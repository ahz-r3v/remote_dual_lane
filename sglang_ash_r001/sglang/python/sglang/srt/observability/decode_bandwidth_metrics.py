"""Online analytical decode-bandwidth metrics for routing.

This module never profiles or synchronizes the GPU.  A completed decode batch
is converted into modeled read bytes, attributed to high/low priority requests,
and accumulated over a short wall-clock window.
"""

from __future__ import annotations

import dataclasses
import os
import threading
import time
from collections import deque
from typing import Deque, Optional, Sequence

from sglang.srt.observability.model_perf_estimator import Qwen35MoePerfEstimator


@dataclasses.dataclass(frozen=True)
class DecodeBandwidthEstimate:
    total_read_gb: float
    high_read_gb: float
    low_read_gb: float
    predicted_step_ms: float
    marginal_request_ms: float
    marginal_context_ms_per_token: float
    active_bw_gb_s: float
    batch_size: int
    attention_kv_gb: float


@dataclasses.dataclass(frozen=True)
class _ReadEvent:
    timestamp: float
    high_read_gb: float
    low_read_gb: float


class DecodeBandwidthMetricsTracker:
    """Thread-safe rolling decode BW accounting for one GPU worker."""

    # v4 extends the section-writer PDMux training domain through P/D=32/96
    # and decoder batch size 1 while retaining the v3 regular model.
    MODEL_VERSION = "qwen36_decode_bw_router_v4"

    # Full-SM regular decoder, fitted on 45 section-writer grid points and
    # validated on eight measurements drawn from four held-out tasks.
    _REGULAR_W_FIXED_GB = 5.1227492235787695
    _REGULAR_W_EXPERT_SPAN_GB = 9.970056636005262
    _REGULAR_W_HALF_BS = 9.730294289277193
    _REGULAR_T0_MS = 5.990268567075247
    _REGULAR_T_B_MS = 0.11043209520010996
    _REGULAR_T_K_MS_PER_GB = 0.21304984569733382

    # Concurrent PDMux model fitted on the 768-point section-writer factorial
    # sweep and validated on 68 held-out factorial points.
    _PDMUX_T0_MS = 3.7612058654573777
    _PDMUX_FIXED_WORK_MS_SM = 143.79209879290548
    _PDMUX_FLOP_MS_SM_PER_TFLOP = 1783.7904609219372
    _PDMUX_STATE_MS_SM_PER_GB = 5.90396185652666e-15
    _PDMUX_WEIGHT_MS_PER_REQ = 0.09153677227403034
    _PDMUX_READ_FIXED_GB = 7.076026132934647
    _PDMUX_READ_GB_PER_REQ = 0.2740861694033353
    _PDMUX_SLOWDOWN_CONSTANT = 5.660489923176078e-20
    _PDMUX_SLOWDOWN_INV_SM = 6.988957093551848
    _PDMUX_SLOWDOWN_FLOP = 1.979741857703191e-17
    _PDMUX_SLOWDOWN_STATE_GB = 4.132258578293496e-18

    @staticmethod
    def request_key(request_id: str) -> str:
        """Return the Router-compatible compact FNV-1a request key."""
        value = 0xCBF29CE484222325
        for byte in str(request_id).encode("utf-8"):
            value ^= byte
            value = (value * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
        return f"{value:016x}"

    def __init__(
        self,
        estimator: Qwen35MoePerfEstimator,
        *,
        window_seconds: float = 10.0,
        clock=time.monotonic,
        wall_clock=time.time,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("decode BW metrics window must be positive")
        self.estimator = estimator
        self.window_seconds = float(window_seconds)
        self._clock = clock
        self._wall_clock = wall_clock
        self._started_at = float(clock())
        self._events: Deque[_ReadEvent] = deque()
        self._lock = threading.Lock()
        self._metrics_generation = 0
        self._latest: Optional[dict] = None
        self._live_state: Optional[tuple] = None

        self._full_layers = len(estimator.full_layer_ids)
        self._linear_layers = len(estimator.linear_layer_ids)
        self._attention_kv_gb_per_token = (
            self._full_layers
            * estimator.full_kv_bytes_per_context_token_per_layer_per_gpu
            / 1e9
        )
        self._mamba_gb_per_request = (
            self._linear_layers
            * estimator.linear_state_bytes_per_request_per_layer_per_gpu
            / 1e9
        )

    @classmethod
    def maybe_create(cls, model_config, tp_size: int):
        enabled = os.environ.get("SGLANG_ENABLE_DECODE_BW_METRICS", "0").lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return None
        model_type = getattr(model_config.hf_text_config, "model_type", None)
        if model_type != "qwen3_5_moe_text" or tp_size != 1:
            return None
        estimator = Qwen35MoePerfEstimator.from_model_config(model_config, tp_size)
        return cls(
            estimator,
            window_seconds=float(
                os.environ.get("SGLANG_DECODE_BW_WINDOW_SECONDS", "10")
            ),
        )

    def _state_components(self, seq_lens: Sequence[int]) -> tuple[float, float]:
        attention_kv_gb = sum(seq_lens) * self._attention_kv_gb_per_token
        mamba_gb = len(seq_lens) * self._mamba_gb_per_request
        return attention_kv_gb, mamba_gb

    def _regular_estimate(self, seq_lens: Sequence[int]) -> tuple[float, float]:
        batch_size = len(seq_lens)
        attention_kv_gb, mamba_gb = self._state_components(seq_lens)
        non_state_gb = self._REGULAR_W_FIXED_GB + (
            self._REGULAR_W_EXPERT_SPAN_GB
            * batch_size
            / (batch_size + self._REGULAR_W_HALF_BS)
        )
        total_read_gb = attention_kv_gb + mamba_gb + non_state_gb
        step_ms = (
            self._REGULAR_T0_MS
            + self._REGULAR_T_B_MS * batch_size
            + self._REGULAR_T_K_MS_PER_GB * attention_kv_gb
        )
        return total_read_gb, step_ms

    def _pdmux_estimate(
        self,
        seq_lens: Sequence[int],
        *,
        decode_sm: int,
        prefill_pressure: float,
    ) -> tuple[float, float]:
        batch_size = len(seq_lens)
        attention_kv_gb, mamba_gb = self._state_components(seq_lens)
        state_gb = attention_kv_gb + mamba_gb
        perf = self.estimator.estimate_decode(
            seq_lens, num_tokens=batch_size, num_logits=batch_size
        )
        flops_t = perf.flops / 1e12
        sm = max(float(decode_sm), 1.0)
        base_ms = (
            self._PDMUX_T0_MS
            + self._PDMUX_FIXED_WORK_MS_SM / sm
            + self._PDMUX_FLOP_MS_SM_PER_TFLOP * flops_t / sm
            + self._PDMUX_STATE_MS_SM_PER_GB * state_gb / sm
            + self._PDMUX_WEIGHT_MS_PER_REQ * batch_size
        )
        slowdown = max(0.0, min(float(prefill_pressure), 1.0)) * (
            self._PDMUX_SLOWDOWN_CONSTANT
            + self._PDMUX_SLOWDOWN_INV_SM / sm
            + self._PDMUX_SLOWDOWN_FLOP * flops_t
            + self._PDMUX_SLOWDOWN_STATE_GB * state_gb
        )
        total_read_gb = (
            state_gb
            + self._PDMUX_READ_FIXED_GB
            + self._PDMUX_READ_GB_PER_REQ * batch_size
        )
        return total_read_gb, base_ms * (1.0 + slowdown)

    def estimate(
        self,
        seq_lens: Sequence[int],
        high_priority: Sequence[bool],
        *,
        worker_kind: str,
        decode_sm: int = 132,
        prefill_pressure: float = 0.0,
    ) -> DecodeBandwidthEstimate:
        if not seq_lens or len(seq_lens) != len(high_priority):
            raise ValueError("decode BW estimate requires one priority per request")
        if worker_kind == "pdmux_decoder":
            total_read_gb, step_ms = self._pdmux_estimate(
                seq_lens,
                decode_sm=decode_sm,
                prefill_pressure=prefill_pressure,
            )

            def estimate_step(lengths):
                return self._pdmux_estimate(
                    lengths,
                    decode_sm=decode_sm,
                    prefill_pressure=prefill_pressure,
                )[1]

        elif worker_kind == "regular_decoder":
            total_read_gb, step_ms = self._regular_estimate(seq_lens)

            def estimate_step(lengths):
                return self._regular_estimate(lengths)[1]

        else:
            raise ValueError(f"unknown decoder worker kind: {worker_kind}")

        # Publish a local marginal form of the same analytical model.  The
        # router uses it to project requests admitted since the latest worker
        # snapshot without duplicating the fitted model coefficients in Rust.
        # Qwen3.6's decode model is locally linear in context length; finite
        # differencing also keeps this correct if the estimator changes.
        plus_one_token_ms = estimate_step([*seq_lens, 1])
        plus_two_tokens_ms = estimate_step([*seq_lens, 2])
        marginal_context_ms_per_token = max(0.0, plus_two_tokens_ms - plus_one_token_ms)
        marginal_request_ms = max(
            0.0,
            plus_one_token_ms - step_ms - marginal_context_ms_per_token,
        )

        attention_kv_gb, _ = self._state_components(seq_lens)
        high_attention_gb = sum(
            seq_len * self._attention_kv_gb_per_token
            for seq_len, is_high in zip(seq_lens, high_priority)
            if is_high
        )
        high_count = sum(high_priority)
        batch_size = len(seq_lens)
        shared_gb = total_read_gb - attention_kv_gb
        high_read_gb = high_attention_gb + shared_gb * high_count / batch_size
        low_read_gb = total_read_gb - high_read_gb
        return DecodeBandwidthEstimate(
            total_read_gb=total_read_gb,
            high_read_gb=high_read_gb,
            low_read_gb=low_read_gb,
            predicted_step_ms=step_ms,
            marginal_request_ms=marginal_request_ms,
            marginal_context_ms_per_token=marginal_context_ms_per_token,
            active_bw_gb_s=1000.0 * total_read_gb / step_ms,
            batch_size=batch_size,
            attention_kv_gb=attention_kv_gb,
        )

    def initialize_idle(
        self,
        *,
        worker_kind: str,
        prefill_sm: int = 0,
        decode_sm: int = 132,
    ) -> None:
        """Publish cold-start capacity before the first serving batch.

        Decode capacity is workload-dependent, so the cold-start value uses a
        configurable nominal batch. The first real decode step replaces it.
        """
        if worker_kind == "regular_decoder":
            nominal_bs = max(
                1,
                int(os.environ.get("SGLANG_REGULAR_BW_NOMINAL_BATCH_SIZE", "32")),
            )
            nominal_context = max(
                1,
                int(os.environ.get("SGLANG_REGULAR_BW_NOMINAL_CONTEXT", "32768")),
            )
            estimate = self.estimate(
                [nominal_context] * nominal_bs,
                [False] * nominal_bs,
                worker_kind=worker_kind,
            )
        elif worker_kind == "pdmux_decoder":
            nominal_bs = max(
                1, int(os.environ.get("SGLANG_PDMUX_BW_NOMINAL_BATCH_SIZE", "24"))
            )
            nominal_context = max(
                1, int(os.environ.get("SGLANG_PDMUX_BW_NOMINAL_CONTEXT", "16384"))
            )
            estimate = self.estimate(
                [nominal_context] * nominal_bs,
                [False] * nominal_bs,
                worker_kind=worker_kind,
                decode_sm=decode_sm,
            )
        else:
            raise ValueError(f"unknown decoder worker kind: {worker_kind}")
        capacity = estimate.active_bw_gb_s
        with self._lock:
            self._latest = {
                "worker_kind": worker_kind,
                "model_version": self.MODEL_VERSION,
                "capacity_gb_s": capacity,
                "used_high_gb_s": 0.0,
                "used_low_gb_s": 0.0,
                "headroom_gb_s": capacity,
                "active_batch_bw_gb_s": 0.0,
                "active_batch_size": 0,
                "active_total_sequence_tokens": 0,
                "queued_batch_size": 0,
                "queued_total_sequence_tokens": 0,
                "active_request_ids": [],
                "queued_request_ids": [],
                "active_sequence_tokens": [],
                "queued_sequence_tokens": [],
                "active_remaining_output_tokens": [],
                "queued_remaining_output_tokens": [],
                "active_priorities": [],
                "queued_priorities": [],
                "active_attention_kv_gb": 0.0,
                "predicted_step_ms": estimate.predicted_step_ms,
                "marginal_request_ms": estimate.marginal_request_ms,
                "marginal_context_ms_per_token": estimate.marginal_context_ms_per_token,
                "window_seconds": self.window_seconds,
                "metrics_generation": self._metrics_generation,
                "collected_at": float(self._wall_clock()),
                "prefill_sm": int(prefill_sm),
                "decode_sm": int(decode_sm),
                "prefill_pressure": 0.0,
            }
            self._live_state = (
                0,
                0,
                0,
                0,
                (),
                (),
                (),
                (),
                (),
                (),
                int(prefill_sm),
                int(decode_sm),
                0.0,
            )

    def update_live_state(
        self,
        active_seq_lens: Sequence[int],
        queued_seq_lens: Sequence[int] = (),
        *,
        active_request_ids: Sequence[str] = (),
        queued_request_ids: Sequence[str] = (),
        active_remaining_output_tokens: Sequence[int] = (),
        queued_remaining_output_tokens: Sequence[int] = (),
        active_priorities: Sequence[str] = (),
        queued_priorities: Sequence[str] = (),
        prefill_sm: int = 0,
        decode_sm: int = 132,
        prefill_pressure: float = 0.0,
    ) -> None:
        """Publish the exact state consumed by router-side admission models.

        This is deliberately separate from rolling bandwidth accounting: a
        `/loads` snapshot must also advance when a request is queued, finishes,
        or the adaptive PDMux SM profile changes without a decode step.
        """
        active_bs = len(active_seq_lens)
        active_total_seq = sum(int(value) for value in active_seq_lens)
        queued_bs = len(queued_seq_lens)
        queued_total_seq = sum(int(value) for value in queued_seq_lens)
        pressure = max(0.0, min(float(prefill_pressure), 1.0))
        active_ids = tuple(str(value) for value in active_request_ids if value)
        queued_ids = tuple(str(value) for value in queued_request_ids if value)
        active_remaining = tuple(
            max(0, int(value)) for value in active_remaining_output_tokens
        )
        queued_remaining = tuple(
            max(0, int(value)) for value in queued_remaining_output_tokens
        )
        active_priority_values = tuple(str(value) for value in active_priorities)
        queued_priority_values = tuple(str(value) for value in queued_priorities)
        live_state = (
            active_bs,
            active_total_seq,
            queued_bs,
            queued_total_seq,
            active_ids,
            queued_ids,
            active_remaining,
            queued_remaining,
            active_priority_values,
            queued_priority_values,
            int(prefill_sm),
            int(decode_sm),
            pressure,
        )
        with self._lock:
            if self._latest is None or live_state == self._live_state:
                return
            self._metrics_generation += 1
            self._live_state = live_state
            self._latest.update(
                {
                    "active_batch_size": active_bs,
                    "active_total_sequence_tokens": active_total_seq,
                    "queued_batch_size": queued_bs,
                    "queued_total_sequence_tokens": queued_total_seq,
                    "active_request_ids": list(active_ids),
                    "queued_request_ids": list(queued_ids),
                    "active_sequence_tokens": [int(value) for value in active_seq_lens],
                    "queued_sequence_tokens": [int(value) for value in queued_seq_lens],
                    "active_remaining_output_tokens": list(active_remaining),
                    "queued_remaining_output_tokens": list(queued_remaining),
                    "active_priorities": list(active_priority_values),
                    "queued_priorities": list(queued_priority_values),
                    "metrics_generation": self._metrics_generation,
                    "collected_at": float(self._wall_clock()),
                    "prefill_sm": int(prefill_sm),
                    "decode_sm": int(decode_sm),
                    "prefill_pressure": pressure,
                }
            )

    def record(
        self,
        seq_lens: Sequence[int],
        high_priority: Sequence[bool],
        *,
        worker_kind: str,
        prefill_sm: int = 0,
        decode_sm: int = 132,
        prefill_pressure: float = 0.0,
        queued_seq_lens: Sequence[int] = (),
        active_request_ids: Sequence[str] = (),
        queued_request_ids: Sequence[str] = (),
        active_remaining_output_tokens: Sequence[int] = (),
        queued_remaining_output_tokens: Sequence[int] = (),
        active_priorities: Sequence[str] = (),
        queued_priorities: Sequence[str] = (),
    ) -> None:
        estimate = self.estimate(
            seq_lens,
            high_priority,
            worker_kind=worker_kind,
            decode_sm=decode_sm,
            prefill_pressure=prefill_pressure,
        )
        now = float(self._clock())
        event = _ReadEvent(now, estimate.high_read_gb, estimate.low_read_gb)
        with self._lock:
            self._events.append(event)
            self._trim_locked(now)
            capacity = estimate.active_bw_gb_s
            self._metrics_generation += 1
            self._latest = {
                "worker_kind": worker_kind,
                "model_version": self.MODEL_VERSION,
                "capacity_gb_s": capacity,
                "used_high_gb_s": 0.0,
                "used_low_gb_s": 0.0,
                "headroom_gb_s": capacity,
                "active_batch_bw_gb_s": estimate.active_bw_gb_s,
                "active_batch_size": estimate.batch_size,
                "active_total_sequence_tokens": sum(int(value) for value in seq_lens),
                "queued_batch_size": len(queued_seq_lens),
                "queued_total_sequence_tokens": sum(
                    int(value) for value in queued_seq_lens
                ),
                "active_request_ids": [
                    str(value) for value in active_request_ids if value
                ],
                "queued_request_ids": [
                    str(value) for value in queued_request_ids if value
                ],
                "active_sequence_tokens": [int(value) for value in seq_lens],
                "queued_sequence_tokens": [int(value) for value in queued_seq_lens],
                "active_remaining_output_tokens": [
                    max(0, int(value)) for value in active_remaining_output_tokens
                ],
                "queued_remaining_output_tokens": [
                    max(0, int(value)) for value in queued_remaining_output_tokens
                ],
                "active_priorities": [str(value) for value in active_priorities],
                "queued_priorities": [str(value) for value in queued_priorities],
                "active_attention_kv_gb": estimate.attention_kv_gb,
                "predicted_step_ms": estimate.predicted_step_ms,
                "marginal_request_ms": estimate.marginal_request_ms,
                "marginal_context_ms_per_token": estimate.marginal_context_ms_per_token,
                "window_seconds": self.window_seconds,
                "metrics_generation": self._metrics_generation,
                "collected_at": float(self._wall_clock()),
                "prefill_sm": int(prefill_sm),
                "decode_sm": int(decode_sm),
                "prefill_pressure": float(prefill_pressure),
            }
            self._live_state = (
                estimate.batch_size,
                sum(int(value) for value in seq_lens),
                len(queued_seq_lens),
                sum(int(value) for value in queued_seq_lens),
                tuple(str(value) for value in active_request_ids if value),
                tuple(str(value) for value in queued_request_ids if value),
                tuple(max(0, int(value)) for value in active_remaining_output_tokens),
                tuple(max(0, int(value)) for value in queued_remaining_output_tokens),
                tuple(str(value) for value in active_priorities),
                tuple(str(value) for value in queued_priorities),
                int(prefill_sm),
                int(decode_sm),
                max(0.0, min(float(prefill_pressure), 1.0)),
            )
            self._refresh_locked(now)

    def _trim_locked(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()

    def _refresh_locked(self, now: float) -> None:
        if self._latest is None:
            return
        self._trim_locked(now)
        elapsed = max(1.0, min(self.window_seconds, now - self._started_at))
        high_bw = sum(item.high_read_gb for item in self._events) / elapsed
        low_bw = sum(item.low_read_gb for item in self._events) / elapsed
        self._latest["used_high_gb_s"] = high_bw
        self._latest["used_low_gb_s"] = low_bw
        # Keep overload visible as negative headroom; clipping it to zero
        # would make all saturated workers indistinguishable to the router.
        self._latest["headroom_gb_s"] = self._latest["capacity_gb_s"] - high_bw - low_bw
        self._latest["collected_at"] = float(self._wall_clock())

    def get_metrics(self) -> Optional[dict]:
        with self._lock:
            self._refresh_locked(float(self._clock()))
            return None if self._latest is None else dict(self._latest)
