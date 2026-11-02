"""Analytical SM-split controller for prefill/decode PDMux.

The controller is intentionally independent from CUDA and scheduler objects so
its policy can be tested without a GPU.  The serving integration records
prefill FLOPs, publishes the latest decode batch resource vector, and applies
the selected pre-created Green Context profile at a batch boundary.
"""

from __future__ import annotations

import dataclasses
import time
from collections import deque
from typing import Deque, Iterable, Optional, Sequence


@dataclasses.dataclass(frozen=True)
class DecodeWorkload:
    batch_size: int = 0
    flops_t: float = 0.0
    state_read_gb: float = 0.0

    @property
    def active(self) -> bool:
        return self.batch_size > 0 and self.flops_t > 0.0 and self.state_read_gb > 0.0


@dataclasses.dataclass(frozen=True)
class SplitPrediction:
    prefill_sm: int
    decode_sm: int
    prefill_peak_tflops: float
    prefill_capacity_tflops: float
    demand_tflops: float
    safe_capacity_tflops: float
    feasible: bool


@dataclasses.dataclass(frozen=True)
class AdaptivePDMuxModel:
    """Coefficients fitted by fit_active_resource_capacity_model.py."""

    prefill_linear_tflops_per_sm: float = 3.7592162614576417
    prefill_diminishing_tflops_per_sm2: float = 0.007148319947983109

    decoder_fixed_ms: float = 3.7612058654573777
    decoder_fixed_work_ms_sm: float = 143.79209879290548
    decoder_flop_ms_sm_per_tflop: float = 1783.7904609219372
    decoder_state_ms_sm_per_gb: float = 5.90396185652666e-15
    decoder_weight_proxy_ms_per_request: float = 0.09153677227403034

    prefill_slowdown_constant: float = 0.028669305129837716
    prefill_slowdown_per_decoder_base_2tb_s: float = 0.12564116059753125
    prefill_slowdown_per_decoder_base_10_tflops_s: float = 0.03245310411879276

    decoder_non_state_fixed_gb_per_step: float = 7.076026132934647
    decoder_non_state_gb_per_request: float = 0.2740861694033353

    def prefill_peak_tflops(self, prefill_sm: int) -> float:
        sm = float(prefill_sm)
        return max(
            0.0,
            self.prefill_linear_tflops_per_sm * sm
            - self.prefill_diminishing_tflops_per_sm2 * sm * sm,
        )

    def decoder_base_rates(
        self, decode_sm: int, workload: DecodeWorkload
    ) -> tuple[float, float]:
        """Return decode base (TFLOP/s, total-read GB/s)."""
        if not workload.active:
            return 0.0, 0.0
        sm = float(decode_sm)
        base_ms = (
            self.decoder_fixed_ms
            + self.decoder_fixed_work_ms_sm / sm
            + self.decoder_flop_ms_sm_per_tflop * workload.flops_t / sm
            + self.decoder_state_ms_sm_per_gb * workload.state_read_gb / sm
            + self.decoder_weight_proxy_ms_per_request * workload.batch_size
        )
        total_read_gb = (
            workload.state_read_gb
            + self.decoder_non_state_fixed_gb_per_step
            + self.decoder_non_state_gb_per_request * workload.batch_size
        )
        return (
            1000.0 * workload.flops_t / base_ms,
            1000.0 * total_read_gb / base_ms,
        )

    def prefill_capacity_tflops(
        self,
        prefill_sm: int,
        decode_sm: int,
        workload: DecodeWorkload,
    ) -> float:
        """Large-prefill-batch capacity under one continuously active decode batch."""
        peak = self.prefill_peak_tflops(prefill_sm)
        if not workload.active:
            return peak
        decoder_tflops, decoder_total_read_gb_s = self.decoder_base_rates(
            decode_sm, workload
        )
        slowdown = (
            self.prefill_slowdown_constant
            + self.prefill_slowdown_per_decoder_base_2tb_s
            * decoder_total_read_gb_s
            / 2000.0
            + self.prefill_slowdown_per_decoder_base_10_tflops_s * decoder_tflops / 10.0
        )
        return peak / (1.0 + slowdown)


class AdaptivePDMuxSMController:
    """Choose the smallest prefill partition that covers rolling FLOP demand."""

    def __init__(
        self,
        splits: Sequence[tuple[int, int]],
        *,
        window_seconds: float = 60.0,
        control_interval_seconds: float = 60.0,
        target_utilization: float = 0.85,
        backlog_drain_seconds: float = 0.0,
        high_priority_prefill_sm_floor: int = 0,
        max_prefill_when_decode_idle: bool = False,
        downscale_hold_seconds: float = 0.0,
        model: Optional[AdaptivePDMuxModel] = None,
        clock=time.monotonic,
    ):
        if not splits:
            raise ValueError("adaptive PDMux requires at least one SM split")
        normalized = sorted({(int(p), int(d)) for p, d in splits})
        if any(p <= 0 or d <= 0 for p, d in normalized):
            raise ValueError(f"adaptive PDMux SM counts must be positive: {normalized}")
        if window_seconds <= 0 or control_interval_seconds <= 0:
            raise ValueError("adaptive PDMux intervals must be positive")
        if not 0.0 < target_utilization <= 1.0:
            raise ValueError("target_utilization must be in (0, 1]")
        if backlog_drain_seconds < 0:
            raise ValueError("backlog_drain_seconds must be non-negative")
        if high_priority_prefill_sm_floor < 0:
            raise ValueError("high_priority_prefill_sm_floor must be non-negative")
        if downscale_hold_seconds < 0:
            raise ValueError("downscale_hold_seconds must be non-negative")

        self.splits = tuple(normalized)
        self.window_seconds = float(window_seconds)
        self.control_interval_seconds = float(control_interval_seconds)
        self.target_utilization = float(target_utilization)
        self.backlog_drain_seconds = float(backlog_drain_seconds)
        self.high_priority_prefill_sm_floor = int(high_priority_prefill_sm_floor)
        self.max_prefill_when_decode_idle = bool(max_prefill_when_decode_idle)
        self.downscale_hold_seconds = float(downscale_hold_seconds)
        self.model = model or AdaptivePDMuxModel()
        self._clock = clock
        # Input-compute demand is recorded once when each request arrives at
        # the prefiller.  It must not depend on when queued work is eventually
        # admitted into a forward batch, otherwise queueing and the current SM
        # split feed back into the traffic-demand signal.
        self._prefill_arrival_flops: Deque[tuple[float, float]] = deque()
        self._started_at = float(clock())
        self._last_control_at: Optional[float] = None
        self._downscale_candidate_since: Optional[float] = None
        self._idle_max_prefill_selected = False

    def _trim(self, now: float) -> None:
        cutoff = now - self.window_seconds
        while (
            self._prefill_arrival_flops and self._prefill_arrival_flops[0][0] < cutoff
        ):
            self._prefill_arrival_flops.popleft()

    def record_prefill_arrival_flops(
        self, flops: float, *, now: Optional[float] = None
    ) -> None:
        if flops <= 0:
            return
        now = self._clock() if now is None else float(now)
        self._prefill_arrival_flops.append((now, float(flops)))
        self._trim(now)

    def ready(self, *, now: Optional[float] = None) -> bool:
        now = self._clock() if now is None else float(now)
        if now - self._started_at < self.window_seconds:
            return False
        return (
            self._last_control_at is None
            or now - self._last_control_at >= self.control_interval_seconds
        )

    def demand_tflops(self, *, now: Optional[float] = None) -> float:
        now = self._clock() if now is None else float(now)
        self._trim(now)
        return sum(flops for _, flops in self._prefill_arrival_flops) / (
            self.window_seconds * 1e12
        )

    def predictions(
        self,
        workload: DecodeWorkload,
        *,
        backlog_flops: float = 0.0,
        now: Optional[float] = None,
    ) -> tuple[SplitPrediction, ...]:
        demand = self.effective_demand_tflops(backlog_flops=backlog_flops, now=now)
        rows = []
        for prefill_sm, decode_sm in self.splits:
            peak = self.model.prefill_peak_tflops(prefill_sm)
            capacity = self.model.prefill_capacity_tflops(
                prefill_sm, decode_sm, workload
            )
            safe_capacity = capacity * self.target_utilization
            rows.append(
                SplitPrediction(
                    prefill_sm=prefill_sm,
                    decode_sm=decode_sm,
                    prefill_peak_tflops=peak,
                    prefill_capacity_tflops=capacity,
                    demand_tflops=demand,
                    safe_capacity_tflops=safe_capacity,
                    feasible=demand <= safe_capacity,
                )
            )
        return tuple(rows)

    def backlog_demand_tflops(self, backlog_flops: float) -> float:
        if self.backlog_drain_seconds <= 0 or backlog_flops <= 0:
            return 0.0
        return float(backlog_flops) / (self.backlog_drain_seconds * 1e12)

    def effective_demand_tflops(
        self, *, backlog_flops: float = 0.0, now: Optional[float] = None
    ) -> float:
        return self.demand_tflops(now=now) + self.backlog_demand_tflops(backlog_flops)

    def choose(
        self,
        workload: DecodeWorkload,
        *,
        backlog_flops: float = 0.0,
        high_priority_pending: bool = False,
        active_prefill_sm: Optional[int] = None,
        now: Optional[float] = None,
        mark_controlled: bool = True,
    ) -> SplitPrediction:
        now = self._clock() if now is None else float(now)
        predictions = self.predictions(workload, backlog_flops=backlog_flops, now=now)
        feasible = [row for row in predictions if row.feasible]
        force_idle_max = self.max_prefill_when_decode_idle and workload.batch_size == 0
        leaving_idle_max = self._idle_max_prefill_selected and workload.batch_size > 0
        if force_idle_max:
            selected = max(predictions, key=lambda row: row.prefill_sm)
        elif high_priority_pending and self.high_priority_prefill_sm_floor > 0:
            protected = [
                row
                for row in predictions
                if row.prefill_sm >= self.high_priority_prefill_sm_floor
            ]
            protected_feasible = [row for row in protected if row.feasible]
            selected = (
                min(protected_feasible, key=lambda row: row.prefill_sm)
                if protected_feasible
                else max(protected or predictions, key=lambda row: row.prefill_sm)
            )
        else:
            selected = (
                min(feasible, key=lambda row: row.prefill_sm)
                if feasible
                else max(predictions, key=lambda row: row.prefill_sm)
            )

        # Do not let a short lull immediately undo an upscale.  A request to
        # increase Prefill SMs is still returned on the first control tick.
        if (
            active_prefill_sm is not None
            and selected.prefill_sm < active_prefill_sm
            and self.downscale_hold_seconds > 0
            and not leaving_idle_max
        ):
            if self._downscale_candidate_since is None:
                self._downscale_candidate_since = now
            if now - self._downscale_candidate_since < self.downscale_hold_seconds:
                active = next(
                    (row for row in predictions if row.prefill_sm == active_prefill_sm),
                    None,
                )
                if active is not None:
                    selected = active
        else:
            self._downscale_candidate_since = None
        self._idle_max_prefill_selected = force_idle_max
        if mark_controlled:
            self._last_control_at = now
        return selected


def validate_splits(
    splits: Iterable[tuple[int, int]], *, multiple: int = 8
) -> tuple[tuple[int, int], ...]:
    normalized = tuple((int(p), int(d)) for p, d in splits)
    if not normalized:
        raise ValueError("adaptive_sm_splits cannot be empty")
    totals = {p + d for p, d in normalized}
    if len(totals) != 1:
        raise ValueError(f"adaptive SM splits must have one total: {normalized}")
    for prefill_sm, decode_sm in normalized:
        if prefill_sm <= 0 or decode_sm <= 0:
            raise ValueError(f"adaptive SM counts must be positive: {normalized}")
        if prefill_sm % multiple or decode_sm % multiple:
            raise ValueError(
                f"adaptive SM counts must be multiples of {multiple}: {normalized}"
            )
    return normalized
