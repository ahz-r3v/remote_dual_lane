"""Independent prefill/decode schedulers for Green Context PDMux."""

import dataclasses
import logging
import threading
import time
from collections import deque
from typing import Deque, List, Optional, Tuple

import torch

from sglang.srt.disaggregation.decode import PDDecodeLaneState
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.environ import envs
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.common import maybe_cache_unfinished_req
from sglang.srt.multiplex.adaptive_pdmux_sm import (
    AdaptivePDMuxSMController,
    DecodeWorkload,
    validate_splits,
)
from sglang.srt.observability.model_perf_estimator import Qwen35MoePerfEstimator
from sglang.srt.utils import DynamicGradMode

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PrefillDecodeSchedulerState:
    """Coordination state shared by the prefill and decode schedulers."""

    ready_batches: Deque[ScheduleBatch] = dataclasses.field(default_factory=deque)
    state_lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)
    resource_lock: threading.RLock = dataclasses.field(default_factory=threading.RLock)
    condition: threading.Condition = dataclasses.field(init=False)
    stop: bool = False
    fatal_error: Optional[BaseException] = None
    decode_running_reqs: int = 0
    decode_executing_bs: int = 0
    decode_ready_reqs: int = 0
    local_prefill_pending_reqs: int = 0
    snapshot_seq: int = 0
    updated_at: float = 0.0
    load_dirty: bool = False
    load_publisher_thread_id: Optional[int] = None
    last_load_publish_at: float = 0.0
    decode_workload: DecodeWorkload = dataclasses.field(default_factory=DecodeWorkload)
    prefill_pressure: float = 0.0
    decode_inflight_prefill_pressure: float = 0.0
    active_profile_index: int = 0
    pending_profile_index: Optional[int] = None
    # A pending request does not stop decode. It becomes armed only after the
    # old-profile prefill has drained; decode then drains at most one step.
    profile_switch_armed: bool = False
    profile_epoch: int = 0
    profile_switch_requested_at: float = 0.0
    profile_switch_armed_at: float = 0.0
    # Set before the decode thread leaves state_lock to prepare/submit work.
    # A paired profile switch cannot commit while this reservation is held.
    decode_launch_in_progress: bool = False
    # True from the moment the Prefill scheduler reserves the ordinary
    # full-resource runtime until that batch has completed and its result has
    # been handed off. A Decode request arriving meanwhile may launch on the
    # active Green Context Decode stream and compete with the ordinary stream;
    # the next Prefill batch returns to the paired Green Context profile.
    full_prefill_active: bool = False

    def __post_init__(self) -> None:
        self.condition = threading.Condition(self.state_lock)


class SchedulerPrefillDecodeMuxMixin:
    """Run eager prefill and graph decode on independent host/GPU streams."""

    def _get_pdmux_active_sm_split(
        self, shared: Optional[PrefillDecodeSchedulerState] = None
    ) -> Tuple[int, int]:
        """Return the SM split used by the currently active lane runtimes.

        ``decode_lane_profile_splits`` is populated only when runtime profile
        switching is enabled.  A fixed Green Context configuration still has
        two PDMux lane runtimes, but its profile table is intentionally empty;
        in that case the split comes directly from the lane configuration.
        """
        model_runner = self.tp_worker.model_runner
        profile_splits = getattr(model_runner, "decode_lane_profile_splits", ())
        profile_index = shared.active_profile_index if shared is not None else 0
        if 0 <= profile_index < len(profile_splits):
            prefill_sm, decode_sm = profile_splits[profile_index]
            return int(prefill_sm), int(decode_sm)

        config = getattr(model_runner, "decode_lane_mux_config", None)
        if config is None:
            raise RuntimeError("PDMux lane configuration is not initialized")
        prefill_sm = int(config.short_green_ctx_sm_count)
        decode_sm = int(config.long_green_ctx_sm_count)
        if prefill_sm <= 0 or decode_sm <= 0:
            raise RuntimeError(
                "PDMux fixed Green Context split must be positive: "
                f"prefill={prefill_sm}, decode={decode_sm}"
            )
        return prefill_sm, decode_sm

    def _init_pdmux_adaptive_sm_calculation(self) -> None:
        config = self.tp_worker.model_runner.decode_lane_mux_config
        if not config.adaptive_prefill_sm_calculation:
            self.pdmux_adaptive_sm_controller = None
            self.pdmux_adaptive_sm_estimator = None
            return
        if not config.prefill_decode_lanes:
            raise ValueError(
                "adaptive_prefill_sm_calculation requires prefill_decode_lanes"
            )
        splits = validate_splits(
            config.adaptive_sm_splits or ((64, 64), (80, 48), (96, 32), (112, 16))
        )
        self.pdmux_adaptive_sm_controller = AdaptivePDMuxSMController(
            splits,
            window_seconds=config.adaptive_window_seconds,
            control_interval_seconds=config.adaptive_control_interval_seconds,
            target_utilization=config.adaptive_target_utilization,
            backlog_drain_seconds=config.adaptive_backlog_drain_seconds,
            high_priority_prefill_sm_floor=(
                config.adaptive_high_priority_prefill_sm_floor
            ),
            # The scheduler implements true full-resource Prefill outside
            # Green Context. Do not also force the paired controller to the
            # 112/16 profile while Decode is idle.
            max_prefill_when_decode_idle=False,
            downscale_hold_seconds=config.adaptive_downscale_hold_seconds,
        )
        self.pdmux_adaptive_sm_estimator = Qwen35MoePerfEstimator.from_model_config(
            self.model_config, self.ps.tp_size
        )
        logger.info(
            "Initialized PDMux SM controller: splits=%s window=%.1fs "
            "interval=%.1fs target_utilization=%.3f backlog_drain=%.1fs "
            "high_priority_floor=%d max_prefill_when_decode_idle=%s "
            "downscale_hold=%.1fs switching=%s",
            splits,
            config.adaptive_window_seconds,
            config.adaptive_control_interval_seconds,
            config.adaptive_target_utilization,
            config.adaptive_backlog_drain_seconds,
            config.adaptive_high_priority_prefill_sm_floor,
            config.adaptive_max_prefill_when_decode_idle,
            config.adaptive_downscale_hold_seconds,
            config.adaptive_sm_switching,
        )

    def _pdmux_prefill_backlog_snapshot(self) -> tuple[float, bool, int]:
        """Estimate queued Prefill work and detect latency-sensitive demand."""
        estimator = getattr(self, "pdmux_adaptive_sm_estimator", None)
        if estimator is None:
            return 0.0, False, 0

        queued_flops = 0.0
        queued_reqs = list(self.waiting_queue)
        for req in queued_reqs:
            input_len = len(req.full_untruncated_fill_ids)
            if input_len <= 0:
                input_len = len(req.origin_input_ids)
            if input_len <= 0:
                continue
            extend_len = min(
                max(int(getattr(req, "extend_input_len", 0)), 0),
                input_len,
            )
            if extend_len <= 0:
                # Prefix matching has not run yet.  Treat the whole prompt as
                # outstanding so queue pressure is never hidden from control.
                extend_len = input_len
            estimate = estimator.estimate_prefill(
                [input_len - extend_len], [extend_len], num_logits=1
            )
            queued_flops += estimate.flops

        high_priority_pending = any(
            getattr(req, "highway_priority", "low") == "high" for req in queued_reqs
        )
        inflight = getattr(self, "prefill_decode_inflight_batch", None)
        if inflight is not None:
            high_priority_pending |= any(
                getattr(req, "highway_priority", "low") == "high"
                for req in inflight.reqs
            )
        return queued_flops, high_priority_pending, len(queued_reqs)

    def _record_pdmux_prefill_scheduled_flops(self, req) -> None:
        """Record cache-adjusted work after the scheduler's native prefix match.

        ``Req.init_next_round_input(tree_cache)`` must run before this method.
        Reusing its exact token-level match avoids an extra radix lookup.  A
        per-request marker prevents chunking/retraction from double counting.
        """
        controller = getattr(self, "pdmux_adaptive_sm_controller", None)
        estimator = getattr(self, "pdmux_adaptive_sm_estimator", None)
        if controller is None or estimator is None:
            return
        if getattr(req, "_pdmux_prefill_flops_recorded", False):
            return
        req._pdmux_prefill_flops_recorded = True

        input_len = len(req.full_untruncated_fill_ids)
        if input_len <= 0:
            return
        extend_len = min(max(int(req.extend_input_len), 0), input_len)
        prefix_len = input_len - extend_len
        if extend_len <= 0:
            return
        num_logits = 1
        if req.return_logprob and req.logprob_start_len >= 0:
            num_logits = max(0, input_len - req.logprob_start_len)
        estimate = estimator.estimate_prefill(
            [prefix_len],
            [extend_len],
            num_logits=num_logits,
        )
        controller.record_prefill_arrival_flops(estimate.flops)

    def _estimate_pdmux_prefill_pressure(self, batch: ScheduleBatch) -> float:
        """Estimate interference pressure for the batch being submitted."""
        controller = getattr(self, "pdmux_adaptive_sm_controller", None)
        estimator = getattr(self, "pdmux_adaptive_sm_estimator", None)
        if controller is None or estimator is None:
            return 0.0
        if batch.prefix_lens is None or batch.extend_lens is None:
            return 0.0
        estimate = estimator.estimate_prefill(
            batch.prefix_lens,
            batch.extend_lens,
            num_logits=self.metrics_reporter._prefill_num_logits(batch),
        )
        shared = getattr(self, "prefill_decode_shared_state", None)
        prefill_sm, _ = self._get_pdmux_active_sm_split(shared)
        peak = controller.model.prefill_peak_tflops(prefill_sm)
        flops_t = estimate.flops / 1e12
        base_ms = 39.50155512477768 + 1000.0 * flops_t / max(peak, 1e-9)
        active_tflops = 1000.0 * flops_t / base_ms
        return max(0.0, min(active_tflops / max(peak, 1e-9), 1.0))

    def _estimate_pdmux_decode_workload(
        self, batch: Optional[ScheduleBatch]
    ) -> DecodeWorkload:
        estimator = getattr(self, "pdmux_adaptive_sm_estimator", None)
        if estimator is None or batch is None or batch.is_empty():
            return DecodeWorkload()
        batch_size = len(batch.reqs)
        if batch.seq_lens_cpu is not None:
            seq_lens = batch.seq_lens_cpu.tolist()
        else:
            seq_lens = [req.seqlen for req in batch.reqs]
        estimate = estimator.estimate_decode(
            seq_lens,
            num_tokens=batch_size,
            num_logits=batch_size,
        )
        return DecodeWorkload(
            batch_size=batch_size,
            flops_t=estimate.flops / 1e12,
            state_read_gb=estimate.kv_cache_read_bytes / 1e9,
        )

    def _maybe_log_pdmux_adaptive_sm_prediction(
        self, shared: PrefillDecodeSchedulerState
    ) -> None:
        controller = getattr(self, "pdmux_adaptive_sm_controller", None)
        if controller is None or not controller.ready():
            return
        with shared.state_lock:
            decode_workload = shared.decode_workload
            active_prefill_sm, _ = self._get_pdmux_active_sm_split(shared)
        backlog_flops, high_priority_pending, queued_reqs = (
            self._pdmux_prefill_backlog_snapshot()
        )
        selected = controller.choose(
            decode_workload,
            backlog_flops=backlog_flops,
            high_priority_pending=high_priority_pending,
            active_prefill_sm=active_prefill_sm,
        )
        predictions = controller.predictions(
            decode_workload, backlog_flops=backlog_flops
        )
        config = self.tp_worker.model_runner.decode_lane_mux_config
        requested_profile = None
        if config.adaptive_sm_switching:
            profile_splits = self.tp_worker.model_runner.decode_lane_profile_splits
            requested_profile = profile_splits.index(
                (selected.prefill_sm, selected.decode_sm)
            )
            with shared.condition:
                if requested_profile == shared.active_profile_index:
                    shared.pending_profile_index = None
                    shared.profile_switch_armed = False
                    shared.profile_switch_armed_at = 0.0
                else:
                    shared.pending_profile_index = requested_profile
                    shared.profile_switch_requested_at = time.monotonic()
                shared.condition.notify_all()
        logger.info(
            "PDMux SM decision: prefill_demand=%.3f TFLOP/s "
            "backlog_demand=%.3f TFLOP/s queued_reqs=%d high_priority=%s "
            "decode_bs=%d decode_flops=%.6f TFLOP/step "
            "decode_state_read=%.3f GB/step selected=%d/%d feasible=%s "
            "active_profile=%d requested_profile=%s switching=%s candidates=%s",
            selected.demand_tflops,
            controller.backlog_demand_tflops(backlog_flops),
            queued_reqs,
            high_priority_pending,
            decode_workload.batch_size,
            decode_workload.flops_t,
            decode_workload.state_read_gb,
            selected.prefill_sm,
            selected.decode_sm,
            selected.feasible,
            shared.active_profile_index,
            requested_profile,
            config.adaptive_sm_switching,
            [
                {
                    "split": f"{row.prefill_sm}/{row.decode_sm}",
                    "capacity": round(row.prefill_capacity_tflops, 3),
                    "safe_capacity": round(row.safe_capacity_tflops, 3),
                    "feasible": row.feasible,
                }
                for row in predictions
            ],
        )

    def get_pdmux_load_snapshot(self) -> Optional[dict]:
        """Return the most recent PDMux-local decode load for /v1/loads."""
        shared = getattr(self, "prefill_decode_shared_state", None)
        if shared is None:
            return None
        with shared.state_lock:
            return {
                "decode_running_reqs": shared.decode_running_reqs,
                "decode_executing_bs": shared.decode_executing_bs,
                "decode_ready_reqs": shared.decode_ready_reqs,
                "local_prefill_pending_reqs": shared.local_prefill_pending_reqs,
                "snapshot_seq": shared.snapshot_seq,
                "updated_at": shared.updated_at,
            }

    @staticmethod
    def _commit_pdmux_load_locked(
        shared: PrefillDecodeSchedulerState, **updates: int
    ) -> bool:
        """Commit an already-computed set of load counters atomically."""
        changed = any(getattr(shared, key) != value for key, value in updates.items())
        if not changed and shared.snapshot_seq != 0:
            return False
        for key, value in updates.items():
            setattr(shared, key, value)
        shared.snapshot_seq += 1
        shared.updated_at = time.time()
        shared.load_dirty = True
        return True

    def _update_pdmux_prefill_load(
        self,
        shared: PrefillDecodeSchedulerState,
    ) -> bool:
        """Update only counters owned by the main/prefill scheduler thread."""
        pending = sum(
            1 for req in self.waiting_queue if getattr(req, "local_decode", False)
        )
        prefill_batch = getattr(self, "prefill_decode_inflight_batch", None)
        if prefill_batch is not None:
            pending += sum(
                1 for req in prefill_batch.reqs if getattr(req, "local_decode", False)
            )
        with shared.condition:
            changed = self._commit_pdmux_load_locked(
                shared, local_prefill_pending_reqs=pending
            )
            if changed:
                shared.condition.notify_all()
        return changed

    def _update_pdmux_decode_load(
        self,
        decode_lane: PDDecodeLaneState,
        shared: PrefillDecodeSchedulerState,
        *,
        admitted_ready_reqs: int = 0,
    ) -> bool:
        """Update only counters owned by the decode scheduler thread.

        ``decode_ready_reqs`` is handoff state. Prefill increments it when a
        completed local prefill is queued; decode subtracts the admitted count
        in the same commit that publishes the new running batch size.
        """
        running = len(decode_lane.running_batch.reqs)
        executing = (
            len(decode_lane.inflight_batch.reqs)
            if decode_lane.inflight_batch is not None
            else 0
        )
        with shared.condition:
            ready = shared.decode_ready_reqs - admitted_ready_reqs
            if ready < 0:
                raise RuntimeError(
                    "PDMux decode-ready accounting underflow: "
                    f"ready={shared.decode_ready_reqs}, admitted={admitted_ready_reqs}"
                )
            changed = self._commit_pdmux_load_locked(
                shared,
                decode_running_reqs=running,
                decode_executing_bs=executing,
                decode_ready_reqs=ready,
            )
            if changed:
                shared.condition.notify_all()
        return changed

    def _publish_pdmux_load_if_dirty(
        self, shared: PrefillDecodeSchedulerState, *, force: bool = False
    ) -> bool:
        """Publish PDMux load from its single writer-owning main thread.

        Besides publishing state changes immediately, send a low-frequency
        heartbeat while the lanes are idle. Decode-bandwidth usage decays with
        wall-clock time, and the router must not mistake a healthy idle worker
        for one whose metrics endpoint has stopped updating.
        """
        publisher_thread_id = shared.load_publisher_thread_id
        if (
            publisher_thread_id is not None
            and threading.get_ident() != publisher_thread_id
        ):
            raise RuntimeError("PDMux load snapshots must be published by main thread")
        now = time.monotonic()
        with shared.state_lock:
            heartbeat_due = now - shared.last_load_publish_at >= 1.0
            if not force and not shared.load_dirty and not heartbeat_due:
                return False
            publishing_seq = shared.snapshot_seq
        self.publish_load_snapshot(force=True)
        with shared.state_lock:
            shared.last_load_publish_at = now
            if shared.snapshot_seq == publishing_seq:
                shared.load_dirty = False
        return True

    def _init_prefill_decode_lane_states(
        self,
    ) -> Tuple[PDDecodeLaneState, PDDecodeLaneState]:
        """Create the prefill/decode streams and capture graphs for decode."""
        if getattr(self, "prefill_decode_lane_states", None) is not None:
            return self.prefill_decode_lane_states

        runtimes = getattr(self.tp_worker.model_runner, "decode_lane_runtimes", [])
        if len(runtimes) < 2:
            raise ValueError("PDMux requires initialized prefill/decode runtimes")

        config = getattr(
            self.tp_worker.model_runner,
            "decode_lane_mux_config",
            None,
        )
        initial_profile_index = 0
        if config is not None and config.adaptive_sm_switching:
            initial_split = (
                config.short_green_ctx_sm_count,
                config.long_green_ctx_sm_count,
            )
            initial_profile_index = (
                self.tp_worker.model_runner.decode_lane_profile_splits.index(
                    initial_split
                )
            )

        lanes = []
        for lane_id, name in enumerate(("prefill", "decode")):
            runtime_lane_id = initial_profile_index * 2 + lane_id
            runtime = runtimes[runtime_lane_id]
            stream = runtime.stream
            if stream is None:
                stream = self.device_module.Stream()
                runtime.stream = stream
            lanes.append(
                PDDecodeLaneState(
                    name=name,
                    lane_id=lane_id,
                    stream=stream,
                    stream_ctx=self.device_module.stream(stream),
                    running_batch=ScheduleBatch(reqs=[], batch_is_full=False),
                    runtime_lane_id=runtime_lane_id,
                )
            )

        if not self.server_args.disable_cuda_graph:
            capture_bs = list(self.server_args.cuda_graph_bs or [])
            if capture_bs:
                logger.info(
                    "Capturing PDMux decode CUDA graphs for %d profile(s) "
                    "with buckets=%s",
                    len(runtimes) // 2,
                    capture_bs,
                )
                capture_bs_by_runtime = [
                    [] if runtime_index % 2 == 0 else capture_bs
                    for runtime_index in range(len(runtimes))
                ]
                self.tp_worker.model_runner.init_decode_lane_cuda_graphs(
                    capture_bs_by_runtime,
                    streams=[runtime.stream for runtime in runtimes],
                )
                lanes[1].graph_enabled = True

        self.prefill_decode_lane_states = tuple(lanes)
        return self.prefill_decode_lane_states

    def _set_pdmux_lane_runtime(
        self, lane: PDDecodeLaneState, runtime_lane_id: int
    ) -> None:
        runtime = self.tp_worker.model_runner.decode_lane_runtimes[runtime_lane_id]
        lane.runtime_lane_id = runtime_lane_id
        lane.stream = runtime.stream
        lane.stream_ctx = self.device_module.stream(runtime.stream)

    def _maybe_commit_pdmux_profile_switch(
        self,
        prefill_lane: PDDecodeLaneState,
        decode_lane: PDDecodeLaneState,
        shared: PrefillDecodeSchedulerState,
    ) -> bool:
        """Atomically switch both logical lanes after all old-profile work drains."""
        with shared.condition:
            target_profile = shared.pending_profile_index
            if target_profile is None or not shared.profile_switch_armed:
                return False
            if (
                prefill_lane.inflight_batch is not None
                or decode_lane.inflight_batch is not None
                or shared.decode_launch_in_progress
            ):
                return False

            runtimes = self.tp_worker.model_runner.decode_lane_runtimes
            prefill_runtime_id = target_profile * 2
            decode_runtime_id = prefill_runtime_id + 1
            if decode_runtime_id >= len(runtimes):
                raise RuntimeError(
                    "adaptive PDMux profile points outside runtime table: "
                    f"profile={target_profile}, runtimes={len(runtimes)}"
                )

            old_profile = shared.active_profile_index
            profile_splits = self.tp_worker.model_runner.decode_lane_profile_splits
            old_prefill_sm, old_decode_sm = profile_splits[old_profile]
            new_prefill_sm, new_decode_sm = profile_splits[target_profile]
            if old_prefill_sm + old_decode_sm != new_prefill_sm + new_decode_sm:
                raise RuntimeError(
                    "adaptive PDMux profile switch must preserve the paired "
                    "SM budget: "
                    f"old={old_prefill_sm}/{old_decode_sm}, "
                    f"new={new_prefill_sm}/{new_decode_sm}"
                )

            # All old-profile GPU work has drained and the launch gate remains
            # closed while holding shared.condition. Switch the shrinking lane
            # first so a future runtime implementation that actively
            # releases/acquires Green Context SMs can never transiently
            # over-allocate the device. These are only two pre-created runtime
            # pointer updates today, so no extra synchronization is introduced.
            commit_started_at = time.perf_counter()
            if new_prefill_sm < old_prefill_sm:
                if new_decode_sm <= old_decode_sm:
                    raise RuntimeError(
                        "invalid adaptive PDMux reallocation: Prefill shrinks "
                        "without Decode growing"
                    )
                shrink_lane = "prefill"
                self._set_pdmux_lane_runtime(prefill_lane, prefill_runtime_id)
                self._set_pdmux_lane_runtime(decode_lane, decode_runtime_id)
            elif new_decode_sm < old_decode_sm:
                if new_prefill_sm <= old_prefill_sm:
                    raise RuntimeError(
                        "invalid adaptive PDMux reallocation: Decode shrinks "
                        "without Prefill growing"
                    )
                shrink_lane = "decode"
                self._set_pdmux_lane_runtime(decode_lane, decode_runtime_id)
                self._set_pdmux_lane_runtime(prefill_lane, prefill_runtime_id)
            else:
                raise RuntimeError(
                    "adaptive PDMux profile switch has no shrinking lane: "
                    f"old={old_prefill_sm}/{old_decode_sm}, "
                    f"new={new_prefill_sm}/{new_decode_sm}"
                )
            pointer_commit_us = (time.perf_counter() - commit_started_at) * 1e6
            shared.active_profile_index = target_profile
            shared.pending_profile_index = None
            shared.profile_switch_armed = False
            shared.profile_epoch += 1
            committed_at = time.monotonic()
            total_wait_ms = (
                (committed_at - shared.profile_switch_requested_at) * 1e3
                if shared.profile_switch_requested_at
                else 0.0
            )
            prefill_drain_ms = (
                (shared.profile_switch_armed_at - shared.profile_switch_requested_at)
                * 1e3
                if shared.profile_switch_requested_at and shared.profile_switch_armed_at
                else 0.0
            )
            decode_pause_ms = (
                (committed_at - shared.profile_switch_armed_at) * 1e3
                if shared.profile_switch_armed_at
                else 0.0
            )
            shared.profile_switch_armed_at = 0.0
            shared.condition.notify_all()

        logger.info(
            "PDMux paired SM profile switched: epoch=%d old=%d(%d/%d) "
            "new=%d(%d/%d) total_wait=%.3f ms prefill_drain=%.3f ms "
            "decode_pause=%.3f ms shrink_lane=%s pointer_commit=%.3f us",
            shared.profile_epoch,
            old_profile,
            profile_splits[old_profile][0],
            profile_splits[old_profile][1],
            target_profile,
            profile_splits[target_profile][0],
            profile_splits[target_profile][1],
            total_wait_ms,
            prefill_drain_ms,
            decode_pause_ms,
            shrink_lane,
            pointer_commit_us,
        )
        self._record_decode_lane_mux_stat("adaptive_profile_switches")
        return True

    @staticmethod
    def _try_reserve_pdmux_decode_launch(
        shared: PrefillDecodeSchedulerState,
    ) -> bool:
        """Reserve the current profile before decode preparation leaves the lock."""
        with shared.condition:
            if shared.profile_switch_armed or shared.decode_launch_in_progress:
                shared.condition.notify_all()
                return False
            shared.decode_launch_in_progress = True
            return True

    def _try_reserve_pdmux_full_prefill(
        self,
        prefill_lane: PDDecodeLaneState,
        shared: PrefillDecodeSchedulerState,
    ) -> bool:
        """Reserve the non-Green-Context Prefill runtime while Decode is idle."""
        model_runner = getattr(getattr(self, "tp_worker", None), "model_runner", None)
        config = getattr(model_runner, "decode_lane_mux_config", None)
        if config is None:
            return False
        runtime_id = getattr(
            model_runner,
            "decode_lane_full_prefill_runtime_id",
            None,
        )
        if not config.adaptive_max_prefill_when_decode_idle or runtime_id is None:
            return False

        with shared.condition:
            decode_idle = (
                shared.decode_running_reqs == 0
                and shared.decode_executing_bs == 0
                and shared.decode_ready_reqs == 0
                and not shared.ready_batches
                and not shared.decode_launch_in_progress
                and not shared.profile_switch_armed
                and not shared.full_prefill_active
            )
            if not decode_idle:
                return False
            shared.full_prefill_active = True
            self._set_pdmux_lane_runtime(prefill_lane, runtime_id)
            shared.condition.notify_all()
        return True

    def _release_pdmux_full_prefill(
        self,
        prefill_lane: PDDecodeLaneState,
        shared: PrefillDecodeSchedulerState,
    ) -> bool:
        """Restore the active Green Context Prefill runtime and unblock Decode."""
        with shared.condition:
            if not shared.full_prefill_active:
                return False
            self._set_pdmux_lane_runtime(prefill_lane, shared.active_profile_index * 2)
            shared.full_prefill_active = False
            shared.condition.notify_all()
        return True

    @staticmethod
    def _arm_pdmux_profile_switch_after_prefill_drain(
        prefill_lane: PDDecodeLaneState,
        shared: PrefillDecodeSchedulerState,
    ) -> bool:
        """Start the short decode drain only after the long prefill drain ends."""
        with shared.condition:
            if (
                shared.pending_profile_index is None
                or shared.profile_switch_armed
                or prefill_lane.inflight_batch is not None
            ):
                return False
            shared.profile_switch_armed = True
            shared.profile_switch_armed_at = time.monotonic()
            shared.condition.notify_all()
            return True

    @staticmethod
    def _release_pdmux_decode_launch(
        shared: PrefillDecodeSchedulerState,
    ) -> None:
        with shared.condition:
            if shared.decode_launch_in_progress:
                shared.decode_launch_in_progress = False
                shared.condition.notify_all()

    def _admit_prefill_decode_ready_batches(
        self,
        decode_lane: PDDecodeLaneState,
        ready_batches: Deque[ScheduleBatch],
    ) -> bool:
        """Merge completed prefills only at a decode-step boundary."""
        if decode_lane.inflight_batch is not None:
            return False
        admitted = False
        while ready_batches:
            ready_batch = ready_batches.popleft()
            if decode_lane.running_batch.is_empty():
                decode_lane.running_batch = ready_batch
            else:
                decode_lane.running_batch.merge_batch(ready_batch)
            admitted = True
        self.running_batch = decode_lane.running_batch
        return admitted

    def _pdmux_batch_decodes_locally(self, batch: ScheduleBatch) -> bool:
        """Return whether an extend batch should retain KV for local decode."""
        return self.disaggregation_mode != DisaggregationMode.PREFILL or all(
            getattr(req, "local_decode", False) for req in batch.reqs
        )

    def _pdmux_prefill_destination_indices(
        self, batch: ScheduleBatch
    ) -> Tuple[List[int], List[int]]:
        """Partition a PDMux prefill without changing its execution order."""
        if self.disaggregation_mode != DisaggregationMode.PREFILL:
            return list(range(len(batch.reqs))), []
        local_indices = [
            i for i, req in enumerate(batch.reqs) if getattr(req, "local_decode", False)
        ]
        local_set = set(local_indices)
        remote_indices = [i for i in range(len(batch.reqs)) if i not in local_set]
        assert len(local_indices) + len(remote_indices) == len(batch.reqs)
        return local_indices, remote_indices

    def _advance_pdmux_remote_prefill(self) -> None:
        """Advance only the remote chunk owned by the PDMux prefill loop.

        Final remote chunks enter the transfer queue during result processing.
        With overlap disabled, an unfinished remote chunk is sent here, matching
        native PD prefill without exposing local requests to its batch-level
        state machine.
        """
        assert self.disaggregation_mode == DisaggregationMode.PREFILL
        req = self.chunked_req
        if req is None or getattr(req, "local_decode", False):
            return

        maybe_cache_unfinished_req(req, self.tree_cache, chunked=True)
        if not self.check_bootstrap(req):
            self.chunked_req = None
            return

        self.running_batch.batch_is_full = False
        if self.enable_overlap:
            req.tmp_end_idx = min(req.fill_len, len(req.origin_input_ids))
        else:
            self.send_kv_chunk(req)

    def _retire_pdmux_disagg_prefill_transfers(
        self, shared: PrefillDecodeSchedulerState
    ) -> bool:
        """Finalize completed remote prefills under the resource lock.

        Polling is deliberately kept outside both locks so an in-progress
        transfer (and its CPU-group collective) cannot stall the decode
        scheduler.  The native PD finalizer polls once more, but only after at
        least one request was observed terminal; all radix-cache and allocator
        mutations performed by that finalizer are serialized with local
        prefill/decode reservation and retirement. Native PD behavior remains
        unchanged.
        """
        if not self.disagg_prefill_inflight_queue:
            return False

        terminal_rids = self.get_transferred_rids()
        if not terminal_rids:
            return False

        with shared.resource_lock:
            done_reqs = self.process_disagg_prefill_inflight_queue(terminal_rids)
        if done_reqs:
            with shared.condition:
                shared.condition.notify_all()
        return bool(done_reqs)

    def _run_prefill_decode_scheduler_loop(
        self,
        decode_lane: PDDecodeLaneState,
        shared: PrefillDecodeSchedulerState,
    ) -> None:
        """Own the complete decode step lifecycle on one persistent thread."""
        model_runner = self.tp_worker.model_runner
        self.device_module.set_device(model_runner.gpu_id)

        try:
            with torch.no_grad():
                decode_done_observed_at = None
                while True:
                    with shared.state_lock:
                        if shared.stop:
                            return

                    made_progress = False
                    if (
                        decode_lane.inflight_batch is not None
                        and decode_lane.done_event is not None
                        and decode_lane.done_event.query()
                    ):
                        # query() only reports readiness; synchronize() on an
                        # already-ready event also attributes asynchronous CUDA
                        # faults to the lane that produced them.
                        decode_lane.done_event.synchronize()
                        decode_done_observed_at = (
                            time.perf_counter(),
                            time.thread_time(),
                        )
                        with self.metrics_reporter.profile_decode_scheduler_cpu(
                            "decode", "decode_result_processing"
                        ):
                            with shared.state_lock:
                                decode_lane.inflight_batch._pdmux_prefill_pressure = (
                                    shared.decode_inflight_prefill_pressure
                                )
                            with shared.resource_lock:
                                self.process_batch_result(
                                    decode_lane.inflight_batch,
                                    decode_lane.inflight_result,
                                )
                            with shared.condition:
                                decode_lane.inflight_batch = None
                                decode_lane.inflight_result = None
                                decode_lane.done_event = None
                                self.running_batch = decode_lane.running_batch
                                self._update_pdmux_decode_load(decode_lane, shared)
                                shared.condition.notify_all()
                            self._record_decode_lane_mux_stat(
                                "pd_lane_decode_completed_batches"
                            )
                        made_progress = True

                    if (
                        decode_lane.inflight_batch is None
                        and self._try_reserve_pdmux_decode_launch(shared)
                    ):
                        batch_to_launch = None
                        with self.metrics_reporter.profile_decode_scheduler_cpu(
                            "decode", "decode_batch_prepare"
                        ):
                            # Move ready batches out of the shared queue first.
                            # State and resource locks are deliberately never
                            # nested.
                            with shared.condition:
                                ready_batches = deque(shared.ready_batches)
                                shared.ready_batches.clear()
                                admitted_ready_reqs = sum(
                                    len(batch.reqs) for batch in ready_batches
                                )

                            # merge_batch(), update_running_batch(), and
                            # prepare_for_decode() may reserve/free request,
                            # KV, radix, and Mamba resources.
                            with shared.resource_lock:
                                self._admit_prefill_decode_ready_batches(
                                    decode_lane, ready_batches
                                )
                                if not decode_lane.running_batch.is_empty():
                                    decode_lane.running_batch = (
                                        self.update_running_batch(
                                            decode_lane.running_batch
                                        )
                                    )
                                self.running_batch = decode_lane.running_batch

                            with shared.condition:
                                self._update_pdmux_decode_load(
                                    decode_lane,
                                    shared,
                                    admitted_ready_reqs=admitted_ready_reqs,
                                )
                                if not decode_lane.running_batch.is_empty():
                                    batch_to_launch = decode_lane.running_batch
                                    decode_lane.inflight_batch = batch_to_launch.copy()

                        if batch_to_launch is not None:
                            decode_workload = self._estimate_pdmux_decode_workload(
                                batch_to_launch
                            )
                            with shared.state_lock:
                                shared.decode_workload = decode_workload
                                shared.decode_inflight_prefill_pressure = (
                                    shared.prefill_pressure
                                )
                                prefill_sm, decode_sm = self._get_pdmux_active_sm_split(
                                    shared
                                )
                            decode_lane.inflight_batch._pdmux_prefill_sm = int(
                                prefill_sm
                            )
                            decode_lane.inflight_batch._pdmux_decode_sm = int(decode_sm)
                            if decode_done_observed_at is not None:
                                now_wall = time.perf_counter()
                                now_cpu = time.thread_time()
                                self.metrics_reporter.record_decode_scheduler_interval(
                                    lane="decode",
                                    stage="decode_done_to_next_submit",
                                    cpu_t=now_cpu - decode_done_observed_at[1],
                                    wall_t=now_wall - decode_done_observed_at[0],
                                )
                                decode_done_observed_at = None
                            result, done_event = (
                                self.run_decode_batch_on_pd_lane_worker(
                                    batch_to_launch, decode_lane
                                )
                            )
                            with shared.condition:
                                decode_lane.inflight_result = result
                                decode_lane.done_event = done_event
                                launched_during_full_prefill = (
                                    shared.full_prefill_active
                                )
                                self._update_pdmux_decode_load(decode_lane, shared)
                                shared.condition.notify_all()
                            self._record_decode_lane_mux_stat(
                                "pd_lane_decode_launched_batches"
                            )
                            if launched_during_full_prefill:
                                self._record_decode_lane_mux_stat(
                                    "pd_lane_decode_launched_during_"
                                    "full_prefill_batches"
                                )
                            made_progress = True
                        else:
                            # Keep the latest non-empty decode resource vector
                            # for control. Clearing it during a sub-millisecond
                            # inter-batch gap makes a once-per-minute sample
                            # randomly alternate between loaded and idle
                            # capacity and causes profile oscillation. Zero
                            # Prefill demand still selects the smallest profile
                            # after a genuinely idle minute.
                            decode_done_observed_at = None

                        # If a batch was launched, inflight_batch now prevents a
                        # switch. If no batch was available, releasing this
                        # reservation lets the main thread commit immediately.
                        self._release_pdmux_decode_launch(shared)

                    if not made_progress:
                        # CUDA completion has no host notification. The short timed
                        # wait releases the GIL while bounding graph replay latency.
                        with shared.condition:
                            if not shared.stop:
                                shared.condition.wait(timeout=0.0002)
        except BaseException as exc:
            logger.exception("Independent PDMux decode scheduler failed")
            with shared.condition:
                shared.fatal_error = exc
                shared.stop = True
                shared.condition.notify_all()

    def _validate_prefill_decode_lane_mux(self) -> None:
        """Validate the deliberately small support envelope of PDMux."""
        if self.device != "cuda":
            raise ValueError("PDMux is CUDA-only")
        if self.ps.tp_size != 1:
            raise ValueError("PDMux currently supports TP=1 only")
        if self.enable_overlap:
            raise ValueError("PDMux requires --disable-overlap-schedule")
        if not self.spec_algorithm.is_none():
            raise ValueError("PDMux does not support speculative decoding")
        if not self.future_map.spec_algo.is_none():
            raise ValueError(
                "PDMux requires a non-speculative FutureMap; its speculative "
                "event and D2H staging state is not shared across lanes"
            )
        runtimes = self.tp_worker.model_runner.decode_lane_runtimes
        if len(runtimes) < 2 or any(
            runtime.sampler is None for runtime in runtimes[:2]
        ):
            raise ValueError("PDMux requires one private sampler per CUDA lane")
        if self.enable_hierarchical_cache:
            raise ValueError("PDMux does not support HiCache yet")
        if self.enable_lora_overlap_loading:
            raise ValueError("PDMux does not support overlapping LoRA loading")

    def _retire_pdmux_prefill(
        self,
        prefill_lane: PDDecodeLaneState,
        execution_batch: Optional[ScheduleBatch],
        shared: PrefillDecodeSchedulerState,
    ) -> Tuple[Optional[ScheduleBatch], bool]:
        """Retire a completed prefill and hand its live requests to decode."""
        if (
            prefill_lane.inflight_batch is None
            or prefill_lane.done_event is None
            or not prefill_lane.done_event.query()
        ):
            return execution_batch, False

        assert execution_batch is not None
        # The query above established readiness, so this does not add a GPU
        # bubble.  It prevents a failed prefill from being misreported by the
        # next decode graph replay on another stream.
        prefill_lane.done_event.synchronize()
        with self.metrics_reporter.profile_decode_scheduler_cpu(
            "prefill", "prefill_result_processing"
        ):
            # Result processing may release finished requests or update radix
            # and Mamba ownership. CUDA completion and the subsequent state
            # handoff stay outside this resource transaction.
            with shared.resource_lock:
                local_indices, remote_indices = self._pdmux_prefill_destination_indices(
                    execution_batch
                )
                self.process_batch_result(
                    prefill_lane.inflight_batch,
                    prefill_lane.inflight_result,
                )
                if remote_indices:
                    self._advance_pdmux_remote_prefill()
                ready_batch = None
                if local_indices:
                    local_batch = self._make_decode_lane_sub_batch(
                        execution_batch, local_indices
                    )
                    if (
                        local_batch.chunked_req is not None
                        and local_batch.chunked_req not in local_batch.reqs
                    ):
                        local_batch.chunked_req = None
                    ready_batch = self._prepare_decode_lane_extend_batch(local_batch)

            with shared.condition:
                if ready_batch is not None:
                    shared.ready_batches.append(ready_batch)
                    self._commit_pdmux_load_locked(
                        shared,
                        decode_ready_reqs=(
                            shared.decode_ready_reqs + len(ready_batch.reqs)
                        ),
                    )
                self.prefill_decode_inflight_batch = None
                shared.prefill_pressure = 0.0
                prefill_lane.inflight_batch = None
                prefill_lane.inflight_result = None
                prefill_lane.done_event = None
                shared.condition.notify_all()
            if self._release_pdmux_full_prefill(prefill_lane, shared):
                self._record_decode_lane_mux_stat(
                    "pd_lane_prefill_full_resource_completed_batches"
                )
            self._record_decode_lane_mux_stat("pd_lane_prefill_completed_batches")
            self._update_pdmux_prefill_load(shared)
        return None, True

    def _launch_pdmux_prefill(
        self,
        prefill_lane: PDDecodeLaneState,
        decode_lane: PDDecodeLaneState,
    ) -> Optional[ScheduleBatch]:
        """Build and submit at most one prefill while decode advances."""
        if prefill_lane.inflight_batch is not None:
            return None
        with self.prefill_decode_shared_state.state_lock:
            if self.prefill_decode_shared_state.pending_profile_index is not None:
                return None
        full_prefill_reserved = self._try_reserve_pdmux_full_prefill(
            prefill_lane, self.prefill_decode_shared_state
        )
        with self.metrics_reporter.profile_decode_scheduler_cpu(
            "prefill", "prefill_batch_prepare"
        ):
            # Batch construction reserves request/KV/Mamba slots and touches
            # radix-cache ownership. Keep only that reservation phase atomic
            # with decode preparation and retirement.
            with self.prefill_decode_shared_state.resource_lock:
                self.running_batch = decode_lane.running_batch
                prefill_batch = self.get_new_batch_prefill()
        if prefill_batch is None or prefill_batch.is_empty():
            if full_prefill_reserved:
                self._release_pdmux_full_prefill(
                    prefill_lane, self.prefill_decode_shared_state
                )
            return None

        assert prefill_batch.forward_mode.is_extend()
        prefill_pressure = self._estimate_pdmux_prefill_pressure(prefill_batch)
        with self.prefill_decode_shared_state.state_lock:
            self.prefill_decode_inflight_batch = prefill_batch
            self.prefill_decode_shared_state.prefill_pressure = prefill_pressure
            if getattr(decode_lane, "inflight_batch", None) is not None:
                self.prefill_decode_shared_state.decode_inflight_prefill_pressure = max(
                    self.prefill_decode_shared_state.decode_inflight_prefill_pressure,
                    prefill_pressure,
                )
        self._update_pdmux_prefill_load(self.prefill_decode_shared_state)
        if self.disaggregation_mode == DisaggregationMode.PREFILL:
            local_indices, remote_indices = self._pdmux_prefill_destination_indices(
                prefill_batch
            )
            self._record_decode_lane_mux_stat(
                "pd_lane_prefill_local_reqs", len(local_indices)
            )
            self._record_decode_lane_mux_stat(
                "pd_lane_prefill_remote_reqs", len(remote_indices)
            )
            if local_indices and remote_indices:
                self._record_decode_lane_mux_stat("pd_lane_prefill_mixed_batches")
        if (
            self.disaggregation_mode == DisaggregationMode.PREFILL
            and not self._pdmux_batch_decodes_locally(prefill_batch)
            and self.enable_staging
        ):
            self.maybe_prefetch_staging_for_batch(prefill_batch)
        with self.metrics_reporter.profile_decode_scheduler_cpu(
            "prefill", "prefill_host_submit"
        ):
            # Never hold state_lock or resource_lock while submitting CUDA
            # work. This is the latency-critical PDMux concurrency boundary.
            result = self.run_batch_on_pd_decode_lane(prefill_batch, prefill_lane)
        with self.prefill_decode_shared_state.state_lock:
            prefill_lane.inflight_batch = prefill_batch.copy()
            prefill_lane.inflight_result = result
        self._record_decode_lane_mux_stat("pd_lane_prefill_launched_batches")
        if full_prefill_reserved:
            self._record_decode_lane_mux_stat(
                "pd_lane_prefill_full_resource_launched_batches"
            )
            logger.info(
                "PDMux full-resource Prefill launched: runtime_id=%d batch_size=%d",
                prefill_lane.runtime_lane_id,
                len(prefill_batch.reqs),
            )
        return prefill_batch

    @DynamicGradMode()
    def event_loop_prefill_decode_lane_mux(self) -> None:
        """Run prefill and decode on independent schedulers and CUDA streams.

        The main thread owns request admission and prefill. A persistent worker
        owns the complete decode lifecycle, including result processing and the
        next CUDA graph replay. ``state_lock`` protects scheduler handoff state;
        ``resource_lock`` protects only request/KV/radix/Mamba ownership
        transitions. Neither lock may cover CUDA submission.
        """
        self._validate_prefill_decode_lane_mux()
        prefill_lane, decode_lane = self._init_prefill_decode_lane_states()
        # CUDA graph capture can take minutes when many profiles are
        # pre-created. Start the traffic window only after capture, otherwise
        # startup time would look like a zero-demand control interval.
        self._init_pdmux_adaptive_sm_calculation()
        shared = PrefillDecodeSchedulerState(
            load_publisher_thread_id=threading.get_ident(),
            active_profile_index=(prefill_lane.runtime_lane_id or 0) // 2,
        )
        self.prefill_decode_shared_state = shared
        prefill_execution_batch = None
        self.prefill_decode_inflight_batch = None
        self.last_batch = None
        self.running_batch = decode_lane.running_batch
        self._update_pdmux_prefill_load(shared)
        self._update_pdmux_decode_load(decode_lane, shared)
        self._publish_pdmux_load_if_dirty(shared, force=True)
        self.enable_staging = (
            self.disaggregation_mode == DisaggregationMode.PREFILL
            and envs.SGLANG_DISAGG_STAGING_BUFFER.get()
        )

        decode_thread = threading.Thread(
            target=self._run_prefill_decode_scheduler_loop,
            args=(decode_lane, shared),
            name="pdmux-decode-scheduler",
            daemon=True,
        )
        decode_thread.start()
        logger.info(
            "Initialized PDMux: prefill=id%s sm=%s, decode=id%s sm=%s thread=%s",
            prefill_lane.lane_id,
            self.tp_worker.model_runner.decode_lane_runtimes[
                prefill_lane.runtime_lane_id
            ].sm_count,
            decode_lane.lane_id,
            self.tp_worker.model_runner.decode_lane_runtimes[
                decode_lane.runtime_lane_id
            ].sm_count,
            decode_thread.name,
        )

        try:
            while True:
                with shared.state_lock:
                    fatal_error = shared.fatal_error
                if fatal_error is not None:
                    raise RuntimeError(
                        "Independent PDMux decode scheduler terminated"
                    ) from fatal_error

                recv_reqs = self.request_receiver.recv_requests()
                if recv_reqs:
                    # Abort handling can release live resources, so serialize
                    # it with decode reservation/retirement. Ordinary request
                    # admission remains owned by this prefill thread.
                    with shared.resource_lock:
                        self.process_input_requests(recv_reqs)
                    self._update_pdmux_prefill_load(shared)

                if self.disaggregation_mode == DisaggregationMode.PREFILL:
                    bootstrapped = (
                        self.disagg_prefill_bootstrap_queue.pop_bootstrapped()
                    )
                    if bootstrapped:
                        with shared.resource_lock:
                            self.waiting_queue.extend(bootstrapped)
                        self._update_pdmux_prefill_load(shared)
                        with shared.condition:
                            shared.condition.notify_all()
                    self._retire_pdmux_disagg_prefill_transfers(shared)

                prefill_execution_batch, made_progress = self._retire_pdmux_prefill(
                    prefill_lane,
                    prefill_execution_batch,
                    shared,
                )

                # A decision first drains only Prefill. Decode keeps using the
                # old paired profile until Prefill reaches its batch boundary.
                self._maybe_log_pdmux_adaptive_sm_prediction(shared)
                self._arm_pdmux_profile_switch_after_prefill_drain(prefill_lane, shared)
                if self._maybe_commit_pdmux_profile_switch(
                    prefill_lane, decode_lane, shared
                ):
                    made_progress = True

                launched_batch = self._launch_pdmux_prefill(
                    prefill_lane,
                    decode_lane,
                )
                if launched_batch is not None:
                    prefill_execution_batch = launched_batch
                    made_progress = True

                self._update_pdmux_prefill_load(shared)
                self._publish_pdmux_load_if_dirty(shared)

                if not made_progress:
                    # Yield the GIL without adding visible request polling latency.
                    with shared.condition:
                        if shared.fatal_error is None:
                            shared.condition.wait(timeout=0.0005)
        finally:
            with shared.condition:
                shared.stop = True
                shared.condition.notify_all()
            decode_thread.join(timeout=5.0)
