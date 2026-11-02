from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml


@dataclass
class DecodeLaneMuxConfig:
    lane_num: int = 2
    policy: str = "short_long_by_seq_len"
    short_context_threshold: int = 4096
    sm_group_num: int = 8
    manual_divisions: list[list[int]] = field(default_factory=list)
    long_attention_active_cta_ratio: float = 0.0
    long_attention_active_cta_limit: int = 0
    long_attention_active_cta_per_sm: int = 2
    long_moe_active_cta_ratio: float = 0.8
    long_moe_active_cta_limit: int = 0
    long_linear_attn_active_cta_limit: int = 0
    short_low_resource_linear: bool = False
    long_low_resource_linear: bool = False
    use_long_cap_graph: bool = True
    enable_green_ctx: bool = False
    short_green_ctx_sm_count: int = 0
    long_green_ctx_sm_count: int = 0
    enable_green_ctx_profile_switch: bool = False
    catchup_short_green_ctx_sm_count: int = 0
    catchup_long_green_ctx_sm_count: int = 0
    short_heavy_budget_steps: int = 0
    long_heavy_budget_steps: int = 0
    enable_green_ctx_budget_full_fallback: bool = False
    forced_full_cooldown_steps: int = 0
    unified_full_min_steps_before_prefill: int = 0
    split_max_short_fraction: float = 1.0
    normal_independent_decode_lanes: bool = False
    # Run one normal EXTEND lane concurrently with one decode lane.  Unlike
    # upstream PDMux this never changes the batch to SPLIT_PREFILL: the prefill
    # forward traverses the complete model once and hands the same Req/KV
    # mapping to the decode scheduler after a CUDA event completes.
    prefill_decode_lanes: bool = False
    # Analytical controller. It can run as shadow-only calculation or drive
    # pre-created paired Green Context profiles.
    adaptive_prefill_sm_calculation: bool = False
    # Pre-create one paired Green Context profile for every adaptive split and
    # switch both logical lanes together at a drained batch boundary.
    adaptive_sm_switching: bool = False
    adaptive_sm_splits: list[list[int]] = field(default_factory=list)
    adaptive_window_seconds: float = 60.0
    adaptive_control_interval_seconds: float = 60.0
    adaptive_target_utilization: float = 0.85
    # Convert queued Prefill work into an additional throughput demand by
    # requiring that the backlog can be drained within this interval.
    adaptive_backlog_drain_seconds: float = 0.0
    # Protect latency-sensitive Prefill requests by reserving at least this
    # many SMs while any high-priority request is queued or executing.
    adaptive_high_priority_prefill_sm_floor: int = 0
    # While the local decoder has no running, executing, or ready request, run
    # Prefill on an ordinary full-resource CUDA stream outside Green Context.
    # A Decode request arriving during that batch may launch immediately on
    # its Green Context stream; after the batch, Prefill returns to the paired
    # Green Context stream while Decode remains active.
    adaptive_max_prefill_when_decode_idle: bool = False
    # Upscaling is immediate; a lower-SM profile must remain desirable for
    # this long before it is allowed to replace the active profile.
    adaptive_downscale_hold_seconds: float = 0.0


@dataclass
class DecodeLaneRuntime:
    """Runtime state owned by one decode lane.

    The two-lane path should pass this object explicitly from the scheduler or
    benchmark entry point. The thread-local helpers below are kept only as a
    compatibility fallback while the rest of the stack is being migrated.
    """

    lane_id: int
    name: str
    stream: Any = None
    sm_count: int = 0
    attn_backend: Any = None
    tp_group: Any = None
    graph_pool: Any = None
    cuda_graph_runner: Any = None
    cuda_graph_runners: dict[str, Any] = field(default_factory=dict)
    # Sampling may run concurrently on different PDMux streams. Keep any
    # backend RNG/workspace/module state lane-private instead of sharing the
    # ModelRunner's default sampler.
    sampler: Any = None
    short_done_event: Any = None
    _eager_decode_registry: Any = None
    _eager_prefill_registry: Any = None


DECODE_LANES: list[DecodeLaneRuntime] = []
_THREAD_LOCAL = threading.local()


def load_decode_lane_mux_config(config_path: Optional[str]) -> DecodeLaneMuxConfig:
    if not config_path:
        return DecodeLaneMuxConfig()

    with open(config_path, "r") as f:
        raw = yaml.safe_load(f) or {}

    lane_num = raw.get("lane_num", 2)
    if lane_num != 2:
        raise ValueError("DecodeLaneMux currently supports exactly two lanes.")
    if int(raw.get("unified_full_min_steps_before_prefill", 0)) < 0:
        raise ValueError("unified_full_min_steps_before_prefill cannot be negative.")
    if bool(raw.get("prefill_decode_lanes", False)) and bool(
        raw.get("normal_independent_decode_lanes", False)
    ):
        raise ValueError(
            "prefill_decode_lanes and normal_independent_decode_lanes are mutually exclusive"
        )
    if bool(raw.get("adaptive_sm_switching", False)):
        if not bool(raw.get("adaptive_prefill_sm_calculation", False)):
            raise ValueError(
                "adaptive_sm_switching requires adaptive_prefill_sm_calculation"
            )
        if not bool(raw.get("prefill_decode_lanes", False)):
            raise ValueError("adaptive_sm_switching requires prefill_decode_lanes")
        if not bool(raw.get("enable_green_ctx", False)):
            raise ValueError("adaptive_sm_switching requires enable_green_ctx")
        if bool(raw.get("enable_green_ctx_profile_switch", False)):
            raise ValueError(
                "adaptive_sm_switching and enable_green_ctx_profile_switch "
                "are mutually exclusive"
            )

    # ASH r001: long_green_ctx_sm_count = -1 (long lane = all remaining SMs, any size) is only defined for plain two-lane decode mux
    if int(raw.get("long_green_ctx_sm_count", 0)) == -1 and (
        bool(raw.get("prefill_decode_lanes", False))
        or bool(raw.get("adaptive_sm_switching", False))
        or bool(raw.get("enable_green_ctx_profile_switch", False))
    ):
        raise ValueError(
            "long_green_ctx_sm_count=-1 is only supported for plain two-lane decode mux "
            "(no prefill_decode_lanes / adaptive_sm_switching / enable_green_ctx_profile_switch)"
        )

    return DecodeLaneMuxConfig(
        lane_num=lane_num,
        policy=raw.get("policy", "short_long_by_seq_len"),
        short_context_threshold=raw.get("short_context_threshold", 4096),
        sm_group_num=raw.get("sm_group_num", 8),
        manual_divisions=raw.get("manual_divisions", []),
        long_attention_active_cta_ratio=raw.get("long_attention_active_cta_ratio", 0.0),
        long_attention_active_cta_limit=raw.get("long_attention_active_cta_limit", 0),
        long_attention_active_cta_per_sm=raw.get("long_attention_active_cta_per_sm", 2),
        long_moe_active_cta_ratio=raw.get("long_moe_active_cta_ratio", 0.8),
        long_moe_active_cta_limit=raw.get("long_moe_active_cta_limit", 0),
        long_linear_attn_active_cta_limit=raw.get(
            "long_linear_attn_active_cta_limit", 0
        ),
        short_low_resource_linear=bool(raw.get("short_low_resource_linear", False)),
        long_low_resource_linear=bool(raw.get("long_low_resource_linear", False)),
        use_long_cap_graph=bool(raw.get("use_long_cap_graph", True)),
        enable_green_ctx=bool(raw.get("enable_green_ctx", False)),
        short_green_ctx_sm_count=int(raw.get("short_green_ctx_sm_count", 0)),
        long_green_ctx_sm_count=int(raw.get("long_green_ctx_sm_count", 0)),
        enable_green_ctx_profile_switch=bool(
            raw.get("enable_green_ctx_profile_switch", False)
        ),
        catchup_short_green_ctx_sm_count=int(
            raw.get("catchup_short_green_ctx_sm_count", 0)
        ),
        catchup_long_green_ctx_sm_count=int(
            raw.get("catchup_long_green_ctx_sm_count", 0)
        ),
        short_heavy_budget_steps=int(raw.get("short_heavy_budget_steps", 0)),
        long_heavy_budget_steps=int(raw.get("long_heavy_budget_steps", 0)),
        enable_green_ctx_budget_full_fallback=bool(
            raw.get("enable_green_ctx_budget_full_fallback", False)
        ),
        forced_full_cooldown_steps=int(raw.get("forced_full_cooldown_steps", 0)),
        unified_full_min_steps_before_prefill=int(
            raw.get("unified_full_min_steps_before_prefill", 0)
        ),
        split_max_short_fraction=float(raw.get("split_max_short_fraction", 1.0)),
        normal_independent_decode_lanes=bool(
            raw.get("normal_independent_decode_lanes", False)
        ),
        prefill_decode_lanes=bool(raw.get("prefill_decode_lanes", False)),
        adaptive_prefill_sm_calculation=bool(
            raw.get("adaptive_prefill_sm_calculation", False)
        ),
        adaptive_sm_switching=bool(raw.get("adaptive_sm_switching", False)),
        adaptive_sm_splits=raw.get("adaptive_sm_splits", []),
        adaptive_window_seconds=float(raw.get("adaptive_window_seconds", 60.0)),
        adaptive_control_interval_seconds=float(
            raw.get("adaptive_control_interval_seconds", 60.0)
        ),
        adaptive_target_utilization=float(raw.get("adaptive_target_utilization", 0.85)),
        adaptive_backlog_drain_seconds=float(
            raw.get("adaptive_backlog_drain_seconds", 0.0)
        ),
        adaptive_high_priority_prefill_sm_floor=int(
            raw.get("adaptive_high_priority_prefill_sm_floor", 0)
        ),
        adaptive_max_prefill_when_decode_idle=bool(
            raw.get("adaptive_max_prefill_when_decode_idle", False)
        ),
        adaptive_downscale_hold_seconds=float(
            raw.get("adaptive_downscale_hold_seconds", 0.0)
        ),
    )


def _resolve_green_ctx_sm_counts(
    short_sm_count: int,
    long_sm_count: int,
    total_sm_count: int,
    multiple: int,
) -> tuple[int, int]:
    # ASH r001: long_sm_count == -1 -> the long lane takes ALL remaining SMs, even if that is not a multiple of the architecture
    # granularity (built through the driver API, see greenctx_driver.py); e.g. short=128 on a 132-SM H200 -> long = 4.
    if long_sm_count == -1:
        if short_sm_count <= 0 or short_sm_count % multiple != 0 or short_sm_count >= total_sm_count:
            raise ValueError(f"long_green_ctx_sm_count=-1 needs 0 < short_green_ctx_sm_count < {total_sm_count}, "
                             f"multiple of {multiple}; got {short_sm_count}")
        return short_sm_count, total_sm_count - short_sm_count
    if short_sm_count <= 0 and long_sm_count <= 0:
        short_sm_count = (total_sm_count // 2) // multiple * multiple
        long_sm_count = (total_sm_count - short_sm_count) // multiple * multiple
    elif short_sm_count <= 0:
        short_sm_count = (total_sm_count - long_sm_count) // multiple * multiple
    elif long_sm_count <= 0:
        long_sm_count = (total_sm_count - short_sm_count) // multiple * multiple

    for name, sm_count in (
        ("short_green_ctx_sm_count", short_sm_count),
        ("long_green_ctx_sm_count", long_sm_count),
    ):
        if sm_count <= 0:
            raise ValueError(f"{name} must be positive when green ctx is enabled")
        if sm_count % multiple != 0:
            raise ValueError(
                f"{name}={sm_count} must be a multiple of {multiple} "
                "for this GPU architecture"
            )

    if short_sm_count + long_sm_count > total_sm_count:
        raise ValueError(
            "decode lane green ctx SM counts exceed available SMs: "
            f"short={short_sm_count}, long={long_sm_count}, total={total_sm_count}"
        )
    return short_sm_count, long_sm_count


def create_decode_lane_green_ctx_streams_by_count(
    short_sm_count: int, long_sm_count: int, gpu_id: int
) -> tuple[list[Any], tuple[int, int]]:
    import torch

    from sgl_kernel import spatial
    from sglang.srt.multiplex.pdmux_context import get_arch_constraints

    device = torch.cuda.current_device()
    _, multiple = get_arch_constraints(torch.cuda.get_device_capability(device))
    total_sm_count = int(spatial.get_sm_available(gpu_id))
    short_sm_count, long_sm_count = _resolve_green_ctx_sm_counts(
        int(short_sm_count), int(long_sm_count), total_sm_count, multiple
    )
    if long_sm_count % multiple != 0:          # ASH r001: remainder lane (long_green_ctx_sm_count = -1)
        from sglang.srt.multiplex.greenctx_driver import create_split_streams

        return create_split_streams(short_sm_count, gpu_id)
    streams = spatial.create_greenctx_stream_by_value(
        short_sm_count, long_sm_count, gpu_id
    )
    return list(streams), (short_sm_count, long_sm_count)


def create_decode_lane_green_ctx_streams(
    config: DecodeLaneMuxConfig, gpu_id: int
) -> tuple[list[Any], tuple[int, int]]:
    """Create green-context streams for the short and long decode lanes."""
    if not config.enable_green_ctx:
        return [], (0, 0)

    return create_decode_lane_green_ctx_streams_by_count(
        config.short_green_ctx_sm_count, config.long_green_ctx_sm_count, gpu_id
    )


def set_decode_lanes(lanes: list[DecodeLaneRuntime]) -> None:
    global DECODE_LANES
    DECODE_LANES = lanes
    _THREAD_LOCAL.current_decode_lane_idx = 0


def get_decode_lanes() -> list[DecodeLaneRuntime]:
    return DECODE_LANES


def set_current_decode_lane_idx(idx: int) -> None:
    if idx < 0 or idx >= len(DECODE_LANES):
        raise ValueError(f"Invalid decode lane index: {idx}")
    _THREAD_LOCAL.current_decode_lane_idx = idx


def get_current_decode_lane_idx() -> int:
    return getattr(_THREAD_LOCAL, "current_decode_lane_idx", 0)


def get_current_decode_lane() -> Optional[DecodeLaneRuntime]:
    if not DECODE_LANES:
        return None
    return DECODE_LANES[get_current_decode_lane_idx()]
