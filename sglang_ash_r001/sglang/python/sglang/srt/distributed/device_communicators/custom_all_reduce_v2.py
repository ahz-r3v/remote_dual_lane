import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, TypeVar

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.jit_kernel.all_reduce import AllReduceAlgo, get_custom_all_reduce_cls
from sglang.srt.distributed import is_in_piecewise_cuda_graph
from sglang.srt.distributed.device_communicators.custom_all_reduce_utils import (
    can_use_custom_all_reduce_with_nvlink,
    is_weak_contiguous,
)
from sglang.srt.utils import is_sm100_supported, log_info_on_rank0

logger = logging.getLogger(__name__)

T = TypeVar("T")

INF = 1 << 60


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class ModeConfig:
    one_shot_push_threshold: int  # below this, use one-shot push
    one_shot_pull_threshold: int  # below this, use one-shot pull


class CustomAllReduceV2:
    def __init__(
        self,
        group: ProcessGroup,
        device: torch.device,
        max_pull_size: Optional[int] = None,
        max_push_size: Optional[int] = None,
        max_pull_blocks: Optional[int] = None,
        max_push_blocks: Optional[int] = None,
    ) -> None:
        _maybe_init_config()
        self.disabled = True
        self.cta_size = 256
        self._debug_custom_ar = _env_flag("SGLANG_DEBUG_CUSTOM_AR")
        self._debug_custom_ar_decision = _env_flag("SGLANG_DEBUG_CUSTOM_AR_DECISION")
        self._debug_custom_ar_limit = _env_int("SGLANG_DEBUG_CUSTOM_AR_LIMIT", 512)
        self._debug_custom_ar_count = 0
        self._debug_custom_ar_decision_count = 0
        if not can_use_custom_all_reduce_v2(group=group, device=device):
            return

        self.group = group
        self.rank = dist.get_rank(group=self.group)
        self.world_size = dist.get_world_size(group=self.group)
        if max_pull_size is None:  # default to 16MB
            max_pull_size = 16 * 1024 * 1024
        if max_push_size is None:  # default to recommended size
            config = THRESHOLD_2_SHOT_MAP[self.world_size]
            max_push_size = config.one_shot_push_threshold
        self.max_pull_size = max_pull_size
        self.max_push_size = max_push_size
        self.max_size = max(max_pull_size, max_push_size)
        num_sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.max_pull_blocks = num_sms if max_pull_blocks is None else max_pull_blocks
        self.max_push_blocks = num_sms if max_push_blocks is None else max_push_blocks
        self.override_shot(None)  # set default config based on world size
        self.override_algo: Optional[AllReduceAlgo] = None
        self.obj = get_custom_all_reduce_cls()(
            rank=self.rank,
            world_size=self.world_size,
            pull_buffer_bytes=self.max_pull_size,
            push_buffer_bytes=self.max_push_size,
            graph_input_count=131072,
            max_pull_blocks=max_pull_blocks,
            max_push_blocks=max_push_blocks,
        )
        self._post_init_obj()
        self.disabled = False
        log_info_on_rank0(logger, "Custom allreduce v2 initialized successfully")
        if self._debug_custom_ar:
            self._debug_print(
                "init",
                push_threshold=self.config.one_shot_push_threshold,
                pull_threshold=self.config.one_shot_pull_threshold,
                max_push_size=self.max_push_size,
                max_pull_size=self.max_pull_size,
                max_push_blocks=self.max_push_blocks,
                max_pull_blocks=self.max_pull_blocks,
                cta_size=self.cta_size,
            )

    def override_shot(self, shot: int | None):
        if shot is None:
            config = THRESHOLD_2_SHOT_MAP[self.world_size]
        else:
            assert shot in (1, 2)
            threshold = INF if shot == 1 else 0
            config = replace(self.config, one_shot_pull_threshold=threshold)
        # need to clip the config thresholds to max sizes to avoid invalid config
        push_threshold = min(config.one_shot_push_threshold, self.max_push_size)
        pull_threshold = min(config.one_shot_pull_threshold, self.max_pull_size)
        self.config: ModeConfig = replace(
            config,
            one_shot_push_threshold=push_threshold,
            one_shot_pull_threshold=pull_threshold,
        )

    @contextmanager
    def capture(self):
        if self.disabled:
            yield
            return
        try:
            self.obj.set_cuda_graph_capture(True)
            yield
        finally:
            self.obj.set_cuda_graph_capture(False)
        # cannot call when graph is capturing
        assert (
            torch.cuda.is_current_stream_capturing() == False
        ), "Cannot register graph inputs while capturing CUDA graph"
        pairs = self.obj.share_graph_inputs()
        handles = [handle for _, handle in pairs]
        offsets = [offset for offset, _ in pairs]
        handles_all = self._share_list(handles)
        offsets_all = self._share_list(offsets)
        result = [list(zip(o, h)) for o, h in zip(offsets_all, handles_all)]
        self.obj.register_inputs(result)
        log_info_on_rank0(logger, f"Registering {len(pairs)} cuda graph addresses")

    def should_custom_ar(self, inp: torch.Tensor) -> bool:
        """Check if the input tensor is suitable for custom all-reduce."""
        inp_size = inp.numel() * inp.element_size()
        if self.disabled:
            self._debug_print_decision(inp, False, "disabled")
            return False
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            self._debug_print_decision(inp, False, "size_not_multiple_of_16")
            return False
        if not is_weak_contiguous(inp):
            self._debug_print_decision(inp, False, "not_weak_contiguous")
            return False
        enabled = inp_size <= self.max_size
        self._debug_print_decision(
            inp, enabled, "custom_ar" if enabled else "larger_than_max_size"
        )
        return enabled

    def custom_all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        if is_in_piecewise_cuda_graph():  # disable inplace optimization
            try:
                self.obj.set_cuda_graph_capture(False)
                return self._all_reduce(input)
            finally:
                self.obj.set_cuda_graph_capture(True)
        return self._all_reduce(input)

    def close(self):
        if not self.disabled and hasattr(self, "obj"):
            self.obj.free(self.group)

    def _all_reduce(self, input: torch.Tensor) -> torch.Tensor:
        """Perform the actual all-reduce via JIT kernel."""
        algo = self._determine_algo(input)
        self._debug_print_all_reduce(input, algo)
        return torch.from_dlpack(self.obj.all_reduce(input, algo))

    def _determine_algo(self, input: torch.Tensor) -> AllReduceAlgo:
        if self.override_algo is not None:
            return self.override_algo
        if _env_flag("SGLANG_CUSTOM_AR_FORCE_PULL", True):
            return AllReduceAlgo.ONE_SHOT_PULL
        input_bytes = input.numel() * input.element_size()
        if input_bytes <= self.config.one_shot_push_threshold:
            return AllReduceAlgo.ONE_SHOT_PUSH
        if input_bytes <= self.config.one_shot_pull_threshold:
            return AllReduceAlgo.ONE_SHOT_PULL
        else:
            return AllReduceAlgo.TWO_SHOT_PULL

    def _debug_print_decision(
        self, input: torch.Tensor, enabled: bool, reason: str
    ) -> None:
        if not self._debug_custom_ar_decision:
            return
        if self._debug_custom_ar_decision_count >= self._debug_custom_ar_limit:
            return
        self._debug_custom_ar_decision_count += 1
        self._debug_print(
            "decision",
            enabled=enabled,
            reason=reason,
            shape=tuple(input.shape),
            dtype=str(input.dtype),
            bytes=input.numel() * input.element_size(),
            max_size=getattr(self, "max_size", None),
        )

    def _debug_print_all_reduce(
        self, input: torch.Tensor, algo: AllReduceAlgo
    ) -> None:
        if not self._debug_custom_ar:
            return
        if self._debug_custom_ar_count >= self._debug_custom_ar_limit:
            return
        self._debug_custom_ar_count += 1

        input_bytes = input.numel() * input.element_size()
        pull_blocks = None
        if not algo.is_push():
            vec_size = 16 // (input.element_size() * 2)
            items_per_block = self.cta_size * vec_size * 2
            needed_blocks = (input.numel() + items_per_block - 1) // items_per_block
            pull_blocks = min(needed_blocks, self.max_pull_blocks)

        self._debug_print(
            "all_reduce",
            idx=self._debug_custom_ar_count,
            algo=algo.name,
            path="push" if algo.is_push() else "pull",
            shape=tuple(input.shape),
            dtype=str(input.dtype),
            bytes=input_bytes,
            numel=input.numel(),
            push_threshold=self.config.one_shot_push_threshold,
            pull_threshold=self.config.one_shot_pull_threshold,
            push_blocks=self.max_push_blocks if algo.is_push() else None,
            pull_blocks=pull_blocks,
            data_ptr=hex(input.data_ptr()),
        )

    def _debug_print(self, event: str, **fields) -> None:
        capturing = "unknown"
        stream = "unknown"
        try:
            capturing = torch.cuda.is_current_stream_capturing()
        except Exception:
            pass
        try:
            stream = torch.cuda.current_stream().cuda_stream
        except Exception:
            pass
        prefix = (
            f"[custom-ar-v2] event={event} pid={os.getpid()} "
            f"local_rank={os.environ.get('LOCAL_RANK', '?')} "
            f"group_rank={getattr(self, 'rank', '?')} "
            f"world_size={getattr(self, 'world_size', '?')} "
            f"obj={id(self)} capturing={capturing} stream={stream}"
        )
        body = " ".join(f"{key}={value}" for key, value in fields.items())
        print(f"{prefix} {body}", flush=True)

    def _post_init_obj(self):
        handles = [self.obj.share_storage()]
        result = self._share_list(handles)
        assert all(len(r) == 1 for r in result)
        result = [h[0] for h in result]
        self.obj.post_init(result)

    def _share_list(self, input: List[T]) -> List[List[T]]:
        input_tensor = torch.tensor(input, dtype=torch.int64, device="cpu")
        gather_list = [torch.empty_like(input_tensor) for _ in range(self.world_size)]
        dist.all_gather(gather_list, input_tensor, group=self.group)
        return [g.tolist() for g in gather_list]

    def __del__(self):
        self.close()


def _maybe_init_config():
    global THRESHOLD_2_SHOT_MAP
    if THRESHOLD_2_SHOT_MAP:
        return
    KB, MB = 1024, 1024 * 1024

    if is_sm100_supported():
        # NOTE: This result is based on benchmarks on B200 GPUs
        THRESHOLD_2_SHOT_MAP = {
            2: ModeConfig(4 * MB, INF),
            3: ModeConfig(4 * MB, 4 * MB),
            4: ModeConfig(2 * MB, 2 * MB),
            5: ModeConfig(2 * MB, 2 * MB),
            6: ModeConfig(1 * MB, 1 * MB),
            7: ModeConfig(896 * KB, 896 * KB),
            8: ModeConfig(720 * KB, 720 * KB),
        }
    else:
        # NOTE: This result is based on benchmarks on H200 GPUs
        THRESHOLD_2_SHOT_MAP = {
            2: ModeConfig(2 * MB, INF),
            3: ModeConfig(512 * KB, 512 * KB),
            4: ModeConfig(384 * KB, 256 * KB),
            5: ModeConfig(256 * KB, 256 * KB),
            6: ModeConfig(192 * KB, 192 * KB),
            7: ModeConfig(192 * KB, 192 * KB),
            8: ModeConfig(160 * KB, 160 * KB),
        }
    # TODO: tune on more GPUs, e.g A100


def can_use_custom_all_reduce_v2(
    group: ProcessGroup,
    device: torch.device,
) -> bool:
    # call _maybe_init_config() to ensure THRESHOLD_2_SHOT_MAP is initialized, since can_use_custom_all_reduce_v2 can be called before CustomAllReduceV2 is initialized
    _maybe_init_config()
    full_nvlink = can_use_custom_all_reduce_with_nvlink(
        group=group,
        device=device,
        supported_world_size=list(THRESHOLD_2_SHOT_MAP.keys()),
        cls_name="CustomAllReduceV2",
    )
    return full_nvlink is True


THRESHOLD_2_SHOT_MAP: Dict[int, ModeConfig] = {}
