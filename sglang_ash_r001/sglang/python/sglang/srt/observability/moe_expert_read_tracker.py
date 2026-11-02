"""Track unique routed experts from the existing MoE alignment histogram."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator, Optional

import torch


@dataclass(frozen=True)
class ExpertReadSnapshot:
    """Pinned-host alignment scratch copied after the measured forward."""

    alignment_scratch_cpu: Optional[torch.Tensor]
    active_layer_ids: tuple[int, ...]
    alignment_num_experts: int
    num_routed_experts: int
    read_bytes_per_expert: int

    @property
    def total_unique_experts(self) -> int:
        if self.alignment_scratch_cpu is None or not self.active_layer_ids:
            return 0

        # The align op receives E + 1 buckets: bucket 0 is the -1 sentinel and
        # buckets 1..E correspond to physical expert ids 0..E-1.  Its prefix
        # scratch occupies E + 2 entries; the fused raw histogram follows it.
        raw_count_offset = self.alignment_num_experts + 2
        first_routed_count = raw_count_offset + 1
        last_routed_count = first_routed_count + self.num_routed_experts
        counts = self.alignment_scratch_cpu[
            list(self.active_layer_ids), first_routed_count:last_routed_count
        ]
        return int(torch.count_nonzero(counts).item())

    @property
    def read_bytes(self) -> int:
        return self.total_unique_experts * self.read_bytes_per_expert


_active_tracker: ContextVar[Optional["ExpertReadTracker"]] = ContextVar(
    "active_expert_read_tracker", default=None
)
_active_layer_id: ContextVar[Optional[int]] = ContextVar(
    "active_expert_read_layer_id", default=None
)


class ExpertReadTracker:
    """Persistent per-stream/graph storage for fused routing histograms."""

    def __init__(
        self,
        *,
        num_layers: int,
        num_experts: int,
        read_bytes_per_expert: int,
        device: torch.device | str,
        record_enabled: bool = True,
    ):
        if num_layers <= 0 or num_experts <= 0:
            raise ValueError("num_layers and num_experts must be positive")
        if read_bytes_per_expert <= 0:
            raise ValueError("read_bytes_per_expert must be positive")
        self.num_layers = int(num_layers)
        self.num_experts = int(num_experts)
        self.read_bytes_per_expert = int(read_bytes_per_expert)
        self.device = torch.device(device)
        self.record_enabled = bool(record_enabled)
        self._alignment_num_experts: Optional[int] = None
        self._alignment_scratch: Optional[torch.Tensor] = None
        self._active_layer_ids: set[int] = set()

    def reset(self) -> None:
        # CUDA graph replay executes no Python; the layer set and persistent
        # graph buffers captured here must therefore survive replay.  Eager
        # forwards repopulate the set on every invocation.
        if not torch.cuda.is_current_stream_capturing():
            self._active_layer_ids.clear()

    @contextmanager
    def capture_layer(self, layer_id: int) -> Iterator[None]:
        if not self.record_enabled:
            yield
            return
        if not 0 <= layer_id < self.num_layers:
            raise ValueError(
                f"layer_id must be in [0, {self.num_layers}), got {layer_id}"
            )
        tracker_token = _active_tracker.set(self)
        layer_token = _active_layer_id.set(int(layer_id))
        try:
            yield
        finally:
            _active_layer_id.reset(layer_token)
            _active_tracker.reset(tracker_token)

    def alignment_scratch_for_layer(
        self, *, layer_id: int, alignment_num_experts: int
    ) -> torch.Tensor:
        if alignment_num_experts < self.num_experts:
            raise ValueError(
                "alignment expert count cannot be smaller than routed expert count"
            )
        if self._alignment_scratch is None:
            # The first E+2 entries preserve the align op's existing cumsum
            # scratch.  The second E+2 entries hold raw assignment counts.
            self._alignment_num_experts = int(alignment_num_experts)
            self._alignment_scratch = torch.empty(
                (self.num_layers, 2 * (alignment_num_experts + 2)),
                dtype=torch.int32,
                device=self.device,
            )
        elif self._alignment_num_experts != alignment_num_experts:
            raise RuntimeError(
                "all tracked MoE layers must use the same physical expert count; "
                f"got {alignment_num_experts} after {self._alignment_num_experts}"
            )
        self._active_layer_ids.add(layer_id)
        return self._alignment_scratch[layer_id]

    def finalize(self) -> ExpertReadSnapshot:
        """Enqueue one D2H copy of the existing alignment scratch."""
        if self._alignment_scratch is None:
            return ExpertReadSnapshot(
                alignment_scratch_cpu=None,
                active_layer_ids=(),
                alignment_num_experts=0,
                num_routed_experts=self.num_experts,
                read_bytes_per_expert=self.read_bytes_per_expert,
            )

        scratch_cpu = torch.empty(
            self._alignment_scratch.shape,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        scratch_cpu.copy_(self._alignment_scratch, non_blocking=True)
        return ExpertReadSnapshot(
            alignment_scratch_cpu=scratch_cpu,
            active_layer_ids=tuple(sorted(self._active_layer_ids)),
            alignment_num_experts=int(self._alignment_num_experts),
            num_routed_experts=self.num_experts,
            read_bytes_per_expert=self.read_bytes_per_expert,
        )


def get_current_moe_alignment_scratch(
    *, alignment_num_experts: int
) -> Optional[torch.Tensor]:
    """Return the active layer's persistent scratch, if metrics are enabled."""
    tracker = _active_tracker.get()
    layer_id = _active_layer_id.get()
    if tracker is None or layer_id is None or not tracker.record_enabled:
        return None
    return tracker.alignment_scratch_for_layer(
        layer_id=layer_id, alignment_num_experts=alignment_num_experts
    )
