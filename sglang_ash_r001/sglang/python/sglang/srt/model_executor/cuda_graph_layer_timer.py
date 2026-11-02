from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class CudaGraphLayerTimerEntry:
    layer_id: int
    phase: str
    start: torch.cuda.Event
    end: torch.cuda.Event


class CudaGraphLayerTimerRecorder:
    def __init__(self, name: str):
        self.name = name
        self.entries: list[CudaGraphLayerTimerEntry] = []

    @contextmanager
    def time(self, layer_id: int, phase: str):
        start = torch.cuda.Event(enable_timing=True, external=True)
        end = torch.cuda.Event(enable_timing=True, external=True)
        start.record(torch.cuda.current_stream())
        try:
            yield
        finally:
            end.record(torch.cuda.current_stream())
            self.entries.append(CudaGraphLayerTimerEntry(layer_id, phase, start, end))

    def elapsed_rows(self) -> list[dict]:
        torch.cuda.synchronize()
        rows = []
        for entry in self.entries:
            rows.append(
                {
                    "timer": self.name,
                    "layer_id": entry.layer_id,
                    "phase": entry.phase,
                    "ms": float(entry.start.elapsed_time(entry.end)),
                }
            )
        return rows


_ACTIVE_RECORDER: Optional[CudaGraphLayerTimerRecorder] = None


def set_cuda_graph_layer_timer_recorder(
    recorder: Optional[CudaGraphLayerTimerRecorder],
) -> Optional[CudaGraphLayerTimerRecorder]:
    global _ACTIVE_RECORDER
    prev = _ACTIVE_RECORDER
    _ACTIVE_RECORDER = recorder
    return prev


def get_cuda_graph_layer_timer_recorder() -> Optional[CudaGraphLayerTimerRecorder]:
    return _ACTIVE_RECORDER


@contextmanager
def cuda_graph_layer_timer_recorder(recorder: Optional[CudaGraphLayerTimerRecorder]):
    prev = set_cuda_graph_layer_timer_recorder(recorder)
    try:
        yield
    finally:
        set_cuda_graph_layer_timer_recorder(prev)


@contextmanager
def cuda_graph_layer_timing(layer_id: int, phase: str):
    recorder = get_cuda_graph_layer_timer_recorder()
    if recorder is None:
        yield
    else:
        with recorder.time(layer_id, phase):
            yield
