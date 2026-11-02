from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import time
import threading
from typing import Callable, Deque, Dict, List, Optional

import torch


class DeviceTimer:
    def __init__(self, reporter: Callable):
        self._intervals: Deque[_TimingInterval] = deque()
        self._reporters: List[Callable] = [reporter]
        self._lock = threading.RLock()

    def add_reporter(self, reporter: Callable):
        self._reporters.append(reporter)

    @contextmanager
    def wrap(self, metadata: Dict, after_end: Optional[Callable] = None):
        interval = _TimingInterval.create()
        with self._lock:
            self._intervals.append(interval)
        try:
            yield
        finally:
            interval.end(metadata=metadata, after_end=after_end)
            self._report()

    def _report(self):
        # Intervals may run on independent CUDA streams and therefore complete
        # out of submission order. A long prefill interval must not head-of-line
        # block already-completed decode intervals behind it.
        with self._lock:
            pending: Deque[_TimingInterval] = deque()
            while self._intervals:
                interval = self._intervals.popleft()
                if interval.ready_event is None or not interval.ready_event.query():
                    pending.append(interval)
                    continue

                elapsed = interval.elapsed_time() / 1000.0
                observability = interval.observability_times()
                for reporter in self._reporters:
                    reporter(t=elapsed, **observability, **interval.metadata)
            self._intervals = pending


class GapTimer(DeviceTimer):
    """Measures GPU idle gaps between consecutive uses of a stream.

    Where DeviceTimer.wrap() measures the duration *inside* a block,
    GapTimer.wrap() measures the time *between* consecutive blocks
    (gap = next_block_start - last_block_end).
    """

    def __init__(self, reporter: Callable):
        super().__init__(reporter)
        self._pending: Optional[_TimingInterval] = None

    @contextmanager
    def wrap(self, metadata: Dict):
        with self._lock:
            if self._pending is not None:
                self._pending.end(metadata=metadata)
                self._intervals.append(self._pending)
                self._pending = None
        self._report()
        try:
            yield
        finally:
            with self._lock:
                self._pending = _TimingInterval.create()

    def cancel(self):
        """Discard a pending gap (e.g. server went idle)."""
        with self._lock:
            self._pending = None


@dataclass
class _TimingInterval:
    start_event: torch.cuda.Event
    end_event: Optional[torch.cuda.Event] = None
    ready_event: Optional[torch.cuda.Event] = None
    metadata: Optional[Dict] = None
    after_end_cpu_seconds: Optional[float] = None

    @staticmethod
    def create():
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        return _TimingInterval(start_event=start_event)

    def end(
        self,
        metadata: Dict,
        after_end: Optional[Callable] = None,
    ):
        end_event = torch.cuda.Event(enable_timing=True)
        end_event.record()

        # Optional observability work is ordered after the measured interval.
        # A second event gates reporting without adding that work to elapsed_time().
        if after_end is not None:
            cpu_start = time.perf_counter()
            after_end()
            after_end_cpu_seconds = time.perf_counter() - cpu_start
            # Timing this event lets us quantify device work (currently the
            # expert-histogram D2H copy) submitted by after_end().
            ready_event = torch.cuda.Event(enable_timing=True)
            ready_event.record()
        else:
            after_end_cpu_seconds = None
            ready_event = end_event

        assert self.end_event is None
        self.end_event = end_event
        self.ready_event = ready_event
        self.metadata = metadata
        self.after_end_cpu_seconds = after_end_cpu_seconds

    def elapsed_time(self) -> float:
        return self.start_event.elapsed_time(self.end_event)

    def observability_times(self) -> Dict[str, float]:
        if self.after_end_cpu_seconds is None:
            return {}
        return {
            "observability_submit_cpu_t": self.after_end_cpu_seconds,
            "observability_device_t": self.end_event.elapsed_time(self.ready_event)
            / 1000.0,
        }
