from __future__ import annotations

import asyncio
import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, Generic, List, Optional, TypeVar

import zmq

T = TypeVar("T")


class FanOutCommunicator(Generic[T]):
    """Fan-out request + collect response primitive over zmq.

    One send is fanned out to `fan_out` recipients; the caller awaits until
    all `fan_out` responses are collected. Supports two modes:
    - "queueing": requests are serialized; concurrent callers wait in a FIFO queue.
    - "watching": concurrent callers share a single in-flight request and all
      receive the same result when it completes.

    Only one request is in-flight at any time in either mode.
    """

    def __init__(self, sender: zmq.Socket, fan_out: int, mode="queueing"):
        self._sender = sender
        self._fan_out = fan_out
        self._mode = mode
        self._result_event: Optional[asyncio.Event] = None
        self._result_values: Optional[List[T]] = None
        self._ready_queue: Deque[asyncio.Event] = deque()

        assert mode in ["queueing", "watching"]

    async def queueing_call(self, obj: T):
        ready_event = asyncio.Event()
        if self._result_event is not None or len(self._ready_queue) > 0:
            self._ready_queue.append(ready_event)
            await ready_event.wait()
            assert self._result_event is None
            assert self._result_values is None

        if obj is not None:
            self._sender.send_pyobj(obj)

        self._result_event = asyncio.Event()
        self._result_values = []
        await self._result_event.wait()
        result_values = self._result_values
        self._result_event = self._result_values = None

        if len(self._ready_queue) > 0:
            self._ready_queue.popleft().set()

        return result_values

    async def watching_call(self, obj):
        if self._result_event is None:
            assert self._result_values is None
            self._result_values = []
            self._result_event = asyncio.Event()

            if obj is not None:
                self._sender.send_pyobj(obj)

        # Capture local refs before await -- after event fires, the first
        # awakened coroutine clears shared state; later awaiters use local refs.
        values = self._result_values
        event = self._result_event
        await event.wait()

        result_values = copy.deepcopy(values)
        if self._result_event is event:
            self._result_event = self._result_values = None
        return result_values

    async def __call__(self, obj):
        if self._mode == "queueing":
            return await self.queueing_call(obj)
        else:
            return await self.watching_call(obj)

    def handle_recv(self, recv_obj: T):
        self._result_values.append(recv_obj)
        if len(self._result_values) == self._fan_out:
            self._result_event.set()

    @staticmethod
    def merge_results(results):
        all_success = all([r.success for r in results])
        all_message = [r.message for r in results]
        all_message = " | ".join(all_message)
        return all_success, all_message


@dataclass
class _KeyedPending(Generic[T]):
    event: asyncio.Event = field(default_factory=asyncio.Event)
    values: List[T] = field(default_factory=list)
    abandoned: bool = False


class KeyedFanOutCommunicator(Generic[T]):
    """Correlate multiple in-flight fan-out controls by a stable request key.

    Unlike ``FanOutCommunicator(mode="queueing")``, this primitive permits
    independent long-lived controls to be outstanding concurrently.  It is
    suitable only when both the input and every output carry the same unique
    key.  Libra needs this because a checkpoint command may remain pending for
    hundreds of decode iterations before its requested token boundary.
    """

    def __init__(
        self,
        sender: zmq.Socket,
        fan_out: int,
        key: Callable[[T], str],
    ) -> None:
        if fan_out < 1:
            raise ValueError("fan_out must be positive")
        self._sender = sender
        self._fan_out = fan_out
        self._key = key
        self._pending: Dict[str, _KeyedPending[T]] = {}

    async def __call__(self, obj: T) -> List[T]:
        key = self._key(obj)
        if not key:
            raise ValueError("keyed communicator requires a non-empty key")
        if key in self._pending:
            raise ValueError(f"duplicate in-flight communicator key: {key}")
        state: _KeyedPending[T] = _KeyedPending()
        self._pending[key] = state
        try:
            self._sender.send_pyobj(obj)
        except Exception:
            self._pending.pop(key, None)
            raise
        try:
            await state.event.wait()
        except asyncio.CancelledError:
            # Scheduler responses may still arrive after an HTTP peer leaves.
            # Retain the keyed slot until every fan-out recipient answers so
            # those late responses cannot be confused with another request.
            state.abandoned = True
            if len(state.values) >= self._fan_out:
                self._pending.pop(key, None)
            raise
        self._pending.pop(key, None)
        return state.values

    def handle_recv(self, recv_obj: T) -> None:
        key = self._key(recv_obj)
        state = self._pending.get(key)
        if state is None:
            # A duplicate or late response after full cleanup is harmless and
            # must not take down the tokenizer output loop.
            return
        if len(state.values) >= self._fan_out:
            return
        state.values.append(recv_obj)
        if len(state.values) == self._fan_out:
            if state.abandoned:
                self._pending.pop(key, None)
            else:
                state.event.set()
