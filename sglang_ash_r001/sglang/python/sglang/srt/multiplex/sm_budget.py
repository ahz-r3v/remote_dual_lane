"""SM budget of the code that is being captured / launched right now (ASH r001, 2026-10-17).

A green-context decode lane only owns part of the GPU. Kernels whose launch configuration depends on the SM count (persistent grids,
split-KV heuristics, tuned Triton configs, cuBLAS algorithm choice) must be configured for the lane's SMs, not for the whole device.
``sm_budget(n)`` is entered while a lane's CUDA graph is captured (and around eager lane forwards); it
  * makes ``current_sm_budget()`` return n (read by FlashAttention, fused-MoE config lookup, MoE sum-reduce selection),
  * sets deep_gemm's SM count,
  * sets the cuBLAS / cuBLASLt SM-count target via torch's SM carveout.
Outside a budget everything sees the full device.
"""
from __future__ import annotations

import contextlib
import functools
from typing import Optional

import torch

_BUDGET: Optional[int] = None
_WARNED = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        import logging
        logging.getLogger(__name__).warning(msg)


@functools.lru_cache(maxsize=None)
def full_sm_count(device: Optional[int] = None) -> int:
    dev = torch.cuda.current_device() if device is None else device
    return int(torch.cuda.get_device_properties(dev).multi_processor_count)


def current_sm_budget() -> int:
    """SMs available to the kernels being configured now (full device outside any lane budget)."""
    return _BUDGET if _BUDGET is not None else full_sm_count()


def is_partial_budget() -> bool:
    return _BUDGET is not None and _BUDGET < full_sm_count()


def _deep_gemm():
    try:
        import deep_gemm  # noqa: F401
        return deep_gemm if hasattr(deep_gemm, 'set_num_sms') else None
    except Exception:
        return None


@contextlib.contextmanager
def sm_budget(n: Optional[int]):
    """Configure SM-count-dependent kernels for ``n`` SMs inside this block (no-op when n is None or the full device)."""
    global _BUDGET
    full = full_sm_count()
    if not n or n >= full:
        yield
        return
    prev = _BUDGET
    dg = _deep_gemm()
    prev_dg = None
    if dg is not None:
        try:
            prev_dg = dg.get_num_sms()
        except Exception as e:  # deep_gemm not usable here (e.g. no CUDA context): skip it, the rest still applies
            _warn_once(f"sm_budget: deep_gemm SM count not adjusted ({e})")
            dg = None
    prev_carveout = torch._C._get_sm_carveout_experimental() if hasattr(torch._C, '_get_sm_carveout_experimental') else None
    _BUDGET = int(n)
    try:
        if dg is not None:
            dg.set_num_sms(int(n))
        if hasattr(torch._C, '_set_sm_carveout_experimental'):
            torch._C._set_sm_carveout_experimental(full - int(n))
        yield
    finally:
        _BUDGET = prev
        if dg is not None and prev_dg is not None:
            dg.set_num_sms(prev_dg)
        if hasattr(torch._C, '_set_sm_carveout_experimental'):
            torch._C._set_sm_carveout_experimental(prev_carveout)


def _find_backends(b, cls, depth=0, seen=None):
    seen = set() if seen is None else seen
    if b is None or id(b) in seen or depth > 4:
        return []
    seen.add(id(b))
    if isinstance(b, cls):
        return [b]
    out = []
    for v in (vars(b).values() if hasattr(b, '__dict__') else []):
        if isinstance(v, (list, tuple)):
            for x in v:
                out += _find_backends(x, cls, depth + 1, seen)
        elif type(v).__module__.startswith('sglang.srt.layers.attention'):
            out += _find_backends(v, cls, depth + 1, seen)
    return out


def set_lane_sm_count(attn_backend, n: Optional[int]) -> int:
    """Record a decode lane's SM count on every FlashAttention (FA3) backend inside ``attn_backend`` (e.g. the full-attention backend
    of HybridLinearAttnBackend). The FA3 backend uses it at capture AND replay to pick the KV split for these SMs. Returns #backends."""
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
    fabs = _find_backends(attn_backend, FlashAttentionBackend)
    for f in fabs:
        f.lane_sm_count = int(n) if n and n < full_sm_count() else None
    return len(fabs)
