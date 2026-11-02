"""Green-context streams through the CUDA driver API (ASH r001, 2026-10-17).

sgl_kernel.spatial.create_greenctx_stream_by_value needs both partitions to be multiples of the architecture granularity (8 SMs on
Hopper), so e.g. "decode lane = 128 SMs, rest = 4 SMs" cannot be built. Here the device is split ONCE with
cuDevSmResourceSplitByCount(1 group, minCount=n_first): the first lane gets that group, the second lane gets the remainder (any size).
Verified on H200 (jobs 148898/149337: fast 128 SMs, remainder 4 SMs, %smid disjoint).
"""
from __future__ import annotations

import ctypes
from typing import Tuple

_cu = None


class _DevResource(ctypes.Structure):  # CUdevResource, RESOURCE_ABI_VERSION 1: type, 92 B padding, union (sm.smCount first)
    _fields_ = [('type', ctypes.c_int), ('pad', ctypes.c_ubyte * 92), ('smCount', ctypes.c_uint), ('rest', ctypes.c_ubyte * 44)]


def _driver():
    global _cu
    if _cu is None:
        _cu = ctypes.CDLL('libcuda.so.1')
    return _cu


def _chk(r: int, what: str) -> None:
    if r != 0:
        raise RuntimeError(f'{what} -> CUresult {r}')


def _stream_on(res_ptr, dev) -> int:
    cu = _driver()
    desc = ctypes.c_void_p(); _chk(cu.cuDevResourceGenerateDesc(ctypes.byref(desc), res_ptr, 1), 'cuDevResourceGenerateDesc')
    g = ctypes.c_void_p(); _chk(cu.cuGreenCtxCreate(ctypes.byref(g), desc, dev, 1), 'cuGreenCtxCreate')   # CU_GREEN_CTX_DEFAULT_STREAM
    s = ctypes.c_void_p(); _chk(cu.cuGreenCtxStreamCreate(ctypes.byref(s), g, 1, 0), 'cuGreenCtxStreamCreate')  # NON_BLOCKING
    return s.value


def create_split_streams(n_first: int, gpu_id: int) -> Tuple[list, Tuple[int, int]]:
    """Two green-context streams: n_first SMs (rounded by the driver to the arch granularity) and all remaining SMs."""
    import torch
    torch.zeros(1, device=f'cuda:{gpu_id}')           # primary context exists before the green contexts
    cu = _driver()
    dev = ctypes.c_int(); _chk(cu.cuDeviceGet(ctypes.byref(dev), gpu_id), 'cuDeviceGet')
    full = _DevResource(); _chk(cu.cuDeviceGetDevResource(dev, ctypes.byref(full), 1), 'cuDeviceGetDevResource')  # CU_DEV_RESOURCE_TYPE_SM
    grp, rem, ng = (_DevResource * 1)(), _DevResource(), ctypes.c_uint(1)
    _chk(cu.cuDevSmResourceSplitByCount(grp, ctypes.byref(ng), ctypes.byref(full), ctypes.byref(rem), 0, int(n_first)),
         'cuDevSmResourceSplitByCount')
    if rem.smCount == 0:
        raise ValueError(f'no SMs left for the second lane after taking {grp[0].smCount}')
    s0 = _stream_on(ctypes.pointer(grp[0]), dev)
    s1 = _stream_on(ctypes.pointer(rem), dev)
    d = torch.device(f'cuda:{gpu_id}')
    return ([torch.cuda.ExternalStream(stream_ptr=s0, device=d), torch.cuda.ExternalStream(stream_ptr=s1, device=d)],
            (int(grp[0].smCount), int(rem.smCount)))
