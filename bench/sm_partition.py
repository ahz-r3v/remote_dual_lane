"""Disjoint green-context SM partitions for two processes on one GPU (greenctx_dual_r001; user 2026-10-15: "GPU1上用GreenContext
划分96/32个SM分别作为fast batch和slow batch"). torch.cuda.GreenContext.create(n) always takes the first n SMs, so two processes
using it would overlap. Here both processes run the same deterministic split through the CUDA driver API (ctypes):
  device split once into groups of N_SLOW SMs; fast = first N_FAST/N_SLOW groups combined, slow = the next group.
enter(role) creates the primary context first (as torch's GreenContext does), then pushes the green context so every later
runtime call / kernel / CUDA graph of this thread runs on that partition. ensure() re-pushes it if a device switch changed the
current context. sm_ids() returns the set of SM ids (%smid) a long Triton kernel actually ran on, eager and from a CUDA graph.
"""
import ctypes, json, os

cu = ctypes.CDLL('libcuda.so.1')


class Res(ctypes.Structure):          # CUdevResource (RESOURCE_ABI_VERSION 1): type, 92 B padding, union of 48 B (sm.smCount first)
    _fields_ = [('type', ctypes.c_int), ('pad', ctypes.c_ubyte * 92), ('smCount', ctypes.c_uint), ('rest', ctypes.c_ubyte * 44)]


def _chk(r, what):
    if r != 0:
        raise RuntimeError(f'{what} -> CUresult {r}')


def make(role, n_fast=96, n_slow=32, dev=0):
    _chk(cu.cuInit(0), 'cuInit')
    d = ctypes.c_int(); _chk(cu.cuDeviceGet(ctypes.byref(d), dev), 'cuDeviceGet')
    full = Res(); _chk(cu.cuDeviceGetDevResource(d, ctypes.byref(full), 1), 'cuDeviceGetDevResource')
    # one split of the whole device into groups of n_slow SMs (re-splitting the remainder is refused, CUresult 915):
    # fast = the first n_fast / n_slow groups combined, slow = the next group -> disjoint by construction
    G = n_fast // n_slow + 1
    grp, rem, ng = (Res * G)(), Res(), ctypes.c_uint(G)
    _chk(cu.cuDevSmResourceSplitByCount(grp, ctypes.byref(ng), ctypes.byref(full), ctypes.byref(rem), 0, n_slow), 'split')
    if ng.value < G:
        raise RuntimeError(f'only {ng.value} groups of {n_slow} SMs')
    if role == 'fast':
        parts, nres = grp, G - 1
    else:
        parts, nres = ctypes.pointer(grp[G - 1]), 1
    n_sms = sum(grp[i].smCount for i in range(G - 1)) if role == 'fast' else grp[G - 1].smCount
    desc = ctypes.c_void_p(); _chk(cu.cuDevResourceGenerateDesc(ctypes.byref(desc), parts, nres), 'generateDesc')
    g = ctypes.c_void_p(); _chk(cu.cuGreenCtxCreate(ctypes.byref(g), desc, d, 1), 'cuGreenCtxCreate')
    ctx = ctypes.c_void_p(); _chk(cu.cuCtxFromGreenCtx(ctypes.byref(ctx), g), 'cuCtxFromGreenCtx')
    return ctx, int(n_sms), int(full.smCount), int(rem.smCount)


def _unused():
    desc = ctypes.c_void_p(); _chk(cu.cuDevResourceGenerateDesc(ctypes.byref(desc), ctypes.byref(part), 1), 'generateDesc')
    g = ctypes.c_void_p(); _chk(cu.cuGreenCtxCreate(ctypes.byref(g), desc, d, 1), 'cuGreenCtxCreate')
    ctx = ctypes.c_void_p(); _chk(cu.cuCtxFromGreenCtx(ctypes.byref(ctx), g), 'cuCtxFromGreenCtx')
    return ctx, int(part.smCount), int(full.smCount), int(rem.smCount)


_CTX = None


def enter(role, n_fast=96, n_slow=32):
    global _CTX
    import torch
    torch.zeros(1, device='cuda:0'); torch.cuda.synchronize()       # primary context exists before the green one is pushed
    ctx, n, full, rem = make(role, n_fast, n_slow)
    _chk(cu.cuCtxPushCurrent_v2(ctx), 'cuCtxPushCurrent'); _CTX = ctx
    # torch creates its per-device stream pool on first use; create it now, while the green context is current, so every
    # pool stream (CUDA-graph capture / replay streams included) belongs to this partition even if a later device switch
    # (peer copies, remote KV allocation) briefly makes the primary context current again.
    _keep = [torch.cuda.Stream(device=0) for _ in range(64)]
    del _keep
    print('ASH_GC2 ' + json.dumps({'role': role, 'sms': n, 'device_sms': full, 'unused_sms': rem}), flush=True)
    return n


def ensure():
    cur = ctypes.c_void_p(); cu.cuCtxGetCurrent(ctypes.byref(cur))
    if _CTX is not None and cur.value != _CTX.value:
        _chk(cu.cuCtxPushCurrent_v2(_CTX), 'cuCtxPushCurrent (re)'); return True
    return False


def sm_ids():
    import torch, triton, triton.language as tl

    @triton.jit
    def k(out_ptr, ITERS: tl.constexpr):
        pid = tl.program_id(0)
        acc = tl.zeros([128], dtype=tl.float32) + pid
        for _ in range(ITERS):
            acc = acc * 1.0000001 + 1.0
        sm = tl.inline_asm_elementwise('mov.u32 $0, %smid;', '=r', [], dtype=tl.int32, is_pure=False, pack=1)
        tl.store(out_ptr + pid, sm + (tl.sum(acc) * 0).to(tl.int32))
    n = 132 * 64
    out = torch.full((n,), -1, dtype=torch.int32, device='cuda')
    k[(n,)](out, ITERS=20000); torch.cuda.synchronize()
    eager = sorted(set(out.tolist()))
    s = torch.cuda.Stream(); g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        k[(n,)](out, ITERS=20000); torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            k[(n,)](out, ITERS=20000)
    out.fill_(-1); g.replay(); torch.cuda.synchronize()
    return eager, sorted(set(out.tolist()))


if __name__ == '__main__':                 # python sm_partition.py fast|slow  -> prints the SM id sets
    import sys
    role = sys.argv[1]; enter(role)
    e, gr = sm_ids()
    print('ASH_GC2_SMIDS ' + json.dumps({'role': role, 'eager_n': len(e), 'graph_n': len(gr), 'eager': e, 'graph': gr}), flush=True)
