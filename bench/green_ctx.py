"""Green-context SM limit for h200_decode_bw_q36fp8_sm_r001 (user 2026-10-14: "就用GreenContext").
enter(n): create a torch.cuda GreenContext with n SMs on device 0 and make it current for this thread, before any other CUDA
work, so every later stream / kernel / CUDA graph of the process (bench_one_batch runs the model in-process for TP1) is
confined to it. check(): counts the distinct SMs that actually ran work, read from %smid by a Triton kernel launched
(a) on the current stream and (b) from a CUDA graph captured on a side stream, as SGLang does.
"""
import json, os
import torch


def enter(n):
    from torch.cuda.green_contexts import GreenContext
    gc = GreenContext.create(n, 0)
    gc.set_context()
    return gc


def check():
    import triton, triton.language as tl

    @triton.jit
    def smid_kernel(out_ptr, ITERS: tl.constexpr):
        pid = tl.program_id(0)
        acc = tl.zeros([128], dtype=tl.float32) + pid
        for _ in range(ITERS):            # keep each program busy so programs spread over all available SMs
            acc = acc * 1.0000001 + 1.0
        sm = tl.inline_asm_elementwise('mov.u32 $0, %smid;', '=r', [], dtype=tl.int32, is_pure=False, pack=1)
        tl.store(out_ptr + pid, sm + (tl.sum(acc) * 0).to(tl.int32))

    n = 132 * 64
    out = torch.full((n,), -1, dtype=torch.int32, device='cuda')
    smid_kernel[(n,)](out, ITERS=20000)
    torch.cuda.synchronize()
    eager = int(torch.unique(out).numel())
    out.fill_(-1)
    s = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s):
        smid_kernel[(n,)](out, ITERS=20000)          # warm-up / compile outside capture
        torch.cuda.synchronize()
        with torch.cuda.graph(g, stream=s):
            smid_kernel[(n,)](out, ITERS=20000)
    out.fill_(-1); g.replay(); torch.cuda.synchronize()
    graph = int(torch.unique(out).numel())
    props = torch.cuda.get_device_properties(0)
    r = {'requested': os.environ.get('ASH_GC_SMS'), 'eager_sms_used': eager, 'graph_sms_used': graph,
         'props_sms': props.multi_processor_count}
    print('ASH_GC_CHECK ' + json.dumps(r), flush=True)
    return r


def fix_sm_count(n):
    """r002 (user 2026-10-14: "按你建议的下一步测试 ... 用真实SGLang测试"): tell SM-count-sized kernels the green context's real
    SM count. (1) torch.cuda.get_device_properties reports n SMs (SGLang utils, deep_gemm JIT warm-up, Triton grids);
    (2) deep_gemm.set_num_sms(n); (3) FA3 flash_attn_with_kvcache / get_scheduler_metadata get sm_margin = full - n."""
    full = torch.cuda.get_device_properties(0).multi_processor_count
    _orig = torch.cuda.get_device_properties

    class _Props:
        def __init__(self, p): self._p = p
        def __getattr__(self, k): return n if k == 'multi_processor_count' else getattr(self._p, k)
        def __repr__(self): return repr(self._p).replace(f'multi_processor_count={full}', f'multi_processor_count={n}')
    torch.cuda.get_device_properties = lambda *a, **k: _Props(_orig(*a, **k))
    import deep_gemm
    deep_gemm.set_num_sms(n)
    margin = full - n
    import sgl_kernel.flash_attn as fa
    seen = set()

    def wrap(f, name):
        def g(*a, **k):
            k.setdefault('sm_margin', margin)
            if name not in seen:
                seen.add(name); print(f'ASH_GC_FIX first call {name} sm_margin={k["sm_margin"]}', flush=True)
            return f(*a, **k)
        return g
    fa.flash_attn_with_kvcache = wrap(fa.flash_attn_with_kvcache, 'flash_attn_with_kvcache')
    fa.get_scheduler_metadata = wrap(fa.get_scheduler_metadata, 'get_scheduler_metadata')
    import sglang.srt.layers.attention.flashattention_backend as fb   # module-level imported names
    for k in ('flash_attn_with_kvcache', 'get_scheduler_metadata'):
        if hasattr(fb, k):
            setattr(fb, k, getattr(fa, k))
    print('ASH_GC_FIX ' + json.dumps({'sms': n, 'props_now': torch.cuda.get_device_properties(0).multi_processor_count,
                                      'deep_gemm_num_sms': deep_gemm.get_num_sms(), 'fa3_sm_margin': margin}), flush=True)


def maybe_enter_from_env():
    n = os.environ.get('ASH_GC_SMS')
    gc = enter(int(n)) if n else None
    if n and os.environ.get('ASH_GC_FIX') == '1':
        fix_sm_count(int(n))
    if os.environ.get('ASH_GC_CHECK', '1') == '1':
        check()
    return gc
