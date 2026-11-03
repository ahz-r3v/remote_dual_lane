"""Minimal micro-batch ring vs sequential slow lane (BF16, 2 x NVLink GPUs; run with CUDA_VISIBLE_DEVICES=<compute>,<kv host>).
Slow lane = 20 SMs (one 8-SM group + the 12-SM remainder of a 15 x 8 split of the device); ring = attention on one part, the rest of
the step on the other part, two micro-batches of 8 requests alternating. Per step: 10 full-attention layers; each followed by the
"rest of the model" (GDN + MoE + projections), emulated by a weight-reading GEMM (M = tokens) calibrated to 2.75 ms/layer for the
whole batch on 20 SMs (real slow lane: ~27.5 ms/step of non-attention kernels). A micro-batch of 8 reads `--mb_weight_frac` (0.69) of
that weight: dense weights in full + fewer unique experts (MoE ~71% of the non-attention time; 8 tokens hit ~57% of the experts
16 tokens hit). Attention = real FA3 on real KV (16 x 32k, local or remote via NVLink peer)."""
import argparse, ctypes, json, math, os, sys
sys.path.insert(0, os.path.join(os.environ['ASH_SGLANG'], 'python'))
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata
from sglang.srt.layers.attention.flashattention_backend import choose_lane_num_splits, fa3_metadata_num_sm
from sglang.srt.multiplex.greenctx_driver import _DevResource, _driver, _chk, create_split_streams

HQ, HK, D, NL = 16, 2, 256, 10
FULL = torch.cuda.get_device_properties(0).multi_processor_count


def green_streams():
    """15 groups of 8 SMs + remainder; returns streams for (8-SM group, remainder, group+remainder) and their SM counts."""
    torch.zeros(1, device='cuda:0'); cu = _driver()
    dev = ctypes.c_int(); _chk(cu.cuDeviceGet(ctypes.byref(dev), 0), 'cuDeviceGet')
    full = _DevResource(); _chk(cu.cuDeviceGetDevResource(dev, ctypes.byref(full), 1), 'cuDeviceGetDevResource')
    want = 15; grp, rem, ng = (_DevResource * want)(), _DevResource(), ctypes.c_uint(want)
    _chk(cu.cuDevSmResourceSplitByCount(grp, ctypes.byref(ng), ctypes.byref(full), ctypes.byref(rem), 0, 8), 'split')
    last = ng.value - 1

    def stream(resources):
        arr = (_DevResource * len(resources))(*resources)
        desc = ctypes.c_void_p(); _chk(cu.cuDevResourceGenerateDesc(ctypes.byref(desc), arr, len(resources)), 'desc')
        g = ctypes.c_void_p(); _chk(cu.cuGreenCtxCreate(ctypes.byref(g), desc, dev, 1), 'green')
        s = ctypes.c_void_p(); _chk(cu.cuGreenCtxStreamCreate(ctypes.byref(s), g, 1, 0), 'stream')
        return torch.cuda.ExternalStream(stream_ptr=s.value, device=torch.device('cuda:0'))
    a, b = grp[last], rem
    # a group and the remainder cannot be combined into one green context (CUresult 915), so the 20-SM lane for the sequential
    # baselines is the usual 112 + rest split (same SM count, possibly other SMs; nothing else runs on the GPU here)
    (_, s20), (_, n20) = create_split_streams(FULL - int(a.smCount + b.smCount), 0)
    return (stream([a]), stream([b]), s20), (int(a.smCount), int(b.smCount), int(n20)), ng.value


def on_peer(shape, dtype):
    t = torch.randn(shape, device='cuda:1', dtype=dtype)
    st = torch._C._construct_storage_from_data_pointer(t.data_ptr(), torch.device('cuda', 0), t.numel() * t.element_size())
    x = torch.empty(0, dtype=dtype, device='cuda:0').set_(st, 0, t.shape, t.stride()); x._owner = t
    return x


def run_timed(fn, streams, iters):
    """fn enqueues one step on the given streams; time from an event before to the join of all streams."""
    s0 = streams[0]; ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    fn(); torch.cuda.synchronize(0)
    ev0.record(s0)
    for s in streams[1:]:
        s.wait_event(ev0)
    for _ in range(iters):
        fn()
    for s in streams[1:]:
        e = torch.cuda.Event(); e.record(s); s0.wait_event(e)
    ev1.record(s0); ev1.synchronize()
    return ev0.elapsed_time(ev1) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--L', type=int, default=32768); ap.add_argument('--other_ms', type=float, default=2.75)
    ap.add_argument('--mb_weight_frac', type=float, default=0.69); ap.add_argument('--iters', type=int, default=5); ap.add_argument('--out')
    a = ap.parse_args()
    torch.ones(1, device='cuda:0').to('cuda:1'); torch.ones(1, device='cuda:1').to('cuda:0')
    (s8, s12, s20), (n8, n12, n20), ng = green_streams()
    print(json.dumps(dict(groups=ng, sms=dict(part_a=n8, part_b=n12, lane=n20))), flush=True)
    L, dt = a.L, torch.bfloat16
    res = dict(L=L, sms=[n8, n12, n20], rows=[])

    # ---- attention: two micro-batches of 8 requests; KV local or remote (each micro-batch its own KV) ----
    def kv(Bm, remote):
        shape = (Bm * L, 1, HK, D)
        return (on_peer(shape, dt), on_peer(shape, dt)) if remote else \
               (torch.randn(shape, device='cuda:0', dtype=dt), torch.randn(shape, device='cuda:0', dtype=dt))
    KV = {r: [kv(8, r), kv(8, r)] for r in (False, True)}
    meta_cache = {}

    def attn(Bm, k, v, n_sms):
        """FA3 decode for Bm requests with its grid limited to n_sms (run on a stream that owns n_sms SMs)."""
        tiles = Bm * HK; s = choose_lane_num_splits(tiles, n_sms); key = (Bm, n_sms)
        if key not in meta_cache:
            seq = torch.full((Bm,), L, device='cuda', dtype=torch.int32); cuq = torch.arange(0, Bm + 1, device='cuda', dtype=torch.int32)
            pt = torch.arange(Bm * L, device='cuda', dtype=torch.int32).view(Bm, L)
            q = torch.randn(Bm, HQ, D, device='cuda', dtype=dt)
            meta = get_scheduler_metadata(batch_size=Bm, max_seqlen_q=1, max_seqlen_k=L, num_heads=HQ, num_heads_k=HK, headdim=D,
                                          cache_seqlens=seq, qkv_dtype=dt, cu_seqlens_q=cuq, page_size=1, causal=True, num_splits=s,
                                          sm_margin=FULL - (fa3_metadata_num_sm(tiles, s, n_sms) if s > 1 else n_sms))
            meta_cache[key] = (seq, cuq, pt, q, meta, s)
        seq, cuq, pt, q, meta, s = meta_cache[key]
        return flash_attn_with_kvcache(q, k, v, cache_seqlens=seq, page_table=pt, cu_seqlens_q=cuq, max_seqlen_q=1,
                                       softmax_scale=1 / math.sqrt(D), causal=True, num_splits=s, scheduler_metadata=meta,
                                       sm_margin=FULL - n_sms)

    # one full-batch attention = two 8-request calls back to back (same bytes as one 16-request call; checked below)
    # ---- rest of the model: weight-reading GEMM, calibrated on the 20-SM lane ----
    K = 2048; N0 = 1 << 21  # 8.6 GB of weights: enough to reach 2.75 ms on 20 SMs
    W = torch.randn(N0, K, device='cuda:0', dtype=dt)
    x16 = torch.randn(16, K, device='cuda:0', dtype=dt); x8 = x16[:8].contiguous()

    def gemm_ms(n, stream):
        f = lambda: torch.matmul(x16, W[:n].t())
        with torch.cuda.stream(stream):
            return run_timed(f, [stream], 5)
    r256 = lambda n: max(256, min(N0, int(n) // 256 * 256))  # aligned row counts (unaligned N made cuBLAS fall back to a slow kernel)
    t0 = gemm_ms(N0, s20); n_full = r256(N0 * a.other_ms / t0)
    n_full = r256(n_full * a.other_ms / gemm_ms(n_full, s20))
    n_mb = r256(n_full * a.mb_weight_frac)
    cal = dict(other_full_ms_20sm=round(gemm_ms(n_full, s20), 3), weights_GB=round(n_full * K * 2 / 1e9, 2),
               other_mb_ms_on_b=round(gemm_ms(n_mb, s12), 3), other_mb_ms_on_a=round(gemm_ms(n_mb, s8), 3),
               other_full_ms_on_b=round(gemm_ms(n_full, s12), 3))
    print(json.dumps(dict(calibration=cal)), flush=True); res['calibration'] = cal

    def other(x, n):
        return torch.matmul(x, W[:n].t())

    # ---- schedules ----
    def sequential(remote, attn_only=False, other_only=False):
        def step():
            with torch.cuda.stream(s20):
                for l in range(NL):
                    if not other_only:
                        for m in range(2):
                            attn(8, *KV[remote][m], n20)
                    if not attn_only:
                        other(x16, n_full)
        return step

    def ring(remote, att_s, att_n, cmp_s):
        def step():
            ev_a = [[torch.cuda.Event() for _ in range(NL)] for _ in range(2)]
            ev_o = [[torch.cuda.Event() for _ in range(NL)] for _ in range(2)]
            for l in range(NL):
                for m in range(2):
                    with torch.cuda.stream(att_s):
                        if l > 0:
                            att_s.wait_event(ev_o[m][l - 1])
                        attn(8, *KV[remote][m], att_n); ev_a[m][l].record(att_s)
                    with torch.cuda.stream(cmp_s):
                        cmp_s.wait_event(ev_a[m][l]); other(x8, n_mb); ev_o[m][l].record(cmp_s)
        return step

    def add(name, fn, streams, **kw):
        ms = run_timed(fn, streams, a.iters)
        row = dict(name=name, step_ms=round(ms, 2), **kw); res['rows'].append(row); print(json.dumps(row), flush=True)
        return ms

    # check: one 16-request call vs two 8-request calls on 20 SMs (remote) - ring uses 8-request calls
    k16 = (on_peer((16 * L, 1, HK, D), dt), on_peer((16 * L, 1, HK, D), dt))
    with torch.cuda.stream(s20):
        one = run_timed(lambda: attn(16, *k16, n20), [s20], 5)
        two = run_timed(lambda: [attn(8, *KV[True][m], n20) for m in range(2)], [s20], 5)
    print(json.dumps(dict(attn_remote_16req_one_call_ms=round(one, 3), two_8req_calls_ms=round(two, 3))), flush=True)
    del k16

    for remote in (False, True):
        tag = 'remote' if remote else 'local'
        att = add(f'{tag}_sequential_attention_only', sequential(remote, attn_only=True), [s20], kv=tag)
        oth = add(f'{tag}_sequential_other_only', sequential(remote, other_only=True), [s20], kv=tag)
        add(f'{tag}_sequential_20sm', sequential(remote), [s20], kv=tag, attention_ms=round(att, 2), other_ms=round(oth, 2))
        def attn_only_8(remote=remote):
            with torch.cuda.stream(s8):
                for _ in range(NL):
                    for m in range(2):
                        attn(8, *KV[remote][m], n8)
        add(f'{tag}_attention_only_on{n8}sm', attn_only_8, [s8], kv=tag)
        add(f'{tag}_ring_attn{n8}_rest{n12}', ring(remote, s8, n8, s12), [s8, s12], kv=tag)
        add(f'{tag}_ring_attn{n12}_rest{n8}', ring(remote, s12, n12, s8), [s12, s8], kv=tag)
    # ---- (4) layer-wise prefetch into a double-buffered local staging area (copy engine), 20-SM lane computes ----
    rem16 = (on_peer((16 * L, 1, HK, D), dt), on_peer((16 * L, 1, HK, D), dt))
    stage = [(torch.empty((16 * L, 1, HK, D), device='cuda:0', dtype=dt), torch.empty((16 * L, 1, HK, D), device='cuda:0', dtype=dt))
             for _ in range(2)]
    cstream = torch.cuda.Stream(device=0)  # ordinary stream: peer copies run on the copy engines, no SMs

    def staged():
        copied = [torch.cuda.Event() for _ in range(NL)]; done = [torch.cuda.Event() for _ in range(NL)]
        cstream.wait_stream(s20)
        with torch.cuda.stream(cstream):
            for i in range(2):
                stage[0][i].copy_(rem16[i]._owner, non_blocking=True)
            copied[0].record(cstream)
        for l in range(NL):
            if l + 1 < NL:
                with torch.cuda.stream(cstream):
                    if l >= 1:
                        cstream.wait_event(done[l - 1])
                    for i in range(2):
                        stage[(l + 1) % 2][i].copy_(rem16[i]._owner, non_blocking=True)
                    copied[l + 1].record(cstream)
            with torch.cuda.stream(s20):
                s20.wait_event(copied[l]); attn(16, *stage[l % 2], n20); done[l].record(s20); other(x16, n_full)
    add('remote_layer_prefetch_20sm', staged, [s20, cstream], kv='remote', extra_local_GB=round(2 * 2 * 16 * L * HK * D * 2 / 1e9, 2))

    # ---- (5) one-shot migration of the whole slow batch's KV (16 requests x 10 layers) to local, then local decode ----
    dst = (torch.empty_like(stage[0][0]), torch.empty_like(stage[0][1]))
    def migrate():
        with torch.cuda.stream(cstream):
            for _ in range(NL):
                for i in range(2):
                    dst[i].copy_(rem16[i]._owner, non_blocking=True)
    mig = run_timed(migrate, [cstream], 3)
    mig_GB = NL * 2 * 16 * L * HK * D * 2 / 1e9
    loc_seq = [r for r in res['rows'] if r['name'] == 'local_sequential_20sm'][0]['step_ms']
    row = dict(name='remote_migrate_all_then_local', migrate_ms=round(mig, 1), migrate_GB=round(mig_GB, 2),
               GBps=round(mig_GB / mig * 1e3, 1), extra_local_GB=round(mig_GB, 2),
               step_ms_amortized={str(n): round(loc_seq + mig / n, 2) for n in (10, 100, 1500)})
    res['rows'].append(row); print(json.dumps(row), flush=True)
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
