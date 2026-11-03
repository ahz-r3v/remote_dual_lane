"""One atomic unit: decode attention kernel <kernel> on a green-context partition of <sms> SMs, local and remote (NVLink peer) KV.
Slow-lane shape of Qwen3.6 full attention: B=16 x L=32768, 16 Q heads / 2 KV heads, head_dim 256, BF16. The same KV bytes are viewed
as page_size 1 (FA3, as in SGLang now) or page_size 16 (FlashInfer). Output is checked against FA3 on local KV.
Run with CUDA_VISIBLE_DEVICES=<compute>,<kv host>:  python attn_kernel_cmp.py <kernel> <sms>
kernels: fa3 | fi_decode_cuda | fi_decode_tc | fi_prefill_fa2 | fi_prefill_fa3     sms: 8 | 16 | 24 (whole 8-SM groups) | 20 (rest of 112)"""
import json, math, os, sys
sys.path.insert(0, os.path.join(os.environ['ASH_SGLANG'], 'python'))
import torch

KERNEL, SMS = sys.argv[1], int(sys.argv[2])
PS = int(sys.argv[3]) if len(sys.argv) > 3 else 16  # FlashInfer page size (SGLang default is 1)
B, L, HQ, HK, D = 16, 32768, 16, 2, 256
FULL = torch.cuda.get_device_properties(0).multi_processor_count
dt = torch.bfloat16


def lane_stream(n):
    if n % 8 == 0:
        from sgl_kernel import spatial
        a, _ = spatial.create_greenctx_stream_by_value(n, FULL // 8 * 8 - n, 0)
        return a
    from sglang.srt.multiplex.greenctx_driver import create_split_streams
    (_, s), (_, got) = create_split_streams(FULL - n, 0); assert got == n, got
    return s


def on_peer(shape):
    t = torch.randn(shape, device='cuda:1', dtype=dt)
    st = torch._C._construct_storage_from_data_pointer(t.data_ptr(), torch.device('cuda', 0), t.numel() * t.element_size())
    x = torch.empty(0, dtype=dt, device='cuda:0').set_(st, 0, t.shape, t.stride()); x._owner = t
    return x


def timed(fn, s, iters=10):
    with torch.cuda.stream(s):
        fn(); s.synchronize()
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record(s)
        for _ in range(iters):
            fn()
        e1.record(s)
    e1.synchronize(); return e0.elapsed_time(e1) / iters


def make_runner(k, v, s):
    """returns fn() -> output [B, HQ, D] for this kernel on KV tensors k, v of shape [B*L, 1, HK, D] (contiguous, per request)."""
    q = make_runner.q
    if KERNEL == 'fa3':
        from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata
        from sglang.srt.layers.attention.flashattention_backend import choose_lane_num_splits, fa3_metadata_num_sm
        tiles = B * HK; ns = choose_lane_num_splits(tiles, SMS)
        seq = torch.full((B,), L, device='cuda', dtype=torch.int32); cuq = torch.arange(0, B + 1, device='cuda', dtype=torch.int32)
        pt = torch.arange(B * L, device='cuda', dtype=torch.int32).view(B, L)
        meta = get_scheduler_metadata(batch_size=B, max_seqlen_q=1, max_seqlen_k=L, num_heads=HQ, num_heads_k=HK, headdim=D,
                                      cache_seqlens=seq, qkv_dtype=dt, cu_seqlens_q=cuq, page_size=1, causal=True, num_splits=ns,
                                      sm_margin=FULL - (fa3_metadata_num_sm(tiles, ns, SMS) if ns > 1 else SMS))
        return lambda: flash_attn_with_kvcache(q, k, v, cache_seqlens=seq, page_table=pt, cu_seqlens_q=cuq, max_seqlen_q=1,
                                               softmax_scale=1 / math.sqrt(D), causal=True, num_splits=ns, scheduler_metadata=meta,
                                               sm_margin=FULL - SMS)
    import flashinfer
    kp, vp = k.view(B * L // PS, PS, HK, D), v.view(B * L // PS, PS, HK, D)
    indptr = torch.arange(0, B + 1, device='cuda', dtype=torch.int32) * (L // PS)
    indices = torch.arange(B * L // PS, device='cuda', dtype=torch.int32)
    last = torch.full((B,), PS, device='cuda', dtype=torch.int32)
    ws = torch.empty(512 << 20, dtype=torch.uint8, device='cuda')
    q3 = q.view(B, HQ, D)
    with torch.cuda.stream(s):
        if KERNEL.startswith('fi_decode'):
            w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, 'NHD', use_tensor_cores=(KERNEL == 'fi_decode_tc'))
            w.plan(indptr, indices, last, HQ, HK, D, PS, q_data_type=dt, kv_data_type=dt, sm_scale=1 / math.sqrt(D))
        else:
            w = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, 'NHD', backend=KERNEL.split('_')[-1])
            qo = torch.arange(0, B + 1, device='cuda', dtype=torch.int32)
            w.plan(qo, indptr, indices, last, HQ, HK, D, PS, causal=False, q_data_type=dt, kv_data_type=dt, sm_scale=1 / math.sqrt(D))
    return lambda: w.run(q3, (kp, vp))


def main():
    torch.ones(1, device='cuda:0').to('cuda:1'); torch.ones(1, device='cuda:1').to('cuda:0')
    s = lane_stream(SMS); torch.manual_seed(0)
    make_runner.q = torch.randn(B, HQ, D, device='cuda:0', dtype=dt)
    shape = (B * L, 1, HK, D); kv_bytes = 2 * B * L * HK * D * 2
    k_loc, v_loc = torch.randn(shape, device='cuda:0', dtype=dt), torch.randn(shape, device='cuda:0', dtype=dt)
    k_rem, v_rem = on_peer(shape), on_peer(shape); k_rem.copy_(k_loc); v_rem.copy_(v_loc)
    # reference: FA3 on all 132 SMs, local KV, default stream
    from sgl_kernel.flash_attn import flash_attn_with_kvcache
    seq = torch.full((B,), L, device='cuda', dtype=torch.int32); cuq = torch.arange(0, B + 1, device='cuda', dtype=torch.int32)
    ref = flash_attn_with_kvcache(make_runner.q, k_loc, v_loc, cache_seqlens=seq, page_table=torch.arange(B * L, device='cuda', dtype=torch.int32).view(B, L),
                                  cu_seqlens_q=cuq, max_seqlen_q=1, softmax_scale=1 / math.sqrt(D), causal=True).float()
    for where, (k, v) in [('local', (k_loc, v_loc)), ('remote', (k_rem, v_rem))]:
        row = dict(kernel=KERNEL, sms=SMS, kv=where, page_size=1 if KERNEL == 'fa3' else PS)
        try:
            fn = make_runner(k, v, s)
            with torch.cuda.stream(s):
                out = fn()
            s.synchronize()
            row['max_diff_vs_fa3'] = float((out.float().view_as(ref) - ref).abs().max())
            ms = timed(fn, s)
            row.update(ms=round(ms, 3), GBps=round(kv_bytes / ms / 1e6, 1), GBps_per_sm=round(kv_bytes / ms / 1e6 / SMS, 1))
        except Exception as e:
            row['error'] = f'{type(e).__name__}: {str(e)[:300]}'
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
