"""GPU check of the FA3 lane split (ASH r001): for decode shapes of Qwen3.6 full attention (16 Q heads, 2 KV heads, head_dim 256,
bf16 KV, page_size 1) compare, per lane SM count, the lane configuration used by FlashAttentionBackend (num_splits from
choose_lane_num_splits, metadata from fa3_metadata_num_sm, kernel sm_margin = full - n) with the unsplit reference, and time it
against the stock configuration (kernel margin only). Run on one H200:
  python test/ash/test_fa3_lane_split_gpu.py [--out f.json] [--sms 96,112,128] [--batches 1,8,32] [--ctx 4096,51200]"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'python'))
import torch  # noqa: E402
from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata  # noqa: E402

from sglang.srt.layers.attention.flashattention_backend import choose_lane_num_splits, fa3_metadata_num_sm  # noqa: E402

HQ, HK, D = 16, 2, 256


def run_case(B, L, kern_margin, meta_margin, ns, q, k, v, iters=20):
    pt = torch.arange(B * L, device='cuda', dtype=torch.int32).view(B, L)
    seq = torch.full((B,), L, device='cuda', dtype=torch.int32); cu = torch.arange(0, B + 1, device='cuda', dtype=torch.int32)
    meta = None
    if meta_margin is not None:
        meta = get_scheduler_metadata(batch_size=B, max_seqlen_q=1, max_seqlen_k=L, num_heads=HQ, num_heads_k=HK, headdim=D,
                                      cache_seqlens=seq, qkv_dtype=torch.bfloat16, cu_seqlens_q=cu, page_size=1, causal=True,
                                      num_splits=ns, sm_margin=meta_margin)
    f = lambda: flash_attn_with_kvcache(q, k, v, cache_seqlens=seq, page_table=pt, cu_seqlens_q=cu, max_seqlen_q=1,
                                        softmax_scale=1 / math.sqrt(D), causal=True, num_splits=ns, scheduler_metadata=meta,
                                        sm_margin=kern_margin)
    out = f(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(iters):
        f()
    e1.record(); torch.cuda.synchronize()
    return out.float(), e0.elapsed_time(e1) / iters * 1e3


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=None)
    ap.add_argument('--sms', default='96,112,128'); ap.add_argument('--batches', default='1,8,32'); ap.add_argument('--ctx', default='4096,51200')
    a = ap.parse_args()
    torch.manual_seed(0)
    full = torch.cuda.get_device_properties(0).multi_processor_count
    res, ok = [], True
    for B in [int(x) for x in a.batches.split(',')]:
        for L in [int(x) for x in a.ctx.split(',')]:
            q = torch.randn(B, HQ, D, device='cuda', dtype=torch.bfloat16)
            k = torch.randn(B * L, 1, HK, D, device='cuda', dtype=torch.bfloat16)
            v = torch.randn(B * L, 1, HK, D, device='cuda', dtype=torch.bfloat16)
            ref, t_ref = run_case(B, L, 0, None, 1, q, k, v)
            for n in [int(x) for x in a.sms.split(',')]:
                tiles = B * HK; s = choose_lane_num_splits(tiles, n); virt = fa3_metadata_num_sm(tiles, s, n)
                o_fix, t_fix = run_case(B, L, full - n, full - virt, s, q, k, v)
                o_stk, t_stk = run_case(B, L, full - n, 0, 0, q, k, v)
                d_fix = float((o_fix - ref).abs().max()); d_stk = float((o_stk - ref).abs().max())
                good = bool(torch.isfinite(o_fix).all()) and d_fix <= max(2e-3, 2 * d_stk + 1e-4)
                ok &= good
                r = dict(B=B, L=L, sms=n, num_splits=s, meta_num_sm=virt, lane_us=round(t_fix, 1), stock_us=round(t_stk, 1),
                         nosplit_full_us=round(t_ref, 1), max_abs_vs_nosplit=d_fix, stock_max_abs_vs_nosplit=d_stk, ok=good)
                res.append(r); print(json.dumps(r), flush=True)
            del k, v; torch.cuda.empty_cache()
    print('ALL_OK' if ok else 'FAILED', flush=True)
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
