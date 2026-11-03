"""FA3 decode over remote (NVLink peer) vs local KV as a function of the number of SMs (persistent CTAs) it gets.
Slow-lane shape of Qwen3.6 full attention: B=16, L=32768, 16 Q heads, 2 KV heads, head_dim 256, page_size 1.
The CTA count is limited with sm_margin (as a green-context lane does); num_splits from the lane rule, plus a sweep.
Run with CUDA_VISIBLE_DEVICES=<compute>,<kv host>:  python fa3_remote_sms.py [--sms 2,4,...] [--dtypes bf16,fp8]"""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.join(os.environ.get('ASH_SGLANG', ''), 'python'))
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata
from sglang.srt.layers.attention.flashattention_backend import choose_lane_num_splits, fa3_metadata_num_sm

HQ, HK, D = 16, 2, 256
FULL = torch.cuda.get_device_properties(0).multi_processor_count


def on_peer(shape, dtype):
    """tensor allocated on cuda:1, aliased into cuda:0 (same trick as dual_lane_bench_r005.py)."""
    t = torch.randn(shape, device='cuda:1', dtype=torch.bfloat16).to(dtype)
    st = torch._C._construct_storage_from_data_pointer(t.data_ptr(), torch.device('cuda', 0), t.numel() * t.element_size())
    a = torch.empty(0, dtype=dtype, device='cuda:0').set_(st, 0, t.shape, t.stride()); a._owner = t
    return a


def timed(f, iters=10):
    f(); torch.cuda.synchronize(0)
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record()
    for _ in range(iters):
        f()
    e1.record(); torch.cuda.synchronize(0)
    return e0.elapsed_time(e1) / iters * 1e3  # us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sms', default='2,4,6,8,10,12,16,20,32,66,132'); ap.add_argument('--B', type=int, default=16)
    ap.add_argument('--L', type=int, default=32768); ap.add_argument('--dtypes', default='bf16,fp8'); ap.add_argument('--out')
    a = ap.parse_args()
    torch.ones(1, device='cuda:0').to('cuda:1'); torch.ones(1, device='cuda:1').to('cuda:0')  # peer access
    B, L = a.B, a.L; tiles = B * HK
    seq = torch.full((B,), L, device='cuda', dtype=torch.int32); cu = torch.arange(0, B + 1, device='cuda', dtype=torch.int32)
    pt = torch.arange(B * L, device='cuda', dtype=torch.int32).view(B, L)
    res = []
    for dt in a.dtypes.split(','):
        dtype = torch.bfloat16 if dt == 'bf16' else torch.float8_e4m3fn
        q = torch.randn(B, HQ, D, device='cuda', dtype=torch.bfloat16).to(dtype)
        kv_bytes = 2 * B * L * HK * D * torch.tensor([], dtype=dtype).element_size()
        kvs = {'local': [torch.randn(B * L, 1, HK, D, device='cuda:0', dtype=torch.bfloat16).to(dtype) for _ in range(2)],
               'remote': [on_peer((B * L, 1, HK, D), dtype) for _ in range(2)]}
        desc = {}
        if dtype != torch.bfloat16:
            one = torch.ones(B, HK, device='cuda', dtype=torch.float32); desc = dict(q_descale=one, k_descale=one, v_descale=one)
        for n in [int(x) for x in a.sms.split(',')]:
            lane_s = choose_lane_num_splits(tiles, n)
            for s in sorted({lane_s, 1, 2, 4, 8, 16}):
                row = dict(dtype=dt, sms=n, num_splits=s, lane_rule=(s == lane_s))
                for where, (k, v) in kvs.items():
                    meta = get_scheduler_metadata(batch_size=B, max_seqlen_q=1, max_seqlen_k=L, num_heads=HQ, num_heads_k=HK,
                                                  headdim=D, cache_seqlens=seq, qkv_dtype=dtype, cu_seqlens_q=cu, page_size=1,
                                                  causal=True, num_splits=s,
                                                  sm_margin=FULL - (fa3_metadata_num_sm(tiles, s, n) if s > 1 else n))
                    f = lambda: flash_attn_with_kvcache(q, k, v, cache_seqlens=seq, page_table=pt, cu_seqlens_q=cu, max_seqlen_q=1,
                                                        softmax_scale=1 / math.sqrt(D), causal=True, num_splits=s,
                                                        scheduler_metadata=meta, sm_margin=FULL - n, **desc)
                    us = timed(f)
                    row[f'{where}_us'] = round(us, 1); row[f'{where}_GBps'] = round(kv_bytes / us / 1e3, 1)
                res.append(row); print(json.dumps(row), flush=True)
        del kvs; torch.cuda.empty_cache()
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
