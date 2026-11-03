"""Microbench (2 x NVLink GPUs; run with CUDA_VISIBLE_DEVICES=<compute>,<kv host>):
Part A  bandwidth-proportional KV split: the slow lane's KV lives in ONE contiguous virtual range built with CUDA VMM, each 2 MB
        chunk backed by local (compute GPU) or remote (KV host) HBM. One unmodified FA3 decode call on a 20-SM green context reads
        it. Sweep the local fraction f, two layouts, with/without a local-HBM hog on the other 112 SMs (stands in for the fast
        lane). Model: t = max(f*B/Bl, (1-f)*B/Br) (both streams concurrent) vs t = f*B/Bl + (1-f)*B/Br (serialized).
Part B  KV transfer schemes for 10 full-attention layers per step, "other compute" between layers emulated with a spin kernel
        (uses no memory bandwidth): all-local (lower bound), in-place remote reads, layer-wise prefetch into a double-buffered
        local staging area (copy engine), and one-shot migration of a request's KV (break-even vs in-place reads)."""
import argparse, json, math, os, sys
sys.path.insert(0, os.path.join(os.environ['ASH_SGLANG'], 'python'))
import torch
from cuda.bindings import driver as cu
from sgl_kernel.flash_attn import flash_attn_with_kvcache, get_scheduler_metadata
from sglang.srt.layers.attention.flashattention_backend import choose_lane_num_splits, fa3_metadata_num_sm
from sglang.srt.multiplex.greenctx_driver import create_split_streams

HQ, HK, D = 16, 2, 256
FULL = torch.cuda.get_device_properties(0).multi_processor_count


def ok(r):
    e = r[0] if isinstance(r, tuple) else r
    if e != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(str(r))
    return r[1] if isinstance(r, tuple) and len(r) > 1 else None


def prop(dev):
    p = cu.CUmemAllocationProp()
    p.type = cu.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    p.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    p.location.id = dev
    return p


class MixedPool:
    """One contiguous VA range; chunk i physically on device place[i] (0 = compute GPU, 1 = KV host); both devices may access."""

    def __init__(self, nbytes, place_fn):
        self.gran = ok(cu.cuMemGetAllocationGranularity(prop(0), cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED))
        n = -(-nbytes // self.gran); self.size = n * self.gran
        self.va = ok(cu.cuMemAddressReserve(self.size, 0, 0, 0))
        self.place = [place_fn(i) for i in range(n)]; self.handles = []
        for i, dev in enumerate(self.place):
            h = ok(cu.cuMemCreate(self.gran, prop(dev), 0)); self.handles.append(h)
            ok(cu.cuMemMap(int(self.va) + i * self.gran, self.gran, 0, h, 0))
        descs = []
        for dev in (0, 1):
            d = cu.CUmemAccessDesc(); d.location.type = cu.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE; d.location.id = dev
            d.flags = cu.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE; descs.append(d)
        ok(cu.cuMemSetAccess(self.va, self.size, descs, len(descs)))

    def tensor(self, shape, dtype):
        st = torch._C._construct_storage_from_data_pointer(int(self.va), torch.device('cuda', 0), self.size)
        return torch.empty(0, dtype=dtype, device='cuda:0').set_(st, 0, shape, torch.empty(shape, device='meta').stride())

    def local_fraction(self):
        return sum(1 for p in self.place if p == 0) / len(self.place)

    def free(self):
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        for i, h in enumerate(self.handles):
            ok(cu.cuMemUnmap(int(self.va) + i * self.gran, self.gran)); ok(cu.cuMemRelease(h))
        ok(cu.cuMemAddressFree(self.va, self.size))


def on_peer(shape, dtype):
    t = torch.empty(shape, device='cuda:1', dtype=dtype)
    st = torch._C._construct_storage_from_data_pointer(t.data_ptr(), torch.device('cuda', 0), t.numel() * t.element_size())
    a = torch.empty(0, dtype=dtype, device='cuda:0').set_(st, 0, t.shape, t.stride()); a._owner = t
    return a


def timed(fn, stream, iters):
    with torch.cuda.stream(stream):
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True); e0.record(stream)
        for _ in range(iters):
            fn()
        e1.record(stream)
    e1.synchronize(); return e0.elapsed_time(e1) / iters  # ms


class Hog:
    """Local-HBM traffic on the fast-lane SMs: back-to-back 1 GiB copies."""

    def __init__(self, stream):
        self.s = stream; n = 1 << 29
        self.a = torch.empty(n, dtype=torch.bfloat16, device='cuda:0').normal_(); self.b = torch.empty_like(self.a)
        self.ms = timed(lambda: self.b.copy_(self.a), stream, 20)
        self.alone_GBps = 2 * self.a.numel() * 2 / self.ms / 1e6

    def start(self, window_ms):
        n = max(4, int(window_ms / self.ms) + 2)
        with torch.cuda.stream(self.s):
            self.e0, self.e1 = torch.cuda.Event(True), torch.cuda.Event(True); self.e0.record(self.s)
            for _ in range(n):
                self.b.copy_(self.a)
            self.e1.record(self.s); self.n = n

    def stop_GBps(self):
        self.e1.synchronize(); return 2 * self.a.numel() * 2 * self.n / self.e0.elapsed_time(self.e1) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--B', type=int, default=16); ap.add_argument('--L', type=int, default=32768); ap.add_argument('--slow_sms', type=int, default=20)
    ap.add_argument('--fracs', default='0,0.25,0.5,0.625,0.75,0.875,1'); ap.add_argument('--other_ms', type=float, default=2.75)
    ap.add_argument('--out'); ap.add_argument('--parts', default='A,B')
    a = ap.parse_args()
    torch.ones(1, device='cuda:0').to('cuda:1'); torch.ones(1, device='cuda:1').to('cuda:0')
    B, L, NS = a.B, a.L, a.slow_sms; tiles = B * HK; dt = torch.bfloat16
    (fast, slow), split = create_split_streams(FULL - NS, 0)
    assert split[1] == NS, split
    print(json.dumps(dict(split=split)), flush=True)
    hog = Hog(fast); print(json.dumps(dict(hog_alone_GBps=round(hog.alone_GBps))), flush=True)
    seq = torch.full((B,), L, device='cuda', dtype=torch.int32); cu_q = torch.arange(0, B + 1, device='cuda', dtype=torch.int32)
    pt = torch.arange(B * L, device='cuda', dtype=torch.int32).view(B, L)
    q = torch.randn(B, HQ, D, device='cuda', dtype=dt)
    shape = (B * L, 1, HK, D); buf_bytes = B * L * HK * D * 2; kv_bytes = 2 * buf_bytes
    meta_cache = {}

    def fa3(k, v, s):
        if s not in meta_cache:
            meta_cache[s] = get_scheduler_metadata(batch_size=B, max_seqlen_q=1, max_seqlen_k=L, num_heads=HQ, num_heads_k=HK, headdim=D,
                                                   cache_seqlens=seq, qkv_dtype=dt, cu_seqlens_q=cu_q, page_size=1, causal=True, num_splits=s,
                                                   sm_margin=FULL - (fa3_metadata_num_sm(tiles, s, NS) if s > 1 else NS))
        return flash_attn_with_kvcache(q, k, v, cache_seqlens=seq, page_table=pt, cu_seqlens_q=cu_q, max_seqlen_q=1,
                                       softmax_scale=1 / math.sqrt(D), causal=True, num_splits=s, scheduler_metadata=meta_cache[s],
                                       sm_margin=FULL - NS)

    lane_s = choose_lane_num_splits(tiles, NS)
    k_ref = torch.randn(shape, device='cuda:0', dtype=dt); v_ref = torch.randn(shape, device='cuda:0', dtype=dt)
    res = dict(B=B, L=L, slow_sms=NS, kv_GB=kv_bytes / 1e9, lane_splits=lane_s, hog_alone_GBps=hog.alone_GBps, partA=[], partB=[])

    def measure(k, v, s, with_hog, iters=8):
        f = lambda: fa3(k, v, s)
        t0 = timed(f, slow, 2)
        if not with_hog:
            return timed(f, slow, iters), None
        hog.start(t0 * iters * 1.05); t = timed(f, slow, iters); return t, hog.stop_GBps()

    if 'A' in a.parts:
        with torch.cuda.stream(slow):
            ref = {s: fa3(k_ref, v_ref, s).clone() for s in sorted({lane_s, 8, 16})}
        torch.cuda.synchronize()
        for frac in [float(x) for x in a.fracs.split(',')]:
            for layout in ['interleaved', 'blocked']:
                gran = ok(cu.cuMemGetAllocationGranularity(prop(0), cu.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED))
                cpr = (L * HK * D * 2) // gran  # chunks per request (per K or V buffer)
                if layout == 'interleaved':
                    place = lambda i, f=frac: 0 if math.floor((i + 1) * f) - math.floor(i * f) >= 1 else 1
                else:
                    place = lambda i, f=frac: 0 if (i % cpr) < round(f * cpr) else 1
                pk, pv = MixedPool(buf_bytes, place), MixedPool(buf_bytes, place)
                k, v = pk.tensor(shape, dt), pv.tensor(shape, dt); k.copy_(k_ref); v.copy_(v_ref); torch.cuda.synchronize()
                fl = pk.local_fraction()
                for s in sorted({lane_s, 8, 16}):
                    with torch.cuda.stream(slow):
                        o = fa3(k, v, s)
                    torch.cuda.synchronize(); diff = float((o.float() - ref[s].float()).abs().max())
                    for h in (False, True):
                        ms, hog_bw = measure(k, v, s, h)
                        row = dict(local_frac=round(fl, 3), layout=layout, splits=s, hog=h, ms=round(ms, 3),
                                   GBps=round(kv_bytes / ms / 1e6, 1), hog_GBps=None if hog_bw is None else round(hog_bw), max_diff=diff)
                        res['partA'].append(row); print(json.dumps(row), flush=True)
                del k, v; pk.free(); pv.free()

    if 'B' in a.parts:
        NL = 10
        sleep_cycles = 1_000_000
        cal = timed(lambda: torch.cuda._sleep(sleep_cycles), slow, 5); sleep_cycles = int(sleep_cycles * a.other_ms / cal)
        other = lambda: torch.cuda._sleep(sleep_cycles)
        loc = [(torch.randn(shape, device='cuda:0', dtype=dt), torch.randn(shape, device='cuda:0', dtype=dt)) for _ in range(2)]
        rem = [(on_peer(shape, dt), on_peer(shape, dt)) for _ in range(2)]
        for kk, vv in rem:
            kk.copy_(k_ref); vv.copy_(v_ref)
        stage = [(torch.empty(shape, device='cuda:0', dtype=dt), torch.empty(shape, device='cuda:0', dtype=dt)) for _ in range(2)]
        cstream = torch.cuda.Stream(device=0)
        s = lane_s

        def step_local():
            for l in range(NL):
                fa3(*loc[l % 2], s); other()

        def step_inplace():
            for l in range(NL):
                fa3(*rem[l % 2], s); other()

        def step_staged():
            copied = [torch.cuda.Event() for _ in range(NL)]; done = [torch.cuda.Event() for _ in range(NL)]
            cstream.wait_stream(slow)
            with torch.cuda.stream(cstream):
                stage[0][0].copy_(rem[0][0]._owner, non_blocking=True); stage[0][1].copy_(rem[0][1]._owner, non_blocking=True)
                copied[0].record(cstream)
            for l in range(NL):
                if l + 1 < NL:
                    with torch.cuda.stream(cstream):
                        if l >= 1:
                            cstream.wait_event(done[l - 1])
                        sk, sv = stage[(l + 1) % 2]; rk, rv = rem[(l + 1) % 2]
                        sk.copy_(rk._owner, non_blocking=True); sv.copy_(rv._owner, non_blocking=True); copied[l + 1].record(cstream)
                slow.wait_event(copied[l]); fa3(*stage[l % 2], s); done[l].record(slow); other()

        for name, fn in [('all_local', step_local), ('inplace_remote', step_inplace), ('staged_prefetch', step_staged)]:
            for h in (False, True):
                t0 = timed(fn, slow, 1)
                if h:
                    hog.start(t0 * 4 * 1.05)
                ms = timed(fn, slow, 4); hb = hog.stop_GBps() if h else None
                row = dict(scheme=name, hog=h, step_ms=round(ms, 2), other_ms_per_layer=a.other_ms, layers=NL,
                           extra_local_GB=round(2 * kv_bytes / 1e9, 2) if name == 'staged_prefetch' else 0.0,
                           hog_GBps=None if hb is None else round(hb))
                res['partB'].append(row); print(json.dumps(row), flush=True)
        # one-shot migration of ONE request's KV (all NL layers, K and V) remote -> local, copy engine
        per_req = L * HK * D * 2
        src = rem[0][0]._owner.view(-1)[: per_req // 2]; dst = torch.empty_like(src, device='cuda:0')
        mig1 = timed(lambda: [dst.copy_(src, non_blocking=True) for _ in range(2 * NL)], cstream, 5)
        row = dict(scheme='migrate_one_request', MB=round(2 * NL * per_req / 1e6), ms=round(mig1, 3),
                   GBps=round(2 * NL * per_req / mig1 / 1e6, 1))
        res['partB'].append(row); print(json.dumps(row), flush=True)
    if a.out:
        json.dump(res, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
