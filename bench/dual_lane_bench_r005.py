"""r005 (2026-10-17): runs on the modified SGLang tree sglang_ash/r001 ("直接修改SGLang源码"): every SM-count-dependent kernel
(FA3 split + grid, deep_gemm, cuBLAS, fused-MoE config, MoE sum-reduce) is configured by SGLang itself for each lane
(multiplex/sm_budget.py, entered by ModelRunner.init_decode_lane_cuda_graph_runner); this harness has NO kernel patches any more.
Kept here (experiment-specific): slow-lane pools + remote KV allocation, decode threads, step timestamps, logits dumps
(ASH_DUMP_LOGITS=1: first 4 requests at decode steps 0 and 50, for numerical comparison across SM splits).
128-SM lane: lane config with long_green_ctx_sm_count: -1 (SGLang builds "128 + remaining 4" through the CUDA driver API).
--- r001 header ---
greenctx_dual_r001, single-process version (user 2026-10-15: "我们就要实现单进程的版本，类似PDMux，他们实现的也是单进程").
One SGLang ModelRunner (one copy of the weights) with the tree's decode-lane mux: lane 0 "short" = FAST batch on a 96-SM green
context, lane 1 "long" = SLOW batch on a disjoint 32-SM green context (streams from sgl_kernel.spatial, as PDMux). Each lane has
its own attention backend, CUDA graphs, sampler and stream, so both decode concurrently inside one process (no MPS).
The slow lane gets its own memory pools (req_to_token + GDN state local; full-attention KV allocated on the PEER GPU and aliased
into this device when ASH_REMOTE_KV=1) by re-running the runner's _init_pools with the pools swapped out; the slow lane's
attention backend and graphs are created while the runner points at those pools, so FA3 reads/writes the remote KV over NVLink.
Prefill: eager, sequential, on the full GPU (default stream); slow prefill uses the slow pools (the design puts it on the
KV-holding GPU; here it is a one-time setup cost). Decode: one Python thread per lane, lane graph replay on the lane stream.
Env: ASH_MODE=fast|slow|both, ASH_REMOTE_KV=0|1, ASH_FAST_B/CTX/STEPS, ASH_SLOW_B/CTX/STEPS, ASH_OUT (dir), ASH_PROFILE_STEP
(cudaProfilerStart/Stop around 3 steps of each lane), ASH_CHECK_SMS=1 (%smid sets per lane stream).
CLI: the usual sglang server args (model path, --enable-decode-lane-mux --decode-lane-mux-config-path lanes_96_32.yaml ...).
"""
import argparse, json, logging, os, sys, threading, time
from array import array
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch

MODE = os.environ.get('ASH_MODE', 'both'); REMOTE = os.environ.get('ASH_REMOTE_KV', '0') == '1'
FB, FCTX, FSTEPS = int(os.environ.get('ASH_FAST_B', 16)), int(os.environ.get('ASH_FAST_CTX', 16384)), int(os.environ.get('ASH_FAST_STEPS', 1500))
SB, SCTX, SSTEPS = int(os.environ.get('ASH_SLOW_B', 4)), int(os.environ.get('ASH_SLOW_CTX', 65536)), int(os.environ.get('ASH_SLOW_STEPS', 1500))
OUT = os.environ.get('ASH_OUT', '.'); os.makedirs(OUT, exist_ok=True)
PROF_STEP = int(os.environ.get('ASH_PROFILE_STEP', '0'))
FULL_SMS = torch.cuda.get_device_properties(0).multi_processor_count
log = lambda **k: print('ASH ' + json.dumps(k), flush=True)

# ---------------- remote KV allocation patch (MHATokenToKVPool buffers on cuda:1, aliased into cuda:0) ----------------
from sglang.srt.mem_cache import memory_pool as mp
_orig_create = mp.MHATokenToKVPool._create_buffers
REMOTE_ON = {'v': False}


def _create_buffers_remote(self):
    if not REMOTE_ON['v']:
        return _orig_create(self)
    real_zeros, n = torch.zeros, {'b': 0}

    def zeros_remote(*a, **k):
        dev = torch.device(k.get('device', 'cpu'))
        if dev.type == 'cuda' and (dev.index or 0) == 0:
            t = real_zeros(*a, **dict(k, device='cuda:1'))
            st = torch._C._construct_storage_from_data_pointer(t.data_ptr(), torch.device('cuda', 0), t.numel() * t.element_size())
            al = torch.empty(0, dtype=t.dtype, device='cuda:0').set_(st, 0, t.shape, t.stride()); al._ash_owner = t
            n['b'] += t.numel() * t.element_size(); return al
        return real_zeros(*a, **k)
    torch.zeros = zeros_remote
    try:
        _orig_create(self)
    finally:
        torch.zeros = real_zeros; torch.cuda.synchronize(1)
    log(remote_kv_GB=n['b'] / 1e9, layers=len(self.k_buffer))


mp.MHATokenToKVPool._create_buffers = _create_buffers_remote

import sglang.bench_one_batch as bob
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs, PortArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.multiplex.decode_lane_context import get_decode_lanes

POOL_ATTRS = ('req_to_token_pool', 'token_to_kv_pool', 'token_to_kv_pool_allocator', 'max_total_num_tokens', 'max_running_requests')


class Pools:
    def __init__(self, runner): self.d = {a: getattr(runner, a) for a in POOL_ATTRS}
    def apply(self, runner):
        for a, v in self.d.items(): setattr(runner, a, v)


class swapped:
    """runner temporarily points at another lane's pools (for backend / graph creation and eager prefill)."""
    def __init__(self, runner, pools, attn_backend=None): self.r, self.p, self.b = runner, pools, attn_backend
    def __enter__(self):
        self.saved = Pools(self.r); self.saved_b = self.r.attn_backend; self.p.apply(self.r)
        if self.b is not None: self.r.attn_backend = self.b
    def __exit__(self, *a): self.saved.apply(self.r); self.r.attn_backend = self.saved_b


def make_slow_pools(runner, max_tokens, max_reqs, remote):
    saved = Pools(runner)
    runner.req_to_token_pool = None; runner.token_to_kv_pool_allocator = None; runner.token_to_kv_pool = None
    runner.max_total_num_tokens = max_tokens; runner.max_running_requests = max_reqs
    mamba_saved = runner.server_args.max_mamba_cache_size; runner.server_args.max_mamba_cache_size = max(2 * max_reqs, 8)
    REMOTE_ON['v'] = remote
    try:
        runner._init_pools()
    finally:
        REMOTE_ON['v'] = False; runner.server_args.max_mamba_cache_size = mamba_saved
    slow = Pools(runner); saved.apply(runner)
    log(slow_pools=dict(tokens=max_tokens, reqs=max_reqs, kv_device=str(slow.d['token_to_kv_pool'].full_kv_pool.k_buffer[0].device)
                        if hasattr(slow.d['token_to_kv_pool'], 'full_kv_pool') else 'n/a', remote=remote))
    return slow


def make_reqs(bs, ctx, seed):
    rng = np.random.default_rng(seed); ids = rng.integers(0, 10000, (bs, ctx), dtype=np.int32)
    sp = SamplingParams(temperature=0, max_new_tokens=100000); reqs = []
    for i in range(bs):
        r = Req(rid=f'{seed}_{i}', origin_input_text='', origin_input_ids=array('q', ids[i]), sampling_params=sp)
        r.full_untruncated_fill_ids = r.origin_input_ids; r.fill_len = len(r.origin_input_ids); r.logprob_start_len = -1
        r.set_extend_input_len(r.fill_len - len(r.prefix_indices)); reqs.append(r)
    return reqs


@torch.no_grad()
def extend_group(reqs, runner, pools):
    tc = bob.TreeCacheNamespace(page_size=runner.server_args.page_size, device=runner.device, token_to_kv_pool_allocator=pools.d['token_to_kv_pool_allocator'])
    batch = ScheduleBatch.init_new(reqs=reqs, req_to_token_pool=pools.d['req_to_token_pool'], token_to_kv_pool_allocator=pools.d['token_to_kv_pool_allocator'],
                                   tree_cache=tc, model_config=runner.model_config, enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE)
    batch.prepare_for_extend()
    if batch.input_ids is None and getattr(batch, 'prefill_input_ids_cpu', None) is not None:
        batch.input_ids = batch.prefill_input_ids_cpu.to(batch.device, non_blocking=True); batch.prefill_input_ids_cpu = None
    fbatch = ForwardBatch.init_new(batch, runner)
    out = runner.forward(fbatch).logits_output
    ids = runner.sample(out, fbatch)
    return ids, batch


def prefill(reqs, runner, pools, backend, group_tokens=8192):
    """chunked prefill (ash_bench.py's grouping) with the runner pointed at `pools` + `backend`; returns (next_ids, batch)."""
    groups, cur, n = [], [], 0
    for r in reqs:
        L = len(r.origin_input_ids)
        if cur and n + L > group_tokens: groups.append(cur); cur, n = [], 0
        cur.append(r); n += L
    groups.append(cur)
    with swapped(runner, pools, backend):
        ids, batch = extend_group(groups[0], runner, pools)
        for g in groups[1:]:
            i2, b2 = extend_group(g, runner, pools); batch.merge_batch(b2); ids = torch.cat([ids, i2])
    torch.cuda.synchronize(); return ids, batch


DUMP_LOGITS = os.environ.get('ASH_DUMP_LOGITS', '1') == '1'


def decode_loop(name, lane, runner, batch, ids, steps, out_f, dump=False):
    """one lane's decode loop: lane graph replay on the lane stream; timestamps per step."""
    stream = lane.stream; done = torch.cuda.Event()
    for i in range(steps):
        batch.input_ids = ids.to(torch.int64); batch.prepare_for_decode()
        fbatch = ForwardBatch.init_new(batch, runner)
        if PROF_STEP and i == PROF_STEP: torch.cuda.cudart().cudaProfilerStart()
        stream.wait_stream(torch.cuda.current_stream())        # batch prep ran on the default stream (as disaggregation/decode.py)
        with torch.cuda.stream(stream):
            out = runner.replay_decode_lane_cuda_graph(fbatch, lane)
            ids = runner.sample(out, fbatch, sampler=lane.sampler); done.record(stream)
        done.synchronize()
        if dump and DUMP_LOGITS and i in (0, 50):
            torch.save(out.next_token_logits[:4].float().cpu(), os.path.join(OUT, f'logits_{name}_{i}.pt'))
        if PROF_STEP and i == PROF_STEP + 2: torch.cuda.cudart().cudaProfilerStop()
        out_f.write(f'{time.time():.6f} {i}\n')
    out_f.flush()


def sm_ids_on(stream):
    import triton, triton.language as tl

    @triton.jit
    def k(out_ptr, ITERS: tl.constexpr):
        pid = tl.program_id(0); acc = tl.zeros([128], dtype=tl.float32) + pid
        for _ in range(ITERS): acc = acc * 1.0000001 + 1.0
        sm = tl.inline_asm_elementwise('mov.u32 $0, %smid;', '=r', [], dtype=tl.int32, is_pure=False, pack=1)
        tl.store(out_ptr + pid, sm + (tl.sum(acc) * 0).to(tl.int32))
    out = torch.full((132 * 32,), -1, dtype=torch.int32, device='cuda')
    with torch.cuda.stream(stream):
        k[(out.numel(),)](out, ITERS=20000)
    stream.synchronize(); return sorted(set(out.tolist()))


def main():
    p = argparse.ArgumentParser(); ServerArgs.add_cli_args(p); args = p.parse_args(); sa = ServerArgs.from_cli_args(args)
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    if REMOTE:
        torch.ones(1, device='cuda:0').to('cuda:1'); torch.ones(1, device='cuda:1').to('cuda:0')     # peer access
    runner, _ = bob.load_model(sa, PortArgs.init_new(sa), 0, 0)
    runner = runner.torch_runner if hasattr(runner, 'torch_runner') else runner
    lanes = get_decode_lanes(); fast_lane, slow_lane = lanes[0], lanes[1]
    runner.init_decode_lane_streams()
    log(lanes=[dict(name=l.name, sm_count=l.sm_count, stream=str(l.stream)) for l in lanes], full_sms=FULL_SMS)
    if os.environ.get('ASH_CHECK_SMS') == '1':
        a, b = sm_ids_on(fast_lane.stream), sm_ids_on(slow_lane.stream)
        log(sm_check=dict(fast=len(a), slow=len(b), overlap=len(set(a) & set(b)), fast_ids=a[:3] + a[-3:], slow_ids=b[:3] + b[-3:]))
    fast_pools = Pools(runner)
    slow_pools = make_slow_pools(runner, SB * (SCTX + SSTEPS + 64), SB, REMOTE) if MODE != 'fast' else None
    # lane backends + graphs: slow lane bound to the slow pools, each lane captured with its own SM count
    if MODE != 'slow':
        fast_lane.attn_backend = runner._get_attention_backend(init_new_workspace=True)
        runner.init_decode_lane_cuda_graph_runner(fast_lane, [FB], fast_lane.stream)   # SGLang sizes kernels for the lane (sm_budget)
    if MODE != 'fast':
        with swapped(runner, slow_pools):
            slow_lane.attn_backend = runner._get_attention_backend(init_new_workspace=True)
            runner.init_decode_lane_cuda_graph_runner(slow_lane, [SB], slow_lane.stream)
        slow_prefill_backend = None
        with swapped(runner, slow_pools):
            slow_prefill_backend = runner._get_attention_backend(init_new_workspace=True)
    log(graphs_ready=True, avail_GB=torch.cuda.mem_get_info(0)[0] / 1e9)
    torch.cuda.empty_cache()
    # prefill (eager, full GPU, sequential)
    jobs = []
    if MODE != 'slow':
        t = time.time(); ids, batch = prefill(make_reqs(FB, FCTX, 1), runner, fast_pools, runner.attn_backend)
        log(prefill_fast_s=time.time() - t); jobs.append(('fast', fast_lane, batch, ids, FSTEPS))
    if MODE != 'fast':
        t = time.time(); ids, batch = prefill(make_reqs(SB, SCTX, 2), runner, slow_pools, slow_prefill_backend)
        log(prefill_slow_s=time.time() - t); jobs.append(('slow', slow_lane, batch, ids, SSTEPS))
    # warm-up 5 steps per lane (sequential), then concurrent decode
    for name, lane, batch, ids, _ in jobs:
        with open(os.devnull, 'w') as dn:
            decode_loop(name, lane, runner, batch, ids, 5, dn)
    files = {name: open(os.path.join(OUT, f'steps_{name}.txt'), 'w') for name, *_ in jobs}
    sig = os.environ.get('ASH_SIGNAL')
    if sig: open(sig, 'w').write(str(time.time()))
    ths = [threading.Thread(target=decode_loop, args=(name, lane, runner, batch, ids, steps, files[name], True)) for name, lane, batch, ids, steps in jobs]
    t0 = time.time()
    for th in ths: th.start()
    for th in ths: th.join()
    for f in files.values(): f.close()
    res = {}
    for name, *_ in jobs:
        t = np.loadtxt(os.path.join(OUT, f'steps_{name}.txt'))[:, 0]; d = np.diff(t[20:]) * 1e3
        res[name] = dict(median_ms=float(np.median(d)), p90_ms=float(np.percentile(d, 90)), n=int(len(d)))
    log(result=res, mode=MODE, remote=REMOTE, wall_s=time.time() - t0)


if __name__ == '__main__':
    main()
