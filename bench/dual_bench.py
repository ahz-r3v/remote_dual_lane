"""greenctx_dual_r001 prototype (user 2026-10-15: "先在H100上做一个验证prototype，GPU0上跑普通decode，GPU1上用GreenContext划分96/32个SM
分别作为fast batch和slow batch，实现我们之前说的那个设计"). One SGLang bench_one_batch process per role (ash_bench.py chunked prefill):
  ASH_ROLE=owner : plain decode (GPU0); its HBM also holds the slow batch's KV.
  ASH_ROLE=fast  : green context with the first 96 SMs of GPU1 (sm_partition.py), full decode, local KV.
  ASH_ROLE=slow  : green context with 32 other SMs of GPU1, full decode (weights + GDN state local); with ASH_REMOTE_KV=1 the
                   full-attention KV pool (MHATokenToKVPool._create_buffers) is allocated on the peer GPU and aliased into this
                   device, so FA3 reads it with TMA over NVLink and new K/V are written there.
fast / slow also get the real SM count (green_ctx.fix_sm_count: device props, deep_gemm.set_num_sms, FA3 sm_margin).
Slow sees CUDA_VISIBLE_DEVICES=<GPU1>,<GPU0>: cuda:0 = compute GPU, cuda:1 = KV GPU. Every decode step of the main run is
timestamped to $ASH_STEP_LOG; $ASH_SIGNAL is created when decoding starts. Optional start barrier: after warm-up the
main run frees cached memory, writes $ASH_READY and waits for $ASH_GO; optional decode barrier: before the first decode
step it writes $ASH_DREADY and waits for $ASH_DGO (so the decode phases of the combined run overlap).
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROLE = os.environ.get('ASH_ROLE', 'owner')
N = {'fast': int(os.environ.get('ASH_FAST_SMS', '96')), 'slow': int(os.environ.get('ASH_SLOW_SMS', '32'))}.get(ROLE)
import torch  # noqa: E402
if ROLE in ('fast', 'slow'):
    import sm_partition as sp
    sp.enter(ROLE, int(os.environ.get('ASH_FAST_SMS', '96')), int(os.environ.get('ASH_SLOW_SMS', '32')))
    if os.environ.get('ASH_REMOTE_KV') == '1':
        torch.ones(1, device='cuda:0').to('cuda:1'); torch.ones(1, device='cuda:1').to('cuda:0')    # peer access
        sp.ensure()
    import green_ctx
    green_ctx.fix_sm_count(N)
    sp.ensure()
import ash_bench  # noqa: E402,F401  (chunked prefill)
import sglang.bench_one_batch as bob  # noqa: E402

if ROLE == 'slow' and os.environ.get('ASH_REMOTE_KV') == '1':
    from sglang.srt.mem_cache import memory_pool as mp
    _orig_create = mp.MHATokenToKVPool._create_buffers

    def remote_create(self):
        real_zeros = torch.zeros
        state = {'bytes': 0}

        def zeros_remote(*a, **k):
            dev = torch.device(k.get('device', 'cpu'))
            if dev.type == 'cuda' and (dev.index or 0) == 0:
                k = dict(k, device='cuda:1'); t = real_zeros(*a, **k)
                st = torch._C._construct_storage_from_data_pointer(t.data_ptr(), torch.device('cuda', 0), t.numel() * t.element_size())
                al = torch.empty(0, dtype=t.dtype, device='cuda:0').set_(st, 0, t.shape, t.stride())
                al._ash_owner = t                                       # keep the peer allocation alive
                state['bytes'] += t.numel() * t.element_size()
                return al
            return real_zeros(*a, **k)
        torch.zeros = zeros_remote
        try:
            _orig_create(self)
        finally:
            torch.zeros = real_zeros
            torch.cuda.synchronize(1); sp.ensure()
        print('ASH_REMOTE_KV ' + json.dumps({'layers': len(self.k_buffer), 'GB_on_peer': state['bytes'] / 1e9,
                                             'k0_device': str(self.k_buffer[0].device)}), flush=True)
    mp.MHATokenToKVPool._create_buffers = remote_create

LOG, SIG = os.environ['ASH_STEP_LOG'], os.environ['ASH_SIGNAL']
_orig = bob.latency_test_run_once


def run_once(run_name, model_runner, *a, **k):
    output_len = a[4] if len(a) > 4 else k.get('output_len')
    main = output_len is not None and output_len > 100
    if main and os.environ.get('ASH_READY') and os.environ.get('ASH_GO'):   # start barrier (combined run)
        torch.cuda.empty_cache()             # hand warm-up activations back to the driver (another process shares this GPU)
        open(os.environ['ASH_READY'], 'w').write(str(time.time()))
        while not os.path.exists(os.environ['ASH_GO']):
            time.sleep(0.05)
    if main:
        if ROLE in ('fast', 'slow'):
            sp.ensure()
            e, g = sp.sm_ids()
            print('ASH_SMS_BEFORE_DECODE ' + json.dumps({'role': ROLE, 'eager': len(e), 'graph': len(g), 'ids_head': e[:3], 'ids_tail': e[-3:]}), flush=True)
        f = open(LOG, 'w'); n = [0]; dec = model_runner.decode

        def logged(*x, **y):
            if n[0] == 0:
                if os.environ.get('ASH_DREADY') and os.environ.get('ASH_DGO'):   # decode barrier: all prefills done first
                    torch.cuda.synchronize(); torch.cuda.empty_cache()           # release this prefill's activations for the next process
                    open(os.environ['ASH_DREADY'], 'w').write(str(time.time()))
                    while not os.path.exists(os.environ['ASH_DGO']):
                        time.sleep(0.01)
                open(SIG, 'w').write(str(time.time()))
            f.write(f'{time.time():.6f} {n[0]}\n'); n[0] += 1
            return dec(*x, **y)
        model_runner.decode = logged
    try:
        return _orig(run_name, model_runner, *a, **k)
    finally:
        if main:
            model_runner.decode = dec; f.close()


bob.latency_test_run_once = run_once
if __name__ == '__main__':
    import argparse, logging
    from sglang.srt.server_args import ServerArgs
    p = argparse.ArgumentParser(); ServerArgs.add_cli_args(p); bob.BenchArgs.add_cli_args(p)
    args = p.parse_args(); sa = ServerArgs.from_cli_args(args); ba = bob.BenchArgs.from_cli_args(args)
    logging.basicConfig(level=getattr(logging, sa.log_level.upper()), format='%(message)s')
    bob.main(sa, ba)
