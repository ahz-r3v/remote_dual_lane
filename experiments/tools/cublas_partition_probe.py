"""Which green-context partitions can run cuBLAS bf16 GEMMs (decode shapes of Qwen3.6, M = 8/16 tokens)?
Partitions: remainder of 120 (12 SMs), remainder of 112 (20 SMs), one 8-SM group, two 8-SM groups (16 SMs). Each GEMM is run
eagerly on the partition's stream, with and without torch's SM carveout (= full - partition SMs). Prints ok / error per case."""
import ctypes, json, os, sys
sys.path.insert(0, os.path.join(os.environ['ASH_SGLANG'], 'python'))
import torch
from sglang.srt.multiplex.greenctx_driver import _DevResource, _driver, _chk, create_split_streams

FULL = torch.cuda.get_device_properties(0).multi_processor_count


def group_streams():
    torch.zeros(1, device='cuda:0'); cu = _driver()
    dev = ctypes.c_int(); _chk(cu.cuDeviceGet(ctypes.byref(dev), 0), 'dev')
    full = _DevResource(); _chk(cu.cuDeviceGetDevResource(dev, ctypes.byref(full), 1), 'res')
    grp, rem, ng = (_DevResource * 16)(), _DevResource(), ctypes.c_uint(16)
    _chk(cu.cuDevSmResourceSplitByCount(grp, ctypes.byref(ng), ctypes.byref(full), ctypes.byref(rem), 0, 8), 'split')

    def stream(rs):
        arr = (_DevResource * len(rs))(*rs)
        d = ctypes.c_void_p(); _chk(cu.cuDevResourceGenerateDesc(ctypes.byref(d), arr, len(rs)), 'desc')
        g = ctypes.c_void_p(); _chk(cu.cuGreenCtxCreate(ctypes.byref(g), d, dev, 1), 'green')
        s = ctypes.c_void_p(); _chk(cu.cuGreenCtxStreamCreate(ctypes.byref(s), g, 1, 0), 'stream')
        return torch.cuda.ExternalStream(stream_ptr=s.value, device=torch.device('cuda:0'))
    out = {'group8': (stream([grp[0]]), 8)}
    try:
        out['group8x2'] = (stream([grp[0], grp[1]]), int(grp[0].smCount + grp[1].smCount))
    except RuntimeError as e:
        print(json.dumps(dict(partition='group8x2', create_error=str(e))), flush=True)
    return out


def main():
    parts = {}
    for n_first in (120, 112):
        (_, s), (_, n) = create_split_streams(n_first, 0); parts[f'rest_of_{n_first}'] = (s, n)
    parts.update(group_streams())
    shapes = [(16, 2048, 4096), (16, 4096, 2048), (16, 2048, 512), (16, 2048, 248320), (8, 2048, 4096), (8, 2048, 248320),
              (16, 2048, 12288), (16, 8192, 2048)]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    for name, (s, n) in parts.items():
        if only and name != only:
            continue
        for carve in (False, True):
            for (M, K, N) in shapes:
                a = torch.randn(M, K, device='cuda:0', dtype=torch.bfloat16); w = torch.randn(N, K, device='cuda:0', dtype=torch.bfloat16)
                torch.cuda.synchronize()
                if carve:
                    torch._C._set_sm_carveout_experimental(FULL - n)
                try:
                    with torch.cuda.stream(s):
                        y = a @ w.t()
                    s.synchronize(); r = 'ok'
                except Exception as e:
                    r = 'ERR ' + str(e).split('\n')[0][:120]
                finally:
                    if carve:
                        torch._C._set_sm_carveout_experimental(None)
                print(json.dumps(dict(partition=name, sms=n, carveout=carve, M=M, K=K, N=N, result=r)), flush=True)
                if r != 'ok':
                    torch.cuda.synchronize()


if __name__ == '__main__':
    main()
