"""NVLink peer-read bandwidth on this node: copy engine (peer memcpy) and SM loads (triton read kernel over a tensor that
lives on cuda:1, aliased into cuda:0) with a limited number of CTAs/SMs. Run with CUDA_VISIBLE_DEVICES=<compute>,<kv host>."""
import torch, triton, triton.language as tl
torch.ones(1, device='cuda:0').to('cuda:1'); torch.ones(1, device='cuda:1').to('cuda:0')   # peer access, as the harness
N = 1 << 30  # bytes
src = torch.empty(N // 2, dtype=torch.bfloat16, device='cuda:1').normal_()
st = torch._C._construct_storage_from_data_pointer(src.data_ptr(), torch.device('cuda', 0), N)
alias = torch.empty(0, dtype=torch.bfloat16, device='cuda:0').set_(st, 0, src.shape, src.stride())
loc = torch.empty_like(alias, device='cuda:0').normal_()
dst = torch.empty_like(loc)

@triton.jit
def rd(x_addr, out, n, CHUNK: tl.constexpr, BLOCK: tl.constexpr):
    x = x_addr.to(tl.pointer_type(tl.bfloat16)); pid = tl.program_id(0); nprog = tl.num_programs(0); acc = tl.zeros([BLOCK], tl.float32)
    for c in range(pid, tl.cdiv(n, CHUNK), nprog):
        for o in range(0, CHUNK, BLOCK):
            idx = c * CHUNK + o + tl.arange(0, BLOCK)
            acc += tl.load(x + idx, mask=idx < n, other=0.).to(tl.float32)
    tl.store(out + pid, tl.sum(acc))

def t(f, it=10):
    f(); torch.cuda.synchronize(0); e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record(); [f() for _ in range(it)]; e1.record(); torch.cuda.synchronize(0); return e0.elapsed_time(e1) / it / 1e3

print('copy_ via alias (same-device view) %.0f GB/s' % (N / t(lambda: dst.copy_(alias)) / 1e9))
print('peer copy cuda:1->cuda:0 (copy engine) %.0f GB/s' % (N / t(lambda: dst.copy_(src, non_blocking=True)) / 1e9))
print('local copy cuda:0 (for scale) %.0f GB/s' % (N / t(lambda: dst.copy_(loc)) / 1e9))
out = torch.empty(4096, device='cuda:0')
for name, x in [('remote', alias), ('local', loc)]:
    for ctas in [20, 40, 80, 132, 264, 528, 1056]:
        for w in [4, 8]:
            bw = N / t(lambda: rd[(ctas,)](x.data_ptr(), out, x.numel(), CHUNK=65536, BLOCK=4096, num_warps=w)) / 1e9
            print(f'SM loads {name:6s} ctas {ctas:5d} warps {w}  {bw:6.0f} GB/s')
