"""Per-step time estimate of the slow lane (20 SMs, 16 requests x 32k, BF16, Qwen3.6, FI attention) for combinations of:
remote KV, fraction kept local (f), lossless compression (ratio c, in-kernel decode), and the micro-batch ring.
Inputs are measurements from this repo (2 x H100 NVL); see MEAS. Model:
  link_ms  = (1 - f) * KV_GB / c / LINK_GBps                      (remote bytes over NVLink)
  attn SM-time = SMs needed to keep the link busy x link_ms + local part / per-SM rate + decode cost
  sequential: step = rest_full + max(link_ms, attention compute on 20 SMs) + gaps
  ring (dynamic SM sharing, perfect packing assumed): step = max(link_ms, (attn SM-time + rest halves SM-time) / 20) + gaps"""
MEAS = dict(
    KV_GB=10.74,          # 16 x 32768 tokens x 10 layers x (K+V) x 2 heads x 256 x 2 B
    LINK_GBps=250.0,      # FA3/FI remote saturation (runs/fa3_remote_sms, runs/attn_kernel_cmp)
    FI_GBps_per_SM=55.0,  # FI tensor-core decode, page_size 1, local (runs/attn_kernel_cmp)
    FI_remote_GBps_per_SM=53.7,  # FI with 4 SMs on remote KV before the link saturates
    rest_full_ms=21.3,    # non-attention kernels/step, B16, 20 SMs, FI run (runs/fi_port)
    rest_half_b=215.0, rest_half_a=3.74,  # rest(B8) = a + b/SMs (runs/real_rest_sms fit)
    gaps_ms=4.6,          # wall - kernel time/step, FI local run (35.8 - 31.2)
    local_fi_ms=35.8, remote_fi_ms=70.5,  # measured wall steps (runs/fi_port)
)
M = MEAS


def scenario(f=0.0, c=1.0, ring=False, decode_sm_ms_per_GB=0.0):
    kv = M['KV_GB']
    link = (1 - f) * kv / c / M['LINK_GBps'] * 1e3
    remote_GB_decompressed = (1 - f) * kv
    local_GB = f * kv
    # SMs needed so that attention keeps the link saturated (decompressed rate = c x link rate)
    k = min(20, (M['LINK_GBps'] * c) / M['FI_remote_GBps_per_SM']) if f < 1 else 0
    attn_sm = k * link + local_GB / M['FI_GBps_per_SM'] * 1e3 + decode_sm_ms_per_GB * remote_GB_decompressed * (c > 1)
    if not ring:
        attn_compute = (kv / (M['FI_GBps_per_SM'] * 20)) * 1e3 + decode_sm_ms_per_GB * remote_GB_decompressed * (c > 1) / 20
        step = M['rest_full_ms'] + max(link, attn_compute) + M['gaps_ms']
    else:
        rest_halves_sm = 2 * M['rest_half_b']           # SM*ms of the two half-batch passes (the a terms overlap the other half)
        step = max(link, (attn_sm + rest_halves_sm) / 20) + M['gaps_ms']
    return round(step, 1), round(link, 1), round(attn_sm, 0)


rows = [
    ('FI local (measured)', None),
    ('FI remote direct (measured)', None),
    ('remote direct, model check', dict()),
    ('remote direct + 30% local', dict(f=0.3)),
    ('remote direct + lossless 1.32x', dict(c=1.32, decode_sm_ms_per_GB=10)),
    ('remote direct + 30% local + lossless', dict(f=0.3, c=1.32, decode_sm_ms_per_GB=10)),
    ('ring', dict(ring=True)),
    ('ring + 30% local', dict(ring=True, f=0.3)),
    ('ring + lossless (decode free)', dict(ring=True, c=1.32)),
    ('ring + lossless (decode 10 SM*ms/GB)', dict(ring=True, c=1.32, decode_sm_ms_per_GB=10)),
    ('ring + 30% local + lossless (decode 10)', dict(ring=True, f=0.3, c=1.32, decode_sm_ms_per_GB=10)),
    ('ring + 15% local', dict(ring=True, f=0.15)),
    ('ring + 50% local', dict(ring=True, f=0.5)),
]
base = M['local_fi_ms']
print(f"{'scenario':42s} {'step ms':>8s} {'vs local':>9s} {'link ms':>8s} {'attn SM*ms':>10s} {'local KV GB':>11s}")
for name, kw in rows:
    if kw is None:
        v = M['local_fi_ms'] if 'local' in name else M['remote_fi_ms']
        print(f"{name:42s} {v:8.1f} {100 * (v / base - 1):+8.0f}% {'':>8s} {'':>10s} {(M['KV_GB'] if 'local' in name else 0):11.1f}")
        continue
    s, l, a = scenario(**kw)
    print(f"{name:42s} {s:8.1f} {100 * (s / base - 1):+8.0f}% {l:8.1f} {a:10.0f} {kw.get('f', 0) * M['KV_GB']:11.1f}")
