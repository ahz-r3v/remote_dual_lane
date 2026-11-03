"""Lossless compressibility of real BF16 KV (from kv_dump.py) - CPU only.
For K and V separately (all 10 full-attention layers, 2 KV heads, head_dim 256):
  * entropy bound: H(sign) + H(exponent) + H(mantissa) per value, exponent modelled per (layer, head) and per (layer, head, channel)
  * fixed-rate block scheme (cheap to decode in-kernel): blocks of 64 values share a base exponent (8 bit); each value stores
    sign + 7-bit mantissa raw and a b-bit exponent offset; values whose offset does not fit go to an exception list (16-bit raw
    value + 16-bit index). Two block directions: along channels (one token, 64 consecutive channels) and along tokens (one channel,
    64 consecutive tokens).
  * reference: lzma on the exponent plane and the mantissa plane.
Usage: python kv_compress_analysis.py runs/kv_dump/kv_seq*.pt"""
import json, lzma, sys
import numpy as np
import torch


def planes(x):
    u = x.contiguous().view(torch.int16).numpy().astype(np.uint16)
    return (u >> 15) & 1, (u >> 7) & 0xFF, u & 0x7F


def entropy(a, axis_groups=None):
    """bits/value; if axis_groups: a is [groups, n] and entropy is averaged over groups (conditional entropy)."""
    if axis_groups is None:
        _, c = np.unique(a, return_counts=True); p = c / c.sum(); return float(-(p * np.log2(p)).sum())
    tot = 0.0
    for g in a:
        _, c = np.unique(g, return_counts=True); p = c / c.sum(); tot += float(-(p * np.log2(p)).sum())
    return tot / len(a)


def block_scheme(exp, b, along):
    """exp: [layers, tokens, heads, dim] uint8 exponents -> bits/value of the fixed-rate block scheme with b-bit offsets."""
    if along == 'channels':
        blocks = exp.reshape(-1, 64)
    else:  # tokens: put token axis last, then block it
        e = np.moveaxis(exp, 1, -1)  # [layers, heads, dim, tokens]
        T = e.shape[-1] // 64 * 64
        blocks = e[..., :T].reshape(-1, 64)
    base = blocks.max(axis=1, keepdims=True).astype(np.int32)
    off = base - blocks.astype(np.int32)                      # >= 0; small offsets = values near the block's largest magnitude
    exc = off >= (1 << b) - 1                                  # top code reserved as the escape
    n = blocks.size
    bits = n * (1 + 7 + b) + blocks.shape[0] * 8 + exc.sum() * 32
    return bits / n, float(exc.mean())


def lzma_bits(a):
    raw = np.ascontiguousarray(a.astype(np.uint8)).tobytes()
    return 8 * len(lzma.compress(raw, preset=6)) / a.size


def main():
    files = sys.argv[1:]
    res = {}
    for name in ['K', 'V']:
        xs = [torch.load(f)[name] for f in files]                    # each [layers, tokens, heads, dim] bf16
        x = torch.cat(xs, dim=1)
        s, e, m = planes(x)
        L, T, Hh, Dd = e.shape
        r = dict(values=int(e.size))
        r['H_sign'] = entropy(s.ravel()); r['H_mant'] = entropy(m.ravel())
        r['H_exp_global'] = entropy(e.ravel())
        r['H_exp_per_layer_head'] = entropy(np.moveaxis(e, 1, -1).reshape(L * Hh, -1), True)
        r['H_exp_per_channel'] = entropy(np.moveaxis(e, 1, -1).reshape(L * Hh * Dd, -1), True)
        for k in ('global', 'per_layer_head', 'per_channel'):
            r[f'ratio_entropy_{k}'] = 16 / (r['H_sign'] + r['H_mant'] + r[f'H_exp_{k}'])
        for along in ('channels', 'tokens'):
            for b in (2, 3, 4):
                bits, exc = block_scheme(e, b, along)
                r[f'block_{along}_b{b}'] = dict(bits_per_value=round(bits, 3), ratio=round(16 / bits, 3), exception_frac=round(exc, 4))
        sub = slice(0, min(T, 4096))
        es, ms = e[:, sub], m[:, sub]
        r['lzma_exp_bits'] = lzma_bits(es); r['lzma_mant_bits'] = lzma_bits(ms)
        r['ratio_lzma_planes'] = 16 / (r['H_sign'] + r['lzma_exp_bits'] + r['lzma_mant_bits'])
        res[name] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}
        print(json.dumps({name: res[name]}, indent=1), flush=True)


if __name__ == '__main__':
    main()
