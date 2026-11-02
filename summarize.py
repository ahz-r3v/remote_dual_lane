#!/usr/bin/env python3
"""Median decode step per lane / GPU for a run_impact.sh output directory (wall-clock step timestamps, steps_*.txt).
Solo runs: median over steps 20.. ; combined runs (both_*): median inside the window where all participants are decoding.
Usage: python3 summarize.py <run dir>
"""
import sys
from pathlib import Path

import numpy as np


def steps(f):
    t = np.loadtxt(f, ndmin=2)[:, 0]
    return t[20:]


def med(t, lo=-np.inf, hi=np.inf):
    d = np.diff(t) * 1e3
    m = (t[1:] > lo) & (t[1:] <= hi)
    return float(np.median(d[m])) if m.sum() > 10 else float('nan')


R = Path(sys.argv[1])
rows = []
for mode in ['full132', 'solo_owner', 'solo_fast', 'solo_slow_local', 'solo_slow_remote', 'both_remote', 'both_local']:
    files = {f.stem.replace('steps_', ''): steps(f) for f in sorted((R / mode).glob('steps_*.txt'))}
    if not files:
        continue
    if mode.startswith('both'):
        lanes = [k for k in files if k in ('fast', 'slow')]
        lo = max(files[k][0] for k in lanes)
        hi = min(files[k][-1] for k in lanes)
        rows.append((mode, {k: med(v, lo, hi) for k, v in files.items()}, hi - lo))
    else:
        rows.append((mode, {k: med(v) for k, v in files.items()}, None))
print(f'{"mode":18s} {"fast/owner/slow median step (ms)":40s} overlap window')
for mode, r, w in rows:
    s = '  '.join(f'{k} {v:.2f}' for k, v in r.items())
    print(f'{mode:18s} {s:40s} {"" if w is None else f"{w:.0f} s"}')
