"""Per-decode-step GPU kernel time by kernel name, from an nsys report (CUDA-graph kernels only = decode, no prefill).
Usage: python kernel_breakdown.py <lanes.nsys-rep or .sqlite> [steps]"""
import sqlite3, subprocess, sys, os
from collections import defaultdict
rep = sys.argv[1]; db = rep.replace('.nsys-rep', '.sqlite')
if not os.path.exists(db):
    subprocess.run(['nsys', 'export', '-t', 'sqlite', '-o', db, rep], check=True, capture_output=True)
c = sqlite3.connect(db)
cols = {r[1] for r in c.execute('pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)')}
g = 'graphNodeId IS NOT NULL AND graphNodeId != 0' if 'graphNodeId' in cols else '1'
rows = c.execute(f'select s.value, k.end-k.start, k.streamId, k.start from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id where {g}').fetchall()
by_stream = defaultdict(int)
for _, d, st, _ in rows: by_stream[st] += d
st = max(by_stream, key=by_stream.get)          # the lane stream that replays the graph
rows = [r for r in rows if r[2] == st]
tot = defaultdict(float); cnt = defaultdict(int)
for n, d, _, _ in rows: tot[n] += d; cnt[n] += 1
steps = int(sys.argv[2]) if len(sys.argv) > 2 else None
if steps is None:   # replays = instances of the most frequent per-step-once kernel ~ guess from the sampler-free minimum
    steps = min(v for v in cnt.values())
T = sum(tot.values())
print(f'stream {st}  graph kernels {len(rows)}  steps~{steps}  kernel time/step {T/steps/1e6:.3f} ms')
for n, v in sorted(tot.items(), key=lambda x: -x[1])[:15]:
    print(f'{v/steps/1e6:8.3f} ms/step  {100*v/T:5.1f}%  x{cnt[n]/steps:5.1f}/step  {n[:90]}')
