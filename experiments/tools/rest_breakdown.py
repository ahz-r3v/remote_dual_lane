"""attention (FA3 device_kernel or FlashInfer paged-KV/merge kernels) / rest (non-attention) GPU time per decode step of the slow lane, from runs/real_rest_sms/<cfg>/lanes.nsys-rep."""
import glob, os, sqlite3, subprocess, sys
R = sys.argv[1] if len(sys.argv) > 1 else 'runs/real_rest_sms'
for d in sorted(glob.glob(f'{R}/sm*_b*')):
    if 'smoke' in d or not os.path.exists(f'{d}/lanes.nsys-rep'):
        continue
    db = f'{d}/lanes.sqlite'
    if not os.path.exists(db):
        subprocess.run(['nsys', 'export', '-t', 'sqlite', '-o', db, f'{d}/lanes.nsys-rep'], check=True, capture_output=True)
    c = sqlite3.connect(db)
    try:
        c.execute('select 1 from CUPTI_ACTIVITY_KIND_KERNEL limit 1')
    except sqlite3.OperationalError:
        print(f'{os.path.basename(d):12s} report not exportable yet (run in progress or stopped)'); continue
    row = c.execute("select streamId from CUPTI_ACTIVITY_KIND_KERNEL where graphNodeId is not null group by streamId order by sum(end-start) desc limit 1").fetchone()
    if row is None:
        print(f'{os.path.basename(d):12s} no decode graph kernels (run failed before decode)'); continue
    st = row[0]
    q = lambda w: c.execute(f"select coalesce(sum(k.end-k.start),0), count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id "
                            f"where k.graphNodeId is not null and k.streamId=? {w}", (st,)).fetchone()
    tot, _ = q('')
    att, _ = q("and (s.value='device_kernel' or s.value like 'Batch%PagedKVCacheKernel' or s.value like '%MergeStates%')")  # FA3 or FlashInfer
    moe, nmoe = q("and s.value='fused_moe_kernel'")
    steps = nmoe / 80  # 2 fused-MoE GEMMs x 40 layers per step (FA3 has no combine kernel when num_splits = 1)
    print(f'{os.path.basename(d):12s} steps {steps:4.0f}  attention {att/steps/1e6:6.2f}  rest {(tot-att)/steps/1e6:6.2f} '
          f'(MoE {moe/steps/1e6:6.2f}, non-MoE {(tot-att-moe)/steps/1e6:6.2f}) ms/step')
