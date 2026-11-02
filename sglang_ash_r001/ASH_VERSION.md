# sglang_ash/r001 — SGLang for ASH method experiments

Copy of the frozen SGLang tree `experiments/amd_prefill_decode_perf_r001/src` (r008 release Python tree), made 2026-10-17.
User decisions: "不行，要直接修改SGLang源码" and "这个新的SGLang需要用作我们后来的方法实验，所以做好点" — fixes go into SGLang source
(not harness monkeypatches); the frozen tree stays untouched for the experiments that already used it.
Full diff vs the frozen tree: `ASH_CHANGES.patch` (`diff -ruN`, regenerate after every change).
Environment on gaas H200: `source env_gaas_h200.sh` (same interpreter / site-packages as the frozen setup, SGLang = this tree).

## What changed (all marked `ASH r001` in the code)

Theme: **kernels whose launch configuration depends on the SM count are configured for the SMs a green-context decode lane
actually owns**, instead of for the whole device. Before this, a lane with 112 or 96 of 132 H200 SMs lost 23–24% decode speed
(FA3 split-KV heuristic), and several other kernels were silently sized for 132 SMs.

| File | Change |
|---|---|
| `srt/multiplex/sm_budget.py` (new) | `sm_budget(n)` context: `current_sm_budget()` = n for code configured inside it; also sets deep_gemm `num_sms` and the cuBLAS/cuBLASLt SM-count target (`torch._C._set_sm_carveout_experimental`). `set_lane_sm_count(backend, n)` records a lane's SM count on every FA3 backend inside a (hybrid) attention backend. |
| `srt/model_executor/model_runner.py` | Lane CUDA-graph capture (`init_decode_lane_cuda_graph_runner`) and eager lane decode/extend run inside `sm_budget(lane.sm_count)`; lane FA3 backends get their SM count when the green-context streams are created and before capture. |
| `srt/layers/attention/flashattention_backend.py` | FA3 calls get `sm_margin = full − budget` under a partial budget (grid = lane SMs). A lane backend picks `num_splits = choose_lane_num_splits(batch × KV heads, lane SMs)` (whole waves) and computes the scheduler metadata with `fa3_metadata_num_sm(...)` so FA3's *dynamic* split rule lands on it (capture, replay and eager). Prefill keeps the original split setting. |
| `srt/layers/moe/.../fused_moe_triton_config.py` | Lookup mechanism only: config file names can carry `,sms=N`; under a partial budget the lane's file is tried first, then the full-device file, then the default. **No tuned files are shipped**, so behaviour = upstream. |
| `srt/layers/moe/.../fused_moe.py` | Lookup mechanism only: the small-batch (≤32 tokens) sum-reduce implementation can be chosen per SM budget from an optional table `configs/triton_<ver>/moe_sum_reduce,device_name=<GPU>.json`. **No table is shipped**, so it stays torch.compile (upstream). |
| `srt/multiplex/decode_lane_context.py`, `srt/multiplex/greenctx_driver.py` (new) | `long_green_ctx_sm_count: -1` = second lane takes all remaining SMs, any size (e.g. 128 + 4 on H200), built with the CUDA driver API; only for plain two-lane decode mux (validated at config load). |
| `test/ash/test_sm_budget.py`, `test/ash/test_fa3_lane_split_gpu.py` (new) | CPU unit tests (split choice, virtual SM count vs FA3's dynamic rule, -1 sentinel, config names, budget context); GPU check of FA3 lane split vs the unsplit reference. |

## MoE tuning: removed (2026-10-17)
User: "完全撤掉MOE调参，恢复之前的配置文件，只保留按SM数查找配置". The configs tuned in ASPIRE job 152080 (and the tuning tool) were
moved out of the tree to `gpu/sglang_ash_r001_smbudget/archive_moe_tune_20261017/`; the `configs/` directory is identical to the
frozen tree again (upstream default MoE config for Qwen3.6-35B-A3B FP8 on H200, as before). Why the tuned configs were faster
(BLOCK_SIZE_M 16 vs 64 for ~1 token per expert in decode) is noted in that archive's results, not used here.

## Validation status
- CPU unit tests: pass (login node, 2026-10-17).
- GPU unit/FA3 checks: pass (ASPIRE 152080: 5/5 unit, 18/18 FA3 lane-split cases).
- GPU (decode benchmark full GPU / 128 / 112 / 96 and same-node reference against
  the frozen tree + r004 patches): gaas job, results in `gpu/sglang_ash_r001_smbudget/runs/<job>/`.

## Not changed / known limits
- FA3 lane split requires the scheduler-metadata path (disabled upstream under DP attention); without it the kernel's own
  heuristic applies.
- Only the FA3 attention backend is lane-aware; other attention backends are not.
- MoE kernels use the upstream default config in every lane (no SM-specific files shipped).
