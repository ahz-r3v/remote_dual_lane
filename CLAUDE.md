# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A reproduction package for a prototype experiment, not a library. Inside one SGLang process on a compute GPU, two decode batches run concurrently on disjoint green-context SM partitions:
- the **fast lane** (112 SMs, KV in local HBM), and
- the **slow lane** (the remaining SMs, 20 on H200). Its full-attention KV pool sits in an NVLink peer GPU's HBM, and FA3 reads and writes it over NVLink.

The peer GPU (the "KV host" or "owner") runs its own plain decode at the same time, so we can measure the slowdown it suffers. `README.md` has the requirements, the expected numbers, and the sanity checks. Its numbers come from the reference run in `reference_results/` (job 152114, 2 x H200).

## Commands

```bash
cp env_template.sh env.sh && source env.sh   # sets PYTHON, PYTHONPATH (bundled SGLang first), caches, HF_HUB_OFFLINE=1
MODEL=/path/Qwen3.6-35B-A3B-FP8 GPUS=0,1 OUT=./runs/try1 bash run_impact.sh                 # full run, ~35 min on H200
MODEL=... GPUS=0,1 OUT=./runs/try1 bash run_impact.sh check slow_remote both_remote          # subset of steps
python3 summarize.py reference_results/steps     # median step per lane; also run automatically at the end of run_impact.sh

# ASH unit tests (CPU only, no GPU needed)
$PYTHON sglang_ash_r001/sglang/test/ash/test_sm_budget.py      # or: pytest sglang_ash_r001/sglang/test/ash/test_sm_budget.py::test_choose_lane_num_splits
# FA3 lane-split GPU check (every line must say ALL_OK)
CUDA_VISIBLE_DEVICES=0 $PYTHON sglang_ash_r001/sglang/test/ash/test_fa3_lane_split_gpu.py --sms 16,20,32 --batches 16 --ctx 4096,32768 --out /tmp/fa3.json
```

There is no build or lint step. Anything that runs the model needs 2 NVLink GPUs with peer access.

## Architecture

**Two-layer split: SGLang source changes vs. harness.**
- `sglang_ash_r001/sglang/python` is a modified SGLang tree. It must be first on `PYTHONPATH`, and a pip-installed `sglang` must never be used. The source changes make every kernel whose launch configuration depends on SM count size itself for the lane's SMs, not the whole device. That covers FA3 splits and grid, deep_gemm, cuBLAS, and the MoE config lookup. The core is `srt/multiplex/sm_budget.py` (the `sm_budget(n)` context and `set_lane_sm_count`). `ModelRunner` enters it for lane graph capture and eager lane decode. Every change is marked `ASH r001` in the code (`grep -rn "ASH r001"`). `ASH_VERSION.md` documents them file by file.
  - Fixes for kernel or SM-sizing behaviour go into the SGLang source, not into harness monkeypatches. This is an explicit user decision recorded in `ASH_VERSION.md`.
  - `ASH_CHANGES.patch` is a `diff -ruN` against the frozen base tree. It is supposed to be regenerated after every SGLang change, but the base tree (`experiments/amd_prefill_decode_perf_r001/src`) is not in this repo.
  - Only the FA3 attention backend is lane-aware. MoE uses upstream default configs in every lane, and no SM-specific tuned files ship.
- `bench/dual_lane_bench_r005.py` holds only the experiment-specific logic:
  - It uses one `ModelRunner`, so the weights are loaded once.
  - The slow lane gets its own pools by re-running `_init_pools` with the runner's pools swapped out (`POOL_ATTRS`, plus a context manager that swaps them in and out). The slow lane's attention backend and CUDA graphs are created while the runner points at those pools.
  - **Remote KV** (`ASH_REMOTE_KV=1`): `MHATokenToKVPool._create_buffers` is wrapped, and `torch.zeros` is temporarily patched so KV buffers are allocated on `cuda:1` and aliased into `cuda:0` via `torch._C._construct_storage_from_data_pointer`.
  - Each lane runs one decode thread that replays its lane graph on the lane stream and writes per-step timestamps to `steps_<lane>.txt`.
  - Prefill is eager and sequential, on the full GPU.
- `bench/dual_bench.py` and `bench/ash_bench.py` are the owner process: SGLang `bench_one_batch` decode plus step logging. In `dual_bench.py` only the `owner` role is used.

**Process/GPU layout in `run_impact.sh`:**
- `GPUS=G0,G1`: G0 = KV host, G1 = compute.
- The lanes process gets `CUDA_VISIBLE_DEVICES=G1,G0`, so inside it `cuda:0` = compute and `cuda:1` = KV host.
- In `both_remote`, the owner process runs on G0. The two processes stay in lockstep through file-based barriers (`ASH_READY`/`ASH_GO`/`ASH_DREADY`/`ASH_DGO`/`ASH_SIGNAL` files in the mode's output dir).
- Shapes and step counts are set at the top of `run_impact.sh` and passed to the harness through `ASH_*` env vars (`ASH_MODE=fast|slow|both`, `ASH_FAST_B/CTX/STEPS`, `ASH_SLOW_*`, `ASH_OUT`, `ASH_CHECK_SMS`, `ASH_DUMP_LOGITS`, `ASH_PROFILE_STEP`).
- Lane SM splits come from YAML (`bench/lanes_112_rest.yaml`). `long_green_ctx_sm_count: -1` means "all remaining SMs", built with the CUDA driver API in `srt/multiplex/greenctx_driver.py`.
- Setting `NSYS=/path/to/nsys` wraps runs in nsys for per-kernel traces.

**Outputs and validation:** each mode writes `runs/<tag>/<mode>/{lanes.log,steps_*.txt,logits_*.pt}` and adds a line to `progress.log`. Check these:
- `sm_check` should show fast 112 / slow 20 / overlap 0.
- `slow_pools` should show `remote: true` in the remote modes.
- Remote-vs-local logits should match exactly (max diff 0).

`summarize.py` takes the median over steps from step 20 onward. For `both_*` modes it only uses the window where both lanes are decoding.

## Follow-up experiments

`experiments/NOTES.md` has three things:
- the results of the 2 x H100 NVL follow-up work (remote-KV cost, FlashInfer port, ring, KV compression, combination model);
- the extra environment the local machine needs (CUDA 13.0 toolchain, `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`);
- the plan.

Scripts are in `experiments/tools/` and small results in `experiments/results/`. Raw outputs go to the gitignored `runs/`.

Rules for GPU work:
- Every GPU process goes through gpuguard (`python3 -m gpuguard run --gpu ... --log ... -- cmd`, from `/home/chengzhi/github/gpuguard`).
- Write GPU tests as atomic units: one config per process, with a gpuguard yield point between units.
