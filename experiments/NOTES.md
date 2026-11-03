# Remote-KV slow lane: experiments on 2 x H100 NVL (2026-11-02 .. 11-03)

These are follow-up experiments to the r001 reproduction package. Everything ran on one node with 2 x H100 NVL (96 GB, 132 SMs,
NV12 bridge). Unless noted otherwise, the slow lane is configured as follows:
- 20 SMs;
- 16 requests x 32k context;
- BF16 KV;
- Qwen3.6-35B-A3B-FP8.

`experiments/tools/` holds the scripts. `experiments/results/<run>/` holds small result files: logs, json, and breakdowns. The raw
nsys reports, sqlite exports and KV dumps are kept outside git, in `runs/` (on this machine a symlink to
`/mnt/vdb/chengzhi/remote_dual_lane_runs`).

Tags used below:
- [M] measured.
- [E] estimated from measurements.

## Framing

The design is called **compute / KV-storage separation**. KV placement is a slow decision: KV can live in any GPU's HBM. Compute
placement (lane, GPU, SM count) can change without moving the KV cache synchronously.
- Separation does not save link bytes. Exact attention reads the whole KV on every step, while a migration moves it once. The gains
  are capacity, no synchronous KV move, and tolerance of frequent rescheduling.
- Remote attention on the KV host is prior work (DistAttention, Adrenaline) and is out of scope.
- Related work and the novelty gap are in the project notes: AQUA and Harvest copy the KV back first; DirectKV does zero-copy reads
  from CPU memory; CloudMatrix384 does UB pooling.

## Environment (this machine)

- **venv**: `/mnt/vdb/chengzhi/venv/sglang-cta-control`. It has the same package versions as the reference run.
- **CUDA_HOME**: `/mnt/vdb/chengzhi/cuda13.0_pip/nvidia/cu13`. This holds CUDA 13.0 nvcc, crt, nvvm, cudart, cccl and curand,
  installed together with `pip install --no-deps --target`, plus `libcudart.so` and `lib64 -> lib` symlinks.
  - The venv's own CUDA mixes nvcc 13.2 with 13.0 headers and has no CCCL, so the DeepGEMM and FlashInfer JIT builds fail with it.
  - Never add a package to that dir with `--target --upgrade`: it replaces the whole `nvidia/` dir.
- **`env.sh`** (not in git) is `env_template.sh` plus: the venv `bin/` on PATH (for ninja), `CUDA_HOME` set as above, and
  `$CUDA_HOME/bin` on PATH.
- **Run settings**:
  - `SGLANG_JIT_DEEPGEMM_PRECOMPILE=0`. Otherwise DeepGEMM precompiles 16384 Ms, which takes more than an hour on both GPUs.
  - `EXTRA_ARGS='--load-format dummy'` (new in `run_impact.sh`) skips the 35 GB checkpoint read. Dummy weights cannot judge accuracy:
    with them, FP8 and BF16 logits were bit-identical.
- **GPU access**: every GPU process goes through gpuguard
  (`PYTHONPATH=/home/chengzhi/github/gpuguard python3 -m gpuguard run --gpu 0,1 --log ... -- cmd`; exit 75 means another job
  disturbed the run, so rerun it).
  - Tests are atomic: one config per process, with a gpuguard yield point between units, so higher-priority jobs can cut in.
  - Never edit a bash driver while it runs.

## Results

### 1. Where the remote-KV time goes [M]
Runs: `diag_slow_dummy4`, `diag_slow_fp8`, `peer_bw`, `fa3_remote_sms`.

| slow lane, dummy weights | local | remote |
|---|---|---|
| BF16 step / FA3 attention per step | 44.92 / 13.08 ms | 76.73 / 43.74 ms |
| FP8 KV step / FA3 attention per step | 42.23 / 10.53 ms | 54.66 / 22.03 ms |

- About 97% of the remote penalty is the FA3 main kernel.
  - Remote FA3 reads about 251 GB/s.
  - A copy-engine peer copy reaches 265 GB/s.
  - So remote FA3 is **NVLink-bandwidth bound** (link-bound, not latency-bound).
- The link saturates at 8 SMs (BF16) and at 12-16 SMs (FP8).
  - Below saturation, a remote SM processes data as fast as a local one: about 42 GB/s/SM for BF16 and about 26 GB/s/SM for FP8.
  - So the per-SM limit is compute, not transfer.
  - num_splits barely matters.

### 2. Bandwidth-proportional split and transfer schemes [M]
Run: `split_transfer`.

**CUDA VMM mixed pool.** One contiguous virtual address range is built from 2 MB chunks, each local or remote.
- It works with **unmodified FA3** and gives bit-identical output.
- t = max(remote_bytes / 250 GB/s, all_bytes / 20-SM compute).
- About 30% of the KV can be remote at no cost: 1.36 ms vs 1.30 ms all-local.
- Under a 112-SM local-HBM hog, all-local slows to 2.11 ms, and 75-87.5% local is just as fast. The hog's bandwidth rises from
  3141 to 3352 GB/s as more slow-lane KV goes remote (the Colloid effect).

**10-layer step.** The spin "other" kernel actually ran about 1.27 ms per layer.

| scheme | step |
|---|---|
| all-local | 25.7 ms |
| in-place remote | 55.7 ms |
| layer prefetch | 43.1 ms (+2.15 GB local; hog -6%) |
| migrate one request (671 MB) | 2.9 ms; breaks even in under 2 steps |

### 3. Micro-batch ring (emulated rest = weight-reading GEMM) [M]
Runs: `ring_bench`, `ring_bench_all`.

| scheme | step | vs local |
|---|---|---|
| local sequential | 39.4 ms | — |
| remote direct | 69.4 ms | +76% |
| ring (attn 8 SMs / rest 12 SMs) | 48.6 ms | +23% (a previous run gave +42%) |
| layer prefetch | 45.5 ms | — (+2.15 GB local) |
| migrate the whole batch (10.7 GB) | 40.7 ms once | — |

- The ring hides the remote penalty completely: a ring with remote KV equals a ring with local KV.
- The cost is on the rest side: fewer SMs, plus weight reads duplicated by splitting the batch.

### 4. Real-weight rest vs SMs [M]
Run: `real_rest_sms`, using group-aligned lanes 104+{8,16,24} and 112+rest.

| SMs | rest per step, B16 / B8 |
|---|---|
| 8 | 45.3 / 30.6 ms |
| 16 | 24.7 / 17.2 ms |
| 20 | 20.8 / 15.6 ms |
| 24 | 18.5 / 13.2 ms |

- Local step for B16: 38.7 ms on 20 SMs, 34.2 ms on 24 SMs.
- Fit: rest = a + b/SMs.
  - B16: a = 4.2, b = 329 SM\*ms.
  - B8: a = 3.7, b = 215 SM\*ms.
  - So the rest takes 1.52x longer on 12 SMs than on 20, and two half-batches cost +31% work.
- **A 12-SM remainder partition (120+rest) fails some cuBLAS GEMMs**: CUBLAS_STATUS_EXECUTION_FAILED, with or without the SM
  carveout (run `cublas_probe`). Whole 8-SM groups work. A carveout switch was tried in `sm_budget.py` and reverted.

### 5. Attention kernel: FA3 vs FlashInfer [M]
Runs: `attn_kernel_cmp`, `fi_port`.

**What FA3 decode is.** It is `FlashAttnFwdSm90` with kBlockM = 128, PackGQA, and PagedKVNonTMA (because page_size is 1).
- Only 8 of 128 M rows are useful: 16 Q heads / 2 KV heads, with 1 token.
- The K/V loads use cp.async, not TMA.

**Per-SM throughput.**

| kernel | GB/s per SM |
|---|---|
| FA3 | ~42 |
| **FlashInfer tensor-core decode** | **~58-62** (page_size 1: -8%) |
| FI CUDA-core decode | 12 |
| FI fa3 backend (qo = 1) | ~3 |

- FlashInfer tensor-core decode reaches 215-223 GB/s remote with 4 SMs.
- Outputs match FA3 (max diff 2e-4).

**Real harness, `--attention-backend flashinfer`.** No code change was needed, and lane graph capture works.

| | FA3 | FlashInfer |
|---|---|---|
| local step | 38.7 ms | 35.8 ms |
| local attention per step | 12.9 ms | 9.9 ms |
| remote step | 69.7 ms | 70.5 ms (link-bound) |

No lane CTA tuning was applied yet.

### 6. KV compression
- **FP8 KV** [M, dummy weights]: halves the link time, see table 1. Accuracy is untested.
- **Lossless, real KV** [M] (`kv_dump`; 2 x 16k tokens of ShareGPT text): sign + mantissa are incompressible (7.97 bits), and the
  exponent carries about 2.5-2.8 bits. That gives:
  - entropy bound: about 1.50-1.53x;
  - lzma: 1.45x;
  - fixed-rate scheme (64-value block base exponent + 4-bit offset + raw sign/mantissa byte, 12 bit/value): **1.32x**, with about
    1e-4 exceptions. It is cheap to decode in-kernel.

### 7. Combination model [E]
Script: `experiments/tools/combo_model.py`; output: `results/kv_dump/combo_model.txt`. Model check: direct remote is estimated at
68.9 ms vs 70.5 ms measured.

| scenario (FI, 20-SM slow lane) | step | vs local 35.8 ms | local KV |
|---|---|---|---|
| remote direct | 70.5 [M] | +97% | 0 |
| direct + 30% local | 56.0 | +56% | 3.2 GB |
| direct + lossless 1.32x | 58.4 | +63% | 0 |
| ring | 47.6 | +33% | 0 |
| **ring + ~30% local** | **36.0** | **+1%** | **3.2 GB** |
| ring + lossless | 37.1 - 41.5 | +4 - 16% | 0 |

The ring becomes compute-bound at about 31.5 ms of SM time. Any link time below that is hidden, so the local fraction should be
about 27%. Compression does not help the ring: decoding costs SMs. It helps direct reads, or a ring with zero local memory.

## Decisions so far (user)
- Out of scope:
  - remote attention on the KV host;
  - AI-side methods (MTP, sparse attention, low-rank KV).
- The direction is to trade compute for transfer, through compression and overlap, and to explore FlashInfer.
- MoE SM-aware tuning was removed on 2026-10-17. Re-enabling it is still the user's call.

## Plan / next steps
1. **Combined microbench**: ring + 30% local + FI. It uses a VMM 30/70 pool and two streams in one 20-SM green context, with
   attention at high priority and FI capped to ~5 CTAs. It validates the main untested assumption: that a dynamically shared ring
   packs SMs well, where the static 8/12 split measured +23-42%.
2. **FI port**:
   - run both lanes (`both_remote` / `both_local`);
   - FI inside the ring bench;
   - lane CTA tuning (`long_attention_active_cta_limit`) and the split policy.
3. **Lossless path**, only if zero local memory is required: fuse the 12-bit block decode into the FI decode producer. Measure
   decode cost, which is assumed to be 10 SM\*ms/GB.
4. **FP8 accuracy** with real weights, if FP8 stays on the table.
5. **Before claiming novelty**, read DirectKV (OSDI'26), Harvest, AQUA and the CloudMatrix384 caching sections in full.

## Known gaps
- The ring's "rest" is emulated in microbenchmarks. No real-model ring exists yet.
- The combination model assumes perfect SM packing and constant gaps.
- Contexts other than 32k are not tested.
- The fast lane does not run concurrently in most experiments.
- The lossless ratios come from 2 ShareGPT sequences only, but the ~1.5x bound does not depend on the data.
