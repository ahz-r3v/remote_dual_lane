# Remote-KV slow lane: reproduction package (r001, 2026-11-02)

The design runs two decode batches concurrently on one GPU, inside one SGLang process:
- **Fast lane** (local-KV lane): 112 SMs, with its KV in local HBM.
- **Slow lane** (remote-KV lane): the remaining 20 SMs. Its weights and GDN state are local, but its full-attention KV pool lives
  in the HBM of an NVLink peer GPU. FA3 reads and writes that KV over NVLink through CUDA peer access.

The peer GPU (the "KV host") keeps running its own normal decode. The experiment measures three things:
1. The slow lane's step time with remote KV, compared with local KV.
2. How much the fast lane slows down beside the slow lane.
3. How much the KV host slows down while it serves the remote reads.

Reference run: ASPIRE 2B, 2 x H200 (NVLink), Qwen3.6-35B-A3B-FP8, PBS job 152114, 2026-10-17.

## Contents
| Path | What |
|---|---|
| `sglang_ash_r001/` | Modified SGLang Python tree, used in place of a pip SGLang. Base: Highway_kernel fork of SGLang at `758ca663` (release/v0.5.13 line, includes the decode-lane mux = PDMux-style green-context lanes). ASH changes make every SM-count-dependent kernel (FA3 split and grid, deep_gemm, cuBLAS) size itself for the lane's SMs. See `ASH_VERSION.md` and `ASH_CHANGES.patch`. `long_green_ctx_sm_count: -1` = slow lane takes all remaining SMs. |
| `bench/dual_lane_bench_r005.py` | Main harness. One `ModelRunner`, so the weights are loaded once. Lane 0 = fast, lane 1 = slow, each with its own pools, attention backend, CUDA graphs and decode thread. **Remote KV** happens when `ASH_REMOTE_KV=1`: `MHATokenToKVPool._create_buffers` is wrapped so that the slow pool's KV tensors are allocated on `cuda:1` (the peer) and aliased into `cuda:0` with `torch._C._construct_storage_from_data_pointer`. The slow lane's FA3 then reads and writes them over NVLink. |
| `bench/dual_bench.py`, `ash_bench.py` | KV-host ("owner") process: plain SGLang `bench_one_batch` decode with step timestamps and start/decode barriers. `sm_partition.py` and `green_ctx.py` are only imported by its unused fast/slow roles. |
| `bench/lanes_112_rest.yaml` | Lane config used: 112 SMs + the rest (20 on H200). `lanes_112_16.yaml` and `lanes_96_32.yaml` are older splits. |
| `run_impact.sh` | The whole experiment, without PBS. A port of `reference_results/run_impact_112_20_original.pbs`. |
| `summarize.py` | Median decode step per lane/GPU from `steps_*.txt`. |
| `env_template.sh` | Environment: package versions and the PYTHONPATH order. |
| `reference_results/` | The reference run: `progress.log`, `env.txt` (hashes and GPUs), each mode's result lines, and raw step timestamps (`steps/`). |

## Requirements
- Two GPUs connected by NVLink, with peer access working (`nvidia-smi topo -m` shows NV*). The reference run used H200 with 132 SMs; on another GPU the "remaining SMs" will differ.
- Python 3.12 and CUDA 13.0, with these packages:
  - torch 2.11.0+cu130
  - triton 3.6.0
  - sglang-kernel 0.4.3 (needs `sgl_kernel.spatial` and FA3)
  - flashinfer-python 0.6.12
  - sgl-deep-gemm 0.1.2
  - transformers 5.8.1
- The SGLang Python package must be the bundled tree, first on PYTHONPATH. Do not use a pip-installed `sglang`.
- Model: HF `Qwen/Qwen3.6-35B-A3B-FP8`, used offline (`HF_HUB_OFFLINE=1`). Peak use is about 0.75 of HBM on the compute GPU and about 0.7 on the KV host, plus the slow lane's KV (16 x 34k tokens) on the KV host.

## Run
```
cp env_template.sh env.sh   # set PYTHON (venv interpreter) and optionally NSYS
source env.sh
MODEL=/path/Qwen3.6-35B-A3B-FP8 GPUS=0,1 OUT=./runs/try1 bash run_impact.sh
# a subset: ... bash run_impact.sh check slow_remote both_remote
```
In `GPUS`, the first id is the KV host (G0) and the second is the compute GPU (G1). The full run takes about 35 min on H200.

The steps run in this order:

| Step | What it does |
|---|---|
| `check` | FA3 lane-split GPU unit check. Every line must say `ALL_OK`. |
| `v2` | Slow lane alone, KV local (the control). |
| `full` | Fast load on the whole GPU (baseline). |
| `owner` | KV host alone. |
| `fast` | Fast lane alone. |
| `slow_remote` | Slow lane alone, KV remote. |
| `both_remote` | Both lanes with the slow KV remote, while the owner decodes on G0. |
| `both_local` | Both lanes with the slow KV local. |

## Sanity checks in `lanes.log` / `progress.log`
- `sm_check`: should read fast 112, slow 20, overlap 0. This is the `%smid` set of each lane stream.
- `slow_pools`: `remote: true` in the `*_remote` modes, plus a `remote_kv_GB` line, which is the KV allocated on the peer.
- `logits_{fast,slow}_{0,50}.pt`: the first 4 requests' logits. Compare remote against local to check that FA3 over NVLink is numerically identical (the reference run gave max diff 0).

## Expected numbers (reference run 152114)
Medians of wall-clock decode steps, in ms, from `python3 summarize.py reference_results/steps`. In combined runs the median is
taken inside the window where both lanes are decoding.

| Mode | Fast (112 SMs) | Slow (20 SMs) | KV host (G0) |
|---|---|---|---|
| Fast load on the full GPU (132 SMs) | 18.71 | - | - |
| Alone | 20.31 | 34.91 (local KV) / 52.44 (remote KV) | 16.58 |
| Both lanes, slow KV remote | 21.62 | 60.17 | 16.90 |
| Both lanes, slow KV local | 22.20 | 46.08 | - |

The ASH notes quote per-kernel figures from the nsys traces of the same run, using the median kernel span per step. For example:
- fast with remote slow = 20.07 ms (+6.4%);
- owner 16.53 -> 16.81 ms (+1.7%);
- slow remote 51.05 ms alone / 51.91 ms with fast.

Those exclude host gaps, so they differ slightly from the wall-clock medians above. Rerun with `NSYS=...` to get the traces.

## Known limits (prototype)
- Prefill is eager and sequential on the full GPU. The slow batch is prefilled straight into the remote pool. In the design, slow
  prefill would run on the KV-holding GPU.
- Memory is budgeted per process, by `--mem-fraction-static`. The owner process and the KV pool aliased from the lanes process share
  G0's HBM without a shared budget.
- Requests are synthetic, with random token ids and a fixed batch (bench_one_batch style). This is not a serving loop, and there is
  no scheduling.
