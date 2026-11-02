# Profiling Patch Archive

This records a set of **profiling / TFLOPS-instrumentation** modifications to
sglang and parks the modified files for reference. It is observability-only:
**no compute kernel and no scheduling logic is changed** — the patch only adds
per-batch TFLOPS + forward-timing measurement to the logs.

## Where things actually live (verified state)

- **Patch source of truth:** `/home/hongrui/Agent_Bench/backend/sglang/patch/`
  (applied to a running container via singularity bind-mount at runtime).
- **conda env `site-packages/sglang`:** the active `.py` files are the **clean,
  un-patched upstream** (newer sglang version — verified: 0 patch markers). The
  patch is **not baked in** there.
- **The `*.py.bak` files in site-packages are NOT this patch's originals.** They
  are backups from a later **sglang version upgrade** (older version). Restoring
  them would *downgrade* those files and break imports against the newer modules
  (e.g. `srt/observability/`). So the originals were **left as-is** (already
  un-patched); nothing was reverted from `.bak`.

## Parked files

Full modified-file content is archived under
`python/sglang/srt/profiling_patch/<path>.py.patch` (suffix `.patch` on the
original `.py` name). 8 files carry real changes:

| file | net change | what it adds |
| --- | --- | --- |
| `managers/scheduler_metrics_mixin.py` | ~160 | FLOPs model + TFLOPS logging (core) |
| `managers/scheduler_output_processor_mixin.py` | ~76 | CPU↔GPU wall-clock calibration |
| `managers/scheduler.py` | ~24 | CUDA events around the forward |
| `managers/utils.py` | ~9 | timing field plumbing |
| `entrypoints/openai/serving_chat.py` | ~43 | request-time stamps |
| `entrypoints/openai/serving_base.py` | ~47 | `received_time` via monotonic clock |
| `entrypoints/openai/serving_completions.py` | ~7 | request-time stamps |
| `entrypoints/openai/protocol.py` | ~3 | timing fields |

> The manifest at `Agent_Bench/backend/sglang/patch/SOURCE_MANIFEST.md` also
> lists `scheduler_runtime_checker_mixin.py`, `io_struct.py`, `metrics/collector.py`,
> and `mem_cache/hiradix_cache.py`, but those have **0 net change** vs the
> pre-patch original (their large `active`-vs-`.bak` diffs are entirely the
> version upgrade, not this patch), so they are **not** archived here.

## What the patch does

1. **Forward timing** (`scheduler.py`): wrap each model forward in
   `torch.cuda.Event(enable_timing=True)` start/end events and attach them to the
   batch result — pure GPU forward time, no extra sync.

2. **FLOPs model + per-batch TFLOPS** (`scheduler_metrics_mixin.py`, the bulk):
   - `_init_flops_factors()` precomputes per-token linear FLOPs and an attention
     FLOPs coefficient from the model config — **MoE-aware** (experts-per-tok,
     router matmul, dense-vs-sparse layers, Qwen3-style `mlp_only_layers` /
     `decoder_sparse_step`).
   - `accumulate_decode_flops()` / `flush_pending_decode_stats()` accumulate
     decode FLOPs + forward time per iter (handles speculative decoding's
     multiple draft tokens) and emit a windowed `TFLOPS / forward_ms / start_ts`
     log line.
   - Prefill attention FLOPs use `attn_qkv_product = Σ extend_len·kv_len`
     (auto-correct under prefix-cache hits).

3. **CPU↔GPU wall-clock calibration** (`scheduler_output_processor_mixin.py`): a
   single cold-start anchor event (`_ANCHOR_EVENT` / `_ANCHOR_WALL`); every later
   batch's wall-clock is derived as `anchor_wall + anchor_event.elapsed_time(evt)`
   so overlap-scheduling CPU-batching skew and long GPU-idle gaps don't corrupt
   the timeline.

4. **OpenAI serving path** (`serving_*.py`, `protocol.py`): `monotonic_time()`
   request timestamps (`received_time`, etc.).

## Re-applying

These are full files from the patch's base version. To re-apply onto a different
sglang version, treat them as a reference and port the additions (the changes
are localized to the functions named above), rather than overwriting — the
base differs from current upstream.
