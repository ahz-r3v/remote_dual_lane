"""
Benchmark short/long decode-lane CUDA graph replay with MoE CTA caps.

This is intentionally scheduler-free: it builds one short decode batch and one
long decode batch, then replays the lane-owned CUDA graphs directly.

Example:
PYTHONPATH=python CUDA_VISIBLE_DEVICES=0 python -m \
  benchmark.kernels.fused_moe_triton.bench_decode_lane_moe_cta \
  --model-path /mnt/vdb/hf_home/hub/models--Qwen--Qwen3-30B-A3B/snapshots/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  --load-format dummy --moe-runner-backend triton --attention-backend triton \
  --short-bs 1 --long-bs 1 --short-input-len 16 --long-input-len 8192 \
  --iters 20 --warmups 5
"""

from __future__ import annotations

import argparse
from array import array
import csv
import dataclasses
import gc
import hashlib
import json
import logging
import multiprocessing
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist

from sglang.bench_one_batch import (
    _maybe_prepare_mlp_sync_batch,
    extend,
    load_model,
    prepare_synthetic_inputs_for_latency_test,
)
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.distributed.parallel_state import destroy_distributed_environment
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.model_executor.conditional_cuda_graph_runner import (
    Qwen3MoeConditionalDecodeRunner,
    branch_switch_index,
)
from sglang.srt.model_executor.cuda_graph_runner import CudaGraphRunner
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.observability.model_perf_estimator import (
    ForwardPerfRecord,
    Qwen35MoePerfEstimator,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import (
    configure_logger,
    kill_process_tree,
    maybe_reindex_device_id,
)
from sglang.srt.utils.device_timer import DeviceTimer


@dataclasses.dataclass
class DecodeLaneBenchArgs:
    short_bs: int = 1
    long_bs: int = 1
    short_input_len: int = 16
    long_input_len: int = 8192
    warmups: int = 5
    iters: int = 20
    long_attention_active_cta_ratio: float = 0.0
    long_attention_active_cta_limit: int = 0
    long_attention_active_cta_per_sm: int = 2
    long_moe_active_cta_ratio: float = 0.8
    long_moe_active_cta_limit: int = 0
    long_linear_attn_active_cta_limit: int = 0
    short_low_resource_linear: bool = False
    long_low_resource_linear: bool = False
    enable_decode_lane_green_ctx: bool = False
    short_green_ctx_sm_count: int = 0
    long_green_ctx_sm_count: int = 0
    result_prefix: str = "benchmark/results/moe_decode_lane_cta"
    make_png: bool = True
    profile_mode: str = "none"
    profile_target: str = "long_full_concurrent"
    profile_output_prefix: Optional[str] = None
    conditional_chunks: int = 0
    run_modes: str = "all"
    concurrent_long_variant: str = "cap"
    conditional_force_cap: bool = False
    validate_tp_branch_consistency: bool = False
    prefill_chunk_size: int = 0
    prefill_token_chunk_size: int = 0
    decode_steps: int = 1
    report_bw_breakdown: bool = False
    swebench_trace_dir: Optional[str] = None
    token_corpus_dir: Optional[str] = None
    token_corpus_split: str = "train"
    short_input_lens: Optional[list[int]] = None

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        parser.add_argument(
            "--short-bs", type=int, default=DecodeLaneBenchArgs.short_bs
        )
        parser.add_argument("--long-bs", type=int, default=DecodeLaneBenchArgs.long_bs)
        parser.add_argument(
            "--short-input-len",
            type=int,
            default=DecodeLaneBenchArgs.short_input_len,
        )
        parser.add_argument(
            "--long-input-len",
            type=int,
            default=DecodeLaneBenchArgs.long_input_len,
        )
        parser.add_argument(
            "--bench-warmups", type=int, default=DecodeLaneBenchArgs.warmups
        )
        parser.add_argument(
            "--bench-iters", type=int, default=DecodeLaneBenchArgs.iters
        )
        parser.add_argument(
            "--long-attention-active-cta-ratio",
            type=float,
            default=DecodeLaneBenchArgs.long_attention_active_cta_ratio,
        )
        parser.add_argument(
            "--long-attention-active-cta-limit",
            type=int,
            default=DecodeLaneBenchArgs.long_attention_active_cta_limit,
        )
        parser.add_argument(
            "--long-attention-active-cta-per-sm",
            type=int,
            default=DecodeLaneBenchArgs.long_attention_active_cta_per_sm,
        )
        parser.add_argument(
            "--long-moe-active-cta-ratio",
            type=float,
            default=DecodeLaneBenchArgs.long_moe_active_cta_ratio,
        )
        parser.add_argument(
            "--long-moe-active-cta-limit",
            type=int,
            default=DecodeLaneBenchArgs.long_moe_active_cta_limit,
        )
        parser.add_argument(
            "--long-linear-attn-active-cta-limit",
            type=int,
            default=DecodeLaneBenchArgs.long_linear_attn_active_cta_limit,
        )
        parser.add_argument(
            "--short-low-resource-linear",
            action=argparse.BooleanOptionalAction,
            default=DecodeLaneBenchArgs.short_low_resource_linear,
            help="Use the experimental low-resource linear kernel in the short lane graph.",
        )
        parser.add_argument(
            "--long-low-resource-linear",
            action=argparse.BooleanOptionalAction,
            default=DecodeLaneBenchArgs.long_low_resource_linear,
            help="Use the experimental low-resource linear kernel in the long lane graph.",
        )
        parser.add_argument(
            "--enable-decode-lane-green-ctx",
            action=argparse.BooleanOptionalAction,
            default=DecodeLaneBenchArgs.enable_decode_lane_green_ctx,
            help=(
                "Use CUDA Green Context streams for the short/long decode lanes. "
                "If SM counts are omitted, a balanced architecture-aligned split "
                "is used."
            ),
        )
        parser.add_argument(
            "--short-green-ctx-sm-count",
            type=int,
            default=DecodeLaneBenchArgs.short_green_ctx_sm_count,
            help="SM count for the short-lane green context; 0 means auto.",
        )
        parser.add_argument(
            "--long-green-ctx-sm-count",
            type=int,
            default=DecodeLaneBenchArgs.long_green_ctx_sm_count,
            help="SM count for the long-lane green context; 0 means auto.",
        )
        parser.add_argument(
            "--result-prefix", type=str, default=DecodeLaneBenchArgs.result_prefix
        )
        parser.add_argument("--no-png", action="store_true")
        parser.add_argument(
            "--profile-mode",
            choices=["none", "torch"],
            default=DecodeLaneBenchArgs.profile_mode,
        )
        parser.add_argument(
            "--profile-target",
            choices=[
                "long_cap_concurrent",
                "long_full_concurrent",
                "long_full_after_short",
            ],
            default=DecodeLaneBenchArgs.profile_target,
        )
        parser.add_argument("--profile-output-prefix", type=str, default=None)
        parser.add_argument(
            "--conditional-chunks",
            type=int,
            default=DecodeLaneBenchArgs.conditional_chunks,
            help="Debug only: limit long conditional graph chunks; 0 means all layers plus tail.",
        )
        parser.add_argument(
            "--run-modes",
            choices=[
                "all",
                "conditional_only",
                "baseline_single_only",
                "baseline_short_only",
                "concurrent_only",
                "baseline_cap_only",
                "baseline_cap_after_short",
            ],
            default=DecodeLaneBenchArgs.run_modes,
            help="Run the full comparison, only baseline+long-cap, or only the conditional switch graph.",
        )
        parser.add_argument(
            "--concurrent-long-variant",
            choices=["cap", "full"],
            default=DecodeLaneBenchArgs.concurrent_long_variant,
            help=(
                "Long-lane graph variant used by run modes that execute one "
                "concurrent mode. 'cap' maps to long_cap_concurrent, 'full' "
                "maps to long_full_concurrent."
            ),
        )
        parser.add_argument(
            "--conditional-force-cap",
            action="store_true",
            help="Capture the conditional runner so long uses cap CTA in both branches.",
        )
        parser.add_argument(
            "--validate-tp-branch-consistency",
            action="store_true",
            help="After replay, all-gather conditional branch traces across TP ranks and fail on divergence.",
        )
        parser.add_argument(
            "--prefill-chunk-size",
            type=int,
            default=DecodeLaneBenchArgs.prefill_chunk_size,
            help=(
                "If > 0, prepare synthetic prompt state through multiple smaller "
                "extend batches and merge them before decode timing. Use 1 to "
                "prefill one request at a time. Prefill time is not measured."
            ),
        )
        parser.add_argument(
            "--prefill-token-chunk-size",
            type=int,
            default=DecodeLaneBenchArgs.prefill_token_chunk_size,
            help="Maximum tokens per extend when prefilling each request; 0 disables token chunking.",
        )
        parser.add_argument(
            "--decode-steps",
            type=int,
            default=DecodeLaneBenchArgs.decode_steps,
            help="Number of consecutive decode graph replays to time for both short and long batches.",
        )
        parser.add_argument(
            "--report-bw-breakdown",
            action="store_true",
            help="Report total, KV-cache, and MoE bandwidth for baseline runs.",
        )
        parser.add_argument(
            "--swebench-trace-dir",
            type=str,
            default=None,
            help="Build synthetic-batch token IDs from real SWE-bench trajectories.",
        )
        parser.add_argument(
            "--token-corpus-dir",
            type=str,
            default=None,
            help="Use a prepared Highway DeepResearch token corpus.",
        )
        parser.add_argument(
            "--token-corpus-split",
            choices=("train", "holdout"),
            default=DecodeLaneBenchArgs.token_corpus_split,
        )
        parser.add_argument(
            "--short-input-lens",
            type=int,
            nargs="+",
            default=None,
            help=(
                "Per-request context lengths for a heterogeneous short-only "
                "batch. The number of values must equal --short-bs."
            ),
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        return cls(
            short_bs=args.short_bs,
            long_bs=args.long_bs,
            short_input_len=args.short_input_len,
            long_input_len=args.long_input_len,
            warmups=args.bench_warmups,
            iters=args.bench_iters,
            long_attention_active_cta_ratio=args.long_attention_active_cta_ratio,
            long_attention_active_cta_limit=args.long_attention_active_cta_limit,
            long_attention_active_cta_per_sm=args.long_attention_active_cta_per_sm,
            long_moe_active_cta_ratio=args.long_moe_active_cta_ratio,
            long_moe_active_cta_limit=args.long_moe_active_cta_limit,
            long_linear_attn_active_cta_limit=args.long_linear_attn_active_cta_limit,
            short_low_resource_linear=args.short_low_resource_linear,
            long_low_resource_linear=args.long_low_resource_linear,
            enable_decode_lane_green_ctx=args.enable_decode_lane_green_ctx,
            short_green_ctx_sm_count=args.short_green_ctx_sm_count,
            long_green_ctx_sm_count=args.long_green_ctx_sm_count,
            result_prefix=args.result_prefix,
            make_png=not args.no_png,
            profile_mode=args.profile_mode,
            profile_target=args.profile_target,
            profile_output_prefix=args.profile_output_prefix,
            conditional_chunks=args.conditional_chunks,
            run_modes=args.run_modes,
            concurrent_long_variant=args.concurrent_long_variant,
            conditional_force_cap=args.conditional_force_cap,
            validate_tp_branch_consistency=args.validate_tp_branch_consistency,
            prefill_chunk_size=args.prefill_chunk_size,
            prefill_token_chunk_size=args.prefill_token_chunk_size,
            decode_steps=args.decode_steps,
            report_bw_breakdown=args.report_bw_breakdown,
            swebench_trace_dir=args.swebench_trace_dir,
            token_corpus_dir=args.token_corpus_dir,
            token_corpus_split=args.token_corpus_split,
            short_input_lens=args.short_input_lens,
        )


class _PerfBreakdownCollector:
    def __init__(self) -> None:
        self.estimates = []

    def report(self, **kwargs) -> None:
        perf_record: Optional[ForwardPerfRecord] = kwargs.get("perf_record")
        if perf_record is not None and perf_record.mode == "decode":
            self.estimates.append(perf_record.resolved_estimate)

    def flush(self, timer: DeviceTimer) -> None:
        torch.cuda.synchronize()
        timer._report()

    def add_to_summary(self, summary: dict, start: int) -> None:
        estimates = self.estimates[start:]
        if not estimates:
            raise RuntimeError("No completed decode perf records for BW breakdown")
        seconds = summary["total_ms_per_step_median"] / 1e3
        fields = {
            "read_bw_gb_s": "read_bytes",
            "write_bw_gb_s": "write_bytes",
            "kv_cache_read_bw_gb_s": "kv_cache_read_bytes",
            "kv_cache_write_bw_gb_s": "kv_cache_write_bytes",
            "moe_read_bw_gb_s": "moe_read_bytes",
        }
        for output_name, estimate_field in fields.items():
            mean_bytes = statistics.mean(
                getattr(estimate, estimate_field) for estimate in estimates
            )
            summary[output_name] = mean_bytes / seconds / 1e9
        summary["perf_record_count"] = len(estimates)


def _swebench_token_ids(
    trace_dir: str, tokenizer, batch_size: int, input_len: int, seed: int
) -> list[list[int]]:
    paths = sorted(Path(trace_dir).glob("agents/*/*.traj.json"))
    if not paths:
        raise ValueError(f"No SWE-bench trajectories found under {trace_dir}")

    candidates = []
    for path in paths:
        messages = json.loads(path.read_text()).get("messages") or []
        chars = 0
        for index, message in enumerate(messages):
            if message.get("role") == "assistant" and chars >= input_len * 2:
                candidates.append((path, index))
            chars += len(json.dumps(message, ensure_ascii=False))

    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    token_batches = []
    sources = []
    for path, message_index in candidates:
        messages = (json.loads(path.read_text()).get("messages") or [])[:message_index]
        try:
            token_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True
            )
        except Exception:
            text = "\n\n".join(
                f"<{message.get('role', 'unknown')}>\n"
                f"{message.get('content') or ''}\n"
                f"{json.dumps(message.get('tool_calls') or [], ensure_ascii=False)}"
                for message in messages
            )
            token_ids = tokenizer.encode(text, add_special_tokens=True)
        if len(token_ids) < input_len:
            continue
        token_batches.append(list(token_ids[-input_len:]))
        sources.append(f"{path.parent.name}:messages[0:{message_index}]")
        if len(token_batches) == batch_size:
            break

    if len(token_batches) != batch_size:
        raise ValueError(
            f"Only found {len(token_batches)} SWE-bench contexts with at least "
            f"{input_len} tokens; need {batch_size}"
        )
    print(
        f"SWE-bench inputs: requests={batch_size} tokens_per_request={input_len} "
        f"unique_trajectories={len({source.split(':', 1)[0] for source in sources})}"
    )
    print(f"SWE-bench first source: {sources[0]}")
    return token_batches


def _deepresearch_token_ids(
    corpus_dir: str, split: str, batch_size: int, input_len: int, seed: int
) -> list[list[int]]:
    repo_root = next(
        (
            parent
            for parent in Path(__file__).resolve().parents
            if (parent / "scripts/analysis/deepresearch_token_corpus.py").is_file()
        ),
        None,
    )
    if repo_root is None:
        raise FileNotFoundError("Cannot locate Highway token-corpus loader")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.analysis.deepresearch_token_corpus import load_token_batch

    batches, selected = load_token_batch(
        corpus_dir,
        split=split,
        batch_size=batch_size,
        input_len=input_len,
        seed=seed,
    )
    print(
        "DeepResearch inputs: "
        f"split={split} requests={batch_size} length={input_len} "
        f"first={selected[0]['task_id']}/{selected[0]['session_id']}"
    )
    return batches


def _swebench_variable_token_ids(
    trace_dir: str, tokenizer, input_lens: list[int], seed: int
) -> list[list[int]]:
    """Build a heterogeneous batch from distinct SWE-bench message prefixes."""
    paths = sorted(Path(trace_dir).glob("agents/*/*.traj.json"))
    if not paths:
        raise ValueError(f"No SWE-bench trajectories found under {trace_dir}")

    minimum_len = min(input_lens)
    candidates = []
    for path in paths:
        messages = json.loads(path.read_text()).get("messages") or []
        chars = 0
        for index, message in enumerate(messages):
            if message.get("role") == "assistant" and chars >= minimum_len * 2:
                candidates.append((path, index, chars))
            chars += len(json.dumps(message, ensure_ascii=False))

    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)
    token_cache: dict[tuple[Path, int], list[int]] = {}
    selected: list[Optional[list[int]]] = [None] * len(input_lens)
    used: set[tuple[Path, int]] = set()
    for output_index, input_len in sorted(
        enumerate(input_lens), key=lambda item: item[1], reverse=True
    ):
        for path, message_index, chars in candidates:
            key = (path, message_index)
            if key in used or chars < input_len * 2:
                continue
            if key not in token_cache:
                messages = (json.loads(path.read_text()).get("messages") or [])[
                    :message_index
                ]
                try:
                    token_ids = tokenizer.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True
                    )
                except Exception:
                    text = "\n\n".join(
                        f"<{message.get('role', 'unknown')}>\n"
                        f"{message.get('content') or ''}\n"
                        f"{json.dumps(message.get('tool_calls') or [], ensure_ascii=False)}"
                        for message in messages
                    )
                    token_ids = tokenizer.encode(text, add_special_tokens=True)
                token_cache[key] = list(token_ids)
            token_ids = token_cache[key]
            if len(token_ids) < input_len:
                continue
            selected[output_index] = token_ids[-input_len:]
            used.add(key)
            break
        if selected[output_index] is None:
            raise ValueError(
                f"No unused SWE-bench context with at least {input_len} tokens"
            )

    print(
        "SWE-bench heterogeneous inputs: "
        f"requests={len(input_lens)} min={min(input_lens)} "
        f"max={max(input_lens)} total={sum(input_lens)}"
    )
    return [tokens for tokens in selected if tokens is not None]


def _write_mux_config(path: Path, bench_args: DecodeLaneBenchArgs) -> None:
    path.write_text(
        "\n".join(
            [
                "lane_num: 2",
                "policy: short_long_by_seq_len",
                "short_context_threshold: 4096",
                "sm_group_num: 8",
                "manual_divisions: []",
                f"long_attention_active_cta_ratio: {bench_args.long_attention_active_cta_ratio}",
                f"long_attention_active_cta_limit: {bench_args.long_attention_active_cta_limit}",
                f"long_attention_active_cta_per_sm: {bench_args.long_attention_active_cta_per_sm}",
                f"long_moe_active_cta_ratio: {bench_args.long_moe_active_cta_ratio}",
                f"long_moe_active_cta_limit: {bench_args.long_moe_active_cta_limit}",
                f"long_linear_attn_active_cta_limit: {bench_args.long_linear_attn_active_cta_limit}",
                f"short_low_resource_linear: {bench_args.short_low_resource_linear}",
                f"long_low_resource_linear: {bench_args.long_low_resource_linear}",
                f"enable_green_ctx: {bench_args.enable_decode_lane_green_ctx}",
                f"short_green_ctx_sm_count: {bench_args.short_green_ctx_sm_count}",
                f"long_green_ctx_sm_count: {bench_args.long_green_ctx_sm_count}",
                "",
            ]
        )
    )


def _prepare_decode_forward_batch(next_token_ids, batch, model_runner) -> ForwardBatch:
    debug_prefill = os.environ.get("SGLANG_DECODE_LANE_DEBUG_PREFILL", "0") == "1"
    debug_prefill_sync = (
        os.environ.get("SGLANG_DECODE_LANE_DEBUG_PREFILL_SYNC", "0") == "1"
    )

    def debug_print(message: str) -> None:
        if debug_prefill:
            try:
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                peak = torch.cuda.max_memory_allocated() / (1024**3)
                mem = (
                    f" allocated={allocated:.2f}GB"
                    f" reserved={reserved:.2f}GB"
                    f" peak={peak:.2f}GB"
                )
            except Exception:
                mem = ""
            print(
                f"[decode-batch-debug t={time.perf_counter():.6f}] {message}{mem}",
                flush=True,
            )

    debug_print(
        f"prepare begin bs={len(batch.reqs)} next_shape={tuple(next_token_ids.shape)}"
    )
    batch.input_ids = next_token_ids.to(torch.int64)
    debug_print("input_ids assigned")
    batch.prepare_for_decode()
    debug_print("prepare_for_decode returned")
    _maybe_prepare_mlp_sync_batch(batch, model_runner)
    debug_print("maybe_prepare_mlp_sync_batch returned")
    fb = ForwardBatch.init_new(batch, model_runner)
    debug_print("ForwardBatch.init_new returned")
    return fb


def _request_token_digest(reqs) -> str:
    digest = hashlib.sha256()
    for req in reqs:
        digest.update(memoryview(req.origin_input_ids))
    return digest.hexdigest()


def _extend_prefill_in_chunks(
    reqs,
    model_runner,
    chunk_size: int,
    rank_print=print,
    label: str = "prefill",
    token_chunk_size: int = 0,
):
    debug_prefill = os.environ.get("SGLANG_DECODE_LANE_DEBUG_PREFILL", "0") == "1"
    debug_prefill_sync = (
        os.environ.get("SGLANG_DECODE_LANE_DEBUG_PREFILL_SYNC", "0") == "1"
    )
    local_rank = (
        dist.get_rank()
        if dist.is_initialized()
        else int(os.environ.get("LOCAL_RANK", "-1"))
    )

    def debug_print(message: str) -> None:
        if debug_prefill:
            try:
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                peak = torch.cuda.max_memory_allocated() / (1024**3)
                mem = (
                    f" allocated={allocated:.2f}GB"
                    f" reserved={reserved:.2f}GB"
                    f" peak={peak:.2f}GB"
                )
            except Exception:
                mem = ""
            print(
                f"[prefill-debug rank={local_rank} t={time.perf_counter():.6f}] "
                f"{message}{mem}",
                flush=True,
            )

    def extend_one_req(req, req_label):
        total_len = len(req.full_untruncated_fill_ids)
        if token_chunk_size <= 0 or total_len <= token_chunk_size:
            return extend([req], model_runner)

        full_input_ids = array("q", req.full_untruncated_fill_ids)
        num_token_chunks = (total_len + token_chunk_size - 1) // token_chunk_size
        ret = None
        for token_chunk_id, end in enumerate(
            range(token_chunk_size, total_len + token_chunk_size, token_chunk_size)
        ):
            end = min(end, total_len)
            if token_chunk_id > 0:
                prefix_len = min(token_chunk_id * token_chunk_size, total_len)
                req.prefix_indices = model_runner.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :prefix_len
                ].to(req.prefix_indices.dtype)
            req.origin_input_ids = array("q", full_input_ids[:end])
            req.full_untruncated_fill_ids = array("q", full_input_ids[:end])
            req.fill_len = end
            req.logprob_start_len = -1
            req.set_extend_input_len(end - len(req.prefix_indices))
            rank_print(
                f"    {req_label} token chunk {token_chunk_id + 1}/"
                f"{num_token_chunks} (tokens={len(req.prefix_indices)}:{end})"
            )
            ret = extend([req], model_runner)

        return ret

    if token_chunk_size > 0:
        merged_batch = None
        next_token_ids = []
        next_token_logits = []
        rank_print(
            f"Preparing {label} one request at a time with token chunks "
            f"(token_chunk_size={token_chunk_size}, reqs={len(reqs)}) ..."
        )
        for req_id, req in enumerate(reqs):
            chunk_next, chunk_logits, chunk_batch = extend_one_req(
                req, f"{label} req {req_id + 1}/{len(reqs)}"
            )
            next_token_ids.append(chunk_next)
            next_token_logits.append(chunk_logits)
            if merged_batch is None:
                merged_batch = chunk_batch
            else:
                merged_batch.merge_batch(chunk_batch)
        return (
            torch.cat(next_token_ids, dim=0),
            torch.cat(next_token_logits, dim=0),
            merged_batch,
        )

    if chunk_size <= 0 or len(reqs) <= chunk_size:
        debug_print(f"{label} single extend begin reqs={len(reqs)}")
        tic = time.perf_counter()
        ret = extend(reqs, model_runner)
        debug_print(
            f"{label} single extend end elapsed_ms={(time.perf_counter() - tic) * 1e3:.3f}"
        )
        return ret

    merged_batch = None
    next_token_ids = []
    next_token_logits = []
    num_chunks = (len(reqs) + chunk_size - 1) // chunk_size
    rank_print(
        f"Preparing {label} in {num_chunks} prefill chunks "
        f"(chunk_size={chunk_size}, reqs={len(reqs)}) ..."
    )
    for chunk_id, start in enumerate(range(0, len(reqs), chunk_size)):
        chunk_reqs = reqs[start : start + chunk_size]
        rank_print(
            f"  {label} prefill chunk {chunk_id + 1}/{num_chunks} "
            f"(reqs={len(chunk_reqs)})"
        )
        debug_print(
            f"{label} chunk {chunk_id + 1}/{num_chunks} begin "
            f"rid={[getattr(req, 'rid', None) for req in chunk_reqs]}"
        )
        tic = time.perf_counter()
        chunk_next, chunk_logits, chunk_batch = extend(chunk_reqs, model_runner)
        debug_print(
            f"{label} chunk {chunk_id + 1}/{num_chunks} extend returned "
            f"elapsed_ms={(time.perf_counter() - tic) * 1e3:.3f}"
        )
        if debug_prefill_sync:
            sync_tic = time.perf_counter()
            torch.cuda.synchronize()
            debug_print(
                f"{label} chunk {chunk_id + 1}/{num_chunks} cuda sync done "
                f"elapsed_ms={(time.perf_counter() - sync_tic) * 1e3:.3f}"
            )
        next_token_ids.append(chunk_next)
        next_token_logits.append(chunk_logits)
        if merged_batch is None:
            merged_batch = chunk_batch
        else:
            debug_print(f"{label} chunk {chunk_id + 1}/{num_chunks} merge begin")
            merged_batch.merge_batch(chunk_batch)
            debug_print(f"{label} chunk {chunk_id + 1}/{num_chunks} merge end")

    return (
        torch.cat(next_token_ids, dim=0),
        torch.cat(next_token_logits, dim=0),
        merged_batch,
    )


def _event_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    end.synchronize()
    return float(start.elapsed_time(end))


@torch.no_grad()
def _replay_once(
    model_runner, short_fb, long_fb, mode: str, decode_steps: int = 1
) -> dict[str, float]:
    debug_replay = os.environ.get("SGLANG_DECODE_LANE_DEBUG_PREFILL", "0") == "1"
    debug_rank = (
        dist.get_rank()
        if dist.is_initialized()
        else int(os.environ.get("LOCAL_RANK", "-1"))
    )

    def debug_print(message: str) -> None:
        if debug_replay:
            try:
                allocated = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                peak = torch.cuda.max_memory_allocated() / (1024**3)
                mem = (
                    f" allocated={allocated:.2f}GB"
                    f" reserved={reserved:.2f}GB"
                    f" peak={peak:.2f}GB"
                )
            except Exception:
                mem = ""
            print(
                f"[replay-debug rank={debug_rank} t={time.perf_counter():.6f}] {message}{mem}",
                flush=True,
            )

    if decode_steps <= 0:
        raise ValueError(f"decode_steps must be positive, got {decode_steps}")
    short_lane = model_runner.decode_lane_runtimes[0]
    long_lane = model_runner.decode_lane_runtimes[1]

    anchor = torch.cuda.Stream()
    start = torch.cuda.Event(enable_timing=True)
    short_start = torch.cuda.Event(enable_timing=True)
    short_end = torch.cuda.Event(enable_timing=True)
    long_start = torch.cuda.Event(enable_timing=True)
    long_end = torch.cuda.Event(enable_timing=True)

    debug_print(f"{mode} begin decode_steps={decode_steps}")
    torch.cuda.synchronize()
    debug_print(f"{mode} initial cuda synchronize returned")
    wall_tic = time.perf_counter()
    with torch.cuda.stream(anchor):
        start.record(anchor)
    debug_print(f"{mode} anchor start recorded")

    with torch.cuda.stream(short_lane.stream):
        short_lane.stream.wait_event(start)
        short_start.record(short_lane.stream)
        for step in range(decode_steps):
            if step == 0 or step == decode_steps - 1 or step % 10 == 0:
                debug_print(f"{mode} short replay step {step} begin")
            model_runner.replay_decode_lane_cuda_graph(short_fb, short_lane)
            if step == 0 or step == decode_steps - 1 or step % 10 == 0:
                debug_print(f"{mode} short replay step {step} returned")
        short_end.record(short_lane.stream)
    debug_print(f"{mode} short enqueue done")

    if mode == "long_full_after_short":
        long_variant = "full"
        wait_for_short = True
    elif mode == "long_full_concurrent":
        long_variant = "full"
        wait_for_short = False
    elif mode == "long_cap_concurrent":
        long_variant = "cap"
        wait_for_short = False
    else:
        raise ValueError(f"unknown mode: {mode}")

    with torch.cuda.stream(long_lane.stream):
        long_lane.stream.wait_event(start)
        if wait_for_short:
            long_lane.stream.wait_event(short_end)
        long_start.record(long_lane.stream)
        for step in range(decode_steps):
            if step == 0 or step == decode_steps - 1 or step % 10 == 0 or step >= 60:
                debug_print(
                    f"{mode} long replay step {step} begin variant={long_variant}"
                )
            model_runner.replay_decode_lane_cuda_graph(
                long_fb, long_lane, graph_variant=long_variant
            )
            if step == 0 or step == decode_steps - 1 or step % 10 == 0 or step >= 60:
                debug_print(
                    f"{mode} long replay step {step} returned variant={long_variant}"
                )
        long_end.record(long_lane.stream)
    debug_print(f"{mode} long enqueue done")

    debug_print(f"{mode} event timing begin short_end")
    short_ms = _event_ms(start, short_end)
    debug_print(f"{mode} event timing short_end returned")
    long_ms = _event_ms(start, long_end)
    debug_print(f"{mode} event timing long_end returned")
    short_replay_ms = _event_ms(short_start, short_end)
    long_replay_ms = _event_ms(long_start, long_end)
    debug_print(f"{mode} final cuda synchronize begin")
    torch.cuda.synchronize()
    debug_print(f"{mode} final cuda synchronize returned")
    wall_ms = (time.perf_counter() - wall_tic) * 1e3
    return {
        "decode_steps": decode_steps,
        "short_ms": short_ms,
        "long_ms": long_ms,
        "total_ms": max(short_ms, long_ms),
        "wall_ms": wall_ms,
        "short_replay_ms": short_replay_ms,
        "long_replay_ms": long_replay_ms,
        "short_ms_per_step": short_ms / decode_steps,
        "long_ms_per_step": long_ms / decode_steps,
        "total_ms_per_step": max(short_ms, long_ms) / decode_steps,
        "short_replay_ms_per_step": short_replay_ms / decode_steps,
        "long_replay_ms_per_step": long_replay_ms / decode_steps,
    }


@torch.no_grad()
def _single_batch_once(
    model_runner, forward_batch, decode_steps: int = 1
) -> dict[str, float]:
    if decode_steps <= 0:
        raise ValueError(f"decode_steps must be positive, got {decode_steps}")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    wall_tic = time.perf_counter()
    start.record(torch.cuda.current_stream())
    for _ in range(decode_steps):
        model_runner.forward(forward_batch)
    end.record(torch.cuda.current_stream())
    batch_ms = _event_ms(start, end)
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_tic) * 1e3
    return {
        "decode_steps": decode_steps,
        "short_ms": batch_ms,
        "long_ms": batch_ms,
        "total_ms": batch_ms,
        "wall_ms": wall_ms,
        "short_replay_ms": batch_ms,
        "long_replay_ms": batch_ms,
        "short_ms_per_step": batch_ms / decode_steps,
        "long_ms_per_step": batch_ms / decode_steps,
        "total_ms_per_step": batch_ms / decode_steps,
        "short_replay_ms_per_step": batch_ms / decode_steps,
        "long_replay_ms_per_step": batch_ms / decode_steps,
    }


def _short_only_baseline_once(
    model_runner, forward_batch, decode_steps: int = 1
) -> dict[str, float]:
    row = _single_batch_once(model_runner, forward_batch, decode_steps=decode_steps)
    row.update(
        {
            "long_ms": 0.0,
            "long_replay_ms": 0.0,
            "long_ms_per_step": 0.0,
            "long_replay_ms_per_step": 0.0,
        }
    )
    return row


def _summarize(values: list[float]) -> dict[str, float]:
    values = sorted(values)
    return {
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "min": values[0],
        "max": values[-1],
        "p90": values[int(0.9 * (len(values) - 1))],
    }


def _summarize_bench_rows(
    mode: str, decode_steps: int, mode_rows: list[dict[str, float]]
) -> dict[str, float]:
    short_stats = _summarize([r["short_ms"] for r in mode_rows])
    long_stats = _summarize([r["long_ms"] for r in mode_rows])
    total_stats = _summarize([r["total_ms"] for r in mode_rows])
    wall_stats = _summarize([r["wall_ms"] for r in mode_rows])
    short_replay_stats = _summarize([r["short_replay_ms"] for r in mode_rows])
    long_replay_stats = _summarize([r["long_replay_ms"] for r in mode_rows])
    total_per_step_stats = _summarize([r["total_ms_per_step"] for r in mode_rows])
    return {
        "mode": mode,
        "decode_steps": decode_steps,
        "short_ms_median": short_stats["median"],
        "short_ms_mean": short_stats["mean"],
        "short_ms_p90": short_stats["p90"],
        "long_ms_median": long_stats["median"],
        "long_ms_mean": long_stats["mean"],
        "long_ms_p90": long_stats["p90"],
        "total_ms_median": total_stats["median"],
        "total_ms_mean": total_stats["mean"],
        "total_ms_p90": total_stats["p90"],
        "wall_ms_median": wall_stats["median"],
        "wall_ms_mean": wall_stats["mean"],
        "wall_ms_p90": wall_stats["p90"],
        "short_replay_ms_median": short_replay_stats["median"],
        "short_replay_ms_mean": short_replay_stats["mean"],
        "short_replay_ms_p90": short_replay_stats["p90"],
        "long_replay_ms_median": long_replay_stats["median"],
        "long_replay_ms_mean": long_replay_stats["mean"],
        "long_replay_ms_p90": long_replay_stats["p90"],
        "total_ms_per_step_median": total_per_step_stats["median"],
        "total_ms_per_step_mean": total_per_step_stats["mean"],
        "total_ms_per_step_p90": total_per_step_stats["p90"],
    }


def _write_results(result_prefix: str, rows: list[dict], summaries: list[dict]) -> None:
    prefix = Path(result_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = prefix.with_suffix(".csv")
    fieldnames = list(rows[0].keys())
    for row in rows[1:]:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = prefix.with_suffix(".json")
    json_path.write_text(json.dumps({"rows": rows, "summaries": summaries}, indent=2))


def _write_layer_timing_results(result_prefix: str, rows: list[dict]) -> None:
    if not rows:
        return
    prefix = Path(result_prefix)
    path = prefix.with_suffix(".layer_timing.csv")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {path}")


def _collect_long_layer_timing(model_runner, mode: str, iteration: int) -> list[dict]:
    if mode == "long_cap_concurrent":
        graph_variant = "cap"
    elif mode in ("long_full_concurrent", "long_full_after_short"):
        graph_variant = "full"
    else:
        return []

    long_lane = model_runner.decode_lane_runtimes[1]
    runner = long_lane.cuda_graph_runners.get(graph_variant)
    recorder = getattr(runner, "layer_timer_recorder", None) if runner else None
    if recorder is None:
        return []

    return [
        {
            "mode": mode,
            "iter": iteration,
            "graph_variant": graph_variant,
            **row,
        }
        for row in recorder.elapsed_rows()
    ]


def _collect_baseline_layer_timing(
    model_runner, mode: str, iteration: int
) -> list[dict]:
    runner = getattr(model_runner, "graph_runner", None)
    recorder = getattr(runner, "layer_timer_recorder", None) if runner else None
    if recorder is None:
        return []

    return [
        {
            "mode": mode,
            "iter": iteration,
            "graph_variant": "single_batch",
            **row,
        }
        for row in recorder.elapsed_rows()
    ]


@torch.no_grad()
def _conditional_once(cond_runner: Qwen3MoeConditionalDecodeRunner) -> dict:
    torch.cuda.synchronize()
    wall_tic = time.perf_counter()
    result = cond_runner.replay()
    torch.cuda.synchronize()
    wall_ms = (time.perf_counter() - wall_tic) * 1e3
    switch = branch_switch_index(result.branch)
    return {
        "short_ms": result.short_ms,
        "long_ms": result.long_ms,
        "total_ms": result.total_ms,
        "wall_ms": wall_ms,
        "short_replay_ms": result.short_ms,
        "long_replay_ms": result.long_ms,
        "cap_chunks": result.cap_chunks,
        "full_chunks": result.full_chunks,
        "switch_chunk": -1 if switch is None else switch,
        "branch": " ".join(result.branch),
        "observed_done": " ".join(str(x) for x in result.observed_done),
    }


def _maybe_write_png(result_prefix: str, summaries: list[dict]) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return None

    labels = [s["mode"] for s in summaries]
    short = [s["short_ms_median"] for s in summaries]
    long = [s["long_ms_median"] for s in summaries]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(labels))
    short_bars = ax.bar([i - 0.18 for i in x], short, width=0.36, label="short")
    long_bars = ax.bar([i + 0.18 for i in x], long, width=0.36, label="long")
    for bars in (short_bars, long_bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Median decode completion latency (ms)")
    ax.set_title("MoE decode lane CTA cap benchmark")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    png_path = Path(result_prefix).with_suffix(".png")
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    return str(png_path)


def _cuda_self_us(event) -> float:
    for attr in ("self_cuda_time_total", "self_device_time_total"):
        value = getattr(event, attr, None)
        if value is not None:
            return float(value)
    return 0.0


def _kernel_group(name: str) -> str:
    lower = name.lower()
    if "rms" in lower or "norm" in lower:
        return "norm"
    if "fused_moe" in lower or "moe" in lower:
        return "moe"
    if (
        "attention" in lower
        or "flash" in lower
        or "paged" in lower
        or "decode" in lower
        or "attn" in lower
        or "_fwd_grouped_kernel_stage" in lower
        or "_fwd_kernel_stage" in lower
        or "fused_rope" in lower
        or "qknorm" in lower
    ):
        return "attention"
    if "topk" in lower or "router" in lower or "routing" in lower or "softmax" in lower:
        return "router_topk"
    if (
        "gemm" in lower
        or "matmul" in lower
        or "::mm" in lower
        or "addmm" in lower
        or "cublas" in lower
        or "nvjet" in lower
    ):
        return "matmul"
    if "copy" in lower or "memcpy" in lower or "fill" in lower or "setitem" in lower:
        return "memory_misc"
    return "other"


def _write_torch_profile(
    result_prefix: str,
    model_runner,
    short_fb,
    long_fb,
    target_mode: str,
) -> dict:
    output_prefix = Path(result_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    trace_path = output_prefix.with_suffix(".trace.json")
    table_path = output_prefix.with_suffix(".profile.txt")
    json_path = output_prefix.with_suffix(".profile.json")

    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        metrics = _replay_once(model_runner, short_fb, long_fb, target_mode)
    torch.cuda.synchronize()

    prof.export_chrome_trace(str(trace_path))
    events = prof.key_averages()
    cuda_events = [
        event
        for event in events
        if _cuda_self_us(event) > 0 and "cuda" not in event.key.lower()
    ]
    cuda_events.sort(key=_cuda_self_us, reverse=True)
    total_cuda_us = sum(_cuda_self_us(event) for event in cuda_events)

    groups: dict[str, float] = {}
    for event in cuda_events:
        group = _kernel_group(event.key)
        groups[group] = groups.get(group, 0.0) + _cuda_self_us(event)

    group_rows = [
        {
            "group": group,
            "cuda_self_ms": us / 1e3,
            "pct_cuda_self": (100.0 * us / total_cuda_us) if total_cuda_us > 0 else 0.0,
        }
        for group, us in sorted(groups.items(), key=lambda item: item[1], reverse=True)
    ]
    top_kernels = [
        {
            "name": event.key,
            "cuda_self_ms": _cuda_self_us(event) / 1e3,
            "pct_cuda_self": (
                100.0 * _cuda_self_us(event) / total_cuda_us
                if total_cuda_us > 0
                else 0.0
            ),
            "count": int(event.count),
        }
        for event in cuda_events[:40]
    ]
    profile_summary = {
        "target_mode": target_mode,
        "metrics": metrics,
        "total_cuda_self_ms": total_cuda_us / 1e3,
        "groups": group_rows,
        "top_kernels": top_kernels,
        "trace_path": str(trace_path),
    }
    json_path.write_text(json.dumps(profile_summary, indent=2))

    with table_path.open("w") as f:
        f.write(f"target_mode: {target_mode}\n")
        f.write(f"metrics: {json.dumps(metrics)}\n")
        f.write(f"total_cuda_self_ms: {total_cuda_us / 1e3:.3f}\n\n")
        f.write("groups:\n")
        for row in group_rows:
            f.write(
                f"  {row['group']:>12s}: {row['cuda_self_ms']:9.3f} ms "
                f"({row['pct_cuda_self']:5.1f}%)\n"
            )
        f.write("\ntop kernels:\n")
        for row in top_kernels:
            f.write(
                f"  {row['cuda_self_ms']:9.3f} ms "
                f"({row['pct_cuda_self']:5.1f}%) x{row['count']:4d} "
                f"{row['name']}\n"
            )

    print(f"Wrote {table_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {trace_path}")
    return profile_summary


def _prepare_server_args_for_decode_lane_bench(
    server_args: ServerArgs, bench_args: DecodeLaneBenchArgs
) -> None:
    result_prefix = Path(bench_args.result_prefix)
    result_prefix.parent.mkdir(parents=True, exist_ok=True)
    mux_config_path = result_prefix.parent / "decode_lane_moe_cta_mux.yaml"
    _write_mux_config(mux_config_path, bench_args)

    server_args.enable_decode_lane_mux = True
    server_args.decode_lane_mux_config_path = str(mux_config_path)
    allow_decode_lane_custom_ar = (
        os.environ.get("SGLANG_DECODE_LANE_ALLOW_CUSTOM_ALL_REDUCE", "0") == "1"
    )
    if (
        server_args.tp_size > 1
        and not server_args.disable_custom_all_reduce
        and not allow_decode_lane_custom_ar
    ):
        server_args.disable_custom_all_reduce = True
        print(
            "DecodeLaneMux TP benchmark disables custom allreduce so short/long "
            "lane CUDA graphs use current-stream NCCL collectives."
        )
    elif (
        server_args.tp_size > 1
        and not server_args.disable_custom_all_reduce
        and allow_decode_lane_custom_ar
    ):
        print(
            "DecodeLaneMux TP benchmark enables experimental custom allreduce "
            "because SGLANG_DECODE_LANE_ALLOW_CUSTOM_ALL_REDUCE=1."
        )
    server_args.disable_piecewise_cuda_graph = True
    server_args.disable_cuda_graph = False
    combined_bs = bench_args.short_bs + bench_args.long_bs
    server_args.cuda_graph_max_bs = max(
        bench_args.short_bs, bench_args.long_bs, combined_bs
    )
    server_args.cuda_graph_bs = sorted(
        {bench_args.short_bs, bench_args.long_bs, combined_bs}
    )


def run_worker(
    server_args: ServerArgs,
    port_args: PortArgs,
    bench_args: DecodeLaneBenchArgs,
    gpu_id: int,
    tp_rank: int,
) -> None:
    _set_envs_and_config(server_args)
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)
    configure_logger(server_args, prefix=f" TP{tp_rank}")
    rank_print = print if tp_rank == 0 else lambda *args, **kwargs: None

    wrapped_runner, tokenizer = load_model(server_args, port_args, gpu_id, tp_rank)
    model_runner = wrapped_runner.torch_runner
    perf_breakdown = None
    perf_timer = None
    if bench_args.report_bw_breakdown:
        perf_breakdown = _PerfBreakdownCollector()
        perf_timer = DeviceTimer(reporter=perf_breakdown.report)
        model_runner.perf_estimator = Qwen35MoePerfEstimator.from_model_config(
            model_runner.model_config, server_args.tp_size
        )
        model_runner.device_timer = perf_timer
    # Model construction may consume process RNG state. Reset here so synthetic
    # token IDs are identical across independently launched benchmark jobs.
    np.random.seed(int(server_args.random_seed))

    if bench_args.run_modes == "baseline_short_only":
        rank_print("Preparing short-only baseline ...")
        rows = []
        summaries = []
        layer_timing_rows = []

        model_runner.graph_runner = CudaGraphRunner(
            model_runner,
            capture_bs_override=[bench_args.short_bs],
            layer_timer_name="baseline_short_only",
        )
        wrapped_runner.clear()
        if bench_args.swebench_trace_dir and bench_args.token_corpus_dir:
            raise ValueError(
                "--swebench-trace-dir and --token-corpus-dir are mutually exclusive"
            )
        if bench_args.short_input_lens is not None:
            if bench_args.token_corpus_dir:
                raise ValueError(
                    "--short-input-lens is not yet supported with --token-corpus-dir"
                )
            if len(bench_args.short_input_lens) != bench_args.short_bs:
                raise ValueError(
                    "--short-input-lens count must equal --short-bs: "
                    f"{len(bench_args.short_input_lens)} != {bench_args.short_bs}"
                )
            custom_inputs = _swebench_variable_token_ids(
                bench_args.swebench_trace_dir,
                tokenizer,
                bench_args.short_input_lens,
                int(server_args.random_seed),
            )
        else:
            if bench_args.token_corpus_dir:
                custom_inputs = _deepresearch_token_ids(
                    bench_args.token_corpus_dir,
                    bench_args.token_corpus_split,
                    bench_args.short_bs,
                    bench_args.short_input_len,
                    int(server_args.random_seed),
                )
            else:
                custom_inputs = (
                    _swebench_token_ids(
                        bench_args.swebench_trace_dir,
                        tokenizer,
                        bench_args.short_bs,
                        bench_args.short_input_len,
                        int(server_args.random_seed),
                    )
                    if bench_args.swebench_trace_dir
                    else None
                )
        short_only_reqs = prepare_synthetic_inputs_for_latency_test(
            bench_args.short_bs, bench_args.short_input_len, custom_inputs
        )
        for i, req in enumerate(short_only_reqs):
            req.rid = i
        short_only_next, _, short_only_batch = _extend_prefill_in_chunks(
            short_only_reqs,
            model_runner,
            bench_args.prefill_chunk_size,
            rank_print,
            "short-only baseline",
        )
        short_only_fb = _prepare_decode_forward_batch(
            short_only_next, short_only_batch, model_runner
        )

        short_only_mode = "baseline_short_only"
        rank_print(f"Warmup {short_only_mode} ...")
        for _ in range(bench_args.warmups):
            _short_only_baseline_once(
                model_runner, short_only_fb, decode_steps=bench_args.decode_steps
            )
        if perf_breakdown is not None:
            perf_breakdown.flush(perf_timer)
            perf_start = len(perf_breakdown.estimates)

        rank_print(f"Benchmark {short_only_mode} ...")
        for i in range(bench_args.iters):
            row = {
                "mode": short_only_mode,
                "iter": i,
                **_short_only_baseline_once(
                    model_runner, short_only_fb, decode_steps=bench_args.decode_steps
                ),
            }
            rows.append(row)
            layer_timing_rows.extend(
                _collect_baseline_layer_timing(model_runner, short_only_mode, i)
            )
        summary = _summarize_bench_rows(short_only_mode, bench_args.decode_steps, rows)
        if perf_breakdown is not None:
            perf_breakdown.flush(perf_timer)
            perf_breakdown.add_to_summary(summary, perf_start)
        summaries.append(summary)
        rank_print(json.dumps(summary, indent=2))

        if tp_rank == 0:
            _write_results(bench_args.result_prefix, rows, summaries)
            _write_layer_timing_results(bench_args.result_prefix, layer_timing_rows)
            png_path = (
                _maybe_write_png(bench_args.result_prefix, summaries)
                if bench_args.make_png
                else None
            )

            print(f"Wrote {Path(bench_args.result_prefix).with_suffix('.csv')}")
            print(f"Wrote {Path(bench_args.result_prefix).with_suffix('.json')}")
            if png_path:
                print(f"Wrote {png_path}")

        if server_args.tp_size > 1:
            destroy_distributed_environment()
        elif dist.is_initialized():
            dist.destroy_process_group()
        return

    rank_print("Building lane CUDA graphs ...")
    model_runner.init_decode_lane_cuda_graphs(
        [[bench_args.short_bs], [bench_args.long_bs]],
        enable_layer_timing=bench_args.run_modes
        in ("all", "baseline_cap_only", "baseline_cap_after_short"),
    )

    rows = []
    summaries = []
    layer_timing_rows = []
    combined_bs = bench_args.short_bs + bench_args.long_bs

    if bench_args.run_modes in (
        "all",
        "baseline_single_only",
        "baseline_cap_only",
        "baseline_cap_after_short",
    ):
        rank_print("Preparing single-batch baseline ...")
        model_runner.graph_runner = CudaGraphRunner(
            model_runner,
            capture_bs_override=[combined_bs],
            layer_timer_name="baseline_single_batch",
        )
        wrapped_runner.clear()
        np.random.seed(int(server_args.random_seed))
        baseline_short_reqs = prepare_synthetic_inputs_for_latency_test(
            bench_args.short_bs, bench_args.short_input_len
        )
        baseline_long_reqs = prepare_synthetic_inputs_for_latency_test(
            bench_args.long_bs, bench_args.long_input_len
        )
        rank_print(
            "Synthetic input hashes: "
            f"seed={server_args.random_seed} "
            f"short_sha256={_request_token_digest(baseline_short_reqs)} "
            f"long_sha256={_request_token_digest(baseline_long_reqs)}"
        )
        baseline_reqs = baseline_short_reqs + baseline_long_reqs
        for i, req in enumerate(baseline_reqs):
            req.rid = i
        baseline_next, _, baseline_batch = _extend_prefill_in_chunks(
            baseline_reqs,
            model_runner,
            bench_args.prefill_chunk_size,
            rank_print,
            "single-batch baseline",
            bench_args.prefill_token_chunk_size,
        )
        baseline_fb = _prepare_decode_forward_batch(
            baseline_next, baseline_batch, model_runner
        )

        baseline_mode = "baseline_single_batch"
        rank_print(f"Warmup {baseline_mode} ...")
        for _ in range(bench_args.warmups):
            _single_batch_once(
                model_runner, baseline_fb, decode_steps=bench_args.decode_steps
            )

        rank_print(f"Benchmark {baseline_mode} ...")
        baseline_rows = []
        for i in range(bench_args.iters):
            row = {
                "mode": baseline_mode,
                "iter": i,
                **_single_batch_once(
                    model_runner, baseline_fb, decode_steps=bench_args.decode_steps
                ),
            }
            baseline_rows.append(row)
            rows.append(row)
            layer_timing_rows.extend(
                _collect_baseline_layer_timing(model_runner, baseline_mode, i)
            )

        short_stats = _summarize([r["short_ms"] for r in baseline_rows])
        long_stats = _summarize([r["long_ms"] for r in baseline_rows])
        total_stats = _summarize([r["total_ms"] for r in baseline_rows])
        wall_stats = _summarize([r["wall_ms"] for r in baseline_rows])
        short_replay_stats = _summarize([r["short_replay_ms"] for r in baseline_rows])
        long_replay_stats = _summarize([r["long_replay_ms"] for r in baseline_rows])
        total_per_step_stats = _summarize(
            [r["total_ms_per_step"] for r in baseline_rows]
        )
        baseline_summary = {
            "mode": baseline_mode,
            "decode_steps": bench_args.decode_steps,
            "short_ms_median": short_stats["median"],
            "short_ms_mean": short_stats["mean"],
            "short_ms_p90": short_stats["p90"],
            "long_ms_median": long_stats["median"],
            "long_ms_mean": long_stats["mean"],
            "long_ms_p90": long_stats["p90"],
            "total_ms_median": total_stats["median"],
            "total_ms_mean": total_stats["mean"],
            "total_ms_p90": total_stats["p90"],
            "wall_ms_median": wall_stats["median"],
            "wall_ms_mean": wall_stats["mean"],
            "wall_ms_p90": wall_stats["p90"],
            "short_replay_ms_median": short_replay_stats["median"],
            "short_replay_ms_mean": short_replay_stats["mean"],
            "short_replay_ms_p90": short_replay_stats["p90"],
            "long_replay_ms_median": long_replay_stats["median"],
            "long_replay_ms_mean": long_replay_stats["mean"],
            "long_replay_ms_p90": long_replay_stats["p90"],
            "total_ms_per_step_median": total_per_step_stats["median"],
            "total_ms_per_step_mean": total_per_step_stats["mean"],
            "total_ms_per_step_p90": total_per_step_stats["p90"],
        }
        summaries.append(baseline_summary)
        rank_print(json.dumps(baseline_summary, indent=2))

        if bench_args.run_modes == "baseline_single_only":
            if tp_rank == 0:
                _write_results(bench_args.result_prefix, rows, summaries)
                _write_layer_timing_results(bench_args.result_prefix, layer_timing_rows)
                png_path = (
                    _maybe_write_png(bench_args.result_prefix, summaries)
                    if bench_args.make_png
                    else None
                )

                print(f"Wrote {Path(bench_args.result_prefix).with_suffix('.csv')}")
                print(f"Wrote {Path(bench_args.result_prefix).with_suffix('.json')}")
                if png_path:
                    print(f"Wrote {png_path}")

            if server_args.tp_size > 1:
                destroy_distributed_environment()
            elif dist.is_initialized():
                dist.destroy_process_group()
            return

        rank_print("Preparing short-only baseline ...")
        model_runner.graph_runner = None
        gc.collect()
        torch.cuda.empty_cache()
        model_runner.graph_runner = CudaGraphRunner(
            model_runner,
            capture_bs_override=[bench_args.short_bs],
            layer_timer_name="baseline_short_only",
        )
        wrapped_runner.clear()
        short_only_reqs = prepare_synthetic_inputs_for_latency_test(
            bench_args.short_bs, bench_args.short_input_len
        )
        for i, req in enumerate(short_only_reqs):
            req.rid = i
        short_only_next, _, short_only_batch = _extend_prefill_in_chunks(
            short_only_reqs,
            model_runner,
            bench_args.prefill_chunk_size,
            rank_print,
            "short-only baseline",
        )
        short_only_fb = _prepare_decode_forward_batch(
            short_only_next, short_only_batch, model_runner
        )

        short_only_mode = "baseline_short_only"
        rank_print(f"Warmup {short_only_mode} ...")
        for _ in range(bench_args.warmups):
            _short_only_baseline_once(
                model_runner, short_only_fb, decode_steps=bench_args.decode_steps
            )

        rank_print(f"Benchmark {short_only_mode} ...")
        short_only_rows = []
        for i in range(bench_args.iters):
            row = {
                "mode": short_only_mode,
                "iter": i,
                **_short_only_baseline_once(
                    model_runner, short_only_fb, decode_steps=bench_args.decode_steps
                ),
            }
            short_only_rows.append(row)
            rows.append(row)
            layer_timing_rows.extend(
                _collect_baseline_layer_timing(model_runner, short_only_mode, i)
            )
        short_only_summary = _summarize_bench_rows(
            short_only_mode, bench_args.decode_steps, short_only_rows
        )
        summaries.append(short_only_summary)
        rank_print(json.dumps(short_only_summary, indent=2))

    if os.environ.get("SGLANG_DECODE_LANE_DEBUG_PREFILL", "0") == "1":
        print(
            f"[bench-debug rank={tp_rank} t={time.perf_counter():.6f}] "
            "concurrent wrapped_runner.clear begin",
            flush=True,
        )
    wrapped_runner.clear()
    if os.environ.get("SGLANG_DECODE_LANE_DEBUG_PREFILL", "0") == "1":
        print(
            f"[bench-debug rank={tp_rank} t={time.perf_counter():.6f}] "
            "concurrent wrapped_runner.clear end",
            flush=True,
        )
    rank_print("Preparing short/long decode batches ...")
    short_reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.short_bs, bench_args.short_input_len
    )
    long_reqs = prepare_synthetic_inputs_for_latency_test(
        bench_args.long_bs, bench_args.long_input_len
    )
    rank_print(
        "Synthetic input hashes: "
        f"seed={server_args.random_seed} "
        f"short_sha256={_request_token_digest(short_reqs)} "
        f"long_sha256={_request_token_digest(long_reqs)}"
    )
    for i, req in enumerate(long_reqs):
        req.rid = bench_args.short_bs + i

    short_next, _, short_batch = _extend_prefill_in_chunks(
        short_reqs,
        model_runner,
        bench_args.prefill_chunk_size,
        rank_print,
        "short lane",
    )
    long_next, _, long_batch = _extend_prefill_in_chunks(
        long_reqs,
        model_runner,
        bench_args.prefill_chunk_size,
        rank_print,
        "long lane",
    )
    short_fb = _prepare_decode_forward_batch(short_next, short_batch, model_runner)
    long_fb = _prepare_decode_forward_batch(long_next, long_batch, model_runner)

    if bench_args.run_modes in (
        "all",
        "concurrent_only",
        "baseline_cap_only",
        "baseline_cap_after_short",
    ):
        selected_concurrent_mode = (
            "long_cap_concurrent"
            if bench_args.concurrent_long_variant == "cap"
            else "long_full_concurrent"
        )
        if bench_args.run_modes == "baseline_cap_only":
            modes = [selected_concurrent_mode]
        elif bench_args.run_modes == "concurrent_only":
            modes = [selected_concurrent_mode]
        elif bench_args.run_modes == "baseline_cap_after_short":
            modes = ["long_cap_concurrent", "long_full_after_short"]
        else:
            modes = [
                "long_cap_concurrent",
                "long_full_concurrent",
                "long_full_after_short",
            ]
        for mode in modes:
            rank_print(f"Warmup {mode} ...")
            for _ in range(bench_args.warmups):
                _replay_once(
                    model_runner,
                    short_fb,
                    long_fb,
                    mode,
                    decode_steps=bench_args.decode_steps,
                )

            rank_print(f"Benchmark {mode} ...")
            mode_rows = []
            for i in range(bench_args.iters):
                row = {
                    "mode": mode,
                    "iter": i,
                    **_replay_once(
                        model_runner,
                        short_fb,
                        long_fb,
                        mode,
                        decode_steps=bench_args.decode_steps,
                    ),
                }
                mode_rows.append(row)
                rows.append(row)
                layer_timing_rows.extend(
                    _collect_long_layer_timing(model_runner, mode, i)
                )

            short_stats = _summarize([r["short_ms"] for r in mode_rows])
            long_stats = _summarize([r["long_ms"] for r in mode_rows])
            total_stats = _summarize([r["total_ms"] for r in mode_rows])
            wall_stats = _summarize([r["wall_ms"] for r in mode_rows])
            short_replay_stats = _summarize([r["short_replay_ms"] for r in mode_rows])
            long_replay_stats = _summarize([r["long_replay_ms"] for r in mode_rows])
            total_per_step_stats = _summarize(
                [r["total_ms_per_step"] for r in mode_rows]
            )
            summary = {
                "mode": mode,
                "decode_steps": bench_args.decode_steps,
                "short_ms_median": short_stats["median"],
                "short_ms_mean": short_stats["mean"],
                "short_ms_p90": short_stats["p90"],
                "long_ms_median": long_stats["median"],
                "long_ms_mean": long_stats["mean"],
                "long_ms_p90": long_stats["p90"],
                "total_ms_median": total_stats["median"],
                "total_ms_mean": total_stats["mean"],
                "total_ms_p90": total_stats["p90"],
                "wall_ms_median": wall_stats["median"],
                "wall_ms_mean": wall_stats["mean"],
                "wall_ms_p90": wall_stats["p90"],
                "short_replay_ms_median": short_replay_stats["median"],
                "short_replay_ms_mean": short_replay_stats["mean"],
                "short_replay_ms_p90": short_replay_stats["p90"],
                "long_replay_ms_median": long_replay_stats["median"],
                "long_replay_ms_mean": long_replay_stats["mean"],
                "long_replay_ms_p90": long_replay_stats["p90"],
                "total_ms_per_step_median": total_per_step_stats["median"],
                "total_ms_per_step_mean": total_per_step_stats["mean"],
                "total_ms_per_step_p90": total_per_step_stats["p90"],
            }
            summaries.append(summary)
            rank_print(json.dumps(summary, indent=2))

    if bench_args.run_modes in (
        "concurrent_only",
        "baseline_cap_only",
        "baseline_cap_after_short",
    ):
        if tp_rank == 0:
            _write_results(bench_args.result_prefix, rows, summaries)
            _write_layer_timing_results(bench_args.result_prefix, layer_timing_rows)
            png_path = (
                _maybe_write_png(bench_args.result_prefix, summaries)
                if bench_args.make_png
                else None
            )

            print(f"Wrote {Path(bench_args.result_prefix).with_suffix('.csv')}")
            print(f"Wrote {Path(bench_args.result_prefix).with_suffix('.json')}")
            if png_path:
                print(f"Wrote {png_path}")

        if server_args.tp_size > 1:
            destroy_distributed_environment()
        elif dist.is_initialized():
            dist.destroy_process_group()
        return

    conditional_mode = "long_conditional_switch"
    if bench_args.conditional_force_cap:
        conditional_mode = "long_conditional_all_cap"
    rank_print("Building long conditional switch graph ...")
    cond_runner = Qwen3MoeConditionalDecodeRunner(
        model_runner,
        short_fb,
        long_fb,
        short_lane=model_runner.decode_lane_runtimes[0],
        long_lane=model_runner.decode_lane_runtimes[1],
        long_attention_active_cta_limit=bench_args.long_attention_active_cta_limit,
        long_attention_active_cta_ratio=bench_args.long_attention_active_cta_ratio,
        long_attention_active_cta_per_sm=bench_args.long_attention_active_cta_per_sm,
        long_moe_active_cta_limit=bench_args.long_moe_active_cta_limit,
        long_moe_active_cta_ratio=bench_args.long_moe_active_cta_ratio,
        long_chunk_count=bench_args.conditional_chunks,
        force_cap_after_short=bench_args.conditional_force_cap,
        validate_tp_branch_consistency=bench_args.validate_tp_branch_consistency,
    )
    cond_runner.capture()

    rank_print(f"Warmup {conditional_mode} ...")
    for _ in range(bench_args.warmups):
        _conditional_once(cond_runner)

    rank_print(f"Benchmark {conditional_mode} ...")
    conditional_rows = []
    for i in range(bench_args.iters):
        row = {
            "mode": conditional_mode,
            "iter": i,
            **_conditional_once(cond_runner),
        }
        conditional_rows.append(row)
        rows.append(row)

    short_stats = _summarize([r["short_ms"] for r in conditional_rows])
    long_stats = _summarize([r["long_ms"] for r in conditional_rows])
    total_stats = _summarize([r["total_ms"] for r in conditional_rows])
    wall_stats = _summarize([r["wall_ms"] for r in conditional_rows])
    short_replay_stats = _summarize([r["short_replay_ms"] for r in conditional_rows])
    long_replay_stats = _summarize([r["long_replay_ms"] for r in conditional_rows])
    cap_chunk_stats = _summarize([r["cap_chunks"] for r in conditional_rows])
    full_chunk_stats = _summarize([r["full_chunks"] for r in conditional_rows])
    switch_chunk_stats = _summarize([r["switch_chunk"] for r in conditional_rows])
    conditional_summary = {
        "mode": conditional_mode,
        "short_ms_median": short_stats["median"],
        "short_ms_mean": short_stats["mean"],
        "short_ms_p90": short_stats["p90"],
        "long_ms_median": long_stats["median"],
        "long_ms_mean": long_stats["mean"],
        "long_ms_p90": long_stats["p90"],
        "total_ms_median": total_stats["median"],
        "total_ms_mean": total_stats["mean"],
        "total_ms_p90": total_stats["p90"],
        "wall_ms_median": wall_stats["median"],
        "wall_ms_mean": wall_stats["mean"],
        "wall_ms_p90": wall_stats["p90"],
        "short_replay_ms_median": short_replay_stats["median"],
        "short_replay_ms_mean": short_replay_stats["mean"],
        "short_replay_ms_p90": short_replay_stats["p90"],
        "long_replay_ms_median": long_replay_stats["median"],
        "long_replay_ms_mean": long_replay_stats["mean"],
        "long_replay_ms_p90": long_replay_stats["p90"],
        "cap_chunks_median": cap_chunk_stats["median"],
        "full_chunks_median": full_chunk_stats["median"],
        "switch_chunk_median": switch_chunk_stats["median"],
        "branch": conditional_rows[-1]["branch"],
        "observed_done": conditional_rows[-1]["observed_done"],
    }
    summaries.append(conditional_summary)
    rank_print(json.dumps(conditional_summary, indent=2))

    if tp_rank == 0:
        _write_results(bench_args.result_prefix, rows, summaries)
        _write_layer_timing_results(bench_args.result_prefix, layer_timing_rows)
        png_path = (
            _maybe_write_png(bench_args.result_prefix, summaries)
            if bench_args.make_png
            else None
        )

        print(f"Wrote {Path(bench_args.result_prefix).with_suffix('.csv')}")
        print(f"Wrote {Path(bench_args.result_prefix).with_suffix('.json')}")
        if png_path:
            print(f"Wrote {png_path}")

    if bench_args.profile_mode == "torch":
        if server_args.tp_size > 1:
            raise ValueError("torch profiling in this benchmark is tp_size=1 only")
        profile_prefix = (
            bench_args.profile_output_prefix
            or f"{bench_args.result_prefix}_{bench_args.profile_target}"
        )
        print(f"Profiling {bench_args.profile_target} ...")
        profile_summary = _write_torch_profile(
            profile_prefix,
            model_runner,
            short_fb,
            long_fb,
            bench_args.profile_target,
        )
        print(json.dumps(profile_summary["groups"], indent=2))

    if server_args.tp_size > 1:
        destroy_distributed_environment()
    elif dist.is_initialized():
        dist.destroy_process_group()


def run(server_args: ServerArgs, bench_args: DecodeLaneBenchArgs) -> None:
    _prepare_server_args_for_decode_lane_bench(server_args, bench_args)
    port_args = PortArgs.init_new(server_args)

    if server_args.tp_size == 1:
        run_worker(server_args, port_args, bench_args, server_args.base_gpu_id, 0)
        return

    mp_context = multiprocessing.get_context("spawn")
    workers = []
    for tp_rank in range(server_args.tp_size):
        with maybe_reindex_device_id(tp_rank) as gpu_id:
            proc = mp_context.Process(
                target=run_worker,
                args=(server_args, port_args, bench_args, gpu_id, tp_rank),
            )
            proc.start()
            workers.append(proc)

    failed = []
    for proc in workers:
        proc.join()
        if proc.exitcode != 0:
            failed.append((proc.pid, proc.exitcode))

    if failed:
        raise RuntimeError(f"TP benchmark worker failed: {failed}")


def main():
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    DecodeLaneBenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    bench_args = DecodeLaneBenchArgs.from_cli_args(args)

    logging.basicConfig(
        level=getattr(logging, server_args.log_level.upper()),
        format="%(message)s",
    )
    try:
        run(server_args, bench_args)
    finally:
        if server_args.tp_size != 1:
            kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
