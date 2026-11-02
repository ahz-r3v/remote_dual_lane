#!/usr/bin/env python3
"""Measure isolated Qwen prefill scaling across CUDA Green Context SM counts.

The benchmark runs one prefill forward at a time.  The companion Green Context
stream is deliberately left idle, so the result measures prefiller scaling
without decoder interference.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.analysis.deepresearch_token_corpus import load_token_batch

from sglang.bench_one_batch import (
    extend,
    load_model,
    prepare_synthetic_inputs_for_latency_test,
)
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.layers.moe import initialize_moe_config
from sglang.srt.layers.quantization.fp4_utils import initialize_fp4_gemm_config
from sglang.srt.layers.quantization.fp8_utils import initialize_fp8_gemm_config
from sglang.srt.observability.model_perf_estimator import (
    ForwardPerfRecord,
    Qwen35MoePerfEstimator,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger
from sglang.srt.utils.device_timer import DeviceTimer


class PerfCollector:
    def __init__(self) -> None:
        self.rows: list[dict] = []

    def report(self, *, t: float, perf_record: ForwardPerfRecord | None, **_) -> None:
        if perf_record is None or perf_record.mode != "prefill":
            return
        estimate = perf_record.resolved_estimate
        self.rows.append(
            {
                "forward_seconds": float(t),
                "num_tokens": int(perf_record.num_tokens),
                "flops": float(estimate.flops),
                "read_bytes": float(estimate.read_bytes),
                "write_bytes": float(estimate.write_bytes),
                "kv_cache_read_bytes": float(estimate.kv_cache_read_bytes),
                "kv_cache_write_bytes": float(estimate.kv_cache_write_bytes),
                "moe_read_bytes": float(estimate.moe_read_bytes),
            }
        )

    def flush(self, timer: DeviceTimer) -> None:
        torch.cuda.synchronize()
        timer._report()

    def discard_pending(self, timer: DeviceTimer) -> None:
        # A failed CUDA forward still closes DeviceTimer.wrap() and can leave an
        # interval waiting to be reported. It must not be attributed to the next
        # independent benchmark point.
        with timer._lock:
            timer._intervals.clear()


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    if len(values) == 1:
        return values[0]
    index = (len(values) - 1) * fraction
    lo = math.floor(index)
    hi = math.ceil(index)
    return values[lo] * (hi - index) + values[hi] * (index - lo)


def mean_field(rows: list[dict], field: str) -> float:
    return statistics.mean(float(row[field]) for row in rows)


def make_inputs(
    batch_size: int,
    input_len: int,
    seed: int,
    token_corpus_dir: Path | None = None,
    token_corpus_split: str = "train",
) -> np.ndarray:
    if token_corpus_dir is not None:
        batches, selected = load_token_batch(
            token_corpus_dir,
            split=token_corpus_split,
            batch_size=batch_size,
            input_len=input_len,
            seed=seed,
        )
        print(
            "DeepResearch inputs: "
            f"split={token_corpus_split} requests={batch_size} length={input_len} "
            f"first={selected[0]['task_id']}/{selected[0]['session_id']}"
        )
        return np.asarray(batches, dtype=np.int32)
    # Keep the exact token IDs identical across SM-count jobs.
    point_seed = seed + batch_size * 1_000_003 + input_len * 101
    rng = np.random.default_rng(point_seed)
    return rng.integers(
        0,
        10_000,
        size=(batch_size, input_len),
        dtype=np.int32,
    )


def create_prefill_stream(
    *, gpu_id: int, requested_sm_count: int
) -> tuple[torch.cuda.Stream, int, int, str]:
    physical_sms = torch.cuda.get_device_properties(gpu_id).multi_processor_count
    if requested_sm_count <= 0:
        return torch.cuda.Stream(device=gpu_id), physical_sms, physical_sms, "full"

    from sgl_kernel import spatial

    physical_available_sms = int(spatial.get_sm_available(gpu_id))
    # Hopper Green Context partitions are allocated in 8-SM groups. H200
    # exposes 132 physical SMs, of which 128 can participate in the paired
    # Green Context split used by production PDMux (for example, 80/48).
    green_total_sms = physical_available_sms // 8 * 8
    idle_sm_count = green_total_sms - requested_sm_count
    if requested_sm_count % 8 != 0 or idle_sm_count < 8 or idle_sm_count % 8 != 0:
        raise ValueError(
            "Hopper Green Context SM counts must be multiples of 8 and leave "
            f"at least 8 SMs for the idle partition; requested={requested_sm_count}, "
            f"physical_available={physical_available_sms}, "
            f"green_partition_total={green_total_sms}"
        )
    prefill_stream, _idle_stream = spatial.create_greenctx_stream_by_value(
        requested_sm_count, idle_sm_count, gpu_id
    )
    return prefill_stream, requested_sm_count, green_total_sms, "green_context"


def run_point(
    *,
    wrapped_runner,
    model_runner,
    timer: DeviceTimer,
    collector: PerfCollector,
    stream: torch.cuda.Stream,
    batch_size: int,
    input_len: int,
    warmups: int,
    iters: int,
    seed: int,
    token_corpus_dir: Path | None,
    token_corpus_split: str,
) -> tuple[list[dict], dict]:
    custom_inputs = make_inputs(
        batch_size,
        input_len,
        seed,
        token_corpus_dir,
        token_corpus_split,
    )

    def run_once() -> dict:
        wrapped_runner.clear()
        reqs = prepare_synthetic_inputs_for_latency_test(
            batch_size, input_len, custom_inputs.tolist()
        )
        start = len(collector.rows)
        with torch.cuda.stream(stream):
            extend(reqs, model_runner)
        stream.synchronize()
        collector.flush(timer)
        new_rows = collector.rows[start:]
        if len(new_rows) != 1:
            raise RuntimeError(
                f"Expected one completed prefill perf record, got {len(new_rows)}"
            )
        return new_rows[0]

    for _ in range(warmups):
        run_once()

    samples = [run_once() for _ in range(iters)]
    times = [float(row["forward_seconds"]) for row in samples]
    median_seconds = statistics.median(times)
    mean_seconds = statistics.mean(times)
    flops = mean_field(samples, "flops")
    read_bytes = mean_field(samples, "read_bytes")
    write_bytes = mean_field(samples, "write_bytes")
    num_tokens = mean_field(samples, "num_tokens")
    summary = {
        "batch_size": batch_size,
        "request_length": input_len,
        "prompt_tokens": int(num_tokens),
        "forward_ms_median": median_seconds * 1e3,
        "forward_ms_mean": mean_seconds * 1e3,
        "forward_ms_p90": percentile(times, 0.90) * 1e3,
        "active_prompt_tokens_per_second": num_tokens / median_seconds,
        "estimated_flops": flops,
        "active_tflops": flops / median_seconds / 1e12,
        "estimated_read_bytes": read_bytes,
        "estimated_write_bytes": write_bytes,
        "active_read_gb_s": read_bytes / median_seconds / 1e9,
        "active_write_gb_s": write_bytes / median_seconds / 1e9,
        "active_total_gb_s": (read_bytes + write_bytes) / median_seconds / 1e9,
        "estimated_kv_cache_read_bytes": mean_field(
            samples, "kv_cache_read_bytes"
        ),
        "estimated_kv_cache_write_bytes": mean_field(
            samples, "kv_cache_write_bytes"
        ),
        "estimated_moe_read_bytes": mean_field(samples, "moe_read_bytes"),
        "iterations": iters,
    }
    return samples, summary


def parse_args() -> tuple[ServerArgs, argparse.Namespace]:
    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--prefill-sm-count", type=int, required=True)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    parser.add_argument("--request-lengths", type=int, nargs="+", required=True)
    parser.add_argument("--bench-warmups", type=int, default=2)
    parser.add_argument("--bench-iters", type=int, default=5)
    parser.add_argument("--result-file", type=Path, required=True)
    parser.add_argument("--token-corpus-dir", type=Path)
    parser.add_argument(
        "--token-corpus-split", choices=("train", "holdout"), default="train"
    )
    args = parser.parse_args()
    return ServerArgs.from_cli_args(args), args


def main() -> None:
    server_args, args = parse_args()
    if server_args.tp_size != 1:
        raise ValueError("This initial prefill SM sweep supports tp_size=1 only")

    # Prefill is eager in the production PDMux path under study. Avoid unrelated
    # CUDA graph capture and make the benchmark's memory requirement predictable.
    server_args.disable_cuda_graph = True
    server_args.disable_piecewise_cuda_graph = True
    server_args.max_running_requests = max(args.batch_sizes)
    server_args.max_mamba_cache_size = max(args.batch_sizes)

    _set_envs_and_config(server_args)
    initialize_moe_config(server_args)
    initialize_fp8_gemm_config(server_args)
    initialize_fp4_gemm_config(server_args)
    configure_logger(server_args, prefix=" PREFILL-SM-SWEEP")
    port_args = PortArgs.init_new(server_args)

    wrapped_runner, _tokenizer = load_model(
        server_args, port_args, server_args.base_gpu_id, 0
    )
    model_runner = wrapped_runner.torch_runner

    collector = PerfCollector()
    timer = DeviceTimer(reporter=collector.report)
    model_runner.perf_estimator = Qwen35MoePerfEstimator.from_model_config(
        model_runner.model_config, server_args.tp_size
    )
    model_runner.device_timer = timer

    stream, actual_sms, green_total_sms, stream_kind = create_prefill_stream(
        gpu_id=server_args.base_gpu_id,
        requested_sm_count=args.prefill_sm_count,
    )
    metadata = {
        "model_path": server_args.model_path,
        "gpu_name": torch.cuda.get_device_name(server_args.base_gpu_id),
        "physical_sm_count": torch.cuda.get_device_properties(
            server_args.base_gpu_id
        ).multi_processor_count,
        "requested_prefill_sm_count": args.prefill_sm_count,
        "actual_prefill_sm_count": actual_sms,
        "green_context_total_sm_count": green_total_sms,
        "stream_kind": stream_kind,
        "batch_sizes": args.batch_sizes,
        "request_lengths": args.request_lengths,
        "warmups": args.bench_warmups,
        "iterations": args.bench_iters,
        "random_seed": server_args.random_seed,
        "token_corpus_dir": (
            str(args.token_corpus_dir.resolve()) if args.token_corpus_dir else None
        ),
        "token_corpus_split": args.token_corpus_split,
        "bandwidth_definition": (
            "estimated useful model read/write bytes divided by CUDA Event "
            "model-forward time; not a physical DRAM counter"
        ),
    }

    output = {"metadata": metadata, "points": []}
    for batch_size in args.batch_sizes:
        for input_len in args.request_lengths:
            print(
                f"Benchmarking sm={actual_sms} bs={batch_size} "
                f"request_length={input_len}",
                flush=True,
            )
            try:
                samples, summary = run_point(
                    wrapped_runner=wrapped_runner,
                    model_runner=model_runner,
                    timer=timer,
                    collector=collector,
                    stream=stream,
                    batch_size=batch_size,
                    input_len=input_len,
                    warmups=args.bench_warmups,
                    iters=args.bench_iters,
                    seed=server_args.random_seed,
                    token_corpus_dir=args.token_corpus_dir,
                    token_corpus_split=args.token_corpus_split,
                )
                summary.update(
                    {
                        "requested_prefill_sm_count": args.prefill_sm_count,
                        "actual_prefill_sm_count": actual_sms,
                        "status": "ok",
                        "samples": samples,
                    }
                )
            except Exception as exc:
                summary = {
                    "batch_size": batch_size,
                    "request_length": input_len,
                    "requested_prefill_sm_count": args.prefill_sm_count,
                    "actual_prefill_sm_count": actual_sms,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"Point failed: {summary['error']}", flush=True)
                # Clear potentially allocated request/cache slots before the
                # next independent point.
                torch.cuda.synchronize()
                wrapped_runner.clear()
                collector.discard_pending(timer)
                gc.collect()
                torch.cuda.empty_cache()
            output["points"].append(summary)
            args.result_file.parent.mkdir(parents=True, exist_ok=True)
            args.result_file.write_text(json.dumps(output, indent=2))

    print(f"Wrote {args.result_file}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
