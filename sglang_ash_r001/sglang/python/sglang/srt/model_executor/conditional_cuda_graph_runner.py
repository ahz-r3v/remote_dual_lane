"""Experimental CUDA conditional-graph runner for decode lane switching.

This runner mirrors the native CUDA conditional-node prototype in
``highway_kernel_cuda_graph``.  It owns one parent CUDA graph:

```
reset
  |-- short_body -> mark_short_done
  `-- check_done_0 -> IF(full_0, cap_0)
      check_done_1 -> IF(full_1, cap_1)
      ...
```

The Python side only supplies the work bodies to capture.  This keeps the native
builder generic and lets SGLang-specific code decide whether a "chunk" is one
decoder layer or a larger layer range.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Callable, Optional

import torch
from torch.utils.cpp_extension import load

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.layers.dp_attention import set_dp_buffer_len, set_is_extend_in_batch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context


_EXTENSION: Optional[ModuleType] = None


def _prepare_build_env() -> None:
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    env_bin = Path(sys.executable).resolve().parent
    os.environ["PATH"] = (
        f"{env_bin}:/usr/local/cuda/bin:{os.environ.get('PATH', '')}"
    )
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")


def load_conditional_cuda_graph_extension(verbose: bool = False) -> ModuleType:
    """JIT-build and return the tiny native conditional graph builder."""

    global _EXTENSION
    if _EXTENSION is not None:
        return _EXTENSION

    _prepare_build_env()
    src = Path(__file__).with_name("conditional_cuda_graph_builder.cu")
    _EXTENSION = load(
        name="sglang_conditional_cuda_graph",
        sources=[str(src)],
        extra_cuda_cflags=["-O3", "--std=c++17"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=verbose,
    )
    return _EXTENSION


@dataclass(frozen=True)
class ConditionalCudaGraphReplayResult:
    chunks: int
    short_ms: float
    long_ms: float
    total_ms: float
    branch: list[str]
    observed_done: list[int]
    cap_chunks: int
    full_chunks: int

    @classmethod
    def from_native(cls, result: dict) -> "ConditionalCudaGraphReplayResult":
        return cls(
            chunks=int(result["chunks"]),
            short_ms=float(result["short_ms"]),
            long_ms=float(result["long_ms"]),
            total_ms=float(result["total_ms"]),
            branch=list(result["branch"]),
            observed_done=[int(x) for x in result["observed_done"]],
            cap_chunks=int(result["cap_chunks"]),
            full_chunks=int(result["full_chunks"]),
        )


ShortBody = Callable[[torch.cuda.ExternalStream], None]
LongChunkBody = Callable[[int, bool, torch.cuda.ExternalStream], None]


class ConditionalDecodeLaneCudaGraphRunner:
    """Capture and replay a parent graph that switches long chunks cap->full.

    ``short_body`` is captured once and must enqueue all short-lane work.
    ``long_chunk_body`` is captured twice for every chunk: once with
    ``full_branch=True`` and once with ``full_branch=False``.  At replay time the
    native graph checks the device-side ``short_done`` flag before each long
    chunk and selects the corresponding branch.

    The runner is intentionally experimental and does not own SGLang static
    buffers yet.  Callers are responsible for preparing graph-stable tensors and
    for ensuring both long branches of each chunk write their output into the
    same next-chunk input buffers.
    """

    def __init__(
        self,
        chunks: int,
        *,
        verbose_build: bool = False,
        name: str = "conditional_decode_lane",
    ) -> None:
        if chunks <= 0:
            raise ValueError("chunks must be positive")
        if not torch.cuda.is_available():
            raise RuntimeError("ConditionalDecodeLaneCudaGraphRunner requires CUDA")
        self.chunks = int(chunks)
        self.name = name
        self.ext = load_conditional_cuda_graph_extension(verbose=verbose_build)
        self.builder = self.ext.ConditionalCudaGraphBuilder(self.chunks)
        self.capture_stream = torch.cuda.ExternalStream(self.builder.stream_ptr())
        self.short_graph_pool = torch.cuda.graph_pool_handle()
        self.long_graph_pool = torch.cuda.graph_pool_handle()
        self._body_graphs: list[torch.cuda.CUDAGraph] = []
        self._captured = False

    def capture(
        self,
        short_body: ShortBody,
        long_chunk_body: LongChunkBody,
        *,
        short_context=None,
        long_full_context=None,
        long_cap_context=None,
    ) -> None:
        """Capture short body and every long chunk branch into one parent graph."""

        if self._captured:
            raise RuntimeError("conditional graph has already been captured")

        torch.cuda.synchronize()
        short_graph = self._capture_body_graph(
            lambda: short_body(self.capture_stream),
            short_context if short_context is not None else nullcontext(),
            self.short_graph_pool,
        )
        self.builder.add_short_child_graph(short_graph.raw_cuda_graph())
        self._body_graphs.append(short_graph)

        for chunk in range(self.chunks):
            self.builder.add_long_chunk(chunk)
            self._capture_long_branch(
                chunk,
                True,
                long_chunk_body,
                long_full_context if long_full_context is not None else nullcontext(),
            )
            self._capture_long_branch(
                chunk,
                False,
                long_chunk_body,
                long_cap_context if long_cap_context is not None else nullcontext(),
            )

        self.builder.instantiate()
        self._captured = True

    def replay(
        self,
        *,
        repeats: int = 1,
        warmup: int = 0,
        launch_priority: int = 0,
    ) -> ConditionalCudaGraphReplayResult:
        if not self._captured:
            raise RuntimeError("capture() must be called before replay()")
        native = self.builder.replay(
            int(repeats), int(warmup), int(launch_priority)
        )
        return ConditionalCudaGraphReplayResult.from_native(dict(native))

    def _capture_long_branch(
        self,
        chunk: int,
        full_branch: bool,
        long_chunk_body: LongChunkBody,
        context,
    ) -> None:
        graph = self._capture_body_graph(
            lambda: long_chunk_body(chunk, full_branch, self.capture_stream),
            context,
            self.long_graph_pool,
        )
        self.builder.add_long_branch_child_graph(
            chunk, full_branch, graph.raw_cuda_graph()
        )
        self._body_graphs.append(graph)

    def _capture_body_graph(self, body: Callable[[], None], context, pool):
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with context:
            with torch.cuda.graph(graph, pool=pool, stream=self.capture_stream):
                body()
        return graph


def branch_switch_index(branch: list[str]) -> Optional[int]:
    """Return the first chunk index that took the full branch, if any."""

    for i, item in enumerate(branch):
        if item == "full":
            return i
    return None


class Qwen3MoeConditionalDecodeRunner:
    """Qwen3-MoE decode-only conditional graph runner.

    This is the first SGLang end-to-end integration target for the highway-style
    conditional graph.  Each TP rank owns and replays an isomorphic local graph
    over its model shard.  The long lane has ``num_layers + 1`` chunks; the final
    chunk runs norm/logits in both branches.
    """

    def __init__(
        self,
        model_runner,
        short_forward_batch: ForwardBatch,
        long_forward_batch: ForwardBatch,
        *,
        short_lane,
        long_lane,
        long_attention_active_cta_limit: int = 0,
        long_attention_active_cta_ratio: float = 0.0,
        long_attention_active_cta_per_sm: int = 0,
        long_moe_active_cta_limit: int = 0,
        long_moe_active_cta_ratio: float = 0.0,
        long_chunk_count: int = 0,
        force_cap_after_short: bool = False,
        validate_tp_branch_consistency: bool = False,
        verbose_build: bool = False,
    ) -> None:
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.model_runner = model_runner
        self.model = model_runner.model
        self.short_fb = short_forward_batch
        self.long_fb = long_forward_batch
        self.short_lane = short_lane
        self.long_lane = long_lane
        self.long_attention_active_cta_limit = max(int(long_attention_active_cta_limit), 0)
        self.long_attention_active_cta_ratio = max(float(long_attention_active_cta_ratio), 0.0)
        self.long_attention_active_cta_per_sm = max(
            int(long_attention_active_cta_per_sm), 0
        )
        self.long_moe_active_cta_limit = max(int(long_moe_active_cta_limit), 0)
        self.long_moe_active_cta_ratio = max(float(long_moe_active_cta_ratio), 0.0)
        self.force_cap_after_short = bool(force_cap_after_short)
        self.validate_tp_branch_consistency = bool(validate_tp_branch_consistency)

        self.language_model = getattr(self.model, "model", None)
        if self.language_model is None or not hasattr(self.language_model, "layers"):
            raise ValueError("expected a Qwen3/Qwen2 style model with .model.layers")
        self.layers = self.language_model.layers
        self.num_layers = int(self.language_model.config.num_hidden_layers)
        self.long_chunk_count = (
            min(int(long_chunk_count), self.num_layers + 1)
            if long_chunk_count > 0
            else self.num_layers + 1
        )
        self.hidden_size = int(self.language_model.config.hidden_size)
        self.dtype = model_runner.dtype
        self.device = model_runner.device
        self.runner = ConditionalDecodeLaneCudaGraphRunner(
            self.long_chunk_count,
            verbose_build=verbose_build,
            name="qwen3_moe_conditional_decode",
        )

        self.long_hidden = [
            torch.empty(
                (self.long_fb.input_ids.shape[0], self.hidden_size),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(2)
        ]
        self.long_residual = [torch.empty_like(self.long_hidden[0]) for _ in range(2)]
        self._keepalive = []

    @torch.no_grad()
    def capture(self) -> None:
        """Capture the parent graph."""

        self._prepare_attention_backend_for_capture(
            self.short_lane.attn_backend, self.short_fb
        )
        self._prepare_attention_backend_for_capture(
            self.long_lane.attn_backend, self.long_fb
        )

        # Warm a few model paths outside native capture so Triton/FlashInfer lazy
        # initialization does not happen while cudaStreamBeginCaptureToGraph is active.
        self._warmup()
        torch.cuda.synchronize()

        self.runner.capture(self._short_body, self._long_chunk_body)

    def replay(
        self,
        *,
        repeats: int = 1,
        warmup: int = 0,
        launch_priority: int = 0,
    ) -> ConditionalCudaGraphReplayResult:
        result = self.runner.replay(
            repeats=repeats, warmup=warmup, launch_priority=launch_priority
        )
        if self.validate_tp_branch_consistency:
            self._validate_tp_branch_consistency(result)
        return result

    def _validate_tp_branch_consistency(
        self, result: ConditionalCudaGraphReplayResult
    ) -> None:
        if self.tp_size <= 1:
            return

        local = {
            "tp_rank": self.tp_rank,
            "branch": tuple(result.branch),
            "observed_done": tuple(result.observed_done),
            "cap_chunks": result.cap_chunks,
            "full_chunks": result.full_chunks,
        }
        gathered = get_tp_group().all_gather_object(local)
        reference = gathered[0]
        mismatches = [
            item
            for item in gathered
            if item["branch"] != reference["branch"]
            or item["observed_done"] != reference["observed_done"]
        ]
        if mismatches:
            raise RuntimeError(
                "conditional decode graph TP branch divergence: "
                f"rank0={reference}, local={local}, gathered={gathered}"
            )

    def _prepare_attention_backend_for_capture(self, attn_backend, fb: ForwardBatch):
        with forward_context(ForwardContext(attn_backend=attn_backend)):
            attn_backend.init_forward_metadata_out_graph(fb, in_capture=True)

    def _short_context(self):
        return forward_context(ForwardContext(attn_backend=self.short_lane.attn_backend))

    def _long_context(self, full_branch: bool):
        if full_branch and not self.force_cap_after_short:
            return forward_context(ForwardContext(attn_backend=self.long_lane.attn_backend))
        return forward_context(
            ForwardContext(
                attn_backend=self.long_lane.attn_backend,
                attention_active_cta_limit=self.long_attention_active_cta_limit,
                attention_active_cta_ratio=self.long_attention_active_cta_ratio,
                attention_active_sm_count=self.long_lane.sm_count,
                attention_active_cta_per_sm=self.long_attention_active_cta_per_sm,
                moe_active_cta_limit=self.long_moe_active_cta_limit,
                moe_active_cta_ratio=self.long_moe_active_cta_ratio,
                moe_active_sm_count=self.long_lane.sm_count,
            )
        )

    def _prepare_in_graph_common(self, fb: ForwardBatch, attn_backend) -> None:
        attn_backend.init_forward_metadata_in_graph(fb)
        fb.dp_local_start_pos = fb.dp_local_num_tokens = None
        is_max_len_padding = (
            fb.dp_padding_mode.is_max_len()
            if fb.dp_padding_mode is not None
            else False
        )
        set_dp_buffer_len(
            fb.global_dp_buffer_len,
            fb.input_ids.shape[0],
            is_max_len_padding,
            None,
        )
        set_is_extend_in_batch(False)

    def _short_body(self, stream: torch.cuda.ExternalStream) -> None:
        with self._short_context():
            self._prepare_in_graph_common(self.short_fb, self.short_lane.attn_backend)
            out = self.model.forward(
                self.short_fb.input_ids,
                self.short_fb.positions,
                self.short_fb,
            )
            self._keepalive.append(out)

    def _long_chunk_body(
        self,
        chunk: int,
        full_branch: bool,
        stream: torch.cuda.ExternalStream,
    ) -> None:
        with self._long_context(full_branch):
            if chunk == 0:
                self._prepare_in_graph_common(self.long_fb, self.long_lane.attn_backend)
            if chunk < self.num_layers:
                self._run_long_layer(chunk)
            else:
                self._run_long_tail()

    def _run_long_layer(self, layer_id: int) -> None:
        if layer_id == 0:
            hidden = self.language_model.embed_tokens(self.long_fb.input_ids)
            self.long_hidden[0].copy_(hidden)
            hidden_in = self.long_hidden[0]
            residual_in = None
        else:
            in_idx = layer_id % 2
            hidden_in = self.long_hidden[in_idx]
            residual_in = self.long_residual[in_idx]

        with get_global_expert_distribution_recorder().with_current_layer(layer_id):
            hidden_out, residual_out = self.layers[layer_id](
                positions=self.long_fb.positions,
                hidden_states=hidden_in,
                residual=residual_in,
                forward_batch=self.long_fb,
                captured_last_layer_outputs=None,
            )

        out_idx = (layer_id + 1) % 2
        self.long_hidden[out_idx].copy_(hidden_out)
        if residual_out is not None:
            self.long_residual[out_idx].copy_(residual_out)

    def _run_long_tail(self) -> None:
        in_idx = self.num_layers % 2
        hidden = self.long_hidden[in_idx]
        residual = self.long_residual[in_idx]
        if hidden.shape[0] != 0:
            hidden, _ = self.language_model.norm(hidden, residual)
        out = self.model.logits_processor(
            self.long_fb.input_ids,
            hidden,
            self.model.lm_head,
            self.long_fb,
        )
        self._keepalive.append(out)

    @torch.no_grad()
    def _warmup(self) -> None:
        # Warm short full path once.
        self._short_body(torch.cuda.current_stream())

        # Warm both long variants.  This mutates the long static buffers, which is
        # fine; chunk 0 reinitializes them from embeddings inside the captured graph.
        for full_branch in (True, False):
            for chunk in range(self.long_chunk_count):
                self._long_chunk_body(chunk, full_branch, torch.cuda.current_stream())

        if hasattr(self.short_lane.attn_backend, "on_after_cuda_graph_warmup"):
            self.short_lane.attn_backend.on_after_cuda_graph_warmup()
        if hasattr(self.long_lane.attn_backend, "on_after_cuda_graph_warmup"):
            self.long_lane.attn_backend.on_after_cuda_graph_warmup()
