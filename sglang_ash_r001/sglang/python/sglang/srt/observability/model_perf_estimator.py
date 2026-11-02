"""Analytical model-work estimator used by MFU/MBU metrics.

The estimator counts useful model operations rather than issued GPU
instructions.  Memory traffic is the minimum model traffic: local weights are
read once per forward, embedding rows are read once per input token, and KV
cache reads/writes are counted explicitly.  Backend-dependent temporary
activation traffic is intentionally excluded.

The validated architectures are the dense Qwen3 decoder family (including
Qwen/Qwen3-32B) and BF16 Qwen3.5 MoE with hybrid full/GDN attention.  Routed
expert weights are deliberately excluded from the Qwen3.5 static estimate;
they are added after the forward from the per-layer set of experts that were
actually selected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.observability.moe_expert_read_tracker import ExpertReadSnapshot


@dataclass(frozen=True)
class PerfEstimate:
    flops: float = 0.0
    read_bytes: float = 0.0
    write_bytes: float = 0.0
    kv_cache_read_bytes: float = 0.0
    kv_cache_write_bytes: float = 0.0
    moe_read_bytes: float = 0.0

    def with_added_moe_read_bytes(self, read_bytes: float) -> "PerfEstimate":
        return PerfEstimate(
            flops=self.flops,
            read_bytes=self.read_bytes + read_bytes,
            write_bytes=self.write_bytes,
            kv_cache_read_bytes=self.kv_cache_read_bytes,
            kv_cache_write_bytes=self.kv_cache_write_bytes,
            moe_read_bytes=self.moe_read_bytes + read_bytes,
        )


@dataclass
class ForwardPerfRecord:
    """Useful model work attached to one timed device interval."""

    mode: str
    estimate: PerfEstimate
    num_tokens: int = 0
    stream_idx: int = -1
    sm_count: int = 0
    total_sm_count: int = 0
    expert_read_snapshot: Optional["ExpertReadSnapshot"] = None

    @property
    def sm_fraction(self) -> float:
        if self.sm_count <= 0 or self.total_sm_count <= 0:
            return 1.0
        return min(self.sm_count / self.total_sm_count, 1.0)

    @property
    def resolved_estimate(self) -> PerfEstimate:
        if self.expert_read_snapshot is None:
            return self.estimate
        return self.estimate.with_added_moe_read_bytes(
            self.expert_read_snapshot.read_bytes
        )


@dataclass(frozen=True)
class DenseQwen3PerfEstimator:
    hidden_size: int
    intermediate_size: int
    num_layers: int
    tp_size: int
    num_q_heads_per_gpu: int
    num_kv_heads_per_gpu: int
    head_dim: int
    vocab_size_per_gpu: int
    weight_bytes: int
    activation_bytes: int
    kv_cache_bytes: int
    tie_word_embeddings: bool

    @classmethod
    def from_model_config(cls, model_config, tp_size: int):
        hf_config = model_config.hf_text_config
        model_type = getattr(hf_config, "model_type", None)
        architectures = set(getattr(hf_config, "architectures", None) or [])
        if model_type != "qwen3" and "Qwen3ForCausalLM" not in architectures:
            raise ValueError(
                "DenseQwen3PerfEstimator only supports dense Qwen3 models, "
                f"got model_type={model_type!r}, architectures={sorted(architectures)!r}"
            )
        if getattr(hf_config, "num_experts", None) is not None:
            raise ValueError("DenseQwen3PerfEstimator does not support MoE models")
        if tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {tp_size}")

        intermediate_size = int(hf_config.intermediate_size)
        vocab_size = int(hf_config.vocab_size)
        if intermediate_size % tp_size != 0:
            raise ValueError(
                f"intermediate_size={intermediate_size} is not divisible by tp_size={tp_size}"
            )
        if vocab_size % tp_size != 0:
            raise ValueError(
                f"vocab_size={vocab_size} is not divisible by tp_size={tp_size}"
            )

        dtype_size = getattr(model_config.dtype, "itemsize", 2)
        if callable(dtype_size):
            dtype_size = dtype_size()

        return cls(
            hidden_size=int(hf_config.hidden_size),
            intermediate_size=intermediate_size,
            num_layers=int(hf_config.num_hidden_layers),
            tp_size=tp_size,
            num_q_heads_per_gpu=int(model_config.get_num_attention_heads(tp_size)),
            num_kv_heads_per_gpu=int(model_config.get_num_kv_heads(tp_size)),
            head_dim=int(getattr(hf_config, "head_dim", 0) or model_config.head_dim),
            vocab_size_per_gpu=vocab_size // tp_size,
            weight_bytes=int(dtype_size),
            activation_bytes=int(dtype_size),
            # TODO: use the resolved KV-cache dtype when quantized KV is enabled.
            kv_cache_bytes=int(dtype_size),
            tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
        )

    @property
    def attention_weight_elements_per_layer_per_gpu(self) -> int:
        qkv_out = self.head_dim * (
            self.num_q_heads_per_gpu + 2 * self.num_kv_heads_per_gpu
        )
        qkv = self.hidden_size * qkv_out
        out_proj = self.hidden_size * self.head_dim * self.num_q_heads_per_gpu
        return qkv + out_proj

    @property
    def mlp_weight_elements_per_layer_per_gpu(self) -> int:
        # gate_proj and up_proj are column-parallel; down_proj is row-parallel.
        return 3 * self.hidden_size * (self.intermediate_size // self.tp_size)

    @property
    def block_linear_flops_per_token_per_gpu(self) -> int:
        matrix_elements = (
            self.attention_weight_elements_per_layer_per_gpu
            + self.mlp_weight_elements_per_layer_per_gpu
        )
        return 2 * self.num_layers * matrix_elements

    @property
    def block_matrix_elements_per_layer_per_gpu(self) -> int:
        return (
            self.attention_weight_elements_per_layer_per_gpu
            + self.mlp_weight_elements_per_layer_per_gpu
        )

    @property
    def block_norm_elements_per_layer_per_gpu(self) -> int:
        # input/post-attention RMSNorm plus shared Q/K head-dimension RMSNorm.
        return 2 * self.hidden_size + 2 * self.head_dim

    @property
    def transformer_weight_bytes_per_gpu(self) -> int:
        matrix_elements = self.num_layers * self.block_matrix_elements_per_layer_per_gpu
        # Two RMSNorm vectors per block plus the final RMSNorm. Qwen3's Q/K
        # RMSNorm parameters have head_dim elements and are shared by all heads.
        norm_elements = (
            self.num_layers * self.block_norm_elements_per_layer_per_gpu
            + self.hidden_size
        )
        return (matrix_elements + norm_elements) * self.weight_bytes

    @property
    def lm_head_weight_bytes_per_gpu(self) -> int:
        return self.hidden_size * self.vocab_size_per_gpu * self.weight_bytes

    @property
    def kv_bytes_per_context_token_per_gpu(self) -> int:
        return (
            2
            * self.num_layers
            * self.num_kv_heads_per_gpu
            * self.head_dim
            * self.kv_cache_bytes
        )

    @property
    def attention_pair_flops_per_gpu(self) -> int:
        # QK dot product and AV product: two MACs, or four FLOPs, per pair.
        return 4 * self.num_layers * self.num_q_heads_per_gpu * self.head_dim

    def _lm_head_flops(self, num_logits: int) -> int:
        return 2 * num_logits * self.hidden_size * self.vocab_size_per_gpu

    def _common_read_bytes(self, num_input_tokens: int, num_logits: int) -> int:
        embedding_rows = num_input_tokens * self.hidden_size * self.activation_bytes
        lm_head = self.lm_head_weight_bytes_per_gpu if num_logits > 0 else 0
        return self.transformer_weight_bytes_per_gpu + embedding_rows + lm_head

    def estimate_prefill(
        self,
        prefix_lens: Sequence[int],
        extend_lens: Sequence[int],
        *,
        num_logits: int,
    ) -> PerfEstimate:
        return self.estimate_prefill_layer_range(
            prefix_lens,
            extend_lens,
            start_layer=0,
            end_layer=self.num_layers,
            num_logits=num_logits,
        )

    def estimate_prefill_layer_range(
        self,
        prefix_lens: Sequence[int],
        extend_lens: Sequence[int],
        *,
        start_layer: int,
        end_layer: int,
        num_logits: int,
    ) -> PerfEstimate:
        """Estimate one split-prefill slice over ``[start_layer, end_layer)``."""
        if len(prefix_lens) != len(extend_lens):
            raise ValueError("prefix_lens and extend_lens must have the same length")
        if any(x < 0 for x in prefix_lens) or any(x < 0 for x in extend_lens):
            raise ValueError("sequence lengths must be non-negative")
        if not 0 <= start_layer < end_layer <= self.num_layers:
            raise ValueError(
                "layer range must satisfy "
                f"0 <= start_layer < end_layer <= {self.num_layers}, got "
                f"[{start_layer}, {end_layer})"
            )

        tokens = sum(extend_lens)
        layer_count = end_layer - start_layer
        # Each request attends only to its own cached prefix and causal chunk.
        context_pairs = sum(
            q_len * prefix_len + q_len * (q_len + 1) // 2
            for prefix_len, q_len in zip(prefix_lens, extend_lens)
        )
        flops = (
            tokens * 2 * layer_count * self.block_matrix_elements_per_layer_per_gpu
            + context_pairs * 4 * layer_count * self.num_q_heads_per_gpu * self.head_dim
            + (self._lm_head_flops(num_logits) if end_layer == self.num_layers else 0)
        )
        # Minimum KV traffic: every active sequence's K/V span is consumed at
        # least once. Cached-prefix K/V comes from the cache; newly appended K/V
        # may be consumed from the cache or an equivalent attention input buffer
        # depending on the backend. Re-reads caused by tiling are deliberately
        # excluded from this model-level estimate.
        kv_read_tokens = sum(
            prefix_len + q_len
            for prefix_len, q_len in zip(prefix_lens, extend_lens)
            if q_len > 0
        )
        layer_weight_bytes = (
            layer_count
            * (
                self.block_matrix_elements_per_layer_per_gpu
                + self.block_norm_elements_per_layer_per_gpu
            )
            * self.weight_bytes
        )
        embedding_read_bytes = (
            tokens * self.hidden_size * self.activation_bytes if start_layer == 0 else 0
        )
        final_read_bytes = (
            self.hidden_size * self.weight_bytes
            + (self.lm_head_weight_bytes_per_gpu if num_logits > 0 else 0)
            if end_layer == self.num_layers
            else 0
        )
        kv_bytes_per_token_for_slice = (
            2
            * layer_count
            * self.num_kv_heads_per_gpu
            * self.head_dim
            * self.kv_cache_bytes
        )
        read_bytes = (
            layer_weight_bytes
            + embedding_read_bytes
            + final_read_bytes
            + kv_read_tokens * kv_bytes_per_token_for_slice
        )
        write_bytes = tokens * kv_bytes_per_token_for_slice
        kv_cache_read_bytes = kv_read_tokens * kv_bytes_per_token_for_slice
        return PerfEstimate(
            flops=float(flops),
            read_bytes=float(read_bytes),
            write_bytes=float(write_bytes),
            kv_cache_read_bytes=float(kv_cache_read_bytes),
            kv_cache_write_bytes=float(write_bytes),
        )

    def estimate_decode(
        self, seq_lens: Sequence[int], *, num_tokens: int, num_logits: int
    ) -> PerfEstimate:
        if num_tokens < 0 or num_logits < 0 or any(x < 0 for x in seq_lens):
            raise ValueError("token counts and sequence lengths must be non-negative")
        if num_tokens == 0:
            return PerfEstimate()

        context_pairs = sum(seq_lens)
        flops = (
            num_tokens * self.block_linear_flops_per_token_per_gpu
            + context_pairs * self.attention_pair_flops_per_gpu
            + self._lm_head_flops(num_logits)
        )
        read_bytes = (
            self._common_read_bytes(num_tokens, num_logits)
            + context_pairs * self.kv_bytes_per_context_token_per_gpu
        )
        write_bytes = num_tokens * self.kv_bytes_per_context_token_per_gpu
        kv_cache_read_bytes = context_pairs * self.kv_bytes_per_context_token_per_gpu
        return PerfEstimate(
            flops=float(flops),
            read_bytes=float(read_bytes),
            write_bytes=float(write_bytes),
            kv_cache_read_bytes=float(kv_cache_read_bytes),
            kv_cache_write_bytes=float(write_bytes),
        )


@dataclass(frozen=True)
class Qwen35MoePerfEstimator:
    """Useful-work estimator for BF16 Qwen3.5 hybrid-attention MoE.

    ``estimate_*`` returns the complete estimate except routed-expert weight
    reads.  Call ``add_routed_expert_reads`` once the GPU tracker has produced
    the sum of per-layer unique routed-expert counts for the timed forward.
    """

    hidden_size: int
    num_layers: int
    layer_types: tuple[str, ...]
    tp_size: int
    num_q_heads_per_gpu: int
    num_kv_heads_per_gpu: int
    head_dim: int
    linear_num_k_heads_per_gpu: int
    linear_num_v_heads_per_gpu: int
    linear_key_head_dim: int
    linear_value_head_dim: int
    linear_conv_kernel: int
    num_experts: int
    top_k: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    vocab_size_per_gpu: int
    weight_bytes: int
    activation_bytes: int
    kv_cache_bytes: int
    linear_state_bytes: int
    tie_word_embeddings: bool
    attn_output_gate: bool

    @classmethod
    def from_model_config(cls, model_config, tp_size: int):
        hf_config = model_config.hf_text_config
        model_type = getattr(hf_config, "model_type", None)
        if model_type != "qwen3_5_moe_text":
            raise ValueError(
                "Qwen35MoePerfEstimator only supports qwen3_5_moe_text, "
                f"got model_type={model_type!r}"
            )
        if tp_size != 1:
            raise ValueError(
                "Qwen35MoePerfEstimator is initially validated only for TP=1, "
                f"got tp_size={tp_size}"
            )

        dtype_size = getattr(model_config.dtype, "itemsize", 2)
        if callable(dtype_size):
            dtype_size = dtype_size()
        if int(dtype_size) != 2:
            raise ValueError(
                "Qwen35MoePerfEstimator is initially validated only for BF16/FP16 "
                f"weights, got itemsize={dtype_size}"
            )

        num_layers = int(hf_config.num_hidden_layers)
        raw_layer_types = tuple(
            getattr(hf_config, "layer_types", None)
            or getattr(hf_config, "layers_block_type")
        )
        if len(raw_layer_types) != num_layers:
            raise ValueError(
                f"expected {num_layers} layer types, got {len(raw_layer_types)}"
            )
        # Transformers checkpoints use ``full_attention`` while SGLang's
        # HybridLayerType normalizes the same layer kind to ``attention``.
        layer_types = tuple(
            "full_attention" if kind == "attention" else kind
            for kind in raw_layer_types
        )
        unsupported = set(layer_types) - {"linear_attention", "full_attention"}
        if unsupported:
            raise ValueError(f"unsupported Qwen3.5 layer types: {sorted(unsupported)}")

        return cls(
            hidden_size=int(hf_config.hidden_size),
            num_layers=num_layers,
            layer_types=layer_types,
            tp_size=tp_size,
            num_q_heads_per_gpu=int(model_config.get_num_attention_heads(tp_size)),
            num_kv_heads_per_gpu=int(model_config.get_num_kv_heads(tp_size)),
            head_dim=int(getattr(hf_config, "head_dim", 0) or model_config.head_dim),
            linear_num_k_heads_per_gpu=int(hf_config.linear_num_key_heads),
            linear_num_v_heads_per_gpu=int(hf_config.linear_num_value_heads),
            linear_key_head_dim=int(hf_config.linear_key_head_dim),
            linear_value_head_dim=int(hf_config.linear_value_head_dim),
            linear_conv_kernel=int(hf_config.linear_conv_kernel_dim),
            num_experts=int(hf_config.num_experts),
            top_k=int(hf_config.num_experts_per_tok),
            moe_intermediate_size=int(hf_config.moe_intermediate_size),
            shared_expert_intermediate_size=int(
                hf_config.shared_expert_intermediate_size
            ),
            vocab_size_per_gpu=int(hf_config.vocab_size),
            weight_bytes=int(dtype_size),
            activation_bytes=int(dtype_size),
            kv_cache_bytes=int(dtype_size),
            linear_state_bytes=4,
            tie_word_embeddings=bool(getattr(hf_config, "tie_word_embeddings", False)),
            attn_output_gate=bool(getattr(hf_config, "attn_output_gate", True)),
        )

    @property
    def full_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            i for i, kind in enumerate(self.layer_types) if kind == "full_attention"
        )

    @property
    def linear_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            i for i, kind in enumerate(self.layer_types) if kind == "linear_attention"
        )

    @property
    def routed_expert_weight_bytes_per_gpu(self) -> int:
        return 3 * self.hidden_size * self.moe_intermediate_size * self.weight_bytes

    @property
    def routed_expert_flops_per_assignment_per_gpu(self) -> int:
        return 6 * self.hidden_size * self.moe_intermediate_size

    @property
    def shared_expert_matrix_elements_per_layer_per_gpu(self) -> int:
        return 3 * self.hidden_size * self.shared_expert_intermediate_size

    @property
    def moe_static_matrix_elements_per_layer_per_gpu(self) -> int:
        # Router, shared SwiGLU expert, and the scalar shared-expert gate.
        return (
            self.hidden_size * self.num_experts
            + self.shared_expert_matrix_elements_per_layer_per_gpu
            + self.hidden_size
        )

    @property
    def moe_flops_per_token_per_layer_per_gpu(self) -> int:
        return 2 * self.moe_static_matrix_elements_per_layer_per_gpu + (
            self.top_k * self.routed_expert_flops_per_assignment_per_gpu
        )

    @property
    def full_attention_matrix_elements_per_layer_per_gpu(self) -> int:
        q_multiplier = 2 if self.attn_output_gate else 1
        qkv_out = self.head_dim * (
            q_multiplier * self.num_q_heads_per_gpu + 2 * self.num_kv_heads_per_gpu
        )
        qkv = self.hidden_size * qkv_out
        out = self.hidden_size * self.head_dim * self.num_q_heads_per_gpu
        return qkv + out

    @property
    def linear_attention_matrix_elements_per_layer_per_gpu(self) -> int:
        key_dim = self.linear_num_k_heads_per_gpu * self.linear_key_head_dim
        value_dim = self.linear_num_v_heads_per_gpu * self.linear_value_head_dim
        conv_dim = 2 * key_dim + value_dim
        qkvz = self.hidden_size * (2 * key_dim + 2 * value_dim)
        ba = self.hidden_size * (2 * self.linear_num_v_heads_per_gpu)
        conv = conv_dim * self.linear_conv_kernel
        out = value_dim * self.hidden_size
        return qkvz + ba + conv + out

    @property
    def linear_recurrence_flops_per_token_per_layer_per_gpu(self) -> int:
        # Decay state (K*V), predict (2*K*V), rank-1 update (2*K*V),
        # and output projection from state (2*K*V).
        return (
            7
            * self.linear_num_v_heads_per_gpu
            * self.linear_key_head_dim
            * self.linear_value_head_dim
        )

    @property
    def linear_state_bytes_per_request_per_layer_per_gpu(self) -> int:
        key_dim = self.linear_num_k_heads_per_gpu * self.linear_key_head_dim
        value_dim = self.linear_num_v_heads_per_gpu * self.linear_value_head_dim
        conv_dim = 2 * key_dim + value_dim
        conv_state = conv_dim * (self.linear_conv_kernel - 1) * self.activation_bytes
        recurrent_state = (
            self.linear_num_v_heads_per_gpu
            * self.linear_value_head_dim
            * self.linear_key_head_dim
            * self.linear_state_bytes
        )
        return conv_state + recurrent_state

    @property
    def full_kv_bytes_per_context_token_per_layer_per_gpu(self) -> int:
        return 2 * self.num_kv_heads_per_gpu * self.head_dim * self.kv_cache_bytes

    @property
    def full_attention_pair_flops_per_layer_per_gpu(self) -> int:
        return 4 * self.num_q_heads_per_gpu * self.head_dim

    @property
    def lm_head_weight_bytes_per_gpu(self) -> int:
        return self.hidden_size * self.vocab_size_per_gpu * self.weight_bytes

    def _layer_static_weight_bytes(self, layer_id: int) -> int:
        attention_matrix_elements = (
            self.full_attention_matrix_elements_per_layer_per_gpu
            if self.layer_types[layer_id] == "full_attention"
            else self.linear_attention_matrix_elements_per_layer_per_gpu
        )
        # Two block norms. Full attention has shared Q/K head-dimension norms;
        # GDN has one RMSNormGated vector of value-head dimension. A_log and
        # dt_bias are stored per value head (A_log in FP32, dt_bias in model dtype).
        if self.layer_types[layer_id] == "full_attention":
            auxiliary_weight_bytes = (
                2 * self.hidden_size + 2 * self.head_dim
            ) * self.weight_bytes
        else:
            auxiliary_weight_bytes = (
                2 * self.hidden_size + self.linear_value_head_dim
            ) * self.weight_bytes + self.linear_num_v_heads_per_gpu * (
                self.linear_state_bytes + self.weight_bytes
            )
        matrix_elements = (
            attention_matrix_elements
            + self.moe_static_matrix_elements_per_layer_per_gpu
        )
        return matrix_elements * self.weight_bytes + auxiliary_weight_bytes

    def _layer_flops_per_token(self, layer_id: int) -> int:
        if self.layer_types[layer_id] == "full_attention":
            attention_flops = 2 * self.full_attention_matrix_elements_per_layer_per_gpu
        else:
            attention_flops = (
                2 * self.linear_attention_matrix_elements_per_layer_per_gpu
                + self.linear_recurrence_flops_per_token_per_layer_per_gpu
            )
        return attention_flops + self.moe_flops_per_token_per_layer_per_gpu

    def add_routed_expert_reads(
        self, estimate: PerfEstimate, total_unique_experts_across_layers: int
    ) -> PerfEstimate:
        if total_unique_experts_across_layers < 0:
            raise ValueError("unique expert count must be non-negative")
        return estimate.with_added_moe_read_bytes(
            total_unique_experts_across_layers * self.routed_expert_weight_bytes_per_gpu
        )

    def estimate_prefill(
        self,
        prefix_lens: Sequence[int],
        extend_lens: Sequence[int],
        *,
        num_logits: int,
    ) -> PerfEstimate:
        return self.estimate_prefill_layer_range(
            prefix_lens,
            extend_lens,
            start_layer=0,
            end_layer=self.num_layers,
            num_logits=num_logits,
        )

    def estimate_prefill_layer_range(
        self,
        prefix_lens: Sequence[int],
        extend_lens: Sequence[int],
        *,
        start_layer: int,
        end_layer: int,
        num_logits: int,
    ) -> PerfEstimate:
        if len(prefix_lens) != len(extend_lens):
            raise ValueError("prefix_lens and extend_lens must have the same length")
        if any(x < 0 for x in prefix_lens) or any(x < 0 for x in extend_lens):
            raise ValueError("sequence lengths must be non-negative")
        if not 0 <= start_layer < end_layer <= self.num_layers:
            raise ValueError(
                "layer range must satisfy "
                f"0 <= start_layer < end_layer <= {self.num_layers}, got "
                f"[{start_layer}, {end_layer})"
            )

        tokens = sum(extend_lens)
        layer_ids = range(start_layer, end_layer)
        full_layers = sum(self.layer_types[i] == "full_attention" for i in layer_ids)
        linear_layers = (end_layer - start_layer) - full_layers
        context_pairs = sum(
            q_len * prefix_len + q_len * (q_len + 1) // 2
            for prefix_len, q_len in zip(prefix_lens, extend_lens)
        )
        flops = tokens * sum(
            self._layer_flops_per_token(i) for i in range(start_layer, end_layer)
        )
        flops += (
            context_pairs
            * full_layers
            * self.full_attention_pair_flops_per_layer_per_gpu
        )
        if end_layer == self.num_layers:
            flops += 2 * num_logits * self.hidden_size * self.vocab_size_per_gpu

        layer_weight_read = sum(
            self._layer_static_weight_bytes(i) for i in range(start_layer, end_layer)
        )
        moe_static_read = (
            (end_layer - start_layer)
            * self.moe_static_matrix_elements_per_layer_per_gpu
            * self.weight_bytes
        )
        embedding_read = (
            tokens * self.hidden_size * self.activation_bytes if start_layer == 0 else 0
        )
        final_read = (
            self.hidden_size * self.weight_bytes
            + (self.lm_head_weight_bytes_per_gpu if num_logits > 0 else 0)
            if end_layer == self.num_layers
            else 0
        )
        full_kv_read_tokens = sum(
            prefix_len + q_len
            for prefix_len, q_len in zip(prefix_lens, extend_lens)
            if q_len > 0
        )
        state_initial_reads = sum(
            prefix_len > 0 and q_len > 0
            for prefix_len, q_len in zip(prefix_lens, extend_lens)
        )
        active_requests = sum(q_len > 0 for q_len in extend_lens)
        kv_cache_read_bytes = (
            full_kv_read_tokens
            * full_layers
            * self.full_kv_bytes_per_context_token_per_layer_per_gpu
            + state_initial_reads
            * linear_layers
            * self.linear_state_bytes_per_request_per_layer_per_gpu
        )
        read_bytes = (
            layer_weight_read + embedding_read + final_read + kv_cache_read_bytes
        )
        write_bytes = (
            tokens
            * full_layers
            * self.full_kv_bytes_per_context_token_per_layer_per_gpu
            + active_requests
            * linear_layers
            * self.linear_state_bytes_per_request_per_layer_per_gpu
        )
        return PerfEstimate(
            flops=float(flops),
            read_bytes=float(read_bytes),
            write_bytes=float(write_bytes),
            kv_cache_read_bytes=float(kv_cache_read_bytes),
            kv_cache_write_bytes=float(write_bytes),
            moe_read_bytes=float(moe_static_read),
        )

    def estimate_decode(
        self, seq_lens: Sequence[int], *, num_tokens: int, num_logits: int
    ) -> PerfEstimate:
        if num_tokens < 0 or num_logits < 0 or any(x < 0 for x in seq_lens):
            raise ValueError("token counts and sequence lengths must be non-negative")
        if num_tokens == 0:
            return PerfEstimate()
        if len(seq_lens) < num_tokens:
            raise ValueError(
                f"decode needs at least {num_tokens} sequence lengths, "
                f"got {len(seq_lens)}"
            )

        full_layers = len(self.full_layer_ids)
        linear_layers = len(self.linear_layer_ids)
        context_pairs = sum(seq_lens[:num_tokens])
        flops = num_tokens * sum(
            self._layer_flops_per_token(i) for i in range(self.num_layers)
        )
        flops += (
            context_pairs
            * full_layers
            * self.full_attention_pair_flops_per_layer_per_gpu
        )
        flops += 2 * num_logits * self.hidden_size * self.vocab_size_per_gpu

        static_read = sum(
            self._layer_static_weight_bytes(i) for i in range(self.num_layers)
        )
        embedding_read = num_tokens * self.hidden_size * self.activation_bytes
        final_read = self.hidden_size * self.weight_bytes + (
            self.lm_head_weight_bytes_per_gpu if num_logits > 0 else 0
        )
        linear_state_traffic = (
            num_tokens
            * linear_layers
            * self.linear_state_bytes_per_request_per_layer_per_gpu
        )
        kv_cache_read_bytes = (
            context_pairs
            * full_layers
            * self.full_kv_bytes_per_context_token_per_layer_per_gpu
            + linear_state_traffic
        )
        moe_static_read = (
            self.num_layers
            * self.moe_static_matrix_elements_per_layer_per_gpu
            * self.weight_bytes
        )
        read_bytes = static_read + embedding_read + final_read + kv_cache_read_bytes
        write_bytes = (
            num_tokens
            * full_layers
            * self.full_kv_bytes_per_context_token_per_layer_per_gpu
            + linear_state_traffic
        )
        return PerfEstimate(
            flops=float(flops),
            read_bytes=float(read_bytes),
            write_bytes=float(write_bytes),
            kv_cache_read_bytes=float(kv_cache_read_bytes),
            kv_cache_write_bytes=float(write_bytes),
            moe_read_bytes=float(moe_static_read),
        )
