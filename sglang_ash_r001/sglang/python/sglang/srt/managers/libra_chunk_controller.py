"""TBT-SLO-aware mixed-prefill budget for the Libra baseline port."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass(frozen=True)
class LibraChunkCoefficients:
    prefill_p0_ms: float
    prefill_p1: float
    prefill_p2: float
    decode_t0_ms: float
    decode_tB_ms: float
    decode_tK_ms: float
    mixed_intercept_ms: float
    mixed_prefill_scale: float
    mixed_decode_scale: float
    kv_gb_per_token: float = 2.048e-5
    full_sm_count: int = 132


class LibraChunkController:
    def __init__(
        self,
        coefficients: LibraChunkCoefficients,
        *,
        tbt_slo_ms: float,
        max_chunk_tokens: int,
        page_size: int,
    ) -> None:
        if tbt_slo_ms <= 0 or max_chunk_tokens <= 0 or page_size <= 0:
            raise ValueError("Libra chunk-controller limits must be positive")
        self.coefficients = coefficients
        self.tbt_slo_ms = tbt_slo_ms
        self.max_chunk_tokens = max_chunk_tokens
        self.page_size = page_size

    @classmethod
    def from_env(
        cls, *, max_chunk_tokens: int, page_size: int
    ) -> Optional["LibraChunkController"]:
        delta_path = os.environ.get("SGLANG_LIBRA_DELTA_PROFILE", "").strip()
        if not delta_path:
            return None
        prefill_path = os.environ.get("SGLANG_LIBRA_PREFILL_MODEL", "").strip()
        decode_path = os.environ.get("SGLANG_LIBRA_DECODE_MODEL", "").strip()
        if not prefill_path or not decode_path:
            raise ValueError(
                "SGLANG_LIBRA_PREFILL_MODEL and SGLANG_LIBRA_DECODE_MODEL are required"
            )
        delta = json.loads(Path(delta_path).read_text(encoding="utf-8"))
        prefill = json.loads(Path(prefill_path).read_text(encoding="utf-8"))
        decode = json.loads(Path(decode_path).read_text(encoding="utf-8"))
        pc = prefill["coefficients"]
        dc = decode["coefficients"]
        mixed = delta["mixed_batch"]
        coefficients = LibraChunkCoefficients(
            prefill_p0_ms=float(pc["p0_fixed_ms"]),
            prefill_p1=float(pc["p1_linear_tflops_per_sm"]),
            prefill_p2=float(pc["p2_diminishing_tflops_per_sm2"]),
            decode_t0_ms=float(dc["t0_ms"]),
            decode_tB_ms=float(dc["tB_ms_per_request"]),
            decode_tK_ms=float(dc["tK_ms_per_attention_kv_gb"]),
            mixed_intercept_ms=float(mixed["intercept_ms"]),
            mixed_prefill_scale=float(mixed["prefill_scale"]),
            mixed_decode_scale=float(mixed["decode_scale"]),
            full_sm_count=int(os.environ.get("SGLANG_LIBRA_FULL_SM_COUNT", "132")),
        )
        return cls(
            coefficients,
            tbt_slo_ms=float(os.environ.get("SGLANG_LIBRA_TBT_SLO_MS", "50")),
            max_chunk_tokens=max_chunk_tokens,
            page_size=page_size,
        )

    def prefill_ms(self, prefix_tokens: int, extend_tokens: int) -> float:
        c = self.coefficients
        extend = max(1, extend_tokens)
        prefix = max(0, prefix_tokens)
        pairs = extend * prefix + extend * (extend + 1) / 2
        work = 0.00498548736 * extend + 1.6384e-7 * pairs + 0.00101711872
        peak = c.prefill_p1 * c.full_sm_count - c.prefill_p2 * c.full_sm_count**2
        if peak <= 0:
            raise ValueError("Libra prefill model predicts non-positive throughput")
        return c.prefill_p0_ms + 1000.0 * work / peak

    def decode_ms(self, contexts: Sequence[int]) -> float:
        if not contexts:
            return 0.0
        c = self.coefficients
        return (
            c.decode_t0_ms
            + c.decode_tB_ms * len(contexts)
            + c.decode_tK_ms * sum(contexts) * c.kv_gb_per_token
        )

    def mixed_ms(
        self, prefix_tokens: int, extend_tokens: int, contexts: Sequence[int]
    ) -> float:
        prefill = self.prefill_ms(prefix_tokens, extend_tokens)
        decode = self.decode_ms(contexts)
        if decode <= 0:
            return prefill
        c = self.coefficients
        return (
            c.mixed_intercept_ms
            + c.mixed_prefill_scale * prefill
            + c.mixed_decode_scale * decode
        )

    def choose(self, prefix_tokens: int, decode_contexts: Sequence[int]) -> int:
        """Return a page-aligned aggregate prefill budget for one iteration."""

        if not decode_contexts:
            return self.max_chunk_tokens
        minimum = min(self.page_size, self.max_chunk_tokens)
        if self.mixed_ms(prefix_tokens, minimum, decode_contexts) > self.tbt_slo_ms:
            # No page-sized mixed prefill can meet the target.  The scheduler
            # must run a decode-only iteration and retry after decode state
            # changes, rather than knowingly violate the TBT SLO.
            return 0
        pages = max(1, self.max_chunk_tokens // self.page_size)
        lo, hi = 1, pages
        while lo < hi:
            mid = (lo + hi + 1) // 2
            tokens = mid * self.page_size
            if self.mixed_ms(prefix_tokens, tokens, decode_contexts) <= self.tbt_slo_ms:
                lo = mid
            else:
                hi = mid - 1
        return min(self.max_chunk_tokens, lo * self.page_size)
