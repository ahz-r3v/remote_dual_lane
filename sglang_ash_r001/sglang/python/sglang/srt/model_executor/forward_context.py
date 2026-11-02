"""Per-forward-call control context.

Owns ``ForwardContext`` — a frozen dataclass holding control configs the model
layer reads at depth via ``get_forward_context()``. The only mandatory field
today is ``attn_backend``; pool refs are derived from ``attn_backend.*``
(every backend caches them at ``__init__``), so a published ``ForwardContext``
is enough to resolve the active pools without a separate global.

``ModelRunner._forward_raw`` publishes a fresh ``ForwardContext`` for the
duration of each forward; callers that need a per-call override (PDmux
per-stream backend, frozen-KV MTP draft loop, TBO per-child dispatch) use
``dataclasses.replace`` and wrap the override scope with ``forward_context()``.

Distinct from ``sglang.srt.compilation.piecewise_context_manager.ForwardContext``,
which collects compilation-time refs for the piecewise CUDA graph backend.

Concurrency: ``_current`` is thread-local. This lets experimental decode lane
multiplexing submit two forwards from two host threads without racing on the
active attention backend.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
    from sglang.srt.mem_cache.memory_pool import KVCache, ReqToTokenPool


@dataclass(frozen=True, slots=True)
class ForwardContext:
    """Per-forward-call control configs. Read via ``get_forward_context()``;
    extend by adding fields here. Frozen so accidental mutation raises at
    write time — use ``dataclasses.replace`` for per-call overrides."""

    attn_backend: AttentionBackend
    attention_active_cta_limit: int = 0
    attention_active_cta_ratio: float = 0.0
    attention_active_sm_count: int = 0
    attention_active_cta_per_sm: int = 0
    moe_active_cta_limit: int = 0
    moe_active_cta_ratio: float = 0.0
    moe_active_sm_count: int = 0
    linear_attn_active_cta_limit: int = 0
    low_resource_linear: bool = False


_THREAD_LOCAL = threading.local()


def set_forward_context(ctx: Optional[ForwardContext]) -> Optional[ForwardContext]:
    """Set the active context; return the previous one for explicit
    save/restore. Prefer the ``forward_context()`` context manager."""
    prev = getattr(_THREAD_LOCAL, "current", None)
    _THREAD_LOCAL.current = ctx
    return prev


def has_forward_context() -> bool:
    return getattr(_THREAD_LOCAL, "current", None) is not None


def get_forward_context() -> ForwardContext:
    current = getattr(_THREAD_LOCAL, "current", None)
    assert current is not None, (
        "no forward context active — call forward_context(...) or set_forward_context(...) "
        "before reading get_forward_context()."
    )
    return current


def get_attn_backend() -> AttentionBackend:
    return get_forward_context().attn_backend


def get_token_to_kv_pool() -> KVCache:
    return get_attn_backend().token_to_kv_pool


def get_req_to_token_pool() -> ReqToTokenPool:
    return get_attn_backend().req_to_token_pool


@contextmanager
def forward_context(ctx: ForwardContext):
    prev = set_forward_context(ctx)
    try:
        yield
    finally:
        set_forward_context(prev)
