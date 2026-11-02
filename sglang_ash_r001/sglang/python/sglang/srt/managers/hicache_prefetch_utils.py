from __future__ import annotations

from typing import Sequence, TypeVar


T = TypeVar("T")


def cacheable_unmatched_tokens(
    fill_ids: Sequence[T], matched_len: int, max_prefix_len: int
) -> Sequence[T]:
    """Return the unmatched range that a later radix lookup may consume."""

    start = max(0, min(int(matched_len), len(fill_ids)))
    stop = max(start, min(int(max_prefix_len), len(fill_ids)))
    return fill_ids[start:stop]
