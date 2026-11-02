#!/usr/bin/env python3
"""Prepare and sample tokenized DeepResearch prompts for microbenchmarks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np


DEFAULT_HOLDOUT_TASKS = ("dr_004", "dr_008", "dr_014", "dr_018")


def _normalize_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI JSON-string tool arguments to Qwen template mappings."""
    normalized = copy.deepcopy(messages)
    for message in normalized:
        for tool_call in message.get("tool_calls") or []:
            function = tool_call.get("function", tool_call)
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                parsed = json.loads(arguments)
                if not isinstance(parsed, dict):
                    raise ValueError("Tool-call arguments JSON must decode to an object")
                function["arguments"] = parsed
    return normalized


def _llm_rows(trace_dir: Path):
    for task_dir in sorted((trace_dir / "tasks").iterdir()):
        if not task_dir.is_dir():
            continue
        for session_path in sorted((task_dir / "sessions").glob("*.jsonl")):
            with session_path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    if row.get("kind") == "llm":
                        yield task_dir.name, session_path.stem, row


def prepare_corpus(
    trace_dir: Path,
    model_path: Path,
    output_dir: Path,
    holdout_tasks: set[str],
) -> dict:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, local_files_only=True, trust_remote_code=True
    )
    all_tokens: list[int] = []
    entries = []
    length_deltas = []
    for task_id, session_id, row in _llm_rows(trace_dir):
        request = row["input"]
        token_ids = tokenizer.apply_chat_template(
            _normalize_messages(request["messages"]),
            tools=request.get("tools"),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
            return_dict=False,
        )
        if not token_ids:
            raise ValueError(f"Empty rendered prompt: {task_id}/{session_id}")
        offset = len(all_tokens)
        all_tokens.extend(int(token) for token in token_ids)
        recorded_length = int(row.get("input_tokens") or 0)
        length_deltas.append(len(token_ids) - recorded_length)
        entries.append(
            {
                "task_id": task_id,
                "session_id": session_id,
                "round_id": int(row["round_id"]),
                "split": "holdout" if task_id in holdout_tasks else "train",
                "offset": offset,
                "length": len(token_ids),
                "recorded_input_tokens": recorded_length,
                "has_tools": bool(request.get("tools")),
            }
        )

    token_array = np.asarray(all_tokens, dtype=np.int32)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "token_ids.npy", token_array, allow_pickle=False)
    token_sha256 = hashlib.sha256(token_array.tobytes()).hexdigest()
    split_counts = {
        split: sum(entry["split"] == split for entry in entries)
        for split in ("train", "holdout")
    }
    manifest = {
        "format": "highway.deepresearch-token-corpus.v1",
        "source_trace_dir": str(trace_dir.resolve()),
        "model_path": str(model_path.resolve()),
        "chat_template": {
            "add_generation_prompt": True,
            "enable_thinking": True,
            "tools_preserved": True,
        },
        "holdout_tasks": sorted(holdout_tasks),
        "entries": entries,
        "entry_count": len(entries),
        "split_counts": split_counts,
        "token_count": int(token_array.size),
        "token_sha256": token_sha256,
        "rendered_minus_recorded_input_tokens": {
            "min": min(length_deltas),
            "max": max(length_deltas),
            "mean": float(np.mean(length_deltas)),
            "median": float(np.median(length_deltas)),
        },
    }
    (output_dir / "index.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def load_token_batch(
    corpus_dir: str | Path,
    *,
    split: str,
    batch_size: int,
    input_len: int,
    seed: int,
) -> tuple[list[list[int]], list[dict]]:
    """Select distinct prompts and return their last ``input_len`` tokens."""
    corpus_dir = Path(corpus_dir)
    manifest = json.loads((corpus_dir / "index.json").read_text())
    candidates = [
        entry
        for entry in manifest["entries"]
        if entry["split"] == split and int(entry["length"]) >= input_len
    ]
    if len(candidates) < batch_size:
        raise ValueError(
            f"Only {len(candidates)} {split} prompts contain at least {input_len} "
            f"tokens; need {batch_size}"
        )
    point_seed = seed + batch_size * 1_000_003 + input_len * 101
    order = np.random.default_rng(point_seed).permutation(len(candidates))[:batch_size]
    selected = [candidates[int(index)] for index in order]
    token_array = np.load(corpus_dir / "token_ids.npy", mmap_mode="r")
    batches = []
    for entry in selected:
        end = int(entry["offset"]) + int(entry["length"])
        start = end - input_len
        batches.append(token_array[start:end].astype(np.int64).tolist())
    return batches, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--holdout-tasks", nargs="+", default=list(DEFAULT_HOLDOUT_TASKS)
    )
    args = parser.parse_args()
    manifest = prepare_corpus(
        args.trace_dir,
        args.model_path,
        args.output_dir,
        set(args.holdout_tasks),
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "entries"}, indent=2))


if __name__ == "__main__":
    main()
