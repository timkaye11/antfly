#!/usr/bin/env python3
"""Materialize a provenance-locked BoolQ dataset for Gemma4 GRPO.

The materializer emits disjoint train/evaluation JSONL files for the typed
``text-grpo`` recipe. It selects balanced yes/no examples in source order,
requires the rendered Gemma4 prompt plus the configured rollout budget to fit
without truncation, and binds source Parquet, tokenizer, materializer, and
output digests in one manifest. The verifier target remains one token while
the sampled completion may contain multiple tokens.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
SCHEMA_VERSION = "antfly_gemma4_grpo_boolq_materialization/v1"
EXPECTED_DATASET_ID = "google/boolq"
EXPECTED_TRAIN_SPLIT = "train"
EXPECTED_EVAL_SPLIT = "validation"
MAX_COMPLETION_TOKENS = 32


class MaterializationError(RuntimeError):
    """The pinned source or deterministic selection contract was violated."""


def validate_length_contract(max_seq_len: int, max_completion_tokens: int) -> None:
    if max_seq_len < 16:
        raise MaterializationError("max sequence length must be at least 16")
    if not 1 <= max_completion_tokens <= MAX_COMPLETION_TOKENS:
        raise MaterializationError(
            f"max completion tokens must be in [1, {MAX_COMPLETION_TOKENS}]"
        )
    if max_completion_tokens >= max_seq_len:
        raise MaterializationError(
            "max completion tokens must be smaller than max sequence length"
        )


@dataclass(frozen=True)
class BoolQExample:
    source_split: str
    source_row_index: int
    source_id: str
    prompt: str
    target: str
    prompt_tokens: int
    target_tokens: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_gemma4_final_prompt(prompt: str) -> str:
    return (
        "<bos><|turn>user\n"
        + prompt
        + "<turn|>\n<|turn>model\n<|channel>final\n<channel|>"
    )


def boolq_prompt(passage: str, question: str) -> str:
    normalized_question = question.strip()
    if not normalized_question.endswith("?"):
        normalized_question += "?"
    return (
        "Read the passage and answer the question with exactly one word: yes or no.\n\n"
        f"Passage: {passage.strip()}\n\nQuestion: {normalized_question}\nAnswer:"
    )


def source_id(split: str, row_index: int, row: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "split": split,
            "row_index": row_index,
            "question": row.get("question"),
            "passage": row.get("passage"),
            "answer": row.get("answer"),
        }
    )


def parse_row(
    row: Mapping[str, Any],
    *,
    split: str,
    row_index: int,
    tokenizer: Any,
    max_seq_len: int,
    max_completion_tokens: int,
) -> BoolQExample | None:
    passage = row.get("passage")
    question = row.get("question")
    answer = row.get("answer")
    if not isinstance(passage, str) or not passage.strip():
        raise MaterializationError(f"{split} row {row_index}: invalid passage")
    if not isinstance(question, str) or not question.strip():
        raise MaterializationError(f"{split} row {row_index}: invalid question")
    if not isinstance(answer, bool):
        raise MaterializationError(f"{split} row {row_index}: answer is not boolean")
    natural_prompt = boolq_prompt(passage, question)
    prompt = render_gemma4_final_prompt(natural_prompt)
    target = "yes" if answer else "no"
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False).ids
    target_ids = tokenizer.encode(target, add_special_tokens=False).ids
    if len(target_ids) != 1:
        raise MaterializationError(
            f"Gemma4 target {target!r} is not exactly one tokenizer token"
        )
    if not prompt_ids or len(prompt_ids) + max_completion_tokens > max_seq_len:
        return None
    return BoolQExample(
        source_split=split,
        source_row_index=row_index,
        source_id=source_id(split, row_index, row),
        prompt=prompt,
        target=target,
        prompt_tokens=len(prompt_ids),
        target_tokens=len(target_ids),
    )


def select_balanced(
    rows: Iterable[Mapping[str, Any]],
    *,
    split: str,
    tokenizer: Any,
    examples: int,
    max_seq_len: int,
    max_completion_tokens: int,
) -> tuple[BoolQExample, ...]:
    if examples < 2 or examples % 2 != 0:
        raise MaterializationError("example count must be even and at least two")
    quota = examples // 2
    selected: list[BoolQExample] = []
    counts = {"yes": 0, "no": 0}
    for row_index, row in enumerate(rows):
        candidate = parse_row(
            row,
            split=split,
            row_index=row_index,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_completion_tokens=max_completion_tokens,
        )
        if candidate is None or counts[candidate.target] >= quota:
            continue
        selected.append(candidate)
        counts[candidate.target] += 1
        if len(selected) == examples:
            break
    if len(selected) != examples:
        raise MaterializationError(
            f"{split}: source cannot provide {quota} admitted rows per label; got {counts}"
        )
    return tuple(selected)


def _write_jsonl(path: Path, examples: Sequence[BoolQExample]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for example in examples:
            payload = {
                "prompt": example.prompt,
                "target": example.target,
                "metadata": {
                    "source": EXPECTED_DATASET_ID,
                    "source_split": example.source_split,
                    "source_row_index": example.source_row_index,
                    "source_id": example.source_id,
                    "prompt_tokens": example.prompt_tokens,
                    "target_tokens": example.target_tokens,
                },
            }
            handle.write(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )


def write_outputs(
    *,
    output_dir: Path,
    train_parquet: Path,
    eval_parquet: Path,
    model_dir: Path,
    revision: str,
    train_rows: int,
    eval_rows: int,
    train_examples: Sequence[BoolQExample],
    eval_examples: Sequence[BoolQExample],
    max_seq_len: int,
    max_completion_tokens: int,
    dependency_versions: Mapping[str, str],
) -> Mapping[str, Any]:
    train_ids = {example.source_id for example in train_examples}
    eval_ids = {example.source_id for example in eval_examples}
    if train_ids & eval_ids:
        raise MaterializationError("train and evaluation source identities overlap")
    output_dir.mkdir(parents=True, exist_ok=False)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    _write_jsonl(train_path, train_examples)
    _write_jsonl(eval_path, eval_examples)
    tokenizer_files = (model_dir / "tokenizer.json", model_dir / "tokenizer_config.json")
    for path in tokenizer_files:
        if not path.is_file():
            raise MaterializationError(f"missing tokenizer input: {path}")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "dataset": {
            "repo_id": EXPECTED_DATASET_ID,
            "revision": revision,
            "train": {
                "split": EXPECTED_TRAIN_SPLIT,
                "source_file": train_parquet.name,
                "source_file_sha256": sha256_file(train_parquet),
                "source_rows": train_rows,
                "materialized_jsonl": train_path.name,
                "materialized_jsonl_sha256": sha256_file(train_path),
            },
            "evaluation": {
                "split": EXPECTED_EVAL_SPLIT,
                "source_file": eval_parquet.name,
                "source_file_sha256": sha256_file(eval_parquet),
                "source_rows": eval_rows,
                "materialized_jsonl": eval_path.name,
                "materialized_jsonl_sha256": sha256_file(eval_path),
            },
            "selection_policy": {
                "ordering": "first-admitted-source-order",
                "dataset_format": "rendered-text-grpo",
                "response_channel": "final",
                "labels": {"yes": len(train_examples) // 2, "no": len(train_examples) // 2},
                "evaluation_labels": {"yes": len(eval_examples) // 2, "no": len(eval_examples) // 2},
                "rendered_prompt_truncation": "forbidden",
                "target_tokens": 1,
                "max_seq_len": max_seq_len,
                "max_completion_tokens": max_completion_tokens,
            },
        },
        "model_dir": str(model_dir),
        "tokenizer_files": {path.name: sha256_file(path) for path in tokenizer_files},
        "train_jsonl": str(train_path),
        "eval_jsonl": str(eval_path),
        "train_source_ids": [example.source_id for example in train_examples],
        "eval_source_ids": [example.source_id for example in eval_examples],
        "train_source_row_indices": [example.source_row_index for example in train_examples],
        "eval_source_row_indices": [example.source_row_index for example in eval_examples],
        "train_prompt_tokens": [example.prompt_tokens for example in train_examples],
        "eval_prompt_tokens": [example.prompt_tokens for example in eval_examples],
        "materializer": str(SCRIPT_PATH),
        "materializer_sha256": sha256_file(SCRIPT_PATH),
        "dependency_versions": dict(dependency_versions),
    }
    manifest["semantic_sha256"] = canonical_sha256(manifest)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def materialize(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.dataset_id != EXPECTED_DATASET_ID:
        raise MaterializationError(f"dataset must be {EXPECTED_DATASET_ID}")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise MaterializationError("revision must be a full lowercase Git commit")
    validate_length_contract(args.max_seq_len, args.max_completion_tokens)
    train_parquet = args.train_parquet.expanduser().resolve()
    eval_parquet = args.eval_parquet.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not train_parquet.is_file() or not eval_parquet.is_file():
        raise MaterializationError("pinned train/evaluation Parquet input is missing")
    if not model_dir.is_dir():
        raise MaterializationError("model directory is missing")
    if args.train_sha256 and sha256_file(train_parquet) != args.train_sha256:
        raise MaterializationError("train Parquet SHA-256 does not match")
    if args.eval_sha256 and sha256_file(eval_parquet) != args.eval_sha256:
        raise MaterializationError("evaluation Parquet SHA-256 does not match")
    try:
        import pyarrow
        import pyarrow.parquet as pq
        import tokenizers
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise MaterializationError("materialization requires pyarrow and tokenizers") from exc
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    train_table = pq.read_table(train_parquet)
    eval_table = pq.read_table(eval_parquet)
    train_examples = select_balanced(
        train_table.to_pylist(),
        split=EXPECTED_TRAIN_SPLIT,
        tokenizer=tokenizer,
        examples=args.examples,
        max_seq_len=args.max_seq_len,
        max_completion_tokens=args.max_completion_tokens,
    )
    eval_examples = select_balanced(
        eval_table.to_pylist(),
        split=EXPECTED_EVAL_SPLIT,
        tokenizer=tokenizer,
        examples=args.eval_examples,
        max_seq_len=args.max_seq_len,
        max_completion_tokens=args.max_completion_tokens,
    )
    return write_outputs(
        output_dir=output_dir,
        train_parquet=train_parquet,
        eval_parquet=eval_parquet,
        model_dir=model_dir,
        revision=args.revision,
        train_rows=train_table.num_rows,
        eval_rows=eval_table.num_rows,
        train_examples=train_examples,
        eval_examples=eval_examples,
        max_seq_len=args.max_seq_len,
        max_completion_tokens=args.max_completion_tokens,
        dependency_versions={
            "pyarrow": pyarrow.__version__,
            "tokenizers": tokenizers.__version__,
        },
    )


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--train-parquet", type=Path, required=True)
    result.add_argument("--eval-parquet", type=Path, required=True)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--dataset-id", default=EXPECTED_DATASET_ID)
    result.add_argument("--revision", required=True)
    result.add_argument("--train-sha256")
    result.add_argument("--eval-sha256")
    result.add_argument("--examples", type=int, default=4)
    result.add_argument("--eval-examples", type=int, default=4)
    result.add_argument("--max-seq-len", type=int, default=128)
    result.add_argument("--max-completion-tokens", type=int, default=1)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        manifest = materialize(parser().parse_args(argv))
    except (MaterializationError, OSError, ValueError) as exc:
        print(f"Gemma4 GRPO BoolQ materialization error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
