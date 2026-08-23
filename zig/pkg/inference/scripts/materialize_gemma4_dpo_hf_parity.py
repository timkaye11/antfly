#!/usr/bin/env python3
"""Materialize a provenance-locked Hugging Face DPO parity dataset.

The output contains both the text JSONL consumed by Antfly and an exact-token
benchmark case consumed by ``run_gemma4_dpo_mlx_benchmark.py``.  Dataset and
tokenizer imports stay lazy so the contract helpers can be unit-tested without
the optional materialization dependencies.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_TOKENIZER_CONTRACT = (
    SCRIPT_DIR.parent / "testdata" / "gemma4_dpo_e2b_seq128_benchmark.json"
)
OUTPUT_SCHEMA_VERSION = "antfly_gemma4_dpo_benchmark_dataset/v1"
MANIFEST_SCHEMA_VERSION = "antfly.hf_dpo_parity_materialization/v1"
EXPECTED_DATASET_ID = "HuggingFaceH4/ultrafeedback_binarized"
EXPECTED_SPLIT = "test_prefs"
FIXED_PROTOCOL = {"cold": 1, "first": 1, "warmup": 3, "measured": 20}
LENGTH_BUCKETS = ((64, 128), (128, 224), (224, 320), (320, 416), (416, 513))
CONTRACT_PROBE_PROMPT = "Answer with one word: yes or no?"


class MaterializationError(RuntimeError):
    """The source, tokenizer, selection, or output violated the parity contract."""


@dataclass(frozen=True)
class TokenizedPreference:
    source_row_index: int
    source_id: str
    score_chosen: float
    score_rejected: float
    prompt: str
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    chosen_text: str
    chosen_token_ids: tuple[int, ...]
    rejected_text: str
    rejected_token_ids: tuple[int, ...]

    @property
    def total_tokens(self) -> int:
        return len(self.prompt_token_ids) + max(
            len(self.chosen_token_ids), len(self.rejected_token_ids)
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def render_gemma4_user_prompt(prompt: str) -> str:
    return (
        "<bos><|turn>user\n"
        + prompt
        + "<turn|>\n<|turn>model\n<|channel>thought\n<channel|>"
    )


def _finite_score(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MaterializationError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MaterializationError(f"{where} must be finite")
    return result


def _messages(value: Any, where: str, prompt: str) -> tuple[str, str]:
    if not isinstance(value, list) or len(value) != 2:
        raise MaterializationError(f"{where} must contain one user and one assistant")
    user, assistant = value
    if not isinstance(user, dict) or not isinstance(assistant, dict):
        raise MaterializationError(f"{where} messages must be objects")
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        raise MaterializationError(f"{where} roles are not user/assistant")
    if user.get("content") != prompt:
        raise MaterializationError(f"{where} user content differs from prompt")
    assistant_text = assistant.get("content")
    if not isinstance(assistant_text, str) or not assistant_text:
        raise MaterializationError(f"{where} assistant content must be non-empty")
    return prompt, assistant_text


def tokenize_source_row(
    row: Mapping[str, Any], source_row_index: int, tokenizer: Any
) -> TokenizedPreference:
    prompt = row.get("prompt")
    source_id = row.get("prompt_id")
    if not isinstance(prompt, str) or not prompt:
        raise MaterializationError(f"row {source_row_index} has no prompt")
    if not isinstance(source_id, str) or not re.fullmatch(r"[0-9a-f]{64}", source_id):
        raise MaterializationError(f"row {source_row_index} has an invalid prompt_id")
    _chosen_prompt, chosen_text = _messages(
        row.get("chosen"), f"row {source_row_index}.chosen", prompt
    )
    _rejected_prompt, rejected_text = _messages(
        row.get("rejected"), f"row {source_row_index}.rejected", prompt
    )
    rendered = render_gemma4_user_prompt(prompt)
    prompt_ids = tuple(tokenizer.encode(rendered, add_special_tokens=False))
    chosen_ids = tuple(tokenizer.encode(chosen_text, add_special_tokens=False))
    rejected_ids = tuple(tokenizer.encode(rejected_text, add_special_tokens=False))
    if not prompt_ids or not chosen_ids or not rejected_ids:
        raise MaterializationError(f"row {source_row_index} tokenized to an empty field")
    return TokenizedPreference(
        source_row_index=source_row_index,
        source_id=source_id,
        score_chosen=_finite_score(
            row.get("score_chosen"), f"row {source_row_index}.score_chosen"
        ),
        score_rejected=_finite_score(
            row.get("score_rejected"), f"row {source_row_index}.score_rejected"
        ),
        prompt=prompt,
        rendered_prompt=rendered,
        prompt_token_ids=prompt_ids,
        chosen_text=chosen_text,
        chosen_token_ids=chosen_ids,
        rejected_text=rejected_text,
        rejected_token_ids=rejected_ids,
    )


def admitted_candidate(example: TokenizedPreference, sequence_length: int) -> bool:
    if example.chosen_token_ids == example.rejected_token_ids:
        return False
    if example.score_chosen - example.score_rejected < 1.0:
        return False
    completion_lengths = (
        len(example.chosen_token_ids),
        len(example.rejected_token_ids),
    )
    if min(completion_lengths) < 32:
        return False
    if max(completion_lengths) / min(completion_lengths) > 4.0:
        return False
    return example.total_tokens <= sequence_length


def select_length_stratified(
    examples: Iterable[TokenizedPreference],
    *,
    sequence_length: int,
    buckets: Sequence[tuple[int, int]] = LENGTH_BUCKETS,
    admitted_occurrence: int = 0,
) -> tuple[TokenizedPreference, ...]:
    if isinstance(admitted_occurrence, bool) or admitted_occurrence < 0:
        raise MaterializationError("admitted occurrence must be nonnegative")
    selected: list[TokenizedPreference | None] = [None] * len(buckets)
    admitted_per_bucket = [0] * len(buckets)
    for example in examples:
        if not admitted_candidate(example, sequence_length):
            continue
        for idx, (lower, upper) in enumerate(buckets):
            if lower <= example.total_tokens < upper:
                if (
                    selected[idx] is None
                    and admitted_per_bucket[idx] == admitted_occurrence
                ):
                    selected[idx] = example
                admitted_per_bucket[idx] += 1
                break
        if all(item is not None for item in selected):
            break
    missing = [
        f"[{lower},{upper})"
        for item, (lower, upper) in zip(selected, buckets)
        if item is None
    ]
    if missing:
        raise MaterializationError(
            "source does not satisfy length buckets: " + ", ".join(missing)
        )
    # Antfly iterates source order inside every epoch.  Preserve that order in
    # both outputs even though selection was stratified by token-length bucket.
    return tuple(
        sorted(
            (item for item in selected if item is not None),
            key=lambda item: item.source_row_index,
        )
    )


def validate_tokenizer_contract(tokenizer: Any, path: Path) -> None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MaterializationError(f"could not read tokenizer contract: {exc}") from exc
    if payload.get("rendered_prompt") != render_gemma4_user_prompt(CONTRACT_PROBE_PROMPT):
        raise MaterializationError("Gemma4 prompt renderer drifted from the locked case")
    checks = (
        (
            payload["rendered_prompt"],
            payload["prompt_token_ids"],
            "prompt token ids",
        ),
        (payload["chosen"]["text"], payload["chosen"]["token_ids"], "chosen ids"),
        (
            payload["rejected"]["text"],
            payload["rejected"]["token_ids"],
            "rejected ids",
        ),
    )
    for text, expected, label in checks:
        actual = tokenizer.encode(text, add_special_tokens=False)
        if actual != expected:
            raise MaterializationError(f"tokenizer drifted for locked {label}")


def _example_payload(example: TokenizedPreference) -> dict[str, Any]:
    return {
        "source_row_index": example.source_row_index,
        "source_id": example.source_id,
        "score_chosen": example.score_chosen,
        "score_rejected": example.score_rejected,
        "rendered_prompt": example.rendered_prompt,
        "prompt_token_ids": list(example.prompt_token_ids),
        "chosen": {
            "text": example.chosen_text,
            "token_ids": list(example.chosen_token_ids),
        },
        "rejected": {
            "text": example.rejected_text,
            "token_ids": list(example.rejected_token_ids),
        },
    }


def write_outputs(
    *,
    output_dir: Path,
    parquet_path: Path,
    source_rows: int,
    dataset_id: str,
    revision: str,
    split: str,
    model_dir: Path,
    sequence_length: int,
    examples: Sequence[TokenizedPreference],
    admitted_occurrence: int = 0,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    jsonl_path = output_dir / "preferences.jsonl"
    with jsonl_path.open("x", encoding="utf-8") as handle:
        for example in examples:
            row = {
                "prompt": example.prompt,
                "chosen": example.chosen_text,
                "rejected": example.rejected_text,
                "metadata": {
                    "source": dataset_id,
                    "source_revision": revision,
                    "source_split": split,
                    "source_row_index": example.source_row_index,
                    "source_id": example.source_id,
                    "score_chosen": example.score_chosen,
                    "score_rejected": example.score_rejected,
                },
            }
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )

    jsonl_sha256 = sha256_file(jsonl_path)
    parquet_sha256 = sha256_file(parquet_path)
    case_payload: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "name": (
            "gemma4-e2b-ultrafeedback-test-prefs-stratified-seq512"
            if admitted_occurrence == 0
            else "gemma4-e2b-ultrafeedback-test-prefs-stratified-seq512-"
            f"admitted-occurrence-{admitted_occurrence}"
        ),
        "model_key": "gemma-4-E2B-it",
        "target_preset": "peft-qv",
        "sequence_length": sequence_length,
        "beta": 0.1,
        "learning_rate": 0.0001,
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.999,
            "epsilon": 1e-8,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
        },
        "protocol": FIXED_PROTOCOL,
        "dataset": {
            "repo_id": dataset_id,
            "revision": revision,
            "split": split,
            "source_file": f"{parquet_path.parent.name}/{parquet_path.name}",
            "source_file_sha256": parquet_sha256,
            "source_rows": source_rows,
            "materialized_jsonl": jsonl_path.name,
            "materialized_jsonl_sha256": jsonl_sha256,
            "selection_policy": {
                "ordering": (
                    "first-admitted-per-length-bucket-then-source-order"
                    if admitted_occurrence == 0
                    else "fixed-admitted-occurrence-per-length-bucket-then-source-order"
                ),
                "length_buckets_half_open": [list(bucket) for bucket in LENGTH_BUCKETS],
                "minimum_score_margin": 1.0,
                "minimum_completion_tokens_per_side": 32,
                "maximum_completion_length_ratio": 4.0,
                "distinct_preference_tokens": "required",
                "truncation": "forbidden",
                **(
                    {"admitted_occurrence_zero_based": admitted_occurrence}
                    if admitted_occurrence != 0
                    else {}
                ),
            },
        },
        "examples": [_example_payload(example) for example in examples],
    }
    case_path = output_dir / "mlx_case.json"
    case_path.write_text(
        json.dumps(case_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    tokenizer_files = [model_dir / "tokenizer.json", model_dir / "tokenizer_config.json"]
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset": case_payload["dataset"],
        "model_dir": str(model_dir),
        "tokenizer_files": {
            path.name: sha256_file(path) for path in tokenizer_files
        },
        "sequence_length": sequence_length,
        "selected_source_row_indices": [item.source_row_index for item in examples],
        "selected_source_ids": [item.source_id for item in examples],
        "selected_total_tokens": [item.total_tokens for item in examples],
        "preferences_jsonl": str(jsonl_path),
        "preferences_jsonl_sha256": jsonl_sha256,
        "mlx_case": str(case_path),
        "mlx_case_sha256": sha256_file(case_path),
        "mlx_case_semantic_sha256": canonical_sha256(case_payload),
        "materializer": str(SCRIPT_PATH),
        "materializer_sha256": sha256_file(SCRIPT_PATH),
        "dependency_versions": {},
    }
    return manifest


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset_id != EXPECTED_DATASET_ID or args.split != EXPECTED_SPLIT:
        raise MaterializationError("this parity contract is locked to UltraFeedback test_prefs")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise MaterializationError("revision must be a full lowercase Git commit")
    if args.sequence_length != 512:
        raise MaterializationError("the real-data parity contract requires sequence_length=512")
    if args.bucket_occurrence < 0:
        raise MaterializationError("bucket occurrence must be nonnegative")

    parquet_path = args.parquet.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not parquet_path.is_file():
        raise MaterializationError(f"source parquet does not exist: {parquet_path}")
    if not model_dir.is_dir():
        raise MaterializationError(f"model directory does not exist: {model_dir}")
    if args.source_sha256 and sha256_file(parquet_path) != args.source_sha256:
        raise MaterializationError("source parquet SHA-256 does not match")

    try:
        import pyarrow
        import pyarrow.parquet as pq
        import tokenizers
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise MaterializationError(
            "materialization requires pyarrow, transformers, and tokenizers"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    validate_tokenizer_contract(tokenizer, args.tokenizer_contract)
    table = pq.read_table(parquet_path)
    raw_rows = table.to_pylist()
    tokenized = (
        tokenize_source_row(row, row_index, tokenizer)
        for row_index, row in enumerate(raw_rows)
    )
    selected = select_length_stratified(
        tokenized,
        sequence_length=args.sequence_length,
        admitted_occurrence=args.bucket_occurrence,
    )
    manifest = write_outputs(
        output_dir=output_dir,
        parquet_path=parquet_path,
        source_rows=len(raw_rows),
        dataset_id=args.dataset_id,
        revision=args.revision,
        split=args.split,
        model_dir=model_dir,
        sequence_length=args.sequence_length,
        examples=selected,
        admitted_occurrence=args.bucket_occurrence,
    )
    manifest["dependency_versions"] = {
        "pyarrow": pyarrow.__version__,
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--parquet", type=Path, required=True)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--dataset-id", default=EXPECTED_DATASET_ID)
    result.add_argument("--revision", required=True)
    result.add_argument("--split", default=EXPECTED_SPLIT)
    result.add_argument("--source-sha256")
    result.add_argument("--sequence-length", type=int, default=512)
    result.add_argument(
        "--bucket-occurrence",
        type=int,
        default=0,
        help="select this zero-based admitted example in every length bucket",
    )
    result.add_argument(
        "--tokenizer-contract", type=Path, default=DEFAULT_TOKENIZER_CONTRACT
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        manifest = materialize(args)
    except MaterializationError as exc:
        print(f"Gemma4 DPO HF materialization error: {exc}")
        return 2
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
