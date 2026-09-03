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
SCHEMA_VERSION_V1 = "antfly_gemma4_grpo_boolq_materialization/v1"
SCHEMA_VERSION = "antfly_gemma4_grpo_boolq_materialization/v2"
SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION_V1, SCHEMA_VERSION})
V2_SELECTION_ORDERING = (
    "source-order-after-token-budget-admission-exact-exclusions-"
    "and-balanced-per-label-offset"
)
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


def validate_materialization_semantic_sha256(manifest: Mapping[str, Any]) -> None:
    """Require the manifest's self-attested semantic identity to be exact."""

    claimed = manifest.get("semantic_sha256")
    if not isinstance(claimed, str) or re.fullmatch(r"[0-9a-f]{64}", claimed) is None:
        raise MaterializationError(
            "BoolQ materialization semantic_sha256 must be a lowercase SHA-256"
        )
    unsigned = dict(manifest)
    del unsigned["semantic_sha256"]
    if canonical_sha256(unsigned) != claimed:
        raise MaterializationError("BoolQ materialization semantic SHA-256 drifted")


def load_excluded_eval_source_ids(
    manifests: Sequence[Path],
) -> tuple[frozenset[str], tuple[Mapping[str, Any], ...]]:
    excluded: set[str] = set()
    evidence: list[Mapping[str, Any]] = []
    seen_paths: set[Path] = set()
    for requested_path in manifests:
        candidate = requested_path.expanduser()
        if candidate.is_symlink():
            raise MaterializationError(
                f"evaluation exclusion manifest cannot be a symlink: {candidate}"
            )
        try:
            path = candidate.resolve(strict=True)
            if not path.is_file():
                raise OSError("not a regular file")
            raw = path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MaterializationError(
                f"cannot load evaluation exclusion manifest {candidate}: {exc}"
            ) from exc
        if path in seen_paths:
            raise MaterializationError(
                f"evaluation exclusion manifest is repeated: {path}"
            )
        seen_paths.add(path)
        if not isinstance(payload, Mapping):
            raise MaterializationError(
                f"evaluation exclusion manifest is not an object: {path}"
            )
        schema_version = payload.get("schema_version")
        if schema_version not in SCHEMA_VERSIONS:
            raise MaterializationError(
                f"unsupported evaluation exclusion manifest schema: {path}"
            )
        raw_ids = payload.get("eval_source_ids")
        if not isinstance(raw_ids, list) or not raw_ids:
            raise MaterializationError(
                f"evaluation exclusion manifest has no source identities: {path}"
            )
        manifest_ids: set[str] = set()
        for value in raw_ids:
            if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                raise MaterializationError(
                    f"evaluation exclusion manifest has an invalid source identity: {path}"
                )
            manifest_ids.add(value)
        if len(manifest_ids) != len(raw_ids):
            raise MaterializationError(
                f"evaluation exclusion manifest has duplicate source identities: {path}"
            )
        excluded.update(manifest_ids)
        evidence.append(
            {
                "path": str(path),
                # Bind the exact bytes parsed above, not a second read that
                # could race a concurrent replacement of the source file.
                "sha256": hashlib.sha256(raw).hexdigest(),
                "schema_version": schema_version,
                "source_id_count": len(manifest_ids),
            }
        )
    return frozenset(excluded), tuple(evidence)


def validate_materialization_selection_contract(
    manifest: Mapping[str, Any],
) -> None:
    """Validate the versioned selection/exclusion portion of a manifest.

    The MLX parity runners consume both historical v1 fixtures and current v2
    holdouts. Keeping this validation beside the materializer prevents those
    consumers from silently accepting a v2 manifest while ignoring the exact
    exclusion evidence that makes a refreshed holdout disjoint.
    """

    schema_version = manifest.get("schema_version")
    if schema_version not in SCHEMA_VERSIONS:
        raise MaterializationError("unsupported BoolQ materialization schema")
    if schema_version == SCHEMA_VERSION_V1:
        return

    dataset = manifest.get("dataset")
    if not isinstance(dataset, Mapping):
        raise MaterializationError("materialization dataset must be an object")
    policy = dataset.get("selection_policy")
    if not isinstance(policy, Mapping):
        raise MaterializationError("materialization selection policy must be an object")
    if policy.get("ordering") != V2_SELECTION_ORDERING:
        raise MaterializationError("v2 materialization selection ordering drifted")

    counts: dict[str, int] = {}
    for field in (
        "train_skip_per_label",
        "evaluation_skip_per_label",
        "evaluation_excluded_source_ids",
        "evaluation_exclusion_evidence_source_ids",
    ):
        value = policy.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MaterializationError(
                f"v2 materialization selection_policy.{field} must be a non-negative integer"
            )
        counts[field] = value

    raw_evidence = manifest.get("evaluation_exclusion_manifests")
    if not isinstance(raw_evidence, list):
        raise MaterializationError(
            "v2 materialization evaluation exclusion evidence must be an array"
        )
    evidence_paths: list[Path] = []
    normalized_evidence: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for index, raw_record in enumerate(raw_evidence):
        if not isinstance(raw_record, Mapping) or set(raw_record) != {
            "path",
            "sha256",
            "schema_version",
            "source_id_count",
        }:
            raise MaterializationError(
                f"v2 evaluation exclusion evidence {index} has the wrong fields"
            )
        path_value = raw_record.get("path")
        digest = raw_record.get("sha256")
        source_id_count = raw_record.get("source_id_count")
        if not isinstance(path_value, str) or not path_value or "\x00" in path_value:
            raise MaterializationError(
                f"v2 evaluation exclusion evidence {index} has an invalid path"
            )
        if path_value in seen_paths:
            raise MaterializationError("v2 evaluation exclusion evidence repeats a path")
        seen_paths.add(path_value)
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise MaterializationError(
                f"v2 evaluation exclusion evidence {index} has an invalid SHA-256"
            )
        if raw_record.get("schema_version") not in SCHEMA_VERSIONS:
            raise MaterializationError(
                f"v2 evaluation exclusion evidence {index} has an unsupported schema"
            )
        if (
            isinstance(source_id_count, bool)
            or not isinstance(source_id_count, int)
            or source_id_count <= 0
        ):
            raise MaterializationError(
                f"v2 evaluation exclusion evidence {index} has an invalid source count"
            )
        evidence_paths.append(Path(path_value))
        normalized_evidence.append(dict(raw_record))

    excluded, verified_evidence = load_excluded_eval_source_ids(evidence_paths)
    if list(verified_evidence) != normalized_evidence:
        raise MaterializationError(
            "v2 evaluation exclusion evidence differs from its source manifests"
        )
    evidence_source_ids = sum(
        int(record["source_id_count"]) for record in verified_evidence
    )
    if counts["evaluation_exclusion_evidence_source_ids"] != evidence_source_ids:
        raise MaterializationError(
            "v2 evaluation exclusion evidence source count is inconsistent"
        )
    if counts["evaluation_excluded_source_ids"] != len(excluded):
        raise MaterializationError(
            "v2 unique evaluation exclusion count is inconsistent"
        )

    for field in ("train_source_ids", "eval_source_ids"):
        source_ids = manifest.get(field)
        if (
            not isinstance(source_ids, list)
            or len(set(source_ids)) != len(source_ids)
            or any(
                not isinstance(source_id, str)
                or re.fullmatch(r"[0-9a-f]{64}", source_id) is None
                for source_id in source_ids
            )
        ):
            raise MaterializationError(f"v2 materialization {field} is invalid")
    if set(manifest["eval_source_ids"]) & excluded:
        raise MaterializationError(
            "v2 materialized evaluation identities overlap the exclusion set"
        )


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
    skip_per_label: int = 0,
    excluded_source_ids: frozenset[str] = frozenset(),
    max_seq_len: int,
    max_completion_tokens: int,
) -> tuple[BoolQExample, ...]:
    if examples < 2 or examples % 2 != 0:
        raise MaterializationError("example count must be even and at least two")
    if skip_per_label < 0:
        raise MaterializationError("per-label skip must be non-negative")
    quota = examples // 2
    selected: list[BoolQExample] = []
    counts = {"yes": 0, "no": 0}
    skipped = {"yes": 0, "no": 0}
    for row_index, row in enumerate(rows):
        candidate = parse_row(
            row,
            split=split,
            row_index=row_index,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_completion_tokens=max_completion_tokens,
        )
        if candidate is None or candidate.source_id in excluded_source_ids:
            continue
        if skipped[candidate.target] < skip_per_label:
            skipped[candidate.target] += 1
            continue
        if counts[candidate.target] >= quota:
            continue
        selected.append(candidate)
        counts[candidate.target] += 1
        if len(selected) == examples:
            break
    if len(selected) != examples:
        raise MaterializationError(
            f"{split}: source cannot skip {skip_per_label} and then provide "
            f"{quota} admitted rows per label; skipped {skipped}, got {counts}"
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
    train_skip_per_label: int,
    eval_skip_per_label: int,
    eval_excluded_source_id_count: int,
    eval_exclusion_evidence: Sequence[Mapping[str, Any]],
    max_seq_len: int,
    max_completion_tokens: int,
    dependency_versions: Mapping[str, str],
) -> Mapping[str, Any]:
    exclusion_evidence_count = sum(
        int(item["source_id_count"]) for item in eval_exclusion_evidence
    )
    if not 0 <= eval_excluded_source_id_count <= exclusion_evidence_count:
        raise MaterializationError(
            "unique evaluation exclusion count is inconsistent with manifest evidence"
        )
    train_ids = {example.source_id for example in train_examples}
    eval_ids = {example.source_id for example in eval_examples}
    if train_ids & eval_ids:
        raise MaterializationError("train and evaluation source identities overlap")
    tokenizer_files = (model_dir / "tokenizer.json", model_dir / "tokenizer_config.json")
    for path in tokenizer_files:
        if not path.is_file():
            raise MaterializationError(f"missing tokenizer input: {path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    train_path = output_dir / "train.jsonl"
    eval_path = output_dir / "eval.jsonl"
    _write_jsonl(train_path, train_examples)
    _write_jsonl(eval_path, eval_examples)
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
                "ordering": V2_SELECTION_ORDERING,
                "train_skip_per_label": train_skip_per_label,
                "evaluation_skip_per_label": eval_skip_per_label,
                "evaluation_excluded_source_ids": eval_excluded_source_id_count,
                "evaluation_exclusion_evidence_source_ids": exclusion_evidence_count,
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
        "evaluation_exclusion_manifests": list(eval_exclusion_evidence),
        "materializer": str(SCRIPT_PATH),
        "materializer_sha256": sha256_file(SCRIPT_PATH),
        "dependency_versions": dict(dependency_versions),
    }
    validate_materialization_selection_contract(manifest)
    manifest["semantic_sha256"] = canonical_sha256(manifest)
    validate_materialization_semantic_sha256(manifest)
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
    excluded_eval_source_ids, eval_exclusion_evidence = (
        load_excluded_eval_source_ids(args.exclude_eval_manifest)
    )
    train_table = pq.read_table(train_parquet)
    eval_table = pq.read_table(eval_parquet)
    train_examples = select_balanced(
        train_table.to_pylist(),
        split=EXPECTED_TRAIN_SPLIT,
        tokenizer=tokenizer,
        examples=args.examples,
        skip_per_label=args.train_skip_per_label,
        max_seq_len=args.max_seq_len,
        max_completion_tokens=args.max_completion_tokens,
    )
    eval_examples = select_balanced(
        eval_table.to_pylist(),
        split=EXPECTED_EVAL_SPLIT,
        tokenizer=tokenizer,
        examples=args.eval_examples,
        skip_per_label=args.eval_skip_per_label,
        excluded_source_ids=excluded_eval_source_ids,
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
        train_skip_per_label=args.train_skip_per_label,
        eval_skip_per_label=args.eval_skip_per_label,
        eval_excluded_source_id_count=len(excluded_eval_source_ids),
        eval_exclusion_evidence=eval_exclusion_evidence,
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
    result.add_argument(
        "--train-skip-per-label",
        type=int,
        default=0,
        help="skip this many admitted train rows for each label before selection",
    )
    result.add_argument(
        "--eval-skip-per-label",
        type=int,
        default=0,
        help="skip this many admitted evaluation rows for each label before selection",
    )
    result.add_argument(
        "--exclude-eval-manifest",
        type=Path,
        action="append",
        default=[],
        help=(
            "exclude exact eval source identities from a prior v1/v2 BoolQ "
            "materialization manifest; repeatable and digest-bound"
        ),
    )
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
