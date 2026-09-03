#!/usr/bin/env python3
"""Run the pinned MLX-LM side of the Gemma4 DPO parity benchmark.

The runner consumes an exact token contract, never tokenizes or downloads,
and uses the same zero-initialized Antfly PEFT adapter as the Zig/Metal run.
MLX imports stay lazy so the contract tests require only the standard library.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import types
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
DEFAULT_CASE_PATH = (
    SCRIPT_DIR.parent.parent
    / "testdata"
    / "gemma4_dpo_e2b_seq128_benchmark.json"
)
CASE_SCHEMA_VERSION = "antfly_gemma4_dpo_benchmark_case/v1"
DATASET_CASE_SCHEMA_VERSION = "antfly_gemma4_dpo_benchmark_dataset/v1"
RESULT_SCHEMA_VERSION = "antfly_gemma4_dpo_mlx_benchmark/v3"
FIXED_PROTOCOL = {"cold": 1, "first": 1, "warmup": 3, "measured": 20}
MODEL_KEYS = ("gemma-4-E2B-it", "gemma-4-E4B-it")

sys.path.insert(0, str(SCRIPT_DIR))
import run_gemma4_lora_mlx_benchmark as locked  # noqa: E402


class DpoBenchmarkContractError(RuntimeError):
    """The benchmark input, environment, or result violated the locked contract."""


@dataclass(frozen=True)
class BenchmarkExample:
    rendered_prompt: str
    prompt_token_ids: tuple[int, ...]
    chosen_text: str
    chosen_token_ids: tuple[int, ...]
    rejected_text: str
    rejected_token_ids: tuple[int, ...]
    source_row_index: int | None = None
    source_id: str | None = None
    score_chosen: float | None = None
    score_rejected: float | None = None


@dataclass(frozen=True)
class BenchmarkCase:
    source_path: Path
    schema_version: str
    name: str
    model_key: str
    target_preset: str
    sequence_length: int
    beta: float
    learning_rate: float
    optimizer: dict[str, float]
    protocol: dict[str, int]
    examples: tuple[BenchmarkExample, ...]
    dataset: dict[str, Any] | None
    semantic_sha256: str

    # Preserve the legacy single-case inspection API used by existing tests
    # and callers while the runtime consumes ``examples`` uniformly.
    @property
    def rendered_prompt(self) -> str:
        return self.examples[0].rendered_prompt

    @property
    def prompt_token_ids(self) -> tuple[int, ...]:
        return self.examples[0].prompt_token_ids

    @property
    def chosen_text(self) -> str:
        return self.examples[0].chosen_text

    @property
    def chosen_token_ids(self) -> tuple[int, ...]:
        return self.examples[0].chosen_token_ids

    @property
    def rejected_text(self) -> str:
        return self.examples[0].rejected_text

    @property
    def rejected_token_ids(self) -> tuple[int, ...]:
        return self.examples[0].rejected_token_ids


def _require_exact_keys(
    value: Mapping[str, Any], expected: set[str], where: str
) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise DpoBenchmarkContractError(
            f"{where} fields drifted (missing={missing}, unknown={unknown})"
        )


def _finite_float(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DpoBenchmarkContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise DpoBenchmarkContractError(f"{where} is outside the admitted range")
    return result


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise DpoBenchmarkContractError(f"{where} must be a positive integer")
    return value


def _nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DpoBenchmarkContractError(f"{where} must be a nonnegative integer")
    return value


def _token_ids(value: Any, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise DpoBenchmarkContractError(f"{where} must be a non-empty token list")
    result: list[int] = []
    for idx, token_id in enumerate(value):
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise DpoBenchmarkContractError(f"{where}[{idx}] is not a token id")
        result.append(token_id)
    return tuple(result)


def _canonical_case_payload(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DpoBenchmarkContractError(f"{where} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise DpoBenchmarkContractError(f"{where} must be hexadecimal") from exc
    if value != value.lower():
        raise DpoBenchmarkContractError(f"{where} must be lowercase")
    return value


def _parse_example(
    payload: Any,
    *,
    sequence_length: int,
    where: str,
    with_source: bool,
) -> BenchmarkExample:
    if not isinstance(payload, dict):
        raise DpoBenchmarkContractError(f"{where} must be an object")
    expected = {
        "rendered_prompt",
        "prompt_token_ids",
        "chosen",
        "rejected",
    }
    if with_source:
        expected |= {
            "source_row_index",
            "source_id",
            "score_chosen",
            "score_rejected",
        }
    _require_exact_keys(payload, expected, where)

    chosen = payload["chosen"]
    rejected = payload["rejected"]
    if not isinstance(chosen, dict) or not isinstance(rejected, dict):
        raise DpoBenchmarkContractError(f"{where} chosen and rejected must be objects")
    _require_exact_keys(chosen, {"text", "token_ids"}, f"{where}.chosen")
    _require_exact_keys(rejected, {"text", "token_ids"}, f"{where}.rejected")
    prompt_ids = _token_ids(payload["prompt_token_ids"], f"{where}.prompt_token_ids")
    chosen_ids = _token_ids(chosen["token_ids"], f"{where}.chosen.token_ids")
    rejected_ids = _token_ids(rejected["token_ids"], f"{where}.rejected.token_ids")
    if len(prompt_ids) + max(len(chosen_ids), len(rejected_ids)) > sequence_length:
        raise DpoBenchmarkContractError(f"{where} exceeds sequence_length")
    if chosen_ids == rejected_ids:
        raise DpoBenchmarkContractError(f"{where} chosen and rejected tokens must differ")
    for value, label in (
        (payload["rendered_prompt"], "rendered_prompt"),
        (chosen["text"], "chosen.text"),
        (rejected["text"], "rejected.text"),
    ):
        if not isinstance(value, str) or not value:
            raise DpoBenchmarkContractError(f"{where}.{label} must be non-empty")

    source_row_index: int | None = None
    source_id: str | None = None
    score_chosen: float | None = None
    score_rejected: float | None = None
    if with_source:
        source_row_index = _nonnegative_int(
            payload["source_row_index"], f"{where}.source_row_index"
        )
        source_id = _sha256_string(payload["source_id"], f"{where}.source_id")
        score_chosen = _finite_float(
            payload["score_chosen"], f"{where}.score_chosen"
        )
        score_rejected = _finite_float(
            payload["score_rejected"], f"{where}.score_rejected"
        )
        if score_chosen <= score_rejected:
            raise DpoBenchmarkContractError(
                f"{where} chosen score must exceed rejected score"
            )
    return BenchmarkExample(
        rendered_prompt=payload["rendered_prompt"],
        prompt_token_ids=prompt_ids,
        chosen_text=chosen["text"],
        chosen_token_ids=chosen_ids,
        rejected_text=rejected["text"],
        rejected_token_ids=rejected_ids,
        source_row_index=source_row_index,
        source_id=source_id,
        score_chosen=score_chosen,
        score_rejected=score_rejected,
    )


def _parse_dataset(payload: Any, source_path: Path) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise DpoBenchmarkContractError("dataset must be an object")
    _require_exact_keys(
        payload,
        {
            "repo_id",
            "revision",
            "split",
            "source_file",
            "source_file_sha256",
            "source_rows",
            "materialized_jsonl",
            "materialized_jsonl_sha256",
            "selection_policy",
        },
        "dataset",
    )
    for key in ("repo_id", "split", "source_file", "materialized_jsonl"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise DpoBenchmarkContractError(f"dataset.{key} must be non-empty")
    revision = payload["revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(char not in "0123456789abcdef" for char in revision)
    ):
        raise DpoBenchmarkContractError("dataset.revision must be a full Git commit")
    _sha256_string(payload["source_file_sha256"], "dataset.source_file_sha256")
    materialized_sha = _sha256_string(
        payload["materialized_jsonl_sha256"],
        "dataset.materialized_jsonl_sha256",
    )
    _positive_int(payload["source_rows"], "dataset.source_rows")
    policy = payload["selection_policy"]
    if not isinstance(policy, dict) or not policy:
        raise DpoBenchmarkContractError("dataset.selection_policy must be an object")

    relative_jsonl = Path(payload["materialized_jsonl"])
    if relative_jsonl.is_absolute() or ".." in relative_jsonl.parts:
        raise DpoBenchmarkContractError("dataset.materialized_jsonl must be a local sibling")
    jsonl_path = (source_path.parent / relative_jsonl).resolve()
    try:
        actual_sha = hashlib.sha256(jsonl_path.read_bytes()).hexdigest()
    except OSError as exc:
        raise DpoBenchmarkContractError(
            f"could not read materialized preference JSONL: {exc}"
        ) from exc
    if actual_sha != materialized_sha:
        raise DpoBenchmarkContractError("materialized preference JSONL SHA-256 drifted")
    return json.loads(json.dumps(payload))


def load_case(path: Path) -> BenchmarkCase:
    source_path = path.expanduser().resolve()
    try:
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DpoBenchmarkContractError(f"could not load DPO case: {exc}") from exc
    if not isinstance(payload, dict):
        raise DpoBenchmarkContractError("DPO case root must be an object")
    schema_version = payload.get("schema_version")
    common_keys = {
        "schema_version",
        "name",
        "model_key",
        "target_preset",
        "sequence_length",
        "beta",
        "learning_rate",
        "optimizer",
        "protocol",
    }
    if schema_version == CASE_SCHEMA_VERSION:
        _require_exact_keys(
            payload,
            common_keys
            | {"rendered_prompt", "prompt_token_ids", "chosen", "rejected"},
            "DPO case",
        )
    elif schema_version == DATASET_CASE_SCHEMA_VERSION:
        _require_exact_keys(
            payload,
            common_keys | {"dataset", "examples"},
            "DPO dataset case",
        )
    else:
        raise DpoBenchmarkContractError("unsupported DPO case schema")

    for key in ("name", "model_key", "target_preset"):
        if not isinstance(payload[key], str) or not payload[key]:
            raise DpoBenchmarkContractError(f"{key} must be a non-empty string")
    if payload["model_key"] not in MODEL_KEYS:
        raise DpoBenchmarkContractError("model_key is outside the Gemma4 DPO matrix")
    sequence_length = _positive_int(payload["sequence_length"], "sequence_length")

    optimizer = payload["optimizer"]
    if not isinstance(optimizer, dict):
        raise DpoBenchmarkContractError("optimizer must be an object")
    _require_exact_keys(
        optimizer,
        {"beta1", "beta2", "epsilon", "weight_decay", "max_grad_norm"},
        "optimizer",
    )
    normalized_optimizer = {
        "beta1": _finite_float(optimizer["beta1"], "optimizer.beta1", positive=True),
        "beta2": _finite_float(optimizer["beta2"], "optimizer.beta2", positive=True),
        "epsilon": _finite_float(
            optimizer["epsilon"], "optimizer.epsilon", positive=True
        ),
        "weight_decay": _finite_float(
            optimizer["weight_decay"], "optimizer.weight_decay"
        ),
        "max_grad_norm": _finite_float(
            optimizer["max_grad_norm"], "optimizer.max_grad_norm", positive=True
        ),
    }
    if not 0.0 < normalized_optimizer["beta1"] < 1.0:
        raise DpoBenchmarkContractError("optimizer.beta1 must be between zero and one")
    if not 0.0 < normalized_optimizer["beta2"] < 1.0:
        raise DpoBenchmarkContractError("optimizer.beta2 must be between zero and one")
    if normalized_optimizer["weight_decay"] < 0.0:
        raise DpoBenchmarkContractError("optimizer.weight_decay cannot be negative")

    protocol = payload["protocol"]
    if not isinstance(protocol, dict):
        raise DpoBenchmarkContractError("protocol must be an object")
    _require_exact_keys(protocol, set(FIXED_PROTOCOL), "protocol")
    normalized_protocol = {
        key: _positive_int(protocol[key], f"protocol.{key}")
        for key in FIXED_PROTOCOL
    }
    if normalized_protocol != FIXED_PROTOCOL:
        raise DpoBenchmarkContractError(
            f"protocol must equal the fixed {FIXED_PROTOCOL} contract"
        )

    dataset: dict[str, Any] | None = None
    if schema_version == CASE_SCHEMA_VERSION:
        examples = (
            _parse_example(
                {
                    "rendered_prompt": payload["rendered_prompt"],
                    "prompt_token_ids": payload["prompt_token_ids"],
                    "chosen": payload["chosen"],
                    "rejected": payload["rejected"],
                },
                sequence_length=sequence_length,
                where="DPO case",
                with_source=False,
            ),
        )
    else:
        raw_examples = payload["examples"]
        if not isinstance(raw_examples, list) or len(raw_examples) <= 1:
            raise DpoBenchmarkContractError(
                "DPO dataset case must contain multiple examples"
            )
        total_updates = sum(FIXED_PROTOCOL.values())
        if total_updates % len(raw_examples) != 0:
            raise DpoBenchmarkContractError(
                "DPO dataset example count must divide the fixed update count"
            )
        examples = tuple(
            _parse_example(
                item,
                sequence_length=sequence_length,
                where=f"examples[{idx}]",
                with_source=True,
            )
            for idx, item in enumerate(raw_examples)
        )
        source_rows = [item.source_row_index for item in examples]
        source_ids = [item.source_id for item in examples]
        if source_rows != sorted(source_rows) or len(set(source_rows)) != len(source_rows):
            raise DpoBenchmarkContractError(
                "DPO dataset examples must use unique source-order rows"
            )
        if len(set(source_ids)) != len(source_ids):
            raise DpoBenchmarkContractError("DPO dataset source ids must be unique")
        dataset = _parse_dataset(payload["dataset"], source_path)

    digest = "sha256:" + hashlib.sha256(_canonical_case_payload(payload)).hexdigest()
    return BenchmarkCase(
        source_path=source_path,
        schema_version=schema_version,
        name=payload["name"],
        model_key=payload["model_key"],
        target_preset=payload["target_preset"],
        sequence_length=sequence_length,
        beta=_finite_float(payload["beta"], "beta", positive=True),
        learning_rate=_finite_float(
            payload["learning_rate"], "learning_rate", positive=True
        ),
        optimizer=normalized_optimizer,
        protocol=normalized_protocol,
        examples=examples,
        dataset=dataset,
        semantic_sha256=digest,
    )


def padded_sequence(
    prompt: Sequence[int], completion: Sequence[int], sequence_length: int
) -> tuple[list[int], list[int]]:
    tokens = [*prompt, *completion]
    if not prompt or not completion or len(tokens) > sequence_length:
        raise DpoBenchmarkContractError("invalid padded preference sequence")
    ids = tokens + [0] * (sequence_length - len(tokens))
    labels = [-100] * sequence_length
    labels[len(prompt) : len(tokens)] = completion
    return ids, labels


def pair_sequence_length(
    example: BenchmarkExample,
    maximum_sequence_length: int,
    bucket_quantum: int | None,
    bucket_minimum: int | None = None,
) -> int:
    """Return one shared chosen/rejected shape matching Antfly's DPO policy."""
    required = len(example.prompt_token_ids) + max(
        len(example.chosen_token_ids), len(example.rejected_token_ids)
    )
    if required > maximum_sequence_length:
        raise DpoBenchmarkContractError("preference pair exceeds sequence_length")
    if bucket_quantum is None:
        if bucket_minimum is not None:
            raise DpoBenchmarkContractError(
                "sequence length bucket minimum requires a quantum"
            )
        return maximum_sequence_length
    if bucket_quantum <= 0:
        raise DpoBenchmarkContractError(
            "sequence length bucket quantum must be positive"
        )
    minimum = bucket_quantum if bucket_minimum is None else bucket_minimum
    if minimum <= 0:
        raise DpoBenchmarkContractError(
            "sequence length bucket minimum must be positive"
        )
    desired = max(required, min(minimum, maximum_sequence_length))
    rounded = ((desired + bucket_quantum - 1) // bucket_quantum) * bucket_quantum
    return min(rounded, maximum_sequence_length)


def summarize_sequence_length_policy(
    case: BenchmarkCase,
    pair_lengths: Sequence[int],
    bucket_quantum: int | None,
    bucket_minimum: int | None,
) -> dict[str, Any]:
    logical_rows = sum(
        2 * len(example.prompt_token_ids)
        + len(example.chosen_token_ids)
        + len(example.rejected_token_ids)
        for example in case.examples
    )
    scheduled_rows = 2 * sum(pair_lengths)
    fixed_rows = 2 * case.sequence_length * len(case.examples)
    avoided = fixed_rows - scheduled_rows
    return {
        "mode": (
            "fixed-pair-padding"
            if bucket_quantum is None
            else "pair-safe-length-buckets"
        ),
        "maximum_sequence_length": case.sequence_length,
        "bucket_quantum": bucket_quantum,
        "bucket_minimum": (
            None
            if bucket_quantum is None
            else bucket_quantum if bucket_minimum is None else bucket_minimum
        ),
        "pair_sequence_lengths": list(pair_lengths),
        "pairs": len(case.examples),
        "logical_branch_rows": logical_rows,
        "scheduled_branch_rows": scheduled_rows,
        "fixed_shape_branch_rows": fixed_rows,
        "padding_rows_avoided": avoided,
        "padding_reduction_fraction": avoided / fixed_rows,
        "minimum_pair_sequence_length": min(pair_lengths),
        "maximum_pair_sequence_length": max(pair_lengths),
        "unique_pair_sequence_lengths": len(set(pair_lengths)),
    }


def dpo_evaluation_metrics(
    policy_chosen: Sequence[float],
    policy_rejected: Sequence[float],
    reference_chosen: Sequence[float],
    reference_rejected: Sequence[float],
    beta: float,
) -> dict[str, Any]:
    lengths = {
        len(policy_chosen),
        len(policy_rejected),
        len(reference_chosen),
        len(reference_rejected),
    }
    if lengths == {0} or len(lengths) != 1:
        raise DpoBenchmarkContractError(
            "DPO evaluation vectors must be non-empty and equally sized"
        )
    rows = []
    for index, (pc, pr, rc, rr) in enumerate(
        zip(policy_chosen, policy_rejected, reference_chosen, reference_rejected)
    ):
        values = (pc, pr, rc, rr, beta)
        if not all(math.isfinite(value) for value in values):
            raise DpoBenchmarkContractError("DPO evaluation contains non-finite values")
        reward_margin = beta * ((pc - rc) - (pr - rr))
        loss = max(0.0, -reward_margin) + math.log1p(
            math.exp(-abs(reward_margin))
        )
        rows.append(
            {
                "index": index,
                "policy_chosen_logp": pc,
                "policy_rejected_logp": pr,
                "reference_chosen_logp": rc,
                "reference_rejected_logp": rr,
                "policy_logp_margin": pc - pr,
                "reference_logp_margin": rc - rr,
                "reward_margin": reward_margin,
                "loss": loss,
                "preferred": reward_margin > 0.0,
            }
        )
    return {
        "examples": len(rows),
        "mean_loss": statistics.mean(row["loss"] for row in rows),
        "mean_reward_margin": statistics.mean(
            row["reward_margin"] for row in rows
        ),
        "accuracy": statistics.mean(1.0 if row["preferred"] else 0.0 for row in rows),
        "rows": rows,
    }


def require_source_checkout(
    root: Path, expected: str, label: str
) -> dict[str, str]:
    try:
        return locked.verify_source_checkout(
            root,
            expected,
            source_name=label,
        )
    except locked.ContractError as exc:
        raise DpoBenchmarkContractError(
            f"could not attest clean {label} source revision: {exc}"
        ) from exc


def require_source_revision(root: Path, expected: str, label: str) -> str:
    checkout = require_source_checkout(root, expected, label)
    return checkout["revision"]


def install_mlx_lm_source_namespace(source_root: Path) -> Path:
    checkout = source_root.expanduser().resolve()
    package_root = checkout / "mlx_lm"
    if not package_root.is_dir():
        raise DpoBenchmarkContractError(
            f"MLX-LM source checkout has no mlx_lm package: {checkout}"
        )
    mlx_lm_package = types.ModuleType("mlx_lm")
    mlx_lm_package.__path__ = [str(package_root)]
    sys.modules["mlx_lm"] = mlx_lm_package
    models_package = types.ModuleType("mlx_lm.models")
    models_package.__path__ = [str(package_root / "models")]
    sys.modules["mlx_lm.models"] = models_package
    tuner_package = types.ModuleType("mlx_lm.tuner")
    tuner_package.__path__ = [str(package_root / "tuner")]
    sys.modules["mlx_lm.tuner"] = tuner_package
    return package_root


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def require_regular_nonsymlink(path: Path, label: str) -> Path:
    requested = path.expanduser()
    if requested.is_symlink():
        raise DpoBenchmarkContractError(f"{label} must not be a symlink")
    try:
        resolved = requested.resolve(strict=True)
    except OSError as exc:
        raise DpoBenchmarkContractError(f"{label} is unavailable: {requested}") from exc
    if not resolved.is_file():
        raise DpoBenchmarkContractError(f"{label} must be a regular file")
    return resolved


def _requested_new_output_path(path: Path, label: str) -> Path:
    requested = Path(os.path.abspath(path.expanduser()))
    if requested.is_symlink() or requested.exists():
        raise DpoBenchmarkContractError(f"{label} already exists: {requested}")
    return requested


def _new_output_path(path: Path, label: str) -> Path:
    requested = _requested_new_output_path(path, label)
    requested.parent.mkdir(parents=True, exist_ok=True)
    if requested.is_symlink() or requested.exists():
        raise DpoBenchmarkContractError(f"{label} already exists: {requested}")
    return requested.parent.resolve(strict=True) / requested.name


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    destination = _new_output_path(path, "benchmark output")
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temp_path = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, destination)
    except FileExistsError as exc:
        raise DpoBenchmarkContractError(
            f"benchmark output already exists: {destination}"
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)


def write_adapter_exclusive(
    path: Path,
    *,
    final_trainables: Mapping[str, Any],
    target_names: Sequence[str],
    adapter: Any,
    mx: Any,
) -> str:
    """Write final MLX trainables back to the exact PEFT/Antfly orientation."""
    destination = _new_output_path(path, "benchmark adapter output")
    mlx_targets = {
        locked.canonicalize_module_name(name): name for name in target_names
    }
    if len(mlx_targets) != len(target_names):
        raise DpoBenchmarkContractError("MLX adapter target names are not canonical")
    serialized: dict[str, Any] = {}
    for (module, role), descriptor in adapter.tensors.items():
        suffix = "lora_a" if role == "lora_A" else "lora_b"
        try:
            mlx_name = f"{mlx_targets[module]}.{suffix}"
            value = final_trainables[mlx_name]
        except KeyError as exc:
            raise DpoBenchmarkContractError(
                f"final MLX adapter is missing {module}.{role}"
            ) from exc
        serialized[descriptor.source_name] = value.T.astype(mx.float32)
    expected_names = {item.source_name for item in adapter.tensors.values()}
    if set(serialized) != expected_names:
        raise DpoBenchmarkContractError("serialized MLX adapter inventory drifted")

    temp_path = destination.with_name(
        f".{destination.stem}.{os.getpid()}.tmp.safetensors"
    )
    try:
        mx.save_safetensors(
            str(temp_path), serialized, metadata={"format": "pt"}
        )
        os.link(temp_path, destination)
    except FileExistsError as exc:
        raise DpoBenchmarkContractError(
            f"benchmark adapter output already exists: {destination}"
        ) from exc
    finally:
        temp_path.unlink(missing_ok=True)
    return "sha256:" + hashlib.sha256(destination.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> dict[str, Any]:
    case = load_case(args.case)
    first_update_output = (
        _requested_new_output_path(
            args.first_update_adapter_output,
            "first-update adapter output",
        )
        if args.first_update_adapter_output is not None
        else None
    )
    final_output = (
        _requested_new_output_path(args.adapter_output, "final adapter output")
        if args.adapter_output is not None
        else None
    )
    if first_update_output is not None and first_update_output == final_output:
        raise DpoBenchmarkContractError(
            "first-update and final adapter outputs must be different paths"
        )
    lock = locked.load_lock(args.lock)
    if case.target_preset not in lock["target_presets"]:
        raise DpoBenchmarkContractError(
            "target_preset is outside the locked Gemma4 DPO matrix"
        )
    model_dir = args.model_dir.expanduser().resolve()
    try:
        locked_model = locked.verify_model_directory(
            lock, case.model_key, model_dir
        )
    except locked.ContractError as exc:
        raise DpoBenchmarkContractError(
            f"model directory differs from the locked {case.model_key} artifact: {exc}"
        ) from exc
    mlx_contract = lock["mlx_reference"]
    locked.force_offline_environment()
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != mlx_contract["python"]:
        raise DpoBenchmarkContractError(
            f"MLX benchmark requires Python {mlx_contract['python']}, found {actual_python}"
        )
    revisions = mlx_contract["source_revisions"]
    mlx_checkout = require_source_checkout(
        args.mlx_source_root, revisions["mlx"], "MLX"
    )
    mlx_lm_checkout = require_source_checkout(
        args.mlx_lm_source_root, revisions["mlx-lm"], "MLX-LM"
    )
    mlx_revision = mlx_checkout["revision"]
    mlx_lm_revision = mlx_lm_checkout["revision"]
    if platform.system() != mlx_contract["required_platform"]:
        raise DpoBenchmarkContractError("MLX benchmark must run on the locked platform")
    if platform.machine() != mlx_contract["required_machine"]:
        raise DpoBenchmarkContractError("MLX benchmark must run on the locked machine")
    if case.protocol["warmup"] != mlx_contract["warmup_steps"]:
        raise DpoBenchmarkContractError("case warmup count differs from the MLX lock")
    if case.protocol["measured"] != mlx_contract["measured_steps"]:
        raise DpoBenchmarkContractError("case measured count differs from the MLX lock")

    try:
        locked.verify_requirements_match_lock(
            lock,
            "mlx_reference",
            locked.MLX_REQUIREMENTS_PATH,
        )
        package_versions = locked.verify_packages(lock, "mlx_reference")
        preverified_native_bundle = locked.verify_mlx_native_build_before_import(
            args,
            lock,
            mlx_checkout,
            mlx_lm_checkout,
        )
    except locked.ContractError as exc:
        raise DpoBenchmarkContractError(
            f"could not attest the pinned MLX native environment: {exc}"
        ) from exc

    mlx_root = Path(mlx_checkout["path"]).resolve()
    mlx_lm_root = Path(mlx_lm_checkout["path"]).resolve()
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(mlx_root / "python"), str(mlx_root)]

    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten, tree_unflatten
    except ImportError as exc:
        raise DpoBenchmarkContractError(
            f"could not import the pinned MLX source environment: {exc}"
        ) from exc

    core_path = Path(mx.__file__ or "").resolve()
    if not _path_is_within(core_path, mlx_root):
        raise DpoBenchmarkContractError(
            f"imported MLX is outside the attested checkout: {core_path}"
        )
    try:
        native_runtime = locked.verify_mlx_native_runtime(
            args,
            lock,
            mlx_checkout,
            mx,
            preverified_native_bundle,
        )
    except locked.ContractError as exc:
        raise DpoBenchmarkContractError(
            f"loaded MLX native runtime differs from its attestation: {exc}"
        ) from exc
    install_mlx_lm_source_namespace(args.mlx_lm_source_root)
    try:
        from mlx_lm.models import gemma4 as mlx_gemma4
        from mlx_lm.tuner.lora import LoRALinear
    except ImportError as exc:
        raise DpoBenchmarkContractError(
            f"could not import the pinned MLX-LM source environment: {exc}"
        ) from exc

    gemma4_source_path = Path(mlx_gemma4.__file__ or "").resolve()
    lora_source_module = sys.modules[LoRALinear.__module__]
    lora_source_path = Path(lora_source_module.__file__ or "").resolve()
    for label, source_path in (
        ("MLX-LM Gemma4", gemma4_source_path),
        ("MLX-LM LoRA", lora_source_path),
    ):
        if not _path_is_within(source_path, mlx_lm_root):
            raise DpoBenchmarkContractError(
                f"imported {label} is outside the attested checkout: {source_path}"
            )

    adapter_dir = args.adapter_dir.expanduser().resolve()
    try:
        manifest = json.loads(
            (adapter_dir / "antfly_finetune_manifest.json").read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise DpoBenchmarkContractError(f"could not load adapter manifest: {exc}") from exc
    if not isinstance(manifest, dict):
        raise DpoBenchmarkContractError("adapter manifest must be an object")
    binding_fields = (
        "base_model_sha256",
        "tokenizer_sha256",
        "chat_template_sha256",
    )
    try:
        prepared_summary = {key: manifest[key] for key in binding_fields}
    except KeyError as exc:
        raise DpoBenchmarkContractError(
            f"adapter manifest is missing model binding: {exc.args[0]}"
        ) from exc
    if any(
        not isinstance(value, str) or len(value) != 64
        for value in prepared_summary.values()
    ):
        raise DpoBenchmarkContractError("adapter manifest model bindings are malformed")
    base_model_provenance = locked.zig_model_provenance(model_dir)
    for field in binding_fields:
        if prepared_summary[field] != base_model_provenance[field]:
            raise DpoBenchmarkContractError(
                f"adapter {field} does not match the benchmark model"
            )
    adapter = locked.inspect_initial_adapter(
        adapter_dir,
        lock,
        case.model_key,
        case.target_preset,
        prepared_summary,
    )

    comparison_path: Path | None = None
    if args.evaluation_adapter is not None:
        comparison_path = require_regular_nonsymlink(
            args.evaluation_adapter,
            "comparison adapter",
        )

    bound_input_paths = [
        *(model_dir / relative for relative in locked_model["files"]),
        *adapter.bound_files,
        case.source_path,
        args.lock.expanduser().resolve(),
        locked.MLX_REQUIREMENTS_PATH,
        SCRIPT_PATH,
        locked.SCRIPT_PATH,
        Path(sys.executable).resolve(),
        gemma4_source_path,
        lora_source_path,
    ]
    if case.dataset is not None:
        bound_input_paths.append(
            (
                case.source_path.parent
                / Path(case.dataset["materialized_jsonl"])
            ).resolve()
        )
    if comparison_path is not None:
        bound_input_paths.append(comparison_path)
    try:
        bound_input_identities = locked.capture_file_identities(bound_input_paths)
    except locked.ContractError as exc:
        raise DpoBenchmarkContractError(
            f"could not bind immutable DPO benchmark inputs: {exc}"
        ) from exc

    bucket_quantum = getattr(args, "sequence_length_bucket_quantum", None)
    bucket_minimum = getattr(args, "sequence_length_bucket_min", None)
    pair_lengths = [
        pair_sequence_length(
            example,
            case.sequence_length,
            bucket_quantum,
            bucket_minimum,
        )
        for example in case.examples
    ]
    sequence_length_policy = summarize_sequence_length_policy(
        case,
        pair_lengths,
        bucket_quantum,
        bucket_minimum,
    )
    padded_examples = []
    for example, pair_length in zip(case.examples, pair_lengths):
        chosen_ids, chosen_labels = padded_sequence(
            example.prompt_token_ids,
            example.chosen_token_ids,
            pair_length,
        )
        rejected_ids, rejected_labels = padded_sequence(
            example.prompt_token_ids,
            example.rejected_token_ids,
            pair_length,
        )
        padded_examples.append(
            (chosen_ids, chosen_labels, rejected_ids, rejected_labels)
        )

    mx.set_default_device(mx.gpu)
    mx.random.seed(42)
    sampler = locked.DarwinProcessMemorySampler()
    sampler.start()
    sampler_active = True
    try:
        load_started = time.perf_counter()
        model, _config = locked.load_locked_mlx_gemma4(
            model_dir,
            mx,
            load_config_fn=lambda path: json.loads(
                (path / "config.json").read_text(encoding="utf-8")
            ),
            get_model_classes_fn=lambda **_kwargs: (
                mlx_gemma4.Model,
                mlx_gemma4.ModelArgs,
            ),
        )
        mx.eval(model.parameters())
        mx.synchronize()
        model.freeze()
        base_inventory = locked.require_bf16_base_model(model, mx)

        chosen = [mx.array([item[0]], dtype=mx.int32) for item in padded_examples]
        chosen_y = [mx.array([item[1]], dtype=mx.int32) for item in padded_examples]
        rejected = [mx.array([item[2]], dtype=mx.int32) for item in padded_examples]
        rejected_y = [mx.array([item[3]], dtype=mx.int32) for item in padded_examples]

        def sequence_logp(current_model: Any, tokens: Any, labels: Any) -> Any:
            logits = current_model(tokens)[:, :-1, :].astype(mx.float32)
            shifted = labels[:, 1:]
            mask = shifted != -100
            safe = mx.where(mask, shifted, mx.zeros_like(shifted))
            losses = nn.losses.cross_entropy(logits, safe)
            return -(losses * mask).sum()

        reference_started = time.perf_counter()
        ref_chosen = [
            sequence_logp(model, tokens, labels)
            for tokens, labels in zip(chosen, chosen_y)
        ]
        ref_rejected = [
            sequence_logp(model, tokens, labels)
            for tokens, labels in zip(rejected, rejected_y)
        ]
        mx.eval(*ref_chosen, *ref_rejected)
        mx.synchronize()
        reference_seconds = time.perf_counter() - reference_started

        targets = locked.target_module_names(
            model, lock, case.model_key, case.target_preset
        )
        target_set = set(targets)
        updates = []
        for name, module in model.named_modules():
            if name not in target_set:
                continue
            if not isinstance(module, nn.Linear):
                raise DpoBenchmarkContractError(f"non-linear LoRA target: {name}")
            updates.append(
                (
                    name,
                    LoRALinear.from_base(module, r=16, scale=2.0, dropout=0.0),
                )
            )
        if {name for name, _module in updates} != target_set:
            raise DpoBenchmarkContractError("incomplete LoRA target conversion")
        model.update_modules(tree_unflatten(updates))
        trainable_inventory = locked.require_exact_trainables(model, targets, mx)
        locked.load_exact_initial_adapter(model, targets, adapter, mx)
        model.train()
        mx.eval(model.state)
        mx.synchronize()
        initial_trainables = {
            name: mx.array(value)
            for name, value in tree_flatten(model.trainable_parameters())
        }
        mx.eval(*initial_trainables.values())
        load_seconds = time.perf_counter() - load_started

        optimizer = optim.AdamW(
            learning_rate=case.learning_rate,
            betas=(case.optimizer["beta1"], case.optimizer["beta2"]),
            eps=case.optimizer["epsilon"],
            weight_decay=case.optimizer["weight_decay"],
            bias_correction=True,
        )

        def dpo_loss(
            current_model: Any,
            chosen_tokens: Any,
            chosen_targets: Any,
            rejected_tokens: Any,
            rejected_targets: Any,
            reference_chosen: Any,
            reference_rejected: Any,
        ) -> Any:
            policy_chosen = sequence_logp(
                current_model, chosen_tokens, chosen_targets
            )
            policy_rejected = sequence_logp(
                current_model, rejected_tokens, rejected_targets
            )
            diff = case.beta * (
                (policy_chosen - reference_chosen)
                - (policy_rejected - reference_rejected)
            )
            return mx.logaddexp(mx.array(0.0, dtype=mx.float32), -diff)

        loss_and_grad = nn.value_and_grad(model, dpo_loss)
        state = [model.state, optimizer.state, mx.random.state]

        def step(
            chosen_tokens: Any,
            chosen_targets: Any,
            rejected_tokens: Any,
            rejected_targets: Any,
            reference_chosen: Any,
            reference_rejected: Any,
        ) -> Any:
            loss, gradients = loss_and_grad(
                model,
                chosen_tokens,
                chosen_targets,
                rejected_tokens,
                rejected_targets,
                reference_chosen,
                reference_rejected,
            )
            gradients, _norm = optim.clip_grad_norm(
                gradients, case.optimizer["max_grad_norm"]
            )
            optimizer.update(model, gradients)
            return loss

        compiled = mx.compile(step, inputs=state, outputs=state)

        update_example_indices: list[int] = []

        def execute() -> tuple[float, float]:
            example_index = len(update_example_indices) % len(case.examples)
            update_example_indices.append(example_index)
            started = time.perf_counter()
            loss = compiled(
                chosen[example_index],
                chosen_y[example_index],
                rejected[example_index],
                rejected_y[example_index],
                ref_chosen[example_index],
                ref_rejected[example_index],
            )
            mx.eval(loss, model.state, optimizer.state)
            mx.synchronize()
            elapsed = time.perf_counter() - started
            value = float(loss.item())
            if not math.isfinite(value):
                raise DpoBenchmarkContractError("non-finite MLX DPO loss")
            return elapsed, value

        cold = execute()
        first_update_trainables: dict[str, Any] | None = None
        if args.first_update_adapter_output is not None:
            # MLX arrays are immutable; retaining these evaluated leaves pins the
            # exact post-cold-update state while subsequent optimizer steps
            # replace the model leaves.  Serialize only after timing completes.
            first_update_trainables = dict(
                tree_flatten(model.trainable_parameters())
            )
            mx.eval(*first_update_trainables.values())
        first = execute()
        warmup = [execute() for _ in range(case.protocol["warmup"])]
        measured = [execute() for _ in range(case.protocol["measured"])]
        if abs(cold[1] - math.log(2.0)) > 1e-5:
            raise DpoBenchmarkContractError(
                f"zero-adapter cold loss is not ln(2): {cold[1]}"
            )
        if not any(abs(loss - cold[1]) > 1e-6 for _seconds, loss in [first, *warmup, *measured]):
            raise DpoBenchmarkContractError("MLX policy did not move after optimization")

        final_trainables = dict(tree_flatten(model.trainable_parameters()))
        delta_squares = [
            ((final_trainables[name] - initial).astype(mx.float32) ** 2).sum()
            for name, initial in initial_trainables.items()
        ]
        delta_maxima = [
            mx.abs(final_trainables[name] - initial).max()
            for name, initial in initial_trainables.items()
        ]
        mx.eval(*delta_squares, *delta_maxima)
        adapter_delta_l2 = math.sqrt(
            sum(float(value.item()) for value in delta_squares)
        )
        adapter_delta_max_abs = max(
            float(value.item()) for value in delta_maxima
        )
        memory = sampler.stop()
        sampler_active = False
    finally:
        if sampler_active:
            sampler.stop()

    def evaluate_current_model() -> dict[str, Any]:
        current_chosen = [
            sequence_logp(model, tokens, labels)
            for tokens, labels in zip(chosen, chosen_y)
        ]
        current_rejected = [
            sequence_logp(model, tokens, labels)
            for tokens, labels in zip(rejected, rejected_y)
        ]
        mx.eval(*current_chosen, *current_rejected)
        mx.synchronize()
        return dpo_evaluation_metrics(
            [float(value.item()) for value in current_chosen],
            [float(value.item()) for value in current_rejected],
            [float(value.item()) for value in ref_chosen],
            [float(value.item()) for value in ref_rejected],
            case.beta,
        )

    model.eval()
    post_training_evaluation = evaluate_current_model()
    comparison_adapter_evaluation: dict[str, Any] | None = None
    if comparison_path is not None:
        comparison_sha256 = "sha256:" + hashlib.sha256(
            comparison_path.read_bytes()
        ).hexdigest()
        comparison_artifact = replace(
            adapter,
            checkpoint=comparison_path,
            checkpoint_sha256=comparison_sha256,
        )
        try:
            locked.load_exact_initial_adapter(
                model, targets, comparison_artifact, mx
            )
        except locked.ContractError as exc:
            raise DpoBenchmarkContractError(
                f"could not load comparison adapter: {exc}"
            ) from exc
        comparison_adapter_evaluation = {
            "path": str(comparison_path),
            "sha256": comparison_sha256,
            **evaluate_current_model(),
        }

    try:
        locked.require_files_unchanged(bound_input_identities)
        locked.require_files_unchanged(native_runtime["bound_file_identities"])
        post_model = locked.verify_model_directory(lock, case.model_key, model_dir)
        if post_model != locked_model:
            raise locked.ContractError("locked model identity drifted during benchmark")
        if locked.verify_packages(lock, "mlx_reference") != package_versions:
            raise locked.ContractError("MLX package environment drifted during benchmark")
        post_mlx_checkout = locked.verify_source_checkout(
            args.mlx_source_root,
            revisions["mlx"],
            source_name="MLX",
        )
        post_mlx_lm_checkout = locked.verify_source_checkout(
            args.mlx_lm_source_root,
            revisions["mlx-lm"],
            source_name="MLX-LM",
        )
        postverified_native_bundle = locked.verify_mlx_native_build_before_import(
            args,
            lock,
            post_mlx_checkout,
            post_mlx_lm_checkout,
        )
        post_native_runtime = locked.verify_mlx_native_runtime(
            args,
            lock,
            post_mlx_checkout,
            mx,
            postverified_native_bundle,
        )
        for field in ("native_artifact_inventory", "build_attestation"):
            if post_native_runtime[field] != native_runtime[field]:
                raise locked.ContractError(
                    f"MLX {field} drifted during DPO benchmark"
                )
    except locked.ContractError as exc:
        raise DpoBenchmarkContractError(
            f"DPO benchmark input postflight failed: {exc}"
        ) from exc

    first_update_adapter_output_path: str | None = None
    first_update_adapter_output_sha256: str | None = None
    if first_update_output is not None:
        destination = first_update_output
        if first_update_trainables is None:
            raise DpoBenchmarkContractError(
                "first-update adapter snapshot was not captured"
            )
        first_update_adapter_output_sha256 = write_adapter_exclusive(
            destination,
            final_trainables=first_update_trainables,
            target_names=targets,
            adapter=adapter,
            mx=mx,
        )
        first_update_adapter_output_path = str(destination.resolve(strict=True))

    adapter_output_path: str | None = None
    adapter_output_sha256: str | None = None
    if final_output is not None:
        destination = final_output
        adapter_output_sha256 = write_adapter_exclusive(
            destination,
            final_trainables=final_trainables,
            target_names=targets,
            adapter=adapter,
            mx=mx,
        )
        adapter_output_path = str(destination.resolve(strict=True))

    measured_seconds = [entry[0] for entry in measured]
    example_summaries = [
        {
            "index": index,
            "source_row_index": example.source_row_index,
            "source_id": example.source_id,
            "score_chosen": example.score_chosen,
            "score_rejected": example.score_rejected,
            "prompt_tokens": len(example.prompt_token_ids),
            "chosen_tokens": len(example.chosen_token_ids),
            "rejected_tokens": len(example.rejected_token_ids),
            "pair_sequence_length": pair_lengths[index],
            "reference_chosen_logp": float(ref_chosen[index].item()),
            "reference_rejected_logp": float(ref_rejected[index].item()),
        }
        for index, example in enumerate(case.examples)
    ]
    case_summary: dict[str, Any] = {
        "schema_version": case.schema_version,
        "name": case.name,
        "path": str(case.source_path),
        "semantic_sha256": case.semantic_sha256,
        "examples": len(case.examples),
        "epochs": sum(case.protocol.values()) // len(case.examples),
        "example_token_counts": [
            {
                "prompt": len(example.prompt_token_ids),
                "chosen": len(example.chosen_token_ids),
                "rejected": len(example.rejected_token_ids),
            }
            for example in case.examples
        ],
    }
    if len(case.examples) == 1:
        case_summary.update(
            {
                "prompt_tokens": len(case.prompt_token_ids),
                "chosen_tokens": len(case.chosen_token_ids),
                "rejected_tokens": len(case.rejected_token_ids),
            }
        )

    payload = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "framework": "mlx-lm",
        "algorithm": "dpo",
        "model_key": case.model_key,
        "sequence_length": case.sequence_length,
        "sequence_length_policy": sequence_length_policy,
        "case": case_summary,
        "dataset": case.dataset,
        "examples": example_summaries,
        "update_example_indices": update_example_indices,
        "protocol": case.protocol,
        "optimizer": {
            "learning_rate": case.learning_rate,
            **case.optimizer,
        },
        "beta": case.beta,
        "load_seconds": load_seconds,
        "reference_precompute_seconds": reference_seconds,
        "reference_chosen_logps": [float(value.item()) for value in ref_chosen],
        "reference_rejected_logps": [float(value.item()) for value in ref_rejected],
        "cold_seconds": cold[0],
        "cold_loss": cold[1],
        "first_seconds": first[0],
        "first_loss": first[1],
        "warmup_seconds": [entry[0] for entry in warmup],
        "warmup_losses": [entry[1] for entry in warmup],
        "measured_seconds": measured_seconds,
        "measured_losses": [entry[1] for entry in measured],
        "median_seconds": statistics.median(measured_seconds),
        "mean_seconds": statistics.mean(measured_seconds),
        "adapter_delta_l2": adapter_delta_l2,
        "adapter_delta_max_abs": adapter_delta_max_abs,
        "first_update_adapter_output_path": first_update_adapter_output_path,
        "first_update_adapter_output_sha256": first_update_adapter_output_sha256,
        "adapter_output_path": adapter_output_path,
        "adapter_output_sha256": adapter_output_sha256,
        "post_training_evaluation": post_training_evaluation,
        "comparison_adapter_evaluation": comparison_adapter_evaluation,
        "adapter_semantic_sha256": adapter.semantic_sha256,
        "base_inventory_sha256": base_inventory["inventory_sha256"],
        "trainable_inventory_sha256": trainable_inventory["inventory_sha256"],
        "peak_phys_footprint_bytes": memory.peak_phys_footprint_bytes,
        "mlx_allocator_peak_bytes": int(mx.get_peak_memory()),
        "mlx_revision": mlx_revision,
        "mlx_lm_revision": mlx_lm_revision,
        "locked_package_versions": package_versions,
        "python_version": actual_python,
        "base_model_provenance": base_model_provenance,
        "locked_model": locked_model,
        "mlx_native_runtime": {
            "native_artifact_inventory": native_runtime[
                "native_artifact_inventory"
            ],
            "build_attestation": native_runtime["build_attestation"],
        },
        "mlx_core_path": str(core_path),
        "mlx_lm_gemma4_path": str(gemma4_source_path),
        "mlx_lm_lora_path": str(lora_source_path),
        "runner_sha256": "sha256:"
        + hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
    }
    if len(case.examples) == 1:
        payload["reference_chosen_logp"] = float(ref_chosen[0].item())
        payload["reference_rejected_logp"] = float(ref_rejected[0].item())
    return payload


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--adapter-dir", type=Path, required=True)
    result.add_argument("--mlx-source-root", type=Path, required=True)
    result.add_argument("--mlx-lm-source-root", type=Path, required=True)
    result.add_argument(
        "--mlx-build-attestation",
        type=Path,
        required=True,
        help="strict local attestation binding the loaded MLX native runtime",
    )
    result.add_argument("--case", type=Path, default=DEFAULT_CASE_PATH)
    result.add_argument("--lock", type=Path, default=locked.LOCK_PATH)
    result.add_argument(
        "--sequence-length-bucket-quantum",
        type=int,
        help="round the maximum chosen/rejected pair length to this quantum",
    )
    result.add_argument(
        "--sequence-length-bucket-min",
        type=int,
        help="minimum pair shape; requires --sequence-length-bucket-quantum",
    )
    result.add_argument("--first-update-adapter-output", type=Path)
    result.add_argument("--adapter-output", type=Path)
    result.add_argument("--evaluation-adapter", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        payload = run(args)
        write_json_exclusive(args.output, payload)
    except (DpoBenchmarkContractError, locked.ContractError) as exc:
        print(f"Gemma4 DPO MLX benchmark contract error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
