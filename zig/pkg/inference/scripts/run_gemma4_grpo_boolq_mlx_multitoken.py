#!/usr/bin/env python3
"""Run matched multi-token Gemma4 GRPO campaigns against MLX.

This runner consumes one completed Antfly Metal campaign and the pinned BoolQ
materialization that produced it. It executes two MLX lanes from the identical
seed adapter:

* ``trace_replay`` trains on Antfly's exact completion sequences and rewards;
* ``native_rollout`` is the retired deterministic ranked multi-token rollout.
  The acceptance loader fails closed for stochastic Antfly reports until this
  lane has a matching categorical sampler and statistical behavioral gates.

Both lanes use a frozen base-equivalent reference, one optimizer update per
completion group, token-normalized GRPO loss, the same hard raw-K3 KL budget,
and the same proportional next-group KL controller as Antfly. The result is a
bounded campaign artifact, not a claim of broad or long-horizon quality parity.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
RESULT_SCHEMA_VERSION = "antfly_gemma4_grpo_boolq_mlx_multitoken/v1"
MATERIALIZATION_SCHEMA_VERSION = "antfly_gemma4_grpo_boolq_materialization/v1"
REWARD_TRACE_SCHEMA_VERSION = "antfly_inference_grpo_reward_trace/v1"
KL_TRACE_SCHEMA_VERSION = "antfly_inference_grpo_kl_control_trace/v2"
GRPO_REPORT_SCHEMA_VERSIONS = frozenset(
    {
        "antfly_inference_finetune_grpo_report/v4",
        "antfly_inference_finetune_grpo_report/v5",
        "antfly_inference_finetune_grpo_report/v6",
        "antfly_inference_finetune_grpo_report/v7",
    }
)
GRPO_EVAL_SCHEMA_VERSIONS = frozenset(
    {
        "antfly_inference_finetune_grpo_evaluation/v2",
        "antfly_inference_finetune_grpo_evaluation/v3",
        "antfly_inference_finetune_grpo_evaluation/v4",
    }
)
MODEL_KEYS = ("gemma-4-E2B-it", "gemma-4-E4B-it")
TARGET_PRESET = "peft-qv"
SEQUENCE_LENGTH = 128
LEARNING_RATE = 1.0e-7
OPTIMIZER = {
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1.0e-8,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}
GRPO = {
    "clip_epsilon": 0.2,
    "initial_kl_coef": 0.04,
    "advantage_epsilon": 1.0e-4,
    "train_max_kl": 0.1,
    "target_kl": 0.01,
    "kl_horizon": 100.0,
    "min_kl_coef": 0.001,
    "max_kl_coef": 1.0,
}

sys.path.insert(0, str(SCRIPT_DIR))
import run_gemma4_grpo_boolq_mlx_parity as legacy  # noqa: E402
import run_gemma4_grpo_mlx_benchmark as microbenchmark  # noqa: E402

locked = microbenchmark.locked


class MultiTokenParityError(RuntimeError):
    """A pinned input, runtime, algorithm, or artifact contract drifted."""


@dataclass(frozen=True)
class CampaignSpec:
    model_key: str
    train_groups: int
    eval_groups: int
    group_size: int
    max_completion_tokens: int

    def validate(self) -> None:
        if self.model_key not in MODEL_KEYS:
            raise MultiTokenParityError("unsupported Gemma4 model key")
        if self.train_groups < 2 or self.eval_groups < 2:
            raise MultiTokenParityError("matched campaigns require at least two train/eval groups")
        if not 2 <= self.group_size <= 8:
            raise MultiTokenParityError("group size must be in [2, 8]")
        if not 2 <= self.max_completion_tokens <= 32:
            raise MultiTokenParityError("multi-token completion budget must be in [2, 32]")
        if self.max_completion_tokens >= SEQUENCE_LENGTH:
            raise MultiTokenParityError("completion budget exceeds the sequence contract")


@dataclass(frozen=True)
class BoolQRow:
    prompt: str
    target: str
    prompt_token_ids: tuple[int, ...]
    source_split: str
    source_row_index: int
    source_id: str


@dataclass(frozen=True)
class TraceCompletion:
    token_ids: tuple[int, ...]
    reward: float


@dataclass(frozen=True)
class TraceGroup:
    prompt_index: int
    completions: tuple[TraceCompletion, ...]

    @property
    def sequences(self) -> tuple[tuple[int, ...], ...]:
        return tuple(completion.token_ids for completion in self.completions)

    @property
    def first_token_ids(self) -> tuple[int, ...]:
        return tuple(completion.token_ids[0] for completion in self.completions)

    @property
    def rewards(self) -> tuple[float, ...]:
        return tuple(completion.reward for completion in self.completions)


@dataclass(frozen=True)
class AcceptanceEvidence:
    root: Path
    config: Mapping[str, Any]
    train_report: Mapping[str, Any]
    eval_report: Mapping[str, Any]
    train_trace: tuple[TraceGroup, ...]
    eval_trace: tuple[TraceGroup, ...]
    trained_adapter_dir: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path, where: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MultiTokenParityError(f"could not load {where}: {exc}") from exc
    if not isinstance(payload, dict):
        raise MultiTokenParityError(f"{where} root must be an object")
    return payload


def finite_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MultiTokenParityError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MultiTokenParityError(f"{where} must be finite")
    return result


def require_close(value: Any, expected: float, where: str, tolerance: float = 1.0e-6) -> None:
    if abs(finite_float(value, where) - expected) > tolerance:
        raise MultiTokenParityError(f"{where} differs from the matched campaign")


def normalized_advantages(rewards: Sequence[float], epsilon: float) -> list[float]:
    if not rewards:
        raise MultiTokenParityError("cannot normalize an empty reward group")
    mean = statistics.mean(rewards)
    variance = (
        sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
        if len(rewards) > 1
        else 0.0
    )
    denominator = math.sqrt(variance) + epsilon
    return [(reward - mean) / denominator for reward in rewards]


def adaptive_kl_update(current: float, mean_kl: float) -> float:
    if not math.isfinite(mean_kl) or mean_kl < 0.0:
        raise MultiTokenParityError("adaptive KL observation must be finite and non-negative")
    proportional_error = min(max(mean_kl / GRPO["target_kl"] - 1.0, -0.2), 0.2)
    updated = current * (1.0 + proportional_error / GRPO["kl_horizon"])
    return min(max(updated, GRPO["min_kl_coef"]), GRPO["max_kl_coef"])


def mean_k3(policy_logps: Sequence[float], reference_logps: Sequence[float]) -> float:
    if not policy_logps or len(policy_logps) != len(reference_logps):
        raise MultiTokenParityError("KL vectors must be non-empty and equal-length")
    values: list[float] = []
    for policy, reference in zip(policy_logps, reference_logps):
        difference = reference - policy
        if not math.isfinite(difference) or difference > 80.0:
            raise MultiTokenParityError("GRPO KL log-ratio is outside the finite f32 contract")
        values.append(max(math.expm1(difference) - difference, 0.0))
    return statistics.mean(values)


def prefix_match_reward(decoded: str, target: str) -> float:
    completion = decoded.strip(" \t\r\n")
    expected = target.strip(" \t\r\n")
    return 1.0 if completion.startswith(expected) else 0.0


def decode_reward(tokenizer: Any, token_ids: Sequence[int], target: str) -> tuple[str, float]:
    decoded = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    return decoded, prefix_match_reward(decoded, target)


def sequence_overlap(
    actual: Sequence[Sequence[int]], expected: Sequence[Sequence[int]]
) -> Mapping[str, Any]:
    actual_tuples = tuple(tuple(item) for item in actual)
    expected_tuples = tuple(tuple(item) for item in expected)
    if (
        not actual_tuples
        or len(actual_tuples) != len(expected_tuples)
        or len(set(actual_tuples)) != len(actual_tuples)
        or len(set(expected_tuples)) != len(expected_tuples)
    ):
        raise MultiTokenParityError("completion groups must be distinct and equal-length")
    overlap = len(set(actual_tuples) & set(expected_tuples))
    actual_first = tuple(item[0] for item in actual_tuples)
    expected_first = tuple(item[0] for item in expected_tuples)
    first_overlap = len(set(actual_first) & set(expected_first))
    return {
        "sequence_overlap": overlap,
        "sequence_recall": overlap / len(expected_tuples),
        "exact_sequence_set": set(actual_tuples) == set(expected_tuples),
        "exact_sequence_order": actual_tuples == expected_tuples,
        "top_sequence_match": actual_tuples[0] == expected_tuples[0],
        "first_token_overlap": first_overlap,
        "first_token_recall": first_overlap / len(expected_first),
        "exact_first_token_order": actual_first == expected_first,
        "top1_first_token_match": actual_first[0] == expected_first[0],
    }


def summarize_overlaps(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise MultiTokenParityError("cannot summarize empty overlap evidence")
    return {
        "groups": len(rows),
        "mean_sequence_recall": statistics.mean(float(row["sequence_recall"]) for row in rows),
        "exact_sequence_set_rate": statistics.mean(1.0 if row["exact_sequence_set"] else 0.0 for row in rows),
        "exact_sequence_order_rate": statistics.mean(1.0 if row["exact_sequence_order"] else 0.0 for row in rows),
        "top_sequence_match_rate": statistics.mean(1.0 if row["top_sequence_match"] else 0.0 for row in rows),
        "mean_first_token_recall": statistics.mean(float(row["first_token_recall"]) for row in rows),
        "exact_first_token_order_rate": statistics.mean(1.0 if row["exact_first_token_order"] else 0.0 for row in rows),
        "top1_first_token_match_rate": statistics.mean(1.0 if row["top1_first_token_match"] else 0.0 for row in rows),
    }


def load_trace(
    path: Path,
    *,
    phase: str,
    expected_groups: int,
    group_size: int,
    max_completion_tokens: int,
) -> tuple[TraceGroup, ...]:
    groups: dict[int, list[TraceCompletion]] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MultiTokenParityError(f"could not load {phase} reward trace: {exc}") from exc
    for line_index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MultiTokenParityError(f"{phase} reward trace contains invalid JSON") from exc
        if not isinstance(row, dict) or row.get("schema_version") != REWARD_TRACE_SCHEMA_VERSION:
            raise MultiTokenParityError(f"{phase} reward trace schema drifted")
        if row.get("phase") != phase or row.get("call_index") != line_index:
            raise MultiTokenParityError(f"{phase} reward trace order drifted")
        prompt_index = row.get("prompt_index")
        if isinstance(prompt_index, bool) or not isinstance(prompt_index, int) or prompt_index < 0:
            raise MultiTokenParityError(f"{phase} prompt index is invalid")
        raw_tokens = row.get("completion_tokens")
        if (
            not isinstance(raw_tokens, list)
            or not 1 <= len(raw_tokens) <= max_completion_tokens
            or any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in raw_tokens)
        ):
            raise MultiTokenParityError(f"{phase} completion token sequence is invalid")
        reward = finite_float(row.get("aggregate_reward"), "aggregate_reward")
        if reward not in (0.0, 1.0):
            raise MultiTokenParityError("BoolQ trace reward must be binary")
        groups.setdefault(prompt_index, []).append(
            TraceCompletion(tuple(raw_tokens), reward)
        )
    if sorted(groups) != list(range(expected_groups)):
        raise MultiTokenParityError(f"{phase} reward trace prompt groups are incomplete")
    result = tuple(TraceGroup(index, tuple(groups[index])) for index in range(expected_groups))
    for group in result:
        if len(group.completions) != group_size:
            raise MultiTokenParityError(f"{phase} reward trace group size drifted")
    return result


def load_materialization(
    path: Path, spec: CampaignSpec, model_dir: Path
) -> Mapping[str, Any]:
    manifest = load_json(path.expanduser().resolve(), "BoolQ materialization manifest")
    if manifest.get("schema_version") != MATERIALIZATION_SCHEMA_VERSION:
        raise MultiTokenParityError("unsupported BoolQ materialization schema")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("repo_id") != "google/boolq":
        raise MultiTokenParityError("materialization is not google/boolq")
    revision = dataset.get("revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(char not in "0123456789abcdef" for char in revision)
    ):
        raise MultiTokenParityError("BoolQ revision must be a full lowercase Git commit")
    policy = dataset.get("selection_policy")
    expected_policy = {
        "dataset_format": "rendered-text-grpo",
        "max_seq_len": SEQUENCE_LENGTH,
        "max_completion_tokens": spec.max_completion_tokens,
        "target_tokens": 1,
        "rendered_prompt_truncation": "forbidden",
        "response_channel": "final",
    }
    if not isinstance(policy, dict) or any(policy.get(key) != value for key, value in expected_policy.items()):
        raise MultiTokenParityError("BoolQ selection policy differs from the campaign")
    for section, manifest_key in (("train", "train_jsonl"), ("evaluation", "eval_jsonl")):
        record = dataset.get(section)
        jsonl_path = Path(str(manifest.get(manifest_key, ""))).expanduser().resolve()
        if not isinstance(record, dict) or not jsonl_path.is_file():
            raise MultiTokenParityError(f"materialized {section} JSONL is missing")
        if record.get("materialized_jsonl_sha256") != sha256_file(jsonl_path):
            raise MultiTokenParityError(f"materialized {section} JSONL SHA-256 drifted")
    train_ids = manifest.get("train_source_ids")
    eval_ids = manifest.get("eval_source_ids")
    if not isinstance(train_ids, list) or not isinstance(eval_ids, list):
        raise MultiTokenParityError("materialization source identities are missing")
    if len(train_ids) < spec.train_groups or len(eval_ids) < spec.eval_groups:
        raise MultiTokenParityError("materialization has too few rows for the campaign")
    if set(train_ids) & set(eval_ids):
        raise MultiTokenParityError("BoolQ train and evaluation identities overlap")
    tokenizer_files = manifest.get("tokenizer_files")
    if not isinstance(tokenizer_files, dict):
        raise MultiTokenParityError("materialization tokenizer fingerprints are missing")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        tokenizer_path = model_dir / name
        if not tokenizer_path.is_file() or tokenizer_files.get(name) != sha256_file(tokenizer_path):
            raise MultiTokenParityError(f"runtime {name} differs from the materialization")
    return manifest


def load_rows(
    path: Path,
    *,
    expected_count: int,
    expected_ids: Sequence[str],
    expected_indices: Sequence[int],
    tokenizer: Any,
    max_completion_tokens: int,
) -> tuple[BoolQRow, ...]:
    rows: list[BoolQRow] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MultiTokenParityError(f"could not load BoolQ JSONL: {exc}") from exc
    for line_index, line in enumerate(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MultiTokenParityError(f"BoolQ JSONL line {line_index + 1} is invalid") from exc
        if not isinstance(payload, dict) or set(payload) != {"prompt", "target", "metadata"}:
            raise MultiTokenParityError("BoolQ JSONL row schema drifted")
        prompt, target, metadata = payload["prompt"], payload["target"], payload["metadata"]
        if not isinstance(prompt, str) or not prompt or target not in ("yes", "no") or not isinstance(metadata, dict):
            raise MultiTokenParityError("BoolQ JSONL row is malformed")
        prompt_ids = tuple(int(value) for value in tokenizer.encode(prompt, add_special_tokens=False).ids)
        target_ids = tuple(int(value) for value in tokenizer.encode(target, add_special_tokens=False).ids)
        if (
            not prompt_ids
            or len(prompt_ids) + max_completion_tokens > SEQUENCE_LENGTH
            or len(target_ids) != 1
            or metadata.get("prompt_tokens") != len(prompt_ids)
            or metadata.get("target_tokens") != 1
        ):
            raise MultiTokenParityError("BoolQ tokenizer/length contract drifted")
        rows.append(
            BoolQRow(
                prompt=prompt,
                target=target,
                prompt_token_ids=prompt_ids,
                source_split=str(metadata.get("source_split", "")),
                source_row_index=int(metadata.get("source_row_index", -1)),
                source_id=str(metadata.get("source_id", "")),
            )
        )
    if len(rows) < expected_count:
        raise MultiTokenParityError("BoolQ JSONL has too few admitted rows")
    selected = tuple(rows[:expected_count])
    if [row.source_id for row in selected] != list(expected_ids[:expected_count]):
        raise MultiTokenParityError("BoolQ source identity order drifted")
    if [row.source_row_index for row in selected] != list(expected_indices[:expected_count]):
        raise MultiTokenParityError("BoolQ source row order drifted")
    return selected


def validate_kl_trace(root: Path, report: Mapping[str, Any], spec: CampaignSpec) -> None:
    telemetry = report.get("kl_control")
    if not isinstance(telemetry, dict):
        raise MultiTokenParityError("Antfly adaptive KL telemetry is missing")
    if (
        telemetry.get("mode") != "adaptive"
        or telemetry.get("budget_policy") != "skip_group"
        or telemetry.get("admitted_groups") != spec.train_groups
        or telemetry.get("rejected_groups") != 0
    ):
        raise MultiTokenParityError("Antfly adaptive KL telemetry counts drifted")
    for field, expected in (
        ("train_max_kl", GRPO["train_max_kl"]),
        ("target_kl", GRPO["target_kl"]),
        ("kl_horizon", GRPO["kl_horizon"]),
        ("initial_kl_coef", GRPO["initial_kl_coef"]),
        ("min_kl_coef", GRPO["min_kl_coef"]),
        ("max_kl_coef", GRPO["max_kl_coef"]),
    ):
        require_close(telemetry.get(field), expected, f"kl_control.{field}")
    trace_path = root / "grpo_kl_control_trace.jsonl"
    if telemetry.get("trace_path") != str(trace_path):
        raise MultiTokenParityError("Antfly KL trace path escaped the campaign root")
    if telemetry.get("trace_digest") != "sha256:" + sha256_file(trace_path):
        raise MultiTokenParityError("Antfly KL trace digest drifted")
    lines = trace_path.read_text(encoding="utf-8").splitlines()
    if len(lines) != spec.train_groups:
        raise MultiTokenParityError("Antfly KL trace group count drifted")
    previous_after: float | None = None
    for index, line in enumerate(lines):
        row = json.loads(line)
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != KL_TRACE_SCHEMA_VERSION
            or row.get("group_index") != index
            or row.get("status") != "admitted"
            or row.get("budget_policy") != "skip_group"
        ):
            raise MultiTokenParityError("Antfly KL trace decision/order drifted")
        before = finite_float(row.get("kl_coef_before"), "kl_coef_before")
        after = finite_float(row.get("kl_coef_after"), "kl_coef_after")
        observed = finite_float(row.get("mean_kl"), "mean_kl")
        if observed > GRPO["train_max_kl"]:
            raise MultiTokenParityError("Antfly admitted a group above the hard KL budget")
        if previous_after is not None and abs(before - previous_after) > 1.0e-7:
            raise MultiTokenParityError("Antfly KL coefficient trajectory is discontinuous")
        expected_after = adaptive_kl_update(before, observed)
        if abs(after - expected_after) > 2.0e-7:
            raise MultiTokenParityError("Antfly adaptive KL update differs from the matched rule")
        previous_after = after


def load_acceptance(
    root: Path,
    manifest: Mapping[str, Any],
    spec: CampaignSpec,
    model_dir: Path,
    adapter_dir: Path,
) -> AcceptanceEvidence:
    evidence_root = root.expanduser().resolve()
    config = load_json(evidence_root / "training_config.json", "Antfly training config")
    train_report = load_json(evidence_root / "grpo_report.json", "Antfly GRPO report")
    eval_report = load_json(evidence_root / "grpo_evaluation_report.json", "Antfly GRPO evaluation report")
    if train_report.get("schema_version") not in GRPO_REPORT_SCHEMA_VERSIONS:
        raise MultiTokenParityError("Antfly GRPO report is not the adaptive-KL schema")
    try:
        legacy.require_native_rollout_sampler_compatibility(train_report)
    except legacy.BoolQParityContractError as exc:
        raise MultiTokenParityError(str(exc)) from exc
    if eval_report.get("schema_version") not in GRPO_EVAL_SCHEMA_VERSIONS:
        raise MultiTokenParityError("Antfly GRPO evaluation is not the raw-KL schema")
    if train_report.get("execution_mode") != "train" or train_report.get("dataset_format") != "rendered-text-grpo":
        raise MultiTokenParityError("Antfly evidence is not optimizer-backed rendered GRPO")
    expected_counts = {
        "groups": spec.train_groups,
        "completions": spec.train_groups * spec.group_size,
        "optimizer_steps": spec.train_groups,
    }
    if any(train_report.get(key) != value for key, value in expected_counts.items()):
        raise MultiTokenParityError("Antfly training counts differ from the matched campaign")
    if train_report.get("policy_backend") != "metal":
        raise MultiTokenParityError("Antfly campaign must run on Metal")
    if eval_report.get("status") != "passed" or eval_report.get("groups") != spec.eval_groups:
        raise MultiTokenParityError("Antfly held-out campaign did not pass")
    if eval_report.get("mask_truncated_completions") is not False:
        raise MultiTokenParityError("Antfly evaluation truncation policy drifted")
    if train_report.get("mean_kl") is None or eval_report.get("mean_kl") is None:
        raise MultiTokenParityError("Antfly raw KL metrics are missing")
    if train_report.get("schema_version") == "antfly_inference_finetune_grpo_report/v7":
        if (
            train_report.get("optimizer_groups") != spec.train_groups
            or train_report.get("zero_reward_std_groups") != 0
            or train_report.get("all_truncated_groups") != 0
            or train_report.get("kl_rejected_groups") != 0
            or float(train_report.get("frac_reward_zero_std", -1.0)) != 0.0
            or float(train_report.get("frac_kl_rejected", -1.0)) != 0.0
            or train_report.get("loss_type") != "bnpo"
            or train_report.get("scale_rewards") != "group"
            or float(train_report.get("epsilon_low", 0.0)) != GRPO["clip_epsilon"]
            or float(train_report.get("epsilon_high", 0.0)) != GRPO["clip_epsilon"]
            or train_report.get("max_completion_tokens") != spec.max_completion_tokens
            or train_report.get("mask_truncated_completions") is not False
            or train_report.get("num_iterations") != 1
        ):
            raise MultiTokenParityError("Antfly GRPO v7 objective semantics drifted")
        truncated = train_report.get("truncated_completions")
        truncated_fraction = train_report.get("frac_completions_truncated")
        if (
            isinstance(truncated, bool)
            or not isinstance(truncated, int)
            or not 0 <= truncated <= expected_counts["completions"]
            or not isinstance(truncated_fraction, (int, float))
            or not math.isclose(
                float(truncated_fraction),
                truncated / expected_counts["completions"],
                rel_tol=0.0,
                abs_tol=1e-7,
            )
        ):
            raise MultiTokenParityError("Antfly truncated-completion telemetry drifted")
    recipe = config.get("recipe")
    if not isinstance(recipe, dict):
        raise MultiTokenParityError("Antfly normalized recipe is missing")
    model = recipe.get("model")
    dataset = recipe.get("dataset")
    adapter = recipe.get("adapter")
    optimizer = recipe.get("optimizer")
    grpo = recipe.get("grpo")
    evaluation = recipe.get("eval")
    if not all(isinstance(value, dict) for value in (model, dataset, adapter, optimizer, grpo, evaluation)):
        raise MultiTokenParityError("Antfly normalized recipe is incomplete")
    if Path(str(model.get("path", ""))).resolve() != model_dir or model.get("family") != "gemma4":
        raise MultiTokenParityError("Antfly model identity differs from the campaign")
    if Path(str(adapter.get("path", ""))).resolve() != adapter_dir:
        raise MultiTokenParityError("Antfly seed adapter differs from the campaign")
    if adapter.get("rank") != 16 or float(adapter.get("alpha", 0.0)) != 32.0:
        raise MultiTokenParityError("Antfly adapter rank/alpha drifted")
    if adapter.get("target_preset") not in (None, TARGET_PRESET):
        raise MultiTokenParityError("Antfly adapter target preset drifted")
    if (
        dataset.get("path") != manifest.get("train_jsonl")
        or evaluation.get("path") != manifest.get("eval_jsonl")
        or dataset.get("max_examples") != spec.train_groups
        or evaluation.get("max_examples") != spec.eval_groups
        or dataset.get("max_seq_len") != SEQUENCE_LENGTH
    ):
        raise MultiTokenParityError("Antfly dataset is not the pinned matched split")
    require_close(optimizer.get("learning_rate"), LEARNING_RATE, "optimizer.learning_rate", 1.0e-14)
    if optimizer.get("epochs") != 1 or optimizer.get("gradient_accumulation_steps") != 1:
        raise MultiTokenParityError("Antfly optimizer schedule drifted")
    require_close(optimizer.get("max_grad_norm"), OPTIMIZER["max_grad_norm"], "optimizer.max_grad_norm")
    if grpo.get("group_size") != spec.group_size or grpo.get("max_completion_tokens") != spec.max_completion_tokens:
        raise MultiTokenParityError("Antfly group/completion shape drifted")
    for field, expected in (
        ("clip_epsilon", GRPO["clip_epsilon"]),
        ("kl_coef", GRPO["initial_kl_coef"]),
        ("train_max_kl", GRPO["train_max_kl"]),
        ("target_kl", GRPO["target_kl"]),
        ("kl_horizon", GRPO["kl_horizon"]),
        ("min_kl_coef", GRPO["min_kl_coef"]),
        ("max_kl_coef", GRPO["max_kl_coef"]),
    ):
        require_close(grpo.get(field), expected, f"grpo.{field}")
    if (
        grpo.get("adaptive_kl") is not True
        or grpo.get("normalize_advantage") is not True
        or grpo.get("loss_type") not in (None, "bnpo")
        or grpo.get("scale_rewards") not in (None, "group")
        or grpo.get("epsilon_high") not in (None, GRPO["clip_epsilon"])
        or grpo.get("mask_truncated_completions") not in (None, False)
    ):
        raise MultiTokenParityError("Antfly adaptive/advantage policy drifted")
    validate_kl_trace(evidence_root, train_report, spec)
    train_trace_path = evidence_root / "grpo_reward_trace.jsonl"
    eval_trace_path = evidence_root / "grpo_evaluation_reward_trace.jsonl"
    for trace_path, report in ((train_trace_path, train_report), (eval_trace_path, eval_report)):
        telemetry = report.get("reward_pipeline")
        if not isinstance(telemetry, dict) or telemetry.get("trace_digest") != "sha256:" + sha256_file(trace_path):
            raise MultiTokenParityError("Antfly reward trace digest drifted")
    trained_adapter_dir = Path(str(train_report.get("trained_adapter_dir", ""))).resolve()
    if trained_adapter_dir.parent != evidence_root or not trained_adapter_dir.is_dir():
        raise MultiTokenParityError("Antfly trained adapter escaped the campaign root")
    return AcceptanceEvidence(
        root=evidence_root,
        config=config,
        train_report=train_report,
        eval_report=eval_report,
        train_trace=load_trace(
            train_trace_path,
            phase="train",
            expected_groups=spec.train_groups,
            group_size=spec.group_size,
            max_completion_tokens=spec.max_completion_tokens,
        ),
        eval_trace=load_trace(
            eval_trace_path,
            phase="evaluation",
            expected_groups=spec.eval_groups,
            group_size=spec.group_size,
            max_completion_tokens=spec.max_completion_tokens,
        ),
        trained_adapter_dir=trained_adapter_dir,
    )


def validate_trace_rewards(
    tokenizer: Any, rows: Sequence[BoolQRow], trace: Sequence[TraceGroup]
) -> None:
    if len(rows) != len(trace):
        raise MultiTokenParityError("BoolQ rows and trace groups differ")
    for row, group in zip(rows, trace):
        for completion in group.completions:
            _decoded, reward = decode_reward(tokenizer, completion.token_ids, row.target)
            if reward != completion.reward:
                raise MultiTokenParityError(
                    f"Antfly reward trace cannot be reproduced for prompt {group.prompt_index}"
                )


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise MultiTokenParityError(f"campaign output already exists: {destination}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    spec = CampaignSpec(
        model_key=args.model_key,
        train_groups=args.train_groups,
        eval_groups=args.eval_groups,
        group_size=args.group_size,
        max_completion_tokens=args.max_completion_tokens,
    )
    spec.validate()
    model_dir = args.model_dir.expanduser().resolve()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    manifest = load_materialization(args.dataset_manifest, spec, model_dir)
    acceptance = load_acceptance(
        args.antfly_run_root,
        manifest,
        spec,
        model_dir,
        adapter_dir,
    )

    lock = locked.load_lock(args.lock)
    mlx_contract = lock["mlx_reference"]
    locked.force_offline_environment()
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != mlx_contract["python"]:
        raise MultiTokenParityError(
            f"MLX campaign requires Python {mlx_contract['python']}, found {actual_python}"
        )
    if (
        platform.system() != mlx_contract["required_platform"]
        or platform.machine() != mlx_contract["required_machine"]
    ):
        raise MultiTokenParityError("MLX campaign must run on the locked Apple platform")
    runtime_root = args.mlx_runtime_root.expanduser().resolve()
    runtime_attestation = legacy.attest_wheel_runtime(
        runtime_root=runtime_root,
        wheel_path=args.mlx_wheel,
        metal_wheel_path=args.mlx_metal_wheel,
        expected_version=mlx_contract["packages"]["mlx"],
    )
    runtime_attestation = {
        **runtime_attestation,
        "locked_source_revision": mlx_contract["source_revisions"]["mlx"],
        "source_revision_verified": False,
    }
    mlx_lm_revision = microbenchmark.require_source_revision(
        args.mlx_lm_source_root,
        mlx_contract["source_revisions"]["mlx-lm"],
        "MLX-LM",
    )

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten
    from tokenizers import tokenizers as tokenizers_native

    core_path = Path(mx.__file__ or "").resolve()
    if not microbenchmark._path_is_within(core_path, runtime_root):
        raise MultiTokenParityError(
            f"imported MLX is outside the attested runtime: {core_path}"
        )
    microbenchmark.install_mlx_lm_source_namespace(args.mlx_lm_source_root)
    from mlx_lm._version import __version__ as mlx_lm_version
    from mlx_lm.models import gemma4 as mlx_gemma4
    from mlx_lm.tuner.lora import LoRALinear

    actual_versions = microbenchmark.require_exact_package_versions(
        {"mlx": str(mx.__version__), "mlx-lm": str(mlx_lm_version)},
        {
            "mlx": mlx_contract["packages"]["mlx"],
            "mlx-lm": mlx_contract["packages"]["mlx-lm"],
        },
    )
    tokenizers_version = str(tokenizers_native.__version__)
    if tokenizers_version != manifest.get("dependency_versions", {}).get("tokenizers"):
        raise MultiTokenParityError("tokenizers version differs from the materializer")
    mlx_lm_root = args.mlx_lm_source_root.expanduser().resolve()
    for label, source_path in (
        ("MLX-LM Gemma4", Path(mlx_gemma4.__file__ or "").resolve()),
        ("MLX-LM LoRA", Path(sys.modules[LoRALinear.__module__].__file__ or "").resolve()),
    ):
        if not microbenchmark._path_is_within(source_path, mlx_lm_root):
            raise MultiTokenParityError(f"imported {label} escaped the attested checkout")

    tokenizer = tokenizers_native.Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    train_rows = load_rows(
        Path(str(manifest["train_jsonl"])),
        expected_count=spec.train_groups,
        expected_ids=manifest["train_source_ids"],
        expected_indices=manifest["train_source_row_indices"],
        tokenizer=tokenizer,
        max_completion_tokens=spec.max_completion_tokens,
    )
    eval_rows = load_rows(
        Path(str(manifest["eval_jsonl"])),
        expected_count=spec.eval_groups,
        expected_ids=manifest["eval_source_ids"],
        expected_indices=manifest["eval_source_row_indices"],
        tokenizer=tokenizer,
        max_completion_tokens=spec.max_completion_tokens,
    )
    validate_trace_rewards(tokenizer, train_rows, acceptance.train_trace)
    validate_trace_rewards(tokenizer, eval_rows, acceptance.eval_trace)

    adapter_manifest = load_json(
        adapter_dir / "antfly_finetune_manifest.json", "seed adapter manifest"
    )
    binding_fields = ("base_model_sha256", "tokenizer_sha256", "chat_template_sha256")
    prepared_summary = {key: adapter_manifest.get(key) for key in binding_fields}
    base_model_provenance = locked.zig_model_provenance(model_dir)
    if prepared_summary != base_model_provenance:
        raise MultiTokenParityError("seed adapter does not match the model")
    seed_adapter = locked.inspect_initial_adapter(
        adapter_dir, lock, spec.model_key, TARGET_PRESET, prepared_summary
    )
    antfly_trained = locked.inspect_initial_adapter(
        acceptance.trained_adapter_dir,
        lock,
        spec.model_key,
        TARGET_PRESET,
        prepared_summary,
    )

    mx.set_default_device(mx.gpu)
    mx.random.seed(42)
    sampler = locked.DarwinProcessMemorySampler()
    sampler.start()
    sampler_active = True
    campaign_started = time.perf_counter()
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
        targets = locked.target_module_names(
            model, lock, spec.model_key, TARGET_PRESET
        )
        target_set = set(targets)
        module_updates = []
        for name, module in model.named_modules():
            if name not in target_set:
                continue
            if not isinstance(module, nn.Linear):
                raise MultiTokenParityError(f"non-linear LoRA target: {name}")
            module_updates.append(
                (name, LoRALinear.from_base(module, r=16, scale=2.0, dropout=0.0))
            )
        if {name for name, _module in module_updates} != target_set:
            raise MultiTokenParityError("incomplete LoRA target conversion")
        model.update_modules(tree_unflatten(module_updates))
        trainable_inventory = locked.require_exact_trainables(model, targets, mx)
        locked.load_exact_initial_adapter(model, targets, seed_adapter, mx)
        model.train()
        mx.eval(model.state)
        mx.synchronize()
        initial_trainables = {
            name: value + mx.zeros_like(value)
            for name, value in tree_flatten(model.trainable_parameters())
        }
        mx.eval(*initial_trainables.values())
        mx.synchronize()
        load_seconds = time.perf_counter() - load_started

        config_payload = load_json(model_dir / "config.json", "Gemma4 config")
        text_config = config_payload.get("text_config")
        if not isinstance(text_config, dict):
            raise MultiTokenParityError("Gemma4 text config is missing")
        eos_token_id = int(text_config.get("eos_token_id", -1))
        if eos_token_id < 0:
            raise MultiTokenParityError("Gemma4 EOS token is invalid")

        def restore_trainables(values: Mapping[str, Any]) -> None:
            model.update(tree_unflatten(list(values.items())), strict=True)
            mx.eval(model.trainable_parameters())
            mx.synchronize()

        def snapshot_trainables() -> Mapping[str, Any]:
            values = {
                name: value + mx.zeros_like(value)
                for name, value in tree_flatten(model.trainable_parameters())
            }
            mx.eval(*values.values())
            mx.synchronize()
            return values

        def reset_to_initial() -> None:
            restore_trainables(initial_trainables)

        def padded_sequence(
            row: BoolQRow, sequence: Sequence[int]
        ) -> tuple[Any, Any, Any]:
            values = [int(token) for token in sequence]
            if not 1 <= len(values) <= spec.max_completion_tokens:
                raise MultiTokenParityError("completion length drifted")
            joined = list(row.prompt_token_ids) + values
            if len(joined) > SEQUENCE_LENGTH:
                raise MultiTokenParityError("completion exceeds the sequence contract")
            return (
                mx.array(
                    [joined + [0] * (SEQUENCE_LENGTH - len(joined))],
                    dtype=mx.int32,
                ),
                mx.array(
                    values + [0] * (spec.max_completion_tokens - len(values)),
                    dtype=mx.int32,
                ),
                mx.array(
                    [1.0] * len(values)
                    + [0.0] * (spec.max_completion_tokens - len(values)),
                    dtype=mx.float32,
                ),
            )

        def padded_sequences(
            row: BoolQRow, sequences: Sequence[Sequence[int]]
        ) -> tuple[Any, Any, Any]:
            if len(sequences) != spec.group_size:
                raise MultiTokenParityError("completion group size drifted")
            rows = [padded_sequence(row, sequence) for sequence in sequences]
            return (
                mx.concatenate([values[0] for values in rows], axis=0),
                mx.stack([values[1] for values in rows], axis=0),
                mx.stack([values[2] for values in rows], axis=0),
            )

        def selected_logps(
            current_model: Any,
            tokens: Any,
            selected: Any,
            mask: Any,
            prompt_length: int,
        ) -> Any:
            logits = current_model(tokens).astype(mx.float32)
            columns = []
            for step in range(spec.max_completion_tokens):
                predictor = logits[:, prompt_length - 1 + step, :]
                logprobs = predictor - mx.logsumexp(
                    predictor, axis=-1, keepdims=True
                )
                token_logps = mx.take_along_axis(
                    logprobs, selected[:, step : step + 1], axis=-1
                )
                columns.append(token_logps)
            return mx.concatenate(columns, axis=1) * mask

        def score_sequences(
            row: BoolQRow, sequences: Sequence[Sequence[int]]
        ) -> list[list[float]]:
            # Keep every candidate score at batch=1. This matches Antfly's
            # completion path and avoids batch-dependent quantized Gemma4
            # logits observed in the initial multi-token parity probe.
            result: list[list[float]] = []
            for sequence in sequences:
                tokens, selected, mask = padded_sequence(row, sequence)
                values = selected_logps(
                    model,
                    tokens,
                    selected[None, :],
                    mask[None, :],
                    len(row.prompt_token_ids),
                )
                mx.eval(values)
                mx.synchronize()
                result.append(
                    [float(value) for value in values[0, : len(sequence)].tolist()]
                )
            return result

        def ranked_group(
            row: BoolQRow,
        ) -> tuple[list[list[int]], list[list[float]]]:
            prompt = list(row.prompt_token_ids)
            prompt_batch = mx.array(
                [prompt + [0] * (SEQUENCE_LENGTH - len(prompt))],
                dtype=mx.int32,
            )
            logits = model(prompt_batch).astype(mx.float32)[0, len(prompt) - 1, :]
            candidate_ids = mx.argpartition(
                -logits, kth=spec.group_size - 1
            )[: spec.group_size]
            order = mx.argsort(-logits[candidate_ids])
            first_tokens = candidate_ids[order]
            first_logprobs = logits - mx.logsumexp(logits)
            first_values = first_logprobs[first_tokens]
            mx.eval(first_tokens, first_values)
            mx.synchronize()
            sequences = [[int(value)] for value in first_tokens.tolist()]
            logps = [[float(value)] for value in first_values.tolist()]
            active = [sequence[0] != eos_token_id for sequence in sequences]
            for _step in range(1, spec.max_completion_tokens):
                active_indices = [index for index, enabled in enumerate(active) if enabled]
                if not active_indices:
                    break
                row_index = len(prompt) + _step - 1
                for completion_index in active_indices:
                    joined = prompt + sequences[completion_index]
                    tokens = mx.array(
                        [joined + [0] * (SEQUENCE_LENGTH - len(joined))],
                        dtype=mx.int32,
                    )
                    predictor = model(tokens).astype(mx.float32)[0, row_index, :]
                    ranked = mx.argpartition(
                        -predictor, kth=spec.group_size - 1
                    )[: spec.group_size]
                    ranked_scores = predictor[ranked]
                    ranked = ranked[mx.argsort(-ranked_scores)]
                    chosen = ranked[completion_index % spec.group_size]
                    chosen_logp = predictor[chosen] - mx.logsumexp(predictor)
                    mx.eval(chosen, chosen_logp)
                    mx.synchronize()
                    token_id = int(chosen.item())
                    sequences[completion_index].append(token_id)
                    logps[completion_index].append(
                        float(chosen_logp.item())
                    )
                    if token_id == eos_token_id:
                        active[completion_index] = False
            return sequences, logps

        def flatten(values: Sequence[Sequence[float]]) -> list[float]:
            return [item for row in values for item in row]

        def grpo_metrics(
            sequences: Sequence[Sequence[int]],
            old_logps: Sequence[Sequence[float]],
            policy_logps: Sequence[Sequence[float]],
            reference_logps: Sequence[Sequence[float]],
            rewards: Sequence[float],
            kl_coef: float,
        ) -> Mapping[str, float]:
            advantages = normalized_advantages(
                rewards, GRPO["advantage_epsilon"]
            )
            pg_values: list[float] = []
            kl_values: list[float] = []
            clipped = 0
            total = 0
            for completion_index, sequence in enumerate(sequences):
                if not (
                    len(sequence)
                    == len(old_logps[completion_index])
                    == len(policy_logps[completion_index])
                    == len(reference_logps[completion_index])
                ):
                    raise MultiTokenParityError("GRPO token/logprob lengths drifted")
                advantage = advantages[completion_index]
                for old, policy, reference in zip(
                    old_logps[completion_index],
                    policy_logps[completion_index],
                    reference_logps[completion_index],
                ):
                    ratio = math.exp(policy - old)
                    unclipped = ratio * advantage
                    clipped_value = (
                        min(
                            max(ratio, 1.0 - GRPO["clip_epsilon"]),
                            1.0 + GRPO["clip_epsilon"],
                        )
                        * advantage
                    )
                    pg_values.append(-min(unclipped, clipped_value))
                    difference = reference - policy
                    raw_kl = max(math.expm1(difference) - difference, 0.0)
                    kl_values.append(raw_kl)
                    clipped += int(clipped_value < unclipped)
                    total += 1
            pg_loss = statistics.mean(pg_values)
            raw_mean_kl = statistics.mean(kl_values)
            kl_loss = kl_coef * raw_mean_kl
            return {
                "loss": pg_loss + kl_loss,
                "pg_loss": pg_loss,
                "kl_loss": kl_loss,
                "mean_kl": raw_mean_kl,
                "clip_fraction": clipped / total,
            }

        def make_optimizer() -> Any:
            return optim.AdamW(
                learning_rate=LEARNING_RATE,
                betas=(OPTIMIZER["beta1"], OPTIMIZER["beta2"]),
                eps=OPTIMIZER["epsilon"],
                weight_decay=OPTIMIZER["weight_decay"],
                bias_correction=True,
            )

        def completion_grpo_loss(
            current_model: Any,
            tokens: Any,
            selected: Any,
            mask: Any,
            old_logps: Any,
            reference_logps: Any,
            advantage: Any,
            kl_coef: Any,
            prompt_length: Any,
            group_token_count: Any,
        ) -> tuple[Any, Any, Any, Any, Any, Any]:
            # Each differentiable completion forward is physically batch=1.
            # The group token count preserves Antfly's group-level reduction.
            logits = current_model(tokens).astype(mx.float32)
            predictor_rows = prompt_length + mx.arange(
                spec.max_completion_tokens, dtype=mx.int32
            ) - 1
            predictors = logits[0, predictor_rows, :]
            logprobs = predictors - mx.logsumexp(
                predictors, axis=-1, keepdims=True
            )
            new_logps = mx.take_along_axis(
                logprobs, selected[:, None], axis=-1
            )[:, 0]
            ratio = mx.exp(new_logps - old_logps)
            pg_unclipped = ratio * advantage
            pg_clipped = mx.clip(
                ratio,
                1.0 - GRPO["clip_epsilon"],
                1.0 + GRPO["clip_epsilon"],
            ) * advantage
            pg_tokens = -mx.minimum(pg_unclipped, pg_clipped)
            difference = reference_logps - new_logps
            raw_kl_tokens = mx.maximum(mx.expm1(difference) - difference, 0.0)
            pg_loss = mx.sum(pg_tokens * mask) / group_token_count
            mean_kl_value = mx.sum(raw_kl_tokens * mask) / group_token_count
            kl_loss = kl_coef * mean_kl_value
            loss = pg_loss + kl_loss
            clip_fraction = mx.sum(
                (pg_clipped < pg_unclipped).astype(mx.float32) * mask
            ) / group_token_count
            return (
                loss,
                pg_loss,
                kl_loss,
                mean_kl_value,
                clip_fraction,
                new_logps * mask,
            )

        completion_loss_and_grad = nn.value_and_grad(model, completion_grpo_loss)

        def reference_score(
            row: BoolQRow, sequences: Sequence[Sequence[int]]
        ) -> list[list[float]]:
            policy_values = snapshot_trainables()
            reset_to_initial()
            try:
                return score_sequences(row, sequences)
            finally:
                restore_trainables(policy_values)

        reference_started = time.perf_counter()
        reset_to_initial()
        trace_reference = [
            score_sequences(row, group.sequences)
            for row, group in zip(train_rows, acceptance.train_trace)
        ]
        reference_precompute_seconds = time.perf_counter() - reference_started

        def train_lane(
            mode: str,
        ) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
            reset_to_initial()
            optimizer = make_optimizer()
            state = [model.state, optimizer.state, mx.random.state]

            def step(
                tokens: Any,
                selected: Any,
                mask: Any,
                old_logps: Any,
                reference_logps: Any,
                advantages: Any,
                kl_coef: Any,
                prompt_length: Any,
            ) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
                group_token_count = mx.sum(mask)
                completion_metrics = []
                gradients = None
                for completion_index in range(spec.group_size):
                    metrics, completion_gradients = completion_loss_and_grad(
                        model,
                        tokens[completion_index : completion_index + 1],
                        selected[completion_index],
                        mask[completion_index],
                        old_logps[completion_index],
                        reference_logps[completion_index],
                        advantages[completion_index],
                        kl_coef,
                        prompt_length,
                        group_token_count,
                    )
                    completion_metrics.append(metrics)
                    gradients = (
                        completion_gradients
                        if gradients is None
                        else tree_map(
                            lambda accumulated, current: accumulated + current,
                            gradients,
                            completion_gradients,
                        )
                    )
                assert gradients is not None
                gradients, grad_norm = optim.clip_grad_norm(
                    gradients, OPTIMIZER["max_grad_norm"]
                )
                optimizer.update(model, gradients)
                metric_totals = list(completion_metrics[0][:5])
                for metrics in completion_metrics[1:]:
                    for metric_index in range(5):
                        metric_totals[metric_index] = (
                            metric_totals[metric_index] + metrics[metric_index]
                        )
                return (
                    *metric_totals,
                    mx.stack([metrics[5] for metrics in completion_metrics], axis=0),
                    grad_norm,
                )

            compiled_step = mx.compile(step, inputs=state, outputs=state)
            coefficient = GRPO["initial_kl_coef"]
            updates: list[dict[str, Any]] = []
            started_lane = time.perf_counter()
            for update_index, (row, expected) in enumerate(
                zip(train_rows, acceptance.train_trace)
            ):
                started = time.perf_counter()
                native_sequences, native_old = ranked_group(row)
                overlap = sequence_overlap(native_sequences, expected.sequences)
                if mode == "trace_replay":
                    sequences = [list(values) for values in expected.sequences]
                    rewards = list(expected.rewards)
                    old_values = score_sequences(row, sequences)
                    reference_values = trace_reference[update_index]
                elif mode == "native_rollout":
                    sequences = native_sequences
                    rewards = [
                        decode_reward(tokenizer, values, row.target)[1]
                        for values in sequences
                    ]
                    old_values = native_old
                    reference_values = reference_score(row, sequences)
                else:
                    raise AssertionError(mode)
                policy_before = score_sequences(row, sequences)
                sampling_rescore_errors = [
                    abs(scored - sampled)
                    for sampled_values, scored_values in zip(old_values, policy_before)
                    for sampled, scored in zip(sampled_values, scored_values)
                ]
                sampling_rescore_max_abs_error = max(
                    sampling_rescore_errors, default=0.0
                )
                if sampling_rescore_max_abs_error > 1.0e-4:
                    raise MultiTokenParityError(
                        f"MLX {mode} group {update_index} sampling/rescore drifted "
                        f"by {sampling_rescore_max_abs_error:.6g}"
                    )
                raw_mean_kl = mean_k3(
                    flatten(policy_before), flatten(reference_values)
                )
                if raw_mean_kl > GRPO["train_max_kl"]:
                    raise MultiTokenParityError(
                        f"MLX {mode} group {update_index} exceeded the pre-update KL budget"
                    )
                next_coefficient = adaptive_kl_update(coefficient, raw_mean_kl)
                tokens, selected, mask = padded_sequences(row, sequences)
                old_array = mx.zeros(
                    (spec.group_size, spec.max_completion_tokens),
                    dtype=mx.float32,
                )
                reference_array = mx.zeros_like(old_array)
                old_host = [
                    values + [0.0] * (spec.max_completion_tokens - len(values))
                    for values in old_values
                ]
                reference_host = [
                    values + [0.0] * (spec.max_completion_tokens - len(values))
                    for values in reference_values
                ]
                old_array = mx.array(old_host, dtype=mx.float32)
                reference_array = mx.array(reference_host, dtype=mx.float32)
                advantages = mx.array(
                    normalized_advantages(
                        rewards, GRPO["advantage_epsilon"]
                    ),
                    dtype=mx.float32,
                )
                outputs = compiled_step(
                    tokens,
                    selected,
                    mask,
                    old_array,
                    reference_array,
                    advantages,
                    mx.array(coefficient, dtype=mx.float32),
                    mx.array(len(row.prompt_token_ids), dtype=mx.int32),
                )
                mx.eval(*outputs, model.state, optimizer.state)
                mx.synchronize()
                loss, pg_loss, kl_loss, observed_kl, clip_fraction, rescored, grad_norm = outputs
                metrics = {
                    "loss": float(loss.item()),
                    "pg_loss": float(pg_loss.item()),
                    "kl_loss": float(kl_loss.item()),
                    "mean_kl": float(observed_kl.item()),
                    "clip_fraction": float(clip_fraction.item()),
                    "preclip_gradient_l2": float(grad_norm.item()),
                }
                if not all(math.isfinite(value) for value in metrics.values()):
                    raise MultiTokenParityError("MLX GRPO produced a non-finite metric")
                if abs(metrics["mean_kl"] - raw_mean_kl) > 2.0e-4:
                    raise MultiTokenParityError("MLX preflight and differentiable KL disagree")
                rescored_host = rescored.tolist()
                rescore_errors = []
                for completion_index, values in enumerate(old_values):
                    for token_index, old_value in enumerate(values):
                        rescore_errors.append(
                            abs(
                                float(rescored_host[completion_index][token_index])
                                - old_value
                            )
                        )
                differentiable_rescore_max_abs_error = max(
                    rescore_errors, default=0.0
                )
                if differentiable_rescore_max_abs_error > 1.0e-4:
                    raise MultiTokenParityError(
                        f"MLX {mode} group {update_index} differentiable rescore "
                        f"drifted by {differentiable_rescore_max_abs_error:.6g}"
                    )
                updates.append(
                    {
                        "update_index": update_index,
                        "source_id": row.source_id,
                        "target": row.target,
                        "completion_token_ids": sequences,
                        "completion_tokens": sum(len(values) for values in sequences),
                        "rewards": rewards,
                        "candidate_overlap_with_antfly": overlap,
                        **metrics,
                        "sampling_rescore_max_abs_error": (
                            sampling_rescore_max_abs_error
                        ),
                        "differentiable_rescore_max_abs_error": (
                            differentiable_rescore_max_abs_error
                        ),
                        "kl_coef_before": coefficient,
                        "kl_coef_after": next_coefficient,
                        "seconds": time.perf_counter() - started,
                    }
                )
                coefficient = next_coefficient
            lane_seconds = time.perf_counter() - started_lane
            adapter_comparison: Mapping[str, Any] | None = None
            if mode == "trace_replay":
                adapter_comparison = legacy._adapter_delta_comparison(
                    model=model,
                    initial_trainables=initial_trainables,
                    antfly_trained=antfly_trained,
                    target_names=targets,
                    mx=mx,
                    tree_flatten=tree_flatten,
                )
            return (
                {
                    "mode": mode,
                    "optimizer_steps": len(updates),
                    "seconds": lane_seconds,
                    "median_update_seconds": statistics.median(
                        row["seconds"] for row in updates
                    ),
                    "mean_update_seconds": statistics.mean(
                        row["seconds"] for row in updates
                    ),
                    "completion_tokens": sum(
                        int(row["completion_tokens"]) for row in updates
                    ),
                    "mean_reward": statistics.mean(
                        reward for row in updates for reward in row["rewards"]
                    ),
                    "mean_loss": statistics.mean(row["loss"] for row in updates),
                    "mean_kl_loss": statistics.mean(
                        row["kl_loss"] for row in updates
                    ),
                    "mean_kl": statistics.mean(row["mean_kl"] for row in updates),
                    "final_kl_coef": coefficient,
                    "max_mean_kl": max(row["mean_kl"] for row in updates),
                    "candidate_overlap_with_antfly": summarize_overlaps(
                        [row["candidate_overlap_with_antfly"] for row in updates]
                    ),
                    "updates": updates,
                },
                adapter_comparison,
            )

        def evaluate_lane() -> Mapping[str, Any]:
            records: list[dict[str, Any]] = []
            started = time.perf_counter()
            for row, expected in zip(eval_rows, acceptance.eval_trace):
                sequences, sampling_logps = ranked_group(row)
                policy_logps = score_sequences(row, sequences)
                reference_logps = reference_score(row, sequences)
                decoded = [decode_reward(tokenizer, values, row.target) for values in sequences]
                rewards = [value[1] for value in decoded]
                metrics = grpo_metrics(
                    sequences,
                    sampling_logps,
                    policy_logps,
                    reference_logps,
                    rewards,
                    GRPO["initial_kl_coef"],
                )
                records.append(
                    {
                        "source_id": row.source_id,
                        "target": row.target,
                        "completion_token_ids": sequences,
                        "decoded_completions": [value[0] for value in decoded],
                        "rewards": rewards,
                        "candidate_overlap_with_antfly": sequence_overlap(
                            sequences, expected.sequences
                        ),
                        **metrics,
                    }
                )
            elapsed = time.perf_counter() - started
            rewards = [reward for record in records for reward in record["rewards"]]
            mean_reward = statistics.mean(rewards)
            reward_variance = statistics.mean(
                (reward - mean_reward) ** 2 for reward in rewards
            )
            return {
                "groups": len(records),
                "completions": len(records) * spec.group_size,
                "tokens": sum(
                    len(sequence)
                    for record in records
                    for sequence in record["completion_token_ids"]
                ),
                "seconds": elapsed,
                "mean_reward": mean_reward,
                "top_rank_mean_reward": statistics.mean(
                    record["rewards"][0] for record in records
                ),
                "positive_reward_group_rate": statistics.mean(
                    1.0 if any(record["rewards"]) else 0.0 for record in records
                ),
                "reward_stddev": math.sqrt(reward_variance),
                "loss": statistics.mean(record["loss"] for record in records),
                "pg_loss": statistics.mean(record["pg_loss"] for record in records),
                "kl_loss": statistics.mean(record["kl_loss"] for record in records),
                "mean_kl": statistics.mean(record["mean_kl"] for record in records),
                "clip_fraction": statistics.mean(
                    record["clip_fraction"] for record in records
                ),
                "candidate_overlap_with_antfly": summarize_overlaps(
                    [record["candidate_overlap_with_antfly"] for record in records]
                ),
                "rows": records,
            }

        reset_to_initial()
        baseline_evaluation = evaluate_lane()
        trace_training, trace_adapter = train_lane("trace_replay")
        trace_evaluation = evaluate_lane()
        native_training, _native_adapter = train_lane("native_rollout")
        native_evaluation = evaluate_lane()

        memory = sampler.stop()
        sampler_active = False
        campaign_seconds = time.perf_counter() - campaign_started
    finally:
        if sampler_active:
            sampler.stop()

    antfly_eval = {
        key: acceptance.eval_report[key]
        for key in (
            "groups",
            "completions",
            "tokens",
            "mean_reward",
            "top_rank_mean_reward",
            "positive_reward_group_rate",
            "reward_stddev",
            "loss",
            "pg_loss",
            "kl_loss",
            "mean_kl",
            "clip_fraction",
        )
    }
    antfly_train_seconds = sum(
        float(acceptance.train_report.get(key) or 0.0)
        for key in (
            "sampling_seconds",
            "policy_rescore_seconds",
            "reference_scoring_seconds",
            "backward_update_seconds",
        )
    )
    antfly_eval_seconds = float(acceptance.eval_report.get("loop_seconds") or 0.0)
    performance = {
        "antfly_train_accounted_seconds": antfly_train_seconds,
        "mlx_native_train_seconds": native_training["seconds"],
        "antfly_to_mlx_train_time_ratio": (
            antfly_train_seconds / native_training["seconds"]
            if native_training["seconds"] > 0.0
            else None
        ),
        "antfly_eval_loop_seconds": antfly_eval_seconds,
        "mlx_native_eval_seconds": native_evaluation["seconds"],
        "antfly_to_mlx_eval_time_ratio": (
            antfly_eval_seconds / native_evaluation["seconds"]
            if native_evaluation["seconds"] > 0.0
            else None
        ),
    }
    evaluation_deltas = {
        key: native_evaluation[key] - antfly_eval[key]
        for key in (
            "mean_reward",
            "top_rank_mean_reward",
            "positive_reward_group_rate",
            "kl_loss",
            "mean_kl",
        )
    }
    minimums = acceptance.eval_report.get("minimums")
    if not isinstance(minimums, dict):
        raise MultiTokenParityError("Antfly evaluation minimums are missing")
    native_passed = (
        native_evaluation["mean_reward"] >= minimums["mean_reward"]
        and native_evaluation["top_rank_mean_reward"]
        >= minimums["top_rank_mean_reward"]
        and native_evaluation["positive_reward_group_rate"]
        >= minimums["positive_reward_group_rate"]
        and native_evaluation["kl_loss"] <= minimums["max_kl_loss"]
    )
    trace_numerical_close = bool(
        trace_adapter
        and trace_adapter["delta_cosine_similarity"] >= 0.95
        and trace_adapter["delta_l2_relative_difference"] <= 0.1
    )
    native_behavior_close = (
        abs(evaluation_deltas["mean_reward"]) <= 1.0 / spec.group_size
        and abs(evaluation_deltas["top_rank_mean_reward"]) <= 1.0 / spec.eval_groups
        and native_passed
    )
    classification = (
        "bounded-behavior-and-update-parity"
        if trace_numerical_close and native_behavior_close
        else "bounded-campaign-with-measured-drift"
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": "completed",
        "scope": (
            f"real-pinned-boolq-{spec.model_key.lower()}-"
            f"{spec.train_groups}x{spec.eval_groups}-group{spec.group_size}-"
            f"max{spec.max_completion_tokens}-adaptive-kl"
        ),
        "classification": classification,
        "claim_boundary": {
            "broad_grpo_performance_parity": False,
            "long_horizon_quality_parity": False,
            "reason": (
                "This is one deterministic BoolQ campaign with a bounded update horizon; "
                "it validates mechanics and matched local behavior only."
            ),
        },
        "contract": {
            "model_key": spec.model_key,
            "target_preset": TARGET_PRESET,
            "sequence_length": SEQUENCE_LENGTH,
            "train_groups": spec.train_groups,
            "eval_groups": spec.eval_groups,
            "group_size": spec.group_size,
            "max_completion_tokens": spec.max_completion_tokens,
            "learning_rate": LEARNING_RATE,
            "optimizer": OPTIMIZER,
            "grpo": GRPO,
            "reward_mode": "prefix-match",
            "reference_mode": "frozen-base-equivalent-seed-adapter",
            "rollout_mode": "deterministic-rank-per-completion-each-token",
            "loss_normalization": "mean-over-all-unmasked-completion-tokens",
        },
        "dataset": {
            "repo_id": manifest["dataset"]["repo_id"],
            "revision": manifest["dataset"]["revision"],
            "manifest_path": str(args.dataset_manifest.expanduser().resolve()),
            "manifest_sha256": sha256_file(args.dataset_manifest.expanduser().resolve()),
            "train_jsonl_sha256": manifest["dataset"]["train"]["materialized_jsonl_sha256"],
            "eval_jsonl_sha256": manifest["dataset"]["evaluation"]["materialized_jsonl_sha256"],
        },
        "antfly": {
            "run_root": str(acceptance.root),
            "grpo_report_sha256": sha256_file(acceptance.root / "grpo_report.json"),
            "evaluation_report_sha256": sha256_file(
                acceptance.root / "grpo_evaluation_report.json"
            ),
            "reward_trace_sha256": sha256_file(
                acceptance.root / "grpo_reward_trace.jsonl"
            ),
            "evaluation_reward_trace_sha256": sha256_file(
                acceptance.root / "grpo_evaluation_reward_trace.jsonl"
            ),
            "kl_control_trace_sha256": sha256_file(
                acceptance.root / "grpo_kl_control_trace.jsonl"
            ),
            "trained_adapter_checkpoint_sha256": sha256_file(
                acceptance.trained_adapter_dir / "adapter_model.safetensors"
            ),
            "training": {
                key: acceptance.train_report[key]
                for key in (
                    "groups",
                    "completions",
                    "tokens",
                    "loss",
                    "pg_loss",
                    "kl_loss",
                    "mean_kl",
                    "mean_reward",
                    "kl_control",
                )
            },
            "evaluation": antfly_eval,
        },
        "mlx": {
            "baseline_evaluation": baseline_evaluation,
            "trace_replay": {
                "training": trace_training,
                "evaluation": trace_evaluation,
                "adapter_delta_comparison_with_antfly": trace_adapter,
            },
            "native_rollout": {
                "training": native_training,
                "evaluation": native_evaluation,
                "evaluation_delta_from_antfly": evaluation_deltas,
                "passed_antfly_quality_minimums": native_passed,
            },
            "performance": performance,
            "load_seconds": load_seconds,
            "reference_precompute_seconds": reference_precompute_seconds,
            "campaign_seconds": campaign_seconds,
            "peak_phys_footprint_bytes": memory.peak_phys_footprint_bytes,
            "mlx_allocator_peak_bytes": int(mx.get_peak_memory()),
            "base_inventory_sha256": base_inventory["inventory_sha256"],
            "trainable_inventory_sha256": trainable_inventory["inventory_sha256"],
            "seed_adapter_semantic_sha256": seed_adapter.semantic_sha256,
            "mlx_runtime_attestation": runtime_attestation,
            "mlx_lm_revision": mlx_lm_revision,
            "package_versions": actual_versions,
            "tokenizers_version": tokenizers_version,
            "python_version": actual_python,
            "mlx_core_path": str(core_path),
        },
        "parity_assessment": {
            "classification": classification,
            "trace_numerical_close": trace_numerical_close,
            "native_behavior_close": native_behavior_close,
            "native_quality_gate": native_passed,
        },
        "base_model_provenance": base_model_provenance,
        "runner_sha256": "sha256:" + hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-key", choices=MODEL_KEYS, required=True)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--adapter-dir", type=Path, required=True)
    result.add_argument("--dataset-manifest", type=Path, required=True)
    result.add_argument("--antfly-run-root", type=Path, required=True)
    result.add_argument("--train-groups", type=int, required=True)
    result.add_argument("--eval-groups", type=int, required=True)
    result.add_argument("--group-size", type=int, default=4)
    result.add_argument("--max-completion-tokens", type=int, default=4)
    result.add_argument("--mlx-runtime-root", type=Path, required=True)
    result.add_argument("--mlx-wheel", type=Path, required=True)
    result.add_argument("--mlx-metal-wheel", type=Path, required=True)
    result.add_argument("--mlx-lm-source-root", type=Path, required=True)
    result.add_argument("--lock", type=Path, default=locked.LOCK_PATH)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        payload = run(args)
        write_json_exclusive(args.output, payload)
    except (
        MultiTokenParityError,
        legacy.BoolQParityContractError,
        microbenchmark.GrpoBenchmarkContractError,
        locked.ContractError,
        OSError,
        ValueError,
    ) as exc:
        print(f"Gemma4 multi-token GRPO MLX campaign error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": payload["status"],
                "scope": payload["scope"],
                "classification": payload["classification"],
                "output": str(args.output.expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
