#!/usr/bin/env python3
"""Run a provenance-locked real-BoolQ Gemma4 GRPO parity campaign in MLX.

The legacy campaign has two complementary lanes starting from the exact
Antfly seed adapter:

* ``trace_replay`` uses Antfly's recorded completion groups. This holds
  prompts, tokens, rewards, update order, optimizer, and reference semantics
  fixed so adapter-update differences are attributable to the implementation.
* ``native_rollout`` implements the retired ranked top-k sampler. The loader
  fails closed for stochastic Antfly evidence until this lane implements
  matching seeded categorical sampling and statistical behavioral gates.

The runner is offline-only. It consumes a pinned BoolQ materialization and a
completed Antfly acceptance root, and imports MLX lazily so contract tests need
only the Python standard library.
"""

from __future__ import annotations

import argparse
import array
from collections import Counter
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
import zipfile
from dataclasses import dataclass
from email.parser import Parser
from pathlib import Path
from typing import Any, Mapping, Sequence

import materialize_gemma4_grpo_boolq as boolq_materializer


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
RESULT_SCHEMA_VERSION = "antfly_gemma4_grpo_boolq_mlx_parity/v2"
MATERIALIZATION_SCHEMA_VERSION = boolq_materializer.SCHEMA_VERSION
MATERIALIZATION_SCHEMA_VERSIONS = boolq_materializer.SCHEMA_VERSIONS
TRACE_SCHEMA_VERSION = "antfly_inference_grpo_reward_trace/v1"
GRPO_REPORT_SCHEMA_VERSIONS = frozenset(
    {
        "antfly_inference_finetune_grpo_report/v3",
        "antfly_inference_finetune_grpo_report/v4",
        "antfly_inference_finetune_grpo_report/v5",
        "antfly_inference_finetune_grpo_report/v6",
        "antfly_inference_finetune_grpo_report/v7",
        "antfly_inference_finetune_grpo_report/v8",
    }
)
GRPO_EVAL_SCHEMA_VERSIONS = frozenset(
    {
        "antfly_inference_finetune_grpo_evaluation/v1",
        "antfly_inference_finetune_grpo_evaluation/v2",
        "antfly_inference_finetune_grpo_evaluation/v3",
        "antfly_inference_finetune_grpo_evaluation/v4",
    }
)
GRPO_KL_TRACE_SCHEMA_VERSION = "antfly_inference_grpo_kl_control_trace/v2"
GRPO_TRAINING_ORDER = {
    "algorithm": "seeded-fisher-yates-per-epoch/v1",
    "stream_derivation": "run-seed-order-domain-epoch-dataset-size/v1",
    "prompt_index_semantics": "original-dataset-index",
}
ANTFLY_LEGACY_REFERENCE_MODE = "compiled-zero-lora"
ANTFLY_BATCHED_REFERENCE_MODE = "compiled-zero-lora-shared-prompt-candidate-row"
ANTFLY_PHASED_EVALUATION_ORDER = (
    "policy-sampling-pass-then-frozen-reference-pass"
)
FIXED_MODEL_KEY = "gemma-4-E2B-it"
FIXED_TARGET_PRESET = "peft-qv"
FIXED_SEQUENCE_LENGTH = 128
FIXED_TRAIN_GROUPS = 8
FIXED_EVAL_GROUPS = 64
FIXED_GROUP_SIZE = 8
FIXED_MAX_COMPLETION_TOKENS = 1
FIXED_LEARNING_RATE = 1.0e-7
FIXED_OPTIMIZER = {
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 1.0e-8,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
}
FIXED_GRPO = {
    "clip_epsilon": 0.2,
    "kl_coef": 0.04,
    "advantage_epsilon": 1.0e-4,
}
BEHAVIORAL_PARITY_LIMITS = {
    "max_baseline_mean_reward_abs_delta": 1.0 / 512.0,
    "max_native_mean_reward_abs_delta": 4.0 / 512.0,
    "max_top_rank_mean_reward_abs_delta": 1.0 / 64.0,
    "min_candidate_recall": 0.98,
    "min_top1_match_rate": 63.0 / 64.0,
}
NUMERICAL_PARITY_LIMITS = {
    "min_adapter_delta_cosine_similarity": 0.99,
    "max_adapter_delta_l2_relative_difference": 0.05,
    "max_adapter_delta_abs_difference": 2.0e-6,
    "max_evaluation_kl_abs_delta": 1.0e-5,
}

sys.path.insert(0, str(SCRIPT_DIR))
import run_gemma4_grpo_mlx_benchmark as microbenchmark  # noqa: E402

locked = microbenchmark.locked


class BoolQParityContractError(RuntimeError):
    """The pinned dataset, Antfly evidence, runtime, or result drifted."""


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
    token_id: int
    reward: float


@dataclass(frozen=True)
class TraceGroup:
    prompt_index: int
    completions: tuple[TraceCompletion, ...]

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(item.token_id for item in self.completions)

    @property
    def rewards(self) -> tuple[float, ...]:
        return tuple(item.reward for item in self.completions)


@dataclass(frozen=True)
class AcceptanceEvidence:
    root: Path
    training_config: Mapping[str, Any]
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


def attest_wheel_runtime(
    *,
    runtime_root: Path,
    wheel_path: Path,
    metal_wheel_path: Path,
    expected_version: str,
) -> Mapping[str, Any]:
    """Bind an extracted MLX runtime to exact, versioned wheel archives."""

    runtime_root = runtime_root.expanduser().resolve()
    wheel_specs = (
        (
            "mlx",
            wheel_path.expanduser().resolve(),
            ("mlx/core.", ".so"),
            (),
        ),
        (
            "mlx-metal",
            metal_wheel_path.expanduser().resolve(),
            (),
            (
                "mlx/lib/libjaccl.dylib",
                "mlx/lib/libmlx.dylib",
                "mlx/lib/mlx.metallib",
            ),
        ),
    )
    wheels: dict[str, Any] = {}
    for expected_name, archive_path, dynamic_member, fixed_members in wheel_specs:
        if not archive_path.is_file():
            raise BoolQParityContractError(
                f"{expected_name} wheel does not exist: {archive_path}"
            )
        try:
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                metadata_names = [
                    name for name in names if name.endswith(".dist-info/METADATA")
                ]
                if len(metadata_names) != 1:
                    raise BoolQParityContractError(
                        f"{expected_name} wheel must contain exactly one METADATA file"
                    )
                metadata = Parser().parsestr(
                    archive.read(metadata_names[0]).decode("utf-8")
                )
                actual_name = str(metadata.get("Name", "")).lower()
                actual_version = str(metadata.get("Version", ""))
                if actual_name != expected_name or actual_version != expected_version:
                    raise BoolQParityContractError(
                        f"wheel contract drifted: expected {expected_name}=={expected_version}, "
                        f"found {actual_name}=={actual_version}"
                    )
                required_members = list(fixed_members)
                if dynamic_member:
                    prefix, suffix = dynamic_member
                    matches = [
                        name
                        for name in names
                        if name.startswith(prefix) and name.endswith(suffix)
                    ]
                    if len(matches) != 1:
                        raise BoolQParityContractError(
                            f"{expected_name} wheel must contain exactly one native core"
                        )
                    required_members.append(matches[0])
                members: dict[str, Any] = {}
                for member in required_members:
                    if member not in names:
                        raise BoolQParityContractError(
                            f"{expected_name} wheel is missing {member}"
                        )
                    archive_bytes = archive.read(member)
                    extracted = runtime_root / member
                    if not extracted.is_file():
                        raise BoolQParityContractError(
                            f"wheel runtime is missing extracted member {extracted}"
                        )
                    extracted_digest = sha256_file(extracted)
                    archive_digest = hashlib.sha256(archive_bytes).hexdigest()
                    if extracted_digest != archive_digest:
                        raise BoolQParityContractError(
                            f"extracted wheel member differs from archive: {member}"
                        )
                    members[member] = {
                        "sha256": archive_digest,
                        "size_bytes": len(archive_bytes),
                    }
        except (OSError, UnicodeDecodeError, zipfile.BadZipFile) as exc:
            raise BoolQParityContractError(
                f"could not attest {expected_name} wheel: {exc}"
            ) from exc
        wheels[expected_name] = {
            "archive_path": str(archive_path),
            "archive_sha256": sha256_file(archive_path),
            "name": expected_name,
            "version": expected_version,
            "members": members,
        }
    return {
        "mode": "versioned-wheel-archive",
        "runtime_root": str(runtime_root),
        "wheels": wheels,
    }


def assess_parity(
    *,
    antfly_evaluation: Mapping[str, float],
    baseline_evaluation: Mapping[str, Any],
    native_evaluation: Mapping[str, Any],
    trace_training: Mapping[str, Any],
    trace_adapter: Mapping[str, Any],
    native_quality_passed: bool,
) -> Mapping[str, Any]:
    """Separate task-behavior agreement from update-level numerical parity."""

    baseline_reward_delta = abs(
        baseline_evaluation["mean_reward"] - antfly_evaluation["mean_reward"]
    )
    baseline_top_rank_delta = abs(
        baseline_evaluation["top_rank_mean_reward"]
        - antfly_evaluation["top_rank_mean_reward"]
    )
    native_reward_delta = abs(
        native_evaluation["mean_reward"] - antfly_evaluation["mean_reward"]
    )
    native_top_rank_delta = abs(
        native_evaluation["top_rank_mean_reward"]
        - antfly_evaluation["top_rank_mean_reward"]
    )
    behavioral_checks = {
        "native_quality_gate": native_quality_passed,
        "baseline_mean_reward": baseline_reward_delta
        <= BEHAVIORAL_PARITY_LIMITS["max_baseline_mean_reward_abs_delta"],
        "baseline_top_rank_mean_reward": baseline_top_rank_delta
        <= BEHAVIORAL_PARITY_LIMITS["max_top_rank_mean_reward_abs_delta"],
        "baseline_candidate_recall": baseline_evaluation[
            "candidate_overlap_with_antfly"
        ]["mean_recall"]
        >= BEHAVIORAL_PARITY_LIMITS["min_candidate_recall"],
        "baseline_top1": baseline_evaluation["candidate_overlap_with_antfly"][
            "top1_match_rate"
        ]
        >= BEHAVIORAL_PARITY_LIMITS["min_top1_match_rate"],
        "native_mean_reward": native_reward_delta
        <= BEHAVIORAL_PARITY_LIMITS["max_native_mean_reward_abs_delta"],
        "native_top_rank_mean_reward": native_top_rank_delta
        <= BEHAVIORAL_PARITY_LIMITS["max_top_rank_mean_reward_abs_delta"],
        "native_candidate_recall": native_evaluation[
            "candidate_overlap_with_antfly"
        ]["mean_recall"]
        >= BEHAVIORAL_PARITY_LIMITS["min_candidate_recall"],
        "native_top1": native_evaluation["candidate_overlap_with_antfly"][
            "top1_match_rate"
        ]
        >= BEHAVIORAL_PARITY_LIMITS["min_top1_match_rate"],
        "trace_replay_exact_training_candidate_sets": trace_training[
            "candidate_overlap_with_antfly"
        ]["exact_set_rate"]
        == 1.0,
    }
    evaluation_kl_delta = abs(
        native_evaluation["kl_loss"] - antfly_evaluation["kl_loss"]
    )
    numerical_checks = {
        "adapter_delta_direction": trace_adapter["delta_cosine_similarity"]
        >= NUMERICAL_PARITY_LIMITS["min_adapter_delta_cosine_similarity"],
        "adapter_delta_norm": trace_adapter["delta_l2_relative_difference"]
        <= NUMERICAL_PARITY_LIMITS["max_adapter_delta_l2_relative_difference"],
        "adapter_delta_elementwise": trace_adapter["delta_max_abs_difference"]
        <= NUMERICAL_PARITY_LIMITS["max_adapter_delta_abs_difference"],
        "evaluation_kl": evaluation_kl_delta
        <= NUMERICAL_PARITY_LIMITS["max_evaluation_kl_abs_delta"],
    }
    behavioral_passed = all(behavioral_checks.values())
    numerical_passed = all(numerical_checks.values())
    if behavioral_passed and numerical_passed:
        classification = "behavioral-and-numerical-parity"
    elif behavioral_passed:
        classification = "behavioral-parity-with-numerical-drift"
    else:
        classification = "parity-failed"
    return {
        "classification": classification,
        "behavioral": {
            "passed": behavioral_passed,
            "limits": BEHAVIORAL_PARITY_LIMITS,
            "checks": behavioral_checks,
            "observed": {
                "baseline_mean_reward_abs_delta": baseline_reward_delta,
                "baseline_top_rank_mean_reward_abs_delta": baseline_top_rank_delta,
                "native_mean_reward_abs_delta": native_reward_delta,
                "native_top_rank_mean_reward_abs_delta": native_top_rank_delta,
            },
        },
        "numerical": {
            "passed": numerical_passed,
            "limits": NUMERICAL_PARITY_LIMITS,
            "checks": numerical_checks,
            "observed": {
                "evaluation_kl_abs_delta": evaluation_kl_delta,
                "adapter_delta_cosine_similarity": trace_adapter[
                    "delta_cosine_similarity"
                ],
                "adapter_delta_l2_relative_difference": trace_adapter[
                    "delta_l2_relative_difference"
                ],
                "adapter_delta_max_abs_difference": trace_adapter[
                    "delta_max_abs_difference"
                ],
            },
        },
    }


def _load_json(path: Path, where: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoolQParityContractError(f"could not load {where}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BoolQParityContractError(f"{where} root must be an object")
    return payload


def _sha256_string(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or value != value.lower()
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise BoolQParityContractError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _positive_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BoolQParityContractError(f"{where} must be a positive integer")
    return value


def _finite_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BoolQParityContractError(f"{where} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise BoolQParityContractError(f"{where} must be finite")
    return result


def load_materialization(path: Path) -> Mapping[str, Any]:
    manifest_path = path.expanduser().resolve()
    manifest = _load_json(manifest_path, "BoolQ materialization manifest")
    if manifest.get("schema_version") not in MATERIALIZATION_SCHEMA_VERSIONS:
        raise BoolQParityContractError("unsupported BoolQ materialization schema")
    try:
        boolq_materializer.validate_materialization_semantic_sha256(manifest)
        boolq_materializer.validate_materialization_selection_contract(manifest)
    except boolq_materializer.MaterializationError as exc:
        raise BoolQParityContractError(str(exc)) from exc
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("repo_id") != "google/boolq":
        raise BoolQParityContractError("materialization is not the pinned BoolQ dataset")
    revision = dataset.get("revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(char not in "0123456789abcdef" for char in revision)
    ):
        raise BoolQParityContractError("BoolQ revision must be a full Git commit")
    policy = dataset.get("selection_policy")
    expected_policy = {
        "dataset_format": "rendered-text-grpo",
        "max_seq_len": FIXED_SEQUENCE_LENGTH,
        "max_completion_tokens": FIXED_MAX_COMPLETION_TOKENS,
        "target_tokens": 1,
        "rendered_prompt_truncation": "forbidden",
        "response_channel": "final",
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != value for key, value in expected_policy.items()
    ):
        raise BoolQParityContractError("BoolQ selection policy differs from the parity contract")
    for section, manifest_key in (("train", "train_jsonl"), ("evaluation", "eval_jsonl")):
        record = dataset.get(section)
        if not isinstance(record, dict):
            raise BoolQParityContractError(f"dataset.{section} must be an object")
        jsonl_path = Path(str(manifest.get(manifest_key, ""))).expanduser().resolve()
        if not jsonl_path.is_file():
            raise BoolQParityContractError(f"missing materialized {section} JSONL")
        expected = _sha256_string(
            record.get("materialized_jsonl_sha256"),
            f"dataset.{section}.materialized_jsonl_sha256",
        )
        if sha256_file(jsonl_path) != expected:
            raise BoolQParityContractError(f"materialized {section} JSONL SHA-256 drifted")
    train_ids = manifest.get("train_source_ids")
    eval_ids = manifest.get("eval_source_ids")
    if not isinstance(train_ids, list) or not isinstance(eval_ids, list):
        raise BoolQParityContractError("materialization source identities are missing")
    if len(train_ids) < FIXED_TRAIN_GROUPS or len(eval_ids) != FIXED_EVAL_GROUPS:
        raise BoolQParityContractError("materialization row counts differ from the fixed campaign")
    if set(train_ids) & set(eval_ids):
        raise BoolQParityContractError("BoolQ train/evaluation source identities overlap")
    return manifest


def load_trace(
    path: Path,
    *,
    phase: str,
    expected_groups: int,
    group_size: int = FIXED_GROUP_SIZE,
) -> tuple[TraceGroup, ...]:
    trace_path = path.expanduser().resolve()
    groups: dict[int, list[TraceCompletion]] = {}
    try:
        lines = trace_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise BoolQParityContractError(f"could not load {phase} reward trace: {exc}") from exc
    for line_index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BoolQParityContractError(
                f"{phase} reward trace line {line_index + 1} is invalid JSON"
            ) from exc
        if not isinstance(row, dict) or row.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise BoolQParityContractError(f"{phase} reward trace schema drifted")
        if row.get("phase") != phase or row.get("call_index") != line_index:
            raise BoolQParityContractError(f"{phase} reward trace order drifted")
        prompt_index = row.get("prompt_index")
        if isinstance(prompt_index, bool) or not isinstance(prompt_index, int) or prompt_index < 0:
            raise BoolQParityContractError(f"{phase} reward trace prompt index is invalid")
        tokens = row.get("completion_tokens")
        if (
            not isinstance(tokens, list)
            or len(tokens) != FIXED_MAX_COMPLETION_TOKENS
            or isinstance(tokens[0], bool)
            or not isinstance(tokens[0], int)
            or tokens[0] < 0
        ):
            raise BoolQParityContractError(f"{phase} reward trace is not one-token GRPO")
        reward = _finite_float(row.get("aggregate_reward"), "aggregate_reward")
        if reward not in (0.0, 1.0):
            raise BoolQParityContractError("BoolQ trace reward must be binary")
        groups.setdefault(prompt_index, []).append(TraceCompletion(tokens[0], reward))
    if sorted(groups) != list(range(expected_groups)):
        raise BoolQParityContractError(f"{phase} reward trace prompt groups are incomplete")
    result = tuple(
        TraceGroup(index, tuple(groups[index])) for index in range(expected_groups)
    )
    if any(len(group.completions) != group_size for group in result):
        raise BoolQParityContractError(f"{phase} reward trace group size drifted")
    return result


def require_antfly_reference_contract(
    train_report: Mapping[str, Any], eval_report: Mapping[str, Any]
) -> None:
    """Admit the legacy scorer or the provenance-locked batched replacement."""

    train_reference_mode = train_report.get("reference_mode")
    if train_reference_mode not in {
        ANTFLY_LEGACY_REFERENCE_MODE,
        ANTFLY_BATCHED_REFERENCE_MODE,
    }:
        raise BoolQParityContractError("Antfly reference semantics drifted")
    if train_reference_mode != ANTFLY_BATCHED_REFERENCE_MODE:
        return
    if eval_report.get("reference_mode") != ANTFLY_BATCHED_REFERENCE_MODE:
        raise BoolQParityContractError(
            "Antfly batched reference semantics differ between train and evaluation"
        )
    if eval_report.get("execution_order") != ANTFLY_PHASED_EVALUATION_ORDER:
        raise BoolQParityContractError(
            "Antfly batched evaluation did not phase-separate policy and reference execution"
        )


def require_v4_kl_control(root: Path, train_report: Mapping[str, Any]) -> None:
    if train_report.get("schema_version") not in {
        "antfly_inference_finetune_grpo_report/v4",
        "antfly_inference_finetune_grpo_report/v5",
        "antfly_inference_finetune_grpo_report/v6",
        "antfly_inference_finetune_grpo_report/v7",
        "antfly_inference_finetune_grpo_report/v8",
    }:
        return
    mean_kl = train_report.get("mean_kl")
    telemetry = train_report.get("kl_control")
    if (
        isinstance(mean_kl, bool)
        or not isinstance(mean_kl, (int, float))
        or not math.isfinite(float(mean_kl))
        or float(mean_kl) < 0.0
        or not isinstance(telemetry, dict)
    ):
        raise BoolQParityContractError("Antfly GRPO v4 KL telemetry is missing")
    trace_path = root / "grpo_kl_control_trace.jsonl"
    reported_path = Path(str(telemetry.get("trace_path", ""))).resolve()
    if reported_path != trace_path or not trace_path.is_file():
        raise BoolQParityContractError("Antfly GRPO v4 KL trace escaped the acceptance root")
    if telemetry.get("trace_digest") != "sha256:" + sha256_file(trace_path):
        raise BoolQParityContractError("Antfly GRPO v4 KL trace digest drifted")
    try:
        rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BoolQParityContractError(f"could not load Antfly GRPO KL trace: {exc}") from exc
    if len(rows) != FIXED_TRAIN_GROUPS:
        raise BoolQParityContractError("Antfly GRPO v4 KL trace group count drifted")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise BoolQParityContractError("Antfly GRPO v4 KL trace row is invalid")
        observed = row.get("mean_kl")
        budget = row.get("train_max_kl")
        if (
            row.get("schema_version") != GRPO_KL_TRACE_SCHEMA_VERSION
            or row.get("group_index") != index
            or row.get("optimizer_steps_before") != index
            or row.get("status") != "admitted"
            or row.get("budget_policy") != "skip_group"
            or isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or float(observed) < 0.0
            or isinstance(budget, bool)
            or not isinstance(budget, (int, float))
            or not math.isfinite(float(budget))
            or float(observed) > float(budget)
        ):
            raise BoolQParityContractError("Antfly GRPO v4 KL admission trace drifted")


def require_native_rollout_sampler_compatibility(
    train_report: Mapping[str, Any],
) -> None:
    mode = train_report.get("sampling_mode")
    if isinstance(mode, str) and "ranked" in mode:
        return
    raise BoolQParityContractError(
        "MLX native_rollout still implements the retired deterministic ranked sampler; "
        "stochastic Antfly evidence is eligible only after the MLX lane implements "
        "seeded categorical temperature/top-p/top-k sampling and statistical gates"
    )


def load_acceptance(root: Path, manifest: Mapping[str, Any]) -> AcceptanceEvidence:
    evidence_root = root.expanduser().resolve()
    config = _load_json(evidence_root / "training_config.json", "Antfly training config")
    train_report = _load_json(evidence_root / "grpo_report.json", "Antfly GRPO report")
    eval_report = _load_json(
        evidence_root / "grpo_evaluation_report.json", "Antfly GRPO evaluation report"
    )
    if train_report.get("schema_version") not in GRPO_REPORT_SCHEMA_VERSIONS:
        raise BoolQParityContractError("Antfly GRPO report schema drifted")
    if (
        train_report.get("schema_version") == "antfly_inference_finetune_grpo_report/v8"
        and train_report.get("training_order") != GRPO_TRAINING_ORDER
    ):
        raise BoolQParityContractError("Antfly GRPO training-order contract drifted")
    if train_report.get("execution_mode") != "train" or train_report.get("dataset_format") != "rendered-text-grpo":
        raise BoolQParityContractError("Antfly evidence is not optimizer-backed rendered GRPO")
    expected_counts = {
        "groups": FIXED_TRAIN_GROUPS,
        "completions": FIXED_TRAIN_GROUPS * FIXED_GROUP_SIZE,
        "optimizer_steps": FIXED_TRAIN_GROUPS,
    }
    if any(train_report.get(key) != value for key, value in expected_counts.items()):
        raise BoolQParityContractError("Antfly training counts differ from the fixed campaign")
    if train_report.get("schema_version") in {
        "antfly_inference_finetune_grpo_report/v7",
        "antfly_inference_finetune_grpo_report/v8",
    }:
        if (
            train_report.get("optimizer_groups") != FIXED_TRAIN_GROUPS
            or train_report.get("zero_reward_std_groups") != 0
            or train_report.get("all_truncated_groups") != 0
            or train_report.get("kl_rejected_groups") != 0
            or float(train_report.get("frac_reward_zero_std", -1.0)) != 0.0
            or float(train_report.get("frac_kl_rejected", -1.0)) != 0.0
            or train_report.get("loss_type") != "bnpo"
            or train_report.get("scale_rewards") != "group"
            or float(train_report.get("epsilon_low", 0.0)) != FIXED_GRPO["clip_epsilon"]
            or float(train_report.get("epsilon_high", 0.0)) != FIXED_GRPO["clip_epsilon"]
            or train_report.get("max_completion_tokens") != FIXED_MAX_COMPLETION_TOKENS
            or train_report.get("mask_truncated_completions") is not False
            or train_report.get("num_iterations") != 1
        ):
            raise BoolQParityContractError("Antfly GRPO v7 objective semantics drifted")
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
            raise BoolQParityContractError("Antfly truncated-completion telemetry drifted")
    if train_report.get("policy_backend") != "metal":
        raise BoolQParityContractError("Antfly parity evidence must use Metal")
    if eval_report.get("schema_version") not in GRPO_EVAL_SCHEMA_VERSIONS:
        raise BoolQParityContractError("Antfly GRPO evaluation schema drifted")
    if eval_report.get("status") != "passed" or eval_report.get("groups") != FIXED_EVAL_GROUPS:
        raise BoolQParityContractError("Antfly held-out campaign did not pass")
    if eval_report.get("mask_truncated_completions") is not False:
        raise BoolQParityContractError("Antfly evaluation truncation policy drifted")
    if eval_report.get("schema_version") in {
        "antfly_inference_finetune_grpo_evaluation/v2",
        "antfly_inference_finetune_grpo_evaluation/v3",
        "antfly_inference_finetune_grpo_evaluation/v4",
    }:
        eval_mean_kl = eval_report.get("mean_kl")
        if (
            isinstance(eval_mean_kl, bool)
            or not isinstance(eval_mean_kl, (int, float))
            or not math.isfinite(float(eval_mean_kl))
            or float(eval_mean_kl) < 0.0
        ):
            raise BoolQParityContractError("Antfly GRPO v2 evaluation KL is invalid")
    require_antfly_reference_contract(train_report, eval_report)
    require_v4_kl_control(evidence_root, train_report)
    require_native_rollout_sampler_compatibility(train_report)

    recipe = config.get("recipe")
    if not isinstance(recipe, dict):
        raise BoolQParityContractError("Antfly training config has no normalized recipe")
    dataset = recipe.get("dataset")
    adapter = recipe.get("adapter")
    optimizer = recipe.get("optimizer")
    grpo = recipe.get("grpo")
    evaluation = recipe.get("eval")
    model = recipe.get("model")
    if not all(isinstance(item, dict) for item in (dataset, adapter, optimizer, grpo, evaluation, model)):
        raise BoolQParityContractError("Antfly normalized recipe is incomplete")
    if model.get("family") != "gemma4":
        raise BoolQParityContractError("Antfly model family is not gemma4")
    if (
        dataset.get("path") != manifest.get("train_jsonl")
        or evaluation.get("path") != manifest.get("eval_jsonl")
        or dataset.get("max_examples") != FIXED_TRAIN_GROUPS
        or evaluation.get("max_examples") != FIXED_EVAL_GROUPS
        or dataset.get("max_seq_len") != FIXED_SEQUENCE_LENGTH
    ):
        raise BoolQParityContractError("Antfly recipe is not bound to the pinned BoolQ split")
    if adapter.get("rank") != 16 or float(adapter.get("alpha", 0.0)) != 32.0 or adapter.get("target_preset") != FIXED_TARGET_PRESET:
        raise BoolQParityContractError("Antfly adapter contract drifted")
    if (
        abs(float(optimizer.get("learning_rate", 0.0)) - FIXED_LEARNING_RATE) > 1.0e-14
        or optimizer.get("epochs") != 1
        or optimizer.get("gradient_accumulation_steps") != 1
        or float(optimizer.get("max_grad_norm", 0.0)) != 1.0
    ):
        raise BoolQParityContractError("Antfly optimizer contract drifted")
    if (
        grpo.get("group_size") != FIXED_GROUP_SIZE
        or grpo.get("max_completion_tokens") != FIXED_MAX_COMPLETION_TOKENS
        or abs(float(grpo.get("clip_epsilon", 0.0)) - FIXED_GRPO["clip_epsilon"]) > 1.0e-6
        or abs(float(grpo.get("kl_coef", 0.0)) - FIXED_GRPO["kl_coef"]) > 1.0e-6
        or grpo.get("normalize_advantage") is not True
        or grpo.get("loss_type") not in (None, "bnpo")
        or grpo.get("scale_rewards") not in (None, "group")
        or grpo.get("epsilon_high") not in (None, FIXED_GRPO["clip_epsilon"])
        or grpo.get("mask_truncated_completions") not in (None, False)
    ):
        raise BoolQParityContractError("Antfly GRPO contract drifted")

    train_trace_path = evidence_root / "grpo_reward_trace.jsonl"
    eval_trace_path = evidence_root / "grpo_evaluation_reward_trace.jsonl"
    telemetry = train_report.get("reward_pipeline")
    eval_telemetry = eval_report.get("reward_pipeline")
    if not isinstance(telemetry, dict) or not isinstance(eval_telemetry, dict):
        raise BoolQParityContractError("Antfly reward telemetry is missing")
    for trace_path, trace_record in ((train_trace_path, telemetry), (eval_trace_path, eval_telemetry)):
        digest = trace_record.get("trace_digest")
        if digest != "sha256:" + sha256_file(trace_path):
            raise BoolQParityContractError("Antfly reward trace digest drifted")
    trained_adapter_dir = Path(str(train_report.get("trained_adapter_dir", ""))).resolve()
    if trained_adapter_dir.parent != evidence_root or not trained_adapter_dir.is_dir():
        raise BoolQParityContractError("Antfly trained adapter escaped the acceptance root")
    return AcceptanceEvidence(
        root=evidence_root,
        training_config=config,
        train_report=train_report,
        eval_report=eval_report,
        train_trace=load_trace(train_trace_path, phase="train", expected_groups=FIXED_TRAIN_GROUPS),
        eval_trace=load_trace(eval_trace_path, phase="evaluation", expected_groups=FIXED_EVAL_GROUPS),
        trained_adapter_dir=trained_adapter_dir,
    )


def normalized_advantages(rewards: Sequence[float], epsilon: float) -> list[float]:
    if not rewards:
        raise BoolQParityContractError("cannot normalize an empty reward group")
    mean = statistics.mean(rewards)
    variance = (
        sum((reward - mean) ** 2 for reward in rewards) / (len(rewards) - 1)
        if len(rewards) > 1
        else 0.0
    )
    denominator = math.sqrt(variance) + epsilon
    return [(reward - mean) / denominator for reward in rewards]


def exact_match_ci(decoded_text: str, target: str) -> float:
    return 1.0 if decoded_text.strip().lower() == target.strip().lower() else 0.0


def candidate_overlap(actual: Sequence[int], expected: Sequence[int]) -> Mapping[str, Any]:
    if not actual or len(actual) != len(expected):
        raise BoolQParityContractError("candidate groups must be non-empty and equal-length")
    actual_counts = Counter(actual)
    expected_counts = Counter(expected)
    overlap = sum((actual_counts & expected_counts).values())
    return {
        "overlap": overlap,
        "recall": overlap / len(expected),
        "exact_set": actual_counts == expected_counts,
        "exact_order": tuple(actual) == tuple(expected),
        "top1_match": actual[0] == expected[0],
    }


def summarize_overlaps(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not rows:
        raise BoolQParityContractError("cannot summarize empty candidate-overlap evidence")
    return {
        "groups": len(rows),
        "mean_overlap": statistics.mean(float(row["overlap"]) for row in rows),
        "mean_recall": statistics.mean(float(row["recall"]) for row in rows),
        "exact_set_rate": statistics.mean(1.0 if row["exact_set"] else 0.0 for row in rows),
        "exact_order_rate": statistics.mean(1.0 if row["exact_order"] else 0.0 for row in rows),
        "top1_match_rate": statistics.mean(1.0 if row["top1_match"] else 0.0 for row in rows),
    }


def write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temp_path = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temp_path.open("x", encoding="utf-8") as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_path, destination)
    except FileExistsError as exc:
        raise BoolQParityContractError(f"parity output already exists: {destination}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def _load_rows(
    path: Path,
    *,
    expected_count: int,
    expected_ids: Sequence[str],
    expected_indices: Sequence[int],
    tokenizer: Any,
) -> tuple[BoolQRow, ...]:
    rows: list[BoolQRow] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise BoolQParityContractError(f"could not load materialized BoolQ JSONL: {exc}") from exc
    for line_index, line in enumerate(lines):
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BoolQParityContractError(f"BoolQ JSONL line {line_index + 1} is invalid") from exc
        if not isinstance(payload, dict) or set(payload) != {"prompt", "target", "metadata"}:
            raise BoolQParityContractError("BoolQ JSONL row schema drifted")
        prompt, target, metadata = payload["prompt"], payload["target"], payload["metadata"]
        if not isinstance(prompt, str) or not prompt or target not in ("yes", "no") or not isinstance(metadata, dict):
            raise BoolQParityContractError("BoolQ JSONL row is malformed")
        token_ids = tuple(int(value) for value in tokenizer.encode(prompt, add_special_tokens=False).ids)
        target_ids = tuple(int(value) for value in tokenizer.encode(target, add_special_tokens=False).ids)
        if (
            not token_ids
            or len(token_ids) + FIXED_MAX_COMPLETION_TOKENS > FIXED_SEQUENCE_LENGTH
            or len(target_ids) != 1
            or metadata.get("prompt_tokens") != len(token_ids)
            or metadata.get("target_tokens") != 1
        ):
            raise BoolQParityContractError("BoolQ tokenizer contract drifted")
        rows.append(
            BoolQRow(
                prompt=prompt,
                target=target,
                prompt_token_ids=token_ids,
                source_split=str(metadata.get("source_split", "")),
                source_row_index=int(metadata.get("source_row_index", -1)),
                source_id=str(metadata.get("source_id", "")),
            )
        )
    if len(rows) < expected_count:
        raise BoolQParityContractError("BoolQ JSONL has too few admitted rows")
    selected = tuple(rows[:expected_count])
    if [row.source_id for row in selected] != list(expected_ids[:expected_count]):
        raise BoolQParityContractError("BoolQ source identity order drifted")
    if [row.source_row_index for row in selected] != list(expected_indices[:expected_count]):
        raise BoolQParityContractError("BoolQ source row order drifted")
    return selected


def _decode_reward(tokenizer: Any, token_id: int, target: str) -> tuple[str, float]:
    decoded = tokenizer.decode([token_id], skip_special_tokens=True)
    return decoded, exact_match_ci(decoded, target)


def _validate_trace_rewards(tokenizer: Any, rows: Sequence[BoolQRow], trace: Sequence[TraceGroup]) -> None:
    if len(rows) != len(trace):
        raise BoolQParityContractError("BoolQ rows and reward trace groups differ")
    for row, group in zip(rows, trace):
        for completion in group.completions:
            _decoded, reward = _decode_reward(tokenizer, completion.token_id, row.target)
            if reward != completion.reward:
                raise BoolQParityContractError(
                    f"Antfly reward trace cannot be reproduced for prompt {group.prompt_index} token {completion.token_id}"
                )


def _adapter_delta_comparison(
    *,
    model: Any,
    initial_trainables: Mapping[str, Any],
    antfly_trained: Any,
    target_names: Sequence[str],
    mx: Any,
    tree_flatten: Any,
) -> Mapping[str, Any]:
    final_trainables = dict(tree_flatten(model.trainable_parameters()))
    loaded = mx.load(str(antfly_trained.checkpoint))
    mlx_targets = {
        locked.canonicalize_module_name(name): name for name in target_names
    }
    antfly_final: dict[str, Any] = {}
    for (module, role), descriptor in antfly_trained.tensors.items():
        suffix = "lora_a" if role == "lora_A" else "lora_b"
        antfly_final[f"{mlx_targets[module]}.{suffix}"] = loaded[descriptor.source_name].T
    names = sorted(initial_trainables)
    if set(final_trainables) != set(names) or set(antfly_final) != set(names):
        raise BoolQParityContractError("adapter comparison inventory drifted")
    mlx_squares = []
    antfly_squares = []
    dots = []
    max_differences = []
    for name in names:
        initial = initial_trainables[name]
        mlx_delta = (final_trainables[name] - initial).astype(mx.float32)
        antfly_delta = (antfly_final[name] - initial).astype(mx.float32)
        mlx_squares.append((mlx_delta * mlx_delta).sum())
        antfly_squares.append((antfly_delta * antfly_delta).sum())
        dots.append((mlx_delta * antfly_delta).sum())
        max_differences.append(mx.abs(mlx_delta - antfly_delta).max())
    mx.eval(*mlx_squares, *antfly_squares, *dots, *max_differences)
    mlx_norm = math.sqrt(sum(float(value.item()) for value in mlx_squares))
    antfly_norm = math.sqrt(sum(float(value.item()) for value in antfly_squares))
    dot = sum(float(value.item()) for value in dots)
    cosine = dot / (mlx_norm * antfly_norm) if mlx_norm and antfly_norm else 0.0
    return {
        "mlx_delta_l2": mlx_norm,
        "antfly_delta_l2": antfly_norm,
        "delta_l2_relative_difference": abs(mlx_norm - antfly_norm) / antfly_norm if antfly_norm else None,
        "delta_cosine_similarity": cosine,
        "delta_max_abs_difference": max(float(value.item()) for value in max_differences),
        "tensor_count": len(names),
    }


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    manifest = load_materialization(args.dataset_manifest)
    acceptance = load_acceptance(args.antfly_run_root, manifest)
    lock = locked.load_lock(args.lock)
    mlx_contract = lock["mlx_reference"]
    locked.force_offline_environment()
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != mlx_contract["python"]:
        raise BoolQParityContractError(
            f"MLX parity requires Python {mlx_contract['python']}, found {actual_python}"
        )
    revisions = mlx_contract["source_revisions"]
    if args.mlx_runtime_mode == "source-checkout":
        mlx_revision: str | None = microbenchmark.require_source_revision(
            args.mlx_source_root, revisions["mlx"], "MLX"
        )
        mlx_runtime_attestation: Mapping[str, Any] = {
            "mode": "clean-source-checkout",
            "runtime_root": str(args.mlx_source_root.expanduser().resolve()),
            "source_revision": mlx_revision,
        }
    else:
        if args.mlx_wheel is None or args.mlx_metal_wheel is None:
            raise BoolQParityContractError(
                "wheel-archive mode requires --mlx-wheel and --mlx-metal-wheel"
            )
        mlx_revision = None
        mlx_runtime_attestation = attest_wheel_runtime(
            runtime_root=args.mlx_source_root,
            wheel_path=args.mlx_wheel,
            metal_wheel_path=args.mlx_metal_wheel,
            expected_version=mlx_contract["packages"]["mlx"],
        )
        mlx_runtime_attestation = {
            **mlx_runtime_attestation,
            "locked_source_revision": revisions["mlx"],
            "source_revision_verified": False,
        }
    mlx_lm_revision = microbenchmark.require_source_revision(
        args.mlx_lm_source_root, revisions["mlx-lm"], "MLX-LM"
    )
    if platform.system() != mlx_contract["required_platform"] or platform.machine() != mlx_contract["required_machine"]:
        raise BoolQParityContractError("MLX parity must run on the locked Apple platform")

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_unflatten
    from tokenizers import tokenizers as tokenizers_native

    Tokenizer = tokenizers_native.Tokenizer

    mlx_root = args.mlx_source_root.expanduser().resolve()
    core_path = Path(mx.__file__ or "").resolve()
    if not microbenchmark._path_is_within(core_path, mlx_root):
        raise BoolQParityContractError(f"imported MLX is outside the attested checkout: {core_path}")
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
        raise BoolQParityContractError("tokenizers version differs from the materializer")
    mlx_lm_root = args.mlx_lm_source_root.expanduser().resolve()
    for label, source_path in (
        ("MLX-LM Gemma4", Path(mlx_gemma4.__file__ or "").resolve()),
        ("MLX-LM LoRA", Path(sys.modules[LoRALinear.__module__].__file__ or "").resolve()),
    ):
        if not microbenchmark._path_is_within(source_path, mlx_lm_root):
            raise BoolQParityContractError(f"imported {label} is outside the attested checkout")

    model_dir = args.model_dir.expanduser().resolve()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    if Path(str(manifest.get("model_dir", ""))).resolve() != model_dir:
        raise BoolQParityContractError("materialization model directory differs from --model-dir")
    tokenizer_path = model_dir / "tokenizer.json"
    if sha256_file(tokenizer_path) != manifest.get("tokenizer_files", {}).get("tokenizer.json"):
        raise BoolQParityContractError("runtime tokenizer differs from the materialization")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    train_rows = _load_rows(
        Path(str(manifest["train_jsonl"])),
        expected_count=FIXED_TRAIN_GROUPS,
        expected_ids=manifest["train_source_ids"],
        expected_indices=manifest["train_source_row_indices"],
        tokenizer=tokenizer,
    )
    eval_rows = _load_rows(
        Path(str(manifest["eval_jsonl"])),
        expected_count=FIXED_EVAL_GROUPS,
        expected_ids=manifest["eval_source_ids"],
        expected_indices=manifest["eval_source_row_indices"],
        tokenizer=tokenizer,
    )
    _validate_trace_rewards(tokenizer, train_rows, acceptance.train_trace)
    _validate_trace_rewards(tokenizer, eval_rows, acceptance.eval_trace)

    adapter_manifest = _load_json(adapter_dir / "antfly_finetune_manifest.json", "seed adapter manifest")
    binding_fields = ("base_model_sha256", "tokenizer_sha256", "chat_template_sha256")
    prepared_summary = {key: adapter_manifest.get(key) for key in binding_fields}
    base_model_provenance = locked.zig_model_provenance(model_dir)
    if prepared_summary != base_model_provenance:
        raise BoolQParityContractError("seed adapter does not match the parity model")
    seed_adapter = locked.inspect_initial_adapter(
        adapter_dir, lock, FIXED_MODEL_KEY, FIXED_TARGET_PRESET, prepared_summary
    )
    antfly_trained = locked.inspect_initial_adapter(
        acceptance.trained_adapter_dir,
        lock,
        FIXED_MODEL_KEY,
        FIXED_TARGET_PRESET,
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
            load_config_fn=lambda path: json.loads((path / "config.json").read_text(encoding="utf-8")),
            get_model_classes_fn=lambda **_kwargs: (mlx_gemma4.Model, mlx_gemma4.ModelArgs),
        )
        mx.eval(model.parameters())
        mx.synchronize()
        model.freeze()
        base_inventory = locked.require_bf16_base_model(model, mx)
        targets = locked.target_module_names(model, lock, FIXED_MODEL_KEY, FIXED_TARGET_PRESET)
        target_set = set(targets)
        module_updates = []
        for name, module in model.named_modules():
            if name not in target_set:
                continue
            if not isinstance(module, nn.Linear):
                raise BoolQParityContractError(f"non-linear LoRA target: {name}")
            module_updates.append(
                (
                    name,
                    LoRALinear.from_base(module, r=16, scale=2.0, dropout=0.0),
                )
            )
        if {name for name, _module in module_updates} != target_set:
            raise BoolQParityContractError("incomplete LoRA target conversion")
        model.update_modules(tree_unflatten(module_updates))
        trainable_inventory = locked.require_exact_trainables(model, targets, mx)
        locked.load_exact_initial_adapter(model, targets, seed_adapter, mx)
        model.train()
        mx.eval(model.state)
        mx.synchronize()
        initial_trainables = {
            name: mx.array(value) for name, value in tree_flatten(model.trainable_parameters())
        }
        mx.eval(*initial_trainables.values())
        load_seconds = time.perf_counter() - load_started

        def reset_to_initial() -> None:
            model.update(tree_unflatten(list(initial_trainables.items())), strict=True)
            mx.eval(model.trainable_parameters())
            mx.synchronize()

        def prompt_tensor(row: BoolQRow) -> tuple[Any, Any]:
            padded = list(row.prompt_token_ids) + [0] * (
                FIXED_SEQUENCE_LENGTH - len(row.prompt_token_ids)
            )
            return (
                mx.array([padded], dtype=mx.int32),
                mx.array([len(row.prompt_token_ids) - 1], dtype=mx.int32),
            )

        def selected_logps(current_model: Any, tokens: Any, row_index: Any, selected: Any) -> Any:
            logits = current_model(tokens).astype(mx.float32)
            predictor = mx.take(logits, row_index, axis=1)[:, 0, :]
            logprobs = predictor - mx.logsumexp(predictor, axis=-1, keepdims=True)
            return mx.take_along_axis(logprobs, selected[None, :], axis=-1)[0]

        def ranked_group(current_model: Any, tokens: Any, row_index: Any) -> tuple[Any, Any]:
            logits = current_model(tokens).astype(mx.float32)
            predictor = mx.take(logits, row_index, axis=1)[0, 0, :]
            candidate_ids = mx.argpartition(-predictor, kth=FIXED_GROUP_SIZE - 1)[:FIXED_GROUP_SIZE]
            order = mx.argsort(-predictor[candidate_ids])
            selected = candidate_ids[order]
            logprobs = predictor - mx.logsumexp(predictor)
            selected_lp = logprobs[selected]
            return selected, selected_lp

        def grpo_loss(
            current_model: Any,
            tokens: Any,
            row_index: Any,
            selected: Any,
            old_logps: Any,
            reference: Any,
            advantages: Any,
        ) -> tuple[Any, Any, Any, Any, Any]:
            new_logps = selected_logps(current_model, tokens, row_index, selected)
            ratio = mx.exp(new_logps - old_logps)
            pg_unclipped = ratio * advantages
            pg_clipped = mx.clip(
                ratio,
                1.0 - FIXED_GRPO["clip_epsilon"],
                1.0 + FIXED_GRPO["clip_epsilon"],
            ) * advantages
            pg_tokens = -mx.minimum(pg_unclipped, pg_clipped)
            diff = reference - new_logps
            kl_tokens = FIXED_GRPO["kl_coef"] * (mx.exp(diff) - diff - 1.0)
            pg_loss = pg_tokens.mean()
            kl_loss = kl_tokens.mean()
            loss = pg_loss + kl_loss
            clip_fraction = (pg_clipped < pg_unclipped).astype(mx.float32).mean()
            return loss, pg_loss, kl_loss, clip_fraction, new_logps

        loss_and_grad = nn.value_and_grad(model, grpo_loss)

        def make_optimizer() -> Any:
            return optim.AdamW(
                learning_rate=FIXED_LEARNING_RATE,
                betas=(FIXED_OPTIMIZER["beta1"], FIXED_OPTIMIZER["beta2"]),
                eps=FIXED_OPTIMIZER["epsilon"],
                weight_decay=FIXED_OPTIMIZER["weight_decay"],
                bias_correction=True,
            )

        def full_reference_logprobs(rows: Sequence[BoolQRow]) -> list[array.array[float]]:
            result: list[array.array[float]] = []
            for row in rows:
                tokens, row_index = prompt_tensor(row)
                logits = model(tokens).astype(mx.float32)
                predictor = mx.take(logits, row_index, axis=1)[0, 0, :]
                values = predictor - mx.logsumexp(predictor)
                mx.eval(values)
                mx.synchronize()
                result.append(array.array("f", values.tolist()))
            return result

        def evaluate_policy(
            rows: Sequence[BoolQRow],
            antfly_trace: Sequence[TraceGroup],
        ) -> tuple[list[dict[str, Any]], list[list[float]]]:
            records: list[dict[str, Any]] = []
            selected_logprobs: list[list[float]] = []
            for row, expected in zip(rows, antfly_trace):
                tokens, row_index = prompt_tensor(row)
                selected, logps = ranked_group(model, tokens, row_index)
                mx.eval(selected, logps)
                mx.synchronize()
                token_ids = [int(value) for value in selected.tolist()]
                policy_logps = [float(value) for value in logps.tolist()]
                decoded_rewards = [_decode_reward(tokenizer, token_id, row.target) for token_id in token_ids]
                records.append(
                    {
                        "source_id": row.source_id,
                        "target": row.target,
                        "completion_token_ids": token_ids,
                        "decoded_completions": [item[0] for item in decoded_rewards],
                        "rewards": [item[1] for item in decoded_rewards],
                        "candidate_overlap": candidate_overlap(token_ids, expected.token_ids),
                    }
                )
                selected_logprobs.append(policy_logps)
            return records, selected_logprobs

        def add_reference_metrics(
            records: list[dict[str, Any]], reference_logps: Sequence[Sequence[float]]
        ) -> Mapping[str, Any]:
            all_rewards: list[float] = []
            group_losses: list[float] = []
            group_pg_losses: list[float] = []
            group_kl_losses: list[float] = []
            top_rewards: list[float] = []
            positive_groups = 0
            for record, reference in zip(records, reference_logps):
                policy = record["policy_logps"]
                rewards = record["rewards"]
                advantages = normalized_advantages(rewards, FIXED_GRPO["advantage_epsilon"])
                pg_tokens = [-advantage for advantage in advantages]
                kl_tokens = []
                for policy_lp, reference_lp in zip(policy, reference):
                    diff = reference_lp - policy_lp
                    kl_tokens.append(
                        FIXED_GRPO["kl_coef"] * (math.exp(diff) - diff - 1.0)
                    )
                pg_loss = statistics.mean(pg_tokens)
                kl_loss = statistics.mean(kl_tokens)
                group_pg_losses.append(pg_loss)
                group_kl_losses.append(kl_loss)
                group_losses.append(pg_loss + kl_loss)
                all_rewards.extend(rewards)
                top_rewards.append(rewards[0])
                positive_groups += int(any(reward > 0.0 for reward in rewards))
                record["reference_logps"] = list(reference)
                record["pg_loss"] = pg_loss
                record["kl_loss"] = kl_loss
            mean_reward = statistics.mean(all_rewards)
            reward_variance = statistics.mean(
                (reward - mean_reward) ** 2 for reward in all_rewards
            )
            overlaps = [record["candidate_overlap"] for record in records]
            return {
                "groups": len(records),
                "completions": len(all_rewards),
                "mean_reward": mean_reward,
                "top_rank_mean_reward": statistics.mean(top_rewards),
                "positive_reward_group_rate": positive_groups / len(records),
                "reward_stddev": math.sqrt(reward_variance),
                "loss": statistics.mean(group_losses),
                "pg_loss": statistics.mean(group_pg_losses),
                "kl_loss": statistics.mean(group_kl_losses),
                "candidate_overlap_with_antfly": summarize_overlaps(overlaps),
                "rows": records,
            }

        # Cache the immutable frozen-reference rows used by trace replay, then
        # establish a native zero-update held-out baseline.
        reference_started = time.perf_counter()
        train_base_logprobs = full_reference_logprobs(train_rows)
        eval_base_logprobs = full_reference_logprobs(eval_rows)
        train_reference = [
            [float(base[token_id]) for token_id in group.token_ids]
            for base, group in zip(train_base_logprobs, acceptance.train_trace)
        ]
        baseline_records, baseline_policy_logps = evaluate_policy(eval_rows, acceptance.eval_trace)
        for record, policy in zip(baseline_records, baseline_policy_logps):
            record["policy_logps"] = policy
        baseline = add_reference_metrics(baseline_records, baseline_policy_logps)
        reference_precompute_seconds = time.perf_counter() - reference_started

        def train_lane(mode: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
            optimizer = make_optimizer()
            state = [model.state, optimizer.state, mx.random.state]

            def step(
                tokens: Any,
                row_index: Any,
                selected: Any,
                old_logps: Any,
                reference: Any,
                advantages: Any,
            ) -> tuple[Any, Any, Any, Any, Any, Any]:
                metrics, gradients = loss_and_grad(
                    model,
                    tokens,
                    row_index,
                    selected,
                    old_logps,
                    reference,
                    advantages,
                )
                gradients, grad_norm = optim.clip_grad_norm(
                    gradients, FIXED_OPTIMIZER["max_grad_norm"]
                )
                optimizer.update(model, gradients)
                return (*metrics, grad_norm)

            compiled_step = mx.compile(step, inputs=state, outputs=state)
            update_rows: list[dict[str, Any]] = []
            lane_started = time.perf_counter()
            for update_index, (row, expected, reference_values) in enumerate(
                zip(train_rows, acceptance.train_trace, train_reference)
            ):
                started = time.perf_counter()
                tokens, row_index = prompt_tensor(row)
                native_ids, native_logps = ranked_group(model, tokens, row_index)
                mx.eval(native_ids, native_logps)
                mx.synchronize()
                native_token_ids = [int(value) for value in native_ids.tolist()]
                if mode == "trace_replay":
                    selected_ids = list(expected.token_ids)
                    rewards = list(expected.rewards)
                    selected = mx.array(selected_ids, dtype=mx.int32)
                    old_logps = selected_logps(model, tokens, row_index, selected)
                    reference = mx.array(reference_values, dtype=mx.float32)
                elif mode == "native_rollout":
                    selected_ids = native_token_ids
                    selected = native_ids
                    old_logps = native_logps
                    rewards = [
                        _decode_reward(tokenizer, token_id, row.target)[1]
                        for token_id in selected_ids
                    ]
                    reference = mx.array(
                        [
                            float(train_base_logprobs[update_index][token_id])
                            for token_id in selected_ids
                        ],
                        dtype=mx.float32,
                    )
                else:
                    raise AssertionError(mode)
                advantages = normalized_advantages(
                    rewards, FIXED_GRPO["advantage_epsilon"]
                )
                advantage_array = mx.array(advantages, dtype=mx.float32)
                outputs = compiled_step(
                    tokens,
                    row_index,
                    selected,
                    old_logps,
                    reference,
                    advantage_array,
                )
                loss, pg_loss, kl_loss, clip_fraction, rescored, grad_norm = outputs
                mx.eval(*outputs, model.state, optimizer.state)
                mx.synchronize()
                sampling_rescore_error = float(mx.max(mx.abs(rescored - old_logps)).item())
                values = [
                    float(loss.item()),
                    float(pg_loss.item()),
                    float(kl_loss.item()),
                    float(clip_fraction.item()),
                    float(grad_norm.item()),
                ]
                if not all(math.isfinite(value) for value in values):
                    raise BoolQParityContractError("MLX GRPO produced a non-finite metric")
                if update_index == 0 and sampling_rescore_error > 1.0e-4:
                    raise BoolQParityContractError("MLX zero-update sampling/rescore parity failed")
                update_rows.append(
                    {
                        "update_index": update_index,
                        "source_id": row.source_id,
                        "target": row.target,
                        "completion_token_ids": selected_ids,
                        "rewards": rewards,
                        "native_candidate_ids": native_token_ids,
                        "candidate_overlap_with_antfly": candidate_overlap(
                            native_token_ids, expected.token_ids
                        ),
                        "loss": values[0],
                        "pg_loss": values[1],
                        "kl_loss": values[2],
                        "clip_fraction": values[3],
                        "preclip_gradient_l2": values[4],
                        "sampling_rescore_max_abs_error": sampling_rescore_error,
                        "seconds": time.perf_counter() - started,
                    }
                )
            lane_seconds = time.perf_counter() - lane_started
            adapter_comparison = _adapter_delta_comparison(
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
                    "optimizer_steps": len(update_rows),
                    "seconds": lane_seconds,
                    "median_update_seconds": statistics.median(
                        row["seconds"] for row in update_rows
                    ),
                    "mean_update_seconds": statistics.mean(
                        row["seconds"] for row in update_rows
                    ),
                    "mean_reward": statistics.mean(
                        reward for row in update_rows for reward in row["rewards"]
                    ),
                    "mean_loss": statistics.mean(row["loss"] for row in update_rows),
                    "mean_kl_loss": statistics.mean(
                        row["kl_loss"] for row in update_rows
                    ),
                    "candidate_overlap_with_antfly": summarize_overlaps(
                        [row["candidate_overlap_with_antfly"] for row in update_rows]
                    ),
                    "updates": update_rows,
                },
                adapter_comparison,
            )

        # Trace-replay lane: exact Antfly candidates and frozen reference rows.
        reset_to_initial()
        trace_train, trace_adapter = train_lane("trace_replay")
        trace_eval_records, trace_eval_policy = evaluate_policy(eval_rows, acceptance.eval_trace)
        trace_eval_reference = [
            [float(base[token_id]) for token_id in record["completion_token_ids"]]
            for base, record in zip(eval_base_logprobs, trace_eval_records)
        ]
        for record, policy in zip(trace_eval_records, trace_eval_policy):
            record["policy_logps"] = policy
        trace_evaluation = add_reference_metrics(trace_eval_records, trace_eval_reference)

        # Native lane: ranked candidates come from the evolving MLX policy;
        # their immutable reference values are gathered from the host-backed
        # zero-LoRA distributions computed before either lane mutated adapters.
        reset_to_initial()
        native_train, native_adapter = train_lane("native_rollout")
        native_eval_records, native_eval_policy = evaluate_policy(eval_rows, acceptance.eval_trace)
        native_eval_reference = [
            [float(base[token_id]) for token_id in record["completion_token_ids"]]
            for base, record in zip(eval_base_logprobs, native_eval_records)
        ]
        for record, policy in zip(native_eval_records, native_eval_policy):
            record["policy_logps"] = policy
        native_evaluation = add_reference_metrics(native_eval_records, native_eval_reference)

        memory = sampler.stop()
        sampler_active = False
        campaign_seconds = time.perf_counter() - campaign_started
    finally:
        if sampler_active:
            sampler.stop()

    antfly_eval = {
        key: acceptance.eval_report[key]
        for key in (
            "mean_reward",
            "top_rank_mean_reward",
            "positive_reward_group_rate",
            "reward_stddev",
            "loss",
            "pg_loss",
            "kl_loss",
            "clip_fraction",
        )
    }
    native_deltas = {
        key: native_evaluation[key] - antfly_eval[key]
        for key in (
            "mean_reward",
            "top_rank_mean_reward",
            "positive_reward_group_rate",
            "kl_loss",
        )
    }
    minimums = acceptance.eval_report.get("minimums")
    if not isinstance(minimums, dict):
        raise BoolQParityContractError("Antfly evaluation minimums are missing")
    native_passed = (
        native_evaluation["mean_reward"] >= minimums["mean_reward"]
        and native_evaluation["top_rank_mean_reward"] >= minimums["top_rank_mean_reward"]
        and native_evaluation["positive_reward_group_rate"] >= minimums["positive_reward_group_rate"]
        and native_evaluation["kl_loss"] <= minimums["max_kl_loss"]
    )
    parity_assessment = assess_parity(
        antfly_evaluation=antfly_eval,
        baseline_evaluation=baseline,
        native_evaluation=native_evaluation,
        trace_training=trace_train,
        trace_adapter=trace_adapter,
        native_quality_passed=native_passed,
    )
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "status": (
            "passed"
            if parity_assessment["behavioral"]["passed"]
            else "failed-parity-gate"
        ),
        "scope": "real-pinned-boolq-e2b-eight-update-grpo",
        "parity_assessment": parity_assessment,
        "dataset": {
            "repo_id": manifest["dataset"]["repo_id"],
            "revision": manifest["dataset"]["revision"],
            "manifest_path": str(args.dataset_manifest.expanduser().resolve()),
            "manifest_sha256": sha256_file(args.dataset_manifest.expanduser().resolve()),
            "train_jsonl_sha256": manifest["dataset"]["train"]["materialized_jsonl_sha256"],
            "eval_jsonl_sha256": manifest["dataset"]["evaluation"]["materialized_jsonl_sha256"],
            "train_groups": FIXED_TRAIN_GROUPS,
            "eval_groups": FIXED_EVAL_GROUPS,
        },
        "contract": {
            "model_key": FIXED_MODEL_KEY,
            "target_preset": FIXED_TARGET_PRESET,
            "sequence_length": FIXED_SEQUENCE_LENGTH,
            "group_size": FIXED_GROUP_SIZE,
            "max_completion_tokens": FIXED_MAX_COMPLETION_TOKENS,
            "learning_rate": FIXED_LEARNING_RATE,
            "optimizer": FIXED_OPTIMIZER,
            "grpo": FIXED_GRPO,
            "reference_mode": "frozen-zero-lora",
            "reward_mode": "exact-match-ci",
            "update_order": "first-eight-pinned-source-order",
        },
        "antfly": {
            "run_root": str(acceptance.root),
            "grpo_report_sha256": sha256_file(acceptance.root / "grpo_report.json"),
            "evaluation_report_sha256": sha256_file(
                acceptance.root / "grpo_evaluation_report.json"
            ),
            "train_reward_trace_sha256": sha256_file(
                acceptance.root / "grpo_reward_trace.jsonl"
            ),
            "evaluation_reward_trace_sha256": sha256_file(
                acceptance.root / "grpo_evaluation_reward_trace.jsonl"
            ),
            "trained_adapter_checkpoint_sha256": sha256_file(
                acceptance.trained_adapter_dir / "adapter_model.safetensors"
            ),
            "training": {
                key: acceptance.train_report[key]
                for key in ("groups", "completions", "loss", "pg_loss", "kl_loss", "mean_reward")
            },
            "evaluation": antfly_eval,
        },
        "mlx": {
            "baseline_evaluation": baseline,
            "trace_replay": {
                "training": trace_train,
                "evaluation": trace_evaluation,
                "adapter_delta_comparison_with_antfly": trace_adapter,
            },
            "native_rollout": {
                "training": native_train,
                "evaluation": native_evaluation,
                "adapter_delta_comparison_with_antfly": native_adapter,
                "evaluation_delta_from_antfly": native_deltas,
                "passed_antfly_quality_minimums": native_passed,
            },
            "load_seconds": load_seconds,
            "reference_precompute_seconds": reference_precompute_seconds,
            "campaign_seconds": campaign_seconds,
            "peak_phys_footprint_bytes": memory.peak_phys_footprint_bytes,
            "mlx_allocator_peak_bytes": int(mx.get_peak_memory()),
            "base_inventory_sha256": base_inventory["inventory_sha256"],
            "trainable_inventory_sha256": trainable_inventory["inventory_sha256"],
            "seed_adapter_semantic_sha256": seed_adapter.semantic_sha256,
            "mlx_revision": mlx_revision,
            "mlx_runtime_attestation": mlx_runtime_attestation,
            "mlx_lm_revision": mlx_lm_revision,
            "package_versions": actual_versions,
            "unused_locked_packages": ["numpy"],
            "tokenizers_version": tokenizers_version,
            "python_version": actual_python,
            "mlx_core_path": str(core_path),
        },
        "base_model_provenance": base_model_provenance,
        "runner_sha256": "sha256:" + hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--adapter-dir", type=Path, required=True)
    result.add_argument("--dataset-manifest", type=Path, required=True)
    result.add_argument("--antfly-run-root", type=Path, required=True)
    result.add_argument("--mlx-source-root", type=Path, required=True)
    result.add_argument(
        "--mlx-runtime-mode",
        choices=("source-checkout", "wheel-archive"),
        default="source-checkout",
    )
    result.add_argument("--mlx-wheel", type=Path)
    result.add_argument("--mlx-metal-wheel", type=Path)
    result.add_argument("--mlx-lm-source-root", type=Path, required=True)
    result.add_argument("--lock", type=Path, default=locked.LOCK_PATH)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        payload = run(args)
        write_json_exclusive(args.output, payload)
    except (BoolQParityContractError, microbenchmark.GrpoBenchmarkContractError, locked.ContractError) as exc:
        print(f"Gemma4 BoolQ GRPO MLX parity contract error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": payload["status"],
                "scope": payload["scope"],
                "classification": payload["parity_assessment"]["classification"],
                "output": str(args.output.expanduser().resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["status"] == "passed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
