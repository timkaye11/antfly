#!/usr/bin/env python3
"""Run matched multi-token Gemma4 GRPO campaigns against MLX.

This runner consumes one completed Antfly Metal campaign and the pinned BoolQ
materialization that produced it. It executes two MLX lanes from the identical
seed adapter:

* ``trace_replay`` trains on Antfly's exact completion sequences and rewards;
* ``native_rollout`` supports the historical ranked rollout and an explicit
  categorical diagnostic mode. The latter matches seeded train/evaluation
  sampling and skipped updates, but cannot publish a parity classification.

Both lanes freeze the initial adapter as their reference, one optimizer update per
admitted completion group, token-normalized GRPO loss, the same hard raw-K3 KL budget,
and the same proportional next-group KL controller as Antfly. The result is a
bounded campaign artifact, not a claim of broad or long-horizon quality parity.
"""

from __future__ import annotations

from gemma4_files import sha256_file

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import materialize_gemma4_grpo_boolq as boolq_materializer


SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent
RESULT_SCHEMA_VERSION = "antfly_gemma4_grpo_boolq_mlx_multitoken/v1"
MATERIALIZATION_SCHEMA_VERSION = boolq_materializer.SCHEMA_VERSION
MATERIALIZATION_SCHEMA_VERSIONS = boolq_materializer.SCHEMA_VERSIONS
REWARD_TRACE_SCHEMA_VERSION = "antfly_inference_grpo_reward_trace/v1"
KL_TRACE_SCHEMA_VERSION = "antfly_inference_grpo_kl_control_trace/v5"
KL_TRACE_SCHEMA_VERSIONS = frozenset(
    {
        "antfly_inference_grpo_kl_control_trace/v2",
        "antfly_inference_grpo_kl_control_trace/v3",
        "antfly_inference_grpo_kl_control_trace/v4",
        KL_TRACE_SCHEMA_VERSION,
    }
)
GRPO_TRAINING_ORDER = {
    "algorithm": "seeded-fisher-yates-per-epoch/v1",
    "stream_derivation": "run-seed-order-domain-epoch-dataset-size/v1",
    "prompt_index_semantics": "original-dataset-index",
}
GRPO_REPORT_SCHEMA_VERSIONS = frozenset(
    {
        "antfly_inference_finetune_grpo_report/v4",
        "antfly_inference_finetune_grpo_report/v5",
        "antfly_inference_finetune_grpo_report/v6",
        "antfly_inference_finetune_grpo_report/v7",
        "antfly_inference_finetune_grpo_report/v8",
        "antfly_inference_finetune_grpo_report/v9",
        "antfly_inference_finetune_grpo_report/v10",
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
from gemma4_grpo_sampling import SamplingPolicy, categorical_rollout_group, _f32  # noqa: E402
from gemma4_mlx_source import attest_mlx_lm_archive  # noqa: E402

locked = microbenchmark.locked


class MultiTokenParityError(RuntimeError):
    """A pinned input, runtime, algorithm, or artifact contract drifted."""


def grpo_loss_terms(
    mx: Any,
    new_logps: Any,
    old_logps: Any,
    reference_logps: Any,
    advantage: Any,
    mask: Any,
    kl_coef: Any,
    group_token_count: Any,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """BNPO terms shared by sequential and coalesced completion execution."""
    ratio = mx.exp(new_logps - old_logps)
    pg_unclipped = ratio * advantage
    pg_clipped = (
        mx.clip(
            ratio,
            1.0 - GRPO["clip_epsilon"],
            1.0 + GRPO["clip_epsilon"],
        )
        * advantage
    )
    pg_tokens = -mx.minimum(pg_unclipped, pg_clipped)
    difference = reference_logps - new_logps
    raw_kl_tokens = mx.maximum(mx.expm1(difference) - difference, 0.0)
    pg_loss = mx.sum(pg_tokens * mask) / group_token_count
    mean_kl_value = mx.sum(raw_kl_tokens * mask) / group_token_count
    kl_loss = kl_coef * mean_kl_value
    loss = pg_loss + kl_loss
    clip_fraction = (
        mx.sum((pg_clipped < pg_unclipped).astype(mx.float32) * mask)
        / group_token_count
    )
    return (
        loss,
        pg_loss,
        kl_loss,
        mean_kl_value,
        clip_fraction,
        new_logps * mask,
    )


def require_captured_tokens(tokens: Sequence[int], captured: frozenset[int]) -> None:
    missing = set(int(token) for token in tokens) - captured
    if missing:
        raise MultiTokenParityError(
            f"frozen PLE replay encountered uncaptured tokens: {sorted(missing)[:8]}"
        )


def compact_frozen_ple_embedding(
    mx: Any, nn: Any, embedding: Any, token_ids: Sequence[int]
) -> tuple[Any, Mapping[str, Any]]:
    """Retain exact frozen rows for an explicitly bounded replay token set."""
    weight = embedding.weight
    ids = sorted(set(int(token) for token in token_ids))
    if weight.ndim != 2 or weight.dtype != mx.bfloat16:
        raise MultiTokenParityError("frozen PLE cache requires a BF16 embedding table")
    if not ids or ids[0] < 0 or ids[-1] >= weight.shape[0]:
        raise MultiTokenParityError("frozen PLE cache token IDs exceed the vocabulary")
    rows = weight[mx.array(ids, dtype=mx.int32)]
    mx.eval(rows)
    # This equality check runs before the full table's last owner is released.
    if not bool(mx.array_equal(rows, weight[mx.array(ids, dtype=mx.int32)]).item()):
        raise MultiTokenParityError("frozen PLE cache changed gathered row bytes")
    lookup = [-1] * weight.shape[0]
    for index, token in enumerate(ids):
        lookup[token] = index
    # Unknown rows are poison as a second line of defense. All Python token
    # constructors reject uncaptured IDs before invoking a compiled forward.
    rows = mx.concatenate(
        [rows, mx.full((1, weight.shape[1]), float("nan"), dtype=weight.dtype)]
    )
    lookup = mx.array(lookup, dtype=mx.int32)
    mx.eval(rows, lookup)

    class FrozenRows(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = rows
            self.lookup = lookup
            self.freeze()

        def __call__(self, tokens: Any) -> Any:
            return self.weight[self.lookup[tokens]]

    return FrozenRows(), {
        "mode": "fixed-token-frozen-ple-rows/v1",
        "original_shape": list(weight.shape),
        "retained_rows": len(ids),
        "token_ids": ids,
        "gathers_verified_exact": True,
        "performance_qualification_eligible": False,
    }


def validate_completion_execution(mode: str, max_completion_tokens: int) -> None:
    if mode not in {
        "sequential",
        "compiled-group",
        "coalesced-single-token",
        "coalesced-single-token-eager",
    }:
        raise MultiTokenParityError("unsupported completion execution mode")
    if mode.startswith("coalesced-single-token") and max_completion_tokens != 1:
        raise MultiTokenParityError(
            "coalesced execution requires one-token completions"
        )


@dataclass(frozen=True)
class RecipeProfile:
    target_preset: str
    sequence_length: int
    learning_rate: float
    advantage_epsilon: float
    min_kl_coef: float = GRPO["min_kl_coef"]
    max_kl_coef: float = GRPO["max_kl_coef"]


RECIPE_PROFILES = {
    "qv-multitoken": RecipeProfile(
        TARGET_PRESET, SEQUENCE_LENGTH, LEARNING_RATE, GRPO["advantage_epsilon"]
    ),
    "all-linear-single-token": RecipeProfile("text-all-linear", 160, 5.0e-8, 1.0e-8),
    "all-linear-single-token-quality": RecipeProfile(
        "text-all-linear",
        160,
        1.0e-8,
        1.0e-8,
        0.04,
        4.0,
    ),
}

ALL_LINEAR_SINGLE_TOKEN_PROFILES = frozenset(
    {
        "all-linear-single-token",
        "all-linear-single-token-quality",
    }
)


@dataclass(frozen=True)
class CampaignSpec:
    model_key: str
    train_groups: int
    eval_groups: int
    group_size: int
    max_completion_tokens: int
    recipe_profile: str = "qv-multitoken"

    @property
    def profile(self) -> RecipeProfile:
        try:
            return RECIPE_PROFILES[self.recipe_profile]
        except KeyError as exc:
            raise MultiTokenParityError("unsupported GRPO recipe profile") from exc

    def validate(self) -> None:
        if self.model_key not in MODEL_KEYS:
            raise MultiTokenParityError("unsupported Gemma4 model key")
        if self.train_groups < 2 or self.eval_groups < 2:
            raise MultiTokenParityError(
                "matched campaigns require at least two train/eval groups"
            )
        profile = self.profile
        if self.recipe_profile in ALL_LINEAR_SINGLE_TOKEN_PROFILES:
            if self.group_size != 16 or self.max_completion_tokens != 1:
                raise MultiTokenParityError(
                    "all-linear single-token profile requires group 16 and one completion token"
                )
        else:
            if not 2 <= self.group_size <= 8:
                raise MultiTokenParityError("group size must be in [2, 8]")
            if not 2 <= self.max_completion_tokens <= 32:
                raise MultiTokenParityError(
                    "multi-token completion budget must be in [2, 32]"
                )
        if self.max_completion_tokens >= profile.sequence_length:
            raise MultiTokenParityError(
                "completion budget exceeds the sequence contract"
            )


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
    eval_report_path: Path
    train_dataset_path: Path
    train_source_ids: tuple[str, ...]
    train_source_row_indices: tuple[int, ...]
    kl_trace_schema_version: str
    train_trace: tuple[TraceGroup, ...]
    eval_trace: tuple[TraceGroup, ...]
    trained_adapter_dir: Path | None


CATEGORICAL_MODES = frozenset(
    {
        "shared-prompt-seeded-categorical-sparse-row",
        "shared-page-prompt-seeded-categorical-incremental-kv",
        "compiled-shared-prompt-seeded-categorical-sparse-row-each-step",
        "shared-prompt-seeded-categorical-sparse-row-each-step",
        "shared-prompt-seeded-categorical",
    }
)


def categorical_contract(
    config: Mapping[str, Any],
    train: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> tuple[int, SamplingPolicy]:
    """Admit a current sampling contract for diagnostics, never acceptance."""
    recipe = config.get("recipe", {})
    if not isinstance(recipe, dict):
        raise MultiTokenParityError("categorical recipe is missing")
    optimizer, grpo = recipe.get("optimizer", {}), recipe.get("grpo", {})
    if not isinstance(optimizer, dict) or not isinstance(grpo, dict):
        raise MultiTokenParityError("categorical optimizer/sampling recipe is missing")
    seed = optimizer.get("seed", 42)
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or not 0 <= seed < 2**64
        or train.get("training_seed") != seed
    ):
        raise MultiTokenParityError("categorical training seed drifted")
    if (
        train.get("schema_version")
        not in {
            "antfly_inference_finetune_grpo_report/v8",
            "antfly_inference_finetune_grpo_report/v9",
            "antfly_inference_finetune_grpo_report/v10",
        }
        or train.get("training_order") != GRPO_TRAINING_ORDER
        or train.get("sampling_mode") not in CATEGORICAL_MODES
        or evaluation.get("schema_version")
        != "antfly_inference_finetune_grpo_evaluation/v4"
    ):
        raise MultiTokenParityError("categorical report/order contract drifted")
    raw = grpo.get("sampling")
    if not isinstance(raw, dict) or set(raw) != {"temperature", "top_p", "top_k"}:
        raise MultiTokenParityError("categorical recipe sampling must be explicit")
    try:
        finite_float(raw["temperature"], "sampling.temperature")
        finite_float(raw["top_p"], "sampling.top_p")
        policy = SamplingPolicy(**raw)
    except (ValueError, TypeError) as exc:
        raise MultiTokenParityError("invalid categorical sampling policy") from exc
    for report, greedy in ((train, False), (evaluation, True)):
        actual = report.get("sampling")
        if (
            not isinstance(actual, dict)
            or set(actual)
            != {
                "temperature",
                "top_p",
                "top_k",
                "first_completion_greedy",
                "algorithm",
                "scoring",
                "stream_derivation",
            }
            or actual.get("scoring") != "temperature-scaled-full-vocabulary/v1"
            or actual.get("algorithm") != "seeded-categorical-temperature-top-k-top-p"
            or actual.get("stream_derivation")
            != "run-seed-domain-epoch-dataset-prompt-index-completion/v2"
            or actual.get("first_completion_greedy") is not greedy
            or type(actual.get("top_k")) is not int
            or actual["top_k"] != policy.top_k
        ):
            raise MultiTokenParityError("categorical sampling phase/top-k drifted")
        for field in ("temperature", "top_p"):
            require_close(
                actual.get(field), getattr(policy, field), f"sampling.{field}"
            )
    return seed, policy


def group_admission(rewards: Sequence[float], mean_kl: float) -> str:
    """Match Zig's ordering with the runner's unmasked completion contract."""
    if not rewards or not all(math.isfinite(value) for value in rewards):
        raise MultiTokenParityError("group rewards must be finite and nonempty")
    if all(value == rewards[0] for value in rewards):
        return "zero-reward-std-skipped"
    if not math.isfinite(mean_kl) or mean_kl < 0:
        raise MultiTokenParityError("group KL must be finite and nonnegative")
    if _f32(mean_kl) > _f32(GRPO["train_max_kl"]):
        return "budget-exceeded-skipped"
    return "admitted"


def campaign_classification(
    trace_close: bool, behavior_close: bool, *, categorical: bool
) -> str:
    if categorical:
        return "categorical-diagnostic-only"
    return (
        "bounded-behavior-and-update-parity"
        if trace_close and behavior_close
        else "bounded-campaign-with-measured-drift"
    )


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


def require_close(
    value: Any, expected: float, where: str, tolerance: float = 1.0e-6
) -> None:
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


def adaptive_kl_update(
    current: float,
    mean_kl: float,
    observed_completions: int = 1,
    *,
    min_kl_coef: float = GRPO["min_kl_coef"],
    max_kl_coef: float = GRPO["max_kl_coef"],
) -> float:
    if not math.isfinite(mean_kl) or mean_kl < 0.0:
        raise MultiTokenParityError(
            "adaptive KL observation must be finite and non-negative"
        )
    if (
        isinstance(observed_completions, bool)
        or not isinstance(observed_completions, int)
        or observed_completions <= 0
    ):
        raise MultiTokenParityError("adaptive KL observed completions must be positive")
    current_f32 = _f32(current)
    mean_kl_f32 = _f32(mean_kl)
    target_f32 = _f32(GRPO["target_kl"])
    horizon_f32 = _f32(GRPO["kl_horizon"])
    min_f32 = _f32(min_kl_coef)
    max_f32 = _f32(max_kl_coef)
    proportional_error = min(max(mean_kl_f32 / target_f32 - 1.0, -0.2), 0.2)
    updated = current_f32 * (
        1.0 + proportional_error * observed_completions / horizon_f32
    )
    return _f32(min(max(updated, min_f32), max_f32))


def controller_observed_completions(
    row: Mapping[str, Any],
    schema_version: str,
    group_size: int,
) -> int:
    if schema_version not in {
        "antfly_inference_grpo_kl_control_trace/v4",
        KL_TRACE_SCHEMA_VERSION,
    }:
        return 1
    observed = row.get("observed_completions")
    if (
        isinstance(observed, bool)
        or not isinstance(observed, int)
        or observed != group_size
    ):
        raise MultiTokenParityError("Antfly KL trace observed-completion count drifted")
    return observed


def mean_k3(policy_logps: Sequence[float], reference_logps: Sequence[float]) -> float:
    if not policy_logps or len(policy_logps) != len(reference_logps):
        raise MultiTokenParityError("KL vectors must be non-empty and equal-length")
    values: list[float] = []
    for policy, reference in zip(policy_logps, reference_logps):
        difference = reference - policy
        if not math.isfinite(difference) or difference > 80.0:
            raise MultiTokenParityError(
                "GRPO KL log-ratio is outside the finite f32 contract"
            )
        values.append(max(math.expm1(difference) - difference, 0.0))
    return statistics.mean(values)


def prefix_match_reward(decoded: str, target: str) -> float:
    completion = decoded.strip(" \t\r\n")
    expected = target.strip(" \t\r\n")
    return 1.0 if completion.startswith(expected) else 0.0


def decode_reward(
    tokenizer: Any, token_ids: Sequence[int], target: str
) -> tuple[str, float]:
    decoded = tokenizer.decode(list(token_ids), skip_special_tokens=True)
    return decoded, prefix_match_reward(decoded, target)


def sequence_overlap(
    actual: Sequence[Sequence[int]],
    expected: Sequence[Sequence[int]],
    *,
    with_replacement: bool = False,
) -> Mapping[str, Any]:
    actual_tuples = tuple(tuple(item) for item in actual)
    expected_tuples = tuple(tuple(item) for item in expected)
    if (
        not actual_tuples
        or len(actual_tuples) != len(expected_tuples)
        or any(not item for item in actual_tuples + expected_tuples)
        or (
            not with_replacement
            and (
                len(set(actual_tuples)) != len(actual_tuples)
                or len(set(expected_tuples)) != len(expected_tuples)
            )
        )
    ):
        raise MultiTokenParityError(
            "completion groups must be distinct and equal-length"
        )
    overlap = (
        sum((Counter(actual_tuples) & Counter(expected_tuples)).values())
        if with_replacement
        else len(set(actual_tuples) & set(expected_tuples))
    )
    actual_first = tuple(item[0] for item in actual_tuples)
    expected_first = tuple(item[0] for item in expected_tuples)
    first_overlap = (
        sum((Counter(actual_first) & Counter(expected_first)).values())
        if with_replacement
        else len(set(actual_first) & set(expected_first))
    )
    return {
        "sequence_overlap": overlap,
        "sequence_recall": overlap / len(expected_tuples),
        "exact_sequence_set": set(actual_tuples) == set(expected_tuples),
        "exact_sequence_multiset": Counter(actual_tuples) == Counter(expected_tuples),
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
        "mean_sequence_recall": statistics.mean(
            float(row["sequence_recall"]) for row in rows
        ),
        "exact_sequence_set_rate": statistics.mean(
            1.0 if row["exact_sequence_set"] else 0.0 for row in rows
        ),
        "exact_sequence_order_rate": statistics.mean(
            1.0 if row["exact_sequence_order"] else 0.0 for row in rows
        ),
        "top_sequence_match_rate": statistics.mean(
            1.0 if row["top_sequence_match"] else 0.0 for row in rows
        ),
        "mean_first_token_recall": statistics.mean(
            float(row["first_token_recall"]) for row in rows
        ),
        "exact_first_token_order_rate": statistics.mean(
            1.0 if row["exact_first_token_order"] else 0.0 for row in rows
        ),
        "top1_first_token_match_rate": statistics.mean(
            1.0 if row["top1_first_token_match"] else 0.0 for row in rows
        ),
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
    previous_prompt: int | None = None
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise MultiTokenParityError(
            f"could not load {phase} reward trace: {exc}"
        ) from exc
    for line_index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MultiTokenParityError(
                f"{phase} reward trace contains invalid JSON"
            ) from exc
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != REWARD_TRACE_SCHEMA_VERSION
        ):
            raise MultiTokenParityError(f"{phase} reward trace schema drifted")
        if row.get("phase") != phase or row.get("call_index") != line_index:
            raise MultiTokenParityError(f"{phase} reward trace order drifted")
        prompt_index = row.get("prompt_index")
        if (
            isinstance(prompt_index, bool)
            or not isinstance(prompt_index, int)
            or prompt_index < 0
        ):
            raise MultiTokenParityError(f"{phase} prompt index is invalid")
        raw_tokens = row.get("completion_tokens")
        if (
            not isinstance(raw_tokens, list)
            or not 1 <= len(raw_tokens) <= max_completion_tokens
            or any(
                isinstance(token, bool) or not isinstance(token, int) or token < 0
                for token in raw_tokens
            )
        ):
            raise MultiTokenParityError(f"{phase} completion token sequence is invalid")
        reward = finite_float(row.get("aggregate_reward"), "aggregate_reward")
        if reward not in (0.0, 1.0):
            raise MultiTokenParityError("BoolQ trace reward must be binary")
        if prompt_index in groups and prompt_index != previous_prompt:
            raise MultiTokenParityError(
                f"{phase} reward trace optimizer groups are interleaved"
            )
        groups.setdefault(prompt_index, []).append(
            TraceCompletion(tuple(raw_tokens), reward)
        )
        previous_prompt = prompt_index
    if sorted(groups) != list(range(expected_groups)):
        raise MultiTokenParityError(
            f"{phase} reward trace prompt groups are incomplete"
        )
    result = tuple(
        TraceGroup(index, tuple(completions)) for index, completions in groups.items()
    )
    for group in result:
        if len(group.completions) != group_size:
            raise MultiTokenParityError(f"{phase} reward trace group size drifted")
    return result


def load_materialization(
    path: Path, spec: CampaignSpec, model_dir: Path
) -> Mapping[str, Any]:
    manifest = load_json(path.expanduser().resolve(), "BoolQ materialization manifest")
    if manifest.get("schema_version") not in MATERIALIZATION_SCHEMA_VERSIONS:
        raise MultiTokenParityError("unsupported BoolQ materialization schema")
    try:
        boolq_materializer.validate_materialization_semantic_sha256(manifest)
        boolq_materializer.validate_materialization_selection_contract(manifest)
    except boolq_materializer.MaterializationError as exc:
        raise MultiTokenParityError(str(exc)) from exc
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("repo_id") != "google/boolq":
        raise MultiTokenParityError("materialization is not google/boolq")
    revision = dataset.get("revision")
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(char not in "0123456789abcdef" for char in revision)
    ):
        raise MultiTokenParityError(
            "BoolQ revision must be a full lowercase Git commit"
        )
    policy = dataset.get("selection_policy")
    expected_policy = {
        "dataset_format": "rendered-text-grpo",
        "max_completion_tokens": spec.max_completion_tokens,
        "target_tokens": 1,
        "rendered_prompt_truncation": "forbidden",
        "response_channel": "final",
    }
    if not isinstance(policy, dict) or any(
        policy.get(key) != value for key, value in expected_policy.items()
    ):
        raise MultiTokenParityError("BoolQ selection policy differs from the campaign")
    admission_length = policy.get("max_seq_len")
    if (
        isinstance(admission_length, bool)
        or not isinstance(admission_length, int)
        or not 16 <= admission_length <= spec.profile.sequence_length
        or (
            spec.recipe_profile == "qv-multitoken"
            and admission_length != spec.profile.sequence_length
        )
    ):
        raise MultiTokenParityError("BoolQ admission length differs from the campaign")
    for section, manifest_key in (
        ("train", "train_jsonl"),
        ("evaluation", "eval_jsonl"),
    ):
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
        raise MultiTokenParityError(
            "materialization tokenizer fingerprints are missing"
        )
    for name in ("tokenizer.json", "tokenizer_config.json"):
        tokenizer_path = model_dir / name
        if not tokenizer_path.is_file() or tokenizer_files.get(name) != sha256_file(
            tokenizer_path
        ):
            raise MultiTokenParityError(
                f"runtime {name} differs from the materialization"
            )
    return manifest


def load_campaign_materialization(
    path: Path,
    evaluation_path: Path | None,
    spec: CampaignSpec,
    model_dir: Path,
) -> Mapping[str, Any]:
    """Bind independently verified train/eval materializations without relabeling either."""
    train = load_materialization(path, spec, model_dir)
    if evaluation_path is None:
        return train
    evaluation = load_materialization(evaluation_path, spec, model_dir)
    for field in ("repo_id", "revision"):
        if train["dataset"][field] != evaluation["dataset"][field]:
            raise MultiTokenParityError("train/evaluation dataset revisions differ")
    for field in ("tokenizer_files", "dependency_versions"):
        if train.get(field) != evaluation.get(field):
            raise MultiTokenParityError(f"train/evaluation {field} differ")
    if set(train["train_source_ids"]) & set(evaluation["eval_source_ids"]):
        raise MultiTokenParityError("campaign train/evaluation identities overlap")
    # This is an in-memory campaign view, not a newly materialized or self-attested dataset.
    view = {key: value for key, value in train.items() if key != "semantic_sha256"}
    view["schema_version"] = "antfly_gemma4_grpo_campaign_dataset/v1"
    for field in (
        "eval_jsonl",
        "eval_prompt_tokens",
        "eval_source_ids",
        "eval_source_row_indices",
        "evaluation_exclusion_manifests",
    ):
        view[field] = evaluation.get(field)
    dataset = dict(train["dataset"])
    dataset.pop("selection_policy")
    dataset["evaluation"] = evaluation["dataset"]["evaluation"]
    dataset["train_selection_policy"] = train["dataset"]["selection_policy"]
    dataset["evaluation_selection_policy"] = evaluation["dataset"]["selection_policy"]
    view["dataset"] = dataset
    view["campaign_manifest_bindings"] = {
        role: {
            "path": str(source.expanduser().resolve()),
            "sha256": sha256_file(source.expanduser().resolve()),
            "semantic_sha256": manifest["semantic_sha256"],
            "selection_policy": manifest["dataset"]["selection_policy"],
        }
        for role, source, manifest in (
            ("train", path, train),
            ("evaluation", evaluation_path, evaluation),
        )
    }
    return view


def load_rows(
    path: Path,
    *,
    expected_count: int,
    expected_ids: Sequence[str],
    expected_indices: Sequence[int],
    tokenizer: Any,
    max_completion_tokens: int,
    sequence_length: int = SEQUENCE_LENGTH,
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
            raise MultiTokenParityError(
                f"BoolQ JSONL line {line_index + 1} is invalid"
            ) from exc
        if not isinstance(payload, dict) or set(payload) != {
            "prompt",
            "target",
            "metadata",
        }:
            raise MultiTokenParityError("BoolQ JSONL row schema drifted")
        prompt, target, metadata = (
            payload["prompt"],
            payload["target"],
            payload["metadata"],
        )
        if (
            not isinstance(prompt, str)
            or not prompt
            or target not in ("yes", "no")
            or not isinstance(metadata, dict)
        ):
            raise MultiTokenParityError("BoolQ JSONL row is malformed")
        prompt_ids = tuple(
            int(value)
            for value in tokenizer.encode(prompt, add_special_tokens=False).ids
        )
        target_ids = tuple(
            int(value)
            for value in tokenizer.encode(target, add_special_tokens=False).ids
        )
        if (
            not prompt_ids
            or len(prompt_ids) + max_completion_tokens > sequence_length
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
    if [row.source_row_index for row in selected] != list(
        expected_indices[:expected_count]
    ):
        raise MultiTokenParityError("BoolQ source row order drifted")
    return selected


def validate_kl_trace(root: Path, report: Mapping[str, Any], spec: CampaignSpec) -> str:
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
    if (
        report.get("schema_version")
        in {
            "antfly_inference_finetune_grpo_report/v9",
            "antfly_inference_finetune_grpo_report/v10",
        }
        and telemetry.get("kl_horizon_unit") != "completion-episodes"
    ):
        raise MultiTokenParityError("Antfly adaptive KL horizon unit drifted")
    for field, expected in (
        ("train_max_kl", GRPO["train_max_kl"]),
        ("target_kl", GRPO["target_kl"]),
        ("kl_horizon", GRPO["kl_horizon"]),
        ("initial_kl_coef", GRPO["initial_kl_coef"]),
        ("min_kl_coef", spec.profile.min_kl_coef),
        ("max_kl_coef", spec.profile.max_kl_coef),
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
    schema_version: str | None = None
    for index, line in enumerate(lines):
        row = json.loads(line)
        row_schema = row.get("schema_version") if isinstance(row, dict) else None
        if row_schema not in KL_TRACE_SCHEMA_VERSIONS:
            raise MultiTokenParityError("Antfly KL trace schema drifted")
        if schema_version is None:
            schema_version = row_schema
        if (
            not isinstance(row, dict)
            or row_schema != schema_version
            or row.get("group_index") != index
            or row.get("status") != "admitted"
            or row.get("budget_policy") != "skip_group"
        ):
            raise MultiTokenParityError("Antfly KL trace decision/order drifted")
        before = finite_float(row.get("kl_coef_before"), "kl_coef_before")
        after = finite_float(row.get("kl_coef_after"), "kl_coef_after")
        if row_schema == KL_TRACE_SCHEMA_VERSION:
            require_close(
                row.get("objective_kl_coef"), before, "objective_kl_coef", 0.0
            )
        observed = finite_float(row.get("mean_kl"), "mean_kl")
        if observed > GRPO["train_max_kl"]:
            raise MultiTokenParityError(
                "Antfly admitted a group above the hard KL budget"
            )
        if previous_after is not None and abs(before - previous_after) > 1.0e-7:
            raise MultiTokenParityError(
                "Antfly KL coefficient trajectory is discontinuous"
            )
        observed_completions = controller_observed_completions(
            row, schema_version, spec.group_size
        )
        expected_after = adaptive_kl_update(
            before,
            observed,
            observed_completions,
            min_kl_coef=spec.profile.min_kl_coef,
            max_kl_coef=spec.profile.max_kl_coef,
        )
        if abs(after - expected_after) > 2.0e-7:
            raise MultiTokenParityError(
                "Antfly adaptive KL update differs from the matched rule"
            )
        previous_after = after
    assert schema_version is not None
    return schema_version


def validate_categorical_groups(
    root: Path,
    report: Mapping[str, Any],
    spec: CampaignSpec,
    trace: Sequence[TraceGroup],
) -> str:
    """Bind skips and Adam step indices to chronological reward/KL evidence."""
    telemetry = report.get("kl_control")
    if (
        not isinstance(telemetry, dict)
        or telemetry.get("mode") != "adaptive"
        or telemetry.get("budget_policy") != "skip_group"
    ):
        raise MultiTokenParityError("categorical KL control contract drifted")
    if (
        report.get("schema_version")
        in {
            "antfly_inference_finetune_grpo_report/v9",
            "antfly_inference_finetune_grpo_report/v10",
        }
        and telemetry.get("kl_horizon_unit") != "completion-episodes"
    ):
        raise MultiTokenParityError("categorical KL horizon unit drifted")
    kl_contract = dict(GRPO)
    kl_contract.update(
        min_kl_coef=spec.profile.min_kl_coef,
        max_kl_coef=spec.profile.max_kl_coef,
    )
    for field in (
        "train_max_kl",
        "target_kl",
        "kl_horizon",
        "initial_kl_coef",
        "min_kl_coef",
        "max_kl_coef",
    ):
        require_close(telemetry.get(field), kl_contract[field], f"kl_control.{field}")
    path = root / "grpo_kl_control_trace.jsonl"
    if telemetry.get("trace_path") != str(path) or telemetry.get(
        "trace_digest"
    ) != "sha256:" + sha256_file(path):
        raise MultiTokenParityError("categorical KL trace identity drifted")
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    schemas = {row.get("schema_version") for row in records if isinstance(row, dict)}
    if not records:
        schema_version = KL_TRACE_SCHEMA_VERSION
    elif len(schemas) == 1 and schemas.issubset(KL_TRACE_SCHEMA_VERSIONS):
        schema_version = next(iter(schemas))
    else:
        raise MultiTokenParityError("categorical KL trace schema drifted")
    cursor = admitted = zero = rejected = 0
    coefficient = GRPO["initial_kl_coef"]
    for index, group in enumerate(trace):
        if all(value == group.rewards[0] for value in group.rewards):
            zero += 1
            continue
        if cursor >= len(records):
            raise MultiTokenParityError("categorical KL decision is missing")
        row = records[cursor]
        cursor += 1
        if (
            not isinstance(row, dict)
            or row.get("schema_version") != schema_version
            or row.get("group_index") != index
            or row.get("epoch_index") != 0
            or row.get("prompt_index") != group.prompt_index
            or row.get("optimizer_steps_before") != admitted
            or row.get("budget_policy") != "skip_group"
        ):
            raise MultiTokenParityError("categorical KL decision/order drifted")
        observed = finite_float(row.get("mean_kl"), "mean_kl")
        status = group_admission(group.rewards, observed)
        if row.get("status") != status:
            raise MultiTokenParityError(
                "categorical KL admission disagrees with budget"
            )
        require_close(row.get("train_max_kl"), GRPO["train_max_kl"], "train_max_kl")
        require_close(row.get("target_kl"), GRPO["target_kl"], "target_kl")
        before = finite_float(row.get("kl_coef_before"), "kl_coef_before")
        require_close(before, coefficient, "kl_coef_before", 2e-7)
        if schema_version == KL_TRACE_SCHEMA_VERSION:
            require_close(
                row.get("objective_kl_coef"),
                before,
                "objective_kl_coef",
                0.0,
            )
        require_close(
            row.get("weighted_kl_loss"),
            coefficient * observed,
            "weighted_kl_loss",
            2e-7,
        )
        if status == "admitted":
            admitted += 1
        if status == "admitted" or schema_version in {
            "antfly_inference_grpo_kl_control_trace/v3",
            "antfly_inference_grpo_kl_control_trace/v4",
            KL_TRACE_SCHEMA_VERSION,
        }:
            observed_completions = controller_observed_completions(
                row, schema_version, spec.group_size
            )
            coefficient = adaptive_kl_update(
                coefficient,
                observed,
                observed_completions,
                min_kl_coef=spec.profile.min_kl_coef,
                max_kl_coef=spec.profile.max_kl_coef,
            )
        if status != "admitted":
            rejected += 1
        require_close(row.get("kl_coef_after"), coefficient, "kl_coef_after", 2e-7)
    if cursor != len(records):
        raise MultiTokenParityError("categorical KL trace contains extra decisions")
    counts = {
        "optimizer_steps": admitted,
        "optimizer_groups": admitted,
        "zero_reward_std_groups": zero,
        "all_truncated_groups": 0,
        "kl_rejected_groups": rejected,
    }
    if (
        len(trace) != spec.train_groups
        or admitted + zero + rejected != spec.train_groups
        or any(
            type(report.get(k)) is not int or report[k] != v for k, v in counts.items()
        )
        or telemetry.get("admitted_groups") != admitted
        or telemetry.get("rejected_groups") != rejected
    ):
        raise MultiTokenParityError("categorical skipped-group counts drifted")
    require_close(
        report.get("frac_reward_zero_std"),
        zero / spec.train_groups,
        "frac_reward_zero_std",
    )
    require_close(
        report.get("frac_kl_rejected"), rejected / spec.train_groups, "frac_kl_rejected"
    )
    return schema_version


def bind_training_dataset(
    config: Mapping[str, Any],
    dataset_path: Path,
    manifest: Mapping[str, Any],
    expected_count: int,
) -> tuple[tuple[str, ...], tuple[int, ...]]:
    """Bind a campaign seed permutation to its admitted source-row multiset."""
    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.is_file():
        raise MultiTokenParityError("Antfly training dataset is missing")
    metadata = config.get("metadata")
    fingerprints = (
        metadata.get("dataset_fingerprints") if isinstance(metadata, dict) else None
    )
    expected_digest = "sha256:" + sha256_file(dataset_path)
    if not isinstance(fingerprints, list) or not any(
        isinstance(item, dict)
        and item.get("label") == "dataset"
        and Path(str(item.get("path", ""))).expanduser().resolve() == dataset_path
        and item.get("digest") == expected_digest
        and item.get("size_bytes") == dataset_path.stat().st_size
        for item in fingerprints
    ):
        raise MultiTokenParityError(
            "Antfly training dataset fingerprint is missing or stale"
        )

    source_ids: list[str] = []
    source_indices: list[int] = []
    try:
        lines = dataset_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            row = json.loads(line)
            row_metadata = row.get("metadata") if isinstance(row, dict) else None
            source_id = (
                row_metadata.get("source_id")
                if isinstance(row_metadata, dict)
                else None
            )
            source_index = (
                row_metadata.get("source_row_index")
                if isinstance(row_metadata, dict)
                else None
            )
            if (
                not isinstance(source_id, str)
                or isinstance(source_index, bool)
                or not isinstance(source_index, int)
            ):
                raise MultiTokenParityError(
                    "Antfly training dataset source identity is invalid"
                )
            source_ids.append(source_id)
            source_indices.append(source_index)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MultiTokenParityError(
            f"could not validate Antfly training dataset: {exc}"
        ) from exc
    if len(source_ids) < expected_count:
        raise MultiTokenParityError("Antfly training dataset row count drifted")
    # The recipe admits a prefix via max_examples; the fingerprint above still
    # binds the entire file, including rows outside that training horizon.
    source_ids = source_ids[:expected_count]
    source_indices = source_indices[:expected_count]
    admitted_pairs = list(zip(source_ids, source_indices))
    manifest_pairs = list(
        zip(
            manifest.get("train_source_ids", [])[:expected_count],
            manifest.get("train_source_row_indices", [])[:expected_count],
        )
    )
    if len(manifest_pairs) != expected_count or Counter(admitted_pairs) != Counter(
        manifest_pairs
    ):
        raise MultiTokenParityError(
            "Antfly training dataset is not the admitted source-row multiset"
        )
    return tuple(source_ids), tuple(source_indices)


def load_acceptance(
    root: Path,
    manifest: Mapping[str, Any],
    spec: CampaignSpec,
    model_dir: Path,
    adapter_dir: Path,
    *,
    categorical_diagnostic: bool = False,
) -> AcceptanceEvidence:
    spec.validate()
    if (
        spec.recipe_profile in ALL_LINEAR_SINGLE_TOKEN_PROFILES
        and not categorical_diagnostic
    ):
        raise MultiTokenParityError(
            "all-linear single-token profile requires categorical diagnostics"
        )
    evidence_root = root.expanduser().resolve()
    config = load_json(evidence_root / "training_config.json", "Antfly training config")
    train_report = load_json(evidence_root / "grpo_report.json", "Antfly GRPO report")
    reported_evaluation = train_report.get("evaluation")
    reported_evaluation_path = (
        reported_evaluation.get("report_path")
        if isinstance(reported_evaluation, dict)
        else None
    )
    if reported_evaluation_path is None:
        eval_report_path = evidence_root / "grpo_evaluation_report.json"
    elif isinstance(reported_evaluation_path, str):
        eval_report_path = Path(reported_evaluation_path).expanduser().resolve()
        if eval_report_path.parent != evidence_root:
            raise MultiTokenParityError(
                "Antfly GRPO evaluation report escaped the campaign root"
            )
    else:
        raise MultiTokenParityError("Antfly GRPO evaluation report path is invalid")
    eval_report = load_json(eval_report_path, "Antfly GRPO evaluation report")
    if train_report.get("schema_version") not in GRPO_REPORT_SCHEMA_VERSIONS:
        raise MultiTokenParityError("Antfly GRPO report is not the adaptive-KL schema")
    if (
        train_report.get("schema_version")
        in {
            "antfly_inference_finetune_grpo_report/v8",
            "antfly_inference_finetune_grpo_report/v9",
            "antfly_inference_finetune_grpo_report/v10",
        }
        and train_report.get("training_order") != GRPO_TRAINING_ORDER
    ):
        raise MultiTokenParityError("Antfly GRPO training-order contract drifted")
    if categorical_diagnostic:
        categorical_contract(config, train_report, eval_report)
    else:
        try:
            legacy.require_native_rollout_sampler_compatibility(train_report)
        except legacy.BoolQParityContractError as exc:
            raise MultiTokenParityError(str(exc)) from exc
    if eval_report.get("schema_version") not in GRPO_EVAL_SCHEMA_VERSIONS:
        raise MultiTokenParityError("Antfly GRPO evaluation is not the raw-KL schema")
    if (
        train_report.get("execution_mode") != "train"
        or train_report.get("dataset_format") != "rendered-text-grpo"
    ):
        raise MultiTokenParityError(
            "Antfly evidence is not optimizer-backed rendered GRPO"
        )
    expected_counts = {
        "groups": spec.train_groups,
        "completions": spec.train_groups * spec.group_size,
    }
    if not categorical_diagnostic:
        expected_counts["optimizer_steps"] = spec.train_groups
    if any(train_report.get(key) != value for key, value in expected_counts.items()):
        raise MultiTokenParityError(
            "Antfly training counts differ from the matched campaign"
        )
    if train_report.get("policy_backend") != "metal":
        raise MultiTokenParityError("Antfly campaign must run on Metal")
    allowed_eval_statuses = (
        {"passed", "failed", "failed-quality-gate"}
        if categorical_diagnostic
        else {"passed"}
    )
    if (
        eval_report.get("status") not in allowed_eval_statuses
        or eval_report.get("groups") != spec.eval_groups
    ):
        raise MultiTokenParityError("Antfly held-out campaign did not pass")
    if eval_report.get("mask_truncated_completions") is not False:
        raise MultiTokenParityError("Antfly evaluation truncation policy drifted")
    if train_report.get("mean_kl") is None or eval_report.get("mean_kl") is None:
        raise MultiTokenParityError("Antfly raw KL metrics are missing")
    if train_report.get("schema_version") in {
        "antfly_inference_finetune_grpo_report/v7",
        "antfly_inference_finetune_grpo_report/v8",
        "antfly_inference_finetune_grpo_report/v9",
        "antfly_inference_finetune_grpo_report/v10",
    }:
        for field in ("epsilon_low", "epsilon_high"):
            value = finite_float(train_report.get(field), f"report.{field}")
            if value not in (GRPO["clip_epsilon"], _f32(GRPO["clip_epsilon"])):
                raise MultiTokenParityError(
                    f"report.{field} differs from the matched campaign"
                )
        if (
            (
                not categorical_diagnostic
                and (
                    train_report.get("optimizer_groups") != spec.train_groups
                    or train_report.get("zero_reward_std_groups") != 0
                    or train_report.get("all_truncated_groups") != 0
                    or train_report.get("kl_rejected_groups") != 0
                    or float(train_report.get("frac_reward_zero_std", -1.0)) != 0.0
                    or float(train_report.get("frac_kl_rejected", -1.0)) != 0.0
                )
            )
            or train_report.get("loss_type") != "bnpo"
            or train_report.get("scale_rewards") != "group"
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
    if not all(
        isinstance(value, dict)
        for value in (model, dataset, adapter, optimizer, grpo, evaluation)
    ):
        raise MultiTokenParityError("Antfly normalized recipe is incomplete")
    if (
        Path(str(model.get("path", ""))).resolve() != model_dir
        or model.get("family") != "gemma4"
    ):
        raise MultiTokenParityError("Antfly model identity differs from the campaign")
    if Path(str(adapter.get("path", ""))).resolve() != adapter_dir:
        raise MultiTokenParityError("Antfly seed adapter differs from the campaign")
    if adapter.get("rank") != 16 or float(adapter.get("alpha", 0.0)) != 32.0:
        raise MultiTokenParityError("Antfly adapter rank/alpha drifted")
    if adapter.get("target_preset") not in (None, spec.profile.target_preset):
        raise MultiTokenParityError("Antfly adapter target preset drifted")
    if spec.recipe_profile in ALL_LINEAR_SINGLE_TOKEN_PROFILES:
        if adapter.get("target_preset") != spec.profile.target_preset:
            raise MultiTokenParityError(
                "all-linear profile requires an explicit target preset"
            )
        if grpo.get("sampling") != {"temperature": 2.0, "top_p": 1.0, "top_k": 32}:
            raise MultiTokenParityError("all-linear profile sampling policy drifted")
    train_dataset_path = Path(str(dataset.get("path", ""))).expanduser().resolve()
    train_source_ids, train_source_row_indices = bind_training_dataset(
        config, train_dataset_path, manifest, spec.train_groups
    )
    if (
        evaluation.get("path") != manifest.get("eval_jsonl")
        or dataset.get("max_examples") != spec.train_groups
        or evaluation.get("max_examples") != spec.eval_groups
        or dataset.get("max_seq_len") != spec.profile.sequence_length
    ):
        raise MultiTokenParityError("Antfly dataset is not the pinned matched split")
    require_close(
        optimizer.get("learning_rate"),
        spec.profile.learning_rate,
        "optimizer.learning_rate",
        1.0e-14,
    )
    if (
        optimizer.get("epochs") != 1
        or optimizer.get("gradient_accumulation_steps") != 1
    ):
        raise MultiTokenParityError("Antfly optimizer schedule drifted")
    require_close(
        optimizer.get("max_grad_norm"),
        OPTIMIZER["max_grad_norm"],
        "optimizer.max_grad_norm",
    )
    if categorical_diagnostic:
        require_close(
            grpo.get("advantage_eps", 1e-4),
            spec.profile.advantage_epsilon,
            "grpo.advantage_eps",
            1e-14,
        )
        if grpo.get("train_max_kl_policy") != "skip_group":
            raise MultiTokenParityError("categorical KL budget policy drifted")
    if (
        grpo.get("group_size") != spec.group_size
        or grpo.get("max_completion_tokens") != spec.max_completion_tokens
    ):
        raise MultiTokenParityError("Antfly group/completion shape drifted")
    for field, expected in (
        ("clip_epsilon", GRPO["clip_epsilon"]),
        ("kl_coef", GRPO["initial_kl_coef"]),
        ("train_max_kl", GRPO["train_max_kl"]),
        ("target_kl", GRPO["target_kl"]),
        ("kl_horizon", GRPO["kl_horizon"]),
        ("min_kl_coef", spec.profile.min_kl_coef),
        ("max_kl_coef", spec.profile.max_kl_coef),
    ):
        require_close(grpo.get(field), expected, f"grpo.{field}")
    if (
        grpo.get("adaptive_kl") is not True
        or (
            grpo.get("normalize_advantage") is not None
            and grpo.get("normalize_advantage") is not True
        )
        or grpo.get("loss_type") not in (None, "bnpo")
        or grpo.get("scale_rewards") not in (None, "group")
        or grpo.get("epsilon_high") not in (None, GRPO["clip_epsilon"])
        or grpo.get("mask_truncated_completions") not in (None, False)
    ):
        raise MultiTokenParityError("Antfly adaptive/advantage policy drifted")
    if categorical_diagnostic:
        kl_trace_schema_version = validate_categorical_groups(
            evidence_root,
            train_report,
            spec,
            load_trace(
                evidence_root / "grpo_reward_trace.jsonl",
                phase="train",
                expected_groups=spec.train_groups,
                group_size=spec.group_size,
                max_completion_tokens=spec.max_completion_tokens,
            ),
        )
    else:
        kl_trace_schema_version = validate_kl_trace(evidence_root, train_report, spec)
    train_trace_path = evidence_root / "grpo_reward_trace.jsonl"
    eval_trace_path = evidence_root / "grpo_evaluation_reward_trace.jsonl"
    for trace_path, report in (
        (train_trace_path, train_report),
        (eval_trace_path, eval_report),
    ):
        telemetry = report.get("reward_pipeline")
        if not isinstance(telemetry, dict) or telemetry.get(
            "trace_digest"
        ) != "sha256:" + sha256_file(trace_path):
            raise MultiTokenParityError("Antfly reward trace digest drifted")
    raw_adapter_dir = train_report.get("trained_adapter_dir")
    quality_rejected = any(
        isinstance(train_report.get(field), dict)
        and train_report[field].get("passed") is False
        for field in ("evaluation", "baseline_relative")
    )
    if raw_adapter_dir is None and categorical_diagnostic and quality_rejected:
        # Failed quality must not force publication of an accepted adapter.
        # Replay/rollout mechanics remain measurable, with adapter parity unavailable.
        trained_adapter_dir = None
    else:
        trained_adapter_dir = Path(str(raw_adapter_dir or "")).resolve()
        if (
            trained_adapter_dir.parent != evidence_root
            or not trained_adapter_dir.is_dir()
        ):
            raise MultiTokenParityError(
                "Antfly trained adapter escaped the campaign root"
            )
    return AcceptanceEvidence(
        root=evidence_root,
        config=config,
        train_report=train_report,
        eval_report=eval_report,
        eval_report_path=eval_report_path,
        train_dataset_path=train_dataset_path,
        train_source_ids=train_source_ids,
        train_source_row_indices=train_source_row_indices,
        kl_trace_schema_version=kl_trace_schema_version,
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
            _decoded, reward = decode_reward(
                tokenizer, completion.token_ids, row.target
            )
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
        raise MultiTokenParityError(
            f"campaign output already exists: {destination}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def write_adapter_exclusive(
    path: Path,
    *,
    final_trainables: Mapping[str, Any],
    target_names: Sequence[str],
    adapter: Any,
    mx: Any,
) -> Mapping[str, Any]:
    """Persist MLX trainables in the seed adapter's canonical orientation."""
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise MultiTokenParityError(
            f"campaign adapter output already exists: {destination}"
        )
    mlx_targets = {locked.canonicalize_module_name(name): name for name in target_names}
    if len(mlx_targets) != len(target_names):
        raise MultiTokenParityError("MLX adapter target names are not canonical")
    serialized: dict[str, Any] = {}
    for (module, role), descriptor in adapter.tensors.items():
        suffix = "lora_a" if role == "lora_A" else "lora_b"
        try:
            value = final_trainables[f"{mlx_targets[module]}.{suffix}"]
        except KeyError as exc:
            raise MultiTokenParityError(
                f"final MLX adapter is missing {module}.{role}"
            ) from exc
        serialized[descriptor.source_name] = value.T.astype(mx.float32)
    expected_names = {item.source_name for item in adapter.tensors.values()}
    if set(serialized) != expected_names:
        raise MultiTokenParityError("serialized MLX adapter inventory drifted")
    temporary = destination.with_name(
        f".{destination.stem}.{os.getpid()}.tmp.safetensors"
    )
    try:
        mx.save_safetensors(str(temporary), serialized, metadata={"format": "pt"})
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise MultiTokenParityError(
            f"campaign adapter output already exists: {destination}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "sha256": sha256_file(destination),
        "tensor_count": len(serialized),
        "orientation": "seed-adapter-source-layout",
    }


def execution_lanes(selection: str, *, categorical: bool) -> tuple[str, ...]:
    if selection == "both":
        return ("trace_replay", "native_rollout")
    if selection not in ("trace-replay", "native-rollout"):
        raise MultiTokenParityError(f"unknown execution lane: {selection}")
    if not categorical:
        raise MultiTokenParityError(
            "individual execution lanes require categorical diagnostics"
        )
    return (selection.replace("-", "_"),)


def diagnostic_execution_shape(
    spec: CampaignSpec,
    *,
    train_prefix_groups: int | None,
    skip_evaluation: bool,
    categorical: bool,
) -> tuple[int, bool]:
    """Resolve a bounded replay only after preserving the full source shape."""
    executed_train_groups = (
        spec.train_groups if train_prefix_groups is None else train_prefix_groups
    )
    if (
        isinstance(executed_train_groups, bool)
        or not isinstance(executed_train_groups, int)
        or not 2 <= executed_train_groups <= spec.train_groups
    ):
        raise MultiTokenParityError(
            "training prefix must contain between two groups and the full source horizon"
        )
    if executed_train_groups != spec.train_groups and not categorical:
        raise MultiTokenParityError("training-prefix replay is diagnostic-only")
    if skip_evaluation and not categorical:
        raise MultiTokenParityError("skipping evaluation is diagnostic-only")
    return executed_train_groups, skip_evaluation


def run(args: argparse.Namespace) -> Mapping[str, Any]:
    run_started = time.monotonic()

    def progress(stage: str, **details: Any) -> None:
        # Flush before expensive work so an externally stopped process leaves
        # its last stage in stderr. These events are not completed result artifacts.
        print(
            json.dumps(
                {
                    "event": "gemma4_grpo_mlx_progress",
                    "stage": stage,
                    "elapsed_seconds": time.monotonic() - run_started,
                    **details,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )

    progress("input-validation")
    lanes = execution_lanes(
        args.execution_lane, categorical=args.categorical_diagnostic
    )
    trace_adapter_output = getattr(args, "trace_adapter_output", None)
    if trace_adapter_output is not None and "trace_replay" not in lanes:
        raise MultiTokenParityError(
            "trace adapter output requires the trace-replay execution lane"
        )
    spec = CampaignSpec(
        model_key=args.model_key,
        train_groups=args.train_groups,
        eval_groups=args.eval_groups,
        group_size=args.group_size,
        max_completion_tokens=args.max_completion_tokens,
        recipe_profile=args.recipe_profile,
    )
    spec.validate()
    validate_completion_execution(args.completion_execution, spec.max_completion_tokens)
    compact_ple = bool(getattr(args, "compact_frozen_ple", False))
    if compact_ple and (
        not args.categorical_diagnostic
        or args.execution_lane != "trace-replay"
        or spec.max_completion_tokens != 1
        or args.activation_mode != "aligned-f32"
    ):
        raise MultiTokenParityError(
            "frozen PLE compaction requires aligned-F32 single-token diagnostic replay"
        )
    skip_evaluation = bool(getattr(args, "skip_evaluation", False))
    executed_train_groups, skip_evaluation = diagnostic_execution_shape(
        spec,
        train_prefix_groups=getattr(args, "train_prefix_groups", None),
        skip_evaluation=skip_evaluation,
        categorical=args.categorical_diagnostic,
    )
    if (
        spec.recipe_profile in ALL_LINEAR_SINGLE_TOKEN_PROFILES
        and not args.categorical_diagnostic
    ):
        raise MultiTokenParityError(
            "all-linear single-token profile requires categorical diagnostics"
        )
    if args.activation_mode == "aligned-f32" and not args.categorical_diagnostic:
        raise MultiTokenParityError(
            "aligned F32 activations require categorical diagnostics"
        )
    if args.capture_initial_training_logits and not args.categorical_diagnostic:
        raise MultiTokenParityError(
            "predictor capture requires categorical diagnostics"
        )
    if args.shared_single_token_scoring and spec.max_completion_tokens != 1:
        raise MultiTokenParityError(
            "shared single-token scoring requires a one-token completion budget"
        )
    model_dir = args.model_dir.expanduser().resolve()
    adapter_dir = args.adapter_dir.expanduser().resolve()
    manifest = load_campaign_materialization(
        args.dataset_manifest, args.evaluation_dataset_manifest, spec, model_dir
    )
    acceptance = load_acceptance(
        args.antfly_run_root,
        manifest,
        spec,
        model_dir,
        adapter_dir,
        categorical_diagnostic=args.categorical_diagnostic,
    )
    controller_unit = getattr(args, "adaptive_kl_controller_unit", "source-contract")
    if controller_unit != "source-contract" and not args.categorical_diagnostic:
        raise MultiTokenParityError(
            "adaptive KL controller overrides require categorical diagnostics"
        )
    controller_uses_completion_episode_horizon = (
        controller_unit == "completion-episodes"
        or acceptance.kl_trace_schema_version
        in {
            "antfly_inference_grpo_kl_control_trace/v4",
            KL_TRACE_SCHEMA_VERSION,
        }
    )
    controller_observations_per_group = (
        spec.group_size if controller_uses_completion_episode_horizon else 1
    )
    controller_advances_rejections = (
        controller_unit == "completion-episodes"
        or acceptance.kl_trace_schema_version
        in {
            "antfly_inference_grpo_kl_control_trace/v3",
            "antfly_inference_grpo_kl_control_trace/v4",
            KL_TRACE_SCHEMA_VERSION,
        }
    )
    sampling_contract = (
        categorical_contract(
            acceptance.config,
            acceptance.train_report,
            acceptance.eval_report,
        )
        if args.categorical_diagnostic
        else None
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
        raise MultiTokenParityError(
            "MLX campaign must run on the locked Apple platform"
        )
    runtime_root = args.mlx_runtime_root.expanduser().resolve()
    progress("runtime-attestation")
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
    mlx_lm_source_attestation = None
    if args.mlx_lm_source_archive is not None:
        mlx_lm_source_attestation = attest_mlx_lm_archive(
            args.mlx_lm_source_root,
            args.mlx_lm_source_archive,
            mlx_contract["source_revisions"]["mlx-lm"],
        )
        mlx_lm_revision = mlx_lm_source_attestation["revision"]
        # Preserve the exact inventory and never import stale generated bytecode.
        sys.dont_write_bytecode = True
    else:
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
        (
            "MLX-LM LoRA",
            Path(sys.modules[LoRALinear.__module__].__file__ or "").resolve(),
        ),
    ):
        if not microbenchmark._path_is_within(source_path, mlx_lm_root):
            raise MultiTokenParityError(
                f"imported {label} escaped the attested checkout"
            )

    progress("dataset-validation")
    tokenizer = tokenizers_native.Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    train_rows = load_rows(
        acceptance.train_dataset_path,
        expected_count=spec.train_groups,
        expected_ids=acceptance.train_source_ids,
        expected_indices=acceptance.train_source_row_indices,
        tokenizer=tokenizer,
        max_completion_tokens=spec.max_completion_tokens,
        sequence_length=spec.profile.sequence_length,
    )
    eval_rows = load_rows(
        Path(str(manifest["eval_jsonl"])),
        expected_count=spec.eval_groups,
        expected_ids=manifest["eval_source_ids"],
        expected_indices=manifest["eval_source_row_indices"],
        tokenizer=tokenizer,
        max_completion_tokens=spec.max_completion_tokens,
        sequence_length=spec.profile.sequence_length,
    )
    train_rows = legacy.rows_in_prompt_order(
        train_rows, [group.prompt_index for group in acceptance.train_trace]
    )
    eval_rows = legacy.rows_in_prompt_order(
        eval_rows, [group.prompt_index for group in acceptance.eval_trace]
    )
    validate_trace_rewards(tokenizer, train_rows, acceptance.train_trace)
    validate_trace_rewards(tokenizer, eval_rows, acceptance.eval_trace)
    execution_train_rows = train_rows[:executed_train_groups]
    execution_train_trace = acceptance.train_trace[:executed_train_groups]

    adapter_manifest = load_json(
        adapter_dir / "antfly_finetune_manifest.json", "seed adapter manifest"
    )
    binding_fields = ("base_model_sha256", "tokenizer_sha256", "chat_template_sha256")
    prepared_summary = {key: adapter_manifest.get(key) for key in binding_fields}
    progress("model-provenance")
    base_model_provenance = locked.zig_model_provenance(model_dir)
    if prepared_summary != base_model_provenance:
        raise MultiTokenParityError("seed adapter does not match the model")
    seed_adapter = locked.inspect_initial_adapter(
        adapter_dir,
        lock,
        spec.model_key,
        spec.profile.target_preset,
        prepared_summary,
        allow_missing_manifest_target_preset=True,
    )
    antfly_trained = (
        locked.inspect_initial_adapter(
            acceptance.trained_adapter_dir,
            lock,
            spec.model_key,
            spec.profile.target_preset,
            prepared_summary,
            allow_missing_manifest_target_preset=True,
        )
        if acceptance.trained_adapter_dir is not None
        else None
    )

    mx.set_default_device(mx.gpu)
    # Bound only reusable free buffers; live model/activation allocations still
    # obey MLX's memory policy and are measured by the process guard.
    if args.mlx_cache_limit_mib is not None:
        mx.set_cache_limit(args.mlx_cache_limit_mib * 1024 * 1024)
    mx.random.seed(42)
    sampler = locked.DarwinProcessMemorySampler()
    sampler.start()
    sampler_active = True
    campaign_started = time.perf_counter()
    try:
        load_started = time.perf_counter()
        progress("model-load")
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
        model.freeze()
        base_inventory = locked.require_bf16_base_model(model, mx)
        frozen_ple_cache = None
        captured_ple_tokens: frozenset[int] | None = None
        if compact_ple:
            text_model = model.language_model.model
            if not text_model.hidden_size_per_layer_input:
                raise MultiTokenParityError("model has no frozen PLE embedding table")
            captured = {0}
            for row in (*train_rows, *eval_rows):
                captured.update(row.prompt_token_ids)
            for group in (*acceptance.train_trace, *acceptance.eval_trace):
                for sequence in group.sequences:
                    captured.update(sequence)
            captured_ple_tokens = frozenset(captured)
            replacement, frozen_ple_cache = compact_frozen_ple_embedding(
                mx, nn, text_model.embed_tokens_per_layer, sorted(captured)
            )
            text_model.embed_tokens_per_layer = replacement
            # The trainable per-layer projection remains in the normal graph.
            mx.clear_cache()
            progress("frozen-ple-compacted", retained_rows=len(captured))
        progress("model-materialization")
        mx.eval(model.parameters())
        mx.synchronize()
        progress("adapter-installation")
        targets = locked.target_module_names(
            model, lock, spec.model_key, spec.profile.target_preset
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
        checkpoint_source_sha256 = None
        if args.gradient_checkpointing:
            from mlx_lm.tuner import trainer as checkpoint_source

            checkpoint_path = Path(checkpoint_source.__file__ or "").resolve()
            if not microbenchmark._path_is_within(checkpoint_path, mlx_lm_root):
                raise MultiTokenParityError(
                    "MLX-LM checkpoint helper escaped the attested source"
                )
            checkpoint_source.grad_checkpoint(model.language_model.model.layers[0])
            checkpoint_source_sha256 = sha256_file(checkpoint_path)
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
        progress("model-ready")

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
            if captured_ple_tokens is not None:
                require_captured_tokens(joined, captured_ple_tokens)
            if len(joined) > spec.profile.sequence_length:
                raise MultiTokenParityError("completion exceeds the sequence contract")
            return (
                mx.array(
                    [joined + [0] * (spec.profile.sequence_length - len(joined))],
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

        def forward(current_model: Any, tokens: Any) -> Any:
            if args.activation_mode == "stock-bf16":
                return current_model(tokens)
            # Same explicit input staging used by the retained F32 numerical
            # comparisons. Frozen checkpoint weights remain BF16.
            text_model = current_model.language_model.model
            embeddings = text_model.embed_tokens(tokens).astype(mx.float32)
            per_layer = (
                text_model._get_per_layer_inputs(tokens, embeddings).astype(mx.float32)
                if text_model.hidden_size_per_layer_input
                else None
            )
            return current_model(
                tokens, input_embeddings=embeddings, per_layer_inputs=per_layer
            )

        policy_temperature = (
            sampling_contract[1].temperature if sampling_contract is not None else 1.0
        )

        def policy_logprobs(logits: Any) -> Any:
            tempered = logits / policy_temperature
            return tempered - mx.logsumexp(tempered, axis=-1, keepdims=True)

        def selected_logps(
            current_model: Any,
            tokens: Any,
            selected: Any,
            mask: Any,
            prompt_length: int,
        ) -> Any:
            logits = forward(current_model, tokens).astype(mx.float32)
            columns = []
            for step in range(spec.max_completion_tokens):
                predictor = logits[:, prompt_length - 1 + step, :]
                logprobs = policy_logprobs(predictor)
                token_logps = mx.take_along_axis(
                    logprobs, selected[:, step : step + 1], axis=-1
                )
                columns.append(token_logps)
            return mx.concatenate(columns, axis=1) * mask

        def score_sequences(
            row: BoolQRow, sequences: Sequence[Sequence[int]]
        ) -> list[list[float]]:
            if args.shared_single_token_scoring and sequences:
                if any(len(sequence) != 1 for sequence in sequences):
                    raise MultiTokenParityError(
                        "shared scoring received a multi-token completion"
                    )
                # Every single-token completion uses the same causal predictor
                # row. Preserve the physical batch=1/padding contract and gather
                # all selected log probabilities from one forward pass.
                tokens, _selected, _mask = padded_sequence(row, sequences[0])
                logits = forward(model, tokens).astype(mx.float32)[
                    0, len(row.prompt_token_ids) - 1, :
                ]
                logprobs = policy_logprobs(logits)
                selected = mx.array(
                    [sequence[0] for sequence in sequences], dtype=mx.int32
                )
                values = logprobs[selected]
                mx.eval(values)
                mx.synchronize()
                return [[float(value)] for value in values.tolist()]
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
                [prompt + [0] * (spec.profile.sequence_length - len(prompt))],
                dtype=mx.int32,
            )
            logits = forward(model, prompt_batch).astype(mx.float32)[
                0, len(prompt) - 1, :
            ]
            first_tokens = mx.array(
                legacy.ranked_tokens(logits.tolist(), spec.group_size), dtype=mx.int32
            )
            first_logprobs = policy_logprobs(logits)
            first_values = first_logprobs[first_tokens]
            mx.eval(first_tokens, first_values)
            mx.synchronize()
            sequences = [[int(value)] for value in first_tokens.tolist()]
            logps = [[float(value)] for value in first_values.tolist()]
            active = [sequence[0] != eos_token_id for sequence in sequences]
            for _step in range(1, spec.max_completion_tokens):
                active_indices = [
                    index for index, enabled in enumerate(active) if enabled
                ]
                if not active_indices:
                    break
                row_index = len(prompt) + _step - 1
                for completion_index in active_indices:
                    joined = prompt + sequences[completion_index]
                    tokens = mx.array(
                        [joined + [0] * (spec.profile.sequence_length - len(joined))],
                        dtype=mx.int32,
                    )
                    predictor = forward(model, tokens).astype(mx.float32)[
                        0, row_index, :
                    ]
                    ranked = legacy.ranked_tokens(predictor.tolist(), spec.group_size)
                    chosen = mx.array(
                        ranked[completion_index % spec.group_size], dtype=mx.int32
                    )
                    chosen_logp = policy_logprobs(predictor)[chosen]
                    mx.eval(chosen, chosen_logp)
                    mx.synchronize()
                    token_id = int(chosen.item())
                    sequences[completion_index].append(token_id)
                    logps[completion_index].append(float(chosen_logp.item()))
                    if token_id == eos_token_id:
                        active[completion_index] = False
            return sequences, logps

        def rollout_group(
            row: BoolQRow,
            prompt_index: int,
            *,
            evaluation: bool = False,
            initial_prediction: dict[str, Any] | None = None,
        ):
            if sampling_contract is None:
                return ranked_group(row)
            seed, policy = sampling_contract

            def predict(prefix: Sequence[int]) -> list[float]:
                if len(prefix) >= spec.profile.sequence_length:
                    raise MultiTokenParityError(
                        "categorical prefix exceeds sequence contract"
                    )
                tokens = mx.array(
                    [list(prefix) + [0] * (spec.profile.sequence_length - len(prefix))],
                    dtype=mx.int32,
                )
                logits = forward(model, tokens).astype(mx.float32)[
                    0, len(prefix) - 1, :
                ]
                mx.eval(logits)
                mx.synchronize()
                values = logits.tolist()
                if initial_prediction is not None and not initial_prediction:
                    initial_prediction.update(
                        source_id=row.source_id,
                        prompt_index=prompt_index,
                        prompt_token_ids=list(prefix),
                        physical_sequence_length=spec.profile.sequence_length,
                        predictor_position=len(prefix) - 1,
                        logits=values,
                    )
                return values

            return categorical_rollout_group(
                predict,
                row.prompt_token_ids,
                run_seed=seed,
                epoch=0,
                prompt_index=prompt_index,
                evaluation=evaluation,
                policy=policy,
                group_size=spec.group_size,
                max_completion_tokens=spec.max_completion_tokens,
                eos_token_id=eos_token_id,
            )

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
            advantages = normalized_advantages(rewards, spec.profile.advantage_epsilon)
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
                learning_rate=spec.profile.learning_rate,
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
            logits = forward(current_model, tokens).astype(mx.float32)
            predictor_rows = (
                prompt_length
                + mx.arange(spec.max_completion_tokens, dtype=mx.int32)
                - 1
            )
            predictors = logits[0, predictor_rows, :]
            logprobs = policy_logprobs(predictors)
            new_logps = mx.take_along_axis(logprobs, selected[:, None], axis=-1)[:, 0]
            return grpo_loss_terms(
                mx,
                new_logps,
                old_logps,
                reference_logps,
                advantage,
                mask,
                kl_coef,
                group_token_count,
            )

        def single_token_group_loss(
            current_model: Any,
            tokens: Any,
            selected: Any,
            mask: Any,
            old_logps: Any,
            reference_logps: Any,
            advantages: Any,
            kl_coef: Any,
            prompt_length: Any,
        ) -> tuple[Any, Any, Any, Any, Any, Any]:
            # One causal predictor serves every completion in this group. The
            # terminal token cannot affect its preceding predictor position.
            logits = forward(current_model, tokens[:1]).astype(mx.float32)
            logprobs = policy_logprobs(logits[0, prompt_length - 1, :])
            new_logps = logprobs[selected[:, 0]]
            return grpo_loss_terms(
                mx,
                new_logps,
                old_logps[:, 0],
                reference_logps[:, 0],
                advantages,
                mask[:, 0],
                kl_coef,
                mx.sum(mask),
            )

        group_loss_and_grad = nn.value_and_grad(model, single_token_group_loss)

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
        progress("reference-precompute")
        reset_to_initial()
        trace_reference = []
        for group_index, (row, group) in enumerate(
            zip(execution_train_rows, execution_train_trace)
        ):
            progress("reference-group", group_index=group_index)
            trace_reference.append(score_sequences(row, group.sequences))
        reference_precompute_seconds = time.perf_counter() - reference_started

        def train_lane(
            mode: str,
        ) -> tuple[Mapping[str, Any], Mapping[str, Any] | None]:
            progress("training-start", lane=mode)
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
                if args.completion_execution.startswith("coalesced-single-token"):
                    metrics, gradients = group_loss_and_grad(
                        model,
                        tokens,
                        selected,
                        mask,
                        old_logps,
                        reference_logps,
                        advantages,
                        kl_coef,
                        prompt_length,
                    )
                    metric_totals = list(metrics[:5])
                    rescored = metrics[5][:, None]
                else:
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
                        if args.completion_execution == "sequential":
                            # Bound the live backward graph to one completion. Keep
                            # the same ordered sum and update only after the group.
                            mx.eval(gradients, completion_metrics)
                    metric_totals = list(completion_metrics[0][:5])
                    for metrics in completion_metrics[1:]:
                        for metric_index in range(5):
                            metric_totals[metric_index] = (
                                metric_totals[metric_index] + metrics[metric_index]
                            )
                    rescored = mx.stack(
                        [metrics[5] for metrics in completion_metrics], axis=0
                    )
                assert gradients is not None
                gradients, grad_norm = optim.clip_grad_norm(
                    gradients, OPTIMIZER["max_grad_norm"]
                )
                optimizer.update(model, gradients)

                return (
                    *metric_totals,
                    rescored,
                    grad_norm,
                )

            compiled_step = (
                step
                if args.completion_execution
                in {"sequential", "coalesced-single-token-eager"}
                else mx.compile(step, inputs=state, outputs=state)
            )
            coefficient = GRPO["initial_kl_coef"]
            updates: list[dict[str, Any]] = []
            skipped: list[dict[str, Any]] = []
            initial_prediction: dict[str, Any] | None = (
                {} if args.capture_initial_training_logits else None
            )
            started_lane = time.perf_counter()
            for update_index, (row, expected) in enumerate(
                zip(execution_train_rows, execution_train_trace)
            ):
                started = time.perf_counter()
                progress("training-rollout", lane=mode, group_index=update_index)
                native_sequences, native_old = rollout_group(
                    row,
                    expected.prompt_index,
                    initial_prediction=initial_prediction
                    if update_index == 0
                    else None,
                )
                overlap = sequence_overlap(
                    native_sequences,
                    expected.sequences,
                    with_replacement=sampling_contract is not None,
                )
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
                raw_mean_kl = mean_k3(flatten(policy_before), flatten(reference_values))
                admission = group_admission(rewards, raw_mean_kl)
                if sampling_contract is not None and admission != "admitted":
                    next_coefficient = coefficient
                    if (
                        admission == "budget-exceeded-skipped"
                        and controller_advances_rejections
                    ):
                        next_coefficient = adaptive_kl_update(
                            coefficient,
                            raw_mean_kl,
                            controller_observations_per_group,
                            min_kl_coef=spec.profile.min_kl_coef,
                            max_kl_coef=spec.profile.max_kl_coef,
                        )
                    progress(
                        "training-skipped",
                        lane=mode,
                        group_index=update_index,
                        reason=admission,
                        optimizer_steps=len(updates),
                    )
                    skipped.append(
                        {
                            "group_index": update_index,
                            "prompt_index": expected.prompt_index,
                            "source_id": row.source_id,
                            "optimizer_steps_before": len(updates),
                            "status": admission,
                            "completion_token_ids": sequences,
                            "completion_tokens": sum(map(len, sequences)),
                            "rewards": rewards,
                            "mean_kl": raw_mean_kl,
                            "kl_coef_before": coefficient,
                            "kl_coef_after": next_coefficient,
                            "sampling_rescore_max_abs_error": sampling_rescore_max_abs_error,
                            "candidate_overlap_with_antfly": overlap,
                            "seconds": time.perf_counter() - started,
                        }
                    )
                    coefficient = next_coefficient
                    continue
                if sampling_contract is None and raw_mean_kl > GRPO["train_max_kl"]:
                    raise MultiTokenParityError(
                        f"MLX {mode} group {update_index} exceeded the pre-update KL budget"
                    )
                next_coefficient = adaptive_kl_update(
                    coefficient,
                    raw_mean_kl,
                    controller_observations_per_group,
                    min_kl_coef=spec.profile.min_kl_coef,
                    max_kl_coef=spec.profile.max_kl_coef,
                )
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
                    normalized_advantages(rewards, spec.profile.advantage_epsilon),
                    dtype=mx.float32,
                )
                progress(
                    "training-update",
                    lane=mode,
                    group_index=update_index,
                    optimizer_steps=len(updates),
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
                (
                    loss,
                    pg_loss,
                    kl_loss,
                    observed_kl,
                    clip_fraction,
                    rescored,
                    grad_norm,
                ) = outputs
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
                    raise MultiTokenParityError(
                        "MLX preflight and differentiable KL disagree"
                    )
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
                differentiable_rescore_max_abs_error = max(rescore_errors, default=0.0)
                if differentiable_rescore_max_abs_error > 1.0e-4:
                    raise MultiTokenParityError(
                        f"MLX {mode} group {update_index} differentiable rescore "
                        f"drifted by {differentiable_rescore_max_abs_error:.6g}"
                    )
                updates.append(
                    {
                        "update_index": len(updates),
                        "group_index": update_index,
                        "prompt_index": expected.prompt_index,
                        "status": "admitted",
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
                progress(
                    "training-updated",
                    lane=mode,
                    group_index=update_index,
                    optimizer_steps=len(updates),
                )
            lane_seconds = time.perf_counter() - started_lane
            groups = sorted(updates + skipped, key=lambda row: row["group_index"])
            adapter_comparison: Mapping[str, Any] | None = None
            adapter_comparison_scope = (
                "matched-full-horizon"
                if executed_train_groups == spec.train_groups
                else "unavailable-antfly-adapter-is-full-horizon"
            )
            if (
                updates
                and antfly_trained is not None
                and executed_train_groups == spec.train_groups
            ):
                adapter_comparison = legacy._adapter_delta_comparison(
                    model=model,
                    initial_trainables=initial_trainables,
                    antfly_trained=antfly_trained,
                    target_names=targets,
                    mx=mx,
                    tree_flatten=tree_flatten,
                )
            adapter_output = None
            if mode == "trace_replay" and trace_adapter_output is not None:
                adapter_output = write_adapter_exclusive(
                    trace_adapter_output,
                    final_trainables=snapshot_trainables(),
                    target_names=targets,
                    adapter=seed_adapter,
                    mx=mx,
                )
            return (
                {
                    "mode": mode,
                    "initial_prediction": initial_prediction,
                    "groups": len(groups),
                    "optimizer_steps": len(updates),
                    "zero_reward_std_groups": sum(
                        row["status"] == "zero-reward-std-skipped" for row in skipped
                    ),
                    "kl_rejected_groups": sum(
                        row["status"] == "budget-exceeded-skipped" for row in skipped
                    ),
                    "adapter_delta_comparison_scope": adapter_comparison_scope,
                    "adapter_output": adapter_output,
                    "skipped_groups": skipped,
                    "seconds": lane_seconds,
                    "median_update_seconds": statistics.median(
                        row["seconds"] for row in updates
                    )
                    if updates
                    else None,
                    "mean_update_seconds": statistics.mean(
                        row["seconds"] for row in updates
                    )
                    if updates
                    else None,
                    "completion_tokens": sum(
                        int(row["completion_tokens"]) for row in groups
                    ),
                    "mean_reward": statistics.mean(
                        reward for row in groups for reward in row["rewards"]
                    ),
                    "mean_loss": statistics.mean(row["loss"] for row in updates)
                    if updates
                    else None,
                    "mean_kl_loss": statistics.mean(row["kl_loss"] for row in updates)
                    if updates
                    else None,
                    "mean_kl": statistics.mean(row["mean_kl"] for row in updates)
                    if updates
                    else None,
                    "final_kl_coef": coefficient,
                    "max_mean_kl": max(row["mean_kl"] for row in groups),
                    "candidate_overlap_with_antfly": summarize_overlaps(
                        [row["candidate_overlap_with_antfly"] for row in groups]
                    ),
                    "updates": updates,
                },
                adapter_comparison,
            )

        def evaluate_lane(lane: str) -> Mapping[str, Any]:
            records: list[dict[str, Any]] = []
            started = time.perf_counter()
            for group_index, (row, expected) in enumerate(
                zip(eval_rows, acceptance.eval_trace)
            ):
                progress("evaluation-group", lane=lane, group_index=group_index)
                sequences, sampling_logps = rollout_group(
                    row, expected.prompt_index, evaluation=True
                )
                policy_logps = score_sequences(row, sequences)
                reference_logps = reference_score(row, sequences)
                decoded = [
                    decode_reward(tokenizer, values, row.target) for values in sequences
                ]
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
                            sequences,
                            expected.sequences,
                            with_replacement=sampling_contract is not None,
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
        baseline_evaluation = None if skip_evaluation else evaluate_lane("baseline")
        trace_training = trace_adapter = trace_evaluation = None
        native_training = native_evaluation = native_adapter = None
        if "trace_replay" in lanes:
            trace_training, trace_adapter = train_lane("trace_replay")
            if not skip_evaluation:
                trace_evaluation = evaluate_lane("trace_replay")
        if "native_rollout" in lanes:
            native_training, native_adapter = train_lane("native_rollout")
            if not skip_evaluation:
                native_evaluation = evaluate_lane("native_rollout")
        progress("model-work-complete")

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
    full_training_horizon = executed_train_groups == spec.train_groups
    performance = {
        "antfly_source_full_train_accounted_seconds": antfly_train_seconds,
        "antfly_train_accounted_seconds": (
            antfly_train_seconds if full_training_horizon else None
        ),
        "mlx_native_train_seconds": native_training["seconds"]
        if native_training
        else None,
        "antfly_to_mlx_train_time_ratio": (
            antfly_train_seconds / native_training["seconds"]
            if native_training and native_training["seconds"] > 0.0
            else None
        ),
        "antfly_eval_loop_seconds": antfly_eval_seconds,
        "mlx_native_eval_seconds": native_evaluation["seconds"]
        if native_evaluation
        else None,
        "antfly_to_mlx_eval_time_ratio": (
            antfly_eval_seconds / native_evaluation["seconds"]
            if native_evaluation and native_evaluation["seconds"] > 0.0
            else None
        ),
    }
    evaluation_deltas = (
        {
            key: native_evaluation[key] - antfly_eval[key]
            for key in (
                "mean_reward",
                "top_rank_mean_reward",
                "positive_reward_group_rate",
                "kl_loss",
                "mean_kl",
            )
        }
        if native_evaluation
        else None
    )
    minimums = acceptance.eval_report.get("minimums")
    if not isinstance(minimums, dict):
        raise MultiTokenParityError("Antfly evaluation minimums are missing")
    native_passed = (
        (
            native_evaluation["mean_reward"] >= minimums["mean_reward"]
            and native_evaluation["top_rank_mean_reward"]
            >= minimums["top_rank_mean_reward"]
            and native_evaluation["positive_reward_group_rate"]
            >= minimums["positive_reward_group_rate"]
            and native_evaluation["kl_loss"] <= minimums["max_kl_loss"]
        )
        if native_evaluation
        else None
    )
    trace_numerical_close = (
        (
            all(
                legacy.adapter_update_checks(
                    trace_adapter, min_cosine=0.95, max_relative_error=0.1
                ).values()
            )
            if trace_adapter is not None
            else None
        )
        if trace_training
        else None
    )
    native_behavior_close = (
        (
            abs(evaluation_deltas["mean_reward"]) <= 1.0 / spec.group_size
            and abs(evaluation_deltas["top_rank_mean_reward"]) <= 1.0 / spec.eval_groups
            and native_passed
        )
        if evaluation_deltas
        else None
    )
    classification = campaign_classification(
        bool(trace_numerical_close),
        bool(native_behavior_close),
        categorical=args.categorical_diagnostic,
    )
    return {
        "schema_version": (
            "antfly_gemma4_grpo_boolq_mlx_multitoken/v2"
            if args.categorical_diagnostic
            else RESULT_SCHEMA_VERSION
        ),
        "status": (
            "diagnostic-lane-completed"
            if len(lanes) == 1
            else "diagnostic-completed"
            if args.categorical_diagnostic
            else "completed"
        ),
        "scope": (
            f"real-pinned-boolq-{spec.model_key.lower()}-"
            f"{executed_train_groups}of{spec.train_groups}x{spec.eval_groups}-group{spec.group_size}-"
            f"max{spec.max_completion_tokens}-adaptive-kl"
        )
        + (f"-{args.execution_lane}" if len(lanes) == 1 else ""),
        "classification": classification,
        "claim_boundary": {
            "categorical_statistical_parity": False,
            "production_acceptance": False,
            "broad_grpo_performance_parity": False,
            "long_horizon_quality_parity": False,
            "reason": (
                "This is one seeded BoolQ campaign with a bounded update horizon; "
                "it validates mechanics and matched local behavior only."
            ),
        },
        "contract": {
            "model_key": spec.model_key,
            "target_preset": spec.profile.target_preset,
            "sequence_length": spec.profile.sequence_length,
            "train_groups": spec.train_groups,
            "executed_train_groups": executed_train_groups,
            "eval_groups": spec.eval_groups,
            "evaluation_executed": not skip_evaluation,
            "group_size": spec.group_size,
            "max_completion_tokens": spec.max_completion_tokens,
            "learning_rate": spec.profile.learning_rate,
            "optimizer": OPTIMIZER,
            "grpo": {
                **GRPO,
                "advantage_epsilon": spec.profile.advantage_epsilon,
                "min_kl_coef": spec.profile.min_kl_coef,
                "max_kl_coef": spec.profile.max_kl_coef,
            },
            "recipe_profile": spec.recipe_profile,
            "execution_lane": args.execution_lane,
            "capture_initial_training_logits": args.capture_initial_training_logits,
            "completion_execution": args.completion_execution,
            "mlx_cache_limit_mib": args.mlx_cache_limit_mib,
            "frozen_ple_cache": frozen_ple_cache,
            "activation_mode": args.activation_mode,
            "single_token_scoring": "shared-prompt"
            if args.shared_single_token_scoring
            else "per-completion",
            "gradient_checkpointing": {
                "enabled": args.gradient_checkpointing,
                "source_sha256": checkpoint_source_sha256,
            },
            "kl_trace_schema_version": acceptance.kl_trace_schema_version,
            "adaptive_kl_controller_unit": controller_unit,
            "adaptive_kl_controller_observations_per_group": controller_observations_per_group,
            "adaptive_kl_observation_policy": (
                "all-kl-observed-groups-completion-episode-horizon"
                if controller_uses_completion_episode_horizon
                else (
                    "all-kl-observed-groups-group-horizon"
                    if acceptance.kl_trace_schema_version
                    == "antfly_inference_grpo_kl_control_trace/v3"
                    else "admitted-groups"
                )
            ),
            "reward_mode": "prefix-match",
            "reference_mode": "frozen-initial-adapter-snapshot",
            "policy_scoring": "temperature-scaled-full-vocabulary/v1",
            "rollout_mode": (
                "seeded-categorical-diagnostic"
                if sampling_contract
                else "deterministic-rank-per-completion-each-token"
            ),
            "sampling": acceptance.train_report.get("sampling"),
            "sampling_seed": sampling_contract[0] if sampling_contract else None,
            "sampling_helper_sha256": sha256_file(
                SCRIPT_DIR / "gemma4_grpo_sampling.py"
            ),
            "loss_normalization": "mean-over-all-unmasked-completion-tokens",
        },
        "dataset": {
            "repo_id": manifest["dataset"]["repo_id"],
            "revision": manifest["dataset"]["revision"],
            "manifest_path": str(args.dataset_manifest.expanduser().resolve()),
            "manifest_sha256": sha256_file(
                args.dataset_manifest.expanduser().resolve()
            ),
            "separate_manifest_bindings": manifest.get("campaign_manifest_bindings"),
            "train_jsonl_sha256": manifest["dataset"]["train"][
                "materialized_jsonl_sha256"
            ],
            "executed_train_jsonl_path": str(acceptance.train_dataset_path),
            "executed_train_jsonl_sha256": sha256_file(acceptance.train_dataset_path),
            "eval_jsonl_sha256": manifest["dataset"]["evaluation"][
                "materialized_jsonl_sha256"
            ],
        },
        "antfly": {
            "run_root": str(acceptance.root),
            "grpo_report_sha256": sha256_file(acceptance.root / "grpo_report.json"),
            "evaluation_report_path": str(acceptance.eval_report_path),
            "evaluation_report_sha256": sha256_file(acceptance.eval_report_path),
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
            )
            if acceptance.trained_adapter_dir is not None
            else None,
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
            }
            if trace_training
            else None,
            "native_rollout": {
                "training": native_training,
                "evaluation": native_evaluation,
                "adapter_delta_comparison_with_antfly": native_adapter,
                "evaluation_delta_from_antfly": evaluation_deltas,
                "passed_antfly_quality_minimums": native_passed,
            }
            if native_training
            else None,
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
            "mlx_lm_source_attestation": mlx_lm_source_attestation,
            "package_versions": actual_versions,
            "tokenizers_version": tokenizers_version,
            "python_version": actual_python,
            "mlx_core_path": str(core_path),
        },
        "parity_assessment": {
            "classification": classification,
            "trace_numerical_close": trace_numerical_close,
            "native_behavior_close": (
                native_behavior_close and not args.categorical_diagnostic
                if native_training
                else None
            ),
            "single_seed_behavior_checks": native_behavior_close,
            "native_quality_gate": native_passed,
        },
        "base_model_provenance": base_model_provenance,
        "runner_sha256": "sha256:"
        + hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest(),
    }


def nonnegative_mib(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("cache limit must be nonnegative")
    return parsed


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model-key", choices=MODEL_KEYS, required=True)
    result.add_argument("--model-dir", type=Path, required=True)
    result.add_argument("--adapter-dir", type=Path, required=True)
    result.add_argument("--dataset-manifest", type=Path, required=True)
    result.add_argument(
        "--evaluation-dataset-manifest",
        type=Path,
        help="Separately attested evaluation materialization; preserves the original training selection",
    )
    result.add_argument("--antfly-run-root", type=Path, required=True)
    result.add_argument("--train-groups", type=int, required=True)
    result.add_argument(
        "--train-prefix-groups",
        type=int,
        help="Execute only this chronological prefix after validating the full source campaign; diagnostic-only",
    )
    result.add_argument("--eval-groups", type=int, required=True)
    result.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip MLX baseline/final evaluation for a focused training replay; diagnostic-only",
    )
    result.add_argument(
        "--recipe-profile",
        choices=tuple(RECIPE_PROFILES),
        default="qv-multitoken",
        help="Pinned comparison recipe; all-linear single-token requires group 16, max 1 and categorical diagnostics",
    )
    result.add_argument("--group-size", type=int, default=4)
    result.add_argument("--max-completion-tokens", type=int, default=4)
    result.add_argument(
        "--completion-execution",
        choices=(
            "compiled-group",
            "sequential",
            "coalesced-single-token",
            "coalesced-single-token-eager",
        ),
        default="compiled-group",
        help="Sequential bounds completion graphs; coalesced modes differentiate one shared predictor, with optional whole-step compilation",
    )
    result.add_argument(
        "--execution-lane",
        choices=("both", "trace-replay", "native-rollout"),
        default="both",
        help="Run independent diagnostic lanes in separate guarded processes",
    )
    result.add_argument(
        "--capture-initial-training-logits",
        action="store_true",
        help="Retain the first training predictor row per executed lane; categorical diagnostics only",
    )
    result.add_argument(
        "--activation-mode",
        choices=("stock-bf16", "aligned-f32"),
        default="stock-bf16",
        help="Aligned F32 stages text/per-layer embeddings in F32 with frozen BF16 weights; diagnostic only",
    )
    result.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Use the pinned MLX-LM layer checkpoint helper to bound backward activation memory",
    )
    result.add_argument(
        "--shared-single-token-scoring",
        action="store_true",
        help="Score one-token completions from their shared causal predictor row at the same physical shape",
    )
    result.add_argument(
        "--compact-frozen-ple",
        action="store_true",
        help="Cache exact frozen PLE rows for bounded single-token trace replay; excludes performance qualification",
    )
    result.add_argument(
        "--mlx-cache-limit-mib",
        type=nonnegative_mib,
        help="Bound MLX reusable free-buffer cache; omitted preserves the runtime default",
    )
    result.add_argument("--mlx-runtime-root", type=Path, required=True)
    result.add_argument("--mlx-wheel", type=Path, required=True)
    result.add_argument("--mlx-metal-wheel", type=Path, required=True)
    result.add_argument("--mlx-lm-source-root", type=Path, required=True)
    result.add_argument(
        "--mlx-lm-source-archive",
        type=Path,
        help="Use a clean extraction of the pinned upstream archive instead of a Git checkout",
    )
    result.add_argument("--lock", type=Path, default=locked.LOCK_PATH)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument(
        "--trace-adapter-output",
        type=Path,
        help="Exclusively persist the final trace-replay adapter as canonical Safetensors",
    )
    result.add_argument(
        "--categorical-diagnostic",
        action="store_true",
        help="Compare current categorical reports without issuing parity/acceptance claims",
    )
    result.add_argument(
        "--adaptive-kl-controller-unit",
        choices=("source-contract", "completion-episodes"),
        default="source-contract",
        help="Diagnostic override for replaying legacy evidence with the completion-episode horizon contract",
    )
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
