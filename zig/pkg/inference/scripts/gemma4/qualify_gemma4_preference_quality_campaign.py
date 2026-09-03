#!/usr/bin/env python3
"""Run a pinned multi-seed, long-horizon Gemma4 preference quality gate.

Each seed is bound into independent, reproducible dimensions: Gemma4 LoRA
initialization, the typed trainer/RNG contract, a deterministic permutation of
the exact same training JSONL rows, and the trainer's per-epoch prompt order.
The campaign bootstraps a
fresh seed-bound adapter for every run from the pinned model and template
adapter configuration. Every seed must complete real optimizer-backed Metal
training, pass the recipe's held-out quality gates, and improve held-out reward
directionally. GRPO significance is decided once across prompt-level reward
changes averaged over all training seeds, keeping the evaluation prompt as the
independent experimental unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import qualify_gemma4_preference_resume as resume_qualifier


SCRIPT_PATH = Path(__file__).resolve()
RESUME_QUALIFIER_PATH = Path(resume_qualifier.__file__).resolve()
SCHEMA_VERSION = "antfly_gemma4_preference_quality_campaign/v7"
MIN_GRPO_OPTIMIZER_GROUP_RATE = 0.25
MAX_GRPO_KL_REJECTED_GROUP_RATE = 0.01
MAX_GRPO_PAIRED_SIGN_TEST_P_VALUE = 0.05
MAX_GRPO_POSITIVE_GROUP_REGRESSIONS = 1
F32_UNIT_ROUNDOFF = 2.0**-24
LONG_HORIZON_UNIT_FLOORS = {"dpo": 40, "grpo": 512}
MAX_FAILURE_SUMMARY_BYTES = 16 * 1024 * 1024
MAX_GRPO_EVALUATION_TRACE_BYTES = 128 * 1024 * 1024


class ContractError(RuntimeError):
    pass


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where}: expected object")
    return value


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{where}: expected integer >= {minimum}")
    return value


def _finite(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}: expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{where}: expected finite number")
    return result


def _nonnegative(value: Any, where: str) -> float:
    result = _finite(value, where)
    if result < 0.0:
        raise ContractError(f"{where}: expected non-negative number")
    return result


def _probability(value: Any, where: str) -> float:
    result = _finite(value, where)
    if result < 0.0 or result > 1.0:
        raise ContractError(f"{where}: expected number in [0, 1]")
    return result


def _grpo_positive_group_requirement(
    groups: int,
    minimum_improvement: float | None,
    maximum_regressions: int | None,
) -> tuple[float, int]:
    """Resolve the stochastic positive-group guardrail to an exact count cap."""
    groups = _integer(groups, "GRPO evaluation groups", 1)
    if minimum_improvement is not None and maximum_regressions is not None:
        raise ContractError(
            "choose either minimum GRPO positive-group improvement or maximum regressions"
        )
    if minimum_improvement is not None:
        return (
            _nonnegative(
                minimum_improvement,
                "minimum GRPO positive-group improvement",
            ),
            0,
        )
    regressions = (
        MAX_GRPO_POSITIVE_GROUP_REGRESSIONS
        if maximum_regressions is None
        else _integer(
            maximum_regressions,
            "maximum GRPO positive-group regressions",
        )
    )
    if regressions > MAX_GRPO_POSITIVE_GROUP_REGRESSIONS:
        raise ContractError(
            "GRPO positive-group regression cap cannot relax the production maximum"
        )
    return (-regressions / groups, regressions)


def _rate_count(rate: float, groups: int, where: str) -> int:
    groups = _integer(groups, f"{where} groups", 1)
    rate = _probability(rate, where)
    value = rate * groups
    count = round(value)
    # Zig reports this metric as f32. Reconstruct the integer count with a
    # bound that covers one correctly rounded f32 rate, while remaining far
    # below half a group for every campaign admitted by the trace-size cap.
    tolerance = max(1e-7, groups * F32_UNIT_ROUNDOFF)
    if not math.isclose(value, count, rel_tol=0.0, abs_tol=tolerance):
        raise ContractError(f"{where} is not an exact group-count rate")
    return count


def _grpo_positive_group_noninferiority(
    baseline_rate: float,
    evaluation_rate: float,
    *,
    groups: int,
    maximum_regressions: int,
) -> dict[str, int | bool]:
    maximum_regressions = _integer(
        maximum_regressions, "maximum GRPO positive-group regressions"
    )
    if maximum_regressions > MAX_GRPO_POSITIVE_GROUP_REGRESSIONS:
        raise ContractError(
            "GRPO positive-group regression cap cannot relax the production maximum"
        )
    baseline_groups = _rate_count(
        baseline_rate, groups, "baseline GRPO positive-group rate"
    )
    evaluation_groups = _rate_count(
        evaluation_rate, groups, "final GRPO positive-group rate"
    )
    change = evaluation_groups - baseline_groups
    return {
        "baseline_groups": baseline_groups,
        "evaluation_groups": evaluation_groups,
        "change": change,
        "maximum_regressions": maximum_regressions,
        "passed": change >= -maximum_regressions,
    }


def _long_horizon_units(
    task: str, examples_per_epoch: int, epochs: int
) -> tuple[int, int]:
    if task not in LONG_HORIZON_UNIT_FLOORS:
        raise ContractError("long-horizon task is unsupported")
    if examples_per_epoch <= 0 or epochs <= 0:
        raise ContractError("long-horizon examples and epochs must be positive")
    units = examples_per_epoch * epochs
    minimum = LONG_HORIZON_UNIT_FLOORS[task]
    if units < minimum:
        raise ContractError(
            f"long-horizon {task.upper()} campaign requires at least "
            f"{minimum} training units; got {examples_per_epoch} x {epochs} = {units}"
        )
    return units, minimum


def _grpo_training_coverage(
    report: Mapping[str, Any],
    expected_units: int,
    minimum_optimizer_group_rate: float,
    maximum_kl_rejected_group_rate: float = MAX_GRPO_KL_REJECTED_GROUP_RATE,
) -> dict[str, int | float]:
    """Validate that the reported GRPO horizon contained useful updates.

    Zero-variance reward groups are valid GRPO observations, but they do not
    cross an optimizer boundary. A long logical horizon can therefore hide a
    nearly empty optimization campaign unless qualification gates the admitted
    fraction explicitly.
    """
    if expected_units <= 0:
        raise ContractError("expected GRPO horizon must be positive")
    minimum_rate = _probability(
        minimum_optimizer_group_rate, "minimum GRPO optimizer-group rate"
    )
    if minimum_rate < MIN_GRPO_OPTIMIZER_GROUP_RATE:
        raise ContractError(
            "GRPO optimizer-group floor cannot relax the production minimum: "
            f"{minimum_rate:.6f} < {MIN_GRPO_OPTIMIZER_GROUP_RATE:.6f}"
        )
    maximum_kl_rejected_rate = _probability(
        maximum_kl_rejected_group_rate,
        "maximum GRPO KL-rejected-group rate",
    )
    if maximum_kl_rejected_rate > MAX_GRPO_KL_REJECTED_GROUP_RATE:
        raise ContractError(
            "GRPO KL-rejected-group ceiling cannot relax the production maximum: "
            f"{maximum_kl_rejected_rate:.6f} > "
            f"{MAX_GRPO_KL_REJECTED_GROUP_RATE:.6f}"
        )
    optimizer_groups = _integer(
        report.get("optimizer_groups"), "report.optimizer_groups", 1
    )
    zero_groups = _integer(
        report.get("zero_reward_std_groups"),
        "report.zero_reward_std_groups",
    )
    all_truncated_groups = _integer(
        report.get("all_truncated_groups"), "report.all_truncated_groups"
    )
    kl_rejected_groups = _integer(
        report.get("kl_rejected_groups"), "report.kl_rejected_groups"
    )
    if (
        optimizer_groups
        + zero_groups
        + all_truncated_groups
        + kl_rejected_groups
        != expected_units
    ):
        raise ContractError("GRPO admitted and skipped groups do not cover the horizon")
    reported_zero_fraction = _probability(
        report.get("frac_reward_zero_std"), "report.frac_reward_zero_std"
    )
    if not math.isclose(
        reported_zero_fraction,
        zero_groups / expected_units,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise ContractError("GRPO zero-variance group fraction is inconsistent")
    reported_kl_fraction = _probability(
        report.get("frac_kl_rejected"), "report.frac_kl_rejected"
    )
    if not math.isclose(
        reported_kl_fraction,
        kl_rejected_groups / expected_units,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise ContractError("GRPO KL-rejected group fraction is inconsistent")
    optimizer_group_rate = optimizer_groups / expected_units
    kl_rejected_group_rate = kl_rejected_groups / expected_units
    if optimizer_group_rate < minimum_rate:
        raise ContractError(
            "GRPO optimizer-group rate is below the campaign floor: "
            f"{optimizer_groups}/{expected_units}={optimizer_group_rate:.6f} "
            f"< {minimum_rate:.6f}"
        )
    if kl_rejected_group_rate > maximum_kl_rejected_rate:
        raise ContractError(
            "GRPO KL-rejected-group rate exceeds the campaign ceiling: "
            f"{kl_rejected_groups}/{expected_units}="
            f"{kl_rejected_group_rate:.6f} > {maximum_kl_rejected_rate:.6f}"
        )
    return {
        "optimizer_groups": optimizer_groups,
        "zero_reward_std_groups": zero_groups,
        "all_truncated_groups": all_truncated_groups,
        "kl_rejected_groups": kl_rejected_groups,
        "optimizer_group_rate": optimizer_group_rate,
        "kl_rejected_group_rate": kl_rejected_group_rate,
    }


def _require_grpo_kl_admissions(
    kl_control: Mapping[str, Any], optimizer_groups: int | float
) -> None:
    expected = _integer(optimizer_groups, "GRPO optimizer groups", 1)
    admitted = _integer(
        kl_control.get("admitted_groups"), "kl_control.admitted_groups"
    )
    if admitted != expected:
        raise ContractError(
            "KL controller admitted-group count does not match optimizer groups"
        )


def _one_sided_exact_sign_test_p_value(wins: int, losses: int) -> float:
    """Return P[X >= wins] for X ~ Binomial(wins + losses, 0.5)."""
    wins = _integer(wins, "paired prompt wins")
    losses = _integer(losses, "paired prompt losses")
    discordant = wins + losses
    if discordant == 0:
        return 1.0
    numerator = sum(
        math.comb(discordant, successes)
        for successes in range(wins, discordant + 1)
    )
    return numerator / (1 << discordant)


def _paired_prompt_reward_deltas(
    baseline_rewards: Sequence[float],
    evaluation_rewards: Sequence[float],
    *,
    group_size: int,
) -> list[float]:
    group_size = _integer(group_size, "GRPO evaluation group size", 1)
    if len(baseline_rewards) != len(evaluation_rewards):
        raise ContractError("paired GRPO reward traces have different lengths")
    if not baseline_rewards or len(baseline_rewards) % group_size != 0:
        raise ContractError("paired GRPO reward traces do not cover complete groups")

    prompt_deltas: list[float] = []
    for start in range(0, len(baseline_rewards), group_size):
        baseline_total = math.fsum(
            _finite(reward, "baseline GRPO evaluation reward")
            for reward in baseline_rewards[start : start + group_size]
        )
        evaluation_total = math.fsum(
            _finite(reward, "final GRPO evaluation reward")
            for reward in evaluation_rewards[start : start + group_size]
        )
        prompt_deltas.append(evaluation_total - baseline_total)
    return prompt_deltas


def _paired_prompt_reward_test(
    baseline_rewards: Sequence[float],
    evaluation_rewards: Sequence[float],
    *,
    group_size: int,
    maximum_p_value: float,
) -> dict[str, int | float | bool]:
    """Test common-random-number reward changes at the prompt boundary.

    Completions in one GRPO group share a prompt, so treating every completion
    as independent would be pseudo-replication. Ties carry no directional
    evidence and are excluded from the exact sign test.
    """
    group_size = _integer(group_size, "GRPO evaluation group size", 1)
    maximum_p_value = _probability(
        maximum_p_value, "maximum GRPO paired sign-test p-value"
    )
    if maximum_p_value == 0.0:
        raise ContractError("maximum GRPO paired sign-test p-value must be positive")
    if maximum_p_value > MAX_GRPO_PAIRED_SIGN_TEST_P_VALUE:
        raise ContractError(
            "GRPO paired sign-test p-value ceiling cannot relax the production maximum"
        )
    prompt_deltas = _paired_prompt_reward_deltas(
        baseline_rewards,
        evaluation_rewards,
        group_size=group_size,
    )
    wins = sum(delta > 0.0 for delta in prompt_deltas)
    losses = sum(delta < 0.0 for delta in prompt_deltas)
    ties = len(prompt_deltas) - wins - losses
    p_value = _one_sided_exact_sign_test_p_value(wins, losses)
    baseline_total = math.fsum(baseline_rewards)
    evaluation_total = math.fsum(evaluation_rewards)
    return {
        "groups": len(prompt_deltas),
        "completions": len(baseline_rewards),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_reward": evaluation_total - baseline_total,
        "mean_reward_improvement": (
            evaluation_total - baseline_total
        ) / len(baseline_rewards),
        "one_sided_exact_p_value": p_value,
        "maximum_p_value": maximum_p_value,
        "directional_passed": wins > losses,
        "significance_passed": p_value <= maximum_p_value,
        "passed": wins > losses and p_value <= maximum_p_value,
    }


def _multi_seed_paired_prompt_reward_test(
    prompt_deltas_by_seed: Mapping[int, Sequence[float]],
    *,
    group_size: int,
    maximum_p_value: float,
) -> dict[str, Any]:
    """Test prompt-level reward changes after averaging over training seeds.

    A prompt evaluated under several training seeds remains one held-out task
    observation, not one independent observation per seed. Averaging its group
    reward delta across seeds therefore avoids both completion-level and
    seed-level pseudo-replication. Every seed must also be directionally
    positive on its own before the aggregate result can pass.
    """
    group_size = _integer(group_size, "GRPO evaluation group size", 1)
    maximum_p_value = _probability(
        maximum_p_value, "maximum GRPO paired sign-test p-value"
    )
    if maximum_p_value == 0.0:
        raise ContractError("maximum GRPO paired sign-test p-value must be positive")
    if maximum_p_value > MAX_GRPO_PAIRED_SIGN_TEST_P_VALUE:
        raise ContractError(
            "GRPO paired sign-test p-value ceiling cannot relax the production maximum"
        )
    if len(prompt_deltas_by_seed) < 3:
        raise ContractError("multi-seed GRPO paired test requires at least three seeds")

    normalized: dict[int, list[float]] = {}
    groups: int | None = None
    per_seed: list[dict[str, int | bool]] = []
    for raw_seed, raw_deltas in prompt_deltas_by_seed.items():
        seed = _integer(raw_seed, "GRPO paired-test seed")
        if seed in normalized:
            raise ContractError("multi-seed GRPO paired test contains a duplicate seed")
        deltas = [
            _finite(delta, f"seed {seed} prompt reward delta")
            for delta in raw_deltas
        ]
        if not deltas:
            raise ContractError("multi-seed GRPO paired test has no prompt deltas")
        if groups is None:
            groups = len(deltas)
        elif len(deltas) != groups:
            raise ContractError(
                "multi-seed GRPO paired test has inconsistent prompt counts"
            )
        wins = sum(delta > 0.0 for delta in deltas)
        losses = sum(delta < 0.0 for delta in deltas)
        per_seed.append(
            {
                "seed": seed,
                "wins": wins,
                "losses": losses,
                "ties": len(deltas) - wins - losses,
                "directional_passed": wins > losses,
            }
        )
        normalized[seed] = deltas

    assert groups is not None
    seed_count = len(normalized)
    prompt_deltas = [
        math.fsum(deltas[prompt_index] for deltas in normalized.values())
        / seed_count
        for prompt_index in range(groups)
    ]
    wins = sum(delta > 0.0 for delta in prompt_deltas)
    losses = sum(delta < 0.0 for delta in prompt_deltas)
    ties = groups - wins - losses
    p_value = _one_sided_exact_sign_test_p_value(wins, losses)
    all_seeds_directional = all(
        bool(seed_result["directional_passed"]) for seed_result in per_seed
    )
    net_group_reward = math.fsum(prompt_deltas)
    return {
        "seeds": seed_count,
        "groups": groups,
        "completions_per_seed": groups * group_size,
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "net_group_reward": net_group_reward,
        "mean_completion_reward_improvement": net_group_reward
        / (groups * group_size),
        "one_sided_exact_p_value": p_value,
        "maximum_p_value": maximum_p_value,
        "all_seeds_directional": all_seeds_directional,
        "per_seed": per_seed,
        "passed": (
            all_seeds_directional
            and wins > losses
            and p_value <= maximum_p_value
        ),
        "independent_unit": "evaluation-prompt",
        "seed_aggregation": "arithmetic-mean-of-prompt-group-reward-deltas",
    }


def _load_grpo_evaluation_rewards(
    path: Path,
    *,
    groups: int,
    group_size: int,
    where: str,
) -> list[float]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"{where}: expected regular non-symlink file")
    size_bytes = path.stat().st_size
    if size_bytes > MAX_GRPO_EVALUATION_TRACE_BYTES:
        raise ContractError(
            f"{where}: trace exceeds input limit: "
            f"{size_bytes} > {MAX_GRPO_EVALUATION_TRACE_BYTES} bytes"
        )
    expected_rows = groups * group_size
    rewards_by_call: dict[int, float] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw in enumerate(handle, 1):
                if not raw.strip():
                    continue
                try:
                    row = _mapping(json.loads(raw), f"{where} row {line_number}")
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ContractError(
                        f"{where} row {line_number}: invalid JSON: {exc}"
                    ) from exc
                if row.get("schema_version") != "antfly_inference_grpo_reward_trace/v1":
                    raise ContractError(f"{where} row {line_number}: unsupported schema")
                if row.get("phase") != "evaluation":
                    raise ContractError(f"{where} row {line_number}: wrong reward phase")
                call_index = _integer(
                    row.get("call_index"), f"{where} row {line_number}.call_index"
                )
                prompt_index = _integer(
                    row.get("prompt_index"),
                    f"{where} row {line_number}.prompt_index",
                )
                if call_index in rewards_by_call:
                    raise ContractError(f"{where}: duplicate call_index {call_index}")
                if call_index >= expected_rows or prompt_index != call_index // group_size:
                    raise ContractError(
                        f"{where}: call/prompt order does not match the seeded group contract"
                    )
                rewards_by_call[call_index] = _finite(
                    row.get("aggregate_reward"),
                    f"{where} row {line_number}.aggregate_reward",
                )
    except OSError as exc:
        raise ContractError(f"{where}: {exc}") from exc
    if len(rewards_by_call) != expected_rows:
        raise ContractError(
            f"{where}: expected {expected_rows} ordered rewards, "
            f"got {len(rewards_by_call)}"
        )
    return [rewards_by_call[index] for index in range(expected_rows)]


def _reported_reward_trace(
    report: Mapping[str, Any], expected_path: Path, where: str
) -> Path:
    reward_pipeline = _mapping(report.get("reward_pipeline"), f"{where}.reward_pipeline")
    evidence = resume_qualifier._require_reported_artifact(
        reward_pipeline.get("trace_path"),
        reward_pipeline.get("trace_digest"),
        expected_path,
        where,
    )
    return Path(str(evidence["path"])).resolve(strict=True)


def _paired_grpo_evaluation_evidence(
    run_root: Path,
    *,
    groups: int,
    group_size: int,
    baseline_summary: Mapping[str, Any],
    evaluation_summary: Mapping[str, Any],
    maximum_p_value: float,
) -> dict[str, Any]:
    baseline_report = _load_json(
        run_root / "grpo_baseline_evaluation_report.json",
        "standalone baseline GRPO evaluation report",
    )
    evaluation_report = _load_json(
        run_root / "grpo-evaluation.json",
        "standalone final GRPO evaluation report",
    )
    baseline_path = _reported_reward_trace(
        baseline_report,
        run_root / "grpo_baseline_evaluation_reward_trace.jsonl",
        "baseline GRPO evaluation reward trace",
    )
    evaluation_path = _reported_reward_trace(
        evaluation_report,
        run_root / "grpo_evaluation_reward_trace.jsonl",
        "final GRPO evaluation reward trace",
    )
    baseline_rewards = _load_grpo_evaluation_rewards(
        baseline_path,
        groups=groups,
        group_size=group_size,
        where="baseline GRPO evaluation reward trace",
    )
    evaluation_rewards = _load_grpo_evaluation_rewards(
        evaluation_path,
        groups=groups,
        group_size=group_size,
        where="final GRPO evaluation reward trace",
    )
    paired = _paired_prompt_reward_test(
        baseline_rewards,
        evaluation_rewards,
        group_size=group_size,
        maximum_p_value=maximum_p_value,
    )
    for name, rewards, summary in (
        ("baseline", baseline_rewards, baseline_summary),
        ("final", evaluation_rewards, evaluation_summary),
    ):
        mean_reward = math.fsum(rewards) / len(rewards)
        top_rank_mean_reward = math.fsum(rewards[::group_size]) / groups
        positive_reward_group_rate = sum(
            any(reward > 0.0 for reward in rewards[start : start + group_size])
            for start in range(0, len(rewards), group_size)
        ) / groups
        for field, computed in (
            ("mean_reward", mean_reward),
            ("top_rank_mean_reward", top_rank_mean_reward),
            ("positive_reward_group_rate", positive_reward_group_rate),
        ):
            reported = _finite(summary.get(field), f"{name} evaluation.{field}")
            if not math.isclose(reported, computed, rel_tol=0.0, abs_tol=1e-7):
                raise ContractError(
                    f"{name} GRPO reward trace disagrees with reported {field}"
                )
    reported_improvement = _finite(
        evaluation_summary.get("mean_reward"), "evaluation.mean_reward"
    ) - _finite(baseline_summary.get("mean_reward"), "baseline.mean_reward")
    if not math.isclose(
        float(paired["mean_reward_improvement"]),
        reported_improvement,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise ContractError(
            "paired GRPO reward traces disagree with reported mean-reward improvement"
        )
    if paired["directional_passed"] is not True:
        raise ContractError(
            "held-out GRPO prompt-level paired reward improvement is not directional"
        )
    return {
        **{name: value for name, value in paired.items() if name != "passed"},
        "per_seed_gate_passed": True,
        "per_seed_significance_required": False,
        "baseline_trace_path": str(baseline_path),
        "baseline_trace_sha256": resume_qualifier._sha256(baseline_path),
        "evaluation_trace_path": str(evaluation_path),
        "evaluation_trace_sha256": resume_qualifier._sha256(evaluation_path),
        "pairing_unit": "prompt-clustered-common-random-number-completion-group",
    }


def _multi_seed_grpo_evaluation_evidence(
    runs: Sequence[Mapping[str, Any]],
    *,
    expected_groups: int,
    maximum_p_value: float,
) -> dict[str, Any]:
    expected_groups = _integer(
        expected_groups, "expected GRPO evaluation groups", 1
    )
    prompt_deltas_by_seed: dict[int, list[float]] = {}
    trace_pairs: list[dict[str, Any]] = []
    group_size: int | None = None
    for run in runs:
        seed = _integer(run.get("seed"), "campaign run seed")
        if seed in prompt_deltas_by_seed:
            raise ContractError("multi-seed GRPO paired evidence contains a duplicate seed")
        quality = _mapping(run.get("quality"), f"seed {seed} quality")
        paired = _mapping(
            quality.get("paired_evaluation"), f"seed {seed} paired evaluation"
        )
        if paired.get("per_seed_gate_passed") is not True:
            raise ContractError(
                f"seed {seed} did not pass the paired directionality gate"
            )
        groups = _integer(paired.get("groups"), f"seed {seed} paired groups", 1)
        completions = _integer(
            paired.get("completions"), f"seed {seed} paired completions", 1
        )
        if groups != expected_groups or completions % groups != 0:
            raise ContractError(
                f"seed {seed} paired evidence has the wrong evaluation shape"
            )
        current_group_size = completions // groups
        if group_size is None:
            group_size = current_group_size
        elif current_group_size != group_size:
            raise ContractError(
                "multi-seed GRPO paired evidence has inconsistent group sizes"
            )

        paths: dict[str, Path] = {}
        for phase in ("baseline", "evaluation"):
            raw_path = paired.get(f"{phase}_trace_path")
            expected_digest = paired.get(f"{phase}_trace_sha256")
            if not isinstance(raw_path, str) or not isinstance(expected_digest, str):
                raise ContractError(
                    f"seed {seed} paired {phase} trace identity is malformed"
                )
            path = Path(raw_path)
            if path.is_symlink() or not path.is_file():
                raise ContractError(
                    f"seed {seed} paired {phase} trace is not a regular file"
                )
            if resume_qualifier._sha256(path) != expected_digest:
                raise ContractError(
                    f"seed {seed} paired {phase} trace changed before aggregation"
                )
            paths[phase] = path

        baseline_rewards = _load_grpo_evaluation_rewards(
            paths["baseline"],
            groups=groups,
            group_size=current_group_size,
            where=f"seed {seed} baseline GRPO evaluation reward trace",
        )
        evaluation_rewards = _load_grpo_evaluation_rewards(
            paths["evaluation"],
            groups=groups,
            group_size=current_group_size,
            where=f"seed {seed} final GRPO evaluation reward trace",
        )
        prompt_deltas = _paired_prompt_reward_deltas(
            baseline_rewards,
            evaluation_rewards,
            group_size=current_group_size,
        )
        wins = sum(delta > 0.0 for delta in prompt_deltas)
        losses = sum(delta < 0.0 for delta in prompt_deltas)
        ties = groups - wins - losses
        for field, computed in (("wins", wins), ("losses", losses), ("ties", ties)):
            if _integer(
                paired.get(field), f"seed {seed} paired {field}"
            ) != computed:
                raise ContractError(
                    f"seed {seed} paired trace disagrees with recorded {field}"
                )
        prompt_deltas_by_seed[seed] = prompt_deltas
        trace_pairs.append(
            {
                "seed": seed,
                "baseline_trace_path": str(paths["baseline"]),
                "baseline_trace_sha256": paired["baseline_trace_sha256"],
                "evaluation_trace_path": str(paths["evaluation"]),
                "evaluation_trace_sha256": paired["evaluation_trace_sha256"],
            }
        )

    assert group_size is not None
    aggregate = _multi_seed_paired_prompt_reward_test(
        prompt_deltas_by_seed,
        group_size=group_size,
        maximum_p_value=maximum_p_value,
    )
    return {
        **aggregate,
        "trace_pairs": trace_pairs,
        "pairing_unit": "prompt-clustered-common-random-number-completion-group",
    }


def _bounded_increase_requirement(
    baseline: float, requested: float, ceiling: float
) -> tuple[float, bool]:
    headroom = max(0.0, ceiling - baseline)
    return min(requested, headroom), requested > headroom


def _bounded_decrease_requirement(
    baseline: float, requested: float, floor: float
) -> tuple[float, bool]:
    headroom = max(0.0, baseline - floor)
    return min(requested, headroom), requested > headroom


def _verify_bounded_requirement(
    baseline_relative: Mapping[str, Any],
    field_prefix: str,
    effective_minimum: float,
    saturated: bool,
    *,
    allow_negative: bool = False,
) -> None:
    reported_minimum = (
        _finite
        if allow_negative
        else _nonnegative
    )(
        baseline_relative.get(f"{field_prefix}_required_improvement"),
        f"baseline_relative.{field_prefix}_required_improvement",
    )
    if not math.isclose(
        reported_minimum,
        effective_minimum,
        rel_tol=1e-6,
        abs_tol=1e-12,
    ):
        raise ContractError(
            f"baseline_relative.{field_prefix} effective requirement mismatch"
        )
    reported_saturated = baseline_relative.get(
        f"{field_prefix}_requirement_saturated"
    )
    if not isinstance(reported_saturated, bool) or reported_saturated != saturated:
        raise ContractError(
            f"baseline_relative.{field_prefix} saturation mismatch"
        )


def _load_json(path: Path, where: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), where)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{where}: {exc}") from exc


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        temporary.unlink(missing_ok=True)
        raise ContractError(f"refusing to replace file: {path}")
    os.replace(temporary, path)
    _fsync_parent(path)


def _write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    if path.exists():
        temporary.unlink(missing_ok=True)
        raise ContractError(f"refusing to replace file: {path}")
    os.replace(temporary, path)
    _fsync_parent(path)


def _fsync_parent(path: Path) -> None:
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _available_failure_artifact(
    path: Path, summary_fields: Sequence[str] = ()
) -> dict[str, Any] | None:
    """Return bounded evidence for an artifact left by a failed run.

    Failure reporting must not trust symlinks or copy an unbounded trainer
    payload into the campaign report. The digest keeps the original artifact
    independently inspectable while the allowlisted summary makes the actual
    failed gate visible at the campaign boundary.
    """
    try:
        if path.is_symlink() or not path.is_file():
            return None
        size_bytes = path.stat().st_size
        evidence: dict[str, Any] = {
            "path": str(path.resolve()),
            "sha256": resume_qualifier._sha256(path),
            "size_bytes": size_bytes,
        }
        if summary_fields:
            if size_bytes > MAX_FAILURE_SUMMARY_BYTES:
                evidence["summary_error"] = (
                    "artifact exceeds failure-summary input limit: "
                    f"{size_bytes} > {MAX_FAILURE_SUMMARY_BYTES} bytes"
                )
                return evidence
            payload = _load_json(path, f"failed-run artifact {path.name}")
            evidence["summary"] = {
                field: payload[field] for field in summary_fields if field in payload
            }
        return evidence
    except (ContractError, OSError) as exc:
        return {
            "path": str(path),
            "evidence_error": str(exc),
        }


def _failed_run_evidence(task: str, seed: int, run_root: Path) -> dict[str, Any]:
    specs: tuple[tuple[str, Path, tuple[str, ...]], ...] = (
        (
            "task_report",
            run_root / f"{task}_report.json",
            (
                "schema_version",
                "execution_mode",
                "examples",
                "groups",
                "optimizer_groups",
                "zero_reward_std_groups",
                "all_truncated_groups",
                "kl_rejected_groups",
                "loss",
                "mean_reward",
                "baseline_evaluation",
                "baseline_relative",
                "evaluation",
                "trained_adapter_dir",
            ),
        ),
        (
            "evaluation_report",
            run_root / f"{task}-evaluation.json",
            (
                "schema_version",
                "status",
                "examples",
                "groups",
                "loss",
                "accuracy",
                "mean_reward_margin",
                "mean_reward",
                "top_rank_mean_reward",
                "positive_reward_group_rate",
                "kl_loss",
                "mean_kl",
                "policy_adapter_digest",
            ),
        ),
        (
            "baseline_evaluation_report",
            run_root / f"{task}_baseline_evaluation_report.json",
            (
                "schema_version",
                "status",
                "examples",
                "groups",
                "loss",
                "accuracy",
                "mean_reward_margin",
                "mean_reward",
                "top_rank_mean_reward",
                "positive_reward_group_rate",
                "kl_loss",
                "mean_kl",
                "policy_adapter_digest",
            ),
        ),
        (
            "baseline_evaluation_reward_trace",
            run_root / f"{task}_baseline_evaluation_reward_trace.jsonl",
            (),
        ),
        (
            "training_report",
            run_root / "training_report.json",
            ("schema_version", "status", "steps"),
        ),
        (
            "recipe_run_manifest",
            run_root / "recipe_run_manifest.json",
            ("schema_version", "status"),
        ),
        ("stdout_log", run_root.with_suffix(".stdout.log"), ()),
        ("stderr_log", run_root.with_suffix(".stderr.log"), ()),
        ("reward_trace", run_root / f"{task}_reward_trace.jsonl", ()),
        (
            "evaluation_reward_trace",
            run_root / f"{task}_evaluation_reward_trace.jsonl",
            (),
        ),
        ("kl_control_trace", run_root / "grpo_kl_control_trace.jsonl", ()),
    )
    artifacts: dict[str, Any] = {}
    for label, path, summary_fields in specs:
        evidence = _available_failure_artifact(path, summary_fields)
        if evidence is not None:
            artifacts[label] = evidence
    return {
        "seed": seed,
        "run_root": str(run_root.resolve()),
        "artifacts": artifacts,
    }


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(item.strip(), 10) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("seeds must be comma-separated u64 values") from exc
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("at least three unique seeds are required")
    if any(seed < 0 or seed > (1 << 64) - 1 for seed in seeds):
        raise argparse.ArgumentTypeError("seeds must fit u64")
    return seeds


def _jsonl_rows(path: Path, where: str = "training dataset") -> list[bytes]:
    try:
        raw_rows = path.read_bytes().splitlines()
    except OSError as exc:
        raise ContractError(f"{where}: {exc}") from exc
    rows = [row for row in raw_rows if row.strip()]
    if len(rows) < 2:
        raise ContractError(f"{where} requires at least two non-empty rows")
    for index, row in enumerate(rows, 1):
        try:
            value = json.loads(row)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"{where} row {index}: invalid JSON: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ContractError(f"{where} row {index}: expected object")
    return rows


def _permuted_rows(rows: Sequence[bytes], seed: int) -> bytes:
    decorated = []
    for index, row in enumerate(rows):
        key = hashlib.sha256(
            seed.to_bytes(8, "little") + index.to_bytes(8, "little") + row
        ).digest()
        decorated.append((key, index, row))
    decorated.sort()
    return b"\n".join(item[2] for item in decorated) + b"\n"


def _row_multiset_sha256(rows: Sequence[bytes]) -> str:
    row_digests = sorted(hashlib.sha256(row).digest() for row in rows)
    return "sha256:" + hashlib.sha256(b"".join(row_digests)).hexdigest()


def _require_permutation_capacity(row_count: int, seed_count: int) -> None:
    capacity = 1
    for factor in range(2, row_count + 1):
        capacity *= factor
        if capacity >= seed_count:
            return
    if capacity < seed_count:
        raise ContractError(
            "selected training rows cannot produce enough distinct data-order seeds"
        )


def _dataset_path(recipe: Mapping[str, Any], field: str) -> Path:
    dataset = _mapping(recipe.get("dataset"), "recipe.dataset")
    if field == "train":
        primary = dataset.get("train_path")
        fallback = dataset.get("path")
    else:
        evaluation = _mapping(recipe.get("eval"), "recipe.eval")
        primary = evaluation.get("path")
        fallback = dataset.get("eval_path")
    try:
        return resume_qualifier._effective_regular_file(
            primary,
            fallback,
            f"{field} dataset",
        )
    except resume_qualifier.ContractError as exc:
        raise ContractError(str(exc)) from exc


def _adapter_bootstrap_spec(adapter: Path) -> dict[str, Any]:
    config = _load_json(adapter / "adapter_config.json", "template adapter config")
    rank = _integer(config.get("r"), "template adapter config.r", 1)
    alpha = _finite(config.get("lora_alpha"), "template adapter config.lora_alpha")
    if alpha <= 0.0:
        raise ContractError("template adapter alpha must be positive")
    raw_targets = config.get("target_modules")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ContractError("template adapter must contain exact target_modules")
    targets: list[str] = []
    for index, value in enumerate(raw_targets):
        if not isinstance(value, str) or not value.strip():
            raise ContractError(
                f"template adapter target_modules[{index}] must be non-empty"
            )
        targets.append(value)
    if len(set(targets)) != len(targets):
        raise ContractError("template adapter target_modules must be unique")
    if config.get("use_dora", False) is not False:
        raise ContractError("independent-initialization campaign does not admit DoRA")
    initializer = config.get("init_lora_weights", True)
    if initializer is not True and initializer != "default":
        raise ContractError(
            "independent-initialization campaign requires standard LoRA initialization"
        )
    return {"rank": rank, "alpha": alpha, "target_modules": targets}


def _bootstrap_seed_adapter(
    binary: Path,
    model: Path,
    template_spec: Mapping[str, Any],
    seed: int,
    adapter_path: Path,
    env: Mapping[str, str],
    log_root: Path,
    timeout: float,
) -> dict[str, Any]:
    command = [
        str(binary),
        "inference",
        "finetune",
        "adapter",
        "bootstrap",
        "gemma4",
        "--model",
        str(model),
        "--out",
        str(adapter_path),
        "--rank",
        str(template_spec["rank"]),
        "--alpha",
        format(float(template_spec["alpha"]), ".17g"),
        "--target-modules",
        ",".join(template_spec["target_modules"]),
        "--initialization-seed",
        str(seed),
    ]
    execution = _run(command, env, log_root, timeout)
    evidence = resume_qualifier._adapter_tree_evidence(
        adapter_path, f"seed-{seed} initialized adapter"
    )
    manifest_path = adapter_path / "antfly_finetune_manifest.json"
    manifest = _load_json(manifest_path, f"seed-{seed} initialized adapter manifest")
    if manifest.get("schema_version") != "antfly_gemma4_finetune/v3":
        raise ContractError("seeded bootstrap did not publish a v3 adapter manifest")
    if manifest.get("status") != "complete":
        raise ContractError("seeded bootstrap manifest is incomplete")
    if _integer(
        manifest.get("initialization_seed"), "adapter manifest.initialization_seed"
    ) != seed:
        raise ContractError("seeded bootstrap manifest attests the wrong seed")
    config = _load_json(
        adapter_path / "adapter_config.json", f"seed-{seed} initialized adapter config"
    )
    if (
        _integer(config.get("r"), "initialized adapter config.r", 1)
        != template_spec["rank"]
        or _finite(config.get("lora_alpha"), "initialized adapter config.lora_alpha")
        != template_spec["alpha"]
        or config.get("target_modules") != template_spec["target_modules"]
    ):
        raise ContractError("seeded bootstrap drifted from the template adapter config")
    return {
        "initialization_seed": seed,
        "path": str(adapter_path),
        "adapter_model_sha256": evidence["adapter_model_sha256"],
        "adapter_tree": evidence["files"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": resume_qualifier._sha256(manifest_path),
        "execution": execution,
    }


def _variant(
    base: Mapping[str, Any],
    task: str,
    seed: int,
    adapter_path: Path,
    train_path: Path,
    run_root: Path,
    epochs: int,
    learning_rate: float | None,
    allow_direct_gguf_training: bool,
) -> dict[str, Any]:
    recipe = json.loads(json.dumps(base))
    dataset = dict(_mapping(recipe.get("dataset"), "recipe.dataset"))
    if dataset.get("train_path") is not None:
        dataset["train_path"] = str(train_path)
    if dataset.get("path") is not None:
        dataset["path"] = str(train_path)
    recipe["dataset"] = dataset
    optimizer = dict(_mapping(recipe.get("optimizer"), "recipe.optimizer"))
    optimizer["seed"] = seed
    optimizer["epochs"] = epochs
    if learning_rate is not None:
        optimizer["learning_rate"] = learning_rate
    recipe["optimizer"] = optimizer
    recipe.pop("checkpoint", None)
    model = dict(_mapping(recipe.get("model"), "recipe.model"))
    if allow_direct_gguf_training:
        model["allow_direct_gguf_training"] = True
    recipe["model"] = model
    adapter = dict(_mapping(recipe.get("adapter"), "recipe.adapter"))
    adapter["path"] = str(adapter_path)
    adapter["initialization_seed"] = seed
    recipe["adapter"] = adapter
    recipe["artifacts"] = {
        "root": str(run_root),
        "trained_adapter_dir": str(run_root / "adapter-trained"),
        "report_path": str(run_root / f"{task}_report.json"),
        "evaluation_report_path": str(run_root / f"{task}-evaluation.json"),
    }
    return recipe


def _run(command: Sequence[str], env: Mapping[str, str], root: Path, timeout: float) -> dict[str, Any]:
    stdout_path = root.with_suffix(".stdout.log")
    stderr_path = root.with_suffix(".stderr.log")
    started = time.monotonic()
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        try:
            completed = subprocess.run(
                list(command),
                env=dict(env),
                stdout=stdout,
                stderr=stderr,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ContractError(f"training timed out after {timeout:g}s") from exc
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        tail = stderr_path.read_bytes()[-6000:].decode(errors="replace")
        raise ContractError(f"training failed with {completed.returncode}; stderr tail:\n{tail}")
    return {
        "command": list(command),
        "returncode": completed.returncode,
        "elapsed_seconds": elapsed,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }


def _validate_run(
    task: str,
    run_root: Path,
    expected_seed: int,
    expected_units: int,
    expected_optimizer_steps: int,
    gradient_accumulation_steps: int,
    expected_epochs: int,
    expected_train_path: Path,
    expected_seed_adapter_path: Path,
    expected_seed_adapter: Mapping[str, Any],
    quality_gates: Mapping[str, float],
    compiled_sampling: bool,
) -> dict[str, Any]:
    report_path = run_root / f"{task}_report.json"
    report = _load_json(report_path, f"{task} report")
    expected_schema = (
        resume_qualifier.DPO_REPORT_SCHEMA_VERSION
        if task == "dpo"
        else resume_qualifier.GRPO_REPORT_SCHEMA_VERSION
    )
    if report.get("schema_version") != expected_schema:
        raise ContractError(f"{task} report has unsupported schema")
    if report.get("execution_mode") != "train" or report.get("policy_backend") != "metal":
        raise ContractError("campaign run was not optimizer-backed Metal training")
    if compiled_sampling:
        if report.get("sampling_mode") != resume_qualifier.COMPILED_GRPO_SAMPLING_MODE:
            raise ContractError(
                "compiled-sampling campaign did not execute the compiled GRPO sampling mode"
            )
        if report.get("incremental_kv") is not None:
            raise ContractError(
                "compiled-sampling campaign unexpectedly emitted incremental-KV telemetry"
            )
    if _integer(report.get("training_seed"), "report.training_seed") != expected_seed:
        raise ContractError("campaign report attests the wrong typed training seed")
    evaluation = _mapping(report.get("evaluation"), "report.evaluation")
    if evaluation.get("passed") is not True:
        raise ContractError("held-out evaluation did not pass")
    resume_qualifier._semantic_report_view(report, task)
    verified_artifacts = resume_qualifier._validate_report_artifacts(
        report, run_root, task
    )
    unit_field = "examples" if task == "dpo" else "groups"
    if _integer(report.get(unit_field), f"report.{unit_field}", 1) != expected_units:
        raise ContractError(f"report.{unit_field} does not cover the full horizon")
    minimum_micro_batch_units = expected_units
    realized_optimizer_steps = expected_optimizer_steps
    grpo_training_coverage: Mapping[str, int | float] | None = None
    if task == "grpo":
        grpo_training_coverage = _grpo_training_coverage(
            report,
            expected_units,
            quality_gates["min_optimizer_group_rate"],
            quality_gates["max_kl_rejected_group_rate"],
        )
        optimizer_groups = int(grpo_training_coverage["optimizer_groups"])
        realized_optimizer_steps = (
            optimizer_groups + gradient_accumulation_steps - 1
        ) // gradient_accumulation_steps
        minimum_micro_batch_units = optimizer_groups
    if _integer(report.get("optimizer_steps"), "report.optimizer_steps", 1) != realized_optimizer_steps:
        raise ContractError("report.optimizer_steps does not cover the admitted long horizon")
    if _integer(report.get("micro_batch_steps"), "report.micro_batch_steps", 1) < minimum_micro_batch_units:
        raise ContractError("report.micro_batch_steps is shorter than the long horizon")

    training_report_path = run_root / "training_report.json"
    training_report = _load_json(training_report_path, "outer training report")
    if (
        training_report.get("schema_version")
        != "antfly_inference_finetune_training_report/v1"
        or training_report.get("status") != "succeeded"
    ):
        raise ContractError("outer training report did not succeed")
    steps = training_report.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        raise ContractError("outer training report must contain exactly one step")
    step = _mapping(steps[0], "outer training report step")
    if step.get("status") != "succeeded" or step.get("exit_code") != 0:
        raise ContractError("outer training step did not succeed")
    realized_recipe = _mapping(training_report.get("recipe"), "outer training recipe")
    realized_optimizer = _mapping(
        realized_recipe.get("optimizer"), "outer training recipe.optimizer"
    )
    if _integer(realized_optimizer.get("epochs"), "realized optimizer.epochs", 1) != expected_epochs:
        raise ContractError("outer training report attests the wrong epoch horizon")
    if _integer(realized_optimizer.get("seed"), "realized optimizer.seed") != expected_seed:
        raise ContractError("outer training report attests the wrong typed training seed")
    realized_adapter = _mapping(
        realized_recipe.get("adapter"), "outer training recipe.adapter"
    )
    realized_adapter_path = realized_adapter.get("path")
    if (
        not isinstance(realized_adapter_path, str)
        or Path(realized_adapter_path).resolve() != expected_seed_adapter_path.resolve()
    ):
        raise ContractError("outer training report attests the wrong initialized adapter")
    if _integer(
        realized_adapter.get("initialization_seed"),
        "realized adapter.initialization_seed",
    ) != expected_seed:
        raise ContractError("outer training report attests the wrong initialization seed")
    realized_dataset = _mapping(
        realized_recipe.get("dataset"), "outer training recipe.dataset"
    )
    realized_train = realized_dataset.get("train_path") or realized_dataset.get("path")
    if not isinstance(realized_train, str) or Path(realized_train).resolve() != expected_train_path.resolve():
        raise ContractError("outer training report attests the wrong seeded dataset")

    manifest_path = run_root / "recipe_run_manifest.json"
    manifest = _load_json(manifest_path, "recipe run manifest")
    if (
        manifest.get("schema_version") != "antfly_inference_finetune_recipe_run/v1"
        or manifest.get("status") != "succeeded"
    ):
        raise ContractError("recipe run manifest did not succeed")
    current_seed_adapter = resume_qualifier._adapter_tree_evidence(
        expected_seed_adapter_path, "initialized adapter after training"
    )
    if (
        current_seed_adapter["adapter_model_sha256"]
        != expected_seed_adapter["adapter_model_sha256"]
        or current_seed_adapter["files"] != expected_seed_adapter["adapter_tree"]
    ):
        raise ContractError("training command changed its initialized adapter input")
    trained = resume_qualifier._adapter_tree_evidence(
        run_root / "adapter-trained", "trained adapter"
    )
    if (
        trained["adapter_model_sha256"]
        == expected_seed_adapter["adapter_model_sha256"]
    ):
        raise ContractError("trained adapter is byte-identical to the seed adapter")

    common_metrics = {
        "train_loss": _finite(report.get("loss"), "report.loss"),
    }
    baseline = _mapping(report.get("baseline_evaluation"), "report.baseline_evaluation")
    baseline_relative = _mapping(
        report.get("baseline_relative"), "report.baseline_relative"
    )
    if baseline_relative.get("passed") is not True:
        raise ContractError("baseline-relative held-out evaluation did not pass")
    expected_baseline_path = run_root / f"{task}_baseline_evaluation_report.json"
    baseline_report_path = baseline.get("report_path")
    if (
        not isinstance(baseline_report_path, str)
        or Path(baseline_report_path).resolve() != expected_baseline_path.resolve()
        or not expected_baseline_path.is_file()
    ):
        raise ContractError("baseline evaluation report path escaped the campaign run")
    if task == "dpo":
        metrics = {
            **common_metrics,
            "eval_loss": _nonnegative(evaluation.get("loss"), "evaluation.loss"),
            "train_accuracy": _probability(report.get("accuracy"), "report.accuracy"),
            "eval_accuracy": _probability(evaluation.get("accuracy"), "evaluation.accuracy"),
            "eval_reward_margin": _finite(
                evaluation.get("mean_reward_margin"), "evaluation.mean_reward_margin"
            ),
            "baseline_eval_loss": _nonnegative(
                baseline.get("loss"), "baseline_evaluation.loss"
            ),
            "baseline_eval_accuracy": _probability(
                baseline.get("accuracy"), "baseline_evaluation.accuracy"
            ),
            "baseline_eval_reward_margin": _finite(
                baseline.get("mean_reward_margin"),
                "baseline_evaluation.mean_reward_margin",
            ),
            "eval_accuracy_improvement": _finite(
                baseline_relative.get("accuracy_improvement"),
                "baseline_relative.accuracy_improvement",
            ),
            "eval_accuracy_required_improvement": _nonnegative(
                baseline_relative.get("accuracy_required_improvement"),
                "baseline_relative.accuracy_required_improvement",
            ),
            "eval_reward_margin_improvement": _finite(
                baseline_relative.get("reward_margin_improvement"),
                "baseline_relative.reward_margin_improvement",
            ),
            "eval_loss_improvement": _finite(
                baseline_relative.get("loss_improvement"),
                "baseline_relative.loss_improvement",
            ),
            "eval_loss_required_improvement": _nonnegative(
                baseline_relative.get("loss_required_improvement"),
                "baseline_relative.loss_required_improvement",
            ),
        }
        if metrics["eval_accuracy"] < quality_gates["min_eval_accuracy"]:
            raise ContractError("held-out DPO accuracy is below the campaign floor")
        if metrics["eval_loss"] > quality_gates["max_eval_loss"]:
            raise ContractError("held-out DPO loss exceeds the campaign ceiling")
        if metrics["eval_reward_margin"] < quality_gates["min_eval_reward_margin"]:
            raise ContractError("held-out DPO reward margin is below the campaign floor")
        accuracy_required, accuracy_saturated = _bounded_increase_requirement(
            metrics["baseline_eval_accuracy"],
            quality_gates["min_eval_accuracy_improvement"],
            1.0,
        )
        loss_required, loss_saturated = _bounded_decrease_requirement(
            metrics["baseline_eval_loss"],
            quality_gates["min_eval_loss_improvement"],
            0.0,
        )
        _verify_bounded_requirement(
            baseline_relative, "accuracy", accuracy_required, accuracy_saturated
        )
        _verify_bounded_requirement(
            baseline_relative, "loss", loss_required, loss_saturated
        )
        if metrics["eval_accuracy_improvement"] < accuracy_required:
            raise ContractError(
                "held-out DPO eval_accuracy_improvement is below the effective baseline-relative floor"
            )
        if (
            metrics["eval_reward_margin_improvement"]
            < quality_gates["min_eval_reward_margin_improvement"]
        ):
            raise ContractError(
                "held-out DPO eval_reward_margin_improvement is below the baseline-relative floor"
            )
        if metrics["eval_loss_improvement"] < loss_required:
            raise ContractError(
                "held-out DPO eval_loss_improvement is below the effective baseline-relative floor"
            )
    else:
        assert grpo_training_coverage is not None
        realized_grpo = _mapping(
            realized_recipe.get("grpo"), "outer training recipe.grpo"
        )
        evaluation_group_size = _integer(
            realized_grpo.get("group_size"), "realized GRPO group_size", 1
        )
        evaluation_groups = _integer(
            evaluation.get("groups"), "evaluation.groups", 1
        )
        if evaluation_groups != _integer(
            quality_gates["expected_eval_groups"],
            "expected GRPO evaluation groups",
            1,
        ):
            raise ContractError(
                "held-out GRPO evaluation does not cover the predeclared group count"
            )
        metrics = {
            **common_metrics,
            "train_optimizer_groups": int(
                grpo_training_coverage["optimizer_groups"]
            ),
            "train_optimizer_group_rate": float(
                grpo_training_coverage["optimizer_group_rate"]
            ),
            "train_zero_reward_std_groups": int(
                grpo_training_coverage["zero_reward_std_groups"]
            ),
            "train_kl_rejected_groups": int(
                grpo_training_coverage["kl_rejected_groups"]
            ),
            "train_kl_rejected_group_rate": float(
                grpo_training_coverage["kl_rejected_group_rate"]
            ),
            "train_mean_reward": _finite(report.get("mean_reward"), "report.mean_reward"),
            "train_mean_kl": _nonnegative(report.get("mean_kl"), "report.mean_kl"),
            "eval_mean_reward": _finite(
                evaluation.get("mean_reward"), "evaluation.mean_reward"
            ),
            "eval_top_rank_mean_reward": _finite(
                evaluation.get("top_rank_mean_reward"),
                "evaluation.top_rank_mean_reward",
            ),
            "eval_positive_reward_group_rate": _probability(
                evaluation.get("positive_reward_group_rate"),
                "evaluation.positive_reward_group_rate",
            ),
            "eval_kl_loss": _nonnegative(
                evaluation.get("kl_loss"), "evaluation.kl_loss"
            ),
            "eval_mean_kl": _nonnegative(evaluation.get("mean_kl"), "evaluation.mean_kl"),
            "baseline_eval_mean_reward": _finite(
                baseline.get("mean_reward"), "baseline_evaluation.mean_reward"
            ),
            "baseline_eval_top_rank_mean_reward": _finite(
                baseline.get("top_rank_mean_reward"),
                "baseline_evaluation.top_rank_mean_reward",
            ),
            "baseline_eval_positive_reward_group_rate": _probability(
                baseline.get("positive_reward_group_rate"),
                "baseline_evaluation.positive_reward_group_rate",
            ),
            "eval_mean_reward_improvement": _finite(
                baseline_relative.get("mean_reward_improvement"),
                "baseline_relative.mean_reward_improvement",
            ),
            "eval_top_rank_mean_reward_improvement": _finite(
                baseline_relative.get("top_rank_mean_reward_improvement"),
                "baseline_relative.top_rank_mean_reward_improvement",
            ),
            "eval_positive_reward_group_rate_improvement": _finite(
                baseline_relative.get("positive_reward_group_rate_improvement"),
                "baseline_relative.positive_reward_group_rate_improvement",
            ),
            "eval_positive_reward_group_rate_required_improvement": _finite(
                baseline_relative.get(
                    "positive_reward_group_rate_required_improvement"
                ),
                "baseline_relative.positive_reward_group_rate_required_improvement",
            ),
        }
        if metrics["eval_mean_reward"] < quality_gates["min_eval_mean_reward"]:
            raise ContractError("held-out GRPO mean reward is below the campaign floor")
        if metrics["eval_top_rank_mean_reward"] < quality_gates["min_eval_top_rank_mean_reward"]:
            raise ContractError("held-out GRPO top-rank reward is below the campaign floor")
        if metrics["eval_positive_reward_group_rate"] < quality_gates["min_eval_positive_reward_group_rate"]:
            raise ContractError("held-out GRPO positive-group rate is below the campaign floor")
        if metrics["eval_kl_loss"] > quality_gates["max_eval_kl_loss"]:
            raise ContractError("held-out GRPO KL loss exceeds the campaign ceiling")
        positive_group_noninferiority = _grpo_positive_group_noninferiority(
            metrics["baseline_eval_positive_reward_group_rate"],
            metrics["eval_positive_reward_group_rate"],
            groups=evaluation_groups,
            maximum_regressions=_integer(
                quality_gates["max_eval_positive_reward_group_regressions"],
                "maximum GRPO positive-group regressions",
            ),
        )
        metrics.update(
            {
                "baseline_eval_positive_reward_groups": int(
                    positive_group_noninferiority["baseline_groups"]
                ),
                "eval_positive_reward_groups": int(
                    positive_group_noninferiority["evaluation_groups"]
                ),
                "eval_positive_reward_group_change": int(
                    positive_group_noninferiority["change"]
                ),
                "max_eval_positive_reward_group_regressions": int(
                    positive_group_noninferiority["maximum_regressions"]
                ),
            }
        )
        if positive_group_noninferiority["passed"] is not True:
            raise ContractError(
                "held-out GRPO positive-group count exceeds the net-regression cap"
            )
        for metric_name, gate_name in (
            ("eval_mean_reward_improvement", "min_eval_mean_reward_improvement"),
            (
                "eval_top_rank_mean_reward_improvement",
                "min_eval_top_rank_mean_reward_improvement",
            ),
        ):
            if metrics[metric_name] < quality_gates[gate_name]:
                raise ContractError(f"held-out GRPO {metric_name} is below the baseline-relative floor")
        positive_rate_required, positive_rate_saturated = (
            _bounded_increase_requirement(
                metrics["baseline_eval_positive_reward_group_rate"],
                quality_gates[
                    "min_eval_positive_reward_group_rate_improvement"
                ],
                1.0,
            )
        )
        _verify_bounded_requirement(
            baseline_relative,
            "positive_reward_group_rate",
            positive_rate_required,
            positive_rate_saturated,
            allow_negative=True,
        )
        if (
            positive_group_noninferiority["maximum_regressions"] == 0
            and metrics["eval_positive_reward_group_rate_improvement"]
            < positive_rate_required
        ):
            raise ContractError(
                "held-out GRPO positive-group improvement is below the effective baseline-relative floor"
            )
        paired_evaluation = _paired_grpo_evaluation_evidence(
            run_root,
            groups=evaluation_groups,
            group_size=evaluation_group_size,
            baseline_summary=baseline,
            evaluation_summary=evaluation,
            maximum_p_value=quality_gates[
                "max_multi_seed_paired_prompt_reward_sign_test_p_value"
            ],
        )
        metrics.update(
            {
                "eval_paired_prompt_reward_wins": int(
                    paired_evaluation["wins"]
                ),
                "eval_paired_prompt_reward_losses": int(
                    paired_evaluation["losses"]
                ),
                "eval_paired_prompt_reward_ties": int(
                    paired_evaluation["ties"]
                ),
                "eval_paired_prompt_reward_one_sided_p_value": float(
                    paired_evaluation["one_sided_exact_p_value"]
                ),
            }
        )
        kl_control = _mapping(report.get("kl_control"), "report.kl_control")
        _require_grpo_kl_admissions(
            kl_control, grpo_training_coverage["optimizer_groups"]
        )
        incremental = report.get("incremental_kv")
        if incremental is not None:
            incremental = _mapping(incremental, "report.incremental_kv")
            if _integer(incremental.get("groups"), "incremental_kv.groups") != expected_units:
                raise ContractError("incremental-KV telemetry does not cover the long horizon")
            if _integer(
                incremental.get("host_logit_fallbacks"),
                "incremental_kv.host_logit_fallbacks",
            ) != 0:
                raise ContractError("incremental-KV campaign used host-logit fallback")

    return {
        "report_path": str(report_path),
        "report_sha256": resume_qualifier._sha256(report_path),
        "adapter_model_sha256": trained["adapter_model_sha256"],
        "adapter_tree": trained["files"],
        "metrics": metrics,
        "paired_evaluation": paired_evaluation if task == "grpo" else None,
        "verified_artifacts": verified_artifacts["evidence"],
        "outer_training_report": {
            "path": str(training_report_path),
            "sha256": resume_qualifier._sha256(training_report_path),
        },
        "baseline_evaluation_report": {
            "path": str(expected_baseline_path),
            "sha256": resume_qualifier._sha256(expected_baseline_path),
        },
        "recipe_run_manifest": {
            "path": str(manifest_path),
            "sha256": resume_qualifier._sha256(manifest_path),
        },
    }


def _metric_summary(runs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    names = sorted(_mapping(runs[0].get("quality"), "run.quality").get("metrics", {}))
    summary: dict[str, Any] = {}
    for name in names:
        values = [float(run["quality"]["metrics"][name]) for run in runs]
        summary[name] = {
            "mean": statistics.fmean(values),
            "population_stddev": statistics.pstdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return summary


def qualify(args: argparse.Namespace) -> Mapping[str, Any]:
    binary = resume_qualifier._regular_file(args.binary, "antfly binary", executable=True)
    recipe_path = resume_qualifier._regular_file(args.recipe, "base recipe")
    base = json.loads(json.dumps(_load_json(recipe_path, "base recipe")))
    task = base.get("recipe")
    if task not in ("dpo", "grpo"):
        raise ContractError("base recipe must be DPO or GRPO")
    if _mapping(base.get("execution"), "recipe.execution").get("mode") != "train":
        raise ContractError("base recipe must set execution.mode=train")
    if base.get("backend") != "metal":
        raise ContractError("base recipe must set backend=metal")
    if args.timeout_seconds <= 0:
        raise ContractError("timeout must be positive")
    if args.learning_rate is not None and (
        not math.isfinite(args.learning_rate) or args.learning_rate <= 0
    ):
        raise ContractError("learning rate must be finite and positive")
    compiled_sampling = bool(getattr(args, "compiled_sampling", False))
    if compiled_sampling and task != "grpo":
        raise ContractError("--compiled-sampling is valid only for GRPO")

    model_config = dict(_mapping(base.get("model"), "recipe.model"))
    adapter_config = dict(_mapping(base.get("adapter"), "recipe.adapter"))
    model_value = model_config.get("path")
    adapter_value = adapter_config.get("path")
    if not isinstance(model_value, str) or not isinstance(adapter_value, str):
        raise ContractError("campaign requires pinned model and seed-adapter paths")
    model = resume_qualifier._closed_immutable_path(Path(model_value), "model")
    template_adapter = resume_qualifier._closed_immutable_path(
        Path(adapter_value), "template adapter"
    )
    if not template_adapter.is_dir():
        raise ContractError("template adapter must be a directory")
    template_adapter_spec = _adapter_bootstrap_spec(template_adapter)
    reference_value = model_config.get("reference_path")
    if reference_value is not None:
        if not isinstance(reference_value, str):
            raise ContractError("campaign requires a pinned reference model path")
        reference = resume_qualifier._closed_immutable_path(
            Path(reference_value), "reference model"
        )
        if reference != model:
            raise ContractError("preference campaign requires model and reference paths to match")
        model_config["reference_path"] = str(reference)
    model_config["path"] = str(model)
    adapter_config["path"] = str(template_adapter)
    if model.is_file() and not args.allow_direct_gguf_training:
        raise ContractError("direct GGUF campaign requires --allow-direct-gguf-training")
    if args.allow_direct_gguf_training and not model.is_file():
        raise ContractError("--allow-direct-gguf-training requires a direct GGUF model")
    if compiled_sampling and args.allow_direct_gguf_training:
        raise ContractError(
            "compiled GRPO sampling is not qualified for direct GGUF training"
        )
    train_path = _dataset_path(base, "train")
    eval_path = _dataset_path(base, "eval")
    dataset_config = dict(_mapping(base.get("dataset"), "recipe.dataset"))
    eval_config = dict(_mapping(base.get("eval"), "recipe.eval"))
    if dataset_config.get("train_path") is not None:
        dataset_config["train_path"] = str(train_path)
    if dataset_config.get("path") is not None:
        dataset_config["path"] = str(train_path)
    if eval_config.get("path") is not None:
        eval_config["path"] = str(eval_path)
    if dataset_config.get("eval_path") is not None:
        dataset_config["eval_path"] = str(eval_path)
    base["model"] = model_config
    base["adapter"] = adapter_config
    base["dataset"] = dataset_config
    base["eval"] = eval_config
    resume_qualifier._apply_compiled_sampling_recipe_contract(
        base, task, compiled_sampling
    )
    rows = _jsonl_rows(train_path)
    configured_max = _mapping(base.get("dataset"), "recipe.dataset").get("max_examples")
    examples_per_epoch = min(
        len(rows),
        _integer(configured_max, "dataset.max_examples", 1)
        if configured_max is not None
        else len(rows),
    )
    selected_rows = rows[:examples_per_epoch]
    _require_permutation_capacity(len(selected_rows), len(args.seeds))
    selected_row_multiset_sha256 = _row_multiset_sha256(selected_rows)
    expected_units, minimum_training_units = _long_horizon_units(
        task, examples_per_epoch, args.epochs
    )
    gradient_accumulation_steps = _integer(
        _mapping(base.get("optimizer"), "recipe.optimizer").get(
            "gradient_accumulation_steps", 1
        ),
        "optimizer.gradient_accumulation_steps",
        1,
    )
    expected_optimizer_steps = (
        expected_units + gradient_accumulation_steps - 1
    ) // gradient_accumulation_steps
    expected_evaluation_groups: int | None = None
    positive_group_requirement: float | None = None
    maximum_positive_group_regressions: int | None = None
    if task == "grpo":
        evaluation_rows = _jsonl_rows(eval_path, "evaluation dataset")
        configured_eval_max = eval_config.get("max_examples")
        if configured_eval_max is None:
            configured_eval_max = dataset_config.get("eval_max_examples")
        expected_evaluation_groups = min(
            len(evaluation_rows),
            _integer(configured_eval_max, "eval.max_examples", 1)
            if configured_eval_max is not None
            else len(evaluation_rows),
        )
        (
            positive_group_requirement,
            maximum_positive_group_regressions,
        ) = _grpo_positive_group_requirement(
            expected_evaluation_groups,
            args.min_grpo_eval_positive_reward_group_rate_improvement,
            args.max_grpo_eval_positive_reward_group_regressions,
        )
    quality_gates = (
        {
            "min_eval_accuracy": args.min_dpo_eval_accuracy,
            "max_eval_loss": args.max_dpo_eval_loss,
            "min_eval_reward_margin": args.min_dpo_eval_reward_margin,
            "min_eval_accuracy_improvement": args.min_dpo_eval_accuracy_improvement,
            "min_eval_reward_margin_improvement": args.min_dpo_eval_reward_margin_improvement,
            "min_eval_loss_improvement": args.min_dpo_eval_loss_improvement,
        }
        if task == "dpo"
        else {
            "min_eval_mean_reward": args.min_grpo_eval_mean_reward,
            "min_eval_top_rank_mean_reward": args.min_grpo_eval_top_rank_mean_reward,
            "min_eval_positive_reward_group_rate": args.min_grpo_eval_positive_reward_group_rate,
            "max_eval_kl_loss": args.max_grpo_eval_kl_loss,
            "min_eval_mean_reward_improvement": args.min_grpo_eval_mean_reward_improvement,
            "min_eval_top_rank_mean_reward_improvement": args.min_grpo_eval_top_rank_mean_reward_improvement,
            "min_eval_positive_reward_group_rate_improvement": positive_group_requirement,
            "max_eval_positive_reward_group_regressions": maximum_positive_group_regressions,
            "expected_eval_groups": expected_evaluation_groups,
            "max_multi_seed_paired_prompt_reward_sign_test_p_value": args.max_grpo_paired_prompt_reward_sign_test_p_value,
            "min_optimizer_group_rate": args.min_grpo_optimizer_group_rate,
            "max_kl_rejected_group_rate": args.max_grpo_kl_rejected_group_rate,
        }
    )
    if any(not math.isfinite(value) for value in quality_gates.values()):
        raise ContractError("quality gates must be finite")
    if task == "dpo":
        _probability(quality_gates["min_eval_accuracy"], "minimum DPO eval accuracy")
        _nonnegative(quality_gates["max_eval_loss"], "maximum DPO eval loss")
        _nonnegative(
            quality_gates["min_eval_accuracy_improvement"],
            "minimum DPO eval accuracy improvement",
        )
        _nonnegative(
            quality_gates["min_eval_reward_margin_improvement"],
            "minimum DPO eval reward-margin improvement",
        )
        _nonnegative(
            quality_gates["min_eval_loss_improvement"],
            "minimum DPO eval loss improvement",
        )
        dpo_minimums = dict(
            _mapping(base["eval"].get("dpo_minimums"), "recipe.eval.dpo_minimums")
        )
        dpo_minimums.update(
            {
                "accuracy": quality_gates["min_eval_accuracy"],
                "max_loss": quality_gates["max_eval_loss"],
                "min_accuracy_improvement": quality_gates["min_eval_accuracy_improvement"],
                "min_reward_margin_improvement": quality_gates["min_eval_reward_margin_improvement"],
                "min_loss_improvement": quality_gates["min_eval_loss_improvement"],
            }
        )
        base["eval"]["dpo_minimums"] = dpo_minimums
    else:
        _probability(
            quality_gates["min_eval_positive_reward_group_rate"],
            "minimum GRPO positive-reward group rate",
        )
        _nonnegative(quality_gates["max_eval_kl_loss"], "maximum GRPO eval KL loss")
        for name in (
            "min_eval_mean_reward_improvement",
            "min_eval_top_rank_mean_reward_improvement",
        ):
            _nonnegative(quality_gates[name], name)
        _finite(
            quality_gates["min_eval_positive_reward_group_rate_improvement"],
            "minimum GRPO positive-group improvement",
        )
        _integer(
            quality_gates["max_eval_positive_reward_group_regressions"],
            "maximum GRPO positive-group regressions",
        )
        _integer(
            quality_gates["expected_eval_groups"],
            "expected GRPO evaluation groups",
            1,
        )
        _probability(
            quality_gates[
                "max_multi_seed_paired_prompt_reward_sign_test_p_value"
            ],
            "maximum multi-seed GRPO paired prompt reward sign-test p-value",
        )
        if (
            quality_gates[
                "max_multi_seed_paired_prompt_reward_sign_test_p_value"
            ]
            == 0.0
        ):
            raise ContractError(
                "maximum multi-seed GRPO paired prompt reward sign-test p-value must be positive"
            )
        if (
            quality_gates[
                "max_multi_seed_paired_prompt_reward_sign_test_p_value"
            ]
            > MAX_GRPO_PAIRED_SIGN_TEST_P_VALUE
        ):
            raise ContractError(
                "GRPO paired sign-test p-value ceiling cannot relax the production maximum"
            )
        _probability(
            quality_gates["min_optimizer_group_rate"],
            "minimum GRPO optimizer-group rate",
        )
        if (
            quality_gates["min_optimizer_group_rate"]
            < MIN_GRPO_OPTIMIZER_GROUP_RATE
        ):
            raise ContractError(
                "GRPO optimizer-group floor cannot relax the production minimum"
            )
        _probability(
            quality_gates["max_kl_rejected_group_rate"],
            "maximum GRPO KL-rejected-group rate",
        )
        if (
            quality_gates["max_kl_rejected_group_rate"]
            > MAX_GRPO_KL_REJECTED_GROUP_RATE
        ):
            raise ContractError(
                "GRPO KL-rejected-group ceiling cannot relax the production maximum"
            )
        grpo_minimums = dict(
            _mapping(base["eval"].get("grpo_minimums"), "recipe.eval.grpo_minimums")
        )
        grpo_minimums.update(
            {
                "mean_reward": quality_gates["min_eval_mean_reward"],
                "top_rank_mean_reward": quality_gates["min_eval_top_rank_mean_reward"],
                "positive_reward_group_rate": quality_gates["min_eval_positive_reward_group_rate"],
                "max_kl_loss": quality_gates["max_eval_kl_loss"],
                "min_mean_reward_improvement": quality_gates["min_eval_mean_reward_improvement"],
                "min_top_rank_mean_reward_improvement": quality_gates["min_eval_top_rank_mean_reward_improvement"],
                "min_positive_reward_group_rate_improvement": quality_gates["min_eval_positive_reward_group_rate_improvement"],
            }
        )
        base["eval"]["grpo_minimums"] = grpo_minimums

    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists():
        raise ContractError(f"output directory already exists: {output_root}")
    immutable_roots = {
        "binary": binary,
        "base_recipe": recipe_path,
        "model": model,
        "template_adapter": template_adapter,
        "train_dataset": train_path,
        "eval_dataset": eval_path,
        "quality_qualifier": SCRIPT_PATH,
        "resume_qualifier": RESUME_QUALIFIER_PATH,
        "environment_policy": resume_qualifier.ENVIRONMENT_POLICY_PATH,
    }
    for name, path in immutable_roots.items():
        if resume_qualifier._paths_overlap(output_root, path):
            raise ContractError(f"output directory overlaps immutable {name}: {path}")
    snapshots_before = {
        name: resume_qualifier._tree_snapshot(path)
        for name, path in immutable_roots.items()
    }
    output_root.mkdir(parents=True)
    inputs_root = output_root / "inputs"
    recipes_root = output_root / "recipes"
    runs_root = output_root / "runs"
    inputs_root.mkdir()
    recipes_root.mkdir()
    runs_root.mkdir()

    env = resume_qualifier._strict_environment()
    effective_contract_env = dict(resume_qualifier.STRICT_METAL_ENV)
    if compiled_sampling:
        env[resume_qualifier.COMPILED_GRPO_SAMPLING_ENV] = "1"
        effective_contract_env[
            resume_qualifier.COMPILED_GRPO_SAMPLING_ENV
        ] = "1"

    runs: list[dict[str, Any]] = []
    initialized_adapters: list[dict[str, Any]] = []
    dataset_digests: set[str] = set()
    multi_seed_paired_evaluation: dict[str, Any] | None = None
    active_seed: int | None = None
    active_run_root: Path | None = None
    try:
        initialized_adapter_digests: set[str] = set()
        initialized_adapter_by_seed: dict[int, dict[str, Any]] = {}
        for seed in args.seeds:
            initialized = _bootstrap_seed_adapter(
                binary,
                model,
                template_adapter_spec,
                seed,
                inputs_root / f"adapter-seed-{seed}",
                env,
                inputs_root / f"bootstrap-seed-{seed}",
                args.timeout_seconds,
            )
            digest = initialized["adapter_model_sha256"]
            if digest in initialized_adapter_digests:
                raise ContractError(
                    "two initialization seeds produced the same adapter checkpoint"
                )
            initialized_adapter_digests.add(digest)
            initialized_adapter_by_seed[seed] = initialized
            initialized_adapters.append(initialized)

        for seed in args.seeds:
            seed_name = f"seed-{seed}"
            seeded_train = inputs_root / f"{seed_name}.jsonl"
            _write_bytes(seeded_train, _permuted_rows(selected_rows, seed))
            if _row_multiset_sha256(_jsonl_rows(seeded_train)) != selected_row_multiset_sha256:
                raise ContractError("seeded dataset changed the selected training-row multiset")
            dataset_digest = resume_qualifier._sha256(seeded_train)
            if dataset_digest in dataset_digests:
                raise ContractError("two seeds produced the same training-row order")
            dataset_digests.add(dataset_digest)
            run_root = runs_root / seed_name
            recipe = _variant(
                base,
                task,
                seed,
                Path(initialized_adapter_by_seed[seed]["path"]),
                seeded_train,
                run_root,
                args.epochs,
                args.learning_rate,
                args.allow_direct_gguf_training,
            )
            variant_path = recipes_root / f"{seed_name}.json"
            _write_json(variant_path, recipe)
            recipe_digest = resume_qualifier._sha256(variant_path)
            command = [str(binary), "inference", "finetune", "run", str(variant_path)]
            active_seed = seed
            active_run_root = run_root
            execution = _run(command, env, runs_root / seed_name, args.timeout_seconds)
            if resume_qualifier._sha256(seeded_train) != dataset_digest:
                raise ContractError("training command changed its seeded dataset")
            if resume_qualifier._sha256(variant_path) != recipe_digest:
                raise ContractError("training command changed its recipe")
            quality = _validate_run(
                task,
                run_root,
                seed,
                expected_units,
                expected_optimizer_steps,
                gradient_accumulation_steps,
                args.epochs,
                seeded_train,
                Path(initialized_adapter_by_seed[seed]["path"]),
                initialized_adapter_by_seed[seed],
                quality_gates,
                compiled_sampling,
            )
            runs.append(
                {
                    "seed": seed,
                    "seed_dimension": "adapter-initialization-plus-typed-training-seed-plus-input-permutation-plus-epoch-order",
                    "initialized_adapter": initialized_adapter_by_seed[seed],
                    "training_dataset_path": str(seeded_train),
                    "training_dataset_sha256": dataset_digest,
                    "recipe_path": str(variant_path),
                    "recipe_sha256": recipe_digest,
                    "execution": execution,
                    "quality": quality,
                }
            )
            active_seed = None
            active_run_root = None

        if task == "grpo":
            assert expected_evaluation_groups is not None
            multi_seed_paired_evaluation = _multi_seed_grpo_evaluation_evidence(
                runs,
                expected_groups=expected_evaluation_groups,
                maximum_p_value=quality_gates[
                    "max_multi_seed_paired_prompt_reward_sign_test_p_value"
                ],
            )
            if multi_seed_paired_evaluation["passed"] is not True:
                raise ContractError(
                    "held-out GRPO prompt-averaged multi-seed reward improvement "
                    "did not pass the production exact sign-test gate"
                )

        adapter_digests = {run["quality"]["adapter_model_sha256"] for run in runs}
        if len(adapter_digests) != len(runs):
            raise ContractError("distinct training seeds produced duplicate final adapters")
        for name, path in immutable_roots.items():
            if resume_qualifier._tree_snapshot(path) != snapshots_before[name]:
                raise ContractError(f"immutable {name} changed during campaign")
        report = {
            "schema_version": SCHEMA_VERSION,
            "status": "pass",
            "task": task,
            "contract": {
                "backend": "metal",
                "seed_dimension": "adapter-initialization-plus-typed-training-seed-plus-input-permutation-plus-epoch-order",
                "seeds": args.seeds,
                "epochs": args.epochs,
                "examples_per_epoch": examples_per_epoch,
                "expected_training_units": expected_units,
                "minimum_training_units": minimum_training_units,
                "selected_training_row_multiset_sha256": selected_row_multiset_sha256,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "maximum_optimizer_steps": expected_optimizer_steps,
                "optimizer_step_policy": (
                    "ceil(admitted_optimizer_groups/gradient_accumulation_steps)"
                    if task == "grpo"
                    else "ceil(examples/gradient_accumulation_steps)"
                ),
                "allow_direct_gguf_training": args.allow_direct_gguf_training,
                "compiled_sampling": compiled_sampling,
                "quality_gates": quality_gates,
                "grpo_paired_statistical_policy": (
                    {
                        "per_seed_requirement": "wins-greater-than-losses",
                        "significance_requirement": "one-sided-exact-sign-test",
                        "significance_experimental_unit": "evaluation-prompt",
                        "seed_aggregation": "arithmetic-mean-of-prompt-group-reward-deltas",
                        "completion_aggregation": "sum-within-prompt-group",
                    }
                    if task == "grpo"
                    else None
                ),
                "bounded_improvement_policy": "requested-minimum-capped-by-valid-metric-headroom-with-non-regression-at-boundary",
                "initialization": {
                    "producer": "antfly inference finetune adapter bootstrap gemma4",
                    "manifest_schema": "antfly_gemma4_finetune/v3",
                    "template_adapter_config": template_adapter_spec,
                    "distinct_adapter_checkpoints": len(initialized_adapter_digests),
                },
                "environment_policy_sha256": resume_qualifier.ENVIRONMENT_POLICY_SHA256,
                "strict_metal_environment": effective_contract_env,
                "sanitized_environment_prefixes": list(
                    resume_qualifier.SANITIZED_ENV_PREFIXES
                ),
                "sanitized_environment_names": sorted(
                    resume_qualifier.SANITIZED_ENV_NAMES
                ),
            },
            "inputs": {
                name: {
                    "path": str(path),
                    "snapshot_sha256": resume_qualifier._snapshot_digest(
                        snapshots_before[name]
                    ),
                    "sha256": resume_qualifier._sha256(path) if path.is_file() else None,
                }
                for name, path in immutable_roots.items()
            },
            "runs": runs,
            "initialized_adapters": initialized_adapters,
            "quality_summary": _metric_summary(runs),
            "multi_seed_paired_evaluation": multi_seed_paired_evaluation,
        }
        _write_json(output_root / "campaign_report.json", report)
        return report
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "fail",
            "task": task,
            "error": str(exc),
            "contract": {
                "backend": "metal",
                "seeds": args.seeds,
                "epochs": args.epochs,
                "examples_per_epoch": examples_per_epoch,
                "expected_training_units": expected_units,
                "minimum_training_units": minimum_training_units,
                "selected_training_row_multiset_sha256": selected_row_multiset_sha256,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "maximum_optimizer_steps": expected_optimizer_steps,
                "optimizer_step_policy": (
                    "ceil(admitted_optimizer_groups/gradient_accumulation_steps)"
                    if task == "grpo"
                    else "ceil(examples/gradient_accumulation_steps)"
                ),
                "allow_direct_gguf_training": args.allow_direct_gguf_training,
                "compiled_sampling": compiled_sampling,
                "quality_gates": quality_gates,
                "grpo_paired_statistical_policy": (
                    {
                        "per_seed_requirement": "wins-greater-than-losses",
                        "significance_requirement": "one-sided-exact-sign-test",
                        "significance_experimental_unit": "evaluation-prompt",
                        "seed_aggregation": "arithmetic-mean-of-prompt-group-reward-deltas",
                        "completion_aggregation": "sum-within-prompt-group",
                    }
                    if task == "grpo"
                    else None
                ),
                "environment_policy_sha256": resume_qualifier.ENVIRONMENT_POLICY_SHA256,
                "strict_metal_environment": effective_contract_env,
            },
            "inputs": {
                name: {
                    "path": str(path),
                    "snapshot_sha256": resume_qualifier._snapshot_digest(
                        snapshots_before[name]
                    ),
                    "sha256": resume_qualifier._sha256(path)
                    if path.is_file()
                    else None,
                }
                for name, path in immutable_roots.items()
            },
            "initialized_adapters": initialized_adapters,
            "completed_runs": runs,
            "multi_seed_paired_evaluation": multi_seed_paired_evaluation,
        }
        if active_seed is not None and active_run_root is not None:
            failure["failed_run"] = _failed_run_evidence(
                task, active_seed, active_run_root
            )
        try:
            _write_json(output_root / "campaign_report.json", failure)
        except Exception:
            pass
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("17,42,991"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--allow-direct-gguf-training", action="store_true")
    parser.add_argument(
        "--compiled-sampling",
        action="store_true",
        help=(
            "qualify the default-off compiled multi-token GRPO sampler; "
            "incremental KV and direct GGUF are forced out of the recipe contract"
        ),
    )
    parser.add_argument("--min-dpo-eval-accuracy", type=float, default=0.4)
    parser.add_argument("--max-dpo-eval-loss", type=float, default=1.0)
    parser.add_argument("--min-dpo-eval-reward-margin", type=float, default=0.0)
    parser.add_argument("--min-dpo-eval-accuracy-improvement", type=float, default=1e-6)
    parser.add_argument(
        "--min-dpo-eval-reward-margin-improvement", type=float, default=1e-6
    )
    parser.add_argument("--min-dpo-eval-loss-improvement", type=float, default=1e-6)
    parser.add_argument("--min-grpo-eval-mean-reward", type=float, default=0.125)
    parser.add_argument(
        "--min-grpo-eval-top-rank-mean-reward", type=float, default=0.125
    )
    parser.add_argument(
        "--min-grpo-eval-positive-reward-group-rate", type=float, default=0.75
    )
    parser.add_argument("--max-grpo-eval-kl-loss", type=float, default=0.004)
    parser.add_argument(
        "--min-grpo-eval-mean-reward-improvement", type=float, default=1e-6
    )
    parser.add_argument(
        "--min-grpo-eval-top-rank-mean-reward-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--min-grpo-eval-positive-reward-group-rate-improvement",
        type=float,
        default=None,
        help=(
            "require a non-negative positive-group rate improvement instead of "
            "the default one-group non-inferiority contract"
        ),
    )
    parser.add_argument(
        "--max-grpo-eval-positive-reward-group-regressions",
        type=int,
        default=None,
        help=(
            "maximum net positive-reward groups that may regress; at most one "
            "is permitted and one is used by default"
        ),
    )
    parser.add_argument(
        "--max-grpo-paired-prompt-reward-sign-test-p-value",
        type=float,
        default=MAX_GRPO_PAIRED_SIGN_TEST_P_VALUE,
        help=(
            "maximum one-sided exact sign-test p-value for prompt-clustered "
            "baseline/final GRPO reward changes averaged over training seeds; "
            "each seed must separately have more prompt wins than losses and "
            "values above the production ceiling are rejected (default: 0.05)"
        ),
    )
    parser.add_argument(
        "--min-grpo-optimizer-group-rate",
        type=float,
        default=MIN_GRPO_OPTIMIZER_GROUP_RATE,
        help=(
            "minimum fraction of the logical GRPO horizon that must contain "
            "reward variation and reach the optimizer; values below the "
            "production floor are rejected (default: 0.25)"
        ),
    )
    parser.add_argument(
        "--max-grpo-kl-rejected-group-rate",
        type=float,
        default=MAX_GRPO_KL_REJECTED_GROUP_RATE,
        help=(
            "maximum fraction of logical GRPO groups that may be safely "
            "skipped by the train-time KL budget; values above the production "
            "ceiling are rejected (default: 0.01)"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = qualify(parse_args(argv))
    except (ContractError, resume_qualifier.ContractError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        f"PASS: {len(report['runs'])}-seed {report['task'].upper()} long-horizon "
        f"quality campaign; report={args_path(report)}"
    )
    return 0


def args_path(report: Mapping[str, Any]) -> str:
    first = report["runs"][0]["recipe_path"]
    return str(Path(first).parents[1] / "campaign_report.json")


if __name__ == "__main__":
    raise SystemExit(main())
