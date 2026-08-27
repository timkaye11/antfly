#!/usr/bin/env python3
"""Run a pinned multi-seed, long-horizon Gemma4 preference quality gate.

Each seed is bound into three independent, reproducible dimensions: Gemma4
LoRA initialization, the typed trainer/RNG contract, and a deterministic
permutation of the exact same training JSONL rows. The campaign bootstraps a
fresh seed-bound adapter for every run from the pinned model and template
adapter configuration. Every seed must complete real optimizer-backed Metal
training and pass the recipe's held-out quality gates.
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


SCHEMA_VERSION = "antfly_gemma4_preference_quality_campaign/v2"


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
) -> None:
    reported_minimum = _nonnegative(
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


def _jsonl_rows(path: Path) -> list[bytes]:
    try:
        raw_rows = path.read_bytes().splitlines()
    except OSError as exc:
        raise ContractError(f"training dataset: {exc}") from exc
    rows = [row for row in raw_rows if row.strip()]
    if len(rows) < 2:
        raise ContractError("multi-seed campaign requires at least two non-empty training rows")
    for index, row in enumerate(rows, 1):
        try:
            value = json.loads(row)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"training dataset row {index}: invalid JSON: {exc}") from exc
        if not isinstance(value, Mapping):
            raise ContractError(f"training dataset row {index}: expected object")
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
    expected_schema = f"antfly_inference_finetune_{task}_report/v7"
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
    if task == "grpo":
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
        reported_fraction = _probability(
            report.get("frac_reward_zero_std"), "report.frac_reward_zero_std"
        )
        if not math.isclose(
            reported_fraction,
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
        metrics = {
            **common_metrics,
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
            "eval_positive_reward_group_rate_required_improvement": _nonnegative(
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
        )
        if (
            metrics["eval_positive_reward_group_rate_improvement"]
            < positive_rate_required
        ):
            raise ContractError(
                "held-out GRPO positive-group improvement is below the effective baseline-relative floor"
            )
        kl_control = _mapping(report.get("kl_control"), "report.kl_control")
        if _integer(kl_control.get("admitted_groups"), "kl_control.admitted_groups") != expected_units:
            raise ContractError("KL controller did not admit the complete long horizon")
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
    if args.epochs < 4:
        raise ContractError("long-horizon campaign requires at least four epochs")
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
    expected_units = examples_per_epoch * args.epochs
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
            "min_eval_positive_reward_group_rate_improvement": args.min_grpo_eval_positive_reward_group_rate_improvement,
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
            "min_eval_positive_reward_group_rate_improvement",
        ):
            _nonnegative(quality_gates[name], name)
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
                    "seed_dimension": "adapter-initialization-plus-typed-training-seed-plus-deterministic-row-order",
                    "initialized_adapter": initialized_adapter_by_seed[seed],
                    "training_dataset_path": str(seeded_train),
                    "training_dataset_sha256": dataset_digest,
                    "recipe_path": str(variant_path),
                    "recipe_sha256": recipe_digest,
                    "execution": execution,
                    "quality": quality,
                }
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
                "seed_dimension": "adapter-initialization-plus-typed-training-seed-plus-deterministic-row-order",
                "seeds": args.seeds,
                "epochs": args.epochs,
                "examples_per_epoch": examples_per_epoch,
                "expected_training_units": expected_units,
                "selected_training_row_multiset_sha256": selected_row_multiset_sha256,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "expected_optimizer_steps": expected_optimizer_steps,
                "allow_direct_gguf_training": args.allow_direct_gguf_training,
                "compiled_sampling": compiled_sampling,
                "quality_gates": quality_gates,
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
        }
        _write_json(output_root / "campaign_report.json", report)
        return report
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "fail",
            "task": task,
            "error": str(exc),
            "initialized_adapters": initialized_adapters,
            "completed_runs": runs,
        }
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
        default=1e-6,
    )
    parser.add_argument(
        "--min-grpo-eval-positive-reward-group-rate-improvement",
        type=float,
        default=1e-6,
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
