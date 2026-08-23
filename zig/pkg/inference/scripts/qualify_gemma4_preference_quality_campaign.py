#!/usr/bin/env python3
"""Run a pinned multi-seed, long-horizon Gemma4 preference quality gate.

Each seed is bound into the typed trainer/RNG contract and defines a
deterministic permutation of the exact same training JSONL rows. The model,
seed adapter, held-out dataset, non-seed optimizer settings, and binary remain
fixed. Every seed must complete real optimizer-backed Metal training and pass
the recipe's held-out quality gates. The initialized adapter remains fixed, so
this measures training-seed and data-order robustness rather than pretending
to be independent model-initialization coverage.
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


SCHEMA_VERSION = "antfly_gemma4_preference_quality_campaign/v1"


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


def _variant(
    base: Mapping[str, Any],
    task: str,
    seed: int,
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
    expected_epochs: int,
    expected_train_path: Path,
    seed_adapter_sha256: str,
    quality_gates: Mapping[str, float],
) -> dict[str, Any]:
    report_path = run_root / f"{task}_report.json"
    report = _load_json(report_path, f"{task} report")
    if report.get("schema_version") != f"antfly_inference_finetune_{task}_report/v6":
        raise ContractError(f"{task} report has unsupported schema")
    if report.get("execution_mode") != "train" or report.get("policy_backend") != "metal":
        raise ContractError("campaign run was not optimizer-backed Metal training")
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
    if _integer(report.get("optimizer_steps"), "report.optimizer_steps", 1) != expected_optimizer_steps:
        raise ContractError("report.optimizer_steps does not cover the exact long horizon")
    if _integer(report.get("micro_batch_steps"), "report.micro_batch_steps", 1) < expected_units:
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
    trained = resume_qualifier._adapter_tree_evidence(
        run_root / "adapter-trained", "trained adapter"
    )
    if trained["adapter_model_sha256"] == seed_adapter_sha256:
        raise ContractError("trained adapter is byte-identical to the seed adapter")

    common_metrics = {
        "train_loss": _finite(report.get("loss"), "report.loss"),
    }
    if task == "dpo":
        metrics = {
            **common_metrics,
            "eval_loss": _nonnegative(evaluation.get("loss"), "evaluation.loss"),
            "train_accuracy": _probability(report.get("accuracy"), "report.accuracy"),
            "eval_accuracy": _probability(evaluation.get("accuracy"), "evaluation.accuracy"),
            "eval_reward_margin": _finite(
                evaluation.get("mean_reward_margin"), "evaluation.mean_reward_margin"
            ),
        }
        if metrics["eval_accuracy"] < quality_gates["min_eval_accuracy"]:
            raise ContractError("held-out DPO accuracy is below the campaign floor")
        if metrics["eval_loss"] > quality_gates["max_eval_loss"]:
            raise ContractError("held-out DPO loss exceeds the campaign ceiling")
        if metrics["eval_reward_margin"] < quality_gates["min_eval_reward_margin"]:
            raise ContractError("held-out DPO reward margin is below the campaign floor")
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
        }
        if metrics["eval_mean_reward"] < quality_gates["min_eval_mean_reward"]:
            raise ContractError("held-out GRPO mean reward is below the campaign floor")
        if metrics["eval_top_rank_mean_reward"] < quality_gates["min_eval_top_rank_mean_reward"]:
            raise ContractError("held-out GRPO top-rank reward is below the campaign floor")
        if metrics["eval_positive_reward_group_rate"] < quality_gates["min_eval_positive_reward_group_rate"]:
            raise ContractError("held-out GRPO positive-group rate is below the campaign floor")
        if metrics["eval_kl_loss"] > quality_gates["max_eval_kl_loss"]:
            raise ContractError("held-out GRPO KL loss exceeds the campaign ceiling")
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

    model_config = dict(_mapping(base.get("model"), "recipe.model"))
    adapter_config = dict(_mapping(base.get("adapter"), "recipe.adapter"))
    model_value = model_config.get("path")
    adapter_value = adapter_config.get("path")
    if not isinstance(model_value, str) or not isinstance(adapter_value, str):
        raise ContractError("campaign requires pinned model and seed-adapter paths")
    model = resume_qualifier._closed_immutable_path(Path(model_value), "model")
    adapter = resume_qualifier._closed_immutable_path(Path(adapter_value), "seed adapter")
    if not adapter.is_dir():
        raise ContractError("seed adapter must be a directory")
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
    adapter_config["path"] = str(adapter)
    if model.is_file() and not args.allow_direct_gguf_training:
        raise ContractError("direct GGUF campaign requires --allow-direct-gguf-training")
    if args.allow_direct_gguf_training and not model.is_file():
        raise ContractError("--allow-direct-gguf-training requires a direct GGUF model")
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
        }
        if task == "dpo"
        else {
            "min_eval_mean_reward": args.min_grpo_eval_mean_reward,
            "min_eval_top_rank_mean_reward": args.min_grpo_eval_top_rank_mean_reward,
            "min_eval_positive_reward_group_rate": args.min_grpo_eval_positive_reward_group_rate,
            "max_eval_kl_loss": args.max_grpo_eval_kl_loss,
        }
    )
    if any(not math.isfinite(value) for value in quality_gates.values()):
        raise ContractError("quality gates must be finite")
    if task == "dpo":
        _probability(quality_gates["min_eval_accuracy"], "minimum DPO eval accuracy")
        _nonnegative(quality_gates["max_eval_loss"], "maximum DPO eval loss")
    else:
        _probability(
            quality_gates["min_eval_positive_reward_group_rate"],
            "minimum GRPO positive-reward group rate",
        )
        _nonnegative(quality_gates["max_eval_kl_loss"], "maximum GRPO eval KL loss")

    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists():
        raise ContractError(f"output directory already exists: {output_root}")
    immutable_roots = {
        "binary": binary,
        "base_recipe": recipe_path,
        "model": model,
        "adapter": adapter,
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
    seed_adapter = resume_qualifier._adapter_tree_evidence(adapter, "seed adapter")

    output_root.mkdir(parents=True)
    inputs_root = output_root / "inputs"
    recipes_root = output_root / "recipes"
    runs_root = output_root / "runs"
    inputs_root.mkdir()
    recipes_root.mkdir()
    runs_root.mkdir()

    env = resume_qualifier._strict_environment()

    runs: list[dict[str, Any]] = []
    dataset_digests: set[str] = set()
    try:
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
                args.epochs,
                seeded_train,
                seed_adapter["adapter_model_sha256"],
                quality_gates,
            )
            runs.append(
                {
                    "seed": seed,
                    "seed_dimension": "typed-training-seed-plus-deterministic-row-order",
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
                "seed_dimension": "typed-training-seed-plus-deterministic-row-order",
                "seeds": args.seeds,
                "epochs": args.epochs,
                "examples_per_epoch": examples_per_epoch,
                "expected_training_units": expected_units,
                "selected_training_row_multiset_sha256": selected_row_multiset_sha256,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "expected_optimizer_steps": expected_optimizer_steps,
                "allow_direct_gguf_training": args.allow_direct_gguf_training,
                "quality_gates": quality_gates,
                "environment_policy_sha256": resume_qualifier.ENVIRONMENT_POLICY_SHA256,
                "strict_metal_environment": resume_qualifier.STRICT_METAL_ENV,
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
    parser.add_argument("--min-dpo-eval-accuracy", type=float, default=0.4)
    parser.add_argument("--max-dpo-eval-loss", type=float, default=1.0)
    parser.add_argument("--min-dpo-eval-reward-margin", type=float, default=0.0)
    parser.add_argument("--min-grpo-eval-mean-reward", type=float, default=0.125)
    parser.add_argument(
        "--min-grpo-eval-top-rank-mean-reward", type=float, default=0.125
    )
    parser.add_argument(
        "--min-grpo-eval-positive-reward-group-rate", type=float, default=0.75
    )
    parser.add_argument("--max-grpo-eval-kl-loss", type=float, default=0.004)
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
