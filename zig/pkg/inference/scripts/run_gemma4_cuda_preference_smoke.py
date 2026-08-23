#!/usr/bin/env python3
"""Run bounded, fail-closed Gemma4 CUDA DPO/GRPO checkpoint smokes.

The runner deliberately does not download or convert checkpoints. It accepts
mounted model directories, inspects only config/index/safetensor headers, and
then delegates all model semantics and training to antfly-inference. When both
objectives are selected, one process and admitted model are reused per model
and repetition; --isolated-processes retains the cold-start control path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Iterable


MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SUPPORTED_RANK2_DTYPES = {"BF16", "F32"}
MATCHED_BENCHMARK_PROTOCOL = {"cold": 1, "first": 1, "warmup": 3, "measured": 20}
MATCHED_BENCHMARK_UPDATES = sum(MATCHED_BENCHMARK_PROTOCOL.values())
PREFERENCE_COMPLETION_EVIDENCE_FILE_NAME = "antfly_preference_run_report.json"
MATCHED_DPO_PROMPT_IDS = [
    2, 105, 2364, 107, 7925, 607, 886, 3658, 236787, 11262, 653,
    951, 236881, 106, 107, 105, 4368, 107, 100, 45518, 107, 101,
]
MATCHED_DPO_CHOSEN_IDS = [4443]
MATCHED_DPO_REJECTED_IDS = [1904]
MATCHED_GRPO_PAYLOAD_IDS = [818, 5279, 529, 7001, 563]
# Both TRL's non-conversational GRPO path and Antfly's already-rendered prompt
# path feed this exact tokenizer output to the decoder without a chat wrapper.
MATCHED_GRPO_PROMPT_IDS = [*MATCHED_GRPO_PAYLOAD_IDS]
RANKED_FIRST_REWARD_TARGET = "__ranked_first__"


class QualificationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    label: str
    path: pathlib.Path


@dataclass(frozen=True)
class ModelPreflight:
    label: str
    path: str
    artifact_kind: str
    shard_count: int
    total_checkpoint_bytes: int
    tensor_count: int
    dtypes: list[str]
    rank2_dtypes: list[str]
    config_sha256: str
    topology: dict[str, int | str | None]


@dataclass(frozen=True)
class PreparedCase:
    objective: str
    run_dir: pathlib.Path
    recipe_path: pathlib.Path
    initial_adapter_dir: pathlib.Path


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_assignment(raw: str, option: str) -> tuple[str, str]:
    label, separator, value = raw.partition("=")
    if not separator or not LABEL_RE.fullmatch(label) or not value:
        raise QualificationError(f"{option} must use LABEL=VALUE with a safe, non-empty label: {raw!r}")
    return label, value


def parse_unique_assignments(values: Iterable[str], option: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for raw in values:
        label, value = parse_assignment(raw, option)
        if label in parsed:
            raise QualificationError(f"duplicate {option} label: {label}")
        parsed[label] = value
    return parsed


def safetensors_header(path: pathlib.Path) -> dict[str, Any]:
    try:
        with path.open("rb") as source:
            size_raw = source.read(8)
            if len(size_raw) != 8:
                raise QualificationError(f"truncated safetensors header length: {path}")
            header_size = int.from_bytes(size_raw, "little")
            if header_size <= 0 or header_size > MAX_SAFETENSORS_HEADER_BYTES:
                raise QualificationError(f"invalid safetensors header size {header_size}: {path}")
            raw = source.read(header_size)
            if len(raw) != header_size:
                raise QualificationError(f"truncated safetensors header: {path}")
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot inspect safetensors header {path}: {error}") from error
    if not isinstance(value, dict):
        raise QualificationError(f"safetensors header must be an object: {path}")
    return value


def checkpoint_shards(model_dir: pathlib.Path) -> tuple[str, list[pathlib.Path]]:
    single = model_dir / "model.safetensors"
    index = model_dir / "model.safetensors.index.json"
    if single.is_file():
        return "safetensors", [single]
    if index.is_file():
        try:
            parsed = json.loads(index.read_text(encoding="utf-8"))
            weight_map = parsed["weight_map"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise QualificationError(f"invalid sharded safetensors index {index}: {error}") from error
        if not isinstance(weight_map, dict) or not weight_map:
            raise QualificationError(f"empty sharded safetensors weight_map: {index}")
        resolved_root = model_dir.resolve()
        shards: list[pathlib.Path] = []
        shard_names: set[str] = set()
        for name in weight_map.values():
            if not isinstance(name, str) or not name:
                raise QualificationError(f"invalid shard name in {index}")
            shard_names.add(name)
        for name in sorted(shard_names):
            shard = (model_dir / name).resolve()
            try:
                shard.relative_to(resolved_root)
            except ValueError as error:
                raise QualificationError(f"shard escapes model directory: {name}") from error
            if not shard.is_file():
                raise QualificationError(f"missing safetensors shard: {shard}")
            shards.append(shard)
        return "sharded_safetensors", shards

    ggufs = sorted(model_dir.glob("*.gguf"))
    if ggufs:
        names = ", ".join(path.name for path in ggufs[:3])
        raise QualificationError(
            f"{model_dir} selects packed GGUF deployment weights ({names}); "
            "Gemma4 CUDA preference training requires native BF16 safetensors and packed QLoRA remains fail-closed"
        )
    raise QualificationError(f"no model.safetensors or model.safetensors.index.json in {model_dir}")


def config_topology(config: dict[str, Any]) -> dict[str, int | str | None]:
    text = config.get("text_config")
    if not isinstance(text, dict):
        text = config

    def value(name: str) -> int | str | None:
        candidate = text.get(name)
        if isinstance(candidate, (int, str)) and not isinstance(candidate, bool):
            return candidate
        return None

    return {
        "model_type": config.get("model_type") if isinstance(config.get("model_type"), str) else None,
        "hidden_size": value("hidden_size"),
        "num_hidden_layers": value("num_hidden_layers"),
        "num_attention_heads": value("num_attention_heads"),
        "num_key_value_heads": value("num_key_value_heads"),
        "head_dim": value("head_dim") or value("attention_head_dim"),
        "intermediate_size": value("intermediate_size"),
        "vocab_size": value("vocab_size"),
    }


def inspect_model(spec: ModelSpec) -> ModelPreflight:
    model_dir = spec.path.resolve()
    if not model_dir.is_dir():
        if model_dir.is_file() and model_dir.suffix.lower() == ".gguf":
            raise QualificationError(
                f"{model_dir} is packed GGUF; Gemma4 CUDA preference training requires a native BF16 safetensors directory"
            )
        raise QualificationError(f"model path is not a directory: {model_dir}")

    # Resolve the selected weight artifact first. A GGUF-only deployment
    # bundle often omits training-side config/tokenizer files; report the
    # decisive packed-weight incompatibility instead of a secondary sidecar
    # omission.
    artifact_kind, shards = checkpoint_shards(model_dir)

    config_path = model_dir / "config.json"
    try:
        config_bytes = config_path.read_bytes()
        config = json.loads(config_bytes)
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot load model config {config_path}: {error}") from error
    if not isinstance(config, dict):
        raise QualificationError(f"model config must be an object: {config_path}")
    model_type = config.get("model_type")
    normalized_model_type = model_type.lower().replace("_", "") if isinstance(model_type, str) else ""
    if not normalized_model_type.startswith("gemma4") or "assistant" in normalized_model_type:
        raise QualificationError(f"expected Gemma4 model_type in {config_path}, found {model_type!r}")

    tokenizer_present = any(
        (model_dir / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "spiece.model", "tokenizer_config.json")
    )
    if not tokenizer_present:
        raise QualificationError(f"no tokenizer artifact found in {model_dir}")

    dtypes: set[str] = set()
    rank2_dtypes: set[str] = set()
    tensor_count = 0
    for shard in shards:
        header = safetensors_header(shard)
        for name, metadata in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(metadata, dict):
                raise QualificationError(f"invalid tensor metadata {name!r} in {shard}")
            dtype = metadata.get("dtype")
            shape = metadata.get("shape")
            if not isinstance(dtype, str) or not isinstance(shape, list):
                raise QualificationError(f"incomplete tensor metadata {name!r} in {shard}")
            dtypes.add(dtype)
            tensor_count += 1
            if len(shape) == 2:
                rank2_dtypes.add(dtype)

    unsupported = rank2_dtypes - SUPPORTED_RANK2_DTYPES
    if unsupported:
        raise QualificationError(
            f"unsupported rank-2 stored-weight dtype(s) for strict CUDA backward in {model_dir}: "
            f"{', '.join(sorted(unsupported))}"
        )
    if "BF16" not in rank2_dtypes:
        raise QualificationError(f"checkpoint has no rank-2 BF16 weights and is not a BF16 qualification artifact: {model_dir}")

    return ModelPreflight(
        label=spec.label,
        path=str(model_dir),
        artifact_kind=artifact_kind,
        shard_count=len(shards),
        total_checkpoint_bytes=sum(path.stat().st_size for path in shards),
        tensor_count=tensor_count,
        dtypes=sorted(dtypes),
        rank2_dtypes=sorted(rank2_dtypes),
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        topology=config_topology(config),
    )


def recipe_for(
    objective: str,
    model: ModelSpec,
    dataset_path: pathlib.Path,
    run_dir: pathlib.Path,
    max_seq_len: int,
    rank: int,
    alpha: float,
    epochs: int = 1,
    initial_adapter_dir: pathlib.Path | None = None,
    matched_benchmark: bool = False,
    ranked_first_reward: bool = False,
) -> dict[str, Any]:
    recipe: dict[str, Any] = {
        "recipe": objective,
        "model": {"path": str(model.path.resolve()), "family": "gemma4"},
        "dataset": {
            "path": str(dataset_path),
            "format": (
                "text-preference" if objective == "dpo" else
                "rendered-text-grpo" if matched_benchmark else
                "text-grpo"
            ),
            "max_examples": 1,
            "max_seq_len": max_seq_len,
        },
        "adapter": {
            "rank": rank,
            "alpha": alpha,
            "target_preset": "peft-qv",
            **({"path": str(initial_adapter_dir)} if initial_adapter_dir is not None else {}),
        },
        "optimizer": {
            "learning_rate": 0.0001,
            "epochs": epochs,
            "gradient_accumulation_steps": 1,
            "max_grad_norm": 1.0,
        },
        "backend": "cuda",
        "artifacts": {
            "root": str(run_dir),
            "adapter_dir": str(run_dir / "adapter-bootstrap"),
            "trained_adapter_dir": str(run_dir / "adapter-trained"),
            "report_path": str(run_dir / f"{objective}_report.json"),
        },
    }
    if objective == "dpo":
        recipe["preference"] = {"beta": 0.1}
    else:
        recipe["grpo"] = {
            "group_size": 2,
            "max_completion_tokens": 1,
            "clip_epsilon": 0.2,
            "kl_coef": 0.04,
            "normalize_advantage": True,
            "reward_mode": "ranked-first" if ranked_first_reward or matched_benchmark else "prefix-match",
        }
    return recipe


def query_gpu_memory_mib(pid: int) -> int | None:
    tool = shutil.which("nvidia-smi")
    if tool is None:
        return None
    try:
        completed = subprocess.run(
            [tool, "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    total = 0
    found = False
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            continue
        try:
            row_pid, memory = int(fields[0]), int(fields[1])
        except ValueError:
            continue
        if row_pid == pid:
            found = True
            total += memory
    return total if found else None


def run_command(
    command: list[str],
    log_path: pathlib.Path,
    timeout_seconds: int,
    matched_benchmark: bool = False,
) -> tuple[int, float, int | None]:
    environment = os.environ.copy()
    environment["TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR"] = "1"
    for name in (
        "TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR",
        "TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK",
        "TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_NODE_IDS",
        "TERMITE_DEBUG_DEVICE_GRAD_NORM",
        "ANTFLY_GEMMA4_DPO_BENCHMARK",
        "ANTFLY_GEMMA4_GRPO_BENCHMARK",
    ):
        environment.pop(name, None)
    if matched_benchmark:
        environment["ANTFLY_GEMMA4_DPO_BENCHMARK"] = "1"
        environment["ANTFLY_GEMMA4_GRPO_BENCHMARK"] = "1"

    started = time.monotonic()
    peak_gpu_mib: int | None = None
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command: " + json.dumps(command) + "\n")
        log.flush()
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=environment, text=True)
        try:
            while process.poll() is None:
                if time.monotonic() - started > timeout_seconds:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    raise QualificationError(f"command timed out after {timeout_seconds}s; see {log_path}")
                observed = query_gpu_memory_mib(process.pid)
                if observed is not None:
                    peak_gpu_mib = max(peak_gpu_mib or 0, observed)
                time.sleep(0.2)
        finally:
            if process.poll() is None:
                process.terminate()
    return process.returncode, time.monotonic() - started, peak_gpu_mib


def require_nonnegative_int(mapping: dict[str, Any], name: str) -> int:
    value = mapping.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise QualificationError(f"report field device_execution.{name} is not a nonnegative integer")
    return value


def validate_trainable_update(report: dict[str, Any], objective: str) -> dict[str, Any]:
    update = report.get("trainable_update")
    if not isinstance(update, dict):
        raise QualificationError(f"{objective} report is missing trainable tensor movement evidence")
    tensor_count = update.get("tensor_count")
    changed_tensor_count = update.get("changed_tensor_count")
    max_abs_delta = update.get("max_abs_delta")
    if not isinstance(tensor_count, int) or isinstance(tensor_count, bool) or tensor_count < 1:
        raise QualificationError(f"{objective} report has no trainable tensors")
    if (
        not isinstance(changed_tensor_count, int)
        or isinstance(changed_tensor_count, bool)
        or changed_tensor_count < 1
        or changed_tensor_count > tensor_count
    ):
        raise QualificationError(f"{objective} report has no valid changed-tensor count")
    if (
        not isinstance(max_abs_delta, (int, float))
        or isinstance(max_abs_delta, bool)
        or not math.isfinite(max_abs_delta)
        or max_abs_delta <= 0
    ):
        raise QualificationError(f"{objective} report has no positive finite trainable-tensor delta")
    return update


def validate_report(path: pathlib.Path, objective: str) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot load {objective} report {path}: {error}") from error
    if not isinstance(report, dict):
        raise QualificationError(f"{objective} report must be an object: {path}")
    if report.get("policy_backend") != "cuda":
        raise QualificationError(f"{objective} report did not use CUDA: {report.get('policy_backend')!r}")
    optimizer_steps = report.get("optimizer_steps")
    if not isinstance(optimizer_steps, int) or optimizer_steps < 1:
        raise QualificationError(f"{objective} report has no optimizer step")
    validate_trainable_update(report, objective)
    evidence = report.get("device_execution")
    if not isinstance(evidence, dict) or evidence.get("schema_version") != "antfly_training_execution_evidence/v1":
        raise QualificationError(f"{objective} report is missing strict device execution evidence")

    train_steps = require_nonnegative_int(evidence, "train_steps")
    eval_steps = require_nonnegative_int(evidence, "eval_steps")
    if train_steps < 1:
        raise QualificationError(f"{objective} report has no compiled training graph step")
    if require_nonnegative_int(evidence, "graph_executor_partitions") < 1:
        raise QualificationError(f"{objective} report has no graph-executor partition")
    if require_nonnegative_int(evidence, "graph_executor_planned_dispatches") < 1:
        raise QualificationError(f"{objective} report has no planned CUDA dispatch")
    for name in (
        "graph_executor_fallback_steps",
        "graph_executor_native_partitions",
        "graph_executor_unsupported_ops",
        "graph_executor_interpreter_fallbacks",
        "graph_executor_true_host_outputs",
        "runtime_input_d2h_bytes",
        "compiled_session_setup_d2h_bytes",
        "graph_execution_h2d_bytes",
        "host_gradient_tensors",
    ):
        if require_nonnegative_int(evidence, name) != 0:
            raise QualificationError(f"{objective} strict CUDA evidence violation: {name} != 0")
    for observed, declared in (
        ("runtime_input_uploads", "declared_runtime_input_uploads"),
        ("runtime_input_upload_bytes", "declared_runtime_input_upload_bytes"),
        ("runtime_input_h2d_bytes", "declared_runtime_input_h2d_bytes"),
    ):
        if require_nonnegative_int(evidence, observed) != require_nonnegative_int(evidence, declared):
            raise QualificationError(f"{objective} strict CUDA evidence violation: {observed} != {declared}")
    total_steps = train_steps + eval_steps
    if require_nonnegative_int(evidence, "graph_execution_d2h_bytes") > total_steps * 4:
        raise QualificationError(f"{objective} graph readback exceeds one f32 loss per graph step")
    if require_nonnegative_int(evidence, "cuda_largest_d2h_transfer_bytes") > 4:
        raise QualificationError(f"{objective} CUDA readback exceeds one f32 scalar")
    if require_nonnegative_int(evidence, "cuda_kernel_launches") < 1:
        raise QualificationError(f"{objective} report has no CUDA kernel launch")
    return report


def validate_matched_benchmark_report(report: dict[str, Any], objective: str) -> dict[str, Any]:
    if report.get("optimizer_steps") != MATCHED_BENCHMARK_UPDATES:
        raise QualificationError(
            f"{objective} matched benchmark requires {MATCHED_BENCHMARK_UPDATES} optimizer steps"
        )
    benchmark = report.get("benchmark")
    if not isinstance(benchmark, dict) or benchmark.get("protocol") != MATCHED_BENCHMARK_PROTOCOL:
        raise QualificationError(f"{objective} report is missing the fixed matched benchmark protocol")

    if objective == "dpo":
        if report.get("examples") != MATCHED_BENCHMARK_UPDATES:
            raise QualificationError("DPO matched benchmark did not execute exactly 25 examples")
        contract = report.get("input_contract")
        expected_contract = {
            "prompt_input_ids": MATCHED_DPO_PROMPT_IDS,
            "chosen_input_ids": MATCHED_DPO_CHOSEN_IDS,
            "rejected_input_ids": MATCHED_DPO_REJECTED_IDS,
        }
        if contract != expected_contract:
            raise QualificationError(f"DPO token contract mismatch: {contract!r}")
        warmup = benchmark.get("warmup_seconds")
        measured = benchmark.get("measured_seconds")
        seconds = [benchmark.get("cold_seconds"), benchmark.get("first_seconds")]
        if isinstance(warmup, list):
            seconds.extend(warmup)
        if isinstance(measured, list):
            seconds.extend(measured)
    else:
        if report.get("groups") != MATCHED_BENCHMARK_UPDATES:
            raise QualificationError("GRPO matched benchmark did not execute exactly 25 groups")
        contract = report.get("input_contract")
        expected_contract = {
            "prompt_input_ids": MATCHED_GRPO_PROMPT_IDS,
            "group_size": 2,
            "max_completion_tokens": 1,
            "sampling": "deterministic-ranked-top-k",
        }
        if contract != expected_contract:
            raise QualificationError(f"GRPO token/sampling contract mismatch: {contract!r}")
        warmup = benchmark.get("warmup")
        measured = benchmark.get("measured")
        updates = [benchmark.get("cold"), benchmark.get("first")]
        if isinstance(warmup, list):
            updates.extend(warmup)
        if isinstance(measured, list):
            updates.extend(measured)
        if not all(isinstance(update, dict) for update in updates):
            raise QualificationError("GRPO matched benchmark update telemetry is malformed")
        seconds = [update.get("seconds") for update in updates]
        for update in updates:
            if update.get("completion_tokens") != 2:
                raise QualificationError("GRPO matched benchmark must score two one-token completions")
            if update.get("mean_reward") != 0.5 or update.get("reward_stddev") != 0.5:
                raise QualificationError("GRPO matched benchmark reward distribution is not [1, 0]")

    if not isinstance(warmup, list) or len(warmup) != MATCHED_BENCHMARK_PROTOCOL["warmup"]:
        raise QualificationError(f"{objective} benchmark warmup telemetry has the wrong length")
    if not isinstance(measured, list) or len(measured) != MATCHED_BENCHMARK_PROTOCOL["measured"]:
        raise QualificationError(f"{objective} benchmark measured telemetry has the wrong length")
    if len(seconds) != MATCHED_BENCHMARK_UPDATES or not all(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0
        for value in seconds
    ):
        raise QualificationError(f"{objective} benchmark contains invalid optimizer-step durations")
    observed_median = benchmark.get("median_seconds")
    observed_mean = benchmark.get("mean_seconds")
    expected_median = statistics.median(seconds[5:])
    expected_mean = statistics.fmean(seconds[5:])
    if not isinstance(observed_median, (int, float)) or not math.isclose(observed_median, expected_median, abs_tol=1e-12):
        raise QualificationError(f"{objective} benchmark median is inconsistent with measured updates")
    if not isinstance(observed_mean, (int, float)) or not math.isclose(observed_mean, expected_mean, abs_tol=1e-12):
        raise QualificationError(f"{objective} benchmark mean is inconsistent with measured updates")

    parity = report.get("initial_logprob_parity")
    if not isinstance(parity, dict) or parity.get("base_equivalent_policy") is not True:
        raise QualificationError(f"{objective} benchmark is missing same-base initial parity")
    parity_fields = (
        ("max_abs_error",) if objective == "dpo" else
        ("sampling_rescore_max_abs_error", "policy_reference_max_abs_error")
    )
    for name in parity_fields:
        value = parity.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value > 1e-4:
            raise QualificationError(f"{objective} initial parity exceeds tolerance: {name}={value!r}")
    return benchmark


def adapter_checkpoint(run_dir: pathlib.Path, which: str) -> pathlib.Path:
    path = run_dir / which / "adapter_model.safetensors"
    if not path.is_file():
        raise QualificationError(f"missing adapter checkpoint: {path}")
    return path


def prepare_case(
    model: ModelSpec,
    objective: str,
    grpo_target: str | None,
    run_dir: pathlib.Path,
    max_seq_len: int,
    rank: int,
    alpha: float,
    matched_benchmark: bool = False,
    initial_adapter_dir: pathlib.Path | None = None,
) -> PreparedCase:
    run_dir.mkdir(parents=True)
    dataset_path = run_dir / f"{objective}.jsonl"
    if objective == "dpo":
        row = {"prompt": "Answer with one word: yes or no?", "chosen": "yes", "rejected": "no"}
    else:
        row = {
            "prompt": "The capital of France is",
            "target": grpo_target or RANKED_FIRST_REWARD_TARGET,
        }
    dataset_path.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    resolved_initial_adapter_dir = initial_adapter_dir or (run_dir / "adapter-bootstrap")
    recipe_path = run_dir / "recipe.json"
    atomic_write_json(
        recipe_path,
        recipe_for(
            objective,
            model,
            dataset_path,
            run_dir,
            max_seq_len,
            rank,
            alpha,
            epochs=MATCHED_BENCHMARK_UPDATES if matched_benchmark else 1,
            initial_adapter_dir=resolved_initial_adapter_dir if initial_adapter_dir is not None else None,
            matched_benchmark=matched_benchmark,
            ranked_first_reward=matched_benchmark or grpo_target is None,
        ),
    )
    return PreparedCase(
        objective=objective,
        run_dir=run_dir,
        recipe_path=recipe_path,
        initial_adapter_dir=resolved_initial_adapter_dir,
    )


def finish_case(
    model: ModelSpec,
    prepared: PreparedCase,
    log_path: pathlib.Path,
    duration_seconds: float,
    peak_gpu_mib: int | None,
    duration_scope: str,
    expected_session: tuple[int, bool] | None = None,
    matched_benchmark: bool = False,
) -> dict[str, Any]:
    objective = prepared.objective
    run_dir = prepared.run_dir
    report_path = run_dir / f"{objective}_report.json"
    report = validate_report(report_path, objective)
    if objective == "grpo":
        try:
            recipe = json.loads(prepared.recipe_path.read_text(encoding="utf-8"))
            expected_reward_mode = recipe["grpo"]["reward_mode"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
            raise QualificationError(f"cannot resolve GRPO reward contract from {prepared.recipe_path}: {error}") from error
        if report.get("reward_mode") != expected_reward_mode:
            raise QualificationError(
                f"{model.label} GRPO reward mode mismatch: "
                f"report={report.get('reward_mode')!r}, recipe={expected_reward_mode!r}"
            )
    benchmark = validate_matched_benchmark_report(report, objective) if matched_benchmark else report.get("benchmark")
    if expected_session is not None:
        expected_index, expected_reuse_hit = expected_session
        session = report.get("preference_session")
        if not isinstance(session, dict):
            raise QualificationError(f"{model.label} {objective} report is missing shared-session telemetry")
        expected = {
            "shared": True,
            "model_admissions": 1,
            "run_index": expected_index,
            "reuse_hit": expected_reuse_hit,
        }
        for name, value in expected.items():
            if session.get(name) != value:
                raise QualificationError(
                    f"{model.label} {objective} shared-session telemetry mismatch: "
                    f"{name}={session.get(name)!r}, expected {value!r}"
                )

    initial_checkpoint = prepared.initial_adapter_dir / "adapter_model.safetensors"
    if not initial_checkpoint.is_file():
        raise QualificationError(f"missing adapter checkpoint: {initial_checkpoint}")
    initial_sha256 = sha256_file(initial_checkpoint)
    trained_sha256 = sha256_file(adapter_checkpoint(run_dir, "adapter-trained"))
    if trained_sha256 == initial_sha256:
        raise QualificationError(f"{model.label} {objective} persisted adapter did not change")
    embedded_report_path = run_dir / "adapter-trained" / PREFERENCE_COMPLETION_EVIDENCE_FILE_NAME
    try:
        if embedded_report_path.read_bytes() != report_path.read_bytes():
            raise QualificationError(
                f"{model.label} {objective} published adapter evidence does not match its external report"
            )
    except OSError as error:
        raise QualificationError(
            f"cannot verify {model.label} {objective} published adapter evidence: {error}"
        ) from error
    return {
        "label": model.label,
        "objective": objective,
        "status": "passed",
        "duration_seconds": duration_seconds,
        "duration_scope": duration_scope,
        "peak_gpu_memory_mib": peak_gpu_mib,
        "initial_adapter_sha256": initial_sha256,
        "trained_adapter_sha256": trained_sha256,
        "loss": report.get("loss"),
        "optimizer_steps": report.get("optimizer_steps"),
        "micro_batch_steps": report.get("micro_batch_steps"),
        "device_execution": report.get("device_execution"),
        "preference_session": report.get("preference_session"),
        "benchmark": benchmark,
        "input_contract": report.get("input_contract"),
        "initial_logprob_parity": report.get("initial_logprob_parity"),
        "reward_mode": report.get("reward_mode"),
        "trainable_update": report.get("trainable_update"),
        "initial_adapter_dir": str(prepared.initial_adapter_dir),
        "embedded_report_path": str(embedded_report_path),
        "report_path": str(report_path),
        "log_path": str(log_path),
    }


def run_case(
    antfly_bin: pathlib.Path,
    model: ModelSpec,
    objective: str,
    grpo_target: str | None,
    run_dir: pathlib.Path,
    max_seq_len: int,
    rank: int,
    alpha: float,
    timeout_seconds: int,
    matched_benchmark: bool = False,
    initial_adapter_dir: pathlib.Path | None = None,
) -> dict[str, Any]:
    prepared = prepare_case(
        model,
        objective,
        grpo_target,
        run_dir,
        max_seq_len,
        rank,
        alpha,
        matched_benchmark,
        initial_adapter_dir,
    )
    log_path = run_dir / "run.log"
    command = [str(antfly_bin), "finetune", "run", str(prepared.recipe_path)]
    return_code, duration_seconds, peak_gpu_mib = run_command(
        command, log_path, timeout_seconds, matched_benchmark
    )
    if return_code != 0:
        raise QualificationError(f"{model.label} {objective} failed with exit code {return_code}; see {log_path}")
    return finish_case(
        model,
        prepared,
        log_path,
        duration_seconds,
        peak_gpu_mib,
        "isolated-process",
        matched_benchmark=matched_benchmark,
    )


def validate_suite_report(path: pathlib.Path, model: ModelSpec, run_count: int) -> dict[str, Any]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise QualificationError(f"cannot load preference suite report {path}: {error}") from error
    if not isinstance(report, dict):
        raise QualificationError(f"preference suite report must be an object: {path}")
    expected = {
        "schema_version": "antfly_inference_gemma4_preference_suite/v2",
        "status": "succeeded",
        "model_path": str(model.path.resolve()),
        "backend": "cuda",
        "runs_planned": run_count,
        "runs_completed": run_count,
        "model_admissions": 1,
        "reuse_hits": run_count - 1,
    }
    for name, value in expected.items():
        if report.get(name) != value:
            raise QualificationError(
                f"preference suite report mismatch: {name}={report.get(name)!r}, expected {value!r}"
            )
    admission_seconds = report.get("model_admission_seconds")
    total_seconds = report.get("total_duration_seconds")
    if not isinstance(admission_seconds, (int, float)) or not math.isfinite(admission_seconds) or admission_seconds <= 0:
        raise QualificationError("preference suite report has no positive model admission duration")
    if not isinstance(total_seconds, (int, float)) or not math.isfinite(total_seconds) or total_seconds <= admission_seconds:
        raise QualificationError("preference suite report has no valid total duration")
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != run_count:
        raise QualificationError("preference suite report has incomplete per-job timings")
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict) or run.get("run_index") != index:
            raise QualificationError("preference suite report has invalid run timing order")
        duration = run.get("duration_seconds")
        if not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration <= 0:
            raise QualificationError("preference suite report has a non-positive job duration")
    return report


def run_shared_suite(
    antfly_bin: pathlib.Path,
    model: ModelSpec,
    objectives: list[str],
    grpo_target: str | None,
    out_dir: pathlib.Path,
    repetition: int,
    max_seq_len: int,
    rank: int,
    alpha: float,
    timeout_seconds: int,
    matched_benchmark: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    initial_adapter_dir = (
        out_dir / model.label / "matched-initial-adapter-rank16-alpha32"
        if matched_benchmark
        else None
    )
    prepared_cases = [
        prepare_case(
            model,
            objective,
            grpo_target,
            out_dir / model.label / objective / f"run-{repetition:02d}",
            max_seq_len,
            rank,
            alpha,
            matched_benchmark,
            initial_adapter_dir,
        )
        for objective in objectives
    ]
    suite_dir = out_dir / model.label / "shared-session" / f"run-{repetition:02d}"
    suite_dir.mkdir(parents=True)
    report_path = suite_dir / "preference_suite_report.json"
    log_path = suite_dir / "run.log"
    command = [
        str(antfly_bin),
        "finetune",
        "run-suite",
        "--report",
        str(report_path),
        *(str(case.recipe_path) for case in prepared_cases),
    ]
    return_code, duration_seconds, peak_gpu_mib = run_command(
        command, log_path, timeout_seconds, matched_benchmark
    )
    if return_code != 0:
        raise QualificationError(
            f"{model.label} shared preference suite failed with exit code {return_code}; see {log_path}"
        )
    suite_report = validate_suite_report(report_path, model, len(prepared_cases))
    run_timings = suite_report["runs"]
    results = [
        finish_case(
            model,
            prepared,
            log_path,
            run_timings[index - 1]["duration_seconds"],
            peak_gpu_mib,
            "shared-session-job",
            (index, index > 1),
            matched_benchmark,
        )
        for index, prepared in enumerate(prepared_cases, start=1)
    ]
    return results, {
        "label": model.label,
        "repetition": repetition,
        "status": "passed",
        "objectives": objectives,
        "duration_seconds": duration_seconds,
        "model_admission_seconds": suite_report["model_admission_seconds"],
        "reported_total_duration_seconds": suite_report["total_duration_seconds"],
        "run_timings": suite_report["runs"],
        "peak_gpu_memory_mib": peak_gpu_mib,
        "report": suite_report,
        "report_path": str(report_path),
        "log_path": str(log_path),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", action="append", required=True, metavar="LABEL=DIR")
    result.add_argument("--grpo-target", action="append", default=[], metavar="LABEL=TEXT")
    result.add_argument("--objective", action="append", choices=("dpo", "grpo"), default=[])
    result.add_argument("--antfly-bin", type=pathlib.Path)
    result.add_argument("--out", type=pathlib.Path)
    result.add_argument("--preflight-only", action="store_true")
    result.add_argument(
        "--matched-benchmark",
        action="store_true",
        help=(
            "run the locked rank-16/alpha-32 25-update DPO/GRPO protocol, "
            "share one immutable initial adapter per model, and require benchmark telemetry"
        ),
    )
    result.add_argument("--repetitions", type=int, default=1)
    result.add_argument("--max-seq-len", type=int, default=128)
    result.add_argument("--rank", type=int)
    result.add_argument("--alpha", type=float, default=32.0)
    result.add_argument("--timeout-seconds", type=int, default=3600)
    result.add_argument(
        "--isolated-processes",
        action="store_true",
        help="run each objective in a fresh process instead of reusing one admitted model for multi-objective suites",
    )
    return result


def validate_grpo_targets(
    models: list[ModelSpec],
    targets: dict[str, str],
    objectives: list[str],
    matched_benchmark: bool,
) -> None:
    if "grpo" not in objectives:
        if targets:
            raise QualificationError("--grpo-target requires the GRPO objective")
        return
    if matched_benchmark and targets:
        raise QualificationError(
            "--matched-benchmark uses deterministic ranked-first rewards and rejects --grpo-target"
        )


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        rank = args.rank if args.rank is not None else (16 if args.matched_benchmark else 8)
        models_raw = parse_unique_assignments(args.model, "--model")
        targets = parse_unique_assignments(args.grpo_target, "--grpo-target")
        unknown_targets = sorted(set(targets) - set(models_raw))
        if unknown_targets:
            raise QualificationError(f"--grpo-target has no matching --model: {', '.join(unknown_targets)}")
        if args.repetitions < 1 or args.max_seq_len < 8 or rank < 1 or args.alpha <= 0 or args.timeout_seconds < 1:
            raise QualificationError("repetitions, sequence length, rank, alpha, and timeout must be positive")
        if args.matched_benchmark and (args.max_seq_len != 128 or rank != 16 or args.alpha != 32.0):
            raise QualificationError(
                "--matched-benchmark locks --max-seq-len=128, --rank=16, and --alpha=32"
            )

        models = [ModelSpec(label, pathlib.Path(path)) for label, path in models_raw.items()]
        preflight = [inspect_model(model) for model in models]
        preflight_payload = [asdict(item) for item in preflight]
        if args.preflight_only:
            print(json.dumps({"status": "passed", "models": preflight_payload}, indent=2, sort_keys=True))
            return 0

        objectives = list(dict.fromkeys(args.objective or ["dpo", "grpo"]))
        validate_grpo_targets(models, targets, objectives, args.matched_benchmark)

        script_dir = pathlib.Path(__file__).resolve().parent
        package_root = script_dir.parent
        antfly_bin = (args.antfly_bin or package_root / "zig-out/bin/antfly-inference").resolve()
        if not antfly_bin.is_file() or not os.access(antfly_bin, os.X_OK):
            raise QualificationError(f"antfly-inference binary is not executable: {antfly_bin}")
        out_dir = (args.out or pathlib.Path("/tmp") / f"antfly-gemma4-cuda-preference-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}").resolve()
        if out_dir.exists():
            raise QualificationError(f"refusing to reuse existing output directory: {out_dir}")
        out_dir.mkdir(parents=True)

        summary: dict[str, Any] = {
            "schema_version": "antfly_gemma4_cuda_preference_smoke/v3",
            "status": "running",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "antfly_bin": str(antfly_bin),
            "models": preflight_payload,
            "objectives": objectives,
            "repetitions": args.repetitions,
            "matched_benchmark": args.matched_benchmark,
            "benchmark_contract": {
                "protocol": MATCHED_BENCHMARK_PROTOCOL,
                "updates_per_objective": MATCHED_BENCHMARK_UPDATES,
                "rank": rank,
                "alpha": args.alpha,
                "max_seq_len": args.max_seq_len,
                "initial_adapter_policy": "one immutable shared Antfly adapter directory per model",
                "dpo_input": {
                    "prompt_input_ids": MATCHED_DPO_PROMPT_IDS,
                    "chosen_input_ids": MATCHED_DPO_CHOSEN_IDS,
                    "rejected_input_ids": MATCHED_DPO_REJECTED_IDS,
                },
                "grpo_input": {
                    "prompt_input_ids": MATCHED_GRPO_PROMPT_IDS,
                    "raw_text_payload_input_ids": MATCHED_GRPO_PAYLOAD_IDS,
                    "reward_mode": "ranked-first",
                    "reward_distribution": [1.0, 0.0],
                    "group_size": 2,
                    "max_completion_tokens": 1,
                    "sampling": "deterministic-ranked-top-k",
                },
            } if args.matched_benchmark else None,
            "execution_mode": (
                "shared-session-suite" if len(objectives) >= 2 and not args.isolated_processes else "isolated-processes"
            ),
            "cases": [],
            "suites": [],
        }
        summary_path = out_dir / "summary.json"
        atomic_write_json(summary_path, summary)
        try:
            for model in models:
                objective_results: dict[str, list[dict[str, Any]]] = {objective: [] for objective in objectives}
                if len(objectives) >= 2 and not args.isolated_processes:
                    for repetition in range(1, args.repetitions + 1):
                        results, suite = run_shared_suite(
                            antfly_bin,
                            model,
                            objectives,
                            targets.get(model.label),
                            out_dir,
                            repetition,
                            args.max_seq_len,
                            rank,
                            args.alpha,
                            args.timeout_seconds,
                            args.matched_benchmark,
                        )
                        summary["suites"].append(suite)
                        for result in results:
                            result["repetition"] = repetition
                            summary["cases"].append(result)
                            objective_results[result["objective"]].append(result)
                        atomic_write_json(summary_path, summary)
                else:
                    initial_adapter_dir = (
                        out_dir / model.label / "matched-initial-adapter-rank16-alpha32"
                        if args.matched_benchmark
                        else None
                    )
                    for objective in objectives:
                        for repetition in range(1, args.repetitions + 1):
                            case_dir = out_dir / model.label / objective / f"run-{repetition:02d}"
                            result = run_case(
                                antfly_bin,
                                model,
                                objective,
                                targets.get(model.label),
                                case_dir,
                                args.max_seq_len,
                                rank,
                                args.alpha,
                                args.timeout_seconds,
                                args.matched_benchmark,
                                initial_adapter_dir,
                            )
                            result["repetition"] = repetition
                            summary["cases"].append(result)
                            objective_results[objective].append(result)
                            atomic_write_json(summary_path, summary)

                for objective, results in objective_results.items():
                    digests = [result["trained_adapter_sha256"] for result in results]
                    losses = [result["loss"] for result in results]
                    if len(set(digests)) != 1 or len(set(map(json.dumps, losses))) != 1:
                        raise QualificationError(
                            f"{model.label} {objective} repetitions were not deterministic: "
                            f"adapter_digests={digests} losses={losses}"
                        )
                    if args.matched_benchmark:
                        medians = [result["benchmark"]["median_seconds"] for result in results]
                        means = [result["benchmark"]["mean_seconds"] for result in results]
                        summary.setdefault("performance", []).append({
                            "label": model.label,
                            "objective": objective,
                            "repetitions": len(results),
                            "median_of_measured_medians_seconds": statistics.median(medians),
                            "mean_of_measured_means_seconds": statistics.fmean(means),
                            "measured_medians_seconds": medians,
                            "measured_means_seconds": means,
                        })
            summary["status"] = "passed"
        except Exception as error:
            summary["status"] = "failed"
            summary["error"] = str(error)
            atomic_write_json(summary_path, summary)
            raise
        atomic_write_json(summary_path, summary)
        print(f"PASS summary={summary_path}")
        return 0
    except QualificationError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
