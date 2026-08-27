#!/usr/bin/env python3
"""Qualify deterministic Gemma4 DPO/GRPO recovery after a real interruption.

The same recipe is executed uninterrupted, interrupted by SIGTERM after a
durable epoch checkpoint, and resumed into a fresh immutable artifact root.
A PASS requires a content-addressed aggregate sidecar, byte-identical final
adapter tensors, exact training/discrete metrics and traces, and a bounded
absolute tolerance only for terminal Metal GRPO KL floats. The harness is
standard-library-only.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import signal
import stat
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "antfly_gemma4_preference_resume_qualification/v2"
CHECKPOINT_TENSOR = "__trainer_state_v2"
CHECKPOINT_SCHEMA_VERSION = 2
CHECKPOINT_FIELD_COUNT = 18
CHECKPOINT_FLOAT_COUNT = CHECKPOINT_FIELD_COUNT * 4
STATE_SCHEMA_VERSION = "antfly_gemma4_preference_checkpoint_state/v1"
ADAPTER_MANIFEST_SCHEMA_V2 = "antfly_gemma4_finetune/v2"
ADAPTER_MANIFEST_SCHEMA_V3 = "antfly_gemma4_finetune/v3"
CANONICAL_EVALUATION_POLICY = (
    "terminal-device-drained-host-weight-snapshot-fresh-backend-private-buffer-reuse-disabled"
)
GRPO_TERMINAL_METAL_ABS_TOLERANCES = {
    "kl_loss": 1e-6,
    "mean_kl": 1e-5,
}
GRPO_TERMINAL_METAL_EVALUATION_ARTIFACT_ABS_TOLERANCES = {
    "loss": 1e-6,
    **GRPO_TERMINAL_METAL_ABS_TOLERANCES,
}
CHECKPOINT_MAGIC = {
    "dpo": 0x44504F2D43504B31,
    "grpo": 0x4752504F43504B31,
}
ENVIRONMENT_POLICY_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "finetune"
    / "gemma4_preference_environment.policy"
)
COMPILED_GRPO_SAMPLING_ENV = "ANTFLY_GEMMA4_GRPO_COMPILED_SAMPLING"
COMPILED_GRPO_SAMPLING_MODE = (
    "compiled-shared-prompt-ranked-sparse-row-each-step"
)
COMPILED_GRPO_POLICY_LOGPROB_MODE = (
    "compiled-token-selection-with-eager-per-completion-token-validated-logprob-rescore"
)


def _load_environment_policy() -> tuple[
    tuple[str, ...], frozenset[str], dict[str, str], str
]:
    try:
        policy_bytes = ENVIRONMENT_POLICY_PATH.read_bytes()
        policy_text = policy_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"cannot load Gemma4 preference environment policy: {ENVIRONMENT_POLICY_PATH}: {exc}"
        ) from exc

    sanitize_prefixes: list[str] = []
    sanitize_names: set[str] = set()
    strict: dict[str, str] = {}
    allowed: dict[str, tuple[str, tuple[str, ...]]] = {}
    denied: set[str] = set()
    admission_directives = {
        "allow-bool",
        "allow-presence",
        "allow-uint",
        "allow-uint-range",
        "allow-fixed",
    }
    for line_number, raw_line in enumerate(policy_text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        directive = fields[0]
        where = f"{ENVIRONMENT_POLICY_PATH}:{line_number}"
        if directive in {"scope-prefix", "sanitize-prefix", "sanitize-exact", "deny"}:
            if len(fields) != 2:
                raise RuntimeError(f"invalid {directive} entry at {where}")
        elif directive in {"strict", "allow-fixed"}:
            if len(fields) != 3:
                raise RuntimeError(f"invalid {directive} entry at {where}")
        elif directive in {"allow-bool", "allow-presence", "allow-uint"}:
            if len(fields) != 2:
                raise RuntimeError(f"invalid {directive} entry at {where}")
        elif directive == "allow-uint-range":
            if len(fields) != 4:
                raise RuntimeError(f"invalid {directive} entry at {where}")
        else:
            raise RuntimeError(f"unknown environment-policy directive at {where}: {directive}")

        if directive == "sanitize-prefix":
            if fields[1] in sanitize_prefixes:
                raise RuntimeError(f"duplicate sanitize prefix at {where}: {fields[1]}")
            sanitize_prefixes.append(fields[1])
        elif directive == "sanitize-exact":
            if fields[1] in sanitize_names:
                raise RuntimeError(f"duplicate sanitized name at {where}: {fields[1]}")
            sanitize_names.add(fields[1])
        elif directive == "strict":
            if fields[1] in strict:
                raise RuntimeError(f"duplicate strict binding at {where}: {fields[1]}")
            strict[fields[1]] = fields[2]
        elif directive == "deny":
            if fields[1] in denied or fields[1] in allowed:
                raise RuntimeError(f"duplicate admission entry at {where}: {fields[1]}")
            denied.add(fields[1])
        elif directive in admission_directives:
            if fields[1] in denied or fields[1] in allowed:
                raise RuntimeError(f"duplicate admission entry at {where}: {fields[1]}")
            allowed[fields[1]] = (directive, tuple(fields[2:]))

    def canonical(name: str, value: str) -> bool:
        directive, arguments = allowed[name]
        if directive == "allow-bool":
            return value in {"0", "1"}
        if directive == "allow-presence":
            return value == "1"
        if directive == "allow-fixed":
            return value == arguments[0]
        try:
            parsed = int(value, 10)
        except ValueError:
            return False
        if parsed < 0:
            return False
        if directive == "allow-uint":
            return True
        return int(arguments[0], 10) <= parsed <= int(arguments[1], 10)

    for name, value in strict.items():
        if name not in allowed or not canonical(name, value):
            raise RuntimeError(f"strict binding is not canonically allowed: {name}={value}")
    if not sanitize_prefixes or not strict:
        raise RuntimeError("environment policy must define sanitization and strict bindings")

    digest = "sha256:" + hashlib.sha256(policy_bytes).hexdigest()
    return tuple(sanitize_prefixes), frozenset(sanitize_names), strict, digest


(
    SANITIZED_ENV_PREFIXES,
    SANITIZED_ENV_NAMES,
    STRICT_METAL_ENV,
    ENVIRONMENT_POLICY_SHA256,
) = _load_environment_policy()


def _strict_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    inherited = os.environ if source is None else source
    result = {
        name: value
        for name, value in inherited.items()
        if name not in SANITIZED_ENV_NAMES
        and not any(name.startswith(prefix) for prefix in SANITIZED_ENV_PREFIXES)
    }
    result.update(STRICT_METAL_ENV)
    return result


def _apply_compiled_sampling_recipe_contract(
    recipe: dict[str, Any], task: str, enabled: bool
) -> None:
    if not enabled:
        return
    if task != "grpo":
        raise ContractError("--compiled-sampling is valid only for GRPO")
    runtime_value = recipe.get("runtime")
    runtime = (
        dict(_mapping(runtime_value, "recipe.runtime"))
        if runtime_value is not None
        else {}
    )
    runtime["grpo_incremental_kv"] = False
    # These fields are refinements of the parent incremental-KV switch.  The
    # recipe contract rejects them when the parent is disabled even if their
    # values are false, so compiled sampling must remove inherited refinements
    # instead of merely turning them off.
    for field in (
        "grpo_incremental_kv_batch_active",
        "grpo_incremental_kv_clone_prompt_tail",
        "grpo_incremental_kv_shadow_exact",
    ):
        runtime.pop(field, None)
    recipe["runtime"] = runtime


METAL_NUMERICAL_POLICY_BOOLEAN_FIELDS = (
    "fused_rms_norm_backward",
    "fused_gqa_attention_backward",
    "fused_linear_cross_entropy",
    "sparse_logits_cross_entropy",
    "bf16_tiled32_m16",
    "bf16_simdgroup_mm",
    "bf16_simdgroup_m64",
    "bf16_forward_simdgroup_m64_packed",
    "bf16_simdgroup_m64_prefix_tail",
    "bf16_backward_tiled32_m16",
    "bf16_backward_small_rows",
    "bf16_backward_simdgroup_mm",
    "bf16_backward_simdgroup_m64",
    "bf16_backward_simdgroup_m64_coalesced",
    "bf16_backward_simdgroup_m64_packed",
    "rms_norm_backward_simdgroup",
    "rms_norm_backward_residual_add",
    "rms_norm_generated",
    "linear_cce_f16_grad",
    "linear_cce_logit_cache",
    "linear_cce_f16_mps_backward",
    "dense_mps_linear",
    "gemma4_bf16_mlp_fusion",
    "gemma4_gate_up_backward_input_sum",
    "q4_0_linear_rms_add_sumsq",
    "eager_rank1_dot_specialization",
    "dense_device_dot_general",
    "lora_forward_fused_branch",
    "lora_forward_generic_rank16",
    "lora_forward_rank1_fused",
    "reference_quant_linear",
    "quant_backward_force_barriers",
    "contiguous_slice_device_view",
    "partition_fused_patterns",
    "partition_runtime_commands",
    "runtime_region_plan",
    "grouped_mps_dot",
    "gather_promote_input",
    "reduce_promote_input",
    "lora_backward_runtime_region",
    "low_rank_lora_backward_runtime_region",
    "rank_adapter_backward_runtime_region",
    "ffn_gelu_backward_runtime_region",
    "gated_gelu_backward_runtime_region",
    "gated_gelu_forward_fusion",
    "masked_softmax_runtime_region",
    "softmax_backward_runtime_region",
    "graph_rank1_dot_specialization",
    "raw_linear_bias_pair_runtime_region",
    "raw_linear_runtime_regions_suppressed",
    "gated_ffn_graph_fusion",
    "gemma_gated_mlp_training_graph_fusion",
    "attention_output_residual_graph_fusion",
    "grouped_lora_a_r16",
    "add3_fusion",
)
ADAPTER_MANIFEST_IDENTITY_FIELDS = (
    "schema_version",
    "artifact_family_version",
    "tensor_key_format",
    "base_model_name_or_path",
    "base_model_sha256",
    "tokenizer_sha256",
    "chat_template_sha256",
    "target_modules",
    "target_preset",
    "rank",
    "alpha",
    "use_dora",
    "use_rslora",
    "initializer",
    "initialization_seed",
    "recursive_lora",
)


class ContractError(RuntimeError):
    """The qualification could not establish the production contract."""


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where}: expected object")
    return value


def _integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{where}: expected integer >= {minimum}")
    return value


def _finite_number(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}: expected finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{where}: expected finite number")
    return result


def _regular_file(path: Path, where: str, *, executable: bool = False) -> Path:
    candidate = path.expanduser()
    try:
        candidate_mode = candidate.lstat().st_mode
    except OSError as exc:
        raise ContractError(f"{where}: cannot stat file: {candidate}: {exc}") from exc
    if stat.S_ISLNK(candidate_mode):
        raise ContractError(f"{where}: file must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    if not stat.S_ISREG(resolved.stat().st_mode):
        raise ContractError(f"{where}: not a regular file: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise ContractError(f"{where}: not executable: {resolved}")
    return resolved


def _closed_immutable_path(path: Path, where: str) -> Path:
    candidate = path.expanduser()
    try:
        candidate_mode = candidate.lstat().st_mode
    except OSError as exc:
        raise ContractError(f"{where}: cannot stat path: {candidate}: {exc}") from exc
    if stat.S_ISLNK(candidate_mode):
        raise ContractError(f"{where}: path must not be a symlink: {candidate}")
    resolved = candidate.resolve(strict=True)
    if resolved.is_file():
        if not stat.S_ISREG(resolved.stat().st_mode):
            raise ContractError(f"{where}: expected a regular file or directory")
        return resolved
    if not resolved.is_dir():
        raise ContractError(f"{where}: expected a regular file or directory")
    for child in resolved.rglob("*"):
        mode = child.lstat().st_mode
        relative = child.relative_to(resolved).as_posix()
        if stat.S_ISLNK(mode):
            raise ContractError(f"{where}: immutable tree contains symlink: {relative}")
        if not stat.S_ISDIR(mode) and not stat.S_ISREG(mode):
            raise ContractError(
                f"{where}: immutable tree contains unsupported entry: {relative}"
            )
    return resolved


def _effective_regular_file(
    primary: Any,
    fallback: Any,
    where: str,
) -> Path:
    values = [value for value in (primary, fallback) if value is not None]
    if not values or any(not isinstance(value, str) for value in values):
        raise ContractError(f"qualification requires a pinned {where} path")
    resolved = [_regular_file(Path(value), where) for value in values]
    if any(path != resolved[0] for path in resolved[1:]):
        raise ContractError(f"conflicting {where} paths")
    return resolved[0]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _load_json(path: Path, where: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), where)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"{where}: invalid JSON at {path}: {exc}") from exc


def _tree_snapshot(root: Path) -> list[dict[str, Any]]:
    resolved = root.expanduser().resolve(strict=True)
    paths = [resolved] if resolved.is_file() else [resolved, *sorted(resolved.rglob("*"))]
    result: list[dict[str, Any]] = []
    for path in paths:
        info = path.lstat()
        entry = {
            "path": "." if path == resolved else path.relative_to(resolved).as_posix(),
            "kind": "symlink" if path.is_symlink() else "directory" if path.is_dir() else "file",
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "inode": info.st_ino,
            "mode": stat.S_IMODE(info.st_mode),
        }
        if stat.S_ISREG(info.st_mode):
            entry["sha256"] = _sha256(path)
        result.append(entry)
    return result


def _paths_overlap(first: Path, second: Path) -> bool:
    first_resolved = first.expanduser().resolve(strict=False)
    second_resolved = second.expanduser().resolve(strict=False)
    return (
        first_resolved == second_resolved
        or first_resolved.is_relative_to(second_resolved)
        or second_resolved.is_relative_to(first_resolved)
    )


def _snapshot_digest(snapshot: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _adapter_tree_evidence(root: Path, where: str) -> dict[str, Any]:
    candidate = root.expanduser()
    try:
        root_mode = candidate.lstat().st_mode
    except OSError as exc:
        raise ContractError(f"{where}: cannot stat adapter directory: {candidate}: {exc}") from exc
    if stat.S_ISLNK(root_mode) or not stat.S_ISDIR(root_mode):
        raise ContractError(f"{where}: adapter root must be a non-symlink directory")
    resolved = candidate.resolve(strict=True)
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(resolved.rglob("*")):
        mode = path.lstat().st_mode
        relative = path.relative_to(resolved).as_posix()
        if stat.S_ISLNK(mode):
            raise ContractError(f"{where}: adapter artifact must not be a symlink: {relative}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ContractError(f"{where}: adapter artifact is not a regular file: {relative}")
        files[relative] = {
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    required = {
        "adapter_model.safetensors",
        "adapter_config.json",
        "antfly_finetune_manifest.json",
    }
    missing = sorted(required - set(files))
    if missing:
        raise ContractError(f"{where}: adapter artifact is incomplete: missing={missing}")
    _load_json(resolved / "adapter_config.json", f"{where} adapter_config.json")
    manifest = _load_json(
        resolved / "antfly_finetune_manifest.json",
        f"{where} antfly_finetune_manifest.json",
    )
    schema_version = manifest.get("schema_version")
    if schema_version not in {
        ADAPTER_MANIFEST_SCHEMA_V2,
        ADAPTER_MANIFEST_SCHEMA_V3,
    }:
        raise ContractError(f"{where}: unsupported Gemma 4 adapter manifest schema")
    if manifest.get("status") != "complete":
        raise ContractError(f"{where}: Gemma 4 adapter manifest is not complete")
    initialization_seed = manifest.get("initialization_seed")
    if schema_version == ADAPTER_MANIFEST_SCHEMA_V2:
        if initialization_seed is not None:
            raise ContractError(
                f"{where}: Gemma 4 v2 adapter manifest must not carry an initialization seed"
            )
    else:
        _integer(
            initialization_seed,
            f"{where} adapter manifest.initialization_seed",
        )
    checkpoint = files["adapter_model.safetensors"]
    if manifest.get("adapter_checkpoint_sha256") != checkpoint["sha256"].removeprefix(
        "sha256:"
    ) or manifest.get("adapter_checkpoint_size_bytes") != checkpoint["size_bytes"]:
        raise ContractError(f"{where}: adapter manifest does not bind checkpoint bytes")
    return {
        "root": str(resolved),
        "files": files,
        "adapter_model_sha256": checkpoint["sha256"],
        "adapter_model_size_bytes": checkpoint["size_bytes"],
        "manifest_identity": {
            field: manifest.get(field) for field in ADAPTER_MANIFEST_IDENTITY_FIELDS
        },
    }


def inspect_training_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = _regular_file(path, "training checkpoint")
    file_size = checkpoint.stat().st_size
    with checkpoint.open("rb") as handle:
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ContractError("training checkpoint: truncated SafeTensors prefix")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size == 0 or header_size > 64 * 1024 * 1024 or 8 + header_size > file_size:
            raise ContractError("training checkpoint: invalid SafeTensors header length")
        raw_header = handle.read(header_size)
        try:
            header = _mapping(
                json.loads(raw_header.decode("utf-8").rstrip(" ")),
                "checkpoint header",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"training checkpoint: invalid SafeTensors header: {exc}") from exc
        descriptor = _mapping(header.get(CHECKPOINT_TENSOR), CHECKPOINT_TENSOR)
        if descriptor.get("dtype") != "F32" or descriptor.get("shape") != [CHECKPOINT_FLOAT_COUNT]:
            raise ContractError(f"{CHECKPOINT_TENSOR}: unexpected dtype or shape")
        offsets = descriptor.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in offsets)
            or offsets[0] < 0
            or offsets[1] - offsets[0] != CHECKPOINT_FLOAT_COUNT * 4
            or 8 + header_size + offsets[1] > file_size
        ):
            raise ContractError(f"{CHECKPOINT_TENSOR}: invalid data offsets")
        handle.seek(8 + header_size + offsets[0])
        raw_values = handle.read(CHECKPOINT_FLOAT_COUNT * 4)
    values = struct.unpack(f"<{CHECKPOINT_FLOAT_COUNT}f", raw_values)
    fields: list[int] = []
    for field_index in range(CHECKPOINT_FIELD_COUNT):
        field = 0
        for chunk_index in range(4):
            value = values[field_index * 4 + chunk_index]
            if not math.isfinite(value) or value < 0 or value > 65535 or value != math.floor(value):
                raise ContractError(f"{CHECKPOINT_TENSOR}: invalid encoded integer")
            field |= int(value) << (chunk_index * 16)
        fields.append(field)
    if fields[0] != CHECKPOINT_SCHEMA_VERSION:
        raise ContractError(f"{CHECKPOINT_TENSOR}: unsupported schema {fields[0]}")
    if fields[2] > fields[1] or fields[4] >= max(fields[5], 1):
        raise ContractError(f"{CHECKPOINT_TENSOR}: inconsistent counters")
    return {
        "schema_version": fields[0],
        "micro_batch_steps": fields[1],
        "optimizer_steps": fields[2],
        "stochastic_steps": fields[3],
        "accumulation_micro_batches": fields[4],
        "configured_accumulation_steps": fields[5],
        "trainer_seed": fields[6],
        "epoch_index": fields[7],
        "next_example_index": fields[8],
        "examples_seen": fields[9],
        "order_seed": fields[10],
        "order_cursor": fields[11],
        "rng_state": fields[12:16],
        "conditional_family_count": fields[16],
        "conditional_family_present_mask": fields[17],
        "size_bytes": file_size,
        "sha256": _sha256(checkpoint),
    }


def inspect_preference_checkpoint(path: Path, task: str) -> dict[str, Any]:
    trainer = inspect_training_checkpoint(path)
    digest_bytes = b"".join(int(word).to_bytes(8, "little") for word in trainer["rng_state"])
    digest = digest_bytes.hex()
    state_path = Path(f"{path}.preference-state-{digest}.json")
    state = _load_json(_regular_file(state_path, "preference checkpoint sidecar"), "checkpoint sidecar")
    if _sha256(state_path) != f"sha256:{digest}":
        raise ContractError("checkpoint sidecar digest does not match trainer progress")
    if state.get("schema_version") != STATE_SCHEMA_VERSION or state.get("task") != task:
        raise ContractError("checkpoint sidecar schema or task mismatch")
    if trainer["order_seed"] != CHECKPOINT_MAGIC[task]:
        raise ContractError("trainer checkpoint has the wrong preference-task marker")
    for field in ("epoch_index", "micro_batch_steps", "optimizer_steps", "accumulation_micro_batches"):
        if _integer(state.get(field), f"checkpoint sidecar.{field}") != trainer[field]:
            raise ContractError(f"checkpoint sidecar {field} does not match trainer checkpoint")
    if (state.get(task) is None) or (state.get("grpo" if task == "dpo" else "dpo") is not None):
        raise ContractError("checkpoint sidecar contains the wrong task aggregate")
    aggregate = _mapping(state.get(task), f"checkpoint sidecar.{task}")
    aggregate_count = aggregate.get("examples_seen" if task == "dpo" else "total_groups")
    if _integer(aggregate_count, "checkpoint sidecar aggregate count") != trainer["examples_seen"]:
        raise ContractError("checkpoint aggregate count does not match trainer progress")
    incremental_kv = aggregate.get("incremental_kv") if task == "grpo" else None
    if incremental_kv is not None:
        incremental_kv = dict(_mapping(incremental_kv, "checkpoint sidecar.grpo.incremental_kv"))
        if _integer(
            incremental_kv.get("groups"),
            "checkpoint sidecar.grpo.incremental_kv.groups",
        ) != trainer["examples_seen"]:
            raise ContractError("checkpoint incremental-KV group count does not match progress")
        if incremental_kv.get("cache_dtype") != "f32":
            raise ContractError("checkpoint incremental-KV dtype is not f32")
    fingerprint = state.get("run_fingerprint_sha256")
    if not isinstance(fingerprint, str) or not fingerprint.startswith("sha256:") or len(fingerprint) != 71:
        raise ContractError("checkpoint sidecar has an invalid run fingerprint")
    return {
        **trainer,
        "state_path": str(state_path.resolve()),
        "state_sha256": f"sha256:{digest}",
        "run_fingerprint_sha256": fingerprint,
        "incremental_kv": incremental_kv,
    }


def _tail(path: Path, limit: int = 6000) -> str:
    try:
        return path.read_bytes()[-limit:].decode("utf-8", errors="replace")
    except OSError:
        return ""


def _run_to_completion(
    command: Sequence[str],
    env: Mapping[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        try:
            result = subprocess.run(
                command,
                env=dict(env),
                stdout=stdout,
                stderr=stderr,
                check=False,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise ContractError(f"training command exceeded {timeout_seconds:g}s") from exc
    if result.returncode != 0:
        raise ContractError(
            f"training command failed with {result.returncode}; stderr tail:\n{_tail(stderr_path)}"
        )
    return {"returncode": result.returncode, "elapsed_seconds": time.monotonic() - started}


def _run_and_interrupt(
    command: Sequence[str],
    env: Mapping[str, str],
    checkpoint_path: Path,
    task: str,
    expected_epoch: int,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
    poll_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started = time.monotonic()
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, env=dict(env), stdout=stdout, stderr=stderr)
        state: dict[str, Any] | None = None
        try:
            while time.monotonic() - started < timeout_seconds:
                returncode = process.poll()
                if returncode is not None:
                    raise ContractError(
                        "interruption target exited before the checkpoint "
                        f"(returncode={returncode}); stderr tail:\n{_tail(stderr_path)}"
                    )
                if checkpoint_path.exists():
                    try:
                        candidate = inspect_preference_checkpoint(checkpoint_path, task)
                    except (ContractError, OSError):
                        candidate = None
                    if candidate is not None:
                        epoch = _integer(candidate["epoch_index"], "checkpoint epoch")
                        if epoch > expected_epoch:
                            raise ContractError(
                                f"checkpoint advanced past interruption boundary: {epoch} > {expected_epoch}"
                            )
                        if epoch == expected_epoch:
                            state = candidate
                            break
                time.sleep(poll_seconds)
            if state is None:
                raise ContractError(
                    f"checkpoint epoch {expected_epoch} was not observed within {timeout_seconds:g}s"
                )
            process.send_signal(signal.SIGTERM)
            try:
                returncode = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = process.wait(timeout=15)
            if returncode == 0:
                raise ContractError("interruption target completed before SIGTERM took effect")
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=15)
    final_state = inspect_preference_checkpoint(checkpoint_path, task)
    if final_state["sha256"] != state["sha256"] or final_state["state_sha256"] != state["state_sha256"]:
        raise ContractError("checkpoint generation changed after the observed interruption boundary")
    if final_state["next_example_index"] != 0 or final_state["order_cursor"] != 0:
        raise ContractError("checkpoint is not at an epoch boundary")
    return (
        {
            "pid": process.pid,
            "returncode": returncode,
            "signal": -returncode if returncode < 0 else None,
            "elapsed_seconds": time.monotonic() - started,
        },
        final_state,
    )


def _write_json(path: Path, payload: Mapping[str, Any], *, exclusive: bool = True) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    mode = "x" if exclusive else "w"
    try:
        with temporary.open(mode, encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive and path.exists():
            raise ContractError(f"refusing to replace file: {path}")
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _recipe_variant(
    base: Mapping[str, Any],
    task: str,
    artifact_root: Path,
    checkpoint_path: Path,
    epochs: int,
    checkpoint_every: int,
    resume: bool,
) -> dict[str, Any]:
    recipe = copy.deepcopy(dict(base))
    optimizer = dict(_mapping(recipe.get("optimizer"), "recipe.optimizer"))
    optimizer["epochs"] = epochs
    recipe["optimizer"] = optimizer
    recipe["checkpoint"] = {
        "every_epochs": checkpoint_every,
        **({"resume_path": str(checkpoint_path)} if resume else {}),
    }
    recipe["artifacts"] = {
        "root": str(artifact_root),
        "trained_adapter_dir": str(artifact_root / "adapter-trained"),
        "report_path": str(artifact_root / f"{task}_report.json"),
        "evaluation_report_path": str(artifact_root / f"{task}-evaluation.json"),
    }
    return recipe


def _semantic_report_view(report: Mapping[str, Any], task: str) -> dict[str, Any]:
    if report.get("evaluation_execution_policy") != CANONICAL_EVALUATION_POLICY:
        raise ContractError("preference report does not attest canonical held-out evaluation")
    numerical_policy = dict(
        _mapping(report.get("metal_numerical_policy"), "report.metal_numerical_policy")
    )
    if numerical_policy.get("schema_version") != "antfly_gemma4_metal_numerical_policy/v2":
        raise ContractError("preference report has an unsupported Metal numerical policy")
    fingerprint_flags = _integer(
        numerical_policy.get("fingerprint_flags"),
        "report.metal_numerical_policy.fingerprint_flags",
    )
    if fingerprint_flags > (1 << 64) - 1:
        raise ContractError("report.metal_numerical_policy.fingerprint_flags exceeds u64")
    _integer(
        numerical_policy.get("sparse_loss_chunk_rows"),
        "report.metal_numerical_policy.sparse_loss_chunk_rows",
        1,
    )
    _integer(
        numerical_policy.get("linear_cce_tile_vocab"),
        "report.metal_numerical_policy.linear_cce_tile_vocab",
        1,
    )
    expected_policy_fields = {
        "schema_version",
        "fingerprint_flags",
        "sparse_loss_chunk_rows",
        "linear_cce_tile_vocab",
        *METAL_NUMERICAL_POLICY_BOOLEAN_FIELDS,
    }
    if set(numerical_policy) != expected_policy_fields:
        missing = sorted(expected_policy_fields - set(numerical_policy))
        unexpected = sorted(set(numerical_policy) - expected_policy_fields)
        raise ContractError(
            "preference report Metal numerical policy field drift: "
            f"missing={missing}, unexpected={unexpected}"
        )
    for field in METAL_NUMERICAL_POLICY_BOOLEAN_FIELDS:
        if not isinstance(numerical_policy[field], bool):
            raise ContractError(f"report.metal_numerical_policy.{field}: expected boolean")
    common = {
        "execution_mode": report.get("execution_mode"),
        "dataset_format": report.get("dataset_format"),
        "training_seed": _integer(
            report.get("training_seed", 42), "report.training_seed"
        ),
        "policy_backend": report.get("policy_backend"),
        "optimizer_steps": _integer(report.get("optimizer_steps"), "report.optimizer_steps", 1),
        "micro_batch_steps": _integer(report.get("micro_batch_steps"), "report.micro_batch_steps", 1),
        "initial_logprob_parity": report.get("initial_logprob_parity"),
        "metal_numerical_policy": numerical_policy,
        "evaluation_execution_policy": report.get("evaluation_execution_policy"),
    }
    evaluation = dict(_mapping(report.get("evaluation"), "report.evaluation"))
    evaluation.pop("report_path", None)
    for field in (
        "sampling_seconds",
        "reference_scoring_seconds",
        "reward_loss_seconds",
        "loop_seconds",
    ):
        evaluation.pop(field, None)
    if evaluation.get("passed") is not True:
        raise ContractError("held-out evaluation did not pass")
    if task == "dpo":
        return {
            **common,
            **{field: report.get(field) for field in ("examples", "loss", "mean_reward_margin", "accuracy", "beta")},
            "policy_scoring_mode": report.get("policy_scoring_mode"),
            "training_microbatch_mode": report.get("training_microbatch_mode"),
            "initial_bucket_signature_parity": report.get("initial_bucket_signature_parity"),
            "sequence_length_policy": report.get("sequence_length_policy"),
            "evaluation": evaluation,
        }
    kl_control = dict(_mapping(report.get("kl_control"), "report.kl_control"))
    kl_control.pop("trace_path", None)
    reward_pipeline = dict(_mapping(report.get("reward_pipeline"), "report.reward_pipeline"))
    reward_pipeline.pop("trace_path", None)
    return {
        **common,
        **{
            field: report.get(field)
            for field in (
                "completions",
                "tokens",
                "groups",
                "loss",
                "pg_loss",
                "kl_loss",
                "mean_kl",
                "clip_fraction",
                "mean_reward",
                "reward_stddev",
                "policy_rescore_completions",
                "sampling_mode",
                "policy_logprob_mode",
                "training_microbatch_mode",
                "training_microbatch_batch_size",
                "training_physical_micro_batches_per_group",
                "reference_mode",
            )
        },
        "kl_control": kl_control,
        "reward_pipeline": reward_pipeline,
        "incremental_kv": report.get("incremental_kv"),
        "evaluation": evaluation,
    }


def _compare_semantic_reports(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    task: str,
    *,
    grpo_terminal_tolerances: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    expected_exact = copy.deepcopy(dict(expected))
    actual_exact = copy.deepcopy(dict(actual))
    comparison: dict[str, Any] = {"mode": "exact"}
    if task == "grpo":
        tolerances = grpo_terminal_tolerances or GRPO_TERMINAL_METAL_ABS_TOLERANCES
        expected_evaluation = dict(
            _mapping(expected_exact.get("evaluation"), "expected evaluation")
        )
        actual_evaluation = dict(
            _mapping(actual_exact.get("evaluation"), "actual evaluation")
        )
        expected_exact["evaluation"] = expected_evaluation
        actual_exact["evaluation"] = actual_evaluation
        fields: dict[str, Any] = {}
        for field, tolerance in tolerances.items():
            want = _finite_number(
                expected_evaluation.pop(field, None),
                f"expected evaluation.{field}",
            )
            got = _finite_number(
                actual_evaluation.pop(field, None),
                f"actual evaluation.{field}",
            )
            absolute_delta = abs(got - want)
            fields[field] = {
                "uninterrupted": want,
                "resumed": got,
                "absolute_delta": absolute_delta,
                "absolute_tolerance": tolerance,
                "passed": absolute_delta <= tolerance,
            }
            if absolute_delta > tolerance:
                raise ContractError(
                    f"resumed terminal Metal {field} drift {absolute_delta:.9g} "
                    f"exceeds tolerance {tolerance:.9g}"
                )
        comparison = {
            "mode": (
                "exact-except-bounded-terminal-metal-grpo-kl"
                if "loss" not in tolerances
                else "exact-except-bounded-terminal-metal-grpo-derived-loss-and-kl"
            ),
            "fields": fields,
        }
    if actual_exact != expected_exact:
        raise ContractError("resumed semantic trajectory differs from uninterrupted execution")
    return comparison


def _require_exact_final_checkpoint_parity(
    uninterrupted: Mapping[str, Any], resumed: Mapping[str, Any]
) -> dict[str, str]:
    checkpoint_sha256 = uninterrupted.get("sha256")
    if not isinstance(checkpoint_sha256, str) or resumed.get("sha256") != checkpoint_sha256:
        raise ContractError(
            "resumed final training checkpoint is not byte-identical to uninterrupted execution"
        )
    state_sha256 = uninterrupted.get("state_sha256")
    if not isinstance(state_sha256, str) or resumed.get("state_sha256") != state_sha256:
        raise ContractError(
            "resumed final preference sidecar is not byte-identical to uninterrupted execution"
        )
    return {
        "training_checkpoint_sha256": checkpoint_sha256,
        "checkpoint_state_sha256": state_sha256,
    }


def _require_changed_adapter(seed_sha256: str, trained_sha256: Any) -> str:
    if not isinstance(trained_sha256, str) or not trained_sha256.startswith("sha256:"):
        raise ContractError("trained adapter is missing a SHA-256 identity")
    if trained_sha256 == seed_sha256:
        raise ContractError("trained adapter is byte-identical to the seed adapter")
    return trained_sha256


def _require_adapter_tree_contract(
    seed_files: Mapping[str, Any], trained_files: Any
) -> None:
    trained = _mapping(trained_files, "trained adapter tree")
    if set(seed_files) != set(trained):
        raise ContractError("trained adapter file inventory differs from the seed adapter")
    mutable = {"adapter_model.safetensors", "antfly_finetune_manifest.json"}
    for relative_path, evidence in seed_files.items():
        if relative_path in mutable:
            continue
        if trained.get(relative_path) != evidence:
            raise ContractError(
                f"trained adapter changed immutable companion file: {relative_path}"
            )


def _require_reported_artifact(
    path_value: Any,
    digest_value: Any,
    expected_path: Path,
    where: str,
) -> dict[str, str]:
    if not isinstance(path_value, str):
        raise ContractError(f"{where}: missing artifact path")
    if not isinstance(digest_value, str) or not digest_value.startswith("sha256:"):
        raise ContractError(f"{where}: missing SHA-256 attestation")
    reported = Path(path_value).expanduser()
    if reported.is_symlink():
        raise ContractError(f"{where}: artifact must not be a symlink")
    resolved = _regular_file(reported, where)
    expected = expected_path.expanduser().resolve(strict=True)
    if resolved != expected:
        raise ContractError(f"{where}: report names the wrong artifact: {resolved}")
    actual_digest = _sha256(resolved)
    if actual_digest != digest_value:
        raise ContractError(f"{where}: reported digest does not match artifact bytes")
    return {"path": str(resolved), "sha256": actual_digest}


def _validate_report_artifacts(
    report: Mapping[str, Any],
    artifact_root: Path,
    task: str,
) -> dict[str, Any]:
    evaluation_summary = _mapping(report.get("evaluation"), "report.evaluation")
    expected_evaluation_path = artifact_root / f"{task}-evaluation.json"
    reported_evaluation_path = evaluation_summary.get("report_path")
    if not isinstance(reported_evaluation_path, str):
        raise ContractError("report.evaluation: missing standalone report path")
    if Path(reported_evaluation_path).expanduser().is_symlink():
        raise ContractError("standalone evaluation report must not be a symlink")
    evaluation_path = _regular_file(
        Path(reported_evaluation_path),
        "standalone evaluation report",
    )
    if evaluation_path != expected_evaluation_path.expanduser().resolve(strict=True):
        raise ContractError("main report names the wrong standalone evaluation report")
    evaluation = dict(_load_json(evaluation_path, "standalone evaluation report"))
    expected_schema = f"antfly_inference_finetune_{task}_evaluation/v{'2' if task == 'dpo' else '3'}"
    if evaluation.get("schema_version") != expected_schema:
        raise ContractError(f"expected {expected_schema} standalone evaluation report")
    if evaluation.get("status") != "passed" or evaluation.get("policy_backend") != "metal":
        raise ContractError("standalone evaluation report did not pass strict Metal admission")
    if evaluation.get("execution_policy") != CANONICAL_EVALUATION_POLICY:
        raise ContractError("standalone evaluation report has the wrong execution policy")
    if evaluation.get("metal_numerical_policy") != report.get("metal_numerical_policy"):
        raise ContractError("standalone evaluation report numerical policy differs from training")
    summary_fields = (
        ("examples", "loss", "mean_reward_margin", "accuracy")
        if task == "dpo"
        else (
            "groups",
            "completions",
            "mean_reward",
            "top_rank_mean_reward",
            "positive_reward_group_rate",
            "reward_stddev",
            "kl_loss",
            "mean_kl",
        )
    )
    if evaluation_summary.get("passed") is not True:
        raise ContractError("main report does not attest a passing held-out evaluation")
    for field in summary_fields:
        if evaluation_summary.get(field) != evaluation.get(field):
            raise ContractError(
                f"main and standalone evaluation reports disagree on {field}"
            )

    evidence: dict[str, Any] = {
        "evaluation_report": {
            "path": str(evaluation_path),
            "sha256": _sha256(evaluation_path),
        }
    }
    if task == "grpo":
        kl_control = _mapping(report.get("kl_control"), "report.kl_control")
        reward_pipeline = _mapping(report.get("reward_pipeline"), "report.reward_pipeline")
        evaluation_reward = _mapping(
            evaluation.get("reward_pipeline"),
            "evaluation.reward_pipeline",
        )
        if evaluation_reward.get("configuration_digest") != reward_pipeline.get(
            "configuration_digest"
        ):
            raise ContractError("training and evaluation reward configurations differ")
        evidence["training_kl_trace"] = _require_reported_artifact(
            kl_control.get("trace_path"),
            kl_control.get("trace_digest"),
            artifact_root / "grpo_kl_control_trace.jsonl",
            "training KL-control trace",
        )
        evidence["training_reward_trace"] = _require_reported_artifact(
            reward_pipeline.get("trace_path"),
            reward_pipeline.get("trace_digest"),
            artifact_root / "grpo_reward_trace.jsonl",
            "training reward trace",
        )
        evidence["evaluation_reward_trace"] = _require_reported_artifact(
            evaluation_reward.get("trace_path"),
            evaluation_reward.get("trace_digest"),
            artifact_root / "grpo_evaluation_reward_trace.jsonl",
            "evaluation reward trace",
        )
        normalized_reward = dict(evaluation_reward)
        normalized_reward.pop("trace_path", None)
        evaluation["reward_pipeline"] = normalized_reward

    for field in (
        "sampling_seconds",
        "reference_scoring_seconds",
        "reward_loss_seconds",
        "loop_seconds",
    ):
        evaluation.pop(field, None)
    return {"evidence": evidence, "semantic_evaluation_report": evaluation}


def _validate_outputs(
    uninterrupted_root: Path,
    resumed_root: Path,
    task: str,
    checkpoint_path: Path,
    expected_resume_epoch: int,
    compiled_sampling: bool = False,
) -> dict[str, Any]:
    uninterrupted = _load_json(uninterrupted_root / f"{task}_report.json", "uninterrupted report")
    resumed = _load_json(resumed_root / f"{task}_report.json", "resumed report")
    expected_schema = f"antfly_inference_finetune_{task}_report/v6"
    if uninterrupted.get("schema_version") != expected_schema or resumed.get("schema_version") != expected_schema:
        raise ContractError(f"expected {expected_schema} reports")
    if uninterrupted.get("policy_backend") != "metal" or resumed.get("policy_backend") != "metal":
        raise ContractError("preference run did not use the strict Metal policy")
    uninterrupted_resume = _mapping(uninterrupted.get("checkpoint_resume"), "uninterrupted checkpoint summary")
    resumed_resume = _mapping(resumed.get("checkpoint_resume"), "resumed checkpoint summary")
    fingerprint = uninterrupted_resume.get("run_fingerprint_sha256")
    if resumed_resume.get("run_fingerprint_sha256") != fingerprint:
        raise ContractError("resumed run fingerprint differs from uninterrupted execution")
    if resumed_resume.get("enabled") is not True:
        raise ContractError("resumed report does not attest checkpoint recovery")
    if _integer(resumed_resume.get("start_epoch"), "resumed start_epoch") != expected_resume_epoch:
        raise ContractError("resumed report recovered the wrong epoch")
    if Path(str(resumed_resume.get("checkpoint_path"))).resolve() != checkpoint_path.resolve():
        raise ContractError("resumed report names the wrong checkpoint")
    checkpoint = inspect_preference_checkpoint(checkpoint_path, task)
    if Path(str(resumed_resume.get("checkpoint_state_path"))).resolve() != Path(checkpoint["state_path"]):
        raise ContractError("resumed report names the wrong checkpoint sidecar")
    if resumed_resume.get("checkpoint_state_sha256") != checkpoint["state_sha256"]:
        raise ContractError("resumed report has the wrong checkpoint sidecar digest")
    if _integer(resumed_resume.get("checkpoint_epoch"), "resumed checkpoint_epoch") != checkpoint["epoch_index"]:
        raise ContractError("resumed report has the wrong durable checkpoint epoch")
    if fingerprint != checkpoint["run_fingerprint_sha256"]:
        raise ContractError("resumed report and checkpoint fingerprints differ")
    uninterrupted_cache_retired = uninterrupted_resume.get(
        "compiled_sampling_execution_cache_retired", False
    )
    resumed_cache_retired = resumed_resume.get(
        "compiled_sampling_execution_cache_retired", False
    )
    if not isinstance(uninterrupted_cache_retired, bool) or not isinstance(
        resumed_cache_retired, bool
    ):
        raise ContractError(
            "compiled-sampling execution-cache retirement attestation must be boolean"
        )
    if compiled_sampling:
        if not uninterrupted_cache_retired:
            raise ContractError(
                "uninterrupted compiled sampling did not retire its checkpoint cache"
            )
        if not resumed_cache_retired:
            raise ContractError(
                "resumed compiled sampling did not retire its bootstrap cache"
            )
    elif uninterrupted_cache_retired or resumed_cache_retired:
        raise ContractError(
            "non-compiled qualification unexpectedly retired a compiled-sampling cache"
        )
    expected = _semantic_report_view(uninterrupted, task)
    actual = _semantic_report_view(resumed, task)
    terminal_metal_float_comparison = _compare_semantic_reports(expected, actual, task)
    uninterrupted_artifacts = _validate_report_artifacts(
        uninterrupted,
        uninterrupted_root,
        task,
    )
    resumed_artifacts = _validate_report_artifacts(resumed, resumed_root, task)
    terminal_evaluation_artifact_float_comparison = _compare_semantic_reports(
        {"evaluation": uninterrupted_artifacts["semantic_evaluation_report"]},
        {"evaluation": resumed_artifacts["semantic_evaluation_report"]},
        task,
        grpo_terminal_tolerances=(
            GRPO_TERMINAL_METAL_EVALUATION_ARTIFACT_ABS_TOLERANCES
            if task == "grpo"
            else None
        ),
    )

    uninterrupted_adapter = _adapter_tree_evidence(
        uninterrupted_root / "adapter-trained",
        "uninterrupted adapter",
    )
    resumed_adapter = _adapter_tree_evidence(
        resumed_root / "adapter-trained",
        "resumed adapter",
    )
    if resumed_adapter["files"] != uninterrupted_adapter["files"]:
        raise ContractError("resumed final adapter tree is not byte-identical")
    adapter_sha = uninterrupted_adapter["adapter_model_sha256"]
    return {
        "run_fingerprint_sha256": fingerprint,
        "adapter_model_sha256": adapter_sha,
        "adapter_model_size_bytes": uninterrupted_adapter["adapter_model_size_bytes"],
        "adapter_tree": uninterrupted_adapter["files"],
        "adapter_manifest_identity": uninterrupted_adapter["manifest_identity"],
        "semantic_report": actual,
        "terminal_metal_float_comparison": terminal_metal_float_comparison,
        "terminal_evaluation_artifact_float_comparison": (
            terminal_evaluation_artifact_float_comparison
        ),
        "compiled_sampling_execution_cache_retirement": {
            "uninterrupted": uninterrupted_cache_retired,
            "resumed": resumed_cache_retired,
        },
        "verified_artifacts": {
            "uninterrupted": uninterrupted_artifacts["evidence"],
            "resumed": resumed_artifacts["evidence"],
        },
    }


def _command(binary: Path, recipe_path: Path) -> list[str]:
    return [str(binary), "inference", "finetune", "run", str(recipe_path)]


def qualify(args: argparse.Namespace) -> Mapping[str, Any]:
    binary = _regular_file(args.binary, "antfly binary", executable=True)
    base_recipe_path = _regular_file(args.recipe, "base recipe")
    base_recipe = copy.deepcopy(dict(_load_json(base_recipe_path, "base recipe")))
    task = base_recipe.get("recipe")
    if task not in ("dpo", "grpo"):
        raise ContractError("base recipe must be optimizer-backed dpo or grpo")
    if _mapping(base_recipe.get("execution"), "recipe.execution").get("mode") != "train":
        raise ContractError("base recipe must set execution.mode=train")
    if base_recipe.get("backend") != "metal":
        raise ContractError("base recipe must set backend=metal")
    if args.epochs < 2 or not 0 < args.interrupt_after_epoch < args.epochs:
        raise ContractError("interrupt epoch must be between 1 and epochs-1")
    if args.timeout_seconds <= 0 or args.poll_seconds <= 0:
        raise ContractError("timeouts must be positive")
    compiled_sampling = bool(getattr(args, "compiled_sampling", False))
    if compiled_sampling and task != "grpo":
        raise ContractError("--compiled-sampling is valid only for GRPO")
    if compiled_sampling and args.incremental_kv:
        raise ContractError(
            "compiled GRPO sampling conflicts with incremental-KV qualification"
        )

    model_config = dict(_mapping(base_recipe.get("model"), "recipe.model"))
    model_value = model_config.get("path")
    if not isinstance(model_value, str):
        raise ContractError("qualification requires a pinned model path")
    model = _closed_immutable_path(Path(model_value), "model")
    if model.is_file() and model.suffix.lower() != ".gguf":
        raise ContractError("model path must be a directory or GGUF file")
    reference_value = model_config.get("reference_path")
    if reference_value is not None:
        if not isinstance(reference_value, str):
            raise ContractError("qualification requires a pinned reference model path")
        reference = _closed_immutable_path(Path(reference_value), "reference model")
        if reference != model:
            raise ContractError("preference qualification requires model and reference paths to match")
        model_config["reference_path"] = str(reference)
    model_config["path"] = str(model)
    direct_gguf_training = args.direct_gguf_training or args.experimental_gguf_qlora
    if model.is_file() and not direct_gguf_training:
        raise ContractError("direct GGUF qualification requires --direct-gguf-training")
    if direct_gguf_training and not model.is_file():
        raise ContractError("--direct-gguf-training requires a direct GGUF model file")
    if direct_gguf_training and args.incremental_kv:
        raise ContractError(
            "direct GGUF GRPO plus incremental KV is not qualified; "
            "use canonical direct-GGUF GRPO or incremental-KV GRPO with safetensors"
        )
    if direct_gguf_training and compiled_sampling:
        raise ContractError(
            "compiled GRPO sampling is not qualified for direct GGUF training"
        )
    if direct_gguf_training:
        model_config["allow_direct_gguf_training"] = True
    base_recipe["model"] = model_config
    if args.incremental_kv:
        if task != "grpo":
            raise ContractError("--incremental-kv is valid only for GRPO")
        runtime = dict(base_recipe.get("runtime") or {})
        runtime.update(
            {
                "grpo_incremental_kv": True,
                "grpo_incremental_kv_batch_active": not args.incremental_kv_serial,
                "grpo_incremental_kv_clone_prompt_tail": args.incremental_kv_clone_prompt_tail,
                "grpo_incremental_kv_shadow_exact": args.incremental_kv_shadow_exact,
            }
        )
        base_recipe["runtime"] = runtime
    _apply_compiled_sampling_recipe_contract(
        base_recipe, task, compiled_sampling
    )
    adapter_config = dict(_mapping(base_recipe.get("adapter"), "recipe.adapter"))
    adapter_value = adapter_config.get("path")
    if not isinstance(adapter_value, str):
        raise ContractError("qualification requires a pinned seed adapter path")
    adapter = _closed_immutable_path(Path(adapter_value), "seed adapter")
    if not adapter.is_dir():
        raise ContractError("seed adapter must be a directory")
    adapter_config["path"] = str(adapter)
    base_recipe["adapter"] = adapter_config

    dataset_config = dict(_mapping(base_recipe.get("dataset"), "recipe.dataset"))
    eval_config = dict(_mapping(base_recipe.get("eval"), "recipe.eval"))
    train_path = _effective_regular_file(
        dataset_config.get("train_path"),
        dataset_config.get("path"),
        "training dataset",
    )
    eval_path = _effective_regular_file(
        eval_config.get("path"),
        dataset_config.get("eval_path"),
        "evaluation dataset",
    )
    if dataset_config.get("train_path") is not None:
        dataset_config["train_path"] = str(train_path)
    if dataset_config.get("path") is not None:
        dataset_config["path"] = str(train_path)
    if eval_config.get("path") is not None:
        eval_config["path"] = str(eval_path)
    if dataset_config.get("eval_path") is not None:
        dataset_config["eval_path"] = str(eval_path)
    base_recipe["dataset"] = dataset_config
    base_recipe["eval"] = eval_config

    output_root = args.output_dir.expanduser().resolve()
    if output_root.exists():
        raise ContractError(f"output directory already exists: {output_root}")
    immutable_inputs = {
        "binary": binary,
        "base recipe": base_recipe_path,
        "model": model,
        "adapter": adapter,
        "training dataset": train_path,
        "evaluation dataset": eval_path,
    }
    for name, path in immutable_inputs.items():
        if _paths_overlap(output_root, path):
            raise ContractError(f"output directory overlaps immutable {name}: {path}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir()
    uninterrupted_root = output_root / "uninterrupted"
    interrupted_root = output_root / "interrupted-unpublished"
    resumed_root = output_root / "resumed"
    checkpoint_name = f"gemma4_{task}_trainer_state.safetensors"
    uninterrupted_checkpoint = uninterrupted_root / checkpoint_name
    interrupted_checkpoint = interrupted_root / checkpoint_name

    recipes: dict[str, Path] = {}
    for name, root, checkpoint, resume in (
        ("uninterrupted", uninterrupted_root, uninterrupted_checkpoint, False),
        ("interrupted", interrupted_root, interrupted_checkpoint, False),
        ("resumed", resumed_root, interrupted_checkpoint, True),
    ):
        path = output_root / f"{name}.recipe.json"
        _write_json(
            path,
            _recipe_variant(
                base_recipe,
                task,
                root,
                checkpoint,
                args.epochs,
                args.interrupt_after_epoch,
                resume,
            ),
        )
        recipes[name] = path

    immutable_roots = {
        "model": model,
        "adapter": adapter,
        "train_dataset": train_path,
        "eval_dataset": eval_path,
    }
    snapshots_before = {name: _tree_snapshot(path) for name, path in immutable_roots.items()}
    binary_sha256 = _sha256(binary)
    base_recipe_sha256 = _sha256(base_recipe_path)
    seed_adapter_evidence = _adapter_tree_evidence(adapter, "seed adapter")
    input_evidence = {
        "binary": {"path": str(binary), "sha256": binary_sha256},
        "base_recipe": {"path": str(base_recipe_path), "sha256": base_recipe_sha256},
        **{
            name: {
                "path": str(path),
                "snapshot_sha256": _snapshot_digest(snapshots_before[name]),
                "sha256": _sha256(path) if path.is_file() else None,
            }
            for name, path in immutable_roots.items()
        },
    }
    input_evidence["adapter"]["checkpoint_sha256"] = seed_adapter_evidence[
        "adapter_model_sha256"
    ]
    input_evidence["adapter"]["files"] = seed_adapter_evidence["files"]
    env = _strict_environment()
    effective_contract_env = dict(STRICT_METAL_ENV)
    if compiled_sampling:
        env[COMPILED_GRPO_SAMPLING_ENV] = "1"
        effective_contract_env[COMPILED_GRPO_SAMPLING_ENV] = "1"
    if args.experimental_gguf_qlora:
        env["ANTFLY_EXPERIMENTAL_GEMMA4_GGUF_QLORA"] = "1"
        effective_contract_env["ANTFLY_EXPERIMENTAL_GEMMA4_GGUF_QLORA"] = "1"

    commands = {name: _command(binary, path) for name, path in recipes.items()}
    uninterrupted_run = _run_to_completion(
        commands["uninterrupted"],
        env,
        output_root / "uninterrupted.stdout.log",
        output_root / "uninterrupted.stderr.log",
        args.timeout_seconds,
    )
    uninterrupted_state = inspect_preference_checkpoint(uninterrupted_checkpoint, task)
    if uninterrupted_state["epoch_index"] != args.epochs:
        raise ContractError("uninterrupted checkpoint did not reach the final epoch")

    interrupted_run, interrupted_state = _run_and_interrupt(
        commands["interrupted"],
        env,
        interrupted_checkpoint,
        task,
        args.interrupt_after_epoch,
        output_root / "interrupted.stdout.log",
        output_root / "interrupted.stderr.log",
        args.timeout_seconds,
        args.poll_seconds,
    )
    if (interrupted_root / "adapter-trained").exists():
        raise ContractError("interrupted command published a final adapter")

    resumed_run = _run_to_completion(
        commands["resumed"],
        env,
        output_root / "resumed.stdout.log",
        output_root / "resumed.stderr.log",
        args.timeout_seconds,
    )
    resumed_state = inspect_preference_checkpoint(interrupted_checkpoint, task)
    if resumed_state["epoch_index"] != args.epochs:
        raise ContractError("resumed checkpoint did not reach the final epoch")
    exact_checkpoint_parity = _require_exact_final_checkpoint_parity(
        uninterrupted_state,
        resumed_state,
    )
    parity = _validate_outputs(
        uninterrupted_root,
        resumed_root,
        task,
        interrupted_checkpoint,
        args.interrupt_after_epoch,
        compiled_sampling,
    )
    parity.update(exact_checkpoint_parity)
    if compiled_sampling:
        semantic_report = _mapping(
            parity.get("semantic_report"), "parity.semantic_report"
        )
        if semantic_report.get("sampling_mode") != COMPILED_GRPO_SAMPLING_MODE:
            raise ContractError(
                "compiled-sampling qualification did not execute the compiled GRPO sampling mode"
            )
        if semantic_report.get("policy_logprob_mode") != COMPILED_GRPO_POLICY_LOGPROB_MODE:
            raise ContractError(
                "compiled-sampling qualification did not use canonical eager policy log-probs"
            )
        if _integer(
            semantic_report.get("policy_rescore_completions"),
            "parity.semantic_report.policy_rescore_completions",
        ) != _integer(
            semantic_report.get("completions"),
            "parity.semantic_report.completions",
            1,
        ):
            raise ContractError(
                "compiled-sampling qualification did not canonically rescore every completion"
            )
        if semantic_report.get("incremental_kv") is not None:
            raise ContractError(
                "compiled-sampling qualification unexpectedly emitted incremental-KV telemetry"
            )
    report_training_seed = _integer(
        parity.get("semantic_report", {}).get("training_seed"),
        "parity.semantic_report.training_seed",
    )
    for checkpoint_name, checkpoint_state in (
        ("uninterrupted_final", uninterrupted_state),
        ("interrupted_boundary", interrupted_state),
        ("resumed_final", resumed_state),
    ):
        if _integer(
            checkpoint_state.get("trainer_seed"),
            f"{checkpoint_name}.trainer_seed",
        ) != report_training_seed:
            raise ContractError(
                f"{checkpoint_name} trainer seed differs from the final report"
            )
    if args.incremental_kv:
        semantic_incremental = _mapping(
            parity.get("semantic_report", {}).get("incremental_kv"),
            "parity.semantic_report.incremental_kv",
        )
        expected_groups = _integer(
            parity.get("semantic_report", {}).get("groups"),
            "parity.semantic_report.groups",
            1,
        )
        if _integer(semantic_incremental.get("groups"), "incremental_kv.groups") != expected_groups:
            raise ContractError("incremental-KV telemetry does not cover the whole resumed run")
        if _integer(
            semantic_incremental.get("host_logit_fallbacks"),
            "incremental_kv.host_logit_fallbacks",
        ) != 0:
            raise ContractError("incremental-KV qualification used a host-logit fallback")
        for checkpoint_name in ("uninterrupted_final", "resumed_final"):
            checkpoint_telemetry = _mapping(
                {
                    "uninterrupted_final": uninterrupted_state,
                    "resumed_final": resumed_state,
                }[checkpoint_name].get("incremental_kv"),
                f"{checkpoint_name}.incremental_kv",
            )
            if checkpoint_telemetry != semantic_incremental:
                raise ContractError(
                    f"{checkpoint_name} incremental-KV telemetry differs from final report"
                )
    _require_changed_adapter(
        input_evidence["adapter"]["checkpoint_sha256"],
        parity.get("adapter_model_sha256"),
    )
    _require_adapter_tree_contract(
        seed_adapter_evidence["files"],
        parity.get("adapter_tree"),
    )
    if parity.get("adapter_manifest_identity") != seed_adapter_evidence["manifest_identity"]:
        raise ContractError("trained adapter changed the seed adapter identity contract")

    for name, path in immutable_roots.items():
        if _tree_snapshot(path) != snapshots_before[name]:
            raise ContractError(f"immutable {name} input changed during qualification")
    if _sha256(binary) != binary_sha256:
        raise ContractError("antfly binary changed during qualification")
    if _sha256(base_recipe_path) != base_recipe_sha256:
        raise ContractError("base recipe changed during qualification")

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "pass",
        "task": task,
        "contract": {
            "backend": "metal",
            "epochs": args.epochs,
            "interrupt_after_epoch": args.interrupt_after_epoch,
            "direct_gguf_training": direct_gguf_training,
            "experimental_direct_gguf_qlora": args.experimental_gguf_qlora,
            "incremental_kv": args.incremental_kv,
            "incremental_kv_batch_active": args.incremental_kv and not args.incremental_kv_serial,
            "incremental_kv_clone_prompt_tail": args.incremental_kv_clone_prompt_tail,
            "incremental_kv_shadow_exact": args.incremental_kv_shadow_exact,
            "compiled_sampling": compiled_sampling,
            "environment_policy_sha256": ENVIRONMENT_POLICY_SHA256,
            "strict_metal_environment": effective_contract_env,
            "sanitized_environment_prefixes": list(SANITIZED_ENV_PREFIXES),
            "sanitized_environment_names": sorted(SANITIZED_ENV_NAMES),
        },
        "inputs": input_evidence,
        "recipes": {name: str(path) for name, path in recipes.items()},
        "commands": commands,
        "runs": {
            "uninterrupted": uninterrupted_run,
            "interrupted": interrupted_run,
            "resumed": resumed_run,
        },
        "checkpoints": {
            "uninterrupted_final": uninterrupted_state,
            "interrupted_boundary": interrupted_state,
            "resumed_final": resumed_state,
        },
        "parity": parity,
    }
    _write_json(output_root / "qualification_report.json", report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--interrupt-after-epoch", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--poll-seconds", type=float, default=0.02)
    parser.add_argument(
        "--direct-gguf-training",
        action="store_true",
        help="admit a direct GGUF base through the typed preference-training recipe contract",
    )
    parser.add_argument(
        "--experimental-gguf-qlora",
        action="store_true",
        help="legacy alias for --direct-gguf-training",
    )
    parser.add_argument(
        "--incremental-kv",
        action="store_true",
        help="qualify checkpointed exact paged-KV GRPO token selection",
    )
    parser.add_argument(
        "--incremental-kv-serial",
        action="store_true",
        help="disable same-position active-candidate batching",
    )
    parser.add_argument(
        "--incremental-kv-clone-prompt-tail",
        action="store_true",
        help="clone the segmented prompt tail on device",
    )
    parser.add_argument(
        "--incremental-kv-shadow-exact",
        action="store_true",
        help="run the exact legacy sampler for the first group",
    )
    parser.add_argument(
        "--compiled-sampling",
        action="store_true",
        help=(
            "qualify the default-off compiled multi-token GRPO sampler; "
            "incremental KV and direct GGUF are rejected"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        report = qualify(parse_args(argv))
    except (ContractError, OSError, ValueError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    comparison_mode = report["parity"]["terminal_metal_float_comparison"]["mode"]
    print(
        f"PASS: interrupted/resumed Gemma4 {report['task'].upper()} Metal trajectory "
        f"({comparison_mode}); adapter={report['parity']['adapter_model_sha256']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
