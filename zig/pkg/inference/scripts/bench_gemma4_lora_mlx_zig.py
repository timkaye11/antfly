#!/usr/bin/env python3
"""Validate same-Mac Gemma4 LoRA benchmark samples and enforce gates.

This harness does not pretend cross-hardware numbers are comparable.  Every
sample is one fresh process, and a campaign is rejected unless Antfly and
MLX-LM used the same locked model, immutable prepared workload, host identity,
protocol, and paired repetition schedule. Framework execution order must
alternate by repetition.

The framework-specific runners are responsible for device synchronization and
emitting one JSON sample conforming to ``gemma4_mlx_benchmark.schema.json``.
This script owns validation, aggregation, and release gate decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from gemma4_oracle_contract import (
    BENCHMARK_PRODUCER_SOURCE_SCHEMA_VERSION,
    BENCHMARK_SAMPLE_SCHEMA_VERSION,
    ContractError,
    LOCK_PATH,
    MLX_NATIVE_ARTIFACT_INVENTORY_SCHEMA_VERSION,
    canonical_mlx_native_artifact_inventory_sha256,
    hardware_fingerprint,
    load_json,
    load_lock,
    lock_digest,
    prefixed_sha256,
    require_exact_keys,
    validate_benchmark_producer_source,
    validate_target_inventory,
    write_json,
)
from run_gemma4_lora_benchmark_campaign import verify_complete_campaign_manifest


FRAMEWORKS = ("antfly-zig-metal", "mlx-lm")
MODELS = ("gemma-4-E2B-it", "gemma-4-E4B-it")
PRESETS = ("peft-qv", "text-all-linear")
SEQUENCE_LENGTHS = (128, 512, 2048)
GRADIENT_ACCUMULATION = (1, 4)
MATRIX_KEY_FIELDS = 9
TIMED_UNIT = "optimizer-step-including-grad-accumulation"
TARGET_INVENTORY_DOMAIN = b"antfly_gemma4_target_inventory/v1\0"
INITIAL_ADAPTER_DOMAIN = b"antfly_gemma4_initial_adapter_semantics/v1\0"
SEMANTIC_CONTRACT_DOMAIN = b"antfly_gemma4_lora_benchmark_semantics/v3\0"
PRECISION_EVIDENCE_SCHEMA_VERSION = "antfly_gemma4_mlx_precision_evidence/v1"
PRECISION_EVIDENCE_REFERENCE_SCHEMA_VERSION = (
    "antfly_gemma4_mlx_precision_evidence_reference/v1"
)
PRECISION_RUNNER_SCHEMA_VERSION = BENCHMARK_PRODUCER_SOURCE_SCHEMA_VERSION
PRECISION_INVENTORY_DOMAIN = b"antfly_gemma4_mlx_precision_tensor_inventory/v1\0"
PRECISION_MOMENT_INVENTORY_DOMAIN = b"antfly_gemma4_mlx_precision_moment_inventory/v1\0"
PRECISION_SAMPLE_BINDING_DOMAIN = b"antfly_gemma4_mlx_precision_sample_binding/v1\0"
EXECUTION_CONTRACT_DOMAIN = b"antfly_gemma4_benchmark_execution_contract/v1\0"
IMPLEMENTATION_IDENTITY_DOMAIN = b"antfly_gemma4_benchmark_implementation_identity/v1\0"
MLX_RUNNER_RELATIVE_PATH = "zig/pkg/inference/scripts/run_gemma4_lora_mlx_benchmark.py"
ANTFLY_RUNNER_RELATIVE_PATH = "zig/pkg/inference/scripts/run_antfly_gemma4_lora_benchmark.py"
SYNC_POINT = "after-optimizer-update-before-timer-stop-every-window"
ZIG_EXECUTION_EVIDENCE_SCHEMA_VERSION = "antfly_gemma4_zig_execution_evidence/v2"
ZIG_PHASE_EVIDENCE_FIELDS = (
    "graph_build_ns", "runtime_input_ns", "train_step_ns", "compile_ns",
    "autodiff_ns", "execute_ns", "extract_ns", "optimizer_update_ns",
    "device_optimizer_ns", "total_ns", "metal_frame_wait_ns", "metal_frame_gpu_ns",
    "graph_executor_plan_build_ns", "graph_executor_buffer_plan_build_ns",
)
ZIG_COMMAND_PLAN_EVIDENCE_FIELDS = (
    "graph_executor_partitions", "graph_executor_command_dispatches",
    "graph_executor_planned_dispatches", "graph_executor_runtime_region_dispatches",
    "graph_executor_runtime_region_active_regions", "graph_executor_runtime_region_covered_nodes",
    "graph_executor_runtime_region_elided_nodes", "graph_executor_runtime_region_plan_compiles",
    "graph_executor_runtime_region_plan_reuses", "graph_executor_plan_cache_hits",
    "graph_executor_plan_cache_misses", "metal_lora_backward_regions",
    "metal_low_rank_lora_backward_regions", "metal_rank_adapter_backward_regions",
    "metal_ffn_gelu_backward_regions", "metal_head_mlp_forward_regions",
    "metal_head_mlp_backward_regions", "metal_gemma4_bf16_gate_up_fused_calls",
    "metal_gemma4_bf16_gate_up_backward_input_sum_fused_calls",
    "metal_linear_cce_forward_calls", "metal_linear_cce_backward_calls",
    "metal_linear_cce_forward_state_hits", "metal_linear_cce_forward_state_misses",
    "metal_linear_cce_peak_scratch_bytes",
    "metal_command_dot_general_dispatches",
    "metal_command_head_dot_dispatches", "metal_command_transpose_dispatches",
    "metal_command_gather_dispatches", "metal_command_reduce_dispatches",
    "metal_command_elementwise_dispatches", "metal_command_activation_dispatches",
    "metal_command_activation_backward_dispatches", "metal_command_other_dispatches",
    "metal_last_frame_compute_encoders", "metal_last_frame_blit_encoders",
    "metal_last_frame_planned_scopes", "metal_last_frame_planned_barriers",
    "metal_last_frame_planned_command_ops",
)
ZIG_COMMAND_ATTRIBUTION_FIELDS = (
    "metal_command_dot_general_dispatches", "metal_command_head_dot_dispatches",
    "metal_command_transpose_dispatches", "metal_command_gather_dispatches",
    "metal_command_reduce_dispatches", "metal_command_elementwise_dispatches",
    "metal_command_activation_dispatches", "metal_command_activation_backward_dispatches",
    "metal_command_other_dispatches",
)


def mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where}: expected object")
    return value


def integer(value: Any, where: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContractError(f"{where}: expected integer >= {minimum}")
    return value


def finite(value: Any, where: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}: expected number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (not positive and result < 0):
        relation = "positive" if positive else "non-negative"
        raise ContractError(f"{where}: expected finite {relation} number")
    return result


def finite_signed(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}: expected number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{where}: expected finite number")
    return result


def string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{where}: expected non-empty string")
    return value


def sha256_identity(value: Any, where: str) -> str:
    result = string(value, where)
    if re.fullmatch(r"sha256:[0-9a-f]{64}", result) is None:
        raise ContractError(f"{where}: expected sha256:<64 lowercase hex characters>")
    return result


def sha256_hex_identity(value: Any, where: str) -> str:
    result = string(value, where)
    if re.fullmatch(r"[0-9a-f]{64}", result) is None:
        raise ContractError(f"{where}: expected 64 lowercase SHA-256 hex characters")
    return result


def benchmark_workload_sha256(
    input_rows: Sequence[Sequence[int]],
    label_rows: Sequence[Sequence[int]],
    attention_mask_rows: Sequence[Sequence[int]],
) -> str:
    """Hash the ordered model-input rows consumed by one optimizer step.

    Rows are ordered exactly as the runner submits them across microbatches and
    gradient-accumulation passes. Length prefixes preserve row boundaries.
    Signed little-endian 64-bit values cover token IDs, attention masks, and the
    -100 label ignore sentinel and are straightforward to reproduce outside
    Python.
    """
    if len(input_rows) != len(label_rows) or len(input_rows) != len(attention_mask_rows) or not input_rows:
        raise ContractError("benchmark workload requires equal, non-empty input, label, and attention-mask row sets")
    hasher = hashlib.sha256(b"antfly_gemma4_benchmark_workload/v1\0")
    hasher.update(struct.pack("<Q", len(input_rows)))
    for row_index, (input_ids, labels, attention_mask) in enumerate(
        zip(input_rows, label_rows, attention_mask_rows)
    ):
        if len(input_ids) != len(labels) or len(input_ids) != len(attention_mask) or not input_ids:
            raise ContractError(
                f"benchmark workload row {row_index} must have equal, non-empty token, label, and attention-mask arrays"
            )
        hasher.update(struct.pack("<Q", len(input_ids)))
        for field, values in (("input_ids", input_ids), ("labels", labels), ("attention_mask", attention_mask)):
            for value_index, value in enumerate(values):
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ContractError(f"benchmark workload {field}[{row_index}][{value_index}] must be an integer")
                if field == "input_ids" and value < 0:
                    raise ContractError(f"benchmark workload input_ids[{row_index}][{value_index}] must be non-negative")
                if field == "labels" and value < 0 and value != -100:
                    raise ContractError(f"benchmark workload labels[{row_index}][{value_index}] permits only -100 as a negative value")
                if field == "attention_mask" and value not in (0, 1):
                    raise ContractError(f"benchmark workload attention_mask[{row_index}][{value_index}] must be zero or one")
                try:
                    hasher.update(struct.pack("<q", value))
                except struct.error as exc:
                    raise ContractError(
                        f"benchmark workload {field}[{row_index}][{value_index}] is outside signed 64-bit range"
                    ) from exc
    return f"sha256:{hasher.hexdigest()}"


def canonical_target_inventory_sha256(canonical_modules: Sequence[str]) -> str:
    """Hash the complete sorted canonical LoRA module inventory."""
    modules = list(canonical_modules)
    if not modules or len(modules) != len(set(modules)):
        raise ContractError("target inventory must be non-empty and contain no duplicates")
    if modules != sorted(modules):
        raise ContractError("target inventory must be sorted by canonical module name")
    hasher = hashlib.sha256(TARGET_INVENTORY_DOMAIN)
    hasher.update(struct.pack("<Q", len(modules)))
    for index, module in enumerate(modules):
        if not isinstance(module, str) or not module.startswith("model."):
            raise ContractError(f"target inventory module {index} is not canonical")
        encoded = module.encode("utf-8")
        hasher.update(struct.pack("<Q", len(encoded)))
        hasher.update(encoded)
    return f"sha256:{hasher.hexdigest()}"


def canonical_initial_adapter_sha256(tensors: Sequence[Mapping[str, Any]]) -> str:
    """Hash canonical F32 LoRA A/B values independently of container format."""
    rows: list[tuple[str, str, tuple[int, ...], bytes]] = []
    for index, raw in enumerate(tensors):
        tensor = mapping(raw, f"initial adapter tensor {index}")
        require_exact_keys(tensor, ("module", "role", "shape", "values"), where=f"initial adapter tensor {index}")
        module = string(tensor["module"], f"initial adapter tensor {index}.module")
        if not module.startswith("model."):
            raise ContractError(f"initial adapter tensor {index}.module is not canonical")
        role = tensor["role"]
        if role not in ("lora_A", "lora_B"):
            raise ContractError(f"initial adapter tensor {index}.role is unsupported")
        raw_shape = tensor["shape"]
        if not isinstance(raw_shape, list) or not raw_shape:
            raise ContractError(f"initial adapter tensor {index}.shape must be non-empty")
        shape = tuple(integer(dim, f"initial adapter tensor {index}.shape[]", 1) for dim in raw_shape)
        expected_values = math.prod(shape)
        raw_values = tensor["values"]
        if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)) or len(raw_values) != expected_values:
            raise ContractError(f"initial adapter tensor {index}.values does not match shape")
        encoded_values = bytearray()
        for value_index, value in enumerate(raw_values):
            number = finite_signed(value, f"initial adapter tensor {index}.values[{value_index}]")
            try:
                encoded = struct.pack("<f", number)
            except (OverflowError, struct.error) as exc:
                raise ContractError(f"initial adapter tensor {index}.values[{value_index}] is outside F32") from exc
            if not math.isfinite(struct.unpack("<f", encoded)[0]):
                raise ContractError(f"initial adapter tensor {index}.values[{value_index}] is non-finite in F32")
            encoded_values.extend(encoded)
        rows.append((module, role, shape, bytes(encoded_values)))
    rows.sort(key=lambda row: (row[0], row[1]))
    identities = [(row[0], row[1]) for row in rows]
    if not rows or len(identities) != len(set(identities)):
        raise ContractError("initial adapter must contain one unique LoRA A/B tensor identity")
    hasher = hashlib.sha256(INITIAL_ADAPTER_DOMAIN)
    hasher.update(struct.pack("<Q", len(rows)))
    for module, role, shape, values in rows:
        for text_value in (module, role, "float32"):
            encoded = text_value.encode("utf-8")
            hasher.update(struct.pack("<Q", len(encoded)))
            hasher.update(encoded)
        hasher.update(struct.pack("<Q", len(shape)))
        for dim in shape:
            hasher.update(struct.pack("<Q", dim))
        hasher.update(struct.pack("<Q", len(values)))
        hasher.update(values)
    return f"sha256:{hasher.hexdigest()}"


def canonical_json_sha256(domain: bytes, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return f"sha256:{hashlib.sha256(domain + encoded).hexdigest()}"


def canonical_tensor_inventory_sha256(tensors: Sequence[Mapping[str, Any]]) -> str:
    """Hash an ordered logical tensor inventory, including names, dtypes, and shapes."""
    encoded = json.dumps(
        list(tensors), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(PRECISION_INVENTORY_DOMAIN + encoded).hexdigest()}"


def canonical_moment_inventory_sha256(moments: Sequence[Mapping[str, Any]]) -> str:
    """Hash an ordered optimizer-moment inventory with its parameter/role binding."""
    encoded = json.dumps(
        list(moments), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(PRECISION_MOMENT_INVENTORY_DOMAIN + encoded).hexdigest()}"


def canonical_precision_sample_binding_sha256(sample: Mapping[str, Any]) -> str:
    """Bind every final sample claim without creating a digest-reference cycle."""
    payload = dict(sample)
    payload.pop("precision_evidence", None)
    return canonical_json_sha256(PRECISION_SAMPLE_BINDING_DOMAIN, payload)


def expected_semantic_contract(lock: Mapping[str, Any], case: Mapping[str, Any]) -> dict[str, Any]:
    frozen = mapping(lock["benchmark_contract"], "benchmark_contract")
    optimizer = dict(mapping(frozen["optimizer"], "benchmark_contract.optimizer"))
    optimizer["gradient_accumulation_steps"] = case["grad_accum"]
    payload: dict[str, Any] = {
        "schema_version": frozen["schema_version"],
        "precision": dict(mapping(frozen["precision"], "benchmark_contract.precision")),
        "optimizer": optimizer,
        "determinism": dict(mapping(frozen["determinism"], "benchmark_contract.determinism")),
        "runtime": dict(mapping(frozen["runtime"], "benchmark_contract.runtime")),
        "memory": dict(mapping(frozen["memory"], "benchmark_contract.memory")),
    }
    binding = {
        "model_key": case["model_key"],
        "revision": case["revision"],
        "local_artifact_sha256": case["local_artifact_sha256"],
        "target_preset": case["target_preset"],
        "rank": case["rank"],
        "alpha": float(case["alpha"]),
        "sequence_length": case["sequence_length"],
        "grad_accum": case["grad_accum"],
        "microbatch": case["microbatch"],
        "prepared": dict(mapping(case["prepared"], "case.prepared")),
        "initial_adapter": dict(mapping(case["initial_adapter"], "case.initial_adapter")),
        "target_inventory": dict(mapping(case["target_inventory"], "case.target_inventory")),
        "contract": payload,
    }
    payload["sha256"] = canonical_json_sha256(SEMANTIC_CONTRACT_DOMAIN, binding)
    return payload


@dataclass(frozen=True)
class Sample:
    path: Path
    payload: dict[str, Any]
    case_key: tuple[Any, ...]
    producer_source: dict[str, Any]


def case_key(case: Mapping[str, Any]) -> tuple[Any, ...]:
    prepared = mapping(case["prepared"], "case.prepared")
    prepared_identity = (
        prepared["schema_version"],
        prepared["artifact_sha256"],
        prepared["example_index"],
        prepared["source_dataset_sha256"],
        prepared["source_record_sha256"],
        prepared["rendered_chat_sha256"],
        prepared["workload_sha256"],
    )
    target_inventory = mapping(case["target_inventory"], "case.target_inventory")
    semantic_identity = (
        prepared_identity,
        case["initial_adapter"]["semantic_sha256"],
        case["initial_adapter"]["tensor_count"],
        target_inventory["sha256"],
        target_inventory["module_count"],
        tuple(target_inventory["canonical_modules"]),
    )
    return (
        case["model_key"],
        case["revision"],
        case["local_artifact_sha256"],
        case["target_preset"],
        case["rank"],
        float(case["alpha"]),
        case["sequence_length"],
        case["grad_accum"],
        case["microbatch"],
        semantic_identity,
    )


def display_matrix_case(key: tuple[Any, ...]) -> str:
    return f"{key[0]}/{key[3]}/seq={key[6]}/ga={key[7]}"


def display_case(key: tuple[Any, ...]) -> str:
    workload = key[MATRIX_KEY_FIELDS][0][6].removeprefix("sha256:")[:12]
    return f"{display_matrix_case(key)}/workload={workload}"


def execution_contract_sha256(sample: Sample) -> str:
    """Identity for execution semantics that must match across paired repetitions."""
    payload = sample.payload
    contract = {
        "case": payload["case"],
        "semantic_contract": payload["semantic_contract"],
        "protocol": payload["protocol"],
    }
    return canonical_json_sha256(EXECUTION_CONTRACT_DOMAIN, contract)


def campaign_implementation_identity(sample: Sample) -> dict[str, Any]:
    """Return the path-independent runtime identity that must be campaign-wide."""
    implementation = mapping(sample.payload["implementation"], "implementation")
    producer = dict(sample.producer_source)
    if sample.payload["framework"] == "antfly-zig-metal":
        return {
            "framework": "antfly-zig-metal",
            "producer_source": producer,
            "antfly": dict(mapping(implementation["antfly"], "implementation.antfly")),
        }

    mlx = mapping(implementation["mlx"], "implementation.mlx")
    inventory = mapping(mlx["native_artifact_inventory"], "implementation.mlx.native_artifact_inventory")
    attestation = mapping(mlx["build_attestation"], "implementation.mlx.build_attestation")
    python = mapping(implementation["python"], "implementation.python")
    # Absolute locations are deliberately excluded. The hashes, revisions,
    # roles, sizes, and relative runtime layout are the portable identity.
    return {
        "framework": "mlx-lm",
        "producer_source": producer,
        "mlx": {
            "version": mlx["version"],
            "source_revision": mlx["source_revision"],
            "source_clean": mlx["source_clean"],
            "native_artifact_inventory": {
                "schema_version": inventory["schema_version"],
                "sha256": inventory["sha256"],
                "artifacts": list(inventory["artifacts"]),
            },
            "build_attestation": {
                key: value for key, value in attestation.items() if key != "path"
            },
        },
        "mlx_lm": dict(mapping(implementation["mlx_lm"], "implementation.mlx_lm")),
        "python": {
            "version": python["version"],
            "executable_sha256": python["executable_sha256"],
        },
    }


def campaign_implementation_identity_sha256(sample: Sample) -> str:
    return canonical_json_sha256(
        IMPLEMENTATION_IDENTITY_DOMAIN,
        campaign_implementation_identity(sample),
    )


def _validate_tensor_records(
    raw_tensors: Any,
    *,
    where: str,
    expected_dtype: str,
) -> list[dict[str, Any]]:
    if not isinstance(raw_tensors, list) or not raw_tensors:
        raise ContractError(f"{where}: tensors must be a non-empty array")
    tensors: list[dict[str, Any]] = []
    for index, raw_tensor in enumerate(raw_tensors):
        tensor = mapping(raw_tensor, f"{where}.tensors[{index}]")
        require_exact_keys(
            tensor,
            ("name", "dtype", "shape"),
            where=f"{where}.tensors[{index}]",
        )
        name = string(tensor["name"], f"{where}.tensors[{index}].name")
        if tensor["dtype"] != expected_dtype:
            raise ContractError(f"{where}.tensors[{index}].dtype must be {expected_dtype}")
        raw_shape = tensor["shape"]
        if not isinstance(raw_shape, list):
            raise ContractError(f"{where}.tensors[{index}].shape must be an array")
        shape = [integer(dim, f"{where}.tensors[{index}].shape[]", 1) for dim in raw_shape]
        tensors.append({"name": name, "dtype": expected_dtype, "shape": shape})
    names = [tensor["name"] for tensor in tensors]
    if names != sorted(names) or len(names) != len(set(names)):
        raise ContractError(f"{where}: tensor names must be unique and sorted")
    return tensors


def _validate_inventory_observation(
    raw_observation: Any,
    *,
    where: str,
    evidence_kind: str,
    expected_dtype: str,
) -> list[dict[str, Any]]:
    observation = mapping(raw_observation, where)
    require_exact_keys(
        observation,
        ("evidence_kind", "dtype", "tensor_count", "inventory_sha256", "tensors"),
        where=where,
    )
    if observation["evidence_kind"] != evidence_kind:
        raise ContractError(f"{where}: unsupported evidence kind")
    if observation["dtype"] != expected_dtype:
        raise ContractError(f"{where}.dtype must be {expected_dtype}")
    tensors = _validate_tensor_records(
        observation["tensors"], where=where, expected_dtype=expected_dtype,
    )
    if integer(observation["tensor_count"], f"{where}.tensor_count", 1) != len(tensors):
        raise ContractError(f"{where}: tensor count differs from the exact inventory")
    if observation["inventory_sha256"] != canonical_tensor_inventory_sha256(tensors):
        raise ContractError(f"{where}: inventory digest mismatch")
    return tensors


def validate_precision_evidence_payload(
    raw_evidence: Any,
    sample: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    where: str = "precision evidence",
) -> dict[str, Any]:
    """Validate closed MLX precision evidence and bind it to one sample/runtime."""
    evidence = dict(mapping(raw_evidence, where))
    require_exact_keys(
        evidence,
        (
            "schema_version", "framework", "oracle_lock_sha256", "sample_binding",
            "runner", "native_runtime", "verified", "not_asserted", "comparison_policy",
            "observations",
        ),
        where=where,
    )
    if evidence["schema_version"] != PRECISION_EVIDENCE_SCHEMA_VERSION:
        raise ContractError(f"{where}: unsupported schema version")
    if evidence["framework"] != "mlx-lm" or sample.get("framework") != "mlx-lm":
        raise ContractError(f"{where}: precision evidence is valid only for MLX-LM samples")
    if evidence["oracle_lock_sha256"] != sample.get("oracle_lock_sha256"):
        raise ContractError(f"{where}: oracle lock digest differs from the sample")

    binding = mapping(evidence["sample_binding"], f"{where}.sample_binding")
    require_exact_keys(
        binding,
        (
            "campaign_id", "run_id", "repetition", "sequence_index", "command_sha256",
            "semantic_contract_sha256", "sample_payload_sha256",
        ),
        where=f"{where}.sample_binding",
    )
    expected_binding = {
        "campaign_id": sample.get("campaign_id"),
        "run_id": sample.get("run_id"),
        "repetition": sample.get("repetition"),
        "sequence_index": sample.get("sequence_index"),
        "command_sha256": mapping(sample.get("implementation"), "implementation").get("command_sha256"),
        "semantic_contract_sha256": mapping(sample.get("semantic_contract"), "semantic_contract").get("sha256"),
        "sample_payload_sha256": canonical_precision_sample_binding_sha256(sample),
    }
    if dict(binding) != expected_binding:
        raise ContractError(f"{where}: sample binding differs from the benchmark sample")

    runner = validate_benchmark_producer_source(
        evidence["runner"],
        expected_entrypoint=MLX_RUNNER_RELATIVE_PATH,
        where=f"{where}.runner",
    )
    sample_producer = validate_benchmark_producer_source(
        mapping(sample.get("implementation"), "implementation").get("producer_source"),
        expected_entrypoint=MLX_RUNNER_RELATIVE_PATH,
        where="implementation.producer_source",
    )
    if runner != sample_producer:
        raise ContractError(f"{where}: precision producer source differs from the benchmark sample")

    implementation = mapping(sample["implementation"], "implementation")
    mlx = mapping(implementation["mlx"], "implementation.mlx")
    mlx_lm = mapping(implementation["mlx_lm"], "implementation.mlx_lm")
    inventory = mapping(mlx["native_artifact_inventory"], "implementation.mlx.native_artifact_inventory")
    attestation = mapping(mlx["build_attestation"], "implementation.mlx.build_attestation")
    native_runtime = mapping(evidence["native_runtime"], f"{where}.native_runtime")
    require_exact_keys(
        native_runtime,
        (
            "mlx_source_revision", "mlx_lm_source_revision", "native_artifact_inventory_sha256",
            "build_attestation_sha256", "build_command_sha256", "precision_policy_sha256",
        ),
        where=f"{where}.native_runtime",
    )
    expected_native_runtime = {
        "mlx_source_revision": mlx.get("source_revision"),
        "mlx_lm_source_revision": mlx_lm.get("source_revision"),
        "native_artifact_inventory_sha256": inventory.get("sha256"),
        "build_attestation_sha256": attestation.get("sha256"),
        "build_command_sha256": attestation.get("build_command_sha256"),
        "precision_policy_sha256": attestation.get("precision_policy_sha256"),
    }
    if dict(native_runtime) != expected_native_runtime:
        raise ContractError(f"{where}: native runtime binding differs from the sample")
    for field in (
        "native_artifact_inventory_sha256", "build_attestation_sha256",
        "build_command_sha256", "precision_policy_sha256",
    ):
        sha256_identity(native_runtime[field], f"{where}.native_runtime.{field}")
    if native_runtime["precision_policy_sha256"] != lock["mlx_reference"]["native_runtime"]["precision_policy_sha256"]:
        raise ContractError(f"{where}: native precision policy differs from the lock")

    precision = mapping(lock["benchmark_contract"]["precision"], "benchmark_contract.precision")
    verified = mapping(evidence["verified"], f"{where}.verified")
    if dict(verified) != dict(mapping(precision["verified"], "benchmark_contract.precision.verified")):
        raise ContractError(f"{where}: verified precision fields differ from the lock")
    not_asserted = evidence["not_asserted"]
    if not isinstance(not_asserted, list) or not_asserted == []:
        raise ContractError(f"{where}.not_asserted must be a non-empty array")
    if not_asserted != precision["not_asserted"]:
        raise ContractError(f"{where}: not_asserted precision fields differ from the lock")
    if not_asserted != ["activation_dtype", "matmul_accumulator_dtype"]:
        raise ContractError(f"{where}: activation and matmul accumulator must remain not_asserted")
    if evidence["comparison_policy"] != precision["comparison_policy"]:
        raise ContractError(f"{where}: diagnostic-only comparison policy differs from the lock")
    if evidence["comparison_policy"] != "diagnostic-only-until-all-comparison-critical-dtypes-are-runtime-proven":
        raise ContractError(f"{where}: precision evidence must remain diagnostic only")

    observations = mapping(evidence["observations"], f"{where}.observations")
    require_exact_keys(
        observations,
        (
            "base_model_storage", "lora_parameter_storage", "gradient_storage",
            "optimizer_moment_storage", "loss",
        ),
        where=f"{where}.observations",
    )
    base_tensors = _validate_inventory_observation(
        observations["base_model_storage"],
        where=f"{where}.observations.base_model_storage",
        evidence_kind="materialized-parameter-inventory",
        expected_dtype="bfloat16",
    )
    if not base_tensors:
        raise ContractError(f"{where}: base-model storage evidence is empty")
    lora_tensors = _validate_inventory_observation(
        observations["lora_parameter_storage"],
        where=f"{where}.observations.lora_parameter_storage",
        evidence_kind="materialized-trainable-parameter-inventory",
        expected_dtype="float32",
    )

    gradient = mapping(observations["gradient_storage"], f"{where}.observations.gradient_storage")
    require_exact_keys(
        gradient,
        ("evidence_kind", "dtype", "tensor_count", "stages"),
        where=f"{where}.observations.gradient_storage",
    )
    if gradient["evidence_kind"] != "compiled-gradient-tree-inventory" or gradient["dtype"] != "float32":
        raise ContractError(f"{where}: gradient storage evidence kind/dtype is invalid")
    if integer(gradient["tensor_count"], f"{where}.observations.gradient_storage.tensor_count", 1) != len(lora_tensors):
        raise ContractError(f"{where}: gradient tensor count differs from LoRA trainables")
    raw_stages = gradient["stages"]
    if not isinstance(raw_stages, list) or len(raw_stages) != 3:
        raise ContractError(f"{where}: gradient evidence must contain raw, accumulated, and clipped stages")
    for index, expected_stage in enumerate(("raw", "accumulated", "clipped")):
        stage = mapping(raw_stages[index], f"{where}.observations.gradient_storage.stages[{index}]")
        require_exact_keys(stage, ("stage", "inventory_sha256", "tensors"), where=f"{where}.gradient stage")
        if stage["stage"] != expected_stage:
            raise ContractError(f"{where}: gradient evidence stages are incomplete or out of order")
        stage_tensors = _validate_tensor_records(
            stage["tensors"],
            where=f"{where}.observations.gradient_storage.{expected_stage}",
            expected_dtype="float32",
        )
        if stage_tensors != lora_tensors:
            raise ContractError(f"{where}: {expected_stage} gradient inventory differs from LoRA trainables")
        if stage["inventory_sha256"] != canonical_tensor_inventory_sha256(stage_tensors):
            raise ContractError(f"{where}: {expected_stage} gradient inventory digest mismatch")

    optimizer = mapping(observations["optimizer_moment_storage"], f"{where}.observations.optimizer_moment_storage")
    require_exact_keys(
        optimizer,
        (
            "evidence_kind", "dtype", "parameter_count", "moment_tensor_count",
            "inventory_sha256", "moments",
        ),
        where=f"{where}.observations.optimizer_moment_storage",
    )
    if optimizer["evidence_kind"] != "materialized-post-cold-adamw-moment-inventory" or optimizer["dtype"] != "float32":
        raise ContractError(f"{where}: optimizer-moment evidence kind/dtype is invalid")
    if integer(optimizer["parameter_count"], f"{where}.optimizer.parameter_count", 1) != len(lora_tensors):
        raise ContractError(f"{where}: optimizer parameter count differs from LoRA trainables")
    raw_moments = optimizer["moments"]
    if not isinstance(raw_moments, list):
        raise ContractError(f"{where}: optimizer moments must be an array")
    moments: list[dict[str, Any]] = []
    for index, raw_moment in enumerate(raw_moments):
        moment = mapping(raw_moment, f"{where}.optimizer.moments[{index}]")
        require_exact_keys(
            moment, ("name", "parameter_name", "role", "dtype", "shape"),
            where=f"{where}.optimizer.moments[{index}]",
        )
        role = moment["role"]
        if role not in ("m", "v"):
            raise ContractError(f"{where}: optimizer moment role must be m or v")
        parameter_name = string(moment["parameter_name"], f"{where}.optimizer.moments[{index}].parameter_name")
        name = string(moment["name"], f"{where}.optimizer.moments[{index}].name")
        if name != f"{parameter_name}.{role}" or moment["dtype"] != "float32":
            raise ContractError(f"{where}: optimizer moment name/dtype is not exact")
        raw_shape = moment["shape"]
        if not isinstance(raw_shape, list):
            raise ContractError(f"{where}: optimizer moment shape must be an array")
        shape = [integer(dim, f"{where}.optimizer.moments[{index}].shape[]", 1) for dim in raw_shape]
        moments.append({
            "name": name, "parameter_name": parameter_name, "role": role,
            "dtype": "float32", "shape": shape,
        })
    if [moment["name"] for moment in moments] != sorted(moment["name"] for moment in moments):
        raise ContractError(f"{where}: optimizer moments must be sorted")
    expected_moments = sorted(
        (
            {
                "name": f"{tensor['name']}.{role}", "parameter_name": tensor["name"],
                "role": role, "dtype": "float32", "shape": tensor["shape"],
            }
            for tensor in lora_tensors
            for role in ("m", "v")
        ),
        key=lambda moment: moment["name"],
    )
    if moments != expected_moments:
        raise ContractError(f"{where}: optimizer m/v inventory differs from exact LoRA trainables")
    if integer(optimizer["moment_tensor_count"], f"{where}.optimizer.moment_tensor_count", 2) != len(moments):
        raise ContractError(f"{where}: optimizer moment count differs from exact inventory")
    if optimizer["inventory_sha256"] != canonical_moment_inventory_sha256(moments):
        raise ContractError(f"{where}: optimizer moment inventory digest mismatch")

    loss = mapping(observations["loss"], f"{where}.observations.loss")
    require_exact_keys(
        loss, ("evidence_kind", "loss_tensor_dtype", "reduction_input_dtype"),
        where=f"{where}.observations.loss",
    )
    if loss != {
        "evidence_kind": "evaluated-training-loss-graph",
        "loss_tensor_dtype": "float32",
        "reduction_input_dtype": "float32",
    }:
        raise ContractError(f"{where}: loss tensor/reduction-input evidence is invalid")

    derived_verified = {
        "base_model_storage_dtype": observations["base_model_storage"]["dtype"],
        "lora_parameter_storage_dtype": observations["lora_parameter_storage"]["dtype"],
        "gradient_storage_dtype": gradient["dtype"],
        "optimizer_moment_storage_dtype": optimizer["dtype"],
        "loss_tensor_dtype": loss["loss_tensor_dtype"],
        "loss_reduction_input_dtype": loss["reduction_input_dtype"],
    }
    if derived_verified != dict(verified):
        raise ContractError(f"{where}: one or more verified fields lack matching runtime evidence")
    return evidence


def validate_precision_evidence_reference(
    sample_path: Path,
    sample: Mapping[str, Any],
    lock: Mapping[str, Any],
    *,
    evidence_path: Path | None = None,
) -> dict[str, Any]:
    reference = mapping(sample.get("precision_evidence"), "precision_evidence")
    require_exact_keys(
        reference, ("schema_version", "relative_path", "artifact_sha256"),
        where="precision_evidence",
    )
    if reference["schema_version"] != PRECISION_EVIDENCE_REFERENCE_SCHEMA_VERSION:
        raise ContractError("precision_evidence: unsupported reference schema")
    relative_path = string(reference["relative_path"], "precision_evidence.relative_path")
    expected_name = (
        evidence_path.expanduser().absolute().name
        if evidence_path is not None
        else sample_path.expanduser().absolute().name + ".precision.json"
    )
    if (
        relative_path != expected_name
        or "/" in relative_path
        or "\\" in relative_path
        or relative_path in (".", "..")
    ):
        raise ContractError("precision_evidence.relative_path must be one sibling file name")
    candidate = evidence_path.expanduser().absolute() if evidence_path is not None else sample_path.expanduser().absolute().with_name(relative_path)
    if candidate.name != relative_path:
        raise ContractError("precision_evidence.relative_path differs from the supplied artifact")
    if candidate.is_symlink() or not candidate.is_file():
        raise ContractError("precision_evidence artifact must be an existing regular non-symlink file")
    if candidate.parent.resolve(strict=True) != sample_path.expanduser().absolute().parent.resolve(strict=True):
        raise ContractError("precision_evidence artifact must be a sibling of the sample")
    sha256_identity(reference["artifact_sha256"], "precision_evidence.artifact_sha256")
    if reference["artifact_sha256"] != prefixed_sha256(candidate):
        raise ContractError("precision_evidence artifact digest mismatch")
    evidence = load_json(candidate)
    return validate_precision_evidence_payload(evidence, sample, lock, where=str(candidate))


def validate_zig_execution_evidence(
    raw: Any,
    *,
    grad_accum: int,
    warmup_steps: int,
    measured_steps: int,
) -> None:
    evidence = mapping(raw, "metrics.execution_evidence")
    require_exact_keys(
        evidence,
        ("schema_version", "optimizer_steps"),
        where="metrics.execution_evidence",
    )
    if evidence["schema_version"] != ZIG_EXECUTION_EVIDENCE_SCHEMA_VERSION:
        raise ContractError("metrics.execution_evidence uses an unsupported schema")
    steps = evidence["optimizer_steps"]
    expected_count = 2 + warmup_steps + measured_steps
    if not isinstance(steps, list) or len(steps) != expected_count:
        raise ContractError(f"metrics.execution_evidence must contain exactly {expected_count} optimizer steps")
    measured_start = 2 + warmup_steps
    for index, raw_step in enumerate(steps):
        step = mapping(raw_step, f"metrics.execution_evidence.optimizer_steps[{index}]")
        require_exact_keys(
            step,
            ("index", "phase", "phase_evidence", "command_plan_evidence"),
            where=f"metrics.execution_evidence.optimizer_steps[{index}]",
        )
        if step["index"] != index:
            raise ContractError("metrics.execution_evidence optimizer-step indexes are not contiguous")
        expected_phase = (
            "cold" if index == 0 else
            "first" if index == 1 else
            "warmup" if index < measured_start else
            "measured"
        )
        if step["phase"] != expected_phase:
            raise ContractError(f"metrics.execution_evidence optimizer step {index} has the wrong phase")

        phase = mapping(step["phase_evidence"], f"execution phase {index}")
        require_exact_keys(phase, ZIG_PHASE_EVIDENCE_FIELDS, where=f"execution phase {index}")
        for field in ZIG_PHASE_EVIDENCE_FIELDS:
            integer(phase[field], f"execution phase {index}.{field}")
        if (phase["compile_ns"] > 0) != (index == 0):
            raise ContractError(f"execution phase {index} compile evidence violates the cold-only contract")

        command = mapping(step["command_plan_evidence"], f"execution command plan {index}")
        require_exact_keys(
            command,
            ZIG_COMMAND_PLAN_EVIDENCE_FIELDS,
            where=f"execution command plan {index}",
        )
        for field in ZIG_COMMAND_PLAN_EVIDENCE_FIELDS:
            integer(command[field], f"execution command plan {index}.{field}")
        if command["graph_executor_partitions"] < grad_accum:
            raise ContractError(f"execution command plan {index} omitted graph partitions")
        dispatches = command["graph_executor_command_dispatches"]
        if dispatches == 0:
            raise ContractError(f"execution command plan {index} omitted graph dispatches")
        if command["graph_executor_planned_dispatches"] > dispatches:
            raise ContractError(f"execution command plan {index} planned dispatches exceed graph dispatches")
        if sum(command[field] for field in ZIG_COMMAND_ATTRIBUTION_FIELDS) > dispatches:
            raise ContractError(f"execution command plan {index} command attribution exceeds graph dispatches")
        cce_forward = command["metal_linear_cce_forward_calls"]
        cce_backward = command["metal_linear_cce_backward_calls"]
        cce_state_events = (
            command["metal_linear_cce_forward_state_hits"]
            + command["metal_linear_cce_forward_state_misses"]
        )
        if cce_state_events != cce_backward or cce_backward != cce_forward or command["metal_linear_cce_forward_state_misses"] != 0:
            raise ContractError(f"execution command plan {index} linear CCE route evidence is inconsistent")
        if (command["metal_linear_cce_peak_scratch_bytes"] != 0) != (cce_forward != 0):
            raise ContractError(f"execution command plan {index} linear CCE scratch evidence is inconsistent")
        hits = command["graph_executor_plan_cache_hits"]
        misses = command["graph_executor_plan_cache_misses"]
        if hits + misses != grad_accum:
            raise ContractError(f"execution command plan {index} has the wrong cache lookup count")
        expected_misses = 1 if index == 0 else 0
        if misses != expected_misses:
            raise ContractError(f"execution command plan {index} has an unexpected cache miss count")
        plan_was_built = misses != 0
        if ((phase["graph_executor_plan_build_ns"] != 0) != plan_was_built or
                (phase["graph_executor_buffer_plan_build_ns"] != 0) != plan_was_built):
            raise ContractError(f"execution command plan {index} differs from plan-build phase evidence")


def validate_sample(
    path: Path,
    lock: Mapping[str, Any],
    lock_path: Path,
    *,
    precision_evidence_path: Path | None = None,
) -> Sample:
    payload = dict(mapping(load_json(path.resolve()), "benchmark sample"))
    framework = payload.get("framework")
    expected_keys = [
        "schema_version", "framework", "oracle_lock_sha256", "campaign_id", "run_id",
        "repetition", "sequence_index", "implementation", "process", "hardware",
        "case", "semantic_contract", "protocol", "metrics",
    ]
    if framework == "mlx-lm":
        expected_keys.append("precision_evidence")
    require_exact_keys(
        payload,
        expected_keys,
        where="benchmark sample",
    )
    if payload["schema_version"] != BENCHMARK_SAMPLE_SCHEMA_VERSION:
        raise ContractError(f"{path}: unsupported benchmark sample schema")
    if payload["framework"] not in FRAMEWORKS:
        raise ContractError(f"{path}: unsupported framework {payload['framework']!r}")
    if payload["oracle_lock_sha256"] != lock_digest(lock_path):
        raise ContractError(f"{path}: oracle lock digest mismatch")
    string(payload["campaign_id"], "campaign_id")
    string(payload["run_id"], "run_id")
    integer(payload["repetition"], "repetition")
    integer(payload["sequence_index"], "sequence_index")

    implementation = mapping(payload["implementation"], "implementation")
    sha256_identity(implementation.get("command_sha256"), "implementation.command_sha256")
    expected_producer_entrypoint = (
        ANTFLY_RUNNER_RELATIVE_PATH
        if payload["framework"] == "antfly-zig-metal"
        else MLX_RUNNER_RELATIVE_PATH
    )
    producer_source = validate_benchmark_producer_source(
        implementation.get("producer_source"),
        expected_entrypoint=expected_producer_entrypoint,
        where="implementation.producer_source",
    )
    if payload["framework"] == "antfly-zig-metal":
        require_exact_keys(
            implementation,
            ("command_sha256", "producer_source", "antfly"),
            where="implementation",
        )
        antfly = mapping(implementation["antfly"], "implementation.antfly")
        require_exact_keys(
            antfly,
            ("version", "source_revision", "source_clean", "executable_sha256"),
            where="implementation.antfly",
        )
        string(antfly["version"], "implementation.antfly.version")
        revision = string(antfly["source_revision"], "implementation.antfly.source_revision")
        if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ContractError("implementation.antfly.source_revision must be a full commit")
        if antfly["source_clean"] is not True:
            raise ContractError("Antfly source checkout must be clean")
        if producer_source["source_revision"] != revision:
            raise ContractError("Antfly wrapper source revision differs from the benchmark executable source")
        sha256_identity(antfly["executable_sha256"], "implementation.antfly.executable_sha256")
    else:
        require_exact_keys(
            implementation,
            ("command_sha256", "producer_source", "mlx", "mlx_lm", "python"),
            where="implementation",
        )
        reference = lock["mlx_reference"]
        mlx = mapping(implementation["mlx"], "implementation.mlx")
        require_exact_keys(
            mlx,
            ("version", "source_revision", "source_clean", "native_artifact_inventory", "build_attestation"),
            where="implementation.mlx",
        )
        if mlx["version"] != reference["packages"]["mlx"]:
            raise ContractError(f"{path}: mlx version is not pinned")
        if mlx["source_revision"] != reference["source_revisions"]["mlx"]:
            raise ContractError(f"{path}: mlx revision is not pinned")
        if mlx["source_clean"] is not True:
            raise ContractError(f"{path}: mlx source checkout must be clean")
        inventory = mapping(mlx["native_artifact_inventory"], "implementation.mlx.native_artifact_inventory")
        require_exact_keys(
            inventory,
            ("schema_version", "sha256", "loaded_core_path", "artifacts"),
            where="implementation.mlx.native_artifact_inventory",
        )
        if inventory["schema_version"] != MLX_NATIVE_ARTIFACT_INVENTORY_SCHEMA_VERSION:
            raise ContractError(f"{path}: unsupported MLX native artifact inventory schema")
        loaded_core_path = Path(string(inventory["loaded_core_path"], "implementation.mlx.native_artifact_inventory.loaded_core_path"))
        if not loaded_core_path.is_absolute():
            raise ContractError("implementation.mlx native loaded core path must be absolute")
        artifacts = inventory["artifacts"]
        if not isinstance(artifacts, list):
            raise ContractError("implementation.mlx native artifacts must be an array")
        inventory_sha256 = canonical_mlx_native_artifact_inventory_sha256(artifacts)
        if inventory["sha256"] != inventory_sha256:
            raise ContractError(f"{path}: MLX native artifact inventory digest mismatch")
        python_extension = next(
            (artifact for artifact in artifacts if artifact.get("role") == "python-extension"),
            None,
        )
        if python_extension is None:
            raise ContractError(f"{path}: MLX native artifact inventory omitted the Python extension")
        if loaded_core_path.name != Path(python_extension["relative_path"]).name:
            raise ContractError(f"{path}: loaded mlx.core path differs from the bound native extension")
        attestation = mapping(mlx["build_attestation"], "implementation.mlx.build_attestation")
        require_exact_keys(
            attestation,
            (
                "schema_version", "path", "sha256", "source_revision", "source_clean",
                "native_artifact_inventory_sha256", "build_command_sha256", "precision_policy_sha256",
            ),
            where="implementation.mlx.build_attestation",
        )
        native_runtime = reference["native_runtime"]
        if attestation["schema_version"] != native_runtime["build_attestation_schema_version"]:
            raise ContractError(f"{path}: unsupported MLX native build attestation schema")
        if not Path(string(attestation["path"], "implementation.mlx.build_attestation.path")).is_absolute():
            raise ContractError("implementation.mlx build attestation path must be absolute")
        sha256_identity(attestation["sha256"], "implementation.mlx.build_attestation.sha256")
        sha256_identity(attestation["build_command_sha256"], "implementation.mlx.build_attestation.build_command_sha256")
        if attestation["source_revision"] != mlx["source_revision"] or attestation["source_clean"] is not True:
            raise ContractError(f"{path}: MLX native build source attestation differs from pinned clean source")
        if attestation["native_artifact_inventory_sha256"] != inventory_sha256:
            raise ContractError(f"{path}: MLX build attestation does not bind the native artifact inventory")
        if attestation["precision_policy_sha256"] != native_runtime["precision_policy_sha256"]:
            raise ContractError(f"{path}: MLX build attestation precision policy differs from the lock")

        mlx_lm = mapping(implementation["mlx_lm"], "implementation.mlx_lm")
        require_exact_keys(mlx_lm, ("version", "source_revision", "source_clean"), where="implementation.mlx_lm")
        if mlx_lm["version"] != reference["packages"]["mlx-lm"]:
            raise ContractError(f"{path}: mlx-lm version is not pinned")
        if mlx_lm["source_revision"] != reference["source_revisions"]["mlx-lm"]:
            raise ContractError(f"{path}: mlx-lm revision is not pinned")
        if mlx_lm["source_clean"] is not True:
            raise ContractError(f"{path}: mlx-lm source checkout must be clean")
        python = mapping(implementation["python"], "implementation.python")
        require_exact_keys(python, ("version", "executable", "executable_sha256"), where="implementation.python")
        python_version = string(python["version"], "implementation.python.version")
        if re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+][A-Za-z0-9.]+)?", python_version) is None:
            raise ContractError("implementation.python.version must be a full runtime version")
        if not python_version.startswith(reference["python"] + "."):
            raise ContractError(f"{path}: Python major/minor is not pinned")
        if not string(python["executable"], "implementation.python.executable").startswith("/"):
            raise ContractError("implementation.python.executable must be absolute")
        sha256_identity(python["executable_sha256"], "implementation.python.executable_sha256")

    process = mapping(payload["process"], "process")
    require_exact_keys(process, ("pid", "started_unix_ns"), where="process")
    integer(process["pid"], "process.pid", 1)
    integer(process["started_unix_ns"], "process.started_unix_ns", 1)

    hardware = mapping(payload["hardware"], "hardware")
    require_exact_keys(hardware, ("platform", "machine", "chip", "memory_bytes", "os_version", "os_build", "metal_device"), where="hardware")
    if hardware["platform"] != lock["mlx_reference"]["required_platform"]:
        raise ContractError(f"{path}: benchmark platform must be Darwin")
    if hardware["machine"] != lock["mlx_reference"]["required_machine"]:
        raise ContractError(f"{path}: benchmark machine must be arm64")
    for name in ("chip", "os_version", "os_build", "metal_device"):
        string(hardware[name], f"hardware.{name}")
    integer(hardware["memory_bytes"], "hardware.memory_bytes", 1)

    case = mapping(payload["case"], "case")
    require_exact_keys(
        case,
        (
            "model_key", "revision", "local_artifact_sha256", "target_preset", "rank", "alpha",
            "sequence_length", "grad_accum", "microbatch", "prepared", "initial_adapter",
            "target_inventory",
        ),
        where="case",
    )
    model_key = case["model_key"]
    if model_key not in lock["models"]:
        raise ContractError(f"{path}: unknown model key")
    if case["revision"] != lock["models"][model_key]["revision"]:
        raise ContractError(f"{path}: model revision is not locked")
    sha256_identity(case["local_artifact_sha256"], "case.local_artifact_sha256")
    if case["target_preset"] not in PRESETS:
        raise ContractError(f"{path}: target preset is not admitted")
    gate = lock["performance_gate"]
    if case["rank"] != gate["rank"] or float(case["alpha"]) != float(gate["alpha"]):
        raise ContractError(f"{path}: rank/alpha differ from the release matrix")
    if case["sequence_length"] not in gate["primary_sequence_lengths"]:
        raise ContractError(f"{path}: sequence length differs from the release matrix")
    if case["grad_accum"] not in gate["gradient_accumulation"]:
        raise ContractError(f"{path}: gradient accumulation differs from the release matrix")
    if case["microbatch"] != gate["microbatch"]:
        raise ContractError(f"{path}: microbatch differs from the release matrix")
    prepared = mapping(case["prepared"], "case.prepared")
    require_exact_keys(
        prepared,
        (
            "schema_version", "artifact_sha256", "example_index", "source_dataset_sha256",
            "source_record_sha256", "rendered_chat_sha256", "workload_sha256",
        ),
        where="case.prepared",
    )
    if prepared["schema_version"] != "gemma4_prepared/v6":
        raise ContractError(f"{path}: benchmark workload requires gemma4_prepared/v6")
    integer(prepared["example_index"], "case.prepared.example_index")
    for field in ("artifact_sha256", "workload_sha256"):
        sha256_identity(prepared[field], f"case.prepared.{field}")
    for field in ("source_dataset_sha256", "source_record_sha256", "rendered_chat_sha256"):
        sha256_hex_identity(prepared[field], f"case.prepared.{field}")

    initial_adapter = mapping(case["initial_adapter"], "case.initial_adapter")
    require_exact_keys(
        initial_adapter,
        ("schema_version", "semantic_sha256", "tensor_count", "tensor_dtype"),
        where="case.initial_adapter",
    )
    if initial_adapter["schema_version"] != "antfly_gemma4_initial_adapter_semantics/v1":
        raise ContractError(f"{path}: unsupported initial adapter semantic schema")
    sha256_identity(initial_adapter["semantic_sha256"], "case.initial_adapter.semantic_sha256")
    integer(initial_adapter["tensor_count"], "case.initial_adapter.tensor_count", 2)
    if initial_adapter["tensor_dtype"] != "float32":
        raise ContractError(f"{path}: initial adapter semantic dtype must be float32")

    target_inventory = mapping(case["target_inventory"], "case.target_inventory")
    require_exact_keys(
        target_inventory,
        ("schema_version", "sha256", "module_count", "canonical_modules"),
        where="case.target_inventory",
    )
    if target_inventory["schema_version"] != "antfly_gemma4_target_inventory/v1":
        raise ContractError(f"{path}: unsupported target inventory schema")
    raw_modules = target_inventory["canonical_modules"]
    if not isinstance(raw_modules, list):
        raise ContractError("case.target_inventory.canonical_modules must be an array")
    modules = [string(module, "case.target_inventory.canonical_modules[]") for module in raw_modules]
    module_count = integer(target_inventory["module_count"], "case.target_inventory.module_count", 1)
    if module_count != len(modules):
        raise ContractError(f"{path}: target inventory count differs from canonical module list")
    if target_inventory["sha256"] != canonical_target_inventory_sha256(modules):
        raise ContractError(f"{path}: target inventory digest does not bind canonical module list")
    validate_target_inventory(lock, model_key, case["target_preset"], modules)
    if initial_adapter["tensor_count"] != 2 * module_count:
        raise ContractError(f"{path}: initial adapter must contain exactly LoRA A and B for every target")

    semantic_contract = mapping(payload["semantic_contract"], "semantic_contract")
    expected_contract = expected_semantic_contract(lock, case)
    semantic_contract_bytes = json.dumps(
        semantic_contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    expected_contract_bytes = json.dumps(
        expected_contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    if semantic_contract_bytes != expected_contract_bytes:
        raise ContractError(f"{path}: semantic contract differs from the locked case contract")
    if payload["framework"] == "mlx-lm":
        validate_precision_evidence_reference(
            path, payload, lock, evidence_path=precision_evidence_path,
        )

    protocol = mapping(payload["protocol"], "protocol")
    require_exact_keys(
        protocol,
        (
            "fresh_process", "cold_optimizer_steps", "cold_step_mutates_optimizer_state",
            "first_steady_steps", "warmup_steps", "measured_steps", "explicit_device_sync",
            "sync_point", "timed_unit",
        ),
        where="protocol",
    )
    reference = lock["mlx_reference"]
    if protocol != {
        "fresh_process": True,
        "cold_optimizer_steps": 1,
        "cold_step_mutates_optimizer_state": True,
        "first_steady_steps": 1,
        "warmup_steps": reference["warmup_steps"],
        "measured_steps": reference["measured_steps"],
        "explicit_device_sync": True,
        "sync_point": SYNC_POINT,
        "timed_unit": TIMED_UNIT,
    }:
        raise ContractError(f"{path}: benchmark protocol differs from the lock")

    metrics = mapping(payload["metrics"], "metrics")
    expected_metric_keys = [
        "load_seconds", "cold_compile_and_step_seconds", "first_steady_step_seconds",
        "step_seconds", "input_tokens", "supervised_tokens", "memory",
    ]
    if payload["framework"] == "antfly-zig-metal":
        expected_metric_keys.append("execution_evidence")
    require_exact_keys(metrics, expected_metric_keys, where="metrics")
    finite(metrics["load_seconds"], "metrics.load_seconds")
    finite(metrics["cold_compile_and_step_seconds"], "metrics.cold_compile_and_step_seconds", positive=True)
    finite(metrics["first_steady_step_seconds"], "metrics.first_steady_step_seconds", positive=True)
    steps = metrics["step_seconds"]
    if not isinstance(steps, list) or len(steps) != reference["measured_steps"]:
        raise ContractError(f"{path}: expected exactly {reference['measured_steps']} measured steps")
    for index, duration in enumerate(steps):
        finite(duration, f"metrics.step_seconds[{index}]", positive=True)
    if payload["framework"] == "antfly-zig-metal":
        validate_zig_execution_evidence(
            metrics["execution_evidence"],
            grad_accum=case["grad_accum"],
            warmup_steps=reference["warmup_steps"],
            measured_steps=reference["measured_steps"],
        )
    integer(metrics["input_tokens"], "metrics.input_tokens", 1)
    integer(metrics["supervised_tokens"], "metrics.supervised_tokens", 1)
    memory = mapping(metrics["memory"], "metrics.memory")
    require_exact_keys(
        memory,
        (
            "process_peak_phys_footprint_bytes", "sampler_interval_ms", "sampler_sample_count",
            "framework_allocator_peak_bytes", "framework_allocator_peak_source", "system_deltas",
        ),
        where="metrics.memory",
    )
    integer(memory["process_peak_phys_footprint_bytes"], "metrics.memory.process_peak_phys_footprint_bytes", 1)
    if memory["sampler_interval_ms"] != lock["benchmark_contract"]["memory"]["sampler_interval_ms"]:
        raise ContractError(f"{path}: process-memory sampler interval differs from the lock")
    integer(memory["sampler_sample_count"], "metrics.memory.sampler_sample_count", 2)
    allocator_peak = memory["framework_allocator_peak_bytes"]
    allocator_source = memory["framework_allocator_peak_source"]
    if payload["framework"] == "mlx-lm":
        integer(allocator_peak, "metrics.memory.framework_allocator_peak_bytes", 1)
        if allocator_source != "mlx-metal-get-peak-memory":
            raise ContractError(f"{path}: MLX allocator peak source is ambiguous")
    elif allocator_peak is not None or allocator_source != "antfly-metal-allocator-unavailable":
        raise ContractError(f"{path}: Antfly allocator peak must be explicitly unavailable")
    system_deltas = mapping(memory["system_deltas"], "metrics.memory.system_deltas")
    require_exact_keys(
        system_deltas,
        (
            "swapins_bytes", "swapouts_bytes", "pageins_bytes", "pageouts_bytes",
            "pressure_available_percent_delta",
        ),
        where="metrics.memory.system_deltas",
    )
    memory_contract = lock["benchmark_contract"]["memory"]
    for field in ("swapins_bytes", "swapouts_bytes", "pageins_bytes", "pageouts_bytes"):
        delta = integer(system_deltas[field], f"metrics.memory.system_deltas.{field}")
        maximum = memory_contract[f"maximum_{field}"]
        if delta > maximum:
            raise ContractError(f"{path}: measured-phase {field} exceeds the locked zero-I/O gate")
    pressure_delta = finite_signed(
        system_deltas["pressure_available_percent_delta"],
        "metrics.memory.system_deltas.pressure_available_percent_delta",
    )
    if not -100 <= pressure_delta <= 100:
        raise ContractError("metrics.memory.system_deltas.pressure_available_percent_delta must be finite in [-100,100]")
    if pressure_delta < memory_contract["minimum_pressure_available_percent_delta"]:
        raise ContractError(f"{path}: measured-phase memory pressure drop exceeds the locked threshold")
    if metrics["supervised_tokens"] > metrics["input_tokens"]:
        raise ContractError(f"{path}: supervised token count exceeds input tokens")
    expected_input_tokens = case["sequence_length"] * case["microbatch"] * case["grad_accum"]
    if metrics["input_tokens"] != expected_input_tokens:
        raise ContractError(
            f"{path}: input_tokens must cover the complete timed optimizer step "
            f"({metrics['input_tokens']} != {expected_input_tokens})"
        )
    return Sample(path.resolve(), payload, case_key(case), producer_source)


def median(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ContractError("cannot summarize an empty metric")
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2


def percentile95(values: Iterable[float]) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ContractError("cannot summarize an empty metric")
    # Nearest-rank p95 is stable and does not invent interpolated timings.
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def summarize_framework(samples: list[Sample]) -> dict[str, Any]:
    per_process_medians = [median(sample.payload["metrics"]["step_seconds"]) for sample in samples]
    all_steps = [duration for sample in samples for duration in sample.payload["metrics"]["step_seconds"]]
    input_tokens = {sample.payload["metrics"]["input_tokens"] for sample in samples}
    supervised_tokens = {sample.payload["metrics"]["supervised_tokens"] for sample in samples}
    if len(input_tokens) != 1 or len(supervised_tokens) != 1:
        raise ContractError("token counts differ across repetitions of one case")
    latency = median(per_process_medians)
    input_count = next(iter(input_tokens))
    supervised_count = next(iter(supervised_tokens))
    return {
        "fresh_processes": len(samples),
        "input_tokens_per_optimizer_step": input_count,
        "supervised_tokens_per_optimizer_step": supervised_count,
        "median_step_seconds": latency,
        "p95_step_seconds": percentile95(all_steps),
        "input_tokens_per_second": input_count / latency,
        "supervised_tokens_per_second": supervised_count / latency,
        "median_process_peak_phys_footprint_bytes": median(
            sample.payload["metrics"]["memory"]["process_peak_phys_footprint_bytes"] for sample in samples
        ),
        "p95_process_peak_phys_footprint_bytes": percentile95(
            sample.payload["metrics"]["memory"]["process_peak_phys_footprint_bytes"] for sample in samples
        ),
        "median_load_seconds": median(sample.payload["metrics"]["load_seconds"] for sample in samples),
        "median_cold_compile_and_step_seconds": median(
            sample.payload["metrics"]["cold_compile_and_step_seconds"] for sample in samples
        ),
        "median_first_steady_step_seconds": median(
            sample.payload["metrics"]["first_steady_step_seconds"] for sample in samples
        ),
    }


def expected_full_matrix(lock: Mapping[str, Any], samples: list[Sample]) -> set[tuple[Any, ...]]:
    artifact_digests: dict[str, str] = {}
    revisions: dict[str, str] = {}
    for sample in samples:
        artifact_digests.setdefault(sample.case_key[0], sample.case_key[2])
        revisions.setdefault(sample.case_key[0], sample.case_key[1])
    if set(artifact_digests) != set(MODELS):
        return set()
    return {
        (
            model,
            revisions[model],
            artifact_digests[model],
            preset,
            lock["performance_gate"]["rank"],
            float(lock["performance_gate"]["alpha"]),
            sequence,
            grad_accum,
            lock["performance_gate"]["microbatch"],
        )
        for model in MODELS
        for preset in PRESETS
        for sequence in SEQUENCE_LENGTHS
        for grad_accum in GRADIENT_ACCUMULATION
    }


def compare_campaign(
    samples: list[Sample],
    lock: Mapping[str, Any],
    *,
    require_full_matrix: bool,
    campaign_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if not samples:
        raise ContractError("no benchmark samples were supplied")
    if require_full_matrix and campaign_manifest_path is None:
        raise ContractError("release comparison requires an orchestrator-published campaign manifest")
    campaign_manifest = None
    if campaign_manifest_path is not None:
        campaign_manifest = verify_complete_campaign_manifest(
            campaign_manifest_path,
            [(sample.path, sample.payload) for sample in samples],
        )
    if require_full_matrix:
        precision = mapping(lock["benchmark_contract"]["precision"], "benchmark_contract.precision")
        if precision["not_asserted"]:
            raise ContractError(
                "release comparison requires runtime proof for every comparison-critical precision field"
            )
    campaign_ids = {sample.payload["campaign_id"] for sample in samples}
    if len(campaign_ids) != 1:
        raise ContractError("all benchmark samples must share one campaign_id")
    hardware_values = {json.dumps(sample.payload["hardware"], sort_keys=True) for sample in samples}
    if len(hardware_values) != 1:
        raise ContractError("all benchmark samples must have byte-equivalent hardware identity")
    run_ids = [sample.payload["run_id"] for sample in samples]
    process_ids = [(sample.payload["process"]["pid"], sample.payload["process"]["started_unix_ns"]) for sample in samples]
    sequence_indices = [sample.payload["sequence_index"] for sample in samples]
    if len(run_ids) != len(set(run_ids)):
        raise ContractError("every sample must have a unique fresh-process run_id")
    if len(process_ids) != len(set(process_ids)):
        raise ContractError("process identity was reused across supposedly fresh samples")
    command_digests = [sample.payload["implementation"]["command_sha256"] for sample in samples]
    if len(command_digests) != len(set(command_digests)):
        raise ContractError("every fresh-process sample must have a unique command invocation digest")
    if len(sequence_indices) != len(set(sequence_indices)):
        raise ContractError("campaign sequence_index values must be unique")
    if sorted(sequence_indices) != list(range(len(sequence_indices))):
        raise ContractError("campaign sequence_index values must be contiguous from zero")

    implementation_identities: dict[str, str] = {}
    for framework in FRAMEWORKS:
        framework_samples = [sample for sample in samples if sample.payload["framework"] == framework]
        identities = {
            campaign_implementation_identity_sha256(sample)
            for sample in framework_samples
        }
        if len(identities) > 1:
            raise ContractError(
                f"{framework}: binary, native runtime, build receipt, Python, or producer source "
                "changed within one campaign"
            )
        if identities:
            implementation_identities[framework] = next(iter(identities))

    grouped: dict[tuple[Any, ...], dict[str, list[Sample]]] = {}
    identities_by_matrix_case: dict[tuple[Any, ...], set[tuple[Any, ...]]] = {}
    for sample in samples:
        grouped.setdefault(sample.case_key, {}).setdefault(sample.payload["framework"], []).append(sample)
        matrix_case = sample.case_key[:MATRIX_KEY_FIELDS]
        identities_by_matrix_case.setdefault(matrix_case, set()).add(sample.case_key[MATRIX_KEY_FIELDS])
    mixed_identities = [
        display_case((*matrix_case, next(iter(identities))))
        for matrix_case, identities in identities_by_matrix_case.items()
        if len(identities) != 1
    ]
    if mixed_identities:
        raise ContractError(
            "matrix cells mix semantic case identities: " + ", ".join(sorted(mixed_identities))
        )
    if require_full_matrix:
        expected = expected_full_matrix(lock, samples)
        actual_matrix = set(identities_by_matrix_case)
        if not expected or actual_matrix != expected:
            missing = sorted(display_matrix_case(key) for key in expected - actual_matrix)
            extra = sorted(display_matrix_case(key) for key in actual_matrix - expected)
            raise ContractError(f"campaign does not contain the full release matrix (missing={missing}, extra={extra})")

    minimum_processes = lock["mlx_reference"]["minimum_fresh_processes"]
    gate = lock["performance_gate"]
    failures: list[str] = []
    cell_rows: list[dict[str, Any]] = []
    throughput_ratios: list[float] = []
    regressed_cells = 0
    for key in sorted(grouped):
        by_framework = grouped[key]
        if set(by_framework) != set(FRAMEWORKS):
            failures.append(f"{display_case(key)}: missing Antfly or MLX-LM samples")
            continue
        repetitions_by_framework: dict[str, dict[int, Sample]] = {}
        semantic_contracts = {
            json.dumps(sample.payload["semantic_contract"], sort_keys=True, separators=(",", ":"))
            for framework_samples in by_framework.values()
            for sample in framework_samples
        }
        if len(semantic_contracts) != 1:
            raise ContractError(f"{display_case(key)}: paired group semantic contracts are not byte-equivalent")
        for framework, framework_samples in by_framework.items():
            repetitions = {sample.payload["repetition"]: sample for sample in framework_samples}
            if len(repetitions) != len(framework_samples):
                failures.append(f"{display_case(key)}/{framework}: duplicate repetition")
            repetitions_by_framework[framework] = repetitions
            if len(framework_samples) < minimum_processes:
                failures.append(f"{display_case(key)}/{framework}: fewer than {minimum_processes} fresh processes")
        execution_contracts = {
            execution_contract_sha256(sample)
            for framework_samples in by_framework.values()
            for sample in framework_samples
        }
        if len(execution_contracts) != 1:
            raise ContractError(f"{display_case(key)}: execution contract changed across paired repetitions")
        repetition_sets = [set(rows) for rows in repetitions_by_framework.values()]
        if repetition_sets[0] != repetition_sets[1]:
            failures.append(f"{display_case(key)}: framework repetition sets differ")
            continue
        repetitions = sorted(repetition_sets[0])
        if repetitions != list(range(len(repetitions))):
            failures.append(f"{display_case(key)}: repetitions must be contiguous from zero")
        first_frameworks: list[str] = []
        for repetition in repetitions:
            left = repetitions_by_framework["antfly-zig-metal"][repetition]
            right = repetitions_by_framework["mlx-lm"][repetition]
            if abs(left.payload["sequence_index"] - right.payload["sequence_index"]) != 1:
                failures.append(f"{display_case(key)}/repetition={repetition}: paired runs are not adjacent")
            first_frameworks.append(
                "antfly-zig-metal" if left.payload["sequence_index"] < right.payload["sequence_index"] else "mlx-lm"
            )
            left_counts = (
                left.payload["metrics"]["input_tokens"],
                left.payload["metrics"]["supervised_tokens"],
            )
            right_counts = (
                right.payload["metrics"]["input_tokens"],
                right.payload["metrics"]["supervised_tokens"],
            )
            if left_counts != right_counts:
                raise ContractError(
                    f"{display_case(key)}/repetition={repetition}: paired input/supervised token counts differ "
                    f"(antfly={left_counts}, mlx-lm={right_counts})"
                )
        if any(first_frameworks[index] == first_frameworks[index - 1] for index in range(1, len(first_frameworks))):
            failures.append(f"{display_case(key)}: framework execution order did not alternate")

        antfly = summarize_framework(by_framework["antfly-zig-metal"])
        mlx = summarize_framework(by_framework["mlx-lm"])
        throughput_ratio = antfly["supervised_tokens_per_second"] / mlx["supervised_tokens_per_second"]
        memory_ratio = (
            antfly["median_process_peak_phys_footprint_bytes"]
            / mlx["median_process_peak_phys_footprint_bytes"]
        )
        throughput_ratios.append(throughput_ratio)
        if throughput_ratio < 1.0:
            regressed_cells += 1
        cell_ok = (
            throughput_ratio >= gate["minimum_cell_throughput_ratio"]
            and memory_ratio <= gate["maximum_peak_memory_ratio"]
        )
        if throughput_ratio < gate["minimum_cell_throughput_ratio"]:
            failures.append(f"{display_case(key)}: throughput ratio {throughput_ratio:.4f} is below the cell floor")
        if memory_ratio > gate["maximum_peak_memory_ratio"]:
            failures.append(f"{display_case(key)}: peak-memory ratio {memory_ratio:.4f} exceeds the ceiling")
        cell_rows.append({
            "case": {
                "model_key": key[0],
                "revision": key[1],
                "local_artifact_sha256": key[2],
                "target_preset": key[3],
                "rank": key[4],
                "alpha": key[5],
                "sequence_length": key[6],
                "grad_accum": key[7],
                "microbatch": key[8],
                "prepared": dict(by_framework["antfly-zig-metal"][0].payload["case"]["prepared"]),
                "initial_adapter": dict(by_framework["antfly-zig-metal"][0].payload["case"]["initial_adapter"]),
                "target_inventory": dict(by_framework["antfly-zig-metal"][0].payload["case"]["target_inventory"]),
            },
            "semantic_contract": dict(by_framework["antfly-zig-metal"][0].payload["semantic_contract"]),
            "antfly": antfly,
            "mlx_lm": mlx,
            "throughput_ratio": throughput_ratio,
            "peak_memory_ratio": memory_ratio,
            "execution_order": first_frameworks,
            "execution_contract_sha256": next(iter(execution_contracts)),
            "ok": cell_ok,
        })
    geomean = math.exp(sum(math.log(value) for value in throughput_ratios) / len(throughput_ratios)) if throughput_ratios else 0.0
    if geomean < gate["minimum_geomean_throughput_ratio"]:
        failures.append(f"geometric-mean throughput ratio {geomean:.4f} is below the release floor")
    regression_fraction = regressed_cells / len(throughput_ratios) if throughput_ratios else 1.0
    if regression_fraction > gate["maximum_regression_fraction"]:
        failures.append(
            f"regressed-cell fraction {regression_fraction:.4f} exceeds "
            f"{gate['maximum_regression_fraction']:.4f}"
        )
    return {
        "schema_version": "antfly_gemma4_lora_benchmark_comparison/v1",
        "ok": not failures,
        "campaign_id": next(iter(campaign_ids)),
        "campaign_manifest": (
            {
                "path": str(campaign_manifest_path.expanduser().resolve()),
                "sha256": prefixed_sha256(campaign_manifest_path.expanduser().resolve()),
                "status": campaign_manifest["status"],
            }
            if campaign_manifest_path is not None and campaign_manifest is not None
            else None
        ),
        "hardware": json.loads(next(iter(hardware_values))),
        "implementation_identities": implementation_identities,
        "full_matrix_required": require_full_matrix,
        "cell_count": len(cell_rows),
        "geomean_throughput_ratio": geomean,
        "regressed_cell_fraction": regression_fraction,
        "cells": cell_rows,
        "failures": failures,
    }


def emit(payload: Any, output: Path | None) -> None:
    if output is None:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        write_json(output, payload)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--lock", type=Path, default=LOCK_PATH)
    root.add_argument("--output", type=Path)
    commands = root.add_subparsers(dest="command", required=True)

    fingerprint = commands.add_parser("fingerprint", help="print a host identity template")
    fingerprint.add_argument("--metal-device", required=True, help="exact Metal device label reported by the runner")

    validate = commands.add_parser("validate", help="validate one benchmark sample")
    validate.add_argument("sample", type=Path)

    compare = commands.add_parser("compare", help="validate and compare a campaign")
    compare.add_argument("samples", type=Path, nargs="+")
    compare.add_argument(
        "--campaign-manifest",
        type=Path,
        help="orchestrator-published COMPLETE.json; required for a full/release comparison",
    )
    compare.add_argument("--allow-partial", action="store_true", help="diagnostic only: do not require all 24 release cells")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        lock = load_lock(args.lock)
        if args.command == "fingerprint":
            if platform.system() != "Darwin" or platform.machine() != "arm64":
                raise ContractError("same-Mac benchmark fingerprint requires Darwin arm64")
            host = hardware_fingerprint()
            required = ("platform", "machine", "chip", "memory_bytes", "os_version", "os_build")
            missing = [name for name in required if name not in host]
            if missing:
                raise ContractError(f"could not resolve required host fields: {missing}")
            payload = {name: host[name] for name in required}
            payload["metal_device"] = args.metal_device
            emit(payload, args.output)
            return 0
        if args.command == "validate":
            sample = validate_sample(args.sample, lock, args.lock)
            payload = {"ok": True, "sample": str(sample.path), "case": display_case(sample.case_key)}
            emit(payload, args.output)
            return 0
        samples = [validate_sample(path, lock, args.lock) for path in args.samples]
        payload = compare_campaign(
            samples,
            lock,
            require_full_matrix=not args.allow_partial,
            campaign_manifest_path=args.campaign_manifest,
        )
        emit(payload, args.output)
        return 0 if payload["ok"] else 1
    except ContractError as exc:
        print(f"Gemma4 same-Mac benchmark contract error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
