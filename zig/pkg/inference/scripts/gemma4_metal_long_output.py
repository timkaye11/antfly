#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Parser and statistical contract for the Gemma4 Metal long-output gate."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable


class BenchmarkContractError(RuntimeError):
    """Raised when a raw benchmark artifact violates the comparison contract."""


_FLOAT = r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
_PERF_PREFIX = (
    r"(?m)^[^\r\n]*?\b"
    r"(?P<logger>common_perf_print|llama_perf_context_print):\s*"
)
_POLICY_ENV_NAMES = (
    "TERMITE_METAL_ENABLE_PREFILL_SG_DIRECT_LOAD",
    "TERMITE_METAL_DISABLE_PREFILL_SG_DIRECT_LOAD",
    "TERMITE_METAL_Q4_0_MMV_VARIANT",
    "TERMITE_METAL_DISABLE_Q4_0_MMV_PORTFOLIO",
    "TERMITE_METAL_TRACE_Q4_0_MMV_VARIANT",
    "TERMITE_METAL_DISABLE_SWA_SCAN_CLAMP",
    "TERMITE_METAL_DISABLE_FAST_PREPARED_FRAME",
    "TERMITE_METAL_FORCE_DIAGNOSTIC_COMMAND_BUFFERS",
    "EXPECT_GENERATED_FLASH_PREFILL_CALLS",
    "EXPECT_GENERATED_FLASH_PREFILL_HD512_CALLS",
    "EXPECT_PREFILL_DIRECT_KV",
    "EXPECT_FAST_PREPARED_FRAME",
    "EXPECT_Q4_0_MMV_VARIANT",
    "EXPECT_SWA_SCAN_CLAMP",
    "EXPECT_LLAMA_METAL_DEVICE",
    "EXPECT_LLAMA_OFFLOADED_LAYERS",
)
_TRUE_ENV_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_ENV_VALUES = frozenset(("0", "false", "no", "off"))


@dataclass(frozen=True)
class LlamaTiming:
    logger: str
    sampling_ms: float
    prompt_ms: float
    prompt_tokens: int
    eval_ms: float
    eval_runs: int
    total_ms: float
    total_tokens: int | None
    unaccounted_ms: float | None
    graphs_reused: int | None


@dataclass(frozen=True)
class LlamaMetalRuntime:
    prepared_device: str
    found_device: str
    default_device: str
    offloaded_layers: int
    total_layers: int


def _unique_match(text: str, pattern: str, label: str, path: Path | str) -> re.Match[str]:
    matches = list(re.finditer(pattern, text))
    if len(matches) != 1:
        raise BenchmarkContractError(
            f"expected exactly one {label}; found {len(matches)}: {path}"
        )
    return matches[0]


def _optional_unique_match(
    text: str,
    pattern: str,
    label: str,
    path: Path | str,
) -> re.Match[str] | None:
    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if len(matches) > 1:
        raise BenchmarkContractError(f"found multiple {label} records: {path}")
    return matches[0] if matches else None


def _last_match(text: str, pattern: str, label: str, path: Path | str) -> re.Match[str]:
    """Return the final counter snapshot; Antfly may print it before and after teardown."""

    matches = list(re.finditer(pattern, text, re.MULTILINE))
    if not matches:
        raise BenchmarkContractError(f"missing {label}: {path}")
    return matches[-1]


def _positive_finite(value: float, label: str, path: Path | str) -> float:
    if not math.isfinite(value) or value <= 0:
        raise BenchmarkContractError(f"invalid {label}={value!r}: {path}")
    return value


def parse_llama_timing(text: str, path: Path | str = "<llama log>") -> LlamaTiming:
    """Parse exactly one complete final perf block, including b10182 log prefixes."""

    sampling = _unique_match(
        text,
        _PERF_PREFIX + rf"sampling time\s*=\s*({_FLOAT}) ms",
        "llama sampling timing",
        path,
    )
    prompt = _unique_match(
        text,
        _PERF_PREFIX + rf"prompt eval time\s*=\s*({_FLOAT}) ms\s*/\s*([0-9]+) tokens",
        "llama prompt timing",
        path,
    )
    evaluation = _unique_match(
        text,
        _PERF_PREFIX + rf"eval time\s*=\s*({_FLOAT}) ms\s*/\s*([0-9]+) runs",
        "llama eval timing",
        path,
    )
    total = _unique_match(
        text,
        _PERF_PREFIX + rf"total time\s*=\s*({_FLOAT}) ms(?:\s*/\s*([0-9]+) tokens)?",
        "llama total timing",
        path,
    )
    if not (sampling.start() < prompt.start() < evaluation.start() < total.start()):
        raise BenchmarkContractError(f"llama timing records are not in final-block order: {path}")

    loggers = {match.group("logger") for match in (sampling, prompt, evaluation, total)}
    if len(loggers) != 1:
        raise BenchmarkContractError(f"llama timing block mixes logger formats: {path}")

    unaccounted = _optional_unique_match(
        text,
        _PERF_PREFIX + rf"unaccounted time\s*=\s*({_FLOAT}) ms",
        "llama unaccounted timing",
        path,
    )
    graphs = _optional_unique_match(
        text,
        _PERF_PREFIX + r"graphs reused\s*=\s*([0-9]+)",
        "llama graph reuse",
        path,
    )
    result = LlamaTiming(
        logger=next(iter(loggers)),
        sampling_ms=_positive_finite(float(sampling.group(2)), "llama sampling time", path),
        prompt_ms=_positive_finite(float(prompt.group(2)), "llama prompt time", path),
        prompt_tokens=int(prompt.group(3)),
        eval_ms=_positive_finite(float(evaluation.group(2)), "llama eval time", path),
        eval_runs=int(evaluation.group(3)),
        total_ms=_positive_finite(float(total.group(2)), "llama total time", path),
        total_tokens=int(total.group(3)) if total.group(3) else None,
        unaccounted_ms=float(unaccounted.group(2)) if unaccounted else None,
        graphs_reused=int(graphs.group(2)) if graphs else None,
    )
    if result.prompt_tokens <= 0 or result.eval_runs <= 0:
        raise BenchmarkContractError(f"llama timing block has non-positive token accounting: {path}")
    if result.unaccounted_ms is not None:
        accounted = result.sampling_ms + result.prompt_ms + result.eval_ms + result.unaccounted_ms
        tolerance_ms = max(1.0, result.total_ms * 0.001)
        if abs(accounted - result.total_ms) > tolerance_ms:
            raise BenchmarkContractError(
                f"llama timing block does not reconcile: accounted={accounted:.3f}ms "
                f"total={result.total_ms:.3f}ms tolerance={tolerance_ms:.3f}ms: {path}"
            )
    return result


def parse_llama_metal_runtime(
    text: str,
    path: Path | str = "<llama log>",
) -> LlamaMetalRuntime:
    """Prove b10182 selected Metal and offloaded the complete model."""

    prepared = _unique_match(
        text,
        r"(?m)^[^\r\n]*\bllama_prepare_model_devices:[^\r\n]*\bMTL[0-9]+\s+\(([^)\r\n]+)\)",
        "llama Metal prepared-device marker",
        path,
    )
    offload = _unique_match(
        text,
        r"(?m)^[^\r\n]*\bload_tensors:\s*offloaded\s+([0-9]+)/([0-9]+)\s+layers to GPU(?:\s|$)",
        "llama GPU layer-offload marker",
        path,
    )
    found = _unique_match(
        text,
        r"(?m)^[^\r\n]*\bggml_metal_init:\s*found device:\s*([^\r\n]+?)\s*$",
        "llama Metal found-device marker",
        path,
    )
    default = _unique_match(
        text,
        r"(?m)^[^\r\n]*\bggml_metal_init:\s*picking default device:\s*([^\r\n]+?)\s*$",
        "llama Metal default-device marker",
        path,
    )
    result = LlamaMetalRuntime(
        prepared_device=prepared.group(1).strip(),
        found_device=found.group(1).strip(),
        default_device=default.group(1).strip(),
        offloaded_layers=int(offload.group(1)),
        total_layers=int(offload.group(2)),
    )
    devices = {result.prepared_device, result.found_device, result.default_device}
    if len(devices) != 1:
        raise BenchmarkContractError(
            "llama Metal device markers disagree: "
            f"prepared={result.prepared_device!r}, found={result.found_device!r}, "
            f"default={result.default_device!r}: {path}"
        )
    if result.offloaded_layers <= 0 or result.total_layers <= 0:
        raise BenchmarkContractError(
            f"llama GPU layer offload is non-positive: "
            f"{result.offloaded_layers}/{result.total_layers}: {path}"
        )
    return result


def stats(values: Iterable[float]) -> dict[str, float]:
    collected = [float(value) for value in values]
    if not collected or any(not math.isfinite(value) or value <= 0 for value in collected):
        raise BenchmarkContractError(f"statistics require non-empty positive finite values: {collected!r}")
    mean = statistics.fmean(collected)
    return {
        "min": min(collected),
        "median": statistics.median(collected),
        "mean": mean,
        "max": max(collected),
        "cv": statistics.pstdev(collected) / mean if mean else 0.0,
    }


def ratio_stats(rows: list[dict[str, Any]], field: str) -> dict[str, float]:
    """Summarize ratios calculated within each paired sample."""

    return stats(float(row[field]) for row in rows)


def _mapping(value: Any, label: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkContractError(f"missing structured {label}: {path}")
    return value


def _exact_int(mapping: dict[str, Any], key: str, expected: int, label: str, path: Path) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkContractError(f"missing integer {label}.{key}: {path}")
    if value != expected:
        raise BenchmarkContractError(f"{label}.{key}={value}, expected {expected}: {path}")
    return value


def _token_line(text: str, name: str, path: Path) -> tuple[str, list[int]]:
    match = _unique_match(text, rf"(?m)^{re.escape(name)}:\s*(.*)$", f"Antfly {name}", path)
    raw = match.group(1).strip()
    try:
        values = [int(part) for part in raw.split()]
    except ValueError as exc:
        raise BenchmarkContractError(f"invalid integer in Antfly {name}: {path}") from exc
    if not values or any(value < 0 for value in values):
        raise BenchmarkContractError(f"invalid Antfly {name}: {path}")
    normalized = " ".join(str(value) for value in values)
    return normalized, values


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_metadata(root: Path, runs: int, requested_tokens: int) -> dict[str, Any]:
    metadata_path = root / "metadata.json"
    metadata = _mapping(json.loads(metadata_path.read_text()), "benchmark metadata", metadata_path)
    if metadata.get("schema") != "antfly.gemma4_metal_long_output.metadata.v2":
        raise BenchmarkContractError(f"unsupported benchmark metadata schema: {metadata_path}")
    if metadata.get("runs") != runs or metadata.get("output_tokens") != requested_tokens:
        raise BenchmarkContractError(f"benchmark metadata does not match requested run shape: {metadata_path}")
    for key in (
        "gguf_sha256",
        "prompt_sha256",
        "llama_cpp_binary_sha256",
        "antfly_binary_sha256",
        "git_tracked_diff_sha256",
        "git_status_sha256",
        "benchmark_harness_sha256",
        "benchmark_parser_sha256",
    ):
        value = metadata.get(key)
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise BenchmarkContractError(f"missing immutable metadata hash {key}: {metadata_path}")
    if not isinstance(metadata.get("git_dirty"), bool):
        raise BenchmarkContractError(f"missing boolean source provenance git_dirty: {metadata_path}")
    source_paths = {
        "benchmark_harness_sha256": Path(__file__).resolve().with_name(
            "benchmark_metal_gemma4_long_output.sh"
        ),
        "benchmark_parser_sha256": Path(__file__).resolve(),
    }
    for key, source_path in source_paths.items():
        current_hash = _file_sha256(source_path)
        if metadata[key] != current_hash:
            raise BenchmarkContractError(
                f"benchmark source hash mismatch for {key}: "
                f"recorded={metadata[key]}, current={current_hash}: {metadata_path}"
            )
    for key in ("llama_cpp_resolved_bin", "llama_cpp_version_output", "llama_cpp_comparator_id"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise BenchmarkContractError(f"missing comparator provenance {key}: {metadata_path}")
    expected_sha = metadata.get("llama_cpp_expected_sha256")
    if expected_sha is not None and expected_sha != metadata["llama_cpp_binary_sha256"]:
        raise BenchmarkContractError(f"llama.cpp comparator hash does not match pinned hash: {metadata_path}")
    policy_env = metadata.get("metal_policy_env")
    if not isinstance(policy_env, dict):
        raise BenchmarkContractError(f"missing Metal policy environment provenance: {metadata_path}")
    for name in _POLICY_ENV_NAMES:
        if name not in policy_env or (
            policy_env[name] is not None and not isinstance(policy_env[name], str)
        ):
            raise BenchmarkContractError(f"missing Metal policy environment value {name}: {metadata_path}")
    return metadata


def _metadata_env_bool(metadata: dict[str, Any], name: str) -> bool | None:
    metadata_path = Path(str(metadata.get("_path", "metadata.json")))
    policy_env = _mapping(metadata.get("metal_policy_env"), "Metal policy environment", metadata_path)
    raw = policy_env.get(name)
    if raw is None or not raw.strip():
        return None
    normalized = raw.strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    raise BenchmarkContractError(f"invalid boolean Metal policy environment {name}={raw!r}: {metadata_path}")


def _required_metadata_env(metadata: dict[str, Any], name: str) -> str:
    metadata_path = Path(str(metadata.get("_path", "metadata.json")))
    policy_env = _mapping(metadata.get("metal_policy_env"), "Metal policy environment", metadata_path)
    raw = policy_env.get(name)
    if not isinstance(raw, str) or not raw.strip():
        raise BenchmarkContractError(
            f"missing required benchmark expectation {name}: {metadata_path}"
        )
    return raw.strip()


def _required_metadata_env_bool(metadata: dict[str, Any], name: str) -> bool:
    value = _metadata_env_bool(metadata, name)
    if value is None:
        metadata_path = Path(str(metadata.get("_path", "metadata.json")))
        raise BenchmarkContractError(
            f"missing required boolean benchmark expectation {name}: {metadata_path}"
        )
    return value


def _required_metadata_env_int(
    metadata: dict[str, Any],
    name: str,
    *,
    positive: bool = False,
) -> int:
    raw = _required_metadata_env(metadata, name)
    if re.fullmatch(r"[0-9]+", raw) is None:
        raise BenchmarkContractError(f"invalid integer benchmark expectation {name}={raw!r}")
    value = int(raw)
    if positive and value <= 0:
        raise BenchmarkContractError(f"benchmark expectation {name} must be positive, got {value}")
    return value


def _expected_q4_mmv_variant(metadata: dict[str, Any]) -> str:
    variant = _required_metadata_env(metadata, "EXPECT_Q4_0_MMV_VARIANT").lower()
    valid_variants = {"any", "nr4-nsg2", "nr8-nsg2", "nr4-nsg4", "nr8-nsg4"}
    if variant not in valid_variants:
        raise BenchmarkContractError(
            f"invalid Q4_0 MMV benchmark expectation: {variant!r}; "
            f"expected one of {sorted(valid_variants)}"
        )
    return variant


def _top_level_env_bool(metadata: dict[str, Any], key: str) -> bool | None:
    raw = metadata.get(key)
    if raw is None or not str(raw).strip():
        return None
    normalized = str(raw).strip().lower()
    if normalized in _TRUE_ENV_VALUES:
        return True
    if normalized in _FALSE_ENV_VALUES:
        return False
    raise BenchmarkContractError(f"invalid benchmark metadata boolean {key}={raw!r}")


def _expect_split_gqa(metadata: dict[str, Any]) -> bool:
    explicit = _top_level_env_bool(metadata, "split_gqa_enable")
    disabled = _top_level_env_bool(metadata, "split_gqa_disable")
    return (explicit is None or explicit) and disabled is not True


def _expect_q4_mmv_portfolio(metadata: dict[str, Any]) -> bool:
    disabled = _metadata_env_bool(metadata, "TERMITE_METAL_DISABLE_Q4_0_MMV_PORTFOLIO")
    raw_variant = metadata["metal_policy_env"]["TERMITE_METAL_Q4_0_MMV_VARIANT"]
    variant = raw_variant.strip().lower() if raw_variant else "auto"
    valid_variants = {"auto", "legacy", "nr4-nsg2", "nr8-nsg2", "nr4-nsg4", "nr8-nsg4"}
    if variant not in valid_variants:
        raise BenchmarkContractError(f"invalid Q4_0 MMV variant in benchmark metadata: {raw_variant!r}")
    return disabled is not True and variant != "legacy"


def collect_rows(
    root: Path,
    runs: int,
    requested_tokens: int,
    metadata: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], str, str]:
    metadata = metadata or _load_metadata(root, runs, requested_tokens)
    rows: list[dict[str, Any]] = []
    reference_ids: str | None = None
    reference_prompt_ids: str | None = None
    expect_split_gqa = _expect_split_gqa(metadata)
    expect_generated_flash_prefill_calls = _required_metadata_env_int(
        metadata,
        "EXPECT_GENERATED_FLASH_PREFILL_CALLS",
    )
    expect_generated_flash_prefill_hd512_calls = _required_metadata_env_int(
        metadata,
        "EXPECT_GENERATED_FLASH_PREFILL_HD512_CALLS",
    )
    expect_prefill_direct_kv = _required_metadata_env_bool(metadata, "EXPECT_PREFILL_DIRECT_KV")
    expect_fast_prepared_frame = _required_metadata_env_bool(metadata, "EXPECT_FAST_PREPARED_FRAME")
    expect_q4_mmv_variant = _expected_q4_mmv_variant(metadata)
    expect_swa_scan_clamp = _required_metadata_env_bool(metadata, "EXPECT_SWA_SCAN_CLAMP")
    expect_llama_metal_device = _required_metadata_env(metadata, "EXPECT_LLAMA_METAL_DEVICE")
    expect_llama_offloaded_layers = _required_metadata_env_int(
        metadata,
        "EXPECT_LLAMA_OFFLOADED_LAYERS",
        positive=True,
    )
    expect_q4_mmv_portfolio = _expect_q4_mmv_portfolio(metadata)
    swa_scan_clamp_enabled = (
        _metadata_env_bool(metadata, "TERMITE_METAL_DISABLE_SWA_SCAN_CLAMP") is not True
    )
    if swa_scan_clamp_enabled != expect_swa_scan_clamp:
        raise BenchmarkContractError(
            "SWA scan clamp policy does not match the benchmark expectation: "
            f"enabled={swa_scan_clamp_enabled}, expected={expect_swa_scan_clamp}"
        )
    decode_frames = requested_tokens - 1
    expected_attention = decode_frames * 42

    for index in range(1, runs + 1):
        antfly_path = root / f"antfly-{index}.json"
        antfly_log_path = root / f"antfly-{index}.log"
        llama_path = root / f"llama-{index}.log"
        try:
            antfly = json.loads(antfly_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkContractError(f"invalid Antfly JSON: {antfly_path}: {exc}") from exc
        antfly_log = antfly_log_path.read_text(errors="replace")
        llama_log = llama_path.read_text(errors="replace")

        if antfly.get("backend") != "metal":
            raise BenchmarkContractError(f"Antfly did not report the Metal backend: {antfly_path}")
        if antfly.get("tokens") != requested_tokens or antfly.get("finish_reason") != "length":
            raise BenchmarkContractError(
                f"Antfly did not generate exactly {requested_tokens} length-limited tokens: {antfly_path}"
            )
        if "speculative" not in antfly or antfly.get("speculative") is not None:
            raise BenchmarkContractError(f"baseline benchmark unexpectedly used MTP/speculative decode: {antfly_path}")
        for draft_field in ("draft_cuda", "draft_cuda_generate"):
            if antfly.get(draft_field) is not None:
                raise BenchmarkContractError(f"baseline benchmark reported {draft_field}: {antfly_path}")
        if "generate-setup: live whole-model executor skipped" not in antfly_log:
            raise BenchmarkContractError(f"Antfly did not enter the compiled generation pipeline: {antfly_log_path}")
        if "gen_debug: executePrefill whole-model fast path" not in antfly_log:
            raise BenchmarkContractError(
                f"Antfly silently fell back from compiled whole-model prefill: {antfly_log_path}"
            )

        ids, id_values = _token_line(antfly_log, "token_ids", antfly_log_path)
        prompt_ids, prompt_id_values = _token_line(antfly_log, "prompt_token_ids", antfly_log_path)
        if len(id_values) != requested_tokens:
            raise BenchmarkContractError(
                f"Antfly token ID count={len(id_values)}, expected {requested_tokens}: {antfly_log_path}"
            )
        json_ids = antfly.get("token_ids")
        if json_ids is not None and json_ids != id_values:
            raise BenchmarkContractError(f"Antfly JSON/log token IDs differ: {antfly_path}")
        if reference_ids is None:
            reference_ids = ids
            reference_prompt_ids = prompt_ids
        elif ids != reference_ids or prompt_ids != reference_prompt_ids:
            raise BenchmarkContractError(
                f"Antfly exact prompt/generated token IDs changed between runs: {antfly_log_path}"
            )

        timing = _mapping(antfly.get("timing_ms"), "Antfly timing", antfly_path)
        antfly_total = _positive_finite(float(timing.get("generate") or 0), "Antfly total", antfly_path)
        antfly_prefill = _positive_finite(float(timing.get("prefill_inner") or 0), "Antfly prefill", antfly_path)
        antfly_decode = _positive_finite(float(timing.get("decode_inner") or 0), "Antfly decode", antfly_path)
        timing_tolerance = max(2.0, antfly_total * 0.001)
        if abs(antfly_prefill + antfly_decode - antfly_total) > timing_tolerance:
            raise BenchmarkContractError(
                f"Antfly phase timing does not reconcile: prefill+decode={antfly_prefill + antfly_decode:.3f}ms "
                f"total={antfly_total:.3f}ms: {antfly_path}"
            )

        llama_metal = parse_llama_metal_runtime(llama_log, llama_path)
        if llama_metal.default_device != expect_llama_metal_device:
            raise BenchmarkContractError(
                f"llama.cpp Metal device={llama_metal.default_device!r}, "
                f"expected {expect_llama_metal_device!r}: {llama_path}"
            )
        if (
            llama_metal.offloaded_layers != expect_llama_offloaded_layers
            or llama_metal.total_layers != expect_llama_offloaded_layers
        ):
            raise BenchmarkContractError(
                f"llama.cpp GPU layer offload={llama_metal.offloaded_layers}/"
                f"{llama_metal.total_layers}, expected all "
                f"{expect_llama_offloaded_layers}/{expect_llama_offloaded_layers}: {llama_path}"
            )
        llama = parse_llama_timing(llama_log, llama_path)
        if llama.prompt_tokens != len(prompt_id_values):
            raise BenchmarkContractError(
                f"prompt token accounting differs: Antfly={len(prompt_id_values)} "
                f"llama={llama.prompt_tokens}: {llama_path}"
            )
        if llama.eval_runs != decode_frames:
            raise BenchmarkContractError(
                f"llama eval runs={llama.eval_runs}, expected {decode_frames}: {llama_path}"
            )
        if llama.total_tokens is not None and llama.total_tokens != llama.prompt_tokens + llama.eval_runs:
            raise BenchmarkContractError(
                f"llama total token accounting={llama.total_tokens}, expected "
                f"{llama.prompt_tokens + llama.eval_runs}: {llama_path}"
            )
        if llama.graphs_reused is not None and llama.graphs_reused < max(0, llama.eval_runs - 2):
            raise BenchmarkContractError(
                f"llama graph reuse={llama.graphs_reused}, expected at least "
                f"{max(0, llama.eval_runs - 2)}: {llama_path}"
            )

        dispatch = _last_match(
            antfly_log,
            (
                r"(?m)^metal_attention_dispatch:.*\bpaged_1x=(\d+)"
                r".*\bdecode_gqa_split=(\d+)"
                r".*\bgenerated_flash_prefill=(\d+)"
                r".*\bgenerated_flash_prefill_hd512=(\d+)"
                r".*\bprefill_direct_kv=(\d+)"
                r".*\bprefill_paged_kv=(\d+)"
            ),
            "attention route counters",
            antfly_log_path,
        )
        paged_calls = int(dispatch.group(1))
        split_calls = int(dispatch.group(2))
        generated_flash_prefill_calls = int(dispatch.group(3))
        generated_flash_prefill_hd512_calls = int(dispatch.group(4))
        prefill_direct_kv_calls = int(dispatch.group(5))
        prefill_paged_kv_calls = int(dispatch.group(6))
        expected_paged = 0 if expect_split_gqa else expected_attention
        expected_split = expected_attention if expect_split_gqa else 0
        if paged_calls != expected_paged or split_calls != expected_split:
            raise BenchmarkContractError(
                f"decode attention routes paged/split={paged_calls}/{split_calls}, "
                f"expected {expected_paged}/{expected_split}: {antfly_log_path}"
            )
        generated_flash_prefill_total = (
            generated_flash_prefill_calls + generated_flash_prefill_hd512_calls
        )
        if (
            generated_flash_prefill_calls != expect_generated_flash_prefill_calls
            or generated_flash_prefill_hd512_calls
            != expect_generated_flash_prefill_hd512_calls
        ):
            raise BenchmarkContractError(
                "generated flash prefill routes="
                f"{generated_flash_prefill_calls}/{generated_flash_prefill_hd512_calls}, "
                "expected exactly "
                f"{expect_generated_flash_prefill_calls}/"
                f"{expect_generated_flash_prefill_hd512_calls}: {antfly_log_path}"
            )
        if prefill_direct_kv_calls + prefill_paged_kv_calls != generated_flash_prefill_total:
            raise BenchmarkContractError(
                "prefill K/V routes do not reconcile with generated flash attention: "
                f"direct+paged={prefill_direct_kv_calls + prefill_paged_kv_calls}, "
                f"flash={generated_flash_prefill_total}: {antfly_log_path}"
            )
        if expect_prefill_direct_kv is True and (
            prefill_direct_kv_calls != generated_flash_prefill_total or prefill_paged_kv_calls != 0
        ):
            raise BenchmarkContractError(
                f"prefill direct K/V route={prefill_direct_kv_calls}/{prefill_paged_kv_calls}, "
                f"expected {generated_flash_prefill_total}/0: {antfly_log_path}"
            )
        if expect_prefill_direct_kv is False and (
            prefill_direct_kv_calls != 0 or prefill_paged_kv_calls != generated_flash_prefill_total
        ):
            raise BenchmarkContractError(
                f"prefill direct K/V rollback={prefill_direct_kv_calls}/{prefill_paged_kv_calls}, "
                f"expected 0/{generated_flash_prefill_total}: {antfly_log_path}"
            )

        prepared_frame = _last_match(
            antfly_log,
            r"(?m)^metal_prepared_frame:\s+fast_path=(\d+)\s+fallback=(\d+)",
            "prepared frame route counters",
            antfly_log_path,
        )
        prepared_frame_fast_path_calls = int(prepared_frame.group(1))
        prepared_frame_fallback_calls = int(prepared_frame.group(2))
        if prepared_frame_fast_path_calls + prepared_frame_fallback_calls != decode_frames:
            raise BenchmarkContractError(
                "prepared frame routes do not reconcile with decode frames: "
                f"fast+fallback={prepared_frame_fast_path_calls + prepared_frame_fallback_calls}, "
                f"decode_frames={decode_frames}: {antfly_log_path}"
            )
        if expect_fast_prepared_frame is True and (
            prepared_frame_fast_path_calls != decode_frames or prepared_frame_fallback_calls != 0
        ):
            raise BenchmarkContractError(
                f"prepared frame routes={prepared_frame_fast_path_calls}/{prepared_frame_fallback_calls}, "
                f"expected {decode_frames}/0: {antfly_log_path}"
            )
        if expect_fast_prepared_frame is False and (
            prepared_frame_fast_path_calls != 0 or prepared_frame_fallback_calls != decode_frames
        ):
            raise BenchmarkContractError(
                f"prepared frame rollback={prepared_frame_fast_path_calls}/{prepared_frame_fallback_calls}, "
                f"expected 0/{decode_frames}: {antfly_log_path}"
            )

        runtime_memory = _unique_match(
            antfly_log,
            r"(?m)^metal_runtime_memory:.*\bframe_retained_mb=(\d+)",
            "Metal runtime memory counters",
            antfly_log_path,
        )
        if int(runtime_memory.group(1)) != 0:
            raise BenchmarkContractError(f"compiled decoder retained a speculative frame: {antfly_log_path}")

        q4_rows = _last_match(
            antfly_log,
            r"(?m)^metal_q4_0_dispatch:.*\blinear_reduce_rows=(\d+)/(\d+)/(\d+)/(\d+)",
            "Q4_0 route counters",
            antfly_log_path,
        )
        q6_rows = _last_match(
            antfly_log,
            r"(?m)^metal_q4_q6_k_dispatch:.*\bq6_linear_reduce_rows=(\d+)/(\d+)/(\d+)/(\d+)",
            "Q6_K route counters",
            antfly_log_path,
        )
        q4_exact = _optional_unique_match(
            antfly_log,
            r"^metal_jit_exact_dispatch:\s+q4_0=(\d+)",
            "Q4_0 exact route counters",
            antfly_log_path,
        )
        q4_pair = _last_match(
            antfly_log,
            r"^metal_q4_0_dispatch:.*\bpair_act_reduce=(\d+)",
            "Q4_0 pair route counters",
            antfly_log_path,
        )
        q4_encode = _optional_unique_match(
            antfly_log,
            r"^metal_q4_0_encode_us:\s+linear_reduce=(\d+)",
            "Q4_0 encode timing",
            antfly_log_path,
        )
        q4_policy = _last_match(
            antfly_log,
            (
                r"(?m)^metal_q4_0_policy:\s+mmv_nr4_nsg2=(\d+)"
                r"\s+mmv_nr8_nsg2=(\d+)"
                r"\s+mmv_nr4_nsg4=(\d+)"
                r"\s+mmv_nr8_nsg4=(\d+)"
                r"\s+mmv_variant_fallbacks=(\d+)"
                r"\s+mm_sg_aligned=(\d+)"
                r"\s+mm_sg_aligned_tail=(\d+)"
                r"\s+mm_sg_unrolled=(\d+)"
            ),
            "Q4_0 policy counters",
            antfly_log_path,
        )
        q4_mmv_variant_calls = [int(q4_policy.group(group)) for group in range(1, 5)]
        q4_mmv_variant_fallbacks = int(q4_policy.group(5))
        q4_mm_sg_aligned_calls = int(q4_policy.group(6))
        q4_mm_sg_aligned_tail_calls = int(q4_policy.group(7))
        q4_mm_sg_unrolled_calls = int(q4_policy.group(8))
        exact_q4_calls = int(q4_exact.group(1)) if q4_exact else 0
        fused_q4_pairs = int(q4_pair.group(1))
        logical_q4_calls = int(q4_rows.group(1)) + exact_q4_calls + 2 * fused_q4_pairs
        expected_q4_calls = decode_frames * 210
        if logical_q4_calls != expected_q4_calls:
            raise BenchmarkContractError(
                f"Q4_0 decode routes={logical_q4_calls}, expected exactly {expected_q4_calls}: {antfly_log_path}"
            )
        if sum(q4_mmv_variant_calls) != int(q4_rows.group(1)):
            raise BenchmarkContractError(
                f"Q4_0 MMV variants={sum(q4_mmv_variant_calls)}, "
                f"expected row-one routes={q4_rows.group(1)}: {antfly_log_path}"
            )
        if expect_q4_mmv_portfolio and q4_mmv_variant_fallbacks != 0:
            raise BenchmarkContractError(
                f"Q4_0 MMV portfolio fallbacks={q4_mmv_variant_fallbacks}, expected 0: {antfly_log_path}"
            )
        if expect_q4_mmv_variant != "any":
            variant_names = ("nr4-nsg2", "nr8-nsg2", "nr4-nsg4", "nr8-nsg4")
            expected_variant_calls = [0, 0, 0, 0]
            expected_variant_calls[variant_names.index(expect_q4_mmv_variant)] = expected_q4_calls
            if (
                int(q4_rows.group(1)) != expected_q4_calls
                or exact_q4_calls != 0
                or fused_q4_pairs != 0
                or q4_mmv_variant_calls != expected_variant_calls
                or q4_mmv_variant_fallbacks != 0
            ):
                observed = ", ".join(
                    f"{name}={calls}"
                    for name, calls in zip(variant_names, q4_mmv_variant_calls, strict=True)
                )
                raise BenchmarkContractError(
                    f"Q4_0 MMV one-hot route is not {expect_q4_mmv_variant}="
                    f"{expected_q4_calls}: {observed}, row_one={q4_rows.group(1)}, "
                    f"exact={exact_q4_calls}, pairs={fused_q4_pairs}, "
                    f"fallbacks={q4_mmv_variant_fallbacks}: {antfly_log_path}"
                )
        if int(q6_rows.group(1)) != requested_tokens:
            raise BenchmarkContractError(
                f"Q6_K LM-head routes={q6_rows.group(1)}, expected exactly {requested_tokens}: {antfly_log_path}"
            )

        runtime = _mapping(antfly.get("runtime"), "Antfly runtime counters", antfly_path)
        decoder = _mapping(
            antfly.get("generation_decoder_runtime"),
            "Antfly generation decoder counters",
            antfly_path,
        )
        _exact_int(runtime, "decode_greedy_calls", decode_frames, "runtime", antfly_path)
        _exact_int(decoder, "forward_attempts", decode_frames, "generation_decoder_runtime", antfly_path)
        metal = _mapping(antfly.get("metal"), "Metal counters", antfly_path)
        if metal.get("native_quant_null") is not False:
            raise BenchmarkContractError(f"Metal native quant route was unavailable: {antfly_path}")
        operators = _mapping(metal.get("runtime_command_operators"), "Metal operator counters", antfly_path)
        _exact_int(operators, "fallback", 0, "metal.runtime_command_operators", antfly_path)
        frame_fallbacks = _mapping(metal.get("frame_fallbacks"), "Metal frame fallback counters", antfly_path)
        for key in ("decode_fallback", "prefill_plan_fail", "prefill_execute_fail"):
            _exact_int(frame_fallbacks, key, 0, "metal.frame_fallbacks", antfly_path)
        quant_plan = _mapping(metal.get("quant_kernel_plan"), "Metal quant plan counters", antfly_path)
        for key in ("fast_path_misses", "unsupported_routes"):
            _exact_int(quant_plan, key, 0, "metal.quant_kernel_plan", antfly_path)
        attention = _mapping(metal.get("attention_dispatch"), "Metal attention counters", antfly_path)
        _exact_int(attention, "paged_1x", expected_paged, "metal.attention_dispatch", antfly_path)
        _exact_int(attention, "decode_gqa_split", expected_split, "metal.attention_dispatch", antfly_path)
        _exact_int(
            attention,
            "generated_flash_prefill",
            generated_flash_prefill_calls,
            "metal.attention_dispatch",
            antfly_path,
        )
        _exact_int(
            attention,
            "generated_flash_prefill_hd512",
            generated_flash_prefill_hd512_calls,
            "metal.attention_dispatch",
            antfly_path,
        )
        _exact_int(
            attention,
            "prefill_direct_kv",
            prefill_direct_kv_calls,
            "metal.attention_dispatch",
            antfly_path,
        )
        _exact_int(
            attention,
            "prefill_paged_kv",
            prefill_paged_kv_calls,
            "metal.attention_dispatch",
            antfly_path,
        )
        prepared = _mapping(metal.get("prepared_frame"), "Metal prepared frame counters", antfly_path)
        _exact_int(
            prepared,
            "fast_path",
            prepared_frame_fast_path_calls,
            "metal.prepared_frame",
            antfly_path,
        )
        _exact_int(
            prepared,
            "fallback",
            prepared_frame_fallback_calls,
            "metal.prepared_frame",
            antfly_path,
        )
        q4_policy_json = _mapping(metal.get("q4_0_policy"), "Metal Q4_0 policy counters", antfly_path)
        for key, expected in zip(
            ("mmv_nr4_nsg2", "mmv_nr8_nsg2", "mmv_nr4_nsg4", "mmv_nr8_nsg4"),
            q4_mmv_variant_calls,
            strict=True,
        ):
            _exact_int(q4_policy_json, key, expected, "metal.q4_0_policy", antfly_path)
        for key, expected in (
            ("mmv_variant_fallbacks", q4_mmv_variant_fallbacks),
            ("mm_sg_aligned", q4_mm_sg_aligned_calls),
            ("mm_sg_aligned_tail", q4_mm_sg_aligned_tail_calls),
            ("mm_sg_unrolled", q4_mm_sg_unrolled_calls),
        ):
            _exact_int(q4_policy_json, key, expected, "metal.q4_0_policy", antfly_path)

        llama_decode_ms = llama.eval_ms + llama.sampling_ms
        antfly_decode_tps = decode_frames * 1000.0 / antfly_decode
        llama_decode_tps = llama.eval_runs * 1000.0 / llama_decode_ms
        rows.append(
            {
                "sample": index,
                "prompt_tokens": len(prompt_id_values),
                "antfly_total_ms": antfly_total,
                "antfly_prefill_ms": antfly_prefill,
                "antfly_decode_ms": antfly_decode,
                "antfly_decode_tok_s": antfly_decode_tps,
                "llama_total_ms": llama.total_ms,
                "llama_prompt_ms": llama.prompt_ms,
                "llama_decode_ms": llama_decode_ms,
                "llama_decode_tok_s": llama_decode_tps,
                "llama_timing_logger": llama.logger,
                "llama_graphs_reused": llama.graphs_reused,
                "llama_metal_device": llama_metal.default_device,
                "llama_offloaded_layers": llama_metal.offloaded_layers,
                "llama_total_layers": llama_metal.total_layers,
                "total_ratio": antfly_total / llama.total_ms,
                "prefill_latency_ratio": antfly_prefill / llama.prompt_ms,
                "decode_latency_ratio": antfly_decode / llama_decode_ms,
                "decode_ratio": antfly_decode_tps / llama_decode_tps,
                "paged_1x_calls": paged_calls,
                "decode_gqa_split_calls": split_calls,
                "generated_flash_prefill_calls": generated_flash_prefill_calls,
                "generated_flash_prefill_hd512_calls": generated_flash_prefill_hd512_calls,
                "prefill_direct_kv_calls": prefill_direct_kv_calls,
                "prefill_paged_kv_calls": prefill_paged_kv_calls,
                "prepared_frame_fast_path_calls": prepared_frame_fast_path_calls,
                "prepared_frame_fallback_calls": prepared_frame_fallback_calls,
                "swa_scan_clamp_enabled": swa_scan_clamp_enabled,
                "frame_retained_mb": int(runtime_memory.group(1)),
                "q4_0_linear_reduce_rows_1": int(q4_rows.group(1)),
                "q4_0_mmv_nr4_nsg2_calls": q4_mmv_variant_calls[0],
                "q4_0_mmv_nr8_nsg2_calls": q4_mmv_variant_calls[1],
                "q4_0_mmv_nr4_nsg4_calls": q4_mmv_variant_calls[2],
                "q4_0_mmv_nr8_nsg4_calls": q4_mmv_variant_calls[3],
                "q4_0_mmv_variant_fallbacks": q4_mmv_variant_fallbacks,
                "q4_0_mm_sg_aligned_calls": q4_mm_sg_aligned_calls,
                "q4_0_mm_sg_aligned_tail_calls": q4_mm_sg_aligned_tail_calls,
                "q4_0_mm_sg_unrolled_calls": q4_mm_sg_unrolled_calls,
                "q4_0_exact_dispatches": exact_q4_calls,
                "q4_0_pair_activation_dispatches": fused_q4_pairs,
                "q6_k_linear_reduce_rows_1": int(q6_rows.group(1)),
                "q4_0_linear_reduce_encode_us": int(q4_encode.group(1)) if q4_encode else None,
                "no_mtp": True,
                "route_contract_passed": True,
            }
        )

    assert reference_ids is not None and reference_prompt_ids is not None
    return rows, reference_ids, reference_prompt_ids


def build_result(
    root: Path,
    runs: int,
    requested_tokens: int,
    max_total_ratio: float,
    min_decode_ratio: float,
    max_cv: float,
    expected_token_ids_sha256: str = "",
) -> dict[str, Any]:
    metadata = _load_metadata(root, runs, requested_tokens)
    rows, token_ids, prompt_token_ids = collect_rows(
        root,
        runs,
        requested_tokens,
        metadata,
    )
    token_ids_sha256 = _digest(token_ids)
    if expected_token_ids_sha256 and token_ids_sha256 != expected_token_ids_sha256.lower():
        raise BenchmarkContractError(
            f"Antfly exact token digest changed: expected {expected_token_ids_sha256.lower()}, "
            f"got {token_ids_sha256}"
        )

    metric_fields = {
        "antfly_total_ms": "antfly_total_ms",
        "llama_total_ms": "llama_total_ms",
        "antfly_prefill_ms": "antfly_prefill_ms",
        "llama_prompt_ms": "llama_prompt_ms",
        "antfly_decode_ms": "antfly_decode_ms",
        "llama_decode_ms": "llama_decode_ms",
        "antfly_decode_tok_s": "antfly_decode_tok_s",
        "llama_decode_tok_s": "llama_decode_tok_s",
    }
    metric_stats = {
        output: stats(row[field] for row in rows)
        for output, field in metric_fields.items()
    }
    paired = {
        "total_latency_ratio": ratio_stats(rows, "total_ratio"),
        "prefill_latency_ratio": ratio_stats(rows, "prefill_latency_ratio"),
        "decode_latency_ratio": ratio_stats(rows, "decode_latency_ratio"),
        "decode_throughput_ratio": ratio_stats(rows, "decode_ratio"),
    }
    cv_metrics = (
        "antfly_total_ms",
        "llama_total_ms",
        "antfly_prefill_ms",
        "llama_prompt_ms",
        "antfly_decode_ms",
        "llama_decode_ms",
    )
    cv_violations = {
        name: metric_stats[name]["cv"]
        for name in cv_metrics
        if metric_stats[name]["cv"] > max_cv
    }
    return {
        "schema": "antfly.gemma4_metal_long_output.v2",
        "metadata": metadata,
        "runs": runs,
        "output_tokens": requested_tokens,
        "prompt_tokens": rows[0]["prompt_tokens"],
        **metric_stats,
        "paired_ratios": paired,
        # Backward-compatible top-level gates now use medians of paired ratios.
        "total_ratio": paired["total_latency_ratio"]["median"],
        "max_total_ratio": max_total_ratio,
        "decode_ratio": paired["decode_throughput_ratio"]["median"],
        "min_decode_ratio": min_decode_ratio,
        "max_cv": max_cv,
        "cv_gate": {
            "metrics": list(cv_metrics),
            "violations": cv_violations,
            "passed": not cv_violations,
        },
        "token_ids": token_ids,
        "token_ids_sha256": token_ids_sha256,
        "expected_token_ids_sha256": expected_token_ids_sha256.lower() or None,
        "prompt_token_ids_sha256": _digest(prompt_token_ids),
        "no_mtp": True,
        "policy_route_expectations": {
            "generated_flash_prefill_calls": _required_metadata_env_int(
                metadata,
                "EXPECT_GENERATED_FLASH_PREFILL_CALLS",
            ),
            "generated_flash_prefill_hd512_calls": _required_metadata_env_int(
                metadata,
                "EXPECT_GENERATED_FLASH_PREFILL_HD512_CALLS",
            ),
            "prefill_direct_kv": _required_metadata_env_bool(metadata, "EXPECT_PREFILL_DIRECT_KV"),
            "fast_prepared_frame": _required_metadata_env_bool(
                metadata,
                "EXPECT_FAST_PREPARED_FRAME",
            ),
            "q4_0_mmv_variant": _expected_q4_mmv_variant(metadata),
            "q4_0_mmv_portfolio": _expect_q4_mmv_portfolio(metadata),
            "swa_scan_clamp": _required_metadata_env_bool(metadata, "EXPECT_SWA_SCAN_CLAMP"),
        },
        "llama_backend_expectations": {
            "metal_device": _required_metadata_env(metadata, "EXPECT_LLAMA_METAL_DEVICE"),
            "offloaded_layers": _required_metadata_env_int(
                metadata,
                "EXPECT_LLAMA_OFFLOADED_LAYERS",
                positive=True,
            ),
        },
        "exact_token_contract_passed": True,
        "route_contract_passed": True,
        "rows": rows,
    }


def gate_errors(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    violations = result["cv_gate"]["violations"]
    if violations:
        rendered = ", ".join(f"{name}={value:.3f}" for name, value in sorted(violations.items()))
        errors.append(f"benchmark phase CV exceeded {result['max_cv']:.3f}: {rendered}")
    if result["total_ratio"] > result["max_total_ratio"]:
        errors.append(
            f"paired Antfly/llama total ratio {result['total_ratio']:.3f} "
            f"exceeds {result['max_total_ratio']:.3f}"
        )
    if result["decode_ratio"] < result["min_decode_ratio"]:
        errors.append(
            f"paired Antfly/llama decode ratio {result['decode_ratio']:.3f} "
            f"below {result['min_decode_ratio']:.3f}"
        )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--max-total-ratio", type=float, required=True)
    parser.add_argument("--min-decode-ratio", type=float, required=True)
    parser.add_argument("--max-cv", type=float, required=True)
    parser.add_argument("--expected-token-ids-sha256", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = build_result(
            args.out_dir,
            args.runs,
            args.output_tokens,
            args.max_total_ratio,
            args.min_decode_ratio,
            args.max_cv,
            args.expected_token_ids_sha256,
        )
    except (BenchmarkContractError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Gemma4 long-output: prompt={result['prompt_tokens']} output={result['output_tokens']} "
        f"Antfly={result['antfly_total_ms']['median']:.1f}ms "
        f"llama={result['llama_total_ms']['median']:.1f}ms "
        f"paired_total_ratio={result['total_ratio']:.3f} "
        f"paired_decode_ratio={result['decode_ratio']:.3f}"
    )
    errors = gate_errors(result)
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
