#!/usr/bin/env python3
"""Summarize Gemma4 E4B QAT/Q4_K CUDA gate artifacts for CI dashboards."""

import argparse
import csv
import json
import math
import pathlib
from typing import Any, Dict, Iterable, List, Optional


IMPORTANT_CUDA_COUNTERS = (
    "download_syncs",
    "launch_scalar",
    "launch_embedding",
    "qkv_fused_q4_0",
    "linear_pair_fused_q4_0",
    "gated_down_fused_q4_0",
    "gated_down_fused_q4_0_tile8",
    "lm_head_argmax_fused_q4_0",
    "lm_head_argmax_fused_q6",
    "lm_head_argmax_fallbacks",
    "mtp_preproject_fused_hits",
    "mtp_preproject_fused_f32_weight_hits",
    "mtp_preproject_fused_bf16_weight_hits",
    "mtp_preproject_fused_f16_weight_hits",
    "mtp_preproject_fused_fallbacks",
    "mtp_masked_select_fused_hits",
    "mtp_masked_select_fused_f32_weight_hits",
    "mtp_masked_select_fused_bf16_weight_hits",
    "mtp_masked_select_fused_f16_weight_hits",
    "mtp_masked_select_fused_fallbacks",
    "mtp_masked_select_hidden_fused_hits",
    "mtp_masked_select_hidden_fused_bf16_hits",
    "mtp_masked_select_hidden_multiblock_hits",
    "mtp_masked_select_hidden_fused_fallbacks",
    "qkv_fallback_unsupported",
    "qkv_kernel_unavailable",
    "linear_pair_fallbacks",
    "gated_down_fallbacks",
    "launch_attention_gqa_decode_fast",
    "launch_attention_gqa_decode_fast_fallbacks",
    "device_kv_compressed_v_reads",
    "device_kv_compressed_v_writes",
    "device_kv_paged_block_table_uploads",
    "device_kv_paged_identity_attention_reads",
    "device_kv_fail_write",
    "graph_capture_persistent_replays",
    "graph_capture_capacity_skips",
    "launches_per_token",
    "add_mul_scalar_fused",
    "linear_activation_slice_fused_q4_0",
    "gated_down_fused_q4_0_precompute",
    "rms_norm_add_weighted_embedding_fused_q6_k",
)

PROVIDER_METRICS = {
    "compressed_kv_decode_tok_s",
    "target_decode_tok_s",
    "long_decode_tok_s",
    "mtp_best_decode_tok_s",
    "mtp_best_active_decode_tok_s",
    "mtp_target_decode_tok_s",
    "resident_e2e_tok_s",
    "soak_aggregate_tok_s",
    "backpressure_accepted_e2e_tok_s",
}


def read_json(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_tsv(path: pathlib.Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def to_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def avg(values: Iterable[float]) -> Optional[float]:
    items = list(values)
    if not items:
        return None
    return sum(items) / len(items)


def percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(math.ceil((pct / 100.0) * len(ordered))) - 1))
    return ordered[index]


def json_run_summary(name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    cuda = data.get("cuda") or {}
    cuda_generate = data.get("cuda_generate") or {}
    runtime = data.get("generation_decoder_runtime") or {}
    counters = {field: cuda.get(field) for field in IMPORTANT_CUDA_COUNTERS if field in cuda}
    return {
        "name": name,
        "tokens": to_int(data.get("tokens")),
        "decode_tok_per_s": to_float(data.get("decode_tok_per_s")),
        "cuda": counters,
        "cuda_generate": {
            "to_float32_calls": to_int(cuda_generate.get("to_float32_calls")),
            "to_float32_bytes": to_int(cuda_generate.get("to_float32_bytes")),
        },
        "generation_decoder_runtime": {
            "device_token_handoff_attempts": to_int(runtime.get("device_token_handoff_attempts")),
            "device_token_handoff_hits": to_int(runtime.get("device_token_handoff_hits")),
            "device_token_handoff_fallbacks": to_int(runtime.get("device_token_handoff_fallbacks")),
            "device_token_handoff_seeds": to_int(runtime.get("device_token_handoff_seeds")),
        },
    }


def collect_json_runs(out_dir: pathlib.Path, names: Iterable[str]) -> List[Dict[str, Any]]:
    runs = []
    for name in names:
        data = read_json(out_dir / f"{name}.json")
        if data is not None:
            runs.append(json_run_summary(name, data))
    return runs


def summarize_rates(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    rates = [rate for rate in (to_float(run.get("decode_tok_per_s")) for run in runs) if rate is not None]
    tokens = [tokens for tokens in (to_int(run.get("tokens")) for run in runs) if tokens is not None]
    counter_values: Dict[str, List[float]] = {}
    for run in runs:
        cuda = run.get("cuda") or {}
        for name, value in cuda.items():
            numeric = to_float(value)
            if numeric is None:
                continue
            counter_values.setdefault(name, []).append(numeric)
    return {
        "runs": runs,
        "count": len(runs),
        "min_decode_tok_per_s": min(rates) if rates else None,
        "avg_decode_tok_per_s": avg(rates),
        "max_decode_tok_per_s": max(rates) if rates else None,
        "min_tokens": min(tokens) if tokens else None,
        "cuda_counter_sums": {name: sum(values) for name, values in counter_values.items()},
        "cuda_counter_mins": {name: min(values) for name, values in counter_values.items()},
        "cuda_counter_maxs": {name: max(values) for name, values in counter_values.items()},
    }


def summarize_tsv(path: pathlib.Path) -> Dict[str, Any]:
    rows = read_tsv(path)
    rates = []
    latencies = []
    completion_tokens = []
    for row in rows:
        rate = to_float(row.get("e2e_tok_s"))
        latency = to_float(row.get("e2e_ms"))
        tokens = to_int(row.get("completion_tokens"))
        if rate is not None:
            rates.append(rate)
        if latency is not None:
            latencies.append(latency)
        if tokens is not None:
            completion_tokens.append(tokens)
    return {
        "path": str(path),
        "row_count": len(rows),
        "avg_e2e_tok_s": avg(rates),
        "min_e2e_tok_s": min(rates) if rates else None,
        "max_e2e_tok_s": max(rates) if rates else None,
        "p50_e2e_ms": percentile(latencies, 50),
        "p95_e2e_ms": percentile(latencies, 95),
        "p99_e2e_ms": percentile(latencies, 99),
        "max_e2e_ms": max(latencies) if latencies else None,
        "total_completion_tokens": sum(completion_tokens) if completion_tokens else None,
        "rows": rows,
    }


def metric_values(path: pathlib.Path) -> Dict[str, float]:
    values = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or " " not in line:
            continue
        name, value = line.rsplit(" ", 1)
        try:
            values[name] = float(value)
        except ValueError:
            continue
    return values


def graph_log_summary(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "present": False}
    text = path.read_text(encoding="utf-8", errors="replace")
    unsafe_markers = [
        marker
        for marker in (
            "unsafe_d2h_copy",
            "unsafe_h2d_copy",
            "unsafe_temp_alloc",
            "CudaGraphCaptureUnsafe",
            "persistent_replay_kv_capacity_exceeded",
            "cuda_graph_capture_probe: discarded",
            "CUDA_ERROR_ILLEGAL_ADDRESS",
        )
        if marker in text
    ]
    return {
        "path": str(path),
        "present": True,
        "replays": text.count("persistent_replayed") + text.count("instantiated_cached_replayed"),
        "unsafe_markers": unsafe_markers,
    }


def summarize_backpressure(out_dir: pathlib.Path, server_log: pathlib.Path) -> Dict[str, Any]:
    path = out_dir / "e4b_qat_resident_backpressure.tsv"
    rows = read_tsv(path)
    accepted_rates = []
    reject_ms = []
    accepted = 0
    rejected = 0
    for row in rows:
        status = to_int(row.get("status"))
        if status == 200:
            accepted += 1
            rate = to_float(row.get("e2e_tok_s"))
            if rate is not None:
                accepted_rates.append(rate)
        elif status == 503:
            rejected += 1
            latency = to_float(row.get("e2e_ms"))
            if latency is not None:
                reject_ms.append(latency)
    return {
        "path": str(path),
        "row_count": len(rows),
        "accepted": accepted,
        "rejected": rejected,
        "avg_accepted_e2e_tok_s": avg(accepted_rates),
        "max_reject_ms": max(reject_ms) if reject_ms else None,
        "metrics": metric_values(out_dir / "e4b_qat_resident_backpressure_metrics.txt"),
        "graph_log": graph_log_summary(server_log),
        "rows": rows,
    }


def summarize_mtp_matrix(summary_path: pathlib.Path) -> Dict[str, Any]:
    rows = read_tsv(summary_path)
    target_rates_by_case: Dict[str, float] = {}
    target_rates: List[float] = []
    candidates: List[Dict[str, Any]] = []

    for row in rows:
        if row.get("status") != "ok":
            continue
        rate = to_float(row.get("decode_tok_s"))
        if rate is None:
            continue
        if row.get("assistant") == "target":
            target_rates_by_case[row.get("case") or ""] = rate
            target_rates.append(rate)

    for row in rows:
        if row.get("status") != "ok" or row.get("assistant") == "target":
            continue
        rate = to_float(row.get("decode_tok_s"))
        if rate is None:
            continue
        case = row.get("case") or ""
        target_rate = target_rates_by_case.get(case)
        decision = row.get("policy_decision") or ""
        candidates.append(
            {
                "assistant": row.get("assistant"),
                "spec_k": to_int(row.get("spec_k")),
                "case": case,
                "decode_tok_s": rate,
                "target_decode_tok_s": target_rate,
                "target_ratio": (
                    rate / target_rate
                    if target_rate not in (None, 0)
                    else None
                ),
                "tokens": to_int(row.get("tokens")),
                "finish_reason": row.get("finish_reason"),
                "policy": row.get("policy"),
                "calibration": row.get("calibration"),
                "policy_decision": decision,
                "graph_replay": row.get("graph_replay"),
                "drafted": to_int(row.get("drafted")),
                "matched": to_int(row.get("matched")),
                "accepted": to_int(row.get("accepted")),
                "accept_permille": to_int(row.get("accept_permille")),
                "materializations": to_int(row.get("materializations")),
                "draft_launches": to_int(row.get("draft_launches")),
                "draft_cross_copies": to_int(row.get("draft_cross_copies")),
                "draft_mtp_preproject_fused_hits": to_int(row.get("draft_mtp_preproject_fused_hits")),
                "draft_mtp_preproject_fused_fallbacks": to_int(row.get("draft_mtp_preproject_fused_fallbacks")),
                "draft_mtp_masked_select_fused_hits": to_int(row.get("draft_mtp_masked_select_fused_hits")),
                "draft_mtp_masked_select_fused_fallbacks": to_int(row.get("draft_mtp_masked_select_fused_fallbacks")),
                "draft_mtp_masked_select_hidden_fused_hits": to_int(row.get("draft_mtp_masked_select_hidden_fused_hits")),
                "draft_mtp_masked_select_hidden_multiblock_hits": to_int(row.get("draft_mtp_masked_select_hidden_multiblock_hits")),
                "draft_mtp_masked_select_hidden_fused_fallbacks": to_int(row.get("draft_mtp_masked_select_hidden_fused_fallbacks")),
                "draft_mtp_masked_argmax_hits": to_int(row.get("draft_mtp_masked_argmax_hits")),
                "draft_mtp_masked_argmax_fallbacks": to_int(row.get("draft_mtp_masked_argmax_fallbacks")),
                "draft_embedding_cross_copies": to_int(row.get("draft_embedding_cross_copies")),
                "profile_draft_ms": to_float(row.get("profile_draft_ms")),
                "profile_draft_embedding_ms": to_float(row.get("profile_draft_embedding_ms")),
                "profile_draft_concat_ms": to_float(row.get("profile_draft_concat_ms")),
                "profile_draft_preprojection_ms": to_float(row.get("profile_draft_preprojection_ms")),
                "profile_draft_assistant_ms": to_float(row.get("profile_draft_assistant_ms")),
                "profile_draft_postprojection_ms": to_float(row.get("profile_draft_postprojection_ms")),
                "profile_draft_argmax_ms": to_float(row.get("profile_draft_argmax_ms")),
                "profile_draft_lm_head_ms": to_float(row.get("profile_draft_lm_head_ms")),
                "profile_draft_selection_ms": to_float(row.get("profile_draft_selection_ms")),
                "draft_graph_begins": to_int(row.get("draft_graph_begins")),
                "draft_graph_persistent_replays": to_int(row.get("draft_graph_persistent_replays")),
                "active": decision in {"active", "forced"},
                "dedicated_runtime_fallbacks": to_int(row.get("dedicated_runtime_fallbacks")),
                "device_verify_commit_fallbacks": to_int(row.get("device_verify_commit_fallbacks")),
            }
        )

    best = max(candidates, key=lambda item: item["decode_tok_s"], default=None)
    active_candidates = [item for item in candidates if item.get("active")]
    best_active = max(active_candidates, key=lambda item: item["decode_tok_s"], default=None)
    return {
        "path": str(summary_path),
        "row_count": len(rows),
        "target_avg_decode_tok_s": avg(target_rates),
        "candidate_count": len(candidates),
        "active_candidate_count": len(active_candidates),
        "best": best,
        "best_active": best_active,
        "candidates": candidates,
    }


def is_mtp_diagnostic_matrix(name: str) -> bool:
    return (
        name.startswith("mtp_acceptance_matrix_")
        or name.startswith("mtp_donor_matrix_")
        or name.startswith("mtp_hidden_ab_")
        or name.startswith("mtp_replay_stability_")
        or name.startswith("mtp_target_equivalence")
    )


def collect_mtp_matrices(out_dir: pathlib.Path) -> Dict[str, Any]:
    matrices: Dict[str, Any] = {}
    for summary_path in sorted(out_dir.glob("*/summary.tsv")):
        name = summary_path.parent.name
        if not name.startswith("mtp_") or is_mtp_diagnostic_matrix(name):
            continue
        matrices[name] = summarize_mtp_matrix(summary_path)
    return matrices


def collect_mtp_replay_stability(out_dir: pathlib.Path) -> Dict[str, Any]:
    path = out_dir / "mtp_replay_stability.json"
    result: Dict[str, Any] = {
        "path": str(path),
        "present": path.exists(),
    }
    if not path.exists():
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        result.update({"ok": False, "load_error": repr(exc)})
        return result
    result.update(payload)
    result["present"] = True
    return result


def collect_mtp_target_equivalence(out_dir: pathlib.Path) -> Dict[str, Any]:
    path = out_dir / "mtp_target_equivalence.json"
    result: Dict[str, Any] = {
        "path": str(path),
        "present": path.exists(),
    }
    if not path.exists():
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        result.update({"ok": False, "load_error": repr(exc)})
        return result
    result.update(payload)
    result["present"] = True
    return result


def collect_mtp_replay_512(out_dir: pathlib.Path) -> Dict[str, Any]:
    path = out_dir / "mtp_replay_512" / "summary.tsv"
    if not path.exists():
        return {
            "path": str(path),
            "present": False,
        }
    result = summarize_mtp_matrix(path)
    result["present"] = True
    return result


def active_or_all_candidates(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates = list(summary.get("candidates") or [])
    active = [item for item in candidates if item.get("active")]
    return active or candidates


def candidate_avg_rate(summary: Dict[str, Any]) -> Optional[float]:
    return avg(
        rate
        for rate in (to_float(item.get("decode_tok_s")) for item in active_or_all_candidates(summary))
        if rate is not None
    )


def default_hidden_selector_disabled(summary: Dict[str, Any]) -> bool:
    rows = active_or_all_candidates(summary)
    return bool(rows) and all(
        to_int(item.get("draft_mtp_masked_select_hidden_fused_hits")) == 0
        and to_int(item.get("draft_mtp_masked_select_hidden_multiblock_hits")) == 0
        for item in rows
    )


def hidden_selector_counters_ok(summary: Dict[str, Any]) -> bool:
    rows = active_or_all_candidates(summary)
    return bool(rows) and all(
        (to_int(item.get("draft_mtp_masked_select_hidden_multiblock_hits")) or 0) > 0
        and to_int(item.get("draft_mtp_masked_select_hidden_fused_fallbacks")) == 0
        and to_int(item.get("draft_mtp_preproject_fused_fallbacks")) == 0
        and to_int(item.get("draft_mtp_masked_argmax_hits")) == 0
        and to_int(item.get("draft_mtp_masked_argmax_fallbacks")) == 0
        for item in rows
    )


def collect_mtp_hidden_ab(out_dir: pathlib.Path) -> Dict[str, Any]:
    pairs: List[Dict[str, Any]] = []
    for default_path in sorted(out_dir.glob("mtp_hidden_ab_default_r*/summary.tsv")):
        repeat = default_path.parent.name.removeprefix("mtp_hidden_ab_default_")
        hidden_path = out_dir / f"mtp_hidden_ab_hidden_{repeat}" / "summary.tsv"
        default_summary = summarize_mtp_matrix(default_path)
        hidden_summary = summarize_mtp_matrix(hidden_path) if hidden_path.exists() else {}
        default_avg = candidate_avg_rate(default_summary)
        hidden_avg = candidate_avg_rate(hidden_summary) if hidden_summary else None
        pairs.append(
            {
                "repeat": repeat,
                "default": default_summary,
                "hidden": hidden_summary,
                "default_avg_decode_tok_s": default_avg,
                "hidden_avg_decode_tok_s": hidden_avg,
                "hidden_over_default_ratio": (
                    hidden_avg / default_avg
                    if hidden_avg is not None and default_avg not in (None, 0)
                    else None
                ),
                "default_hidden_selector_disabled": default_hidden_selector_disabled(default_summary),
                "hidden_selector_counters_ok": hidden_selector_counters_ok(hidden_summary) if hidden_summary else False,
            }
        )
    return {
        "present": bool(pairs),
        "pair_count": len(pairs),
        "pairs": pairs,
        "default_hidden_selector_disabled": bool(pairs) and all(pair["default_hidden_selector_disabled"] for pair in pairs),
        "hidden_selector_counters_ok": bool(pairs) and all(pair["hidden_selector_counters_ok"] for pair in pairs),
        "min_ratio": min(
            (
                ratio
                for ratio in (to_float(pair.get("hidden_over_default_ratio")) for pair in pairs)
                if ratio is not None
            ),
            default=None,
        ),
    }


def collect_mtp_acceptance_matrix(out_dir: pathlib.Path) -> Dict[str, Any]:
    matrices: Dict[str, Any] = {}
    all_candidates: List[Dict[str, Any]] = []
    for summary_path in sorted(out_dir.glob("mtp_acceptance_matrix_*/summary.tsv")):
        name = summary_path.parent.name
        summary = summarize_mtp_matrix(summary_path)
        matrices[name] = summary
        for candidate in summary.get("candidates") or []:
            item = dict(candidate)
            item["matrix"] = name
            all_candidates.append(item)
    best_acceptance = max(
        all_candidates,
        key=lambda item: (
            to_int(item.get("accept_permille")) if to_int(item.get("accept_permille")) is not None else -1,
            to_float(item.get("decode_tok_s")) if to_float(item.get("decode_tok_s")) is not None else -1.0,
        ),
        default=None,
    )
    return {
        "present": bool(matrices),
        "combo_count": len(matrices),
        "candidate_count": len(all_candidates),
        "best_acceptance": best_acceptance,
        "matrices": matrices,
    }


def collect_mtp_donor_matrix(out_dir: pathlib.Path) -> Dict[str, Any]:
    matrices: Dict[str, Any] = {}
    all_candidates: List[Dict[str, Any]] = []
    for summary_path in sorted(out_dir.glob("mtp_donor_matrix_*/summary.tsv")):
        name = summary_path.parent.name
        summary = summarize_mtp_matrix(summary_path)
        matrices[name] = summary
        for candidate in summary.get("candidates") or []:
            item = dict(candidate)
            item["matrix"] = name
            all_candidates.append(item)
    best_acceptance = max(
        all_candidates,
        key=lambda item: (
            to_int(item.get("accept_permille")) if to_int(item.get("accept_permille")) is not None else -1,
            to_float(item.get("decode_tok_s")) if to_float(item.get("decode_tok_s")) is not None else -1.0,
        ),
        default=None,
    )
    best_active = max(
        (item for item in all_candidates if item.get("active")),
        key=lambda item: to_float(item.get("decode_tok_s")) if to_float(item.get("decode_tok_s")) is not None else -1.0,
        default=None,
    )
    return {
        "present": bool(matrices),
        "combo_count": len(matrices),
        "candidate_count": len(all_candidates),
        "best_acceptance": best_acceptance,
        "best_active": best_active,
        "matrices": matrices,
    }


def mtp_metric_values(matrices: Dict[str, Any]) -> Dict[str, Optional[float]]:
    best_decode: Optional[float] = None
    best_active_decode: Optional[float] = None
    target_rates: List[float] = []
    for matrix in matrices.values():
        best = matrix.get("best") or {}
        best_active = matrix.get("best_active") or {}
        best_rate = to_float(best.get("decode_tok_s"))
        best_active_rate = to_float(best_active.get("decode_tok_s"))
        target_rate = to_float(matrix.get("target_avg_decode_tok_s"))
        if best_rate is not None:
            best_decode = best_rate if best_decode is None else max(best_decode, best_rate)
        if best_active_rate is not None:
            best_active_decode = best_active_rate if best_active_decode is None else max(best_active_decode, best_active_rate)
        if target_rate is not None:
            target_rates.append(target_rate)
    return {
        "mtp_best_decode_tok_s": best_decode,
        "mtp_best_active_decode_tok_s": best_active_decode,
        "mtp_target_decode_tok_s": avg(target_rates),
    }


def collect_steps(out_dir: pathlib.Path) -> Dict[str, Any]:
    rows = read_tsv(out_dir / "steps.tsv")
    failures = [row for row in rows if row.get("status") == "fail"]
    return {
        "path": str(out_dir / "steps.tsv"),
        "present": bool(rows),
        "rows": rows,
        "failures": failures,
    }


def provider_metric_values(report: Dict[str, Any]) -> Dict[str, Optional[float]]:
    values = {
        "compressed_kv_decode_tok_s": report["compressed_kv"]["qat"].get("avg_decode_tok_per_s"),
        "target_decode_tok_s": report["target"]["qat"].get("avg_decode_tok_per_s"),
        "long_decode_tok_s": report["long_context"]["qat"].get("avg_decode_tok_per_s"),
        "resident_e2e_tok_s": report["resident"]["qat"].get("avg_e2e_tok_s"),
        "soak_aggregate_tok_s": report["soak"].get("aggregate_tok_s"),
        "backpressure_accepted_e2e_tok_s": report["backpressure"].get("avg_accepted_e2e_tok_s"),
    }
    values.update(mtp_metric_values(report.get("mtp") or {}))
    return values


def provider_baseline_value(entry: Dict[str, Any]) -> Optional[float]:
    for key in ("tok_s", "tokens_per_second", "decode_tok_per_s", "e2e_tok_s", "value"):
        value = to_float(entry.get(key))
        if value is not None:
            return value
    return None


def normalize_provider_baselines(payload: Any, source: str) -> List[Dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, list):
        raw_entries = payload
    elif isinstance(payload, dict):
        raw_entries = (
            payload.get("baselines")
            or payload.get("providers")
            or payload.get("benchmarks")
            or payload.get("comparisons")
            or ([payload] if provider_baseline_value(payload) is not None else [])
        )
    else:
        return []

    entries = []
    for index, raw_entry in enumerate(raw_entries):
        if not isinstance(raw_entry, dict):
            continue
        provider_tok_s = provider_baseline_value(raw_entry)
        provider = raw_entry.get("provider") or raw_entry.get("name")
        metric = raw_entry.get("metric")
        stat = raw_entry.get("stat") or raw_entry.get("baseline_stat")
        rate_source = raw_entry.get("rate_source")
        label = raw_entry.get("label") or raw_entry.get("id")
        entries.append(
            {
                "source": source,
                "index": index,
                "provider": provider or f"provider_{index + 1}",
                "label": label or f"{provider or f'provider_{index + 1}'}:{metric or 'resident_e2e_tok_s'}:{stat or index + 1}",
                "provider_explicit": bool(provider),
                "metric": metric or "resident_e2e_tok_s",
                "metric_explicit": bool(metric),
                "stat": stat,
                "rate_source": rate_source,
                "tok_s": provider_tok_s,
                "min_ratio": to_float(raw_entry.get("min_ratio")),
                "hardware": raw_entry.get("hardware") or raw_entry.get("device"),
                "model": raw_entry.get("model"),
                "tokens": to_int(raw_entry.get("tokens") or raw_entry.get("max_tokens") or raw_entry.get("generated_tokens")),
                "workload": raw_entry.get("workload") or raw_entry.get("prompt_name") or raw_entry.get("prompt"),
                "measured_at": raw_entry.get("measured_at") or raw_entry.get("date"),
                "source_url": raw_entry.get("source_url") or raw_entry.get("url"),
                "context": raw_entry.get("context"),
                "url": raw_entry.get("url"),
                "notes": raw_entry.get("notes"),
            }
        )
    return entries


def load_provider_baselines(args: argparse.Namespace) -> Dict[str, Any]:
    baselines = []
    errors = []
    for path_text in args.provider_baseline:
        path = pathlib.Path(path_text)
        try:
            payload = read_json(path)
            if payload is None:
                errors.append(f"{path}: missing")
                continue
            baselines.extend(normalize_provider_baselines(payload, str(path)))
        except Exception as exc:
            errors.append(f"{path}: {exc!r}")
    for index, inline_json in enumerate(args.provider_baseline_json):
        try:
            payload = json.loads(inline_json)
            baselines.extend(normalize_provider_baselines(payload, f"inline:{index + 1}"))
        except Exception as exc:
            errors.append(f"inline:{index + 1}: {exc!r}")
    return {"baselines": baselines, "errors": errors}


def check_name_fragment(value: Any) -> str:
    text = str(value or "unknown")
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in text)


def provider_metadata_missing(entry: Dict[str, Any]) -> List[str]:
    missing = []
    if not entry.get("provider_explicit"):
        missing.append("provider")
    if not entry.get("metric_explicit"):
        missing.append("metric")
    if entry.get("metric") not in PROVIDER_METRICS:
        missing.append("metric_supported")
    if entry.get("tok_s") is None:
        missing.append("tok_s")
    for field in ("model", "hardware", "tokens", "workload", "measured_at", "source_url"):
        value = entry.get(field)
        if value is None or value == "":
            missing.append(field)
    return missing


def compare_provider_baselines(report: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    loaded = load_provider_baselines(args)
    local_metrics = provider_metric_values(report)
    comparisons = []
    for entry in loaded["baselines"]:
        metric = entry["metric"]
        local_tok_s = local_metrics.get(metric)
        provider_tok_s = entry.get("tok_s")
        min_ratio = entry.get("min_ratio") if entry.get("min_ratio") is not None else args.min_provider_ratio
        ratio = local_tok_s / provider_tok_s if local_tok_s is not None and provider_tok_s not in (None, 0) else None
        missing_metadata = provider_metadata_missing(entry)
        comparisons.append(
            {
                **entry,
                "local_tok_s": local_tok_s,
                "provider_tok_s": provider_tok_s,
                "ratio": ratio,
                "required_ratio": min_ratio,
                "metadata_missing": missing_metadata,
                "metadata_ok": not missing_metadata,
                "ok": ratio is not None and ratio >= min_ratio,
            }
        )
    return {
        "local_metrics": local_metrics,
        "comparisons": comparisons,
        "load_errors": loaded["errors"],
        "required": bool(args.require_provider_comparison),
        "require_metadata": bool(args.require_provider_metadata),
        "min_provider_ratio": args.min_provider_ratio,
    }


def parse_competitive_floor(raw: str) -> Dict[str, Any]:
    if "=" in raw:
        metric, value = raw.split("=", 1)
    elif ":" in raw:
        metric, value = raw.split(":", 1)
    else:
        raise ValueError(f"competitive floor must use metric=tok_s: {raw!r}")
    metric = metric.strip()
    if metric not in PROVIDER_METRICS:
        raise ValueError(f"unsupported competitive floor metric {metric!r}; expected one of {sorted(PROVIDER_METRICS)}")
    floor = to_float(value.strip())
    if floor is None:
        raise ValueError(f"competitive floor for {metric} is not numeric: {value!r}")
    return {
        "metric": metric,
        "floor_tok_s": floor,
    }


def competitive_floor_comparison(report: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    local_metrics = provider_metric_values(report)
    floors = []
    errors = []
    for raw in args.competitive_floor:
        try:
            floor = parse_competitive_floor(raw)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        metric = floor["metric"]
        local_tok_s = local_metrics.get(metric)
        floors.append(
            {
                **floor,
                "local_tok_s": local_tok_s,
                "ok": local_tok_s is not None and local_tok_s >= floor["floor_tok_s"],
            }
        )
    return {
        "required": bool(args.require_competitive_floor),
        "load_errors": errors,
        "local_metrics": local_metrics,
        "floors": floors,
    }


def add_verdict_check(verdict: Dict[str, Any], name: str, ok: bool, detail: str) -> None:
    check = {"name": name, "ok": bool(ok), "detail": detail}
    verdict["checks"].append(check)
    if not ok:
        verdict["failures"].append(f"{name}: {detail}")


def summary_counter(summary: Dict[str, Any], reducer: str, name: str) -> Optional[float]:
    counters = summary.get(f"cuda_counter_{reducer}s") or {}
    return to_float(counters.get(name))


def add_min_counter_check(
    verdict: Dict[str, Any],
    check_name: str,
    summary: Dict[str, Any],
    counter_name: str,
    floor: int,
) -> None:
    if floor <= 0:
        return
    value = summary_counter(summary, "min", counter_name)
    add_verdict_check(
        verdict,
        check_name,
        value is not None and value >= floor,
        f"min={value} floor={floor}",
    )


def add_max_counter_check(
    verdict: Dict[str, Any],
    check_name: str,
    summary: Dict[str, Any],
    counter_name: str,
    ceiling: float,
) -> None:
    if math.isinf(ceiling):
        return
    value = summary_counter(summary, "max", counter_name)
    add_verdict_check(
        verdict,
        check_name,
        value is not None and value <= ceiling,
        f"max={value} ceiling={ceiling}",
    )


def make_provider_baseline_verdict(provider_comparison: Dict[str, Any]) -> Dict[str, Any]:
    verdict: Dict[str, Any] = {"ok": True, "checks": [], "failures": []}
    load_errors = provider_comparison.get("load_errors") or []
    add_verdict_check(
        verdict,
        "provider_baseline_load",
        not load_errors,
        f"errors={load_errors}",
    )
    comparisons = provider_comparison.get("comparisons") or []
    add_verdict_check(
        verdict,
        "provider_baseline_entries",
        bool(comparisons),
        f"entries={len(comparisons)}",
    )
    for comparison in comparisons:
        provider_name = check_name_fragment(comparison.get("label") or comparison.get("provider"))
        add_verdict_check(
            verdict,
            "provider_metadata_" + provider_name,
            bool(comparison.get("metadata_ok")),
            f"missing={comparison.get('metadata_missing')}",
        )
    verdict["ok"] = not verdict["failures"]
    return verdict


def make_verdict(report: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    verdict: Dict[str, Any] = {"ok": True, "checks": [], "failures": []}

    step_failures = report["steps"].get("failures") or []
    add_verdict_check(
        verdict,
        "gate_steps",
        not step_failures,
        f"failures={len(step_failures)}",
    )

    environment = report.get("environment")
    if environment is not None:
        status = environment.get("status")
        add_verdict_check(
            verdict,
            "cuda_environment_not_failed",
            status != "fail",
            f"status={status}",
        )
    if args.require_cuda_environment:
        add_verdict_check(
            verdict,
            "cuda_environment_present",
            environment is not None and environment.get("status") == "ok",
            f"status={(environment or {}).get('status')}",
        )
        if environment is not None:
            add_verdict_check(
                verdict,
                "cuda_environment_device",
                bool(environment.get("device_name") and environment.get("compute_capability")),
                f"device={environment.get('device_name')} sm={environment.get('compute_capability')}",
            )

    target = report["target"]
    target_qat = target["qat"]
    target_q4k = target["q4k"]
    target_ratio = target.get("qat_over_q4k_ratio")
    compressed_kv_qat = report["compressed_kv"]["qat"]
    if args.require_compressed_kv:
        add_verdict_check(
            verdict,
            "compressed_kv_qat_runs",
            int(compressed_kv_qat.get("count") or 0) > 0,
            f"runs={compressed_kv_qat.get('count')}",
        )
        add_verdict_check(
            verdict,
            "compressed_kv_qat_min_tokens",
            (compressed_kv_qat.get("min_tokens") or 0) >= args.min_compressed_kv_tokens,
            f"min_tokens={compressed_kv_qat.get('min_tokens')} floor={args.min_compressed_kv_tokens}",
        )
        add_verdict_check(
            verdict,
            "compressed_kv_qat_avg_tok_s",
            (compressed_kv_qat.get("avg_decode_tok_per_s") or 0.0) >= args.min_compressed_kv_tok_s,
            f"avg={compressed_kv_qat.get('avg_decode_tok_per_s')} floor={args.min_compressed_kv_tok_s}",
        )
        add_min_counter_check(
            verdict,
            "compressed_kv_graph_replays",
            compressed_kv_qat,
            "graph_capture_persistent_replays",
            args.min_compressed_kv_graph_replays,
        )
        add_max_counter_check(
            verdict,
            "compressed_kv_download_syncs",
            compressed_kv_qat,
            "download_syncs",
            args.max_compressed_kv_download_syncs,
        )
        add_max_counter_check(
            verdict,
            "compressed_kv_capacity_skips",
            compressed_kv_qat,
            "graph_capture_capacity_skips",
            args.max_compressed_kv_capacity_skips,
        )
        add_min_counter_check(
            verdict,
            "compressed_kv_compressed_v_reads",
            compressed_kv_qat,
            "device_kv_compressed_v_reads",
            args.min_compressed_kv_compressed_v_reads,
        )
        add_min_counter_check(
            verdict,
            "compressed_kv_compressed_v_writes",
            compressed_kv_qat,
            "device_kv_compressed_v_writes",
            args.min_compressed_kv_compressed_v_writes,
        )
        add_min_counter_check(
            verdict,
            "compressed_kv_paged_uploads",
            compressed_kv_qat,
            "device_kv_paged_block_table_uploads",
            args.min_compressed_kv_paged_uploads,
        )
        add_min_counter_check(
            verdict,
            "compressed_kv_identity_attention_reads",
            compressed_kv_qat,
            "device_kv_paged_identity_attention_reads",
            args.min_compressed_kv_identity_attention_reads,
        )
        add_min_counter_check(
            verdict,
            "compressed_kv_fast_gqa",
            compressed_kv_qat,
            "launch_attention_gqa_decode_fast",
            args.min_compressed_kv_fast_gqa,
        )
        add_max_counter_check(
            verdict,
            "compressed_kv_write_fallbacks",
            compressed_kv_qat,
            "device_kv_fail_write",
            args.max_compressed_kv_fail_writes,
        )
    if args.require_target:
        add_verdict_check(
            verdict,
            "target_qat_runs",
            int(target_qat.get("count") or 0) >= args.min_target_qat_repeats,
            f"runs={target_qat.get('count')} expected={args.min_target_qat_repeats}",
        )
        add_verdict_check(
            verdict,
            "target_qat_min_tokens",
            (target_qat.get("min_tokens") or 0) >= args.min_target_qat_tokens,
            f"min_tokens={target_qat.get('min_tokens')} expected={args.min_target_qat_tokens}",
        )
        add_verdict_check(
            verdict,
            "target_qat_avg_tok_s",
            (target_qat.get("avg_decode_tok_per_s") or 0.0) >= args.min_target_qat_tok_s,
            f"avg={target_qat.get('avg_decode_tok_per_s')} floor={args.min_target_qat_tok_s}",
        )
        add_verdict_check(
            verdict,
            "target_qat_min_tok_s",
            (target_qat.get("min_decode_tok_per_s") or 0.0) >= args.min_target_qat_run_tok_s,
            f"min={target_qat.get('min_decode_tok_per_s')} floor={args.min_target_qat_run_tok_s}",
        )
        add_verdict_check(
            verdict,
            "target_q4k_baseline_present",
            int(target_q4k.get("count") or 0) > 0,
            f"runs={target_q4k.get('count')}",
        )
        add_verdict_check(
            verdict,
            "target_qat_over_q4k",
            target_ratio is not None and target_ratio >= args.min_target_qat_over_q4k_ratio,
            f"ratio={target_ratio} floor={args.min_target_qat_over_q4k_ratio}",
        )

    if args.require_long:
        long_qat = report["long_context"]["qat"]
        add_verdict_check(
            verdict,
            "long_qat_present",
            int(long_qat.get("count") or 0) > 0,
            f"runs={long_qat.get('count')}",
        )
        add_verdict_check(
            verdict,
            "long_qat_tokens",
            (long_qat.get("min_tokens") or 0) >= args.min_long_qat_tokens,
            f"min_tokens={long_qat.get('min_tokens')} floor={args.min_long_qat_tokens}",
        )
        add_verdict_check(
            verdict,
            "long_qat_tok_s",
            (long_qat.get("avg_decode_tok_per_s") or 0.0) >= args.min_long_qat_tok_s,
            f"avg={long_qat.get('avg_decode_tok_per_s')} floor={args.min_long_qat_tok_s}",
        )

    resident = report["resident"]
    resident_qat = resident["qat"]
    resident_q4k = resident["q4k"]
    resident_ratio = resident.get("qat_over_q4k_ratio")
    if args.require_resident:
        add_verdict_check(
            verdict,
            "resident_qat_rows",
            int(resident_qat.get("row_count") or 0) >= args.min_resident_qat_repeats,
            f"rows={resident_qat.get('row_count')} expected={args.min_resident_qat_repeats}",
        )
        add_verdict_check(
            verdict,
            "resident_qat_tokens",
            (resident_qat.get("total_completion_tokens") or 0) >= args.min_resident_qat_tokens * args.min_resident_qat_repeats,
            f"tokens={resident_qat.get('total_completion_tokens')} expected={args.min_resident_qat_tokens * args.min_resident_qat_repeats}",
        )
        add_verdict_check(
            verdict,
            "resident_qat_avg_tok_s",
            (resident_qat.get("avg_e2e_tok_s") or 0.0) >= args.min_resident_qat_tok_s,
            f"avg={resident_qat.get('avg_e2e_tok_s')} floor={args.min_resident_qat_tok_s}",
        )
        graph_log = resident.get("graph_log") or {}
        if graph_log.get("present"):
            add_verdict_check(
                verdict,
                "resident_graph_safe",
                not graph_log.get("unsafe_markers"),
                f"unsafe_markers={graph_log.get('unsafe_markers')}",
            )
    if args.require_resident_q4k:
        add_verdict_check(
            verdict,
            "resident_q4k_rows",
            int(resident_q4k.get("row_count") or 0) >= args.min_resident_qat_repeats,
            f"rows={resident_q4k.get('row_count')} expected={args.min_resident_qat_repeats}",
        )
        add_verdict_check(
            verdict,
            "resident_qat_over_q4k",
            resident_ratio is not None and resident_ratio >= args.min_resident_qat_over_q4k_ratio,
            f"ratio={resident_ratio} floor={args.min_resident_qat_over_q4k_ratio}",
        )

    soak = report["soak"]
    if args.require_soak:
        add_verdict_check(
            verdict,
            "soak_rows",
            int(soak.get("row_count") or 0) >= args.min_soak_requests,
            f"rows={soak.get('row_count')} expected={args.min_soak_requests}",
        )
        add_verdict_check(
            verdict,
            "soak_aggregate_tok_s",
            (soak.get("aggregate_tok_s") or 0.0) >= args.min_soak_aggregate_tok_s,
            f"aggregate={soak.get('aggregate_tok_s')} floor={args.min_soak_aggregate_tok_s}",
        )
        add_verdict_check(
            verdict,
            "soak_min_request_tok_s",
            (soak.get("min_e2e_tok_s") or 0.0) >= args.min_soak_request_tok_s,
            f"min={soak.get('min_e2e_tok_s')} floor={args.min_soak_request_tok_s}",
        )
        add_verdict_check(
            verdict,
            "soak_p95_e2e_ms",
            soak.get("p95_e2e_ms") is not None and soak.get("p95_e2e_ms") <= args.max_soak_p95_e2e_ms,
            f"p95={soak.get('p95_e2e_ms')} max={args.max_soak_p95_e2e_ms}",
        )

    backpressure = report["backpressure"]
    if args.require_backpressure:
        add_verdict_check(
            verdict,
            "backpressure_rows",
            int(backpressure.get("row_count") or 0) >= args.min_backpressure_accepted + args.min_backpressure_rejected,
            f"rows={backpressure.get('row_count')} expected_min={args.min_backpressure_accepted + args.min_backpressure_rejected}",
        )
        add_verdict_check(
            verdict,
            "backpressure_accepted",
            int(backpressure.get("accepted") or 0) >= args.min_backpressure_accepted,
            f"accepted={backpressure.get('accepted')} floor={args.min_backpressure_accepted}",
        )
        add_verdict_check(
            verdict,
            "backpressure_rejected",
            int(backpressure.get("rejected") or 0) >= args.min_backpressure_rejected,
            f"rejected={backpressure.get('rejected')} floor={args.min_backpressure_rejected}",
        )
        add_verdict_check(
            verdict,
            "backpressure_reject_latency",
            backpressure.get("max_reject_ms") is not None and backpressure.get("max_reject_ms") <= args.max_backpressure_reject_ms,
            f"max_reject_ms={backpressure.get('max_reject_ms')} max={args.max_backpressure_reject_ms}",
        )
        metrics = backpressure.get("metrics") or {}
        add_verdict_check(
            verdict,
            "backpressure_queue_rejections_metric",
            metrics.get("antfly_inference_request_queue_rejections_total", 0.0) >= int(backpressure.get("rejected") or 0),
            f"metric={metrics.get('antfly_inference_request_queue_rejections_total')} rejected={backpressure.get('rejected')}",
        )
        graph_log = backpressure.get("graph_log") or {}
        if graph_log.get("present"):
            add_verdict_check(
                verdict,
                "backpressure_graph_safe",
                not graph_log.get("unsafe_markers"),
                f"unsafe_markers={graph_log.get('unsafe_markers')}",
            )

    mtp = report.get("mtp") or {}
    if args.require_mtp:
        add_verdict_check(
            verdict,
            "mtp_matrices_present",
            bool(mtp),
            f"matrices={len(mtp)}",
        )
    for matrix_name, matrix in mtp.items():
        target_rate = to_float(matrix.get("target_avg_decode_tok_s"))
        candidates = matrix.get("candidates") or []
        add_verdict_check(
            verdict,
            "mtp_" + check_name_fragment(matrix_name) + "_target_rows",
            target_rate is not None,
            f"target_avg={target_rate}",
        )
        for candidate in candidates:
            if not candidate.get("active"):
                continue
            candidate_name = "mtp_" + check_name_fragment(
                f"{matrix_name}_{candidate.get('assistant')}_k{candidate.get('spec_k')}_{candidate.get('case')}"
            )
            ratio = to_float(candidate.get("target_ratio"))
            tokens = to_int(candidate.get("tokens"))
            enforce_speed = tokens is None or tokens >= args.min_mtp_active_speed_tokens
            add_verdict_check(
                verdict,
                candidate_name + "_active_ratio",
                (not enforce_speed) or (ratio is not None and ratio >= args.min_mtp_active_speed_ratio),
                (
                    f"ratio={ratio} floor={args.min_mtp_active_speed_ratio} "
                    f"rate={candidate.get('decode_tok_s')} target={candidate.get('target_decode_tok_s')} "
                    f"tokens={tokens} min_tokens={args.min_mtp_active_speed_tokens}"
                ),
            )
            dedicated_fallbacks = to_int(candidate.get("dedicated_runtime_fallbacks"))
            if dedicated_fallbacks is not None:
                add_verdict_check(
                    verdict,
                    candidate_name + "_dedicated_runtime",
                    dedicated_fallbacks == 0,
                    f"dedicated_runtime_fallbacks={dedicated_fallbacks}",
                )
            device_fallbacks = to_int(candidate.get("device_verify_commit_fallbacks"))
            if device_fallbacks is not None:
                add_verdict_check(
                    verdict,
                    candidate_name + "_device_verify_commit",
                    device_fallbacks == 0,
                    f"device_verify_commit_fallbacks={device_fallbacks}",
                )
            if args.require_mtp_preproject_fusion:
                preproject_hits = to_int(candidate.get("draft_mtp_preproject_fused_hits"))
                preproject_fallbacks = to_int(candidate.get("draft_mtp_preproject_fused_fallbacks"))
                add_verdict_check(
                    verdict,
                    candidate_name + "_preproject_fusion_hits",
                    preproject_hits is not None and preproject_hits > 0,
                    f"draft_mtp_preproject_fused_hits={preproject_hits}",
                )
                add_verdict_check(
                    verdict,
                    candidate_name + "_preproject_fusion_fallbacks",
                    preproject_fallbacks is not None and preproject_fallbacks == 0,
                    f"draft_mtp_preproject_fused_fallbacks={preproject_fallbacks}",
                )
            if args.require_mtp_masked_select_fusion:
                masked_select_hits = to_int(candidate.get("draft_mtp_masked_select_hidden_multiblock_hits"))
                masked_select_fallbacks = to_int(candidate.get("draft_mtp_masked_select_hidden_fused_fallbacks"))
                add_verdict_check(
                    verdict,
                    candidate_name + "_masked_select_fusion_hits",
                    masked_select_hits is not None and masked_select_hits > 0,
                    f"draft_mtp_masked_select_hidden_multiblock_hits={masked_select_hits}",
                )
                add_verdict_check(
                    verdict,
                    candidate_name + "_masked_select_fusion_fallbacks",
                    masked_select_fallbacks is not None and masked_select_fallbacks == 0,
                    f"draft_mtp_masked_select_hidden_fused_fallbacks={masked_select_fallbacks}",
                )

    replay_stability = report.get("mtp_replay_stability") or {}
    if replay_stability.get("present"):
        add_verdict_check(
            verdict,
            "mtp_replay_stability_ok",
            bool(replay_stability.get("ok")),
            f"row_count={replay_stability.get('row_count')} errors={replay_stability.get('errors')}",
        )
        add_verdict_check(
            verdict,
            "mtp_replay_stability_rows",
            int(replay_stability.get("row_count") or 0) > 0,
            f"row_count={replay_stability.get('row_count')}",
        )

    target_equivalence = report.get("mtp_target_equivalence") or {}
    if args.require_mtp_target_equivalence or target_equivalence.get("present"):
        add_verdict_check(
            verdict,
            "mtp_target_equivalence_present",
            bool(target_equivalence.get("present")),
            f"present={target_equivalence.get('present')}",
        )
    if target_equivalence.get("present"):
        add_verdict_check(
            verdict,
            "mtp_target_equivalence_ok",
            bool(target_equivalence.get("ok")),
            f"targets={target_equivalence.get('target_case_count')} candidates={target_equivalence.get('candidate_count')} errors={target_equivalence.get('errors')}",
        )

    replay_512 = report.get("mtp_replay_512") or {}
    if replay_512.get("present"):
        add_verdict_check(
            verdict,
            "mtp_replay_512_candidates",
            int(replay_512.get("candidate_count") or 0) > 0,
            f"candidates={replay_512.get('candidate_count')}",
        )

    hidden_ab = report.get("mtp_hidden_ab") or {}
    if args.require_mtp_hidden_ab or hidden_ab.get("present"):
        add_verdict_check(
            verdict,
            "mtp_hidden_ab_present",
            bool(hidden_ab.get("present")),
            f"present={hidden_ab.get('present')}",
        )
    if hidden_ab.get("present"):
        ratios = [
            ratio
            for ratio in (to_float(pair.get("hidden_over_default_ratio")) for pair in hidden_ab.get("pairs") or [])
            if ratio is not None
        ]
        ratio_pass = (
            int(hidden_ab.get("pair_count") or 0) >= args.min_mtp_hidden_ab_pairs
            and len(ratios) >= args.min_mtp_hidden_ab_pairs
            and all(ratio >= args.min_mtp_hidden_ab_ratio for ratio in ratios)
        )
        counters_ok = bool(hidden_ab.get("hidden_selector_counters_ok")) and bool(hidden_ab.get("default_hidden_selector_disabled"))
        hidden_ab["promotion_recommendation"] = (
            "fail"
            if not counters_ok
            else ("promote" if ratio_pass else "keep_opt_in")
        )
        hidden_ab["promotion_ratio_pass"] = ratio_pass
        hidden_ab["promotion_ratio_floor"] = args.min_mtp_hidden_ab_ratio
        add_verdict_check(
            verdict,
            "mtp_hidden_ab_pairs",
            int(hidden_ab.get("pair_count") or 0) >= args.min_mtp_hidden_ab_pairs,
            f"pairs={hidden_ab.get('pair_count')} floor={args.min_mtp_hidden_ab_pairs}",
        )
        add_verdict_check(
            verdict,
            "mtp_hidden_ab_default_selector_off",
            bool(hidden_ab.get("default_hidden_selector_disabled")),
            f"default_hidden_selector_disabled={hidden_ab.get('default_hidden_selector_disabled')}",
        )
        add_verdict_check(
            verdict,
            "mtp_hidden_ab_hidden_counters",
            bool(hidden_ab.get("hidden_selector_counters_ok")),
            f"hidden_selector_counters_ok={hidden_ab.get('hidden_selector_counters_ok')} min_ratio={hidden_ab.get('min_ratio')}",
        )

    acceptance_matrix = report.get("mtp_acceptance_matrix") or {}
    if acceptance_matrix.get("present"):
        add_verdict_check(
            verdict,
            "mtp_acceptance_matrix_combos",
            int(acceptance_matrix.get("combo_count") or 0) >= 8,
            f"combos={acceptance_matrix.get('combo_count')}",
        )
        add_verdict_check(
            verdict,
            "mtp_acceptance_matrix_candidates",
            int(acceptance_matrix.get("candidate_count") or 0) > 0,
            f"candidates={acceptance_matrix.get('candidate_count')}",
        )

    donor_matrix = report.get("mtp_donor_matrix") or {}
    if args.require_mtp_donor_matrix or donor_matrix.get("present"):
        add_verdict_check(
            verdict,
            "mtp_donor_matrix_present",
            bool(donor_matrix.get("present")),
            f"present={donor_matrix.get('present')}",
        )
    if donor_matrix.get("present"):
        add_verdict_check(
            verdict,
            "mtp_donor_matrix_combos",
            int(donor_matrix.get("combo_count") or 0) >= 2,
            f"combos={donor_matrix.get('combo_count')}",
        )
        add_verdict_check(
            verdict,
            "mtp_donor_matrix_candidates",
            int(donor_matrix.get("candidate_count") or 0) > 0,
            f"candidates={donor_matrix.get('candidate_count')}",
        )

    if args.require_mtp_benefit:
        best_active = ((mtp.get("mtp_e4b_qat") or {}).get("best_active") or {})
        best_active_ratio = to_float(best_active.get("target_ratio"))
        add_verdict_check(
            verdict,
            "mtp_e4b_qat_best_active_benefit",
            best_active_ratio is not None and best_active_ratio >= args.min_mtp_benefit_ratio,
            f"ratio={best_active_ratio} floor={args.min_mtp_benefit_ratio} candidate={best_active}",
        )
        replay_best = (replay_512.get("best_active") or {}) if replay_512.get("present") else {}
        replay_ratio = to_float(replay_best.get("target_ratio"))
        add_verdict_check(
            verdict,
            "mtp_replay_512_best_active_benefit",
            replay_ratio is not None and replay_ratio >= args.min_mtp_replay_512_benefit_ratio,
            f"ratio={replay_ratio} floor={args.min_mtp_replay_512_benefit_ratio} candidate={replay_best}",
        )

    provider_comparison = report.get("provider_comparison") or {}
    if args.require_provider_comparison or provider_comparison.get("comparisons"):
        load_errors = provider_comparison.get("load_errors") or []
        add_verdict_check(
            verdict,
            "provider_baseline_load",
            not load_errors,
            f"errors={load_errors}",
        )
    if args.require_provider_comparison:
        comparisons = provider_comparison.get("comparisons") or []
        add_verdict_check(
            verdict,
            "provider_comparison_present",
            bool(comparisons),
            f"comparisons={len(comparisons)}",
        )
        for comparison in comparisons:
            if args.require_provider_metadata:
                metadata_name = "provider_metadata_" + check_name_fragment(comparison.get("label") or comparison.get("provider"))
                add_verdict_check(
                    verdict,
                    metadata_name,
                    bool(comparison.get("metadata_ok")),
                    f"missing={comparison.get('metadata_missing')}",
                )
            name = "provider_comparison_" + check_name_fragment(comparison.get("label") or comparison.get("provider"))
            add_verdict_check(
                verdict,
                name,
                bool(comparison.get("ok")),
                (
                    f"metric={comparison.get('metric')} rate_source={comparison.get('rate_source')} stat={comparison.get('stat')} local={comparison.get('local_tok_s')} "
                    f"provider={comparison.get('provider_tok_s')} ratio={comparison.get('ratio')} "
                    f"floor={comparison.get('required_ratio')}"
                ),
            )

    competitive = report.get("competitive_floor") or {}
    if args.require_competitive_floor or competitive.get("floors") or competitive.get("load_errors"):
        errors = competitive.get("load_errors") or []
        add_verdict_check(
            verdict,
            "competitive_floor_parse",
            not errors,
            f"errors={errors}",
        )
    if args.require_competitive_floor:
        floors = competitive.get("floors") or []
        add_verdict_check(
            verdict,
            "competitive_floor_present",
            bool(floors),
            f"floors={len(floors)}",
        )
        for floor in floors:
            metric = check_name_fragment(floor.get("metric"))
            add_verdict_check(
                verdict,
                "competitive_floor_" + metric,
                bool(floor.get("ok")),
                f"local={floor.get('local_tok_s')} floor={floor.get('floor_tok_s')}",
            )

    verdict["ok"] = not verdict["failures"]
    return verdict


def validate_provider_baselines_only(args: argparse.Namespace) -> int:
    report: Dict[str, Any] = {
        "compressed_kv": {"qat": {}},
        "target": {"qat": {}},
        "long_context": {"qat": {}},
        "resident": {"qat": {}},
        "soak": {},
        "backpressure": {},
        "mtp": {},
        "mtp_target_equivalence": {},
        "mtp_replay_stability": {},
        "mtp_replay_512": {},
        "mtp_acceptance_matrix": {},
        "mtp_donor_matrix": {},
    }
    provider_comparison = compare_provider_baselines(report, args)
    validation = {
        "provider_comparison": provider_comparison,
        "verdict": make_provider_baseline_verdict(provider_comparison),
    }
    output = pathlib.Path(args.output)
    output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "provider_baseline_validation="
        f"{output} verdict={'ok' if validation['verdict']['ok'] else 'fail'} "
        f"entries={len(provider_comparison.get('comparisons') or [])}"
    )
    if not validation["verdict"]["ok"]:
        for failure in validation["verdict"]["failures"]:
            print(f"verdict_failure: {failure}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir")
    parser.add_argument("--output", required=True)
    parser.add_argument("--validate-provider-baselines-only", action="store_true")
    parser.add_argument("--require-cuda-environment", action="store_true")
    parser.add_argument("--require-compressed-kv", action="store_true")
    parser.add_argument("--require-target", action="store_true")
    parser.add_argument("--require-long", action="store_true")
    parser.add_argument("--require-resident", action="store_true")
    parser.add_argument("--require-resident-q4k", action="store_true")
    parser.add_argument("--require-soak", action="store_true")
    parser.add_argument("--require-backpressure", action="store_true")
    parser.add_argument("--require-mtp", action="store_true")
    parser.add_argument("--require-mtp-target-equivalence", action="store_true")
    parser.add_argument("--require-mtp-donor-matrix", action="store_true")
    parser.add_argument("--require-mtp-benefit", action="store_true")
    parser.add_argument("--require-mtp-preproject-fusion", action="store_true")
    parser.add_argument("--require-mtp-masked-select-fusion", action="store_true")
    parser.add_argument("--require-mtp-hidden-ab", action="store_true")
    parser.add_argument("--require-provider-comparison", action="store_true")
    parser.add_argument("--require-provider-metadata", action="store_true")
    parser.add_argument("--provider-baseline", action="append", default=[])
    parser.add_argument("--provider-baseline-json", action="append", default=[])
    parser.add_argument("--min-provider-ratio", type=float, default=1.0)
    parser.add_argument("--competitive-floor", action="append", default=[])
    parser.add_argument("--require-competitive-floor", action="store_true")
    parser.add_argument("--min-compressed-kv-tok-s", type=float, default=0.0)
    parser.add_argument("--min-compressed-kv-tokens", type=int, default=0)
    parser.add_argument("--min-compressed-kv-graph-replays", type=int, default=0)
    parser.add_argument("--max-compressed-kv-download-syncs", type=float, default=float("inf"))
    parser.add_argument("--max-compressed-kv-capacity-skips", type=float, default=float("inf"))
    parser.add_argument("--min-compressed-kv-compressed-v-reads", type=int, default=0)
    parser.add_argument("--min-compressed-kv-compressed-v-writes", type=int, default=0)
    parser.add_argument("--min-compressed-kv-paged-uploads", type=int, default=0)
    parser.add_argument("--min-compressed-kv-identity-attention-reads", type=int, default=0)
    parser.add_argument("--min-compressed-kv-fast-gqa", type=int, default=0)
    parser.add_argument("--max-compressed-kv-fail-writes", type=float, default=float("inf"))
    parser.add_argument("--min-target-qat-tok-s", type=float, default=0.0)
    parser.add_argument("--min-target-qat-run-tok-s", type=float, default=0.0)
    parser.add_argument("--min-target-qat-over-q4k-ratio", type=float, default=0.0)
    parser.add_argument("--min-target-qat-tokens", type=int, default=0)
    parser.add_argument("--min-target-qat-repeats", type=int, default=1)
    parser.add_argument("--min-long-qat-tok-s", type=float, default=0.0)
    parser.add_argument("--min-long-qat-tokens", type=int, default=0)
    parser.add_argument("--min-resident-qat-tok-s", type=float, default=0.0)
    parser.add_argument("--min-resident-qat-over-q4k-ratio", type=float, default=0.0)
    parser.add_argument("--min-resident-qat-tokens", type=int, default=0)
    parser.add_argument("--min-resident-qat-repeats", type=int, default=1)
    parser.add_argument("--min-soak-requests", type=int, default=1)
    parser.add_argument("--min-soak-aggregate-tok-s", type=float, default=0.0)
    parser.add_argument("--min-soak-request-tok-s", type=float, default=0.0)
    parser.add_argument("--max-soak-p95-e2e-ms", type=float, default=float("inf"))
    parser.add_argument("--min-backpressure-accepted", type=int, default=1)
    parser.add_argument("--min-backpressure-rejected", type=int, default=1)
    parser.add_argument("--max-backpressure-reject-ms", type=float, default=float("inf"))
    parser.add_argument("--min-mtp-active-speed-ratio", type=float, default=1.0)
    parser.add_argument("--min-mtp-active-speed-tokens", type=int, default=1)
    parser.add_argument("--min-mtp-benefit-ratio", type=float, default=1.02)
    parser.add_argument("--min-mtp-replay-512-benefit-ratio", type=float, default=1.02)
    parser.add_argument("--min-mtp-hidden-ab-ratio", type=float, default=1.03)
    parser.add_argument("--min-mtp-hidden-ab-pairs", type=int, default=2)
    args = parser.parse_args()

    if args.validate_provider_baselines_only:
        args.require_provider_metadata = True
        args.require_provider_comparison = True
        return validate_provider_baselines_only(args)
    if not args.out_dir:
        parser.error("--out-dir is required unless --validate-provider-baselines-only is used")

    out_dir = pathlib.Path(args.out_dir)
    output = pathlib.Path(args.output)
    environment = read_json(out_dir / "cuda_environment.json")

    target_qat_names = ["target_e4b_qat", "e4b_qat_cuda"] + [
        f"target_e4b_qat_run{i}" for i in range(1, 17)
    ] + [
        f"e4b_qat_cuda_run{i}" for i in range(1, 17)
    ]
    target_q4k_names = ["target_e4b_q4k", "e4b_q4k_baseline"]
    target_qat = summarize_rates(collect_json_runs(out_dir, target_qat_names))
    target_q4k_runs = collect_json_runs(out_dir, target_q4k_names)
    target_q4k = summarize_rates(target_q4k_runs)
    target_qat_rate = target_qat.get("avg_decode_tok_per_s")
    target_q4k_rate = target_q4k.get("avg_decode_tok_per_s")

    compressed_kv_runs = collect_json_runs(out_dir, ["e4b_qat_compressed_kv"])
    compressed_kv_qat = summarize_rates(compressed_kv_runs)
    compressed_kv_rate = compressed_kv_qat.get("avg_decode_tok_per_s")
    long_runs = collect_json_runs(out_dir, ["e4b_qat_long", "e4b_qat_cuda_long"])
    resident_qat = summarize_tsv(out_dir / "e4b_qat_resident_cuda_server.tsv")
    resident_q4k = summarize_tsv(out_dir / "e4b_q4k_resident_cuda_server.tsv")
    resident_qat_rate = resident_qat.get("avg_e2e_tok_s")
    resident_q4k_rate = resident_q4k.get("avg_e2e_tok_s")

    soak = summarize_tsv(out_dir / "e4b_qat_resident_soak.tsv")
    soak_meta = read_json(out_dir / "e4b_qat_resident_soak_meta.json") or {}
    soak["aggregate_tok_s"] = to_float(soak_meta.get("aggregate_tok_s"))
    soak["meta"] = soak_meta

    server_log = out_dir / "e4b_qat_resident_server.log"
    mtp = collect_mtp_matrices(out_dir)
    report: Dict[str, Any] = {
        "out_dir": str(out_dir),
        "environment": environment,
        "steps": collect_steps(out_dir),
        "compressed_kv": {
            "qat": compressed_kv_qat,
        },
        "target": {
            "qat": target_qat,
            "q4k": target_q4k,
            "qat_over_q4k_ratio": (
                target_qat_rate / target_q4k_rate
                if target_qat_rate is not None and target_q4k_rate not in (None, 0)
                else None
            ),
        },
        "long_context": {
            "qat": summarize_rates(long_runs),
        },
        "resident": {
            "qat": resident_qat,
            "q4k": resident_q4k,
            "qat_over_q4k_ratio": (
                resident_qat_rate / resident_q4k_rate
                if resident_qat_rate is not None and resident_q4k_rate not in (None, 0)
                else None
            ),
            "graph_log": graph_log_summary(server_log),
        },
        "soak": soak,
        "backpressure": summarize_backpressure(out_dir, server_log),
        "mtp": mtp,
        "mtp_target_equivalence": collect_mtp_target_equivalence(out_dir),
        "mtp_replay_stability": collect_mtp_replay_stability(out_dir),
        "mtp_replay_512": collect_mtp_replay_512(out_dir),
        "mtp_hidden_ab": collect_mtp_hidden_ab(out_dir),
        "mtp_acceptance_matrix": collect_mtp_acceptance_matrix(out_dir),
        "mtp_donor_matrix": collect_mtp_donor_matrix(out_dir),
    }
    report["provider_comparison"] = compare_provider_baselines(report, args)
    report["competitive_floor"] = competitive_floor_comparison(report, args)
    report["verdict"] = make_verdict(report, args)
    report["ok"] = not report["steps"]["failures"] and bool(report["verdict"]["ok"])

    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    env_label = "unknown"
    if environment:
        env_label = f"{environment.get('device_name', 'unknown')}/{environment.get('compute_capability', 'unknown')}"
    target_ratio = report["target"]["qat_over_q4k_ratio"]
    resident_ratio = report["resident"]["qat_over_q4k_ratio"]
    print(
        "qat_production_summary="
        f"{output} env={env_label} "
        f"verdict={'ok' if report['verdict']['ok'] else 'fail'} "
        f"compressed_kv_qat_avg={compressed_kv_rate if compressed_kv_rate is not None else 'n/a'} "
        f"target_qat_avg={target_qat_rate if target_qat_rate is not None else 'n/a'} "
        f"target_qat_over_q4k={target_ratio if target_ratio is not None else 'n/a'} "
        f"resident_qat_avg={resident_qat_rate if resident_qat_rate is not None else 'n/a'} "
        f"resident_qat_over_q4k={resident_ratio if resident_ratio is not None else 'n/a'}"
    )
    if not report["verdict"]["ok"]:
        for failure in report["verdict"]["failures"]:
            print(f"verdict_failure: {failure}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
