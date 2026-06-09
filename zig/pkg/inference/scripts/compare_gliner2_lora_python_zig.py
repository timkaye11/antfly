#!/usr/bin/env python3
"""Run a small GLiNER2 LoRA apples-to-apples smoke comparison.

The script intentionally keeps the comparison narrow: same local model dir,
same NER smoke JSONL, same LoRA rank/alpha/dropout/target groups, one backend
for Zig, and one JSON report with raw command outputs plus extracted metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any


DEFAULT_PYTHON = "/private/tmp/gliner2-parity-venv/bin/python"
DEFAULT_MODEL_DIR = "/private/tmp/termite-models/gliner2"
DEFAULT_PYTHON_MODEL = "fastino/gliner2-base-v1"
DEFAULT_OUT_DIR = "/private/tmp/termite-gliner2-apples-to-apples"
DEFAULT_LABELS = "person,organization,location"
LORA_TARGETS = "encoder,span_rep,classifier,count_embed,count_pred"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def inference_dir() -> Path:
    return repo_root() / "zig" / "pkg" / "inference"


def default_train_data() -> Path:
    return inference_dir() / "testdata" / "gliner2_ner_smoke.jsonl"


def parse_label_csv(labels_csv: str) -> set[str]:
    return {label.strip() for label in labels_csv.split(",") if label.strip()}


def summarize_upstream_output(output: dict[str, Any], labels: set[str], allowed_labels: set[str] | None) -> dict[str, int]:
    counts = {
        "entity_mentions": 0,
        "classifications": len(output.get("classifications", []) or []),
        "json_structures": len(output.get("json_structures", []) or []),
        "relations": len(output.get("relations", []) or []),
    }
    for label, mentions in (output.get("entities") or {}).items():
        if allowed_labels is not None and label not in allowed_labels:
            continue
        labels.add(label)
        counts["entity_mentions"] += len(mentions or [])
    return counts


def normalize_python_record(record: dict[str, Any], allowed_labels: set[str] | None) -> tuple[dict[str, Any], dict[str, int], set[str]]:
    labels: set[str] = set()
    if "input" in record and "output" in record:
        output = dict(record.get("output") or {})
        if allowed_labels is not None and "entities" in output:
            output["entities"] = {
                label: mentions
                for label, mentions in (output.get("entities") or {}).items()
                if label in allowed_labels
            }
        counts = summarize_upstream_output(output, labels, allowed_labels)
        return {"input": record["input"], "output": output}, counts, labels

    grouped: dict[str, list[str]] = {}
    for ent in record.get("entities", []):
        label = ent["label"]
        if allowed_labels is not None and label not in allowed_labels:
            continue
        grouped.setdefault(label, []).append(ent["text"])
        labels.add(label)
    return (
        {"input": record["text"], "output": {"entities": grouped}},
        {"entity_mentions": sum(len(v) for v in grouped.values()), "classifications": 0, "json_structures": 0, "relations": 0},
        labels,
    )


def convert_to_python_jsonl(src: Path, dst: Path, allowed_labels: set[str] | None = None) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    examples = 0
    mentions = 0
    task_counts = {"classifications": 0, "json_structures": 0, "relations": 0}
    labels: set[str] = set()
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            normalized, counts, record_labels = normalize_python_record(record, allowed_labels)
            mentions += counts["entity_mentions"]
            for key in task_counts:
                task_counts[key] += counts[key]
            labels.update(record_labels)
            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            examples += 1
    return {"examples": examples, "mentions": mentions, "labels": sorted(labels), **task_counts, "path": str(dst)}


def summarize_python_jsonl(src: Path, allowed_labels: set[str] | None = None) -> dict[str, Any]:
    examples = 0
    mentions = 0
    task_counts = {"classifications": 0, "json_structures": 0, "relations": 0}
    labels: set[str] = set()
    with src.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            _, counts, record_labels = normalize_python_record(record, allowed_labels)
            mentions += counts["entity_mentions"]
            for key in task_counts:
                task_counts[key] += counts[key]
            labels.update(record_labels)
            examples += 1
    return {"examples": examples, "mentions": mentions, "labels": sorted(labels), **task_counts}


def convert_limited_to_python_jsonl(src: Path, dst: Path, max_examples: int, allowed_labels: set[str] | None = None) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    examples = 0
    mentions = 0
    task_counts = {"classifications": 0, "json_structures": 0, "relations": 0}
    labels: set[str] = set()
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if examples >= max_examples:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            normalized, counts, record_labels = normalize_python_record(record, allowed_labels)
            mentions += counts["entity_mentions"]
            for key in task_counts:
                task_counts[key] += counts[key]
            labels.update(record_labels)
            fout.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            examples += 1
    return {"examples": examples, "mentions": mentions, "labels": sorted(labels), **task_counts, "path": str(dst)}


def prepare_python_model_dir(model_dir: Path, out_dir: Path) -> Path:
    """Create a Python-loader-compatible view of the local GLiNER2 bundle.

    The Termite local model cache may omit `encoder_config.model_type`; the
    upstream Python package delegates that file to Hugging Face AutoConfig,
    which requires it.  Patch only the comparison copy and symlink large files.
    """
    dst = out_dir / "python_model"
    dst.mkdir(parents=True, exist_ok=True)
    root_config = dst / "config.json"
    if not (model_dir / "config.json").exists() and not root_config.exists():
        root_config.write_text(
            json.dumps(
                {
                    "model_name": "microsoft/deberta-v3-base",
                    "model_type": "extractor",
                    "counting_layer": "count_lstm_v2",
                    "token_pooling": "first",
                    "max_width": 8,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    tokenizer_config = dst / "tokenizer_config.json"
    if not (model_dir / "tokenizer_config.json").exists() and not tokenizer_config.exists():
        tokenizer_config.write_text(
            json.dumps(
                {
                    "tokenizer_class": "DebertaV2TokenizerFast",
                    "model_max_length": 512,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    for src in model_dir.iterdir():
        target = dst / src.name
        if src.name == "encoder_config":
            target.mkdir(exist_ok=True)
            for child in src.iterdir():
                child_target = target / child.name
                if child.name == "config.json":
                    cfg = json.loads(child.read_text(encoding="utf-8"))
                    cfg.setdefault("model_type", "deberta-v2")
                    cfg.setdefault("relative_attention", True)
                    cfg.setdefault("position_biased_input", False)
                    cfg.setdefault("pos_att_type", ["p2c", "c2p"])
                    cfg.setdefault("max_relative_positions", -1)
                    cfg.setdefault("norm_rel_ebd", "layer_norm")
                    cfg.setdefault("share_att_key", True)
                    cfg.setdefault("type_vocab_size", 0)
                    cfg.setdefault("layer_norm_eps", 1e-7)
                    child_target.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                elif not child_target.exists():
                    child_target.symlink_to(child)
        elif not target.exists():
            target.symlink_to(src)
    return dst


def run_command(cmd: list[str], cwd: Path, timeout: int | None = None, env: dict[str, str] | None = None) -> dict[str, Any]:
    started = time.time()
    run_env = {**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    if env:
        run_env.update(env)
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env=run_env,
    )
    return {
        "argv": cmd,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "elapsed_seconds": time.time() - started,
        "output": proc.stdout,
    }


def trim_result_output(result: dict[str, Any], max_chars: int) -> None:
    output = result.get("output")
    if not isinstance(output, str) or max_chars <= 0 or len(output) <= max_chars:
        return
    head_len = max_chars // 4
    tail_len = max_chars - head_len
    result["output_truncated"] = True
    result["output_original_chars"] = len(output)
    result["output_head_chars"] = head_len
    result["output_tail_chars"] = tail_len
    result["output"] = (
        output[:head_len]
        + f"\n\n... <truncated {len(output) - max_chars} chars; kept head={head_len} tail={tail_len}> ...\n\n"
        + output[-tail_len:]
    )


def load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def as_float_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def parse_zig_output(output: str) -> dict[str, Any]:
    def parse_float(value: str | None) -> float | None:
        if value is None:
            return None
        lowered = value.lower()
        if lowered in {"nan", "+nan", "-nan"}:
            return float("nan")
        if lowered in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if lowered in {"-inf", "-infinity"}:
            return float("-inf")
        return float(value)

    float_pattern = r"([-+]?(?:nan|inf|infinity|[0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?|\.[0-9]+(?:[eE][-+]?[0-9]+)?))"
    step = re.search(rf"loss={float_pattern}\s+grad_norm={float_pattern}\s+supervised_tok/s={float_pattern}", output)
    final = re.search(rf"final avg loss={float_pattern}", output)
    loaded = re.search(r"loaded\s+(\d+)\s+weights", output)
    parity_debug = extract_prefixed_json(output, "SPAN_PARITY_DEBUG ")
    preprocess_debug = extract_prefixed_json(output, "SPAN_PREPROCESS_DEBUG ")
    component_debug = extract_prefixed_json(output, "SPAN_COMPONENT_DEBUG ")
    total_component_debug = extract_prefixed_json(output, "GLINER2_TOTAL_LOSS_COMPONENT_DEBUG ")
    return {
        "step_loss": parse_float(step.group(1)) if step else None,
        "grad_norm": parse_float(step.group(2)) if step else None,
        "supervised_tok_per_s": parse_float(step.group(3)) if step else None,
        "final_avg_loss": parse_float(final.group(1)) if final else None,
        "loaded_weight_count": int(loaded.group(1)) if loaded else None,
        "span_parity_debug": parity_debug[-1] if parity_debug else None,
        "span_preprocess_debug": preprocess_debug[-1] if preprocess_debug else None,
        "span_component_debug": component_debug[-1] if component_debug else None,
        "gliner2_total_loss_components": total_component_debug[-1] if total_component_debug else None,
    }


def _trim_trailing(values: list[Any], pad: Any = 0) -> list[Any]:
    out = list(values or [])
    while out and out[-1] == pad:
        out.pop()
    return out


def _first_mismatch(path: str, left: Any, right: Any) -> dict[str, Any] | None:
    if isinstance(left, list) and isinstance(right, list):
        limit = min(len(left), len(right))
        for idx in range(limit):
            mismatch = _first_mismatch(f"{path}[{idx}]", left[idx], right[idx])
            if mismatch is not None:
                return mismatch
        if len(left) != len(right):
            return {"field": path, "index": limit, "python": len(left), "zig": len(right), "kind": "length"}
        return None
    if left != right:
        return {"field": path, "python": left, "zig": right}
    return None


def _zig_schema_special_indices(zig: dict[str, Any]) -> list[list[int]]:
    counts = zig.get("schema_special_counts") or []
    flat = zig.get("schema_special_positions") or []
    if not counts:
        return []
    max_schemas = len(counts)
    width = len(flat) // max_schemas if max_schemas else 0
    out: list[list[int]] = []
    for schema_idx, count in enumerate(counts):
        start = schema_idx * width
        out.append([int(x) for x in flat[start:start + int(count)] if int(x) >= 0])
    while out and not out[-1]:
        out.pop()
    return out


def _zig_schema_special_indices_for_sample(zig: dict[str, Any], sample_idx: int) -> list[list[int]]:
    counts_all = zig.get("schema_special_counts_all") or []
    flat_all = zig.get("schema_special_positions_all") or []
    max_schemas = int(zig.get("max_schemas") or len(zig.get("schema_special_counts") or []))
    max_schema_specials = int(zig.get("max_schema_specials") or 0)
    if not counts_all or not flat_all or max_schemas <= 0 or max_schema_specials <= 0:
        return _zig_schema_special_indices(zig) if sample_idx == 0 else []
    counts_start = sample_idx * max_schemas
    pos_start = sample_idx * max_schemas * max_schema_specials
    out: list[list[int]] = []
    for schema_idx in range(max_schemas):
        count_pos = counts_start + schema_idx
        if count_pos >= len(counts_all):
            break
        count = int(counts_all[count_pos])
        row_start = pos_start + schema_idx * max_schema_specials
        out.append([int(x) for x in flat_all[row_start:row_start + count] if int(x) >= 0])
    while out and not out[-1]:
        out.pop()
    return out


def _python_structure_positive_count(structure_labels: Any) -> int:
    count = 0
    for item in structure_labels or []:
        if not isinstance(item, list) or len(item) < 2:
            continue
        if not isinstance(item[1], list):
            continue
        for span_group in item[1] or []:
            if not isinstance(span_group, list):
                continue
            for field in span_group or []:
                if isinstance(field, list):
                    count += sum(1 for sub in field if sub not in (None, [-1, -1], (-1, -1)))
                elif field not in (None, [-1, -1], (-1, -1)):
                    count += 1
    return count


def _zig_sample_slice(zig: dict[str, Any], field: str, sample_idx: int, width: int, fallback_field: str | None = None) -> list[Any]:
    values = zig.get(field) or []
    if not values and fallback_field:
        values = zig.get(fallback_field) or []
    start = sample_idx * width
    end = start + width
    return values[start:end]


def _zig_task_types_for_sample(zig: dict[str, Any], sample_idx: int, count: int) -> list[str]:
    task_id_to_name = {1: "entities", 2: "json_structures", 3: "relations", 4: "classifications"}
    max_schemas = int(zig.get("max_schemas") or len(zig.get("task_type_ids") or []))
    ids = _zig_sample_slice(zig, "task_type_ids_all", sample_idx, max_schemas, "task_type_ids")
    return [task_id_to_name.get(int(x), "unknown") for x in ids[:count]]


def _compare_preprocess_sample(py: dict[str, Any], zig: dict[str, Any], sample_idx: int) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    max_length = int(zig.get("max_length") or len(zig.get("input_ids") or []))
    max_words = int(zig.get("max_words_per_sample") or len(zig.get("first_token_positions") or []))
    max_spans = int(zig.get("max_spans") or 0)
    entity_types = int(zig.get("num_entity_types") or 0)
    word_count = int(py.get("text_word_counts") or 0)
    py_task_types = py.get("task_types", [])

    comparisons = {
        f"samples[{sample_idx}].input_ids": (
            _trim_trailing(py.get("input_ids", []), 0),
            _trim_trailing(_zig_sample_slice(zig, "input_ids_all", sample_idx, max_length, "input_ids"), 0),
        ),
        f"samples[{sample_idx}].attention_mask": (
            _trim_trailing(py.get("attention_mask", []), 0),
            _trim_trailing(_zig_sample_slice(zig, "attention_mask_all", sample_idx, max_length, "attention_mask"), 0),
        ),
        f"samples[{sample_idx}].text_word_indices": (
            py.get("text_word_indices", []),
            _zig_sample_slice(zig, "first_token_positions_all", sample_idx, max_words, "first_token_positions")[:word_count],
        ),
        f"samples[{sample_idx}].schema_special_indices": (
            py.get("schema_special_indices", []),
            _zig_schema_special_indices_for_sample(zig, sample_idx),
        ),
        f"samples[{sample_idx}].task_types": (
            py_task_types,
            _zig_task_types_for_sample(zig, sample_idx, len(py_task_types)),
        ),
    }
    for field, (left, right) in comparisons.items():
        mismatch = _first_mismatch(field, left, right)
        if mismatch is not None:
            mismatches.append(mismatch)

    py_positive = _python_structure_positive_count(py.get("structure_labels"))
    span_label_width = max_spans * entity_types
    zig_labels = _zig_sample_slice(zig, "span_labels_all", sample_idx, span_label_width, "span_labels")
    zig_positive = int(round(sum(float(x) for x in zig_labels)))
    if py_positive != zig_positive:
        mismatches.append({ "field": f"samples[{sample_idx}].structure_positive_count", "python": py_positive, "zig": zig_positive })
    return mismatches


def compare_preprocess_debug(py: dict[str, Any] | None, zig: dict[str, Any] | None) -> tuple[bool, list[dict[str, Any]]]:
    mismatches: list[dict[str, Any]] = []
    if not py or not zig:
        return False, [{"field": "preprocess_debug", "python": bool(py), "zig": bool(zig), "kind": "missing"}]

    task_id_to_name = {1: "entities", 2: "json_structures", 3: "relations", 4: "classifications"}
    comparisons = {
        "input_ids": (_trim_trailing(py.get("input_ids", []), 0), _trim_trailing(zig.get("input_ids", []), 0)),
        "attention_mask": (_trim_trailing(py.get("attention_mask", []), 0), _trim_trailing(zig.get("attention_mask", []), 0)),
        "text_word_indices": (py.get("text_word_indices", []), (zig.get("first_token_positions", []) or [])[: len(py.get("text_word_indices", []))]),
        "schema_special_indices": (py.get("schema_special_indices", []), _zig_schema_special_indices(zig)),
        "task_types": (py.get("task_types", []), [task_id_to_name.get(int(x), "unknown") for x in (zig.get("task_type_ids") or [])[: len(py.get("task_types", []))]]),
    }
    for field, (left, right) in comparisons.items():
        mismatch = _first_mismatch(field, left, right)
        if mismatch is not None:
            mismatches.append(mismatch)

    py_positive = _python_structure_positive_count(py.get("structure_labels"))
    zig_labels = zig.get("span_labels") or []
    zig_first_sample_width = int(zig.get("max_spans") or 0) * int(zig.get("num_entity_types") or 0)
    if zig_first_sample_width <= 0:
        zig_first_sample_width = len(zig_labels)
    zig_positive = int(round(sum(float(x) for x in zig_labels[:zig_first_sample_width])))
    if py_positive != zig_positive:
        mismatches.append({"field": "structure_positive_count", "python": py_positive, "zig": zig_positive})

    return not mismatches, mismatches[:10]


def compare_preprocess_debug_samples(py_samples: list[dict[str, Any]] | None, zig: dict[str, Any] | None) -> tuple[bool, list[dict[str, Any]]]:
    if not py_samples:
        return compare_preprocess_debug(None, zig)
    if not zig:
        return False, [{"field": "preprocess_debug", "python": True, "zig": False, "kind": "missing"}]
    mismatches: list[dict[str, Any]] = []
    for fallback_idx, sample in enumerate(py_samples):
        sample_idx = int(sample.get("sample_idx", fallback_idx))
        mismatches.extend(_compare_preprocess_sample(sample, zig, sample_idx))
        if len(mismatches) >= 10:
            break
    return not mismatches, mismatches[:10]


def compare_component_losses(py: dict[str, Any] | None, zig: dict[str, Any] | None, tolerance: float) -> tuple[bool, dict[str, Any]]:
    fields = ["classification_loss", "structure_loss", "count_loss", "total_loss"]
    if not py or not zig:
        return False, {"missing": {"python": bool(py), "zig": bool(zig)}}
    deltas: dict[str, Any] = {}
    ok = True
    for field in fields:
        pv = py.get(field)
        zv = zig.get(field)
        if pv is None or zv is None:
            deltas[field] = {"python": pv, "zig": zv, "delta": None, "ok": False}
            ok = False
            continue
        if not finite_number(pv) or not finite_number(zv):
            deltas[field] = {"python": pv, "zig": zv, "delta": None, "ok": False, "reason": "non_finite"}
            ok = False
            continue
        delta = float(zv) - float(pv)
        field_ok = abs(delta) <= tolerance
        deltas[field] = {"python": float(pv), "zig": float(zv), "delta": delta, "ok": field_ok}
        ok = ok and field_ok
    return ok, deltas


def summarize_preprocess_tasks(samples: list[dict[str, Any]] | None) -> dict[str, Any]:
    task_counts: dict[str, int] = {}
    sample_summaries: list[dict[str, Any]] = []
    for fallback_idx, sample in enumerate(samples or []):
        sample_idx = int(sample.get("sample_idx", fallback_idx))
        local_counts: dict[str, int] = {}
        for task_type in sample.get("task_types", []) or []:
            name = str(task_type)
            task_counts[name] = task_counts.get(name, 0) + 1
            local_counts[name] = local_counts.get(name, 0) + 1
        sample_summaries.append({
            "sample_idx": sample_idx,
            "task_counts": local_counts,
            "has_non_entity_task": any(k != "entities" for k in local_counts),
            "structure_positive_count": _python_structure_positive_count(sample.get("structure_labels")),
        })
    return {
        "sample_count": len(sample_summaries),
        "task_counts": task_counts,
        "sample_summaries": sample_summaries,
        "non_entity_task_count": sum(v for k, v in task_counts.items() if k != "entities"),
    }


def summarize_component_deltas(deltas: dict[str, Any]) -> dict[str, Any]:
    failing: list[dict[str, Any]] = []
    for name, payload in deltas.items():
        if not isinstance(payload, dict) or payload.get("ok") is True:
            continue
        delta = payload.get("delta")
        failing.append({
            "component": name,
            "python": payload.get("python"),
            "zig": payload.get("zig"),
            "delta": delta,
            "abs_delta": abs(float(delta)) if delta is not None else None,
        })
    failing.sort(key=lambda item: item["abs_delta"] if item["abs_delta"] is not None else -1.0, reverse=True)
    return {
        "failing_components": failing,
        "largest_failing_component": failing[0]["component"] if failing else None,
    }


def summarize_metal_readiness(args: argparse.Namespace, report: dict[str, Any], zig_step_rows: list[dict[str, Any]], zig_manifest: dict[str, Any]) -> dict[str, Any] | None:
    if args.zig_backend != "metal" and not args.zig_build_metal:
        return None

    optimizer_backends = sorted({
        str(row.get("optimizer_backend"))
        for row in zig_step_rows
        if row.get("optimizer_backend") is not None
    })
    max_device_resident_transfer_count = max(
        [int(row.get("device_resident_transfer_count") or 0) for row in zig_step_rows] or [0]
    )
    max_device_trainable_bytes = max(
        [int(row.get("device_trainable_bytes") or 0) for row in zig_step_rows] or [0]
    )
    total_command_dispatches = sum(
        int(row.get("graph_executor_command_dispatches") or 0) for row in zig_step_rows
    )
    total_planned_dispatches = sum(
        int(row.get("graph_executor_planned_dispatches") or 0) for row in zig_step_rows
    )
    total_interpreter_fallbacks = sum(
        int(row.get("graph_executor_interpreter_fallbacks") or 0) for row in zig_step_rows
    )
    total_host_outputs = sum(
        int(row.get("graph_executor_host_outputs") or 0) for row in zig_step_rows
    )
    manifest_backend = str(zig_manifest.get("backend") or "")
    manifest_objective = str(zig_manifest.get("objective") or "")
    zig_metrics = report.get("zig", {}).get("metrics", {})
    checks = {
        "zig_returncode_ok": report.get("zig", {}).get("returncode") == 0,
        "manifest_backend_is_metal": manifest_backend.lower() == "metal",
        "manifest_objective_matches_request": manifest_objective == args.zig_objective,
        "finite_step_loss": finite_number(zig_metrics.get("step_loss")) or finite_number(zig_metrics.get("final_avg_loss")),
        "finite_grad_norm": finite_number(zig_metrics.get("grad_norm")) if zig_metrics.get("grad_norm") is not None else True,
        "optimizer_backend_is_metal": optimizer_backends == ["metal"] if optimizer_backends else False,
        "device_resident_transfers_zero": max_device_resident_transfer_count == 0,
        "device_trainables_resident": max_device_trainable_bytes > 0,
    }
    warnings: list[str] = []
    if total_command_dispatches == 0 and total_planned_dispatches == 0:
        warnings.append("Metal run reported no graph command/planned dispatches; check for interpreter-only execution")
    if total_interpreter_fallbacks > 0:
        warnings.append(f"Metal run reported {total_interpreter_fallbacks} interpreter fallbacks")
    if total_host_outputs > 0:
        warnings.append(f"Metal run reported {total_host_outputs} host outputs")
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "warnings": warnings,
        "manifest_backend": manifest_backend or None,
        "manifest_objective": manifest_objective or None,
        "optimizer_backends": optimizer_backends,
        "max_device_resident_transfer_count": max_device_resident_transfer_count,
        "max_device_trainable_bytes": max_device_trainable_bytes,
        "total_graph_command_dispatches": total_command_dispatches,
        "total_graph_planned_dispatches": total_planned_dispatches,
        "total_graph_interpreter_fallbacks": total_interpreter_fallbacks,
        "total_graph_host_outputs": total_host_outputs,
    }


def parse_op_stat_items(payload: str) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for item in payload.split(","):
        item = item.strip()
        if not item or ":count=" not in item:
            continue
        parts = item.split(":")
        name = parts[0]
        values: dict[str, float] = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, raw_value = part.split("=", 1)
            try:
                values[key] = float(raw_value)
            except ValueError:
                pass
        if values:
            stats[name] = values
    return stats


def parse_zig_op_stats(output: str) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    prefixes = {
        "metal_partition_command_ops:": "command_ops",
        "metal_partition_fallback_ops:": "fallback_ops",
        "metal_partition_host_output_ops:": "host_output_ops",
    }
    for line in output.splitlines():
        for prefix, key in prefixes.items():
            if line.startswith(prefix):
                parsed[key] = parse_op_stat_items(line[len(prefix):].strip())
    return parsed


def parse_bool_token(value: str) -> bool | None:
    if value == "true":
        return True
    if value == "false":
        return False
    return None


def parse_dot_shape_items(payload: str) -> list[dict[str, Any]]:
    shapes: list[dict[str, Any]] = []
    shape_re = re.compile(
        r"(?P<lhs0>-?\d+)x(?P<lhs1>-?\d+)\*(?P<rhs0>-?\d+)x(?P<rhs1>-?\d+)->(?P<out0>-?\d+)x(?P<out1>-?\d+)"
    )
    for item in payload.split(","):
        item = item.strip()
        if not item or item == "none":
            continue
        shape_part, _, stat_payload = item.partition(":")
        match = shape_re.fullmatch(shape_part)
        if not match:
            continue
        groups = match.groupdict()
        values: dict[str, str] = {}
        for part in stat_payload.split(":"):
            if "=" not in part:
                continue
            key, raw_value = part.split("=", 1)
            values[key] = raw_value
        shape: dict[str, Any] = {
            "lhs": [int(groups["lhs0"]), int(groups["lhs1"])],
            "rhs": [int(groups["rhs0"]), int(groups["rhs1"])],
            "out": [int(groups["out0"]), int(groups["out1"])],
        }
        for key in ("count",):
            if key in values:
                shape[key] = int(values[key])
        for key in ("total_ms", "avg_ms"):
            if key in values:
                shape[key] = float(values[key])
        for key in ("phase", "family", "pos", "node"):
            if key in values:
                shape[key] = values[key]
        if "lhs" in values:
            shape["lhs_op"] = values["lhs"]
        if "rhs" in values:
            shape["rhs_op"] = values["rhs"]
        for key in ("rhs_transpose", "rhs_parameter", "rhs_lora", "raw_linear"):
            if key in values:
                shape[key] = parse_bool_token(values[key])
        shapes.append(shape)
    return shapes


def parse_zig_op_runs(output: str) -> dict[str, Any]:
    dot_shapes: list[dict[str, Any]] = []
    for line in output.splitlines():
        prefixes = ("metal_partition_command_dot_shapes:", "metal_partition_dot_shapes:")
        prefix = next((p for p in prefixes if line.startswith(p)), None)
        if prefix is None:
            continue
        if prefix == "metal_partition_command_dot_shapes:":
            dot_shapes.extend(parse_dot_shape_items(line[len(prefix):].strip()))
            continue
        top_index = line.find(" top=")
        if top_index < 0:
            continue
        dot_shapes.extend(parse_dot_shape_items(line[top_index + len(" top="):]))
    dot_shapes.sort(key=lambda item: (item.get("total_ms", 0), item.get("count", 0)), reverse=True)
    return {
        "dot_shapes": dot_shapes,
        "top_dot_shapes": dot_shapes[:16],
    }


def extract_prefixed_json(output: str, prefix: str) -> list[dict[str, Any]]:
    def normalize_nonfinite_constants(payload: str) -> str:
        payload = re.sub(r'(?<![A-Za-z0-9_+\-.])-nan(?![A-Za-z0-9_+\-.])', 'NaN', payload, flags=re.IGNORECASE)
        payload = re.sub(r'(?<![A-Za-z0-9_+\-.])\+?nan(?![A-Za-z0-9_+\-.])', 'NaN', payload, flags=re.IGNORECASE)
        payload = re.sub(r'(?<![A-Za-z0-9_+\-.])-inf(?:inity)?(?![A-Za-z0-9_+\-.])', '-Infinity', payload, flags=re.IGNORECASE)
        payload = re.sub(r'(?<![A-Za-z0-9_+\-.])\+?inf(?:inity)?(?![A-Za-z0-9_+\-.])', 'Infinity', payload, flags=re.IGNORECASE)
        return payload

    payloads: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            payload = normalize_nonfinite_constants(line[len(prefix):])
            payloads.append(json.loads(payload))
        except json.JSONDecodeError:
            pass
    return payloads


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    return rows


def python_training_script() -> str:
    return r'''
import argparse, inspect, json, math, os, pathlib, time, types
import torch
import torch.nn.functional as F
import gliner2.model as gliner2_model
from gliner2.model import Extractor
from gliner2.training.lora import save_lora_adapter
from gliner2.training.trainer import ExtractorCollator, TrainingConfig, GLiNER2Trainer
from transformers import AutoConfig

p = argparse.ArgumentParser()
p.add_argument("--model-dir", required=True)
p.add_argument("--train-data", required=True)
p.add_argument("--out-dir", required=True)
p.add_argument("--steps", type=int, required=True)
p.add_argument("--batch-size", type=int, required=True)
p.add_argument("--seq-len", type=int, required=True)
p.add_argument("--max-span-width", type=int, required=True)
p.add_argument("--learning-rate", type=float, required=True)
p.add_argument("--weight-decay", type=float, required=True)
p.add_argument("--lora-rank", type=int, required=True)
p.add_argument("--lora-alpha", type=float, required=True)
p.add_argument("--lora-dropout", type=float, required=True)
p.add_argument("--lora-targets", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--span-negative-mask-rate", type=float, required=True)
p.add_argument("--disable-model-dropout", action="store_true")
p.add_argument("--dump-parity", action="store_true")
p.add_argument("--no-train-shuffle", action="store_true")
args = p.parse_args()

out = pathlib.Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)

_extractor_config_from_pretrained = Extractor.config_class.from_pretrained
def _local_file_aware_extractor_config(cls, path_or_repo_id, *cfg_args, **cfg_kwargs):
    if isinstance(path_or_repo_id, (str, os.PathLike)) and os.path.isfile(path_or_repo_id):
        return cls.from_json_file(os.fspath(path_or_repo_id))
    return _extractor_config_from_pretrained(path_or_repo_id, *cfg_args, **cfg_kwargs)
Extractor.config_class.from_pretrained = classmethod(_local_file_aware_extractor_config)

_auto_config_from_pretrained = AutoConfig.from_pretrained
def _local_file_aware_auto_config(path_or_repo_id, *cfg_args, **cfg_kwargs):
    if isinstance(path_or_repo_id, (str, os.PathLike)) and os.path.isfile(path_or_repo_id):
        return _auto_config_from_pretrained(os.path.dirname(os.fspath(path_or_repo_id)), *cfg_args, **cfg_kwargs)
    return _auto_config_from_pretrained(path_or_repo_id, *cfg_args, **cfg_kwargs)
AutoConfig.from_pretrained = _local_file_aware_auto_config
gliner2_model.AutoConfig.from_pretrained = _local_file_aware_auto_config

started = time.time()
model = Extractor.from_pretrained(args.model_dir, map_location="cpu")
model.max_width = args.max_span_width
model.config.max_width = args.max_span_width
if hasattr(model, "span_rep") and hasattr(model.span_rep, "span_rep_layer") and hasattr(model.span_rep.span_rep_layer, "max_width"):
    model.span_rep.span_rep_layer.max_width = args.max_span_width
disabled_dropout_modules = 0
if args.disable_model_dropout:
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
            disabled_dropout_modules += 1
        elif isinstance(module, torch.nn.MultiheadAttention) and module.dropout != 0.0:
            module.dropout = 0.0
            disabled_dropout_modules += 1
config_kwargs = {
    "output_dir": str(out),
    "experiment_name": "gliner2_python_zig_smoke",
    "num_epochs": 1,
    "max_steps": args.steps,
    "batch_size": args.batch_size,
    "eval_batch_size": args.batch_size,
    "gradient_accumulation_steps": 1,
    "encoder_lr": args.learning_rate,
    "task_lr": args.learning_rate,
    "weight_decay": args.weight_decay,
    "max_grad_norm": 1.0,
    "scheduler_type": "constant",
    "warmup_steps": 0,
    "warmup_ratio": 0.0,
    "fp16": False,
    "bf16": False,
    "eval_strategy": "no",
    "save_total_limit": 1,
    "save_best": False,
    "logging_steps": 1,
    "logging_first_step": True,
    "report_to_wandb": False,
    "early_stopping": False,
    "num_workers": 0,
    "pin_memory": False,
    "seed": args.seed,
    "deterministic": True,
    "max_train_samples": args.steps * args.batch_size,
    "use_lora": True,
    "lora_r": args.lora_rank,
    "lora_alpha": args.lora_alpha,
    "lora_dropout": args.lora_dropout,
    "lora_target_modules": args.lora_targets.split(","),
    "save_adapter_only": True,
}
supported_config_args = set(inspect.signature(TrainingConfig).parameters)
cfg = TrainingConfig(**{k: v for k, v in config_kwargs.items() if k in supported_config_args})
trainer = GLiNER2Trainer(model, cfg)
initial_adapter_dir = out / "initial_adapter"
save_lora_adapter(trainer.model, initial_adapter_dir)
sampling = trainer.processor.sampling_config
sampling.remove_json_structure_prob = 0.0
sampling.shuffle_json_fields = False
sampling.remove_json_field_prob = 0.0
sampling.synthetic_entity_label_prob = 0.0
sampling.shuffle_entities = False
sampling.remove_entity_prob = 0.0
sampling.remove_entities_prob = 0.0
sampling.remove_relations_prob = 0.0
sampling.swap_head_tail_prob = 0.0
sampling.remove_classification_prob = 0.0
sampling.shuffle_classification_labels = False
sampling.remove_classification_label_prob = 0.0
sampling.synthetic_label_prob = 0.0
sampling.include_true_label_prob = 1.0
if args.no_train_shuffle:
    def _process_json_structures_ordered(self, schema, schemas, labels, types, sampling):
        if "json_structures" not in schema:
            return
        json_descs = schema.get("json_descriptions", {})
        groups = {}
        for item in schema["json_structures"]:
            for parent, fields in item.items():
                groups.setdefault(parent, []).append(fields)
        for parent, occurrences in groups.items():
            chosen = []
            seen_fields = set()
            for occ in occurrences:
                for field in occ.keys():
                    if field not in seen_fields:
                        seen_fields.add(field)
                        chosen.append(field)
            if not chosen:
                continue
            spans = []
            for occ in occurrences:
                spans.append([occ.get(field) for field in chosen])
            uniq = []
            seen_spans = set()
            for span in spans:
                key = tuple(tuple(item) if isinstance(item, list) else item for item in span)
                if key not in seen_spans:
                    seen_spans.add(key)
                    uniq.append(span)
            if all(all(cell is None or cell == "" for cell in span) for span in uniq):
                count = 0
                uniq = []
            else:
                count = len(uniq)
            labels.append([count, uniq])
            schemas.append(self._transform_schema(
                parent,
                chosen,
                self.C_TOKEN,
                label_descriptions=json_descs.get(parent, {}),
                example_mode="descriptions" if json_descs.get(parent, {}) else "none",
            ))
            types.append("json_structures")
    trainer.processor._process_json_structures = types.MethodType(_process_json_structures_ordered, trainer.processor)
python_step_timings = []
original_create_dataloader = trainer._create_dataloader
original_prepare_data = trainer._prepare_data

class TimedTrainingLoader:
    def __init__(self, inner, timings):
        self.inner = inner
        self.timings = timings

    def __len__(self):
        return len(self.inner)

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __iter__(self):
        for local_step, batch in enumerate(self.inner):
            started = time.perf_counter()
            try:
                yield batch
            finally:
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                try:
                    batch_size = len(batch)
                except TypeError:
                    batch_size = None
                self.timings.append({
                    "step": len(self.timings) + 1,
                    "loader_step": local_step + 1,
                    "batch_size": batch_size,
                    "step_wall_ms": elapsed_ms,
                })

def create_dataloader_with_timing(*call_args, **call_kwargs):
    is_training = call_kwargs.get("is_training")
    if is_training is None and len(call_args) >= 4:
        is_training = call_args[3]
    if args.no_train_shuffle and is_training:
        if "shuffle" in call_kwargs:
            call_kwargs["shuffle"] = False
        elif len(call_args) >= 3:
            call_args = (*call_args[:2], False, *call_args[3:])
    loader = original_create_dataloader(*call_args, **call_kwargs)
    if is_training:
        return TimedTrainingLoader(loader, python_step_timings)
    return loader

def prepare_data_no_shuffle(data, is_train=True):
    if args.no_train_shuffle and is_train:
        return original_prepare_data(data, is_train=False)
    return original_prepare_data(data, is_train=is_train)

trainer._prepare_data = prepare_data_no_shuffle
trainer._create_dataloader = create_dataloader_with_timing
parity_debug = []
preprocess_debug = []
preprocess_debug_samples = []
component_debug = []
sample_loss_debug = []
if args.dump_parity:
    def _finite(value):
        value = float(value)
        if math.isfinite(value):
            return value
        return None

    def _mean_or_zero(tensor):
        if tensor.numel() == 0:
            return 0.0
        return _finite(tensor.float().mean().detach().cpu().item())

    def _token_piece_count(token):
        tokenizer = getattr(trainer.processor, "tokenizer", None)
        if tokenizer is None:
            return 1
        try:
            pieces = tokenizer.tokenize(token)
            return max(1, len(pieces))
        except Exception:
            return 1

    def _derive_schema_special_indices(schema_tokens_list):
        special_tokens = {"[P]", "[C]", "[E]", "[R]", "[L]"}
        out = []
        pos = 0
        for schema_idx, schema_tokens in enumerate(schema_tokens_list or []):
            if schema_idx > 0:
                pos += 1
            schema_out = []
            for token in schema_tokens or []:
                if token in special_tokens:
                    schema_out.append(pos)
                pos += _token_piece_count(token)
            out.append(schema_out)
        return out

    def _derive_text_word_indices(schema_tokens_list, text_tokens):
        pos = 0
        for schema_idx, schema_tokens in enumerate(schema_tokens_list or []):
            if schema_idx > 0:
                pos += 1
            for token in schema_tokens or []:
                pos += _token_piece_count(token)
        if schema_tokens_list:
            pos += 1
        out = []
        for token in text_tokens or []:
            out.append(pos)
            pos += _token_piece_count(token)
        return out

    original_collator_call = ExtractorCollator.__call__
    def _preprocess_sample_debug(out, sample_idx, text_tokens, schema_tokens_list, text_word_indices, text_word_counts, schema_special_indices, structure_labels, task_types):
        sample_text_tokens = text_tokens[sample_idx] if text_tokens and sample_idx < len(text_tokens) else []
        sample_schema_tokens = schema_tokens_list[sample_idx] if schema_tokens_list and sample_idx < len(schema_tokens_list) else []
        word_count = int(text_word_counts[sample_idx]) if text_word_counts and sample_idx < len(text_word_counts) else len(sample_text_tokens)
        derived_text_word_indices = _derive_text_word_indices(sample_schema_tokens, sample_text_tokens)
        derived_schema_special_indices = _derive_schema_special_indices(sample_schema_tokens)
        return {
            "sample_idx": sample_idx,
            "input_ids": out.input_ids[sample_idx].detach().cpu().tolist(),
            "attention_mask": out.attention_mask[sample_idx].detach().cpu().tolist(),
            "text_word_indices": (
                text_word_indices[sample_idx, :word_count].detach().cpu().tolist()
                if text_word_indices is not None and text_word_counts else derived_text_word_indices
            ),
            "text_word_counts": word_count,
            "schema_special_indices": schema_special_indices[sample_idx] if schema_special_indices else derived_schema_special_indices,
            "text_tokens": sample_text_tokens,
            "schema_tokens_list": sample_schema_tokens,
            "structure_labels": structure_labels[sample_idx] if structure_labels else [],
            "task_types": task_types[sample_idx] if task_types else [],
        }

    def collator_call_with_debug(self, batch):
        original_is_training = self.is_training
        if args.no_train_shuffle:
            self.is_training = False
        try:
            out = original_collator_call(self, batch)
        finally:
            self.is_training = original_is_training
        if not preprocess_debug and getattr(out, "input_ids", None) is not None and len(out) > 0:
            text_word_indices = getattr(out, "text_word_indices", None)
            text_word_counts = getattr(out, "text_word_counts", None)
            schema_special_indices = getattr(out, "schema_special_indices", None)
            text_tokens = getattr(out, "text_tokens", None)
            schema_tokens_list = getattr(out, "schema_tokens_list", None)
            structure_labels = getattr(out, "structure_labels", None)
            task_types = getattr(out, "task_types", None)
            sample_count = int(out.input_ids.shape[0])
            for sample_idx in range(sample_count):
                preprocess_debug_samples.append(_preprocess_sample_debug(
                    out,
                    sample_idx,
                    text_tokens,
                    schema_tokens_list,
                    text_word_indices,
                    text_word_counts,
                    schema_special_indices,
                    structure_labels,
                    task_types,
                ))
            if preprocess_debug_samples:
                first = dict(preprocess_debug_samples[0])
                first["text_word_counts"] = [item["text_word_counts"] for item in preprocess_debug_samples]
                first.pop("sample_idx", None)
                preprocess_debug.append(first)
        return out

    ExtractorCollator.__call__ = collator_call_with_debug
    original_compute_span_rep = trainer.model.compute_span_rep
    span_hidden_debug = []

    def compute_span_rep_with_hidden_debug(self, token_embeddings):
        out = original_compute_span_rep(token_embeddings)
        if not span_hidden_debug:
            span_hidden_debug.append(token_embeddings.detach())
        return out

    trainer.model.compute_span_rep = types.MethodType(compute_span_rep_with_hidden_debug, trainer.model)

    def compute_struct_loss_with_debug(self, span_rep, schema_emb, structure, span_mask, masking_rate=args.span_negative_mask_rate):
        gold_count = min(structure[0], 19)
        struct_proj = self.count_embed(schema_emb[1:], gold_count)
        scores = torch.einsum('lkd,bpd->bplk', span_rep, struct_proj)
        labs = torch.zeros_like(scores)

        for i in range(gold_count):
            gold_spans = structure[1][i]
            for k, span in enumerate(gold_spans):
                if span is None or span == (-1, -1):
                    continue
                if isinstance(span, tuple):
                    start, end = span
                    width = end - start
                    if 0 <= start < scores.shape[2] and 0 <= width < scores.shape[3]:
                        labs[i, k, start, width] = 1
                elif isinstance(span, list):
                    for sub in span:
                        if sub is None or sub == (-1, -1):
                            continue
                        start, end = sub
                        width = end - start
                        if 0 <= start < scores.shape[2] and 0 <= width < scores.shape[3]:
                            labs[i, k, start, width] = 1

        if masking_rate > 0.0 and self.training:
            negative = (labs == 0)
            random_mask = torch.rand_like(scores) < masking_rate
            to_mask = negative & random_mask
            loss_mask = (~to_mask).float()
        else:
            loss_mask = torch.ones_like(scores)

        bce = F.binary_cross_entropy_with_logits(scores, labs, reduction="none")
        masked_bce = bce * loss_mask
        span_valid = (~span_mask[0]).float()
        final_mask = loss_mask.view(loss_mask.shape[0], loss_mask.shape[1], -1) * span_valid
        final_tensor = masked_bce.view(masked_bce.shape[0], masked_bce.shape[1], -1) * span_valid
        loss = final_tensor.sum()

        if not parity_debug:
            pos_scores = scores[labs > 0]
            neg_scores = scores[labs <= 0]
            final_bce = bce.view(bce.shape[0], bce.shape[1], -1) * final_mask
            labs_flat = labs.view(labs.shape[0], labs.shape[1], -1)
            final_pos = final_bce[labs_flat > 0]
            final_neg = final_bce[labs_flat <= 0]
            valid_negative = (labs_flat <= 0) & (final_mask > 0)
            top_valid_negative_logits = []
            if valid_negative.any():
                flat_scores = scores.view(scores.shape[0], scores.shape[1], -1)
                masked_final_bce = final_bce.masked_fill(~valid_negative, float("-inf"))
                top_k = min(5, int(valid_negative.sum().detach().cpu().item()))
                top_values, top_indices = torch.topk(masked_final_bce.flatten(), k=top_k)
                entity_count = scores.shape[1]
                span_width = scores.shape[3]
                flat_span_count = scores.shape[2] * scores.shape[3]
                for value, flat_index_tensor in zip(top_values, top_indices):
                    flat_index = int(flat_index_tensor.detach().cpu().item())
                    count_idx_top = flat_index // (entity_count * flat_span_count)
                    remainder = flat_index % (entity_count * flat_span_count)
                    entity_idx_top = remainder // flat_span_count
                    span_flat = remainder % flat_span_count
                    start_idx_top = span_flat // span_width
                    width_idx_top = span_flat % span_width
                    top_valid_negative_logits.append({
                        "count_index": count_idx_top,
                        "entity": entity_idx_top,
                        "row": start_idx_top * span_width + width_idx_top,
                        "start": start_idx_top,
                        "width": width_idx_top,
                        "logit": _finite(flat_scores[count_idx_top, entity_idx_top, span_flat].detach().cpu().item()),
                        "bce": _finite(value.detach().cpu().item()),
                    })
            positive_indices = (labs > 0).nonzero(as_tuple=False)
            if positive_indices.numel() > 0 and not component_debug:
                count_idx = int(positive_indices[0, 0].detach().cpu().item())
                entity_idx = int(positive_indices[0, 1].detach().cpu().item())
                start_idx = int(positive_indices[0, 2].detach().cpu().item())
                width_idx = int(positive_indices[0, 3].detach().cpu().item())
                span_vec = span_rep[start_idx, width_idx]
                schema_vec = schema_emb[1 + entity_idx]
                projection_vec = struct_proj[count_idx, entity_idx]
                count_gru_state_vec = None
                count_state_vec = None
                count_in_project_vec = None
                count_layer0_vec = None
                count_layer1_vec = None
                count_out0_vec = None
                count_out2_vec = None
                with torch.no_grad():
                    count_embed = self.count_embed
                    label_embs = schema_emb[1:]
                    max_count = int(getattr(count_embed, "max_count", 20))
                    debug_gold_count = min(int(gold_count), max_count)
                    if debug_gold_count > 0:
                        debug_count_idx = torch.arange(max_count, device=label_embs.device)[:debug_gold_count]
                        debug_pos_seq = count_embed.pos_embedding(debug_count_idx)
                        debug_pos_seq = debug_pos_seq.unsqueeze(1).expand(-1, label_embs.shape[0], -1)
                        debug_h0 = label_embs.unsqueeze(0)
                        debug_gru_state, _ = count_embed.gru(debug_pos_seq, debug_h0)
                        debug_count_state = debug_gru_state + debug_h0.expand_as(debug_gru_state)
                        debug_transformer = count_embed.transformer
                        debug_in_project = debug_transformer.in_projector(debug_count_state)
                        debug_layer = debug_in_project
                        debug_layer0 = None
                        debug_layer1 = None
                        for layer_idx, layer in enumerate(debug_transformer.transformer.layers):
                            debug_layer = layer(debug_layer)
                            if layer_idx == 0:
                                debug_layer0 = debug_layer
                            elif layer_idx == 1:
                                debug_layer1 = debug_layer
                        debug_joined = torch.cat([debug_layer, debug_count_state], dim=-1)
                        debug_out0 = torch.relu(debug_transformer.out_projector[0](debug_joined))
                        debug_out2 = torch.relu(debug_transformer.out_projector[2](debug_out0))
                        count_gru_state_vec = debug_gru_state[count_idx, entity_idx]
                        count_state_vec = debug_count_state[count_idx, entity_idx]
                        count_in_project_vec = debug_in_project[count_idx, entity_idx]
                        count_layer0_vec = debug_layer0[count_idx, entity_idx] if debug_layer0 is not None else debug_layer[count_idx, entity_idx]
                        count_layer1_vec = debug_layer1[count_idx, entity_idx] if debug_layer1 is not None else debug_layer[count_idx, entity_idx]
                        count_out0_vec = debug_out0[count_idx, entity_idx]
                        count_out2_vec = debug_out2[count_idx, entity_idx]

                def _vec_norm(vec):
                    return _finite(vec.float().norm().detach().cpu().item()) if vec is not None else 0.0

                def _vec_mean(vec):
                    return _finite(vec.float().mean().detach().cpu().item()) if vec is not None else 0.0

                top_negative_debug = {}
                if top_valid_negative_logits:
                    neg = top_valid_negative_logits[0]
                    neg_count_idx = int(neg["count_index"])
                    neg_entity_idx = int(neg["entity"])
                    neg_start_idx = int(neg["start"])
                    neg_width_idx = int(neg["width"])
                    neg_end_idx = neg_start_idx + neg_width_idx
                    neg_span_vec = span_rep[neg_start_idx, neg_width_idx]
                    neg_start_hidden_vec = None
                    neg_end_hidden_vec = None
                    if span_hidden_debug:
                        token_embeddings = span_hidden_debug[0]
                        if 0 <= neg_start_idx < token_embeddings.shape[0]:
                            neg_start_hidden_vec = token_embeddings[neg_start_idx]
                        if 0 <= neg_end_idx < token_embeddings.shape[0]:
                            neg_end_hidden_vec = token_embeddings[neg_end_idx]
                    neg_schema_hidden_vec = schema_emb[1 + neg_entity_idx]
                    neg_projection_vec = struct_proj[neg_count_idx, neg_entity_idx]
                    top_negative_debug = {
                        "negative_row": int(neg["row"]),
                        "negative_entity": neg_entity_idx,
                        "negative_count_index": neg_count_idx,
                        "negative_start_index": neg_start_idx,
                        "negative_width_index": neg_width_idx,
                        "negative_end_index": neg_end_idx,
                        "negative_logit": _finite(neg["logit"]),
                        "negative_start_hidden_norm": _vec_norm(neg_start_hidden_vec),
                        "negative_end_hidden_norm": _vec_norm(neg_end_hidden_vec),
                        "negative_schema_hidden_norm": _vec_norm(neg_schema_hidden_vec),
                        "negative_span_rep_norm": _finite(neg_span_vec.float().norm().detach().cpu().item()),
                        "negative_schema_projection_norm": _finite(neg_projection_vec.float().norm().detach().cpu().item()),
                        "negative_projected_schema_dot": _finite(torch.dot(neg_span_vec.flatten().float(), neg_projection_vec.flatten().float()).detach().cpu().item()),
                        "negative_start_hidden_mean": _vec_mean(neg_start_hidden_vec),
                        "negative_end_hidden_mean": _vec_mean(neg_end_hidden_vec),
                        "negative_schema_hidden_mean": _vec_mean(neg_schema_hidden_vec),
                        "negative_span_rep_mean": _finite(neg_span_vec.float().mean().detach().cpu().item()),
                        "negative_schema_projection_mean": _finite(neg_projection_vec.float().mean().detach().cpu().item()),
                    }

                component_payload = {
                    "positive_row": start_idx * scores.shape[3] + width_idx,
                    "positive_entity": entity_idx,
                    "schema_row": entity_idx,
                    "count_index": count_idx,
                    "start_index": start_idx,
                    "width_index": width_idx,
                    "span_rep_norm": _finite(span_vec.float().norm().detach().cpu().item()),
                    "schema_hidden_norm": _finite(schema_vec.float().norm().detach().cpu().item()),
                    "count_gru_state_norm": _vec_norm(count_gru_state_vec),
                    "count_state_norm": _vec_norm(count_state_vec),
                    "count_in_project_norm": _vec_norm(count_in_project_vec),
                    "count_layer0_norm": _vec_norm(count_layer0_vec),
                    "count_layer1_norm": _vec_norm(count_layer1_vec),
                    "count_out0_norm": _vec_norm(count_out0_vec),
                    "count_out2_norm": _vec_norm(count_out2_vec),
                    "schema_projection_norm": _finite(projection_vec.float().norm().detach().cpu().item()),
                    "projected_schema_dot": _finite(torch.dot(span_vec.flatten().float(), projection_vec.flatten().float()).detach().cpu().item()),
                    "span_rep_mean": _finite(span_vec.float().mean().detach().cpu().item()),
                    "count_gru_state_mean": _vec_mean(count_gru_state_vec),
                    "count_state_mean": _vec_mean(count_state_vec),
                    "count_in_project_mean": _vec_mean(count_in_project_vec),
                    "count_layer0_mean": _vec_mean(count_layer0_vec),
                    "count_layer1_mean": _vec_mean(count_layer1_vec),
                    "count_out0_mean": _vec_mean(count_out0_vec),
                    "count_out2_mean": _vec_mean(count_out2_vec),
                    "schema_projection_mean": _finite(projection_vec.float().mean().detach().cpu().item()),
                }
                component_payload.update(top_negative_debug)
                component_debug.append(component_payload)
            parity_debug.append({
                "gold_count": int(gold_count),
                "scores_shape": list(scores.shape),
                "labs_sum": _finite(labs.sum().detach().cpu().item()),
                "loss_mask_sum": _finite(loss_mask.sum().detach().cpu().item()),
                "final_mask_sum": _finite(final_mask.sum().detach().cpu().item()),
                "valid_span_count": _finite(span_valid.sum().detach().cpu().item()),
                "scores_min": _finite(scores.min().detach().cpu().item()),
                "scores_max": _finite(scores.max().detach().cpu().item()),
                "scores_mean": _finite(scores.mean().detach().cpu().item()),
                "positive_scores_mean": _mean_or_zero(pos_scores),
                "negative_scores_mean": _mean_or_zero(neg_scores),
                "bce_unweighted_sum": _finite(bce.sum().detach().cpu().item()),
                "bce_negative_masked_sum": _finite(masked_bce.sum().detach().cpu().item()),
                "bce_final_positive_sum": _finite(final_pos.sum().detach().cpu().item()) if final_pos.numel() else 0.0,
                "bce_final_negative_sum": _finite(final_neg.sum().detach().cpu().item()) if final_neg.numel() else 0.0,
                "bce_final_sum": _finite(loss.detach().cpu().item()),
                "top_valid_negative_logits": top_valid_negative_logits,
                "masking_rate": float(masking_rate),
            })
        return loss

    Extractor.compute_struct_loss = compute_struct_loss_with_debug
    trainer.model.compute_struct_loss = types.MethodType(compute_struct_loss_with_debug, trainer.model)
    original_compute_sample_loss = trainer.model._compute_sample_loss
    def compute_sample_loss_with_debug(self, *sample_args, **sample_kwargs):
        out = original_compute_sample_loss(*sample_args, **sample_kwargs)
        if len(sample_loss_debug) < args.batch_size:
            sample_loss_debug.append({
                "classification_loss": _finite(out["classification"].detach().cpu().item()),
                "structure_loss": _finite(out["structure"].detach().cpu().item()),
                "count_loss": _finite(out["count"].detach().cpu().item()),
            })
        return out
    trainer.model._compute_sample_loss = types.MethodType(compute_sample_loss_with_debug, trainer.model)
result = trainer.train(train_data=args.train_data)
trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in trainer.model.parameters())
payload = {
    "elapsed_seconds": time.time() - started,
    "total_steps": result.get("total_steps"),
    "samples_per_second": result.get("samples_per_second"),
    "train_metrics_history": result.get("train_metrics_history", []),
    "step_timings": python_step_timings,
    "total_step_wall_ms": sum(item["step_wall_ms"] for item in python_step_timings),
    "avg_step_wall_ms": (
        sum(item["step_wall_ms"] for item in python_step_timings) / len(python_step_timings)
        if python_step_timings else None
    ),
    "trainable_parameters": trainable,
    "total_parameters": total,
    "torch_version": torch.__version__,
    "disabled_dropout_modules": disabled_dropout_modules,
    "initial_adapter_checkpoint": str(initial_adapter_dir / "adapter_weights.safetensors"),
    "span_parity_debug": parity_debug[0] if parity_debug else None,
    "span_preprocess_debug": preprocess_debug[0] if preprocess_debug else None,
    "span_preprocess_debug_samples": preprocess_debug_samples,
    "span_component_debug": component_debug[0] if component_debug else None,
    "gliner2_total_loss_components": (
        {
            "classification_loss": sum((item.get("classification_loss") or 0.0) for item in sample_loss_debug),
            "structure_loss": sum((item.get("structure_loss") or 0.0) for item in sample_loss_debug),
            "count_loss": sum((item.get("count_loss") or 0.0) for item in sample_loss_debug),
            "total_loss": sum((item.get("classification_loss") or 0.0) + (item.get("structure_loss") or 0.0) + (item.get("count_loss") or 0.0) for item in sample_loss_debug),
            "sample_count": len(sample_loss_debug),
        }
        if sample_loss_debug else None
    ),
}
(out / "comparison_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
print("PYTHON_GLINER2_COMPARISON " + json.dumps(payload, sort_keys=True))
'''


def run_python_side(args: argparse.Namespace, py_train_data: Path, out_dir: Path) -> dict[str, Any]:
    script = out_dir / "run_python_gliner2_train.py"
    script.write_text(python_training_script(), encoding="utf-8")
    python_model = str(args.python_model)
    if Path(python_model).exists():
        python_model = str(prepare_python_model_dir(Path(python_model), out_dir))
    cmd = [
        args.python_bin,
        str(script),
        "--model-dir", python_model,
        "--train-data", str(py_train_data),
        "--out-dir", str(out_dir / "python"),
        "--steps", str(args.steps),
        "--batch-size", str(args.batch_size),
        "--seq-len", str(args.seq_len),
        "--max-span-width", str(args.max_span_width),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--lora-targets", args.lora_targets,
        "--seed", str(args.seed),
        "--span-negative-mask-rate", str(args.span_negative_mask_rate),
    ]
    if args.disable_python_model_dropout:
        cmd.append("--disable-model-dropout")
    if args.dump_parity:
        cmd.append("--dump-parity")
    if args.dump_preprocess_parity:
        cmd.append("--no-train-shuffle")
    result = run_command(cmd, repo_root(), timeout=args.timeout_seconds, env={"PYTHONHASHSEED": "0"})
    metrics_path = out_dir / "python" / "comparison_metrics.json"
    result["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return result


def run_zig_side(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    zig_local_cache = out_dir / "zig-local-cache"
    zig_global_cache = out_dir / "zig-global-cache"
    if zig_local_cache.exists():
        shutil.rmtree(zig_local_cache)
    if zig_global_cache.exists():
        shutil.rmtree(zig_global_cache)
    zig_local_cache.mkdir(parents=True, exist_ok=True)
    zig_global_cache.mkdir(parents=True, exist_ok=True)
    enable_metal = args.zig_build_metal if args.zig_build_metal is not None else args.zig_backend == "metal"
    enable_mlx = args.zig_build_mlx if args.zig_build_mlx is not None else args.zig_backend == "mlx"
    cmd = [
        "zig", "build",
        "--cache-dir", str(zig_local_cache),
        "--global-cache-dir", str(zig_global_cache),
        f"-Dmlx={'true' if enable_mlx else 'false'}",
        "-Donnx=false",
        f"-Dmetal={'true' if enable_metal else 'false'}",
    ]
    if args.zig_optimize is not None:
        cmd.append(f"-Doptimize={args.zig_optimize}")
    cmd.extend([
        "train-gliner2-autodiff",
        "--",
        "--model-dir", str(args.model_dir),
        "--train-data", str(args.train_data),
        "--out-dir", str(out_dir / "zig"),
        "--epochs", "1",
        "--batch-size", str(args.batch_size),
        "--max-examples", str(args.steps * args.batch_size),
        "--seq-len", str(args.seq_len),
        "--learning-rate", str(args.learning_rate),
        "--weight-decay", str(args.weight_decay),
        "--backend", args.zig_backend,
        "--objective", args.zig_objective,
        "--max-span-width", str(args.max_span_width),
        "--span-loss", "bce",
        "--span-loss-reduction", args.span_loss_reduction,
        "--span-positive-weight", str(args.span_positive_weight),
        "--span-negative-weight", str(args.span_negative_weight),
        "--span-hard-negative-weight", str(args.span_hard_negative_weight),
        "--span-negative-mask-rate", str(args.span_negative_mask_rate),
        "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--lora-targets", args.lora_targets,
        "--seed", str(args.seed),
    ])
    if args.zig_objective != "gliner2-total-loss":
        cmd.extend([
            "--entity-types", args.entity_types,
            "--num-classes", str(len([x for x in args.entity_types.split(",") if x]) + 1),
        ])
    if args.zig_lora_only_trainables:
        cmd.append("--lora-only-trainables")
    initial_adapter_checkpoint = out_dir / "python" / "initial_adapter" / "adapter_weights.safetensors"
    if initial_adapter_checkpoint.exists():
        cmd.extend(["--initial-adapter-checkpoint", str(initial_adapter_checkpoint)])
    if args.dump_parity:
        cmd.append("--dump-span-parity")
    zig_env: dict[str, str] = {}
    if args.zig_backend == "metal" and args.zig_training_graph_executor:
        zig_env["TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR"] = "1"
    result = run_command(cmd, inference_dir(), timeout=args.timeout_seconds, env=zig_env)
    result["cache_fallback"] = False
    if result.get("returncode") != 0 and "manifest_create PermissionDenied" in result.get("output", ""):
        fallback_cmd = ["zig", "build", *cmd[6:]]
        fallback = run_command(fallback_cmd, inference_dir(), timeout=args.timeout_seconds, env=zig_env)
        fallback["cache_fallback"] = True
        fallback["cache_fallback_reason"] = "isolated Zig cache failed with manifest_create PermissionDenied; retried with repo-local cache"
        fallback["initial_isolated_cache_failure_tail"] = result.get("output", "")[-4000:]
        result = fallback
    result["metrics"] = parse_zig_output(result["output"])
    result["training_metrics"] = load_jsonl(out_dir / "zig" / "training_metrics.jsonl")
    result["training_manifest"] = load_json_file(out_dir / "zig" / "training_manifest.json")
    return result


def main() -> int:
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__ or ""),
    )
    p.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    p.add_argument("--python-model", default=DEFAULT_PYTHON_MODEL)
    p.add_argument("--train-data", type=Path, default=default_train_data())
    p.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    p.add_argument("--python-bin", default=DEFAULT_PYTHON)
    p.add_argument("--entity-types", default=DEFAULT_LABELS)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--max-span-width", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--span-loss-reduction", default="sum", choices=["mean", "sum"])
    p.add_argument("--span-positive-weight", type=float, default=1.0)
    p.add_argument("--span-negative-weight", type=float, default=1.0)
    p.add_argument("--span-hard-negative-weight", type=float, default=1.0)
    p.add_argument("--span-negative-mask-rate", type=float, default=0.5)
    p.add_argument(
        "--disable-python-model-dropout",
        action="store_true",
        help="Set Python nn.Dropout modules to p=0 for deterministic objective parity; LoRA dropout remains controlled by --lora-dropout",
    )
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-targets", default=LORA_TARGETS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--zig-backend", default="native", choices=["native", "metal", "mlx", "auto"])
    p.add_argument(
        "--zig-training-graph-executor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR for Zig Metal runs",
    )
    p.add_argument("--zig-objective", default="span-start", choices=["token", "span-start", "gliner2-total-loss"])
    p.add_argument(
        "--zig-lora-only-trainables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match upstream GLiNER2 LoRA training by freezing regular task-head params and optimizing only LoRA params",
    )
    p.add_argument("--zig-build-metal", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--zig-build-mlx", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--zig-optimize",
        choices=["Debug", "ReleaseSafe", "ReleaseFast", "ReleaseSmall"],
        default="ReleaseFast",
        help="Zig optimization mode for the training binary (default: ReleaseFast; pass Debug for instrumentation-heavy debugging)",
    )
    p.add_argument("--dump-parity", action="store_true", help="Collect first-batch span objective logits/label/mask stats from both implementations")
    p.add_argument("--dump-preprocess-parity", action="store_true", help="Collect first-batch preprocessing metadata from both implementations")
    p.add_argument(
        "--loss-parity-tolerance",
        type=float,
        default=1e-4,
        help="Maximum absolute Python/Zig loss delta for valid loss parity when both sides run",
    )
    p.add_argument(
        "--perf-target-only-python",
        action="store_true",
        help="Treat Python as a timing target only; report Python/Zig loss delta but mark loss parity invalid",
    )
    p.add_argument("--timeout-seconds", type=int, default=900)
    p.add_argument(
        "--max-command-output-chars",
        type=int,
        default=200_000,
        help="Bound each captured command output in comparison_report.json after metrics are parsed (0 keeps full output)",
    )
    p.add_argument("--skip-python", action="store_true")
    p.add_argument("--skip-zig", action="store_true")
    p.add_argument("--keep-out-dir", action="store_true")
    args = p.parse_args()
    if args.dump_preprocess_parity:
        args.dump_parity = True
    args.model_dir = args.model_dir.expanduser().resolve()
    args.train_data = args.train_data.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if Path(str(args.python_model)).exists():
        args.python_model = str(Path(str(args.python_model)).expanduser().resolve())

    if not args.keep_out_dir and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    allowed_labels = parse_label_csv(args.entity_types)
    source_task_summary = summarize_python_jsonl(args.train_data, allowed_labels)
    converted = convert_limited_to_python_jsonl(
        args.train_data,
        args.out_dir / "python_train.jsonl",
        args.steps * args.batch_size,
        allowed_labels,
    )
    report: dict[str, Any] = {
        "task": "gliner2_lora_python_zig_apples_to_apples",
        "config": {
            "model_dir": str(args.model_dir),
            "python_model": str(args.python_model),
            "train_data": str(args.train_data),
            "converted_python_train_data": converted,
            "source_train_data_summary": source_task_summary,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "learning_rate": args.learning_rate,
            "span_loss_reduction": args.span_loss_reduction,
            "span_positive_weight": args.span_positive_weight,
            "span_negative_weight": args.span_negative_weight,
            "span_hard_negative_weight": args.span_hard_negative_weight,
            "span_negative_mask_rate": args.span_negative_mask_rate,
            "disable_python_model_dropout": args.disable_python_model_dropout,
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_targets": args.lora_targets,
            "seed": args.seed,
            "zig_backend": args.zig_backend,
            "zig_training_graph_executor": args.zig_training_graph_executor,
            "zig_objective": args.zig_objective,
            "zig_lora_only_trainables": args.zig_lora_only_trainables,
            "zig_build_metal": args.zig_build_metal,
            "zig_build_mlx": args.zig_build_mlx,
            "zig_optimize": args.zig_optimize,
            "dump_parity": args.dump_parity,
            "dump_preprocess_parity": args.dump_preprocess_parity,
            "loss_parity_tolerance": args.loss_parity_tolerance,
            "perf_target_only_python": args.perf_target_only_python,
        },
    }

    if not args.skip_python:
        report["python"] = run_python_side(args, args.out_dir / "python_train.jsonl", args.out_dir)
    if not args.skip_zig:
        report["zig"] = run_zig_side(args, args.out_dir)

    zig_step_rows = [row for row in report.get("zig", {}).get("training_metrics", []) if row.get("event") == "step"]
    python_step_rows = report.get("python", {}).get("metrics", {}).get("train_metrics_history", [])
    py_loss = None
    if python_step_rows:
        py_loss = as_float_or_none(python_step_rows[-1].get("loss"))
    zig_final_step_loss = as_float_or_none(zig_step_rows[-1].get("loss")) if zig_step_rows else None
    zig_epoch_avg_loss = as_float_or_none(report.get("zig", {}).get("metrics", {}).get("final_avg_loss"))
    zig_loss = zig_final_step_loss if zig_final_step_loss is not None else zig_epoch_avg_loss
    step_loss_deltas: list[dict[str, Any]] = []
    for py_row, zig_row in zip(python_step_rows, zig_step_rows):
        py_step_loss = as_float_or_none(py_row.get("loss"))
        zig_step_loss = as_float_or_none(zig_row.get("loss"))
        delta = zig_step_loss - py_step_loss if py_step_loss is not None and zig_step_loss is not None else None
        step_loss_deltas.append({
            "step": py_row.get("step", zig_row.get("step")),
            "python": py_step_loss,
            "zig": zig_step_loss,
            "delta": delta,
            "ok": delta is not None and abs(delta) <= args.loss_parity_tolerance,
        })
    step_loss_counts_match = bool(python_step_rows) and len(python_step_rows) == len(zig_step_rows)
    step_loss_parity_matches = step_loss_counts_match and all(row.get("ok") for row in step_loss_deltas)
    largest_step_loss_delta = max(
        step_loss_deltas,
        key=lambda row: abs(float(row["delta"])) if row.get("delta") is not None else -1.0,
        default=None,
    )
    zig_op_stats = parse_zig_op_stats(report.get("zig", {}).get("output", ""))
    zig_op_runs = parse_zig_op_runs(report.get("zig", {}).get("output", ""))
    python_trainer_elapsed = report.get("python", {}).get("metrics", {}).get("elapsed_seconds")
    python_step_timings = report.get("python", {}).get("metrics", {}).get("step_timings", [])
    python_total_step_ms = report.get("python", {}).get("metrics", {}).get("total_step_wall_ms")
    python_avg_step_ms = report.get("python", {}).get("metrics", {}).get("avg_step_wall_ms")
    zig_total_trainer_ms = sum(float(row.get("trainer_total_ms") or 0.0) for row in zig_step_rows) if zig_step_rows else None
    zig_avg_trainer_ms = (zig_total_trainer_ms / len(zig_step_rows)) if zig_total_trainer_ms is not None and zig_step_rows else None
    python_warm_step_timings = python_step_timings[1:] if len(python_step_timings) > 1 else []
    python_warm_total_step_ms = (
        sum(float(row.get("step_wall_ms") or 0.0) for row in python_warm_step_timings)
        if python_warm_step_timings
        else None
    )
    python_warm_avg_step_ms = (
        python_warm_total_step_ms / len(python_warm_step_timings)
        if python_warm_total_step_ms is not None and python_warm_step_timings
        else None
    )
    python_preprocess_debug = report.get("python", {}).get("metrics", {}).get("span_preprocess_debug")
    python_preprocess_debug_samples = report.get("python", {}).get("metrics", {}).get("span_preprocess_debug_samples")
    python_task_breakdown = summarize_preprocess_tasks(python_preprocess_debug_samples)
    zig_preprocess_debug = report.get("zig", {}).get("metrics", {}).get("span_preprocess_debug")
    if python_preprocess_debug_samples:
        preprocess_matches, preprocess_mismatches = compare_preprocess_debug_samples(python_preprocess_debug_samples, zig_preprocess_debug)
    else:
        preprocess_matches, preprocess_mismatches = compare_preprocess_debug(python_preprocess_debug, zig_preprocess_debug)
    python_total_components = report.get("python", {}).get("metrics", {}).get("gliner2_total_loss_components")
    zig_total_components = report.get("zig", {}).get("metrics", {}).get("gliner2_total_loss_components")
    component_loss_matches, component_loss_deltas = compare_component_losses(python_total_components, zig_total_components, args.loss_parity_tolerance)
    component_loss_focus = summarize_component_deltas(component_loss_deltas)
    zig_manifest = report.get("zig", {}).get("training_manifest", {})
    zig_warm_step_rows = zig_step_rows[1:] if len(zig_step_rows) > 1 else []
    zig_warm_total_trainer_ms = (
        sum(float(row.get("trainer_total_ms") or 0.0) for row in zig_warm_step_rows)
        if zig_warm_step_rows
        else None
    )
    zig_warm_avg_trainer_ms = (
        zig_warm_total_trainer_ms / len(zig_warm_step_rows)
        if zig_warm_total_trainer_ms is not None and zig_warm_step_rows
        else None
    )
    zig_epoch_metrics = next((row for row in report.get("zig", {}).get("training_metrics", []) if row.get("event") == "epoch"), {})
    def zig_step_sum(key: str) -> float | None:
        if not zig_step_rows:
            return None
        return sum(float(row.get(key) or 0.0) for row in zig_step_rows)

    def zig_step_avg(key: str) -> float | None:
        total = zig_step_sum(key)
        return (total / len(zig_step_rows)) if total is not None and zig_step_rows else None

    trainable_parity_warning = None if args.zig_lora_only_trainables else "Zig is training regular task-head params in addition to LoRA params; upstream GLiNER2 LoRA freezes non-LoRA params"
    non_entity_task_count = (
        int(converted.get("classifications", 0))
        + int(converted.get("json_structures", 0))
        + int(converted.get("relations", 0))
    )
    source_non_entity_task_count = (
        int(source_task_summary.get("classifications", 0))
        + int(source_task_summary.get("json_structures", 0))
        + int(source_task_summary.get("relations", 0))
    )
    non_entity_task_warning = (
        f"Python fixture includes {source_non_entity_task_count} non-entity task annotations; Zig currently derives extractive span targets and does not train upstream classification/count/relation objectives"
        if source_non_entity_task_count > 0
        else None
    )
    entity_only_structure_parity = (
        args.zig_objective == "span-start"
        and args.span_loss_reduction == "sum"
        and args.span_positive_weight == 1.0
        and args.span_negative_weight == 1.0
        and args.span_hard_negative_weight == 1.0
        and source_non_entity_task_count == 0
    )
    full_loss_components_supported = False
    upstream_preprocessing_supported = False
    if args.zig_objective == "gliner2-total-loss":
        full_loss_components_supported = True
        upstream_preprocessing_supported = preprocess_matches
        objective_parity_warning = None if preprocess_matches and component_loss_matches else (
            "Zig gliner2-total-loss has graph-native structure/classification/count loss components, "
            "but full accuracy parity still requires matching upstream multi-schema preprocessing and component-level Python/Zig loss checks"
        )
        zig_objective_semantics = "flattened schema-conditioned structure_loss plus graph-native classification_loss and count_loss"
    elif entity_only_structure_parity:
        objective_parity_warning = "Current objective parity is scoped to upstream entity-only structure_loss with gold_count=1; GLiNER2 classification, count_loss, relations, and multi-structure count>1 losses are not covered by this benchmark"
        zig_objective_semantics = "entity-only structure_loss-compatible flattened span/start-width BCE"
    elif non_entity_task_warning is not None:
        objective_parity_warning = non_entity_task_warning
        zig_objective_semantics = "span-start BCE over extractive mentions derived from upstream-format records"
    elif args.zig_objective == "span-start":
        objective_parity_warning = "Zig span-start settings differ from upstream entity-only structure_loss settings; use sum reduction and unit positive/negative/hard-negative weights for the closest current parity run"
        zig_objective_semantics = "span-start BCE surrogate"
    else:
        objective_parity_warning = "Zig token-classification objective does not match upstream GLiNER2Trainer structure_loss training"
        zig_objective_semantics = "token classification"
    loss_delta = (zig_loss - py_loss) if zig_loss is not None and py_loss is not None else None
    if args.zig_objective == "gliner2-total-loss":
        valid_loss_parity = (
            not args.perf_target_only_python
            and trainable_parity_warning is None
            and preprocess_matches
            and component_loss_matches
            and step_loss_parity_matches
            and loss_delta is not None
            and abs(loss_delta) <= args.loss_parity_tolerance
        )
    else:
        valid_loss_parity = (
            not args.perf_target_only_python
            and trainable_parity_warning is None
            and entity_only_structure_parity
            and step_loss_parity_matches
            and loss_delta is not None
            and abs(loss_delta) <= args.loss_parity_tolerance
        )
    loss_parity_warning = None
    if args.perf_target_only_python:
        loss_parity_warning = "Python is being used as a timing target only; Python/Zig loss parity is intentionally not asserted"
    elif loss_delta is None:
        loss_parity_warning = "Python/Zig loss parity was not evaluated because one side did not report loss"
    elif not step_loss_counts_match:
        loss_parity_warning = f"Python/Zig step counts differ for loss parity: python={len(python_step_rows)} zig={len(zig_step_rows)}"
    elif not step_loss_parity_matches:
        step = largest_step_loss_delta.get("step") if largest_step_loss_delta else None
        delta = largest_step_loss_delta.get("delta") if largest_step_loss_delta else None
        loss_parity_warning = f"Python/Zig per-step loss parity failed at step {step}: delta {delta:.9g} exceeds tolerance {args.loss_parity_tolerance:.9g}"
    elif not valid_loss_parity:
        loss_parity_warning = f"Python/Zig loss delta {loss_delta:.9g} exceeds tolerance {args.loss_parity_tolerance:.9g} or objective/trainable parity is incomplete"
    report["summary"] = {
        "python_returncode": report.get("python", {}).get("returncode"),
        "zig_returncode": report.get("zig", {}).get("returncode"),
        "python_elapsed_seconds": report.get("python", {}).get("elapsed_seconds"),
        "zig_elapsed_seconds": report.get("zig", {}).get("elapsed_seconds"),
        "python_trainer_elapsed_seconds": python_trainer_elapsed,
        "python_step_count": len(python_step_timings) if python_step_timings else None,
        "python_total_step_wall_ms": python_total_step_ms,
        "python_avg_step_wall_ms": python_avg_step_ms,
        "python_warm_step_count": len(python_warm_step_timings) if python_warm_step_timings else None,
        "python_warm_total_step_wall_ms": python_warm_total_step_ms,
        "python_warm_avg_step_wall_ms": python_warm_avg_step_ms,
        "zig_step_count": len(zig_step_rows) if zig_step_rows else None,
        "zig_total_trainer_ms": zig_total_trainer_ms,
        "zig_avg_trainer_ms": zig_avg_trainer_ms,
        "zig_warm_step_count": len(zig_warm_step_rows) if zig_warm_step_rows else None,
        "zig_warm_total_trainer_ms": zig_warm_total_trainer_ms,
        "zig_warm_avg_trainer_ms": zig_warm_avg_trainer_ms,
        "zig_epoch_wall_ms": zig_epoch_metrics.get("epoch_wall_ms"),
        "zig_epoch_supervised_tokens_per_second": zig_epoch_metrics.get("supervised_tokens_per_second"),
        "zig_graph_executor_partitions_avg": zig_step_avg("graph_executor_partitions"),
        "zig_graph_executor_command_dispatches_avg": zig_step_avg("graph_executor_command_dispatches"),
        "zig_graph_executor_planned_dispatches_avg": zig_step_avg("graph_executor_planned_dispatches"),
        "zig_graph_executor_interpreter_fallbacks_avg": zig_step_avg("graph_executor_interpreter_fallbacks"),
        "zig_graph_executor_host_outputs_avg": zig_step_avg("graph_executor_host_outputs"),
        "zig_graph_executor_regions_avg": zig_step_avg("graph_executor_regions"),
        "zig_graph_executor_runtime_region_dispatches_avg": zig_step_avg("graph_executor_runtime_region_dispatches"),
        "zig_graph_executor_runtime_region_active_regions_avg": zig_step_avg("graph_executor_runtime_region_active_regions"),
        "zig_graph_executor_runtime_region_covered_nodes_avg": zig_step_avg("graph_executor_runtime_region_covered_nodes"),
        "zig_graph_executor_runtime_region_elided_nodes_avg": zig_step_avg("graph_executor_runtime_region_elided_nodes"),
        "zig_graph_executor_runtime_region_plan_compiles_avg": zig_step_avg("graph_executor_runtime_region_plan_compiles"),
        "zig_graph_executor_runtime_region_plan_reuses_avg": zig_step_avg("graph_executor_runtime_region_plan_reuses"),
        "zig_graph_executor_plan_build_ms_avg": zig_step_avg("graph_executor_plan_build_ms"),
        "zig_graph_executor_buffer_plan_build_ms_avg": zig_step_avg("graph_executor_buffer_plan_build_ms"),
        "zig_graph_executor_plan_cache_hits_avg": zig_step_avg("graph_executor_plan_cache_hits"),
        "zig_graph_executor_plan_cache_misses_avg": zig_step_avg("graph_executor_plan_cache_misses"),
        "zig_metal_frame_wait_ms_avg": zig_step_avg("metal_frame_wait_ms"),
        "zig_metal_frame_gpu_ms_avg": zig_step_avg("metal_frame_gpu_ms"),
        "zig_metal_last_frame_compute_encoders_avg": zig_step_avg("metal_last_frame_compute_encoders"),
        "zig_metal_last_frame_blit_encoders_avg": zig_step_avg("metal_last_frame_blit_encoders"),
        "zig_metal_last_frame_planned_scopes_avg": zig_step_avg("metal_last_frame_planned_scopes"),
        "zig_metal_last_frame_planned_barriers_avg": zig_step_avg("metal_last_frame_planned_barriers"),
        "zig_metal_last_frame_planned_command_ops_avg": zig_step_avg("metal_last_frame_planned_command_ops"),
        "zig_metal_deberta_encoder_plan_attempts_avg": zig_step_avg("metal_deberta_encoder_plan_attempts"),
        "zig_metal_deberta_encoder_plan_successes_avg": zig_step_avg("metal_deberta_encoder_plan_successes"),
        "zig_metal_deberta_encoder_plan_reuses_avg": zig_step_avg("metal_deberta_encoder_plan_reuses"),
        "zig_metal_deberta_encoder_plan_failures_avg": zig_step_avg("metal_deberta_encoder_plan_failures"),
        "zig_metal_deberta_encoder_layer_attempts_avg": zig_step_avg("metal_deberta_encoder_layer_attempts"),
        "zig_metal_deberta_encoder_layer_successes_avg": zig_step_avg("metal_deberta_encoder_layer_successes"),
        "zig_metal_deberta_encoder_layer_fallbacks_avg": zig_step_avg("metal_deberta_encoder_layer_fallbacks"),
        "zig_metal_deberta_relative_qk_pair_calls_avg": zig_step_avg("metal_deberta_relative_qk_pair_calls"),
        "zig_metal_deberta_relative_qk_pair_fallbacks_avg": zig_step_avg("metal_deberta_relative_qk_pair_fallbacks"),
        "zig_metal_deberta_ffn_fused_calls_avg": zig_step_avg("metal_deberta_ffn_fused_calls"),
        "zig_metal_deberta_ffn_fused_mps_matmuls_avg": zig_step_avg("metal_deberta_ffn_fused_mps_matmuls"),
        "zig_metal_deberta_ffn_fused_fallbacks_avg": zig_step_avg("metal_deberta_ffn_fused_fallbacks"),
        "zig_metal_deberta_attention_flash_calls_avg": zig_step_avg("metal_deberta_attention_flash_calls"),
        "zig_metal_deberta_attention_gemm_calls_avg": zig_step_avg("metal_deberta_attention_gemm_calls"),
        "zig_metal_deberta_attention_gemm_fallbacks_avg": zig_step_avg("metal_deberta_attention_gemm_fallbacks"),
        "zig_metal_deberta_attention_legacy_calls_avg": zig_step_avg("metal_deberta_attention_legacy_calls"),
        "zig_dot_general_command_count": zig_op_stats.get("command_ops", {}).get("dot_general", {}).get("count"),
        "zig_dot_general_command_total_ms": zig_op_stats.get("command_ops", {}).get("dot_general", {}).get("total_ms"),
        "zig_dot_general_command_avg_ms": zig_op_stats.get("command_ops", {}).get("dot_general", {}).get("avg_ms"),
        "zig_gather_fallback_count": zig_op_stats.get("fallback_ops", {}).get("gather", {}).get("count"),
        "zig_gather_fallback_total_ms": zig_op_stats.get("fallback_ops", {}).get("gather", {}).get("total_ms"),
        "zig_gather_host_output_count": zig_op_stats.get("host_output_ops", {}).get("gather", {}).get("count"),
        "zig_gather_host_output_total_ms": zig_op_stats.get("host_output_ops", {}).get("gather", {}).get("total_ms"),
        "zig_top_dot_shapes": zig_op_runs.get("top_dot_shapes", []),
        "python_cpu_step_gate_ms": python_avg_step_ms,
        "zig_beats_python_cpu_step_time": (
            zig_avg_trainer_ms < python_avg_step_ms
            if zig_avg_trainer_ms is not None and python_avg_step_ms is not None
            else None
        ),
        "zig_beats_python_cpu_warm_step_time": (
            zig_warm_avg_trainer_ms < python_warm_avg_step_ms
            if zig_warm_avg_trainer_ms is not None and python_warm_avg_step_ms is not None
            else None
        ),
        "trainer_speedup_python_over_zig": (
            python_trainer_elapsed / (zig_total_trainer_ms / 1000.0)
            if python_trainer_elapsed is not None and zig_total_trainer_ms not in (None, 0)
            else None
        ),
        "warm_step_wall_speedup_python_over_zig": (
            python_warm_total_step_ms / zig_warm_total_trainer_ms
            if python_warm_total_step_ms not in (None, 0) and zig_warm_total_trainer_ms not in (None, 0)
            else None
        ),
        "zig_warm_step_wall_slowdown_vs_python": (
            zig_warm_total_trainer_ms / python_warm_total_step_ms
            if python_warm_total_step_ms not in (None, 0) and zig_warm_total_trainer_ms not in (None, 0)
            else None
        ),
        "step_wall_speedup_python_over_zig": (
            python_total_step_ms / zig_total_trainer_ms
            if python_total_step_ms not in (None, 0) and zig_total_trainer_ms not in (None, 0)
            else None
        ),
        "zig_step_wall_slowdown_vs_python": (
            zig_total_trainer_ms / python_total_step_ms
            if python_total_step_ms not in (None, 0) and zig_total_trainer_ms not in (None, 0)
            else None
        ),
        "python_last_loss": py_loss,
        "zig_final_step_loss": zig_final_step_loss,
        "zig_final_avg_loss": zig_epoch_avg_loss,
        "loss_delta_zig_minus_python": loss_delta,
        "step_loss_parity_matches": step_loss_parity_matches,
        "step_loss_counts_match": step_loss_counts_match,
        "step_loss_deltas": step_loss_deltas,
        "largest_step_loss_delta": largest_step_loss_delta,
        "preprocess_parity_matches": preprocess_matches,
        "preprocess_parity_mismatches": preprocess_mismatches,
        "component_loss_parity_matches": component_loss_matches,
        "component_loss_deltas": component_loss_deltas,
        "component_loss_focus": component_loss_focus,
        "python_preprocess_task_breakdown": python_task_breakdown,
        "zig_manifest_backend": zig_manifest.get("backend"),
        "zig_manifest_objective": zig_manifest.get("objective"),
        "metal_readiness": summarize_metal_readiness(args, report, zig_step_rows, zig_manifest),
        "loss_parity_tolerance": args.loss_parity_tolerance,
        "valid_loss_parity": valid_loss_parity,
        "loss_parity_warning": loss_parity_warning,
        "perf_target_only_python": args.perf_target_only_python,
        "python_objective_semantics": "upstream GLiNER2Trainer total_loss = classification_loss + structure_loss + count_loss; LoRA mode freezes non-LoRA params",
        "zig_objective_semantics": zig_objective_semantics,
        "required_loss_components": ["classification_loss", "structure_loss", "count_loss"],
        "zig_loss_components_supported": {
            "classification_loss": full_loss_components_supported,
            "structure_loss": args.zig_objective in ("span-start", "gliner2-total-loss"),
            "count_loss": full_loss_components_supported,
        },
        "zig_preprocessing_matches_upstream": upstream_preprocessing_supported,
        "valid_full_loss_parity": bool(
            valid_loss_parity
            and full_loss_components_supported
            and upstream_preprocessing_supported
            and args.zig_objective == "gliner2-total-loss"
        ),
        "entity_only_structure_parity": entity_only_structure_parity,
        "non_entity_task_annotations": source_non_entity_task_count,
        "training_slice_non_entity_task_annotations": non_entity_task_count,
        "trainable_parity_warning": trainable_parity_warning,
        "objective_parity_warning": objective_parity_warning,
        "semantic_parity_warning": objective_parity_warning if trainable_parity_warning is None else f"{trainable_parity_warning}; {objective_parity_warning}",
    }

    for section in ("python", "zig"):
        if section in report:
            trim_result_output(report[section], args.max_command_output_chars)

    report_path = args.out_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"comparison report: {report_path}")
    print(json.dumps(report["summary"], indent=2))
    ok = all(
        section not in report or report[section].get("returncode") == 0
        for section in ("python", "zig")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
