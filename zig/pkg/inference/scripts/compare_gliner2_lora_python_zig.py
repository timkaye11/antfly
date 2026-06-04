#!/usr/bin/env python3
"""Run a small GLiNER2 LoRA apples-to-apples smoke comparison.

The script intentionally keeps the comparison narrow: same local model dir,
same NER smoke JSONL, same LoRA rank/alpha/dropout/target groups, one backend
for Zig, and one JSON report with raw command outputs plus extracted metrics.
"""

from __future__ import annotations

import argparse
import json
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


def convert_to_python_jsonl(src: Path, dst: Path, allowed_labels: set[str] | None = None) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    examples = 0
    mentions = 0
    labels: set[str] = set()
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            record = json.loads(line)
            grouped: dict[str, list[str]] = {}
            for ent in record.get("entities", []):
                label = ent["label"]
                if allowed_labels is not None and label not in allowed_labels:
                    continue
                grouped.setdefault(label, []).append(ent["text"])
                labels.add(label)
                mentions += 1
            fout.write(json.dumps({"input": record["text"], "output": {"entities": grouped}}, ensure_ascii=False) + "\n")
            examples += 1
    return {"examples": examples, "mentions": mentions, "labels": sorted(labels), "path": str(dst)}


def convert_limited_to_python_jsonl(src: Path, dst: Path, max_examples: int, allowed_labels: set[str] | None = None) -> dict[str, Any]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    examples = 0
    mentions = 0
    labels: set[str] = set()
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if examples >= max_examples:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            grouped: dict[str, list[str]] = {}
            for ent in record.get("entities", []):
                label = ent["label"]
                if allowed_labels is not None and label not in allowed_labels:
                    continue
                grouped.setdefault(label, []).append(ent["text"])
                labels.add(label)
                mentions += 1
            fout.write(json.dumps({"input": record["text"], "output": {"entities": grouped}}, ensure_ascii=False) + "\n")
            examples += 1
    return {"examples": examples, "mentions": mentions, "labels": sorted(labels), "path": str(dst)}


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


def parse_zig_output(output: str) -> dict[str, Any]:
    step = re.search(r"loss=([0-9.eE+-]+)\s+grad_norm=([0-9.eE+-]+)\s+supervised_tok/s=([0-9.eE+-]+)", output)
    final = re.search(r"final avg loss=([0-9.eE+-]+)", output)
    loaded = re.search(r"loaded\s+(\d+)\s+weights", output)
    parity_debug = extract_prefixed_json(output, "SPAN_PARITY_DEBUG ")
    preprocess_debug = extract_prefixed_json(output, "SPAN_PREPROCESS_DEBUG ")
    component_debug = extract_prefixed_json(output, "SPAN_COMPONENT_DEBUG ")
    return {
        "step_loss": float(step.group(1)) if step else None,
        "grad_norm": float(step.group(2)) if step else None,
        "supervised_tok_per_s": float(step.group(3)) if step else None,
        "final_avg_loss": float(final.group(1)) if final else None,
        "loaded_weight_count": int(loaded.group(1)) if loaded else None,
        "span_parity_debug": parity_debug[-1] if parity_debug else None,
        "span_preprocess_debug": preprocess_debug[-1] if preprocess_debug else None,
        "span_component_debug": component_debug[-1] if component_debug else None,
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
        r":count=(?P<count>\d+):rhs_transpose=(?P<rhs_transpose>true|false)"
        r":rhs_parameter=(?P<rhs_parameter>true|false):rhs_lora=(?P<rhs_lora>true|false):raw_linear=(?P<raw_linear>true|false)"
    )
    for item in payload.split(","):
        item = item.strip()
        if not item or item == "none":
            continue
        match = shape_re.fullmatch(item)
        if not match:
            continue
        groups = match.groupdict()
        shapes.append({
            "lhs": [int(groups["lhs0"]), int(groups["lhs1"])],
            "rhs": [int(groups["rhs0"]), int(groups["rhs1"])],
            "out": [int(groups["out0"]), int(groups["out1"])],
            "count": int(groups["count"]),
            "rhs_transpose": parse_bool_token(groups["rhs_transpose"]),
            "rhs_parameter": parse_bool_token(groups["rhs_parameter"]),
            "rhs_lora": parse_bool_token(groups["rhs_lora"]),
            "raw_linear": parse_bool_token(groups["raw_linear"]),
        })
    return shapes


def parse_zig_op_runs(output: str) -> dict[str, Any]:
    dot_shapes: list[dict[str, Any]] = []
    for line in output.splitlines():
        prefix = "metal_partition_dot_shapes:"
        if not line.startswith(prefix):
            continue
        top_index = line.find(" top=")
        if top_index < 0:
            continue
        dot_shapes.extend(parse_dot_shape_items(line[top_index + len(" top="):]))
    dot_shapes.sort(key=lambda item: item["count"], reverse=True)
    return {
        "dot_shapes": dot_shapes,
        "top_dot_shapes": dot_shapes[:16],
    }


def extract_prefixed_json(output: str, prefix: str) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for line in output.splitlines():
        if not line.startswith(prefix):
            continue
        try:
            payloads.append(json.loads(line[len(prefix):]))
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
import argparse, json, math, os, pathlib, time, types
import torch
import torch.nn.functional as F
import gliner2.model as gliner2_model
from gliner2.model import Extractor
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
p.add_argument("--lora-rank", type=int, required=True)
p.add_argument("--lora-alpha", type=float, required=True)
p.add_argument("--lora-dropout", type=float, required=True)
p.add_argument("--lora-targets", required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--span-negative-mask-rate", type=float, required=True)
p.add_argument("--disable-model-dropout", action="store_true")
p.add_argument("--dump-parity", action="store_true")
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
cfg = TrainingConfig(
    output_dir=str(out),
    experiment_name="gliner2_python_zig_smoke",
    num_epochs=1,
    max_steps=args.steps,
    batch_size=args.batch_size,
    eval_batch_size=args.batch_size,
    gradient_accumulation_steps=1,
    encoder_lr=args.learning_rate,
    task_lr=args.learning_rate,
    weight_decay=0.0,
    max_grad_norm=1.0,
    scheduler_type="constant",
    warmup_steps=0,
    warmup_ratio=0.0,
    fp16=False,
    bf16=False,
    eval_strategy="no",
    save_total_limit=1,
    save_best=False,
    logging_steps=1,
    logging_first_step=True,
    report_to_wandb=False,
    early_stopping=False,
    num_workers=0,
    pin_memory=False,
    seed=args.seed,
    deterministic=True,
    max_train_samples=args.steps * args.batch_size,
    max_len=args.seq_len,
    use_lora=True,
    lora_r=args.lora_rank,
    lora_alpha=args.lora_alpha,
    lora_dropout=args.lora_dropout,
    lora_target_modules=args.lora_targets.split(","),
    save_adapter_only=True,
)
trainer = GLiNER2Trainer(model, cfg)
sampling = trainer.processor.sampling_config
sampling.synthetic_entity_label_prob = 0.0
sampling.shuffle_entities = False
sampling.remove_entity_prob = 0.0
sampling.remove_entities_prob = 0.0
python_step_timings = []
original_create_dataloader = trainer._create_dataloader

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
    loader = original_create_dataloader(*call_args, **call_kwargs)
    is_training = call_kwargs.get("is_training")
    if is_training is None and len(call_args) >= 4:
        is_training = call_args[3]
    if is_training:
        return TimedTrainingLoader(loader, python_step_timings)
    return loader

trainer._create_dataloader = create_dataloader_with_timing
parity_debug = []
preprocess_debug = []
component_debug = []
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

    original_collator_call = ExtractorCollator.__call__
    def collator_call_with_debug(self, batch):
        out = original_collator_call(self, batch)
        if not preprocess_debug and getattr(out, "input_ids", None) is not None and len(out) > 0:
            preprocess_debug.append({
                "input_ids": out.input_ids[0].detach().cpu().tolist(),
                "attention_mask": out.attention_mask[0].detach().cpu().tolist(),
                "text_word_indices": (
                    out.text_word_indices[0, :out.text_word_counts[0]].detach().cpu().tolist()
                    if out.text_word_indices is not None and out.text_word_counts else []
                ),
                "text_word_counts": list(out.text_word_counts or []),
                "schema_special_indices": out.schema_special_indices[0] if out.schema_special_indices else [],
                "text_tokens": out.text_tokens[0] if out.text_tokens else [],
                "schema_tokens_list": out.schema_tokens_list[0] if out.schema_tokens_list else [],
                "structure_labels": out.structure_labels[0] if out.structure_labels else [],
                "task_types": out.task_types[0] if out.task_types else [],
            })
        return out

    ExtractorCollator.__call__ = collator_call_with_debug

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
            positive_indices = (labs > 0).nonzero(as_tuple=False)
            if positive_indices.numel() > 0 and not component_debug:
                count_idx = int(positive_indices[0, 0].detach().cpu().item())
                entity_idx = int(positive_indices[0, 1].detach().cpu().item())
                start_idx = int(positive_indices[0, 2].detach().cpu().item())
                width_idx = int(positive_indices[0, 3].detach().cpu().item())
                span_vec = span_rep[start_idx, width_idx]
                schema_vec = schema_emb[1 + entity_idx]
                projection_vec = struct_proj[count_idx, entity_idx]
                component_debug.append({
                    "positive_row": start_idx * scores.shape[3] + width_idx,
                    "positive_entity": entity_idx,
                    "schema_row": entity_idx,
                    "count_index": count_idx,
                    "start_index": start_idx,
                    "width_index": width_idx,
                    "span_rep_norm": _finite(span_vec.float().norm().detach().cpu().item()),
                    "schema_hidden_norm": _finite(schema_vec.float().norm().detach().cpu().item()),
                    "schema_projection_norm": _finite(projection_vec.float().norm().detach().cpu().item()),
                    "projected_schema_dot": _finite(torch.dot(span_vec.flatten().float(), projection_vec.flatten().float()).detach().cpu().item()),
                    "span_rep_mean": _finite(span_vec.float().mean().detach().cpu().item()),
                    "schema_projection_mean": _finite(projection_vec.float().mean().detach().cpu().item()),
                })
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
                "masking_rate": float(masking_rate),
            })
        return loss

    Extractor.compute_struct_loss = compute_struct_loss_with_debug
    trainer.model.compute_struct_loss = types.MethodType(compute_struct_loss_with_debug, trainer.model)
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
    "span_parity_debug": parity_debug[0] if parity_debug else None,
    "span_preprocess_debug": preprocess_debug[0] if preprocess_debug else None,
    "span_component_debug": component_debug[0] if component_debug else None,
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
    result = run_command(cmd, repo_root(), timeout=args.timeout_seconds)
    metrics_path = out_dir / "python" / "comparison_metrics.json"
    result["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return result


def run_zig_side(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    zig_global_cache = out_dir / "zig-global-cache"
    zig_global_cache.mkdir(parents=True, exist_ok=True)
    enable_metal = args.zig_build_metal if args.zig_build_metal is not None else args.zig_backend == "metal"
    enable_mlx = args.zig_build_mlx if args.zig_build_mlx is not None else args.zig_backend == "mlx"
    cmd = [
        "zig", "build",
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
        "--backend", args.zig_backend,
        "--objective", args.zig_objective,
        "--entity-types", args.entity_types,
        "--num-classes", str(len([x for x in args.entity_types.split(",") if x]) + 1),
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
    if args.zig_lora_only_trainables:
        cmd.append("--lora-only-trainables")
    if args.dump_parity:
        cmd.append("--dump-span-parity")
    zig_env: dict[str, str] = {}
    if args.zig_backend == "metal" and args.zig_training_graph_executor:
        zig_env["TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR"] = "1"
    result = run_command(cmd, inference_dir(), timeout=args.timeout_seconds, env=zig_env)
    result["metrics"] = parse_zig_output(result["output"])
    result["training_metrics"] = load_jsonl(out_dir / "zig" / "training_metrics.jsonl")
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
    p.add_argument("--zig-objective", default="span-start", choices=["token", "span-start"])
    p.add_argument(
        "--zig-lora-only-trainables",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match upstream GLiNER2 LoRA training by freezing regular task-head params and optimizing only LoRA params",
    )
    p.add_argument("--zig-build-metal", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--zig-build-mlx", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--zig-optimize", choices=["Debug", "ReleaseSafe", "ReleaseFast", "ReleaseSmall"], default=None)
    p.add_argument("--dump-parity", action="store_true", help="Collect first-batch span objective logits/label/mask stats from both implementations")
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
    p.add_argument("--skip-python", action="store_true")
    p.add_argument("--skip-zig", action="store_true")
    p.add_argument("--keep-out-dir", action="store_true")
    args = p.parse_args()

    if not args.keep_out_dir and args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    converted = convert_limited_to_python_jsonl(
        args.train_data,
        args.out_dir / "python_train.jsonl",
        args.steps * args.batch_size,
        parse_label_csv(args.entity_types),
    )
    report: dict[str, Any] = {
        "task": "gliner2_lora_python_zig_apples_to_apples",
        "config": {
            "model_dir": str(args.model_dir),
            "python_model": str(args.python_model),
            "train_data": str(args.train_data),
            "converted_python_train_data": converted,
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
            "loss_parity_tolerance": args.loss_parity_tolerance,
            "perf_target_only_python": args.perf_target_only_python,
        },
    }

    if not args.skip_python:
        report["python"] = run_python_side(args, args.out_dir / "python_train.jsonl", args.out_dir)
    if not args.skip_zig:
        report["zig"] = run_zig_side(args, args.out_dir)

    py_loss = None
    if report.get("python", {}).get("metrics", {}).get("train_metrics_history"):
        py_loss = report["python"]["metrics"]["train_metrics_history"][-1].get("loss")
    zig_loss = report.get("zig", {}).get("metrics", {}).get("final_avg_loss")
    zig_step_rows = [row for row in report.get("zig", {}).get("training_metrics", []) if row.get("event") == "step"]
    zig_op_stats = parse_zig_op_stats(report.get("zig", {}).get("output", ""))
    zig_op_runs = parse_zig_op_runs(report.get("zig", {}).get("output", ""))
    python_trainer_elapsed = report.get("python", {}).get("metrics", {}).get("elapsed_seconds")
    python_step_timings = report.get("python", {}).get("metrics", {}).get("step_timings", [])
    python_total_step_ms = report.get("python", {}).get("metrics", {}).get("total_step_wall_ms")
    python_avg_step_ms = report.get("python", {}).get("metrics", {}).get("avg_step_wall_ms")
    zig_total_trainer_ms = sum(float(row.get("trainer_total_ms") or 0.0) for row in zig_step_rows) if zig_step_rows else None
    zig_avg_trainer_ms = (zig_total_trainer_ms / len(zig_step_rows)) if zig_total_trainer_ms is not None and zig_step_rows else None
    zig_epoch_metrics = next((row for row in report.get("zig", {}).get("training_metrics", []) if row.get("event") == "epoch"), {})
    def zig_step_sum(key: str) -> float | None:
        if not zig_step_rows:
            return None
        return sum(float(row.get(key) or 0.0) for row in zig_step_rows)

    def zig_step_avg(key: str) -> float | None:
        total = zig_step_sum(key)
        return (total / len(zig_step_rows)) if total is not None and zig_step_rows else None

    trainable_parity_warning = None if args.zig_lora_only_trainables else "Zig is training regular task-head params in addition to LoRA params; upstream GLiNER2 LoRA freezes non-LoRA params"
    entity_only_structure_parity = (
        args.zig_objective == "span-start"
        and args.span_loss_reduction == "sum"
        and args.span_positive_weight == 1.0
        and args.span_negative_weight == 1.0
        and args.span_hard_negative_weight == 1.0
    )
    if entity_only_structure_parity:
        objective_parity_warning = "Current objective parity is scoped to upstream entity-only structure_loss with gold_count=1; GLiNER2 classification, count_loss, relations, and multi-structure count>1 losses are not covered by this benchmark"
        zig_objective_semantics = "entity-only structure_loss-compatible flattened span/start-width BCE"
    elif args.zig_objective == "span-start":
        objective_parity_warning = "Zig span-start settings differ from upstream entity-only structure_loss settings; use sum reduction and unit positive/negative/hard-negative weights for the closest current parity run"
        zig_objective_semantics = "span-start BCE surrogate"
    else:
        objective_parity_warning = "Zig token-classification objective does not match upstream GLiNER2Trainer structure_loss training"
        zig_objective_semantics = "token classification"
    loss_delta = (zig_loss - py_loss) if zig_loss is not None and py_loss is not None else None
    valid_loss_parity = (
        not args.perf_target_only_python
        and trainable_parity_warning is None
        and entity_only_structure_parity
        and loss_delta is not None
        and abs(loss_delta) <= args.loss_parity_tolerance
    )
    loss_parity_warning = None
    if args.perf_target_only_python:
        loss_parity_warning = "Python is being used as a timing target only; Python/Zig loss parity is intentionally not asserted"
    elif loss_delta is None:
        loss_parity_warning = "Python/Zig loss parity was not evaluated because one side did not report loss"
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
        "zig_step_count": len(zig_step_rows) if zig_step_rows else None,
        "zig_total_trainer_ms": zig_total_trainer_ms,
        "zig_avg_trainer_ms": zig_avg_trainer_ms,
        "zig_epoch_wall_ms": zig_epoch_metrics.get("epoch_wall_ms"),
        "zig_epoch_supervised_tokens_per_second": zig_epoch_metrics.get("supervised_tokens_per_second"),
        "zig_graph_executor_partitions_avg": zig_step_avg("graph_executor_partitions"),
        "zig_graph_executor_command_dispatches_avg": zig_step_avg("graph_executor_command_dispatches"),
        "zig_graph_executor_planned_dispatches_avg": zig_step_avg("graph_executor_planned_dispatches"),
        "zig_graph_executor_interpreter_fallbacks_avg": zig_step_avg("graph_executor_interpreter_fallbacks"),
        "zig_graph_executor_host_outputs_avg": zig_step_avg("graph_executor_host_outputs"),
        "zig_graph_executor_regions_avg": zig_step_avg("graph_executor_regions"),
        "zig_graph_executor_runtime_region_dispatches_avg": zig_step_avg("graph_executor_runtime_region_dispatches"),
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
        "trainer_speedup_python_over_zig": (
            python_trainer_elapsed / (zig_total_trainer_ms / 1000.0)
            if python_trainer_elapsed is not None and zig_total_trainer_ms not in (None, 0)
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
        "zig_final_avg_loss": zig_loss,
        "loss_delta_zig_minus_python": loss_delta,
        "loss_parity_tolerance": args.loss_parity_tolerance,
        "valid_loss_parity": valid_loss_parity,
        "loss_parity_warning": loss_parity_warning,
        "perf_target_only_python": args.perf_target_only_python,
        "python_objective_semantics": "upstream GLiNER2Trainer total_loss = classification_loss + structure_loss + count_loss; LoRA mode freezes non-LoRA params",
        "zig_objective_semantics": zig_objective_semantics,
        "entity_only_structure_parity": entity_only_structure_parity,
        "trainable_parity_warning": trainable_parity_warning,
        "objective_parity_warning": objective_parity_warning,
        "semantic_parity_warning": objective_parity_warning if trainable_parity_warning is None else f"{trainable_parity_warning}; {objective_parity_warning}",
    }

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
