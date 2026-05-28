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


DEFAULT_PYTHON = "/private/tmp/gliner2-parity-py311/bin/python"
DEFAULT_MODEL_DIR = "/private/tmp/termite-models/gliner2"
DEFAULT_PYTHON_MODEL = "fastino/gliner2-base-v1"
DEFAULT_OUT_DIR = "/private/tmp/termite-gliner2-apples-to-apples"
DEFAULT_LABELS = "person,organization,location"
LORA_TARGETS = "encoder,span_rep,classifier,count_embed,count_pred"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def termite_dir() -> Path:
    return repo_root() / "zig" / "pkg" / "termite"


def default_train_data() -> Path:
    return termite_dir() / "testdata" / "gliner2_ner_smoke.jsonl"


def convert_to_python_jsonl(src: Path, dst: Path) -> dict[str, Any]:
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
                grouped.setdefault(label, []).append(ent["text"])
                labels.add(label)
                mentions += 1
            fout.write(json.dumps({"input": record["text"], "output": {"entities": grouped}}, ensure_ascii=False) + "\n")
            examples += 1
    return {"examples": examples, "mentions": mentions, "labels": sorted(labels), "path": str(dst)}


def convert_limited_to_python_jsonl(src: Path, dst: Path, max_examples: int) -> dict[str, Any]:
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
    for src in model_dir.iterdir():
        target = dst / src.name
        if src.name == "encoder_config":
            target.mkdir(exist_ok=True)
            for child in src.iterdir():
                child_target = target / child.name
                if child.name == "config.json":
                    cfg = json.loads(child.read_text(encoding="utf-8"))
                    cfg.setdefault("model_type", "deberta-v2")
                    child_target.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                elif not child_target.exists():
                    child_target.symlink_to(child)
        elif not target.exists():
            target.symlink_to(src)
    return dst


def run_command(cmd: list[str], cwd: Path, timeout: int | None = None) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        env={**os.environ, "TOKENIZERS_PARALLELISM": "false"},
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
    return {
        "step_loss": float(step.group(1)) if step else None,
        "grad_norm": float(step.group(2)) if step else None,
        "supervised_tok_per_s": float(step.group(3)) if step else None,
        "final_avg_loss": float(final.group(1)) if final else None,
        "loaded_weight_count": int(loaded.group(1)) if loaded else None,
    }


def python_training_script() -> str:
    return r'''
import argparse, json, math, pathlib, time
import torch
from gliner2.model import Extractor
from gliner2.training.trainer import TrainingConfig, GLiNER2Trainer

p = argparse.ArgumentParser()
p.add_argument("--model-dir", required=True)
p.add_argument("--train-data", required=True)
p.add_argument("--out-dir", required=True)
p.add_argument("--steps", type=int, required=True)
p.add_argument("--batch-size", type=int, required=True)
p.add_argument("--seq-len", type=int, required=True)
p.add_argument("--learning-rate", type=float, required=True)
p.add_argument("--lora-rank", type=int, required=True)
p.add_argument("--lora-alpha", type=float, required=True)
p.add_argument("--lora-dropout", type=float, required=True)
p.add_argument("--lora-targets", required=True)
p.add_argument("--seed", type=int, required=True)
args = p.parse_args()

out = pathlib.Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)

started = time.time()
model = Extractor.from_pretrained(args.model_dir, map_location="cpu")
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
result = trainer.train(train_data=args.train_data)
trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
total = sum(p.numel() for p in trainer.model.parameters())
payload = {
    "elapsed_seconds": time.time() - started,
    "total_steps": result.get("total_steps"),
    "samples_per_second": result.get("samples_per_second"),
    "train_metrics_history": result.get("train_metrics_history", []),
    "trainable_parameters": trainable,
    "total_parameters": total,
    "torch_version": torch.__version__,
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
        "--learning-rate", str(args.learning_rate),
        "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--lora-targets", args.lora_targets,
        "--seed", str(args.seed),
    ]
    result = run_command(cmd, repo_root(), timeout=args.timeout_seconds)
    metrics_path = out_dir / "python" / "comparison_metrics.json"
    result["metrics"] = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    return result


def run_zig_side(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    zig_global_cache = out_dir / "zig-global-cache"
    zig_global_cache.mkdir(parents=True, exist_ok=True)
    cmd = [
        "zig", "build",
        "--global-cache-dir", str(zig_global_cache),
        "-Dmlx=false", "-Donnx=false", "-Dmetal=false",
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
        "--objective", "span-start",
        "--entity-types", args.entity_types,
        "--num-classes", str(len([x for x in args.entity_types.split(",") if x]) + 1),
        "--max-span-width", str(args.max_span_width),
        "--span-loss", "bce",
        "--span-loss-reduction", args.span_loss_reduction,
        "--span-positive-weight", str(args.span_positive_weight),
        "--span-negative-weight", str(args.span_negative_weight),
        "--span-hard-negative-weight", str(args.span_hard_negative_weight),
        "--lora-rank", str(args.lora_rank),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--lora-targets", args.lora_targets,
        "--seed", str(args.seed),
    ]
    result = run_command(cmd, termite_dir(), timeout=args.timeout_seconds)
    result["metrics"] = parse_zig_output(result["output"])
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
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=float, default=32.0)
    p.add_argument("--lora-dropout", type=float, default=0.1)
    p.add_argument("--lora-targets", default=LORA_TARGETS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--zig-backend", default="native", choices=["native", "metal", "mlx", "auto"])
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
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
            "lora_targets": args.lora_targets,
            "seed": args.seed,
            "zig_backend": args.zig_backend,
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
    report["summary"] = {
        "python_returncode": report.get("python", {}).get("returncode"),
        "zig_returncode": report.get("zig", {}).get("returncode"),
        "python_last_loss": py_loss,
        "zig_final_avg_loss": zig_loss,
        "loss_delta_zig_minus_python": (zig_loss - py_loss) if zig_loss is not None and py_loss is not None else None,
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
