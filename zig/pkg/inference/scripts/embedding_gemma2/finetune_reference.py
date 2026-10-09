#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pinned PyTorch/PEFT reference for native EmbeddingGemma 2 LoRA training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import resource
import struct
import time
from pathlib import Path

MRL_DIMS = (128, 256, 512, 768)
TARGET_LEAVES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
BASE_WEIGHT_SHA256 = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
TOKENIZER_SHA256 = "4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def splitmix64(state: int) -> tuple[int, int]:
    mask = (1 << 64) - 1
    state = (state + 0x9E3779B97F4A7C15) & mask
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return state, (z ^ (z >> 31)) & mask


def shuffle_epoch(count: int, seed: int, epoch: int) -> list[int]:
    values = list(range(count)); mask = (1 << 64) - 1
    state = (seed ^ ((epoch * 0xD1342543DE82EF95) & mask)) & mask
    remaining = count
    while remaining > 1:
        bound = remaining; threshold = ((1 << 64) - bound) % bound
        state, value = splitmix64(state)
        while value < threshold: state, value = splitmix64(state)
        chosen = value % bound; remaining -= 1
        values[remaining], values[chosen] = values[chosen], values[remaining]
    return values


def batch_indices(count: int, seed: int, completed_steps: int, logical_batch: int) -> list[int]:
    if count <= 0 or logical_batch <= 0 or completed_steps < 0:
        raise ValueError("invalid batch schedule")
    steps_per_epoch = (count + logical_batch - 1) // logical_batch
    epoch, step_in_epoch = divmod(completed_steps, steps_per_epoch)
    cursor = step_in_epoch * logical_batch
    return shuffle_epoch(count, seed, epoch)[cursor:min(count, cursor + logical_batch)]


class PreparedDataset:
    """Bounded random access over native authenticated indexed shards."""
    def __init__(self, directory: Path):
        self.directory = directory; self.manifest = json.loads((directory / "manifest.json").read_text())
        if self.manifest.get("format") != "antfly.embeddinggemma2.retrieval-shards/v1": raise ValueError("invalid prepared dataset format")
        if self.manifest.get("tokenizer_sha256") != TOKENIZER_SHA256: raise ValueError("prepared tokenizer identity mismatch")
        if file_sha256(directory / "index.bin") != self.manifest["index_sha256"]: raise ValueError("prepared index checksum mismatch")
        for shard in self.manifest["shards"]:
            path = directory / shard["name"]
            if path.stat().st_size != shard["bytes"] or file_sha256(path) != shard["sha256"]: raise ValueError(f"prepared shard mismatch: {path}")
        self.index = (directory / "index.bin").open("rb"); header = self.index.read(16)
        if header[:8] != b"EG2IDX1\0" or struct.unpack("<Q", header[8:])[0] != self.manifest["records"]: raise ValueError("prepared index header mismatch")
        self.train_entries = []
        for record in range(self.manifest["records"]):
            entry = self.index.read(56)
            if len(entry) != 56: raise ValueError("truncated prepared index")
            if entry[48] == 0: self.train_entries.append(record)
        if len(self.train_entries) != self.manifest["train_records"]: raise ValueError("prepared train count mismatch")

    def __len__(self): return len(self.train_entries)
    def close(self):
        if not self.index.closed: self.index.close()
    def __getitem__(self, item):
        self.index.seek(16 + self.train_entries[item] * 56); entry = self.index.read(56)
        if len(entry) != 56 or entry[48] != 0: raise ValueError("prepared train index changed")
        shard, length, offset = struct.unpack("<IIQ", entry[:16])
        if shard >= len(self.manifest["shards"]): raise ValueError("invalid prepared shard index")
        with (self.directory / self.manifest["shards"][shard]["name"]).open("rb") as source:
            source.seek(offset); payload = source.read(length)
        if len(payload) != length or hashlib.sha256(payload).digest() != entry[16:48]: raise ValueError("prepared record checksum mismatch")
        row = json.loads(payload)
        if row["split"] != "train": raise ValueError("prepared split/index mismatch")
        return row


def pad_ids(rows, device):
    import torch
    width = max(len(row) for row in rows)
    ids = torch.zeros((len(rows), width), dtype=torch.long, device=device); mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        ids[index, :len(row)] = torch.tensor(row, dtype=torch.long, device=device); mask[index, :len(row)] = 1
    return {"input_ids": ids, "attention_mask": mask, "_lengths": tuple(len(row) for row in rows)}


def learning_rate(completed: int, total: int, peak: float = 1e-4, warmup_fraction: float = 0.05) -> float:
    """Exact embedding_gemma2_job.zig completed-step schedule."""
    if total <= 0 or completed < 0 or completed >= total:
        raise ValueError("invalid optimizer step")
    warmup = math.floor(total * warmup_fraction)
    step = completed + 1
    if warmup and step <= warmup:
        return peak * step / warmup
    if total == warmup:
        return peak
    return peak * (total - completed) / (total - warmup)


def target_module_names(model) -> list[str]:
    """Resolve only text encoder linears and its final embedding projection."""
    import torch

    names = []
    for name, module in model.named_modules():
        canonical = name.removeprefix("model.")
        if not canonical.startswith("language_model.") or not isinstance(module, torch.nn.Linear):
            continue
        if canonical.endswith(".embedding_projection") or canonical.rsplit(".", 1)[-1] in TARGET_LEAVES:
            names.append(name)
    if not names or not any(n.removeprefix("model.") == "language_model.embedding_projection" for n in names):
        raise RuntimeError("the pinned text encoder/projection target inventory was not found")
    return sorted(names)


def pool(outputs, mask):
    import torch

    hidden = outputs.last_hidden_state.float()
    weighted = hidden * mask.unsqueeze(-1).float()
    return torch.nn.functional.normalize(weighted.sum(1) / mask.sum(1, keepdim=True).clamp_min(1), dim=-1)


def mrl_loss(query, documents, positive_index, allowed, temperature: float):
    import torch

    losses = []
    for dim in MRL_DIMS:
        q = torch.nn.functional.normalize(query[:, :dim], dim=-1)
        d = torch.nn.functional.normalize(documents[:, :dim], dim=-1)
        logits = (q @ d.T / temperature).masked_fill(~allowed, torch.finfo(q.dtype).min)
        losses.append(torch.nn.functional.cross_entropy(logits, positive_index))
    return torch.stack(losses).mean()


def sliced_inputs(encoded, start: int, end: int):
    lengths = encoded["_lengths"][start:end]
    if not lengths: raise ValueError("empty microbatch slice")
    width = max(lengths)
    return {name: encoded[name][start:end, :width] for name in ("input_ids", "attention_mask")}


def forward_representation(model, part, precision: str):
    import torch

    enabled = precision == "bf16"
    with torch.autocast(device_type=part["input_ids"].device.type, dtype=torch.bfloat16, enabled=enabled):
        outputs = model(**part)
    return pool(outputs, part["attention_mask"])


def gradcache_representations(model, encoded, microbatch: int, precision: str):
    """First GradCache pass: cache FP32 representations without activations."""
    import torch

    chunks = []
    with torch.no_grad():
        for start in range(0, encoded["input_ids"].shape[0], microbatch):
            part = sliced_inputs(encoded, start, start + microbatch)
            chunks.append(forward_representation(model, part, precision))
    cached = torch.cat(chunks).detach().requires_grad_(True)
    return cached


def gradcache_replay(model, encoded, cached_gradient, microbatch: int, precision: str):
    """Second pass: replay each microbatch using gradients of cached reps."""
    for start in range(0, encoded["input_ids"].shape[0], microbatch):
        part = sliced_inputs(encoded, start, start + microbatch)
        representation = forward_representation(model, part, precision)
        (representation * cached_gradient[start:start + representation.shape[0]]).sum().backward()


def state_hash(model, optimizer) -> str:
    import torch

    digest = hashlib.sha256()
    trainables = [(name, value) for name, value in sorted(model.named_parameters()) if value.requires_grad]
    for name, value in trainables:
        digest.update(name.encode() + b"\0")
        digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
    # Optimizer state is included canonically after at least one update.
    for name, parameter in trainables:
        digest.update(name.encode() + b"\0optimizer\0")
        for key, value in sorted(optimizer.state[parameter].items()):
            digest.update(str(key).encode() + b"\0")
            if torch.is_tensor(value):
                digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
            else:
                digest.update(repr(value).encode())
    return digest.hexdigest()


def canonical_adapter_name(name: str) -> str:
    name = name.removeprefix("base_model.model.")
    return name.replace(".lora_A.default.weight", ".lora_A.weight").replace(".lora_B.default.weight", ".lora_B.weight")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--microbatch-size", type=int, default=2)
    parser.add_argument("--logical-batch-size", type=int, default=64)
    parser.add_argument("--max-negatives", type=int, default=4)
    parser.add_argument("--rank", type=int, choices=(8, 16, 32), default=16)
    parser.add_argument("--alpha", type=float, default=32.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", choices=("eager","sdpa"), default="eager")
    parser.add_argument("--trace-full", action="store_true")
    parser.add_argument("--trace-tensors-dir", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--initial-adapter", type=Path)
    parser.add_argument("--base-weight-sha256", default=BASE_WEIGHT_SHA256)
    parser.add_argument("--disable-bf16-reduced-precision-reduction", action="store_true",
                        help="diagnostic: force FP32 accumulation/reduction for BF16 CUDA matmuls")
    parser.add_argument("--paired-control-dir", type=Path,
                        help="qualification-only file grant directory for resident paired timing")
    args = parser.parse_args()
    if args.steps < 1 or args.microbatch_size < 1 or args.logical_batch_size < 1 or args.microbatch_size > args.logical_batch_size:
        parser.error("invalid step or batch size")
    if not 0 <= args.max_negatives <= 16:
        parser.error("max-negatives must be in [0,16]")
    if not math.isfinite(args.alpha) or not 0 < args.alpha <= 1024:
        parser.error("alpha must be finite and in (0,1024]")
    if not 0 <= args.seed < 1 << 64:
        parser.error("seed must be in [0,2^64)")
    actual_weight_sha = file_sha256(args.model / "model.safetensors")
    if actual_weight_sha != args.base_weight_sha256:
        raise ValueError("base model.safetensors identity mismatch")

    import numpy as np
    import peft
    import torch
    import transformers
    from transformers import EmbeddingGemma2Model

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = not args.disable_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_tf32 = False

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    base = EmbeddingGemma2Model.from_pretrained(args.model, local_files_only=True, torch_dtype=dtype,
                                                attn_implementation=args.attention).to(args.device)
    targets = target_module_names(base)
    config = peft.LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none", target_modules=targets, task_type="FEATURE_EXTRACTION")
    model = peft.get_peft_model(base, config)
    if args.initial_adapter:
        from safetensors.torch import load_file
        initial = load_file(args.initial_adapter, device=args.device)
        expected_adapter = peft.get_peft_model_state_dict(model)
        if set(initial) != set(expected_adapter): raise ValueError("initial adapter tensor set does not match the exact 169-target PEFT inventory")
        result = peft.set_peft_model_state_dict(model, initial)
        if result.unexpected_keys: raise ValueError(f"unexpected initial adapter tensors: {result.unexpected_keys}")
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    if not parameters or any(parameter.dtype != torch.float32 for parameter in parameters):
        raise RuntimeError("reference requires FP32 adapter masters over BF16 base compute")
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    completed = 0
    if args.resume:
        # RNG APIs require CPU ByteTensors; optimizer.load_state_dict moves
        # its tensors to the already-resident parameter devices.
        saved = torch.load(args.resume, map_location="cpu", weights_only=False)
        result = peft.set_peft_model_state_dict(model, saved["adapter"])
        if result.unexpected_keys: raise ValueError("resume checkpoint contains unexpected adapter tensors")
        optimizer.load_state_dict(saved["optimizer"])
        random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"]); torch.set_rng_state(saved["torch_rng"])
        if torch.cuda.is_available(): torch.cuda.set_rng_state_all(saved["cuda_rng"])
        completed = int(saved["completed_steps"])

    rows = PreparedDataset(args.prepared_dir); prepared_manifest = rows.manifest
    if not rows: raise ValueError("prepared input is empty")
    if prepared_manifest["max_negatives"] != args.max_negatives:
        raise ValueError("prepared max_negatives does not match the job")
    total = args.total_steps or (completed + args.steps)
    traces, timings = [], []
    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
    for _ in range(args.steps):
        batch = [rows[index] for index in batch_indices(len(rows), args.seed, completed, args.logical_batch_size)]
        if len({row["query_id"] for row in batch}) != len(batch) or len({row["query_content_sha256"] for row in batch}) != len(batch):
            raise ValueError("duplicate contrastive query in logical batch")
        query_ids = [row["query_ids"] for row in batch]
        documents, positives = [], []
        def add_candidate(document):
            for index, candidate in enumerate(documents):
                if candidate["id"] == document["id"] and candidate["content_sha256"] != document["content_sha256"]:
                    raise ValueError("candidate ID maps to conflicting content")
                if candidate["content_sha256"] == document["content_sha256"]:
                    if candidate["input_ids"] != document["input_ids"]:
                        raise ValueError("candidate content maps to conflicting token IDs")
                    candidate["_aliases"].add(document["id"])
                    return index
            candidate = dict(document)
            candidate["_aliases"] = {document["id"]}
            documents.append(candidate)
            return len(documents) - 1
        # Native batching fixes every positive index before appending the
        # deduplicated hard-negative tail.
        for row in batch:
            positives.append(add_candidate(row["positive"]))
        for row in batch:
            for document in row.get("negatives", [])[:args.max_negatives]: add_candidate(document)
        document_ids = [document["input_ids"] for document in documents]
        q_inputs = pad_ids(query_ids, args.device); d_inputs = pad_ids(document_ids, args.device)
        allowed = torch.ones((len(batch), len(document_ids)), dtype=torch.bool, device=args.device)
        for query_index, row in enumerate(batch):
            known = set(row.get("known_positive_ids", []))
            for candidate_index, candidate in enumerate(documents):
                if candidate_index != positives[query_index] and known.intersection(candidate["_aliases"]): allowed[query_index, candidate_index] = False
        if torch.cuda.is_available(): torch.cuda.synchronize()
        observed={"queries":len(batch),"candidates":len(document_ids),
                  "query_token_min":min(map(len,query_ids)),"query_token_max":max(map(len,query_ids)),
                  "document_token_min":min(map(len,document_ids)),"document_token_max":max(map(len,document_ids))}
        if args.paired_control_dir:
            args.paired_control_dir.mkdir(parents=True, exist_ok=True)
            grant = args.paired_control_dir / f"grant-{completed:08d}"
            print(json.dumps({"event":"ready","step":completed,**observed}), flush=True)
            while not grant.exists(): time.sleep(0.001)
            grant.unlink()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        q = gradcache_representations(model, q_inputs, args.microbatch_size, args.precision)
        d = gradcache_representations(model, d_inputs, args.microbatch_size, args.precision)
        loss = mrl_loss(q, d, torch.tensor(positives, device=args.device), allowed, 0.05)
        loss.backward()
        q_gradient, d_gradient = q.grad.detach(), d.grad.detach()
        gradcache_replay(model, q_inputs, q_gradient, args.microbatch_size, args.precision)
        gradcache_replay(model, d_inputs, d_gradient, args.microbatch_size, args.precision)
        preclip_gradients = {}
        if args.trace_full or args.trace_tensors_dir:
            preclip_gradients = {canonical_adapter_name(name): value.grad.detach().clone()
                                 for name,value in model.named_parameters() if value.requires_grad}
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        lr = learning_rate(completed, total)
        for group in optimizer.param_groups: group["lr"] = lr
        optimizer.step(); completed += 1
        if torch.cuda.is_available(): torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000
        timings.append(elapsed_ms)
        loss_value = float(loss.detach())
        if args.paired_control_dir:
            print(json.dumps({"event":"step","step":completed - 1,"loss":loss_value,**observed,
                              "step_ms":elapsed_ms,"trace_mode":bool(args.trace_full or args.trace_tensors_dir)}), flush=True)
            continuation = args.paired_control_dir / f"continue-{completed - 1:08d}"
            while not continuation.exists(): time.sleep(0.001)
            continuation.unlink()
        gradients = {}; adapter_parameters = {}; trace_tensors = {}
        embedding_entry = {}
        if args.trace_full or args.trace_tensors_dir:
            embedding = torch.cat((q.detach(), d.detach()), dim=0).float().cpu().contiguous()
            for name, value in model.named_parameters():
                if value.requires_grad:
                    canonical = canonical_adapter_name(name)
                    gradient = preclip_gradients[canonical].float().cpu().contiguous()
                    parameter = value.detach().float().cpu().contiguous()
                    if args.trace_full:
                        gradients[canonical] = gradient.flatten().tolist(); adapter_parameters[canonical] = parameter.flatten().tolist()
                    else:
                        trace_tensors[f"gradient.{canonical}"] = gradient; trace_tensors[f"parameter.{canonical}"] = parameter
            embedding_entry = {"pooled": embedding.tolist()}
        if args.trace_tensors_dir:
            from safetensors.torch import save_file
            args.trace_tensors_dir.mkdir(parents=True, exist_ok=True)
            trace_tensors["embedding.pooled"] = embedding.contiguous()
            trace_path = args.trace_tensors_dir / f"step-{completed - 1:06d}.safetensors"
            save_file(trace_tensors, str(trace_path), metadata={"format":"pt","precision":args.precision})
            gradients = {name.removeprefix("gradient."): {"file":str(trace_path),"tensor":name} for name in trace_tensors if name.startswith("gradient.")}
            adapter_parameters = {name.removeprefix("parameter."): {"file":str(trace_path),"tensor":name} for name in trace_tensors if name.startswith("parameter.")}
            embedding_entry = {"pooled":{"file":str(trace_path),"tensor":"embedding.pooled"}}
        traces.append({"step": completed - 1, "loss": loss_value, "learning_rate": lr,
                       "embedding": embedding_entry, "gradients": gradients,
                       "adapter_parameters": adapter_parameters})

    checkpoint = args.output.with_suffix(".checkpoint.pt")
    torch.save({"adapter": peft.get_peft_model_state_dict(model), "optimizer": optimizer.state_dict(), "completed_steps": completed,
                "python_rng": random.getstate(), "numpy_rng": np.random.get_state(), "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}, checkpoint)
    report = {"schema": "antfly.embedding_gemma2.finetune_reference.v1", "steps": traces,
              "resume": {"checkpoint": str(checkpoint), "next_step_state_sha256": state_hash(model, optimizer),
                         "uninterrupted_next_step_sha256": None, "resumed_next_step_sha256": None},
              "cells": {"reference": {"step_ms": timings,
                         "peak_device_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                         "peak_host_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                         "trace_mode": args.trace_full or args.trace_tensors_dir is not None}},
              "provenance": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
                             "precision": args.precision, "device": str(args.device), "targets": targets,
                             "attention": args.attention,
                             "loaded_attention": getattr(base.config, "_attn_implementation", None),
                             "adamw_foreach": optimizer.defaults.get("foreach"), "adamw_fused": optimizer.defaults.get("fused"),
                             "clip_grad_foreach": None,
                             "gradient_trace_stage": "pre-clip",
                             "base_weight_sha256": actual_weight_sha,
                             "prepared_manifest_sha256": file_sha256(args.prepared_dir / "manifest.json"),
                             "preprocessing_sha256": prepared_manifest["preprocessing_sha256"],
                             "padding_policy": "per-micro-active-v1",
                             "compute_profile": "bf16-edges-fp32-masters-v1",
                             "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
                             "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                             "schedule": "embedding_gemma2_job.zig/completed-step-v1", "seed": args.seed,
                             "logical_batch_size": args.logical_batch_size, "microbatch_size": args.microbatch_size,
                             "max_negatives": args.max_negatives, "rank": args.rank, "alpha": args.alpha}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    rows.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
