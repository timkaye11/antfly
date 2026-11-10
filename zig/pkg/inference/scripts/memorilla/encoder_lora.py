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

"""Research-only encoder LoRA + 16-slot reader experiment; never a serving artifact.

Decoder weights stay frozen but its forward remains differentiable during
training. No source text is passed to the decoder. The frozen encoder control
uses the same adapter architecture, initialized to zero effect and left frozen.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

SYSTEM = "Answer using the supplied memory. Return only the complete exact name or code, preserving case, hyphens and leading zeros. If the requested fact is missing, return UNKNOWN."
SCHEMA = "antfly.memorilla.encoder-lora-research.v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_checkpoint(path, state):
    temporary = path.with_suffix(".partial")
    with temporary.open("wb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def continuation_contract(parent, current, completed_step):
    """An explicit larger-budget phase; ordinary resume still requires equality."""
    for field in (
        "schema",
        "arm",
        "memory_slots",
        "rank",
        "alpha",
        "target_modules",
        "seed",
        "gradient_accumulation",
        "reader_lr",
        "encoder_lr",
        "dropout",
        "precision",
        "dataset_sha256",
        "system_prompt",
        "decoder_frozen",
        "source_text_in_decoder",
        "heldout_opened",
        "rollout_qualified",
    ):
        if parent[field] != current[field]:
            raise ValueError("continuation changes training contract: " + field)
    if parent["versions"]["torch"] != current["versions"]["torch"]:
        raise ValueError("continuation changes PyTorch version")
    if completed_step != parent["steps"] or current["steps"] <= completed_step:
        raise ValueError("continuation requires a completed shorter budget")


def data_rows(seed, split, groups):
    """Related counterfactuals stay together; all values are split-disjoint."""
    rng = random.Random(f"memorilla-compression-{seed}-{split}")
    rows = []
    used = set()

    def word():
        while True:
            value = "".join(
                rng.choice("bcdfghjklmnprstvz") + rng.choice("aeiou") for _ in range(3)
            ).capitalize()
            if value not in used:
                used.add(value)
                return value

    def code():
        while True:
            value = (
                rng.choice("BCDFGHJKLMNPQRSTVWXYZ")
                + rng.choice("abcdefghijkmnpqrstuvwxyz")
                + str(rng.randrange(2, 10))
                + "-00"
                + str(rng.randrange(1000, 10000))
            )
            if value not in used:
                used.add(value)
                return value

    for group in range(groups):
        site, other = word() + " Depot", word() + " Depot"
        owner = word() + " " + word()
        distractor = word() + " " + word()
        first, second, wrong = code(), code(), code()
        template = group % 3 if split != "ood" else 3

        def documents(value):
            facts = [
                f"The access register at {site} assigns its brass key to {owner}. Its access code is {value}.",
                f"At {site}, the brass-key holder is {owner}. The code for this depot is {value}.",
                f"Depot: {site}. Brass key owner: {owner}. Access code: {value}.",
                f"{owner} is entrusted with the brass key for {site}, whose access code is recorded as {value}.",
            ]
            return [
                facts[template],
                f"The brass key at {other} belongs to {distractor}. The code at {other} is {wrong}.",
            ]

        for suffix, question, value, answer, category in [
            (
                "code-a",
                f"What is the exact access code for {site}?",
                first,
                first,
                "code",
            ),
            (
                "code-b",
                f"What is the exact access code for {site}?",
                second,
                second,
                "code",
            ),
            ("owner", f"Who holds the brass key at {site}?", first, owner, "name"),
            (
                "missing",
                f"What is the exact access code for {site}?"
                if group % 2 == 0
                else f"Who holds the brass key at {site}?",
                first,
                "UNKNOWN",
                "missing",
            ),
        ]:
            docs = documents(value)
            if category == "missing":
                docs[0] = (
                    f"At {site}, the brass key holder is {owner}. Its access code is not recorded."
                    if group % 2 == 0
                    else f"At {site}, the access code is {first}. Its brass key holder is not recorded."
                )
            rows.append(
                {
                    "id": f"{split}-{group:04d}-{suffix}",
                    "group": f"{split}-{group:04d}",
                    "template": template,
                    "question": question,
                    "documents": docs,
                    "answer": answer,
                    "category": category,
                    "entities": [site, other, owner, distractor],
                }
            )
    return rows


def make_data(out, seed=104729, train_groups=128, eval_groups=8):
    out.mkdir(parents=True, exist_ok=False)
    splits = {
        s: data_rows(seed, s, train_groups if s == "train" else eval_groups)
        for s in ("train", "dev", "test", "ood")
    }
    values = {
        s: {r["answer"] for r in rows if r["category"] != "missing"}
        for s, rows in splits.items()
    }
    documents = {
        s: {d for r in rows for d in r["documents"]} for s, rows in splits.items()
    }
    entities = {
        s: {e for r in rows for e in r["entities"]} for s, rows in splits.items()
    }
    for i, a in enumerate(splits):
        for b in list(splits)[i + 1 :]:
            if (
                values[a] & values[b]
                or documents[a] & documents[b]
                or entities[a] & entities[b]
            ):
                raise ValueError("split collision; choose a new seed")
    manifest = {
        "schema": SCHEMA,
        "seed": seed,
        "development_only": True,
        "rollout_qualified": False,
        "split_policy": "Counterfactual groups, documents, non-UNKNOWN answers and entities disjoint; OOD additionally holds out phrasing",
        "files": {},
    }
    for split, rows in splits.items():
        path = out / f"{split}.json"
        save_json(path, {"schema": SCHEMA, "split": split, "cases": rows})
        manifest["files"][split] = {
            "sha256": digest(path),
            "cases": len(rows),
            "groups": len({r["group"] for r in rows}),
        }
    save_json(out / "manifest.json", manifest)
    return manifest


def load_split(directory, name):
    manifest = json.loads((directory / "manifest.json").read_text())
    path = directory / f"{name}.json"
    if (
        manifest["schema"] != SCHEMA
        or digest(path) != manifest["files"][name]["sha256"]
    ):
        raise ValueError("dataset manifest mismatch")
    value = json.loads(path.read_text())
    if (
        value["split"] != name
        or len(value["cases"]) != manifest["files"][name]["cases"]
    ):
        raise ValueError("dataset split mismatch")
    return value["cases"]


class LoRALinear(nn.Module):
    def __init__(self, base, rank=8, alpha=16):
        super().__init__()
        if not isinstance(base, nn.Linear) or not 1 <= rank <= min(
            base.in_features, base.out_features
        ):
            raise ValueError("invalid linear adapter")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.A = nn.Parameter(
            torch.empty(
                rank, base.in_features, device=base.weight.device, dtype=torch.float32
            )
        )
        self.B = nn.Parameter(
            torch.zeros(
                base.out_features, rank, device=base.weight.device, dtype=torch.float32
            )
        )
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        original = self.base(x)
        update = F.linear(F.linear(x.to(self.A.dtype), self.A), self.B)
        return original + (update * (self.alpha / self.rank)).to(original.dtype)


def inject_lora(encoder, rank, alpha, trainable):
    adapters = {}
    for name, module in list(encoder.named_modules()):
        if name.rsplit(".", 1)[-1] not in {"q_proj", "v_proj"}:
            continue
        parent_name, leaf = name.rsplit(".", 1)
        adapter = LoRALinear(module, rank, alpha)
        adapter.A.requires_grad_(trainable)
        adapter.B.requires_grad_(trainable)
        setattr(encoder.get_submodule(parent_name), leaf, adapter)
        adapters[name] = adapter
    if not adapters:
        raise ValueError("no encoder Q/V projection adapters installed")
    return adapters


def adapter_state(adapters):
    return {
        name + "." + key: getattr(module, key).detach().cpu().clone()
        for name, module in adapters.items()
        for key in ("A", "B")
    }


def restore_adapters(adapters, state):
    expected = {name + "." + key for name in adapters for key in ("A", "B")}
    if set(state) != expected:
        raise ValueError("adapter checkpoint geometry differs")
    with torch.no_grad():
        for name, module in adapters.items():
            for key in ("A", "B"):
                getattr(module, key).copy_(state[name + "." + key])


def order_index(seed, absolute, count):
    epoch, index = divmod(absolute, count)
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order[index]


def teacher_inputs(prefix, embedding, target_ids):
    """Only answer positions contribute loss, including its terminating EOS."""
    if prefix.ndim != 2 or target_ids.ndim != 1 or target_ids.numel() < 1:
        raise ValueError("invalid teacher-forced inputs")
    return torch.cat([prefix, embedding(target_ids[:-1])], dim=0).unsqueeze(0)


def frozen_versions(*models):
    return [
        (p, p._version, p.data_ptr())
        for m in models
        for p in m.parameters()
        if not p.requires_grad
    ]


def assert_frozen(snapshot):
    if any(
        p.requires_grad
        or p.grad is not None
        or p._version != version
        or p.data_ptr() != ptr
        for p, version, ptr in snapshot
    ):
        raise RuntimeError("a frozen base parameter changed or acquired a gradient")


def scores(rows):
    result = {"total": len(rows), "correct": sum(r["correct"] for r in rows)}
    for category in ("code", "name", "missing"):
        chosen = [r for r in rows if r["category"] == category]
        result[category] = {
            "correct": sum(r["correct"] for r in chosen),
            "total": len(chosen),
        }
    pairs = {}
    for r in rows:
        if r["category"] == "code":
            pairs.setdefault(r["group"], []).append(r)
    result["counterfactual_pairs"] = {
        "correct": sum(
            len(v) == 2
            and all(r["correct"] for r in v)
            and len({r["actual"] for r in v}) == 2
            for v in pairs.values()
        ),
        "total": len(pairs),
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    data = sub.add_parser("make-data")
    data.add_argument("--output", type=Path, required=True)
    data.add_argument("--seed", type=int, default=104729)
    data.add_argument("--train-groups", type=int, default=128)
    data.add_argument("--eval-groups", type=int, default=8)
    train = sub.add_parser("train")
    for name in (
        "encoder",
        "decoder",
        "checkpoint",
        "upstream-source",
        "data",
        "output",
    ):
        train.add_argument("--" + name, type=Path, required=True)
    train.add_argument("--arm", choices=("reader-only", "encoder-lora"), required=True)
    train.add_argument("--steps", type=int, default=128)
    train.add_argument("--accum", type=int, default=4)
    train.add_argument("--eval-every", type=int, default=32)
    train.add_argument("--rank", type=int, default=8)
    train.add_argument("--alpha", type=int, default=16)
    train.add_argument("--reader-lr", type=float, default=5e-5)
    train.add_argument("--encoder-lr", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=23)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--continue-from", type=Path)
    args = p.parse_args()
    if args.command == "make-data":
        if not 1 <= args.train_groups <= 100000 or not 1 <= args.eval_groups <= 10000:
            p.error("bounded nonzero corpus sizes required")
        print(
            json.dumps(
                make_data(args.output, args.seed, args.train_groups, args.eval_groups)
            )
        )
        return
    run(args)


def run(args):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
    from assets import verify_base_model
    from decoder_reference import prepare_prompt
    from encoder_reference import normalize_features
    from gpu_quality_reference import generate
    from reference import load_upstream
    from released_reference import checkpoint_proof, source_recipe

    if (
        not 1 <= args.steps <= 4096
        or not 1 <= args.accum <= 16
        or not 1 <= args.eval_every <= args.steps
    ):
        raise ValueError("invalid bounded training budget")
    if not 0 < args.reader_lr <= 1e-3 or not 0 < args.encoder_lr <= 1e-3:
        raise ValueError("invalid learning rate")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    import subprocess

    occupied = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip()
    if occupied:
        raise RuntimeError("GPU already occupied; refusing a competing training run")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    protocol = {
        "schema": SCHEMA,
        "arm": args.arm,
        "memory_slots": 16,
        "rank": args.rank,
        "alpha": args.alpha,
        "target_modules": ["q_proj", "v_proj"],
        "seed": args.seed,
        "steps": args.steps,
        "gradient_accumulation": args.accum,
        "reader_lr": args.reader_lr,
        "encoder_lr": args.encoder_lr,
        "dropout": 0,
        "precision": "BF16 encoder/decoder SDPA; FP32 trainable weights; FP16->BF16 feature channel",
        "dataset_sha256": digest(args.data / "manifest.json"),
        "script_sha256": digest(Path(__file__)),
        "system_prompt": SYSTEM,
        "decoder_frozen": True,
        "source_text_in_decoder": False,
        "heldout_opened": False,
        "rollout_qualified": False,
        "versions": {"torch": str(torch.__version__)},
    }
    protocol["eval_every"] = args.eval_every
    if args.resume:
        previous_protocol = json.loads((args.output / "protocol.json").read_text())
        if "continuation" in previous_protocol:
            protocol["continuation"] = previous_protocol["continuation"]
            if (
                args.continue_from is not None
                and str(args.continue_from.resolve())
                != protocol["continuation"]["directory"]
            ):
                raise ValueError("resume changes parent run")
    elif args.continue_from is not None:
        parent = args.continue_from.resolve()
        parent_protocol = json.loads((parent / "protocol.json").read_text())
        parent_result = json.loads((parent / "result.json").read_text())
        if not parent_result["completed"]:
            raise ValueError("parent run did not complete")
        completed_step = parent_result["history"][-1]["step"]
        continuation_contract(parent_protocol, protocol, completed_step)
        protocol["continuation"] = {
            "directory": str(parent),
            "step": completed_step,
            "protocol_sha256": digest(parent / "protocol.json"),
            "checkpoint_sha256": digest(parent / "resume.pt"),
        }
    if (
        args.resume
        and json.loads((args.output / "protocol.json").read_text()) != protocol
    ):
        raise ValueError("resume protocol differs; preserve run and choose new output")
    save_json(args.output / "protocol.json", protocol)
    if not args.resume:
        (args.output / "harness_source.py").write_bytes(Path(__file__).read_bytes())
        if digest(args.output / "harness_source.py") != protocol["script_sha256"]:
            raise ValueError("trainer changed during source archival")
    config, reader_proof = checkpoint_proof(args.checkpoint)
    if config["num_memories"] != 16:
        raise ValueError("experiment requires released 16-slot initialization")
    proof = {
        "encoder": verify_base_model(args.encoder, "encoder"),
        "decoder": verify_base_model(args.decoder, "decoder"),
        "reader": reader_proof,
        "upstream": source_recipe(args.upstream_source),
    }
    save_json(args.output / "base-proofs.json", proof)
    train_rows, dev_rows = load_split(args.data, "train"), load_split(args.data, "dev")
    if len({r["group"] for r in train_rows} & {r["group"] for r in dev_rows}):
        raise ValueError("train/dev group overlap")
    tokenizer_e = AutoTokenizer.from_pretrained(
        args.encoder, local_files_only=True, trust_remote_code=False
    )
    tokenizer_e.padding_side = "right"
    encoder = (
        AutoModel.from_pretrained(
            args.encoder,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .requires_grad_(False)
        .cuda()
    )
    adapters = inject_lora(encoder, args.rank, args.alpha, args.arm == "encoder-lora")
    encoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    weights = torch.load(
        args.checkpoint / "memory.pt", map_location="cpu", weights_only=True, mmap=True
    )
    reader = (
        load_upstream(args.upstream_source, config, weights)
        .float()
        .cuda()
        .requires_grad_(True)
    )
    del weights
    for module in reader.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0
        if isinstance(module, nn.MultiheadAttention):
            module.dropout = 0
    tokenizer_d = AutoTokenizer.from_pretrained(
        args.decoder, local_files_only=True, trust_remote_code=False
    )
    decoder = (
        AutoModelForCausalLM.from_pretrained(
            args.decoder,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .requires_grad_(False)
        .cuda()
    )
    decoder.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    decoder.config.use_cache = False
    if decoder.config.vocab_size != 151936:
        raise ValueError("decoder vocabulary must remain unchanged")
    groups = [{"params": list(reader.parameters()), "lr": args.reader_lr}]
    if args.arm == "encoder-lora":
        groups.append(
            {
                "params": [p for p in encoder.parameters() if p.requires_grad],
                "lr": args.encoder_lr,
            }
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=0)
    trainable = [p for group in optimizer.param_groups for p in group["params"]]
    frozen = frozen_versions(encoder, decoder)
    result_path = args.output / "result.json"
    result = {
        "completed": False,
        "rollout_qualified": False,
        "arm": args.arm,
        "history": [],
        "evaluations": [],
        "reader_parameters": sum(p.numel() for p in reader.parameters()),
        "encoder_trainable_parameters": sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        ),
        "encoder_base_and_decoder_frozen": True,
    }
    start_step = 0
    if args.resume or args.continue_from is not None:
        resume_path = (
            args.output / "resume.pt"
            if args.resume
            else args.continue_from / "resume.pt"
        )
        if (
            not args.resume
            and digest(resume_path) != protocol["continuation"]["checkpoint_sha256"]
        ):
            raise ValueError("parent checkpoint changed")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=True)
        if args.resume and checkpoint["protocol"] != protocol:
            raise ValueError("checkpoint protocol differs")
        if not args.resume:
            continuation_contract(checkpoint["protocol"], protocol, checkpoint["step"])
        reader.load_state_dict(checkpoint["reader"])
        restore_adapters(adapters, checkpoint["adapters"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["cpu_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        start_step = checkpoint["step"]
        result = checkpoint["result"]
        result["completed"] = False
        result.pop("frozen_base_versions_unchanged", None)
        result.pop("error", None)
        result["phase_start_step"] = start_step
        del checkpoint
        # Restoring reader/adapters does not touch frozen base parameters.
    frozen = frozen_versions(encoder, decoder)
    save_json(result_path, result)

    def modes(training):
        encoder.train(training)
        reader.train(training)
        decoder.train(training)

    def prefix(case, training):
        tokens = tokenizer_e(
            case["documents"] + [case["question"]], padding=True, return_tensors="pt"
        ).to("cuda")
        if tokens.input_ids.shape[1] > 128:
            raise ValueError(
                "encoder input exceeds explicit 128-token experiment budget"
            )
        context = (
            contextlib.nullcontext()
            if training and args.arm == "encoder-lora"
            else torch.no_grad()
        )
        with context:
            states = encoder(**tokens, use_cache=False).last_hidden_state
            features = (
                normalize_features(states, tokens.attention_mask)
                .half()
                .bfloat16()
                .float()
            )
        docs, query = features[:-1].unsqueeze(0), features[-1:].contiguous()
        mask = torch.zeros((1, docs.shape[1]), dtype=torch.bool, device="cuda")
        memories = reader(docs, mask, query)[0]
        if memories.shape != (16, 4096):
            raise ValueError("reader memory geometry changed")
        _, embeds, _ = prepare_prompt(
            tokenizer_d,
            decoder.get_input_embeddings(),
            memories,
            system_prompt=SYSTEM,
            question=case["question"],
            device="cuda",
        )
        return embeds

    def evaluate(step):
        modes(False)
        outputs = []
        started = time.monotonic()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for case in dev_rows:
                answer = generate(decoder, tokenizer_d, prefix(case, False), 24)
                outputs.append(
                    {
                        "id": case["id"],
                        "group": case["group"],
                        "category": case["category"],
                        "actual": answer["answer"],
                        "expected": case["answer"],
                        "correct": answer["answer"].strip() == case["answer"],
                        "finish_reason": answer["finish_reason"],
                    }
                )
        evaluation = {
            "step": step,
            "split": "dev",
            "scores": scores(outputs),
            "seconds": time.monotonic() - started,
            "rows": outputs,
        }
        result["evaluations"].append(evaluation)
        save_json(result_path, result)
        print(
            json.dumps(
                {"event": "evaluation", "step": step, "scores": evaluation["scores"]}
            ),
            flush=True,
        )
        assert_frozen(frozen)
        modes(True)

    def checkpoint(step):
        assert_frozen(frozen)
        state = {
            "protocol": protocol,
            "step": step,
            "reader": {k: v.detach().cpu() for k, v in reader.state_dict().items()},
            "adapters": adapter_state(adapters),
            "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "result": result,
        }
        atomic_checkpoint(args.output / "resume.pt", state)

    try:
        if not args.resume:
            expected_rows = result["evaluations"][-1]["rows"] if start_step else None
            evaluate(start_step)
            if expected_rows is not None:
                previous_answers = {r["id"]: r["actual"] for r in expected_rows}
                if any(
                    r["actual"] != previous_answers[r["id"]]
                    for r in result["evaluations"][-1]["rows"]
                ):
                    raise ValueError(
                        "continuation does not reproduce parent development answers"
                    )
        for step in range(start_step, args.steps):
            modes(True)
            optimizer.zero_grad(set_to_none=True)
            started, total_loss = time.monotonic(), 0.0
            for micro in range(args.accum):
                case = train_rows[
                    order_index(args.seed, step * args.accum + micro, len(train_rows))
                ]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    embeddings = prefix(case, True)
                    ids = tokenizer_d(
                        case["answer"], add_special_tokens=False
                    ).input_ids + [151645]
                    target = torch.tensor(ids, device="cuda", dtype=torch.long)
                    if target.numel() > 24:
                        raise ValueError("answer exceeds experiment generation cap")
                    inputs = teacher_inputs(
                        embeddings, decoder.get_input_embeddings(), target
                    )
                    output = decoder(
                        inputs_embeds=inputs, use_cache=False, logits_to_keep=len(ids)
                    )
                    loss = F.cross_entropy(output.logits[0].float(), target)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                total_loss += float(loss.detach()) / args.accum
                (loss / args.accum).backward()
                del embeddings, inputs, output, loss
            reader_grad = (
                sum(
                    float(p.grad.detach().float().square().sum())
                    for p in reader.parameters()
                    if p.grad is not None
                )
                ** 0.5
            )
            lora_grad = (
                sum(
                    float(p.grad.detach().float().square().sum())
                    for p in encoder.parameters()
                    if p.requires_grad and p.grad is not None
                )
                ** 0.5
            )
            if (
                not math.isfinite(reader_grad)
                or reader_grad == 0
                or args.arm == "encoder-lora"
                and (not math.isfinite(lora_grad) or lora_grad == 0)
            ):
                raise RuntimeError("missing/nonfinite reader or encoder gradient")
            assert_frozen(frozen)
            norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            )
            optimizer.step()
            torch.cuda.synchronize()
            row = {
                "step": step + 1,
                "loss": total_loss,
                "reader_grad_norm": reader_grad,
                "encoder_lora_grad_norm": lora_grad,
                "total_grad_norm": norm,
                "seconds": time.monotonic() - started,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            result["history"].append(row)
            save_json(result_path, result)
            print(json.dumps({"event": "step", **row}), flush=True)
            if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
                evaluate(step + 1)
                checkpoint(step + 1)
        result["completed"] = True
        result["frozen_base_versions_unchanged"] = True
        save_json(result_path, result)
        checkpoint(args.steps)
        from safetensors.torch import save_file

        save_file(
            {k: v.detach().cpu().contiguous() for k, v in reader.state_dict().items()},
            str(args.output / "reader.safetensors"),
        )
        save_file(
            adapter_state(adapters), str(args.output / "encoder-lora.safetensors")
        )
        save_json(
            args.output / "candidate.json",
            {
                "schema": SCHEMA,
                "rollout_qualified": False,
                "experimental_only": True,
                "memory_slots": 16,
                "base_proofs_sha256": digest(args.output / "base-proofs.json"),
                "reader_sha256": digest(args.output / "reader.safetensors"),
                "encoder_lora_sha256": digest(args.output / "encoder-lora.safetensors"),
                "protocol_sha256": digest(args.output / "protocol.json"),
            },
        )
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save_json(result_path, result)
        raise


if __name__ == "__main__":
    main()
