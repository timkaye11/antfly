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

"""Reload research candidates and distinguish training fit from dev generalization.

Existing test/ood splits stay sealed. Optional post-freeze challenges use fresh
facts generated only after all candidate hashes have been checked. Neither expected answers nor category labels enter encoder inputs
or decoder prompts. This is a Python research diagnostic, not serving promotion.
"""

from __future__ import annotations

import argparse
import json
import secrets
import subprocess
from pathlib import Path

import torch
from safetensors.torch import load_file

from assets import verify_base_model
from decoder_reference import prepare_prompt
from encoder_reference import normalize_features
from encoder_lora import (
    SYSTEM,
    data_rows,
    digest,
    inject_lora,
    load_split,
    restore_adapters,
    save_json,
    scores,
)
from encoder_lora_corpus import inventories
from gpu_quality_reference import generate
from reference import load_upstream
from released_reference import checkpoint_proof, source_recipe


def fresh_challenge(seed, groups, train, dev):
    # Reject colliding groups individually so large qualification suites remain
    # feasible even when training contains many thousands of invented entities.
    for attempt in range(128):
        effective = seed + attempt * 65537
        reserved = [a | b for a, b in zip(inventories(train), inventories(dev))]
        rows = []
        complete = True
        for label, source in [
            ("postfreeze", "postfreeze"),
            ("postfreeze-phrasing", "ood"),
        ]:
            pool = data_rows(effective, source, max(groups * 2, 16))
            accepted = 0
            for offset in range(0, len(pool), 4):
                group = pool[offset : offset + 4]
                first, second = group[:2]
                old = second["answer"]
                new = first["answer"][:-1] + str((int(first["answer"][-1]) + 1) % 10)
                second["answer"] = new
                second["documents"] = [d.replace(old, new) for d in second["documents"]]
                own = inventories(group)
                if any(a & b for a, b in zip(own, reserved)):
                    continue
                for a, b in zip(own, reserved):
                    b.update(a)
                for row in group:
                    row["id"] = label + ":" + row["id"]
                    row["group"] = label + ":" + row["group"]
                    row["cohort"] = label
                rows.extend(group)
                accepted += 1
                if accepted == groups:
                    break
            if accepted != groups:
                complete = False
                break
        if complete:
            return effective, rows
    raise ValueError("could not generate disjoint post-freeze facts")


def main():
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "encoder",
        "decoder",
        "checkpoint",
        "upstream-source",
        "data",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--candidate", type=Path, action="append", required=True)
    p.add_argument("--fresh-groups", type=int, default=0)
    p.add_argument("--fresh-seed", type=int)
    args = p.parse_args()
    if not 0 <= args.fresh_groups <= 1000:
        p.error("fresh groups must be between zero and 1000")
    if not 1 <= len(args.candidate) <= 2:
        p.error("one or two candidates required")
    if subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True
    ).strip():
        raise RuntimeError("GPU busy")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(23)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config, _ = checkpoint_proof(args.checkpoint)
    proofs = {
        "encoder": verify_base_model(args.encoder, "encoder"),
        "decoder": verify_base_model(args.decoder, "decoder"),
        "upstream": source_recipe(args.upstream_source),
    }
    save_json(args.output / "base-proofs.json", proofs)
    protocols = []
    for directory in args.candidate:
        candidate = json.loads((directory / "candidate.json").read_text())
        protocol = json.loads((directory / "protocol.json").read_text())
        if (
            candidate["rollout_qualified"]
            or candidate["memory_slots"] != 16
            or protocol["memory_slots"] != 16
        ):
            raise ValueError("not an unqualified 16-slot research candidate")
        for field, file in [
            ("reader_sha256", "reader.safetensors"),
            ("encoder_lora_sha256", "encoder-lora.safetensors"),
            ("protocol_sha256", "protocol.json"),
            ("base_proofs_sha256", "base-proofs.json"),
        ]:
            if candidate[field] != digest(directory / file):
                raise ValueError("candidate artifact changed")
        if (
            protocol["dataset_sha256"] != digest(args.data / "manifest.json")
            or protocol["system_prompt"] != SYSTEM
        ):
            raise ValueError("candidate/data/prompt mismatch")
        if not json.loads((directory / "result.json").read_text())["completed"]:
            raise ValueError("candidate training did not complete")
        protocols.append(protocol)
    if any(
        (a["rank"], a["alpha"], a["seed"], a["precision"])
        != (
            protocols[0]["rank"],
            protocols[0]["alpha"],
            protocols[0]["seed"],
            protocols[0]["precision"],
        )
        for a in protocols
    ):
        raise ValueError("incompatible candidate protocols")
    dev = load_split(args.data, "dev")
    train = load_split(args.data, "train")
    groups = list(dict.fromkeys(r["group"] for r in train))[:8]
    train_probe = [r for r in train if r["group"] in groups]
    fresh_rows = []
    if args.fresh_groups:
        seed, fresh_rows = fresh_challenge(
            args.fresh_seed if args.fresh_seed is not None else secrets.randbits(63),
            args.fresh_groups,
            train,
            dev,
        )
        save_json(
            args.output / "fresh-cases.json",
            {
                "seed": seed,
                "generated_after_candidate_hash_checks": True,
                "candidate_manifest_hashes": [
                    digest(d / "candidate.json") for d in args.candidate
                ],
                "candidate_protocol_hashes": [
                    digest(d / "protocol.json") for d in args.candidate
                ],
                "dataset_sha256": digest(args.data / "manifest.json"),
                "cases": fresh_rows,
            },
        )
    enc_tok = AutoTokenizer.from_pretrained(args.encoder, local_files_only=True)
    enc_tok.padding_side = "right"
    dec_tok = AutoTokenizer.from_pretrained(args.decoder, local_files_only=True)
    encoder = (
        AutoModel.from_pretrained(
            args.encoder,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .requires_grad_(False)
        .cuda()
        .eval()
    )
    adapters = inject_lora(encoder, protocols[0]["rank"], protocols[0]["alpha"], False)
    decoder = (
        AutoModelForCausalLM.from_pretrained(
            args.decoder,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .requires_grad_(False)
        .cuda()
        .eval()
    )
    result = {
        "schema": "antfly.memorilla.encoder-lora-reload-audit.v1",
        "completed": False,
        "rollout_qualified": False,
        "heldout_opened": bool(fresh_rows),
        "old_qualification_splits_opened": False,
        "postfreeze_facts_generated": len(fresh_rows),
        "source_sha256": digest(Path(__file__)),
        "candidates": [],
    }
    save_json(args.output / "result.json", result)
    for directory, protocol in zip(args.candidate, protocols):
        restore_adapters(
            adapters, load_file(str(directory / "encoder-lora.safetensors"))
        )
        reader = (
            load_upstream(
                args.upstream_source,
                config,
                load_file(str(directory / "reader.safetensors")),
            )
            .float()
            .cuda()
            .requires_grad_(False)
            .eval()
        )
        entry = {
            "arm": protocol["arm"],
            "candidate": directory.name,
            "candidate_sha256": digest(directory / "candidate.json"),
            "reconstruct_every": protocol.get("reconstruct_every", 0),
            "protocol_sha256": digest(directory / "protocol.json"),
            "splits": {},
        }
        result["candidates"].append(entry)
        for split, cases in [("dev", dev), ("train-probe", train_probe)] + (
            [("postfreeze", fresh_rows)] if fresh_rows else []
        ):
            rows, code_vectors = [], {}
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for case in cases:
                    tokens = enc_tok(
                        case["documents"] + [case["question"]],
                        padding=True,
                        return_tensors="pt",
                    ).to("cuda")
                    if tokens.input_ids.shape[1] > 128:
                        raise ValueError("encoder bound exceeded")
                    features = (
                        normalize_features(
                            encoder(**tokens, use_cache=False).last_hidden_state,
                            tokens.attention_mask,
                        )
                        .half()
                        .bfloat16()
                        .float()
                    )
                    docs, query = features[:-1].unsqueeze(0), features[-1:]
                    memories = reader(
                        docs,
                        torch.zeros(
                            (1, docs.shape[1]), dtype=torch.bool, device="cuda"
                        ),
                        query,
                    )[0]
                    if tuple(memories.shape) != (16, 4096):
                        raise ValueError("candidate changed memory geometry")
                    _, embeds, _ = prepare_prompt(
                        dec_tok,
                        decoder.get_input_embeddings(),
                        memories,
                        system_prompt=SYSTEM,
                        question=case["question"],
                        device="cuda",
                    )
                    output = generate(decoder, dec_tok, embeds, 24)
                    rows.append(
                        {
                            "id": case["id"],
                            "group": case["group"],
                            "category": case["category"],
                            "cohort": case.get("cohort", split),
                            "actual": output["answer"],
                            "expected": case["answer"],
                            "correct": output["answer"].strip() == case["answer"],
                            "finish_reason": output["finish_reason"],
                        }
                    )
                    if case["category"] == "code":
                        code_vectors.setdefault(case["group"], []).append(
                            features[0].float().cpu()
                        )
            pairs = [
                float(torch.nn.functional.cosine_similarity(v[0], v[1], dim=0))
                for v in code_vectors.values()
                if len(v) == 2
            ]
            summary = {
                "scores": scores(rows),
                "rows": rows,
                "code_pair_embedding_cosine_mean": sum(pairs) / len(pairs)
                if pairs
                else None,
            }
            if split == "dev":
                previous = json.loads((directory / "result.json").read_text())[
                    "evaluations"
                ][-1]["rows"]
                expected = {r["id"]: r["actual"] for r in previous}
                summary["reload_text_matches"] = sum(
                    r["actual"] == expected[r["id"]] for r in rows
                )
                if summary["reload_text_matches"] != len(rows):
                    entry["splits"][split] = summary
                    save_json(args.output / "result.json", result)
                    raise ValueError(
                        "reloaded candidate does not reproduce final development outputs"
                    )
            entry["splits"][split] = summary
            save_json(args.output / "result.json", result)
            print(
                json.dumps(
                    {
                        "arm": entry["arm"],
                        "candidate": entry["candidate"],
                        "split": split,
                        **summary["scores"],
                        "code_pair_embedding_cosine_mean": summary[
                            "code_pair_embedding_cosine_mean"
                        ],
                    }
                ),
                flush=True,
            )
        del reader
    result["completed"] = True
    save_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
