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
facts generated only after all candidate hashes have been checked. QA inputs never use expected answers or category labels. An explicit optional
oracle-prefix control uses gold answers, is labeled separately, and never counts
toward QA accuracy or qualification. This is a Python research diagnostic, not serving promotion.
"""

from __future__ import annotations

import argparse
import copy
import json
import secrets
import random
import re
import subprocess
from pathlib import Path

import torch
from safetensors.torch import load_file

from assets import verify_base_model
from decoder_reference import prepare_prompt
from encoder_reference import normalize_features
from encoder_lora import (
    SYSTEM,
    CounterfactualGroupSampler,
    RECONSTRUCTION_SYSTEM,
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


def literal_memory_ids(tokenizer, answer, kind):
    text = " " + answer
    if kind == "characters":
        pieces = [tokenizer.encode(c, add_special_tokens=False) for c in text]
        if any(len(p) != 1 for p in pieces):
            raise ValueError("oracle character does not have a single token ID")
        ids = [p[0] for p in pieces]
    elif kind == "bpe":
        ids = tokenizer.encode(text, add_special_tokens=False)
    else:
        raise ValueError("unknown oracle prefix representation")
    pad = tokenizer.encode(" ", add_special_tokens=False)
    if len(pad) != 1 or len(ids) > 16:
        raise ValueError("oracle prefix does not fit exactly 16 slots")
    return ids + pad * (16 - len(ids))


def readout_prompt(question, format):
    routes = [
        ("What is the exact access code for ", "Code"),
        ("Who holds the brass key at ", "Name"),
    ]
    for prefix, field in routes:
        if question.startswith(prefix) and question.endswith("?"):
            site = question[len(prefix) : -1]
            if not site:
                break
            prompt = f"Reconstruct the brass-key record for {site}. Return exactly two lines: Name: followed by the holder's name; Code: followed by the access code. Use UNKNOWN for an absent field."
            if format == "spelled":
                prompt += " For each present field, spell its value with one space between characters. Separate words in a name with / . Keep UNKNOWN unspelled."
            return field, prompt
    raise ValueError(
        "structured readout supports only the declared name/code question forms"
    )


def parse_readout(answer, field, format):
    lines = answer.strip().splitlines()
    if (
        len(lines) != 2
        or not lines[0].startswith("Name: ")
        or not lines[1].startswith("Code: ")
    ):
        return None
    values = {"Name": lines[0][6:].strip(), "Code": lines[1][6:].strip()}
    value = values[field]
    if value == "UNKNOWN":
        return value
    if format == "spelled":
        parts = value.split(" / ") if field == "Name" else [value]
        pattern = (
            r"[A-Za-z-](?: [A-Za-z-])*"
            if field == "Name"
            else r"[A-Za-z0-9-](?: [A-Za-z0-9-])*"
        )
        if any(re.fullmatch(pattern, part) is None for part in parts):
            return None
        return (" " if field == "Name" else "").join(
            part.replace(" ", "") for part in parts
        )
    pattern = r"[A-Za-z-]+(?: [A-Za-z-]+)*" if field == "Name" else r"[A-Za-z0-9-]+"
    return value if re.fullmatch(pattern, value) else None


def prior_dataset_hashes(protocol):
    hashes, seen = set(), set()
    while "initialization" in protocol:
        initial = protocol["initialization"]
        directory = Path(initial["directory"])
        if str(directory) in seen or len(seen) >= 8:
            raise ValueError("invalid initialization lineage")
        seen.add(str(directory))
        if digest(directory / "candidate.json") != initial["candidate_sha256"]:
            raise ValueError("initialization candidate lineage changed")
        candidate = json.loads((directory / "candidate.json").read_text())
        if digest(directory / "protocol.json") != candidate["protocol_sha256"]:
            raise ValueError("initialization protocol lineage changed")
        protocol = json.loads((directory / "protocol.json").read_text())
        hashes.add(protocol["dataset_sha256"])
    return hashes


def qa_corpus(rows):
    return [{k: v for k, v in r.items() if k != "reconstruction"} for r in rows]


def training_exposures(protocol, count, train=None):
    visits = [{"qa": 0, "reconstruction": 0} for _ in range(count)]
    total = protocol["steps"] * protocol["gradient_accumulation"]
    every = protocol.get("reconstruct_every", 0)
    if protocol.get("sampling", "row") == "counterfactual-group":
        if train is None or len(train) != count:
            raise ValueError("group exposure accounting needs the actual training rows")
        sampler = CounterfactualGroupSampler(train, protocol["seed"])
        for absolute in range(total):
            task = "reconstruction" if every and absolute % every == every - 1 else "qa"
            visits[sampler.index(absolute)][task] += 1
        return visits
    for epoch in range((total + count - 1) // count):
        order = list(range(count))
        random.Random(protocol["seed"] + epoch).shuffle(order)
        for offset, index in enumerate(order[: min(count, total - epoch * count)]):
            absolute = epoch * count + offset
            task = "reconstruction" if every and absolute % every == every - 1 else "qa"
            visits[index][task] += 1
    return visits


def select_training_probe(train, protocol):
    visits = training_exposures(protocol, len(train), train)
    groups = {}
    for index, row in enumerate(train):
        groups.setdefault(row["group"], []).append(index)
    selected = sorted(
        groups,
        key=lambda g: (
            -sum(sum(visits[i].values()) > 0 for i in groups[g]),
            -sum(sum(visits[i].values()) for i in groups[g]),
            g,
        ),
    )[:8]
    return [r for r in train if r["group"] in selected]


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
            alternative_names = data_rows(
                effective, "name-counterfactual-" + source, max(groups * 2, 16)
            )
            accepted = 0
            for offset in range(0, len(pool), 4):
                group = pool[offset : offset + 4]
                first, second = group[:2]
                old = second["answer"]
                new = first["answer"][:-1] + str((int(first["answer"][-1]) + 1) % 10)
                second["answer"] = new
                second["documents"] = [d.replace(old, new) for d in second["documents"]]
                alternative = copy.deepcopy(group[2])
                previous_name = alternative["answer"]
                alternative["answer"] = alternative_names[offset + 2]["answer"]
                if alternative["answer"] in group[2]["entities"]:
                    continue
                alternative["documents"] = [
                    d.replace(previous_name, alternative["answer"])
                    for d in alternative["documents"]
                ]
                alternative["entities"][2] = alternative["answer"]
                alternative["id"] += "-b"
                group.append(alternative)
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
    p.add_argument("--candidate-data", type=Path, action="append")
    p.add_argument("--prior-training-data", type=Path, action="append", default=[])
    p.add_argument("--oracle-prefix-probe", action="store_true")
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument(
        "--oracle-question", choices=("original", "copy-value"), default="original"
    )
    p.add_argument("--readout-format", choices=("fields", "spelled"))
    p.add_argument("--fresh-groups", type=int, default=0)
    p.add_argument("--fresh-seed", type=int)
    args = p.parse_args()
    if args.oracle_only and (not args.oracle_prefix_probe or args.fresh_groups):
        p.error(
            "oracle-only requires the explicit control flag and cannot open fresh qualification"
        )
    if not 0 <= args.fresh_groups <= 1000:
        p.error("fresh groups must be between zero and 1000")
    if args.candidate_data is not None and len(args.candidate_data) != len(
        args.candidate
    ):
        p.error("provide one candidate-data directory per candidate")
    candidate_data = args.candidate_data or [args.data] * len(args.candidate)
    if not 1 <= len(args.candidate) <= 3:
        p.error("one to three candidates required")
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
    canonical = {s: load_split(args.data, s) for s in ("train", "dev")}
    for directory, data_directory in zip(args.candidate, candidate_data):
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
            protocol["dataset_sha256"] != digest(data_directory / "manifest.json")
            or protocol["system_prompt"] != SYSTEM
        ):
            raise ValueError("candidate/data/prompt mismatch")
        if not json.loads((directory / "result.json").read_text())["completed"]:
            raise ValueError("candidate training did not complete")
        if any(
            qa_corpus(load_split(data_directory, split)) != qa_corpus(canonical[split])
            for split in ("train", "dev")
        ):
            raise ValueError(
                "candidate corpora have different QA facts or evaluation inputs"
            )
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
    train_probe = select_training_probe(train, protocols[0])
    fresh_rows = []
    prior_rows, prior_hashes = [], set()
    if args.fresh_groups:
        needed = set().union(*(prior_dataset_hashes(p) for p in protocols)) - {
            p["dataset_sha256"] for p in protocols
        }
        for directory in args.prior_training_data:
            value = digest(directory / "manifest.json")
            if value not in needed:
                raise ValueError(
                    "prior training corpus is not in candidate initialization lineage"
                )
            prior_rows.extend(load_split(directory, "train"))
            prior_hashes.add(value)
        if needed != prior_hashes:
            raise ValueError(
                "fresh challenge requires --prior-training-data for every earlier training corpus"
            )
        seed, fresh_rows = fresh_challenge(
            args.fresh_seed if args.fresh_seed is not None else secrets.randbits(63),
            args.fresh_groups,
            train + prior_rows,
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
                "prior_training_manifest_hashes": sorted(prior_hashes),
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
        "readout_format": args.readout_format,
        "oracle_prefix_probe": args.oracle_prefix_probe,
        "oracle_only": args.oracle_only,
        "oracle_question": args.oracle_question,
        "readout_scope": "Two declared name/code question forms; deterministic parsing, no correction, gold routing or fallback.",
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
        exposures = {
            r["id"]: v
            for r, v in zip(train, training_exposures(protocol, len(train), train))
        }
        entry = {
            "arm": protocol["arm"],
            "candidate": directory.name,
            "candidate_sha256": digest(directory / "candidate.json"),
            "reconstruct_every": protocol.get("reconstruct_every", 0),
            "protocol_sha256": digest(directory / "protocol.json"),
            "training_data_manifest_sha256": protocol["dataset_sha256"],
            "evaluation_data_manifest_sha256": digest(args.data / "manifest.json"),
            "splits": {},
        }
        result["candidates"].append(entry)
        plan = (
            []
            if args.oracle_only
            else (
                [("dev", dev), ("train-probe", train_probe)]
                + ([("dev-readout", dev)] if args.readout_format else [])
                + ([("postfreeze", fresh_rows)] if fresh_rows else [])
            )
        )
        for split, cases in plan:
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
                    structured = args.readout_format is not None and split in (
                        "dev-readout",
                        "postfreeze",
                    )
                    field, decoder_question = (
                        readout_prompt(case["question"], args.readout_format)
                        if structured
                        else (None, case["question"])
                    )
                    _, embeds, _ = prepare_prompt(
                        dec_tok,
                        decoder.get_input_embeddings(),
                        memories,
                        system_prompt=RECONSTRUCTION_SYSTEM if structured else SYSTEM,
                        question=decoder_question,
                        device="cuda",
                    )
                    output = generate(
                        decoder, dec_tok, embeds, 64 if structured else 24
                    )
                    parsed = (
                        (
                            parse_readout(output["answer"], field, args.readout_format)
                            if output["finish_reason"] == "eos"
                            else None
                        )
                        if structured
                        else output["answer"]
                    )
                    actual = parsed if parsed is not None else ""
                    rows.append(
                        {
                            "id": case["id"],
                            "group": case["group"],
                            "category": case["category"],
                            "cohort": case.get("cohort", split),
                            "actual": actual,
                            "raw_answer": output["answer"],
                            "answer_format": args.readout_format
                            if structured
                            else "direct",
                            "parse_valid": parsed is not None,
                            "decoder_seconds": output["decoder_seconds"],
                            "generated_tokens": len(output["token_ids"]),
                            "expected": case["answer"],
                            "correct": actual.strip() == case["answer"],
                            "finish_reason": output["finish_reason"],
                            "training_exposure": exposures[case["id"]]
                            if split == "train-probe"
                            else None,
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
            if split == "postfreeze":
                summary["name_counterfactual_pairs"] = scores(
                    [dict(r, category="code") for r in rows if r["category"] == "name"]
                )["counterfactual_pairs"]
                summary["name_pairs_by_cohort"] = {
                    cohort: scores(
                        [
                            dict(r, category="code")
                            for r in rows
                            if r["category"] == "name" and r["cohort"] == cohort
                        ]
                    )["counterfactual_pairs"]
                    for cohort in sorted({r["cohort"] for r in rows})
                }
            if split == "train-probe":
                summary["exposure_scores"] = {}
                for label, predicate in [
                    ("qa_seen", lambda v: v["qa"] > 0),
                    (
                        "reconstruction_only_seen",
                        lambda v: v["qa"] == 0 and v["reconstruction"] > 0,
                    ),
                    ("unseen_row", lambda v: sum(v.values()) == 0),
                ]:
                    selected = [r for r in rows if predicate(r["training_exposure"])]
                    selected_scores = scores(selected)
                    summary["exposure_scores"][label] = {
                        key: selected_scores[key]
                        for key in ("total", "correct", "code", "name", "missing")
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
        if args.oracle_prefix_probe:
            # Explicit upper-bound control: gold answers enter these memory
            # slots. These scores NEVER count as latent QA or qualification.
            entry["oracle_prefix_controls"] = {}
            control_cases = dev[:8] + dev[-8:]
            for kind in ("characters", "bpe"):
                control_rows = []
                with (
                    torch.inference_mode(),
                    torch.autocast("cuda", dtype=torch.bfloat16),
                ):
                    for case in control_cases:
                        ids = torch.tensor(
                            literal_memory_ids(dec_tok, case["answer"], kind),
                            device="cuda",
                        )
                        oracle = decoder.get_input_embeddings()(ids).float()
                        _, embeds, _ = prepare_prompt(
                            dec_tok,
                            decoder.get_input_embeddings(),
                            oracle,
                            system_prompt=SYSTEM,
                            question="What exact value is written in the memory? Return only that value."
                            if args.oracle_question == "copy-value"
                            else case["question"],
                            device="cuda",
                        )
                        output = generate(decoder, dec_tok, embeds, 24)
                        control_rows.append(
                            {
                                "id": case["id"],
                                "group": case["group"],
                                "category": case["category"],
                                "expected": case["answer"],
                                "actual": output["answer"],
                                "correct": output["answer"].strip() == case["answer"],
                                "finish_reason": output["finish_reason"],
                            }
                        )
                entry["oracle_prefix_controls"][kind] = {
                    "uses_gold_in_memory": True,
                    "counts_toward_qa": False,
                    "scores": scores(control_rows),
                    "rows": control_rows,
                }
                save_json(args.output / "result.json", result)
                print(
                    json.dumps(
                        {
                            "event": "oracle-prefix-control",
                            "representation": kind,
                            "uses_gold_in_memory": True,
                            "counts_toward_qa": False,
                            "scores": scores(control_rows),
                        }
                    ),
                    flush=True,
                )
        del reader
    result["completed"] = True
    save_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
