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
    COPY_HEAD_TEMPERATURE,
    DEFAULT_LORA_TARGETS,
    TRUNCATED_RECORDS,
    CopyHead,
    copy_with_contiguous_runs,
    copy_with_unique_positions,
    SlotRefiner,
    EntityTagEmbedding,
    asked_field,
    assert_shared_token_ids,
    candidate_artifacts,
    copy_type_mask,
    document_entity_tags,
    encoder_special_suffix,
    entity_tags_enabled,
    looped_read,
    reader_loops_enabled,
    lexical_value_slots,
    check_dual_layer,
    check_dual_layer_flags,
    check_encoder_layer,
    encoder_states,
    encoder_states_pair,
    RELEASED_MEMORY_SLOTS,
    SYSTEM,
    CounterfactualGroupSampler,
    MemoryTokenSnapper,
    RECONSTRUCTION_SYSTEM,
    data_rows,
    digest,
    document_features,
    document_features_dual,
    document_token_embeddings,
    feature_channel,
    inject_lora,
    load_split,
    memory_slot_count,
    prefixed_documents,
    restore_adapters,
    save_json,
    scores,
    feature_row_tags,
    self_context_documents,
    self_context_states,
    token_ids_embeddings,
)
from encoder_lora_corpus import inventories
from gpu_quality_reference import generate
from reference import load_upstream
from released_reference import checkpoint_proof, source_recipe


def validate_reference_mode(candidate, protocol, result, create):
    if create:
        if (
            candidate.get("candidate_kind") != "parameter_average"
            or not candidate.get("reference_pending")
            or protocol.get("postprocessing", {}).get("method") != "parameter_average"
            or result["completed"]
            or result["evaluations"]
        ):
            raise ValueError(
                "first reference creation requires an unmeasured derived candidate"
            )
    elif not result["completed"] or candidate.get("reference_pending", False):
        raise ValueError("candidate lacks a completed development reference")


def literal_memory_ids(tokenizer, answer, kind, slots=RELEASED_MEMORY_SLOTS):
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
    if len(pad) != 1 or len(ids) > slots:
        raise ValueError(f"oracle prefix does not fit exactly {slots} slots")
    return ids + pad * (slots - len(ids))


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
        if str(directory) in seen or len(seen) >= 16:
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
    p.add_argument("--create-derived-reference", action="store_true")
    p.add_argument("--oracle-prefix-probe", action="store_true")
    p.add_argument("--oracle-only", action="store_true")
    p.add_argument(
        "--oracle-question", choices=("original", "copy-value"), default="original"
    )
    p.add_argument("--readout-format", choices=("fields", "spelled"))
    # "all" replaces every predicted memory vector by the exact nearest decoder
    # input-embedding row before the decoder reads it (one matmul against the
    # table). Oracle-prefix controls are already literal rows and stay as they are.
    p.add_argument("--snap-memory", choices=("none", "all"), default="none")
    # Evaluation-time confidence rule on top of snapping: a slot whose nearest
    # token is below this cosine is blanked to the record's pad (" ") token so
    # a misspelled site or digit no longer reaches the decoder; abstention
    # replaces a wrong value, and a correct code next to a blanked site can be
    # answered. Recorded in the result; needs --snap-memory all.
    p.add_argument("--snap-blank-below", type=float)
    # Evaluation-time rule for copy-head candidates: no two copying slots may
    # read the same document token position (``copy_with_unique_positions``).
    p.add_argument("--copy-unique-positions", action="store_true")
    # Stronger rule: each run of copying slots reads one contiguous source span
    # (``copy_with_contiguous_runs``); exclusive with --copy-unique-positions.
    p.add_argument("--copy-contiguous-runs", action="store_true")
    # With --copy-contiguous-runs: absorb a neighbouring non-copying slot whose
    # copy distribution peaks at the run's next contiguous position.
    p.add_argument("--copy-extend-runs", action="store_true")
    p.add_argument(
        "--document-space-prefix",
        choices=("protocol", "on", "off"),
        default="protocol",
        help="override the candidate protocol's document prefix (inference-only test)",
    )
    p.add_argument("--fresh-groups", type=int, default=0)
    p.add_argument("--fresh-seed", type=int)
    args = p.parse_args()
    if args.create_derived_reference and (
        args.fresh_groups
        or args.oracle_prefix_probe
        or args.readout_format
        or args.snap_memory != "none"
    ):
        p.error("derived reference creation is a direct development-only measurement")
    if args.snap_blank_below is not None and (
        args.snap_memory != "all" or not 0 < args.snap_blank_below < 1
    ):
        p.error("--snap-blank-below needs --snap-memory all and a cosine in (0, 1)")
    if args.copy_unique_positions and args.copy_contiguous_runs:
        p.error("choose one of --copy-unique-positions / --copy-contiguous-runs")
    if args.copy_extend_runs and not args.copy_contiguous_runs:
        p.error("--copy-extend-runs needs --copy-contiguous-runs")
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
    # Other tenants run short jobs on the shared GPU; small co-resident usage
    # cannot change the measurements (only timing) and is recorded, while a
    # large one is refused as before.
    cotenant_mib = sum(
        int(line.split(",")[1].strip().split()[0])
        for line in subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader",
            ],
            text=True,
        ).splitlines()
        if line.strip()
    )
    if cotenant_mib > 16384:
        raise RuntimeError("GPU busy")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(23)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config, _ = checkpoint_proof(args.checkpoint)
    if config["num_memories"] != RELEASED_MEMORY_SLOTS:
        raise ValueError("audit requires the released 16-slot reader checkpoint")
    proofs = {
        "encoder": verify_base_model(args.encoder, "encoder"),
        "decoder": verify_base_model(args.decoder, "decoder"),
        "upstream": source_recipe(args.upstream_source),
    }
    save_json(args.output / "base-proofs.json", proofs)
    protocols, manifests = [], []
    canonical = {s: load_split(args.data, s) for s in ("train", "dev")}
    for directory, data_directory in zip(args.candidate, candidate_data):
        candidate = json.loads((directory / "candidate.json").read_text())
        protocol = json.loads((directory / "protocol.json").read_text())
        if (
            candidate["rollout_qualified"]
            or candidate["memory_slots"] != protocol["memory_slots"]
            or memory_slot_count(protocol["memory_slots"]) != candidate["memory_slots"]
        ):
            raise ValueError("not an unqualified 16..64-slot research candidate")
        for field, file in candidate_artifacts(protocol):
            if candidate.get(field) != digest(directory / file):
                raise ValueError("candidate artifact changed")
        if ("copy_head_sha256" in candidate) != bool(protocol.get("copy_head", False)):
            raise ValueError("candidate copy head artifact disagrees with its protocol")
        if ("entity_tags_sha256" in candidate) != entity_tags_enabled(protocol):
            raise ValueError(
                "candidate entity tag artifact disagrees with its protocol"
            )
        if ("slot_refiner_sha256" in candidate) != reader_loops_enabled(protocol):
            raise ValueError(
                "candidate slot refiner artifact disagrees with its protocol"
            )
        if (
            protocol["dataset_sha256"] != digest(data_directory / "manifest.json")
            or protocol["system_prompt"] != SYSTEM
        ):
            raise ValueError("candidate/data/prompt mismatch")
        validate_reference_mode(
            candidate,
            protocol,
            json.loads((directory / "result.json").read_text()),
            args.create_derived_reference,
        )
        if (
            candidate.get("development_reference_sha256")
            and digest(directory / "development-reference.json")
            != candidate["development_reference_sha256"]
        ):
            raise ValueError("candidate development reference changed")
        if any(
            qa_corpus(load_split(data_directory, split)) != qa_corpus(canonical[split])
            for split in ("train", "dev")
        ):
            raise ValueError(
                "candidate corpora have different QA facts or evaluation inputs"
            )
        protocols.append(protocol)
        manifests.append(candidate)
    # Adapters are injected once, so every candidate must share their geometry.
    target_modules = protocols[0].get("target_modules", list(DEFAULT_LORA_TARGETS))
    if any(
        (
            a["rank"],
            a["alpha"],
            a["seed"],
            a["precision"],
            a.get("target_modules", list(DEFAULT_LORA_TARGETS)),
        )
        != (
            protocols[0]["rank"],
            protocols[0]["alpha"],
            protocols[0]["seed"],
            protocols[0]["precision"],
            target_modules,
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
    adapters = inject_lora(
        encoder,
        protocols[0]["rank"],
        protocols[0]["alpha"],
        False,
        tuple(target_modules),
    )
    for protocol in protocols:
        check_encoder_layer(
            protocol.get("encoder_layer"), encoder.config.num_hidden_layers
        )
        check_dual_layer(protocol.get("dual_layer"), encoder.config.num_hidden_layers)
        check_dual_layer_flags(
            protocol.get("document_features", "pooled"),
            protocol.get("encoder_layer"),
            protocol.get("dual_layer"),
        )
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
    snapper = (
        MemoryTokenSnapper(decoder.get_input_embeddings())
        if args.snap_memory == "all"
        else None
    )
    blank_id = None
    if args.snap_blank_below is not None:
        pad = dec_tok.encode(" ", add_special_tokens=False)
        if len(pad) != 1:
            raise ValueError("record pad must be a single token")
        blank_id = pad[0]
    if any(p.get("copy_head", False) for p in protocols):
        # Copy sources are decoder embedding rows at the document token ids,
        # exact only while the encoder tokenizes the documents identically
        # (checked for every document spelling a candidate may declare,
        # including the single ids behind self-context states).
        for prefix, doubled in sorted(
            {
                (
                    bool(p.get("document_space_prefix", False)),
                    bool(p.get("document_self_context", False)),
                )
                for p in protocols
            }
        ):
            assert_shared_token_ids(
                enc_tok,
                dec_tok,
                prefixed_documents(dev[0]["documents"], prefix),
                doubled,
            )
    result = {
        "schema": "antfly.memorilla.encoder-lora-reload-audit.v1",
        "completed": False,
        "rollout_qualified": False,
        "heldout_opened": bool(fresh_rows),
        "old_qualification_splits_opened": False,
        "postfreeze_facts_generated": len(fresh_rows),
        "readout_format": args.readout_format,
        "snap_memory": args.snap_memory,
        "gpu_cotenant_mib_at_start": cotenant_mib,
        "snap_blank_below": args.snap_blank_below,
        "copy_unique_positions": bool(args.copy_unique_positions),
        "copy_contiguous_runs": bool(args.copy_contiguous_runs),
        "copy_extend_runs": bool(args.copy_extend_runs),
        "oracle_prefix_probe": args.oracle_prefix_probe,
        "oracle_only": args.oracle_only,
        "creates_derived_references": args.create_derived_reference,
        "oracle_question": args.oracle_question,
        "readout_scope": "Two declared name/code question forms; deterministic parsing, no correction, gold routing or fallback.",
        "source_sha256": digest(Path(__file__)),
        "candidates": [],
    }
    save_json(args.output / "result.json", result)
    for directory, protocol, candidate_manifest in zip(
        args.candidate, protocols, manifests
    ):
        restore_adapters(
            adapters, load_file(str(directory / "encoder-lora.safetensors"))
        )
        # A candidate trained with expanded queries carries [1, slots, width]
        # memory queries; build the pinned module at that count so the strict
        # load checks every tensor, including the expanded one.
        slots = protocol["memory_slots"]
        reader = (
            load_upstream(
                args.upstream_source,
                dict(config, num_memories=slots),
                load_file(str(directory / "reader.safetensors")),
            )
            .float()
            .cuda()
            .requires_grad_(False)
            .eval()
        )
        if reader.num_memories != slots:
            raise ValueError("candidate reader slots differ from its protocol")
        copy_head = None
        if protocol.get("copy_head", False):
            copy_head = (
                CopyHead(
                    4096, protocol.get("copy_head_temperature", COPY_HEAD_TEMPERATURE)
                )
                .cuda()
                .requires_grad_(False)
                .eval()
            )
            copy_head.load_state_dict(
                load_file(str(directory / "copy-head.safetensors"))
            )
        # Per-token entity types as trained: the oracle tags of each document
        # feed the candidate's tag table; with the type restriction the asked
        # value's slots copy only tokens of the asked field's type.
        entity_tags = protocol.get("entity_tags", "none")
        tag_embedding = None
        if entity_tags_enabled(protocol):
            tag_embedding = (
                EntityTagEmbedding(encoder.config.hidden_size)
                .cuda()
                .requires_grad_(False)
                .eval()
            )
            tag_embedding.load_state_dict(
                load_file(str(directory / "entity-tags.safetensors"))
            )
        # Looped readout as trained: the refiner feeds each pass's final
        # slots into the next pass's initial slots.
        reader_loops = protocol.get("reader_loops", 1)
        slot_refiner = None
        if reader_loops_enabled(protocol):
            slot_refiner = (
                SlotRefiner(reader.memory_queries.shape[-1])
                .cuda()
                .requires_grad_(False)
                .eval()
            )
            slot_refiner.load_state_dict(
                load_file(str(directory / "slot-refiner.safetensors"))
            )
        copy_restrict = bool(protocol.get("copy_restrict_by_type", False))
        if copy_restrict and (
            copy_head is None
            or tag_embedding is None
            or protocol.get("lexical_record", "none") == "none"
        ):
            raise ValueError(
                "type-restricted copying needs a copy head, tags and record"
            )
        exposures = {
            r["id"]: v
            for r, v in zip(train, training_exposures(protocol, len(train), train))
        }
        feature_mode = protocol.get("document_features", "pooled")
        encoder_layer = protocol.get("encoder_layer")
        dual_layer = protocol.get("dual_layer")
        # The candidate's document spelling for the encoder and copy sources
        # only; oracle controls and readout prompts read the raw case.
        space_prefix = bool(protocol.get("document_space_prefix", False))
        if args.document_space_prefix != "protocol":
            # Inference-only override; the result entry records the setting used.
            space_prefix = args.document_space_prefix == "on"
        # Index-time document context as trained: the encoder reads each
        # document doubled and keeps the second copy's states, and/or every
        # token row gets its document's pooled vector mixed in.
        self_context = bool(protocol.get("document_self_context", False))
        # Runs recorded with ``encoder_special_rows`` keep the encoder's
        # appended special rows in the self-context selection and tag them as
        # untyped; earlier candidates reproduce their count-only keep.
        suffix = (
            encoder_special_suffix(enc_tok)
            if protocol.get("encoder_special_rows", False)
            else 0
        )
        pooled_context = bool(protocol.get("document_pooled_context", False))
        if pooled_context and feature_mode == "pooled":
            raise ValueError("pooled document context needs per-token features")
        entry = {
            "arm": protocol["arm"],
            "candidate": directory.name,
            "candidate_sha256": digest(directory / "candidate.json"),
            "reconstruct_every": protocol.get("reconstruct_every", 0),
            "document_features": feature_mode,
            "encoder_layer": encoder_layer,
            **({} if dual_layer is None else {"dual_layer": dual_layer}),
            "memory_slots": slots,
            "lexical_loss_weight": protocol.get("lexical_loss_weight", 0),
            "lexical_record": protocol.get("lexical_record", "none"),
            "copy_head": copy_head is not None,
            "reader_loops": reader_loops,
            "document_space_prefix": space_prefix,
            "document_self_context": self_context,
            "document_pooled_context": pooled_context,
            **({"document_self_context_fallbacks": 0} if self_context else {}),
            "entity_tags": entity_tags,
            "copy_restrict_by_type": copy_restrict,
            "snap_memory": args.snap_memory,
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
                    documents = prefixed_documents(case["documents"], space_prefix)
                    texts, counts, copy_ids = documents + [case["question"]], None, None
                    if self_context:
                        doubled, counts, single_ids, fallbacks = self_context_documents(
                            enc_tok, documents
                        )
                        texts = doubled + [case["question"]]
                        copy_ids = torch.cat(single_ids)
                        entry["document_self_context_fallbacks"] += fallbacks
                    tokens = enc_tok(texts, padding=True, return_tensors="pt").to(
                        "cuda"
                    )
                    if tokens.input_ids.shape[1] > 128:
                        raise ValueError("encoder bound exceeded")
                    mask = tokens.attention_mask
                    tags = copy_tags = None
                    if tag_embedding is not None:
                        id_tags = [
                            document_entity_tags(enc_tok, text, case, self_context)
                            for text in documents
                        ]
                        copy_tags = torch.cat(id_tags).to("cuda")
                        tags = torch.cat(
                            [feature_row_tags(t, suffix) for t in id_tags]
                        ).to("cuda")
                    if feature_mode == "tokens-dual":
                        states = encoder_states_pair(encoder, tokens, dual_layer)
                        if counts is not None:
                            states, mask = self_context_states(
                                states, mask, counts, suffix
                            )
                        final, layer = states
                        docs, query = document_features_dual(
                            final,
                            layer,
                            mask,
                            pooled_context=pooled_context,
                            tag_embedding=tag_embedding,
                            tags=tags,
                        )
                        # The pooled-cosine diagnostic below reads the final
                        # layer regardless of the reader's feature mode.
                        states = final
                    else:
                        states = encoder_states(encoder, tokens, encoder_layer)
                        if counts is not None:
                            states, mask = self_context_states(
                                states, mask, counts, suffix
                            )
                        docs, query = document_features(
                            states,
                            mask,
                            feature_mode,
                            pooled_context=pooled_context,
                            tag_embedding=tag_embedding,
                            tags=tags,
                        )
                    memories = looped_read(
                        reader,
                        slot_refiner,
                        docs,
                        torch.zeros(
                            (1, docs.shape[1]), dtype=torch.bool, device="cuda"
                        ),
                        query,
                        reader_loops,
                    )[0]
                    if tuple(memories.shape) != (slots, 4096):
                        raise ValueError("candidate changed memory geometry")
                    copied = {}
                    if copy_head is not None:
                        # Same gated copy as training, before any snapping;
                        # under self-context the sources are the single ids
                        # the kept states stand for.
                        embedding = decoder.get_input_embeddings()
                        sources = (
                            token_ids_embeddings(embedding, copy_ids)
                            if copy_ids is not None
                            else document_token_embeddings(
                                dec_tok, embedding, documents
                            )[0]
                        )
                        slot_mask = None
                        if copy_restrict:
                            if copy_tags.shape[0] != sources.shape[0]:
                                raise ValueError("entity tags and copy sources differ")
                            record = protocol["lexical_record"]
                            slot_mask = copy_type_mask(
                                copy_tags[None],
                                torch.ones_like(copy_tags[None], dtype=torch.bool),
                                lexical_value_slots(
                                    dec_tok,
                                    case,
                                    record,
                                    record in TRUNCATED_RECORDS,
                                    slots,
                                )[None].to("cuda"),
                                [asked_field(case["question"])],
                            )[0]
                        if args.copy_unique_positions:
                            memories, gate, reassigned = copy_with_unique_positions(
                                copy_head, memories, sources, slot_mask
                            )
                            copied = {"copy_reassigned_slots": reassigned}
                        elif args.copy_contiguous_runs:
                            memories, gate, moved = copy_with_contiguous_runs(
                                copy_head,
                                memories,
                                sources,
                                slot_mask,
                                extend=args.copy_extend_runs,
                            )
                            copied = {"copy_moved_slots": moved}
                        else:
                            memories, _, gate = copy_head(
                                memories, sources, None, slot_mask
                            )
                            copied = {}
                        copied["copy_gate_mean"] = round(float(gate.mean()), 4)
                    snap = {}
                    if snapper is not None:
                        memories, snap_ids, snap_top1 = snapper(memories)
                        blanked = 0
                        if args.snap_blank_below is not None:
                            low = snap_top1 < args.snap_blank_below
                            blanked = int(low.sum())
                            if blanked:
                                pad_row = decoder.get_input_embeddings().weight[
                                    blank_id
                                ].detach()
                                memories = memories.clone()
                                memories[low] = pad_row.to(memories.dtype)
                                snap_ids = snap_ids.clone()
                                snap_ids[low] = blank_id
                        snap = {
                            "snapped_text": "".join(
                                dec_tok.decode([int(i)]) for i in snap_ids
                            ),
                            "snap_min_top1_cosine": round(float(snap_top1.min()), 4),
                            "snap_top1_cosines": [
                                round(float(c), 3) for c in snap_top1
                            ],
                            **(
                                {"snap_blanked_slots": blanked}
                                if args.snap_blank_below is not None
                                else {}
                            ),
                        }
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
                            **copied,
                            **snap,
                        }
                    )
                    if case["category"] == "code":
                        # Diagnostic on the pooled first-document embedding,
                        # independent of the candidate's reader feature mode.
                        code_vectors.setdefault(case["group"], []).append(
                            feature_channel(normalize_features(states, mask))[0]
                            .float()
                            .cpu()
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
            if split == "dev" and args.create_derived_reference:
                summary["reference_created"] = True
            if split == "dev" and snapper is not None:
                # The stored development reference was measured on unsnapped
                # memories; snapped outputs cannot reproduce it by construction.
                summary["reload_text_matches"] = None
                summary["reload_check"] = "skipped: snap-memory all"
            if split == "dev" and not args.create_derived_reference and snapper is None:
                previous = (
                    json.loads((directory / "development-reference.json").read_text())[
                        "rows"
                    ]
                    if candidate_manifest.get("development_reference_sha256")
                    else json.loads((directory / "result.json").read_text())[
                        "evaluations"
                    ][-1]["rows"]
                )
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
                        "snap_memory": args.snap_memory,
                        **summary["scores"],
                        "code_pair_embedding_cosine_mean": summary[
                            "code_pair_embedding_cosine_mean"
                        ],
                    }
                ),
                flush=True,
            )
        if args.create_derived_reference:
            summary = entry["splits"]["dev"]
            reference = {
                "schema": "antfly.memorilla.derived-development-reference.v1",
                "reader_sha256": candidate_manifest["reader_sha256"],
                "encoder_lora_sha256": candidate_manifest["encoder_lora_sha256"],
                "dataset_sha256": protocol["dataset_sha256"],
                "evaluator_sha256": digest(Path(__file__)),
                "scores": summary["scores"],
                "rows": summary["rows"],
            }
            save_json(directory / "development-reference.json", reference)
            model_result = json.loads((directory / "result.json").read_text())
            model_result.update(completed=True, reference_pending=False)
            model_result["evaluations"] = [
                {
                    "step": protocol["steps"],
                    "split": "dev",
                    "source": "derived-first-measurement",
                    "scores": summary["scores"],
                    "rows": summary["rows"],
                }
            ]
            save_json(directory / "result.json", model_result)
            save_json(directory / "unmeasured-candidate.json", candidate_manifest)
            entry["unmeasured_candidate_sha256"] = digest(
                directory / "unmeasured-candidate.json"
            )
            candidate_manifest["reference_pending"] = False
            candidate_manifest["development_reference_sha256"] = digest(
                directory / "development-reference.json"
            )
            save_json(directory / "candidate.json", candidate_manifest)
            entry["candidate_sha256"] = digest(directory / "candidate.json")
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
                            literal_memory_ids(dec_tok, case["answer"], kind, slots),
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
        del reader, copy_head, tag_embedding, slot_refiner
    result["completed"] = True
    save_json(args.output / "result.json", result)


if __name__ == "__main__":
    main()
