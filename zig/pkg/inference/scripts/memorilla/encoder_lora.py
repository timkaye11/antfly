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

"""Research-only encoder LoRA + memory reader experiment; never a serving artifact.

The reader starts from the released 16-slot checkpoint; ``--memory-slots N``
(16..64) expands its learned queries so a run trains an N-slot reader.

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
import re
import time

import torch
from torch import nn
from torch.nn import functional as F

from encoder_reference import normalize_features

SYSTEM = "Answer using the supplied memory. Return only the complete exact name or code, preserving case, hyphens and leading zeros. If the requested fact is missing, return UNKNOWN."
SCHEMA = "antfly.memorilla.encoder-lora-research.v1"
RECONSTRUCTION_SYSTEM = "Reconstruct the requested record using the supplied memory. Preserve every character, capitalization, hyphen and leading zero. Use UNKNOWN for missing fields."
DOCUMENT_FEATURE_MODES = (
    "pooled",
    "tokens",
    "tokens-question",
    "chunks4",
    "chunks8",
    "tokens-dual",
)
RELEASED_MEMORY_SLOTS = 16
MAX_MEMORY_SLOTS = 64
LEXICAL_RECORDS = (
    "none",
    "owner-code",
    "asked-first",
    "site-asked",
    "asked-site",
    "asked-site-bare",
)
# Records whose asked field leads may drop trailing tokens beyond the slots.
TRUNCATED_RECORDS = ("asked-first", "asked-site", "asked-site-bare")
LEXICAL_OBJECTIVES = ("cosine", "token-ce")
# Canonical order for encoder adapter targets (attention, then MLP).
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
DEFAULT_LORA_TARGETS = ("q_proj", "v_proj")
LEXICAL_RECIPES = {
    "cosine": "frozen-decoder-token-embedding-cosine-plus-half-squared-log-norm-ratio-v2",
    "token-ce": "frozen-decoder-vocabulary-cross-entropy-tau0.05-plus-half-squared-log-norm-ratio-v3",
}
CODE_PATTERN = re.compile(r"[A-Z][a-z]\d-00\d{4}")
# Softmax temperature of the copy head's cosine attention over document tokens.
COPY_HEAD_TEMPERATURE = 0.05
# Record label tokens the copy head must generate rather than copy, even when
# a document happens to contain them; the space pad is excluded as well.
COPY_LABEL_TEXTS = (" code", " key holder", " at", " UNKNOWN")
# Per-token entity types exposed to the reader and copy head (``--entity-tags``):
# none, site, person, access code, and the literal "not recorded" marker.
ENTITY_TAG_MODES = ("none", "oracle")
ENTITY_TYPES = ("none", "site", "person", "code", "missing")
ENTITY_TYPE_NONE, ENTITY_TYPE_SITE, ENTITY_TYPE_PERSON = 0, 1, 2
ENTITY_TYPE_CODE, ENTITY_TYPE_MISSING = 3, 4
MISSING_MARKER = "not recorded"
# Document token types a value slot may copy from, by the asked field; the
# "not recorded" marker is always allowed so a missing fact can be copied.
COPY_FIELD_TYPES = {
    "name": (ENTITY_TYPE_PERSON, ENTITY_TYPE_MISSING),
    "code": (ENTITY_TYPE_CODE, ENTITY_TYPE_MISSING),
}


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
    for field, default in [
        ("reconstruct_every", 0),
        ("reconstruction_system", None),
        ("initialization", None),
        ("sampling", "row"),
        ("sampling_recipe", None),
        ("document_features", "pooled"),
        ("lexical_loss_weight", 0),
        ("lexical_record", "none"),
        ("lexical_recipe", None),
        ("memory_scale_init", "none"),
        ("lexical_warmup_steps", 0),
        ("lexical_objective", "cosine"),
        ("lexical_warmup_batch", 1),
        ("joint_reader_lr", None),
        ("lexical_warmup_schedule", "constant"),
        ("lexical_name_weight", 1.0),
        ("lexical_warmup_cache", False),
        ("lexical_warmup_train_encoder", False),
        ("encoder_layer", None),
        ("dual_layer", None),
        ("copy_head", False),
        ("copy_head_temperature", None),
        ("copy_pointer_weight", 0),
        ("copy_gate_weight", 1.0),
        ("copy_head_lr", None),
        ("entity_tag_lr", None),
        ("entity_tag_max_norm", None),
        ("reader_loops", 1),
        ("slot_refiner_lr", None),
        ("document_space_prefix", False),
        ("document_self_context", False),
        ("document_pooled_context", False),
        ("entity_tags", "none"),
        ("copy_restrict_by_type", False),
    ]:
        if parent.get(field, default) != current.get(field, default):
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


def lora_targets(spec):
    """Parse a comma-separated list of adapter leaf names into canonical order."""
    names = [name.strip() for name in str(spec).split(",")]
    if (
        not names
        or len(set(names)) != len(names)
        or any(name not in LORA_TARGET_MODULES for name in names)
    ):
        raise ValueError(
            "adapter targets must be distinct names from "
            + ",".join(LORA_TARGET_MODULES)
        )
    return tuple(name for name in LORA_TARGET_MODULES if name in names)


def initialization_adds_targets(parent, child):
    """A child may add adapter targets (the new modules start at zero effect)
    but never drop one of the parent's; returns whether any were added."""
    parent, child = set(parent), set(child)
    if not parent or not parent <= child:
        raise ValueError("initialization drops parent adapter targets")
    return parent != child


def inject_lora(encoder, rank, alpha, trainable, targets=DEFAULT_LORA_TARGETS):
    targets = set(targets)
    if not targets <= set(LORA_TARGET_MODULES):
        raise ValueError("unknown adapter target module")
    adapters = {}
    for name, module in list(encoder.named_modules()):
        if name.rsplit(".", 1)[-1] not in targets:
            continue
        parent_name, leaf = name.rsplit(".", 1)
        adapter = LoRALinear(module, rank, alpha)
        adapter.A.requires_grad_(trainable)
        adapter.B.requires_grad_(trainable)
        setattr(encoder.get_submodule(parent_name), leaf, adapter)
        adapters[name] = adapter
    if not adapters:
        raise ValueError("no encoder projection adapters installed")
    return adapters


def check_encoder_layer(layer, num_hidden_layers):
    """``None`` keeps the final normalized state; otherwise 1..num_hidden_layers."""
    if layer is None:
        return None
    if (
        isinstance(layer, bool)
        or not isinstance(layer, int)
        or not 1 <= layer <= num_hidden_layers
    ):
        raise ValueError("encoder layer must be within 1..%d" % num_hidden_layers)
    return layer


def encoder_states(encoder, tokens, layer):
    """Per-token encoder states [B, T, width]: the final normalized state when
    ``layer`` is None, else ``hidden_states[layer]`` (``hidden_states[0]`` is the
    embedding output, so index N is the output of layer N, un-normalized below
    the last layer)."""
    if layer is None:
        return encoder(**tokens, use_cache=False).last_hidden_state
    hidden = encoder(**tokens, use_cache=False, output_hidden_states=True).hidden_states
    check_encoder_layer(layer, len(hidden) - 1)
    return hidden[layer]


def check_dual_layer(layer, num_hidden_layers):
    """``None`` disables the second key set; otherwise 1..num_hidden_layers-1
    (the final layer already supplies the first key set)."""
    if layer is None:
        return None
    if (
        isinstance(layer, bool)
        or not isinstance(layer, int)
        or not 1 <= layer < num_hidden_layers
    ):
        raise ValueError("dual layer must be within 1..%d" % (num_hidden_layers - 1))
    return layer


def check_dual_layer_flags(document_features, encoder_layer, dual_layer):
    """``tokens-dual`` and ``--dual-layer`` come together, on top of the final
    layer (``--encoder-layer`` stays unset); the upper bound waits for the model."""
    if (document_features == "tokens-dual") != (dual_layer is not None):
        raise ValueError(
            "tokens-dual document features require --dual-layer and vice versa"
        )
    if dual_layer is not None and (dual_layer < 1 or encoder_layer is not None):
        raise ValueError(
            "dual layer must be within 1..num_hidden_layers-1 beside the final layer"
        )


def encoder_states_pair(encoder, tokens, dual_layer):
    """One forward -> (final normalized states, ``hidden_states[dual_layer]``),
    each [B, T, width], for the ``tokens-dual`` document features."""
    output = encoder(**tokens, use_cache=False, output_hidden_states=True)
    check_dual_layer(dual_layer, len(output.hidden_states) - 1)
    return output.last_hidden_state, output.hidden_states[dual_layer]


def adapter_state(adapters):
    return {
        name + "." + key: getattr(module, key).detach().cpu().clone()
        for name, module in adapters.items()
        for key in ("A", "B")
    }


def restore_adapters(adapters, state, allow_missing=False):
    """Copy adapter tensors from ``state``. With ``allow_missing`` a run that
    adapts more modules than its parent restores the shared adapters whole and
    leaves the new ones at zero effect (B = 0); extra or partial entries and
    strict-mode mismatches are refused."""
    expected = {name + "." + key for name in adapters for key in ("A", "B")}
    present = set(state)
    if present != expected and not (allow_missing and present < expected):
        raise ValueError("adapter checkpoint geometry differs")
    for name in adapters:
        keys = {name + ".A", name + ".B"}
        if keys & present and not keys <= present:
            raise ValueError("adapter checkpoint geometry differs")
    with torch.no_grad():
        for name, module in adapters.items():
            if name + ".A" not in present:
                module.B.zero_()
                continue
            for key in ("A", "B"):
                target, value = getattr(module, key), state[name + "." + key]
                if tuple(target.shape) != tuple(value.shape):
                    raise ValueError("adapter checkpoint geometry differs")
                target.copy_(value)


def order_index(seed, absolute, count):
    epoch, index = divmod(absolute, count)
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order[index]


class CounterfactualGroupSampler:
    """Every eight-example batch contains both values for identical questions."""

    def __init__(self, rows, seed):
        grouped = {}
        for index, row in enumerate(rows):
            grouped.setdefault(row["group"], []).append(index)
        self.groups, self.seed, self.orders = [], seed, {}
        for indices in grouped.values():
            code = [i for i in indices if rows[i]["category"] == "code"]
            name = [i for i in indices if rows[i]["category"] == "name"]
            missing = [i for i in indices if rows[i]["category"] == "missing"]
            if len(code) != 2 or len(name) != 2 or len(missing) != 1:
                raise ValueError(
                    "group sampler requires two codes, two names and one missing row"
                )
            for pair in [code, name]:
                a, b = [rows[i] for i in pair]
                if a["question"] != b["question"] or a["answer"] == b["answer"]:
                    raise ValueError(
                        "counterfactual values must differ for identical questions"
                    )
            if rows[missing[0]]["answer"] != "UNKNOWN":
                raise ValueError("invalid missing-fact target")
            self.groups.append((code, name, missing[0]))

    def index(self, absolute_microstep):
        group_absolute, slot = divmod(absolute_microstep, 8)
        epoch, offset = divmod(group_absolute, len(self.groups))
        if epoch not in self.orders:
            order = list(range(len(self.groups)))
            random.Random(self.seed + epoch).shuffle(order)
            self.orders[epoch] = order
        code, name, missing = self.groups[self.orders[epoch][offset]]
        layout = (
            code
            + name
            + [missing, missing]
            + (code if group_absolute % 2 == 0 else name)
        )
        return layout[slot]


def reconstruction_task(case, absolute_microstep, every):
    if every and absolute_microstep % every == every - 1:
        task = case.get("reconstruction")
        if not task or set(task) != {"question", "answer"}:
            raise ValueError("scheduled reconstruction requires explicit field targets")
        return task["question"], task["answer"], RECONSTRUCTION_SYSTEM, True
    return case["question"], case["answer"], SYSTEM, False


def memory_slot_count(slots):
    """The trained reader's slot count: the released 16 or an expansion <= 64."""
    if (
        isinstance(slots, bool)
        or not isinstance(slots, int)
        or not RELEASED_MEMORY_SLOTS <= slots <= MAX_MEMORY_SLOTS
    ):
        raise ValueError("memory slots must be an integer within 16..64")
    return slots


def candidate_artifacts(protocol):
    """(candidate hash field, file) pairs a candidate directory must carry; a
    copy-head run adds its gated copy module and an entity-tag run its tag
    embedding table."""
    files = [
        ("base_proofs_sha256", "base-proofs.json"),
        ("reader_sha256", "reader.safetensors"),
        ("encoder_lora_sha256", "encoder-lora.safetensors"),
        ("protocol_sha256", "protocol.json"),
    ]
    if protocol.get("copy_head", False):
        files.append(("copy_head_sha256", "copy-head.safetensors"))
    if entity_tags_enabled(protocol):
        files.append(("entity_tags_sha256", "entity-tags.safetensors"))
    if reader_loops_enabled(protocol):
        files.append(("slot_refiner_sha256", "slot-refiner.safetensors"))
    return files


def reader_loops_enabled(protocol):
    """Whether a protocol reads its memories in more than one looped pass (and
    therefore trains a slot refiner between passes)."""
    loops = protocol.get("reader_loops", 1)
    if isinstance(loops, bool) or not isinstance(loops, int) or not 1 <= loops <= 4:
        raise ValueError("reader loops must be an integer in 1..4")
    return loops > 1


def entity_tags_enabled(protocol):
    """Whether a protocol trains per-token entity-type embeddings."""
    mode = protocol.get("entity_tags", "none")
    if mode not in ENTITY_TAG_MODES:
        raise ValueError("unknown entity tag mode: " + str(mode))
    return mode != "none"


def candidate_proof(directory, slots=RELEASED_MEMORY_SLOTS):
    """Verify an initialization candidate; its reader may have the released 16
    slots (expanded on load when the run trains more) or exactly ``slots``."""
    candidate = json.loads((directory / "candidate.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    if (
        candidate.get("schema") != SCHEMA
        or protocol.get("schema") != SCHEMA
        or candidate["rollout_qualified"]
        or candidate["memory_slots"] not in (RELEASED_MEMORY_SLOTS, slots)
        or protocol.get("memory_slots", RELEASED_MEMORY_SLOTS)
        != candidate["memory_slots"]
        or candidate.get("reference_pending", False)
    ):
        raise ValueError(
            "initialization must be an unqualified candidate with 16 or the run's slots"
        )
    for field, file in candidate_artifacts(protocol):
        if candidate.get(field) != digest(directory / file):
            raise ValueError("initialization artifact changed: " + file)
    if ("copy_head_sha256" in candidate) != bool(protocol.get("copy_head", False)):
        raise ValueError("candidate copy head artifact disagrees with its protocol")
    if ("entity_tags_sha256" in candidate) != entity_tags_enabled(protocol):
        raise ValueError("candidate entity tag artifact disagrees with its protocol")
    if ("slot_refiner_sha256" in candidate) != reader_loops_enabled(protocol):
        raise ValueError("candidate slot refiner artifact disagrees with its protocol")
    if not json.loads((directory / "result.json").read_text())["completed"]:
        raise ValueError("initialization run did not complete")
    return protocol


def expand_memory_queries(reader, slots, seed):
    """Grow the reader's learned memory queries [1, current, width] to ``slots``.

    The existing rows stay bit-identical. Row ``i >= current`` copies row
    ``i mod current`` plus Gaussian noise with std 0.01 times that row's std, so
    new slots start as near-duplicates attention can specialize. The noise comes
    from a dedicated CPU generator, leaving the global RNG stream untouched. A
    no-op when ``slots`` equals the current count; shrinking is refused.
    """
    import dataclasses

    queries = reader.memory_queries
    if queries.ndim != 3 or queries.shape[0] != 1:
        raise ValueError("reader memory queries must be [1, slots, width]")
    current = queries.shape[1]
    if reader.config.num_memories != current:
        raise ValueError("reader config disagrees with its memory queries")
    if slots < current:
        raise ValueError("memory slots cannot shrink below the loaded reader")
    if slots == current:
        return reader
    with torch.no_grad():
        source = queries.detach()[0]
        base = source[torch.arange(current, slots, device=source.device) % current]
        generator = torch.Generator().manual_seed(seed)
        noise = torch.randn(base.shape, generator=generator, dtype=torch.float32)
        noise = noise.to(base) * 0.01 * base.float().std(dim=-1, keepdim=True).to(base)
        expanded = torch.cat([source, base + noise], dim=0).unsqueeze(0).contiguous()
    reader.memory_queries = nn.Parameter(expanded, requires_grad=queries.requires_grad)
    reader.config = dataclasses.replace(reader.config, num_memories=slots)
    return reader


def feature_channel(features):
    """The released FP16 storage -> BF16 load channel, kept differentiable."""
    return features.half().bfloat16().float()


def document_features(
    states,
    attention_mask,
    mode="pooled",
    pooled_context=False,
    tag_embedding=None,
    tags=None,
):
    """Encoder states for [documents..., question] -> (docs [1, M, width], query).

    The question is always last-token pooled exactly as the released encoder
    path. ``pooled`` reproduces that path for documents bit-for-bit; ``tokens``
    passes every active document token as its own L2-normalized vector
    (``tokens-question`` adds the question's tokens as further keys);
    ``chunksK`` mean-pools K contiguous near-equal token spans per document.
    ``pooled_context`` adds each key source's own pooled vector to its rows
    before their normalization (``add_pooled_context``); the ``pooled`` mode
    has no token rows and rejects it. ``tag_embedding`` with ``tags`` (one
    entity type per document token, over the documents only) adds each token's
    type vector (``EntityTagEmbedding.offsets``) to its row after the pooled
    context and before the final normalization; question tokens under
    ``tokens-question`` carry no type. A zero table leaves the rows
    bit-identical.
    """
    if mode not in DOCUMENT_FEATURE_MODES:
        raise ValueError("unknown document feature mode")
    if mode == "tokens-dual":
        raise ValueError("tokens-dual needs both layers; use document_features_dual")
    if pooled_context and mode == "pooled":
        raise ValueError("pooled document context needs per-token document features")
    if (tag_embedding is None) != (tags is None):
        raise ValueError("entity tags need both the embedding table and the tags")
    if tags is not None and mode not in ("tokens", "tokens-question"):
        raise ValueError("entity tags need per-token document features")
    if (
        states.ndim != 3
        or states.shape[0] < 2
        or tuple(states.shape[:2]) != tuple(attention_mask.shape)
    ):
        raise ValueError("document features need [documents..., question] states")
    pooled = feature_channel(normalize_features(states, attention_mask))
    query = pooled[-1:].contiguous()
    if mode == "pooled":
        return pooled[:-1].unsqueeze(0), query
    chunks = int(mode[len("chunks") :]) if mode.startswith("chunks") else 0
    pieces = []
    # ``tokens-question`` also exposes the question's own token states as keys,
    # so slots can copy the asked entity's spelling straight from the question.
    key_count = states.shape[0] if mode == "tokens-question" else states.shape[0] - 1
    for index in range(key_count):
        rows = states[index][attention_mask[index] == 1]
        if chunks:
            if rows.shape[0] < chunks:
                raise ValueError("document has fewer encoder tokens than chunks")
            rows = torch.stack(
                [c.float().mean(dim=0) for c in torch.tensor_split(rows, chunks, dim=0)]
            )
        rows = rows.float()
        if pooled_context:
            rows = add_pooled_context(rows, pooled[index])
        pieces.append(rows)
    docs = torch.cat(pieces, dim=0)
    if (
        not torch.isfinite(docs).all().item()
        or torch.any(docs.norm(dim=-1) == 0).item()
    ):
        raise ValueError("encoder produced a nonfinite or zero document vector")
    if tags is not None:
        document_rows = docs.shape[0]
        if mode == "tokens-question":
            document_rows -= pieces[-1].shape[0]
        if tags.ndim != 1 or tags.shape[0] != document_rows:
            raise ValueError("entity tags need one type per document token")
        tags = tags.to(docs.device, torch.long)
        if mode == "tokens-question":
            tags = torch.cat([tags, tags.new_zeros(pieces[-1].shape[0])])
        docs = docs + tag_embedding.offsets(docs, tags)
    return feature_channel(F.normalize(docs, p=2, dim=-1)).unsqueeze(0), query


def document_features_dual(
    final_states,
    layer_states,
    attention_mask,
    pooled_context=False,
    tag_embedding=None,
    tags=None,
):
    """``tokens-dual``: (docs [1, 2M, width], query) from two layers of one forward.

    The final normalized layer binds a question to the right document across
    phrasings while an intermediate layer spells names better, so the reader
    sees both as separate keys: the ``tokens`` rows of the final states followed
    by the ``tokens`` rows of the intermediate states, each L2-normalized through
    the same channel. The query stays the final-layer pooled question.
    ``pooled_context`` applies per layer with that layer's own pooled vectors;
    the same ``tags`` (a token's type does not depend on the layer) and table
    apply to both layers' rows.
    """
    if tuple(final_states.shape) != tuple(layer_states.shape):
        raise ValueError("dual layer states must match the final states' geometry")
    docs, query = document_features(
        final_states,
        attention_mask,
        "tokens",
        pooled_context=pooled_context,
        tag_embedding=tag_embedding,
        tags=tags,
    )
    layer_docs, _ = document_features(
        layer_states,
        attention_mask,
        "tokens",
        pooled_context=pooled_context,
        tag_embedding=tag_embedding,
        tags=tags,
    )
    return torch.cat([docs, layer_docs], dim=1), query


def add_pooled_context(rows, pooled):
    """Document token rows [T, width] plus the document's own pooled unit vector
    [width] -> [T, width]: ``normalize(normalize(row) + pooled)`` per row.

    The causal encoder leaves a token's state blind to everything written
    after it; the document's pooled (last-token) vector has seen the whole
    text, so mixing it into every row gives each key a document-level cue
    while the row keeps its own spelling. Rows stay unit norm.
    """
    if rows.ndim != 2 or rows.shape[0] == 0 or tuple(pooled.shape) != (rows.shape[1],):
        raise ValueError("pooled context needs [T, width] rows and a [width] vector")
    mixed = F.normalize(rows.float(), p=2, dim=-1) + pooled.float().unsqueeze(0)
    if torch.any(mixed.norm(dim=-1) < 1e-6).item():
        raise ValueError("pooled context cancelled a document token row")
    return F.normalize(mixed, p=2, dim=-1)


class EntityTagEmbedding(nn.Module):
    """Learned per-token entity-type vectors added to the document token rows.

    GLiNER-style span typing: a bidirectional tagger would mark every entity
    span at index time, and the reader and copy head then see each token's
    type next to its causal state, which is what a name written before its
    site and holder cue lacks. The table starts at zero, so a fresh run reads
    exactly the untagged features, and lives in FP32 at the encoder width.
    ``offsets`` scales each type vector by its row's norm (detached), so the
    addition is the same for unit rows and for raw states; the caller's L2
    normalization (``document_features``) follows.
    """

    def __init__(self, width=2560, types=len(ENTITY_TYPES)):
        super().__init__()
        if (
            not isinstance(width, int)
            or width < 1
            or not isinstance(types, int)
            or types < 2
        ):
            raise ValueError(
                "entity tag embedding needs a positive width and >=2 types"
            )
        self.width, self.types = width, types
        self.table = nn.Embedding(types, width)
        with torch.no_grad():
            self.table.weight.zero_()
        self.float()

    def offsets(self, rows, tags):
        """``rows`` [T, width] with ``tags`` [T] long -> [T, width] FP32 type
        vectors scaled to each row's norm; zero for untyped tokens at init."""
        if (
            rows.ndim != 2
            or rows.shape[1] != self.width
            or tags.shape != (rows.shape[0],)
            or tags.dtype != torch.long
            or (tags.numel() and not bool(((tags >= 0) & (tags < self.types)).all()))
        ):
            raise ValueError("entity tag offsets need [T, width] rows and [T] type ids")
        with torch.autocast(rows.device.type, enabled=False):
            scale = rows.detach().float().norm(dim=-1, keepdim=True)
            return scale * self.table(tags.to(rows.device)).float()

    def forward(self, rows, tags):
        """Unit-normalized ``rows + emb[tags]``; the document feature path adds
        the offsets itself before its own normalization channel."""
        with torch.autocast(rows.device.type, enabled=False):
            return F.normalize(rows.float() + self.offsets(rows, tags), p=2, dim=-1)


def entity_tag_state(tag_embedding):
    return {k: v.detach().cpu().clone() for k, v in tag_embedding.state_dict().items()}


def cap_entity_tag_rows(tag_embedding, max_norm):
    """Scale every tag row whose norm exceeds ``max_norm`` down to it (in place,
    no gradient); rows at or below the cap are untouched."""
    if not isinstance(max_norm, (int, float)) or max_norm <= 0:
        raise ValueError("entity tag norm cap must be positive")
    with torch.no_grad():
        table = tag_embedding.table.weight
        norms = table.norm(dim=-1, keepdim=True)
        table.mul_(torch.clamp(max_norm / norms.clamp_min(1e-12), max=1.0))


def oracle_entity_tags(tokenizer, document, case):
    """Oracle per-token entity types of one document text -> LongTensor [T].

    The ceiling measurement for a learned tagger: spans come from the case's
    public entity list (``entities`` = [site, other site, owner, distractor]),
    the access-code pattern and the literal "not recorded" marker, never from
    the QA label. A token is typed when its character span overlaps an entity
    span (the fast tokenizer's offset mapping), so a sentence-initial name's
    subwords are typed like a mid-sentence one. Every entity found in the text
    must cover at least one token.
    """
    entities = case.get("entities")
    if not isinstance(entities, list) or len(entities) != 4:
        raise ValueError("oracle entity tags need [site, other, owner, distractor]")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("oracle entity tags need a fast tokenizer with offsets")
    encoded = tokenizer(document, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    if not ids or len(offsets) != len(ids):
        raise ValueError("document tokenized to nothing or without offsets")
    if ids != tokenizer.encode(document, add_special_tokens=False):
        raise ValueError("offset tokenization disagrees with the plain encoding")
    spans = []
    for type_id, texts in (
        (ENTITY_TYPE_SITE, entities[:2]),
        (ENTITY_TYPE_PERSON, entities[2:]),
        (ENTITY_TYPE_MISSING, [MISSING_MARKER]),
    ):
        for text in texts:
            if not isinstance(text, str) or not text:
                raise ValueError("entity names must be nonempty strings")
            spans.extend(
                (m.start(), m.end(), type_id)
                for m in re.finditer(re.escape(text), document)
            )
    spans.extend(
        (m.start(), m.end(), ENTITY_TYPE_CODE) for m in CODE_PATTERN.finditer(document)
    )
    tags = [ENTITY_TYPE_NONE] * len(ids)
    for start, end, type_id in spans:
        covered = False
        for index, (token_start, token_end) in enumerate(offsets):
            if token_start < end and token_end > start:
                if tags[index] not in (ENTITY_TYPE_NONE, type_id):
                    raise ValueError("entity spans overlap on one token")
                tags[index] = type_id
                covered = True
        if not covered:
            raise ValueError("an entity span covers no token")
    return torch.tensor(tags, dtype=torch.long)


def document_entity_tags(tokenizer, document, case, self_context=False):
    """Oracle tags aligned with a document's feature rows: the document's own
    ids, or under ``self_context`` the single ids the kept states stand for
    (the ids of ``" " + document`` when the doubled text fell back)."""
    if self_context:
        _, count, _, fallback = self_context_tokens(tokenizer, document)
        tags = oracle_entity_tags(
            tokenizer, " " + document if fallback else document, case
        )
        if tags.numel() != count:
            raise ValueError("entity tags do not cover the self-context ids")
        return tags
    return oracle_entity_tags(tokenizer, document, case)


def pad_entity_tags(rows):
    """Right-pad per-case [T_i] tag vectors into [B, T] with type none (0)."""
    if not rows or any(
        r.ndim != 1 or r.numel() == 0 or r.dtype != torch.long for r in rows
    ):
        raise ValueError("entity tag batch needs nonempty [T] long rows")
    longest = max(r.shape[0] for r in rows)
    batch = torch.zeros((len(rows), longest), dtype=torch.long, device=rows[0].device)
    for index, row in enumerate(rows):
        batch[index, : row.shape[0]] = row.to(batch.device)
    return batch


def asked_field(question):
    """Which record field the public question asks for; never reads the label."""
    lowered = question.lower()
    code = "access code" in lowered
    name = "brass key" in lowered or "brass-key" in lowered
    if code == name:
        raise ValueError("question does not name exactly one record field")
    return "code" if code else "name"


def lexical_record_text(case, record="owner-code"):
    """Canonical `` {holder}, {code}`` record read from the site's source document.

    Uses only source text and the public entity list, never the QA label.
    Missing fields are spelled UNKNOWN exactly as the QA abstention target.
    ``asked-first`` orders the field the question asks for first, so a record
    truncated to the memory budget always keeps the complete asked field.
    ``site-asked`` keeps only the asked field, labeled and bound to the site;
    ``asked-site`` puts that labeled field first and the site binding last.
    """
    if record not in LEXICAL_RECORDS[1:]:
        raise ValueError("unknown lexical record form")
    site, name_value, code_value = lexical_record_fields(case)
    if record in ("asked-site", "asked-site-bare"):
        # Asked field first, so its tokens occupy fixed slots regardless of the
        # site's token count; the site binding trails (read 64/64 by the frozen
        # decoder in target-readability). ``asked-site-bare`` drops the " code"
        # label: a code is ten decoder tokens, and with the label a site of
        # four tokens plus " Depot" overflows the 16-slot record (17), so the
        # trailing " Depot" was cut and the decoder read a mismatched site.
        if asked_field(case["question"]) == "code":
            if record == "asked-site-bare":
                return f" {code_value} at {site}"
            return f" code {code_value} at {site}"
        return f" key holder {name_value} at {site}"
    if record == "site-asked":
        # The frozen decoder binds a labeled field to the asked site; a bare
        # or site-free value is read as a missing fact (target-readability).
        if asked_field(case["question"]) == "code":
            return f" {site} code {code_value}"
        return f" {site} key holder {name_value}"
    if record == "asked-first" and asked_field(case["question"]) == "code":
        return f" {code_value}, {name_value}"
    return f" {name_value}, {code_value}"


def lexical_record_fields(case):
    """(site, holder value, code value) of the site's source document; missing
    fields are UNKNOWN. Reads only source text and the public entity list."""
    entities = case.get("entities")
    if not isinstance(entities, list) or len(entities) != 4:
        raise ValueError("lexical record requires [site, other, owner, distractor]")
    site, other_site, owner, _ = entities
    sources = [d for d in case["documents"] if site in d and other_site not in d]
    if len(sources) != 1:
        raise ValueError("lexical record needs exactly one source document per site")
    source = sources[0]
    codes = CODE_PATTERN.findall(source)
    holder, missing = owner in source, MISSING_MARKER in source
    if len(codes) == 1 and holder and not missing:
        return site, owner, codes[0]
    if missing and holder and not codes:
        return site, owner, "UNKNOWN"
    if missing and not holder and len(codes) == 1:
        return site, "UNKNOWN", codes[0]
    raise ValueError("source document does not hold exactly one holder/code record")


def lexical_value_slots(
    tokenizer, case, record, truncate=False, slots=RELEASED_MEMORY_SLOTS
):
    """Bool [slots]: the record slots whose token covers the asked field's value.

    Found through the fast tokenizer's offsets over the canonical record text
    (the same ids as ``lexical_target_ids``); label, site, separator and pad
    slots are False. UNKNOWN values count as the asked value as well.
    """
    text = lexical_record_text(case, record)
    _, name_value, code_value = lexical_record_fields(case)
    value = code_value if asked_field(case["question"]) == "code" else name_value
    if text.count(value) != 1:
        raise ValueError("asked value must occur exactly once in the record")
    start = text.index(value)
    end = start + len(value)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("value slots need a fast tokenizer with offsets")
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    target = lexical_target_ids(tokenizer, text, truncate, slots)
    kept = min(len(ids), slots) if truncate else len(ids)
    if ids[:kept] != target[:kept]:
        raise ValueError("offset tokenization disagrees with the lexical target ids")
    mask = torch.zeros(slots, dtype=torch.bool)
    for index in range(kept):
        token_start, token_end = offsets[index]
        mask[index] = token_start < end and token_end > start
    if not bool(mask.any()):
        raise ValueError("asked value covers no record slot")
    return mask


def copy_type_mask(tags, doc_mask, value_slots, fields):
    """Per-slot copy restriction -> bool [B, slots, T].

    Slots marked in ``value_slots`` [B, slots] may only copy document tokens
    whose entity type (``tags`` [B, T]) fits the asked field (``fields``, one
    of ``COPY_FIELD_TYPES`` per case); every other slot keeps every real token
    (``doc_mask`` [B, T]). Every slot must keep at least one token.
    """
    if (
        tags.ndim != 2
        or doc_mask.shape != tags.shape
        or doc_mask.dtype != torch.bool
        or tags.dtype != torch.long
        or value_slots.ndim != 2
        or value_slots.shape[0] != tags.shape[0]
        or value_slots.dtype != torch.bool
        or len(fields) != tags.shape[0]
        or any(field not in COPY_FIELD_TYPES for field in fields)
    ):
        raise ValueError(
            "copy type mask needs tags/doc mask [B, T], value slots [B, slots]"
            " and a known asked field per case"
        )
    mask = doc_mask[:, None, :].expand(-1, value_slots.shape[1], -1).clone()
    for index, field in enumerate(fields):
        allowed = torch.tensor(
            COPY_FIELD_TYPES[field], dtype=torch.long, device=tags.device
        )
        typed = torch.isin(tags[index], allowed) & doc_mask[index]
        mask[index, value_slots[index]] = typed
    if not bool(mask.any(dim=-1).all()):
        raise ValueError("a restricted slot has no document token of its type")
    return mask


def lexical_target_ids(tokenizer, text, truncate=False, slots=RELEASED_MEMORY_SLOTS):
    """Decoder token ids of a <=``slots``-token record padded with the space token."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    pad = tokenizer.encode(" ", add_special_tokens=False)
    if truncate:
        ids = ids[:slots]
    if len(pad) != 1 or not 1 <= len(ids) <= slots:
        raise ValueError(f"lexical record does not fit exactly {slots} memory slots")
    return ids + pad * (slots - len(ids))


def lexical_targets(
    tokenizer, embedding, text, truncate=False, slots=RELEASED_MEMORY_SLOTS
):
    """Frozen decoder input embeddings of a <=``slots``-token record padded with
    spaces.

    With ``truncate`` the trailing tokens beyond the slots are dropped; the
    caller guarantees the complete asked field precedes them.
    """
    if embedding.weight.requires_grad:
        raise ValueError("lexical teacher requires frozen decoder embeddings")
    ids = torch.tensor(
        lexical_target_ids(tokenizer, text, truncate, slots),
        device=embedding.weight.device,
    )
    with torch.no_grad():
        return embedding(ids).detach().float()


def lexical_alignment_loss(predicted, target, slots=RELEASED_MEMORY_SLOTS):
    """Per slot: (1 - cosine) + 0.5 * (log ||p|| - log ||t||)^2, averaged.

    Training-only auxiliary objective in FP32; the teacher never replaces the
    predicted memory in the decoder forward. The log-ratio norm term keeps the
    gradient bounded when the reader's output scale is far from the embedding
    scale, where a squared error would dominate the QA objective.
    """
    if (
        predicted.ndim != 2
        or predicted.shape[0] != slots
        or tuple(predicted.shape) != tuple(target.shape)
    ):
        raise ValueError(
            f"lexical supervision requires predicted/target [{slots}, width]"
        )
    with torch.autocast(predicted.device.type, enabled=False):
        predicted, target = predicted.float(), target.detach().float()
        target_norm = target.norm(dim=-1)
        if not bool((target_norm > 0).all()):
            raise ValueError("lexical targets must have nonzero norm")
        cosine = F.cosine_similarity(predicted, target, dim=-1)
        log_ratio = torch.log(predicted.norm(dim=-1).clamp_min(1e-6)) - torch.log(
            target_norm
        )
        return ((1 - cosine) + 0.5 * log_ratio.square()).mean()


class LexicalVocabularyTeacher:
    """Frozen, L2-normalized decoder embedding table for token cross-entropy."""

    def __init__(self, embedding, temperature=0.05):
        if embedding.weight.requires_grad or not 0 < temperature <= 1:
            raise ValueError("vocabulary teacher needs frozen embeddings and 0<tau<=1")
        with torch.no_grad():
            weight = embedding.weight.detach().float()
            self.norms = weight.norm(dim=-1).clamp_min(1e-6)
            self.table = F.normalize(weight, dim=-1)
        self.temperature = temperature


def snap_memory_to_tokens(memories, table_normalized, table):
    """Replace each memory vector by the exact nearest decoder input-embedding row.

    Cosine search in FP32 over ``table_normalized`` ([V, width], unit rows);
    the returned rows are ``table[ids]`` in the table's own dtype, so the
    decoder reads exactly what it reads for text. Returns
    ``(snapped [slots, width], ids [slots], top1_cosine [slots])``. This is an
    evaluation-time policy, not a training objective: serving needs one matmul
    against the embedding table, and encoder and decoder token ids coincide.
    """
    if (
        memories.ndim != 2
        or table_normalized.ndim != 2
        or tuple(table_normalized.shape) != tuple(table.shape)
        or memories.shape[1] != table.shape[1]
        or memories.shape[0] < 1
    ):
        raise ValueError(
            "memory snapping requires memories [slots, width] and tables [V, width]"
        )
    with torch.autocast(memories.device.type, enabled=False):
        unit = F.normalize(memories.detach().float(), dim=-1)
        cosine = unit @ table_normalized.float().T
        top1, ids = cosine.max(dim=-1)
        return table.index_select(0, ids), ids, top1


class MemoryTokenSnapper:
    """Frozen decoder embedding table prepared for nearest-token memory snapping."""

    def __init__(self, embedding):
        if embedding.weight.requires_grad:
            raise ValueError("memory snapper needs frozen embeddings")
        with torch.no_grad():
            self.table = embedding.weight.detach()
            self.table_normalized = F.normalize(self.table.float(), dim=-1)

    def __call__(self, memories):
        return snap_memory_to_tokens(memories, self.table_normalized, self.table)


class CopyHead(nn.Module):
    """Gated copy of exact document token embeddings into the memory slots.

    Encoder and decoder share one vocabulary, so every value token a record
    needs (name subwords, code characters, site tokens) is present among the
    document token ids; only label tokens must be generated. Each slot attends
    over the frozen decoder embeddings of those tokens (cosine / temperature)
    and a sigmoid gate mixes the copied row with the generated memory:
    ``output = g * copied + (1 - g) * memories``. The query starts at identity
    and the gate bias at -2 (g ~ 0.12), so a fresh head mostly generates.
    Everything runs in FP32.
    """

    def __init__(self, width=4096, temperature=COPY_HEAD_TEMPERATURE):
        super().__init__()
        if not isinstance(width, int) or width < 1 or not 0 < temperature <= 1:
            raise ValueError("copy head needs a positive width and 0<tau<=1")
        self.width, self.temperature = width, temperature
        self.query = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, 1)
        with torch.no_grad():
            self.query.weight.copy_(torch.eye(width))
            self.gate.weight.zero_()
            self.gate.bias.fill_(-2.0)
        self.float()

    def forward(self, memories, token_embeddings, token_mask=None, slot_mask=None):
        """``memories`` [slots, width] with ``token_embeddings`` [T, width], an
        optional [T] bool mask and an optional per-slot [slots, T] bool mask,
        or the batched [B, slots, width] / [B, T, width] / [B, T] /
        [B, slots, T] forms -> ``(output, copy_probs, gate)`` with the same
        leading geometry; ``copy_probs`` is [.., slots, T] and ``gate``
        [.., slots]. Masked tokens receive zero probability."""
        single = memories.ndim == 2
        if single:
            memories, token_embeddings = (
                memories.unsqueeze(0),
                token_embeddings.unsqueeze(0),
            )
            if token_mask is not None:
                token_mask = token_mask.unsqueeze(0)
            if slot_mask is not None:
                slot_mask = slot_mask.unsqueeze(0)
        output, logits, gate_logits = self.forward_detailed(
            memories, token_embeddings, token_mask, slot_mask
        )
        probs, gate = logits.softmax(dim=-1), torch.sigmoid(gate_logits)
        if single:
            return output[0], probs[0], gate[0]
        return output, probs, gate

    def forward_detailed(
        self, memories, token_embeddings, token_mask=None, slot_mask=None
    ):
        """Batched form only: [B, slots, width] memories over [B, T, width]
        tokens -> ``(output [B, slots, width], copy_logits [B, slots, T],
        gate_logits [B, slots])``. The copy logits are the pre-softmax cosine /
        temperature scores with ``-inf`` at masked tokens and the gate logits
        the pre-sigmoid gate, so pointer supervision can score them directly.
        ``slot_mask`` [B, slots, T] (default None: every real token) further
        restricts each slot to its own token subset, such as the tokens of the
        asked field's entity type; every slot must keep one valid token."""
        if token_mask is None:
            token_mask = torch.ones(
                token_embeddings.shape[:2], dtype=torch.bool, device=memories.device
            )
        if (
            memories.ndim != 3
            or token_embeddings.ndim != 3
            or memories.shape[2] != self.width
            or token_embeddings.shape[2] != self.width
            or token_embeddings.shape[0] != memories.shape[0]
            or token_mask.shape != token_embeddings.shape[:2]
            or token_mask.dtype != torch.bool
            or not bool(token_mask.any(dim=1).all())
        ):
            raise ValueError(
                "copy head needs memories [B, slots, width], tokens [B, T, width]"
                " and a bool mask with one valid token per case"
            )
        allowed = token_mask[:, None, :]
        if slot_mask is not None:
            if (
                slot_mask.shape
                != (memories.shape[0], memories.shape[1], token_embeddings.shape[1])
                or slot_mask.dtype != torch.bool
            ):
                raise ValueError("copy slot mask needs bool [B, slots, T]")
            allowed = allowed & slot_mask
            if not bool(allowed.any(dim=-1).all()):
                raise ValueError("copy slot mask leaves a slot without tokens")
        with torch.autocast(memories.device.type, enabled=False):
            memories = memories.float()
            tokens = token_embeddings.detach().float()
            unit_query = F.normalize(self.query(memories), dim=-1)
            unit_tokens = F.normalize(tokens, dim=-1)
            logits = unit_query @ unit_tokens.transpose(1, 2) / self.temperature
            logits = logits.masked_fill(~allowed, float("-inf"))
            probs = logits.softmax(dim=-1)
            copied = probs @ tokens
            gate_logits = self.gate(memories)
            gate = torch.sigmoid(gate_logits)
            output = gate * copied + (1 - gate) * memories
        return output, logits, gate_logits.squeeze(-1)


def copy_with_unique_positions(
    copy_head, memories, token_embeddings, slot_mask=None, min_gate=0.5
):
    """Inference-time copy with no two copying slots on one source position.

    ``memories`` [S, width] and ``token_embeddings`` [T, width] go through the
    copy head exactly as ``CopyHead.forward``; then the slots whose gate is at
    least ``min_gate`` are visited in descending order of their peak copy
    probability, each claiming its argmax source position. A slot whose peak
    position is already claimed is re-pointed to its best unclaimed position
    and receives the hard copy of that token (gate-mixed with its memory);
    every other slot keeps the soft output. Returns ``(output [S, width],
    gate [S], reassigned slots)``. A code's own prefix digit copied again into
    a later digit slot, or a name subword copied twice, is exactly a shared
    position, which this rule forbids without touching the trained weights.
    """
    if memories.ndim != 2 or token_embeddings.ndim != 2:
        raise ValueError("unique copy needs [S, width] memories and [T, width] tokens")
    if not 0 <= min_gate <= 1:
        raise ValueError("unique copy needs a gate threshold in [0, 1]")
    output, logits, gate_logits = copy_head.forward_detailed(
        memories.unsqueeze(0),
        token_embeddings.unsqueeze(0),
        None,
        None if slot_mask is None else slot_mask.unsqueeze(0),
    )
    output, logits, gate = output[0], logits[0], torch.sigmoid(gate_logits[0])
    probs = logits.softmax(dim=-1)
    peak, position = probs.max(dim=-1)
    order = sorted(
        (s for s in range(probs.shape[0]) if float(gate[s]) >= min_gate),
        key=lambda s: -float(peak[s]),
    )
    taken, reassigned = set(), 0
    tokens = token_embeddings.detach().float()
    for slot in order:
        wanted = int(position[slot])
        if wanted not in taken:
            taken.add(wanted)
            continue
        row = probs[slot].clone()
        row[list(taken)] = -1.0
        best = int(row.argmax())
        if float(row[best]) < 0:
            continue
        taken.add(best)
        reassigned += 1
        g = gate[slot]
        output[slot] = g * tokens[best] + (1 - g) * memories[slot].float()
    return output, gate, reassigned


def copy_with_contiguous_runs(
    copy_head,
    memories,
    token_embeddings,
    slot_mask=None,
    min_gate=0.5,
    extend=False,
    return_spans=False,
    boundary=None,
    label_slots=None,
    split_jumps=None,
):
    """Inference-time copy where each run of copying slots reads one contiguous
    source span.

    Every value a record copies (a code, a name, a site) is a contiguous
    substring of its document, and the record lays its tokens out in order, so
    a maximal run of consecutive slots whose gate is at least ``min_gate``
    should read consecutive source positions ``start, start+1, ...``. For each
    run the start maximizing the summed log copy probability along that
    diagonal is chosen and every slot of the run receives the hard copy of its
    token (gate-mixed with its memory); slots below the gate keep the soft
    output. Runs claim positions in order of their best score so two runs never
    share a position (a later run skips starts overlapping claimed positions
    when any other start is possible). Returns ``(output [S, width], gate
    [S], moved slots)``, counting slots whose span position differs from their
    own argmax. Digit swaps inside a code ("Mh7-007579" for "Mh7-001579") are
    exactly off-diagonal copies, which this rule forbids.
    """
    if memories.ndim != 2 or token_embeddings.ndim != 2:
        raise ValueError("contiguous copy needs [S, width] memories and [T, width] tokens")
    if not 0 <= min_gate <= 1:
        raise ValueError("contiguous copy needs a gate threshold in [0, 1]")
    output, logits, gate_logits = copy_head.forward_detailed(
        memories.unsqueeze(0),
        token_embeddings.unsqueeze(0),
        None,
        None if slot_mask is None else slot_mask.unsqueeze(0),
    )
    output, logits, gate = output[0], logits[0], torch.sigmoid(gate_logits[0])
    log_probs = logits.log_softmax(dim=-1)
    argmax = log_probs.argmax(dim=-1)
    copying = [float(g) >= min_gate for g in gate]
    if label_slots is not None:
        # Slots whose own memory is a record label (" at", " key holder",
        # " code", the pad) generate rather than copy whatever their gate
        # says; a label slot with a high gate would otherwise glue the value
        # runs on both sides into one span ("Cipapuu Ruzepu" for
        # "Cipapu at Ruzepu").
        if len(label_slots) != len(copying):
            raise ValueError("label slot mask needs one flag per slot")
        copying = [c and not bool(l) for c, l in zip(copying, label_slots)]
    runs, start = [], None
    for slot, flag in enumerate(copying + [False]):
        if flag and start is None:
            start = slot
        elif not flag and start is not None:
            runs.append((start, slot))
            start = None
    if split_jumps is not None and runs:
        # Two values glued by a label slot whose gate rose above the threshold
        # ("Cipapu at Ruzepu" read as one run) show up as a jump in the slots'
        # argmax source positions: inside one value consecutive slots point
        # at consecutive positions give or take a duplicate, across the glue
        # they jump. Split a run where the step differs from +1 by more than
        # ``split_jumps``; a single slot left between two runs is the label
        # and stops copying.
        pieces = []
        for begin, end in runs:
            cut = begin
            for slot in range(begin + 1, end):
                if abs(int(argmax[slot]) - int(argmax[slot - 1]) - 1) > split_jumps:
                    pieces.append((cut, slot))
                    cut = slot
            pieces.append((cut, end))
        runs = []
        for index, (begin, end) in enumerate(pieces):
            sandwiched = (
                end - begin == 1
                and index > 0
                and index + 1 < len(pieces)
                and pieces[index - 1][1] == begin
                and pieces[index + 1][0] == end
            )
            if sandwiched:
                copying[begin] = False
            else:
                runs.append((begin, end))
    positions = log_probs.shape[1]
    if extend and runs:
        # A run boundary is decided by the gate, which can fall under the
        # threshold on a value's first or last subword (the slot then generates
        # a label-like vector and the name loses "oco" or the site loses
        # " Mo"). A neighbouring non-copying slot whose own copy distribution
        # peaks at the position just beyond the run's current span joins the
        # run: it is copying the value's next token, whatever its gate says.
        # Absorbed slots are forced to copy (gate 1) since their gate was wrong.
        forced = set()
        grown = []
        for index, (begin, end) in enumerate(runs):
            prev_end = runs[index - 1][1] if index else 0
            next_begin = runs[index + 1][0] if index + 1 < len(runs) else len(copying)
            run_start = int(argmax[begin])
            run_stop = int(argmax[end - 1])
            while begin > prev_end and int(argmax[begin - 1]) == run_start - 1:
                begin -= 1
                run_start -= 1
                forced.add(begin)
            while end < next_begin and int(argmax[end]) == run_stop + 1:
                forced.add(end)
                end += 1
                run_stop += 1
            grown.append((begin, end))
        runs = grown
    else:
        forced = set()
    scored = []
    for begin, end in runs:
        length = end - begin
        if length > positions:
            continue
        scores = torch.stack(
            [
                sum(log_probs[begin + k, s + k] for k in range(length))
                for s in range(positions - length + 1)
            ]
        )
        scored.append((begin, end, scores))
    scored.sort(key=lambda item: -float(item[2].max()))
    tokens = token_embeddings.detach().float()
    taken, moved, spans = set(), 0, []
    for begin, end, scores in scored:
        length = end - begin
        order = torch.argsort(scores, descending=True).tolist()
        chosen = next(
            (
                s
                for s in order
                if torch.isfinite(scores[s])
                and not any(s + k in taken for k in range(length))
            ),
            None,
        )
        if chosen is not None and boundary is not None:
            # ``boundary`` = (tokenizer, source ids, margin): among the starts
            # within ``margin`` log-probability of the best, prefer one whose
            # span begins and ends on word boundaries of the source; a span
            # that is off by one token ("ebo" of "Hohebo" read as "eb"+"obo")
            # otherwise survives and word completion then lengthens the
            # wrong answer instead of fixing it.
            tokenizer, source_ids, margin = boundary
            aligned = next(
                (
                    s
                    for s in order
                    if torch.isfinite(scores[s])
                    and float(scores[chosen] - scores[s]) <= margin
                    and not any(s + k in taken for k in range(length))
                    and not token_continues_word(tokenizer, source_ids[s])
                    and (
                        s + length >= len(source_ids)
                        or not token_continues_word(tokenizer, source_ids[s + length])
                    )
                ),
                None,
            )
            if aligned is not None:
                chosen = aligned
        if chosen is None:
            chosen = next((s for s in order if torch.isfinite(scores[s])), None)
        if chosen is None:
            continue
        spans.append((begin, end, chosen))
        for k in range(length):
            slot, source = begin + k, chosen + k
            taken.add(source)
            if int(argmax[slot]) != source:
                moved += 1
            g = gate[slot] if slot not in forced else gate.new_ones(())
            output[slot] = g * tokens[source] + (1 - g) * memories[slot].float()
    if return_spans:
        return output, gate, moved, sorted(spans)
    return output, gate, moved


def token_continues_word(tokenizer, token_id):
    """Whether a decoder token continues the word before it: no leading space
    and only letters, digits or hyphens (code pieces such as ``-00``)."""
    text = tokenizer.decode([int(token_id)])
    body = text.strip()
    return bool(body) and not text.startswith(" ") and all(
        c.isalnum() or c == "-" for c in body
    )


def token_starts_word(tokenizer, token_id):
    """Whether a decoder token begins a new alphanumeric word (leading space,
    letters or digits), as the " Depot" of a site after its name."""
    text = tokenizer.decode([int(token_id)])
    body = text.strip()
    return text.startswith(" ") and bool(body) and body.isalnum()


def complete_copied_words(
    slot_ids, spans, source_ids, tokenizer, pad_id, overflow="truncate"
):
    """Snapped slot ids [S] -> ids [S] where every copied span starts and ends
    on a word boundary of its source document.

    ``spans`` are ``(begin, end, start)`` from ``copy_with_contiguous_runs``:
    record slots ``begin:end`` read source positions ``start:start+end-begin``.
    A span whose last source token is followed by word-continuing tokens (the
    "oco" of "Daboco" after a run that stopped at "Dab") takes them; a span
    whose first token itself continues a word (the "zeba" of " Mozeba") takes
    the preceding tokens back to the word start. Later slots shift right into
    the trailing pads; when the record overflows, trailing pads go first and
    then the final tokens, as the training records were truncated; with
    ``overflow="drop-last-run"`` an overflowing record instead loses its last
    copied span entirely (the site after a long code or name), since the
    decoder abstains on a truncated site but answers from a value whose site
    is simply absent. The same policy drops the last span of a record that is
    full (no pad left) when its source continues with a new word (" Depot"
    after " Cibipu"): the training record was truncated there and the decoder
    reads the shortened site as a mismatch. Returns ``(ids, inserted)``.
    """
    if overflow not in ("truncate", "drop-last-run"):
        raise ValueError("unknown overflow policy")
    slot_ids = [int(i) for i in slot_ids]
    source_ids = [int(i) for i in source_ids]
    size, cursor, inserted, pieces = len(slot_ids), 0, 0, []
    last_run, last_source_end = None, None
    for begin, end, start in sorted(spans):
        if not 0 <= begin < end <= size or not 0 <= start <= len(source_ids) - (end - begin):
            raise ValueError("copied span does not fit the record or the source")
        pieces.extend(slot_ids[cursor:begin])
        first, last = start, start + (end - begin)
        while first > 0 and token_continues_word(tokenizer, source_ids[first]):
            first -= 1
            inserted += 1
        while last < len(source_ids) and token_continues_word(tokenizer, source_ids[last]):
            last += 1
            inserted += 1
        last_run = (len(pieces), len(pieces) + (last - first))
        last_source_end = last
        pieces.extend(source_ids[first:last])
        cursor = end
    pieces.extend(slot_ids[cursor:])
    while len(pieces) > size and pieces and pieces[-1] == pad_id:
        pieces.pop()
    if overflow == "drop-last-run" and last_run is not None:
        full = len(pieces) >= size and pad_id not in pieces[last_run[1] :]
        continues = (
            full
            and last_source_end < len(source_ids)
            and token_starts_word(tokenizer, source_ids[last_source_end])
        )
        if len(pieces) > size or continues:
            pieces = pieces[: last_run[0]] + pieces[last_run[1] :]
            while len(pieces) > size and pieces and pieces[-1] == pad_id:
                pieces.pop()
    while len(pieces) > size:
        pieces.pop()
    pieces.extend([pad_id] * (size - len(pieces)))
    return pieces, inserted


def copy_excluded_ids(tokenizer, labels=COPY_LABEL_TEXTS):
    """Token ids a slot must generate rather than copy: the space pad and every
    token of the record label texts. Any of them may occur in a document
    (`` at`` does), so copyability is decided on the target id, not the text."""
    pad = tokenizer.encode(" ", add_special_tokens=False)
    if len(pad) != 1:
        raise ValueError("copy exclusion needs a single-token space pad")
    excluded = set(pad)
    for text in labels:
        excluded.update(tokenizer.encode(text, add_special_tokens=False))
    return excluded


def copyable_mask(target_ids, doc_ids, doc_mask, excluded_ids):
    """Per slot, whether the lexical target token can be copied from the case's
    documents -> ``(copyable [B, slots] bool, match [B, slots, T] bool)``.

    A slot is copyable when its target id occurs at a real (unmasked) document
    position and is not one of ``excluded_ids`` (pad and label tokens, which
    the head must generate). ``match`` marks every matching position of the
    copyable slots and is all False on the others.
    """
    if (
        target_ids.ndim != 2
        or doc_ids.ndim != 2
        or doc_mask.shape != doc_ids.shape
        or doc_ids.shape[0] != target_ids.shape[0]
        or doc_mask.dtype != torch.bool
        or target_ids.dtype != torch.long
        or doc_ids.dtype != torch.long
    ):
        raise ValueError(
            "copyable mask needs target ids [B, slots], document ids [B, T]"
            " and a bool [B, T] mask"
        )
    match = (target_ids[:, :, None] == doc_ids[:, None, :]) & doc_mask[:, None, :]
    copyable = match.any(dim=-1)
    if excluded_ids:
        excluded = torch.tensor(
            sorted(excluded_ids), dtype=torch.long, device=target_ids.device
        )
        copyable &= ~torch.isin(target_ids, excluded)
    return copyable, match & copyable[:, :, None]


def pointer_losses(copy_logits, gate_logits, match, copyable, doc_mask):
    """Direct copy-head supervision -> ``(pointer_loss, gate_loss)`` FP32.

    ``pointer_loss`` is -log of the copy probability mass on the matching
    positions (log-sum-exp over matches minus log-sum-exp over all valid
    positions), averaged over copyable slots (zero when there are none);
    ``gate_loss`` is BCE of the gate logit toward 1 on copyable and 0 on other
    slots, averaged over every slot. Masked positions never carry mass.
    """
    if (
        copy_logits.ndim != 3
        or match.shape != copy_logits.shape
        or copyable.shape != copy_logits.shape[:2]
        or gate_logits.shape != copyable.shape
        or doc_mask.shape != (copy_logits.shape[0], copy_logits.shape[2])
        or match.dtype != torch.bool
        or copyable.dtype != torch.bool
        or doc_mask.dtype != torch.bool
        or not bool(doc_mask.any(dim=1).all())
    ):
        raise ValueError(
            "pointer losses need copy logits [B, slots, T], gate logits [B, slots],"
            " bool match/copyable/document masks and one valid token per case"
        )
    if not torch.equal(match.any(dim=-1), copyable) or bool(
        (match & ~doc_mask[:, None, :]).any()
    ):
        raise ValueError("match positions must be valid and cover the copyable slots")
    with torch.autocast(copy_logits.device.type, enabled=False):
        logits = copy_logits.float().masked_fill(~doc_mask[:, None, :], float("-inf"))
        total = logits.logsumexp(dim=-1)
        if bool(copyable.any()):
            matched = logits.masked_fill(~match, float("-inf")).logsumexp(dim=-1)
            pointer_loss = (total - matched)[copyable].mean()
        else:
            pointer_loss = logits.new_zeros(())
        gate_loss = F.binary_cross_entropy_with_logits(
            gate_logits.float(), copyable.float()
        )
    return pointer_loss, gate_loss


def prefixed_documents(documents, enabled):
    """``documents`` each prefixed with one space when ``enabled`` (a text that
    already starts with a space is left alone), else an unchanged copy. The
    shared Qwen tokenizer segments a sentence-initial word differently from the
    same word mid-sentence ("Mujapo" -> M/uj/apo versus " Muj"/apo), so a
    record's first subword is missing from an owner-first document's ids and
    the copy head cannot copy it; the prefix makes both segmentations agree."""
    if not enabled:
        return list(documents)
    return [text if text.startswith(" ") else " " + text for text in documents]


def self_context_tokens(tokenizer, document):
    """Causal self-context for one document -> ``(doubled, n, single_ids, fallback)``.

    The Qwen3 embedding encoder is causal: a token's state sees only the
    tokens to its left, so a name written before the site and holder cue of
    its sentence carries no trace of either. Encoding the document twice and
    keeping the states of the second copy gives each of its ``n`` tokens the
    whole document as left context. The second copy must tokenize to the
    document's own ids: a text that starts with a space (the prefixed form)
    is simply repeated, any other is repeated after one space. When the
    trailing ids still differ (a sentence-initial word re-segments mid-text)
    the single ids fall back to those of ``" " + document`` (``fallback``
    True, so the kept states and copy sources agree); anything else raises.
    """
    if not isinstance(document, str) or not document.strip():
        raise ValueError("self context needs a nonblank document text")
    single = tokenizer.encode(document, add_special_tokens=False)
    doubled = (
        document + document if document.startswith(" ") else document + " " + document
    )
    doubled_ids = tokenizer.encode(doubled, add_special_tokens=False)
    fallback = False
    if not single or doubled_ids[-len(single) :] != single:
        single = tokenizer.encode(" " + document, add_special_tokens=False)
        fallback = True
    if not single or doubled_ids[-len(single) :] != single:
        raise ValueError("doubled document does not end with its own token ids")
    return doubled, len(single), single, fallback


def self_context_documents(tokenizer, documents):
    """``self_context_tokens`` over a case's documents -> (doubled texts, kept
    token counts, single id LongTensors, number of fallbacks)."""
    if not documents:
        raise ValueError("self context needs at least one document")
    doubled, counts, ids, fallbacks = [], [], [], 0
    for text in documents:
        text_doubled, count, single, fallback = self_context_tokens(tokenizer, text)
        doubled.append(text_doubled)
        counts.append(count)
        ids.append(torch.tensor(single, dtype=torch.long))
        fallbacks += int(fallback)
    return doubled, counts, ids, fallbacks


def select_last_rows(states, attention_mask, counts):
    """Right-padded [B, T, width] states -> the last ``counts[b]`` active rows of
    each row, right-padded again -> (states [B, max(counts), width], mask).

    Under self-context the kept rows are the second copy's states; the
    question row passes its full active length to stay whole.
    """
    if (
        states.ndim != 3
        or tuple(states.shape[:2]) != tuple(attention_mask.shape)
        or len(counts) != states.shape[0]
    ):
        raise ValueError(
            "last-row selection needs [B, T, width] states and a count per row"
        )
    lengths = attention_mask.sum(dim=1).tolist()
    for index, (count, length) in enumerate(zip(counts, lengths)):
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= length
            or int(attention_mask[index, :length].sum()) != length
        ):
            raise ValueError(
                "last-row selection needs right padding and 1..length counts"
            )
    longest = max(counts)
    selected = states.new_zeros((states.shape[0], longest, states.shape[2]))
    mask = torch.zeros(
        (states.shape[0], longest),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    for index, (count, length) in enumerate(zip(counts, lengths)):
        selected[index, :count] = states[index, length - count : length]
        mask[index, :count] = 1
    return selected, mask


def self_context_states(states, attention_mask, counts, suffix=0):
    """Keep each document's last ``counts[i] + suffix`` rows and the final
    question row whole, for one [docs..., question] tensor or a (final, layer)
    pair. ``suffix`` is the number of special rows the encoder tokenizer
    appends after a document's own ids (``encoder_special_suffix``), so the
    kept rows are the whole second copy followed by those rows."""
    if attention_mask.shape[0] != len(counts) + 1:
        raise ValueError("self context needs one kept count per document")
    if isinstance(suffix, bool) or not isinstance(suffix, int) or suffix < 0:
        raise ValueError("self context needs a nonnegative special-row suffix")
    counts = [c + suffix for c in counts] + [int(attention_mask[-1].sum())]
    if isinstance(states, tuple):
        final, mask = select_last_rows(states[0], attention_mask, counts)
        layer, layer_mask = select_last_rows(states[1], attention_mask, counts)
        if not torch.equal(mask, layer_mask):
            raise ValueError("dual layers disagree on kept rows")
        return (final, layer), mask
    return select_last_rows(states, attention_mask, counts)


def encoder_special_suffix(tokenizer):
    """How many special tokens the encoder tokenizer appends after a text's
    own ids (Qwen3-Embedding adds ``<|endoftext|>``; the decoder adds none).

    Those rows are encoder states without a document token behind them: they
    carry no copy-source id and no entity type, so id-aligned vectors are
    padded by this count to line up with a document's state rows
    (``feature_row_tags``) and the self-context keep includes them.
    """
    probe = "The brass key at Site A belongs to Nobody."
    full = list(tokenizer(probe).input_ids)
    plain = tokenizer.encode(probe, add_special_tokens=False)
    if not plain or full[: len(plain)] != plain:
        raise ValueError("encoder tokenizer prepends special tokens")
    return len(full) - len(plain)


def feature_row_tags(tags, suffix):
    """Id-aligned entity tags [T] -> tags over one document's state rows
    [T + suffix]: the appended special rows are untyped (``ENTITY_TYPE_NONE``)."""
    if tags.ndim != 1 or tags.dtype != torch.long:
        raise ValueError("feature row tags need a [T] long vector")
    if isinstance(suffix, bool) or not isinstance(suffix, int) or suffix < 0:
        raise ValueError("feature row tags need a nonnegative special-row suffix")
    if suffix == 0:
        return tags
    return torch.cat([tags, tags.new_zeros(suffix)])


def document_token_ids(tokenizer, documents):
    """Concatenated token ids of ``documents`` (no special tokens) as a LongTensor."""
    if not documents or any(not isinstance(d, str) for d in documents):
        raise ValueError("document token ids need a nonempty list of texts")
    ids = [
        i
        for text in documents
        for i in tokenizer.encode(text, add_special_tokens=False)
    ]
    if not ids:
        raise ValueError("documents tokenized to nothing")
    return torch.tensor(ids, dtype=torch.long)


def assert_shared_token_ids(
    encoder_tokenizer, decoder_tokenizer, documents, self_context=False
):
    """The copy head reads decoder embeddings at the document token ids; that is
    exact only while the encoder tokenizes the same documents to the same ids
    (under ``self_context`` the single ids the kept states stand for)."""
    if self_context:
        encoder_ids = torch.cat(self_context_documents(encoder_tokenizer, documents)[2])
        decoder_ids = torch.cat(self_context_documents(decoder_tokenizer, documents)[2])
    else:
        encoder_ids = document_token_ids(encoder_tokenizer, documents)
        decoder_ids = document_token_ids(decoder_tokenizer, documents)
    if not torch.equal(encoder_ids, decoder_ids):
        raise ValueError("encoder and decoder tokenizers disagree on document ids")
    return decoder_ids


def token_ids_embeddings(embedding, ids):
    """Frozen decoder input-embedding rows for ``ids`` [T] -> [T, width] FP32."""
    if embedding.weight.requires_grad:
        raise ValueError("copy sources need frozen decoder embeddings")
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("copy sources need a nonempty [T] id vector")
    with torch.no_grad():
        return embedding(ids.to(embedding.weight.device)).detach().float()


def document_token_embeddings(tokenizer, embedding, documents):
    """Copy sources for one case -> ``(embeddings [T, width] FP32, ids [T])``."""
    ids = document_token_ids(tokenizer, documents)
    return token_ids_embeddings(embedding, ids), ids


def pad_token_embeddings(rows):
    """Right-pad per-case ``(embeddings [T_i, width], ids [T_i])`` pairs into
    ``(embeddings [B, T, width], ids [B, T], mask [B, T])``; mask is True at
    real tokens and padded ids are zero."""
    if not rows or any(e.ndim != 2 or e.shape[0] == 0 for e, _ in rows):
        raise ValueError("token embedding batch needs nonempty [T, width] rows")
    width, longest = rows[0][0].shape[1], max(e.shape[0] for e, _ in rows)
    batch = rows[0][0].new_zeros((len(rows), longest, width))
    ids = torch.zeros((len(rows), longest), dtype=torch.long, device=batch.device)
    mask = torch.zeros((len(rows), longest), dtype=torch.bool, device=batch.device)
    for index, (embeddings, row_ids) in enumerate(rows):
        if embeddings.shape[1] != width or row_ids.shape != (embeddings.shape[0],):
            raise ValueError("token embedding batch geometry differs")
        batch[index, : embeddings.shape[0]] = embeddings
        ids[index, : embeddings.shape[0]] = row_ids.to(ids.device)
        mask[index, : embeddings.shape[0]] = True
    return batch, ids, mask


def copy_head_state(copy_head):
    return {k: v.detach().cpu().clone() for k, v in copy_head.state_dict().items()}


def slot_refiner_state(refiner):
    return {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}


class SlotRefiner(nn.Module):
    """Feedback between looped reader passes: causal self-attention over the
    previous pass's normalized slots, then a zero-initialized projection.

    The reader fills its slots in one parallel shot, so nothing stops two
    slots from copying the same name subword or skipping one. A second pass
    whose initial slot ``i`` is the learned query plus a function of what
    slots ``<= i`` held after the first pass lets the readout condition on its
    own earlier choices (a looped, pointer-network-like refinement). The zero
    projection makes pass two identical to pass one at init, so a child adding
    loops reproduces its parent exactly.
    """

    def __init__(self, width=2560, heads=8):
        super().__init__()
        if (
            isinstance(width, bool)
            or not isinstance(width, int)
            or width < 1
            or isinstance(heads, bool)
            or not isinstance(heads, int)
            or heads < 1
            or width % heads
        ):
            raise ValueError("slot refiner needs a positive width divisible by heads")
        self.width = width
        self.norm = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads, batch_first=True)
        self.proj = nn.Linear(width, width)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
        self.float()

    def forward(self, prev_slots):
        """``prev_slots`` [B, S, width] -> feedback [B, S, width] FP32; slot
        ``i`` attends only to slots ``<= i``."""
        if (
            not torch.is_tensor(prev_slots)
            or prev_slots.ndim != 3
            or prev_slots.shape[-1] != self.width
            or prev_slots.shape[1] < 1
            or not prev_slots.is_floating_point()
        ):
            raise ValueError("slot refiner needs [B, S, width] floating slots")
        rows = self.norm(prev_slots.float())
        count = rows.shape[1]
        causal = torch.ones(count, count, dtype=torch.bool, device=rows.device).triu(1)
        attended, _ = self.attn(rows, rows, rows, attn_mask=causal, need_weights=False)
        return self.proj(attended)


def reader_pass(reader, docs, mask, question, initial_slots):
    """One upstream ``MemoryModule.forward`` from the given initial slots.

    Reproduces the pinned forward step for step (question conditioning,
    document norm, cross-attention + feed-forward blocks, final norm and
    projection) with the reader's own submodules, so with
    ``reader.memory_queries.expand(B, -1, -1)`` the memories equal
    ``reader(docs, mask, question)`` exactly. Returns ``(memories [B, S,
    output], final_slots [B, S, width])`` where the final slots are the
    normalized pre-projection states a later pass feeds back on.
    """
    config = reader.config
    if config.retrieval_init or config.num_self_attn_layers > 0:
        raise ValueError("looped reading needs learned queries and no document encoder")
    if (
        not torch.is_tensor(initial_slots)
        or initial_slots.ndim != 3
        or initial_slots.shape[0] != docs.shape[0]
        or initial_slots.shape[1] != reader.num_memories
        or initial_slots.shape[2] != reader.memory_queries.shape[-1]
    ):
        raise ValueError("initial slots need [B, slots, width] geometry")
    question = question.unsqueeze(1)
    conditioned, _ = reader.query_cond_attn(initial_slots, question, question)
    slots = reader.query_cond_norm(initial_slots + conditioned)
    rows = reader.doc_norm(docs)
    for cross_attn_norm, cross_attn, ffn_norm, ffn in zip(
        reader.cross_attn_norms,
        reader.cross_attn_layers,
        reader.ffn_norms,
        reader.ffn_layers,
        strict=True,
    ):
        attended, _ = cross_attn(cross_attn_norm(slots), rows, rows, key_padding_mask=mask)
        slots = slots + reader.attn_dropout(attended)
        slots = slots + ffn(ffn_norm(slots))
    final = reader.layer_norm(slots)
    return reader.memory_projection(final), final


def looped_read(reader, refiner, docs, mask, question, loops=1):
    """Memories [B, S, output] from ``loops`` reader passes; pass ``k > 1``
    starts from the learned queries plus the refiner's feedback on pass
    ``k - 1``'s final slots. ``loops == 1`` is the plain reader call."""
    if isinstance(loops, bool) or not isinstance(loops, int) or not 1 <= loops <= 4:
        raise ValueError("reader loops must be an integer in 1..4")
    if (loops > 1) != (refiner is not None):
        raise ValueError("looped reading needs a slot refiner exactly when loops > 1")
    if loops == 1:
        return reader(docs, mask, question)
    base = reader.memory_queries.expand(docs.shape[0], -1, -1)
    memories, final = reader_pass(reader, docs, mask, question, base)
    for _ in range(loops - 1):
        memories, final = reader_pass(
            reader, docs, mask, question, base + refiner(final).to(base.dtype)
        )
    return memories


def lexical_slot_weights(tokenizer, case, record, slots, name_weight, truncate):
    """Per-slot loss weights: the asked name's value tokens get ``name_weight``.

    Name subwords are the last slots to converge; the weights are normalized to
    mean one so the overall lexical scale is unchanged.
    """
    weights = torch.ones(slots)
    if name_weight == 1.0 or asked_field(case["question"]) != "name":
        return weights
    text = lexical_record_text(case, record)
    ids = lexical_target_ids(tokenizer, text, truncate, slots)
    label = len(tokenizer.encode(" key holder", add_special_tokens=False))
    value = text.removeprefix(" key holder ").split(" at ")[0]
    count = len(tokenizer.encode(" " + value, add_special_tokens=False))
    if value == "UNKNOWN" or label + count > len(ids):
        return weights
    weights[label : label + count] = name_weight
    return weights * (slots / weights.sum())


def lexical_token_ce_loss(
    predicted, target_ids, teacher, slots=RELEASED_MEMORY_SLOTS, weights=None
):
    """Per slot: CE over the whole vocabulary of cosine/tau logits, plus the same
    squared log-norm ratio toward the target rows as the cosine objective.

    The contrast against every other token penalizes collapsing all slots onto
    the common embedding direction, which pure cosine regression rewards early.
    ``weights`` ([B, slots] or [slots], mean one) rebalance the per-slot CE.
    """
    if predicted.ndim == 2:
        predicted, target_ids = predicted.unsqueeze(0), target_ids.unsqueeze(0)
        if weights is not None and weights.ndim == 1:
            weights = weights.unsqueeze(0)
    if (
        predicted.ndim != 3
        or predicted.shape[1] != slots
        or target_ids.shape != predicted.shape[:2]
        or predicted.shape[2] != teacher.table.shape[1]
    ):
        raise ValueError(
            f"token supervision requires predicted [B, {slots}, width] and ids"
        )
    with torch.autocast(predicted.device.type, enabled=False):
        # One vocabulary matmul for the whole batch: the table read, not the
        # arithmetic, dominates on unified-memory devices.
        predicted = predicted.float().reshape(-1, predicted.shape[2])
        ids = target_ids.reshape(-1)
        unit = F.normalize(predicted, dim=-1)
        logits = unit @ teacher.table.T / teacher.temperature
        if weights is None:
            ce = F.cross_entropy(logits, ids)
        else:
            if tuple(weights.shape) != tuple(target_ids.shape):
                raise ValueError("slot weights must match the target id geometry")
            flat = weights.reshape(-1).to(logits.device, torch.float32)
            ce = (
                F.cross_entropy(logits, ids, reduction="none") * flat
            ).sum() / flat.sum()
        target_norm = teacher.norms.index_select(0, ids)
        log_ratio = torch.log(predicted.norm(dim=-1).clamp_min(1e-6)) - torch.log(
            target_norm
        )
        return ce + 0.5 * log_ratio.square().mean()


def adapter_digest(adapters):
    """Hash of the adapter tensors; a feature cache is only valid while frozen."""
    digest_state = hashlib.sha256()
    for key, value in sorted(adapter_state(adapters).items()):
        digest_state.update(key.encode())
        digest_state.update(value.float().contiguous().numpy().tobytes())
    return digest_state.hexdigest()


def cached_case_states(cache, texts):
    """Right-pad cached per-text encoder states into [n, T, width] plus mask."""
    rows = [cache[text] for text in texts]
    if not rows or any(r.ndim != 2 or r.shape[0] == 0 for r in rows):
        raise ValueError("cached states need nonempty [tokens, width] rows")
    longest, width = max(r.shape[0] for r in rows), rows[0].shape[1]
    states = rows[0].new_zeros((len(rows), longest, width))
    mask = torch.zeros((len(rows), longest), dtype=torch.long)
    for index, row in enumerate(rows):
        if row.shape[1] != width:
            raise ValueError("cached state widths differ")
        states[index, : row.shape[0]] = row
        mask[index, : row.shape[0]] = 1
    return states, mask


def cached_case_states_dual(cache, texts):
    """Dual-layer cache entries are ``{"final": rows, "layer": rows}`` per text;
    pad both alike -> (final [n, T, width], layer [n, T, width], mask)."""
    final, mask = cached_case_states({t: cache[t]["final"] for t in texts}, texts)
    layer, layer_mask = cached_case_states({t: cache[t]["layer"] for t in texts}, texts)
    if not torch.equal(mask, layer_mask):
        raise ValueError("dual cache layers disagree on token counts")
    return final, layer, mask


def warmup_lr_scale(step, total, schedule, ramp=100):
    """Learning-rate multiplier for warmup step `step` of `total`.

    ``cosine``: linear ramp over the first ``ramp`` steps, then cosine decay to
    a 1% floor at the end of the warmup phase (the sweep's best setting).
    """
    if schedule == "constant":
        return 1.0
    if schedule != "cosine" or not 0 <= step < total:
        raise ValueError("unknown warmup schedule or step outside the phase")
    ramp = min(ramp, max(total // 10, 1))
    if step < ramp:
        return (step + 1) / ramp
    progress = (step - ramp) / max(total - ramp, 1)
    return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress))


def pad_document_batch(docs):
    """Stack per-case document features [1, M_i, W] into [B, M, W] plus a
    key padding mask with True at padded rows."""
    if not docs or any(d.ndim != 3 or d.shape[0] != 1 for d in docs):
        raise ValueError("document batch needs [1, M, width] entries")
    width, longest = docs[0].shape[2], max(d.shape[1] for d in docs)
    batch = docs[0].new_zeros((len(docs), longest, width))
    mask = torch.ones((len(docs), longest), dtype=torch.bool, device=batch.device)
    for index, d in enumerate(docs):
        if d.shape[2] != width:
            raise ValueError("document batch widths differ")
        batch[index, : d.shape[1]] = d[0]
        mask[index, : d.shape[1]] = False
    return batch, mask


def rescale_memory_projection(reader, memories, target_norm):
    """Scale the reader's final projection so predicted memories match the
    teacher embedding norm. A pure reparametrization of the same module.

    The released reader emits memories two orders of magnitude larger than the
    decoder's token embeddings, which the lexical objective cannot close.
    """
    norms = torch.stack([m.detach().float().norm(dim=-1).mean() for m in memories])
    current = float(norms.mean())
    if not current > 0 or not target_norm > 0 or not norms.isfinite().all():
        raise ValueError("memory rescale needs finite positive norms")
    factor = target_norm / current
    projection = reader.memory_projection
    with torch.no_grad():
        projection.weight.mul_(factor)
        if projection.bias is not None:
            projection.bias.mul_(factor)
    return factor


def lexical_settings(weight, record, objective="cosine"):
    if objective not in LEXICAL_OBJECTIVES:
        raise ValueError("unknown lexical objective")
    if record not in LEXICAL_RECORDS or not 0 <= weight <= 10:
        raise ValueError(
            "lexical loss weight must be within [0, 10] with a known record"
        )
    if (weight > 0) != (record != "none"):
        raise ValueError(
            "lexical supervision needs both a positive weight and a record form"
        )
    return {
        "lexical_loss_weight": weight,
        "lexical_record": record,
        "lexical_objective": objective,
        "lexical_recipe": LEXICAL_RECIPES[objective] if weight > 0 else None,
    }


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
    if any("cohort" in r for r in rows):
        result["cohorts"] = {
            cohort: scores(
                [
                    {k: v for k, v in r.items() if k != "cohort"}
                    for r in rows
                    if r.get("cohort") == cohort
                ]
            )
            for cohort in sorted({r.get("cohort", "unspecified") for r in rows})
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
    train.add_argument("--memory-slots", type=int, default=RELEASED_MEMORY_SLOTS)
    train.add_argument("--reader-lr", type=float, default=5e-5)
    train.add_argument("--encoder-lr", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=23)
    train.add_argument("--resume", action="store_true")
    train.add_argument("--continue-from", type=Path)
    train.add_argument("--initialize-from", type=Path)
    train.add_argument("--reconstruct-every", type=int, default=0)
    train.add_argument(
        "--sampling", choices=("row", "counterfactual-group"), default="row"
    )
    train.add_argument(
        "--document-features", choices=DOCUMENT_FEATURE_MODES, default="pooled"
    )
    train.add_argument("--lexical-loss-weight", type=float, default=0.0)
    train.add_argument("--lexical-record", choices=LEXICAL_RECORDS, default="none")
    train.add_argument(
        "--memory-scale-init", choices=("none", "lexical-target"), default="none"
    )
    train.add_argument("--lexical-warmup-steps", type=int, default=0)
    train.add_argument(
        "--lexical-objective", choices=LEXICAL_OBJECTIVES, default="cosine"
    )
    train.add_argument("--lexical-warmup-batch", type=int, default=1)
    train.add_argument("--lexical-warmup-cache", type=Path)
    train.add_argument("--lexical-warmup-train-encoder", action="store_true")
    train.add_argument("--joint-reader-lr", type=float)
    train.add_argument(
        "--lexical-warmup-schedule", choices=("constant", "cosine"), default="constant"
    )
    train.add_argument("--lexical-name-weight", type=float, default=1.0)
    train.add_argument("--lora-targets", default=",".join(DEFAULT_LORA_TARGETS))
    train.add_argument("--encoder-layer", type=int)
    train.add_argument("--dual-layer", type=int)
    train.add_argument("--copy-head", action="store_true")
    train.add_argument("--copy-pointer-weight", type=float, default=0.0)
    # Multiplier on the gate BCE inside the pointer term; a label slot whose
    # gate rose above 0.5 glued two value runs in 4 of 480 wide cases.
    train.add_argument("--copy-gate-weight", type=float, default=1.0)
    train.add_argument("--copy-head-lr", type=float)
    train.add_argument("--entity-tag-lr", type=float)
    train.add_argument("--entity-tag-max-norm", type=float)
    train.add_argument("--reader-loops", type=int, default=1)
    train.add_argument("--slot-refiner-lr", type=float)
    train.add_argument("--document-space-prefix", action="store_true")
    train.add_argument("--document-self-context", action="store_true")
    train.add_argument("--document-pooled-context", action="store_true")
    train.add_argument("--entity-tags", choices=ENTITY_TAG_MODES, default="none")
    train.add_argument("--copy-restrict-by-type", action="store_true")
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
    slots = memory_slot_count(args.memory_slots)
    targets = lora_targets(args.lora_targets)
    if args.encoder_layer is not None and args.encoder_layer < 1:
        raise ValueError("encoder layer must be within 1..num_hidden_layers")
    check_dual_layer_flags(args.document_features, args.encoder_layer, args.dual_layer)
    if args.reconstruct_every not in (0, 2, 3, 4):
        raise ValueError("reconstruction cadence must be disabled or 2/3/4")
    if args.initialize_from is not None and args.continue_from is not None:
        raise ValueError(
            "weight initialization and optimizer continuation are distinct"
        )
    if args.sampling == "counterfactual-group" and args.accum != 8:
        raise ValueError("counterfactual-group sampling requires accumulation eight")
    lexical = lexical_settings(
        args.lexical_loss_weight, args.lexical_record, args.lexical_objective
    )
    if args.memory_scale_init != "none" and args.lexical_loss_weight <= 0:
        raise ValueError("memory scale initialization requires lexical supervision")
    if not 0 <= args.lexical_warmup_steps <= args.steps or (
        args.lexical_warmup_steps and args.lexical_loss_weight <= 0
    ):
        raise ValueError("lexical warmup needs lexical supervision within the budget")
    if not 1 <= args.lexical_warmup_batch <= 64 or (
        args.lexical_warmup_batch > 1 and not args.lexical_warmup_steps
    ):
        raise ValueError("warmup batch must be 1..64 and needs warmup steps")
    if args.joint_reader_lr is not None and (
        not 0 < args.joint_reader_lr <= 1e-3 or not args.lexical_warmup_steps
    ):
        raise ValueError("joint reader learning rate needs a warmup phase")
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
        "memory_slots": slots,
        "rank": args.rank,
        "alpha": args.alpha,
        "target_modules": list(targets),
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
    protocol["sampling"] = args.sampling
    protocol["sampling_recipe"] = (
        "code-name-pairs-two-missing-repeat-pair-v1"
        if args.sampling == "counterfactual-group"
        else None
    )
    protocol["reconstruct_every"] = args.reconstruct_every
    protocol["reconstruction_system"] = (
        RECONSTRUCTION_SYSTEM if args.reconstruct_every else None
    )
    protocol["document_features"] = args.document_features
    protocol["encoder_layer"] = args.encoder_layer
    protocol["dual_layer"] = args.dual_layer
    protocol.update(lexical)
    protocol["memory_scale_init"] = args.memory_scale_init
    protocol["lexical_warmup_steps"] = args.lexical_warmup_steps
    protocol["lexical_warmup_batch"] = args.lexical_warmup_batch
    protocol["joint_reader_lr"] = args.joint_reader_lr
    protocol["lexical_warmup_schedule"] = args.lexical_warmup_schedule
    protocol["lexical_name_weight"] = args.lexical_name_weight
    if not 1.0 <= args.lexical_name_weight <= 8.0 or (
        args.lexical_name_weight != 1.0
        and (
            args.lexical_objective != "token-ce"
            or args.lexical_record not in ("asked-site", "asked-site-bare")
        )
    ):
        raise ValueError(
            "name weighting needs the token-ce objective and asked-site record"
        )
    if args.lexical_warmup_schedule != "constant" and not args.lexical_warmup_steps:
        raise ValueError("a warmup schedule needs warmup steps")
    protocol["lexical_warmup_cache"] = args.lexical_warmup_cache is not None
    protocol["lexical_warmup_train_encoder"] = args.lexical_warmup_train_encoder
    protocol["copy_head"] = bool(args.copy_head)
    protocol["copy_head_temperature"] = (
        COPY_HEAD_TEMPERATURE if args.copy_head else None
    )
    if not 0 <= args.copy_pointer_weight <= 10:
        raise ValueError("copy pointer weight must be within 0..10")
    if args.copy_pointer_weight and (
        not args.copy_head
        or args.lexical_loss_weight <= 0
        or args.lexical_objective != "token-ce"
    ):
        raise ValueError(
            "pointer supervision needs --copy-head and the token-ce lexical objective"
        )
    if args.copy_head_lr is not None and (
        not 0 < args.copy_head_lr <= 1e-3 or not args.copy_head
    ):
        raise ValueError("copy head learning rate needs --copy-head and 0<lr<=1e-3")
    protocol["copy_pointer_weight"] = args.copy_pointer_weight
    if not 0 < args.copy_gate_weight <= 10:
        raise ValueError("copy gate weight must be within (0, 10]")
    protocol["copy_gate_weight"] = args.copy_gate_weight
    protocol["copy_head_lr"] = args.copy_head_lr
    # The tag table is five rows; at the reader's rate it moves a few percent
    # of a unit row in 2048 steps (copy8: norms 0.02-0.11), so it may take its
    # own, higher rate (same cosine schedule) to test typing at all.
    if args.entity_tag_lr is not None and (
        not 0 < args.entity_tag_lr <= 1e-1 or args.entity_tags == "none"
    ):
        raise ValueError("entity tag learning rate needs --entity-tags and 0<lr<=1e-1")
    protocol["entity_tag_lr"] = args.entity_tag_lr
    # Type vectors are added at each row's norm, so a row norm above one
    # outweighs the token itself (copy10 at 5e-3: site 1.9, missing 3.2 by
    # step 256 and dev fell to 40/64); the cap renormalizes rows after every
    # optimizer step so typing stays a hint next to the token's identity.
    if args.entity_tag_max_norm is not None and (
        not 0 < args.entity_tag_max_norm <= 2 or args.entity_tags == "none"
    ):
        raise ValueError("entity tag norm cap needs --entity-tags and 0<cap<=2")
    protocol["entity_tag_max_norm"] = args.entity_tag_max_norm
    # Looped readout: the reader runs ``reader_loops`` passes and a slot
    # refiner (causal slot self-attention, zero-initialized projection) feeds
    # each pass's final slots into the next pass's initial slots.
    if isinstance(args.reader_loops, bool) or not 1 <= args.reader_loops <= 4:
        raise ValueError("--reader-loops must be 1..4")
    protocol["reader_loops"] = args.reader_loops
    if args.slot_refiner_lr is not None and (
        not 0 < args.slot_refiner_lr <= 1e-3 or args.reader_loops == 1
    ):
        raise ValueError("slot refiner learning rate needs --reader-loops > 1 and 0<lr<=1e-3")
    protocol["slot_refiner_lr"] = args.slot_refiner_lr
    reader_loops = args.reader_loops
    space_prefix = bool(args.document_space_prefix)
    protocol["document_space_prefix"] = space_prefix
    # Both index-time only: the encoder reads each document twice and keeps the
    # second copy's states, and/or every token row gets its document's pooled
    # vector mixed in; the question and the decoder are untouched.
    self_context = bool(args.document_self_context)
    pooled_context = bool(args.document_pooled_context)
    if pooled_context and args.document_features == "pooled":
        raise ValueError("pooled document context needs per-token document features")
    protocol["document_self_context"] = self_context
    protocol["document_pooled_context"] = pooled_context
    # The encoder tokenizer appends special rows after a document's ids; runs
    # recorded with this flag keep them in the self-context selection and tag
    # them as untyped, instead of the earlier count-only keep (the evaluator
    # reproduces whichever the protocol records).
    protocol["encoder_special_rows"] = True
    # Index-time entity typing: every document token row carries a learned
    # vector for its (oracle) entity type; the copy head may then be held to
    # the asked field's type on the value slots. Both default off.
    protocol["entity_tags"] = args.entity_tags
    entity_tags = entity_tags_enabled(protocol)
    if entity_tags and args.document_features not in (
        "tokens",
        "tokens-question",
        "tokens-dual",
    ):
        raise ValueError("entity tags need per-token document features")
    copy_restrict = bool(args.copy_restrict_by_type)
    if copy_restrict and (
        not args.copy_head or not entity_tags or args.lexical_record == "none"
    ):
        raise ValueError(
            "type-restricted copying needs --copy-head, --entity-tags and a"
            " lexical record"
        )
    protocol["copy_restrict_by_type"] = copy_restrict
    if args.lexical_warmup_train_encoder and (
        args.lexical_warmup_cache is not None
        or not args.lexical_warmup_steps
        or args.arm != "encoder-lora"
    ):
        raise ValueError(
            "training the encoder during warmup needs live encoding and adapters"
        )
    if args.lexical_warmup_cache is not None and not args.lexical_warmup_steps:
        raise ValueError("warmup feature cache needs a warmup phase")
    if args.initialize_from is not None:
        initialization = args.initialize_from.resolve()
        initial_protocol = candidate_proof(initialization, slots)
        if (
            initial_protocol["rank"],
            initial_protocol["alpha"],
            initial_protocol["arm"],
        ) != (args.rank, args.alpha, args.arm):
            raise ValueError("initialization adapter contract differs")
        parent_targets = tuple(
            initial_protocol.get("target_modules", list(DEFAULT_LORA_TARGETS))
        )
        initialization_adds_targets(parent_targets, targets)
        parent_copy_head = bool(initial_protocol.get("copy_head", False))
        if parent_copy_head and not args.copy_head:
            raise ValueError(
                "initialization parent trained a copy head; pass --copy-head"
            )
        parent_space_prefix = bool(initial_protocol.get("document_space_prefix", False))
        parent_self_context = bool(initial_protocol.get("document_self_context", False))
        parent_pooled_context = bool(
            initial_protocol.get("document_pooled_context", False)
        )
        parent_entity_tags = initial_protocol.get("entity_tags", "none")
        if entity_tags_enabled(initial_protocol) and not entity_tags:
            raise ValueError(
                "initialization parent trained entity tags; pass --entity-tags"
            )
        parent_copy_restrict = bool(
            initial_protocol.get("copy_restrict_by_type", False)
        )
        parent_reader_loops = initial_protocol.get("reader_loops", 1)
        reader_loops_enabled(initial_protocol)
        if parent_reader_loops > reader_loops:
            raise ValueError(
                "initialization parent read in more loops; pass at least --reader-loops "
                + str(parent_reader_loops)
            )
        protocol["initialization"] = {
            "directory": str(initialization),
            "candidate_sha256": digest(initialization / "candidate.json"),
            "optimizer_reset": True,
            # Reader/adapter weights are feature-mode agnostic; record the
            # parent's source geometry so a changed mode is explicit lineage.
            "parent_document_features": initial_protocol.get(
                "document_features", "pooled"
            ),
            # A 16-slot parent initializes a wider run: its reader loads whole
            # and the learned queries are then expanded to this run's count.
            "parent_memory_slots": initial_protocol.get(
                "memory_slots", RELEASED_MEMORY_SLOTS
            ),
            # Shared adapters restore whole; any added target modules start at
            # zero effect, so the parent's answers must still reproduce.
            "parent_target_modules": list(parent_targets),
            # Reader and adapter weights are layer-agnostic; a different source
            # layer feeds them new features and is explicit lineage.
            "parent_encoder_layer": initial_protocol.get("encoder_layer"),
            # Only present for a dual-layer parent so earlier lineage stays equal.
            **(
                {}
                if initial_protocol.get("dual_layer") is None
                else {"parent_dual_layer": initial_protocol["dual_layer"]}
            ),
            # Only present when a copy head is involved: a parent with one
            # restores it, a child adding one starts it fresh at init.
            **(
                {"parent_copy_head": parent_copy_head}
                if args.copy_head or parent_copy_head
                else {}
            ),
            # Only present when either side prefixes documents with a space:
            # a changed prefix re-segments the encoder input and the copy
            # sources, so it is explicit lineage like a changed layer.
            **(
                {"parent_document_space_prefix": parent_space_prefix}
                if space_prefix or parent_space_prefix
                else {}
            ),
            # Likewise for the two document-context mechanisms: each changes
            # the features the same weights read, so a change is lineage.
            **(
                {"parent_document_self_context": parent_self_context}
                if self_context or parent_self_context
                else {}
            ),
            **(
                {"parent_document_pooled_context": parent_pooled_context}
                if pooled_context or parent_pooled_context
                else {}
            ),
            # Only present when either side types entity tokens: a parent with
            # a tag table restores it, a child adding one starts at zero (the
            # parent's features reproduce exactly); dropping it is refused.
            **(
                {"parent_entity_tags": parent_entity_tags}
                if entity_tags or parent_entity_tags != "none"
                else {}
            ),
            # Only present when either side loops the reader: a parent's
            # refiner restores, a child adding loops starts it at zero (the
            # parent reproduces exactly); fewer loops than the parent is refused.
            **(
                {"parent_reader_loops": parent_reader_loops}
                if reader_loops > 1 or parent_reader_loops > 1
                else {}
            ),
            # Restricting the copy head by type changes its attention, so a
            # change is explicit lineage like a fresh copy head.
            **(
                {"parent_copy_restrict_by_type": parent_copy_restrict}
                if copy_restrict or parent_copy_restrict
                else {}
            ),
        }

    if args.resume:
        previous_protocol = json.loads((args.output / "protocol.json").read_text())
        if "initialization" in previous_protocol and args.initialize_from is None:
            protocol["initialization"] = previous_protocol["initialization"]
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
        if "initialization" in parent_protocol:
            protocol["initialization"] = parent_protocol["initialization"]
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
    if config["num_memories"] != RELEASED_MEMORY_SLOTS:
        raise ValueError("experiment requires released 16-slot initialization")
    proof = {
        "encoder": verify_base_model(args.encoder, "encoder"),
        "decoder": verify_base_model(args.decoder, "decoder"),
        "reader": reader_proof,
        "upstream": source_recipe(args.upstream_source),
    }
    save_json(args.output / "base-proofs.json", proof)
    if (
        args.initialize_from is not None
        and json.loads((args.initialize_from / "base-proofs.json").read_text()) != proof
    ):
        raise ValueError("initialization base assets differ")
    train_rows, dev_rows = load_split(args.data, "train"), load_split(args.data, "dev")
    if args.reconstruct_every and any("reconstruction" not in r for r in train_rows):
        raise ValueError("reconstruction fields missing from training data")
    if len({r["group"] for r in train_rows} & {r["group"] for r in dev_rows}):
        raise ValueError("train/dev group overlap")
    group_sampler = (
        CounterfactualGroupSampler(train_rows, args.seed)
        if args.sampling == "counterfactual-group"
        else None
    )
    tokenizer_e = AutoTokenizer.from_pretrained(
        args.encoder, local_files_only=True, trust_remote_code=False
    )
    tokenizer_e.padding_side = "right"
    encoder_suffix = encoder_special_suffix(tokenizer_e)
    if (
        args.initialize_from is not None
        and not args.resume
        and bool(initial_protocol.get("document_self_context", False))
        and not initial_protocol.get("encoder_special_rows", False)
    ):
        raise ValueError(
            "self-context parent kept count-only rows; continuation would change its features"
        )
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
    adapters = inject_lora(
        encoder, args.rank, args.alpha, args.arm == "encoder-lora", targets
    )
    encoder_layer = check_encoder_layer(
        args.encoder_layer, encoder.config.num_hidden_layers
    )
    dual_layer = check_dual_layer(args.dual_layer, encoder.config.num_hidden_layers)
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
    if args.initialize_from is not None and not args.resume:
        from safetensors.torch import load_file

        # The parent's reader loads at its own slot count (16 or this run's).
        expand_memory_queries(
            reader, protocol["initialization"]["parent_memory_slots"], args.seed
        )
        reader.load_state_dict(
            load_file(str(args.initialize_from / "reader.safetensors"))
        )
        restore_adapters(
            adapters,
            load_file(str(args.initialize_from / "encoder-lora.safetensors")),
            allow_missing=initialization_adds_targets(
                protocol["initialization"]["parent_target_modules"], targets
            ),
        )
    expand_memory_queries(reader, slots, args.seed)
    if reader.num_memories != slots or reader.memory_queries.shape[1] != slots:
        raise ValueError("reader memory slot expansion failed")
    decoder.config.use_cache = False
    if decoder.config.vocab_size != 151936:
        raise ValueError("decoder vocabulary must remain unchanged")
    vocabulary_teacher = (
        LexicalVocabularyTeacher(decoder.get_input_embeddings())
        if args.lexical_loss_weight > 0 and args.lexical_objective == "token-ce"
        else None
    )

    def case_documents(case):
        """The encoder's and copy head's view of a case's documents: the raw
        texts, or each with the single-space prefix when the flag is on."""
        return prefixed_documents(case["documents"], space_prefix)

    def case_texts(case):
        """Feature keys of a case: its (possibly prefixed) documents then the
        raw question; also the warmup cache keys."""
        return case_documents(case) + [case["question"]]

    def encoder_texts(case):
        """(texts the encoder tokenizes, kept token counts per document or
        None): under self-context each document is doubled and only the second
        copy's ``n`` states are kept; the question is always read whole."""
        documents = case_documents(case)
        if not self_context:
            return documents + [case["question"]], None
        doubled, counts, _, fallbacks = self_context_documents(tokenizer_e, documents)
        if fallbacks:
            result["document_self_context_fallbacks"] += fallbacks
        return doubled + [case["question"]], counts

    def document_ids(text):
        """Copy-source ids of one document: the ids the kept states stand for."""
        if self_context:
            return torch.tensor(
                self_context_tokens(tokenizer_d, text)[2], dtype=torch.long
            )
        return document_token_ids(tokenizer_d, [text])

    def check_budget(tokens):
        if tokens.input_ids.shape[1] > 128:
            raise ValueError(
                "encoder input exceeds explicit 128-token experiment budget"
                + (
                    " (documents are doubled under --document-self-context)"
                    if self_context
                    else ""
                )
            )

    def kept(states, attention_mask, counts):
        """Self-context keeps each document's last ``counts`` rows and the
        question whole; otherwise the states pass unchanged."""
        if counts is None:
            return states, attention_mask
        return self_context_states(states, attention_mask, counts, encoder_suffix)

    copy_head = None
    if args.copy_head:
        # Copy sources are decoder embedding rows at the document token ids;
        # the encoder must tokenize those documents identically for the copy
        # to be exact (also as the single ids behind the self-context states).
        assert_shared_token_ids(tokenizer_e, tokenizer_d, case_documents(train_rows[0]))
        if self_context:
            assert_shared_token_ids(
                tokenizer_e, tokenizer_d, case_documents(train_rows[0]), True
            )
        copy_head = CopyHead(4096, COPY_HEAD_TEMPERATURE).cuda()
        if (
            args.initialize_from is not None
            and not args.resume
            and protocol["initialization"]["parent_copy_head"]
        ):
            from safetensors.torch import load_file

            copy_head.load_state_dict(
                load_file(str(args.initialize_from / "copy-head.safetensors"))
            )
    slot_refiner = None
    if reader_loops > 1:
        slot_refiner = SlotRefiner(reader.memory_queries.shape[-1]).cuda()
        if (
            args.initialize_from is not None
            and not args.resume
            and protocol["initialization"]["parent_reader_loops"] > 1
        ):
            from safetensors.torch import load_file

            slot_refiner.load_state_dict(
                load_file(str(args.initialize_from / "slot-refiner.safetensors"))
            )
    tag_embedding = None
    if entity_tags:
        tag_embedding = EntityTagEmbedding(encoder.config.hidden_size).cuda()
        if (
            args.initialize_from is not None
            and not args.resume
            and protocol["initialization"]["parent_entity_tags"] != "none"
        ):
            from safetensors.torch import load_file

            tag_embedding.load_state_dict(
                load_file(str(args.initialize_from / "entity-tags.safetensors"))
            )
        # Every training document must tag cleanly before any update runs.
        for case in train_rows:
            for text in case_documents(case):
                document_entity_tags(tokenizer_e, text, case, self_context)
    if copy_restrict:
        truncate = args.lexical_record in TRUNCATED_RECORDS
        for case in train_rows:
            lexical_value_slots(tokenizer_d, case, args.lexical_record, truncate, slots)

    def lexical_loss_for(memories, case):
        text = lexical_record_text(case, args.lexical_record)
        truncate = args.lexical_record in TRUNCATED_RECORDS
        if vocabulary_teacher is not None:
            ids = torch.tensor(
                lexical_target_ids(tokenizer_d, text, truncate, slots), device="cuda"
            )
            return lexical_token_ce_loss(
                memories,
                ids,
                vocabulary_teacher,
                slots,
                lexical_slot_weights(
                    tokenizer_d,
                    case,
                    args.lexical_record,
                    slots,
                    args.lexical_name_weight,
                    truncate,
                ).to("cuda"),
            )
        return lexical_alignment_loss(
            memories,
            lexical_targets(
                tokenizer_d,
                decoder.get_input_embeddings(),
                text,
                truncate=truncate,
                slots=slots,
            ),
            slots,
        )

    def lexical_loss_for_batch(memories, cases):
        """[B, slots, width] memories against their cases' records, one table read."""
        if vocabulary_teacher is None:
            return torch.stack(
                [lexical_loss_for(memories[j], case) for j, case in enumerate(cases)]
            ).mean()
        truncate = args.lexical_record in TRUNCATED_RECORDS
        ids = torch.tensor(
            [
                lexical_target_ids(
                    tokenizer_d,
                    lexical_record_text(case, args.lexical_record),
                    truncate,
                    slots,
                )
                for case in cases
            ],
            device="cuda",
        )
        weights = torch.stack(
            [
                lexical_slot_weights(
                    tokenizer_d,
                    case,
                    args.lexical_record,
                    slots,
                    args.lexical_name_weight,
                    truncate,
                )
                for case in cases
            ]
        ).to("cuda")
        return lexical_token_ce_loss(memories, ids, vocabulary_teacher, slots, weights)

    if args.lexical_loss_weight > 0:
        # Every training record must fit the memory slots before any update runs.
        for case in train_rows:
            lexical_targets(
                tokenizer_d,
                decoder.get_input_embeddings(),
                lexical_record_text(case, args.lexical_record),
                truncate=args.lexical_record in TRUNCATED_RECORDS,
                slots=slots,
            )
    projection_parameters = list(reader.memory_projection.parameters())
    projection_ids = {id(p) for p in projection_parameters}
    groups = [
        {
            "params": [p for p in reader.parameters() if id(p) not in projection_ids],
            "lr": args.reader_lr,
            "reader_body": True,
        },
        # Rescaling the projection by f and training it at lr*f is the same as
        # training the unscaled module with a folded output scale (AdamW).
        {"params": projection_parameters, "lr": args.reader_lr, "projection": True},
    ]
    copy_head_lr = (
        args.copy_head_lr if args.copy_head_lr is not None else args.reader_lr
    )
    if copy_head is not None:
        groups.append(
            {
                "params": list(copy_head.parameters()),
                "lr": copy_head_lr,
                "copy_head": True,
            }
        )
    entity_tag_lr = (
        args.entity_tag_lr if args.entity_tag_lr is not None else args.reader_lr
    )
    if tag_embedding is not None:
        # The tag table follows the reader body's schedule at its own rate.
        groups.append(
            {
                "params": list(tag_embedding.parameters()),
                "lr": entity_tag_lr,
                "entity_tags": True,
            }
        )
    slot_refiner_lr = (
        args.slot_refiner_lr if args.slot_refiner_lr is not None else args.reader_lr
    )
    if slot_refiner is not None:
        # The refiner follows the reader body's schedule at its own rate.
        groups.append(
            {
                "params": list(slot_refiner.parameters()),
                "lr": slot_refiner_lr,
                "slot_refiner": True,
            }
        )
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
        **(
            {}
            if copy_head is None
            else {
                "copy_head_parameters": sum(p.numel() for p in copy_head.parameters())
            }
        ),
        **(
            {}
            if tag_embedding is None
            else {
                "entity_tag_parameters": sum(
                    p.numel() for p in tag_embedding.parameters()
                )
            }
        ),
        **(
            {}
            if slot_refiner is None
            else {
                "slot_refiner_parameters": sum(
                    p.numel() for p in slot_refiner.parameters()
                )
            }
        ),
        "encoder_trainable_parameters": sum(
            p.numel() for p in encoder.parameters() if p.requires_grad
        ),
        "encoder_base_and_decoder_frozen": True,
        # Documents whose doubled text re-segmented their first word, so the
        # kept states stand for the ids of the space-prefixed text instead.
        **({"document_self_context_fallbacks": 0} if self_context else {}),
        "encoder_special_suffix": encoder_suffix,
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
        if copy_head is not None:
            if "copy_head" not in checkpoint:
                raise ValueError("checkpoint lacks the copy head state")
            copy_head.load_state_dict(checkpoint["copy_head"])
        if tag_embedding is not None:
            if "entity_tags" not in checkpoint:
                raise ValueError("checkpoint lacks the entity tag state")
            tag_embedding.load_state_dict(checkpoint["entity_tags"])
        if slot_refiner is not None:
            if "slot_refiner" not in checkpoint:
                raise ValueError("checkpoint lacks the slot refiner state")
            slot_refiner.load_state_dict(checkpoint["slot_refiner"])
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
        if copy_head is not None:
            copy_head.train(training)
        if tag_embedding is not None:
            tag_embedding.train(training)
        if slot_refiner is not None:
            slot_refiner.train(training)

    def read(docs, mask, query):
        """The reader over padded document features, looped through the slot
        refiner when the protocol reads in more than one pass."""
        return looped_read(reader, slot_refiner, docs, mask, query, reader_loops)

    def document_tags(text, case):
        """Oracle entity types of one document's feature rows (and copy
        sources): cached warmup tags when present, else computed on the
        encoder tokenization the kept states stand for."""
        if text in warmup_tags:
            return warmup_tags[text]
        return document_entity_tags(tokenizer_e, text, case, self_context)

    def case_copy_tags(case):
        """[T] entity types over a case's copy-source ids (document tokens)."""
        return torch.cat([document_tags(text, case) for text in case_documents(case)])

    def case_tags(case):
        """[M] entity types over a case's document state rows (each document's
        ids then its untyped special rows), or None without tags."""
        if tag_embedding is None:
            return None
        return torch.cat(
            [
                feature_row_tags(document_tags(text, case), encoder_suffix)
                for text in case_documents(case)
            ]
        )

    copy_gates = []

    def copy_sources(cases):
        """Frozen decoder embeddings of each case's document tokens, padded ->
        ([B, T, width] FP32, [B, T] ids, [B, T] mask). Cached warmup runs reuse the ids
        stored next to the encoder states; live runs tokenize with the decoder
        tokenizer (asserted equal to the encoder's ids at startup)."""
        embedding = decoder.get_input_embeddings()
        rows = []
        for case in cases:
            ids = torch.cat(
                [
                    warmup_ids[text] if text in warmup_ids else document_ids(text)
                    for text in case_documents(case)
                ]
            )
            rows.append((token_ids_embeddings(embedding, ids.to("cuda")), ids))
        return pad_token_embeddings(rows)

    def copy_detailed(memories, cases):
        """Copy head over [B, slots, width] memories of ``cases`` -> (output,
        copy_logits [B, slots, T], gate_logits [B, slots], document ids [B, T],
        document mask [B, T]); the logits feed pointer supervision."""
        if copy_head is None:
            raise RuntimeError("pointer supervision needs the copy head")
        sources, ids, mask = copy_sources(cases)
        slot_mask = copy_slot_mask(cases, ids, mask) if copy_restrict else None
        output, copy_logits, gate_logits = copy_head.forward_detailed(
            memories, sources, mask, slot_mask
        )
        copy_gates.append(float(torch.sigmoid(gate_logits.detach().float()).mean()))
        return output, copy_logits, gate_logits, ids, mask, slot_mask

    def copy_slot_mask(cases, ids, mask):
        """Type restriction [B, slots, T] for the copy head: the asked value's
        record slots copy only tokens of the asked field's entity type (or the
        "not recorded" marker); label, site and pad slots keep every token."""
        tags = pad_entity_tags([case_copy_tags(case) for case in cases]).to(ids.device)
        if tags.shape != ids.shape:
            raise ValueError("entity tags and copy-source ids differ in geometry")
        truncate = args.lexical_record in TRUNCATED_RECORDS
        value_slots = torch.stack(
            [
                lexical_value_slots(
                    tokenizer_d, case, args.lexical_record, truncate, slots
                )
                for case in cases
            ]
        ).to(ids.device)
        fields = [asked_field(case["question"]) for case in cases]
        return copy_type_mask(tags, mask, value_slots, fields)

    def copy(memories, cases):
        """Apply the copy head to [B, slots, width] memories of ``cases``."""
        if copy_head is None:
            return memories
        return copy_detailed(memories, cases)[0]

    pointer_excluded = (
        copy_excluded_ids(tokenizer_d) if args.copy_pointer_weight > 0 else None
    )
    pointer_values, gate_values, copyable_fractions = [], [], []

    def pointer_term(pointer_inputs, cases):
        """Weighted pointer + gate loss of a batch's copy-head logits against
        the cases' lexical records; records the step statistics."""
        copy_logits, gate_logits, doc_ids, doc_mask, slot_mask = pointer_inputs
        truncate = args.lexical_record in TRUNCATED_RECORDS
        target_ids = torch.tensor(
            [
                lexical_target_ids(
                    tokenizer_d,
                    lexical_record_text(case, args.lexical_record),
                    truncate,
                    slots,
                )
                for case in cases
            ],
            device=doc_ids.device,
        )
        copyable, match = copyable_mask(target_ids, doc_ids, doc_mask, pointer_excluded)
        if slot_mask is not None:
            # A restricted slot can only copy from its type's tokens; a target
            # occurrence outside them carries no mass and is not copyable.
            match = match & slot_mask
            copyable = match.any(dim=-1)
        pointer_loss, gate_loss = pointer_losses(
            copy_logits, gate_logits, match, copyable, doc_mask
        )
        if not torch.isfinite(pointer_loss) or not torch.isfinite(gate_loss):
            raise RuntimeError("nonfinite pointer loss")
        pointer_values.append(float(pointer_loss.detach()))
        gate_values.append(float(gate_loss.detach()))
        copyable_fractions.append(float(copyable.float().mean()))
        return args.copy_pointer_weight * (
            pointer_loss + args.copy_gate_weight * gate_loss
        )

    def encode(tokens):
        """Encoder states for the configured source: one tensor, or the
        (final, intermediate) pair for ``tokens-dual``."""
        if dual_layer is None:
            return encoder_states(encoder, tokens, encoder_layer)
        return encoder_states_pair(encoder, tokens, dual_layer)

    def select(states, begin, end):
        if dual_layer is None:
            return states[begin:end]
        return tuple(s[begin:end] for s in states)

    def features(states, attention_mask, tags=None):
        if dual_layer is None:
            return document_features(
                states,
                attention_mask,
                args.document_features,
                pooled_context=pooled_context,
                tag_embedding=tag_embedding,
                tags=tags,
            )
        return document_features_dual(
            states[0],
            states[1],
            attention_mask,
            pooled_context=pooled_context,
            tag_embedding=tag_embedding,
            tags=tags,
        )

    def memory(case, training, raw=False, detailed=False):
        """One case's memories [slots, 4096]; ``raw`` skips the copy head and
        ``detailed`` also returns the copy head's pointer inputs."""
        texts, counts = encoder_texts(case)
        tokens = tokenizer_e(texts, padding=True, return_tensors="pt").to("cuda")
        check_budget(tokens)
        context = (
            contextlib.nullcontext()
            if training and args.arm == "encoder-lora"
            else torch.no_grad()
        )
        with context:
            states, mask = kept(encode(tokens), tokens.attention_mask, counts)
        # Feature assembly runs outside the encoder's no-grad context so the
        # tag table trains even when the encoder rests (same values either way).
        docs, query = features(states, mask, case_tags(case))
        mask = torch.zeros((1, docs.shape[1]), dtype=torch.bool, device="cuda")
        memories = read(docs, mask, query)
        if memories.shape != (1, slots, 4096):
            raise ValueError("reader memory geometry changed")
        if raw:
            return memories[0]
        if detailed:
            output, *pointer_inputs = copy_detailed(memories, [case])
            return output[0], pointer_inputs
        return copy(memories, [case])[0]

    def batched_memories(cases, detailed=False):
        """Warmup-only: one no-grad encoder pass over every text of `cases`,
        then the reader over padded document features -> [B, slots, 4096]
        (with the copy head's pointer inputs when ``detailed``)."""
        docs, queries = [], []
        if warmup_cache:
            for case in cases:
                texts = case_texts(case)
                if dual_layer is None:
                    states, mask = cached_case_states(warmup_cache, texts)
                    states = states.to("cuda")
                else:
                    final, layer, mask = cached_case_states_dual(warmup_cache, texts)
                    states = (final.to("cuda"), layer.to("cuda"))
                d, q = features(states, mask.to("cuda"), case_tags(case))
                docs.append(d)
                queries.append(q)
        else:
            encoded = [encoder_texts(case) for case in cases]
            texts = [t for case_inputs, _ in encoded for t in case_inputs]
            tokens = tokenizer_e(texts, padding=True, return_tensors="pt").to("cuda")
            check_budget(tokens)
            # Adapters train during warmup only when explicitly requested; the
            # cached path is unavailable then because the features move.
            context = (
                contextlib.nullcontext()
                if args.lexical_warmup_train_encoder
                else torch.no_grad()
            )
            with context:
                states = encode(tokens)
            offset = 0
            for case, (case_inputs, counts) in zip(cases, encoded):
                count = len(case_inputs)
                case_states, case_mask = kept(
                    select(states, offset, offset + count),
                    tokens.attention_mask[offset : offset + count],
                    counts,
                )
                d, q = features(case_states, case_mask, case_tags(case))
                docs.append(d)
                queries.append(q)
                offset += count
        batch, mask = pad_document_batch(docs)
        memories = read(batch, mask, torch.cat(queries, dim=0))
        if memories.shape != (len(cases), slots, 4096):
            raise ValueError("reader memory geometry changed")
        if detailed:
            output, *pointer_inputs = copy_detailed(memories, cases)
            return output, pointer_inputs
        return copy(memories, cases)

    def prompt(memories, question, system_prompt):
        _, embeds, _ = prepare_prompt(
            tokenizer_d,
            decoder.get_input_embeddings(),
            memories,
            system_prompt=system_prompt,
            question=question,
            device="cuda",
        )
        return embeds

    def prefix(case, training, decoder_question=None, system_prompt=SYSTEM):
        return prompt(
            memory(case, training),
            case["question"] if decoder_question is None else decoder_question,
            system_prompt,
        )

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
                        "cohort": case.get("cohort", "development"),
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
            **({} if copy_head is None else {"copy_head": copy_head_state(copy_head)}),
            **(
                {}
                if tag_embedding is None
                else {"entity_tags": entity_tag_state(tag_embedding)}
            ),
            **(
                {}
                if slot_refiner is None
                else {"slot_refiner": slot_refiner_state(slot_refiner)}
            ),
            "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "result": result,
        }
        atomic_checkpoint(args.output / "resume.pt", state)

    warmup_cache, warmup_ids, warmup_tags = {}, {}, {}
    if args.lexical_warmup_cache is not None and start_step < args.lexical_warmup_steps:
        # Warmup never trains the encoder, so its token states are constant:
        # encode every distinct training text once (reused across runs when
        # encoder, adapters, data and feature contract match).
        unique_texts = sorted({t for case in train_rows for t in case_texts(case)})
        # Cache keys are the feature texts; under self-context a document is
        # encoded doubled and only its second copy's states are stored.
        document_texts = {t for case in train_rows for t in case_documents(case)}
        if self_context and document_texts & {case["question"] for case in train_rows}:
            raise ValueError("a self-context document text doubles as a question")
        cache_manifest = {
            "schema": SCHEMA,
            "encoder": proof["encoder"],
            "adapters_sha256": adapter_digest(adapters),
            "dataset_sha256": protocol["dataset_sha256"],
            "texts_sha256": hashlib.sha256(
                "\n".join(unique_texts).encode()
            ).hexdigest(),
            "dtype": "bfloat16",
            # Only present for a selected layer so existing caches stay valid.
            **({} if encoder_layer is None else {"encoder_layer": encoder_layer}),
            # Dual caches hold both layers per text and never mix with others.
            **(
                {}
                if dual_layer is None
                else {"document_features": "tokens-dual", "dual_layer": dual_layer}
            ),
            # Copy-head caches also store each text's decoder token ids.
            **({"copy_head": True} if copy_head is not None else {}),
            # Prefixed texts are cached under their prefixed keys and never
            # mix with an unprefixed cache of the same corpus.
            **({"document_space_prefix": True} if space_prefix else {}),
            # Self-context caches hold the second copy's states per document
            # (with the encoder's appended special rows).
            **(
                {"document_self_context": True, "encoder_special_rows": True}
                if self_context
                else {}
            ),
            # Entity-tag caches also store each document's per-token types.
            **({"entity_tags": args.entity_tags} if tag_embedding is not None else {}),
        }
        cache_dir = args.lexical_warmup_cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        manifest_path, tensor_path, ids_path, tags_path = (
            cache_dir / "manifest.json",
            cache_dir / "states.pt",
            cache_dir / "ids.pt",
            cache_dir / "tags.pt",
        )
        started_cache = time.monotonic()
        if (
            manifest_path.exists()
            and tensor_path.exists()
            and (copy_head is None or ids_path.exists())
            and (tag_embedding is None or tags_path.exists())
            and json.loads(manifest_path.read_text()) == cache_manifest
        ):
            warmup_cache = torch.load(
                tensor_path, map_location="cpu", weights_only=True
            )
            if copy_head is not None:
                warmup_ids = torch.load(ids_path, map_location="cpu", weights_only=True)
            if tag_embedding is not None:
                warmup_tags = torch.load(
                    tags_path, map_location="cpu", weights_only=True
                )
            cache_event = "loaded"
        else:
            modes(False)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                for begin in range(0, len(unique_texts), 64):
                    chunk = unique_texts[begin : begin + 64]
                    inputs, keeps = list(chunk), [slice(None)] * len(chunk)
                    if self_context:
                        inputs, keeps = [], []
                        for text in chunk:
                            if text in document_texts:
                                doubled, count, _, fallback = self_context_tokens(
                                    tokenizer_e, text
                                )
                                result["document_self_context_fallbacks"] += int(
                                    fallback
                                )
                                inputs.append(doubled)
                                keeps.append(slice(-(count + encoder_suffix), None))
                            else:
                                inputs.append(text)
                                keeps.append(slice(None))
                    tokens = tokenizer_e(inputs, padding=True, return_tensors="pt").to(
                        "cuda"
                    )
                    check_budget(tokens)
                    states = encode(tokens)
                    for index, (text, mask, keep) in enumerate(
                        zip(chunk, tokens.attention_mask, keeps)
                    ):
                        if dual_layer is None:
                            warmup_cache[text] = (
                                states[index][mask == 1][keep].to(torch.bfloat16).cpu()
                            )
                        else:
                            warmup_cache[text] = {
                                "final": states[0][index][mask == 1][keep]
                                .to(torch.bfloat16)
                                .cpu(),
                                "layer": states[1][index][mask == 1][keep]
                                .to(torch.bfloat16)
                                .cpu(),
                            }
            torch.save(warmup_cache, tensor_path)
            if copy_head is not None:
                warmup_ids = {
                    text: document_ids(text)
                    if text in document_texts
                    else document_token_ids(tokenizer_d, [text])
                    for text in unique_texts
                }
                torch.save(warmup_ids, ids_path)
            if tag_embedding is not None:
                # Tags depend on a case's entity list; a document text that
                # recurs across cases must tag identically to be cacheable.
                warmup_tags = {}
                for case in train_rows:
                    for text in case_documents(case):
                        tags = document_entity_tags(
                            tokenizer_e, text, case, self_context
                        )
                        if text in warmup_tags and not torch.equal(
                            warmup_tags[text], tags
                        ):
                            raise ValueError("a document text tags differently")
                        warmup_tags[text] = tags
                torch.save(warmup_tags, tags_path)
            save_json(manifest_path, cache_manifest)
            cache_event = "encoded"
        if (
            set(warmup_cache) != set(unique_texts)
            or (copy_head is not None and set(warmup_ids) != set(unique_texts))
            or (tag_embedding is not None and set(warmup_tags) != document_texts)
        ):
            raise ValueError("warmup feature cache does not cover the training texts")
        if tag_embedding is not None and any(
            (
                warmup_cache[text]["final"]
                if dual_layer is not None
                else warmup_cache[text]
            ).shape[0]
            != warmup_tags[text].numel() + encoder_suffix
            for text in document_texts
        ):
            # Every kept document state has exactly one entity type (the
            # encoder's appended special rows are untyped).
            raise ValueError("cached document states and entity tags differ")
        if copy_head is not None and any(
            (
                warmup_cache[text]["final"]
                if dual_layer is not None
                else warmup_cache[text]
            ).shape[0]
            != warmup_ids[text].numel() + encoder_suffix
            for text in document_texts
        ):
            # Every kept document state but the encoder's appended special
            # rows has exactly one copy-source id.
            raise ValueError("cached document states and copy-source ids differ")
        # The cached path must reproduce the live encoder features closely
        # (with the pooled context derived from the cached last rows).
        probe_case = train_rows[0]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            live = memory(probe_case, False).float()
            cached = batched_memories([probe_case])[0].float()
        cache_gap = float(
            (live - cached).abs().max() / live.abs().max().clamp_min(1e-6)
        )
        if not cache_gap < 0.1:
            raise ValueError("cached warmup features diverge from the live encoder")
        result["warmup_cache"] = {
            "event": cache_event,
            "texts": len(unique_texts),
            "seconds": time.monotonic() - started_cache,
            "relative_memory_gap": cache_gap,
        }
        print(
            json.dumps({"event": "warmup-cache", **result["warmup_cache"]}), flush=True
        )
    rescaled = False
    if (
        args.memory_scale_init == "lexical-target"
        and not args.resume
        and args.continue_from is None
    ):
        modes(False)
        sample = [
            train_rows[order_index(args.seed, i, len(train_rows))] for i in range(32)
        ]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            # The projection is rescaled on the reader's own output; the copy
            # head mixes in embedding rows that already have the target norm.
            memories = [memory(case, False, raw=True) for case in sample]
            target_norm = float(
                torch.stack(
                    [
                        lexical_targets(
                            tokenizer_d,
                            decoder.get_input_embeddings(),
                            lexical_record_text(case, args.lexical_record),
                            truncate=args.lexical_record in TRUNCATED_RECORDS,
                            slots=slots,
                        )
                        .norm(dim=-1)
                        .mean()
                        for case in sample
                    ]
                ).mean()
            )
        result["memory_scale_factor"] = rescale_memory_projection(
            reader, memories, target_norm
        )
        for group in optimizer.param_groups:
            if group.get("projection"):
                group["lr"] = args.reader_lr * result["memory_scale_factor"]
        result["memory_scale_target_norm"] = target_norm
        rescaled = True
        del memories
        print(
            json.dumps(
                {
                    "event": "memory-scale-init",
                    "factor": result["memory_scale_factor"],
                    "target_norm": target_norm,
                }
            ),
            flush=True,
        )
    try:
        if not args.resume:
            expected_rows = result["evaluations"][-1]["rows"] if start_step else None
            evaluate(start_step)
            if args.initialize_from is not None:
                previous = json.loads(
                    (args.initialize_from / "result.json").read_text()
                )["evaluations"][-1]["rows"]
                expected = {r["id"]: r["actual"] for r in previous}
                overlap = [
                    r for r in result["evaluations"][-1]["rows"] if r["id"] in expected
                ]
                matches = sum(r["actual"] == expected[r["id"]] for r in overlap)
                same_features = (
                    protocol["initialization"]["parent_document_features"]
                    == args.document_features
                )
                same_slots = protocol["initialization"]["parent_memory_slots"] == slots
                same_layer = (
                    protocol["initialization"].get("parent_encoder_layer")
                    == encoder_layer
                    and protocol["initialization"].get("parent_dual_layer")
                    == dual_layer
                )
                # A fresh copy head mixes 12% copied rows into the parent's
                # memories, so its answers are only expected to reproduce
                # when the parent trained the same head.
                same_copy_head = protocol["initialization"].get(
                    "parent_copy_head", False
                ) == (copy_head is not None)
                # A changed space prefix re-segments the encoder input, so
                # the parent's answers are not expected to reproduce either.
                same_space_prefix = (
                    protocol["initialization"].get(
                        "parent_document_space_prefix", False
                    )
                    == space_prefix
                )
                same_self_context = (
                    protocol["initialization"].get(
                        "parent_document_self_context", False
                    )
                    == self_context
                )
                same_pooled_context = (
                    protocol["initialization"].get(
                        "parent_document_pooled_context", False
                    )
                    == pooled_context
                )
                # A fresh zero tag table reproduces the parent's features
                # exactly, so only a changed copy restriction relaxes the check.
                same_copy_restrict = (
                    protocol["initialization"].get(
                        "parent_copy_restrict_by_type", False
                    )
                    == copy_restrict
                )
                # A zero-initialized refiner makes every added pass identical
                # to the first, so adding loops to a loop-less parent still
                # reproduces; more loops on a parent's trained refiner do not.
                parent_loops = protocol["initialization"].get("parent_reader_loops", 1)
                same_reader_loops = parent_loops == 1 or parent_loops == reader_loops
                if len(overlap) != len(previous) or (
                    same_features
                    and same_slots
                    and same_layer
                    and same_copy_head
                    and same_space_prefix
                    and same_self_context
                    and same_pooled_context
                    and same_copy_restrict
                    and same_reader_loops
                    and not rescaled
                    and matches != len(overlap)
                ):
                    raise ValueError(
                        "initialized weights do not reproduce parent development answers"
                    )
                # A changed document feature mode or encoder layer feeds the
                # same weights different inputs, and expanded queries add new
                # slots to the memory frame, so only the overlap is recorded.
                result["initialization_development_text_matches"] = matches
                result["initialization_same_document_features"] = same_features
                result["initialization_same_memory_slots"] = same_slots
                result["initialization_same_encoder_layer"] = same_layer
                if (
                    copy_head is not None
                    or "parent_copy_head" in protocol["initialization"]
                ):
                    result["initialization_same_copy_head"] = same_copy_head
                if "parent_document_space_prefix" in protocol["initialization"]:
                    result["initialization_same_document_space_prefix"] = (
                        same_space_prefix
                    )
                if "parent_document_self_context" in protocol["initialization"]:
                    result["initialization_same_document_self_context"] = (
                        same_self_context
                    )
                if "parent_document_pooled_context" in protocol["initialization"]:
                    result["initialization_same_document_pooled_context"] = (
                        same_pooled_context
                    )
                if "parent_copy_restrict_by_type" in protocol["initialization"]:
                    result["initialization_same_copy_restrict_by_type"] = (
                        same_copy_restrict
                    )
                if "parent_reader_loops" in protocol["initialization"]:
                    result["initialization_same_reader_loops"] = same_reader_loops
                save_json(result_path, result)
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
            copy_gates.clear()
            pointer_values.clear()
            gate_values.clear()
            copyable_fractions.clear()
            pointing = args.copy_pointer_weight > 0
            started, total_loss = time.monotonic(), 0.0
            qa_losses, reconstruction_losses, target_tokens = [], [], 0
            lexical_losses = []
            warming = step < args.lexical_warmup_steps
            if warming and args.lexical_warmup_schedule != "constant":
                scale = warmup_lr_scale(
                    step, args.lexical_warmup_steps, args.lexical_warmup_schedule
                )
                for group in optimizer.param_groups:
                    if group.get("projection"):
                        group["lr"] = (
                            args.reader_lr
                            * result.get("memory_scale_factor", 1.0)
                            * scale
                        )
                    elif group.get("copy_head"):
                        group["lr"] = copy_head_lr * scale
                    elif group.get("entity_tags"):
                        group["lr"] = entity_tag_lr * scale
                    elif group.get("slot_refiner"):
                        group["lr"] = slot_refiner_lr * scale
                    elif group.get("reader_body"):
                        group["lr"] = args.reader_lr * scale
            if (
                args.joint_reader_lr is not None
                and step == args.lexical_warmup_steps
                and start_step <= step
            ):
                for group in optimizer.param_groups:
                    if group.get("projection"):
                        group["lr"] = args.joint_reader_lr * result.get(
                            "memory_scale_factor", 1.0
                        )
                    elif group.get("copy_head"):
                        # An explicit copy-head rate stays through the joint phase.
                        group["lr"] = (
                            args.copy_head_lr
                            if args.copy_head_lr is not None
                            else args.joint_reader_lr
                        )
                    elif group.get("entity_tags"):
                        # An explicit tag rate stays through the joint phase.
                        group["lr"] = (
                            args.entity_tag_lr
                            if args.entity_tag_lr is not None
                            else args.joint_reader_lr
                        )
                    elif group.get("slot_refiner"):
                        # An explicit refiner rate stays through the joint phase.
                        group["lr"] = (
                            args.slot_refiner_lr
                            if args.slot_refiner_lr is not None
                            else args.joint_reader_lr
                        )
                    elif group.get("reader_body"):
                        group["lr"] = args.joint_reader_lr
                print(json.dumps({"event": "joint-phase", "step": step}), flush=True)
            for micro in range(args.accum):
                absolute_microstep = step * args.accum + micro
                index = (
                    group_sampler.index(absolute_microstep)
                    if group_sampler is not None
                    else order_index(args.seed, absolute_microstep, len(train_rows))
                )
                case = train_rows[index]
                question, answer, system, reconstructing = reconstruction_task(
                    case, step * args.accum + micro, args.reconstruct_every
                )
                if warming:
                    # Lexical-only warmup: the reader regresses onto record
                    # embeddings without a decoder pass; encoder adapters rest.
                    # A warmup batch spends consecutive sampler positions.
                    width = args.lexical_warmup_batch
                    batch_cases = [
                        train_rows[
                            group_sampler.index(absolute_microstep * width + j)
                            if group_sampler is not None
                            else order_index(
                                args.seed,
                                absolute_microstep * width + j,
                                len(train_rows),
                            )
                        ]
                        for j in range(width)
                    ]
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        if pointing:
                            memories, pointer_inputs = batched_memories(
                                batch_cases, detailed=True
                            )
                        else:
                            memories, pointer_inputs = (
                                batched_memories(batch_cases),
                                None,
                            )
                    lexical_loss = lexical_loss_for_batch(memories, batch_cases)
                    if not torch.isfinite(lexical_loss):
                        raise RuntimeError("nonfinite lexical loss")
                    lexical_losses.append(float(lexical_loss.detach()))
                    loss = args.lexical_loss_weight * lexical_loss
                    if pointing:
                        # Direct copy-head supervision: the lexical CE alone
                        # leaves the gate unpressured once the reader fits.
                        loss = loss + pointer_term(pointer_inputs, batch_cases)
                    total_loss += float(loss.detach()) / args.accum
                    (loss / args.accum).backward()
                    del memories, pointer_inputs, lexical_loss, loss
                    continue
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    if pointing:
                        memories, pointer_inputs = memory(case, True, detailed=True)
                    else:
                        memories, pointer_inputs = memory(case, True), None
                    embeddings = prompt(memories, question, system)
                    ids = tokenizer_d(answer, add_special_tokens=False).input_ids + [
                        151645
                    ]
                    target = torch.tensor(ids, device="cuda", dtype=torch.long)
                    if target.numel() > (64 if reconstructing else 24):
                        raise ValueError("answer exceeds explicit training target cap")
                    inputs = teacher_inputs(
                        embeddings, decoder.get_input_embeddings(), target
                    )
                    output = decoder(
                        inputs_embeds=inputs, use_cache=False, logits_to_keep=len(ids)
                    )
                    loss = F.cross_entropy(output.logits[0].float(), target)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                loss_value = float(loss.detach())
                (reconstruction_losses if reconstructing else qa_losses).append(
                    loss_value
                )
                if args.lexical_loss_weight > 0:
                    # Auxiliary FP32 pull toward the frozen decoder's embeddings
                    # of the canonical record; the decoder still read `memories`.
                    lexical_loss = lexical_loss_for(memories, case)
                    if not torch.isfinite(lexical_loss):
                        raise RuntimeError("nonfinite lexical loss")
                    lexical_losses.append(float(lexical_loss.detach()))
                    loss = loss + args.lexical_loss_weight * lexical_loss
                    if pointing:
                        loss = loss + pointer_term(pointer_inputs, [case])
                    loss_value = float(loss.detach())
                    del lexical_loss
                target_tokens += len(ids)
                total_loss += loss_value / args.accum
                (loss / args.accum).backward()
                del memories, pointer_inputs, embeddings, inputs, output, loss
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
                and not (warming and not args.lexical_warmup_train_encoder)
                and (not math.isfinite(lora_grad) or lora_grad == 0)
            ):
                raise RuntimeError("missing/nonfinite reader or encoder gradient")
            assert_frozen(frozen)
            norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
            )
            optimizer.step()
            if tag_embedding is not None and args.entity_tag_max_norm is not None:
                cap_entity_tag_rows(tag_embedding, args.entity_tag_max_norm)
            torch.cuda.synchronize()
            row = {
                "step": step + 1,
                "loss": total_loss,
                "qa_loss": sum(qa_losses) / len(qa_losses) if qa_losses else None,
                "reconstruction_loss": sum(reconstruction_losses)
                / len(reconstruction_losses)
                if reconstruction_losses
                else None,
                "lexical_loss": sum(lexical_losses) / len(lexical_losses)
                if lexical_losses
                else None,
                "qa_examples": len(qa_losses),
                "lexical_warmup": warming,
                "reconstruction_examples": len(reconstruction_losses),
                "target_tokens": target_tokens,
                "reader_grad_norm": reader_grad,
                "encoder_lora_grad_norm": lora_grad,
                "total_grad_norm": norm,
                "seconds": time.monotonic() - started,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            }
            if copy_head is not None:
                row["copy_gate_mean"] = (
                    sum(copy_gates) / len(copy_gates) if copy_gates else None
                )
            if pointing:
                row["pointer_loss"] = (
                    sum(pointer_values) / len(pointer_values)
                    if pointer_values
                    else None
                )
                row["gate_loss"] = (
                    sum(gate_values) / len(gate_values) if gate_values else None
                )
                row["copyable_fraction"] = (
                    sum(copyable_fractions) / len(copyable_fractions)
                    if copyable_fractions
                    else None
                )
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
        if copy_head is not None:
            save_file(
                {k: v.contiguous() for k, v in copy_head_state(copy_head).items()},
                str(args.output / "copy-head.safetensors"),
            )
        if tag_embedding is not None:
            save_file(
                {k: v.contiguous() for k, v in entity_tag_state(tag_embedding).items()},
                str(args.output / "entity-tags.safetensors"),
            )
        if slot_refiner is not None:
            save_file(
                {k: v.contiguous() for k, v in slot_refiner_state(slot_refiner).items()},
                str(args.output / "slot-refiner.safetensors"),
            )
        save_json(
            args.output / "candidate.json",
            {
                "schema": SCHEMA,
                "rollout_qualified": False,
                "experimental_only": True,
                "memory_slots": slots,
                **{
                    field: digest(args.output / file)
                    for field, file in candidate_artifacts(protocol)
                },
            },
        )
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        save_json(result_path, result)
        raise


if __name__ == "__main__":
    main()
