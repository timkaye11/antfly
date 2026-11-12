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

"""Export immutable research candidates from selected or averaged checkpoints.

A parameter average needs its first development reference before initialization
or qualification; it is not presented as a measured trained checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import torch
from safetensors.torch import save_file

from encoder_lora import (
    SCHEMA,
    candidate_artifacts,
    digest,
    entity_tags_enabled,
    memory_slot_count,
    reader_loops_enabled,
    save_json,
)


def derive_candidate(
    run, snapshots, output, weights=None, allow_incomplete_parent=False
):
    protocol = json.loads((run / "protocol.json").read_text())
    parent = json.loads((run / "result.json").read_text())
    incomplete = not parent["completed"]
    if incomplete and not allow_incomplete_parent:
        raise ValueError("parent must finish with unchanged frozen base weights")
    if parent.get("error") or (
        not incomplete and not parent.get("frozen_base_versions_unchanged")
    ):
        raise ValueError("parent must finish with unchanged frozen base weights")
    if incomplete and len(snapshots) != 1:
        raise ValueError("an interrupted parent yields single selected checkpoints")
    slots = memory_slot_count(protocol["memory_slots"])
    if digest(run / "harness_source.py") != protocol["script_sha256"]:
        raise ValueError("parent source or memory contract differs")
    if not 1 <= len(snapshots) <= 4:
        raise ValueError("one to four checkpoints required")
    weights = [1 / len(snapshots)] * len(snapshots) if weights is None else weights
    if (
        len(weights) != len(snapshots)
        or any(not 0 < w <= 1 for w in weights)
        or abs(sum(weights) - 1) > 1e-6
    ):
        raise ValueError("positive checkpoint weights must sum to one")
    torch.set_num_threads(4)
    states = [
        torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        for p in snapshots
    ]
    for state in states:
        if state["protocol"] != protocol or not 1 <= state["step"] <= protocol["steps"]:
            raise ValueError("snapshot belongs to another run or budget")
        queries = state["reader"].get("memory_queries")
        if queries is not None and tuple(queries.shape[:2]) != (1, slots):
            raise ValueError("snapshot reader slots differ from the protocol")
        if (
            len(state["result"]["history"]) != state["step"]
            or state["result"]["evaluations"][-1]["step"] != state["step"]
        ):
            raise ValueError("snapshot lacks a matching development evaluation")
    pending = len(states) > 1
    latest = max(states, key=lambda s: s["step"])
    derived = copy.deepcopy(protocol)
    derived["steps"] = latest["step"]
    derived["postprocessing"] = {
        "method": "parameter_average" if pending else "selected_checkpoint",
        "source_run": str(run.resolve()),
        "parent_planned_steps": protocol["steps"],
        "parent_protocol_sha256": digest(run / "protocol.json"),
        "checkpoint_steps": [s["step"] for s in states],
        "checkpoint_sha256": [digest(p) for p in snapshots],
        "weights": weights,
        "exporter_sha256": digest(Path(__file__)),
        "optimizer_reset_required": True,
    }
    output.mkdir(parents=True, exist_ok=False)
    tensor_files = [
        ("reader", "reader.safetensors"),
        ("adapters", "encoder-lora.safetensors"),
    ]
    if protocol.get("copy_head", False):
        # A copy-head run's snapshots carry the gated copy module as well.
        if any("copy_head" not in s for s in states):
            raise ValueError("copy-head snapshot lacks the copy head state")
        tensor_files.append(("copy_head", "copy-head.safetensors"))
    elif any("copy_head" in s for s in states):
        raise ValueError("snapshot carries a copy head its protocol does not declare")
    if entity_tags_enabled(protocol):
        # An entity-tag run's snapshots carry the tag embedding table as well.
        if any("entity_tags" not in s for s in states):
            raise ValueError("entity-tag snapshot lacks the tag embedding state")
        tensor_files.append(("entity_tags", "entity-tags.safetensors"))
    elif any("entity_tags" in s for s in states):
        raise ValueError("snapshot carries entity tags its protocol does not declare")
    if reader_loops_enabled(protocol):
        # A looped run's snapshots carry the slot refiner as well.
        if any("slot_refiner" not in s for s in states):
            raise ValueError("looped snapshot lacks the slot refiner state")
        tensor_files.append(("slot_refiner", "slot-refiner.safetensors"))
    elif any("slot_refiner" in s for s in states):
        raise ValueError("snapshot carries a slot refiner its protocol does not declare")
    for field, file in tensor_files:
        keys = set(states[0][field])
        if any(set(s[field]) != keys for s in states):
            raise ValueError("snapshot tensor inventories differ")
        tensors = {}
        for key in sorted(keys):
            source = states[0][field][key]
            if not source.is_floating_point():
                raise ValueError("nonfloating tensor in research candidate")
            value = torch.zeros_like(source, dtype=torch.float32)
            for state, weight in zip(states, weights):
                tensor = state[field][key]
                if (
                    tensor.shape != source.shape
                    or tensor.dtype != source.dtype
                    or not torch.isfinite(tensor).all()
                ):
                    raise ValueError("invalid or incompatible snapshot tensor")
                value.add_(tensor.float(), alpha=weight)
            tensors[key] = value.to(source.dtype).contiguous()
        save_file(tensors, str(output / file))
    for file in ["base-proofs.json", "harness_source.py"]:
        shutil.copy2(run / file, output / file)
    result = copy.deepcopy(latest["result"])
    result.update(
        completed=not pending,
        reference_pending=pending,
        selected_checkpoint=True,
        frozen_base_versions_unchanged=True,
    )
    if pending:
        result["source_training_evaluations"] = result.pop("evaluations")
        result["evaluations"] = []
    save_json(output / "protocol.json", derived)
    save_json(output / "result.json", result)
    candidate = {
        "schema": SCHEMA,
        "rollout_qualified": False,
        "experimental_only": True,
        "memory_slots": slots,
        "candidate_kind": "parameter_average" if pending else "selected_checkpoint",
        "parent_completed": not incomplete,
        "reference_pending": pending,
        "resume_supported": False,
        **{
            field: digest(output / file) for field, file in candidate_artifacts(derived)
        },
    }
    save_json(output / "candidate.json", candidate)
    return candidate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, action="append", required=True)
    p.add_argument("--weight", type=float, action="append")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--allow-incomplete-parent", action="store_true")
    args = p.parse_args()
    print(
        derive_candidate(
            args.run,
            args.snapshot,
            args.output,
            args.weight,
            allow_incomplete_parent=args.allow_incomplete_parent,
        )
    )


if __name__ == "__main__":
    main()
