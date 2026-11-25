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

"""Strict parser for test-only Metal checkpoint owner-ledger evidence."""


from __future__ import annotations
import json
from pathlib import Path
import time
import math
import re

LEDGER_MARKER = "metal_checkpoint_qualification "
PHASE_SCHEMA = "antfly.metal-checkpoint-phase.v1"
RELEASE_SCHEMA = "antfly.metal-checkpoint-release.v1"
# The private test ingress pins Gemma-4 E4B. Zig derives and attests these from
# the loaded model layout; the collector pins the qualified cell independently.
# Independently pinned from the qualified GGUF metadata: every block executes
# attention, while only non-shared-KV blocks append donor state. E2B declares
# block_count=35/shared_kv_layers=20; E4B declares 42/18.
QUALIFIED_MODEL_GEOMETRY = {
    "google/gemma-4-E2B-it-qat-q4_0-gguf": (35, 15),
    "google/gemma-4-E4B-it-qat-q4_0-gguf": (42, 24),
}
KV_BOUNDARY_FIELDS = (
    "kv_materialized_tokens", "pending_sampled_tokens", "kv_boundary_proven",
    "binding_attention_calls_per_materialized_token",
    "binding_donor_appends_per_materialized_token",
)


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def ledger_rows(path, offset):
    rows = []
    with Path(path).open("rb") as stream:
        stream.seek(offset)
        for raw in stream:
            text = raw.decode("utf-8", errors="strict")
            marker = text.find(LEDGER_MARKER)
            if marker < 0:
                continue
            value = json.loads(text[marker + len(LEDGER_MARKER):])
            require(type(value) is dict, "invalid Metal owner ledger row")
            rows.append(value)
    return rows


def wait_ledger(path, offset, phases=("generation_committed", "final_release"),
                timeout=5):
    deadline = time.monotonic() + timeout
    rows = []
    while time.monotonic() < deadline:
        rows = ledger_rows(path, offset)
        if all(any(row.get("phase") == phase for row in rows) for phase in phases):
            return rows
        time.sleep(.01)
    return rows


def _binding_boundary_contract(ready, current, expected_model):
    """Prove a sampled final token that has not undergone a subsequent forward."""
    integer_fields = (
        "kv_materialized_tokens", "pending_sampled_tokens",
        "binding_attention_calls_per_materialized_token",
        "binding_donor_appends_per_materialized_token",
    )
    require(
        all(type(row.get(field)) is int and row[field] >= 0
            for row in (ready, current) for field in integer_fields) and
        all(type(row.get(field)) is int and row[field] >= 0
            for row in (ready, current)
            for field in ("logical_tokens", "binding_logical_tokens",
                          "attention_calls", "donor_appends")) and
        ready.get("kv_boundary_proven") is True and
        current.get("kv_boundary_proven") is True,
        "checkpoint restore lacks typed KV-boundary proof",
    )
    require(
        ready["pending_sampled_tokens"] == 0 and
        ready["kv_materialized_tokens"] == ready.get("logical_tokens") and
        ready.get("binding_logical_tokens") == ready["kv_materialized_tokens"] and
        current["pending_sampled_tokens"] in (0, 1) and
        current.get("logical_tokens") ==
            current["kv_materialized_tokens"] + current["pending_sampled_tokens"] and
        current.get("binding_logical_tokens") == current["kv_materialized_tokens"],
        "checkpoint restore KV boundary does not reconcile",
    )
    materialized_delta = (
        current["kv_materialized_tokens"] - ready["kv_materialized_tokens"]
    )
    attention_delta = current.get("attention_calls", -1) - ready.get(
        "attention_calls", -1
    )
    append_delta = current.get("donor_appends", -1) - ready.get(
        "donor_appends", -1
    )
    attention_rate = current["binding_attention_calls_per_materialized_token"]
    append_rate = current["binding_donor_appends_per_materialized_token"]
    require(expected_model in QUALIFIED_MODEL_GEOMETRY,
            "checkpoint restore model geometry is not qualified")
    expected_attention_rate, expected_append_rate = QUALIFIED_MODEL_GEOMETRY[expected_model]
    require(
        ready["binding_attention_calls_per_materialized_token"] ==
            attention_rate == expected_attention_rate and
        ready["binding_donor_appends_per_materialized_token"] ==
            append_rate == expected_append_rate,
        "checkpoint restore KV rates differ from qualified model geometry",
    )
    require(
        materialized_delta > 0 and
        attention_delta == materialized_delta * attention_rate and
        append_delta == materialized_delta * append_rate,
        "checkpoint restore KV activity does not prove its materialized boundary",
    )


def generation_contract(rows, expected_route, require_binding=True,
                        expected_representation=None, expected_model=None,
                        expected_encoded_heads=None, expected_encoded_rows=None):
    committed = [row for row in rows if row.get("schema") == PHASE_SCHEMA and
                 row.get("phase") == "generation_committed"]
    released = [row for row in rows if row.get("schema") == RELEASE_SCHEMA and
                row.get("phase") == "final_release"]
    require(len(committed) == len(released) == 1,
            "request lacks unique committed/release owner ledger")
    current, final = committed[0], released[0]
    identity = (current.get("request"), current.get("route"), current.get("owner_id"))
    require(identity == (final.get("request"), final.get("route"),
                         final.get("owner_id")), "owner ledger attribution changed")
    require(current.get("route") == expected_route,
            "owner ledger route differs from requested workload")
    require(all(current.get(field) is True for field in
                ("owner_available", "physical_available", "combined_available")) and
            current.get("invalid") is False and
            type(current.get("logical_tokens")) is int and
            all(type(current.get(field)) is int and current[field] >= 0 for field in
                ("validation_ns", "upload_ns", "install_ns", "suffix_prefill_ns",
                 "attention_calls", "donor_appends")) and
            type(current.get("tracked_combined_live_bytes")) is int and
            current["tracked_combined_live_bytes"] > 0 and
            type(current.get("tracked_combined_peak_bytes")) is int and
            current["tracked_combined_peak_bytes"] >= current["tracked_combined_live_bytes"] and
            final.get("released") is True and final.get("scope_complete") is True and
            final.get("full_owner_physical_available") is True and
            final.get("measured_available") is True and
            type(final.get("initial_live_bytes")) is int and
            final["initial_live_bytes"] >= 0 and
            type(final.get("measured_committed_bytes")) is int and
            final["measured_committed_bytes"] > 0 and
            final.get("post_release_live_bytes") == 0,
            "incomplete physical owner ledger")
    require(current["tracked_combined_live_bytes"] >= final["initial_live_bytes"] and
            current["tracked_combined_live_bytes"] - final["initial_live_bytes"] ==
            final["measured_committed_bytes"],
            "phase and release committed-byte measurements do not reconcile")
    if require_binding:
        ready = [row for row in rows if row.get("schema") == PHASE_SCHEMA and
                 row.get("phase") == "resumed_ready"]
        require(len(ready) == 1 and
                (ready[0].get("request"), ready[0].get("route"),
                 ready[0].get("owner_id")) == identity,
                "checkpoint restore lacks unique resumed-ready ledger")
        require(current.get("binding") is True,
                "checkpoint restore binding is incomplete")
        proof_fields_present = any(
            field in row for row in (ready[0], current) for field in KV_BOUNDARY_FIELDS
        )
        if proof_fields_present:
            _binding_boundary_contract(ready[0], current, expected_model)
        else:
            require(current.get("binding_logical_tokens") == current["logical_tokens"],
                    "checkpoint restore binding is incomplete")
        require(expected_representation in ("f16", "fp8", "rotated4") and
                current.get("binding_representation") == expected_representation,
                "checkpoint resident representation differs")
        for field in ("binding_fp8_heads", "binding_fp8_rows",
                      "binding_rotated4_heads", "binding_rotated4_rows",
                      "binding_exact_resident_bytes", "binding_encoded_resident_bytes",
                      "binding_metadata_resident_bytes", "binding_allocated_bytes"):
            require(type(current.get(field)) is int and current[field] >= 0,
                    "invalid checkpoint resident accounting: " + field)
        require(current["binding_exact_resident_bytes"] +
                current["binding_encoded_resident_bytes"] +
                current["binding_metadata_resident_bytes"] ==
                current["binding_allocated_bytes"],
                "checkpoint resident byte accounting does not reconcile")
        require(current["binding_allocated_bytes"] > 0 and
                current["attention_calls"] > 0,
                "checkpoint restore lacks resident execution proof")
        if expected_representation == "fp8":
            require(current["binding_fp8_heads"] > 0 and
                    current["binding_fp8_rows"] > 0 and
                    current["binding_rotated4_heads"] == 0 and
                    current["binding_rotated4_rows"] == 0 and
                    current["binding_encoded_resident_bytes"] > 0,
                    "FP8 restore lacks resident encoded storage proof")
        elif expected_representation == "rotated4":
            require(current["binding_rotated4_heads"] > 0 and
                    current["binding_rotated4_rows"] > 0 and
                    current["binding_fp8_heads"] == 0 and
                    current["binding_fp8_rows"] == 0 and
                    current["binding_encoded_resident_bytes"] > 0,
                    "rotated4 restore lacks resident encoded storage proof")
            require(expected_encoded_heads is None or
                    current["binding_rotated4_heads"] == expected_encoded_heads,
                    "rotated4 resident head count differs from checkpoint receipt")
            require(expected_encoded_rows is None or
                    current["binding_rotated4_rows"] == expected_encoded_rows,
                    "rotated4 resident row count differs from checkpoint receipt")
        else:
            require(current["binding_exact_resident_bytes"] > 0 and
                    current["binding_fp8_heads"] == 0 and
                    current["binding_fp8_rows"] == 0 and
                    current["binding_rotated4_heads"] == 0 and
                    current["binding_rotated4_rows"] == 0 and
                    current["binding_encoded_resident_bytes"] == 0,
                    "f16 restore unexpectedly claims encoded storage")
    else:
        require(current.get("binding") is False and
                current.get("binding_representation") == "none" and
                all(current.get(field) == 0 for field in (
                    "binding_logical_tokens", "binding_allocated_bytes",
                    "binding_fp8_heads", "binding_fp8_rows",
                    "binding_rotated4_heads", "binding_rotated4_rows",
                    "binding_exact_resident_bytes",
                    "binding_encoded_resident_bytes",
                    "binding_metadata_resident_bytes",
                )), "exact Metal request unexpectedly installed checkpoint state")
    return {"committed": current, "release": final,
            "committed_context_bytes": final["measured_committed_bytes"],
            "tracked_combined_live_bytes": current["tracked_combined_live_bytes"],
            "release_measured_committed_bytes": final["measured_committed_bytes"],
            "initial_baseline_bytes": final["initial_live_bytes"],
            "measurement_reconciliation_delta_bytes":
                current["tracked_combined_live_bytes"] - final["initial_live_bytes"] -
                final["measured_committed_bytes"]}

# Successive-owner cycle evidence.


def cycle_require(condition, message):
    if not condition:
        raise ValueError(message)

def integer(value, label, minimum=0):
    cycle_require(type(value) is int and value >= minimum, "invalid " + label)
    return value

def number(value, label, minimum=0):
    cycle_require(type(value) in (int, float) and math.isfinite(value) and value >= minimum,
            "invalid " + label)
    return value

def digest(value, label):
    cycle_require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) and
            value != "0" * 64, "invalid " + label)
    return value

def byte_digest(value, label):
    cycle_require(type(value) is list and len(value) == 32 and
            all(type(x) is int and 0 <= x <= 255 for x in value) and any(value),
            "invalid " + label)
    return bytes(value).hex()

def validate_report(value, frozen_input, *, input_sha256, model_directory, loaded_identity):
    """All numerical bounds are the frozen runtime contract, never caller overrides."""
    cycle_require(type(value) is dict and value.get("schema") ==
            "antfly.metal-checkpoint-cycles.report.v3", "unsupported cycle report")
    request = frozen_input["request"]
    count = integer(request["cycles"], "requested cycles", 1)
    cycle_require(count <= 20, "too many cycles")
    append = integer(request["append_tokens"], "append tokens", 1)
    answers = integer(request["answer_tokens"], "answer tokens", 1)
    cycle_require(append <= 2048 and answers <= 32, "invalid continuation request")
    initial = integer(frozen_input["expected_prompt_tokens"], "initial tokens", 2)
    cycle_require(initial + count * (append + answers) <= 32768 - 512,
            "logical capacity exceeds query-tail reserve")
    encoding = request["encoding"]
    cycle_require(encoding in ("f16", "fp8", "rotated4"), "invalid encoding")
    cycle_require(value.get("model") == frozen_input["model"] and
            value.get("encoding") == encoding and
            value.get("model_directory") == str(model_directory) and
            value.get("input_sha256") == digest(input_sha256, "input hash"),
            "cycle input/model binding mismatch")
    identity = value.get("model_identity")
    cycle_require(type(identity) is dict and set(identity) == {"weights", "tokenizer", "template", "layout"},
            "invalid loaded identity fields")
    cycle_require({key: byte_digest(raw, key) for key, raw in identity.items()} == loaded_identity,
            "loaded model identity mismatch")
    integer(value.get("load_generation"), "load generation", 1)
    for key in ("rendered_prompt_sha256", "prompt_tokens_sha256", "implementation_sha256"):
        digest(value.get(key), key)
    report = value.get("report")
    cycle_require(type(report) is dict and report.get("passed") is True and
            report.get("diagnostic_only") is True and report.get("production_ready") is False,
            "invalid cycle evidence authority")
    reference = request.get("reference", [])
    cycle_require(type(reference) is list and all(type(token) is int and token >= 0 for token in reference) and
            (not reference or len(reference) == count * answers),
            "invalid frozen reference")
    cycle_require(report.get("frozen_reference") is bool(reference), "reference provenance mismatch")
    cycle_require(type(report.get("completed")) is int and report["completed"] == count and
            report.get("initial_tokens") == initial and
            report.get("final_tokens") == initial + (count - 1) * append + count * answers,
            "incomplete cycle trajectory")
    rows = report.get("rows")
    cycle_require(type(rows) is list and len(rows) == 20, "invalid bounded row array")
    active = rows[:count]
    owner = integer(active[0].get("owner_id"), "installed owner", 1)
    previous_revision = 0
    exact_nll = compacted_nll = 0.0
    greedy = comparisons = 0
    checkpoint_hashes = []
    for index, row in enumerate(active):
        cycle_require(type(row) is dict and type(row.get("cycle")) is int and row["cycle"] == index + 1 and
                row.get("owner_id") == owner and type(row.get("owner_id")) is int and
                row.get("logical_tokens") == initial + index * append + (index + 1) * answers,
                "invalid persistent owner trajectory")
        source_revision = integer(row.get("source_revision"), "source revision", 1)
        cycle_require(source_revision > previous_revision and
                type(row.get("installed_revision")) is int and
                row["installed_revision"] == source_revision + 1, "invalid install revision")
        previous_revision = row["installed_revision"]
        cycle_require(row.get("query_scope_released") is True and
                type(row.get("query_scope_delta_bytes")) is int and row["query_scope_delta_bytes"] == 0 and
                row.get("query_uses_source_owner") is True and
                type(row.get("query_owner_id")) is int and row["query_owner_id"] == owner,
                "query clone escaped parent allocation scope")
        peak = integer(row.get("query_owner_peak_bytes"), "query peak", 1)
        # generation.collectAttentionMatchQueries reports the fresh forwarding
        # scope's peak, not the parent owner's lifetime high-water mark. A new
        # scope can shrink after compaction; its cap and full release still bind.
        cycle_require(peak <= 8 * 1024**3, "query scope peak exceeded")
        for key in ("completed_chunks", "validated_heads", "attention_calls", "donor_appends", "fp64_accelerated_value_solves"):
            integer(row.get(key), key, 1)
        cycle_require(integer(row.get("admitted_peak_bytes"), "admitted bytes", 1) <= 8 * 1024**3,
                "aggregate admission exceeded")
        cycle_require(number(row.get("max_head_mass_error"), "head mass error") <= .05 and
                number(row.get("max_head_output_rmse"), "head output error") <= .10,
                "head numerical acceptance exceeded")
        if index:
            integer(row.get("ragged_repeat_calls"), "installed query attention", 1)
        source_bytes = integer(row.get("source_global_bytes"), "source global bytes", 1)
        candidate_bytes = integer(row.get("candidate_global_bytes"), "candidate global bytes", 1)
        threshold = request["initial_minimum_savings_ratio"] if index == 0 else request["incremental_minimum_savings_ratio"]
        cycle_require(number(threshold, "savings threshold", 1) >= (1.5 if index == 0 else 1) and
                candidate_bytes < source_bytes and source_bytes / candidate_bytes + 1e-12 >= threshold,
                "cycle did not meet frozen global savings")
        fp8 = integer(row.get("fp8_heads"), "FP8 heads")
        rot4 = integer(row.get("rotated4_heads"), "rotated4 heads")
        cycle_require((encoding == "f16" and fp8 == rot4 == 0) or
                (encoding == "fp8" and fp8 > 0 and rot4 == 0) or
                (encoding == "rotated4" and rot4 > 0 and fp8 == 0),
                "requested resident codec was not exercised")
        byte_digest(row.get("immutable_rows_sha256"), "historical rows hash")
        checkpoint_hashes.append(byte_digest(row.get("checkpoint_sha256"), "checkpoint hash"))
        compared = integer(row.get("comparisons"), "comparisons", 1)
        matched = integer(row.get("greedy_matches"), "greedy matches")
        cycle_require(compared == answers and matched <= compared, "invalid continuation comparison count")
        exact_nll += number(row.get("exact_nll"), "exact NLL")
        compacted_nll += number(row.get("compacted_nll"), "compacted NLL")
        number(row.get("max_logit_relative_l2"), "logit drift")
        comparisons += compared
        greedy += matched
    cycle_require(len(set(checkpoint_hashes)) == count, "checkpoint trajectory replayed")
    delta = (compacted_nll - exact_nll) / comparisons
    cycle_require(delta <= math.log(1.05), "continuation NLL acceptance exceeded")
    for label in ("exact_release", "compacted_release"):
        receipt = report.get(label)
        cycle_require(type(receipt) is dict and receipt.get("released") is True and
                type(receipt.get("post_release_live_bytes")) is int and receipt["post_release_live_bytes"] == 0,
                "cycle owner did not release")
    return {"completed_cycles": count, "owner_id": owner, "comparisons": comparisons,
            "greedy_matches": greedy, "nll_delta_per_token": delta,
            "frozen_reference": bool(reference), "checkpoint_sha256": checkpoint_hashes,
            "diagnostic_only": True, "production_ready": False}
