#!/usr/bin/env python3
"""Audit one full-ingestion compaction performance cell.

This deliberately has a schema separate from the checkpoint-restore benchmark.
The input must be the raw campaign emitted by Source822's monotonic collector.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
import math
import random
from pathlib import Path
_child_spec=importlib.util.spec_from_file_location("inclusive_child_artifacts",Path(__file__).with_name("child_artifacts.py"))
if _child_spec is None or _child_spec.loader is None: raise RuntimeError("child artifact verifier unavailable")
child_artifacts=importlib.util.module_from_spec(_child_spec); _child_spec.loader.exec_module(child_artifacts)

INPUT_SCHEMA = "antfly.attention-matching-full-ttft-campaign.v1"
OUTPUT_SCHEMA = "antfly.compaction-full-ingestion-performance.v1"
ATTEMPT_SCHEMA = "antfly.attention-matching-full-ttft-attempt.v1"
VARIANTS = ("exact", "packed_exact", "attention_matching")
ORDERS = set(itertools.permutations(VARIANTS))
REQUIRED_PHASES = {
    "exact": {"request_dispatch", "exact_prefill", "first_token"},
    "packed_exact": {"request_dispatch", "packed_exact_prefill", "first_token"},
    "attention_matching": {"request_dispatch", "source_prefill", "query_collection",
                           "fit_validation", "export", "upload", "install", "first_token"},
}
ALLOWED_EXTRA_PHASES = {"future_generation", "snapshot", "suffix_prefill", "compiled_handoff",
                        "export_prepare", "client_checkpoint_receive"}

def need(value, message):
    if not value:
        raise ValueError(message)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def finite(value, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)

def percentile(values, fraction):
    values = sorted(values)
    need(values, "empty statistic")
    index = (len(values) - 1) * fraction
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return values[low]
    return values[low] + (values[high] - values[low]) * (index - low)

def bootstrap_bound(rows, statistic, upper, seed):
    rng = random.Random(seed)
    estimates = []
    for _ in range(10_000):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        estimates.append(statistic(sample))
    return percentile(estimates, .95 if upper else .05)

def validate_attempt(row, variant, seen_owners, seen_artifacts):
    need(type(row) is dict and row.get("completed") is True, "incomplete attempt")
    for field in ("ttft_ms", "decode_tokens_per_second", "committed_context_bytes"):
        need(finite(row.get(field), positive=True), f"invalid {variant} {field}")
    need(row.get("swapout_growth_bytes") == 0, f"{variant} paged")
    authority = row.get("full_ttft_authority")
    need(type(authority) is dict and authority.get("schema") == ATTEMPT_SCHEMA,
         "restore-only observation rejected")
    need(authority.get("clock") == "posix_CLOCK_MONOTONIC" and
         authority.get("timing_scope") == "full_request_pipeline_to_first_public_content" and
         authority.get("first_public_content_kind") == "nonempty_public_delta",
         "invalid inclusive timing authority")
    start, finish = authority.get("outer_start_ns"), authority.get("first_public_content_ns")
    need(type(start) is int and type(finish) is int and 0 <= start < finish,
         "invalid inclusive interval")
    phases = authority.get("phases")
    need(type(phases) is list and phases, "missing phase ledger")
    names, cursor = [], start
    allowed = REQUIRED_PHASES[variant] | ALLOWED_EXTRA_PHASES
    for phase in phases:
        need(type(phase) is dict and set(phase) == {"name", "start_ns", "end_ns", "authority"},
             "phase closure")
        need(phase["name"] in allowed and phase["start_ns"] == cursor and
             type(phase["end_ns"]) is int and cursor < phase["end_ns"] <= finish and
             phase["authority"] in ("client_monotonic_boundary", "server_monotonic_event"),
             "noncontiguous phase ledger")
        names.append(phase["name"])
        cursor = phase["end_ns"]
    need(cursor == finish and len(names) == len(set(names)) and REQUIRED_PHASES[variant] <= set(names),
         "incomplete inclusive phase ledger")
    need(abs(row["ttft_ms"] - (finish - start) / 1e6) <= 1e-6, "TTFT clock mismatch")
    owner, artifact = authority.get("fresh_owner_id"), authority.get("fresh_artifact_id")
    need(isinstance(owner, str) and owner and owner not in seen_owners, "owner reused")
    need(isinstance(artifact, str) and artifact and artifact not in seen_artifacts, "artifact reused")
    seen_owners.add(owner); seen_artifacts.add(artifact)
    restore = authority.get("restore_ttft_ms")
    if variant == "attention_matching":
        need(finite(restore, positive=True) and restore <= row["ttft_ms"], "secondary restore timing")
        need(row.get("published") is True and row.get("exact_fallback") is False,
             "compaction was not published")
    else:
        need(restore is None, "exact arm used restore timing")

def audit(campaign, campaign_ref, execution_authority):
    need(type(execution_authority) is dict and
         execution_authority.get("schema") == "antfly.current-full-ingestion-server-launch.v1" and
         execution_authority.get("passed") is True and
         execution_authority.get("qualification_claim") is False and
         execution_authority.get("performance_scope") == "full_request_ingestion",
         "execution authority")
    for field in ("source_freeze", "full_cli_receipt", "binary", "matrix", "profile",
                  "model_config", "request_config", "native_identity", "model_artifact"):
        value = execution_authority.get(field)
        need(type(value) is dict and set(value) == {"path", "sha256"} and
             isinstance(value["path"], str) and isinstance(value["sha256"], str) and
             len(value["sha256"]) == 64, f"missing {field} authority")
    children = execution_authority.get("matrix_children")
    need(type(children) is dict and set(children) == {"candidate_binding_path",
         "optimal_compiled_exact_binding_path", "packed_exact_binding_path"} and
         all(type(value) is dict and set(value) == {"path", "sha256"} for value in children.values()),
         "matrix child authority")
    header = execution_authority.get("performance_header")
    need(type(header) is dict and all(campaign.get(key) == value for key, value in header.items()),
         "campaign provenance/header join")
    need(campaign.get("schema") == INPUT_SCHEMA and campaign.get("passed") is True,
         "raw campaign failed")
    child_artifacts.from_campaign(campaign)
    warmups, samples = campaign.get("warmups"), campaign.get("samples")
    need(type(warmups) is list and len(warmups) == 5, "requires exactly five warmup triplets")
    need(type(samples) is list and len(samples) == 30, "requires exactly thirty measured triplets")
    need(campaign.get("all_six_orders_present") is True, "campaign omitted an order")
    order_counts = {order: 0 for order in ORDERS}
    seen_pairs, seen_prompts, seen_owners, seen_artifacts = set(), set(), set(), set()
    for measured, rows in ((False, warmups), (True, samples)):
        for row in rows:
            need(type(row) is dict and set(row.get("order", ())) == set(VARIANTS), "triplet order")
            if measured:
                order_counts[tuple(row["order"])] += 1
            pair, prompt = row.get("pair_id"), row.get("prompt_sha256")
            need(isinstance(pair, str) and pair and pair not in seen_pairs, "pair reused")
            need(isinstance(prompt, str) and len(prompt) == 64 and prompt not in seen_prompts,
                 "prompt reused")
            seen_pairs.add(pair); seen_prompts.add(prompt)
            context, reserved = row.get("context_tokens"), row.get("reserved_output_tokens")
            expected_prompt = 8128 if context == 8192 else 32512 if context == 32768 else None
            expected_reserved = 64 if context == 8192 else 256 if context == 32768 else None
            need(row.get("prompt_tokens") == expected_prompt and reserved == expected_reserved and
                 row.get("output_tokens") == expected_reserved, "wrong formal token geometry")
            for variant in VARIANTS:
                validate_attempt(row.get(variant), variant, seen_owners, seen_artifacts)
            exact, packed, match = row["exact"], row["packed_exact"], row["attention_matching"]
            need(exact.get("output_sha256") == packed.get("output_sha256"),
                 "exact baselines disagree")
            baseline_bytes = min(exact["committed_context_bytes"], packed["committed_context_bytes"])
            need(match["committed_context_bytes"] <= .75 * baseline_bytes,
                 "per-sample committed-context savings below 25 percent")
    need(set(order_counts) == ORDERS and set(order_counts.values()) == {5},
         "thirty measured triplets must use every order five times")

    packed_ttft = lambda rows: percentile([x["packed_exact"]["ttft_ms"] / x["exact"]["ttft_ms"] for x in rows], .95)
    packed_decode = lambda rows: percentile([x["packed_exact"]["decode_tokens_per_second"] / x["exact"]["decode_tokens_per_second"] for x in rows], .50)
    match_ttft = lambda rows: percentile([x["attention_matching"]["ttft_ms"] / min(x["exact"]["ttft_ms"], x["packed_exact"]["ttft_ms"]) for x in rows], .95)
    match_decode = lambda rows: percentile([x["attention_matching"]["decode_tokens_per_second"] / x["exact"]["decode_tokens_per_second"] for x in rows], .50)
    savings = lambda rows: percentile([1 - x["attention_matching"]["committed_context_bytes"] / min(x["exact"]["committed_context_bytes"], x["packed_exact"]["committed_context_bytes"]) for x in rows], .50)
    seed = campaign_ref["sha256"]
    bounds = {
        "packed_exact_ttft_ratio_upper_95": bootstrap_bound(samples, packed_ttft, True, seed + ":packed-ttft"),
        "packed_exact_decode_ratio_lower_95": bootstrap_bound(samples, packed_decode, False, seed + ":packed-decode"),
        "attention_matching_ttft_ratio_to_fastest_exact_upper_95": bootstrap_bound(samples, match_ttft, True, seed + ":match-ttft"),
        "attention_matching_decode_ratio_lower_95": bootstrap_bound(samples, match_decode, False, seed + ":match-decode"),
        "committed_context_savings_vs_smallest_exact_lower_95": bootstrap_bound(samples, savings, False, seed + ":savings"),
    }
    gates = {
        "packed_exact_ttft": bounds["packed_exact_ttft_ratio_upper_95"] <= 1.05,
        "packed_exact_decode": bounds["packed_exact_decode_ratio_lower_95"] >= 1 / 1.05,
        "attention_matching_ttft": bounds["attention_matching_ttft_ratio_to_fastest_exact_upper_95"] <= 1.25,
        "attention_matching_decode": bounds["attention_matching_decode_ratio_lower_95"] >= .90,
        "committed_context_savings": bounds["committed_context_savings_vs_smallest_exact_lower_95"] >= .25,
    }
    return {"schema": OUTPUT_SCHEMA, "passed": all(gates.values()), "qualification_credit": False,
            "performance_scope": "full_request_ingestion_through_first_public_content_and_decode",
            "restore_performance_report_substitute": False, "raw_campaign": campaign_ref,
            "execution_authority": {"source_freeze": execution_authority["source_freeze"],
                 "full_cli_receipt": execution_authority["full_cli_receipt"],
                 "binary": execution_authority["binary"], "matrix": execution_authority["matrix"],
                 "matrix_children": children, "profile": execution_authority["profile"],
                 "model_config": execution_authority["model_config"],
                 "request_config": execution_authority["request_config"],
                 "native_identity": execution_authority["native_identity"],
                 "model_artifact": execution_authority["model_artifact"]},
            "warmup_triplets": 5, "measured_triplets": 30,
            "order_counts": {"/".join(k): v for k, v in sorted(order_counts.items())},
            "bounds": bounds, "gates": gates}

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--campaign-sha256", required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--authority-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    need(args.campaign.is_absolute() and args.campaign.is_file() and
         sha256(args.campaign) == args.campaign_sha256, "campaign pin")
    need(args.authority.is_absolute() and args.authority.is_file() and
         sha256(args.authority) == args.authority_sha256, "authority pin")
    need(args.output.is_absolute() and not args.output.exists(), "fresh output")
    result = audit(json.loads(args.campaign.read_bytes()),
                   {"path": str(args.campaign), "sha256": args.campaign_sha256},
                   json.loads(args.authority.read_bytes()))
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return 0 if result["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
