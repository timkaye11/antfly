#!/usr/bin/env python3
"""Fail-closed artifact gate for Gemma4 GRPO incremental KV rollout.

The candidate must preserve the baseline's sampled completions, reward/KL
traces, optimizer trajectory, held-out metrics, and final adapter bytes. Timing
and path fields are intentionally excluded from semantic equality; the
candidate must additionally prove that aligned prompt pages were reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "antfly_gemma4_grpo_incremental_kv_parity/v1"
TRAIN_SCHEMAS = {
    "antfly_inference_finetune_grpo_report/v4",
    "antfly_inference_finetune_grpo_report/v5",
    "antfly_inference_finetune_grpo_report/v6",
}
EVAL_SCHEMAS = {
    "antfly_inference_finetune_grpo_evaluation/v2",
    "antfly_inference_finetune_grpo_evaluation/v3",
}
EXACT_FILES = (
    "grpo_reward_trace.jsonl",
    "grpo_evaluation_reward_trace.jsonl",
    "grpo_kl_control_trace.jsonl",
    "adapter-trained/adapter_model.safetensors",
)
TRAIN_SEMANTIC_FIELDS = (
    "execution_mode",
    "dataset_format",
    "completions",
    "tokens",
    "groups",
    "loss",
    "pg_loss",
    "kl_loss",
    "mean_kl",
    "clip_fraction",
    "mean_reward",
    "reward_stddev",
    "policy_backend",
    "optimizer_steps",
    "micro_batch_steps",
    "policy_logprob_mode",
    "policy_rescore_completions",
    "training_microbatch_mode",
    "reference_mode",
    "reference_cache",
    "initial_logprob_parity",
)
EVAL_SEMANTIC_FIELDS = (
    "status",
    "dataset_fingerprint",
    "policy_adapter_digest",
    "policy_backend",
    "groups",
    "completions",
    "tokens",
    "prompt_overlap_count",
    "mean_reward",
    "top_rank_mean_reward",
    "positive_reward_group_rate",
    "reward_stddev",
    "loss",
    "pg_loss",
    "kl_loss",
    "mean_kl",
    "clip_fraction",
    "minimums",
    "reference_mode",
    "execution_order",
)


class ParityError(RuntimeError):
    pass


def load_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ParityError(f"could not load {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise ParityError(f"{label} must be a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ParityError(f"could not hash {path}: {exc}") from exc
    return "sha256:" + digest.hexdigest()


def select_fields(report: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
    missing = [field for field in fields if field not in report]
    if missing:
        raise ParityError(f"report is missing semantic fields: {', '.join(missing)}")
    return {field: report[field] for field in fields}


def require_equal(label: str, baseline: Any, candidate: Any) -> None:
    if baseline != candidate:
        raise ParityError(f"{label} differs between full-prefix and incremental-KV runs")


def require_incremental_telemetry(
    report: Mapping[str, Any],
    *,
    expected_groups: int,
    expected_completions: int,
    expected_tokens: int,
    label: str,
    require_active_batching: bool = False,
    require_prompt_tail_cloning: bool = False,
) -> Mapping[str, Any]:
    telemetry = report.get("incremental_kv")
    if not isinstance(telemetry, dict):
        raise ParityError(f"{label} incremental KV telemetry is missing")
    integer_fields = (
        "groups",
        "prompt_prefill_forwards",
        "prompt_tail_prefill_forwards",
        "prompt_tail_prefill_candidates",
        "decode_forwards",
        "exact_logprob_rescore_forwards",
        "resident_ranked_token_selections",
        "host_logit_fallbacks",
        "shared_prompt_tokens",
        "reused_candidate_prompt_tokens",
        "cache_page_tokens",
    )
    for field in integer_fields:
        value = telemetry.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ParityError(f"{label} incremental KV telemetry field {field} is invalid")
    if telemetry["groups"] != expected_groups:
        raise ParityError(f"{label} incremental KV group count drifted")
    if telemetry["prompt_prefill_forwards"] != expected_groups:
        raise ParityError(f"{label} did not execute exactly one canonical prompt prefill per group")
    if telemetry["decode_forwards"] == 0:
        raise ParityError(f"{label} did not execute incremental decode forwards")
    if telemetry["exact_logprob_rescore_forwards"] != expected_completions:
        raise ParityError(f"{label} did not exactly rescore every incremental completion")
    # Selecting rank r requires r+1 resident argmax passes with progressively
    # suppressed winners. The dispatch count therefore exceeds the generated
    # token count whenever candidates use ranks above zero.
    if telemetry["resident_ranked_token_selections"] < expected_tokens:
        raise ParityError(f"{label} did not rank every generated token on device")
    if telemetry["host_logit_fallbacks"] != 0:
        raise ParityError(f"{label} downloaded eager logits to the host")
    if telemetry["shared_prompt_tokens"] == 0 or telemetry["reused_candidate_prompt_tokens"] == 0:
        raise ParityError(f"{label} did not reuse aligned prompt pages")
    if telemetry["cache_page_tokens"] != 16 or telemetry.get("cache_dtype") != "f32":
        raise ParityError(f"{label} KV cache geometry is outside the qualified exactness lane")
    if require_active_batching:
        if telemetry.get("active_candidate_batching") is not True:
            raise ParityError(f"{label} did not enable active-candidate decode batching")
        for field in ("decode_forward_candidates", "max_decode_batch_size"):
            value = telemetry.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ParityError(
                    f"{label} incremental KV batching field {field} is invalid"
                )
        if telemetry["decode_forward_candidates"] <= telemetry["decode_forwards"]:
            raise ParityError(f"{label} did not amortize any decode forward")
        if telemetry["max_decode_batch_size"] < 2:
            raise ParityError(f"{label} did not execute a multi-candidate decode batch")
    if require_prompt_tail_cloning:
        if telemetry.get("prompt_tail_cloning") is not True:
            raise ParityError(f"{label} did not enable segmented prompt-tail clone fan-out")
        for field in ("prompt_tail_clone_candidates", "prompt_tail_clone_tokens"):
            value = telemetry.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ParityError(f"{label} prompt-tail clone field {field} is invalid")
        tail_groups = telemetry["prompt_tail_prefill_forwards"]
        if tail_groups <= 0 or tail_groups > expected_groups:
            raise ParityError(f"{label} segmented prompt-tail group accounting is invalid")
        if telemetry["prompt_tail_prefill_candidates"] != tail_groups:
            raise ParityError(f"{label} prompt-tail replay candidate accounting drifted")
        if expected_groups <= 0 or expected_completions % expected_groups != 0:
            raise ParityError(f"{label} completion-group geometry is invalid")
        clone_fanout = expected_completions // expected_groups - 1
        expected_clone_candidates = tail_groups * clone_fanout
        if clone_fanout <= 0 or telemetry["prompt_tail_clone_candidates"] != expected_clone_candidates:
            raise ParityError(f"{label} did not fan the segmented prompt-tail KV out to every remaining candidate")
    return telemetry


def validate(
    baseline_root: Path,
    candidate_root: Path,
    model_key: str,
    *,
    require_active_batching: bool = False,
    require_prompt_tail_cloning: bool = False,
) -> dict[str, Any]:
    baseline_root = baseline_root.expanduser().resolve()
    candidate_root = candidate_root.expanduser().resolve()
    baseline_train = load_json(baseline_root / "grpo_report.json", "baseline train report")
    candidate_train = load_json(candidate_root / "grpo_report.json", "candidate train report")
    baseline_eval = load_json(
        baseline_root / "grpo_evaluation_report.json", "baseline evaluation report"
    )
    candidate_eval = load_json(
        candidate_root / "grpo_evaluation_report.json", "candidate evaluation report"
    )
    if baseline_train.get("schema_version") not in TRAIN_SCHEMAS:
        raise ParityError("baseline train report schema is unsupported")
    if candidate_train.get("schema_version") not in {
        "antfly_inference_finetune_grpo_report/v5",
        "antfly_inference_finetune_grpo_report/v6",
    }:
        raise ParityError("candidate train report must use an incremental-KV v5/v6 schema")
    if baseline_eval.get("schema_version") not in EVAL_SCHEMAS:
        raise ParityError("baseline evaluation report schema is unsupported")
    if candidate_eval.get("schema_version") != "antfly_inference_finetune_grpo_evaluation/v3":
        raise ParityError("candidate evaluation report must use the incremental-KV v3 schema")
    if candidate_train.get("sampling_mode") != "shared-page-prompt-ranked-incremental-kv":
        raise ParityError("candidate did not report the incremental KV sampling mode")

    require_equal(
        "training semantics",
        select_fields(baseline_train, TRAIN_SEMANTIC_FIELDS),
        select_fields(candidate_train, TRAIN_SEMANTIC_FIELDS),
    )
    require_equal(
        "evaluation semantics",
        select_fields(baseline_eval, EVAL_SEMANTIC_FIELDS),
        select_fields(candidate_eval, EVAL_SEMANTIC_FIELDS),
    )
    train_telemetry = require_incremental_telemetry(
        candidate_train,
        expected_groups=int(candidate_train["groups"]),
        expected_completions=int(candidate_train["completions"]),
        expected_tokens=int(candidate_train["tokens"]),
        label="training",
        require_active_batching=require_active_batching,
        require_prompt_tail_cloning=require_prompt_tail_cloning,
    )
    eval_telemetry = require_incremental_telemetry(
        candidate_eval,
        expected_groups=int(candidate_eval["groups"]),
        expected_completions=int(candidate_eval["completions"]),
        expected_tokens=int(candidate_eval["tokens"]),
        label="evaluation",
        require_active_batching=require_active_batching,
        require_prompt_tail_cloning=require_prompt_tail_cloning,
    )

    exact_hashes: dict[str, str] = {}
    for relative in EXACT_FILES:
        baseline_hash = sha256_file(baseline_root / relative)
        candidate_hash = sha256_file(candidate_root / relative)
        require_equal(relative, baseline_hash, candidate_hash)
        exact_hashes[relative] = candidate_hash

    initial_parity = candidate_train.get("initial_logprob_parity")
    if not isinstance(initial_parity, dict) or initial_parity.get("sampling_rescore_max_abs_error") != 0:
        raise ParityError("candidate sampling/rescore parity is not exact zero")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "model_key": model_key,
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "required_active_candidate_batching": require_active_batching,
        "required_prompt_tail_cloning": require_prompt_tail_cloning,
        "exact_hashes": exact_hashes,
        "training_incremental_kv": train_telemetry,
        "evaluation_incremental_kv": eval_telemetry,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--model-key", choices=("gemma-4-E2B-it", "gemma-4-E4B-it"), required=True)
    parser.add_argument(
        "--require-active-batching",
        action="store_true",
        help="require proof that multi-candidate paged decode forwards were executed",
    )
    parser.add_argument(
        "--require-prompt-tail-cloning",
        action="store_true",
        help="require proof that every non-page-aligned prompt used one segmented canonical tail cloned to every private candidate",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = validate(
            args.baseline_root,
            args.candidate_root,
            args.model_key,
            require_active_batching=args.require_active_batching,
            require_prompt_tail_cloning=args.require_prompt_tail_cloning,
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    except ParityError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
