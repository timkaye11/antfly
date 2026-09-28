#!/usr/bin/env python3
"""Evaluate paired one-token yes/no SFT answers in a pinned MLX runtime.

This development diagnostic uses native-prepared tokens and exact adapter
weights. Passing is not native-generation parity, sealed acceptance, or CUDA
oracle qualification. Native loss and saved-adapter reload need separate checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import qualify_gemma4_preference_quality_campaign as quality


class AnswerQualityError(ValueError):
    pass


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def validate_examples(
    summary: dict, expected_examples: int
) -> tuple[list[dict], list[int]]:
    if type(expected_examples) is not int or expected_examples < 1:
        raise AnswerQualityError("expected example count must be positive")
    examples = summary.get("examples")
    if (
        summary.get("schema_version") != "gemma4_prepared/v6"
        or summary.get("examples_truncated") != 0
        or summary.get("examples_seen") != expected_examples
        or not isinstance(examples, list)
        or len(examples) != expected_examples
    ):
        raise AnswerQualityError(
            "expected complete, untruncated v6 evaluation examples"
        )
    groups: set[str] = set()
    prompts: set[tuple[int, ...]] = set()
    answers: set[int] = set()
    for row in examples:
        group = row.get("source_group_id")
        prompt = row.get("prompt_input_ids")
        response = row.get("response_input_ids")
        if (
            not isinstance(group, str)
            or not group
            or group in groups
            or not isinstance(prompt, list)
            or not prompt
            or any(type(token) is not int or token < 0 for token in prompt)
            or tuple(prompt) in prompts
        ):
            raise AnswerQualityError("evaluation needs unique groups and prompt tokens")
        if (
            not isinstance(response, list)
            or len(response) != 3
            or any(type(token) is not int or token < 0 for token in response)
            or response[1:] != [106, 107]
        ):
            raise AnswerQualityError("expected answer token, end-of-turn, newline")
        if (
            row.get("was_truncated") is not False
            or row.get("turn_count") != 2
            or row.get("input_ids") != prompt + response
            or row.get("labels") != [-100] * len(prompt) + response
            or row.get("num_supervised_tokens") != 3
        ):
            raise AnswerQualityError("prepared prompt/response supervision mismatch")
        groups.add(group)
        prompts.add(tuple(prompt))
        answers.add(response[0])
    if answers != {1904, 4443}:
        raise AnswerQualityError("expected both locked Gemma yes/no token IDs")
    return examples, sorted(answers)


def summarize_rows(rows: list[dict]) -> dict:
    if not rows:
        raise AnswerQualityError("answer evaluation is empty")
    return {
        "rows": rows,
        **{
            field: sum(row[key] for row in rows) / len(rows)
            for field, key in (
                ("accuracy", "correct"),
                ("valid_answer_rate", "valid_answer"),
                ("forced_choice_accuracy", "forced_choice_correct"),
                ("complete_answer_accuracy", "complete_answer_correct"),
            )
        },
    }


def score_answers(baseline: list[dict], trained: list[dict]) -> dict:
    """Recompute correctness, pairing by identity rather than trusting row order."""

    def checked(rows: list[dict]) -> dict[str, dict]:
        result = {}
        for row in rows:
            group = row.get("source_group_id")
            target = row.get("target_token_id")
            prediction = row.get("predicted_token_id")
            forced = row.get("forced_choice_token_id")
            following = row.get("next_token_id")
            if not isinstance(group, str) or not group or group in result:
                raise AnswerQualityError("duplicate or missing answer group")
            if (
                type(target) is not int
                or target not in (1904, 4443)
                or type(prediction) is not int
                or prediction < 0
                or type(forced) is not int
                or forced not in (1904, 4443)
                or (
                    following is not None
                    and (type(following) is not int or following < 0)
                )
            ):
                raise AnswerQualityError("invalid answer token IDs")
            result[group] = {
                **row,
                "correct": prediction == target,
                "valid_answer": prediction in (1904, 4443),
                "forced_choice_correct": forced == target,
                "complete_answer_correct": prediction == target and following == 106,
            }
        return result

    before, after = checked(baseline), checked(trained)
    if not before or before.keys() != after.keys():
        raise AnswerQualityError("baseline and trained evaluation groups differ")
    ids = sorted(before)
    if any(
        before[key]["target_token_id"] != after[key]["target_token_id"] for key in ids
    ):
        raise AnswerQualityError("paired answer targets differ")
    evaluations = {
        "baseline": summarize_rows([before[key] for key in ids]),
        "trained": summarize_rows([after[key] for key in ids]),
    }
    paired = quality._paired_prompt_reward_test(
        [float(before[key]["complete_answer_correct"]) for key in ids],
        [float(after[key]["complete_answer_correct"]) for key in ids],
        group_size=1,
        maximum_p_value=0.05,
    )
    forced_ok = (
        evaluations["trained"]["forced_choice_accuracy"]
        >= evaluations["baseline"]["forced_choice_accuracy"]
    )
    first_improved = (
        evaluations["trained"]["accuracy"] > evaluations["baseline"]["accuracy"]
    )
    return {
        "evaluations": evaluations,
        "paired_test": paired,
        "forced_choice_nonregression": forced_ok,
        "first_token_improved": first_improved,
        "passed": paired["passed"] and forced_ok and first_improved,
    }


def evaluate(args: argparse.Namespace) -> dict:
    # Keep Metal imports behind CLI execution so contract tests need no GPU.
    import run_gemma4_grpo_boolq_mlx_parity as legacy
    import run_gemma4_grpo_mlx_benchmark as micro
    from gemma4_mlx_source import attest_mlx_lm_archive

    locked = micro.locked
    bound_paths = [
        Path(__file__),
        Path(quality.__file__),
        args.prepared_eval,
        args.mlx_attestation,
        locked.LOCK_PATH,
    ]
    for directory in (args.initial_adapter, args.trained_adapter):
        bound_paths.extend(
            directory / name
            for name in (
                "adapter_model.safetensors",
                "adapter_config.json",
                "antfly_finetune_manifest.json",
            )
        )
    bindings = {str(path.resolve()): sha256(path) for path in bound_paths}
    summary = read(args.prepared_eval)["summary"]
    examples, answer_ids = validate_examples(summary, args.expected_examples)
    lock = locked.load_lock(locked.LOCK_PATH)
    locked.force_offline_environment()
    locked.verify_model_directory(lock, args.model_id, args.model)
    locked.require_prepared_model_binding(summary, args.model)
    prior = read(args.mlx_attestation)
    runtime = legacy.attest_wheel_runtime(
        runtime_root=Path(prior["runtime"]["runtime_root"]),
        wheel_path=Path(prior["runtime"]["wheels"]["mlx"]["archive_path"]),
        metal_wheel_path=Path(prior["runtime"]["wheels"]["mlx-metal"]["archive_path"]),
        expected_version=lock["mlx_reference"]["packages"]["mlx"],
    )
    source = attest_mlx_lm_archive(
        Path(prior["source"]["source_root"]),
        Path(prior["source"]["archive_path"]),
        lock["mlx_reference"]["source_revisions"]["mlx-lm"],
    )
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_unflatten

    if (
        not Path(mx.__file__)
        .resolve()
        .is_relative_to(Path(runtime["runtime_root"]).resolve())
    ):
        raise AnswerQualityError("imported MLX differs from attested runtime")
    micro.install_mlx_lm_source_namespace(Path(source["source_root"]))
    from mlx_lm.models import gemma4
    from mlx_lm.tuner.lora import LoRALinear

    mx.set_default_device(mx.gpu)
    mx.set_cache_limit(256 * 1024**2)
    mx.random.seed(42)
    model, _ = locked.load_locked_mlx_gemma4(
        args.model,
        mx,
        load_config_fn=lambda path: read(path / "config.json"),
        get_model_classes_fn=lambda **kwargs: (gemma4.Model, gemma4.ModelArgs),
    )
    model.freeze()
    base_inventory = locked.require_bf16_base_model(model, mx)
    targets = locked.target_module_names(model, lock, args.model_id, args.target_preset)
    updates = []
    gate = lock["performance_gate"]
    for name, module in model.named_modules():
        if name in set(targets):
            if not isinstance(module, nn.Linear):
                raise AnswerQualityError(f"nonlinear LoRA target: {name}")
            updates.append(
                (
                    name,
                    LoRALinear.from_base(
                        module,
                        r=gate["rank"],
                        scale=gate["alpha"] / gate["rank"],
                        dropout=0.0,
                    ),
                )
            )
    if {name for name, _ in updates} != set(targets):
        raise AnswerQualityError("model LoRA target inventory differs")
    model.update_modules(tree_unflatten(updates))
    locked.require_exact_trainables(model, targets, mx)
    model.eval()
    evaluations = {}
    for label, directory in (
        ("baseline", args.initial_adapter),
        ("trained", args.trained_adapter),
    ):
        adapter = locked.inspect_initial_adapter(
            directory, lock, args.model_id, args.target_preset, summary
        )
        locked.load_exact_initial_adapter(model, targets, adapter, mx)
        mx.eval(model.parameters())
        mx.synchronize()
        rows = []
        for index, row in enumerate(examples):
            logits = model(mx.array([row["prompt_input_ids"]], dtype=mx.int32))[
                0, -1
            ].astype(mx.float32)
            if not bool(mx.all(mx.isfinite(logits)).item()):
                raise AnswerQualityError("nonfinite answer logits")
            prediction = int(mx.argmax(logits).item())
            forced = answer_ids[
                int(mx.argmax(logits[mx.array(answer_ids, dtype=mx.int32)]).item())
            ]
            following = None
            if prediction in answer_ids:
                continuation = model(
                    mx.array([row["prompt_input_ids"] + [prediction]], dtype=mx.int32)
                )[0, -1]
                if not bool(mx.all(mx.isfinite(continuation)).item()):
                    raise AnswerQualityError("nonfinite continuation logits")
                following = int(mx.argmax(continuation).item())
            rows.append(
                {
                    "source_group_id": row["source_group_id"],
                    "target_token_id": row["response_input_ids"][0],
                    "predicted_token_id": prediction,
                    "forced_choice_token_id": forced,
                    "next_token_id": following,
                }
            )
            if (index + 1) % 32 == 0:
                print(json.dumps({"phase": label, "evaluated": index + 1}), flush=True)
        evaluations[label] = rows
    result = score_answers(evaluations["baseline"], evaluations["trained"])
    for label, directory in (
        ("baseline", args.initial_adapter),
        ("trained", args.trained_adapter),
    ):
        result["evaluations"][label]["adapter_sha256"] = bindings[
            str((directory / "adapter_model.safetensors").resolve())
        ]
    for path, digest in bindings.items():
        if sha256(Path(path)) != digest:
            raise AnswerQualityError(f"input changed during evaluation: {path}")
    return {
        "schema_version": "antfly.gemma4.sft-answer-quality/v1",
        "classification": "paired development yes/no answers in pinned MLX; not production qualification",
        "production_qualified": False,
        "input_sha256": bindings,
        "runtime_attestation": runtime,
        "source_attestation": source,
        "base_inventory": base_inventory,
        **result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--model-id", choices=("gemma-4-E2B-it", "gemma-4-E4B-it"), required=True
    )
    parser.add_argument("--prepared-eval", type=Path, required=True)
    parser.add_argument("--expected-examples", type=int, required=True)
    parser.add_argument("--initial-adapter", type=Path, required=True)
    parser.add_argument("--trained-adapter", type=Path, required=True)
    parser.add_argument(
        "--target-preset", choices=("peft-qv", "text-all-linear"), default="peft-qv"
    )
    parser.add_argument(
        "--mlx-attestation",
        type=Path,
        required=True,
        help="runtime/source archive locations; hashes reverified",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    result = evaluate(args)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps({"passed": result["passed"], "paired_test": result["paired_test"]})
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
