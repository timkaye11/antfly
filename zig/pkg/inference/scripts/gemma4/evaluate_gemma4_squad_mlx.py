#!/usr/bin/env python3
"""Generate paired multi-token SQuAD answers in the pinned MLX reference.

Uses native-prepared prompts, complete greedy answers and article-grouped
scoring. This is a task-quality check; native generation and CUDA numerical
parity remain separate requirements. No network access occurs during a run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from score_gemma4_squad_sft import score_pairs


def read(path: Path):
    return json.loads(path.read_text())


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_prepared(
    summary: dict, references: list[dict], selection: dict, source_dataset: Path
) -> list[dict]:
    from gemma4_oracle_contract import verify_prepared_source_dataset

    expected = selection["splits"]["eval"]["examples"]
    if (
        summary.get("schema_version") != "gemma4_prepared/v6"
        or summary.get("examples_seen") != expected
        or summary.get("examples_truncated") != 0
        or summary.get("source_split") != "eval"
        or sha(source_dataset) != selection["splits"]["eval"]["dataset_sha256"]
    ):
        raise ValueError("prepared evaluation differs from the frozen selection")
    # Native provenance hashes the split, leaf filename and bytes in a domain;
    # the selection manifest records the raw file SHA. Require both identities.
    verify_prepared_source_dataset(summary, source_dataset)
    refs = {r["id"]: r for r in references}
    rows = summary.get("examples")
    if (
        not isinstance(rows, list)
        or len(rows) != expected
        or len(references) != expected
        or len(refs) != expected
        or {r["source_id"] for r in rows} != refs.keys()
    ):
        raise ValueError("prepared/reference identities or counts differ")
    prompts = set()
    groups = set()
    for row in rows:
        prompt, response = row["prompt_input_ids"], row["response_input_ids"]
        if (
            not isinstance(prompt, list)
            or not prompt
            or not isinstance(response, list)
            or len(response) < 4
            or len(response) > 66
            or response[-2:] != [106, 107]
            or any(type(t) is not int or t < 0 for t in prompt + response)
            or row["input_ids"] != prompt + response
            or row["labels"] != [-100] * len(prompt) + response
            or row["num_supervised_tokens"] != len(response)
            or row["was_truncated"] is not False
            or row["turn_count"] != 2
            or len(prompt + response) > 512
            or tuple(prompt) in prompts
            or row["source_group_id"] in groups
        ):
            raise ValueError("invalid, duplicate or truncated multi-token supervision")
        prompts.add(tuple(prompt))
        groups.add(row["source_group_id"])
    return sorted(rows, key=lambda row: row["source_id"])


def greedy_tokens(prompt: list[int], next_token, *, use_cache: bool) -> list[int]:
    """Allow 64 answer tokens and one final end-of-turn prediction."""
    generated = []
    for step in range(65):
        inputs = prompt + generated if not use_cache or step == 0 else [generated[-1]]
        token = next_token(inputs)
        if type(token) is not int or token < 0:
            raise ValueError("invalid generated token")
        generated.append(token)
        if token == 106:
            break
    return generated


def evaluate(args) -> dict:
    import run_gemma4_grpo_boolq_mlx_parity as legacy
    import run_gemma4_grpo_mlx_benchmark as micro
    import score_gemma4_squad_sft as scorer
    from gemma4_mlx_source import attest_mlx_lm_archive
    from tokenizers import Tokenizer

    locked = micro.locked
    selection = read(args.selection_manifest)
    if sha(args.references) != selection["splits"]["eval"]["references_sha256"]:
        raise ValueError("references differ from frozen selection")
    references = read(args.references)
    summary = read(args.prepared_eval)["summary"]
    examples = validate_prepared(summary, references, selection, args.source_dataset)
    bound = [
        Path(__file__),
        Path(scorer.__file__),
        args.selection_manifest,
        args.references,
        args.prepared_eval,
        args.source_dataset,
        args.mlx_attestation,
        locked.LOCK_PATH,
        args.model / "tokenizer.json",
    ]
    for directory in (args.initial_adapter, args.trained_adapter):
        bound.extend(
            directory / name
            for name in (
                "adapter_model.safetensors",
                "adapter_config.json",
                "antfly_finetune_manifest.json",
            )
        )
    bindings = {str(path.resolve()): sha(path) for path in bound}
    lock = locked.load_lock(locked.LOCK_PATH)
    locked.force_offline_environment()
    locked.verify_model_directory(lock, args.model_id, args.model)
    locked.require_prepared_model_binding(summary, args.model)
    if sha(args.model / "tokenizer.json") != selection["tokenizer_sha256"]:
        raise ValueError("selection tokenizer differs from model")
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
        raise ValueError("imported MLX differs from attested runtime")
    micro.install_mlx_lm_source_namespace(Path(source["source_root"]))
    from mlx_lm.models import gemma4
    from mlx_lm.models.cache import make_prompt_cache
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
    inventory = locked.require_bf16_base_model(model, mx)
    targets = locked.target_module_names(model, lock, args.model_id, "text-all-linear")
    gate = lock["performance_gate"]
    updates = []
    for name, module in model.named_modules():
        if name in set(targets):
            if not isinstance(module, nn.Linear):
                raise ValueError("nonlinear adapter target: " + name)
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
        raise ValueError("adapter target inventory differs")
    model.update_modules(tree_unflatten(updates))
    locked.require_exact_trainables(model, targets, mx)
    model.eval()
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    # Bind imported repository helpers as well as this entry point. The pinned
    # upstream MLX source and wheel inventories are attested independently.
    helper_root = Path(__file__).resolve().parent
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.parent == helper_root and path.suffix == ".py":
                bindings.setdefault(str(path), sha(path))
    predictions = {}
    cache_comparisons = []
    for label, adapter_dir in (
        ("baseline", args.initial_adapter),
        ("trained", args.trained_adapter),
    ):
        adapter = locked.inspect_initial_adapter(
            adapter_dir, lock, args.model_id, "text-all-linear", summary
        )
        locked.load_exact_initial_adapter(model, targets, adapter, mx)
        mx.eval(model.parameters())
        mx.synchronize()
        rows = []
        for index, row in enumerate(examples):

            def generate(use_cache):
                cache = make_prompt_cache(model) if use_cache else None

                def next_token(inputs):
                    logits = model(mx.array([inputs], dtype=mx.int32), cache=cache)[
                        0, -1
                    ].astype(mx.float32)
                    if not bool(mx.all(mx.isfinite(logits)).item()):
                        raise ValueError("nonfinite generation logits")
                    return int(mx.argmax(logits).item())

                tokens = greedy_tokens(
                    row["prompt_input_ids"], next_token, use_cache=use_cache
                )
                mx.synchronize()
                return tokens

            tokens = generate(True)
            if index < 8:
                uncached = generate(False)
                if tokens != uncached:
                    raise ValueError(
                        f"cached and uncached generation differ: {label} {row['source_id']}"
                    )
                cache_comparisons.append(
                    {
                        "phase": label,
                        "id": row["source_id"],
                        "matched_tokens": len(tokens),
                    }
                )
            terminated = tokens[-1] == 106
            answer_ids = tokens[:-1] if terminated else tokens
            rows.append(
                {
                    "id": row["source_id"],
                    "text": tokenizer.decode(answer_ids, skip_special_tokens=False),
                    "token_ids": tokens,
                    "terminated": terminated,
                }
            )
            if (index + 1) % 16 == 0:
                print(json.dumps({"phase": label, "evaluated": index + 1}), flush=True)
        predictions[label] = rows
    result = score_pairs(references, predictions["baseline"], predictions["trained"])
    for path, expected in bindings.items():
        if sha(Path(path)) != expected:
            raise ValueError("input changed during evaluation: " + path)
    return {
        **result,
        "schema": "antfly.gemma4.squad-sft-evaluation/v1",
        "model_id": args.model_id,
        "target_preset": "text-all-linear",
        "initialization_seed": read(
            args.initial_adapter / "antfly_finetune_manifest.json"
        )["initialization_seed"],
        "input_sha256": bindings,
        "runtime_attestation": runtime,
        "source_attestation": source,
        "base_inventory": inventory,
        "predictions": predictions,
        "cache_comparisons": cache_comparisons,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "initial-adapter",
        "trained-adapter",
        "prepared-eval",
        "references",
        "selection-manifest",
        "source-dataset",
        "mlx-attestation",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--model-id", choices=("gemma-4-E2B-it", "gemma-4-E4B-it"), required=True
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output already exists")
    result = evaluate(args)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "checks": result["checks"],
                "paired_article_test": result["paired_article_test"],
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
