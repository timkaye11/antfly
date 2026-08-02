#!/usr/bin/env python3
"""Fail-closed Gemma 4 26B-A4B token parity tooling.

The tool is split into two argparse subcommands so the expensive pinned
llama.cpp oracle only ever runs when a reference is (re)recorded:

* ``record-reference`` is the ONLY mode that launches llama.cpp.  It runs the
  pinned, token-instrumented binary once per KV-cache dtype lane (``f16``,
  ``f32``, or both), strictly sequentially, and writes a reference bundle
  directory containing ``reference.json`` (schema
  ``antfly.gemma4_a4b_reference.v1``) plus the raw llama.cpp logs.  The bundle
  pins the prompt, token IDs, llama binary/model/tokenizer SHA-256 hashes, and
  any optional oracle instrumentation markers:
  ``reference_route: <layer> <k> <expert_id...>`` and
  ``reference_logit_sample: <index> <value>``.  Absent markers are recorded as
  empty lists, never an error.

* ``verify-reference`` never launches llama.cpp (passing ``--llama-bin`` is a
  usage error).  It re-hashes the model and fails closed on drift from the
  bundle (``--allow-model-hash-mismatch`` downgrades that to a warning).  By
  default it runs only the production compact Antfly engine (``metal`` backend
  with ``--memory-profile 2gbs``) per recorded lane; ``--full-matrix`` adds the
  native and canonical-Metal engines sequentially.  Gates are fail-closed:
  exact prompt token IDs, exact output token IDs, and the recorded KV-cache
  dtype provenance.  When the reference carries expert routes the Antfly run is
  executed with ``TERMITE_DEBUG_GPT_LAST_ROW=1`` so Antfly prints
  ``layer<N>_routes: <expert>:<weight> ...`` lines (the format emitted by
  ``maybeDebugMoeRoutesLastRow`` in ``gpt.zig``); the top-8 expert ID sets are
  compared per layer using the last emitted line per layer on both sides.
  Logit samples (``logit_sample: <index> <value>`` on the Antfly side, last
  occurrence per index) must match the reference within an absolute tolerance
  of 0.01.  Optional data missing on either side skips the gate and records it
  in the summary ``skipped_gates`` — never a silent pass claim.  Results land
  in ``summary.json`` (schema ``antfly.gemma4_a4b_parity.v3``).

Exit codes: 0 parity pass, 1 gate/parity failure, 2 usage error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any


PINNED_LLAMA_COMMIT = "32703b42d6d8a74336bf27613705cbc76e2363ee"
REFERENCE_SCHEMA = "antfly.gemma4_a4b_reference.v1"
SUMMARY_SCHEMA = "antfly.gemma4_a4b_parity.v3"
ROUTE_SUMMARY_SCHEMA = "antfly.gemma4_a4b_route_parity.v1"
LOGIT_SAMPLE_TOLERANCE = 0.01
ROUTE_TOP_K_COMPARED = 8

ANTFLY_TOKEN_RE = re.compile(r"(?m)^token_ids:\s*(?P<ids>[0-9 ]+?)\s*$")
ANTFLY_PROMPT_RE = re.compile(r"(?m)^prompt_token_ids:\s*(?P<ids>[0-9 ]+?)\s*$")
LLAMA_TOKEN_RE = re.compile(r"(?m)^reference_token_id:\s*(?P<id>\d+)\s*$")
LLAMA_PROMPT_COUNT_RE = re.compile(r"number of tokens in prompt\s*=\s*(?P<count>\d+)")
# llama.cpp's logger prefixes verbose-prompt lines with its elapsed-time and
# level columns (for example, ``1.40.130.003 I      2 -> '<bos>'``).  Capture
# the token immediately before the arrow instead of assuming a bare line.
LLAMA_PROMPT_TOKEN_RE = re.compile(r"(?m)^.*?\b(?P<id>\d+)\s+->\s+'")
ANTFLY_CACHE_DTYPE_RE = re.compile(
    r"(?m)^cache_dtype_effective:\s*requested=(?P<requested>[a-z0-9_]+)\s+"
    r"effective=(?P<effective>[a-z0-9_]+)\s*$"
)
# Optional oracle instrumentation markers recorded by record-reference.
REFERENCE_ROUTE_RE = re.compile(
    r"(?m)^reference_route:\s*(?P<layer>\d+)\s+(?P<k>\d+)\s+(?P<ids>\d+(?:\s+\d+)*)\s*$"
)
REFERENCE_LOGIT_SAMPLE_RE = re.compile(
    r"(?m)^reference_logit_sample:\s*(?P<index>\d+)\s+(?P<value>[-+0-9.eE]+)\s*$"
)
# Antfly-side markers.  maybeDebugMoeRoutesLastRow (gpt.zig) prints
# ``layer{d}_routes:`` followed by space-separated ``expert:weight`` pairs with
# six fractional digits, e.g. ``layer3_routes: 12:0.483210 7:-0.211001``.
ANTFLY_ROUTE_RE = re.compile(r"(?m)^layer(?P<layer>\d+)_routes:\s*(?P<pairs>.*?)\s*$")
ANTFLY_ROUTE_PAIR_RE = re.compile(
    r"^(?P<id>\d+):(?P<weight>-?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?)$"
)
ANTFLY_LOGIT_SAMPLE_RE = re.compile(
    r"(?m)^logit_sample:\s*(?P<index>\d+)\s+(?P<value>[-+0-9.eE]+)\s*$"
)


class ParityError(RuntimeError):
    pass


def _ids(match: re.Match[str] | None, label: str) -> list[int]:
    if match is None:
        raise ParityError(f"missing {label}")
    return [int(value) for value in match.group("ids").split()]


def parse_antfly_output(text: str) -> tuple[list[int], list[int]]:
    return (
        _ids(ANTFLY_PROMPT_RE.search(text), "Antfly prompt_token_ids"),
        _ids(ANTFLY_TOKEN_RE.search(text), "Antfly token_ids"),
    )


def parse_llama_output(text: str) -> tuple[list[int], list[int]]:
    count_match = LLAMA_PROMPT_COUNT_RE.search(text)
    if count_match is None:
        raise ParityError("missing llama.cpp verbose prompt token count")
    prompt_count = int(count_match.group("count"))
    prompt_ids = [int(match.group("id")) for match in LLAMA_PROMPT_TOKEN_RE.finditer(text)]
    if len(prompt_ids) < prompt_count:
        raise ParityError(
            f"llama.cpp printed {len(prompt_ids)} prompt IDs, expected {prompt_count}"
        )
    # Verbose logging can contain token-piece lines for optional suffixes after
    # the prompt.  Only the first declared prompt_count entries are canonical.
    prompt_ids = prompt_ids[:prompt_count]
    token_ids = [int(match.group("id")) for match in LLAMA_TOKEN_RE.finditer(text)]
    if not token_ids:
        raise ParityError(
            "llama.cpp emitted no reference_token_id markers; use the pinned "
            "token-instrumented completion binary"
        )
    return prompt_ids, token_ids


def parse_antfly_cache_dtype(text: str) -> tuple[str, str]:
    match = ANTFLY_CACHE_DTYPE_RE.search(text)
    if match is None:
        raise ParityError("missing Antfly cache_dtype_effective provenance")
    return match.group("requested"), match.group("effective")


def _float(raw: str, label: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise ParityError(f"malformed {label} value {raw!r}") from exc


def parse_reference_routes(text: str) -> list[dict[str, Any]]:
    """Parse optional ``reference_route: <layer> <k> <expert_id...>`` markers."""
    routes: list[dict[str, Any]] = []
    for match in REFERENCE_ROUTE_RE.finditer(text):
        expert_ids = [int(value) for value in match.group("ids").split()]
        top_k = int(match.group("k"))
        if top_k != len(expert_ids):
            raise ParityError(
                f"malformed reference_route marker: k={top_k} but "
                f"{len(expert_ids)} expert IDs: {match.group(0).strip()!r}"
            )
        routes.append(
            {
                "layer": int(match.group("layer")),
                "top_k": top_k,
                "expert_ids": expert_ids,
            }
        )
    return routes


def parse_reference_logit_samples(text: str) -> list[dict[str, Any]]:
    """Parse optional ``reference_logit_sample: <index> <value>`` markers."""
    return [
        {
            "index": int(match.group("index")),
            "value": _float(match.group("value"), "reference_logit_sample"),
        }
        for match in REFERENCE_LOGIT_SAMPLE_RE.finditer(text)
    ]


def parse_antfly_routes(text: str) -> list[dict[str, Any]]:
    """Parse ``layer<N>_routes: <expert>:<weight> ...`` debug lines."""
    routes: list[dict[str, Any]] = []
    for match in ANTFLY_ROUTE_RE.finditer(text):
        pairs = match.group("pairs").split()
        if not pairs:
            raise ParityError(f"malformed Antfly route line: {match.group(0).strip()!r}")
        expert_ids: list[int] = []
        route_weights: list[float] = []
        for pair in pairs:
            pair_match = ANTFLY_ROUTE_PAIR_RE.match(pair)
            if pair_match is None:
                raise ParityError(
                    f"malformed Antfly route pair {pair!r} in {match.group(0).strip()!r}"
                )
            expert_ids.append(int(pair_match.group("id")))
            route_weights.append(_float(pair_match.group("weight"), "Antfly route weight"))
        routes.append(
            {
                "layer": int(match.group("layer")),
                "expert_ids": expert_ids,
                "route_weights": route_weights,
            }
        )
    return routes


def parse_antfly_logit_samples(text: str) -> list[dict[str, Any]]:
    """Parse ``logit_sample: <index> <value>`` lines from Antfly output."""
    return [
        {
            "index": int(match.group("index")),
            "value": _float(match.group("value"), "Antfly logit_sample"),
        }
        for match in ANTFLY_LOGIT_SAMPLE_RE.finditer(text)
    ]


def _last_by_key(entries: list[dict[str, Any]], key: str) -> tuple[list[int], dict[int, dict[str, Any]]]:
    """Deduplicate to the last occurrence per key, preserving first-seen order."""
    order: list[int] = []
    by_key: dict[int, dict[str, Any]] = {}
    for entry in entries:
        value = entry[key]
        if value not in by_key:
            order.append(value)
        by_key[value] = entry
    return order, by_key


def compare_route_sets(
    label: str,
    reference_routes: list[dict[str, Any]],
    antfly_routes: list[dict[str, Any]],
) -> list[str]:
    """Compare ordered top-8 expert IDs per layer, last emitted line per layer."""
    ref_order, ref_by_layer = _last_by_key(reference_routes, "layer")
    _, antfly_by_layer = _last_by_key(antfly_routes, "layer")
    failures: list[str] = []
    for layer in ref_order:
        expected = ref_by_layer[layer]["expert_ids"][:ROUTE_TOP_K_COMPARED]
        antfly_route = antfly_by_layer.get(layer)
        if antfly_route is None:
            failures.append(f"{label} layer {layer} routes missing from Antfly output")
            continue
        actual = antfly_route["expert_ids"][:ROUTE_TOP_K_COMPARED]
        if expected != actual:
            failures.append(
                f"{label} layer {layer} ordered expert mismatch: "
                f"reference={expected} antfly={actual}"
            )
    return failures


def compare_logit_samples(
    label: str,
    reference_samples: list[dict[str, Any]],
    antfly_samples: list[dict[str, Any]],
    tolerance: float = LOGIT_SAMPLE_TOLERANCE,
) -> list[str]:
    """Compare logit samples by index (last occurrence per index on each side)."""
    ref_order, ref_by_index = _last_by_key(reference_samples, "index")
    _, antfly_by_index = _last_by_key(antfly_samples, "index")
    failures: list[str] = []
    for index in ref_order:
        antfly_sample = antfly_by_index.get(index)
        if antfly_sample is None:
            failures.append(f"{label} logit sample index {index} missing from Antfly output")
            continue
        error = abs(ref_by_index[index]["value"] - antfly_sample["value"])
        if error > tolerance:
            failures.append(
                f"{label} logit sample index {index} abs error {error:.6f} "
                f"exceeds {tolerance}"
            )
    return failures


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(
    command: list[str],
    environment: dict[str, str],
    log_path: Path,
    timeout: float,
) -> str:
    started = time.monotonic()
    completed = subprocess.run(
        command,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
    text = completed.stdout
    log_path.write_text(text, encoding="utf-8")
    if completed.returncode != 0:
        raise ParityError(
            f"command exited {completed.returncode} after {time.monotonic() - started:.1f}s: "
            f"{' '.join(command)} (log: {log_path})"
        )
    return text


def _llama_version(llama: Path) -> str:
    return subprocess.check_output(
        [str(llama), "--version"], stderr=subprocess.STDOUT, text=True
    ).strip()


def _antfly_command(
    binary: Path,
    model: Path,
    prompt: str,
    tokens: int,
    backend: str,
    compact: bool,
    cache_dtype: str,
    memory_budget_mb: int = 2048,
    device_routing: str = "off",
    seed: int = 42,
) -> list[str]:
    command = [
        str(binary),
        "generate",
        str(model),
        prompt,
        "--backend",
        backend,
        "--raw-prompt",
        "--max-tokens",
        str(tokens),
        "--temperature",
        "0",
        "--seed",
        str(seed),
        "--cache-dtype",
        cache_dtype,
        "--ignore-eos",
        "--print-token-ids",
        "--print-prompt-token-ids",
        "--print-timing",
    ]
    if compact:
        command.extend(
            (
                "--memory-profile", "2gbs",
                "--memory-budget-mb", str(memory_budget_mb),
                "--compact-device-routing", device_routing,
            )
        )
        if cache_dtype == "f32":
            # The production compact profile intentionally budgets an FP16 KV
            # cache.  The F32 lane is a correctness diagnostic, so explicitly
            # widen only its KV allowance rather than silently changing the
            # profile or cross-comparing different dtypes.
            command.extend(("--kv-budget-mb", "512"))
    elif backend == "native":
        # The native oracle maps the complete 13.4 GiB artifact.  Antfly's
        # conservative default host budget is 5 GiB, so make the exceptional
        # correctness-oracle residency explicit instead of weakening runtime
        # defaults globally.
        command.extend(("--host-budget-mb", "32768", "--combined-budget-mb", "32768"))
    return command


def _llama_command(
    binary: Path,
    model: Path,
    prompt: str,
    tokens: int,
    gpu_layers: int,
    cache_dtype: str,
) -> list[str]:
    return [
        str(binary),
        "-m",
        str(model),
        "--no-conversation",
        "--no-jinja",
        "--special",
        "-p",
        prompt,
        "-n",
        str(tokens),
        "-c",
        "4096",
        "-b",
        "512",
        "-ub",
        "512",
        "-ngl",
        str(gpu_layers),
        "-lv",
        "4",
        "--log-colors",
        "off",
        "-ctk",
        cache_dtype,
        "-ctv",
        cache_dtype,
        "--temp",
        "0",
        "--ignore-eos",
        "--no-display-prompt",
        "--verbose-prompt",
        "--no-warmup",
    ]


def _require_file(path: Path, label: str, executable: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise ParityError(f"missing {label}: {resolved}")
    if executable and not os.access(resolved, os.X_OK):
        raise ParityError(f"{label} is not executable: {resolved}")
    return resolved


def _first_difference(expected: list[int], actual: list[int]) -> str:
    for index, (lhs, rhs) in enumerate(zip(expected, actual)):
        if lhs != rhs:
            return f"index {index}: expected {lhs}, got {rhs}"
    if len(expected) != len(actual):
        return f"length: expected {len(expected)}, got {len(actual)}"
    return "none"


def _lanes(cache_dtype: str) -> list[str]:
    return ["f16", "f32"] if cache_dtype == "both" else [cache_dtype]


def record_reference(args: argparse.Namespace) -> dict[str, Any]:
    """Run the pinned llama.cpp oracle per lane and write the reference bundle."""
    llama = _require_file(args.llama_bin, "llama.cpp binary", executable=True)
    model = _require_file(args.model, "GGUF model")
    tokenizer = (
        _require_file(args.tokenizer_file, "tokenizer file")
        if args.tokenizer_file
        else None
    )
    args.out_dir.mkdir(parents=True, exist_ok=False)

    version = _llama_version(llama)
    expected_commit = args.expected_llama_commit.lower()
    # llama.cpp's default version string abbreviates the revision to seven
    # hexadecimal characters.  The binary SHA below provides the stronger
    # provenance identity for the locally instrumented build.
    if expected_commit and expected_commit[:7] not in version.lower():
        raise ParityError(
            f"llama.cpp version does not contain pinned commit {expected_commit[:7]}: {version!r}"
        )

    environment = dict(os.environ)
    lanes = _lanes(args.cache_dtype)
    lane_docs: dict[str, dict[str, Any]] = {}
    # Lanes run strictly sequentially: the oracle is never launched in parallel.
    for lane in lanes:
        log_path = args.out_dir / f"llama.{lane}.log"
        output = _run(
            _llama_command(llama, model, args.prompt, args.tokens, args.llama_gpu_layers, lane),
            environment,
            log_path,
            args.timeout,
        )
        prompt_ids, token_ids = parse_llama_output(output)
        if len(token_ids) != args.tokens:
            raise ParityError(
                f"{lane}/llama emitted {len(token_ids)} token IDs, "
                f"expected {args.tokens}: {log_path}"
            )
        lane_docs[lane] = {
            "cache_dtype": lane,
            "prompt_token_ids": prompt_ids,
            "token_ids": token_ids,
            # Optional oracle instrumentation; absent markers record empty lists.
            "routes": parse_reference_routes(output),
            "logit_samples": parse_reference_logit_samples(output),
            "log": log_path.name,
        }

    reference = {
        "schema": REFERENCE_SCHEMA,
        "prompt": args.prompt,
        "tokens": args.tokens,
        "cache_dtype_lanes": lanes,
        "lanes": lane_docs,
        "model": str(model),
        "model_sha256": _sha256(model),
        "tokenizer_file": str(tokenizer) if tokenizer else None,
        "tokenizer_sha256": _sha256(tokenizer) if tokenizer else None,
        "llama_binary": str(llama),
        "llama_binary_sha256": _sha256(llama),
        "llama_version": version,
        "expected_llama_commit": expected_commit,
        "llama_gpu_layers": args.llama_gpu_layers,
    }
    (args.out_dir / "reference.json").write_text(
        json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return reference


def _load_reference(reference_arg: Path) -> tuple[Path, dict[str, Any]]:
    path = reference_arg.expanduser().resolve()
    if path.is_dir():
        path = path / "reference.json"
    if not path.is_file():
        raise ParityError(f"missing reference bundle: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    schema = document.get("schema")
    if schema != REFERENCE_SCHEMA:
        raise ParityError(f"unsupported reference schema {schema!r}; expected {REFERENCE_SCHEMA}")
    for key in ("prompt", "tokens", "cache_dtype_lanes", "lanes", "model_sha256"):
        if key not in document:
            raise ParityError(f"reference.json missing key {key!r}: {path}")
    tokens = document["tokens"]
    if not isinstance(tokens, int) or tokens <= 0:
        raise ParityError(f"reference.json tokens must be a positive integer, got {tokens!r}")
    lanes = document["cache_dtype_lanes"]
    if not lanes:
        raise ParityError(f"reference.json has no cache dtype lanes: {path}")
    for lane in lanes:
        lane_doc = document["lanes"].get(lane)
        if lane_doc is None:
            raise ParityError(f"reference.json missing lane {lane!r}: {path}")
        for key in ("prompt_token_ids", "token_ids", "routes", "logit_samples"):
            if key not in lane_doc:
                raise ParityError(f"reference lane {lane!r} missing key {key!r}: {path}")
    return path, document


def _verify_engine(
    *,
    label: str,
    backend: str,
    compact: bool,
    lane: str,
    lane_ref: dict[str, Any],
    lane_dir: Path,
    antfly: Path,
    model: Path,
    prompt: str,
    tokens: int,
    timeout: float,
    memory_budget_mb: int,
    device_routing: str,
    seed: int,
) -> tuple[dict[str, Any], list[str], list[str]]:
    overrides: dict[str, str] = {}
    if label == "metal_canonical":
        overrides["ANTFLY_GEMMA4_DISABLE_COMPACT_MOE"] = "1"
    if lane_ref["routes"]:
        # The route lines come from maybeDebugMoeRoutesLastRow (gpt.zig), which
        # only prints when TERMITE_DEBUG_GPT_LAST_ROW is enabled.
        overrides["TERMITE_DEBUG_GPT_LAST_ROW"] = "1"
    environment = dict(os.environ)
    environment.update(overrides)

    command = _antfly_command(
        antfly,
        model,
        prompt,
        tokens,
        backend,
        compact,
        lane,
        memory_budget_mb,
        device_routing,
        seed,
    )
    log_path = lane_dir / f"{label}.log"
    output = _run(command, environment, log_path, timeout)

    prompt_ids, token_ids = parse_antfly_output(output)
    if len(token_ids) != tokens:
        raise ParityError(
            f"{lane}/{label} emitted {len(token_ids)} token IDs, "
            f"expected {tokens}: {log_path}"
        )
    requested, effective = parse_antfly_cache_dtype(output)
    if requested != lane or effective != lane:
        raise ParityError(
            f"{lane}/{label} cache dtype drifted: requested={requested} effective={effective}"
        )

    failures: list[str] = []
    skipped: list[str] = []
    gates: dict[str, dict[str, str]] = {}

    def gate(name: str, gate_failures: list[str] | None, skip_detail: str = "") -> None:
        if gate_failures is None:
            gates[name] = {"status": "skipped", "detail": skip_detail}
            skipped.append(f"{lane}/{label} {name}: {skip_detail}")
        elif gate_failures:
            gates[name] = {"status": "failed", "detail": "; ".join(gate_failures)}
            failures.extend(gate_failures)
        else:
            gates[name] = {"status": "passed", "detail": ""}

    prompt_failures = []
    if prompt_ids != lane_ref["prompt_token_ids"]:
        prompt_failures.append(
            f"{lane}/{label} prompt mismatch: "
            f"{_first_difference(lane_ref['prompt_token_ids'], prompt_ids)}"
        )
    gate("prompt_token_ids", prompt_failures)

    token_failures = []
    if token_ids != lane_ref["token_ids"]:
        token_failures.append(
            f"{lane}/{label} output mismatch: "
            f"{_first_difference(lane_ref['token_ids'], token_ids)}"
        )
    gate("output_token_ids", token_failures)

    antfly_routes = parse_antfly_routes(output)
    if not lane_ref["routes"]:
        gate("expert_routes", None, "reference recorded no route markers")
    elif not antfly_routes:
        gate("expert_routes", None, "Antfly emitted no layer<N>_routes lines")
    else:
        gate(
            "expert_routes",
            compare_route_sets(f"{lane}/{label}", lane_ref["routes"], antfly_routes),
        )

    antfly_logits = parse_antfly_logit_samples(output)
    if not lane_ref["logit_samples"]:
        gate("logit_samples", None, "reference recorded no logit sample markers")
    elif not antfly_logits:
        gate("logit_samples", None, "Antfly emitted no logit_sample lines")
    else:
        gate(
            "logit_samples",
            compare_logit_samples(
                f"{lane}/{label}", lane_ref["logit_samples"], antfly_logits
            ),
        )

    result = {
        "command": command,
        "log": str(log_path),
        "environment_overrides": overrides,
        "cache_dtype_requested": requested,
        "cache_dtype_effective": effective,
        "cache_dtype_provenance": "Antfly cache_dtype_effective log",
        "memory_budget_mb": memory_budget_mb if compact else None,
        "device_routing": device_routing if compact else None,
        "seed": seed,
        "prompt_token_ids": prompt_ids,
        "token_ids": token_ids,
        "gates": gates,
    }
    return result, failures, skipped


def verify_reference(args: argparse.Namespace) -> dict[str, Any]:
    """Replay recorded lanes against Antfly engines; never launches llama.cpp."""
    antfly = _require_file(args.antfly_bin, "Antfly binary", executable=True)
    model = _require_file(args.model, "GGUF model")
    reference_path, reference = _load_reference(args.reference)

    model_sha256 = _sha256(model)
    model_hash_match = model_sha256 == reference["model_sha256"]
    if not model_hash_match:
        message = (
            f"model hash mismatch: reference={reference['model_sha256']} "
            f"actual={model_sha256} ({model})"
        )
        if not args.allow_model_hash_mismatch:
            raise ParityError(f"{message}; pass --allow-model-hash-mismatch to override")
        print(f"parity warning: {message}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=False)
    prompt = reference["prompt"]
    tokens = reference["tokens"]
    lanes = reference["cache_dtype_lanes"]
    engines_selected: list[tuple[str, str, bool]] = []
    if args.full_matrix:
        engines_selected.append(("native", "native", False))
        engines_selected.append(("metal_canonical", "metal", False))
    engines_selected.append(("metal_compact", "metal", True))

    lane_results: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    skipped_gates: list[str] = []
    for lane in lanes:
        lane_ref = reference["lanes"][lane]
        lane_dir = args.out_dir / lane
        lane_dir.mkdir(parents=True, exist_ok=False)
        lane_failures: list[str] = []
        lane_skipped: list[str] = []
        engine_results: dict[str, dict[str, Any]] = {}
        for label, backend, compact in engines_selected:
            result, engine_failures, engine_skipped = _verify_engine(
                label=label,
                backend=backend,
                compact=compact,
                lane=lane,
                lane_ref=lane_ref,
                lane_dir=lane_dir,
                antfly=antfly,
                model=model,
                prompt=prompt,
                tokens=tokens,
                timeout=args.timeout,
                memory_budget_mb=args.memory_budget_mb,
                device_routing=args.device_routing,
                seed=args.seed,
            )
            engine_results[label] = result
            lane_failures.extend(engine_failures)
            lane_skipped.extend(engine_skipped)
        lane_results[lane] = {
            "cache_dtype": lane,
            "passed": not lane_failures,
            "engines": engine_results,
            "failures": lane_failures,
            "skipped_gates": lane_skipped,
        }
        failures.extend(lane_failures)
        skipped_gates.extend(lane_skipped)

    reference_prompt_tokens, _ = _reference_counts(reference)
    summary = {
        "schema": SUMMARY_SCHEMA,
        "mode": "verify-reference",
        "passed": not failures,
        "full_matrix": bool(args.full_matrix),
        "engines_selected": [label for label, _, _ in engines_selected],
        "prompt": prompt,
        "tokens": tokens,
        "prompt_tokens": reference_prompt_tokens,
        "model": str(model),
        "model_sha256": model_sha256,
        "model_sha256_expected": reference["model_sha256"],
        "model_hash_match": model_hash_match,
        "allow_model_hash_mismatch": bool(args.allow_model_hash_mismatch),
        "antfly_binary": str(antfly),
        "antfly_binary_sha256": _sha256(antfly),
        "reference_path": str(reference_path),
        "reference_schema": reference["schema"],
        "llama_binary_sha256": reference.get("llama_binary_sha256"),
        "llama_version": reference.get("llama_version"),
        "expected_llama_commit": reference.get("expected_llama_commit"),
        "tokenizer_sha256": reference.get("tokenizer_sha256"),
        "cache_dtype_lanes": lanes,
        "lanes": lane_results,
        "failures": failures,
        "skipped_gates": skipped_gates,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures:
        raise ParityError("; ".join(failures))
    return summary


def _reference_counts(reference: dict[str, Any]) -> tuple[int, int]:
    """Derive (prompt_tokens, output_tokens) from the reference lanes."""
    first_lane = reference["cache_dtype_lanes"][0]
    lane_doc = reference["lanes"][first_lane]
    return len(lane_doc["prompt_token_ids"]), reference["tokens"]


def compare_compact_routes(args: argparse.Namespace) -> dict[str, Any]:
    """Compare streamed and full-resident compact routes without llama.cpp."""
    antfly = _require_file(args.antfly_bin, "Antfly binary", executable=True)
    model = _require_file(args.model, "GGUF model")
    args.out_dir.mkdir(parents=True, exist_ok=False)
    results: dict[str, dict[str, Any]] = {}
    for route, budget in (("off", args.streamed_memory_budget_mb), ("full", args.full_memory_budget_mb)):
        command = _antfly_command(
            antfly,
            model,
            args.prompt,
            args.tokens,
            "metal",
            True,
            args.cache_dtype,
            budget,
            route,
            args.seed,
        )
        log_path = args.out_dir / f"{route}.log"
        output = _run(command, dict(os.environ), log_path, args.timeout)
        prompt_ids, token_ids = parse_antfly_output(output)
        requested, effective = parse_antfly_cache_dtype(output)
        if requested != args.cache_dtype or effective != args.cache_dtype:
            raise ParityError(
                f"{route} cache dtype drifted: requested={requested} effective={effective}"
            )
        if len(token_ids) != args.tokens:
            raise ParityError(
                f"{route} emitted {len(token_ids)} token IDs, expected {args.tokens}: {log_path}"
            )
        results[route] = {
            "command": command,
            "log": str(log_path),
            "memory_budget_mb": budget,
            "prompt_token_ids": prompt_ids,
            "token_ids": token_ids,
            "cache_dtype_requested": requested,
            "cache_dtype_effective": effective,
        }
    failures: list[str] = []
    if results["off"]["prompt_token_ids"] != results["full"]["prompt_token_ids"]:
        failures.append(
            "prompt mismatch: "
            + _first_difference(
                results["off"]["prompt_token_ids"], results["full"]["prompt_token_ids"]
            )
        )
    if results["off"]["token_ids"] != results["full"]["token_ids"]:
        failures.append(
            "output mismatch: "
            + _first_difference(results["off"]["token_ids"], results["full"]["token_ids"])
        )
    summary = {
        "schema": ROUTE_SUMMARY_SCHEMA,
        "passed": not failures,
        "model": str(model),
        "model_sha256": _sha256(model),
        "prompt": args.prompt,
        "tokens": args.tokens,
        "seed": args.seed,
        "cache_dtype": args.cache_dtype,
        "routes": results,
        "failures": failures,
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures:
        raise ParityError("; ".join(failures))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record = subparsers.add_parser(
        "record-reference",
        help="run the pinned llama.cpp oracle per lane and write a reference bundle",
    )
    record.add_argument("--llama-bin", type=Path, required=True)
    record.add_argument("--model", type=Path, required=True)
    record.add_argument("--out-dir", type=Path, required=True)
    record.add_argument("--prompt", default="Hello upon-")
    record.add_argument("--tokens", type=int, default=8)
    record.add_argument("--timeout", type=float, default=300.0)
    record.add_argument("--expected-llama-commit", default=PINNED_LLAMA_COMMIT)
    record.add_argument(
        "--cache-dtype",
        choices=("f16", "f32", "both"),
        default="both",
        help="record explicit, isolated KV-cache lanes (default: both)",
    )
    record.add_argument(
        "--llama-gpu-layers",
        type=int,
        default=0,
        help="offload count for the correctness oracle; CPU avoids whole-model Metal residency",
    )
    record.add_argument(
        "--tokenizer-file",
        type=Path,
        default=None,
        help="optional tokenizer artifact to hash into the reference bundle",
    )

    verify = subparsers.add_parser(
        "verify-reference",
        help="verify Antfly engines against a recorded bundle; never launches llama.cpp",
    )
    verify.add_argument("--antfly-bin", type=Path, required=True)
    verify.add_argument("--model", type=Path, required=True)
    verify.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="reference bundle directory (or its reference.json)",
    )
    verify.add_argument("--out-dir", type=Path, required=True)
    verify.add_argument("--timeout", type=float, default=300.0)
    verify.add_argument("--memory-budget-mb", type=int, default=2048)
    verify.add_argument(
        "--device-routing",
        choices=("off", "partial", "full", "auto"),
        default="off",
    )
    verify.add_argument("--seed", type=int, default=42)
    verify.add_argument(
        "--full-matrix",
        action="store_true",
        help="also run the native and canonical-Metal engines (default: compact only)",
    )
    verify.add_argument(
        "--allow-model-hash-mismatch",
        action="store_true",
        help="downgrade the fail-closed model hash check to a warning",
    )
    verify.add_argument(
        "--llama-bin",
        type=Path,
        default=None,
        help="rejected: verify-reference never launches llama.cpp",
    )
    routes = subparsers.add_parser(
        "compare-compact-routes",
        help="compare exact tokens from streamed/off and full-resident compact routes",
    )
    routes.add_argument("--antfly-bin", type=Path, required=True)
    routes.add_argument("--model", type=Path, required=True)
    routes.add_argument("--out-dir", type=Path, required=True)
    routes.add_argument("--prompt", default="Hello upon-")
    routes.add_argument("--tokens", type=int, default=32)
    routes.add_argument("--timeout", type=float, default=600.0)
    routes.add_argument("--cache-dtype", choices=("f16", "f32"), default="f16")
    routes.add_argument("--streamed-memory-budget-mb", type=int, default=2048)
    routes.add_argument("--full-memory-budget-mb", type=int, default=16384)
    routes.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "record-reference" and args.tokens <= 0:
        print("parity error: --tokens must be positive", file=sys.stderr)
        return 2
    if args.command == "verify-reference" and args.llama_bin is not None:
        print(
            "usage error: verify-reference never launches llama.cpp; record the "
            "oracle with record-reference and drop --llama-bin",
            file=sys.stderr,
        )
        return 2
    if args.command == "verify-reference" and (
        args.memory_budget_mb < 2048 or args.seed < 0
    ):
        print(
            "usage error: --memory-budget-mb must be at least 2048 and --seed must be non-negative",
            file=sys.stderr,
        )
        return 2
    if args.command == "compare-compact-routes" and (
        args.tokens <= 0
        or args.timeout <= 0
        or args.seed < 0
        or args.streamed_memory_budget_mb < 2048
        or args.full_memory_budget_mb < 2048
    ):
        print("usage error: invalid compare-compact-routes limits", file=sys.stderr)
        return 2
    try:
        if args.command == "record-reference":
            reference = record_reference(args)
            prompt_tokens, output_tokens = _reference_counts(reference)
            route_count = sum(
                len(lane_doc["routes"]) for lane_doc in reference["lanes"].values()
            )
            logit_count = sum(
                len(lane_doc["logit_samples"]) for lane_doc in reference["lanes"].values()
            )
            print(
                f"reference recorded: lanes={','.join(reference['cache_dtype_lanes'])} "
                f"prompt_tokens={prompt_tokens} output_tokens={output_tokens} "
                f"routes={route_count} logit_samples={logit_count} "
                f"bundle={args.out_dir}"
            )
        elif args.command == "verify-reference":
            summary = verify_reference(args)
            # Counts derive from the reference lanes carried in the summary; the
            # v2 tool crashed here with a KeyError by assuming a top-level
            # summary["engines"]["llama"] entry that never existed.
            print(
                f"exact parity passed: engines={','.join(summary['engines_selected'])} "
                f"lanes={','.join(summary['cache_dtype_lanes'])} "
                f"prompt_tokens={summary['prompt_tokens']} "
                f"output_tokens={summary['tokens']} "
                f"skipped_gates={len(summary['skipped_gates'])}"
            )
        else:
            summary = compare_compact_routes(args)
            print(
                f"compact route parity passed: prompt_tokens="
                f"{len(summary['routes']['off']['prompt_token_ids'])} "
                f"output_tokens={summary['tokens']} summary={args.out_dir / 'summary.json'}"
            )
    except (ParityError, OSError, subprocess.SubprocessError, ValueError) as exc:
        print(f"parity error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
