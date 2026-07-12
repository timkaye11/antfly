#!/usr/bin/env python3
"""Collect reproducible Gemma 4 CUDA L4 release evidence.

The production target deliberately keeps generated attention, generated E2B
Q8-intermediate and exact-F32 FFN, and generated Q6_K LM-head candidates
disabled. This command proves the fixed E2B QAT decode contract against
llama.cpp and records a separate 12B Q4_K_M deterministic/replay regression
without changing that policy.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
from typing import Any


RELEASE_SCHEMA = "antfly.gemma4_cuda_l4_release_gate.v1"
E2B_PROMPT = "Here is a sentence about ants:"
E2B_ANTFLY_TOKENS = 255
E2B_LLAMA_TOKENS = 256
GEMMA12B_PROMPT = "Write one sentence about ants."
GEMMA12B_TOKENS = 256
CAPTURE_KV_CAPACITY = 544
MAX_TOK_S_CV = 0.02
DEFAULT_MIN_COMPARABLE_RATIO = 0.80
TOKEN_IDS_RE = re.compile(r"^token_ids:(?P<ids>(?:\s+-?\d+)*)\s*$", re.MULTILINE)

# These values are applied after the caller environment.  The lower-case
# aliases matter because the shared tuning shell function accepts both forms.
FROZEN_PROFILE = {
    "ANTFLY_PREFILL_CHUNK_SIZE": "32",
    "ANTFLY_CACHE_DTYPE": "f32",
    "LLAMA_CACHE_TYPE_K": "f32",
    "LLAMA_CACHE_TYPE_V": "f32",
    "ANTFLY_CAPTURE_FORCE_KV_CAPACITY": str(CAPTURE_KV_CAPACITY),
    "ANTFLY_DECODE_GRAPH_REPLAY": "required",
    "ANTFLY_SERVER_DECODE_GRAPH_REPLAY": "required",
    "antfly_decode_graph_replay": "required",
    "ANTFLY_SERVER_DISABLE_CONTINUOUS_BATCHING": "1",
    "ANTFLY_INFERENCE_DISABLE_CONTINUOUS_BATCHING": "1",
    "ANTFLY_INFERENCE_CUDA_TEMP_SLOT_PERIOD": "853",
    "ANTFLY_CUDA_TEMP_SLOT_PERIOD": "853",
    "antfly_cuda_temp_slot_period": "853",
    "ANTFLY_INFERENCE_CUDA_TEMP_SLOT_SKIP": "2500",
    "ANTFLY_CUDA_TEMP_SLOT_SKIP": "2500",
    "antfly_cuda_temp_slot_skip": "2500",
    "ANTFLY_GENERATED_ATTENTION_DECODE": "0",
    "ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_DECODE": "0",
    "antfly_generated_attention_decode": "0",
    "ANTFLY_GENERATED_ATTENTION_SCORE_PREWORK": "0",
    "ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_SCORE_PREWORK": "0",
    "antfly_generated_attention_score_prework": "0",
    "ANTFLY_TURBOQUANT_SPLIT_ATTENTION": "0",
    "ANTFLY_INFERENCE_CUDA_TURBOQUANT_SPLIT_ATTENTION": "0",
    "antfly_turboquant_split_attention": "0",
    "ANTFLY_INFERENCE_CUDA_GENERATED_Q6_K_Q8_1_LM_HEAD_ARGMAX": "0",
    "antfly_generated_q6_k_q8_1_lm_head_argmax": "0",
    "ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN": "0",
    "ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_CATALOG_FFN_CANDIDATES": "0",
    "antfly_generated_q4_0_e2b_ffn": "0",
    "ANTFLY_INFERENCE_CUDA_GENERATED_Q4_0_E2B_FFN_EXACT": "0",
    "antfly_generated_q4_0_e2b_ffn_exact": "0",
    "ANTFLY_INFERENCE_CUDA_Q4_0_LM_HEAD_Q8_1_ARGMAX": "1",
    "antfly_q4_0_q8_1_lm_head_argmax": "1",
    "REQUIRE_GRAPH_REPLAY": "1",
    "REQUIRE_GENERATED_ATTENTION": "0",
    "REQUIRE_LM_HEAD_ARGMAX": "1",
    "REQUIRE_GENERATED_Q6_LM_HEAD_ARGMAX": "0",
    "REQUIRE_GENERATED_E2B_FFN": "0",
    "ANTFLY_Q4_0_Q8_1_PREFILL_ROWS": "1",
    "ANTFLY_Q4_0_GATE_UP_ACTIVATION_Q8_1_PRECOMPUTE": "0",
    "ANTFLY_GQA_PREFILL_FAST": "1",
    "ANTFLY_GQA_PREFILL_TILED": "0",
    "ANTFLY_GQA_PREFILL_MMA": "0",
    "ANTFLY_Q4_0_LINEAR_Q8_1_TILE4_W8_MIN_IN_DIM": "2048",
    "ANTFLY_Q4_0_LINEAR_Q8_1_ROWS8_C4": "1",
    "ANTFLY_Q4_0_PAIR_ACTIVATION_Q8_1_ROWS8_C2": "1",
    "ANTFLY_Q4_0_PAIR_ACTIVATION_Q8_1_ROWS16_C1": "1",
    "ANTFLY_CUDA_GEMMA_PREFILL_PREWARM": "1",
    "ANTFLY_CUDA_PREFILL_FIRST_TOKEN": "1",
    "ANTFLY_CUDA_PREFILL_FIRST_TOKEN_COALESCE_TOKENS": "2048",
    "ANTFLY_CUDA_PROFILE_PREFILL_OPS": "0",
    "ANTFLY_CUDA_PROFILE_DECODE": "0",
    "ANTFLY_RMS_NORM_BF16_MIRROR": "0",
    "ANTFLY_BF16_RESIDENT_WEIGHTS": "0",
    "ANTFLY_HYBRID_BF16_PREFILL": "0",
    "ANTFLY_PLE_MODEL_PROJ_BF16": "0",
    "ANTFLY_INFERENCE_CUDA_ASYNC_I32_DOWNLOAD_STAGING": "1",
    "ANTFLY_INFERENCE_CUDA_GREEDY_PENDING_TOKEN_READBACK": "1",
}


def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    repo = repo_root()
    inference = repo / "zig/pkg/inference"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=pathlib.Path, default=inference / "zig-out/bin/antfly-inference")
    parser.add_argument("--llama-cpp-bin", type=pathlib.Path, default=pathlib.Path("/tmp/llama.cpp/build/bin/llama-completion"))
    parser.add_argument(
        "--e2b-model",
        type=pathlib.Path,
        default=repo / ".models/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf",
    )
    parser.add_argument(
        "--gemma12b-q4-model",
        type=pathlib.Path,
        default=repo / ".models/google/gemma-4-12B-it-q4_k/gemma-4-12B-it-Q4_K_M.gguf",
    )
    parser.add_argument("--wrapper", type=pathlib.Path, default=inference / "scripts/with_gemma4_qat_cuda_tuning.sh")
    parser.add_argument("--matrix-script", type=pathlib.Path, default=inference / "scripts/benchmark_gemma4_cuda_matrix.py")
    parser.add_argument("--artifact-check-script", type=pathlib.Path, default=inference / "scripts/regen-cuda-artifacts.sh")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("/tmp/antfly-gemma4-cuda-l4-release"))
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout-sec", type=int, default=720)
    parser.add_argument(
        "--enforce-performance",
        action="store_true",
        help="fail unless the fixed E2B benchmark reaches --min-comparable-ratio",
    )
    parser.add_argument("--min-comparable-ratio", type=float, default=DEFAULT_MIN_COMPARABLE_RATIO)
    parser.add_argument(
        "--verify-artifacts",
        action="store_true",
        help="run the non-mutating CUDA artifact freshness check before benchmarking",
    )
    parser.add_argument(
        "--skip-12b",
        action="store_true",
        help="diagnostic-only: skip the required 12B Q4_K_M deterministic/replay evidence",
    )
    parser.add_argument(
        "--no-require-l4",
        dest="require_l4",
        action="store_false",
        help="diagnostic-only: record GPU identity without requiring NVIDIA L4 / SM89",
    )
    parser.set_defaults(require_l4=True)
    return parser.parse_args()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def path_provenance(path: pathlib.Path) -> dict[str, Any]:
    resolved = path.resolve()
    result: dict[str, Any] = {"path": str(resolved), "exists": path.exists()}
    if not path.exists():
        return result
    if path.is_file():
        result.update({"kind": "file", "bytes": path.stat().st_size, "sha256": sha256_file(path)})
        return result
    if not path.is_dir():
        result["kind"] = "other"
        return result

    digest = hashlib.sha256()
    total_bytes = 0
    count = 0
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        child_digest = sha256_file(child)
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(child_digest.encode("ascii"))
        digest.update(b"\n")
        total_bytes += child.stat().st_size
        count += 1
    result.update({
        "kind": "directory",
        "file_count": count,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    })
    return result


def command_output(*command: str) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def git_provenance(repo: pathlib.Path) -> dict[str, Any]:
    status = command_output("git", "-C", str(repo), "status", "--porcelain=v1")
    return {
        "commit": command_output("git", "-C", str(repo), "rev-parse", "HEAD"),
        "describe": command_output("git", "-C", str(repo), "describe", "--always", "--dirty", "--long"),
        "dirty": bool(status),
        "status_sha256": hashlib.sha256((status or "").encode("utf-8")).hexdigest(),
    }


def gpu_provenance() -> dict[str, Any]:
    output = command_output(
        "nvidia-smi",
        "--query-gpu=index,name,driver_version,compute_cap",
        "--format=csv,noheader",
    )
    all_devices = []
    if output:
        for line in output.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) >= 4:
                all_devices.append({
                    "index": fields[0],
                    "name": fields[1],
                    "driver_version": fields[2],
                    "compute_capability": fields[3],
                })
            else:
                all_devices.append({"raw": line})
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    requested_indexes = [item.strip() for item in visible.split(",")] if visible else []
    if requested_indexes and all(item.isdecimal() for item in requested_indexes):
        devices = [device for device in all_devices if device.get("index") in requested_indexes]
    else:
        devices = all_devices
    return {
        "query": output,
        "cuda_visible_devices": visible,
        "all_devices": all_devices,
        "devices": devices,
    }


def l4_errors(gpu: dict[str, Any]) -> list[str]:
    devices = gpu.get("devices")
    if not isinstance(devices, list) or len(devices) != 1:
        return ["expected exactly one NVIDIA L4 / SM89 device"]
    device = devices[0]
    if device.get("name") != "NVIDIA L4":
        return [f"expected NVIDIA L4, got {device.get('name')!r}"]
    if device.get("compute_capability") not in {"8.9", "8.9 "}:
        return [f"expected compute capability 8.9, got {device.get('compute_capability')!r}"]
    return []


def diagnostic_mode_errors(require_l4: bool, skip_12b: bool) -> list[str]:
    errors = []
    if not require_l4:
        errors.append("--no-require-l4 is diagnostic-only; release evidence requires NVIDIA L4 / SM89")
    if skip_12b:
        errors.append("--skip-12b is diagnostic-only; release evidence requires 12B Q4_K_M replay proof")
    return errors


def artifact_provenance(repo: pathlib.Path) -> dict[str, dict[str, Any]]:
    root = repo / "zig/pkg/inference/src/ops/cuda"
    paths = {
        "generated_manifest": root / "generated/quant_kernel_artifacts.json",
        "runtime_bundle_source": root / "artifacts/inference_cuda_kernels.cu",
        "runtime_ptx": root / "artifacts/inference_cuda_kernels.ptx",
        "runtime_fatbin": root / "artifacts/inference_cuda_kernels.fatbin",
        "runtime_sm89_cubin": root / "artifacts/inference_cuda_kernels_sm89.cubin",
    }
    return {name: path_provenance(path) for name, path in paths.items()}


def frozen_profile() -> dict[str, str]:
    """Return a new profile mapping so callers cannot mutate the contract."""
    return dict(FROZEN_PROFILE)


def release_environment(args: argparse.Namespace) -> dict[str, str]:
    environment = os.environ.copy()
    # A release result must not inherit an accidental experiment switch.  Keep
    # CUDA_VISIBLE_DEVICES and ordinary process settings, but replace all
    # benchmark/tuning knobs with the profile below.
    controlled_prefixes = ("ANTFLY_", "antfly_", "LLAMA_", "REQUIRE_", "MIN_", "MAX_")
    controlled_names = {"MODEL", "PROMPT", "TIMEOUT", "WARMUPS", "REPEATS"}
    for name in tuple(environment):
        if name.startswith(controlled_prefixes) or name in controlled_names:
            environment.pop(name)
    environment.update(frozen_profile())
    environment.update({
        "ANTFLY_BIN": str(args.binary.resolve()),
        "LLAMA_CPP_BIN": str(args.llama_cpp_bin.resolve()),
        "MODEL": str(args.e2b_model.resolve()),
        "PROMPT": E2B_PROMPT,
        "ANTFLY_TOKENS": str(E2B_ANTFLY_TOKENS),
        "LLAMA_TOKENS": str(E2B_LLAMA_TOKENS),
        "MIN_LLAMA_THROUGHPUT_RATIO": "0",
        "MIN_COMPARABLE_THROUGHPUT_RATIO": "0",
        "MIN_ANTFLY_TOK_S": "0",
        "MAX_ANTFLY_TOK_S_CV": str(MAX_TOK_S_CV),
        "TIMEOUT": str(args.timeout_sec),
    })
    return environment


def matrix_command(args: argparse.Namespace) -> list[str]:
    minimum_ratio = args.min_comparable_ratio if args.enforce_performance else 0.0
    return [
        sys.executable,
        str(args.matrix_script),
        "--output-dir", str(args.output_dir / "e2b"),
        "--prompt", E2B_PROMPT,
        "--lengths", str(E2B_LLAMA_TOKENS),
        "--target-length", str(E2B_LLAMA_TOKENS),
        "--warmups", str(args.warmups),
        "--repeats", str(args.repeats),
        "--min-antfly-tok-s", "0",
        "--min-comparable-ratio", str(minimum_ratio),
        "--max-cv", str(MAX_TOK_S_CV),
        "--require-graph-replay",
        "--no-require-generated-attention",
        "--no-require-generated-q6-lm-head-argmax",
        "--collect-only",
    ]


def reset_matrix_outputs(output_dir: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    matrix_path = output_dir / "e2b/matrix_summary.json"
    pair_path = output_dir / f"e2b/tokens-{E2B_LLAMA_TOKENS}/paired_summary.json"
    matrix_path.unlink(missing_ok=True)
    pair_path.unlink(missing_ok=True)
    return matrix_path, pair_path


def matrix_process_errors(returncode: int) -> list[str]:
    return [] if returncode == 0 else [f"E2B matrix exited {returncode}"]


def write_json(path: pathlib.Path, value: object) -> None:
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_logged(command: list[str], environment: dict[str, str], log_path: pathlib.Path, timeout_sec: int) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=repo_root(),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_sec,
            check=False,
        )
        output = completed.stdout
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        partial = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        output = partial + f"\ncommand timed out after {timeout_sec}s\n"
        returncode = 124
    except OSError as exc:
        output = f"could not start command: {exc}\n"
        returncode = 127
    log_path.write_text(output, encoding="utf-8")
    return returncode, output


def int_value(value: object) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def float_value(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) else 0.0


def e2b_contract_errors(matrix: dict[str, Any], enforce_performance: bool, minimum_ratio: float) -> list[str]:
    errors: list[str] = []
    entries = matrix.get("entries")
    if not isinstance(entries, list) or len(entries) != 1:
        return ["expected exactly one fixed 256-token E2B matrix entry"]
    entry = entries[0]
    if int_value(entry.get("output_tokens")) != E2B_LLAMA_TOKENS:
        errors.append(f"expected {E2B_LLAMA_TOKENS}-token E2B matrix entry")
    if not bool(entry.get("pair_ok")):
        errors.append("fixed E2B paired benchmark did not pass its replay/route contract")
    if not bool(entry.get("graph_replay_ok")):
        errors.append("fixed E2B paired benchmark did not retain persistent graph replay")
    ratio = float_value(entry.get("comparable_ratio"))
    if ratio <= 0.0:
        errors.append("fixed E2B benchmark reported non-positive comparable throughput ratio")
    if enforce_performance and ratio < minimum_ratio:
        errors.append(f"E2B comparable throughput ratio {ratio:.3f} is below required {minimum_ratio:.3f}")
    if enforce_performance and not bool(matrix.get("passed")):
        errors.append("E2B performance matrix did not pass its configured release threshold")
    return errors


def e2b_pair_contract_errors(pair: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    comparison = pair.get("comparison") if isinstance(pair.get("comparison"), dict) else {}
    expected = {
        "antfly_tokens": E2B_ANTFLY_TOKENS,
        "llama_tokens": E2B_LLAMA_TOKENS,
        "antfly_cache_dtype": "f32",
        "llama_cache_type_k": "f32",
        "llama_cache_type_v": "f32",
        "antfly_generated_attention_decode": "0",
        "antfly_generated_q6_k_q8_1_lm_head_argmax": "0",
        "antfly_generated_q4_0_e2b_ffn": "0",
        "antfly_generated_q4_0_e2b_ffn_exact": "0",
        "antfly_q4_0_q8_1_lm_head_argmax": "1",
    }
    for key, value in expected.items():
        if str(comparison.get(key)) != str(value):
            errors.append(f"E2B fixed profile drifted: {key}={comparison.get(key)!r}, expected {value!r}")
    if not bool(pair.get("ok")):
        errors.append("E2B paired summary did not pass")
    if not bool(pair.get("ok_graph_replay")):
        errors.append("E2B paired summary did not pass persistent graph replay")
    if not bool(pair.get("ok_lm_head_argmax")):
        errors.append("E2B paired summary did not pass the production Q4_0 LM-head route")
    rows = pair.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("E2B paired summary has no measured rows")
        return errors
    for index, row in enumerate(rows, start=1):
        if int_value(row.get("antfly_generated_q6_lm_head_argmax")) != 0:
            errors.append(f"E2B sample {index} unexpectedly used generated Q6 LM-head")
        if int_value(row.get("antfly_generated_q6_lm_head_argmax_fallbacks")) != 0:
            errors.append(f"E2B sample {index} reported generated Q6 LM-head fallbacks")
        for key in (
            "antfly_generated_e2b_pair",
            "antfly_generated_e2b_down",
            "antfly_generated_e2b_pair_fallbacks",
            "antfly_generated_e2b_down_fallbacks",
            "antfly_generated_e2b_exact_pair",
            "antfly_generated_e2b_exact_down",
            "antfly_generated_e2b_exact_pair_fallbacks",
            "antfly_generated_e2b_exact_down_fallbacks",
            "antfly_generated_attention",
        ):
            if int_value(row.get(key)) != 0:
                errors.append(f"E2B sample {index} unexpectedly used disabled candidate counter {key}")
    return errors


def generation_command(args: argparse.Namespace, model: pathlib.Path, prompt: str, tokens: int, timing_path: pathlib.Path) -> list[str]:
    return [
        str(args.wrapper),
        str(args.binary),
        "generate",
        str(model),
        prompt,
        "--backend", "cuda",
        "--combined-budget-mb", "22000",
        "--backend-budget-mb", "19000",
        "--kv-budget-mb", "1024",
        "--scratch-budget-mb", "2048",
        "--prefill-chunk-size", "32",
        "--max-tokens", str(tokens),
        "--temperature", "0",
        "--raw-prompt",
        "--no-chat-template",
        "--ignore-eos",
        "--cache-dtype", "f32",
        "--print-token-count",
        "--print-token-ids",
        "--print-timing",
        "--json-timing", str(timing_path),
    ]


def parse_token_ids(output: str) -> list[int]:
    match = TOKEN_IDS_RE.search(output)
    return [int(value) for value in match.group("ids").split()] if match else []


def run_generation_case(
    args: argparse.Namespace,
    environment: dict[str, str],
    model: pathlib.Path,
    prompt: str,
    tokens: int,
    output_dir: pathlib.Path,
    stem: str,
) -> dict[str, Any]:
    timing_path = output_dir / f"{stem}.json"
    log_path = output_dir / f"{stem}.log"
    command = generation_command(args, model, prompt, tokens, timing_path)
    returncode, output = run_logged(command, environment, log_path, args.timeout_sec)
    result: dict[str, Any] = {
        "command": command,
        "log": str(log_path),
        "timing": str(timing_path),
        "returncode": returncode,
        "token_ids": parse_token_ids(output),
    }
    if timing_path.is_file():
        try:
            result["timing_data"] = json.loads(timing_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            result["timing_error"] = f"invalid timing JSON: {exc}"
    else:
        result["timing_error"] = "command did not write timing JSON"
    return result


def disabled_candidate_errors(timing: dict[str, Any], label: str) -> list[str]:
    cuda = timing.get("cuda") if isinstance(timing.get("cuda"), dict) else {}
    counters = (
        "launch_attention_gqa_decode_generated",
        "launch_attention_gqa_decode_score_prework",
        "lm_head_argmax_generated_q6_k_q8_1_hits",
        "lm_head_argmax_generated_q6_k_q8_1_fallbacks",
        "q4_0_generated_e2b_pair_q8_hits",
        "q4_0_generated_e2b_down_q8_hits",
        "q4_0_generated_e2b_pair_q8_fallbacks",
        "q4_0_generated_e2b_down_q8_fallbacks",
        "q4_0_generated_e2b_exact_pair_f32_hits",
        "q4_0_generated_e2b_exact_down_f32_hits",
        "q4_0_generated_e2b_exact_pair_f32_fallbacks",
        "q4_0_generated_e2b_exact_down_f32_fallbacks",
    )
    return [f"{label} unexpectedly used disabled candidate counter {counter}" for counter in counters if int_value(cuda.get(counter)) != 0]


def generation_replay_errors(run: dict[str, Any], tokens: int, label: str) -> list[str]:
    errors: list[str] = []
    if int_value(run.get("returncode")) != 0:
        errors.append(f"{label} command exited {run.get('returncode')}")
    timing = run.get("timing_data")
    if not isinstance(timing, dict):
        errors.append(f"{label} has no readable timing JSON")
        return errors
    ids = run.get("token_ids")
    if not isinstance(ids, list) or len(ids) != tokens:
        errors.append(f"{label} emitted {len(ids) if isinstance(ids, list) else 0} token IDs, expected {tokens}")
    if int_value(timing.get("tokens")) != tokens:
        errors.append(f"{label} timing reported {timing.get('tokens')!r} tokens, expected {tokens}")
    if float_value(timing.get("decode_tok_per_s")) <= 0.0:
        errors.append(f"{label} reported non-positive decode throughput")
    cuda = timing.get("cuda") if isinstance(timing.get("cuda"), dict) else {}
    minimum_replays = max(1, tokens - 8)
    if int_value(cuda.get("graph_capture_persistent_replays")) < minimum_replays:
        errors.append(f"{label} persistent graph replays below {minimum_replays}")
    if int_value(cuda.get("graph_capture_discards")) != 0:
        errors.append(f"{label} reported graph capture discards")
    if int_value(cuda.get("graph_capture_capacity_skips")) != 0:
        errors.append(f"{label} reported graph capture capacity skips")
    errors.extend(disabled_candidate_errors(timing, label))
    return errors


def gemma12b_evidence(
    args: argparse.Namespace,
    environment: dict[str, str],
    model_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    output_dir = args.output_dir / "gemma12b"
    output_dir.mkdir(parents=True, exist_ok=True)
    first = run_generation_case(
        args, environment, args.gemma12b_q4_model, GEMMA12B_PROMPT, GEMMA12B_TOKENS, output_dir, "run-1"
    )
    second = run_generation_case(
        args, environment, args.gemma12b_q4_model, GEMMA12B_PROMPT, GEMMA12B_TOKENS, output_dir, "run-2"
    )
    replay_errors = generation_replay_errors(first, GEMMA12B_TOKENS, "12B run 1")
    replay_errors.extend(generation_replay_errors(second, GEMMA12B_TOKENS, "12B run 2"))
    errors = list(replay_errors)
    first_ids = first.get("token_ids") if isinstance(first.get("token_ids"), list) else []
    second_ids = second.get("token_ids") if isinstance(second.get("token_ids"), list) else []
    if first_ids != second_ids:
        mismatch = next(
            (index for index, pair in enumerate(zip(first_ids, second_ids)) if pair[0] != pair[1]),
            min(len(first_ids), len(second_ids)),
        )
        errors.append(f"12B deterministic token IDs differ at index {mismatch}")
    token_sha256 = canonical_sha256(first_ids) if first_ids else None
    return {
        "model": model_provenance or path_provenance(args.gemma12b_q4_model),
        "prompt": GEMMA12B_PROMPT,
        "tokens": GEMMA12B_TOKENS,
        "runs": [first, second],
        "token_ids_equal": first_ids == second_ids,
        "token_ids_sha256": token_sha256,
        "checks": {
            "deterministic_tokens": first_ids == second_ids and len(first_ids) == GEMMA12B_TOKENS,
            "replay_and_routes": not replay_errors,
        },
        "errors": errors,
        "passed": not errors,
    }


def artifact_check(args: argparse.Namespace, environment: dict[str, str]) -> dict[str, Any]:
    log_path = args.output_dir / "artifact_check.log"
    command = [str(args.artifact_check_script), "--check", "--all"]
    returncode, _ = run_logged(command, environment, log_path, args.timeout_sec)
    return {"command": command, "log": str(log_path), "returncode": returncode, "passed": returncode == 0}


def require_paths(args: argparse.Namespace) -> list[str]:
    paths = {
        "Antfly binary": args.binary,
        "llama.cpp binary": args.llama_cpp_bin,
        "E2B QAT GGUF": args.e2b_model,
        "tuning wrapper": args.wrapper,
        "CUDA matrix script": args.matrix_script,
    }
    if args.verify_artifacts:
        paths["CUDA artifact check script"] = args.artifact_check_script
    if not args.skip_12b:
        paths["12B Q4_K_M GGUF"] = args.gemma12b_q4_model
    return [f"missing {label}: {path}" for label, path in paths.items() if not path.exists()]


def main() -> None:
    args = parse_args()
    if args.warmups < 0 or args.repeats < 1 or args.timeout_sec < 1:
        raise SystemExit("warmups must be non-negative; repeats and timeout-sec must be positive")
    if not math.isfinite(args.min_comparable_ratio) or args.min_comparable_ratio <= 0.0:
        raise SystemExit("min-comparable-ratio must be a finite positive number")

    missing = require_paths(args)
    if missing:
        raise SystemExit("\n".join(missing))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profile = frozen_profile()
    environment = release_environment(args)
    gpu = gpu_provenance()
    artifacts = artifact_provenance(repo_root())
    gemma12b_model_provenance = path_provenance(args.gemma12b_q4_model)
    provenance = {
        "schema": RELEASE_SCHEMA,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git": git_provenance(repo_root()),
        "gpu": gpu,
        "binary": path_provenance(args.binary),
        "llama_cpp_binary": path_provenance(args.llama_cpp_bin),
        "e2b_model": path_provenance(args.e2b_model),
        "gemma12b_q4_model": gemma12b_model_provenance,
        "artifacts": artifacts,
        "frozen_profile": profile,
        "frozen_profile_sha256": canonical_sha256(profile),
        "benchmark_contract": {
            "prompt": E2B_PROMPT,
            "antfly_tokens": E2B_ANTFLY_TOKENS,
            "llama_tokens": E2B_LLAMA_TOKENS,
            "cache_dtype": "f32",
            "graph_replay": "required",
            "max_tok_s_cv": MAX_TOK_S_CV,
            "continuous_batching": "disabled",
        },
    }
    write_json(args.output_dir / "release_provenance.json", provenance)

    errors = diagnostic_mode_errors(args.require_l4, args.skip_12b)
    if args.require_l4:
        errors.extend(l4_errors(gpu))
    missing_artifacts = [name for name, value in artifacts.items() if not value.get("exists")]
    if missing_artifacts:
        errors.append(f"missing CUDA provenance artifacts: {', '.join(missing_artifacts)}")

    artifact_result: dict[str, Any] | None = None
    if args.verify_artifacts:
        artifact_result = artifact_check(args, environment)
        if not artifact_result["passed"]:
            errors.append("CUDA artifact freshness check failed")

    matrix_path, pair_path = reset_matrix_outputs(args.output_dir)
    matrix_log = args.output_dir / "e2b_matrix.log"
    matrix_returncode, _ = run_logged(matrix_command(args), environment, matrix_log, args.timeout_sec)
    errors.extend(matrix_process_errors(matrix_returncode))
    matrix: dict[str, Any] = {}
    if matrix_path.is_file():
        try:
            matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"could not parse E2B matrix summary: {exc}")
    else:
        errors.append("E2B matrix did not write matrix_summary.json")
    if matrix:
        errors.extend(e2b_contract_errors(matrix, args.enforce_performance, args.min_comparable_ratio))

    pair: dict[str, Any] = {}
    if pair_path.is_file():
        try:
            pair = json.loads(pair_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            errors.append(f"could not parse E2B paired summary: {exc}")
    else:
        errors.append("E2B matrix did not write paired_summary.json")
    if pair:
        errors.extend(e2b_pair_contract_errors(pair))

    gemma12b: dict[str, Any]
    if args.skip_12b:
        gemma12b = {"skipped": True, "passed": False, "reason": "--skip-12b"}
    else:
        gemma12b = gemma12b_evidence(args, environment, gemma12b_model_provenance)
        errors.extend(gemma12b["errors"])

    result = {
        "schema": RELEASE_SCHEMA,
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "config": {
            "enforce_performance": args.enforce_performance,
            "min_comparable_ratio": args.min_comparable_ratio,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "require_l4": args.require_l4,
            "verify_artifacts": args.verify_artifacts,
            "skip_12b": args.skip_12b,
        },
        "provenance": "release_provenance.json",
        "artifact_check": artifact_result,
        "e2b": {
            "matrix_log": str(matrix_log),
            "matrix_exit_code": matrix_returncode,
            "matrix_summary": str(matrix_path),
            "paired_summary": str(pair_path),
            "target": (matrix.get("entries") or [None])[0] if matrix else None,
        },
        "gemma12b": gemma12b,
        "errors": errors,
        "passed": not errors,
    }
    summary_path = args.output_dir / "release_summary.json"
    write_json(summary_path, result)
    target = result["e2b"]["target"] or {}
    print(
        "gemma4_cuda_l4_release_gate "
        f"comparable_ratio={float_value(target.get('comparable_ratio')):.3f} "
        f"enforced={str(args.enforce_performance).lower()} "
        f"passed={str(not errors).lower()} output={summary_path}"
    )
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
