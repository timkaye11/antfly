#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fast fake-binary tests for the Gemma4 Metal A/B benchmark contract."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_metal_gemma4_ab import (  # noqa: E402
    BenchmarkContractError,
    _route_expectations,
    build_summary,
)


BENCHMARK = SCRIPT_DIR / "benchmark_metal_gemma4_ab.py"
PROMPT_TOKENS = 20
OUTPUT_TOKENS = 128


def token_digest(count: int) -> str:
    value = " ".join(str(index) for index in range(count))
    return hashlib.sha256(value.encode()).hexdigest()


def executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


FAKE_ANTFLY = r'''#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
tokens = int(args[args.index("--max-tokens") + 1])
json_path = Path(args[args.index("--json-timing") + 1])
label = json_path.stem
variant = os.environ.get("BENCH_VARIANT", "missing")
if variant not in ("baseline", "candidate"):
    raise SystemExit(f"missing variant: {variant}")
if variant == "baseline" and os.environ.get("CANDIDATE_ONLY") is not None:
    raise SystemExit("candidate environment leaked into baseline")
order_log = os.environ.get("ORDER_LOG")
if order_log:
    with Path(order_log).open("a") as stream:
        stream.write(label + "\n")

stage_enabled = os.environ.get("TERMITE_METAL_STAGE_TIMING") == "1"
pair_decode = os.environ.get("TERMITE_METAL_ENABLE_Q4_0_PAIR_ACTIVATION_FUSION") == "1"
pair_prefill = os.environ.get("TERMITE_METAL_ENABLE_Q4_0_PAIR_ACTIVATION_MM") == "1"
concurrent = os.environ.get("TERMITE_METAL_ENABLE_CONCURRENT_PLANNED_DISPATCH") == "1"
if concurrent and os.environ.get("TERMITE_METAL_DISABLE_CONCURRENT_PLANNED_DISPATCH") == "1":
    raise SystemExit("concurrent dispatch simultaneously enabled and disabled")

decode_frames = tokens - 1
attention = 42 * decode_frames
decode_pairs = 42 * decode_frames if pair_decode else 0
prefill_pairs = 42 if pair_prefill else 0
q4_row_one = 210 * decode_frames - 2 * decode_pairs
q4_row_65 = 342 - 2 * prefill_pairs

if stage_enabled:
    prefill_ms, decode_ms = 20000, 30000
elif tokens <= 4:
    prefill_ms = 100 if variant == "baseline" else 99
    decode_ms = 30 if variant == "baseline" else 29
else:
    prefill_ms = 4000 if variant == "baseline" else 3980
    decode_ms = 6000 if variant == "baseline" else 5820
total_ms = prefill_ms + decode_ms

if stage_enabled:
    decode_start = int(os.environ["TERMITE_METAL_STAGE_TIMING_DECODE_START"])
    decode_stride = int(os.environ["TERMITE_METAL_STAGE_TIMING_DECODE_STRIDE"])
    decode_max = int(os.environ["TERMITE_METAL_STAGE_TIMING_DECODE_MAX"])
    decode_sampled = 0
    if decode_frames > decode_start:
        decode_sampled = min(decode_max, ((decode_frames - 1 - decode_start) // decode_stride) + 1)
    prefill_stage = {
        "frames": 1,
        "gpu": 100,
        "attention": 20,
        "ffn": 40,
        "ple": 10,
        "tail": 10,
        "embedding": 10,
        "other": 10,
    }
    decode_stage = {
        "frames": decode_sampled,
        "gpu": 100 * decode_sampled,
        "attention": 20 * decode_sampled,
        "ffn": 40 * decode_sampled,
        "ple": 10 * decode_sampled,
        "tail": 10 * decode_sampled,
        "embedding": 10 * decode_sampled,
        "other": 10 * decode_sampled,
    }
    stage = {
        "scope": "runtime_frame",
        "enabled": 1,
        "supported": 1,
        "complete": 1,
        "samples": 8 * (1 + decode_sampled),
        "failures": 0,
        "prefill": prefill_stage,
        "decode": decode_stage,
    }
else:
    empty = {
        "frames": 0,
        "gpu": 0,
        "attention": 0,
        "ffn": 0,
        "ple": 0,
        "tail": 0,
        "embedding": 0,
        "other": 0,
    }
    stage = {
        "scope": "runtime_frame",
        "enabled": 0,
        "supported": 0,
        "complete": 0,
        "samples": 0,
        "failures": 0,
        "prefill": empty,
        "decode": empty,
    }

q4_variants = {
    "mmv_nr4_nsg2": q4_row_one,
    "mmv_nr8_nsg2": 0,
    "mmv_nr4_nsg4": 0,
    "mmv_nr8_nsg4": 0,
    "mmv_variant_fallbacks": 0,
    "mm_sg_aligned": 342,
    "mm_sg_aligned_tail": 265,
    "mm_sg_unrolled": 0,
}
payload = {
    "backend": "metal",
    "tokens": tokens,
    "token_ids": list(range(tokens)),
    "finish_reason": "length",
    "timing_ms": {
        "generate": total_ms,
        "prefill_inner": prefill_ms,
        "decode_inner": decode_ms,
    },
    "speculative": None,
    "draft_cuda": None,
    "draft_cuda_generate": None,
    "runtime": {"decode_greedy_calls": decode_frames},
    "generation_decoder_runtime": {"forward_attempts": decode_frames},
    "metal": {
        "native_quant_null": False,
        "runtime_command_operators": {"fallback": 0},
        "attention_dispatch": {
            "paged_1x": 0,
            "decode_gqa_split": attention,
            "generated_flash_prefill": 35,
            "generated_flash_prefill_hd512": 7,
            "prefill_direct_kv": 0,
            "prefill_paged_kv": 42,
        },
        "prepared_frame": {"fast_path": decode_frames, "fallback": 0},
        "stage_timing_ns": stage,
        "q4_0_policy": q4_variants,
        "frame_fallbacks": {
            "decode_fallback": 0,
            "prefill_plan_fail": 0,
            "prefill_execute_fail": 0,
        },
        "quant_kernel_plan": {"fast_path_misses": 0, "unsupported_routes": 0},
    },
}
json_path.write_text(json.dumps(payload))

print("generate-setup: live whole-model executor skipped")
print("gen_debug: executePrefill whole-model fast path seq_len=20")
print("prompt_token_ids:", " ".join(str(index) for index in range(20)))
print("token_ids:", " ".join(str(index) for index in range(tokens)))
print(
    f"metal_attention_dispatch: paged_1x=0 decode_gqa_split={attention} "
    "generated_decode_1x=0 generated_flash_prefill=35 "
    "generated_flash_prefill_hd512=7 prefill_direct_kv=0 prefill_paged_kv=42 "
    "generated_rms_norm=0"
)
print(f"metal_prepared_frame: fast_path={decode_frames} fallback=0")
print("metal_runtime_memory: frame_retained_mb=0")
print(
    f"metal_q4_0_dispatch: linear_reduce_rows={q4_row_one}/0/0/{q4_row_65} "
    f"pair_act_reduce={decode_pairs + prefill_pairs}"
)
print(
    f"metal_q4_0_policy: mmv_nr4_nsg2={q4_row_one} mmv_nr8_nsg2=0 "
    "mmv_nr4_nsg4=0 mmv_nr8_nsg4=0 mmv_variant_fallbacks=0 "
    "mm_sg_aligned=342 mm_sg_aligned_tail=265 mm_sg_unrolled=0"
)
print(
    f"metal_q4_0_pair_activation_policy: mmv_nr4_nsg2={decode_pairs} "
    "mmv_nr8_nsg2=0 mmv_nr4_nsg4=0 mmv_nr8_nsg4=0 mmv_variant_fallbacks=0 "
    f"mm_m32_n64_aligned=0 mm_m32_n64_tail={prefill_pairs} "
    "mm_m32_n32_aligned=0 mm_m32_n32_tail=0 mm_variant_fallbacks=0"
)
print(f"metal_q4_q6_k_dispatch: q6_linear_reduce_rows={tokens}/0/0/0")
p = stage["prefill"]
d = stage["decode"]
print(
    f"metal_stage_timing_ns: enabled={stage['enabled']} supported={stage['supported']} "
    f"complete={stage['complete']} scope=runtime_frame prefill_frames={p['frames']} prefill_gpu={p['gpu']} "
    f"prefill_attention={p['attention']} prefill_ffn={p['ffn']} prefill_ple={p['ple']} "
    f"prefill_tail={p['tail']} prefill_embedding={p['embedding']} prefill_other={p['other']} "
    f"decode_frames={d['frames']} decode_gpu={d['gpu']} decode_attention={d['attention']} "
    f"decode_ffn={d['ffn']} decode_ple={d['ple']} decode_tail={d['tail']} "
    f"decode_embedding={d['embedding']} decode_other={d['other']} "
    f"samples={stage['samples']} failures={stage['failures']}"
)
'''


class RouteFormulaTests(unittest.TestCase):
    def test_split_and_pair_formulas_at_gate_shapes(self) -> None:
        for tokens in (4, 128, 300):
            with self.subTest(tokens=tokens):
                split = _route_expectations("split_ffn", tokens)
                self.assertEqual(split["q4_row_one"], 210 * (tokens - 1))
                self.assertEqual(split["decode_pairs"], 0)
                self.assertEqual(split["q4_row_65_plus"], 342)
                paired = _route_expectations("pair_decode_prefill", tokens)
                self.assertEqual(paired["q4_row_one"], 126 * (tokens - 1))
                self.assertEqual(paired["decode_pairs"], 42 * (tokens - 1))
                self.assertEqual(paired["q4_row_65_plus"], 258)
                self.assertEqual(paired["prefill_pairs"], 42)
                self.assertEqual(
                    paired["q4_row_one"] + 2 * paired["decode_pairs"],
                    paired["logical_decode_q4"],
                )
                self.assertEqual(
                    paired["q4_row_65_plus"] + 2 * paired["prefill_pairs"],
                    paired["logical_prefill_q4"],
                )


class HarnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="gemma4-metal-ab-test-")
        self.tmp = Path(self.tempdir.name)
        self.model = self.tmp / "model.gguf"
        self.model.write_bytes(b"fake gguf")
        self.binary = self.tmp / "antfly-inference"
        executable(self.binary, FAKE_ANTFLY)
        self.order_log = self.tmp / "order.log"

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def paired_command(self, out: Path) -> list[str]:
        return [
            sys.executable,
            str(BENCHMARK),
            "run",
            "--out-dir",
            str(out),
            "--experiment-id",
            "fixture-paired",
            "--model",
            str(self.model),
            "--antfly-bin",
            str(self.binary),
            "--prompt",
            "fixture prompt",
            "--expected-prompt-tokens",
            str(PROMPT_TOKENS),
            "--expected-prompt-token-ids-sha256",
            token_digest(PROMPT_TOKENS),
            "--output-tokens",
            str(OUTPUT_TOKENS),
            "--expected-token-ids-sha256",
            token_digest(OUTPUT_TOKENS),
            "--runs",
            "2",
            "--warmups",
            "1",
            "--warmup-output-tokens",
            "4",
            "--expected-warmup-token-ids-sha256",
            token_digest(4),
            "--cooldown-seconds",
            "0",
            "--common-env",
            f"ORDER_LOG={self.order_log}",
            "--baseline-env",
            "BENCH_VARIANT=baseline",
            "--candidate-env",
            "BENCH_VARIANT=candidate",
            "--candidate-env",
            "CANDIDATE_ONLY=1",
            "--max-total-latency-ratio",
            "0.99",
            "--max-prefill-latency-ratio",
            "1.0",
            "--max-decode-latency-ratio",
            "0.98",
            "--min-decode-throughput-ratio",
            "1.02",
            "--min-target-wins",
            "2",
            "--max-cv",
            "0.01",
        ]

    def run_paired(self, out: Path) -> subprocess.CompletedProcess[str]:
        environment = os.environ.copy()
        environment["STAGE_TIMING_RUNS"] = "1"
        environment["CANDIDATE_ONLY"] = "inherited-value"
        completed = subprocess.run(
            self.paired_command(out),
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            logs = "\n".join(
                f"[{path.name}]\n{path.read_text(errors='replace')}"
                for path in sorted(out.glob("*.log"))
            )
            self.fail(
                f"paired harness exited {completed.returncode}:\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}\nlogs:\n{logs}"
            )
        return completed

    def test_full_paired_contract_order_env_isolation_and_stage_exclusion(self) -> None:
        out = self.tmp / "paired"
        completed = self.run_paired(out)
        self.assertIn("passed=True", completed.stdout)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["schema"], "antfly.gemma4_metal_ab.v1")
        self.assertTrue(summary["passed"])
        self.assertEqual(len(summary["performance_samples"]), 4)
        self.assertEqual(len(summary["stage_timing_samples"]), 2)
        self.assertEqual(len(summary["warmup_samples"]), 2)
        self.assertAlmostEqual(summary["metrics"]["candidate_total_ms"]["median"], 9800)
        self.assertAlmostEqual(
            summary["paired_ratios"]["candidate_decode_latency_ratio"]["median"],
            0.97,
        )
        self.assertEqual(summary["target_wins"], 2)
        self.assertTrue(summary["checks"]["stage_timing_excluded_from_performance"])
        self.assertEqual(
            summary["metadata"]["stage_timing_contract"]["sampling"],
            {"prefill_max": 1, "decode_start": 32, "decode_stride": 64, "decode_max": 5},
        )
        self.assertEqual(
            summary["metadata"]["stage_timing_contract"]["scope"], "runtime_frame"
        )
        self.assertIsNone(summary["metadata"]["effective_baseline_env"]["CANDIDATE_ONLY"])
        self.assertEqual(summary["metadata"]["effective_candidate_env"]["CANDIDATE_ONLY"], "1")
        self.assertEqual(
            self.order_log.read_text().splitlines(),
            [
                "warmup-baseline-01",
                "warmup-candidate-01",
                "performance-baseline-01",
                "performance-candidate-01",
                "performance-candidate-02",
                "performance-baseline-02",
                "stage_timing-baseline-01",
                "stage_timing-candidate-01",
            ],
        )

    def test_stage_timing_failure_is_fail_closed(self) -> None:
        out = self.tmp / "bad-stage"
        self.run_paired(out)
        log_path = out / "stage_timing-candidate-01.log"
        log_path.write_text(log_path.read_text().replace("complete=1", "complete=0"))
        json_path = out / "stage_timing-candidate-01.json"
        payload = json.loads(json_path.read_text())
        payload["metal"]["stage_timing_ns"]["complete"] = 0
        json_path.write_text(json.dumps(payload))
        with self.assertRaisesRegex(BenchmarkContractError, "complete=0"):
            build_summary(out)

    def test_pair_decode_and_prefill_route_profile_is_exact(self) -> None:
        out = self.tmp / "pair-routes"
        command = self.paired_command(out)
        command.extend(
            (
                "--candidate-route-profile",
                "pair_decode_prefill",
                "--candidate-env",
                "TERMITE_METAL_ENABLE_Q4_0_PAIR_ACTIVATION_FUSION=1",
                "--candidate-env",
                "TERMITE_METAL_ENABLE_Q4_0_PAIR_ACTIVATION_MM=1",
            )
        )
        environment = os.environ.copy()
        environment["STAGE_TIMING_RUNS"] = "0"
        environment["CANDIDATE_ONLY"] = "inherited-value"
        completed = subprocess.run(
            command,
            env=environment,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn("passed=True", completed.stdout)
        summary = json.loads((out / "summary.json").read_text())
        candidate = next(
            sample
            for sample in summary["performance_samples"]
            if sample["variant"] == "candidate"
        )
        routes = candidate["routes"]
        self.assertEqual(routes["q4_linear_reduce_rows"], [16_002, 0, 0, 258])
        self.assertEqual(routes["q4_pair_activation_decode"], 5_334)
        self.assertEqual(
            routes["q4_pair_activation_policy"]["mmv_nr4_nsg2"], 5_334
        )
        self.assertEqual(
            routes["q4_pair_activation_policy"]["mm_m32_n64_tail"], 42
        )
        self.assertEqual(routes["q4_pair_activation_total"], 5_376)

    def test_pair_prefill_route_profile_counts_the_shared_pair_counter(self) -> None:
        out = self.tmp / "pair-prefill-routes"
        command = self.paired_command(out)
        command.extend(
            (
                "--target-phase",
                "prefill",
                "--candidate-route-profile",
                "pair_prefill",
                "--candidate-env",
                "TERMITE_METAL_ENABLE_Q4_0_PAIR_ACTIVATION_MM=1",
                "--max-prefill-latency-ratio",
                "1.0",
                "--max-decode-latency-ratio",
                "1.0",
                "--min-decode-throughput-ratio",
                "1.0",
            )
        )
        completed = subprocess.run(command, text=True, capture_output=True, check=True)
        self.assertIn("passed=True", completed.stdout)
        summary = json.loads((out / "summary.json").read_text())
        candidate = next(
            sample
            for sample in summary["performance_samples"]
            if sample["variant"] == "candidate"
        )
        routes = candidate["routes"]
        self.assertEqual(routes["q4_pair_activation_decode"], 0)
        self.assertEqual(routes["q4_pair_activation_total"], 42)
        self.assertEqual(routes["q4_pair_activation_policy"]["mm_m32_n64_tail"], 42)

    def test_regression_and_cv_gates_use_only_performance_samples(self) -> None:
        out = self.tmp / "regression"
        self.run_paired(out)
        path = out / "performance-candidate-02.json"
        payload = json.loads(path.read_text())
        payload["timing_ms"].update(
            {"generate": 14000, "prefill_inner": 4000, "decode_inner": 10000}
        )
        path.write_text(json.dumps(payload))
        summary = build_summary(out)
        self.assertFalse(summary["passed"])
        self.assertIn("candidate_decode_ms", summary["cv_violations"])
        self.assertNotEqual(
            summary["paired_ratios"]["candidate_decode_latency_ratio"]["median"],
            30000 / 6000,
        )

    def test_extra_artifact_and_binary_provenance_are_rejected(self) -> None:
        out = self.tmp / "provenance"
        self.run_paired(out)
        (out / "unexpected.json").write_text("{}")
        with self.assertRaisesRegex(BenchmarkContractError, "unexpected A/B JSON"):
            build_summary(out)
        (out / "unexpected.json").unlink()
        self.binary.write_text(self.binary.read_text() + "\n# changed\n")
        with self.assertRaisesRegex(BenchmarkContractError, "antfly_binary_sha256"):
            build_summary(out)

    def test_short_concurrency_determinism_gate(self) -> None:
        out = self.tmp / "determinism"
        command = [
            sys.executable,
            str(BENCHMARK),
            "run",
            "--out-dir",
            str(out),
            "--experiment-id",
            "concurrency-short",
            "--mode",
            "determinism",
            "--model",
            str(self.model),
            "--antfly-bin",
            str(self.binary),
            "--prompt",
            "fixture prompt",
            "--expected-prompt-tokens",
            str(PROMPT_TOKENS),
            "--expected-prompt-token-ids-sha256",
            token_digest(PROMPT_TOKENS),
            "--output-tokens",
            "64",
            "--expected-token-ids-sha256",
            token_digest(64),
            "--runs",
            "3",
            "--warmups",
            "0",
            "--cooldown-seconds",
            "0",
            "--candidate-route-profile",
            "concurrent_split",
            "--candidate-env",
            "BENCH_VARIANT=candidate",
            "--candidate-env",
            "TERMITE_METAL_ENABLE_CONCURRENT_PLANNED_DISPATCH=1",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=True)
        self.assertIn("digests=1 passed=True", completed.stdout)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(len(summary["determinism_samples"]), 3)
        self.assertEqual(summary["unique_output_token_digests"], [token_digest(64)])

        log_path = out / "determinism-candidate-02.log"
        log_path.write_text(log_path.read_text().replace("token_ids: 0 1 2", "token_ids: 9 1 2"))
        with self.assertRaisesRegex(BenchmarkContractError, "output token digest"):
            build_summary(out)

    def test_stage_only_mode_has_no_performance_statistics(self) -> None:
        out = self.tmp / "stage-only"
        command = [
            sys.executable,
            str(BENCHMARK),
            "run",
            "--out-dir",
            str(out),
            "--experiment-id",
            "stage-only",
            "--mode",
            "stage",
            "--model",
            str(self.model),
            "--antfly-bin",
            str(self.binary),
            "--prompt",
            "fixture prompt",
            "--expected-prompt-tokens",
            str(PROMPT_TOKENS),
            "--expected-prompt-token-ids-sha256",
            token_digest(PROMPT_TOKENS),
            "--output-tokens",
            str(OUTPUT_TOKENS),
            "--expected-token-ids-sha256",
            token_digest(OUTPUT_TOKENS),
            "--warmups",
            "0",
            "--stage-timing-runs",
            "1",
            "--cooldown-seconds",
            "0",
            "--candidate-env",
            "BENCH_VARIANT=candidate",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=True)
        self.assertIn("runs=1 passed=True", completed.stdout)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["mode"], "stage")
        self.assertEqual(summary["performance_samples"], [])
        self.assertEqual(len(summary["stage_timing_samples"]), 1)
        self.assertTrue(summary["checks"]["no_performance_samples"])


if __name__ == "__main__":
    unittest.main()
