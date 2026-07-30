#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Fast contract tests for the paired Gemma4 long-output benchmark."""

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

from gemma4_metal_long_output import (  # noqa: E402
    BenchmarkContractError,
    build_result,
    gate_errors,
    parse_llama_metal_runtime,
    parse_llama_timing,
    ratio_stats,
)


BENCHMARK = SCRIPT_DIR / "benchmark_metal_gemma4_long_output.sh"
OUTPUT_TOKENS = 300
TOKEN_IDS = " ".join(str(index) for index in range(OUTPUT_TOKENS))
TOKEN_IDS_SHA256 = hashlib.sha256(TOKEN_IDS.encode()).hexdigest()


def executable(path: Path, source: str) -> None:
    path.write_text(source)
    path.chmod(0o755)


def b10182_perf_block(tokens: int = OUTPUT_TOKENS) -> str:
    return "\n".join(
        (
            "0.17.923.567 I common_perf_print: sampling time = 50.00 ms / 300 runs",
            "0.17.923.567 I common_perf_print: prompt eval time = 4000.00 ms / 2003 tokens",
            f"0.17.923.567 I common_perf_print: eval time = 5600.00 ms / {tokens - 1} runs",
            f"0.17.923.567 I common_perf_print: total time = 9655.00 ms / {2003 + tokens - 1} tokens",
            "0.17.923.567 I common_perf_print: unaccounted time = 5.00 ms / 0.1 %",
            f"0.17.923.567 I common_perf_print: graphs reused = {tokens - 3}",
        )
    )


class ParserContractTests(unittest.TestCase):
    def test_b10182_timestamp_prefix_is_parsed(self) -> None:
        timing = parse_llama_timing(b10182_perf_block())
        self.assertEqual(timing.logger, "common_perf_print")
        self.assertEqual(timing.prompt_tokens, 2003)
        self.assertEqual(timing.eval_runs, 299)
        self.assertEqual(timing.total_tokens, 2302)
        self.assertEqual(timing.graphs_reused, 297)
        self.assertAlmostEqual(timing.total_ms, 9655.0)

    def test_b10182_metal_backend_markers_are_parsed(self) -> None:
        runtime = parse_llama_metal_runtime(
            "\n".join(
                (
                    "llama_prepare_model_devices: using device MTL0 (Apple M4)",
                    "load_tensors: offloaded 43/43 layers to GPU",
                    "ggml_metal_init: found device: Apple M4",
                    "ggml_metal_init: picking default device: Apple M4",
                )
            )
        )
        self.assertEqual(runtime.default_device, "Apple M4")
        self.assertEqual(runtime.offloaded_layers, 43)
        self.assertEqual(runtime.total_layers, 43)

    def test_timing_block_must_be_unique_and_complete(self) -> None:
        block = b10182_perf_block()
        with self.assertRaisesRegex(BenchmarkContractError, "exactly one"):
            parse_llama_timing(block + "\n" + block)
        with self.assertRaisesRegex(BenchmarkContractError, "exactly one"):
            parse_llama_timing("\n".join(block.splitlines()[:-4]))

    def test_ratios_are_summarized_pairwise(self) -> None:
        rows = [
            {"ratio": 100.0},
            {"ratio": 0.1},
            {"ratio": 10.0},
        ]
        self.assertEqual(ratio_stats(rows, "ratio")["median"], 10.0)

    def test_phase_cv_violations_are_gated(self) -> None:
        result = {
            "cv_gate": {"violations": {"antfly_decode_ms": 0.08}},
            "max_cv": 0.03,
            "total_ratio": 1.0,
            "max_total_ratio": 1.1,
            "decode_ratio": 1.0,
            "min_decode_ratio": 0.9,
        }
        self.assertIn("antfly_decode_ms=0.080", gate_errors(result)[0])


class HarnessContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="gemma4-long-output-test-")
        self.tmp = Path(self.tempdir.name)
        self.model = self.tmp / "model.gguf"
        self.model.write_bytes(b"fake gguf")
        self.antfly = self.tmp / "antfly"
        self.llama = self.tmp / "llama-completion"

        executable(
            self.antfly,
            """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import sys

args = sys.argv[1:]
tokens = int(args[args.index("--max-tokens") + 1])
split_enable = os.environ.get("TERMITE_METAL_ENABLE_DECODE_GQA_SPLIT")
split_gqa = split_enable in (None, "1") and os.environ.get("TERMITE_METAL_DISABLE_DECODE_GQA_SPLIT") != "1"
def env_true(name):
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")

prefill_direct = (
    env_true("TERMITE_METAL_ENABLE_PREFILL_SG_DIRECT_LOAD")
    and not env_true("TERMITE_METAL_DISABLE_PREFILL_SG_DIRECT_LOAD")
)
fast_prepared = (
    os.environ.get("TERMITE_METAL_DISABLE_FAST_PREPARED_FRAME") != "1"
    and os.environ.get("TERMITE_METAL_FORCE_DIAGNOSTIC_COMMAND_BUFFERS") != "1"
)
decode_frames = tokens - 1
attention_calls = decode_frames * 42
flash_local = 35
flash_hd512 = 7
q4_row_one = decode_frames * 210
timing_path = Path(args[args.index("--json-timing") + 1])
timing_path.write_text(json.dumps({
    "backend": "metal",
    "tokens": tokens,
    "token_ids": list(range(tokens)),
    "finish_reason": "length",
    "timing_ms": {"generate": 10000, "prefill_inner": 4000, "decode_inner": 6000},
    "speculative": None,
    "draft_cuda": None,
    "draft_cuda_generate": None,
    "runtime": {"decode_greedy_calls": decode_frames},
    "generation_decoder_runtime": {"forward_attempts": decode_frames},
    "metal": {
        "native_quant_null": False,
        "runtime_command_operators": {"fallback": 0},
        "attention_dispatch": {
            "paged_1x": 0 if split_gqa else attention_calls,
            "decode_gqa_split": attention_calls if split_gqa else 0,
            "generated_flash_prefill": flash_local,
            "generated_flash_prefill_hd512": flash_hd512,
            "prefill_direct_kv": flash_local + flash_hd512 if prefill_direct else 0,
            "prefill_paged_kv": 0 if prefill_direct else flash_local + flash_hd512,
        },
        "prepared_frame": {
            "fast_path": decode_frames if fast_prepared else 0,
            "fallback": 0 if fast_prepared else decode_frames,
        },
        "q4_0_policy": {
            "mmv_nr4_nsg2": q4_row_one,
            "mmv_nr8_nsg2": 0,
            "mmv_nr4_nsg4": 0,
            "mmv_nr8_nsg4": 0,
            "mmv_variant_fallbacks": 0,
            "mm_sg_aligned": 342,
            "mm_sg_aligned_tail": 342,
            "mm_sg_unrolled": 0,
        },
        "frame_fallbacks": {
            "decode_fallback": 0,
            "prefill_plan_fail": 0,
            "prefill_execute_fail": 0,
        },
        "quant_kernel_plan": {"fast_path_misses": 0, "unsupported_routes": 0},
    },
}))
print("generate-setup: live whole-model executor skipped")
print("gen_debug: executePrefill whole-model fast path seq_len=2003")
print("prompt_token_ids:", " ".join(str(i) for i in range(2003)))
print("token_ids:", " ".join(str(i) for i in range(tokens)))
print(
    f"metal_attention_dispatch: paged_1x={0 if split_gqa else attention_calls} "
    f"decode_gqa_split={attention_calls if split_gqa else 0} "
    f"generated_decode_1x=0 generated_flash_prefill={flash_local} "
    f"generated_flash_prefill_hd512={flash_hd512} "
    f"prefill_direct_kv={flash_local + flash_hd512 if prefill_direct else 0} "
    f"prefill_paged_kv={0 if prefill_direct else flash_local + flash_hd512} "
    "generated_rms_norm=0"
)
print(
    f"metal_prepared_frame: fast_path={decode_frames if fast_prepared else 0} "
    f"fallback={0 if fast_prepared else decode_frames}"
)
print("metal_runtime_memory: frame_retained_mb=0")
print(f"metal_q4_0_dispatch: linear_reduce_rows={q4_row_one}/0/0/0 pair_act_reduce=0")
print(
    f"metal_q4_0_policy: mmv_nr4_nsg2={q4_row_one} mmv_nr8_nsg2=0 "
    "mmv_nr4_nsg4=0 mmv_nr8_nsg4=0 mmv_variant_fallbacks=0 "
    "mm_sg_aligned=342 mm_sg_aligned_tail=342 mm_sg_unrolled=0"
)
print(f"metal_q4_q6_k_dispatch: q6_linear_reduce_rows={tokens}/0/0/0")
print("metal_jit_exact_dispatch: q4_0=0")
print("metal_q4_0_encode_us: linear_reduce=1234")
""",
        )
        executable(
            self.llama,
            """#!/usr/bin/env python3
import sys

args = sys.argv[1:]
if "--version" in args:
    print("version: 10182 (afeebe103)")
    print("built with AppleClang 21.0.0 for Darwin arm64")
    raise SystemExit(0)
if "-lv" not in args or args[args.index("-lv") + 1] != "4":
    raise SystemExit("benchmark must request llama log verbosity 4")
if "--log-colors" not in args or args[args.index("--log-colors") + 1] != "off":
    raise SystemExit("benchmark must disable llama log colors")
tokens = int(args[args.index("-n") + 1])
print("llama_prepare_model_devices: using device MTL0 (Apple M4)")
print("load_tensors: offloaded 43/43 layers to GPU")
print("ggml_metal_init: found device: Apple M4")
print("ggml_metal_init: picking default device: Apple M4")
print("0.17.923.567 I common_perf_print: sampling time = 50.00 ms / 300 runs")
print("0.17.923.567 I common_perf_print: prompt eval time = 4000.00 ms / 2003 tokens")
print(f"0.17.923.567 I common_perf_print: eval time = 5600.00 ms / {tokens - 1} runs")
print(f"0.17.923.567 I common_perf_print: total time = 9655.00 ms / {2003 + tokens - 1} tokens")
print("0.17.923.567 I common_perf_print: unaccounted time = 5.00 ms / 0.1 %")
print(f"0.17.923.567 I common_perf_print: graphs reused = {tokens - 3}")
""",
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def environment(self, out: Path) -> dict[str, str]:
        env = os.environ.copy()
        env.update(
            {
                "MODEL": str(self.model),
                "GGUF": str(self.model),
                "ANTFLY_BIN": str(self.antfly),
                "LLAMA_CPP_BIN": str(self.llama),
                "OUT_DIR": str(out),
                "OUTPUT_TOKENS": str(OUTPUT_TOKENS),
                "WARMUPS": "0",
                "RUNS": "2",
                "COOLDOWN_SECONDS": "0",
                "MAX_CV": "0.01",
                "EXPECTED_TOKEN_IDS_SHA256": TOKEN_IDS_SHA256,
                "EXPECTED_LLAMA_CPP_SHA256": hashlib.sha256(self.llama.read_bytes()).hexdigest(),
            }
        )
        env.pop("TERMITE_METAL_ENABLE_DECODE_GQA_SPLIT", None)
        env.pop("TERMITE_METAL_DISABLE_DECODE_GQA_SPLIT", None)
        for name in (
            "TERMITE_METAL_ENABLE_PREFILL_SG_DIRECT_LOAD",
            "TERMITE_METAL_DISABLE_PREFILL_SG_DIRECT_LOAD",
            "TERMITE_METAL_Q4_0_MMV_VARIANT",
            "TERMITE_METAL_DISABLE_Q4_0_MMV_PORTFOLIO",
            "TERMITE_METAL_TRACE_Q4_0_MMV_VARIANT",
            "TERMITE_METAL_DISABLE_SWA_SCAN_CLAMP",
            "TERMITE_METAL_DISABLE_FAST_PREPARED_FRAME",
            "TERMITE_METAL_FORCE_DIAGNOSTIC_COMMAND_BUFFERS",
            "EXPECT_GENERATED_FLASH_PREFILL_CALLS",
            "EXPECT_GENERATED_FLASH_PREFILL_HD512_CALLS",
            "EXPECT_PREFILL_DIRECT_KV",
            "EXPECT_FAST_PREPARED_FRAME",
            "EXPECT_Q4_0_MMV_VARIANT",
            "EXPECT_SWA_SCAN_CLAMP",
            "EXPECT_LLAMA_METAL_DEVICE",
            "EXPECT_LLAMA_OFFLOADED_LAYERS",
        ):
            env.pop(name, None)
        return env

    def run_harness(self, out: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", str(BENCHMARK)],
            env=env or self.environment(out),
            check=True,
            text=True,
            capture_output=True,
        )

    def test_full_harness_contract_and_provenance(self) -> None:
        out = self.tmp / "results"
        completed = self.run_harness(out)
        self.assertIn("prompt=2003 output=300", completed.stdout)
        summary = json.loads((out / "summary.json").read_text())
        self.assertEqual(summary["schema"], "antfly.gemma4_metal_long_output.v2")
        self.assertEqual(summary["prompt_tokens"], 2003)
        self.assertEqual(summary["output_tokens"], 300)
        self.assertLessEqual(summary["total_ratio"], 1.10)
        self.assertGreaterEqual(summary["decode_ratio"], 0.90)
        self.assertEqual(summary["total_ratio"], summary["paired_ratios"]["total_latency_ratio"]["median"])
        self.assertTrue(summary["cv_gate"]["passed"])
        self.assertEqual(
            set(summary["cv_gate"]["metrics"]),
            {
                "antfly_total_ms",
                "llama_total_ms",
                "antfly_prefill_ms",
                "llama_prompt_ms",
                "antfly_decode_ms",
                "llama_decode_ms",
            },
        )
        self.assertEqual(summary["token_ids_sha256"], TOKEN_IDS_SHA256)
        self.assertTrue(summary["exact_token_contract_passed"])
        self.assertTrue(summary["no_mtp"])
        self.assertTrue(summary["route_contract_passed"])
        row = summary["rows"][0]
        self.assertEqual(row["paged_1x_calls"], 0)
        self.assertEqual(row["decode_gqa_split_calls"], 12_558)
        self.assertEqual(row["generated_flash_prefill_calls"], 35)
        self.assertEqual(row["generated_flash_prefill_hd512_calls"], 7)
        self.assertEqual(row["prefill_direct_kv_calls"], 0)
        self.assertEqual(row["prefill_paged_kv_calls"], 42)
        self.assertEqual(row["prepared_frame_fast_path_calls"], 299)
        self.assertEqual(row["prepared_frame_fallback_calls"], 0)
        self.assertTrue(row["swa_scan_clamp_enabled"])
        self.assertEqual(row["llama_metal_device"], "Apple M4")
        self.assertEqual(row["llama_offloaded_layers"], 43)
        self.assertEqual(row["llama_total_layers"], 43)
        self.assertEqual(row["q4_0_linear_reduce_rows_1"], 62_790)
        self.assertEqual(row["q4_0_mmv_nr4_nsg2_calls"], 62_790)
        self.assertEqual(row["q4_0_mmv_variant_fallbacks"], 0)
        self.assertEqual(row["q4_0_mm_sg_aligned_calls"], 342)
        self.assertEqual(row["q4_0_mm_sg_aligned_tail_calls"], 342)
        self.assertEqual(row["q6_k_linear_reduce_rows_1"], 300)
        self.assertEqual(row["q4_0_linear_reduce_encode_us"], 1234)
        metadata = summary["metadata"]
        self.assertEqual(metadata["schema"], "antfly.gemma4_metal_long_output.metadata.v2")
        self.assertEqual(metadata["llama_cpp_build"], 10182)
        self.assertEqual(metadata["llama_cpp_commit"], "afeebe103")
        self.assertEqual(metadata["llama_cpp_full_commit"], "afeebe103bd99cda8f5dfaefcabadf890db7fda7")
        self.assertEqual(metadata["llama_cpp_binary_sha256"], hashlib.sha256(self.llama.read_bytes()).hexdigest())
        self.assertEqual(metadata["antfly_binary_sha256"], hashlib.sha256(self.antfly.read_bytes()).hexdigest())
        self.assertIn("built with AppleClang", metadata["llama_cpp_version_output"])
        repo_root = SCRIPT_DIR.parents[3]
        git_status = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain=v1", "--untracked-files=all"],
            env={**os.environ, "LC_ALL": "C"},
        )
        tracked_diff = subprocess.check_output(
            ["git", "-C", str(repo_root), "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
            env={**os.environ, "LC_ALL": "C"},
        )
        self.assertEqual(metadata["git_dirty"], bool(git_status.rstrip(b"\n")))
        self.assertEqual(metadata["git_status_sha256"], hashlib.sha256(git_status).hexdigest())
        self.assertEqual(
            metadata["git_tracked_diff_sha256"],
            hashlib.sha256(tracked_diff).hexdigest(),
        )
        for key in (
            "git_tracked_diff_sha256",
            "git_status_sha256",
            "benchmark_harness_sha256",
            "benchmark_parser_sha256",
        ):
            self.assertRegex(metadata[key], r"^[0-9a-f]{64}$")
        self.assertEqual(
            metadata["benchmark_harness_sha256"],
            hashlib.sha256(BENCHMARK.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            metadata["benchmark_parser_sha256"],
            hashlib.sha256((SCRIPT_DIR / "gemma4_metal_long_output.py").read_bytes()).hexdigest(),
        )
        self.assertEqual(
            set(metadata["metal_policy_env"]),
            {
                "TERMITE_METAL_ENABLE_PREFILL_SG_DIRECT_LOAD",
                "TERMITE_METAL_DISABLE_PREFILL_SG_DIRECT_LOAD",
                "TERMITE_METAL_Q4_0_MMV_VARIANT",
                "TERMITE_METAL_DISABLE_Q4_0_MMV_PORTFOLIO",
                "TERMITE_METAL_TRACE_Q4_0_MMV_VARIANT",
                "TERMITE_METAL_DISABLE_SWA_SCAN_CLAMP",
                "TERMITE_METAL_DISABLE_FAST_PREPARED_FRAME",
                "TERMITE_METAL_FORCE_DIAGNOSTIC_COMMAND_BUFFERS",
                "EXPECT_GENERATED_FLASH_PREFILL_CALLS",
                "EXPECT_GENERATED_FLASH_PREFILL_HD512_CALLS",
                "EXPECT_PREFILL_DIRECT_KV",
                "EXPECT_FAST_PREPARED_FRAME",
                "EXPECT_Q4_0_MMV_VARIANT",
                "EXPECT_SWA_SCAN_CLAMP",
                "EXPECT_LLAMA_METAL_DEVICE",
                "EXPECT_LLAMA_OFFLOADED_LAYERS",
            },
        )
        self.assertTrue(summary["policy_route_expectations"]["q4_0_mmv_portfolio"])
        self.assertEqual(summary["policy_route_expectations"]["generated_flash_prefill_calls"], 35)
        self.assertEqual(summary["policy_route_expectations"]["generated_flash_prefill_hd512_calls"], 7)
        self.assertFalse(summary["policy_route_expectations"]["prefill_direct_kv"])
        self.assertTrue(summary["policy_route_expectations"]["fast_prepared_frame"])
        self.assertEqual(summary["policy_route_expectations"]["q4_0_mmv_variant"], "nr4-nsg2")
        self.assertTrue(summary["policy_route_expectations"]["swa_scan_clamp"])
        self.assertEqual(summary["llama_backend_expectations"]["metal_device"], "Apple M4")
        self.assertEqual(summary["llama_backend_expectations"]["offloaded_layers"], 43)

        with self.assertRaisesRegex(BenchmarkContractError, "exact token digest changed"):
            build_result(out, 2, OUTPUT_TOKENS, 1.10, 0.90, 0.01, "0" * 64)

        antfly_json = out / "antfly-1.json"
        payload = json.loads(antfly_json.read_text())
        payload["speculative"] = {"accepted": 1}
        antfly_json.write_text(json.dumps(payload))
        with self.assertRaisesRegex(BenchmarkContractError, "MTP/speculative"):
            build_result(out, 2, OUTPUT_TOKENS, 1.10, 0.90, 0.01)

    def test_paged_attention_rollback_route(self) -> None:
        out = self.tmp / "paged-results"
        env = self.environment(out)
        env["TERMITE_METAL_DISABLE_DECODE_GQA_SPLIT"] = "1"
        self.run_harness(out, env)
        row = json.loads((out / "summary.json").read_text())["rows"][0]
        self.assertEqual(row["paged_1x_calls"], 12_558)
        self.assertEqual(row["decode_gqa_split_calls"], 0)

    def test_explicit_policy_rollback_routes_and_metadata(self) -> None:
        out = self.tmp / "policy-rollback-results"
        env = self.environment(out)
        env.update(
            {
                "TERMITE_METAL_DISABLE_PREFILL_SG_DIRECT_LOAD": "1",
                "TERMITE_METAL_DISABLE_Q4_0_MMV_PORTFOLIO": "1",
                "TERMITE_METAL_DISABLE_SWA_SCAN_CLAMP": "1",
                "TERMITE_METAL_DISABLE_FAST_PREPARED_FRAME": "1",
                "TERMITE_METAL_FORCE_DIAGNOSTIC_COMMAND_BUFFERS": "1",
                "EXPECT_PREFILL_DIRECT_KV": "0",
                "EXPECT_FAST_PREPARED_FRAME": "0",
                "EXPECT_Q4_0_MMV_VARIANT": "any",
                "EXPECT_SWA_SCAN_CLAMP": "0",
            }
        )
        self.run_harness(out, env)
        summary = json.loads((out / "summary.json").read_text())
        row = summary["rows"][0]
        self.assertEqual(row["prefill_direct_kv_calls"], 0)
        self.assertEqual(row["prefill_paged_kv_calls"], 42)
        self.assertEqual(row["prepared_frame_fast_path_calls"], 0)
        self.assertEqual(row["prepared_frame_fallback_calls"], 299)
        self.assertFalse(row["swa_scan_clamp_enabled"])
        self.assertFalse(summary["policy_route_expectations"]["prefill_direct_kv"])
        self.assertFalse(summary["policy_route_expectations"]["fast_prepared_frame"])
        self.assertFalse(summary["policy_route_expectations"]["q4_0_mmv_portfolio"])
        self.assertEqual(summary["policy_route_expectations"]["q4_0_mmv_variant"], "any")
        self.assertFalse(summary["policy_route_expectations"]["swa_scan_clamp"])
        policy_env = summary["metadata"]["metal_policy_env"]
        self.assertEqual(policy_env["TERMITE_METAL_DISABLE_SWA_SCAN_CLAMP"], "1")

    def test_direct_kv_route_requires_explicit_enable(self) -> None:
        out = self.tmp / "explicit-direct-route"
        env = self.environment(out)
        env["TERMITE_METAL_ENABLE_PREFILL_SG_DIRECT_LOAD"] = "1"
        env["EXPECT_PREFILL_DIRECT_KV"] = "1"
        self.run_harness(out, env)
        row = json.loads((out / "summary.json").read_text())["rows"][0]
        self.assertEqual(row["prefill_direct_kv_calls"], 42)
        self.assertEqual(row["prefill_paged_kv_calls"], 0)

    def test_explicit_direct_kv_expectation_rejects_paged_route(self) -> None:
        out = self.tmp / "wrong-direct-route"
        env = self.environment(out)
        env["TERMITE_METAL_DISABLE_PREFILL_SG_DIRECT_LOAD"] = "1"
        env["EXPECT_PREFILL_DIRECT_KV"] = "1"
        completed = subprocess.run(
            ["bash", str(BENCHMARK)],
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("prefill direct K/V route", completed.stderr)

    def test_exact_flash_prefill_route_counts_are_required(self) -> None:
        for field, log_key, observed, expected_message in (
            ("generated_flash_prefill", "generated_flash_prefill", 34, "34/7"),
            ("generated_flash_prefill_hd512", "generated_flash_prefill_hd512", 6, "35/6"),
        ):
            with self.subTest(field=field):
                out = self.tmp / f"wrong-{field}"
                self.run_harness(out)
                antfly_log = out / "antfly-1.log"
                original = 35 if field == "generated_flash_prefill" else 7
                antfly_log.write_text(
                    antfly_log.read_text().replace(
                        f"{log_key}={original}",
                        f"{log_key}={observed}",
                    )
                )
                antfly_json = out / "antfly-1.json"
                payload = json.loads(antfly_json.read_text())
                payload["metal"]["attention_dispatch"][field] = observed
                antfly_json.write_text(json.dumps(payload))
                with self.assertRaisesRegex(
                    BenchmarkContractError,
                    f"generated flash prefill routes={expected_message}",
                ):
                    build_result(out, 2, OUTPUT_TOKENS, 1.10, 0.90, 0.01)

    def test_q4_mmv_canonical_variant_must_be_one_hot(self) -> None:
        out = self.tmp / "wrong-q4-variant"
        self.run_harness(out)
        antfly_log = out / "antfly-1.log"
        antfly_log.write_text(
            antfly_log.read_text().replace(
                "mmv_nr4_nsg2=62790 mmv_nr8_nsg2=0",
                "mmv_nr4_nsg2=0 mmv_nr8_nsg2=62790",
            )
        )
        antfly_json = out / "antfly-1.json"
        payload = json.loads(antfly_json.read_text())
        payload["metal"]["q4_0_policy"]["mmv_nr4_nsg2"] = 0
        payload["metal"]["q4_0_policy"]["mmv_nr8_nsg2"] = 62_790
        antfly_json.write_text(json.dumps(payload))
        with self.assertRaisesRegex(BenchmarkContractError, "Q4_0 MMV one-hot route"):
            build_result(out, 2, OUTPUT_TOKENS, 1.10, 0.90, 0.01)

    def test_swa_scan_clamp_requires_explicit_rollback_expectation(self) -> None:
        out = self.tmp / "unexpected-swa-rollback"
        env = self.environment(out)
        env["TERMITE_METAL_DISABLE_SWA_SCAN_CLAMP"] = "1"
        completed = subprocess.run(
            ["bash", str(BENCHMARK)],
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("SWA scan clamp policy", completed.stderr)

    def test_llama_metal_device_marker_is_required(self) -> None:
        out = self.tmp / "missing-llama-metal-marker"
        self.run_harness(out)
        llama_log = out / "llama-1.log"
        llama_log.write_text(
            llama_log.read_text().replace(
                "ggml_metal_init: found device: Apple M4",
                "ggml_metal_init: skipped device: Apple M4",
            )
        )
        with self.assertRaisesRegex(BenchmarkContractError, "Metal found-device marker"):
            build_result(out, 2, OUTPUT_TOKENS, 1.10, 0.90, 0.01)

    def test_llama_device_identity_and_full_offload_are_required(self) -> None:
        device_out = self.tmp / "wrong-llama-device"
        self.run_harness(device_out)
        device_log = device_out / "llama-1.log"
        device_log.write_text(device_log.read_text().replace("Apple M4", "Apple M3"))
        with self.assertRaisesRegex(BenchmarkContractError, "Metal device='Apple M3'"):
            build_result(device_out, 2, OUTPUT_TOKENS, 1.10, 0.90, 0.01)

        offload_out = self.tmp / "partial-llama-offload"
        self.run_harness(offload_out)
        offload_log = offload_out / "llama-1.log"
        offload_log.write_text(offload_log.read_text().replace("43/43", "42/43"))
        with self.assertRaisesRegex(BenchmarkContractError, "GPU layer offload=42/43"):
            build_result(offload_out, 2, OUTPUT_TOKENS, 1.10, 0.90, 0.01)

    def test_benchmark_source_provenance_is_fail_closed(self) -> None:
        out = self.tmp / "bad-source-provenance"
        self.run_harness(out)
        metadata_path = out / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["benchmark_parser_sha256"] = "0" * 64
        metadata_path.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(BenchmarkContractError, "benchmark source hash mismatch"):
            build_result(out, 2, OUTPUT_TOKENS, 1.10, 0.90, 0.01)

    def test_q4_mmv_portfolio_fallback_is_rejected(self) -> None:
        out = self.tmp / "q4-mmv-fallback"
        self.run_harness(out)
        antfly_log = out / "antfly-1.log"
        antfly_log.write_text(
            antfly_log.read_text().replace("mmv_variant_fallbacks=0", "mmv_variant_fallbacks=1")
        )
        antfly_json = out / "antfly-1.json"
        payload = json.loads(antfly_json.read_text())
        payload["metal"]["q4_0_policy"]["mmv_variant_fallbacks"] = 1
        antfly_json.write_text(json.dumps(payload))
        with self.assertRaisesRegex(BenchmarkContractError, "MMV portfolio fallbacks=1"):
            build_result(out, 2, OUTPUT_TOKENS, 1.10, 0.90, 0.01)

    def test_comparator_hash_pin_fails_before_benchmark(self) -> None:
        out = self.tmp / "bad-comparator"
        env = self.environment(out)
        env["EXPECTED_LLAMA_CPP_SHA256"] = "0" * 64
        completed = subprocess.run(
            ["bash", str(BENCHMARK)],
            env=env,
            check=False,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("binary SHA-256 mismatch", completed.stderr)
        self.assertFalse((out / "antfly-1.json").exists())


if __name__ == "__main__":
    unittest.main()
