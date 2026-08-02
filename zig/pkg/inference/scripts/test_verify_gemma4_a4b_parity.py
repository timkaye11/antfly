#!/usr/bin/env python3
"""Hermetic contract tests for verify_gemma4_a4b_parity.py (no real binaries)."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).with_name("verify_gemma4_a4b_parity.py")
SPEC = importlib.util.spec_from_file_location("verify_gemma4_a4b_parity", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MOD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MOD
SPEC.loader.exec_module(MOD)


@contextlib.contextmanager
def _patched(**attributes):
    saved = {name: getattr(MOD, name) for name in attributes}
    for name, value in attributes.items():
        setattr(MOD, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(MOD, name, value)


def _llama_output(extra: str = "") -> str:
    return (
        "llama_completion: number of tokens in prompt = 2\n"
        "1.40.130.003 I      2 -> '<bos>'\n"
        "1.40.136.965 I   9259 -> 'Hello'\n"
        "reference_token_id: 111\n"
        "reference_token_id: 222\n" + extra
    )


LLAMA_MARKERS = (
    "reference_route: 0 2 3 7\n"
    "reference_route: 1 2 4 9\n"
    "reference_logit_sample: 0 1.25\n"
    "reference_logit_sample: 5 -0.5\n"
)

MATCHING_ROUTE_LINES = (
    "layer0_routes: 3:0.600000 7:0.400000\n"
    "layer1_routes: 4:0.700000 9:0.300000\n"
)

MATCHING_LOGIT_LINES = (
    "logit_sample: 0 1.259000\n"
    "logit_sample: 5 -0.495000\n"
)

REFERENCE_ROUTES = [
    {"layer": 0, "top_k": 2, "expert_ids": [3, 7]},
    {"layer": 1, "top_k": 2, "expert_ids": [4, 9]},
]

REFERENCE_LOGIT_SAMPLES = [
    {"index": 0, "value": 1.25},
    {"index": 5, "value": -0.5},
]


def _antfly_output(lane: str, extra: str = "") -> str:
    return (
        "prompt_token_ids: 2 9259\n"
        "token_ids: 111 222\n"
        f"cache_dtype_effective: requested={lane} effective={lane}\n" + extra
    )


class AntflyRunRecorder:
    """Monkeypatch stand-in for MOD._run that fakes Antfly engine output."""

    def __init__(self, extra_for_lane=None):
        self.calls = []
        self.extra_for_lane = extra_for_lane or (lambda lane: "")

    def __call__(self, command, environment, log_path, timeout):
        self.calls.append((list(command), dict(environment), Path(log_path)))
        lane = command[command.index("--cache-dtype") + 1]
        return _antfly_output(lane, self.extra_for_lane(lane))


class ParityParserTests(unittest.TestCase):
    def test_parse_antfly_output(self) -> None:
        prompt, output = MOD.parse_antfly_output(
            "prompt_token_ids: 2 9259\ntoken_ids: 3324 236772 616\n"
        )
        self.assertEqual([2, 9259], prompt)
        self.assertEqual([3324, 236772, 616], output)

    def test_parse_llama_output(self) -> None:
        prompt, output = MOD.parse_llama_output(
            "llama_completion: number of tokens in prompt = 2\n"
            "1.40.130.003 I      2 -> '<bos>'\n"
            "1.40.136.965 I   9259 -> 'Hello'\n"
            "reference_token_id: 3324\n"
            "reference_token_id: 236772\n"
        )
        self.assertEqual([2, 9259], prompt)
        self.assertEqual([3324, 236772], output)

    def test_first_difference(self) -> None:
        self.assertEqual("index 2: expected 3, got 4", MOD._first_difference([1, 2, 3], [1, 2, 4]))
        self.assertEqual("length: expected 2, got 1", MOD._first_difference([1, 2], [1]))

    def test_parse_antfly_cache_dtype(self) -> None:
        requested, effective = MOD.parse_antfly_cache_dtype(
            "cache_dtype_effective: requested=f16 effective=f16\n"
        )
        self.assertEqual(("f16", "f16"), (requested, effective))

    def test_parse_antfly_routes_gpt_zig_format(self) -> None:
        # maybeDebugMoeRoutesLastRow prints ``layer{d}_routes:`` followed by
        # space-separated ``expert:weight`` pairs with {d:.6} weights.
        routes = MOD.parse_antfly_routes(
            "noise line\n"
            "layer0_routes: 12:0.483210 7:0.211001\n"
            "layer17_routes: 3:-0.125000 90:0.000001\n"
        )
        self.assertEqual(2, len(routes))
        self.assertEqual({"layer": 0, "expert_ids": [12, 7], "route_weights": [0.48321, 0.211001]}, routes[0])
        self.assertEqual(17, routes[1]["layer"])
        self.assertEqual([3, 90], routes[1]["expert_ids"])
        self.assertAlmostEqual(-0.125, routes[1]["route_weights"][0])

    def test_parse_antfly_routes_rejects_malformed_pairs(self) -> None:
        with self.assertRaises(MOD.ParityError):
            MOD.parse_antfly_routes("layer0_routes: 12:0.5 seven:0.5\n")
        with self.assertRaises(MOD.ParityError):
            MOD.parse_antfly_routes("layer0_routes:\n")

    def test_parse_reference_markers(self) -> None:
        routes = MOD.parse_reference_routes("reference_route: 3 4 1 2 3 4\n")
        self.assertEqual([{"layer": 3, "top_k": 4, "expert_ids": [1, 2, 3, 4]}], routes)
        samples = MOD.parse_reference_logit_samples(
            "reference_logit_sample: 0 1.25\nreference_logit_sample: 9 -3.5e-2\n"
        )
        self.assertEqual([0, 9], [sample["index"] for sample in samples])
        self.assertAlmostEqual(-0.035, samples[1]["value"])

    def test_parse_reference_markers_absent_is_empty_never_error(self) -> None:
        self.assertEqual([], MOD.parse_reference_routes(_llama_output()))
        self.assertEqual([], MOD.parse_reference_logit_samples(_llama_output()))

    def test_parse_reference_route_k_mismatch_fails_closed(self) -> None:
        with self.assertRaises(MOD.ParityError):
            MOD.parse_reference_routes("reference_route: 0 3 1 2\n")

    def test_compare_route_sets_uses_last_line_per_layer(self) -> None:
        reference = [{"layer": 0, "top_k": 2, "expert_ids": [3, 7]}]
        antfly = MOD.parse_antfly_routes(
            "layer0_routes: 1:0.900000 2:0.100000\n"  # earlier decode step
            "layer0_routes: 3:0.600000 7:0.400000\n"  # last line wins
        )
        self.assertEqual([], MOD.compare_route_sets("f16/x", reference, antfly))

    def test_compare_route_sets_reports_mismatch_and_missing_layer(self) -> None:
        reference = [
            {"layer": 0, "top_k": 2, "expert_ids": [3, 7]},
            {"layer": 1, "top_k": 2, "expert_ids": [4, 9]},
        ]
        antfly = MOD.parse_antfly_routes("layer0_routes: 3:0.500000 8:0.500000\n")
        failures = MOD.compare_route_sets("f16/x", reference, antfly)
        self.assertEqual(2, len(failures))
        self.assertIn("ordered expert mismatch", failures[0])
        self.assertIn("layer 1 routes missing", failures[1])

    def test_compare_logit_samples_tolerance(self) -> None:
        reference = [{"index": 0, "value": 1.25}]
        within = [{"index": 0, "value": 1.259}]
        outside = [{"index": 0, "value": 1.262}]
        self.assertEqual([], MOD.compare_logit_samples("f16/x", reference, within))
        failures = MOD.compare_logit_samples("f16/x", reference, outside)
        self.assertEqual(1, len(failures))
        self.assertIn("abs error", failures[0])
        missing = MOD.compare_logit_samples("f16/x", reference, [{"index": 3, "value": 1.25}])
        self.assertIn("missing from Antfly output", missing[0])

    def test_native_oracle_explicitly_widens_host_budget(self) -> None:
        command = MOD._antfly_command(
            Path("/tmp/antfly"),
            Path("/tmp/model.gguf"),
            "hello",
            1,
            "native",
            False,
            "f16",
        )
        self.assertIn("--host-budget-mb", command)
        self.assertIn("--combined-budget-mb", command)

        compact = MOD._antfly_command(
            Path("/tmp/antfly"),
            Path("/tmp/model.gguf"),
            "hello",
            1,
            "metal",
            True,
            "f16",
        )
        self.assertNotIn("--host-budget-mb", compact)
        self.assertEqual("2gbs", compact[compact.index("--memory-profile") + 1])
        self.assertEqual("2048", compact[compact.index("--memory-budget-mb") + 1])
        self.assertEqual("off", compact[compact.index("--compact-device-routing") + 1])
        self.assertEqual("42", compact[compact.index("--seed") + 1])
        self.assertEqual("f16", compact[compact.index("--cache-dtype") + 1])

    def test_f32_compact_lane_widens_only_kv_budget(self) -> None:
        compact = MOD._antfly_command(
            Path("/tmp/antfly"),
            Path("/tmp/model.gguf"),
            "hello",
            1,
            "metal",
            True,
            "f32",
        )
        self.assertEqual("f32", compact[compact.index("--cache-dtype") + 1])
        self.assertEqual("512", compact[compact.index("--kv-budget-mb") + 1])

    def test_llama_cache_lane_sets_both_k_and_v(self) -> None:
        command = MOD._llama_command(
            Path("/tmp/llama"), Path("/tmp/model.gguf"), "hello", 1, 0, "f32"
        )
        self.assertEqual("f32", command[command.index("-ctk") + 1])
        self.assertEqual("f32", command[command.index("-ctv") + 1])


class RecordVerifyWorkspace:
    """Shared temp-dir fixture: stub binaries, model, and reference bundles."""

    def __init__(self, root: Path):
        self.root = root
        self.model = root / "model.gguf"
        self.model.write_bytes(b"MODEL BYTES")
        self.antfly = root / "antfly"
        self.antfly.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.antfly.chmod(0o755)
        self.llama = root / "llama"
        self.llama.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        self.llama.chmod(0o755)
        self.out_serial = 0

    def out_dir(self) -> Path:
        self.out_serial += 1
        return self.root / f"out{self.out_serial}"

    def write_bundle(
        self,
        lanes=("f16", "f32"),
        tokens=2,
        routes=None,
        logit_samples=None,
        model_sha256=None,
    ) -> Path:
        self.out_serial += 1
        bundle = self.root / f"bundle{self.out_serial}"
        bundle.mkdir()
        lane_docs = {}
        for lane in lanes:
            lane_docs[lane] = {
                "cache_dtype": lane,
                "prompt_token_ids": [2, 9259],
                "token_ids": [111, 222],
                "routes": list(routes or []),
                "logit_samples": list(logit_samples or []),
                "log": f"llama.{lane}.log",
            }
        reference = {
            "schema": MOD.REFERENCE_SCHEMA,
            "prompt": "Hello upon-",
            "tokens": tokens,
            "cache_dtype_lanes": list(lanes),
            "lanes": lane_docs,
            "model": str(self.model),
            "model_sha256": model_sha256 or MOD._sha256(self.model),
            "tokenizer_file": None,
            "tokenizer_sha256": None,
            "llama_binary": str(self.llama),
            "llama_binary_sha256": MOD._sha256(self.llama),
            "llama_version": "stub (32703b4)",
            "expected_llama_commit": MOD.PINNED_LLAMA_COMMIT,
            "llama_gpu_layers": 0,
        }
        (bundle / "reference.json").write_text(
            json.dumps(reference, indent=2) + "\n", encoding="utf-8"
        )
        return bundle


class RecordReferenceTests(unittest.TestCase):
    def _record(self, workspace: RecordVerifyWorkspace, extra_args=(), markers=""):
        calls = []

        def fake_run(command, environment, log_path, timeout):
            calls.append((list(command), dict(environment), Path(log_path)))
            Path(log_path).write_text("raw llama log\n", encoding="utf-8")
            return _llama_output(markers)

        out_dir = workspace.out_dir()
        with _patched(_run=fake_run, _llama_version=lambda llama: "version: 4242 (32703b42)"):
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = MOD.main(
                    [
                        "record-reference",
                        "--llama-bin",
                        str(workspace.llama),
                        "--model",
                        str(workspace.model),
                        "--out-dir",
                        str(out_dir),
                        "--tokens",
                        "2",
                        *extra_args,
                    ]
                )
        return rc, out_dir, calls, stdout.getvalue()

    def test_record_writes_reference_bundle_with_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            rc, out_dir, calls, stdout = self._record(workspace, markers=LLAMA_MARKERS)
            self.assertEqual(0, rc, stdout)
            self.assertEqual(2, len(calls), "one sequential llama run per lane")
            self.assertEqual("f16", calls[0][0][calls[0][0].index("-ctk") + 1])
            self.assertEqual("f32", calls[1][0][calls[1][0].index("-ctk") + 1])
            reference = json.loads((out_dir / "reference.json").read_text(encoding="utf-8"))
            self.assertEqual(MOD.REFERENCE_SCHEMA, reference["schema"])
            self.assertEqual(
                hashlib.sha256(b"MODEL BYTES").hexdigest(), reference["model_sha256"]
            )
            self.assertEqual(MOD._sha256(workspace.llama), reference["llama_binary_sha256"])
            self.assertEqual(["f16", "f32"], reference["cache_dtype_lanes"])
            for lane in ("f16", "f32"):
                lane_doc = reference["lanes"][lane]
                self.assertEqual([2, 9259], lane_doc["prompt_token_ids"])
                self.assertEqual([111, 222], lane_doc["token_ids"])
                self.assertEqual(REFERENCE_ROUTES, lane_doc["routes"])
                self.assertEqual(
                    [0, 5], [sample["index"] for sample in lane_doc["logit_samples"]]
                )
                self.assertTrue((out_dir / f"llama.{lane}.log").is_file(), "raw llama log copied")
            self.assertIn("reference recorded:", stdout)
            self.assertIn("prompt_tokens=2", stdout)

    def test_record_absent_markers_are_empty_lists_not_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            rc, out_dir, _, stdout = self._record(workspace, markers="")
            self.assertEqual(0, rc, stdout)
            reference = json.loads((out_dir / "reference.json").read_text(encoding="utf-8"))
            self.assertEqual([], reference["lanes"]["f16"]["routes"])
            self.assertEqual([], reference["lanes"]["f16"]["logit_samples"])
            self.assertIn("routes=0 logit_samples=0", stdout)

    def test_record_hashes_optional_tokenizer_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            tokenizer = workspace.root / "tokenizer.json"
            tokenizer.write_bytes(b"TOKENIZER")
            rc, out_dir, _, _ = self._record(
                workspace, extra_args=("--tokenizer-file", str(tokenizer))
            )
            self.assertEqual(0, rc)
            reference = json.loads((out_dir / "reference.json").read_text(encoding="utf-8"))
            self.assertEqual(hashlib.sha256(b"TOKENIZER").hexdigest(), reference["tokenizer_sha256"])

    def test_record_rejects_unpinned_llama_version(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            with _patched(
                _run=lambda *a, **k: _llama_output(),
                _llama_version=lambda llama: "version: 4242 (deadbee)",
            ):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    rc = MOD.main(
                        [
                            "record-reference",
                            "--llama-bin",
                            str(workspace.llama),
                            "--model",
                            str(workspace.model),
                            "--out-dir",
                            str(workspace.out_dir()),
                        ]
                    )
            self.assertEqual(1, rc)
            self.assertIn("pinned commit", stderr.getvalue())

    def test_record_nonpositive_tokens_is_usage_error(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = MOD.main(
                [
                    "record-reference",
                    "--llama-bin",
                    "/nonexistent/llama",
                    "--model",
                    "/nonexistent/model.gguf",
                    "--out-dir",
                    "/nonexistent/out",
                    "--tokens",
                    "0",
                ]
            )
        self.assertEqual(2, rc)
        self.assertIn("--tokens must be positive", stderr.getvalue())


class VerifyReferenceTests(unittest.TestCase):
    def _verify(self, workspace, bundle, extra_args=(), extra_for_lane=None):
        recorder = AntflyRunRecorder(extra_for_lane)
        out_dir = workspace.out_dir()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with _patched(_run=recorder):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                rc = MOD.main(
                    [
                        "verify-reference",
                        "--antfly-bin",
                        str(workspace.antfly),
                        "--model",
                        str(workspace.model),
                        "--reference",
                        str(bundle),
                        "--out-dir",
                        str(out_dir),
                        *extra_args,
                    ]
                )
        return rc, out_dir, recorder.calls, stdout.getvalue(), stderr.getvalue()

    def _summary(self, out_dir: Path) -> dict:
        return json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    def test_default_runs_exactly_one_compact_engine_per_lane(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            bundle = workspace.write_bundle()
            rc, out_dir, calls, stdout, stderr = self._verify(workspace, bundle)
            self.assertEqual(0, rc, stderr)
            self.assertEqual(2, len(calls), "one compact engine per recorded lane")
            for command, environment, _ in calls:
                self.assertIn("--memory-profile", command)
                self.assertEqual("2gbs", command[command.index("--memory-profile") + 1])
                self.assertEqual("metal", command[command.index("--backend") + 1])
            summary = self._summary(out_dir)
            self.assertEqual(MOD.SUMMARY_SCHEMA, summary["schema"])
            self.assertTrue(summary["passed"])
            self.assertEqual(["metal_compact"], summary["engines_selected"])
            for lane in ("f16", "f32"):
                engine = summary["lanes"][lane]["engines"]["metal_compact"]
                self.assertEqual(lane, engine["cache_dtype_requested"])
                self.assertEqual(lane, engine["cache_dtype_effective"])
                self.assertEqual(
                    "Antfly cache_dtype_effective log", engine["cache_dtype_provenance"]
                )

    def test_success_path_prints_summary_without_keyerror_regression(self) -> None:
        # The v2 tool crashed after a PASSING run: main() read
        # summary["engines"]["llama"], but engines were nested per lane.
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            bundle = workspace.write_bundle()
            rc, _, _, stdout, stderr = self._verify(workspace, bundle)
            self.assertEqual(0, rc, stderr)
            self.assertIn("exact parity passed:", stdout)
            self.assertIn("prompt_tokens=2", stdout)
            self.assertIn("output_tokens=2", stdout)
            self.assertIn("lanes=f16,f32", stdout)

    def test_compare_compact_routes_records_exact_off_vs_full_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            calls = []

            def fake_run(command, environment, log_path, timeout):
                calls.append(list(command))
                Path(log_path).write_text("route log\n", encoding="utf-8")
                return _antfly_output("f16")

            out_dir = workspace.out_dir()
            with _patched(_run=fake_run):
                rc = MOD.main(
                    [
                        "compare-compact-routes",
                        "--antfly-bin", str(workspace.antfly),
                        "--model", str(workspace.model),
                        "--out-dir", str(out_dir),
                        "--tokens", "2",
                    ]
                )
            self.assertEqual(0, rc)
            self.assertEqual(2, len(calls))
            self.assertEqual("off", calls[0][calls[0].index("--compact-device-routing") + 1])
            self.assertEqual("full", calls[1][calls[1].index("--compact-device-routing") + 1])
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["passed"])
            self.assertEqual(MOD.ROUTE_SUMMARY_SCHEMA, summary["schema"])

    def test_full_matrix_runs_three_engines_per_lane(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            bundle = workspace.write_bundle()
            rc, out_dir, calls, _, stderr = self._verify(
                workspace, bundle, extra_args=("--full-matrix",)
            )
            self.assertEqual(0, rc, stderr)
            self.assertEqual(6, len(calls), "native + canonical + compact per lane")
            summary = self._summary(out_dir)
            self.assertEqual(
                ["native", "metal_canonical", "metal_compact"], summary["engines_selected"]
            )
            canonical_calls = [
                (command, environment)
                for command, environment, _ in calls
                if "--memory-profile" not in command
                and command[command.index("--backend") + 1] == "metal"
            ]
            self.assertEqual(2, len(canonical_calls))
            for _, environment in canonical_calls:
                self.assertEqual("1", environment.get("ANTFLY_GEMMA4_DISABLE_COMPACT_MOE"))
            for lane in ("f16", "f32"):
                overrides = summary["lanes"][lane]["engines"]["metal_compact"][
                    "environment_overrides"
                ]
                self.assertNotIn("ANTFLY_GEMMA4_DISABLE_COMPACT_MOE", overrides)

    def test_model_hash_mismatch_fails_closed_before_running_engines(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            bundle = workspace.write_bundle(model_sha256="f" * 64)
            rc, out_dir, calls, _, stderr = self._verify(workspace, bundle)
            self.assertEqual(1, rc)
            self.assertEqual(0, len(calls), "no engine may run on hash mismatch")
            self.assertIn("model hash mismatch", stderr)
            self.assertFalse((out_dir / "summary.json").exists())

    def test_model_hash_mismatch_override_warns_and_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            bundle = workspace.write_bundle(model_sha256="f" * 64)
            rc, out_dir, calls, _, stderr = self._verify(
                workspace, bundle, extra_args=("--allow-model-hash-mismatch",)
            )
            self.assertEqual(0, rc, stderr)
            self.assertEqual(2, len(calls))
            self.assertIn("parity warning: model hash mismatch", stderr)
            summary = self._summary(out_dir)
            self.assertFalse(summary["model_hash_match"])
            self.assertTrue(summary["allow_model_hash_mismatch"])

    def test_route_and_logit_gates_pass_when_present_and_matching(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            bundle = workspace.write_bundle(
                routes=REFERENCE_ROUTES, logit_samples=REFERENCE_LOGIT_SAMPLES
            )
            rc, out_dir, calls, _, stderr = self._verify(
                workspace,
                bundle,
                extra_for_lane=lambda lane: MATCHING_ROUTE_LINES + MATCHING_LOGIT_LINES,
            )
            self.assertEqual(0, rc, stderr)
            for _, environment, _ in calls:
                self.assertEqual(
                    "1",
                    environment.get("TERMITE_DEBUG_GPT_LAST_ROW"),
                    "route verification must enable the gpt.zig route debug lines",
                )
            summary = self._summary(out_dir)
            self.assertEqual([], summary["skipped_gates"])
            gates = summary["lanes"]["f16"]["engines"]["metal_compact"]["gates"]
            self.assertEqual("passed", gates["expert_routes"]["status"])
            self.assertEqual("passed", gates["logit_samples"]["status"])

    def test_route_gate_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            bundle = workspace.write_bundle(routes=REFERENCE_ROUTES)
            mismatched = (
                "layer0_routes: 3:0.600000 7:0.400000\n"
                "layer1_routes: 5:0.700000 4:0.300000\n"  # 5 instead of 9
            )
            rc, out_dir, _, _, stderr = self._verify(
                workspace, bundle, extra_for_lane=lambda lane: mismatched
            )
            self.assertEqual(1, rc)
            self.assertIn("ordered expert mismatch", stderr)
            summary = self._summary(out_dir)
            self.assertFalse(summary["passed"])
            gates = summary["lanes"]["f16"]["engines"]["metal_compact"]["gates"]
            self.assertEqual("failed", gates["expert_routes"]["status"])

    def test_route_gate_absent_on_antfly_side_is_recorded_skip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            bundle = workspace.write_bundle(routes=REFERENCE_ROUTES)
            rc, out_dir, _, _, stderr = self._verify(workspace, bundle)
            self.assertEqual(0, rc, stderr)
            summary = self._summary(out_dir)
            self.assertTrue(summary["passed"])
            gates = summary["lanes"]["f16"]["engines"]["metal_compact"]["gates"]
            self.assertEqual("skipped", gates["expert_routes"]["status"])
            self.assertIn("no layer<N>_routes lines", gates["expert_routes"]["detail"])
            self.assertTrue(
                any("expert_routes" in entry for entry in summary["skipped_gates"])
            )

    def test_optional_gates_absent_in_reference_are_recorded_skips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            bundle = workspace.write_bundle()
            rc, out_dir, calls, _, _ = self._verify(workspace, bundle)
            self.assertEqual(0, rc)
            for _, environment, _ in calls:
                overrides_free = "TERMITE_DEBUG_GPT_LAST_ROW" not in environment or (
                    environment.get("TERMITE_DEBUG_GPT_LAST_ROW")
                    == dict(MOD.os.environ).get("TERMITE_DEBUG_GPT_LAST_ROW")
                )
                self.assertTrue(overrides_free, "no route debug env without reference routes")
            summary = self._summary(out_dir)
            # 2 lanes x 1 engine x (expert_routes + logit_samples)
            self.assertEqual(4, len(summary["skipped_gates"]))
            gates = summary["lanes"]["f32"]["engines"]["metal_compact"]["gates"]
            self.assertEqual("skipped", gates["expert_routes"]["status"])
            self.assertEqual("skipped", gates["logit_samples"]["status"])

    def test_logit_gate_tolerance_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            bundle = workspace.write_bundle(logit_samples=REFERENCE_LOGIT_SAMPLES)
            off_by_too_much = "logit_sample: 0 1.300000\nlogit_sample: 5 -0.500000\n"
            rc, out_dir, _, _, stderr = self._verify(
                workspace, bundle, extra_for_lane=lambda lane: off_by_too_much
            )
            self.assertEqual(1, rc)
            self.assertIn("abs error", stderr)
            summary = self._summary(out_dir)
            gates = summary["lanes"]["f16"]["engines"]["metal_compact"]["gates"]
            self.assertEqual("failed", gates["logit_samples"]["status"])

    def test_output_token_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            bundle = workspace.write_bundle()

            def wrong_tokens(command, environment, log_path, timeout):
                lane = command[command.index("--cache-dtype") + 1]
                return (
                    "prompt_token_ids: 2 9259\n"
                    "token_ids: 111 999\n"
                    f"cache_dtype_effective: requested={lane} effective={lane}\n"
                )

            out_dir = workspace.out_dir()
            stderr = io.StringIO()
            with _patched(_run=wrong_tokens):
                with contextlib.redirect_stderr(stderr):
                    rc = MOD.main(
                        [
                            "verify-reference",
                            "--antfly-bin",
                            str(workspace.antfly),
                            "--model",
                            str(workspace.model),
                            "--reference",
                            str(bundle),
                            "--out-dir",
                            str(out_dir),
                        ]
                    )
            self.assertEqual(1, rc)
            self.assertIn("output mismatch", stderr.getvalue())
            self.assertIn("index 1: expected 222, got 999", stderr.getvalue())

    def test_verify_rejects_llama_bin_as_usage_error(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = MOD.main(
                [
                    "verify-reference",
                    "--antfly-bin",
                    "/nonexistent/antfly",
                    "--model",
                    "/nonexistent/model.gguf",
                    "--reference",
                    "/nonexistent/bundle",
                    "--out-dir",
                    "/nonexistent/out",
                    "--llama-bin",
                    "/nonexistent/llama",
                ]
            )
        self.assertEqual(2, rc)
        self.assertIn("never launches llama.cpp", stderr.getvalue())

    def test_verify_accepts_reference_json_path_and_rejects_bad_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            bundle = workspace.write_bundle()
            rc, _, calls, _, stderr = self._verify(workspace, bundle / "reference.json")
            self.assertEqual(0, rc, stderr)
            self.assertEqual(2, len(calls))

            reference_path = bundle / "reference.json"
            document = json.loads(reference_path.read_text(encoding="utf-8"))
            document["schema"] = "antfly.other.v9"
            reference_path.write_text(json.dumps(document), encoding="utf-8")
            rc, _, calls, _, stderr = self._verify(workspace, bundle)
            self.assertEqual(1, rc)
            self.assertEqual(0, len(calls))
            self.assertIn("unsupported reference schema", stderr)


class RecordThenVerifyRoundTripTests(unittest.TestCase):
    def test_record_then_full_matrix_verify_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = RecordVerifyWorkspace(Path(temp_dir))
            bundle = workspace.out_dir()

            def fake_llama_run(command, environment, log_path, timeout):
                Path(log_path).write_text("raw llama log\n", encoding="utf-8")
                return _llama_output(LLAMA_MARKERS)

            with _patched(
                _run=fake_llama_run,
                _llama_version=lambda llama: "version: 4242 (32703b42)",
            ):
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = MOD.main(
                        [
                            "record-reference",
                            "--llama-bin",
                            str(workspace.llama),
                            "--model",
                            str(workspace.model),
                            "--out-dir",
                            str(bundle),
                            "--tokens",
                            "2",
                        ]
                    )
            self.assertEqual(0, rc)

            recorder = AntflyRunRecorder(
                lambda lane: MATCHING_ROUTE_LINES + MATCHING_LOGIT_LINES
            )
            out_dir = workspace.out_dir()
            stdout = io.StringIO()
            with _patched(_run=recorder):
                with contextlib.redirect_stdout(stdout):
                    rc = MOD.main(
                        [
                            "verify-reference",
                            "--antfly-bin",
                            str(workspace.antfly),
                            "--model",
                            str(workspace.model),
                            "--reference",
                            str(bundle),
                            "--out-dir",
                            str(out_dir),
                            "--full-matrix",
                        ]
                    )
            self.assertEqual(0, rc)
            self.assertEqual(6, len(recorder.calls))
            self.assertIn("exact parity passed:", stdout.getvalue())
            summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["passed"])
            self.assertEqual([], summary["skipped_gates"])


if __name__ == "__main__":
    unittest.main()
