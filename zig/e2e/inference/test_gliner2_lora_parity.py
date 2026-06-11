# Copyright 2026 Antfly, Inc.
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

"""GLiNER2 LoRA fine-tuning Python/Zig parity gate.

Shells out to scripts/compare_gliner2_lora_python_zig.py with --strict on the
small NER smoke fixture. The compare harness trains the same model with the
same seed/data/LoRA-init on both the upstream Python GLiNER2 trainer and the
Zig train-gliner2-autodiff CLI, then asserts per-component and per-step loss
parity plus an adapter round-trip back into Python.

The native backend is the CI-trusted reference (≤1e-9 vs Python). The Metal
sibling (test_gliner2_lora_metal_strict_parity) runs the same config on the
Metal graph executor and additionally gates the metal-readiness signals
(real GPU dispatches, Metal optimizer backend, zero device-resident transfers,
no graph-executor fallback reasons, bounded interpreter fallbacks) — Metal
reproduces native step-for-step, so it passes at the same loss tolerances.

Skipped unless the local GLiNER2 model bundle and parity Python environment
are available (this test trains a full DeBERTa-v3 encoder on CPU and builds
the Zig training CLI, so it is opt-in like the other real-model gates); the
Metal sibling additionally skips off macOS.

Environment overrides:
    TERMITE_GLINER2_PARITY_MODEL_DIR   (default: /private/tmp/termite-models/gliner2)
    TERMITE_GLINER2_PARITY_PYTHON     (default: /private/tmp/gliner2-parity-venv/bin/python)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

def _find_repo_root() -> Path:
    # Walk up until the directory containing zig/pkg/inference is found, so the
    # gate resolves correctly regardless of how deep e2e/inference is nested
    # (the file lives at <repo>/zig/e2e/inference/).
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "zig" / "pkg" / "inference" / "scripts" / "compare_gliner2_lora_python_zig.py").exists():
            return parent
    # Fall back to the historical assumption (file three levels under the root).
    return here.parents[3]


REPO_ROOT = _find_repo_root()
INFERENCE_DIR = REPO_ROOT / "zig" / "pkg" / "inference"
COMPARE_SCRIPT = INFERENCE_DIR / "scripts" / "compare_gliner2_lora_python_zig.py"
TRAIN_FIXTURE = INFERENCE_DIR / "testdata" / "gliner2_ner_smoke.jsonl"
ALL_TASK_FIXTURE = INFERENCE_DIR / "testdata" / "gliner2_all_task_smoke.jsonl"

# Phase 5 parity-envelope fixtures: (fixture file, steps, fixture contains
# classification tasks). steps is chosen so steps * batch_size(=2) covers the
# fixture; the odd-sized fixtures (9 examples) deliberately leave a partial
# final batch of one example.
FULL_TASK_PARITY_FIXTURES = [
    pytest.param("gliner2_ner_smoke.jsonl", 1, False, id="entity-only"),
    pytest.param("gliner2_cls_smoke.jsonl", 5, True, id="cls"),
    pytest.param("gliner2_json_smoke.jsonl", 4, False, id="json"),
    pytest.param("gliner2_rel_smoke.jsonl", 4, False, id="rel"),
    pytest.param("gliner2_alltask_smoke.jsonl", 5, True, id="alltask"),
    pytest.param("gliner2_multicount_smoke.jsonl", 4, False, id="multicount"),
    pytest.param("gliner2_negative_smoke.jsonl", 5, True, id="negative"),
]

MODEL_DIR = Path(os.environ.get("TERMITE_GLINER2_PARITY_MODEL_DIR", "/private/tmp/termite-models/gliner2"))
PARITY_PYTHON = Path(os.environ.get("TERMITE_GLINER2_PARITY_PYTHON", "/private/tmp/gliner2-parity-venv/bin/python"))


def _skip_reason() -> str | None:
    if not (MODEL_DIR / "model.safetensors").exists():
        return f"GLiNER2 model bundle not found at {MODEL_DIR}"
    if not PARITY_PYTHON.exists():
        return f"parity Python environment not found at {PARITY_PYTHON}"
    if shutil.which("zig") is None:
        return "zig compiler not on PATH"
    if not COMPARE_SCRIPT.exists():
        return f"compare script not found at {COMPARE_SCRIPT}"
    if not TRAIN_FIXTURE.exists():
        return f"train fixture not found at {TRAIN_FIXTURE}"
    return None


_reason = _skip_reason()
pytestmark = [
    pytest.mark.skipif(_reason is not None, reason=_reason or ""),
    pytest.mark.slow,
    pytest.mark.model_integration,
]


def test_gliner2_lora_python_zig_strict_parity(tmp_path: Path):
    out_dir = tmp_path / "gliner2-lora-parity"
    cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--strict",
        "--deterministic",
        "--model-dir", str(MODEL_DIR),
        "--python-model", str(MODEL_DIR),
        "--python-bin", str(PARITY_PYTHON),
        "--train-data", str(TRAIN_FIXTURE),
        "--out-dir", str(out_dir),
        "--zig-objective", "gliner2-total-loss",
        "--zig-backend", "native",
        "--steps", "1",
        "--batch-size", "2",
        "--seq-len", "64",
        "--max-span-width", "4",
        "--lora-rank", "4",
        "--lora-alpha", "8",
        "--span-loss-reduction", "sum",
        "--span-positive-weight", "1",
        "--span-negative-weight", "1",
        "--span-hard-negative-weight", "1",
        "--seed", "42",
        "--dump-preprocess-parity",
        "--loss-parity-tolerance", "1e-4",
        # The adapter round-trip compares trained adapter weights AND forward
        # logits. Adapter weights match to ~3.5e-10, but the span-score logits
        # pass through saturated pretrained projections that amplify f32
        # weight noise to ~1.7e-4, so the logit gate uses 5e-4 (the measured
        # noise floor is ~1.7e-4; real divergences are orders of magnitude
        # larger).
        "--adapter-roundtrip-tolerance", "5e-4",
        "--timeout-seconds", "1800",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=3000,
    )
    report_path = out_dir / "comparison_report.json"
    tail = proc.stdout[-8000:]
    assert proc.returncode == 0, f"strict parity run failed (exit {proc.returncode}):\n{tail}"
    assert report_path.exists(), f"comparison report missing at {report_path}:\n{tail}"

    summary = json.loads(report_path.read_text(encoding="utf-8"))["summary"]
    assert summary["python_returncode"] == 0
    assert summary["zig_returncode"] == 0
    strict_checks = summary.get("strict_checks", {})
    # Every comparison that ran must have passed under --strict.
    failed = {name: value for name, value in strict_checks.items() if value is False}
    assert not failed, f"strict parity checks failed: {failed}\n{tail}"
    # The headline comparisons must actually have run (not been skipped).
    for required in ("component_loss_parity_matches", "step_loss_parity_matches", "preprocess_parity_matches"):
        assert strict_checks.get(required) is True, (
            f"expected strict check '{required}' to run and pass, got {strict_checks.get(required)!r}\n{tail}"
        )


def _metal_skip_reason() -> str | None:
    # Metal is macOS-only; the harness builds the trainer with -Dmetal=true and
    # would fail (not skip) elsewhere, so gate the Metal gate on darwin plus the
    # shared bundle/python/zig requirements. TERMITE_GLINER2_PARITY_METAL forces
    # the gate on (=1) or off (=0) regardless of platform.
    override = os.environ.get("TERMITE_GLINER2_PARITY_METAL")
    if override == "0":
        return "Metal parity gate disabled via TERMITE_GLINER2_PARITY_METAL=0"
    if override != "1" and sys.platform != "darwin":
        return f"Metal backend is macOS-only (sys.platform={sys.platform})"
    return _skip_reason()


def test_gliner2_lora_metal_strict_parity(tmp_path: Path):
    """Metal sibling of test_gliner2_lora_python_zig_strict_parity (Phase 1).

    Runs the same proven entity-only single-step parity config on the Metal
    backend with the training graph executor enabled, and additionally gates the
    metal-readiness signals (real GPU command dispatches, Metal optimizer
    backend, zero device-resident transfers for trainables, empty graph-executor
    fallback reasons, and per-op interpreter fallbacks under an explicit
    ceiling). Metal reproduces the CI-trusted native backend step-for-step, so
    the loss/component/preprocess/round-trip checks pass at the same tolerances
    as the native gate; this test additionally proves the step genuinely ran on
    the GPU rather than silently falling back to the interpreter.
    """
    metal_reason = _metal_skip_reason()
    if metal_reason is not None:
        pytest.skip(metal_reason)
    out_dir = tmp_path / "gliner2-lora-metal-parity"
    cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--strict",
        "--deterministic",
        "--model-dir", str(MODEL_DIR),
        "--python-model", str(MODEL_DIR),
        "--python-bin", str(PARITY_PYTHON),
        "--train-data", str(TRAIN_FIXTURE),
        "--out-dir", str(out_dir),
        "--zig-objective", "gliner2-total-loss",
        "--zig-backend", "metal",
        "--zig-training-graph-executor",
        "--steps", "1",
        "--batch-size", "2",
        "--seq-len", "64",
        "--max-span-width", "4",
        "--lora-rank", "4",
        "--lora-alpha", "8",
        "--span-loss-reduction", "sum",
        "--span-positive-weight", "1",
        "--span-negative-weight", "1",
        "--span-hard-negative-weight", "1",
        "--seed", "42",
        "--dump-preprocess-parity",
        "--loss-parity-tolerance", "1e-4",
        # Same logit noise-floor rationale as the native strict test above.
        "--adapter-roundtrip-tolerance", "5e-4",
        "--timeout-seconds", "1800",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=3000,
    )
    report_path = out_dir / "comparison_report.json"
    tail = proc.stdout[-8000:]
    assert proc.returncode == 0, f"strict Metal parity run failed (exit {proc.returncode}):\n{tail}"
    assert report_path.exists(), f"comparison report missing at {report_path}:\n{tail}"

    summary = json.loads(report_path.read_text(encoding="utf-8"))["summary"]
    assert summary["python_returncode"] == 0
    assert summary["zig_returncode"] == 0
    strict_checks = summary.get("strict_checks", {})
    failed = {name: value for name, value in strict_checks.items() if value is False}
    assert not failed, f"strict Metal parity checks failed: {failed}\n{tail}"
    # The headline parity comparisons plus the metal-readiness gates must all
    # have RUN and PASSED (not been skipped) — a skipped metal-readiness check
    # would mean the run was not actually a graph-executor Metal step.
    required = (
        "component_loss_parity_matches",
        "step_loss_parity_matches",
        "preprocess_parity_matches",
        "metal_manifest_backend_is_metal",
        "metal_optimizer_backend_is_metal",
        "metal_device_resident_transfers_zero",
        "metal_finite_step_loss",
        "metal_graph_executor_dispatches_nonzero",
        "metal_graph_executor_fallback_reasons_empty",
        "metal_interpreter_fallbacks_within_threshold",
    )
    for check in required:
        assert strict_checks.get(check) is True, (
            f"expected strict check '{check}' to run and pass on Metal, "
            f"got {strict_checks.get(check)!r}\n{tail}"
        )


def test_gliner2_lora_all_task_multi_step_roundtrip(tmp_path: Path):
    """3-step all-task training parity with an lr-scaled adapter round-trip gate.

    Multi-step (steps > 1) round-trips cannot reuse the 1-step tolerances.
    After step 1 the two implementations' adapter weights differ by inherent
    f32 noise; on the second forward pass that noise perturbs gradients whose
    true magnitude sits below the noise floor, flipping their sign between
    implementations. AdamW normalizes any nonzero gradient element to a
    ~lr-scale update (|update| = lr * m_hat / (sqrt(v_hat) + eps) -> ~0.744*lr
    on a parameter's first effective step, -> lr asymptotically), so each
    sign-flipped noise element can legitimately diverge by up to ~2*lr per
    optimizer step even when both implementations are correct. This is pure
    noise amplification, not a formula bug: the per-step optimizer-parity dump
    (--dump-optimizer-parity) shows gradients matching to ~1e-5, Adam m/v
    tracking those gradients exactly, and per-parameter step counts identical.

    Tolerances (lr=1e-3, steps=3, measured on the optimizer-parity runs):
      * weights:  2*lr*steps = 6e-3.  Measured noise floor 2.7e-3; the
        pre-fix lora_A step-count-lag bug produced 3.0e-3 in weights but its
        signature was the systematic per-tensor abs-sum drift (10-12x larger
        pre-fix) and 0.9 span-score deltas.
      * logits:   0.3.  Span scores route adapter-weight noise through
        saturated pretrained projections; measured floor is ~0.12 at 3 steps,
        while the pre-fix optimizer bug produced 0.91 — real divergences sit
        well above this gate, the noise floor sits ~2.5x below it.
      * per-step loss: 2.5e-4.  Step-2 loss is ~34.6 (count/structure loss on
        the all-task fixture); the measured cross-implementation delta is
        8.8e-5 and scales with the amplified weight noise, so the 1-step 1e-4
        gate is kept only for the strict 1-step test above.
    """
    out_dir = tmp_path / "gliner2-lora-all-task-multi-step"
    # 3 steps x batch 2 needs 6 examples; cycle the 4-line fixture.
    lines = [line for line in ALL_TASK_FIXTURE.read_text(encoding="utf-8").splitlines() if line.strip()]
    fixture = tmp_path / "gliner2_all_task_x6.jsonl"
    fixture.write_text("\n".join((lines * 2)[:6]) + "\n", encoding="utf-8")
    cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--strict",
        "--deterministic",
        "--model-dir", str(MODEL_DIR),
        "--python-model", str(MODEL_DIR),
        "--python-bin", str(PARITY_PYTHON),
        "--train-data", str(fixture),
        "--out-dir", str(out_dir),
        "--zig-objective", "gliner2-total-loss",
        "--zig-backend", "native",
        "--steps", "3",
        "--batch-size", "2",
        "--seq-len", "64",
        "--max-span-width", "4",
        "--lora-rank", "4",
        "--lora-alpha", "8",
        "--span-loss-reduction", "sum",
        "--span-positive-weight", "1",
        "--span-negative-weight", "1",
        "--span-hard-negative-weight", "1",
        "--seed", "42",
        "--dump-optimizer-parity",
        "--loss-parity-tolerance", "2.5e-4",
        "--adapter-roundtrip-tolerance", "0.3",
        "--adapter-roundtrip-weights-tolerance", "6e-3",
        "--timeout-seconds", "1800",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=3000,
    )
    report_path = out_dir / "comparison_report.json"
    tail = proc.stdout[-8000:]
    assert proc.returncode == 0, f"all-task multi-step parity run failed (exit {proc.returncode}):\n{tail}"
    assert report_path.exists(), f"comparison report missing at {report_path}:\n{tail}"

    report = json.loads(report_path.read_text(encoding="utf-8"))
    summary = report["summary"]
    assert summary["python_returncode"] == 0
    assert summary["zig_returncode"] == 0
    strict_checks = summary.get("strict_checks", {})
    failed = {name: value for name, value in strict_checks.items() if value is False}
    assert not failed, f"strict parity checks failed: {failed}\n{tail}"
    for required in (
        "component_loss_parity_matches",
        "step_loss_parity_matches",
        "preprocess_parity_matches",
        "adapter_roundtrip_ok",
    ):
        assert strict_checks.get(required) is True, (
            f"expected strict check '{required}' to run and pass, got {strict_checks.get(required)!r}\n{tail}"
        )
    # The optimizer-parity dump must show identical per-parameter Adam step
    # counts: the lora_A zero-grad step-count lag (fixed via
    # syncZeroGradLoraOptimizerSteps in train_gliner2_autodiff.zig) produced
    # python=k / zig=k-1 lags here before the fix.
    optimizer_parity = report.get("optimizer_parity", {})
    assert optimizer_parity.get("ran") is True, f"optimizer parity dump missing\n{tail}"
    for row in optimizer_parity.get("steps", []):
        assert row["step_count_mismatch_count"] == 0, (
            f"Adam per-parameter step counts diverged at step {row['step']}: "
            f"{row['step_count_mismatches']}\n{tail}"
        )


@pytest.mark.parametrize(("fixture_name", "steps", "has_classifications"), FULL_TASK_PARITY_FIXTURES)
def test_gliner2_lora_full_task_strict_parity(
    tmp_path: Path, fixture_name: str, steps: int, has_classifications: bool
):
    """Phase 5 parity-envelope gates: per-task-type fixtures under
    --strict --require-full-task-parity (native backend).

    The flag converts the previously warning-only/scoped comparisons into
    failing checks: the component-loss, preprocessing, and (for fixtures with
    classification tasks) classification-debug comparisons must RUN and match,
    and the trainable/objective parity warnings must be empty.
    """
    fixture = INFERENCE_DIR / "testdata" / fixture_name
    assert fixture.exists(), f"fixture missing at {fixture}"
    out_dir = tmp_path / f"gliner2-lora-full-task-{fixture.stem.replace('_', '-')}"
    cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        "--strict",
        "--require-full-task-parity",
        "--deterministic",
        "--model-dir", str(MODEL_DIR),
        "--python-model", str(MODEL_DIR),
        "--python-bin", str(PARITY_PYTHON),
        "--train-data", str(fixture),
        "--out-dir", str(out_dir),
        "--zig-objective", "gliner2-total-loss",
        "--zig-backend", "native",
        "--steps", str(steps),
        "--batch-size", "2",
        "--seq-len", "64",
        "--max-span-width", "4",
        "--lora-rank", "4",
        "--lora-alpha", "8",
        "--span-loss-reduction", "sum",
        "--span-positive-weight", "1",
        "--span-negative-weight", "1",
        "--span-hard-negative-weight", "1",
        "--seed", "42",
        "--dump-preprocess-parity",
        "--loss-parity-tolerance", "1e-4",
        # Same logit noise-floor rationale as the 1-step strict test above.
        "--adapter-roundtrip-tolerance", "5e-4",
        "--timeout-seconds", "1800",
    ]
    proc = subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=3000,
    )
    report_path = out_dir / "comparison_report.json"
    tail = proc.stdout[-8000:]
    assert proc.returncode == 0, (
        f"full-task strict parity run failed for {fixture_name} (exit {proc.returncode}):\n{tail}"
    )
    assert report_path.exists(), f"comparison report missing at {report_path}:\n{tail}"

    summary = json.loads(report_path.read_text(encoding="utf-8"))["summary"]
    assert summary["python_returncode"] == 0
    assert summary["zig_returncode"] == 0
    assert summary.get("require_full_task_parity") is True, tail
    strict_checks = summary.get("strict_checks", {})
    failed = {name: value for name, value in strict_checks.items() if value is False}
    assert not failed, f"strict parity checks failed for {fixture_name}: {failed}\n{tail}"
    required = [
        "component_loss_parity_matches",
        "step_loss_parity_matches",
        "preprocess_parity_matches",
        "valid_full_loss_parity",
        "full_task_component_loss_comparison_ran",
        "full_task_preprocess_comparison_ran",
        "full_task_loss_parity_comparison_ran",
        "full_task_trainable_parity",
        "full_task_objective_parity",
    ]
    if has_classifications:
        required.extend(["classification_debug_matches", "full_task_classification_debug_ran"])
    for check in required:
        assert strict_checks.get(check) is True, (
            f"expected strict check '{check}' to run and pass for {fixture_name}, "
            f"got {strict_checks.get(check)!r}\n{tail}"
        )
