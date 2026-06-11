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

Skipped unless the local GLiNER2 model bundle and parity Python environment
are available (this test trains a full DeBERTa-v3 encoder on CPU and builds
the Zig training CLI, so it is opt-in like the other real-model gates).

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

REPO_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_DIR = REPO_ROOT / "zig" / "pkg" / "inference"
COMPARE_SCRIPT = INFERENCE_DIR / "scripts" / "compare_gliner2_lora_python_zig.py"
TRAIN_FIXTURE = INFERENCE_DIR / "testdata" / "gliner2_ner_smoke.jsonl"
ALL_TASK_FIXTURE = INFERENCE_DIR / "testdata" / "gliner2_all_task_smoke.jsonl"

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
