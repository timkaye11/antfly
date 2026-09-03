#!/usr/bin/env python3

from __future__ import annotations

import json
import math
import signal
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
from safetensors.numpy import save_file

import export_gemma4_lora_zig_oracle as oracle
from gemma4_oracle_contract import ContractError, prefixed_sha256


MODULE = "model.layers.0.self_attn.q_proj"
SOURCE_A = f"{MODULE}.weight.lora_A.weight"
SOURCE_B = f"{MODULE}.weight.lora_B.weight"
SLOT_A = f"{MODULE}.weight.lora_A"
SLOT_B = f"{MODULE}.weight.lora_B"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _encode_u64_fields(fields: list[int]) -> np.ndarray[Any, np.dtype[np.float32]]:
    return np.asarray(
        [(value >> (chunk * 16)) & 0xFFFF for value in fields for chunk in range(4)],
        dtype=np.float32,
    )


class CaptureFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.capture_dir = root / "capture"
        self.capture_dir.mkdir()
        self.candidate_dir = root / "candidate"
        self.candidate_dir.mkdir()
        self.source_dir = root / "source"
        self.source_dir.mkdir()
        self.request_path = root / "request.json"
        self.steps = 1
        self.seed = 42
        self.prepared = {"labels": [-100, -100, 3]}
        self.initial = {
            SOURCE_A: np.asarray([[1.0, 2.0]], dtype=np.float32),
            SOURCE_B: np.asarray([[0.0], [0.0]], dtype=np.float32),
        }
        self.updated = {
            SOURCE_A: np.asarray([[0.99999, 1.99998]], dtype=np.float32),
            SOURCE_B: np.asarray([[-0.001], [0.001]], dtype=np.float32),
        }
        save_file(self.initial, str(self.source_dir / "adapter_model.safetensors"))
        save_file(self.updated, str(self.candidate_dir / "adapter_model.safetensors"))
        self.source_adapter = self._adapter(self.source_dir, self.initial)
        self.candidate_adapter = self._adapter(self.candidate_dir, self.updated)
        self.request = {
            "schema_version": oracle.REQUEST_SCHEMA_VERSION,
            "implementation": {
                "version": "test",
                "executable_sha256": "sha256:" + "1" * 64,
                "source_revision": "2" * 40,
                "backend": "native",
                "metal_device": None,
            },
            "bindings": {
                "oracle_lock_sha256": "sha256:" + "3" * 64,
                "model_key": "gemma-4-E2B-it",
                "model_revision": "4" * 40,
                "local_artifact_sha256": "sha256:" + "5" * 64,
                "base_model_sha256": "6" * 64,
                "initial_adapter_sha256": self.source_adapter["adapter_model_sha256"],
                "train_prepared_sha256": "sha256:" + "7" * 64,
                "source_dataset_sha256": "8" * 64,
                "example_index": 0,
                "target_preset": "peft-qv",
                "rank": 1,
                "alpha": 2.0,
                "target_count": 1,
            },
            "training": {
                "optimizer": "adamw",
                "seed": self.seed,
                "steps": self.steps,
                "learning_rate": 0.001,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "grad_accum_steps": 1,
                "supervised_token_normalization": "mean",
                "dropout": 0.0,
                "use_cache": False,
            },
        }
        _write_json(self.request_path, self.request)
        self.targets = [self._target(SOURCE_A, SLOT_A, [1, 2]), self._target(SOURCE_B, SLOT_B, [2, 1])]
        self.gradients = {
            f"gradient::{SLOT_A}": np.zeros((1, 2), dtype=np.float32),
            f"gradient::{SLOT_B}": np.asarray([[0.25], [-0.5]], dtype=np.float32),
        }
        save_file(self.gradients, str(self.capture_dir / "raw_gradients.safetensors"))
        checkpoint: dict[str, np.ndarray[Any, np.dtype[np.float32]]] = {
            "__trainer_counters": _encode_u64_fields([1, 1]),
            "__trainer_state_v2": _encode_u64_fields([2, 1, 1, 1, 0, 1, 42, 1, 0, 1, 123, 0, 1, 2, 3, 4, 0, 0]),
        }
        for source_name, slot in ((SOURCE_A, SLOT_A), (SOURCE_B, SLOT_B)):
            checkpoint[f"weight::{slot}"] = self.updated[source_name]
            checkpoint[f"adam_m::{slot}"] = np.full(self.updated[source_name].shape, 0.1, dtype=np.float32)
            checkpoint[f"adam_v::{slot}"] = np.full(self.updated[source_name].shape, 0.01, dtype=np.float32)
            checkpoint[f"adam_step::{slot}"] = np.asarray([1], dtype=np.float32)
            checkpoint[f"grad_accum::{slot}"] = np.zeros(self.updated[source_name].shape, dtype=np.float32)
        save_file(checkpoint, str(self.capture_dir / "trainer_checkpoint.safetensors"))
        predictor = 1
        token_ids = oracle._stable_probe_token_ids(3, 17, predictor, self.seed)
        self.payload = {
            "schema_version": oracle.CAPTURE_SCHEMA_VERSION,
            "request_sha256": prefixed_sha256(self.request_path),
            "implementation": self.request["implementation"],
            "bindings": self.request["bindings"],
            "training": self.request["training"],
            "result": {
                "loss_history": [1.25],
                "raw_gradient_norm": math.sqrt(0.25**2 + 0.5**2),
                "supervised_tokens": 1,
                "logit_probes": [{
                    "predictor_position": predictor,
                    "target_token_id": 3,
                    "token_ids": token_ids,
                    "values": [float(index) / 10 for index in range(len(token_ids))],
                    "logsumexp": 2.5,
                }],
                "targets": self.targets,
                "execution": {
                    "optimizer_steps": 1,
                    "micro_batch_steps": 1,
                    "metal_optimizer_steps": 0,
                    "graph_executor_steps": 0,
                    "graph_executor_fallback_steps": 0,
                    "graph_executor_native_partitions": 0,
                    "graph_executor_unsupported_ops": 0,
                    "graph_executor_interpreter_fallbacks": 0,
                    "graph_executor_true_host_outputs": 0,
                },
            },
            "artifacts": {},
        }
        self.refresh_capture()

    @staticmethod
    def _target(source_name: str, slot: str, shape: list[int]) -> dict[str, Any]:
        return {
            "source_name": source_name,
            "trainer_slot_name": slot,
            "shape": shape,
            "gradient_storage_key": f"gradient::{slot}",
            "checkpoint_weight_storage_key": f"weight::{slot}",
            "checkpoint_m_storage_key": f"adam_m::{slot}",
            "checkpoint_v_storage_key": f"adam_v::{slot}",
        }

    @staticmethod
    def _adapter(root: Path, tensors: dict[str, np.ndarray[Any, np.dtype[np.float32]]]) -> dict[str, Any]:
        by_identity = {
            oracle.canonicalize_adapter_tensor_name(name): {
                "source_name": name,
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
                "values": value.reshape(-1).tolist(),
            }
            for name, value in tensors.items()
        }
        checkpoint = root / "adapter_model.safetensors"
        return {
            "checkpoint": str(checkpoint),
            "adapter_model_sha256": prefixed_sha256(checkpoint),
            "tensors": by_identity,
            "inventory": sorted(f"{module}:{role}" for module, role in by_identity),
        }

    def refresh_capture(self) -> None:
        raw = self.capture_dir / "raw_gradients.safetensors"
        checkpoint = self.capture_dir / "trainer_checkpoint.safetensors"
        candidate = self.candidate_dir / "adapter_model.safetensors"
        self.payload["artifacts"] = {
            "raw_gradients": {"path": raw.name, "sha256": prefixed_sha256(raw), "size_bytes": raw.stat().st_size},
            "trainer_checkpoint": {"path": checkpoint.name, "sha256": prefixed_sha256(checkpoint), "size_bytes": checkpoint.stat().st_size},
            "candidate_adapter_model": {"sha256": prefixed_sha256(candidate), "size_bytes": candidate.stat().st_size},
        }
        _write_json(self.capture_dir / "capture.json", self.payload)
        files = []
        for name in ("capture.json", "raw_gradients.safetensors", "trainer_checkpoint.safetensors"):
            path = self.capture_dir / name
            files.append({
                "name": name,
                "sha256": prefixed_sha256(path).removeprefix("sha256:"),
                "size_bytes": path.stat().st_size,
            })
        _write_json(self.capture_dir / "run_manifest.json", {
            "schema_version": oracle.RUN_MANIFEST_SCHEMA_VERSION,
            "status": "complete",
            "artifact_family_version": oracle.ARTIFACT_FAMILY_VERSION,
            "artifacts": files,
        })

    def validate(self) -> dict[str, Any]:
        return oracle.validate_zig_capture(
            self.capture_dir,
            self.request_path,
            self.request,
            self.candidate_dir,
            self.source_adapter,
            self.candidate_adapter,
            self.prepared,
            17,
        )


class ZigOracleExporterTests(unittest.TestCase):
    def test_child_is_terminated_when_the_wrapper_is_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            process = mock.Mock()
            process.pid = 12345
            process.poll.return_value = None
            process.wait.side_effect = [KeyboardInterrupt(), -signal.SIGTERM]
            with (
                mock.patch.object(oracle.subprocess, "Popen", return_value=process),
                mock.patch.object(oracle.os, "killpg") as killpg,
                self.assertRaises(KeyboardInterrupt),
            ):
                oracle._run_child(
                    ["unused"],
                    {},
                    root / "stdout.log",
                    root / "stderr.log",
                    1,
                )
            killpg.assert_called_once_with(process.pid, signal.SIGTERM)
            process.wait.assert_has_calls([mock.call(timeout=1), mock.call(timeout=10)])

    def test_child_environment_is_offline_and_sanitizes_training_controls(self) -> None:
        source = {
            "PATH": "/usr/bin",
            "TERMITE_METAL_DISABLE_LINEAR_CCE": "1",
            "ANTFLY_GEMMA4_GRPO_COMPILED_SAMPLING": "1",
            "ANTFLY_EXPERIMENTAL_GEMMA4_GGUF_QLORA": "1",
            "HF_HUB_OFFLINE": "0",
        }
        metal = oracle._oracle_environment(source, "metal")
        self.assertEqual(metal["PATH"], source["PATH"])
        self.assertNotIn("TERMITE_METAL_DISABLE_LINEAR_CCE", metal)
        self.assertNotIn("ANTFLY_GEMMA4_GRPO_COMPILED_SAMPLING", metal)
        self.assertNotIn("ANTFLY_EXPERIMENTAL_GEMMA4_GGUF_QLORA", metal)
        self.assertEqual(metal["HF_HUB_OFFLINE"], "1")
        self.assertEqual(
            {name: metal[name] for name in oracle.STRICT_METAL_ENV},
            oracle.STRICT_METAL_ENV,
        )

        native = oracle._oracle_environment(source, "native")
        self.assertFalse(any(name.startswith("TERMITE_") for name in native))

    def test_stable_probe_projection_matches_zig_fixture(self) -> None:
        self.assertEqual(
            oracle._stable_probe_token_ids(3, 17, 2, 42),
            [0, 1, 2, 3, 10, 12, 14, 15, 16],
        )

    def test_capture_validation_and_trace_tensor_packaging(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptureFixture(Path(temporary))
            capture = fixture.validate()
            trace_path = Path(temporary) / "trace.safetensors"
            targets, entries, grad_norm = oracle._build_trace_tensor_store(
                trace_path,
                capture,
                fixture.source_adapter,
                fixture.candidate_adapter,
            )
            self.assertEqual(len(targets), 2)
            self.assertEqual(len(entries), 10)
            self.assertEqual(grad_norm, fixture.payload["result"]["raw_gradient_norm"])
            self.assertTrue(trace_path.is_file())

    def test_capture_rejects_noncanonical_probe_projection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptureFixture(Path(temporary))
            fixture.payload["result"]["logit_probes"][0]["token_ids"][-1] = 13
            fixture.refresh_capture()
            with self.assertRaisesRegex(ContractError, "deterministic token projection"):
                fixture.validate()

    def test_capture_rejects_unbound_checkpoint_counter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptureFixture(Path(temporary))
            checkpoint_path = fixture.capture_dir / "trainer_checkpoint.safetensors"
            from safetensors import safe_open

            with safe_open(str(checkpoint_path), framework="np", device="cpu") as source:
                tensors = {name: source.get_tensor(name) for name in source.keys()}
            tensors["__trainer_counters"] = _encode_u64_fields([2, 1])
            save_file(tensors, str(checkpoint_path))
            fixture.refresh_capture()
            with self.assertRaisesRegex(ContractError, "counters differ"):
                fixture.validate()

    def test_capture_rejects_unexpected_publication_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CaptureFixture(Path(temporary))
            (fixture.capture_dir / "unexpected.txt").write_text("nope")
            with self.assertRaisesRegex(ContractError, "file set differs"):
                fixture.validate()


if __name__ == "__main__":
    unittest.main()
