#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

import compare_gemma4_grpo_prefix_adapters as comparison


class PrefixAdapterComparisonTests(unittest.TestCase):
    def _write_mlx_provenance(self, root: Path) -> tuple[Path, Path]:
        adapter = root / 'mlx.safetensors'
        adapter.write_bytes(b'matched adapter')
        adapter_digest = hashlib.sha256(adapter.read_bytes()).hexdigest()
        runner_digest = 'a' * 64
        report = root / 'mlx-report.json'
        report.write_text(json.dumps({
            'schema_version': 'antfly_gemma4_grpo_boolq_mlx_multitoken/v2',
            'status': 'diagnostic-lane-completed',
            'runner_sha256': f'sha256:{runner_digest}',
            'contract': {'execution_lane': 'trace-replay', 'executed_train_groups': 260},
            'mlx': {'trace_replay': {'training': {
                'mode': 'trace_replay', 'groups': 260, 'optimizer_steps': 242,
                'adapter_output': {
                    'path': str(adapter.resolve()), 'sha256': adapter_digest,
                    'tensor_count': 686,
                },
            }}},
        }))
        status = root / 'mlx-status.json'
        status.write_text(json.dumps({
            'schema_version': 'antfly_gemma4_guarded_execution/v1',
            'status': 'completed', 'returncode': 0,
            'runner_sha256_at_start': runner_digest,
            'runner_sha256_at_finish': runner_digest,
            'adapter_output_path': str(adapter.resolve()),
            'adapter_output_sha256': adapter_digest,
            'output_path': str(report.resolve()),
            'output_sha256': hashlib.sha256(report.read_bytes()).hexdigest(),
        }))
        return adapter, status

    def test_mlx_provenance_binds_exact_guarded_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            adapter, status = self._write_mlx_provenance(Path(temp))
            result = comparison.validate_mlx_run(status, adapter, 260, 242, 686)
            self.assertEqual(hashlib.sha256(adapter.read_bytes()).hexdigest(), result['adapter_sha256'])

    def test_mlx_provenance_rejects_status_report_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            adapter, status = self._write_mlx_provenance(root)
            payload = json.loads(status.read_text())
            payload['adapter_output_sha256'] = '0' * 64
            status.write_text(json.dumps(payload))
            with self.assertRaises(comparison.PrefixAdapterParityError):
                comparison.validate_mlx_run(status, adapter, 260, 242, 686)

    def test_identical_matched_updates_pass(self) -> None:
        name = 'model.language_model.layers.0.self_attn.q_proj.weight.lora_A.weight'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed = root / 'seed.safetensors'
            metal = root / 'metal.safetensors'
            mlx = root / 'mlx.safetensors'
            initial = np.arange(8, dtype=np.float32).reshape(2, 4)
            update = np.linspace(-1e-4, 1e-4, 8, dtype=np.float32).reshape(2, 4)
            save_file({name: initial}, str(seed))
            save_file({
                'weight::model.language_model.layers.0.self_attn.q_proj.weight.lora_A':
                    (initial + update).reshape(-1),
                '__trainer_counters': np.zeros(2, dtype=np.float32),
            }, str(metal))
            save_file({name: initial + update}, str(mlx))
            result = comparison.compare_adapter_files(seed, metal, mlx)
            self.assertTrue(result['passed'])
            self.assertEqual(1, result['metrics']['tensor_count'])
            self.assertAlmostEqual(1.0, result['metrics']['delta_cosine_similarity'])
            self.assertEqual(0.0, result['metrics']['delta_vector_l2_relative_error'])

    def test_inventory_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            seed = root / 'seed.safetensors'
            metal = root / 'metal.safetensors'
            mlx = root / 'mlx.safetensors'
            name = 'model.language_model.layers.0.self_attn.q_proj.weight.lora_A.weight'
            save_file({name: np.zeros((2, 4), dtype=np.float32)}, str(seed))
            save_file({'different': np.zeros((2, 4), dtype=np.float32)}, str(mlx))
            save_file({'__trainer_counters': np.zeros(2, dtype=np.float32)}, str(metal))
            with self.assertRaises(comparison.PrefixAdapterParityError):
                comparison.compare_adapter_files(seed, metal, mlx)


if __name__ == '__main__':
    unittest.main()
