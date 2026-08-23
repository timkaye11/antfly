from __future__ import annotations

import json
import os
import stat
import struct
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import qualify_gemma4_metal_resume as qualifier


FAKE_ANTFLY = r'''#!/usr/bin/env python3
import json
import os
import pathlib
import struct
import sys
import time

args = sys.argv[1:]
def value(flag):
    return args[args.index(flag) + 1]

out = pathlib.Path(value("--out"))
checkpoint = pathlib.Path(value("--checkpoint-path"))
epochs = int(value("--epochs"))
boundary = int(value("--checkpoint-every-epochs"))
resume = "--resume" in args

def encode(fields):
    chunks = []
    for field in fields:
        chunks.extend(float((field >> (index * 16)) & 0xffff) for index in range(4))
    return chunks

def write_checkpoint(epoch):
    fields = [2, epoch, epoch, epoch, 0, 1, 42, epoch, 0, epoch, 100 + epoch, 0, 1, 2, 3, 4, 0, 0]
    values = encode(fields)
    fingerprint = [float(index) for index in range(32)]
    state_bytes = struct.pack("<72f", *values)
    fingerprint_bytes = struct.pack("<32f", *fingerprint)
    header = {
        "__metadata__": {"format": "pt"},
        "__trainer_state_v2": {"dtype": "F32", "shape": [72], "data_offsets": [0, len(state_bytes)]},
        "__run_fingerprint": {"dtype": "F32", "shape": [32], "data_offsets": [len(state_bytes), len(state_bytes) + len(fingerprint_bytes)]},
    }
    raw = json.dumps(header, separators=(",", ":")).encode()
    padding = b" " * ((8 - len(raw) % 8) % 8)
    temporary = checkpoint.with_suffix(".tmp")
    temporary.write_bytes(struct.pack("<Q", len(raw) + len(padding)) + raw + padding + state_bytes + fingerprint_bytes)
    os.replace(temporary, checkpoint)

def metric(epoch):
    return {
        "examples_seen": 1, "supervised_tokens_seen": 8,
        "teacher_examples_seen": 0, "teacher_supervised_tokens_seen": 0,
        "mean_teacher_temperature": 0.0, "average_loss": 2.0 - epoch / 10,
        "mean_grad_norm": 0.5, "optimizer_steps": 1,
        "graph_executor_steps": 1, "graph_executor_fallback_steps": 0,
        "graph_executor_native_partitions": 0, "graph_executor_unsupported_ops": 0,
        "graph_executor_interpreter_fallbacks": 0, "graph_executor_true_host_outputs": 0,
        "metal_optimizer_steps": 1,
    }

if out.name == "interrupted-unpublished":
    write_checkpoint(boundary)
    while True:
        time.sleep(1)

start = boundary if resume else 0
for epoch in range(start + 1, epochs + 1):
    write_checkpoint(epoch)
out.mkdir()
(out / "adapter_model.safetensors").write_bytes(b"identical-final-adapter")
model = pathlib.Path(value("--model"))
sidecars = ["adapter_config.json", "antfly_finetune_manifest.json", "run_manifest.json", "train_eval_report.json"]
if model.suffix.lower() != ".gguf":
    sidecars.extend(["tokenizer.json", "tokenizer_config.json"])
for name in sidecars:
    (out / name).write_text("{}\n")
report = {
    "contract_version": "training_report/v1",
    "artifact_family_version": "gemma4_lora/v1alpha1",
    "task": "gemma4_lora_train_eval",
    "report": {
        "trainer_kind": "real_autodiff_causal_lm_v1",
        "backend_kind": "metal",
        "run_fingerprint_sha256": "a" * 64,
        "epochs": epochs,
        "checkpoint_resume": {"enabled": resume, "start_epoch": start},
        "epoch_history": [metric(epoch) for epoch in range(start + 1, epochs + 1)],
        "after": {"average_loss": 1.0},
    },
}
(out / "training_report.json").write_text(json.dumps(report))
config = {"training": {"run_fingerprint_sha256": "a" * 64}, "backend_policy": {"selected": "metal"}}
(out / "training_config.json").write_text(json.dumps(config))
'''


class ResumeQualificationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.binary = self.root / "antfly"
        self.binary.write_text(FAKE_ANTFLY, encoding="utf-8")
        self.binary.chmod(self.binary.stat().st_mode | stat.S_IXUSR)
        self.model = self.root / "model"
        self.adapter = self.root / "adapter"
        self.model.mkdir()
        self.adapter.mkdir()
        (self.model / "config.json").write_text("{}\n", encoding="utf-8")
        (self.adapter / "adapter_model.safetensors").write_bytes(b"seed")
        self.train = self.root / "train.json"
        self.eval = self.root / "eval.json"
        self.train.write_text("{}\n", encoding="utf-8")
        self.eval.write_text("{}\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def args(self) -> Namespace:
        return Namespace(
            binary=self.binary,
            model=self.model,
            adapter=self.adapter,
            train_prepared=self.train,
            eval_prepared=self.eval,
            output_dir=self.root / "qualification",
            epochs=2,
            interrupt_after_epoch=1,
            learning_rate=1e-4,
            max_examples=1,
            eval_max_examples=1,
            grad_accum=1,
            max_grad_norm=1.0,
            activation_checkpoint_interval=0,
            seed=42,
            timeout_seconds=5.0,
            poll_seconds=0.01,
            experimental_gguf_qlora=False,
        )

    def test_end_to_end_exact_resume_contract(self) -> None:
        report = qualifier.qualify(self.args())
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["checkpoints"]["interrupted_boundary"]["epoch_index"], 1)
        self.assertEqual(report["checkpoints"]["resumed_final"]["epoch_index"], 2)
        self.assertTrue((self.root / "qualification" / "qualification_report.json").is_file())

    def test_existing_output_fails_closed(self) -> None:
        args = self.args()
        args.output_dir.mkdir()
        with self.assertRaisesRegex(qualifier.ContractError, "already exists"):
            qualifier.qualify(args)

    def test_direct_gguf_model_artifact_is_snapshotted_and_accepted(self) -> None:
        gguf = self.root / "model.gguf"
        gguf.write_bytes(b"GGUF-pinned-model")
        args = self.args()
        args.model = gguf
        args.experimental_gguf_qlora = True
        report = qualifier.qualify(args)
        self.assertEqual(report["inputs"]["model"]["kind"], "gguf-file")
        self.assertEqual(report["inputs"]["model"]["sha256"], qualifier._sha256(gguf))
        self.assertEqual(report["inputs"]["model"]["snapshot_sha256"], qualifier._snapshot_digest(qualifier._tree_snapshot(gguf)))
        self.assertTrue(report["contract"]["experimental_direct_gguf_qlora"])
        self.assertEqual(
            report["contract"]["strict_metal_environment"]["ANTFLY_EXPERIMENTAL_GEMMA4_GGUF_QLORA"],
            "1",
        )
        self.assertFalse((self.root / "qualification" / "uninterrupted" / "tokenizer.json").exists())

    def test_direct_gguf_requires_explicit_experimental_admission(self) -> None:
        gguf = self.root / "model.gguf"
        gguf.write_bytes(b"GGUF-pinned-model")
        args = self.args()
        args.model = gguf
        with self.assertRaisesRegex(qualifier.ContractError, "--experimental-gguf-qlora"):
            qualifier.qualify(args)

    def test_checkpoint_parser_rejects_wrong_schema(self) -> None:
        path = self.root / "bad.safetensors"
        fields = [99] + [0] * 17
        values = []
        for field in fields:
            values.extend(float((field >> (index * 16)) & 0xffff) for index in range(4))
        data = struct.pack("<72f", *values)
        header = json.dumps({
            "__trainer_state_v2": {"dtype": "F32", "shape": [72], "data_offsets": [0, len(data)]}
        }, separators=(",", ":")).encode()
        padding = b" " * ((8 - len(header) % 8) % 8)
        path.write_bytes(struct.pack("<Q", len(header) + len(padding)) + header + padding + data)
        with self.assertRaisesRegex(qualifier.ContractError, "unsupported schema"):
            qualifier.inspect_training_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
