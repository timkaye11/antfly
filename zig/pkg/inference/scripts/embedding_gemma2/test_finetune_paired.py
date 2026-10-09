#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exercise resident pairing without loading a model or requiring a GPU."""

import json
import queue
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from run_finetune_paired import Worker, checked_job


FAKE_TRAINER = """
import argparse,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--paired-control-dir',type=Path,required=True);a=p.parse_args()
def gate(prefix,step):
    path=a.paired_control_dir/f'{prefix}-{step:08d}'
    while not path.exists(): time.sleep(.001)
    path.unlink()
for step in range(2):
    print(json.dumps({'event':'ready','step':step}),flush=True)
    gate('grant',step)
    print(json.dumps({'event':'step','step':step,'step_ms':1.,'loss':.1,'trace_mode':False}),flush=True)
    gate('continue',step)
"""


class PairingTests(unittest.TestCase):
    def test_post_step_barrier_keeps_next_batch_out_of_partner_timing(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(Worker, "_monitor", lambda self: None):
            root = Path(temporary)
            script = root / "trainer.py"
            script.write_text(FAKE_TRAINER)
            workers = [Worker([sys.executable, str(script)], root / name, {}) for name in ("native", "reference")]
            deadline = time.monotonic() + 10
            try:
                for worker in workers:
                    worker.wait_event("ready", 0, deadline)
                workers[0].grant("grant", 0)
                workers[0].wait_event("step", 0, deadline)
                with self.assertRaises(queue.Empty):
                    workers[0].events.get(timeout=.05)
                workers[1].grant("grant", 0)
                workers[1].wait_event("step", 0, deadline)
                for worker in workers:
                    worker.grant("continue", 0)
                for worker in workers:
                    worker.wait_event("ready", 1, deadline)
                    worker.grant("grant", 1)
                    worker.wait_event("step", 1, deadline)
                    worker.grant("continue", 1)
                for worker in workers:
                    worker.finish(deadline)
            finally:
                for worker in workers:
                    worker.close()

    def test_job_admission_uses_partial_epoch_steps_and_rejects_different_math(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "manifest.json").write_text(json.dumps({"train_records": 5, "query_max_tokens": 32,
                "document_max_tokens": 64, "max_negatives": 1}))
            job = {"model_dir": "/base", "prepared_dir": str(root), "output_dir": "/unused",
                   "logical_batch_size": 2, "query_max_tokens": 32, "document_max_tokens": 64,
                   "max_negatives": 1}
            path = root / "job.json"
            path.write_text(json.dumps(job))
            self.assertEqual(checked_job(path, 3)["checkpoint_every_steps"], 4)
            with self.assertRaises(ValueError):
                checked_job(path, 4)
            job["optimizer"] = {"learning_rate": .01}
            path.write_text(json.dumps(job))
            with self.assertRaisesRegex(ValueError, "optimizer"):
                checked_job(path, 2)


if __name__ == "__main__":
    unittest.main()
