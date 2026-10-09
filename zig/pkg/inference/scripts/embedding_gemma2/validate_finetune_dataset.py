#!/usr/bin/env python3
# Copyright 2026 Antfly, Inc.
# SPDX-License-Identifier: Apache-2.0
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

"""Validate pinned dataset snapshots, split leakage, and train-only mining."""

from __future__ import annotations
import argparse, json
from pathlib import Path
from finetune_contract import canonical_hash, load_json, validate_dataset_manifest

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest = load_json(args.manifest)
    counts = validate_dataset_manifest(manifest)
    report = {"schema":"antfly.embedding_gemma2.finetune_dataset_validation.v1","pass":True,"manifest_sha256":canonical_hash(manifest),"counts":counts}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True)+"\n")
    print(json.dumps(report, sort_keys=True))
    return 0
if __name__ == "__main__": raise SystemExit(main())

