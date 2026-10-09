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

"""Compare an uninterrupted run with a checkpoint/resume run."""

from __future__ import annotations
import argparse,json
from pathlib import Path
from finetune_contract import load_json

def main()->int:
    p=argparse.ArgumentParser();p.add_argument("--uninterrupted",type=Path,required=True);p.add_argument("--resumed",type=Path,required=True);p.add_argument("--report",type=Path,required=True);a=p.parse_args()
    uninterrupted,resumed=load_json(a.uninterrupted),load_json(a.resumed)
    left=uninterrupted.get("resume",{}).get("next_step_state_sha256");right=resumed.get("resume",{}).get("next_step_state_sha256")
    if not left or not right: raise ValueError("both runs must report next-step state hashes")
    provenance_equal=uninterrupted.get("provenance")==resumed.get("provenance")
    report={"schema":"antfly.embedding_gemma2.finetune_resume.v1","pass":left==right and provenance_equal,"uninterrupted_next_step_sha256":left,"resumed_next_step_sha256":right,"provenance_equal":provenance_equal}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n");print(json.dumps({"pass":report["pass"]}));return 0 if report["pass"] else 1
if __name__=="__main__":raise SystemExit(main())
