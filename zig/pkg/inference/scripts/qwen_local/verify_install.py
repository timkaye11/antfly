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

"""Check a copied Qwen install against the pinned manifests and record the host.

Run on the destination after copying the model directories. It verifies every
manifest file's layout, size and SHA-256, records chip, memory, OS and where
each model is stored, and, given the rebuilt runners, the memory plan each
derives on this host (including Metal's recommended working set). Read-only:
model files are never modified.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

from machine_profile import dry_run, physical_memory_bytes

SCRIPTS = Path(__file__).resolve().parent.parent
MANIFESTS = {
    "flash": SCRIPTS / "qwen4exp" / "artifact.json",
    "27b": SCRIPTS / "qwen35_streamed" / "artifact.json",
}
# The file each runner is pointed at, relative to its model directory.
ENTRY = {
    "flash": "UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
    "27b": "Qwen3.8-27B-UD-Q4_K_M.gguf",
}
# Sidecars the runners read and pin; the rest of a manifest is provenance.
REQUIRED_SIDECARS = (
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
)


def sha256(path, progress=None):
    hasher = hashlib.sha256()
    done = 0
    with open(path, "rb", buffering=0) as stream:
        while block := stream.read(8 << 20):
            hasher.update(block)
            done += len(block)
            if progress:
                progress(done)
    return hasher.hexdigest()


def check_files(root, manifest, hash_files=True, log=None):
    """Compare `root` with a manifest. Returns per-file rows; never raises on a mismatch."""
    rows = []
    for item in manifest["files"]:
        path = Path(root) / item["path"]
        optional = (
            not item["path"].endswith(".gguf")
            and Path(item["path"]).name not in REQUIRED_SIDECARS
        )
        row = {"path": item["path"], "expected_bytes": item["bytes"], "ok": False}
        rows.append(row)
        if not path.is_file():
            row["error"] = "missing"
            # Provenance files do not affect execution.
            row["ok"] = optional
            continue
        row["bytes"] = path.stat().st_size
        if row["bytes"] != item["bytes"]:
            row["error"] = "size mismatch"
            continue
        if hash_files:
            started = time.monotonic()
            if log:
                log(f"hashing {item['path']} ({item['bytes']} bytes)")
            row["sha256"] = sha256(path)
            row["hash_seconds"] = time.monotonic() - started
            if row["sha256"] != item["sha256"]:
                row["error"] = "sha256 mismatch"
                continue
        row["ok"] = True
    return rows


def output(*command):
    try:
        return subprocess.check_output(command, text=True, timeout=30).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def storage(path):
    """Where a model lives: bus protocol and free space decide Flash throughput."""
    info = {"path": str(path)}
    df = output("df", "-k", str(path))
    if df:
        fields = df.splitlines()[-1].split()
        info.update(
            device=fields[0], available_bytes=int(fields[3]) * 1024, mount=fields[-1]
        )
        details = output("diskutil", "info", fields[0]) or ""
        for line in details.splitlines():
            key, _, value = line.partition(":")
            if key.strip() in (
                "Protocol",
                "Device Location",
                "Solid State",
                "File System Personality",
            ):
                info[key.strip().lower().replace(" ", "_")] = value.strip()
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--flash-dir",
        type=Path,
        help="directory containing UD-Q4_K_XL/ and the Flash sidecars",
    )
    p.add_argument(
        "--27b-dir",
        dest="b27_dir",
        type=Path,
        help="directory containing the 27B GGUF and sidecars",
    )
    p.add_argument(
        "--flash-binary", type=Path, help="rebuilt antfly-qwen4exp, for its memory plan"
    )
    p.add_argument(
        "--27b-binary",
        dest="b27_binary",
        type=Path,
        help="rebuilt antfly-qwen35-streamed",
    )
    p.add_argument(
        "--sizes-only", action="store_true", help="skip hashing (layout and sizes only)"
    )
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    roots = {"flash": a.flash_dir, "27b": a.b27_dir}
    binaries = {"flash": a.flash_binary, "27b": a.b27_binary}
    if not any(roots.values()):
        p.error("pass --flash-dir and/or --27b-dir")
    report = {
        "schema": 1,
        "passed": False,
        "hashed": not a.sizes_only,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "model": output("sysctl", "-n", "hw.model"),
            "chip": output("sysctl", "-n", "machdep.cpu.brand_string"),
            "physical_memory_bytes": physical_memory_bytes(),
            "macos": output("sw_vers", "-productVersion"),
        },
        "models": {},
    }
    for family, root in roots.items():
        if root is None:
            continue
        manifest = json.loads(MANIFESTS[family].read_text())
        rows = check_files(
            root, manifest, not a.sizes_only, log=lambda m: print(m, flush=True)
        )
        entry = {
            "directory": str(root),
            "entry_point": str(root / ENTRY[family]),
            "storage": storage(root),
            "files": rows,
            "passed": all(row["ok"] for row in rows),
        }
        if binaries[family] is not None:
            # The plan this host derives, without loading weights.
            try:
                entry["plan"] = dry_run(binaries[family], root / ENTRY[family], family)
            except (OSError, subprocess.SubprocessError, ValueError) as error:
                entry["plan_error"] = f"{type(error).__name__}: {error}"
                entry["passed"] = False
        report["models"][family] = entry
        for row in rows:
            if not row["ok"]:
                print(family, row["path"], row.get("error"), file=sys.stderr)
    report["passed"] = all(entry["passed"] for entry in report["models"].values())
    a.output.write_text(json.dumps(report, indent=2) + "\n")
    for family, entry in report["models"].items():
        plan = entry.get("plan", {})
        profile = plan.get("machine_profile", {})
        print(
            family,
            "ok" if entry["passed"] else "FAILED",
            entry["storage"].get("protocol", "unknown bus"),
            f"metal_working_set={profile.get('metal_recommended_working_set_bytes')}",
            f"process_cap={plan.get('process_cap_bytes')}",
            f"expert_cache={plan.get('expert_cache_bytes')}"
            if family == "flash"
            else "",
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
