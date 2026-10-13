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

"""Harness view of the runners' machine profile.

The binaries derive the process envelope, the expert cache and the context
limit (`src/runtime/machine_profile.zig`). The harness never repeats that
arithmetic: it passes explicit overrides through unchanged and checks its
evidence against what the binary reports it admitted.
"""

import json
import os
import subprocess
from pathlib import Path

MIB = 1024**2
# Mirrors machine_profile.zig `max_context`; test_evidence.py checks they agree.
MAX_CONTEXT = 4096
VERIFY_MODES = ("full", "stamp")
FLAGS = {"27b": "--experimental-qwen35-streamed", "flash": "--experimental-qwen4exp"}


def physical_memory_bytes():
    try:
        return int(subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True))
    except (OSError, subprocess.CalledProcessError, ValueError):
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def explicit_mb(name, value, physical=None):
    """Validate an explicit MiB override; None means "let the binary derive"."""
    if value is None:
        return None
    limit = (physical_memory_bytes() if physical is None else physical) // MIB
    if type(value) is not int or not 1 <= value <= limit:
        raise ValueError(f"{name} must be 1..{limit} MiB on this host")
    return value


def budget_arguments(family, cache_mb=None, combined_mb=None, verify=None):
    """Runner flags for the given overrides. Omitted flags are derived."""
    result = []
    if combined_mb is not None:
        result += ["--combined-budget-mb", str(combined_mb)]
    if cache_mb is not None:
        if family != "flash":
            raise ValueError("an expert cache budget requires family flash")
        result += ["--moe-cache-mb", str(cache_mb)]
    if verify is not None:
        if verify not in VERIFY_MODES:
            raise ValueError(f"verify must be one of {VERIFY_MODES}")
        result += ["--verify", verify]
    return result


def dry_run(binary, model, family, cache_mb=None, combined_mb=None, context=None):
    """Ask the binary for the plan it would admit, without loading weights."""
    command = [
        str(Path(binary).resolve()),
        str(model),
        "-",
        FLAGS[family],
        "--backend",
        "metal",
        "--speculation-policy",
        "off",
        "--dry-run",
    ]
    if context is not None:
        command += ["--context-size", str(context)]
    command += budget_arguments(family, cache_mb, combined_mb)
    output = subprocess.run(
        command, capture_output=True, text=True, timeout=120, check=True
    ).stdout
    return json.loads(output)
