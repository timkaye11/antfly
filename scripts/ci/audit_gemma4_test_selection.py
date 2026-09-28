#!/usr/bin/env python3
"""Fail when a named inference test added by a change has no CI log entry."""

import argparse
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def added_tests(diff: str) -> list[tuple[str, str]]:
    source = ""
    cases = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            source = line[6:]
        match = re.match(r'^\+\s*test "([^"]+)"', line)
        if match and source.startswith("zig/pkg/inference/src/"):
            module = (
                source.removeprefix("zig/pkg/inference/src/")
                .removesuffix(".zig")
                .replace("/", ".")
            )
            cases.append((source, module + ".test." + match[1]))
    return cases


def audit(diff: str, log: str) -> dict:
    cases = added_tests(diff)
    missing = [name for _, name in cases if name + "..." not in log]
    skipped = [name for _, name in cases if name + "...SKIP" in log]
    return {
        "added_named_tests": len(cases),
        "missing": missing,
        "optional_skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="HEAD^")
    parser.add_argument("logs", nargs="+", type=Path)
    args = parser.parse_args()
    base = args.base or "HEAD^"
    if base == "0" * 40:
        base = "HEAD^"
    diff = subprocess.check_output(
        ["git", "diff", base, "--", "zig/pkg/inference/src"], cwd=ROOT, text=True
    )
    result = audit(diff, "\n".join(path.read_text() for path in args.logs))
    print(json.dumps(result, indent=2))
    return bool(result["missing"])


if __name__ == "__main__":
    raise SystemExit(main())
