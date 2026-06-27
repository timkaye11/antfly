#!/usr/bin/env python3
"""Convert `antfly-inference cuda-info --smoke` output to stable JSON."""

import argparse
import csv
import json
import pathlib
from typing import Any, Optional, Tuple


def parse_scalar(value: str) -> Any:
    text = value.strip()
    lower = text.lower()
    if lower in {"true", "yes", "available"}:
        return True if lower != "available" else text
    if lower in {"false", "no", "unavailable"}:
        return False if lower != "unavailable" else text
    try:
        return int(text)
    except ValueError:
        return text


def step_status(steps_path: pathlib.Path, step_name: str) -> Tuple[Optional[str], Optional[str]]:
    if not steps_path.exists():
        return None, None
    with steps_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("step") == step_name:
                return row.get("status"), row.get("detail")
    return None, None


def parse_smoke_log(log_path: pathlib.Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_log": str(log_path),
        "capabilities": {},
        "smoke": {},
    }
    if not log_path.exists():
        result["parse_error"] = "missing_log"
        return result

    for raw_line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key.startswith("capability_"):
            result["capabilities"][key[len("capability_") :]] = bool(parse_scalar(value))
            continue
        if key == "smoke":
            parts = value.rsplit(" ", 1)
            if len(parts) == 2:
                result["smoke"][parts[0]] = parts[1]
            else:
                result["smoke"][value] = ""
            continue
        result[key] = parse_scalar(value)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--status")
    parser.add_argument("--detail")
    parser.add_argument("--steps")
    parser.add_argument("--step", default="cuda_smoke")
    args = parser.parse_args()

    log_path = pathlib.Path(args.log)
    out_path = pathlib.Path(args.out)
    status = args.status
    detail = args.detail
    if args.steps:
        step_status_value, step_detail = step_status(pathlib.Path(args.steps), args.step)
        status = status or step_status_value
        detail = detail or step_detail

    payload = parse_smoke_log(log_path)
    payload["status"] = status or ("ok" if log_path.exists() else "skip")
    if detail:
        payload["detail"] = detail

    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
