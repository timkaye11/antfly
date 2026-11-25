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

"""Strict adapter for antfly.context-release-build.v1/v2/v3/v4 evidence.

This module never upgrades historical/scoped receipts. It owns the complete
archive, toolchain, command, resource, log, test, and executable audit and returns
only fields derived from the validated receipt.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import stat
import stat as stat_mod
import math
import zipfile



MAX_SOURCE_FILES = 16384
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 768 * 1024 * 1024

MAX_REPORT = 16 * 1024 * 1024

def require(ok, reason):
    if not ok:
        raise ValueError(reason)

def read_json(path, expected_sha256=None):
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_REPORT + 1)
    require(len(raw) <= MAX_REPORT, "evidence exceeds byte limit")
    if expected_sha256 is not None:
        require(hashlib.sha256(raw).hexdigest() == expected_sha256, "evidence digest mismatch")
    return decode_json(raw)

def decode_json(raw):
    def bad(value):
        raise ValueError("nonfinite JSON number: " + value)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(raw, parse_constant=bad, object_pairs_hook=pairs)

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()

def campaign_file(root, entry, label, max_bytes=None):
    """Resolve and hash one immutable campaign input without trusting its label."""
    require(type(entry) is dict and set(entry) == {"path", "sha256"}, "missing " + label + " file provenance")
    require(isinstance(entry["path"], str) and entry["path"], "invalid " + label + " file path")
    require(isinstance(entry["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]),
            "invalid " + label + " file digest")
    relative = Path(entry["path"])
    require(not relative.is_absolute() and ".." not in relative.parts, label + " path escapes campaign")
    unresolved = root / relative
    require(not unresolved.is_symlink(), label + " file must not be a symlink")
    path = unresolved.resolve()
    require(path.is_relative_to(root) and path.is_file(), "unverifiable " + label + " file")
    if max_bytes is not None:
        require(path.stat().st_size <= max_bytes, label + " exceeds byte limit")
    require(sha(path) == entry["sha256"], label + " file digest mismatch")
    return path

CLI_BUILD_MAX_LOG_BYTES = 256 * 1024 * 1024

FULL_RELEASE_BUILD_FIELDS_V1 = {
    "schema", "evidence_scope", "release_eligible", "serving_qualified", "created_utc", "sealed_utc",
    "head", "zig_version", "tools", "binary", "build", "resource", "stdout", "stderr", "tests", "source",
    "source_archive", "source_verified_after_build",
}

FULL_RELEASE_BUILD_FIELDS_V2 = FULL_RELEASE_BUILD_FIELDS_V1 | {"compiler_profile"}

FULL_RELEASE_BUILD_FIELDS = FULL_RELEASE_BUILD_FIELDS_V1

FULL_RELEASE_SCHEMA_V1 = "antfly.context-release-build.v1"

FULL_RELEASE_SCHEMA_V2 = "antfly.context-release-build.v2"

FULL_RELEASE_SCHEMA_V3 = "antfly.context-release-build.v3"

FULL_RELEASE_SCHEMA_V4 = "antfly.context-release-build.v4"

FULL_RELEASE_EXPECTED_STRIPPED_PROFILE = {"strip_debug_info": True, "have_error_return_tracing": False}

FULL_RELEASE_COMPILER_PROFILE_PREFIX = "context_compiler_profile: "

FULL_RELEASE_SOURCE_NOTE = (
    "Complete Git-visible Zig tree inputs are archived; deleted binds tracked removals; Git-ignored caches and outputs are excluded; gitlinks bind clean or uninitialized submodules."
)

FULL_RELEASE_ARCHIVE_MANIFEST = "FULL-RELEASE-SOURCE.json"

FULL_RELEASE_MAX_SOURCE_FILES = MAX_SOURCE_FILES

FULL_RELEASE_MAX_ARCHIVE_BYTES = MAX_ARCHIVE_BYTES

FULL_RELEASE_MAX_UNCOMPRESSED_BYTES = MAX_UNCOMPRESSED_BYTES

FULL_RELEASE_COMMAND = [
    "/usr/bin/env", "ANTFLY_ZIG_MAX_RSS=8589934592", "python3", "tools/run_bounded_zig_build.py",
    "build", "inference-test-context", "--verbose", "-j1", "-Dmetal=true", "-Dcuda=false", "-Donnx=false",
    "-Doptimize=ReleaseFast",
]

FULL_RELEASE_STRIPPED_COMMAND = [*FULL_RELEASE_COMMAND, "-Dstrip=true"]

FULL_RELEASE_CURRENT_STRIPPED_COMMAND = [
    argument for argument in FULL_RELEASE_COMMAND if argument != "-Donnx=false"
] + ["-Dstrip=true"]

FULL_RELEASE_16G_STRIPPED_COMMAND = [
    "ANTFLY_ZIG_MAX_RSS=17179869184" if argument == "ANTFLY_ZIG_MAX_RSS=8589934592" else argument
    for argument in FULL_RELEASE_CURRENT_STRIPPED_COMMAND
] + ["-Dcontext-test-compile-memory-mb=16384"]

FULL_RELEASE_CLEARED_ENV_PREFIXES = ["ANTFLY_", "TERMITE_", "MTL_", "METAL_", "CUDA_", "GGML_", "ZIG_"]

FULL_RELEASE_ENVIRONMENT = {
    "inherit_non_runtime_environment": True,
    "cleared_prefixes": FULL_RELEASE_CLEARED_ENV_PREFIXES,
    "set_by_command": {"ANTFLY_ZIG_MAX_RSS": "8589934592"},
}

FULL_RELEASE_16G_ENVIRONMENT = {
    **FULL_RELEASE_ENVIRONMENT,
    "set_by_command": {"ANTFLY_ZIG_MAX_RSS": "17179869184"},
}

def _full_release_resource_contract(receipt, expected_command):
    require(type(receipt) is dict and receipt.get("schema") == "antfly.benchmark_resources.v1" and
            receipt.get("command") == expected_command and receipt.get("pass") is True and
            receipt.get("violation") is None and "cleanup_error" not in receipt,
            "full release build resource guard failed")
    require(type(receipt.get("returncode")) is int and not isinstance(receipt["returncode"], bool) and
            receipt["returncode"] == 0 and type(receipt.get("samples")) is int and
            not isinstance(receipt["samples"], bool) and receipt["samples"] > 0 and
            type(receipt.get("pid")) is int and not isinstance(receipt["pid"], bool) and
            receipt["pid"] > 0, "invalid full release build completion")
    limits = receipt.get("limits")
    require(type(limits) is dict and set(limits) == {"timeout_seconds", "max_rss_mib", "min_free_percent",
                                                     "max_swap_growth_mib", "min_disk_free_mib"},
            "invalid full release build limits")
    timeout = number(limits["timeout_seconds"], "full release build timeout", 1)
    rss = number(limits["max_rss_mib"], "full release build RSS cap", 1)
    free = number(limits["min_free_percent"], "full release build memory floor")
    swap = number(limits["max_swap_growth_mib"], "full release build swap cap")
    disk = number(limits["min_disk_free_mib"], "full release build disk floor")
    require(timeout <= 3600 and rss <= 16384 and free >= 15 and 0 <= swap <= 512 and disk >= 4096,
            "insufficient full release build limits")
    peak = number(receipt.get("peak_group_rss_mib"), "full release measured RSS")
    minimum_free = number(receipt.get("min_free_percent"), "full release measured free memory")
    initial_free = number(receipt.get("initial_free_percent"), "full release initial free memory")
    swap_growth = number(receipt.get("swapout_growth_mib"), "full release measured swap growth")
    minimum_disk = number(receipt.get("min_disk_free_mib"), "full release measured disk")
    elapsed = number(receipt.get("elapsed_seconds"), "full release build elapsed")
    require(0 <= peak <= rss and free <= minimum_free <= initial_free <= 100 and
            0 <= swap_growth <= swap and minimum_disk >= disk and 0 < elapsed <= timeout,
            "full release build exceeded resource limits")

def full_release_build_contract(root, receipt_path, expected_binary):
    """Audit source/build evidence only; serving, numerical, quality, and soak remain separate gates."""
    receipt = read_json(receipt_path)
    require(type(receipt) is dict, "invalid full release build receipt")
    require(receipt.get("release_eligible") is not False and
            receipt.get("evidence_scope") == "full_context_release_build",
            "release-ineligible or scoped build receipt cannot support full release")
    schema = receipt.get("schema")
    expected_fields = (FULL_RELEASE_BUILD_FIELDS_V1 if schema == FULL_RELEASE_SCHEMA_V1 else
                       FULL_RELEASE_BUILD_FIELDS_V2 if schema in {FULL_RELEASE_SCHEMA_V2,
                                                                  FULL_RELEASE_SCHEMA_V3,
                                                                  FULL_RELEASE_SCHEMA_V4} else None)
    require(expected_fields is not None and set(receipt) == expected_fields and
            receipt["release_eligible"] is True and receipt["serving_qualified"] is False and
            receipt["source_verified_after_build"] is True,
            "invalid full release build evidence")
    require(isinstance(receipt["created_utc"], str) and receipt["created_utc"] and
            isinstance(receipt["sealed_utc"], str) and receipt["sealed_utc"] and
            re.fullmatch(r"[0-9a-f]{40}", receipt["head"] or "") and
            isinstance(receipt["zig_version"], str) and receipt["zig_version"],
            "invalid full release build identity")
    tools = receipt["tools"]
    require(type(tools) is dict and set(tools) == {"python3", "zig"},
            "invalid full release toolchain")
    for name, entry in tools.items():
        require(type(entry) is dict and set(entry) == {"path", "sha256", "bytes", "version"} and
                isinstance(entry["path"], str) and Path(entry["path"]).is_absolute() and
                re.fullmatch(r"[0-9a-f]{64}", entry["sha256"] or "") and
                type(entry["bytes"]) is int and not isinstance(entry["bytes"], bool) and entry["bytes"] > 0 and
                isinstance(entry["version"], str) and entry["version"],
                "invalid full release tool identity")
        tool_path = Path(entry["path"])
        require(not tool_path.is_symlink() and tool_path.is_file() and os.access(tool_path, os.X_OK) and
                tool_path.stat().st_size == entry["bytes"] and sha(tool_path) == entry["sha256"],
                "full release build tool changed")
    require(receipt["zig_version"] == tools["zig"]["version"], "full release Zig version mismatch")

    binary = receipt["binary"]
    require(type(binary) is dict and set(binary) == {"path", "sha256", "bytes"} and
            binary["path"] == expected_binary["path"] and binary["sha256"] == expected_binary["sha256"] and
            type(binary["bytes"]) is int and not isinstance(binary["bytes"], bool) and binary["bytes"] > 0,
            "full release build receipt does not bind executable")
    binary_path = campaign_file(root, {"path": binary["path"], "sha256": binary["sha256"]},
                                "full release binary", 1024 * 1024 * 1024)
    require(binary_path.stat().st_size == binary["bytes"] and os.access(binary_path, os.X_OK),
            "invalid full release executable")

    build = receipt["build"]
    expected_build_fields = ({"cwd", "command", "target", "optimize", "metal", "cuda", "environment"}
                             if schema in {FULL_RELEASE_SCHEMA_V3, FULL_RELEASE_SCHEMA_V4} else
                             {"cwd", "command", "target", "optimize", "metal", "cuda", "onnx", "environment"})
    require(type(build) is dict and set(build) == expected_build_fields and
            build["cwd"] == "zig" and build["target"] == "inference-test-context" and
            build["optimize"] == "ReleaseFast" and build["metal"] is True and
            build["cuda"] is False and
            (schema in {FULL_RELEASE_SCHEMA_V3, FULL_RELEASE_SCHEMA_V4} or build["onnx"] is False) and
            build["environment"] == (FULL_RELEASE_16G_ENVIRONMENT if schema == FULL_RELEASE_SCHEMA_V4 else FULL_RELEASE_ENVIRONMENT),
            "invalid full release build configuration")
    command = build["command"]
    expected_command = (FULL_RELEASE_COMMAND if schema == FULL_RELEASE_SCHEMA_V1 else
                        FULL_RELEASE_STRIPPED_COMMAND if schema == FULL_RELEASE_SCHEMA_V2 else
                        FULL_RELEASE_CURRENT_STRIPPED_COMMAND if schema == FULL_RELEASE_SCHEMA_V3 else
                        FULL_RELEASE_16G_STRIPPED_COMMAND)
    require(command == expected_command, "invalid full release guarded command")
    if schema in {FULL_RELEASE_SCHEMA_V2, FULL_RELEASE_SCHEMA_V3, FULL_RELEASE_SCHEMA_V4}:
        compiler_profile = receipt["compiler_profile"]
        require(type(compiler_profile) is dict and
                set(compiler_profile) == {"strip_debug_info", "have_error_return_tracing"} and
                type(compiler_profile.get("strip_debug_info")) is bool and
                type(compiler_profile.get("have_error_return_tracing")) is bool and
                compiler_profile == FULL_RELEASE_EXPECTED_STRIPPED_PROFILE,
                "invalid stripped compiler profile")

    refs = {}
    for field, maximum, allow_empty in (("resource", MAX_REPORT, False),
                                         ("stdout", CLI_BUILD_MAX_LOG_BYTES, True),
                                         ("stderr", CLI_BUILD_MAX_LOG_BYTES, True)):
        ref = receipt[field]
        refs[field] = campaign_file(root, ref, "full release build " + field, maximum)
        require(allow_empty or refs[field].stat().st_size > 0, "empty full release build " + field)
    resource = read_json(refs["resource"], receipt["resource"]["sha256"])
    _full_release_resource_contract(resource, command)
    if schema == FULL_RELEASE_SCHEMA_V4:
        require(resource["limits"]["max_rss_mib"] == 16384, "V4 compiler watchdog differs from declared budget")
    stderr = refs["stderr"].read_text(encoding="utf-8", errors="replace")
    require(not any(line.startswith("error:") for line in stderr.splitlines()),
            "full release compiler log contains an error")
    profile_rows = [line[len(FULL_RELEASE_COMPILER_PROFILE_PREFIX):] for line in stderr.splitlines()
                    if line.startswith(FULL_RELEASE_COMPILER_PROFILE_PREFIX)]
    if schema in {FULL_RELEASE_SCHEMA_V2, FULL_RELEASE_SCHEMA_V3, FULL_RELEASE_SCHEMA_V4}:
        require(len(profile_rows) == 1, "stripped full release requires exactly one compiler profile observation")
        try:
            observed_profile = json.loads(profile_rows[0])
        except json.JSONDecodeError as error:
            raise ValueError("invalid compiler profile observation") from error
        require(type(observed_profile) is dict and
                set(observed_profile) == {"schema_version", "strip_debug_info", "have_error_return_tracing"} and
                type(observed_profile.get("schema_version")) is int and
                not isinstance(observed_profile["schema_version"], bool) and observed_profile["schema_version"] == 1 and
                type(observed_profile.get("strip_debug_info")) is bool and
                type(observed_profile.get("have_error_return_tracing")) is bool and
                {key: observed_profile[key] for key in FULL_RELEASE_EXPECTED_STRIPPED_PROFILE} ==
                receipt["compiler_profile"], "compiler profile receipt/log mismatch")

    tests = receipt["tests"]
    require(type(tests) is dict and set(tests) == {"selected", "passed", "skipped", "failed", "leaked"} and
            all(type(tests[field]) is int and not isinstance(tests[field], bool) and tests[field] >= 0
                for field in tests) and tests["selected"] > 0 and tests["passed"] > 0 and
            tests["passed"] + tests["skipped"] == tests["selected"] and
            tests["failed"] == 0 and tests["leaked"] == 0,
            "full release compiled tests did not pass")
    summaries = re.findall(r"(?m)^(\d+) selected; (\d+) passed; (\d+) skipped(?:; (\d+) failed; (\d+) leaked)?\.$", stderr)
    require(summaries, "full release build log lacks test summary")
    selected, passed, skipped, failed, leaked = summaries[-1]
    require((int(selected), int(passed), int(skipped), int(failed or 0), int(leaked or 0)) ==
            tuple(tests[field] for field in ("selected", "passed", "skipped", "failed", "leaked")),
            "full release test summary mismatch")

    source = receipt["source"]
    require(type(source) is dict and set(source) == {"contract", "files", "deleted", "gitlinks"} and
            source["contract"] == FULL_RELEASE_SOURCE_NOTE and type(source["files"]) is dict and
            len(source["files"]) <= FULL_RELEASE_MAX_SOURCE_FILES and type(source["deleted"]) is list and
            len(source["deleted"]) <= FULL_RELEASE_MAX_SOURCE_FILES and type(source["gitlinks"]) is dict and
            len(source["gitlinks"]) <= FULL_RELEASE_MAX_SOURCE_FILES,
            "invalid full release source manifest")
    def source_name(name):
        if not (isinstance(name, str) and name.startswith("zig/") and not Path(name).is_absolute() and
                all(part not in ("", ".", "..") for part in Path(name).parts)):
            return False
        return True
    require(all(source_name(name) and re.fullmatch(r"[0-9a-f]{64}", digest or "")
                for name, digest in source["files"].items()) and
            all(source_name(name) for name in source["deleted"]) and
            all(source_name(name) and re.fullmatch(r"[0-9a-f]{40}", object_id or "")
                for name, object_id in source["gitlinks"].items()) and
            not set(source["files"]).intersection(source["deleted"]) and
            not set(source["gitlinks"]).intersection(set(source["files"]) | set(source["deleted"])),
            "invalid full release source entry")

    archive = receipt["source_archive"]
    require(type(archive) is dict and set(archive) == {"path", "sha256", "bytes", "files"} and
            type(archive["bytes"]) is int and not isinstance(archive["bytes"], bool) and
            0 < archive["bytes"] <= FULL_RELEASE_MAX_ARCHIVE_BYTES and
            archive["files"] == len(source["files"]), "invalid full release source archive")
    archive_path = campaign_file(root, {"path": archive["path"], "sha256": archive["sha256"]},
                                 "full release source archive", FULL_RELEASE_MAX_ARCHIVE_BYTES)
    require(archive_path.stat().st_size == archive["bytes"], "full release source archive size mismatch")
    # The archive is sealed before the compiler starts. Build/test/binary evidence
    # is bound by the outer receipt after the same source snapshot is rehashed.
    manifest = {"head": receipt["head"], "source": source, "build": build, "tools": tools}
    expected_manifest = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    expected_members = {FULL_RELEASE_ARCHIVE_MANIFEST, *source["files"]}
    with zipfile.ZipFile(archive_path) as bundle:
        infos = bundle.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)) and set(names) == expected_members and
                sum(info.file_size for info in infos) <= FULL_RELEASE_MAX_UNCOMPRESSED_BYTES,
                "full release source archive membership mismatch")
        for info in infos:
            require(not info.is_dir() and
                    (info.filename == FULL_RELEASE_ARCHIVE_MANIFEST or source_name(info.filename)) and
                    not stat_mod.S_ISLNK(info.external_attr >> 16),
                    "unsafe full release source archive member")
        require(bundle.read(FULL_RELEASE_ARCHIVE_MANIFEST) == expected_manifest,
                "full release source archive manifest mismatch")
        for name, digest in source["files"].items():
            require(hashlib.sha256(bundle.read(name)).hexdigest() == digest,
                    "full release archived source digest mismatch")
    return {"release_build_evidence": True, "head": receipt["head"],
            "binary_sha256": binary["sha256"], "source_archive_sha256": archive["sha256"]}

def number(value, label, minimum=0):
    require(type(value) in (int, float) and math.isfinite(value) and value >= minimum, "invalid " + label)
    return value

def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(receipt_path, expected_binary_path=None, require_campaign_root=None):
    """Return validated build identity; every authority field comes from strict release evidence."""
    supplied_receipt = Path(receipt_path)
    _require(not supplied_receipt.is_symlink(), "invalid full release receipt path")
    receipt_path = supplied_receipt.resolve(strict=True)
    _require(receipt_path.is_file() and receipt_path.stat().st_size <= 16 * 1024 * 1024,
             "invalid full release receipt path")
    receipt = json.loads(receipt_path.read_text())
    _require(type(receipt) is dict and receipt.get("schema") in {"antfly.context-release-build.v1", "antfly.context-release-build.v2", "antfly.context-release-build.v3", "antfly.context-release-build.v4"} and
             receipt.get("evidence_scope") == "full_context_release_build" and
             receipt.get("release_eligible") is True and receipt.get("serving_qualified") is False and
             receipt.get("source_verified_after_build") is True,
             "historical, scoped, or release-ineligible build receipt")
    binary = receipt.get("binary")
    _require(type(binary) is dict and set(binary) == {"path", "sha256", "bytes"} and
             isinstance(binary["path"], str) and re.fullmatch(r"[0-9a-f]{64}", binary.get("sha256", "")) and
             type(binary["bytes"]) is int and not isinstance(binary["bytes"], bool) and binary["bytes"] > 0,
             "invalid full release binary binding")
    root = receipt_path.parent.resolve()
    if require_campaign_root is not None:
        _require(root == Path(require_campaign_root).resolve(strict=True),
                 "full release evidence and campaign must share one sealed root")
    # Lazy import prevents capture/release import cycles. This is the sole full
    # evidence authority, rather than a second weaker validator.
    audited = full_release_build_contract(root, receipt_path,
                                                   {"path": binary["path"], "sha256": binary["sha256"]})
    binary_path = (root / binary["path"]).resolve(strict=True)
    _require(binary_path.is_relative_to(root) and binary_path.is_file() and not binary_path.is_symlink() and
             os.access(binary_path, os.X_OK) and binary_path.stat().st_size == binary["bytes"] and
             digest(binary_path) == binary["sha256"], "full release executable changed")
    if expected_binary_path is not None:
        _require(Path(expected_binary_path).resolve(strict=True) == binary_path,
                 "selected executable is not the full release executable")
    return {
        "receipt_path": receipt_path,
        "receipt_sha256": digest(receipt_path),
        "binary_path": binary_path,
        "binary_sha256": audited["binary_sha256"],
        "source_archive_sha256": audited["source_archive_sha256"],
        "head": audited["head"],
        "source": receipt["source"],
        "receipt": receipt,
    }

BUNDLE_FIELDS = ("binary", "resource", "stdout", "stderr", "source_archive")


def _destination_path(root, relative_text, created_directories):
    relative = Path(relative_text)
    _require(not relative.is_absolute() and relative.parts and
             all(part not in ("", ".", "..") for part in relative.parts),
             "unsafe build artifact ref")
    parent = root
    for part in relative.parts[:-1]:
        parent = parent / part
        if parent.exists() or parent.is_symlink():
            _require(not parent.is_symlink() and parent.is_dir(),
                     "unsafe build evidence destination parent")
        else:
            parent.mkdir()
            created_directories.append(parent)
    target = parent / relative.parts[-1]
    _require(not target.exists() and not target.is_symlink(),
             "build evidence destination collision")
    return target


def _copy_exact(source, target, copied):
    # Register ownership immediately after exclusive creation so an exception
    # during the first read/write still removes the partial evidence file.
    with source.open("rb") as incoming:
        outgoing = target.open("xb")
        copied.append(target)
        try:
            for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        finally:
            outgoing.close()


def materialize_bundle(receipt_path, campaign_root, receipt_name="full-release-build.json"):
    """Copy an exact, closed release evidence graph; never rewrite the receipt."""
    source = validate(receipt_path)
    supplied_destination = Path(campaign_root)
    _require(not supplied_destination.is_symlink(), "invalid build evidence destination")
    destination = supplied_destination.resolve(strict=True)
    _require(destination.is_dir(), "invalid build evidence destination")
    receipt = source["receipt"]
    copied = []
    created_directories = []
    try:
        for field in BUNDLE_FIELDS:
            ref = receipt[field]
            _require(type(ref) is dict and isinstance(ref.get("path"), str), "invalid build artifact ref")
            relative = Path(ref["path"])
            _require(not relative.is_absolute() and all(part not in ("", ".", "..") for part in relative.parts),
                     "unsafe build artifact ref")
            src = (source["receipt_path"].parent / relative).resolve(strict=True)
            dst = _destination_path(destination, ref["path"], created_directories)
            _copy_exact(src, dst, copied)
            if field == "binary":
                dst.chmod(src.stat().st_mode)
        target = _destination_path(destination, receipt_name, created_directories)
        _copy_exact(source["receipt_path"], target, copied)
        result = validate(target, require_campaign_root=destination)
        _require(result["receipt_sha256"] == source["receipt_sha256"], "copied full release receipt changed")
        return result
    except Exception:
        for path in reversed(copied):
            try: path.unlink()
            except FileNotFoundError: pass
        for path in reversed(created_directories):
            try: path.rmdir()
            except (FileNotFoundError, OSError): pass
        raise


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate")
    check.add_argument("--receipt", type=Path, required=True)
    check.add_argument("--binary", type=Path)
    copy = sub.add_parser("materialize")
    copy.add_argument("--receipt", type=Path, required=True)
    copy.add_argument("--campaign-root", type=Path, required=True)
    args = parser.parse_args()
    result = (validate(args.receipt, args.binary) if args.command == "validate" else
              materialize_bundle(args.receipt, args.campaign_root))
    print(json.dumps({"receipt": str(result["receipt_path"]),
                      "receipt_sha256": result["receipt_sha256"],
                      "binary": str(result["binary_path"]),
                      "binary_sha256": result["binary_sha256"],
                      "source_archive_sha256": result["source_archive_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
