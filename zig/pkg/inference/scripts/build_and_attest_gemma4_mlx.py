#!/usr/bin/env python3
"""Build and attest the pinned MLX native runtime with fetching disabled.

This command is intentionally standard-library-only.  It does not install
Python packages or fetch build dependencies.  Instead, the operator supplies a
strict manifest of local dependency trees and build tools whose digests are
verified before and after the build.  The pinned MLX checkout is built with
FetchContent fully disconnected, then the exact native runtime surface is
hashed using the benchmark contract's canonical inventory function.  This
command does not itself impose an OS-level network sandbox; production builds
must additionally run it in a deny-network worker.

The final attestation is published last, atomically, and without replacement.
Its ``build_command_sha256`` is the SHA-256 of the retained strict build
receipt, which contains the exact argv, environment, inputs, log identity, and
outputs needed to audit or replay the build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import pwd
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from gemma4_oracle_contract import (
    ContractError,
    LOCK_PATH,
    MLX_NATIVE_ARTIFACT_INVENTORY_SCHEMA_VERSION,
    canonical_mlx_native_artifact_inventory_sha256,
    load_json,
    load_lock,
    lock_digest,
    prefixed_sha256,
    require_exact_keys,
)


BUILD_INPUTS_SCHEMA_VERSION = "antfly_mlx_native_build_inputs/v1"
BUILD_COMMAND_SCHEMA_VERSION = "antfly_mlx_native_build_command/v1"
BUILD_RECEIPT_SCHEMA_VERSION = "antfly_mlx_native_build_receipt/v1"
DEPENDENCY_TREE_DOMAIN = b"antfly_mlx_native_dependency_tree/v1\0"

SCRIPT_PATH = Path(__file__).resolve()
RECEIPT_NAME = "antfly-native-build-receipt.json"
ATTESTATION_NAME = "antfly-native-build.json"
BUILD_LOG_NAME = "antfly-native-build.log"

# MLX 0.31.2 conditionally adds and installs the standalone JACCL library when
# its CMake configure sees macOS SDK 26.2 or newer. The locked native runtime
# deliberately admits that branch and binds libjaccl.dylib as a fourth role.
JACCL_SDK_MINIMUM = (26, 2)
GIT_PATH = Path("/usr/bin/git")
GENERATED_INSTALL_TREES = ("include", "share", "lib/cmake")
SYSTEM_BUILD_COMMANDS = {
    "bash": Path("/bin/bash"),
    "basename": Path("/usr/bin/basename"),
    "cat": Path("/bin/cat"),
    "env": Path("/usr/bin/env"),
    "git": GIT_PATH,
    "grep": Path("/usr/bin/grep"),
    "mkdir": Path("/bin/mkdir"),
    "sh": Path("/bin/sh"),
    "tail": Path("/usr/bin/tail"),
    "tr": Path("/usr/bin/tr"),
    "xcrun": Path("/usr/bin/xcrun"),
    "zsh": Path("/bin/zsh"),
}
SELECTED_XCODE_TOOLS = (
    "ar",
    "clang",
    "clang++",
    "install_name_tool",
    "ld",
    "metal",
    "ranlib",
    "strip",
)

DEPENDENCY_NAMES = ("fmt", "json", "metal_cpp", "nanobind")
TOOL_NAMES = ("clang", "clangxx", "cmake", "ninja", "python")
DEPENDENCY_SENTINELS = {
    "fmt": ("CMakeLists.txt", "include/fmt/format.h"),
    "json": ("CMakeLists.txt", "single_include/nlohmann/json.hpp"),
    "metal_cpp": ("Metal/Metal.hpp",),
    "nanobind": ("CMakeLists.txt", "include/nanobind/nanobind.h"),
}
TOOL_VERSION_ARGV = {
    "clang": ("--version",),
    "clangxx": ("--version",),
    "cmake": ("--version",),
    "ninja": ("--version",),
    "python": ("--version",),
}

_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)+$")
_MACHO_MAGICS = {
    b"\xce\xfa\xed\xfe",
    b"\xcf\xfa\xed\xfe",
    b"\xca\xfe\xba\xbe",
    b"\xca\xfe\xba\xbf",
    b"\xfe\xed\xfa\xce",
    b"\xfe\xed\xfa\xcf",
    b"\xbe\xba\xfe\xca",
    b"\xbf\xba\xfe\xca",
}


@dataclass(frozen=True)
class DependencyPin:
    name: str
    path: Path
    tree_sha256: str


@dataclass(frozen=True)
class ToolPin:
    name: str
    path: Path
    sha256: str


@dataclass(frozen=True)
class BuildInputs:
    path: Path
    sha256: str
    jobs: int
    macos_deployment_target: str
    macos_sdk_version: str
    metal_toolchain_identifier: str
    developer_dir: Path
    user_home: Path
    dependencies: tuple[DependencyPin, ...]
    tools: tuple[ToolPin, ...]


@dataclass(frozen=True)
class BuildPaths:
    source: Path
    output: Path
    package_root: Path
    receipt: Path
    attestation: Path
    log: Path


def _mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where}: expected object")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{where}: expected non-empty string")
    return value


def _sha256(value: Any, where: str) -> str:
    result = _string(value, where)
    if _SHA256.fullmatch(result) is None:
        raise ContractError(f"{where}: expected sha256:<64 lowercase hex>")
    return result


def _int(value: Any, where: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{where}: expected integer")
    if value < minimum or value > maximum:
        raise ContractError(f"{where}: expected value in [{minimum}, {maximum}]")
    return value


def _numeric_version(value: Any, where: str) -> tuple[str, tuple[int, ...]]:
    text = _string(value, where)
    if _VERSION.fullmatch(text) is None:
        raise ContractError(f"{where}: expected dotted numeric version")
    return text, tuple(int(part) for part in text.split("."))


def require_jaccl_inventory_sdk_version(version: str) -> None:
    _text, components = _numeric_version(version, "macos_sdk_version")
    padded = components + (0,) * max(0, len(JACCL_SDK_MINIMUM) - len(components))
    if padded < JACCL_SDK_MINIMUM:
        raise ContractError(
            "the locked MLX 0.31.2 native runtime requires macOS SDK 26.2 or "
            "newer so the JACCL libjaccl.dylib is present in the four-artifact inventory"
        )


def _canonical_path(value: Any, where: str, *, kind: str) -> Path:
    if isinstance(value, os.PathLike):
        path = Path(value)
    else:
        path = Path(_string(value, where))
    if not path.is_absolute():
        raise ContractError(f"{where}: path must be absolute")
    if path.is_symlink():
        raise ContractError(f"{where}: symbolic links are forbidden")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ContractError(f"{where}: could not resolve path: {exc}") from exc
    if resolved != path:
        raise ContractError(f"{where}: path must already be canonical ({resolved})")
    if kind == "file" and not path.is_file():
        raise ContractError(f"{where}: expected regular file")
    if kind == "directory" and not path.is_dir():
        raise ContractError(f"{where}: expected directory")
    return path


def _run_text(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    cwd: Path | None = None,
) -> str:
    completed = subprocess.run(
        list(argv),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=None if env is None else dict(env),
        cwd=None if cwd is None else str(cwd),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no diagnostic"
        raise ContractError(f"command failed ({' '.join(argv)}): {detail}")
    return completed.stdout.strip()


def dependency_tree_regular_paths(root: Path) -> tuple[Path, ...]:
    root = _canonical_path(root, "dependency root", kind="directory")
    paths: list[Path] = []
    for directory, directory_names, file_names in os.walk(root, topdown=True, followlinks=False):
        base = Path(directory)
        retained_directories: list[str] = []
        for name in sorted(directory_names):
            candidate = base / name
            if candidate.is_symlink():
                raise ContractError(f"dependency tree contains symbolic link: {candidate}")
            if name == ".git":
                continue
            retained_directories.append(name)
        directory_names[:] = retained_directories
        for name in sorted(file_names):
            candidate = base / name
            if candidate.is_symlink():
                raise ContractError(f"dependency tree contains symbolic link: {candidate}")
            if name == ".git":
                continue
            info = candidate.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ContractError(f"dependency tree contains special file: {candidate}")
            paths.append(candidate)
    if not paths:
        raise ContractError(f"dependency tree is empty: {root}")
    return tuple(sorted(paths, key=lambda path: path.relative_to(root).as_posix()))


def dependency_tree_identity(root: Path) -> dict[str, Any]:
    """Return a canonical, closed identity for a local source dependency.

    Git administrative data is excluded so an archive and an equivalent clean
    checkout hash identically.  Every other entry is closed: symbolic links,
    devices, sockets, and other special files fail rather than being skipped.
    """
    root = _canonical_path(root, "dependency root", kind="directory")
    entries: list[tuple[str, int, int, str]] = []
    for candidate in dependency_tree_regular_paths(root):
        info = candidate.stat()
        relative_text = candidate.relative_to(root).as_posix()
        executable = 1 if info.st_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) else 0
        entries.append((relative_text, executable, info.st_size, prefixed_sha256(candidate)))
    entries.sort(key=lambda entry: entry[0])
    encoded = json.dumps(
        entries,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = "sha256:" + hashlib.sha256(DEPENDENCY_TREE_DOMAIN + encoded).hexdigest()
    return {
        "path": str(root),
        "tree_sha256": digest,
        "file_count": len(entries),
        "total_bytes": sum(entry[2] for entry in entries),
    }


def _validate_dependency_pin(name: str, value: Any) -> DependencyPin:
    raw = _mapping(value, f"dependencies.{name}")
    require_exact_keys(raw, ("path", "tree_sha256"), where=f"dependencies.{name}")
    path = _canonical_path(raw["path"], f"dependencies.{name}.path", kind="directory")
    for sentinel in DEPENDENCY_SENTINELS[name]:
        candidate = path / sentinel
        if candidate.is_symlink() or not candidate.is_file():
            raise ContractError(f"dependencies.{name}: missing regular source file {sentinel}")
    return DependencyPin(
        name=name,
        path=path,
        tree_sha256=_sha256(raw["tree_sha256"], f"dependencies.{name}.tree_sha256"),
    )


def _validate_tool_pin(name: str, value: Any) -> ToolPin:
    raw = _mapping(value, f"tools.{name}")
    require_exact_keys(raw, ("path", "sha256"), where=f"tools.{name}")
    path = _canonical_path(raw["path"], f"tools.{name}.path", kind="file")
    if not os.access(path, os.X_OK):
        raise ContractError(f"tools.{name}.path must be executable")
    if path.read_bytes()[:4] not in _MACHO_MAGICS:
        raise ContractError(f"tools.{name}.path must be a native Mach-O executable, not a wrapper")
    if name == "cmake" and path.name != "cmake":
        raise ContractError("tools.cmake.path must name the cmake executable")
    if name == "ninja" and path.name != "ninja":
        raise ContractError("tools.ninja.path must name the ninja executable")
    if name == "python" and path != Path(sys.executable).resolve(strict=True):
        raise ContractError("tools.python.path must equal the interpreter running this command")
    return ToolPin(name=name, path=path, sha256=_sha256(raw["sha256"], f"tools.{name}.sha256"))


def load_build_inputs(path: Path) -> BuildInputs:
    unresolved = path.expanduser().absolute()
    if unresolved.is_symlink() or not unresolved.is_file():
        raise ContractError("build inputs manifest must be a regular non-symlink file")
    path = unresolved.resolve(strict=True)
    raw = _mapping(load_json(path), "MLX native build inputs")
    require_exact_keys(
        raw,
        (
            "schema_version",
            "jobs",
            "macos_deployment_target",
            "macos_sdk_version",
            "metal_toolchain_identifier",
            "developer_dir",
            "user_home",
            "dependencies",
            "tools",
        ),
        where="MLX native build inputs",
    )
    if raw["schema_version"] != BUILD_INPUTS_SCHEMA_VERSION:
        raise ContractError("unsupported MLX native build inputs schema")
    jobs = _int(raw["jobs"], "jobs", minimum=1, maximum=256)
    deployment, components = _numeric_version(
        raw["macos_deployment_target"], "macos_deployment_target"
    )
    if components < (14, 0):
        raise ContractError("MLX requires macOS deployment target 14.0 or newer")
    sdk_version, _sdk_components = _numeric_version(
        raw["macos_sdk_version"], "macos_sdk_version"
    )
    require_jaccl_inventory_sdk_version(sdk_version)
    metal_toolchain_identifier = _string(
        raw["metal_toolchain_identifier"], "metal_toolchain_identifier"
    )
    if re.fullmatch(
        r"com\.apple\.dt\.toolchain\.Metal\.[0-9]+(?:\.[0-9]+)+",
        metal_toolchain_identifier,
    ) is None:
        raise ContractError("metal_toolchain_identifier is malformed")
    developer_dir = _canonical_path(raw["developer_dir"], "developer_dir", kind="directory")
    user_home = _canonical_path(raw["user_home"], "user_home", kind="directory")
    account_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve(strict=True)
    if user_home != account_home:
        raise ContractError(f"user_home {user_home} != current account home {account_home}")

    dependencies_raw = _mapping(raw["dependencies"], "dependencies")
    require_exact_keys(dependencies_raw, DEPENDENCY_NAMES, where="dependencies")
    dependencies = tuple(_validate_dependency_pin(name, dependencies_raw[name]) for name in DEPENDENCY_NAMES)
    if len({pin.path for pin in dependencies}) != len(dependencies):
        raise ContractError("dependency source roots must be distinct")

    tools_raw = _mapping(raw["tools"], "tools")
    require_exact_keys(tools_raw, TOOL_NAMES, where="tools")
    tools = tuple(_validate_tool_pin(name, tools_raw[name]) for name in TOOL_NAMES)
    return BuildInputs(
        path=path,
        sha256=prefixed_sha256(path),
        jobs=jobs,
        macos_deployment_target=deployment,
        macos_sdk_version=sdk_version,
        metal_toolchain_identifier=metal_toolchain_identifier,
        developer_dir=developer_dir,
        user_home=user_home,
        dependencies=dependencies,
        tools=tools,
    )


def _git_environment() -> dict[str, str]:
    return {
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
    }


def _git(source: Path, *arguments: str) -> str:
    return _run_text(
        (str(GIT_PATH), "-C", str(source), *arguments),
        env=_git_environment(),
    )


def verify_pinned_source(source_arg: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    unresolved = source_arg.expanduser().absolute()
    if unresolved.is_symlink():
        raise ContractError("MLX source checkout may not be a symbolic link")
    source = unresolved.resolve(strict=True)
    if source != unresolved:
        raise ContractError(f"MLX source checkout path must be canonical ({source})")
    expected_revision = lock["mlx_reference"]["source_revisions"]["mlx"]
    revision = _git(source, "rev-parse", "HEAD")
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ContractError("MLX source revision is not a full lowercase commit")
    if revision != expected_revision:
        raise ContractError(f"MLX source revision {revision} != locked {expected_revision}")
    top = Path(_git(source, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if top != source:
        raise ContractError(f"MLX source must name the checkout root, found {top}")
    status = _git(
        source,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise ContractError("MLX checkout must be clean, including submodules")
    submodules = _git(source, "submodule", "status", "--recursive")
    if any(line and line[0] != " " for line in submodules.splitlines()):
        raise ContractError("MLX submodules must be initialized at recorded revisions")
    timestamp = _git(source, "show", "-s", "--format=%ct", "HEAD")
    if not timestamp.isdigit():
        raise ContractError("could not determine MLX source commit timestamp")
    version_header = source / "mlx" / "version.h"
    if version_header.is_symlink() or not version_header.is_file():
        raise ContractError("MLX source is missing mlx/version.h")
    version_text = version_header.read_text(encoding="utf-8")
    parts: list[str] = []
    for name in ("MAJOR", "MINOR", "PATCH"):
        match = re.search(rf"^#define MLX_VERSION_{name} ([0-9]+)$", version_text, re.MULTILINE)
        if match is None:
            raise ContractError(f"MLX source version header is missing {name}")
        parts.append(match.group(1))
    version = ".".join(parts)
    expected_version = lock["mlx_reference"]["packages"]["mlx"]
    if version != expected_version:
        raise ContractError(f"MLX source version {version} != locked {expected_version}")
    return {
        "path": str(source),
        "revision": revision,
        "commit_unix_seconds": int(timestamp),
        "version": version,
    }


def resolve_build_paths(source: Path, output_arg: Path, *, must_exist: bool) -> BuildPaths:
    output_unresolved = output_arg.expanduser().absolute()
    if output_unresolved.is_symlink():
        raise ContractError("build output may not be a symbolic link")
    if not output_unresolved.is_relative_to(source):
        raise ContractError("build output must reside inside the pinned MLX checkout")
    if output_unresolved == source:
        raise ContractError("build output may not be the MLX checkout root")
    relative_output = output_unresolved.relative_to(source)
    current = source
    for component in relative_output.parts[:-1]:
        current = current / component
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise ContractError(f"build output parent must be a regular directory: {current}")
    output = output_unresolved.resolve(strict=False)
    if not output.is_relative_to(source):
        raise ContractError("build output escapes the pinned MLX checkout")
    ignored = subprocess.run(
        (
            str(GIT_PATH),
            "-C",
            str(source),
            "check-ignore",
            "-q",
            "--no-index",
            "--",
            str(output),
        ),
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        env=_git_environment(),
    )
    if ignored.returncode != 0:
        detail = ignored.stderr.strip()
        suffix = f": {detail}" if detail else ""
        raise ContractError(f"build output must be covered by MLX .gitignore{suffix}")
    if must_exist:
        if output.is_symlink() or not output.is_dir():
            raise ContractError("build output must be an existing regular directory")
        output = output.resolve(strict=True)
    elif output.exists():
        raise ContractError("build output already exists; builds are no-replace")
    return BuildPaths(
        source=source,
        output=output,
        package_root=source / "python" / "mlx",
        receipt=output / RECEIPT_NAME,
        attestation=output / ATTESTATION_NAME,
        log=output / BUILD_LOG_NAME,
    )


def _verify_dependency_locations(inputs: BuildInputs, source: Path) -> None:
    for pin in inputs.dependencies:
        if pin.path.is_relative_to(source) or source.is_relative_to(pin.path):
            raise ContractError(f"dependency {pin.name} must be outside the MLX checkout")


def ignored_untracked_files(source: Path) -> tuple[Path, ...]:
    completed = subprocess.run(
        (
            str(GIT_PATH),
            "-C",
            str(source),
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "-z",
        ),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip() or "no diagnostic"
        raise ContractError(f"could not inventory ignored MLX files: {detail}")
    paths: list[Path] = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        try:
            relative_text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ContractError("ignored MLX path is not valid UTF-8") from exc
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"git reported unsafe ignored path: {relative_text!r}")
        paths.append((source / relative).absolute())
    return tuple(sorted(paths, key=str))


def require_closed_ignored_files(
    source: Path,
    *,
    build_output: Path | None = None,
    native_paths: Sequence[Path] = (),
) -> None:
    allowed_native = {path.absolute() for path in native_paths}
    unexpected = [
        path
        for path in ignored_untracked_files(source)
        if path not in allowed_native
        and (build_output is None or not path.is_relative_to(build_output))
    ]
    if unexpected:
        raise ContractError(
            "MLX checkout contains ignored files outside the closed build outputs "
            f"(unexpected={[str(path) for path in unexpected]})"
        )


def remove_generated_install_trees(paths: BuildPaths) -> list[str]:
    """Remove only known non-runtime CMake install trees from the source copy.

    MLX stage-0 installs headers and CMake package metadata beside the native
    runtime before setuptools copies that tree into ``python/mlx``. JACCL uses
    ``lib/cmake`` while MLX uses ``share``. These trees are
    legitimate build products but are not loaded benchmark artifacts.  The
    builder proves each exact tree is untracked, ignored, non-symlink, and free
    of special files before removing it; any other ignored residue remains a
    hard failure.
    """
    removed: list[str] = []
    for name in GENERATED_INSTALL_TREES:
        target = paths.package_root / name
        if not target.exists() and not target.is_symlink():
            continue
        if target.is_symlink() or not target.is_dir():
            raise ContractError(f"generated MLX install tree is not a regular directory: {target}")
        relative = target.relative_to(paths.source).as_posix()
        if _git(paths.source, "ls-files", "--", relative):
            raise ContractError(f"refusing to remove tracked MLX install tree: {relative}")
        ignored = subprocess.run(
            (
                str(GIT_PATH),
                "-C",
                str(paths.source),
                "check-ignore",
                "-q",
                "--no-index",
                "--",
                str(target),
            ),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            env=_git_environment(),
        )
        if ignored.returncode != 0:
            raise ContractError(f"generated MLX install tree is not Git-ignored: {relative}")
        for candidate in target.rglob("*"):
            if candidate.is_symlink():
                raise ContractError(f"generated MLX install tree contains symbolic link: {candidate}")
            info = candidate.stat()
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ContractError(f"generated MLX install tree contains special file: {candidate}")
        shutil.rmtree(target)
        if target.exists() or target.is_symlink():
            raise ContractError(f"failed to remove generated MLX install tree: {target}")
        removed.append(relative)
    return removed


def verify_dependency_pins(inputs: BuildInputs) -> list[dict[str, Any]]:
    identities: list[dict[str, Any]] = []
    for pin in inputs.dependencies:
        identity = dependency_tree_identity(pin.path)
        if identity["tree_sha256"] != pin.tree_sha256:
            raise ContractError(
                f"dependency {pin.name} tree digest {identity['tree_sha256']} != pinned {pin.tree_sha256}"
            )
        identities.append({"name": pin.name, **identity})
    return identities


def _tool_environment(inputs: BuildInputs) -> dict[str, str]:
    tool_paths = {pin.name: pin.path for pin in inputs.tools}
    path_dirs: list[str] = []
    for name in ("cmake", "ninja"):
        parent = str(tool_paths[name].parent)
        if parent not in path_dirs:
            path_dirs.append(parent)
    path_dirs.extend(("/usr/bin", "/bin"))
    return {
        "DEVELOPER_DIR": str(inputs.developer_dir),
        "HOME": str(inputs.user_home),
        "LC_ALL": "C",
        "PATH": ":".join(path_dirs),
        "TOOLCHAINS": inputs.metal_toolchain_identifier,
    }


def resolve_bare_build_commands(
    path_value: str,
    expected_commands: Mapping[str, Path],
) -> dict[str, Any]:
    resolved: dict[str, Any] = {}
    for name, expected in expected_commands.items():
        found = shutil.which(name, path=path_value)
        if found is None:
            raise ContractError(f"required bare build command is absent from closed PATH: {name}")
        actual = Path(found).resolve(strict=True)
        expected_resolved = expected.resolve(strict=True)
        if actual != expected_resolved:
            raise ContractError(
                f"bare build command {name} resolves to {actual}, expected {expected_resolved}"
            )
        resolved[name] = {
            "path": str(actual),
            "sha256": prefixed_sha256(actual),
        }
    return resolved


def _python_build_environment_identity(
    python_path: Path,
    environment: Mapping[str, str],
    source: Path,
) -> dict[str, Any]:
    probe = r'''
import hashlib
import importlib.metadata as metadata
import json
import pathlib
import sys
import setuptools.command.bdist_wheel as bdist_wheel_module

def distribution_identity(name):
    distribution = metadata.distribution(name)
    rows = []
    for item in sorted(distribution.files or (), key=str):
        path = pathlib.Path(distribution.locate_file(item))
        if path.is_symlink():
            raise RuntimeError(f"{name} contains a symbolic link: {path}")
        if not path.is_file():
            continue
        data = path.read_bytes()
        rows.append((str(item), len(data), hashlib.sha256(data).hexdigest()))
    encoded = json.dumps(rows, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    return {
        "version": distribution.version,
        "file_count": len(rows),
        "total_bytes": sum(row[1] for row in rows),
        "files_sha256": "sha256:" + hashlib.sha256(
            b"antfly_python_build_distribution/v1\\0" + encoded
        ).hexdigest(),
    }

bdist_wheel_path = pathlib.Path(bdist_wheel_module.__file__).resolve()
bdist_wheel_data = bdist_wheel_path.read_bytes()
print(json.dumps({
    "executable": str(pathlib.Path(sys.executable).resolve()),
    "version": sys.version.split()[0],
    "prefix": str(pathlib.Path(sys.prefix).resolve()),
    "base_prefix": str(pathlib.Path(sys.base_prefix).resolve()),
    "build_frontend": {
        "bdist_wheel_path": str(bdist_wheel_path),
        "bdist_wheel_sha256": "sha256:" + hashlib.sha256(bdist_wheel_data).hexdigest(),
    },
    "packages": {
        "setuptools": distribution_identity("setuptools"),
    },
}, sort_keys=True, separators=(",", ":")))
'''
    probe_environment = {
        **environment,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
    }
    raw = _run_text(
        (str(python_path), "-B", "-c", probe),
        env=probe_environment,
        cwd=source,
    )
    try:
        identity = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError("pinned build Python emitted malformed provenance") from exc
    if not isinstance(identity, Mapping) or set(identity) != {
        "executable", "version", "prefix", "base_prefix", "build_frontend", "packages"
    }:
        raise ContractError("pinned build Python emitted an incomplete provenance identity")
    if Path(_string(identity["executable"], "python.executable")) != python_path:
        raise ContractError("pinned build Python executed a different interpreter")
    if not isinstance(identity["packages"], Mapping) or set(identity["packages"]) != {"setuptools"}:
        raise ContractError("pinned build Python must expose its setuptools identity")
    build_frontend = identity["build_frontend"]
    if not isinstance(build_frontend, Mapping) or set(build_frontend) != {
        "bdist_wheel_path", "bdist_wheel_sha256"
    }:
        raise ContractError("pinned build Python must expose its bdist_wheel module identity")
    return dict(identity)


def inspect_toolchain(
    inputs: BuildInputs,
    lock: Mapping[str, Any],
    source: Path,
) -> dict[str, Any]:
    if platform.system() != lock["mlx_reference"]["required_platform"]:
        raise ContractError("MLX native build requires the locked Darwin platform")
    if platform.machine() != lock["mlx_reference"]["required_machine"]:
        raise ContractError("MLX native build requires the locked arm64 machine")
    actual_python = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual_python != lock["mlx_reference"]["python"]:
        raise ContractError(
            f"MLX native build requires Python {lock['mlx_reference']['python']}, found {actual_python}"
        )
    tool_env = _tool_environment(inputs)
    tool_paths = {pin.name: pin.path for pin in inputs.tools}
    expected_commands = {
        "cmake": tool_paths["cmake"],
        "ninja": tool_paths["ninja"],
        **SYSTEM_BUILD_COMMANDS,
    }
    resolved_commands = resolve_bare_build_commands(tool_env["PATH"], expected_commands)
    tools: dict[str, Any] = {}
    for pin in inputs.tools:
        actual_sha256 = prefixed_sha256(pin.path)
        if actual_sha256 != pin.sha256:
            raise ContractError(f"tool {pin.name} digest {actual_sha256} != pinned {pin.sha256}")
        tools[pin.name] = {
            "path": str(pin.path),
            "sha256": actual_sha256,
            "version_output": _run_text((str(pin.path), *TOOL_VERSION_ARGV[pin.name]), env=tool_env),
        }
    system_tools: dict[str, Any] = {}
    system_tool_paths = {"xcodebuild": Path("/usr/bin/xcodebuild"), **SYSTEM_BUILD_COMMANDS}
    for name, path in system_tool_paths.items():
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"required system tool is not a regular file: {path}")
        system_tools[name] = {"path": str(path), "sha256": prefixed_sha256(path)}
    sdk_version = _run_text(("/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-version"), env=tool_env)
    if sdk_version != inputs.macos_sdk_version:
        raise ContractError(
            f"selected macOS SDK {sdk_version!r} != pinned {inputs.macos_sdk_version!r}"
        )
    require_jaccl_inventory_sdk_version(sdk_version)
    sdk_path = Path(
        _run_text(("/usr/bin/xcrun", "--sdk", "macosx", "--show-sdk-path"), env=tool_env)
    ).resolve(strict=True)
    xcode_version = _run_text(("/usr/bin/xcodebuild", "-version"), env=tool_env)
    selected_xcode_tools: dict[str, Any] = {}
    for name in SELECTED_XCODE_TOOLS:
        selected = Path(
            _run_text(("/usr/bin/xcrun", "--sdk", "macosx", "--find", name), env=tool_env)
        ).resolve(strict=True)
        if selected.is_symlink() or not selected.is_file() or not os.access(selected, os.X_OK):
            raise ContractError(f"xcrun selected an invalid {name} executable: {selected}")
        selected_xcode_tools[name] = {
            "path": str(selected),
            "sha256": prefixed_sha256(selected),
        }
    for pin_name, xcrun_name in (("clang", "clang"), ("clangxx", "clang++")):
        selected = Path(selected_xcode_tools[xcrun_name]["path"])
        if tool_paths[pin_name] != selected:
            raise ContractError(
                f"pinned {pin_name} {tool_paths[pin_name]} != xcrun-selected {selected}"
            )
    python_path = next(pin.path for pin in inputs.tools if pin.name == "python")
    python_identity = _python_build_environment_identity(python_path, tool_env, source)
    if ".".join(python_identity["version"].split(".")[:2]) != lock["mlx_reference"]["python"]:
        raise ContractError("pinned build Python child version differs from the oracle lock")
    return {
        "platform": platform.system(),
        "machine": platform.machine(),
        "os_version": platform.mac_ver()[0],
        "python": {**python_identity, "sha256": prefixed_sha256(python_path)},
        "developer_dir": str(inputs.developer_dir),
        "metal_toolchain_identifier": inputs.metal_toolchain_identifier,
        "xcode_version": xcode_version,
        "sdk_path": str(sdk_path),
        "sdk_version": sdk_version,
        "tools": tools,
        "system_tools": system_tools,
        "resolved_bare_commands": resolved_commands,
        "selected_xcode_tools": selected_xcode_tools,
    }


def _require_no_whitespace(path: Path, where: str) -> None:
    if any(character.isspace() for character in str(path)):
        raise ContractError(f"{where} may not contain whitespace because MLX setup.py splits CMAKE_ARGS")


def build_command(
    paths: BuildPaths,
    source_identity: Mapping[str, Any],
    inputs: BuildInputs,
    toolchain: Mapping[str, Any],
) -> dict[str, Any]:
    dependencies = {pin.name: pin.path for pin in inputs.dependencies}
    tools = {pin.name: pin.path for pin in inputs.tools}
    for name, path in (*dependencies.items(), *tools.items()):
        _require_no_whitespace(path, name)
    sdk_path = Path(toolchain["sdk_path"])
    _require_no_whitespace(sdk_path, "sdk_path")
    cmake_args = (
        "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
        f"-DFETCHCONTENT_SOURCE_DIR_FMT={dependencies['fmt']}",
        f"-DFETCHCONTENT_SOURCE_DIR_JSON={dependencies['json']}",
        f"-DFETCHCONTENT_SOURCE_DIR_METAL_CPP={dependencies['metal_cpp']}",
        f"-DFETCHCONTENT_SOURCE_DIR_NANOBIND={dependencies['nanobind']}",
        "-DMLX_BUILD_METAL=ON",
        "-DMLX_METAL_JIT=OFF",
        "-DMLX_BUILD_PYTHON_STUBS=OFF",
        "-DMLX_BUILD_CPU=ON",
        # The locked benchmark model is Safetensors-only. Disabling GGUF keeps
        # gguf-tools outside the closed native build dependency surface.
        "-DMLX_BUILD_GGUF=OFF",
        "-DMLX_BUILD_SAFETENSORS=ON",
        "-DMLX_USE_CCACHE=OFF",
        f"-DCMAKE_C_COMPILER={tools['clang']}",
        f"-DCMAKE_CXX_COMPILER={tools['clangxx']}",
        # clang and clang++ resolve to the same pinned Mach-O in Xcode. Preserve
        # clang++ link semantics explicitly after canonicalizing the symlink.
        "-DCMAKE_CXX_FLAGS=--driver-mode=g++",
        f"-DCMAKE_MAKE_PROGRAM={tools['ninja']}",
        f"-DCMAKE_OSX_DEPLOYMENT_TARGET={inputs.macos_deployment_target}",
        f"-DCMAKE_OSX_SYSROOT={sdk_path}",
    )
    environment = {
        "ARCHFLAGS": "-arch arm64",
        "CCACHE_DISABLE": "1",
        "CMAKE_ARGS": " ".join(cmake_args),
        "CMAKE_BUILD_PARALLEL_LEVEL": str(inputs.jobs),
        "CMAKE_GENERATOR": "Ninja",
        "DEVELOPER_DIR": str(inputs.developer_dir),
        "DEV_RELEASE": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": str(inputs.user_home),
        "LANG": "C",
        "LC_ALL": "C",
        "MACOSX_DEPLOYMENT_TARGET": inputs.macos_deployment_target,
        "MLX_BUILD_STAGE": "0",
        "PATH": _tool_environment(inputs)["PATH"],
        "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        "PIP_NO_INDEX": "1",
        "PYPI_RELEASE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "SDKROOT": toolchain["sdk_path"],
        "SOURCE_DATE_EPOCH": str(source_identity["commit_unix_seconds"]),
        "TOOLCHAINS": inputs.metal_toolchain_identifier,
        "TMPDIR": str(paths.output / "tmp"),
        "ZERO_AR_DATE": "1",
    }
    argv = (
        str(tools["python"]),
        "setup.py",
        "build_ext",
        "--inplace",
        "--build-temp",
        str(paths.output / "cmake"),
        "--build-lib",
        str(paths.output / "python"),
    )
    return {
        "schema_version": BUILD_COMMAND_SCHEMA_VERSION,
        "argv": list(argv),
        "cwd": str(paths.source),
        "environment": environment,
        "network_policy": {
            "fetchcontent_disconnected": True,
            "network_isolation": "not_enforced",
            "pip_index_disabled": True,
            "top_level_shell": False,
            "upstream_shell_commands": True,
        },
    }


def _relevant_native_paths(package_root: Path) -> set[Path]:
    if package_root.is_symlink() or not package_root.is_dir():
        raise ContractError(f"MLX Python package root is not a regular directory: {package_root}")
    library_root = package_root / "lib"
    paths: set[Path] = set()
    for candidate in package_root.glob("*.so"):
        if candidate.is_symlink():
            raise ContractError(f"MLX native runtime contains symbolic link: {candidate}")
        if candidate.is_file():
            paths.add(candidate.absolute())
    if library_root.exists():
        if library_root.is_symlink() or not library_root.is_dir():
            raise ContractError("MLX package lib path must be a regular directory")
        for candidate in library_root.rglob("*"):
            if candidate.is_symlink():
                raise ContractError(f"MLX native runtime contains symbolic link: {candidate}")
            if candidate.is_file() and candidate.suffix in (".dylib", ".metallib"):
                paths.add(candidate.absolute())
    return paths


def require_no_native_artifacts(package_root: Path) -> None:
    existing = sorted(str(path) for path in _relevant_native_paths(package_root))
    if existing:
        raise ContractError(
            "MLX checkout already contains native build products; use a fresh clean checkout "
            f"(found={existing})"
        )


def discover_native_inventory(
    package_root: Path,
    native_contract: Mapping[str, Any],
) -> dict[str, Any]:
    relevant = _relevant_native_paths(package_root)
    core_candidates = sorted(package_root.glob("core.*.so"))
    if len(core_candidates) != 1:
        raise ContractError(f"expected exactly one built mlx.core extension, found {len(core_candidates)}")
    expected = {
        "jaccl-runtime-dylib": package_root / "lib" / "libjaccl.dylib",
        "metal-library": package_root / "lib" / "mlx.metallib",
        "python-extension": core_candidates[0],
        "runtime-dylib": package_root / "lib" / "libmlx.dylib",
    }
    expected_paths = {path.absolute() for path in expected.values() if path.is_file() and not path.is_symlink()}
    if len(expected_paths) != len(expected) or relevant != expected_paths:
        missing = sorted(str(path) for path in expected.values() if not path.is_file() or path.is_symlink())
        extra = sorted(str(path) for path in relevant - expected_paths)
        raise ContractError(
            "MLX native runtime differs from the closed four-artifact inventory "
            f"(missing={missing}, extra={extra})"
        )
    artifacts: list[dict[str, Any]] = []
    for role in native_contract["artifact_roles"]:
        path = expected[role]
        size = path.stat().st_size
        if size <= 0:
            raise ContractError(f"MLX native artifact is empty: {path}")
        artifacts.append(
            {
                "role": role,
                "relative_path": path.relative_to(package_root).as_posix(),
                "size_bytes": size,
                "sha256": prefixed_sha256(path),
            }
        )
    digest = canonical_mlx_native_artifact_inventory_sha256(artifacts)
    return {
        "schema_version": MLX_NATIVE_ARTIFACT_INVENTORY_SCHEMA_VERSION,
        "sha256": digest,
        "artifacts": artifacts,
    }


def _file_identity(path: Path, *, relative_to: Path | None = None) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ContractError(f"expected regular non-symlink file: {path}")
    relative_path = path.name if relative_to is None else path.relative_to(relative_to).as_posix()
    return {
        "relative_path": relative_path,
        "size_bytes": path.stat().st_size,
        "sha256": prefixed_sha256(path),
    }


def atomic_publish_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish canonical JSON atomically without ever replacing a target."""
    if path.parent.is_symlink() or not path.parent.is_dir():
        raise ContractError("publication parent must be a regular directory")
    data = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    linked = False
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        os.chmod(temporary, 0o444)
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as exc:
            raise ContractError(f"refusing to replace existing evidence: {path}") from exc
        linked = True
        try:
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError as exc:
            try:
                path.unlink()
                linked = False
            except OSError as cleanup_exc:
                raise ContractError(
                    f"evidence directory fsync failed ({exc}) and cleanup failed ({cleanup_exc})"
                ) from cleanup_exc
            raise ContractError(f"evidence directory fsync failed: {exc}") from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if not linked:
        raise ContractError(f"failed to publish evidence: {path}")
    try:
        matches = load_json(path) == payload
    except (ContractError, OSError) as exc:
        matches = False
        readback_error: Exception | None = exc
    else:
        readback_error = None
    if not matches:
        try:
            path.unlink()
            directory_descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError as cleanup_exc:
            raise ContractError(
                f"published JSON failed read-back and cleanup failed: {cleanup_exc}"
            ) from cleanup_exc
        detail = "" if readback_error is None else f": {readback_error}"
        raise ContractError(f"published JSON failed exact read-back verification: {path}{detail}")


def _create_output(paths: BuildPaths) -> None:
    paths.output.parent.mkdir(parents=True, exist_ok=True)
    current = paths.source
    for component in paths.output.parent.relative_to(paths.source).parts:
        current = current / component
        if current.is_symlink() or not current.is_dir():
            raise ContractError(f"build output parent must be a regular directory: {current}")
    try:
        os.mkdir(paths.output, 0o755)
    except FileExistsError as exc:
        raise ContractError("build output already exists; builds are no-replace") from exc
    (paths.output / "tmp").mkdir(mode=0o700)


def _execute_build(command: Mapping[str, Any], log_path: Path) -> tuple[int, int, int]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(log_path, flags, 0o444)
    except FileExistsError as exc:
        raise ContractError(f"refusing to replace existing build log: {log_path}") from exc
    started = time.time_ns()
    try:
        with os.fdopen(descriptor, "wb") as log:
            completed = subprocess.run(
                list(command["argv"]),
                cwd=command["cwd"],
                env=dict(command["environment"]),
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log.flush()
            os.fsync(log.fileno())
    finally:
        finished = time.time_ns()
    return completed.returncode, started, finished


def _receipt(
    *,
    lock_sha256: str,
    source_identity: Mapping[str, Any],
    inputs: BuildInputs,
    dependencies: Sequence[Mapping[str, Any]],
    toolchain: Mapping[str, Any],
    command: Mapping[str, Any],
    paths: BuildPaths,
    inventory: Mapping[str, Any],
    started_unix_ns: int,
    finished_unix_ns: int,
) -> dict[str, Any]:
    return {
        "schema_version": BUILD_RECEIPT_SCHEMA_VERSION,
        "oracle_lock_sha256": lock_sha256,
        "source": {
            "path": source_identity["path"],
            "revision": source_identity["revision"],
            "version": source_identity["version"],
            "clean_before_build": True,
            "clean_after_build": True,
        },
        "inputs": {
            "build_inputs_path": str(inputs.path),
            "build_inputs_sha256": inputs.sha256,
            "builder_path": str(SCRIPT_PATH),
            "builder_sha256": prefixed_sha256(SCRIPT_PATH),
            "dependencies": list(dependencies),
            "toolchain": toolchain,
        },
        "build_command": dict(command),
        "build_result": {
            "exit_code": 0,
            "started_unix_ns": started_unix_ns,
            "finished_unix_ns": finished_unix_ns,
            "log": _file_identity(paths.log, relative_to=paths.output),
        },
        "outputs": {"native_artifact_inventory": dict(inventory)},
    }


def _attestation(
    lock: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    inventory: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    native_contract = lock["mlx_reference"]["native_runtime"]
    return {
        "schema_version": native_contract["build_attestation_schema_version"],
        "source_revision": source_identity["revision"],
        "source_clean": True,
        "native_artifact_inventory_sha256": inventory["sha256"],
        "build_command_sha256": prefixed_sha256(receipt_path),
        "precision_policy_sha256": native_contract["precision_policy_sha256"],
    }


def _validate_receipt_shape(receipt: Mapping[str, Any]) -> None:
    require_exact_keys(
        receipt,
        (
            "schema_version",
            "oracle_lock_sha256",
            "source",
            "inputs",
            "build_command",
            "build_result",
            "outputs",
        ),
        where="MLX native build receipt",
    )
    if receipt["schema_version"] != BUILD_RECEIPT_SCHEMA_VERSION:
        raise ContractError("unsupported MLX native build receipt schema")
    source = _mapping(receipt["source"], "receipt.source")
    require_exact_keys(
        source,
        ("path", "revision", "version", "clean_before_build", "clean_after_build"),
        where="receipt.source",
    )
    inputs = _mapping(receipt["inputs"], "receipt.inputs")
    require_exact_keys(
        inputs,
        (
            "build_inputs_path",
            "build_inputs_sha256",
            "builder_path",
            "builder_sha256",
            "dependencies",
            "toolchain",
        ),
        where="receipt.inputs",
    )
    command = _mapping(receipt["build_command"], "receipt.build_command")
    require_exact_keys(
        command,
        ("schema_version", "argv", "cwd", "environment", "network_policy"),
        where="receipt.build_command",
    )
    if command["schema_version"] != BUILD_COMMAND_SCHEMA_VERSION:
        raise ContractError("unsupported MLX native build command schema")
    result = _mapping(receipt["build_result"], "receipt.build_result")
    require_exact_keys(
        result,
        ("exit_code", "started_unix_ns", "finished_unix_ns", "log"),
        where="receipt.build_result",
    )
    if result["exit_code"] != 0:
        raise ContractError("MLX native build receipt does not record successful completion")
    for field in ("started_unix_ns", "finished_unix_ns"):
        if isinstance(result[field], bool) or not isinstance(result[field], int) or result[field] <= 0:
            raise ContractError(f"receipt.build_result.{field} must be a positive integer")
    if result["finished_unix_ns"] < result["started_unix_ns"]:
        raise ContractError("MLX native build receipt timestamps are reversed")
    outputs = _mapping(receipt["outputs"], "receipt.outputs")
    require_exact_keys(outputs, ("native_artifact_inventory",), where="receipt.outputs")


def _validate_attestation_shape(attestation: Mapping[str, Any]) -> None:
    require_exact_keys(
        attestation,
        (
            "schema_version",
            "source_revision",
            "source_clean",
            "native_artifact_inventory_sha256",
            "build_command_sha256",
            "precision_policy_sha256",
        ),
        where="MLX native build attestation",
    )
    for field in (
        "native_artifact_inventory_sha256",
        "build_command_sha256",
        "precision_policy_sha256",
    ):
        _sha256(attestation[field], f"attestation.{field}")


def verify_published_build(
    paths: BuildPaths,
    lock: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    inputs: BuildInputs,
    dependencies: Sequence[Mapping[str, Any]],
    toolchain: Mapping[str, Any],
) -> dict[str, Any]:
    for path, name in ((paths.receipt, "receipt"), (paths.attestation, "attestation"), (paths.log, "build log")):
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"MLX native build {name} must be a regular non-symlink file")
    receipt = _mapping(load_json(paths.receipt), "MLX native build receipt")
    _validate_receipt_shape(receipt)
    expected_command = build_command(paths, source_identity, inputs, toolchain)
    expected_inventory = discover_native_inventory(paths.package_root, lock["mlx_reference"]["native_runtime"])
    native_paths = tuple(
        paths.package_root / artifact["relative_path"]
        for artifact in expected_inventory["artifacts"]
    )
    require_closed_ignored_files(
        paths.source,
        build_output=paths.output,
        native_paths=native_paths,
    )
    expected_receipt_values = {
        "oracle_lock_sha256": lock_digest(LOCK_PATH),
        "source": {
            "path": source_identity["path"],
            "revision": source_identity["revision"],
            "version": source_identity["version"],
            "clean_before_build": True,
            "clean_after_build": True,
        },
        "inputs": {
            "build_inputs_path": str(inputs.path),
            "build_inputs_sha256": inputs.sha256,
            "builder_path": str(SCRIPT_PATH),
            "builder_sha256": prefixed_sha256(SCRIPT_PATH),
            "dependencies": list(dependencies),
            "toolchain": toolchain,
        },
        "build_command": expected_command,
        "outputs": {"native_artifact_inventory": expected_inventory},
    }
    for field, expected in expected_receipt_values.items():
        if receipt[field] != expected:
            raise ContractError(f"MLX native build receipt {field} differs from current inputs")
    if receipt["build_result"]["log"] != _file_identity(paths.log, relative_to=paths.output):
        raise ContractError("MLX native build log identity differs from receipt")

    attestation = _mapping(load_json(paths.attestation), "MLX native build attestation")
    _validate_attestation_shape(attestation)
    expected_attestation = _attestation(lock, source_identity, expected_inventory, paths.receipt)
    if attestation != expected_attestation:
        raise ContractError("MLX native build attestation differs from receipt/runtime/lock")
    verify_pinned_source(paths.source, lock)
    return {
        "ok": True,
        "receipt": str(paths.receipt),
        "receipt_sha256": prefixed_sha256(paths.receipt),
        "attestation": str(paths.attestation),
        "attestation_sha256": prefixed_sha256(paths.attestation),
        "native_artifact_inventory_sha256": expected_inventory["sha256"],
        "source_revision": source_identity["revision"],
    }


def verify_attestation_bundle(
    attestation_arg: Path,
    source_arg: Path,
    lock: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify the fixed evidence bundle for benchmark-runner admission.

    The six-field attestation is intentionally small, but its
    ``build_command_sha256`` names the exact sibling receipt.  Benchmark
    admission must therefore verify that receipt rather than treating the
    digest as an opaque well-formed string.  This entry point derives the
    pinned input manifest from the receipt, revalidates every source,
    dependency, toolchain, log, lock, and native-runtime identity, and returns
    the regular files the runner should hold stable through the sample.
    """
    attestation_unresolved = attestation_arg.expanduser().absolute()
    if attestation_unresolved.is_symlink() or not attestation_unresolved.is_file():
        raise ContractError("MLX native build attestation must be a regular non-symlink file")
    attestation_path = attestation_unresolved.resolve(strict=True)
    if attestation_path.name != ATTESTATION_NAME:
        raise ContractError(f"MLX native build attestation must be named {ATTESTATION_NAME}")

    source_identity = verify_pinned_source(source_arg, lock)
    source = Path(source_identity["path"])
    paths = resolve_build_paths(source, attestation_path.parent, must_exist=True)
    if paths.attestation != attestation_path:
        raise ContractError("MLX native build attestation is not the fixed file in its build output")
    if paths.receipt.is_symlink() or not paths.receipt.is_file():
        raise ContractError(
            f"MLX native build receipt must be the regular sibling {RECEIPT_NAME}"
        )

    receipt = _mapping(load_json(paths.receipt), "MLX native build receipt")
    _validate_receipt_shape(receipt)
    receipt_inputs = _mapping(receipt["inputs"], "receipt.inputs")
    inputs_path = Path(
        _string(receipt_inputs["build_inputs_path"], "receipt.inputs.build_inputs_path")
    )
    inputs = load_build_inputs(inputs_path)
    _verify_dependency_locations(inputs, source)
    dependencies = verify_dependency_pins(inputs)
    toolchain = inspect_toolchain(inputs, lock, source)
    verified = verify_published_build(
        paths,
        lock,
        source_identity,
        inputs,
        dependencies,
        toolchain,
    )

    bound_paths = {
        paths.receipt,
        paths.attestation,
        paths.log,
        inputs.path,
        SCRIPT_PATH,
        LOCK_PATH,
        *(pin.path for pin in inputs.tools),
        *(
            Path(identity["path"])
            for identity in toolchain["system_tools"].values()
        ),
        *(
            Path(identity["path"])
            for identity in toolchain["selected_xcode_tools"].values()
        ),
        *(
            path
            for pin in inputs.dependencies
            for path in dependency_tree_regular_paths(pin.path)
        ),
    }
    return {
        **verified,
        "bound_paths": [str(path) for path in sorted(bound_paths, key=str)],
    }


def _preflight(args: argparse.Namespace, *, output_must_exist: bool) -> tuple[Any, ...]:
    lock = load_lock(LOCK_PATH)
    source_identity = verify_pinned_source(args.mlx_source, lock)
    source = Path(source_identity["path"])
    paths = resolve_build_paths(source, args.build_output, must_exist=output_must_exist)
    inputs = load_build_inputs(args.inputs)
    _verify_dependency_locations(inputs, source)
    dependencies = verify_dependency_pins(inputs)
    toolchain = inspect_toolchain(inputs, lock, source)
    return lock, source_identity, paths, inputs, dependencies, toolchain


def build(args: argparse.Namespace) -> dict[str, Any]:
    lock, source_identity, paths, inputs, dependencies, toolchain = _preflight(
        args, output_must_exist=False
    )
    lock_sha256_before = lock_digest(LOCK_PATH)
    builder_sha256_before = prefixed_sha256(SCRIPT_PATH)
    require_no_native_artifacts(paths.package_root)
    require_closed_ignored_files(paths.source)
    _create_output(paths)
    command = build_command(paths, source_identity, inputs, toolchain)
    returncode, started, finished = _execute_build(command, paths.log)
    if returncode != 0:
        raise ContractError(
            f"MLX native build failed with exit code {returncode}; retained log: {paths.log}"
        )
    remove_generated_install_trees(paths)
    # Revalidate every mutable input after the compiler has stopped.
    source_identity_after = verify_pinned_source(paths.source, lock)
    if source_identity_after != source_identity:
        raise ContractError("MLX source identity drifted during native build")
    if prefixed_sha256(inputs.path) != inputs.sha256:
        raise ContractError("MLX native build inputs manifest drifted during build")
    if lock_digest(LOCK_PATH) != lock_sha256_before:
        raise ContractError("Gemma4 oracle lock drifted during MLX native build")
    if prefixed_sha256(SCRIPT_PATH) != builder_sha256_before:
        raise ContractError("MLX native build command drifted during execution")
    dependencies_after = verify_dependency_pins(inputs)
    if dependencies_after != dependencies:
        raise ContractError("MLX native dependency tree drifted during build")
    toolchain_after = inspect_toolchain(inputs, lock, paths.source)
    if toolchain_after != toolchain:
        raise ContractError("MLX native toolchain drifted during build")
    inventory = discover_native_inventory(paths.package_root, lock["mlx_reference"]["native_runtime"])
    native_paths = tuple(
        paths.package_root / artifact["relative_path"]
        for artifact in inventory["artifacts"]
    )
    require_closed_ignored_files(
        paths.source,
        build_output=paths.output,
        native_paths=native_paths,
    )
    receipt = _receipt(
        lock_sha256=lock_sha256_before,
        source_identity=source_identity,
        inputs=inputs,
        dependencies=dependencies,
        toolchain=toolchain,
        command=command,
        paths=paths,
        inventory=inventory,
        started_unix_ns=started,
        finished_unix_ns=finished,
    )
    atomic_publish_json(paths.receipt, receipt)
    atomic_publish_json(paths.attestation, _attestation(lock, source_identity, inventory, paths.receipt))
    return verify_published_build(
        paths,
        lock,
        source_identity,
        inputs,
        dependencies_after,
        toolchain_after,
    )


def verify(args: argparse.Namespace) -> dict[str, Any]:
    lock, source_identity, paths, inputs, dependencies, toolchain = _preflight(
        args, output_must_exist=True
    )
    return verify_published_build(
        paths,
        lock,
        source_identity,
        inputs,
        dependencies,
        toolchain,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and attest the pinned MLX 0.31.2 native runtime with "
            "FetchContent disconnected."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        (
            "build",
            "perform one no-replace FetchContent-disconnected build and publish its attestation",
        ),
        ("verify", "rehash and verify an existing build receipt and attestation"),
    ):
        command = subparsers.add_parser(name, help=help_text)
        command.add_argument("--mlx-source", type=Path, required=True)
        command.add_argument(
            "--build-output",
            type=Path,
            required=True,
            help="New (build) or existing (verify) Git-ignored directory inside the MLX checkout.",
        )
        command.add_argument("--inputs", type=Path, required=True, help="Strict local dependency/tool pin manifest.")
    file_digest = subparsers.add_parser(
        "file-digest",
        help="compute the SHA-256 used to pin one build executable",
    )
    file_digest.add_argument("path", type=Path)
    tree = subparsers.add_parser(
        "tree-digest",
        help="compute the canonical digest used to pin one local dependency tree",
    )
    tree.add_argument("path", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "tree-digest":
            payload = dependency_tree_identity(args.path)
        elif args.command == "file-digest":
            path = _canonical_path(args.path, "file", kind="file")
            payload = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": prefixed_sha256(path),
            }
        elif args.command == "build":
            payload = build(args)
        else:
            payload = verify(args)
    except ContractError as exc:
        parser.error(str(exc))
    print(json.dumps(payload, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
