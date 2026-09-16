"""Attest the pinned MLX-LM source archive without relying on a parent Git checkout."""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


# Official codeload archive for the same revision pinned by gemma4_oracle.lock.json.
MLX_LM_ARCHIVE_SHA256 = {
    "ed1fca4cef15a824c5f1702c80f70b4cffc8e4dd": "67e1a52f9b86551a24eab1aa2681c26a391819925bb10d14792b96a88303ebc7",
}


class SourceArchiveError(ValueError):
    pass


def attest_mlx_lm_archive(root: Path, archive: Path, revision: str) -> dict[str, Any]:
    expected_digest = MLX_LM_ARCHIVE_SHA256.get(revision)
    if expected_digest is None:
        raise SourceArchiveError("no pinned MLX-LM archive digest for this revision")
    if root.is_symlink() or not root.is_dir():
        raise SourceArchiveError("MLX-LM archive source root must be a real directory")
    root = root.resolve()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != expected_digest:
        raise SourceArchiveError(
            "MLX-LM archive SHA-256 differs from the pinned revision"
        )
    prefix = f"mlx-lm-{revision}"
    inventory: dict[str, str] = {}
    directories: set[str] = set()
    seen: set[str] = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source.getmembers():
            path = PurePosixPath(member.name)
            if (
                path.is_absolute()
                or not path.parts
                or path.parts[0] != prefix
                or ".." in path.parts
            ):
                raise SourceArchiveError(
                    "MLX-LM archive member escapes its pinned root"
                )
            relative = PurePosixPath(*path.parts[1:]).as_posix()
            if relative in seen:
                raise SourceArchiveError("MLX-LM archive repeats a path")
            seen.add(relative)
            local = root.joinpath(*path.parts[1:])
            if local.is_symlink() or not local.resolve().is_relative_to(root):
                raise SourceArchiveError(
                    "MLX-LM source contains a symlink or escaped path"
                )
            if member.isdir():
                if not local.is_dir():
                    raise SourceArchiveError(
                        f"MLX-LM source directory missing: {relative}"
                    )
                if relative != ".":
                    directories.add(relative)
            elif member.isfile():
                payload = source.extractfile(member)
                if payload is None or not local.is_file():
                    raise SourceArchiveError(f"MLX-LM source file missing: {relative}")
                expected = hashlib.sha256(payload.read()).hexdigest()
                if hashlib.sha256(local.read_bytes()).hexdigest() != expected:
                    raise SourceArchiveError(f"MLX-LM source file changed: {relative}")
                inventory[relative] = expected
            else:
                raise SourceArchiveError(
                    "MLX-LM archive contains an unsupported member"
                )
    actual_files: set[str] = set()
    actual_directories: set[str] = set()
    for local in root.rglob("*"):
        if local.is_symlink():
            raise SourceArchiveError("MLX-LM source contains a symlink")
        relative = local.relative_to(root).as_posix()
        if local.is_file():
            actual_files.add(relative)
        elif local.is_dir():
            actual_directories.add(relative)
        else:
            raise SourceArchiveError("MLX-LM source contains an unsupported file")
    if actual_files != set(inventory) or actual_directories != directories:
        raise SourceArchiveError(
            "MLX-LM source inventory differs from the archive; use a clean extraction without bytecode caches"
        )
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return {
        "mode": "pinned-upstream-archive",
        "revision": revision,
        "archive_path": str(archive.resolve()),
        "archive_sha256": digest,
        "source_root": str(root),
        "source_files": len(inventory),
        "source_inventory_sha256": hashlib.sha256(encoded).hexdigest(),
    }
