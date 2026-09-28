"""Small standard-library-only file primitives shared by Gemma tooling."""

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 hex digest, preserving filesystem errors."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
