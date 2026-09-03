"""Fail-closed contracts shared by the Gemma4 LoRA reference tools.

The correctness oracle deliberately consumes Antfly's already-tokenized
``input_ids`` and ``labels``.  Tokenizer/chat-template conformance is a
separate exact comparison; it cannot be hidden inside a numerical tolerance.

This module is standard-library-only.  Torch, PEFT, Safetensors, and MLX are
imported lazily by the commands that need them, after their installed versions
and local artifact identities have been verified against the checked-in lock.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LOCK_SCHEMA_VERSION = "antfly_gemma4_oracle_lock/v3"
TRACE_SCHEMA_VERSION = "antfly_gemma4_lora_trace/v1"
PUBLICATION_SCHEMA_VERSION = "antfly_gemma4_oracle_publication/v1"
ZIG_ORACLE_PACKAGING_PACKAGES = ("numpy", "safetensors")
BENCHMARK_SAMPLE_SCHEMA_VERSION = "antfly_gemma4_lora_benchmark_sample/v5"
BENCHMARK_PRECISION_POLICY_DOMAIN = b"antfly_gemma4_precision_policy/v1\0"
MLX_NATIVE_ARTIFACT_INVENTORY_SCHEMA_VERSION = "antfly_mlx_native_artifact_inventory/v2"
MLX_NATIVE_ARTIFACT_INVENTORY_DOMAIN = b"antfly_mlx_native_artifact_inventory/v2\0"
MLX_NATIVE_ARTIFACT_ROLES = (
    "jaccl-runtime-dylib",
    "metal-library",
    "python-extension",
    "runtime-dylib",
)
BENCHMARK_PRODUCER_SOURCE_SCHEMA_VERSION = "antfly_gemma4_benchmark_producer_source/v1"
BENCHMARK_PRODUCER_SOURCE_DOMAIN = b"antfly_gemma4_benchmark_producer_source/v1\0"
BENCHMARK_PRODUCER_RELATIVE_PATHS = (
    "zig/pkg/inference/scripts/gemma4/bench_gemma4_lora_mlx_zig.py",
    "zig/pkg/inference/scripts/gemma4/build_and_attest_gemma4_mlx.py",
    "zig/pkg/inference/scripts/gemma4/gemma4_oracle.lock.json",
    "zig/pkg/inference/scripts/gemma4/gemma4_oracle_contract.py",
    "zig/pkg/inference/scripts/gemma4/requirements-gemma4-mlx-reference.txt",
    "zig/pkg/inference/scripts/gemma4/run_antfly_gemma4_lora_benchmark.py",
    "zig/pkg/inference/scripts/gemma4/run_gemma4_lora_benchmark_campaign.py",
    "zig/pkg/inference/scripts/gemma4/run_gemma4_lora_mlx_benchmark.py",
)
PREPARED_SCHEMA_VERSION = "gemma4_prepared/v6"
ANTFLY_ADAPTER_MANIFEST_SCHEMA_V2 = "antfly_gemma4_finetune/v2"
ANTFLY_ADAPTER_MANIFEST_SCHEMA_V3 = "antfly_gemma4_finetune/v3"
ANTFLY_ADAPTER_MANIFEST_SCHEMA_VERSIONS = (
    ANTFLY_ADAPTER_MANIFEST_SCHEMA_V2,
    ANTFLY_ADAPTER_MANIFEST_SCHEMA_V3,
)
ANTFLY_ADAPTER_KEY_FORMAT = "antfly_gemma4_adapter_keys/v1"
STOCK_PEFT_KEY_FORMAT = "stock-peft/v1"
ANTFLY_PEFT_EXPORT_MANIFEST_SCHEMA_VERSION = "antfly_gemma4_peft_export/v1"
IGNORE_LABEL = -100
LOCK_PATH = Path(__file__).with_name("gemma4_oracle.lock.json")

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)+(?:[-+][A-Za-z0-9.]+)?$")


class ContractError(ValueError):
    """A release-relevant input did not satisfy its frozen contract."""


def benchmark_precision_policy_sha256(precision: Mapping[str, Any]) -> str:
    """Bind the closed precision policy used by native build attestations."""
    encoded = json.dumps(
        precision,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(BENCHMARK_PRECISION_POLICY_DOMAIN + encoded).hexdigest()


def canonical_mlx_native_artifact_inventory_sha256(
    artifacts: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the complete sorted MLX native runtime surface."""
    if not isinstance(artifacts, Sequence) or isinstance(artifacts, (str, bytes)):
        raise ContractError("MLX native artifact inventory must be an array")
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(artifacts):
        artifact = _require_mapping(raw, f"MLX native artifact inventory[{index}]")
        require_exact_keys(
            artifact,
            ("role", "relative_path", "size_bytes", "sha256"),
            where=f"MLX native artifact inventory[{index}]",
        )
        role = _require_string(artifact["role"], f"MLX native artifact inventory[{index}].role")
        relative_text = _require_string(
            artifact["relative_path"], f"MLX native artifact inventory[{index}].relative_path"
        )
        relative = Path(relative_text)
        if relative.is_absolute() or relative_text != relative.as_posix() or ".." in relative.parts:
            raise ContractError("MLX native artifact inventory paths must be normalized relative paths")
        size = _require_int(artifact["size_bytes"], f"MLX native artifact inventory[{index}].size_bytes", minimum=1)
        digest = _require_string(artifact["sha256"], f"MLX native artifact inventory[{index}].sha256")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ContractError("MLX native artifact inventory SHA-256 is malformed")
        normalized.append(
            {"role": role, "relative_path": relative_text, "size_bytes": size, "sha256": digest}
        )
    roles = tuple(item["role"] for item in normalized)
    if roles != MLX_NATIVE_ARTIFACT_ROLES:
        raise ContractError(
            "MLX native artifact inventory must contain the exact sorted Metal runtime roles"
        )
    names = {item["role"]: Path(item["relative_path"]).name for item in normalized}
    if (
        names["jaccl-runtime-dylib"] != "libjaccl.dylib"
        or names["metal-library"] != "mlx.metallib"
        or names["runtime-dylib"] != "libmlx.dylib"
    ):
        raise ContractError("MLX native artifact inventory has an unexpected runtime library name")
    extension_name = names["python-extension"]
    if not extension_name.startswith("core.") or not extension_name.endswith(".so"):
        raise ContractError("MLX native artifact inventory does not bind the loaded mlx.core extension")
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(MLX_NATIVE_ARTIFACT_INVENTORY_DOMAIN + encoded).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> Any:
    """Load JSON while rejecting duplicate keys and non-finite numbers."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContractError(f"could not read {path}: {exc}") from exc

    def reject_constant(value: str) -> None:
        raise ContractError(f"non-finite JSON number is forbidden: {value}")

    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ContractError(f"invalid JSON in {path}: {exc}") from exc


def require_exact_keys(value: Mapping[str, Any], required: Iterable[str], *, where: str) -> None:
    expected = set(required)
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        pieces = []
        if missing:
            pieces.append("missing=" + ",".join(missing))
        if unknown:
            pieces.append("unknown=" + ",".join(unknown))
        raise ContractError(f"{where}: field contract mismatch ({'; '.join(pieces)})")


def _require_mapping(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{where}: expected object")
    return value


def _require_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise ContractError(f"{where}: expected array")
    return value


def _require_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{where}: expected non-empty string")
    return value


def _require_int(value: Any, where: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{where}: expected integer")
    if minimum is not None and value < minimum:
        raise ContractError(f"{where}: expected value >= {minimum}")
    return value


def _finite_float(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{where}: expected number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{where}: expected finite number")
    return result


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                hasher.update(chunk)
    except OSError as exc:
        raise ContractError(f"could not hash {path}: {exc}") from exc
    return hasher.hexdigest()


def git_blob_sha1(path: Path, size: int | None = None) -> str:
    file_size = path.stat().st_size if size is None else size
    hasher = hashlib.sha1(usedforsecurity=False)
    hasher.update(f"blob {file_size}\0".encode("ascii"))
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def prefixed_sha256(path: Path) -> str:
    return f"sha256:{sha256_file(path)}"


def canonical_benchmark_producer_source_sha256(
    *,
    relative_path: str,
    source_revision: str,
    source_tree: str,
    files: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the closed source manifest for one benchmark producer entrypoint."""
    payload = {
        "relative_path": relative_path,
        "source_revision": source_revision,
        "source_tree": source_tree,
        "files": [dict(item) for item in files],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(BENCHMARK_PRODUCER_SOURCE_DOMAIN + encoded).hexdigest()


def validate_benchmark_producer_source(
    raw: Any,
    *,
    expected_entrypoint: str | None = None,
    where: str = "benchmark producer source",
) -> dict[str, Any]:
    """Validate a content-closed producer source attestation.

    The producing process establishes the Git facts while the checkout is live.
    This validator then makes the complete file inventory and its commit/tree
    binding immutable inside the sample, without depending on absolute paths.
    """
    source = dict(_require_mapping(raw, where))
    require_exact_keys(
        source,
        (
            "schema_version", "relative_path", "source_revision", "source_tree",
            "source_clean", "source_sha256", "files", "manifest_sha256",
        ),
        where=where,
    )
    if source["schema_version"] != BENCHMARK_PRODUCER_SOURCE_SCHEMA_VERSION:
        raise ContractError(f"{where}: unsupported schema version")
    relative_path = _require_string(source["relative_path"], f"{where}.relative_path")
    if expected_entrypoint is not None and relative_path != expected_entrypoint:
        raise ContractError(f"{where}: entrypoint differs from the admitted benchmark runner")
    if relative_path not in BENCHMARK_PRODUCER_RELATIVE_PATHS:
        raise ContractError(f"{where}: unknown producer entrypoint")
    for field in ("source_revision", "source_tree"):
        value = _require_string(source[field], f"{where}.{field}")
        if _HEX40.fullmatch(value) is None:
            raise ContractError(f"{where}.{field}: expected a full Git object id")
    if source["source_clean"] is not True:
        raise ContractError(f"{where}: producer checkout must be clean")
    entrypoint_sha256 = _require_string(source["source_sha256"], f"{where}.source_sha256")
    manifest_sha256 = _require_string(source["manifest_sha256"], f"{where}.manifest_sha256")
    for field, value in (("source_sha256", entrypoint_sha256), ("manifest_sha256", manifest_sha256)):
        if re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None:
            raise ContractError(f"{where}.{field}: malformed SHA-256")

    raw_files = _require_list(source["files"], f"{where}.files")
    files: list[dict[str, str]] = []
    for index, raw_file in enumerate(raw_files):
        item = _require_mapping(raw_file, f"{where}.files[{index}]")
        require_exact_keys(item, ("relative_path", "source_sha256"), where=f"{where}.files[{index}]")
        path = _require_string(item["relative_path"], f"{where}.files[{index}].relative_path")
        digest = _require_string(item["source_sha256"], f"{where}.files[{index}].source_sha256")
        relative = Path(path)
        if relative.is_absolute() or path != relative.as_posix() or ".." in relative.parts:
            raise ContractError(f"{where}: source paths must be normalized relative paths")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ContractError(f"{where}: malformed source file SHA-256")
        files.append({"relative_path": path, "source_sha256": digest})
    expected_paths = list(BENCHMARK_PRODUCER_RELATIVE_PATHS)
    if [item["relative_path"] for item in files] != expected_paths:
        raise ContractError(f"{where}: producer source inventory is incomplete, duplicated, or unsorted")
    entrypoints = [item for item in files if item["relative_path"] == relative_path]
    if len(entrypoints) != 1 or entrypoints[0]["source_sha256"] != entrypoint_sha256:
        raise ContractError(f"{where}: entrypoint digest differs from the closed source inventory")
    expected_manifest = canonical_benchmark_producer_source_sha256(
        relative_path=relative_path,
        source_revision=source["source_revision"],
        source_tree=source["source_tree"],
        files=files,
    )
    if manifest_sha256 != expected_manifest:
        raise ContractError(f"{where}: producer source manifest digest mismatch")
    source["files"] = files
    return source


def attest_benchmark_producer_source(
    entrypoint: Path,
    *,
    expected_entrypoint: str,
) -> dict[str, Any]:
    """Attest a complete clean checkout and exact committed producer closure."""
    script = entrypoint.expanduser().resolve(strict=True)

    def git(root: Path, *arguments: str, text: bool = True) -> subprocess.CompletedProcess[Any]:
        try:
            return subprocess.run(
                ("git", "-C", str(root), *arguments),
                check=True,
                capture_output=True,
                text=text,
            )
        except (OSError, subprocess.CalledProcessError) as exc:
            detail = getattr(exc, "stderr", b"" if not text else "") or str(exc)
            if isinstance(detail, bytes):
                detail = detail.decode("utf-8", errors="replace")
            raise ContractError(f"could not attest benchmark producer source: {str(detail).strip()}") from exc

    probe = git(script.parent, "rev-parse", "--show-toplevel")
    root = Path(probe.stdout.strip()).resolve(strict=True)
    try:
        relative_entrypoint = script.relative_to(root).as_posix()
    except ValueError as exc:
        raise ContractError("benchmark producer entrypoint is outside its Git checkout") from exc
    if relative_entrypoint != expected_entrypoint:
        raise ContractError("benchmark producer entrypoint differs from its admitted repository path")
    revision = git(root, "rev-parse", "HEAD").stdout.strip()
    source_tree = git(root, "rev-parse", "HEAD^{tree}").stdout.strip()
    if _HEX40.fullmatch(revision) is None or _HEX40.fullmatch(source_tree) is None:
        raise ContractError("benchmark producer revision/tree is not a full Git object id")
    status = git(root, "status", "--porcelain=v1", "--untracked-files=all").stdout
    if status:
        raise ContractError("benchmark producer checkout must be completely clean, including untracked files")

    files: list[dict[str, str]] = []
    for relative_path in BENCHMARK_PRODUCER_RELATIVE_PATHS:
        path = root / relative_path
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"benchmark producer source must be a regular tracked file: {relative_path}")
        git(root, "ls-files", "--error-unmatch", "--", relative_path)
        committed = git(root, "show", f"{revision}:{relative_path}", text=False).stdout
        actual = path.read_bytes()
        if committed != actual:
            raise ContractError(f"benchmark producer source differs from commit {revision}: {relative_path}")
        files.append(
            {
                "relative_path": relative_path,
                "source_sha256": "sha256:" + hashlib.sha256(actual).hexdigest(),
            }
        )
    entrypoint_sha256 = next(
        item["source_sha256"] for item in files if item["relative_path"] == relative_entrypoint
    )
    result = {
        "schema_version": BENCHMARK_PRODUCER_SOURCE_SCHEMA_VERSION,
        "relative_path": relative_entrypoint,
        "source_revision": revision,
        "source_tree": source_tree,
        "source_clean": True,
        "source_sha256": entrypoint_sha256,
        "files": files,
        "manifest_sha256": canonical_benchmark_producer_source_sha256(
            relative_path=relative_entrypoint,
            source_revision=revision,
            source_tree=source_tree,
            files=files,
        ),
    }
    return validate_benchmark_producer_source(result, expected_entrypoint=expected_entrypoint)


def build_evidence_ledger(root: Path) -> dict[str, Any]:
    """Build a closed, deterministic inventory for an oracle publication."""
    publication_candidate = root.expanduser()
    if publication_candidate.is_symlink():
        raise ContractError(f"oracle publication root cannot be a symlink: {publication_candidate}")
    publication_root = publication_candidate.resolve()
    if not publication_root.is_dir():
        raise ContractError(f"oracle publication root is not a directory: {publication_root}")
    complete = publication_root / "COMPLETE.json"
    if complete.exists():
        raise ContractError("COMPLETE.json must be written only after the evidence ledger is built")
    files: list[dict[str, Any]] = []
    for path in sorted(publication_root.rglob("*"), key=lambda item: item.relative_to(publication_root).as_posix()):
        if path.is_symlink():
            raise ContractError(f"oracle publication cannot contain a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ContractError(f"oracle publication contains a non-regular file: {path}")
        relative = path.relative_to(publication_root).as_posix()
        files.append({
            "path": relative,
            "sha256": prefixed_sha256(path),
            "size_bytes": path.stat().st_size,
        })
    if not files:
        raise ContractError("oracle publication cannot be empty")
    return {
        "schema_version": PUBLICATION_SCHEMA_VERSION,
        "status": "complete",
        "files": files,
    }


def validate_evidence_ledger(trace_path: Path) -> tuple[str, dict[str, dict[str, Any]]]:
    """Validate COMPLETE.json as a closed inventory and return its digest/files."""
    trace_candidate = trace_path.expanduser()
    if trace_candidate.is_symlink():
        raise ContractError(f"oracle trace cannot be a symlink: {trace_candidate}")
    resolved_trace = trace_candidate.resolve()
    root = resolved_trace.parent
    complete = root / "COMPLETE.json"
    if not complete.is_file() or complete.is_symlink():
        raise ContractError(f"complete oracle evidence requires {complete}")
    marker = _require_mapping(load_json(complete), "COMPLETE.json")
    require_exact_keys(marker, ("schema_version", "status", "files"), where="COMPLETE.json")
    if marker["schema_version"] != PUBLICATION_SCHEMA_VERSION or marker["status"] != "complete":
        raise ContractError("COMPLETE.json has an unsupported publication contract")

    declared: dict[str, dict[str, Any]] = {}
    declared_order: list[str] = []
    for index, raw_entry in enumerate(_require_list(marker["files"], "COMPLETE.json.files")):
        entry = _require_mapping(raw_entry, f"COMPLETE.json.files[{index}]")
        require_exact_keys(entry, ("path", "sha256", "size_bytes"), where=f"COMPLETE.json.files[{index}]")
        relative_text = _require_string(entry["path"], f"COMPLETE.json.files[{index}].path")
        relative = Path(relative_text)
        if relative_text != relative.as_posix() or relative.is_absolute() or ".." in relative.parts:
            raise ContractError(f"COMPLETE.json.files[{index}].path must be a normalized safe relative path")
        if relative_text == "COMPLETE.json" or relative_text in declared:
            raise ContractError(f"COMPLETE.json contains a duplicate or self-referential path: {relative_text}")
        digest = _require_string(entry["sha256"], f"COMPLETE.json.files[{index}].sha256")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
            raise ContractError(f"COMPLETE.json.files[{index}].sha256 is malformed")
        size_bytes = _require_int(entry["size_bytes"], f"COMPLETE.json.files[{index}].size_bytes", minimum=0)
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ContractError(f"COMPLETE.json evidence file is missing or not regular: {relative_text}")
        if path.stat().st_size != size_bytes or prefixed_sha256(path) != digest:
            raise ContractError(f"COMPLETE.json evidence digest/size mismatch: {relative_text}")
        declared[relative_text] = dict(entry)
        declared_order.append(relative_text)
    if declared_order != sorted(declared_order):
        raise ContractError("COMPLETE.json files must be sorted by normalized path")

    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ContractError(f"oracle publication cannot contain a symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ContractError(f"oracle publication contains a non-regular file: {path}")
        relative = path.relative_to(root).as_posix()
        if relative != "COMPLETE.json":
            actual.add(relative)
    if actual != set(declared):
        missing = sorted(set(declared) - actual)
        uncommitted = sorted(actual - set(declared))
        raise ContractError(
            f"COMPLETE.json is not a closed evidence inventory "
            f"(missing={missing}, uncommitted={uncommitted})"
        )
    trace_relative = resolved_trace.relative_to(root).as_posix()
    if trace_relative not in declared:
        raise ContractError("COMPLETE.json does not commit the requested trace")
    return prefixed_sha256(complete), declared


def lock_digest(lock_path: Path = LOCK_PATH) -> str:
    return prefixed_sha256(lock_path.resolve())


def validate_lock(lock: Any) -> dict[str, Any]:
    lock = dict(_require_mapping(lock, "oracle lock"))
    require_exact_keys(
        lock,
        (
            "schema_version",
            "lock_policy",
            "models",
            "python_oracle",
            "mlx_reference",
            "target_presets",
            "target_inventory",
            "training_contract",
            "benchmark_contract",
            "tolerance_profiles",
            "performance_gate",
            "references",
        ),
        where="oracle lock",
    )
    if lock["schema_version"] != LOCK_SCHEMA_VERSION:
        raise ContractError(f"unsupported oracle lock schema: {lock['schema_version']!r}")

    policy = _require_mapping(lock["lock_policy"], "lock_policy")
    require_exact_keys(
        policy,
        (
            "network_access",
            "model_loading",
            "unknown_fields",
            "comparison_input",
            "prepared_schema",
            "label_ignore_index",
            "causal_label_shift",
            "loss_reduction",
        ),
        where="lock_policy",
    )
    expected_policy = {
        "network_access": "forbidden-during-run",
        "model_loading": "local-files-only",
        "unknown_fields": "reject",
        "comparison_input": "exact-antfly-input-ids-and-labels",
        "prepared_schema": PREPARED_SCHEMA_VERSION,
        "label_ignore_index": IGNORE_LABEL,
        "causal_label_shift": 1,
    }
    for key, expected in expected_policy.items():
        if policy[key] != expected:
            raise ContractError(f"lock_policy.{key}: {policy[key]!r} != {expected!r}")
    _require_string(policy["loss_reduction"], "lock_policy.loss_reduction")

    models = _require_mapping(lock["models"], "models")
    if set(models) != {"gemma-4-E2B-it", "gemma-4-E4B-it"}:
        raise ContractError("models must contain exactly Gemma4 E2B-it and E4B-it")
    expected_repositories = {
        "gemma-4-E2B-it": "google/gemma-4-E2B-it",
        "gemma-4-E4B-it": "google/gemma-4-E4B-it",
    }
    for model_key, raw_model in models.items():
        model = _require_mapping(raw_model, f"models.{model_key}")
        require_exact_keys(model, ("repo_id", "revision", "architecture", "files"), where=f"models.{model_key}")
        if model["repo_id"] != expected_repositories[model_key]:
            raise ContractError(f"models.{model_key}.repo_id does not match the model key")
        if not _HEX40.fullmatch(_require_string(model["revision"], f"models.{model_key}.revision")):
            raise ContractError(f"models.{model_key}.revision must be a full commit")
        if model["architecture"] != "Gemma4ForConditionalGeneration":
            raise ContractError(f"models.{model_key}.architecture is not Gemma4")
        files = _require_mapping(model["files"], f"models.{model_key}.files")
        required_files = {"config.json", "chat_template.jinja", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
        if set(files) != required_files:
            raise ContractError(f"models.{model_key}.files must pin exactly {sorted(required_files)}")
        for name, raw_spec in files.items():
            spec = _require_mapping(raw_spec, f"models.{model_key}.files.{name}")
            if set(spec) not in ({"size", "sha256"}, {"size", "git_blob_sha1"}):
                raise ContractError(f"models.{model_key}.files.{name}: require one exact digest")
            _require_int(spec["size"], f"models.{model_key}.files.{name}.size", minimum=1)
            digest_key = "sha256" if "sha256" in spec else "git_blob_sha1"
            matcher = _HEX64 if digest_key == "sha256" else _HEX40
            if not matcher.fullmatch(_require_string(spec[digest_key], f"models.{model_key}.files.{name}.{digest_key}")):
                raise ContractError(f"models.{model_key}.files.{name}.{digest_key}: malformed digest")

    python_oracle = _require_mapping(lock["python_oracle"], "python_oracle")
    require_exact_keys(
        python_oracle,
        ("python", "packages", "source_revisions", "execution"),
        where="python_oracle",
    )
    if re.fullmatch(r"[0-9]+\.[0-9]+", _require_string(python_oracle["python"], "python_oracle.python")) is None:
        raise ContractError("python_oracle.python must pin a major.minor runtime")
    execution = _require_mapping(python_oracle["execution"], "python_oracle.execution")
    require_exact_keys(
        execution,
        (
            "attention_implementation", "device", "dropout", "dtype", "gradient_dtype",
            "optimizer_state_dtype", "use_cache", "deterministic_algorithms",
        ),
        where="python_oracle.execution",
    )
    expected_execution = {
        "attention_implementation": "eager",
        "device": "cuda",
        "dropout": 0.0,
        "dtype": "bfloat16",
        "gradient_dtype": "float32",
        "optimizer_state_dtype": "float32",
        "use_cache": False,
        "deterministic_algorithms": True,
    }
    if execution != expected_execution:
        raise ContractError("python_oracle.execution differs from the deterministic BF16 lane")

    mlx_reference = _require_mapping(lock["mlx_reference"], "mlx_reference")
    require_exact_keys(
        mlx_reference,
        (
            "python", "packages", "source_revisions", "native_runtime", "role", "minimum_fresh_processes",
            "warmup_steps", "measured_steps", "required_platform", "required_machine",
        ),
        where="mlx_reference",
    )
    if mlx_reference["role"] != "same-mac-performance-reference-only":
        raise ContractError("MLX-LM must remain performance-reference-only")
    if re.fullmatch(r"[0-9]+\.[0-9]+", _require_string(mlx_reference["python"], "mlx_reference.python")) is None:
        raise ContractError("mlx_reference.python must pin a major.minor runtime")
    if mlx_reference["required_platform"] != "Darwin" or mlx_reference["required_machine"] != "arm64":
        raise ContractError("MLX-LM reference hardware must be Darwin arm64")
    _require_int(mlx_reference["minimum_fresh_processes"], "mlx_reference.minimum_fresh_processes", minimum=5)
    _require_int(mlx_reference["warmup_steps"], "mlx_reference.warmup_steps", minimum=1)
    _require_int(mlx_reference["measured_steps"], "mlx_reference.measured_steps", minimum=5)
    mlx_native_runtime = _require_mapping(mlx_reference["native_runtime"], "mlx_reference.native_runtime")
    require_exact_keys(
        mlx_native_runtime,
        (
            "extension_module",
            "artifact_roles",
            "minimum_macos_sdk_version",
            "build_attestation_schema_version",
            "precision_policy_sha256",
        ),
        where="mlx_reference.native_runtime",
    )
    if mlx_native_runtime["extension_module"] != "mlx.core":
        raise ContractError("mlx_reference native extension module is unsupported")
    if mlx_native_runtime["build_attestation_schema_version"] != "antfly_mlx_native_build_attestation/v1":
        raise ContractError("mlx_reference native build attestation schema is unsupported")
    if tuple(_require_list(mlx_native_runtime["artifact_roles"], "mlx_reference.native_runtime.artifact_roles")) != MLX_NATIVE_ARTIFACT_ROLES:
        raise ContractError("mlx_reference native artifact roles are not the exact sorted runtime surface")
    if mlx_native_runtime["minimum_macos_sdk_version"] != "26.2":
        raise ContractError("mlx_reference native runtime must require the JACCL-capable macOS SDK")
    if re.fullmatch(
        r"sha256:[0-9a-f]{64}",
        _require_string(mlx_native_runtime["precision_policy_sha256"], "mlx_reference.native_runtime.precision_policy_sha256"),
    ) is None:
        raise ContractError("mlx_reference precision policy digest is malformed")

    for env_name in ("python_oracle", "mlx_reference"):
        env = _require_mapping(lock[env_name], env_name)
        packages = _require_mapping(env.get("packages"), f"{env_name}.packages")
        expected_packages = {
            "python_oracle": {"accelerate", "huggingface-hub", "numpy", "peft", "safetensors", "torch", "transformers"},
            "mlx_reference": {"mlx", "mlx-lm", "numpy"},
        }[env_name]
        if set(packages) != expected_packages:
            raise ContractError(f"{env_name}.packages must be exactly {sorted(expected_packages)}")
        for package, version in packages.items():
            if not _VERSION.fullmatch(_require_string(version, f"{env_name}.packages.{package}")):
                raise ContractError(f"{env_name}.packages.{package}: version is not exact")
        revisions = _require_mapping(env.get("source_revisions"), f"{env_name}.source_revisions")
        expected_sources = {
            "python_oracle": {"peft", "transformers"},
            "mlx_reference": {"mlx", "mlx-lm"},
        }[env_name]
        if set(revisions) != expected_sources:
            raise ContractError(f"{env_name}.source_revisions must be exactly {sorted(expected_sources)}")
        for source, revision in revisions.items():
            if not _HEX40.fullmatch(_require_string(revision, f"{env_name}.source_revisions.{source}")):
                raise ContractError(f"{env_name}.source_revisions.{source}: expected full commit")

    presets = _require_mapping(lock["target_presets"], "target_presets")
    if set(presets) != {"peft-qv", "text-all-linear"}:
        raise ContractError("target_presets: unsupported preset set")
    for name, raw_targets in presets.items():
        targets = [_require_string(item, f"target_presets.{name}[]") for item in _require_list(raw_targets, f"target_presets.{name}")]
        if not targets or len(set(targets)) != len(targets):
            raise ContractError(f"target_presets.{name}: targets must be non-empty and unique")

    target_inventory = _require_mapping(lock["target_inventory"], "target_inventory")
    if set(target_inventory) != set(models):
        raise ContractError("target_inventory must cover exactly the locked models")
    for model_key, raw_model_inventory in target_inventory.items():
        model_inventory = _require_mapping(raw_model_inventory, f"target_inventory.{model_key}")
        if set(model_inventory) != set(presets):
            raise ContractError(f"target_inventory.{model_key} must cover exactly the target presets")
        for preset, raw_counts in model_inventory.items():
            counts = _require_mapping(raw_counts, f"target_inventory.{model_key}.{preset}")
            if set(counts) != set(presets[preset]):
                raise ContractError(
                    f"target_inventory.{model_key}.{preset} suffixes differ from target_presets"
                )
            for suffix, count in counts.items():
                _require_int(count, f"target_inventory.{model_key}.{preset}.{suffix}", minimum=1)

    training_contract = _require_mapping(lock["training_contract"], "training_contract")
    require_exact_keys(
        training_contract,
        (
            "optimizer", "seed", "steps", "learning_rate", "betas", "eps",
            "weight_decay", "max_grad_norm", "grad_accum_steps",
            "supervised_token_normalization",
        ),
        where="training_contract",
    )
    if training_contract["optimizer"] != "adamw":
        raise ContractError("training_contract.optimizer must be adamw")
    _require_int(training_contract["seed"], "training_contract.seed", minimum=0)
    steps = [_require_int(item, "training_contract.steps[]", minimum=1) for item in _require_list(training_contract["steps"], "training_contract.steps")]
    if steps != [1, 2, 8]:
        raise ContractError("training_contract.steps must be exactly [1,2,8]")
    if _finite_float(training_contract["learning_rate"], "training_contract.learning_rate") <= 0:
        raise ContractError("training_contract.learning_rate must be positive")
    betas = [_finite_float(item, "training_contract.betas[]") for item in _require_list(training_contract["betas"], "training_contract.betas")]
    if len(betas) != 2 or any(beta < 0 or beta >= 1 for beta in betas):
        raise ContractError("training_contract.betas must contain two values in [0,1)")
    if _finite_float(training_contract["eps"], "training_contract.eps") <= 0:
        raise ContractError("training_contract.eps must be positive")
    for field in ("weight_decay", "max_grad_norm"):
        if _finite_float(training_contract[field], f"training_contract.{field}") < 0:
            raise ContractError(f"training_contract.{field} must be non-negative")
    if training_contract["grad_accum_steps"] != 1:
        raise ContractError("training_contract.grad_accum_steps must be one")
    if training_contract["supervised_token_normalization"] != "mean":
        raise ContractError("training_contract normalization must be mean")

    benchmark_contract = _require_mapping(lock["benchmark_contract"], "benchmark_contract")
    require_exact_keys(
        benchmark_contract,
        ("schema_version", "precision", "optimizer", "determinism", "runtime", "memory"),
        where="benchmark_contract",
    )
    if benchmark_contract["schema_version"] != "antfly_gemma4_lora_benchmark_semantics/v3":
        raise ContractError("benchmark_contract.schema_version is unsupported")
    expected_benchmark_contract = {
        "precision": {
            "verified": {
                "base_model_storage_dtype": "bfloat16",
                "lora_parameter_storage_dtype": "float32",
                "gradient_storage_dtype": "float32",
                "optimizer_moment_storage_dtype": "float32",
                "loss_tensor_dtype": "float32",
                "loss_reduction_input_dtype": "float32",
            },
            "not_asserted": [
                "activation_dtype",
                "matmul_accumulator_dtype",
            ],
            "comparison_policy": "diagnostic-only-until-all-comparison-critical-dtypes-are-runtime-proven",
        },
        "optimizer": {
            "name": "adamw",
            "learning_rate": 0.001,
            "schedule": "constant-no-warmup",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "bias_correction": True,
            "weight_decay": 0.01,
            "weight_decay_scope": "all-and-only-lora-a-and-lora-b-parameters",
            "gradient_clip_kind": "global-l2-norm",
            "gradient_clip_max_norm": 1.0,
            "gradient_clip_epsilon": 1e-6,
            "gradient_clip_order": "after-accumulation-before-adamw-update",
            "gradient_accumulation_reduction": "arithmetic-mean-of-per-microbatch-mean-supervised-token-gradients",
            "partial_accumulation_windows": "forbidden",
        },
        "determinism": {
            "seed": 42,
            "sample_order": "fixed-prepared-workload-order-without-shuffle",
            "dropout": 0.0,
            "lora_a_initializer": "kaiming-uniform-a-sqrt5-fan-in",
            "lora_b_initializer": "zeros",
            "initial_adapter_policy": "load-byte-equivalent-canonical-f32-semantics",
        },
        "runtime": {
            "attention_kv_cache": False,
            "activation_checkpointing": False,
            "training_checkpoint_io": "disabled",
            "compiled_graph_cache": "enabled-process-local-shape-keyed",
            "compile_policy": "lazy-first-full-optimizer-window-included-in-cold-metric",
            "filesystem_cache_policy": "not-flushed-between-alternating-fresh-processes",
            "per_step_device_sync": "after-optimizer-update-before-timer-stop-every-window",
        },
        "memory": {
            "process_peak_definition": "maximum-darwin-proc-pid-rusage-v4-ri-phys-footprint-sample",
            "sampler_interval_ms": 10,
            "framework_allocator_peak": "separate-diagnostic-not-cross-framework-gate",
            "system_delta_source": "darwin-vm-stat-pages-and-memory-pressure-before-after",
            "system_delta_window": "measured-optimizer-steps-only",
            "maximum_swapins_bytes": 0,
            "maximum_swapouts_bytes": 0,
            "maximum_pageins_bytes": 0,
            "maximum_pageouts_bytes": 0,
            "minimum_pressure_available_percent_delta": -5.0,
        },
    }
    for section, expected in expected_benchmark_contract.items():
        actual = _require_mapping(benchmark_contract[section], f"benchmark_contract.{section}")
        if actual != expected:
            raise ContractError(f"benchmark_contract.{section} differs from the frozen same-Mac contract")
    if mlx_native_runtime["precision_policy_sha256"] != benchmark_precision_policy_sha256(
        _require_mapping(benchmark_contract["precision"], "benchmark_contract.precision")
    ):
        raise ContractError("mlx_reference native runtime precision policy digest differs from benchmark_contract.precision")

    tolerance_profiles = _require_mapping(lock["tolerance_profiles"], "tolerance_profiles")
    tolerance_keys = {
        "loss_abs", "loss_rel", "grad_norm_abs", "grad_norm_rel",
        "logits_max_abs", "logits_rel_l2", "logits_cosine_min",
        "gradient_max_abs", "gradient_rel_l2", "gradient_cosine_min",
        "state_max_abs", "state_rel_l2", "state_cosine_min",
    }
    if set(tolerance_profiles) != {"tiny-f32", "native-metal-bf16", "hf-zig-bf16", "resume"}:
        raise ContractError("tolerance_profiles must contain exactly the four release lanes")
    for name, raw_profile in tolerance_profiles.items():
        profile_value = _require_mapping(raw_profile, f"tolerance_profiles.{name}")
        require_exact_keys(profile_value, tolerance_keys, where=f"tolerance_profiles.{name}")
        for key, value in profile_value.items():
            number = _finite_float(value, f"tolerance_profiles.{name}.{key}")
            if key.endswith("cosine_min"):
                if not -1 <= number <= 1:
                    raise ContractError(f"tolerance_profiles.{name}.{key}: expected [-1,1]")
            elif number < 0:
                raise ContractError(f"tolerance_profiles.{name}.{key}: expected non-negative")

    performance = _require_mapping(lock["performance_gate"], "performance_gate")
    require_exact_keys(
        performance,
        (
            "minimum_geomean_throughput_ratio", "minimum_cell_throughput_ratio",
            "maximum_peak_memory_ratio", "maximum_regression_fraction",
            "primary_sequence_lengths", "gradient_accumulation", "microbatch", "rank", "alpha",
        ),
        where="performance_gate",
    )
    for field in (
        "minimum_geomean_throughput_ratio", "minimum_cell_throughput_ratio",
        "maximum_peak_memory_ratio", "maximum_regression_fraction", "alpha",
    ):
        value = _finite_float(performance[field], f"performance_gate.{field}")
        if value <= 0:
            raise ContractError(f"performance_gate.{field} must be positive")
    if performance["maximum_regression_fraction"] > 1:
        raise ContractError("performance_gate.maximum_regression_fraction must be <= 1")
    for field in ("microbatch", "rank"):
        _require_int(performance[field], f"performance_gate.{field}", minimum=1)
    for field in ("primary_sequence_lengths", "gradient_accumulation"):
        values = [_require_int(item, f"performance_gate.{field}[]", minimum=1) for item in _require_list(performance[field], f"performance_gate.{field}")]
        if not values or len(values) != len(set(values)):
            raise ContractError(f"performance_gate.{field} must be non-empty and unique")
    references = _require_mapping(lock["references"], "references")
    if not references or any(not isinstance(url, str) or not url.startswith("https://") for url in references.values()):
        raise ContractError("references must be non-empty HTTPS URLs")
    return lock


def load_lock(path: Path = LOCK_PATH) -> dict[str, Any]:
    return validate_lock(load_json(path.resolve()))


def parse_requirements(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ContractError(f"{path}:{line_number}: dependency must use exactly ==")
        package, version = line.split("==", 1)
        if not package or not _VERSION.fullmatch(version):
            raise ContractError(f"{path}:{line_number}: malformed exact dependency pin")
        if package in result:
            raise ContractError(f"{path}:{line_number}: duplicate dependency {package}")
        result[package] = version
    if not result:
        raise ContractError(f"{path}: no dependencies")
    return result


def verify_requirements_match_lock(lock: Mapping[str, Any], environment: str, path: Path) -> dict[str, str]:
    expected = dict(_require_mapping(lock[environment], environment)["packages"])
    actual = parse_requirements(path)
    if actual != expected:
        raise ContractError(f"{path.name} does not exactly match {environment}.packages")
    return actual


def verify_requirements_match_lock_subset(
    lock: Mapping[str, Any],
    environment: str,
    packages: Sequence[str],
    path: Path,
) -> dict[str, str]:
    locked = _require_mapping(
        _require_mapping(lock[environment], environment)["packages"],
        f"{environment}.packages",
    )
    if not packages or len(set(packages)) != len(packages):
        raise ContractError("requirements subset must be non-empty and unique")
    missing = sorted(set(packages) - set(locked))
    if missing:
        raise ContractError(f"requirements subset is absent from {environment}.packages: {missing}")
    expected = {
        name: _require_string(locked[name], f"{environment}.packages.{name}")
        for name in packages
    }
    actual = parse_requirements(path)
    if actual != expected:
        raise ContractError(
            f"{path.name} does not exactly match the requested "
            f"{environment}.packages subset"
        )
    return actual


def verify_python_runtime(lock: Mapping[str, Any]) -> str:
    expected = _require_string(_require_mapping(lock["python_oracle"], "python_oracle")["python"], "python_oracle.python")
    actual = f"{sys.version_info.major}.{sys.version_info.minor}"
    if actual != expected:
        raise ContractError(f"Gemma4 oracle requires Python {expected}, found {actual}")
    return actual


def verify_packages(
    lock: Mapping[str, Any],
    environment: str,
    *,
    actual_versions: Mapping[str, str] | None = None,
) -> dict[str, str]:
    expected = dict(_require_mapping(lock[environment], environment)["packages"])
    actual: dict[str, str] = {}
    for package in expected:
        if actual_versions is not None:
            actual[package] = actual_versions.get(package, "<missing>")
            continue
        try:
            actual[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual[package] = "<missing>"
    mismatches = [
        f"{name}={actual[name]} (expected {version})"
        for name, version in expected.items()
        if actual[name] != version
    ]
    if mismatches:
        raise ContractError(f"{environment} dependency mismatch: " + ", ".join(mismatches))
    return actual


def verify_source_checkout(path: Path, expected_revision: str, *, source_name: str) -> dict[str, str]:
    source = path.expanduser().resolve()
    if not source.is_dir():
        raise ContractError(f"{source_name} checkout is not a directory: {source}")
    head = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    revision = head.stdout.strip() if head.returncode == 0 else ""
    if revision != expected_revision:
        raise ContractError(f"{source_name} revision {revision or '<unknown>'} != pinned {expected_revision}")
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain=v1", "--untracked-files=all"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if dirty.returncode != 0 or dirty.stdout.strip():
        raise ContractError(f"{source_name} checkout must be clean")
    return {"path": str(source), "revision": revision}


def verify_import_source(imported: Any, checkout: Path, *, source_name: str) -> str:
    imported_from = Path(inspect.getfile(imported)).resolve()
    checkout = checkout.expanduser().resolve()
    if not imported_from.is_relative_to(checkout):
        raise ContractError(f"{source_name} imported from unpinned source: {imported_from}")
    return str(imported_from)


def verify_model_directory(lock: Mapping[str, Any], model_key: str, model_dir: Path) -> dict[str, Any]:
    models = _require_mapping(lock["models"], "models")
    if model_key not in models:
        raise ContractError(f"unknown locked model: {model_key}")
    model = _require_mapping(models[model_key], f"models.{model_key}")
    root = model_dir.expanduser().resolve()
    if not root.is_dir():
        raise ContractError(f"model directory does not exist: {root}")
    verified: dict[str, dict[str, Any]] = {}
    aggregate = hashlib.sha256()
    aggregate.update(b"antfly_gemma4_locked_model/v1\0")
    aggregate.update(model_key.encode("utf-8") + b"\0")
    aggregate.update(model["revision"].encode("ascii") + b"\0")
    for relative_path, raw_spec in sorted(_require_mapping(model["files"], f"models.{model_key}.files").items()):
        if Path(relative_path).is_absolute() or ".." in Path(relative_path).parts:
            raise ContractError(f"unsafe locked model path: {relative_path}")
        spec = _require_mapping(raw_spec, f"models.{model_key}.files.{relative_path}")
        path = root / relative_path
        if not path.is_file():
            raise ContractError(f"required locked model file is missing: {path}")
        size = path.stat().st_size
        if size != spec["size"]:
            raise ContractError(f"{path}: size {size} != pinned {spec['size']}")
        if "sha256" in spec:
            digest_kind = "sha256"
            digest = sha256_file(path)
        else:
            digest_kind = "git_blob_sha1"
            digest = git_blob_sha1(path, size)
        if digest != spec[digest_kind]:
            raise ContractError(f"{path}: {digest_kind} {digest} != pinned {spec[digest_kind]}")
        verified[relative_path] = {"size": size, digest_kind: digest}
        aggregate.update(relative_path.encode("utf-8") + b"\0")
        aggregate.update(str(size).encode("ascii") + b"\0")
        aggregate.update(digest_kind.encode("ascii") + b":" + digest.encode("ascii") + b"\0")
    return {
        "model_key": model_key,
        "repo_id": model["repo_id"],
        "revision": model["revision"],
        "directory": str(root),
        "files": verified,
        "local_artifact_sha256": f"sha256:{aggregate.hexdigest()}",
    }


def _validate_int_array(value: Any, where: str, *, labels: bool = False) -> list[int]:
    result: list[int] = []
    for index, item in enumerate(_require_list(value, where)):
        number = _require_int(item, f"{where}[{index}]")
        if labels:
            if number < 0 and number != IGNORE_LABEL:
                raise ContractError(f"{where}[{index}]: only {IGNORE_LABEL} may be negative")
        elif number < 0:
            raise ContractError(f"{where}[{index}]: token ids must be non-negative")
        result.append(number)
    if not result:
        raise ContractError(f"{where}: cannot be empty")
    return result


def _zig_hash_length(hasher: Any, value: int) -> None:
    hasher.update(struct.pack("<Q", value))


def _zig_hash_bytes(hasher: Any, value: str | bytes) -> None:
    encoded = value.encode("utf-8") if isinstance(value, str) else value
    _zig_hash_length(hasher, len(encoded))
    hasher.update(encoded)


def _zig_hash_ints(hasher: Any, values: Sequence[int]) -> None:
    _zig_hash_length(hasher, len(values))
    for value in values:
        hasher.update(struct.pack("<i", int(value)))


def _zig_hash_usizes(hasher: Any, values: Sequence[int]) -> None:
    _zig_hash_length(hasher, len(values))
    for value in values:
        _zig_hash_length(hasher, int(value))


def _zig_hash_f32s(hasher: Any, values: Sequence[float]) -> None:
    _zig_hash_length(hasher, len(values))
    for value in values:
        hasher.update(struct.pack("<f", float(value)))


def _zig_hash_strings(hasher: Any, values: Sequence[str]) -> None:
    _zig_hash_length(hasher, len(values))
    for value in values:
        _zig_hash_bytes(hasher, value)


def _optional_string(value: Any, where: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ContractError(f"{where}: expected string or null")
    return value


def _fingerprint_prepared_examples(examples: Sequence[Mapping[str, Any]], domain: str) -> str:
    """Mirror Zig's prepared-example fingerprint for one explicit domain."""
    hasher = hashlib.sha256()
    _zig_hash_bytes(hasher, domain)
    _zig_hash_length(hasher, len(examples))
    modes = {"instruction": 0, "completion": 1}
    for index, example in enumerate(examples):
        mode = example.get("mode")
        if mode not in modes:
            raise ContractError(f"examples[{index}].mode: unsupported value {mode!r}")
        _zig_hash_length(hasher, modes[mode])
        for field in ("prompt_input_ids", "response_input_ids", "input_ids"):
            values = example.get(field, [])
            if not isinstance(values, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in values):
                raise ContractError(f"examples[{index}].{field}: expected integer array")
            _zig_hash_ints(hasher, values)
        for field in ("image_paths", "audio_paths"):
            values = example.get(field, [])
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ContractError(f"examples[{index}].{field}: expected string array")
            _zig_hash_strings(hasher, values)
        for field in (
            "source_id", "source_group_id", "source_name", "source_record_sha256", "rendered_chat_sha256",
        ):
            _zig_hash_bytes(hasher, _optional_string(example.get(field), f"examples[{index}].{field}"))
        media_hashes = example.get("media_content_sha256", [])
        if not isinstance(media_hashes, list) or any(not isinstance(value, str) for value in media_hashes):
            raise ContractError(f"examples[{index}].media_content_sha256: expected string array")
        _zig_hash_strings(hasher, media_hashes)
        for field in ("image_token_counts", "audio_token_counts"):
            values = example.get(field, [])
            if not isinstance(values, list) or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
                raise ContractError(f"examples[{index}].{field}: expected non-negative integer array")
            _zig_hash_usizes(hasher, values)
        labels = example.get("labels", [])
        if not isinstance(labels, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in labels):
            raise ContractError(f"examples[{index}].labels: expected integer array")
        _zig_hash_ints(hasher, labels)
        for field in (
            "num_prompt_tokens", "num_response_tokens", "num_input_tokens", "num_supervised_tokens", "turn_count",
        ):
            _zig_hash_length(hasher, _require_int(example.get(field, 0), f"examples[{index}].{field}", minimum=0))
        for field in ("has_tool_calls", "has_tool_messages"):
            value = example.get(field, False)
            if not isinstance(value, bool):
                raise ContractError(f"examples[{index}].{field}: expected boolean")
            _zig_hash_length(hasher, int(value))
        teacher_ids = example.get("teacher_top_k_token_ids", [])
        if not isinstance(teacher_ids, list) or any(isinstance(value, bool) or not isinstance(value, int) for value in teacher_ids):
            raise ContractError(f"examples[{index}].teacher_top_k_token_ids: expected integer array")
        _zig_hash_ints(hasher, teacher_ids)
        teacher_probs = example.get("teacher_top_k_probs", [])
        if not isinstance(teacher_probs, list):
            raise ContractError(f"examples[{index}].teacher_top_k_probs: expected number array")
        _zig_hash_f32s(hasher, [_finite_float(value, f"examples[{index}].teacher_top_k_probs") for value in teacher_probs])
        _zig_hash_length(hasher, _require_int(example.get("teacher_top_k", 0), f"examples[{index}].teacher_top_k", minimum=0))
        _zig_hash_f32s(hasher, [_finite_float(example.get("teacher_temperature", 1.0), f"examples[{index}].teacher_temperature")])
        was_truncated = example.get("was_truncated", False)
        if not isinstance(was_truncated, bool):
            raise ContractError(f"examples[{index}].was_truncated: expected boolean")
        _zig_hash_length(hasher, int(was_truncated))
        _zig_hash_length(hasher, _require_int(example.get("turns_dropped_from_left", 0), f"examples[{index}].turns_dropped_from_left", minimum=0))
        _zig_hash_bytes(hasher, _optional_string(example.get("policy_version"), f"examples[{index}].policy_version"))
    return hasher.hexdigest()


def fingerprint_prepared_examples_v2(examples: Sequence[Mapping[str, Any]]) -> str:
    """Mirror Zig's ``gemma4_prepared_examples/v2`` fingerprint exactly."""
    return _fingerprint_prepared_examples(examples, "gemma4_prepared_examples/v2")


def fingerprint_prepared_examples_v3(examples: Sequence[Mapping[str, Any]]) -> str:
    """Mirror Zig's ``gemma4_prepared_examples/v3`` fingerprint exactly."""
    return _fingerprint_prepared_examples(examples, "gemma4_prepared_examples/v3")


def resolve_dataset_jsonl_files(path: Path, split: str) -> list[Path]:
    source = path.expanduser().resolve()
    if source.is_file() and source.name != "manifest.json":
        return [source]
    manifest_path = source if source.is_file() else source / "manifest.json"
    if manifest_path.is_file():
        manifest = _require_mapping(load_json(manifest_path), "dataset manifest")
        files = _require_list(manifest.get("files"), "dataset manifest.files")
        result = []
        for index, raw_entry in enumerate(files):
            entry = _require_mapping(raw_entry, f"dataset manifest.files[{index}]")
            if entry.get("kind") != "jsonl" or entry.get("split") != split:
                continue
            relative = Path(_require_string(entry.get("relative_path"), f"dataset manifest.files[{index}].relative_path"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ContractError("dataset manifest contains an unsafe relative path")
            result.append((manifest_path.parent / relative).resolve())
        result.sort(key=str)
    elif source.is_dir():
        result = sorted(
            (item.resolve() for item in source.iterdir() if item.is_file() and item.suffix == ".jsonl" and item.name.startswith(split + "-")),
            key=str,
        )
    else:
        raise ContractError(f"source dataset path does not exist: {source}")
    if not result or any(not item.is_file() for item in result):
        raise ContractError(f"source dataset has no complete JSONL files for split {split!r}")
    return result


def fingerprint_dataset_source(path: Path, split: str) -> str:
    files = resolve_dataset_jsonl_files(path, split)
    hasher = hashlib.sha256()
    _zig_hash_bytes(hasher, "gemma4_dataset_source/v1")
    _zig_hash_bytes(hasher, split)
    _zig_hash_length(hasher, len(files))
    for source in files:
        _zig_hash_bytes(hasher, "jsonl")
        _zig_hash_bytes(hasher, source.name)
        _zig_hash_bytes(hasher, source.read_bytes())
    return hasher.hexdigest()


def verify_prepared_source_dataset(summary: Mapping[str, Any], path: Path | None = None) -> dict[str, str]:
    split = _require_string(summary.get("source_split"), "prepared.summary.source_split")
    source_path = path if path is not None else Path(_require_string(summary.get("source_dataset_path"), "prepared.summary.source_dataset_path"))
    actual = fingerprint_dataset_source(source_path, split)
    expected = _require_string(summary.get("source_dataset_sha256"), "prepared.summary.source_dataset_sha256")
    if actual != expected:
        raise ContractError(f"prepared source dataset SHA-256 {actual} != recorded {expected}")
    return {"path": str(source_path.expanduser().resolve()), "split": split, "sha256": actual}


def load_prepared_example(path: Path, example_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _require_mapping(load_json(path.resolve()), "prepared artifact")
    require_exact_keys(payload, ("summary",), where="prepared artifact")
    summary = dict(_require_mapping(payload["summary"], "prepared.summary"))
    if summary.get("schema_version") != PREPARED_SCHEMA_VERSION:
        raise ContractError(f"prepared.summary.schema_version must be {PREPARED_SCHEMA_VERSION}")
    if summary.get("artifact_family_version") != "gemma4_lora/v1alpha1":
        raise ContractError("prepared artifact family is not Gemma4 LoRA")
    examples = _require_list(summary.get("examples"), "prepared.summary.examples")
    if not 0 <= example_index < len(examples):
        raise ContractError(f"prepared example index {example_index} is out of range")
    if _require_int(summary.get("examples_seen"), "prepared.summary.examples_seen", minimum=1) != len(examples):
        raise ContractError("prepared examples_seen does not match examples length")
    if summary.get("examples_with_images", 0) != 0 or summary.get("examples_with_audio", 0) != 0:
        raise ContractError("Gemma4 correctness oracle admits text-only prepared artifacts")
    fingerprint = summary.get("prepared_examples_sha256")
    if not isinstance(fingerprint, str) or not _HEX64.fullmatch(fingerprint):
        raise ContractError("prepared_examples_sha256 is missing or malformed")

    for field in ("base_model_sha256", "tokenizer_sha256", "chat_template_sha256", "source_dataset_sha256"):
        value = summary.get(field)
        if not isinstance(value, str) or not _HEX64.fullmatch(value):
            raise ContractError(f"prepared.summary.{field} is missing or malformed")
    _require_string(summary.get("source_dataset_path"), "prepared.summary.source_dataset_path")
    source_split = _require_string(summary.get("source_split"), "prepared.summary.source_split")
    source_revision = _require_string(summary.get("source_revision"), "prepared.summary.source_revision")

    max_input = 0
    max_supervised = 0
    normalized: list[dict[str, Any]] = []
    source_records: set[str] = set()
    for index, raw_example in enumerate(examples):
        example = dict(_require_mapping(raw_example, f"prepared.summary.examples[{index}]"))
        input_ids = _validate_int_array(example.get("input_ids"), f"examples[{index}].input_ids")
        labels = _validate_int_array(example.get("labels"), f"examples[{index}].labels", labels=True)
        if len(input_ids) != len(labels):
            raise ContractError(f"examples[{index}]: input_ids and labels lengths differ")
        if _require_int(example.get("num_input_tokens"), f"examples[{index}].num_input_tokens") != len(input_ids):
            raise ContractError(f"examples[{index}]: num_input_tokens is inconsistent")
        supervised = sum(label != IGNORE_LABEL for label in labels)
        if supervised == 0 or not any(labels[position] != IGNORE_LABEL for position in range(1, len(labels))):
            raise ContractError(f"examples[{index}]: no causal supervised tokens")
        if _require_int(example.get("num_supervised_tokens"), f"examples[{index}].num_supervised_tokens") != supervised:
            raise ContractError(f"examples[{index}]: num_supervised_tokens is inconsistent")
        for field in ("image_paths", "audio_paths", "image_token_counts", "audio_token_counts"):
            if example.get(field, []) != []:
                raise ContractError(f"examples[{index}].{field}: multimodal data is not admitted")
        if example.get("media_content_sha256", []) != []:
            raise ContractError(f"examples[{index}].media_content_sha256: text-only examples require an empty array")
        source_id = _require_string(example.get("source_id"), f"examples[{index}].source_id")
        source_group_id = _require_string(example.get("source_group_id"), f"examples[{index}].source_group_id")
        source_name = _require_string(example.get("source_name"), f"examples[{index}].source_name")
        record_digest = _require_string(example.get("source_record_sha256"), f"examples[{index}].source_record_sha256")
        rendered_digest = _require_string(example.get("rendered_chat_sha256"), f"examples[{index}].rendered_chat_sha256")
        if not _HEX64.fullmatch(record_digest) or not _HEX64.fullmatch(rendered_digest):
            raise ContractError(f"examples[{index}]: source/rendered digest is malformed")
        if record_digest in source_records:
            raise ContractError(f"examples[{index}]: duplicate source record digest")
        source_records.add(record_digest)
        max_input = max(max_input, len(input_ids))
        max_supervised = max(max_supervised, supervised)
        normalized.append({
            "input_ids": input_ids,
            "labels": labels,
            "source_id": source_id,
            "source_group_id": source_group_id,
            "source_name": source_name,
            "source_record_sha256": record_digest,
            "rendered_chat_sha256": rendered_digest,
        })
    max_seq_len = _require_int(summary.get("max_seq_len"), "prepared.summary.max_seq_len", minimum=1)
    if max_input > max_seq_len:
        raise ContractError("prepared example exceeds max_seq_len")
    if summary.get("max_input_tokens") != max_input or summary.get("max_supervised_tokens") != max_supervised:
        raise ContractError("prepared aggregate token counters are inconsistent")
    actual_examples_fingerprint = fingerprint_prepared_examples_v3(
        [dict(_require_mapping(example, "prepared example")) for example in examples]
    )
    if actual_examples_fingerprint != fingerprint:
        raise ContractError("prepared_examples_sha256 does not match the v6 example payload")
    selected = normalized[example_index]
    return summary, {
        "schema_version": PREPARED_SCHEMA_VERSION,
        "artifact_sha256": prefixed_sha256(path.resolve()),
        "example_index": example_index,
        "input_ids": selected["input_ids"],
        "labels": selected["labels"],
        "source_dataset_sha256": summary["source_dataset_sha256"],
        "source_split": source_split,
        "source_revision": source_revision,
        "source_id": selected["source_id"],
        "source_group_id": selected["source_group_id"],
        "source_name": selected["source_name"],
        "source_record_sha256": selected["source_record_sha256"],
        "rendered_chat_sha256": selected["rendered_chat_sha256"],
    }


_ADAPTER_NAME = re.compile(
    r"^(?P<module>.+?)(?:\.weight)?\.lora_(?P<role>[AB])(?:\.[^.]+)?\.weight$"
)
_PREFIXES = (
    "base_model.model.model.language_model.",
    "base_model.model.language_model.",
    "base_model.model.",
    "model.language_model.",
    "language_model.",
)
_PLE_ALIASES = (
    ("per_layer_model_projection", "per_layer_input.per_layer_model_proj"),
    ("per_layer_input_gate", "per_layer_input.inp_gate"),
    ("per_layer_projection", "per_layer_input.proj"),
)


def canonicalize_module_name(name: str) -> str:
    result = _require_string(name, "module name")
    changed = True
    while changed:
        changed = False
        for prefix in _PREFIXES:
            if result.startswith(prefix):
                result = result[len(prefix):]
                changed = True
                break
    if result.startswith(("layers.", "per_layer_input.")):
        result = "model." + result
    if result.endswith(".weight"):
        result = result[: -len(".weight")]
    for legacy, canonical in _PLE_ALIASES:
        result = result.replace(legacy, canonical)
    if not result or ".." in result or result.startswith(".") or result.endswith("."):
        raise ContractError(f"could not canonicalize module name: {name!r}")
    return result


def canonicalize_adapter_tensor_name(name: str) -> tuple[str, str]:
    match = _ADAPTER_NAME.fullmatch(_require_string(name, "adapter tensor name"))
    if match is None:
        raise ContractError(f"unsupported adapter tensor name: {name}")
    role = f"lora_{match.group('role')}"
    return canonicalize_module_name(match.group("module")), role


def canonical_adapter_inventory(names: Iterable[str]) -> list[str]:
    inventory = [f"{module}:{role}" for module, role in (canonicalize_adapter_tensor_name(name) for name in names)]
    if len(inventory) != len(set(inventory)):
        raise ContractError("adapter contains duplicate canonical tensor names")
    return sorted(inventory)


def validate_target_inventory(
    lock: Mapping[str, Any],
    model_key: str,
    target_preset: str,
    modules: Iterable[str],
) -> dict[str, int]:
    """Require the complete model-specific target inventory for one preset."""
    try:
        expected_raw = lock["target_inventory"][model_key][target_preset]
    except (KeyError, TypeError) as exc:
        raise ContractError(f"no locked target inventory for {model_key}/{target_preset}") from exc
    expected = {
        _require_string(suffix, "target inventory suffix"): _require_int(
            count,
            f"target_inventory.{model_key}.{target_preset}.{suffix}",
            minimum=1,
        )
        for suffix, count in _require_mapping(expected_raw, "target inventory").items()
    }
    actual = {suffix: 0 for suffix in expected}
    canonical_modules = [canonicalize_module_name(module) for module in modules]
    if len(canonical_modules) != len(set(canonical_modules)):
        raise ContractError("target module inventory contains duplicates")
    suffixes = sorted(expected, key=len, reverse=True)
    unmatched: list[str] = []
    for module in canonical_modules:
        matched = next(
            (suffix for suffix in suffixes if module == suffix or module.endswith("." + suffix)),
            None,
        )
        if matched is None:
            unmatched.append(module)
        else:
            actual[matched] += 1
    if unmatched or actual != expected:
        raise ContractError(
            f"incomplete target inventory for {model_key}/{target_preset} "
            f"(expected={expected}, actual={actual}, unmatched={unmatched})"
        )
    return actual


def _shape_product(shape: Sequence[int]) -> int:
    product = 1
    for dimension in shape:
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ContractError(f"invalid tensor shape: {shape!r}")
        product *= dimension
    return product


@dataclass(frozen=True)
class TensorEntry:
    shape: tuple[int, ...]
    dtype: str
    storage_key: str | None
    inline_values: tuple[float, ...] | None


class TensorStore:
    """Read either small inline test tensors or a hash-bound Safetensors file."""

    def __init__(self, trace_path: Path, descriptor: Mapping[str, Any]):
        self.trace_path = trace_path.resolve()
        self.descriptor = dict(descriptor)
        require_exact_keys(self.descriptor, ("format", "entries") if self.descriptor.get("format") == "inline-f64/v1" else ("format", "path", "sha256", "entries"), where="tensor_store")
        self.format = self.descriptor.get("format")
        if self.format not in ("inline-f64/v1", "safetensors/v1"):
            raise ContractError(f"unsupported tensor store format: {self.format!r}")
        raw_entries = _require_mapping(self.descriptor["entries"], "tensor_store.entries")
        if not raw_entries:
            raise ContractError("tensor_store.entries cannot be empty")
        entries: dict[str, TensorEntry] = {}
        storage_keys: set[str] = set()
        for logical_name, raw_entry in raw_entries.items():
            entry = _require_mapping(raw_entry, f"tensor_store.entries.{logical_name}")
            expected = ("shape", "dtype", "values") if self.format == "inline-f64/v1" else ("shape", "dtype", "storage_key")
            require_exact_keys(entry, expected, where=f"tensor_store.entries.{logical_name}")
            shape = tuple(_require_int(dim, f"{logical_name}.shape", minimum=1) for dim in _require_list(entry["shape"], f"{logical_name}.shape"))
            if not shape:
                raise ContractError(f"{logical_name}: scalar tensors are not part of the LoRA trace contract")
            dtype = _require_string(entry["dtype"], f"{logical_name}.dtype")
            if self.format == "inline-f64/v1":
                if dtype != "float64":
                    raise ContractError(f"{logical_name}: inline-f64/v1 requires dtype=float64")
                values = tuple(_finite_float(item, f"{logical_name}.values") for item in _require_list(entry["values"], f"{logical_name}.values"))
                if len(values) != _shape_product(shape):
                    raise ContractError(f"{logical_name}: inline value count does not match shape")
                entries[logical_name] = TensorEntry(shape, dtype, None, values)
            else:
                if dtype != "float32":
                    raise ContractError(f"{logical_name}: v1 release state tensors require dtype=float32")
                storage_key = _require_string(entry["storage_key"], f"{logical_name}.storage_key")
                if storage_key in storage_keys:
                    raise ContractError(f"Safetensors storage key is reused: {storage_key}")
                storage_keys.add(storage_key)
                entries[logical_name] = TensorEntry(shape, dtype, storage_key, None)
        self.entries = entries
        self.storage_keys = storage_keys
        self.external_path: Path | None = None
        if self.format == "safetensors/v1":
            relative_text = _require_string(self.descriptor["path"], "tensor_store.path")
            relative = Path(relative_text)
            if relative_text != relative.as_posix() or relative.is_absolute() or ".." in relative.parts:
                raise ContractError("tensor_store.path must be a normalized safe relative path")
            external_candidate = self.trace_path.parent / relative
            if external_candidate.is_symlink():
                raise ContractError("tensor_store.path cannot reference a symlink")
            external = external_candidate.resolve()
            if not external.is_relative_to(self.trace_path.parent):
                raise ContractError("tensor_store.path escapes the trace directory")
            if not external.is_file():
                raise ContractError(f"tensor store file is missing: {external}")
            expected_sha = _require_string(self.descriptor["sha256"], "tensor_store.sha256")
            if expected_sha != prefixed_sha256(external):
                raise ContractError("tensor store SHA-256 mismatch")
            self.external_path = external

    def get(self, logical_name: str) -> tuple[tuple[int, ...], list[float]]:
        try:
            entry = self.entries[logical_name]
        except KeyError as exc:
            raise ContractError(f"tensor store is missing logical tensor {logical_name}") from exc
        if entry.inline_values is not None:
            return entry.shape, list(entry.inline_values)
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise ContractError("safetensors is required to read an external trace tensor store") from exc
        assert self.external_path is not None and entry.storage_key is not None
        with safe_open(str(self.external_path), framework="np", device="cpu") as source:
            actual_keys = set(source.keys())
            if actual_keys != self.storage_keys:
                missing = sorted(self.storage_keys - actual_keys)
                uncommitted = sorted(actual_keys - self.storage_keys)
                raise ContractError(
                    f"Safetensors keys differ from the trace manifest "
                    f"(missing={missing}, uncommitted={uncommitted})"
                )
            if entry.storage_key not in actual_keys:
                raise ContractError(f"Safetensors store is missing {entry.storage_key}")
            tensor = source.get_tensor(entry.storage_key)
        actual_shape = tuple(int(dim) for dim in tensor.shape)
        if actual_shape != entry.shape:
            raise ContractError(f"{logical_name}: Safetensors shape {actual_shape} != manifest {entry.shape}")
        actual_dtype = str(tensor.dtype)
        if actual_dtype != entry.dtype:
            raise ContractError(
                f"{logical_name}: Safetensors dtype {actual_dtype!r} != manifest {entry.dtype!r}"
            )
        values = [float(value) for value in tensor.reshape(-1)]
        if any(not math.isfinite(value) for value in values):
            raise ContractError(f"{logical_name}: tensor contains non-finite values")
        return entry.shape, values


@dataclass
class ValidatedTrace:
    path: Path
    payload: dict[str, Any]
    tensors: TensorStore
    trace_sha256: str
    evidence_manifest_sha256: str | None
    recomputed_grad_norm: float


def validate_trace(
    path: Path,
    lock: Mapping[str, Any],
    *,
    lock_path: Path = LOCK_PATH,
    allow_synthetic: bool = False,
) -> ValidatedTrace:
    trace_candidate = path.expanduser()
    if trace_candidate.is_symlink():
        raise ContractError(f"trace must not be a symlink: {trace_candidate}")
    trace_path = trace_candidate.resolve()
    if not trace_path.is_file():
        raise ContractError(f"trace must be a regular file: {trace_path}")
    trace_sha256 = prefixed_sha256(trace_path)
    payload = dict(_require_mapping(load_json(trace_path), "trace"))
    require_exact_keys(
        payload,
        (
            "schema_version", "producer", "oracle_lock_sha256", "model", "prepared",
            "training", "metrics", "logit_probes", "target_tensors", "tensor_store", "artifact",
        ),
        where="trace",
    )
    if payload["schema_version"] != TRACE_SCHEMA_VERSION:
        raise ContractError(f"unsupported trace schema: {payload['schema_version']!r}")
    if payload["oracle_lock_sha256"] != lock_digest(lock_path):
        raise ContractError("trace was produced with a different oracle lock")

    producer = _require_mapping(payload["producer"], "producer")
    require_exact_keys(producer, ("name", "version", "source_revision", "hardware"), where="producer")
    if producer["name"] not in ("hf-peft", "antfly-zig-native", "antfly-zig-metal", "synthetic-test"):
        raise ContractError(f"unsupported trace producer: {producer['name']!r}")
    _require_string(producer["version"], "producer.version")
    _require_string(producer["source_revision"], "producer.source_revision")
    producer_hardware = _require_mapping(producer["hardware"], "producer.hardware")
    evidence_manifest_sha256: str | None = None
    if producer["name"] == "synthetic-test":
        if not allow_synthetic:
            raise ContractError("synthetic-test traces are accepted only by explicit test code")
        if (trace_path.parent / "COMPLETE.json").exists():
            evidence_manifest_sha256, _ = validate_evidence_ledger(trace_path)
    else:
        evidence_manifest_sha256, _ = validate_evidence_ledger(trace_path)
    if producer["name"] == "hf-peft":
        expected_revision = lock["python_oracle"]["source_revisions"]["transformers"]
        if producer["source_revision"] != expected_revision:
            raise ContractError("HF trace source revision does not match the lock")
        expected_version = ";".join(
            f"{name}={version}"
            for name, version in sorted(lock["python_oracle"]["packages"].items())
        )
        if producer["version"] != expected_version:
            raise ContractError("HF trace package versions do not match the lock")
        execution = lock["python_oracle"]["execution"]
        if (
            producer_hardware.get("torch_device") != execution["device"]
            or producer_hardware.get("dtype") != execution["dtype"]
        ):
            raise ContractError("HF trace device/dtype do not match the lock")
        _require_string(producer_hardware.get("cuda_device"), "producer.hardware.cuda_device")
        _require_string(producer_hardware.get("cuda_runtime"), "producer.hardware.cuda_runtime")
    elif producer["name"] in ("antfly-zig-native", "antfly-zig-metal"):
        if _HEX40.fullmatch(producer["source_revision"]) is None:
            raise ContractError("Antfly trace source_revision must be a full clean Git commit")
        expected_backend = "native" if producer["name"] == "antfly-zig-native" else "metal"
        executable_sha256 = producer_hardware.get("executable_sha256")
        if (
            producer_hardware.get("backend") != expected_backend
            or not isinstance(executable_sha256, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", executable_sha256) is None
            or producer_hardware.get("git_dirty") is not False
        ):
            raise ContractError(
                "Antfly trace must bind the matching backend, a clean source tree, and executable SHA-256"
            )
        if producer_hardware.get("platform") != "Darwin" or producer_hardware.get("machine") != "arm64":
            raise ContractError("Antfly Gemma4 release traces require Darwin arm64 hardware identity")
        for field in ("os_version", "os_build", "chip"):
            _require_string(producer_hardware.get(field), f"producer.hardware.{field}")
        _require_int(producer_hardware.get("memory_bytes"), "producer.hardware.memory_bytes", minimum=1)
        if producer["name"] == "antfly-zig-metal":
            _require_string(producer_hardware.get("metal_device"), "producer.hardware.metal_device")

    model = _require_mapping(payload["model"], "model")
    require_exact_keys(model, ("key", "repo_id", "revision", "local_artifact_sha256"), where="model")
    model_key = _require_string(model["key"], "model.key")
    if model_key not in lock["models"]:
        raise ContractError(f"trace model is not locked: {model_key}")
    locked_model = lock["models"][model_key]
    if model["repo_id"] != locked_model["repo_id"] or model["revision"] != locked_model["revision"]:
        raise ContractError("trace model identity does not match the lock")
    if not isinstance(model["local_artifact_sha256"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", model["local_artifact_sha256"]):
        raise ContractError("model.local_artifact_sha256 is malformed")

    prepared = _require_mapping(payload["prepared"], "prepared")
    prepared_keys = (
        "schema_version", "artifact_sha256", "example_index", "input_ids", "labels",
        "source_dataset_sha256", "source_split", "source_revision", "source_id",
        "source_group_id", "source_name", "source_record_sha256", "rendered_chat_sha256",
    )
    require_exact_keys(prepared, prepared_keys, where="prepared")
    if prepared["schema_version"] != PREPARED_SCHEMA_VERSION:
        raise ContractError("trace prepared schema is not v6")
    if not isinstance(prepared["artifact_sha256"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", prepared["artifact_sha256"]):
        raise ContractError("prepared.artifact_sha256 is malformed")
    _require_int(prepared["example_index"], "prepared.example_index", minimum=0)
    input_ids = _validate_int_array(prepared["input_ids"], "prepared.input_ids")
    labels = _validate_int_array(prepared["labels"], "prepared.labels", labels=True)
    if len(input_ids) != len(labels):
        raise ContractError("trace input_ids and labels lengths differ")
    for field in ("source_dataset_sha256", "source_record_sha256", "rendered_chat_sha256"):
        if not isinstance(prepared[field], str) or not _HEX64.fullmatch(prepared[field]):
            raise ContractError(f"prepared.{field} is malformed")
    for field in ("source_split", "source_revision", "source_id", "source_group_id", "source_name"):
        _require_string(prepared[field], f"prepared.{field}")
    supervised_positions = [index for index in range(1, len(labels)) if labels[index] != IGNORE_LABEL]
    if not supervised_positions:
        raise ContractError("trace has no causal supervised token")

    training = _require_mapping(payload["training"], "training")
    training_keys = (
        "optimizer", "seed", "step", "rank", "alpha", "scale", "target_preset", "learning_rate", "betas",
        "eps", "weight_decay", "max_grad_norm", "grad_accum_steps",
        "supervised_token_normalization", "dropout", "use_cache",
    )
    require_exact_keys(training, training_keys, where="training")
    if training["optimizer"] != "adamw":
        raise ContractError("v1 numerical traces require AdamW")
    _require_int(training["seed"], "training.seed", minimum=0)
    _require_int(training["step"], "training.step", minimum=1)
    rank = _require_int(training["rank"], "training.rank", minimum=1)
    alpha = _finite_float(training["alpha"], "training.alpha")
    scale = _finite_float(training["scale"], "training.scale")
    if alpha <= 0 or not math.isclose(scale, alpha / rank, rel_tol=1e-12, abs_tol=1e-12):
        raise ContractError("training.scale must equal alpha/rank")
    if training["target_preset"] not in lock["target_presets"]:
        raise ContractError("training.target_preset is not locked")
    if _finite_float(training["learning_rate"], "training.learning_rate") <= 0:
        raise ContractError("training.learning_rate must be positive")
    betas = _require_list(training["betas"], "training.betas")
    if len(betas) != 2 or not all(0 <= _finite_float(beta, "training.beta") < 1 for beta in betas):
        raise ContractError("training.betas must contain two values in [0,1)")
    if _finite_float(training["eps"], "training.eps") <= 0:
        raise ContractError("training.eps must be positive")
    if _finite_float(training["weight_decay"], "training.weight_decay") < 0:
        raise ContractError("training.weight_decay must be non-negative")
    if _finite_float(training["max_grad_norm"], "training.max_grad_norm") < 0:
        raise ContractError("training.max_grad_norm must be non-negative")
    if training["grad_accum_steps"] != 1 or training["supervised_token_normalization"] != "mean":
        raise ContractError("v1 numerical traces require one-step supervised-token mean semantics")
    if training["dropout"] != 0.0 or training["use_cache"] is not False:
        raise ContractError("v1 numerical traces require dropout=0 and use_cache=false")
    trajectory = lock["training_contract"]
    if (
        training["seed"] != trajectory["seed"]
        or training["step"] not in trajectory["steps"]
        or float(training["learning_rate"]) != float(trajectory["learning_rate"])
        or [float(value) for value in training["betas"]] != [float(value) for value in trajectory["betas"]]
        or float(training["eps"]) != float(trajectory["eps"])
        or float(training["weight_decay"]) != float(trajectory["weight_decay"])
        or float(training["max_grad_norm"]) != float(trajectory["max_grad_norm"])
        or training["grad_accum_steps"] != trajectory["grad_accum_steps"]
        or training["supervised_token_normalization"] != trajectory["supervised_token_normalization"]
    ):
        raise ContractError("trace optimizer trajectory differs from training_contract")

    metrics = _require_mapping(payload["metrics"], "metrics")
    require_exact_keys(metrics, ("loss", "loss_history", "grad_norm", "supervised_tokens"), where="metrics")
    _finite_float(metrics["loss"], "metrics.loss")
    history = [_finite_float(value, "metrics.loss_history[]") for value in _require_list(metrics["loss_history"], "metrics.loss_history")]
    if len(history) != training["step"] or not math.isclose(history[-1], float(metrics["loss"]), rel_tol=0, abs_tol=0):
        raise ContractError("metrics.loss_history must contain one exact entry per step and end at loss")
    claimed_grad_norm = _finite_float(metrics["grad_norm"], "metrics.grad_norm")
    if claimed_grad_norm < 0:
        raise ContractError("metrics.grad_norm must be non-negative")
    if metrics["supervised_tokens"] != len(supervised_positions):
        raise ContractError("metrics.supervised_tokens is inconsistent with labels")

    probe_positions: set[int] = set()
    for index, raw_probe in enumerate(_require_list(payload["logit_probes"], "logit_probes")):
        probe = _require_mapping(raw_probe, f"logit_probes[{index}]")
        require_exact_keys(probe, ("predictor_position", "target_token_id", "token_ids", "values", "logsumexp"), where=f"logit_probes[{index}]")
        predictor = _require_int(probe["predictor_position"], f"logit_probes[{index}].predictor_position", minimum=0)
        if predictor in probe_positions:
            raise ContractError("duplicate logit predictor position")
        probe_positions.add(predictor)
        if predictor + 1 >= len(labels) or labels[predictor + 1] == IGNORE_LABEL:
            raise ContractError("logit probe does not correspond to a supervised causal target")
        if probe["target_token_id"] != labels[predictor + 1]:
            raise ContractError("logit probe target does not match prepared labels")
        token_ids = _validate_int_array(probe["token_ids"], f"logit_probes[{index}].token_ids")
        if len(set(token_ids)) != len(token_ids):
            raise ContractError("logit probe token ids must be unique")
        values = [_finite_float(value, f"logit_probes[{index}].values") for value in _require_list(probe["values"], f"logit_probes[{index}].values")]
        if len(values) != len(token_ids) or probe["target_token_id"] not in token_ids:
            raise ContractError("logit probe values/token ids are inconsistent")
        _finite_float(probe["logsumexp"], f"logit_probes[{index}].logsumexp")
    if probe_positions != {position - 1 for position in supervised_positions}:
        raise ContractError("trace must probe every supervised causal position exactly once")

    tensor_store_descriptor = _require_mapping(payload["tensor_store"], "tensor_store")
    expected_store_format = "inline-f64/v1" if producer["name"] == "synthetic-test" else "safetensors/v1"
    if tensor_store_descriptor.get("format") != expected_store_format:
        raise ContractError(
            f"producer {producer['name']} requires tensor_store.format={expected_store_format}"
        )
    tensor_store = TensorStore(trace_path, tensor_store_descriptor)
    target_tensors = _require_list(payload["target_tensors"], "target_tensors")
    identities: set[tuple[str, str]] = set()
    modules: dict[str, set[str]] = {}
    inventory: list[str] = []
    logical_names: set[str] = set()
    global_gradient_sq = 0.0
    for index, raw_target in enumerate(target_tensors):
        target = _require_mapping(raw_target, f"target_tensors[{index}]")
        require_exact_keys(target, ("canonical_name", "source_name", "role", "shape", "gradient_expectation", "logical_tensors"), where=f"target_tensors[{index}]")
        canonical = canonicalize_module_name(_require_string(target["canonical_name"], f"target_tensors[{index}].canonical_name"))
        role = target["role"]
        if role not in ("lora_A", "lora_B"):
            raise ContractError(f"target_tensors[{index}].role is unsupported")
        source_identity = canonicalize_adapter_tensor_name(_require_string(target["source_name"], f"target_tensors[{index}].source_name"))
        if source_identity != (canonical, role):
            raise ContractError(f"target_tensors[{index}] source/canonical names disagree")
        identity = (canonical, role)
        if identity in identities:
            raise ContractError(f"duplicate target tensor: {identity}")
        identities.add(identity)
        modules.setdefault(canonical, set()).add(role)
        inventory.append(f"{canonical}:{role}")
        shape = tuple(_require_int(dim, f"target_tensors[{index}].shape", minimum=1) for dim in _require_list(target["shape"], f"target_tensors[{index}].shape"))
        if len(shape) != 2 or (role == "lora_A" and shape[0] != rank) or (role == "lora_B" and shape[1] != rank):
            raise ContractError(f"target_tensors[{index}] shape is incompatible with rank {rank}")
        if target["gradient_expectation"] not in ("active", "zero-by-zero-b-initialization"):
            raise ContractError("unsupported gradient expectation")
        logical = _require_mapping(target["logical_tensors"], f"target_tensors[{index}].logical_tensors")
        require_exact_keys(logical, ("initial", "gradient", "updated", "optimizer_m", "optimizer_v"), where=f"target_tensors[{index}].logical_tensors")
        gradient: list[float] | None = None
        for state_name, logical_name in logical.items():
            logical_name = _require_string(logical_name, f"target_tensors[{index}].logical_tensors.{state_name}")
            if logical_name in logical_names:
                raise ContractError(f"logical tensor is reused: {logical_name}")
            logical_names.add(logical_name)
            stored_shape, values = tensor_store.get(logical_name)
            if stored_shape != shape:
                raise ContractError(f"{logical_name}: shape does not match target manifest")
            if any(not math.isfinite(value) for value in values):
                raise ContractError(f"{logical_name}: non-finite value")
            if state_name == "gradient":
                gradient = values
        assert gradient is not None
        gradient_sq = math.fsum(value * value for value in gradient)
        global_gradient_sq = math.fsum((global_gradient_sq, gradient_sq))
        gradient_norm = math.sqrt(gradient_sq)
        if target["gradient_expectation"] == "active" and gradient_norm == 0:
            raise ContractError(f"active target has an all-zero gradient: {canonical}:{role}")
        if target["gradient_expectation"] == "zero-by-zero-b-initialization" and gradient_norm != 0:
            raise ContractError(f"expected-zero target has a nonzero gradient: {canonical}:{role}")
    if any(roles != {"lora_A", "lora_B"} for roles in modules.values()):
        raise ContractError("every target module must contain exactly one LoRA A/B pair")
    recomputed_grad_norm = math.sqrt(global_gradient_sq)
    if not math.isclose(claimed_grad_norm, recomputed_grad_norm, rel_tol=1e-9, abs_tol=1e-12):
        raise ContractError(
            f"metrics.grad_norm is not bound to the serialized raw gradients "
            f"(reported={claimed_grad_norm}, recomputed={recomputed_grad_norm})"
        )
    artifact = _require_mapping(payload["artifact"], "artifact")
    require_exact_keys(
        artifact,
        ("adapter_config_semantics", "adapter_model_sha256", "tensor_inventory", "key_layout", "policy_source"),
        where="artifact",
    )
    semantics = _require_mapping(artifact["adapter_config_semantics"], "artifact.adapter_config_semantics")
    require_exact_keys(semantics, ("peft_type", "task_type", "r", "lora_alpha", "target_preset", "target_modules", "use_dora", "lora_dropout"), where="artifact.adapter_config_semantics")
    if semantics["peft_type"] != "LORA" or semantics["task_type"] != "CAUSAL_LM":
        raise ContractError("artifact adapter is not PEFT causal-LM LoRA")
    if semantics["r"] != rank or float(semantics["lora_alpha"]) != alpha:
        raise ContractError("artifact adapter rank/alpha do not match training")
    if semantics["target_preset"] != training["target_preset"]:
        raise ContractError("artifact target preset does not match training")
    if semantics["use_dora"] is not False or float(semantics["lora_dropout"]) != 0.0:
        raise ContractError("artifact uses unsupported DoRA or LoRA dropout")
    configured_modules = sorted(canonicalize_module_name(item) for item in _require_list(semantics["target_modules"], "artifact.target_modules"))
    if configured_modules != sorted(modules):
        raise ContractError("adapter config target modules do not exactly match the tensor inventory")
    validate_target_inventory(
        lock,
        model_key,
        training["target_preset"],
        configured_modules,
    )
    if not isinstance(artifact["adapter_model_sha256"], str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", artifact["adapter_model_sha256"]):
        raise ContractError("artifact.adapter_model_sha256 is malformed")
    artifact_inventory = _require_list(artifact["tensor_inventory"], "artifact.tensor_inventory")
    if artifact_inventory != sorted(inventory) or len(artifact_inventory) != len(set(artifact_inventory)):
        raise ContractError("artifact tensor inventory does not match target tensors")
    if artifact["key_layout"] not in (STOCK_PEFT_KEY_FORMAT, ANTFLY_ADAPTER_KEY_FORMAT):
        raise ContractError("artifact key layout is unsupported")
    if artifact["policy_source"] not in (
        "explicit-lock-policy",
        "antfly-finetune-manifest/v2",
        "antfly-finetune-manifest/v3",
        "antfly-peft-export/v1",
    ):
        raise ContractError("artifact target policy source is unsupported")
    if prefixed_sha256(trace_path) != trace_sha256:
        raise ContractError("trace changed while it was being validated")
    return ValidatedTrace(
        trace_path,
        payload,
        tensor_store,
        trace_sha256,
        evidence_manifest_sha256,
        recomputed_grad_norm,
    )


@dataclass(frozen=True)
class VectorMetrics:
    count: int
    max_abs: float
    rel_l2: float
    cosine: float


def vector_metrics(reference: Sequence[float], candidate: Sequence[float]) -> VectorMetrics:
    if len(reference) != len(candidate) or not reference:
        raise ContractError("numeric vectors must be non-empty and have equal lengths")
    if any(not math.isfinite(float(value)) for value in reference) or any(
        not math.isfinite(float(value)) for value in candidate
    ):
        raise ContractError("numeric vectors must contain only finite values")
    diff_sq = sum((float(right) - float(left)) ** 2 for left, right in zip(reference, candidate))
    ref_sq = sum(float(value) ** 2 for value in reference)
    cand_sq = sum(float(value) ** 2 for value in candidate)
    dot = sum(float(left) * float(right) for left, right in zip(reference, candidate))
    max_abs = max(abs(float(right) - float(left)) for left, right in zip(reference, candidate))
    if ref_sq == 0 and cand_sq == 0:
        cosine = 1.0
    elif ref_sq == 0 or cand_sq == 0:
        cosine = 0.0
    else:
        cosine = dot / math.sqrt(ref_sq * cand_sq)
        cosine = max(-1.0, min(1.0, cosine))
    rel_l2 = math.sqrt(diff_sq) / max(math.sqrt(ref_sq), 1e-30)
    return VectorMetrics(len(reference), max_abs, rel_l2, cosine)


def scalar_within(reference: float, candidate: float, *, abs_tol: float, rel_tol: float) -> tuple[bool, float]:
    bound = max(abs_tol, rel_tol * abs(reference))
    return abs(candidate - reference) <= bound, bound


def _comparison_contract(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": payload["model"],
        "prepared": payload["prepared"],
        "training": payload["training"],
    }


_PROFILE_PRODUCER_PAIRS: dict[str, set[tuple[str, str]]] = {
    "tiny-f32": {("synthetic-test", "synthetic-test")},
    "native-metal-bf16": {("antfly-zig-native", "antfly-zig-metal")},
    "hf-zig-bf16": {("hf-peft", "antfly-zig-metal")},
    "resume": {
        ("antfly-zig-native", "antfly-zig-native"),
        ("antfly-zig-metal", "antfly-zig-metal"),
    },
}


def _require_profile_producer_pair(
    reference: ValidatedTrace,
    candidate: ValidatedTrace,
    profile_name: str,
) -> None:
    pair = (
        str(reference.payload["producer"]["name"]),
        str(candidate.payload["producer"]["name"]),
    )
    allowed = _PROFILE_PRODUCER_PAIRS.get(profile_name)
    if allowed is None or pair not in allowed:
        raise ContractError(
            f"profile {profile_name!r} does not admit producer pair {pair!r}; "
            f"expected one of {sorted(allowed or ())}"
        )


def compare_traces(
    reference: ValidatedTrace,
    candidate: ValidatedTrace,
    lock: Mapping[str, Any],
    profile_name: str,
) -> dict[str, Any]:
    profiles = _require_mapping(lock["tolerance_profiles"], "tolerance_profiles")
    if profile_name not in profiles:
        raise ContractError(f"unknown tolerance profile: {profile_name}")
    _require_profile_producer_pair(reference, candidate, profile_name)
    profile = profiles[profile_name]
    ref_producer = reference.payload["producer"]
    cand_producer = candidate.payload["producer"]
    if profile_name == "native-metal-bf16":
        if ref_producer["source_revision"] != cand_producer["source_revision"]:
            raise ContractError("native-metal-bf16 requires one Antfly source commit")
        identity_fields = ("platform", "machine", "os_version", "os_build", "chip", "memory_bytes")
        ref_identity = tuple(ref_producer["hardware"].get(field) for field in identity_fields)
        cand_identity = tuple(cand_producer["hardware"].get(field) for field in identity_fields)
        if ref_identity != cand_identity:
            raise ContractError("native-metal-bf16 requires byte-equivalent host hardware identity")
    elif profile_name == "resume":
        if (
            ref_producer["version"] != cand_producer["version"]
            or ref_producer["source_revision"] != cand_producer["source_revision"]
            or ref_producer["hardware"] != cand_producer["hardware"]
        ):
            raise ContractError("resume comparison requires the exact same binary and hardware identity")

    failures: list[str] = []
    comparisons: dict[str, Any] = {}
    contract_equal = _comparison_contract(reference.payload) == _comparison_contract(candidate.payload)
    if not contract_equal:
        failures.append("model/prepared/training contracts differ")
    comparisons["exact_contract"] = contract_equal

    ref_metrics = reference.payload["metrics"]
    cand_metrics = candidate.payload["metrics"]
    loss_ok, loss_bound = scalar_within(
        float(ref_metrics["loss"]),
        float(cand_metrics["loss"]),
        abs_tol=float(profile["loss_abs"]),
        rel_tol=float(profile["loss_rel"]),
    )
    comparisons["loss"] = {
        "reference": ref_metrics["loss"],
        "candidate": cand_metrics["loss"],
        "absolute_delta": abs(float(cand_metrics["loss"]) - float(ref_metrics["loss"])),
        "bound": loss_bound,
        "ok": loss_ok,
    }
    if not loss_ok:
        failures.append("loss exceeded tolerance")
    grad_norm_ok, grad_norm_bound = scalar_within(
        reference.recomputed_grad_norm,
        candidate.recomputed_grad_norm,
        abs_tol=float(profile["grad_norm_abs"]),
        rel_tol=float(profile["grad_norm_rel"]),
    )
    comparisons["grad_norm"] = {
        "reference": reference.recomputed_grad_norm,
        "candidate": candidate.recomputed_grad_norm,
        "absolute_delta": abs(candidate.recomputed_grad_norm - reference.recomputed_grad_norm),
        "bound": grad_norm_bound,
        "ok": grad_norm_ok,
    }
    if not grad_norm_ok:
        failures.append("global raw-gradient norm exceeded tolerance")
    if len(ref_metrics["loss_history"]) != len(cand_metrics["loss_history"]):
        failures.append("loss history lengths differ")
    else:
        history_rows = []
        for index, (left, right) in enumerate(zip(ref_metrics["loss_history"], cand_metrics["loss_history"])):
            ok, bound = scalar_within(float(left), float(right), abs_tol=float(profile["loss_abs"]), rel_tol=float(profile["loss_rel"]))
            history_rows.append({"step": index + 1, "absolute_delta": abs(float(right) - float(left)), "bound": bound, "ok": ok})
            if not ok:
                failures.append(f"loss history step {index + 1} exceeded tolerance")
        comparisons["loss_history"] = history_rows

    ref_probes = {row["predictor_position"]: row for row in reference.payload["logit_probes"]}
    cand_probes = {row["predictor_position"]: row for row in candidate.payload["logit_probes"]}
    if set(ref_probes) != set(cand_probes):
        failures.append("logit probe positions differ")
    probe_rows = []
    for position in sorted(set(ref_probes) & set(cand_probes)):
        left = ref_probes[position]
        right = cand_probes[position]
        if left["token_ids"] != right["token_ids"] or left["target_token_id"] != right["target_token_id"]:
            failures.append(f"logit probe token contract differs at position {position}")
            continue
        metrics = vector_metrics(left["values"], right["values"])
        lse_ok, lse_bound = scalar_within(float(left["logsumexp"]), float(right["logsumexp"]), abs_tol=float(profile["loss_abs"]), rel_tol=float(profile["loss_rel"]))
        ok = (
            metrics.max_abs <= profile["logits_max_abs"]
            and metrics.rel_l2 <= profile["logits_rel_l2"]
            and metrics.cosine >= profile["logits_cosine_min"]
            and lse_ok
        )
        probe_rows.append({
            "predictor_position": position,
            "rel_l2": metrics.rel_l2,
            "cosine": metrics.cosine,
            "max_abs": metrics.max_abs,
            "max_abs_limit": profile["logits_max_abs"],
            "logsumexp_absolute_delta": abs(float(right["logsumexp"]) - float(left["logsumexp"])),
            "logsumexp_bound": lse_bound,
            "ok": ok,
        })
        if not ok:
            failures.append(f"logit probe {position} exceeded tolerance")
    comparisons["logit_probes"] = probe_rows

    def targets(trace: ValidatedTrace) -> dict[tuple[str, str], Mapping[str, Any]]:
        return {(row["canonical_name"], row["role"]): row for row in trace.payload["target_tensors"]}

    ref_targets = targets(reference)
    cand_targets = targets(candidate)
    if set(ref_targets) != set(cand_targets):
        missing = sorted(set(ref_targets) - set(cand_targets))
        extra = sorted(set(cand_targets) - set(ref_targets))
        failures.append(f"target tensor inventory differs (missing={missing}, extra={extra})")
    target_rows = []
    for identity in sorted(set(ref_targets) & set(cand_targets)):
        left = ref_targets[identity]
        right = cand_targets[identity]
        if left["shape"] != right["shape"] or left["gradient_expectation"] != right["gradient_expectation"]:
            failures.append(f"target metadata differs for {identity[0]}:{identity[1]}")
            continue
        row: dict[str, Any] = {"canonical_name": identity[0], "role": identity[1], "states": {}}
        for state_name in ("initial", "gradient", "updated", "optimizer_m", "optimizer_v"):
            _, left_values = reference.tensors.get(left["logical_tensors"][state_name])
            _, right_values = candidate.tensors.get(right["logical_tensors"][state_name])
            metrics = vector_metrics(left_values, right_values)
            if state_name == "gradient":
                max_abs_limit = profile["gradient_max_abs"]
                rel_limit = profile["gradient_rel_l2"]
                cosine_limit = profile["gradient_cosine_min"]
            else:
                max_abs_limit = profile["state_max_abs"]
                rel_limit = profile["state_rel_l2"]
                cosine_limit = profile["state_cosine_min"]
            ok = (
                metrics.max_abs <= max_abs_limit
                and metrics.rel_l2 <= rel_limit
                and metrics.cosine >= cosine_limit
            )
            row["states"][state_name] = {
                "count": metrics.count,
                "rel_l2": metrics.rel_l2,
                "cosine": metrics.cosine,
                "max_abs": metrics.max_abs,
                "max_abs_limit": max_abs_limit,
                "ok": ok,
            }
            if not ok:
                failures.append(f"{identity[0]}:{identity[1]} {state_name} exceeded tolerance")
        target_rows.append(row)
    comparisons["target_tensors"] = target_rows

    ref_artifact = reference.payload["artifact"]
    cand_artifact = candidate.payload["artifact"]
    artifact_semantics_equal = (
        ref_artifact["adapter_config_semantics"] == cand_artifact["adapter_config_semantics"]
        and ref_artifact["tensor_inventory"] == cand_artifact["tensor_inventory"]
    )
    comparisons["artifact_semantics"] = {
        "ok": artifact_semantics_equal,
        "byte_hashes_equal": ref_artifact["adapter_model_sha256"] == cand_artifact["adapter_model_sha256"],
        "reference_key_layout": ref_artifact["key_layout"],
        "candidate_key_layout": cand_artifact["key_layout"],
        "canonical_name_normalization_applied": ref_artifact["key_layout"] != cand_artifact["key_layout"],
        "direct_bidirectional_interoperability_proven": False,
        "note": "byte hashes and normalized key parity are evidence; direct stock PEFT load remains a separate real-model gate",
    }
    if not artifact_semantics_equal:
        failures.append("adapter artifact semantics differ")
    return {
        "schema_version": "antfly_gemma4_lora_comparison/v1",
        "ok": not failures,
        "profile": profile_name,
        "reference_trace": str(reference.path),
        "candidate_trace": str(candidate.path),
        "reference_trace_sha256": reference.trace_sha256,
        "candidate_trace_sha256": candidate.trace_sha256,
        "reference_evidence_manifest_sha256": reference.evidence_manifest_sha256,
        "candidate_evidence_manifest_sha256": candidate.evidence_manifest_sha256,
        "reference_producer": reference.payload["producer"],
        "candidate_producer": candidate.payload["producer"],
        "comparisons": comparisons,
        "failures": failures,
    }


def hardware_fingerprint() -> dict[str, Any]:
    """Return auditable host identity; benchmark matching is stricter."""
    result: dict[str, Any] = {
        "platform": platform.system(),
        "machine": platform.machine(),
        "os_version": platform.mac_ver()[0] if platform.system() == "Darwin" else platform.release(),
        "python": platform.python_version(),
    }
    if platform.system() == "Darwin":
        for key, sysctl_name in (
            ("chip", "machdep.cpu.brand_string"),
            ("memory_bytes", "hw.memsize"),
            ("os_build", "kern.osversion"),
        ):
            query = subprocess.run(
                ["sysctl", "-n", sysctl_name],
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            value = query.stdout.strip() if query.returncode == 0 else ""
            if value:
                result[key] = int(value) if key == "memory_bytes" else value
    return result


def write_json(path: Path, payload: Any) -> None:
    """Write canonical JSON without replacing an existing evidence file."""
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(target, flags, 0o644)
    except FileExistsError as exc:
        raise ContractError(f"refusing to replace existing evidence: {target}") from exc
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        # Keep an incomplete, visibly unparseable artifact rather than risk
        # replacing evidence from a previous run.
        raise


def _read_antfly_adapter_manifest(adapter_dir: Path) -> dict[str, Any] | None:
    path = adapter_dir.expanduser().resolve() / "antfly_finetune_manifest.json"
    if not path.exists():
        return None
    manifest = dict(_require_mapping(load_json(path), "antfly_finetune_manifest"))
    fields = (
        "schema_version", "status", "artifact_family_version", "base_model_name_or_path",
        "base_model_sha256", "tokenizer_sha256", "chat_template_sha256", "target_modules",
        "target_preset", "rank", "alpha", "use_dora", "use_rslora", "initializer", "recursive_lora",
        "tensor_key_format", "adapter_checkpoint_sha256", "adapter_checkpoint_size_bytes",
    )
    schema_version = manifest.get("schema_version")
    if schema_version not in ANTFLY_ADAPTER_MANIFEST_SCHEMA_VERSIONS:
        raise ContractError("unsupported Antfly adapter manifest schema")
    if schema_version == ANTFLY_ADAPTER_MANIFEST_SCHEMA_V3:
        require_exact_keys(manifest, fields + ("initialization_seed",), where="antfly_finetune_manifest")
        _require_int(manifest["initialization_seed"], "manifest.initialization_seed", minimum=0)
    else:
        # Older v2 artifacts omit the field. Writers that share the current
        # typed manifest may serialize it as null; accept only that equivalent.
        allowed = set(fields) | {"initialization_seed"}
        unknown = sorted(set(manifest) - allowed)
        missing = sorted(set(fields) - set(manifest))
        if unknown or missing or manifest.get("initialization_seed") is not None:
            raise ContractError(
                "antfly_finetune_manifest keys do not match v2 "
                f"(missing={missing}, unknown={unknown})"
            )
    if manifest["tensor_key_format"] != ANTFLY_ADAPTER_KEY_FORMAT:
        raise ContractError(
            "Antfly adapter manifest tensor_key_format must be "
            f"{ANTFLY_ADAPTER_KEY_FORMAT}"
        )
    if manifest["status"] != "complete" or manifest["artifact_family_version"] != "gemma4_lora/v1alpha1":
        raise ContractError("Antfly adapter manifest is not a complete Gemma4 LoRA artifact")
    _require_string(manifest["base_model_name_or_path"], "manifest.base_model_name_or_path")
    for field in ("base_model_sha256", "tokenizer_sha256", "chat_template_sha256", "adapter_checkpoint_sha256"):
        if not isinstance(manifest[field], str) or not _HEX64.fullmatch(manifest[field]):
            raise ContractError(f"manifest.{field} is missing or malformed")
    adapter_checkpoint = adapter_dir.expanduser().resolve() / "adapter_model.safetensors"
    if not adapter_checkpoint.is_file():
        raise ContractError("Antfly adapter checkpoint is missing")
    checkpoint_size = _require_int(
        manifest["adapter_checkpoint_size_bytes"],
        "manifest.adapter_checkpoint_size_bytes",
        minimum=1,
    )
    if adapter_checkpoint.stat().st_size != checkpoint_size:
        raise ContractError("Antfly adapter checkpoint size does not match its manifest")
    if sha256_file(adapter_checkpoint) != manifest["adapter_checkpoint_sha256"]:
        raise ContractError("Antfly adapter checkpoint SHA-256 does not match its manifest")
    targets = [canonicalize_module_name(item) for item in _require_list(manifest["target_modules"], "manifest.target_modules")]
    if not targets or len(targets) != len(set(targets)):
        raise ContractError("manifest target modules must be non-empty and unique")
    manifest["target_modules"] = targets
    _require_int(manifest["rank"], "manifest.rank", minimum=1)
    if _finite_float(manifest["alpha"], "manifest.alpha") <= 0:
        raise ContractError("manifest.alpha must be positive")
    if manifest["use_dora"] is not False or manifest["use_rslora"] is not False:
        raise ContractError("DoRA/RS-LoRA are outside the Gemma4 v1 oracle lane")
    if manifest["recursive_lora"] is not None:
        raise ContractError("recursive LoRA is outside the standard Gemma4 parity lane")
    if manifest["target_preset"] not in ("peft-qv", "text-all-linear"):
        raise ContractError("manifest.target_preset is missing or unsupported")
    return manifest


def _read_antfly_peft_export_manifest(adapter_dir: Path) -> dict[str, Any] | None:
    root = adapter_dir.expanduser().resolve()
    path = root / "antfly_peft_export.json"
    if not path.exists():
        return None
    manifest = dict(_require_mapping(load_json(path), "antfly_peft_export"))
    fields = (
        "schema_version", "status", "source_artifact_family_version",
        "source_tensor_key_format", "destination_tensor_key_format",
        "source_adapter_model_sha256", "destination_adapter_model_sha256",
        "destination_adapter_model_size_bytes", "adapter_config_sha256",
        "base_model_name_or_path", "base_model_sha256", "tokenizer_sha256",
        "chat_template_sha256", "target_preset", "tensor_count",
    )
    require_exact_keys(manifest, fields, where="antfly_peft_export")
    if manifest["schema_version"] != ANTFLY_PEFT_EXPORT_MANIFEST_SCHEMA_VERSION:
        raise ContractError("unsupported Antfly PEFT export manifest schema")
    if manifest["status"] != "complete" or manifest["source_artifact_family_version"] != "gemma4_lora/v1alpha1":
        raise ContractError("Antfly PEFT export is not a complete Gemma4 LoRA artifact")
    if manifest["source_tensor_key_format"] != ANTFLY_ADAPTER_KEY_FORMAT:
        raise ContractError("Antfly PEFT export source tensor-key format is invalid")
    if manifest["destination_tensor_key_format"] != STOCK_PEFT_KEY_FORMAT:
        raise ContractError("Antfly PEFT export destination tensor-key format is invalid")
    _require_string(manifest["base_model_name_or_path"], "peft_export.base_model_name_or_path")
    for field in (
        "source_adapter_model_sha256", "destination_adapter_model_sha256",
        "adapter_config_sha256", "base_model_sha256", "tokenizer_sha256",
        "chat_template_sha256",
    ):
        if not isinstance(manifest[field], str) or not _HEX64.fullmatch(manifest[field]):
            raise ContractError(f"peft_export.{field} is missing or malformed")
    if manifest["target_preset"] not in ("peft-qv", "text-all-linear"):
        raise ContractError("peft_export.target_preset is missing or unsupported")
    _require_int(manifest["tensor_count"], "peft_export.tensor_count", minimum=2)

    checkpoint = root / "adapter_model.safetensors"
    config = root / "adapter_config.json"
    if not checkpoint.is_file() or not config.is_file():
        raise ContractError("Antfly PEFT export payload is incomplete")
    checkpoint_size = _require_int(
        manifest["destination_adapter_model_size_bytes"],
        "peft_export.destination_adapter_model_size_bytes",
        minimum=1,
    )
    if checkpoint.stat().st_size != checkpoint_size:
        raise ContractError("Antfly PEFT export checkpoint size does not match its manifest")
    if sha256_file(checkpoint) != manifest["destination_adapter_model_sha256"]:
        raise ContractError("Antfly PEFT export checkpoint SHA-256 does not match its manifest")
    if sha256_file(config) != manifest["adapter_config_sha256"]:
        raise ContractError("Antfly PEFT export config SHA-256 does not match its manifest")
    return manifest


def read_adapter_config(adapter_dir: Path, *, target_preset: str | None = None) -> dict[str, Any]:
    path = adapter_dir.expanduser().resolve() / "adapter_config.json"
    config = dict(_require_mapping(load_json(path), "adapter_config"))
    required = ("peft_type", "task_type", "r", "lora_alpha", "target_modules")
    missing = [key for key in required if key not in config]
    if missing:
        raise ContractError(f"adapter_config.json is missing {missing}")
    rank = _require_int(config["r"], "adapter_config.r", minimum=1)
    alpha = _finite_float(config["lora_alpha"], "adapter_config.lora_alpha")
    if alpha <= 0:
        raise ContractError("adapter_config.lora_alpha must be positive")
    if config["peft_type"] != "LORA" or config["task_type"] != "CAUSAL_LM":
        raise ContractError("adapter is not PEFT causal-LM LoRA")
    if config.get("use_dora", False) is not False:
        raise ContractError("DoRA is outside the Gemma4 v1 oracle lane")
    if config.get("use_rslora", False) is not False:
        raise ContractError("RS-LoRA is outside the Gemma4 v1 oracle lane")
    if config.get("modules_to_save") not in (None, []):
        raise ContractError("modules_to_save is outside the standard LoRA parity lane")
    dropout = _finite_float(config.get("lora_dropout", 0.0), "adapter_config.lora_dropout")
    if dropout != 0.0:
        raise ContractError("LoRA dropout must be zero for numerical parity")
    targets = [canonicalize_module_name(item) for item in _require_list(config["target_modules"], "adapter_config.target_modules")]
    if not targets or len(targets) != len(set(targets)):
        raise ContractError("adapter target modules must be non-empty and unique")
    manifest = _read_antfly_adapter_manifest(adapter_dir)
    export_manifest = _read_antfly_peft_export_manifest(adapter_dir)
    if manifest is not None and export_manifest is not None:
        raise ContractError("adapter cannot contain both internal and PEFT export manifests")
    if manifest is not None:
        if target_preset is not None and target_preset != manifest["target_preset"]:
            raise ContractError("explicit target preset conflicts with Antfly adapter manifest")
        initializer = config.get("init_lora_weights", True)
        configured_initializer = None if initializer is True else initializer
        manifest_initializer = manifest["initializer"]
        if (
            config.get("base_model_name_or_path") != manifest["base_model_name_or_path"]
            or rank != manifest["rank"]
            or alpha != float(manifest["alpha"])
            or targets != manifest["target_modules"]
            or configured_initializer != manifest_initializer
        ):
            raise ContractError("Antfly adapter manifest and PEFT config disagree")
        resolved_preset = manifest["target_preset"]
        policy_source = f"antfly-finetune-manifest/{manifest['schema_version'].rsplit('/', 1)[-1]}"
        provenance = {
            "base_model_sha256": manifest["base_model_sha256"],
            "tokenizer_sha256": manifest["tokenizer_sha256"],
            "chat_template_sha256": manifest["chat_template_sha256"],
            "tensor_key_format": manifest["tensor_key_format"],
            "adapter_checkpoint_sha256": manifest["adapter_checkpoint_sha256"],
            "adapter_checkpoint_size_bytes": manifest["adapter_checkpoint_size_bytes"],
            "manifest_sha256": prefixed_sha256(adapter_dir.expanduser().resolve() / "antfly_finetune_manifest.json"),
            "initialization_seed": manifest.get("initialization_seed"),
        }
    elif export_manifest is not None:
        if target_preset is not None and target_preset != export_manifest["target_preset"]:
            raise ContractError("explicit target preset conflicts with Antfly PEFT export manifest")
        if config.get("base_model_name_or_path") != export_manifest["base_model_name_or_path"]:
            raise ContractError("Antfly PEFT export manifest and PEFT config disagree")
        resolved_preset = export_manifest["target_preset"]
        policy_source = "antfly-peft-export/v1"
        provenance = {
            "base_model_sha256": export_manifest["base_model_sha256"],
            "tokenizer_sha256": export_manifest["tokenizer_sha256"],
            "chat_template_sha256": export_manifest["chat_template_sha256"],
            "tensor_key_format": export_manifest["destination_tensor_key_format"],
            "adapter_checkpoint_sha256": export_manifest["destination_adapter_model_sha256"],
            "adapter_checkpoint_size_bytes": export_manifest["destination_adapter_model_size_bytes"],
            "source_adapter_checkpoint_sha256": export_manifest["source_adapter_model_sha256"],
            "tensor_count": export_manifest["tensor_count"],
            "manifest_sha256": prefixed_sha256(adapter_dir.expanduser().resolve() / "antfly_peft_export.json"),
        }
    else:
        if target_preset is None:
            raise ContractError(
                "a stock PEFT adapter without antfly_finetune_manifest.json requires "
                "an explicit target preset"
            )
        if target_preset not in ("peft-qv", "text-all-linear"):
            raise ContractError("explicit target preset is unsupported")
        resolved_preset = target_preset
        policy_source = "explicit-lock-policy"
        provenance = None
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": alpha,
        "target_preset": resolved_preset,
        "target_modules": targets,
        "use_dora": False,
        "lora_dropout": dropout,
        "policy_source": policy_source,
        "provenance": provenance,
    }


def _adapter_key_layout(name: str) -> str:
    if re.search(r"\.weight\.lora_[AB](?:\.[^.]+)?\.weight$", name):
        return ANTFLY_ADAPTER_KEY_FORMAT
    if re.search(r"\.lora_[AB](?:\.[^.]+)?\.weight$", name):
        return STOCK_PEFT_KEY_FORMAT
    raise ContractError(f"unsupported adapter tensor key layout: {name}")


def antfly_to_stock_peft_tensor_name(name: str) -> str:
    """Translate one Antfly weight-qualified LoRA key to stock PEFT layout.

    Antfly keys share the frozen-weight namespace used by the Zig trainer, for
    example ``model.layers.0.self_attn.q_proj.weight.lora_A.weight``. Stock
    PEFT checkpoints omit that frozen ``.weight`` segment and add their model
    wrapper prefix. This is intentionally one-way; reverse loading remains a
    separate round-trip gate.
    """
    if _adapter_key_layout(name) != ANTFLY_ADAPTER_KEY_FORMAT:
        raise ContractError("Antfly-to-PEFT translation requires an Antfly tensor key")
    module, role = canonicalize_adapter_tensor_name(name)
    return f"base_model.model.{module}.{role}.weight"


def inspect_adapter_artifact(adapter_dir: Path, *, target_preset: str | None = None) -> dict[str, Any]:
    root = adapter_dir.expanduser().resolve()
    config = read_adapter_config(root, target_preset=target_preset)
    semantics = {key: config[key] for key in ("peft_type", "task_type", "r", "lora_alpha", "target_preset", "target_modules", "use_dora", "lora_dropout")}
    checkpoint = root / "adapter_model.safetensors"
    if not checkpoint.is_file():
        raise ContractError(f"adapter checkpoint is missing: {checkpoint}")
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise ContractError("safetensors is required to inspect an adapter artifact") from exc
    tensors: dict[tuple[str, str], dict[str, Any]] = {}
    layouts: set[str] = set()
    with safe_open(str(checkpoint), framework="np", device="cpu") as source:
        keys = list(source.keys())
        for key in keys:
            layouts.add(_adapter_key_layout(key))
            identity = canonicalize_adapter_tensor_name(key)
            if identity in tensors:
                raise ContractError(f"duplicate canonical adapter tensor: {identity}")
            tensor = source.get_tensor(key)
            values = [float(value) for value in tensor.reshape(-1)]
            if any(not math.isfinite(value) for value in values):
                raise ContractError(f"adapter tensor contains non-finite data: {key}")
            tensors[identity] = {
                "source_name": key,
                "shape": tuple(int(dim) for dim in tensor.shape),
                "dtype": str(tensor.dtype),
                "values": values,
            }
    modules: dict[str, set[str]] = {}
    for (module, role), entry in tensors.items():
        modules.setdefault(module, set()).add(role)
        shape = entry["shape"]
        if len(shape) != 2:
            raise ContractError(f"adapter tensor is not rank 2: {entry['source_name']}")
        if role == "lora_A" and shape[0] != semantics["r"]:
            raise ContractError(f"LoRA A rank mismatch: {entry['source_name']}")
        if role == "lora_B" and shape[1] != semantics["r"]:
            raise ContractError(f"LoRA B rank mismatch: {entry['source_name']}")
    if not modules or any(roles != {"lora_A", "lora_B"} for roles in modules.values()):
        raise ContractError("adapter must contain one A/B pair for every module")
    if len(layouts) != 1:
        raise ContractError(f"adapter mixes tensor key layouts: {sorted(layouts)}")
    key_layout = next(iter(layouts))
    has_internal_manifest = config["policy_source"] in (
        "antfly-finetune-manifest/v2",
        "antfly-finetune-manifest/v3",
    )
    if has_internal_manifest and key_layout != ANTFLY_ADAPTER_KEY_FORMAT:
        raise ContractError("Antfly manifest declares internal keys but checkpoint uses stock PEFT keys")
    if not has_internal_manifest and key_layout != STOCK_PEFT_KEY_FORMAT:
        raise ContractError("Antfly internal keys require an antfly_finetune_manifest.json sidecar")
    if config["policy_source"] == "antfly-peft-export/v1" and len(keys) != config["provenance"]["tensor_count"]:
        raise ContractError("Antfly PEFT export tensor count does not match its manifest")
    configured_targets = semantics["target_modules"]
    if has_internal_manifest:
        if sorted(modules) != configured_targets:
            raise ContractError("Antfly adapter config targets do not exactly match tensor modules")
    else:
        unmatched_modules = [
            module
            for module in modules
            if not any(module == target or module.endswith("." + target) for target in configured_targets)
        ]
        unused_targets = [
            target
            for target in configured_targets
            if not any(module == target or module.endswith("." + target) for module in modules)
        ]
        if unmatched_modules or unused_targets:
            raise ContractError(
                "stock PEFT adapter config targets do not resolve exactly to its tensor modules "
                f"(unmatched_modules={unmatched_modules}, unused_targets={unused_targets})"
            )
    # Compare resolved tensor modules, not PEFT's potentially suffix-based
    # selection rules. Preserve the serialized rules separately for audit.
    semantics["target_modules"] = sorted(modules)
    return {
        "directory": str(root),
        "checkpoint": str(checkpoint),
        "adapter_model_sha256": prefixed_sha256(checkpoint),
        "semantics": semantics,
        "configured_target_modules": configured_targets,
        "policy_source": config["policy_source"],
        "provenance": config["provenance"],
        "key_layout": key_layout,
        "inventory": sorted(f"{module}:{role}" for module, role in tensors),
        "tensors": tensors,
    }
