"""Bounded portable KV checkpoint transport. Model compatibility is server-validated."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import os
import struct
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Protocol

CONTENT_TYPE = "application/vnd.antfly.attachments.v1"
CONTENT_TYPE_V2 = "application/vnd.antfly.attachments.v2"
CHECKPOINT_MIME = b"application/vnd.antfly.kv-checkpoint"
MAX_METADATA_BYTES = 2 * 1024 * 1024
MAX_CHECKPOINT_BYTES = 2 * 1024 * 1024 * 1024
CHUNK_BYTES = 64 * 1024
AFATT001 = b"AFATT001"
AFATT002 = b"AFATT002"
MAX_PREPARED_METADATA_BYTES = 16 * 1024


@dataclass(frozen=True)
class NativeKvCheckpoint:
    """A checksum-verified checkpoint saved atomically to ``path``.

    ``metadata['outcome']`` distinguishes compaction from an unchanged fallback.
    Fingerprints identify the model revision; the server checks compatibility
    and payload semantics on every resume. This object does not authorize reuse
    with another model or attest task quality.
    """

    path: Path
    metadata: Mapping[str, Any]
    sha256: str
    identity: Mapping[str, str]
    envelope_version: int = 1
    prepared_metadata_base64: str | None = None
    producer_outcome: str | None = None


class Reader(Protocol):
    def read(self, size: int, /) -> bytes: ...


class CheckpointSource(Reader, Protocol):
    """Seekable file descriptor used for stable, bounded checkpoint uploads."""

    def fileno(self) -> int: ...

    def seek(self, offset: int, /) -> int: ...


class ChunkReader:
    """Adapt bounded HTTP chunks without accumulating a response-sized buffer."""

    def __init__(self, chunks: Iterator[bytes]):
        self.chunks = iter(chunks)
        self.pending = b""
        self.offset = 0

    def read(self, size: int) -> bytes:
        if size <= 0:
            return b""
        while self.offset == len(self.pending):
            try:
                self.pending = next(self.chunks)
            except StopIteration:
                return b""
            self.offset = 0
            if not self.pending:
                continue
            if len(self.pending) > CHUNK_BYTES:
                raise ValueError("HTTP checkpoint chunk exceeds the bounded reader capacity")
        end = min(len(self.pending), self.offset + size)
        result = self.pending[self.offset : end]
        self.offset = end
        return result


def _read_exact(reader: Reader, size: int) -> bytes:
    result = bytearray()
    while len(result) < size:
        data = reader.read(min(CHUNK_BYTES, size - len(result)))
        if not data or len(data) > size - len(result):
            raise ValueError("truncated native checkpoint transfer")
        result.extend(data)
    return bytes(result)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate checkpoint metadata field")
        result[key] = value
    return result


def _invalid_constant(_: str) -> Any:
    raise ValueError("nonfinite checkpoint metadata")


def _validate_metadata(metadata: Any, size: int, model: str) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be an object")
    if (
        metadata.get("version") != "attention_match_v1"
        or metadata.get("wire_version") not in (2, 3, 4)
        or metadata.get("model") != model
    ):
        raise ValueError("checkpoint metadata has an unexpected version or model")
    if metadata["wire_version"] == 4 and metadata.get("encoding") != "rotated4":
        raise ValueError("checkpoint metadata encoding disagrees with wire version")
    if metadata["wire_version"] == 3 and metadata.get("encoding") not in ("f16", "fp8"):
        raise ValueError("checkpoint metadata has an unsupported encoding")
    if metadata["wire_version"] == 2 and metadata.get("encoding", "f32") != "f32":
        raise ValueError("checkpoint metadata encoding disagrees with wire version")
    for name in ("wire_version", "logical_tokens", "checkpoint_bytes", "validated_heads", "preserved_heads"):
        if type(metadata.get(name)) is not int or metadata[name] < 0:
            raise ValueError(f"invalid checkpoint metadata field: {name}")
    fp8 = metadata.get("encoding") == "fp8"
    for name in ("fp8_heads", "fp8_fallback_heads"):
        if fp8 or name in metadata:
            if type(metadata.get(name)) is not int or metadata[name] < 0:
                raise ValueError(f"invalid checkpoint metadata for wire encoding: {name}")
    quantized = metadata.get("fp8_heads", 0)
    fallback = metadata.get("fp8_fallback_heads", 0)
    if (fp8 and quantized + fallback != metadata["validated_heads"]) or (not fp8 and (quantized or fallback)):
        raise ValueError("checkpoint head accounting disagrees with wire encoding")
    rotated = metadata.get("encoding") == "rotated4"
    rotated_fields = (
        "rotated4_heads",
        "rotated4_fallback_heads",
        "rotated4_eligible_rows",
        "rotated4_compressed_rows",
        "rotated4_exact_rows",
    )
    for name in rotated_fields:
        if rotated or name in metadata:
            if type(metadata.get(name)) is not int or metadata[name] < 0:
                raise ValueError(f"invalid checkpoint metadata for wire encoding: {name}")
    heads, fallback, eligible, compressed, exact = (metadata.get(name, 0) for name in rotated_fields)
    if rotated:
        if (
            heads + fallback != metadata["validated_heads"]
            or compressed + exact != eligible
            or bool(heads) != bool(compressed)
            or heads > compressed
            or fallback > exact
        ):
            raise ValueError("checkpoint row/head accounting disagrees with wire encoding")
    elif any((heads, fallback, eligible, compressed, exact)):
        raise ValueError("checkpoint row/head accounting disagrees with wire encoding")
    if metadata["checkpoint_bytes"] != size or not 0 < metadata["logical_tokens"] <= 32768:
        raise ValueError("checkpoint metadata has an invalid size or logical length")
    ratio = metadata.get("global_savings_ratio")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError("invalid checkpoint savings ratio")
    try:
        ratio = float(ratio)
    except OverflowError as exc:
        raise ValueError("invalid checkpoint savings ratio") from exc
    if not math.isfinite(ratio):
        raise ValueError("invalid checkpoint savings ratio")
    outcome = metadata.get("outcome")
    if outcome == "compacted":
        if ratio < 1.5 or metadata["validated_heads"] == 0:
            raise ValueError("checkpoint does not attest successful compaction")
    elif outcome == "unchanged_insufficient_savings":
        if ratio != 1 or metadata["validated_heads"] != 0:
            raise ValueError("invalid unchanged checkpoint outcome")
    else:
        raise ValueError("unsupported checkpoint outcome")
    return metadata


def _validate_v2_metadata(raw: bytes) -> tuple[dict[str, Any], str, dict[str, Any]]:
    metadata = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(metadata, dict) or set(metadata) not in (
        {"schema", "prepared_metadata_base64"},
        {"schema", "prepared_metadata_base64", "request_json"},
    ):
        raise ValueError("invalid AFATT002 metadata fields")
    if metadata.get("schema") != "antfly.native-kv-envelope.v2":
        raise ValueError("invalid AFATT002 metadata schema")
    prepared = metadata.get("prepared_metadata_base64")
    if not isinstance(prepared, str) or not prepared.isascii():
        raise ValueError("invalid AFATT002 producer metadata")
    try:
        decoded = base64.b64decode(prepared, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise ValueError("invalid AFATT002 producer metadata") from exc
    if (
        not decoded
        or len(decoded) > MAX_PREPARED_METADATA_BYTES
        or base64.b64encode(decoded).decode("ascii") != prepared
    ):
        raise ValueError("invalid AFATT002 producer metadata")
    producer = json.loads(decoded, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    if not isinstance(producer, dict) or producer.get("schema") != "antfly.native-kv-prepared-producer.v1":
        raise ValueError("invalid AFATT002 producer metadata")
    required = {
        "schema",
        "model",
        "model_artifact_sha256_hex",
        "model_tokenizer_sha256_hex",
        "model_template_sha256_hex",
        "model_layout_sha256_hex",
        "profile_sha256_hex",
        "profile_semantic_sha256_hex",
        "source_sha256_hex",
        "binary_sha256_hex",
        "policy_name",
        "policy_sha256_hex",
        "admission",
        "chunk_rows",
        "reject_exact_active_fixed_point",
        "encoding",
        "logical_revision",
        "logical_tokens",
        "checkpoint_bytes",
        "outcome",
    }
    optional = {"quality_evidence_core_sha256_hex"}
    fields = set(producer)
    if not required <= fields or not fields <= required | optional:
        raise ValueError("invalid AFATT002 producer fields")
    for name in (key for key in required if key.endswith("_sha256_hex")):
        value = producer.get(name)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("invalid AFATT002 producer digest")
    if "quality_evidence_core_sha256_hex" in producer:
        value = producer["quality_evidence_core_sha256_hex"]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or value == "0" * 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("invalid AFATT002 producer quality evidence digest")
    for name in ("logical_revision", "logical_tokens", "checkpoint_bytes", "chunk_rows"):
        if type(producer.get(name)) is not int or producer[name] < 0:
            raise ValueError("invalid AFATT002 producer integer")
    if type(producer.get("reject_exact_active_fixed_point")) is not bool:
        raise ValueError("invalid AFATT002 producer policy")
    for name in ("model", "policy_name", "admission", "encoding"):
        if not isinstance(producer.get(name), str) or not producer[name]:
            raise ValueError("invalid AFATT002 producer string")
    outcome = producer.get("outcome")
    if outcome not in ("compacted", "unchanged_insufficient_savings"):
        raise ValueError("invalid AFATT002 producer outcome")
    request_json = metadata.get("request_json")
    if request_json is not None:
        if not isinstance(request_json, str):
            raise ValueError("AFATT002 request_json must be a JSON string")
        json.loads(request_json, object_pairs_hook=_unique_object, parse_constant=_invalid_constant)
    return metadata, prepared, producer


def _copy_checkpoint(
    reader: Reader,
    size: int,
    output: BinaryIO | None = None,
    *,
    metadata: dict[str, Any] | None = None,
    envelope_hash: Any | None = None,
) -> tuple[str, dict[str, str], int, str]:
    if not 192 <= size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("native checkpoint exceeds file limits")
    header = _read_exact(reader, 152)
    magic, version, flags, payload_size = struct.unpack("<8sIIQ", header[:24])
    if (
        magic != b"ANTFLYKV"
        or version not in (2, 3, 4)
        or payload_size != size - 184
        or (version == 2 and flags != 0)
        or (version == 3 and flags not in (0, 1))
        or (version == 4 and flags != 0)
    ):
        raise ValueError("invalid native checkpoint header")
    if metadata is not None and (
        version != metadata["wire_version"]
        or (version == 3 and flags != int(metadata["encoding"] == "fp8"))
        or (version == 4 and metadata.get("encoding") != "rotated4")
    ):
        raise ValueError("checkpoint metadata disagrees with wire encoding")
    integrity = hashlib.sha256(header)
    file_hash = hashlib.sha256(header)
    if envelope_hash is not None:
        envelope_hash.update(header)
    if output is not None:
        output.write(header)
    remaining = size - 184
    logical_tokens: int | None = None
    while remaining:
        block = _read_exact(reader, min(CHUNK_BYTES, remaining))
        if logical_tokens is None:
            logical_tokens = struct.unpack("<Q", block[:8])[0]
        integrity.update(block)
        file_hash.update(block)
        if envelope_hash is not None:
            envelope_hash.update(block)
        if output is not None:
            output.write(block)
        remaining -= len(block)
    footer = _read_exact(reader, 32)
    if integrity.digest() != footer:
        raise ValueError("native checkpoint checksum mismatch")
    if output is not None:
        output.write(footer)
    file_hash.update(footer)
    if envelope_hash is not None:
        envelope_hash.update(footer)
    assert logical_tokens is not None
    if not 0 < logical_tokens <= 32768:
        raise ValueError("native checkpoint logical length is invalid")
    identity = {
        name: header[24 + i * 32 : 56 + i * 32].hex()
        for i, name in enumerate(("weights", "tokenizer", "template", "layout"))
    }
    encoding = {2: "f32", 4: "rotated4"}.get(version, "fp8" if flags == 1 else "f16")
    return file_hash.hexdigest(), identity, logical_tokens, encoding


def _validate_producer_checkpoint(
    producer: dict[str, Any], model: str, size: int, logical_tokens: int, encoding: str, identity: Mapping[str, str]
) -> None:
    if (
        producer["model"] != model
        or producer["checkpoint_bytes"] != size
        or producer["logical_tokens"] != logical_tokens
        or producer["encoding"] != encoding
        or producer["model_artifact_sha256_hex"] != identity["weights"]
        or producer["model_tokenizer_sha256_hex"] != identity["tokenizer"]
        or producer["model_template_sha256_hex"] != identity["template"]
        or producer["model_layout_sha256_hex"] != identity["layout"]
    ):
        raise ValueError("AFATT002 producer disagrees with checkpoint")


def receive_checkpoint(
    reader: Reader, destination: str | Path, model: str, *, content_type: str | None = None
) -> NativeKvCheckpoint:
    """Leave an existing destination intact on any failure before atomic publication."""
    raw_header = _read_exact(reader, 40)
    header = struct.unpack("<8sQIIIIQ", raw_header)
    magic, metadata_bytes, count, flags, mime_bytes, attachment_flags, size = header
    if magic not in (AFATT001, AFATT002) or (count, flags, mime_bytes, attachment_flags) != (
        1,
        0,
        len(CHECKPOINT_MIME),
        0,
    ):
        raise ValueError("invalid native checkpoint envelope")
    if content_type is not None and (content_type, magic) not in (
        (CONTENT_TYPE, AFATT001),
        (CONTENT_TYPE_V2, AFATT002),
    ):
        raise ValueError("native checkpoint media type and envelope version disagree")
    if not 0 < metadata_bytes <= MAX_METADATA_BYTES or not 192 <= size <= MAX_CHECKPOINT_BYTES:
        raise ValueError("native checkpoint transfer exceeds limits")
    raw_metadata = _read_exact(reader, metadata_bytes)
    if magic == AFATT001:
        metadata = _validate_metadata(
            json.loads(raw_metadata, object_pairs_hook=_unique_object, parse_constant=_invalid_constant), size, model
        )
        prepared = None
        producer_outcome = metadata["outcome"]
        envelope_hash = None
    else:
        metadata, prepared, producer = _validate_v2_metadata(raw_metadata)
        producer_outcome = producer["outcome"]
        envelope_hash = hashlib.sha256(raw_header + raw_metadata + CHECKPOINT_MIME)
    if _read_exact(reader, mime_bytes) != CHECKPOINT_MIME:
        raise ValueError("invalid checkpoint attachment MIME")
    destination = Path(destination)
    partial = destination.with_name(destination.name + ".part")
    created = False
    try:
        with partial.open("xb") as output:
            created = True
            digest, identity, logical_tokens, encoding = _copy_checkpoint(
                reader, size, output, metadata=metadata if magic == AFATT001 else None, envelope_hash=envelope_hash
            )
            if magic == AFATT001 and logical_tokens != metadata["logical_tokens"]:
                raise ValueError("checkpoint metadata disagrees with saved logical length")
            if magic == AFATT002:
                _validate_producer_checkpoint(producer, model, size, logical_tokens, encoding, identity)
            if envelope_hash is not None and _read_exact(reader, 32) != envelope_hash.digest():
                raise ValueError("AFATT002 envelope checksum mismatch")
            if reader.read(1):
                raise ValueError("trailing native checkpoint data")
            output.flush()
            os.fsync(output.fileno())
        os.replace(partial, destination)
    finally:
        if created:
            partial.unlink(missing_ok=True)
    return NativeKvCheckpoint(
        destination, metadata, digest, identity, 2 if magic == AFATT002 else 1, prepared, producer_outcome
    )


def checkpoint_upload(
    source: CheckpointSource, metadata: dict[str, Any], prepared_metadata_base64: str | None = None
) -> tuple[int, Iterator[bytes]]:
    """Validate a file before sending; stream the same descriptor without replay.

    Includes V4 rotated4 files; model compatibility is checked by the server.
    Payload/codec semantics are validated by the server before installation.
    """
    before = os.fstat(source.fileno())
    size = before.st_size
    _, identity, logical_tokens, encoding = _copy_checkpoint(source, size)
    if source.read(1):
        raise ValueError("trailing native checkpoint data")
    after = os.fstat(source.fileno())
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("checkpoint changed during validation")
    source.seek(0)
    request = json.dumps(metadata, separators=(",", ":"), sort_keys=True, allow_nan=False)
    if prepared_metadata_base64 is None:
        encoded = request.encode()
        magic = AFATT001
    else:
        _, _, producer = _validate_v2_metadata(
            json.dumps(
                {"schema": "antfly.native-kv-envelope.v2", "prepared_metadata_base64": prepared_metadata_base64},
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
        model = metadata.get("model")
        if not isinstance(model, str):
            raise ValueError("AFATT002 request model is required")
        _validate_producer_checkpoint(producer, model, size, logical_tokens, encoding, identity)
        encoded = json.dumps(
            {
                "prepared_metadata_base64": prepared_metadata_base64,
                "request_json": request,
                "schema": "antfly.native-kv-envelope.v2",
            },
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode()
        magic = AFATT002
    if not 0 < len(encoded) <= MAX_METADATA_BYTES:
        raise ValueError("native checkpoint metadata exceeds limits")
    prefix = struct.pack("<8sQIIIIQ", magic, len(encoded), 1, 0, len(CHECKPOINT_MIME), 0, size)
    prefix += encoded + CHECKPOINT_MIME

    def chunks() -> Iterator[bytes]:
        envelope_hash = hashlib.sha256(prefix) if magic == AFATT002 else None
        for offset in range(0, len(prefix), CHUNK_BYTES):
            yield prefix[offset : offset + CHUNK_BYTES]
        remaining = size
        while remaining:
            block = _read_exact(source, min(CHUNK_BYTES, remaining))
            if envelope_hash is not None:
                envelope_hash.update(block)
            yield block
            remaining -= len(block)
        end = os.fstat(source.fileno())
        if source.read(1) or (before.st_size, before.st_mtime_ns) != (end.st_size, end.st_mtime_ns):
            raise ValueError("checkpoint changed during upload")
        if envelope_hash is not None:
            yield envelope_hash.digest()

    return len(prefix) + size + (32 if magic == AFATT002 else 0), chunks()


def checkpoint_upload_content_type(prepared_metadata_base64: str | None) -> str:
    return CONTENT_TYPE_V2 if prepared_metadata_base64 is not None else CONTENT_TYPE
