"""Portable checkpoint streaming, publication, and cancellation contracts."""

import base64
import hashlib
import io
import json
import struct
from pathlib import Path

import httpx
import pytest

from antfly import AntflyClient, AntflyException, NativeKvCheckpoint
from antfly.client_generated.models import InferenceGenerateRequest
from antfly.client_generated.models.inference_generate_usage import InferenceGenerateUsage
from antfly.native_kv import (
    CHECKPOINT_MIME,
    CHUNK_BYTES,
    CONTENT_TYPE,
    CONTENT_TYPE_V2,
    MAX_CHECKPOINT_BYTES,
    ChunkReader,
    checkpoint_upload,
    receive_checkpoint,
    _validate_v2_metadata,
)


def fixture(payload_size=40, outcome="compacted", version=2, encoding="f32"):
    payload = struct.pack("<Q", 12) + b"x" * payload_size
    header = struct.pack("<8sIIQ", b"ANTFLYKV", version, int(encoding == "fp8"), len(payload)) + bytes(range(128))
    checkpoint = header + payload + hashlib.sha256(header + payload).digest()
    metadata = {
        "version": "attention_match_v1",
        "wire_version": version,
        "encoding": encoding,
        "model": "gemma",
        "outcome": outcome,
        "logical_tokens": 12,
        "checkpoint_bytes": len(checkpoint),
        "global_savings_ratio": 2 if outcome == "compacted" else 1,
        "validated_heads": 2 if outcome == "compacted" else 0,
        "preserved_heads": 6,
    }
    if encoding == "fp8":
        metadata.update(
            fp8_heads=1 if outcome == "compacted" else 0, fp8_fallback_heads=1 if outcome == "compacted" else 0
        )
    if encoding == "rotated4":
        metadata.update(
            rotated4_heads=1 if outcome == "compacted" else 0,
            rotated4_fallback_heads=1 if outcome == "compacted" else 0,
            rotated4_eligible_rows=8 if outcome == "compacted" else 0,
            rotated4_compressed_rows=5 if outcome == "compacted" else 0,
            rotated4_exact_rows=3 if outcome == "compacted" else 0,
        )
    encoded = json.dumps(metadata).encode()
    frame = struct.pack("<8sQIIIIQ", b"AFATT001", len(encoded), 1, 0, len(CHECKPOINT_MIME), 0, len(checkpoint))
    return frame + encoded + CHECKPOINT_MIME + checkpoint, checkpoint, metadata


def fixture_v2(payload_size=40, outcome="compacted", quality_evidence_core=None):
    _, checkpoint, _ = fixture(payload_size)
    identity = [bytes(range(start, start + 32)).hex() for start in range(0, 128, 32)]
    producer_value = {
            "schema": "antfly.native-kv-prepared-producer.v1",
            "model": "gemma",
            "model_artifact_sha256_hex": identity[0],
            "model_tokenizer_sha256_hex": identity[1],
            "model_template_sha256_hex": identity[2],
            "model_layout_sha256_hex": identity[3],
            "profile_sha256_hex": "1" * 64,
            "profile_semantic_sha256_hex": "2" * 64,
            "source_sha256_hex": "3" * 64,
            "binary_sha256_hex": "4" * 64,
            "policy_name": "cumulative256_exact_cycle_v1",
            "policy_sha256_hex": "5" * 64,
            "admission": "cumulative_ladder_4_3_2",
            "chunk_rows": 256,
            "reject_exact_active_fixed_point": True,
            "encoding": "f32",
            "logical_revision": 7,
            "logical_tokens": 12,
            "checkpoint_bytes": len(checkpoint),
            "outcome": outcome,
        }
    if quality_evidence_core is not None:
        producer_value["quality_evidence_core_sha256_hex"] = quality_evidence_core
    producer = json.dumps(
        producer_value,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    prepared = base64.b64encode(producer).decode("ascii")
    outer = {
        "prepared_metadata_base64": prepared,
        "schema": "antfly.native-kv-envelope.v2",
    }
    encoded = json.dumps(outer, separators=(",", ":"), sort_keys=True).encode()
    header = struct.pack("<8sQIIIIQ", b"AFATT002", len(encoded), 1, 0, len(CHECKPOINT_MIME), 0, len(checkpoint))
    authenticated = header + encoded + CHECKPOINT_MIME + checkpoint
    return authenticated + hashlib.sha256(authenticated).digest(), checkpoint, outer, producer


def test_v2_producer_accepts_only_authenticated_optional_quality_core():
    for quality in (None, "a" * 64):
        _, _, outer, _ = fixture_v2(quality_evidence_core=quality)
        _, _, producer = _validate_v2_metadata(json.dumps(outer).encode())
        assert producer.get("quality_evidence_core_sha256_hex") == quality
    for invalid in (None, "", "0" * 64, "A" * 64, "g" * 64, "a" * 63):
        _, _, outer, producer = fixture_v2()
        value = json.loads(producer)
        value["quality_evidence_core_sha256_hex"] = invalid
        outer["prepared_metadata_base64"] = base64.b64encode(json.dumps(value).encode()).decode()
        with pytest.raises(ValueError, match="quality evidence digest"):
            _validate_v2_metadata(json.dumps(outer).encode())
    _, _, outer, producer = fixture_v2()
    value = json.loads(producer)
    value["unknown_optional"] = "a" * 64
    outer["prepared_metadata_base64"] = base64.b64encode(json.dumps(value).encode()).decode()
    with pytest.raises(ValueError, match="producer fields"):
        _validate_v2_metadata(json.dumps(outer).encode())
    duplicate = producer[:-1] + b',"model":"duplicate"}'
    outer["prepared_metadata_base64"] = base64.b64encode(duplicate).decode()
    with pytest.raises(ValueError, match="duplicate checkpoint metadata field"):
        _validate_v2_metadata(json.dumps(outer).encode())


class Stream(httpx.SyncByteStream):
    def __init__(self, data, *, fail=False):
        self.data = data
        self.fail = fail
        self.closed = False

    def __iter__(self):
        for i in range(0, len(self.data), 137):
            yield self.data[i : i + 137]
        if self.fail:
            raise httpx.ReadError("connection lost")

    def close(self):
        self.closed = True


def client_with_transport(handler):
    client = AntflyClient("http://test", token="secret")
    headers = client._client.get_httpx_client().headers
    client._client.set_httpx_client(
        httpx.Client(base_url="http://test", headers=headers, transport=httpx.MockTransport(handler))
    )
    return client


@pytest.mark.parametrize("outcome", ["compacted", "unchanged_insufficient_savings"])
def test_export_streams_and_publishes_verified_checkpoint(tmp_path, outcome):
    frame, checkpoint, metadata = fixture(CHUNK_BYTES * 3, outcome)
    destination = tmp_path / "saved.kv"
    destination.write_bytes(b"previous")
    stream = Stream(frame)

    def handle(request):
        assert request.headers["Authorization"] == "Bearer secret"
        assert [value.strip() for value in request.headers["Accept"].split(",")] == [CONTENT_TYPE_V2, CONTENT_TYPE]
        assert json.loads(request.read())["stream"] is False
        assert destination.read_bytes() == b"previous"
        return httpx.Response(200, headers={"Content-Type": CONTENT_TYPE}, stream=stream)

    client = client_with_transport(handle)
    result = client.prepare_native_kv_checkpoint(InferenceGenerateRequest(model="gemma", messages=[]), destination)
    assert isinstance(result, NativeKvCheckpoint)
    assert result.metadata == metadata
    assert destination.read_bytes() == checkpoint
    assert result.sha256 == hashlib.sha256(checkpoint).hexdigest()
    assert result.identity["weights"] == bytes(range(32)).hex()
    assert stream.closed
    assert not destination.with_suffix(".kv.part").exists()


def test_every_truncation_and_bad_integrity_preserve_existing_file(tmp_path):
    frame, _, _ = fixture()
    destination = tmp_path / "saved.kv"
    destination.write_bytes(b"previous")
    for n in range(len(frame)):
        with pytest.raises(ValueError):
            receive_checkpoint(io.BytesIO(frame[:n]), destination, "gemma")
        assert destination.read_bytes() == b"previous"
        assert not Path(str(destination) + ".part").exists()
    for offset in (0, 16, 20, 24, 28, frame.index(CHECKPOINT_MIME), len(frame) - 33, len(frame) - 1):
        corrupt = bytearray(frame)
        corrupt[offset] ^= 1
        with pytest.raises(ValueError):
            receive_checkpoint(io.BytesIO(corrupt), destination, "gemma")
        assert destination.read_bytes() == b"previous"
    with pytest.raises(ValueError, match="trailing"):
        receive_checkpoint(io.BytesIO(frame + b"extra"), destination, "gemma")


def test_afatt002_footer_is_verified_before_publish_and_opaque_producer_is_forwarded(tmp_path):
    frame, checkpoint, outer, producer = fixture_v2(CHUNK_BYTES + 7)
    destination = tmp_path / "saved-v2.kv"
    destination.write_bytes(b"previous")
    saved = receive_checkpoint(io.BytesIO(frame), destination, "gemma")
    assert destination.read_bytes() == checkpoint
    assert saved.envelope_version == 2
    assert saved.prepared_metadata_base64 == outer["prepared_metadata_base64"]
    assert saved.producer_outcome == "compacted"
    assert base64.b64decode(saved.prepared_metadata_base64) == producer

    request = {"model": "gemma", "messages": [{"role": "user", "content": "next"}]}
    with destination.open("rb") as source:
        length, chunks = checkpoint_upload(source, request, saved.prepared_metadata_base64)
        wire = b"".join(chunks)
    assert len(wire) == length
    assert wire[:8] == b"AFATT002"
    metadata_len = struct.unpack_from("<Q", wire, 8)[0]
    upload_outer = json.loads(wire[40 : 40 + metadata_len])
    assert upload_outer["prepared_metadata_base64"] == outer["prepared_metadata_base64"]
    assert json.loads(upload_outer["request_json"]) == request
    assert wire[-32:] == hashlib.sha256(wire[:-32]).digest()

    for cut in (1, 31, 32, len(frame) - 1):
        destination.write_bytes(b"previous")
        with pytest.raises(ValueError):
            receive_checkpoint(io.BytesIO(frame[:-cut]), destination, "gemma")
        assert destination.read_bytes() == b"previous"
    corrupt = bytearray(frame)
    corrupt[-1] ^= 1
    with pytest.raises(ValueError, match="envelope checksum"):
        receive_checkpoint(io.BytesIO(corrupt), destination, "gemma")
    assert destination.read_bytes() == b"previous"

    for offset in (41, len(frame) - 33):
        corrupt = bytearray(frame)
        corrupt[offset] ^= 1
        with pytest.raises(ValueError):
            receive_checkpoint(io.BytesIO(corrupt), destination, "gemma")
        assert destination.read_bytes() == b"previous"


def test_public_client_negotiates_v2_and_resume_preserves_opaque_producer(tmp_path):
    frame, checkpoint, outer, _ = fixture_v2(CHUNK_BYTES + 7)
    destination = tmp_path / "client-v2.kv"
    requests = []

    def handler(request):
        requests.append(request)
        if request.url.path.endswith("/checkpoint"):
            assert [value.strip() for value in request.headers["Accept"].split(",")] == [
                CONTENT_TYPE_V2,
                CONTENT_TYPE,
            ]
            return httpx.Response(200, headers={"Content-Type": CONTENT_TYPE_V2}, stream=Stream(frame))
        assert request.url.path.endswith("/resume")
        assert request.headers["Content-Type"] == CONTENT_TYPE_V2
        wire = request.read()
        assert wire[:8] == b"AFATT002"
        metadata_len = struct.unpack_from("<Q", wire, 8)[0]
        upload_outer = json.loads(wire[40 : 40 + metadata_len])
        assert upload_outer["prepared_metadata_base64"] == outer["prepared_metadata_base64"]
        assert json.loads(upload_outer["request_json"])["model"] == "gemma"
        assert wire[-32:] == hashlib.sha256(wire[:-32]).digest()
        return httpx.Response(200, headers={"Content-Type": "application/json"}, json={})

    client = client_with_transport(handler)
    request = InferenceGenerateRequest(model="gemma", messages=[])
    saved = client.prepare_native_kv_checkpoint(request, destination)
    assert destination.read_bytes() == checkpoint
    assert saved.prepared_metadata_base64 == outer["prepared_metadata_base64"]
    with client._native_kv_resume_response(request, saved, stream=False):
        pass
    assert len(requests) == 2


def test_afatt002_preserves_unchanged_outcome(tmp_path):
    frame, _, _, _ = fixture_v2(outcome="unchanged_insufficient_savings")
    saved = receive_checkpoint(io.BytesIO(frame), tmp_path / "unchanged.kv", "gemma")
    assert saved.producer_outcome == "unchanged_insufficient_savings"


@pytest.mark.parametrize(
    "field,value",
    [
        ("model", "other"),
        ("logical_tokens", 11),
        ("checkpoint_bytes", 1),
        ("encoding", "f16"),
        ("model_layout_sha256_hex", "0" * 64),
    ],
)
def test_afatt002_rejects_producer_checkpoint_mismatch(tmp_path, field, value):
    frame, checkpoint, outer, _ = fixture_v2()
    producer = json.loads(base64.b64decode(outer["prepared_metadata_base64"]))
    producer[field] = value
    outer["prepared_metadata_base64"] = base64.b64encode(
        json.dumps(producer, separators=(",", ":"), sort_keys=True).encode()
    ).decode()
    encoded = json.dumps(outer, separators=(",", ":"), sort_keys=True).encode()
    header = struct.pack("<8sQIIIIQ", b"AFATT002", len(encoded), 1, 0, len(CHECKPOINT_MIME), 0, len(checkpoint))
    authenticated = header + encoded + CHECKPOINT_MIME + checkpoint
    wire = authenticated + hashlib.sha256(authenticated).digest()
    with pytest.raises(ValueError, match="disagrees"):
        receive_checkpoint(io.BytesIO(wire), tmp_path / "mismatch.kv", "gemma", content_type=CONTENT_TYPE_V2)


def test_afatt002_upload_rechecks_model_and_checkpoint(tmp_path):
    frame, _, _, _ = fixture_v2()
    saved = receive_checkpoint(io.BytesIO(frame), tmp_path / "saved.kv", "gemma", content_type=CONTENT_TYPE_V2)
    with saved.path.open("rb") as source, pytest.raises(ValueError, match="disagrees"):
        checkpoint_upload(source, {"model": "other"}, saved.prepared_metadata_base64)


def test_afatt001_remains_unchanged_and_afatt002_rejects_noncanonical_or_object_request(tmp_path):
    legacy, _, _ = fixture()
    assert receive_checkpoint(io.BytesIO(legacy), tmp_path / "legacy.kv", "gemma").envelope_version == 1
    frame, _, outer, _ = fixture_v2()
    metadata_len = struct.unpack_from("<Q", frame, 8)[0]
    metadata = json.loads(frame[40 : 40 + metadata_len])
    metadata["request_json"] = {"model": "gemma"}
    encoded = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()
    bad_header = struct.pack(
        "<8sQIIIIQ",
        b"AFATT002",
        len(encoded),
        1,
        0,
        len(CHECKPOINT_MIME),
        0,
        len(frame) - 40 - metadata_len - len(CHECKPOINT_MIME) - 32,
    )
    with pytest.raises(ValueError, match="JSON string"):
        receive_checkpoint(io.BytesIO(bad_header + encoded + frame[40 + metadata_len :]), tmp_path / "bad.kv", "gemma")

    raw = base64.b64decode(outer["prepared_metadata_base64"])
    metadata = dict(outer, prepared_metadata_base64=base64.b64encode(raw).decode().rstrip("="))
    encoded = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode()
    bad_header = struct.pack(
        "<8sQIIIIQ",
        b"AFATT002",
        len(encoded),
        1,
        0,
        len(CHECKPOINT_MIME),
        0,
        len(frame) - 40 - metadata_len - len(CHECKPOINT_MIME) - 32,
    )
    with pytest.raises(ValueError, match="producer metadata"):
        receive_checkpoint(
            io.BytesIO(bad_header + encoded + frame[40 + metadata_len :]), tmp_path / "bad-base64.kv", "gemma"
        )

    with pytest.raises(ValueError, match="media type"):
        receive_checkpoint(io.BytesIO(frame), tmp_path / "bad-media.kv", "gemma", content_type=CONTENT_TYPE)
    with pytest.raises(ValueError, match="media type"):
        receive_checkpoint(io.BytesIO(legacy), tmp_path / "bad-legacy-media.kv", "gemma", content_type=CONTENT_TYPE_V2)


def test_cross_sdk_afatt002_fixture_is_exact(tmp_path):
    fixture_path = Path(__file__).parents[4] / "specs" / "testdata" / "native_kv_afatt002_v1.json"
    shared = json.loads(fixture_path.read_text())
    wire = bytes.fromhex(shared["wire_hex"])
    assert hashlib.sha256(wire).hexdigest() == shared["wire_sha256"]
    saved = receive_checkpoint(io.BytesIO(wire), tmp_path / "shared.kv", "gemma", content_type=shared["content_type"])
    assert saved.prepared_metadata_base64 == shared["prepared_metadata_base64"]
    assert saved.producer_outcome == "compacted"


def test_failed_export_closes_response_and_keeps_previous_checkpoint(tmp_path):
    frame, _, _ = fixture()
    destination = tmp_path / "saved.kv"
    destination.write_bytes(b"previous")
    stream = Stream(frame, fail=True)
    client = client_with_transport(lambda _: httpx.Response(200, headers={"Content-Type": CONTENT_TYPE}, stream=stream))
    with pytest.raises(httpx.ReadError):
        client.prepare_native_kv_checkpoint(InferenceGenerateRequest(model="gemma", messages=[]), destination)
    assert stream.closed
    assert destination.read_bytes() == b"previous"
    assert not Path(str(destination) + ".part").exists()


def test_staging_ownership_limits_model_and_metadata_are_checked(tmp_path):
    frame, _, _ = fixture()
    destination = tmp_path / "saved.kv"
    partial = Path(str(destination) + ".part")
    partial.write_bytes(b"another writer")
    with pytest.raises(FileExistsError):
        receive_checkpoint(io.BytesIO(frame), destination, "gemma")
    assert partial.read_bytes() == b"another writer"
    partial.unlink()
    with pytest.raises(ValueError, match="model"):
        receive_checkpoint(io.BytesIO(frame), destination, "different-model")
    oversized = bytearray(frame[:40])
    struct.pack_into("<Q", oversized, 32, MAX_CHECKPOINT_BYTES + 1)
    with pytest.raises(ValueError, match="limits"):
        receive_checkpoint(io.BytesIO(oversized), destination, "gemma")
    mismatch = frame.replace(b'"logical_tokens": 12', b'"logical_tokens": 13')
    with pytest.raises(ValueError, match="logical length"):
        receive_checkpoint(io.BytesIO(mismatch), destination, "gemma")
    assert not destination.exists()
    assert not partial.exists()


def test_disk_failure_never_replaces_previous_checkpoint(tmp_path, monkeypatch):
    from antfly import native_kv

    frame, _, _ = fixture()
    destination = tmp_path / "saved.kv"
    destination.write_bytes(b"previous")

    def fail(_):
        raise OSError("disk failure")

    monkeypatch.setattr(native_kv.os, "fsync", fail)
    with pytest.raises(OSError):
        receive_checkpoint(io.BytesIO(frame), destination, "gemma")
    assert destination.read_bytes() == b"previous"
    assert not Path(str(destination) + ".part").exists()


def test_upload_validation_and_chunks_are_bounded(tmp_path):
    _, checkpoint, _ = fixture(CHUNK_BYTES * 3)
    path = tmp_path / "saved.kv"
    path.write_bytes(checkpoint)

    class BoundedFile:
        def __init__(self, file):
            self.file = file

        def read(self, n):
            assert 0 < n <= CHUNK_BYTES
            return self.file.read(n)

        def seek(self, offset):
            return self.file.seek(offset)

        def fileno(self):
            return self.file.fileno()

    with path.open("rb") as file:
        length, chunks = checkpoint_upload(BoundedFile(file), {"model": "gemma", "messages": []})
        blocks = list(chunks)
    assert all(len(block) <= CHUNK_BYTES for block in blocks)
    wire = b"".join(blocks)
    assert length == len(wire)
    assert wire.endswith(checkpoint)
    assert wire[:8] == b"AFATT001"
    path.write_bytes(checkpoint[:-1] + bytes([checkpoint[-1] ^ 1]))
    with path.open("rb") as source, pytest.raises(ValueError, match="checksum"):
        checkpoint_upload(source, {})


def test_resume_json_and_early_closed_sse_use_bounded_upload(tmp_path):
    _, checkpoint, _ = fixture(CHUNK_BYTES * 2)
    path = tmp_path / "saved.kv"
    path.write_bytes(checkpoint)
    requests = []
    stream = Stream(
        b'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"gemma",'
        b'"choices":[{"index":0,"delta":{"content":"ok"}}]}\n\ndata: [DONE]\n\n'
    )

    def handle(request):
        assert request.headers["Authorization"] == "Bearer secret"
        assert request.headers["Content-Type"] == CONTENT_TYPE
        wire = request.read()
        assert int(request.headers["Content-Length"]) == len(wire)
        assert wire.endswith(checkpoint)
        metadata_length = struct.unpack_from("<Q", wire, 8)[0]
        metadata = json.loads(wire[40 : 40 + metadata_length])
        requests.append(metadata)
        if metadata["stream"]:
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, stream=stream)
        return httpx.Response(
            200,
            json={
                "id": "c",
                "object": "chat.completion",
                "created": 1,
                "model": "gemma",
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13, "cached_prompt_tokens": 12},
            },
        )

    client = client_with_transport(handle)
    request = InferenceGenerateRequest(model="gemma", messages=[])
    response = client.resume_native_kv_checkpoint(request, path)
    assert isinstance(response.usage, InferenceGenerateUsage)
    assert response.usage.cached_prompt_tokens == 12
    with client.resume_native_kv_checkpoint_stream(request, path) as chunks:
        assert next(chunks).choices[0].delta.content == "ok"
    assert stream.closed
    assert [item["stream"] for item in requests] == [False, True]
    assert path.read_bytes() == checkpoint


def test_resume_rejects_invalid_file_before_any_request(tmp_path):
    path = tmp_path / "saved.kv"
    path.write_bytes(b"broken")
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(500)

    client = client_with_transport(handle)
    with pytest.raises(AntflyException, match="file limits"):
        client.resume_native_kv_checkpoint(InferenceGenerateRequest(model="gemma", messages=[]), path)
    assert not calls


def test_chunk_reader_rejects_unbounded_source_chunk():
    reader = ChunkReader(iter([b"x" * (CHUNK_BYTES + 1)]))
    with pytest.raises(ValueError, match="bounded"):
        reader.read(1)


def test_chunk_reader_skips_empty_chunks_and_retains_partial_reads():
    reader = ChunkReader(iter([b"", b"abc", b"", b"d", b""]))
    assert reader.read(2) == b"ab"
    assert reader.read(2) == b"c"
    assert reader.read(2) == b"d"
    assert reader.read(2) == b""


@pytest.mark.parametrize("encoding", ["f16", "fp8"])
def test_v3_encoding_streams_and_checks_metadata(tmp_path, encoding):
    frame, checkpoint, metadata = fixture(version=3, encoding=encoding)
    destination = tmp_path / "v3.kv"
    saved = receive_checkpoint(io.BytesIO(frame), destination, "gemma")
    assert saved.metadata["encoding"] == encoding
    assert destination.read_bytes() == checkpoint
    with destination.open("rb") as source:
        length, chunks = checkpoint_upload(source, {"model": "gemma"})
        uploaded = b"".join(chunks)
    assert len(uploaded) == length
    assert uploaded.endswith(checkpoint)
    wrong = frame.replace(b'"wire_version": 3', b'"wire_version": 2', 1)
    with pytest.raises(ValueError):
        receive_checkpoint(io.BytesIO(wrong), destination, "gemma")
    assert destination.read_bytes() == checkpoint
    wrong = frame.replace(
        b'"encoding": "' + encoding.encode() + b'"',
        b'"encoding": "' + (b"fp8" if encoding == "f16" else b"f16") + b'"',
        1,
    )
    with pytest.raises(ValueError, match="wire encoding"):
        receive_checkpoint(io.BytesIO(wrong), destination, "gemma")


@pytest.mark.parametrize("encoding", ["f32", "f16", "fp8", "rotated4"])
def test_checkpoint_encoding_option_reaches_transport(tmp_path, encoding):
    frame, _, _ = fixture(version={"f32": 2, "f16": 3, "fp8": 3, "rotated4": 4}[encoding], encoding=encoding)

    def handler(request):
        assert request.headers["X-Antfly-KV-Encoding"] == encoding
        return httpx.Response(200, headers={"Content-Type": CONTENT_TYPE}, stream=Stream(frame))

    client = client_with_transport(handler)
    saved = client.prepare_native_kv_checkpoint(
        InferenceGenerateRequest(model="gemma", messages=[]), tmp_path / "saved.kv", encoding=encoding
    )
    assert saved.metadata["encoding"] == encoding
    with pytest.raises(ValueError, match="encoding"):
        client.prepare_native_kv_checkpoint(
            InferenceGenerateRequest(model="gemma", messages=[]), tmp_path / "saved.kv", encoding="unknown"
        )


@pytest.mark.parametrize(
    "field,value",
    [("fp8_heads", -1), ("fp8_heads", True), ("fp8_heads", 3), ("fp8_fallback_heads", None), ("fp8_fallback_heads", 0)],
)
def test_invalid_fp8_accounting_does_not_replace_saved_file(tmp_path, field, value):
    frame, checkpoint, metadata = fixture(version=3, encoding="fp8")
    destination = tmp_path / "saved.kv"
    receive_checkpoint(io.BytesIO(frame), destination, "gemma")
    metadata[field] = value
    encoded = json.dumps(metadata).encode()
    header = struct.pack("<8sQIIIIQ", b"AFATT001", len(encoded), 1, 0, len(CHECKPOINT_MIME), 0, len(checkpoint))
    with pytest.raises(ValueError, match="wire encoding"):
        receive_checkpoint(io.BytesIO(header + encoded + CHECKPOINT_MIME + checkpoint), destination, "gemma")
    assert destination.read_bytes() == checkpoint


def test_diagnostic_rotated4_checkpoint_upload_preserves_bytes(tmp_path):
    _, checkpoint, _ = fixture(CHUNK_BYTES * 2, version=4, encoding="rotated4")
    path = tmp_path / "rotated4.kv"
    path.write_bytes(checkpoint)
    request = {"model": "gemma", "messages": [], "stream": False}
    with path.open("rb") as source:
        length, chunks = checkpoint_upload(source, request)
        uploaded = b"".join(chunks)
    assert len(uploaded) == length
    assert uploaded.endswith(checkpoint)
    assert path.read_bytes() == checkpoint


@pytest.mark.parametrize("flags", [1, 2, 0xFFFFFFFF])
def test_rotated4_upload_rejects_unknown_flags_before_sending(tmp_path, flags):
    _, checkpoint, _ = fixture(version=4, encoding="rotated4")
    corrupted = bytearray(checkpoint)
    struct.pack_into("<I", corrupted, 12, flags)
    corrupted[-32:] = hashlib.sha256(corrupted[:-32]).digest()
    path = tmp_path / "rotated4.kv"
    path.write_bytes(corrupted)
    with path.open("rb") as source, pytest.raises(ValueError, match="header"):
        checkpoint_upload(source, {"model": "gemma"})


@pytest.mark.parametrize("outcome", ["compacted", "unchanged_insufficient_savings"])
def test_rotated4_export_publishes_verified_metadata(tmp_path, outcome):
    frame, checkpoint, metadata = fixture(version=4, encoding="rotated4", outcome=outcome)
    destination = tmp_path / "saved.kv"
    saved = receive_checkpoint(io.BytesIO(frame), destination, "gemma")
    assert saved.metadata == metadata
    assert destination.read_bytes() == checkpoint


@pytest.mark.parametrize(
    "field,value",
    [
        ("rotated4_heads", -1),
        ("rotated4_heads", True),
        ("rotated4_heads", 3),
        ("rotated4_fallback_heads", None),
        ("rotated4_fallback_heads", 0),
        ("rotated4_eligible_rows", 7),
        ("rotated4_compressed_rows", 0),
        ("rotated4_exact_rows", 4),
        ("rotated4_compressed_rows", False),
        ("fp8_heads", 1),
    ],
)
def test_rotated4_bad_accounting_preserves_existing_checkpoint(tmp_path, field, value):
    frame, checkpoint, metadata = fixture(version=4, encoding="rotated4")
    destination = tmp_path / "saved.kv"
    receive_checkpoint(io.BytesIO(frame), destination, "gemma")
    metadata[field] = value
    encoded = json.dumps(metadata).encode()
    header = struct.pack("<8sQIIIIQ", b"AFATT001", len(encoded), 1, 0, len(CHECKPOINT_MIME), 0, len(checkpoint))
    with pytest.raises(ValueError, match="wire encoding"):
        receive_checkpoint(io.BytesIO(header + encoded + CHECKPOINT_MIME + checkpoint), destination, "gemma")
    assert destination.read_bytes() == checkpoint


def test_f16_metadata_cannot_claim_rotated_rows(tmp_path):
    _, checkpoint, metadata = fixture(version=3, encoding="f16")
    metadata.update(
        rotated4_heads=1,
        rotated4_fallback_heads=1,
        rotated4_eligible_rows=8,
        rotated4_compressed_rows=5,
        rotated4_exact_rows=3,
    )
    encoded = json.dumps(metadata).encode()
    header = struct.pack("<8sQIIIIQ", b"AFATT001", len(encoded), 1, 0, len(CHECKPOINT_MIME), 0, len(checkpoint))
    with pytest.raises(ValueError, match="wire encoding"):
        receive_checkpoint(io.BytesIO(header + encoded + CHECKPOINT_MIME + checkpoint), tmp_path / "saved.kv", "gemma")


@pytest.mark.parametrize("heads,fallback,compressed,exact", [(2, 0, 1, 7), (1, 1, 8, 0)])
def test_rotated4_each_reported_head_requires_eligible_rows(tmp_path, heads, fallback, compressed, exact):
    _, checkpoint, metadata = fixture(version=4, encoding="rotated4")
    metadata.update(
        rotated4_heads=heads,
        rotated4_fallback_heads=fallback,
        rotated4_compressed_rows=compressed,
        rotated4_exact_rows=exact,
    )
    encoded = json.dumps(metadata).encode()
    header = struct.pack("<8sQIIIIQ", b"AFATT001", len(encoded), 1, 0, len(CHECKPOINT_MIME), 0, len(checkpoint))
    with pytest.raises(ValueError, match="wire encoding"):
        receive_checkpoint(io.BytesIO(header + encoded + CHECKPOINT_MIME + checkpoint), tmp_path / "saved.kv", "gemma")
