use antfly_sdk::native_kv::{CHECKPOINT_MIME, CONTENT_TYPE_V2, receive_to_path, write_upload};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use sha2::{Digest, Sha256};
use std::io::Cursor;

#[derive(serde::Deserialize)]
struct SharedFixture {
    content_type: String,
    prepared_metadata_base64: String,
    wire_hex: String,
    wire_sha256: String,
}

fn hex_decode(value: &str) -> Vec<u8> {
    assert_eq!(value.len() % 2, 0);
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn checkpoint() -> Vec<u8> {
    let payload = [12u64.to_le_bytes().as_slice(), &[7u8; 32]].concat();
    let mut value = Vec::from(b"ANTFLYKV".as_slice());
    value.extend_from_slice(&2u32.to_le_bytes());
    value.extend_from_slice(&0u32.to_le_bytes());
    value.extend_from_slice(&(payload.len() as u64).to_le_bytes());
    value.extend(0u8..128);
    value.extend_from_slice(&payload);
    let digest = Sha256::digest(&value);
    value.extend_from_slice(&digest);
    value
}

fn frame(outcome: &str) -> (Vec<u8>, String) {
    let checkpoint = checkpoint();
    let ids: Vec<String> = (0..4)
        .map(|i| (i * 32..i * 32 + 32).map(|b| format!("{b:02x}")).collect())
        .collect();
    let producer = serde_json::json!({
        "schema":"antfly.native-kv-prepared-producer.v1","model":"gemma",
        "model_artifact_sha256_hex":ids[0],"model_tokenizer_sha256_hex":ids[1],
        "model_template_sha256_hex":ids[2],"model_layout_sha256_hex":ids[3],
        "profile_sha256_hex":"1".repeat(64),"profile_semantic_sha256_hex":"2".repeat(64),
        "source_sha256_hex":"3".repeat(64),"binary_sha256_hex":"4".repeat(64),
        "policy_name":"cumulative256_exact_cycle_v1","policy_sha256_hex":"5".repeat(64),
        "admission":"cumulative_ladder_4_3_2","chunk_rows":256,"reject_exact_active_fixed_point":true,
        "encoding":"f32","logical_revision":7,"logical_tokens":12,"checkpoint_bytes":checkpoint.len(),
        "outcome":outcome
    });
    let prepared = BASE64.encode(serde_json::to_vec(&producer).unwrap());
    let metadata = format!(
        "{{\"prepared_metadata_base64\":\"{prepared}\",\"schema\":\"antfly.native-kv-envelope.v2\"}}"
    );
    let mut header = [0; 40];
    header[..8].copy_from_slice(b"AFATT002");
    header[8..16].copy_from_slice(&(metadata.len() as u64).to_le_bytes());
    header[16..20].copy_from_slice(&1u32.to_le_bytes());
    header[24..28].copy_from_slice(&(CHECKPOINT_MIME.len() as u32).to_le_bytes());
    header[32..40].copy_from_slice(&(checkpoint.len() as u64).to_le_bytes());
    let mut result = [
        header.as_slice(),
        metadata.as_bytes(),
        CHECKPOINT_MIME,
        &checkpoint,
    ]
    .concat();
    result.extend_from_slice(&Sha256::digest(&result));
    (result, prepared)
}

fn legacy_frame() -> Vec<u8> {
    let checkpoint = checkpoint();
    let metadata = serde_json::to_vec(&serde_json::json!({
        "version":"attention_match_v1","wire_version":2,"encoding":"f32","model":"gemma",
        "outcome":"compacted","logical_tokens":12,"checkpoint_bytes":checkpoint.len(),
        "global_savings_ratio":2.0,"validated_heads":1,"preserved_heads":0
    }))
    .unwrap();
    let mut header = [0; 40];
    header[..8].copy_from_slice(b"AFATT001");
    header[8..16].copy_from_slice(&(metadata.len() as u64).to_le_bytes());
    header[16..20].copy_from_slice(&1u32.to_le_bytes());
    header[24..28].copy_from_slice(&(CHECKPOINT_MIME.len() as u32).to_le_bytes());
    header[32..40].copy_from_slice(&(checkpoint.len() as u64).to_le_bytes());
    [
        header.as_slice(),
        metadata.as_slice(),
        CHECKPOINT_MIME,
        checkpoint.as_slice(),
    ]
    .concat()
}

#[test]
fn v2_roundtrip_preserves_producer_and_rehashes_request() {
    let (wire, prepared) = frame("compacted");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("saved.kv");
    let saved = receive_to_path(&mut Cursor::new(&wire), &path, CONTENT_TYPE_V2, "gemma").unwrap();
    assert_eq!(saved.producer_outcome, "compacted");
    assert_eq!(
        saved.prepared_metadata_base64.as_deref(),
        Some(prepared.as_str())
    );
    let mut upload = Vec::new();
    let (length, media) = write_upload(
        &mut Cursor::new(std::fs::read(path).unwrap()),
        &mut upload,
        br#"{"model":"gemma"}"#,
        Some(&prepared),
    )
    .unwrap();
    assert_eq!(media, CONTENT_TYPE_V2);
    assert_eq!(length as usize, upload.len());
    assert_eq!(&upload[..8], b"AFATT002");
    assert_eq!(
        &upload[upload.len() - 32..],
        Sha256::digest(&upload[..upload.len() - 32]).as_slice()
    );
    assert!(
        write_upload(
            &mut Cursor::new(std::fs::read(&saved.path).unwrap()),
            &mut Vec::new(),
            br#"{"model":"other"}"#,
            Some(&prepared)
        )
        .is_err()
    );
}

#[test]
fn v2_truncation_mutation_and_media_mismatch_never_replace() {
    let (wire, _) = frame("compacted");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("saved.kv");
    std::fs::write(&path, b"old").unwrap();
    for cut in [1, 31, 32] {
        assert!(
            receive_to_path(
                &mut Cursor::new(&wire[..wire.len() - cut]),
                &path,
                CONTENT_TYPE_V2,
                "gemma"
            )
            .is_err()
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"old");
    }
    let mut corrupt = wire.clone();
    *corrupt.last_mut().unwrap() ^= 1;
    assert!(receive_to_path(&mut Cursor::new(corrupt), &path, CONTENT_TYPE_V2, "gemma").is_err());
    for offset in [41, wire.len() - 33] {
        let mut corrupt = wire.clone();
        corrupt[offset] ^= 1;
        assert!(
            receive_to_path(&mut Cursor::new(corrupt), &path, CONTENT_TYPE_V2, "gemma").is_err()
        );
        assert_eq!(std::fs::read(&path).unwrap(), b"old");
    }
    assert!(
        receive_to_path(
            &mut Cursor::new(wire),
            &path,
            "application/vnd.antfly.attachments.v1",
            "gemma"
        )
        .is_err()
    );
    assert_eq!(std::fs::read(path).unwrap(), b"old");
}

#[test]
fn preexisting_partial_is_never_removed_or_replaced() {
    let (wire, _) = frame("compacted");
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("saved.kv");
    let partial = dir.path().join("saved.kv.part");
    std::fs::write(&path, b"destination").unwrap();
    std::fs::write(&partial, b"other-owner").unwrap();
    assert!(receive_to_path(&mut Cursor::new(wire), &path, CONTENT_TYPE_V2, "gemma").is_err());
    assert_eq!(std::fs::read(path).unwrap(), b"destination");
    assert_eq!(std::fs::read(partial).unwrap(), b"other-owner");
}

#[test]
fn legacy_metadata_is_checked_against_expected_model_and_checkpoint() {
    let wire = legacy_frame();
    let dir = tempfile::tempdir().unwrap();
    let saved = receive_to_path(
        &mut Cursor::new(&wire),
        &dir.path().join("legacy.kv"),
        "application/vnd.antfly.attachments.v1",
        "gemma",
    )
    .unwrap();
    assert_eq!(saved.envelope_version, 1);
    assert_eq!(
        saved.identity[0],
        (0..32).map(|b| format!("{b:02x}")).collect::<String>()
    );
    assert!(
        receive_to_path(
            &mut Cursor::new(wire),
            &dir.path().join("wrong.kv"),
            "application/vnd.antfly.attachments.v1",
            "other"
        )
        .is_err()
    );
}

#[test]
fn unchanged_outcome_is_preserved() {
    let (wire, _) = frame("unchanged_insufficient_savings");
    let dir = tempfile::tempdir().unwrap();
    let saved = receive_to_path(
        &mut Cursor::new(wire),
        &dir.path().join("unchanged.kv"),
        CONTENT_TYPE_V2,
        "gemma",
    )
    .unwrap();
    assert_eq!(saved.producer_outcome, "unchanged_insufficient_savings");
}

#[test]
fn cross_sdk_fixture_is_exact() {
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../specs/testdata/native_kv_afatt002_v1.json");
    let fixture: SharedFixture =
        serde_json::from_slice(&std::fs::read(fixture_path).unwrap()).unwrap();
    let wire = hex_decode(&fixture.wire_hex);
    assert_eq!(format!("{:x}", Sha256::digest(&wire)), fixture.wire_sha256);
    let dir = tempfile::tempdir().unwrap();
    let saved = receive_to_path(
        &mut Cursor::new(wire),
        &dir.path().join("shared.kv"),
        &fixture.content_type,
        "gemma",
    )
    .unwrap();
    assert_eq!(
        saved.prepared_metadata_base64.as_deref(),
        Some(fixture.prepared_metadata_base64.as_str())
    );
    assert_eq!(saved.producer_outcome, "compacted");
}
