// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Bounded AFATT001/AFATT002 native-KV transport. Generated API files do not
//! own this wire contract. Publish only the path returned after verification.

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use serde::Deserialize;
use serde::de::{IgnoredAny, MapAccess, Visitor};
use sha2::{Digest, Sha256};
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const CONTENT_TYPE_V1: &str = "application/vnd.antfly.attachments.v1";
pub const CONTENT_TYPE_V2: &str = "application/vnd.antfly.attachments.v2";
pub const CHECKPOINT_MIME: &[u8] = b"application/vnd.antfly.kv-checkpoint";
pub const MAX_METADATA_BYTES: u64 = 2 * 1024 * 1024;
pub const MAX_PREPARED_METADATA_BYTES: usize = 16 * 1024;
pub const MAX_CHECKPOINT_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const CHUNK: usize = 64 * 1024;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    Invalid(&'static str),
}
impl From<std::io::Error> for Error {
    fn from(v: std::io::Error) -> Self {
        Self::Io(v)
    }
}
impl From<serde_json::Error> for Error {
    fn from(v: serde_json::Error) -> Self {
        Self::Json(v)
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => e.fmt(f),
            Self::Json(e) => e.fmt(f),
            Self::Invalid(s) => f.write_str(s),
        }
    }
}
impl std::error::Error for Error {}

#[derive(Debug, Clone)]
pub struct PortableCheckpoint {
    pub path: PathBuf,
    pub envelope_version: u8,
    pub metadata_json: Vec<u8>,
    pub prepared_metadata_base64: Option<String>,
    pub producer_outcome: String,
    pub checkpoint_sha256: [u8; 32],
    pub identity: [String; 4],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OuterV2 {
    schema: String,
    prepared_metadata_base64: String,
    request_json: Option<String>,
}

struct ProducerSummary {
    schema: String,
    model: String,
    model_artifact_sha256_hex: String,
    model_tokenizer_sha256_hex: String,
    model_template_sha256_hex: String,
    model_layout_sha256_hex: String,
    encoding: String,
    logical_tokens: u64,
    checkpoint_bytes: u64,
    outcome: String,
}

impl<'de> Deserialize<'de> for ProducerSummary {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct ProducerVisitor;
        impl<'de> Visitor<'de> for ProducerVisitor {
            type Value = ProducerSummary;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("a prepared Producer object with unique fields")
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<Self::Value, M::Error> {
                let mut names = std::collections::HashSet::new();
                let mut schema = None;
                let mut model = None;
                let mut model_artifact_sha256_hex = None;
                let mut model_tokenizer_sha256_hex = None;
                let mut model_template_sha256_hex = None;
                let mut model_layout_sha256_hex = None;
                let mut encoding = None;
                let mut logical_tokens = None;
                let mut checkpoint_bytes = None;
                let mut outcome = None;
                while let Some(name) = map.next_key::<String>()? {
                    if !names.insert(name.clone()) {
                        return Err(serde::de::Error::custom("duplicate Producer field"));
                    }
                    match name.as_str() {
                        "schema" => schema = Some(map.next_value::<String>()?),
                        "model" => model = Some(map.next_value::<String>()?),
                        "model_artifact_sha256_hex" => {
                            model_artifact_sha256_hex = Some(map.next_value::<String>()?)
                        }
                        "model_tokenizer_sha256_hex" => {
                            model_tokenizer_sha256_hex = Some(map.next_value::<String>()?)
                        }
                        "model_template_sha256_hex" => {
                            model_template_sha256_hex = Some(map.next_value::<String>()?)
                        }
                        "model_layout_sha256_hex" => {
                            model_layout_sha256_hex = Some(map.next_value::<String>()?)
                        }
                        "encoding" => encoding = Some(map.next_value::<String>()?),
                        "logical_tokens" => logical_tokens = Some(map.next_value::<u64>()?),
                        "checkpoint_bytes" => checkpoint_bytes = Some(map.next_value::<u64>()?),
                        "outcome" => outcome = Some(map.next_value::<String>()?),
                        "profile_sha256_hex"
                        | "profile_semantic_sha256_hex"
                        | "source_sha256_hex"
                        | "binary_sha256_hex"
                        | "policy_sha256_hex" => {
                            let value = map.next_value::<String>()?;
                            if !valid_digest(&value) {
                                return Err(serde::de::Error::custom("invalid Producer digest"));
                            }
                        }
                        "policy_name" | "admission" => {
                            if map.next_value::<String>()?.is_empty() {
                                return Err(serde::de::Error::custom("empty Producer string"));
                            }
                        }
                        "chunk_rows" | "logical_revision" => {
                            map.next_value::<u64>()?;
                        }
                        "reject_exact_active_fixed_point" => {
                            map.next_value::<bool>()?;
                        }
                        _ => {
                            map.next_value::<IgnoredAny>()?;
                        }
                    }
                }
                const EXPECTED: [&str; 20] = [
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
                ];
                if names.len() != EXPECTED.len()
                    || EXPECTED.iter().any(|name| !names.contains(*name))
                {
                    return Err(serde::de::Error::custom("invalid Producer fields"));
                }
                Ok(ProducerSummary {
                    schema: schema.ok_or_else(|| serde::de::Error::missing_field("schema"))?,
                    model: model.ok_or_else(|| serde::de::Error::missing_field("model"))?,
                    model_artifact_sha256_hex: model_artifact_sha256_hex.ok_or_else(|| {
                        serde::de::Error::missing_field("model_artifact_sha256_hex")
                    })?,
                    model_tokenizer_sha256_hex: model_tokenizer_sha256_hex.ok_or_else(|| {
                        serde::de::Error::missing_field("model_tokenizer_sha256_hex")
                    })?,
                    model_template_sha256_hex: model_template_sha256_hex.ok_or_else(|| {
                        serde::de::Error::missing_field("model_template_sha256_hex")
                    })?,
                    model_layout_sha256_hex: model_layout_sha256_hex.ok_or_else(|| {
                        serde::de::Error::missing_field("model_layout_sha256_hex")
                    })?,
                    encoding: encoding
                        .ok_or_else(|| serde::de::Error::missing_field("encoding"))?,
                    logical_tokens: logical_tokens
                        .ok_or_else(|| serde::de::Error::missing_field("logical_tokens"))?,
                    checkpoint_bytes: checkpoint_bytes
                        .ok_or_else(|| serde::de::Error::missing_field("checkpoint_bytes"))?,
                    outcome: outcome.ok_or_else(|| serde::de::Error::missing_field("outcome"))?,
                })
            }
        }
        deserializer.deserialize_map(ProducerVisitor)
    }
}

fn u32le(b: &[u8]) -> u32 {
    u32::from_le_bytes(b.try_into().unwrap())
}
fn u64le(b: &[u8]) -> u64 {
    u64::from_le_bytes(b.try_into().unwrap())
}
fn put32(b: &mut [u8], v: u32) {
    b.copy_from_slice(&v.to_le_bytes())
}
fn put64(b: &mut [u8], v: u64) {
    b.copy_from_slice(&v.to_le_bytes())
}

fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

fn validate_prepared(prepared: &str) -> Result<ProducerSummary, Error> {
    let decoded = BASE64
        .decode(prepared.as_bytes())
        .map_err(|_| Error::Invalid("producer base64"))?;
    if decoded.is_empty()
        || decoded.len() > MAX_PREPARED_METADATA_BYTES
        || BASE64.encode(&decoded) != prepared
    {
        return Err(Error::Invalid("producer metadata"));
    }
    let producer: ProducerSummary = serde_json::from_slice(&decoded)?;
    if producer.schema != "antfly.native-kv-prepared-producer.v1" {
        return Err(Error::Invalid("producer schema"));
    }
    if !matches!(
        producer.outcome.as_str(),
        "compacted" | "unchanged_insufficient_savings"
    ) {
        return Err(Error::Invalid("producer outcome"));
    }
    if producer.model.is_empty()
        || ![
            &producer.model_artifact_sha256_hex,
            &producer.model_tokenizer_sha256_hex,
            &producer.model_template_sha256_hex,
            &producer.model_layout_sha256_hex,
        ]
        .into_iter()
        .all(|v| valid_digest(v))
    {
        return Err(Error::Invalid("producer identity"));
    }
    Ok(producer)
}

fn validate_v2(raw: &[u8]) -> Result<(String, String), Error> {
    let outer: OuterV2 = serde_json::from_slice(raw)?;
    if outer.schema != "antfly.native-kv-envelope.v2" {
        return Err(Error::Invalid("AFATT002 schema"));
    }
    let producer = validate_prepared(&outer.prepared_metadata_base64)?;
    if let Some(request) = outer.request_json {
        let _: serde_json::Value = serde_json::from_str(&request)?;
    }
    Ok((outer.prepared_metadata_base64, producer.outcome))
}

fn copy_n<R: Read, W: Write>(
    r: &mut R,
    w: &mut W,
    mut n: u64,
    envelope: &mut Option<Sha256>,
    file: &mut Sha256,
) -> Result<(), Error> {
    let mut buf = [0; CHUNK];
    while n != 0 {
        let take = n.min(CHUNK as u64) as usize;
        r.read_exact(&mut buf[..take])?;
        w.write_all(&buf[..take])?;
        file.update(&buf[..take]);
        if let Some(h) = envelope {
            h.update(&buf[..take]);
        }
        n -= take as u64;
    }
    Ok(())
}

struct CheckpointFacts {
    digest: [u8; 32],
    logical_tokens: u64,
    encoding: &'static str,
    identity: [[u8; 32]; 4],
}

fn validate_match(
    producer: &ProducerSummary,
    expected_model: &str,
    size: u64,
    facts: &CheckpointFacts,
) -> Result<(), Error> {
    let ids = [
        &producer.model_artifact_sha256_hex,
        &producer.model_tokenizer_sha256_hex,
        &producer.model_template_sha256_hex,
        &producer.model_layout_sha256_hex,
    ];
    if producer.model != expected_model
        || producer.checkpoint_bytes != size
        || producer.logical_tokens != facts.logical_tokens
        || producer.encoding != facts.encoding
        || ids
            .into_iter()
            .zip(facts.identity)
            .any(|(a, b)| a != &hex(&b))
    {
        return Err(Error::Invalid("producer/checkpoint mismatch"));
    }
    Ok(())
}

fn copy_checkpoint<R: Read, W: Write>(
    r: &mut R,
    w: &mut W,
    size: u64,
    envelope: &mut Option<Sha256>,
) -> Result<CheckpointFacts, Error> {
    if !(192..=MAX_CHECKPOINT_BYTES).contains(&size) {
        return Err(Error::Invalid("checkpoint size"));
    }
    let mut header = [0; 152];
    r.read_exact(&mut header)?;
    w.write_all(&header)?;
    let version = u32le(&header[8..12]);
    let flags = u32le(&header[12..16]);
    if &header[..8] != b"ANTFLYKV"
        || !(2..=4).contains(&version)
        || u64le(&header[16..24]) != size - 184
        || (version == 2 && flags != 0)
        || (version == 3 && flags > 1)
        || (version == 4 && flags != 0)
    {
        return Err(Error::Invalid("checkpoint header"));
    }
    let mut hash = Sha256::new();
    hash.update(header);
    if let Some(h) = envelope {
        h.update(header);
    }
    let mut first = [0; 8];
    r.read_exact(&mut first)?;
    w.write_all(&first)?;
    hash.update(first);
    if let Some(h) = envelope {
        h.update(first);
    }
    let logical_tokens = u64le(&first);
    if !(1..=32768).contains(&logical_tokens) {
        return Err(Error::Invalid("logical tokens"));
    }
    copy_n(r, w, size - 192, envelope, &mut hash)?;
    let mut footer = [0; 32];
    r.read_exact(&mut footer)?;
    if hash.clone().finalize().as_slice() != footer {
        return Err(Error::Invalid("checkpoint checksum"));
    }
    w.write_all(&footer)?;
    hash.update(footer);
    if let Some(h) = envelope {
        h.update(footer);
    }
    let identity = std::array::from_fn(|i| header[24 + i * 32..56 + i * 32].try_into().unwrap());
    let encoding = match (version, flags) {
        (2, 0) => "f32",
        (3, 0) => "f16",
        (3, 1) => "fp8",
        (4, 0) => "rotated4",
        _ => return Err(Error::Invalid("checkpoint encoding")),
    };
    Ok(CheckpointFacts {
        digest: hash.finalize().into(),
        logical_tokens,
        encoding,
        identity,
    })
}

pub fn receive_to_path<R: Read>(
    r: &mut R,
    destination: &Path,
    content_type: &str,
    expected_model: &str,
) -> Result<PortableCheckpoint, Error> {
    let mut header = [0; 40];
    r.read_exact(&mut header)?;
    let version = match (&header[..8], content_type) {
        (b"AFATT001", CONTENT_TYPE_V1) => 1,
        (b"AFATT002", CONTENT_TYPE_V2) => 2,
        _ => return Err(Error::Invalid("media type/version mismatch")),
    };
    let metadata_len = u64le(&header[8..16]);
    let mime_len = u32le(&header[24..28]) as usize;
    let checkpoint_len = u64le(&header[32..40]);
    if metadata_len == 0
        || metadata_len > MAX_METADATA_BYTES
        || u32le(&header[16..20]) != 1
        || u32le(&header[20..24]) != 0
        || mime_len != CHECKPOINT_MIME.len()
        || u32le(&header[28..32]) != 0
        || !(192..=MAX_CHECKPOINT_BYTES).contains(&checkpoint_len)
    {
        return Err(Error::Invalid("envelope header"));
    }
    let mut metadata = vec![0; metadata_len as usize];
    r.read_exact(&mut metadata)?;
    let legacy = if version == 1 {
        Some(serde_json::from_slice::<serde_json::Value>(&metadata)?)
    } else {
        None
    };
    let (prepared, outcome, producer) = if version == 2 {
        let (p, o) = validate_v2(&metadata)?;
        let producer = validate_prepared(&p)?;
        (Some(p), o, Some(producer))
    } else {
        let v = legacy.as_ref().unwrap();
        (
            None,
            v.get("outcome")
                .and_then(|x| x.as_str())
                .ok_or(Error::Invalid("legacy outcome"))?
                .to_owned(),
            None,
        )
    };
    let mut mime = vec![0; mime_len];
    r.read_exact(&mut mime)?;
    if mime != CHECKPOINT_MIME {
        return Err(Error::Invalid("MIME"));
    }
    let mut envelope = (version == 2).then(|| {
        let mut h = Sha256::new();
        h.update(header);
        h.update(&metadata);
        h.update(&mime);
        h
    });
    let partial = destination.with_extension(format!(
        "{}part",
        destination
            .extension()
            .and_then(|x| x.to_str())
            .map(|x| format!("{x}."))
            .unwrap_or_default()
    ));
    let mut created = false;
    let result = (|| {
        let mut out = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&partial)?;
        created = true;
        let facts = copy_checkpoint(r, &mut out, checkpoint_len, &mut envelope)?;
        if let Some(p) = &producer {
            validate_match(p, expected_model, checkpoint_len, &facts)?;
        }
        if let Some(v) = &legacy {
            let model = v
                .get("model")
                .and_then(|x| x.as_str())
                .ok_or(Error::Invalid("legacy model"))?;
            let bytes = v
                .get("checkpoint_bytes")
                .and_then(|x| x.as_u64())
                .ok_or(Error::Invalid("legacy bytes"))?;
            let tokens = v
                .get("logical_tokens")
                .and_then(|x| x.as_u64())
                .ok_or(Error::Invalid("legacy tokens"))?;
            let encoding = v.get("encoding").and_then(|x| x.as_str()).unwrap_or("f32");
            if model != expected_model
                || bytes != checkpoint_len
                || tokens != facts.logical_tokens
                || encoding != facts.encoding
            {
                return Err(Error::Invalid("legacy metadata/checkpoint mismatch"));
            }
        }
        if let Some(hash) = envelope.take() {
            let mut footer = [0; 32];
            r.read_exact(&mut footer)?;
            if hash.finalize().as_slice() != footer {
                return Err(Error::Invalid("AFATT002 checksum"));
            }
        }
        let mut extra = [0; 1];
        if r.read(&mut extra)? != 0 {
            return Err(Error::Invalid("trailing data"));
        }
        out.sync_all()?;
        std::fs::rename(&partial, destination)?;
        let identity = facts.identity.map(|value| hex(&value));
        Ok(PortableCheckpoint {
            path: destination.to_owned(),
            envelope_version: version,
            metadata_json: metadata,
            prepared_metadata_base64: prepared,
            producer_outcome: outcome,
            checkpoint_sha256: facts.digest,
            identity,
        })
    })();
    if result.is_err() && created {
        let _ = std::fs::remove_file(&partial);
    }
    result
}

fn v2_outer(prepared: &str, request: &str) -> Result<Vec<u8>, Error> {
    validate_prepared(prepared)?;
    let _: serde_json::Value = serde_json::from_str(request)?;
    Ok(format!("{{\"prepared_metadata_base64\":{},\"request_json\":{},\"schema\":\"antfly.native-kv-envelope.v2\"}}", serde_json::to_string(prepared)?, serde_json::to_string(request)?).into_bytes())
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub fn write_upload<R: Read + Seek, W: Write>(
    source: &mut R,
    sink: &mut W,
    request_json: &[u8],
    prepared: Option<&str>,
) -> Result<(u64, &'static str), Error> {
    let size = source.seek(SeekFrom::End(0))?;
    source.seek(SeekFrom::Start(0))?;
    let mut none = None;
    let facts = copy_checkpoint(source, &mut std::io::sink(), size, &mut none)?;
    let mut extra = [0; 1];
    if source.read(&mut extra)? != 0 {
        return Err(Error::Invalid("trailing checkpoint"));
    }
    source.seek(SeekFrom::Start(0))?;
    let request = std::str::from_utf8(request_json).map_err(|_| Error::Invalid("request UTF-8"))?;
    let request_value: serde_json::Value = serde_json::from_slice(request_json)?;
    let (magic, metadata, content_type) = if let Some(p) = prepared {
        let producer = validate_prepared(p)?;
        let model = request_value
            .get("model")
            .and_then(|v| v.as_str())
            .ok_or(Error::Invalid("request model"))?;
        validate_match(&producer, model, size, &facts)?;
        (
            b"AFATT002".as_slice(),
            v2_outer(p, request)?,
            CONTENT_TYPE_V2,
        )
    } else {
        (
            b"AFATT001".as_slice(),
            request_json.to_vec(),
            CONTENT_TYPE_V1,
        )
    };
    if metadata.is_empty() || metadata.len() as u64 > MAX_METADATA_BYTES {
        return Err(Error::Invalid("metadata size"));
    }
    let mut header = [0; 40];
    header[..8].copy_from_slice(magic);
    put64(&mut header[8..16], metadata.len() as u64);
    put32(&mut header[16..20], 1);
    put32(&mut header[24..28], CHECKPOINT_MIME.len() as u32);
    put64(&mut header[32..40], size);
    sink.write_all(&header)?;
    sink.write_all(&metadata)?;
    sink.write_all(CHECKPOINT_MIME)?;
    let mut envelope = prepared.map(|_| {
        let mut h = Sha256::new();
        h.update(header);
        h.update(&metadata);
        h.update(CHECKPOINT_MIME);
        h
    });
    let mut ignored = Sha256::new();
    copy_n(source, sink, size, &mut envelope, &mut ignored)?;
    if let Some(hash) = envelope {
        sink.write_all(&hash.finalize())?;
    }
    Ok((
        40 + metadata.len() as u64
            + CHECKPOINT_MIME.len() as u64
            + size
            + if prepared.is_some() { 32 } else { 0 },
        content_type,
    ))
}
