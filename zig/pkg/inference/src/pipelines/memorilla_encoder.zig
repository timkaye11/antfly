// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Managed encoder-to-memory bridge. Authentication/source closure belongs to
//! its caller; producer metadata is an identity check, never an access grant.
const std = @import("std");
const manager = @import("../server/model_manager.zig");
const factory = @import("../architectures/session_factory.zig");
const qualified = @import("../models/memorilla_qualification.zig");
const catalog = @import("../registry/memorilla_catalog.zig");
const token_features = @import("memorilla_token_features.zig");
const session_mod = @import("../backends/session.zig");
const Tensor = @import("../backends/backends.zig").Tensor;
const c_file = @import("../util/c_file.zig");
const Control = @import("../execution_control.zig").InferenceExecutionControl;
const memory = @import("../runtime/root.zig").tier.memory;
const A = std.mem.Allocator;
pub const width = qualified.width;
/// Request text ceiling shared by every document and the question (bytes).
pub const max_text_bytes: usize = 8192;

pub const v1_profile: qualified.Profile = .qwen3_encoder_f32_v1;
pub const research_profile: qualified.Profile = .qwen3_encoder_lora_research_f32_v1;
/// Research candidates ship their lock next to the weights (<= 1 MiB).
pub const research_lock_file = "memorilla_lock.json";
pub const research_lock_max_bytes: usize = 1024 * 1024;

pub const Producer = struct {
    /// The actual qualification tag that produced these features.
    profile: qualified.Profile,
    publication: [32]u8,
    identity: [64]u8,
    /// Lowercase sha256 hex of the profile's source lock: v1 pins the embedded
    /// `memorilla_v1_base_models.json`; the research tag pins the candidate
    /// directory's `memorilla_lock.json` bytes.
    source_lock_sha256: [64]u8,
    /// Lowercase sha256 hex of the canonical manifest for `profile`.
    recipe_sha256: [64]u8,
};

/// sha256 hex of the embedded v1 base-model lock.
pub fn v1SourceLockSha256() [64]u8 {
    var source: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(qualified.lock_bytes, &source, .{});
    return std.fmt.bytesToHex(source, .lower);
}

/// sha256 hex of `<model_dir>/memorilla_lock.json`. The loaded model exposes no
/// parsed lock, so the research producer pin is computed from its directory.
pub fn researchSourceLockSha256(a: A, model_dir: []const u8) ![64]u8 {
    const path = try std.fs.path.join(a, &.{ model_dir, research_lock_file });
    defer a.free(path);
    const bytes = c_file.readFileMax(a, path, research_lock_max_bytes) catch |err| switch (err) {
        error.FileNotFound => return error.MemorillaLockNotFound,
        else => return err,
    };
    defer a.free(bytes);
    if (bytes.len == 0) return error.InvalidMemorillaLock;
    var source: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &source, .{});
    return std.fmt.bytesToHex(source, .lower);
}

/// Canonical manifest digest for an encoder profile.
pub fn recipeSha256(profile: qualified.Profile) ![64]u8 {
    const source: []const u8 = switch (profile) {
        v1_profile => catalog.encoder_manifest_sha256,
        research_profile => catalog.research_encoder_manifest_sha256,
        else => return error.IncompatibleMemorillaProducer,
    };
    if (source.len != 64) return error.IncompatibleMemorillaProducer;
    var out: [64]u8 = undefined;
    @memcpy(&out, source);
    return out;
}

fn isLowerHex(text: []const u8) bool {
    for (text) |c| if (!((c >= '0' and c <= '9') or (c >= 'a' and c <= 'f'))) return false;
    return true;
}

/// Stored-vector identity checks do not replace caller authorization or source
/// version closure. The caller supplies the exact expected publication pin and
/// the profile it is serving. The v1 lock is embedded and verified here; the
/// research lock lives in the candidate directory, so this only requires a
/// well-formed digest and `requireProducerSourceLock` pins the exact value.
pub fn requireProducer(producer: Producer, expected_publication: [32]u8, expected: qualified.Profile) !void {
    if (producer.profile != expected or
        !std.mem.eql(u8, &producer.publication, &expected_publication) or
        !std.mem.eql(u8, &producer.identity, &qualified.executionIdentity(expected)) or
        !std.mem.eql(u8, &producer.recipe_sha256, &(try recipeSha256(expected))))
        return error.IncompatibleMemorillaProducer;
    if (expected == v1_profile) {
        if (!std.mem.eql(u8, &producer.source_lock_sha256, &v1SourceLockSha256())) return error.IncompatibleMemorillaProducer;
    } else if (!isLowerHex(&producer.source_lock_sha256)) return error.IncompatibleMemorillaProducer;
}

/// Pin a research producer to one candidate directory's lock digest
/// (`researchSourceLockSha256`).
pub fn requireProducerSourceLock(producer: Producer, expected_source_lock_sha256: [64]u8) !void {
    if (!std.mem.eql(u8, &producer.source_lock_sha256, &expected_source_lock_sha256)) return error.IncompatibleMemorillaProducer;
}

pub const Features = struct {
    allocator: A,
    document_count: usize,
    question_tokens: usize,
    documents_fp16: []f16,
    question_fp16: []f16,
    /// Exact BF16-valued features widened to F32 for the memory compute API.
    documents: []f32,
    question: []f32,
    question_text: []u8,
    document_sha256: [][32]u8,
    producer: Producer,
    lease: ?memory.AdmissionLease,

    pub fn deinit(self: *Features) void {
        self.allocator.free(self.documents_fp16);
        self.allocator.free(self.question_fp16);
        self.allocator.free(self.documents);
        self.allocator.free(self.question);
        self.allocator.free(self.question_text);
        self.allocator.free(self.document_sha256);
        if (self.lease) |*lease| lease.release();
        self.* = undefined;
    }
};

pub const canonicalQuestion = qualified.canonicalQuestion;

pub const StoredValue = struct { fp16: f16, memory: f32 };
pub fn storedValue(value: f32) !StoredValue {
    if (!std.math.isFinite(value)) return error.NonFiniteMemorillaFeature;
    const half: f16 = @floatCast(value);
    if (!std.math.isFinite(half)) return error.NonFiniteMemorillaFeature;
    const widened: f32 = @floatCast(half);
    const bits: u32 = @bitCast(widened);
    const rounded = bits +% (0x7fff + ((bits >> 16) & 1));
    const bf16: u32 = (rounded >> 16) << 16;
    return .{ .fp16 = half, .memory = @bitCast(bf16) };
}

fn copyFeature(source: []const f32, stored: []f16, destination: []f32) !void {
    if (source.len != width or stored.len != width or destination.len != width) return error.InvalidMemorillaFeatureWidth;
    var norm: f64 = 0;
    for (source) |value| {
        if (!std.math.isFinite(value)) return error.NonFiniteMemorillaFeature;
        norm += @as(f64, value) * value;
    }
    if (@abs(@sqrt(norm) - 1) > 1e-4) return error.UnnormalizedMemorillaFeature;
    for (source, stored, destination) |value, *half, *memory_value| {
        const converted = try storedValue(value);
        half.* = converted.fp16;
        memory_value.* = converted.memory;
    }
}
pub const convertFeature = copyFeature;

const TokenBudget = struct {
    total: usize = 0,
    question_text: []const u8,
    question_tokens: usize = 0,
    fn observe(raw: *anyopaque, text: []const u8, ids: []const i32, mask: []const i32) anyerror!void {
        const self: *@This() = @ptrCast(@alignCast(raw));
        if (ids.len != mask.len) return error.InvalidMemorillaTokens;
        var active: usize = 0;
        var last: i32 = -1;
        for (ids, mask) |id, used| {
            if (used != 0 and used != 1) return error.InvalidMemorillaTokens;
            if (used == 1) {
                active += 1;
                last = id;
            }
        }
        if (active == 0 or active > qualified.max_text_tokens or last != 151643) return error.MemorillaTextLimitExceeded;
        self.total = try std.math.add(usize, self.total, active);
        if (self.total > 16384) return error.MemorillaTextLimitExceeded;
        if (std.mem.eql(u8, text, self.question_text)) {
            if (self.question_tokens != 0 and self.question_tokens != active) return error.InvalidMemorillaTokens;
            self.question_tokens = active;
        }
    }
};

/// The caller retains its handle and authorization closure. This call adds an
/// independent lifetime pin and keeps the complete output admitted until deinit.
pub fn encode(a: A, handle: *const manager.ModelHandle, documents: []const []const u8, question: []const u8, limits: memory.Limits, control: Control) !Features {
    return encodeImpl(a, handle, documents, question, limits, control, false);
}

/// Query execution over supplied stored features encodes only the question.
/// Its empty documents arrays are intentional and never encode an empty corpus.
pub fn encodeQuestion(a: A, handle: *const manager.ModelHandle, question: []const u8, limits: memory.Limits, control: Control) !Features {
    return encodeImpl(a, handle, &.{}, question, limits, control, true);
}

fn encodeImpl(a: A, handle: *const manager.ModelHandle, documents: []const []const u8, question: []const u8, limits: memory.Limits, control: Control, question_only: bool) !Features {
    try control.check();
    if (documents.len == 0 and !question_only) return error.EmptyMemorillaCollection;
    if (documents.len > qualified.max_documents or question.len > 8192) return error.MemorillaTextLimitExceeded;
    const canonical = try canonicalQuestion(question);
    if (canonical.len == 0) return error.EmptyMemorillaQuestion;
    for (documents) |text| {
        if (text.len > 8192) return error.MemorillaTextLimitExceeded;
        _ = try std.unicode.Utf8View.init(text);
    }
    const backing = @import("antfly_platform").allocator.reclaimingScratch(a);
    var pin = handle.duplicate();
    defer pin.release();
    const model = pin.get();
    try qualified.requireRecipe(v1_profile, model.manifest);
    try factory.requireMemorillaProfile(model.session, v1_profile);
    const publication = model.manifest.managed_publication orelse return error.MemorillaManagedProducerRequired;
    const elements = try std.math.mul(usize, documents.len + 1, width);
    const owned_bytes = try std.math.add(usize, try std.math.mul(usize, elements, 10), canonical.len + documents.len * 32 + (documents.len + 1) * @sizeOf([]const u8));
    var lease = try pin.manager.acquireRunResourceAmounts(.cpu, limits, .{ .host_scratch_bytes = owned_bytes });
    var owns_lease = true;
    defer if (owns_lease) lease.release();
    try control.check();
    var assets = try model.acquireEmbeddingAssetLeaseWithControl(false, control);
    defer assets.release();
    var pipeline = blk: {
        try model.lockEmbeddingAssetsWithControl(control);
        defer model.unlockEmbeddingAssets();
        try model.ensurePrimaryEmbeddingAssetsLockedWithControl(true, false, control);
        break :blk model.embeddingPipelineLocked(backing);
    };
    try model.verifyEmbeddingIdentityWithControl(control);
    const identity = model.embedding_identity orelse return error.InvalidEmbeddingIdentity;
    pipeline.execution_control = control;
    pipeline.execution_lane = &model.embedding_execution_lane;
    const lane = try model.embedding_execution_lane.acquire(control);
    defer lane.release();
    // This owner explicitly admits all retained full-width F32 output rows.
    // Keep bounded physical batches without the Gemma-only retained-row floor.
    pipeline.execution_lane_held = true;
    pipeline.config.max_length = qualified.max_text_tokens;
    pipeline.config.batch_size = 4;
    pipeline.config.strict_max_length = true;
    pipeline.config.reject_truncation = true;
    var budget = TokenBudget{ .question_text = canonical };
    pipeline.tokenization_observer = .{ .ptr = &budget, .observe = TokenBudget.observe };
    const texts = try backing.alloc([]const u8, documents.len + 1);
    defer backing.free(texts);
    @memcpy(texts[0..documents.len], documents);
    texts[documents.len] = canonical;
    const vectors = try pipeline.embed(texts);
    defer {
        for (vectors) |vector| backing.free(vector);
        backing.free(vectors);
    }
    try control.check();
    try model.verifyEmbeddingIdentityWithControl(control);
    try factory.requireMemorillaProfile(model.session, v1_profile);
    if (vectors.len != texts.len) return error.InvalidMemorillaFeatureWidth;
    if (budget.question_tokens == 0) return error.InvalidMemorillaTokens;
    const documents_fp16 = try backing.alloc(f16, documents.len * width);
    errdefer backing.free(documents_fp16);
    const question_fp16 = try backing.alloc(f16, width);
    errdefer backing.free(question_fp16);
    const document_features = try backing.alloc(f32, documents.len * width);
    errdefer backing.free(document_features);
    const question_features = try backing.alloc(f32, width);
    errdefer backing.free(question_features);
    const question_text = try backing.dupe(u8, canonical);
    errdefer backing.free(question_text);
    const document_sha256 = try backing.alloc([32]u8, documents.len);
    errdefer backing.free(document_sha256);
    for (documents, vectors[0..documents.len], 0..) |text, vector, i| {
        try control.check();
        try copyFeature(vector, documents_fp16[i * width ..][0..width], document_features[i * width ..][0..width]);
        std.crypto.hash.sha2.Sha256.hash(text, &document_sha256[i], .{});
    }
    try copyFeature(vectors[documents.len], question_fp16, question_features);
    const recipe_hash = try recipeSha256(v1_profile);
    try control.check();
    owns_lease = false;
    return .{ .allocator = backing, .document_count = documents.len, .question_tokens = budget.question_tokens, .documents_fp16 = documents_fp16, .question_fp16 = question_fp16, .documents = document_features, .question = question_features, .question_text = question_text, .document_sha256 = document_sha256, .producer = .{ .profile = v1_profile, .publication = publication, .identity = identity, .source_lock_sha256 = v1SourceLockSha256(), .recipe_sha256 = recipe_hash }, .lease = lease };
}

// --- per-token (copy17 `tokens-dual`) encoder entry ---

/// Reader input for the research profile. Owns `features` and keeps the
/// complete host footprint admitted until deinit.
pub const TokenFeatures = struct {
    allocator: A,
    features: token_features.Features,
    /// Active length of the question row (its ids plus EOS).
    question_tokens: usize,
    producer: Producer,
    lease: memory.AdmissionLease,

    pub fn deinit(self: *TokenFeatures) void {
        self.features.deinit();
        self.lease.release();
        self.* = undefined;
    }
};

const row_bytes: usize = width * @sizeOf(f32);
const padded_length: usize = token_features.max_padded_length;

/// Host bytes retained or transiently touched by `encodeTokenFeatures` for
/// `documents` documents whose texts (plus the question) total `text_bytes`:
/// both host output tensors at the padded ceiling, the `[2M, width]` feature
/// rows plus the row-selection temporaries, the query, copy ids, the i64 ids
/// and mask, the per-row i32 encodings, bookkeeping slices and the prefixed
/// and doubled text copies.
pub fn tokenFeatureHostBytes(documents: usize, text_bytes: usize) !usize {
    if (documents == 0) return error.EmptyMemorillaCollection;
    const rows = try std.math.add(usize, documents, 1);
    const positions = try std.math.mul(usize, rows, padded_length);
    const document_positions = try std.math.mul(usize, documents, padded_length);
    var total = try std.math.mul(usize, positions, row_bytes * 2);
    total = try std.math.add(usize, total, try std.math.mul(usize, document_positions, row_bytes * 4));
    total = try std.math.add(usize, total, row_bytes);
    total = try std.math.add(usize, total, try std.math.mul(usize, document_positions, 2 * @sizeOf(u32)));
    total = try std.math.add(usize, total, try std.math.mul(usize, positions, 2 * @sizeOf(i64) + @sizeOf(i32)));
    total = try std.math.add(usize, total, try std.math.mul(usize, rows, 4 * @sizeOf(usize) + @sizeOf([]const u8)));
    total = try std.math.add(usize, total, try std.math.mul(usize, text_bytes, 3));
    return std.math.add(usize, total, try std.math.mul(usize, documents, 2));
}

/// Explicit admission for one tapped encoder run of `batch` rows padded to
/// `seq_len`: i64 ids and mask plus the tap scalar as inputs, two F32
/// `[batch, seq_len, width]` host outputs, and a conservative activation
/// workspace of eight F32 row-widths per padded position at the 128 ceiling.
/// The session's own admission turns `workspace_bytes` into the backend
/// scratch that `archRunAdmitted` requires for a Memorilla encoder.
pub fn tokenFeatureRunRequest(batch: usize, seq_len: usize) !session_mod.RunRequest {
    if (batch < 2) return error.InvalidMemorillaDocumentCount;
    if (seq_len == 0 or seq_len > padded_length) return error.MemorillaEncoderBoundExceeded;
    const positions = try std.math.mul(usize, batch, seq_len);
    return .{
        .batch = batch,
        .sequence = seq_len,
        .input_bytes = try std.math.add(usize, try std.math.mul(usize, positions, 2 * @sizeOf(i64)), @sizeOf(i64)),
        .output_bytes = try std.math.mul(usize, positions, row_bytes * 2),
        .workspace_bytes = try std.math.mul(usize, try std.math.mul(usize, batch, padded_length), row_bytes * 8),
    };
}

/// Per-token `tokens-dual` features for the research encoder profile. The
/// caller retains its handle and authorization closure and passes the already
/// canonical question (`canonicalQuestion`); documents are encoded verbatim
/// after the reference's space prefix. Same guards as `encode`: profile tag,
/// managed publication, embedding identity before and after, embedding asset
/// lease and execution lane, with the forward admitted through the session.
pub fn encodeTokenFeatures(a: A, handle: *const manager.ModelHandle, documents: []const []const u8, question: []const u8, limits: memory.Limits, control: Control) !TokenFeatures {
    try control.check();
    if (documents.len == 0) return error.EmptyMemorillaCollection;
    if (documents.len > qualified.max_documents or question.len > max_text_bytes) return error.MemorillaTextLimitExceeded;
    const canonical = try canonicalQuestion(question);
    if (canonical.len == 0) return error.EmptyMemorillaQuestion;
    if (!std.mem.eql(u8, canonical, question)) return error.NonCanonicalMemorillaQuestion;
    var text_bytes: usize = question.len;
    for (documents) |text| {
        if (text.len > max_text_bytes) return error.MemorillaTextLimitExceeded;
        _ = try std.unicode.Utf8View.init(text);
        if (std.mem.trim(u8, text, &std.ascii.whitespace).len == 0) return error.BlankMemorillaDocument;
        text_bytes += text.len;
    }
    const host_bytes = try tokenFeatureHostBytes(documents.len, text_bytes);
    const profile = research_profile;
    const backing = @import("antfly_platform").allocator.reclaimingScratch(a);
    var pin = handle.duplicate();
    defer pin.release();
    const model = pin.get();
    try qualified.requireRecipe(profile, model.manifest);
    try factory.requireMemorillaProfile(model.session, profile);
    const publication = model.manifest.managed_publication orelse return error.MemorillaManagedProducerRequired;
    var lease = try pin.manager.acquireRunResourceAmounts(.cpu, limits, .{ .host_scratch_bytes = host_bytes });
    var owns_lease = true;
    defer if (owns_lease) lease.release();
    try control.check();
    var assets = try model.acquireEmbeddingAssetLeaseWithControl(false, control);
    defer assets.release();
    {
        try model.lockEmbeddingAssetsWithControl(control);
        defer model.unlockEmbeddingAssets();
        try model.ensurePrimaryEmbeddingAssetsLockedWithControl(true, false, control);
    }
    try model.verifyEmbeddingIdentityWithControl(control);
    const identity = model.embedding_identity orelse return error.InvalidEmbeddingIdentity;
    const tokenizer = model.getTokenizer();
    const lane = try model.embedding_execution_lane.acquire(control);
    defer lane.release();
    var batch = try token_features.prepare(backing, tokenizer, documents, question);
    defer batch.deinit();
    try control.check();
    const shape = [_]i64{ @intCast(batch.batch), @intCast(batch.seq_len) };
    var ids = try Tensor.initInt64(backing, "input_ids", &shape, batch.input_ids);
    defer ids.deinit();
    const mask_values = try backing.alloc(i64, batch.input_ids.len);
    defer backing.free(mask_values);
    for (0..batch.batch) |row| {
        for (0..batch.seq_len) |position| mask_values[row * batch.seq_len + position] = if (position < batch.active[row]) 1 else 0;
    }
    var mask = try Tensor.initInt64(backing, "attention_mask", &shape, mask_values);
    defer mask.deinit();
    var tap = try Tensor.initInt64(backing, factory.gpt_hidden_states_tap_input, &.{1}, &.{@intCast(token_features.tap_layer)});
    defer tap.deinit();
    var permit = try model.session.admit(try tokenFeatureRunRequest(batch.batch, batch.seq_len));
    defer permit.deinit();
    try control.check();
    const outputs = try permit.runWithControl(&.{ ids, mask, tap }, backing, control);
    defer {
        for (outputs) |*output| output.deinit();
        backing.free(outputs);
    }
    try control.check();
    try model.verifyEmbeddingIdentityWithControl(control);
    try factory.requireMemorillaProfile(model.session, profile);
    if (outputs.len != 2 or outputs[0].dtype != .f32 or outputs[1].dtype != .f32 or
        !std.mem.eql(u8, outputs[1].name, factory.gptHiddenStatesTapName(token_features.tap_layer)))
        return error.UnexpectedMemorillaEncoderOutputs;
    var features = try token_features.features(backing, outputs[0].asFloat32(), outputs[1].asFloat32(), &batch);
    errdefer features.deinit();
    const source_lock = try researchSourceLockSha256(backing, model.model_dir);
    const recipe_hash = try recipeSha256(profile);
    try control.check();
    owns_lease = false;
    return .{
        .allocator = backing,
        .features = features,
        .question_tokens = batch.active[batch.batch - 1],
        .producer = .{ .profile = profile, .publication = publication, .identity = identity, .source_lock_sha256 = source_lock, .recipe_sha256 = recipe_hash },
        .lease = lease,
    };
}

test "Memorilla question rendering matches Python strip including Unicode boundaries" {
    try std.testing.expectEqualStrings("María", try canonicalQuestion("\x1c\t\u{3000}María\u{a0}\u{202f}"));
    try std.testing.expectError(error.EmptyMemorillaQuestion, canonicalQuestion("\u{2007}\r\n"));
    try std.testing.expectEqualStrings("x\u{200b}", try canonicalQuestion(" x\u{200b} "));
    try std.testing.expectError(error.InvalidUtf8, canonicalQuestion("\xff"));
    try std.testing.expectError(error.InvalidMemorillaQuestion, canonicalQuestion("Use <|memory|> directly"));
}

test "Memorilla storage casts FP16 before BF16 with ties to even" {
    const result = try storedValue(1.00390625);
    try std.testing.expectEqual(@as(f32, 1), result.memory);
    try std.testing.expectEqual(@as(f32, 1.00390625), @as(f32, @floatCast(result.fp16)));
    try std.testing.expectEqual(@as(f32, 1.015625), (try storedValue(1.01171875)).memory);
    try std.testing.expectError(error.NonFiniteMemorillaFeature, storedValue(std.math.nan(f32)));
}

test "Memorilla encoder rejects cancellation empty collection and injected question before touching a handle" {
    var untouched: manager.ModelHandle = undefined;
    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.Cancelled;
        }
    };
    try std.testing.expectError(error.Cancelled, encode(std.testing.allocator, &untouched, &.{"doc"}, "question", .{}, .{ .check_fn = Cancel.check }));
    try std.testing.expectError(error.EmptyMemorillaCollection, encode(std.testing.allocator, &untouched, &.{}, "question", .{}, .{}));
    try std.testing.expectError(error.InvalidMemorillaQuestion, encodeQuestion(std.testing.allocator, &untouched, "<|memory|>", .{}, .{}));
}

test "Memorilla token-feature encoder rejects deadline collection and text limits before touching a handle" {
    var untouched: manager.ModelHandle = undefined;
    const a = std.testing.allocator;
    try std.testing.expectError(error.Timeout, encodeTokenFeatures(a, &untouched, &.{"doc"}, "question", .{}, .{ .deadline_ns = 0 }));
    try std.testing.expectError(error.EmptyMemorillaCollection, encodeTokenFeatures(a, &untouched, &.{}, "question", .{}, .{}));
    const many: [qualified.max_documents + 1][]const u8 = @splat("doc");
    try std.testing.expectError(error.MemorillaTextLimitExceeded, encodeTokenFeatures(a, &untouched, &many, "question", .{}, .{}));
    const long = try a.alloc(u8, max_text_bytes + 1);
    defer a.free(long);
    @memset(long, 'x');
    try std.testing.expectError(error.MemorillaTextLimitExceeded, encodeTokenFeatures(a, &untouched, &.{long}, "question", .{}, .{}));
    try std.testing.expectError(error.MemorillaTextLimitExceeded, encodeTokenFeatures(a, &untouched, &.{"doc"}, long, .{}, .{}));
    try std.testing.expectError(error.BlankMemorillaDocument, encodeTokenFeatures(a, &untouched, &.{" \t\n"}, "question", .{}, .{}));
    try std.testing.expectError(error.InvalidUtf8, encodeTokenFeatures(a, &untouched, &.{"\xff"}, "question", .{}, .{}));
    try std.testing.expectError(error.EmptyMemorillaQuestion, encodeTokenFeatures(a, &untouched, &.{"doc"}, " \u{3000}", .{}, .{}));
    try std.testing.expectError(error.NonCanonicalMemorillaQuestion, encodeTokenFeatures(a, &untouched, &.{"doc"}, " question ", .{}, .{}));
    try std.testing.expectError(error.InvalidMemorillaQuestion, encodeTokenFeatures(a, &untouched, &.{"doc"}, "<|memory|>", .{}, .{}));
}

test "Memorilla token-feature lease sizing covers outputs features inputs and text" {
    // rows = 2, positions = 256, document positions = 128, row = 10240 bytes:
    // 256*20480 + 128*40960 + 10240 + 128*8 + 256*20 + 2*48 + 3*10 + 2.
    try std.testing.expectEqual(@as(usize, 10_502_272), try tokenFeatureHostBytes(1, 10));
    try std.testing.expect(try tokenFeatureHostBytes(64, 0) > try tokenFeatureHostBytes(63, 8192 * 65));
    try std.testing.expectError(error.EmptyMemorillaCollection, tokenFeatureHostBytes(0, 0));
    try std.testing.expectError(error.Overflow, tokenFeatureHostBytes(std.math.maxInt(usize), 0));
    const request = try tokenFeatureRunRequest(3, 50);
    try std.testing.expectEqual(@as(usize, 3), request.batch);
    try std.testing.expectEqual(@as(usize, 50), request.sequence);
    try std.testing.expectEqual(@as(usize, 150 * 16 + 8), request.input_bytes);
    try std.testing.expectEqual(@as(?usize, 150 * 2560 * 4 * 2), request.output_bytes);
    try std.testing.expectEqual(@as(usize, 3 * 128 * 2560 * 4 * 8), request.workspace_bytes);
    try std.testing.expect(!request.workspace_planned);
    try std.testing.expectError(error.InvalidMemorillaDocumentCount, tokenFeatureRunRequest(1, 8));
    try std.testing.expectError(error.MemorillaEncoderBoundExceeded, tokenFeatureRunRequest(2, 129));
    try std.testing.expectError(error.MemorillaEncoderBoundExceeded, tokenFeatureRunRequest(2, 0));
}

test "Memorilla producer check is profile specific" {
    const publication: [32]u8 = @splat(7);
    const v1 = Producer{ .profile = v1_profile, .publication = publication, .identity = qualified.executionIdentity(v1_profile), .source_lock_sha256 = v1SourceLockSha256(), .recipe_sha256 = try recipeSha256(v1_profile) };
    try requireProducer(v1, publication, v1_profile);
    try std.testing.expectError(error.IncompatibleMemorillaProducer, requireProducer(v1, @splat(8), v1_profile));
    try std.testing.expectError(error.IncompatibleMemorillaProducer, requireProducer(v1, publication, .qwen3_decoder_f32_kv_f32_v1));
    var stale = v1;
    stale.source_lock_sha256[0] = if (stale.source_lock_sha256[0] == '0') '1' else '0';
    try std.testing.expectError(error.IncompatibleMemorillaProducer, requireProducer(stale, publication, v1_profile));
    try std.testing.expectError(error.IncompatibleMemorillaProducer, recipeSha256(.qwen3_decoder_f32_kv_f32_v1));
    const lock: [64]u8 = @splat('a');
    const candidate = Producer{ .profile = research_profile, .publication = publication, .identity = qualified.executionIdentity(research_profile), .source_lock_sha256 = lock, .recipe_sha256 = try recipeSha256(research_profile) };
    try requireProducer(candidate, publication, research_profile);
    try requireProducerSourceLock(candidate, lock);
    try std.testing.expectError(error.IncompatibleMemorillaProducer, requireProducerSourceLock(candidate, @splat('b')));
    try std.testing.expectError(error.IncompatibleMemorillaProducer, requireProducer(candidate, publication, v1_profile));
    try std.testing.expectError(error.IncompatibleMemorillaProducer, requireProducer(v1, publication, research_profile));
    try std.testing.expect(!std.mem.eql(u8, &candidate.recipe_sha256, &v1.recipe_sha256));
    try std.testing.expect(!std.mem.eql(u8, &candidate.identity, &v1.identity));
    var upper = candidate;
    upper.source_lock_sha256[3] = 'A';
    try std.testing.expectError(error.IncompatibleMemorillaProducer, requireProducer(upper, publication, research_profile));
}
