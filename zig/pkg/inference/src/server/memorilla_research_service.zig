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

//! Text-input composition for the encoder-LoRA research candidate: per-token
//! encoder features, research reader bundle, copy head, snapped contiguous
//! spans and the unchanged decoder injection. Same lease, budget and fence
//! discipline as `memorilla_service`; nothing here is a qualified release.
const std = @import("std");
const platform = @import("antfly_platform");
const common = @import("memorilla_service.zig");
const manager_mod = @import("model_manager.zig");
const qualified = @import("../models/memorilla_qualification.zig");
const research = @import("../models/memorilla_research_runtime.zig");
const encoder = @import("../pipelines/memorilla_encoder.zig");
const memory_pipeline = @import("../pipelines/memorilla.zig");
const token_features = @import("../pipelines/memorilla_token_features.zig");
const copy = @import("../pipelines/memorilla_copy.zig");
const grounding = @import("../pipelines/memorilla_grounding.zig");
const decode = @import("../pipelines/memorilla_generation.zig");
const generation = @import("../pipelines/generation.zig");
const factory = @import("../architectures/session_factory.zig");
const gpt = @import("../architectures/gpt.zig");
const memory = @import("../runtime/root.zig").tier.memory;
const execution = @import("../ops/execution_scope.zig");
const ops = @import("../ops/ops.zig");
const Control = @import("../execution_control.zig").InferenceExecutionControl;
const file = @import("../util/c_file.zig");
const PhysicalBudgetScope = if (@import("build_options").enable_cuda) @import("../ops/cuda/cuda_compute.zig").CudaCompute.MemorillaDeviceBudgetScope else struct {
    fn deinit(_: *@This()) void {}
};
const A = std.mem.Allocator;

pub const Profile = common.Profile;
pub const Paths = common.Paths;
pub const Options = common.Options;
pub const Stage = common.Stage;
pub const encoder_profile: qualified.Profile = .qwen3_encoder_lora_research_f32_v1;
pub const decoder_profile: qualified.Profile = .qwen3_decoder_f32_kv_f32_v1;

/// Slots are fixed by the decoder's trained memory frame; the reader bundle
/// must agree (checked at open).
pub const slots: usize = 16;
/// Upper bound on reader rows per document: two layers of at most 64 kept
/// rows (a document and its EOS fit a 128-token padded row at most twice).
pub const max_rows_per_document: usize = 128;
/// Reader scratch per uploaded row: the rows themselves, their norm, two
/// cross-attention key/value projections and headroom (plan 5.5).
const reader_scratch_per_row: usize = qualified.width * @sizeOf(f32) * 6;
/// Decoder-side transients for the copy/snap stages: source rows, the
/// 16 x vocab score matrix and the completed memory rows.
const decoder_copy_scratch_bytes: usize = 32 * 1024 * 1024;
/// Host chunk for the one-time embedding-table norm computation.
const table_norm_chunk_rows: usize = 8192;

/// Outcome of the answer-format guard applied to the decoder's text.
pub const AnswerGuard = enum { accepted, rejected };

/// The literal abstention answer of the trained protocol.
pub const unknown_answer: []const u8 = "UNKNOWN";
/// Rule 2: the longest accepted answer in bytes (a three-word name or a code
/// is far shorter; a sentence or an echoed frame is not).
pub const guard_max_bytes: usize = 64;
/// Rule 3: the most whitespace-separated words an answer may contain.
pub const guard_max_words: usize = 6;
/// Rule 5: the memory frame literal the decoder echoes when the record left
/// the copied slots; as a standalone word it never belongs to a short fact.
const frame_word: []const u8 = "Memory";

fn isGuardByte(byte: u8) bool {
    // Rule 6: letters, digits, space, hyphen, period and apostrophe only.
    // Hyphens appear in codes and double-barrelled names, periods in
    // initials, apostrophes in names such as O'Neil.
    return std.ascii.isAlphanumeric(byte) or byte == ' ' or byte == '-' or byte == '.' or byte == '\'';
}

/// Answer-format guard for direct-answer research serving. The research task
/// answers are short exact facts (a name of one to three words, a short code,
/// or the literal UNKNOWN); the frozen decoder has no format constraint and,
/// when the copied record misses the fact, can echo its own memory frame as
/// prose. The guard is general (no corpus-specific patterns) and turns such
/// output into an abstention. Rules, applied to the whitespace-trimmed text:
///
/// 1. the literal `UNKNOWN` is accepted (it is the protocol's abstention);
/// 2. empty text, or text longer than `guard_max_bytes`, is rejected;
/// 3. more than `guard_max_words` whitespace-separated words is rejected;
/// 4. any control character, including newline and tab, is rejected (a
///    second line is never part of a short fact);
/// 5. the memory frame literal `Memory` as a standalone word, or the
///    `<|memory|>` marker, is rejected (frame echo);
/// 6. any byte outside letters, digits, space, '-', '.', '\'' is rejected
///    (punctuation, braces, quotes and non-ASCII bytes are not short facts
///    in this protocol).
pub fn guardAnswer(text: []const u8) AnswerGuard {
    const trimmed = std.mem.trim(u8, text, &std.ascii.whitespace);
    if (std.mem.eql(u8, trimmed, unknown_answer)) return .accepted;
    if (trimmed.len == 0 or trimmed.len > guard_max_bytes) return .rejected;
    if (std.mem.indexOf(u8, trimmed, decode.marker) != null) return .rejected;
    var words: usize = 0;
    var it = std.mem.tokenizeAny(u8, trimmed, &std.ascii.whitespace);
    while (it.next()) |word| {
        words += 1;
        if (words > guard_max_words) return .rejected;
        if (std.mem.eql(u8, word, frame_word)) return .rejected;
    }
    for (trimmed) |byte| {
        if (std.ascii.isControl(byte)) return .rejected;
        if (!isGuardByte(byte)) return .rejected;
    }
    return .accepted;
}

/// Replaces a rejected generation's text with the abstention answer. The
/// token ids stay as decoded so callers can see what the decoder produced;
/// `finish_reason` is a static string in `GenerationResult` (its `deinit`
/// frees only text, reasoning text and ids).
fn abstainGeneration(result: *generation.GenerationResult) !void {
    const replacement = try result.allocator.dupe(u8, unknown_answer);
    result.allocator.free(result.text);
    result.text = replacement;
    result.finish_reason = "abstained";
}

pub const OwnedResult = struct {
    generation: generation.GenerationResult,
    answer_guard: AnswerGuard,
    encoder_execution_identity: [64]u8,
    decoder_execution_identity: [64]u8,
    module_sha256: [64]u8,
    candidate_sha256: [64]u8,
    document_count: usize,
    question_tokens: usize,
    response_lease: memory.AdmissionLease,

    pub fn deinit(self: *OwnedResult) void {
        self.generation.deinit();
        self.response_lease.release();
        self.* = undefined;
    }
};

/// Pre-admission validation shared with the HTTP handler. Only policy and
/// shape; no model, reader or GPU state is consulted.
pub fn validateInputs(profile: Profile, question: []const u8, documents: []const []const u8, identity: []const u8, max_new_tokens: usize) !void {
    try (platform.inference_settings.Memorilla{ .enabled = true, .profiles = &.{profile} }).validate();
    if (profile.reader_kind != .encoder_lora_research) return error.UnsupportedMemorillaProfile;
    if (question.len > grounding.max_document_bytes) return error.MemorillaTextLimitExceeded;
    _ = try qualified.canonicalQuestion(question);
    if (!std.mem.eql(u8, identity, &qualified.executionIdentity(encoder_profile))) return error.IncompatibleMemorillaProducer;
    if (max_new_tokens == 0 or max_new_tokens > profile.max_new_tokens) return error.MemorillaGenerationLimitExceeded;
    if (documents.len == 0) return error.EmptyMemorillaCollection;
    if (documents.len > profile.max_documents) return error.MemorillaCollectionBudgetExceeded;
    try grounding.validateSources(documents, documents.len);
}

fn requireLimits(limits: memory.Limits) !void {
    if (limits.host_limit_bytes == 0 or limits.backend_limit_bytes == 0 or limits.combined_limit_bytes == 0 or
        limits.kv_limit_bytes == 0 or limits.scratch_limit_bytes == 0) return error.MemorillaFiniteBudgetRequired;
}

fn notify(options: Options, stage: Stage, manager: *manager_mod.ModelManager) !void {
    if (options.stage) |callback| try callback(options.context, stage, manager);
}

fn add(x: usize, y: usize) !usize {
    return std.math.add(usize, x, y) catch error.MemorillaResourceOverflow;
}

fn mul(x: usize, y: usize) !usize {
    return std.math.mul(usize, x, y) catch error.MemorillaResourceOverflow;
}

/// The system prompt is part of the candidate's trained prompt recipe. Read
/// it from the bundle before any model loads so prompt budgets fail early;
/// the opened reader must later report the same text.
fn bundleSystemPrompt(a: A, bundle_directory: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ bundle_directory, "protocol.json" });
    defer a.free(path);
    const bytes = try file.readFileMax(a, path, 256 * 1024);
    defer a.free(bytes);
    var parsed = try std.json.parseFromSlice(std.json.Value, a, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidMemorillaManifest;
    const prompt = parsed.value.object.get("system_prompt") orelse return error.InvalidMemorillaManifest;
    if (prompt != .string or prompt.string.len == 0 or prompt.string.len > 4096 or
        std.mem.indexOf(u8, prompt.string, decode.marker) != null) return error.InvalidMemorillaManifest;
    return a.dupe(u8, prompt.string);
}

fn preflightQuestion(a: A, io: std.Io, manager: *manager_mod.ModelManager, selected: @import("../registry/model_selection.zig").ResolvedModel, canonical: []const u8, control: Control) !usize {
    try control.check();
    var assets = try common.acquireGenerationAssets(a, io, manager, selected, encoder_profile, control);
    defer assets.deinit();
    const ids = try assets.tokenizer().encode(a, canonical);
    defer a.free(ids);
    // The research batch appends EOS itself and bounds every row at 128.
    if (ids.len == 0) return error.InvalidMemorillaTokens;
    if (ids.len + 1 > token_features.max_padded_length) return error.MemorillaEncoderBoundExceeded;
    try selected.verify(a, io);
    try control.check();
    return ids.len + 1;
}

pub fn generate(a: A, io: std.Io, manager: *manager_mod.ModelManager, profile: Profile, paths: Paths, question: []const u8, documents: []const []const u8, identity: []const u8, max_new_tokens: usize, limits: memory.Limits, request_limits: platform.resource_policy.InferenceRequestLimits, control: Control, options_in: Options) !OwnedResult {
    var options = options_in;
    options.request_limits = request_limits;
    var result = try generateImpl(a, io, manager, profile, paths, question, documents, identity, max_new_tokens, limits, control, options);
    errdefer result.deinit();
    try control.check();
    return result;
}

/// Host F32 row norms of the decoder's input-embedding table, computed once
/// per opened reader in bounded device-row chunks. The snapper divides GPU
/// dot products by them.
fn ensureTableNorms(a: A, owner: *research.Owner, cb: *const ops.ComputeBackend, config: gpt.Config, control: Control) ![]const f32 {
    if (owner.table_norms) |norms| {
        if (owner.table_norms_vocab != config.vocab_size or norms.len != config.vocab_size) return error.ModelArtifactsChanging;
        return norms;
    }
    const weight = try gpt.getEmbeddingWeight(cb, config);
    defer cb.free(weight);
    const norms = try owner.allocator.alloc(f32, config.vocab_size);
    errdefer owner.allocator.free(norms);
    var start: usize = 0;
    while (start < config.vocab_size) : (start += table_norm_chunk_rows) {
        try control.check();
        const count = @min(table_norm_chunk_rows, config.vocab_size - start);
        const chunk = try cb.sliceRows2D(a, weight, start, count, config.hidden_size);
        defer cb.free(chunk);
        const host = try cb.toFloat32(chunk, a);
        defer a.free(host);
        if (host.len != count * config.hidden_size) return error.UnexpectedEmbeddingTable;
        const partial = try copy.rowNorms(a, host, count, config.hidden_size);
        defer a.free(partial);
        @memcpy(norms[start..][0..count], partial);
    }
    owner.setTableNorms(norms);
    return owner.table_norms.?;
}

/// Everything the copy stages produce on the host; device transients are
/// released inside `copyAndSnap` so generation starts from resident weights.
const Snapped = struct {
    a: A,
    completed_ids: []i32,
    rows: []f32,

    fn deinit(self: *Snapped) void {
        self.a.free(self.completed_ids);
        self.a.free(self.rows);
    }
};

fn copyAndSnap(a: A, owner: *research.Owner, decoder_cb: *const ops.ComputeBackend, config: gpt.Config, tokenizer: @import("inference_tokenizer").Tokenizer, memories: []const f32, copy_ids: []const u32, control: Control) !Snapped {
    const hidden = config.hidden_size;
    if (copy_ids.len == 0) return error.EmptyMemorillaCollection;
    const source_ids = try a.alloc(i32, copy_ids.len);
    defer a.free(source_ids);
    const wide = try a.alloc(i64, copy_ids.len);
    defer a.free(wide);
    for (copy_ids, source_ids, wide) |id, *narrow, *w| {
        if (id >= config.vocab_size) return error.InvalidMemorillaTokens;
        narrow.* = @intCast(id);
        w.* = id;
    }
    const norms = try ensureTableNorms(a, owner, decoder_cb, config, control);
    const weight = try gpt.getEmbeddingWeight(decoder_cb, config);
    defer decoder_cb.free(weight);
    const sources = blk: {
        const rows = try decoder_cb.embeddingLookup(weight, wide, copy_ids.len, hidden);
        defer decoder_cb.free(rows);
        break :blk try decoder_cb.toFloat32(rows, a);
    };
    defer a.free(sources);
    try control.check();
    var fwd = try copy.forward(a, &owner.copy_head, memories, slots, sources, copy_ids.len);
    defer fwd.deinit();
    var runs = try copy.contiguousRunsForward(a, &fwd, memories, sources, copy.default_min_gate);
    defer runs.deinit();
    const scores = blk: {
        const output_ct = try decoder_cb.fromFloat32Shape(runs.output, &.{ @intCast(slots), @intCast(hidden) });
        defer decoder_cb.free(output_ct);
        const scores_ct = try decoder_cb.linearNoBias(output_ct, weight, slots, hidden, config.vocab_size);
        defer decoder_cb.free(scores_ct);
        break :blk try decoder_cb.toFloat32(scores_ct, a);
    };
    defer a.free(scores);
    const output_norms = try copy.rowNorms(a, runs.output, slots, hidden);
    defer a.free(output_norms);
    const snapped = try copy.snapIds(a, scores, norms, output_norms, slots, config.vocab_size);
    defer a.free(snapped);
    try control.check();
    // The pad is the single-space token of the decoder vocabulary; the
    // trained records end in it.
    const pad = try tokenizer.encode(a, " ");
    defer a.free(pad);
    if (pad.len != 1 or pad[0] != copy.default_pad_id) return error.InvalidMemorillaTokenizer;
    const completed_ids = blk: {
        var completed = try copy.completeCopiedWords(a, tokenizer, snapped, runs.spans, source_ids, pad[0]);
        defer completed.deinit();
        break :blk try a.dupe(i32, completed.ids);
    };
    errdefer a.free(completed_ids);
    const completed_wide = try a.alloc(i64, slots);
    defer a.free(completed_wide);
    for (completed_ids, completed_wide) |id, *w| {
        if (id < 0 or id >= config.vocab_size) return error.InvalidMemorillaTokens;
        w.* = id;
    }
    const rows = blk: {
        const ct = try decoder_cb.embeddingLookup(weight, completed_wide, slots, hidden);
        defer decoder_cb.free(ct);
        break :blk try decoder_cb.toFloat32(ct, a);
    };
    errdefer a.free(rows);
    if (rows.len != slots * hidden) return error.UnexpectedEmbeddingTable;
    return .{ .a = a, .completed_ids = completed_ids, .rows = rows };
}

fn generateImpl(a: A, io: std.Io, manager: *manager_mod.ModelManager, profile: Profile, paths: Paths, question: []const u8, documents: []const []const u8, identity: []const u8, max_new_tokens: usize, limits: memory.Limits, control: Control, options: Options) !OwnedResult {
    try control.check();
    try validateInputs(profile, question, documents, identity, max_new_tokens);
    try common.validateRequestLimits(options.request_limits, documents.len, max_new_tokens);
    try requireLimits(limits);
    try qualified.requireCudaEnvironment();
    if (paths.encoder.publication == null or paths.decoder.publication == null) return error.MemorillaManagedProducerRequired;
    try paths.encoder.verify(a, io);
    try paths.decoder.verify(a, io);
    const encoder_control = paths.encoder.bind(control);
    const decoder_control = paths.decoder.bind(control);
    const canonical = try qualified.canonicalQuestion(question);

    var response_lease = try manager.acquireRunResourceAmounts(.cpu, limits, .{ .host_scratch_bytes = 16 * 1024 * 1024 });
    var transfer_response = false;
    defer if (!transfer_response) response_lease.release();
    const backing = platform.allocator.reclaimingScratch(a);
    const preflight_question_tokens = try preflightQuestion(backing, io, manager, paths.encoder, canonical, encoder_control);
    try common.validateInputTokenLimit(options.request_limits, preflight_question_tokens);
    const system_prompt = try bundleSystemPrompt(backing, profile.memory_module_path);
    defer backing.free(system_prompt);
    var assets = try common.acquireGenerationAssets(backing, io, manager, paths.decoder, decoder_profile, decoder_control);
    defer assets.deinit();
    const config = try assets.decoderConfig(backing, paths.decoder.path);
    const bounded = decode.Limits{ .max_new_tokens = profile.max_new_tokens };
    var text = try decode.prepareText(backing, assets.tokenizer(), assets.chatTemplate() orelse return error.MemorillaChatTemplateRequired, config, canonical, .{ .system_prompt = system_prompt }, bounded, decoder_control);
    defer text.deinit();
    try common.validateInputTokenLimit(options.request_limits, text.input_ids.len);
    if (options.prepared_text) |callback| try callback(options.context, &text);
    const plan = try decode.ResourcePlan.create(config, .cuda, text.input_ids.len, text.rendered.len, slots, max_new_tokens, bounded);

    // Reader admission is bounded by the row budget implied by the document
    // count (features cannot exceed it; the encoder enforces the 128-token
    // row bound). Host scratch covers the two feature layers on the host.
    const max_rows = try mul(documents.len, max_rows_per_document);
    const reader_request = memory.AdmissionAmounts{
        .host_scratch_bytes = try add(16 * 1024 * 1024, try mul(max_rows, qualified.width * @sizeOf(f32))),
        .backend_scratch_bytes = try add(128 * 1024 * 1024, try mul(max_rows, reader_scratch_per_row)),
    };
    const copy_request = memory.AdmissionAmounts{ .backend_scratch_bytes = decoder_copy_scratch_bytes };
    const decoder_request = try (try memory.AdmissionAmounts.fromEstimate(plan.decoder).merge(.fromEstimate(plan.host))).merge(copy_request);
    const request_amounts = try decoder_request.merge(reader_request);
    var request_lease = try manager.acquireRunResourceAmounts(.gpu, limits, request_amounts);
    defer request_lease.release();
    var decoder_budget = memory.RunBudget.init(try common.requestBudgetLimits(limits, decoder_request));
    var reader_budget = memory.RunBudget.init(try common.requestBudgetLimits(limits, reader_request));
    var reservation = try plan.reserve(&decoder_budget);
    defer reservation.deinit();
    const copy_estimate = memory.Estimate{ .prompt_tokens = 0, .retained_tokens = 0, .kv_bytes = 0, .kv_tier = .backend, .scratch_bytes = decoder_copy_scratch_bytes, .scratch_tier = .backend };
    try decoder_budget.reserveEstimate(copy_estimate);
    defer decoder_budget.releaseEstimate(copy_estimate);
    const reader_estimate = memory.Estimate{ .prompt_tokens = 0, .retained_tokens = 0, .kv_bytes = 0, .kv_tier = .backend, .scratch_bytes = reader_request.backend_scratch_bytes, .scratch_tier = .backend };
    try reader_budget.reserveEstimate(reader_estimate);
    defer reader_budget.releaseEstimate(reader_estimate);
    const reader_host_estimate = memory.Estimate{ .prompt_tokens = 0, .retained_tokens = 0, .kv_bytes = 0, .kv_tier = .host, .scratch_bytes = reader_request.host_scratch_bytes, .scratch_tier = .host };
    try reader_budget.reserveEstimate(reader_host_estimate);
    defer reader_budget.releaseEstimate(reader_host_estimate);
    try notify(options, .prepared, manager);

    var encoder_handle = try common.acquireResidentModel(manager, paths.encoder.path, encoder_control);
    defer encoder_handle.release();
    const encoder_model = encoder_handle.get();
    try encoder_model.verifyMemorillaIdentityWithControl(encoder_control);
    try notify(options, .encoder_loaded, manager);
    var features = try encoder.encodeTokenFeatures(backing, &encoder_handle, documents, canonical, limits, encoder_control);
    defer features.deinit();
    try encoder.requireProducer(features.producer, paths.encoder.publication.?, encoder_profile);
    // The research lock is per directory: pin the producer to the encoder
    // directory this request resolved, not just to the contract identity.
    try encoder.requireProducerSourceLock(features.producer, try encoder.researchSourceLockSha256(backing, paths.encoder.path));
    if (features.features.rows == 0 or features.features.rows > max_rows) return error.MemorillaCollectionBudgetExceeded;
    try notify(options, .question_encoded, manager);
    try control.check();
    try paths.encoder.verify(a, io);
    try paths.decoder.verify(a, io);

    var decoder_handle = try common.acquireResidentModel(manager, paths.decoder.path, decoder_control);
    defer decoder_handle.release();
    const decoder_model = decoder_handle.get();
    try decoder_model.verifyMemorillaIdentityWithControl(decoder_control);
    try notify(options, .decoder_loaded, manager);
    var native_guard = try control.enterUninterruptible(.uninterruptible);
    defer native_guard.deinit();
    var reader_handle = try decoder_model.memorilla_research_reader_cache.acquire(decoder_model.allocator, io, manager, profile.memory_module_path, limits, control);
    defer reader_handle.deinit();
    var reader_request_clean = false;
    defer if (!reader_request_clean) reader_handle.invalidate();
    const reader = reader_handle.owner();
    if (reader.bundle.release or reader.bundle.config.num_memories != slots or reader.bundle.config.output_dim != config.hidden_size) return error.UnsupportedMemorillaProfile;
    if (!std.mem.eql(u8, reader.system_prompt, system_prompt)) return error.ModelArtifactsChanging;
    // One candidate, three artifacts: the merged encoder's lock and the reader
    // bundle must name the same candidate.json digest.
    {
        var lock = try qualified.loadResearchLock(backing, paths.encoder.path);
        defer lock.deinit();
        if (!std.mem.eql(u8, &lock.candidate_sha256, &reader.candidate_sha256)) return error.MemorillaCandidateMismatch;
    }
    try notify(options, .reader_loaded, manager);
    const module_sha256 = reader.bundle.weights_sha256;
    const candidate_sha256 = reader.candidate_sha256;

    var reader_backend = try reader.computeBackendWithBudget(&reader_budget);
    reader_backend.execution_control = control;
    reader_backend.execution_budget = &reader_budget;
    var reader_scope = try execution.Scope.acquire(control, .{ .mutex = &reader.execution_mutex });
    defer reader_scope.deinit();
    var reader_physical: PhysicalBudgetScope = if (comptime @import("build_options").enable_cuda)
        try reader.compute.beginMemorillaDeviceBudget(reader_request.backendTotalBytes())
    else
        return error.CudaNotEnabled;
    defer reader_physical.deinit();
    try reader_scope.bind(reader_backend);
    const output = blk: {
        defer {
            reader.releaseRequestResources() catch @panic("Memorilla research reader request cleanup failed; worker restart required");
            reader_request_clean = true;
        }
        const rows = features.features.rows;
        const mask = try backing.alloc(bool, rows);
        defer backing.free(mask);
        @memset(mask, false);
        try encoder_model.verifyMemorillaIdentityWithControl(encoder_control);
        try decoder_model.verifyMemorillaIdentityWithControl(decoder_control);
        try factory.requireMemorillaCudaExecution(&reader_backend);
        // Reader: per-token rows of both layers, one batch, all rows visible.
        const memories = blk2: {
            const docs = try reader_backend.fromFloat32Shape(features.features.docs, &.{ @intCast(rows), @intCast(qualified.width) });
            defer reader_backend.free(docs);
            const query = try reader_backend.fromFloat32Shape(features.features.query, &.{ 1, @intCast(qualified.width) });
            defer reader_backend.free(query);
            const memories_ct = try memory_pipeline.forward(&reader_backend, backing, reader.bundle.config, .{ .documents = docs, .question = query, .padding_mask = mask, .batch = 1, .num_documents = rows, .max_document_rows = max_rows }, null);
            defer reader_backend.free(memories_ct);
            break :blk2 try reader_backend.toFloat32(memories_ct, backing);
        };
        defer backing.free(memories);
        try factory.requireMemorillaCudaExecution(&reader_backend);
        try control.check();
        // Copy head, snapping and word completion. Decoder-side transients
        // run under the decoder's own execution scope and are released before
        // generation, which requires the decoder to hold only resident weights.
        var decoder_backend = try factory.getCoordinatedCudaComputeBackend(decoder_model.session, &decoder_budget, @intFromPtr(&reservation));
        defer decoder_backend.deinit();
        var snapped = blk3: {
            decoder_backend.execution_control = decoder_control;
            decoder_backend.execution_budget = &decoder_budget;
            var decoder_scope = try execution.Scope.acquire(control, .{ .mutex = decoder_model.nativeGenerationMutex(), .lane = &decoder_model.native_reader_execution_lane });
            defer decoder_scope.deinit();
            try decoder_scope.bind(decoder_backend);
            try factory.requireMemorillaCudaExecution(&decoder_backend);
            const result = try copyAndSnap(backing, reader, &decoder_backend, config, assets.tokenizer(), memories, features.features.copy_ids, control);
            errdefer {
                var owned = result;
                owned.deinit();
            }
            try factory.requireMemorillaCudaExecution(&decoder_backend);
            break :blk3 result;
        };
        defer snapped.deinit();
        try control.check();
        const rows_ct = try reader_backend.fromFloat32Shape(snapped.rows, &.{ @intCast(slots), @intCast(config.hidden_size) });
        defer reader_backend.free(rows_ct);
        var decoder = try decode.Decoder.fromQualifiedSession(decoder_model.session, &decoder_backend, assets.tokenizer(), decoder_model.nativeGenerationMutex());
        decoder.execution_lane = &decoder_model.native_reader_execution_lane;
        break :blk try decode.generate(a, decoder, &text, .{ .backend = &reader_backend, .tensor = rows_ct }, .{ .max_new_tokens = max_new_tokens, .teacher_forced_ids = options.teacher_forced_ids, .trace = options.trace, .ignore_eos = options.ignore_eos }, &reservation, bounded, decoder_control);
    };
    var result = output;
    errdefer result.deinit();
    // Serving policy for the research profile: an answer that is not a short
    // fact (plan 7.1) is published as an abstention, never as prose.
    const answer_guard = guardAnswer(result.text);
    if (answer_guard == .rejected) try abstainGeneration(&result);
    if (comptime @import("build_options").enable_cuda) {
        if (options.reader_physical_resources) |record| {
            const snapshot = reader.compute.ctx.device_allocations.snapshot();
            try record(options.context, .{ .baseline = reader_physical.baseline, .held_transient_bytes = reader_request.backendTotalBytes(), .limit = snapshot.limit, .live = snapshot.live, .lifetime_peak = snapshot.peak });
        }
    }
    try encoder_model.verifyMemorillaIdentityWithControl(encoder_control);
    try decoder_model.verifyMemorillaIdentityWithControl(decoder_control);
    try paths.encoder.verify(a, io);
    try paths.decoder.verify(a, io);
    try reader_handle.verify(control);
    try control.check();
    try notify(options, .decoded, manager);
    try control.check();
    if (features.question_tokens != preflight_question_tokens) return error.InvalidMemorillaTokens;
    transfer_response = true;
    return .{
        .generation = result,
        .answer_guard = answer_guard,
        .encoder_execution_identity = qualified.executionIdentity(encoder_profile),
        .decoder_execution_identity = qualified.executionIdentity(decoder_profile),
        .module_sha256 = module_sha256,
        .candidate_sha256 = candidate_sha256,
        .document_count = documents.len,
        .question_tokens = features.question_tokens,
        .response_lease = response_lease,
    };
}

test "Memorilla research validation rejects kind identity bounds and texts before models" {
    const released = Profile{ .name = "stage3", .encoder = "encoder", .decoder = "decoder", .memory_module_path = "reader" };
    var p = released;
    p.reader_kind = .encoder_lora_research;
    const identity = qualified.executionIdentity(encoder_profile);
    const docs = [_][]const u8{ "Alice Smith holds the brass key for Mozeba Depot.", "The code at Tutevo Depot is Jc4-001234." };
    try validateInputs(p, "\xc2\xa0question\xc2\xa0", &docs, &identity, 24);
    try std.testing.expectError(error.UnsupportedMemorillaProfile, validateInputs(released, "q", &docs, &identity, 24));
    try std.testing.expectError(error.EmptyMemorillaCollection, validateInputs(p, "q", &.{}, &identity, 1));
    try std.testing.expectError(error.IncompatibleMemorillaProducer, validateInputs(p, "q", &docs, "other", 1));
    try std.testing.expectError(error.IncompatibleMemorillaProducer, validateInputs(p, "q", &docs, &qualified.executionIdentity(.qwen3_encoder_f32_v1), 1));
    try std.testing.expectError(error.MemorillaGenerationLimitExceeded, validateInputs(p, "q", &docs, &identity, 129));
    try std.testing.expectError(error.InvalidMemorillaSourceText, validateInputs(p, "q", &.{" \n"}, &identity, 1));
    try std.testing.expectError(error.InvalidMemorillaSourceText, validateInputs(p, "q", &.{"a\x00b"}, &identity, 1));
    const long: [8193]u8 = @splat('a');
    try std.testing.expectError(error.MemorillaSourceTextLimitExceeded, validateInputs(p, "q", &.{&long}, &identity, 1));
    try std.testing.expectError(error.MemorillaTextLimitExceeded, validateInputs(p, &long, &docs, &identity, 1));
    var small = p;
    small.max_documents = 1;
    try std.testing.expectError(error.MemorillaCollectionBudgetExceeded, validateInputs(small, "q", &docs, &identity, 1));
}

test "Memorilla research service cancellation precedes component resolution and model loading" {
    try std.testing.expectError(error.Timeout, generate(std.testing.allocator, std.testing.io, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, .{}, .{ .deadline_ns = 0 }, .{}));
}

test "Memorilla research admission bounds scale with the document count" {
    try std.testing.expectEqual(@as(usize, 128), try mul(1, max_rows_per_document));
    try std.testing.expectEqual(@as(usize, 64 * 128), try mul(64, max_rows_per_document));
    try std.testing.expectError(error.MemorillaResourceOverflow, mul(std.math.maxInt(usize), 2));
    try std.testing.expectEqual(qualified.width * 4 * 6, reader_scratch_per_row);
}

test "Memorilla research answer guard accepts short facts and the abstention literal" {
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer("Kg9-009369"));
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer("Komode Tuloco"));
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer("UNKNOWN"));
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer("  UNKNOWN\n"));
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer("Mary-Ann O'Neil"));
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer(" Alice J. Smith "));
}

test "Memorilla research answer guard rejects frame echo and malformed answers" {
    // The observed wide-challenge failure: the decoder echoed its memory frame.
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer("Memory key holder with the brass key for Zehisi Depot  \nMemory"));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer(""));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer(" \t\n"));
    const long: [65]u8 = @splat('a');
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer(&long));
    const max: [64]u8 = @splat('a');
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer(&max));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer("one two three four five six seven"));
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer("one two three four five six"));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer("Memory"));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer("Zehisi Memory"));
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer("Memoryville"));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer("Alice\nSmith"));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer("Alice\x01Smith"));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer("<|memory|>Alice"));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer("{Alice}"));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer("Alice, Smith"));
    // Lowercase "unknown" is a well-formed single word; only the exact literal
    // is treated as the abstention, the rest is ordinary accepted text.
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer("unknown"));
}

test "Memorilla research answer guard abstention replaces text and keeps ids" {
    const a = std.testing.allocator;
    var result = generation.GenerationResult{
        .text = try a.dupe(u8, "Memory key holder\nMemory"),
        .token_ids = try a.dupe(i32, &.{ 5, 6, 7 }),
        .tokens_used = 3,
        .finish_reason = "stop",
        .allocator = a,
    };
    defer result.deinit();
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer(result.text));
    try abstainGeneration(&result);
    try std.testing.expectEqualStrings("UNKNOWN", result.text);
    try std.testing.expectEqualStrings("abstained", result.finish_reason);
    try std.testing.expectEqual(@as(usize, 3), result.token_ids.?.len);
    try std.testing.expectEqual(@as(usize, 3), result.tokens_used);
}
