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

//! Hybrid research memory (docs/plans/memorilla-copy17-native.md sections
//! 8.4-8.7): the released reader's 16 soft rows (general recall) composed
//! with the copy17 candidate's 16-row exact record, read by the frozen
//! decoder, over ONE resident encoder. Production composition of
//! `scripts/memorilla/qwen_hybrid_eval.py --record-gate --answer-canonicalize`
//! (merged prompt, soft rows first, full record rows):
//!
//! 1. soft half: the stock `qwen3_encoder_f32_v1` encoder pools every document
//!    and the question (last token, L2, FP16 -> BF16 channel), and the
//!    released reader (`released_memory_module_path`, e.g. stage3) turns them
//!    into 16 rows;
//! 2. record half: the same encoder, with the hybrid bundle's encoder LoRA
//!    applied to its q/v projections at run time, produces the per-token
//!    `tokens-dual` features; the research reader, copy head, contiguous
//!    runs, snapping and word completion produce the 16 record ids, their
//!    spans and the source map;
//! 3. record gate (plan 7.4): open when one copied span snapped every slot at
//!    cosine >= 0.85. Open: system = helpful prompt + " " + the candidate's
//!    exact-answer prompt, 32 rows = soft then record embedding rows. Closed:
//!    the helpful prompt with the 16 soft rows alone;
//! 4. greedy decode, then (gate open) answer canonicalization against the
//!    record text, the frame-echo answer guard and citations of the record's
//!    copied spans evaluated on the published answer.
//!
//! Same lease, budget and fence discipline as `memorilla_research_service`.
//! Nothing here is a qualified release.
const std = @import("std");
const platform = @import("antfly_platform");
const tok = @import("inference_tokenizer");
const common = @import("memorilla_service.zig");
const research_service = @import("memorilla_research_service.zig");
const manager_mod = @import("model_manager.zig");
const selection = @import("../registry/model_selection.zig");
const qualified = @import("../models/memorilla_qualification.zig");
const encoder_lora = @import("../models/memorilla_encoder_lora.zig");
const encoder = @import("../pipelines/memorilla_encoder.zig");
const memory_pipeline = @import("../pipelines/memorilla.zig");
const token_features = @import("../pipelines/memorilla_token_features.zig");
const copy = @import("../pipelines/memorilla_copy.zig");
const grounding = @import("../pipelines/memorilla_grounding.zig");
const decode = @import("../pipelines/memorilla_generation.zig");
const generation = @import("../pipelines/generation.zig");
const factory = @import("../architectures/session_factory.zig");
const memory = @import("../runtime/root.zig").tier.memory;
const execution = @import("../ops/execution_scope.zig");
const Control = @import("../execution_control.zig").InferenceExecutionControl;
const PhysicalBudgetScope = if (@import("build_options").enable_cuda) @import("../ops/cuda/cuda_compute.zig").CudaCompute.MemorillaDeviceBudgetScope else struct {
    fn deinit(_: *@This()) void {}
};
const A = std.mem.Allocator;

pub const Profile = common.Profile;
/// `encoder` is the stock managed encoder (both passes), `decoder` the
/// managed decoder.
pub const Paths = common.Paths;
pub const Options = common.Options;
pub const Stage = common.Stage;
pub const encoder_profile: qualified.Profile = .qwen3_encoder_f32_v1;
pub const decoder_profile: qualified.Profile = .qwen3_decoder_f32_kv_f32_v1;

/// Rows per reader (the decoder's trained memory frame).
pub const slots: usize = 16;
/// Gate open: soft rows then record rows.
pub const hybrid_rows: usize = 2 * slots;
/// The released reader's prompt (upstream `HELPFUL_PROMPT`).
pub const helpful_prompt: []const u8 = "You are a helpful assistant.";
/// Plan 7.4: a copied span is trusted when every slot snapped at least this.
pub const record_gate_cosine: f32 = 0.85;
/// `canonicalize_answer`'s minimum squashed tail length.
pub const tail_min_chars: usize = 3;
/// Pooled vectors of documents longer than the stock encoder's pooled limit
/// come from sentence-greedy chunks of at most this many tokens (EOS
/// included).
pub const chunk_max_tokens: usize = 128;
/// Upper bound on the per-token record batch (rows x padded length). The
/// evaluator has none; depot cases use at most 219 positions and squad at
/// most 1525. This keeps one request's encoder activations bounded.
pub const max_record_positions: usize = 16384;

pub const RecordGate = enum { open, closed };
pub const AnswerGuard = research_service.AnswerGuard;
pub const Citation = copy.Citation;
pub const Citations = copy.Citations;
pub const unknown_answer = research_service.unknown_answer;

pub const OwnedResult = struct {
    /// Owns `record_text` and `raw_text`.
    allocator: A,
    generation: generation.GenerationResult,
    answer_guard: AnswerGuard,
    record_gate: RecordGate,
    /// The published text was re-spelled from the record.
    canonicalized: bool,
    /// Rows the decoder read: 32 (gate open) or 16 (closed).
    memory_tokens: usize,
    encoder_execution_identity: [64]u8,
    decoder_execution_identity: [64]u8,
    /// Research reader weights (record half).
    module_sha256: [64]u8,
    /// Released reader weights (soft half).
    soft_module_sha256: [64]u8,
    candidate_sha256: [64]u8,
    /// Run-time encoder LoRA applied to the record pass.
    adapter_sha256: [64]u8,
    document_count: usize,
    /// Pooled vectors read by the soft half (documents, or their chunks).
    soft_rows: usize,
    question_tokens: usize,
    /// Copied spans of the record with their document byte ranges; empty
    /// when the gate is closed.
    citations: copy.Citations,
    /// Diagnostics: the decoded record (`hybrid_record_text`), the trimmed
    /// decoder answer before canonicalization (`answer_raw`) and the smallest
    /// snap cosine over the record slots.
    record_text: []u8,
    raw_text: []u8,
    min_snap_cosine: f32,
    response_lease: memory.AdmissionLease,

    pub fn deinit(self: *OwnedResult) void {
        self.allocator.free(self.record_text);
        self.allocator.free(self.raw_text);
        self.generation.deinit();
        self.citations.deinit();
        self.response_lease.release();
        self.* = undefined;
    }
};

// --- policy helpers (pure, unit tested) ---------------------------------------

/// `qwen_hybrid_eval.system_prompt_text("merged")`: helpful then exact.
pub fn mergedSystemPrompt(a: A, exact_prompt: []const u8) ![]u8 {
    return std.mem.concat(a, u8, &.{ helpful_prompt, " ", exact_prompt });
}

/// `record_gate_open`: true when at least one copied span (record slots
/// `begin..end` before word completion) has every slot's top-1 snap cosine
/// at or above `threshold`. The evaluator compares the cosines it recorded,
/// rounded to four decimals; so does this.
pub fn recordGateOpen(spans: []const copy.Span, cosines: []const f32, threshold: f32) !bool {
    if (!(threshold >= 0 and threshold <= 1)) return error.InvalidRecordGate;
    for (spans) |span| {
        if (!(span.begin < span.end and span.end <= cosines.len)) return error.InvalidCopySpan;
        var lowest: f32 = std.math.inf(f32);
        for (cosines[span.begin..span.end]) |cosine| lowest = @min(lowest, round4(cosine));
        if (lowest >= threshold) return true;
    }
    return false;
}

fn round4(value: f32) f32 {
    const scaled: f64 = @as(f64, value) * 10000.0;
    return @floatCast(@round(scaled) / 10000.0);
}

/// Python `str.strip()` (Unicode whitespace), as the evaluator's `generate`
/// applies to the decoded answer. Invalid UTF-8 is returned unstripped.
pub fn pythonStrip(text: []const u8) []const u8 {
    const view = std.unicode.Utf8View.init(text) catch return text;
    var it = view.iterator();
    var offset: usize = 0;
    var first: ?usize = null;
    var last: usize = 0;
    while (it.nextCodepoint()) |cp| {
        const n = std.unicode.utf8CodepointSequenceLength(cp) catch unreachable;
        if (!qualified.pythonWhitespace(cp)) {
            if (first == null) first = offset;
            last = offset + n;
        }
        offset += n;
    }
    return if (first) |start| text[start..last] else text[text.len..];
}

fn isRecordWordByte(byte: u8) bool {
    return std.ascii.isAlphanumeric(byte) or byte == '-';
}

/// `squash`: lower case (ASCII), without hyphens or whitespace.
fn squash(a: A, text: []const u8) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(a);
    const view = std.unicode.Utf8View.init(text) catch {
        for (text) |byte| if (byte != '-' and !std.ascii.isWhitespace(byte)) try out.append(a, std.ascii.toLower(byte));
        return out.toOwnedSlice(a);
    };
    var it = view.iterator();
    while (it.nextCodepointSlice()) |piece| {
        const cp = std.unicode.utf8Decode(piece) catch unreachable;
        if (cp == '-' or qualified.pythonWhitespace(cp)) continue;
        if (piece.len == 1) try out.append(a, std.ascii.toLower(piece[0])) else try out.appendSlice(a, piece);
    }
    return out.toOwnedSlice(a);
}

fn codepoints(text: []const u8) usize {
    return std.unicode.utf8CountCodepoints(text) catch text.len;
}

/// `canonicalize_answer` (`qwen_hybrid_eval.py`, TAIL_MIN_CHARS = 3): re-spell a
/// decoded answer from the record it was read from.
///
/// (a) An answer occurring in `record_text` case-insensitively and aligned to
/// record word boundaries (letters, digits, `-`) takes the record's casing.
/// (b) Otherwise an answer that is the tail of one to three consecutive record
/// words (`[A-Za-z0-9][A-Za-z0-9-]*`, hyphens and whitespace ignored) becomes
/// that word sequence joined by single spaces; the squashed tail needs at
/// least three characters, and a tail starting inside a word is accepted only
/// for an answer carrying a digit. UNKNOWN and empty answers pass through.
/// The result is the stripped answer when nothing applies. Case folding is
/// ASCII-only (the evaluator uses Unicode `str.lower`).
pub fn canonicalizeAnswer(a: A, answer: []const u8, record_text: []const u8) ![]u8 {
    const raw = pythonStrip(answer);
    if (raw.len == 0 or std.mem.eql(u8, raw, unknown_answer)) return a.dupe(u8, raw);
    if (raw.len <= record_text.len) {
        var start: usize = 0;
        while (start + raw.len <= record_text.len) : (start += 1) {
            if (!std.ascii.eqlIgnoreCase(record_text[start..][0..raw.len], raw)) continue;
            const end = start + raw.len;
            const before_ok = start == 0 or !isRecordWordByte(record_text[start - 1]);
            const after_ok = end == record_text.len or !isRecordWordByte(record_text[end]);
            if (before_ok and after_ok) return a.dupe(u8, record_text[start..end]);
        }
    }
    const target = try squash(a, raw);
    defer a.free(target);
    if (codepoints(target) < tail_min_chars) return a.dupe(u8, raw);
    var words: std.ArrayListUnmanaged([]const u8) = .empty;
    defer words.deinit(a);
    var i: usize = 0;
    while (i < record_text.len) {
        if (!std.ascii.isAlphanumeric(record_text[i])) {
            i += 1;
            continue;
        }
        var j = i + 1;
        while (j < record_text.len and isRecordWordByte(record_text[j])) j += 1;
        try words.append(a, record_text[i..j]);
        i = j;
    }
    var has_digit = false;
    for (raw) |byte| has_digit = has_digit or std.ascii.isDigit(byte);
    var joined: std.ArrayListUnmanaged(u8) = .empty;
    defer joined.deinit(a);
    for ([_]usize{ 1, 2, 3 }) |size| {
        if (words.items.len < size) break;
        for (0..words.items.len - size + 1) |index| {
            const sequence = words.items[index..][0..size];
            // Squashed words concatenate: word k starts at the summed lengths
            // of the squashed words before it.
            joined.clearRetainingCapacity();
            var starts: [3]usize = undefined;
            for (sequence, 0..) |word, k| {
                starts[k] = joined.items.len;
                for (word) |byte| if (byte != '-') try joined.append(a, std.ascii.toLower(byte));
            }
            if (!std.mem.endsWith(u8, joined.items, target)) continue;
            const tail_start = joined.items.len - target.len;
            const aligned = std.mem.indexOfScalar(usize, starts[0..size], tail_start) != null;
            if (has_digit or aligned) return std.mem.join(a, " ", sequence);
        }
    }
    return a.dupe(u8, raw);
}

/// Hybrid answer guard. General answers can be sentences, so only the
/// frame-echo rules of the research guard (plan 7.1) apply, to the
/// whitespace-trimmed text: the literal UNKNOWN is accepted; empty text, any
/// control character (including newline and tab), the `<|memory|>` marker or
/// the frame literal `Memory` as a standalone word is rejected. There is no
/// length, word-count or character-set limit.
pub fn guardAnswer(text: []const u8) AnswerGuard {
    const trimmed = std.mem.trim(u8, text, &std.ascii.whitespace);
    if (std.mem.eql(u8, trimmed, unknown_answer)) return .accepted;
    if (trimmed.len == 0) return .rejected;
    if (std.mem.indexOf(u8, trimmed, decode.marker) != null) return .rejected;
    for (trimmed) |byte| if (std.ascii.isControl(byte)) return .rejected;
    var it = std.mem.tokenizeAny(u8, trimmed, &std.ascii.whitespace);
    while (it.next()) |word| if (std.mem.eql(u8, word, "Memory")) return .rejected;
    return .accepted;
}

/// Appends `piece` decoded leniently: valid UTF-8 is copied, an invalid or
/// truncated sequence becomes one U+FFFD (Python `bytes.decode(errors="replace")`
/// on a token piece).
fn appendLossy(a: A, out: *std.ArrayListUnmanaged(u8), piece: []const u8) !void {
    var i: usize = 0;
    while (i < piece.len) {
        const length = std.unicode.utf8ByteSequenceLength(piece[i]) catch {
            try out.appendSlice(a, "\u{fffd}");
            i += 1;
            continue;
        };
        var continued: usize = 0;
        while (continued + 1 < length and i + 1 + continued < piece.len and piece[i + 1 + continued] & 0xc0 == 0x80) continued += 1;
        if (continued + 1 == length and std.unicode.utf8ValidateSlice(piece[i..][0..length])) {
            try out.appendSlice(a, piece[i..][0..length]);
        } else {
            try out.appendSlice(a, "\u{fffd}");
        }
        i += 1 + continued;
    }
}

/// `"".join(dec_tok.decode([i]) for i in ids)`: the record text the
/// evaluator canonicalizes against, decoded token by token.
pub fn recordText(a: A, tokenizer: tok.Tokenizer, ids: []const i32) ![]u8 {
    var out: std.ArrayListUnmanaged(u8) = .empty;
    errdefer out.deinit(a);
    for (ids) |id| {
        const piece = try tokenizer.decode(a, &.{id});
        defer a.free(piece);
        try appendLossy(a, &out, piece);
    }
    return out.toOwnedSlice(a);
}

// --- pooled chunking ----------------------------------------------------------

const Range = struct { start: usize, end: usize };

fn tokenCount(a: A, tokenizer: tok.Tokenizer, text: []const u8) !usize {
    const ids = try tokenizer.encode(a, text);
    defer a.free(ids);
    // The pooled pass appends EOS.
    return ids.len + 1;
}

fn trimRange(text: []const u8, range: Range) ?Range {
    var start = range.start;
    var end = range.end;
    while (start < end and std.ascii.isWhitespace(text[start])) start += 1;
    while (end > start and std.ascii.isWhitespace(text[end - 1])) end -= 1;
    return if (start < end) .{ .start = start, .end = end } else null;
}

/// Greedily merges consecutive `units` (each fitting alone) into the longest
/// runs whose covering text fits `max_tokens`.
fn packRanges(a: A, tokenizer: tok.Tokenizer, text: []const u8, units: []const Range, max_tokens: usize, out: *std.ArrayListUnmanaged(Range)) !void {
    var current: ?Range = null;
    for (units) |unit| {
        if (current) |held| {
            const candidate = Range{ .start = held.start, .end = unit.end };
            if (try tokenCount(a, tokenizer, text[candidate.start..candidate.end]) <= max_tokens) {
                current = candidate;
                continue;
            }
            try out.append(a, held);
        }
        current = unit;
    }
    if (current) |held| try out.append(a, held);
}

/// Splits one oversized sentence into word groups, and an oversized word into
/// pieces of at most `max_tokens - 1` bytes on UTF-8 boundaries (a byte-level
/// BPE token covers at least one byte).
fn splitUnit(a: A, tokenizer: tok.Tokenizer, text: []const u8, unit: Range, max_tokens: usize, out: *std.ArrayListUnmanaged(Range)) !void {
    var words: std.ArrayListUnmanaged(Range) = .empty;
    defer words.deinit(a);
    var i = unit.start;
    while (i < unit.end) {
        while (i < unit.end and std.ascii.isWhitespace(text[i])) i += 1;
        if (i == unit.end) break;
        var j = i;
        while (j < unit.end and !std.ascii.isWhitespace(text[j])) j += 1;
        if (try tokenCount(a, tokenizer, text[i..j]) <= max_tokens) {
            try words.append(a, .{ .start = i, .end = j });
        } else {
            var piece_start = i;
            while (piece_start < j) {
                var piece_end = @min(j, piece_start + max_tokens - 1);
                while (piece_end > piece_start + 1 and piece_end < j and text[piece_end] & 0xc0 == 0x80) piece_end -= 1;
                try words.append(a, .{ .start = piece_start, .end = piece_end });
                piece_start = piece_end;
            }
        }
        i = j;
    }
    try packRanges(a, tokenizer, text, words.items, max_tokens, out);
}

/// Sentence-greedy chunks of `text` whose pooled encodings (EOS included) fit
/// `max_tokens`: sentences end after `.`, `!` or `?` followed by whitespace,
/// or at a newline; consecutive sentences are merged while they fit, an
/// oversized sentence is split at whitespace (and an oversized word at
/// UTF-8 boundaries). Chunks are trimmed slices of `text`, in order.
pub fn chunkDocument(a: A, tokenizer: tok.Tokenizer, text: []const u8, max_tokens: usize) ![]Range {
    if (max_tokens < 2) return error.MemorillaTextLimitExceeded;
    var units: std.ArrayListUnmanaged(Range) = .empty;
    defer units.deinit(a);
    var start: usize = 0;
    for (text, 0..) |byte, i| {
        const boundary = byte == '\n' or ((byte == '.' or byte == '!' or byte == '?') and (i + 1 == text.len or std.ascii.isWhitespace(text[i + 1])));
        if (!boundary) continue;
        if (trimRange(text, .{ .start = start, .end = i + 1 })) |sentence| {
            if (try tokenCount(a, tokenizer, text[sentence.start..sentence.end]) <= max_tokens) try units.append(a, sentence) else try splitUnit(a, tokenizer, text, sentence, max_tokens, &units);
        }
        start = i + 1;
    }
    if (trimRange(text, .{ .start = start, .end = text.len })) |sentence| {
        if (try tokenCount(a, tokenizer, text[sentence.start..sentence.end]) <= max_tokens) try units.append(a, sentence) else try splitUnit(a, tokenizer, text, sentence, max_tokens, &units);
    }
    var chunks: std.ArrayListUnmanaged(Range) = .empty;
    errdefer chunks.deinit(a);
    try packRanges(a, tokenizer, text, units.items, max_tokens, &chunks);
    return chunks.toOwnedSlice(a);
}

// --- validation and preflight -------------------------------------------------

/// Pre-admission validation shared with the HTTP handler. Only policy and
/// shape; no model, reader or GPU state is consulted. The identity is the
/// stock encoder's: both passes run on it.
pub fn validateInputs(profile: Profile, question: []const u8, documents: []const []const u8, identity: []const u8, max_new_tokens: usize) !void {
    try (platform.inference_settings.Memorilla{ .enabled = true, .profiles = &.{profile} }).validate();
    if (profile.reader_kind != .hybrid_research or profile.released_memory_module_path == null) return error.UnsupportedMemorillaProfile;
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

/// Elementwise maximum: one admission that covers either decoder plan.
pub fn maxAmounts(x: memory.AdmissionAmounts, y: memory.AdmissionAmounts) memory.AdmissionAmounts {
    var out: memory.AdmissionAmounts = .{};
    inline for (@typeInfo(memory.AdmissionAmounts).@"struct".field_names) |name| @field(out, name) = @max(@field(x, name), @field(y, name));
    return out;
}

/// Encoder-side facts computed with the stock encoder's tokenizer before any
/// model loads, so limits fail early and admission is exact.
const Preflight = struct {
    a: A,
    question_tokens: usize,
    /// Pooled texts (slices of the caller's documents): one per document, or
    /// its chunks when it exceeds the pooled limit.
    soft_texts: [][]const u8,
    /// Selected per-token rows of the record pass (`2 * sum(n_i + 1)`).
    record_rows: usize,

    fn deinit(self: *Preflight) void {
        self.a.free(self.soft_texts);
    }
};

fn preflightEncoder(a: A, io: std.Io, manager: *manager_mod.ModelManager, selected: selection.ResolvedModel, canonical: []const u8, documents: []const []const u8, control: Control) !Preflight {
    try control.check();
    var assets = try common.acquireGenerationAssets(a, io, manager, selected, encoder_profile, control);
    defer assets.deinit();
    const tokenizer = assets.tokenizer();
    // The pooled question, exactly as the released preflight counts it.
    const question_tokens = blk: {
        var tokens = try tokenizer.encodeForModel(a, canonical, qualified.max_text_tokens + 1);
        defer tokens.deinit();
        if (tokens.ids.len != tokens.attention_mask.len) return error.InvalidMemorillaTokens;
        var active: usize = 0;
        var last: i32 = -1;
        for (tokens.ids, tokens.attention_mask) |id, used| {
            if (used != 0 and used != 1) return error.InvalidMemorillaTokens;
            if (used == 1) {
                active += 1;
                last = id;
            }
        }
        if (active > qualified.max_text_tokens) return error.MemorillaTextLimitExceeded;
        if (active == 0 or last != token_features.eos_id) return error.InvalidMemorillaTokens;
        break :blk active;
    };
    var soft: std.ArrayListUnmanaged([]const u8) = .empty;
    errdefer soft.deinit(a);
    for (documents) |document| {
        try control.check();
        if (try tokenCount(a, tokenizer, document) <= qualified.max_text_tokens) {
            try soft.append(a, document);
            continue;
        }
        const chunks = try chunkDocument(a, tokenizer, document, chunk_max_tokens);
        defer a.free(chunks);
        for (chunks) |chunk| try soft.append(a, document[chunk.start..chunk.end]);
    }
    if (soft.items.len == 0) return error.EmptyMemorillaCollection;
    if (soft.items.len > qualified.max_documents) return error.MemorillaCollectionBudgetExceeded;
    // The record batch: bounds, rows and the copy-source map are recomputed
    // by the encoder pass and must agree.
    var batch = try token_features.prepareBounded(a, tokenizer, documents, canonical, token_features.hybrid_max_padded_length);
    defer batch.deinit();
    if (try mul(batch.batch, batch.seq_len) > max_record_positions) return error.MemorillaEncoderBoundExceeded;
    const record_rows = try mul(try token_features.selectedRowCount(batch.counts), 2);
    try selected.verify(a, io);
    try control.check();
    return .{ .a = a, .question_tokens = question_tokens, .soft_texts = try soft.toOwnedSlice(a), .record_rows = record_rows };
}

// --- service --------------------------------------------------------------------

pub fn generate(a: A, io: std.Io, manager: *manager_mod.ModelManager, profile: Profile, paths: Paths, question: []const u8, documents: []const []const u8, identity: []const u8, max_new_tokens: usize, limits: memory.Limits, request_limits: platform.resource_policy.InferenceRequestLimits, control: Control, options_in: Options) !OwnedResult {
    var options = options_in;
    options.request_limits = request_limits;
    var result = try generateImpl(a, io, manager, profile, paths, question, documents, identity, max_new_tokens, limits, control, options);
    errdefer result.deinit();
    try control.check();
    return result;
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
    const released_directory = profile.released_memory_module_path orelse return error.UnsupportedMemorillaProfile;

    var response_lease = try manager.acquireRunResourceAmounts(.cpu, limits, .{ .host_scratch_bytes = 32 * 1024 * 1024 });
    var transfer_response = false;
    defer if (!transfer_response) response_lease.release();
    const backing = platform.allocator.reclaimingScratch(a);

    // Candidate pins read before any model loads: the run-time adapters and
    // the exact-answer prompt of the hybrid bundle.
    var adapter = try encoder_lora.loadBundle(backing, profile.memory_module_path, encoder_lora.qwen3_embedding_4b);
    defer adapter.deinit();
    const exact_prompt = try research_service.bundleSystemPrompt(backing, profile.memory_module_path);
    defer backing.free(exact_prompt);
    const merged_prompt = try mergedSystemPrompt(backing, exact_prompt);
    defer backing.free(merged_prompt);

    var preflight = try preflightEncoder(backing, io, manager, paths.encoder, canonical, documents, encoder_control);
    defer preflight.deinit();
    try common.validateInputTokenLimit(options.request_limits, preflight.question_tokens);

    var assets = try common.acquireGenerationAssets(backing, io, manager, paths.decoder, decoder_profile, decoder_control);
    defer assets.deinit();
    const config = try assets.decoderConfig(backing, paths.decoder.path);
    const bounded = decode.Limits{ .max_new_tokens = profile.max_new_tokens };
    const template = assets.chatTemplate() orelse return error.MemorillaChatTemplateRequired;
    // Both prompts are prepared up front; the gate picks one after the record.
    var open_text = try decode.prepareText(backing, assets.tokenizer(), template, config, canonical, .{ .system_prompt = merged_prompt, .memory_rows = hybrid_rows }, bounded, decoder_control);
    defer open_text.deinit();
    var closed_text = try decode.prepareText(backing, assets.tokenizer(), template, config, canonical, .{ .system_prompt = helpful_prompt, .memory_rows = slots }, bounded, decoder_control);
    defer closed_text.deinit();
    try common.validateInputTokenLimit(options.request_limits, @max(open_text.input_ids.len, closed_text.input_ids.len));
    if (options.prepared_text) |callback| try callback(options.context, &open_text);
    const open_plan = try decode.ResourcePlan.create(config, .cuda, open_text.input_ids.len, open_text.rendered.len, hybrid_rows, max_new_tokens, bounded);
    const closed_plan = try decode.ResourcePlan.create(config, .cuda, closed_text.input_ids.len, closed_text.rendered.len, slots, max_new_tokens, bounded);

    // Admission: the decoder plan covering either prompt plus the copy/snap
    // transients, the released reader over the pooled rows and the research
    // reader over the per-token rows (plus the uploaded 32 memory rows).
    const soft_request = memory.AdmissionAmounts{ .host_scratch_bytes = 16 * 1024 * 1024, .backend_scratch_bytes = try add(128 * 1024 * 1024, try mul(qualified.max_documents, qualified.width * 16 * @sizeOf(f32))) };
    const record_request = memory.AdmissionAmounts{
        .host_scratch_bytes = try add(16 * 1024 * 1024, try mul(preflight.record_rows, qualified.width * @sizeOf(f32))),
        .backend_scratch_bytes = try add(try add(128 * 1024 * 1024, try mul(preflight.record_rows, research_service.reader_scratch_per_row)), try mul(hybrid_rows, config.hidden_size * @sizeOf(f32))),
    };
    const copy_request = memory.AdmissionAmounts{ .backend_scratch_bytes = research_service.decoder_copy_scratch_bytes };
    const open_decoder = try (try memory.AdmissionAmounts.fromEstimate(open_plan.decoder).merge(.fromEstimate(open_plan.host))).merge(copy_request);
    const closed_decoder = try (try memory.AdmissionAmounts.fromEstimate(closed_plan.decoder).merge(.fromEstimate(closed_plan.host))).merge(copy_request);
    const decoder_request = maxAmounts(open_decoder, closed_decoder);
    const request_amounts = try (try decoder_request.merge(soft_request)).merge(record_request);
    var request_lease = try manager.acquireRunResourceAmounts(.gpu, limits, request_amounts);
    defer request_lease.release();
    var decoder_budget = memory.RunBudget.init(try common.requestBudgetLimits(limits, decoder_request));
    var soft_budget = memory.RunBudget.init(try common.requestBudgetLimits(limits, soft_request));
    var record_budget = memory.RunBudget.init(try common.requestBudgetLimits(limits, record_request));
    // Reserve the 32-row plan; a closed gate swaps it for the 16-row plan in
    // place (the reservation's address identifies the decoder owner).
    var reservation = try open_plan.reserve(&decoder_budget);
    defer reservation.deinit();
    const copy_estimate = memory.Estimate{ .prompt_tokens = 0, .retained_tokens = 0, .kv_bytes = 0, .kv_tier = .backend, .scratch_bytes = research_service.decoder_copy_scratch_bytes, .scratch_tier = .backend };
    try decoder_budget.reserveEstimate(copy_estimate);
    defer decoder_budget.releaseEstimate(copy_estimate);
    inline for (.{ .{ &soft_budget, soft_request }, .{ &record_budget, record_request } }) |entry| {
        const backend_estimate = memory.Estimate{ .prompt_tokens = 0, .retained_tokens = 0, .kv_bytes = 0, .kv_tier = .backend, .scratch_bytes = entry[1].backend_scratch_bytes, .scratch_tier = .backend };
        try entry[0].reserveEstimate(backend_estimate);
        const host_estimate = memory.Estimate{ .prompt_tokens = 0, .retained_tokens = 0, .kv_bytes = 0, .kv_tier = .host, .scratch_bytes = entry[1].host_scratch_bytes, .scratch_tier = .host };
        entry[0].reserveEstimate(host_estimate) catch |err| {
            entry[0].releaseEstimate(backend_estimate);
            return err;
        };
    }
    defer inline for (.{ .{ &soft_budget, soft_request }, .{ &record_budget, record_request } }) |entry| {
        entry[0].releaseEstimate(.{ .prompt_tokens = 0, .retained_tokens = 0, .kv_bytes = 0, .kv_tier = .host, .scratch_bytes = entry[1].host_scratch_bytes, .scratch_tier = .host });
        entry[0].releaseEstimate(.{ .prompt_tokens = 0, .retained_tokens = 0, .kv_bytes = 0, .kv_tier = .backend, .scratch_bytes = entry[1].backend_scratch_bytes, .scratch_tier = .backend });
    };
    try notify(options, .prepared, manager);

    // One resident encoder serves both passes.
    var encoder_handle = try common.acquireResidentModel(manager, paths.encoder.path, encoder_control);
    defer encoder_handle.release();
    const encoder_model = encoder_handle.get();
    try encoder_model.verifyMemorillaIdentityWithControl(encoder_control);
    try notify(options, .encoder_loaded, manager);
    // Soft pass: pooled vectors without adapters.
    var pooled = try encoder.encode(backing, &encoder_handle, preflight.soft_texts, canonical, limits, encoder_control);
    defer pooled.deinit();
    if (!std.mem.eql(u8, pooled.question_text, canonical)) return error.NonCanonicalMemorillaQuestion;
    try encoder.requireProducer(pooled.producer, paths.encoder.publication.?, encoder_profile);
    if (pooled.document_count != preflight.soft_texts.len or pooled.question_tokens != preflight.question_tokens) return error.InvalidMemorillaTokens;
    // Record pass: per-token features with the pinned adapters at run time.
    var record_features = try encoder.encodeHybridTokenFeatures(backing, &encoder_handle, documents, canonical, &adapter, limits, encoder_control);
    defer record_features.deinit();
    try encoder.requireProducer(record_features.producer, paths.encoder.publication.?, encoder_profile);
    if (record_features.adapter_sha256 == null or !std.mem.eql(u8, &record_features.adapter_sha256.?, &adapter.sha256)) return error.MemorillaCandidateMismatch;
    if (record_features.features.rows != preflight.record_rows) return error.MemorillaCollectionBudgetExceeded;
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

    // Soft half: the released reader, borrowed and retired before the record
    // half (the two reader caches are never held together).
    const soft_rows, const soft_module_sha256 = blk: {
        var soft_handle = try decoder_model.memorilla_reader_cache.acquire(decoder_model.allocator, io, manager, released_directory, limits, control);
        defer soft_handle.deinit();
        var soft_clean = false;
        defer if (!soft_clean) soft_handle.invalidate();
        const reader = soft_handle.owner();
        try reader.bundle.config.validate(true);
        if (!reader.bundle.release or reader.bundle.config.num_memories != slots or reader.bundle.config.output_dim != config.hidden_size) return error.UnsupportedMemorillaProfile;
        const rows = pooled.document_count;
        const mask = try backing.alloc(bool, rows);
        defer backing.free(mask);
        @memset(mask, false);
        var soft_backend = try reader.computeBackendWithBudget(&soft_budget);
        soft_backend.execution_control = control;
        soft_backend.execution_budget = &soft_budget;
        var soft_scope = try execution.Scope.acquire(control, .{ .mutex = &reader.execution_mutex });
        defer soft_scope.deinit();
        var soft_physical: PhysicalBudgetScope = if (comptime @import("build_options").enable_cuda)
            try reader.compute.beginMemorillaDeviceBudget(soft_request.backendTotalBytes())
        else
            return error.CudaNotEnabled;
        defer soft_physical.deinit();
        try soft_scope.bind(soft_backend);
        const values = inner: {
            defer {
                reader.releaseRequestResources() catch @panic("Memorilla released reader request cleanup failed; worker restart required");
                soft_clean = true;
            }
            const docs = try soft_backend.fromFloat32Shape(pooled.documents, &.{ @intCast(rows), @intCast(qualified.width) });
            defer soft_backend.free(docs);
            const query = try soft_backend.fromFloat32Shape(pooled.question, &.{ 1, @intCast(qualified.width) });
            defer soft_backend.free(query);
            try encoder_model.verifyMemorillaIdentityWithControl(encoder_control);
            try decoder_model.verifyMemorillaIdentityWithControl(decoder_control);
            try factory.requireMemorillaCudaExecution(&soft_backend);
            const memories = try memory_pipeline.forward(&soft_backend, backing, reader.bundle.config, .{ .documents = docs, .question = query, .padding_mask = mask, .batch = 1, .num_documents = rows, .max_document_rows = qualified.max_documents }, null);
            defer soft_backend.free(memories);
            try factory.requireMemorillaCudaExecution(&soft_backend);
            break :inner try soft_backend.toFloat32(memories, backing);
        };
        errdefer backing.free(values);
        if (values.len != slots * config.hidden_size) return error.InvalidMemorillaMemoryRows;
        try soft_handle.verify(control);
        break :blk .{ values, reader.bundle.weights_sha256 };
    };
    defer backing.free(soft_rows);
    try control.check();

    // Record half: the research reader of the hybrid bundle.
    var reader_handle = try decoder_model.memorilla_research_reader_cache.acquire(decoder_model.allocator, io, manager, profile.memory_module_path, limits, control);
    defer reader_handle.deinit();
    var reader_request_clean = false;
    defer if (!reader_request_clean) reader_handle.invalidate();
    const reader = reader_handle.owner();
    if (reader.bundle.release or reader.bundle.config.num_memories != slots or reader.bundle.config.output_dim != config.hidden_size) return error.UnsupportedMemorillaProfile;
    if (!std.mem.eql(u8, reader.system_prompt, exact_prompt)) return error.ModelArtifactsChanging;
    // One candidate: the adapters were verified against this very manifest,
    // and the reader names the same candidate.json.
    if (!std.mem.eql(u8, &reader.bundle.manifest_sha256, &adapter.manifest_sha256)) return error.ModelArtifactsChanging;
    if (!std.mem.eql(u8, &reader.candidate_sha256, &adapter.candidate_sha256)) return error.MemorillaCandidateMismatch;
    try notify(options, .reader_loaded, manager);
    const module_sha256 = reader.bundle.weights_sha256;
    const candidate_sha256 = reader.candidate_sha256;

    var reader_backend = try reader.computeBackendWithBudget(&record_budget);
    reader_backend.execution_control = control;
    reader_backend.execution_budget = &record_budget;
    var reader_scope = try execution.Scope.acquire(control, .{ .mutex = &reader.execution_mutex });
    defer reader_scope.deinit();
    var reader_physical: PhysicalBudgetScope = if (comptime @import("build_options").enable_cuda)
        try reader.compute.beginMemorillaDeviceBudget(record_request.backendTotalBytes())
    else
        return error.CudaNotEnabled;
    defer reader_physical.deinit();
    try reader_scope.bind(reader_backend);
    var completed_spans: []copy.CompletedSpan = &.{};
    defer backing.free(completed_spans);
    var gate: RecordGate = .closed;
    var record_text: []u8 = &.{};
    defer backing.free(record_text);
    var min_snap_cosine: f32 = 0;
    const output = blk: {
        defer {
            reader.releaseRequestResources() catch @panic("Memorilla research reader request cleanup failed; worker restart required");
            reader_request_clean = true;
        }
        const rows = record_features.features.rows;
        const mask = try backing.alloc(bool, rows);
        defer backing.free(mask);
        @memset(mask, false);
        try encoder_model.verifyMemorillaIdentityWithControl(encoder_control);
        try decoder_model.verifyMemorillaIdentityWithControl(decoder_control);
        try factory.requireMemorillaCudaExecution(&reader_backend);
        const memories = blk2: {
            const docs = try reader_backend.fromFloat32Shape(record_features.features.docs, &.{ @intCast(rows), @intCast(qualified.width) });
            defer reader_backend.free(docs);
            const query = try reader_backend.fromFloat32Shape(record_features.features.query, &.{ 1, @intCast(qualified.width) });
            defer reader_backend.free(query);
            const memories_ct = try memory_pipeline.forward(&reader_backend, backing, reader.bundle.config, .{ .documents = docs, .question = query, .padding_mask = mask, .batch = 1, .num_documents = rows, .max_document_rows = preflight.record_rows }, null);
            defer reader_backend.free(memories_ct);
            break :blk2 try reader_backend.toFloat32(memories_ct, backing);
        };
        defer backing.free(memories);
        try factory.requireMemorillaCudaExecution(&reader_backend);
        try control.check();
        var decoder_backend = try factory.getCoordinatedCudaComputeBackend(decoder_model.session, &decoder_budget, @intFromPtr(&reservation));
        defer decoder_backend.deinit();
        var snapped = blk3: {
            decoder_backend.execution_control = decoder_control;
            decoder_backend.execution_budget = &decoder_budget;
            var decoder_scope = try execution.Scope.acquire(control, .{ .mutex = decoder_model.nativeGenerationMutex(), .lane = &decoder_model.native_reader_execution_lane });
            defer decoder_scope.deinit();
            try decoder_scope.bind(decoder_backend);
            try factory.requireMemorillaCudaExecution(&decoder_backend);
            const result = try research_service.copyAndSnap(backing, reader, &decoder_backend, config, assets.tokenizer(), memories, record_features.features.copy_ids, control);
            errdefer {
                var owned = result;
                owned.deinit();
            }
            try factory.requireMemorillaCudaExecution(&decoder_backend);
            break :blk3 result;
        };
        defer snapped.deinit();
        completed_spans = snapped.spans;
        snapped.spans = &.{};
        min_snap_cosine = std.math.inf(f32);
        for (snapped.cosines) |cosine| min_snap_cosine = @min(min_snap_cosine, cosine);
        gate = if (try recordGateOpen(snapped.copy_spans, snapped.cosines, record_gate_cosine)) .open else .closed;
        record_text = try recordText(backing, assets.tokenizer(), snapped.completed_ids);
        try control.check();
        // Compose the memory rows on the host: soft rows first, then (gate
        // open) the record's embedding rows; one F32 upload for injection.
        const memory_rows: usize = if (gate == .open) hybrid_rows else slots;
        const composed = try backing.alloc(f32, memory_rows * config.hidden_size);
        defer backing.free(composed);
        @memcpy(composed[0 .. slots * config.hidden_size], soft_rows);
        if (gate == .open) @memcpy(composed[slots * config.hidden_size ..], snapped.rows);
        if (gate == .closed) {
            reservation.deinit();
            reservation = try closed_plan.reserve(&decoder_budget);
        }
        const text = if (gate == .open) &open_text else &closed_text;
        const rows_ct = try reader_backend.fromFloat32Shape(composed, &.{ @intCast(memory_rows), @intCast(config.hidden_size) });
        defer reader_backend.free(rows_ct);
        var decoder = try decode.Decoder.fromQualifiedSession(decoder_model.session, &decoder_backend, assets.tokenizer(), decoder_model.nativeGenerationMutex());
        decoder.execution_lane = &decoder_model.native_reader_execution_lane;
        break :blk try decode.generate(a, decoder, text, .{ .backend = &reader_backend, .tensor = rows_ct }, .{ .max_new_tokens = max_new_tokens, .teacher_forced_ids = options.teacher_forced_ids, .trace = options.trace, .ignore_eos = options.ignore_eos }, &reservation, bounded, decoder_control);
    };
    var result = output;
    errdefer result.deinit();

    // Publish: the stripped decoder answer, re-spelled from the record when
    // the decoder read it, then the frame-echo guard and the citations.
    const raw_text = try a.dupe(u8, pythonStrip(result.text));
    errdefer a.free(raw_text);
    const published = if (gate == .open)
        try canonicalizeAnswer(result.allocator, raw_text, record_text)
    else
        try result.allocator.dupe(u8, raw_text);
    result.allocator.free(result.text);
    result.text = published;
    const answer_guard = guardAnswer(result.text);
    if (answer_guard == .rejected) try research_service.abstainGeneration(&result);
    const canonicalized = answer_guard == .accepted and !std.mem.eql(u8, result.text, raw_text);
    var citations = if (gate == .open)
        try copy.buildCitations(a, completed_spans, record_features.features.sources, documents, research_service.citedAnswer(result.text, answer_guard))
    else
        copy.Citations.none(a, false);
    errdefer citations.deinit();
    const owned_record_text = try a.dupe(u8, record_text);
    errdefer a.free(owned_record_text);
    if (comptime @import("build_options").enable_cuda) {
        if (options.reader_physical_resources) |record| {
            const snapshot = reader.compute.ctx.device_allocations.snapshot();
            try record(options.context, .{ .baseline = reader_physical.baseline, .held_transient_bytes = record_request.backendTotalBytes(), .limit = snapshot.limit, .live = snapshot.live, .lifetime_peak = snapshot.peak });
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
    if (record_features.question_tokens == 0) return error.InvalidMemorillaTokens;
    transfer_response = true;
    return .{
        .allocator = a,
        .generation = result,
        .answer_guard = answer_guard,
        .record_gate = gate,
        .canonicalized = canonicalized,
        .memory_tokens = if (gate == .open) hybrid_rows else slots,
        .encoder_execution_identity = qualified.executionIdentity(encoder_profile),
        .decoder_execution_identity = qualified.executionIdentity(decoder_profile),
        .module_sha256 = module_sha256,
        .soft_module_sha256 = soft_module_sha256,
        .candidate_sha256 = candidate_sha256,
        .adapter_sha256 = adapter.sha256,
        .document_count = documents.len,
        .soft_rows = pooled.document_count,
        .question_tokens = preflight.question_tokens,
        .citations = citations,
        .record_text = owned_record_text,
        .raw_text = raw_text,
        .min_snap_cosine = min_snap_cosine,
        .response_lease = response_lease,
    };
}

// --- tests ----------------------------------------------------------------------

test "Memorilla hybrid canonicalization mirrors the evaluator cases" {
    const a = std.testing.allocator;
    const cases = [_][3][]const u8{
        // (a) case flips take the record's casing.
        .{ "WZ3-003545", " code Wz3-003545 at Dociru Depot", "Wz3-003545" },
        .{ "hidehi hokifu", " key holder Hidehi Hokifu at Kopene Depot ", "Hidehi Hokifu" },
        // (b) prefix drops grow back to the record word (hyphen ignored).
        .{ "3005601", " code Te3-005601 at Zorutu ", "Te3-005601" },
        .{ "4672", " code Xd4-004672 at Kuruli Depot ", "Xd4-004672" },
        .{ "001618", " code Pp7-001618 at Soceso Depot ", "Pp7-001618" },
        // A whole-word match is left as the record spells it, never extended.
        .{ "Hokifu", " key holder Hidehi Hokifu at Kopene Depot ", "Hokifu" },
        // Bare digits never grow into a record word (three-character tail);
        // the shortest real prefix drop still completes.
        .{ "4", " code holder4 10 and 50 centavos ", "4" },
        .{ "0", " code UNKNOWN at 12-10 with under 2 ", "0" },
        .{ "10", " code UNKNOWN at 12-10 with under 2 ", "10" },
        .{ "672", " code Xd4-004672 at Kuruli Depot ", "Xd4-004672" },
        // Multi-word tails complete to the word sequence.
        .{ "Fihunu at Kopene", " key holder Faketa Fihunu at Kopene Depot ", "Fihunu at Kopene" },
        .{ "Tuloco", " key holder Komode Tuloco ", "Tuloco" },
        // Free-form words never grow into longer words.
        .{ "yes", " code holder Hayes at 1 or 0 ", "yes" },
        .{ "France", " key holder region in France ", "France" },
        // No relation to the record: unchanged (whitespace stripped).
        .{ "JESAVO DEGEBE", " key holder Jesavo Degebo at Gihera ", "JESAVO DEGEBE" },
        .{ " Rome ", " code holder at 1060s ", "Rome" },
        .{ "UNKNOWN", " code Te3-005601 ", "UNKNOWN" },
        .{ "", " code Te3-005601 ", "" },
        .{ "-", " code Te3-005601 ", "-" },
    };
    for (cases) |case| {
        const got = try canonicalizeAnswer(a, case[0], case[1]);
        defer a.free(got);
        std.testing.expectEqualStrings(case[2], got) catch |err| {
            std.debug.print("canonicalize({s}, {s})\n", .{ case[0], case[1] });
            return err;
        };
    }
}

test "Memorilla hybrid canonicalization handles multi-word tails and Unicode whitespace" {
    const a = std.testing.allocator;
    // A digit-carrying tail may start inside a word and spans up to three words.
    const coded = try canonicalizeAnswer(a, "9-009369 at Pipofo", " code Kg9-009369 at Pipofo Depot ");
    defer a.free(coded);
    try std.testing.expectEqualStrings("Kg9-009369 at Pipofo", coded);
    // Without a digit the tail must start at a word start.
    const named = try canonicalizeAnswer(a, "ode Tuloco", " key holder Komode Tuloco ");
    defer a.free(named);
    try std.testing.expectEqualStrings("ode Tuloco", named);
    // Python strip removes Unicode whitespace around the answer.
    const spaced = try canonicalizeAnswer(a, "\u{a0}Kg9-009369\u{3000}", " code Kg9-009369 ");
    defer a.free(spaced);
    try std.testing.expectEqualStrings("Kg9-009369", spaced);
    try std.testing.expectEqualStrings("x y", pythonStrip("\u{2003}x y\u{85}"));
    try std.testing.expectEqualStrings("", pythonStrip(" \t\n"));
}

test "Memorilla hybrid record gate trusts a clean copied span" {
    var cosines: [16]f32 = @splat(1.0);
    // No copied span: the soft rows go alone.
    try std.testing.expect(!try recordGateOpen(&.{}, &cosines, record_gate_cosine));
    // A clean span opens the gate.
    try std.testing.expect(try recordGateOpen(&.{.{ .begin = 2, .end = 6, .start = 0 }}, &cosines, record_gate_cosine));
    // A fence-sitting slot inside the only span closes it.
    var fence = cosines;
    fence[4] = 0.7;
    try std.testing.expect(!try recordGateOpen(&.{.{ .begin = 2, .end = 6, .start = 0 }}, &fence, record_gate_cosine));
    // Another clean span is enough.
    try std.testing.expect(try recordGateOpen(&.{ .{ .begin = 2, .end = 6, .start = 0 }, .{ .begin = 8, .end = 10, .start = 20 } }, &fence, record_gate_cosine));
    // Low cosines outside the spans (generated label slots) do not matter.
    var noisy: [16]f32 = @splat(0.3);
    noisy[2] = 0.85;
    noisy[3] = 0.9;
    noisy[4] = 1.0;
    noisy[5] = 0.86;
    try std.testing.expect(try recordGateOpen(&.{.{ .begin = 2, .end = 6, .start = 0 }}, &noisy, record_gate_cosine));
    try std.testing.expect(!try recordGateOpen(&.{.{ .begin = 2, .end = 6, .start = 0 }}, &noisy, 0.9));
    // The evaluator compares four-decimal cosines: 0.84996 counts as 0.85.
    noisy[5] = 0.84996;
    try std.testing.expect(try recordGateOpen(&.{.{ .begin = 2, .end = 6, .start = 0 }}, &noisy, record_gate_cosine));
    noisy[5] = 0.8494;
    try std.testing.expect(!try recordGateOpen(&.{.{ .begin = 2, .end = 6, .start = 0 }}, &noisy, record_gate_cosine));
    try std.testing.expectError(error.InvalidCopySpan, recordGateOpen(&.{.{ .begin = 10, .end = 17, .start = 0 }}, &cosines, record_gate_cosine));
    try std.testing.expectError(error.InvalidRecordGate, recordGateOpen(&.{.{ .begin = 2, .end = 6, .start = 0 }}, &cosines, 1.5));
}

test "Memorilla hybrid guard rejects only frame echo" {
    // General answers: sentences, punctuation and length are fine.
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer("The Normans were in Normandy in the 10th and 11th centuries, according to the passage."));
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer("Denmark, Iceland and Norway"));
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer("\"Rollo\" (c. 860 - c. 930)"));
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer("Kg9-009369"));
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer(" UNKNOWN\n"));
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer("Memoryville"));
    try std.testing.expectEqual(AnswerGuard.accepted, guardAnswer("Zürich"));
    // Frame echo.
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer("Memory key holder with the brass key for Zehisi Depot  \nMemory"));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer("Zehisi Memory"));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer("<|memory|>Alice"));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer("First line\nsecond line"));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer("tab\tseparated"));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer(""));
    try std.testing.expectEqual(AnswerGuard.rejected, guardAnswer(" \t\n"));
}

test "Memorilla hybrid merged prompt and lenient record pieces" {
    const a = std.testing.allocator;
    const merged = try mergedSystemPrompt(a, "Answer using the supplied memory.");
    defer a.free(merged);
    try std.testing.expectEqualStrings("You are a helpful assistant. Answer using the supplied memory.", merged);
    var out: std.ArrayListUnmanaged(u8) = .empty;
    defer out.deinit(a);
    try appendLossy(a, &out, " Z\xc3\xbcrich");
    try appendLossy(a, &out, "\xe2\x80");
    try appendLossy(a, &out, "\x80x");
    try std.testing.expectEqualStrings(" Zürich\u{fffd}\u{fffd}x", out.items);
}

test "Memorilla hybrid admission covers either decoder plan" {
    const open = memory.AdmissionAmounts{ .host_scratch_bytes = 10, .backend_kv_bytes = 30, .backend_scratch_bytes = 5 };
    const closed = memory.AdmissionAmounts{ .host_scratch_bytes = 12, .backend_kv_bytes = 20, .backend_scratch_bytes = 5 };
    const both = maxAmounts(open, closed);
    try std.testing.expectEqual(@as(usize, 12), both.host_scratch_bytes);
    try std.testing.expectEqual(@as(usize, 30), both.backend_kv_bytes);
    try std.testing.expectEqual(@as(usize, 5), both.backend_scratch_bytes);
    try std.testing.expectEqual(@as(usize, 0), both.host_weight_bytes);
}

test "Memorilla hybrid validation rejects kind identity bounds and texts before models" {
    const research_kind = Profile{ .name = "copy17", .encoder = "encoder", .decoder = "decoder", .memory_module_path = "bundle", .reader_kind = .encoder_lora_research };
    var p = research_kind;
    p.reader_kind = .hybrid_research;
    p.released_memory_module_path = "stage3";
    const identity = qualified.executionIdentity(encoder_profile);
    const docs = [_][]const u8{ "Alice Smith holds the brass key for Mozeba Depot.", "The Normans gave their name to Normandy." };
    try validateInputs(p, "\xc2\xa0question\xc2\xa0", &docs, &identity, 48);
    try std.testing.expectError(error.UnsupportedMemorillaProfile, validateInputs(research_kind, "q", &docs, &identity, 24));
    try std.testing.expectError(error.EmptyMemorillaCollection, validateInputs(p, "q", &.{}, &identity, 1));
    try std.testing.expectError(error.IncompatibleMemorillaProducer, validateInputs(p, "q", &docs, "other", 1));
    // The research encoder identity is not the stock encoder that serves both passes.
    try std.testing.expectError(error.IncompatibleMemorillaProducer, validateInputs(p, "q", &docs, &qualified.executionIdentity(.qwen3_encoder_lora_research_f32_v1), 1));
    try std.testing.expectError(error.MemorillaGenerationLimitExceeded, validateInputs(p, "q", &docs, &identity, 129));
    try std.testing.expectError(error.InvalidMemorillaSourceText, validateInputs(p, "q", &.{" \n"}, &identity, 1));
    const long: [8193]u8 = @splat('a');
    try std.testing.expectError(error.MemorillaSourceTextLimitExceeded, validateInputs(p, "q", &.{&long}, &identity, 1));
    try std.testing.expectError(error.MemorillaTextLimitExceeded, validateInputs(p, &long, &docs, &identity, 1));
    var small = p;
    small.max_documents = 1;
    try std.testing.expectError(error.MemorillaCollectionBudgetExceeded, validateInputs(small, "q", &docs, &identity, 1));
    var missing = p;
    missing.released_memory_module_path = null;
    try std.testing.expectError(error.InvalidMemorillaConfig, validateInputs(missing, "q", &docs, &identity, 1));
}

test "Memorilla hybrid service cancellation precedes component resolution and model loading" {
    try std.testing.expectError(error.Timeout, generate(std.testing.allocator, std.testing.io, undefined, undefined, undefined, undefined, undefined, undefined, undefined, undefined, .{}, .{ .deadline_ns = 0 }, .{}));
}

/// Word-level stand-in tokenizer for the chunker: every whitespace-separated
/// word (punctuation attached) is one token.
const WordCountTokenizer = struct {
    var sentinel: u8 = 0;
    fn handle() tok.Tokenizer {
        return .{ .ptr = &sentinel, .vtable = &.{ .encode = encode, .encodeInto = append, .encodeForModel = unused, .encodeGeneration = unusedGeneration, .decode = decodeIds, .specialTokens = specials, .vocabSize = size, .deinit = deinit } };
    }
    fn encode(_: *anyopaque, a: A, text: []const u8) anyerror![]i32 {
        var count: usize = 0;
        var it = std.mem.tokenizeAny(u8, text, &std.ascii.whitespace);
        while (it.next()) |_| count += 1;
        const ids = try a.alloc(i32, count);
        @memset(ids, 7);
        return ids;
    }
    fn append(raw: *anyopaque, a: A, text: []const u8, out: *std.ArrayListUnmanaged(i32)) anyerror!void {
        const ids = try encode(raw, a, text);
        defer a.free(ids);
        try out.appendSlice(a, ids);
    }
    fn decodeIds(_: *anyopaque, a: A, _: []const i32) anyerror![]u8 {
        return a.dupe(u8, "");
    }
    fn unused(_: *anyopaque, _: A, _: []const u8, _: usize) anyerror!tok.EncodeResult {
        return error.NotUsed;
    }
    fn unusedGeneration(_: *anyopaque, _: A, _: []const u8, _: usize, _: bool) anyerror!tok.EncodeResult {
        return error.NotUsed;
    }
    fn specials(_: *anyopaque) tok.SpecialTokens {
        return .{};
    }
    fn size(_: *anyopaque) usize {
        return 16;
    }
    fn deinit(_: *anyopaque) void {}
};

test "Memorilla hybrid chunker packs sentences greedily within the token bound" {
    const a = std.testing.allocator;
    const tokenizer = WordCountTokenizer.handle();
    // Bound 5 = four words plus EOS.
    const text = "One two three. Four five.\nSix seven eight nine ten eleven. Twelve!";
    const chunks = try chunkDocument(a, tokenizer, text, 5);
    defer a.free(chunks);
    var got: std.ArrayListUnmanaged([]const u8) = .empty;
    defer got.deinit(a);
    for (chunks) |chunk| try got.append(a, text[chunk.start..chunk.end]);
    const expected = [_][]const u8{ "One two three.", "Four five.", "Six seven eight nine", "ten eleven. Twelve!" };
    try std.testing.expectEqual(expected.len, got.items.len);
    for (expected, got.items) |want, have| try std.testing.expectEqualStrings(want, have);
    // Every chunk fits the bound.
    for (got.items) |chunk| try std.testing.expect(try tokenCount(a, tokenizer, chunk) <= 5);
    const one = try chunkDocument(a, tokenizer, "  short text  ", 5);
    defer a.free(one);
    try std.testing.expectEqual(@as(usize, 1), one.len);
    try std.testing.expectEqualStrings("short text", "  short text  "[one[0].start..one[0].end]);
    try std.testing.expectError(error.MemorillaTextLimitExceeded, chunkDocument(a, tokenizer, "x", 1));
}
