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

//! Memorilla copy17 `tokens-dual` encoder features (pure host F32).
//!
//! Mirrors `scripts/memorilla/encoder_lora.py` (`prefixed_documents`,
//! `self_context_tokens`, `self_context_states`, `document_features_dual`,
//! `feature_channel`) and the batch built by `encoder_lora_eval.py`:
//!
//! 1. every document gets one leading space unless it already starts with one;
//! 2. each document is encoded twice (`D + D`); the second copy's ids must equal
//!    `encode(D)` (else `encode(" " + D)`, else error) and are the copy sources;
//! 3. rows `[D1+D1, ..., Dn+Dn, question]`, each followed by EOS 151643, right
//!    padded with 151643 to a common length `<= 128`;
//! 4. from both returned state tensors (final norm output and HF
//!    `hidden_states[24]`) keep each document's last `n_i + 1` active rows and
//!    the question row whole;
//! 5. rows are L2-normalized in F32 (eps 1e-12) and sent through the stored
//!    feature channel F32 -> FP16 (RNE) -> BF16 (RNE) -> F32; `docs` is the
//!    final-layer rows followed by the layer-24 rows (`[2M, 2560]`), `query`
//!    the question's last active final-layer row.
const std = @import("std");
const tok = @import("inference_tokenizer");
const qualified = @import("../models/memorilla_qualification.zig");
const encoder = @import("memorilla_encoder.zig");
const A = std.mem.Allocator;

pub const width: usize = qualified.width;
/// `<|endoftext|>`: appended to every row by the HF post-processor and used as
/// the pad id (mask 0).
pub const eos_id: i32 = 151643;
pub const pad_id: i32 = eos_id;
/// The candidate was trained and evaluated under this encoder bound.
pub const max_padded_length: usize = 128;
/// HF `hidden_states[k]` index of the intermediate layer (0-based layer 23).
pub const tap_layer: usize = 24;
pub const normalize_eps: f32 = 1e-12;

/// `prefixed_documents`: one leading space unless the text already has one.
pub fn prefixDocument(a: A, text: []const u8) ![]u8 {
    if (std.mem.startsWith(u8, text, " ")) return a.dupe(u8, text);
    const out = try a.alloc(u8, text.len + 1);
    out[0] = ' ';
    @memcpy(out[1..], text);
    return out;
}

fn encodeIds(a: A, tokenizer: tok.Tokenizer, text: []const u8) ![]i32 {
    const ids = try tokenizer.encode(a, text);
    errdefer a.free(ids);
    for (ids) |id| if (id < 0 or id == eos_id) return error.InvalidMemorillaTokens;
    return ids;
}

fn endsWith(ids: []const i32, tail: []const i32) bool {
    return tail.len <= ids.len and std.mem.eql(i32, ids[ids.len - tail.len ..], tail);
}

/// One document's causal self-context: the doubled text, how many trailing
/// ids belong to the second copy, and those ids (the copy sources).
pub const SelfContextDocument = struct {
    doubled: []u8,
    single_ids: []i32,
    fallback: bool,
};

/// `self_context_tokens`. `document` is normally already space prefixed; a
/// text without the prefix is repeated after one space, exactly as in Python.
pub fn selfContextDocument(a: A, tokenizer: tok.Tokenizer, document: []const u8) !SelfContextDocument {
    if (std.mem.trim(u8, document, &std.ascii.whitespace).len == 0) return error.BlankMemorillaDocument;
    const separator: []const u8 = if (std.mem.startsWith(u8, document, " ")) "" else " ";
    const doubled = try std.mem.concat(a, u8, &.{ document, separator, document });
    errdefer a.free(doubled);
    const doubled_ids = try encodeIds(a, tokenizer, doubled);
    defer a.free(doubled_ids);

    var single = try encodeIds(a, tokenizer, document);
    var fallback = false;
    if (single.len == 0 or !endsWith(doubled_ids, single)) {
        a.free(single);
        const spaced = try std.mem.concat(a, u8, &.{ " ", document });
        defer a.free(spaced);
        single = try encodeIds(a, tokenizer, spaced);
        fallback = true;
    }
    errdefer a.free(single);
    if (single.len == 0 or !endsWith(doubled_ids, single)) return error.MemorillaSelfContextMismatch;
    return .{ .doubled = doubled, .single_ids = single, .fallback = fallback };
}

/// `self_context_documents` over a case: doubled texts, kept counts `n_i` and
/// the concatenated copy sources.
pub const SelfContext = struct {
    a: A,
    texts: [][]u8,
    counts: []usize,
    copy_ids: []u32,
    fallbacks: usize,

    pub fn deinit(self: *SelfContext) void {
        for (self.texts) |text| self.a.free(text);
        self.a.free(self.texts);
        self.a.free(self.counts);
        self.a.free(self.copy_ids);
        self.* = undefined;
    }
};

pub fn selfContextTokens(a: A, tokenizer: tok.Tokenizer, documents: []const []const u8) !SelfContext {
    if (documents.len == 0 or documents.len > qualified.max_documents) return error.InvalidMemorillaDocumentCount;
    const texts = try a.alloc([]u8, documents.len);
    var built: usize = 0;
    errdefer {
        for (texts[0..built]) |text| a.free(text);
        a.free(texts);
    }
    const counts = try a.alloc(usize, documents.len);
    errdefer a.free(counts);
    var copy_ids = std.ArrayListUnmanaged(u32).empty;
    errdefer copy_ids.deinit(a);
    var fallbacks: usize = 0;
    for (documents, texts, counts) |document, *text, *count| {
        const context = try selfContextDocument(a, tokenizer, document);
        defer a.free(context.single_ids);
        text.* = context.doubled;
        built += 1;
        count.* = context.single_ids.len;
        fallbacks += @intFromBool(context.fallback);
        try copy_ids.ensureUnusedCapacity(a, context.single_ids.len);
        for (context.single_ids) |id| copy_ids.appendAssumeCapacity(@intCast(id));
    }
    return .{
        .a = a,
        .texts = texts,
        .counts = counts,
        .copy_ids = try copy_ids.toOwnedSlice(a),
        .fallbacks = fallbacks,
    };
}

/// The encoder batch `[D1+D1, ..., Dn+Dn, question]`, EOS appended and right
/// padded. `input_ids` is row-major `[batch, seq_len]` i64 as the `.gpt`
/// session run expects; `active[row]` counts the row's real tokens including
/// its EOS. `counts` and `copy_ids` are the documents' self-context values.
pub const Batch = struct {
    a: A,
    input_ids: []i64,
    batch: usize,
    seq_len: usize,
    active: []usize,
    counts: []usize,
    copy_ids: []u32,
    fallbacks: usize,

    pub fn documentCount(self: *const Batch) usize {
        return self.batch - 1;
    }

    pub fn deinit(self: *Batch) void {
        self.a.free(self.input_ids);
        self.a.free(self.active);
        self.a.free(self.counts);
        self.a.free(self.copy_ids);
        self.* = undefined;
    }
};

/// Tokenizes `context.texts ++ [question]` into one right-padded batch. The
/// question gets no prefix and no instruction. `context` stays owned by the
/// caller; its counts and copy ids are copied into the batch.
pub fn assembleBatch(a: A, tokenizer: tok.Tokenizer, context: *const SelfContext, question: []const u8) !Batch {
    const batch = context.texts.len + 1;
    const rows = try a.alloc([]i32, batch);
    var encoded: usize = 0;
    defer {
        for (rows[0..encoded]) |row| a.free(row);
        a.free(rows);
    }
    const active = try a.alloc(usize, batch);
    errdefer a.free(active);
    var seq_len: usize = 0;
    for (0..batch) |row| {
        const text = if (row < context.texts.len) context.texts[row] else question;
        rows[row] = try encodeIds(a, tokenizer, text);
        encoded += 1;
        active[row] = rows[row].len + 1;
        if (active[row] > max_padded_length) return error.MemorillaEncoderBoundExceeded;
        seq_len = @max(seq_len, active[row]);
    }
    if (std.mem.trim(u8, question, &std.ascii.whitespace).len == 0) return error.BlankMemorillaQuestion;
    const input_ids = try a.alloc(i64, batch * seq_len);
    errdefer a.free(input_ids);
    @memset(input_ids, pad_id);
    for (rows, 0..) |row, index| {
        const target = input_ids[index * seq_len ..][0..seq_len];
        for (row, target[0..row.len]) |id, *out| out.* = id;
        target[row.len] = eos_id;
    }
    for (context.counts, active[0..context.texts.len]) |count, row_active| {
        if (count + 1 > row_active) return error.MemorillaSelfContextMismatch;
    }
    const counts = try a.dupe(usize, context.counts);
    errdefer a.free(counts);
    const copy_ids = try a.dupe(u32, context.copy_ids);
    return .{
        .a = a,
        .input_ids = input_ids,
        .batch = batch,
        .seq_len = seq_len,
        .active = active,
        .counts = counts,
        .copy_ids = copy_ids,
        .fallbacks = context.fallbacks,
    };
}

/// Prefix, self-context and batch assembly in one step (section 1.2 steps 1-3).
pub fn prepare(a: A, tokenizer: tok.Tokenizer, documents: []const []const u8, question: []const u8) !Batch {
    if (documents.len == 0 or documents.len > qualified.max_documents) return error.InvalidMemorillaDocumentCount;
    const prefixed = try a.alloc([]u8, documents.len);
    var built: usize = 0;
    defer {
        for (prefixed[0..built]) |text| a.free(text);
        a.free(prefixed);
    }
    for (documents, prefixed) |document, *text| {
        text.* = try prefixDocument(a, document);
        built += 1;
    }
    var context = try selfContextTokens(a, tokenizer, prefixed);
    defer context.deinit();
    return assembleBatch(a, tokenizer, &context, question);
}

/// Number of selected document rows `M = sum(n_i + 1)`.
pub fn selectedRowCount(counts: []const usize) !usize {
    var total: usize = 0;
    for (counts) |count| total = try std.math.add(usize, total, try std.math.add(usize, count, 1));
    return total;
}

fn checkStates(states: []const f32, batch: usize, seq_len: usize, row_width: usize, active: []const usize) !void {
    if (batch < 2 or seq_len == 0 or row_width == 0 or active.len != batch) return error.InvalidMemorillaStates;
    const expected = try std.math.mul(usize, try std.math.mul(usize, batch, seq_len), row_width);
    if (states.len != expected) return error.InvalidMemorillaStates;
    for (active) |length| if (length == 0 or length > seq_len) return error.InvalidMemorillaStates;
}

/// `self_context_states` for the documents of one `[batch, seq_len, row_width]`
/// host tensor: rows `0..batch-1` are documents, the last row is the question
/// (not selected here). Returns `[M, row_width]` with each document's last
/// `counts[i] + 1` active rows in order (second copy, then its EOS state).
pub fn selectRows(
    a: A,
    states: []const f32,
    batch: usize,
    seq_len: usize,
    row_width: usize,
    active: []const usize,
    counts: []const usize,
) ![]f32 {
    try checkStates(states, batch, seq_len, row_width, active);
    if (counts.len != batch - 1) return error.InvalidMemorillaStates;
    const rows = try selectedRowCount(counts);
    const out = try a.alloc(f32, try std.math.mul(usize, rows, row_width));
    errdefer a.free(out);
    var cursor: usize = 0;
    for (counts, active[0 .. batch - 1], 0..) |count, length, index| {
        const keep = count + 1;
        if (keep > length) return error.InvalidMemorillaStates;
        const first = length - keep;
        const source = states[(index * seq_len + first) * row_width ..][0 .. keep * row_width];
        @memcpy(out[cursor * row_width ..][0 .. keep * row_width], source);
        cursor += keep;
    }
    std.debug.assert(cursor == rows);
    return out;
}

/// The question's last active row (its EOS state), a view into `states`.
pub fn questionLastRow(states: []const f32, batch: usize, seq_len: usize, row_width: usize, active: []const usize) ![]const f32 {
    try checkStates(states, batch, seq_len, row_width, active);
    const row = batch - 1;
    return states[(row * seq_len + active[row] - 1) * row_width ..][0..row_width];
}

/// `F.normalize(x, p=2, dim=-1)` in F32: `x / max(||x||, 1e-12)`. A zero or
/// non-finite row is an encoder failure, as in Python.
pub fn normalizeRow(row: []f32) !void {
    var sum: f64 = 0;
    for (row) |value| {
        if (!std.math.isFinite(value)) return error.NonFiniteMemorillaFeature;
        sum += @as(f64, value) * value;
    }
    const norm: f32 = @floatCast(@sqrt(sum));
    if (norm == 0) return error.ZeroMemorillaFeature;
    const denominator = @max(norm, normalize_eps);
    for (row) |*value| value.* /= denominator;
}

/// `feature_channel`: F32 -> FP16 (RNE, subnormals kept) -> BF16 (RNE) -> F32.
pub fn featureChannel(row: []f32) !void {
    for (row) |*value| value.* = (try encoder.storedValue(value.*)).memory;
}

pub fn normalizeAndChannel(row: []f32) !void {
    try normalizeRow(row);
    try featureChannel(row);
}

fn normalizeRows(values: []f32, row_width: usize) !void {
    var offset: usize = 0;
    while (offset < values.len) : (offset += row_width) try normalizeAndChannel(values[offset..][0..row_width]);
}

/// Reader input: `docs` `[rows, width]` = final-layer rows then layer-24 rows
/// (`rows = 2M`), `query` `[width]`, the copy sources and per-document counts.
pub const Features = struct {
    a: A,
    docs: []f32,
    rows: usize,
    query: []f32,
    copy_ids: []u32,
    counts: []usize,

    pub fn deinit(self: *Features) void {
        self.a.free(self.docs);
        self.a.free(self.query);
        self.a.free(self.copy_ids);
        self.a.free(self.counts);
        self.* = undefined;
    }
};

/// `document_features_dual` over the two host tensors returned for `batch`:
/// `final` is `last_hidden_state`, `tap` is `hidden_states_24`, both
/// `[batch.batch, batch.seq_len, width]` F32.
pub fn features(a: A, final: []const f32, tap: []const f32, batch: *const Batch) !Features {
    return featuresWithWidth(a, final, tap, batch, width);
}

pub fn featuresWithWidth(a: A, final: []const f32, tap: []const f32, batch: *const Batch, row_width: usize) !Features {
    if (final.len != tap.len) return error.InvalidMemorillaStates;
    const final_rows = try selectRows(a, final, batch.batch, batch.seq_len, row_width, batch.active, batch.counts);
    defer a.free(final_rows);
    const tap_rows = try selectRows(a, tap, batch.batch, batch.seq_len, row_width, batch.active, batch.counts);
    defer a.free(tap_rows);
    const docs = try a.alloc(f32, final_rows.len + tap_rows.len);
    errdefer a.free(docs);
    @memcpy(docs[0..final_rows.len], final_rows);
    @memcpy(docs[final_rows.len..], tap_rows);
    try normalizeRows(docs, row_width);

    const query = try a.dupe(f32, try questionLastRow(final, batch.batch, batch.seq_len, row_width, batch.active));
    errdefer a.free(query);
    try normalizeAndChannel(query);

    const copy_ids = try a.dupe(u32, batch.copy_ids);
    errdefer a.free(copy_ids);
    const counts = try a.dupe(usize, batch.counts);
    return .{
        .a = a,
        .docs = docs,
        .rows = docs.len / row_width,
        .query = query,
        .copy_ids = copy_ids,
        .counts = counts,
    };
}

// --- tests ---

/// Word-level stand-in for the Qwen tokenizer: a word preceded by a space maps
/// to `hash(word)`, a sentence-initial word (no space before it) to
/// `hash(word) + 100_000`, mirroring "Mujapo" -> M/uj/apo versus " Muj"/apo.
const WordTokenizer = struct {
    var sentinel: u8 = 0;
    fn hash(word: []const u8) i32 {
        var value: u32 = 7;
        for (word) |byte| value = value *% 31 +% byte;
        return @intCast(value % 50_000 + 1);
    }
    fn encode(_: *anyopaque, a: A, text: []const u8) ![]i32 {
        var ids = std.ArrayListUnmanaged(i32).empty;
        errdefer ids.deinit(a);
        var start: usize = 0;
        var index: usize = 0;
        while (index <= text.len) : (index += 1) {
            if (index == text.len or text[index] == ' ') {
                if (index > start) {
                    const initial = start == 0;
                    try ids.append(a, hash(text[start..index]) + @as(i32, if (initial) 100_000 else 0));
                }
                start = index + 1;
            }
        }
        return ids.toOwnedSlice(a);
    }
    fn append(raw: *anyopaque, a: A, text: []const u8, out: *std.ArrayListUnmanaged(i32)) !void {
        const ids = try encode(raw, a, text);
        defer a.free(ids);
        try out.appendSlice(a, ids);
    }
    fn decode(_: *anyopaque, a: A, _: []const i32) ![]u8 {
        return a.dupe(u8, "");
    }
    fn unused(_: *anyopaque, _: A, _: []const u8, _: usize) !tok.EncodeResult {
        return error.NotUsed;
    }
    fn unusedGeneration(_: *anyopaque, _: A, _: []const u8, _: usize, _: bool) !tok.EncodeResult {
        return error.NotUsed;
    }
    fn specials(_: *anyopaque) tok.SpecialTokens {
        return .{};
    }
    fn size(_: *anyopaque) usize {
        return 200_000;
    }
    fn deinit(_: *anyopaque) void {}
    fn handle() tok.Tokenizer {
        return .{ .ptr = &sentinel, .vtable = &.{ .encode = encode, .encodeInto = append, .encodeForModel = unused, .encodeGeneration = unusedGeneration, .decode = decode, .specialTokens = specials, .vocabSize = size, .deinit = deinit } };
    }
};

/// Every id is its position: a doubled text never ends with its own ids.
const PositionTokenizer = struct {
    var sentinel: u8 = 0;
    fn encode(_: *anyopaque, a: A, text: []const u8) ![]i32 {
        const out = try a.alloc(i32, text.len);
        for (out, 0..) |*id, index| id.* = @intCast(index);
        return out;
    }
    fn handle() tok.Tokenizer {
        return .{ .ptr = &sentinel, .vtable = &.{ .encode = encode, .encodeInto = WordTokenizer.append, .encodeForModel = WordTokenizer.unused, .encodeGeneration = WordTokenizer.unusedGeneration, .decode = WordTokenizer.decode, .specialTokens = WordTokenizer.specials, .vocabSize = WordTokenizer.size, .deinit = WordTokenizer.deinit } };
    }
};

test "memorilla token features prefix each document with one space" {
    const a = std.testing.allocator;
    const prefixed = try prefixDocument(a, "Antfly stores versions.");
    defer a.free(prefixed);
    try std.testing.expectEqualStrings(" Antfly stores versions.", prefixed);
    const kept = try prefixDocument(a, " already prefixed");
    defer a.free(kept);
    try std.testing.expectEqualStrings(" already prefixed", kept);
    const empty = try prefixDocument(a, "");
    defer a.free(empty);
    try std.testing.expectEqualStrings(" ", empty);
}

test "memorilla token features self-context doubles documents with the fallback rule" {
    const a = std.testing.allocator;
    const tokenizer = WordTokenizer.handle();
    const muj = WordTokenizer.hash("Mujapo");
    const depot = WordTokenizer.hash("Depot");

    // Prefixed text: the doubled text ends with encode(D) directly.
    const prefixed = try selfContextDocument(a, tokenizer, " Mujapo Depot");
    defer a.free(prefixed.doubled);
    defer a.free(prefixed.single_ids);
    try std.testing.expectEqualStrings(" Mujapo Depot Mujapo Depot", prefixed.doubled);
    try std.testing.expectEqualSlices(i32, &.{ muj, depot }, prefixed.single_ids);
    try std.testing.expect(!prefixed.fallback);

    // Sentence-initial word re-segments mid-text: fall back to encode(" " + D).
    const raw = try selfContextDocument(a, tokenizer, "Mujapo Depot");
    defer a.free(raw.doubled);
    defer a.free(raw.single_ids);
    try std.testing.expectEqualStrings("Mujapo Depot Mujapo Depot", raw.doubled);
    try std.testing.expectEqualSlices(i32, &.{ muj, depot }, raw.single_ids);
    try std.testing.expect(raw.fallback);

    try std.testing.expectError(error.MemorillaSelfContextMismatch, selfContextDocument(a, PositionTokenizer.handle(), " abc"));
    try std.testing.expectError(error.BlankMemorillaDocument, selfContextDocument(a, tokenizer, "   "));

    var context = try selfContextTokens(a, tokenizer, &.{ " Mujapo Depot", "Site code 7" });
    defer context.deinit();
    try std.testing.expectEqual(@as(usize, 2), context.texts.len);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3 }, context.counts);
    try std.testing.expectEqual(@as(usize, 1), context.fallbacks);
    const expected_copy = [_]u32{ @intCast(muj), @intCast(depot), @intCast(WordTokenizer.hash("Site")), @intCast(WordTokenizer.hash("code")), @intCast(WordTokenizer.hash("7")) };
    try std.testing.expectEqualSlices(u32, &expected_copy, context.copy_ids);
    try std.testing.expectError(error.InvalidMemorillaDocumentCount, selfContextTokens(a, tokenizer, &.{}));
}

test "memorilla token features batch appends EOS, right pads and bounds the length" {
    const a = std.testing.allocator;
    const tokenizer = WordTokenizer.handle();
    var batch = try prepare(a, tokenizer, &.{ "a b", " c" }, "q r");
    defer batch.deinit();
    try std.testing.expectEqual(@as(usize, 3), batch.batch);
    try std.testing.expectEqual(@as(usize, 2), batch.documentCount());
    try std.testing.expectEqual(@as(usize, 5), batch.seq_len);
    try std.testing.expectEqualSlices(usize, &.{ 5, 3, 3 }, batch.active);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, batch.counts);
    try std.testing.expectEqual(@as(usize, 0), batch.fallbacks);
    const ha = WordTokenizer.hash("a");
    const hb = WordTokenizer.hash("b");
    const hc = WordTokenizer.hash("c");
    const hq = WordTokenizer.hash("q") + 100_000;
    const hr = WordTokenizer.hash("r");
    const expected = [_]i64{
        ha,     hb,     ha,     hb,     eos_id,
        hc,     hc,     eos_id, pad_id, pad_id,
        hq,     hr,     eos_id, pad_id, pad_id,
    };
    try std.testing.expectEqualSlices(i64, &expected, batch.input_ids);
    try std.testing.expectEqualSlices(u32, &.{ @intCast(ha), @intCast(hb), @intCast(hc) }, batch.copy_ids);

    // 64 words doubled -> 128 ids + EOS = 129 > 128.
    var long_buf: [64 * 4]u8 = undefined;
    var long_len: usize = 0;
    for (0..64) |index| long_len += (try std.fmt.bufPrint(long_buf[long_len..], " w{d}", .{index})).len;
    const long = long_buf[0..long_len];
    try std.testing.expectError(error.MemorillaEncoderBoundExceeded, prepare(a, tokenizer, &.{long}, "q"));
    // 63 words doubled -> 126 ids + EOS = 127 fits.
    var fits = try prepare(a, tokenizer, &.{long[0 .. long.len - " w63".len]}, "q");
    defer fits.deinit();
    try std.testing.expectEqual(@as(usize, 127), fits.seq_len);
    try std.testing.expectError(error.BlankMemorillaQuestion, prepare(a, tokenizer, &.{"a"}, " "));
}

test "memorilla token features select the last n_i + 1 rows per document and the question EOS row" {
    const a = std.testing.allocator;
    const batch: usize = 3;
    const seq_len: usize = 5;
    const row_width: usize = 2;
    var states: [batch * seq_len * row_width]f32 = undefined;
    for (0..batch) |b| for (0..seq_len) |t| {
        const value: f32 = @floatFromInt(b * 100 + t);
        states[(b * seq_len + t) * row_width] = value;
        states[(b * seq_len + t) * row_width + 1] = -value;
    };
    const active = [_]usize{ 5, 3, 3 };
    const counts = [_]usize{ 1, 1 };
    const rows = try selectRows(a, &states, batch, seq_len, row_width, &active, &counts);
    defer a.free(rows);
    try std.testing.expectEqualSlices(f32, &.{ 3, -3, 4, -4, 101, -101, 102, -102 }, rows);
    const question = try questionLastRow(&states, batch, seq_len, row_width, &active);
    try std.testing.expectEqualSlices(f32, &.{ 202, -202 }, question);
    try std.testing.expectError(error.InvalidMemorillaStates, selectRows(a, &states, batch, seq_len, row_width, &active, &.{ 5, 1 }));
    try std.testing.expectError(error.InvalidMemorillaStates, selectRows(a, &states, batch, seq_len, row_width, &.{ 5, 3, 6 }, &counts));
    try std.testing.expectError(error.InvalidMemorillaStates, selectRows(a, states[0 .. states.len - 1], batch, seq_len, row_width, &active, &counts));
}

test "memorilla token features normalize in F32 and round through FP16 then BF16" {
    var row = [_]f32{ 3, 4, 0, 0 };
    try normalizeRow(&row);
    try std.testing.expectEqualSlices(f32, &.{ 0.6, 0.8, 0, 0 }, &row);
    var zero = [_]f32{ 0, 0 };
    try std.testing.expectError(error.ZeroMemorillaFeature, normalizeRow(&zero));
    var bad = [_]f32{ 1, std.math.nan(f32) };
    try std.testing.expectError(error.NonFiniteMemorillaFeature, normalizeRow(&bad));

    // FP16 ties to even: 1 + 2^-11 sits between 1 and 1 + 2^-10 -> 1.
    var tie_down = [_]f32{1.0 + 0.00048828125};
    try featureChannel(&tie_down);
    try std.testing.expectEqual(@as(f32, 1.0), tie_down[0]);
    // 1 + 3 * 2^-11 rounds up to 1 + 2^-9 in FP16 (even mantissa), then BF16
    // (7 mantissa bits) drops 2^-9 (below half of 2^-7) -> 1.
    try std.testing.expectEqual(@as(f16, 1.001953125), (try encoder.storedValue(1.0 + 3 * 0.00048828125)).fp16);
    var up_then_down = [_]f32{1.0 + 3 * 0.00048828125};
    try featureChannel(&up_then_down);
    try std.testing.expectEqual(@as(f32, 1.0), up_then_down[0]);
    // BF16 ties to even: 1 + 2^-8 (exact in FP16) -> 1; 1 + 3 * 2^-8 -> 1 + 2^-6.
    var bf_tie = [_]f32{ 1.0 + 0.00390625, 1.0 + 3 * 0.00390625 };
    try featureChannel(&bf_tie);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 1.015625 }, &bf_tie);
    // FP16 subnormals survive both roundings.
    var tiny = [_]f32{std.math.pow(f32, 2, -20)};
    try featureChannel(&tiny);
    try std.testing.expectEqual(std.math.pow(f32, 2, -20), tiny[0]);
    // Both halves of a normalized vector pass through the same channel.
    var pair = [_]f32{ 0.6, 0.8 };
    try featureChannel(&pair);
    try std.testing.expectEqual(@as(f32, 0.6015625), pair[0]);
    try std.testing.expectEqual(@as(f32, 0.80078125), pair[1]);
}

test "memorilla token features build docs as final rows then tap rows and the final-layer query" {
    const a = std.testing.allocator;
    const tokenizer = WordTokenizer.handle();
    var batch = try prepare(a, tokenizer, &.{ "a b", " c" }, "q r");
    defer batch.deinit();
    const row_width: usize = 2;
    const total = batch.batch * batch.seq_len * row_width;
    const final = try a.alloc(f32, total);
    defer a.free(final);
    const tap = try a.alloc(f32, total);
    defer a.free(tap);
    for (0..batch.batch) |b| for (0..batch.seq_len) |t| {
        const base: f32 = @floatFromInt(b * 100 + t + 1);
        final[(b * batch.seq_len + t) * row_width] = base;
        final[(b * batch.seq_len + t) * row_width + 1] = 0;
        tap[(b * batch.seq_len + t) * row_width] = 0;
        tap[(b * batch.seq_len + t) * row_width + 1] = -base;
    };
    var result = try featuresWithWidth(a, final, tap, &batch, row_width);
    defer result.deinit();
    // M = (2 + 1) + (1 + 1) = 5 rows per layer.
    try std.testing.expectEqual(@as(usize, 10), result.rows);
    try std.testing.expectEqual(@as(usize, 20), result.docs.len);
    for (result.docs[0..10], 0..) |value, index| try std.testing.expectEqual(@as(f32, if (index % 2 == 0) 1 else 0), value);
    for (result.docs[10..], 0..) |value, index| try std.testing.expectEqual(@as(f32, if (index % 2 == 0) 0 else -1), value);
    try std.testing.expectEqualSlices(f32, &.{ 1, 0 }, result.query);
    try std.testing.expectEqualSlices(u32, batch.copy_ids, result.copy_ids);
    try std.testing.expectEqualSlices(usize, &.{ 2, 1 }, result.counts);
    try std.testing.expectError(error.InvalidMemorillaStates, featuresWithWidth(a, final, tap[1..], &batch, row_width));
}

test "memorilla token features agree with the Qwen3 encoder tokenizer" {
    const platform = @import("antfly_platform");
    const directory = platform.env.getenv("ANTFLY_MEMORILLA_ENCODER_MODEL_DIR") orelse return error.SkipZigTest;
    const a = std.testing.allocator;
    const path = try std.fs.path.join(a, &.{ directory, "tokenizer.json" });
    defer a.free(path);
    const bytes = try @import("../util/c_file.zig").readFile(a, path);
    defer a.free(bytes);
    const hf = try @import("inference_hf_tokenizer").HfTokenizer.loadFromBytes(a, bytes);
    defer hf.deinitSelf();
    const tokenizer = hf.tokenizer();

    // scripts/memorilla/fixtures/encoder-tokenizer.json "short_document": the
    // pipeline appends 151643; `encode` itself matches add_special_tokens=False.
    const short = try tokenizer.encode(a, "Antfly stores immutable source versions.");
    defer a.free(short);
    try std.testing.expectEqualSlices(i32, &.{ 17117, 21642, 10533, 51572, 2530, 10795, 13 }, short);

    // A prefixed record starts with one leading-space token and its doubled
    // text ends with its own ids (no fallback).
    const context = try selfContextDocument(a, tokenizer, " At Mujapo Depot the owner code is QX7.");
    defer a.free(context.doubled);
    defer a.free(context.single_ids);
    try std.testing.expect(!context.fallback);
    const first = try tokenizer.decodeToken(a, context.single_ids[0], false);
    defer a.free(first);
    try std.testing.expectEqualStrings(" At", first);
    const second = try tokenizer.decodeToken(a, context.single_ids[1], false);
    defer a.free(second);
    try std.testing.expect(std.mem.startsWith(u8, second, " "));

    // The sentence-initial segmentation differs ("Mujapo" -> M/uj/apo), so an
    // unprefixed document takes the fallback and still yields the spaced ids.
    const raw = try selfContextDocument(a, tokenizer, "Mujapo Depot owner code QX7.");
    defer a.free(raw.doubled);
    defer a.free(raw.single_ids);
    const spaced = try tokenizer.encode(a, " Mujapo Depot owner code QX7.");
    defer a.free(spaced);
    try std.testing.expectEqualSlices(i32, spaced, raw.single_ids);

    var batch = try prepare(a, tokenizer, &.{ "At Mujapo Depot the owner code is QX7.", " Site Rendal Yard, holder Ana." }, "Which code belongs to Mujapo Depot?");
    defer batch.deinit();
    try std.testing.expectEqual(@as(usize, 3), batch.batch);
    try std.testing.expect(batch.seq_len <= max_padded_length);
    for (batch.active, 0..) |length, row| {
        try std.testing.expectEqual(@as(i64, eos_id), batch.input_ids[row * batch.seq_len + length - 1]);
        if (row < 2) try std.testing.expect(batch.counts[row] + 1 <= length);
    }
    try std.testing.expectEqual(batch.counts[0] + batch.counts[1], batch.copy_ids.len);
}
