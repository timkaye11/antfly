// Copyright 2026 Antfly, Inc.
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

const std = @import("std");
const compat = @import("../io/compat.zig");

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Character-level span for one chunk within a document.
pub const FusedChunkBoundary = struct {
    start_char: u32,
    end_char: u32,
    start_token: u32 = 0,
    end_token: u32 = 0,
};

/// One training example: a document with pre-labeled chunk boundaries.
pub const FusedSample = struct {
    text: []const u8,
    chunk_boundaries: []FusedChunkBoundary,
    /// Optional paraphrase texts for contrastive learning.
    positive_texts: []const []const u8,
    /// Optional per-sample hard negative texts (Feature 7).
    hard_negatives: []const []const u8 = &.{},
    /// Benchmark metadata: true only for the prepared record that represents
    /// the end of the original Chonky evaluation stream.
    benchmark_include_final_boundary: bool = false,
};

/// Fixed-size batch for model training/inference.
/// All values are pre-tokenized. Dimensions: [batch_size * max_seq_len] flat.
pub const FusedBatch = struct {
    batch_size: usize,
    max_seq_len: usize,
    max_chunks: usize,

    /// Token IDs: [batch_size * max_seq_len] i32
    input_ids: []i32,
    /// Attention mask: [batch_size * max_seq_len] i32 (1=real, 0=pad)
    attention_mask: []i32,
    /// Boundary labels: [batch_size * max_seq_len] f32
    /// 1.0 at the first token of each chunk boundary after the first chunk.
    boundary_labels: []f32,
    /// Sentence-like boundary candidate mask: [batch_size * max_seq_len] f32.
    /// 1.0 for valid tokens that start after sentence punctuation and begin
    /// with an uppercase byte. Gold boundaries are a sparse subset of these.
    boundary_candidate_mask: []f32,
    /// Chunk start token indices: [batch_size * max_chunks] i32
    chunk_starts: []i32,
    /// Chunk end token indices (exclusive): [batch_size * max_chunks] i32
    chunk_ends: []i32,
    /// Chunk validity mask: [batch_size * max_chunks] f32 (1.0=valid, 0.0=padding)
    chunk_mask: []f32,
    /// Sample indices back into the source dataset (for debugging).
    sample_indices: []usize,

    // Hard negatives (Feature 7): null when no samples have hard_negatives.
    // hard_neg_ids: [batch_size * max_negatives * max_seq_len]
    hard_neg_ids: ?[]i32 = null,
    // hard_neg_mask: [batch_size * max_negatives * max_seq_len]
    hard_neg_mask: ?[]i32 = null,
    num_negatives: usize = 0,

    pub fn deinit(self: *FusedBatch, allocator: std.mem.Allocator) void {
        allocator.free(self.input_ids);
        allocator.free(self.attention_mask);
        allocator.free(self.boundary_labels);
        allocator.free(self.boundary_candidate_mask);
        allocator.free(self.chunk_starts);
        allocator.free(self.chunk_ends);
        allocator.free(self.chunk_mask);
        allocator.free(self.sample_indices);
        if (self.hard_neg_ids) |h| allocator.free(h);
        if (self.hard_neg_mask) |h| allocator.free(h);
        self.* = undefined;
    }
};

/// Aggregate statistics over a loaded dataset.
pub const FusedDatasetStats = struct {
    num_samples: usize = 0,
    avg_text_chars: f64 = 0,
    avg_chunks_per_sample: f64 = 0,
    min_chunks: usize = 0,
    max_chunks: usize = 0,
    samples_with_contrastive_positives: usize = 0,
    samples_with_boundary_targets: usize = 0,
    total_boundary_targets: usize = 0,
};

pub const LoadedSamples = struct {
    arena: std.heap.ArenaAllocator,
    dataset_root: []const u8,
    samples: []FusedSample,

    pub fn deinit(self: *LoadedSamples) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

// ---------------------------------------------------------------------------
// Private parsing types (JSONL wire format)
// ---------------------------------------------------------------------------

const RawChunk = struct {
    start_char: ?u32 = null,
    end_char: ?u32 = null,
    start: ?u32 = null,
    end: ?u32 = null,
    start_token: ?u32 = null,
    end_token: ?u32 = null,
};

const RawRecord = struct {
    text: []const u8,
    chunks: ?[]const RawChunk = null,
    chunk_boundaries: ?[]const RawChunk = null,
    /// Prepared benchmark window marker. The Chonky-compatible final document
    /// separator is counted only for the true final prepared window.
    source_format: ?[]const u8 = null,
    benchmark_include_final_boundary: ?bool = null,
    /// Legacy flat positive strings, or a permissive JSON value for compatibility.
    positives: ?std.json.Value = null,
    /// Go production schema: [{"chunk_idx": 0, "positive_text": "..."}].
    positive_pairs: ?std.json.Value = null,
    /// Feature 7: accepts legacy flat strings or Go production objects:
    /// [{"chunk_idx": 0, "negatives": ["...", "..."]}].
    hard_negatives: ?std.json.Value = null,
};

fn appendPositiveTextsFromJson(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged([]const u8),
    value: std.json.Value,
) !void {
    switch (value) {
        .string => |s| try out.append(allocator, s),
        .array => |arr| {
            for (arr.items) |item| try appendPositiveTextsFromJson(allocator, out, item);
        },
        .object => |obj| {
            if (obj.get("positive_text")) |text_val| {
                try appendPositiveTextsFromJson(allocator, out, text_val);
            } else if (obj.get("text")) |text_val| {
                try appendPositiveTextsFromJson(allocator, out, text_val);
            } else if (obj.get("positives")) |texts_val| {
                try appendPositiveTextsFromJson(allocator, out, texts_val);
            }
        },
        else => {},
    }
}

fn appendHardNegativesFromJson(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged([]const u8),
    value: std.json.Value,
) !void {
    switch (value) {
        .string => |s| try out.append(allocator, s),
        .array => |arr| {
            for (arr.items) |item| try appendHardNegativesFromJson(allocator, out, item);
        },
        .object => |obj| {
            if (obj.get("negatives")) |texts_val| {
                try appendHardNegativesFromJson(allocator, out, texts_val);
            } else if (obj.get("negative_text")) |text_val| {
                try appendHardNegativesFromJson(allocator, out, text_val);
            } else if (obj.get("text")) |text_val| {
                try appendHardNegativesFromJson(allocator, out, text_val);
            }
        },
        else => {},
    }
}

fn flattenPositiveTexts(
    allocator: std.mem.Allocator,
    rec: RawRecord,
) ![]const []const u8 {
    var texts = std.ArrayListUnmanaged([]const u8).empty;
    defer texts.deinit(allocator);

    if (rec.positives) |value| try appendPositiveTextsFromJson(allocator, &texts, value);
    if (rec.positive_pairs) |value| try appendPositiveTextsFromJson(allocator, &texts, value);

    if (texts.items.len == 0) return &.{};
    return try texts.toOwnedSlice(allocator);
}

fn flattenHardNegatives(
    allocator: std.mem.Allocator,
    rec: RawRecord,
) ![]const []const u8 {
    var texts = std.ArrayListUnmanaged([]const u8).empty;
    defer texts.deinit(allocator);

    if (rec.hard_negatives) |value| try appendHardNegativesFromJson(allocator, &texts, value);

    if (texts.items.len == 0) return &.{};
    return try texts.toOwnedSlice(allocator);
}

fn rawRecordBenchmarkIncludeFinalBoundary(rec: RawRecord) bool {
    return rec.benchmark_include_final_boundary orelse
        (if (rec.source_format) |source_format|
            std.mem.eql(u8, source_format, "final")
        else
            false);
}

fn rawRecordHasUsableChunks(raw_chunks: []const RawChunk, benchmark_include_final_boundary: bool) bool {
    return raw_chunks.len >= 2 or (benchmark_include_final_boundary and raw_chunks.len == 1);
}

// ---------------------------------------------------------------------------
// File resolution (mirrors reranker_data.zig exactly)
// ---------------------------------------------------------------------------

const ResolvedFiles = struct {
    allocator: std.mem.Allocator,
    base_dir: []const u8,
    paths: [][]const u8,

    fn deinit(self: *ResolvedFiles) void {
        self.allocator.free(self.base_dir);
        for (self.paths) |path| self.allocator.free(path);
        self.allocator.free(self.paths);
        self.* = undefined;
    }
};

fn resolveJsonlFiles(allocator: std.mem.Allocator, path: []const u8, split: ?[]const u8) !ResolvedFiles {
    if (std.mem.trim(u8, path, " \t\r\n").len == 0) return error.EmptyPath;
    const stat = try compat.cwd().statFile(compat.io(), path, .{});
    if (stat.kind == .file) {
        const one = try allocator.alloc([]const u8, 1);
        errdefer allocator.free(one);
        one[0] = try allocator.dupe(u8, path);
        errdefer allocator.free(one[0]);
        const base_dir = try allocator.dupe(u8, std.fs.path.dirname(path) orelse ".");
        errdefer allocator.free(base_dir);
        return .{
            .allocator = allocator,
            .base_dir = base_dir,
            .paths = one,
        };
    }
    if (stat.kind != .directory) return error.UnsupportedPathType;

    var dir = try compat.cwd().openDir(compat.io(), path, .{ .iterate = true });
    defer dir.close(compat.io());
    var iter = dir.iterate();
    var paths = std.ArrayListUnmanaged([]const u8).empty;
    errdefer {
        for (paths.items) |owned_path| allocator.free(owned_path);
        paths.deinit(allocator);
    }
    while (try iter.next(compat.io())) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ".jsonl")) continue;
        if (split) |want_split| {
            if (!std.mem.startsWith(u8, entry.name, want_split) or
                entry.name.len <= want_split.len or
                entry.name[want_split.len] != '-')
            {
                continue;
            }
        }
        const joined = try std.fs.path.join(allocator, &.{ path, entry.name });
        paths.append(allocator, joined) catch |err| {
            allocator.free(joined);
            return err;
        };
    }
    if (paths.items.len == 0) return error.NoJsonlFilesForSplit;
    std.mem.sort([]const u8, paths.items, {}, lessThanString);
    const base_dir = try allocator.dupe(u8, path);
    errdefer allocator.free(base_dir);
    return .{
        .allocator = allocator,
        .base_dir = base_dir,
        .paths = try paths.toOwnedSlice(allocator),
    };
}

fn lessThanString(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.lessThan(u8, lhs, rhs);
}

// ---------------------------------------------------------------------------
// JSONL loading
// ---------------------------------------------------------------------------

const max_fused_jsonl_file_bytes: usize = 1024 * 1024 * 1024;

fn loadSamplesFromFile(
    allocator: std.mem.Allocator,
    path: []const u8,
    out: *std.ArrayListUnmanaged(FusedSample),
) !void {
    const data = try compat.cwd().readFileAlloc(compat.io(), path, allocator, .limited(max_fused_jsonl_file_bytes));
    var lines = std.mem.tokenizeScalar(u8, data, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r");
        if (line.len == 0) continue;
        const rec = try std.json.parseFromSliceLeaky(RawRecord, allocator, line, .{
            .ignore_unknown_fields = true,
        });

        // Skip records with empty text.
        if (std.mem.trim(u8, rec.text, " \t\r\n").len == 0) continue;

        // Prefer `chunks`, fall back to `chunk_boundaries`.
        const raw_chunks: []const RawChunk = rec.chunks orelse rec.chunk_boundaries orelse &.{};
        const benchmark_include_final_boundary = rawRecordBenchmarkIncludeFinalBoundary(rec);

        // Training samples need at least one supervised boundary. Benchmark
        // eval may keep a final one-chunk window to score Chonky's document-end
        // separator exactly once.
        if (!rawRecordHasUsableChunks(raw_chunks, benchmark_include_final_boundary)) continue;

        // Map raw chunks to canonical boundaries, supporting both field-name variants.
        const boundaries = try allocator.alloc(FusedChunkBoundary, raw_chunks.len);
        for (raw_chunks, 0..) |rc, i| {
            boundaries[i] = .{
                .start_char = rc.start_char orelse rc.start orelse 0,
                .end_char = rc.end_char orelse rc.end orelse 0,
                .start_token = rc.start_token orelse 0,
                .end_token = rc.end_token orelse 0,
            };
        }

        // Map optional positives from both the legacy flat schema and the Go
        // production `positive_pairs` schema.
        const positives = try flattenPositiveTexts(allocator, rec);

        // Map optional hard negatives (Feature 7) from both flat strings and
        // Go production chunk-scoped negative lists.
        const hard_negs = try flattenHardNegatives(allocator, rec);

        try out.append(allocator, .{
            .text = rec.text,
            .chunk_boundaries = boundaries,
            .positive_texts = positives,
            .hard_negatives = hard_negs,
            .benchmark_include_final_boundary = benchmark_include_final_boundary,
        });
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

pub fn loadSamples(
    allocator: std.mem.Allocator,
    path: []const u8,
    split: ?[]const u8,
) !LoadedSamples {
    var resolved = try resolveJsonlFiles(allocator, path, split);
    defer resolved.deinit();

    var arena = std.heap.ArenaAllocator.init(allocator);
    errdefer arena.deinit();
    const arena_alloc = arena.allocator();

    var samples = std.ArrayListUnmanaged(FusedSample).empty;
    defer samples.deinit(arena_alloc);

    for (resolved.paths) |resolved_path| {
        try loadSamplesFromFile(arena_alloc, resolved_path, &samples);
    }

    return .{
        .arena = arena,
        .dataset_root = try arena_alloc.dupe(u8, resolved.base_dir),
        .samples = try samples.toOwnedSlice(arena_alloc),
    };
}

pub fn computeStats(samples: []const FusedSample) FusedDatasetStats {
    var stats = FusedDatasetStats{ .num_samples = samples.len };
    if (samples.len == 0) return stats;

    var total_chars: usize = 0;
    var total_chunks: usize = 0;
    var with_contrastive_positives: usize = 0;
    var with_boundary_targets: usize = 0;
    var total_boundary_targets: usize = 0;
    stats.min_chunks = samples[0].chunk_boundaries.len;
    stats.max_chunks = samples[0].chunk_boundaries.len;

    for (samples) |s| {
        total_chars += s.text.len;
        total_chunks += s.chunk_boundaries.len;
        stats.min_chunks = @min(stats.min_chunks, s.chunk_boundaries.len);
        stats.max_chunks = @max(stats.max_chunks, s.chunk_boundaries.len);
        if (s.positive_texts.len > 0) with_contrastive_positives += 1;
        if (s.chunk_boundaries.len > 1) {
            with_boundary_targets += 1;
            total_boundary_targets += s.chunk_boundaries.len - 1;
        }
    }

    const n = @as(f64, @floatFromInt(samples.len));
    stats.avg_text_chars = @as(f64, @floatFromInt(total_chars)) / n;
    stats.avg_chunks_per_sample = @as(f64, @floatFromInt(total_chunks)) / n;
    stats.samples_with_contrastive_positives = with_contrastive_positives;
    stats.samples_with_boundary_targets = with_boundary_targets;
    stats.total_boundary_targets = total_boundary_targets;
    return stats;
}

/// Convert a character-level [start_char, end_char) span to token indices.
///
/// offsets[t] = {byte_start, byte_end} for token t.
/// Returns {start_token, end_token} where end_token is exclusive.
///
/// Tokens where offsets[t] == {0, 0} and t > 0 are treated as special/padding
/// tokens and are skipped during the search.
pub fn charToTokenBoundary(
    start_char: u32,
    end_char: u32,
    offsets: []const [2]u32,
) struct { start_token: u32, end_token: u32 } {
    if (offsets.len == 0) return .{ .start_token = 0, .end_token = 0 };

    // Find the first token whose byte range contains start_char.
    var start_tok: u32 = 0;
    var found_start = false;
    for (offsets, 0..) |off, t| {
        // Skip zero-span special tokens (e.g. [CLS], [SEP]) at positions > 0.
        if (t > 0 and off[0] == 0 and off[1] == 0) continue;
        if (off[0] <= start_char and start_char < off[1]) {
            start_tok = @intCast(t);
            found_start = true;
            break;
        }
    }
    if (!found_start) {
        // start_char is beyond all tokens — return an empty range clamped to the end.
        const last: u32 = @intCast(offsets.len);
        return .{ .start_token = last, .end_token = last };
    }

    // Find the last token whose byte range contains end_char (exclusive condition:
    // off[0] < end_char <= off[1] means end_char falls within or at the end of token t).
    var end_tok: u32 = start_tok;
    var found_end = false;
    for (offsets, 0..) |off, t| {
        if (t > 0 and off[0] == 0 and off[1] == 0) continue;
        if (off[0] < end_char and end_char <= off[1]) {
            end_tok = @as(u32, @intCast(t)) + 1;
            found_end = true;
            break;
        }
    }
    if (!found_end) {
        return .{ .start_token = start_tok, .end_token = start_tok };
    }

    // Clamp end_tok to valid range.
    const max_tok: u32 = @intCast(offsets.len);
    end_tok = @min(end_tok, max_tok);

    return .{ .start_token = start_tok, .end_token = end_tok };
}

fn previousNonWhitespaceByte(text: []const u8, start: usize) ?u8 {
    var i = @min(start, text.len);
    while (i > 0) {
        i -= 1;
        const c = text[i];
        if (!std.ascii.isWhitespace(c)) return c;
    }
    return null;
}

fn firstNonWhitespaceByteAtOrAfter(text: []const u8, start: usize) ?u8 {
    var i = @min(start, text.len);
    while (i < text.len) : (i += 1) {
        const c = text[i];
        if (!std.ascii.isWhitespace(c)) return c;
    }
    return null;
}

fn isSentencePunctuation(c: u8) bool {
    return c == '.' or c == '?' or c == '!';
}

fn isSentenceLikeBoundaryCandidate(text: []const u8, token_offset: [2]u32) bool {
    if (token_offset[0] == 0 and token_offset[1] == 0) return false;
    if (token_offset[0] >= text.len) return false;
    const start: usize = @intCast(token_offset[0]);
    const prev = previousNonWhitespaceByte(text, start) orelse return false;
    if (!isSentencePunctuation(prev)) return false;
    const first = firstNonWhitespaceByteAtOrAfter(text, start) orelse return false;
    return std.ascii.isUpper(first);
}

fn fillBoundaryCandidateMask(text: []const u8, offsets: []const [2]u32, out: []f32) void {
    const n = @min(offsets.len, out.len);
    for (0..n) |i| {
        out[i] = if (isSentenceLikeBoundaryCandidate(text, offsets[i])) 1.0 else 0.0;
    }
    if (n < out.len) @memset(out[n..], 0.0);
}

fn useGoChunkEndCompat(ctx: anytype) bool {
    const T = @TypeOf(ctx);
    return switch (@typeInfo(T)) {
        .pointer => |ptr| if (@hasField(ptr.child, "go_chunk_end_compat")) ctx.go_chunk_end_compat else false,
        else => false,
    };
}

/// Assemble a fixed-size token batch from a subset of samples addressed by `indices`.
///
/// token_fn signature:
///   fn(ctx: anytype, text: []const u8, out_ids: []i32, out_mask: []i32, out_offsets: ?[][2]u32) usize
///
/// token_fn fills the pre-allocated out_ids and out_mask slices (length max_seq_len),
/// optionally fills out_offsets with per-token byte ranges, and returns the actual
/// number of tokens produced (<= out_ids.len).
///
/// Pass `{}` as ctx for context-free (plain function) token_fn.
///
/// All output slices are allocated from `allocator`; caller owns them via FusedBatch.deinit.
pub fn assembleTokenBatch(
    allocator: std.mem.Allocator,
    samples: []const FusedSample,
    indices: []const usize,
    max_seq_len: usize,
    max_chunks: usize,
    ctx: anytype,
    token_fn: anytype,
) !FusedBatch {
    const batch_size = indices.len;
    const seq_elems = batch_size * max_seq_len;
    const chunk_elems = batch_size * max_chunks;

    const input_ids = try allocator.alloc(i32, seq_elems);
    errdefer allocator.free(input_ids);
    @memset(input_ids, 0);

    const attention_mask = try allocator.alloc(i32, seq_elems);
    errdefer allocator.free(attention_mask);
    @memset(attention_mask, 0);

    const boundary_labels = try allocator.alloc(f32, seq_elems);
    errdefer allocator.free(boundary_labels);
    @memset(boundary_labels, 0);

    const boundary_candidate_mask = try allocator.alloc(f32, seq_elems);
    errdefer allocator.free(boundary_candidate_mask);
    @memset(boundary_candidate_mask, 0);

    const chunk_starts = try allocator.alloc(i32, chunk_elems);
    errdefer allocator.free(chunk_starts);
    @memset(chunk_starts, 0);

    const chunk_ends = try allocator.alloc(i32, chunk_elems);
    errdefer allocator.free(chunk_ends);
    @memset(chunk_ends, 0);

    const chunk_mask = try allocator.alloc(f32, chunk_elems);
    errdefer allocator.free(chunk_mask);
    @memset(chunk_mask, 0);

    const sample_indices = try allocator.alloc(usize, batch_size);
    errdefer allocator.free(sample_indices);

    // Per-sample scratch buffer for character-to-token offset mapping.
    // Reused across all samples; zeroed before each call so unused entries are {0,0}.
    const offsets_scratch = try allocator.alloc([2]u32, max_seq_len);
    defer allocator.free(offsets_scratch);
    const go_chunk_end_compat = useGoChunkEndCompat(ctx);

    for (indices, 0..) |sample_idx, i| {
        sample_indices[i] = sample_idx;
        const sample = samples[sample_idx];

        const ids_slice = input_ids[i * max_seq_len .. (i + 1) * max_seq_len];
        const mask_slice = attention_mask[i * max_seq_len .. (i + 1) * max_seq_len];
        const candidate_slice = boundary_candidate_mask[i * max_seq_len .. (i + 1) * max_seq_len];

        @memset(offsets_scratch, .{ 0, 0 });

        const n_tokens = token_fn(ctx, sample.text, ids_slice, mask_slice, offsets_scratch);
        const active_offsets = offsets_scratch[0..@min(n_tokens, max_seq_len)];
        fillBoundaryCandidateMask(sample.text, active_offsets, candidate_slice);

        // Map character boundaries to valid token spans. Match the Go
        // builder's contract: skip unmapped/truncated spans entirely, label all
        // valid post-first boundaries, and only truncate the chunk arrays.
        const c_base = i * max_chunks;
        var valid_chunk_idx: usize = 0;

        for (sample.chunk_boundaries) |boundary| {
            const tok_span = charToTokenBoundary(
                boundary.start_char,
                boundary.end_char,
                active_offsets,
            );
            const resolved_start = if (boundary.start_token > 0) boundary.start_token else tok_span.start_token;
            const resolved_end = if (boundary.end_token > 0) boundary.end_token else tok_span.end_token;
            if (resolved_end <= resolved_start) continue;

            // Set boundary label at the first token of every chunk after the first.
            // This marks the positions where chunk splits occur in the sequence.
            if (valid_chunk_idx > 0 and resolved_start < max_seq_len) {
                boundary_labels[i * max_seq_len + resolved_start] = 1.0;
            }

            if (valid_chunk_idx < max_chunks) {
                var stored_end = resolved_end;
                if (go_chunk_end_compat and boundary.end_token == 0) {
                    var trim_end_for_go = false;
                    if (tok_span.end_token > 0 and tok_span.end_token - 1 < active_offsets.len) {
                        const prev_off = active_offsets[tok_span.end_token - 1];
                        trim_end_for_go = prev_off[1] == boundary.end_char and prev_off[0] + 1 == prev_off[1];
                    }
                    if (trim_end_for_go and stored_end > resolved_start) stored_end -= 1;
                }
                chunk_starts[c_base + valid_chunk_idx] = @intCast(resolved_start);
                chunk_ends[c_base + valid_chunk_idx] = @intCast(stored_end);
                chunk_mask[c_base + valid_chunk_idx] = 1.0;
            }
            valid_chunk_idx += 1;
        }
    }

    // Feature 7: Hard negatives — collect up to max_negatives per sample.
    const max_negatives: usize = 7;
    var any_hard_negs = false;
    for (indices) |sample_idx| {
        if (samples[sample_idx].hard_negatives.len > 0) {
            any_hard_negs = true;
            break;
        }
    }

    var hard_neg_ids_opt: ?[]i32 = null;
    var hard_neg_mask_opt: ?[]i32 = null;
    var actual_num_negatives: usize = 0;

    if (any_hard_negs) {
        const neg_elems = batch_size * max_negatives * max_seq_len;
        const hn_ids = try allocator.alloc(i32, neg_elems);
        errdefer allocator.free(hn_ids);
        @memset(hn_ids, 0);
        const hn_mask = try allocator.alloc(i32, neg_elems);
        errdefer allocator.free(hn_mask);
        @memset(hn_mask, 0);

        for (indices, 0..) |sample_idx, i| {
            const sample = samples[sample_idx];
            const n_neg = @min(sample.hard_negatives.len, max_negatives);
            if (n_neg > actual_num_negatives) actual_num_negatives = n_neg;
            for (0..n_neg) |ni| {
                const base = (i * max_negatives + ni) * max_seq_len;
                const neg_ids_slice = hn_ids[base .. base + max_seq_len];
                const neg_mask_slice = hn_mask[base .. base + max_seq_len];
                // Temporary i32 slices for token_fn output
                _ = token_fn(ctx, sample.hard_negatives[ni], neg_ids_slice, neg_mask_slice, null);
            }
        }

        hard_neg_ids_opt = hn_ids;
        hard_neg_mask_opt = hn_mask;
    }

    return .{
        .batch_size = batch_size,
        .max_seq_len = max_seq_len,
        .max_chunks = max_chunks,
        .input_ids = input_ids,
        .attention_mask = attention_mask,
        .boundary_labels = boundary_labels,
        .boundary_candidate_mask = boundary_candidate_mask,
        .chunk_starts = chunk_starts,
        .chunk_ends = chunk_ends,
        .chunk_mask = chunk_mask,
        .sample_indices = sample_indices,
        .hard_neg_ids = hard_neg_ids_opt,
        .hard_neg_mask = hard_neg_mask_opt,
        .num_negatives = actual_num_negatives,
    };
}

/// Mean-pool token-level encoder features into one L2-normalized dense vector
/// per chunk, matching the Go fused chunker late-chunking output.
///
/// features: [batch_size * max_seq_len * hidden_size]
/// returns:  [batch_size * max_chunks * hidden_size]
pub fn meanPoolChunkEmbeddings(
    allocator: std.mem.Allocator,
    features: []const f32,
    batch: *const FusedBatch,
    hidden_size: usize,
) ![]f32 {
    const B = batch.batch_size;
    const S = batch.max_seq_len;
    const C = batch.max_chunks;
    if (features.len < B * S * hidden_size) return error.FeatureBufferTooSmall;

    const out = try allocator.alloc(f32, B * C * hidden_size);
    errdefer allocator.free(out);
    @memset(out, 0);

    for (0..B) |b| {
        for (0..C) |c| {
            const chunk_idx = b * C + c;
            if (batch.chunk_mask[chunk_idx] <= 0.5) continue;

            var start_i = batch.chunk_starts[chunk_idx];
            var end_i = batch.chunk_ends[chunk_idx];
            if (start_i < 0) start_i = 0;
            if (end_i < 0) end_i = 0;

            const start = @min(@as(usize, @intCast(start_i)), S);
            var end = @min(@as(usize, @intCast(end_i)), S);
            if (end <= start and start < S) end = start + 1;
            if (end <= start) continue;

            const dst = out[chunk_idx * hidden_size .. (chunk_idx + 1) * hidden_size];
            var count: usize = 0;
            for (start..end) |t| {
                if (batch.attention_mask[b * S + t] == 0) continue;
                const src = features[(b * S + t) * hidden_size .. (b * S + t + 1) * hidden_size];
                for (src, dst) |v, *acc| acc.* += v;
                count += 1;
            }
            if (count == 0) {
                @memset(dst, 0);
                continue;
            }
            const inv_count = 1.0 / @as(f32, @floatFromInt(count));
            for (dst) |*v| v.* *= inv_count;

            var norm_sq: f32 = 0;
            for (dst) |v| norm_sq += v * v;
            const inv_norm = 1.0 / @sqrt(norm_sq + 1e-12);
            for (dst) |*v| v.* *= inv_norm;
        }
    }

    return out;
}

/// Scatter gradients from L2-normalized mean-pooled chunk embeddings back to
/// token features.
///
/// `feature_grads` is updated in place and must be shaped
/// [batch_size * max_seq_len * hidden_size]. `chunk_grads` is shaped
/// [batch_size * max_chunks * hidden_size].
pub fn addMeanPoolChunkEmbeddingGradToFeatures(
    allocator: std.mem.Allocator,
    feature_grads: []f32,
    features: []const f32,
    chunk_grads: []const f32,
    chunk_starts: []const i32,
    chunk_ends: []const i32,
    chunk_mask: []const f32,
    attention_mask: []const f32,
    batch_size: usize,
    max_seq_len: usize,
    max_chunks: usize,
    hidden_size: usize,
) !void {
    if (feature_grads.len < batch_size * max_seq_len * hidden_size) return error.FeatureGradBufferTooSmall;
    if (features.len < batch_size * max_seq_len * hidden_size) return error.FeatureBufferTooSmall;
    if (chunk_grads.len < batch_size * max_chunks * hidden_size) return error.ChunkGradBufferTooSmall;
    if (chunk_starts.len < batch_size * max_chunks or
        chunk_ends.len < batch_size * max_chunks or
        chunk_mask.len < batch_size * max_chunks) return error.ChunkMetadataTooSmall;
    if (attention_mask.len < batch_size * max_seq_len) return error.AttentionMaskTooSmall;

    const pooled = try allocator.alloc(f32, hidden_size);
    defer allocator.free(pooled);

    for (0..batch_size) |b| {
        for (0..max_chunks) |c| {
            const chunk_idx = b * max_chunks + c;
            if (chunk_mask[chunk_idx] <= 0.5) continue;

            var start_i = chunk_starts[chunk_idx];
            var end_i = chunk_ends[chunk_idx];
            if (start_i < 0) start_i = 0;
            if (end_i < 0) end_i = 0;

            const start = @min(@as(usize, @intCast(start_i)), max_seq_len);
            var end = @min(@as(usize, @intCast(end_i)), max_seq_len);
            if (end <= start and start < max_seq_len) end = start + 1;
            if (end <= start) continue;

            var count: usize = 0;
            for (start..end) |t| {
                if (attention_mask[b * max_seq_len + t] > 0.5) count += 1;
            }
            if (count == 0) continue;

            const inv_count = 1.0 / @as(f32, @floatFromInt(count));
            const src = chunk_grads[chunk_idx * hidden_size .. (chunk_idx + 1) * hidden_size];
            @memset(pooled, 0);
            for (start..end) |t| {
                if (attention_mask[b * max_seq_len + t] <= 0.5) continue;
                const row = features[(b * max_seq_len + t) * hidden_size .. (b * max_seq_len + t + 1) * hidden_size];
                for (row, pooled) |v, *acc| acc.* += v;
            }
            for (0..hidden_size) |h| {
                pooled[h] *= inv_count;
            }

            var norm_sq: f32 = 0;
            for (pooled) |v| {
                norm_sq += v * v;
            }
            const norm = @sqrt(norm_sq + 1e-12);
            const inv_norm = 1.0 / norm;

            var dot_y_g: f32 = 0;
            for (pooled, src) |p, g| {
                const y = p * inv_norm;
                dot_y_g += y * g;
            }

            for (start..end) |t| {
                if (attention_mask[b * max_seq_len + t] <= 0.5) continue;
                const dst = feature_grads[(b * max_seq_len + t) * hidden_size .. (b * max_seq_len + t + 1) * hidden_size];
                for (0..hidden_size) |h| {
                    const y = pooled[h] * inv_norm;
                    const d_pooled = (src[h] - y * dot_y_g) * inv_norm;
                    dst[h] += d_pooled * inv_count;
                }
            }
        }
    }
}

/// Mean-pool full sequence representations into one vector per sequence.
pub fn meanPoolSequenceEmbeddings(
    allocator: std.mem.Allocator,
    features: []const f32,
    attention_mask: []const i32,
    num_sequences: usize,
    seq_len: usize,
    hidden_size: usize,
) ![]f32 {
    if (features.len < num_sequences * seq_len * hidden_size) return error.FeatureBufferTooSmall;
    if (attention_mask.len < num_sequences * seq_len) return error.AttentionMaskTooSmall;

    const out = try allocator.alloc(f32, num_sequences * hidden_size);
    errdefer allocator.free(out);
    @memset(out, 0);

    for (0..num_sequences) |s| {
        const dst = out[s * hidden_size .. (s + 1) * hidden_size];
        var count: usize = 0;
        for (0..seq_len) |t| {
            if (attention_mask[s * seq_len + t] == 0) continue;
            const src = features[(s * seq_len + t) * hidden_size .. (s * seq_len + t + 1) * hidden_size];
            for (src, dst) |v, *acc| acc.* += v;
            count += 1;
        }
        if (count == 0) continue;
        const inv_count = 1.0 / @as(f32, @floatFromInt(count));
        for (dst) |*v| v.* *= inv_count;
    }

    return out;
}

// ---------------------------------------------------------------------------
// Length Bucketing (Feature 8)
// ---------------------------------------------------------------------------

/// Sort sample indices into buckets by text length for efficient batching.
/// Returns a new order that groups similar-length texts together.
/// Divides `indices` into windows of `bucket_size` and sorts each window
/// by `samples[idx].text.len` ascending.  This reduces padding waste by
/// keeping similar-length texts in the same batch.
pub fn sortByLength(
    allocator: std.mem.Allocator,
    samples: []const FusedSample,
    indices: []const usize,
    bucket_size: usize,
) ![]usize {
    const result = try allocator.dupe(usize, indices);
    errdefer allocator.free(result);

    const effective_bucket = @max(bucket_size, 1);
    var start: usize = 0;
    while (start < result.len) {
        const end = @min(start + effective_bucket, result.len);
        const window = result[start..end];
        std.mem.sort(usize, window, samples, struct {
            fn lessThan(ctx: []const FusedSample, a: usize, b: usize) bool {
                return ctx[a].text.len < ctx[b].text.len;
            }
        }.lessThan);
        start = end;
    }
    return result;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "charToTokenBoundary: basic span mapping" {
    // Token layout over "Hello world foo":
    //   t0: bytes [0, 5)  "Hello"
    //   t1: bytes [5, 6)  " "
    //   t2: bytes [6, 11) "world"
    //   t3: bytes [11,12) " "
    //   t4: bytes [12,15) "foo"
    //   t5: {0,0}         special/padding (t>0, skipped)
    var offsets = [_][2]u32{
        .{ 0, 5 },
        .{ 5, 6 },
        .{ 6, 11 },
        .{ 11, 12 },
        .{ 12, 15 },
        .{ 0, 0 },
    };

    // "Hello" only => [0, 1)
    {
        const r = charToTokenBoundary(0, 5, &offsets);
        try std.testing.expectEqual(@as(u32, 0), r.start_token);
        try std.testing.expectEqual(@as(u32, 1), r.end_token);
    }
    // "world" only => [2, 3)
    {
        const r = charToTokenBoundary(6, 11, &offsets);
        try std.testing.expectEqual(@as(u32, 2), r.start_token);
        try std.testing.expectEqual(@as(u32, 3), r.end_token);
    }
    // "Hello world" => [0, 3)
    {
        const r = charToTokenBoundary(0, 11, &offsets);
        try std.testing.expectEqual(@as(u32, 0), r.start_token);
        try std.testing.expectEqual(@as(u32, 3), r.end_token);
    }
    // "foo" => [4, 5)
    {
        const r = charToTokenBoundary(12, 15, &offsets);
        try std.testing.expectEqual(@as(u32, 4), r.start_token);
        try std.testing.expectEqual(@as(u32, 5), r.end_token);
    }
    // Span starts inside the tokenized window but ends after it; match the Go
    // trainer by returning an empty span so the chunk is masked out.
    {
        const r = charToTokenBoundary(12, 100, &offsets);
        try std.testing.expectEqual(@as(u32, 4), r.start_token);
        try std.testing.expectEqual(@as(u32, 4), r.end_token);
    }
    // start_char beyond all tokens => clamped to end
    {
        const r = charToTokenBoundary(100, 110, &offsets);
        try std.testing.expectEqual(@as(u32, 6), r.start_token);
        try std.testing.expectEqual(@as(u32, 6), r.end_token);
    }
}

test "assembleTokenBatch masks chunks outside tokenized window" {
    const allocator = std.testing.allocator;
    var boundaries = [_]FusedChunkBoundary{
        .{ .start_char = 0, .end_char = 3 },
        .{ .start_char = 100, .end_char = 110 },
    };
    const samples = [_]FusedSample{.{
        .text = "abc def",
        .chunk_boundaries = &boundaries,
        .positive_texts = &.{},
    }};
    const indices = [_]usize{0};

    const token_fn = struct {
        fn call(_: void, _: []const u8, out_ids: []i32, out_mask: []i32, out_offsets: ?[][2]u32) usize {
            @memset(out_ids, 0);
            @memset(out_mask, 0);
            out_ids[0] = 10;
            out_ids[1] = 11;
            out_mask[0] = 1;
            out_mask[1] = 1;
            if (out_offsets) |offsets| {
                @memset(offsets, .{ 0, 0 });
                offsets[0] = .{ 0, 3 };
                offsets[1] = .{ 4, 7 };
            }
            return 2;
        }
    }.call;

    var batch = try assembleTokenBatch(allocator, &samples, &indices, 4, 2, {}, token_fn);
    defer batch.deinit(allocator);

    try std.testing.expectEqualSlices(f32, &.{ 1.0, 0.0 }, batch.chunk_mask);
    try std.testing.expectEqualSlices(i32, &.{ 0, 0 }, batch.chunk_starts);
    try std.testing.expectEqualSlices(i32, &.{ 1, 0 }, batch.chunk_ends);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 0, 0 }, batch.boundary_labels);
}

test "assembleTokenBatch honors provided token boundaries" {
    const allocator = std.testing.allocator;
    var boundaries = [_]FusedChunkBoundary{
        .{ .start_char = 0, .end_char = 3 },
        .{ .start_char = 100, .end_char = 110, .start_token = 1, .end_token = 2 },
    };
    const samples = [_]FusedSample{.{
        .text = "abc def",
        .chunk_boundaries = &boundaries,
        .positive_texts = &.{},
    }};
    const indices = [_]usize{0};

    const token_fn = struct {
        fn call(_: void, _: []const u8, out_ids: []i32, out_mask: []i32, out_offsets: ?[][2]u32) usize {
            @memset(out_ids, 0);
            @memset(out_mask, 0);
            out_ids[0] = 10;
            out_ids[1] = 11;
            out_mask[0] = 1;
            out_mask[1] = 1;
            if (out_offsets) |offsets| {
                @memset(offsets, .{ 0, 0 });
                offsets[0] = .{ 0, 3 };
                offsets[1] = .{ 4, 7 };
            }
            return 2;
        }
    }.call;

    var batch = try assembleTokenBatch(allocator, &samples, &indices, 4, 2, {}, token_fn);
    defer batch.deinit(allocator);

    try std.testing.expectEqualSlices(f32, &.{ 1.0, 1.0 }, batch.chunk_mask);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1 }, batch.chunk_starts);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2 }, batch.chunk_ends);
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 0, 0 }, batch.boundary_labels);
}

test "assembleTokenBatch marks sentence-like boundary candidates" {
    const allocator = std.testing.allocator;
    var boundaries = [_]FusedChunkBoundary{
        .{ .start_char = 0, .end_char = 4 },
        .{ .start_char = 5, .end_char = 8 },
    };
    const samples = [_]FusedSample{.{
        .text = "One. Two",
        .chunk_boundaries = &boundaries,
        .positive_texts = &.{},
    }};
    const indices = [_]usize{0};

    const token_fn = struct {
        fn call(_: void, _: []const u8, out_ids: []i32, out_mask: []i32, out_offsets: ?[][2]u32) usize {
            @memset(out_ids, 0);
            @memset(out_mask, 0);
            out_ids[0] = 10;
            out_ids[1] = 11;
            out_ids[2] = 12;
            out_mask[0] = 1;
            out_mask[1] = 1;
            out_mask[2] = 1;
            if (out_offsets) |offsets| {
                @memset(offsets, .{ 0, 0 });
                offsets[0] = .{ 0, 3 };
                offsets[1] = .{ 3, 4 };
                offsets[2] = .{ 5, 8 };
            }
            return 3;
        }
    }.call;

    var batch = try assembleTokenBatch(allocator, &samples, &indices, 4, 2, {}, token_fn);
    defer batch.deinit(allocator);

    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1, 0 }, batch.boundary_candidate_mask);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0, 1, 0 }, batch.boundary_labels);
}

test "meanPoolChunkEmbeddings averages and normalizes chunk token ranges with attention mask" {
    const allocator = std.testing.allocator;
    var input_ids = [_]i32{ 1, 2, 3, 4 };
    var attention_mask = [_]i32{ 1, 1, 0, 1 };
    var boundary_labels = [_]f32{ 0, 1, 0, 0 };
    var boundary_candidate_mask = [_]f32{ 0, 1, 0, 0 };
    var chunk_starts = [_]i32{ 0, 1, 2 };
    var chunk_ends = [_]i32{ 2, 4, 4 };
    var chunk_mask = [_]f32{ 1, 1, 0 };
    var sample_indices = [_]usize{0};
    const batch = FusedBatch{
        .batch_size = 1,
        .max_seq_len = 4,
        .max_chunks = 3,
        .input_ids = &input_ids,
        .attention_mask = &attention_mask,
        .boundary_labels = &boundary_labels,
        .boundary_candidate_mask = &boundary_candidate_mask,
        .chunk_starts = &chunk_starts,
        .chunk_ends = &chunk_ends,
        .chunk_mask = &chunk_mask,
        .sample_indices = &sample_indices,
    };
    const features = [_]f32{
        1,  10,
        3,  30,
        99, 99,
        5,  50,
    };

    const pooled = try meanPoolChunkEmbeddings(allocator, &features, &batch, 2);
    defer allocator.free(pooled);

    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / @sqrt(@as(f32, 404.0))), pooled[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0 / @sqrt(@as(f32, 404.0))), pooled[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0 / @sqrt(@as(f32, 1616.0))), pooled[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 40.0 / @sqrt(@as(f32, 1616.0))), pooled[3], 1e-6);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0 }, pooled[4..6]);
}

test "addMeanPoolChunkEmbeddingGradToFeatures scatters pooled gradients" {
    const allocator = std.testing.allocator;
    var feature_grads = [_]f32{
        0, 0,
        0, 0,
        0, 0,
        0, 0,
    };
    const chunk_grads = [_]f32{
        2, 20,
        3, 30,
    };
    const features = [_]f32{
        1,  10,
        3,  30,
        99, 99,
        5,  50,
    };
    const chunk_starts = [_]i32{ 0, 1 };
    const chunk_ends = [_]i32{ 2, 4 };
    const chunk_mask = [_]f32{ 1, 1 };
    const attention_mask = [_]f32{ 1, 1, 0, 1 };

    try addMeanPoolChunkEmbeddingGradToFeatures(
        allocator,
        &feature_grads,
        &features,
        &chunk_grads,
        &chunk_starts,
        &chunk_ends,
        &chunk_mask,
        &attention_mask,
        1,
        4,
        2,
        2,
    );

    for (feature_grads[0..6]) |value| {
        try std.testing.expectApproxEqAbs(@as(f32, 0), value, 1e-4);
    }
    try std.testing.expectEqualSlices(f32, &.{ 0, 0 }, feature_grads[4..6]);
    try std.testing.expectApproxEqAbs(@as(f32, 0), feature_grads[6], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 0), feature_grads[7], 1e-4);
}

test "meanPoolSequenceEmbeddings averages each valid sequence" {
    const allocator = std.testing.allocator;
    const features = [_]f32{
        1,  10,
        3,  30,
        5,  50,
        7,  70,
        9,  90,
        11, 110,
    };
    const mask = [_]i32{
        1, 0, 1,
        0, 1, 1,
    };

    const pooled = try meanPoolSequenceEmbeddings(allocator, &features, &mask, 2, 3, 2);
    defer allocator.free(pooled);

    try std.testing.expectEqualSlices(f32, &.{ 3, 30 }, pooled[0..2]);
    try std.testing.expectEqualSlices(f32, &.{ 10, 100 }, pooled[2..4]);
}

test "computeStats: hardcoded samples" {
    const boundaries_a = [_]FusedChunkBoundary{
        .{ .start_char = 0, .end_char = 10 },
        .{ .start_char = 10, .end_char = 20 },
        .{ .start_char = 20, .end_char = 30 },
    };
    const boundaries_b = [_]FusedChunkBoundary{
        .{ .start_char = 0, .end_char = 5 },
        .{ .start_char = 5, .end_char = 15 },
    };
    const positives_a = [_][]const u8{"paraphrase of sample a"};

    const samples = [_]FusedSample{
        .{
            .text = "hello world sample A", // 20 chars
            .chunk_boundaries = @constCast(&boundaries_a),
            .positive_texts = @constCast(&positives_a),
        },
        .{
            .text = "sample B text", // 13 chars
            .chunk_boundaries = @constCast(&boundaries_b),
            .positive_texts = &.{},
        },
    };

    const stats = computeStats(&samples);
    try std.testing.expectEqual(@as(usize, 2), stats.num_samples);
    try std.testing.expectEqual(@as(usize, 2), stats.min_chunks);
    try std.testing.expectEqual(@as(usize, 3), stats.max_chunks);
    try std.testing.expectEqual(@as(usize, 1), stats.samples_with_contrastive_positives);
    try std.testing.expectEqual(@as(usize, 2), stats.samples_with_boundary_targets);
    try std.testing.expectEqual(@as(usize, 3), stats.total_boundary_targets);
    // avg_chunks: (3 + 2) / 2 = 2.5
    try std.testing.expectApproxEqAbs(@as(f64, 2.5), stats.avg_chunks_per_sample, 1e-6);
    // avg_text_chars: (20 + 13) / 2 = 16.5
    try std.testing.expectApproxEqAbs(@as(f64, 16.5), stats.avg_text_chars, 1e-6);
}

test "JSON parsing: both chunk field names and both coordinate field names" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    // `chunks` + `{start_char, end_char}`
    const line_a =
        \\{"text":"hello world","chunks":[{"start_char":0,"end_char":5},{"start_char":6,"end_char":11}]}
    ;
    const rec_a = try std.json.parseFromSliceLeaky(RawRecord, aa, line_a, .{
        .ignore_unknown_fields = true,
    });
    try std.testing.expectEqualStrings("hello world", rec_a.text);
    try std.testing.expect(rec_a.chunks != null);
    try std.testing.expectEqual(@as(usize, 2), rec_a.chunks.?.len);
    try std.testing.expectEqual(@as(?u32, 0), rec_a.chunks.?[0].start_char);
    try std.testing.expectEqual(@as(?u32, 5), rec_a.chunks.?[0].end_char);
    try std.testing.expectEqual(@as(?u32, null), rec_a.chunks.?[0].start);

    // `chunk_boundaries` + `{start, end}` + `positives`
    const line_b =
        \\{"text":"foo bar baz","chunk_boundaries":[{"start":0,"end":3},{"start":4,"end":7}],"positives":["alt text"]}
    ;
    const rec_b = try std.json.parseFromSliceLeaky(RawRecord, aa, line_b, .{
        .ignore_unknown_fields = true,
    });
    try std.testing.expectEqualStrings("foo bar baz", rec_b.text);
    try std.testing.expect(rec_b.chunk_boundaries != null);
    try std.testing.expectEqual(@as(usize, 2), rec_b.chunk_boundaries.?.len);
    try std.testing.expectEqual(@as(?u32, 0), rec_b.chunk_boundaries.?[0].start);
    try std.testing.expectEqual(@as(?u32, 3), rec_b.chunk_boundaries.?[0].end);
    try std.testing.expectEqual(@as(?u32, null), rec_b.chunk_boundaries.?[0].start_char);
    try std.testing.expect(rec_b.positives != null);
    const positives_b = try flattenPositiveTexts(aa, rec_b);
    try std.testing.expectEqual(@as(usize, 1), positives_b.len);
    try std.testing.expectEqualStrings("alt text", positives_b[0]);
}

test "JSON parsing: final benchmark one-chunk records remain usable" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const final_line =
        \\{"text":"last paragraph","chunks":[{"start_char":0,"end_char":14}],"source_format":"final"}
    ;
    const final_rec = try std.json.parseFromSliceLeaky(RawRecord, aa, final_line, .{
        .ignore_unknown_fields = true,
    });
    const final_chunks: []const RawChunk = final_rec.chunks orelse &.{};
    try std.testing.expect(rawRecordBenchmarkIncludeFinalBoundary(final_rec));
    try std.testing.expect(rawRecordHasUsableChunks(final_chunks, rawRecordBenchmarkIncludeFinalBoundary(final_rec)));

    const middle_line =
        \\{"text":"middle paragraph","chunks":[{"start_char":0,"end_char":16}],"source_format":"chonky-tokens"}
    ;
    const middle_rec = try std.json.parseFromSliceLeaky(RawRecord, aa, middle_line, .{
        .ignore_unknown_fields = true,
    });
    const middle_chunks: []const RawChunk = middle_rec.chunks orelse &.{};
    try std.testing.expect(!rawRecordBenchmarkIncludeFinalBoundary(middle_rec));
    try std.testing.expect(!rawRecordHasUsableChunks(middle_chunks, rawRecordBenchmarkIncludeFinalBoundary(middle_rec)));

    const explicit_false_line =
        \\{"text":"override","chunks":[{"start_char":0,"end_char":8}],"source_format":"final","benchmark_include_final_boundary":false}
    ;
    const explicit_false_rec = try std.json.parseFromSliceLeaky(RawRecord, aa, explicit_false_line, .{
        .ignore_unknown_fields = true,
    });
    try std.testing.expect(!rawRecordBenchmarkIncludeFinalBoundary(explicit_false_rec));
}

test "JSON parsing: Go production positive_pairs and hard_negatives flatten" {
    const allocator = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const aa = arena.allocator();

    const line =
        \\{
        \\  "text":"alpha beta gamma delta",
        \\  "chunk_boundaries":[
        \\    {"start_char":0,"end_char":10,"start_token":1,"end_token":3},
        \\    {"start_char":11,"end_char":22,"start_token":3,"end_token":5}
        \\  ],
        \\  "positive_pairs":[
        \\    {"chunk_idx":0,"positive_text":"related alpha beta"},
        \\    {"chunk_idx":1,"positive_text":"related gamma delta"}
        \\  ],
        \\  "hard_negatives":[
        \\    {"chunk_idx":0,"negatives":["unrelated finance","unrelated sports"]},
        \\    {"chunk_idx":1,"negatives":["unrelated cooking"]}
        \\  ]
        \\}
    ;
    const rec = try std.json.parseFromSliceLeaky(RawRecord, aa, line, .{
        .ignore_unknown_fields = true,
    });

    const positives = try flattenPositiveTexts(aa, rec);
    try std.testing.expectEqual(@as(usize, 2), positives.len);
    try std.testing.expectEqualStrings("related alpha beta", positives[0]);
    try std.testing.expectEqualStrings("related gamma delta", positives[1]);

    const negatives = try flattenHardNegatives(aa, rec);
    try std.testing.expectEqual(@as(usize, 3), negatives.len);
    try std.testing.expectEqualStrings("unrelated finance", negatives[0]);
    try std.testing.expectEqualStrings("unrelated sports", negatives[1]);
    try std.testing.expectEqualStrings("unrelated cooking", negatives[2]);
}
