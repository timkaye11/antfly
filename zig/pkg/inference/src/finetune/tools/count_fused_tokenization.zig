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
const inference = @import("inference_internal");

const fused_chunker_data = inference.finetune.fused_chunker_data;
const TokenizerBatch = inference.finetune.tokenizer_batch.TokenizerBatch;
const TokenFnCtx = inference.finetune.tokenizer_batch.TokenFnCtx;

const fnv_offset: u64 = 14695981039346656037;
const fnv_prime: u64 = 1099511628211;

fn hashU64(hash: *u64, value: u64) void {
    var v = value;
    for (0..8) |_| {
        hash.* ^= v & 0xff;
        hash.* *%= fnv_prime;
        v >>= 8;
    }
}

fn hashI32(hash: *u64, value: i32) void {
    hashU64(hash, @as(u32, @bitCast(value)));
}

fn usage() void {
    std.debug.print(
        \\usage: count-fused-tokenization --data <jsonl> --model-dir <dir> [--split <name>] [--max-seq-len <n>] [--max-chunks <n>] [--batch-size <n>] [--offset <n>] [--limit <n>] [--analysis-pos-weight <f>] [--json] [--dump-first] [--dump-batch] [--inspect-alignment] [--inspect-boundary-context]
        \\
    , .{});
}

fn balancedPositiveWeight(valid: u64, gold: u64) f64 {
    if (valid == 0 or gold == 0 or gold >= valid) return 0.0;
    return @as(f64, @floatFromInt(valid - gold)) / @as(f64, @floatFromInt(gold));
}

fn weightedPositivePrior(valid: u64, gold: u64, pos_weight: f64) f64 {
    if (valid == 0 or gold == 0 or !std.math.isFinite(pos_weight) or pos_weight <= 0.0) return 0.0;
    const positives = @as(f64, @floatFromInt(gold));
    const negatives = @as(f64, @floatFromInt(valid - gold));
    const weighted_positives = positives * pos_weight;
    const denom = negatives + weighted_positives;
    if (denom <= 0.0) return 0.0;
    return weighted_positives / denom;
}

fn printI32Array(values: []const i32) void {
    std.debug.print("[", .{});
    for (values, 0..) |v, i| {
        if (i > 0) std.debug.print(",", .{});
        std.debug.print("{d}", .{v});
    }
    std.debug.print("]", .{});
}

fn printUsizeArray(values: []const usize) void {
    std.debug.print("[", .{});
    for (values, 0..) |v, i| {
        if (i > 0) std.debug.print(",", .{});
        std.debug.print("{d}", .{v});
    }
    std.debug.print("]", .{});
}

fn printUsizeArrayWithOffset(values: []const usize, offset: usize) void {
    std.debug.print("[", .{});
    for (values, 0..) |v, i| {
        if (i > 0) std.debug.print(",", .{});
        std.debug.print("{d}", .{offset + v});
    }
    std.debug.print("]", .{});
}

fn printBoundaryPositions(labels: []const f32, mask: []const i32) void {
    std.debug.print("[", .{});
    var first = true;
    for (labels, 0..) |label, i| {
        if (i >= mask.len or mask[i] == 0) break;
        if (label <= 0.5) continue;
        if (!first) std.debug.print(",", .{});
        first = false;
        std.debug.print("{d}", .{i});
    }
    std.debug.print("]", .{});
}

fn printChunkSpans(starts: []const i32, ends: []const i32, mask: []const f32) void {
    std.debug.print("[", .{});
    for (starts, 0..) |start, i| {
        if (i > 0) std.debug.print(",", .{});
        const end = if (i < ends.len) ends[i] else 0;
        const m: u8 = if (i < mask.len and mask[i] > 0.5) 1 else 0;
        std.debug.print("{{\"idx\":{d},\"start\":{d},\"end\":{d},\"mask\":{d}}}", .{ i, start, end, m });
    }
    std.debug.print("]", .{});
}

fn printSourceChunkBoundaries(chunks: []const fused_chunker_data.FusedChunkBoundary) void {
    std.debug.print("[", .{});
    for (chunks, 0..) |chunk, i| {
        if (i > 0) std.debug.print(",", .{});
        std.debug.print(
            "{{\"idx\":{d},\"start_char\":{d},\"end_char\":{d},\"start_token\":{d},\"end_token\":{d}}}",
            .{ i, chunk.start_char, chunk.end_char, chunk.start_token, chunk.end_token },
        );
    }
    std.debug.print("]", .{});
}

fn printSourceTokenMappings(chunks: []const fused_chunker_data.FusedChunkBoundary, offsets: [][2]u32) void {
    std.debug.print("[", .{});
    for (chunks, 0..) |chunk, i| {
        if (i > 0) std.debug.print(",", .{});
        const span = fused_chunker_data.charToTokenBoundary(chunk.start_char, chunk.end_char, offsets);
        const resolved_start = if (chunk.start_token > 0) chunk.start_token else span.start_token;
        const resolved_end = if (chunk.end_token > 0) chunk.end_token else span.end_token;
        const valid: u8 = if (resolved_end > resolved_start) 1 else 0;
        const prev_start = if (span.end_token > 0 and span.end_token - 1 < offsets.len) offsets[span.end_token - 1][0] else 0;
        const prev_end = if (span.end_token > 0 and span.end_token - 1 < offsets.len) offsets[span.end_token - 1][1] else 0;
        const end_start = if (span.end_token < offsets.len) offsets[span.end_token][0] else 0;
        const end_end = if (span.end_token < offsets.len) offsets[span.end_token][1] else 0;
        std.debug.print(
            "{{\"idx\":{d},\"start_char\":{d},\"end_char\":{d},\"raw_start\":{d},\"raw_end\":{d},\"resolved_start\":{d},\"resolved_end\":{d},\"valid\":{d},\"prev_off\":[{d},{d}],\"end_off\":[{d},{d}]}}",
            .{
                i,
                chunk.start_char,
                chunk.end_char,
                span.start_token,
                span.end_token,
                resolved_start,
                resolved_end,
                valid,
                prev_start,
                prev_end,
                end_start,
                end_end,
            },
        );
    }
    std.debug.print("]", .{});
}

fn validTokenCount(mask: []const i32) usize {
    var n: usize = 0;
    for (mask) |m| {
        if (m == 0) break;
        n += 1;
    }
    return n;
}

const AlignmentInspection = struct {
    source_chunks: u64 = 0,
    source_valid_chunks: u64 = 0,
    source_invalid_chunks: u64 = 0,
    expected_post_first_boundaries: u64 = 0,
    active_boundary_labels: u64 = 0,
    labels_on_padding: u64 = 0,
    labels_not_at_resolved_chunk_start: u64 = 0,
    labels_on_first_chunk_start: u64 = 0,
    missing_label_at_resolved_chunk_start: u64 = 0,
    resolved_chunk_start_attention_zero: u64 = 0,
    valid_chunk_span_invalid: u64 = 0,
    stored_chunk_missing_label: u64 = 0,
    samples_with_label_count_mismatch: u64 = 0,
    samples_with_any_violation: u64 = 0,
};

fn alignmentInspectionPassed(inspection: AlignmentInspection) bool {
    return inspection.labels_on_padding == 0 and
        inspection.labels_not_at_resolved_chunk_start == 0 and
        inspection.labels_on_first_chunk_start == 0 and
        inspection.missing_label_at_resolved_chunk_start == 0 and
        inspection.resolved_chunk_start_attention_zero == 0 and
        inspection.valid_chunk_span_invalid == 0 and
        inspection.stored_chunk_missing_label == 0 and
        inspection.samples_with_label_count_mismatch == 0 and
        inspection.samples_with_any_violation == 0;
}

const NumericStats = struct {
    count: u64 = 0,
    sum: f64 = 0,
    min: f64 = std.math.inf(f64),
    max: f64 = -std.math.inf(f64),

    fn add(self: *NumericStats, value: f64) void {
        if (!std.math.isFinite(value)) return;
        self.count += 1;
        self.sum += value;
        self.min = @min(self.min, value);
        self.max = @max(self.max, value);
    }

    fn mean(self: NumericStats) f64 {
        return if (self.count == 0) 0 else self.sum / @as(f64, @floatFromInt(self.count));
    }

    fn minOrZero(self: NumericStats) f64 {
        return if (self.count == 0) 0 else self.min;
    }

    fn maxOrZero(self: NumericStats) f64 {
        return if (self.count == 0) 0 else self.max;
    }
};

const BoundaryContextInspection = struct {
    active_boundaries: u64 = 0,
    samples_with_active_boundaries: u64 = 0,
    start_char_past_text: u64 = 0,
    end_char_past_text: u64 = 0,
    whitespace_before: u64 = 0,
    newline_before: u64 = 0,
    line_start_like: u64 = 0,
    blank_line_before: u64 = 0,
    sentence_punctuation_before: u64 = 0,
    punctuation_before: u64 = 0,
    uppercase_start: u64 = 0,
    lowercase_start: u64 = 0,
    digit_start: u64 = 0,
    bullet_start: u64 = 0,
    token_position_first_quarter: u64 = 0,
    token_position_middle_half: u64 = 0,
    token_position_last_quarter: u64 = 0,
    token_position: NumericStats = .{},
    token_position_ratio: NumericStats = .{},
    previous_chunk_token_len: NumericStats = .{},
    current_chunk_token_len: NumericStats = .{},
    previous_chunk_char_len: NumericStats = .{},
    current_chunk_char_len: NumericStats = .{},
    source_gap_chars: NumericStats = .{},
    candidate_tokens: u64 = 0,
    candidate_sentence_punctuation_before: u64 = 0,
    candidate_sentence_punctuation_gold: u64 = 0,
    candidate_sentence_like_uppercase: u64 = 0,
    candidate_sentence_like_uppercase_gold: u64 = 0,
    candidate_uppercase_start: u64 = 0,
    candidate_uppercase_gold: u64 = 0,
    candidate_line_start_like: u64 = 0,
    candidate_line_start_gold: u64 = 0,

    fn rate(count: u64, denom: u64) f64 {
        return if (denom == 0) 0 else @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(denom));
    }
};

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

fn isLineStartLike(text: []const u8, start: usize) bool {
    var i = @min(start, text.len);
    while (i > 0) {
        const c = text[i - 1];
        if (c == '\n') return true;
        if (c == ' ' or c == '\t' or c == '\r') {
            i -= 1;
            continue;
        }
        return false;
    }
    return true;
}

fn hasBlankLineBefore(text: []const u8, start: usize) bool {
    var i = @min(start, text.len);
    var newline_count: u8 = 0;
    while (i > 0) {
        i -= 1;
        const c = text[i];
        if (c == '\n') {
            newline_count += 1;
            if (newline_count >= 2) return true;
            continue;
        }
        if (c == ' ' or c == '\t' or c == '\r') continue;
        return false;
    }
    return false;
}

fn isSentencePunctuation(c: u8) bool {
    return c == '.' or c == '?' or c == '!';
}

fn isBoundaryPunctuation(c: u8) bool {
    return isSentencePunctuation(c) or c == ':' or c == ';' or c == ')' or c == ']' or c == '}';
}

fn inspectBoundaryContext(
    inspection: *BoundaryContextInspection,
    sample: *const fused_chunker_data.FusedSample,
    labels: []const f32,
    attention_mask: []const i32,
    offsets: [][2]u32,
) void {
    var sample_active = false;
    const valid_tokens = validTokenCount(attention_mask);

    var valid_chunk_idx: usize = 0;
    var prev_start_token: u32 = 0;
    var prev_end_token: u32 = 0;
    var prev_start_char: u32 = 0;
    var prev_end_char: u32 = 0;

    for (sample.chunk_boundaries) |boundary| {
        if (boundary.start_char > sample.text.len) inspection.start_char_past_text += 1;
        if (boundary.end_char > sample.text.len) inspection.end_char_past_text += 1;

        const span = fused_chunker_data.charToTokenBoundary(boundary.start_char, boundary.end_char, offsets);
        const resolved_start = if (boundary.start_token > 0) boundary.start_token else span.start_token;
        const resolved_end = if (boundary.end_token > 0) boundary.end_token else span.end_token;
        if (resolved_end <= resolved_start) continue;

        const start_idx: usize = @intCast(resolved_start);
        const label_active = valid_chunk_idx > 0 and
            start_idx < labels.len and
            start_idx < attention_mask.len and
            attention_mask[start_idx] != 0 and
            labels[start_idx] > 0.5;

        if (label_active) {
            inspection.active_boundaries += 1;
            sample_active = true;

            const start_char: usize = @min(@as(usize, @intCast(boundary.start_char)), sample.text.len);
            if (start_char > 0 and std.ascii.isWhitespace(sample.text[start_char - 1])) inspection.whitespace_before += 1;
            if (start_char > 0 and sample.text[start_char - 1] == '\n') inspection.newline_before += 1;
            if (isLineStartLike(sample.text, start_char)) inspection.line_start_like += 1;
            if (hasBlankLineBefore(sample.text, start_char)) inspection.blank_line_before += 1;
            if (previousNonWhitespaceByte(sample.text, start_char)) |prev| {
                if (isSentencePunctuation(prev)) inspection.sentence_punctuation_before += 1;
                if (isBoundaryPunctuation(prev)) inspection.punctuation_before += 1;
            }
            if (firstNonWhitespaceByteAtOrAfter(sample.text, start_char)) |first| {
                if (std.ascii.isUpper(first)) inspection.uppercase_start += 1;
                if (std.ascii.isLower(first)) inspection.lowercase_start += 1;
                if (std.ascii.isDigit(first)) inspection.digit_start += 1;
                if (first == '-' or first == '*' or first == '#') inspection.bullet_start += 1;
            }

            const token_pos_f: f64 = @floatFromInt(start_idx);
            inspection.token_position.add(token_pos_f);
            const ratio = if (valid_tokens == 0) 0 else token_pos_f / @as(f64, @floatFromInt(valid_tokens));
            inspection.token_position_ratio.add(ratio);
            if (ratio < 0.25) {
                inspection.token_position_first_quarter += 1;
            } else if (ratio < 0.75) {
                inspection.token_position_middle_half += 1;
            } else {
                inspection.token_position_last_quarter += 1;
            }

            inspection.previous_chunk_token_len.add(@floatFromInt(prev_end_token - prev_start_token));
            inspection.current_chunk_token_len.add(@floatFromInt(resolved_end - resolved_start));
            inspection.previous_chunk_char_len.add(@floatFromInt(prev_end_char - prev_start_char));
            inspection.current_chunk_char_len.add(@floatFromInt(boundary.end_char - boundary.start_char));
            const source_gap = if (boundary.start_char >= prev_end_char)
                boundary.start_char - prev_end_char
            else
                0;
            inspection.source_gap_chars.add(@floatFromInt(source_gap));
        }

        prev_start_token = resolved_start;
        prev_end_token = resolved_end;
        prev_start_char = boundary.start_char;
        prev_end_char = boundary.end_char;
        valid_chunk_idx += 1;
    }

    if (sample_active) inspection.samples_with_active_boundaries += 1;
}

fn inspectBoundaryCandidateContext(
    inspection: *BoundaryContextInspection,
    sample: *const fused_chunker_data.FusedSample,
    labels: []const f32,
    attention_mask: []const i32,
    offsets: [][2]u32,
) void {
    const token_count = @min(@min(labels.len, attention_mask.len), offsets.len);
    for (0..token_count) |token_idx| {
        if (attention_mask[token_idx] == 0) continue;
        const off = offsets[token_idx];
        if (off[0] == 0 and off[1] == 0) continue;
        if (off[0] >= sample.text.len) continue;

        inspection.candidate_tokens += 1;
        const is_gold = labels[token_idx] > 0.5;
        const start_char: usize = @intCast(off[0]);
        const sentence_punctuation = if (previousNonWhitespaceByte(sample.text, start_char)) |prev|
            isSentencePunctuation(prev)
        else
            false;
        const uppercase_start = if (firstNonWhitespaceByteAtOrAfter(sample.text, start_char)) |first|
            std.ascii.isUpper(first)
        else
            false;
        const line_start = isLineStartLike(sample.text, start_char);

        if (sentence_punctuation) {
            inspection.candidate_sentence_punctuation_before += 1;
            if (is_gold) inspection.candidate_sentence_punctuation_gold += 1;
        }
        if (sentence_punctuation and uppercase_start) {
            inspection.candidate_sentence_like_uppercase += 1;
            if (is_gold) inspection.candidate_sentence_like_uppercase_gold += 1;
        }
        if (uppercase_start) {
            inspection.candidate_uppercase_start += 1;
            if (is_gold) inspection.candidate_uppercase_gold += 1;
        }
        if (line_start) {
            inspection.candidate_line_start_like += 1;
            if (is_gold) inspection.candidate_line_start_gold += 1;
        }
    }
}

fn inspectSampleAlignment(
    inspection: *AlignmentInspection,
    sample: *const fused_chunker_data.FusedSample,
    labels: []const f32,
    attention_mask: []const i32,
    chunk_starts: []const i32,
    chunk_ends: []const i32,
    chunk_mask: []const f32,
    offsets: [][2]u32,
) void {
    var sample_expected: u64 = 0;
    var sample_active_labels: u64 = 0;
    var sample_violation = false;

    for (labels, 0..) |label, t| {
        if (label <= 0.5) continue;
        sample_active_labels += 1;
        inspection.active_boundary_labels += 1;
        if (t >= attention_mask.len or attention_mask[t] == 0) {
            inspection.labels_on_padding += 1;
            sample_violation = true;
        }
    }

    var valid_chunk_idx: usize = 0;
    for (sample.chunk_boundaries) |boundary| {
        inspection.source_chunks += 1;
        const span = fused_chunker_data.charToTokenBoundary(boundary.start_char, boundary.end_char, offsets);
        const resolved_start = if (boundary.start_token > 0) boundary.start_token else span.start_token;
        const resolved_end = if (boundary.end_token > 0) boundary.end_token else span.end_token;
        if (resolved_end <= resolved_start) {
            inspection.source_invalid_chunks += 1;
            continue;
        }

        inspection.source_valid_chunks += 1;
        const start_usize: usize = @intCast(resolved_start);
        if (start_usize >= attention_mask.len or attention_mask[start_usize] == 0) {
            inspection.resolved_chunk_start_attention_zero += 1;
            sample_violation = true;
        }

        if (valid_chunk_idx == 0) {
            if (start_usize < labels.len and labels[start_usize] > 0.5) {
                inspection.labels_on_first_chunk_start += 1;
                sample_violation = true;
            }
        } else {
            sample_expected += 1;
            inspection.expected_post_first_boundaries += 1;
            if (start_usize >= labels.len or labels[start_usize] <= 0.5) {
                inspection.missing_label_at_resolved_chunk_start += 1;
                sample_violation = true;
            }
        }
        valid_chunk_idx += 1;
    }

    for (labels, 0..) |label, token_idx| {
        if (label <= 0.5) continue;
        var matches_resolved_start = false;
        var valid_idx: usize = 0;
        for (sample.chunk_boundaries) |boundary| {
            const span = fused_chunker_data.charToTokenBoundary(boundary.start_char, boundary.end_char, offsets);
            const resolved_start = if (boundary.start_token > 0) boundary.start_token else span.start_token;
            const resolved_end = if (boundary.end_token > 0) boundary.end_token else span.end_token;
            if (resolved_end <= resolved_start) continue;
            if (valid_idx > 0 and @as(usize, @intCast(resolved_start)) == token_idx) {
                matches_resolved_start = true;
                break;
            }
            valid_idx += 1;
        }
        if (!matches_resolved_start) {
            inspection.labels_not_at_resolved_chunk_start += 1;
            sample_violation = true;
        }
    }

    for (chunk_mask, 0..) |m, i| {
        if (m <= 0.5) continue;
        if (i >= chunk_starts.len or i >= chunk_ends.len or chunk_ends[i] <= chunk_starts[i]) {
            inspection.valid_chunk_span_invalid += 1;
            sample_violation = true;
            continue;
        }
        if (i > 0) {
            const start = chunk_starts[i];
            if (start < 0) {
                inspection.stored_chunk_missing_label += 1;
                sample_violation = true;
                continue;
            }
            const start_idx: usize = @intCast(start);
            if (start_idx >= labels.len or labels[start_idx] <= 0.5) {
                inspection.stored_chunk_missing_label += 1;
                sample_violation = true;
            }
        }
    }

    if (sample_active_labels != sample_expected) {
        inspection.samples_with_label_count_mismatch += 1;
        sample_violation = true;
    }
    if (sample_violation) inspection.samples_with_any_violation += 1;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    var data_path: ?[]const u8 = null;
    var model_dir: ?[]const u8 = null;
    var split: ?[]const u8 = null;
    var max_seq_len: usize = 384;
    var max_chunks: usize = 32;
    var batch_size: usize = 8;
    var sample_offset: usize = 0;
    var limit: usize = 0;
    var dump_first = false;
    var dump_batch = false;
    var json_output = false;
    var inspect_alignment = false;
    var inspect_boundary_context_flag = false;
    var analysis_pos_weight: f64 = 0.0;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--data")) {
            data_path = args.next() orelse return error.MissingData;
        } else if (std.mem.eql(u8, arg, "--model-dir")) {
            model_dir = args.next() orelse return error.MissingModelDir;
        } else if (std.mem.eql(u8, arg, "--split")) {
            split = args.next() orelse return error.MissingSplit;
        } else if (std.mem.eql(u8, arg, "--max-seq-len")) {
            max_seq_len = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingMaxSeqLen, 10);
        } else if (std.mem.eql(u8, arg, "--max-chunks")) {
            max_chunks = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingMaxChunks, 10);
        } else if (std.mem.eql(u8, arg, "--batch-size")) {
            batch_size = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingBatchSize, 10);
        } else if (std.mem.eql(u8, arg, "--offset")) {
            sample_offset = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingOffset, 10);
        } else if (std.mem.eql(u8, arg, "--limit")) {
            limit = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingLimit, 10);
        } else if (std.mem.eql(u8, arg, "--analysis-pos-weight")) {
            analysis_pos_weight = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingAnalysisPosWeight);
        } else if (std.mem.eql(u8, arg, "--dump-first")) {
            dump_first = true;
        } else if (std.mem.eql(u8, arg, "--dump-batch")) {
            dump_batch = true;
        } else if (std.mem.eql(u8, arg, "--json")) {
            json_output = true;
        } else if (std.mem.eql(u8, arg, "--inspect-alignment")) {
            inspect_alignment = true;
        } else if (std.mem.eql(u8, arg, "--inspect-boundary-context")) {
            inspect_boundary_context_flag = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            usage();
            return;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }

    const path = data_path orelse {
        usage();
        return error.MissingData;
    };
    const mdir = model_dir orelse {
        usage();
        return error.MissingModelDir;
    };

    var loaded = try fused_chunker_data.loadSamples(allocator, path, split);
    defer loaded.deinit();
    if (sample_offset > loaded.samples.len) return error.InvalidOffset;
    const sample_end = if (limit > 0)
        @min(sample_offset + limit, loaded.samples.len)
    else
        loaded.samples.len;
    const samples = loaded.samples[sample_offset..sample_end];
    const dataset_stats = fused_chunker_data.computeStats(samples);

    var tokenizer = try TokenizerBatch.loadFromDir(allocator, mdir, max_seq_len);
    defer tokenizer.deinit();
    var tok_ctx = tokenizer.makeTokenFnCtx();

    var valid: u64 = 0;
    var gold: u64 = 0;
    var batches: u64 = 0;
    var ids_hash: u64 = fnv_offset;
    var mask_hash: u64 = fnv_offset;
    var offsets_hash: u64 = fnv_offset;
    var labels_hash: u64 = fnv_offset;
    var chunks_hash: u64 = fnv_offset;
    var sample_indices_hash: u64 = fnv_offset;
    var samples_with_active_boundary_labels: u64 = 0;
    var alignment_inspection = AlignmentInspection{};
    var boundary_context_inspection = BoundaryContextInspection{};

    const offsets = try allocator.alloc([2]u32, max_seq_len);
    defer allocator.free(offsets);
    const ids = try allocator.alloc(i32, max_seq_len);
    defer allocator.free(ids);
    const mask = try allocator.alloc(i32, max_seq_len);
    defer allocator.free(mask);

    var start: usize = 0;
    while (start < samples.len) {
        const end = @min(start + batch_size, samples.len);
        const count = end - start;
        const indices = try allocator.alloc(usize, count);
        defer allocator.free(indices);
        for (indices, 0..) |*idx, i| idx.* = start + i;
        for (indices) |idx| hashU64(&sample_indices_hash, sample_offset + idx);

        var batch = try fused_chunker_data.assembleTokenBatch(
            allocator,
            samples,
            indices,
            max_seq_len,
            max_chunks,
            &tok_ctx,
            TokenFnCtx.call,
        );
        defer batch.deinit(allocator);

        batches += 1;
        const total = count * max_seq_len;
        var batch_gold_by_sample = try allocator.alloc(u64, count);
        defer allocator.free(batch_gold_by_sample);
        @memset(batch_gold_by_sample, 0);
        for (0..total) |i| {
            hashI32(&ids_hash, batch.input_ids[i]);
            hashI32(&mask_hash, batch.attention_mask[i]);
            hashU64(&labels_hash, if (batch.boundary_labels[i] > 0.5) 1 else 0);
            if (batch.attention_mask[i] != 0) {
                valid += 1;
                if (batch.boundary_labels[i] > 0.5) {
                    gold += 1;
                    batch_gold_by_sample[i / max_seq_len] += 1;
                }
            }
        }
        for (batch_gold_by_sample) |sample_gold| {
            if (sample_gold > 0) samples_with_active_boundary_labels += 1;
        }
        for (0..count * max_chunks) |i| {
            hashI32(&chunks_hash, batch.chunk_starts[i]);
            hashI32(&chunks_hash, batch.chunk_ends[i]);
            hashU64(&chunks_hash, if (batch.chunk_mask[i] > 0.5) 1 else 0);
        }
        if (inspect_alignment or inspect_boundary_context_flag) {
            for (0..count) |bi| {
                const sample_index = batch.sample_indices[bi];
                const seq_start = bi * max_seq_len;
                const chunk_start = bi * max_chunks;
                @memset(offsets, .{ 0, 0 });
                @memset(ids, 0);
                @memset(mask, 0);
                const n_tokens = TokenFnCtx.call(&tok_ctx, samples[sample_index].text, ids, mask, offsets);
                const active_offsets = offsets[0..@min(n_tokens, max_seq_len)];
                inspectSampleAlignment(
                    &alignment_inspection,
                    &samples[sample_index],
                    batch.boundary_labels[seq_start .. seq_start + max_seq_len],
                    batch.attention_mask[seq_start .. seq_start + max_seq_len],
                    batch.chunk_starts[chunk_start .. chunk_start + max_chunks],
                    batch.chunk_ends[chunk_start .. chunk_start + max_chunks],
                    batch.chunk_mask[chunk_start .. chunk_start + max_chunks],
                    active_offsets,
                );
                if (inspect_boundary_context_flag) {
                    inspectBoundaryContext(
                        &boundary_context_inspection,
                        &samples[sample_index],
                        batch.boundary_labels[seq_start .. seq_start + max_seq_len],
                        batch.attention_mask[seq_start .. seq_start + max_seq_len],
                        active_offsets,
                    );
                    inspectBoundaryCandidateContext(
                        &boundary_context_inspection,
                        &samples[sample_index],
                        batch.boundary_labels[seq_start .. seq_start + max_seq_len],
                        batch.attention_mask[seq_start .. seq_start + max_seq_len],
                        active_offsets,
                    );
                }
            }
        }

        start = end;
    }

    for (samples) |sample| {
        @memset(offsets, .{ 0, 0 });
        @memset(ids, 0);
        @memset(mask, 0);
        _ = TokenFnCtx.call(&tok_ctx, sample.text, ids, mask, offsets);
        for (offsets) |off| {
            hashU64(&offsets_hash, off[0]);
            hashU64(&offsets_hash, off[1]);
        }
    }

    const gold_rate = if (valid == 0) 0.0 else @as(f64, @floatFromInt(gold)) / @as(f64, @floatFromInt(valid));
    const non_boundary_tokens = valid - gold;
    const balanced_pos_weight = balancedPositiveWeight(valid, gold);
    const weighted_prior_pos1 = weightedPositivePrior(valid, gold, 1.0);
    const weighted_prior_pos5 = weightedPositivePrior(valid, gold, 5.0);
    const weighted_prior_balanced = weightedPositivePrior(valid, gold, balanced_pos_weight);
    const weighted_prior_analysis = weightedPositivePrior(valid, gold, analysis_pos_weight);
    if (json_output) {
        std.debug.print(
            "{{\"tool\":\"zig_fused_tokenization_parity\",\"schema_version\":3,\"offset\":{d},\"samples\":{d},\"batches\":{d},\"max_seq_len\":{d},\"max_chunks\":{d},\"valid_tokens\":{d},\"boundary_gold_tokens\":{d},\"non_boundary_tokens\":{d},\"gold_rate\":{d:.9},\"balanced_pos_weight\":{d:.9},\"weighted_positive_prior_pos_weight_1\":{d:.9},\"weighted_positive_prior_pos_weight_5\":{d:.9},\"weighted_positive_prior_balanced\":{d:.9},\"analysis_pos_weight\":{d:.9},\"weighted_positive_prior_analysis\":{d:.9},\"samples_with_active_boundary_labels\":{d},\"contrastive_pos_samples\":{d},\"boundary_target_samples\":{d},\"boundary_targets\":{d},\"hashes\":{{\"sample_indices\":\"{x}\",\"input_ids\":\"{x}\",\"attention_mask\":\"{x}\",\"offsets\":\"{x}\",\"labels\":\"{x}\",\"chunks\":\"{x}\"}}",
            .{
                sample_offset,
                samples.len,
                batches,
                max_seq_len,
                max_chunks,
                valid,
                gold,
                non_boundary_tokens,
                gold_rate,
                balanced_pos_weight,
                weighted_prior_pos1,
                weighted_prior_pos5,
                weighted_prior_balanced,
                analysis_pos_weight,
                weighted_prior_analysis,
                samples_with_active_boundary_labels,
                dataset_stats.samples_with_contrastive_positives,
                dataset_stats.samples_with_boundary_targets,
                dataset_stats.total_boundary_targets,
                sample_indices_hash,
                ids_hash,
                mask_hash,
                offsets_hash,
                labels_hash,
                chunks_hash,
            },
        );
        if (inspect_alignment) {
            const status: []const u8 = if (alignmentInspectionPassed(alignment_inspection)) "passed" else "failed";
            std.debug.print(
                ",\"alignment_inspection\":{{\"status\":\"{s}\",\"source_chunks\":{d},\"source_valid_chunks\":{d},\"source_invalid_chunks\":{d},\"expected_post_first_boundaries\":{d},\"active_boundary_labels\":{d},\"labels_on_padding\":{d},\"labels_not_at_resolved_chunk_start\":{d},\"labels_on_first_chunk_start\":{d},\"missing_label_at_resolved_chunk_start\":{d},\"resolved_chunk_start_attention_zero\":{d},\"valid_chunk_span_invalid\":{d},\"stored_chunk_missing_label\":{d},\"samples_with_label_count_mismatch\":{d},\"samples_with_any_violation\":{d}}}",
                .{
                    status,
                    alignment_inspection.source_chunks,
                    alignment_inspection.source_valid_chunks,
                    alignment_inspection.source_invalid_chunks,
                    alignment_inspection.expected_post_first_boundaries,
                    alignment_inspection.active_boundary_labels,
                    alignment_inspection.labels_on_padding,
                    alignment_inspection.labels_not_at_resolved_chunk_start,
                    alignment_inspection.labels_on_first_chunk_start,
                    alignment_inspection.missing_label_at_resolved_chunk_start,
                    alignment_inspection.resolved_chunk_start_attention_zero,
                    alignment_inspection.valid_chunk_span_invalid,
                    alignment_inspection.stored_chunk_missing_label,
                    alignment_inspection.samples_with_label_count_mismatch,
                    alignment_inspection.samples_with_any_violation,
                },
            );
        }
        if (inspect_boundary_context_flag) {
            const denom = boundary_context_inspection.active_boundaries;
            std.debug.print(
                ",\"boundary_context\":{{\"active_boundaries\":{d},\"samples_with_active_boundaries\":{d},\"start_char_past_text\":{d},\"end_char_past_text\":{d}",
                .{
                    boundary_context_inspection.active_boundaries,
                    boundary_context_inspection.samples_with_active_boundaries,
                    boundary_context_inspection.start_char_past_text,
                    boundary_context_inspection.end_char_past_text,
                },
            );
            std.debug.print(
                ",\"rates\":{{\"whitespace_before\":{d:.9},\"newline_before\":{d:.9},\"line_start_like\":{d:.9},\"blank_line_before\":{d:.9},\"sentence_punctuation_before\":{d:.9},\"punctuation_before\":{d:.9},\"uppercase_start\":{d:.9},\"lowercase_start\":{d:.9},\"digit_start\":{d:.9},\"bullet_start\":{d:.9},\"token_position_first_quarter\":{d:.9},\"token_position_middle_half\":{d:.9},\"token_position_last_quarter\":{d:.9}}}",
                .{
                    BoundaryContextInspection.rate(boundary_context_inspection.whitespace_before, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.newline_before, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.line_start_like, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.blank_line_before, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.sentence_punctuation_before, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.punctuation_before, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.uppercase_start, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.lowercase_start, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.digit_start, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.bullet_start, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.token_position_first_quarter, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.token_position_middle_half, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.token_position_last_quarter, denom),
                },
            );
            std.debug.print(
                ",\"token_position\":{{\"mean\":{d:.6},\"min\":{d:.6},\"max\":{d:.6}}},\"token_position_ratio\":{{\"mean\":{d:.9},\"min\":{d:.9},\"max\":{d:.9}}}",
                .{
                    boundary_context_inspection.token_position.mean(),
                    boundary_context_inspection.token_position.minOrZero(),
                    boundary_context_inspection.token_position.maxOrZero(),
                    boundary_context_inspection.token_position_ratio.mean(),
                    boundary_context_inspection.token_position_ratio.minOrZero(),
                    boundary_context_inspection.token_position_ratio.maxOrZero(),
                },
            );
            std.debug.print(
                ",\"previous_chunk_token_len\":{{\"mean\":{d:.6},\"min\":{d:.6},\"max\":{d:.6}}},\"current_chunk_token_len\":{{\"mean\":{d:.6},\"min\":{d:.6},\"max\":{d:.6}}}",
                .{
                    boundary_context_inspection.previous_chunk_token_len.mean(),
                    boundary_context_inspection.previous_chunk_token_len.minOrZero(),
                    boundary_context_inspection.previous_chunk_token_len.maxOrZero(),
                    boundary_context_inspection.current_chunk_token_len.mean(),
                    boundary_context_inspection.current_chunk_token_len.minOrZero(),
                    boundary_context_inspection.current_chunk_token_len.maxOrZero(),
                },
            );
            std.debug.print(
                ",\"previous_chunk_char_len\":{{\"mean\":{d:.6},\"min\":{d:.6},\"max\":{d:.6}}},\"current_chunk_char_len\":{{\"mean\":{d:.6},\"min\":{d:.6},\"max\":{d:.6}}},\"source_gap_chars\":{{\"mean\":{d:.6},\"min\":{d:.6},\"max\":{d:.6}}}}}",
                .{
                    boundary_context_inspection.previous_chunk_char_len.mean(),
                    boundary_context_inspection.previous_chunk_char_len.minOrZero(),
                    boundary_context_inspection.previous_chunk_char_len.maxOrZero(),
                    boundary_context_inspection.current_chunk_char_len.mean(),
                    boundary_context_inspection.current_chunk_char_len.minOrZero(),
                    boundary_context_inspection.current_chunk_char_len.maxOrZero(),
                    boundary_context_inspection.source_gap_chars.mean(),
                    boundary_context_inspection.source_gap_chars.minOrZero(),
                    boundary_context_inspection.source_gap_chars.maxOrZero(),
                },
            );
            std.debug.print(
                ",\"candidate_context\":{{\"candidate_tokens\":{d},\"sentence_punctuation_before\":{{\"count\":{d},\"gold\":{d},\"gold_rate\":{d:.9},\"gold_coverage\":{d:.9}}},\"sentence_like_uppercase\":{{\"count\":{d},\"gold\":{d},\"gold_rate\":{d:.9},\"gold_coverage\":{d:.9}}},\"uppercase_start\":{{\"count\":{d},\"gold\":{d},\"gold_rate\":{d:.9},\"gold_coverage\":{d:.9}}},\"line_start_like\":{{\"count\":{d},\"gold\":{d},\"gold_rate\":{d:.9},\"gold_coverage\":{d:.9}}}}}",
                .{
                    boundary_context_inspection.candidate_tokens,
                    boundary_context_inspection.candidate_sentence_punctuation_before,
                    boundary_context_inspection.candidate_sentence_punctuation_gold,
                    BoundaryContextInspection.rate(boundary_context_inspection.candidate_sentence_punctuation_gold, boundary_context_inspection.candidate_sentence_punctuation_before),
                    BoundaryContextInspection.rate(boundary_context_inspection.candidate_sentence_punctuation_gold, denom),
                    boundary_context_inspection.candidate_sentence_like_uppercase,
                    boundary_context_inspection.candidate_sentence_like_uppercase_gold,
                    BoundaryContextInspection.rate(boundary_context_inspection.candidate_sentence_like_uppercase_gold, boundary_context_inspection.candidate_sentence_like_uppercase),
                    BoundaryContextInspection.rate(boundary_context_inspection.candidate_sentence_like_uppercase_gold, denom),
                    boundary_context_inspection.candidate_uppercase_start,
                    boundary_context_inspection.candidate_uppercase_gold,
                    BoundaryContextInspection.rate(boundary_context_inspection.candidate_uppercase_gold, boundary_context_inspection.candidate_uppercase_start),
                    BoundaryContextInspection.rate(boundary_context_inspection.candidate_uppercase_gold, denom),
                    boundary_context_inspection.candidate_line_start_like,
                    boundary_context_inspection.candidate_line_start_gold,
                    BoundaryContextInspection.rate(boundary_context_inspection.candidate_line_start_gold, boundary_context_inspection.candidate_line_start_like),
                    BoundaryContextInspection.rate(boundary_context_inspection.candidate_line_start_gold, denom),
                },
            );
        }
    } else {
        std.debug.print(
            "zig offset={d} samples={d} batches={d} valid={d} gold={d} non_boundary={d} gold_rate={d:.6} balanced_pos_weight={d:.6} weighted_prior_pos1={d:.6} weighted_prior_pos5={d:.6} weighted_prior_balanced={d:.6} analysis_pos_weight={d:.6} weighted_prior_analysis={d:.6} active_boundary_samples={d} contrastive_pos_samples={d} boundary_target_samples={d} boundary_targets={d} sample_indices_hash={x} ids_hash={x} mask_hash={x} offsets_hash={x} labels_hash={x} chunks_hash={x}\n",
            .{
                sample_offset,
                samples.len,
                batches,
                valid,
                gold,
                non_boundary_tokens,
                gold_rate,
                balanced_pos_weight,
                weighted_prior_pos1,
                weighted_prior_pos5,
                weighted_prior_balanced,
                analysis_pos_weight,
                weighted_prior_analysis,
                samples_with_active_boundary_labels,
                dataset_stats.samples_with_contrastive_positives,
                dataset_stats.samples_with_boundary_targets,
                dataset_stats.total_boundary_targets,
                sample_indices_hash,
                ids_hash,
                mask_hash,
                offsets_hash,
                labels_hash,
                chunks_hash,
            },
        );
        if (inspect_alignment) {
            std.debug.print(
                "alignment_inspection status={s} source_chunks={d} source_valid_chunks={d} source_invalid_chunks={d} expected_post_first_boundaries={d} active_boundary_labels={d} labels_on_padding={d} labels_not_at_resolved_chunk_start={d} labels_on_first_chunk_start={d} missing_label_at_resolved_chunk_start={d} resolved_chunk_start_attention_zero={d} valid_chunk_span_invalid={d} stored_chunk_missing_label={d} samples_with_label_count_mismatch={d} samples_with_any_violation={d}\n",
                .{
                    if (alignmentInspectionPassed(alignment_inspection)) "passed" else "failed",
                    alignment_inspection.source_chunks,
                    alignment_inspection.source_valid_chunks,
                    alignment_inspection.source_invalid_chunks,
                    alignment_inspection.expected_post_first_boundaries,
                    alignment_inspection.active_boundary_labels,
                    alignment_inspection.labels_on_padding,
                    alignment_inspection.labels_not_at_resolved_chunk_start,
                    alignment_inspection.labels_on_first_chunk_start,
                    alignment_inspection.missing_label_at_resolved_chunk_start,
                    alignment_inspection.resolved_chunk_start_attention_zero,
                    alignment_inspection.valid_chunk_span_invalid,
                    alignment_inspection.stored_chunk_missing_label,
                    alignment_inspection.samples_with_label_count_mismatch,
                    alignment_inspection.samples_with_any_violation,
                },
            );
        }
        if (inspect_boundary_context_flag) {
            const denom = boundary_context_inspection.active_boundaries;
            std.debug.print(
                "boundary_context active={d} samples={d} start_char_past_text={d} end_char_past_text={d} whitespace_before={d:.4} newline_before={d:.4} line_start_like={d:.4} blank_line_before={d:.4} sentence_punctuation_before={d:.4} punctuation_before={d:.4} uppercase_start={d:.4} lowercase_start={d:.4} digit_start={d:.4} token_pos_mean={d:.2} token_pos_ratio_mean={d:.4} prev_chunk_tokens_mean={d:.2} current_chunk_tokens_mean={d:.2} prev_chunk_chars_mean={d:.2} current_chunk_chars_mean={d:.2} source_gap_chars_mean={d:.2}\n",
                .{
                    boundary_context_inspection.active_boundaries,
                    boundary_context_inspection.samples_with_active_boundaries,
                    boundary_context_inspection.start_char_past_text,
                    boundary_context_inspection.end_char_past_text,
                    BoundaryContextInspection.rate(boundary_context_inspection.whitespace_before, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.newline_before, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.line_start_like, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.blank_line_before, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.sentence_punctuation_before, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.punctuation_before, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.uppercase_start, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.lowercase_start, denom),
                    BoundaryContextInspection.rate(boundary_context_inspection.digit_start, denom),
                    boundary_context_inspection.token_position.mean(),
                    boundary_context_inspection.token_position_ratio.mean(),
                    boundary_context_inspection.previous_chunk_token_len.mean(),
                    boundary_context_inspection.current_chunk_token_len.mean(),
                    boundary_context_inspection.previous_chunk_char_len.mean(),
                    boundary_context_inspection.current_chunk_char_len.mean(),
                    boundary_context_inspection.source_gap_chars.mean(),
                },
            );
            std.debug.print(
                "candidate_context tokens={d} sentence_punctuation count={d} gold={d} gold_rate={d:.6} sentence_like_uppercase count={d} gold={d} gold_rate={d:.6} uppercase count={d} gold={d} gold_rate={d:.6} line_start count={d} gold={d} gold_rate={d:.6}\n",
                .{
                    boundary_context_inspection.candidate_tokens,
                    boundary_context_inspection.candidate_sentence_punctuation_before,
                    boundary_context_inspection.candidate_sentence_punctuation_gold,
                    BoundaryContextInspection.rate(boundary_context_inspection.candidate_sentence_punctuation_gold, boundary_context_inspection.candidate_sentence_punctuation_before),
                    boundary_context_inspection.candidate_sentence_like_uppercase,
                    boundary_context_inspection.candidate_sentence_like_uppercase_gold,
                    BoundaryContextInspection.rate(boundary_context_inspection.candidate_sentence_like_uppercase_gold, boundary_context_inspection.candidate_sentence_like_uppercase),
                    boundary_context_inspection.candidate_uppercase_start,
                    boundary_context_inspection.candidate_uppercase_gold,
                    BoundaryContextInspection.rate(boundary_context_inspection.candidate_uppercase_gold, boundary_context_inspection.candidate_uppercase_start),
                    boundary_context_inspection.candidate_line_start_like,
                    boundary_context_inspection.candidate_line_start_gold,
                    BoundaryContextInspection.rate(boundary_context_inspection.candidate_line_start_gold, boundary_context_inspection.candidate_line_start_like),
                },
            );
        }
    }

    if (dump_first and samples.len > 0) {
        const one = [_]usize{0};
        var batch = try fused_chunker_data.assembleTokenBatch(
            allocator,
            samples,
            &one,
            max_seq_len,
            max_chunks,
            &tok_ctx,
            TokenFnCtx.call,
        );
        defer batch.deinit(allocator);

        var valid_first: usize = 0;
        for (batch.attention_mask[0..max_seq_len]) |m| {
            if (m == 0) break;
            valid_first += 1;
        }
        if (json_output) {
            std.debug.print(",\"first_sample\":{{\"valid_tokens\":{d},\"chunks\":[", .{valid_first});
            for (0..@min(samples[0].chunk_boundaries.len, max_chunks)) |i| {
                if (i > 0) std.debug.print(",", .{});
                const label = if (batch.chunk_starts[i] >= 0 and @as(usize, @intCast(batch.chunk_starts[i])) < max_seq_len)
                    batch.boundary_labels[@intCast(batch.chunk_starts[i])]
                else
                    0.0;
                std.debug.print(
                    "{{\"idx\":{d},\"start_char\":{d},\"end_char\":{d},\"start_token\":{d},\"end_token\":{d},\"mask\":{d},\"label_at_start\":{d}}}",
                    .{
                        i,
                        samples[0].chunk_boundaries[i].start_char,
                        samples[0].chunk_boundaries[i].end_char,
                        batch.chunk_starts[i],
                        batch.chunk_ends[i],
                        batch.chunk_mask[i],
                        label,
                    },
                );
            }
            std.debug.print("],\"boundary_positions\":[", .{});
            var first_position = true;
            for (0..valid_first) |i| {
                if (batch.boundary_labels[i] <= 0.5) continue;
                if (!first_position) std.debug.print(",", .{});
                first_position = false;
                std.debug.print("{d}", .{i});
            }
            std.debug.print("]}}", .{});
        } else {
            std.debug.print("first valid={d} chunks={d}\n", .{ valid_first, samples[0].chunk_boundaries.len });
            for (0..@min(samples[0].chunk_boundaries.len, max_chunks)) |i| {
                const label = if (batch.chunk_starts[i] >= 0 and @as(usize, @intCast(batch.chunk_starts[i])) < max_seq_len)
                    batch.boundary_labels[@intCast(batch.chunk_starts[i])]
                else
                    0.0;
                std.debug.print(
                    "first chunk {d} char=[{d},{d}) token=[{d},{d}) mask={d} label_at_start={d}\n",
                    .{
                        i,
                        samples[0].chunk_boundaries[i].start_char,
                        samples[0].chunk_boundaries[i].end_char,
                        batch.chunk_starts[i],
                        batch.chunk_ends[i],
                        batch.chunk_mask[i],
                        label,
                    },
                );
            }
        }
    }
    if (dump_batch and samples.len > 0) {
        const count = @min(batch_size, samples.len);
        const indices = try allocator.alloc(usize, count);
        defer allocator.free(indices);
        for (indices, 0..) |*idx, i| idx.* = i;
        var batch = try fused_chunker_data.assembleTokenBatch(
            allocator,
            samples,
            indices,
            max_seq_len,
            max_chunks,
            &tok_ctx,
            TokenFnCtx.call,
        );
        defer batch.deinit(allocator);

        if (json_output) {
            std.debug.print(",\"first_batch\":{{\"sample_indices\":", .{});
            printUsizeArrayWithOffset(batch.sample_indices, sample_offset);
            std.debug.print(",\"samples\":[", .{});
            for (0..count) |bi| {
                if (bi > 0) std.debug.print(",", .{});
                const seq_start = bi * max_seq_len;
                const seq_end = seq_start + max_seq_len;
                const chunk_start = bi * max_chunks;
                const chunk_end = chunk_start + max_chunks;
                const ids_slice = batch.input_ids[seq_start..seq_end];
                const mask_slice = batch.attention_mask[seq_start..seq_end];
                const labels_slice = batch.boundary_labels[seq_start..seq_end];
                std.debug.print("{{\"sample_index\":{d},\"valid_tokens\":{d},\"input_ids\":", .{
                    sample_offset + batch.sample_indices[bi],
                    validTokenCount(mask_slice),
                });
                printI32Array(ids_slice);
                std.debug.print(",\"attention_mask\":", .{});
                printI32Array(mask_slice);
                std.debug.print(",\"boundary_positions\":", .{});
                printBoundaryPositions(labels_slice, mask_slice);
                std.debug.print(",\"source_chunk_boundaries\":", .{});
                printSourceChunkBoundaries(samples[batch.sample_indices[bi]].chunk_boundaries);
                std.debug.print(",\"chunk_spans\":", .{});
                printChunkSpans(
                    batch.chunk_starts[chunk_start..chunk_end],
                    batch.chunk_ends[chunk_start..chunk_end],
                    batch.chunk_mask[chunk_start..chunk_end],
                );
                std.debug.print("}}", .{});
            }
            std.debug.print("]}}", .{});
            std.debug.print(",\"debug_source_token_mappings\":[", .{});
            for (0..count) |bi| {
                if (bi > 0) std.debug.print(",", .{});
                const sample_index = batch.sample_indices[bi];
                @memset(offsets, .{ 0, 0 });
                @memset(ids, 0);
                @memset(mask, 0);
                const n_tokens = TokenFnCtx.call(&tok_ctx, samples[sample_index].text, ids, mask, offsets);
                const active_offsets = offsets[0..@min(n_tokens, max_seq_len)];
                std.debug.print("{{\"sample_index\":{d},\"mappings\":", .{sample_offset + sample_index});
                printSourceTokenMappings(samples[sample_index].chunk_boundaries, active_offsets);
                std.debug.print("}}", .{});
            }
            std.debug.print("]", .{});
        } else {
            std.debug.print("first_batch sample_indices=", .{});
            printUsizeArrayWithOffset(batch.sample_indices, sample_offset);
            std.debug.print("\n", .{});
            for (0..count) |bi| {
                const seq_start = bi * max_seq_len;
                const chunk_start = bi * max_chunks;
                std.debug.print("batch sample {d} valid={d} boundary_positions=", .{
                    sample_offset + batch.sample_indices[bi],
                    validTokenCount(batch.attention_mask[seq_start .. seq_start + max_seq_len]),
                });
                printBoundaryPositions(
                    batch.boundary_labels[seq_start .. seq_start + max_seq_len],
                    batch.attention_mask[seq_start .. seq_start + max_seq_len],
                );
                std.debug.print(" chunks=", .{});
                printChunkSpans(
                    batch.chunk_starts[chunk_start .. chunk_start + max_chunks],
                    batch.chunk_ends[chunk_start .. chunk_start + max_chunks],
                    batch.chunk_mask[chunk_start .. chunk_start + max_chunks],
                );
                std.debug.print("\n", .{});
            }
        }
    }
    if (json_output) std.debug.print("}}\n", .{});
}
