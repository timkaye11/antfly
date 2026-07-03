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

// Eval binary for the fused chunker-embedder boundary head.
//
// Usage:
//   eval_fused_chunker --data <path> --checkpoint <file> [options]
//
// Options:
//   --data <path>        JSONL eval data path (file or directory)
//   --checkpoint <file>  Checkpoint file written by train_fused_chunker
//   --split <name>       Dataset split filter (default: "val")
//   --batch-size <n>     Batch size (default: 32)
//   --max-examples <n>   Maximum eval examples (default: 0 = all)
//   --hidden-size <n>    Hidden size (default: 768)
//   --max-seq-len <n>    Max seq len (default: 384)
//   --max-chunks <n>     Max chunks per sample (default: 32)
//   --boundary-feature-mode token|prev-diff|prev-current-diff|prev-current-diff-concat|prev-current-next-diff-concat|window-context-diff
//   --boundary-alignment-dump-dir <dir>
//   --boundary-alignment-dump-top-k <n>
//   --backend native|metal|auto  Compute backend (default: auto)
//   --compiled-segment-forward   Compiled MPSGraph encoder forward (default: off;
//                                env ANTFLY_FUSED_CHUNKER_COMPILED_SEGMENT_FORWARD=1)

const std = @import("std");
const build_options = @import("build_options");
const native_compute = @import("../../ops/native_compute.zig");
const metal_compute = if (build_options.enable_metal) @import("../../ops/metal_compute.zig") else struct {};
const gpu_hosted_store = @import("../../ops/gpu_hosted_store.zig");
const metal_runtime = if (build_options.enable_metal) @import("../../backends/metal_runtime.zig") else struct {
    pub fn metalDeviceAvailable() bool {
        return false;
    }
};
const ops_mod = @import("../../ops/ops.zig");
const ComputeBackend = ops_mod.ComputeBackend;
const fused_chunker_data = @import("../fused_chunker_data.zig");
const fused_chunker_loss = @import("../fused_chunker_loss.zig");
const fused_chunker_lora = @import("../lora_adapter_set.zig");
const fused_chunker_train = @import("../fused_chunker_train.zig");
const fused_chunker_compiled_forward = @import("../fused_chunker_compiled_forward.zig");
const modern_bert = @import("../../architectures/modern_bert.zig");
const tokenizer_batch_mod = @import("../tokenizer_batch.zig");
const TokenizerBatch = tokenizer_batch_mod.TokenizerBatch;
const TokenFnCtx = tokenizer_batch_mod.TokenFnCtx;
const weight_source_mod = @import("../../models/weight_source.zig");
const SafetensorsSource = weight_source_mod.SafetensorsSource;
const LoadedWeight = weight_source_mod.LoadedWeight;
const safetensors = @import("../../models/safetensors.zig");
const tensor_mod = @import("../../backends/tensor.zig");
const compat = @import("../../io/compat.zig");
const FusedTrainer = fused_chunker_train.FusedTrainer;
const FusedTrainingConfig = fused_chunker_train.FusedTrainingConfig;
const fused_chunker_weights = @import("../fused_chunker_weights.zig");
const MetalWeightStore = fused_chunker_weights.MetalWeightStore;
const deinitMetalWeightStore = fused_chunker_weights.deinitMetalWeightStore;
const deinitNativeWeightStore = fused_chunker_weights.deinitNativeWeightStore;
const stripEncoderPrefix = fused_chunker_weights.stripEncoderPrefix;
const normalizeModernBertWeightName = fused_chunker_weights.normalizeModernBertWeightName;
const loadSafetensorsIntoNativeStore = fused_chunker_weights.loadSafetensorsIntoNativeStore;
const loadSafetensorsIntoMetalStore = fused_chunker_weights.loadSafetensorsIntoMetalStore;

const print = std.debug.print;
const chonky_boundary_metric = "chonky_character_separator_f1";
const token_boundary_metric = "token_boundary_f1";
const fused_chunker_lora_target_modules = [_][]const u8{
    "query_proj",
    "value_proj",
    "key_proj",
    "out_proj",
    "wo",
};

// ---------------------------------------------------------------------------
// Dense retrieval evaluation metrics
// ---------------------------------------------------------------------------

pub const RetrievalMetrics = struct {
    recall_at_1: f64,
    recall_at_10: f64,
    ndcg_at_10: f64,
    mrr: f64,
    num_queries: usize,
};

pub const BenchmarkBaseline = struct {
    name: []const u8,
    overall_ndcg_at_10: f64,
};

const BenchmarkDatasetResult = struct {
    name: []const u8,
    f1: f64,
    precision: f64,
    recall: f64,
    tp: u64,
    fp: u64,
    fn_count: u64,
};

const BenchmarkBoundaryLane = struct {
    dataset_metric: []const u8,
    internal_phase20_best_f1: f64,
    fixed_threshold_f1: f64,
    best_threshold_f1: f64,
    best_threshold: f64,
    average_precision: f64,
    precision_at_gold_count: f64,
    recall_at_gold_count: f64,
    f1_at_gold_count: f64,
    threshold_at_gold_count: f64,
    max_rank_f1: f64,
    max_rank_precision: f64,
    max_rank_recall: f64,
    max_rank_threshold: f64,
    sample_oracle_count_samples: u64,
    sample_oracle_count_topk_f1: f64,
    sample_oracle_count_topk_precision: f64,
    sample_oracle_count_topk_recall: f64,
    sample_oracle_count_topk_tp: u64,
    sample_oracle_count_topk_fp: u64,
    sample_oracle_count_topk_fn: u64,
    sample_oracle_count_nms_f1: f64,
    sample_oracle_count_nms_precision: f64,
    sample_oracle_count_nms_recall: f64,
    sample_oracle_count_nms_tp: u64,
    sample_oracle_count_nms_fp: u64,
    sample_oracle_count_nms_fn: u64,
    sample_oracle_count_nms_radius: u32,
    sample_oracle_count_length_window_f1: f64,
    sample_oracle_count_length_window_precision: f64,
    sample_oracle_count_length_window_recall: f64,
    sample_oracle_count_length_window_tp: u64,
    sample_oracle_count_length_window_fp: u64,
    sample_oracle_count_length_window_fn: u64,
    sample_oracle_count_length_window_min_radius: u32,
    sample_oracle_count_length_window_radius_fraction: f64,
    gold_positive_mean_rank: f64,
    gold_positive_mean_rank_percentile: f64,
    gold_positive_median_rank: u64,
    gold_positive_median_rank_percentile: f64,
    gold_positive_p90_rank: u64,
    gold_positive_p90_rank_percentile: f64,
    gold_positive_p99_rank: u64,
    gold_positive_p99_rank_percentile: f64,
    gold_positive_worst_rank: u64,
    gold_positive_top_5x_count: u64,
    gold_positive_top_5x_recall: f64,
    gold_positive_top_10x_count: u64,
    gold_positive_top_10x_recall: f64,
    dataset_results: []const BenchmarkDatasetResult,
};

const BenchmarkRetrievalLane = struct {
    output_dimension: u32,
    overall_ndcg_at_10: f64,
    recall_at_1: f64,
    recall_at_10: f64,
    mrr: f64,
    queries: usize,
    baselines: []const BenchmarkBaseline,
};

const BenchmarkSourceEval = struct {
    checkpoint: []const u8,
    data: []const u8,
    split: []const u8,
    backend: []const u8,
    samples: usize,
    max_seq_len: u32,
    max_chunks: u32,
};

const BenchmarkResultFile = struct {
    schema_version: []const u8 = "fused_chunker_benchmark_results/v1",
    source_eval: BenchmarkSourceEval,
    boundary_f1: BenchmarkBoundaryLane,
    retrieval_ndcg: ?BenchmarkRetrievalLane,
};

const BenchmarkResultInput = struct {
    dataset_name: []const u8,
    checkpoint_path: []const u8,
    data_path: []const u8,
    split: []const u8,
    backend_name: []const u8,
    samples: usize,
    max_seq_len: u32,
    max_chunks: u32,
    output_dimension: u32,
    internal_phase20_best_f1: f64,
    summary: fused_chunker_train.EvalSummary,
    separator_metrics: ?SeparatorBoundaryMetrics = null,
    retrieval: ?RetrievalMetrics,
    baselines: []const BenchmarkBaseline,
};

pub const SeparatorBoundaryMetrics = struct {
    precision: f64,
    recall: f64,
    f1: f64,
    tp: usize,
    fp: usize,
    fn_count: usize,
};

/// Chonky-compatible separator F1 over sorted, unique separator offsets.
///
/// Chonky's published eval turns paragraph separators into a sequence labeling
/// task and reports seqeval overall_f1. For single-position separator labels,
/// exact matching of sorted separator offsets is equivalent and much cheaper to
/// run inside the Zig benchmark harness.
pub fn computeSortedSeparatorBoundaryMetrics(
    gold_offsets: []const u32,
    predicted_offsets: []const u32,
) SeparatorBoundaryMetrics {
    var gi: usize = 0;
    var pi: usize = 0;
    var tp: usize = 0;
    var fp: usize = 0;
    var fn_count: usize = 0;

    while (gi < gold_offsets.len and pi < predicted_offsets.len) {
        const gold = gold_offsets[gi];
        const pred = predicted_offsets[pi];
        if (gold == pred) {
            tp += 1;
            gi += 1;
            pi += 1;
        } else if (pred < gold) {
            fp += 1;
            pi += 1;
        } else {
            fn_count += 1;
            gi += 1;
        }
    }
    fp += predicted_offsets.len - pi;
    fn_count += gold_offsets.len - gi;

    const precision = if (tp + fp == 0) 0.0 else @as(f64, @floatFromInt(tp)) / @as(f64, @floatFromInt(tp + fp));
    const recall = if (tp + fn_count == 0) 0.0 else @as(f64, @floatFromInt(tp)) / @as(f64, @floatFromInt(tp + fn_count));
    const f1 = if (precision + recall == 0.0) 0.0 else 2.0 * precision * recall / (precision + recall);
    return .{
        .precision = precision,
        .recall = recall,
        .f1 = f1,
        .tp = tp,
        .fp = fp,
        .fn_count = fn_count,
    };
}

const ChonkySeparatorAccumulator = struct {
    tp: usize = 0,
    fp: usize = 0,
    fn_count: usize = 0,

    fn addSample(
        self: *ChonkySeparatorAccumulator,
        allocator: std.mem.Allocator,
        sample: fused_chunker_data.FusedSample,
        logits: []const f32,
        mask: []const f32,
        offsets: []const [2]u32,
        threshold: f32,
    ) !void {
        if (sample.text.len == 0) return;
        if (sample.text.len > std.math.maxInt(u32)) return error.BenchmarkTextTooLong;

        var gold = std.ArrayListUnmanaged(u32).empty;
        defer gold.deinit(allocator);
        try appendGoldChonkySeparatorOffsets(allocator, &gold, sample, offsets);
        const gold_offsets = sortAndUniqueU32(gold.items);

        var predicted = std.ArrayListUnmanaged(u32).empty;
        defer predicted.deinit(allocator);
        try appendPredictedChonkySeparatorOffsets(
            allocator,
            &predicted,
            sample.text,
            sample.benchmark_include_final_boundary,
            logits,
            mask,
            offsets,
            threshold,
        );
        const predicted_offsets = sortAndUniqueU32(predicted.items);

        const metrics = computeSortedSeparatorBoundaryMetrics(gold_offsets, predicted_offsets);
        self.tp += metrics.tp;
        self.fp += metrics.fp;
        self.fn_count += metrics.fn_count;
    }

    fn finish(self: ChonkySeparatorAccumulator) SeparatorBoundaryMetrics {
        const precision = if (self.tp + self.fp == 0) 0.0 else @as(f64, @floatFromInt(self.tp)) / @as(f64, @floatFromInt(self.tp + self.fp));
        const recall = if (self.tp + self.fn_count == 0) 0.0 else @as(f64, @floatFromInt(self.tp)) / @as(f64, @floatFromInt(self.tp + self.fn_count));
        const f1 = if (precision + recall == 0.0) 0.0 else 2.0 * precision * recall / (precision + recall);
        return .{
            .precision = precision,
            .recall = recall,
            .f1 = f1,
            .tp = self.tp,
            .fp = self.fp,
            .fn_count = self.fn_count,
        };
    }
};

fn u32LessThan(_: void, a: u32, b: u32) bool {
    return a < b;
}

fn sortAndUniqueU32(items: []u32) []u32 {
    if (items.len == 0) return items;
    std.mem.sort(u32, items, {}, u32LessThan);
    var write: usize = 1;
    for (items[1..]) |value| {
        if (value == items[write - 1]) continue;
        items[write] = value;
        write += 1;
    }
    return items[0..write];
}

fn appendGoldChonkySeparatorOffsets(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u32),
    sample: fused_chunker_data.FusedSample,
    offsets: []const [2]u32,
) !void {
    if (sample.text.len == 0) return;
    if (sample.text.len > std.math.maxInt(u32)) return error.BenchmarkTextTooLong;

    var valid_chunk_idx: usize = 0;
    for (sample.chunk_boundaries) |chunk| {
        const span = fused_chunker_data.charToTokenBoundary(chunk.start_char, chunk.end_char, offsets);
        const resolved_start = if (chunk.start_token > 0) chunk.start_token else span.start_token;
        const resolved_end = if (chunk.end_token > 0) chunk.end_token else span.end_token;
        if (resolved_end <= resolved_start) continue;
        valid_chunk_idx += 1;
        if (valid_chunk_idx == 1) continue;
        if (chunk.start_char == 0) continue;
        try out.append(allocator, chunk.start_char - 1);
    }
    if (sample.benchmark_include_final_boundary) {
        try out.append(allocator, @as(u32, @intCast(sample.text.len - 1)));
    }
}

fn appendPredictedChonkySeparatorOffsets(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u32),
    text: []const u8,
    include_final_boundary: bool,
    logits: []const f32,
    mask: []const f32,
    offsets: []const [2]u32,
    threshold: f32,
) !void {
    const text_len = text.len;
    if (text_len == 0) return;
    if (text_len > std.math.maxInt(u32)) return error.BenchmarkTextTooLong;
    const token_count = @min(@min(mask.len, offsets.len), logits.len / 2);
    const final_offset: u32 = @intCast(text_len - 1);

    for (0..token_count) |token_idx| {
        if (mask[token_idx] <= 0.5) continue;
        const off = offsets[token_idx];
        if (off[0] == 0 and off[1] == 0) continue;
        if (off[0] == 0) continue;
        const probability = fused_chunker_loss.positiveBoundaryProbability(logits[token_idx * 2], logits[token_idx * 2 + 1]);
        if (probability <= threshold) continue;
        try out.append(allocator, tokenStartSeparatorOffset(text, off, final_offset) orelse continue);
    }
    if (include_final_boundary) {
        try out.append(allocator, final_offset);
    }
}

fn tokenStartSeparatorOffset(text: []const u8, offset: [2]u32, final_offset: u32) ?u32 {
    if (text.len == 0) return null;
    if (offset[0] >= text.len) return null;
    var token_start: usize = offset[0];
    const token_end = @min(@as(usize, offset[1]), text.len);
    while (token_start < token_end and std.ascii.isWhitespace(text[token_start])) {
        token_start += 1;
    }
    if (token_start == 0) return null;
    return @min(@as(u32, @intCast(token_start - 1)), final_offset);
}

fn accumulateChonkySeparatorBatch(
    allocator: std.mem.Allocator,
    acc: *ChonkySeparatorAccumulator,
    tokenizer: *TokenizerBatch,
    samples: []const fused_chunker_data.FusedSample,
    indices: []const usize,
    logits: []const f32,
    mask: []const f32,
    max_seq_len: usize,
    threshold: f32,
) !void {
    var tok_ctx = tokenizer.makeTokenFnCtx();
    const ids = try allocator.alloc(i32, max_seq_len);
    defer allocator.free(ids);
    const token_mask = try allocator.alloc(i32, max_seq_len);
    defer allocator.free(token_mask);
    const offsets = try allocator.alloc([2]u32, max_seq_len);
    defer allocator.free(offsets);

    for (indices, 0..) |sample_index, local_idx| {
        @memset(ids, 0);
        @memset(token_mask, 0);
        @memset(offsets, .{ 0, 0 });

        const sample = samples[sample_index];
        const n_tokens = TokenFnCtx.call(&tok_ctx, sample.text, ids, token_mask, offsets);
        const active_offsets = offsets[0..@min(n_tokens, max_seq_len)];
        const token_base = local_idx * max_seq_len;
        try acc.addSample(
            allocator,
            sample,
            logits[token_base * 2 .. (token_base + max_seq_len) * 2],
            mask[token_base .. token_base + max_seq_len],
            active_offsets,
            threshold,
        );
    }
}

const BoundaryAlignmentCandidate = struct {
    probability: f32,
    flat_index: usize,
};

const BoundaryAlignmentFeatureStats = struct {
    mean: f32,
    rms: f32,
    l2: f32,
    max_abs: f32,
    first0: f32,
    first1: f32,
    first2: f32,
    first3: f32,
};

const BoundaryAlignmentChunkMarkers = struct {
    chunk_start_index: ?usize = null,
    chunk_end_exclusive_index: ?usize = null,
    previous_chunk_end_index: ?usize = null,
};

fn boundaryAlignmentCandidateGreaterThan(_: void, a: BoundaryAlignmentCandidate, b: BoundaryAlignmentCandidate) bool {
    if (a.probability == b.probability) return a.flat_index < b.flat_index;
    return a.probability > b.probability;
}

fn boundaryAlignmentDumpPath(
    allocator: std.mem.Allocator,
    dump_dir: []const u8,
    eval_label: []const u8,
    step: u64,
) ![]u8 {
    return try std.fmt.allocPrint(allocator, "{s}/boundary_alignment_{s}_step_{d}.jsonl", .{ dump_dir, eval_label, step });
}

fn boundaryAlignmentChunkMarkers(
    batch: *const fused_chunker_data.FusedBatch,
    local_sample_idx: usize,
    token_idx: usize,
) BoundaryAlignmentChunkMarkers {
    const base = local_sample_idx * batch.max_chunks;
    var markers = BoundaryAlignmentChunkMarkers{};
    for (0..batch.max_chunks) |chunk_idx| {
        if (batch.chunk_mask[base + chunk_idx] <= 0.5) continue;
        const start = batch.chunk_starts[base + chunk_idx];
        const end = batch.chunk_ends[base + chunk_idx];
        if (start >= 0 and @as(usize, @intCast(start)) == token_idx) {
            markers.chunk_start_index = chunk_idx;
        }
        if (end >= 0 and @as(usize, @intCast(end)) == token_idx) {
            markers.chunk_end_exclusive_index = chunk_idx;
        }
        if (end > 0 and @as(usize, @intCast(end - 1)) == token_idx) {
            markers.previous_chunk_end_index = chunk_idx;
        }
    }
    return markers;
}

fn nearestBoundaryGoldDelta(
    batch: *const fused_chunker_data.FusedBatch,
    local_sample_idx: usize,
    token_idx: usize,
) ?i32 {
    const base = local_sample_idx * batch.max_seq_len;
    var best_delta: ?i32 = null;
    var best_abs: u32 = std.math.maxInt(u32);
    for (0..batch.max_seq_len) |candidate_idx| {
        if (batch.boundary_labels[base + candidate_idx] <= 0.5) continue;
        const delta = @as(i32, @intCast(candidate_idx)) - @as(i32, @intCast(token_idx));
        const abs_delta: u32 = @intCast(if (delta < 0) -delta else delta);
        if (abs_delta < best_abs) {
            best_abs = abs_delta;
            best_delta = delta;
        }
    }
    return best_delta;
}

fn boundaryAlignmentFeatureStats(
    features: []const f32,
    flat_index: usize,
    hidden_size: usize,
) ?BoundaryAlignmentFeatureStats {
    if (hidden_size == 0) return null;
    const start = flat_index * hidden_size;
    const end = start + hidden_size;
    if (end > features.len) return null;

    var sum: f64 = 0;
    var sum_sq: f64 = 0;
    var max_abs: f32 = 0;
    for (features[start..end]) |value| {
        sum += value;
        sum_sq += @as(f64, value) * @as(f64, value);
        const abs_value = @abs(value);
        if (abs_value > max_abs) max_abs = abs_value;
    }
    const hidden_f64: f64 = @floatFromInt(hidden_size);
    const row = features[start..end];
    return .{
        .mean = @floatCast(sum / hidden_f64),
        .rms = @floatCast(@sqrt(sum_sq / hidden_f64)),
        .l2 = @floatCast(@sqrt(sum_sq)),
        .max_abs = max_abs,
        .first0 = row[0],
        .first1 = if (hidden_size > 1) row[1] else 0,
        .first2 = if (hidden_size > 2) row[2] else 0,
        .first3 = if (hidden_size > 3) row[3] else 0,
    };
}

fn writeBoundaryAlignmentRecord(
    writer: *std.Io.Writer,
    eval_label: []const u8,
    step: u64,
    batch_idx: usize,
    batch: *const fused_chunker_data.FusedBatch,
    logits: []const f32,
    features: []const f32,
    hidden_size: usize,
    flat_index: usize,
    record_kind: []const u8,
    top_probability_rank: ?usize,
    probability_rank: ?usize,
    valid_token_count: usize,
) !void {
    const local_sample_idx = flat_index / batch.max_seq_len;
    const token_idx = flat_index % batch.max_seq_len;
    const logit0 = logits[flat_index * 2 + 0];
    const logit1 = logits[flat_index * 2 + 1];
    const probability = fused_chunker_loss.positiveBoundaryProbability(logit0, logit1);
    const markers = boundaryAlignmentChunkMarkers(batch, local_sample_idx, token_idx);
    const probability_rank_percentile: ?f64 = if (probability_rank) |rank|
        @as(f64, @floatFromInt(rank)) / @as(f64, @floatFromInt(valid_token_count))
    else
        null;
    try std.json.Stringify.value(.{
        .event = "boundary_alignment_token",
        .schema_version = 2,
        .eval = eval_label,
        .step = step,
        .batch = batch_idx,
        .record_kind = record_kind,
        .top_probability_rank = top_probability_rank,
        .probability_rank = probability_rank,
        .probability_rank_percentile = probability_rank_percentile,
        .valid_token_count = valid_token_count,
        .eval_sample_index = batch.sample_indices[local_sample_idx],
        .local_sample_index = local_sample_idx,
        .sample_token_index = token_idx,
        .token_id = batch.input_ids[flat_index],
        .attention = batch.attention_mask[flat_index],
        .gold_boundary = batch.boundary_labels[flat_index] > 0.5,
        .probability = probability,
        .margin = logit1 - logit0,
        .logit0 = logit0,
        .logit1 = logit1,
        .nearest_gold_delta = nearestBoundaryGoldDelta(batch, local_sample_idx, token_idx),
        .chunk_start_index = markers.chunk_start_index,
        .chunk_end_exclusive_index = markers.chunk_end_exclusive_index,
        .previous_chunk_end_index = markers.previous_chunk_end_index,
        .feature_stats = boundaryAlignmentFeatureStats(features, flat_index, hidden_size),
    }, .{}, writer);
    try writer.writeByte('\n');
}

fn writeBoundaryAlignmentDumpBatch(
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    eval_label: []const u8,
    step: u64,
    batch_idx: usize,
    batch: *const fused_chunker_data.FusedBatch,
    logits: []const f32,
    features: []const f32,
    hidden_size: usize,
    top_k: usize,
) !void {
    const total_tokens = batch.batch_size * batch.max_seq_len;

    var candidates = std.ArrayListUnmanaged(BoundaryAlignmentCandidate).empty;
    defer candidates.deinit(allocator);
    try candidates.ensureTotalCapacity(allocator, total_tokens);

    for (0..total_tokens) |flat_index| {
        if (batch.attention_mask[flat_index] == 0) continue;
        const logit0 = logits[flat_index * 2 + 0];
        const logit1 = logits[flat_index * 2 + 1];
        try candidates.append(allocator, .{
            .probability = fused_chunker_loss.positiveBoundaryProbability(logit0, logit1),
            .flat_index = flat_index,
        });
    }

    std.mem.sort(BoundaryAlignmentCandidate, candidates.items, {}, boundaryAlignmentCandidateGreaterThan);

    const rank_by_flat_index = try allocator.alloc(usize, total_tokens);
    defer allocator.free(rank_by_flat_index);
    @memset(rank_by_flat_index, 0);
    for (candidates.items, 0..) |candidate, rank| {
        rank_by_flat_index[candidate.flat_index] = rank + 1;
    }

    for (0..total_tokens) |flat_index| {
        if (batch.boundary_labels[flat_index] > 0.5) {
            try writeBoundaryAlignmentRecord(
                writer,
                eval_label,
                step,
                batch_idx,
                batch,
                logits,
                features,
                hidden_size,
                flat_index,
                "gold_boundary",
                null,
                if (rank_by_flat_index[flat_index] == 0) null else rank_by_flat_index[flat_index],
                candidates.items.len,
            );
        }
    }

    const limit = @min(top_k, candidates.items.len);
    for (candidates.items[0..limit], 0..) |candidate, rank| {
        try writeBoundaryAlignmentRecord(
            writer,
            eval_label,
            step,
            batch_idx,
            batch,
            logits,
            features,
            hidden_size,
            candidate.flat_index,
            "top_probability",
            rank + 1,
            rank + 1,
            candidates.items.len,
        );
    }
}

fn binaryDcgAtKFromRankedMatches(matches: []const bool, k: usize) f64 {
    const top_k = @min(matches.len, k);
    var dcg: f64 = 0.0;
    for (0..top_k) |rank_idx| {
        if (!matches[rank_idx]) continue;
        dcg += 1.0 / @log2(@as(f64, @floatFromInt(rank_idx + 2)));
    }
    return dcg;
}

fn binaryIdealDcgAtK(positive_count: usize, k: usize) f64 {
    const ideal_len = @min(positive_count, k);
    var idcg: f64 = 0.0;
    for (0..ideal_len) |rank_idx| {
        idcg += 1.0 / @log2(@as(f64, @floatFromInt(rank_idx + 2)));
    }
    return idcg;
}

pub fn binaryNdcgAtKFromRankedMatches(matches: []const bool, positive_count: usize, k: usize) f64 {
    if (positive_count == 0) return 0.0;
    const idcg = binaryIdealDcgAtK(positive_count, k);
    if (idcg == 0.0) return 0.0;
    return binaryDcgAtKFromRankedMatches(matches, k) / idcg;
}

/// Compute retrieval metrics from chunk embeddings.
///
/// embeddings:  [num_chunks * embed_dim] — chunk embeddings (need not be pre-normalised)
/// doc_ids:     [num_chunks] — chunks sharing a doc_id are positives
/// chunk_mask:  [num_chunks] — 1.0 valid, 0.0 padding
/// num_chunks:  total number of chunks (including padding)
/// embed_dim:   embedding dimension
///
/// Algorithm (O(V²) where V = number of valid chunks):
///   1. Filter to valid chunks (chunk_mask > 0).
///   2. L2-normalise each valid embedding.
///   3. For each query i: compute dot-product similarity to all other valid chunks,
///      rank by descending similarity (excluding self), find positives (same doc_id).
///   4. Accumulate Recall@1, Recall@10, MRR; average over queries with at least one positive.
pub fn computeRetrievalMetrics(
    allocator: std.mem.Allocator,
    embeddings: []const f32,
    doc_ids: []const u32,
    chunk_mask: []const f32,
    num_chunks: usize,
    embed_dim: usize,
) !RetrievalMetrics {
    // Step 1: collect valid indices.
    var valid_idx = std.ArrayListUnmanaged(usize).empty;
    defer valid_idx.deinit(allocator);

    for (0..num_chunks) |i| {
        if (chunk_mask[i] > 0.5) {
            try valid_idx.append(allocator, i);
        }
    }
    const V = valid_idx.items.len;

    if (V < 2) {
        return RetrievalMetrics{
            .recall_at_1 = 0,
            .recall_at_10 = 0,
            .ndcg_at_10 = 0,
            .mrr = 0,
            .num_queries = 0,
        };
    }

    // Step 2: L2-normalise valid embeddings into a compact [V * embed_dim] buffer.
    const norm_vecs = try allocator.alloc(f32, V * embed_dim);
    defer allocator.free(norm_vecs);

    const compact_doc_ids = try allocator.alloc(u32, V);
    defer allocator.free(compact_doc_ids);

    for (0..V) |ci| {
        const orig_i = valid_idx.items[ci];
        compact_doc_ids[ci] = doc_ids[orig_i];

        const src = embeddings[orig_i * embed_dim .. orig_i * embed_dim + embed_dim];
        const dst = norm_vecs[ci * embed_dim .. ci * embed_dim + embed_dim];

        var sum_sq: f32 = 0;
        for (src) |v| sum_sq += v * v;
        const inv_norm: f32 = if (sum_sq > 1e-24) 1.0 / @sqrt(sum_sq) else 0;
        for (src, dst) |s, *d| d.* = s * inv_norm;
    }

    // Step 3: for each query, compute similarities, rank, accumulate metrics.
    const sims = try allocator.alloc(f32, V);
    defer allocator.free(sims);

    // Reusable scratch buffer for sorting indices.
    const rank_buf = try allocator.alloc(usize, V);
    defer allocator.free(rank_buf);

    var sum_r1: f64 = 0;
    var sum_r10: f64 = 0;
    var sum_ndcg10: f64 = 0;
    var sum_mrr: f64 = 0;
    var num_queries: usize = 0;

    for (0..V) |qi| {
        // Check whether this query has any positives.
        var positive_count: usize = 0;
        for (0..V) |j| {
            if (j != qi and compact_doc_ids[j] == compact_doc_ids[qi]) {
                positive_count += 1;
            }
        }
        if (positive_count == 0) continue;

        // Compute dot-product similarities to all other valid chunks.
        const qi_base = qi * embed_dim;
        for (0..V) |j| {
            if (j == qi) {
                sims[j] = -std.math.inf(f32); // exclude self from ranking
                continue;
            }
            const qj_base = j * embed_dim;
            var dot: f32 = 0;
            for (0..embed_dim) |k| {
                dot += norm_vecs[qi_base + k] * norm_vecs[qj_base + k];
            }
            sims[j] = dot;
        }

        // Build rank buffer [0..V] and partial-sort to find the top-10.
        // We use a simple selection approach: track the top-K indices by sim value.
        // For V up to ~512 a full sort is fine.
        for (0..V) |k| rank_buf[k] = k;
        // Sort descending by similarity.
        std.sort.pdq(usize, rank_buf, sims, struct {
            fn lessThan(sim_slice: []const f32, a: usize, b: usize) bool {
                return sim_slice[a] > sim_slice[b]; // descending
            }
        }.lessThan);

        // Recall@1
        if (compact_doc_ids[rank_buf[0]] == compact_doc_ids[qi]) {
            sum_r1 += 1.0;
        }

        // Recall@10 and MRR
        var found_r10 = false;
        var first_positive_rank: usize = 0; // 1-based, 0 means not found yet
        const top_k = @min(V - 1, 10); // at most V-1 non-self results
        var dcg10: f64 = 0.0;
        for (0..top_k) |rank_idx| {
            const j = rank_buf[rank_idx];
            if (compact_doc_ids[j] == compact_doc_ids[qi]) {
                dcg10 += 1.0 / @log2(@as(f64, @floatFromInt(rank_idx + 2)));
                if (!found_r10) {
                    sum_r10 += 1.0;
                    found_r10 = true;
                }
                if (first_positive_rank == 0) {
                    first_positive_rank = rank_idx + 1; // 1-based
                }
            }
        }
        const idcg10 = binaryIdealDcgAtK(positive_count, 10);
        if (idcg10 > 0.0) sum_ndcg10 += dcg10 / idcg10;
        // If the first positive wasn't in top-10, search the rest for MRR.
        if (first_positive_rank == 0) {
            for (top_k..V) |rank_idx| {
                const j = rank_buf[rank_idx];
                if (j == qi) continue;
                if (compact_doc_ids[j] == compact_doc_ids[qi]) {
                    first_positive_rank = rank_idx + 1;
                    break;
                }
            }
        }
        if (first_positive_rank > 0) {
            sum_mrr += 1.0 / @as(f64, @floatFromInt(first_positive_rank));
        }

        num_queries += 1;
    }

    if (num_queries == 0) {
        return RetrievalMetrics{
            .recall_at_1 = 0,
            .recall_at_10 = 0,
            .ndcg_at_10 = 0,
            .mrr = 0,
            .num_queries = 0,
        };
    }

    const nq_f: f64 = @floatFromInt(num_queries);
    return RetrievalMetrics{
        .recall_at_1 = sum_r1 / nq_f,
        .recall_at_10 = sum_r10 / nq_f,
        .ndcg_at_10 = sum_ndcg10 / nq_f,
        .mrr = sum_mrr / nq_f,
        .num_queries = num_queries,
    };
}

test "separator boundary metrics match exact sorted offsets" {
    const gold = [_]u32{ 4, 12, 20, 28 };
    const pred = [_]u32{ 4, 10, 20, 32 };
    const metrics = computeSortedSeparatorBoundaryMetrics(&gold, &pred);
    try std.testing.expectEqual(@as(usize, 2), metrics.tp);
    try std.testing.expectEqual(@as(usize, 2), metrics.fp);
    try std.testing.expectEqual(@as(usize, 2), metrics.fn_count);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), metrics.precision, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), metrics.recall, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), metrics.f1, 1e-12);
}

test "chonky separator accumulator maps token starts and final boundary" {
    const allocator = std.testing.allocator;
    var chunks = [_]fused_chunker_data.FusedChunkBoundary{
        .{ .start_char = 0, .end_char = 10 },
        .{ .start_char = 11, .end_char = 22 },
    };
    const sample = fused_chunker_data.FusedSample{
        .text = "alpha beta gamma delta",
        .chunk_boundaries = &chunks,
        .positive_texts = &.{},
        .benchmark_include_final_boundary = true,
    };
    const offsets = [_][2]u32{
        .{ 0, 5 },
        .{ 6, 10 },
        .{ 11, 16 },
        .{ 17, 22 },
    };
    const mask = [_]f32{ 1, 1, 1, 1 };
    const logits = [_]f32{
        3,  -3,
        3,  -3,
        -3, 3,
        3,  -3,
    };

    var acc = ChonkySeparatorAccumulator{};
    try acc.addSample(allocator, sample, &logits, &mask, &offsets, 0.5);
    const metrics = acc.finish();
    try std.testing.expectEqual(@as(usize, 2), metrics.tp);
    try std.testing.expectEqual(@as(usize, 0), metrics.fp);
    try std.testing.expectEqual(@as(usize, 0), metrics.fn_count);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), metrics.f1, 1e-12);
}

test "chonky separator accumulator skips final boundary for split windows" {
    const allocator = std.testing.allocator;
    var chunks = [_]fused_chunker_data.FusedChunkBoundary{
        .{ .start_char = 0, .end_char = 10 },
        .{ .start_char = 11, .end_char = 22 },
    };
    const sample = fused_chunker_data.FusedSample{
        .text = "alpha beta gamma delta",
        .chunk_boundaries = &chunks,
        .positive_texts = &.{},
        .benchmark_include_final_boundary = false,
    };
    const offsets = [_][2]u32{
        .{ 0, 5 },
        .{ 6, 10 },
        .{ 11, 16 },
        .{ 17, 22 },
    };
    const mask = [_]f32{ 1, 1, 1, 1 };
    const logits = [_]f32{
        3,  -3,
        3,  -3,
        -3, 3,
        3,  -3,
    };

    var acc = ChonkySeparatorAccumulator{};
    try acc.addSample(allocator, sample, &logits, &mask, &offsets, 0.5);
    const metrics = acc.finish();
    try std.testing.expectEqual(@as(usize, 1), metrics.tp);
    try std.testing.expectEqual(@as(usize, 0), metrics.fp);
    try std.testing.expectEqual(@as(usize, 0), metrics.fn_count);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), metrics.f1, 1e-12);
}

test "chonky separator accumulator handles token offset with leading separator whitespace" {
    const allocator = std.testing.allocator;
    var chunks = [_]fused_chunker_data.FusedChunkBoundary{
        .{ .start_char = 0, .end_char = 5 },
        .{ .start_char = 6, .end_char = 10 },
    };
    const sample = fused_chunker_data.FusedSample{
        .text = "alpha beta",
        .chunk_boundaries = &chunks,
        .positive_texts = &.{},
        .benchmark_include_final_boundary = false,
    };
    const offsets = [_][2]u32{
        .{ 0, 5 },
        .{ 5, 10 },
    };
    const mask = [_]f32{ 1, 1 };
    const logits = [_]f32{
        3,  -3,
        -3, 3,
    };

    var acc = ChonkySeparatorAccumulator{};
    try acc.addSample(allocator, sample, &logits, &mask, &offsets, 0.5);
    const metrics = acc.finish();
    try std.testing.expectEqual(@as(usize, 1), metrics.tp);
    try std.testing.expectEqual(@as(usize, 0), metrics.fp);
    try std.testing.expectEqual(@as(usize, 0), metrics.fn_count);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), metrics.f1, 1e-12);
}

test "chonky separator accumulator filters source boundaries outside tokenized window" {
    const allocator = std.testing.allocator;
    var chunks = [_]fused_chunker_data.FusedChunkBoundary{
        .{ .start_char = 0, .end_char = 5 },
        .{ .start_char = 6, .end_char = 10 },
        .{ .start_char = 50, .end_char = 60 },
    };
    const sample = fused_chunker_data.FusedSample{
        .text = "alpha beta trailing text beyond window",
        .chunk_boundaries = &chunks,
        .positive_texts = &.{},
        .benchmark_include_final_boundary = false,
    };
    const offsets = [_][2]u32{
        .{ 0, 5 },
        .{ 5, 10 },
    };
    const mask = [_]f32{ 1, 1 };
    const logits = [_]f32{
        3,  -3,
        -3, 3,
    };

    var acc = ChonkySeparatorAccumulator{};
    try acc.addSample(allocator, sample, &logits, &mask, &offsets, 0.5);
    const metrics = acc.finish();
    try std.testing.expectEqual(@as(usize, 1), metrics.tp);
    try std.testing.expectEqual(@as(usize, 0), metrics.fp);
    try std.testing.expectEqual(@as(usize, 0), metrics.fn_count);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), metrics.f1, 1e-12);
}

test "binary ndcg uses Voyage-style top-k ranking discount" {
    const matches = [_]bool{ false, true, true, false };
    const ndcg = binaryNdcgAtKFromRankedMatches(&matches, 2, 10);
    const expected = (1.0 / @log2(@as(f64, 3.0)) + 1.0 / @log2(@as(f64, 4.0))) /
        (1.0 + 1.0 / @log2(@as(f64, 3.0)));
    try std.testing.expectApproxEqAbs(expected, ndcg, 1e-12);
}

test "dense retrieval metrics report perfect ndcg for nearest positive chunks" {
    const allocator = std.testing.allocator;
    const embeddings = [_]f32{
        1.0, 0.0,
        0.0, 1.0,
        0.9, 0.1,
        0.1, 0.9,
    };
    const doc_ids = [_]u32{ 0, 1, 0, 1 };
    const mask = [_]f32{ 1, 1, 1, 1 };
    const metrics = try computeRetrievalMetrics(allocator, &embeddings, &doc_ids, &mask, 4, 2);
    try std.testing.expectEqual(@as(usize, 4), metrics.num_queries);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), metrics.recall_at_1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), metrics.recall_at_10, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), metrics.ndcg_at_10, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), metrics.mrr, 1e-12);
}

pub fn parseBenchmarkBaseline(raw: []const u8) !BenchmarkBaseline {
    const sep = std.mem.lastIndexOfScalar(u8, raw, ':') orelse return error.InvalidBenchmarkBaseline;
    if (sep == 0 or sep + 1 >= raw.len) return error.InvalidBenchmarkBaseline;
    const name = raw[0..sep];
    const ndcg = try std.fmt.parseFloat(f64, raw[sep + 1 ..]);
    if (!std.math.isFinite(ndcg) or ndcg < 0.0 or ndcg > 1.0) return error.InvalidBenchmarkBaseline;
    return .{
        .name = name,
        .overall_ndcg_at_10 = ndcg,
    };
}

fn parseBoundaryFeatureMode(raw: []const u8) !fused_chunker_train.BoundaryFeatureMode {
    if (std.mem.eql(u8, raw, "token")) return .token;
    if (std.mem.eql(u8, raw, "prev-diff")) return .prev_diff;
    if (std.mem.eql(u8, raw, "prev-current-diff")) return .prev_current_diff;
    if (std.mem.eql(u8, raw, "prev-current-diff-concat")) return .prev_current_diff_concat;
    if (std.mem.eql(u8, raw, "prev-current-next-diff-concat")) return .prev_current_next_diff_concat;
    if (std.mem.eql(u8, raw, "window-context-diff")) return .window_context_diff;
    return error.InvalidBoundaryFeatureMode;
}

pub fn renderBenchmarkResultsJson(allocator: std.mem.Allocator, input: BenchmarkResultInput) ![]u8 {
    const dataset_boundary = input.separator_metrics orelse SeparatorBoundaryMetrics{
        .precision = input.summary.boundary_precision,
        .recall = input.summary.boundary_recall,
        .f1 = input.summary.boundary_f1,
        .tp = @intCast(input.summary.boundary_tp),
        .fp = @intCast(input.summary.boundary_fp),
        .fn_count = @intCast(input.summary.boundary_fn),
    };
    const dataset_metric = if (input.separator_metrics != null) chonky_boundary_metric else token_boundary_metric;
    const dataset_results = [_]BenchmarkDatasetResult{.{
        .name = input.dataset_name,
        .f1 = dataset_boundary.f1,
        .precision = dataset_boundary.precision,
        .recall = dataset_boundary.recall,
        .tp = @intCast(dataset_boundary.tp),
        .fp = @intCast(dataset_boundary.fp),
        .fn_count = @intCast(dataset_boundary.fn_count),
    }};
    const retrieval_lane: ?BenchmarkRetrievalLane = if (input.retrieval) |retrieval| .{
        .output_dimension = input.output_dimension,
        .overall_ndcg_at_10 = retrieval.ndcg_at_10,
        .recall_at_1 = retrieval.recall_at_1,
        .recall_at_10 = retrieval.recall_at_10,
        .mrr = retrieval.mrr,
        .queries = retrieval.num_queries,
        .baselines = input.baselines,
    } else null;
    const file = BenchmarkResultFile{
        .source_eval = .{
            .checkpoint = input.checkpoint_path,
            .data = input.data_path,
            .split = input.split,
            .backend = input.backend_name,
            .samples = input.samples,
            .max_seq_len = input.max_seq_len,
            .max_chunks = input.max_chunks,
        },
        .boundary_f1 = .{
            .dataset_metric = dataset_metric,
            .internal_phase20_best_f1 = input.internal_phase20_best_f1,
            .fixed_threshold_f1 = input.summary.boundary_f1,
            .best_threshold_f1 = input.summary.best_boundary_f1,
            .best_threshold = input.summary.best_boundary_threshold,
            .average_precision = input.summary.average_precision,
            .precision_at_gold_count = input.summary.precision_at_gold_count,
            .recall_at_gold_count = input.summary.recall_at_gold_count,
            .f1_at_gold_count = input.summary.f1_at_gold_count,
            .threshold_at_gold_count = input.summary.threshold_at_gold_count,
            .max_rank_f1 = input.summary.max_rank_f1,
            .max_rank_precision = input.summary.max_rank_precision,
            .max_rank_recall = input.summary.max_rank_recall,
            .max_rank_threshold = input.summary.max_rank_threshold,
            .sample_oracle_count_samples = input.summary.sample_oracle_count_samples,
            .sample_oracle_count_topk_f1 = input.summary.sample_oracle_count_topk_f1,
            .sample_oracle_count_topk_precision = input.summary.sample_oracle_count_topk_precision,
            .sample_oracle_count_topk_recall = input.summary.sample_oracle_count_topk_recall,
            .sample_oracle_count_topk_tp = input.summary.sample_oracle_count_topk_tp,
            .sample_oracle_count_topk_fp = input.summary.sample_oracle_count_topk_fp,
            .sample_oracle_count_topk_fn = input.summary.sample_oracle_count_topk_fn,
            .sample_oracle_count_nms_f1 = input.summary.sample_oracle_count_nms_f1,
            .sample_oracle_count_nms_precision = input.summary.sample_oracle_count_nms_precision,
            .sample_oracle_count_nms_recall = input.summary.sample_oracle_count_nms_recall,
            .sample_oracle_count_nms_tp = input.summary.sample_oracle_count_nms_tp,
            .sample_oracle_count_nms_fp = input.summary.sample_oracle_count_nms_fp,
            .sample_oracle_count_nms_fn = input.summary.sample_oracle_count_nms_fn,
            .sample_oracle_count_nms_radius = input.summary.sample_oracle_count_nms_radius,
            .sample_oracle_count_length_window_f1 = input.summary.sample_oracle_count_length_window_f1,
            .sample_oracle_count_length_window_precision = input.summary.sample_oracle_count_length_window_precision,
            .sample_oracle_count_length_window_recall = input.summary.sample_oracle_count_length_window_recall,
            .sample_oracle_count_length_window_tp = input.summary.sample_oracle_count_length_window_tp,
            .sample_oracle_count_length_window_fp = input.summary.sample_oracle_count_length_window_fp,
            .sample_oracle_count_length_window_fn = input.summary.sample_oracle_count_length_window_fn,
            .sample_oracle_count_length_window_min_radius = input.summary.sample_oracle_count_length_window_min_radius,
            .sample_oracle_count_length_window_radius_fraction = input.summary.sample_oracle_count_length_window_radius_fraction,
            .gold_positive_mean_rank = input.summary.gold_positive_mean_rank,
            .gold_positive_mean_rank_percentile = input.summary.gold_positive_mean_rank_percentile,
            .gold_positive_median_rank = input.summary.gold_positive_median_rank,
            .gold_positive_median_rank_percentile = input.summary.gold_positive_median_rank_percentile,
            .gold_positive_p90_rank = input.summary.gold_positive_p90_rank,
            .gold_positive_p90_rank_percentile = input.summary.gold_positive_p90_rank_percentile,
            .gold_positive_p99_rank = input.summary.gold_positive_p99_rank,
            .gold_positive_p99_rank_percentile = input.summary.gold_positive_p99_rank_percentile,
            .gold_positive_worst_rank = input.summary.gold_positive_worst_rank,
            .gold_positive_top_5x_count = input.summary.gold_positive_top_5x_count,
            .gold_positive_top_5x_recall = input.summary.gold_positive_top_5x_recall,
            .gold_positive_top_10x_count = input.summary.gold_positive_top_10x_count,
            .gold_positive_top_10x_recall = input.summary.gold_positive_top_10x_recall,
            .dataset_results = &dataset_results,
        },
        .retrieval_ndcg = retrieval_lane,
    };

    var buffer: std.Io.Writer.Allocating = .init(allocator);
    errdefer buffer.deinit();
    try std.json.Stringify.value(file, .{ .whitespace = .indent_2 }, &buffer.writer);
    try buffer.writer.writeByte('\n');
    return buffer.toOwnedSlice();
}

fn writeBenchmarkResultsJson(allocator: std.mem.Allocator, path: []const u8, input: BenchmarkResultInput) !void {
    const json = try renderBenchmarkResultsJson(allocator, input);
    defer allocator.free(json);
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = path, .data = json });
}

test "parse benchmark baseline accepts name and ndcg" {
    const baseline = try parseBenchmarkBaseline("fixed_500_50_same_encoder:0.7125");
    try std.testing.expectEqualStrings("fixed_500_50_same_encoder", baseline.name);
    try std.testing.expectApproxEqAbs(@as(f64, 0.7125), baseline.overall_ndcg_at_10, 1e-12);
    try std.testing.expectError(error.InvalidBenchmarkBaseline, parseBenchmarkBaseline("bad"));
    try std.testing.expectError(error.InvalidBenchmarkBaseline, parseBenchmarkBaseline("bad:1.2"));
}

test "render benchmark results JSON matches verifier schema" {
    const allocator = std.testing.allocator;
    const summary = fused_chunker_train.EvalSummary{
        .boundary_f1 = 0.82,
        .boundary_precision = 0.83,
        .boundary_recall = 0.81,
        .boundary_tp = 82,
        .boundary_fp = 17,
        .boundary_fn = 19,
        .best_boundary_f1 = 0.84,
        .best_boundary_threshold = 0.42,
        .average_precision = 0.85,
        .precision_at_gold_count = 0.80,
        .recall_at_gold_count = 0.79,
        .f1_at_gold_count = 0.795,
        .threshold_at_gold_count = 0.51,
        .max_rank_f1 = 0.86,
        .max_rank_precision = 0.84,
        .max_rank_recall = 0.88,
        .max_rank_threshold = 0.47,
    };
    const retrieval = RetrievalMetrics{
        .recall_at_1 = 0.55,
        .recall_at_10 = 0.91,
        .ndcg_at_10 = 0.74,
        .mrr = 0.63,
        .num_queries = 12,
    };
    const baselines = [_]BenchmarkBaseline{.{
        .name = "fixed_500_50_same_encoder",
        .overall_ndcg_at_10 = 0.68,
    }};
    const json = try renderBenchmarkResultsJson(allocator, .{
        .dataset_name = "bookcorpus",
        .checkpoint_path = "ckpt.safetensors",
        .data_path = "eval.jsonl",
        .split = "val",
        .backend_name = "native",
        .samples = 8,
        .max_seq_len = 384,
        .max_chunks = 32,
        .output_dimension = 256,
        .internal_phase20_best_f1 = 0.86,
        .summary = summary,
        .retrieval = retrieval,
        .baselines = &baselines,
    });
    defer allocator.free(json);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings("fused_chunker_benchmark_results/v1", obj.get("schema_version").?.string);
    const boundary = obj.get("boundary_f1").?.object;
    try std.testing.expectEqualStrings(token_boundary_metric, boundary.get("dataset_metric").?.string);
    try std.testing.expectApproxEqAbs(@as(f64, 0.86), boundary.get("internal_phase20_best_f1").?.float, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.85), boundary.get("average_precision").?.float, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.86), boundary.get("max_rank_f1").?.float, 1e-12);
    const datasets = boundary.get("dataset_results").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), datasets.len);
    try std.testing.expectEqualStrings("bookcorpus", datasets[0].object.get("name").?.string);
    const retrieval_obj = obj.get("retrieval_ndcg").?.object;
    try std.testing.expectApproxEqAbs(@as(f64, 0.74), retrieval_obj.get("overall_ndcg_at_10").?.float, 1e-12);
}

test "render benchmark results uses chonky separator metrics for dataset result" {
    const allocator = std.testing.allocator;
    const summary = fused_chunker_train.EvalSummary{
        .boundary_f1 = 0.20,
        .boundary_precision = 0.25,
        .boundary_recall = 0.166,
        .boundary_tp = 1,
        .boundary_fp = 3,
        .boundary_fn = 5,
        .best_boundary_f1 = 0.30,
        .best_boundary_threshold = 0.4,
    };
    const separator = SeparatorBoundaryMetrics{
        .precision = 0.75,
        .recall = 0.60,
        .f1 = 2.0 / 3.0,
        .tp = 6,
        .fp = 2,
        .fn_count = 4,
    };

    const json = try renderBenchmarkResultsJson(allocator, .{
        .dataset_name = "bookcorpus",
        .checkpoint_path = "/tmp/checkpoint.safetensors",
        .data_path = "/tmp/bookcorpus.jsonl",
        .split = "val",
        .backend_name = "native",
        .samples = 8,
        .max_seq_len = 384,
        .max_chunks = 32,
        .output_dimension = 256,
        .internal_phase20_best_f1 = 0.80,
        .summary = summary,
        .separator_metrics = separator,
        .retrieval = null,
        .baselines = &.{},
    });
    defer allocator.free(json);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const boundary = parsed.value.object.get("boundary_f1").?.object;
    try std.testing.expectEqualStrings(chonky_boundary_metric, boundary.get("dataset_metric").?.string);
    const dataset = boundary.get("dataset_results").?.array.items[0].object;
    try std.testing.expectApproxEqAbs(@as(f64, 2.0 / 3.0), dataset.get("f1").?.float, 1e-12);
    try std.testing.expectEqual(@as(i64, 6), dataset.get("tp").?.integer);
    try std.testing.expectEqual(@as(i64, 2), dataset.get("fp").?.integer);
    try std.testing.expectEqual(@as(i64, 4), dataset.get("fn_count").?.integer);
}

const FusedBackend = enum {
    auto,
    metal,
    native,
};

const Options = struct {
    data_path: []const u8,
    checkpoint_path: []const u8,
    model_dir: ?[]const u8 = null,
    split: []const u8 = "val",
    results_out: ?[]const u8 = null,
    benchmark_dataset_name: ?[]const u8 = null,
    internal_phase20_best_f1: ?f64 = null,
    output_dimension: ?u32 = null,
    retrieval_baselines: []const BenchmarkBaseline = &.{},
    batch_size: u32 = 32,
    max_examples: usize = 0,
    hidden_size: u32 = 768,
    num_layers: u32 = 22,
    max_seq_len: u32 = 384,
    max_chunks: u32 = 32,
    intermediate_size: u32 = 1152,
    boundary_feature_mode: fused_chunker_train.BoundaryFeatureMode = .token,
    boundary_alignment_dump_dir: ?[]const u8 = null,
    boundary_alignment_dump_top_k: usize = 32,
    backend: FusedBackend = .auto,
    /// Opt-in compiled MPSGraph segmented encoder forward (requires Metal,
    /// encoder weights, and LoRA adapters in the checkpoint). Eager fallback
    /// on any failure. Also settable via
    /// ANTFLY_FUSED_CHUNKER_COMPILED_SEGMENT_FORWARD=1.
    compiled_segment_forward: bool = false,
};

fn printUsage() void {
    print(
        \\Usage: eval-fused-chunker --data <path> --checkpoint <file> [options]
        \\
        \\Options:
        \\  --data <path>            JSONL eval data path (file or directory)
        \\  --checkpoint <file>      Checkpoint file written by train_fused_chunker
        \\  --model-dir <dir>        Model directory (tokenizer + encoder weights)
        \\  --split <name>           Dataset split filter (default: "val")
        \\  --results-out <path>     Write fused_chunker_benchmark_results/v1 JSON
        \\  --benchmark-dataset-name <name>
        \\                           Dataset name to emit in --results-out (default: data basename)
        \\  --internal-phase20-best-f1 <f>
        \\                           Internal validation gate F1 to emit (default: this eval's best F1)
        \\  --output-dimension <n>   Dense output dimension for Voyage target mapping (default: hidden size)
        \\  --baseline <name:ndcg>   Add local retrieval baseline for verifier comparison; repeatable
        \\  --batch-size <n>         Batch size (default: 32)
        \\  --max-examples <n>       Maximum eval examples (default: 0 = all)
        \\  --hidden-size <n>        Hidden size (default: 768)
        \\  --num-layers <n>         ModernBERT layer count (default: 22)
        \\  --max-layers <n>         Alias for --num-layers
        \\  --max-seq-len <n>        Max seq len (default: 384)
        \\  --max-chunks <n>         Max chunks per sample (default: 32)
        \\  --intermediate-size <n>  ModernBERT intermediate_size (default: 1152)
        \\  --boundary-feature-mode token|prev-diff|prev-current-diff|prev-current-diff-concat|prev-current-next-diff-concat|window-context-diff Boundary head input feature view (default: token)
        \\  --boundary-alignment-dump-dir <dir> Write JSONL token alignment dump for gold and high-probability boundary tokens
        \\  --boundary-alignment-dump-top-k <n> High-probability tokens to dump per eval batch (default: 32)
        \\  --backend native|metal|auto  Compute backend (default: auto; auto prefers Metal)
        \\  --compiled-segment-forward   Run the encoder forward through compiled MPSGraph segment sessions
        \\                           (default: off; env ANTFLY_FUSED_CHUNKER_COMPILED_SEGMENT_FORWARD=1;
        \\                           requires --model-dir and a LoRA checkpoint; eager fallback on failure)
        \\
    , .{});
}

// Weight-store loading helpers (normalizeModernBertWeightName, QKV splits,
// safetensors loaders, store deinit) live in ../fused_chunker_weights.zig and
// are shared with the serving pipeline (src/pipelines/fused_chunking.zig).

const FusedLoRALoadResult = struct {
    count: usize = 0,
    rank: u32 = 0,
    alpha: f32 = 0.0,
};

const ParsedRuntimeLoRAKey = struct {
    layer_idx: u32,
    module_name: []const u8,
    is_a: bool,
};

const LoRARuntimeTarget = struct {
    scope: []const u8,
    projection: []const u8,
};

fn loraRuntimeTarget(module_name: []const u8) ?LoRARuntimeTarget {
    if (std.mem.eql(u8, module_name, "query_proj") or
        std.mem.eql(u8, module_name, "key_proj") or
        std.mem.eql(u8, module_name, "value_proj"))
    {
        return .{ .scope = "attn", .projection = module_name };
    }
    if (std.mem.eql(u8, module_name, "out_proj")) return .{ .scope = "attn", .projection = "Wo" };
    if (std.mem.eql(u8, module_name, "wo")) return .{ .scope = "mlp", .projection = "Wo" };
    return null;
}

fn readOptionalScalarF32(reader: *const safetensors.MMapReader, name: []const u8) !?f32 {
    if (!reader.header.tensors.contains(name)) return null;
    var tensor = try reader.readTensor(name);
    defer tensor.deinit();
    if (tensor.dtype != .f32) return error.InvalidFusedLoRAScalar;
    const values = tensor.asFloat32();
    if (values.len != 1) return error.InvalidFusedLoRAScalar;
    return values[0];
}

fn rankFromLoRATensor(name: []const u8, shape: []const i64) !u32 {
    if (shape.len != 2 or shape[0] <= 0 or shape[1] <= 0) return error.InvalidFusedLoRATensor;
    if (std.mem.endsWith(u8, name, ".lora_a")) return @intCast(shape[0]);
    if (std.mem.endsWith(u8, name, ".lora_b")) return @intCast(shape[1]);
    return error.InvalidFusedLoRATensor;
}

const NormalizedLoRATensor = struct {
    key: []const u8,
    values: []const f32,
    shape: [2]i64,
    rank: u32,
    owned_values: ?[]f32 = null,

    fn deinit(self: *NormalizedLoRATensor, allocator: std.mem.Allocator) void {
        if (self.owned_values) |values| allocator.free(values);
        self.* = undefined;
    }
};

fn transpose2DF32Alloc(allocator: std.mem.Allocator, values: []const f32, rows: usize, cols: usize) ![]f32 {
    if (values.len != rows * cols) return error.InvalidFusedLoRATensor;
    const out = try allocator.alloc(f32, values.len);
    for (0..rows) |r| {
        for (0..cols) |c| {
            out[c * rows + r] = values[r * cols + c];
        }
    }
    return out;
}

fn parseRuntimeLoRAKey(key: []const u8) !?ParsedRuntimeLoRAKey {
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, key, prefix)) return null;
    var rest = key[prefix.len..];
    const layer_end = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer_idx = try std.fmt.parseUnsigned(u32, rest[0..layer_end], 10);
    rest = rest[layer_end + 1 ..];

    const scope_end = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const scope = rest[0..scope_end];
    rest = rest[scope_end + 1 ..];

    const projection_end = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const projection = rest[0..projection_end];
    const suffix = rest[projection_end + 1 ..];
    const is_a = if (std.mem.eql(u8, suffix, "lora_a"))
        true
    else if (std.mem.eql(u8, suffix, "lora_b"))
        false
    else
        return null;

    const module_name: []const u8 = if (std.mem.eql(u8, scope, "attn")) blk: {
        if (std.mem.eql(u8, projection, "Wo")) break :blk "out_proj";
        if (std.mem.eql(u8, projection, "query_proj") or
            std.mem.eql(u8, projection, "key_proj") or
            std.mem.eql(u8, projection, "value_proj"))
        {
            break :blk projection;
        }
        return null;
    } else if (std.mem.eql(u8, scope, "mlp") and std.mem.eql(u8, projection, "Wo"))
        "wo"
    else
        return null;

    return .{
        .layer_idx = layer_idx,
        .module_name = module_name,
        .is_a = is_a,
    };
}

fn inspectFusedLoRAMetadata(
    allocator: std.mem.Allocator,
    checkpoint_path: []const u8,
) !FusedLoRALoadResult {
    const file_bytes = try compat.cwd().readFileAlloc(compat.io(), checkpoint_path, allocator, .unlimited);
    var reader = safetensors.MMapReader.fromBytes(allocator, file_bytes) catch |err| {
        allocator.free(file_bytes);
        return err;
    };
    defer reader.deinit();

    var result = FusedLoRALoadResult{};
    if (try readOptionalScalarF32(&reader, "lora_rank")) |rank_value| {
        if (rank_value < 0) return error.InvalidFusedLoRAScalar;
        result.rank = @intFromFloat(rank_value);
    }
    result.alpha = (try readOptionalScalarF32(&reader, "lora_alpha")) orelse 32.0;

    const names = try reader.header.tensorNames(allocator);
    defer allocator.free(names);

    for (names) |name| {
        if (std.mem.indexOf(u8, name, ".lora_") == null and
            std.mem.indexOf(u8, name, "/lora_") == null)
        {
            continue;
        }
        var tensor = try reader.readTensor(name);
        defer tensor.deinit();
        if (tensor.dtype != .f32 or tensor.shape.len != 2) return error.InvalidFusedLoRATensor;

        var key_buf: [256]u8 = undefined;
        var normalized = (try normalizeFusedLoRATensor(
            allocator,
            name,
            tensor.asFloat32(),
            tensor.shape,
            &key_buf,
        )) orelse continue;
        defer normalized.deinit(allocator);

        const parsed = (try parseRuntimeLoRAKey(normalized.key)) orelse continue;
        _ = parsed;
        const tensor_rank = normalized.rank;
        if (result.rank == 0) {
            result.rank = tensor_rank;
        } else if (result.rank != tensor_rank) {
            return error.FusedLoRARankMismatch;
        }
        result.count += 1;
    }

    if (result.count == 0) return .{};
    if (result.rank == 0 or result.alpha == 0.0) return error.InvalidFusedLoRAScalar;
    return result;
}

fn copyNormalizedLoRATensorIntoAdapter(
    adapter: *fused_chunker_lora.LoRAAdapterSet,
    normalized: NormalizedLoRATensor,
) !bool {
    const parsed = (try parseRuntimeLoRAKey(normalized.key)) orelse return false;
    const layer = adapter.get(parsed.layer_idx, parsed.module_name) orelse return false;
    const dest = if (parsed.is_a) layer.A else layer.B;
    if (normalized.values.len != dest.len) return error.InvalidFusedLoRATensor;

    if (parsed.is_a) {
        if (normalized.shape[0] != @as(i64, @intCast(adapter.config.rank)) or
            normalized.shape[1] != @as(i64, @intCast(layer.in_features)))
        {
            return error.InvalidFusedLoRATensor;
        }
    } else {
        if (normalized.shape[0] != @as(i64, @intCast(layer.out_features)) or
            normalized.shape[1] != @as(i64, @intCast(adapter.config.rank)))
        {
            return error.InvalidFusedLoRATensor;
        }
    }
    @memcpy(dest, normalized.values);
    return true;
}

fn loadFusedLoRAAdapterSetFromCheckpoint(
    allocator: std.mem.Allocator,
    checkpoint_path: []const u8,
    adapter: *fused_chunker_lora.LoRAAdapterSet,
) !usize {
    const file_bytes = try compat.cwd().readFileAlloc(compat.io(), checkpoint_path, allocator, .unlimited);
    var reader = safetensors.MMapReader.fromBytes(allocator, file_bytes) catch |err| {
        allocator.free(file_bytes);
        return err;
    };
    defer reader.deinit();

    const names = try reader.header.tensorNames(allocator);
    defer allocator.free(names);

    var loaded: usize = 0;
    for (names) |name| {
        if (std.mem.indexOf(u8, name, ".lora_") == null and
            std.mem.indexOf(u8, name, "/lora_") == null)
        {
            continue;
        }
        var tensor = try reader.readTensor(name);
        defer tensor.deinit();
        if (tensor.dtype != .f32 or tensor.shape.len != 2) return error.InvalidFusedLoRATensor;

        var key_buf: [256]u8 = undefined;
        var normalized = (try normalizeFusedLoRATensor(
            allocator,
            name,
            tensor.asFloat32(),
            tensor.shape,
            &key_buf,
        )) orelse continue;
        defer normalized.deinit(allocator);

        if (try copyNormalizedLoRATensorIntoAdapter(adapter, normalized)) loaded += 1;
    }
    return loaded;
}

fn normalizeGoFusedLoRAName(name: []const u8, key_buf: *[256]u8) !?struct {
    key: []const u8,
    is_a: bool,
} {
    var clean = name;
    if (std.mem.startsWith(u8, clean, "var:/")) clean = clean["var:/".len..];
    if (std.mem.startsWith(u8, clean, "fused_chunker_embedder/")) {
        clean = clean["fused_chunker_embedder/".len..];
    }
    if (!std.mem.startsWith(u8, clean, "encoder/layer/")) return null;
    var rest = clean["encoder/layer/".len..];
    const layer_end = std.mem.indexOfScalar(u8, rest, '/') orelse return null;
    const layer = try std.fmt.parseUnsigned(u32, rest[0..layer_end], 10);
    rest = rest[layer_end + 1 ..];

    var scope: []const u8 = undefined;
    var projection: []const u8 = undefined;
    var component: []const u8 = undefined;

    if (std.mem.startsWith(u8, rest, "attn/")) {
        rest = rest["attn/".len..];
        const proj_end = std.mem.indexOfScalar(u8, rest, '/') orelse return null;
        scope = "attn";
        projection = rest[0..proj_end];
        component = rest[proj_end + 1 ..];
    } else if (std.mem.startsWith(u8, rest, "attention/self/")) {
        rest = rest["attention/self/".len..];
        const proj_end = std.mem.indexOfScalar(u8, rest, '/') orelse return null;
        scope = "attn";
        projection = rest[0..proj_end];
        component = rest[proj_end + 1 ..];
    } else if (std.mem.startsWith(u8, rest, "mlp/Wo/")) {
        scope = "mlp";
        projection = "Wo";
        component = rest["mlp/Wo/".len..];
    } else {
        return null;
    }

    const suffix = if (std.mem.eql(u8, component, "lora_A"))
        "lora_a"
    else if (std.mem.eql(u8, component, "lora_B"))
        "lora_b"
    else
        return null;

    return .{
        .key = std.fmt.bufPrint(key_buf, "model.layers.{d}.{s}.{s}.{s}", .{ layer, scope, projection, suffix }) catch return error.NameTooLong,
        .is_a = std.mem.eql(u8, suffix, "lora_a"),
    };
}

fn normalizeFusedLoRATensor(
    allocator: std.mem.Allocator,
    name: []const u8,
    values: []const f32,
    shape: []const i64,
    key_buf: *[256]u8,
) !?NormalizedLoRATensor {
    if (shape.len != 2 or shape[0] <= 0 or shape[1] <= 0) return error.InvalidFusedLoRATensor;

    if (std.mem.endsWith(u8, name, ".lora_a") or std.mem.endsWith(u8, name, ".lora_b")) {
        return .{
            .key = name,
            .values = values,
            .shape = .{ shape[0], shape[1] },
            .rank = try rankFromLoRATensor(name, shape),
        };
    }

    const go_name = (try normalizeGoFusedLoRAName(name, key_buf)) orelse return null;
    const rows: usize = @intCast(shape[0]);
    const cols: usize = @intCast(shape[1]);
    const transposed = try transpose2DF32Alloc(allocator, values, rows, cols);
    return .{
        .key = go_name.key,
        .values = transposed,
        .shape = .{ shape[1], shape[0] },
        .rank = if (go_name.is_a) @intCast(shape[1]) else @intCast(shape[0]),
        .owned_values = transposed,
    };
}

test "eval fused LoRA normalization accepts Go checkpoint tensors" {
    const allocator = std.testing.allocator;

    const a_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const a_shape = [_]i64{ 3, 2 };
    var a_key_buf: [256]u8 = undefined;
    var normalized_a = (try normalizeFusedLoRATensor(
        allocator,
        "var:/fused_chunker_embedder/encoder/layer/7/mlp/Wo/lora_A",
        a_values[0..],
        a_shape[0..],
        &a_key_buf,
    )).?;
    defer normalized_a.deinit(allocator);

    try std.testing.expectEqualStrings("model.layers.7.mlp.Wo.lora_a", normalized_a.key);
    try std.testing.expectEqual(@as(u32, 2), normalized_a.rank);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, normalized_a.shape[0..]);
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 5, 2, 4, 6 }, normalized_a.values);

    const b_values = [_]f32{ 10, 11, 12, 13, 14, 15, 16, 17 };
    const b_shape = [_]i64{ 2, 4 };
    var b_key_buf: [256]u8 = undefined;
    var normalized_b = (try normalizeFusedLoRATensor(
        allocator,
        "fused_chunker_embedder/encoder/layer/3/attn/query_proj/lora_B",
        b_values[0..],
        b_shape[0..],
        &b_key_buf,
    )).?;
    defer normalized_b.deinit(allocator);

    try std.testing.expectEqualStrings("model.layers.3.attn.query_proj.lora_b", normalized_b.key);
    try std.testing.expectEqual(@as(u32, 2), normalized_b.rank);
    try std.testing.expectEqualSlices(i64, &.{ 4, 2 }, normalized_b.shape[0..]);
    try std.testing.expectEqualSlices(f32, &.{ 10, 14, 11, 15, 12, 16, 13, 17 }, normalized_b.values);

    const old_b_values = [_]f32{ 20, 21, 22, 23 };
    const old_b_shape = [_]i64{ 2, 2 };
    var old_b_key_buf: [256]u8 = undefined;
    var normalized_old_b = (try normalizeFusedLoRATensor(
        allocator,
        "var:/fused_chunker_embedder/encoder/layer/4/attention/self/value_proj/lora_B",
        old_b_values[0..],
        old_b_shape[0..],
        &old_b_key_buf,
    )).?;
    defer normalized_old_b.deinit(allocator);

    try std.testing.expectEqualStrings("model.layers.4.attn.value_proj.lora_b", normalized_old_b.key);
    try std.testing.expectEqual(@as(u32, 2), normalized_old_b.rank);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, normalized_old_b.shape[0..]);
    try std.testing.expectEqualSlices(f32, &.{ 20, 22, 21, 23 }, normalized_old_b.values);

    const zig_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const zig_shape = [_]i64{ 2, 3 };
    var zig_key_buf: [256]u8 = undefined;
    var normalized_zig = (try normalizeFusedLoRATensor(
        allocator,
        "model.layers.1.mlp.Wo.lora_a",
        zig_values[0..],
        zig_shape[0..],
        &zig_key_buf,
    )).?;
    defer normalized_zig.deinit(allocator);

    try std.testing.expectEqualStrings("model.layers.1.mlp.Wo.lora_a", normalized_zig.key);
    try std.testing.expectEqual(@as(u32, 2), normalized_zig.rank);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, normalized_zig.shape[0..]);
    try std.testing.expect(normalized_zig.values.ptr == zig_values[0..].ptr);
}

fn insertLoRATensorIntoNativeStore(
    allocator: std.mem.Allocator,
    weight_store: *native_compute.WeightStore,
    key: []const u8,
    values: []const f32,
    shape: []const i64,
) !void {
    if (shape.len != 2 or shape[0] <= 0 or shape[1] <= 0) return error.InvalidFusedLoRATensor;
    const shape_i64 = [_]i64{ shape[0], shape[1] };
    const owned_key = try allocator.dupe(u8, key);
    errdefer allocator.free(owned_key);
    var tensor = try tensor_mod.Tensor.initFloat32(allocator, owned_key, &shape_i64, values);
    errdefer tensor.deinit();
    try weight_store.resident_weights.put(allocator, owned_key, LoadedWeight{ .tensor = tensor });
}

fn insertLoRATensorIntoMetalStore(
    allocator: std.mem.Allocator,
    weight_store: *MetalWeightStore,
    key: []const u8,
    values: []const f32,
    shape: []const i64,
) !void {
    if (comptime !build_options.enable_metal) return error.MetalBackendUnavailable;
    if (shape.len != 2 or shape[0] <= 0 or shape[1] <= 0) return error.InvalidFusedLoRATensor;
    const shape_i64 = [_]i64{ shape[0], shape[1] };
    if (weight_store.lazy_weights.getPtr(key)) |entry| {
        const src_bytes = std.mem.sliceAsBytes(values);
        if (entry.host_loaded) |*loaded| {
            if (loaded.tensor.dtype == .f32 and
                loaded.tensor.shape.len == shape_i64.len and
                loaded.tensor.shape[0] == shape_i64[0] and
                loaded.tensor.shape[1] == shape_i64[1] and
                loaded.tensor.data.len == src_bytes.len)
            {
                @memcpy(loaded.tensor.data, src_bytes);
                entry.active_tier = .host;
                entry.loaded_bytes = loaded.tensor.data.len;
                return;
            }
            loaded.deinit();
        }
        if (entry.quantized_storage) |*storage| storage.deinit();
        var tensor = try tensor_mod.Tensor.initFloat32(allocator, key, &shape_i64, values);
        errdefer tensor.deinit();
        entry.* = .{
            .tensor_ref = undefined,
            .host_loaded = .{ .tensor = tensor },
            .active_tier = .host,
            .loaded_bytes = tensor.data.len,
        };
        return;
    }
    const owned_key = try allocator.dupe(u8, key);
    errdefer allocator.free(owned_key);
    var tensor = try tensor_mod.Tensor.initFloat32(allocator, owned_key, &shape_i64, values);
    errdefer tensor.deinit();
    try weight_store.lazy_weights.put(allocator, owned_key, .{
        .tensor_ref = undefined,
        .host_loaded = .{ .tensor = tensor },
        .active_tier = .host,
        .loaded_bytes = tensor.data.len,
    });
}

fn refreshLoRAWeightsForForward(
    allocator: std.mem.Allocator,
    use_metal: bool,
    native_weight_store: *native_compute.WeightStore,
    metal_weight_store: *MetalWeightStore,
    adapter: *const fused_chunker_lora.LoRAAdapterSet,
) !void {
    const rank: usize = @intCast(adapter.config.rank);
    for (adapter.layers) |*ll| {
        const target = loraRuntimeTarget(ll.module_name) orelse continue;

        var key_buf_a: [128]u8 = undefined;
        var key_buf_b: [128]u8 = undefined;
        const key_a = try std.fmt.bufPrint(
            &key_buf_a,
            "model.layers.{d}.{s}.{s}.lora_a",
            .{ ll.layer_idx, target.scope, target.projection },
        );
        const key_b = try std.fmt.bufPrint(
            &key_buf_b,
            "model.layers.{d}.{s}.{s}.lora_b",
            .{ ll.layer_idx, target.scope, target.projection },
        );

        if (use_metal) {
            try insertLoRATensorIntoMetalStore(allocator, metal_weight_store, key_a, ll.A, &.{ @intCast(rank), @intCast(ll.in_features) });
            try insertLoRATensorIntoMetalStore(allocator, metal_weight_store, key_b, ll.B, &.{ @intCast(ll.out_features), @intCast(rank) });
        } else {
            try insertLoRATensorIntoNativeStore(allocator, native_weight_store, key_a, ll.A, &.{ @intCast(rank), @intCast(ll.in_features) });
            try insertLoRATensorIntoNativeStore(allocator, native_weight_store, key_b, ll.B, &.{ @intCast(ll.out_features), @intCast(rank) });
        }
    }
}

fn loadFusedLoRAFromCheckpoint(
    allocator: std.mem.Allocator,
    checkpoint_path: []const u8,
    use_metal: bool,
    native_weight_store: *native_compute.WeightStore,
    metal_weight_store: *MetalWeightStore,
) !FusedLoRALoadResult {
    const file_bytes = try compat.cwd().readFileAlloc(compat.io(), checkpoint_path, allocator, .unlimited);
    var reader = safetensors.MMapReader.fromBytes(allocator, file_bytes) catch |err| {
        allocator.free(file_bytes);
        return switch (err) {
            error.FileTooSmall,
            error.HeaderTooLarge,
            error.EmptyHeader,
            error.FileTruncated,
            error.InvalidHeader,
            error.UnsupportedDType,
            error.MissingShape,
            error.InvalidShape,
            error.MissingOffsets,
            error.InvalidOffset,
            => FusedLoRALoadResult{},
            else => err,
        };
    };
    defer reader.deinit();

    var result = FusedLoRALoadResult{};
    if (try readOptionalScalarF32(&reader, "lora_rank")) |rank_value| {
        if (rank_value < 0) return error.InvalidFusedLoRAScalar;
        result.rank = @intFromFloat(rank_value);
    }
    result.alpha = (try readOptionalScalarF32(&reader, "lora_alpha")) orelse 32.0;

    const names = try reader.header.tensorNames(allocator);
    defer allocator.free(names);

    for (names) |name| {
        if (std.mem.indexOf(u8, name, ".lora_") == null and
            std.mem.indexOf(u8, name, "/lora_") == null)
        {
            continue;
        }
        var tensor = try reader.readTensor(name);
        defer tensor.deinit();
        if (tensor.dtype != .f32 or tensor.shape.len != 2) return error.InvalidFusedLoRATensor;

        var key_buf: [256]u8 = undefined;
        var normalized = (try normalizeFusedLoRATensor(
            allocator,
            name,
            tensor.asFloat32(),
            tensor.shape,
            &key_buf,
        )) orelse continue;
        defer normalized.deinit(allocator);

        const tensor_rank = normalized.rank;
        if (result.rank == 0) {
            result.rank = tensor_rank;
        } else if (result.rank != tensor_rank) {
            return error.FusedLoRARankMismatch;
        }

        const expected_len: usize = @as(usize, @intCast(normalized.shape[0])) * @as(usize, @intCast(normalized.shape[1]));
        if (normalized.values.len != expected_len) return error.InvalidFusedLoRATensor;

        if (use_metal) {
            try insertLoRATensorIntoMetalStore(allocator, metal_weight_store, normalized.key, normalized.values, normalized.shape[0..]);
        } else {
            try insertLoRATensorIntoNativeStore(allocator, native_weight_store, normalized.key, normalized.values, normalized.shape[0..]);
        }
        result.count += 1;
    }

    if (result.count == 0) return FusedLoRALoadResult{};
    if (result.rank == 0 or result.alpha == 0.0) return error.InvalidFusedLoRAScalar;
    return result;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next(); // skip argv[0]

    var data_path: ?[]const u8 = null;
    var checkpoint_path: ?[]const u8 = null;
    var model_dir: ?[]const u8 = null;
    var split: []const u8 = "val";
    var batch_size: u32 = 32;
    var max_examples: usize = 0;
    var hidden_size: u32 = 768;
    var num_layers: u32 = 22;
    var max_seq_len: u32 = 384;
    var max_chunks: u32 = 32;
    var intermediate_size: u32 = 1152;
    var boundary_feature_mode: fused_chunker_train.BoundaryFeatureMode = .token;
    var boundary_alignment_dump_dir: ?[]const u8 = null;
    var boundary_alignment_dump_top_k: usize = 32;
    var results_out: ?[]const u8 = null;
    var benchmark_dataset_name: ?[]const u8 = null;
    var internal_phase20_best_f1: ?f64 = null;
    var output_dimension: ?u32 = null;
    var retrieval_baselines = std.ArrayListUnmanaged(BenchmarkBaseline).empty;
    defer retrieval_baselines.deinit(allocator);
    var backend: @TypeOf((Options{
        .data_path = "",
        .checkpoint_path = "",
    }).backend) = .auto;
    var compiled_segment_forward: bool = fused_chunker_compiled_forward.envEnabled();

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--data")) {
            data_path = args.next() orelse {
                print("error: --data requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--checkpoint")) {
            checkpoint_path = args.next() orelse {
                print("error: --checkpoint requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--model-dir")) {
            model_dir = args.next() orelse {
                print("error: --model-dir requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--split")) {
            split = args.next() orelse {
                print("error: --split requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--results-out")) {
            results_out = args.next() orelse {
                print("error: --results-out requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--benchmark-dataset-name")) {
            benchmark_dataset_name = args.next() orelse {
                print("error: --benchmark-dataset-name requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--internal-phase20-best-f1")) {
            const value = args.next() orelse {
                print("error: --internal-phase20-best-f1 requires a value\n", .{});
                std.process.exit(1);
            };
            internal_phase20_best_f1 = std.fmt.parseFloat(f64, value) catch {
                print("error: invalid --internal-phase20-best-f1 value: {s}\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--output-dimension")) {
            const value = args.next() orelse {
                print("error: --output-dimension requires a value\n", .{});
                std.process.exit(1);
            };
            output_dimension = std.fmt.parseUnsigned(u32, value, 10) catch {
                print("error: invalid --output-dimension value: {s}\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--baseline")) {
            const value = args.next() orelse {
                print("error: --baseline requires a value\n", .{});
                std.process.exit(1);
            };
            const baseline = parseBenchmarkBaseline(value) catch {
                print("error: invalid --baseline value '{s}', expected name:ndcg\n", .{value});
                std.process.exit(1);
            };
            try retrieval_baselines.append(allocator, baseline);
        } else if (std.mem.eql(u8, arg, "--batch-size")) {
            const value = args.next() orelse {
                print("error: --batch-size requires a value\n", .{});
                std.process.exit(1);
            };
            batch_size = std.fmt.parseUnsigned(u32, value, 10) catch {
                print("error: invalid --batch-size value: {s}\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--max-examples")) {
            const value = args.next() orelse {
                print("error: --max-examples requires a value\n", .{});
                std.process.exit(1);
            };
            max_examples = std.fmt.parseUnsigned(usize, value, 10) catch {
                print("error: invalid --max-examples value: {s}\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--hidden-size")) {
            const value = args.next() orelse {
                print("error: --hidden-size requires a value\n", .{});
                std.process.exit(1);
            };
            hidden_size = std.fmt.parseUnsigned(u32, value, 10) catch {
                print("error: invalid --hidden-size value: {s}\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--num-layers") or std.mem.eql(u8, arg, "--max-layers")) {
            const value = args.next() orelse {
                print("error: {s} requires a value\n", .{arg});
                std.process.exit(1);
            };
            num_layers = std.fmt.parseUnsigned(u32, value, 10) catch {
                print("error: invalid {s} value: {s}\n", .{ arg, value });
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--max-seq-len")) {
            const value = args.next() orelse {
                print("error: --max-seq-len requires a value\n", .{});
                std.process.exit(1);
            };
            max_seq_len = std.fmt.parseUnsigned(u32, value, 10) catch {
                print("error: invalid --max-seq-len value: {s}\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--max-chunks")) {
            const value = args.next() orelse {
                print("error: --max-chunks requires a value\n", .{});
                std.process.exit(1);
            };
            max_chunks = std.fmt.parseUnsigned(u32, value, 10) catch {
                print("error: invalid --max-chunks value: {s}\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--intermediate-size")) {
            const value = args.next() orelse {
                print("error: --intermediate-size requires a value\n", .{});
                std.process.exit(1);
            };
            intermediate_size = std.fmt.parseUnsigned(u32, value, 10) catch {
                print("error: invalid --intermediate-size value: {s}\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--boundary-feature-mode")) {
            const value = args.next() orelse {
                print("error: --boundary-feature-mode requires a value\n", .{});
                std.process.exit(1);
            };
            boundary_feature_mode = parseBoundaryFeatureMode(value) catch {
                print("error: unknown boundary feature mode '{s}': expected token, prev-diff, prev-current-diff, prev-current-diff-concat, prev-current-next-diff-concat, or window-context-diff\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--boundary-alignment-dump-dir")) {
            boundary_alignment_dump_dir = args.next() orelse {
                print("error: --boundary-alignment-dump-dir requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--boundary-alignment-dump-top-k")) {
            const value = args.next() orelse {
                print("error: --boundary-alignment-dump-top-k requires a value\n", .{});
                std.process.exit(1);
            };
            boundary_alignment_dump_top_k = std.fmt.parseUnsigned(usize, value, 10) catch {
                print("error: invalid --boundary-alignment-dump-top-k value: {s}\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--backend")) {
            const value = args.next() orelse {
                print("error: --backend requires a value\n", .{});
                std.process.exit(1);
            };
            if (std.mem.eql(u8, value, "native")) {
                backend = .native;
            } else if (std.mem.eql(u8, value, "metal")) {
                backend = .metal;
            } else if (std.mem.eql(u8, value, "auto")) {
                backend = .auto;
            } else {
                print("error: unknown backend '{s}': expected native, metal, or auto\n", .{value});
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, arg, "--compiled-segment-forward")) {
            compiled_segment_forward = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return;
        } else {
            print("error: unknown argument: {s}\n", .{arg});
            printUsage();
            std.process.exit(1);
        }
    }

    const opts = Options{
        .data_path = data_path orelse {
            print("error: --data is required\n", .{});
            printUsage();
            std.process.exit(1);
        },
        .checkpoint_path = checkpoint_path orelse {
            print("error: --checkpoint is required\n", .{});
            printUsage();
            std.process.exit(1);
        },
        .model_dir = model_dir,
        .split = split,
        .results_out = results_out,
        .benchmark_dataset_name = benchmark_dataset_name,
        .internal_phase20_best_f1 = internal_phase20_best_f1,
        .output_dimension = output_dimension,
        .retrieval_baselines = retrieval_baselines.items,
        .batch_size = batch_size,
        .max_examples = max_examples,
        .hidden_size = hidden_size,
        .num_layers = num_layers,
        .max_seq_len = max_seq_len,
        .max_chunks = max_chunks,
        .intermediate_size = intermediate_size,
        .boundary_feature_mode = boundary_feature_mode,
        .boundary_alignment_dump_dir = boundary_alignment_dump_dir,
        .boundary_alignment_dump_top_k = boundary_alignment_dump_top_k,
        .backend = backend,
        .compiled_segment_forward = compiled_segment_forward,
    };

    const metal_available = if (comptime build_options.enable_metal) metal_runtime.metalDeviceAvailable() else false;
    const selected_backend: FusedBackend = switch (opts.backend) {
        .auto => if (metal_available) .metal else .native,
        .metal => blk: {
            if (!metal_available) {
                print("error: --backend metal requested but Metal is not compiled in or no Metal device is available\n", .{});
                std.process.exit(1);
            }
            break :blk .metal;
        },
        .native => .native,
    };
    const use_metal = selected_backend == .metal;

    // Checkpoints are trained/validated with the Metal device dense-linear
    // fast path disabled; score them under the same forward unless the
    // caller explicitly overrode the escape hatch.
    if (use_metal and fused_chunker_weights.ensureMetalDenseLinearForwardParityDefault()) {
        print(
            "metal dense-linear device forward disabled for checkpoint parity (set {s}=0 to probe the device path)\n",
            .{fused_chunker_weights.metal_dense_linear_forward_env},
        );
    }

    // Set up compute backend for optional encoder forward and boundary-head eval.
    var weight_store = native_compute.WeightStore{
        .allocator = allocator,
        .resident_weights = .{},
        .lazy_weights = .{},
    };
    defer deinitNativeWeightStore(allocator, &weight_store);

    var native_backend = native_compute.NativeCompute.init(allocator, &weight_store, null);

    var metal_weight_store: MetalWeightStore = undefined;
    var metal_backend: if (build_options.enable_metal) metal_compute.MetalCompute else void = undefined;

    const cb: ComputeBackend = if (use_metal) blk: {
        if (comptime build_options.enable_metal) {
            metal_weight_store = MetalWeightStore{
                .allocator = allocator,
                .prefix = "",
                .lazy_weights = .{},
            };
            metal_compute.initPrefetchQueue(&metal_weight_store, allocator);
            metal_backend = try metal_compute.MetalCompute.init(allocator, &metal_weight_store, null);
            break :blk metal_backend.computeBackend();
        } else unreachable;
    } else native_backend.computeBackend();
    defer if (use_metal) {
        if (comptime build_options.enable_metal) deinitMetalWeightStore(allocator, &metal_weight_store);
    };
    defer if (use_metal) {
        if (comptime build_options.enable_metal) metal_backend.deinit();
    };

    print("backend: {s}\n", .{if (use_metal) "metal" else "native"});

    var tokenizer_opt: ?TokenizerBatch = null;
    defer if (tokenizer_opt) |*tb| tb.deinit();

    var encoder_loaded = false;
    if (opts.model_dir) |mdir| {
        tokenizer_opt = TokenizerBatch.loadFromDir(allocator, mdir, opts.max_seq_len) catch |err| blk: {
            print("warning: could not load tokenizer from {s}: {}\n", .{ mdir, err });
            break :blk null;
        };
        if (tokenizer_opt != null) print("tokenizer loaded from {s}\n", .{mdir});

        var st_path_buf: [512]u8 = undefined;
        const st_path = std.fmt.bufPrint(&st_path_buf, "{s}/model.safetensors", .{mdir}) catch null;
        if (st_path) |p| {
            if (use_metal) {
                if (loadSafetensorsIntoMetalStore(allocator, &metal_weight_store, p)) |loaded_count| {
                    encoder_loaded = true;
                    print("loaded {d} encoder weights into Metal store from {s}\n", .{ loaded_count, p });
                } else |err| {
                    print("warning: could not load encoder weights into Metal store from {s}: {}\n", .{ p, err });
                }
            } else {
                if (loadSafetensorsIntoNativeStore(allocator, &weight_store, p)) |loaded_count| {
                    encoder_loaded = true;
                    print("loaded {d} encoder weights from {s}\n", .{ loaded_count, p });
                } else |err| {
                    print("warning: could not load encoder weights from {s}: {}\n", .{ p, err });
                }
            }
        }
    } else {
        print("model_dir=none (encoder features and retrieval embeddings will be zero-filled)\n", .{});
    }

    // Set up trainer (owns the boundary head weights)
    const config = FusedTrainingConfig{
        .hidden_size = opts.hidden_size,
        .max_seq_len = opts.max_seq_len,
        .max_chunks = opts.max_chunks,
        .batch_size = opts.batch_size,
        .boundary_feature_mode = opts.boundary_feature_mode,
    };
    var trainer = try FusedTrainer.init(allocator, config, &cb);
    defer trainer.deinit();

    // Load checkpoint
    trainer.loadCheckpoint(allocator, opts.checkpoint_path) catch |err| {
        print("error: failed to load checkpoint '{s}': {}\n", .{ opts.checkpoint_path, err });
        std.process.exit(1);
    };

    var eval_lora_rank: u32 = 0;
    var eval_lora_alpha: f32 = 0.0;
    var eval_lora_adapters_opt: ?fused_chunker_lora.LoRAAdapterSet = null;
    defer if (eval_lora_adapters_opt) |*adapter| adapter.deinit();
    if (encoder_loaded) {
        const lora_meta = inspectFusedLoRAMetadata(
            allocator,
            opts.checkpoint_path,
        ) catch |err| blk: {
            print("warning: could not inspect LoRA tensors from checkpoint '{s}': {}\n", .{ opts.checkpoint_path, err });
            break :blk FusedLoRALoadResult{};
        };
        if (lora_meta.count > 0) {
            eval_lora_adapters_opt = try fused_chunker_lora.LoRAAdapterSet.init(
                allocator,
                fused_chunker_lora.LoRAConfig{
                    .rank = lora_meta.rank,
                    .alpha = lora_meta.alpha,
                    .target_modules = fused_chunker_lora_target_modules[0..],
                    .num_layers = opts.num_layers,
                },
                @intCast(opts.hidden_size),
                @intCast(opts.intermediate_size),
            );
            const loaded_lora_tensors = try loadFusedLoRAAdapterSetFromCheckpoint(
                allocator,
                opts.checkpoint_path,
                &eval_lora_adapters_opt.?,
            );
            try refreshLoRAWeightsForForward(
                allocator,
                use_metal,
                &weight_store,
                &metal_weight_store,
                &eval_lora_adapters_opt.?,
            );
            eval_lora_rank = lora_meta.rank;
            eval_lora_alpha = lora_meta.alpha;
            print("loaded {d}/{d} LoRA tensors from checkpoint via adapter set (rank={d} alpha={d:.3})\n", .{
                loaded_lora_tensors,
                lora_meta.count,
                lora_meta.rank,
                lora_meta.alpha,
            });
        }
    }

    // Opt-in compiled MPSGraph segmented encoder forward (capture-free).
    // Sessions compile lazily on the first full batch; any failure latches
    // the eager fallback for the rest of the run.
    var compiled_eval_forward_opt: ?fused_chunker_compiled_forward.CompiledEvalForward = null;
    defer if (compiled_eval_forward_opt) |*cef| cef.deinit();
    if (opts.compiled_segment_forward) {
        if (!encoder_loaded or tokenizer_opt == null) {
            print("compiled_segment_forward requested but encoder weights/tokenizer are unavailable; using eager forward\n", .{});
        } else if (eval_lora_adapters_opt == null or eval_lora_rank == 0) {
            print("compiled_segment_forward requested but the checkpoint has no LoRA adapters; using eager forward\n", .{});
        } else if (fused_chunker_compiled_forward.graphConfigForEval(
            opts.hidden_size,
            opts.num_layers,
            opts.intermediate_size,
        )) |eval_graph_config| {
            compiled_eval_forward_opt = try fused_chunker_compiled_forward.CompiledEvalForward.init(
                allocator,
                eval_graph_config,
                1,
                eval_lora_rank,
                eval_lora_alpha,
            );
            print("compiled_segment_forward=on rank={d} alpha={d:.3} (eager fallback on failure)\n", .{
                eval_lora_rank,
                eval_lora_alpha,
            });
        } else |err| {
            print("compiled_segment_forward requested but unsupported for this config ({s}); using eager forward\n", .{@errorName(err)});
        }
    }

    // Load eval samples
    var loaded = fused_chunker_data.loadSamples(allocator, opts.data_path, opts.split) catch |err| {
        print("error: failed to load eval data from '{s}': {}\n", .{ opts.data_path, err });
        std.process.exit(1);
    };
    defer loaded.deinit();

    const samples = if (opts.max_examples > 0 and opts.max_examples < loaded.samples.len)
        loaded.samples[0..opts.max_examples]
    else
        loaded.samples;
    if (samples.len == 0) {
        print("error: no samples found in '{s}' for split '{s}'\n", .{ opts.data_path, opts.split });
        std.process.exit(1);
    }

    print("Loaded {d}/{d} eval samples from '{s}' (split='{s}')\n", .{
        samples.len,
        loaded.samples.len,
        opts.data_path,
        opts.split,
    });

    var boundary_acc = fused_chunker_train.BoundaryEvalAccumulator{};
    defer boundary_acc.deinit(allocator);
    var chonky_separator_acc = ChonkySeparatorAccumulator{};
    var chonky_separator_available = false;

    // Accumulate chunk-level data for dense retrieval evaluation.
    // chunk_embeddings_all: flat [total_chunks * hs] from mean-pooled encoder features.
    // chunk_mask_all:       flat [total_chunks] — 1.0 valid
    // chunk_doc_ids_all:    flat [total_chunks] — global sample index for each chunk
    var chunk_embeddings_all = std.ArrayListUnmanaged(f32).empty;
    defer chunk_embeddings_all.deinit(allocator);

    var chunk_mask_all = std.ArrayListUnmanaged(f32).empty;
    defer chunk_mask_all.deinit(allocator);

    var chunk_doc_ids_all = std.ArrayListUnmanaged(u32).empty;
    defer chunk_doc_ids_all.deinit(allocator);

    // Batch up samples
    const bs: usize = @intCast(opts.batch_size);
    const msl: usize = @intCast(opts.max_seq_len);
    const mc: usize = @intCast(opts.max_chunks);
    const hs: usize = @intCast(opts.hidden_size);

    var alignment_path: ?[]u8 = null;
    defer if (alignment_path) |path| allocator.free(path);
    var alignment_file: ?std.Io.File = null;
    if (opts.boundary_alignment_dump_dir) |dump_dir| {
        try compat.cwd().createDirPath(compat.io(), dump_dir);
        const path = try boundaryAlignmentDumpPath(allocator, dump_dir, opts.split, 0);
        alignment_path = path;
        alignment_file = try compat.cwd().createFile(compat.io(), path, .{ .truncate = true });
        print("Writing boundary alignment dump to {s}\n", .{path});
    }
    defer if (alignment_file) |*file| file.close(compat.io());
    var alignment_buf: [8192]u8 = undefined;
    var alignment_writer = if (alignment_file) |*file| file.writerStreaming(compat.io(), &alignment_buf) else null;
    defer if (alignment_writer) |*writer| writer.interface.flush() catch {};

    // Dummy tokeniser: fills ids/mask with zeros, returns max_seq_len tokens
    const dummy_token_fn = struct {
        fn call(
            _ctx: void,
            text: []const u8,
            out_ids: []i32,
            out_mask: []i32,
            out_offsets: ?[][2]u32,
        ) usize {
            _ = _ctx;
            _ = text;
            @memset(out_ids, 0);
            @memset(out_mask, 1);
            if (out_offsets) |off| @memset(off, .{ 0, 0 });
            return out_ids.len;
        }
    }.call;

    var sample_idx: usize = 0;
    var eval_batch_idx: usize = 0;
    while (sample_idx < samples.len) {
        const end = @min(sample_idx + bs, samples.len);
        const count = end - sample_idx;

        // Build index slice for this batch
        const indices = try allocator.alloc(usize, count);
        defer allocator.free(indices);
        for (0..count) |k| indices[k] = sample_idx + k;

        var batch: fused_chunker_data.FusedBatch = undefined;
        if (tokenizer_opt) |*tb| {
            var tok_ctx = tb.makeTokenFnCtx();
            batch = try fused_chunker_data.assembleTokenBatch(
                allocator,
                samples,
                indices,
                msl,
                mc,
                &tok_ctx,
                TokenFnCtx.call,
            );
        } else {
            batch = try fused_chunker_data.assembleTokenBatch(
                allocator,
                samples,
                indices,
                msl,
                mc,
                {},
                dummy_token_fn,
            );
        }
        defer batch.deinit(allocator);

        // total tokens in this batch = batch_size * max_seq_len (all tokens active)
        const total_tokens: usize = count * msl;

        const features = if (encoder_loaded and tokenizer_opt != null) blk: {
            const ids_i64 = try allocator.alloc(i64, total_tokens);
            defer allocator.free(ids_i64);
            for (batch.input_ids[0..total_tokens], ids_i64) |id32, *id64| id64.* = @intCast(id32);

            const mask_i64 = try allocator.alloc(i64, total_tokens);
            defer allocator.free(mask_i64);
            for (batch.attention_mask[0..total_tokens], mask_i64) |m32, *m64| m64.* = @intCast(m32);

            const bert_config = modern_bert.Config{
                .hidden_size = opts.hidden_size,
                .num_hidden_layers = opts.num_layers,
                .intermediate_size = opts.intermediate_size,
                .lora_rank = eval_lora_rank,
                .lora_alpha = eval_lora_alpha,
            };
            // Opt-in compiled segmented forward; partial trailing batches
            // and any compiled failure fall back to the eager forward.
            if (compiled_eval_forward_opt) |*cef| {
                if (cef.shouldUse(count, bs)) {
                    if (cef.forward(
                        &cb,
                        bert_config,
                        ids_i64,
                        mask_i64,
                        count,
                        msl,
                        eval_lora_adapters_opt.?.layers,
                    )) |compiled_features| {
                        break :blk compiled_features;
                    } else |err| {
                        cef.failed = true;
                        print(
                            "warning: compiled segment forward failed on eval batch starting at sample {d}: {s}; falling back to eager forward\n",
                            .{ sample_idx, @errorName(err) },
                        );
                    }
                }
            }
            break :blk modern_bert.forward(
                &cb,
                allocator,
                bert_config,
                ids_i64,
                mask_i64,
                count,
                msl,
            ) catch |err| fblk: {
                print("warning: encoder forward failed on eval batch starting at sample {d}: {}\n", .{ sample_idx, err });
                const zeros = try allocator.alloc(f32, total_tokens * hs);
                @memset(zeros, 0.0);
                break :fblk zeros;
            };
        } else zblk: {
            const zeros = try allocator.alloc(f32, total_tokens * hs);
            @memset(zeros, 0.0);
            break :zblk zeros;
        };
        defer allocator.free(features);

        // Build one-hot boundary labels [total_tokens * 2] from flat boundary_labels
        const labels = try allocator.alloc(f32, total_tokens * 2);
        defer allocator.free(labels);
        for (0..total_tokens) |t| {
            const is_boundary = batch.boundary_labels[t] > 0.5;
            labels[t * 2 + 0] = if (is_boundary) 0.0 else 1.0;
            labels[t * 2 + 1] = if (is_boundary) 1.0 else 0.0;
        }

        // Build f32 attention mask [total_tokens] from i32 attention_mask
        const mask = try allocator.alloc(f32, total_tokens);
        defer allocator.free(mask);
        for (0..total_tokens) |t| {
            mask[t] = if (batch.attention_mask[t] != 0) 1.0 else 0.0;
        }

        const logits = try trainer.evaluateBoundaryLogitsOwnedWithMask(allocator, features, mask, total_tokens);
        defer allocator.free(logits);

        try boundary_acc.addLogitsBySample(allocator, logits, labels, mask, msl);
        if (alignment_writer) |*writer| {
            try writeBoundaryAlignmentDumpBatch(
                allocator,
                &writer.interface,
                opts.split,
                0,
                eval_batch_idx + 1,
                &batch,
                logits,
                features,
                hs,
                opts.boundary_alignment_dump_top_k,
            );
        }
        if (tokenizer_opt) |*tb| {
            try accumulateChonkySeparatorBatch(
                allocator,
                &chonky_separator_acc,
                tb,
                samples,
                indices,
                logits,
                mask,
                msl,
                0.5,
            );
            chonky_separator_available = true;
        }

        const num_chunks_batch = count * mc;
        const chunk_embeddings = try fused_chunker_data.meanPoolChunkEmbeddings(allocator, features, &batch, hs);
        defer allocator.free(chunk_embeddings);
        try chunk_embeddings_all.appendSlice(allocator, chunk_embeddings);
        // chunk_mask from the assembled batch
        try chunk_mask_all.appendSlice(allocator, batch.chunk_mask[0..num_chunks_batch]);
        // doc_ids: the global sample index for each chunk position in the batch.
        // Chunk position c within the batch belongs to sample (sample_idx + c / mc).
        for (0..num_chunks_batch) |c| {
            const global_sample = sample_idx + c / mc;
            try chunk_doc_ids_all.append(allocator, @intCast(global_sample));
        }

        eval_batch_idx += 1;
        sample_idx = end;
    }

    const summary = try boundary_acc.finish(allocator);
    const chonky_separator_metrics: ?SeparatorBoundaryMetrics = if (chonky_separator_available)
        chonky_separator_acc.finish()
    else
        null;

    // Print results
    print("\n=== Eval Results ===\n", .{});
    print("Checkpoint:  {s}\n", .{opts.checkpoint_path});
    print("Data:        {s}\n", .{opts.data_path});
    print("Batches:     {d}\n", .{summary.num_batches});
    print("F1:          {d:.4}\n", .{summary.boundary_f1});
    print("Precision:   {d:.4}\n", .{summary.boundary_precision});
    print("Recall:      {d:.4}\n", .{summary.boundary_recall});
    print("Counts:      tp={d} fp={d} fn={d}\n", .{ summary.boundary_tp, summary.boundary_fp, summary.boundary_fn });
    print("Best F1:     {d:.4} @ threshold={d:.2}\n", .{ summary.best_boundary_f1, summary.best_boundary_threshold });
    print("Best Prec:   {d:.4}\n", .{summary.best_boundary_precision});
    print("Best Recall: {d:.4}\n", .{summary.best_boundary_recall});
    print("Best Counts: tp={d} fp={d} fn={d}\n", .{ summary.best_boundary_tp, summary.best_boundary_fp, summary.best_boundary_fn });
    print("AP:          {d:.6}\n", .{summary.average_precision});
    print("P@Gold:      {d:.4} R@Gold: {d:.4} F1@Gold: {d:.4} threshold={d:.6} pred_rate={d:.6}\n", .{
        summary.precision_at_gold_count,
        summary.recall_at_gold_count,
        summary.f1_at_gold_count,
        summary.threshold_at_gold_count,
        summary.predicted_positive_rate_at_gold_count,
    });
    print("Rank Best:   f1={d:.4} precision={d:.4} recall={d:.4} threshold={d:.6} pred_rate={d:.6}\n", .{
        summary.max_rank_f1,
        summary.max_rank_precision,
        summary.max_rank_recall,
        summary.max_rank_threshold,
        summary.max_rank_predicted_positive_rate,
    });
    print("Rank Counts: top_gold tp={d} fp={d} fn={d} best tp={d} fp={d} fn={d}\n", .{
        summary.gold_count_tp,
        summary.gold_count_fp,
        summary.gold_count_fn,
        summary.max_rank_tp,
        summary.max_rank_fp,
        summary.max_rank_fn,
    });
    if (summary.sample_oracle_count_samples > 0) {
        print("Sample Oracle: samples={d} topk_f1={d:.4} topk_p={d:.4} topk_r={d:.4} nms_f1={d:.4} nms_p={d:.4} nms_r={d:.4} nms_radius={d} length_window_f1={d:.4} length_window_p={d:.4} length_window_r={d:.4} length_window_min_radius={d} length_window_radius_fraction={d:.3}\n", .{
            summary.sample_oracle_count_samples,
            summary.sample_oracle_count_topk_f1,
            summary.sample_oracle_count_topk_precision,
            summary.sample_oracle_count_topk_recall,
            summary.sample_oracle_count_nms_f1,
            summary.sample_oracle_count_nms_precision,
            summary.sample_oracle_count_nms_recall,
            summary.sample_oracle_count_nms_radius,
            summary.sample_oracle_count_length_window_f1,
            summary.sample_oracle_count_length_window_precision,
            summary.sample_oracle_count_length_window_recall,
            summary.sample_oracle_count_length_window_min_radius,
            summary.sample_oracle_count_length_window_radius_fraction,
        });
        print("Sample Counts: topk tp={d} fp={d} fn={d} nms tp={d} fp={d} fn={d} length_window tp={d} fp={d} fn={d}\n", .{
            summary.sample_oracle_count_topk_tp,
            summary.sample_oracle_count_topk_fp,
            summary.sample_oracle_count_topk_fn,
            summary.sample_oracle_count_nms_tp,
            summary.sample_oracle_count_nms_fp,
            summary.sample_oracle_count_nms_fn,
            summary.sample_oracle_count_length_window_tp,
            summary.sample_oracle_count_length_window_fp,
            summary.sample_oracle_count_length_window_fn,
        });
    }
    print("Gold Ranks:  mean={d:.2}({d:.4}) median={d}({d:.4}) p90={d}({d:.4}) p99={d}({d:.4}) worst={d}\n", .{
        summary.gold_positive_mean_rank,
        summary.gold_positive_mean_rank_percentile,
        summary.gold_positive_median_rank,
        summary.gold_positive_median_rank_percentile,
        summary.gold_positive_p90_rank,
        summary.gold_positive_p90_rank_percentile,
        summary.gold_positive_p99_rank,
        summary.gold_positive_p99_rank_percentile,
        summary.gold_positive_worst_rank,
    });
    print("Gold Recall: top5x={d}/{d:.4} top10x={d}/{d:.4}\n", .{
        summary.gold_positive_top_5x_count,
        summary.gold_positive_top_5x_recall,
        summary.gold_positive_top_10x_count,
        summary.gold_positive_top_10x_recall,
    });
    print("Valid Tok:   {d}\n", .{summary.valid_tokens});
    print("Gold Pos:    {d} ({d:.6})\n", .{ summary.gold_positives, summary.gold_positive_rate });
    print("Pred Pos:    {d} ({d:.6})\n", .{ summary.predicted_positives, summary.predicted_positive_rate });
    print("Best Pred:   {d} ({d:.6})\n", .{ summary.best_predicted_positives, summary.best_predicted_positive_rate });
    print("Mean P(+):   gold_pos={d:.4} gold_neg={d:.4}\n", .{
        summary.mean_positive_probability_gold_positive,
        summary.mean_positive_probability_gold_negative,
    });
    if (chonky_separator_metrics) |separator| {
        print("Chonky F1:   {d:.4} precision={d:.4} recall={d:.4} counts tp={d} fp={d} fn={d}\n", .{
            separator.f1,
            separator.precision,
            separator.recall,
            separator.tp,
            separator.fp,
            separator.fn_count,
        });
    }
    print("Mean Margin: gold_pos={d:.4} gold_neg={d:.4}\n", .{
        summary.mean_boundary_margin_gold_positive,
        summary.mean_boundary_margin_gold_negative,
    });
    print("Mean Logits: gold_pos=({d:.4},{d:.4}) gold_neg=({d:.4},{d:.4})\n", .{
        summary.mean_logit0_gold_positive,
        summary.mean_logit1_gold_positive,
        summary.mean_logit0_gold_negative,
        summary.mean_logit1_gold_negative,
    });
    fused_chunker_train.printBoundaryQualityDiagnostics("eval_quality", summary);

    // Dense retrieval evaluation.
    // Only run when chunk embeddings are non-zero; without --model-dir the eval
    // path intentionally falls back to zero-filled features for boundary-only checks.
    const emb_slice = chunk_embeddings_all.items;
    var emb_sum: f32 = 0;
    for (emb_slice) |v| emb_sum += @abs(v);

    var retrieval_result: ?RetrievalMetrics = null;
    if (emb_sum > 0.0) {
        const total_chunks_all = chunk_mask_all.items.len;
        const retrieval = try computeRetrievalMetrics(
            allocator,
            emb_slice,
            chunk_doc_ids_all.items,
            chunk_mask_all.items,
            total_chunks_all,
            hs,
        );
        retrieval_result = retrieval;
        print("\n=== Dense Retrieval Metrics ===\n", .{});
        print("Queries:     {d}\n", .{retrieval.num_queries});
        print("Recall@1:    {d:.4}\n", .{retrieval.recall_at_1});
        print("Recall@10:   {d:.4}\n", .{retrieval.recall_at_10});
        print("NDCG@10:     {d:.4}\n", .{retrieval.ndcg_at_10});
        print("MRR:         {d:.4}\n", .{retrieval.mrr});
    } else {
        print("\n(Dense retrieval metrics skipped: chunk embeddings are zero-filled)\n", .{});
    }

    if (opts.results_out) |results_path| {
        const dataset_name = opts.benchmark_dataset_name orelse std.fs.path.basename(opts.data_path);
        const backend_name = if (use_metal) "metal" else "native";
        try writeBenchmarkResultsJson(allocator, results_path, .{
            .dataset_name = dataset_name,
            .checkpoint_path = opts.checkpoint_path,
            .data_path = opts.data_path,
            .split = opts.split,
            .backend_name = backend_name,
            .samples = samples.len,
            .max_seq_len = opts.max_seq_len,
            .max_chunks = opts.max_chunks,
            .output_dimension = opts.output_dimension orelse opts.hidden_size,
            .internal_phase20_best_f1 = opts.internal_phase20_best_f1 orelse summary.best_boundary_f1,
            .summary = summary,
            .separator_metrics = chonky_separator_metrics,
            .retrieval = retrieval_result,
            .baselines = opts.retrieval_baselines,
        });
        print("Wrote benchmark results: {s}\n", .{results_path});
    }
}
