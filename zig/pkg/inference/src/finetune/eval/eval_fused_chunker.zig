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
//   --backend native|metal|auto  Compute backend (default: auto)

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
const fused_chunker_train = @import("../fused_chunker_train.zig");
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
const MetalWeightStore = if (build_options.enable_metal) gpu_hosted_store.WeightStore else void;

const print = std.debug.print;

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
    internal_phase20_best_f1: f64,
    fixed_threshold_f1: f64,
    best_threshold_f1: f64,
    best_threshold: f64,
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

pub fn renderBenchmarkResultsJson(allocator: std.mem.Allocator, input: BenchmarkResultInput) ![]u8 {
    const dataset_results = [_]BenchmarkDatasetResult{.{
        .name = input.dataset_name,
        .f1 = input.summary.boundary_f1,
        .precision = input.summary.boundary_precision,
        .recall = input.summary.boundary_recall,
        .tp = input.summary.boundary_tp,
        .fp = input.summary.boundary_fp,
        .fn_count = input.summary.boundary_fn,
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
            .internal_phase20_best_f1 = input.internal_phase20_best_f1,
            .fixed_threshold_f1 = input.summary.boundary_f1,
            .best_threshold_f1 = input.summary.best_boundary_f1,
            .best_threshold = input.summary.best_boundary_threshold,
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
    try std.testing.expectApproxEqAbs(@as(f64, 0.86), boundary.get("internal_phase20_best_f1").?.float, 1e-12);
    const datasets = boundary.get("dataset_results").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), datasets.len);
    try std.testing.expectEqualStrings("bookcorpus", datasets[0].object.get("name").?.string);
    const retrieval_obj = obj.get("retrieval_ndcg").?.object;
    try std.testing.expectApproxEqAbs(@as(f64, 0.74), retrieval_obj.get("overall_ndcg_at_10").?.float, 1e-12);
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
    backend: FusedBackend = .auto,
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
        \\  --backend native|metal|auto  Compute backend (default: auto; auto prefers Metal)
        \\
    , .{});
}

fn deinitMetalWeightStore(allocator: std.mem.Allocator, weight_store: *MetalWeightStore) void {
    if (comptime !build_options.enable_metal) return;
    metal_compute.deinitPrefetchQueue(weight_store);
    var it = weight_store.lazy_weights.iterator();
    while (it.next()) |entry| {
        allocator.free(entry.key_ptr.*);
        if (entry.value_ptr.host_loaded) |*loaded| loaded.deinit();
        if (entry.value_ptr.quantized_storage) |*storage| storage.deinit();
    }
    weight_store.lazy_weights.deinit(allocator);
}

fn deinitNativeWeightStore(allocator: std.mem.Allocator, weight_store: *native_compute.WeightStore) void {
    native_compute.deinitPrefetchQueue(weight_store);
    var it = weight_store.resident_weights.iterator();
    while (it.next()) |entry| {
        allocator.free(entry.key_ptr.*);
        entry.value_ptr.deinit();
    }
    weight_store.resident_weights.deinit(allocator);
    weight_store.lazy_weights.deinit(allocator);
}

fn stripEncoderPrefix(name: []const u8) []const u8 {
    const prefix = "encoder.";
    if (std.mem.startsWith(u8, name, prefix)) return name[prefix.len..];
    return name;
}

fn normalizeModernBertWeightName(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    const stripped = stripEncoderPrefix(name);
    if (std.mem.startsWith(u8, stripped, "layers.") or
        std.mem.startsWith(u8, stripped, "embeddings.") or
        std.mem.startsWith(u8, stripped, "final_norm."))
    {
        return std.fmt.allocPrint(allocator, "model.{s}", .{stripped});
    }
    return allocator.dupe(u8, stripped);
}

fn cloneLoadedWeight(allocator: std.mem.Allocator, loaded: LoadedWeight) !LoadedWeight {
    if (loaded.quantized or loaded.quantized_storage != null) return error.UnsupportedQuantizedEvalWeight;
    const owned_data = try allocator.dupe(u8, loaded.tensor.data);
    errdefer allocator.free(owned_data);
    const owned_shape = try allocator.dupe(i64, loaded.tensor.shape);
    errdefer allocator.free(owned_shape);
    return .{
        .tensor = .{
            .data = owned_data,
            .dtype = loaded.tensor.dtype,
            .shape = owned_shape,
            .name = "",
            .allocator = allocator,
            .owns_data = true,
            .owns_shape = true,
        },
        .quantized = false,
    };
}

fn modernBertQkvLayer(name: []const u8) ?[]const u8 {
    const prefix = "model.layers.";
    const suffix = ".attn.Wqkv.weight";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    if (!std.mem.endsWith(u8, name, suffix)) return null;
    return name[prefix.len .. name.len - suffix.len];
}

fn copyModernBertQkvPart(
    allocator: std.mem.Allocator,
    loaded: LoadedWeight,
    part: usize,
) !struct { data: []f32, hidden: usize } {
    if (loaded.tensor.dtype != .f32) return error.UnsupportedModernBertQkvTensor;
    if (loaded.tensor.shape.len != 2) return error.InvalidModernBertQkvTensor;
    if (loaded.tensor.shape[0] < 0 or loaded.tensor.shape[1] < 0) return error.InvalidModernBertQkvTensor;
    const rows: usize = @intCast(loaded.tensor.shape[0]);
    const hidden: usize = @intCast(loaded.tensor.shape[1]);
    if (rows != hidden * 3) return error.InvalidModernBertQkvTensor;

    const elems = hidden * hidden;
    const elem_start = part * elems;
    const byte_start = elem_start * @sizeOf(f32);
    const byte_end = byte_start + elems * @sizeOf(f32);
    if (byte_end > loaded.tensor.data.len) return error.InvalidModernBertQkvTensor;

    const data = try allocator.alloc(f32, elems);
    errdefer allocator.free(data);
    @memcpy(std.mem.sliceAsBytes(data), loaded.tensor.data[byte_start..byte_end]);
    return .{ .data = data, .hidden = hidden };
}

fn insertModernBertQkvSplitsIntoNativeStore(
    allocator: std.mem.Allocator,
    weight_store: *native_compute.WeightStore,
    normalized_name: []const u8,
    loaded: LoadedWeight,
) !usize {
    const layer = modernBertQkvLayer(normalized_name) orelse return 0;
    const projections = [_][]const u8{ "query_proj", "key_proj", "value_proj" };
    var inserted: usize = 0;
    for (projections, 0..) |projection, part| {
        const split = try copyModernBertQkvPart(allocator, loaded, part);
        defer allocator.free(split.data);
        const split_name = try std.fmt.allocPrint(allocator, "model.layers.{s}.attn.{s}.weight", .{ layer, projection });
        errdefer allocator.free(split_name);
        const shape = [_]i64{ @intCast(split.hidden), @intCast(split.hidden) };
        var tensor = try tensor_mod.Tensor.initFloat32(allocator, split_name, &shape, split.data);
        errdefer tensor.deinit();
        try weight_store.resident_weights.put(allocator, split_name, weight_source_mod.LoadedWeight{ .tensor = tensor });
        inserted += 1;
    }
    return inserted;
}

fn insertModernBertQkvSplitsIntoMetalStore(
    allocator: std.mem.Allocator,
    weight_store: *MetalWeightStore,
    normalized_name: []const u8,
    loaded: LoadedWeight,
) !usize {
    if (comptime !build_options.enable_metal) return error.MetalBackendUnavailable;
    const layer = modernBertQkvLayer(normalized_name) orelse return 0;
    const projections = [_][]const u8{ "query_proj", "key_proj", "value_proj" };
    var inserted: usize = 0;
    for (projections, 0..) |projection, part| {
        const split = try copyModernBertQkvPart(allocator, loaded, part);
        defer allocator.free(split.data);
        const split_name = try std.fmt.allocPrint(allocator, "model.layers.{s}.attn.{s}.weight", .{ layer, projection });
        errdefer allocator.free(split_name);
        const shape = [_]i64{ @intCast(split.hidden), @intCast(split.hidden) };
        var tensor = try tensor_mod.Tensor.initFloat32(allocator, split_name, &shape, split.data);
        errdefer tensor.deinit();
        try weight_store.lazy_weights.put(allocator, split_name, .{
            .tensor_ref = undefined,
            .host_loaded = .{ .tensor = tensor },
            .active_tier = .host,
            .loaded_bytes = tensor.data.len,
        });
        inserted += 1;
    }
    return inserted;
}

const FusedLoRALoadResult = struct {
    count: usize = 0,
    rank: u32 = 0,
    alpha: f32 = 0.0,
};

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

fn loadSafetensorsIntoNativeStore(
    allocator: std.mem.Allocator,
    weight_store: *native_compute.WeightStore,
    st_path: []const u8,
) !usize {
    var source = try SafetensorsSource.initAbsolute(allocator, st_path);
    errdefer source.weightSource().deinit();
    const ws = source.weightSource();
    const names = try ws.listNames(allocator);
    defer allocator.free(names);

    var loaded_count: usize = 0;
    for (names) |name| {
        var loaded = ws.getTensor(name) catch continue;
        defer loaded.deinit();
        var owned_loaded = try cloneLoadedWeight(allocator, loaded);
        errdefer owned_loaded.deinit();
        const owned_name = try normalizeModernBertWeightName(allocator, name);
        errdefer allocator.free(owned_name);
        try weight_store.resident_weights.put(allocator, owned_name, owned_loaded);
        loaded_count += 1;
        loaded_count += try insertModernBertQkvSplitsIntoNativeStore(allocator, weight_store, owned_name, loaded);
    }
    source.weightSource().deinit();
    return loaded_count;
}

fn loadSafetensorsIntoMetalStore(
    allocator: std.mem.Allocator,
    weight_store: *MetalWeightStore,
    st_path: []const u8,
) !usize {
    if (comptime !build_options.enable_metal) return error.MetalBackendUnavailable;
    var source = try SafetensorsSource.initAbsolute(allocator, st_path);
    errdefer source.weightSource().deinit();
    const ws = source.weightSource();
    const names = try ws.listNames(allocator);
    defer allocator.free(names);

    var loaded_count: usize = 0;
    for (names) |name| {
        var loaded = ws.getTensor(name) catch continue;
        defer loaded.deinit();
        var owned_loaded = try cloneLoadedWeight(allocator, loaded);
        errdefer owned_loaded.deinit();
        const owned_name = try normalizeModernBertWeightName(allocator, name);
        errdefer allocator.free(owned_name);
        try weight_store.lazy_weights.put(allocator, owned_name, .{
            .tensor_ref = undefined,
            .host_loaded = owned_loaded,
            .active_tier = .host,
            .loaded_bytes = owned_loaded.tensor.data.len,
        });
        loaded_count += 1;
        loaded_count += try insertModernBertQkvSplitsIntoMetalStore(allocator, weight_store, owned_name, loaded);
    }
    source.weightSource().deinit();
    return loaded_count;
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
        .backend = backend,
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
    if (encoder_loaded) {
        const lora_load = loadFusedLoRAFromCheckpoint(
            allocator,
            opts.checkpoint_path,
            use_metal,
            &weight_store,
            &metal_weight_store,
        ) catch |err| blk: {
            print("warning: could not load LoRA tensors from checkpoint '{s}': {}\n", .{ opts.checkpoint_path, err });
            break :blk FusedLoRALoadResult{};
        };
        if (lora_load.count > 0) {
            eval_lora_rank = lora_load.rank;
            eval_lora_alpha = lora_load.alpha;
            print("loaded {d} LoRA tensors from checkpoint (rank={d} alpha={d:.3})\n", .{
                lora_load.count,
                lora_load.rank,
                lora_load.alpha,
            });
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

        try trainer.evaluateBatchInto(allocator, &boundary_acc, features, labels, mask, total_tokens);

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

        sample_idx = end;
    }

    const summary = boundary_acc.finish();

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
    print("Valid Tok:   {d}\n", .{summary.valid_tokens});
    print("Gold Pos:    {d} ({d:.6})\n", .{ summary.gold_positives, summary.gold_positive_rate });
    print("Pred Pos:    {d} ({d:.6})\n", .{ summary.predicted_positives, summary.predicted_positive_rate });
    print("Best Pred:   {d} ({d:.6})\n", .{ summary.best_predicted_positives, summary.best_predicted_positive_rate });
    print("Mean P(+):   gold_pos={d:.4} gold_neg={d:.4}\n", .{
        summary.mean_positive_probability_gold_positive,
        summary.mean_positive_probability_gold_negative,
    });
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
            .retrieval = retrieval_result,
            .baselines = opts.retrieval_baselines,
        });
        print("Wrote benchmark results: {s}\n", .{results_path});
    }
}
