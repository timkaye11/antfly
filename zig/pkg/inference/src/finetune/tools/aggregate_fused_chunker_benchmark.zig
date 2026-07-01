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
const compat = @import("termite_io_compat");

const expected_results_schema = "fused_chunker_benchmark_results/v1";

const Options = struct {
    out_path: []const u8,
    input_paths: []const []const u8,
};

const ParsedDatasetResult = struct {
    name: []const u8,
    f1: f64,
    precision: ?f64 = null,
    recall: ?f64 = null,
    tp: ?u64 = null,
    fp: ?u64 = null,
    fn_count: ?u64 = null,
};

const ParsedBoundaryLane = struct {
    dataset_metric: ?[]const u8 = null,
    internal_phase20_best_f1: f64,
    fixed_threshold_f1: f64,
    best_threshold_f1: f64,
    best_threshold: ?f64 = null,
    dataset_results: []const ParsedDatasetResult,
};

const ParsedBaseline = struct {
    name: []const u8,
    overall_ndcg_at_10: ?f64 = null,
    ndcg_at_10: ?f64 = null,
};

const ParsedRetrievalLane = struct {
    output_dimension: ?u32 = null,
    overall_ndcg_at_10: ?f64 = null,
    ndcg_at_10: ?f64 = null,
    voyage_context_4_target_ndcg_at_10: ?f64 = null,
    distance_to_voyage_context_4_target: ?f64 = null,
    best_local_baseline_ndcg_at_10: ?f64 = null,
    relative_gain_over_best_local_baseline: ?f64 = null,
    recall_at_1: ?f64 = null,
    recall_at_10: ?f64 = null,
    mrr: ?f64 = null,
    queries: ?usize = null,
    baselines: ?[]const ParsedBaseline = null,
};

const ParsedBenchmarkFile = struct {
    schema_version: []const u8,
    boundary_f1: ?ParsedBoundaryLane = null,
    retrieval_ndcg: ?ParsedRetrievalLane = null,
};

const OutputDatasetResult = struct {
    name: []const u8,
    f1: f64,
    precision: ?f64 = null,
    recall: ?f64 = null,
    tp: ?u64 = null,
    fp: ?u64 = null,
    fn_count: ?u64 = null,
};

const OutputBoundaryLane = struct {
    dataset_metric: ?[]const u8 = null,
    internal_phase20_best_f1: f64,
    fixed_threshold_f1: f64,
    best_threshold_f1: f64,
    best_threshold: ?f64 = null,
    dataset_results: []const OutputDatasetResult,
};

const OutputBaseline = struct {
    name: []const u8,
    overall_ndcg_at_10: f64,
};

const OutputRetrievalLane = struct {
    output_dimension: ?u32 = null,
    overall_ndcg_at_10: f64,
    voyage_context_4_target_ndcg_at_10: ?f64 = null,
    distance_to_voyage_context_4_target: ?f64 = null,
    best_local_baseline_ndcg_at_10: ?f64 = null,
    relative_gain_over_best_local_baseline: ?f64 = null,
    recall_at_1: ?f64 = null,
    recall_at_10: ?f64 = null,
    mrr: ?f64 = null,
    queries: usize,
    baselines: []const OutputBaseline,
};

const OutputBenchmarkFile = struct {
    schema_version: []const u8 = expected_results_schema,
    aggregated: bool = true,
    source_results: []const []const u8,
    boundary_f1: OutputBoundaryLane,
    retrieval_ndcg: ?OutputRetrievalLane = null,
};

const BaselineAccumulator = struct {
    name: []const u8,
    weighted_ndcg_sum: f64 = 0,
    weight_sum: f64 = 0,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const opts = try parseOptions(allocator, &args) orelse return;
    defer allocator.free(opts.input_paths);

    var input_bytes = std.ArrayListUnmanaged([]const u8).empty;
    defer {
        for (input_bytes.items) |bytes| allocator.free(bytes);
        input_bytes.deinit(allocator);
    }
    for (opts.input_paths) |path| {
        const bytes = try compat.cwd().readFileAlloc(compat.io(), path, allocator, .limited(64 * 1024 * 1024));
        try input_bytes.append(allocator, bytes);
    }

    const output = try aggregateBenchmarkJson(allocator, input_bytes.items, opts.input_paths);
    defer allocator.free(output);
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = opts.out_path, .data = output });

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(compat.io(), &stdout_buf);
    try std.json.Stringify.value(.{
        .status = "written",
        .out = opts.out_path,
        .inputs = opts.input_paths.len,
    }, .{ .whitespace = .indent_2 }, &stdout.interface);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}

fn parseOptions(allocator: std.mem.Allocator, args: *std.process.Args.Iterator) !?Options {
    var out_path: ?[]const u8 = null;
    var input_paths = std.ArrayListUnmanaged([]const u8).empty;
    errdefer input_paths.deinit(allocator);

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--out")) {
            out_path = args.next() orelse return error.MissingOutPath;
        } else if (std.mem.eql(u8, arg, "--input")) {
            try input_paths.append(allocator, args.next() orelse return error.MissingInputPath);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            usage();
            input_paths.deinit(allocator);
            return null;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }

    if (out_path == null or input_paths.items.len == 0) {
        usage();
        return if (out_path == null) error.MissingOutPath else error.MissingInputPath;
    }

    return .{
        .out_path = out_path.?,
        .input_paths = try input_paths.toOwnedSlice(allocator),
    };
}

pub fn aggregateBenchmarkJson(
    allocator: std.mem.Allocator,
    input_jsons: []const []const u8,
    source_paths: []const []const u8,
) ![]u8 {
    if (input_jsons.len == 0 or input_jsons.len != source_paths.len) return error.InvalidAggregateInputs;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var datasets = std.ArrayListUnmanaged(OutputDatasetResult).empty;
    defer datasets.deinit(allocator);
    var baseline_accs = std.ArrayListUnmanaged(BaselineAccumulator).empty;
    defer baseline_accs.deinit(allocator);

    var boundary_lane_count: usize = 0;
    var boundary_dataset_metric: ?[]const u8 = null;
    var min_internal_phase20_best_f1 = std.math.inf(f64);
    var best_threshold_sum: f64 = 0;
    var best_threshold_count: usize = 0;
    var best_threshold_f1_sum: f64 = 0;
    var fixed_threshold_f1_sum: f64 = 0;
    var fixed_counts_tp: u64 = 0;
    var fixed_counts_fp: u64 = 0;
    var fixed_counts_fn: u64 = 0;
    var all_dataset_counts_present = true;

    var retrieval_count: usize = 0;
    var retrieval_weight_sum: f64 = 0;
    var retrieval_ndcg_sum: f64 = 0;
    var retrieval_target_sum: f64 = 0;
    var retrieval_target_weight: f64 = 0;
    var retrieval_queries: usize = 0;
    var recall_at_1_sum: f64 = 0;
    var recall_at_1_weight: f64 = 0;
    var recall_at_10_sum: f64 = 0;
    var recall_at_10_weight: f64 = 0;
    var mrr_sum: f64 = 0;
    var mrr_weight: f64 = 0;
    var output_dimension: ?u32 = null;
    var output_dimension_consistent = true;

    for (input_jsons) |json| {
        const parsed = try std.json.parseFromSliceLeaky(ParsedBenchmarkFile, arena_alloc, json, .{ .ignore_unknown_fields = true });
        if (!std.mem.eql(u8, parsed.schema_version, expected_results_schema)) return error.InvalidBenchmarkResults;

        if (parsed.boundary_f1) |boundary| {
            if (boundary.dataset_results.len == 0) return error.MissingBoundaryDatasetResults;
            if (boundary.dataset_metric) |metric| {
                if (boundary_dataset_metric) |existing| {
                    if (!std.mem.eql(u8, existing, metric)) return error.MixedBoundaryDatasetMetrics;
                } else {
                    boundary_dataset_metric = metric;
                }
            }
            if (!std.math.isFinite(boundary.internal_phase20_best_f1) or
                !std.math.isFinite(boundary.fixed_threshold_f1) or
                !std.math.isFinite(boundary.best_threshold_f1))
            {
                return error.InvalidBoundaryResults;
            }
            boundary_lane_count += 1;
            min_internal_phase20_best_f1 = @min(min_internal_phase20_best_f1, boundary.internal_phase20_best_f1);
            fixed_threshold_f1_sum += boundary.fixed_threshold_f1;
            best_threshold_f1_sum += boundary.best_threshold_f1;
            if (boundary.best_threshold) |threshold| {
                if (!std.math.isFinite(threshold)) return error.InvalidBoundaryResults;
                best_threshold_sum += threshold;
                best_threshold_count += 1;
            }
            for (boundary.dataset_results) |dataset| {
                if (!std.math.isFinite(dataset.f1)) return error.InvalidBoundaryDatasetResult;
                try appendDatasetResult(allocator, &datasets, dataset);
                if (dataset.tp) |tp| {
                    if (dataset.fp) |fp| {
                        if (dataset.fn_count) |fn_count| {
                            fixed_counts_tp += tp;
                            fixed_counts_fp += fp;
                            fixed_counts_fn += fn_count;
                        } else {
                            all_dataset_counts_present = false;
                        }
                    } else {
                        all_dataset_counts_present = false;
                    }
                } else {
                    all_dataset_counts_present = false;
                }
            }
        }

        if (parsed.retrieval_ndcg) |retrieval| {
            const ndcg = retrieval.overall_ndcg_at_10 orelse retrieval.ndcg_at_10 orelse return error.MissingRetrievalNdcg;
            if (!std.math.isFinite(ndcg)) return error.InvalidRetrievalResults;
            const queries = retrieval.queries orelse 1;
            const weight: f64 = @floatFromInt(@max(queries, 1));
            retrieval_count += 1;
            retrieval_weight_sum += weight;
            retrieval_ndcg_sum += ndcg * weight;
            retrieval_queries += queries;
            if (retrieval.voyage_context_4_target_ndcg_at_10) |target| {
                if (!std.math.isFinite(target)) return error.InvalidRetrievalResults;
                retrieval_target_sum += target * weight;
                retrieval_target_weight += weight;
            } else if (retrieval.output_dimension) |dim| {
                const target = voyageContext4TargetForDimension(dim);
                retrieval_target_sum += target * weight;
                retrieval_target_weight += weight;
            }
            if (retrieval.recall_at_1) |value| {
                if (!std.math.isFinite(value)) return error.InvalidRetrievalResults;
                recall_at_1_sum += value * weight;
                recall_at_1_weight += weight;
            }
            if (retrieval.recall_at_10) |value| {
                if (!std.math.isFinite(value)) return error.InvalidRetrievalResults;
                recall_at_10_sum += value * weight;
                recall_at_10_weight += weight;
            }
            if (retrieval.mrr) |value| {
                if (!std.math.isFinite(value)) return error.InvalidRetrievalResults;
                mrr_sum += value * weight;
                mrr_weight += weight;
            }
            if (retrieval.output_dimension) |dim| {
                if (output_dimension) |existing| {
                    if (existing != dim) output_dimension_consistent = false;
                } else {
                    output_dimension = dim;
                }
            } else {
                output_dimension_consistent = false;
            }
            if (retrieval.baselines) |baselines| {
                for (baselines) |baseline| {
                    const baseline_ndcg = baseline.overall_ndcg_at_10 orelse baseline.ndcg_at_10 orelse return error.InvalidRetrievalBaseline;
                    if (!std.math.isFinite(baseline_ndcg)) return error.InvalidRetrievalBaseline;
                    try addBaselineObservation(allocator, &baseline_accs, baseline.name, baseline_ndcg, weight);
                }
            }
        }
    }

    if (boundary_lane_count == 0 or datasets.items.len == 0) return error.MissingBoundaryResults;
    const boundary_count_f: f64 = @floatFromInt(boundary_lane_count);
    const can_recompute_fixed_threshold_from_dataset_counts = if (boundary_dataset_metric) |metric|
        std.mem.eql(u8, metric, "token_boundary_f1")
    else
        true;
    const fixed_threshold_f1 = if (can_recompute_fixed_threshold_from_dataset_counts and
        all_dataset_counts_present and
        fixed_counts_tp + fixed_counts_fp + fixed_counts_fn > 0)
        f1FromCounts(fixed_counts_tp, fixed_counts_fp, fixed_counts_fn)
    else
        fixed_threshold_f1_sum / boundary_count_f;
    const boundary = OutputBoundaryLane{
        .dataset_metric = boundary_dataset_metric,
        .internal_phase20_best_f1 = if (std.math.isFinite(min_internal_phase20_best_f1))
            min_internal_phase20_best_f1
        else
            best_threshold_f1_sum / boundary_count_f,
        .fixed_threshold_f1 = fixed_threshold_f1,
        .best_threshold_f1 = best_threshold_f1_sum / boundary_count_f,
        .best_threshold = if (best_threshold_count == 0)
            null
        else
            best_threshold_sum / @as(f64, @floatFromInt(best_threshold_count)),
        .dataset_results = datasets.items,
    };

    var baselines = std.ArrayListUnmanaged(OutputBaseline).empty;
    defer baselines.deinit(allocator);
    for (baseline_accs.items) |acc| {
        if (acc.weight_sum <= 0) return error.InvalidRetrievalBaseline;
        try baselines.append(allocator, .{
            .name = acc.name,
            .overall_ndcg_at_10 = acc.weighted_ndcg_sum / acc.weight_sum,
        });
    }

    var best_baseline_ndcg: ?f64 = null;
    for (baselines.items) |baseline| {
        best_baseline_ndcg = if (best_baseline_ndcg) |best|
            @max(best, baseline.overall_ndcg_at_10)
        else
            baseline.overall_ndcg_at_10;
    }

    const retrieval_lane: ?OutputRetrievalLane = if (retrieval_count == 0) null else lane: {
        const overall = retrieval_ndcg_sum / retrieval_weight_sum;
        const target = if (retrieval_target_weight > 0)
            retrieval_target_sum / retrieval_target_weight
        else if (output_dimension_consistent)
            if (output_dimension) |dim| voyageContext4TargetForDimension(dim) else null
        else
            null;
        break :lane .{
            .output_dimension = if (output_dimension_consistent) output_dimension else null,
            .overall_ndcg_at_10 = overall,
            .voyage_context_4_target_ndcg_at_10 = target,
            .distance_to_voyage_context_4_target = if (target) |value| value - overall else null,
            .best_local_baseline_ndcg_at_10 = best_baseline_ndcg,
            .relative_gain_over_best_local_baseline = if (best_baseline_ndcg) |best|
                if (best > 0.0) (overall - best) / best else null
            else
                null,
            .recall_at_1 = if (recall_at_1_weight > 0) recall_at_1_sum / recall_at_1_weight else null,
            .recall_at_10 = if (recall_at_10_weight > 0) recall_at_10_sum / recall_at_10_weight else null,
            .mrr = if (mrr_weight > 0) mrr_sum / mrr_weight else null,
            .queries = retrieval_queries,
            .baselines = baselines.items,
        };
    };

    const output = OutputBenchmarkFile{
        .source_results = source_paths,
        .boundary_f1 = boundary,
        .retrieval_ndcg = retrieval_lane,
    };
    var buffer: std.Io.Writer.Allocating = .init(allocator);
    errdefer buffer.deinit();
    try std.json.Stringify.value(output, .{ .whitespace = .indent_2 }, &buffer.writer);
    try buffer.writer.writeByte('\n');
    return buffer.toOwnedSlice();
}

fn appendDatasetResult(
    allocator: std.mem.Allocator,
    datasets: *std.ArrayListUnmanaged(OutputDatasetResult),
    dataset: ParsedDatasetResult,
) !void {
    for (datasets.items) |existing| {
        if (std.mem.eql(u8, existing.name, dataset.name)) return error.DuplicateBoundaryDatasetResult;
    }
    try datasets.append(allocator, .{
        .name = dataset.name,
        .f1 = dataset.f1,
        .precision = dataset.precision,
        .recall = dataset.recall,
        .tp = dataset.tp,
        .fp = dataset.fp,
        .fn_count = dataset.fn_count,
    });
}

fn addBaselineObservation(
    allocator: std.mem.Allocator,
    baselines: *std.ArrayListUnmanaged(BaselineAccumulator),
    name: []const u8,
    ndcg: f64,
    weight: f64,
) !void {
    for (baselines.items) |*baseline| {
        if (std.mem.eql(u8, baseline.name, name)) {
            baseline.weighted_ndcg_sum += ndcg * weight;
            baseline.weight_sum += weight;
            return;
        }
    }
    try baselines.append(allocator, .{
        .name = name,
        .weighted_ndcg_sum = ndcg * weight,
        .weight_sum = weight,
    });
}

fn f1FromCounts(tp: u64, fp: u64, fn_count: u64) f64 {
    const precision = if (tp + fp == 0) 0.0 else @as(f64, @floatFromInt(tp)) / @as(f64, @floatFromInt(tp + fp));
    const recall = if (tp + fn_count == 0) 0.0 else @as(f64, @floatFromInt(tp)) / @as(f64, @floatFromInt(tp + fn_count));
    return if (precision + recall == 0) 0.0 else 2.0 * precision * recall / (precision + recall);
}

fn voyageContext4TargetForDimension(output_dimension: u32) f64 {
    return switch (output_dimension) {
        256 => 0.8054,
        512 => 0.8266,
        1024 => 0.8381,
        else => 0.8440,
    };
}

fn usage() void {
    std.debug.print(
        \\usage: aggregate-fused-chunker-benchmark --out <path> --input <path> [--input <path> ...]
        \\
        \\Aggregates per-dataset eval-fused-chunker result JSON files into one
        \\fused_chunker_benchmark_results/v1 file suitable for
        \\verify-fused-chunker-benchmark.
        \\
    , .{});
}

test "aggregate benchmark results combines datasets and retrieval baselines" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{
        \\{
        \\  "schema_version":"fused_chunker_benchmark_results/v1",
        \\  "boundary_f1":{
        \\    "dataset_metric":"chonky_character_separator_f1",
        \\    "internal_phase20_best_f1":0.82,
        \\    "fixed_threshold_f1":0.80,
        \\    "best_threshold_f1":0.82,
        \\    "best_threshold":0.43,
        \\    "dataset_results":[{"name":"bookcorpus","f1":0.82,"tp":82,"fp":18,"fn_count":18}]
        \\  },
        \\  "retrieval_ndcg":{
        \\    "output_dimension":256,
        \\    "overall_ndcg_at_10":0.74,
        \\    "recall_at_1":0.5,
        \\    "recall_at_10":0.9,
        \\    "mrr":0.61,
        \\    "queries":10,
        \\    "baselines":[{"name":"fixed","overall_ndcg_at_10":0.68},{"name":"chonky","overall_ndcg_at_10":0.70}]
        \\  }
        \\}
        ,
        \\{
        \\  "schema_version":"fused_chunker_benchmark_results/v1",
        \\  "boundary_f1":{
        \\    "dataset_metric":"chonky_character_separator_f1",
        \\    "internal_phase20_best_f1":0.81,
        \\    "fixed_threshold_f1":0.72,
        \\    "best_threshold_f1":0.74,
        \\    "best_threshold":0.45,
        \\    "dataset_results":[{"name":"paul_graham","f1":0.72,"tp":72,"fp":28,"fn_count":28}]
        \\  },
        \\  "retrieval_ndcg":{
        \\    "output_dimension":256,
        \\    "overall_ndcg_at_10":0.80,
        \\    "recall_at_1":0.6,
        \\    "recall_at_10":1.0,
        \\    "mrr":0.71,
        \\    "queries":30,
        \\    "baselines":[{"name":"fixed","overall_ndcg_at_10":0.70},{"name":"chonky","overall_ndcg_at_10":0.72}]
        \\  }
        \\}
        ,
    };
    const paths = [_][]const u8{ "bookcorpus.json", "paul_graham.json" };
    const output = try aggregateBenchmarkJson(allocator, &inputs, &paths);
    defer allocator.free(output);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, output, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings(expected_results_schema, obj.get("schema_version").?.string);
    const boundary = obj.get("boundary_f1").?.object;
    try std.testing.expectEqualStrings("chonky_character_separator_f1", boundary.get("dataset_metric").?.string);
    try std.testing.expectApproxEqAbs(@as(f64, 0.81), boundary.get("internal_phase20_best_f1").?.float, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.76), boundary.get("fixed_threshold_f1").?.float, 1e-12);
    try std.testing.expectEqual(@as(usize, 2), boundary.get("dataset_results").?.array.items.len);
    const retrieval = obj.get("retrieval_ndcg").?.object;
    try std.testing.expectApproxEqAbs(@as(f64, 0.785), retrieval.get("overall_ndcg_at_10").?.float, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8054), retrieval.get("voyage_context_4_target_ndcg_at_10").?.float, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0204), retrieval.get("distance_to_voyage_context_4_target").?.float, 1e-12);
    const baselines = retrieval.get("baselines").?.array.items;
    try std.testing.expectEqual(@as(usize, 2), baselines.len);
}

test "aggregate benchmark results preserves source fixed f1 when counts are absent" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{
        \\{
        \\  "schema_version":"fused_chunker_benchmark_results/v1",
        \\  "boundary_f1":{
        \\    "internal_phase20_best_f1":0.80,
        \\    "fixed_threshold_f1":0.78,
        \\    "best_threshold_f1":0.80,
        \\    "dataset_results":[
        \\      {"name":"bookcorpus","f1":0.82},
        \\      {"name":"en_judgements","f1":0.32}
        \\    ]
        \\  }
        \\}
        ,
    };
    const paths = [_][]const u8{"combined.json"};
    const output = try aggregateBenchmarkJson(allocator, &inputs, &paths);
    defer allocator.free(output);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, output, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const boundary = parsed.value.object.get("boundary_f1").?.object;
    try std.testing.expectApproxEqAbs(@as(f64, 0.78), boundary.get("fixed_threshold_f1").?.float, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.80), boundary.get("best_threshold_f1").?.float, 1e-12);
}

test "aggregate benchmark results rejects duplicate dataset names" {
    const allocator = std.testing.allocator;
    const inputs = [_][]const u8{
        \\{"schema_version":"fused_chunker_benchmark_results/v1","boundary_f1":{"internal_phase20_best_f1":0.8,"fixed_threshold_f1":0.8,"best_threshold_f1":0.8,"dataset_results":[{"name":"bookcorpus","f1":0.8}]}}
        ,
        \\{"schema_version":"fused_chunker_benchmark_results/v1","boundary_f1":{"internal_phase20_best_f1":0.8,"fixed_threshold_f1":0.8,"best_threshold_f1":0.8,"dataset_results":[{"name":"bookcorpus","f1":0.81}]}}
        ,
    };
    const paths = [_][]const u8{ "a.json", "b.json" };
    try std.testing.expectError(error.DuplicateBoundaryDatasetResult, aggregateBenchmarkJson(allocator, &inputs, &paths));
}
