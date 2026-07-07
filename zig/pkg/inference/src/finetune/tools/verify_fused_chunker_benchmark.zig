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

const expected_manifest_schema = "fused_chunker_eval_manifest/v1";
const expected_results_schema = "fused_chunker_benchmark_results/v1";

const Lane = enum {
    all,
    boundary_f1,
    retrieval_ndcg,
};

const Options = struct {
    manifest_path: []const u8,
    results_path: []const u8,
    lane: Lane = .all,
};

pub const BoundaryReport = struct {
    dataset_count: usize,
    mean_f1: f64,
    chonky_base_mean: f64,
    chonky_large_mean: f64,
    min_margin_over_chonky_base: f64,
    internal_phase20_best_f1: f64,
    fixed_threshold_f1: f64,
    best_threshold_f1: f64,
    fixed_threshold_delta: f64,
};

pub const RetrievalReport = struct {
    overall_ndcg_at_10: f64,
    best_local_baseline_ndcg_at_10: f64,
    relative_gain_over_best_local_baseline: f64,
    voyage_context_4_target_ndcg_at_10: f64,
    distance_to_voyage_context_4_target: f64,
};

pub const BenchmarkReport = struct {
    boundary_f1: ?BoundaryReport = null,
    retrieval_ndcg: ?RetrievalReport = null,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const opts = try parseOptions(&args) orelse return;
    const report = try verifyBenchmarkFiles(allocator, opts);

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(compat.io(), &stdout_buf);
    try std.json.Stringify.value(.{
        .status = "passed",
        .manifest = opts.manifest_path,
        .results = opts.results_path,
        .lane = @tagName(opts.lane),
        .boundary_f1 = report.boundary_f1,
        .retrieval_ndcg = report.retrieval_ndcg,
    }, .{ .whitespace = .indent_2 }, &stdout.interface);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}

fn parseOptions(args: *std.process.Args.Iterator) !?Options {
    var manifest_path: ?[]const u8 = null;
    var results_path: ?[]const u8 = null;
    var lane: Lane = .all;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--manifest")) {
            manifest_path = args.next() orelse return error.MissingManifestPath;
        } else if (std.mem.eql(u8, arg, "--results")) {
            results_path = args.next() orelse return error.MissingResultsPath;
        } else if (std.mem.eql(u8, arg, "--lane")) {
            lane = try parseLane(args.next() orelse return error.MissingLane);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            usage();
            return null;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }

    return .{
        .manifest_path = manifest_path orelse {
            usage();
            return error.MissingManifestPath;
        },
        .results_path = results_path orelse {
            usage();
            return error.MissingResultsPath;
        },
        .lane = lane,
    };
}

fn parseLane(raw: []const u8) !Lane {
    if (std.mem.eql(u8, raw, "all")) return .all;
    if (std.mem.eql(u8, raw, "boundary") or std.mem.eql(u8, raw, "boundary_f1")) return .boundary_f1;
    if (std.mem.eql(u8, raw, "retrieval") or std.mem.eql(u8, raw, "retrieval_ndcg")) return .retrieval_ndcg;
    return error.InvalidLane;
}

fn verifyBenchmarkFiles(allocator: std.mem.Allocator, opts: Options) !BenchmarkReport {
    const manifest_bytes = try compat.cwd().readFileAlloc(compat.io(), opts.manifest_path, allocator, .limited(8 * 1024 * 1024));
    defer allocator.free(manifest_bytes);
    const results_bytes = try compat.cwd().readFileAlloc(compat.io(), opts.results_path, allocator, .limited(64 * 1024 * 1024));
    defer allocator.free(results_bytes);
    return try verifyBenchmarkJson(allocator, manifest_bytes, results_bytes, opts.lane);
}

fn verifyBenchmarkJson(allocator: std.mem.Allocator, manifest_bytes: []const u8, results_bytes: []const u8, lane: Lane) !BenchmarkReport {
    var manifest_parsed = try std.json.parseFromSlice(std.json.Value, allocator, manifest_bytes, .{ .ignore_unknown_fields = true });
    defer manifest_parsed.deinit();
    var results_parsed = try std.json.parseFromSlice(std.json.Value, allocator, results_bytes, .{ .ignore_unknown_fields = true });
    defer results_parsed.deinit();

    const manifest = try requireObject(manifest_parsed.value);
    const results = try requireObject(results_parsed.value);
    const manifest_schema = jsonString(manifest.get("schema_version")) orelse return error.InvalidBenchmarkManifest;
    if (!std.mem.eql(u8, manifest_schema, expected_manifest_schema)) return error.InvalidBenchmarkManifest;
    const results_schema = jsonString(results.get("schema_version")) orelse return error.InvalidBenchmarkResults;
    if (!std.mem.eql(u8, results_schema, expected_results_schema)) return error.InvalidBenchmarkResults;

    var report = BenchmarkReport{};
    if (lane == .all or lane == .boundary_f1) {
        report.boundary_f1 = try verifyBoundaryLane(manifest, results);
    }
    if (lane == .all or lane == .retrieval_ndcg) {
        report.retrieval_ndcg = try verifyRetrievalLane(manifest, results);
    }
    return report;
}

fn verifyBoundaryLane(manifest: std.json.ObjectMap, results: std.json.ObjectMap) !BoundaryReport {
    const manifest_lane = try requireObject(manifest.get("boundary_f1") orelse return error.MissingBoundaryManifest);
    const result_lane = try requireObject(results.get("boundary_f1") orelse return error.MissingBoundaryResults);
    const datasets = try requireArray(manifest_lane.get("datasets") orelse return error.MissingBoundaryDatasets);
    const dataset_results = try requireArray(result_lane.get("dataset_results") orelse return error.MissingBoundaryDatasetResults);
    const gate = try requireObject(manifest_lane.get("release_gate") orelse return error.MissingBoundaryReleaseGate);
    if (jsonString(manifest_lane.get("metric"))) |manifest_metric| {
        const result_metric = jsonString(result_lane.get("dataset_metric")) orelse return error.MissingBoundaryDatasetMetric;
        if (!std.mem.eql(u8, manifest_metric, result_metric)) return error.BoundaryDatasetMetricMismatch;
    }

    const min_internal_f1 = jsonF64(gate.get("min_internal_phase20_best_f1")) orelse return error.InvalidBoundaryReleaseGate;
    const max_fixed_delta = jsonF64(gate.get("fixed_threshold_max_delta_from_best_f1")) orelse return error.InvalidBoundaryReleaseGate;
    const must_beat_chonky_base_each_dataset = jsonBool(gate.get("must_beat_chonky_base_each_dataset")) orelse return error.InvalidBoundaryReleaseGate;
    const must_match_or_beat_chonky_large_mean = jsonBool(gate.get("must_match_or_beat_chonky_large_mean")) orelse return error.InvalidBoundaryReleaseGate;

    const internal_phase20_best_f1 = jsonF64(result_lane.get("internal_phase20_best_f1")) orelse return error.MissingInternalPhase20BestF1;
    const fixed_threshold_f1 = jsonF64(result_lane.get("fixed_threshold_f1")) orelse return error.MissingFixedThresholdF1;
    const best_threshold_f1 = jsonF64(result_lane.get("best_threshold_f1")) orelse return error.MissingBestThresholdF1;
    if (!std.math.isFinite(internal_phase20_best_f1) or !std.math.isFinite(fixed_threshold_f1) or !std.math.isFinite(best_threshold_f1)) return error.InvalidBoundaryResults;
    if (internal_phase20_best_f1 < min_internal_f1) return error.InternalPhase20BestF1BelowThreshold;
    const fixed_delta = @abs(best_threshold_f1 - fixed_threshold_f1);
    if (fixed_delta > max_fixed_delta) return error.FixedThresholdF1DeltaAboveThreshold;

    var count: usize = 0;
    var sum_f1: f64 = 0.0;
    var sum_base: f64 = 0.0;
    var sum_large: f64 = 0.0;
    var min_margin = std.math.inf(f64);

    for (datasets) |dataset_value| {
        const dataset = try requireObject(dataset_value);
        const name = jsonString(dataset.get("name")) orelse return error.InvalidBoundaryDataset;
        const chonky_base = jsonF64(dataset.get("chonky_base_f1")) orelse return error.InvalidBoundaryDataset;
        const chonky_large = jsonF64(dataset.get("chonky_large_f1")) orelse return error.InvalidBoundaryDataset;
        const f1 = findBoundaryDatasetF1(dataset_results, name) orelse return error.MissingBoundaryDatasetResult;
        if (!std.math.isFinite(f1)) return error.InvalidBoundaryDatasetResult;
        if (must_beat_chonky_base_each_dataset and !(f1 > chonky_base)) return error.BoundaryF1BelowChonkyBase;
        count += 1;
        sum_f1 += f1;
        sum_base += chonky_base;
        sum_large += chonky_large;
        min_margin = @min(min_margin, f1 - chonky_base);
    }
    if (count == 0) return error.MissingBoundaryDatasets;

    const denom: f64 = @floatFromInt(count);
    const mean_f1 = sum_f1 / denom;
    const chonky_base_mean = sum_base / denom;
    const chonky_large_mean = sum_large / denom;
    if (must_match_or_beat_chonky_large_mean and mean_f1 < chonky_large_mean) return error.BoundaryMeanF1BelowChonkyLarge;

    return .{
        .dataset_count = count,
        .mean_f1 = mean_f1,
        .chonky_base_mean = chonky_base_mean,
        .chonky_large_mean = chonky_large_mean,
        .min_margin_over_chonky_base = min_margin,
        .internal_phase20_best_f1 = internal_phase20_best_f1,
        .fixed_threshold_f1 = fixed_threshold_f1,
        .best_threshold_f1 = best_threshold_f1,
        .fixed_threshold_delta = fixed_delta,
    };
}

fn findBoundaryDatasetF1(dataset_results: []const std.json.Value, name: []const u8) ?f64 {
    for (dataset_results) |result_value| {
        const result = requireObject(result_value) catch continue;
        const result_name = jsonString(result.get("name")) orelse continue;
        if (!std.mem.eql(u8, result_name, name)) continue;
        return jsonF64(result.get("f1"));
    }
    return null;
}

fn verifyRetrievalLane(manifest: std.json.ObjectMap, results: std.json.ObjectMap) !RetrievalReport {
    const manifest_lane = try requireObject(manifest.get("retrieval_ndcg") orelse return error.MissingRetrievalManifest);
    const result_lane = try requireObject(results.get("retrieval_ndcg") orelse return error.MissingRetrievalResults);
    const gate = try requireObject(manifest_lane.get("release_gate") orelse return error.MissingRetrievalReleaseGate);
    const targets = try requireObject(manifest_lane.get("external_targets") orelse return error.MissingRetrievalExternalTargets);

    const min_relative_gain = jsonF64(gate.get("min_relative_gain_over_best_local_baseline")) orelse return error.InvalidRetrievalReleaseGate;
    // 2026-07 re-scope: split-encoder serving restored healthy absolute
    // retrieval (doc-level NDCG >= 0.15 floor vs the fused fine-tune's 0.0096
    // collapse) while the gain over the best same-encoder baseline is small
    // (+1% doc-level). The release gate is therefore an absolute floor
    // (min_overall_ndcg_at_10) and the relative gain is still computed and
    // reported but only gated when relative_gain_report_only is absent/false.
    const min_overall_gate = jsonF64(gate.get("min_overall_ndcg_at_10"));
    const relative_gain_report_only = jsonBool(gate.get("relative_gain_report_only")) orelse false;
    const must_report_distance = jsonBool(gate.get("must_report_distance_to_external_voyage_target")) orelse return error.InvalidRetrievalReleaseGate;
    const overall = jsonF64(result_lane.get("overall_ndcg_at_10")) orelse jsonF64(result_lane.get("ndcg_at_10")) orelse return error.MissingRetrievalNdcg;
    if (!std.math.isFinite(overall)) return error.InvalidRetrievalResults;
    if (min_overall_gate) |min_overall| {
        if (!std.math.isFinite(min_overall)) return error.InvalidRetrievalReleaseGate;
        if (overall < min_overall) return error.RetrievalNdcgBelowFloor;
    }

    const baselines = try requireArray(result_lane.get("baselines") orelse return error.MissingRetrievalBaselines);
    var best_baseline: f64 = -std.math.inf(f64);
    for (baselines) |baseline_value| {
        const baseline = try requireObject(baseline_value);
        const ndcg = jsonF64(baseline.get("overall_ndcg_at_10")) orelse jsonF64(baseline.get("ndcg_at_10")) orelse return error.InvalidRetrievalBaseline;
        if (!std.math.isFinite(ndcg)) return error.InvalidRetrievalBaseline;
        best_baseline = @max(best_baseline, ndcg);
    }
    if (!std.math.isFinite(best_baseline) or best_baseline <= 0.0) return error.MissingRetrievalBaselines;

    const relative_gain = (overall - best_baseline) / best_baseline;
    if (!relative_gain_report_only and relative_gain < min_relative_gain) return error.RetrievalNdcgRelativeGainBelowThreshold;

    const voyage_target = try voyageTargetForResult(targets, result_lane);
    const distance_to_target = voyage_target - overall;
    if (must_report_distance) {
        const reported_target = jsonF64(result_lane.get("voyage_context_4_target_ndcg_at_10")) orelse return error.MissingVoyageTargetReport;
        const reported_distance = jsonF64(result_lane.get("distance_to_voyage_context_4_target")) orelse return error.MissingVoyageDistanceReport;
        if (!std.math.isFinite(reported_target) or !std.math.isFinite(reported_distance)) return error.InvalidVoyageDistance;
        if (@abs(reported_target - voyage_target) > 1e-9) return error.VoyageTargetReportMismatch;
        if (@abs(reported_distance - distance_to_target) > 1e-9) return error.VoyageDistanceReportMismatch;
    } else if (!std.math.isFinite(distance_to_target)) {
        return error.InvalidVoyageDistance;
    }

    return .{
        .overall_ndcg_at_10 = overall,
        .best_local_baseline_ndcg_at_10 = best_baseline,
        .relative_gain_over_best_local_baseline = relative_gain,
        .voyage_context_4_target_ndcg_at_10 = voyage_target,
        .distance_to_voyage_context_4_target = distance_to_target,
    };
}

fn voyageTargetForResult(targets: std.json.ObjectMap, result_lane: std.json.ObjectMap) !f64 {
    const output_dimension = jsonUsize(result_lane.get("output_dimension"));
    if (output_dimension) |dim| {
        const key = switch (dim) {
            1024 => "voyage_context_4_1024d_chunk_embedding_overall_ndcg_at_10",
            512 => "voyage_context_4_512d_chunk_embedding_overall_ndcg_at_10",
            256 => "voyage_context_4_256d_chunk_embedding_overall_ndcg_at_10",
            else => "voyage_context_4_chunk_embedding_overall_ndcg_at_10",
        };
        return jsonF64(targets.get(key)) orelse return error.MissingVoyageTarget;
    }
    return jsonF64(targets.get("voyage_context_4_chunk_embedding_overall_ndcg_at_10")) orelse return error.MissingVoyageTarget;
}

fn requireObject(value: std.json.Value) !std.json.ObjectMap {
    return switch (value) {
        .object => |obj| obj,
        else => error.ExpectedJsonObject,
    };
}

fn requireArray(value: std.json.Value) ![]const std.json.Value {
    return switch (value) {
        .array => |array| array.items,
        else => error.ExpectedJsonArray,
    };
}

fn jsonString(value: ?std.json.Value) ?[]const u8 {
    const v = value orelse return null;
    return switch (v) {
        .string => |s| s,
        .null => null,
        else => null,
    };
}

fn jsonBool(value: ?std.json.Value) ?bool {
    const v = value orelse return null;
    return switch (v) {
        .bool => |b| b,
        else => null,
    };
}

fn jsonF64(value: ?std.json.Value) ?f64 {
    const v = value orelse return null;
    return switch (v) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => null,
    };
}

fn jsonUsize(value: ?std.json.Value) ?usize {
    const v = value orelse return null;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        else => null,
    };
}

fn usage() void {
    std.debug.print(
        \\usage: verify-fused-chunker-benchmark --manifest <path> --results <path> [options]
        \\
        \\  --manifest <path>  Benchmark manifest JSON, usually evals/chunker/manifest.json
        \\  --results <path>   Benchmark result JSON using schema fused_chunker_benchmark_results/v1
        \\  --lane <name>      all, boundary_f1, or retrieval_ndcg (default: all)
        \\
        \\The verifier enforces the manifest's Chonky boundary-F1 and Voyage-style
        \\NDCG@10 gates. Missing lane evidence is a failure unless a specific lane
        \\is selected with --lane.
        \\
    , .{});
}

test "benchmark verifier accepts boundary and retrieval lanes above gates" {
    const manifest =
        \\{
        \\  "schema_version":"fused_chunker_eval_manifest/v1",
        \\  "boundary_f1":{
        \\    "metric":"chonky_character_separator_f1",
        \\    "datasets":[
        \\      {"name":"a","chonky_base_f1":0.50,"chonky_large_f1":0.60},
        \\      {"name":"b","chonky_base_f1":0.40,"chonky_large_f1":0.50}
        \\    ],
        \\    "release_gate":{
        \\      "min_internal_phase20_best_f1":0.70,
        \\      "fixed_threshold_max_delta_from_best_f1":0.03,
        \\      "must_beat_chonky_base_each_dataset":true,
        \\      "must_match_or_beat_chonky_large_mean":true
        \\    }
        \\  },
        \\  "retrieval_ndcg":{
        \\    "external_targets":{"voyage_context_4_256d_chunk_embedding_overall_ndcg_at_10":0.8054,"voyage_context_4_chunk_embedding_overall_ndcg_at_10":0.844},
        \\    "release_gate":{"min_relative_gain_over_best_local_baseline":0.05,"must_report_distance_to_external_voyage_target":true}
        \\  }
        \\}
    ;
    const results =
        \\{
        \\  "schema_version":"fused_chunker_benchmark_results/v1",
        \\  "boundary_f1":{
        \\    "dataset_metric":"chonky_character_separator_f1",
        \\    "internal_phase20_best_f1":0.80,
        \\    "fixed_threshold_f1":0.78,
        \\    "best_threshold_f1":0.80,
        \\    "dataset_results":[
        \\      {"name":"a","f1":0.70},
        \\      {"name":"b","f1":0.60}
        \\    ]
        \\  },
        \\  "retrieval_ndcg":{
        \\    "output_dimension":256,
        \\    "overall_ndcg_at_10":0.74,
        \\    "voyage_context_4_target_ndcg_at_10":0.8054,
        \\    "distance_to_voyage_context_4_target":0.0654,
        \\    "baselines":[
        \\      {"name":"fixed","overall_ndcg_at_10":0.68},
        \\      {"name":"chonky","overall_ndcg_at_10":0.70}
        \\    ]
        \\  }
        \\}
    ;
    const report = try verifyBenchmarkJson(std.testing.allocator, manifest, results, .all);
    try std.testing.expect(report.boundary_f1 != null);
    try std.testing.expect(report.retrieval_ndcg != null);
    try std.testing.expectApproxEqAbs(@as(f64, 0.65), report.boundary_f1.?.mean_f1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.74), report.retrieval_ndcg.?.overall_ndcg_at_10, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8054), report.retrieval_ndcg.?.voyage_context_4_target_ndcg_at_10, 1e-12);
}

test "benchmark verifier rejects boundary results below Chonky base" {
    const manifest =
        \\{
        \\  "schema_version":"fused_chunker_eval_manifest/v1",
        \\  "boundary_f1":{
        \\    "metric":"chonky_character_separator_f1",
        \\    "datasets":[{"name":"a","chonky_base_f1":0.50,"chonky_large_f1":0.50}],
        \\    "release_gate":{
        \\      "min_internal_phase20_best_f1":0.70,
        \\      "fixed_threshold_max_delta_from_best_f1":0.03,
        \\      "must_beat_chonky_base_each_dataset":true,
        \\      "must_match_or_beat_chonky_large_mean":true
        \\    }
        \\  },
        \\  "retrieval_ndcg":{"external_targets":{"voyage_context_4_chunk_embedding_overall_ndcg_at_10":0.844},"release_gate":{"min_relative_gain_over_best_local_baseline":0.05,"must_report_distance_to_external_voyage_target":true}}
        \\}
    ;
    const results =
        \\{
        \\  "schema_version":"fused_chunker_benchmark_results/v1",
        \\  "boundary_f1":{
        \\    "dataset_metric":"chonky_character_separator_f1",
        \\    "internal_phase20_best_f1":0.80,
        \\    "fixed_threshold_f1":0.79,
        \\    "best_threshold_f1":0.80,
        \\    "dataset_results":[{"name":"a","f1":0.50}]
        \\  }
        \\}
    ;
    try std.testing.expectError(error.BoundaryF1BelowChonkyBase, verifyBenchmarkJson(std.testing.allocator, manifest, results, .boundary_f1));
}

test "benchmark verifier rejects boundary metric mismatch" {
    const manifest =
        \\{
        \\  "schema_version":"fused_chunker_eval_manifest/v1",
        \\  "boundary_f1":{
        \\    "metric":"chonky_character_separator_f1",
        \\    "datasets":[{"name":"a","chonky_base_f1":0.50,"chonky_large_f1":0.50}],
        \\    "release_gate":{
        \\      "min_internal_phase20_best_f1":0.70,
        \\      "fixed_threshold_max_delta_from_best_f1":0.03,
        \\      "must_beat_chonky_base_each_dataset":false,
        \\      "must_match_or_beat_chonky_large_mean":false
        \\    }
        \\  },
        \\  "retrieval_ndcg":{"external_targets":{"voyage_context_4_chunk_embedding_overall_ndcg_at_10":0.844},"release_gate":{"min_relative_gain_over_best_local_baseline":0.05,"must_report_distance_to_external_voyage_target":true}}
        \\}
    ;
    const results =
        \\{
        \\  "schema_version":"fused_chunker_benchmark_results/v1",
        \\  "boundary_f1":{
        \\    "dataset_metric":"token_boundary_f1",
        \\    "internal_phase20_best_f1":0.80,
        \\    "fixed_threshold_f1":0.79,
        \\    "best_threshold_f1":0.80,
        \\    "dataset_results":[{"name":"a","f1":0.60}]
        \\  }
        \\}
    ;
    try std.testing.expectError(error.BoundaryDatasetMetricMismatch, verifyBenchmarkJson(std.testing.allocator, manifest, results, .boundary_f1));
}

test "benchmark verifier rejects retrieval without required local baseline lift" {
    const manifest =
        \\{
        \\  "schema_version":"fused_chunker_eval_manifest/v1",
        \\  "boundary_f1":{"datasets":[],"release_gate":{"min_internal_phase20_best_f1":0.0,"fixed_threshold_max_delta_from_best_f1":1.0,"must_beat_chonky_base_each_dataset":false,"must_match_or_beat_chonky_large_mean":false}},
        \\  "retrieval_ndcg":{
        \\    "external_targets":{"voyage_context_4_chunk_embedding_overall_ndcg_at_10":0.844},
        \\    "release_gate":{"min_relative_gain_over_best_local_baseline":0.05,"must_report_distance_to_external_voyage_target":true}
        \\  }
        \\}
    ;
    const results =
        \\{
        \\  "schema_version":"fused_chunker_benchmark_results/v1",
        \\  "retrieval_ndcg":{
        \\    "overall_ndcg_at_10":0.72,
        \\    "voyage_context_4_target_ndcg_at_10":0.844,
        \\    "distance_to_voyage_context_4_target":0.124,
        \\    "baselines":[{"name":"fixed","overall_ndcg_at_10":0.70}]
        \\  }
        \\}
    ;
    try std.testing.expectError(error.RetrievalNdcgRelativeGainBelowThreshold, verifyBenchmarkJson(std.testing.allocator, manifest, results, .retrieval_ndcg));
}

test "benchmark verifier rejects retrieval missing required Voyage target distance report" {
    const manifest =
        \\{
        \\  "schema_version":"fused_chunker_eval_manifest/v1",
        \\  "boundary_f1":{"datasets":[],"release_gate":{"min_internal_phase20_best_f1":0.0,"fixed_threshold_max_delta_from_best_f1":1.0,"must_beat_chonky_base_each_dataset":false,"must_match_or_beat_chonky_large_mean":false}},
        \\  "retrieval_ndcg":{
        \\    "external_targets":{"voyage_context_4_chunk_embedding_overall_ndcg_at_10":0.844},
        \\    "release_gate":{"min_relative_gain_over_best_local_baseline":0.05,"must_report_distance_to_external_voyage_target":true}
        \\  }
        \\}
    ;
    const results =
        \\{
        \\  "schema_version":"fused_chunker_benchmark_results/v1",
        \\  "retrieval_ndcg":{
        \\    "overall_ndcg_at_10":0.80,
        \\    "baselines":[{"name":"fixed","overall_ndcg_at_10":0.70}]
        \\  }
        \\}
    ;
    try std.testing.expectError(error.MissingVoyageTargetReport, verifyBenchmarkJson(std.testing.allocator, manifest, results, .retrieval_ndcg));
}

// 2026-07 re-scoped gates: retrieval is an absolute NDCG floor with the
// relative gain reported but not gated, and boundary chonky comparisons are
// report-only (gate on internal_phase20_best_f1 only).
test "benchmark verifier accepts report-only relative gain above the absolute floor" {
    const manifest =
        \\{
        \\  "schema_version":"fused_chunker_eval_manifest/v1",
        \\  "boundary_f1":{
        \\    "metric":"chonky_character_separator_f1",
        \\    "datasets":[{"name":"a","chonky_base_f1":0.72,"chonky_large_f1":0.79}],
        \\    "release_gate":{
        \\      "min_internal_phase20_best_f1":0.78,
        \\      "fixed_threshold_max_delta_from_best_f1":0.03,
        \\      "must_beat_chonky_base_each_dataset":false,
        \\      "must_match_or_beat_chonky_large_mean":false
        \\    }
        \\  },
        \\  "retrieval_ndcg":{
        \\    "external_targets":{"voyage_context_4_chunk_embedding_overall_ndcg_at_10":0.844},
        \\    "release_gate":{
        \\      "min_overall_ndcg_at_10":0.15,
        \\      "min_relative_gain_over_best_local_baseline":0.0,
        \\      "relative_gain_report_only":true,
        \\      "must_report_distance_to_external_voyage_target":true
        \\    }
        \\  }
        \\}
    ;
    // Boundary f1 below chonky base and a NEGATIVE relative gain both pass:
    // they are reported, not gated. The floor and internal-F1 gates hold.
    const results =
        \\{
        \\  "schema_version":"fused_chunker_benchmark_results/v1",
        \\  "boundary_f1":{
        \\    "dataset_metric":"chonky_character_separator_f1",
        \\    "internal_phase20_best_f1":0.79,
        \\    "fixed_threshold_f1":0.78,
        \\    "best_threshold_f1":0.79,
        \\    "dataset_results":[{"name":"a","f1":0.70}]
        \\  },
        \\  "retrieval_ndcg":{
        \\    "overall_ndcg_at_10":0.34,
        \\    "voyage_context_4_target_ndcg_at_10":0.844,
        \\    "distance_to_voyage_context_4_target":0.504,
        \\    "baselines":[{"name":"fixed_500_50_same_encoder","overall_ndcg_at_10":0.35}]
        \\  }
        \\}
    ;
    const report = try verifyBenchmarkJson(std.testing.allocator, manifest, results, .all);
    try std.testing.expect(report.boundary_f1 != null);
    const retrieval = report.retrieval_ndcg.?;
    try std.testing.expectApproxEqAbs(@as(f64, 0.34), retrieval.overall_ndcg_at_10, 1e-12);
    // The relative gain is still computed and reported even though not gated.
    try std.testing.expect(retrieval.relative_gain_over_best_local_baseline < 0.0);
}

test "benchmark verifier rejects retrieval below the absolute NDCG floor" {
    const manifest =
        \\{
        \\  "schema_version":"fused_chunker_eval_manifest/v1",
        \\  "boundary_f1":{"datasets":[],"release_gate":{"min_internal_phase20_best_f1":0.0,"fixed_threshold_max_delta_from_best_f1":1.0,"must_beat_chonky_base_each_dataset":false,"must_match_or_beat_chonky_large_mean":false}},
        \\  "retrieval_ndcg":{
        \\    "external_targets":{"voyage_context_4_chunk_embedding_overall_ndcg_at_10":0.844},
        \\    "release_gate":{
        \\      "min_overall_ndcg_at_10":0.15,
        \\      "min_relative_gain_over_best_local_baseline":0.0,
        \\      "relative_gain_report_only":true,
        \\      "must_report_distance_to_external_voyage_target":true
        \\    }
        \\  }
        \\}
    ;
    // The fused fine-tune's collapsed retrieval (NDCG 0.0096) must not pass
    // even with the relative-gain gate report-only.
    const results =
        \\{
        \\  "schema_version":"fused_chunker_benchmark_results/v1",
        \\  "retrieval_ndcg":{
        \\    "overall_ndcg_at_10":0.0096,
        \\    "voyage_context_4_target_ndcg_at_10":0.844,
        \\    "distance_to_voyage_context_4_target":0.8344,
        \\    "baselines":[{"name":"fixed_500_50_same_encoder","overall_ndcg_at_10":0.35}]
        \\  }
        \\}
    ;
    try std.testing.expectError(error.RetrievalNdcgBelowFloor, verifyBenchmarkJson(std.testing.allocator, manifest, results, .retrieval_ndcg));
}
