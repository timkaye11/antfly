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
    run_path: []const u8,
    out_path: []const u8,
    output_dimension: ?u32 = null,
    baselines: []const BaselineInput = &.{},
};

const BaselineInput = struct {
    name: []const u8,
    path: []const u8,
};

const RawHit = struct {
    id: ?[]const u8 = null,
    doc_id: ?[]const u8 = null,
    chunk_id: ?[]const u8 = null,
    score: ?f64 = null,
};

const RawRecord = struct {
    query_id: ?[]const u8 = null,
    relevant_ids: ?[]const []const u8 = null,
    relevant: ?[]const []const u8 = null,
    ranked_ids: ?[]const []const u8 = null,
    results: ?[]const RawHit = null,
    hits: ?[]const RawHit = null,
};

const RetrievalMetrics = struct {
    queries: usize = 0,
    recall_at_1: f64 = 0,
    recall_at_10: f64 = 0,
    ndcg_at_10: f64 = 0,
    mrr: f64 = 0,
};

const OutputBaseline = struct {
    name: []const u8,
    overall_ndcg_at_10: f64,
    recall_at_1: f64,
    recall_at_10: f64,
    mrr: f64,
    queries: usize,
};

const OutputRetrievalLane = struct {
    output_dimension: ?u32 = null,
    overall_ndcg_at_10: f64,
    voyage_context_4_target_ndcg_at_10: f64,
    distance_to_voyage_context_4_target: f64,
    best_local_baseline_ndcg_at_10: ?f64 = null,
    relative_gain_over_best_local_baseline: ?f64 = null,
    recall_at_1: f64,
    recall_at_10: f64,
    mrr: f64,
    queries: usize,
    baselines: []const OutputBaseline,
};

const OutputFile = struct {
    schema_version: []const u8 = expected_results_schema,
    retrieval_ndcg: OutputRetrievalLane,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const opts = try parseOptions(allocator, &args) orelse return;
    defer allocator.free(opts.baselines);

    const run_bytes = try compat.cwd().readFileAlloc(compat.io(), opts.run_path, allocator, .limited(512 * 1024 * 1024));
    defer allocator.free(run_bytes);
    const output = try scoreRetrievalBenchmarkJson(allocator, run_bytes, opts.output_dimension, opts.baselines);
    defer allocator.free(output);
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = opts.out_path, .data = output });

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(compat.io(), &stdout_buf);
    try std.json.Stringify.value(.{
        .status = "written",
        .run = opts.run_path,
        .out = opts.out_path,
        .baselines = opts.baselines.len,
    }, .{ .whitespace = .indent_2 }, &stdout.interface);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}

fn parseOptions(allocator: std.mem.Allocator, args: *std.process.Args.Iterator) !?Options {
    var run_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var output_dimension: ?u32 = null;
    var baselines = std.ArrayListUnmanaged(BaselineInput).empty;
    errdefer baselines.deinit(allocator);

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--run")) {
            run_path = args.next() orelse return error.MissingRunPath;
        } else if (std.mem.eql(u8, arg, "--out")) {
            out_path = args.next() orelse return error.MissingOutPath;
        } else if (std.mem.eql(u8, arg, "--output-dimension")) {
            output_dimension = try std.fmt.parseUnsigned(u32, args.next() orelse return error.MissingOutputDimension, 10);
        } else if (std.mem.eql(u8, arg, "--baseline")) {
            try baselines.append(allocator, try parseBaselineInput(args.next() orelse return error.MissingBaseline));
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            usage();
            baselines.deinit(allocator);
            return null;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }
    if (run_path == null or out_path == null) {
        usage();
        return if (run_path == null) error.MissingRunPath else error.MissingOutPath;
    }
    return .{
        .run_path = run_path.?,
        .out_path = out_path.?,
        .output_dimension = output_dimension,
        .baselines = try baselines.toOwnedSlice(allocator),
    };
}

fn parseBaselineInput(raw: []const u8) !BaselineInput {
    const sep = std.mem.indexOfScalar(u8, raw, '=') orelse std.mem.indexOfScalar(u8, raw, ':') orelse return error.InvalidBaseline;
    if (sep == 0 or sep + 1 >= raw.len) return error.InvalidBaseline;
    return .{
        .name = raw[0..sep],
        .path = raw[sep + 1 ..],
    };
}

pub fn scoreRetrievalBenchmarkJson(
    allocator: std.mem.Allocator,
    run_jsonl: []const u8,
    output_dimension: ?u32,
    baseline_inputs: []const BaselineInput,
) ![]u8 {
    const primary = try computeRetrievalMetricsFromJsonl(allocator, run_jsonl);
    var baselines = try allocator.alloc(OutputBaseline, baseline_inputs.len);
    errdefer allocator.free(baselines);
    var best_baseline_ndcg: ?f64 = null;
    for (baseline_inputs, 0..) |baseline, i| {
        const raw = try compat.cwd().readFileAlloc(compat.io(), baseline.path, allocator, .limited(512 * 1024 * 1024));
        defer allocator.free(raw);
        const metrics = try computeRetrievalMetricsFromJsonl(allocator, raw);
        best_baseline_ndcg = if (best_baseline_ndcg) |best|
            @max(best, metrics.ndcg_at_10)
        else
            metrics.ndcg_at_10;
        baselines[i] = .{
            .name = baseline.name,
            .overall_ndcg_at_10 = metrics.ndcg_at_10,
            .recall_at_1 = metrics.recall_at_1,
            .recall_at_10 = metrics.recall_at_10,
            .mrr = metrics.mrr,
            .queries = metrics.queries,
        };
    }

    const voyage_target = voyageContext4TargetForDimension(output_dimension);
    const relative_gain = if (best_baseline_ndcg) |best|
        if (best > 0.0) (primary.ndcg_at_10 - best) / best else null
    else
        null;
    const output = OutputFile{ .retrieval_ndcg = .{
        .output_dimension = output_dimension,
        .overall_ndcg_at_10 = primary.ndcg_at_10,
        .voyage_context_4_target_ndcg_at_10 = voyage_target,
        .distance_to_voyage_context_4_target = voyage_target - primary.ndcg_at_10,
        .best_local_baseline_ndcg_at_10 = best_baseline_ndcg,
        .relative_gain_over_best_local_baseline = relative_gain,
        .recall_at_1 = primary.recall_at_1,
        .recall_at_10 = primary.recall_at_10,
        .mrr = primary.mrr,
        .queries = primary.queries,
        .baselines = baselines,
    } };
    var buffer: std.Io.Writer.Allocating = .init(allocator);
    errdefer buffer.deinit();
    try std.json.Stringify.value(output, .{ .whitespace = .indent_2 }, &buffer.writer);
    try buffer.writer.writeByte('\n');
    allocator.free(baselines);
    return buffer.toOwnedSlice();
}

pub fn computeRetrievalMetricsFromJsonl(allocator: std.mem.Allocator, jsonl: []const u8) !RetrievalMetrics {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var sum_r1: f64 = 0;
    var sum_r10: f64 = 0;
    var sum_ndcg10: f64 = 0;
    var sum_mrr: f64 = 0;
    var queries: usize = 0;

    var lines = std.mem.tokenizeScalar(u8, jsonl, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r\n");
        if (line.len == 0) continue;
        const record = try std.json.parseFromSliceLeaky(RawRecord, arena_alloc, line, .{ .ignore_unknown_fields = true });
        const relevant = record.relevant_ids orelse record.relevant orelse return error.MissingRelevantIds;
        if (relevant.len == 0) continue;
        const ranked = try rankedIdsFromRecord(allocator, record);
        defer allocator.free(ranked);
        if (ranked.len == 0) return error.MissingRankedIds;

        queries += 1;
        if (containsId(relevant, ranked[0])) sum_r1 += 1.0;
        var found_top10 = false;
        var first_rank: usize = 0;
        const top_k = @min(ranked.len, 10);
        var dcg10: f64 = 0;
        for (ranked[0..top_k], 0..) |id, idx| {
            if (containsId(relevant, id)) {
                if (!found_top10) {
                    sum_r10 += 1.0;
                    found_top10 = true;
                }
                if (first_rank == 0) first_rank = idx + 1;
                dcg10 += 1.0 / @log2(@as(f64, @floatFromInt(idx + 2)));
            }
        }
        if (first_rank == 0) {
            for (ranked[top_k..], top_k..) |id, idx| {
                if (containsId(relevant, id)) {
                    first_rank = idx + 1;
                    break;
                }
            }
        }
        if (first_rank > 0) sum_mrr += 1.0 / @as(f64, @floatFromInt(first_rank));
        const idcg10 = idealDcg(@min(relevant.len, 10));
        if (idcg10 > 0) sum_ndcg10 += dcg10 / idcg10;
    }

    if (queries == 0) return error.NoRetrievalQueries;
    const denom: f64 = @floatFromInt(queries);
    return .{
        .queries = queries,
        .recall_at_1 = sum_r1 / denom,
        .recall_at_10 = sum_r10 / denom,
        .ndcg_at_10 = sum_ndcg10 / denom,
        .mrr = sum_mrr / denom,
    };
}

fn voyageContext4TargetForDimension(output_dimension: ?u32) f64 {
    return switch (output_dimension orelse 0) {
        256 => 0.8054,
        512 => 0.8266,
        1024 => 0.8381,
        else => 0.8440,
    };
}

fn rankedIdsFromRecord(allocator: std.mem.Allocator, record: RawRecord) ![]const []const u8 {
    if (record.ranked_ids) |ids| return allocator.dupe([]const u8, ids);
    const hits = record.results orelse record.hits orelse return error.MissingRankedIds;
    var ids = try allocator.alloc([]const u8, hits.len);
    errdefer allocator.free(ids);
    for (hits, 0..) |hit, i| {
        ids[i] = hit.id orelse hit.doc_id orelse hit.chunk_id orelse return error.InvalidRetrievalHit;
    }
    return ids;
}

fn containsId(ids: []const []const u8, needle: []const u8) bool {
    for (ids) |id| {
        if (std.mem.eql(u8, id, needle)) return true;
    }
    return false;
}

fn idealDcg(positive_count: usize) f64 {
    var out: f64 = 0;
    for (0..positive_count) |idx| {
        out += 1.0 / @log2(@as(f64, @floatFromInt(idx + 2)));
    }
    return out;
}

fn jsonF64(value: std.json.Value) !f64 {
    return switch (value) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.ExpectedJsonNumber,
    };
}

fn usage() void {
    std.debug.print(
        \\usage: score-fused-chunker-retrieval-benchmark --run <jsonl> --out <json> [options]
        \\
        \\  --output-dimension <n>     Dense output dimension for Voyage target mapping
        \\  --baseline <name=path>     Baseline run JSONL to score; repeatable
        \\
        \\Run JSONL records must include relevant_ids/relevant and either ranked_ids
        \\or results/hits with id, doc_id, or chunk_id fields.
        \\
    , .{});
}

test "retrieval metrics compute binary ndcg recall and mrr" {
    const jsonl =
        \\{"query_id":"q1","relevant_ids":["d2"],"ranked_ids":["d1","d2","d3"]}
        \\{"query_id":"q2","relevant_ids":["d4","d5"],"results":[{"id":"d5"},{"id":"d4"},{"id":"d9"}]}
        \\
    ;
    const metrics = try computeRetrievalMetricsFromJsonl(std.testing.allocator, jsonl);
    try std.testing.expectEqual(@as(usize, 2), metrics.queries);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), metrics.recall_at_1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), metrics.recall_at_10, 1e-12);
    const expected_q1 = (1.0 / @log2(@as(f64, 3.0))) / 1.0;
    const expected_q2 = 1.0;
    try std.testing.expectApproxEqAbs((expected_q1 + expected_q2) / 2.0, metrics.ndcg_at_10, 1e-12);
    try std.testing.expectApproxEqAbs((0.5 + 1.0) / 2.0, metrics.mrr, 1e-12);
}

test "retrieval result JSON matches benchmark schema" {
    const allocator = std.testing.allocator;
    const jsonl =
        \\{"query_id":"q1","relevant_ids":["d1"],"ranked_ids":["d1","d2"]}
        \\
    ;
    const output = try scoreRetrievalBenchmarkJson(allocator, jsonl, 256, &.{});
    defer allocator.free(output);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, output, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings(expected_results_schema, obj.get("schema_version").?.string);
    const retrieval = obj.get("retrieval_ndcg").?.object;
    try std.testing.expectEqual(@as(i64, 256), retrieval.get("output_dimension").?.integer);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), try jsonF64(retrieval.get("overall_ndcg_at_10").?), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8054), try jsonF64(retrieval.get("voyage_context_4_target_ndcg_at_10").?), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, -0.1946), try jsonF64(retrieval.get("distance_to_voyage_context_4_target").?), 1e-12);
}
