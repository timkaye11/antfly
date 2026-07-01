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

const analysis_schema_version = "fused_chunker_boundary_alignment_analysis/v1";

const Options = struct {
    alignment_path: ?[]const u8 = null,
    analysis_out_path: ?[]const u8 = null,
};

const AlignmentInspection = struct {
    record_count: usize = 0,
    skipped_record_count: usize = 0,
    gold_boundary_records: usize = 0,
    top_probability_records: usize = 0,
    top_probability_gold_records: usize = 0,
    top_probability_false_positive_records: usize = 0,
    top_probability_chunk_start_records: usize = 0,
    top_probability_previous_chunk_end_records: usize = 0,
    top_probability_near_gold_within_1_records: usize = 0,
    top_probability_near_gold_within_2_records: usize = 0,
    top_probability_before_nearest_gold_records: usize = 0,
    top_probability_after_nearest_gold_records: usize = 0,
    top_probability_exact_gold_records: usize = 0,
    gold_top_1pct_count: usize = 0,
    gold_top_5pct_count: usize = 0,
    gold_top_10pct_count: usize = 0,
    gold_rank_percentiles: std.ArrayListUnmanaged(f64) = .empty,
    gold_probability_sum: f64 = 0,
    gold_margin_sum: f64 = 0,
    top_false_probability_sum: f64 = 0,
    top_false_margin_sum: f64 = 0,
    token_surface_stats: std.AutoHashMapUnmanaged(u64, TokenSurfaceStats) = .empty,

    fn deinit(self: *AlignmentInspection, allocator: std.mem.Allocator) void {
        self.gold_rank_percentiles.deinit(allocator);
        self.token_surface_stats.deinit(allocator);
        self.* = undefined;
    }
};

const TokenSurfaceStats = struct {
    gold_boundary_count: usize = 0,
    top_false_positive_count: usize = 0,
    gold_probability_sum: f64 = 0,
    top_false_probability_sum: f64 = 0,
};

const TokenSurfaceConfusion = struct {
    boundary_like_top_false_positive_records: usize = 0,
    boundary_like_top_false_positive_rate: ?f64 = null,
    top_false_hot_token_id: ?u64 = null,
    top_false_hot_token_count: usize = 0,
    top_false_hot_token_rate: ?f64 = null,
    top_false_hot_token_gold_boundary_count: usize = 0,
    top_false_hot_token_false_to_gold_ratio: ?f64 = null,
    top_false_hot_token_mean_false_probability: ?f64 = null,
    top_false_hot_token_mean_gold_probability: ?f64 = null,
    token_surface_confounded: bool = false,
};

const AlignmentFailureModes = struct {
    no_gold_boundary_records: bool = false,
    gold_ranks_buried: bool = false,
    top_false_positives_dominate: bool = false,
    top_false_positives_near_gold: bool = false,
    top_tokens_shifted_to_previous_chunk_end: bool = false,
    top_tokens_shifted_before_gold: bool = false,
    top_tokens_shifted_after_gold: bool = false,
    gold_probability_below_top_false_positive: bool = false,
    token_surface_confounded: bool = false,
};

const AlignmentAnalysis = struct {
    schema_version: []const u8 = analysis_schema_version,
    status: []const u8 = "analyzed",
    alignment: []const u8,
    record_count: usize,
    skipped_record_count: usize,
    gold_boundary_records: usize,
    top_probability_records: usize,
    top_probability_gold_records: usize,
    top_probability_false_positive_records: usize,
    top_probability_false_positive_rate: ?f64,
    top_probability_chunk_start_records: usize,
    top_probability_previous_chunk_end_records: usize,
    top_probability_near_gold_within_1_records: usize,
    top_probability_near_gold_within_2_records: usize,
    top_probability_before_nearest_gold_records: usize,
    top_probability_after_nearest_gold_records: usize,
    top_probability_exact_gold_records: usize,
    gold_mean_rank_percentile: ?f64,
    gold_median_rank_percentile: ?f64,
    gold_p90_rank_percentile: ?f64,
    gold_p99_rank_percentile: ?f64,
    gold_top_1pct_recall: ?f64,
    gold_top_5pct_recall: ?f64,
    gold_top_10pct_recall: ?f64,
    mean_gold_probability: ?f64,
    mean_top_false_positive_probability: ?f64,
    mean_gold_margin: ?f64,
    mean_top_false_positive_margin: ?f64,
    mean_gold_probability_minus_top_false_positive: ?f64,
    mean_gold_margin_minus_top_false_positive: ?f64,
    token_surface_confusion: TokenSurfaceConfusion,
    failure_modes: AlignmentFailureModes,
    stop_long_training: bool,
    next_target: []const u8,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const opts = try parseOptions(&args) orelse return;
    const alignment_path = opts.alignment_path orelse return error.MissingAlignment;
    const analysis = try analyzeAlignmentDumpFile(allocator, alignment_path);
    if (opts.analysis_out_path) |path| {
        const output = try stringifyAnalysis(allocator, analysis);
        defer allocator.free(output);
        try compat.cwd().writeFile(compat.io(), .{ .sub_path = path, .data = output });
    }

    var stdout_buf: [8192]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(compat.io(), &stdout_buf);
    try std.json.Stringify.value(analysis, .{ .whitespace = .indent_2 }, &stdout.interface);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}

fn parseOptions(args: *std.process.Args.Iterator) !?Options {
    var opts = Options{};
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--alignment")) {
            opts.alignment_path = args.next() orelse return error.MissingAlignment;
        } else if (std.mem.eql(u8, arg, "--analysis-out")) {
            opts.analysis_out_path = args.next() orelse return error.MissingAnalysisOut;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            usage();
            return null;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }
    if (opts.alignment_path == null) {
        usage();
        return error.MissingAlignment;
    }
    return opts;
}

fn usage() void {
    std.debug.print(
        \\usage: analyze-fused-chunker-boundary-alignment-dump --alignment <boundary_alignment_*.jsonl> [--analysis-out <path>]
        \\
    , .{});
}

fn analyzeAlignmentDumpFile(allocator: std.mem.Allocator, alignment_path: []const u8) !AlignmentAnalysis {
    const bytes = try compat.cwd().readFileAlloc(compat.io(), alignment_path, allocator, .limited(128 * 1024 * 1024));
    defer allocator.free(bytes);
    return analyzeAlignmentDumpJsonl(allocator, alignment_path, bytes);
}

fn analyzeAlignmentDumpJsonl(allocator: std.mem.Allocator, alignment_path: []const u8, bytes: []const u8) !AlignmentAnalysis {
    var inspection = try inspectAlignmentDumpJsonl(allocator, bytes);
    defer inspection.deinit(allocator);

    const rank_stats = computeRankStats(allocator, inspection.gold_rank_percentiles.items);
    const mean_gold_probability = meanOrNull(inspection.gold_probability_sum, inspection.gold_boundary_records);
    const mean_top_false_probability = meanOrNull(inspection.top_false_probability_sum, inspection.top_probability_false_positive_records);
    const mean_gold_margin = meanOrNull(inspection.gold_margin_sum, inspection.gold_boundary_records);
    const mean_top_false_margin = meanOrNull(inspection.top_false_margin_sum, inspection.top_probability_false_positive_records);
    const token_surface = computeTokenSurfaceConfusion(&inspection);
    const modes = classifyAlignment(inspection, rank_stats, mean_gold_probability, mean_top_false_probability, token_surface);

    return .{
        .alignment = alignment_path,
        .record_count = inspection.record_count,
        .skipped_record_count = inspection.skipped_record_count,
        .gold_boundary_records = inspection.gold_boundary_records,
        .top_probability_records = inspection.top_probability_records,
        .top_probability_gold_records = inspection.top_probability_gold_records,
        .top_probability_false_positive_records = inspection.top_probability_false_positive_records,
        .top_probability_false_positive_rate = if (inspection.top_probability_records == 0)
            null
        else
            @as(f64, @floatFromInt(inspection.top_probability_false_positive_records)) /
                @as(f64, @floatFromInt(inspection.top_probability_records)),
        .top_probability_chunk_start_records = inspection.top_probability_chunk_start_records,
        .top_probability_previous_chunk_end_records = inspection.top_probability_previous_chunk_end_records,
        .top_probability_near_gold_within_1_records = inspection.top_probability_near_gold_within_1_records,
        .top_probability_near_gold_within_2_records = inspection.top_probability_near_gold_within_2_records,
        .top_probability_before_nearest_gold_records = inspection.top_probability_before_nearest_gold_records,
        .top_probability_after_nearest_gold_records = inspection.top_probability_after_nearest_gold_records,
        .top_probability_exact_gold_records = inspection.top_probability_exact_gold_records,
        .gold_mean_rank_percentile = rank_stats.mean,
        .gold_median_rank_percentile = rank_stats.median,
        .gold_p90_rank_percentile = rank_stats.p90,
        .gold_p99_rank_percentile = rank_stats.p99,
        .gold_top_1pct_recall = recallOrNull(inspection.gold_top_1pct_count, inspection.gold_boundary_records),
        .gold_top_5pct_recall = recallOrNull(inspection.gold_top_5pct_count, inspection.gold_boundary_records),
        .gold_top_10pct_recall = recallOrNull(inspection.gold_top_10pct_count, inspection.gold_boundary_records),
        .mean_gold_probability = mean_gold_probability,
        .mean_top_false_positive_probability = mean_top_false_probability,
        .mean_gold_margin = mean_gold_margin,
        .mean_top_false_positive_margin = mean_top_false_margin,
        .mean_gold_probability_minus_top_false_positive = diffOrNull(mean_gold_probability, mean_top_false_probability),
        .mean_gold_margin_minus_top_false_positive = diffOrNull(mean_gold_margin, mean_top_false_margin),
        .token_surface_confusion = token_surface,
        .failure_modes = modes,
        .stop_long_training = shouldStopLongTraining(modes),
        .next_target = chooseNextTarget(modes),
    };
}

fn inspectAlignmentDumpJsonl(allocator: std.mem.Allocator, bytes: []const u8) !AlignmentInspection {
    var out = AlignmentInspection{};
    errdefer out.deinit(allocator);

    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r\n");
        if (line.len == 0) continue;

        var parsed = std.json.parseFromSlice(std.json.Value, allocator, line, .{}) catch {
            out.skipped_record_count += 1;
            continue;
        };
        defer parsed.deinit();
        if (parsed.value != .object) {
            out.skipped_record_count += 1;
            continue;
        }
        const obj = parsed.value.object;
        const event = jsonString(obj.get("event")) orelse "";
        if (!std.mem.eql(u8, event, "boundary_alignment_token")) {
            out.skipped_record_count += 1;
            continue;
        }

        out.record_count += 1;
        const record_kind = jsonString(obj.get("record_kind")) orelse "";
        const gold_boundary = jsonBool(obj.get("gold_boundary")) orelse false;
        const probability = jsonF64(obj.get("probability")) orelse 0;
        const margin = jsonF64(obj.get("margin")) orelse 0;
        const token_id = jsonU64(obj.get("token_id"));

        if (std.mem.eql(u8, record_kind, "gold_boundary")) {
            out.gold_boundary_records += 1;
            out.gold_probability_sum += probability;
            out.gold_margin_sum += margin;
            if (token_id) |id| {
                var entry = try out.token_surface_stats.getOrPut(allocator, id);
                if (!entry.found_existing) entry.value_ptr.* = .{};
                entry.value_ptr.gold_boundary_count += 1;
                entry.value_ptr.gold_probability_sum += probability;
            }
            if (recordRankPercentile(obj)) |rank_percentile| {
                try out.gold_rank_percentiles.append(allocator, rank_percentile);
                if (rank_percentile <= 0.01) out.gold_top_1pct_count += 1;
                if (rank_percentile <= 0.05) out.gold_top_5pct_count += 1;
                if (rank_percentile <= 0.10) out.gold_top_10pct_count += 1;
            }
        } else if (std.mem.eql(u8, record_kind, "top_probability")) {
            out.top_probability_records += 1;
            if (gold_boundary) {
                out.top_probability_gold_records += 1;
            } else {
                out.top_probability_false_positive_records += 1;
                out.top_false_probability_sum += probability;
                out.top_false_margin_sum += margin;
                if (token_id) |id| {
                    var entry = try out.token_surface_stats.getOrPut(allocator, id);
                    if (!entry.found_existing) entry.value_ptr.* = .{};
                    entry.value_ptr.top_false_positive_count += 1;
                    entry.value_ptr.top_false_probability_sum += probability;
                }
            }
            if (jsonUsize(obj.get("chunk_start_index")) != null) out.top_probability_chunk_start_records += 1;
            if (jsonUsize(obj.get("previous_chunk_end_index")) != null) out.top_probability_previous_chunk_end_records += 1;
            if (jsonI64(obj.get("nearest_gold_delta"))) |delta| {
                const abs_delta = if (delta < 0) -delta else delta;
                if (abs_delta == 0) out.top_probability_exact_gold_records += 1;
                if (abs_delta <= 1) out.top_probability_near_gold_within_1_records += 1;
                if (abs_delta <= 2) out.top_probability_near_gold_within_2_records += 1;
                if (delta > 0) out.top_probability_before_nearest_gold_records += 1;
                if (delta < 0) out.top_probability_after_nearest_gold_records += 1;
            }
        }
    }

    return out;
}

fn recordRankPercentile(obj: std.json.ObjectMap) ?f64 {
    if (jsonF64(obj.get("probability_rank_percentile"))) |rank_percentile| return rank_percentile;
    const rank = jsonUsize(obj.get("probability_rank")) orelse return null;
    const valid = jsonUsize(obj.get("valid_token_count")) orelse return null;
    if (rank == 0 or valid == 0) return null;
    return @as(f64, @floatFromInt(rank)) / @as(f64, @floatFromInt(valid));
}

const RankStats = struct {
    mean: ?f64 = null,
    median: ?f64 = null,
    p90: ?f64 = null,
    p99: ?f64 = null,
};

fn computeRankStats(allocator: std.mem.Allocator, values: []const f64) RankStats {
    if (values.len == 0) return .{};
    const sorted = allocator.dupe(f64, values) catch return .{};
    defer allocator.free(sorted);
    std.mem.sort(f64, sorted, {}, f64LessThan);

    var sum: f64 = 0;
    for (sorted) |value| sum += value;
    return .{
        .mean = sum / @as(f64, @floatFromInt(sorted.len)),
        .median = percentile(sorted, 0.50),
        .p90 = percentile(sorted, 0.90),
        .p99 = percentile(sorted, 0.99),
    };
}

fn f64LessThan(_: void, a: f64, b: f64) bool {
    return a < b;
}

fn percentile(sorted: []const f64, q: f64) ?f64 {
    if (sorted.len == 0) return null;
    const clamped = @min(@max(q, 0), 1);
    const idx_f = clamped * @as(f64, @floatFromInt(sorted.len - 1));
    const idx: usize = @intFromFloat(@round(idx_f));
    return sorted[idx];
}

fn computeTokenSurfaceConfusion(inspection: *const AlignmentInspection) TokenSurfaceConfusion {
    var out = TokenSurfaceConfusion{};
    var hot_token_stats: ?TokenSurfaceStats = null;

    var it = inspection.token_surface_stats.iterator();
    while (it.next()) |entry| {
        const stats = entry.value_ptr.*;
        if (stats.top_false_positive_count == 0) continue;
        if (stats.gold_boundary_count > 0) {
            out.boundary_like_top_false_positive_records += stats.top_false_positive_count;
        }
        if (stats.top_false_positive_count > out.top_false_hot_token_count) {
            out.top_false_hot_token_id = entry.key_ptr.*;
            out.top_false_hot_token_count = stats.top_false_positive_count;
            hot_token_stats = stats;
        }
    }

    out.boundary_like_top_false_positive_rate = recallOrNull(
        out.boundary_like_top_false_positive_records,
        inspection.top_probability_false_positive_records,
    );
    out.top_false_hot_token_rate = recallOrNull(
        out.top_false_hot_token_count,
        inspection.top_probability_false_positive_records,
    );

    if (hot_token_stats) |stats| {
        out.top_false_hot_token_gold_boundary_count = stats.gold_boundary_count;
        out.top_false_hot_token_false_to_gold_ratio = if (stats.gold_boundary_count == 0)
            null
        else
            @as(f64, @floatFromInt(stats.top_false_positive_count)) /
                @as(f64, @floatFromInt(stats.gold_boundary_count));
        out.top_false_hot_token_mean_false_probability = meanOrNull(
            stats.top_false_probability_sum,
            stats.top_false_positive_count,
        );
        out.top_false_hot_token_mean_gold_probability = meanOrNull(
            stats.gold_probability_sum,
            stats.gold_boundary_count,
        );
    }

    const boundary_like_rate = out.boundary_like_top_false_positive_rate orelse 0;
    const hot_rate = out.top_false_hot_token_rate orelse 0;
    const hot_ratio = out.top_false_hot_token_false_to_gold_ratio orelse 0;
    out.token_surface_confounded =
        boundary_like_rate >= 0.25 or
        (out.top_false_hot_token_gold_boundary_count > 0 and hot_rate >= 0.10 and hot_ratio >= 2.0);
    return out;
}

fn classifyAlignment(
    inspection: AlignmentInspection,
    rank_stats: RankStats,
    mean_gold_probability: ?f64,
    mean_top_false_probability: ?f64,
    token_surface: TokenSurfaceConfusion,
) AlignmentFailureModes {
    const top_false_rate = if (inspection.top_probability_records == 0)
        0
    else
        @as(f64, @floatFromInt(inspection.top_probability_false_positive_records)) /
            @as(f64, @floatFromInt(inspection.top_probability_records));
    const near_gold_false_rate = if (inspection.top_probability_false_positive_records == 0)
        0
    else
        @as(f64, @floatFromInt(inspection.top_probability_near_gold_within_2_records -| inspection.top_probability_exact_gold_records)) /
            @as(f64, @floatFromInt(inspection.top_probability_false_positive_records));
    const previous_end_rate = if (inspection.top_probability_records == 0)
        0
    else
        @as(f64, @floatFromInt(inspection.top_probability_previous_chunk_end_records)) /
            @as(f64, @floatFromInt(inspection.top_probability_records));
    const before_rate = if (inspection.top_probability_records == 0)
        0
    else
        @as(f64, @floatFromInt(inspection.top_probability_before_nearest_gold_records)) /
            @as(f64, @floatFromInt(inspection.top_probability_records));
    const after_rate = if (inspection.top_probability_records == 0)
        0
    else
        @as(f64, @floatFromInt(inspection.top_probability_after_nearest_gold_records)) /
            @as(f64, @floatFromInt(inspection.top_probability_records));
    const probability_gap = diffOrNull(mean_gold_probability, mean_top_false_probability) orelse 0;

    return .{
        .no_gold_boundary_records = inspection.gold_boundary_records == 0,
        .gold_ranks_buried = (rank_stats.median orelse 0) > 0.25 or (rank_stats.p90 orelse 0) > 0.75,
        .top_false_positives_dominate = top_false_rate >= 0.80,
        .top_false_positives_near_gold = near_gold_false_rate >= 0.50,
        .top_tokens_shifted_to_previous_chunk_end = previous_end_rate >= 0.50,
        .top_tokens_shifted_before_gold = before_rate >= 0.50,
        .top_tokens_shifted_after_gold = after_rate >= 0.50,
        .gold_probability_below_top_false_positive = mean_gold_probability != null and
            mean_top_false_probability != null and probability_gap < 0,
        .token_surface_confounded = token_surface.token_surface_confounded,
    };
}

fn shouldStopLongTraining(modes: AlignmentFailureModes) bool {
    return modes.no_gold_boundary_records or
        modes.gold_ranks_buried or
        modes.top_false_positives_dominate or
        modes.gold_probability_below_top_false_positive or
        modes.token_surface_confounded;
}

fn chooseNextTarget(modes: AlignmentFailureModes) []const u8 {
    if (modes.no_gold_boundary_records) return "fix_boundary_label_generation_or_truncation";
    if (modes.top_tokens_shifted_to_previous_chunk_end) return "compare_start_vs_previous_end_boundary_labeling";
    if (modes.top_false_positives_near_gold and (modes.top_tokens_shifted_before_gold or modes.top_tokens_shifted_after_gold)) return "check_one_token_boundary_offset_shift";
    if (modes.token_surface_confounded) return "add_same_surface_hard_negative_or_context_features";
    if (modes.gold_probability_below_top_false_positive) return "check_boundary_label_polarity_and_feature_label_alignment";
    if (modes.gold_ranks_buried) return "improve_boundary_feature_separation_before_long_training";
    return "fit_validation_threshold_or_continue_short_probe";
}

fn meanOrNull(sum: f64, count: usize) ?f64 {
    if (count == 0) return null;
    return sum / @as(f64, @floatFromInt(count));
}

fn recallOrNull(count: usize, denom: usize) ?f64 {
    if (denom == 0) return null;
    return @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(denom));
}

fn diffOrNull(a: ?f64, b: ?f64) ?f64 {
    const av = a orelse return null;
    const bv = b orelse return null;
    return av - bv;
}

fn stringifyAnalysis(allocator: std.mem.Allocator, analysis: AlignmentAnalysis) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();
    try std.json.Stringify.value(analysis, .{ .whitespace = .indent_2 }, &out.writer);
    try out.writer.writeByte('\n');
    return out.toOwnedSlice();
}

fn jsonString(value: ?std.json.Value) ?[]const u8 {
    const v = value orelse return null;
    return switch (v) {
        .string => |s| s,
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

fn jsonI64(value: ?std.json.Value) ?i64 {
    const v = value orelse return null;
    return switch (v) {
        .integer => |i| i,
        .float => |f| if (std.math.isFinite(f)) @intFromFloat(f) else null,
        else => null,
    };
}

fn jsonUsize(value: ?std.json.Value) ?usize {
    const v = value orelse return null;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        .float => |f| if (std.math.isFinite(f) and f >= 0) @intFromFloat(f) else null,
        else => null,
    };
}

fn jsonU64(value: ?std.json.Value) ?u64 {
    const v = value orelse return null;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        .float => |f| if (std.math.isFinite(f) and f >= 0) @intFromFloat(f) else null,
        else => null,
    };
}

test "alignment dump analysis detects buried gold and false positives" {
    const jsonl =
        \\{"event":"boundary_alignment_token","record_kind":"gold_boundary","gold_boundary":true,"probability":0.20,"margin":-1.0,"probability_rank_percentile":0.80}
        \\{"event":"boundary_alignment_token","record_kind":"gold_boundary","gold_boundary":true,"probability":0.30,"margin":-0.5,"probability_rank_percentile":0.70}
        \\{"event":"boundary_alignment_token","record_kind":"top_probability","gold_boundary":false,"probability":0.90,"margin":2.0,"top_probability_rank":1,"probability_rank_percentile":0.01,"nearest_gold_delta":1,"previous_chunk_end_index":0}
        \\{"event":"boundary_alignment_token","record_kind":"top_probability","gold_boundary":false,"probability":0.80,"margin":1.5,"top_probability_rank":2,"probability_rank_percentile":0.02,"nearest_gold_delta":1,"previous_chunk_end_index":1}
        \\
    ;
    const analysis = try analyzeAlignmentDumpJsonl(std.testing.allocator, "memory.jsonl", jsonl);
    try std.testing.expectEqual(@as(usize, 2), analysis.gold_boundary_records);
    try std.testing.expectEqual(@as(usize, 2), analysis.top_probability_false_positive_records);
    try std.testing.expect(analysis.failure_modes.gold_ranks_buried);
    try std.testing.expect(analysis.failure_modes.top_false_positives_dominate);
    try std.testing.expect(analysis.failure_modes.top_tokens_shifted_to_previous_chunk_end);
    try std.testing.expect(analysis.failure_modes.gold_probability_below_top_false_positive);
    try std.testing.expectEqualStrings("compare_start_vs_previous_end_boundary_labeling", analysis.next_target);
}

test "alignment dump analysis accepts high-ranked gold records" {
    const jsonl =
        \\{"event":"boundary_alignment_token","record_kind":"gold_boundary","gold_boundary":true,"probability":0.90,"margin":2.0,"probability_rank_percentile":0.01}
        \\{"event":"boundary_alignment_token","record_kind":"gold_boundary","gold_boundary":true,"probability":0.85,"margin":1.7,"probability_rank_percentile":0.02}
        \\{"event":"boundary_alignment_token","record_kind":"top_probability","gold_boundary":true,"probability":0.90,"margin":2.0,"top_probability_rank":1,"probability_rank_percentile":0.01,"nearest_gold_delta":0,"chunk_start_index":1}
        \\{"event":"boundary_alignment_token","record_kind":"top_probability","gold_boundary":true,"probability":0.85,"margin":1.7,"top_probability_rank":2,"probability_rank_percentile":0.02,"nearest_gold_delta":0,"chunk_start_index":2}
        \\
    ;
    const analysis = try analyzeAlignmentDumpJsonl(std.testing.allocator, "memory.jsonl", jsonl);
    try std.testing.expectEqual(@as(usize, 0), analysis.top_probability_false_positive_records);
    try std.testing.expect(!analysis.failure_modes.gold_ranks_buried);
    try std.testing.expect(!analysis.failure_modes.top_false_positives_dominate);
    try std.testing.expect(!analysis.stop_long_training);
    try std.testing.expectEqualStrings("fit_validation_threshold_or_continue_short_probe", analysis.next_target);
}

test "alignment dump analysis detects token surface shortcut" {
    const jsonl =
        \\{"event":"boundary_alignment_token","record_kind":"gold_boundary","token_id":380,"gold_boundary":true,"probability":0.70,"margin":0.8,"probability_rank_percentile":0.04}
        \\{"event":"boundary_alignment_token","record_kind":"gold_boundary","token_id":754,"gold_boundary":true,"probability":0.65,"margin":0.6,"probability_rank_percentile":0.05}
        \\{"event":"boundary_alignment_token","record_kind":"top_probability","token_id":380,"gold_boundary":false,"probability":0.92,"margin":2.4,"top_probability_rank":1,"probability_rank_percentile":0.01,"nearest_gold_delta":40}
        \\{"event":"boundary_alignment_token","record_kind":"top_probability","token_id":380,"gold_boundary":false,"probability":0.90,"margin":2.2,"top_probability_rank":2,"probability_rank_percentile":0.02,"nearest_gold_delta":-30}
        \\{"event":"boundary_alignment_token","record_kind":"top_probability","token_id":380,"gold_boundary":false,"probability":0.88,"margin":2.0,"top_probability_rank":3,"probability_rank_percentile":0.03,"nearest_gold_delta":20}
        \\{"event":"boundary_alignment_token","record_kind":"top_probability","token_id":754,"gold_boundary":false,"probability":0.86,"margin":1.8,"top_probability_rank":4,"probability_rank_percentile":0.04,"nearest_gold_delta":-20}
        \\
    ;
    const analysis = try analyzeAlignmentDumpJsonl(std.testing.allocator, "memory.jsonl", jsonl);
    try std.testing.expect(analysis.failure_modes.token_surface_confounded);
    try std.testing.expect(analysis.token_surface_confusion.token_surface_confounded);
    try std.testing.expectEqual(@as(?u64, 380), analysis.token_surface_confusion.top_false_hot_token_id);
    try std.testing.expectEqual(@as(usize, 3), analysis.token_surface_confusion.top_false_hot_token_count);
    try std.testing.expectEqual(@as(usize, 1), analysis.token_surface_confusion.top_false_hot_token_gold_boundary_count);
    try std.testing.expectEqualStrings("add_same_surface_hard_negative_or_context_features", analysis.next_target);
}
