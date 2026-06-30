// Copyright 2026 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

const std = @import("std");
const compat = @import("termite_io_compat");

const manifest_file_name = "fused_training_manifest.json";
const metrics_file_name = "fused_training_metrics.jsonl";
const expected_schema_version = "fused_chunker_training/v1";
const expected_artifact_family_version = "fused_chunker_phase20/v1";

const Options = struct {
    out_dir: []const u8,
    manifest_path: ?[]const u8 = null,
    metrics_path: ?[]const u8 = null,
    require_complete: bool = false,
    require_backend: ?[]const u8 = null,
    require_mpsgraph_vjp: bool = false,
    min_steps: usize = 1,
    min_fixed_f1: ?f64 = null,
    min_best_f1: ?f64 = null,
    min_mean_positive_probability_gap: ?f64 = null,
    max_avg_step_ms: ?f64 = null,
    max_peak_rss_bytes: ?usize = null,
    max_vjp_fallbacks: u64 = 0,
    require_encoder_neftune: ?bool = null,
};

const ManifestInspection = struct {
    status: []const u8,
    backend: []const u8,
    total_steps: usize,
    peak_resident_bytes: usize,
    final_checkpoint_path: ?[]const u8,
    best_val_f1: f64,
    encoder_neftune: ?bool,

    fn deinit(self: ManifestInspection, allocator: std.mem.Allocator) void {
        allocator.free(self.status);
        allocator.free(self.backend);
        if (self.final_checkpoint_path) |path| allocator.free(path);
    }
};

const MetricsInspection = struct {
    record_count: usize = 0,
    step_count: usize = 0,
    validation_count: usize = 0,
    all_losses_finite: bool = true,
    first_loss: ?f64 = null,
    final_loss: ?f64 = null,
    total_step_ms: f64 = 0,
    max_peak_resident_bytes: usize = 0,
    max_vjp_fallbacks: u64 = 0,
    mpsgraph_step_count: usize = 0,
    non_mpsgraph_vjp_step_count: usize = 0,
    best_fixed_f1: f64 = 0,
    best_threshold_f1: f64 = 0,
    probability_gap_count: usize = 0,
    min_mean_positive_probability_gap: f64 = std.math.inf(f64),

    fn avgStepMs(self: MetricsInspection) f64 {
        if (self.step_count == 0) return 0;
        return self.total_step_ms / @as(f64, @floatFromInt(self.step_count));
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const opts = try parseOptions(&args) orelse return;
    try validate(allocator, opts);
}

fn parseOptions(args: *std.process.Args.Iterator) !?Options {
    var out_dir: ?[]const u8 = null;
    var opts = Options{ .out_dir = "" };
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--out-dir")) {
            out_dir = args.next() orelse return error.MissingOutDir;
        } else if (std.mem.eql(u8, arg, "--manifest")) {
            opts.manifest_path = args.next() orelse return error.MissingManifest;
        } else if (std.mem.eql(u8, arg, "--metrics")) {
            opts.metrics_path = args.next() orelse return error.MissingMetrics;
        } else if (std.mem.eql(u8, arg, "--require-complete")) {
            opts.require_complete = true;
        } else if (std.mem.eql(u8, arg, "--require-backend")) {
            opts.require_backend = args.next() orelse return error.MissingBackend;
        } else if (std.mem.eql(u8, arg, "--require-mpsgraph-vjp")) {
            opts.require_mpsgraph_vjp = true;
        } else if (std.mem.eql(u8, arg, "--min-steps")) {
            opts.min_steps = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingMinSteps, 10);
        } else if (std.mem.eql(u8, arg, "--min-fixed-f1")) {
            opts.min_fixed_f1 = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinFixedF1);
        } else if (std.mem.eql(u8, arg, "--min-best-f1")) {
            opts.min_best_f1 = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinBestF1);
        } else if (std.mem.eql(u8, arg, "--min-mean-positive-probability-gap")) {
            opts.min_mean_positive_probability_gap = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinMeanPositiveProbabilityGap);
        } else if (std.mem.eql(u8, arg, "--max-avg-step-ms")) {
            opts.max_avg_step_ms = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMaxAvgStepMs);
        } else if (std.mem.eql(u8, arg, "--max-peak-rss-gb")) {
            opts.max_peak_rss_bytes = try parseMemoryGigabytesToBytes(args.next() orelse return error.MissingMaxPeakRssGb);
        } else if (std.mem.eql(u8, arg, "--max-vjp-fallbacks")) {
            opts.max_vjp_fallbacks = try std.fmt.parseUnsigned(u64, args.next() orelse return error.MissingMaxVjpFallbacks, 10);
        } else if (std.mem.eql(u8, arg, "--require-encoder-neftune")) {
            opts.require_encoder_neftune = try parseBoolFlag(args.next() orelse return error.MissingRequireEncoderNeftune);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            usage();
            return null;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }
    opts.out_dir = out_dir orelse {
        usage();
        return error.MissingOutDir;
    };
    return opts;
}

fn validate(allocator: std.mem.Allocator, opts: Options) !void {
    const default_manifest_path = try std.fs.path.join(allocator, &.{ opts.out_dir, manifest_file_name });
    defer allocator.free(default_manifest_path);
    const default_metrics_path = try std.fs.path.join(allocator, &.{ opts.out_dir, metrics_file_name });
    defer allocator.free(default_metrics_path);
    const manifest_path = opts.manifest_path orelse default_manifest_path;
    const metrics_path = opts.metrics_path orelse default_metrics_path;

    const manifest = try inspectManifest(allocator, manifest_path);
    defer manifest.deinit(allocator);
    if (opts.require_complete and !std.mem.eql(u8, manifest.status, "complete")) return error.TrainingNotComplete;
    if (opts.require_backend) |backend| {
        if (!std.mem.eql(u8, manifest.backend, backend)) return error.TrainingBackendMismatch;
    }
    if (opts.require_encoder_neftune) |expected| {
        const actual = manifest.encoder_neftune orelse return error.MissingEncoderNeftuneManifestField;
        if (actual != expected) return error.EncoderNeftuneMismatch;
    }
    if (manifest.final_checkpoint_path) |path| {
        _ = try compat.cwd().statFile(compat.io(), path, .{});
    }

    const metrics = try inspectMetrics(allocator, metrics_path);
    if (metrics.step_count < opts.min_steps) return error.StepCountBelowThreshold;
    if (!metrics.all_losses_finite) return error.NonFiniteLoss;
    if (metrics.max_vjp_fallbacks > opts.max_vjp_fallbacks) return error.VjpFallbacksAboveThreshold;
    if (opts.require_mpsgraph_vjp) {
        if (metrics.mpsgraph_step_count == 0) return error.MissingMpsGraphVjp;
        if (metrics.non_mpsgraph_vjp_step_count > 0) return error.NonMpsGraphVjpRuntime;
    }
    if (opts.min_fixed_f1) |min_f1| {
        if (metrics.best_fixed_f1 < min_f1 and manifest.best_val_f1 < min_f1) return error.FixedF1BelowThreshold;
    }
    if (opts.min_best_f1) |min_f1| {
        if (metrics.best_threshold_f1 < min_f1) return error.BestF1BelowThreshold;
    }
    if (opts.min_mean_positive_probability_gap) |min_gap| {
        if (metrics.probability_gap_count == 0) return error.MissingBoundaryProbabilityDiagnostics;
        if (metrics.min_mean_positive_probability_gap < min_gap) return error.BoundaryProbabilityGapBelowThreshold;
    }
    if (opts.max_avg_step_ms) |max_ms| {
        if (metrics.avgStepMs() > max_ms) return error.AvgStepMsAboveThreshold;
    }
    if (opts.max_peak_rss_bytes) |max_bytes| {
        const peak = @max(metrics.max_peak_resident_bytes, manifest.peak_resident_bytes);
        if (peak > max_bytes) return error.PeakResidentBytesAboveThreshold;
    }

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(compat.io(), &stdout_buf);
    try std.json.Stringify.value(.{
        .status = "passed",
        .manifest = manifest_path,
        .metrics = metrics_path,
        .training_status = manifest.status,
        .backend = manifest.backend,
        .encoder_neftune = manifest.encoder_neftune,
        .steps = metrics.step_count,
        .avg_step_ms = metrics.avgStepMs(),
        .max_peak_resident_bytes = @max(metrics.max_peak_resident_bytes, manifest.peak_resident_bytes),
        .max_vjp_fallbacks = metrics.max_vjp_fallbacks,
        .mpsgraph_vjp_steps = metrics.mpsgraph_step_count,
        .best_fixed_f1 = metrics.best_fixed_f1,
        .best_threshold_f1 = metrics.best_threshold_f1,
        .min_mean_positive_probability_gap = if (metrics.probability_gap_count > 0) metrics.min_mean_positive_probability_gap else null,
        .first_loss = metrics.first_loss,
        .final_loss = metrics.final_loss,
    }, .{ .whitespace = .indent_2 }, &stdout.interface);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}

fn inspectManifest(allocator: std.mem.Allocator, path: []const u8) !ManifestInspection {
    const bytes = try compat.cwd().readFileAlloc(compat.io(), path, allocator, .limited(8 * 1024 * 1024));
    defer allocator.free(bytes);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidTrainingManifest;
    const obj = parsed.value.object;
    const schema = jsonString(obj.get("schema_version")) orelse return error.InvalidTrainingManifest;
    if (!std.mem.eql(u8, schema, expected_schema_version)) return error.InvalidTrainingManifest;
    const family = jsonString(obj.get("artifact_family_version")) orelse return error.InvalidTrainingManifest;
    if (!std.mem.eql(u8, family, expected_artifact_family_version)) return error.InvalidTrainingManifest;
    const status = try allocator.dupe(u8, jsonString(obj.get("status")) orelse return error.InvalidTrainingManifest);
    errdefer allocator.free(status);
    const backend = try allocator.dupe(u8, jsonString(obj.get("backend")) orelse return error.InvalidTrainingManifest);
    errdefer allocator.free(backend);
    const checkpoint_path = if (jsonString(obj.get("final_checkpoint_path"))) |checkpoint|
        try allocator.dupe(u8, checkpoint)
    else
        null;
    errdefer if (checkpoint_path) |checkpoint| allocator.free(checkpoint);
    return .{
        .status = status,
        .backend = backend,
        .total_steps = jsonUsize(obj.get("total_steps")) orelse 0,
        .peak_resident_bytes = jsonUsize(obj.get("peak_resident_bytes")) orelse 0,
        .final_checkpoint_path = checkpoint_path,
        .best_val_f1 = jsonF64(obj.get("best_val_f1")) orelse 0,
        .encoder_neftune = jsonBool(obj.get("encoder_neftune")),
    };
}

fn inspectMetrics(allocator: std.mem.Allocator, path: []const u8) !MetricsInspection {
    const bytes = try compat.cwd().readFileAlloc(compat.io(), path, allocator, .limited(128 * 1024 * 1024));
    defer allocator.free(bytes);
    var out = MetricsInspection{};
    var lines = std.mem.splitScalar(u8, bytes, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r\n");
        if (line.len == 0) continue;
        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, line, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidMetricsRecord;
        const obj = parsed.value.object;
        out.record_count += 1;
        const event = jsonString(obj.get("event")) orelse return error.InvalidMetricsRecord;
        if (std.mem.eql(u8, event, "step")) {
            out.step_count += 1;
            const loss = jsonF64(obj.get("loss")) orelse return error.InvalidMetricsRecord;
            if (!std.math.isFinite(loss)) out.all_losses_finite = false;
            if (out.first_loss == null) out.first_loss = loss;
            out.final_loss = loss;
            const step_ms = jsonF64(obj.get("step_wall_ms")) orelse return error.InvalidMetricsRecord;
            if (!std.math.isFinite(step_ms) or step_ms <= 0) return error.InvalidPerformanceMetrics;
            out.total_step_ms += step_ms;
            out.max_peak_resident_bytes = @max(out.max_peak_resident_bytes, jsonUsize(obj.get("peak_resident_bytes")) orelse 0);
            out.max_vjp_fallbacks = @max(out.max_vjp_fallbacks, jsonU64(obj.get("vjp_interpreter_fallbacks")) orelse 0);
            const runtime = jsonString(obj.get("vjp_runtime")) orelse "none";
            if (std.mem.eql(u8, runtime, "mpsgraph")) out.mpsgraph_step_count += 1 else if (!std.mem.eql(u8, runtime, "none")) out.non_mpsgraph_vjp_step_count += 1;
        } else if (std.mem.startsWith(u8, event, "validation_")) {
            out.validation_count += 1;
            out.best_fixed_f1 = @max(out.best_fixed_f1, jsonF64(obj.get("f1")) orelse 0);
            out.best_threshold_f1 = @max(out.best_threshold_f1, jsonF64(obj.get("best_f1")) orelse 0);
            if (jsonF64(obj.get("mean_positive_probability_gold_positive"))) |pos| {
                if (jsonF64(obj.get("mean_positive_probability_gold_negative"))) |neg| {
                    if (std.math.isFinite(pos) and std.math.isFinite(neg)) {
                        out.probability_gap_count += 1;
                        out.min_mean_positive_probability_gap = @min(out.min_mean_positive_probability_gap, pos - neg);
                    }
                }
            }
        }
    }
    return out;
}

fn parseMemoryGigabytesToBytes(raw: []const u8) !usize {
    const value = try std.fmt.parseFloat(f64, raw);
    if (!std.math.isFinite(value) or value < 0) return error.InvalidMemoryThreshold;
    return @intFromFloat(value * 1024.0 * 1024.0 * 1024.0);
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

fn parseBoolFlag(raw: []const u8) !bool {
    if (std.mem.eql(u8, raw, "1") or
        std.mem.eql(u8, raw, "true") or
        std.mem.eql(u8, raw, "TRUE") or
        std.mem.eql(u8, raw, "yes") or
        std.mem.eql(u8, raw, "YES"))
    {
        return true;
    }
    if (std.mem.eql(u8, raw, "0") or
        std.mem.eql(u8, raw, "false") or
        std.mem.eql(u8, raw, "FALSE") or
        std.mem.eql(u8, raw, "no") or
        std.mem.eql(u8, raw, "NO"))
    {
        return false;
    }
    return error.InvalidBoolFlag;
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

fn jsonU64(value: ?std.json.Value) ?u64 {
    const v = value orelse return null;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        else => null,
    };
}

fn usage() void {
    std.debug.print(
        \\usage: validate-fused-chunker-run --out-dir <dir> [options]
        \\
        \\  --manifest <path>          Manifest path (default: <out-dir>/fused_training_manifest.json)
        \\  --metrics <path>           Metrics JSONL path (default: <out-dir>/fused_training_metrics.jsonl)
        \\  --require-complete         Require manifest status=complete
        \\  --require-backend <name>    Require backend name, for example metal
        \\  --require-mpsgraph-vjp     Require at least one MPSGraph VJP step and no non-MPSGraph VJP runtime
        \\  --require-encoder-neftune <bool>
        \\                             Require manifest encoder_neftune to match true/false
        \\  --min-steps <n>            Minimum step records (default: 1)
        \\  --min-fixed-f1 <f>         Minimum fixed-threshold validation F1
        \\  --min-best-f1 <f>          Minimum best-threshold validation F1
        \\  --min-mean-positive-probability-gap <f>
        \\                             Minimum mean P(boundary) gap for gold-positive minus gold-negative labels
        \\  --max-avg-step-ms <f>      Maximum average step wall time
        \\  --max-peak-rss-gb <f>      Maximum peak RSS in GiB
        \\  --max-vjp-fallbacks <n>    Maximum VJP interpreter fallbacks (default: 0)
        \\
    , .{});
}
