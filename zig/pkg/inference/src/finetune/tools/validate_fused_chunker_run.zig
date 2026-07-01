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
    min_validations: usize = 0,
    min_fixed_f1: ?f64 = null,
    min_best_f1: ?f64 = null,
    min_calibrated_f1: ?f64 = null,
    min_average_precision: ?f64 = null,
    min_max_rank_f1: ?f64 = null,
    min_mean_positive_probability_gap: ?f64 = null,
    min_last_fixed_f1: ?f64 = null,
    min_last_best_f1: ?f64 = null,
    min_last_calibrated_f1: ?f64 = null,
    min_last_average_precision: ?f64 = null,
    min_last_max_rank_f1: ?f64 = null,
    min_last_mean_positive_probability_gap: ?f64 = null,
    checkpoint_eval_results_path: ?[]const u8 = null,
    min_checkpoint_fixed_f1: ?f64 = null,
    min_checkpoint_best_f1: ?f64 = null,
    min_checkpoint_average_precision: ?f64 = null,
    min_checkpoint_max_rank_f1: ?f64 = null,
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
    recommended_boundary_threshold_available: bool = false,
    recommended_boundary_threshold: ?f64 = null,
    recommended_boundary_threshold_f1: ?f64 = null,
    recommended_boundary_threshold_source: ?[]const u8 = null,
    recommended_boundary_threshold_full_validation: ?bool = null,

    fn deinit(self: ManifestInspection, allocator: std.mem.Allocator) void {
        allocator.free(self.status);
        allocator.free(self.backend);
        if (self.final_checkpoint_path) |path| allocator.free(path);
        if (self.recommended_boundary_threshold_source) |source| allocator.free(source);
    }
};

const CheckpointEvalInspection = struct {
    fixed_threshold_f1: f64 = 0,
    best_threshold_f1: f64 = 0,
    average_precision: f64 = 0,
    precision_at_gold_count: f64 = 0,
    max_rank_f1: f64 = 0,
    retrieval_ndcg_at_10: ?f64 = null,
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
    best_calibrated_f1: f64 = 0,
    best_calibrated_threshold: ?f64 = null,
    best_average_precision: f64 = 0,
    best_max_rank_f1: f64 = 0,
    last_fixed_f1: f64 = 0,
    last_threshold_f1: f64 = 0,
    last_calibrated_f1: f64 = 0,
    last_calibrated_threshold: ?f64 = null,
    last_calibrated_available: bool = false,
    last_average_precision: f64 = 0,
    last_max_rank_f1: f64 = 0,
    last_mean_positive_probability_gap: ?f64 = null,
    probability_gap_count: usize = 0,
    calibrated_f1_count: usize = 0,
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
        } else if (std.mem.eql(u8, arg, "--min-validations")) {
            opts.min_validations = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingMinValidations, 10);
        } else if (std.mem.eql(u8, arg, "--min-fixed-f1")) {
            opts.min_fixed_f1 = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinFixedF1);
        } else if (std.mem.eql(u8, arg, "--min-best-f1")) {
            opts.min_best_f1 = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinBestF1);
        } else if (std.mem.eql(u8, arg, "--min-calibrated-f1")) {
            opts.min_calibrated_f1 = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinCalibratedF1);
        } else if (std.mem.eql(u8, arg, "--min-average-precision")) {
            opts.min_average_precision = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinAveragePrecision);
        } else if (std.mem.eql(u8, arg, "--min-max-rank-f1")) {
            opts.min_max_rank_f1 = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinMaxRankF1);
        } else if (std.mem.eql(u8, arg, "--min-mean-positive-probability-gap")) {
            opts.min_mean_positive_probability_gap = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinMeanPositiveProbabilityGap);
        } else if (std.mem.eql(u8, arg, "--min-last-fixed-f1")) {
            opts.min_last_fixed_f1 = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinLastFixedF1);
        } else if (std.mem.eql(u8, arg, "--min-last-best-f1")) {
            opts.min_last_best_f1 = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinLastBestF1);
        } else if (std.mem.eql(u8, arg, "--min-last-calibrated-f1")) {
            opts.min_last_calibrated_f1 = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinLastCalibratedF1);
        } else if (std.mem.eql(u8, arg, "--min-last-average-precision")) {
            opts.min_last_average_precision = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinLastAveragePrecision);
        } else if (std.mem.eql(u8, arg, "--min-last-max-rank-f1")) {
            opts.min_last_max_rank_f1 = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinLastMaxRankF1);
        } else if (std.mem.eql(u8, arg, "--min-last-mean-positive-probability-gap")) {
            opts.min_last_mean_positive_probability_gap = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinLastMeanPositiveProbabilityGap);
        } else if (std.mem.eql(u8, arg, "--checkpoint-eval-results")) {
            opts.checkpoint_eval_results_path = args.next() orelse return error.MissingCheckpointEvalResults;
        } else if (std.mem.eql(u8, arg, "--min-checkpoint-fixed-f1")) {
            opts.min_checkpoint_fixed_f1 = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinCheckpointFixedF1);
        } else if (std.mem.eql(u8, arg, "--min-checkpoint-best-f1")) {
            opts.min_checkpoint_best_f1 = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinCheckpointBestF1);
        } else if (std.mem.eql(u8, arg, "--min-checkpoint-average-precision")) {
            opts.min_checkpoint_average_precision = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinCheckpointAveragePrecision);
        } else if (std.mem.eql(u8, arg, "--min-checkpoint-max-rank-f1")) {
            opts.min_checkpoint_max_rank_f1 = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingMinCheckpointMaxRankF1);
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
    try validateMinimumValidationCount(metrics, opts.min_validations);
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
    if (opts.min_calibrated_f1) |min_f1| {
        if (metrics.calibrated_f1_count == 0) return error.MissingCalibratedBoundaryMetrics;
        if (metrics.best_calibrated_f1 < min_f1) return error.CalibratedF1BelowThreshold;
    }
    if (opts.min_average_precision) |min_ap| {
        if (metrics.best_average_precision < min_ap) return error.AveragePrecisionBelowThreshold;
    }
    if (opts.min_max_rank_f1) |min_rank_f1| {
        if (metrics.best_max_rank_f1 < min_rank_f1) return error.MaxRankF1BelowThreshold;
    }
    if (opts.min_mean_positive_probability_gap) |min_gap| {
        if (metrics.probability_gap_count == 0) return error.MissingBoundaryProbabilityDiagnostics;
        if (metrics.min_mean_positive_probability_gap < min_gap) return error.BoundaryProbabilityGapBelowThreshold;
    }
    try validateLastValidationThresholds(metrics, opts);
    if (opts.max_avg_step_ms) |max_ms| {
        if (metrics.avgStepMs() > max_ms) return error.AvgStepMsAboveThreshold;
    }
    if (opts.max_peak_rss_bytes) |max_bytes| {
        const peak = @max(metrics.max_peak_resident_bytes, manifest.peak_resident_bytes);
        if (peak > max_bytes) return error.PeakResidentBytesAboveThreshold;
    }

    const checkpoint_eval = if (opts.checkpoint_eval_results_path) |path|
        try inspectCheckpointEvalResults(allocator, path)
    else
        null;
    if (opts.min_checkpoint_fixed_f1 != null or
        opts.min_checkpoint_best_f1 != null or
        opts.min_checkpoint_average_precision != null or
        opts.min_checkpoint_max_rank_f1 != null)
    {
        const eval = checkpoint_eval orelse return error.MissingCheckpointEvalResults;
        if (opts.min_checkpoint_fixed_f1) |min_f1| {
            if (eval.fixed_threshold_f1 < min_f1) return error.CheckpointFixedF1BelowThreshold;
        }
        if (opts.min_checkpoint_best_f1) |min_f1| {
            if (eval.best_threshold_f1 < min_f1) return error.CheckpointBestF1BelowThreshold;
        }
        if (opts.min_checkpoint_average_precision) |min_ap| {
            if (eval.average_precision < min_ap) return error.CheckpointAveragePrecisionBelowThreshold;
        }
        if (opts.min_checkpoint_max_rank_f1) |min_rank_f1| {
            if (eval.max_rank_f1 < min_rank_f1) return error.CheckpointMaxRankF1BelowThreshold;
        }
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
        .recommended_boundary_threshold_available = manifest.recommended_boundary_threshold_available,
        .recommended_boundary_threshold = if (manifest.recommended_boundary_threshold_available) manifest.recommended_boundary_threshold else null,
        .recommended_boundary_threshold_f1 = if (manifest.recommended_boundary_threshold_available) manifest.recommended_boundary_threshold_f1 else null,
        .recommended_boundary_threshold_source = manifest.recommended_boundary_threshold_source,
        .recommended_boundary_threshold_full_validation = manifest.recommended_boundary_threshold_full_validation,
        .steps = metrics.step_count,
        .validation_count = metrics.validation_count,
        .avg_step_ms = metrics.avgStepMs(),
        .max_peak_resident_bytes = @max(metrics.max_peak_resident_bytes, manifest.peak_resident_bytes),
        .max_vjp_fallbacks = metrics.max_vjp_fallbacks,
        .mpsgraph_vjp_steps = metrics.mpsgraph_step_count,
        .best_fixed_f1 = metrics.best_fixed_f1,
        .best_threshold_f1 = metrics.best_threshold_f1,
        .best_calibrated_f1 = if (metrics.calibrated_f1_count > 0) metrics.best_calibrated_f1 else null,
        .best_calibrated_threshold = metrics.best_calibrated_threshold,
        .best_average_precision = metrics.best_average_precision,
        .best_max_rank_f1 = metrics.best_max_rank_f1,
        .last_fixed_f1 = metrics.last_fixed_f1,
        .last_best_threshold_f1 = metrics.last_threshold_f1,
        .last_calibrated_f1 = if (metrics.last_calibrated_available) metrics.last_calibrated_f1 else null,
        .last_calibrated_threshold = metrics.last_calibrated_threshold,
        .last_calibrated_available = metrics.last_calibrated_available,
        .last_average_precision = metrics.last_average_precision,
        .last_max_rank_f1 = metrics.last_max_rank_f1,
        .min_mean_positive_probability_gap = if (metrics.probability_gap_count > 0) metrics.min_mean_positive_probability_gap else null,
        .last_mean_positive_probability_gap = metrics.last_mean_positive_probability_gap,
        .checkpoint_eval = checkpoint_eval,
        .first_loss = metrics.first_loss,
        .final_loss = metrics.final_loss,
    }, .{ .whitespace = .indent_2 }, &stdout.interface);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}

fn inspectManifest(allocator: std.mem.Allocator, path: []const u8) !ManifestInspection {
    const bytes = try compat.cwd().readFileAlloc(compat.io(), path, allocator, .limited(8 * 1024 * 1024));
    defer allocator.free(bytes);
    return inspectManifestJson(allocator, bytes);
}

fn inspectManifestJson(allocator: std.mem.Allocator, bytes: []const u8) !ManifestInspection {
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
    const recommended_threshold_available = jsonBool(obj.get("recommended_boundary_threshold_available")) orelse false;
    const recommended_threshold_source = if (jsonString(obj.get("recommended_boundary_threshold_source"))) |source|
        try allocator.dupe(u8, source)
    else
        null;
    errdefer if (recommended_threshold_source) |source| allocator.free(source);
    return .{
        .status = status,
        .backend = backend,
        .total_steps = jsonUsize(obj.get("total_steps")) orelse 0,
        .peak_resident_bytes = jsonUsize(obj.get("peak_resident_bytes")) orelse 0,
        .final_checkpoint_path = checkpoint_path,
        .best_val_f1 = jsonF64(obj.get("best_val_f1")) orelse 0,
        .encoder_neftune = jsonBool(obj.get("encoder_neftune")),
        .recommended_boundary_threshold_available = recommended_threshold_available,
        .recommended_boundary_threshold = jsonF64(obj.get("recommended_boundary_threshold")),
        .recommended_boundary_threshold_f1 = jsonF64(obj.get("recommended_boundary_threshold_f1")),
        .recommended_boundary_threshold_source = recommended_threshold_source,
        .recommended_boundary_threshold_full_validation = jsonBool(obj.get("recommended_boundary_threshold_full_validation")),
    };
}

fn inspectMetrics(allocator: std.mem.Allocator, path: []const u8) !MetricsInspection {
    const bytes = try compat.cwd().readFileAlloc(compat.io(), path, allocator, .limited(128 * 1024 * 1024));
    defer allocator.free(bytes);
    return inspectMetricsJsonl(allocator, bytes);
}

fn inspectMetricsJsonl(allocator: std.mem.Allocator, bytes: []const u8) !MetricsInspection {
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
            const fixed_f1 = jsonF64(obj.get("f1")) orelse 0;
            const best_f1 = jsonF64(obj.get("best_f1")) orelse 0;
            const calibrated_available = jsonBool(obj.get("calibrated_threshold_available")) orelse false;
            const calibrated_f1 = if (calibrated_available)
                finiteJsonF64(obj.get("calibrated_boundary_f1"))
            else
                null;
            const calibrated_threshold = if (calibrated_available)
                finiteJsonF64(obj.get("calibrated_boundary_threshold"))
            else
                null;
            const average_precision = jsonF64(obj.get("average_precision")) orelse 0;
            const max_rank_f1 = jsonF64(obj.get("max_rank_f1")) orelse 0;
            out.best_fixed_f1 = @max(out.best_fixed_f1, fixed_f1);
            out.best_threshold_f1 = @max(out.best_threshold_f1, best_f1);
            if (calibrated_f1) |f1| {
                const first_calibrated = out.calibrated_f1_count == 0;
                out.calibrated_f1_count += 1;
                out.last_calibrated_available = true;
                if (first_calibrated or f1 > out.best_calibrated_f1) {
                    out.best_calibrated_f1 = f1;
                    out.best_calibrated_threshold = calibrated_threshold;
                }
                out.last_calibrated_f1 = f1;
                out.last_calibrated_threshold = calibrated_threshold;
            } else {
                out.last_calibrated_available = false;
                out.last_calibrated_f1 = 0;
                out.last_calibrated_threshold = null;
            }
            out.best_average_precision = @max(out.best_average_precision, average_precision);
            out.best_max_rank_f1 = @max(out.best_max_rank_f1, max_rank_f1);
            out.last_fixed_f1 = fixed_f1;
            out.last_threshold_f1 = best_f1;
            out.last_average_precision = average_precision;
            out.last_max_rank_f1 = max_rank_f1;
            out.last_mean_positive_probability_gap = null;
            if (jsonF64(obj.get("mean_positive_probability_gold_positive"))) |pos| {
                if (jsonF64(obj.get("mean_positive_probability_gold_negative"))) |neg| {
                    if (std.math.isFinite(pos) and std.math.isFinite(neg)) {
                        const gap = pos - neg;
                        out.probability_gap_count += 1;
                        out.min_mean_positive_probability_gap = @min(out.min_mean_positive_probability_gap, gap);
                        out.last_mean_positive_probability_gap = gap;
                    }
                }
            }
        }
    }
    return out;
}

fn inspectCheckpointEvalResults(allocator: std.mem.Allocator, path: []const u8) !CheckpointEvalInspection {
    const bytes = try compat.cwd().readFileAlloc(compat.io(), path, allocator, .limited(64 * 1024 * 1024));
    defer allocator.free(bytes);
    return inspectCheckpointEvalResultsJson(allocator, bytes);
}

fn inspectCheckpointEvalResultsJson(allocator: std.mem.Allocator, bytes: []const u8) !CheckpointEvalInspection {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidCheckpointEvalResults;
    const obj = parsed.value.object;
    const schema = jsonString(obj.get("schema_version")) orelse return error.InvalidCheckpointEvalResults;
    if (!std.mem.eql(u8, schema, "fused_chunker_benchmark_results/v1")) return error.InvalidCheckpointEvalResults;

    const boundary = try jsonObject(obj.get("boundary_f1") orelse return error.InvalidCheckpointEvalResults);
    const retrieval = if (obj.get("retrieval_ndcg")) |retrieval_value| blk: {
        if (retrieval_value == .null) break :blk null;
        break :blk try jsonObject(retrieval_value);
    } else null;

    const out = CheckpointEvalInspection{
        .fixed_threshold_f1 = finiteJsonF64(boundary.get("fixed_threshold_f1")) orelse return error.InvalidCheckpointEvalResults,
        .best_threshold_f1 = finiteJsonF64(boundary.get("best_threshold_f1")) orelse return error.InvalidCheckpointEvalResults,
        .average_precision = finiteJsonF64(boundary.get("average_precision")) orelse return error.InvalidCheckpointEvalResults,
        .precision_at_gold_count = finiteJsonF64(boundary.get("precision_at_gold_count")) orelse 0,
        .max_rank_f1 = finiteJsonF64(boundary.get("max_rank_f1")) orelse return error.InvalidCheckpointEvalResults,
        .retrieval_ndcg_at_10 = if (retrieval) |retrieval_obj|
            finiteJsonF64(retrieval_obj.get("overall_ndcg_at_10"))
        else
            null,
    };
    return out;
}

fn validateMinimumValidationCount(metrics: MetricsInspection, min_validations: usize) !void {
    if (metrics.validation_count < min_validations) return error.ValidationCountBelowThreshold;
}

fn validateLastValidationThresholds(metrics: MetricsInspection, opts: Options) !void {
    const requires_last_validation =
        opts.min_last_fixed_f1 != null or
        opts.min_last_best_f1 != null or
        opts.min_last_calibrated_f1 != null or
        opts.min_last_average_precision != null or
        opts.min_last_max_rank_f1 != null or
        opts.min_last_mean_positive_probability_gap != null;
    if (!requires_last_validation) return;
    if (metrics.validation_count == 0) return error.MissingLastValidationMetrics;
    if (opts.min_last_fixed_f1) |min_f1| {
        if (metrics.last_fixed_f1 < min_f1) return error.LastFixedF1BelowThreshold;
    }
    if (opts.min_last_best_f1) |min_f1| {
        if (metrics.last_threshold_f1 < min_f1) return error.LastBestF1BelowThreshold;
    }
    if (opts.min_last_calibrated_f1) |min_f1| {
        if (!metrics.last_calibrated_available) return error.MissingLastCalibratedBoundaryMetrics;
        if (metrics.last_calibrated_f1 < min_f1) return error.LastCalibratedF1BelowThreshold;
    }
    if (opts.min_last_average_precision) |min_ap| {
        if (metrics.last_average_precision < min_ap) return error.LastAveragePrecisionBelowThreshold;
    }
    if (opts.min_last_max_rank_f1) |min_rank_f1| {
        if (metrics.last_max_rank_f1 < min_rank_f1) return error.LastMaxRankF1BelowThreshold;
    }
    if (opts.min_last_mean_positive_probability_gap) |min_gap| {
        const last_gap = metrics.last_mean_positive_probability_gap orelse return error.MissingLastBoundaryProbabilityDiagnostics;
        if (last_gap < min_gap) return error.LastBoundaryProbabilityGapBelowThreshold;
    }
}

test "manifest inspection captures recommended boundary threshold" {
    const manifest_json =
        \\{
        \\  "schema_version":"fused_chunker_training/v1",
        \\  "artifact_family_version":"fused_chunker_phase20/v1",
        \\  "status":"complete",
        \\  "backend":"metal",
        \\  "total_steps":100,
        \\  "peak_resident_bytes":1024,
        \\  "final_checkpoint_path":"/tmp/checkpoint_final.safetensors",
        \\  "best_val_f1":0.31,
        \\  "encoder_neftune":false,
        \\  "recommended_boundary_threshold_available":true,
        \\  "recommended_boundary_threshold":0.89,
        \\  "recommended_boundary_threshold_f1":0.47,
        \\  "recommended_boundary_threshold_source":"validation_epoch",
        \\  "recommended_boundary_threshold_full_validation":true
        \\}
    ;
    const manifest = try inspectManifestJson(std.testing.allocator, manifest_json);
    defer manifest.deinit(std.testing.allocator);

    try std.testing.expect(manifest.recommended_boundary_threshold_available);
    try std.testing.expectApproxEqAbs(@as(f64, 0.89), manifest.recommended_boundary_threshold.?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.47), manifest.recommended_boundary_threshold_f1.?, 1e-12);
    try std.testing.expect(std.mem.eql(u8, "validation_epoch", manifest.recommended_boundary_threshold_source.?));
    try std.testing.expectEqual(true, manifest.recommended_boundary_threshold_full_validation.?);
}

test "metrics inspection captures validation average precision and rank sweep" {
    const metrics_jsonl =
        \\{"event":"step","loss":1.0,"step_wall_ms":10.0,"peak_resident_bytes":1024,"vjp_interpreter_fallbacks":0,"vjp_runtime":"mpsgraph"}
        \\{"event":"validation_step","f1":0.12,"best_f1":0.30,"calibrated_threshold_available":true,"calibrated_boundary_threshold":0.83,"calibrated_boundary_f1":0.27,"average_precision":0.42,"max_rank_f1":0.37,"mean_positive_probability_gold_positive":0.61,"mean_positive_probability_gold_negative":0.49}
        \\{"event":"validation_step","f1":0.15,"best_f1":0.28,"calibrated_threshold_available":true,"calibrated_boundary_threshold":0.83,"calibrated_boundary_f1":0.26,"average_precision":0.41,"max_rank_f1":0.35,"mean_positive_probability_gold_positive":0.58,"mean_positive_probability_gold_negative":0.50}
        \\
    ;
    const metrics = try inspectMetricsJsonl(std.testing.allocator, metrics_jsonl);
    try std.testing.expectEqual(@as(usize, 1), metrics.step_count);
    try std.testing.expectEqual(@as(usize, 2), metrics.validation_count);
    try std.testing.expectEqual(@as(usize, 2), metrics.calibrated_f1_count);
    try std.testing.expectApproxEqAbs(@as(f64, 0.15), metrics.best_fixed_f1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.30), metrics.best_threshold_f1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.27), metrics.best_calibrated_f1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.83), metrics.best_calibrated_threshold.?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.42), metrics.best_average_precision, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.37), metrics.best_max_rank_f1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.15), metrics.last_fixed_f1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.28), metrics.last_threshold_f1, 1e-12);
    try std.testing.expect(metrics.last_calibrated_available);
    try std.testing.expectApproxEqAbs(@as(f64, 0.26), metrics.last_calibrated_f1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.83), metrics.last_calibrated_threshold.?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.41), metrics.last_average_precision, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.35), metrics.last_max_rank_f1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.08), metrics.min_mean_positive_probability_gap, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.08), metrics.last_mean_positive_probability_gap.?, 1e-12);
    try validateMinimumValidationCount(metrics, 2);
    try std.testing.expectError(error.ValidationCountBelowThreshold, validateMinimumValidationCount(metrics, 3));
}

test "last validation gates catch post-peak collapse" {
    const metrics_jsonl =
        \\{"event":"step","loss":1.0,"step_wall_ms":10.0,"peak_resident_bytes":1024,"vjp_interpreter_fallbacks":0,"vjp_runtime":"mpsgraph"}
        \\{"event":"validation_step","f1":0.80,"best_f1":0.82,"calibrated_threshold_available":true,"calibrated_boundary_f1":0.81,"calibrated_boundary_threshold":0.91,"average_precision":0.83,"max_rank_f1":0.84,"mean_positive_probability_gold_positive":0.62,"mean_positive_probability_gold_negative":0.50}
        \\{"event":"validation_step","f1":0.20,"best_f1":0.25,"calibrated_threshold_available":true,"calibrated_boundary_f1":0.21,"calibrated_boundary_threshold":0.91,"average_precision":0.22,"max_rank_f1":0.24,"mean_positive_probability_gold_positive":0.51,"mean_positive_probability_gold_negative":0.50}
        \\
    ;
    const metrics = try inspectMetricsJsonl(std.testing.allocator, metrics_jsonl);
    try std.testing.expectApproxEqAbs(@as(f64, 0.80), metrics.best_fixed_f1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.81), metrics.best_calibrated_f1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.20), metrics.last_fixed_f1, 1e-12);
    try validateLastValidationThresholds(metrics, .{
        .out_dir = "",
        .min_last_fixed_f1 = 0.19,
        .min_last_best_f1 = 0.24,
        .min_last_calibrated_f1 = 0.20,
        .min_last_average_precision = 0.21,
        .min_last_max_rank_f1 = 0.23,
        .min_last_mean_positive_probability_gap = 0.0,
    });
    try std.testing.expectError(error.LastFixedF1BelowThreshold, validateLastValidationThresholds(metrics, .{
        .out_dir = "",
        .min_last_fixed_f1 = 0.766,
    }));
    try std.testing.expectError(error.LastBoundaryProbabilityGapBelowThreshold, validateLastValidationThresholds(metrics, .{
        .out_dir = "",
        .min_last_mean_positive_probability_gap = 0.05,
    }));
    try std.testing.expectError(error.LastCalibratedF1BelowThreshold, validateLastValidationThresholds(metrics, .{
        .out_dir = "",
        .min_last_calibrated_f1 = 0.50,
    }));
}

test "calibrated validation gates require calibrated records" {
    const metrics_jsonl =
        \\{"event":"step","loss":1.0,"step_wall_ms":10.0,"peak_resident_bytes":1024,"vjp_interpreter_fallbacks":0,"vjp_runtime":"mpsgraph"}
        \\{"event":"validation_step","f1":0.20,"best_f1":0.25,"average_precision":0.22,"max_rank_f1":0.24,"mean_positive_probability_gold_positive":0.51,"mean_positive_probability_gold_negative":0.50}
        \\
    ;
    const metrics = try inspectMetricsJsonl(std.testing.allocator, metrics_jsonl);
    try std.testing.expectEqual(@as(usize, 0), metrics.calibrated_f1_count);
    try std.testing.expect(!metrics.last_calibrated_available);
    try std.testing.expectError(error.MissingLastCalibratedBoundaryMetrics, validateLastValidationThresholds(metrics, .{
        .out_dir = "",
        .min_last_calibrated_f1 = 0.10,
    }));
}

test "zero calibrated F1 still records calibrated threshold" {
    const metrics_jsonl =
        \\{"event":"step","loss":1.0,"step_wall_ms":10.0,"peak_resident_bytes":1024,"vjp_interpreter_fallbacks":0,"vjp_runtime":"mpsgraph"}
        \\{"event":"validation_step","f1":0.02,"best_f1":0.03,"calibrated_threshold_available":true,"calibrated_boundary_threshold":0.995,"calibrated_boundary_f1":0.0,"average_precision":0.01,"max_rank_f1":0.02}
        \\
    ;
    const metrics = try inspectMetricsJsonl(std.testing.allocator, metrics_jsonl);
    try std.testing.expectEqual(@as(usize, 1), metrics.calibrated_f1_count);
    try std.testing.expect(metrics.last_calibrated_available);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), metrics.best_calibrated_f1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.995), metrics.best_calibrated_threshold.?, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.995), metrics.last_calibrated_threshold.?, 1e-12);
}

test "checkpoint eval inspection requires persisted AP and rank metrics" {
    const eval_json =
        \\{
        \\  "schema_version":"fused_chunker_benchmark_results/v1",
        \\  "boundary_f1":{
        \\    "fixed_threshold_f1":0.21,
        \\    "best_threshold_f1":0.44,
        \\    "average_precision":0.55,
        \\    "precision_at_gold_count":0.62,
        \\    "max_rank_f1":0.49
        \\  },
        \\  "retrieval_ndcg":{"overall_ndcg_at_10":0.71}
        \\}
    ;
    const eval = try inspectCheckpointEvalResultsJson(std.testing.allocator, eval_json);
    try std.testing.expectApproxEqAbs(@as(f64, 0.21), eval.fixed_threshold_f1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.44), eval.best_threshold_f1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.55), eval.average_precision, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.62), eval.precision_at_gold_count, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.49), eval.max_rank_f1, 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 0.71), eval.retrieval_ndcg_at_10.?, 1e-12);

    const missing_ap_json =
        \\{
        \\  "schema_version":"fused_chunker_benchmark_results/v1",
        \\  "boundary_f1":{
        \\    "fixed_threshold_f1":0.21,
        \\    "best_threshold_f1":0.44,
        \\    "max_rank_f1":0.49
        \\  }
        \\}
    ;
    try std.testing.expectError(error.InvalidCheckpointEvalResults, inspectCheckpointEvalResultsJson(std.testing.allocator, missing_ap_json));
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

fn jsonObject(value: std.json.Value) !std.json.ObjectMap {
    return switch (value) {
        .object => |obj| obj,
        else => error.InvalidJsonObject,
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

fn finiteJsonF64(value: ?std.json.Value) ?f64 {
    const value_f = jsonF64(value) orelse return null;
    if (!std.math.isFinite(value_f)) return null;
    return value_f;
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
        \\  --min-validations <n>      Minimum validation records (default: 0)
        \\  --min-fixed-f1 <f>         Minimum fixed-threshold validation F1
        \\  --min-best-f1 <f>          Minimum best-threshold validation F1
        \\  --min-calibrated-f1 <f>    Minimum weighted-CE calibrated-threshold validation F1
        \\  --min-average-precision <f>
        \\                             Minimum validation average precision
        \\  --min-max-rank-f1 <f>      Minimum validation exact rank-sweep F1
        \\  --min-mean-positive-probability-gap <f>
        \\                             Minimum mean P(boundary) gap for gold-positive minus gold-negative labels
        \\  --min-last-fixed-f1 <f>    Minimum latest validation fixed-threshold F1
        \\  --min-last-best-f1 <f>     Minimum latest validation best-threshold F1
        \\  --min-last-calibrated-f1 <f>
        \\                             Minimum latest weighted-CE calibrated-threshold F1
        \\  --min-last-average-precision <f>
        \\                             Minimum latest validation average precision
        \\  --min-last-max-rank-f1 <f>
        \\                             Minimum latest validation exact rank-sweep F1
        \\  --min-last-mean-positive-probability-gap <f>
        \\                             Minimum latest validation gold-positive minus gold-negative probability gap
        \\  --checkpoint-eval-results <path>
        \\                             eval-fused-chunker --results-out JSON for the persisted checkpoint
        \\  --min-checkpoint-fixed-f1 <f>
        \\                             Minimum persisted-checkpoint fixed-threshold F1
        \\  --min-checkpoint-best-f1 <f>
        \\                             Minimum persisted-checkpoint best-threshold F1
        \\  --min-checkpoint-average-precision <f>
        \\                             Minimum persisted-checkpoint boundary average precision
        \\  --min-checkpoint-max-rank-f1 <f>
        \\                             Minimum persisted-checkpoint exact rank-sweep F1
        \\  --max-avg-step-ms <f>      Maximum average step wall time
        \\  --max-peak-rss-gb <f>      Maximum peak RSS in GiB
        \\  --max-vjp-fallbacks <n>    Maximum VJP interpreter fallbacks (default: 0)
        \\
    , .{});
}
