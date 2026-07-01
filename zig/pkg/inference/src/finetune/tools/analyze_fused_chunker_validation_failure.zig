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

const metrics_file_name = "fused_training_metrics.jsonl";
const analysis_schema_version = "fused_chunker_validation_failure_analysis/v1";
const release_boundary_f1_floor = 0.766;

const Options = struct {
    out_dir: ?[]const u8 = null,
    metrics_path: ?[]const u8 = null,
    analysis_out_path: ?[]const u8 = null,
};

const ValidationSnapshot = struct {
    step: ?usize = null,
    samples: ?usize = null,
    total_samples: ?usize = null,
    fixed_f1: f64 = 0,
    precision: f64 = 0,
    recall: f64 = 0,
    tp: u64 = 0,
    fp: u64 = 0,
    fn_count: u64 = 0,
    best_threshold_f1: f64 = 0,
    best_threshold: ?f64 = null,
    best_tp: u64 = 0,
    best_fp: u64 = 0,
    best_fn: u64 = 0,
    calibrated_threshold_available: bool = false,
    calibrated_boundary_threshold: ?f64 = null,
    calibrated_boundary_f1: ?f64 = null,
    calibrated_boundary_precision: ?f64 = null,
    calibrated_boundary_recall: ?f64 = null,
    calibrated_boundary_tp: u64 = 0,
    calibrated_boundary_fp: u64 = 0,
    calibrated_boundary_fn: u64 = 0,
    calibrated_predicted_positive_rate: ?f64 = null,
    fitted_threshold_available: bool = false,
    fitted_boundary_threshold: ?f64 = null,
    fitted_boundary_f1: ?f64 = null,
    fitted_boundary_precision: ?f64 = null,
    fitted_boundary_recall: ?f64 = null,
    fitted_boundary_tp: u64 = 0,
    fitted_boundary_fp: u64 = 0,
    fitted_boundary_fn: u64 = 0,
    fitted_predicted_positive_rate: ?f64 = null,
    average_precision: f64 = 0,
    precision_at_gold_count: f64 = 0,
    max_rank_f1: f64 = 0,
    max_rank_precision: f64 = 0,
    max_rank_recall: f64 = 0,
    max_rank_threshold: ?f64 = null,
    gold_positive_mean_rank_percentile: ?f64 = null,
    gold_positive_median_rank_percentile: ?f64 = null,
    gold_positive_p90_rank_percentile: ?f64 = null,
    gold_positive_p99_rank_percentile: ?f64 = null,
    gold_positive_top_5x_recall: ?f64 = null,
    gold_positive_top_10x_recall: ?f64 = null,
    valid_tokens: ?usize = null,
    gold_positives: ?usize = null,
    gold_positive_rate: ?f64 = null,
    predicted_positive_rate: ?f64 = null,
    best_predicted_positive_rate: ?f64 = null,
    mean_positive_probability_gold_positive: ?f64 = null,
    mean_positive_probability_gold_negative: ?f64 = null,
    mean_boundary_margin_gold_positive: ?f64 = null,
    mean_boundary_margin_gold_negative: ?f64 = null,

    fn probabilityGap(self: ValidationSnapshot) ?f64 {
        const pos = self.mean_positive_probability_gold_positive orelse return null;
        const neg = self.mean_positive_probability_gold_negative orelse return null;
        return pos - neg;
    }
};

const FrozenFeatureProbeSnapshot = struct {
    train_examples: usize = 0,
    val_examples: usize = 0,
    train_offset: usize = 0,
    val_offset: usize = 0,
    epochs: u32 = 0,
    final_step: u32 = 0,
    final_boundary_loss: ?f64 = null,
    train_best_f1: f64 = 0,
    train_fixed_f1: f64 = 0,
    train_max_rank_f1: f64 = 0,
    train_average_precision: f64 = 0,
    train_probability_gap: ?f64 = null,
    val_best_f1: ?f64 = null,
    val_fixed_f1: ?f64 = null,
    val_max_rank_f1: ?f64 = null,
    val_average_precision: ?f64 = null,
    val_probability_gap: ?f64 = null,

    fn trainGeneralizesPoorly(self: FrozenFeatureProbeSnapshot) bool {
        const train_good = self.train_best_f1 >= 0.50 or self.train_average_precision >= 0.50 or self.train_max_rank_f1 >= 0.50;
        const val_best = self.val_best_f1 orelse 0.0;
        const val_ap = self.val_average_precision orelse 0.0;
        const val_rank = self.val_max_rank_f1 orelse 0.0;
        const val_poor = val_best < 0.15 and val_ap < 0.05 and val_rank < 0.15;
        return train_good and val_poor;
    }
};

const MetricsInspection = struct {
    record_count: usize = 0,
    step_count: usize = 0,
    validation_count: usize = 0,
    train_validation_count: usize = 0,
    first_loss: ?f64 = null,
    final_loss: ?f64 = null,
    first_boundary_loss: ?f64 = null,
    final_boundary_loss: ?f64 = null,
    min_boundary_loss: ?f64 = null,
    max_vjp_fallbacks: u64 = 0,
    mpsgraph_step_count: usize = 0,
    non_mpsgraph_vjp_step_count: usize = 0,
    latest_validation: ?ValidationSnapshot = null,
    latest_train_validation: ?ValidationSnapshot = null,
    best_fixed_validation: ?ValidationSnapshot = null,
    best_threshold_validation: ?ValidationSnapshot = null,
    best_average_precision_validation: ?ValidationSnapshot = null,
    best_max_rank_validation: ?ValidationSnapshot = null,
    frozen_feature_probe: ?FrozenFeatureProbeSnapshot = null,
};

const FailureModes = struct {
    no_validation_records: bool = false,
    fixed_threshold_underconfident: bool = false,
    best_threshold_extremely_low: bool = false,
    predicted_zero_at_fixed_threshold: bool = false,
    rank_near_chance: bool = false,
    probability_gap_inverted: bool = false,
    probability_gap_too_small: bool = false,
    gold_ranks_buried: bool = false,
    train_rank_good_validation_rank_poor: bool = false,
    train_and_validation_rank_poor: bool = false,
    overpredicting_fixed_threshold: bool = false,
    calibrated_threshold_underpredicting: bool = false,
    calibrated_threshold_worse_than_best: bool = false,
    fitted_threshold_underpredicting: bool = false,
    fitted_threshold_worse_than_best: bool = false,
    collapse_after_peak: bool = false,
    train_loss_down_but_validation_poor: bool = false,
    best_threshold_below_release_floor: bool = false,
    frozen_feature_train_memorizes_heldout_poor: bool = false,
    frozen_feature_heldout_probability_gap_too_small: bool = false,
};

const Analysis = struct {
    schema_version: []const u8 = analysis_schema_version,
    status: []const u8 = "analyzed",
    metrics: []const u8,
    step_count: usize,
    validation_count: usize,
    train_validation_count: usize,
    latest_validation: ?ValidationSnapshot,
    latest_train_validation: ?ValidationSnapshot,
    best_fixed_validation: ?ValidationSnapshot,
    best_threshold_validation: ?ValidationSnapshot,
    best_average_precision_validation: ?ValidationSnapshot,
    best_max_rank_validation: ?ValidationSnapshot,
    frozen_feature_probe: ?FrozenFeatureProbeSnapshot,
    first_loss: ?f64,
    final_loss: ?f64,
    first_boundary_loss: ?f64,
    final_boundary_loss: ?f64,
    min_boundary_loss: ?f64,
    max_vjp_fallbacks: u64,
    mpsgraph_step_count: usize,
    non_mpsgraph_vjp_step_count: usize,
    latest_probability_gap: ?f64,
    latest_train_probability_gap: ?f64,
    latest_train_best_f1_minus_validation_best_f1: ?f64,
    latest_train_calibrated_f1_minus_validation_calibrated_f1: ?f64,
    latest_train_best_f1_minus_validation_fitted_f1: ?f64,
    latest_fitted_f1_minus_best_f1: ?f64,
    latest_train_ap_minus_validation_ap: ?f64,
    latest_train_max_rank_f1_minus_validation_max_rank_f1: ?f64,
    frozen_feature_train_best_f1_minus_val_best_f1: ?f64,
    frozen_feature_train_ap_minus_val_ap: ?f64,
    release_boundary_f1_floor: f64 = release_boundary_f1_floor,
    latest_best_f1_gap_to_release_floor: ?f64,
    failure_modes: FailureModes,
    stop_long_training: bool,
    next_target: []const u8,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const opts = try parseOptions(&args) orelse return;
    const metrics_path = try resolveMetricsPath(allocator, opts);
    defer allocator.free(metrics_path);

    const analysis = try analyzeMetricsFile(allocator, metrics_path);
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
        if (std.mem.eql(u8, arg, "--out-dir")) {
            opts.out_dir = args.next() orelse return error.MissingOutDir;
        } else if (std.mem.eql(u8, arg, "--metrics")) {
            opts.metrics_path = args.next() orelse return error.MissingMetrics;
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
    if (opts.metrics_path == null and opts.out_dir == null) {
        usage();
        return error.MissingMetrics;
    }
    return opts;
}

fn resolveMetricsPath(allocator: std.mem.Allocator, opts: Options) ![]const u8 {
    if (opts.metrics_path) |path| return allocator.dupe(u8, path);
    return std.fs.path.join(allocator, &.{ opts.out_dir.?, metrics_file_name });
}

fn analyzeMetricsFile(allocator: std.mem.Allocator, metrics_path: []const u8) !Analysis {
    const bytes = try compat.cwd().readFileAlloc(compat.io(), metrics_path, allocator, .limited(128 * 1024 * 1024));
    defer allocator.free(bytes);
    return analyzeMetricsJsonl(allocator, metrics_path, bytes);
}

fn analyzeMetricsJsonl(allocator: std.mem.Allocator, metrics_path: []const u8, bytes: []const u8) !Analysis {
    const metrics = try inspectMetricsJsonl(allocator, bytes);
    const modes = classifyFailure(metrics);
    const latest = metrics.latest_validation;
    const latest_gap = if (latest) |validation| validation.probabilityGap() else null;
    const latest_train = metrics.latest_train_validation;
    const latest_train_gap = if (latest_train) |validation| validation.probabilityGap() else null;
    const train_best_f1_delta = if (latest_train != null and latest != null)
        latest_train.?.best_threshold_f1 - latest.?.best_threshold_f1
    else
        null;
    const train_calibrated_delta = if (latest_train != null and latest != null and latest_train.?.calibrated_boundary_f1 != null and latest.?.calibrated_boundary_f1 != null)
        latest_train.?.calibrated_boundary_f1.? - latest.?.calibrated_boundary_f1.?
    else
        null;
    const train_best_vs_fitted_delta = if (latest_train != null and latest != null and latest.?.fitted_boundary_f1 != null)
        latest_train.?.best_threshold_f1 - latest.?.fitted_boundary_f1.?
    else
        null;
    const fitted_vs_best_delta = if (latest != null and latest.?.fitted_boundary_f1 != null)
        latest.?.fitted_boundary_f1.? - latest.?.best_threshold_f1
    else
        null;
    const train_ap_delta = if (latest_train != null and latest != null)
        latest_train.?.average_precision - latest.?.average_precision
    else
        null;
    const train_max_rank_delta = if (latest_train != null and latest != null)
        latest_train.?.max_rank_f1 - latest.?.max_rank_f1
    else
        null;
    const frozen_feature_best_delta = if (metrics.frozen_feature_probe) |probe|
        if (probe.val_best_f1) |val_best| probe.train_best_f1 - val_best else null
    else
        null;
    const frozen_feature_ap_delta = if (metrics.frozen_feature_probe) |probe|
        if (probe.val_average_precision) |val_ap| probe.train_average_precision - val_ap else null
    else
        null;
    const latest_best_gap = if (latest) |validation|
        release_boundary_f1_floor - validation.best_threshold_f1
    else
        null;
    const stop_long_training = shouldStopLongTraining(modes);
    return .{
        .metrics = metrics_path,
        .step_count = metrics.step_count,
        .validation_count = metrics.validation_count,
        .train_validation_count = metrics.train_validation_count,
        .latest_validation = metrics.latest_validation,
        .latest_train_validation = metrics.latest_train_validation,
        .best_fixed_validation = metrics.best_fixed_validation,
        .best_threshold_validation = metrics.best_threshold_validation,
        .best_average_precision_validation = metrics.best_average_precision_validation,
        .best_max_rank_validation = metrics.best_max_rank_validation,
        .frozen_feature_probe = metrics.frozen_feature_probe,
        .first_loss = metrics.first_loss,
        .final_loss = metrics.final_loss,
        .first_boundary_loss = metrics.first_boundary_loss,
        .final_boundary_loss = metrics.final_boundary_loss,
        .min_boundary_loss = metrics.min_boundary_loss,
        .max_vjp_fallbacks = metrics.max_vjp_fallbacks,
        .mpsgraph_step_count = metrics.mpsgraph_step_count,
        .non_mpsgraph_vjp_step_count = metrics.non_mpsgraph_vjp_step_count,
        .latest_probability_gap = latest_gap,
        .latest_train_probability_gap = latest_train_gap,
        .latest_train_best_f1_minus_validation_best_f1 = train_best_f1_delta,
        .latest_train_calibrated_f1_minus_validation_calibrated_f1 = train_calibrated_delta,
        .latest_train_best_f1_minus_validation_fitted_f1 = train_best_vs_fitted_delta,
        .latest_fitted_f1_minus_best_f1 = fitted_vs_best_delta,
        .latest_train_ap_minus_validation_ap = train_ap_delta,
        .latest_train_max_rank_f1_minus_validation_max_rank_f1 = train_max_rank_delta,
        .frozen_feature_train_best_f1_minus_val_best_f1 = frozen_feature_best_delta,
        .frozen_feature_train_ap_minus_val_ap = frozen_feature_ap_delta,
        .latest_best_f1_gap_to_release_floor = latest_best_gap,
        .failure_modes = modes,
        .stop_long_training = stop_long_training,
        .next_target = chooseNextTarget(modes),
    };
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
            const loss = jsonF64(obj.get("loss"));
            if (loss) |value| {
                if (out.first_loss == null) out.first_loss = value;
                out.final_loss = value;
            }
            if (jsonF64(obj.get("boundary_loss"))) |boundary_loss| {
                if (out.first_boundary_loss == null) out.first_boundary_loss = boundary_loss;
                out.final_boundary_loss = boundary_loss;
                out.min_boundary_loss = if (out.min_boundary_loss) |current| @min(current, boundary_loss) else boundary_loss;
            }
            out.max_vjp_fallbacks = @max(out.max_vjp_fallbacks, jsonU64(obj.get("vjp_interpreter_fallbacks")) orelse 0);
            const runtime = jsonString(obj.get("vjp_runtime")) orelse "none";
            if (std.mem.eql(u8, runtime, "mpsgraph")) {
                out.mpsgraph_step_count += 1;
            } else if (!std.mem.eql(u8, runtime, "none")) {
                out.non_mpsgraph_vjp_step_count += 1;
            }
        } else if (std.mem.startsWith(u8, event, "validation_")) {
            const validation = validationFromJson(obj);
            out.validation_count += 1;
            out.latest_validation = validation;
            if (out.best_fixed_validation == null or validation.fixed_f1 > out.best_fixed_validation.?.fixed_f1) {
                out.best_fixed_validation = validation;
            }
            if (out.best_threshold_validation == null or validation.best_threshold_f1 > out.best_threshold_validation.?.best_threshold_f1) {
                out.best_threshold_validation = validation;
            }
            if (out.best_average_precision_validation == null or validation.average_precision > out.best_average_precision_validation.?.average_precision) {
                out.best_average_precision_validation = validation;
            }
            if (out.best_max_rank_validation == null or validation.max_rank_f1 > out.best_max_rank_validation.?.max_rank_f1) {
                out.best_max_rank_validation = validation;
            }
        } else if (std.mem.startsWith(u8, event, "train_validation_")) {
            out.train_validation_count += 1;
            out.latest_train_validation = validationFromJson(obj);
        } else if (std.mem.eql(u8, event, "frozen_feature_probe")) {
            out.frozen_feature_probe = frozenFeatureProbeFromJson(obj);
        }
    }
    return out;
}

fn frozenFeatureProbeFromJson(obj: std.json.ObjectMap) FrozenFeatureProbeSnapshot {
    return .{
        .train_examples = jsonUsize(obj.get("train_examples")) orelse 0,
        .val_examples = jsonUsize(obj.get("val_examples")) orelse 0,
        .train_offset = jsonUsize(obj.get("train_offset")) orelse 0,
        .val_offset = jsonUsize(obj.get("val_offset")) orelse 0,
        .epochs = @intCast(jsonU64(obj.get("epochs")) orelse 0),
        .final_step = @intCast(jsonU64(obj.get("final_step")) orelse 0),
        .final_boundary_loss = jsonF64(obj.get("final_boundary_loss")),
        .train_best_f1 = jsonF64(obj.get("train_best_f1")) orelse 0,
        .train_fixed_f1 = jsonF64(obj.get("train_fixed_f1")) orelse 0,
        .train_max_rank_f1 = jsonF64(obj.get("train_max_rank_f1")) orelse 0,
        .train_average_precision = jsonF64(obj.get("train_average_precision")) orelse 0,
        .train_probability_gap = jsonF64(obj.get("train_probability_gap")),
        .val_best_f1 = jsonF64(obj.get("val_best_f1")),
        .val_fixed_f1 = jsonF64(obj.get("val_fixed_f1")),
        .val_max_rank_f1 = jsonF64(obj.get("val_max_rank_f1")),
        .val_average_precision = jsonF64(obj.get("val_average_precision")),
        .val_probability_gap = jsonF64(obj.get("val_probability_gap")),
    };
}

fn validationFromJson(obj: std.json.ObjectMap) ValidationSnapshot {
    return .{
        .step = jsonUsize(obj.get("step")),
        .samples = jsonUsize(obj.get("samples")),
        .total_samples = jsonUsize(obj.get("total_samples")),
        .fixed_f1 = jsonF64(obj.get("f1")) orelse 0,
        .precision = jsonF64(obj.get("precision")) orelse 0,
        .recall = jsonF64(obj.get("recall")) orelse 0,
        .tp = jsonU64(obj.get("tp")) orelse 0,
        .fp = jsonU64(obj.get("fp")) orelse 0,
        .fn_count = jsonU64(obj.get("fn")) orelse 0,
        .best_threshold_f1 = jsonF64(obj.get("best_f1")) orelse 0,
        .best_threshold = jsonF64(obj.get("best_threshold")),
        .best_tp = jsonU64(obj.get("best_tp")) orelse 0,
        .best_fp = jsonU64(obj.get("best_fp")) orelse 0,
        .best_fn = jsonU64(obj.get("best_fn")) orelse 0,
        .calibrated_threshold_available = jsonBool(obj.get("calibrated_threshold_available")) orelse false,
        .calibrated_boundary_threshold = jsonF64(obj.get("calibrated_boundary_threshold")),
        .calibrated_boundary_f1 = jsonF64(obj.get("calibrated_boundary_f1")),
        .calibrated_boundary_precision = jsonF64(obj.get("calibrated_boundary_precision")),
        .calibrated_boundary_recall = jsonF64(obj.get("calibrated_boundary_recall")),
        .calibrated_boundary_tp = jsonU64(obj.get("calibrated_boundary_tp")) orelse 0,
        .calibrated_boundary_fp = jsonU64(obj.get("calibrated_boundary_fp")) orelse 0,
        .calibrated_boundary_fn = jsonU64(obj.get("calibrated_boundary_fn")) orelse 0,
        .calibrated_predicted_positive_rate = jsonF64(obj.get("calibrated_predicted_positive_rate")),
        .fitted_threshold_available = jsonBool(obj.get("fitted_threshold_available")) orelse false,
        .fitted_boundary_threshold = jsonF64(obj.get("fitted_boundary_threshold")),
        .fitted_boundary_f1 = jsonF64(obj.get("fitted_boundary_f1")),
        .fitted_boundary_precision = jsonF64(obj.get("fitted_boundary_precision")),
        .fitted_boundary_recall = jsonF64(obj.get("fitted_boundary_recall")),
        .fitted_boundary_tp = jsonU64(obj.get("fitted_boundary_tp")) orelse 0,
        .fitted_boundary_fp = jsonU64(obj.get("fitted_boundary_fp")) orelse 0,
        .fitted_boundary_fn = jsonU64(obj.get("fitted_boundary_fn")) orelse 0,
        .fitted_predicted_positive_rate = jsonF64(obj.get("fitted_predicted_positive_rate")),
        .average_precision = jsonF64(obj.get("average_precision")) orelse 0,
        .precision_at_gold_count = jsonF64(obj.get("precision_at_gold_count")) orelse 0,
        .max_rank_f1 = jsonF64(obj.get("max_rank_f1")) orelse 0,
        .max_rank_precision = jsonF64(obj.get("max_rank_precision")) orelse 0,
        .max_rank_recall = jsonF64(obj.get("max_rank_recall")) orelse 0,
        .max_rank_threshold = jsonF64(obj.get("max_rank_threshold")),
        .gold_positive_mean_rank_percentile = jsonF64(obj.get("gold_positive_mean_rank_percentile")),
        .gold_positive_median_rank_percentile = jsonF64(obj.get("gold_positive_median_rank_percentile")),
        .gold_positive_p90_rank_percentile = jsonF64(obj.get("gold_positive_p90_rank_percentile")),
        .gold_positive_p99_rank_percentile = jsonF64(obj.get("gold_positive_p99_rank_percentile")),
        .gold_positive_top_5x_recall = jsonF64(obj.get("gold_positive_top_5x_recall")),
        .gold_positive_top_10x_recall = jsonF64(obj.get("gold_positive_top_10x_recall")),
        .valid_tokens = jsonUsize(obj.get("valid_tokens")),
        .gold_positives = jsonUsize(obj.get("gold_positives")),
        .gold_positive_rate = jsonF64(obj.get("gold_positive_rate")),
        .predicted_positive_rate = jsonF64(obj.get("predicted_positive_rate")),
        .best_predicted_positive_rate = jsonF64(obj.get("best_predicted_positive_rate")),
        .mean_positive_probability_gold_positive = jsonF64(obj.get("mean_positive_probability_gold_positive")),
        .mean_positive_probability_gold_negative = jsonF64(obj.get("mean_positive_probability_gold_negative")),
        .mean_boundary_margin_gold_positive = jsonF64(obj.get("mean_boundary_margin_gold_positive")),
        .mean_boundary_margin_gold_negative = jsonF64(obj.get("mean_boundary_margin_gold_negative")),
    };
}

fn classifyFailure(metrics: MetricsInspection) FailureModes {
    var modes = FailureModes{};
    if (metrics.frozen_feature_probe) |probe| {
        modes.frozen_feature_train_memorizes_heldout_poor = probe.trainGeneralizesPoorly();
        if (probe.val_probability_gap) |gap| {
            modes.frozen_feature_heldout_probability_gap_too_small = gap < 0.05;
        }
    }
    const latest = metrics.latest_validation orelse {
        modes.no_validation_records = true;
        return modes;
    };
    const probability_gap = latest.probabilityGap();
    const gold_rate = latest.gold_positive_rate orelse 0;
    const fixed_pred_rate = latest.predicted_positive_rate orelse 0;
    const best_threshold = latest.best_threshold orelse 1;

    modes.predicted_zero_at_fixed_threshold = fixed_pred_rate <= 0.000001;
    modes.best_threshold_extremely_low = best_threshold <= 0.03;
    modes.fixed_threshold_underconfident =
        latest.fixed_f1 <= 0.001 and
        modes.predicted_zero_at_fixed_threshold and
        modes.best_threshold_extremely_low;
    modes.best_threshold_below_release_floor = latest.best_threshold_f1 < release_boundary_f1_floor;
    const fitted_f1 = latest.fitted_boundary_f1 orelse 0;
    const fitted_pred_rate = latest.fitted_predicted_positive_rate orelse 0;
    const fitted_close_to_best =
        latest.fitted_threshold_available and
        latest.best_threshold_f1 > 0 and
        fitted_f1 >= latest.best_threshold_f1 * 0.75;
    if (latest.fitted_threshold_available) {
        modes.fitted_threshold_underpredicting =
            fitted_pred_rate <= @max(0.000001, gold_rate * 0.25) and
            latest.best_threshold_f1 > 0.02;
        modes.fitted_threshold_worse_than_best =
            latest.best_threshold_f1 > 0.02 and
            fitted_f1 <= latest.best_threshold_f1 * 0.50;
    }
    if (latest.calibrated_threshold_available) {
        const calibrated_f1 = latest.calibrated_boundary_f1 orelse 0;
        const calibrated_pred_rate = latest.calibrated_predicted_positive_rate orelse 0;
        modes.calibrated_threshold_underpredicting =
            !fitted_close_to_best and
            calibrated_pred_rate <= @max(0.000001, gold_rate * 0.25) and
            latest.best_threshold_f1 > 0.02;
        modes.calibrated_threshold_worse_than_best =
            !fitted_close_to_best and
            latest.best_threshold_f1 > 0.02 and
            calibrated_f1 <= latest.best_threshold_f1 * 0.25;
    }
    modes.rank_near_chance =
        latest.max_rank_f1 <= 0.05 or
        latest.average_precision <= @max(0.02, gold_rate * 3.0);
    if (probability_gap) |gap| {
        modes.probability_gap_inverted = gap <= 0;
        modes.probability_gap_too_small = gap > 0 and gap < 0.01;
    }
    if (latest.gold_positive_p90_rank_percentile) |p90| {
        modes.gold_ranks_buried = p90 >= 0.50;
    } else if (latest.gold_positive_median_rank_percentile) |median| {
        modes.gold_ranks_buried = median >= 0.25;
    }
    if (gold_rate > 0) {
        modes.overpredicting_fixed_threshold =
            fixed_pred_rate >= @max(0.05, gold_rate * 20.0);
    }
    if (metrics.latest_train_validation) |train| {
        const train_rank_good = train.max_rank_f1 >= 0.30 or train.average_precision >= 0.20;
        const val_rank_poor = latest.max_rank_f1 < 0.10 or latest.average_precision < 0.05;
        modes.train_rank_good_validation_rank_poor = train_rank_good and val_rank_poor;
        modes.train_and_validation_rank_poor = !train_rank_good and val_rank_poor;
    }
    if (metrics.best_threshold_validation) |best_validation| {
        modes.collapse_after_peak =
            best_validation.best_threshold_f1 >= 0.10 and
            latest.best_threshold_f1 <= best_validation.best_threshold_f1 * 0.5;
    }
    if (metrics.first_loss) |first_loss| {
        if (metrics.final_loss) |final_loss| {
            modes.train_loss_down_but_validation_poor =
                final_loss < first_loss * 0.5 and latest.max_rank_f1 < 0.10;
        }
    }
    return modes;
}

fn shouldStopLongTraining(modes: FailureModes) bool {
    return modes.no_validation_records or
        modes.frozen_feature_train_memorizes_heldout_poor or
        modes.rank_near_chance or
        modes.gold_ranks_buried or
        modes.train_and_validation_rank_poor or
        modes.train_rank_good_validation_rank_poor or
        modes.probability_gap_inverted or
        modes.calibrated_threshold_underpredicting or
        modes.calibrated_threshold_worse_than_best or
        modes.fitted_threshold_underpredicting or
        modes.fitted_threshold_worse_than_best or
        modes.best_threshold_below_release_floor or
        modes.train_loss_down_but_validation_poor or
        modes.collapse_after_peak;
}

fn chooseNextTarget(modes: FailureModes) []const u8 {
    if (modes.frozen_feature_train_memorizes_heldout_poor) return "improve_encoder_boundary_features_or_add_contextual_boundary_objective_before_long_run";
    if (modes.no_validation_records) return "enable_step_validation_before_more_training";
    if (modes.probability_gap_inverted) return "check_boundary_label_polarity_and_feature_label_alignment";
    if (modes.train_rank_good_validation_rank_poor) return "inspect_validation_distribution_and_overfit_gap";
    if (modes.train_and_validation_rank_poor) return "inspect_train_boundary_feature_label_alignment";
    if (modes.rank_near_chance or modes.gold_ranks_buried or modes.train_loss_down_but_validation_poor) return "dump_train_vs_val_gold_positive_ranks_and_boundary_features";
    if (modes.fitted_threshold_underpredicting or modes.fitted_threshold_worse_than_best) return "fit_train_threshold_or_temperature_calibration";
    if (modes.calibrated_threshold_underpredicting or modes.calibrated_threshold_worse_than_best) return "fit_validation_threshold_or_temperature_calibration";
    if (modes.best_threshold_below_release_floor) return "improve_boundary_rank_quality_before_long_run";
    if (modes.overpredicting_fixed_threshold) return "reduce_positive_bias_before_long_run";
    if (modes.fixed_threshold_underconfident) return "defer_calibration_until_rank_quality_improves";
    if (modes.collapse_after_peak) return "compare_latest_vs_best_checkpoint_deltas";
    return "continue_gated_probe_with_release_floor_latest_metrics";
}

fn stringifyAnalysis(allocator: std.mem.Allocator, analysis: Analysis) ![]const u8 {
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
        .null => null,
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

fn jsonBool(value: ?std.json.Value) ?bool {
    const v = value orelse return null;
    return switch (v) {
        .bool => |b| b,
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
        \\usage: analyze-fused-chunker-validation-failure (--out-dir <run-dir>|--metrics <metrics.jsonl>) [--analysis-out <path>]
        \\
        \\Reads fused_training_metrics.jsonl and emits a structured boundary-quality failure diagnosis.
        \\
    , .{});
}

test "classifies underconfident near-chance validation failure" {
    const metrics_jsonl =
        \\{"event":"step","loss":1.4,"boundary_loss":1.2,"step_wall_ms":10.0,"vjp_runtime":"mpsgraph","vjp_interpreter_fallbacks":0}
        \\{"event":"step","loss":0.09,"boundary_loss":0.03,"step_wall_ms":10.0,"vjp_runtime":"mpsgraph","vjp_interpreter_fallbacks":0}
        \\{"event":"validation_step","step":500,"f1":0,"precision":0,"recall":0,"tp":0,"fp":0,"fn":819,"best_f1":0.030157342553138733,"best_threshold":0.009999999776482582,"best_tp":69,"best_fp":3688,"best_fn":750,"average_precision":0.009320156648755074,"precision_at_gold_count":0.01587301678955555,"max_rank_f1":0.031102096661925316,"gold_positive_rate":0.004370867274701595,"predicted_positive_rate":0,"best_predicted_positive_rate":0.02005048654973507,"mean_positive_probability_gold_positive":0.0037623147945851088,"mean_positive_probability_gold_negative":0.0019845045171678066}
        \\
    ;
    const analysis = try analyzeMetricsJsonl(std.testing.allocator, "metrics.jsonl", metrics_jsonl);
    try std.testing.expectEqual(@as(usize, 2), analysis.step_count);
    try std.testing.expectEqual(@as(usize, 1), analysis.validation_count);
    try std.testing.expect(analysis.failure_modes.fixed_threshold_underconfident);
    try std.testing.expect(analysis.failure_modes.rank_near_chance);
    try std.testing.expect(analysis.failure_modes.probability_gap_too_small);
    try std.testing.expect(analysis.failure_modes.train_loss_down_but_validation_poor);
    try std.testing.expect(analysis.stop_long_training);
    try std.testing.expectEqualStrings("dump_train_vs_val_gold_positive_ranks_and_boundary_features", analysis.next_target);
}

test "classifies overprediction separately from underconfidence" {
    const metrics_jsonl =
        \\{"event":"step","loss":0.7,"boundary_loss":0.4,"step_wall_ms":10.0,"vjp_runtime":"mpsgraph","vjp_interpreter_fallbacks":0}
        \\{"event":"validation_step","step":100,"f1":0.01069,"precision":0.0054,"recall":0.50,"best_f1":0.01152,"best_threshold":0.5,"average_precision":0.008,"max_rank_f1":0.012,"gold_positive_rate":0.004484,"predicted_positive_rate":0.327131,"mean_positive_probability_gold_positive":0.52,"mean_positive_probability_gold_negative":0.48}
        \\
    ;
    const analysis = try analyzeMetricsJsonl(std.testing.allocator, "metrics.jsonl", metrics_jsonl);
    try std.testing.expect(analysis.failure_modes.overpredicting_fixed_threshold);
    try std.testing.expect(!analysis.failure_modes.fixed_threshold_underconfident);
}

test "classifies calibrated threshold failure separately from rank failure" {
    const metrics_jsonl =
        \\{"event":"step","loss":0.7,"boundary_loss":0.4,"step_wall_ms":10.0,"vjp_runtime":"mpsgraph","vjp_interpreter_fallbacks":0}
        \\{"event":"validation_step","step":100,"f1":0.12,"precision":0.08,"recall":0.24,"best_f1":0.31,"best_threshold":0.72,"average_precision":0.12,"max_rank_f1":0.24,"gold_positive_rate":0.01,"predicted_positive_rate":0.03,"calibrated_threshold_available":true,"calibrated_boundary_threshold":0.995,"calibrated_boundary_f1":0.0,"calibrated_predicted_positive_rate":0.0,"mean_positive_probability_gold_positive":0.62,"mean_positive_probability_gold_negative":0.40}
        \\
    ;
    const analysis = try analyzeMetricsJsonl(std.testing.allocator, "metrics.jsonl", metrics_jsonl);
    try std.testing.expect(!analysis.failure_modes.rank_near_chance);
    try std.testing.expect(analysis.failure_modes.calibrated_threshold_underpredicting);
    try std.testing.expect(analysis.failure_modes.calibrated_threshold_worse_than_best);
    try std.testing.expect(analysis.stop_long_training);
    try std.testing.expectEqualStrings("fit_validation_threshold_or_temperature_calibration", analysis.next_target);
}

test "classifies inverted probability gap" {
    const metrics_jsonl =
        \\{"event":"step","loss":1.0,"boundary_loss":0.8,"step_wall_ms":10.0,"vjp_runtime":"mpsgraph","vjp_interpreter_fallbacks":0}
        \\{"event":"validation_step","step":100,"f1":0.003,"best_f1":0.010,"best_threshold":0.07,"average_precision":0.004,"max_rank_f1":0.011,"gold_positive_rate":0.004,"predicted_positive_rate":0.002,"mean_positive_probability_gold_positive":0.11,"mean_positive_probability_gold_negative":0.12}
        \\
    ;
    const analysis = try analyzeMetricsJsonl(std.testing.allocator, "metrics.jsonl", metrics_jsonl);
    try std.testing.expect(analysis.failure_modes.probability_gap_inverted);
    try std.testing.expect(analysis.stop_long_training);
    try std.testing.expectEqualStrings("check_boundary_label_polarity_and_feature_label_alignment", analysis.next_target);
}

test "classifies buried gold ranks from rank diagnostics" {
    const metrics_jsonl =
        \\{"event":"step","loss":1.0,"boundary_loss":0.8,"step_wall_ms":10.0,"vjp_runtime":"mpsgraph","vjp_interpreter_fallbacks":0}
        \\{"event":"validation_step","step":100,"f1":0.12,"best_f1":0.19,"best_threshold":0.20,"average_precision":0.06,"max_rank_f1":0.08,"gold_positive_rate":0.01,"predicted_positive_rate":0.04,"gold_positive_median_rank_percentile":0.30,"gold_positive_p90_rank_percentile":0.72,"gold_positive_top_5x_recall":0.08,"gold_positive_top_10x_recall":0.12,"mean_positive_probability_gold_positive":0.21,"mean_positive_probability_gold_negative":0.19}
        \\
    ;
    const analysis = try analyzeMetricsJsonl(std.testing.allocator, "metrics.jsonl", metrics_jsonl);
    try std.testing.expect(analysis.failure_modes.gold_ranks_buried);
    try std.testing.expect(analysis.stop_long_training);
    try std.testing.expectEqualStrings("dump_train_vs_val_gold_positive_ranks_and_boundary_features", analysis.next_target);
}

test "classifies train good validation poor as overfit or distribution gap" {
    const metrics_jsonl =
        \\{"event":"step","loss":0.20,"boundary_loss":0.05,"step_wall_ms":10.0,"vjp_runtime":"mpsgraph","vjp_interpreter_fallbacks":0}
        \\{"event":"train_validation_step","step":100,"f1":0.31,"best_f1":0.42,"best_threshold":0.14,"average_precision":0.28,"max_rank_f1":0.37,"gold_positive_rate":0.01,"predicted_positive_rate":0.03,"mean_positive_probability_gold_positive":0.21,"mean_positive_probability_gold_negative":0.08}
        \\{"event":"validation_step","step":100,"f1":0.0,"best_f1":0.03,"best_threshold":0.01,"average_precision":0.009,"max_rank_f1":0.03,"gold_positive_rate":0.01,"predicted_positive_rate":0.0,"mean_positive_probability_gold_positive":0.03,"mean_positive_probability_gold_negative":0.02}
        \\
    ;
    const analysis = try analyzeMetricsJsonl(std.testing.allocator, "metrics.jsonl", metrics_jsonl);
    try std.testing.expectEqual(@as(usize, 1), analysis.train_validation_count);
    try std.testing.expect(analysis.failure_modes.train_rank_good_validation_rank_poor);
    try std.testing.expect(!analysis.failure_modes.train_and_validation_rank_poor);
    try std.testing.expectApproxEqAbs(@as(f64, 0.39), analysis.latest_train_best_f1_minus_validation_best_f1.?, 1e-12);
    try std.testing.expectEqualStrings("inspect_validation_distribution_and_overfit_gap", analysis.next_target);
}

test "classifies train and validation both poor as alignment issue" {
    const metrics_jsonl =
        \\{"event":"step","loss":0.20,"boundary_loss":0.05,"step_wall_ms":10.0,"vjp_runtime":"mpsgraph","vjp_interpreter_fallbacks":0}
        \\{"event":"train_validation_step","step":100,"f1":0.0,"best_f1":0.04,"best_threshold":0.01,"average_precision":0.010,"max_rank_f1":0.04,"gold_positive_rate":0.01,"predicted_positive_rate":0.0,"mean_positive_probability_gold_positive":0.03,"mean_positive_probability_gold_negative":0.02}
        \\{"event":"validation_step","step":100,"f1":0.0,"best_f1":0.03,"best_threshold":0.01,"average_precision":0.009,"max_rank_f1":0.03,"gold_positive_rate":0.01,"predicted_positive_rate":0.0,"mean_positive_probability_gold_positive":0.03,"mean_positive_probability_gold_negative":0.02}
        \\
    ;
    const analysis = try analyzeMetricsJsonl(std.testing.allocator, "metrics.jsonl", metrics_jsonl);
    try std.testing.expect(analysis.failure_modes.train_and_validation_rank_poor);
    try std.testing.expect(!analysis.failure_modes.train_rank_good_validation_rank_poor);
    try std.testing.expectEqualStrings("inspect_train_boundary_feature_label_alignment", analysis.next_target);
}

test "classifies frozen feature train memorization with poor heldout transfer" {
    const metrics_jsonl =
        \\{"event":"frozen_feature_probe_train_after","step":240,"samples":16,"total_samples":16,"f1":0.8965517,"best_f1":0.9285714,"average_precision":0.9492121,"max_rank_f1":0.9285714,"mean_positive_probability_gold_positive":0.875232,"mean_positive_probability_gold_negative":0.006471}
        \\{"event":"frozen_feature_probe_val_after","step":240,"samples":16,"total_samples":16,"f1":0.0571429,"best_f1":0.0800000,"average_precision":0.0139744,"max_rank_f1":0.0816327,"mean_positive_probability_gold_positive":0.060598,"mean_positive_probability_gold_negative":0.010664}
        \\{"event":"frozen_feature_probe","status":"complete","train_examples":16,"val_examples":16,"train_offset":0,"val_offset":1024,"epochs":30,"final_step":240,"final_boundary_loss":0.004645,"train_best_f1":0.9285714,"train_fixed_f1":0.8965517,"train_max_rank_f1":0.9285714,"train_average_precision":0.9492121,"train_probability_gap":0.8687615,"val_best_f1":0.0800000,"val_fixed_f1":0.0571429,"val_max_rank_f1":0.0816327,"val_average_precision":0.0139744,"val_probability_gap":0.0499344}
        \\
    ;
    const analysis = try analyzeMetricsJsonl(std.testing.allocator, "metrics.jsonl", metrics_jsonl);
    try std.testing.expect(analysis.failure_modes.no_validation_records);
    try std.testing.expect(analysis.failure_modes.frozen_feature_train_memorizes_heldout_poor);
    try std.testing.expect(analysis.failure_modes.frozen_feature_heldout_probability_gap_too_small);
    try std.testing.expectApproxEqAbs(@as(f64, 0.8485714), analysis.frozen_feature_train_best_f1_minus_val_best_f1.?, 1e-7);
    try std.testing.expectEqualStrings("improve_encoder_boundary_features_or_add_contextual_boundary_objective_before_long_run", analysis.next_target);
}
