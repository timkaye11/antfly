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
const termite = @import("termite_internal");

const gliner2_data = termite.finetune.gliner2_data;
const gliner2_autodiff = termite.finetune.gliner2_real_autodiff;
const adapter_eval = @import("eval_gliner2_autodiff_adapter.zig");

const Options = struct {
    model_dir: []const u8,
    adapter_dir: []const u8,
    eval_data: []const u8,
    entity_types_csv: []const u8,
    seq_len: usize = 32,
    max_span_width: usize = 4,
    objective: gliner2_autodiff.GlinerObjective = .span_start,
    max_examples: usize = 0,
    min_prediction_score: f32 = 0.03,
    nms_overlap_threshold: ?f64 = 0.0,
    max_predictions_per_example: ?usize = null,
    top_k_per_label: ?usize = null,
    best_span_per_label_start: bool = false,
    sweep_thresholds: []const f32 = &.{},
    min_precision: ?f64 = null,
    min_recall: ?f64 = null,
    min_f1: ?f64 = null,
};

const LabelMetric = struct {
    label: []const u8,
    gold: usize = 0,
    predicted: usize = 0,
    correct: usize = 0,
    precision: f64 = 0.0,
    recall: f64 = 0.0,
    f1: f64 = 0.0,
};

const QualitySummary = struct {
    model_dir: []const u8,
    adapter_dir: []const u8,
    eval_data: []const u8,
    entity_types: []const []const u8,
    objective: []const u8,
    seq_len: usize,
    max_span_width: usize,
    example_count: usize,
    gold_entity_count: usize,
    predicted_entity_count: usize,
    correct_entity_count: usize,
    false_positive_count: usize,
    false_negative_count: usize,
    precision: f64,
    recall: f64,
    f1: f64,
    min_prediction_score: f32,
    nms_overlap_threshold: ?f64,
    max_predictions_per_example: ?usize,
    top_k_per_label: ?usize,
    best_span_per_label_start: bool,
    per_label_score_stats: []const LabelScoreStats,
    per_label: []const LabelMetric,
    threshold_sweep: []const ThresholdMetric,
    status: []const u8,
};

const LabelScoreStats = struct {
    label: []const u8,
    count: usize = 0,
    min: f32 = 0.0,
    max: f32 = 0.0,
    mean: f64 = 0.0,

    fn observe(self: *LabelScoreStats, score: f32) void {
        if (self.count == 0) {
            self.min = score;
            self.max = score;
            self.mean = score;
            self.count = 1;
            return;
        }
        self.min = @min(self.min, score);
        self.max = @max(self.max, score);
        self.mean = ((self.mean * @as(f64, @floatFromInt(self.count))) + score) / @as(f64, @floatFromInt(self.count + 1));
        self.count += 1;
    }
};

const ThresholdMetric = struct {
    min_prediction_score: f32,
    predicted_entity_count: usize,
    correct_entity_count: usize,
    precision: f64,
    recall: f64,
    f1: f64,
};

const EvalAccumulator = struct {
    threshold: f32,
    label_metrics: []LabelMetric,
    gold_total: usize = 0,
    predicted_total: usize = 0,
    correct_total: usize = 0,

    fn init(allocator: std.mem.Allocator, threshold: f32, entity_types: []const []const u8) !EvalAccumulator {
        const label_metrics = try allocator.alloc(LabelMetric, entity_types.len);
        for (label_metrics, entity_types) |*metric, label| metric.* = .{ .label = label };
        return .{ .threshold = threshold, .label_metrics = label_metrics };
    }

    fn deinit(self: *EvalAccumulator, allocator: std.mem.Allocator) void {
        allocator.free(self.label_metrics);
        self.* = undefined;
    }

    fn addGold(self: *EvalAccumulator, label_idx: usize) void {
        self.gold_total += 1;
        self.label_metrics[label_idx].gold += 1;
    }

    fn addPrediction(self: *EvalAccumulator, label_idx: usize, correct: bool) void {
        self.predicted_total += 1;
        self.label_metrics[label_idx].predicted += 1;
        if (correct) {
            self.correct_total += 1;
            self.label_metrics[label_idx].correct += 1;
        }
    }

    fn finish(self: *EvalAccumulator) void {
        for (self.label_metrics) |*metric| finishMetric(metric);
    }

    fn precision(self: *const EvalAccumulator) f64 {
        return ratio(self.correct_total, self.predicted_total);
    }

    fn recall(self: *const EvalAccumulator) f64 {
        return ratio(self.correct_total, self.gold_total);
    }

    fn f1(self: *const EvalAccumulator) f64 {
        return f1Score(self.precision(), self.recall());
    }

    fn thresholdMetric(self: *const EvalAccumulator) ThresholdMetric {
        return .{
            .min_prediction_score = self.threshold,
            .predicted_entity_count = self.predicted_total,
            .correct_entity_count = self.correct_total,
            .precision = self.precision(),
            .recall = self.recall(),
            .f1 = self.f1(),
        };
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const opts = try parseOptions(init.minimal.args, allocator) orelse return;
    defer allocator.free(opts.sweep_thresholds);
    const entity_types = try parseCsv(allocator, opts.entity_types_csv);
    defer freeStringSlice(allocator, entity_types);

    var loaded = try gliner2_data.loadExamples(allocator, opts.eval_data, null);
    defer loaded.deinit();

    const limit = if (opts.max_examples == 0) loaded.examples.len else @min(opts.max_examples, loaded.examples.len);
    if (limit == 0) return error.NoEvalExamples;

    const decode_floor = minDecodeThreshold(opts);
    var main_accum = try EvalAccumulator.init(allocator, opts.min_prediction_score, entity_types);
    defer main_accum.deinit(allocator);
    const score_stats = try allocator.alloc(LabelScoreStats, entity_types.len);
    defer allocator.free(score_stats);
    for (score_stats, entity_types) |*stat, label| stat.* = .{ .label = label };
    var sweep_accums = try allocator.alloc(EvalAccumulator, opts.sweep_thresholds.len);
    defer allocator.free(sweep_accums);
    for (opts.sweep_thresholds, 0..) |threshold, idx| {
        sweep_accums[idx] = try EvalAccumulator.init(allocator, threshold, entity_types);
    }
    defer for (sweep_accums) |*accum| accum.deinit(allocator);

    for (loaded.examples[0..limit]) |ex| {
        for (ex.entities) |ent| {
            if (indexOfLabel(entity_types, ent.label)) |label_idx| {
                main_accum.addGold(label_idx);
                for (sweep_accums) |*accum| accum.addGold(label_idx);
            }
        }

        var summary = try adapter_eval.evalSavedAdapterText(allocator, .{
            .model_dir = opts.model_dir,
            .adapter_dir = opts.adapter_dir,
            .text = ex.text,
            .entity_types_csv = opts.entity_types_csv,
            .seq_len = opts.seq_len,
            .max_span_width = opts.max_span_width,
            .objective_override = opts.objective,
            .prediction_threshold = decode_floor,
        });
        defer summary.deinit(allocator);

        observeScores(score_stats, entity_types, summary.predictions);
        try scoreExample(allocator, &main_accum, ex, entity_types, summary.predictions, opts);
        for (sweep_accums) |*accum| try scoreExample(allocator, accum, ex, entity_types, summary.predictions, opts);
    }

    main_accum.finish();
    for (sweep_accums) |*accum| accum.finish();
    const precision = main_accum.precision();
    const recall = main_accum.recall();
    const f1 = main_accum.f1();

    if (opts.min_precision) |threshold| {
        if (!std.math.isFinite(threshold) or threshold < 0 or threshold > 1) return error.InvalidQualityThreshold;
        if (precision < threshold) return error.EntityPrecisionBelowThreshold;
    }
    if (opts.min_recall) |threshold| {
        if (!std.math.isFinite(threshold) or threshold < 0 or threshold > 1) return error.InvalidQualityThreshold;
        if (recall < threshold) return error.EntityRecallBelowThreshold;
    }
    if (opts.min_f1) |threshold| {
        if (!std.math.isFinite(threshold) or threshold < 0 or threshold > 1) return error.InvalidQualityThreshold;
        if (f1 < threshold) return error.EntityF1BelowThreshold;
    }

    const threshold_sweep = try allocator.alloc(ThresholdMetric, sweep_accums.len);
    defer allocator.free(threshold_sweep);
    for (sweep_accums, threshold_sweep) |accum, *metric| metric.* = accum.thresholdMetric();

    const stdout = std.Io.File.stdout();
    var buf: [32768]u8 = undefined;
    var writer = stdout.writer(init.io, &buf);
    try std.json.Stringify.value(QualitySummary{
        .model_dir = opts.model_dir,
        .adapter_dir = opts.adapter_dir,
        .eval_data = opts.eval_data,
        .entity_types = entity_types,
        .objective = objectiveName(opts.objective),
        .seq_len = opts.seq_len,
        .max_span_width = opts.max_span_width,
        .example_count = limit,
        .gold_entity_count = main_accum.gold_total,
        .predicted_entity_count = main_accum.predicted_total,
        .correct_entity_count = main_accum.correct_total,
        .false_positive_count = main_accum.predicted_total - main_accum.correct_total,
        .false_negative_count = main_accum.gold_total - main_accum.correct_total,
        .precision = precision,
        .recall = recall,
        .f1 = f1,
        .min_prediction_score = opts.min_prediction_score,
        .nms_overlap_threshold = opts.nms_overlap_threshold,
        .max_predictions_per_example = opts.max_predictions_per_example,
        .top_k_per_label = opts.top_k_per_label,
        .best_span_per_label_start = opts.best_span_per_label_start,
        .per_label_score_stats = score_stats,
        .per_label = main_accum.label_metrics,
        .threshold_sweep = threshold_sweep,
        .status = "passed",
    }, .{ .whitespace = .indent_2 }, &writer.interface);
    try writer.interface.writeByte('\n');
    try writer.interface.flush();
}

fn observeScores(score_stats: []LabelScoreStats, entity_types: []const []const u8, predictions: []const adapter_eval.TopEntity) void {
    for (predictions) |prediction| {
        if (indexOfLabel(entity_types, prediction.label)) |label_idx| score_stats[label_idx].observe(prediction.score);
    }
}

fn minDecodeThreshold(opts: Options) f32 {
    var threshold = opts.min_prediction_score;
    for (opts.sweep_thresholds) |candidate| threshold = @min(threshold, candidate);
    return threshold;
}

fn scoreExample(
    allocator: std.mem.Allocator,
    accum: *EvalAccumulator,
    ex: gliner2_data.Example,
    entity_types: []const []const u8,
    predictions: []const adapter_eval.TopEntity,
    opts: Options,
) !void {
    const selected = try selectPredictions(allocator, predictions, accum.threshold, opts);
    defer allocator.free(selected);
    var gold_matched = try allocator.alloc(bool, ex.entities.len);
    defer allocator.free(gold_matched);
    @memset(gold_matched, false);

    for (selected) |prediction_idx| {
        const prediction = predictions[prediction_idx];
        const pred_label_idx = indexOfLabel(entity_types, prediction.label) orelse continue;
        var correct = false;
        for (ex.entities, 0..) |ent, gold_idx| {
            if (gold_matched[gold_idx]) continue;
            if (std.mem.eql(u8, ent.label, prediction.label) and ent.start == prediction.start and ent.end == prediction.end) {
                gold_matched[gold_idx] = true;
                correct = true;
                break;
            }
        }
        accum.addPrediction(pred_label_idx, correct);
    }
}

fn selectPredictions(
    allocator: std.mem.Allocator,
    predictions: []const adapter_eval.TopEntity,
    threshold: f32,
    opts: Options,
) ![]usize {
    var candidates = std.ArrayListUnmanaged(usize).empty;
    defer candidates.deinit(allocator);
    for (predictions, 0..) |prediction, idx| {
        if (prediction.score >= threshold) try candidates.append(allocator, idx);
    }
    std.mem.sort(usize, candidates.items, predictions, predictionBetterThan);
    if (opts.nms_overlap_threshold == null and !opts.best_span_per_label_start and opts.top_k_per_label == null and opts.max_predictions_per_example == null) return candidates.toOwnedSlice(allocator);

    var selected = std.ArrayListUnmanaged(usize).empty;
    errdefer selected.deinit(allocator);
    var per_label_counts = std.StringHashMapUnmanaged(usize){};
    defer per_label_counts.deinit(allocator);
    for (candidates.items) |candidate_idx| {
        const candidate = predictions[candidate_idx];
        if (opts.max_predictions_per_example) |max_predictions| {
            if (selected.items.len >= max_predictions) break;
        }
        if (opts.top_k_per_label) |top_k| {
            if ((per_label_counts.get(candidate.label) orelse 0) >= top_k) continue;
        }
        var suppressed = false;
        for (selected.items) |selected_idx| {
            const existing = predictions[selected_idx];
            if (!std.mem.eql(u8, candidate.label, existing.label)) continue;
            if (opts.best_span_per_label_start and candidate.start == existing.start) {
                suppressed = true;
                break;
            }
            if (opts.nms_overlap_threshold) |overlap_threshold| {
                if (spanOverlapRatio(candidate.start, candidate.end, existing.start, existing.end) > overlap_threshold) {
                    suppressed = true;
                    break;
                }
            }
        }
        if (!suppressed) {
            try selected.append(allocator, candidate_idx);
            if (opts.top_k_per_label != null) {
                const prior = per_label_counts.get(candidate.label) orelse 0;
                try per_label_counts.put(allocator, candidate.label, prior + 1);
            }
        }
    }
    return selected.toOwnedSlice(allocator);
}

fn predictionBetterThan(predictions: []const adapter_eval.TopEntity, lhs_idx: usize, rhs_idx: usize) bool {
    const lhs = predictions[lhs_idx];
    const rhs = predictions[rhs_idx];
    if (lhs.score != rhs.score) return lhs.score > rhs.score;
    const lhs_len = lhs.end - lhs.start;
    const rhs_len = rhs.end - rhs.start;
    if (lhs_len != rhs_len) return lhs_len < rhs_len;
    if (lhs.start != rhs.start) return lhs.start < rhs.start;
    return lhs.end < rhs.end;
}

fn spanOverlapRatio(a_start: usize, a_end: usize, b_start: usize, b_end: usize) f64 {
    if (a_end <= a_start or b_end <= b_start) return 0.0;
    const start = @max(a_start, b_start);
    const end = @min(a_end, b_end);
    if (end <= start) return 0.0;
    const intersection = end - start;
    const union_len = @max(a_end, b_end) - @min(a_start, b_start);
    if (union_len == 0) return 0.0;
    return @as(f64, @floatFromInt(intersection)) / @as(f64, @floatFromInt(union_len));
}

fn finishMetric(metric: *LabelMetric) void {
    metric.precision = ratio(metric.correct, metric.predicted);
    metric.recall = ratio(metric.correct, metric.gold);
    metric.f1 = f1Score(metric.precision, metric.recall);
}

fn ratio(num: usize, denom: usize) f64 {
    if (denom == 0) return 0.0;
    return @as(f64, @floatFromInt(num)) / @as(f64, @floatFromInt(denom));
}

fn f1Score(precision: f64, recall: f64) f64 {
    const denom = precision + recall;
    if (denom == 0.0) return 0.0;
    return 2.0 * precision * recall / denom;
}

fn parseOptions(args_in: std.process.Args, allocator: std.mem.Allocator) !?Options {
    var args = try std.process.Args.Iterator.initAllocator(args_in, allocator);
    defer args.deinit();
    _ = args.next();

    const model_dir = args.next() orelse return usageError();
    if (std.mem.eql(u8, model_dir, "--help") or std.mem.eql(u8, model_dir, "-h")) {
        printUsage();
        return null;
    }
    var opts = Options{
        .model_dir = model_dir,
        .adapter_dir = args.next() orelse return usageError(),
        .eval_data = args.next() orelse return usageError(),
        .entity_types_csv = args.next() orelse return usageError(),
    };
    var sweep_thresholds = std.ArrayListUnmanaged(f32).empty;
    errdefer sweep_thresholds.deinit(allocator);

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--seq-len")) {
            opts.seq_len = try parseUsize(args.next() orelse return usageError());
        } else if (std.mem.eql(u8, arg, "--max-span-width")) {
            opts.max_span_width = try parseUsize(args.next() orelse return usageError());
        } else if (std.mem.eql(u8, arg, "--objective")) {
            opts.objective = try parseObjective(args.next() orelse return usageError());
        } else if (std.mem.eql(u8, arg, "--max-examples")) {
            opts.max_examples = try parseUsize(args.next() orelse return usageError());
        } else if (std.mem.eql(u8, arg, "--min-prediction-score")) {
            opts.min_prediction_score = try std.fmt.parseFloat(f32, args.next() orelse return usageError());
        } else if (std.mem.eql(u8, arg, "--sweep-thresholds")) {
            try parseThresholdCsv(allocator, args.next() orelse return usageError(), &sweep_thresholds);
        } else if (std.mem.eql(u8, arg, "--nms-overlap")) {
            opts.nms_overlap_threshold = try std.fmt.parseFloat(f64, args.next() orelse return usageError());
        } else if (std.mem.eql(u8, arg, "--no-nms")) {
            opts.nms_overlap_threshold = null;
        } else if (std.mem.eql(u8, arg, "--max-predictions-per-example")) {
            opts.max_predictions_per_example = try parseUsize(args.next() orelse return usageError());
        } else if (std.mem.eql(u8, arg, "--top-k-per-label")) {
            opts.top_k_per_label = try parseUsize(args.next() orelse return usageError());
        } else if (std.mem.eql(u8, arg, "--best-span-per-label-start")) {
            opts.best_span_per_label_start = true;
        } else if (std.mem.eql(u8, arg, "--min-precision")) {
            opts.min_precision = try std.fmt.parseFloat(f64, args.next() orelse return usageError());
        } else if (std.mem.eql(u8, arg, "--min-recall")) {
            opts.min_recall = try std.fmt.parseFloat(f64, args.next() orelse return usageError());
        } else if (std.mem.eql(u8, arg, "--min-f1")) {
            opts.min_f1 = try std.fmt.parseFloat(f64, args.next() orelse return usageError());
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return null;
        } else {
            return usageError();
        }
    }
    if (opts.nms_overlap_threshold) |threshold| {
        if (!std.math.isFinite(threshold) or threshold < 0 or threshold > 1) return error.InvalidNmsOverlapThreshold;
    }
    opts.sweep_thresholds = try sweep_thresholds.toOwnedSlice(allocator);
    return opts;
}

fn parseUsize(value: []const u8) !usize {
    return std.fmt.parseUnsigned(usize, value, 10);
}

fn parseObjective(value: []const u8) !gliner2_autodiff.GlinerObjective {
    if (std.mem.eql(u8, value, "token")) return .token;
    if (std.mem.eql(u8, value, "span-start") or std.mem.eql(u8, value, "span_start")) return .span_start;
    return error.InvalidObjective;
}

fn objectiveName(objective: gliner2_autodiff.GlinerObjective) []const u8 {
    return switch (objective) {
        .token => "token",
        .span_start => "span-start",
    };
}

fn parseCsv(allocator: std.mem.Allocator, value: []const u8) ![][]const u8 {
    var out = std.ArrayListUnmanaged([]const u8).empty;
    errdefer {
        for (out.items) |item| allocator.free(item);
        out.deinit(allocator);
    }
    var iter = std.mem.splitScalar(u8, value, ',');
    while (iter.next()) |raw| {
        const item = std.mem.trim(u8, raw, " \t\r\n");
        if (item.len == 0) continue;
        try out.append(allocator, try allocator.dupe(u8, item));
    }
    if (out.items.len == 0) return error.EmptyCsv;
    return try out.toOwnedSlice(allocator);
}

fn parseThresholdCsv(allocator: std.mem.Allocator, value: []const u8, out: *std.ArrayListUnmanaged(f32)) !void {
    var iter = std.mem.splitScalar(u8, value, ',');
    while (iter.next()) |raw| {
        const item = std.mem.trim(u8, raw, " \t\r\n");
        if (item.len == 0) continue;
        const threshold = try std.fmt.parseFloat(f32, item);
        if (!std.math.isFinite(threshold) or threshold < 0 or threshold > 1) return error.InvalidQualityThreshold;
        try out.append(allocator, threshold);
    }
}

fn freeStringSlice(allocator: std.mem.Allocator, values: []const []const u8) void {
    for (values) |value| allocator.free(value);
    allocator.free(values);
}

fn indexOfLabel(labels: []const []const u8, label: []const u8) ?usize {
    for (labels, 0..) |candidate, idx| {
        if (std.mem.eql(u8, candidate, label)) return idx;
    }
    return null;
}

fn usageError() error{InvalidArguments} {
    printUsage();
    return error.InvalidArguments;
}

fn printUsage() void {
    std.debug.print(
        \\usage: eval-gliner2-autodiff-adapter-dataset <model_dir> <adapter_dir> <eval_jsonl_or_dir> <entity_types_csv> [options]
        \\
        \\Reports exact-match entity precision/recall/F1 for a saved GLiNER2 autodiff adapter.
        \\Evaluates all decoded entities above --min-prediction-score.
        \\
        \\options:
        \\  --seq-len N
        \\  --max-span-width N
        \\  --objective token|span-start
        \\  --max-examples N
        \\  --min-prediction-score FLOAT
        \\  --sweep-thresholds CSV
        \\  --nms-overlap FLOAT
        \\  --no-nms
        \\  --max-predictions-per-example N
        \\  --top-k-per-label N
        \\  --best-span-per-label-start
        \\  --min-precision FLOAT
        \\  --min-recall FLOAT
        \\  --min-f1 FLOAT
        \\
    , .{});
}
