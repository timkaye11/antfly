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

const gliner2 = termite.finetune.gliner2;
const gliner2_data = termite.finetune.gliner2_data;
const validation = termite.finetune.gliner2_run_validation;

const train_gliner2_autodiff = @import("train/train_gliner2_autodiff.zig");
const eval_gliner2_autodiff_adapter = @import("tools/eval_gliner2_autodiff_adapter.zig");
const materialize_gliner2_lora = @import("tools/materialize_gliner2_lora.zig");

const CommandMain = *const fn (std.process.Init) anyerror!void;

const Options = struct {
    model_dir: []const u8,
    train_data: []const u8,
    eval_data: []const u8,
    out_dir: []const u8,
    entity_types_csv: []const u8,

    epochs: []const u8 = "5",
    batch_size: []const u8 = "1",
    max_examples: []const u8 = "100",
    seq_len: []const u8 = "256",
    learning_rate: []const u8 = "1e-3",
    lora_rank: []const u8 = "16",
    lora_alpha: []const u8 = "32",
    objective: []const u8 = "span-start",
    max_span_width: []const u8 = "4",
    max_grad_norm: []const u8 = "1.0",
    grad_accum: []const u8 = "1",
    seed: []const u8 = "42",
    backend: []const u8 = "auto",
    compiled_required: bool = false,
    num_classes_override: ?[]const u8 = null,

    min_train_examples: usize = 100,
    min_eval_examples: usize = 20,
    min_total_entities: usize = 100,
    min_unique_labels: usize = 3,
    min_target_coverage_ratio: f64 = 0.95,
    min_positive_span_labels: usize = 100,
    min_positive_rate_per_label: f64 = 0.0,

    min_steps: ?usize = 100,
    min_supervised_tokens: ?usize = 1000,
    min_entity_tokens: ?usize = 100,
    min_supervised_tokens_per_second: ?f64 = null,
    max_avg_step_wall_ms: ?f64 = null,
    max_total_execute_ms: ?f64 = null,
    max_peak_resident_bytes: ?usize = null,
    max_device_resident_transfer_count: ?u64 = null,
    min_device_trainable_bytes: ?usize = null,
    require_loss_decrease: bool = true,

    eval_text: ?[]const u8 = null,
    expect_text: ?[]const u8 = null,
    expect_label: ?[]const u8 = null,
    min_score: ?[]const u8 = null,
    skip_semantic_eval: bool = false,

    materialized_dir: ?[]const u8 = null,
    dry_run: bool = false,
};

const ReadinessGateSummary = struct {
    model_dir: []const u8,
    train_data: []const u8,
    eval_data: []const u8,
    out_dir: []const u8,
    entity_types: []const []const u8,
    objective: []const u8,
    num_classes: usize,
    train_dataset: gliner2_data.DatasetReadinessSummary,
    eval_dataset: gliner2_data.DatasetReadinessSummary,
    run_validation: validation.RunValidationSummary,
    lora_inspection: gliner2.LoRABundleInspectionSummary,
    semantic_eval_required: bool,
    materialized_dir: ?[]const u8,
    status: []const u8 = "passed",
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const opts = try parseOptions(&args) orelse return;
    try runReadiness(init, allocator, opts);
}

fn runReadiness(init: std.process.Init, allocator: std.mem.Allocator, opts: Options) !void {
    if (opts.dry_run) {
        try printDryRun(init, opts);
        return;
    }

    const entity_types = try parseCsvOwned(allocator, opts.entity_types_csv);
    defer freeStringList(allocator, entity_types);
    if (entity_types.len == 0) return error.NoEntityTypesProvided;

    var train_loaded = try gliner2_data.loadExamples(allocator, opts.train_data, null);
    defer train_loaded.deinit();
    var eval_loaded = try gliner2_data.loadExamples(allocator, opts.eval_data, null);
    defer eval_loaded.deinit();

    const seq_len = try std.fmt.parseUnsigned(usize, opts.seq_len, 10);
    const max_span_width = try std.fmt.parseUnsigned(usize, opts.max_span_width, 10);
    const batch_size = try std.fmt.parseUnsigned(usize, opts.batch_size, 10);

    var train_readiness = try gliner2_data.evaluateDatasetReadiness(
        allocator,
        train_loaded.examples,
        entity_types,
        seq_len,
        max_span_width,
        batch_size,
        .{
            .min_examples = opts.min_train_examples,
            .min_total_entities = opts.min_total_entities,
            .min_unique_labels = opts.min_unique_labels,
            .min_target_entities = opts.min_total_entities,
            .min_target_coverage_ratio = opts.min_target_coverage_ratio,
            .require_all_examples_with_target = false,
            .min_positive_span_labels = opts.min_positive_span_labels,
            .min_positive_rate_per_label = opts.min_positive_rate_per_label,
        },
    );
    errdefer gliner2_data.freeDatasetReadinessSummary(allocator, &train_readiness);
    if (!train_readiness.passed) return error.TrainDatasetReadinessFailed;

    var eval_readiness = try gliner2_data.evaluateDatasetReadiness(
        allocator,
        eval_loaded.examples,
        entity_types,
        seq_len,
        max_span_width,
        batch_size,
        .{
            .min_examples = opts.min_eval_examples,
            .min_total_entities = @min(opts.min_total_entities, opts.min_eval_examples),
            .min_unique_labels = opts.min_unique_labels,
            .min_target_entities = @min(opts.min_total_entities, opts.min_eval_examples),
            .min_target_coverage_ratio = opts.min_target_coverage_ratio,
            .require_all_examples_with_target = false,
            .min_positive_span_labels = @min(opts.min_positive_span_labels, opts.min_eval_examples),
            .min_positive_rate_per_label = opts.min_positive_rate_per_label,
        },
    );
    errdefer gliner2_data.freeDatasetReadinessSummary(allocator, &eval_readiness);
    if (!eval_readiness.passed) return error.EvalDatasetReadinessFailed;

    const num_classes = try resolveNumClasses(allocator, train_loaded.examples, entity_types, opts.num_classes_override);
    var num_classes_buf: [32]u8 = undefined;
    const num_classes_arg = try std.fmt.bufPrint(&num_classes_buf, "{d}", .{num_classes});

    const train_args = [_][]const u8{
        "--model-dir",      opts.model_dir,
        "--train-data",     opts.train_data,
        "--out-dir",        opts.out_dir,
        "--epochs",         opts.epochs,
        "--batch-size",     opts.batch_size,
        "--max-examples",   opts.max_examples,
        "--seq-len",        opts.seq_len,
        "--num-classes",    num_classes_arg,
        "--learning-rate",  opts.learning_rate,
        "--lora-rank",      opts.lora_rank,
        "--lora-alpha",     opts.lora_alpha,
        "--objective",      opts.objective,
        "--max-span-width", opts.max_span_width,
        "--max-grad-norm",  opts.max_grad_norm,
        "--grad-accum",     opts.grad_accum,
        "--seed",           opts.seed,
        "--backend",        opts.backend,
    };
    var train_args_list = std.ArrayListUnmanaged([]const u8).empty;
    defer train_args_list.deinit(allocator);
    try train_args_list.appendSlice(allocator, &train_args);
    if (opts.compiled_required) try train_args_list.append(allocator, "--compiled-required");
    try runCommand(init, allocator, "train-gliner2-autodiff", train_gliner2_autodiff.main, train_args_list.items);

    const metal_required = std.mem.eql(u8, opts.backend, "metal") or opts.compiled_required;
    const min_device_trainable_bytes: ?usize = opts.min_device_trainable_bytes orelse if (metal_required) @as(usize, 1) else null;
    var run_summary = try validation.validateRun(allocator, opts.out_dir, .{
        .require_loss_decrease = opts.require_loss_decrease,
        .min_supervised_tokens_per_second = opts.min_supervised_tokens_per_second,
        .max_avg_step_wall_ms = opts.max_avg_step_wall_ms,
        .max_total_execute_ms = opts.max_total_execute_ms,
        .max_peak_resident_bytes = opts.max_peak_resident_bytes,
        .min_examples = opts.min_train_examples,
        .min_steps = opts.min_steps,
        .min_entity_labels = opts.min_unique_labels,
        .min_supervised_tokens = opts.min_supervised_tokens,
        .min_entity_tokens = opts.min_entity_tokens,
        .require_backend = if (std.mem.eql(u8, opts.backend, "metal")) "Metal" else null,
        .require_optimizer_backend = if (metal_required) "metal" else null,
        .max_device_resident_transfer_count = opts.max_device_resident_transfer_count,
        .min_device_trainable_bytes = min_device_trainable_bytes,
    });
    errdefer validation.freeRunValidationSummary(allocator, &run_summary);

    var lora_summary = try gliner2.inspectLoRABundle(allocator, opts.model_dir, opts.out_dir);
    errdefer gliner2.freeLoRABundleInspectionSummary(allocator, &lora_summary);
    if (lora_summary.resolved_tensor_count == 0) return error.NoPeftAdapterTensors;

    const semantic_required = !opts.skip_semantic_eval;
    if (semantic_required) {
        const eval_text = opts.eval_text orelse return error.MissingSemanticEvalText;
        const expect_label = opts.expect_label orelse return error.MissingSemanticExpectedLabel;
        const min_score = opts.min_score orelse return error.MissingSemanticMinScore;
        var eval_args_list = std.ArrayListUnmanaged([]const u8).empty;
        defer eval_args_list.deinit(allocator);
        try eval_args_list.appendSlice(allocator, &.{
            opts.model_dir,
            opts.out_dir,
            eval_text,
            opts.entity_types_csv,
            "--seq-len",
            opts.seq_len,
            "--max-span-width",
            opts.max_span_width,
            "--objective",
            opts.objective,
            "--expect-label",
            expect_label,
            "--min-score",
            min_score,
        });
        if (opts.expect_text) |expect_text| {
            try eval_args_list.appendSlice(allocator, &.{ "--expect-text", expect_text });
        }
        try runCommand(init, allocator, "eval-gliner2-autodiff-adapter", eval_gliner2_autodiff_adapter.main, eval_args_list.items);
    }

    if (opts.materialized_dir) |materialized_dir| {
        const materialize_args = [_][]const u8{ opts.model_dir, opts.out_dir, materialized_dir };
        try runCommand(init, allocator, "materialize-gliner2-lora", materialize_gliner2_lora.main, &materialize_args);
    }

    const report = ReadinessGateSummary{
        .model_dir = opts.model_dir,
        .train_data = opts.train_data,
        .eval_data = opts.eval_data,
        .out_dir = opts.out_dir,
        .entity_types = entity_types,
        .objective = opts.objective,
        .num_classes = num_classes,
        .train_dataset = train_readiness,
        .eval_dataset = eval_readiness,
        .run_validation = run_summary,
        .lora_inspection = lora_summary,
        .semantic_eval_required = semantic_required,
        .materialized_dir = opts.materialized_dir,
    };
    try printJson(init, report);

    gliner2.freeLoRABundleInspectionSummary(allocator, &lora_summary);
    validation.freeRunValidationSummary(allocator, &run_summary);
    gliner2_data.freeDatasetReadinessSummary(allocator, &eval_readiness);
    gliner2_data.freeDatasetReadinessSummary(allocator, &train_readiness);
}

fn resolveNumClasses(
    allocator: std.mem.Allocator,
    examples: []const gliner2_data.Example,
    entity_types: []const []const u8,
    override: ?[]const u8,
) !usize {
    if (override) |value| return try std.fmt.parseUnsigned(usize, value, 10);
    const vocab = try gliner2_data.buildLabelVocab(allocator, examples, null);
    defer {
        for (vocab) |label| allocator.free(label);
        allocator.free(vocab);
    }
    return @max(vocab.len, entity_types.len) + 1;
}

fn runCommand(init: std.process.Init, allocator: std.mem.Allocator, argv0: []const u8, main_fn: CommandMain, args: []const []const u8) !void {
    printCommand(argv0, args);

    var owned = try allocator.alloc([:0]u8, args.len + 1);
    defer {
        for (owned) |arg| allocator.free(arg);
        allocator.free(owned);
    }
    var vector = try allocator.alloc([*:0]const u8, args.len + 1);
    defer allocator.free(vector);

    owned[0] = try allocator.dupeZ(u8, argv0);
    vector[0] = owned[0].ptr;
    for (args, 0..) |arg, idx| {
        owned[idx + 1] = try allocator.dupeZ(u8, arg);
        vector[idx + 1] = owned[idx + 1].ptr;
    }

    var command_init = init;
    command_init.minimal.args = .{ .vector = vector };
    try main_fn(command_init);
}

fn printCommand(argv0: []const u8, args: []const []const u8) void {
    std.debug.print("+ {s}", .{argv0});
    for (args) |arg| std.debug.print(" {s}", .{arg});
    std.debug.print("\n", .{});
}

fn parseCsvOwned(allocator: std.mem.Allocator, csv: []const u8) ![][]const u8 {
    var out = std.ArrayListUnmanaged([]const u8).empty;
    errdefer {
        for (out.items) |item| allocator.free(item);
        out.deinit(allocator);
    }
    var it = std.mem.splitScalar(u8, csv, ',');
    while (it.next()) |raw| {
        const trimmed = std.mem.trim(u8, raw, " \t\r\n");
        if (trimmed.len == 0) continue;
        try out.append(allocator, try allocator.dupe(u8, trimmed));
    }
    return out.toOwnedSlice(allocator);
}

fn freeStringList(allocator: std.mem.Allocator, values: []const []const u8) void {
    for (values) |value| allocator.free(value);
    allocator.free(values);
}

fn printDryRun(init: std.process.Init, opts: Options) !void {
    const stdout = std.Io.File.stdout();
    var buf: [4096]u8 = undefined;
    var writer = stdout.writer(init.io, &buf);
    try writer.interface.print(
        \\gliner2_production_readiness_dry_run: true
        \\model_dir: {s}
        \\train_data: {s}
        \\eval_data: {s}
        \\out_dir: {s}
        \\entity_types: {s}
        \\objective: {s}
        \\semantic_eval_required: {}
        \\
    , .{
        opts.model_dir,
        opts.train_data,
        opts.eval_data,
        opts.out_dir,
        opts.entity_types_csv,
        opts.objective,
        !opts.skip_semantic_eval,
    });
    try writer.interface.flush();
}

fn printJson(init: std.process.Init, value: anytype) !void {
    const stdout = std.Io.File.stdout();
    var buf: [32768]u8 = undefined;
    var writer = stdout.writer(init.io, &buf);
    try std.json.Stringify.value(value, .{ .whitespace = .indent_2 }, &writer.interface);
    try writer.interface.writeByte('\n');
    try writer.interface.flush();
}

fn parseOptions(args: *std.process.Args.Iterator) !?Options {
    const model_dir = args.next() orelse return usageError();
    if (std.mem.eql(u8, model_dir, "--help") or std.mem.eql(u8, model_dir, "-h")) {
        printUsage();
        return null;
    }

    var opts = Options{
        .model_dir = model_dir,
        .train_data = args.next() orelse return usageError(),
        .eval_data = args.next() orelse return usageError(),
        .out_dir = args.next() orelse return usageError(),
        .entity_types_csv = args.next() orelse return usageError(),
    };

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--epochs")) {
            opts.epochs = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--batch-size")) {
            opts.batch_size = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--max-examples")) {
            opts.max_examples = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--seq-len")) {
            opts.seq_len = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--learning-rate") or std.mem.eql(u8, arg, "--lr")) {
            opts.learning_rate = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--lora-rank")) {
            opts.lora_rank = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--lora-alpha")) {
            opts.lora_alpha = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--objective")) {
            opts.objective = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--max-span-width")) {
            opts.max_span_width = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--max-grad-norm")) {
            opts.max_grad_norm = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--grad-accum")) {
            opts.grad_accum = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--seed")) {
            opts.seed = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--backend")) {
            opts.backend = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--compiled-required")) {
            opts.compiled_required = true;
        } else if (std.mem.eql(u8, arg, "--num-classes")) {
            opts.num_classes_override = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--min-train-examples")) {
            opts.min_train_examples = try parseUsizeArg(args, arg);
        } else if (std.mem.eql(u8, arg, "--min-eval-examples")) {
            opts.min_eval_examples = try parseUsizeArg(args, arg);
        } else if (std.mem.eql(u8, arg, "--min-total-entities")) {
            opts.min_total_entities = try parseUsizeArg(args, arg);
        } else if (std.mem.eql(u8, arg, "--min-unique-labels")) {
            opts.min_unique_labels = try parseUsizeArg(args, arg);
        } else if (std.mem.eql(u8, arg, "--min-target-coverage-ratio")) {
            opts.min_target_coverage_ratio = try parseF64Arg(args, arg);
        } else if (std.mem.eql(u8, arg, "--min-positive-span-labels")) {
            opts.min_positive_span_labels = try parseUsizeArg(args, arg);
        } else if (std.mem.eql(u8, arg, "--min-positive-rate-per-label")) {
            opts.min_positive_rate_per_label = try parseF64Arg(args, arg);
        } else if (std.mem.eql(u8, arg, "--min-steps")) {
            opts.min_steps = try parseUsizeArg(args, arg);
        } else if (std.mem.eql(u8, arg, "--min-supervised-tokens")) {
            opts.min_supervised_tokens = try parseUsizeArg(args, arg);
        } else if (std.mem.eql(u8, arg, "--min-entity-tokens")) {
            opts.min_entity_tokens = try parseUsizeArg(args, arg);
        } else if (std.mem.eql(u8, arg, "--min-supervised-tokens-per-second")) {
            opts.min_supervised_tokens_per_second = try parseF64Arg(args, arg);
        } else if (std.mem.eql(u8, arg, "--max-avg-step-wall-ms")) {
            opts.max_avg_step_wall_ms = try parseF64Arg(args, arg);
        } else if (std.mem.eql(u8, arg, "--max-total-execute-ms")) {
            opts.max_total_execute_ms = try parseF64Arg(args, arg);
        } else if (std.mem.eql(u8, arg, "--max-peak-resident-bytes")) {
            opts.max_peak_resident_bytes = try parseUsizeArg(args, arg);
        } else if (std.mem.eql(u8, arg, "--max-device-resident-transfer-count")) {
            opts.max_device_resident_transfer_count = try parseU64Arg(args, arg);
        } else if (std.mem.eql(u8, arg, "--min-device-trainable-bytes")) {
            opts.min_device_trainable_bytes = try parseUsizeArg(args, arg);
        } else if (std.mem.eql(u8, arg, "--allow-flat-loss")) {
            opts.require_loss_decrease = false;
        } else if (std.mem.eql(u8, arg, "--eval-text")) {
            opts.eval_text = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--expect-text")) {
            opts.expect_text = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--expect-label")) {
            opts.expect_label = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--min-score")) {
            opts.min_score = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--skip-semantic-eval")) {
            opts.skip_semantic_eval = true;
        } else if (std.mem.eql(u8, arg, "--materialized-dir")) {
            opts.materialized_dir = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--dry-run")) {
            opts.dry_run = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return null;
        } else {
            return usageError();
        }
    }
    return opts;
}

fn parseUsizeArg(args: *std.process.Args.Iterator, name: []const u8) !usize {
    return std.fmt.parseUnsigned(usize, args.next() orelse {
        std.debug.print("error: missing value for {s}\n", .{name});
        return error.InvalidArguments;
    }, 10);
}

fn parseU64Arg(args: *std.process.Args.Iterator, name: []const u8) !u64 {
    return std.fmt.parseUnsigned(u64, args.next() orelse {
        std.debug.print("error: missing value for {s}\n", .{name});
        return error.InvalidArguments;
    }, 10);
}

fn parseF64Arg(args: *std.process.Args.Iterator, name: []const u8) !f64 {
    return std.fmt.parseFloat(f64, args.next() orelse {
        std.debug.print("error: missing value for {s}\n", .{name});
        return error.InvalidArguments;
    });
}

fn usageError() error{InvalidArguments} {
    printUsage();
    return error.InvalidArguments;
}

fn printUsage() void {
    std.debug.print(
        \\usage: gliner2-production-readiness <model_dir> <train_jsonl_or_dir> <eval_jsonl_or_dir> <out_dir> <entity_types_csv> [options]
        \\
        \\Runs the production-readiness gate:
        \\  dataset readiness -> train-gliner2-autodiff -> validate run artifacts -> inspect bundle -> semantic eval -> optional materialization.
        \\
        \\Required semantic options unless --skip-semantic-eval is set:
        \\  --eval-text TEXT
        \\  --expect-label LABEL
        \\  --min-score FLOAT
        \\
        \\Common options:
        \\  --objective token|span-start      Training/eval objective (default: span-start)
        \\  --epochs N                       Training epochs (default: 5)
        \\  --max-examples N                 Training cap (default: 100)
        \\  --seq-len N                      Sequence length (default: 256)
        \\  --batch-size N                   Batch size (default: 1)
        \\  --learning-rate FLOAT            Learning rate (default: 1e-3)
        \\  --backend auto|metal|mlx|native  Training backend (default: auto)
        \\  --compiled-required              Fail if requested compiled backend falls back
        \\  --materialized-dir DIR           Also materialize merged model artifacts
        \\  --dry-run                        Print the gate shape without touching model/data files
        \\
        \\Production threshold options:
        \\  --min-train-examples N
        \\  --min-eval-examples N
        \\  --min-total-entities N
        \\  --min-unique-labels N
        \\  --min-target-coverage-ratio FLOAT
        \\  --min-positive-span-labels N
        \\  --min-steps N
        \\  --min-supervised-tokens N
        \\  --min-entity-tokens N
        \\  --min-supervised-tokens-per-second FLOAT
        \\  --max-avg-step-wall-ms FLOAT
        \\  --max-total-execute-ms FLOAT
        \\  --max-peak-resident-bytes N
        \\  --max-device-resident-transfer-count N
        \\  --min-device-trainable-bytes N
        \\
    , .{});
}
