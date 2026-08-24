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
const platform = @import("antfly_platform");
const ml = @import("ml");
const finetune = @import("gemma4.zig");
const gemma4_real = @import("gemma4_real_autodiff.zig");
const gemma4_mm_real = @import("gemma4_multimodal_real_autodiff.zig");
const real_autodiff = @import("real_autodiff_trainer.zig");
const safetensors_checkpoint = @import("safetensors_checkpoint.zig");
const graph_bridge = @import("graph_bridge.zig");
const gemma_graph = @import("../architectures/gemma_graph.zig");
const manifest_mod = @import("../models/manifest.zig");
const safetensors = @import("../models/safetensors.zig");
const build_options = @import("build_options");
const run_contract = @import("../run/contract.zig");
const artifact_writer = @import("../run/artifact_writer.zig");
const artifact_publication = @import("artifact_publication.zig");
const ops_mod = @import("../ops/ops.zig");
const ComputeBackend = ops_mod.ComputeBackend;
const pjrt_mod = if (build_options.enable_pjrt) @import("pjrt") else struct {
    pub const pjrt = struct {
        pub const Client = void;
    };
};

const TrainerMode = enum { auto, surrogate, autodiff };

pub const AutodiffCliInvocation = struct {
    base_model_dir: []const u8,
    adapter_dir: []const u8,
    train_prepared_path: []const u8,
    eval_prepared_path: []const u8,
    output_dir: []const u8,
    backend: []const u8,
    max_examples: []const u8,
    eval_max_examples: []const u8,
    epochs: []const u8,
    learning_rate: []const u8,
    gguf_projector_path: ?[]const u8 = null,
};

pub fn appendAutodiffCliArgs(
    allocator: std.mem.Allocator,
    command: *std.ArrayListUnmanaged([]const u8),
    invocation: AutodiffCliInvocation,
) !void {
    try command.appendSlice(allocator, &.{
        invocation.base_model_dir,
        invocation.adapter_dir,
        invocation.train_prepared_path,
        invocation.output_dir,
        "--trainer",
        "autodiff",
        "--backend",
        invocation.backend,
        "--max-examples",
        invocation.max_examples,
        "--eval-prepared",
        invocation.eval_prepared_path,
        "--eval-max-examples",
        invocation.eval_max_examples,
        "--epochs",
        invocation.epochs,
        "--lr",
        invocation.learning_rate,
        "--max-grad-norm",
        "1.0",
        "--grad-accum",
        "1",
    });
    if (invocation.gguf_projector_path) |path| try command.appendSlice(allocator, &.{ "--gguf-projector", path });
}

const AutodiffEpochSummary = struct {
    examples_seen: usize = 0,
    supervised_tokens_seen: usize = 0,
    teacher_examples_seen: usize = 0,
    teacher_supervised_tokens_seen: usize = 0,
    mean_teacher_temperature: f64 = 0,
    average_loss: f64 = 0,
    mean_grad_norm: f64 = 0,
    optimizer_steps: usize = 0,
    graph_executor_steps: u64 = 0,
    graph_executor_fallback_steps: u64 = 0,
    graph_executor_partitions: u64 = 0,
    graph_executor_command_dispatches: u64 = 0,
    graph_executor_native_partitions: u64 = 0,
    graph_executor_unsupported_ops: u64 = 0,
    graph_executor_interpreter_fallbacks: u64 = 0,
    graph_executor_runtime_region_dispatches: u64 = 0,
    graph_executor_true_host_outputs: u64 = 0,
    metal_optimizer_steps: u64 = 0,
};

const CliOptions = struct {
    learning_rate: f32 = 0.001,
    max_examples: usize = 32,
    eval_max_examples: usize = 0,
    epochs: usize = 1,
    layer_name: ?[]const u8 = null,
    max_grad_norm: f32 = 1.0,
    grad_accum_steps: u32 = 1,
    activation_checkpoint_interval: u32 = 0,
    llrd_decay: f32 = 1.0,
    use_schedule_free: bool = false,
    trainer_mode: TrainerMode = .autodiff,
    backend_kind: ?gemma4_real.BackendKind = null,
    gguf_projector_path: ?[]const u8 = null,
    eval_prepared_inputs_path: ?[]const u8 = null,

    fn effectiveEvalMaxExamples(self: CliOptions) usize {
        return if (self.eval_max_examples > 0) self.eval_max_examples else self.max_examples;
    }
};

const AutodiffExecutionPolicy = struct {
    engine: real_autodiff.TrainingExecutionEngine,
    compiled_required: bool,
    strict_metal_execution: bool,
};

fn autodiffExecutionPolicy(backend_kind: gemma4_real.BackendKind) AutodiffExecutionPolicy {
    const metal = backend_kind == .metal;
    return .{
        .engine = if (metal) .compiled_device else .interpreter,
        .compiled_required = metal,
        .strict_metal_execution = metal,
    };
}

const MultimodalPreparedStats = struct {
    examples_with_media: usize = 0,
    total_image_inputs: usize = 0,
    total_audio_inputs: usize = 0,
    total_image_soft_tokens: usize = 0,
    total_audio_soft_tokens: usize = 0,
};

test "gemma4 autodiff CLI invocation wires a distinct heldout prepared artifact" {
    const allocator = std.testing.allocator;
    var command = std.ArrayListUnmanaged([]const u8).empty;
    defer command.deinit(allocator);
    try appendAutodiffCliArgs(allocator, &command, .{
        .base_model_dir = "/base",
        .adapter_dir = "/adapter",
        .train_prepared_path = "/run/prepared.json",
        .eval_prepared_path = "/run/prepared.eval.json",
        .output_dir = "/run/train_out_native",
        .backend = "native",
        .max_examples = "8",
        .eval_max_examples = "2",
        .epochs = "1",
        .learning_rate = "0.0003",
    });

    var maybe_eval_flag: ?usize = null;
    for (command.items, 0..) |arg, idx| {
        if (std.mem.eql(u8, arg, "--eval-prepared")) {
            maybe_eval_flag = idx;
            break;
        }
    }
    const eval_flag = maybe_eval_flag orelse return error.MissingEvaluationPreparedInputs;
    try std.testing.expect(eval_flag + 1 < command.items.len);
    try std.testing.expectEqualStrings("/run/prepared.eval.json", command.items[eval_flag + 1]);
    try std.testing.expect(!std.mem.eql(u8, command.items[2], command.items[eval_flag + 1]));
}

const ReportContext = struct {
    prepared_inputs_path: []const u8,
    learning_rate: f32,
    max_examples: usize,
    eval_max_examples: usize,
    epochs: usize,
    layer_name: ?[]const u8,
    max_grad_norm: f32,
    grad_accum_steps: u32,
    activation_checkpoint_interval: u32,
    llrd_decay: f32,
    use_schedule_free: bool,
    backend_label: []const u8,
};

const ImmutableRunPublication = artifact_publication.ImmutableDirectoryPublication;

test "gemma4 whole-run publication is immutable and cleans owned staging" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const cwd = std.Io.Dir.cwd();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const existing_dir = try std.fs.path.join(allocator, &.{ root, "existing" });
    defer allocator.free(existing_dir);
    try cwd.createDirPath(io, existing_dir);
    const sentinel_path = try std.fs.path.join(allocator, &.{ existing_dir, "sentinel.txt" });
    defer allocator.free(sentinel_path);
    try cwd.writeFile(io, .{ .sub_path = sentinel_path, .data = "known-good" });

    try std.testing.expectError(
        error.Gemma4RunOutputAlreadyExists,
        ImmutableRunPublication.init(allocator, io, existing_dir),
    );
    const sentinel = try cwd.readFileAlloc(io, sentinel_path, allocator, .limited(32));
    defer allocator.free(sentinel);
    try std.testing.expectEqualStrings("known-good", sentinel);

    const failed_dir = try std.fs.path.join(allocator, &.{ root, "failed" });
    defer allocator.free(failed_dir);
    var failed_staging_copy: []u8 = undefined;
    {
        var failed = try ImmutableRunPublication.init(allocator, io, failed_dir);
        defer failed.deinit();
        failed_staging_copy = try allocator.dupe(u8, failed.staging_dir);
        try cwd.createDirPath(io, failed.staging_dir);
        failed.claimStaging();
        const partial_path = try std.fs.path.join(allocator, &.{ failed.staging_dir, "partial.json" });
        defer allocator.free(partial_path);
        try cwd.writeFile(io, .{ .sub_path = partial_path, .data = "partial" });
    }
    defer allocator.free(failed_staging_copy);
    try std.testing.expectError(error.FileNotFound, cwd.access(io, failed_staging_copy, .{}));
    try std.testing.expectError(error.FileNotFound, cwd.access(io, failed_dir, .{}));

    const fresh_dir = try std.fs.path.join(allocator, &.{ root, "fresh" });
    defer allocator.free(fresh_dir);
    var fresh = try ImmutableRunPublication.init(allocator, io, fresh_dir);
    defer fresh.deinit();
    try cwd.createDirPath(io, fresh.staging_dir);
    fresh.claimStaging();
    const complete_path = try std.fs.path.join(allocator, &.{ fresh.staging_dir, "training_report.json" });
    defer allocator.free(complete_path);
    try cwd.writeFile(io, .{ .sub_path = complete_path, .data = "complete" });
    try fresh.publish();

    const published_path = try std.fs.path.join(allocator, &.{ fresh_dir, "training_report.json" });
    defer allocator.free(published_path);
    const published = try cwd.readFileAlloc(io, published_path, allocator, .limited(32));
    defer allocator.free(published);
    try std.testing.expectEqualStrings("complete", published);

    const late_dir = try std.fs.path.join(allocator, &.{ root, "late-collision" });
    defer allocator.free(late_dir);
    var late = try ImmutableRunPublication.init(allocator, io, late_dir);
    defer late.deinit();
    try cwd.createDirPath(io, late.staging_dir);
    late.claimStaging();
    try cwd.createDirPath(io, late_dir);
    const late_sentinel_path = try std.fs.path.join(allocator, &.{ late_dir, "sentinel.txt" });
    defer allocator.free(late_sentinel_path);
    try cwd.writeFile(io, .{ .sub_path = late_sentinel_path, .data = "late-winner" });
    try std.testing.expectError(error.Gemma4RunOutputAlreadyExists, late.publish());
    const late_sentinel = try cwd.readFileAlloc(io, late_sentinel_path, allocator, .limited(32));
    defer allocator.free(late_sentinel);
    try std.testing.expectEqualStrings("late-winner", late_sentinel);
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    var argv: std.ArrayListUnmanaged([]const u8) = .empty;
    defer argv.deinit(allocator);
    while (args.next()) |arg| try argv.append(allocator, arg);
    try runFromArgs(allocator, init.io, argv.items);
}

pub fn runFromArgs(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !void {
    if (argv.len < 4) return usageError();

    const base_model_dir = argv[0];
    const adapter_model_dir = argv[1];
    const prepared_inputs_path = argv[2];
    const out_dir = argv[3];

    var opts = CliOptions{};
    var positional_count: usize = 0;
    var i: usize = 4;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--lr") or std.mem.eql(u8, arg, "--learning-rate")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.learning_rate = try std.fmt.parseFloat(f32, argv[i]);
        } else if (std.mem.eql(u8, arg, "--max-examples")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.max_examples = try std.fmt.parseUnsigned(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--eval-max-examples")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.eval_max_examples = try std.fmt.parseUnsigned(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--eval-prepared")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.eval_prepared_inputs_path = argv[i];
        } else if (std.mem.eql(u8, arg, "--epochs")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.epochs = try std.fmt.parseUnsigned(usize, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--layer-name") or std.mem.eql(u8, arg, "--layer")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.layer_name = argv[i];
        } else if (std.mem.eql(u8, arg, "--max-grad-norm")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.max_grad_norm = try std.fmt.parseFloat(f32, argv[i]);
        } else if (std.mem.eql(u8, arg, "--grad-accum")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.grad_accum_steps = try std.fmt.parseUnsigned(u32, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--activation-checkpoint-interval")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.activation_checkpoint_interval = try std.fmt.parseUnsigned(u32, argv[i], 10);
            if (opts.activation_checkpoint_interval == 0) return usageError();
        } else if (std.mem.eql(u8, arg, "--llrd-decay")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.llrd_decay = try std.fmt.parseFloat(f32, argv[i]);
        } else if (std.mem.eql(u8, arg, "--schedule-free")) {
            opts.use_schedule_free = true;
        } else if (std.mem.eql(u8, arg, "--backend")) {
            i += 1;
            if (i >= argv.len) return usageError();
            const val = argv[i];
            if (std.mem.eql(u8, val, "native")) {
                opts.backend_kind = .native;
            } else if (std.mem.eql(u8, val, "metal")) {
                opts.backend_kind = .metal;
            } else return usageError();
        } else if (std.mem.eql(u8, arg, "--trainer")) {
            i += 1;
            if (i >= argv.len) return usageError();
            const val = argv[i];
            if (std.mem.eql(u8, val, "auto")) {
                opts.trainer_mode = .auto;
            } else if (std.mem.eql(u8, val, "surrogate")) {
                opts.trainer_mode = .surrogate;
            } else if (std.mem.eql(u8, val, "autodiff")) {
                opts.trainer_mode = .autodiff;
            } else return usageError();
        } else if (std.mem.eql(u8, arg, "--gguf-projector")) {
            i += 1;
            if (i >= argv.len) return usageError();
            opts.gguf_projector_path = argv[i];
        } else {
            switch (positional_count) {
                0 => opts.learning_rate = try std.fmt.parseFloat(f32, arg),
                1 => opts.max_examples = try std.fmt.parseUnsigned(usize, arg, 10),
                2 => opts.epochs = try std.fmt.parseUnsigned(usize, arg, 10),
                3 => opts.layer_name = arg,
                else => return usageError(),
            }
            positional_count += 1;
        }
    }

    const actual_mode = resolveTrainerMode(opts.trainer_mode);
    if (actual_mode == .autodiff and opts.backend_kind == null) return error.MissingBackend;
    if (actual_mode == .autodiff) {
        if (opts.eval_prepared_inputs_path == null) return error.MissingEvaluationPreparedInputs;
        try validateAutodiffTrainingOptions(opts);
        try validateMetalGraphExecutorEnvironment(opts.backend_kind.?);
        try validateAutodiffBaseArtifact(allocator, base_model_dir, opts.backend_kind.?);
    }

    var prepared = try finetune.loadPreparedInputsSummary(allocator, prepared_inputs_path);
    defer finetune.freePreparedInputsSummary(allocator, &prepared);

    var eval_prepared: ?finetune.PreparedInputsSummary = null;
    defer if (eval_prepared) |*summary| finetune.freePreparedInputsSummary(allocator, summary);
    if (actual_mode == .autodiff) {
        eval_prepared = try finetune.loadPreparedInputsSummary(allocator, opts.eval_prepared_inputs_path.?);
    }

    switch (actual_mode) {
        .surrogate => try runSurrogate(io, allocator, base_model_dir, adapter_model_dir, prepared_inputs_path, out_dir, prepared, opts),
        .autodiff => try runAutodiff(io, allocator, base_model_dir, adapter_model_dir, prepared_inputs_path, opts.eval_prepared_inputs_path.?, out_dir, prepared, eval_prepared.?, opts),
        .auto => unreachable,
    }
}

fn resolveTrainerMode(requested: TrainerMode) TrainerMode {
    return switch (requested) {
        .surrogate, .autodiff => requested,
        // `auto` is retained as a compatibility spelling, but production
        // training must never turn a missing real model into a successful
        // surrogate run.
        .auto => .autodiff,
    };
}

fn validateAutodiffTrainingOptions(opts: CliOptions) !void {
    if (!std.math.isFinite(opts.learning_rate) or opts.learning_rate <= 0) return error.InvalidLearningRate;
    if (opts.epochs == 0) return error.InvalidEpochCount;
    if (opts.grad_accum_steps == 0) return error.InvalidGradientAccumulation;
    if (!std.math.isFinite(opts.max_grad_norm) or opts.max_grad_norm < 0) return error.InvalidMaxGradNorm;
    if (opts.layer_name != null) return error.LayerScopedAutodiffNotYetSupported;
    if (!std.math.approxEqAbs(f32, opts.llrd_decay, 1.0, 1e-6)) return error.LayerWiseDecayNotYetSupportedForAutodiff;
    if (opts.use_schedule_free) return error.ScheduleFreeNotYetSupportedForAutodiff;
}

fn validateMetalGraphExecutorEnvironment(backend_kind: gemma4_real.BackendKind) !void {
    if (backend_kind != .metal) return;
    try validateMetalGraphExecutorFlags(
        platform.env.getenvBoolDefault("TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR", false),
        platform.env.getenvBoolDefault("TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR", false),
        platform.env.getenv("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_NODE_IDS") != null,
        platform.env.getenvBoolDefault("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK", false),
        platform.env.getenvBoolDefault("TERMITE_DEBUG_DEVICE_GRAD_NORM", false),
    );
}

fn validateMetalGraphExecutorFlags(
    enabled: bool,
    disabled: bool,
    parity_node_ids: bool,
    parity_check: bool,
    debug_device_grad_norm: bool,
) !void {
    if (!enabled) return error.Gemma4MetalTrainingGraphExecutorRequired;
    if (disabled) return error.Gemma4MetalTrainingGraphExecutorDisabled;
    if (parity_node_ids) return error.Gemma4MetalTrainingParityDiagnosticForbidden;
    if (parity_check) return error.Gemma4MetalTrainingParityCheckForbidden;
    if (debug_device_grad_norm) return error.Gemma4MetalDebugGradientReadbackForbidden;
}

const ValidatedAdapterConfig = struct {
    rank: u32,
    alpha: f32,
};

fn validateAutodiffAdapterConfig(inspected: finetune.InspectionSummary) !ValidatedAdapterConfig {
    if (inspected.use_dora orelse false) return error.DoRAAutodiffNotYetSupported;
    try finetune.validateLoRAInitializerBaseCompatibility(inspected.init_lora_weights);

    const rank = inspected.lora_rank orelse return error.MissingAdapterConfig;
    if (rank == 0 or rank > std.math.maxInt(u32)) return error.InvalidLoRARank;

    const alpha = inspected.lora_alpha orelse return error.MissingAdapterConfig;
    if (!std.math.isFinite(alpha) or alpha <= 0 or alpha > std.math.floatMax(f32)) return error.InvalidLoRAAlpha;
    return .{ .rank = @intCast(rank), .alpha = @floatCast(alpha) };
}

fn validateAutodiffBaseArtifact(
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    backend_kind: gemma4_real.BackendKind,
) !void {
    var manifest = try manifest_mod.loadFromDir(allocator, base_model_dir);
    defer manifest.deinit();

    const artifact_kind = manifest.nativeWeightArtifactKind() orelse return;
    try validateAutodiffBaseCapabilities(artifact_kind == .gguf);
    if (backend_kind != .metal) return;

    switch (artifact_kind) {
        .gguf => unreachable,
        .safetensors => try validateMetalStoredWeightBackward(
            allocator,
            manifest.safetensors_path,
            null,
        ),
        .sharded_safetensors => try validateMetalStoredWeightBackward(
            allocator,
            null,
            manifest.safetensors_index_path,
        ),
    }
}

fn validateMetalStoredWeightBackward(
    allocator: std.mem.Allocator,
    single_path: ?[]const u8,
    index_path: ?[]const u8,
) !void {
    var dependencies = try safetensors.inspectArtifactDependencies(allocator, single_path, index_path);
    defer dependencies.deinit();

    const first_tensor_path: usize = if (single_path != null) 0 else 1;
    for (dependencies.paths[first_tensor_path..]) |path| {
        var reader = try safetensors.MMapReader.openFileAbsolute(allocator, path);
        defer reader.deinit();
        var tensor_it = reader.header.tensors.iterator();
        while (tensor_it.next()) |entry| {
            const meta = entry.value_ptr.*;
            // Rank-2 BF16 has a device-resident frozen-linear dX path; F16 does not.
            if (meta.shape.len == 2 and meta.dtype == .f16) {
                return error.MetalStoredWeightBackwardDTypeNotYetSupported;
            }
        }
    }
}

fn validateAutodiffBaseCapabilities(has_gguf_weights: bool) !void {
    // BF16 frozen linears have a device-only dX kernel. Packed GGUF weights do
    // not yet: admitting them would require an unbounded host dequantization
    // and violate both the compiled/no-host contract and QLoRA memory bounds.
    if (has_gguf_weights) return error.GgufAutodiffBackwardNotYetSupported;
}

test "gemma4 auto trainer is fail-closed real autodiff" {
    try std.testing.expectEqual(TrainerMode.autodiff, resolveTrainerMode(.auto));
    try std.testing.expectEqual(TrainerMode.autodiff, resolveTrainerMode(.autodiff));
    try std.testing.expectEqual(TrainerMode.surrogate, resolveTrainerMode(.surrogate));
}

test "gemma4 autodiff CLI requires an explicit backend before opening artifacts" {
    try std.testing.expectError(
        error.MissingBackend,
        runFromArgs(std.testing.allocator, std.testing.io, &.{ "missing-base", "missing-adapter", "missing-prepared", "missing-out" }),
    );
}

test "gemma4 Metal execution policy is strict for train and eval" {
    const metal = autodiffExecutionPolicy(.metal);
    try std.testing.expectEqual(real_autodiff.TrainingExecutionEngine.compiled_device, metal.engine);
    try std.testing.expect(metal.compiled_required);
    try std.testing.expect(metal.strict_metal_execution);

    const native = autodiffExecutionPolicy(.native);
    try std.testing.expectEqual(real_autodiff.TrainingExecutionEngine.interpreter, native.engine);
    try std.testing.expect(!native.compiled_required);
    try std.testing.expect(!native.strict_metal_execution);

    try std.testing.expectError(
        error.Gemma4MetalTrainingGraphExecutorRequired,
        validateMetalGraphExecutorFlags(false, false, false, false, false),
    );
    try std.testing.expectError(
        error.Gemma4MetalTrainingGraphExecutorDisabled,
        validateMetalGraphExecutorFlags(true, true, false, false, false),
    );
    try std.testing.expectError(
        error.Gemma4MetalTrainingParityDiagnosticForbidden,
        validateMetalGraphExecutorFlags(true, false, true, false, false),
    );
    try std.testing.expectError(
        error.Gemma4MetalTrainingParityCheckForbidden,
        validateMetalGraphExecutorFlags(true, false, false, true, false),
    );
    try std.testing.expectError(
        error.Gemma4MetalDebugGradientReadbackForbidden,
        validateMetalGraphExecutorFlags(true, false, false, false, true),
    );
    try validateMetalGraphExecutorFlags(true, false, false, false, false);
}

test "gemma4 strict Metal step evidence rejects every fallback surface" {
    const valid = real_autodiff.StepProfile{
        .optimizer_backend = .metal,
        .graph_executor_partitions = 1,
        .graph_executor_command_dispatches = 1,
        .graph_executor_device_outputs = 3,
    };
    try real_autodiff.validateStrictMetalStepEvidence(valid, 0, 2, 2, .train);
    try real_autodiff.validateStrictMetalStepEvidence(valid, 0, 2, 2, .eval);
    try std.testing.expectError(error.StrictMetalGradientNotDeviceResident, real_autodiff.validateStrictMetalStepEvidence(valid, 0, 0, 2, .eval));
    var evidence: gemma4_real.CausalLmMetrics = .{};
    gemma4_real.recordStepExecutionEvidence(&evidence, .{
        .loss = 1,
        .grad_norm = 1,
        .step = 1,
        .optimizer_stepped = true,
        .profile = valid,
    });
    try std.testing.expectEqual(@as(u64, 1), evidence.graph_executor_steps);
    try std.testing.expectEqual(@as(u64, 1), evidence.graph_executor_partitions);
    try std.testing.expectEqual(@as(u64, 1), evidence.graph_executor_command_dispatches);
    try std.testing.expectEqual(@as(u64, 1), evidence.metal_optimizer_steps);

    var profile = valid;
    profile.graph_executor_fallback_reason = "fallback";
    try std.testing.expectError(error.StrictMetalGraphExecutorFallback, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_partitions = 0;
    try std.testing.expectError(error.StrictMetalGraphExecutorDidNotRun, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_command_dispatches = 0;
    try std.testing.expectError(error.StrictMetalGraphExecutorDidNotDispatch, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_native_partitions = 1;
    try std.testing.expectError(error.StrictMetalNativePartition, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_unsupported_ops = 1;
    try std.testing.expectError(error.StrictMetalUnsupportedOperation, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_interpreter_fallbacks = 1;
    try std.testing.expectError(error.StrictMetalInterpreterFallback, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_runtime_region_dispatches = 1;
    profile.graph_executor_runtime_region_active_regions = 1;
    try real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train);
    profile.graph_executor_runtime_region_fallbacks = 1;
    try std.testing.expectError(error.StrictMetalRuntimeRegion, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_host_output_runtime_region = 1;
    try std.testing.expectError(error.StrictMetalRuntimeRegion, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    profile = valid;
    profile.graph_executor_true_host_outputs = 1;
    try std.testing.expectError(error.StrictMetalHostOutput, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
    try std.testing.expectError(error.StrictMetalGradientNotDeviceResident, real_autodiff.validateStrictMetalStepEvidence(valid, 1, 2, 2, .train));
    profile = valid;
    profile.optimizer_backend = .host;
    try real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .eval);
    try std.testing.expectError(error.StrictMetalOptimizerRequired, real_autodiff.validateStrictMetalStepEvidence(profile, 0, 2, 2, .train));
}

test "gemma4 autodiff training options reject zero-update configurations" {
    try validateAutodiffTrainingOptions(.{ .backend_kind = .native });
    try std.testing.expectError(error.InvalidLearningRate, validateAutodiffTrainingOptions(.{ .backend_kind = .native, .learning_rate = 0 }));
    try std.testing.expectError(error.InvalidEpochCount, validateAutodiffTrainingOptions(.{ .backend_kind = .native, .epochs = 0 }));
    try std.testing.expectError(error.InvalidGradientAccumulation, validateAutodiffTrainingOptions(.{ .backend_kind = .native, .grad_accum_steps = 0 }));
    try std.testing.expectError(error.InvalidMaxGradNorm, validateAutodiffTrainingOptions(.{ .backend_kind = .native, .max_grad_norm = -1 }));
}

test "gemma4 unsupported autodiff options fail before artifact IO" {
    const prefix = [_][]const u8{
        "missing-base",
        "missing-adapter",
        "missing-prepared",
        "missing-out",
        "--eval-prepared",
        "missing-eval-prepared",
        "--backend",
        "native",
    };
    try std.testing.expectError(
        error.LayerScopedAutodiffNotYetSupported,
        runFromArgs(std.testing.allocator, std.testing.io, &(prefix ++ .{ "--layer-name", "model.layers.0" })),
    );
    try std.testing.expectError(
        error.LayerWiseDecayNotYetSupportedForAutodiff,
        runFromArgs(std.testing.allocator, std.testing.io, &(prefix ++ .{ "--llrd-decay", "0.9" })),
    );
    try std.testing.expectError(
        error.ScheduleFreeNotYetSupportedForAutodiff,
        runFromArgs(std.testing.allocator, std.testing.io, &(prefix ++ .{"--schedule-free"})),
    );
}

test "gemma4 autodiff adapter admission rejects unsupported and zero-update configs" {
    var inspected = finetune.InspectionSummary{
        .artifact_family_version = finetune.artifact_family_version,
        .variant = .adapter_only,
        .model_dir = "adapter",
        .lora_rank = 8,
        .lora_alpha = 16,
    };
    const valid = try validateAutodiffAdapterConfig(inspected);
    try std.testing.expectEqual(@as(u32, 8), valid.rank);
    try std.testing.expectEqual(@as(f32, 16), valid.alpha);

    inspected.use_dora = true;
    try std.testing.expectError(error.DoRAAutodiffNotYetSupported, validateAutodiffAdapterConfig(inspected));
    inspected.use_dora = false;

    inspected.init_lora_weights = "pissa";
    try std.testing.expectError(error.LoRAInitializerRequiresAdjustedBase, validateAutodiffAdapterConfig(inspected));
    inspected.init_lora_weights = "loftq";
    try std.testing.expectError(error.LoRAInitializerRequiresAdjustedBase, validateAutodiffAdapterConfig(inspected));
    inspected.init_lora_weights = "eva";

    inspected.lora_rank = 0;
    try std.testing.expectError(error.InvalidLoRARank, validateAutodiffAdapterConfig(inspected));
    if (comptime @bitSizeOf(usize) > @bitSizeOf(u32)) {
        inspected.lora_rank = @as(usize, std.math.maxInt(u32)) + 1;
        try std.testing.expectError(error.InvalidLoRARank, validateAutodiffAdapterConfig(inspected));
    }
    inspected.lora_rank = 8;

    inspected.lora_alpha = 0;
    try std.testing.expectError(error.InvalidLoRAAlpha, validateAutodiffAdapterConfig(inspected));
    inspected.lora_alpha = std.math.nan(f64);
    try std.testing.expectError(error.InvalidLoRAAlpha, validateAutodiffAdapterConfig(inspected));
    inspected.lora_alpha = std.math.inf(f64);
    try std.testing.expectError(error.InvalidLoRAAlpha, validateAutodiffAdapterConfig(inspected));
    inspected.lora_alpha = @as(f64, std.math.floatMax(f32)) * 2;
    try std.testing.expectError(error.InvalidLoRAAlpha, validateAutodiffAdapterConfig(inspected));
}

test "gemma4 autodiff rejects GGUF bases until packed-weight backward exists" {
    try std.testing.expectError(
        error.GgufAutodiffBackwardNotYetSupported,
        validateAutodiffBaseCapabilities(true),
    );
    try validateAutodiffBaseCapabilities(false);
}

test "gemma4 Metal autodiff admits native-dense BF16 stored weights from headers only" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const header =
        \\{"weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]}}
    ;
    var bytes: [8 + header.len + 8]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memset(bytes[8 + header.len ..], 0);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = &bytes });

    const model_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(model_dir);
    try validateAutodiffBaseArtifact(allocator, model_dir, .native);
    try validateAutodiffBaseArtifact(allocator, model_dir, .metal);
}

test "gemma4 Metal autodiff rejects native-dense F16 stored weights before backend or prepared IO" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const header =
        \\{"weight":{"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}}
    ;
    var bytes: [8 + header.len + 8]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memset(bytes[8 + header.len ..], 0);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = &bytes });

    const model_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(model_dir);
    try std.testing.expectError(
        error.MetalStoredWeightBackwardDTypeNotYetSupported,
        validateAutodiffBaseArtifact(allocator, model_dir, .metal),
    );
}

test "gemma4 Metal autodiff rejects mixed BF16 and F16 stored weights before backend or prepared IO" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const header =
        \\{"bf16_weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"f16_weight":{"dtype":"F16","shape":[2,2],"data_offsets":[8,16]}}
    ;
    var bytes: [8 + header.len + 16]u8 = undefined;
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    @memset(bytes[8 + header.len ..], 0);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "model.safetensors", .data = &bytes });

    const model_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(model_dir);
    try std.testing.expectError(
        error.MetalStoredWeightBackwardDTypeNotYetSupported,
        validateAutodiffBaseArtifact(allocator, model_dir, .metal),
    );
}

test "gemma4 autodiff admits selected safetensors when a deployment GGUF is colocated" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const model_dir = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(model_dir);
    const checkpoint_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(checkpoint_path);
    const values = [_]f32{ 1, 0, 0, 1 };
    try safetensors_checkpoint.save(allocator, checkpoint_path, &.{.{
        .name = "weight",
        .shape = &.{ 2, 2 },
        .data = &values,
    }});
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "export.gguf", .data = "GGUFstub" });

    var manifest = try manifest_mod.loadFromDir(allocator, model_dir);
    defer manifest.deinit();

    try std.testing.expectEqual(
        manifest_mod.NativeWeightArtifactKind.safetensors,
        manifest.nativeWeightArtifactKind().?,
    );
    try std.testing.expect(manifest.gguf_path != null);
    try validateAutodiffBaseArtifact(allocator, model_dir, .native);
    try validateAutodiffBaseArtifact(allocator, model_dir, .metal);

    try std.testing.expectError(
        error.GgufAutodiffBackwardNotYetSupported,
        validateAutodiffBaseArtifact(allocator, manifest.gguf_path.?, .metal),
    );
}

test "gemma4 autodiff rejects a real DoRA bootstrap before backend and output mutation" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const checkpoint_path = try std.fs.path.join(allocator, &.{ root, finetune.checkpoint_file_name });
    defer allocator.free(checkpoint_path);
    const adapter_dir = try std.fs.path.join(allocator, &.{ root, "adapter" });
    defer allocator.free(adapter_dir);
    const prepared_path = try std.fs.path.join(allocator, &.{ root, "prepared.json" });
    defer allocator.free(prepared_path);
    const out_dir = try std.fs.path.join(allocator, &.{ root, "out" });
    defer allocator.free(out_dir);

    const base_values = [_]f32{ 1, 0, 0, 1 };
    try tmp.dir.writeFile(io, .{ .sub_path = finetune.hf_config_file_name, .data =
        \\{"model_type":"gemma4_text","max_position_embeddings":32}
    });
    try safetensors_checkpoint.save(allocator, checkpoint_path, &.{.{
        .name = "model.layers.0.self_attn.q_proj.weight",
        .shape = &.{ 2, 2 },
        .data = &base_values,
    }});
    var bootstrap = try finetune.bootstrapLoRABundle(allocator, root, adapter_dir, .{
        .rank = 1,
        .alpha = 1,
        .target_modules = &.{"q_proj"},
        .use_dora = true,
    });
    defer finetune.freeBootstrapSummary(allocator, &bootstrap);

    var prompt_ids = [_]i32{1};
    var response_ids = [_]i32{2};
    var input_ids = [_]i32{ 1, 2 };
    var labels = [_]i32{ -100, 2 };
    var examples = [_]finetune.PreparedExampleInput{.{
        .mode = .instruction,
        .prompt_input_ids = &prompt_ids,
        .response_input_ids = &response_ids,
        .num_prompt_tokens = 1,
        .num_response_tokens = 1,
        .input_ids = &input_ids,
        .labels = &labels,
        .num_input_tokens = 2,
        .num_supervised_tokens = 1,
        .source_identity_sha256 = "1111111111111111111111111111111111111111111111111111111111111111",
    }};
    var provenance = try finetune.fingerprintGemma4Model(allocator, root);
    defer provenance.deinit(allocator);
    const examples_digest = try finetune.fingerprintPreparedExamplesAlloc(allocator, &examples);
    defer allocator.free(examples_digest);
    try finetune.savePreparedInputsSummary(allocator, prepared_path, .{
        .artifact_family_version = finetune.artifact_family_version,
        .model_dir = root,
        .schema_version = finetune.prepared_schema_v4,
        .base_model_sha256 = provenance.base_model_sha256,
        .tokenizer_sha256 = provenance.tokenizer_sha256,
        .chat_template_sha256 = provenance.chat_template_sha256,
        .prepared_examples_sha256 = examples_digest,
        .max_examples = 1,
        .examples_seen = 1,
        .max_seq_len = 2,
        .max_prompt_tokens = 1,
        .max_response_tokens = 1,
        .max_input_tokens = 2,
        .max_supervised_tokens = 1,
        .examples = &examples,
    });

    try tmp.dir.createDirPath(io, "out");
    try tmp.dir.writeFile(io, .{ .sub_path = "out/sentinel.txt", .data = "preserve" });
    try std.testing.expectError(
        error.DoRAAutodiffNotYetSupported,
        runFromArgs(allocator, io, &.{ root, adapter_dir, prepared_path, out_dir, "--backend", "native", "--eval-prepared", prepared_path }),
    );

    const sentinel = try tmp.dir.readFileAlloc(io, "out/sentinel.txt", allocator, .limited(16));
    defer allocator.free(sentinel);
    try std.testing.expectEqualStrings("preserve", sentinel);
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "out/adapter_model.safetensors", .{}));
    try std.testing.expectError(error.FileNotFound, tmp.dir.statFile(io, "out/training_report.json", .{}));
}

fn runAutodiff(
    io: std.Io,
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    prepared_inputs_path: []const u8,
    eval_prepared_inputs_path: []const u8,
    out_dir: []const u8,
    prepared: finetune.PreparedInputsSummary,
    eval_prepared: finetune.PreparedInputsSummary,
    opts: CliOptions,
) !void {
    var adapter_inspect = try finetune.inspectCheckpoint(allocator, adapter_model_dir);
    defer finetune.freeInspectionSummary(allocator, &adapter_inspect);
    const validated_adapter = try validateAutodiffAdapterConfig(adapter_inspect);
    const recursive_shared_block_size = adapter_inspect.recursive_shared_block_size;

    const graph_config = try gemma4_real.loadGraphConfig(allocator, base_model_dir);
    _ = try finetune.validatePreparedSequenceAdmission(prepared, graph_config.max_position_embeddings);
    _ = try finetune.validatePreparedSequenceAdmission(eval_prepared, graph_config.max_position_embeddings);
    try finetune.validatePreparedEvalDisjoint(allocator, prepared.examples, eval_prepared.examples);
    var provenance = try finetune.fingerprintGemma4Model(allocator, base_model_dir);
    defer provenance.deinit(allocator);
    try finetune.validatePreparedModelProvenance(prepared, provenance);
    try finetune.validatePreparedModelProvenance(eval_prepared, provenance);
    try finetune.validateAdapterModelProvenance(adapter_inspect, provenance);

    var publication = try ImmutableRunPublication.init(allocator, io, out_dir);
    defer publication.deinit();

    const bootstrap = gemma4_real.findFirstSupervisedExample(prepared.examples) orelse return error.NoTrainingData;
    const is_multimodal = finetune.preparedExamplesHaveMedia(prepared.examples);
    const eval_is_multimodal = finetune.preparedExamplesHaveMedia(eval_prepared.examples);
    if ((is_multimodal or eval_is_multimodal) and opts.gguf_projector_path == null) return error.MissingGgufProjector;
    var maybe_projector_fingerprint: ?finetune.ProjectorFingerprint = null;
    defer if (maybe_projector_fingerprint) |*fp| finetune.freeProjectorFingerprint(allocator, fp);
    if (is_multimodal or eval_is_multimodal) {
        maybe_projector_fingerprint = try finetune.fingerprintProjectorFile(allocator, opts.gguf_projector_path.?);
        if (is_multimodal) try validatePreparedProjectorFingerprint(prepared, maybe_projector_fingerprint.?);
        if (eval_is_multimodal) try validatePreparedProjectorFingerprint(eval_prepared, maybe_projector_fingerprint.?);
    }
    const mm_stats = summarizeMultimodalPrepared(prepared.examples);
    const backend_kind = opts.backend_kind orelse return error.MissingBackend;
    const execution_policy = autodiffExecutionPolicy(backend_kind);
    const target_modules = adapter_inspect.target_modules orelse finetune.default_lora_target_modules[0..];
    const lora_config = ml.graph.lora.LoRAConfig{
        .rank = validated_adapter.rank,
        .alpha = validated_adapter.alpha,
        .target_patterns = target_modules,
        .strict_target_patterns = true,
        .sharing = if (adapter_inspect.recursive_lora_enabled) .by_use else .by_weight,
    };

    const before = try evaluateAutodiff(
        allocator,
        base_model_dir,
        adapter_model_dir,
        eval_prepared.examples,
        opts.effectiveEvalMaxExamples(),
        graph_config,
        lora_config,
        backend_kind,
        opts.max_grad_norm,
        opts.gguf_projector_path,
        if (maybe_projector_fingerprint) |fp| fp.sha256 else null,
    );

    var backend = try gemma4_real.loadBackendForModelDir(allocator, base_model_dir, backend_kind);
    defer backend.deinit();

    var trainer = try real_autodiff.RealAutodiffTrainer.init(allocator, backend.backendPtr(), .{
        .lora = lora_config,
        .optimizer = .{},
        .lr_schedule = .{ .constant = opts.learning_rate },
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .hidden_size_hint = graph_config.hidden_size,
        .num_layers_hint = graph_config.num_hidden_layers,
        .execution_engine = execution_policy.engine,
        .compiled_required = execution_policy.compiled_required,
        .strict_metal_execution = execution_policy.strict_metal_execution,
        .checkpoint_config = if (opts.activation_checkpoint_interval > 0) .{
            .strategy = .every_n_layers,
            .layer_interval = opts.activation_checkpoint_interval,
        } else null,
    });
    defer trainer.deinit();
    var maybe_text_ctx: ?gemma4_real.GemmaAutodiffCtx = null;
    var maybe_mm_ctx: ?gemma4_mm_real.MultimodalCtx = null;
    if (is_multimodal) {
        const tokenizer = try gemma4_mm_real.loadTokenizerForModelDir(allocator, base_model_dir);
        maybe_mm_ctx = if (recursive_shared_block_size) |shared_block_size|
            gemma4_mm_real.MultimodalCtx.initRecursive(allocator, backend.backendPtr(), graph_config, opts.gguf_projector_path.?, maybe_projector_fingerprint.?.sha256, tokenizer, shared_block_size)
        else
            gemma4_mm_real.MultimodalCtx.init(allocator, backend.backendPtr(), graph_config, opts.gguf_projector_path.?, maybe_projector_fingerprint.?.sha256, tokenizer);
        try gemma4_mm_real.initializeTrainerFromAdapterDir(
            allocator,
            &trainer,
            &maybe_mm_ctx.?,
            adapter_model_dir,
            bootstrap,
            @intCast(prepared.max_seq_len),
        );
    } else {
        maybe_text_ctx = if (recursive_shared_block_size) |shared_block_size|
            gemma4_real.GemmaAutodiffCtx.initRecursive(graph_config, shared_block_size)
        else
            gemma4_real.GemmaAutodiffCtx.init(graph_config);
        try gemma4_real.initializeTrainerFromAdapterDir(
            allocator,
            &trainer,
            &maybe_text_ctx.?,
            adapter_model_dir,
            bootstrap,
            @intCast(prepared.max_seq_len),
        );
    }
    defer if (maybe_mm_ctx) |*ctx| ctx.deinit();

    const epoch_history = try allocator.alloc(AutodiffEpochSummary, opts.epochs);
    defer allocator.free(epoch_history);
    for (0..opts.epochs) |epoch_idx| {
        const metrics = if (is_multimodal)
            try gemma4_mm_real.trainPreparedExamples(
                allocator,
                &trainer,
                &maybe_mm_ctx.?,
                prepared.examples,
                opts.max_examples,
                @intCast(prepared.max_seq_len),
            )
        else
            try gemma4_real.trainPreparedExamples(
                allocator,
                &trainer,
                &maybe_text_ctx.?,
                prepared.examples,
                opts.max_examples,
                @intCast(prepared.max_seq_len),
            );
        try validateAutodiffEpoch(metrics);
        epoch_history[epoch_idx] = .{
            .examples_seen = metrics.examples_seen,
            .supervised_tokens_seen = metrics.supervised_tokens_seen,
            .teacher_examples_seen = metrics.teacher_examples_seen,
            .teacher_supervised_tokens_seen = metrics.teacher_supervised_tokens_seen,
            .mean_teacher_temperature = metrics.mean_teacher_temperature,
            .average_loss = metrics.average_loss,
            .mean_grad_norm = metrics.mean_grad_norm,
            .optimizer_steps = metrics.optimizer_steps,
            .graph_executor_steps = metrics.graph_executor_steps,
            .graph_executor_fallback_steps = metrics.graph_executor_fallback_steps,
            .graph_executor_partitions = metrics.graph_executor_partitions,
            .graph_executor_command_dispatches = metrics.graph_executor_command_dispatches,
            .graph_executor_native_partitions = metrics.graph_executor_native_partitions,
            .graph_executor_unsupported_ops = metrics.graph_executor_unsupported_ops,
            .graph_executor_interpreter_fallbacks = metrics.graph_executor_interpreter_fallbacks,
            .graph_executor_runtime_region_dispatches = metrics.graph_executor_runtime_region_dispatches,
            .graph_executor_true_host_outputs = metrics.graph_executor_true_host_outputs,
            .metal_optimizer_steps = metrics.metal_optimizer_steps,
        };
        std.log.info(
            "gemma4 autodiff: epoch={d}/{d} loss={d:.4} examples={d} tokens={d} updates={d}",
            .{ epoch_idx + 1, opts.epochs, metrics.average_loss, metrics.examples_seen, metrics.supervised_tokens_seen, metrics.optimizer_steps },
        );
    }

    try gemma4_real.saveTrainerAsGemmaBundle(allocator, &trainer, base_model_dir, adapter_model_dir, publication.staging_dir);
    publication.claimStaging();
    const after = try evaluateAutodiff(
        allocator,
        base_model_dir,
        publication.staging_dir,
        eval_prepared.examples,
        opts.effectiveEvalMaxExamples(),
        graph_config,
        lora_config,
        backend_kind,
        opts.max_grad_norm,
        opts.gguf_projector_path,
        if (maybe_projector_fingerprint) |fp| fp.sha256 else null,
    );

    const report_payload = .{
        .artifact_family_version = finetune.artifact_family_version,
        .trainer_kind = if (is_multimodal) "real_autodiff_multimodal_causal_lm_v1" else "real_autodiff_causal_lm_v1",
        .prepared_inputs_path = prepared_inputs_path,
        .eval_prepared_inputs_path = eval_prepared_inputs_path,
        .saved_adapter_checkpoint = finetune.adapter_checkpoint_file_name,
        .learning_rate = opts.learning_rate,
        .max_examples = opts.max_examples,
        .eval_max_examples = opts.effectiveEvalMaxExamples(),
        .epochs = opts.epochs,
        .layer_name = opts.layer_name,
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .activation_checkpoint_interval = opts.activation_checkpoint_interval,
        .llrd_decay = opts.llrd_decay,
        .use_schedule_free = opts.use_schedule_free,
        .backend_kind = backend_kind,
        .multimodal = .{
            .enabled = is_multimodal,
            .gguf_projector_path = opts.gguf_projector_path,
            .gguf_projector_sha256 = if (maybe_projector_fingerprint) |fp| fp.sha256 else null,
            .gguf_projector_size_bytes = if (maybe_projector_fingerprint) |fp| fp.size_bytes else null,
            .projected_media_cache_entries = if (maybe_mm_ctx) |*ctx| ctx.projectedMediaCacheEntries() else 0,
            .projected_media_cache_hits = if (maybe_mm_ctx) |*ctx| ctx.projected_media_cache_hits else 0,
            .projected_media_cache_misses = if (maybe_mm_ctx) |*ctx| ctx.projected_media_cache_misses else 0,
            .examples_with_media = mm_stats.examples_with_media,
            .total_image_inputs = mm_stats.total_image_inputs,
            .total_audio_inputs = mm_stats.total_audio_inputs,
            .total_image_soft_tokens = mm_stats.total_image_soft_tokens,
            .total_audio_soft_tokens = mm_stats.total_audio_soft_tokens,
        },
        .prepared_dataset = .{
            .schema_version = prepared.schema_version,
            .prepared_examples_sha256 = prepared.prepared_examples_sha256,
            .examples_seen = prepared.examples_seen,
            .max_seq_len = prepared.max_seq_len,
            .max_input_tokens = prepared.max_input_tokens,
            .max_supervised_tokens = prepared.max_supervised_tokens,
            .examples_with_tool_calls = prepared.examples_with_tool_calls,
            .examples_with_tool_results = prepared.examples_with_tool_messages,
            .examples_with_multiturn = prepared.examples_with_multiturn,
            .examples_with_images = prepared.examples_with_images,
            .examples_with_audio = prepared.examples_with_audio,
            .examples_truncated = prepared.examples_truncated,
            .max_turns_dropped = prepared.max_turns_dropped,
        },
        .teacher_provenance = .{
            .base_model_sha256 = prepared.teacher_base_model_sha256,
            .tokenizer_sha256 = prepared.teacher_tokenizer_sha256,
            .chat_template_sha256 = prepared.teacher_chat_template_sha256,
            .gguf_projector_sha256 = prepared.teacher_gguf_projector_sha256,
        },
        .evaluation_dataset = .{
            .schema_version = eval_prepared.schema_version,
            .examples_seen = eval_prepared.examples_seen,
            .max_seq_len = eval_prepared.max_seq_len,
            .max_input_tokens = eval_prepared.max_input_tokens,
            .max_supervised_tokens = eval_prepared.max_supervised_tokens,
            .prepared_examples_sha256 = eval_prepared.prepared_examples_sha256,
        },
        .before = before,
        .epoch_history = epoch_history,
        .after = after,
    };
    try writeRunOutputs(io, allocator, publication.staging_dir, base_model_dir, adapter_model_dir, if (is_multimodal) "autodiff_multimodal" else "autodiff", report_payload, .{
        .prepared_inputs_path = prepared_inputs_path,
        .learning_rate = opts.learning_rate,
        .max_examples = opts.max_examples,
        .eval_max_examples = opts.effectiveEvalMaxExamples(),
        .epochs = opts.epochs,
        .layer_name = opts.layer_name,
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .activation_checkpoint_interval = opts.activation_checkpoint_interval,
        .llrd_decay = opts.llrd_decay,
        .use_schedule_free = opts.use_schedule_free,
        .backend_label = backend_kind.label(),
    });
    try publication.publish();
}

fn validateAutodiffEpoch(metrics: gemma4_real.CausalLmMetrics) !void {
    if (metrics.examples_seen == 0 or metrics.supervised_tokens_seen == 0) return error.NoTrainingData;
    if (!std.math.isFinite(metrics.average_loss)) return error.NonFiniteTrainingLoss;
    if (metrics.optimizer_steps == 0) return error.NoOptimizerSteps;
}

test "gemma4 autodiff epochs require a finite loss and an optimizer update" {
    try validateAutodiffEpoch(.{
        .examples_seen = 1,
        .supervised_tokens_seen = 2,
        .average_loss = 1.0,
        .optimizer_steps = 1,
    });
    try std.testing.expectError(error.NoTrainingData, validateAutodiffEpoch(.{}));
    try std.testing.expectError(error.NoOptimizerSteps, validateAutodiffEpoch(.{
        .examples_seen = 1,
        .supervised_tokens_seen = 2,
        .average_loss = 1.0,
    }));
}

fn evaluateAutodiff(
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    examples: []const finetune.PreparedExampleInput,
    max_examples: usize,
    graph_config: gemma_graph.Config,
    lora_config: ml.graph.lora.LoRAConfig,
    backend_kind: gemma4_real.BackendKind,
    max_grad_norm: f32,
    gguf_projector_path: ?[]const u8,
    gguf_projector_sha256: ?[]const u8,
) !gemma4_real.CausalLmMetrics {
    const execution_policy = autodiffExecutionPolicy(backend_kind);
    const limit = if (max_examples > 0 and max_examples < examples.len) max_examples else examples.len;
    const selected_examples = examples[0..limit];
    const bootstrap = gemma4_real.findFirstSupervisedExample(selected_examples) orelse return error.NoEvaluationData;
    const selected_seq_len: u32 = @intCast(limitExampleSeqLen(selected_examples, graph_config));
    var backend = try gemma4_real.loadBackendForModelDir(allocator, base_model_dir, backend_kind);
    defer backend.deinit();
    const is_multimodal = countMultimodalExamples(selected_examples) > 0;
    var adapter_inspect = try finetune.inspectCheckpoint(allocator, adapter_model_dir);
    defer finetune.freeInspectionSummary(allocator, &adapter_inspect);
    const recursive_shared_block_size = adapter_inspect.recursive_shared_block_size;

    const eval_accum_steps: u32 = @intCast(@min(limit + 1, @as(usize, std.math.maxInt(u32))));

    var trainer = try real_autodiff.RealAutodiffTrainer.init(allocator, backend.backendPtr(), .{
        .lora = lora_config,
        .optimizer = .{},
        .lr_schedule = .{ .constant = 0.0 },
        .max_grad_norm = max_grad_norm,
        .grad_accum_steps = eval_accum_steps,
        .hidden_size_hint = graph_config.hidden_size,
        .num_layers_hint = graph_config.num_hidden_layers,
        .execution_engine = execution_policy.engine,
        .compiled_required = execution_policy.compiled_required,
        .strict_metal_execution = execution_policy.strict_metal_execution,
    });
    defer trainer.deinit();
    if (is_multimodal) {
        const projector_path = gguf_projector_path orelse return error.MissingGgufProjector;
        const projector_sha256 = gguf_projector_sha256 orelse return error.MissingPreparedProjectorFingerprint;
        const tokenizer = try gemma4_mm_real.loadTokenizerForModelDir(allocator, base_model_dir);
        var ctx = if (recursive_shared_block_size) |shared_block_size|
            gemma4_mm_real.MultimodalCtx.initRecursive(allocator, backend.backendPtr(), graph_config, projector_path, projector_sha256, tokenizer, shared_block_size)
        else
            gemma4_mm_real.MultimodalCtx.init(allocator, backend.backendPtr(), graph_config, projector_path, projector_sha256, tokenizer);
        defer ctx.deinit();
        try gemma4_mm_real.initializeTrainerFromAdapterDir(
            allocator,
            &trainer,
            &ctx,
            adapter_model_dir,
            bootstrap,
            selected_seq_len,
        );
        return gemma4_mm_real.evaluatePreparedExamples(
            allocator,
            &trainer,
            &ctx,
            selected_examples,
            0,
            selected_seq_len,
        );
    } else {
        var ctx = if (recursive_shared_block_size) |shared_block_size|
            gemma4_real.GemmaAutodiffCtx.initRecursive(graph_config, shared_block_size)
        else
            gemma4_real.GemmaAutodiffCtx.init(graph_config);
        try gemma4_real.initializeTrainerFromAdapterDir(
            allocator,
            &trainer,
            &ctx,
            adapter_model_dir,
            bootstrap,
            selected_seq_len,
        );
        return gemma4_real.evaluatePreparedExamples(
            allocator,
            &trainer,
            &ctx,
            selected_examples,
            0,
            selected_seq_len,
        );
    }
}

fn limitExampleSeqLen(
    examples: []const finetune.PreparedExampleInput,
    graph_config: gemma_graph.Config,
) usize {
    _ = graph_config;
    var max_len: usize = 1;
    for (examples) |example| {
        if (example.num_input_tokens > max_len) max_len = example.num_input_tokens;
    }
    return max_len;
}

test "gemma4 eval graph sizing only inspects the selected example slice" {
    const examples = [_]finetune.PreparedExampleInput{
        .{ .mode = .instruction, .prompt_input_ids = &.{}, .response_input_ids = &.{}, .num_prompt_tokens = 0, .num_response_tokens = 0, .num_input_tokens = 16 },
        .{ .mode = .instruction, .prompt_input_ids = &.{}, .response_input_ids = &.{}, .num_prompt_tokens = 0, .num_response_tokens = 0, .num_input_tokens = 2048 },
    };
    const config = gemma_graph.Config{ .family = .gemma };
    try std.testing.expectEqual(@as(usize, 16), limitExampleSeqLen(examples[0..1], config));
    try std.testing.expectEqual(@as(usize, 2048), limitExampleSeqLen(&examples, config));
}

fn runSurrogate(
    io: std.Io,
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    prepared_inputs_path: []const u8,
    out_dir: []const u8,
    prepared: finetune.PreparedInputsSummary,
    opts: CliOptions,
) !void {
    if (finetune.preparedExamplesHaveMedia(prepared.examples)) return error.MultimodalRequiresAutodiffTrainer;

    const backend_ptr: ?*const ComputeBackend = null;

    var bundle = try finetune.loadLoRABundleScoped(allocator, base_model_dir, adapter_model_dir, opts.layer_name);
    defer bundle.deinit();
    var publication = try ImmutableRunPublication.init(allocator, io, out_dir);
    defer publication.deinit();

    const PjrtClientT = if (build_options.enable_pjrt) ?pjrt_mod.pjrt.Client else void;
    var pjrt_client_storage: PjrtClientT = if (comptime build_options.enable_pjrt) null else {};
    if (comptime build_options.enable_pjrt) {
        pjrt_client_storage = pjrt_mod.pjrt.Client.initFromEnv(allocator) catch |err| blk: {
            std.log.warn("PJRT client init failed ({s}); LoRA gradients will use CPU", .{@errorName(err)});
            break :blk null;
        };
    }
    defer if (comptime build_options.enable_pjrt) {
        if (pjrt_client_storage) |*client| client.deinit();
    };

    const PjrtStepsT = if (build_options.enable_pjrt) ?[]?graph_bridge.LoRAPjrtTrainStep else void;
    var pjrt_lora_steps: PjrtStepsT = if (comptime build_options.enable_pjrt) null else {};
    if (comptime build_options.enable_pjrt) {
        if (pjrt_client_storage) |*pjrt_client| {
            const steps = try allocator.alloc(?graph_bridge.LoRAPjrtTrainStep, bundle.layers.len);
            @memset(steps, null);
            var compiled_count: usize = 0;
            for (bundle.layers, 0..) |*layer, li| {
                var layer_graph = graph_bridge.LoRALinearGraph.init(
                    allocator,
                    3,
                    layer.input_dim,
                    layer.output_dim,
                    layer.rank,
                    bundle.lora_alpha,
                ) catch continue;
                steps[li] = graph_bridge.compileLoRALinearPjrtStep(allocator, &layer_graph, pjrt_client) catch blk: {
                    layer_graph.deinit();
                    break :blk null;
                };
                if (steps[li] != null) {
                    layer_graph.deinit();
                    compiled_count += 1;
                }
            }
            std.log.info("PJRT: compiled {d}/{d} LoRA layers", .{ compiled_count, bundle.layers.len });
            pjrt_lora_steps = steps;
        }
    }
    defer if (comptime build_options.enable_pjrt) {
        if (pjrt_lora_steps) |steps| {
            for (steps) |*step_opt| if (step_opt.*) |*step| step.deinit();
            allocator.free(steps);
        }
    };

    const before = try finetune.evaluatePreparedExamples(allocator, &bundle, prepared.examples, .{
        .max_examples = opts.effectiveEvalMaxExamples(),
        .layer_name = opts.layer_name,
    });

    const epoch_history = try allocator.alloc(finetune.TrainEpochSummary, opts.epochs);
    defer allocator.free(epoch_history);
    for (0..opts.epochs) |epoch_idx| {
        epoch_history[epoch_idx] = try finetune.trainPreparedExamplesEpoch(allocator, &bundle, prepared.examples, .{
            .learning_rate = opts.learning_rate,
            .max_examples = opts.max_examples,
            .layer_name = opts.layer_name,
            .max_grad_norm = opts.max_grad_norm,
            .grad_accum_steps = opts.grad_accum_steps,
            .llrd_decay = opts.llrd_decay,
            .use_schedule_free = opts.use_schedule_free,
            .compute_backend = backend_ptr,
            .pjrt_lora_steps = if (comptime build_options.enable_pjrt) pjrt_lora_steps else {},
        });
    }
    const after = try finetune.evaluatePreparedExamples(allocator, &bundle, prepared.examples, .{
        .max_examples = opts.effectiveEvalMaxExamples(),
        .layer_name = opts.layer_name,
    });

    try publication.createStaging();
    try finetune.saveLoRABundleToStaging(&bundle, publication.staging_dir);

    const report_payload = .{
        .artifact_family_version = finetune.artifact_family_version,
        .trainer_kind = "surrogate_lora_turn_aware_v2",
        .prepared_inputs_path = prepared_inputs_path,
        .saved_adapter_checkpoint = finetune.adapter_checkpoint_file_name,
        .learning_rate = opts.learning_rate,
        .max_examples = opts.max_examples,
        .eval_max_examples = opts.effectiveEvalMaxExamples(),
        .epochs = opts.epochs,
        .layer_name = opts.layer_name,
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .activation_checkpoint_interval = opts.activation_checkpoint_interval,
        .llrd_decay = opts.llrd_decay,
        .use_schedule_free = opts.use_schedule_free,
        .multimodal = .{
            .enabled = false,
            .gguf_projector_path = @as(?[]const u8, null),
            .examples_with_media = @as(usize, 0),
            .total_image_inputs = @as(usize, 0),
            .total_audio_inputs = @as(usize, 0),
            .total_image_soft_tokens = @as(usize, 0),
            .total_audio_soft_tokens = @as(usize, 0),
        },
        .prepared_dataset = .{
            .schema_version = prepared.schema_version,
            .examples_seen = prepared.examples_seen,
            .max_seq_len = prepared.max_seq_len,
            .max_input_tokens = prepared.max_input_tokens,
            .max_supervised_tokens = prepared.max_supervised_tokens,
            .examples_with_tool_calls = prepared.examples_with_tool_calls,
            .examples_with_tool_results = prepared.examples_with_tool_messages,
            .examples_with_multiturn = prepared.examples_with_multiturn,
            .examples_with_images = prepared.examples_with_images,
            .examples_with_audio = prepared.examples_with_audio,
            .examples_truncated = prepared.examples_truncated,
            .max_turns_dropped = prepared.max_turns_dropped,
        },
        .before = before,
        .epoch_history = epoch_history,
        .after = after,
    };
    try writeRunOutputs(io, allocator, publication.staging_dir, base_model_dir, adapter_model_dir, "surrogate", report_payload, .{
        .prepared_inputs_path = prepared_inputs_path,
        .learning_rate = opts.learning_rate,
        .max_examples = opts.max_examples,
        .eval_max_examples = opts.effectiveEvalMaxExamples(),
        .epochs = opts.epochs,
        .layer_name = opts.layer_name,
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .activation_checkpoint_interval = opts.activation_checkpoint_interval,
        .llrd_decay = opts.llrd_decay,
        .use_schedule_free = opts.use_schedule_free,
        .backend_label = "surrogate",
    });
    try publication.publish();
}

fn writeRunOutputs(
    io: std.Io,
    allocator: std.mem.Allocator,
    out_dir: []const u8,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    trainer_name: []const u8,
    report_payload: anytype,
    ctx: ReportContext,
) !void {
    const training_config_path = try std.fs.path.join(allocator, &.{ out_dir, "training_config.json" });
    defer allocator.free(training_config_path);
    try artifact_writer.writeJsonFile(allocator, training_config_path, .{
        .contract_version = run_contract.training_config_version,
        .artifact_family_version = finetune.artifact_family_version,
        .task = "gemma4_lora_train_eval",
        .inputs = .{
            .base_model_dir = base_model_dir,
            .adapter_model_dir = adapter_model_dir,
            .prepared_inputs_path = ctx.prepared_inputs_path,
        },
        .training = .{
            .trainer = trainer_name,
            .learning_rate = ctx.learning_rate,
            .max_examples = ctx.max_examples,
            .eval_max_examples = ctx.eval_max_examples,
            .epochs = ctx.epochs,
            .layer_name = ctx.layer_name,
            .max_grad_norm = ctx.max_grad_norm,
            .grad_accum_steps = ctx.grad_accum_steps,
            .activation_checkpoint_interval = ctx.activation_checkpoint_interval,
            .llrd_decay = ctx.llrd_decay,
            .use_schedule_free = ctx.use_schedule_free,
        },
        .backend_policy = .{
            .selected = ctx.backend_label,
            .preferred = ctx.backend_label,
        },
        .distributed = .{
            .enabled = false,
            .backend = ctx.backend_label,
            .rank = 0,
            .world_size = 1,
            .primary_rank = 0,
        },
    });

    const report_path = try std.fs.path.join(allocator, &.{ out_dir, "train_eval_report.json" });
    defer allocator.free(report_path);
    try artifact_writer.writeJsonFile(allocator, report_path, report_payload);

    const training_report_path = try std.fs.path.join(allocator, &.{ out_dir, "training_report.json" });
    defer allocator.free(training_report_path);
    try artifact_writer.writeJsonFile(allocator, training_report_path, .{
        .contract_version = run_contract.training_report_version,
        .artifact_family_version = finetune.artifact_family_version,
        .task = "gemma4_lora_train_eval",
        .backend_policy = .{
            .selected = ctx.backend_label,
            .preferred = ctx.backend_label,
        },
        .distributed = .{
            .enabled = false,
            .backend = ctx.backend_label,
            .rank = 0,
            .world_size = 1,
            .primary_rank = 0,
        },
        .report = report_payload,
    });

    const stdout = std.Io.File.stdout();
    var buf: [4096]u8 = undefined;
    var writer = stdout.writer(io, &buf);
    try std.json.Stringify.value(.{
        .before = report_payload.before,
        .epoch_history = report_payload.epoch_history,
        .after = report_payload.after,
    }, .{ .whitespace = .indent_2 }, &writer.interface);
    try writer.interface.writeByte('\n');
    try writer.interface.flush();
}

fn usageError() error{InvalidArguments} {
    std.debug.print(
        \\usage: train-eval-gemma4-lora-bundle <base_model_dir> <adapter_model_dir> <prepared_inputs_json> <out_dir> [options]
        \\
        \\Positional (legacy):
        \\  [learning_rate] [max_examples] [epochs] [layer_name]
        \\
        \\Flags:
        \\  --trainer auto|surrogate|autodiff   Trainer implementation (default: autodiff; auto is an alias)
        \\  --lr, --learning-rate <f32>         Learning rate (default: 0.001)
        \\  --max-examples <usize>              Max examples per epoch (default: 32)
        \\  --eval-prepared <path>              Required disjoint prepared evaluation artifact
        \\  --eval-max-examples <usize>         Max examples for before/after eval (default: --max-examples)
        \\  --epochs <usize>                    Number of epochs (default: 1)
        \\  --layer-name, --layer <str>         Scope to a specific layer name
        \\  --max-grad-norm <f32>               Gradient norm clipping threshold (default: 1.0, 0=disabled)
        \\  --grad-accum <u32>                  Gradient accumulation steps (default: 1)
        \\  --activation-checkpoint-interval N  Recompute between every N layer boundaries
        \\  --llrd-decay <f32>                  Surrogate-only layer-wise LR decay (default: 1.0)
        \\  --schedule-free                     Surrogate-only schedule-free AdamW
        \\  --backend native|metal               Required compute backend; no implicit fallback
        \\  --gguf-projector <path>             Required for multimodal autodiff examples; path to Gemma4 projector GGUF
        \\
        \\example: train-eval-gemma4-lora-bundle /tmp/gemma4-base /tmp/gemma4-lora /tmp/gemma4_inputs.json /tmp/out \
        \\           --trainer autodiff --backend metal --lr 0.0003 --max-examples 64 --epochs 3 --max-grad-norm 1.0 --grad-accum 4
        \\
    , .{});
    return error.InvalidArguments;
}

fn countMultimodalExamples(examples: []const finetune.PreparedExampleInput) usize {
    var count: usize = 0;
    for (examples) |example| {
        if (example.image_paths.len > 0 or example.audio_paths.len > 0) count += 1;
    }
    return count;
}

fn validatePreparedProjectorFingerprint(
    prepared: finetune.PreparedInputsSummary,
    actual: finetune.ProjectorFingerprint,
) !void {
    const expected_sha256 = prepared.gguf_projector_sha256 orelse return error.MissingPreparedProjectorFingerprint;
    const expected_size = prepared.gguf_projector_size_bytes orelse return error.MissingPreparedProjectorFingerprint;
    if (!std.mem.eql(u8, expected_sha256, actual.sha256)) return error.ProjectorFingerprintMismatch;
    if (expected_size != actual.size_bytes) return error.ProjectorFingerprintMismatch;
}

fn summarizeMultimodalPrepared(examples: []const finetune.PreparedExampleInput) MultimodalPreparedStats {
    var stats = MultimodalPreparedStats{};
    for (examples) |example| {
        if (example.image_paths.len > 0 or example.audio_paths.len > 0) {
            stats.examples_with_media += 1;
        }
        stats.total_image_inputs += example.image_paths.len;
        stats.total_audio_inputs += example.audio_paths.len;
        for (example.image_token_counts) |count| stats.total_image_soft_tokens += count;
        for (example.audio_token_counts) |count| stats.total_audio_soft_tokens += count;
    }
    return stats;
}
