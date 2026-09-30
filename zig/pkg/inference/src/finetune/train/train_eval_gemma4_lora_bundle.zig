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
const ml = @import("ml");
const inference = @import("inference_internal");
const finetune = inference.finetune.gemma4;
const gemma4_real = inference.finetune.gemma4_real_autodiff;
const gemma4_mm_real = inference.finetune.gemma4_multimodal_real_autodiff;
const real_autodiff = inference.finetune.real_autodiff_trainer;
const distributed_runtime = inference.finetune.distributed_runtime;
const graph_bridge = inference.finetune.graph_bridge;
const gemma_graph = @import("../../architectures/gemma_graph.zig");
const build_options = @import("build_options");
const run_contract = @import("../../run/contract.zig");
const artifact_writer = @import("../../run/artifact_writer.zig");
const ops_mod = @import("../../ops/ops.zig");
const ComputeBackend = ops_mod.ComputeBackend;
const pjrt_mod = if (build_options.enable_pjrt) @import("pjrt") else struct {
    pub const pjrt = struct {
        pub const Client = void;
    };
};

const TrainerMode = enum { auto, surrogate, autodiff };

const AutodiffEpochSummary = struct {
    duration_ns: u64 = 0,
    examples_per_second: f64 = 0,
    examples_seen: usize = 0,
    supervised_tokens_seen: usize = 0,
    teacher_examples_seen: usize = 0,
    teacher_supervised_tokens_seen: usize = 0,
    mean_teacher_temperature: f64 = 0,
    average_loss: f64 = 0,
    mean_grad_norm: f64 = 0,
    optimizer_steps: usize = 0,
};

const CliOptions = struct {
    learning_rate: f32 = 0.001,
    max_examples: usize = 32,
    eval_max_examples: usize = 0,
    epochs: usize = 1,
    layer_name: ?[]const u8 = null,
    max_grad_norm: f32 = 1.0,
    grad_accum_steps: u32 = 1,
    llrd_decay: f32 = 1.0,
    use_schedule_free: bool = false,
    trainer_mode: TrainerMode = .auto,
    gguf_projector_path: ?[]const u8 = null,

    fn effectiveEvalMaxExamples(self: CliOptions) usize {
        return if (self.eval_max_examples > 0) self.eval_max_examples else self.max_examples;
    }
};

const MultimodalPreparedStats = struct {
    examples_with_media: usize = 0,
    total_image_inputs: usize = 0,
    total_audio_inputs: usize = 0,
    total_image_soft_tokens: usize = 0,
    total_audio_soft_tokens: usize = 0,
};

const ReportContext = struct {
    prepared_inputs_path: []const u8,
    learning_rate: f32,
    max_examples: usize,
    eval_max_examples: usize,
    epochs: usize,
    layer_name: ?[]const u8,
    max_grad_norm: f32,
    grad_accum_steps: u32,
    llrd_decay: f32,
    use_schedule_free: bool,
};

fn monotonicNowNs() u64 {
    var ts: std.posix.timespec = undefined;
    if (std.posix.errno(std.posix.system.clock_gettime(.MONOTONIC, &ts)) != .SUCCESS) return 0;
    return @intCast(@as(i128, ts.sec) * std.time.ns_per_s + ts.nsec);
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
    var validate_only = false;
    var validate_local = false;
    var positional_count: usize = 0;
    var i: usize = 4;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--validate-only")) {
            validate_only = true;
        } else if (std.mem.eql(u8, arg, "--validate-local-only")) {
            validate_only = true;
            validate_local = true;
        } else if (std.mem.eql(u8, arg, "--lr") or std.mem.eql(u8, arg, "--learning-rate")) {
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
            if (!std.mem.eql(u8, val, "native") and !std.mem.eql(u8, val, "native") and !std.mem.eql(u8, val, "auto")) return usageError();
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

    var prepared = try finetune.loadPreparedInputsSummary(allocator, prepared_inputs_path);
    defer finetune.freePreparedInputsSummary(allocator, &prepared);

    if (validate_only) {
        const count = if (opts.max_examples == 0) prepared.examples.len else @min(opts.max_examples, prepared.examples.len);
        if (prepared.examples_with_images > 0 or prepared.examples_with_audio > 0 or opts.grad_accum_steps != 1) return error.UnsupportedDistributedGemma4Options;
        if (count == 0) return error.NoTrainingData;
        if (!validate_local and (count < 2 or count % 2 != 0)) return error.DistributedRequiresEvenExamples;
        for (prepared.examples[0..count]) |example| if (example.num_supervised_tokens == 0) return error.DistributedRequiresSupervisedExamples;
        return;
    }
    var distributed = try distributed_runtime.openFromEnv(allocator);
    defer if (distributed) |*context| context.deinit();
    const actual_mode = try resolveTrainerMode(allocator, base_model_dir, prepared, opts.trainer_mode, opts);
    if (distributed != null and actual_mode != .autodiff) return error.DistributedGemma4RequiresAutodiff;
    switch (actual_mode) {
        .surrogate => try runSurrogate(io, allocator, base_model_dir, adapter_model_dir, prepared_inputs_path, out_dir, prepared, opts),
        .autodiff => try runAutodiff(io, allocator, base_model_dir, adapter_model_dir, prepared_inputs_path, out_dir, prepared, opts, if (distributed) |*context| context else null),
        .auto => unreachable,
    }
}

fn resolveTrainerMode(
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    prepared: finetune.PreparedInputsSummary,
    requested: TrainerMode,
    opts: CliOptions,
) !TrainerMode {
    return switch (requested) {
        .surrogate, .autodiff => requested,
        .auto => blk: {
            const is_multimodal = prepared.examples_with_images > 0 or prepared.examples_with_audio > 0;
            if (is_multimodal) {
                if (opts.gguf_projector_path == null) break :blk .autodiff;
                _ = gemma4_real.loadGraphConfig(allocator, base_model_dir) catch |err| switch (err) {
                    error.FileNotFound,
                    error.UnsupportedGemmaMoeConfig,
                    error.UnsupportedPositionEncoding,
                    => return error.UnsupportedMultimodalAutodiffBaseModelLayout,
                    else => return err,
                };
                break :blk .autodiff;
            }
            const requires_surrogate =
                opts.layer_name != null or
                !std.math.approxEqAbs(f32, opts.llrd_decay, 1.0, 1e-6) or
                opts.use_schedule_free;
            if (requires_surrogate) break :blk .surrogate;
            _ = gemma4_real.loadGraphConfig(allocator, base_model_dir) catch |err| switch (err) {
                error.FileNotFound,
                error.UnsupportedGemmaMoeConfig,
                error.UnsupportedPositionEncoding,
                => break :blk .surrogate,
                else => return err,
            };
            break :blk .autodiff;
        },
    };
}

fn runAutodiff(
    io: std.Io,
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    prepared_inputs_path: []const u8,
    out_dir: []const u8,
    prepared: finetune.PreparedInputsSummary,
    opts: CliOptions,
    distributed: ?*distributed_runtime.Context,
) !void {
    if (opts.layer_name != null) return error.LayerScopedAutodiffNotYetSupported;
    if (!std.math.approxEqAbs(f32, opts.llrd_decay, 1.0, 1e-6)) return error.LayerWiseDecayNotYetSupportedForAutodiff;
    if (opts.use_schedule_free) return error.ScheduleFreeNotYetSupportedForAutodiff;

    const bootstrap = gemma4_real.findFirstSupervisedExample(prepared.examples) orelse return error.NoTrainingData;
    const is_multimodal = prepared.examples_with_images > 0 or prepared.examples_with_audio > 0;
    if (distributed != null and (is_multimodal or opts.grad_accum_steps != 1))
        return error.UnsupportedDistributedGemma4Options;
    const train_limit = if (opts.max_examples > 0 and opts.max_examples < prepared.examples.len) opts.max_examples else prepared.examples.len;
    if (distributed != null and (train_limit < 2 or train_limit % 2 != 0)) return error.DistributedRequiresEvenExamples;
    if (distributed != null) for (prepared.examples[0..train_limit]) |example| {
        if (example.num_supervised_tokens == 0) return error.DistributedRequiresSupervisedExamples;
    };
    const local_examples = if (distributed) |context| blk: {
        const items = try allocator.alloc(finetune.PreparedExampleInput, train_limit / 2);
        for (items, 0..) |*item, index| item.* = prepared.examples[2 * index + context.rank()];
        break :blk items;
    } else null;
    defer if (local_examples) |items| allocator.free(items);
    if (is_multimodal and opts.gguf_projector_path == null) return error.MissingGgufProjector;
    var maybe_projector_fingerprint: ?finetune.ProjectorFingerprint = null;
    defer if (maybe_projector_fingerprint) |*fp| finetune.freeProjectorFingerprint(allocator, fp);
    if (is_multimodal) {
        maybe_projector_fingerprint = try finetune.fingerprintProjectorFile(allocator, opts.gguf_projector_path.?);
        try validatePreparedProjectorFingerprint(prepared, maybe_projector_fingerprint.?);
    }
    const mm_stats = summarizeMultimodalPrepared(prepared.examples);
    const graph_config = try gemma4_real.loadGraphConfig(allocator, base_model_dir);
    const backend_kind: gemma4_real.BackendKind = .native;
    var adapter_inspect = try finetune.inspectCheckpoint(allocator, adapter_model_dir);
    defer finetune.freeInspectionSummary(allocator, &adapter_inspect);
    const recursive_shared_block_size = adapter_inspect.recursive_shared_block_size;

    const lora_rank = adapter_inspect.lora_rank orelse return error.MissingAdapterConfig;
    const lora_alpha = @as(f32, @floatCast(adapter_inspect.lora_alpha orelse return error.MissingAdapterConfig));
    const target_modules = adapter_inspect.target_modules orelse finetune.default_lora_target_modules[0..];
    const lora_config = ml.graph.lora.LoRAConfig{
        .rank = @intCast(lora_rank),
        .alpha = lora_alpha,
        .target_patterns = target_modules,
        .sharing = if (adapter_inspect.recursive_lora_enabled) .by_use else .by_weight,
    };

    const before = try evaluateAutodiff(
        allocator,
        base_model_dir,
        adapter_model_dir,
        prepared.examples,
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
        .reduce_grads = if (distributed != null) distributed_runtime.Context.reduce else null,
        .reduce_grads_ctx = if (distributed) |context| @ptrCast(context) else null,
        .hidden_size_hint = graph_config.hidden_size,
        .num_layers_hint = graph_config.num_hidden_layers,
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
    if (distributed) |context| try context.verifyWeightsAgree(&trainer);

    const epoch_history = try allocator.alloc(AutodiffEpochSummary, opts.epochs);
    defer allocator.free(epoch_history);
    for (0..opts.epochs) |epoch_idx| {
        const epoch_started_ns = monotonicNowNs();
        const metrics = if (is_multimodal)
            try gemma4_mm_real.trainPreparedExamples(
                allocator,
                &trainer,
                &maybe_mm_ctx.?,
                local_examples orelse prepared.examples,
                if (distributed != null) 0 else opts.max_examples,
                @intCast(prepared.max_seq_len),
            )
        else
            try gemma4_real.trainPreparedExamples(
                allocator,
                &trainer,
                &maybe_text_ctx.?,
                local_examples orelse prepared.examples,
                if (distributed != null) 0 else opts.max_examples,
                @intCast(prepared.max_seq_len),
            );
        const epoch_duration_ns = monotonicNowNs() -| epoch_started_ns;
        epoch_history[epoch_idx] = .{
            .duration_ns = epoch_duration_ns,
            .examples_per_second = if (epoch_duration_ns > 0)
                @as(f64, @floatFromInt(metrics.examples_seen)) * @as(f64, std.time.ns_per_s) / @as(f64, @floatFromInt(epoch_duration_ns))
            else
                0,
            .examples_seen = metrics.examples_seen,
            .supervised_tokens_seen = metrics.supervised_tokens_seen,
            .teacher_examples_seen = metrics.teacher_examples_seen,
            .teacher_supervised_tokens_seen = metrics.teacher_supervised_tokens_seen,
            .mean_teacher_temperature = metrics.mean_teacher_temperature,
            .average_loss = metrics.average_loss,
            .mean_grad_norm = metrics.mean_grad_norm,
            .optimizer_steps = metrics.optimizer_steps,
        };
        var progress_buffer: [2048]u8 = undefined;
        var progress_writer = std.Io.File.stdout().writerStreaming(io, &progress_buffer);
        try std.json.Stringify.value(.{
            .event = "progress",
            .family = "gemma4",
            .phase = "epoch",
            .rank = if (distributed) |context| context.rank() else @as(u8, 0),
            .epoch = epoch_idx + 1,
            .epochs = opts.epochs,
            .metrics = epoch_history[epoch_idx],
        }, .{}, &progress_writer.interface);
        try progress_writer.interface.writeByte('\n');
        try progress_writer.interface.flush();
        std.log.info(
            "gemma4 autodiff: epoch={d}/{d} loss={d:.4} examples={d} tokens={d} updates={d} duration_ns={d} examples_per_second={d:.3}",
            .{ epoch_idx + 1, opts.epochs, metrics.average_loss, metrics.examples_seen, metrics.supervised_tokens_seen, metrics.optimizer_steps, epoch_duration_ns, epoch_history[epoch_idx].examples_per_second },
        );
    }

    if (distributed) |context| try context.verifyWeightsAgree(&trainer);
    try gemma4_real.saveTrainerAsGemmaBundle(allocator, &trainer, base_model_dir, adapter_model_dir, out_dir);
    const after = try evaluateAutodiff(
        allocator,
        base_model_dir,
        out_dir,
        prepared.examples,
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
        .saved_adapter_checkpoint = finetune.adapter_checkpoint_file_name,
        .learning_rate = opts.learning_rate,
        .max_examples = opts.max_examples,
        .eval_max_examples = opts.effectiveEvalMaxExamples(),
        .epochs = opts.epochs,
        .layer_name = opts.layer_name,
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .llrd_decay = opts.llrd_decay,
        .use_schedule_free = opts.use_schedule_free,
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
    try writeRunOutputs(io, allocator, out_dir, base_model_dir, adapter_model_dir, if (is_multimodal) "autodiff_multimodal" else "autodiff", report_payload, .{
        .prepared_inputs_path = prepared_inputs_path,
        .learning_rate = opts.learning_rate,
        .max_examples = opts.max_examples,
        .eval_max_examples = opts.effectiveEvalMaxExamples(),
        .epochs = opts.epochs,
        .layer_name = opts.layer_name,
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .llrd_decay = opts.llrd_decay,
        .use_schedule_free = opts.use_schedule_free,
    });
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
    const bootstrap = gemma4_real.findFirstSupervisedExample(examples) orelse return error.NoTrainingData;
    var backend = try gemma4_real.loadBackendForModelDir(allocator, base_model_dir, backend_kind);
    defer backend.deinit();
    const is_multimodal = countMultimodalExamples(examples) > 0;
    var adapter_inspect = try finetune.inspectCheckpoint(allocator, adapter_model_dir);
    defer finetune.freeInspectionSummary(allocator, &adapter_inspect);
    const recursive_shared_block_size = adapter_inspect.recursive_shared_block_size;

    const limit = if (max_examples > 0 and max_examples < examples.len) max_examples else examples.len;
    const eval_accum_steps: u32 = @intCast(@min(limit + 1, @as(usize, std.math.maxInt(u32))));

    var trainer = try real_autodiff.RealAutodiffTrainer.init(allocator, backend.backendPtr(), .{
        .lora = lora_config,
        .optimizer = .{},
        .lr_schedule = .{ .constant = 0.0 },
        .max_grad_norm = max_grad_norm,
        .grad_accum_steps = eval_accum_steps,
        .hidden_size_hint = graph_config.hidden_size,
        .num_layers_hint = graph_config.num_hidden_layers,
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
            @intCast(limitExampleSeqLen(examples, graph_config)),
        );
        return gemma4_mm_real.evaluatePreparedExamples(
            allocator,
            &trainer,
            &ctx,
            examples,
            max_examples,
            @intCast(limitExampleSeqLen(examples, graph_config)),
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
            @intCast(limitExampleSeqLen(examples, graph_config)),
        );
        return gemma4_real.evaluatePreparedExamples(
            allocator,
            &trainer,
            &ctx,
            examples,
            max_examples,
            @intCast(limitExampleSeqLen(examples, graph_config)),
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
    if (prepared.examples_with_images > 0 or prepared.examples_with_audio > 0) return error.MultimodalRequiresAutodiffTrainer;

    const backend_ptr: ?*const ComputeBackend = null;

    var bundle = try finetune.loadLoRABundleScoped(allocator, base_model_dir, adapter_model_dir, opts.layer_name);
    defer bundle.deinit();

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

    try finetune.saveLoRABundle(&bundle, out_dir);

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
    try writeRunOutputs(io, allocator, out_dir, base_model_dir, adapter_model_dir, "surrogate", report_payload, .{
        .prepared_inputs_path = prepared_inputs_path,
        .learning_rate = opts.learning_rate,
        .max_examples = opts.max_examples,
        .eval_max_examples = opts.effectiveEvalMaxExamples(),
        .epochs = opts.epochs,
        .layer_name = opts.layer_name,
        .max_grad_norm = opts.max_grad_norm,
        .grad_accum_steps = opts.grad_accum_steps,
        .llrd_decay = opts.llrd_decay,
        .use_schedule_free = opts.use_schedule_free,
    });
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
            .llrd_decay = ctx.llrd_decay,
            .use_schedule_free = ctx.use_schedule_free,
        },
        .backend_policy = .{
            .selected = "native",
            .preferred = "native",
        },
        .distributed = .{
            .enabled = false,
            .backend = "native",
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
            .selected = "native",
            .preferred = "native",
        },
        .distributed = .{
            .enabled = false,
            .backend = "native",
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
        \\  --trainer auto|surrogate|autodiff   Trainer implementation (default: auto)
        \\  --validate-only                  Validate distributed text inputs without loading weights
        \\  --validate-local-only            Validate local text inputs without loading weights
        \\  --lr, --learning-rate <f32>         Learning rate (default: 0.001)
        \\  --max-examples <usize>              Max examples per epoch (default: 32)
        \\  --eval-max-examples <usize>         Max examples for before/after eval (default: --max-examples)
        \\  --epochs <usize>                    Number of epochs (default: 1)
        \\  --layer-name, --layer <str>         Scope to a specific layer name
        \\  --max-grad-norm <f32>               Gradient norm clipping threshold (default: 1.0, 0=disabled)
        \\  --grad-accum <u32>                  Gradient accumulation steps (default: 1)
        \\  --llrd-decay <f32>                  Surrogate-only layer-wise LR decay (default: 1.0)
        \\  --schedule-free                     Surrogate-only schedule-free AdamW
        \\  --backend auto|native                 Compute backend for gradient math (default: auto)
        \\  --gguf-projector <path>             Required for multimodal autodiff examples; path to Gemma4 projector GGUF
        \\
        \\example: train-eval-gemma4-lora-bundle /tmp/gemma4-base /tmp/gemma4-lora /tmp/gemma4_inputs.json /tmp/out \
        \\           --trainer autodiff --lr 0.0003 --max-examples 64 --epochs 3 --max-grad-norm 1.0 --grad-accum 4
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
