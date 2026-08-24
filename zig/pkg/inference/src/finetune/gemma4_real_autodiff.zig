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
const build_options = @import("build_options");
const ml = @import("ml");
const c_file = @import("../util/c_file.zig");
const gemma_graph = @import("../architectures/gemma_graph.zig");
const artifact_publication = @import("artifact_publication.zig");
const real_autodiff = @import("real_autodiff_trainer.zig");
const gemma4 = @import("gemma4.zig");
const compat = @import("../io/compat.zig");
const graph_input_binder = @import("graph_input_binder.zig");
const weight_source_mod = @import("../models/weight_source.zig");
const SafetensorsSource = weight_source_mod.SafetensorsSource;
const session_factory = @import("../architectures/session_factory.zig");
const Session = @import("../backends/session.zig").Session;
const ops_mod = @import("../ops/ops.zig");
const interpreter = @import("../graph/interpreter.zig");
const Tensor = @import("../backends/tensor.zig").Tensor;
const ComputeBackend = ops_mod.ComputeBackend;
const CT = ops_mod.CT;

const Builder = ml.graph.Builder;
const Graph = ml.graph.Graph;
const NodeId = ml.graph.NodeId;
const Shape = ml.graph.Shape;

fn copyTensorFloat32(dst: []f32, tensor: *const Tensor) !void {
    if (tensor.dtype != .f32) return error.AdapterShapeMismatch;
    if (tensor.data.len != dst.len * @sizeOf(f32)) return error.AdapterShapeMismatch;

    if (tensor.asFloat32IfAligned()) |values| {
        if (values.len != dst.len) return error.AdapterShapeMismatch;
        @memcpy(dst, values);
        return;
    }

    for (dst, 0..) |*value, idx| {
        const offset = idx * @sizeOf(f32);
        const bits = std.mem.readInt(u32, tensor.data[offset..][0..@sizeOf(f32)], .little);
        value.* = @bitCast(bits);
    }
}

pub const CausalLmMetrics = struct {
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

pub fn recordStepExecutionEvidence(metrics: *CausalLmMetrics, step: real_autodiff.StepResult) void {
    const profile = step.profile;
    if (profile.graph_executor_partitions > 0) metrics.graph_executor_steps += 1;
    if (profile.graph_executor_fallback_reason != null) metrics.graph_executor_fallback_steps += 1;
    metrics.graph_executor_partitions += profile.graph_executor_partitions;
    metrics.graph_executor_command_dispatches += profile.graph_executor_command_dispatches;
    metrics.graph_executor_native_partitions += profile.graph_executor_native_partitions;
    metrics.graph_executor_unsupported_ops += profile.graph_executor_unsupported_ops;
    metrics.graph_executor_interpreter_fallbacks += profile.graph_executor_interpreter_fallbacks;
    metrics.graph_executor_runtime_region_dispatches += profile.graph_executor_runtime_region_dispatches;
    metrics.graph_executor_true_host_outputs += profile.graph_executor_true_host_outputs;
    if (step.optimizer_stepped and profile.optimizer_backend == .metal) metrics.metal_optimizer_steps += 1;
}

pub const TeacherTopKOptions = struct {
    top_k: usize = 8,
    temperature: f32 = 1.0,
    max_examples: usize = 0,
};

pub const TeacherTopKSummary = struct {
    examples_seen: usize = 0,
    examples_written: usize = 0,
    supervised_tokens_seen: usize = 0,
    top_k: usize = 0,
    temperature: f32 = 1.0,
};

pub const BackendKind = enum {
    native,
    metal,

    pub fn label(self: BackendKind) []const u8 {
        return @tagName(self);
    }
};

pub const LoadedBackend = struct {
    kind: BackendKind,
    compute_backend: ComputeBackend,
    session: Session,

    pub fn backendPtr(self: *LoadedBackend) *const ComputeBackend {
        return &self.compute_backend;
    }

    pub fn deinit(self: *LoadedBackend) void {
        self.compute_backend.deinit();
        self.session.close();
        self.* = undefined;
    }
};

pub const GemmaAutodiffCtx = struct {
    graph_config: gemma_graph.Config,
    graph_options: gemma_graph.BuildOptions = .{},
    built: ?gemma_graph.GemmaGraph = null,
    lm_logits: ?NodeId = null,

    pub fn init(graph_config: gemma_graph.Config) GemmaAutodiffCtx {
        return .{ .graph_config = graph_config };
    }

    pub fn initRecursive(graph_config: gemma_graph.Config, shared_block_size: usize) GemmaAutodiffCtx {
        return .{
            .graph_config = graph_config,
            .graph_options = .{ .recursive_shared_block_size = @intCast(shared_block_size) },
        };
    }

    pub fn buildForward(
        ctx_opaque: *anyopaque,
        bld: *Builder,
        input_ids: ml.graph.NodeId,
        attention_mask: ml.graph.NodeId,
        batch: u32,
        seq_len: u32,
    ) anyerror!ml.graph.NodeId {
        _ = attention_mask;
        const self: *GemmaAutodiffCtx = @ptrCast(@alignCast(ctx_opaque));
        self.built = try gemma_graph.buildForwardGraphWithOptions(bld, self.graph_config, batch, seq_len, .{
            .input_ids = input_ids,
            .rope_cos = ml.graph.null_node,
            .rope_sin = ml.graph.null_node,
        }, self.graph_options);
        return self.built.?.output_node;
    }

    pub fn buildLoss(
        ctx_opaque: *anyopaque,
        bld: *Builder,
        forward_output: ml.graph.NodeId,
        targets: ml.graph.NodeId,
    ) anyerror!ml.graph.NodeId {
        const self: *GemmaAutodiffCtx = @ptrCast(@alignCast(ctx_opaque));
        const target_shape = bld.graph.node(targets).output_shape;
        if (target_shape.rank() == 2 and target_shape.dim(1) != self.graph_config.vocab_size) {
            return self.buildSparseCausalLoss(bld, forward_output, targets);
        }
        const logits = try self.buildLogits(bld, forward_output);
        return bld.crossEntropyLoss(logits, targets);
    }

    /// Memory-bounded causal LM loss. `targets` is [M, 2], containing
    /// [predictor_row, token_id] pairs for the M supervised tokens. The tied
    /// vocabulary projection is evaluated a few supervised rows at a time so
    /// the graph never owns a [sequence, vocabulary] activation.
    fn buildSparseCausalLoss(
        self: *GemmaAutodiffCtx,
        bld: *Builder,
        forward_output: NodeId,
        targets: NodeId,
    ) !NodeId {
        const out_shape = bld.graph.node(forward_output).output_shape;
        const total_rows: u32 = @intCast(out_shape.dim(0) * out_shape.dim(1));
        const hidden_size: u32 = @intCast(out_shape.dim(2));
        const target_shape = bld.graph.node(targets).output_shape;
        const supervised_rows: u32 = @intCast(target_shape.dim(0));
        const target_columns: u32 = @intCast(target_shape.dim(1));
        if (supervised_rows == 0) return error.NoSupervisedTokens;
        if (target_columns < sparse_target_columns) return error.InvalidTeacherDistillationTargets;
        const teacher_top_k: u32 = if (target_columns == sparse_target_columns)
            0
        else blk: {
            if (target_columns < sparse_teacher_fixed_columns or
                (target_columns - sparse_teacher_fixed_columns) % 2 != 0)
            {
                return error.InvalidTeacherDistillationTargets;
            }
            break :blk (target_columns - sparse_teacher_fixed_columns) / 2;
        };

        const hidden_flat = try bld.reshape(forward_output, Shape.init(.f32, &.{ @as(i64, @intCast(total_rows)), @as(i64, @intCast(hidden_size)) }));
        const predictor_rows_2d = try bld.sliceLastDim(targets, 0, 1);
        const predictor_rows = try bld.reshape(predictor_rows_2d, Shape.init(.f32, &.{@as(i64, @intCast(supervised_rows))}));
        const supervised_hidden = try bld.embeddingLookup(hidden_flat, predictor_rows, supervised_rows, hidden_size);
        const labels = if (teacher_top_k == 0) try bld.sliceLastDim(targets, 1, 2) else null;
        const teacher_temperatures = if (teacher_top_k > 0) try bld.sliceLastDim(targets, 1, 2) else null;
        const teacher_ids = if (teacher_top_k > 0) try bld.sliceLastDim(targets, 2, 2 + teacher_top_k) else null;
        const teacher_probs = if (teacher_top_k > 0) try bld.sliceLastDim(targets, 2 + teacher_top_k, target_columns) else null;
        const lm_head_w = try self.buildLmHeadWeight(bld, hidden_size);

        const vocab_size: usize = @intCast(self.graph_config.vocab_size);
        const vocab_ids = try bld.graph.allocator.alloc(f32, vocab_size);
        defer bld.graph.allocator.free(vocab_ids);
        for (vocab_ids, 0..) |*value, idx| value.* = @floatFromInt(idx);
        const vocab_row = try bld.tensorConst(vocab_ids, Shape.init(.f32, &.{ 1, @as(i64, @intCast(vocab_size)) }));

        var total_loss: ?NodeId = null;
        var start: u32 = 0;
        while (start < supervised_rows) : (start += sparse_loss_chunk_rows) {
            const end = @min(start + sparse_loss_chunk_rows, supervised_rows);
            const chunk_rows = end - start;
            const hidden_chunk = try sliceRows2d(bld, supervised_hidden, start, end, hidden_size);
            const raw_logits = try bld.linearNoBias(hidden_chunk, lm_head_w, chunk_rows, hidden_size, self.graph_config.vocab_size);
            const unscaled_logits = try self.applyFinalLogitSoftcap(bld, raw_logits);
            const logits = if (teacher_top_k > 0) blk: {
                const temperature_chunk = try sliceRows2d(bld, teacher_temperatures.?, start, end, 1);
                const temperature_bc = try broadcast2d(bld, temperature_chunk, chunk_rows, self.graph_config.vocab_size);
                break :blk try bld.div(unscaled_logits, temperature_bc);
            } else unscaled_logits;
            const dense_targets = if (teacher_top_k == 0) blk: {
                const label_chunk = try sliceRows2d(bld, labels.?, start, end, 1);
                break :blk try oneHotTargets(bld, label_chunk, vocab_row, chunk_rows, self.graph_config.vocab_size);
            } else blk: {
                const id_chunk = try sliceRows2d(bld, teacher_ids.?, start, end, teacher_top_k);
                const prob_chunk = try sliceRows2d(bld, teacher_probs.?, start, end, teacher_top_k);
                var mixed: ?NodeId = null;
                for (0..teacher_top_k) |ki| {
                    const label = try bld.sliceLastDim(id_chunk, @intCast(ki), @intCast(ki + 1));
                    const probability = try bld.sliceLastDim(prob_chunk, @intCast(ki), @intCast(ki + 1));
                    const one_hot = try oneHotTargets(bld, label, vocab_row, chunk_rows, self.graph_config.vocab_size);
                    const probability_bc = try broadcast2d(bld, probability, chunk_rows, self.graph_config.vocab_size);
                    const weighted = try bld.mul(one_hot, probability_bc);
                    mixed = if (mixed) |acc| try bld.add(acc, weighted) else weighted;
                }
                break :blk mixed orelse return error.InvalidTeacherDistillationTargets;
            };
            const chunk_loss = try bld.crossEntropyLoss(logits, dense_targets);
            const weight = try bld.scalarConst(.f32, @as(f32, @floatFromInt(chunk_rows)) / @as(f32, @floatFromInt(supervised_rows)));
            const weighted = try bld.mul(chunk_loss, weight);
            total_loss = if (total_loss) |acc| try bld.add(acc, weighted) else weighted;
        }
        return total_loss.?;
    }

    pub fn buildLogits(
        self: *GemmaAutodiffCtx,
        bld: *Builder,
        forward_output: ml.graph.NodeId,
    ) !ml.graph.NodeId {
        const out_shape = bld.graph.node(forward_output).output_shape;
        const total_rows: u32 = @intCast(out_shape.dim(0) * out_shape.dim(1));
        const hidden_size: u32 = @intCast(out_shape.dim(2));

        const hidden_flat = try bld.reshape(forward_output, Shape.init(.f32, &.{ @as(i64, @intCast(total_rows)), @as(i64, @intCast(hidden_size)) }));
        const lm_head_w = try self.buildLmHeadWeight(bld, hidden_size);
        const raw_logits = try bld.linearNoBias(hidden_flat, lm_head_w, total_rows, hidden_size, self.graph_config.vocab_size);
        const logits = try self.applyFinalLogitSoftcap(bld, raw_logits);
        self.lm_logits = logits;
        return logits;
    }

    fn applyFinalLogitSoftcap(self: *GemmaAutodiffCtx, bld: *Builder, logits: NodeId) !NodeId {
        const softcap = self.graph_config.final_logit_softcapping;
        if (softcap <= 0.0) return logits;
        const scale = try bld.scalarConst(.f32, softcap);
        return bld.mul(try bld.tanhOp(try bld.div(logits, scale)), scale);
    }

    fn buildLmHeadWeight(self: *GemmaAutodiffCtx, bld: *Builder, hidden_size: u32) !NodeId {
        if (self.graph_config.weight_tying) {
            var name_buf: [256]u8 = undefined;
            const name = try prefixedModelName(&name_buf, self.graph_config, "model.embed_tokens.weight");
            return bld.parameter(name, Shape.init(.f32, &.{ @as(i64, @intCast(self.graph_config.vocab_size)), @as(i64, @intCast(hidden_size)) }));
        }
        return bld.parameter("lm_head.weight", Shape.init(.f32, &.{ @as(i64, @intCast(self.graph_config.vocab_size)), @as(i64, @intCast(hidden_size)) }));
    }

    pub fn remapGraphNodes(ctx_opaque: *anyopaque, id_map: []const NodeId) anyerror!void {
        const self: *GemmaAutodiffCtx = @ptrCast(@alignCast(ctx_opaque));
        if (self.built) |*built| {
            built.input_ids_node = id_map[built.input_ids_node];
            if (built.rope_cos_node != ml.graph.null_node) built.rope_cos_node = id_map[built.rope_cos_node];
            if (built.rope_sin_node != ml.graph.null_node) built.rope_sin_node = id_map[built.rope_sin_node];
            built.output_node = id_map[built.output_node];
        }
        if (self.lm_logits) |node_id| self.lm_logits = id_map[node_id];
    }
};

const sparse_target_columns: i64 = 2;
const sparse_teacher_fixed_columns: u32 = 2;
const sparse_loss_chunk_rows: u32 = 1;

fn sliceRows2d(bld: *Builder, input: NodeId, start: u32, end: u32, columns: u32) !NodeId {
    var attrs = ml.graph.node.SliceAttrs{};
    attrs.num_axes = 2;
    attrs.starts[0] = start;
    attrs.starts[1] = 0;
    attrs.limits[0] = end;
    attrs.limits[1] = columns;
    attrs.strides[0] = 1;
    attrs.strides[1] = 1;
    return bld.graph.addNode(.{
        .op = .{ .slice = attrs },
        .output_shape = Shape.init(.f32, &.{ @as(i64, @intCast(end - start)), @as(i64, @intCast(columns)) }),
        .inputs = .{ input, ml.graph.null_node, ml.graph.null_node, ml.graph.null_node },
        .num_inputs = 1,
    });
}

fn broadcast2d(bld: *Builder, input: NodeId, rows: u32, columns: u32) !NodeId {
    const shape = Shape.init(.f32, &.{ @as(i64, @intCast(rows)), @as(i64, @intCast(columns)) });
    var attrs = ml.graph.node.BroadcastAttrs{ .target_shape = shape };
    attrs.broadcast_axes[0] = 0;
    attrs.broadcast_axes[1] = 1;
    attrs.num_axes = 2;
    return bld.graph.addNode(.{
        .op = .{ .broadcast_in_dim = attrs },
        .output_shape = shape,
        .inputs = .{ input, ml.graph.null_node, ml.graph.null_node, ml.graph.null_node },
        .num_inputs = 1,
    });
}

fn oneHotTargets(bld: *Builder, labels: NodeId, vocab_row: NodeId, rows: u32, vocab_size: u32) !NodeId {
    const labels_bc = try broadcast2d(bld, labels, rows, vocab_size);
    const vocab_bc = try broadcast2d(bld, vocab_row, rows, vocab_size);
    const diff = try bld.absOp(try bld.sub(labels_bc, vocab_bc));
    const half = try bld.scalarConst(.f32, 0.5);
    return bld.graph.addNode(.{
        .op = .{ .less_than = {} },
        .output_shape = Shape.init(.f32, &.{ @as(i64, @intCast(rows)), @as(i64, @intCast(vocab_size)) }),
        .inputs = .{ diff, half, ml.graph.null_node, ml.graph.null_node },
        .num_inputs = 2,
    });
}

fn prefixedModelName(buf: *[256]u8, config: gemma_graph.Config, name: []const u8) ![]const u8 {
    if (config.weight_prefix.len == 0 or !std.mem.startsWith(u8, name, "model.")) return name;
    return std.fmt.bufPrint(buf, "{s}.{s}", .{ config.weight_prefix, name["model.".len..] }) catch error.NameTooLong;
}

pub const OwnedTrainerInput = struct {
    input_ids: []i64,
    attention_mask: []f32,
    targets: []f32,
    supervised_tokens: usize,
    trainer_input: real_autodiff.TrainerInput,

    pub fn deinit(self: *OwnedTrainerInput, allocator: std.mem.Allocator) void {
        allocator.free(self.input_ids);
        allocator.free(self.attention_mask);
        allocator.free(self.targets);
        self.* = undefined;
    }
};

pub fn loadGraphConfig(allocator: std.mem.Allocator, model_dir: []const u8) !gemma_graph.Config {
    const config = try session_factory.loadGptConfigMetadataFromModelDir(allocator, model_dir);
    try gemma_graph.validateConfig(config);
    return config;
}

pub fn loadBackendForModelDir(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    backend_kind: BackendKind,
) !LoadedBackend {
    const session = switch (backend_kind) {
        .native => try session_factory.createNativeSession(allocator, model_dir),
        .metal => try session_factory.createMetalSession(allocator, model_dir),
    };
    errdefer session.close();
    const compute_backend = try session_factory.getComputeBackend(session, allocator);
    errdefer compute_backend.deinit();
    return .{
        .kind = backend_kind,
        .compute_backend = compute_backend,
        .session = session,
    };
}

pub fn makeTrainerInputForExample(
    allocator: std.mem.Allocator,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
) !OwnedTrainerInput {
    return makeTrainerInputForExampleWeighted(allocator, ctx, example, seq_len, null, null);
}

pub fn makeTrainerInputForExampleScaled(
    allocator: std.mem.Allocator,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    token_scale_override: ?f32,
) !OwnedTrainerInput {
    return makeTrainerInputForExampleWeighted(allocator, ctx, example, seq_len, token_scale_override, null);
}

pub fn makeTrainerInputForTokenLogprobGrads(
    allocator: std.mem.Allocator,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    logprob_grads: []const f32,
) !OwnedTrainerInput {
    const rows_f: f32 = @floatFromInt(seq_len);
    const token_scales = try allocator.alloc(f32, logprob_grads.len);
    defer allocator.free(token_scales);
    for (logprob_grads, 0..) |grad, idx| token_scales[idx] = -grad * rows_f;
    return makeTrainerInputForExampleWeighted(allocator, ctx, example, seq_len, null, token_scales);
}

fn makeTrainerInputForExampleWeighted(
    allocator: std.mem.Allocator,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    token_scale_override: ?f32,
    token_scales: ?[]const f32,
) !OwnedTrainerInput {
    const seq_len_usize: usize = @intCast(seq_len);
    const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
    const rows = seq_len_usize;
    const valid_tokens = try validatePreparedExample(example, rows, vocab_size);
    if (valid_tokens == 0) return error.NoSupervisedTokens;

    const input_ids = try allocator.alloc(i64, rows);
    errdefer allocator.free(input_ids);
    @memset(input_ids, 0);
    const attention_mask = try allocator.alloc(f32, rows);
    errdefer allocator.free(attention_mask);
    @memset(attention_mask, 0.0);

    const usable = @min(example.input_ids.len, rows);
    for (0..usable) |i| {
        input_ids[i] = example.input_ids[i];
        attention_mask[i] = 1.0;
    }

    const label_limit = @min(example.labels.len, usable);
    if (token_scales) |scales| {
        if (scales.len != valid_tokens) return error.GradientShapeMismatch;
    }

    const teacher_fields_present = example.teacher_top_k != 0 or
        example.teacher_top_k_token_ids.len != 0 or
        example.teacher_top_k_probs.len != 0;
    if (teacher_fields_present and !exampleHasTeacherTargets(example))
        return error.InvalidTeacherDistillationTargets;

    const sparse_hard_targets = token_scales == null and
        token_scale_override == null and
        !teacher_fields_present;
    const sparse_teacher_targets = token_scales == null and
        token_scale_override == null and
        teacher_fields_present;
    const sparse_teacher_columns = if (sparse_teacher_targets)
        try std.math.add(usize, sparse_teacher_fixed_columns, try std.math.mul(usize, 2, example.teacher_top_k))
    else
        0;
    const targets_shape = if (sparse_hard_targets)
        Shape.init(.f32, &.{ @as(i64, @intCast(valid_tokens)), sparse_target_columns })
    else if (sparse_teacher_targets)
        Shape.init(.f32, &.{ @as(i64, @intCast(valid_tokens)), @as(i64, @intCast(sparse_teacher_columns)) })
    else
        Shape.init(.f32, &.{ @as(i64, @intCast(rows)), @as(i64, @intCast(vocab_size)) });
    const target_elements = if (sparse_hard_targets)
        valid_tokens * @as(usize, @intCast(sparse_target_columns))
    else if (sparse_teacher_targets)
        valid_tokens * sparse_teacher_columns
    else
        rows * vocab_size;
    const targets = try allocator.alloc(f32, target_elements);
    errdefer allocator.free(targets);
    @memset(targets, 0.0);

    if (sparse_hard_targets) {
        var supervised_idx: usize = 0;
        for (1..label_limit) |i| {
            const label = example.labels[i];
            if (label < 0) continue;
            const token_idx: usize = @intCast(label);
            if (token_idx >= vocab_size) return error.LabelOutOfRange;
            // Decoder row i-1 predicts token i. Store only the rows that
            // participate in SFT so host and device memory stay O(tokens),
            // independent of vocabulary size.
            targets[supervised_idx * 2] = @floatFromInt(i - 1);
            targets[supervised_idx * 2 + 1] = @floatFromInt(token_idx);
            supervised_idx += 1;
        }
    } else if (sparse_teacher_targets) {
        try fillSparseTeacherTopKTargets(targets, sparse_teacher_columns, rows, vocab_size, example);
    } else {
        const default_row_scale: f32 = token_scale_override orelse
            @as(f32, @floatFromInt(rows)) / @as(f32, @floatFromInt(valid_tokens));

        var supervised_idx: usize = 0;
        for (1..label_limit) |i| {
            const label = example.labels[i];
            if (label < 0) continue;
            const idx: usize = @intCast(label);
            if (idx >= vocab_size) return error.LabelOutOfRange;
            const row_scale = if (token_scales) |scales| scales[supervised_idx] else default_row_scale;
            targets[(i - 1) * vocab_size + idx] = row_scale;
            supervised_idx += 1;
        }
    }

    return .{
        .input_ids = input_ids,
        .attention_mask = attention_mask,
        .targets = targets,
        .supervised_tokens = valid_tokens,
        .trainer_input = .{
            .ctx = @ptrCast(ctx),
            .build_forward = &GemmaAutodiffCtx.buildForward,
            .build_loss = &GemmaAutodiffCtx.buildLoss,
            .input_ids = input_ids,
            .attention_mask = attention_mask,
            .targets = targets,
            .targets_shape = targets_shape,
            .batch = 1,
            .seq_len = seq_len,
            .bind_arch_inputs = null,
            .remap_graph_nodes = &GemmaAutodiffCtx.remapGraphNodes,
        },
    };
}

fn validatePreparedExample(
    example: *const gemma4.PreparedExampleInput,
    rows: usize,
    vocab_size: usize,
) !usize {
    if (example.labels.len != example.input_ids.len) return error.InvalidPreparedExampleShape;
    const usable = @min(example.input_ids.len, rows);
    if (example.num_input_tokens > usable) return error.PreparedExampleExceedsSequenceLength;

    for (example.input_ids[0..usable]) |token_id| {
        if (token_id < 0 or @as(usize, @intCast(token_id)) >= vocab_size) return error.InputTokenOutOfRange;
    }

    var supervised_tokens: usize = 0;
    for (example.labels, 0..) |label, row| {
        if (label == -100) continue;
        if (label < 0) return error.InvalidPreparedLabel;
        if (row == 0) return error.InvalidCausalLabel;
        if (@as(usize, @intCast(label)) >= vocab_size) return error.LabelOutOfRange;
        if (row >= usable) return error.PreparedExampleExceedsSequenceLength;
        supervised_tokens += 1;
    }
    if (supervised_tokens != example.num_supervised_tokens) return error.SupervisedTokenCountMismatch;
    return supervised_tokens;
}

pub fn makeTrainerInputForLogprobCoeff(
    allocator: std.mem.Allocator,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    logprob_coeff: f32,
) !OwnedTrainerInput {
    const rows_f: f32 = @floatFromInt(seq_len);
    return makeTrainerInputForExampleScaled(allocator, ctx, example, seq_len, -logprob_coeff * rows_f);
}

pub fn tokenLogprobsForPromptCompletion(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    prompt: []const i32,
    completion: []const i32,
    seq_len: u32,
    out_logps: []f32,
) !void {
    if (completion.len != out_logps.len) return error.LogpLenMismatch;
    if (prompt.len == 0) return error.EmptyPrompt;
    if (completion.len == 0) return error.EmptyCompletion;
    const total_len = prompt.len + completion.len;
    if (total_len > seq_len) return error.SequenceTooLong;

    const joined = try concatPromptCompletion(allocator, prompt, completion);
    defer allocator.free(joined);
    const logits = try executeLogitsForInputIds(allocator, trainer, ctx, joined, seq_len);
    defer allocator.free(logits);
    const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
    for (completion, 0..) |token_id, comp_idx| {
        const row_idx = prompt.len + comp_idx - 1;
        const row = logits[row_idx * vocab_size ..][0..vocab_size];
        out_logps[comp_idx] = logProbAtToken(row, @intCast(token_id));
    }
}

pub fn sampleCompletionRanked(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    prompt: []const i32,
    seq_len: u32,
    max_completion_tokens: usize,
    rank: usize,
    eos_token_id: ?i32,
    out_tokens: *std.ArrayList(i32),
    out_logps: *std.ArrayList(f32),
) !void {
    if (prompt.len == 0) return error.EmptyPrompt;
    var seq = std.ArrayList(i32).empty;
    defer seq.deinit(allocator);
    try seq.appendSlice(allocator, prompt);

    var step: usize = 0;
    while (step < max_completion_tokens and seq.items.len < seq_len) : (step += 1) {
        const logits = try executeLogitsForInputIds(allocator, trainer, ctx, seq.items, seq_len);
        defer allocator.free(logits);
        const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
        const row = logits[(seq.items.len - 1) * vocab_size ..][0..vocab_size];
        const token_id = try selectRankedToken(allocator, row, rank);
        const token_logp = logProbAtToken(row, token_id);
        try out_tokens.append(allocator, @intCast(token_id));
        try out_logps.append(allocator, token_logp);
        try seq.append(allocator, @intCast(token_id));
        if (eos_token_id) |eos_id| if (token_id == @as(usize, @intCast(eos_id))) break;
    }
    if (out_tokens.items.len == 0) return error.EmptyCompletion;
}

pub fn fillTeacherTopKTargets(
    targets: []f32,
    rows: usize,
    vocab_size: usize,
    example: *const gemma4.PreparedExampleInput,
) !bool {
    const top_k = example.teacher_top_k;
    if (top_k == 0) return false;
    if (example.teacher_top_k_token_ids.len != example.teacher_top_k_probs.len) return error.InvalidTeacherDistillationTargets;
    if (example.teacher_top_k_token_ids.len % top_k != 0) return error.InvalidTeacherDistillationTargets;
    const teacher_rows = example.teacher_top_k_token_ids.len / top_k;
    if (teacher_rows == 0) return false;
    if (teacher_rows < @min(example.labels.len, rows)) return error.InvalidTeacherDistillationTargets;

    var active_rows: usize = 0;
    for (1..@min(example.labels.len, rows)) |row| {
        if (example.labels[row] != -100) active_rows += 1;
    }
    if (active_rows == 0) return false;
    const temperature = example.teacher_temperature;
    if (!std.math.isFinite(temperature) or temperature <= 0) return error.InvalidTeacherTemperature;
    const distillation_scale = temperature * temperature;
    const row_scale = (@as(f32, @floatFromInt(rows)) / @as(f32, @floatFromInt(active_rows))) * distillation_scale;

    for (1..@min(example.labels.len, rows)) |row| {
        if (example.labels[row] == -100) continue;
        const base = (row - 1) * top_k;
        var prob_sum: f32 = 0.0;
        for (0..top_k) |ki| {
            const prob = example.teacher_top_k_probs[base + ki];
            if (!std.math.isFinite(prob) or prob < 0) return error.InvalidTeacherDistillationTargets;
            prob_sum += prob;
        }
        if (!std.math.isFinite(prob_sum) or prob_sum <= 0) return error.InvalidTeacherDistillationTargets;
        for (0..top_k) |ki| {
            const token_id = example.teacher_top_k_token_ids[base + ki];
            if (token_id < 0) continue;
            const idx: usize = @intCast(token_id);
            if (idx >= vocab_size) return error.LabelOutOfRange;
            targets[(row - 1) * vocab_size + idx] += row_scale * (example.teacher_top_k_probs[base + ki] / prob_sum);
        }
    }
    return true;
}

fn fillSparseTeacherTopKTargets(
    targets: []f32,
    target_columns: usize,
    rows: usize,
    vocab_size: usize,
    example: *const gemma4.PreparedExampleInput,
) !void {
    const top_k = example.teacher_top_k;
    const expected_columns = std.math.add(usize, sparse_teacher_fixed_columns, std.math.mul(usize, 2, top_k) catch return error.InvalidTeacherDistillationTargets) catch
        return error.InvalidTeacherDistillationTargets;
    if (top_k == 0 or target_columns != expected_columns) return error.InvalidTeacherDistillationTargets;
    if (example.teacher_top_k_token_ids.len != example.teacher_top_k_probs.len) return error.InvalidTeacherDistillationTargets;
    if (example.teacher_top_k_token_ids.len % top_k != 0) return error.InvalidTeacherDistillationTargets;
    const teacher_rows = example.teacher_top_k_token_ids.len / top_k;
    if (teacher_rows == 0 or teacher_rows < @min(example.labels.len, rows)) return error.InvalidTeacherDistillationTargets;

    var active_rows: usize = 0;
    for (1..@min(example.labels.len, rows)) |row| {
        if (example.labels[row] != -100) active_rows += 1;
    }
    if (active_rows == 0 or targets.len != active_rows * target_columns) return error.InvalidTeacherDistillationTargets;
    const temperature = example.teacher_temperature;
    if (!std.math.isFinite(temperature) or temperature <= 0) return error.InvalidTeacherTemperature;
    const distillation_scale = temperature * temperature;

    var active_row: usize = 0;
    for (1..@min(example.labels.len, rows)) |row| {
        if (example.labels[row] == -100) continue;
        // Decoder row `row - 1` predicts label token `row`, so read and
        // supervise the teacher distribution from that predictor row.
        const teacher_base = (row - 1) * top_k;
        var prob_sum: f32 = 0.0;
        for (0..top_k) |ki| {
            const prob = example.teacher_top_k_probs[teacher_base + ki];
            if (!std.math.isFinite(prob) or prob < 0) return error.InvalidTeacherDistillationTargets;
            const token_id = example.teacher_top_k_token_ids[teacher_base + ki];
            if (token_id < 0 or @as(usize, @intCast(token_id)) >= vocab_size) return error.LabelOutOfRange;
            prob_sum += prob;
        }
        if (!std.math.isFinite(prob_sum) or prob_sum <= 0) return error.InvalidTeacherDistillationTargets;

        const target_base = active_row * target_columns;
        targets[target_base] = @floatFromInt(row - 1);
        targets[target_base + 1] = temperature;
        for (0..top_k) |ki| {
            targets[target_base + 2 + ki] = @floatFromInt(example.teacher_top_k_token_ids[teacher_base + ki]);
            targets[target_base + 2 + top_k + ki] = distillation_scale * (example.teacher_top_k_probs[teacher_base + ki] / prob_sum);
        }
        active_row += 1;
    }
}

fn exampleHasTeacherTargets(example: *const gemma4.PreparedExampleInput) bool {
    return example.teacher_top_k > 0 and
        example.teacher_top_k_token_ids.len > 0 and
        example.teacher_top_k_probs.len > 0;
}

pub fn materializeTeacherTopKTargets(
    allocator: std.mem.Allocator,
    base_model_dir: []const u8,
    prepared: *gemma4.PreparedInputsSummary,
    backend_kind: BackendKind,
    options: TeacherTopKOptions,
) !TeacherTopKSummary {
    if (gemma4.preparedExamplesHaveMedia(prepared.examples)) return error.MultimodalTeacherMaterializationNotYetSupported;
    if (options.top_k == 0) return error.InvalidTeacherTopK;
    if (!std.math.isFinite(options.temperature) or options.temperature <= 0) return error.InvalidTeacherTemperature;

    var teacher_provenance = try gemma4.fingerprintGemma4Model(allocator, base_model_dir);
    defer teacher_provenance.deinit(allocator);
    try gemma4.validatePreparedModelProvenance(prepared.*, teacher_provenance);

    const graph_config = try loadGraphConfig(allocator, base_model_dir);
    const vocab_size: usize = @intCast(graph_config.vocab_size);
    if (options.top_k > vocab_size) return error.InvalidTeacherTopK;
    const seq_len: usize = @intCast(try gemma4.validatePreparedSequenceAdmission(prepared.*, graph_config.max_position_embeddings));

    var backend = try loadBackendForModelDir(allocator, base_model_dir, backend_kind);
    defer backend.deinit();

    var graph = Graph.init(allocator);
    defer graph.deinit();
    var bld = Builder.init(&graph);
    const ids_shape = Shape.init(.f32, &.{ 1, @as(i64, @intCast(seq_len)) });
    const input_ids_node = try bld.parameter("__teacher_input_ids", ids_shape);
    const attention_mask_node = try bld.parameter("__teacher_attention_mask", ids_shape);
    var ctx = GemmaAutodiffCtx.init(graph_config);
    const hidden = try GemmaAutodiffCtx.buildForward(@ptrCast(&ctx), &bld, input_ids_node, attention_mask_node, 1, @intCast(seq_len));
    const logits_node = try ctx.buildLogits(&bld, hidden);
    try graph.markOutput(logits_node);

    const limit = if (options.max_examples > 0 and options.max_examples < prepared.examples.len) options.max_examples else prepared.examples.len;
    var summary = TeacherTopKSummary{
        .top_k = options.top_k,
        .temperature = options.temperature,
    };
    for (prepared.examples[0..limit]) |*example| {
        if (example.input_ids.len == 0) continue;
        const logits = try forwardTeacherLogitsForExample(
            allocator,
            backend.backendPtr(),
            &graph,
            input_ids_node,
            attention_mask_node,
            example,
            seq_len,
            vocab_size,
        );
        defer allocator.free(logits);

        const token_ids = try allocator.alloc(i32, seq_len * options.top_k);
        errdefer allocator.free(token_ids);
        const probs = try allocator.alloc(f32, seq_len * options.top_k);
        errdefer allocator.free(probs);
        try fillTopKFromLogits(token_ids, probs, logits, seq_len, vocab_size, options.top_k, options.temperature);

        replaceTeacherTargets(allocator, example, token_ids, probs, options.top_k, options.temperature);
        summary.examples_written += 1;
        summary.supervised_tokens_seen += example.num_supervised_tokens;
    }
    summary.examples_seen = limit;
    try gemma4.refreshPreparedExamplesFingerprint(allocator, prepared);
    if (gemma4.preparedExamplesHaveTeacherTargets(prepared.examples)) {
        try gemma4.bindPreparedTeacherProvenance(allocator, prepared, teacher_provenance, null);
    }
    return summary;
}

fn forwardTeacherLogitsForExample(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    graph: *const Graph,
    input_ids_node: NodeId,
    attention_mask_node: NodeId,
    example: *const gemma4.PreparedExampleInput,
    seq_len: usize,
    vocab_size: usize,
) ![]f32 {
    const input_ids = try allocator.alloc(f32, seq_len);
    defer allocator.free(input_ids);
    const attention_mask = try allocator.alloc(f32, seq_len);
    defer allocator.free(attention_mask);
    @memset(input_ids, 0.0);
    @memset(attention_mask, 0.0);
    const usable = @min(example.input_ids.len, seq_len);
    for (0..usable) |idx| {
        input_ids[idx] = @floatFromInt(example.input_ids[idx]);
        attention_mask[idx] = 1.0;
    }

    const dims = [_]i32{ 1, @intCast(seq_len) };
    const input_ct = try cb.fromFloat32Shape(input_ids, &dims);
    defer cb.free(input_ct);
    const mask_ct = try cb.fromFloat32Shape(attention_mask, &dims);
    defer cb.free(mask_ct);
    const rt_inputs = [_]interpreter.RuntimeInput{
        .{ .node_id = input_ids_node, .value = input_ct },
        .{ .node_id = attention_mask_node, .value = mask_ct },
    };
    var exec_result = try interpreter.execute(allocator, graph, cb, .{ .runtime_inputs = &rt_inputs });
    defer exec_result.deinit(cb);
    if (exec_result.outputs.len != 1) return error.InvalidTeacherLogits;
    const logits = try cb.toFloat32(exec_result.outputs[0], allocator);
    errdefer allocator.free(logits);
    if (logits.len != seq_len * vocab_size) return error.InvalidTeacherLogits;
    return logits;
}

pub fn fillTopKFromLogits(
    token_ids: []i32,
    probs: []f32,
    logits: []const f32,
    rows: usize,
    vocab_size: usize,
    top_k: usize,
    temperature: f32,
) !void {
    if (token_ids.len != rows * top_k or probs.len != rows * top_k) return error.InvalidTeacherTopK;
    var row: usize = 0;
    while (row < rows) : (row += 1) {
        const out_base = row * top_k;
        for (0..top_k) |slot| {
            token_ids[out_base + slot] = -1;
            probs[out_base + slot] = -std.math.inf(f32);
        }
        const row_logits = logits[row * vocab_size ..][0..vocab_size];
        for (row_logits, 0..) |logit, token_idx| {
            if (std.math.isNan(logit)) continue;
            const score = logit / temperature;
            var insert_at: ?usize = null;
            for (0..top_k) |slot| {
                if (score > probs[out_base + slot]) {
                    insert_at = slot;
                    break;
                }
            }
            if (insert_at) |slot| {
                var move_idx = top_k - 1;
                while (move_idx > slot) : (move_idx -= 1) {
                    probs[out_base + move_idx] = probs[out_base + move_idx - 1];
                    token_ids[out_base + move_idx] = token_ids[out_base + move_idx - 1];
                }
                probs[out_base + slot] = score;
                token_ids[out_base + slot] = @intCast(token_idx);
            }
        }

        const max_score = probs[out_base];
        if (token_ids[out_base] < 0 or max_score == -std.math.inf(f32)) return error.InvalidTeacherLogits;
        var sum_exp: f32 = 0.0;
        for (0..top_k) |slot| {
            const value = @exp(probs[out_base + slot] - max_score);
            probs[out_base + slot] = value;
            sum_exp += value;
        }
        if (sum_exp <= 0 or std.math.isNan(sum_exp)) return error.InvalidTeacherLogits;
        for (0..top_k) |slot| probs[out_base + slot] /= sum_exp;
    }
}

pub fn replaceTeacherTargets(
    allocator: std.mem.Allocator,
    example: *gemma4.PreparedExampleInput,
    token_ids: []i32,
    probs: []f32,
    top_k: usize,
    temperature: f32,
) void {
    if (example.teacher_top_k_token_ids.len > 0) allocator.free(example.teacher_top_k_token_ids);
    if (example.teacher_top_k_probs.len > 0) allocator.free(example.teacher_top_k_probs);
    example.teacher_top_k_token_ids = token_ids;
    example.teacher_top_k_probs = probs;
    example.teacher_top_k = top_k;
    example.teacher_temperature = temperature;
}

pub fn findFirstSupervisedExample(examples: []const gemma4.PreparedExampleInput) ?*const gemma4.PreparedExampleInput {
    for (examples) |*example| {
        if (example.num_supervised_tokens > 0) return example;
    }
    return null;
}

pub fn initializeTrainerFromAdapterDir(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    adapter_model_dir: []const u8,
    bootstrap_example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
) !void {
    // Fail before graph/backend mutation if the adapter config and checkpoint
    // do not describe the same exact A/B/DoRA target inventory.
    try gemma4.validateLoRAAdapterInventory(allocator, adapter_model_dir);
    var bootstrap = try makeTrainerInputForExample(allocator, ctx, bootstrap_example, seq_len);
    defer bootstrap.deinit(allocator);
    try trainer.ensureGraphBuilt(bootstrap.trainer_input);

    var inspection = try gemma4.inspectCheckpoint(allocator, adapter_model_dir);
    defer gemma4.freeInspectionSummary(allocator, &inspection);
    const checkpoint_path = inspection.adapter_checkpoint_path orelse return error.MissingAdapterCheckpoint;

    var source = try SafetensorsSource.initAbsolute(allocator, checkpoint_path);
    defer source.weightSource().deinit();
    const ws = source.weightSource();

    for (trainer.lora_params.items) |*slot| {
        const tensor_name = try mapTrainerSlotNameToGemmaAdapterTensor(allocator, slot.name);
        defer allocator.free(tensor_name);

        var loaded = try ws.getTensor(tensor_name);
        defer loaded.deinit();

        if (loaded.tensor.shape.len != slot.dims.len) return error.AdapterShapeMismatch;
        for (slot.dims, 0..) |want_dim, idx| {
            if (loaded.tensor.shape[idx] != @as(i64, want_dim)) return error.AdapterShapeMismatch;
        }

        try copyTensorFloat32(slot.weights, &loaded.tensor);
        @memset(slot.grad_accum, 0.0);
    }
}

pub fn trainPreparedExamples(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    examples: []const gemma4.PreparedExampleInput,
    max_examples: usize,
    seq_len: u32,
) !CausalLmMetrics {
    return runPreparedExamples(allocator, trainer, ctx, examples, max_examples, seq_len, .train);
}

pub fn evaluatePreparedExamples(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    examples: []const gemma4.PreparedExampleInput,
    max_examples: usize,
    seq_len: u32,
) !CausalLmMetrics {
    return runPreparedExamples(allocator, trainer, ctx, examples, max_examples, seq_len, .eval);
}

fn runPreparedExamples(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    examples: []const gemma4.PreparedExampleInput,
    max_examples: usize,
    seq_len: u32,
    mode: enum { train, eval },
) !CausalLmMetrics {
    var metrics = CausalLmMetrics{};
    const limit = if (max_examples > 0 and max_examples < examples.len) max_examples else examples.len;
    var total_weighted_loss: f64 = 0;
    var total_weighted_grad_norm: f64 = 0;
    var total_weighted_teacher_temperature: f64 = 0;

    for (examples[0..limit]) |*example| {
        const effective_supervised_tokens = try validatePreparedExample(
            example,
            @intCast(seq_len),
            @intCast(ctx.graph_config.vocab_size),
        );
        if (effective_supervised_tokens == 0) continue;
        var input = try makeTrainerInputForExample(allocator, ctx, example, seq_len);
        defer input.deinit(allocator);
        const step = switch (mode) {
            .train => try trainer.step(input.trainer_input),
            .eval => try trainer.evaluate(input.trainer_input),
        };
        const weight: f64 = @floatFromInt(input.supervised_tokens);
        metrics.examples_seen += 1;
        metrics.supervised_tokens_seen += input.supervised_tokens;
        if (exampleHasTeacherTargets(example)) {
            metrics.teacher_examples_seen += 1;
            metrics.teacher_supervised_tokens_seen += input.supervised_tokens;
            total_weighted_teacher_temperature += @as(f64, example.teacher_temperature) * weight;
        }
        total_weighted_loss += @as(f64, step.loss) * weight;
        total_weighted_grad_norm += @as(f64, step.grad_norm) * weight;
        recordStepExecutionEvidence(&metrics, step);
        if (step.optimizer_stepped) metrics.optimizer_steps += 1;
    }

    // Do not discard the final partial accumulation window at an epoch
    // boundary. Besides preserving every example's gradient, this leaves the
    // trainer in the only state that can be checkpointed safely.
    if (mode == .train) {
        if (try trainer.flushAccumulatedGradients()) |_| {
            metrics.optimizer_steps += 1;
            if (trainer.config.strict_metal_execution) metrics.metal_optimizer_steps += 1;
        }
    }

    if (metrics.supervised_tokens_seen > 0) {
        const denom: f64 = @floatFromInt(metrics.supervised_tokens_seen);
        metrics.average_loss = total_weighted_loss / denom;
        metrics.mean_grad_norm = total_weighted_grad_norm / denom;
    }
    if (metrics.teacher_supervised_tokens_seen > 0) {
        metrics.mean_teacher_temperature = total_weighted_teacher_temperature / @as(f64, @floatFromInt(metrics.teacher_supervised_tokens_seen));
    }
    return metrics;
}

pub fn saveTrainerAsGemmaBundle(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    base_model_dir: []const u8,
    adapter_model_dir: []const u8,
    out_dir: []const u8,
) !void {
    try trainer.syncDeviceTrainablesToHost();

    var adapter_inspect = try gemma4.inspectCheckpoint(allocator, adapter_model_dir);
    defer gemma4.freeInspectionSummary(allocator, &adapter_inspect);

    var tensors = std.ArrayList(WriteTensorF32).empty;
    defer tensors.deinit(allocator);
    var owned_names = std.ArrayList([]const u8).empty;
    defer {
        for (owned_names.items) |name| allocator.free(name);
        owned_names.deinit(allocator);
    }
    var owned_shapes = std.ArrayList([]const usize).empty;
    defer {
        for (owned_shapes.items) |shape| allocator.free(shape);
        owned_shapes.deinit(allocator);
    }

    const slot_count = trainer.lora_params.items.len;
    try tensors.ensureTotalCapacity(allocator, slot_count);
    try owned_names.ensureTotalCapacity(allocator, slot_count);
    try owned_shapes.ensureTotalCapacity(allocator, slot_count);

    for (trainer.lora_params.items) |slot| {
        const owned = blk: {
            const mapped_name = try mapTrainerSlotNameToGemmaAdapterTensor(allocator, slot.name);
            errdefer allocator.free(mapped_name);
            const dims = try dimsToUsize(allocator, slot.dims);
            break :blk .{ .name = mapped_name, .dims = dims };
        };
        owned_names.appendAssumeCapacity(owned.name);
        owned_shapes.appendAssumeCapacity(owned.dims);
        tensors.appendAssumeCapacity(.{
            .name = owned.name,
            .shape = owned.dims,
            .data = slot.weights,
        });
    }

    const base_name = adapter_inspect.base_model_name_or_path orelse base_model_dir;
    const rank = adapter_inspect.lora_rank orelse return error.MissingAdapterConfig;
    const alpha = @as(f32, @floatCast(adapter_inspect.lora_alpha orelse return error.MissingAdapterConfig));
    const target_modules = adapter_inspect.target_modules orelse gemma4.default_lora_target_modules[0..];
    try writeAndPublishGemmaBundle(allocator, out_dir, tensors.items, .{
        .base_model_name_or_path = base_name,
        .base_model_sha256 = adapter_inspect.base_model_sha256,
        .tokenizer_sha256 = adapter_inspect.tokenizer_sha256,
        .chat_template_sha256 = adapter_inspect.chat_template_sha256,
        .rank = rank,
        .alpha = alpha,
        .target_modules = target_modules,
        .recursive_config = .{
            .enabled = adapter_inspect.recursive_lora_enabled,
            .source_num_layers = adapter_inspect.recursive_source_num_layers orelse 0,
            .shared_block_size = adapter_inspect.recursive_shared_block_size orelse 0,
            .loop_count = adapter_inspect.recursive_loop_count orelse 0,
            .init_strategy = adapter_inspect.recursive_init_strategy orelse "average_residual_svd",
        },
        .tokenizer_config_path = adapter_inspect.tokenizer_config_path,
        .tokenizer_path = adapter_inspect.tokenizer_path,
        .special_tokens_map_path = adapter_inspect.special_tokens_map_path,
    });
}

pub fn sequenceLogprobForExample(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
) !f32 {
    // Scoring needs the full vocabulary logits node. Force the dense graph
    // signature used by preference training; ordinary SFT uses the bounded
    // sparse-loss graph and intentionally does not materialize this tensor.
    var owned = try makeTrainerInputForExampleScaled(allocator, ctx, example, seq_len, 1.0);
    defer owned.deinit(allocator);
    try trainer.ensureGraphBuilt(owned.trainer_input);

    var gs = &trainer.graph_state.?;
    const logits_node = ctx.lm_logits orelse return error.MissingTrainerLogitsNode;

    var rt = std.AutoHashMapUnmanaged(NodeId, CT).empty;
    defer {
        var it = rt.iterator();
        while (it.next()) |entry| trainer.compute_backend.free(entry.value_ptr.*);
        rt.deinit(allocator);
    }

    const input_placeholder = graph_input_binder.PlaceholderInfo{
        .node_id = gs.input_ids_node,
        .name = "__input_ids",
        .shape = gs.graph.node(gs.input_ids_node).output_shape,
    };
    const mask_placeholder = graph_input_binder.PlaceholderInfo{
        .node_id = gs.attention_mask_node,
        .name = "__attention_mask",
        .shape = gs.graph.node(gs.attention_mask_node).output_shape,
    };

    const input_ct = try graph_input_binder.bindI64(trainer.compute_backend, allocator, input_placeholder, owned.input_ids);
    try rt.put(allocator, gs.input_ids_node, input_ct);
    const mask_ct = try graph_input_binder.bindF32(trainer.compute_backend, allocator, mask_placeholder, owned.attention_mask);
    try rt.put(allocator, gs.attention_mask_node, mask_ct);

    for (trainer.lora_params.items) |slot| {
        const dims = try allocator.alloc(i32, slot.dims.len);
        defer allocator.free(dims);
        @memcpy(dims, slot.dims);
        const ct = try trainer.compute_backend.fromFloat32Shape(slot.weights, dims);
        try rt.put(allocator, slot.node_id, ct);
    }

    const saved_outputs = try allocator.dupe(NodeId, gs.graph.outputs.items);
    defer {
        gs.graph.outputs.clearRetainingCapacity();
        for (saved_outputs) |node_id| gs.graph.outputs.append(allocator, node_id) catch {};
        allocator.free(saved_outputs);
    }
    gs.graph.outputs.clearRetainingCapacity();
    try gs.graph.markOutput(logits_node);

    var rt_inputs = std.ArrayList(interpreter.RuntimeInput).empty;
    defer rt_inputs.deinit(allocator);
    {
        var it = rt.iterator();
        while (it.next()) |entry| {
            try rt_inputs.append(allocator, .{
                .node_id = entry.key_ptr.*,
                .value = entry.value_ptr.*,
            });
        }
    }

    var exec_result = try interpreter.execute(allocator, &gs.graph, trainer.compute_backend, .{
        .runtime_inputs = rt_inputs.items,
    });
    defer exec_result.deinit(trainer.compute_backend);

    const logits = try trainer.compute_backend.toFloat32(exec_result.outputs[0], allocator);
    defer allocator.free(logits);

    const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
    var sum_logp: f32 = 0.0;
    const rows = @min(example.labels.len, @as(usize, @intCast(seq_len)));
    for (1..rows) |row_idx| {
        const label = example.labels[row_idx];
        if (label < 0) continue;
        const token_idx: usize = @intCast(label);
        if (token_idx >= vocab_size) return error.LabelOutOfRange;
        const predictor_row = row_idx - 1;
        const row = logits[predictor_row * vocab_size ..][0..vocab_size];
        sum_logp += logProbAtToken(row, token_idx);
    }
    return sum_logp;
}

fn executeLogitsForInputIds(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    raw_input_ids: []const i32,
    seq_len: u32,
) ![]f32 {
    const rows: usize = @intCast(seq_len);
    if (raw_input_ids.len == 0) return error.EmptyPrompt;
    if (raw_input_ids.len > rows) return error.SequenceTooLong;

    const input_ids = try allocator.alloc(i64, rows);
    defer allocator.free(input_ids);
    @memset(input_ids, 0);
    const attention_mask = try allocator.alloc(f32, rows);
    defer allocator.free(attention_mask);
    @memset(attention_mask, 0.0);
    for (raw_input_ids, 0..) |token_id, idx| {
        input_ids[idx] = token_id;
        attention_mask[idx] = 1.0;
    }

    const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
    const targets = try allocator.alloc(f32, rows * vocab_size);
    defer allocator.free(targets);
    @memset(targets, 0.0);

    const trainer_input = real_autodiff.TrainerInput{
        .ctx = @ptrCast(ctx),
        .build_forward = &GemmaAutodiffCtx.buildForward,
        .build_loss = &GemmaAutodiffCtx.buildLoss,
        .input_ids = input_ids,
        .attention_mask = attention_mask,
        .targets = targets,
        .targets_shape = Shape.init(.f32, &.{ @as(i64, @intCast(rows)), @as(i64, @intCast(vocab_size)) }),
        .batch = 1,
        .seq_len = seq_len,
        .bind_arch_inputs = null,
        .remap_graph_nodes = &GemmaAutodiffCtx.remapGraphNodes,
    };

    try trainer.ensureGraphBuilt(trainer_input);
    var gs = &trainer.graph_state.?;
    const logits_node = ctx.lm_logits orelse return error.MissingTrainerLogitsNode;

    var rt = std.AutoHashMapUnmanaged(NodeId, CT).empty;
    defer {
        var it = rt.iterator();
        while (it.next()) |entry| trainer.compute_backend.free(entry.value_ptr.*);
        rt.deinit(allocator);
    }

    const input_placeholder = graph_input_binder.PlaceholderInfo{
        .node_id = gs.input_ids_node,
        .name = "__input_ids",
        .shape = gs.graph.node(gs.input_ids_node).output_shape,
    };
    const mask_placeholder = graph_input_binder.PlaceholderInfo{
        .node_id = gs.attention_mask_node,
        .name = "__attention_mask",
        .shape = gs.graph.node(gs.attention_mask_node).output_shape,
    };
    const targets_placeholder = graph_input_binder.PlaceholderInfo{
        .node_id = gs.targets_node,
        .name = "__targets",
        .shape = gs.graph.node(gs.targets_node).output_shape,
    };

    const input_ct = try graph_input_binder.bindI64(trainer.compute_backend, allocator, input_placeholder, input_ids);
    try rt.put(allocator, gs.input_ids_node, input_ct);
    const mask_ct = try graph_input_binder.bindF32(trainer.compute_backend, allocator, mask_placeholder, attention_mask);
    try rt.put(allocator, gs.attention_mask_node, mask_ct);
    const targets_ct = try graph_input_binder.bindF32(trainer.compute_backend, allocator, targets_placeholder, targets);
    try rt.put(allocator, gs.targets_node, targets_ct);

    for (trainer.lora_params.items) |slot| {
        const dims = try allocator.alloc(i32, slot.dims.len);
        defer allocator.free(dims);
        @memcpy(dims, slot.dims);
        const ct = try trainer.compute_backend.fromFloat32Shape(slot.weights, dims);
        try rt.put(allocator, slot.node_id, ct);
    }

    const saved_outputs = try allocator.dupe(NodeId, gs.graph.outputs.items);
    defer {
        gs.graph.outputs.clearRetainingCapacity();
        for (saved_outputs) |node_id| gs.graph.outputs.append(allocator, node_id) catch {};
        allocator.free(saved_outputs);
    }
    gs.graph.outputs.clearRetainingCapacity();
    try gs.graph.markOutput(logits_node);

    var rt_inputs = std.ArrayList(interpreter.RuntimeInput).empty;
    defer rt_inputs.deinit(allocator);
    {
        var it = rt.iterator();
        while (it.next()) |entry| {
            try rt_inputs.append(allocator, .{
                .node_id = entry.key_ptr.*,
                .value = entry.value_ptr.*,
            });
        }
    }

    var exec_result = try interpreter.execute(allocator, &gs.graph, trainer.compute_backend, .{
        .runtime_inputs = rt_inputs.items,
    });
    defer exec_result.deinit(trainer.compute_backend);
    return trainer.compute_backend.toFloat32(exec_result.outputs[0], allocator);
}

fn concatPromptCompletion(allocator: std.mem.Allocator, prompt: []const i32, completion: []const i32) ![]i32 {
    const out = try allocator.alloc(i32, prompt.len + completion.len);
    @memcpy(out[0..prompt.len], prompt);
    @memcpy(out[prompt.len..], completion);
    return out;
}

fn selectRankedToken(allocator: std.mem.Allocator, logits: []const f32, rank: usize) !usize {
    const Entry = struct {
        idx: usize,
        value: f32,
    };
    var entries = try allocator.alloc(Entry, logits.len);
    defer allocator.free(entries);
    for (logits, 0..) |value, idx| {
        entries[idx] = .{ .idx = idx, .value = value };
    }
    std.sort.heap(Entry, entries, {}, struct {
        fn lessThan(_: void, lhs: Entry, rhs: Entry) bool {
            return lhs.value > rhs.value;
        }
    }.lessThan);
    return entries[@min(rank, entries.len - 1)].idx;
}

fn logProbAtToken(logits: []const f32, token_id: usize) f32 {
    var max_logit = logits[0];
    for (logits[1..]) |value| {
        if (value > max_logit) max_logit = value;
    }
    var sum_exp: f64 = 0.0;
    for (logits) |value| {
        sum_exp += @exp(@as(f64, value - max_logit));
    }
    const log_z = @as(f64, max_logit) + @log(sum_exp);
    return @as(f32, @floatCast(@as(f64, logits[token_id]) - log_z));
}

const WriteTensorF32 = struct {
    name: []const u8,
    shape: []const usize,
    data: []const f32,
};

const GemmaBundleWriteSpec = struct {
    base_model_name_or_path: []const u8,
    base_model_sha256: ?[]const u8 = null,
    tokenizer_sha256: ?[]const u8 = null,
    chat_template_sha256: ?[]const u8 = null,
    rank: usize,
    alpha: f32,
    target_modules: []const []const u8,
    recursive_config: @import("recursive_lora.zig").Config = .{},
    tokenizer_config_path: ?[]const u8 = null,
    tokenizer_path: ?[]const u8 = null,
    special_tokens_map_path: ?[]const u8 = null,
};

fn writeAndPublishGemmaBundle(
    allocator: std.mem.Allocator,
    out_dir: []const u8,
    tensors: []const WriteTensorF32,
    spec: GemmaBundleWriteSpec,
) !void {
    var publication = artifact_publication.ImmutableDirectoryPublication.init(allocator, compat.io(), out_dir) catch |err| switch (err) {
        error.Gemma4RunOutputAlreadyExists => return error.GemmaBundleOutputAlreadyExists,
        else => return err,
    };
    defer publication.deinit();
    try publication.createStaging();

    const adapter_checkpoint_path = try std.fs.path.join(allocator, &.{ publication.staging_dir, gemma4.adapter_checkpoint_file_name });
    defer allocator.free(adapter_checkpoint_path);
    const adapter_config_path = try std.fs.path.join(allocator, &.{ publication.staging_dir, gemma4.adapter_config_file_name });
    defer allocator.free(adapter_config_path);

    try writeHeaderAndTensorsF32(allocator, adapter_checkpoint_path, tensors);
    try writeAdapterConfigJson(
        allocator,
        adapter_config_path,
        spec.base_model_name_or_path,
        spec.base_model_sha256,
        spec.tokenizer_sha256,
        spec.chat_template_sha256,
        spec.rank,
        spec.alpha,
        spec.target_modules,
        spec.recursive_config,
    );
    try copySupportingArtifactIfPresent(allocator, spec.tokenizer_config_path, publication.staging_dir, gemma4.tokenizer_config_file_name);
    try copySupportingArtifactIfPresent(allocator, spec.tokenizer_path, publication.staging_dir, gemma4.tokenizer_file_name);
    try copySupportingArtifactIfPresent(allocator, spec.special_tokens_map_path, publication.staging_dir, gemma4.special_tokens_map_file_name);

    try gemma4.validateLoRAAdapterInventory(allocator, publication.staging_dir);

    publication.publish() catch |err| switch (err) {
        error.Gemma4RunOutputAlreadyExists => return error.GemmaBundleOutputAlreadyExists,
        else => return err,
    };
}

fn mapTrainerSlotNameToGemmaAdapterTensor(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    if (try mapUseSiteTrainerSlotNameToLoopTensor(allocator, name)) |mapped| return mapped;
    if (std.mem.endsWith(u8, name, ".lora_A")) {
        return std.fmt.allocPrint(allocator, "{s}.weight", .{name});
    }
    if (std.mem.endsWith(u8, name, ".lora_B")) {
        return std.fmt.allocPrint(allocator, "{s}.weight", .{name});
    }
    return try allocator.dupe(u8, name);
}

fn mapUseSiteTrainerSlotNameToLoopTensor(allocator: std.mem.Allocator, name: []const u8) !?[]u8 {
    const suffix_a = ".lora_A";
    const suffix_b = ".lora_B";
    const kind: u8, const without_suffix = blk: {
        if (std.mem.endsWith(u8, name, suffix_a)) break :blk .{ 'A', name[0 .. name.len - suffix_a.len] };
        if (std.mem.endsWith(u8, name, suffix_b)) break :blk .{ 'B', name[0 .. name.len - suffix_b.len] };
        return null;
    };
    const marker = ".use_";
    const marker_pos = std.mem.lastIndexOf(u8, without_suffix, marker) orelse return null;
    const digits = without_suffix[marker_pos + marker.len ..];
    if (digits.len == 0) return null;
    _ = std.fmt.parseUnsigned(usize, digits, 10) catch return null;
    return try std.fmt.allocPrint(
        allocator,
        "{s}.loop_{s}.lora_{c}.weight",
        .{ without_suffix[0..marker_pos], digits, kind },
    );
}

fn dimsToUsize(allocator: std.mem.Allocator, dims: []const i32) ![]usize {
    const out = try allocator.alloc(usize, dims.len);
    for (dims, 0..) |dim, i| out[i] = @intCast(dim);
    return out;
}

fn writeHeaderAndTensorsF32(allocator: std.mem.Allocator, path: []const u8, tensors: []const WriteTensorF32) !void {
    _ = allocator;
    var header_buf: std.Io.Writer.Allocating = .init(std.heap.page_allocator);
    defer header_buf.deinit();
    const writer = &header_buf.writer;
    try writer.writeByte('{');
    var offset: u64 = 0;
    for (tensors, 0..) |tensor, idx| {
        if (idx != 0) try writer.writeByte(',');
        const byte_len = tensor.data.len * @sizeOf(f32);
        try writer.print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{tensor.name});
        for (tensor.shape, 0..) |dim, dim_idx| {
            if (dim_idx != 0) try writer.writeByte(',');
            try writer.print("{}", .{dim});
        }
        try writer.print("],\"data_offsets\":[{},{}]}}", .{ offset, offset + byte_len });
        offset += byte_len;
    }
    try writer.writeByte('}');

    var file = try compat.cwd().createFile(compat.io(), path, .{ .truncate = true });
    defer file.close(compat.io());
    var len_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &len_buf, header_buf.written().len, .little);
    try file.writeStreamingAll(compat.io(), &len_buf);
    try file.writeStreamingAll(compat.io(), header_buf.written());
    for (tensors) |tensor| {
        for (tensor.data) |item| {
            const bits: u32 = @bitCast(item);
            var bits_buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &bits_buf, bits, .little);
            try file.writeStreamingAll(compat.io(), &bits_buf);
        }
    }
}

fn writeAdapterConfigJson(
    allocator: std.mem.Allocator,
    path: []const u8,
    base_model_name_or_path: []const u8,
    base_model_sha256: ?[]const u8,
    tokenizer_sha256: ?[]const u8,
    chat_template_sha256: ?[]const u8,
    rank: usize,
    alpha: f32,
    target_modules: []const []const u8,
    recursive_config: @import("recursive_lora.zig").Config,
) !void {
    var buffer: std.Io.Writer.Allocating = .init(allocator);
    defer buffer.deinit();
    if (recursive_config.enabled) {
        try std.json.Stringify.value(.{
            .base_model_name_or_path = base_model_name_or_path,
            .antfly_base_model_sha256 = base_model_sha256,
            .antfly_tokenizer_sha256 = tokenizer_sha256,
            .antfly_chat_template_sha256 = chat_template_sha256,
            .peft_type = "LORA",
            .task_type = "CAUSAL_LM",
            .r = rank,
            .lora_alpha = alpha,
            .target_modules = target_modules,
            .recursive_lora = .{
                .enabled = true,
                .source_num_layers = recursive_config.source_num_layers,
                .shared_block_size = recursive_config.shared_block_size,
                .loop_count = recursive_config.loop_count,
                .init_strategy = recursive_config.init_strategy,
            },
        }, .{ .whitespace = .indent_2 }, &buffer.writer);
    } else {
        try std.json.Stringify.value(.{
            .base_model_name_or_path = base_model_name_or_path,
            .antfly_base_model_sha256 = base_model_sha256,
            .antfly_tokenizer_sha256 = tokenizer_sha256,
            .antfly_chat_template_sha256 = chat_template_sha256,
            .peft_type = "LORA",
            .task_type = "CAUSAL_LM",
            .r = rank,
            .lora_alpha = alpha,
            .target_modules = target_modules,
        }, .{ .whitespace = .indent_2 }, &buffer.writer);
    }
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = path, .data = buffer.written() });
}

fn copySupportingArtifactIfPresent(
    allocator: std.mem.Allocator,
    maybe_src_path: ?[]const u8,
    out_dir: []const u8,
    file_name: []const u8,
) !void {
    const src_path = maybe_src_path orelse return;
    const contents = try c_file.readFile(allocator, src_path);
    defer allocator.free(contents);
    const dst_path = try std.fs.path.join(allocator, &.{ out_dir, file_name });
    defer allocator.free(dst_path);
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = dst_path, .data = contents });
}

test "gemma4 bundle publication preserves immutable output and publishes fresh directory" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    const out_dir = try std.fs.path.join(allocator, &.{ root, "trained" });
    defer allocator.free(out_dir);
    try compat.cwd().createDirPath(compat.io(), out_dir);

    const checkpoint_path = try std.fs.path.join(allocator, &.{ out_dir, gemma4.adapter_checkpoint_file_name });
    defer allocator.free(checkpoint_path);
    const config_path = try std.fs.path.join(allocator, &.{ out_dir, gemma4.adapter_config_file_name });
    defer allocator.free(config_path);
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = checkpoint_path, .data = "known-good-checkpoint" });
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = config_path, .data = "known-good-config" });

    const missing_support_path = try std.fs.path.join(allocator, &.{ root, "missing-tokenizer-config.json" });
    defer allocator.free(missing_support_path);
    const a_data = [_]f32{ 1.0, 2.0 };
    const b_data = [_]f32{ 0.0, 0.0 };
    const tensors = [_]WriteTensorF32{
        .{
            .name = "model.layers.0.self_attn.q_proj.weight.lora_A.weight",
            .shape = &.{ 1, 2 },
            .data = &a_data,
        },
        .{
            .name = "model.layers.0.self_attn.q_proj.weight.lora_B.weight",
            .shape = &.{ 2, 1 },
            .data = &b_data,
        },
    };
    const base_spec = GemmaBundleWriteSpec{
        .base_model_name_or_path = "base-model",
        .rank = 1,
        .alpha = 1,
        .target_modules = &.{"model.layers.0.self_attn.q_proj"},
    };
    var failed = false;
    writeAndPublishGemmaBundle(allocator, out_dir, &tensors, .{
        .base_model_name_or_path = base_spec.base_model_name_or_path,
        .rank = base_spec.rank,
        .alpha = base_spec.alpha,
        .target_modules = base_spec.target_modules,
        .tokenizer_config_path = missing_support_path,
    }) catch {
        failed = true;
    };
    try std.testing.expect(failed);

    const preserved_checkpoint = try c_file.readFile(allocator, checkpoint_path);
    defer allocator.free(preserved_checkpoint);
    try std.testing.expectEqualStrings("known-good-checkpoint", preserved_checkpoint);
    const preserved_config = try c_file.readFile(allocator, config_path);
    defer allocator.free(preserved_config);
    try std.testing.expectEqualStrings("known-good-config", preserved_config);

    try std.testing.expectError(
        error.GemmaBundleOutputAlreadyExists,
        writeAndPublishGemmaBundle(allocator, out_dir, &tensors, base_spec),
    );
    const still_preserved_checkpoint = try c_file.readFile(allocator, checkpoint_path);
    defer allocator.free(still_preserved_checkpoint);
    try std.testing.expectEqualStrings("known-good-checkpoint", still_preserved_checkpoint);
    const still_preserved_config = try c_file.readFile(allocator, config_path);
    defer allocator.free(still_preserved_config);
    try std.testing.expectEqualStrings("known-good-config", still_preserved_config);

    const fresh_out_dir = try std.fs.path.join(allocator, &.{ root, "trained-fresh" });
    defer allocator.free(fresh_out_dir);
    try writeAndPublishGemmaBundle(allocator, fresh_out_dir, &tensors, base_spec);
    var inspection = try gemma4.inspectCheckpoint(allocator, fresh_out_dir);
    defer gemma4.freeInspectionSummary(allocator, &inspection);
    try std.testing.expect(inspection.has_adapter_weights);
    try std.testing.expectEqual(@as(?usize, 1), inspection.lora_rank);
    try std.testing.expectEqualStrings("base-model", inspection.base_model_name_or_path.?);
}

test "gemma4 makeTrainerInputForExample builds bounded sparse causal targets" {
    const allocator = std.testing.allocator;
    var ctx = GemmaAutodiffCtx.init(.{
        .family = .gemma,
        .hidden_size = 16,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .attention_head_dim = 4,
        .intermediate_size = 32,
        .vocab_size = 32,
        .position_encoding = .rope,
        .norm_type = .rms_norm,
        .activation = .gelu_new,
        .norm_eps = 1e-6,
        .norm_weight_offset = 1.0,
        .final_logit_softcapping = 30.0,
    });
    var prompt_input_ids = [_]i32{ 1, 2 };
    var response_input_ids = [_]i32{ 3, 4 };
    var input_ids = [_]i32{ 1, 2, 3, 4 };
    var labels = [_]i32{ -100, -100, 3, 4 };
    const ex = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = prompt_input_ids[0..],
        .response_input_ids = response_input_ids[0..],
        .num_prompt_tokens = 2,
        .num_response_tokens = 2,
        .input_ids = input_ids[0..],
        .labels = labels[0..],
        .num_input_tokens = 4,
        .num_supervised_tokens = 2,
    };

    var owned = try makeTrainerInputForExample(allocator, &ctx, &ex, 4);
    defer owned.deinit(allocator);

    try std.testing.expectEqual(@as(i64, 1), owned.input_ids[0]);
    try std.testing.expectEqual(@as(usize, 4), owned.targets.len);
    try std.testing.expectEqual(@as(u8, 2), owned.trainer_input.targets_shape.rank());
    try std.testing.expectEqual(@as(i64, 2), owned.trainer_input.targets_shape.dim(0));
    try std.testing.expectEqual(sparse_target_columns, owned.trainer_input.targets_shape.dim(1));
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 3.0, 2.0, 4.0 }, owned.targets);
}

test "gemma4 prepared examples reject malformed labels and count drift" {
    const allocator = std.testing.allocator;
    var ctx = GemmaAutodiffCtx.init(.{
        .family = .gemma,
        .hidden_size = 16,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .attention_head_dim = 4,
        .intermediate_size = 32,
        .vocab_size = 32,
        .position_encoding = .rope,
        .norm_type = .rms_norm,
        .activation = .gelu_new,
        .norm_eps = 1e-6,
        .norm_weight_offset = 1.0,
    });
    var input_ids = [_]i32{ 1, 2, 3, 4 };
    var labels = [_]i32{ -100, -100, 3, 4 };
    var ex = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = input_ids[0..2],
        .response_input_ids = input_ids[2..4],
        .num_prompt_tokens = 2,
        .num_response_tokens = 2,
        .input_ids = input_ids[0..],
        .labels = labels[0..],
        .num_input_tokens = 4,
        .num_supervised_tokens = 2,
    };

    labels[2] = -1;
    try std.testing.expectError(error.InvalidPreparedLabel, makeTrainerInputForExample(allocator, &ctx, &ex, 4));
    labels[2] = 32;
    try std.testing.expectError(error.LabelOutOfRange, makeTrainerInputForExample(allocator, &ctx, &ex, 4));
    labels[2] = 3;
    ex.num_supervised_tokens = 1;
    try std.testing.expectError(error.SupervisedTokenCountMismatch, makeTrainerInputForExample(allocator, &ctx, &ex, 4));
}

test "gemma4 sparse causal targets do not scale with vocabulary size" {
    const allocator = std.testing.allocator;
    var ctx = GemmaAutodiffCtx.init(.{
        .family = .gemma,
        .hidden_size = 16,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .attention_head_dim = 4,
        .intermediate_size = 32,
        .vocab_size = 262_144,
        .position_encoding = .rope,
        .norm_type = .rms_norm,
        .activation = .gelu_new,
        .norm_eps = 1e-6,
        .norm_weight_offset = 1.0,
    });
    var prompt_input_ids = [_]i32{ 1, 2 };
    var response_input_ids = [_]i32{ 3, 4 };
    var input_ids = [_]i32{ 1, 2, 3, 4 };
    var labels = [_]i32{ -100, -100, 3, 4 };
    const ex = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = prompt_input_ids[0..],
        .response_input_ids = response_input_ids[0..],
        .num_prompt_tokens = 2,
        .num_response_tokens = 2,
        .input_ids = input_ids[0..],
        .labels = labels[0..],
        .num_input_tokens = 4,
        .num_supervised_tokens = 2,
    };

    var owned = try makeTrainerInputForExample(allocator, &ctx, &ex, 4);
    defer owned.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 4), owned.targets.len);
}

test "gemma4 sparse causal loss graph projects only supervised rows" {
    const allocator = std.testing.allocator;
    var ctx = GemmaAutodiffCtx.init(.{
        .family = .gemma,
        .hidden_size = 16,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .attention_head_dim = 4,
        .intermediate_size = 32,
        .vocab_size = 32,
        .position_encoding = .rope,
        .norm_type = .rms_norm,
        .activation = .gelu_new,
        .norm_eps = 1e-6,
        .norm_weight_offset = 1.0,
        .final_logit_softcapping = 30.0,
    });
    var graph = Graph.init(allocator);
    defer graph.deinit();
    var bld = Builder.init(&graph);
    const hidden = try bld.parameter("hidden", Shape.init(.f32, &.{ 1, 4, 16 }));
    const targets = try bld.parameter("targets", Shape.init(.f32, &.{ 2, sparse_target_columns }));

    const loss = try GemmaAutodiffCtx.buildLoss(@ptrCast(&ctx), &bld, hidden, targets);
    try std.testing.expectEqual(@as(i64, 1), graph.node(loss).output_shape.numElements().?);
    var saw_tanh = false;
    for (0..graph.nodeCount()) |idx| {
        switch (graph.node(@intCast(idx)).op) {
            .tanh => saw_tanh = true,
            else => {},
        }
    }
    try std.testing.expect(saw_tanh);
}

test "gemma4 sparse teacher loss never constructs a sequence by vocabulary target" {
    const allocator = std.testing.allocator;
    var ctx = GemmaAutodiffCtx.init(.{
        .family = .gemma,
        .hidden_size = 16,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .attention_head_dim = 4,
        .intermediate_size = 32,
        .vocab_size = 32,
        .position_encoding = .rope,
        .norm_type = .rms_norm,
        .activation = .gelu_new,
        .norm_eps = 1e-6,
        .norm_weight_offset = 1.0,
    });
    var graph = Graph.init(allocator);
    defer graph.deinit();
    var bld = Builder.init(&graph);
    const hidden = try bld.parameter("hidden", Shape.init(.f32, &.{ 1, 4, 16 }));
    const targets = try bld.parameter("targets", Shape.init(.f32, &.{ 2, 6 }));

    const loss = try GemmaAutodiffCtx.buildLoss(@ptrCast(&ctx), &bld, hidden, targets);
    try std.testing.expectEqual(@as(i64, 1), graph.node(loss).output_shape.numElements().?);
    var saw_temperature_divide = false;
    for (0..graph.nodeCount()) |idx| {
        const node = graph.node(@intCast(idx));
        const shape = node.output_shape;
        try std.testing.expect(!(shape.rank() == 2 and shape.dim(0) == 4 and shape.dim(1) == 32));
        switch (node.op) {
            .div => saw_temperature_divide = true,
            else => {},
        }
    }
    try std.testing.expect(saw_temperature_divide);
}

test "gemma4 loadGraphConfig supports a GGUF manifest directory" {
    const raw_path = std.c.getenv("ANTFLY_GEMMA4_FINETUNE_TEST_MODEL") orelse
        return error.SkipZigTest;
    const config = try loadGraphConfig(std.testing.allocator, std.mem.span(raw_path));
    try std.testing.expect(config.family == .gemma);
    try std.testing.expect(config.hidden_size > 0);
    try std.testing.expect(config.num_hidden_layers > 0);
    try std.testing.expect(config.vocab_size > 0);
}

test "gemma4 makeTrainerInputForTokenLogprobGrads builds per-token weighted targets" {
    const allocator = std.testing.allocator;
    var ctx = GemmaAutodiffCtx.init(.{
        .family = .gemma,
        .hidden_size = 16,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .attention_head_dim = 4,
        .intermediate_size = 32,
        .vocab_size = 32,
        .position_encoding = .rope,
        .norm_type = .rms_norm,
        .activation = .gelu_new,
        .norm_eps = 1e-6,
        .norm_weight_offset = 1.0,
    });
    var prompt_input_ids = [_]i32{ 1, 2 };
    var response_input_ids = [_]i32{ 3, 4 };
    var input_ids = [_]i32{ 1, 2, 3, 4 };
    var labels = [_]i32{ -100, -100, 3, 4 };
    const ex = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = prompt_input_ids[0..],
        .response_input_ids = response_input_ids[0..],
        .num_prompt_tokens = 2,
        .num_response_tokens = 2,
        .input_ids = input_ids[0..],
        .labels = labels[0..],
        .num_input_tokens = 4,
        .num_supervised_tokens = 2,
    };

    var owned = try makeTrainerInputForTokenLogprobGrads(allocator, &ctx, &ex, 4, &.{ 0.25, -0.5 });
    defer owned.deinit(allocator);

    try std.testing.expectApproxEqAbs(@as(f32, -1.0), owned.targets[1 * 32 + 3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), owned.targets[2 * 32 + 4], 1e-6);
}

test "gemma4 teacher soft targets are sparse and aligned to causal predictor rows" {
    const allocator = std.testing.allocator;
    var ctx = GemmaAutodiffCtx.init(.{
        .family = .gemma,
        .hidden_size = 16,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .attention_head_dim = 4,
        .intermediate_size = 32,
        .vocab_size = 262_144,
        .position_encoding = .rope,
        .norm_type = .rms_norm,
        .activation = .gelu_new,
        .norm_eps = 1e-6,
        .norm_weight_offset = 1.0,
    });
    var prompt_input_ids = [_]i32{ 1, 2 };
    var response_input_ids = [_]i32{ 3, 4 };
    var input_ids = [_]i32{ 1, 2, 3, 4 };
    var labels = [_]i32{ -100, -100, 3, 4 };
    var teacher_ids = [_]i32{
        0, 0,
        7, 8,
        9, 10,
        0, 0,
    };
    var teacher_probs = [_]f32{
        0.0,  0.0,
        0.75, 0.25,
        0.2,  0.8,
        0.0,  0.0,
    };
    const ex = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = prompt_input_ids[0..],
        .response_input_ids = response_input_ids[0..],
        .num_prompt_tokens = 2,
        .num_response_tokens = 2,
        .input_ids = input_ids[0..],
        .labels = labels[0..],
        .num_input_tokens = 4,
        .num_supervised_tokens = 2,
        .teacher_top_k_token_ids = teacher_ids[0..],
        .teacher_top_k_probs = teacher_probs[0..],
        .teacher_top_k = 2,
        .teacher_temperature = 1.0,
    };

    var owned = try makeTrainerInputForExample(allocator, &ctx, &ex, 4);
    defer owned.deinit(allocator);

    try std.testing.expectEqual(@as(i64, 2), owned.trainer_input.targets_shape.dim(0));
    try std.testing.expectEqual(@as(i64, 6), owned.trainer_input.targets_shape.dim(1));
    try std.testing.expectEqualSlices(f32, &.{
        1.0, 1.0, 7.0, 8.0,  0.75, 0.25,
        2.0, 1.0, 9.0, 10.0, 0.2,  0.8,
    }, owned.targets);
}

test "gemma4 dense and sparse teacher targets reject infinite values" {
    var prompt_input_ids = [_]i32{1};
    var response_input_ids = [_]i32{2};
    var input_ids = [_]i32{ 1, 2 };
    var labels = [_]i32{ -100, 2 };
    var teacher_ids = [_]i32{ 3, 0 };
    var teacher_probs = [_]f32{ 0.5, 0.0 };
    var example = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = prompt_input_ids[0..],
        .response_input_ids = response_input_ids[0..],
        .num_prompt_tokens = 1,
        .num_response_tokens = 1,
        .input_ids = input_ids[0..],
        .labels = labels[0..],
        .num_input_tokens = 2,
        .num_supervised_tokens = 1,
        .teacher_top_k_token_ids = teacher_ids[0..],
        .teacher_top_k_probs = teacher_probs[0..],
        .teacher_top_k = 1,
        .teacher_temperature = std.math.inf(f32),
    };
    var dense_targets = [_]f32{0} ** 16;
    var sparse_targets = [_]f32{0} ** 4;

    try std.testing.expectError(
        error.InvalidTeacherTemperature,
        fillTeacherTopKTargets(&dense_targets, 2, 8, &example),
    );
    try std.testing.expectError(
        error.InvalidTeacherTemperature,
        fillSparseTeacherTopKTargets(&sparse_targets, 4, 2, 8, &example),
    );

    example.teacher_temperature = 1.0;
    teacher_probs[0] = std.math.inf(f32);
    try std.testing.expectError(
        error.InvalidTeacherDistillationTargets,
        fillTeacherTopKTargets(&dense_targets, 2, 8, &example),
    );
    try std.testing.expectError(
        error.InvalidTeacherDistillationTargets,
        fillSparseTeacherTopKTargets(&sparse_targets, 4, 2, 8, &example),
    );
}

test "gemma4 makeTrainerInputForTokenLogprobGrads aligns completion weights with predictor rows" {
    const allocator = std.testing.allocator;
    var ctx = GemmaAutodiffCtx.init(.{
        .family = .gemma,
        .hidden_size = 16,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .attention_head_dim = 4,
        .intermediate_size = 32,
        .vocab_size = 32,
        .position_encoding = .rope,
        .norm_type = .rms_norm,
        .activation = .gelu_new,
        .norm_eps = 1e-6,
        .norm_weight_offset = 1.0,
    });
    var prompt_input_ids = [_]i32{ 1, 2, 3 };
    var response_input_ids = [_]i32{ 5, 7, 9 };
    var input_ids = [_]i32{ 1, 2, 3, 5, 7, 9 };
    var labels = [_]i32{ -100, -100, -100, 5, 7, 9 };
    const ex = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = prompt_input_ids[0..],
        .response_input_ids = response_input_ids[0..],
        .num_prompt_tokens = 3,
        .num_response_tokens = 3,
        .input_ids = input_ids[0..],
        .labels = labels[0..],
        .num_input_tokens = 6,
        .num_supervised_tokens = 3,
    };

    var owned = try makeTrainerInputForTokenLogprobGrads(allocator, &ctx, &ex, 6, &.{ 0.25, -0.5, 1.5 });
    defer owned.deinit(allocator);

    for (0..2) |row_idx| {
        const row = owned.targets[row_idx * 32 ..][0..32];
        for (row) |value| try std.testing.expectEqual(@as(f32, 0.0), value);
    }
    try std.testing.expectApproxEqAbs(@as(f32, -1.5), owned.targets[2 * 32 + 5], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), owned.targets[3 * 32 + 7], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -9.0), owned.targets[4 * 32 + 9], 1e-6);
    const final_row = owned.targets[5 * 32 ..][0..32];
    for (final_row) |value| try std.testing.expectEqual(@as(f32, 0.0), value);
}

test "gemma4 teacher targets carry temperature and standard distillation scale" {
    const allocator = std.testing.allocator;
    var ctx = GemmaAutodiffCtx.init(.{
        .family = .gemma,
        .hidden_size = 16,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .attention_head_dim = 4,
        .intermediate_size = 32,
        .vocab_size = 32,
        .position_encoding = .rope,
        .norm_type = .rms_norm,
        .activation = .gelu_new,
        .norm_eps = 1e-6,
        .norm_weight_offset = 1.0,
    });
    var prompt_input_ids = [_]i32{1};
    var response_input_ids = [_]i32{2};
    var input_ids = [_]i32{ 1, 2 };
    var labels = [_]i32{ -100, 2 };
    var teacher_ids = [_]i32{
        5, 6,
        0, 0,
    };
    var teacher_probs = [_]f32{
        0.25, 0.75,
        0.0,  0.0,
    };
    const ex = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = prompt_input_ids[0..],
        .response_input_ids = response_input_ids[0..],
        .num_prompt_tokens = 1,
        .num_response_tokens = 1,
        .input_ids = input_ids[0..],
        .labels = labels[0..],
        .num_input_tokens = 2,
        .num_supervised_tokens = 1,
        .teacher_top_k_token_ids = teacher_ids[0..],
        .teacher_top_k_probs = teacher_probs[0..],
        .teacher_top_k = 2,
        .teacher_temperature = 2.0,
    };

    var owned = try makeTrainerInputForExample(allocator, &ctx, &ex, 2);
    defer owned.deinit(allocator);

    try std.testing.expectEqualSlices(f32, &.{ 0.0, 2.0, 5.0, 6.0, 1.0, 3.0 }, owned.targets);

    // For T=2 the expected student-logit gradient is
    // T * (softmax(student_logits / T) - teacher_probs). Verify the formula
    // differs from the old T^2-scaled, untempered gradient, including direction.
    const student_logits = [_]f32{ 1.0, -1.0 };
    const teacher_distribution = [_]f32{ 0.8, 0.2 };
    const temperature: f32 = 2.0;
    const tempered_p0 = @exp(student_logits[0] / temperature) /
        (@exp(student_logits[0] / temperature) + @exp(student_logits[1] / temperature));
    const correct_gradient0 = temperature * (tempered_p0 - teacher_distribution[0]);
    const untempered_p0 = @exp(student_logits[0]) / (@exp(student_logits[0]) + @exp(student_logits[1]));
    const old_gradient0 = temperature * temperature * (untempered_p0 - teacher_distribution[0]);
    try std.testing.expect(correct_gradient0 < 0.0);
    try std.testing.expect(old_gradient0 > 0.0);
}
