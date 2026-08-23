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
const gpt_arch = @import("../architectures/gpt.zig");
const generation = @import("../pipelines/generation.zig");
const runtime = @import("../runtime/root.zig");
const real_autodiff = @import("real_autodiff_trainer.zig");
const gemma4 = @import("gemma4.zig");
const artifact_publication = @import("artifact_publication.zig");
const compat = @import("../io/compat.zig");
const graph_input_binder = @import("graph_input_binder.zig");
const weight_source_mod = @import("../models/weight_source.zig");
const metal_compute_mod = @import("../ops/metal_compute.zig");
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
    /// Non-padding tokens carried by examples that reached train/eval.
    logical_input_tokens_seen: usize = 0,
    /// Transformer rows actually scheduled after optional length bucketing.
    scheduled_input_tokens_seen: usize = 0,
    /// Rows the caller's fixed-shape policy would have scheduled.
    fixed_shape_input_tokens_seen: usize = 0,
    padding_tokens_avoided: usize = 0,
    length_bucketed_examples: usize = 0,
    min_scheduled_sequence_length: u32 = 0,
    max_scheduled_sequence_length: u32 = 0,
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
    runtime_input_uploads: u64 = 0,
    runtime_input_upload_bytes: u64 = 0,
    runtime_input_h2d_bytes: u64 = 0,
    runtime_input_d2h_bytes: u64 = 0,
    declared_runtime_input_uploads: u64 = 0,
    declared_runtime_input_upload_bytes: u64 = 0,
    declared_runtime_input_h2d_bytes: u64 = 0,
    graph_execution_h2d_bytes: u64 = 0,
    graph_execution_d2h_bytes: u64 = 0,
    training_runtime_h2d_bytes: u64 = 0,
    training_runtime_d2h_bytes: u64 = 0,
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
    metrics.runtime_input_uploads += profile.runtime_input_uploads;
    metrics.runtime_input_upload_bytes += profile.runtime_input_upload_bytes;
    metrics.runtime_input_h2d_bytes += profile.runtime_input_h2d_bytes;
    metrics.runtime_input_d2h_bytes += profile.runtime_input_d2h_bytes;
    metrics.declared_runtime_input_uploads += profile.declared_runtime_input_uploads;
    metrics.declared_runtime_input_upload_bytes += profile.declared_runtime_input_upload_bytes;
    metrics.declared_runtime_input_h2d_bytes += profile.declared_runtime_input_h2d_bytes;
    metrics.graph_execution_h2d_bytes += profile.graph_execution_h2d_bytes;
    metrics.graph_execution_d2h_bytes += profile.graph_execution_d2h_bytes;
    metrics.training_runtime_h2d_bytes += profile.training_runtime_h2d_bytes;
    metrics.training_runtime_d2h_bytes += profile.training_runtime_d2h_bytes;
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

pub const DpoPairGraphMode = enum(u8) {
    split_batch1,
    batched_forward,
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
    /// Set by the Metal training command after strict backend admission. Keep
    /// false for native/reference callers so their decomposed VJP remains an
    /// independent correctness control.
    enable_fused_rms_norm_backward: bool = false,
    /// Experimental differentiable GQA forward/backward. The command layer
    /// enables this only for admitted Metal training when explicitly opted in.
    enable_fused_gqa_attention_backward: bool = false,
    /// Backend admission override. Strict Metal training enables the qualified
    /// bounded CCE primitive; native/reference callers remain null so their
    /// decomposed loss stays an independent correctness control.
    enable_fused_linear_cross_entropy: ?bool = null,
    built: ?gemma_graph.GemmaGraph = null,
    lm_logits: ?NodeId = null,
    /// Non-null only for the compiled DPO pair graph. The graph runs chosen
    /// and rejected as two independent batch rows, reduces each response to a
    /// sequence log-probability, and differentiates the DPO scalar directly.
    /// Keeping the bucket in the captured graph context makes shape-cache
    /// restores deterministic when ordinary SFT/scoring graphs coexist.
    dpo_pair_bucket_rows: ?u32 = null,

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
        bld.fuse_rms_norm_backward = self.enable_fused_rms_norm_backward;
        bld.fuse_gqa_attention_backward = self.enable_fused_gqa_attention_backward;
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
        if (self.dpo_pair_bucket_rows) |bucket_rows| {
            return self.buildDpoPairLoss(bld, forward_output, targets, bucket_rows);
        }
        const target_shape = bld.graph.node(targets).output_shape;
        if (target_shape.rank() == 2 and target_shape.dim(1) != self.graph_config.vocab_size) {
            return self.buildSparseCausalLoss(bld, forward_output, targets);
        }
        const logits = try self.buildLogits(bld, forward_output);
        return bld.crossEntropyLoss(logits, targets);
    }

    /// Whole-objective DPO builder used by RealAutodiffTrainer's coupled
    /// objective hook. Each preference side is a genuine batch-1 Gemma graph,
    /// preserving the qualified single-example numerics. The chosen branch is
    /// reduced to a scalar before the rejected branch is constructed, which
    /// gives checkpointing and the buffer planner a branch boundary at which
    /// large forward activations can be released or recomputed.
    pub fn buildDpoPairObjective(
        ctx_opaque: *anyopaque,
        bld: *Builder,
        input_ids: NodeId,
        attention_mask: NodeId,
        targets: NodeId,
        batch: u32,
        seq_len: u32,
    ) anyerror!real_autodiff.ObjectiveGraph {
        const self: *GemmaAutodiffCtx = @ptrCast(@alignCast(ctx_opaque));
        const bucket_rows = self.dpo_pair_bucket_rows orelse return error.MissingDpoPairGraphMode;
        if (batch != 2) return error.InvalidDpoPairGraphShape;
        const target_shape = bld.graph.node(targets).output_shape;
        const expected_rows = @as(i64, @intCast(bucket_rows)) * 2 + 1;
        if (target_shape.rank() != 2 or
            target_shape.dim(0) != expected_rows or
            target_shape.dim(1) != @as(i64, dpo_pair_target_columns))
        {
            return error.InvalidDpoPairTargetShape;
        }

        bld.fuse_rms_norm_backward = self.enable_fused_rms_norm_backward;
        bld.fuse_gqa_attention_backward = self.enable_fused_gqa_attention_backward;
        self.lm_logits = null;
        const chosen_ids = try sliceRows2d(bld, input_ids, 0, 1, seq_len);
        const chosen_mask = try sliceRows2d(bld, attention_mask, 0, 1, seq_len);
        const rejected_ids = try sliceRows2d(bld, input_ids, 1, 2, seq_len);
        const rejected_mask = try sliceRows2d(bld, attention_mask, 1, 2, seq_len);
        const chosen_targets = try sliceRows2d(bld, targets, 0, bucket_rows, dpo_pair_target_columns);
        const rejected_targets = try sliceRows2d(bld, targets, bucket_rows, bucket_rows * 2, dpo_pair_target_columns);
        const metadata = try sliceRows2d(bld, targets, bucket_rows * 2, bucket_rows * 2 + 1, dpo_pair_target_columns);

        const chosen_graph = try gemma_graph.buildForwardGraphWithOptions(bld, self.graph_config, 1, seq_len, .{
            .input_ids = chosen_ids,
            .rope_cos = ml.graph.null_node,
            .rope_sin = ml.graph.null_node,
        }, self.graph_options);
        _ = chosen_mask;
        const chosen_count = try bld.reshape(try bld.sliceLastDim(metadata, 2, 3), Shape.scalar(.f32));
        const chosen_logp = try self.buildDpoBranchLogp(bld, chosen_graph.output_node, chosen_targets, bucket_rows, chosen_count);

        const rejected_graph = try gemma_graph.buildForwardGraphWithOptions(bld, self.graph_config, 1, seq_len, .{
            .input_ids = rejected_ids,
            .rope_cos = ml.graph.null_node,
            .rope_sin = ml.graph.null_node,
        }, self.graph_options);
        _ = rejected_mask;
        const rejected_count = try bld.reshape(try bld.sliceLastDim(metadata, 3, 4), Shape.scalar(.f32));
        const rejected_logp = try self.buildDpoBranchLogp(bld, rejected_graph.output_node, rejected_targets, bucket_rows, rejected_count);

        self.built = rejected_graph;
        const ref_margin = try bld.reshape(try bld.sliceLastDim(metadata, 0, 1), Shape.scalar(.f32));
        const beta = try bld.reshape(try bld.sliceLastDim(metadata, 1, 2), Shape.scalar(.f32));
        const policy_margin = try bld.sub(chosen_logp, rejected_logp);
        const reward_margin = try bld.mul(beta, try bld.sub(policy_margin, ref_margin));
        const loss = try bld.neg(try bld.logOp(try bld.sigmoid(reward_margin)));
        return .{ .forward_output = rejected_graph.output_node, .loss = loss };
    }

    fn buildDpoBranchLogp(
        self: *GemmaAutodiffCtx,
        bld: *Builder,
        forward_output: NodeId,
        branch_targets: NodeId,
        bucket_rows: u32,
        active_count: NodeId,
    ) !NodeId {
        const out_shape = bld.graph.node(forward_output).output_shape;
        if (out_shape.rank() != 3 or out_shape.dim(0) != 1) return error.InvalidDpoPairGraphShape;
        const seq_len: u32 = @intCast(out_shape.dim(1));
        const hidden_size: u32 = @intCast(out_shape.dim(2));
        const hidden_flat = try bld.reshape(
            forward_output,
            Shape.init(.f32, &.{ @as(i64, @intCast(seq_len)), @as(i64, @intCast(hidden_size)) }),
        );
        const rows_2d = try bld.sliceLastDim(branch_targets, 0, 1);
        const rows = try bld.reshape(rows_2d, Shape.init(.f32, &.{@as(i64, @intCast(bucket_rows))}));
        const labels = try bld.sliceLastDim(branch_targets, 1, 2);
        const hidden = try bld.embeddingLookup(hidden_flat, rows, bucket_rows, hidden_size);
        const lm_head_w = try self.buildLmHeadWeight(bld, hidden_size);
        const mean_ce = try bld.linearCrossEntropyLoss(hidden, lm_head_w, labels, .{
            .rows = bucket_rows,
            .in_dim = hidden_size,
            .vocab_size = self.graph_config.vocab_size,
            .logit_softcap = self.graph_config.final_logit_softcapping,
            .ignore_index = -100,
            .frozen_weight = true,
        });
        return bld.neg(try bld.mul(mean_ce, active_count));
    }

    /// Exact batch-one-pair DPO objective:
    ///
    ///   -log(sigmoid(beta * ((logp_c - logp_r) - ref_margin)))
    ///
    /// `forward_output` contains two independent causal streams in its batch
    /// dimension. The target tensor contains two fixed-size sparse response
    /// blocks followed by one metadata row. Fused linear-CE returns a mean
    /// over non-ignored labels; multiplying by each active-token count recovers
    /// the summed sequence log-probability used by the host DPO oracle.
    fn buildDpoPairLoss(
        self: *GemmaAutodiffCtx,
        bld: *Builder,
        forward_output: NodeId,
        targets: NodeId,
        bucket_rows: u32,
    ) !NodeId {
        self.lm_logits = null;
        const out_shape = bld.graph.node(forward_output).output_shape;
        if (out_shape.rank() != 3 or out_shape.dim(0) != 2) return error.InvalidDpoPairGraphShape;
        const seq_len: u32 = @intCast(out_shape.dim(1));
        const hidden_size: u32 = @intCast(out_shape.dim(2));
        const target_shape = bld.graph.node(targets).output_shape;
        const expected_rows = @as(i64, @intCast(bucket_rows)) * 2 + 1;
        if (target_shape.rank() != 2 or
            target_shape.dim(0) != expected_rows or
            target_shape.dim(1) != @as(i64, dpo_pair_target_columns))
        {
            return error.InvalidDpoPairTargetShape;
        }

        const hidden_flat = try bld.reshape(
            forward_output,
            Shape.init(.f32, &.{ @as(i64, @intCast(seq_len * 2)), @as(i64, @intCast(hidden_size)) }),
        );
        const chosen_targets = try sliceRows2d(bld, targets, 0, bucket_rows, dpo_pair_target_columns);
        const rejected_targets = try sliceRows2d(bld, targets, bucket_rows, bucket_rows * 2, dpo_pair_target_columns);
        const metadata = try sliceRows2d(bld, targets, bucket_rows * 2, bucket_rows * 2 + 1, dpo_pair_target_columns);

        const chosen_rows_2d = try bld.sliceLastDim(chosen_targets, 0, 1);
        const rejected_rows_2d = try bld.sliceLastDim(rejected_targets, 0, 1);
        const chosen_rows = try bld.reshape(chosen_rows_2d, Shape.init(.f32, &.{@as(i64, @intCast(bucket_rows))}));
        const rejected_rows = try bld.reshape(rejected_rows_2d, Shape.init(.f32, &.{@as(i64, @intCast(bucket_rows))}));
        const chosen_labels = try bld.sliceLastDim(chosen_targets, 1, 2);
        const rejected_labels = try bld.sliceLastDim(rejected_targets, 1, 2);
        const chosen_hidden = try bld.embeddingLookup(hidden_flat, chosen_rows, bucket_rows, hidden_size);
        const rejected_hidden = try bld.embeddingLookup(hidden_flat, rejected_rows, bucket_rows, hidden_size);
        const lm_head_w = try self.buildLmHeadWeight(bld, hidden_size);

        const chosen_mean_ce = try bld.linearCrossEntropyLoss(chosen_hidden, lm_head_w, chosen_labels, .{
            .rows = bucket_rows,
            .in_dim = hidden_size,
            .vocab_size = self.graph_config.vocab_size,
            .logit_softcap = self.graph_config.final_logit_softcapping,
            .ignore_index = -100,
            .frozen_weight = true,
        });
        const rejected_mean_ce = try bld.linearCrossEntropyLoss(rejected_hidden, lm_head_w, rejected_labels, .{
            .rows = bucket_rows,
            .in_dim = hidden_size,
            .vocab_size = self.graph_config.vocab_size,
            .logit_softcap = self.graph_config.final_logit_softcapping,
            .ignore_index = -100,
            .frozen_weight = true,
        });

        const ref_margin = try bld.reshape(try bld.sliceLastDim(metadata, 0, 1), Shape.scalar(.f32));
        const beta = try bld.reshape(try bld.sliceLastDim(metadata, 1, 2), Shape.scalar(.f32));
        const chosen_count = try bld.reshape(try bld.sliceLastDim(metadata, 2, 3), Shape.scalar(.f32));
        const rejected_count = try bld.reshape(try bld.sliceLastDim(metadata, 3, 4), Shape.scalar(.f32));
        const chosen_logp = try bld.neg(try bld.mul(chosen_mean_ce, chosen_count));
        const rejected_logp = try bld.neg(try bld.mul(rejected_mean_ce, rejected_count));
        const policy_margin = try bld.sub(chosen_logp, rejected_logp);
        const reward_margin = try bld.mul(beta, try bld.sub(policy_margin, ref_margin));
        return bld.neg(try bld.logOp(try bld.sigmoid(reward_margin)));
    }

    /// Memory-bounded causal LM loss. Ordinary hard-label `targets` are
    /// [M, 2] `[predictor_row, token_id]` pairs. Signed preference targets are
    /// [M, 4] `[predictor_row, token_id, cross_entropy_weight, reserved]`
    /// rows; zero-weight padding keeps a small set of reusable graph shapes.
    /// The tied vocabulary projection is evaluated a bounded number of
    /// supervised rows at a time so the graph never owns a
    /// [sequence, vocabulary] target tensor.
    fn buildSparseCausalLoss(
        self: *GemmaAutodiffCtx,
        bld: *Builder,
        forward_output: NodeId,
        targets: NodeId,
    ) !NodeId {
        self.lm_logits = null;
        const out_shape = bld.graph.node(forward_output).output_shape;
        const total_rows: u32 = @intCast(out_shape.dim(0) * out_shape.dim(1));
        const hidden_size: u32 = @intCast(out_shape.dim(2));
        const target_shape = bld.graph.node(targets).output_shape;
        const supervised_rows: u32 = @intCast(target_shape.dim(0));
        const target_columns: u32 = @intCast(target_shape.dim(1));
        if (supervised_rows == 0) return error.NoSupervisedTokens;
        if (target_columns < sparse_target_columns) return error.InvalidTeacherDistillationTargets;
        const weighted_hard_targets = target_columns == weighted_hard_target_columns;
        const uniform_weighted_targets = target_columns == uniform_weighted_target_columns;
        const fused_linear_ce_enabled = self.enable_fused_linear_cross_entropy orelse cutLinearCrossEntropyEnabled();
        if (uniform_weighted_targets and !fused_linear_ce_enabled) {
            return error.UniformWeightedTargetsRequireFusedLinearCrossEntropy;
        }
        const teacher_top_k: u32 = if (target_columns == sparse_target_columns or weighted_hard_targets or uniform_weighted_targets)
            0
        else blk: {
            if ((target_columns - 1) % 2 != 0) return error.InvalidTeacherDistillationTargets;
            break :blk (target_columns - 1) / 2;
        };

        const hidden_flat = try bld.reshape(forward_output, Shape.init(.f32, &.{ @as(i64, @intCast(total_rows)), @as(i64, @intCast(hidden_size)) }));
        const predictor_rows_2d = try bld.sliceLastDim(targets, 0, 1);
        const predictor_rows = try bld.reshape(predictor_rows_2d, Shape.init(.f32, &.{@as(i64, @intCast(supervised_rows))}));
        const supervised_hidden = try bld.embeddingLookup(hidden_flat, predictor_rows, supervised_rows, hidden_size);
        const labels = if (teacher_top_k == 0) try bld.sliceLastDim(targets, 1, 2) else null;
        const hard_label_weights = if (weighted_hard_targets) try bld.sliceLastDim(targets, 2, 3) else null;
        const teacher_ids = if (teacher_top_k > 0) try bld.sliceLastDim(targets, 1, 1 + teacher_top_k) else null;
        const teacher_probs = if (teacher_top_k > 0) try bld.sliceLastDim(targets, 1 + teacher_top_k, target_columns) else null;
        const lm_head_w = try self.buildLmHeadWeight(bld, hidden_size);

        const loss_chunk_rows = sparseLossChunkRows();
        if (teacher_top_k == 0 and !weighted_hard_targets and fused_linear_ce_enabled) {
            if (uniform_weighted_targets and supervised_rows > loss_chunk_rows) return error.UniformWeightedLossExceedsFusedChunk;
            // Gemma PEFT keeps the tied vocabulary projection frozen. The
            // graph-native loss computes hard-label CE and d_hidden without
            // owning a [chunk_rows, vocab_size] logits activation. Its builder
            // and VJP both fail closed if a weight gradient is requested.
            var total_loss: ?NodeId = null;
            var start: u32 = 0;
            while (start < supervised_rows) : (start += loss_chunk_rows) {
                const end = @min(start + loss_chunk_rows, supervised_rows);
                const chunk_rows = end - start;
                const hidden_chunk = try sliceRows2d(bld, supervised_hidden, start, end, hidden_size);
                const label_chunk = try sliceRows2d(bld, labels.?, start, end, 1);
                const chunk_loss = try bld.linearCrossEntropyLoss(hidden_chunk, lm_head_w, label_chunk, .{
                    .rows = chunk_rows,
                    .in_dim = hidden_size,
                    .vocab_size = self.graph_config.vocab_size,
                    .logit_softcap = self.graph_config.final_logit_softcapping,
                    .ignore_index = -100,
                    .frozen_weight = true,
                });
                if (uniform_weighted_targets) {
                    // Fused CE is a mean over non-ignored rows. The first
                    // target row carries `-logprob_coeff * active_rows`, so
                    // this product is exactly `logprob_coeff * sum(logp)`.
                    const scales = try bld.sliceLastDim(targets, 2, 3);
                    const first_scale_2d = try sliceRows2d(bld, scales, 0, 1, 1);
                    const first_scale = try bld.reshape(first_scale_2d, Shape.scalar(.f32));
                    return bld.mul(chunk_loss, first_scale);
                }
                const weight = try bld.scalarConst(.f32, @as(f32, @floatFromInt(chunk_rows)) / @as(f32, @floatFromInt(supervised_rows)));
                const weighted = try bld.mul(chunk_loss, weight);
                total_loss = if (total_loss) |acc| try bld.add(acc, weighted) else weighted;
            }
            return total_loss.?;
        }

        // Materialized-logits chunks remain the production path until the
        // combined linear-CE backward-input kernel is trajectory-qualified.
        // Hard labels select the exact target log-probability sparsely;
        // teacher top-k retains its dense soft-target semantics.
        const use_sparse_hard_labels = teacher_top_k == 0 and !sparseLogitsCrossEntropyDisabled();
        const vocab_row = if (!use_sparse_hard_labels) blk: {
            const vocab_size: usize = @intCast(self.graph_config.vocab_size);
            const vocab_ids = try bld.graph.allocator.alloc(f32, vocab_size);
            defer bld.graph.allocator.free(vocab_ids);
            for (vocab_ids, 0..) |*value, idx| value.* = @floatFromInt(idx);
            break :blk try bld.tensorConst(vocab_ids, Shape.init(.f32, &.{ 1, @as(i64, @intCast(vocab_size)) }));
        } else ml.graph.null_node;

        var total_loss: ?NodeId = null;
        var start: u32 = 0;
        while (start < supervised_rows) : (start += loss_chunk_rows) {
            const end = @min(start + loss_chunk_rows, supervised_rows);
            const chunk_rows = end - start;
            const hidden_chunk = try sliceRows2d(bld, supervised_hidden, start, end, hidden_size);
            const raw_logits = try bld.linearNoBias(hidden_chunk, lm_head_w, chunk_rows, hidden_size, self.graph_config.vocab_size);
            const logits = try self.applyFinalLogitSoftcap(bld, raw_logits);
            if (supervised_rows <= loss_chunk_rows) self.lm_logits = logits;
            const chunk_loss = if (use_sparse_hard_labels) blk: {
                const label_chunk = try sliceRows2d(bld, labels.?, start, end, 1);
                if (weighted_hard_targets) {
                    const weight_chunk = try sliceRows2d(bld, hard_label_weights.?, start, end, 1);
                    const selected = try sparseSelectedHardLabelLogprobs(bld, logits, label_chunk, chunk_rows, self.graph_config.vocab_size);
                    break :blk try sparseWeightedSelectedLogprobCrossEntropy(bld, selected, weight_chunk);
                }
                break :blk try sparseHardLabelCrossEntropy(bld, logits, label_chunk, chunk_rows, self.graph_config.vocab_size);
            } else blk: {
                const dense_targets = if (teacher_top_k == 0) dense: {
                    const label_chunk = try sliceRows2d(bld, labels.?, start, end, 1);
                    break :dense try oneHotTargets(bld, label_chunk, vocab_row, chunk_rows, self.graph_config.vocab_size);
                } else dense: {
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
                    break :dense mixed orelse return error.InvalidTeacherDistillationTargets;
                };
                break :blk try bld.crossEntropyLoss(logits, dense_targets);
            };
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

    pub fn captureGraphContext(ctx_opaque: *anyopaque, allocator: std.mem.Allocator) anyerror!?*anyopaque {
        const self: *GemmaAutodiffCtx = @ptrCast(@alignCast(ctx_opaque));
        const snapshot = try allocator.create(GemmaAutodiffCtx);
        snapshot.* = self.*;
        return @ptrCast(snapshot);
    }

    pub fn restoreGraphContext(ctx_opaque: *anyopaque, state: ?*anyopaque) anyerror!void {
        const self: *GemmaAutodiffCtx = @ptrCast(@alignCast(ctx_opaque));
        const snapshot: *const GemmaAutodiffCtx = @ptrCast(@alignCast(state orelse return error.MissingGraphContextState));
        self.* = snapshot.*;
    }

    pub fn deinitGraphContext(state: ?*anyopaque, allocator: std.mem.Allocator) void {
        const snapshot: *GemmaAutodiffCtx = @ptrCast(@alignCast(state orelse return));
        allocator.destroy(snapshot);
    }
};

/// The compact attention VJP is a production default only for the exact E2B
/// topology qualified by the sequence-512 fixed-work campaign. E4B remains on
/// the decomposed graph: its one-step update was close, but the small ordering
/// difference compounded into a materially different 25-step DPO trajectory.
/// The enable variable is therefore an explicit research override for other
/// shapes, while the disable variable remains the unconditional kill switch.
pub fn fusedGqaAttentionExperimentEnabled(config: gemma_graph.Config) bool {
    if (std.c.getenv("TERMITE_METAL_DISABLE_GEMMA_GQA_ATTENTION_FUSION") != null) return false;
    if (std.c.getenv("TERMITE_METAL_ENABLE_GEMMA_GQA_ATTENTION_FUSION") != null) return true;
    return qualifiedE2BTrainingTopology(config);
}

/// Exact Gemma 4 E2B topology covered by the locked sequence-512 training
/// campaign. Optimization policies with model-size-dependent numerical or
/// lifetime risk use this pure predicate for their production default.
pub fn qualifiedE2BTrainingTopology(config: gemma_graph.Config) bool {
    return config.family == .gemma and
        !config.usesMoe() and
        !config.gemma4_mtp_assistant and
        config.hidden_size == 1536 and
        config.num_hidden_layers == 35 and
        config.num_attention_heads == 8 and
        config.effectiveKVHeads() == 1 and
        (config.num_global_key_value_heads == 0 or config.num_global_key_value_heads == 1) and
        config.headDim() == 256 and
        config.global_head_dim == 512 and
        config.intermediate_size == 6144 and
        config.sliding_window == 512 and
        config.sliding_window_pattern == 5 and
        config.num_kv_shared_layers == 20 and
        config.ple_hidden_size == 256;
}

test "gemma4 compact GQA backward defaults only to the qualified E2B topology" {
    var config = gemma_graph.Config{
        .family = .gemma,
        .hidden_size = 1536,
        .num_hidden_layers = 35,
        .num_attention_heads = 8,
        .num_key_value_heads = 1,
        .attention_head_dim = 256,
        .intermediate_size = 6144,
        .sliding_window = 512,
        .sliding_window_pattern = 5,
        .num_kv_shared_layers = 20,
        .global_head_dim = 512,
        .ple_hidden_size = 256,
    };
    try std.testing.expect(qualifiedE2BTrainingTopology(config));

    // E4B's 8Q/2KV topology passed one-step parity but failed the retained
    // multi-step trajectory gate, so changing only its distinguishing KV
    // geometry must fail closed.
    config.num_key_value_heads = 2;
    try std.testing.expect(!qualifiedE2BTrainingTopology(config));

    config.num_key_value_heads = 1;
    config.hidden_size = 2560;
    try std.testing.expect(!qualifiedE2BTrainingTopology(config));
}

const sparse_target_columns: i64 = 2;
const weighted_hard_target_columns: i64 = 4;
const uniform_weighted_target_columns: i64 = 6;
const dpo_pair_target_columns: u32 = 4;
const dpo_pair_graph_variant_marker: u64 = 0x44504f5f50414952; // "DPO_PAIR"
const max_sparse_loss_chunk_rows: u32 = 512;
const default_sparse_loss_chunk_rows: u32 = max_sparse_loss_chunk_rows;
const weighted_logprob_target_top_k: usize = 8;
const weighted_logprob_target_columns: usize = 1 + 2 * weighted_logprob_target_top_k;

fn parseSparseLossChunkRows(raw: []const u8) ?u32 {
    const parsed = std.fmt.parseInt(u32, raw, 10) catch return null;
    if (parsed == 0 or parsed > max_sparse_loss_chunk_rows) return null;
    return parsed;
}

/// Batch a bounded number of supervised rows through the tied vocabulary
/// projection. Gemma 4's 262144-row embedding table is much larger than the
/// logits workspace, so each projection amortizes dispatch and frozen-weight
/// traffic across up to 512 targets. Sparse label selection avoids a dense
/// one-hot tensor and keeps the qualified sequence-512 path below the MLX peak
/// footprint. The override permits smaller memory tiers; the disable flag
/// restores the former one-row rollback.
fn sparseLossChunkRows() u32 {
    if (std.c.getenv("TERMITE_GEMMA4_DISABLE_BATCHED_SPARSE_LOSS") != null) return 1;
    const raw = std.c.getenv("TERMITE_GEMMA4_SPARSE_LOSS_CHUNK_ROWS") orelse return default_sparse_loss_chunk_rows;
    return parseSparseLossChunkRows(std.mem.span(raw)) orelse default_sparse_loss_chunk_rows;
}

/// Resolved graph-construction policy used by Gemma preference-run
/// attestation. Keep this as the single source of truth for the chunking
/// decision so checkpoint identity cannot drift from the graph builder.
pub fn resolvedSparseLossChunkRows() u32 {
    return sparseLossChunkRows();
}

fn cutLinearCrossEntropyEnabled() bool {
    // Diagnostic override for callers that do not make an explicit backend
    // admission decision. Production strict-Metal training sets the typed
    // context field after backend selection; the environment remains useful
    // for focused tests without making native/reference behavior implicit.
    if (envFlagEnabled("TERMITE_METAL_DISABLE_LINEAR_CCE")) return false;
    const raw = std.c.getenv("TERMITE_ENABLE_CUT_LINEAR_CROSS_ENTROPY") orelse return false;
    return raw[0] == '1' or raw[0] == 't' or raw[0] == 'T' or raw[0] == 'y' or raw[0] == 'Y';
}

fn sparseLogitsCrossEntropyDisabled() bool {
    return envFlagEnabled("TERMITE_GEMMA4_DISABLE_SPARSE_LOGITS_CROSS_ENTROPY");
}

/// Whether the bounded sparse-logits objective is admitted for newly built
/// graphs. Exposed for the preference numerical-policy report and fingerprint.
pub fn resolvedSparseLogitsCrossEntropyEnabled() bool {
    return !sparseLogitsCrossEntropyDisabled();
}

fn envFlagEnabled(name: [*:0]const u8) bool {
    const raw = std.c.getenv(name) orelse return false;
    return raw[0] == '1' or raw[0] == 't' or raw[0] == 'T' or raw[0] == 'y' or raw[0] == 'Y';
}

/// Keep preference graphs in four response-length buckets for the production
/// sequence-512 lane. Longer contexts round to 512-row multiples. Padding rows
/// carry zero loss weight, so this only trades a bounded amount of projection
/// work for compiled-graph reuse across heterogeneous examples.
fn weightedHardTargetRows(supervised_tokens: usize) !usize {
    if (supervised_tokens == 0) return error.NoSupervisedTokens;
    if (supervised_tokens <= 64) return 64;
    if (supervised_tokens <= 128) return 128;
    if (supervised_tokens <= 256) return 256;
    if (supervised_tokens <= max_sparse_loss_chunk_rows) return max_sparse_loss_chunk_rows;
    const chunks = try std.math.divCeil(usize, supervised_tokens, max_sparse_loss_chunk_rows);
    return std.math.mul(usize, chunks, max_sparse_loss_chunk_rows);
}

/// Public preference-scheduler view of the compact weighted-target bucket.
/// DPO uses the maximum chosen/rejected value so both branches share one
/// compiled signature without changing either branch's nonzero target rows.
pub fn preferenceTargetRows(supervised_tokens: usize) !usize {
    return weightedHardTargetRows(supervised_tokens);
}

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

/// Exact hard-label CE over already-materialized logits without constructing
/// a dense one-hot target. The ordinary gather/scatter-add autodiff path keeps
/// the projection VJP on the same qualified graph kernels as dense CE.
fn sparseHardLabelCrossEntropy(
    bld: *Builder,
    logits: NodeId,
    labels: NodeId,
    rows: u32,
    vocab_size: u32,
) !NodeId {
    const selected = try sparseSelectedHardLabelLogprobs(bld, logits, labels, rows, vocab_size);
    return bld.neg(try bld.reduceMean(selected, &.{0}));
}

/// Signed hard-label CE used by DPO/GRPO. Weights are allowed to be negative:
/// `-mean(weight[i] * log_prob[i])` is exactly the scalar objective whose VJP
/// produces the caller-supplied log-probability coefficients. Zero-weight
/// padding rows make preference graph signatures independent of exact response
/// length without changing the objective.
fn sparseWeightedHardLabelCrossEntropy(
    bld: *Builder,
    logits: NodeId,
    labels: NodeId,
    weights: NodeId,
    rows: u32,
    vocab_size: u32,
) !NodeId {
    const selected = try sparseSelectedHardLabelLogprobs(bld, logits, labels, rows, vocab_size);
    return sparseWeightedSelectedLogprobCrossEntropy(bld, selected, weights);
}

fn sparseWeightedSelectedLogprobCrossEntropy(
    bld: *Builder,
    selected_logprobs: NodeId,
    weights: NodeId,
) !NodeId {
    const weighted = try bld.mul(selected_logprobs, weights);
    return bld.neg(try bld.reduceMean(weighted, &.{0}));
}

fn sparseSelectedHardLabelLogprobs(
    bld: *Builder,
    logits: NodeId,
    labels: NodeId,
    rows: u32,
    vocab_size: u32,
) !NodeId {
    const rows_i: i64 = @intCast(rows);
    const vocab_i: i64 = @intCast(vocab_size);
    const logits_shape = bld.graph.node(logits).output_shape;
    if (logits_shape.rank() != 2 or logits_shape.dim(0) != rows_i or logits_shape.dim(1) != vocab_i) {
        return error.ShapeMismatch;
    }
    const log_probs = try bld.logSoftmax(logits);
    const labels_flat = try bld.reshape(labels, Shape.init(.f32, &.{rows_i}));
    const label_indices = try bld.convertDtype(labels_flat, .i64);

    // Gather is row-oriented. Transpose to [vocab, rows], select each label's
    // row, then take the diagonal so result[i] is log_probs[i, labels[i]].
    // This avoids forming flat indices above f32's exact-integer range while
    // keeping the intermediate bounded at rows squared rather than rows*vocab.
    const by_vocab = try bld.transpose(log_probs, &.{ 1, 0 });
    const selected_by_label = try bld.gather(by_vocab, label_indices, Shape.init(.f32, &.{ rows_i, rows_i }));
    // Retain a singleton feature dimension because Metal's resident
    // gather/scatter-add contract is a 2-D row table.
    const selected_flat = try bld.reshape(selected_by_label, Shape.init(.f32, &.{ rows_i * rows_i, 1 }));

    const offsets = try bld.graph.allocator.alloc(f32, rows);
    defer bld.graph.allocator.free(offsets);
    for (offsets, 0..) |*offset, row| {
        offset.* = @floatFromInt(row * (@as(usize, @intCast(rows)) + 1));
    }
    const row_offsets = try bld.tensorConst(offsets, Shape.init(.f32, &.{rows_i}));
    const diagonal_indices = try bld.convertDtype(row_offsets, .i64);
    return bld.gather(selected_flat, diagonal_indices, Shape.init(.f32, &.{ rows_i, 1 }));
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

/// Configure the two private-buffer reuse tiers for one complete preference
/// run. Entry is frame-drained; the returned scope restores both policies on
/// every exit path. In-frame reuse is admitted only by the caller after graph
/// liveness, encoder-fence, and trajectory-parity gates have passed.
pub fn configureMetalBufferReuseForPreferenceRun(
    trainer: *real_autodiff.RealAutodiffTrainer,
    in_frame_reuse_enabled: bool,
    completion_cache_enabled: bool,
) !metal_compute_mod.MetalCompute.BufferReuseOverride {
    if (trainer.compute_backend.kind() == .metal) {
        try trainer.compute_backend.decoderRuntimeSubmitAndWaitFrame();
    }
    return metal_compute_mod.MetalCompute.overrideBufferReuseWithCompletionCache(
        trainer.compute_backend,
        in_frame_reuse_enabled,
        completion_cache_enabled,
    );
}

pub fn metalCompletionCacheStats(
    trainer: *real_autodiff.RealAutodiffTrainer,
) metal_compute_mod.MetalCompute.CompletionCacheStats {
    return metal_compute_mod.MetalCompute.completionCacheStats(trainer.compute_backend);
}

pub fn makeTrainerInputForExample(
    allocator: std.mem.Allocator,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
) !OwnedTrainerInput {
    return makeTrainerInputForExampleWeighted(allocator, ctx, example, seq_len, null, null, null);
}

pub fn makeTrainerInputForExampleScaled(
    allocator: std.mem.Allocator,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    token_scale_override: ?f32,
) !OwnedTrainerInput {
    return makeTrainerInputForExampleWeighted(allocator, ctx, example, seq_len, token_scale_override, null, null);
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
    return makeTrainerInputForExampleWeighted(allocator, ctx, example, seq_len, null, token_scales, null);
}

/// Builds one contamination-free GRPO backward over independent batch rows.
/// Each row owns one prompt/completion sequence, so ordinary causal attention
/// and RoPE already provide the same document isolation and position reset as
/// explicit sequence packing without requiring a block-diagonal attention
/// mask. The sparse targets use flattened predictor rows and scale their
/// weights so the scalar objective is exactly
/// `sum(logprob_grads[i] * token_logprobs[i])` across the complete group.
pub fn makeTrainerInputForTokenLogprobGradBatch(
    allocator: std.mem.Allocator,
    ctx: *GemmaAutodiffCtx,
    examples: []const gemma4.PreparedExampleInput,
    seq_len: u32,
    logprob_grads: []const []const f32,
) !OwnedTrainerInput {
    ctx.dpo_pair_bucket_rows = null;
    if (examples.len == 0 or examples.len != logprob_grads.len) {
        return error.InvalidCompletionGroup;
    }

    const batch: u32 = std.math.cast(u32, examples.len) orelse return error.InvalidCompletionGroup;
    const rows: usize = @intCast(seq_len);
    const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
    const batch_elements = std.math.mul(usize, examples.len, rows) catch return error.SequenceTooLong;

    var total_supervised_tokens: usize = 0;
    for (examples, logprob_grads) |*example, gradients| {
        const supervised_tokens = try validatePreparedExample(example, rows, vocab_size);
        if (supervised_tokens == 0 or gradients.len != supervised_tokens) {
            return error.GradientShapeMismatch;
        }
        for (gradients) |gradient| {
            if (!std.math.isFinite(gradient)) return error.NonFiniteLogprobGradient;
        }
        total_supervised_tokens = std.math.add(usize, total_supervised_tokens, supervised_tokens) catch
            return error.SequenceTooLong;
    }
    const target_rows = try weightedHardTargetRows(total_supervised_tokens);
    const target_elements = std.math.mul(usize, target_rows, @as(usize, weighted_hard_target_columns)) catch
        return error.SequenceTooLong;

    const input_ids = try allocator.alloc(i64, batch_elements);
    errdefer allocator.free(input_ids);
    @memset(input_ids, 0);
    const attention_mask = try allocator.alloc(f32, batch_elements);
    errdefer allocator.free(attention_mask);
    @memset(attention_mask, 0.0);
    const targets = try allocator.alloc(f32, target_elements);
    errdefer allocator.free(targets);
    @memset(targets, 0.0);
    for (0..target_rows) |target_row| {
        targets[target_row * @as(usize, weighted_hard_target_columns) + 1] = -100.0;
    }

    const target_scale: f32 = @floatFromInt(target_rows);
    var target_row: usize = 0;
    for (examples, logprob_grads, 0..) |*example, gradients, batch_idx| {
        const row_base = batch_idx * rows;
        const usable = @min(example.input_ids.len, rows);
        for (example.input_ids[0..usable], 0..) |token_id, token_idx| {
            input_ids[row_base + token_idx] = token_id;
            attention_mask[row_base + token_idx] = 1.0;
        }

        var gradient_idx: usize = 0;
        for (example.labels[0..usable], 0..) |label, label_row| {
            if (label < 0) continue;
            if (label_row == 0 or gradient_idx >= gradients.len or target_row >= target_rows) {
                return error.GradientShapeMismatch;
            }
            const target_base = target_row * @as(usize, weighted_hard_target_columns);
            targets[target_base] = @floatFromInt(row_base + label_row - 1);
            targets[target_base + 1] = @floatFromInt(label);
            // sparseWeightedSelectedLogprobCrossEntropy negates the mean.
            // Multiplying by the fixed target bucket recovers the requested
            // summed log-probability VJP independent of padding rows.
            targets[target_base + 2] = -gradients[gradient_idx] * target_scale;
            target_row += 1;
            gradient_idx += 1;
        }
        if (gradient_idx != gradients.len) return error.GradientShapeMismatch;
    }
    if (target_row != total_supervised_tokens) return error.GradientShapeMismatch;

    return .{
        .input_ids = input_ids,
        .attention_mask = attention_mask,
        .targets = targets,
        .supervised_tokens = total_supervised_tokens,
        .trainer_input = .{
            .ctx = @ptrCast(ctx),
            .build_forward = &GemmaAutodiffCtx.buildForward,
            .build_loss = &GemmaAutodiffCtx.buildLoss,
            .input_ids = input_ids,
            .attention_mask = attention_mask,
            .targets = targets,
            .targets_shape = Shape.init(.f32, &.{
                @as(i64, @intCast(target_rows)),
                weighted_hard_target_columns,
            }),
            .batch = batch,
            .seq_len = seq_len,
            .bind_arch_inputs = null,
            .remap_graph_nodes = &GemmaAutodiffCtx.remapGraphNodes,
            .capture_graph_context = &GemmaAutodiffCtx.captureGraphContext,
            .restore_graph_context = &GemmaAutodiffCtx.restoreGraphContext,
            .deinit_graph_context = &GemmaAutodiffCtx.deinitGraphContext,
        },
    };
}

/// Builds one backward input for candidate completions that are all one token
/// from the same prompt. Multiple target coefficients occupy the same causal
/// predictor row, so one backward pass is exactly the sum of the candidate
/// log-probability gradients before optimizer reduction. This is shared by
/// one-token GRPO groups and chosen/rejected DPO pairs.
pub fn makeTrainerInputForSingleTokenCandidatesLogprobGrads(
    allocator: std.mem.Allocator,
    ctx: *GemmaAutodiffCtx,
    representative_example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    completion_token_ids: []const i32,
    logprob_grads: []const f32,
) !OwnedTrainerInput {
    ctx.dpo_pair_bucket_rows = null;
    if (completion_token_ids.len == 0 or completion_token_ids.len != logprob_grads.len) {
        return error.GradientShapeMismatch;
    }
    if (representative_example.num_supervised_tokens != 1) return error.ExpectedSingleSupervisedToken;

    const rows: usize = @intCast(seq_len);
    const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
    const valid_tokens = try validatePreparedExample(representative_example, rows, vocab_size);
    if (valid_tokens != 1) return error.ExpectedSingleSupervisedToken;

    const input_ids = try allocator.alloc(i64, rows);
    errdefer allocator.free(input_ids);
    @memset(input_ids, 0);
    const attention_mask = try allocator.alloc(f32, rows);
    errdefer allocator.free(attention_mask);
    @memset(attention_mask, 0.0);
    const usable = @min(representative_example.input_ids.len, rows);
    for (0..usable) |idx| {
        input_ids[idx] = representative_example.input_ids[idx];
        attention_mask[idx] = 1.0;
    }

    var predictor_row: ?usize = null;
    for (representative_example.labels, 0..) |label, label_row| {
        if (label < 0) continue;
        if (label_row == 0 or predictor_row != null) return error.ExpectedSingleSupervisedToken;
        predictor_row = label_row - 1;
    }
    const row = predictor_row orelse return error.ExpectedSingleSupervisedToken;
    if (row >= rows) return error.PreparedExampleExceedsSequenceLength;

    var distinct_token_ids: [weighted_logprob_target_top_k]i32 = @splat(0);
    var token_weights: [weighted_logprob_target_top_k]f32 = @splat(0.0);
    var distinct_count: usize = 0;
    for (completion_token_ids, logprob_grads) |token_id, grad| {
        if (token_id < 0 or @as(usize, @intCast(token_id)) >= vocab_size) return error.InputTokenOutOfRange;
        if (!std.math.isFinite(grad)) return error.NonFiniteLogprobGradient;
        var slot: ?usize = null;
        for (distinct_token_ids[0..distinct_count], 0..) |existing, idx| {
            if (existing == token_id) {
                slot = idx;
                break;
            }
        }
        if (slot == null) {
            if (distinct_count == weighted_logprob_target_top_k) return error.TooManyDistinctCompletionTokens;
            slot = distinct_count;
            distinct_token_ids[distinct_count] = token_id;
            distinct_count += 1;
        }
        token_weights[slot.?] += -grad;
    }

    const targets = try allocator.alloc(f32, weighted_logprob_target_columns);
    errdefer allocator.free(targets);
    @memset(targets, 0.0);
    targets[0] = @floatFromInt(row);
    for (0..weighted_logprob_target_top_k) |idx| {
        targets[1 + idx] = @floatFromInt(distinct_token_ids[idx]);
        targets[1 + weighted_logprob_target_top_k + idx] = token_weights[idx];
    }

    return .{
        .input_ids = input_ids,
        .attention_mask = attention_mask,
        .targets = targets,
        .supervised_tokens = 1,
        .trainer_input = .{
            .ctx = @ptrCast(ctx),
            .build_forward = &GemmaAutodiffCtx.buildForward,
            .build_loss = &GemmaAutodiffCtx.buildLoss,
            .input_ids = input_ids,
            .attention_mask = attention_mask,
            .targets = targets,
            .targets_shape = Shape.init(.f32, &.{ 1, weighted_logprob_target_columns }),
            .batch = 1,
            .seq_len = seq_len,
            .bind_arch_inputs = null,
            .remap_graph_nodes = &GemmaAutodiffCtx.remapGraphNodes,
            .capture_graph_context = &GemmaAutodiffCtx.captureGraphContext,
            .restore_graph_context = &GemmaAutodiffCtx.restoreGraphContext,
            .deinit_graph_context = &GemmaAutodiffCtx.deinitGraphContext,
        },
    };
}

pub fn makeTrainerInputForSingleTokenCompletionGroupLogprobGrads(
    allocator: std.mem.Allocator,
    ctx: *GemmaAutodiffCtx,
    representative_example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    completion_token_ids: []const i32,
    logprob_grads: []const f32,
) !OwnedTrainerInput {
    return makeTrainerInputForSingleTokenCandidatesLogprobGrads(
        allocator,
        ctx,
        representative_example,
        seq_len,
        completion_token_ids,
        logprob_grads,
    );
}

/// Builds the runtime payload for one fully compiled DPO update. Chosen and
/// rejected occupy independent batch rows, so causal attention cannot cross
/// between them. `split_batch1` builds two exact batch-1 forwards under one
/// objective; `batched_forward` builds one batch-2 forward and applies the
/// same scalar DPO loss through `buildLoss`. The final target row carries only
/// scalar DPO metadata:
/// `[reference_margin, beta, chosen_token_count, rejected_token_count]`.
pub fn makeTrainerInputForDpoPair(
    allocator: std.mem.Allocator,
    ctx: *GemmaAutodiffCtx,
    chosen: *const gemma4.PreparedExampleInput,
    rejected: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    graph_mode: DpoPairGraphMode,
    reference_chosen_logp: f32,
    reference_rejected_logp: f32,
    beta: f32,
) !OwnedTrainerInput {
    if (!std.math.isFinite(reference_chosen_logp) or
        !std.math.isFinite(reference_rejected_logp) or
        !std.math.isFinite(beta) or beta <= 0.0)
    {
        return error.InvalidDpoPairMetadata;
    }

    const rows: usize = @intCast(seq_len);
    const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
    const chosen_tokens = try validatePreparedExample(chosen, rows, vocab_size);
    const rejected_tokens = try validatePreparedExample(rejected, rows, vocab_size);
    if (chosen_tokens == 0 or rejected_tokens == 0) return error.NoSupervisedTokens;
    const bucket_rows_usize = try weightedHardTargetRows(@max(chosen_tokens, rejected_tokens));
    const bucket_rows: u32 = @intCast(bucket_rows_usize);
    const batch_elements = try std.math.mul(usize, rows, 2);
    const sparse_rows = try std.math.add(usize, try std.math.mul(usize, bucket_rows_usize, 2), 1);
    const target_columns: usize = dpo_pair_target_columns;
    const target_elements = try std.math.mul(usize, sparse_rows, target_columns);

    const input_ids = try allocator.alloc(i64, batch_elements);
    errdefer allocator.free(input_ids);
    @memset(input_ids, 0);
    const attention_mask = try allocator.alloc(f32, batch_elements);
    errdefer allocator.free(attention_mask);
    @memset(attention_mask, 0.0);
    for (0..@min(chosen.input_ids.len, rows)) |idx| {
        input_ids[idx] = chosen.input_ids[idx];
        attention_mask[idx] = 1.0;
    }
    for (0..@min(rejected.input_ids.len, rows)) |idx| {
        input_ids[rows + idx] = rejected.input_ids[idx];
        attention_mask[rows + idx] = 1.0;
    }

    const targets = try allocator.alloc(f32, target_elements);
    errdefer allocator.free(targets);
    @memset(targets, 0.0);
    for (0..bucket_rows_usize * 2) |target_row| {
        targets[target_row * target_columns + 1] = -100.0;
    }
    try fillDpoPairTargetBlock(targets, 0, bucket_rows_usize, 0, chosen, rows, vocab_size);
    const rejected_row_offset = switch (graph_mode) {
        .split_batch1 => 0,
        .batched_forward => rows,
    };
    try fillDpoPairTargetBlock(targets, bucket_rows_usize, bucket_rows_usize, rejected_row_offset, rejected, rows, vocab_size);
    const metadata_base = bucket_rows_usize * 2 * target_columns;
    targets[metadata_base] = reference_chosen_logp - reference_rejected_logp;
    targets[metadata_base + 1] = beta;
    targets[metadata_base + 2] = @floatFromInt(chosen_tokens);
    targets[metadata_base + 3] = @floatFromInt(rejected_tokens);

    ctx.dpo_pair_bucket_rows = bucket_rows;
    return .{
        .input_ids = input_ids,
        .attention_mask = attention_mask,
        .targets = targets,
        .supervised_tokens = chosen_tokens + rejected_tokens,
        .trainer_input = .{
            .ctx = @ptrCast(ctx),
            .build_forward = &GemmaAutodiffCtx.buildForward,
            .build_loss = &GemmaAutodiffCtx.buildLoss,
            .build_objective = switch (graph_mode) {
                .split_batch1 => &GemmaAutodiffCtx.buildDpoPairObjective,
                .batched_forward => null,
            },
            .input_ids = input_ids,
            .attention_mask = attention_mask,
            .targets = targets,
            .targets_shape = Shape.init(.f32, &.{ @as(i64, @intCast(sparse_rows)), @as(i64, dpo_pair_target_columns) }),
            .batch = 2,
            .seq_len = seq_len,
            .bind_arch_inputs = null,
            .remap_graph_nodes = &GemmaAutodiffCtx.remapGraphNodes,
            .capture_graph_context = &GemmaAutodiffCtx.captureGraphContext,
            .restore_graph_context = &GemmaAutodiffCtx.restoreGraphContext,
            .deinit_graph_context = &GemmaAutodiffCtx.deinitGraphContext,
            .graph_variant = .{ dpo_pair_graph_variant_marker, bucket_rows, @intFromEnum(graph_mode), 0 },
        },
    };
}

fn fillDpoPairTargetBlock(
    targets: []f32,
    target_row_start: usize,
    target_row_capacity: usize,
    flattened_row_offset: usize,
    example: *const gemma4.PreparedExampleInput,
    seq_len: usize,
    vocab_size: usize,
) !void {
    const usable = @min(example.input_ids.len, seq_len);
    const label_limit = @min(example.labels.len, usable);
    var supervised_idx: usize = 0;
    for (1..label_limit) |label_row| {
        const label = example.labels[label_row];
        if (label < 0) continue;
        if (supervised_idx == target_row_capacity) return error.DpoPairTargetBucketOverflow;
        const token_id: usize = @intCast(label);
        if (token_id >= vocab_size) return error.LabelOutOfRange;
        const target_base = (target_row_start + supervised_idx) * @as(usize, dpo_pair_target_columns);
        targets[target_base] = @floatFromInt(flattened_row_offset + label_row - 1);
        targets[target_base + 1] = @floatFromInt(token_id);
        supervised_idx += 1;
    }
}

fn makeTrainerInputForExampleWeighted(
    allocator: std.mem.Allocator,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    token_scale_override: ?f32,
    token_scales: ?[]const f32,
    uniform_logprob_coeff: ?f32,
) !OwnedTrainerInput {
    return makeTrainerInputForExampleWeightedScheduled(
        allocator,
        ctx,
        example,
        seq_len,
        token_scale_override,
        token_scales,
        uniform_logprob_coeff,
        null,
    );
}

fn makeTrainerInputForExampleWeightedScheduled(
    allocator: std.mem.Allocator,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    token_scale_override: ?f32,
    token_scales: ?[]const f32,
    uniform_logprob_coeff: ?f32,
    weighted_target_rows_override: ?usize,
) !OwnedTrainerInput {
    ctx.dpo_pair_bucket_rows = null;
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

    if (uniform_logprob_coeff) |coeff| if (!std.math.isFinite(coeff)) return error.NonFiniteLogprobGradient;
    const requested_weighted_targets = token_scales != null or token_scale_override != null or uniform_logprob_coeff != null;
    const minimum_weighted_target_rows = if (requested_weighted_targets) try weightedHardTargetRows(valid_tokens) else 0;
    const requested_weighted_target_rows = if (weighted_target_rows_override) |target_rows| rows: {
        if (!requested_weighted_targets or
            target_rows < minimum_weighted_target_rows or
            (try weightedHardTargetRows(target_rows)) != target_rows)
        {
            return error.InvalidPreferenceTargetRows;
        }
        break :rows target_rows;
    } else minimum_weighted_target_rows;
    // The compact uniform target layout stores an ignore-index-padded label
    // bucket plus one aggregate scale. That contract is interpreted only by
    // fused linear CE. Materialized logits need an explicit weight per row;
    // selecting the ordinary weighted layout there is both exact and prevents
    // the -100 padding label from reaching gather as a vocabulary index.
    const fused_linear_ce_enabled = ctx.enable_fused_linear_cross_entropy orelse cutLinearCrossEntropyEnabled();
    const uniform_weighted_targets = uniform_logprob_coeff != null and
        requested_weighted_target_rows <= max_sparse_loss_chunk_rows and
        fused_linear_ce_enabled;
    const weighted_hard_targets = requested_weighted_targets and !uniform_weighted_targets;
    const sparse_hard_targets = !weighted_hard_targets and
        token_scale_override == null and
        !teacher_fields_present;
    const sparse_teacher_targets = !weighted_hard_targets and
        teacher_fields_present;
    const weighted_target_rows = if (requested_weighted_targets) requested_weighted_target_rows else 0;
    const sparse_teacher_columns = if (sparse_teacher_targets)
        try std.math.add(usize, 1, try std.math.mul(usize, 2, example.teacher_top_k))
    else
        0;
    const targets_shape = if (uniform_weighted_targets)
        Shape.init(.f32, &.{ @as(i64, @intCast(weighted_target_rows)), uniform_weighted_target_columns })
    else if (weighted_hard_targets)
        Shape.init(.f32, &.{ @as(i64, @intCast(weighted_target_rows)), weighted_hard_target_columns })
    else if (sparse_hard_targets)
        Shape.init(.f32, &.{ @as(i64, @intCast(valid_tokens)), sparse_target_columns })
    else if (sparse_teacher_targets)
        Shape.init(.f32, &.{ @as(i64, @intCast(valid_tokens)), @as(i64, @intCast(sparse_teacher_columns)) })
    else
        Shape.init(.f32, &.{ @as(i64, @intCast(rows)), @as(i64, @intCast(vocab_size)) });
    const target_elements = if (uniform_weighted_targets)
        weighted_target_rows * @as(usize, @intCast(uniform_weighted_target_columns))
    else if (weighted_hard_targets)
        weighted_target_rows * @as(usize, @intCast(weighted_hard_target_columns))
    else if (sparse_hard_targets)
        valid_tokens * @as(usize, @intCast(sparse_target_columns))
    else if (sparse_teacher_targets)
        valid_tokens * sparse_teacher_columns
    else
        rows * vocab_size;
    const targets = try allocator.alloc(f32, target_elements);
    errdefer allocator.free(targets);
    @memset(targets, 0.0);

    if (uniform_weighted_targets) {
        const target_columns: usize = @intCast(uniform_weighted_target_columns);
        for (0..weighted_target_rows) |target_row| targets[target_row * target_columns + 1] = -100.0;
        var supervised_idx: usize = 0;
        for (1..label_limit) |i| {
            const label = example.labels[i];
            if (label < 0) continue;
            const token_idx: usize = @intCast(label);
            if (token_idx >= vocab_size) return error.LabelOutOfRange;
            const target_base = supervised_idx * target_columns;
            targets[target_base] = @floatFromInt(i - 1);
            targets[target_base + 1] = @floatFromInt(token_idx);
            supervised_idx += 1;
        }
        targets[2] = -uniform_logprob_coeff.? * @as(f32, @floatFromInt(valid_tokens));
    } else if (weighted_hard_targets) {
        // Preserve the dense CE contract exactly. Dense targets were averaged
        // over every sequence row; sparse targets are averaged over the
        // response-length bucket. Rescale each target by bucket_rows/seq_rows
        // so both scalar loss and adapter VJP remain unchanged.
        const bucket_over_sequence = @as(f32, @floatFromInt(weighted_target_rows)) /
            @as(f32, @floatFromInt(rows));
        const default_row_scale: f32 = token_scale_override orelse if (uniform_logprob_coeff) |coeff|
            -coeff * @as(f32, @floatFromInt(rows))
        else
            @as(f32, @floatFromInt(rows)) / @as(f32, @floatFromInt(valid_tokens));
        var supervised_idx: usize = 0;
        for (1..label_limit) |i| {
            const label = example.labels[i];
            if (label < 0) continue;
            const token_idx: usize = @intCast(label);
            if (token_idx >= vocab_size) return error.LabelOutOfRange;
            const dense_row_scale = if (token_scales) |scales| scales[supervised_idx] else default_row_scale;
            const target_base = supervised_idx * @as(usize, @intCast(weighted_hard_target_columns));
            targets[target_base] = @floatFromInt(i - 1);
            targets[target_base + 1] = @floatFromInt(token_idx);
            targets[target_base + 2] = dense_row_scale * bucket_over_sequence;
            supervised_idx += 1;
        }
    } else if (sparse_hard_targets) {
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
            .capture_graph_context = &GemmaAutodiffCtx.captureGraphContext,
            .restore_graph_context = &GemmaAutodiffCtx.restoreGraphContext,
            .deinit_graph_context = &GemmaAutodiffCtx.deinitGraphContext,
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
    return makeTrainerInputForExampleWeighted(allocator, ctx, example, seq_len, null, null, logprob_coeff);
}

pub fn makeTrainerInputForLogprobCoeffScheduled(
    allocator: std.mem.Allocator,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    weighted_target_rows: usize,
    logprob_coeff: f32,
) !OwnedTrainerInput {
    return makeTrainerInputForExampleWeightedScheduled(
        allocator,
        ctx,
        example,
        seq_len,
        null,
        null,
        logprob_coeff,
        weighted_target_rows,
    );
}

pub fn singleTokenCandidateLogprobsForPrompt(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    prompt: []const i32,
    candidate_token_ids: []const i32,
    seq_len: u32,
    out_logps: []f32,
) !void {
    return singleTokenCandidateLogprobsForPromptWithBindings(
        allocator,
        trainer,
        ctx,
        prompt,
        candidate_token_ids,
        seq_len,
        out_logps,
        null,
    );
}

pub fn singleTokenCandidateLogprobsForPromptFrozenBase(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    prompt: []const i32,
    candidate_token_ids: []const i32,
    seq_len: u32,
    out_logps: []f32,
    frozen_lora: *const FrozenBaseLoraBindings,
) !void {
    return singleTokenCandidateLogprobsForPromptWithBindings(
        allocator,
        trainer,
        ctx,
        prompt,
        candidate_token_ids,
        seq_len,
        out_logps,
        frozen_lora,
    );
}

fn singleTokenCandidateLogprobsForPromptWithBindings(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    prompt: []const i32,
    candidate_token_ids: []const i32,
    seq_len: u32,
    out_logps: []f32,
    frozen_lora: ?*const FrozenBaseLoraBindings,
) !void {
    if (prompt.len == 0) return error.EmptyPrompt;
    if (prompt.len >= @as(usize, @intCast(seq_len))) return error.NoCompletionBudget;
    if (candidate_token_ids.len == 0 or candidate_token_ids.len != out_logps.len) {
        return error.LogpLenMismatch;
    }

    // Every candidate is predicted by the final prompt row. Project that one
    // hidden row once instead of materializing [sequence, vocabulary] logits
    // independently for each candidate completion.
    const logits = try executeSparseLogitsForInputIds(
        allocator,
        trainer,
        ctx,
        prompt,
        seq_len,
        frozen_lora,
    );
    defer allocator.free(logits);

    const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
    if (logits.len != vocab_size) return error.InvalidTrainerLogits;
    for (candidate_token_ids, out_logps) |token_id, *out_logp| {
        if (token_id < 0 or @as(usize, @intCast(token_id)) >= vocab_size) {
            return error.InputTokenOutOfRange;
        }
        out_logp.* = logProbAtToken(logits, @intCast(token_id));
    }
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
    return tokenLogprobsForPromptCompletionWithBindings(
        allocator,
        trainer,
        ctx,
        prompt,
        completion,
        seq_len,
        out_logps,
        null,
        false,
    );
}

pub fn tokenLogprobsForPromptCompletionSparseRows(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    prompt: []const i32,
    completion: []const i32,
    seq_len: u32,
    out_logps: []f32,
) !void {
    return tokenLogprobsForPromptCompletionWithBindings(
        allocator,
        trainer,
        ctx,
        prompt,
        completion,
        seq_len,
        out_logps,
        null,
        true,
    );
}

pub fn tokenLogprobsForPromptCompletionFrozenBase(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    prompt: []const i32,
    completion: []const i32,
    seq_len: u32,
    out_logps: []f32,
    frozen_lora: *const FrozenBaseLoraBindings,
) !void {
    return tokenLogprobsForPromptCompletionWithBindings(
        allocator,
        trainer,
        ctx,
        prompt,
        completion,
        seq_len,
        out_logps,
        frozen_lora,
        false,
    );
}

pub fn tokenLogprobsForPromptCompletionSparseRowsFrozenBase(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    prompt: []const i32,
    completion: []const i32,
    seq_len: u32,
    out_logps: []f32,
    frozen_lora: *const FrozenBaseLoraBindings,
) !void {
    return tokenLogprobsForPromptCompletionWithBindings(
        allocator,
        trainer,
        ctx,
        prompt,
        completion,
        seq_len,
        out_logps,
        frozen_lora,
        true,
    );
}

/// Scores a completion group with one padded batch forward and one sparse
/// vocabulary projection. Predictor rows are global rows in the flattened
/// [batch, sequence, hidden] activation. Returning false is an explicit,
/// allocation-free admission failure: callers must retain their per-
/// completion scorer when the requested group exceeds the qualified sparse
/// projection bound.
pub fn tokenLogprobsForPromptCompletionGroupSparseRows(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    prompt: []const i32,
    completions: []const []const i32,
    seq_len: u32,
    out_logps: []const []f32,
) !bool {
    return tokenLogprobsForPromptCompletionGroupSparseRowsWithBindings(
        allocator,
        trainer,
        ctx,
        prompt,
        completions,
        seq_len,
        out_logps,
        null,
    );
}

pub fn tokenLogprobsForPromptCompletionGroupSparseRowsFrozenBase(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    prompt: []const i32,
    completions: []const []const i32,
    seq_len: u32,
    out_logps: []const []f32,
    frozen_lora: *const FrozenBaseLoraBindings,
) !bool {
    return tokenLogprobsForPromptCompletionGroupSparseRowsWithBindings(
        allocator,
        trainer,
        ctx,
        prompt,
        completions,
        seq_len,
        out_logps,
        frozen_lora,
    );
}

fn tokenLogprobsForPromptCompletionGroupSparseRowsWithBindings(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    prompt: []const i32,
    completions: []const []const i32,
    seq_len: u32,
    out_logps: []const []f32,
    frozen_lora: ?*const FrozenBaseLoraBindings,
) !bool {
    if (prompt.len == 0) return error.EmptyPrompt;
    if (completions.len == 0 or completions.len != out_logps.len) return error.InvalidCompletionGroup;

    const rows: usize = @intCast(seq_len);
    var selected_row_count: usize = 0;
    for (completions, out_logps) |completion, output| {
        if (completion.len == 0) return error.EmptyCompletion;
        if (completion.len != output.len) return error.LogpLenMismatch;
        const total_len = std.math.add(usize, prompt.len, completion.len) catch return error.SequenceTooLong;
        if (total_len > rows) return error.SequenceTooLong;
        selected_row_count = std.math.add(usize, selected_row_count, completion.len) catch
            return error.InvalidLogitsRowSelection;
    }
    // The Metal tied-head kernels choose different accumulation geometry for
    // tiny M. That is fast but can move log-probabilities enough to violate
    // GRPO's sampling/rescore contract. Keep the projected M at least one
    // legacy sequence wide, with real rows first and duplicate padding after
    // them. The transformer still runs the completion group as one batch and
    // the caller ignores padding logits, while the LM head retains the same
    // M family as the batch-1 sequence-wide sampler.
    const projection_row_count = @max(selected_row_count, rows);
    if (projection_row_count > sparseLossChunkRows()) return false;

    const joined = try allocator.alloc([]const i32, completions.len);
    defer allocator.free(joined);
    var joined_count: usize = 0;
    defer for (joined[0..joined_count]) |sequence| allocator.free(sequence);

    const predictor_rows = try allocator.alloc(usize, projection_row_count);
    defer allocator.free(predictor_rows);
    for (completions, joined) |completion, *joined_sequence| {
        joined_sequence.* = try concatPromptCompletion(allocator, prompt, completion);
        joined_count += 1;
    }
    try fillCompletionGroupPredictorRows(prompt.len, completions, rows, predictor_rows[0..selected_row_count]);
    @memset(predictor_rows[selected_row_count..], predictor_rows[selected_row_count - 1]);

    const logits = try executeSparseLogitsForInputIdBatchAtRows(
        allocator,
        trainer,
        ctx,
        joined,
        seq_len,
        predictor_rows,
        frozen_lora,
    );
    defer allocator.free(logits);

    const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
    if (logits.len != projection_row_count * vocab_size) return error.InvalidTrainerLogits;
    var selected_idx: usize = 0;
    for (completions, out_logps) |completion, output| {
        for (completion, output) |token_id, *out_logp| {
            if (token_id < 0 or @as(usize, @intCast(token_id)) >= vocab_size) return error.InputTokenOutOfRange;
            const row = logits[selected_idx * vocab_size ..][0..vocab_size];
            out_logp.* = logProbAtToken(row, @intCast(token_id));
            selected_idx += 1;
        }
    }
    return true;
}

fn fillCompletionGroupPredictorRows(
    prompt_len: usize,
    completions: []const []const i32,
    seq_len: usize,
    out_rows: []usize,
) !void {
    if (prompt_len == 0) return error.EmptyPrompt;
    var output_idx: usize = 0;
    for (completions, 0..) |completion, batch_idx| {
        if (completion.len == 0) return error.EmptyCompletion;
        const joined_len = std.math.add(usize, prompt_len, completion.len) catch return error.SequenceTooLong;
        if (joined_len > seq_len) return error.SequenceTooLong;
        const batch_row_base = std.math.mul(usize, batch_idx, seq_len) catch
            return error.InvalidLogitsRowSelection;
        for (completion, 0..) |_, completion_idx| {
            if (output_idx >= out_rows.len) return error.LogpLenMismatch;
            out_rows[output_idx] = batch_row_base + prompt_len + completion_idx - 1;
            output_idx += 1;
        }
    }
    if (output_idx != out_rows.len) return error.LogpLenMismatch;
}

fn tokenLogprobsForPromptCompletionWithBindings(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    prompt: []const i32,
    completion: []const i32,
    seq_len: u32,
    out_logps: []f32,
    frozen_lora: ?*const FrozenBaseLoraBindings,
    sparse_rows: bool,
) !void {
    if (completion.len != out_logps.len) return error.LogpLenMismatch;
    if (prompt.len == 0) return error.EmptyPrompt;
    if (completion.len == 0) return error.EmptyCompletion;
    const total_len = prompt.len + completion.len;
    if (total_len > seq_len) return error.SequenceTooLong;

    const joined = try concatPromptCompletion(allocator, prompt, completion);
    defer allocator.free(joined);
    const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
    if (!sparse_rows) {
        const logits = try executeLogitsForInputIds(allocator, trainer, ctx, joined, seq_len, frozen_lora);
        defer allocator.free(logits);
        for (completion, 0..) |token_id, comp_idx| {
            const predictor_row = prompt.len + comp_idx - 1;
            const row = logits[predictor_row * vocab_size ..][0..vocab_size];
            out_logps[comp_idx] = logProbAtToken(row, @intCast(token_id));
        }
        return;
    }

    const max_rows_per_execution: usize = sparseLossChunkRows();
    var start: usize = 0;
    while (start < completion.len) {
        const end = @min(start + max_rows_per_execution, completion.len);
        const predictor_rows = try allocator.alloc(usize, end - start);
        defer allocator.free(predictor_rows);
        for (predictor_rows, 0..) |*row, local_idx| row.* = prompt.len + start + local_idx - 1;
        const logits = try executeSparseLogitsForInputIdsAtRows(
            allocator,
            trainer,
            ctx,
            joined,
            seq_len,
            predictor_rows,
            frozen_lora,
        );
        defer allocator.free(logits);
        for (completion[start..end], 0..) |token_id, local_idx| {
            const row = logits[local_idx * vocab_size ..][0..vocab_size];
            out_logps[start + local_idx] = logProbAtToken(row, @intCast(token_id));
        }
        start = end;
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
        const logits = try executeLogitsForInputIds(allocator, trainer, ctx, seq.items, seq_len, null);
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

pub const GrpoIncrementalKvTelemetry = struct {
    groups: usize = 0,
    prompt_prefill_forwards: usize = 0,
    prompt_tail_prefill_forwards: usize = 0,
    prompt_tail_prefill_candidates: usize = 0,
    max_prompt_tail_batch_size: usize = 0,
    prompt_tail_clone_candidates: usize = 0,
    prompt_tail_clone_tokens: usize = 0,
    prompt_tail_cloning: bool = false,
    decode_forwards: usize = 0,
    decode_forward_candidates: usize = 0,
    max_decode_batch_size: usize = 0,
    active_candidate_batching: bool = false,
    exact_logprob_rescore_forwards: usize = 0,
    resident_ranked_token_selections: usize = 0,
    host_logit_fallbacks: usize = 0,
    shared_prompt_tokens: usize = 0,
    reused_candidate_prompt_tokens: usize = 0,
    cache_page_tokens: usize = 0,
    cache_dtype: []const u8 = "f32",
};

/// Exactness-first GRPO rollout sampler. One candidate performs the canonical
/// prompt prefill; every other candidate retains its complete KV pages and
/// independently replays only the unaligned tail. Diverged candidates then
/// advance with one-token paged decode. Admission is deliberately limited to
/// dense Gemma Q/V LoRA with by-weight sharing and no dropout until broader
/// adapter targets pass the same replay gates.
pub const GrpoIncrementalKvSampler = struct {
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    config: gemma_graph.Config,
    kv_manager: runtime.kv.manager.KvManager,
    kv_storage: runtime.kv.storage_runtime.KvStorageRuntime,
    pool_id: runtime.kv.block.KvPoolId,
    batch_active_candidates: bool,
    clone_prompt_tails: bool,
    telemetry: GrpoIncrementalKvTelemetry,

    pub fn init(
        allocator: std.mem.Allocator,
        trainer: *real_autodiff.RealAutodiffTrainer,
        ctx: *const GemmaAutodiffCtx,
        batch_active_candidates: bool,
        clone_prompt_tails: bool,
    ) !GrpoIncrementalKvSampler {
        if (trainer.compute_backend.kind() != .metal) return error.GrpoIncrementalKvRequiresMetal;
        if (ctx.graph_config.family != .gemma or ctx.graph_config.usesMoe()) {
            return error.UnsupportedGrpoIncrementalKvModel;
        }
        if (trainer.config.lora.dropout != 0.0) return error.GrpoIncrementalKvRequiresZeroDropout;
        if (trainer.config.lora.sharing != .by_weight) return error.GrpoIncrementalKvRequiresByWeightLoRA;

        const adapters = try trainer.loraAdapterBaseNames();
        if (adapters.len == 0) return error.MissingLoRAAdapters;
        var q_count: usize = 0;
        var v_count: usize = 0;
        for (adapters) |adapter| {
            if (std.mem.endsWith(u8, adapter.base_name, ".self_attn.q_proj.weight")) {
                q_count += 1;
            } else if (std.mem.endsWith(u8, adapter.base_name, ".self_attn.v_proj.weight")) {
                v_count += 1;
            } else {
                return error.UnsupportedGrpoIncrementalKvLoRATarget;
            }
        }
        if (q_count == 0 or v_count == 0) return error.IncompleteGrpoIncrementalKvLoRAInventory;
        try trainer.ensureResidentLoraWeightsForRuntime();
        for (adapters) |adapter| {
            _ = (try trainer.residentLoraBinding(adapter.base_name)) orelse
                return error.LoRAWeightNotResident;
        }

        const pool_config = generation.kvPoolConfig(.metal, .f32, ctx.graph_config, false);
        var kv_manager = runtime.kv.manager.KvManager.init(allocator);
        errdefer kv_manager.deinit();
        const pool_id = try kv_manager.addPool(pool_config);
        var kv_storage = try runtime.kv.storage_runtime.KvStorageRuntime.init(allocator, pool_config);
        errdefer kv_storage.deinit();
        try trainer.compute_backend.provisionKvDeviceWriteHook(&kv_storage);

        return .{
            .allocator = allocator,
            .trainer = trainer,
            .config = ctx.graph_config,
            .kv_manager = kv_manager,
            .kv_storage = kv_storage,
            .pool_id = pool_id,
            .batch_active_candidates = batch_active_candidates,
            .clone_prompt_tails = clone_prompt_tails,
            .telemetry = .{
                .cache_page_tokens = pool_config.page_size_tokens,
                .active_candidate_batching = batch_active_candidates,
                .prompt_tail_cloning = clone_prompt_tails,
            },
        };
    }

    pub fn deinit(self: *GrpoIncrementalKvSampler) void {
        self.trainer.compute_backend.decoderRuntimeSubmitAndWaitFrame() catch {
            if (self.trainer.compute_backend.decoderRuntimeHasActiveFrame()) {
                self.trainer.compute_backend.decoderRuntimeCancelFrame() catch {};
            }
        };
        self.kv_storage.deinit();
        self.kv_manager.deinit();
        self.* = undefined;
    }

    pub fn resetTelemetry(self: *GrpoIncrementalKvSampler) void {
        const page_tokens = self.telemetry.cache_page_tokens;
        self.telemetry = .{
            .cache_page_tokens = page_tokens,
            .active_candidate_batching = self.batch_active_candidates,
            .prompt_tail_cloning = self.clone_prompt_tails,
        };
    }

    fn requireCheckpointQuiescence(self: *GrpoIncrementalKvSampler) !void {
        try self.drainFrame();
        if (self.trainer.compute_backend.decoderRuntimeHasActiveFrame() or
            self.kv_manager.activeSequenceCount() != 0 or
            self.kv_storage.activeSequenceCount() != 0)
        {
            return error.GrpoIncrementalKvCheckpointNotQuiescent;
        }
    }

    fn validateCheckpointTelemetry(
        self: *const GrpoIncrementalKvSampler,
        telemetry: GrpoIncrementalKvTelemetry,
        expected_groups: usize,
    ) !void {
        if (telemetry.groups != expected_groups or
            telemetry.prompt_prefill_forwards != expected_groups or
            telemetry.cache_page_tokens != self.telemetry.cache_page_tokens or
            !std.mem.eql(u8, telemetry.cache_dtype, "f32") or
            telemetry.active_candidate_batching != self.batch_active_candidates or
            telemetry.prompt_tail_cloning != self.clone_prompt_tails or
            telemetry.host_logit_fallbacks > telemetry.resident_ranked_token_selections or
            telemetry.prompt_tail_clone_tokens < telemetry.prompt_tail_clone_candidates or
            telemetry.reused_candidate_prompt_tokens < telemetry.shared_prompt_tokens)
        {
            return error.InvalidGrpoIncrementalKvCheckpointState;
        }
        if (expected_groups == 0 and
            (telemetry.prompt_tail_prefill_forwards != 0 or
                telemetry.prompt_tail_prefill_candidates != 0 or
                telemetry.max_prompt_tail_batch_size != 0 or
                telemetry.prompt_tail_clone_candidates != 0 or
                telemetry.prompt_tail_clone_tokens != 0 or
                telemetry.decode_forwards != 0 or
                telemetry.decode_forward_candidates != 0 or
                telemetry.max_decode_batch_size != 0 or
                telemetry.exact_logprob_rescore_forwards != 0 or
                telemetry.resident_ranked_token_selections != 0 or
                telemetry.host_logit_fallbacks != 0 or
                telemetry.shared_prompt_tokens != 0 or
                telemetry.reused_candidate_prompt_tokens != 0))
        {
            return error.InvalidGrpoIncrementalKvCheckpointState;
        }
    }

    /// Returns cumulative sampler telemetry only after all submitted Metal
    /// work is complete and every logical/device sequence has released its KV
    /// references. Transient pages are deliberately not checkpointed: every
    /// completed GRPO group is an exact replay boundary.
    pub fn checkpointTelemetry(self: *GrpoIncrementalKvSampler) !GrpoIncrementalKvTelemetry {
        try self.requireCheckpointQuiescence();
        try self.validateCheckpointTelemetry(self.telemetry, self.telemetry.groups);
        return self.telemetry;
    }

    /// Restores the cumulative part of an incremental sampler at an epoch
    /// boundary. Cache geometry and execution mode must match the newly
    /// initialized sampler; live KV pages always begin empty.
    pub fn restoreCheckpointTelemetry(
        self: *GrpoIncrementalKvSampler,
        checkpoint: GrpoIncrementalKvTelemetry,
        expected_groups: usize,
    ) !void {
        try self.requireCheckpointQuiescence();
        if (self.telemetry.groups != 0) return error.InvalidGrpoIncrementalKvCheckpointState;
        try self.validateCheckpointTelemetry(checkpoint, expected_groups);
        self.telemetry = checkpoint;
        self.telemetry.cache_dtype = "f32";
    }

    fn runtimeAdapterApply(
        context: *anyopaque,
        cb: *const ComputeBackend,
        allocator: std.mem.Allocator,
        base_name: []const u8,
        input: CT,
        base_output: CT,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) anyerror!?CT {
        _ = allocator;
        const self: *GrpoIncrementalKvSampler = @ptrCast(@alignCast(context));
        const binding = (try self.trainer.residentLoraBinding(base_name)) orelse return null;
        if (binding.in_dim != in_dim or binding.out_dim != out_dim) return error.InvalidLoRAAdapterShape;
        const fused = (try cb.loraLinearBranch(&.{
            .input = input,
            .base = base_output,
            .lora_a = binding.lora_a,
            .lora_b = binding.lora_b,
            .rows = rows,
            .in_dim = in_dim,
            .rank = binding.rank,
            .out_dim = out_dim,
            .scale = binding.scale,
        })) orelse return error.GrpoIncrementalKvLoRAForwardUnavailable;
        cb.free(fused.after_a);
        cb.free(fused.after_b);
        return fused.output;
    }

    fn decodeContext(
        self: *GrpoIncrementalKvSampler,
        state: *generation.NativeDecodeState,
        total_sequence_len: usize,
        query_sequence_len: usize,
    ) gpt_arch.DecodeContext {
        var decode_context = state.gptDecodeContext(total_sequence_len, query_sequence_len);
        decode_context.runtime_linear_adapter = .{
            .context = @ptrCast(self),
            .applyFn = &runtimeAdapterApply,
        };
        return decode_context;
    }

    fn attachRuntimeAdapter(
        self: *GrpoIncrementalKvSampler,
        decode_context: *gpt_arch.DecodeContext,
    ) void {
        decode_context.runtime_linear_adapter = .{
            .context = @ptrCast(self),
            .applyFn = &runtimeAdapterApply,
        };
    }

    fn drainFrame(self: *GrpoIncrementalKvSampler) !void {
        try self.trainer.compute_backend.decoderRuntimeSubmitAndWaitFrame();
    }

    fn cancelFrameAfterError(self: *GrpoIncrementalKvSampler) void {
        self.drainFrame() catch {
            if (self.trainer.compute_backend.decoderRuntimeHasActiveFrame()) {
                self.trainer.compute_backend.decoderRuntimeCancelFrame() catch {};
            }
        };
    }

    fn selectTopRankedTokensResidentAtRow(
        self: *GrpoIncrementalKvSampler,
        logits: CT,
        row_index: usize,
        out_token_ids: []usize,
    ) !void {
        if (out_token_ids.len == 0 or out_token_ids.len > 8) return error.InvalidRankedTokenCount;
        const vocab_size: usize = @intCast(self.config.vocab_size);
        var suppressed: [8]i32 = undefined;
        for (out_token_ids, 0..) |*token_id, rank| {
            var choices: ?[]u32 = null;
            if (rank == 0) {
                choices = try self.trainer.compute_backend.decoderRuntimeArgmaxRowsSuppress(
                    logits,
                    row_index,
                    1,
                    vocab_size,
                    suppressed[0..0],
                    self.allocator,
                );
            }
            if (choices == null) {
                choices = try self.trainer.compute_backend.argmaxRowsSuppress(
                    logits,
                    row_index,
                    1,
                    vocab_size,
                    suppressed[0..rank],
                    self.allocator,
                );
            }
            if (choices) |owned_choices| {
                defer self.allocator.free(owned_choices);
                if (owned_choices.len != 1) return error.InvalidLogitsShape;
                token_id.* = owned_choices[0];
                suppressed[rank] = @intCast(owned_choices[0]);
                continue;
            }

            const host_logits = try self.trainer.compute_backend.toFloat32(logits, self.allocator);
            defer self.allocator.free(host_logits);
            const row_start = std.math.mul(usize, row_index, vocab_size) catch return error.InvalidLogitsShape;
            if (row_start + vocab_size > host_logits.len) return error.InvalidLogitsShape;
            try selectTopRankedTokens(self.allocator, host_logits[row_start..][0..vocab_size], out_token_ids);
            self.telemetry.host_logit_fallbacks += 1;
            self.telemetry.resident_ranked_token_selections += out_token_ids.len;
            return;
        }
        self.telemetry.resident_ranked_token_selections += out_token_ids.len;
    }

    fn selectTopRankedTokensResident(
        self: *GrpoIncrementalKvSampler,
        logits: CT,
        out_token_ids: []usize,
    ) !void {
        try self.selectTopRankedTokensResidentAtRow(logits, 0, out_token_ids);
    }

    fn selectRankedTokenResident(
        self: *GrpoIncrementalKvSampler,
        logits: CT,
        row_index: usize,
        rank: usize,
    ) !usize {
        const clamped_rank = @min(rank, @as(usize, @intCast(self.config.vocab_size)) - 1);
        var ranked: [8]usize = undefined;
        if (clamped_rank >= ranked.len) return error.InvalidRankedTokenCount;
        try self.selectTopRankedTokensResidentAtRow(logits, row_index, ranked[0 .. clamped_rank + 1]);
        return ranked[clamped_rank];
    }

    pub fn sampleCompletionGroupRanked(
        self: *GrpoIncrementalKvSampler,
        prompt: []const i32,
        seq_len: usize,
        max_completion_tokens: usize,
        rank_cap: usize,
        eos_token_id: ?i32,
        out_tokens: []std.ArrayList(i32),
        out_logps: []std.ArrayList(f32),
    ) !void {
        if (prompt.len == 0) return error.EmptyPrompt;
        if (prompt.len >= seq_len) return error.NoCompletionBudget;
        if (max_completion_tokens == 0) return error.EmptyCompletion;
        if (out_tokens.len == 0 or out_tokens.len != out_logps.len) return error.InvalidCompletionGroup;
        if (rank_cap == 0) return error.InvalidCompletionRankCap;
        for (out_tokens, out_logps) |tokens, logps| {
            if (tokens.items.len != 0 or logps.items.len != 0) return error.NonEmptyCompletionOutput;
        }

        errdefer self.cancelFrameAfterError();
        const group_size = out_tokens.len;
        const states = try self.allocator.alloc(generation.NativeDecodeState, group_size);
        defer self.allocator.free(states);
        var initialized_states: usize = 0;
        defer for (states[0..initialized_states]) |*state| state.deinit();
        for (states) |*state| {
            state.* = generation.NativeDecodeState.initPaged(self.allocator, &self.kv_manager, self.pool_id, null);
            state.kv_storage = &self.kv_storage;
            state.configureForGptConfig(self.config);
            initialized_states += 1;
        }

        const prompt_ids = try self.allocator.alloc(i64, prompt.len);
        defer self.allocator.free(prompt_ids);
        for (prompt, 0..) |token_id, index| prompt_ids[index] = token_id;

        const page_tokens = self.telemetry.cache_page_tokens;
        const shared_prompt_tokens = prompt.len - (prompt.len % page_tokens);
        const prompt_tail = prompt_ids[shared_prompt_tokens..];
        const segmented_clone_source = self.clone_prompt_tails and
            shared_prompt_tokens > 0 and
            prompt_tail.len > 0 and
            states.len > 1;

        const first_logits = if (segmented_clone_source) blk: {
            // Build the canonical state in the same aligned-prefix + segmented
            // tail shape used by private candidates. This produces both the
            // exact first-token logits and one qualified tail representation,
            // avoiding a redundant full-prompt tail projection.
            try states[0].notePrefill(shared_prompt_tokens);
            var prefix_context = self.decodeContext(
                &states[0],
                shared_prompt_tokens,
                shared_prompt_tokens,
            );
            try gpt_arch.forwardHiddenOnlyWithCudaReplay(
                self.trainer.compute_backend,
                self.allocator,
                self.config,
                prompt_ids[0..shared_prompt_tokens],
                1,
                shared_prompt_tokens,
                &prefix_context,
                "gemma4.grpo_kv_prompt_aligned_prefix",
            );
            try self.drainFrame();
            self.telemetry.prompt_prefill_forwards += 1;

            try states[0].appendPrefillChunk(prompt_tail.len);
            var tail_context = self.decodeContext(&states[0], prompt.len, prompt_tail.len);
            const logits = try gpt_arch.forwardLastLogitsTensorWithCudaReplay(
                self.trainer.compute_backend,
                self.allocator,
                self.config,
                prompt_tail,
                1,
                prompt.len,
                &tail_context,
                "gemma4.grpo_kv_prompt_segmented_tail",
            );
            try self.drainFrame();
            self.telemetry.prompt_tail_prefill_forwards += 1;
            self.telemetry.prompt_tail_prefill_candidates += 1;
            self.telemetry.max_prompt_tail_batch_size = @max(self.telemetry.max_prompt_tail_batch_size, 1);
            break :blk logits;
        } else blk: {
            try states[0].notePrefill(prompt.len);
            var prompt_context = self.decodeContext(&states[0], prompt.len, prompt.len);
            const logits = try gpt_arch.forwardLastLogitsTensorWithCudaReplay(
                self.trainer.compute_backend,
                self.allocator,
                self.config,
                prompt_ids,
                1,
                prompt.len,
                &prompt_context,
                "gemma4.grpo_kv_prompt",
            );
            self.telemetry.prompt_prefill_forwards += 1;
            break :blk logits;
        };
        defer self.trainer.compute_backend.free(first_logits);

        if (self.clone_prompt_tails and !segmented_clone_source and prompt_tail.len > 0 and states.len > 1) {
            // The canonical prompt must be globally visible before the first
            // private candidate replays its segmented tail. That replay is the
            // exact clone source for the remaining private candidates.
            try self.drainFrame();
        }
        for (states[1..]) |*state| {
            if (shared_prompt_tokens > 0) {
                try state.attachSharedPagedPrefix(&states[0], shared_prompt_tokens);
                self.telemetry.shared_prompt_tokens += shared_prompt_tokens;
                self.telemetry.reused_candidate_prompt_tokens += shared_prompt_tokens;
            }
        }
        if (prompt_tail.len > 0 and states.len > 1) {
            if (self.clone_prompt_tails) {
                for (states[1..]) |*state| {
                    if (shared_prompt_tokens == 0) {
                        try state.notePrefill(prompt.len);
                    } else {
                        try state.appendPrefillChunk(prompt_tail.len);
                    }
                }

                var clone_source: *generation.NativeDecodeState = &states[0];
                var first_clone_index: usize = 1;
                if (!segmented_clone_source) {
                    // A tail projected as part of a canonical full-prompt
                    // graph is not bit-equivalent to the qualified segmented
                    // replay. Short/no-prefix prompts retain the proven
                    // replay-one fallback before cloning the rest.
                    var clone_source_context = self.decodeContext(&states[1], prompt.len, prompt_tail.len);
                    try gpt_arch.forwardHiddenOnlyWithCudaReplay(
                        self.trainer.compute_backend,
                        self.allocator,
                        self.config,
                        prompt_tail,
                        1,
                        prompt.len,
                        &clone_source_context,
                        "gemma4.grpo_kv_prompt_tail_clone_source",
                    );
                    try self.drainFrame();
                    self.telemetry.prompt_tail_prefill_forwards += 1;
                    self.telemetry.prompt_tail_prefill_candidates += 1;
                    self.telemetry.max_prompt_tail_batch_size = @max(self.telemetry.max_prompt_tail_batch_size, 1);
                    clone_source = &states[1];
                    first_clone_index = 2;
                }

                if (first_clone_index < states.len) {
                    if (!try self.trainer.compute_backend.decoderRuntimeBeginFrame()) {
                        return error.GrpoPromptTailCloneFrameUnavailable;
                    }
                    errdefer self.cancelFrameAfterError();
                    for (states[first_clone_index..]) |*state| {
                        try state.clonePagedTailFrom(clone_source, prompt_tail.len);
                        self.telemetry.prompt_tail_clone_candidates += 1;
                        self.telemetry.prompt_tail_clone_tokens += prompt_tail.len;
                    }
                    try self.drainFrame();
                }
            } else {
                for (states[1..]) |*state| {
                    if (shared_prompt_tokens == 0) {
                        try state.notePrefill(prompt.len);
                    } else {
                        try state.appendPrefillChunk(prompt_tail.len);
                    }
                    var tail_context = self.decodeContext(state, prompt.len, prompt_tail.len);
                    try gpt_arch.forwardHiddenOnlyWithCudaReplay(
                        self.trainer.compute_backend,
                        self.allocator,
                        self.config,
                        prompt_tail,
                        1,
                        prompt.len,
                        &tail_context,
                        "gemma4.grpo_kv_prompt_tail",
                    );
                    try self.drainFrame();
                    self.telemetry.prompt_tail_prefill_forwards += 1;
                    self.telemetry.prompt_tail_prefill_candidates += 1;
                    self.telemetry.max_prompt_tail_batch_size = @max(self.telemetry.max_prompt_tail_batch_size, 1);
                }
            }
        }

        const active = try self.allocator.alloc(bool, group_size);
        defer self.allocator.free(active);
        @memset(active, true);
        const active_indices = try self.allocator.alloc(usize, group_size);
        defer self.allocator.free(active_indices);
        const active_states = try self.allocator.alloc(*generation.NativeDecodeState, group_size);
        defer self.allocator.free(active_states);
        const decode_inputs = try self.allocator.alloc(i64, group_size);
        defer self.allocator.free(decode_inputs);

        const vocab_size: usize = @intCast(self.config.vocab_size);
        const ranked_count = @min(@min(rank_cap, group_size), vocab_size);
        const ranked_tokens = try self.allocator.alloc(usize, ranked_count);
        defer self.allocator.free(ranked_tokens);
        try self.selectTopRankedTokensResident(first_logits, ranked_tokens);
        for (0..group_size) |completion_idx| {
            const token_id = ranked_tokens[@min(completion_idx % rank_cap, ranked_count - 1)];
            try out_tokens[completion_idx].append(self.allocator, @intCast(token_id));
            try out_logps[completion_idx].append(self.allocator, 0.0);
            if (eos_token_id) |eos_id| {
                if (token_id == @as(usize, @intCast(eos_id))) active[completion_idx] = false;
            }
        }

        var step: usize = 1;
        while (step < max_completion_tokens) : (step += 1) {
            var active_count: usize = 0;
            for (0..group_size) |completion_idx| {
                if (!active[completion_idx]) continue;
                if (prompt.len + step >= seq_len) {
                    active[completion_idx] = false;
                    continue;
                }
                active_indices[active_count] = completion_idx;
                active_states[active_count] = &states[completion_idx];
                decode_inputs[active_count] = out_tokens[completion_idx].items[step - 1];
                active_count += 1;
            }
            if (active_count == 0) break;

            const total_sequence_len = prompt.len + step;
            if (self.batch_active_candidates and active_count > 1) {
                for (active_states[0..active_count]) |state| try state.appendGeneratedToken();
                var owned_context = try generation.buildOwnedBatchDecodeContext(
                    self.allocator,
                    active_states[0..active_count],
                    total_sequence_len,
                    1,
                );
                defer owned_context.deinit();
                self.attachRuntimeAdapter(&owned_context.context);
                const logits = try gpt_arch.forwardLastLogitsTensorWithCudaReplay(
                    self.trainer.compute_backend,
                    self.allocator,
                    self.config,
                    decode_inputs[0..active_count],
                    active_count,
                    total_sequence_len,
                    &owned_context.context,
                    "gemma4.grpo_kv_decode_batch",
                );
                defer self.trainer.compute_backend.free(logits);
                self.telemetry.decode_forwards += 1;
                self.telemetry.decode_forward_candidates += active_count;
                self.telemetry.max_decode_batch_size = @max(self.telemetry.max_decode_batch_size, active_count);
                for (active_indices[0..active_count], 0..) |completion_idx, row_index| {
                    const token_id = try self.selectRankedTokenResident(logits, row_index, completion_idx % rank_cap);
                    try out_tokens[completion_idx].append(self.allocator, @intCast(token_id));
                    try out_logps[completion_idx].append(self.allocator, 0.0);
                    if (eos_token_id) |eos_id| {
                        if (token_id == @as(usize, @intCast(eos_id))) active[completion_idx] = false;
                    }
                }
            } else {
                for (active_indices[0..active_count]) |completion_idx| {
                    try states[completion_idx].appendGeneratedToken();
                    const decode_input = [_]i64{out_tokens[completion_idx].items[step - 1]};
                    var decode_context = self.decodeContext(&states[completion_idx], total_sequence_len, 1);
                    const logits = try gpt_arch.forwardLastLogitsTensorWithCudaReplay(
                        self.trainer.compute_backend,
                        self.allocator,
                        self.config,
                        &decode_input,
                        1,
                        total_sequence_len,
                        &decode_context,
                        "gemma4.grpo_kv_decode",
                    );
                    defer self.trainer.compute_backend.free(logits);
                    self.telemetry.decode_forwards += 1;
                    self.telemetry.decode_forward_candidates += 1;
                    self.telemetry.max_decode_batch_size = @max(self.telemetry.max_decode_batch_size, 1);
                    const token_id = try self.selectRankedTokenResident(logits, 0, completion_idx % rank_cap);
                    try out_tokens[completion_idx].append(self.allocator, @intCast(token_id));
                    try out_logps[completion_idx].append(self.allocator, 0.0);
                    if (eos_token_id) |eos_id| {
                        if (token_id == @as(usize, @intCast(eos_id))) active[completion_idx] = false;
                    }
                }
            }
        }

        try self.drainFrame();
        self.telemetry.groups += 1;
        for (out_tokens) |tokens| {
            if (tokens.items.len == 0) return error.EmptyCompletion;
        }
    }
};

/// Samples a ranked completion group while sharing the prompt-only forward pass.
/// Every completion uses `completion_idx % rank_cap` at each decoding step, which
/// preserves the deterministic diversity contract of `sampleCompletionRanked`.
/// Once prefixes diverge, subsequent decoding steps execute independently.
pub fn sampleCompletionGroupRanked(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    prompt: []const i32,
    seq_len: u32,
    max_completion_tokens: usize,
    rank_cap: usize,
    eos_token_id: ?i32,
    sparse_multi_token_projection: bool,
    out_tokens: []std.ArrayList(i32),
    out_logps: []std.ArrayList(f32),
) !void {
    if (prompt.len == 0) return error.EmptyPrompt;
    if (prompt.len >= seq_len) return error.NoCompletionBudget;
    if (max_completion_tokens == 0) return error.EmptyCompletion;
    if (out_tokens.len == 0 or out_tokens.len != out_logps.len) return error.InvalidCompletionGroup;
    if (rank_cap == 0) return error.InvalidCompletionRankCap;
    for (out_tokens, out_logps) |tokens, logps| {
        if (tokens.items.len != 0 or logps.items.len != 0) return error.NonEmptyCompletionOutput;
    }

    const group_size = out_tokens.len;
    const sequences = try allocator.alloc(std.ArrayList(i32), group_size);
    defer allocator.free(sequences);
    var initialized_sequences: usize = 0;
    defer for (sequences[0..initialized_sequences]) |*seq| seq.deinit(allocator);
    for (sequences) |*seq| {
        seq.* = .empty;
        initialized_sequences += 1;
        try seq.appendSlice(allocator, prompt);
    }

    const active = try allocator.alloc(bool, group_size);
    defer allocator.free(active);
    @memset(active, true);

    var step: usize = 0;
    while (step < max_completion_tokens) : (step += 1) {
        if (step == 0) {
            const sparse_row_projection = max_completion_tokens == 1 or sparse_multi_token_projection;
            const logits = if (sparse_row_projection)
                try executeSparseLogitsForInputIds(allocator, trainer, ctx, prompt, seq_len, null)
            else
                try executeLogitsForInputIds(allocator, trainer, ctx, prompt, seq_len, null);
            defer allocator.free(logits);
            const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
            const row = if (sparse_row_projection)
                logits[0..vocab_size]
            else
                logits[(prompt.len - 1) * vocab_size ..][0..vocab_size];
            const ranked_count = @min(@min(rank_cap, group_size), vocab_size);
            const ranked_tokens = try allocator.alloc(usize, ranked_count);
            defer allocator.free(ranked_tokens);
            try selectTopRankedTokens(allocator, row, ranked_tokens);
            const log_z = logNormalizer(row);
            for (0..group_size) |completion_idx| {
                const token_id = ranked_tokens[@min(completion_idx % rank_cap, ranked_count - 1)];
                const token_logp = logProbAtTokenWithNormalizer(row, token_id, log_z);
                try out_tokens[completion_idx].append(allocator, @intCast(token_id));
                try out_logps[completion_idx].append(allocator, token_logp);
                try sequences[completion_idx].append(allocator, @intCast(token_id));
                if (eos_token_id) |eos_id| {
                    if (token_id == @as(usize, @intCast(eos_id))) active[completion_idx] = false;
                }
            }
            continue;
        }

        var any_active = false;
        for (0..group_size) |completion_idx| {
            if (!active[completion_idx]) continue;
            const seq = &sequences[completion_idx];
            if (seq.items.len >= seq_len) {
                active[completion_idx] = false;
                continue;
            }
            any_active = true;
            const logits = if (sparse_multi_token_projection)
                try executeSparseLogitsForInputIds(allocator, trainer, ctx, seq.items, seq_len, null)
            else
                try executeLogitsForInputIds(allocator, trainer, ctx, seq.items, seq_len, null);
            defer allocator.free(logits);
            const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
            const row = if (sparse_multi_token_projection)
                logits[0..vocab_size]
            else
                logits[(seq.items.len - 1) * vocab_size ..][0..vocab_size];
            const token_id = try selectRankedToken(allocator, row, completion_idx % rank_cap);
            const token_logp = logProbAtToken(row, token_id);
            try out_tokens[completion_idx].append(allocator, @intCast(token_id));
            try out_logps[completion_idx].append(allocator, token_logp);
            try seq.append(allocator, @intCast(token_id));
            if (eos_token_id) |eos_id| {
                if (token_id == @as(usize, @intCast(eos_id))) active[completion_idx] = false;
            }
        }
        if (!any_active) break;
    }

    for (out_tokens) |tokens| {
        if (tokens.items.len == 0) return error.EmptyCompletion;
    }
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
    const expected_columns = std.math.add(usize, 1, std.math.mul(usize, 2, top_k) catch return error.InvalidTeacherDistillationTargets) catch
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
        for (0..top_k) |ki| {
            targets[target_base + 1 + ki] = @floatFromInt(example.teacher_top_k_token_ids[teacher_base + ki]);
            targets[target_base + 1 + top_k + ki] = distillation_scale * (example.teacher_top_k_probs[teacher_base + ki] / prob_sum);
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
    if (prepared.examples_with_images > 0 or prepared.examples_with_audio > 0) return error.MultimodalTeacherMaterializationNotYetSupported;
    if (options.top_k == 0) return error.InvalidTeacherTopK;
    if (!std.math.isFinite(options.temperature) or options.temperature <= 0) return error.InvalidTeacherTemperature;

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
    return (try runPreparedExamplesRange(allocator, trainer, ctx, examples, .{
        .max_examples = max_examples,
        .seq_len = seq_len,
        .flush_at_end = true,
    }, .train)).metrics;
}

pub fn evaluatePreparedExamples(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    examples: []const gemma4.PreparedExampleInput,
    max_examples: usize,
    seq_len: u32,
) !CausalLmMetrics {
    return (try runPreparedExamplesRange(allocator, trainer, ctx, examples, .{
        .max_examples = max_examples,
        .seq_len = seq_len,
    }, .eval)).metrics;
}

pub const PreparedExamplesRunOptions = struct {
    start_example_index: usize = 0,
    max_examples: usize = 0,
    seq_len: u32,
    /// Independently round each causal row to a bounded shape. Examples are
    /// never concatenated or reordered, so attention isolation, RoPE origins,
    /// and optimizer order remain identical to fixed-padding training.
    sequence_length_buckets: ?SequenceLengthBuckets = null,
    /// Leave an incomplete accumulation window resident so it can be paired
    /// atomically with `saveTrainingCheckpoint`. Epoch-boundary callers keep
    /// the default and flush the final partial window.
    flush_at_end: bool = true,
    /// Benchmark-only observer. Both fields must be supplied together. The
    /// callback runs once for every admitted training microbatch, after an
    /// explicit strict-Metal synchronization when that microbatch closes an
    /// optimizer window. Ordinary training leaves both fields null.
    benchmark_observer_context: ?*anyopaque = null,
    benchmark_observer: ?PreparedMicrostepObserver = null,
};

pub const SequenceLengthBuckets = struct {
    quantum: u32,
    /// Zero selects one quantum. Values larger than the run maximum are
    /// clamped to that maximum, preserving a valid fixed-shape rollback.
    minimum: u32 = 0,
};

/// Select a shape for one independent causal row. The returned length always
/// contains every logical input token and never exceeds `max_seq_len`, even
/// when that maximum is not itself a multiple of the requested quantum.
pub fn sequenceLengthForExample(
    required_tokens: usize,
    max_seq_len: u32,
    buckets: ?SequenceLengthBuckets,
) !u32 {
    if (max_seq_len == 0) return error.InvalidSequenceLength;
    const required = std.math.cast(u32, required_tokens) orelse return error.SequenceTooLong;
    if (required > max_seq_len) return error.PreparedExampleExceedsSequenceLength;
    const policy = buckets orelse return max_seq_len;
    if (policy.quantum == 0) return error.InvalidSequenceLengthBucketQuantum;

    const minimum = if (policy.minimum == 0) policy.quantum else policy.minimum;
    const desired = @max(required, @min(minimum, max_seq_len));
    const rounded = std.math.mul(
        u32,
        try std.math.divCeil(u32, desired, policy.quantum),
        policy.quantum,
    ) catch return error.SequenceTooLong;
    return @min(rounded, max_seq_len);
}

pub const PreparedMicrostepObservation = struct {
    step: real_autodiff.StepResult,
    started_ns: u64,
    finished_ns: u64,
    input_tokens: usize,
    supervised_tokens: usize,
    explicit_device_sync: bool,
};

pub const PreparedMicrostepObserver = *const fn (
    context: *anyopaque,
    observation: PreparedMicrostepObservation,
) anyerror!void;

pub const PreparedExamplesRunSummary = struct {
    metrics: CausalLmMetrics,
    next_example_index: usize,
    exhausted: bool,
};

pub fn trainPreparedExamplesRange(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    examples: []const gemma4.PreparedExampleInput,
    options: PreparedExamplesRunOptions,
) !PreparedExamplesRunSummary {
    return runPreparedExamplesRange(allocator, trainer, ctx, examples, options, .train);
}

pub fn evaluatePreparedExamplesRange(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    examples: []const gemma4.PreparedExampleInput,
    options: PreparedExamplesRunOptions,
) !PreparedExamplesRunSummary {
    return runPreparedExamplesRange(allocator, trainer, ctx, examples, options, .eval);
}

const PreparedExamplesRunMode = enum { train, eval };

fn runPreparedExamplesRange(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    examples: []const gemma4.PreparedExampleInput,
    options: PreparedExamplesRunOptions,
    mode: PreparedExamplesRunMode,
) !PreparedExamplesRunSummary {
    try validatePreparedObserverOptions(options, mode);
    if (options.start_example_index > examples.len) return error.PreparedExampleCursorOutOfRange;
    var metrics = CausalLmMetrics{};
    const remaining = examples.len - options.start_example_index;
    const count = if (options.max_examples > 0) @min(options.max_examples, remaining) else remaining;
    const end_example_index = options.start_example_index + count;
    var total_weighted_loss: f64 = 0;
    var total_weighted_grad_norm: f64 = 0;
    var total_weighted_teacher_temperature: f64 = 0;

    for (examples[options.start_example_index..end_example_index], options.start_example_index..) |*example, example_index| {
        const effective_supervised_tokens = try validatePreparedExample(
            example,
            @intCast(options.seq_len),
            @intCast(ctx.graph_config.vocab_size),
        );
        if (effective_supervised_tokens == 0) {
            if (mode == .train) advanceTrainingProgress(trainer, example_index + 1);
            continue;
        }
        const scheduled_seq_len = try sequenceLengthForExample(
            example.num_input_tokens,
            options.seq_len,
            options.sequence_length_buckets,
        );
        var input = try makeTrainerInputForExample(allocator, ctx, example, scheduled_seq_len);
        defer input.deinit(allocator);
        const step_started_ns = monotonicNowNs();
        const step = switch (mode) {
            .train => try trainer.step(input.trainer_input),
            .eval => try trainer.evaluate(input.trainer_input),
        };
        var explicit_device_sync = false;
        if (mode == .train and options.benchmark_observer != null and step.optimizer_stepped) {
            try trainer.synchronizeMetalForBenchmark();
            explicit_device_sync = true;
        }
        const step_finished_ns = monotonicNowNs();
        if (options.benchmark_observer) |observe| {
            try observe(options.benchmark_observer_context.?, .{
                .step = step,
                .started_ns = step_started_ns,
                .finished_ns = step_finished_ns,
                .input_tokens = input.trainer_input.input_ids.len,
                .supervised_tokens = input.supervised_tokens,
                .explicit_device_sync = explicit_device_sync,
            });
        }
        const weight: f64 = @floatFromInt(input.supervised_tokens);
        metrics.examples_seen += 1;
        metrics.supervised_tokens_seen += input.supervised_tokens;
        metrics.logical_input_tokens_seen +|= example.num_input_tokens;
        metrics.scheduled_input_tokens_seen +|= scheduled_seq_len;
        metrics.fixed_shape_input_tokens_seen +|= options.seq_len;
        metrics.padding_tokens_avoided +|= options.seq_len - scheduled_seq_len;
        metrics.length_bucketed_examples += @intFromBool(scheduled_seq_len != options.seq_len);
        if (metrics.min_scheduled_sequence_length == 0) {
            metrics.min_scheduled_sequence_length = scheduled_seq_len;
        } else {
            metrics.min_scheduled_sequence_length = @min(metrics.min_scheduled_sequence_length, scheduled_seq_len);
        }
        metrics.max_scheduled_sequence_length = @max(metrics.max_scheduled_sequence_length, scheduled_seq_len);
        if (exampleHasTeacherTargets(example)) {
            metrics.teacher_examples_seen += 1;
            metrics.teacher_supervised_tokens_seen += input.supervised_tokens;
            total_weighted_teacher_temperature += @as(f64, example.teacher_temperature) * weight;
        }
        total_weighted_loss += @as(f64, step.loss) * weight;
        total_weighted_grad_norm += @as(f64, step.grad_norm) * weight;
        recordStepExecutionEvidence(&metrics, step);
        if (step.optimizer_stepped) metrics.optimizer_steps += 1;
        if (mode == .train) advanceTrainingProgress(trainer, example_index + 1);
    }

    // Do not discard the final partial accumulation window at an epoch
    // boundary. Besides preserving every example's gradient, this leaves the
    // trainer in the only state that can be checkpointed safely.
    if (mode == .train and options.flush_at_end) {
        if (options.benchmark_observer != null and trainer.accumulatedMicroBatches() != 0) {
            return error.BenchmarkIncompleteOptimizerWindow;
        }
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
    return .{
        .metrics = metrics,
        .next_example_index = end_example_index,
        .exhausted = end_example_index == examples.len,
    };
}

fn validatePreparedObserverOptions(options: PreparedExamplesRunOptions, mode: PreparedExamplesRunMode) !void {
    if ((options.benchmark_observer == null) != (options.benchmark_observer_context == null)) {
        return error.InvalidBenchmarkObserver;
    }
    if (options.benchmark_observer != null and mode != .train) {
        return error.BenchmarkObserverRequiresTraining;
    }
}

fn monotonicNowNs() u64 {
    var ts: std.posix.timespec = undefined;
    switch (std.posix.errno(std.posix.system.clock_gettime(.MONOTONIC, &ts))) {
        .SUCCESS => return @intCast(@as(i128, ts.sec) * std.time.ns_per_s + ts.nsec),
        else => return 0,
    }
}

test "gemma4 prepared microstep observer admission is paired and training-only" {
    const noop = struct {
        fn observe(_: *anyopaque, _: PreparedMicrostepObservation) anyerror!void {}
    }.observe;
    var context: u8 = 0;
    const base = PreparedExamplesRunOptions{ .seq_len = 8 };
    try validatePreparedObserverOptions(base, .train);
    var missing_context = base;
    missing_context.benchmark_observer = noop;
    try std.testing.expectError(error.InvalidBenchmarkObserver, validatePreparedObserverOptions(missing_context, .train));
    var admitted = missing_context;
    admitted.benchmark_observer_context = &context;
    try validatePreparedObserverOptions(admitted, .train);
    try std.testing.expectError(error.BenchmarkObserverRequiresTraining, validatePreparedObserverOptions(admitted, .eval));
}

test "gemma4 independent row length buckets are bounded and contain every token" {
    const buckets = SequenceLengthBuckets{ .quantum = 16, .minimum = 32 };
    try std.testing.expectEqual(@as(u32, 32), try sequenceLengthForExample(1, 128, buckets));
    try std.testing.expectEqual(@as(u32, 32), try sequenceLengthForExample(32, 128, buckets));
    try std.testing.expectEqual(@as(u32, 48), try sequenceLengthForExample(33, 128, buckets));
    try std.testing.expectEqual(@as(u32, 128), try sequenceLengthForExample(127, 128, buckets));
    try std.testing.expectEqual(@as(u32, 125), try sequenceLengthForExample(124, 125, buckets));
    try std.testing.expectEqual(@as(u32, 128), try sequenceLengthForExample(33, 128, null));
    try std.testing.expectError(
        error.PreparedExampleExceedsSequenceLength,
        sequenceLengthForExample(129, 128, buckets),
    );
    try std.testing.expectError(
        error.InvalidSequenceLengthBucketQuantum,
        sequenceLengthForExample(1, 128, .{ .quantum = 0 }),
    );
}

fn advanceTrainingProgress(trainer: *real_autodiff.RealAutodiffTrainer, next_example_index: usize) void {
    var progress = trainer.trainingProgress();
    progress.next_example_index = @intCast(next_example_index);
    progress.order_cursor +|= 1;
    progress.examples_seen +|= 1;
    trainer.setTrainingProgress(progress);
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
        .target_preset = adapter_inspect.target_preset,
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

fn putOwnedLogitsRuntimeInput(
    allocator: std.mem.Allocator,
    compute_backend: *const ComputeBackend,
    runtime_inputs: *std.AutoHashMapUnmanaged(NodeId, CT),
    owned_values: *std.ArrayList(CT),
    node_id: NodeId,
    value: CT,
) !void {
    errdefer compute_backend.free(value);
    try runtime_inputs.put(allocator, node_id, value);
    try owned_values.append(allocator, value);
}

/// Reusable device tensors that bind every LoRA parameter to zero while the
/// base-model graph and weights remain shared with the live policy trainer.
/// This gives preference objectives an exact frozen-base scorer without a
/// second model allocation or temporary mutation of optimizer-owned weights.
pub const FrozenBaseLoraBindings = struct {
    allocator: std.mem.Allocator,
    compute_backend: *const ComputeBackend,
    values: []CT,

    pub fn init(
        allocator: std.mem.Allocator,
        trainer: *real_autodiff.RealAutodiffTrainer,
    ) !FrozenBaseLoraBindings {
        const values = try allocator.alloc(CT, trainer.lora_params.items.len);
        var initialized: usize = 0;
        errdefer {
            for (values[0..initialized]) |value| trainer.compute_backend.free(value);
            allocator.free(values);
        }
        for (trainer.lora_params.items, 0..) |slot, idx| {
            const zeros = try allocator.alloc(f32, slot.weights.len);
            defer allocator.free(zeros);
            @memset(zeros, 0.0);
            values[idx] = try trainer.compute_backend.fromFloat32Shape(zeros, slot.dims);
            initialized += 1;
        }
        return .{
            .allocator = allocator,
            .compute_backend = trainer.compute_backend,
            .values = values,
        };
    }

    pub fn deinit(self: *FrozenBaseLoraBindings) void {
        for (self.values) |value| self.compute_backend.free(value);
        self.allocator.free(self.values);
        self.* = undefined;
    }

    pub fn debugMaxAbsPrefix(self: *const FrozenBaseLoraBindings, max_bindings: usize) !f32 {
        var max_abs: f32 = 0.0;
        for (self.values[0..@min(self.values.len, max_bindings)]) |value| {
            const data = try self.compute_backend.toFloat32(value, self.allocator);
            defer self.allocator.free(data);
            for (data) |item| max_abs = @max(max_abs, @abs(item));
        }
        return max_abs;
    }
};

fn bindLogitsTrainableSlots(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    slots: []const real_autodiff.RealAutodiffTrainer.ParamSlot,
    runtime_inputs: *std.AutoHashMapUnmanaged(NodeId, CT),
    owned_values: *std.ArrayList(CT),
) !void {
    for (slots) |slot| {
        const resident_weight = if (slot.device) |device|
            device.weight
        else
            slot.eval_device_weight;
        if (resident_weight) |weight| {
            try runtime_inputs.put(allocator, slot.node_id, weight);
        } else {
            const weight = try trainer.compute_backend.fromFloat32Shape(slot.weights, slot.dims);
            try putOwnedLogitsRuntimeInput(
                allocator,
                trainer.compute_backend,
                runtime_inputs,
                owned_values,
                slot.node_id,
                weight,
            );
        }
    }
}

fn bindTrainerLogitsTrainables(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    runtime_inputs: *std.AutoHashMapUnmanaged(NodeId, CT),
    owned_values: *std.ArrayList(CT),
    frozen_lora: ?*const FrozenBaseLoraBindings,
) !void {
    if (frozen_lora) |bindings| {
        if (bindings.values.len != trainer.lora_params.items.len) return error.FrozenLoraBindingMismatch;
        for (trainer.lora_params.items, bindings.values) |slot, value| {
            try runtime_inputs.put(allocator, slot.node_id, value);
        }
    } else {
        try bindLogitsTrainableSlots(allocator, trainer, trainer.lora_params.items, runtime_inputs, owned_values);
    }
    try bindLogitsTrainableSlots(allocator, trainer, trainer.regular_params.items, runtime_inputs, owned_values);
}

fn executeOwnedTrainerGraphOutput(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    owned: *const OwnedTrainerInput,
    output_node: NodeId,
    frozen_lora: ?*const FrozenBaseLoraBindings,
) ![]f32 {
    var gs = &trainer.graph_state.?;
    var rt = std.AutoHashMapUnmanaged(NodeId, CT).empty;
    defer rt.deinit(allocator);
    var owned_runtime_values = std.ArrayList(CT).empty;
    defer {
        for (owned_runtime_values.items) |value| trainer.compute_backend.free(value);
        owned_runtime_values.deinit(allocator);
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

    const input_ct = try graph_input_binder.bindI64(trainer.compute_backend, allocator, input_placeholder, owned.input_ids);
    try putOwnedLogitsRuntimeInput(allocator, trainer.compute_backend, &rt, &owned_runtime_values, gs.input_ids_node, input_ct);
    const mask_ct = try graph_input_binder.bindF32(trainer.compute_backend, allocator, mask_placeholder, owned.attention_mask);
    try putOwnedLogitsRuntimeInput(allocator, trainer.compute_backend, &rt, &owned_runtime_values, gs.attention_mask_node, mask_ct);
    const targets_ct = try graph_input_binder.bindF32(trainer.compute_backend, allocator, targets_placeholder, owned.targets);
    try putOwnedLogitsRuntimeInput(allocator, trainer.compute_backend, &rt, &owned_runtime_values, gs.targets_node, targets_ct);
    try bindTrainerLogitsTrainables(allocator, trainer, &rt, &owned_runtime_values, frozen_lora);

    // Sequence-logprob scoring asks for the trainer loss node. Execute it
    // through the same cached, pruned forward graph used by live-policy
    // evaluation so reference and policy scoring have identical forward
    // semantics. The runtime map is keyed by source-graph NodeIds;
    // CompiledTrainSession remaps those IDs into its loss-only clone while
    // preserving the explicit zero-LoRA bindings supplied above.
    //
    // This is especially important for fused ops with a training-forward
    // alternate: executing the source graph directly would run the fused
    // inference kernel for the frozen reference while autodiff runs the
    // proven decomposed forward for the policy.
    if (output_node == gs.loss_node) {
        _ = try trainer.ensureCompiledEvalSessionBuilt();
        var result = try trainer.compiled_eval_session.?.execute(trainer.compute_backend, rt);
        defer result.deinit();
        const values = try allocator.alloc(f32, 1);
        values[0] = result.loss;
        return values;
    }

    const saved_outputs = try allocator.dupe(NodeId, gs.graph.outputs.items);
    defer {
        gs.graph.outputs.clearRetainingCapacity();
        for (saved_outputs) |node_id| gs.graph.outputs.append(allocator, node_id) catch {};
        allocator.free(saved_outputs);
    }
    gs.graph.outputs.clearRetainingCapacity();
    try gs.graph.markOutput(output_node);

    var rt_inputs = std.ArrayList(interpreter.RuntimeInput).empty;
    defer rt_inputs.deinit(allocator);
    var it = rt.iterator();
    while (it.next()) |entry| {
        try rt_inputs.append(allocator, .{
            .node_id = entry.key_ptr.*,
            .value = entry.value_ptr.*,
        });
    }

    if (trainer.compute_backend.kind() == .metal) {
        try trainer.compute_backend.decoderRuntimeSubmitAndWaitFrame();
    }
    var planned_encoder_coalescing_suspended = false;
    if (trainer.compute_backend.kind() == .metal) {
        planned_encoder_coalescing_suspended = try trainer.compute_backend.decoderRuntimePushPlannedEncoderCoalescingSuppression();
        if (!planned_encoder_coalescing_suspended) return error.PlannedEncoderCoalescingSuppressionUnavailable;
    }
    defer if (planned_encoder_coalescing_suspended) {
        if (trainer.compute_backend.decoderRuntimeHasActiveFrame()) {
            trainer.compute_backend.decoderRuntimeCancelFrame() catch {};
        }
        trainer.compute_backend.decoderRuntimePopPlannedEncoderCoalescingSuppression() catch {};
    };
    errdefer if (trainer.compute_backend.decoderRuntimeHasActiveFrame()) {
        trainer.compute_backend.decoderRuntimeCancelFrame() catch {};
    };
    var exec_result = try interpreter.execute(allocator, &gs.graph, trainer.compute_backend, .{
        .runtime_inputs = rt_inputs.items,
    });
    defer exec_result.deinit(trainer.compute_backend);
    if (trainer.compute_backend.kind() == .metal) {
        try trainer.compute_backend.decoderRuntimeSubmitAndWaitFrame();
        if (comptime build_options.enable_metal) {
            try metal_compute_mod.MetalCompute.syncOutputTensor(trainer.compute_backend, exec_result.outputs[0]);
        }
        try trainer.compute_backend.decoderRuntimePopPlannedEncoderCoalescingSuppression();
        planned_encoder_coalescing_suspended = false;
    }
    return trainer.compute_backend.toFloat32(exec_result.outputs[0], allocator);
}

pub fn sequenceLogprobForExample(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
) !f32 {
    return sequenceLogprobForExampleWithBindings(allocator, trainer, ctx, example, seq_len, null, null);
}

pub fn sequenceLogprobForExampleScheduled(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    weighted_target_rows: usize,
) !f32 {
    return sequenceLogprobForExampleWithBindings(
        allocator,
        trainer,
        ctx,
        example,
        seq_len,
        null,
        weighted_target_rows,
    );
}

pub fn sequenceLogprobForExampleFrozenBase(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    frozen_lora: *const FrozenBaseLoraBindings,
) !f32 {
    return sequenceLogprobForExampleWithBindings(allocator, trainer, ctx, example, seq_len, frozen_lora, null);
}

pub fn sequenceLogprobForExampleFrozenBaseScheduled(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    weighted_target_rows: usize,
    frozen_lora: *const FrozenBaseLoraBindings,
) !f32 {
    return sequenceLogprobForExampleWithBindings(
        allocator,
        trainer,
        ctx,
        example,
        seq_len,
        frozen_lora,
        weighted_target_rows,
    );
}

fn sequenceLogprobForExampleWithBindings(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    example: *const gemma4.PreparedExampleInput,
    seq_len: u32,
    frozen_lora: ?*const FrozenBaseLoraBindings,
    weighted_target_rows: ?usize,
) !f32 {
    // DPO uses one coefficient for every completion token. The uniform sparse
    // target routes scoring through frozen-head fused linear CE, producing the
    // exact summed sequence log-probability without materializing logits.
    var owned = if (weighted_target_rows) |target_rows|
        try makeTrainerInputForLogprobCoeffScheduled(allocator, ctx, example, seq_len, target_rows, 1.0)
    else
        try makeTrainerInputForLogprobCoeff(allocator, ctx, example, seq_len, 1.0);
    defer owned.deinit(allocator);
    if (frozen_lora == null) {
        // The live policy uses trainer-owned resident LoRA weights, so it can
        // execute through the cached, pruned loss-only compiled session. The
        // reference scorer still supplies explicit zero-LoRA bindings below.
        // Keeping that uncommon precompute route separate avoids mutating
        // optimizer-owned device weights while removing two eager forward
        // traversals from every DPO update.
        const result = try trainer.evaluate(owned.trainer_input);
        if (!std.math.isFinite(result.loss)) return error.InvalidSequenceLogprob;
        return result.loss;
    }
    try trainer.ensureGraphBuilt(owned.trainer_input);
    const loss_values = try executeOwnedTrainerGraphOutput(allocator, trainer, &owned, trainer.graph_state.?.loss_node, frozen_lora);
    defer allocator.free(loss_values);
    if (loss_values.len != 1 or !std.math.isFinite(loss_values[0])) return error.InvalidSequenceLogprob;
    return loss_values[0];
}

fn executeLogitsForInputIds(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    raw_input_ids: []const i32,
    seq_len: u32,
    frozen_lora: ?*const FrozenBaseLoraBindings,
) ![]f32 {
    return executeLogitsForInputIdsConfigured(
        allocator,
        trainer,
        ctx,
        raw_input_ids,
        seq_len,
        null,
        frozen_lora,
    );
}

fn executeSparseLogitsForInputIds(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    raw_input_ids: []const i32,
    seq_len: u32,
    frozen_lora: ?*const FrozenBaseLoraBindings,
) ![]f32 {
    if (raw_input_ids.len == 0) return error.EmptyPrompt;
    const predictor_rows = [_]usize{raw_input_ids.len - 1};
    return executeSparseLogitsForInputIdsAtRows(
        allocator,
        trainer,
        ctx,
        raw_input_ids,
        seq_len,
        &predictor_rows,
        frozen_lora,
    );
}

fn executeSparseLogitsForInputIdsAtRows(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    raw_input_ids: []const i32,
    seq_len: u32,
    predictor_rows: []const usize,
    frozen_lora: ?*const FrozenBaseLoraBindings,
) ![]f32 {
    return executeLogitsForInputIdsConfigured(
        allocator,
        trainer,
        ctx,
        raw_input_ids,
        seq_len,
        predictor_rows,
        frozen_lora,
    );
}

fn executeLogitsForInputIdsConfigured(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    raw_input_ids: []const i32,
    seq_len: u32,
    selected_predictor_rows: ?[]const usize,
    frozen_lora: ?*const FrozenBaseLoraBindings,
) ![]f32 {
    const batches = [_][]const i32{raw_input_ids};
    return executeLogitsForInputIdBatchesConfigured(
        allocator,
        trainer,
        ctx,
        &batches,
        seq_len,
        selected_predictor_rows,
        frozen_lora,
    );
}

fn executeSparseLogitsForInputIdBatchAtRows(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    raw_input_id_batches: []const []const i32,
    seq_len: u32,
    predictor_rows: []const usize,
    frozen_lora: ?*const FrozenBaseLoraBindings,
) ![]f32 {
    return executeLogitsForInputIdBatchesConfigured(
        allocator,
        trainer,
        ctx,
        raw_input_id_batches,
        seq_len,
        predictor_rows,
        frozen_lora,
    );
}

fn executeLogitsForInputIdBatchesConfigured(
    allocator: std.mem.Allocator,
    trainer: *real_autodiff.RealAutodiffTrainer,
    ctx: *GemmaAutodiffCtx,
    raw_input_id_batches: []const []const i32,
    seq_len: u32,
    selected_predictor_rows: ?[]const usize,
    frozen_lora: ?*const FrozenBaseLoraBindings,
) ![]f32 {
    const rows: usize = @intCast(seq_len);
    if (raw_input_id_batches.len == 0) return error.EmptyPrompt;
    const batch: u32 = std.math.cast(u32, raw_input_id_batches.len) orelse return error.InvalidCompletionGroup;
    const total_rows = std.math.mul(usize, raw_input_id_batches.len, rows) catch return error.SequenceTooLong;
    for (raw_input_id_batches) |raw_input_ids| {
        if (raw_input_ids.len == 0) return error.EmptyPrompt;
        if (raw_input_ids.len > rows) return error.SequenceTooLong;
    }
    if (selected_predictor_rows) |predictor_rows| {
        if (predictor_rows.len == 0 or predictor_rows.len > sparseLossChunkRows()) return error.InvalidLogitsRowSelection;
        for (predictor_rows) |row| {
            if (row >= total_rows) return error.InvalidLogitsRowSelection;
            const batch_idx = row / rows;
            const local_row = row % rows;
            if (local_row >= raw_input_id_batches[batch_idx].len) return error.InvalidLogitsRowSelection;
        }
    }

    const input_ids = try allocator.alloc(i64, total_rows);
    defer allocator.free(input_ids);
    @memset(input_ids, 0);
    const attention_mask = try allocator.alloc(f32, total_rows);
    defer allocator.free(attention_mask);
    @memset(attention_mask, 0.0);
    for (raw_input_id_batches, 0..) |raw_input_ids, batch_idx| {
        const batch_row_base = batch_idx * rows;
        for (raw_input_ids, 0..) |token_id, idx| {
            input_ids[batch_row_base + idx] = token_id;
            attention_mask[batch_row_base + idx] = 1.0;
        }
    }

    const vocab_size: usize = @intCast(ctx.graph_config.vocab_size);
    const target_elements = if (selected_predictor_rows) |predictor_rows|
        predictor_rows.len * weighted_logprob_target_columns
    else
        total_rows * vocab_size;
    const targets = try allocator.alloc(f32, target_elements);
    defer allocator.free(targets);
    @memset(targets, 0.0);
    if (selected_predictor_rows) |predictor_rows| {
        for (predictor_rows, 0..) |predictor_row, idx| {
            targets[idx * weighted_logprob_target_columns] = @floatFromInt(predictor_row);
        }
    }

    const targets_shape = if (selected_predictor_rows) |predictor_rows|
        Shape.init(.f32, &.{
            @as(i64, @intCast(predictor_rows.len)),
            @as(i64, @intCast(weighted_logprob_target_columns)),
        })
    else
        Shape.init(.f32, &.{ @as(i64, @intCast(total_rows)), @as(i64, @intCast(vocab_size)) });

    const trainer_input = real_autodiff.TrainerInput{
        .ctx = @ptrCast(ctx),
        .build_forward = &GemmaAutodiffCtx.buildForward,
        .build_loss = &GemmaAutodiffCtx.buildLoss,
        .input_ids = input_ids,
        .attention_mask = attention_mask,
        .targets = targets,
        .targets_shape = targets_shape,
        .batch = batch,
        .seq_len = seq_len,
        .bind_arch_inputs = null,
        .remap_graph_nodes = &GemmaAutodiffCtx.remapGraphNodes,
        .capture_graph_context = &GemmaAutodiffCtx.captureGraphContext,
        .restore_graph_context = &GemmaAutodiffCtx.restoreGraphContext,
        .deinit_graph_context = &GemmaAutodiffCtx.deinitGraphContext,
    };

    try trainer.ensureGraphBuilt(trainer_input);
    var gs = &trainer.graph_state.?;
    const logits_node = ctx.lm_logits orelse return error.MissingTrainerLogitsNode;

    var rt = std.AutoHashMapUnmanaged(NodeId, CT).empty;
    defer rt.deinit(allocator);
    var owned_runtime_values = std.ArrayList(CT).empty;
    defer {
        for (owned_runtime_values.items) |value| trainer.compute_backend.free(value);
        owned_runtime_values.deinit(allocator);
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
    try putOwnedLogitsRuntimeInput(allocator, trainer.compute_backend, &rt, &owned_runtime_values, gs.input_ids_node, input_ct);
    const mask_ct = try graph_input_binder.bindF32(trainer.compute_backend, allocator, mask_placeholder, attention_mask);
    try putOwnedLogitsRuntimeInput(allocator, trainer.compute_backend, &rt, &owned_runtime_values, gs.attention_mask_node, mask_ct);
    const targets_ct = try graph_input_binder.bindF32(trainer.compute_backend, allocator, targets_placeholder, targets);
    try putOwnedLogitsRuntimeInput(allocator, trainer.compute_backend, &rt, &owned_runtime_values, gs.targets_node, targets_ct);
    try bindTrainerLogitsTrainables(allocator, trainer, &rt, &owned_runtime_values, frozen_lora);

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

    // The sampler executes the forward graph through the eager interpreter,
    // outside the compiled partition executor that normally promotes and
    // synchronizes graph outputs. Close any frame left open by an earlier
    // host read before this execution, then drain this execution before
    // reading its logits. Without these boundaries, a runtime-pooled output
    // buffer can cycle back to the sampler with bytes from an older update.
    if (trainer.compute_backend.kind() == .metal) {
        try trainer.compute_backend.decoderRuntimeSubmitAndWaitFrame();
    }
    var planned_encoder_coalescing_suspended = false;
    if (trainer.compute_backend.kind() == .metal) {
        planned_encoder_coalescing_suspended = try trainer.compute_backend.decoderRuntimePushPlannedEncoderCoalescingSuppression();
        if (!planned_encoder_coalescing_suspended) return error.PlannedEncoderCoalescingSuppressionUnavailable;
    }
    defer if (planned_encoder_coalescing_suspended) {
        if (trainer.compute_backend.decoderRuntimeHasActiveFrame()) {
            trainer.compute_backend.decoderRuntimeCancelFrame() catch {};
        }
        trainer.compute_backend.decoderRuntimePopPlannedEncoderCoalescingSuppression() catch {};
    };
    errdefer if (trainer.compute_backend.decoderRuntimeHasActiveFrame()) {
        trainer.compute_backend.decoderRuntimeCancelFrame() catch {};
    };
    var exec_result = try interpreter.execute(allocator, &gs.graph, trainer.compute_backend, .{
        .runtime_inputs = rt_inputs.items,
    });
    defer exec_result.deinit(trainer.compute_backend);
    if (trainer.compute_backend.kind() == .metal) {
        try trainer.compute_backend.decoderRuntimeSubmitAndWaitFrame();
        if (comptime build_options.enable_metal) {
            try metal_compute_mod.MetalCompute.syncOutputTensor(trainer.compute_backend, exec_result.outputs[0]);
        }
        try trainer.compute_backend.decoderRuntimePopPlannedEncoderCoalescingSuppression();
        planned_encoder_coalescing_suspended = false;
    }
    return trainer.compute_backend.toFloat32(exec_result.outputs[0], allocator);
}

fn concatPromptCompletion(allocator: std.mem.Allocator, prompt: []const i32, completion: []const i32) ![]i32 {
    const out = try allocator.alloc(i32, prompt.len + completion.len);
    @memcpy(out[0..prompt.len], prompt);
    @memcpy(out[prompt.len..], completion);
    return out;
}

fn selectRankedToken(allocator: std.mem.Allocator, logits: []const f32, rank: usize) !usize {
    if (logits.len == 0) return error.EmptyLogits;
    const clamped_rank = @min(rank, logits.len - 1);
    const ranked = try allocator.alloc(usize, clamped_rank + 1);
    defer allocator.free(ranked);
    try selectTopRankedTokens(allocator, logits, ranked);
    return ranked[clamped_rank];
}

fn selectTopRankedTokens(
    allocator: std.mem.Allocator,
    logits: []const f32,
    out_token_ids: []usize,
) !void {
    if (logits.len == 0) return error.EmptyLogits;
    if (out_token_ids.len == 0 or out_token_ids.len > logits.len) return error.InvalidRankedTokenCount;
    const Entry = struct {
        idx: usize,
        value: f32,
    };
    const entries = try allocator.alloc(Entry, out_token_ids.len);
    defer allocator.free(entries);
    var entry_count: usize = 0;
    for (logits, 0..) |value, idx| {
        const candidate = Entry{ .idx = idx, .value = value };
        var insert_at: usize = 0;
        while (insert_at < entry_count) : (insert_at += 1) {
            const current = entries[insert_at];
            if (candidate.value > current.value or
                (candidate.value == current.value and candidate.idx < current.idx)) break;
        }
        if (insert_at >= out_token_ids.len) continue;
        const new_count = @min(entry_count + 1, out_token_ids.len);
        var move_idx = new_count - 1;
        while (move_idx > insert_at) : (move_idx -= 1) {
            entries[move_idx] = entries[move_idx - 1];
        }
        entries[insert_at] = candidate;
        entry_count = new_count;
    }
    for (entries, 0..) |entry, idx| out_token_ids[idx] = entry.idx;
}

fn logProbAtToken(logits: []const f32, token_id: usize) f32 {
    return logProbAtTokenWithNormalizer(logits, token_id, logNormalizer(logits));
}

fn logNormalizer(logits: []const f32) f64 {
    var max_logit = logits[0];
    for (logits[1..]) |value| {
        if (value > max_logit) max_logit = value;
    }
    var sum_exp: f64 = 0.0;
    for (logits) |value| {
        sum_exp += @exp(@as(f64, value - max_logit));
    }
    return @as(f64, max_logit) + @log(sum_exp);
}

fn logProbAtTokenWithNormalizer(logits: []const f32, token_id: usize, log_z: f64) f32 {
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
    target_preset: ?[]const u8 = null,
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
    const adapter_manifest_path = try std.fs.path.join(allocator, &.{ publication.staging_dir, gemma4.adapter_manifest_file_name });
    defer allocator.free(adapter_manifest_path);

    try writeHeaderAndTensorsF32(allocator, adapter_checkpoint_path, tensors);
    const adapter_write_options = gemma4.AdapterConfigWriteOptions{
        .base_model_name_or_path = spec.base_model_name_or_path,
        .base_model_sha256 = spec.base_model_sha256,
        .tokenizer_sha256 = spec.tokenizer_sha256,
        .chat_template_sha256 = spec.chat_template_sha256,
        .rank = spec.rank,
        .alpha = spec.alpha,
        .target_modules = spec.target_modules,
        .target_preset = spec.target_preset,
        .recursive_lora = spec.recursive_config,
    };
    try gemma4.writeAdapterConfigJson(allocator, adapter_config_path, adapter_write_options);
    try gemma4.writeAdapterManifestJson(allocator, adapter_manifest_path, adapter_write_options);
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
        .base_model_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        .tokenizer_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        .chat_template_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
        .rank = 1,
        .alpha = 1,
        .target_modules = &.{"model.layers.0.self_attn.q_proj"},
    };
    var failed = false;
    writeAndPublishGemmaBundle(allocator, out_dir, &tensors, .{
        .base_model_name_or_path = base_spec.base_model_name_or_path,
        .base_model_sha256 = base_spec.base_model_sha256,
        .tokenizer_sha256 = base_spec.tokenizer_sha256,
        .chat_template_sha256 = base_spec.chat_template_sha256,
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

test "gemma4 sparse loss chunk parser enforces bounded positive rows" {
    try std.testing.expectEqual(max_sparse_loss_chunk_rows, default_sparse_loss_chunk_rows);
    try std.testing.expectEqual(@as(?u32, 1), parseSparseLossChunkRows("1"));
    try std.testing.expectEqual(@as(?u32, 8), parseSparseLossChunkRows("8"));
    try std.testing.expectEqual(@as(?u32, 64), parseSparseLossChunkRows("64"));
    try std.testing.expectEqual(@as(?u32, max_sparse_loss_chunk_rows), parseSparseLossChunkRows("512"));
    try std.testing.expectEqual(@as(?u32, null), parseSparseLossChunkRows("0"));
    try std.testing.expectEqual(@as(?u32, null), parseSparseLossChunkRows("513"));
    try std.testing.expectEqual(@as(?u32, null), parseSparseLossChunkRows("invalid"));
}

test "gemma4 sparse logits hard-label CE has a bounded differentiable graph" {
    const allocator = std.testing.allocator;
    var graph = Graph.init(allocator);
    defer graph.deinit();
    var bld = Builder.init(&graph);

    const logits = try bld.parameter("logits", Shape.init(.f32, &.{ 3, 5 }));
    const labels = try bld.parameter("labels", Shape.init(.f32, &.{ 3, 1 }));
    const loss = try sparseHardLabelCrossEntropy(&bld, logits, labels, 3, 5);
    try graph.markOutput(loss);

    var gather_count: usize = 0;
    for (0..graph.nodeCount()) |idx| {
        const node = graph.node(@intCast(idx));
        switch (node.op) {
            .gather => gather_count += 1,
            .constant => try std.testing.expect(
                !(node.output_shape.rank() == 2 and
                    node.output_shape.dim(0) == 3 and
                    node.output_shape.dim(1) == 5),
            ),
            else => {},
        }
    }
    try std.testing.expectEqual(@as(usize, 2), gather_count);

    const logits_values = [_]f32{
        0.2,  -0.3, 1.1,  0.7,  -0.4,
        1.4,  0.1,  -0.8, 0.6,  0.0,
        -0.2, 0.9,  0.3,  -0.5, 1.2,
    };
    const label_values = [_]f32{ 2, 0, 4 };
    const max_error = try ml.graph.grad_check.checkGradients(
        allocator,
        &graph,
        loss,
        &.{logits},
        &.{ &logits_values, &label_values },
        1e-3,
    );
    try std.testing.expect(max_error < 5e-3);
}

test "gemma4 sparse hard-label loss emits bounded fused linear CE chunks" {
    if (std.c.getenv("TERMITE_GEMMA4_DISABLE_BATCHED_SPARSE_LOSS") != null or
        std.c.getenv("TERMITE_GEMMA4_SPARSE_LOSS_CHUNK_ROWS") != null)
    {
        return error.SkipZigTest;
    }

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
    ctx.enable_fused_linear_cross_entropy = true;
    var graph = Graph.init(allocator);
    defer graph.deinit();
    var bld = Builder.init(&graph);
    const hidden = try bld.parameter("hidden", Shape.init(.f32, &.{ 1, 600, 16 }));
    const targets = try bld.parameter("targets", Shape.init(.f32, &.{ 600, sparse_target_columns }));

    _ = try GemmaAutodiffCtx.buildLoss(@ptrCast(&ctx), &bld, hidden, targets);
    var loss_rows: [2]u32 = undefined;
    var loss_count: usize = 0;
    var vocabulary_projection_count: usize = 0;
    for (0..graph.nodeCount()) |idx| {
        switch (graph.node(@intCast(idx)).op) {
            .fused_linear_cross_entropy_loss => |attrs| {
                try std.testing.expect(loss_count < loss_rows.len);
                try std.testing.expect(attrs.frozen_weight);
                try std.testing.expectEqual(@as(u32, 16), attrs.in_dim);
                try std.testing.expectEqual(@as(u32, 32), attrs.vocab_size);
                try std.testing.expectEqual(@as(f32, 30.0), attrs.logit_softcap);
                loss_rows[loss_count] = attrs.rows;
                loss_count += 1;
            },
            .fused_linear_no_bias => |attrs| {
                if (attrs.in_dim != 16 or attrs.out_dim != 32) continue;
                vocabulary_projection_count += 1;
            },
            else => {},
        }
    }
    try std.testing.expectEqual(@as(usize, 2), loss_count);
    try std.testing.expectEqualSlices(u32, &.{ 512, 88 }, loss_rows[0..loss_count]);
    try std.testing.expectEqual(@as(usize, 0), vocabulary_projection_count);
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
    const targets = try bld.parameter("targets", Shape.init(.f32, &.{ 2, 5 }));

    const loss = try GemmaAutodiffCtx.buildLoss(@ptrCast(&ctx), &bld, hidden, targets);
    try std.testing.expectEqual(@as(i64, 1), graph.node(loss).output_shape.numElements().?);
    var teacher_projection_count: usize = 0;
    var hard_label_fused_count: usize = 0;
    for (0..graph.nodeCount()) |idx| {
        const node = graph.node(@intCast(idx));
        const shape = node.output_shape;
        try std.testing.expect(!(shape.rank() == 2 and shape.dim(0) == 4 and shape.dim(1) == 32));
        switch (node.op) {
            .fused_linear_no_bias => |attrs| {
                if (attrs.in_dim == 16 and attrs.out_dim == 32) teacher_projection_count += 1;
            },
            .fused_linear_cross_entropy_loss => hard_label_fused_count += 1,
            else => {},
        }
    }
    // Soft teacher targets retain their existing dense-target chunk path.
    try std.testing.expectEqual(@as(usize, 1), teacher_projection_count);
    try std.testing.expectEqual(@as(usize, 0), hard_label_fused_count);
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

    try std.testing.expectEqual(@as(i64, 64), owned.trainer_input.targets_shape.dim(0));
    try std.testing.expectEqual(weighted_hard_target_columns, owned.trainer_input.targets_shape.dim(1));
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 3.0, -16.0, 0.0 }, owned.targets[0..4]);
    try std.testing.expectEqualSlices(f32, &.{ 2.0, 4.0, 32.0, 0.0 }, owned.targets[4..8]);
    for (owned.targets[8..]) |value| try std.testing.expectEqual(@as(f32, 0.0), value);
}

test "gemma4 sequence logprob coefficient chooses backend-compatible target contract" {
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
    const example = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = input_ids[0..2],
        .response_input_ids = input_ids[2..],
        .num_prompt_tokens = 2,
        .num_response_tokens = 2,
        .input_ids = &input_ids,
        .labels = &labels,
        .num_input_tokens = input_ids.len,
        .num_supervised_tokens = 2,
    };

    var fallback_owned = try makeTrainerInputForLogprobCoeff(allocator, &ctx, &example, 4, 0.25);
    defer fallback_owned.deinit(allocator);

    try std.testing.expectEqual(@as(i64, 64), fallback_owned.trainer_input.targets_shape.dim(0));
    try std.testing.expectEqual(weighted_hard_target_columns, fallback_owned.trainer_input.targets_shape.dim(1));
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 3.0, -16.0, 0.0 }, fallback_owned.targets[0..4]);
    try std.testing.expectEqualSlices(f32, &.{ 2.0, 4.0, -16.0, 0.0 }, fallback_owned.targets[4..8]);

    ctx.enable_fused_linear_cross_entropy = true;
    var owned = try makeTrainerInputForLogprobCoeff(allocator, &ctx, &example, 4, 0.25);
    defer owned.deinit(allocator);

    try std.testing.expectEqual(@as(i64, 64), owned.trainer_input.targets_shape.dim(0));
    try std.testing.expectEqual(uniform_weighted_target_columns, owned.trainer_input.targets_shape.dim(1));
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 3.0, -0.5, 0.0, 0.0, 0.0 }, owned.targets[0..6]);
    try std.testing.expectEqualSlices(f32, &.{ 2.0, 4.0, 0.0, 0.0, 0.0, 0.0 }, owned.targets[6..12]);
    var row: usize = 2;
    while (row < 64) : (row += 1) {
        const target = owned.targets[row * 6 ..][0..6];
        try std.testing.expectEqual(@as(f32, 0.0), target[0]);
        try std.testing.expectEqual(@as(f32, -100.0), target[1]);
        for (target[2..]) |value| try std.testing.expectEqual(@as(f32, 0.0), value);
    }

    var pair_scheduled = try makeTrainerInputForLogprobCoeffScheduled(
        allocator,
        &ctx,
        &example,
        4,
        128,
        0.25,
    );
    defer pair_scheduled.deinit(allocator);
    try std.testing.expectEqual(@as(i64, 128), pair_scheduled.trainer_input.targets_shape.dim(0));
    try std.testing.expectEqualSlices(f32, owned.targets[0..12], pair_scheduled.targets[0..12]);
    try std.testing.expectError(
        error.InvalidPreferenceTargetRows,
        makeTrainerInputForLogprobCoeffScheduled(allocator, &ctx, &example, 4, 32, 0.25),
    );
    try std.testing.expectError(
        error.InvalidPreferenceTargetRows,
        makeTrainerInputForLogprobCoeffScheduled(allocator, &ctx, &example, 4, 65, 0.25),
    );

    var graph = Graph.init(allocator);
    defer graph.deinit();
    var bld = Builder.init(&graph);
    const hidden = try bld.parameter("hidden", Shape.init(.f32, &.{ 1, 4, 16 }));
    const targets = try bld.parameter("targets", owned.trainer_input.targets_shape);
    _ = try GemmaAutodiffCtx.buildLoss(@ptrCast(&ctx), &bld, hidden, targets);
    var fused_cce_count: usize = 0;
    var vocabulary_projection_count: usize = 0;
    for (0..graph.nodeCount()) |idx| switch (graph.node(@intCast(idx)).op) {
        .fused_linear_cross_entropy_loss => fused_cce_count += 1,
        .fused_linear_no_bias => |attrs| if (attrs.in_dim == 16 and attrs.out_dim == 32) {
            vocabulary_projection_count += 1;
        },
        else => {},
    };
    try std.testing.expectEqual(@as(usize, 1), fused_cce_count);
    try std.testing.expectEqual(@as(usize, 0), vocabulary_projection_count);
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
    try std.testing.expectEqual(@as(i64, 5), owned.trainer_input.targets_shape.dim(1));
    try std.testing.expectEqualSlices(f32, &.{
        1.0, 7.0, 8.0,  0.75, 0.25,
        2.0, 9.0, 10.0, 0.2,  0.8,
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
    var sparse_targets = [_]f32{0} ** 3;

    try std.testing.expectError(
        error.InvalidTeacherTemperature,
        fillTeacherTopKTargets(&dense_targets, 2, 8, &example),
    );
    try std.testing.expectError(
        error.InvalidTeacherTemperature,
        fillSparseTeacherTopKTargets(&sparse_targets, 3, 2, 8, &example),
    );

    example.teacher_temperature = 1.0;
    teacher_probs[0] = std.math.inf(f32);
    try std.testing.expectError(
        error.InvalidTeacherDistillationTargets,
        fillTeacherTopKTargets(&dense_targets, 2, 8, &example),
    );
    try std.testing.expectError(
        error.InvalidTeacherDistillationTargets,
        fillSparseTeacherTopKTargets(&sparse_targets, 3, 2, 8, &example),
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

    try std.testing.expectEqual(@as(i64, 64), owned.trainer_input.targets_shape.dim(0));
    try std.testing.expectEqualSlices(f32, &.{ 2.0, 5.0, -16.0, 0.0 }, owned.targets[0..4]);
    try std.testing.expectEqualSlices(f32, &.{ 3.0, 7.0, 32.0, 0.0 }, owned.targets[4..8]);
    try std.testing.expectEqualSlices(f32, &.{ 4.0, 9.0, -96.0, 0.0 }, owned.targets[8..12]);
    for (owned.targets[12..]) |value| try std.testing.expectEqual(@as(f32, 0.0), value);
}

test "gemma4 makeTrainerInputForExample scales teacher soft targets by temperature squared" {
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

    try std.testing.expectEqualSlices(f32, &.{ 0.0, 5.0, 6.0, 1.0, 3.0 }, owned.targets);
}

test "gemma4 ranked token selection returns deterministic top k" {
    const logits = [_]f32{ 0.5, 3.0, -2.0, 3.0, 1.5, 0.5 };
    var ranked: [4]usize = undefined;
    try selectTopRankedTokens(std.testing.allocator, &logits, &ranked);
    try std.testing.expectEqualSlices(usize, &.{ 1, 3, 4, 0 }, &ranked);
    try std.testing.expectEqual(@as(usize, 1), try selectRankedToken(std.testing.allocator, &logits, 0));
    try std.testing.expectEqual(@as(usize, 2), try selectRankedToken(std.testing.allocator, &logits, 99));
}

test "gemma4 multi-token GRPO predictor rows preserve flattened batch ownership" {
    const first = [_]i32{ 11, 12 };
    const second = [_]i32{13};
    const third = [_]i32{ 14, 15, 16 };
    const completions = [_][]const i32{ &first, &second, &third };
    var predictor_rows: [6]usize = undefined;
    try fillCompletionGroupPredictorRows(3, &completions, 8, &predictor_rows);
    try std.testing.expectEqualSlices(usize, &.{ 2, 3, 10, 18, 19, 20 }, &predictor_rows);

    var too_short: [6]usize = undefined;
    try std.testing.expectError(
        error.SequenceTooLong,
        fillCompletionGroupPredictorRows(6, &completions, 8, &too_short),
    );
    var wrong_count: [5]usize = undefined;
    try std.testing.expectError(
        error.LogpLenMismatch,
        fillCompletionGroupPredictorRows(3, &completions, 8, &wrong_count),
    );
}

test "gemma4 multi-token GRPO backward batches isolated rows with summed weights" {
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

    var first_prompt = [_]i32{ 1, 2 };
    var first_response = [_]i32{ 3, 4 };
    var first_ids = [_]i32{ 1, 2, 3, 4 };
    var first_labels = [_]i32{ -100, -100, 3, 4 };
    var second_prompt = [_]i32{ 5, 6, 7 };
    var second_response = [_]i32{8};
    var second_ids = [_]i32{ 5, 6, 7, 8 };
    var second_labels = [_]i32{ -100, -100, -100, 8 };
    const examples = [_]gemma4.PreparedExampleInput{
        .{
            .mode = .instruction,
            .prompt_input_ids = &first_prompt,
            .response_input_ids = &first_response,
            .num_prompt_tokens = 2,
            .num_response_tokens = 2,
            .input_ids = &first_ids,
            .labels = &first_labels,
            .num_input_tokens = 4,
            .num_supervised_tokens = 2,
        },
        .{
            .mode = .instruction,
            .prompt_input_ids = &second_prompt,
            .response_input_ids = &second_response,
            .num_prompt_tokens = 3,
            .num_response_tokens = 1,
            .input_ids = &second_ids,
            .labels = &second_labels,
            .num_input_tokens = 4,
            .num_supervised_tokens = 1,
        },
    };
    const first_grads = [_]f32{ 0.25, -0.5 };
    const second_grads = [_]f32{1.5};
    const gradient_rows = [_][]const f32{ &first_grads, &second_grads };

    var owned = try makeTrainerInputForTokenLogprobGradBatch(
        allocator,
        &ctx,
        &examples,
        6,
        &gradient_rows,
    );
    defer owned.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 2), owned.trainer_input.batch);
    try std.testing.expectEqual(@as(u32, 6), owned.trainer_input.seq_len);
    try std.testing.expectEqual(@as(usize, 3), owned.supervised_tokens);
    try std.testing.expectEqual(@as(i64, 64), owned.trainer_input.targets_shape.dim(0));
    try std.testing.expectEqual(@as(i64, 4), owned.trainer_input.targets_shape.dim(1));
    try std.testing.expectEqualSlices(i64, &.{ 1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0 }, owned.input_ids);
    try std.testing.expectEqualSlices(f32, &.{ 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0 }, owned.attention_mask);
    try std.testing.expectEqualSlices(f32, &.{
        1, 3, -16, 0,
        2, 4, 32,  0,
        8, 8, -96, 0,
    }, owned.targets[0..12]);
    try std.testing.expectEqual(@as(f32, -100), owned.targets[13]);
}

test "gemma4 single-token GRPO group coalesces weighted targets on one predictor row" {
    const allocator = std.testing.allocator;
    var ctx = GemmaAutodiffCtx.init(.{
        .family = .gemma,
        .hidden_size = 16,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .attention_head_dim = 4,
        .intermediate_size = 32,
        .vocab_size = 16,
        .position_encoding = .rope,
        .norm_type = .rms_norm,
        .activation = .gelu_new,
        .norm_eps = 1e-6,
        .norm_weight_offset = 1.0,
    });
    var prompt_input_ids = [_]i32{ 1, 2, 3 };
    var response_input_ids = [_]i32{5};
    var input_ids = [_]i32{ 1, 2, 3, 5 };
    var labels = [_]i32{ -100, -100, -100, 5 };
    const example = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = &prompt_input_ids,
        .response_input_ids = &response_input_ids,
        .num_prompt_tokens = 3,
        .num_response_tokens = 1,
        .input_ids = &input_ids,
        .labels = &labels,
        .num_input_tokens = 4,
        .num_supervised_tokens = 1,
    };

    var owned = try makeTrainerInputForSingleTokenCompletionGroupLogprobGrads(
        allocator,
        &ctx,
        &example,
        6,
        &.{ 5, 7, 5 },
        &.{ 0.25, -0.5, 1.5 },
    );
    defer owned.deinit(allocator);

    try std.testing.expectEqual(@as(i64, 1), owned.trainer_input.targets_shape.dim(0));
    try std.testing.expectEqual(@as(i64, 17), owned.trainer_input.targets_shape.dim(1));
    try std.testing.expectEqual(@as(f32, 2.0), owned.targets[0]);
    try std.testing.expectEqual(@as(f32, 5.0), owned.targets[1]);
    try std.testing.expectEqual(@as(f32, 7.0), owned.targets[2]);
    try std.testing.expectApproxEqAbs(@as(f32, -1.75), owned.targets[9], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), owned.targets[10], 1e-6);
}

test "gemma4 single-token DPO pair coalesces opposing logprob gradients" {
    const allocator = std.testing.allocator;
    var ctx = GemmaAutodiffCtx.init(.{
        .family = .gemma,
        .hidden_size = 16,
        .num_hidden_layers = 2,
        .num_attention_heads = 4,
        .num_key_value_heads = 2,
        .attention_head_dim = 4,
        .intermediate_size = 32,
        .vocab_size = 16,
        .position_encoding = .rope,
        .norm_type = .rms_norm,
        .activation = .gelu_new,
        .norm_eps = 1e-6,
        .norm_weight_offset = 1.0,
    });
    var prompt_input_ids = [_]i32{ 1, 2, 3 };
    var response_input_ids = [_]i32{5};
    var input_ids = [_]i32{ 1, 2, 3, 5 };
    var labels = [_]i32{ -100, -100, -100, 5 };
    const example = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = &prompt_input_ids,
        .response_input_ids = &response_input_ids,
        .num_prompt_tokens = 3,
        .num_response_tokens = 1,
        .input_ids = &input_ids,
        .labels = &labels,
        .num_input_tokens = 4,
        .num_supervised_tokens = 1,
    };

    var owned = try makeTrainerInputForSingleTokenCandidatesLogprobGrads(
        allocator,
        &ctx,
        &example,
        6,
        &.{ 5, 7 },
        &.{ -0.25, 0.25 },
    );
    defer owned.deinit(allocator);

    try std.testing.expectEqual(@as(i64, 1), owned.trainer_input.targets_shape.dim(0));
    try std.testing.expectEqual(@as(i64, 17), owned.trainer_input.targets_shape.dim(1));
    try std.testing.expectEqual(@as(f32, 2.0), owned.targets[0]);
    try std.testing.expectEqual(@as(f32, 5.0), owned.targets[1]);
    try std.testing.expectEqual(@as(f32, 7.0), owned.targets[2]);
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), owned.targets[9], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, -0.25), owned.targets[10], 1e-6);
}

test "gemma4 compiled DPO pair input keeps causal streams and metadata separate" {
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
    var prompt = [_]i32{ 1, 2, 3 };
    var chosen_response = [_]i32{ 5, 7 };
    var rejected_response = [_]i32{ 11, 13, 17 };
    var chosen_ids = [_]i32{ 1, 2, 3, 5, 7 };
    var rejected_ids = [_]i32{ 1, 2, 3, 11, 13, 17 };
    var chosen_labels = [_]i32{ -100, -100, -100, 5, 7 };
    var rejected_labels = [_]i32{ -100, -100, -100, 11, 13, 17 };
    const chosen = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = &prompt,
        .response_input_ids = &chosen_response,
        .num_prompt_tokens = prompt.len,
        .num_response_tokens = chosen_response.len,
        .input_ids = &chosen_ids,
        .labels = &chosen_labels,
        .num_input_tokens = chosen_ids.len,
        .num_supervised_tokens = chosen_response.len,
    };
    const rejected = gemma4.PreparedExampleInput{
        .mode = .instruction,
        .prompt_input_ids = &prompt,
        .response_input_ids = &rejected_response,
        .num_prompt_tokens = prompt.len,
        .num_response_tokens = rejected_response.len,
        .input_ids = &rejected_ids,
        .labels = &rejected_labels,
        .num_input_tokens = rejected_ids.len,
        .num_supervised_tokens = rejected_response.len,
    };

    var owned = try makeTrainerInputForDpoPair(allocator, &ctx, &chosen, &rejected, 8, .split_batch1, -5.0, -3.0, 0.1);
    defer owned.deinit(allocator);

    try std.testing.expectEqual(@as(u32, 2), owned.trainer_input.batch);
    try std.testing.expectEqual(@as(usize, 16), owned.input_ids.len);
    try std.testing.expectEqual(@as(i64, 129), owned.trainer_input.targets_shape.dim(0));
    try std.testing.expectEqual(@as(i64, 4), owned.trainer_input.targets_shape.dim(1));
    try std.testing.expectEqual(@as(?u32, 64), ctx.dpo_pair_bucket_rows);
    try std.testing.expectEqual(dpo_pair_graph_variant_marker, owned.trainer_input.graph_variant[0]);
    try std.testing.expectEqual(@as(u64, @intFromEnum(DpoPairGraphMode.split_batch1)), owned.trainer_input.graph_variant[2]);
    try std.testing.expect(owned.trainer_input.build_objective != null);
    try std.testing.expectEqualSlices(f32, &.{ 2.0, 5.0, 0.0, 0.0 }, owned.targets[0..4]);
    try std.testing.expectEqualSlices(f32, &.{ 3.0, 7.0, 0.0, 0.0 }, owned.targets[4..8]);
    const rejected_base = 64 * 4;
    try std.testing.expectEqualSlices(f32, &.{ 2.0, 11.0, 0.0, 0.0 }, owned.targets[rejected_base..][0..4]);
    try std.testing.expectEqualSlices(f32, &.{ 3.0, 13.0, 0.0, 0.0 }, owned.targets[rejected_base + 4 ..][0..4]);
    try std.testing.expectEqualSlices(f32, &.{ 4.0, 17.0, 0.0, 0.0 }, owned.targets[rejected_base + 8 ..][0..4]);
    const metadata_base = 128 * 4;
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), owned.targets[metadata_base], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), owned.targets[metadata_base + 1], 1e-6);
    try std.testing.expectEqual(@as(f32, 2.0), owned.targets[metadata_base + 2]);
    try std.testing.expectEqual(@as(f32, 3.0), owned.targets[metadata_base + 3]);

    var batched = try makeTrainerInputForDpoPair(allocator, &ctx, &chosen, &rejected, 8, .batched_forward, -5.0, -3.0, 0.1);
    defer batched.deinit(allocator);
    try std.testing.expect(batched.trainer_input.build_objective == null);
    try std.testing.expectEqual(@as(u64, @intFromEnum(DpoPairGraphMode.batched_forward)), batched.trainer_input.graph_variant[2]);
    try std.testing.expectEqualSlices(f32, &.{ 10.0, 11.0, 0.0, 0.0 }, batched.targets[rejected_base..][0..4]);
    try std.testing.expectEqualSlices(f32, &.{ 11.0, 13.0, 0.0, 0.0 }, batched.targets[rejected_base + 4 ..][0..4]);
    try std.testing.expectEqualSlices(f32, &.{ 12.0, 17.0, 0.0, 0.0 }, batched.targets[rejected_base + 8 ..][0..4]);
}
