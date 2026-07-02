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

// Segmented backpropagation through the encoder for LoRA gradient computation.
//
// The production path mirrors GoMLX's compiled backward shape: build fixed-size
// ModernBERT segment VJP graphs, compile them through MPSGraph by default, and
// execute each segment from the output layer back toward the embeddings. The
// older direct LoRA scatter path remains available when exact adapter gradients
// are disabled, but the Phase-20 parity runner is intended to use MPSGraph
// segment VJPs for the real backward pass.

const std = @import("std");
const ml = @import("ml");
const platform = @import("antfly_platform");
const modern_bert_graph = @import("../architectures/modern_bert_graph.zig");
const lora = @import("../finetune/lora.zig");
const fused_chunker_lora = @import("../finetune/lora_adapter_set.zig");
const graph_input_binder = @import("../finetune/graph_input_binder.zig");
const ops = @import("../ops/ops.zig");
const training = @import("training.zig");
const mpsgraph_executor = @import("mpsgraph_executor.zig");

const Graph = ml.graph.Graph;
const Builder = ml.graph.Builder;
const NodeId = ml.graph.NodeId;
const Shape = ml.graph.Shape;
const null_node = ml.graph.null_node;
const graph_lora = ml.graph.lora;
const CT = ops.CT;
const ComputeBackend = ops.ComputeBackend;

const segment_lora_targets = [_][]const u8{
    "attn.query_proj.weight",
    "attn.key_proj.weight",
    "attn.value_proj.weight",
    "attn.Wo.weight",
    "mlp.Wo.weight",
};

/// Opt-in fast path for segment VJP execution (default OFF): cache RoPE
/// tables and frozen base-weight host buffers per session, share the two
/// attention-bias variants across segments within a step, and feed the
/// MPSGraph executable from stable host buffers instead of per-exec backend
/// tensors. Byte-identical inputs; only redundant copies are removed.
pub fn vjpFeedCacheEnabled() bool {
    return platform.env.getenvBoolDefault("ANTFLY_FUSED_CHUNKER_VJP_FEED_CACHE", false);
}

/// Per-step cache for the ModernBERT segment attention biases. Every layer in
/// a step shares one of exactly two bias tensors (local sliding-window or
/// global), both pure functions of the step's attention mask — build each at
/// most once per step instead of once per segment execution.
pub const SharedSegmentAttnBias = struct {
    allocator: std.mem.Allocator,
    batch: u32,
    seq_len: u32,
    num_heads: u32,
    local_attention_window: u32,
    /// Borrowed for the lifetime of the step.
    attention_mask: []const f32,
    local_bias: ?[]f32 = null,
    global_bias: ?[]f32 = null,

    pub fn init(
        allocator: std.mem.Allocator,
        batch: u32,
        seq_len: u32,
        num_heads: u32,
        local_attention_window: u32,
        attention_mask: []const f32,
    ) SharedSegmentAttnBias {
        return .{
            .allocator = allocator,
            .batch = batch,
            .seq_len = seq_len,
            .num_heads = num_heads,
            .local_attention_window = local_attention_window,
            .attention_mask = attention_mask,
        };
    }

    pub fn deinit(self: *SharedSegmentAttnBias) void {
        if (self.local_bias) |buf| self.allocator.free(buf);
        if (self.global_bias) |buf| self.allocator.free(buf);
        self.* = undefined;
    }

    pub fn get(self: *SharedSegmentAttnBias, is_local_attention: bool) ![]const f32 {
        const slot = if (is_local_attention) &self.local_bias else &self.global_bias;
        if (slot.* == null) {
            slot.* = try buildSegmentAttnBias(
                self.allocator,
                self.attention_mask,
                self.batch,
                self.seq_len,
                self.num_heads,
                is_local_attention,
                self.local_attention_window,
            );
        }
        return slot.*.?;
    }
};

/// Cached RoPE cos/sin table for one layer binding.
const RopeTable = struct {
    cos: []f32,
    sin: []f32,
};

fn segmentVjpExecutionStrategy() training.CompiledExecutionStrategy {
    // Full segment VJPs are intended to mirror GoMLX's compiled backward graph:
    // MPSGraph is the primary path, with explicit overrides for diagnostics.
    const raw = platform.env.getenv("ANTFLY_FUSED_CHUNKER_ENCODER_VJP_EXECUTION") orelse
        platform.env.getenv("ANTFLY_FUSED_CHUNKER_ENCODER_VJP_BACKEND") orelse
        return .mpsgraph_preferred;
    if (std.mem.eql(u8, raw, "interpreter")) return .interpreter;
    if (std.mem.eql(u8, raw, "partitioned") or std.mem.eql(u8, raw, "metal")) return .partitioned_preferred;
    if (std.mem.eql(u8, raw, "partitioned_required") or std.mem.eql(u8, raw, "metal_required")) return .partitioned_required;
    if (std.mem.eql(u8, raw, "mpsgraph")) return .mpsgraph_preferred;
    if (std.mem.eql(u8, raw, "mpsgraph_required")) return .mpsgraph_required;
    return .mpsgraph_preferred;
}

// ----------------------------------------------------------------------------
// LayerActivation
// ----------------------------------------------------------------------------

/// The input tensor captured at a single linear projection during a forward
/// pass.  Used as the left-hand side when computing LoRA gradients.
pub const LayerActivation = struct {
    layer_idx: u32,
    /// Module name, e.g. "query_proj" or "value_proj".  Not owned — points
    /// into a string literal or the LoRALayer's module_name buffer.
    module_name: []const u8,
    /// Owned flat buffer: [seq_len * in_features] in row-major order.
    input: []f32,
    in_features: usize,
    out_features: usize,
    seq_len: usize,

    pub fn deinit(self: *LayerActivation, allocator: std.mem.Allocator) void {
        allocator.free(self.input);
        self.* = undefined;
    }
};

// ----------------------------------------------------------------------------
// EncoderActivations
// ----------------------------------------------------------------------------

/// A collection of LayerActivation records captured during one encoder forward
/// pass — one entry per targeted (layer_idx, module_name) pair.
pub const EncoderActivations = struct {
    allocator: std.mem.Allocator,
    /// Owned slice grown via addActivation.
    activations: std.ArrayListUnmanaged(LayerActivation),
    seq_len: usize,
    hidden_size: usize,

    pub fn init(
        allocator: std.mem.Allocator,
        capacity: usize,
        seq_len: usize,
        hidden_size: usize,
    ) !EncoderActivations {
        var list = std.ArrayListUnmanaged(LayerActivation).empty;
        try list.ensureTotalCapacity(allocator, capacity);
        return .{
            .allocator = allocator,
            .activations = list,
            .seq_len = seq_len,
            .hidden_size = hidden_size,
        };
    }

    pub fn deinit(self: *EncoderActivations) void {
        for (self.activations.items) |*act| {
            act.deinit(self.allocator);
        }
        self.activations.deinit(self.allocator);
        self.* = undefined;
    }

    /// Append an activation record, copying the input slice.
    ///
    /// `input` must have length `seq_len * in_features`; the data is copied
    /// into an owned buffer so the caller may safely reuse its memory.
    pub fn addActivation(
        self: *EncoderActivations,
        layer_idx: u32,
        module_name: []const u8,
        input: []const f32,
        in_features: usize,
        out_features: usize,
    ) !void {
        const owned_input = try self.allocator.dupe(f32, input);
        errdefer self.allocator.free(owned_input);

        try self.activations.append(self.allocator, .{
            .layer_idx = layer_idx,
            .module_name = module_name,
            .input = owned_input,
            .in_features = in_features,
            .out_features = out_features,
            .seq_len = self.seq_len,
        });
    }
};

// ----------------------------------------------------------------------------
// ModernBERT segmented VJP session
// ----------------------------------------------------------------------------

pub const SegmentVJPResult = struct {
    allocator: std.mem.Allocator,
    hidden_grad: ?[]f32 = null,
    loss: f32,
    profile: SegmentVJPProfile = .{},

    pub fn deinit(self: *SegmentVJPResult) void {
        if (self.hidden_grad) |buf| self.allocator.free(buf);
        self.* = undefined;
    }
};

pub const NamedGradient = struct {
    name: []u8,
    values: []f32,

    pub fn deinit(self: *NamedGradient, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        allocator.free(self.values);
        self.* = undefined;
    }
};

pub const LayerBackwardSubstageInputs = struct {
    upstream_grad: []const f32,
    primary: ?[]const f32 = null,
    q_raw: ?[]const f32 = null,
    k_raw: ?[]const f32 = null,
    v_raw: ?[]const f32 = null,
    hidden_in_aux: ?[]const f32 = null,
    hidden_after_attn_aux: ?[]const f32 = null,
    gate_input_aux: ?[]const f32 = null,
    gate_value_aux: ?[]const f32 = null,
    attention_mask: []const f32,
};

pub const LayerBackwardSubstageResult = struct {
    allocator: std.mem.Allocator,
    stage: modern_bert_graph.LayerBackwardSubstage,
    loss: f32,
    profile: SegmentVJPProfile = .{},
    stage_grads: []NamedGradient = &.{},
    adapter_grads: []NamedGradient = &.{},

    pub fn deinit(self: *LayerBackwardSubstageResult) void {
        for (self.stage_grads) |*grad| grad.deinit(self.allocator);
        if (self.stage_grads.len > 0) self.allocator.free(self.stage_grads);
        for (self.adapter_grads) |*grad| grad.deinit(self.allocator);
        if (self.adapter_grads.len > 0) self.allocator.free(self.adapter_grads);
        self.* = undefined;
    }
};

pub const SegmentVJPProfile = struct {
    runtime_input_ns: u64 = 0,
    compiled_execute_ns: u64 = 0,
    compiled_extract_ns: u64 = 0,
    compiled_total_ns: u64 = 0,
    accumulate_ns: u64 = 0,
    total_ns: u64 = 0,
    partitioned_runtime: bool = false,
    mpsgraph_runtime: bool = false,
    partitions_executed: u64 = 0,
    backend_command_dispatches: u64 = 0,
    planned_operator_dispatches: u64 = 0,
    interpreter_fallbacks: u64 = 0,
    graph_region_dispatches: u64 = 0,

    pub fn add(self: *SegmentVJPProfile, other: SegmentVJPProfile) void {
        self.runtime_input_ns += other.runtime_input_ns;
        self.compiled_execute_ns += other.compiled_execute_ns;
        self.compiled_extract_ns += other.compiled_extract_ns;
        self.compiled_total_ns += other.compiled_total_ns;
        self.accumulate_ns += other.accumulate_ns;
        self.total_ns += other.total_ns;
        self.partitioned_runtime = self.partitioned_runtime or other.partitioned_runtime;
        self.mpsgraph_runtime = self.mpsgraph_runtime or other.mpsgraph_runtime;
        self.partitions_executed += other.partitions_executed;
        self.backend_command_dispatches += other.backend_command_dispatches;
        self.planned_operator_dispatches += other.planned_operator_dispatches;
        self.interpreter_fallbacks += other.interpreter_fallbacks;
        self.graph_region_dispatches += other.graph_region_dispatches;
    }
};

const ParamBinding = struct {
    node_id: NodeId,
    name: []const u8,

    fn deinit(self: *ParamBinding, allocator: std.mem.Allocator) void {
        allocator.free(self.name);
        self.* = undefined;
    }
};

const AdapterBinding = struct {
    base_name: []const u8,
    lora_a_name: []const u8,
    lora_b_name: []const u8,
    lora_a_id: NodeId,
    lora_b_id: NodeId,
    lora_a_rows: usize,
    lora_a_cols: usize,
    lora_b_rows: usize,
    lora_b_cols: usize,
    layer_idx: u32,
    module_name: []const u8,

    fn deinit(self: *AdapterBinding, allocator: std.mem.Allocator) void {
        allocator.free(self.base_name);
        allocator.free(self.lora_a_name);
        allocator.free(self.lora_b_name);
        allocator.free(self.module_name);
        self.* = undefined;
    }
};

const SegmentLayerInputBinding = struct {
    layer_idx: u32,
    attn_bias_id: NodeId,
    rope_cos_id: NodeId,
    rope_sin_id: NodeId,
    is_local_attention: bool,
    rope_theta: f32,
};

const stage_mlp_wo_names = [_][]const u8{"__substage_mlp_wo_input"};
const stage_mlp_gelu_input_names = [_][]const u8{"__substage_gelu_input"};
const stage_mlp_gate_value_names = [_][]const u8{"__substage_gate_value"};
const stage_mlp_gate_input_names = [_][]const u8{"__substage_gate_input"};
const stage_mlp_wi_output_names = [_][]const u8{"__substage_wi_output"};
const stage_mlp_norm_output_names = [_][]const u8{"__substage_mlp_norm_output"};
const stage_mlp_hidden_after_attn_names = [_][]const u8{"__substage_hidden_after_attn"};
const stage_attn_out_proj_names = [_][]const u8{"__substage_attn_merged"};
const stage_attention_core_names = [_][]const u8{ "__substage_q_raw", "__substage_k_raw", "__substage_v_raw" };
const stage_attention_core_post_rope_names = [_][]const u8{ "__substage_q_rope", "__substage_k_rope", "__substage_v_attention_input" };
const stage_attention_scores_raw_names = [_][]const u8{"__substage_scores_raw"};
const stage_attention_scores_masked_names = [_][]const u8{"__substage_scores_masked"};
const stage_attention_probs_names = [_][]const u8{ "__substage_attention_probs", "__substage_v_attention_input" };
const stage_qkv_proj_names = [_][]const u8{"__substage_attn_normed"};
const stage_qkv_proj_split_names = [_][]const u8{ "__substage_q_attn_normed", "__substage_k_attn_normed", "__substage_v_attn_normed" };
const stage_attn_norm_hidden_in_names = [_][]const u8{"__substage_hidden_in"};

fn substageInputParamNames(stage: modern_bert_graph.LayerBackwardSubstage) []const []const u8 {
    return switch (stage) {
        .mlp_wo => stage_mlp_wo_names[0..],
        .mlp_gelu_input => stage_mlp_gelu_input_names[0..],
        .mlp_gate_value => stage_mlp_gate_value_names[0..],
        .mlp_gate_input => stage_mlp_gate_input_names[0..],
        .mlp_wi_output => stage_mlp_wi_output_names[0..],
        .mlp_norm_output => stage_mlp_norm_output_names[0..],
        .mlp_hidden_after_attn => stage_mlp_hidden_after_attn_names[0..],
        .attn_out_proj => stage_attn_out_proj_names[0..],
        .attention_core => stage_attention_core_names[0..],
        .attention_core_post_rope => stage_attention_core_post_rope_names[0..],
        .attention_scores_raw => stage_attention_scores_raw_names[0..],
        .attention_scores_masked => stage_attention_scores_masked_names[0..],
        .attention_probs => stage_attention_probs_names[0..],
        .qkv_proj => stage_qkv_proj_names[0..],
        .qkv_proj_split => stage_qkv_proj_split_names[0..],
        .attn_norm_hidden_in => stage_attn_norm_hidden_in_names[0..],
    };
}

fn substageUsesAttentionBias(stage: modern_bert_graph.LayerBackwardSubstage) bool {
    return switch (stage) {
        .attention_core, .attention_core_post_rope, .attention_scores_raw, .qkv_proj, .qkv_proj_split, .attn_norm_hidden_in => true,
        else => false,
    };
}

fn substageUsesRope(stage: modern_bert_graph.LayerBackwardSubstage) bool {
    return switch (stage) {
        .attention_core, .qkv_proj, .qkv_proj_split, .attn_norm_hidden_in => true,
        else => false,
    };
}

fn substagePrimaryInputIsAttentionMatrix(stage: modern_bert_graph.LayerBackwardSubstage) bool {
    return switch (stage) {
        .attention_scores_raw, .attention_scores_masked, .attention_probs => true,
        else => false,
    };
}

fn substageUsesVAttentionInput(stage: modern_bert_graph.LayerBackwardSubstage) bool {
    return switch (stage) {
        .attention_scores_raw, .attention_scores_masked, .attention_probs => true,
        else => false,
    };
}

fn substagePrimaryInputName(stage: modern_bert_graph.LayerBackwardSubstage) ?[]const u8 {
    return switch (stage) {
        .mlp_wo => "__substage_mlp_wo_input",
        .mlp_gelu_input => "__substage_gelu_input",
        .mlp_gate_value => "__substage_gate_value",
        .mlp_gate_input => "__substage_gate_input",
        .mlp_wi_output => "__substage_wi_output",
        .mlp_norm_output => "__substage_mlp_norm_output",
        .mlp_hidden_after_attn => "__substage_hidden_after_attn",
        .attn_out_proj => "__substage_attn_merged",
        .attention_scores_raw => "__substage_scores_raw",
        .attention_scores_masked => "__substage_scores_masked",
        .attention_probs => "__substage_attention_probs",
        .qkv_proj => "__substage_attn_normed",
        .attn_norm_hidden_in => "__substage_hidden_in",
        .attention_core, .attention_core_post_rope, .qkv_proj_split => null,
    };
}

fn substagePrimaryInputCols(stage: modern_bert_graph.LayerBackwardSubstage, hidden_size: usize, intermediate_size: usize) usize {
    return switch (stage) {
        .mlp_wo,
        .mlp_gelu_input,
        .mlp_gate_value,
        .mlp_gate_input,
        => intermediate_size,
        .mlp_wi_output => intermediate_size * 2,
        else => hidden_size,
    };
}

pub const ModernBertLayerBackwardSubstageSession = struct {
    allocator: std.mem.Allocator,
    config: modern_bert_graph.Config,
    batch: u32,
    seq_len: u32,
    layer_idx: u32,
    stage: modern_bert_graph.LayerBackwardSubstage,
    primary_input_id: NodeId,
    q_raw_id: NodeId,
    k_raw_id: NodeId,
    v_raw_id: NodeId,
    hidden_in_aux_id: NodeId,
    hidden_after_attn_aux_id: NodeId,
    gate_input_aux_id: NodeId,
    gate_value_aux_id: NodeId,
    upstream_grad_id: NodeId,
    attn_bias_id: NodeId,
    rope_cos_id: NodeId,
    rope_sin_id: NodeId,
    layer_input: SegmentLayerInputBinding,
    base_params: []ParamBinding,
    adapters: []AdapterBinding,
    trainable_param_names: [][]const u8,
    session: training.CompiledTrainSession,

    pub fn init(
        allocator: std.mem.Allocator,
        config: modern_bert_graph.Config,
        batch: u32,
        seq_len: u32,
        layer_idx: u32,
        stage: modern_bert_graph.LayerBackwardSubstage,
        lora_rank: u32,
        lora_alpha: f32,
        execution_strategy: training.CompiledExecutionStrategy,
    ) !ModernBertLayerBackwardSubstageSession {
        if (lora_rank == 0) return error.LoRARankRequired;
        if (layer_idx >= config.num_hidden_layers) return error.InvalidLayerRange;

        var g = Graph.init(allocator);
        defer g.deinit();
        var bld = Builder.init(&g);

        const graph = try modern_bert_graph.buildEncoderLayerBackwardSubstageGraph(
            &bld,
            config,
            stage,
            batch,
            seq_len,
            layer_idx,
        );

        var lora_result = try graph_lora.injectLoRA(allocator, &g, .{
            .rank = lora_rank,
            .alpha = lora_alpha,
            .target_patterns = segment_lora_targets[0..],
        });
        defer lora_result.deinit();

        var base_params_list: std.ArrayListUnmanaged(ParamBinding) = .empty;
        errdefer {
            for (base_params_list.items) |*p| p.deinit(allocator);
            base_params_list.deinit(allocator);
        }
        for (lora_result.graph.parameters.items) |param_id| {
            const n = lora_result.graph.node(param_id);
            if (n.op != .parameter) continue;
            const name = lora_result.graph.parameterName(n);
            if (std.mem.startsWith(u8, name, "__")) continue;
            if (std.mem.indexOf(u8, name, ".lora_A") != null or
                std.mem.indexOf(u8, name, ".lora_B") != null)
            {
                continue;
            }
            const owned_name = try allocator.dupe(u8, name);
            errdefer allocator.free(owned_name);
            try base_params_list.append(allocator, .{
                .node_id = param_id,
                .name = owned_name,
            });
        }

        var adapter_list: std.ArrayListUnmanaged(AdapterBinding) = .empty;
        errdefer {
            for (adapter_list.items) |*a| a.deinit(allocator);
            adapter_list.deinit(allocator);
        }
        for (lora_result.adapter.adapterNames()) |info| {
            const stable_base_name = stripLoRASuffix(info.lora_a_name) orelse info.base_name;
            const mapped = moduleFromModernBertBaseName(stable_base_name) catch continue;
            const base_name = try allocator.dupe(u8, stable_base_name);
            errdefer allocator.free(base_name);
            const a_name = try allocator.dupe(u8, info.lora_a_name);
            errdefer allocator.free(a_name);
            const b_name = try allocator.dupe(u8, info.lora_b_name);
            errdefer allocator.free(b_name);
            const module_name = try allocator.dupe(u8, mapped.module_name);
            errdefer allocator.free(module_name);
            const a_shape = lora_result.graph.node(info.lora_a_id).output_shape;
            const b_shape = lora_result.graph.node(info.lora_b_id).output_shape;
            try adapter_list.append(allocator, .{
                .base_name = base_name,
                .lora_a_name = a_name,
                .lora_b_name = b_name,
                .lora_a_id = info.lora_a_id,
                .lora_b_id = info.lora_b_id,
                .lora_a_rows = @intCast(a_shape.dim(0)),
                .lora_a_cols = @intCast(a_shape.dim(1)),
                .lora_b_rows = @intCast(b_shape.dim(0)),
                .lora_b_cols = @intCast(b_shape.dim(1)),
                .layer_idx = mapped.layer_idx,
                .module_name = module_name,
            });
        }

        const stage_names = substageInputParamNames(stage);
        var trainable_names = try allocator.alloc([]const u8, stage_names.len + adapter_list.items.len * 2);
        errdefer allocator.free(trainable_names);
        var tn: usize = 0;
        for (stage_names) |name| {
            trainable_names[tn] = name;
            tn += 1;
        }
        for (adapter_list.items) |adapter| {
            trainable_names[tn] = adapter.lora_a_name;
            tn += 1;
            trainable_names[tn] = adapter.lora_b_name;
            tn += 1;
        }

        var session = try training.CompiledTrainSession.init(
            allocator,
            &lora_result.graph,
            graph.loss_node,
            .{
                .trainable_params = trainable_names,
                .execution_strategy = execution_strategy,
            },
        );
        errdefer session.deinit();

        const layer_is_global = modern_bert_graph.isGlobalAttentionLayer(config, layer_idx);
        return .{
            .allocator = allocator,
            .config = config,
            .batch = batch,
            .seq_len = seq_len,
            .layer_idx = layer_idx,
            .stage = stage,
            .primary_input_id = graph.primary_input_node,
            .q_raw_id = graph.q_raw_node,
            .k_raw_id = graph.k_raw_node,
            .v_raw_id = graph.v_raw_node,
            .hidden_in_aux_id = graph.hidden_in_aux_node,
            .hidden_after_attn_aux_id = graph.hidden_after_attn_aux_node,
            .gate_input_aux_id = graph.gate_input_aux_node,
            .gate_value_aux_id = graph.gate_value_aux_node,
            .upstream_grad_id = graph.upstream_grad_node,
            .attn_bias_id = graph.attn_bias_node,
            .rope_cos_id = graph.rope_cos_node,
            .rope_sin_id = graph.rope_sin_node,
            .layer_input = .{
                .layer_idx = layer_idx,
                .attn_bias_id = graph.attn_bias_node,
                .rope_cos_id = graph.rope_cos_node,
                .rope_sin_id = graph.rope_sin_node,
                .is_local_attention = !layer_is_global,
                .rope_theta = modern_bert_graph.ropeThetaForLayer(config, layer_idx),
            },
            .base_params = try base_params_list.toOwnedSlice(allocator),
            .adapters = try adapter_list.toOwnedSlice(allocator),
            .trainable_param_names = trainable_names,
            .session = session,
        };
    }

    pub fn deinit(self: *ModernBertLayerBackwardSubstageSession) void {
        self.session.deinit();
        for (self.base_params) |*p| p.deinit(self.allocator);
        self.allocator.free(self.base_params);
        for (self.adapters) |*adapter| adapter.deinit(self.allocator);
        self.allocator.free(self.adapters);
        self.allocator.free(self.trainable_param_names);
        self.* = undefined;
    }

    pub fn execute(
        self: *ModernBertLayerBackwardSubstageSession,
        cb: *const ComputeBackend,
        inputs: LayerBackwardSubstageInputs,
        lora_layers: []fused_chunker_lora.LoRALayer,
    ) !LayerBackwardSubstageResult {
        const total_start_ns = nowNs();
        const total: usize = @as(usize, self.batch) * @as(usize, self.seq_len);
        const H: usize = @intCast(self.config.hidden_size);
        const I: usize = @intCast(self.config.intermediate_size);
        if (inputs.upstream_grad.len != total * H) return error.InvalidSegmentTensorShape;
        if (inputs.attention_mask.len != total) return error.InvalidAttentionMaskShape;

        var rt = std.AutoHashMapUnmanaged(NodeId, CT){};
        defer rt.deinit(self.allocator);
        var owned_cts: std.ArrayListUnmanaged(CT) = .empty;
        defer {
            for (owned_cts.items) |ct| cb.free(ct);
            owned_cts.deinit(self.allocator);
        }

        try self.putOwnedTensor(cb, &rt, &owned_cts, self.upstream_grad_id, inputs.upstream_grad, &.{ @intCast(total), @intCast(H) });
        if (substagePrimaryInputName(self.stage) != null) {
            const primary = inputs.primary orelse return error.MissingSubstageInput;
            if (substagePrimaryInputIsAttentionMatrix(self.stage)) {
                const bh: usize = @as(usize, self.batch) * @as(usize, self.config.num_attention_heads);
                const seq: usize = @intCast(self.seq_len);
                if (primary.len != bh * seq * seq) return error.InvalidSegmentTensorShape;
                try self.putOwnedTensor(cb, &rt, &owned_cts, self.primary_input_id, primary, &.{ @intCast(bh), @intCast(seq), @intCast(seq) });
            } else {
                const primary_cols = substagePrimaryInputCols(self.stage, H, I);
                if (primary.len != total * primary_cols) return error.InvalidSegmentTensorShape;
                try self.putOwnedTensor(cb, &rt, &owned_cts, self.primary_input_id, primary, &.{ @intCast(total), @intCast(primary_cols) });
            }
        }
        if (self.stage == .attention_core or self.stage == .attention_core_post_rope) {
            const q = inputs.q_raw orelse return error.MissingSubstageInput;
            const k = inputs.k_raw orelse return error.MissingSubstageInput;
            const v = inputs.v_raw orelse return error.MissingSubstageInput;
            if (q.len != total * H or k.len != total * H or v.len != total * H) return error.InvalidSegmentTensorShape;
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.q_raw_id, q, &.{ @intCast(total), @intCast(H) });
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.k_raw_id, k, &.{ @intCast(total), @intCast(H) });
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.v_raw_id, v, &.{ @intCast(total), @intCast(H) });
        }
        if (substageUsesVAttentionInput(self.stage)) {
            const v = inputs.v_raw orelse return error.MissingSubstageInput;
            if (v.len != total * H) return error.InvalidSegmentTensorShape;
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.v_raw_id, v, &.{ @intCast(total), @intCast(H) });
        }
        if (self.stage == .qkv_proj_split) {
            const primary = inputs.primary orelse return error.MissingSubstageInput;
            if (primary.len != total * H) return error.InvalidSegmentTensorShape;
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.q_raw_id, primary, &.{ @intCast(total), @intCast(H) });
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.k_raw_id, primary, &.{ @intCast(total), @intCast(H) });
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.v_raw_id, primary, &.{ @intCast(total), @intCast(H) });
        }
        if (self.hidden_in_aux_id != null_node) {
            const hidden_in = inputs.hidden_in_aux orelse return error.MissingSubstageInput;
            if (hidden_in.len != total * H) return error.InvalidSegmentTensorShape;
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.hidden_in_aux_id, hidden_in, &.{ @intCast(total), @intCast(H) });
        }
        if (self.hidden_after_attn_aux_id != null_node) {
            const hidden_after_attn = inputs.hidden_after_attn_aux orelse return error.MissingSubstageInput;
            if (hidden_after_attn.len != total * H) return error.InvalidSegmentTensorShape;
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.hidden_after_attn_aux_id, hidden_after_attn, &.{ @intCast(total), @intCast(H) });
        }
        if (self.gate_input_aux_id != null_node) {
            const gate_input = inputs.gate_input_aux orelse return error.MissingSubstageInput;
            if (gate_input.len != total * I) return error.InvalidSegmentTensorShape;
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.gate_input_aux_id, gate_input, &.{ @intCast(total), @intCast(I) });
        }
        if (self.gate_value_aux_id != null_node) {
            const gate_value = inputs.gate_value_aux orelse return error.MissingSubstageInput;
            if (gate_value.len != total * I) return error.InvalidSegmentTensorShape;
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.gate_value_aux_id, gate_value, &.{ @intCast(total), @intCast(I) });
        }
        if (substageUsesAttentionBias(self.stage)) {
            const attn_bias = try buildSegmentAttnBias(
                self.allocator,
                inputs.attention_mask,
                self.batch,
                self.seq_len,
                self.config.num_attention_heads,
                self.layer_input.is_local_attention,
                self.config.local_attention_window,
            );
            defer self.allocator.free(attn_bias);
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.attn_bias_id, attn_bias, &.{
                @intCast(self.batch * self.config.num_attention_heads),
                @intCast(self.seq_len),
                @intCast(self.seq_len),
            });
        }

        if (substageUsesRope(self.stage)) {
            const rope = try graph_input_binder.QwenPlaceholderPrep.buildRopeCosSin(
                self.allocator,
                self.seq_len,
                self.config.head_dim,
                self.layer_input.rope_theta,
            );
            defer {
                self.allocator.free(rope.cos);
                self.allocator.free(rope.sin);
            }
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.rope_cos_id, rope.cos, &.{ @intCast(self.seq_len), @intCast(self.config.head_dim) });
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.rope_sin_id, rope.sin, &.{ @intCast(self.seq_len), @intCast(self.config.head_dim) });
        }

        for (self.base_params) |binding| {
            const weight = cb.getWeight(binding.name) catch |err| switch (err) {
                error.MissingWeight => if (std.mem.endsWith(u8, binding.name, ".bias")) blk: {
                    const zero_bias = try self.allocator.alloc(f32, self.config.hidden_size);
                    defer self.allocator.free(zero_bias);
                    @memset(zero_bias, 0);
                    const dims = [_]i32{@intCast(self.config.hidden_size)};
                    break :blk try cb.fromFloat32Shape(zero_bias, &dims);
                } else {
                    std.log.warn("substage VJP missing base weight: {s}", .{binding.name});
                    return err;
                },
                else => return err,
            };
            errdefer cb.free(weight);
            try rt.put(self.allocator, binding.node_id, weight);
            try owned_cts.append(self.allocator, weight);
        }

        for (self.adapters) |adapter| {
            if (findLoRALayer(lora_layers, adapter.layer_idx, adapter.module_name)) |layer| {
                try self.putOwnedTensor(cb, &rt, &owned_cts, adapter.lora_a_id, layer.A, &.{
                    @intCast(adapter.lora_a_rows),
                    @intCast(adapter.lora_a_cols),
                });
                try self.putOwnedTensor(cb, &rt, &owned_cts, adapter.lora_b_id, layer.B, &.{
                    @intCast(adapter.lora_b_rows),
                    @intCast(adapter.lora_b_cols),
                });
            } else {
                try self.putZeroOwnedTensor(cb, &rt, &owned_cts, adapter.lora_a_id, adapter.lora_a_rows, adapter.lora_a_cols);
                try self.putZeroOwnedTensor(cb, &rt, &owned_cts, adapter.lora_b_id, adapter.lora_b_rows, adapter.lora_b_cols);
            }
        }
        const runtime_input_ns = elapsedNs(total_start_ns);

        var step_result = try self.session.execute(cb, rt);
        defer step_result.deinit();

        var stage_grads_list: std.ArrayListUnmanaged(NamedGradient) = .empty;
        errdefer {
            for (stage_grads_list.items) |*grad| grad.deinit(self.allocator);
            stage_grads_list.deinit(self.allocator);
        }
        for (substageInputParamNames(self.stage)) |name| {
            const values = step_result.gradients.get(name) orelse {
                std.log.warn("substage VJP missing gradient stage={s} name={s} available_gradients={d}", .{
                    @tagName(self.stage),
                    name,
                    step_result.gradients.count(),
                });
                return error.MissingSubstageGradient;
            };
            const owned_name = try self.allocator.dupe(u8, name);
            errdefer self.allocator.free(owned_name);
            const owned_values = try self.allocator.dupe(f32, values);
            errdefer self.allocator.free(owned_values);
            try stage_grads_list.append(self.allocator, .{
                .name = owned_name,
                .values = owned_values,
            });
        }

        var adapter_grads_list: std.ArrayListUnmanaged(NamedGradient) = .empty;
        errdefer {
            for (adapter_grads_list.items) |*grad| grad.deinit(self.allocator);
            adapter_grads_list.deinit(self.allocator);
        }
        for (self.adapters) |adapter| {
            if (step_result.gradients.get(adapter.lora_a_name)) |grad_a| {
                const name = try std.fmt.allocPrint(self.allocator, "layer_{d:0>2}_{s}_A", .{ adapter.layer_idx, adapter.module_name });
                errdefer self.allocator.free(name);
                const values = try self.allocator.dupe(f32, grad_a);
                errdefer self.allocator.free(values);
                try adapter_grads_list.append(self.allocator, .{ .name = name, .values = values });
            }
            if (step_result.gradients.get(adapter.lora_b_name)) |grad_b| {
                const name = try std.fmt.allocPrint(self.allocator, "layer_{d:0>2}_{s}_B", .{ adapter.layer_idx, adapter.module_name });
                errdefer self.allocator.free(name);
                const values = try self.allocator.dupe(f32, grad_b);
                errdefer self.allocator.free(values);
                try adapter_grads_list.append(self.allocator, .{ .name = name, .values = values });
            }
        }

        const stage_grads = try stage_grads_list.toOwnedSlice(self.allocator);
        stage_grads_list = .empty;
        errdefer {
            for (stage_grads) |*grad| grad.deinit(self.allocator);
            if (stage_grads.len > 0) self.allocator.free(stage_grads);
        }

        const adapter_grads = try adapter_grads_list.toOwnedSlice(self.allocator);
        adapter_grads_list = .empty;
        errdefer {
            for (adapter_grads) |*grad| grad.deinit(self.allocator);
            if (adapter_grads.len > 0) self.allocator.free(adapter_grads);
        }

        return .{
            .allocator = self.allocator,
            .stage = self.stage,
            .loss = step_result.loss,
            .profile = .{
                .runtime_input_ns = runtime_input_ns,
                .compiled_execute_ns = step_result.profile.execute_ns,
                .compiled_extract_ns = step_result.profile.extract_ns,
                .compiled_total_ns = step_result.profile.total_ns,
                .total_ns = elapsedNs(total_start_ns),
                .partitioned_runtime = step_result.profile.partitioned_runtime,
                .mpsgraph_runtime = step_result.profile.mpsgraph_runtime,
                .partitions_executed = step_result.profile.partitions_executed,
                .backend_command_dispatches = step_result.profile.backend_command_dispatches,
                .planned_operator_dispatches = step_result.profile.planned_operator_dispatches,
                .interpreter_fallbacks = step_result.profile.interpreter_fallbacks,
                .graph_region_dispatches = step_result.profile.graph_region_dispatches,
            },
            .stage_grads = stage_grads,
            .adapter_grads = adapter_grads,
        };
    }

    fn putOwnedTensor(
        self: *ModernBertLayerBackwardSubstageSession,
        cb: *const ComputeBackend,
        rt: *std.AutoHashMapUnmanaged(NodeId, CT),
        owned_cts: *std.ArrayListUnmanaged(CT),
        node_id: NodeId,
        data: []const f32,
        dims: []const i32,
    ) !void {
        const ct = try cb.fromFloat32Shape(data, dims);
        errdefer cb.free(ct);
        try rt.put(self.allocator, node_id, ct);
        try owned_cts.append(self.allocator, ct);
    }

    fn putZeroOwnedTensor(
        self: *ModernBertLayerBackwardSubstageSession,
        cb: *const ComputeBackend,
        rt: *std.AutoHashMapUnmanaged(NodeId, CT),
        owned_cts: *std.ArrayListUnmanaged(CT),
        node_id: NodeId,
        rows: usize,
        cols: usize,
    ) !void {
        const zeros = try self.allocator.alloc(f32, rows * cols);
        defer self.allocator.free(zeros);
        @memset(zeros, 0.0);
        try self.putOwnedTensor(cb, rt, owned_cts, node_id, zeros, &.{
            @intCast(rows),
            @intCast(cols),
        });
    }
};

/// Reusable compiled synthetic-loss VJP for a ModernBERT encoder segment.
///
/// This mirrors the Go trainer's segmented backward foundation:
/// `loss = sum(segment_forward(hidden_in) * stop_gradient(upstream_grad))`.
/// Differentiating this scalar wrt `hidden_in` returns the upstream VJP for the
/// preceding segment, while differentiating wrt injected LoRA params returns
/// exact adapter gradients for the segment.
pub const ModernBertSegmentVJPSession = struct {
    allocator: std.mem.Allocator,
    config: modern_bert_graph.Config,
    batch: u32,
    seq_len: u32,
    start_layer: u32,
    end_layer: u32,
    hidden_in_id: NodeId,
    upstream_grad_id: NodeId,
    attn_bias_id: NodeId,
    rope_cos_id: NodeId,
    rope_sin_id: NodeId,
    layer_inputs: []SegmentLayerInputBinding,
    base_params: []ParamBinding,
    adapters: []AdapterBinding,
    trainable_param_names: [][]const u8,
    include_hidden_grad: bool,
    include_adapter_grads: bool,
    is_local_attention: bool,
    session: training.CompiledTrainSession,
    /// Opt-in host-feed fast path (ANTFLY_FUSED_CHUNKER_VJP_FEED_CACHE=1).
    feed_cache_enabled: bool = false,
    /// Lazily-built RoPE tables, parallel to `layer_inputs`.
    rope_cache: ?[]RopeTable = null,
    /// Lazily-cached frozen base-weight host buffers, parallel to `base_params`.
    base_weight_cache: ?[]?[]f32 = null,

    pub fn init(
        allocator: std.mem.Allocator,
        config: modern_bert_graph.Config,
        batch: u32,
        seq_len: u32,
        start_layer: u32,
        end_layer: u32,
        lora_rank: u32,
        lora_alpha: f32,
    ) !ModernBertSegmentVJPSession {
        return initWithOptions(
            allocator,
            config,
            batch,
            seq_len,
            start_layer,
            end_layer,
            lora_rank,
            lora_alpha,
            true,
        );
    }

    pub fn initWithOptions(
        allocator: std.mem.Allocator,
        config: modern_bert_graph.Config,
        batch: u32,
        seq_len: u32,
        start_layer: u32,
        end_layer: u32,
        lora_rank: u32,
        lora_alpha: f32,
        include_hidden_grad: bool,
    ) !ModernBertSegmentVJPSession {
        return initWithGradientOptions(
            allocator,
            config,
            batch,
            seq_len,
            start_layer,
            end_layer,
            lora_rank,
            lora_alpha,
            include_hidden_grad,
            true,
        );
    }

    pub fn initWithGradientOptions(
        allocator: std.mem.Allocator,
        config: modern_bert_graph.Config,
        batch: u32,
        seq_len: u32,
        start_layer: u32,
        end_layer: u32,
        lora_rank: u32,
        lora_alpha: f32,
        include_hidden_grad: bool,
        include_adapter_grads: bool,
    ) !ModernBertSegmentVJPSession {
        return initWithGradientOptionsAndStrategy(
            allocator,
            config,
            batch,
            seq_len,
            start_layer,
            end_layer,
            lora_rank,
            lora_alpha,
            include_hidden_grad,
            include_adapter_grads,
            segmentVjpExecutionStrategy(),
        );
    }

    pub fn initWithGradientOptionsAndStrategy(
        allocator: std.mem.Allocator,
        config: modern_bert_graph.Config,
        batch: u32,
        seq_len: u32,
        start_layer: u32,
        end_layer: u32,
        lora_rank: u32,
        lora_alpha: f32,
        include_hidden_grad: bool,
        include_adapter_grads: bool,
        execution_strategy: training.CompiledExecutionStrategy,
    ) !ModernBertSegmentVJPSession {
        if (lora_rank == 0) return error.LoRARankRequired;
        if (start_layer >= end_layer or end_layer > config.num_hidden_layers) return error.InvalidLayerRange;
        const segment_len = end_layer - start_layer;

        var g = Graph.init(allocator);
        defer g.deinit();
        var bld = Builder.init(&g);

        const total: i64 = @intCast(batch * seq_len);
        const hidden_i: i64 = @intCast(config.hidden_size);
        const head_dim_i: i64 = @intCast(config.head_dim);
        const bh_i: i64 = @intCast(batch * config.num_attention_heads);
        const seq_i: i64 = @intCast(seq_len);

        const hidden_in = try bld.parameter(
            "__segment_hidden_in",
            Shape.init(.f32, &.{ total, hidden_i }),
        );
        const upstream_grad = try bld.parameter(
            "__segment_upstream_grad",
            Shape.init(.f32, &.{ total, hidden_i }),
        );

        const layer_bindings = try allocator.alloc(SegmentLayerInputBinding, @intCast(segment_len));
        errdefer allocator.free(layer_bindings);
        const graph_layer_inputs = try allocator.alloc(modern_bert_graph.SegmentLayerInputs, @intCast(segment_len));
        defer allocator.free(graph_layer_inputs);

        var offset: u32 = 0;
        while (offset < segment_len) : (offset += 1) {
            const offset_usize: usize = @intCast(offset);
            const layer_idx = start_layer + offset;
            const layer_is_global = modern_bert_graph.isGlobalAttentionLayer(config, layer_idx);
            const rope_theta = modern_bert_graph.ropeThetaForLayer(config, layer_idx);

            var name_buf: [96]u8 = undefined;
            const attn_bias_name = try std.fmt.bufPrint(&name_buf, "__segment_attn_bias_layer_{d}", .{layer_idx});
            const attn_bias = try bld.parameter(
                attn_bias_name,
                Shape.init(.f32, &.{ bh_i, seq_i, seq_i }),
            );
            const rope_cos_name = try std.fmt.bufPrint(&name_buf, "__segment_rope_cos_layer_{d}", .{layer_idx});
            const rope_cos = try bld.parameter(
                rope_cos_name,
                Shape.init(.f32, &.{ seq_i, head_dim_i }),
            );
            const rope_sin_name = try std.fmt.bufPrint(&name_buf, "__segment_rope_sin_layer_{d}", .{layer_idx});
            const rope_sin = try bld.parameter(
                rope_sin_name,
                Shape.init(.f32, &.{ seq_i, head_dim_i }),
            );

            const binding = SegmentLayerInputBinding{
                .layer_idx = layer_idx,
                .attn_bias_id = attn_bias,
                .rope_cos_id = rope_cos,
                .rope_sin_id = rope_sin,
                .is_local_attention = !layer_is_global,
                .rope_theta = rope_theta,
            };
            layer_bindings[offset_usize] = binding;
            graph_layer_inputs[offset_usize] = .{
                .attn_bias_node = attn_bias,
                .rope_cos_node = rope_cos,
                .rope_sin_node = rope_sin,
            };
        }

        const segment = try modern_bert_graph.buildEncoderSegmentGraphWithLayerInputs(
            &bld,
            config,
            hidden_in,
            upstream_grad,
            graph_layer_inputs,
            batch,
            seq_len,
            start_layer,
            end_layer,
        );

        var lora_result = try graph_lora.injectLoRA(allocator, &g, .{
            .rank = lora_rank,
            .alpha = lora_alpha,
            .target_patterns = segment_lora_targets[0..],
        });
        defer lora_result.deinit();

        var base_params_list: std.ArrayListUnmanaged(ParamBinding) = .empty;
        errdefer {
            for (base_params_list.items) |*p| p.deinit(allocator);
            base_params_list.deinit(allocator);
        }
        for (lora_result.graph.parameters.items) |param_id| {
            const n = lora_result.graph.node(param_id);
            if (n.op != .parameter) continue;
            const name = lora_result.graph.parameterName(n);
            if (std.mem.startsWith(u8, name, "__")) continue;
            if (std.mem.indexOf(u8, name, ".lora_A") != null or
                std.mem.indexOf(u8, name, ".lora_B") != null)
            {
                continue;
            }
            const owned_name = try allocator.dupe(u8, name);
            errdefer allocator.free(owned_name);
            try base_params_list.append(allocator, .{
                .node_id = param_id,
                .name = owned_name,
            });
        }

        var adapter_list: std.ArrayListUnmanaged(AdapterBinding) = .empty;
        errdefer {
            for (adapter_list.items) |*a| a.deinit(allocator);
            adapter_list.deinit(allocator);
        }
        for (lora_result.adapter.adapterNames()) |info| {
            const stable_base_name = stripLoRASuffix(info.lora_a_name) orelse info.base_name;
            const mapped = moduleFromModernBertBaseName(stable_base_name) catch continue;
            const base_name = try allocator.dupe(u8, stable_base_name);
            errdefer allocator.free(base_name);
            const a_name = try allocator.dupe(u8, info.lora_a_name);
            errdefer allocator.free(a_name);
            const b_name = try allocator.dupe(u8, info.lora_b_name);
            errdefer allocator.free(b_name);
            const module_name = try allocator.dupe(u8, mapped.module_name);
            errdefer allocator.free(module_name);
            const a_shape = lora_result.graph.node(info.lora_a_id).output_shape;
            const b_shape = lora_result.graph.node(info.lora_b_id).output_shape;
            try adapter_list.append(allocator, .{
                .base_name = base_name,
                .lora_a_name = a_name,
                .lora_b_name = b_name,
                .lora_a_id = info.lora_a_id,
                .lora_b_id = info.lora_b_id,
                .lora_a_rows = @intCast(a_shape.dim(0)),
                .lora_a_cols = @intCast(a_shape.dim(1)),
                .lora_b_rows = @intCast(b_shape.dim(0)),
                .lora_b_cols = @intCast(b_shape.dim(1)),
                .layer_idx = mapped.layer_idx,
                .module_name = module_name,
            });
        }

        const hidden_grad_params: usize = if (include_hidden_grad) 1 else 0;
        const adapter_grad_params: usize = if (include_adapter_grads) adapter_list.items.len * 2 else 0;
        var trainable_names = try allocator.alloc([]const u8, hidden_grad_params + adapter_grad_params);
        errdefer allocator.free(trainable_names);
        var tn: usize = 0;
        if (include_hidden_grad) {
            trainable_names[tn] = "__segment_hidden_in";
            tn += 1;
        }
        if (include_adapter_grads) {
            for (adapter_list.items) |adapter| {
                trainable_names[tn] = adapter.lora_a_name;
                tn += 1;
                trainable_names[tn] = adapter.lora_b_name;
                tn += 1;
            }
        }

        var session = try training.CompiledTrainSession.init(
            allocator,
            &lora_result.graph,
            segment.loss_node,
            .{
                .trainable_params = trainable_names,
                .execution_strategy = execution_strategy,
            },
        );
        errdefer session.deinit();

        return .{
            .allocator = allocator,
            .config = config,
            .batch = batch,
            .seq_len = seq_len,
            .start_layer = start_layer,
            .end_layer = end_layer,
            .hidden_in_id = hidden_in,
            .upstream_grad_id = upstream_grad,
            .attn_bias_id = layer_bindings[0].attn_bias_id,
            .rope_cos_id = layer_bindings[0].rope_cos_id,
            .rope_sin_id = layer_bindings[0].rope_sin_id,
            .layer_inputs = layer_bindings,
            .base_params = try base_params_list.toOwnedSlice(allocator),
            .adapters = try adapter_list.toOwnedSlice(allocator),
            .trainable_param_names = trainable_names,
            .include_hidden_grad = include_hidden_grad,
            .include_adapter_grads = include_adapter_grads,
            .is_local_attention = segment_len == 1 and layer_bindings[0].is_local_attention,
            .session = session,
            .feed_cache_enabled = vjpFeedCacheEnabled(),
        };
    }

    pub fn deinit(self: *ModernBertSegmentVJPSession) void {
        self.session.deinit();
        self.allocator.free(self.layer_inputs);
        for (self.base_params) |*p| p.deinit(self.allocator);
        self.allocator.free(self.base_params);
        for (self.adapters) |*adapter| adapter.deinit(self.allocator);
        self.allocator.free(self.adapters);
        self.allocator.free(self.trainable_param_names);
        if (self.rope_cache) |tables| {
            for (tables) |table| {
                self.allocator.free(table.cos);
                self.allocator.free(table.sin);
            }
            self.allocator.free(tables);
        }
        if (self.base_weight_cache) |cache| {
            for (cache) |slot| {
                if (slot) |buf| self.allocator.free(buf);
            }
            self.allocator.free(cache);
        }
        self.* = undefined;
    }

    pub fn execute(
        self: *ModernBertSegmentVJPSession,
        cb: *const ComputeBackend,
        hidden_in: []const f32,
        upstream_grad: []const f32,
        attention_mask: []const f32,
        lora_layers: []fused_chunker_lora.LoRALayer,
    ) !SegmentVJPResult {
        return self.executeWithOptions(
            cb,
            hidden_in,
            upstream_grad,
            attention_mask,
            lora_layers,
            true,
        );
    }

    pub fn executeWithOptions(
        self: *ModernBertSegmentVJPSession,
        cb: *const ComputeBackend,
        hidden_in: []const f32,
        upstream_grad: []const f32,
        attention_mask: []const f32,
        lora_layers: []fused_chunker_lora.LoRALayer,
        accumulate_adapter_grads: bool,
    ) !SegmentVJPResult {
        return self.executeWithSharedBias(
            cb,
            hidden_in,
            upstream_grad,
            attention_mask,
            lora_layers,
            accumulate_adapter_grads,
            null,
        );
    }

    /// Like `executeWithOptions`, but may consume a per-step shared
    /// attention-bias cache. When the VJP feed cache is enabled the inputs
    /// are fed as stable host buffers (cached rope tables, cached frozen
    /// base weights, shared attention biases); otherwise this is identical
    /// to the classic backend-tensor path.
    pub fn executeWithSharedBias(
        self: *ModernBertSegmentVJPSession,
        cb: *const ComputeBackend,
        hidden_in: []const f32,
        upstream_grad: []const f32,
        attention_mask: []const f32,
        lora_layers: []fused_chunker_lora.LoRALayer,
        accumulate_adapter_grads: bool,
        shared_bias: ?*SharedSegmentAttnBias,
    ) !SegmentVJPResult {
        const total_start_ns = nowNs();
        const total: usize = @as(usize, self.batch) * @as(usize, self.seq_len);
        const H: usize = @intCast(self.config.hidden_size);
        if (hidden_in.len != total * H or upstream_grad.len != total * H) return error.InvalidSegmentTensorShape;
        if (attention_mask.len != total) return error.InvalidAttentionMaskShape;

        var rt = std.AutoHashMapUnmanaged(NodeId, CT){};
        defer rt.deinit(self.allocator);
        var owned_cts: std.ArrayListUnmanaged(CT) = .empty;
        defer {
            for (owned_cts.items) |ct| cb.free(ct);
            owned_cts.deinit(self.allocator);
        }
        var host_feeds: std.ArrayListUnmanaged(training.HostFeed) = .empty;
        defer host_feeds.deinit(self.allocator);
        var scratch_host: std.ArrayListUnmanaged([]f32) = .empty;
        defer {
            for (scratch_host.items) |buf| self.allocator.free(buf);
            scratch_host.deinit(self.allocator);
        }

        if (self.feed_cache_enabled) {
            try self.prepareHostFeeds(
                cb,
                &host_feeds,
                &scratch_host,
                hidden_in,
                upstream_grad,
                attention_mask,
                lora_layers,
                shared_bias,
            );
        } else {
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.hidden_in_id, hidden_in, &.{ @intCast(total), @intCast(H) });
            try self.putOwnedTensor(cb, &rt, &owned_cts, self.upstream_grad_id, upstream_grad, &.{ @intCast(total), @intCast(H) });

            for (self.layer_inputs) |binding| {
                const attn_bias = try buildSegmentAttnBias(
                    self.allocator,
                    attention_mask,
                    self.batch,
                    self.seq_len,
                    self.config.num_attention_heads,
                    binding.is_local_attention,
                    self.config.local_attention_window,
                );
                defer self.allocator.free(attn_bias);
                try self.putOwnedTensor(cb, &rt, &owned_cts, binding.attn_bias_id, attn_bias, &.{
                    @intCast(self.batch * self.config.num_attention_heads),
                    @intCast(self.seq_len),
                    @intCast(self.seq_len),
                });

                const rope = try graph_input_binder.QwenPlaceholderPrep.buildRopeCosSin(
                    self.allocator,
                    self.seq_len,
                    self.config.head_dim,
                    binding.rope_theta,
                );
                defer {
                    self.allocator.free(rope.cos);
                    self.allocator.free(rope.sin);
                }
                try self.putOwnedTensor(cb, &rt, &owned_cts, binding.rope_cos_id, rope.cos, &.{ @intCast(self.seq_len), @intCast(self.config.head_dim) });
                try self.putOwnedTensor(cb, &rt, &owned_cts, binding.rope_sin_id, rope.sin, &.{ @intCast(self.seq_len), @intCast(self.config.head_dim) });
            }

            for (self.base_params) |binding| {
                const weight = cb.getWeight(binding.name) catch |err| switch (err) {
                    error.MissingWeight => if (std.mem.endsWith(u8, binding.name, ".bias")) blk: {
                        const zero_bias = try self.allocator.alloc(f32, self.config.hidden_size);
                        defer self.allocator.free(zero_bias);
                        @memset(zero_bias, 0);
                        const dims = [_]i32{@intCast(self.config.hidden_size)};
                        break :blk try cb.fromFloat32Shape(zero_bias, &dims);
                    } else {
                        std.log.warn("segment VJP missing base weight: {s}", .{binding.name});
                        return err;
                    },
                    else => return err,
                };
                errdefer cb.free(weight);
                try rt.put(self.allocator, binding.node_id, weight);
                try owned_cts.append(self.allocator, weight);
            }

            for (self.adapters) |adapter| {
                if (findLoRALayer(lora_layers, adapter.layer_idx, adapter.module_name)) |layer| {
                    if (layer.A.len != adapter.lora_a_rows * adapter.lora_a_cols or
                        layer.B.len != adapter.lora_b_rows * adapter.lora_b_cols)
                    {
                        return error.InvalidLoRAAdapterShape;
                    }
                    try self.putOwnedTensor(cb, &rt, &owned_cts, adapter.lora_a_id, layer.A, &.{
                        @intCast(adapter.lora_a_rows),
                        @intCast(adapter.lora_a_cols),
                    });
                    try self.putOwnedTensor(cb, &rt, &owned_cts, adapter.lora_b_id, layer.B, &.{
                        @intCast(adapter.lora_b_rows),
                        @intCast(adapter.lora_b_cols),
                    });
                } else {
                    std.log.warn("segment VJP missing LoRA adapter layer={d} module={s}; binding zero adapter tensors", .{ adapter.layer_idx, adapter.module_name });
                    try self.putZeroOwnedTensor(cb, &rt, &owned_cts, adapter.lora_a_id, adapter.lora_a_rows, adapter.lora_a_cols);
                    try self.putZeroOwnedTensor(cb, &rt, &owned_cts, adapter.lora_b_id, adapter.lora_b_rows, adapter.lora_b_cols);
                }
            }
        }
        const runtime_input_ns = elapsedNs(total_start_ns);

        var step_result = if (self.feed_cache_enabled)
            try self.session.executeWithHostFeeds(cb, null, host_feeds.items)
        else
            try self.session.execute(cb, rt);
        defer step_result.deinit();

        const accumulate_start_ns = nowNs();
        const hidden_grad = if (self.include_hidden_grad) blk: {
            const hidden_grad_src = step_result.gradients.get("__segment_hidden_in") orelse return error.MissingHiddenGradient;
            const copied = try self.allocator.dupe(f32, hidden_grad_src);
            errdefer self.allocator.free(copied);
            break :blk copied;
        } else null;

        if (accumulate_adapter_grads) {
            for (self.adapters) |adapter| {
                const layer = findLoRALayer(lora_layers, adapter.layer_idx, adapter.module_name) orelse continue;
                if (step_result.gradients.get(adapter.lora_a_name)) |grad_a| {
                    if (grad_a.len != layer.grad_A.len) return error.InvalidLoRAGradShape;
                    for (layer.grad_A, grad_a) |*dst, gval| dst.* += gval;
                }
                if (step_result.gradients.get(adapter.lora_b_name)) |grad_b| {
                    if (grad_b.len != layer.grad_B.len) return error.InvalidLoRAGradShape;
                    for (layer.grad_B, grad_b) |*dst, gval| dst.* += gval;
                }
            }
        }
        const accumulate_ns = elapsedNs(accumulate_start_ns);

        return .{
            .allocator = self.allocator,
            .hidden_grad = hidden_grad,
            .loss = step_result.loss,
            .profile = .{
                .runtime_input_ns = runtime_input_ns,
                .compiled_execute_ns = step_result.profile.execute_ns,
                .compiled_extract_ns = step_result.profile.extract_ns,
                .compiled_total_ns = step_result.profile.total_ns,
                .accumulate_ns = accumulate_ns,
                .total_ns = elapsedNs(total_start_ns),
                .partitioned_runtime = step_result.profile.partitioned_runtime,
                .mpsgraph_runtime = step_result.profile.mpsgraph_runtime,
                .partitions_executed = step_result.profile.partitions_executed,
                .backend_command_dispatches = step_result.profile.backend_command_dispatches,
                .planned_operator_dispatches = step_result.profile.planned_operator_dispatches,
                .interpreter_fallbacks = step_result.profile.interpreter_fallbacks,
                .graph_region_dispatches = step_result.profile.graph_region_dispatches,
            },
        };
    }

    /// Build the host-feed list for one execution. Values are identical to
    /// the backend-tensor path; buffers with step lifetime are tracked in
    /// `scratch_host`, session-lifetime buffers live in the caches.
    fn prepareHostFeeds(
        self: *ModernBertSegmentVJPSession,
        cb: *const ComputeBackend,
        host_feeds: *std.ArrayListUnmanaged(training.HostFeed),
        scratch_host: *std.ArrayListUnmanaged([]f32),
        hidden_in: []const f32,
        upstream_grad: []const f32,
        attention_mask: []const f32,
        lora_layers: []fused_chunker_lora.LoRALayer,
        shared_bias: ?*SharedSegmentAttnBias,
    ) !void {
        try host_feeds.append(self.allocator, .{ .node_id = self.hidden_in_id, .data = hidden_in });
        try host_feeds.append(self.allocator, .{ .node_id = self.upstream_grad_id, .data = upstream_grad });

        try self.ensureRopeCache(cb);
        const rope_tables = self.rope_cache.?;
        for (self.layer_inputs, rope_tables) |binding, table| {
            const attn_bias: []const f32 = if (shared_bias) |bias_cache| blk: {
                break :blk try bias_cache.get(binding.is_local_attention);
            } else blk: {
                const owned = try buildSegmentAttnBias(
                    self.allocator,
                    attention_mask,
                    self.batch,
                    self.seq_len,
                    self.config.num_attention_heads,
                    binding.is_local_attention,
                    self.config.local_attention_window,
                );
                errdefer self.allocator.free(owned);
                try scratch_host.append(self.allocator, owned);
                break :blk owned;
            };
            try host_feeds.append(self.allocator, .{ .node_id = binding.attn_bias_id, .data = attn_bias });
            try host_feeds.append(self.allocator, .{ .node_id = binding.rope_cos_id, .data = table.cos });
            try host_feeds.append(self.allocator, .{ .node_id = binding.rope_sin_id, .data = table.sin });
        }

        // Frozen base weights: cache the host copies across steps. When the
        // backend lacks linearLoRA the training loop merges LoRA deltas into
        // the weight store around the forward pass, so skip the cross-step
        // cache and fetch fresh copies instead.
        const can_cache_base_weights = cb.vtable.linearLoRA != null;
        if (self.base_weight_cache == null) {
            const cache = try self.allocator.alloc(?[]f32, self.base_params.len);
            @memset(cache, null);
            self.base_weight_cache = cache;
        }
        const weight_cache = self.base_weight_cache.?;
        for (self.base_params, weight_cache) |binding, *slot| {
            if (slot.*) |cached| {
                try host_feeds.append(self.allocator, .{ .node_id = binding.node_id, .data = cached });
                continue;
            }
            const host = try self.fetchBaseWeightHost(cb, binding.name);
            errdefer self.allocator.free(host);
            if (can_cache_base_weights) {
                slot.* = host;
            } else {
                try scratch_host.append(self.allocator, host);
            }
            try host_feeds.append(self.allocator, .{ .node_id = binding.node_id, .data = host });
        }

        for (self.adapters) |adapter| {
            if (findLoRALayer(lora_layers, adapter.layer_idx, adapter.module_name)) |layer| {
                if (layer.A.len != adapter.lora_a_rows * adapter.lora_a_cols or
                    layer.B.len != adapter.lora_b_rows * adapter.lora_b_cols)
                {
                    return error.InvalidLoRAAdapterShape;
                }
                try host_feeds.append(self.allocator, .{ .node_id = adapter.lora_a_id, .data = layer.A });
                try host_feeds.append(self.allocator, .{ .node_id = adapter.lora_b_id, .data = layer.B });
            } else {
                std.log.warn("segment VJP missing LoRA adapter layer={d} module={s}; binding zero adapter tensors", .{ adapter.layer_idx, adapter.module_name });
                const zero_a = try self.allocator.alloc(f32, adapter.lora_a_rows * adapter.lora_a_cols);
                errdefer self.allocator.free(zero_a);
                @memset(zero_a, 0.0);
                try scratch_host.append(self.allocator, zero_a);
                const zero_b = try self.allocator.alloc(f32, adapter.lora_b_rows * adapter.lora_b_cols);
                errdefer self.allocator.free(zero_b);
                @memset(zero_b, 0.0);
                try scratch_host.append(self.allocator, zero_b);
                try host_feeds.append(self.allocator, .{ .node_id = adapter.lora_a_id, .data = zero_a });
                try host_feeds.append(self.allocator, .{ .node_id = adapter.lora_b_id, .data = zero_b });
            }
        }
    }

    fn ensureRopeCache(self: *ModernBertSegmentVJPSession, cb: *const ComputeBackend) !void {
        if (self.rope_cache != null) return;
        self.rope_cache = try buildRopeTables(
            self.allocator,
            cb,
            self.layer_inputs,
            self.seq_len,
            self.config.head_dim,
        );
    }

    fn fetchBaseWeightHost(
        self: *ModernBertSegmentVJPSession,
        cb: *const ComputeBackend,
        name: []const u8,
    ) ![]f32 {
        return fetchBaseWeightHostBuffer(self.allocator, cb, name, self.config.hidden_size);
    }

    fn putOwnedTensor(
        self: *ModernBertSegmentVJPSession,
        cb: *const ComputeBackend,
        rt: *std.AutoHashMapUnmanaged(NodeId, CT),
        owned_cts: *std.ArrayListUnmanaged(CT),
        node_id: NodeId,
        data: []const f32,
        dims: []const i32,
    ) !void {
        const ct = try cb.fromFloat32Shape(data, dims);
        errdefer cb.free(ct);
        try rt.put(self.allocator, node_id, ct);
        try owned_cts.append(self.allocator, ct);
    }

    fn putZeroOwnedTensor(
        self: *ModernBertSegmentVJPSession,
        cb: *const ComputeBackend,
        rt: *std.AutoHashMapUnmanaged(NodeId, CT),
        owned_cts: *std.ArrayListUnmanaged(CT),
        node_id: NodeId,
        rows: usize,
        cols: usize,
    ) !void {
        const zeros = try self.allocator.alloc(f32, rows * cols);
        defer self.allocator.free(zeros);
        @memset(zeros, 0.0);
        try self.putOwnedTensor(cb, rt, owned_cts, node_id, zeros, &.{
            @intCast(rows),
            @intCast(cols),
        });
    }
};

// ----------------------------------------------------------------------------
// ModernBERT compiled segment forward session
// ----------------------------------------------------------------------------

/// Per-layer capture buffers produced by one compiled segment forward.
/// These mirror what the eager capture path downloads:
///   attn_normed   — Q/K/V projection input ("query_proj"/"key_proj"/"value_proj")
///   attn_merged   — attention output before attn.Wo ("out_proj")
///   mlp_activated — GeGLU activation before mlp.Wo ("wo")
pub const SegmentForwardLayerCapture = struct {
    layer_idx: u32,
    attn_normed: []f32,
    attn_merged: []f32,
    mlp_activated: []f32,
    /// [total * hidden] — this layer's output (the next layer's input).
    hidden_out: []f32,
};

pub const SegmentForwardResult = struct {
    allocator: std.mem.Allocator,
    /// [total * hidden] — segment output hidden state (alias of the last
    /// layer's `hidden_out`; owned by `layers`, not freed separately).
    hidden_out: []const f32,
    layers: []SegmentForwardLayerCapture,

    pub fn deinit(self: *SegmentForwardResult) void {
        for (self.layers) |capture| {
            self.allocator.free(capture.attn_normed);
            self.allocator.free(capture.attn_merged);
            self.allocator.free(capture.mlp_activated);
            self.allocator.free(capture.hidden_out);
        }
        self.allocator.free(self.layers);
        self.* = undefined;
    }
};

/// Compiled forward pass over a ModernBERT encoder segment, built from the
/// SAME `modern_bert_graph.encoderLayer` body the segment VJP sessions use
/// (so a compiled forward matches the forward recompute the VJP graphs
/// already perform), with LoRA injected via the same target patterns.
///
/// Executes through the MPSGraph runtime only; init fails when the segment
/// graph cannot be lowered, letting callers fall back to the eager forward.
pub const ModernBertSegmentForwardSession = struct {
    allocator: std.mem.Allocator,
    config: modern_bert_graph.Config,
    batch: u32,
    seq_len: u32,
    start_layer: u32,
    end_layer: u32,
    hidden_in_id: NodeId,
    layer_inputs: []SegmentLayerInputBinding,
    base_params: []ParamBinding,
    adapters: []AdapterBinding,
    compiled: mpsgraph_executor.CompiledGraph,
    rope_cache: ?[]RopeTable = null,
    base_weight_cache: ?[]?[]f32 = null,

    pub fn init(
        allocator: std.mem.Allocator,
        config: modern_bert_graph.Config,
        batch: u32,
        seq_len: u32,
        start_layer: u32,
        end_layer: u32,
        lora_rank: u32,
        lora_alpha: f32,
    ) !ModernBertSegmentForwardSession {
        if (lora_rank == 0) return error.LoRARankRequired;
        if (start_layer >= end_layer or end_layer > config.num_hidden_layers) return error.InvalidLayerRange;
        const segment_len = end_layer - start_layer;

        var g = Graph.init(allocator);
        defer g.deinit();
        var bld = Builder.init(&g);

        const total: i64 = @intCast(batch * seq_len);
        const hidden_i: i64 = @intCast(config.hidden_size);
        const head_dim_i: i64 = @intCast(config.head_dim);
        const bh_i: i64 = @intCast(batch * config.num_attention_heads);
        const seq_i: i64 = @intCast(seq_len);

        const hidden_in = try bld.parameter(
            "__segment_hidden_in",
            Shape.init(.f32, &.{ total, hidden_i }),
        );

        const layer_bindings = try allocator.alloc(SegmentLayerInputBinding, @intCast(segment_len));
        errdefer allocator.free(layer_bindings);
        const graph_layer_inputs = try allocator.alloc(modern_bert_graph.SegmentLayerInputs, @intCast(segment_len));
        defer allocator.free(graph_layer_inputs);

        var offset: u32 = 0;
        while (offset < segment_len) : (offset += 1) {
            const offset_usize: usize = @intCast(offset);
            const layer_idx = start_layer + offset;
            const layer_is_global = modern_bert_graph.isGlobalAttentionLayer(config, layer_idx);
            const rope_theta = modern_bert_graph.ropeThetaForLayer(config, layer_idx);

            var name_buf: [96]u8 = undefined;
            const attn_bias_name = try std.fmt.bufPrint(&name_buf, "__segment_attn_bias_layer_{d}", .{layer_idx});
            const attn_bias = try bld.parameter(
                attn_bias_name,
                Shape.init(.f32, &.{ bh_i, seq_i, seq_i }),
            );
            const rope_cos_name = try std.fmt.bufPrint(&name_buf, "__segment_rope_cos_layer_{d}", .{layer_idx});
            const rope_cos = try bld.parameter(
                rope_cos_name,
                Shape.init(.f32, &.{ seq_i, head_dim_i }),
            );
            const rope_sin_name = try std.fmt.bufPrint(&name_buf, "__segment_rope_sin_layer_{d}", .{layer_idx});
            const rope_sin = try bld.parameter(
                rope_sin_name,
                Shape.init(.f32, &.{ seq_i, head_dim_i }),
            );

            layer_bindings[offset_usize] = .{
                .layer_idx = layer_idx,
                .attn_bias_id = attn_bias,
                .rope_cos_id = rope_cos,
                .rope_sin_id = rope_sin,
                .is_local_attention = !layer_is_global,
                .rope_theta = rope_theta,
            };
            graph_layer_inputs[offset_usize] = .{
                .attn_bias_node = attn_bias,
                .rope_cos_node = rope_cos,
                .rope_sin_node = rope_sin,
            };
        }

        const layer_nodes = try allocator.alloc(modern_bert_graph.SegmentForwardLayerNodes, @intCast(segment_len));
        defer allocator.free(layer_nodes);

        _ = try modern_bert_graph.buildEncoderSegmentForwardGraph(
            &bld,
            config,
            hidden_in,
            graph_layer_inputs,
            layer_nodes,
            batch,
            seq_len,
            start_layer,
            end_layer,
        );

        // Output order consumed by execute(): per layer
        // [attn_normed, attn_merged, mlp_activated, hidden_out].
        for (layer_nodes) |nodes| {
            try g.markOutput(nodes.attn_normed_node);
            try g.markOutput(nodes.attn_merged_node);
            try g.markOutput(nodes.mlp_activated_node);
            try g.markOutput(nodes.hidden_out_node);
        }

        var lora_result = try graph_lora.injectLoRA(allocator, &g, .{
            .rank = lora_rank,
            .alpha = lora_alpha,
            .target_patterns = segment_lora_targets[0..],
        });
        defer lora_result.deinit();

        var base_params_list: std.ArrayListUnmanaged(ParamBinding) = .empty;
        errdefer {
            for (base_params_list.items) |*p| p.deinit(allocator);
            base_params_list.deinit(allocator);
        }
        for (lora_result.graph.parameters.items) |param_id| {
            const n = lora_result.graph.node(param_id);
            if (n.op != .parameter) continue;
            const name = lora_result.graph.parameterName(n);
            if (std.mem.startsWith(u8, name, "__")) continue;
            if (std.mem.indexOf(u8, name, ".lora_A") != null or
                std.mem.indexOf(u8, name, ".lora_B") != null)
            {
                continue;
            }
            const owned_name = try allocator.dupe(u8, name);
            errdefer allocator.free(owned_name);
            try base_params_list.append(allocator, .{
                .node_id = param_id,
                .name = owned_name,
            });
        }

        var adapter_list: std.ArrayListUnmanaged(AdapterBinding) = .empty;
        errdefer {
            for (adapter_list.items) |*a| a.deinit(allocator);
            adapter_list.deinit(allocator);
        }
        for (lora_result.adapter.adapterNames()) |info| {
            const stable_base_name = stripLoRASuffix(info.lora_a_name) orelse info.base_name;
            const mapped = moduleFromModernBertBaseName(stable_base_name) catch continue;
            const base_name = try allocator.dupe(u8, stable_base_name);
            errdefer allocator.free(base_name);
            const a_name = try allocator.dupe(u8, info.lora_a_name);
            errdefer allocator.free(a_name);
            const b_name = try allocator.dupe(u8, info.lora_b_name);
            errdefer allocator.free(b_name);
            const module_name = try allocator.dupe(u8, mapped.module_name);
            errdefer allocator.free(module_name);
            const a_shape = lora_result.graph.node(info.lora_a_id).output_shape;
            const b_shape = lora_result.graph.node(info.lora_b_id).output_shape;
            try adapter_list.append(allocator, .{
                .base_name = base_name,
                .lora_a_name = a_name,
                .lora_b_name = b_name,
                .lora_a_id = info.lora_a_id,
                .lora_b_id = info.lora_b_id,
                .lora_a_rows = @intCast(a_shape.dim(0)),
                .lora_a_cols = @intCast(a_shape.dim(1)),
                .lora_b_rows = @intCast(b_shape.dim(0)),
                .lora_b_cols = @intCast(b_shape.dim(1)),
                .layer_idx = mapped.layer_idx,
                .module_name = module_name,
            });
        }

        var compiled = try mpsgraph_executor.CompiledGraph.compile(allocator, &lora_result.graph);
        errdefer compiled.deinit();

        return .{
            .allocator = allocator,
            .config = config,
            .batch = batch,
            .seq_len = seq_len,
            .start_layer = start_layer,
            .end_layer = end_layer,
            .hidden_in_id = hidden_in,
            .layer_inputs = layer_bindings,
            .base_params = try base_params_list.toOwnedSlice(allocator),
            .adapters = try adapter_list.toOwnedSlice(allocator),
            .compiled = compiled,
        };
    }

    pub fn deinit(self: *ModernBertSegmentForwardSession) void {
        self.compiled.deinit();
        self.allocator.free(self.layer_inputs);
        for (self.base_params) |*p| p.deinit(self.allocator);
        self.allocator.free(self.base_params);
        for (self.adapters) |*adapter| adapter.deinit(self.allocator);
        self.allocator.free(self.adapters);
        if (self.rope_cache) |tables| {
            for (tables) |table| {
                self.allocator.free(table.cos);
                self.allocator.free(table.sin);
            }
            self.allocator.free(tables);
        }
        if (self.base_weight_cache) |cache| {
            for (cache) |slot| {
                if (slot) |buf| self.allocator.free(buf);
            }
            self.allocator.free(cache);
        }
        self.* = undefined;
    }

    pub fn execute(
        self: *ModernBertSegmentForwardSession,
        cb: *const ComputeBackend,
        hidden_in: []const f32,
        attention_mask: []const f32,
        lora_layers: []fused_chunker_lora.LoRALayer,
        shared_bias: ?*SharedSegmentAttnBias,
    ) !SegmentForwardResult {
        const total: usize = @as(usize, self.batch) * @as(usize, self.seq_len);
        const H: usize = @intCast(self.config.hidden_size);
        const I: usize = @intCast(self.config.intermediate_size);
        if (hidden_in.len != total * H) return error.InvalidSegmentTensorShape;
        if (attention_mask.len != total) return error.InvalidAttentionMaskShape;

        var host_feeds: std.ArrayListUnmanaged(mpsgraph_executor.HostInput) = .empty;
        defer host_feeds.deinit(self.allocator);
        var scratch_host: std.ArrayListUnmanaged([]f32) = .empty;
        defer {
            for (scratch_host.items) |buf| self.allocator.free(buf);
            scratch_host.deinit(self.allocator);
        }

        try host_feeds.append(self.allocator, .{ .node_id = self.hidden_in_id, .data = hidden_in });

        if (self.rope_cache == null) {
            self.rope_cache = try buildRopeTables(
                self.allocator,
                cb,
                self.layer_inputs,
                self.seq_len,
                self.config.head_dim,
            );
        }
        for (self.layer_inputs, self.rope_cache.?) |binding, table| {
            const attn_bias: []const f32 = if (shared_bias) |bias_cache| blk: {
                break :blk try bias_cache.get(binding.is_local_attention);
            } else blk: {
                const owned = try buildSegmentAttnBias(
                    self.allocator,
                    attention_mask,
                    self.batch,
                    self.seq_len,
                    self.config.num_attention_heads,
                    binding.is_local_attention,
                    self.config.local_attention_window,
                );
                errdefer self.allocator.free(owned);
                try scratch_host.append(self.allocator, owned);
                break :blk owned;
            };
            try host_feeds.append(self.allocator, .{ .node_id = binding.attn_bias_id, .data = attn_bias });
            try host_feeds.append(self.allocator, .{ .node_id = binding.rope_cos_id, .data = table.cos });
            try host_feeds.append(self.allocator, .{ .node_id = binding.rope_sin_id, .data = table.sin });
        }

        const can_cache_base_weights = cb.vtable.linearLoRA != null;
        if (self.base_weight_cache == null) {
            const cache = try self.allocator.alloc(?[]f32, self.base_params.len);
            @memset(cache, null);
            self.base_weight_cache = cache;
        }
        const weight_cache = self.base_weight_cache.?;
        for (self.base_params, weight_cache) |binding, *slot| {
            if (slot.*) |cached| {
                try host_feeds.append(self.allocator, .{ .node_id = binding.node_id, .data = cached });
                continue;
            }
            const host = try fetchBaseWeightHostBuffer(self.allocator, cb, binding.name, self.config.hidden_size);
            errdefer self.allocator.free(host);
            if (can_cache_base_weights) {
                slot.* = host;
            } else {
                try scratch_host.append(self.allocator, host);
            }
            try host_feeds.append(self.allocator, .{ .node_id = binding.node_id, .data = host });
        }

        for (self.adapters) |adapter| {
            if (findLoRALayer(lora_layers, adapter.layer_idx, adapter.module_name)) |layer| {
                if (layer.A.len != adapter.lora_a_rows * adapter.lora_a_cols or
                    layer.B.len != adapter.lora_b_rows * adapter.lora_b_cols)
                {
                    return error.InvalidLoRAAdapterShape;
                }
                try host_feeds.append(self.allocator, .{ .node_id = adapter.lora_a_id, .data = layer.A });
                try host_feeds.append(self.allocator, .{ .node_id = adapter.lora_b_id, .data = layer.B });
            } else {
                const zero_a = try self.allocator.alloc(f32, adapter.lora_a_rows * adapter.lora_a_cols);
                errdefer self.allocator.free(zero_a);
                @memset(zero_a, 0.0);
                try scratch_host.append(self.allocator, zero_a);
                const zero_b = try self.allocator.alloc(f32, adapter.lora_b_rows * adapter.lora_b_cols);
                errdefer self.allocator.free(zero_b);
                @memset(zero_b, 0.0);
                try scratch_host.append(self.allocator, zero_b);
                try host_feeds.append(self.allocator, .{ .node_id = adapter.lora_a_id, .data = zero_a });
                try host_feeds.append(self.allocator, .{ .node_id = adapter.lora_b_id, .data = zero_b });
            }
        }

        var exec_result = try self.compiled.executeWithHostInputs(
            self.allocator,
            cb,
            null,
            host_feeds.items,
        );
        var exec_owned = true;
        defer if (exec_owned) exec_result.deinit();

        const num_layers: usize = @intCast(self.end_layer - self.start_layer);
        if (exec_result.outputs.len != num_layers * 4) return error.InvalidSegmentForwardOutputs;
        for (0..num_layers) |i| {
            if (exec_result.outputs[i * 4 + 0].len != total * H or
                exec_result.outputs[i * 4 + 1].len != total * H or
                exec_result.outputs[i * 4 + 2].len != total * I or
                exec_result.outputs[i * 4 + 3].len != total * H)
            {
                return error.InvalidSegmentForwardOutputs;
            }
        }

        const layers = try self.allocator.alloc(SegmentForwardLayerCapture, num_layers);
        errdefer self.allocator.free(layers);
        for (layers, 0..) |*capture, i| {
            capture.* = .{
                .layer_idx = self.start_layer + @as(u32, @intCast(i)),
                .attn_normed = exec_result.outputs[i * 4 + 0],
                .attn_merged = exec_result.outputs[i * 4 + 1],
                .mlp_activated = exec_result.outputs[i * 4 + 2],
                .hidden_out = exec_result.outputs[i * 4 + 3],
            };
        }

        // Take ownership of the output slices; free only the container.
        self.allocator.free(exec_result.outputs);
        exec_owned = false;

        return .{
            .allocator = self.allocator,
            .hidden_out = layers[num_layers - 1].hidden_out,
            .layers = layers,
        };
    }
};

fn nowNs() u64 {
    return platform.time.monotonicNs();
}

fn elapsedNs(start_ns: u64) u64 {
    const end_ns = nowNs();
    return if (end_ns > start_ns) end_ns - start_ns else 0;
}

/// Build the per-layer RoPE cos/sin tables for a segment session, one table
/// per layer binding. Tables are pure functions of (seq_len, head_dim, theta)
/// and safe to cache for the lifetime of the session.
fn buildRopeTables(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    layer_inputs: []const SegmentLayerInputBinding,
    seq_len: u32,
    head_dim: u32,
) ![]RopeTable {
    const tables = try allocator.alloc(RopeTable, layer_inputs.len);
    var filled: usize = 0;
    errdefer {
        for (tables[0..filled]) |table| {
            allocator.free(table.cos);
            allocator.free(table.sin);
        }
        allocator.free(tables);
    }
    for (layer_inputs, tables) |binding, *table| {
        // Reuse an already-built table when this layer shares its theta with
        // an earlier layer (ModernBERT alternates between exactly two thetas).
        var reused = false;
        for (layer_inputs[0..filled], tables[0..filled]) |prev_binding, prev_table| {
            if (prev_binding.rope_theta == binding.rope_theta) {
                const cos = try allocator.dupe(f32, prev_table.cos);
                errdefer allocator.free(cos);
                const sin = try allocator.dupe(f32, prev_table.sin);
                table.* = .{ .cos = cos, .sin = sin };
                reused = true;
                break;
            }
        }
        if (reused) {
            filled += 1;
            continue;
        }
        table.* = try buildBackendRopeTable(allocator, cb, seq_len, head_dim, binding.rope_theta) orelse blk: {
            const rope = try graph_input_binder.QwenPlaceholderPrep.buildRopeCosSin(
                allocator,
                seq_len,
                head_dim,
                binding.rope_theta,
            );
            break :blk .{ .cos = rope.cos, .sin = rope.sin };
        };
        filled += 1;
    }
    return tables;
}

/// Capture the backend's own RoPE cos/sin values so the graph-side rotation
/// multiplies by EXACTLY the factors the eager `cb.rope` kernel applies.
///
/// The eager Metal kernel computes `angle = pos * pow(theta, -2j/dim)` in f32
/// per element; a host `math.pow` differs from the kernel's `pow` by ~1 ulp,
/// and at positions of a few hundred that 1-ulp frequency difference becomes
/// a ~3e-5 absolute cos/sin difference — a systematic per-layer forward error
/// that compounds across the encoder stack.
///
/// Rotating the basis pattern (x0=1, x1=0) per pair returns
/// `out0 = cos, out1 = sin` with the kernel's exact rounding.
/// Returns null when the backend cannot run the rope op (caller falls back to
/// the host-computed table).
fn buildBackendRopeTable(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    seq_len: u32,
    head_dim: u32,
    theta: f32,
) !?RopeTable {
    const sl: usize = @intCast(seq_len);
    const hd: usize = @intCast(head_dim);
    const half = hd / 2;
    if (half == 0) return null;
    const total = sl * hd;

    const basis = try allocator.alloc(f32, total);
    defer allocator.free(basis);
    for (0..sl) |pos| {
        for (0..half) |j| {
            basis[pos * hd + 2 * j] = 1.0;
            basis[pos * hd + 2 * j + 1] = 0.0;
        }
    }

    const basis_ct = cb.fromFloat32Shape(basis, &.{ @intCast(sl), @intCast(hd) }) catch return null;
    defer cb.free(basis_ct);
    // consecutive_pairs=true matches the eager ModernBERT forward
    // (`cb.rope(..., true)` in modern_bert.zig).
    const roped_ct = cb.rope(basis_ct, sl, hd, hd, theta, 1.0, 0, true) catch return null;
    defer cb.free(roped_ct);
    const roped = cb.toFloat32(roped_ct, allocator) catch return null;
    defer allocator.free(roped);
    if (roped.len != total) return null;

    const cos_buf = try allocator.alloc(f32, total);
    errdefer allocator.free(cos_buf);
    const sin_buf = try allocator.alloc(f32, total);
    errdefer allocator.free(sin_buf);
    for (0..sl) |pos| {
        for (0..half) |j| {
            const c = roped[pos * hd + 2 * j];
            const s = roped[pos * hd + 2 * j + 1];
            // Split-convention layout consumed by the graph rope lowering:
            // pair j reads column j of the [seq, head_dim] table (the second
            // half duplicates the first, mirroring buildRopeCosSin).
            cos_buf[pos * hd + j] = c;
            sin_buf[pos * hd + j] = s;
            cos_buf[pos * hd + half + j] = c;
            sin_buf[pos * hd + half + j] = s;
        }
    }
    return .{ .cos = cos_buf, .sin = sin_buf };
}

/// Download one frozen base weight as an owned host f32 buffer, with the
/// same missing-bias fallback as the backend-tensor path.
fn fetchBaseWeightHostBuffer(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    name: []const u8,
    hidden_size: u32,
) ![]f32 {
    const weight = cb.getWeight(name) catch |err| {
        if (err == error.MissingWeight and std.mem.endsWith(u8, name, ".bias")) {
            const zero_bias = try allocator.alloc(f32, hidden_size);
            @memset(zero_bias, 0);
            return zero_bias;
        }
        if (err == error.MissingWeight) {
            std.log.warn("segment session missing base weight: {s}", .{name});
        }
        return err;
    };
    defer cb.free(weight);
    return cb.toFloat32(weight, allocator);
}

fn buildSegmentAttnBias(
    allocator: std.mem.Allocator,
    attention_mask: []const f32,
    batch: u32,
    seq_len: u32,
    num_heads: u32,
    is_local_attention: bool,
    local_attention_window: u32,
) ![]f32 {
    const sl: usize = @intCast(seq_len);
    const b: usize = @intCast(batch);
    const nh: usize = @intCast(num_heads);
    if (attention_mask.len != b * sl) return error.InvalidAttentionMaskShape;

    const total = b * nh * sl * sl;
    const bias = try allocator.alloc(f32, total);
    errdefer allocator.free(bias);

    const window_half: usize = @intCast(local_attention_window / 2);
    var idx: usize = 0;
    for (0..b) |bi| {
        for (0..nh) |_| {
            for (0..sl) |qi| {
                for (0..sl) |ki| {
                    const mask_idx = bi * sl + ki;
                    const diff = if (qi >= ki) qi - ki else ki - qi;
                    const outside_window = is_local_attention and diff > window_half;
                    bias[idx] = if (attention_mask[mask_idx] < 0.5 or outside_window) -1.0e9 else 0.0;
                    idx += 1;
                }
            }
        }
    }

    return bias;
}

const ModuleMapping = struct {
    layer_idx: u32,
    module_name: []const u8,
};

fn moduleFromModernBertBaseName(base_name: []const u8) !ModuleMapping {
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, base_name, prefix)) return error.UnsupportedLoRABaseName;
    var rest = base_name[prefix.len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.UnsupportedLoRABaseName;
    const layer_idx = try std.fmt.parseInt(u32, rest[0..dot], 10);
    rest = rest[dot + 1 ..];

    if (std.mem.eql(u8, rest, "attn.query_proj.weight")) return .{ .layer_idx = layer_idx, .module_name = "query_proj" };
    if (std.mem.eql(u8, rest, "attn.key_proj.weight")) return .{ .layer_idx = layer_idx, .module_name = "key_proj" };
    if (std.mem.eql(u8, rest, "attn.value_proj.weight")) return .{ .layer_idx = layer_idx, .module_name = "value_proj" };
    if (std.mem.eql(u8, rest, "attn.Wo.weight")) return .{ .layer_idx = layer_idx, .module_name = "out_proj" };
    if (std.mem.eql(u8, rest, "mlp.Wo.weight")) return .{ .layer_idx = layer_idx, .module_name = "wo" };
    return error.UnsupportedLoRABaseName;
}

fn stripLoRASuffix(name: []const u8) ?[]const u8 {
    if (std.mem.endsWith(u8, name, ".lora_A")) return name[0 .. name.len - ".lora_A".len];
    if (std.mem.endsWith(u8, name, ".lora_B")) return name[0 .. name.len - ".lora_B".len];
    return null;
}

fn findLoRALayer(
    lora_layers: []fused_chunker_lora.LoRALayer,
    layer_idx: u32,
    module_name: []const u8,
) ?*fused_chunker_lora.LoRALayer {
    for (lora_layers) |*layer| {
        if (layer.layer_idx == layer_idx and std.mem.eql(u8, layer.module_name, module_name)) return layer;
    }
    return null;
}

// ----------------------------------------------------------------------------
// backwardLoRA
// ----------------------------------------------------------------------------

/// Accumulate LoRA gradients for all captured activations given the gradient
/// of the loss with respect to the final encoder output.
///
/// Parameters
/// ----------
/// allocator   — scratch allocator forwarded to lora.accumulateLinearLoRAGrads
/// activations — captured inputs from the encoder forward pass
/// d_output    — [seq_len * hidden_size] — dL/d(encoder_final_output)
/// lora_layers — mutable slice of LoRALayer; grad_A and grad_B are accumulated
/// alpha       — LoRA scaling factor (usually LoRAConfig.alpha)
///
/// For each activation, the function looks up the matching LoRALayer by
/// (layer_idx, module_name) and calls accumulateLayerLoRAGrads.  Activations
/// whose corresponding LoRALayer cannot be found are silently skipped.
pub fn backwardLoRA(
    allocator: std.mem.Allocator,
    activations: *const EncoderActivations,
    d_output: []const f32,
    lora_layers: []fused_chunker_lora.LoRALayer,
    alpha: f32,
) !void {
    for (activations.activations.items) |*act| {
        // Find the LoRALayer matching this (layer_idx, module_name) pair.
        const matched_layer: ?*fused_chunker_lora.LoRALayer = blk: {
            for (lora_layers) |*ll| {
                if (ll.layer_idx == act.layer_idx and
                    std.mem.eql(u8, ll.module_name, act.module_name))
                {
                    break :blk ll;
                }
            }
            break :blk null;
        };

        const lora_layer = matched_layer orelse continue;

        // Build an output-gradient slice of the right length for this projection.
        // Because we are using the encoder's final output gradient as an approximation
        // we need a buffer of [seq_len * out_features].  When out_features equals
        // hidden_size (the common case for query/value projections) d_output can be
        // used directly.  Otherwise we allocate a zero-padded / truncated view.
        const needed = act.seq_len * act.out_features;
        const out_grad: []const f32 = if (d_output.len == needed) blk: {
            break :blk d_output;
        } else blk: {
            const buf = try allocator.alloc(f32, needed);
            errdefer allocator.free(buf);
            const copy_len = @min(d_output.len, needed);
            @memcpy(buf[0..copy_len], d_output[0..copy_len]);
            if (copy_len < needed) @memset(buf[copy_len..], 0);
            break :blk buf;
        };
        defer if (out_grad.ptr != d_output.ptr) allocator.free(@constCast(out_grad));

        try accumulateLayerLoRAGrads(allocator, act, out_grad, lora_layer, alpha);
    }
}

// ----------------------------------------------------------------------------
// accumulateLayerLoRAGrads  (private helper)
// ----------------------------------------------------------------------------

/// Accumulate grad_A and grad_B for one (activation, LoRALayer) pair.
///
/// Delegates to lora.accumulateLinearLoRAGrads which expects:
///   grad_a         [rank * in_features]
///   grad_b         [out_features * rank]
///   input_rows     seq_len
///   input_cols     in_features
///   inputs         [seq_len * in_features]
///   output_cols    out_features
///   output_grads   [seq_len * out_features]
///   adapter_a      Matrix — A: [rank, in_features]
///   adapter_b      Matrix — B: [out_features, rank]
///   alpha          f32
fn accumulateLayerLoRAGrads(
    allocator: std.mem.Allocator,
    act: *const LayerActivation,
    output_grad: []const f32,
    lora_layer: *fused_chunker_lora.LoRALayer,
    alpha: f32,
) !void {
    const rank = lora_layer.A.len / lora_layer.in_features;
    if (rank == 0) return;

    const adapter_a = try allocator.alloc(f32, lora_layer.in_features * rank);
    defer allocator.free(adapter_a);
    const adapter_b = try allocator.alloc(f32, rank * lora_layer.out_features);
    defer allocator.free(adapter_b);
    const grad_a = try allocator.alloc(f32, lora_layer.in_features * rank);
    defer allocator.free(grad_a);
    const grad_b = try allocator.alloc(f32, rank * lora_layer.out_features);
    defer allocator.free(grad_b);

    for (0..rank) |r| {
        for (0..lora_layer.in_features) |i| {
            adapter_a[i * rank + r] = lora_layer.A[r * lora_layer.in_features + i];
        }
    }
    for (0..lora_layer.out_features) |o| {
        for (0..rank) |r| {
            adapter_b[r * lora_layer.out_features + o] = lora_layer.B[o * rank + r];
        }
    }
    @memset(grad_a, 0);
    @memset(grad_b, 0);

    lora.accumulateLinearLoRAGrads(
        grad_a,
        grad_b,
        act.seq_len, // input_rows
        act.in_features,
        act.input, // inputs
        act.out_features,
        output_grad, // output_grads
        .{ .rows = lora_layer.in_features, .cols = rank, .data = adapter_a },
        .{ .rows = rank, .cols = lora_layer.out_features, .data = adapter_b },
        alpha,
    );

    for (0..lora_layer.in_features) |i| {
        for (0..rank) |r| {
            lora_layer.grad_A[r * lora_layer.in_features + i] += grad_a[i * rank + r];
        }
    }
    for (0..rank) |r| {
        for (0..lora_layer.out_features) |o| {
            lora_layer.grad_B[o * rank + r] += grad_b[r * lora_layer.out_features + o];
        }
    }
}

// ----------------------------------------------------------------------------
// backwardLoRADirect
// ----------------------------------------------------------------------------

/// Apply LoRA backprop using captures from modern_bert.ActivationBuffer.
///
/// Accepts raw slices extracted from an ActivationBuffer rather than importing
/// modern_bert.zig (to avoid circular dependencies).
///
/// cap_layer_indices: []const u32    — layer_idx per capture
/// cap_module_names:  []const []const u8  — module_name per capture
/// cap_inputs:        []const []const f32 — input[i] is [total * in_features] for capture i
/// cap_in_features:   []const usize
/// cap_out_features:  []const usize
/// d_output:          []const f32    — dL/d(encoder_output), [total * hidden_size]
/// lora_layers:       []LoRALayer    — grad_A/grad_B accumulated in-place
/// alpha:             f32
pub fn backwardLoRADirect(
    cb: ops.ComputeBackend,
    allocator: std.mem.Allocator,
    cap_layer_indices: []const u32,
    cap_module_names: []const []const u8,
    cap_inputs: []const []const f32,
    cap_in_features: []const usize,
    cap_out_features: []const usize,
    d_output: []const f32,
    lora_layers: []fused_chunker_lora.LoRALayer,
    alpha: f32,
) !void {
    const n = cap_layer_indices.len;
    std.debug.assert(cap_module_names.len == n);
    std.debug.assert(cap_inputs.len == n);
    std.debug.assert(cap_in_features.len == n);
    std.debug.assert(cap_out_features.len == n);

    for (0..n) |i| {
        const layer_idx = cap_layer_indices[i];
        const module_name = cap_module_names[i];
        const input = cap_inputs[i];
        const in_features = cap_in_features[i];
        const out_features = cap_out_features[i];

        // Find the LoRALayer matching this (layer_idx, module_name) pair.
        const matched_layer: ?*fused_chunker_lora.LoRALayer = blk: {
            for (lora_layers) |*ll| {
                if (ll.layer_idx == layer_idx and
                    std.mem.eql(u8, ll.module_name, module_name))
                {
                    break :blk ll;
                }
            }
            break :blk null;
        };

        const lora_layer = matched_layer orelse continue;

        // Derive seq_len from the input buffer length and in_features.
        const seq_len = if (in_features > 0) input.len / in_features else 0;

        // Build an output-gradient slice of [seq_len * out_features].
        const needed = seq_len * out_features;
        const out_grad: []const f32 = if (d_output.len == needed) blk: {
            break :blk d_output;
        } else blk: {
            const buf = try allocator.alloc(f32, needed);
            errdefer allocator.free(buf);
            const copy_len = @min(d_output.len, needed);
            @memcpy(buf[0..copy_len], d_output[0..copy_len]);
            if (copy_len < needed) @memset(buf[copy_len..], 0);
            break :blk buf;
        };
        defer if (out_grad.ptr != d_output.ptr) allocator.free(@constCast(out_grad));

        // Try the GPU-accelerated path first; fall back to the CPU path if the
        // backend does not provide it.
        const rank = if (in_features > 0) lora_layer.A.len / in_features else 0;
        const scale = if (rank > 0) alpha / @as(f32, @floatFromInt(rank)) else 0.0;
        const gpu_used = try cb.accumulateLoRAGrads(
            allocator,
            lora_layer.grad_A,
            lora_layer.grad_B,
            input,
            out_grad,
            lora_layer.A,
            lora_layer.B,
            seq_len,
            in_features,
            out_features,
            rank,
            scale,
        );
        if (!gpu_used) {
            // Build a synthetic LayerActivation from the raw slices so we can
            // reuse accumulateLayerLoRAGrads without duplicating BLAS logic.
            // The input slice is borrowed (not owned); we do not free it here.
            const act = LayerActivation{
                .layer_idx = layer_idx,
                .module_name = module_name,
                .input = @constCast(input),
                .in_features = in_features,
                .out_features = out_features,
                .seq_len = seq_len,
            };
            try accumulateLayerLoRAGrads(allocator, &act, out_grad, lora_layer, alpha);
        }
    }
}

/// Backpropagate through a row-wise LayerNorm:
/// y = ((x - mean) / sqrt(var + eps)) * gamma + beta.
///
/// Returns an owned [rows * hidden] gradient with respect to x. This is used
/// as a cheap chain-rule correction before the current direct LoRA-gradient
/// approximation consumes dL/d(final_encoder_output).
pub fn backpropLayerNormInput(
    allocator: std.mem.Allocator,
    grad_out: []const f32,
    input: []const f32,
    gamma: []const f32,
    rows: usize,
    hidden: usize,
    eps: f32,
) ![]f32 {
    if (grad_out.len != rows * hidden or input.len != rows * hidden or gamma.len != hidden) {
        return error.InvalidLayerNormBackwardShape;
    }

    const grad_in = try allocator.alloc(f32, rows * hidden);
    errdefer allocator.free(grad_in);
    if (hidden == 0) return grad_in;

    const hidden_f: f32 = @floatFromInt(hidden);
    for (0..rows) |row| {
        const base = row * hidden;
        const x = input[base .. base + hidden];
        const dy = grad_out[base .. base + hidden];
        const dx = grad_in[base .. base + hidden];

        var mean: f32 = 0.0;
        for (x) |v| mean += v;
        mean /= hidden_f;

        var var_sum: f32 = 0.0;
        for (x) |v| {
            const centered = v - mean;
            var_sum += centered * centered;
        }
        const variance = var_sum / hidden_f;
        const inv_std = 1.0 / @sqrt(variance + eps);

        var sum_dyg: f32 = 0.0;
        var sum_dyg_xhat: f32 = 0.0;
        for (0..hidden) |col| {
            const xhat = (x[col] - mean) * inv_std;
            const dyg = dy[col] * gamma[col];
            sum_dyg += dyg;
            sum_dyg_xhat += dyg * xhat;
        }

        for (0..hidden) |col| {
            const xhat = (x[col] - mean) * inv_std;
            const dyg = dy[col] * gamma[col];
            dx[col] = (inv_std / hidden_f) * (hidden_f * dyg - sum_dyg - xhat * sum_dyg_xhat);
        }
    }

    return grad_in;
}

// ----------------------------------------------------------------------------
// Tests
// ----------------------------------------------------------------------------

test "EncoderActivations init and deinit" {
    const allocator = std.testing.allocator;

    var acts = try EncoderActivations.init(allocator, 4, 8, 16);
    defer acts.deinit();

    try std.testing.expectEqual(@as(usize, 8), acts.seq_len);
    try std.testing.expectEqual(@as(usize, 16), acts.hidden_size);
    try std.testing.expectEqual(@as(usize, 0), acts.activations.items.len);
}

test "EncoderActivations addActivation copies data" {
    const allocator = std.testing.allocator;

    const seq_len: usize = 4;
    const in_features: usize = 8;

    var acts = try EncoderActivations.init(allocator, 2, seq_len, in_features);
    defer acts.deinit();

    // Build an input buffer with a known pattern.
    var input_buf: [4 * 8]f32 = undefined;
    for (&input_buf, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i)) * 0.1;

    try acts.addActivation(0, "query_proj", &input_buf, in_features, in_features);

    try std.testing.expectEqual(@as(usize, 1), acts.activations.items.len);
    const act = &acts.activations.items[0];
    try std.testing.expectEqual(@as(u32, 0), act.layer_idx);
    try std.testing.expectEqualStrings("query_proj", act.module_name);
    try std.testing.expectEqual(@as(usize, seq_len * in_features), act.input.len);

    // Mutate the original buffer — the stored copy must be unaffected.
    input_buf[0] = 9999.0;
    try std.testing.expect(act.input[0] != 9999.0);
}

test "moduleFromModernBertBaseName maps checkpoint names to adapter modules" {
    const q = try moduleFromModernBertBaseName("model.layers.21.attn.query_proj.weight");
    try std.testing.expectEqual(@as(u32, 21), q.layer_idx);
    try std.testing.expectEqualStrings("query_proj", q.module_name);

    const out = try moduleFromModernBertBaseName("model.layers.21.attn.Wo.weight");
    try std.testing.expectEqual(@as(u32, 21), out.layer_idx);
    try std.testing.expectEqualStrings("out_proj", out.module_name);

    const wo = try moduleFromModernBertBaseName("model.layers.21.mlp.Wo.weight");
    try std.testing.expectEqual(@as(u32, 21), wo.layer_idx);
    try std.testing.expectEqualStrings("wo", wo.module_name);

    try std.testing.expectError(
        error.UnsupportedLoRABaseName,
        moduleFromModernBertBaseName("model.layers.21.mlp.Wi.weight"),
    );
}

test "ModernBertSegmentVJPSession init compiles tiny global segment" {
    const allocator = std.testing.allocator;
    const config = modern_bert_graph.Config{
        .vocab_size = 32,
        .hidden_size = 8,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .head_dim = 4,
        .intermediate_size = 16,
        .max_position_embeddings = 8,
        .global_attn_every_n_layers = 1,
    };

    var session = try ModernBertSegmentVJPSession.init(
        allocator,
        config,
        1,
        4,
        0,
        1,
        2,
        4.0,
    );
    defer session.deinit();

    try std.testing.expectEqual(@as(usize, 14), session.base_params.len);
    try std.testing.expectEqual(@as(usize, 5), session.adapters.len);
    try std.testing.expectEqual(@as(usize, 11), session.trainable_param_names.len);
    try std.testing.expectEqualStrings("__segment_hidden_in", session.trainable_param_names[0]);
}

test "ModernBertSegmentVJPSession can omit unused hidden gradient" {
    const allocator = std.testing.allocator;
    const config = modern_bert_graph.Config{
        .vocab_size = 32,
        .hidden_size = 8,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .head_dim = 4,
        .intermediate_size = 16,
        .max_position_embeddings = 8,
        .global_attn_every_n_layers = 1,
    };

    var session = try ModernBertSegmentVJPSession.initWithOptions(
        allocator,
        config,
        1,
        4,
        0,
        1,
        2,
        4.0,
        false,
    );
    defer session.deinit();

    try std.testing.expect(!session.include_hidden_grad);
    try std.testing.expectEqual(@as(usize, 10), session.trainable_param_names.len);
    for (session.trainable_param_names) |name| {
        try std.testing.expect(!std.mem.eql(u8, name, "__segment_hidden_in"));
    }
}

test "ModernBertSegmentVJPSession can request hidden gradient without adapter gradients" {
    const allocator = std.testing.allocator;
    const config = modern_bert_graph.Config{
        .vocab_size = 32,
        .hidden_size = 8,
        .num_hidden_layers = 1,
        .num_attention_heads = 2,
        .head_dim = 4,
        .intermediate_size = 16,
        .max_position_embeddings = 8,
        .global_attn_every_n_layers = 1,
    };

    var session = try ModernBertSegmentVJPSession.initWithGradientOptions(
        allocator,
        config,
        1,
        4,
        0,
        1,
        2,
        4.0,
        true,
        false,
    );
    defer session.deinit();

    try std.testing.expect(session.include_hidden_grad);
    try std.testing.expect(!session.include_adapter_grads);
    try std.testing.expectEqual(@as(usize, 1), session.trainable_param_names.len);
    try std.testing.expectEqualStrings("__segment_hidden_in", session.trainable_param_names[0]);
}

test "ModernBertSegmentVJPSession accepts one-layer local-attention segments" {
    const allocator = std.testing.allocator;
    const config = modern_bert_graph.Config{
        .vocab_size = 32,
        .hidden_size = 8,
        .num_hidden_layers = 2,
        .num_attention_heads = 2,
        .head_dim = 4,
        .intermediate_size = 16,
        .max_position_embeddings = 8,
        .global_attn_every_n_layers = 3,
    };

    var session = try ModernBertSegmentVJPSession.init(
        allocator,
        config,
        1,
        4,
        1,
        2,
        2,
        4.0,
    );
    defer session.deinit();

    try std.testing.expect(session.is_local_attention);
    try std.testing.expectEqual(config.local_rope_theta, session.layer_inputs[0].rope_theta);
}

test "ModernBertSegmentVJPSession accepts mixed local and global multi-layer segments" {
    const allocator = std.testing.allocator;
    const config = modern_bert_graph.Config{
        .vocab_size = 32,
        .hidden_size = 8,
        .num_hidden_layers = 4,
        .num_attention_heads = 2,
        .head_dim = 4,
        .intermediate_size = 16,
        .max_position_embeddings = 8,
        .global_attn_every_n_layers = 3,
    };

    var session = try ModernBertSegmentVJPSession.init(
        allocator,
        config,
        1,
        4,
        2,
        4,
        2,
        4.0,
    );
    defer session.deinit();

    try std.testing.expectEqual(@as(usize, 2), session.layer_inputs.len);
    try std.testing.expect(session.layer_inputs[0].is_local_attention);
    try std.testing.expect(!session.layer_inputs[1].is_local_attention);
    try std.testing.expectEqual(config.local_rope_theta, session.layer_inputs[0].rope_theta);
    try std.testing.expectEqual(config.rope_theta, session.layer_inputs[1].rope_theta);
}

test "SharedSegmentAttnBias matches per-segment bias construction" {
    const allocator = std.testing.allocator;
    const mask = [_]f32{ 1.0, 1.0, 0.0, 1.0 };

    var shared = SharedSegmentAttnBias.init(allocator, 1, 4, 2, 2, &mask);
    defer shared.deinit();

    const shared_local = try shared.get(true);
    const shared_global = try shared.get(false);
    // Cached: repeated lookups return the same buffer.
    try std.testing.expectEqual(shared_local.ptr, (try shared.get(true)).ptr);

    const direct_local = try buildSegmentAttnBias(allocator, &mask, 1, 4, 2, true, 2);
    defer allocator.free(direct_local);
    const direct_global = try buildSegmentAttnBias(allocator, &mask, 1, 4, 2, false, 2);
    defer allocator.free(direct_global);

    try std.testing.expectEqualSlices(f32, direct_local, shared_local);
    try std.testing.expectEqualSlices(f32, direct_global, shared_global);
}

test "buildSegmentAttnBias combines padding and local window masks" {
    const allocator = std.testing.allocator;
    const mask = [_]f32{ 1.0, 1.0, 0.0, 1.0 };
    const bias = try buildSegmentAttnBias(allocator, &mask, 1, 4, 1, true, 2);
    defer allocator.free(bias);

    try std.testing.expectEqual(@as(usize, 16), bias.len);
    try std.testing.expectEqual(@as(f32, 0.0), bias[0 * 4 + 0]);
    try std.testing.expectEqual(@as(f32, 0.0), bias[0 * 4 + 1]);
    try std.testing.expect(bias[0 * 4 + 2] < -1.0e8);
    try std.testing.expect(bias[3 * 4 + 0] < -1.0e8);
    try std.testing.expect(bias[3 * 4 + 2] < -1.0e8);
    try std.testing.expectEqual(@as(f32, 0.0), bias[3 * 4 + 3]);
}

test "backwardLoRA smoke — grad_A and grad_B become non-zero" {
    const allocator = std.testing.allocator;

    const seq_len: usize = 3;
    const hidden_size: usize = 8;
    const rank: u32 = 2;
    const alpha: f32 = 4.0;

    // Create one activation with a constant input pattern.
    var acts = try EncoderActivations.init(allocator, 1, seq_len, hidden_size);
    defer acts.deinit();

    var input_data: [3 * 8]f32 = undefined;
    for (&input_data, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1)) * 0.1;

    try acts.addActivation(0, "query_proj", &input_data, hidden_size, hidden_size);

    // Create one LoRALayer matching (layer_idx=0, module_name="query_proj").
    // Use a fixed-length stack array so backwardLoRA can take a mutable slice of it.
    var lora_layers: [1]fused_chunker_lora.LoRALayer = undefined;
    lora_layers[0] = try fused_chunker_lora.LoRALayer.init(
        allocator,
        0,
        "query_proj",
        hidden_size,
        hidden_size,
        rank,
    );
    defer lora_layers[0].deinit();
    for (lora_layers[0].B, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i + 1)) * 0.001;
    }

    // Ensure grad accumulators start at zero.
    lora_layers[0].zeroGrads();

    // Build a non-zero output gradient [seq_len * hidden_size].
    var d_output: [3 * 8]f32 = undefined;
    for (&d_output, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1)) * 0.05;

    try backwardLoRA(allocator, &acts, &d_output, &lora_layers, alpha);

    // After backward, at least one element in grad_A and grad_B must be non-zero.
    var any_nonzero_a = false;
    for (lora_layers[0].grad_A) |v| {
        if (v != 0.0) {
            any_nonzero_a = true;
            break;
        }
    }
    var any_nonzero_b = false;
    for (lora_layers[0].grad_B) |v| {
        if (v != 0.0) {
            any_nonzero_b = true;
            break;
        }
    }

    try std.testing.expect(any_nonzero_a);
    try std.testing.expect(any_nonzero_b);
}

test "backwardLoRA skips unmatched activations" {
    const allocator = std.testing.allocator;

    const seq_len: usize = 2;
    const hidden_size: usize = 4;

    // Activation targets layer 0, "query_proj".
    var acts = try EncoderActivations.init(allocator, 1, seq_len, hidden_size);
    defer acts.deinit();

    var input_data: [2 * 4]f32 = undefined;
    @memset(&input_data, 1.0);
    try acts.addActivation(0, "query_proj", &input_data, hidden_size, hidden_size);

    // LoRALayer targets layer 1, "value_proj" — no match with the activation above.
    var lora_layers: [1]fused_chunker_lora.LoRALayer = undefined;
    lora_layers[0] = try fused_chunker_lora.LoRALayer.init(
        allocator,
        1, // different layer_idx
        "value_proj", // different module_name
        hidden_size,
        hidden_size,
        2,
    );
    defer lora_layers[0].deinit();
    lora_layers[0].zeroGrads();

    var d_output: [2 * 4]f32 = undefined;
    @memset(&d_output, 0.5);

    try backwardLoRA(allocator, &acts, &d_output, &lora_layers, 4.0);

    // Grads must remain zero because there was no matching layer.
    for (lora_layers[0].grad_A) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
    for (lora_layers[0].grad_B) |v| try std.testing.expectEqual(@as(f32, 0.0), v);
}

test "backpropLayerNormInput matches finite differences" {
    const allocator = std.testing.allocator;

    const rows: usize = 1;
    const hidden: usize = 4;
    const eps: f32 = 1e-5;
    var input = [_]f32{ 0.3, -0.7, 1.2, 0.5 };
    const gamma = [_]f32{ 0.9, 1.1, -0.8, 0.7 };
    const grad_out = [_]f32{ 0.2, -0.4, 0.6, -0.1 };

    const grad = try backpropLayerNormInput(allocator, &grad_out, &input, &gamma, rows, hidden, eps);
    defer allocator.free(grad);

    const delta: f32 = 1e-3;
    for (0..hidden) |i| {
        input[i] += delta;
        const plus = layerNormSyntheticLoss(&input, &gamma, &grad_out, hidden, eps);
        input[i] -= 2.0 * delta;
        const minus = layerNormSyntheticLoss(&input, &gamma, &grad_out, hidden, eps);
        input[i] += delta;

        const numeric = (plus - minus) / (2.0 * delta);
        try std.testing.expectApproxEqAbs(numeric, grad[i], 2e-3);
    }
}

fn layerNormSyntheticLoss(input: []const f32, gamma: []const f32, grad_out: []const f32, hidden: usize, eps: f32) f32 {
    const hidden_f: f32 = @floatFromInt(hidden);
    var mean: f32 = 0.0;
    for (input) |v| mean += v;
    mean /= hidden_f;

    var var_sum: f32 = 0.0;
    for (input) |v| {
        const centered = v - mean;
        var_sum += centered * centered;
    }
    const inv_std = 1.0 / @sqrt(var_sum / hidden_f + eps);

    var loss: f32 = 0.0;
    for (0..hidden) |i| {
        const y = (input[i] - mean) * inv_std * gamma[i];
        loss += grad_out[i] * y;
    }
    return loss;
}
