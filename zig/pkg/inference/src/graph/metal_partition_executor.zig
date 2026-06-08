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
const build_options = @import("build_options");
const platform = @import("antfly_platform");

const contracts = @import("backend_contracts.zig");
const ops_mod = @import("../ops/ops.zig");
const interpreter = @import("interpreter.zig");
const partition_mod = @import("partition.zig");
const metal_capabilities = @import("metal_capabilities.zig");
const buffer_plan_mod = @import("buffer_plan.zig");
const operator_plan_mod = @import("operator_plan.zig");
const device_mesh_mod = @import("device_mesh.zig");
const gpu_hosted_store_mod = @import("../ops/gpu_hosted_store.zig");
const metal_compute_mod = @import("../ops/metal_compute.zig");
const weight_source_mod = @import("../models/weight_source.zig");
const quant_codec = @import("../gguf/quant_codec.zig");
const transpose_utils = @import("transpose_utils.zig");
const metal_runtime_mod = if (build_options.enable_metal) @import("../backends/metal_runtime.zig") else struct {
    pub fn metalDeviceAvailable() bool {
        return false;
    }
};

const Graph = ml.graph.Graph;
const NodeId = ml.graph.NodeId;
const null_node = ml.graph.null_node;
const Shape = ml.graph.Shape;

const CT = contracts.CT;
const ComputeBackend = ops_mod.ComputeBackend;
const PartitionExecutor = partition_mod.PartitionExecutor;
const DeviceId = device_mesh_mod.DeviceId;
const GraphPlanSlot = ops_mod.GraphPlanSlot;
const QuantizedStorage = weight_source_mod.QuantizedStorage;
const OperatorPlan = operator_plan_mod.OperatorPlan;

const max_graph_plan_slots = 26;

const MetalExecutionKind = enum {
    command,
    metadata_alias,
    descriptor_materialization,
    constant_materialization,
};

const OpExecutionCount = struct {
    name: []const u8 = "",
    count: usize = 0,
    total_ns: u64 = 0,
};

const OpExecutionStats = struct {
    command_counts: [96]OpExecutionCount = [_]OpExecutionCount{.{}} ** 96,
    command_used: usize = 0,
    command_class_counts: [160]CommandExecutionSummary = [_]CommandExecutionSummary{.{}} ** 160,
    command_class_used: usize = 0,
    fallback_counts: [96]OpExecutionCount = [_]OpExecutionCount{.{}} ** 96,
    fallback_used: usize = 0,
    host_output_counts: [96]OpExecutionCount = [_]OpExecutionCount{.{}} ** 96,
    host_output_used: usize = 0,
    host_output_reason_counts: [32]OpExecutionCount = [_]OpExecutionCount{.{}} ** 32,
    host_output_reason_used: usize = 0,
    runtime_region_counts: [32]OpExecutionCount = [_]OpExecutionCount{.{}} ** 32,
    runtime_region_used: usize = 0,
    dot_command_shapes: [128]DotShapeExecutionSummary = [_]DotShapeExecutionSummary{.{}} ** 128,
    dot_command_shape_used: usize = 0,

    fn recordCommand(self: *OpExecutionStats, name: []const u8, elapsed_ns: u64) void {
        recordOpCount(&self.command_counts, &self.command_used, name, elapsed_ns);
    }

    fn recordCommandClass(self: *OpExecutionStats, graph: *const Graph, node_id: NodeId, node_pos: usize, elapsed_ns: u64) void {
        recordCommandExecutionSummary(graph, node_id, node_pos, &self.command_class_counts, &self.command_class_used, elapsed_ns);
    }

    fn recordFallback(self: *OpExecutionStats, name: []const u8, elapsed_ns: u64) void {
        recordOpCount(&self.fallback_counts, &self.fallback_used, name, elapsed_ns);
    }

    fn recordHostOutput(self: *OpExecutionStats, name: []const u8, elapsed_ns: u64) void {
        recordOpCount(&self.host_output_counts, &self.host_output_used, name, elapsed_ns);
    }

    fn recordHostOutputReason(self: *OpExecutionStats, name: []const u8, elapsed_ns: u64) void {
        recordOpCount(&self.host_output_reason_counts, &self.host_output_reason_used, name, elapsed_ns);
    }

    fn recordRuntimeRegion(self: *OpExecutionStats, name: []const u8, elapsed_ns: u64) void {
        recordOpCount(&self.runtime_region_counts, &self.runtime_region_used, name, elapsed_ns);
    }

    fn recordDotCommand(self: *OpExecutionStats, graph: *const Graph, node_id: NodeId, node_pos: usize, elapsed_ns: u64) void {
        recordDotShapeExecutionSummary(graph, node_id, node_pos, &self.dot_command_shapes, &self.dot_command_shape_used, elapsed_ns);
    }
};

const ExecutorLoopProfile = struct {
    nodes: usize = 0,
    executed_nodes: usize = 0,
    partition_view_ns: u64 = 0,
    graph_plan_ns: u64 = 0,
    materialize_runtime_inputs_ns: u64 = 0,
    materialize_parameters_ns: u64 = 0,
    materialize_constants_ns: u64 = 0,
    begin_frame_ns: u64 = 0,
    runtime_region_plan_ns: u64 = 0,
    execution_ns: u64 = 0,
    stats_ns: u64 = 0,
    alias_clone_ns: u64 = 0,
    free_expired_ns: u64 = 0,
    submit_frame_ns: u64 = 0,
    boundary_outputs_ns: u64 = 0,

    fn print(self: ExecutorLoopProfile, label: []const u8) void {
        std.debug.print(
            "{s}: nodes={d}:executed={d}:partition_view_ms={d:.3}:graph_plan_ms={d:.3}:runtime_inputs_ms={d:.3}:parameters_ms={d:.3}:constants_ms={d:.3}:begin_frame_ms={d:.3}:runtime_plan_ms={d:.3}:execution_ms={d:.3}:stats_ms={d:.3}:alias_clone_ms={d:.3}:free_expired_ms={d:.3}:submit_frame_ms={d:.3}:boundary_outputs_ms={d:.3}:accounted_ms={d:.3}\n",
            .{
                label,
                self.nodes,
                self.executed_nodes,
                nsToMs(self.partition_view_ns),
                nsToMs(self.graph_plan_ns),
                nsToMs(self.materialize_runtime_inputs_ns),
                nsToMs(self.materialize_parameters_ns),
                nsToMs(self.materialize_constants_ns),
                nsToMs(self.begin_frame_ns),
                nsToMs(self.runtime_region_plan_ns),
                nsToMs(self.execution_ns),
                nsToMs(self.stats_ns),
                nsToMs(self.alias_clone_ns),
                nsToMs(self.free_expired_ns),
                nsToMs(self.submit_frame_ns),
                nsToMs(self.boundary_outputs_ns),
                nsToMs(self.partition_view_ns +
                    self.graph_plan_ns +
                    self.materialize_runtime_inputs_ns +
                    self.materialize_parameters_ns +
                    self.materialize_constants_ns +
                    self.begin_frame_ns +
                    self.runtime_region_plan_ns +
                    self.execution_ns +
                    self.stats_ns +
                    self.alias_clone_ns +
                    self.free_expired_ns +
                    self.submit_frame_ns +
                    self.boundary_outputs_ns),
            },
        );
    }
};

const RuntimeRegionKind = enum(u8) {
    none = 0,
    raw_linear_dot,
    raw_linear_pair,
    raw_linear_bias,
    raw_linear_bias_pair,
    lora_linear,
    lora_linear_qkv,
    deberta_attention,
    lora_backward,
    ffn_gelu_backward,
    q_linear,
    linear_qkv,
    grouped_linear_qkv_slice,
    rms_norm_grouped_linear_qkv_slice,
    attention_output_residual,
    rms_norm_gated_ffn_residual,
    gated_ffn_residual,
    ple_residual,
};

const RuntimeRegion = union(RuntimeRegionKind) {
    none: void,
    raw_linear_dot: RawLinearDotPattern,
    raw_linear_pair: RawLinearPairPattern,
    raw_linear_bias: RawLinearBiasPattern,
    raw_linear_bias_pair: RawLinearBiasPairPattern,
    lora_linear: LoraLinearPattern,
    lora_linear_qkv: LoraLinearQkvPattern,
    deberta_attention: DebertaAttentionPattern,
    lora_backward: LoraBackwardPattern,
    ffn_gelu_backward: FfnGeluBackwardPattern,
    q_linear: QLinearPattern,
    linear_qkv: LinearNoBiasQkvPattern,
    grouped_linear_qkv_slice: GroupedLinearQkvSlicePattern,
    rms_norm_grouped_linear_qkv_slice: RmsNormGroupedLinearQkvSlicePattern,
    attention_output_residual: AttentionOutputResidualPattern,
    rms_norm_gated_ffn_residual: RmsNormGatedFfnResidualPattern,
    gated_ffn_residual: GatedFfnResidualPattern,
    ple_residual: PleResidualPattern,
};

const PreparedQkvRegion = struct {
    q_slot: usize,
    k_slot: usize,
    v_slot: usize,
};

const PreparedLinearRegion = struct {
    linear_slot: usize,
};

const PreparedLinearPairRegion = struct {
    first_slot: usize,
    second_slot: usize,
};

const PreparedRmsNormGroupedQkvRegion = struct {
    norm_slot: usize,
    qkv: PreparedQkvRegion,
};

const PreparedAttentionOutputResidualRegion = struct {
    linear_slot: usize,
    pre_linear_rms_norm_slot: ?usize = null,
    post_linear_rms_norm_slot: ?usize = null,
};

const PreparedRmsNormGatedFfnResidualRegion = struct {
    norm_slot: usize,
    ffn: PreparedGatedFfnResidualRegion,
};

const PreparedGatedFfnResidualRegion = struct {
    gate_slot: usize,
    up_slot: usize,
    down_slot: usize,
    post_down_rms_norm_slot: ?usize = null,
};

const PreparedPleResidualRegion = struct {
    gate_slot: usize,
    projection_slot: usize,
    post_norm_slot: usize,
};

const PreparedRuntimeRegion = union(RuntimeRegionKind) {
    none: void,
    raw_linear_dot: PreparedLinearRegion,
    raw_linear_pair: PreparedLinearPairRegion,
    raw_linear_bias: PreparedLinearRegion,
    raw_linear_bias_pair: PreparedLinearPairRegion,
    lora_linear: void,
    lora_linear_qkv: void,
    deberta_attention: void,
    lora_backward: void,
    ffn_gelu_backward: void,
    q_linear: PreparedLinearRegion,
    linear_qkv: PreparedQkvRegion,
    grouped_linear_qkv_slice: PreparedQkvRegion,
    rms_norm_grouped_linear_qkv_slice: PreparedRmsNormGroupedQkvRegion,
    attention_output_residual: PreparedAttentionOutputResidualRegion,
    rms_norm_gated_ffn_residual: PreparedRmsNormGatedFfnResidualRegion,
    gated_ffn_residual: PreparedGatedFfnResidualRegion,
    ple_residual: PreparedPleResidualRegion,
};

const RuntimeRegionPlan = struct {
    node_count: usize = 0,
    value_count: usize = 0,
    first_node: NodeId = null_node,
    last_node: NodeId = null_node,
    regions_by_pos: []RuntimeRegion = &.{},
    prepared_by_pos: []PreparedRuntimeRegion = &.{},
    attention_input_max_first_node: []NodeId = &.{},
    region_count: usize = 0,
    covered_node_count: usize = 0,
    elided_node_count: usize = 0,

    fn deinit(self: *RuntimeRegionPlan, allocator: std.mem.Allocator) void {
        allocator.free(self.regions_by_pos);
        allocator.free(self.prepared_by_pos);
        allocator.free(self.attention_input_max_first_node);
        self.* = .{};
    }

    fn matches(self: RuntimeRegionPlan, node_ids: []const NodeId, value_count: usize) bool {
        if (self.regions_by_pos.len != node_ids.len) return false;
        if (self.prepared_by_pos.len != node_ids.len) return false;
        if (self.attention_input_max_first_node.len != value_count) return false;
        if (self.node_count != node_ids.len or self.value_count != value_count) return false;
        if (node_ids.len == 0) return self.first_node == null_node and self.last_node == null_node;
        return self.first_node == node_ids[0] and self.last_node == node_ids[node_ids.len - 1];
    }

    fn regionAt(self: RuntimeRegionPlan, node_pos: usize, node_id: NodeId, node_ids: []const NodeId) RuntimeRegion {
        if (node_pos >= self.regions_by_pos.len or node_pos >= node_ids.len) return .{ .none = {} };
        if (node_ids[node_pos] != node_id) return .{ .none = {} };
        return self.regions_by_pos[node_pos];
    }

    fn preparedPtrAt(self: RuntimeRegionPlan, node_pos: usize, node_id: NodeId, node_ids: []const NodeId) ?*PreparedRuntimeRegion {
        if (node_pos >= self.prepared_by_pos.len or node_pos >= node_ids.len) return null;
        if (node_ids[node_pos] != node_id) return null;
        return &self.prepared_by_pos[node_pos];
    }

    fn needsAttentionInputAfterNode(self: RuntimeRegionPlan, input_id: NodeId, node_id: NodeId) bool {
        if (input_id == null_node) return false;
        const input_index: usize = @intCast(input_id);
        if (input_index >= self.attention_input_max_first_node.len) return false;
        const max_first_node = self.attention_input_max_first_node[input_index];
        return max_first_node != null_node and max_first_node > node_id;
    }
};

const CachedPartitionBufferView = struct {
    partition_index: u32 = 0,
    partition_count: usize = 0,
    slot_count: usize = 0,
    transfer_count: usize = 0,
    first_slot_node: NodeId = null_node,
    last_slot_node: NodeId = null_node,
    backend: contracts.BackendKind = .native,
    view: buffer_plan_mod.PartitionBufferView,

    fn init(
        buffer_plan: *const buffer_plan_mod.BufferPlan,
        partition_plan: *const partition_mod.PartitionPlan,
        partition_index: u32,
        view: buffer_plan_mod.PartitionBufferView,
    ) CachedPartitionBufferView {
        return .{
            .partition_index = partition_index,
            .partition_count = partition_plan.partitions.len,
            .slot_count = buffer_plan.slots.len,
            .transfer_count = buffer_plan.transfers.len,
            .first_slot_node = if (buffer_plan.slots.len == 0) null_node else buffer_plan.slots[0].node_id,
            .last_slot_node = if (buffer_plan.slots.len == 0) null_node else buffer_plan.slots[buffer_plan.slots.len - 1].node_id,
            .backend = view.backend,
            .view = view,
        };
    }

    fn deinit(self: *CachedPartitionBufferView, allocator: std.mem.Allocator) void {
        self.view.deinit(allocator);
        self.* = undefined;
    }

    fn matches(
        self: CachedPartitionBufferView,
        buffer_plan: *const buffer_plan_mod.BufferPlan,
        partition_plan: *const partition_mod.PartitionPlan,
        partition_index: u32,
    ) bool {
        if (partition_index >= partition_plan.partitions.len) return false;
        return self.partition_index == partition_index and
            self.partition_count == partition_plan.partitions.len and
            self.slot_count == buffer_plan.slots.len and
            self.transfer_count == buffer_plan.transfers.len and
            self.first_slot_node == (if (buffer_plan.slots.len == 0) null_node else buffer_plan.slots[0].node_id) and
            self.last_slot_node == (if (buffer_plan.slots.len == 0) null_node else buffer_plan.slots[buffer_plan.slots.len - 1].node_id) and
            self.backend == partition_plan.partitions[@intCast(partition_index)].backend;
    }

    fn traceMismatch(
        self: CachedPartitionBufferView,
        buffer_plan: *const buffer_plan_mod.BufferPlan,
        partition_plan: *const partition_mod.PartitionPlan,
        partition_index: u32,
    ) void {
        if (!tracePartitionViewCacheEnabled()) return;
        const current_backend = if (partition_index < partition_plan.partitions.len)
            partition_plan.partitions[@intCast(partition_index)].backend
        else
            .native;
        std.debug.print(
            "partition_view_cache_miss: cached_partition={d} current_partition={d} cached_partitions={d} current_partitions={d} cached_slots={d} current_slots={d} cached_transfers={d} current_transfers={d} cached_first={} current_first={} cached_last={} current_last={} cached_backend={s} current_backend={s}\n",
            .{
                self.partition_index,
                partition_index,
                self.partition_count,
                partition_plan.partitions.len,
                self.slot_count,
                buffer_plan.slots.len,
                self.transfer_count,
                buffer_plan.transfers.len,
                self.first_slot_node,
                if (buffer_plan.slots.len == 0) null_node else buffer_plan.slots[0].node_id,
                self.last_slot_node,
                if (buffer_plan.slots.len == 0) null_node else buffer_plan.slots[buffer_plan.slots.len - 1].node_id,
                @tagName(self.backend),
                @tagName(current_backend),
            },
        );
    }
};

const PartitionBufferViewResult = struct {
    view: buffer_plan_mod.PartitionBufferView,
    cache_hit: bool = false,
};

const RuntimeFrameIneligibleReason = enum {
    none,
    no_regions,
    missing_qkv,
    missing_attention,
    missing_ffn,
    missing_ple,
    single_row,
    non_layer_order,
    shape_mismatch,
    missing_model_metadata,
};

const RuntimeFrameEligibility = struct {
    layers: usize = 0,
    reason: RuntimeFrameIneligibleReason = .none,

    fn eligible(self: RuntimeFrameEligibility) bool {
        return self.reason == .none and self.layers > 0;
    }
};

const RuntimeFrameLayerShape = struct {
    rows: usize,
    hidden_size: usize,
    attention_input_size: usize = 0,
};

const RuntimeFrameQkvMetadata = struct {
    layer_index: usize,
    rows: usize,
    hidden_size: usize,
    q_dim: usize,
    kv_dim: usize,
    q_weight_id: NodeId,
    k_weight_id: ?NodeId = null,
    v_weight_id: ?NodeId = null,
};

const RuntimeFrameLayerMetadata = struct {
    layer_index: usize,
    shares_kv: bool,
    kv_layer_index: usize,
    kv_heads: usize,
    head_dim: usize,
    intermediate_size: usize,
    hidden_size: usize,
    attention_input_size: usize,
    ple_hidden_size: usize,
    activation: ops_mod.DecoderRuntimeActivationKind,
};

const RuntimeFrameMetadata = struct {
    rows: usize,
    layer_count: usize,
    hidden_size: usize,
    num_attention_heads: usize,
    global_head_dim: usize,
    ple_hidden_size: usize,
    activation: ops_mod.DecoderRuntimeActivationKind,
};

pub fn isMetalDeviceResident(cb: *const ComputeBackend, tensor: CT) bool {
    if (cb.kind() != .metal) return false;
    if (comptime !build_options.enable_metal) return false;
    return metal_compute_mod.MetalCompute.debugHasDeviceTensor(cb, tensor);
}

fn isMetalResidentOrQuantizedDescriptor(cb: *const ComputeBackend, tensor: CT) bool {
    if (isMetalDeviceResident(cb, tensor)) return true;
    if (cb.kind() != .metal) return false;
    if (comptime !build_options.enable_metal) return false;
    return metal_compute_mod.MetalCompute.getQuantizedStorage(cb, tensor) != null;
}

fn isMetalStorageAlias(cb: *const ComputeBackend, lhs: CT, rhs: CT) bool {
    if (cb.kind() != .metal) return false;
    if (comptime !build_options.enable_metal) return false;
    return metal_compute_mod.MetalCompute.debugSharesStorage(cb, lhs, rhs);
}

fn classifyMetalExecutionKind(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    node_id: NodeId,
) MetalExecutionKind {
    const output = valueFor(values, node_id) orelse return .command;
    const node = graph.node(node_id);
    const inputs = node.getInputs();
    switch (node.op) {
        .constant, .fused_zero_tensor => return .constant_materialization,
        .reshape, .slice => {
            if (inputs.len == 0) return .command;
            const input = valueFor(values, inputs[0]) orelse return .command;
            if (isMetalStorageAlias(cb, input, output)) return .metadata_alias;
        },
        .concat_prim => {
            if (comptime build_options.enable_metal) {
                if (cb.kind() == .metal) {
                    if (metal_compute_mod.MetalCompute.getQuantizedStorage(cb, output) != null) {
                        return .descriptor_materialization;
                    }
                }
            }
        },
        else => {},
    }
    return .command;
}

pub fn makeMetalDeviceResident(cb: *const ComputeBackend, tensor: CT) !?CT {
    if (cb.kind() != .metal) return null;
    if (comptime !build_options.enable_metal) return null;
    return metal_compute_mod.MetalCompute.makeDeviceResident(cb, tensor);
}

fn promoteMetalOutputIfNeeded(cb: *const ComputeBackend, output: ?CT) !?CT {
    const ct = output orelse return null;
    if (isMetalDeviceResident(cb, ct)) return ct;
    if (try makeMetalDeviceResident(cb, ct)) |device_ct| {
        if (device_ct != ct) cb.free(ct);
        return device_ct;
    }
    return ct;
}

const TemporaryMetalResidentValue = struct {
    value: CT,
    owned: bool = false,

    fn deinit(self: TemporaryMetalResidentValue, cb: *const ComputeBackend) void {
        if (self.owned) cb.free(self.value);
    }
};

fn temporaryMetalResidentValue(cb: *const ComputeBackend, tensor: CT) !TemporaryMetalResidentValue {
    if (isMetalDeviceResident(cb, tensor)) return .{ .value = tensor };
    if (try makeMetalDeviceResident(cb, tensor)) |device_ct| {
        return .{ .value = device_ct, .owned = device_ct != tensor };
    }
    return .{ .value = tensor };
}

pub const MetalGraphPlanAllocation = struct {
    allocation: buffer_plan_mod.AllocationId,
    graph_slot: usize,
    bytes: usize,
};

pub const MetalPartitionGraphPlan = struct {
    slots: []const GraphPlanSlot,
    allocations: []const MetalGraphPlanAllocation,

    pub fn deinit(self: *MetalPartitionGraphPlan, allocator: std.mem.Allocator) void {
        allocator.free(self.slots);
        allocator.free(self.allocations);
    }
};

pub const MetalPartitionExecutor = struct {
    allocator: std.mem.Allocator,
    graph: *const Graph,
    backend: *const ComputeBackend,
    pe: PartitionExecutor = undefined,
    owned: bool = false,
    partition_view: ?CachedPartitionBufferView = null,
    runtime_region_plan: ?RuntimeRegionPlan = null,

    const vtable = PartitionExecutor.VTable{
        .execute = &executeFn,
        .deinit = &deinitFn,
    };

    pub fn initBorrowed(
        allocator: std.mem.Allocator,
        graph: *const Graph,
        backend: *const ComputeBackend,
    ) MetalPartitionExecutor {
        return .{
            .allocator = allocator,
            .graph = graph,
            .backend = backend,
        };
    }

    pub fn create(
        allocator: std.mem.Allocator,
        graph: *const Graph,
        backend: *const ComputeBackend,
    ) !*MetalPartitionExecutor {
        const exec = try allocator.create(MetalPartitionExecutor);
        exec.* = .{
            .allocator = allocator,
            .graph = graph,
            .backend = backend,
            .owned = true,
        };
        exec.pe = .{ .ptr = exec, .vtable = &vtable };
        return exec;
    }

    pub fn partitionExecutor(self: *MetalPartitionExecutor) *const PartitionExecutor {
        self.pe = .{ .ptr = self, .vtable = &vtable };
        return &self.pe;
    }

    fn executeFn(
        ctx: *anyopaque,
        values: []?CT,
        value_device: []DeviceId,
        node_ids: []const NodeId,
        device_id: DeviceId,
        exec_ctx: PartitionExecutor.ExecutionContext,
    ) anyerror!void {
        const self: *MetalPartitionExecutor = @ptrCast(@alignCast(ctx));
        return self.execute(values, value_device, node_ids, device_id, exec_ctx);
    }

    fn deinitFn(ctx: *anyopaque) void {
        const self: *MetalPartitionExecutor = @ptrCast(@alignCast(ctx));
        if (self.runtime_region_plan) |*plan| {
            plan.deinit(self.allocator);
            self.runtime_region_plan = null;
        }
        if (self.partition_view) |*view| {
            view.deinit(self.allocator);
            self.partition_view = null;
        }
        if (self.owned) self.allocator.destroy(self);
    }

    fn partitionBufferView(
        self: *MetalPartitionExecutor,
        allocator: std.mem.Allocator,
        buffer_plan: *const buffer_plan_mod.BufferPlan,
        partition_plan: *const partition_mod.PartitionPlan,
        partition_index: u32,
        transient: *?buffer_plan_mod.PartitionBufferView,
    ) !PartitionBufferViewResult {
        if (self.owned and partitionViewCacheEnabled()) {
            if (self.partition_view) |view| {
                if (view.matches(buffer_plan, partition_plan, partition_index)) {
                    if (tracePartitionViewCacheEnabled()) std.debug.print("partition_view_cache_hit: partition={d} slots={d} transfers={d}\n", .{ partition_index, buffer_plan.slots.len, buffer_plan.transfers.len });
                    return .{ .view = view.view, .cache_hit = true };
                }
                view.traceMismatch(buffer_plan, partition_plan, partition_index);
                var old = self.partition_view.?;
                old.deinit(self.allocator);
                self.partition_view = null;
            }
            const view = try buffer_plan.partitionView(self.allocator, partition_plan, partition_index);
            self.partition_view = CachedPartitionBufferView.init(buffer_plan, partition_plan, partition_index, view);
            if (tracePartitionViewCacheEnabled()) std.debug.print("partition_view_cache_store: partition={d} slots={d} transfers={d}\n", .{ partition_index, buffer_plan.slots.len, buffer_plan.transfers.len });
            return .{ .view = self.partition_view.?.view };
        }

        if (tracePartitionViewCacheEnabled()) std.debug.print("partition_view_cache_bypass: owned={} enabled={} partition={d} slots={d} transfers={d}\n", .{ self.owned, partitionViewCacheEnabled(), partition_index, buffer_plan.slots.len, buffer_plan.transfers.len });
        transient.* = try buffer_plan.partitionView(allocator, partition_plan, partition_index);
        return .{ .view = transient.*.? };
    }

    fn runtimeRegionPlan(
        self: *MetalPartitionExecutor,
        allocator: std.mem.Allocator,
        graph: *const Graph,
        node_ids: []const NodeId,
        value_count: usize,
        reachable: []const bool,
        last_use: []const u32,
        stats: ?*PartitionExecutor.ExecutionStats,
        transient: *?RuntimeRegionPlan,
    ) !RuntimeRegionPlan {
        if (runtimeRegionPlanDisabled()) return .{};

        if (self.owned) {
            if (self.runtime_region_plan) |plan| {
                if (plan.matches(node_ids, value_count)) {
                    if (stats) |s| s.runtime_region_plan_reuses += 1;
                    return plan;
                }
                var old = self.runtime_region_plan.?;
                old.deinit(self.allocator);
                self.runtime_region_plan = null;
            }
            self.runtime_region_plan = try buildRuntimeRegionPlan(self.allocator, graph, node_ids, value_count, reachable, last_use);
            if (stats) |s| {
                s.runtime_region_plan_compiles += 1;
                s.runtime_region_plan_regions += self.runtime_region_plan.?.region_count;
            }
            return self.runtime_region_plan.?;
        }

        transient.* = try buildRuntimeRegionPlan(allocator, graph, node_ids, value_count, reachable, last_use);
        if (stats) |s| {
            s.runtime_region_plan_compiles += 1;
            s.runtime_region_plan_regions += transient.*.?.region_count;
        }
        return transient.*.?;
    }

    fn execute(
        self: *MetalPartitionExecutor,
        values: []?CT,
        value_device: []DeviceId,
        node_ids: []const NodeId,
        device_id: DeviceId,
        exec_ctx: PartitionExecutor.ExecutionContext,
    ) !void {
        const allocator = exec_ctx.allocator orelse self.allocator;
        const graph = exec_ctx.graph orelse self.graph;
        const cb = exec_ctx.backend orelse self.backend;
        const reachable = exec_ctx.reachable orelse return error.MissingPartitionExecutionContext;
        const last_use = exec_ctx.last_use orelse return error.MissingPartitionExecutionContext;
        const buffer_plan = exec_ctx.buffer_plan orelse return error.MissingPartitionExecutionContext;
        const partition_plan = exec_ctx.partition_plan orelse return error.MissingPartitionExecutionContext;
        const trace_nodes = traceMetalGraphNodesEnabled();
        const progress_interval = metalGraphProgressInterval();
        const progress_start = metalGraphProgressStart();
        const progress_end = metalGraphProgressEnd();
        const trace_progress = progress_interval != 0 or progress_start != std.math.maxInt(usize) or progress_end != std.math.maxInt(usize);
        const collect_op_stats = metalPartitionOpStatsEnabled();
        const collect_loop_profile = metalPartitionLoopProfileEnabled();
        const collect_residency_stats = metalPartitionResidencyStatsEnabled() or collect_op_stats or traceMetalHostOutputsEnabled();
        var op_execution_stats = OpExecutionStats{};
        var loop_profile = ExecutorLoopProfile{};
        const partition_index = try partitionIndexForNodes(buffer_plan, node_ids);
        if (trace_nodes) std.debug.print("graph_executor_node_trace: executor_begin partition={d} nodes={d}\n", .{ partition_index, node_ids.len });
        if (trace_progress) std.debug.print("metal_partition_progress: phase=executor_begin partition={d} nodes={d}\n", .{ partition_index, node_ids.len });

        var transient_partition_view: ?buffer_plan_mod.PartitionBufferView = null;
        defer if (transient_partition_view) |*view| view.deinit(allocator);
        const partition_view_start_ns = if (collect_loop_profile) metalPartitionNowNs() else 0;
        const partition_view_result = try self.partitionBufferView(
            allocator,
            buffer_plan,
            partition_plan,
            partition_index,
            &transient_partition_view,
        );
        const partition_view = partition_view_result.view;
        if (!partition_view_result.cache_hit) try validatePartitionView(partition_view, node_ids);
        if (collect_loop_profile) loop_profile.partition_view_ns += metalPartitionElapsedNs(partition_view_start_ns, metalPartitionNowNs());
        if (trace_progress) std.debug.print("metal_partition_progress: phase=partition_view_ready partition={d} slots={d} transfers_in={d} transfers_out={d}\n", .{ partition_index, partition_view.slots.len, partition_view.transfers_in.len, partition_view.transfers_out.len });
        if (trace_nodes) {
            std.debug.print(
                "graph_executor_node_trace: partition_view partition={d} slots={d} transfers_in={d} transfers_out={d}\n",
                .{ partition_index, partition_view.slots.len, partition_view.transfers_in.len, partition_view.transfers_out.len },
            );
        }

        const graph_plan_start_ns = if (collect_loop_profile) metalPartitionNowNs() else 0;
        var metal_graph_plan = try buildMetalGraphPlan(allocator, buffer_plan, partition_view);
        defer metal_graph_plan.deinit(allocator);
        if (trace_nodes) printMetalGraphPlanTrace(partition_index, metal_graph_plan);
        if (trace_progress) std.debug.print("metal_partition_progress: phase=reserve_graph_slots_begin partition={d} slots={d}\n", .{ partition_index, metal_graph_plan.slots.len });
        _ = try cb.reserveGraphPlanSlots(metal_graph_plan.slots);
        if (trace_progress) std.debug.print("metal_partition_progress: phase=reserve_graph_slots_end partition={d}\n", .{partition_index});
        if (trace_nodes) std.debug.print("graph_executor_node_trace: graph_plan_reserved partition={d}\n", .{partition_index});
        if (collect_loop_profile) loop_profile.graph_plan_ns += metalPartitionElapsedNs(graph_plan_start_ns, metalPartitionNowNs());
        if (exec_ctx.stats) |stats| {
            stats.graph_plan_slots_reserved += metal_graph_plan.slots.len;
            for (metal_graph_plan.slots) |slot| stats.graph_plan_bytes_reserved += slot.bytes;
        }

        const options = exec_ctx.options orelse interpreter.ExecuteOptions{
            .attention = if (exec_ctx.attention) |attention| attention.* else null,
            .embedding_ids = exec_ctx.embedding_ids,
        };

        var local_owned_runtime_transfers = std.AutoHashMapUnmanaged(NodeId, void).empty;
        defer local_owned_runtime_transfers.deinit(allocator);
        var effective_exec_ctx = exec_ctx;
        if (effective_exec_ctx.owned_runtime_transfers == null) {
            effective_exec_ctx.owned_runtime_transfers = &local_owned_runtime_transfers;
        }

        var rt_map = std.AutoHashMapUnmanaged(NodeId, CT).empty;
        defer rt_map.deinit(allocator);
        var donated = std.AutoHashMapUnmanaged(NodeId, void).empty;
        defer donated.deinit(allocator);
        if (options.runtime_inputs) |inputs| {
            for (inputs, 0..) |ri, idx| {
                try rt_map.put(allocator, ri.node_id, ri.value);
                if (options.donate) |donate| {
                    if (idx < donate.len and donate[idx]) try donated.put(allocator, ri.node_id, {});
                }
            }
        }

        if (trace_nodes) std.debug.print("graph_executor_node_trace: materialize_runtime_inputs_begin partition={d}\n", .{partition_index});
        if (trace_progress) std.debug.print("metal_partition_progress: phase=materialize_runtime_inputs_begin partition={d}\n", .{partition_index});
        const runtime_inputs_start_ns = if (collect_loop_profile) metalPartitionNowNs() else 0;
        try materializePartitionRuntimeInputs(
            allocator,
            values,
            value_device,
            node_ids,
            device_id,
            effective_exec_ctx,
            cb,
            rt_map,
        );
        if (collect_loop_profile) loop_profile.materialize_runtime_inputs_ns += metalPartitionElapsedNs(runtime_inputs_start_ns, metalPartitionNowNs());
        if (trace_progress) std.debug.print("metal_partition_progress: phase=materialize_runtime_inputs_end partition={d}\n", .{partition_index});
        if (trace_nodes) std.debug.print("graph_executor_node_trace: materialize_runtime_inputs_end partition={d}\n", .{partition_index});

        if (trace_nodes) std.debug.print("graph_executor_node_trace: materialize_parameters_begin partition={d}\n", .{partition_index});
        if (trace_progress) std.debug.print("metal_partition_progress: phase=materialize_parameters_begin partition={d}\n", .{partition_index});
        const parameters_start_ns = if (collect_loop_profile) metalPartitionNowNs() else 0;
        try materializePartitionParameters(
            graph,
            cb,
            values,
            value_device,
            node_ids,
            reachable,
            device_id,
            rt_map,
            exec_ctx.stats,
        );
        if (collect_loop_profile) loop_profile.materialize_parameters_ns += metalPartitionElapsedNs(parameters_start_ns, metalPartitionNowNs());
        if (trace_progress) std.debug.print("metal_partition_progress: phase=materialize_parameters_end partition={d}\n", .{partition_index});
        if (trace_nodes) std.debug.print("graph_executor_node_trace: materialize_parameters_end partition={d}\n", .{partition_index});

        if (trace_nodes) std.debug.print("graph_executor_node_trace: materialize_constants_begin partition={d}\n", .{partition_index});
        if (trace_progress) std.debug.print("metal_partition_progress: phase=materialize_constants_begin partition={d}\n", .{partition_index});
        const constants_start_ns = if (collect_loop_profile) metalPartitionNowNs() else 0;
        try materializePartitionConstants(
            graph,
            cb,
            values,
            value_device,
            node_ids,
            reachable,
            device_id,
        );
        if (collect_loop_profile) loop_profile.materialize_constants_ns += metalPartitionElapsedNs(constants_start_ns, metalPartitionNowNs());
        if (trace_progress) std.debug.print("metal_partition_progress: phase=materialize_constants_end partition={d}\n", .{partition_index});
        if (trace_nodes) std.debug.print("graph_executor_node_trace: materialize_constants_end partition={d}\n", .{partition_index});

        if (trace_nodes) std.debug.print("graph_executor_node_trace: begin_frame_begin partition={d}\n", .{partition_index});
        if (trace_progress) std.debug.print("metal_partition_progress: phase=begin_frame_begin partition={d}\n", .{partition_index});
        const begin_frame_start_ns = if (collect_loop_profile) metalPartitionNowNs() else 0;
        var frame_active = if (metalPartitionFrameDisabled()) false else try cb.decoderRuntimeBeginFrame();
        errdefer if (frame_active) cb.decoderRuntimeCancelFrame() catch {};
        if (collect_loop_profile) loop_profile.begin_frame_ns += metalPartitionElapsedNs(begin_frame_start_ns, metalPartitionNowNs());
        if (trace_progress) std.debug.print("metal_partition_progress: phase=begin_frame_end partition={d} active={}\n", .{ partition_index, frame_active });
        if (trace_nodes) std.debug.print("graph_executor_node_trace: begin_frame_end partition={d} active={}\n", .{ partition_index, frame_active });
        const planned_scope = if (frame_active and !metalPartitionPlannedScopeDisabled())
            try metal_compute_mod.MetalCompute.beginPlannedGraphScope(cb, .ffn)
        else
            metal_compute_mod.MetalCompute.PlannedGraphScope{};
        errdefer metal_compute_mod.MetalCompute.endPlannedGraphScope(cb, planned_scope) catch {};

        var exec_state = interpreter.ExecState{
            .attention_layer = if (exec_ctx.attention_layer) |layer| layer.* else 0,
            .options = options,
            .last_use = last_use,
            .pair_second = if (exec_ctx.pair_second) |pair| pair.* else null,
        };
        defer exec_state.freeMoeState();

        const skipped_nodes = try allocator.alloc(bool, values.len);
        defer allocator.free(skipped_nodes);
        @memset(skipped_nodes, false);

        var transient_runtime_region_plan: ?RuntimeRegionPlan = null;
        defer if (transient_runtime_region_plan) |*plan| plan.deinit(allocator);
        const runtime_region_plan_start_ns = if (collect_loop_profile) metalPartitionNowNs() else 0;
        const runtime_region_plan = try self.runtimeRegionPlan(
            allocator,
            graph,
            node_ids,
            values.len,
            reachable,
            last_use,
            exec_ctx.stats,
            &transient_runtime_region_plan,
        );
        if (collect_loop_profile) loop_profile.runtime_region_plan_ns += metalPartitionElapsedNs(runtime_region_plan_start_ns, metalPartitionNowNs());
        if (trace_progress) std.debug.print("metal_partition_progress: phase=runtime_region_plan_ready partition={d} regions={d}\n", .{ partition_index, runtime_region_plan.region_count });
        if (traceRuntimeRegionsEnabled()) printRuntimeRegionPlanSummary(graph, runtime_region_plan, partition_index);
        if (metalPartitionOpRunsEnabled()) printMetalPartitionOpRuns(graph, node_ids, reachable, last_use, partition_index);
        if (exec_ctx.stats) |stats| {
            stats.runtime_region_plan_active_regions += runtime_region_plan.region_count;
            stats.runtime_region_plan_covered_nodes += runtime_region_plan.covered_node_count;
            stats.runtime_region_plan_elided_nodes += runtime_region_plan.elided_node_count;
            recordRuntimeFrameEligibilityStats(stats, analyzeRuntimeFrameEligibility(runtime_region_plan));
            if (runtimeFrameMetadataFromPlan(graph, runtime_region_plan) != null) {
                stats.runtime_frame_metadata_ready += 1;
            }
        }

        var node_pos: usize = 0;
        while (node_pos < node_ids.len) : (node_pos += 1) {
            const node_id = node_ids[node_pos];
            if (collect_loop_profile) loop_profile.nodes += 1;
            const i: usize = @intCast(node_id);
            if (i >= reachable.len or !reachable[i]) continue;
            if (i < skipped_nodes.len and skipped_nodes[i]) continue;

            if (rt_map.contains(node_id)) {
                value_device[i] = device_id;
                continue;
            }

            if (graph.node(node_id).op == .parameter and values[i] != null) {
                value_device[i] = device_id;
                continue;
            }

            if (graph.node(node_id).op == .fused_from_float32) continue;

            if (shouldDeferScaleMulForAdd(graph, node_id, reachable, last_use)) {
                skipped_nodes[i] = true;
                continue;
            }
            if (shouldDeferElementwiseMulForAdd(graph, node_id, reachable, last_use)) {
                skipped_nodes[i] = true;
                continue;
            }
            if (isPreMaterializedConstantOp(graph.node(node_id).op)) {
                if (values[i] != null) {
                    if (exec_ctx.stats) |stats| {
                        stats.constant_materializations += 1;
                        if (!collect_residency_stats) {
                            stats.device_resident_outputs += 1;
                        } else if (isMetalResidentOrQuantizedDescriptor(cb, values[i].?)) {
                            stats.device_resident_outputs += 1;
                        } else {
                            stats.host_materialized_outputs += 1;
                            if (traceMetalHostOutputsEnabled()) traceMetalHostOutput(graph, node_id, "pre_materialized_constant");
                        }
                    }
                    value_device[i] = device_id;
                    continue;
                }
            }

            if (shouldDeferTransposeForLinearDot(graph, node_id, reachable, last_use)) {
                skipped_nodes[i] = true;
                continue;
            }

            const trace_node_progress = traceMetalGraphProgressNode(node_pos, progress_interval, progress_start, progress_end);
            if (trace_node_progress) {
                printMetalProgressNode("node_begin", graph, partition_index, node_pos, node_ids.len, node_id);
            }
            if (trace_nodes) printMetalNodeTraceBegin(graph, node_id);
            if (trace_nodes) printMetalNodeTraceInputs(graph, cb, values, node_id);
            const op_start_ns = if (collect_op_stats) metalPartitionNowNs() else 0;
            const execution_start_ns = if (collect_loop_profile) metalPartitionNowNs() else 0;

            const op_plan = partition_plan.operatorPlanForNode(node_id);
            var execution_kind: ?MetalExecutionKind = null;
            if (trace_node_progress) std.debug.print("metal_partition_progress: phase=planned_region_begin partition={d} pos={d} node={}\n", .{ partition_index, node_pos, node_id });
            const runtime_region = runtime_region_plan.regionAt(node_pos, node_id, node_ids);
            const region_start_ns = if (collect_op_stats) metalPartitionNowNs() else 0;
            if (try tryExecutePlannedRuntimeRegion(
                runtime_region,
                runtime_region_plan.preparedPtrAt(node_pos, node_id, node_ids),
                allocator,
                graph,
                cb,
                values,
                value_device,
                node_ids,
                node_pos,
                reachable,
                device_id,
                effective_exec_ctx,
                &exec_state,
                skipped_nodes,
                last_use,
                rt_map,
                donated,
            )) {
                if (trace_node_progress) std.debug.print("metal_partition_progress: phase=planned_region_hit partition={d} pos={d} node={}\n", .{ partition_index, node_pos, node_id });
                execution_kind = .command;
                if (exec_ctx.stats) |stats| stats.runtime_region_plan_dispatches += 1;
                if (collect_op_stats) {
                    op_execution_stats.recordRuntimeRegion(@tagName(runtime_region), metalPartitionElapsedNs(region_start_ns, metalPartitionNowNs()));
                }
            } else {
                if (trace_node_progress) std.debug.print("metal_partition_progress: phase=planned_region_miss partition={d} pos={d} node={}\n", .{ partition_index, node_pos, node_id });
                if (trace_node_progress) std.debug.print("metal_partition_progress: phase=fused_pattern_begin partition={d} pos={d} node={}\n", .{ partition_index, node_pos, node_id });
                if (try tryExecuteFusedMetalGraphPattern(
                    allocator,
                    graph,
                    cb,
                    values,
                    value_device,
                    node_ids,
                    node_pos,
                    reachable,
                    device_id,
                    effective_exec_ctx,
                    &exec_state,
                    skipped_nodes,
                    last_use,
                    rt_map,
                    donated,
                )) {
                    if (trace_node_progress) std.debug.print("metal_partition_progress: phase=fused_pattern_hit partition={d} pos={d} node={}\n", .{ partition_index, node_pos, node_id });
                    execution_kind = .command;
                } else {
                    if (trace_node_progress) std.debug.print("metal_partition_progress: phase=fused_pattern_miss partition={d} pos={d} node={}\n", .{ partition_index, node_pos, node_id });
                    if (trace_node_progress) std.debug.print("metal_partition_progress: phase=metal_command_begin partition={d} pos={d} node={}\n", .{ partition_index, node_pos, node_id });
                    const command_output_opt = if (!metalPartitionRuntimeCommandsDisabled())
                        try tryExecuteMetalCommand(allocator, graph, cb, values, node_id, op_plan, &exec_state)
                    else
                        null;
                    if (command_output_opt) |command_output| {
                        if (trace_node_progress) std.debug.print("metal_partition_progress: phase=metal_command_hit partition={d} pos={d} node={}\n", .{ partition_index, node_pos, node_id });
                        values[i] = command_output;
                        execution_kind = classifyMetalExecutionKind(graph, cb, values, node_id);
                    } else {
                        if (trace_node_progress) std.debug.print("metal_partition_progress: phase=metal_command_miss partition={d} pos={d} node={}\n", .{ partition_index, node_pos, node_id });
                        if (trace_node_progress) std.debug.print("metal_partition_progress: phase=interpreter_begin partition={d} pos={d} node={}\n", .{ partition_index, node_pos, node_id });
                        if (trace_nodes or collect_op_stats) printInterpreterFallbackNullInputs(graph, values, node_id, partition_index, node_pos);
                        values[i] = try interpreter.executeNode(graph, cb, values, node_id, &exec_state);
                        if (trace_node_progress) std.debug.print("metal_partition_progress: phase=interpreter_end partition={d} pos={d} node={}\n", .{ partition_index, node_pos, node_id });
                    }
                }
            }
            if (collect_loop_profile) {
                loop_profile.executed_nodes += 1;
                loop_profile.execution_ns += metalPartitionElapsedNs(execution_start_ns, metalPartitionNowNs());
            }
            const stats_start_ns = if (collect_loop_profile) metalPartitionNowNs() else 0;
            if (graph.node(node_id).op == .constant) {
                if (values[i]) |current| {
                    if (!isMetalDeviceResident(cb, current)) {
                        if (try makeMetalDeviceResident(cb, current)) |device_value| {
                            if (device_value != current) {
                                cb.free(current);
                                values[i] = device_value;
                            }
                        }
                    }
                }
            }
            if (exec_ctx.stats) |stats| {
                const op_name = @tagName(graph.node(node_id).op);
                const elapsed_ns = if (collect_op_stats) metalPartitionElapsedNs(op_start_ns, metalPartitionNowNs()) else 0;
                if (execution_kind) |kind| {
                    switch (kind) {
                        .command => {
                            stats.backend_command_dispatches += 1;
                            if (op_plan != null) stats.planned_operator_dispatches += 1;
                            if (collect_op_stats) {
                                op_execution_stats.recordCommand(op_name, elapsed_ns);
                                op_execution_stats.recordCommandClass(graph, node_id, node_pos, elapsed_ns);
                                op_execution_stats.recordDotCommand(graph, node_id, node_pos, elapsed_ns);
                            }
                        },
                        .metadata_alias => stats.metadata_aliases += 1,
                        .descriptor_materialization => stats.descriptor_materializations += 1,
                        .constant_materialization => stats.constant_materializations += 1,
                    }
                } else {
                    stats.interpreter_fallbacks += 1;
                    if (collect_op_stats) op_execution_stats.recordFallback(op_name, elapsed_ns);
                }
                if (collect_residency_stats) {
                    const output_resident = isMetalResidentOrQuantizedDescriptor(cb, values[i].?);
                    if (output_resident) {
                        stats.device_resident_outputs += 1;
                    } else {
                        stats.host_materialized_outputs += 1;
                        if (collect_op_stats) {
                            op_execution_stats.recordHostOutput(op_name, elapsed_ns);
                            op_execution_stats.recordHostOutputReason(hostOutputReasonName(execution_kind), elapsed_ns);
                        }
                        if (traceMetalHostOutputsEnabled()) traceMetalHostOutput(graph, node_id, hostOutputReasonName(execution_kind));
                    }
                    recordGemmaRuntimeResidency(stats, graph, node_id, output_resident);
                } else if (execution_kind != null) {
                    stats.device_resident_outputs += 1;
                } else {
                    stats.host_materialized_outputs += 1;
                }
            }
            if (collect_loop_profile) loop_profile.stats_ns += metalPartitionElapsedNs(stats_start_ns, metalPartitionNowNs());
            value_device[i] = device_id;
            const traced_command = if (execution_kind) |kind| kind == .command else false;
            if (trace_nodes) printMetalNodeTraceEnd(graph, cb, node_id, values[i].?, traced_command);

            const alias_clone_start_ns = if (collect_loop_profile) metalPartitionNowNs() else 0;
            try interpreter.cloneOutputIfAliasedInputWouldBeFreed(
                allocator,
                graph,
                cb,
                values,
                node_id,
                last_use,
                rt_map,
                donated,
            );
            if (collect_loop_profile) loop_profile.alias_clone_ns += metalPartitionElapsedNs(alias_clone_start_ns, metalPartitionNowNs());

            const free_expired_start_ns = if (collect_loop_profile) metalPartitionNowNs() else 0;
            try freeExpiredInputs(
                allocator,
                graph,
                cb,
                values,
                value_device,
                node_id,
                device_id,
                last_use,
                runtime_region_plan,
                rt_map,
                donated,
                effective_exec_ctx,
            );
            if (collect_loop_profile) loop_profile.free_expired_ns += metalPartitionElapsedNs(free_expired_start_ns, metalPartitionNowNs());

            if (i < skipped_nodes.len and skipped_nodes[i]) {
                values[i] = null;
            }
            if (trace_node_progress) {
                printMetalProgressNode("node_end", graph, partition_index, node_pos, node_ids.len, node_id);
            }
        }

        if (frame_active) {
            try metal_compute_mod.MetalCompute.endPlannedGraphScope(cb, planned_scope);
            if (trace_progress) std.debug.print("metal_partition_progress: phase=submit_frame_begin partition={d}\n", .{partition_index});
            const submit_frame_start_ns = if (collect_loop_profile) metalPartitionNowNs() else 0;
            try cb.decoderRuntimeSubmitAndWaitFrame();
            if (collect_loop_profile) loop_profile.submit_frame_ns += metalPartitionElapsedNs(submit_frame_start_ns, metalPartitionNowNs());
            frame_active = false;
            if (trace_progress) std.debug.print("metal_partition_progress: phase=submit_frame_end partition={d}\n", .{partition_index});
        } else {
            try metal_compute_mod.MetalCompute.endPlannedGraphScope(cb, planned_scope);
        }

        if (exec_ctx.materialize_boundary_outputs) {
            if (trace_progress) std.debug.print("metal_partition_progress: phase=materialize_boundary_outputs_begin partition={d}\n", .{partition_index});
            if (exec_ctx.stats) |stats| {
                stats.boundary_output_materializations += countPartitionBoundaryOutputs(partition_view);
            }
            const boundary_outputs_start_ns = if (collect_loop_profile) metalPartitionNowNs() else 0;
            try evalPartitionBoundaryOutputs(cb, values, partition_view);
            if (collect_loop_profile) loop_profile.boundary_outputs_ns += metalPartitionElapsedNs(boundary_outputs_start_ns, metalPartitionNowNs());
            if (trace_progress) std.debug.print("metal_partition_progress: phase=materialize_boundary_outputs_end partition={d}\n", .{partition_index});
        }

        if (exec_ctx.attention_layer) |layer| layer.* = exec_state.attention_layer;
        if (exec_ctx.pair_second) |pair| pair.* = exec_state.pair_second;
        if (collect_op_stats) {
            printOpExecutionStats("metal_partition_command_ops", &op_execution_stats.command_counts, op_execution_stats.command_used);
            printCommandExecutionStats("metal_partition_command_classes", &op_execution_stats.command_class_counts, op_execution_stats.command_class_used);
            printOpExecutionStats("metal_partition_runtime_regions", &op_execution_stats.runtime_region_counts, op_execution_stats.runtime_region_used);
            printDotShapeExecutionStats("metal_partition_command_dot_shapes", &op_execution_stats.dot_command_shapes, op_execution_stats.dot_command_shape_used);
            printOpExecutionStats("metal_partition_fallback_ops", &op_execution_stats.fallback_counts, op_execution_stats.fallback_used);
            printOpExecutionStats("metal_partition_host_output_ops", &op_execution_stats.host_output_counts, op_execution_stats.host_output_used);
            printOpExecutionStats("metal_partition_host_output_reasons", &op_execution_stats.host_output_reason_counts, op_execution_stats.host_output_reason_used);
        }
        if (collect_loop_profile) loop_profile.print("metal_partition_loop_profile");
        if (trace_progress) std.debug.print("metal_partition_progress: phase=executor_end partition={d}\n", .{partition_index});
    }
};

fn recordOpCount(counts: []OpExecutionCount, used: *usize, name: []const u8, elapsed_ns: u64) void {
    for (counts[0..used.*]) |*entry| {
        if (std.mem.eql(u8, entry.name, name)) {
            entry.count += 1;
            entry.total_ns += elapsed_ns;
            return;
        }
    }
    if (used.* >= counts.len) return;
    counts[used.*] = .{ .name = name, .count = 1, .total_ns = elapsed_ns };
    used.* += 1;
}

fn sortOpExecutionCounts(counts: []OpExecutionCount) void {
    std.mem.sort(OpExecutionCount, counts, {}, struct {
        fn lessThan(_: void, a: OpExecutionCount, b: OpExecutionCount) bool {
            if (a.total_ns == b.total_ns) {
                if (a.count == b.count) return std.mem.lessThan(u8, a.name, b.name);
                return a.count > b.count;
            }
            return a.total_ns > b.total_ns;
        }
    }.lessThan);
}

fn printOpExecutionStats(label: []const u8, counts: []OpExecutionCount, used: usize) void {
    var sorted_buf = [_]OpExecutionCount{.{}} ** 96;
    const n = @min(used, sorted_buf.len);
    @memcpy(sorted_buf[0..n], counts[0..n]);
    sortOpExecutionCounts(sorted_buf[0..n]);
    std.debug.print("{s}: ", .{label});
    if (n == 0) {
        std.debug.print("none\n", .{});
        return;
    }
    const limit = @min(n, 16);
    for (sorted_buf[0..limit], 0..) |entry, idx| {
        if (idx > 0) std.debug.print(",", .{});
        const avg_ms = if (entry.count == 0) 0.0 else nsToMs(entry.total_ns) / @as(f64, @floatFromInt(entry.count));
        std.debug.print("{s}:count={d}:total_ms={d:.3}:avg_ms={d:.3}", .{ entry.name, entry.count, nsToMs(entry.total_ns), avg_ms });
    }
    if (n > limit) std.debug.print(",...", .{});
    std.debug.print("\n", .{});
}

const CommandExecutionSummary = struct {
    op_name: []const u8 = "",
    phase: []const u8 = "",
    family: []const u8 = "",
    source_op: []const u8 = "",
    first_node: NodeId = null_node,
    first_pos: usize = 0,
    last_node: NodeId = null_node,
    last_pos: usize = 0,
    count: usize = 0,
    total_ns: u64 = 0,
};

fn sameCommandExecutionSummary(a: CommandExecutionSummary, b: CommandExecutionSummary) bool {
    return std.mem.eql(u8, a.op_name, b.op_name) and
        std.mem.eql(u8, a.phase, b.phase) and
        std.mem.eql(u8, a.family, b.family) and
        std.mem.eql(u8, a.source_op, b.source_op);
}

fn recordCommandExecutionSummary(
    graph: *const Graph,
    node_id: NodeId,
    node_pos: usize,
    summaries: *[160]CommandExecutionSummary,
    used: *usize,
    elapsed_ns: u64,
) void {
    if (node_id == null_node or node_id >= graph.nodeCount()) return;
    const node = graph.node(node_id);
    const classification = commandSourceClassification(graph, node_id, 4);
    const summary = CommandExecutionSummary{
        .op_name = @tagName(std.meta.activeTag(node.op)),
        .phase = classification.phase,
        .family = classification.family,
        .source_op = classification.source_op,
        .first_node = node_id,
        .first_pos = node_pos,
        .last_node = node_id,
        .last_pos = node_pos,
    };
    for (summaries[0..used.*]) |*entry| {
        if (!sameCommandExecutionSummary(entry.*, summary)) continue;
        entry.count += 1;
        entry.total_ns += elapsed_ns;
        entry.last_node = node_id;
        entry.last_pos = node_pos;
        return;
    }
    if (used.* >= summaries.len) return;
    summaries[used.*] = summary;
    summaries[used.*].count = 1;
    summaries[used.*].total_ns = elapsed_ns;
    used.* += 1;
}

fn sortCommandExecutionSummaries(summaries: []CommandExecutionSummary) void {
    std.mem.sort(CommandExecutionSummary, summaries, {}, struct {
        fn lessThan(_: void, a: CommandExecutionSummary, b: CommandExecutionSummary) bool {
            if (a.total_ns == b.total_ns) {
                if (a.count == b.count) return std.mem.lessThan(u8, a.op_name, b.op_name);
                return a.count > b.count;
            }
            return a.total_ns > b.total_ns;
        }
    }.lessThan);
}

fn printCommandExecutionStats(label: []const u8, summaries: []const CommandExecutionSummary, used: usize) void {
    var sorted_buf = [_]CommandExecutionSummary{.{}} ** 160;
    const n = @min(used, sorted_buf.len);
    @memcpy(sorted_buf[0..n], summaries[0..n]);
    sortCommandExecutionSummaries(sorted_buf[0..n]);
    std.debug.print("{s}: ", .{label});
    if (n == 0) {
        std.debug.print("none\n", .{});
        return;
    }
    const limit = @min(n, 24);
    for (sorted_buf[0..limit], 0..) |entry, idx| {
        if (idx > 0) std.debug.print(",", .{});
        const avg_ms = if (entry.count == 0) 0.0 else nsToMs(entry.total_ns) / @as(f64, @floatFromInt(entry.count));
        std.debug.print(
            "{s}:phase={s}:family={s}:source={s}:count={d}:total_ms={d:.3}:avg_ms={d:.3}:pos={d}-{d}:node={}-{}",
            .{
                entry.op_name,
                entry.phase,
                entry.family,
                entry.source_op,
                entry.count,
                nsToMs(entry.total_ns),
                avg_ms,
                entry.first_pos,
                entry.last_pos,
                entry.first_node,
                entry.last_node,
            },
        );
    }
    if (n > limit) std.debug.print(",...", .{});
    std.debug.print("\n", .{});
}

fn sameDotShapeExecutionSummary(a: DotShapeExecutionSummary, b: DotShapeExecutionSummary) bool {
    return a.lhs0 == b.lhs0 and
        a.lhs1 == b.lhs1 and
        a.rhs0 == b.rhs0 and
        a.rhs1 == b.rhs1 and
        a.out0 == b.out0 and
        a.out1 == b.out1 and
        a.rhs_transpose == b.rhs_transpose and
        a.rhs_parameter == b.rhs_parameter and
        a.rhs_lora == b.rhs_lora and
        std.mem.eql(u8, a.phase, b.phase) and
        std.mem.eql(u8, a.family, b.family) and
        std.mem.eql(u8, a.lhs_source_op, b.lhs_source_op) and
        std.mem.eql(u8, a.rhs_source_op, b.rhs_source_op);
}

fn recordDotShapeExecutionSummary(
    graph: *const Graph,
    node_id: NodeId,
    node_pos: usize,
    summaries: *[128]DotShapeExecutionSummary,
    used: *usize,
    elapsed_ns: u64,
) void {
    if (node_id == null_node or node_id >= graph.nodeCount()) return;
    const node = graph.node(node_id);
    switch (node.op) {
        .dot_general => {},
        else => return,
    }
    if (node.num_inputs < 2) return;
    const lhs_id = node.inputs[0];
    const rhs_id = node.inputs[1];
    if (lhs_id == null_node or rhs_id == null_node or lhs_id >= graph.nodeCount() or rhs_id >= graph.nodeCount()) return;
    const lhs_shape = graph.node(lhs_id).output_shape;
    const rhs_shape = graph.node(rhs_id).output_shape;
    const out_shape = node.output_shape;
    if (lhs_shape.rank() != 2 or rhs_shape.rank() != 2 or out_shape.rank() != 2) return;

    const lhs_source = dotSourceInfo(graph, lhs_id);
    const rhs_source = dotSourceInfo(graph, rhs_id);

    const summary = DotShapeExecutionSummary{
        .lhs0 = lhs_shape.dims[0],
        .lhs1 = lhs_shape.dims[1],
        .rhs0 = rhs_shape.dims[0],
        .rhs1 = rhs_shape.dims[1],
        .out0 = out_shape.dims[0],
        .out1 = out_shape.dims[1],
        .rhs_transpose = rhs_source.is_transpose,
        .rhs_parameter = rhs_source.is_parameter,
        .rhs_lora = rhs_source.is_lora,
        .phase = classifyDotPhase(lhs_source, rhs_source),
        .family = classifyDotParameterFamily(rhs_source.parameter_name orelse lhs_source.parameter_name),
        .lhs_source_op = lhs_source.op_name,
        .rhs_source_op = rhs_source.op_name,
        .first_node = node_id,
        .first_pos = node_pos,
        .last_node = node_id,
        .last_pos = node_pos,
    };
    for (summaries[0..used.*]) |*entry| {
        if (!sameDotShapeExecutionSummary(entry.*, summary)) continue;
        entry.count += 1;
        entry.total_ns += elapsed_ns;
        entry.last_node = node_id;
        entry.last_pos = node_pos;
        return;
    }
    if (used.* >= summaries.len) return;
    summaries[used.*] = summary;
    summaries[used.*].count = 1;
    summaries[used.*].total_ns = elapsed_ns;
    used.* += 1;
}

fn sortDotShapeExecutionSummaries(summaries: []DotShapeExecutionSummary) void {
    std.mem.sort(DotShapeExecutionSummary, summaries, {}, struct {
        fn lessThan(_: void, a: DotShapeExecutionSummary, b: DotShapeExecutionSummary) bool {
            if (a.total_ns == b.total_ns) {
                if (a.count == b.count) {
                    if (a.lhs0 != b.lhs0) return a.lhs0 < b.lhs0;
                    if (a.lhs1 != b.lhs1) return a.lhs1 < b.lhs1;
                    if (a.rhs0 != b.rhs0) return a.rhs0 < b.rhs0;
                    return a.rhs1 < b.rhs1;
                }
                return a.count > b.count;
            }
            return a.total_ns > b.total_ns;
        }
    }.lessThan);
}

fn printDotShapeExecutionStats(label: []const u8, summaries: []const DotShapeExecutionSummary, used: usize) void {
    var sorted_buf = [_]DotShapeExecutionSummary{.{}} ** 128;
    const n = @min(used, sorted_buf.len);
    @memcpy(sorted_buf[0..n], summaries[0..n]);
    sortDotShapeExecutionSummaries(sorted_buf[0..n]);
    std.debug.print("{s}: ", .{label});
    if (n == 0) {
        std.debug.print("none\n", .{});
        return;
    }
    const limit = @min(n, 16);
    for (sorted_buf[0..limit], 0..) |entry, idx| {
        if (idx > 0) std.debug.print(",", .{});
        const avg_ms = if (entry.count == 0) 0.0 else nsToMs(entry.total_ns) / @as(f64, @floatFromInt(entry.count));
        std.debug.print(
            "{d}x{d}*{d}x{d}->{d}x{d}:count={d}:total_ms={d:.3}:avg_ms={d:.3}:pos={d}-{d}:node={}-{}:phase={s}:family={s}:lhs={s}:rhs={s}:rhs_transpose={}:rhs_parameter={}:rhs_lora={}",
            .{
                entry.lhs0,
                entry.lhs1,
                entry.rhs0,
                entry.rhs1,
                entry.out0,
                entry.out1,
                entry.count,
                nsToMs(entry.total_ns),
                avg_ms,
                entry.first_pos,
                entry.last_pos,
                entry.first_node,
                entry.last_node,
                entry.phase,
                entry.family,
                entry.lhs_source_op,
                entry.rhs_source_op,
                entry.rhs_transpose,
                entry.rhs_parameter,
                entry.rhs_lora,
            },
        );
    }
    if (n > limit) std.debug.print(",...", .{});
    std.debug.print("\n", .{});
}

fn printInterpreterFallbackNullInputs(
    graph: *const Graph,
    values: []?CT,
    node_id: NodeId,
    partition_index: usize,
    node_pos: usize,
) void {
    const node = graph.node(node_id);
    var missing = false;
    for (node.getInputs()) |input_id| {
        const input_index: usize = @intCast(input_id);
        if (input_index >= values.len or values[input_index] == null) {
            missing = true;
            break;
        }
    }
    if (!missing) return;
    std.debug.print("metal_partition_interpreter_null_inputs: partition={d} pos={d} node={} op={s} inputs=", .{
        partition_index,
        node_pos,
        node_id,
        @tagName(node.op),
    });
    for (node.getInputs(), 0..) |input_id, idx| {
        if (idx > 0) std.debug.print(",", .{});
        const input_index: usize = @intCast(input_id);
        const input_op = if (input_id < graph.nodeCount()) @tagName(graph.node(input_id).op) else "invalid";
        const state = if (input_index < values.len and values[input_index] != null) "set" else "null";
        std.debug.print("{}:{s}:{s}", .{ input_id, input_op, state });
    }
    std.debug.print("\n", .{});
}

fn metalPartitionOpStatsEnabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_PARTITION_OP_STATS", false);
}

fn metalPartitionLoopProfileEnabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_PARTITION_LOOP_PROFILE", false);
}

fn metalPartitionResidencyStatsEnabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_PARTITION_RESIDENCY_STATS", false);
}

fn traceMetalHostOutputsEnabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_TRACE_HOST_OUTPUTS", false);
}

fn metalPartitionOpRunsEnabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_PARTITION_OP_RUNS", false);
}

fn hostOutputReasonName(execution_kind: ?MetalExecutionKind) []const u8 {
    return if (execution_kind) |kind| switch (kind) {
        .command => "command_host_output",
        .metadata_alias => "metadata_alias_host_output",
        .descriptor_materialization => "descriptor_materialization_host_output",
        .constant_materialization => "constant_materialization_host_output",
    } else "interpreter_fallback_host_output";
}

fn traceMetalHostOutput(graph: *const Graph, node_id: NodeId, reason: []const u8) void {
    const node = graph.node(node_id);
    std.debug.print(
        "metal_partition_host_output_trace: reason={s} node={d} op={s} shape={any}\n",
        .{ reason, node_id, @tagName(node.op), node.output_shape },
    );
}

const OpRunSummary = struct {
    name: []const u8 = "",
    count: usize = 0,
};

const LongOpRun = struct {
    name: []const u8 = "",
    start_pos: usize = 0,
    end_pos: usize = 0,
    count: usize = 0,
};

const DotShapeRunSummary = struct {
    lhs0: i64 = 0,
    lhs1: i64 = 0,
    rhs0: i64 = 0,
    rhs1: i64 = 0,
    out0: i64 = 0,
    out1: i64 = 0,
    rhs_transpose: bool = false,
    rhs_parameter: bool = false,
    rhs_lora: bool = false,
    raw_linear_match: bool = false,
    phase: []const u8 = "",
    family: []const u8 = "",
    lhs_source_op: []const u8 = "",
    rhs_source_op: []const u8 = "",
    count: usize = 0,
};

const DotShapeExecutionSummary = struct {
    lhs0: i64 = 0,
    lhs1: i64 = 0,
    rhs0: i64 = 0,
    rhs1: i64 = 0,
    out0: i64 = 0,
    out1: i64 = 0,
    rhs_transpose: bool = false,
    rhs_parameter: bool = false,
    rhs_lora: bool = false,
    phase: []const u8 = "",
    family: []const u8 = "",
    lhs_source_op: []const u8 = "",
    rhs_source_op: []const u8 = "",
    first_node: NodeId = null_node,
    first_pos: usize = 0,
    last_node: NodeId = null_node,
    last_pos: usize = 0,
    count: usize = 0,
    total_ns: u64 = 0,
};

const DotSourceInfo = struct {
    op_name: []const u8 = "",
    is_transpose: bool = false,
    is_parameter: bool = false,
    is_lora: bool = false,
    parameter_name: ?[]const u8 = null,
};

const CommandSourceClassification = struct {
    phase: []const u8 = "activation",
    family: []const u8 = "activation",
    source_op: []const u8 = "none",
};

fn commandSourceClassification(graph: *const Graph, node_id: NodeId, depth: usize) CommandSourceClassification {
    if (node_id == null_node or node_id >= graph.nodeCount()) return .{};
    const node = graph.node(node_id);
    if (std.meta.activeTag(node.op) == .dot_general and node.num_inputs >= 2) {
        const lhs_source = dotSourceInfo(graph, node.inputs[0]);
        const rhs_source = dotSourceInfo(graph, node.inputs[1]);
        return .{
            .phase = classifyDotPhase(lhs_source, rhs_source),
            .family = classifyDotParameterFamily(rhs_source.parameter_name orelse lhs_source.parameter_name),
            .source_op = rhs_source.op_name,
        };
    }
    if (commandParameterName(graph, node_id, depth)) |name| {
        return .{
            .phase = "parameter_ancestry",
            .family = classifyDotParameterFamily(name),
            .source_op = "parameter",
        };
    }
    if (hasTransposedActivationAncestor(graph, node_id, depth)) {
        return .{
            .phase = "activation_transpose",
            .family = "activation",
            .source_op = "transpose(other)",
        };
    }
    return .{
        .phase = "activation",
        .family = "activation",
        .source_op = @tagName(std.meta.activeTag(node.op)),
    };
}

fn commandParameterName(graph: *const Graph, node_id: NodeId, depth: usize) ?[]const u8 {
    if (depth == 0 or node_id == null_node or node_id >= graph.nodeCount()) return null;
    const node = graph.node(node_id);
    if (std.meta.activeTag(node.op) == .parameter) return graph.parameterName(node);
    for (node.getInputs()) |input_id| {
        if (commandParameterName(graph, input_id, depth - 1)) |name| return name;
    }
    return null;
}

fn hasTransposedActivationAncestor(graph: *const Graph, node_id: NodeId, depth: usize) bool {
    if (depth == 0 or node_id == null_node or node_id >= graph.nodeCount()) return false;
    const node = graph.node(node_id);
    if (std.meta.activeTag(node.op) == .transpose) {
        if (node.num_inputs == 0 or node.inputs[0] == null_node or node.inputs[0] >= graph.nodeCount()) return true;
        return std.meta.activeTag(graph.node(node.inputs[0]).op) != .parameter;
    }
    for (node.getInputs()) |input_id| {
        if (hasTransposedActivationAncestor(graph, input_id, depth - 1)) return true;
    }
    return false;
}

fn dotSourceInfo(graph: *const Graph, node_id: NodeId) DotSourceInfo {
    if (node_id == null_node or node_id >= graph.nodeCount()) return .{ .op_name = "invalid" };
    const node = graph.node(node_id);
    switch (node.op) {
        .transpose => {
            if (node.num_inputs == 0 or node.inputs[0] == null_node or node.inputs[0] >= graph.nodeCount()) {
                return .{ .op_name = "transpose", .is_transpose = true };
            }
            const source = graph.node(node.inputs[0]);
            if (std.meta.activeTag(source.op) == .parameter) {
                const name = graph.parameterName(source);
                return .{
                    .op_name = "transpose(parameter)",
                    .is_transpose = true,
                    .is_parameter = true,
                    .is_lora = isLoRAAdapterParameterName(name),
                    .parameter_name = name,
                };
            }
            return .{
                .op_name = if (std.meta.activeTag(source.op) == .dot_general) "transpose(dot_general)" else "transpose(other)",
                .is_transpose = true,
            };
        },
        .parameter => {
            const name = graph.parameterName(node);
            return .{
                .op_name = "parameter",
                .is_parameter = true,
                .is_lora = isLoRAAdapterParameterName(name),
                .parameter_name = name,
            };
        },
        else => return .{ .op_name = @tagName(std.meta.activeTag(node.op)) },
    }
}

fn classifyDotParameterFamily(name_opt: ?[]const u8) []const u8 {
    const name = name_opt orelse return "activation";
    if (std.mem.indexOf(u8, name, "lora_A") != null) return "lora_A";
    if (std.mem.indexOf(u8, name, "lora_B") != null) return "lora_B";
    if (std.mem.indexOf(u8, name, "query") != null or std.mem.indexOf(u8, name, "q_proj") != null) return "attention_q";
    if (std.mem.indexOf(u8, name, "key") != null or std.mem.indexOf(u8, name, "k_proj") != null) return "attention_k";
    if (std.mem.indexOf(u8, name, "value") != null or std.mem.indexOf(u8, name, "v_proj") != null) return "attention_v";
    if (std.mem.indexOf(u8, name, "attention.output.dense") != null or std.mem.indexOf(u8, name, "out_proj") != null) return "attention_out";
    if (std.mem.indexOf(u8, name, "intermediate.dense") != null or std.mem.indexOf(u8, name, "linear1") != null) return "ffn_up";
    if (std.mem.indexOf(u8, name, "output.dense") != null or std.mem.indexOf(u8, name, "linear2") != null) return "ffn_down";
    if (std.mem.indexOf(u8, name, "LayerNorm") != null or std.mem.indexOf(u8, name, "layer_norm") != null or std.mem.indexOf(u8, name, "norm") != null) return "norm";
    if (std.mem.indexOf(u8, name, "embeddings") != null or std.mem.indexOf(u8, name, "embedding") != null) return "embedding";
    if (std.mem.indexOf(u8, name, "classifier") != null or std.mem.indexOf(u8, name, "span_rep") != null or std.mem.indexOf(u8, name, "count_") != null) return "head";
    return "parameter_other";
}

fn classifyDotPhase(lhs: DotSourceInfo, rhs: DotSourceInfo) []const u8 {
    if (rhs.is_parameter) return "forward_parameter";
    if (lhs.is_parameter) return "backward_input";
    if (lhs.is_transpose and !rhs.is_parameter) return "backward_weight";
    if (rhs.is_transpose and !rhs.is_parameter) return "backward_activation";
    return "activation";
}

fn recordOpRunCount(counts: *[96]OpRunSummary, used: *usize, name: []const u8) void {
    for (counts[0..used.*]) |*entry| {
        if (std.mem.eql(u8, entry.name, name)) {
            entry.count += 1;
            return;
        }
    }
    if (used.* >= counts.len) return;
    counts[used.*] = .{ .name = name, .count = 1 };
    used.* += 1;
}

fn insertLongOpRun(runs: *[12]LongOpRun, run: LongOpRun) void {
    if (run.count == 0) return;
    var insert_at: ?usize = null;
    for (runs, 0..) |entry, idx| {
        if (run.count > entry.count) {
            insert_at = idx;
            break;
        }
    }
    const idx = insert_at orelse return;
    var move_idx = runs.len - 1;
    while (move_idx > idx) : (move_idx -= 1) {
        runs[move_idx] = runs[move_idx - 1];
    }
    runs[idx] = run;
}

fn lessOpRunCount(_: void, a: OpRunSummary, b: OpRunSummary) bool {
    if (a.count == b.count) return std.mem.lessThan(u8, a.name, b.name);
    return a.count > b.count;
}

fn sameDotShapeRunSummary(a: DotShapeRunSummary, b: DotShapeRunSummary) bool {
    return a.lhs0 == b.lhs0 and
        a.lhs1 == b.lhs1 and
        a.rhs0 == b.rhs0 and
        a.rhs1 == b.rhs1 and
        a.out0 == b.out0 and
        a.out1 == b.out1 and
        a.rhs_transpose == b.rhs_transpose and
        a.rhs_parameter == b.rhs_parameter and
        a.rhs_lora == b.rhs_lora and
        a.raw_linear_match == b.raw_linear_match and
        std.mem.eql(u8, a.phase, b.phase) and
        std.mem.eql(u8, a.family, b.family) and
        std.mem.eql(u8, a.lhs_source_op, b.lhs_source_op) and
        std.mem.eql(u8, a.rhs_source_op, b.rhs_source_op);
}

fn recordDotShapeRunSummary(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    last_use: []const u32,
    summaries: *[96]DotShapeRunSummary,
    used: *usize,
) void {
    const node_id = node_ids[node_pos];
    const node = graph.node(node_id);
    switch (node.op) {
        .dot_general => {},
        else => return,
    }
    if (node.num_inputs < 2) return;
    const lhs_id = node.inputs[0];
    const rhs_id = node.inputs[1];
    if (lhs_id == null_node or rhs_id == null_node) return;
    const lhs_shape = graph.node(lhs_id).output_shape;
    const rhs_shape = graph.node(rhs_id).output_shape;
    const out_shape = node.output_shape;
    if (lhs_shape.rank() != 2 or rhs_shape.rank() != 2 or out_shape.rank() != 2) return;

    const lhs_source = dotSourceInfo(graph, lhs_id);
    const rhs_source = dotSourceInfo(graph, rhs_id);

    var summary = DotShapeRunSummary{
        .lhs0 = lhs_shape.dims[0],
        .lhs1 = lhs_shape.dims[1],
        .rhs0 = rhs_shape.dims[0],
        .rhs1 = rhs_shape.dims[1],
        .out0 = out_shape.dims[0],
        .out1 = out_shape.dims[1],
        .rhs_transpose = rhs_source.is_transpose,
        .rhs_parameter = rhs_source.is_parameter,
        .rhs_lora = rhs_source.is_lora,
        .raw_linear_match = matchRawLinearDotPattern(graph, node_ids, node_pos, reachable, last_use) != null,
        .phase = classifyDotPhase(lhs_source, rhs_source),
        .family = classifyDotParameterFamily(rhs_source.parameter_name orelse lhs_source.parameter_name),
        .lhs_source_op = lhs_source.op_name,
        .rhs_source_op = rhs_source.op_name,
    };

    for (summaries[0..used.*]) |*entry| {
        if (sameDotShapeRunSummary(entry.*, summary)) {
            entry.count += 1;
            return;
        }
    }
    if (used.* >= summaries.len) return;
    summary.count = 1;
    summaries[used.*] = summary;
    used.* += 1;
}

fn lessDotShapeRunSummary(_: void, a: DotShapeRunSummary, b: DotShapeRunSummary) bool {
    if (a.count == b.count) {
        if (a.lhs0 != b.lhs0) return a.lhs0 < b.lhs0;
        if (a.lhs1 != b.lhs1) return a.lhs1 < b.lhs1;
        if (a.rhs0 != b.rhs0) return a.rhs0 < b.rhs0;
        return a.rhs1 < b.rhs1;
    }
    return a.count > b.count;
}

fn printMetalPartitionOpRuns(
    graph: *const Graph,
    node_ids: []const NodeId,
    reachable: []const bool,
    last_use: []const u32,
    partition_index: usize,
) void {
    var counts = [_]OpRunSummary{.{}} ** 96;
    var counts_used: usize = 0;
    var longest = [_]LongOpRun{.{}} ** 12;
    var dot_shapes = [_]DotShapeRunSummary{.{}} ** 96;
    var dot_shapes_used: usize = 0;

    var reachable_nodes: usize = 0;
    var current_name: []const u8 = "";
    var current_start: usize = 0;
    var current_count: usize = 0;

    for (node_ids, 0..) |node_id, node_pos| {
        const i: usize = @intCast(node_id);
        if (i >= reachable.len or !reachable[i]) continue;
        const name = @tagName(graph.node(node_id).op);
        reachable_nodes += 1;
        recordOpRunCount(&counts, &counts_used, name);
        recordDotShapeRunSummary(graph, node_ids, node_pos, reachable, last_use, &dot_shapes, &dot_shapes_used);

        if (current_count == 0) {
            current_name = name;
            current_start = node_pos;
            current_count = 1;
            continue;
        }
        if (std.mem.eql(u8, current_name, name)) {
            current_count += 1;
            continue;
        }
        insertLongOpRun(&longest, .{
            .name = current_name,
            .start_pos = current_start,
            .end_pos = node_pos - 1,
            .count = current_count,
        });
        current_name = name;
        current_start = node_pos;
        current_count = 1;
    }
    if (current_count != 0) {
        insertLongOpRun(&longest, .{
            .name = current_name,
            .start_pos = current_start,
            .end_pos = if (node_ids.len == 0) 0 else node_ids.len - 1,
            .count = current_count,
        });
    }

    std.sort.pdq(OpRunSummary, counts[0..counts_used], {}, lessOpRunCount);
    std.sort.pdq(DotShapeRunSummary, dot_shapes[0..dot_shapes_used], {}, lessDotShapeRunSummary);

    std.debug.print("metal_partition_op_runs: partition={d} reachable_nodes={d} distinct_ops={d} top=", .{ partition_index, reachable_nodes, counts_used });
    const top_limit = @min(counts_used, 16);
    if (top_limit == 0) {
        std.debug.print("none", .{});
    } else {
        for (counts[0..top_limit], 0..) |entry, idx| {
            if (idx > 0) std.debug.print(",", .{});
            std.debug.print("{s}:{d}", .{ entry.name, entry.count });
        }
    }
    std.debug.print(" long_runs=", .{});
    var printed_runs: usize = 0;
    for (longest) |run| {
        if (run.count == 0) continue;
        if (printed_runs > 0) std.debug.print(",", .{});
        std.debug.print("{s}:{d}@{d}-{d}", .{ run.name, run.count, run.start_pos, run.end_pos });
        printed_runs += 1;
    }
    if (printed_runs == 0) std.debug.print("none", .{});
    std.debug.print("\n", .{});

    std.debug.print("metal_partition_dot_shapes: partition={d} distinct_shapes={d} top=", .{ partition_index, dot_shapes_used });
    const dot_limit = @min(dot_shapes_used, 16);
    if (dot_limit == 0) {
        std.debug.print("none", .{});
    } else {
        for (dot_shapes[0..dot_limit], 0..) |entry, idx| {
            if (idx > 0) std.debug.print(",", .{});
            std.debug.print(
                "{d}x{d}*{d}x{d}->{d}x{d}:count={d}:phase={s}:family={s}:lhs={s}:rhs={s}:rhs_transpose={}:rhs_parameter={}:rhs_lora={}:raw_linear={}",
                .{
                    entry.lhs0,
                    entry.lhs1,
                    entry.rhs0,
                    entry.rhs1,
                    entry.out0,
                    entry.out1,
                    entry.count,
                    entry.phase,
                    entry.family,
                    entry.lhs_source_op,
                    entry.rhs_source_op,
                    entry.rhs_transpose,
                    entry.rhs_parameter,
                    entry.rhs_lora,
                    entry.raw_linear_match,
                },
            );
        }
    }
    std.debug.print("\n", .{});
}

fn metalPartitionNowNs() u64 {
    return platform.time.monotonicNs();
}

fn metalPartitionElapsedNs(start_ns: u64, end_ns: u64) u64 {
    if (end_ns <= start_ns) return 0;
    return end_ns - start_ns;
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn traceMetalGraphNodesEnabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_GRAPH_EXECUTOR_TRACE_NODES", false);
}

fn metalGraphProgressInterval() usize {
    return platform.env.getenvUsize("TERMITE_METAL_PARTITION_PROGRESS_INTERVAL") orelse 0;
}

fn metalGraphProgressStart() usize {
    return platform.env.getenvUsize("TERMITE_METAL_PARTITION_PROGRESS_START") orelse std.math.maxInt(usize);
}

fn metalGraphProgressEnd() usize {
    return platform.env.getenvUsize("TERMITE_METAL_PARTITION_PROGRESS_END") orelse std.math.maxInt(usize);
}

fn metalPartitionFrameDisabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_PARTITION_DISABLE_FRAME", false);
}

fn metalPartitionPlannedScopeDisabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_PARTITION_DISABLE_PLANNED_SCOPE", false);
}

fn partitionViewCacheEnabled() bool {
    if (platform.env.getenvBoolDefault("TERMITE_METAL_DISABLE_PARTITION_VIEW_CACHE", false)) return false;
    if (platform.env.getenvBoolDefault("TERMITE_METAL_ENABLE_PARTITION_VIEW_CACHE", false)) return true;
    return platform.env.getenvBoolDefault("TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR", false) and
        !platform.env.getenvBoolDefault("TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR", false);
}

fn tracePartitionViewCacheEnabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_TRACE_PARTITION_VIEW_CACHE", false);
}

fn metalPartitionFusedPatternsDisabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_PARTITION_DISABLE_FUSED_PATTERNS", false);
}

fn metalPartitionRuntimeCommandsDisabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_PARTITION_DISABLE_RUNTIME_COMMANDS", false);
}

fn fusedPatternProbingDisabled(exec_ctx: PartitionExecutor.ExecutionContext) bool {
    if (metalPartitionFusedPatternsDisabled()) return true;
    if (exec_ctx.options) |options| return options.skip_metal_fused_patterns;
    return false;
}

fn traceMetalGraphProgressNode(node_pos: usize, interval: usize, start: usize, end: usize) bool {
    if (start != std.math.maxInt(usize) and node_pos >= start and node_pos <= end) return true;
    return interval != 0 and (node_pos == 0 or node_pos % interval == 0);
}

fn printMetalProgressNode(
    phase: []const u8,
    graph: *const Graph,
    partition_index: usize,
    node_pos: usize,
    node_count: usize,
    node_id: NodeId,
) void {
    const n = graph.node(node_id);
    const inputs = n.getInputs();
    const in0 = if (inputs.len > 0) inputs[0] else null_node;
    const in1 = if (inputs.len > 1) inputs[1] else null_node;
    const in2 = if (inputs.len > 2) inputs[2] else null_node;
    std.debug.print(
        "metal_partition_progress: phase={s} partition={d} pos={d}/{d} node={} op={s} out_shape={any} in0={} in0_op={s} in0_shape={any} in1={} in1_op={s} in1_shape={any} in2={} in2_op={s} in2_shape={any}\n",
        .{
            phase,
            partition_index,
            node_pos,
            node_count,
            node_id,
            @tagName(n.op),
            n.output_shape,
            in0,
            if (in0 != null_node) @tagName(graph.node(in0).op) else "none",
            if (in0 != null_node) graph.node(in0).output_shape else Shape.scalar(.f32),
            in1,
            if (in1 != null_node) @tagName(graph.node(in1).op) else "none",
            if (in1 != null_node) graph.node(in1).output_shape else Shape.scalar(.f32),
            in2,
            if (in2 != null_node) @tagName(graph.node(in2).op) else "none",
            if (in2 != null_node) graph.node(in2).output_shape else Shape.scalar(.f32),
        },
    );
}

fn traceMetalGraphFusionsEnabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_TRACE_GRAPH_FUSIONS", false);
}

fn traceRuntimeRegionsEnabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_TRACE_RUNTIME_REGIONS", false);
}

fn traceLoraQkvMatchingEnabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_TRACE_LORA_QKV_MATCH", false);
}

fn traceDebertaAttentionMatchingEnabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_TRACE_DEBERTA_ATTENTION_MATCH", false);
}

fn debertaAttentionRuntimeRegionEnabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_ENABLE_DEBERTA_ATTENTION_RUNTIME_REGION", false);
}

fn runtimeRegionPlanDisabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_DISABLE_RUNTIME_REGION_PLAN", false);
}

fn loraBackwardRuntimeRegionEnabled() bool {
    if (platform.env.getenvBoolDefault("TERMITE_METAL_DISABLE_LORA_BACKWARD_RUNTIME_REGION", false)) return false;
    return platform.env.getenvBoolDefault("TERMITE_METAL_ENABLE_LORA_BACKWARD_RUNTIME_REGION", true);
}

fn ffnGeluBackwardRuntimeRegionEnabled() bool {
    if (platform.env.getenvBoolDefault("TERMITE_METAL_DISABLE_FFN_GELU_BACKWARD_RUNTIME_REGION", false)) return false;
    if (platform.env.getenvBoolDefault("TERMITE_METAL_ENABLE_FFN_GELU_BACKWARD_RUNTIME_REGION", false)) return true;
    return platform.env.getenvBoolDefault("TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR", false) and
        !platform.env.getenvBoolDefault("TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR", false);
}

fn rank1DotSpecializationEnabled() bool {
    if (platform.env.getenvBoolDefault("TERMITE_METAL_DISABLE_RANK1_DOT_SPECIALIZATION", false)) return false;
    if (platform.env.getenvBoolDefault("TERMITE_METAL_ENABLE_RANK1_DOT_SPECIALIZATION", false)) return true;
    return platform.env.getenvBoolDefault("TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR", false) and
        !platform.env.getenvBoolDefault("TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR", false);
}

fn rawLinearBiasPairRuntimeRegionEnabled() bool {
    if (platform.env.getenvBoolDefault("TERMITE_METAL_DISABLE_RAW_LINEAR_BIAS_PAIR_RUNTIME_REGION", false)) return false;
    return platform.env.getenvBoolDefault("TERMITE_METAL_ENABLE_RAW_LINEAR_BIAS_PAIR_RUNTIME_REGION", true);
}

fn gatedFfnGraphFusionDisabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_DISABLE_GATED_FFN_GRAPH_FUSION", false);
}

fn gatedFfnGraphFusionEnabled() bool {
    if (gatedFfnGraphFusionDisabled()) return false;
    return platform.env.getenvBoolDefault("TERMITE_METAL_ENABLE_GATED_FFN_GRAPH_FUSION", true);
}

fn attentionOutputResidualGraphFusionDisabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_METAL_DISABLE_ATTENTION_OUTPUT_RESIDUAL_GRAPH_FUSION", false);
}

fn attentionOutputResidualGraphFusionEnabled() bool {
    if (attentionOutputResidualGraphFusionDisabled()) return false;
    return platform.env.getenvBoolDefault("TERMITE_METAL_ENABLE_ATTENTION_OUTPUT_RESIDUAL_GRAPH_FUSION", true);
}

fn buildRuntimeRegionPlan(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    node_ids: []const NodeId,
    value_count: usize,
    reachable: []const bool,
    last_use: []const u32,
) !RuntimeRegionPlan {
    const regions = try allocator.alloc(RuntimeRegion, node_ids.len);
    errdefer allocator.free(regions);
    @memset(regions, .{ .none = {} });
    const prepared = try allocator.alloc(PreparedRuntimeRegion, node_ids.len);
    errdefer allocator.free(prepared);
    @memset(prepared, .{ .none = {} });
    const attention_input_max_first_node = try allocator.alloc(NodeId, value_count);
    errdefer allocator.free(attention_input_max_first_node);
    @memset(attention_input_max_first_node, null_node);
    const null_values = try allocator.alloc(?CT, value_count);
    defer allocator.free(null_values);
    @memset(null_values, null);

    const skipped = try allocator.alloc(bool, value_count);
    defer allocator.free(skipped);
    @memset(skipped, false);

    var region_count: usize = 0;
    for (node_ids, 0..) |node_id, node_pos| {
        const i: usize = @intCast(node_id);
        if (i >= reachable.len or !reachable[i]) continue;
        if (i < skipped.len and skipped[i]) continue;

        if (rawLinearBiasPairRuntimeRegionEnabled()) {
            if (matchRawLinearBiasPairPattern(graph, node_ids, node_pos, reachable, last_use, skipped)) |pattern| {
                regions[node_pos] = .{ .raw_linear_bias_pair = pattern };
                markRawLinearBiasPairSkipped(skipped, pattern);
                region_count += 1;
                continue;
            }
        }
        if (matchRawLinearBiasPattern(graph, node_ids, node_pos, reachable, last_use, skipped)) |pattern| {
            regions[node_pos] = .{ .raw_linear_bias = pattern };
            markRawLinearBiasSkipped(skipped, pattern);
            region_count += 1;
            continue;
        }
        if (matchRawLinearPairPattern(graph, node_ids, node_pos, reachable, last_use, skipped)) |pattern| {
            regions[node_pos] = .{ .raw_linear_pair = pattern };
            markRawLinearPairSkipped(skipped, pattern);
            region_count += 1;
            continue;
        }
        if (matchRawLinearDotPattern(graph, node_ids, node_pos, reachable, last_use)) |pattern| {
            regions[node_pos] = .{ .raw_linear_dot = pattern };
            region_count += 1;
            continue;
        }
        if (matchLoraLinearQkvPattern(graph, node_ids, node_pos, reachable, skipped)) |pattern| {
            regions[node_pos] = .{ .lora_linear_qkv = pattern };
            markLoraLinearQkvSkipped(skipped, pattern);
            region_count += 1;
            continue;
        }
        if (matchLoraLinearPattern(graph, node_ids, node_pos, reachable, skipped)) |pattern| {
            regions[node_pos] = .{ .lora_linear = pattern };
            markLoraLinearSkipped(skipped, pattern);
            region_count += 1;
            continue;
        }
        if (debertaAttentionRuntimeRegionEnabled()) {
            if (matchDebertaAttentionPattern(graph, node_ids, node_pos, reachable, skipped, regions)) |pattern| {
                regions[node_pos] = .{ .deberta_attention = pattern };
                markDebertaAttentionSkipped(graph, skipped, pattern);
                recordAttentionInputMaxFirstNode(attention_input_max_first_node, pattern);
                region_count += 1;
                continue;
            }
        }
        if (loraBackwardRuntimeRegionEnabled()) {
            if (matchLoraBackwardPattern(graph, node_ids, node_pos, reachable, skipped)) |pattern| {
                regions[node_pos] = .{ .lora_backward = pattern };
                markLoraBackwardSkipped(skipped, pattern);
                region_count += 1;
                continue;
            }
        }
        if (ffnGeluBackwardRuntimeRegionEnabled()) {
            if (matchFfnGeluBackwardPattern(graph, node_ids, node_pos, reachable, skipped)) |pattern| {
                regions[node_pos] = .{ .ffn_gelu_backward = pattern };
                markFfnGeluBackwardSkipped(skipped, pattern);
                region_count += 1;
                continue;
            }
        }

        if (matchRmsNormGroupedLinearQkvSlicePattern(graph, null_values, node_ids, node_pos, reachable, skipped)) |pattern| {
            regions[node_pos] = .{ .rms_norm_grouped_linear_qkv_slice = pattern };
            markRmsNormGroupedLinearQkvSkipped(skipped, pattern);
            region_count += 1;
            continue;
        }
        if (matchGroupedLinearQkvSlicePattern(graph, null_values, node_ids, node_pos, reachable, skipped)) |pattern| {
            regions[node_pos] = .{ .grouped_linear_qkv_slice = pattern };
            markGroupedLinearQkvSkipped(skipped, pattern);
            region_count += 1;
            continue;
        }
        if (matchLinearNoBiasQkvPattern(graph, null_values, node_ids, node_pos, reachable, skipped)) |pattern| {
            regions[node_pos] = .{ .linear_qkv = pattern };
            markLinearNoBiasQkvSkipped(skipped, pattern);
            region_count += 1;
            continue;
        }
        if (matchQLinearPattern(graph, null_values, node_ids, node_pos, reachable, skipped)) |pattern| {
            regions[node_pos] = .{ .q_linear = pattern };
            region_count += 1;
            continue;
        }

        if (attentionOutputResidualGraphFusionEnabled()) {
            if (matchAttentionOutputResidualPattern(graph, node_ids, node_pos, reachable, skipped, last_use)) |pattern| {
                regions[node_pos] = .{ .attention_output_residual = pattern };
                markAttentionOutputResidualSkipped(skipped, pattern);
                region_count += 1;
                continue;
            }
        }

        if (gatedFfnGraphFusionEnabled()) {
            if (matchRmsNormGatedFfnResidualPattern(graph, node_ids, node_pos, reachable, skipped, last_use)) |pattern| {
                regions[node_pos] = .{ .rms_norm_gated_ffn_residual = pattern };
                markRmsNormGatedFfnResidualSkipped(skipped, pattern);
                region_count += 1;
                continue;
            }
            if (matchGatedFfnResidualPattern(graph, node_ids, node_pos, reachable, skipped, last_use)) |pattern| {
                regions[node_pos] = .{ .gated_ffn_residual = pattern };
                markGatedFfnResidualSkipped(skipped, pattern);
                region_count += 1;
                continue;
            }
        }

        if (matchPleResidualPattern(graph, node_ids, node_pos, reachable, skipped)) |pattern| {
            regions[node_pos] = .{ .ple_residual = pattern };
            markPleResidualSkipped(skipped, pattern);
            region_count += 1;
            continue;
        }
    }

    return .{
        .node_count = node_ids.len,
        .value_count = value_count,
        .first_node = if (node_ids.len == 0) null_node else node_ids[0],
        .last_node = if (node_ids.len == 0) null_node else node_ids[node_ids.len - 1],
        .regions_by_pos = regions,
        .prepared_by_pos = prepared,
        .attention_input_max_first_node = attention_input_max_first_node,
        .region_count = region_count,
        .covered_node_count = runtimeRegionCoveredNodeCount(node_ids, regions, skipped),
        .elided_node_count = runtimeRegionElidedNodeCount(node_ids, regions, skipped),
    };
}

fn recordAttentionInputMaxFirstNode(max_first_by_input: []NodeId, pattern: DebertaAttentionPattern) void {
    recordAttentionInputFirstNode(max_first_by_input, pattern.q_id, pattern.first_node_id);
    recordAttentionInputFirstNode(max_first_by_input, pattern.k_id, pattern.first_node_id);
    recordAttentionInputFirstNode(max_first_by_input, pattern.v_id, pattern.first_node_id);
    recordAttentionInputFirstNode(max_first_by_input, pattern.q_r_id, pattern.first_node_id);
    recordAttentionInputFirstNode(max_first_by_input, pattern.k_r_id, pattern.first_node_id);
    recordAttentionInputFirstNode(max_first_by_input, pattern.attn_bias_id, pattern.first_node_id);
}

fn recordAttentionInputFirstNode(max_first_by_input: []NodeId, input_id: NodeId, first_node_id: NodeId) void {
    if (input_id == null_node) return;
    const input_index: usize = @intCast(input_id);
    if (input_index >= max_first_by_input.len) return;
    if (max_first_by_input[input_index] == null_node or first_node_id > max_first_by_input[input_index]) {
        max_first_by_input[input_index] = first_node_id;
    }
}

fn runtimeRegionCoveredNodeCount(node_ids: []const NodeId, regions: []const RuntimeRegion, skipped: []const bool) usize {
    var count: usize = 0;
    for (node_ids, 0..) |node_id, pos| {
        if (pos < regions.len and std.meta.activeTag(regions[pos]) != .none) {
            count += 1;
            continue;
        }
        const i: usize = @intCast(node_id);
        if (i < skipped.len and skipped[i]) count += 1;
    }
    return count;
}

fn runtimeRegionElidedNodeCount(node_ids: []const NodeId, regions: []const RuntimeRegion, skipped: []const bool) usize {
    var count: usize = 0;
    for (node_ids, 0..) |node_id, pos| {
        if (pos < regions.len and std.meta.activeTag(regions[pos]) != .none) continue;
        const i: usize = @intCast(node_id);
        if (i < skipped.len and skipped[i]) count += 1;
    }
    return count;
}

const RuntimeRegionSummary = struct {
    kind: RuntimeRegionKind = .none,
    count: usize = 0,
    first_pos: usize = 0,
    last_pos: usize = 0,
};

fn recordRuntimeRegionSummary(summaries: *[32]RuntimeRegionSummary, used: *usize, kind: RuntimeRegionKind, pos: usize) void {
    if (kind == .none) return;
    for (summaries[0..used.*]) |*summary| {
        if (summary.kind != kind) continue;
        summary.count += 1;
        summary.last_pos = pos;
        return;
    }
    if (used.* >= summaries.len) return;
    summaries[used.*] = .{
        .kind = kind,
        .count = 1,
        .first_pos = pos,
        .last_pos = pos,
    };
    used.* += 1;
}

fn printRuntimeRegionPlanSummary(graph: *const Graph, plan: RuntimeRegionPlan, partition_index: usize) void {
    var summaries = [_]RuntimeRegionSummary{.{}} ** 32;
    var used: usize = 0;
    for (plan.regions_by_pos, 0..) |region, pos| {
        recordRuntimeRegionSummary(&summaries, &used, std.meta.activeTag(region), pos);
    }
    const eligibility = analyzeRuntimeFrameEligibility(plan);
    const metadata_ready = runtimeFrameMetadataFromPlan(graph, plan) != null;
    std.debug.print(
        "runtime_region_plan_summary: partition={d} regions={d} frame_reason={s} frame_layers={d} metadata_ready={} kinds=",
        .{ partition_index, plan.region_count, @tagName(eligibility.reason), eligibility.layers, metadata_ready },
    );
    if (used == 0) {
        std.debug.print("none", .{});
    } else {
        for (summaries[0..used], 0..) |summary, idx| {
            if (idx > 0) std.debug.print(",", .{});
            std.debug.print("{s}:{d}@{d}-{d}", .{
                @tagName(summary.kind),
                summary.count,
                summary.first_pos,
                summary.last_pos,
            });
        }
    }
    std.debug.print("\n", .{});
}

fn runtimeFrameLayerShapeFromQkv(region: RuntimeRegion) ?RuntimeFrameLayerShape {
    return switch (region) {
        .q_linear => |pattern| .{ .rows = pattern.rows, .hidden_size = pattern.in_dim, .attention_input_size = pattern.out_dim },
        .linear_qkv => |pattern| .{ .rows = pattern.rows, .hidden_size = pattern.in_dim, .attention_input_size = pattern.q_out_dim },
        .grouped_linear_qkv_slice => |pattern| .{ .rows = pattern.rows, .hidden_size = pattern.in_dim, .attention_input_size = pattern.q_out_dim },
        .rms_norm_grouped_linear_qkv_slice => |pattern| .{ .rows = pattern.qkv.rows, .hidden_size = pattern.qkv.in_dim, .attention_input_size = pattern.qkv.q_out_dim },
        .lora_linear_qkv => |pattern| .{ .rows = pattern.rows, .hidden_size = pattern.in_dim, .attention_input_size = pattern.q_out_dim },
        else => null,
    };
}

fn runtimeFrameLayerShapeFromAttention(pattern: AttentionOutputResidualPattern) RuntimeFrameLayerShape {
    return .{ .rows = pattern.rows, .hidden_size = pattern.hidden_size, .attention_input_size = pattern.attention_input_size };
}

fn runtimeFrameLayerShapeFromFfn(region: RuntimeRegion) ?RuntimeFrameLayerShape {
    return switch (region) {
        .rms_norm_gated_ffn_residual => |pattern| .{ .rows = pattern.ffn.rows, .hidden_size = pattern.ffn.hidden_size },
        .gated_ffn_residual => |pattern| .{ .rows = pattern.rows, .hidden_size = pattern.hidden_size },
        else => null,
    };
}

fn runtimeFrameLayerShapeFromPle(pattern: PleResidualPattern) RuntimeFrameLayerShape {
    return .{ .rows = pattern.rows, .hidden_size = pattern.hidden_size };
}

fn runtimeFrameShapesMatch(lhs: RuntimeFrameLayerShape, rhs: RuntimeFrameLayerShape) bool {
    if (lhs.rows != rhs.rows or lhs.hidden_size != rhs.hidden_size) return false;
    if (lhs.attention_input_size != 0 and rhs.attention_input_size != 0 and lhs.attention_input_size != rhs.attention_input_size) return false;
    return true;
}

fn runtimeFrameQkvMetadataFromRegion(graph: *const Graph, region: RuntimeRegion) ?RuntimeFrameQkvMetadata {
    return switch (region) {
        .q_linear => |pattern| .{
            .layer_index = layerIndexForWeight(graph, pattern.weight_id) orelse return null,
            .rows = pattern.rows,
            .hidden_size = pattern.in_dim,
            .q_dim = pattern.out_dim,
            .kv_dim = 0,
            .q_weight_id = pattern.weight_id,
        },
        .linear_qkv => |pattern| .{
            .layer_index = layerIndexForWeight(graph, pattern.q_weight_id) orelse return null,
            .rows = pattern.rows,
            .hidden_size = pattern.in_dim,
            .q_dim = pattern.q_out_dim,
            .kv_dim = pattern.kv_out_dim,
            .q_weight_id = pattern.q_weight_id,
            .k_weight_id = pattern.k_weight_id,
            .v_weight_id = pattern.v_weight_id,
        },
        .grouped_linear_qkv_slice => |pattern| .{
            .layer_index = layerIndexForWeight(graph, pattern.q_weight_id) orelse return null,
            .rows = pattern.rows,
            .hidden_size = pattern.in_dim,
            .q_dim = pattern.q_out_dim,
            .kv_dim = pattern.kv_out_dim,
            .q_weight_id = pattern.q_weight_id,
            .k_weight_id = pattern.k_weight_id,
            .v_weight_id = pattern.v_weight_id,
        },
        .rms_norm_grouped_linear_qkv_slice => |pattern| .{
            .layer_index = layerIndexForWeight(graph, pattern.qkv.q_weight_id) orelse return null,
            .rows = pattern.qkv.rows,
            .hidden_size = pattern.qkv.in_dim,
            .q_dim = pattern.qkv.q_out_dim,
            .kv_dim = pattern.qkv.kv_out_dim,
            .q_weight_id = pattern.qkv.q_weight_id,
            .k_weight_id = pattern.qkv.k_weight_id,
            .v_weight_id = pattern.qkv.v_weight_id,
        },
        .lora_linear_qkv => |pattern| .{
            .layer_index = layerIndexForWeight(graph, pattern.q_base_weight_id) orelse return null,
            .rows = pattern.rows,
            .hidden_size = pattern.in_dim,
            .q_dim = pattern.q_out_dim,
            .kv_dim = pattern.kv_out_dim,
            .q_weight_id = pattern.q_base_weight_id,
            .k_weight_id = pattern.k_base_weight_id,
            .v_weight_id = pattern.v_base_weight_id,
        },
        else => null,
    };
}

fn runtimeFrameLayerMetadata(
    graph: *const Graph,
    qkv: RuntimeFrameQkvMetadata,
    attention: AttentionOutputResidualPattern,
    ffn_region: RuntimeRegion,
    ple: PleResidualPattern,
) ?RuntimeFrameLayerMetadata {
    const attention_node = graph.node(attention.attention_id);
    const attention_attrs = switch (attention_node.op) {
        .fused_gqa_causal_attention => |attrs| attrs,
        else => return null,
    };
    const num_heads: usize = attention_attrs.num_heads;
    const num_kv_heads: usize = if (attention_attrs.num_kv_heads == 0) attention_attrs.num_heads else attention_attrs.num_kv_heads;
    const head_dim: usize = attention_attrs.head_dim;
    if (num_heads == 0 or num_kv_heads == 0 or head_dim == 0) return null;
    const attention_input_size = num_heads * head_dim;
    if (attention_input_size != qkv.q_dim or attention_input_size != attention.attention_input_size) return null;
    if (attention.hidden_size != qkv.hidden_size or ple.hidden_size != qkv.hidden_size) return null;
    if (attention.rows != qkv.rows or ple.rows != qkv.rows) return null;

    const ffn_shape = runtimeFrameLayerShapeFromFfn(ffn_region) orelse return null;
    if (ffn_shape.rows != qkv.rows or ffn_shape.hidden_size != qkv.hidden_size) return null;
    const ffn = switch (ffn_region) {
        .rms_norm_gated_ffn_residual => |pattern| pattern.ffn,
        .gated_ffn_residual => |pattern| pattern,
        else => return null,
    };
    if (ffn.activation != ple.activation) return null;

    const shares_kv = attention_attrs.skip_kv_write;
    const kv_dim = num_kv_heads * head_dim;
    if (!shares_kv and qkv.kv_dim != kv_dim) return null;
    if (shares_kv and qkv.kv_dim != 0) return null;
    const kv_layer_index: usize = if (attention_attrs.layer_index == std.math.maxInt(u32))
        qkv.layer_index
    else
        attention_attrs.layer_index;

    return .{
        .layer_index = qkv.layer_index,
        .shares_kv = shares_kv,
        .kv_layer_index = kv_layer_index,
        .kv_heads = num_kv_heads,
        .head_dim = head_dim,
        .intermediate_size = ffn.intermediate_size,
        .hidden_size = qkv.hidden_size,
        .attention_input_size = attention_input_size,
        .ple_hidden_size = ple.ple_hidden_size,
        .activation = ffn.activation,
    };
}

fn traceRuntimeFrameMetadataDeclined(
    reason: []const u8,
    layer_index: usize,
    qkv: ?RuntimeFrameQkvMetadata,
    attention: ?AttentionOutputResidualPattern,
    ffn: RuntimeRegion,
    ple: ?PleResidualPattern,
) void {
    if (!platform.env.getenvBoolDefault("TERMITE_METAL_TRACE_FRAME_METADATA", false)) return;
    const q_layer = if (qkv) |q| q.layer_index else std.math.maxInt(usize);
    const q_rows = if (qkv) |q| q.rows else 0;
    const q_hidden = if (qkv) |q| q.hidden_size else 0;
    const q_dim = if (qkv) |q| q.q_dim else 0;
    const attn_id = if (attention) |a| a.attention_id else null_node;
    const attn_rows = if (attention) |a| a.rows else 0;
    const attn_hidden = if (attention) |a| a.hidden_size else 0;
    const attn_dim = if (attention) |a| a.attention_input_size else 0;
    const ffn_shape = runtimeFrameLayerShapeFromFfn(ffn);
    const ffn_rows = if (ffn_shape) |shape| shape.rows else 0;
    const ffn_hidden = if (ffn_shape) |shape| shape.hidden_size else 0;
    const ple_rows = if (ple) |p| p.rows else 0;
    const ple_hidden = if (ple) |p| p.hidden_size else 0;
    const ple_dim = if (ple) |p| p.ple_hidden_size else 0;
    std.debug.print(
        "runtime_frame_metadata_declined reason={s} layer_count={d} q_layer={d} q_rows={d} q_hidden={d} q_dim={d} attn={d} attn_rows={d} attn_hidden={d} attn_dim={d} ffn_rows={d} ffn_hidden={d} ple_rows={d} ple_hidden={d} ple_dim={d}\n",
        .{ reason, layer_index, q_layer, q_rows, q_hidden, q_dim, attn_id, attn_rows, attn_hidden, attn_dim, ffn_rows, ffn_hidden, ple_rows, ple_hidden, ple_dim },
    );
}

fn runtimeFrameMetadataFromPlan(graph: *const Graph, plan: RuntimeRegionPlan) ?RuntimeFrameMetadata {
    if (plan.region_count == 0) return null;

    var phase: enum { qkv, attention, ffn, ple } = .qkv;
    var pending_qkv: ?RuntimeFrameQkvMetadata = null;
    var pending_attention: ?AttentionOutputResidualPattern = null;
    var pending_ffn: RuntimeRegion = .{ .none = {} };
    var rows: usize = 0;
    var hidden_size: usize = 0;
    var num_attention_heads: usize = 0;
    var global_head_dim: usize = 0;
    var ple_hidden_size: usize = 0;
    var activation: ?ops_mod.DecoderRuntimeActivationKind = null;
    var layer_count: usize = 0;

    for (plan.regions_by_pos) |region| {
        switch (region) {
            .none, .raw_linear_dot, .raw_linear_pair, .raw_linear_bias, .raw_linear_bias_pair, .lora_linear, .lora_backward, .ffn_gelu_backward => continue,
            .deberta_attention => {
                if (phase != .attention or pending_qkv == null) {
                    traceRuntimeFrameMetadataDeclined("attention_phase", layer_count, pending_qkv, pending_attention, pending_ffn, null);
                    return null;
                }
                phase = .ffn;
                pending_attention = null;
            },
            .q_linear, .linear_qkv, .grouped_linear_qkv_slice, .rms_norm_grouped_linear_qkv_slice, .lora_linear_qkv => {
                if (phase != .qkv) {
                    traceRuntimeFrameMetadataDeclined("qkv_phase", layer_count, pending_qkv, pending_attention, pending_ffn, null);
                    return null;
                }
                pending_qkv = runtimeFrameQkvMetadataFromRegion(graph, region) orelse {
                    traceRuntimeFrameMetadataDeclined("qkv_metadata", layer_count, null, pending_attention, pending_ffn, null);
                    return null;
                };
                phase = .attention;
            },
            .attention_output_residual => |pattern| {
                if (phase != .attention or pending_qkv == null) {
                    traceRuntimeFrameMetadataDeclined("attention_phase", layer_count, pending_qkv, pending_attention, pending_ffn, null);
                    return null;
                }
                pending_attention = pattern;
                phase = .ffn;
            },
            .rms_norm_gated_ffn_residual, .gated_ffn_residual => {
                if (phase != .ffn or pending_attention == null) {
                    traceRuntimeFrameMetadataDeclined("ffn_phase", layer_count, pending_qkv, pending_attention, pending_ffn, null);
                    return null;
                }
                pending_ffn = region;
                phase = .ple;
            },
            .ple_residual => |pattern| {
                if (phase != .ple) {
                    traceRuntimeFrameMetadataDeclined("ple_phase", layer_count, pending_qkv, pending_attention, pending_ffn, pattern);
                    return null;
                }
                const layer = runtimeFrameLayerMetadata(
                    graph,
                    pending_qkv orelse return null,
                    pending_attention orelse return null,
                    pending_ffn,
                    pattern,
                ) orelse {
                    traceRuntimeFrameMetadataDeclined("layer_metadata", layer_count, pending_qkv, pending_attention, pending_ffn, pattern);
                    return null;
                };
                if (layer.layer_index != layer_count) {
                    traceRuntimeFrameMetadataDeclined("layer_index_order", layer_count, pending_qkv, pending_attention, pending_ffn, pattern);
                    return null;
                }
                if (rows == 0) {
                    rows = pattern.rows;
                    hidden_size = layer.hidden_size;
                    ple_hidden_size = layer.ple_hidden_size;
                    activation = layer.activation;
                    num_attention_heads = if (layer.head_dim == 0) return null else layer.attention_input_size / layer.head_dim;
                    global_head_dim = if (layer.shares_kv) layer.head_dim else 0;
                } else {
                    if (rows != pattern.rows or hidden_size != layer.hidden_size) {
                        traceRuntimeFrameMetadataDeclined("frame_shape_mismatch", layer_count, pending_qkv, pending_attention, pending_ffn, pattern);
                        return null;
                    }
                    if (ple_hidden_size != layer.ple_hidden_size) {
                        traceRuntimeFrameMetadataDeclined("ple_size_mismatch", layer_count, pending_qkv, pending_attention, pending_ffn, pattern);
                        return null;
                    }
                    if (activation.? != layer.activation) {
                        traceRuntimeFrameMetadataDeclined("activation_mismatch", layer_count, pending_qkv, pending_attention, pending_ffn, pattern);
                        return null;
                    }
                    const heads = if (layer.head_dim == 0) {
                        traceRuntimeFrameMetadataDeclined("zero_head_dim", layer_count, pending_qkv, pending_attention, pending_ffn, pattern);
                        return null;
                    } else layer.attention_input_size / layer.head_dim;
                    if (num_attention_heads != heads) {
                        traceRuntimeFrameMetadataDeclined("num_heads_mismatch", layer_count, pending_qkv, pending_attention, pending_ffn, pattern);
                        return null;
                    }
                    if (layer.shares_kv) {
                        if (global_head_dim < layer.head_dim) {
                            global_head_dim = layer.head_dim;
                        }
                    }
                }
                layer_count += 1;
                pending_qkv = null;
                pending_attention = null;
                pending_ffn = .{ .none = {} };
                phase = .qkv;
            },
        }
    }

    if (phase != .qkv or layer_count == 0) {
        traceRuntimeFrameMetadataDeclined("final_phase", layer_count, pending_qkv, pending_attention, pending_ffn, null);
        return null;
    }
    return .{
        .rows = rows,
        .layer_count = layer_count,
        .hidden_size = hidden_size,
        .num_attention_heads = num_attention_heads,
        .global_head_dim = global_head_dim,
        .ple_hidden_size = ple_hidden_size,
        .activation = activation orelse return null,
    };
}

fn analyzeRuntimeFrameEligibility(plan: RuntimeRegionPlan) RuntimeFrameEligibility {
    if (plan.region_count == 0) return .{ .reason = .no_regions };

    var phase: enum { qkv, attention, ffn, ple } = .qkv;
    var layer_shape: ?RuntimeFrameLayerShape = null;
    var frame_rows: usize = 0;
    var layers: usize = 0;

    for (plan.regions_by_pos) |region| {
        switch (region) {
            .none, .raw_linear_dot, .raw_linear_pair, .raw_linear_bias, .raw_linear_bias_pair, .lora_linear, .lora_backward, .ffn_gelu_backward => continue,
            .deberta_attention => {
                if (phase != .attention) return .{ .layers = layers, .reason = .non_layer_order };
                phase = .ffn;
            },
            .q_linear, .linear_qkv, .grouped_linear_qkv_slice, .rms_norm_grouped_linear_qkv_slice, .lora_linear_qkv => {
                if (phase != .qkv) return .{ .layers = layers, .reason = .non_layer_order };
                layer_shape = runtimeFrameLayerShapeFromQkv(region) orelse return .{ .layers = layers, .reason = .non_layer_order };
                phase = .attention;
            },
            .attention_output_residual => |pattern| {
                if (phase == .qkv) return .{ .layers = layers, .reason = .missing_qkv };
                if (phase != .attention) return .{ .layers = layers, .reason = .non_layer_order };
                const attention_shape = runtimeFrameLayerShapeFromAttention(pattern);
                if (layer_shape) |shape| {
                    if (!runtimeFrameShapesMatch(shape, attention_shape)) return .{ .layers = layers, .reason = .shape_mismatch };
                } else {
                    return .{ .layers = layers, .reason = .missing_qkv };
                }
                layer_shape = attention_shape;
                phase = .ffn;
            },
            .rms_norm_gated_ffn_residual, .gated_ffn_residual => {
                if (phase == .qkv) return .{ .layers = layers, .reason = .missing_qkv };
                if (phase == .attention) return .{ .layers = layers, .reason = .missing_attention };
                if (phase == .ple) return .{ .layers = layers, .reason = .non_layer_order };
                const ffn_shape = runtimeFrameLayerShapeFromFfn(region) orelse return .{ .layers = layers, .reason = .non_layer_order };
                if (layer_shape) |shape| {
                    if (!runtimeFrameShapesMatch(shape, ffn_shape)) return .{ .layers = layers, .reason = .shape_mismatch };
                } else {
                    return .{ .layers = layers, .reason = .missing_attention };
                }
                phase = .ple;
            },
            .ple_residual => |pattern| {
                if (phase == .qkv) return .{ .layers = layers, .reason = .missing_qkv };
                if (phase == .attention) return .{ .layers = layers, .reason = .missing_attention };
                if (phase == .ffn) return .{ .layers = layers, .reason = .missing_ffn };
                const ple_shape = runtimeFrameLayerShapeFromPle(pattern);
                if (layer_shape) |shape| {
                    if (!runtimeFrameShapesMatch(shape, ple_shape)) return .{ .layers = layers, .reason = .shape_mismatch };
                } else {
                    return .{ .layers = layers, .reason = .missing_attention };
                }
                if (frame_rows == 0) frame_rows = ple_shape.rows;
                layers += 1;
                layer_shape = null;
                phase = .qkv;
            },
        }
    }

    if (layers == 0 and phase == .qkv) return .{ .reason = .no_regions };
    if (phase == .attention) return .{ .layers = layers, .reason = .missing_attention };
    if (phase == .ffn) return .{ .layers = layers, .reason = .missing_ffn };
    if (phase == .ple) return .{ .layers = layers, .reason = .missing_ple };
    if (frame_rows <= 1) return .{ .layers = layers, .reason = .single_row };

    // The current graph plan proves layer structure, but not yet the model
    // metadata needed by the whole-frame decoder runtime: head layout, RoPE/KV
    // policy, PLE vectors, and stable per-layer slot numbering.
    return .{ .layers = layers, .reason = .missing_model_metadata };
}

fn recordRuntimeFrameEligibilityStats(stats: *PartitionExecutor.ExecutionStats, eligibility: RuntimeFrameEligibility) void {
    if (eligibility.reason != .no_regions) stats.runtime_frame_candidates += 1;
    if (eligibility.eligible()) {
        stats.runtime_frame_eligible += 1;
        return;
    }
    switch (eligibility.reason) {
        .none => {},
        .no_regions => stats.runtime_frame_ineligible_no_regions += 1,
        .missing_qkv => stats.runtime_frame_ineligible_missing_qkv += 1,
        .missing_attention => stats.runtime_frame_ineligible_missing_attention += 1,
        .missing_ffn => stats.runtime_frame_ineligible_missing_ffn += 1,
        .missing_ple => stats.runtime_frame_ineligible_missing_ple += 1,
        .single_row => stats.runtime_frame_ineligible_single_row += 1,
        .non_layer_order => stats.runtime_frame_ineligible_non_layer_order += 1,
        .shape_mismatch => stats.runtime_frame_ineligible_shape_mismatch += 1,
        .missing_model_metadata => stats.runtime_frame_ineligible_missing_model_metadata += 1,
    }
}

fn markSkipped(skipped: []bool, node_id: NodeId) void {
    const i: usize = @intCast(node_id);
    if (i < skipped.len) skipped[i] = true;
}

fn markLinearNoBiasQkvSkipped(skipped: []bool, pattern: LinearNoBiasQkvPattern) void {
    markSkipped(skipped, pattern.k_id);
    markSkipped(skipped, pattern.v_id);
}

fn markLoraLinearSkipped(skipped: []bool, pattern: LoraLinearPattern) void {
    if (pattern.dropout_mul_id) |dropout_mul_id| markSkipped(skipped, dropout_mul_id);
    markSkipped(skipped, pattern.after_a_id);
    markSkipped(skipped, pattern.after_b_id);
    markSkipped(skipped, pattern.scaled_id);
}

fn markLoraLinearQkvSkipped(skipped: []bool, pattern: LoraLinearQkvPattern) void {
    markLoraLinearSkipped(skipped, pattern.q);
    markLoraLinearSkipped(skipped, pattern.k);
    markLoraLinearSkipped(skipped, pattern.v);
    markSkipped(skipped, pattern.k.add_id);
    markSkipped(skipped, pattern.v.add_id);
}

fn markDebertaAttentionSkipped(graph: *const Graph, skipped: []bool, pattern: DebertaAttentionPattern) void {
    var node_id = pattern.first_node_id;
    while (node_id <= pattern.output_id) : (node_id += 1) {
        if (debertaAttentionNodeFeedsOutsideConsumer(
            graph,
            node_id,
            pattern.first_node_id,
            pattern.output_id,
            pattern.output_id,
            @intCast(pattern.output_id - pattern.first_node_id + 1),
        )) continue;
        markSkipped(skipped, node_id);
        if (node_id == std.math.maxInt(NodeId)) break;
    }
}

fn debertaAttentionNodeFeedsOutsideConsumer(
    graph: *const Graph,
    node_id: NodeId,
    first_id: NodeId,
    last_id: NodeId,
    output_id: NodeId,
    depth_remaining: usize,
) bool {
    if (node_id == output_id) return false;
    if (depth_remaining == 0) return false;

    var candidate_id: NodeId = 0;
    while (candidate_id < graph.nodeCount()) : (candidate_id += 1) {
        const candidate = graph.node(candidate_id);
        for (candidate.getInputs()) |input_id| {
            if (input_id != node_id) continue;
            if (candidate_id < first_id or candidate_id > last_id) return true;
            if (candidate_id == output_id) continue;
            if (debertaAttentionNodeFeedsOutsideConsumer(
                graph,
                candidate_id,
                first_id,
                last_id,
                output_id,
                depth_remaining - 1,
            )) return true;
        }
    }
    return false;
}

fn markGroupedLinearQkvSkipped(skipped: []bool, pattern: GroupedLinearQkvSlicePattern) void {
    markSkipped(skipped, pattern.linear_id);
    markSkipped(skipped, pattern.q_slice_id);
    markSkipped(skipped, pattern.k_slice_id);
    markSkipped(skipped, pattern.v_slice_id);
}

fn markRmsNormGroupedLinearQkvSkipped(skipped: []bool, pattern: RmsNormGroupedLinearQkvSlicePattern) void {
    markSkipped(skipped, pattern.norm_id);
    markGroupedLinearQkvSkipped(skipped, pattern.qkv);
}

fn markAttentionOutputResidualSkipped(skipped: []bool, pattern: AttentionOutputResidualPattern) void {
    if (pattern.pre_linear_norm_id) |norm_id| markSkipped(skipped, norm_id);
    markSkipped(skipped, pattern.linear_id);
    if (pattern.post_linear_norm_id) |norm_id| markSkipped(skipped, norm_id);
    markSkipped(skipped, pattern.add_id);
}

fn markRmsNormGatedFfnResidualSkipped(skipped: []bool, pattern: RmsNormGatedFfnResidualPattern) void {
    markGatedFfnResidualSkipped(skipped, pattern.ffn);
}

fn markGatedFfnResidualSkipped(skipped: []bool, pattern: GatedFfnResidualPattern) void {
    markSkipped(skipped, pattern.pair_second_id);
    markSkipped(skipped, pattern.activation_id);
    markSkipped(skipped, pattern.multiply_id);
    markSkipped(skipped, pattern.down_id);
    if (pattern.post_down_norm_id) |norm_id| markSkipped(skipped, norm_id);
    markSkipped(skipped, pattern.add_id);
}

fn markPleResidualSkipped(skipped: []bool, pattern: PleResidualPattern) void {
    markSkipped(skipped, pattern.activation_id);
    markSkipped(skipped, pattern.multiply_id);
    markSkipped(skipped, pattern.projection_id);
    markSkipped(skipped, pattern.post_norm_id);
    markSkipped(skipped, pattern.add_id);
}

fn printMetalGraphPlanTrace(partition_index: u32, plan: MetalPartitionGraphPlan) void {
    var total_bytes: usize = 0;
    for (plan.slots) |slot| total_bytes += slot.bytes;
    std.debug.print(
        "graph_executor_node_trace: graph_plan partition={d} slots={d} bytes={d}",
        .{ partition_index, plan.slots.len, total_bytes },
    );
    for (plan.slots) |slot| {
        std.debug.print(" slot{d}={d}", .{ slot.slot, slot.bytes });
    }
    std.debug.print("\n", .{});
}

fn printMetalNodeTraceBegin(graph: *const Graph, node_id: NodeId) void {
    const n = graph.node(node_id);
    switch (n.op) {
        .parameter => {
            std.debug.print(
                "graph_executor_node_trace: begin node={d} op=parameter name={s} shape={any}\n",
                .{ node_id, graph.parameterName(n), n.output_shape },
            );
        },
        else => {
            std.debug.print(
                "graph_executor_node_trace: begin node={d} op={s} shape={any}\n",
                .{ node_id, @tagName(n.op), n.output_shape },
            );
        },
    }
}

fn printMetalNodeTraceInputs(graph: *const Graph, cb: *const ComputeBackend, values: []?CT, node_id: NodeId) void {
    const n = graph.node(node_id);
    for (n.getInputs(), 0..) |input_id, input_index| {
        const input_node = graph.node(input_id);
        const value = valueFor(values, input_id);
        const device = if (value) |ct| isMetalDeviceResident(cb, ct) else false;
        const quant = if (comptime build_options.enable_metal)
            if (value) |ct| metal_compute_mod.MetalCompute.getQuantizedStorage(cb, ct) != null else false
        else
            false;
        const runtime_quant = if (comptime build_options.enable_metal)
            if (value) |ct| metal_compute_mod.MetalCompute.debugHasRuntimeQuantizedStorage(cb, ct) else false
        else
            false;
        if (value) |ct| {
            const tensor_shape = cb.tensorShape(ct, std.heap.page_allocator) catch null;
            defer if (tensor_shape) |shape| std.heap.page_allocator.free(shape);
            if (tensor_shape) |shape| {
                std.debug.print(
                    "graph_executor_node_trace: input node={d} input_index={d} input_node={d} op={s} graph_shape={any} tensor_shape={any} device={} quant={} runtime_quant={}\n",
                    .{ node_id, input_index, input_id, @tagName(input_node.op), input_node.output_shape, shape, device, quant, runtime_quant },
                );
            } else {
                std.debug.print(
                    "graph_executor_node_trace: input node={d} input_index={d} input_node={d} op={s} graph_shape={any} tensor_shape=<unavailable> device={} quant={} runtime_quant={}\n",
                    .{ node_id, input_index, input_id, @tagName(input_node.op), input_node.output_shape, device, quant, runtime_quant },
                );
            }
        } else {
            std.debug.print(
                "graph_executor_node_trace: input node={d} input_index={d} input_node={d} op={s} graph_shape={any} tensor_shape=<null> device={} quant={} runtime_quant={}\n",
                .{ node_id, input_index, input_id, @tagName(input_node.op), input_node.output_shape, device, quant, runtime_quant },
            );
        }
    }
}

fn printMetalNodeTraceEnd(graph: *const Graph, cb: *const ComputeBackend, node_id: NodeId, output: CT, used_command: bool) void {
    const n = graph.node(node_id);
    const quant = if (comptime build_options.enable_metal)
        metal_compute_mod.MetalCompute.getQuantizedStorage(cb, output) != null
    else
        false;
    const runtime_quant = if (comptime build_options.enable_metal)
        metal_compute_mod.MetalCompute.debugHasRuntimeQuantizedStorage(cb, output)
    else
        false;
    std.debug.print(
        "graph_executor_node_trace: end node={d} op={s} command={} device={} quant={} runtime_quant={}\n",
        .{
            node_id,
            @tagName(n.op),
            used_command,
            isMetalDeviceResident(cb, output),
            quant,
            runtime_quant,
        },
    );
}

const GemmaRuntimeResidencyCategory = enum {
    qkv,
    o_proj,
    mlp_proj,
    attention_matmul,
    rms_norm,
    softmax,
    residual_add,
    elementwise_mul,
};

fn recordGemmaRuntimeResidency(
    stats: *PartitionExecutor.ExecutionStats,
    graph: *const Graph,
    node_id: NodeId,
    hit: bool,
) void {
    const category = classifyGemmaRuntimeResidencyNode(graph, node_id) orelse return;
    switch (category) {
        .qkv => if (hit) {
            stats.gemma_qkv_hits += 1;
        } else {
            stats.gemma_qkv_fallbacks += 1;
        },
        .o_proj => if (hit) {
            stats.gemma_o_proj_hits += 1;
        } else {
            stats.gemma_o_proj_fallbacks += 1;
        },
        .mlp_proj => if (hit) {
            stats.gemma_mlp_proj_hits += 1;
        } else {
            stats.gemma_mlp_proj_fallbacks += 1;
        },
        .attention_matmul => if (hit) {
            stats.gemma_attention_matmul_hits += 1;
        } else {
            stats.gemma_attention_matmul_fallbacks += 1;
        },
        .rms_norm => if (hit) {
            stats.gemma_rms_norm_hits += 1;
        } else {
            stats.gemma_rms_norm_fallbacks += 1;
        },
        .softmax => if (hit) {
            stats.gemma_softmax_hits += 1;
        } else {
            stats.gemma_softmax_fallbacks += 1;
        },
        .residual_add => if (hit) {
            stats.gemma_residual_add_hits += 1;
        } else {
            stats.gemma_residual_add_fallbacks += 1;
        },
        .elementwise_mul => if (hit) {
            stats.gemma_elementwise_mul_hits += 1;
        } else {
            stats.gemma_elementwise_mul_fallbacks += 1;
        },
    }
}

fn tryExecuteFusedMetalGraphPattern(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    device_id: DeviceId,
    exec_ctx: PartitionExecutor.ExecutionContext,
    exec_state: *interpreter.ExecState,
    skipped_nodes: []bool,
    last_use: []const u32,
    rt_map: std.AutoHashMapUnmanaged(NodeId, CT),
    donated: std.AutoHashMapUnmanaged(NodeId, void),
) !bool {
    if (fusedPatternProbingDisabled(exec_ctx)) return false;
    if (try tryExecuteRmsNormGroupedLinearQkvSlicePattern(
        allocator,
        graph,
        cb,
        values,
        value_device,
        node_ids,
        node_pos,
        reachable,
        device_id,
        exec_ctx,
        skipped_nodes,
        last_use,
        rt_map,
        donated,
    )) return true;
    if (try tryExecuteGroupedLinearQkvSlicePattern(
        allocator,
        graph,
        cb,
        values,
        value_device,
        node_ids,
        node_pos,
        reachable,
        device_id,
        exec_ctx,
        skipped_nodes,
        last_use,
        rt_map,
        donated,
    )) return true;
    if (try tryExecuteLinearNoBiasQkvPattern(
        allocator,
        graph,
        cb,
        values,
        value_device,
        node_ids,
        node_pos,
        reachable,
        device_id,
        exec_ctx,
        skipped_nodes,
        last_use,
        rt_map,
        donated,
    )) return true;
    if (try tryExecuteAttentionOutputResidualPattern(
        graph,
        cb,
        values,
        value_device,
        node_ids,
        node_pos,
        reachable,
        device_id,
        exec_ctx,
        exec_state,
        skipped_nodes,
        last_use,
    )) return true;
    if (try tryExecuteRmsNormGatedFfnResidualPattern(
        graph,
        cb,
        values,
        value_device,
        node_ids,
        node_pos,
        reachable,
        device_id,
        skipped_nodes,
        last_use,
        exec_ctx.stats,
    )) return true;
    if (try tryExecuteGatedFfnResidualPattern(
        graph,
        cb,
        values,
        value_device,
        node_ids,
        node_pos,
        reachable,
        device_id,
        skipped_nodes,
        last_use,
        exec_ctx.stats,
    )) return true;
    if (try tryExecutePleResidualPattern(
        graph,
        cb,
        values,
        value_device,
        node_ids,
        node_pos,
        reachable,
        device_id,
        skipped_nodes,
        exec_ctx.stats,
    )) return true;
    return tryExecuteLinearNoBiasPairPattern(
        allocator,
        graph,
        cb,
        values,
        value_device,
        node_ids,
        node_pos,
        reachable,
        device_id,
        exec_ctx,
        skipped_nodes,
        last_use,
        rt_map,
        donated,
    );
}

fn preparedRuntimeRegionMatches(region: RuntimeRegion, prepared: PreparedRuntimeRegion) bool {
    return std.meta.activeTag(region) == std.meta.activeTag(prepared);
}

fn preparedRuntimeRegionSlotCount(prepared: PreparedRuntimeRegion) u64 {
    return switch (prepared) {
        .none => 0,
        .raw_linear_dot => 1,
        .raw_linear_pair => 2,
        .raw_linear_bias => 1,
        .raw_linear_bias_pair => 2,
        .lora_linear => 0,
        .lora_linear_qkv => 0,
        .deberta_attention => 0,
        .lora_backward => 0,
        .ffn_gelu_backward => 0,
        .q_linear => 1,
        .linear_qkv, .grouped_linear_qkv_slice => 3,
        .rms_norm_grouped_linear_qkv_slice => 4,
        .attention_output_residual => |slots| 1 +
            @as(u64, if (slots.pre_linear_rms_norm_slot != null) 1 else 0) +
            @as(u64, if (slots.post_linear_rms_norm_slot != null) 1 else 0),
        .rms_norm_gated_ffn_residual => |slots| 1 + preparedRuntimeRegionSlotCount(.{ .gated_ffn_residual = slots.ffn }),
        .gated_ffn_residual => |slots| 3 + @as(u64, if (slots.post_down_rms_norm_slot != null) 1 else 0),
        .ple_residual => 3,
    };
}

fn ensurePreparedLinearSlot(
    cb: *const ComputeBackend,
    values: []?CT,
    weight_id: NodeId,
    in_dim: usize,
    out_dim: usize,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?usize {
    const weight = valueFor(values, weight_id) orelse return null;
    if (stats) |s| s.runtime_prepare_slot_calls += 1;
    return try cb.decoderRuntimeEnsureLinearSlot(&.{
        .weight = weight,
        .bias = null,
        .in_dim = in_dim,
        .out_dim = out_dim,
    });
}

fn ensurePreparedLinearSlotWithOptionalBias(
    cb: *const ComputeBackend,
    values: []?CT,
    weight_id: NodeId,
    bias_id: ?NodeId,
    in_dim: usize,
    out_dim: usize,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?usize {
    const weight = valueFor(values, weight_id) orelse return null;
    const bias = if (bias_id) |id| valueFor(values, id) orelse return null else null;
    if (stats) |s| s.runtime_prepare_slot_calls += 1;
    return try cb.decoderRuntimeEnsureLinearSlot(&.{
        .weight = weight,
        .bias = bias,
        .in_dim = in_dim,
        .out_dim = out_dim,
    });
}

fn ensurePreparedRmsNormSlot(
    cb: *const ComputeBackend,
    values: []?CT,
    weight_id: NodeId,
    hidden_size: usize,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?usize {
    const weight = valueFor(values, weight_id) orelse return null;
    if (stats) |s| s.runtime_prepare_slot_calls += 1;
    return try cb.decoderRuntimeEnsureRmsNormSlot(&.{
        .weight = weight,
        .hidden_size = hidden_size,
    });
}

fn prepareQLinearRegion(
    cb: *const ComputeBackend,
    values: []?CT,
    pattern: QLinearPattern,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?PreparedLinearRegion {
    const linear_slot = (try ensurePreparedLinearSlot(
        cb,
        values,
        pattern.weight_id,
        pattern.in_dim,
        pattern.out_dim,
        stats,
    )) orelse return null;
    return .{ .linear_slot = linear_slot };
}

fn prepareQkvRegion(
    cb: *const ComputeBackend,
    values: []?CT,
    q_weight_id: NodeId,
    k_weight_id: NodeId,
    v_weight_id: NodeId,
    in_dim: usize,
    q_out_dim: usize,
    kv_out_dim: usize,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?PreparedQkvRegion {
    const q_slot = (try ensurePreparedLinearSlot(cb, values, q_weight_id, in_dim, q_out_dim, stats)) orelse return null;
    const k_slot = (try ensurePreparedLinearSlot(cb, values, k_weight_id, in_dim, kv_out_dim, stats)) orelse return null;
    const v_slot = (try ensurePreparedLinearSlot(cb, values, v_weight_id, in_dim, kv_out_dim, stats)) orelse return null;
    return .{
        .q_slot = q_slot,
        .k_slot = k_slot,
        .v_slot = v_slot,
    };
}

fn prepareLinearNoBiasQkvRegion(
    cb: *const ComputeBackend,
    values: []?CT,
    pattern: LinearNoBiasQkvPattern,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?PreparedQkvRegion {
    return prepareQkvRegion(
        cb,
        values,
        pattern.q_weight_id,
        pattern.k_weight_id,
        pattern.v_weight_id,
        pattern.in_dim,
        pattern.q_out_dim,
        pattern.kv_out_dim,
        stats,
    );
}

fn prepareGroupedLinearQkvSliceRegion(
    cb: *const ComputeBackend,
    values: []?CT,
    pattern: GroupedLinearQkvSlicePattern,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?PreparedQkvRegion {
    return prepareQkvRegion(
        cb,
        values,
        pattern.q_weight_id,
        pattern.k_weight_id,
        pattern.v_weight_id,
        pattern.in_dim,
        pattern.q_out_dim,
        pattern.kv_out_dim,
        stats,
    );
}

fn prepareRmsNormGroupedLinearQkvSliceRegion(
    cb: *const ComputeBackend,
    values: []?CT,
    pattern: RmsNormGroupedLinearQkvSlicePattern,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?PreparedRmsNormGroupedQkvRegion {
    const norm_slot = (try ensurePreparedRmsNormSlot(cb, values, pattern.norm_weight_id, pattern.norm_dim, stats)) orelse return null;
    const qkv = (try prepareGroupedLinearQkvSliceRegion(cb, values, pattern.qkv, stats)) orelse return null;
    return .{
        .norm_slot = norm_slot,
        .qkv = qkv,
    };
}

fn prepareAttentionOutputResidualRegion(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    pattern: AttentionOutputResidualPattern,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?PreparedAttentionOutputResidualRegion {
    const linear_inputs = graph.node(pattern.linear_id).getInputs();
    if (linear_inputs.len < 2) return null;
    const linear_slot = (try ensurePreparedLinearSlot(
        cb,
        values,
        linear_inputs[1],
        pattern.attention_input_size,
        pattern.hidden_size,
        stats,
    )) orelse return null;

    var pre_linear_slot: ?usize = null;
    if (pattern.pre_linear_norm_id) |norm_id| {
        const norm_inputs = graph.node(norm_id).getInputs();
        if (norm_inputs.len < 2) return null;
        pre_linear_slot = (try ensurePreparedRmsNormSlot(
            cb,
            values,
            norm_inputs[1],
            pattern.attention_input_size,
            stats,
        )) orelse return null;
    }

    var post_linear_slot: ?usize = null;
    if (pattern.post_linear_norm_id) |norm_id| {
        const norm_inputs = graph.node(norm_id).getInputs();
        if (norm_inputs.len < 2) return null;
        post_linear_slot = (try ensurePreparedRmsNormSlot(
            cb,
            values,
            norm_inputs[1],
            pattern.hidden_size,
            stats,
        )) orelse return null;
    }

    return .{
        .linear_slot = linear_slot,
        .pre_linear_rms_norm_slot = pre_linear_slot,
        .post_linear_rms_norm_slot = post_linear_slot,
    };
}

fn prepareGatedFfnResidualRegion(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    pattern: GatedFfnResidualPattern,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?PreparedGatedFfnResidualRegion {
    const pair_inputs = graph.node(pattern.pair_id).getInputs();
    if (pair_inputs.len < 3) return null;
    const gate_slot = (try ensurePreparedLinearSlot(
        cb,
        values,
        pair_inputs[1],
        pattern.hidden_size,
        pattern.intermediate_size,
        stats,
    )) orelse return null;
    const up_slot = (try ensurePreparedLinearSlot(
        cb,
        values,
        pair_inputs[2],
        pattern.hidden_size,
        pattern.intermediate_size,
        stats,
    )) orelse return null;

    const down_inputs = graph.node(pattern.down_id).getInputs();
    if (down_inputs.len < 2) return null;
    const down_slot = (try ensurePreparedLinearSlot(
        cb,
        values,
        down_inputs[1],
        pattern.intermediate_size,
        pattern.hidden_size,
        stats,
    )) orelse return null;

    var post_down_slot: ?usize = null;
    if (pattern.post_down_norm_id) |norm_id| {
        const norm_inputs = graph.node(norm_id).getInputs();
        if (norm_inputs.len < 2) return null;
        post_down_slot = (try ensurePreparedRmsNormSlot(
            cb,
            values,
            norm_inputs[1],
            pattern.hidden_size,
            stats,
        )) orelse return null;
    }

    return .{
        .gate_slot = gate_slot,
        .up_slot = up_slot,
        .down_slot = down_slot,
        .post_down_rms_norm_slot = post_down_slot,
    };
}

fn prepareRmsNormGatedFfnResidualRegion(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    pattern: RmsNormGatedFfnResidualPattern,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?PreparedRmsNormGatedFfnResidualRegion {
    const norm_slot = (try ensurePreparedRmsNormSlot(
        cb,
        values,
        pattern.norm_weight_id,
        pattern.norm_dim,
        stats,
    )) orelse return null;
    const ffn = (try prepareGatedFfnResidualRegion(graph, cb, values, pattern.ffn, stats)) orelse return null;
    return .{ .norm_slot = norm_slot, .ffn = ffn };
}

fn preparePleResidualRegion(
    cb: *const ComputeBackend,
    values: []?CT,
    pattern: PleResidualPattern,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?PreparedPleResidualRegion {
    const gate_slot = (try ensurePreparedLinearSlot(
        cb,
        values,
        pattern.gate_weight_id,
        pattern.hidden_size,
        pattern.ple_hidden_size,
        stats,
    )) orelse return null;
    const projection_slot = (try ensurePreparedLinearSlot(
        cb,
        values,
        pattern.projection_weight_id,
        pattern.ple_hidden_size,
        pattern.hidden_size,
        stats,
    )) orelse return null;
    const post_norm_slot = (try ensurePreparedRmsNormSlot(
        cb,
        values,
        pattern.post_norm_weight_id,
        pattern.hidden_size,
        stats,
    )) orelse return null;
    return .{
        .gate_slot = gate_slot,
        .projection_slot = projection_slot,
        .post_norm_slot = post_norm_slot,
    };
}

fn prepareRuntimeRegion(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    region: RuntimeRegion,
    prepared_region: ?*PreparedRuntimeRegion,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?PreparedRuntimeRegion {
    if (prepared_region) |prepared_ptr| {
        if (preparedRuntimeRegionMatches(region, prepared_ptr.*)) {
            if (stats) |s| s.runtime_prepare_slot_cache_hits += preparedRuntimeRegionSlotCount(prepared_ptr.*);
            return prepared_ptr.*;
        }
    }

    const prepared: PreparedRuntimeRegion = switch (region) {
        .none => return null,
        .raw_linear_dot => |pattern| .{
            .raw_linear_dot = (try prepareRawLinearDotRegion(cb, values, pattern, stats)) orelse return null,
        },
        .raw_linear_pair => |pattern| .{
            .raw_linear_pair = (try prepareRawLinearPairRegion(cb, values, pattern, stats)) orelse return null,
        },
        .raw_linear_bias => |pattern| .{
            .raw_linear_bias = (try prepareRawLinearBiasRegion(cb, values, pattern, stats)) orelse return null,
        },
        .raw_linear_bias_pair => |pattern| .{
            .raw_linear_bias_pair = (try prepareRawLinearBiasPairRegion(cb, values, pattern, stats)) orelse return null,
        },
        .lora_linear => .{ .lora_linear = {} },
        .lora_linear_qkv => .{ .lora_linear_qkv = {} },
        .deberta_attention => .{ .deberta_attention = {} },
        .lora_backward => .{ .lora_backward = {} },
        .ffn_gelu_backward => .{ .ffn_gelu_backward = {} },
        .q_linear => |pattern| .{
            .q_linear = (try prepareQLinearRegion(cb, values, pattern, stats)) orelse return null,
        },
        .linear_qkv => |pattern| .{
            .linear_qkv = (try prepareLinearNoBiasQkvRegion(cb, values, pattern, stats)) orelse return null,
        },
        .grouped_linear_qkv_slice => |pattern| .{
            .grouped_linear_qkv_slice = (try prepareGroupedLinearQkvSliceRegion(cb, values, pattern, stats)) orelse return null,
        },
        .rms_norm_grouped_linear_qkv_slice => |pattern| .{
            .rms_norm_grouped_linear_qkv_slice = (try prepareRmsNormGroupedLinearQkvSliceRegion(cb, values, pattern, stats)) orelse return null,
        },
        .attention_output_residual => |pattern| .{
            .attention_output_residual = (try prepareAttentionOutputResidualRegion(graph, cb, values, pattern, stats)) orelse return null,
        },
        .rms_norm_gated_ffn_residual => |pattern| .{
            .rms_norm_gated_ffn_residual = (try prepareRmsNormGatedFfnResidualRegion(graph, cb, values, pattern, stats)) orelse return null,
        },
        .gated_ffn_residual => |pattern| .{
            .gated_ffn_residual = (try prepareGatedFfnResidualRegion(graph, cb, values, pattern, stats)) orelse return null,
        },
        .ple_residual => |pattern| .{
            .ple_residual = (try preparePleResidualRegion(cb, values, pattern, stats)) orelse return null,
        },
    };
    if (prepared_region) |prepared_ptr| prepared_ptr.* = prepared;
    return prepared;
}

fn tryExecutePlannedRuntimeRegion(
    region: RuntimeRegion,
    prepared_region: ?*PreparedRuntimeRegion,
    allocator: std.mem.Allocator,
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    device_id: DeviceId,
    exec_ctx: PartitionExecutor.ExecutionContext,
    exec_state: *interpreter.ExecState,
    skipped_nodes: []bool,
    last_use: []const u32,
    rt_map: std.AutoHashMapUnmanaged(NodeId, CT),
    donated: std.AutoHashMapUnmanaged(NodeId, void),
) !bool {
    _ = node_ids;
    _ = node_pos;
    _ = reachable;
    const prepared = (try prepareRuntimeRegion(graph, cb, values, region, prepared_region, exec_ctx.stats)) orelse {
        if (std.meta.activeTag(region) != .none) {
            if (exec_ctx.stats) |stats| stats.runtime_region_fallbacks += 1;
        }
        return false;
    };
    return switch (region) {
        .none => false,
        .raw_linear_dot => |pattern| executeRawLinearDotPattern(
            graph,
            cb,
            values,
            value_device,
            device_id,
            exec_ctx.stats,
            pattern,
            switch (prepared) {
                .raw_linear_dot => |slots| slots,
                else => return false,
            },
        ),
        .raw_linear_pair => |pattern| executeRawLinearPairPattern(
            graph,
            cb,
            values,
            value_device,
            device_id,
            exec_ctx.stats,
            pattern,
            switch (prepared) {
                .raw_linear_pair => |slots| slots,
                else => return false,
            },
            skipped_nodes,
        ),
        .raw_linear_bias => |pattern| executeRawLinearBiasPattern(
            graph,
            cb,
            values,
            value_device,
            device_id,
            exec_ctx.stats,
            pattern,
            switch (prepared) {
                .raw_linear_bias => |slots| slots,
                else => return false,
            },
            skipped_nodes,
        ),
        .raw_linear_bias_pair => |pattern| executeRawLinearBiasPairPattern(
            graph,
            cb,
            values,
            value_device,
            device_id,
            exec_ctx.stats,
            pattern,
            switch (prepared) {
                .raw_linear_bias_pair => |slots| slots,
                else => return false,
            },
            skipped_nodes,
        ),
        .lora_linear => |pattern| executeLoraLinearPattern(
            graph,
            cb,
            values,
            value_device,
            device_id,
            exec_ctx.stats,
            pattern,
        ),
        .lora_linear_qkv => |pattern| executeLoraLinearQkvPattern(
            graph,
            cb,
            values,
            value_device,
            device_id,
            exec_ctx.stats,
            pattern,
            skipped_nodes,
        ),
        .deberta_attention => |pattern| executeDebertaAttentionPattern(
            graph,
            cb,
            allocator,
            values,
            value_device,
            device_id,
            exec_ctx.stats,
            pattern,
            skipped_nodes,
        ),
        .lora_backward => |pattern| executeLoraBackwardPattern(
            graph,
            cb,
            values,
            value_device,
            device_id,
            exec_ctx.stats,
            pattern,
            skipped_nodes,
        ),
        .ffn_gelu_backward => |pattern| executeFfnGeluBackwardPattern(
            graph,
            cb,
            values,
            value_device,
            device_id,
            exec_ctx.stats,
            exec_ctx.partition_plan,
            pattern,
            skipped_nodes,
        ),
        .q_linear => |pattern| executeQLinearPattern(
            graph,
            cb,
            values,
            value_device,
            device_id,
            exec_ctx.stats,
            pattern,
            switch (prepared) {
                .q_linear => |slots| slots,
                else => return false,
            },
        ),
        .linear_qkv => |pattern| executeLinearNoBiasQkvPattern(
            allocator,
            graph,
            cb,
            values,
            value_device,
            device_id,
            exec_ctx,
            skipped_nodes,
            last_use,
            rt_map,
            donated,
            pattern,
            switch (prepared) {
                .linear_qkv => |slots| slots,
                else => return false,
            },
        ),
        .grouped_linear_qkv_slice => |pattern| executeGroupedLinearQkvSlicePattern(
            allocator,
            graph,
            cb,
            values,
            value_device,
            device_id,
            exec_ctx,
            skipped_nodes,
            last_use,
            rt_map,
            donated,
            pattern,
            switch (prepared) {
                .grouped_linear_qkv_slice => |slots| slots,
                else => return false,
            },
        ),
        .rms_norm_grouped_linear_qkv_slice => |pattern| executeRmsNormGroupedLinearQkvSlicePattern(
            allocator,
            graph,
            cb,
            values,
            value_device,
            device_id,
            exec_ctx,
            skipped_nodes,
            last_use,
            rt_map,
            donated,
            pattern,
            switch (prepared) {
                .rms_norm_grouped_linear_qkv_slice => |slots| slots,
                else => return false,
            },
        ),
        .attention_output_residual => |pattern| executeAttentionOutputResidualPattern(
            graph,
            cb,
            values,
            value_device,
            device_id,
            exec_ctx,
            exec_state,
            skipped_nodes,
            pattern,
            switch (prepared) {
                .attention_output_residual => |slots| slots,
                else => return false,
            },
        ),
        .rms_norm_gated_ffn_residual => |pattern| executeRmsNormGatedFfnResidualPattern(
            graph,
            cb,
            values,
            value_device,
            device_id,
            skipped_nodes,
            exec_ctx.stats,
            pattern,
            switch (prepared) {
                .rms_norm_gated_ffn_residual => |slots| slots,
                else => return false,
            },
        ),
        .gated_ffn_residual => |pattern| executeMatchedGatedFfnResidualPattern(
            graph,
            cb,
            values,
            value_device,
            device_id,
            skipped_nodes,
            exec_ctx.stats,
            pattern,
            switch (prepared) {
                .gated_ffn_residual => |slots| slots,
                else => return false,
            },
        ),
        .ple_residual => |pattern| executePleResidualPattern(
            graph,
            cb,
            values,
            value_device,
            device_id,
            skipped_nodes,
            exec_ctx.stats,
            pattern,
            switch (prepared) {
                .ple_residual => |slots| slots,
                else => return false,
            },
        ),
    } catch |err| switch (err) {
        error.UnsupportedOperation,
        error.UnsupportedPrimitiveOp,
        error.UnsupportedShape,
        error.ShapeMismatch,
        error.UnsupportedTensorType,
        => false,
        else => return err,
    };
}

const AttentionOutputResidualPattern = struct {
    attention_id: NodeId,
    pre_linear_norm_id: ?NodeId,
    linear_id: NodeId,
    post_linear_norm_id: ?NodeId,
    add_id: NodeId,
    residual_id: NodeId,
    rows: usize,
    attention_input_size: usize,
    hidden_size: usize,
    eps: f32,

    fn elidedNodeCount(self: AttentionOutputResidualPattern) u64 {
        return 2 + @as(u64, if (self.pre_linear_norm_id != null) 1 else 0) +
            @as(u64, if (self.post_linear_norm_id != null) 1 else 0);
    }
};

fn tryExecuteAttentionOutputResidualPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    device_id: DeviceId,
    exec_ctx: PartitionExecutor.ExecutionContext,
    exec_state: *interpreter.ExecState,
    skipped_nodes: []bool,
    last_use: []const u32,
) !bool {
    if (!attentionOutputResidualGraphFusionEnabled()) return false;
    const pattern = matchAttentionOutputResidualPattern(graph, node_ids, node_pos, reachable, skipped_nodes, last_use) orelse return false;
    const prepared = (try prepareAttentionOutputResidualRegion(graph, cb, values, pattern, exec_ctx.stats)) orelse return false;
    return executeAttentionOutputResidualPattern(
        graph,
        cb,
        values,
        value_device,
        device_id,
        exec_ctx,
        exec_state,
        skipped_nodes,
        pattern,
        prepared,
    );
}

fn executeAttentionOutputResidualPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    exec_ctx: PartitionExecutor.ExecutionContext,
    exec_state: *interpreter.ExecState,
    skipped_nodes: []bool,
    pattern: AttentionOutputResidualPattern,
    prepared: PreparedAttentionOutputResidualRegion,
) !bool {
    const attention_node = graph.node(pattern.attention_id);
    const attention_inputs = attention_node.getInputs();
    const attention_attrs = switch (attention_node.op) {
        .fused_gqa_causal_attention => |attrs| attrs,
        else => return false,
    };

    const residual = valueFor(values, pattern.residual_id) orelse return false;
    const attention_output = (try executeRuntimeGqaCausalAttention(
        cb,
        values,
        attention_inputs,
        attention_attrs,
        attention_node.num_inputs,
        exec_state,
    )) orelse return false;
    errdefer cb.free(attention_output);

    const planned_scope = try metal_compute_mod.MetalCompute.beginPlannedGraphScope(cb, .attention_project);
    defer metal_compute_mod.MetalCompute.endPlannedGraphScope(cb, planned_scope) catch {};
    const fused = cb.runAttentionOutputResidual(&.{
        .attention_output = attention_output,
        .residual = residual,
        .rows = pattern.rows,
        .attention_input_size = pattern.attention_input_size,
        .hidden_size = pattern.hidden_size,
        .linear_slot = prepared.linear_slot,
        .pre_linear_rms_norm_slot = prepared.pre_linear_rms_norm_slot,
        .post_linear_rms_norm_slot = prepared.post_linear_rms_norm_slot,
        .eps = pattern.eps,
    }) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch, error.UnsupportedTensorType => null,
        else => return err,
    };

    if (fused) |output| {
        if (output != attention_output) cb.free(attention_output);
        values[@intCast(pattern.attention_id)] = output;
        values[@intCast(pattern.add_id)] = output;
        value_device[@intCast(pattern.attention_id)] = device_id;
        value_device[@intCast(pattern.add_id)] = device_id;
        skipped_nodes[@intCast(pattern.attention_id)] = true;
        if (pattern.pre_linear_norm_id) |norm_id| skipped_nodes[@intCast(norm_id)] = true;
        skipped_nodes[@intCast(pattern.linear_id)] = true;
        if (pattern.post_linear_norm_id) |norm_id| skipped_nodes[@intCast(norm_id)] = true;
        skipped_nodes[@intCast(pattern.add_id)] = true;
        if (exec_ctx.stats) |stats| {
            recordMetalGraphRegion(stats, .attention, pattern.elidedNodeCount());
            stats.fused_graph_pattern_dispatches += 1;
            stats.fused_graph_nodes_elided += pattern.elidedNodeCount();
            stats.metal_attention_output_residual_fusions += 1;
        }
        if (traceMetalGraphFusionsEnabled()) {
            std.debug.print(
                "metal_graph_fusion_trace: attention_output_residual executed attention={d} linear={d} post_norm={?d} add={d} rows={d} attention_dim={d} hidden={d}\n",
                .{ pattern.attention_id, pattern.linear_id, pattern.post_linear_norm_id, pattern.add_id, pattern.rows, pattern.attention_input_size, pattern.hidden_size },
            );
        }
        return true;
    }

    values[@intCast(pattern.attention_id)] = attention_output;
    value_device[@intCast(pattern.attention_id)] = device_id;
    if (exec_ctx.stats) |stats| {
        stats.graph_region_fallbacks += 1;
        stats.metal_attention_output_residual_partial_fallbacks += 1;
    }
    return true;
}

fn matchAttentionOutputResidualPattern(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
    last_use: []const u32,
) ?AttentionOutputResidualPattern {
    const attention_id = node_ids[node_pos];
    const attention = graph.node(attention_id);
    const attention_attrs = switch (attention.op) {
        .fused_gqa_causal_attention => |attrs| attrs,
        else => return null,
    };
    if (attention_attrs.num_heads == 0 or attention_attrs.head_dim == 0) return null;
    const attention_input_size = @as(usize, attention_attrs.num_heads) * @as(usize, attention_attrs.head_dim);

    var linear_input_id = attention_id;
    var pre_linear_norm_id: ?NodeId = null;
    var eps: f32 = 0.0;
    if (findSingleInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, attention_id, &isMatchingPostDownRmsNorm)) |norm_id| {
        const norm = graph.node(norm_id);
        const norm_attrs = switch (norm.op) {
            .fused_rms_norm => |attrs| attrs,
            else => return null,
        };
        if (norm_attrs.dim == attention_input_size) {
            pre_linear_norm_id = norm_id;
            linear_input_id = norm_id;
            eps = norm_attrs.eps;
        }
    }

    const linear_id = findSingleInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, linear_input_id, &isPlainLinearNoBiasNode) orelse return null;
    const linear = graph.node(linear_id);
    const linear_attrs = switch (linear.op) {
        .fused_linear_no_bias => |attrs| attrs,
        else => return null,
    };
    if (linear_attrs.in_dim != attention_input_size or linear_attrs.out_dim == 0 or linear_attrs.rows == 0) return null;

    var add_lhs_id = linear_id;
    var post_linear_norm_id: ?NodeId = null;
    if (findSingleInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, linear_id, &isMatchingPostDownRmsNorm)) |norm_id| {
        const norm = graph.node(norm_id);
        const norm_attrs = switch (norm.op) {
            .fused_rms_norm => |attrs| attrs,
            else => return null,
        };
        if (norm_attrs.dim == linear_attrs.out_dim) {
            if (pre_linear_norm_id != null and norm_attrs.eps != eps) return null;
            post_linear_norm_id = norm_id;
            add_lhs_id = norm_id;
            eps = norm_attrs.eps;
        }
    }

    const add_id = findSingleInputNodeAsBinaryLhs(graph, node_ids, node_pos + 1, reachable, skipped_nodes, add_lhs_id, &isAddNode) orelse return null;
    const add_inputs = graph.node(add_id).getInputs();
    if (add_inputs.len < 2) return null;
    const residual_id = if (add_inputs[0] == add_lhs_id) add_inputs[1] else add_inputs[0];
    if (residual_id == null_node) return null;

    if (pre_linear_norm_id) |norm_id| {
        if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, attention_id, &.{norm_id})) return null;
        if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, norm_id, &.{linear_id})) return null;
        if (!nodeLastUseIs(last_use, attention_id, norm_id)) return null;
        if (!nodeLastUseIs(last_use, norm_id, linear_id)) return null;
    } else {
        if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, attention_id, &.{linear_id})) return null;
        if (!nodeLastUseIs(last_use, attention_id, linear_id)) return null;
    }
    if (post_linear_norm_id) |norm_id| {
        if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, linear_id, &.{norm_id})) return null;
        if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, norm_id, &.{add_id})) return null;
        if (!nodeLastUseIs(last_use, linear_id, norm_id)) return null;
        if (!nodeLastUseIs(last_use, norm_id, add_id)) return null;
    } else {
        if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, linear_id, &.{add_id})) return null;
        if (!nodeLastUseIs(last_use, linear_id, add_id)) return null;
    }

    return .{
        .attention_id = attention_id,
        .pre_linear_norm_id = pre_linear_norm_id,
        .linear_id = linear_id,
        .post_linear_norm_id = post_linear_norm_id,
        .add_id = add_id,
        .residual_id = residual_id,
        .rows = linear_attrs.rows,
        .attention_input_size = attention_input_size,
        .hidden_size = linear_attrs.out_dim,
        .eps = eps,
    };
}

const GatedFfnResidualPattern = struct {
    pair_id: NodeId,
    pair_second_id: NodeId,
    activation_id: NodeId,
    multiply_id: NodeId,
    down_id: NodeId,
    post_down_norm_id: ?NodeId,
    add_id: NodeId,
    residual_id: NodeId,
    activation: ops_mod.DecoderRuntimeActivationKind,
    hidden_size: usize,
    intermediate_size: usize,
    rows: usize,
    eps: f32,

    fn elidedNodeCount(self: GatedFfnResidualPattern) u64 {
        return 6 + @as(u64, if (self.post_down_norm_id != null) 1 else 0);
    }
};

const RmsNormGatedFfnResidualPattern = struct {
    norm_id: NodeId,
    norm_input_id: NodeId,
    norm_weight_id: NodeId,
    norm_dim: usize,
    norm_eps: f32,
    ffn: GatedFfnResidualPattern,

    fn elidedNodeCount(self: RmsNormGatedFfnResidualPattern) u64 {
        return 1 + self.ffn.elidedNodeCount();
    }
};

fn tryExecuteRmsNormGatedFfnResidualPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    device_id: DeviceId,
    skipped_nodes: []bool,
    last_use: []const u32,
    stats: ?*PartitionExecutor.ExecutionStats,
) !bool {
    if (!gatedFfnGraphFusionEnabled()) return false;
    const pattern = matchRmsNormGatedFfnResidualPattern(graph, node_ids, node_pos, reachable, skipped_nodes, last_use) orelse return false;
    const prepared = (try prepareRmsNormGatedFfnResidualRegion(graph, cb, values, pattern, stats)) orelse return false;
    return executeRmsNormGatedFfnResidualPattern(
        graph,
        cb,
        values,
        value_device,
        device_id,
        skipped_nodes,
        stats,
        pattern,
        prepared,
    );
}

fn executeRmsNormGatedFfnResidualPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    skipped_nodes: []bool,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: RmsNormGatedFfnResidualPattern,
    prepared: PreparedRmsNormGatedFfnResidualRegion,
) !bool {
    const input = valueFor(values, pattern.norm_input_id) orelse return traceGatedFfnDeclined("missing_rms_input", pattern.norm_input_id);
    const normed = cb.decoderRuntimeApplyRmsNorm(&.{
        .slot = prepared.norm_slot,
        .input = input,
        .hidden_size = pattern.norm_dim,
        .eps = pattern.norm_eps,
    }) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => return traceGatedFfnDeclined("rms_norm_unavailable", pattern.norm_id),
        else => return err,
    } orelse return traceGatedFfnDeclined("rms_norm_unavailable", pattern.norm_id);
    errdefer cb.free(normed);

    const output = try executeGatedFfnResidualPattern(
        graph,
        cb,
        values,
        value_device,
        device_id,
        skipped_nodes,
        stats,
        pattern.ffn,
        normed,
        pattern.norm_id,
        false,
        1,
        prepared.ffn,
    );
    if (output == null) return false;
    cb.free(normed);
    return true;
}

fn matchRmsNormGatedFfnResidualPattern(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
    last_use: []const u32,
) ?RmsNormGatedFfnResidualPattern {
    const norm_id = node_ids[node_pos];
    const norm = graph.node(norm_id);
    const norm_attrs = switch (norm.op) {
        .fused_rms_norm => |attrs| attrs,
        else => return null,
    };
    if (norm_attrs.dim == 0) return null;
    const norm_inputs = norm.getInputs();
    if (norm_inputs.len < 2) return null;
    const pair_id = findSingleInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, norm_id, &isLinearNoBiasPairNode) orelse return null;
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, norm_id, &.{pair_id})) return null;
    const pair_pos = findNodePos(node_ids, pair_id) orelse return null;
    const ffn = matchGatedFfnResidualPattern(graph, node_ids, pair_pos, reachable, skipped_nodes, last_use) orelse return null;

    return .{
        .norm_id = norm_id,
        .norm_input_id = norm_inputs[0],
        .norm_weight_id = norm_inputs[1],
        .norm_dim = norm_attrs.dim,
        .norm_eps = norm_attrs.eps,
        .ffn = ffn,
    };
}

fn tryExecuteGatedFfnResidualPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    device_id: DeviceId,
    skipped_nodes: []bool,
    last_use: []const u32,
    stats: ?*PartitionExecutor.ExecutionStats,
) !bool {
    if (!gatedFfnGraphFusionEnabled()) return false;
    const pattern = matchGatedFfnResidualPattern(graph, node_ids, node_pos, reachable, skipped_nodes, last_use) orelse {
        if (traceMetalGraphFusionsEnabled()) traceGatedFfnResidualCandidate(graph, node_ids, node_pos, reachable, skipped_nodes, last_use);
        return false;
    };
    const prepared = (try prepareGatedFfnResidualRegion(graph, cb, values, pattern, stats)) orelse return false;
    return executeMatchedGatedFfnResidualPattern(
        graph,
        cb,
        values,
        value_device,
        device_id,
        skipped_nodes,
        stats,
        pattern,
        prepared,
    );
}

fn executeMatchedGatedFfnResidualPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    skipped_nodes: []bool,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: GatedFfnResidualPattern,
    prepared: PreparedGatedFfnResidualRegion,
) !bool {
    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: gated_ffn match pair={d} pair_second={d} activation={d} multiply={d} down={d} post_norm={?d} add={d} residual={d} rows={d} hidden={d} intermediate={d} activation_kind={s}\n",
            .{
                pattern.pair_id,
                pattern.pair_second_id,
                pattern.activation_id,
                pattern.multiply_id,
                pattern.down_id,
                pattern.post_down_norm_id,
                pattern.add_id,
                pattern.residual_id,
                pattern.rows,
                pattern.hidden_size,
                pattern.intermediate_size,
                @tagName(pattern.activation),
            },
        );
    }
    const pair = graph.node(pattern.pair_id);
    const pair_inputs = pair.getInputs();
    if (pair_inputs.len < 3) return traceGatedFfnDeclined("short_pair_inputs", pattern.pair_id);
    const input = valueFor(values, pair_inputs[0]) orelse return traceGatedFfnDeclined("missing_input", pair_inputs[0]);
    return (try executeGatedFfnResidualPattern(
        graph,
        cb,
        values,
        value_device,
        device_id,
        skipped_nodes,
        stats,
        pattern,
        input,
        pattern.pair_id,
        true,
        0,
        prepared,
    )) != null;
}

fn executeGatedFfnResidualPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    skipped_nodes: []bool,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: GatedFfnResidualPattern,
    input: CT,
    current_output_id: NodeId,
    publish_pair_output: bool,
    extra_elided_nodes: u64,
    prepared: PreparedGatedFfnResidualRegion,
) !?CT {
    const pair = graph.node(pattern.pair_id);
    const pair_inputs = pair.getInputs();
    if (pair_inputs.len < 3) return traceGatedFfnDeclinedNull("short_pair_inputs", pattern.pair_id);
    const residual = valueFor(values, pattern.residual_id) orelse return traceGatedFfnDeclinedNull("missing_residual", pattern.residual_id);

    var post_down_weight: ?CT = null;
    const post_down_slot = if (pattern.post_down_norm_id) |norm_id| blk: {
        const norm = graph.node(norm_id);
        const norm_inputs = norm.getInputs();
        if (norm_inputs.len < 2) return traceGatedFfnDeclinedNull("short_norm_inputs", norm_id);
        const norm_weight = valueFor(values, norm_inputs[1]) orelse return traceGatedFfnDeclinedNull("missing_norm_weight", norm_inputs[1]);
        post_down_weight = norm_weight;
        break :blk prepared.post_down_rms_norm_slot orelse return traceGatedFfnDeclinedNull("norm_slot_unavailable", norm_id);
    } else null;

    const planned_scope = try metal_compute_mod.MetalCompute.beginPlannedGraphScope(cb, .ffn);
    defer metal_compute_mod.MetalCompute.endPlannedGraphScope(cb, planned_scope) catch {};
    const output = (try cb.runGatedFfnResidual(&.{
        .gate_linear_slot = prepared.gate_slot,
        .up_linear_slot = prepared.up_slot,
        .down_linear_slot = prepared.down_slot,
        .input = input,
        .residual = residual,
        .post_down_rms_norm_slot = post_down_slot,
        .post_down_rms_norm_weight = post_down_weight,
        .hidden_size = pattern.hidden_size,
        .intermediate_size = pattern.intermediate_size,
        .eps = pattern.eps,
        .activation = pattern.activation,
    })) orelse return traceGatedFfnDeclinedNull("backend_returned_null", pattern.pair_id);

    values[@intCast(current_output_id)] = output;
    if (publish_pair_output) values[@intCast(pattern.pair_id)] = output;
    values[@intCast(pattern.add_id)] = output;
    value_device[@intCast(current_output_id)] = device_id;
    if (publish_pair_output) value_device[@intCast(pattern.pair_id)] = device_id;
    value_device[@intCast(pattern.add_id)] = device_id;

    skipped_nodes[@intCast(current_output_id)] = true;
    skipped_nodes[@intCast(pattern.pair_id)] = true;
    skipped_nodes[@intCast(pattern.pair_second_id)] = true;
    skipped_nodes[@intCast(pattern.activation_id)] = true;
    skipped_nodes[@intCast(pattern.multiply_id)] = true;
    skipped_nodes[@intCast(pattern.down_id)] = true;
    if (pattern.post_down_norm_id) |norm_id| skipped_nodes[@intCast(norm_id)] = true;
    skipped_nodes[@intCast(pattern.add_id)] = true;
    if (stats) |s| {
        recordMetalGraphRegion(s, .ffn, pattern.elidedNodeCount() + extra_elided_nodes);
        s.fused_graph_pattern_dispatches += 1;
        s.fused_graph_nodes_elided += pattern.elidedNodeCount() + extra_elided_nodes;
        s.metal_gated_ffn_residual_fusions += 1;
        if (extra_elided_nodes != 0) s.gemma_rms_norm_hits += 1;
    }
    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: gated_ffn executed pair={d} add={d}\n",
            .{ pattern.pair_id, pattern.add_id },
        );
    }
    return output;
}

const PleResidualPattern = struct {
    gate_id: NodeId,
    activation_id: NodeId,
    multiply_id: NodeId,
    projection_id: NodeId,
    post_norm_id: NodeId,
    add_id: NodeId,
    hidden_id: NodeId,
    ple_id: NodeId,
    gate_weight_id: NodeId,
    projection_weight_id: NodeId,
    post_norm_weight_id: NodeId,
    rows: usize,
    hidden_size: usize,
    ple_hidden_size: usize,
    eps: f32,
    activation: ops_mod.DecoderRuntimeActivationKind,

    fn elidedNodeCount(_: PleResidualPattern) u64 {
        return 5;
    }
};

fn tryExecutePleResidualPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    device_id: DeviceId,
    skipped_nodes: []bool,
    stats: ?*PartitionExecutor.ExecutionStats,
) !bool {
    const pattern = matchPleResidualPattern(graph, node_ids, node_pos, reachable, skipped_nodes) orelse return false;
    const prepared = (try preparePleResidualRegion(cb, values, pattern, stats)) orelse return false;
    return executePleResidualPattern(
        graph,
        cb,
        values,
        value_device,
        device_id,
        skipped_nodes,
        stats,
        pattern,
        prepared,
    );
}

fn executePleResidualPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    skipped_nodes: []bool,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: PleResidualPattern,
    prepared: PreparedPleResidualRegion,
) !bool {
    _ = graph;
    const hidden = valueFor(values, pattern.hidden_id) orelse return false;
    const ple = valueFor(values, pattern.ple_id) orelse return false;

    const planned_scope = try metal_compute_mod.MetalCompute.beginPlannedGraphScope(cb, .ple);
    defer metal_compute_mod.MetalCompute.endPlannedGraphScope(cb, planned_scope) catch {};
    const output = (try runMetalPleResidual(
        cb,
        hidden,
        ple,
        prepared.gate_slot,
        prepared.projection_slot,
        prepared.post_norm_slot,
        pattern.hidden_size,
        pattern.ple_hidden_size,
        pattern.eps,
        pattern.activation,
    )) orelse return tracePleDeclined("backend_unavailable", pattern.gate_id);

    values[@intCast(pattern.gate_id)] = output;
    values[@intCast(pattern.add_id)] = output;
    value_device[@intCast(pattern.gate_id)] = device_id;
    value_device[@intCast(pattern.add_id)] = device_id;
    skipped_nodes[@intCast(pattern.gate_id)] = true;
    skipped_nodes[@intCast(pattern.activation_id)] = true;
    skipped_nodes[@intCast(pattern.multiply_id)] = true;
    skipped_nodes[@intCast(pattern.projection_id)] = true;
    skipped_nodes[@intCast(pattern.post_norm_id)] = true;
    skipped_nodes[@intCast(pattern.add_id)] = true;
    if (stats) |s| {
        recordMetalGraphRegion(s, .ple, pattern.elidedNodeCount());
        s.fused_graph_pattern_dispatches += 1;
        s.fused_graph_nodes_elided += pattern.elidedNodeCount();
    }
    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: ple_residual executed gate={d} activation={d} multiply={d} projection={d} post_norm={d} add={d} rows={d} hidden={d} ple_hidden={d}\n",
            .{ pattern.gate_id, pattern.activation_id, pattern.multiply_id, pattern.projection_id, pattern.post_norm_id, pattern.add_id, pattern.rows, pattern.hidden_size, pattern.ple_hidden_size },
        );
    }
    return true;
}

fn runMetalPleResidual(
    cb: *const ComputeBackend,
    hidden: CT,
    ple: CT,
    gate_linear_slot: usize,
    projection_linear_slot: usize,
    post_norm_slot: usize,
    hidden_size: usize,
    ple_hidden_size: usize,
    eps: f32,
    activation: ops_mod.DecoderRuntimeActivationKind,
) !?CT {
    if (cb.kind() != .metal) return null;
    if (comptime !build_options.enable_metal) return null;
    return metal_compute_mod.MetalCompute.applyPleResidual(
        cb,
        hidden,
        ple,
        gate_linear_slot,
        projection_linear_slot,
        post_norm_slot,
        hidden_size,
        ple_hidden_size,
        eps,
        activation,
    );
}

fn matchPleResidualPattern(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?PleResidualPattern {
    const gate_id = node_ids[node_pos];
    const gate = graph.node(gate_id);
    const gate_attrs = switch (gate.op) {
        .fused_linear_no_bias => |attrs| attrs,
        else => return null,
    };
    if (gate_attrs.num_projections != 0 or gate_attrs.rows == 0 or gate_attrs.in_dim == 0 or gate_attrs.out_dim == 0) return tracePleDeclinedNull("unsupported_gate_attrs", gate_id);
    const gate_inputs = gate.getInputs();
    if (gate_inputs.len < 2) return tracePleDeclinedNull("short_gate_inputs", gate_id);
    const hidden_id = gate_inputs[0];
    const gate_weight_id = gate_inputs[1];

    const activation_id = findSingleInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, gate_id, &isSupportedGatedFfnActivation) orelse return tracePleDeclinedNull("missing_activation", gate_id);
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, gate_id, &.{activation_id})) return tracePleDeclinedNull("escaped_gate", gate_id);
    const activation_pos = findNodePos(node_ids, activation_id) orelse return tracePleDeclinedNull("activation_not_in_partition", activation_id);
    const activation = activationKindForGraphNode(graph.node(activation_id)) orelse return tracePleDeclinedNull("unsupported_activation", activation_id);

    const multiply_id = findSingleInputNodeAsBinaryLhs(graph, node_ids, activation_pos + 1, reachable, skipped_nodes, activation_id, &isMultiplyNode) orelse return tracePleDeclinedNull("missing_multiply", activation_id);
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, activation_id, &.{multiply_id})) return tracePleDeclinedNull("escaped_activation", activation_id);
    const multiply = graph.node(multiply_id);
    const multiply_inputs = multiply.getInputs();
    if (multiply_inputs.len < 2) return tracePleDeclinedNull("short_multiply_inputs", multiply_id);
    const ple_id = if (multiply_inputs[0] == activation_id) multiply_inputs[1] else multiply_inputs[0];
    if (ple_id == null_node or ple_id == hidden_id) return tracePleDeclinedNull("invalid_ple_input", multiply_id);

    const multiply_pos = findNodePos(node_ids, multiply_id) orelse return tracePleDeclinedNull("multiply_not_in_partition", multiply_id);
    const projection_id = findSingleInputNode(graph, node_ids, multiply_pos + 1, reachable, skipped_nodes, multiply_id, &isPlainLinearNoBiasNode) orelse return tracePleDeclinedNull("missing_projection", multiply_id);
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, multiply_id, &.{projection_id})) return tracePleDeclinedNull("escaped_multiply", multiply_id);
    const projection = graph.node(projection_id);
    const projection_attrs = switch (projection.op) {
        .fused_linear_no_bias => |attrs| attrs,
        else => return tracePleDeclinedNull("projection_not_linear", projection_id),
    };
    if (projection_attrs.rows != gate_attrs.rows or
        projection_attrs.in_dim != gate_attrs.out_dim or
        projection_attrs.out_dim != gate_attrs.in_dim)
    {
        return tracePleDeclinedNull("projection_shape_mismatch", projection_id);
    }
    const projection_inputs = projection.getInputs();
    if (projection_inputs.len < 2) return tracePleDeclinedNull("short_projection_inputs", projection_id);
    const projection_weight_id = projection_inputs[1];

    const projection_pos = findNodePos(node_ids, projection_id) orelse return tracePleDeclinedNull("projection_not_in_partition", projection_id);
    const post_norm_id = findSingleInputNode(graph, node_ids, projection_pos + 1, reachable, skipped_nodes, projection_id, &isMatchingPostDownRmsNorm) orelse return tracePleDeclinedNull("missing_post_norm", projection_id);
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, projection_id, &.{post_norm_id})) return tracePleDeclinedNull("escaped_projection", projection_id);
    const post_norm = graph.node(post_norm_id);
    const post_norm_attrs = switch (post_norm.op) {
        .fused_rms_norm => |attrs| attrs,
        else => return tracePleDeclinedNull("post_norm_not_rms", post_norm_id),
    };
    if (post_norm_attrs.dim != gate_attrs.in_dim) return tracePleDeclinedNull("post_norm_dim_mismatch", post_norm_id);
    const post_norm_inputs = post_norm.getInputs();
    if (post_norm_inputs.len < 2) return tracePleDeclinedNull("short_post_norm_inputs", post_norm_id);
    const post_norm_weight_id = post_norm_inputs[1];

    const post_norm_pos = findNodePos(node_ids, post_norm_id) orelse return tracePleDeclinedNull("post_norm_not_in_partition", post_norm_id);
    const add_id = findBinaryInputNode(graph, node_ids, post_norm_pos + 1, reachable, skipped_nodes, hidden_id, post_norm_id, &isAddNode) orelse return tracePleDeclinedNull("missing_add", post_norm_id);
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, post_norm_id, &.{add_id})) return tracePleDeclinedNull("escaped_post_norm", post_norm_id);

    return .{
        .gate_id = gate_id,
        .activation_id = activation_id,
        .multiply_id = multiply_id,
        .projection_id = projection_id,
        .post_norm_id = post_norm_id,
        .add_id = add_id,
        .hidden_id = hidden_id,
        .ple_id = ple_id,
        .gate_weight_id = gate_weight_id,
        .projection_weight_id = projection_weight_id,
        .post_norm_weight_id = post_norm_weight_id,
        .rows = gate_attrs.rows,
        .hidden_size = gate_attrs.in_dim,
        .ple_hidden_size = gate_attrs.out_dim,
        .eps = post_norm_attrs.eps,
        .activation = activation,
    };
}

fn tracePleDeclinedNull(reason: []const u8, node_id: NodeId) ?PleResidualPattern {
    _ = tracePleDeclined(reason, node_id);
    return null;
}

fn tracePleDeclined(reason: []const u8, node_id: NodeId) bool {
    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: ple_residual declined reason={s} node={d}\n",
            .{ reason, node_id },
        );
    }
    return false;
}

fn traceGatedFfnDeclined(reason: []const u8, node_id: NodeId) bool {
    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: gated_ffn declined reason={s} node={d}\n",
            .{ reason, node_id },
        );
    }
    return false;
}

fn traceGatedFfnDeclinedNull(reason: []const u8, node_id: NodeId) ?CT {
    _ = traceGatedFfnDeclined(reason, node_id);
    return null;
}

fn matchGatedFfnResidualPattern(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
    last_use: []const u32,
) ?GatedFfnResidualPattern {
    const pair_id = node_ids[node_pos];
    const pair = graph.node(pair_id);
    const pair_attrs = switch (pair.op) {
        .fused_linear_no_bias_pair => |attrs| attrs,
        else => return null,
    };
    const pair_inputs = pair.getInputs();
    if (pair_inputs.len < 3) return null;
    if (pair_attrs.rows == 0 or pair_attrs.in_dim == 0 or pair_attrs.out_dim == 0) return null;

    const norm = graph.node(pair_inputs[0]);
    const residual_id = switch (norm.op) {
        .fused_rms_norm => blk: {
            const norm_inputs = norm.getInputs();
            if (norm_inputs.len < 1) return null;
            break :blk norm_inputs[0];
        },
        else => return null,
    };

    const activation_id = findSingleInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, pair_id, &isSupportedGatedFfnActivation) orelse return null;
    const activation = activationKindForGraphNode(graph.node(activation_id)) orelse return null;
    const pair_second_id = findSingleInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, pair_id, &isPairSecondMarker) orelse return null;
    const multiply_id = findBinaryInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, activation_id, pair_second_id, &isMultiplyNode) orelse return null;
    const down_id = findSingleInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, multiply_id, &isPlainLinearNoBiasNode) orelse return null;
    const down = graph.node(down_id);
    const down_attrs = switch (down.op) {
        .fused_linear_no_bias => |attrs| attrs,
        else => return null,
    };
    if (down_attrs.num_projections != 0 or
        down_attrs.rows != pair_attrs.rows or
        down_attrs.in_dim != pair_attrs.out_dim or
        down_attrs.out_dim != pair_attrs.in_dim)
    {
        return null;
    }

    var post_down_norm_id: ?NodeId = null;
    var add_lhs_id = down_id;
    var eps: f32 = 0.0;
    if (findSingleInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, down_id, &isMatchingPostDownRmsNorm)) |norm_id| {
        const post_norm = graph.node(norm_id);
        const norm_attrs = switch (post_norm.op) {
            .fused_rms_norm => |attrs| attrs,
            else => return null,
        };
        if (norm_attrs.dim == down_attrs.out_dim) {
            post_down_norm_id = norm_id;
            add_lhs_id = norm_id;
            eps = norm_attrs.eps;
        }
    }

    const add_id = findBinaryInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, add_lhs_id, residual_id, &isAddNode) orelse return null;
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, pair_id, &.{ activation_id, pair_second_id })) return null;
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, activation_id, &.{multiply_id})) return null;
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, pair_second_id, &.{multiply_id})) return null;
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, multiply_id, &.{down_id})) return null;
    if (post_down_norm_id) |norm_id| {
        if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, down_id, &.{norm_id})) return null;
        if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, norm_id, &.{add_id})) return null;
    } else if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, down_id, &.{add_id})) return null;

    if (!nodeLastUseIs(last_use, pair_id, pair_second_id)) return null;
    if (!nodeLastUseIs(last_use, activation_id, multiply_id)) return null;
    if (!nodeLastUseIs(last_use, pair_second_id, multiply_id)) return null;
    if (!nodeLastUseIs(last_use, multiply_id, down_id)) return null;
    if (post_down_norm_id) |norm_id| {
        if (!nodeLastUseIs(last_use, down_id, norm_id)) return null;
        if (!nodeLastUseIs(last_use, norm_id, add_id)) return null;
    } else if (!nodeLastUseIs(last_use, down_id, add_id)) return null;

    return .{
        .pair_id = pair_id,
        .pair_second_id = pair_second_id,
        .activation_id = activation_id,
        .multiply_id = multiply_id,
        .down_id = down_id,
        .post_down_norm_id = post_down_norm_id,
        .add_id = add_id,
        .residual_id = residual_id,
        .activation = activation,
        .hidden_size = pair_attrs.in_dim,
        .intermediate_size = pair_attrs.out_dim,
        .rows = pair_attrs.rows,
        .eps = eps,
    };
}

fn traceGatedFfnResidualCandidate(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
    last_use: []const u32,
) void {
    const pair_id = node_ids[node_pos];
    const pair = graph.node(pair_id);
    const pair_attrs = switch (pair.op) {
        .fused_linear_no_bias_pair => |attrs| attrs,
        else => return,
    };
    const pair_inputs = pair.getInputs();
    if (pair_inputs.len < 3) {
        std.debug.print("metal_graph_fusion_trace: gated_ffn miss pair={d} reason=short_pair_inputs\n", .{pair_id});
        return;
    }
    if (pair_attrs.rows == 0 or pair_attrs.in_dim == 0 or pair_attrs.out_dim == 0) {
        std.debug.print("metal_graph_fusion_trace: gated_ffn miss pair={d} reason=zero_dims rows={d} hidden={d} intermediate={d}\n", .{
            pair_id,
            pair_attrs.rows,
            pair_attrs.in_dim,
            pair_attrs.out_dim,
        });
        return;
    }

    const norm = graph.node(pair_inputs[0]);
    const residual_id = switch (norm.op) {
        .fused_rms_norm => blk: {
            const norm_inputs = norm.getInputs();
            if (norm_inputs.len < 1) {
                std.debug.print("metal_graph_fusion_trace: gated_ffn miss pair={d} reason=short_prenorm_inputs norm={d}\n", .{ pair_id, pair_inputs[0] });
                return;
            }
            break :blk norm_inputs[0];
        },
        else => {
            std.debug.print("metal_graph_fusion_trace: gated_ffn miss pair={d} reason=input_not_prenorm input={d} input_op={s}\n", .{ pair_id, pair_inputs[0], @tagName(norm.op) });
            return;
        },
    };

    const activation_id = findSingleInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, pair_id, &isSupportedGatedFfnActivation) orelse {
        std.debug.print("metal_graph_fusion_trace: gated_ffn miss pair={d} reason=missing_activation residual={d}\n", .{ pair_id, residual_id });
        return;
    };
    const pair_second_id = findSingleInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, pair_id, &isPairSecondMarker) orelse {
        std.debug.print("metal_graph_fusion_trace: gated_ffn miss pair={d} activation={d} reason=missing_pair_second\n", .{ pair_id, activation_id });
        return;
    };
    const multiply_id = findBinaryInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, activation_id, pair_second_id, &isMultiplyNode) orelse {
        std.debug.print("metal_graph_fusion_trace: gated_ffn miss pair={d} activation={d} pair_second={d} reason=missing_multiply\n", .{ pair_id, activation_id, pair_second_id });
        return;
    };
    const down_id = findSingleInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, multiply_id, &isPlainLinearNoBiasNode) orelse {
        std.debug.print("metal_graph_fusion_trace: gated_ffn miss pair={d} multiply={d} reason=missing_down_linear\n", .{ pair_id, multiply_id });
        return;
    };
    const down = graph.node(down_id);
    const down_attrs = switch (down.op) {
        .fused_linear_no_bias => |attrs| attrs,
        else => unreachable,
    };
    if (down_attrs.num_projections != 0 or
        down_attrs.rows != pair_attrs.rows or
        down_attrs.in_dim != pair_attrs.out_dim or
        down_attrs.out_dim != pair_attrs.in_dim)
    {
        std.debug.print(
            "metal_graph_fusion_trace: gated_ffn miss pair={d} down={d} reason=down_shape_mismatch pair_rows={d} pair_in={d} pair_out={d} down_rows={d} down_in={d} down_out={d} down_proj={d}\n",
            .{
                pair_id,
                down_id,
                pair_attrs.rows,
                pair_attrs.in_dim,
                pair_attrs.out_dim,
                down_attrs.rows,
                down_attrs.in_dim,
                down_attrs.out_dim,
                down_attrs.num_projections,
            },
        );
        return;
    }

    var post_down_norm_id: ?NodeId = null;
    var add_lhs_id = down_id;
    if (findSingleInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, down_id, &isMatchingPostDownRmsNorm)) |norm_id| {
        const post_norm = graph.node(norm_id);
        const norm_attrs = switch (post_norm.op) {
            .fused_rms_norm => |attrs| attrs,
            else => unreachable,
        };
        if (norm_attrs.dim == down_attrs.out_dim) {
            post_down_norm_id = norm_id;
            add_lhs_id = norm_id;
        }
    }

    const add_id = findBinaryInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, add_lhs_id, residual_id, &isAddNode) orelse {
        std.debug.print("metal_graph_fusion_trace: gated_ffn miss pair={d} down={d} post_norm={?d} residual={d} reason=missing_residual_add\n", .{ pair_id, down_id, post_down_norm_id, residual_id });
        return;
    };

    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, pair_id, &.{ activation_id, pair_second_id })) {
        traceUnexpectedUses(graph, reachable, skipped_nodes, pair_id, &.{ activation_id, pair_second_id }, "pair_extra_use");
        return;
    }
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, activation_id, &.{multiply_id})) {
        traceUnexpectedUses(graph, reachable, skipped_nodes, activation_id, &.{multiply_id}, "activation_extra_use");
        return;
    }
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, pair_second_id, &.{multiply_id})) {
        traceUnexpectedUses(graph, reachable, skipped_nodes, pair_second_id, &.{multiply_id}, "pair_second_extra_use");
        return;
    }
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, multiply_id, &.{down_id})) {
        traceUnexpectedUses(graph, reachable, skipped_nodes, multiply_id, &.{down_id}, "multiply_extra_use");
        return;
    }
    if (post_down_norm_id) |norm_id| {
        if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, down_id, &.{norm_id})) {
            traceUnexpectedUses(graph, reachable, skipped_nodes, down_id, &.{norm_id}, "down_extra_use");
            return;
        }
        if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, norm_id, &.{add_id})) {
            traceUnexpectedUses(graph, reachable, skipped_nodes, norm_id, &.{add_id}, "norm_extra_use");
            return;
        }
    } else if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, down_id, &.{add_id})) {
        traceUnexpectedUses(graph, reachable, skipped_nodes, down_id, &.{add_id}, "down_extra_use");
        return;
    }

    if (!nodeLastUseIs(last_use, pair_id, pair_second_id)) return traceLastUseMismatch(last_use, pair_id, pair_second_id, "pair_last_use");
    if (!nodeLastUseIs(last_use, activation_id, multiply_id)) return traceLastUseMismatch(last_use, activation_id, multiply_id, "activation_last_use");
    if (!nodeLastUseIs(last_use, pair_second_id, multiply_id)) return traceLastUseMismatch(last_use, pair_second_id, multiply_id, "pair_second_last_use");
    if (!nodeLastUseIs(last_use, multiply_id, down_id)) return traceLastUseMismatch(last_use, multiply_id, down_id, "multiply_last_use");
    if (post_down_norm_id) |norm_id| {
        if (!nodeLastUseIs(last_use, down_id, norm_id)) return traceLastUseMismatch(last_use, down_id, norm_id, "down_last_use");
        if (!nodeLastUseIs(last_use, norm_id, add_id)) return traceLastUseMismatch(last_use, norm_id, add_id, "norm_last_use");
    } else if (!nodeLastUseIs(last_use, down_id, add_id)) return traceLastUseMismatch(last_use, down_id, add_id, "down_last_use");

    std.debug.print("metal_graph_fusion_trace: gated_ffn miss pair={d} reason=unknown_after_trace add={d}\n", .{ pair_id, add_id });
}

fn traceUnexpectedUses(
    graph: *const Graph,
    reachable: []const bool,
    skipped_nodes: []const bool,
    node_id: NodeId,
    expected: []const NodeId,
    reason: []const u8,
) void {
    std.debug.print("metal_graph_fusion_trace: gated_ffn miss node={d} reason={s} expected=", .{ node_id, reason });
    for (expected, 0..) |expected_id, i| {
        if (i != 0) std.debug.print(",", .{});
        std.debug.print("{d}", .{expected_id});
    }
    std.debug.print(" actual=", .{});
    var first = true;
    for (0..graph.nodeCount()) |raw_candidate| {
        const candidate_id: NodeId = @intCast(raw_candidate);
        if (raw_candidate >= reachable.len or !reachable[raw_candidate]) continue;
        if (raw_candidate < skipped_nodes.len and skipped_nodes[raw_candidate]) continue;
        const candidate = graph.node(candidate_id);
        for (candidate.getInputs()) |input_id| {
            if (input_id != node_id) continue;
            if (!first) std.debug.print(",", .{});
            first = false;
            std.debug.print("{d}:{s}", .{ candidate_id, @tagName(candidate.op) });
            break;
        }
    }
    std.debug.print("\n", .{});
}

fn traceLastUseMismatch(last_use: []const u32, node_id: NodeId, expected: NodeId, reason: []const u8) void {
    const idx: usize = @intCast(node_id);
    const actual: u32 = if (idx < last_use.len) last_use[idx] else std.math.maxInt(u32);
    std.debug.print(
        "metal_graph_fusion_trace: gated_ffn miss node={d} reason={s} expected_last_use={d} actual_last_use={d}\n",
        .{ node_id, reason, expected, actual },
    );
}

fn findSingleInputNode(
    graph: *const Graph,
    node_ids: []const NodeId,
    start_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
    input_id: NodeId,
    predicate: *const fn (*const ml.graph.Node) bool,
) ?NodeId {
    for (node_ids[start_pos..]) |candidate_id| {
        const candidate_index: usize = @intCast(candidate_id);
        if (candidate_index >= reachable.len or !reachable[candidate_index]) continue;
        if (candidate_index < skipped_nodes.len and skipped_nodes[candidate_index]) continue;
        const candidate = graph.node(candidate_id);
        if (!predicate(candidate)) continue;
        const inputs = candidate.getInputs();
        if (inputs.len >= 1 and inputs[0] == input_id) return candidate_id;
    }
    return null;
}

fn findBinaryInputNode(
    graph: *const Graph,
    node_ids: []const NodeId,
    start_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
    lhs_id: NodeId,
    rhs_id: NodeId,
    predicate: *const fn (*const ml.graph.Node) bool,
) ?NodeId {
    for (node_ids[start_pos..]) |candidate_id| {
        const candidate_index: usize = @intCast(candidate_id);
        if (candidate_index >= reachable.len or !reachable[candidate_index]) continue;
        if (candidate_index < skipped_nodes.len and skipped_nodes[candidate_index]) continue;
        const candidate = graph.node(candidate_id);
        if (!predicate(candidate)) continue;
        const inputs = candidate.getInputs();
        if (inputs.len < 2) continue;
        if ((inputs[0] == lhs_id and inputs[1] == rhs_id) or
            (inputs[0] == rhs_id and inputs[1] == lhs_id))
        {
            return candidate_id;
        }
    }
    return null;
}

fn findSingleInputNodeAsBinaryLhs(
    graph: *const Graph,
    node_ids: []const NodeId,
    start_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
    lhs_id: NodeId,
    predicate: *const fn (*const ml.graph.Node) bool,
) ?NodeId {
    for (node_ids[start_pos..]) |candidate_id| {
        const candidate_index: usize = @intCast(candidate_id);
        if (candidate_index >= reachable.len or !reachable[candidate_index]) continue;
        if (candidate_index < skipped_nodes.len and skipped_nodes[candidate_index]) continue;
        const candidate = graph.node(candidate_id);
        if (!predicate(candidate)) continue;
        const inputs = candidate.getInputs();
        if (inputs.len >= 2 and (inputs[0] == lhs_id or inputs[1] == lhs_id)) return candidate_id;
    }
    return null;
}

fn hasOnlyExpectedUses(
    graph: *const Graph,
    reachable: []const bool,
    skipped_nodes: []const bool,
    node_id: NodeId,
    expected: []const NodeId,
) bool {
    var seen: usize = 0;
    for (0..graph.nodeCount()) |raw_candidate| {
        const candidate_id: NodeId = @intCast(raw_candidate);
        if (raw_candidate >= reachable.len or !reachable[raw_candidate]) continue;
        if (raw_candidate < skipped_nodes.len and skipped_nodes[raw_candidate]) continue;
        const candidate = graph.node(candidate_id);
        var uses_candidate = false;
        for (candidate.getInputs()) |input_id| {
            if (input_id == node_id) {
                uses_candidate = true;
                break;
            }
        }
        if (!uses_candidate) continue;
        for (expected) |expected_id| {
            if (candidate_id == expected_id) {
                seen += 1;
                break;
            }
        } else return false;
    }
    return seen == expected.len;
}

fn hasExpectedReachableUseThrough(
    graph: *const Graph,
    reachable: []const bool,
    skipped_nodes: []const bool,
    node_id: NodeId,
    predicate: *const fn (*const ml.graph.Node) bool,
    passthrough: *const fn (*const ml.graph.Node) bool,
    depth: usize,
) bool {
    if (depth == 0) return false;
    for (0..graph.nodeCount()) |raw_candidate| {
        const candidate_id: NodeId = @intCast(raw_candidate);
        if (raw_candidate >= reachable.len or !reachable[raw_candidate]) continue;
        if (raw_candidate < skipped_nodes.len and skipped_nodes[raw_candidate]) continue;
        const candidate = graph.node(candidate_id);
        var uses_node = false;
        for (candidate.getInputs()) |input_id| {
            if (input_id == node_id) {
                uses_node = true;
                break;
            }
        }
        if (!uses_node) continue;
        if (predicate(candidate)) return true;
        if (passthrough(candidate) and hasExpectedReachableUseThrough(
            graph,
            reachable,
            skipped_nodes,
            candidate_id,
            predicate,
            passthrough,
            depth - 1,
        )) return true;
    }
    return false;
}

fn nodeLastUseIs(last_use: []const u32, node_id: NodeId, expected: NodeId) bool {
    const idx: usize = @intCast(node_id);
    return idx < last_use.len and last_use[idx] == @as(u32, @intCast(expected));
}

fn isSupportedGatedFfnActivation(node: *const ml.graph.Node) bool {
    return activationKindForGraphNode(node) != null;
}

fn activationKindForGraphNode(node: *const ml.graph.Node) ?ops_mod.DecoderRuntimeActivationKind {
    return switch (node.op) {
        .fused_gelu => .gelu,
        .fused_silu => .silu,
        .fused_relu => .relu,
        .fused_quick_gelu => .quick_gelu,
        else => null,
    };
}

fn isPairSecondMarker(node: *const ml.graph.Node) bool {
    return node.op == .fused_to_float32;
}

fn isMultiplyNode(node: *const ml.graph.Node) bool {
    return node.op == .mul or node.op == .fused_elem_multiply;
}

fn isAddNode(node: *const ml.graph.Node) bool {
    return node.op == .add or node.op == .fused_elem_add;
}

fn isMatchingPostDownRmsNorm(node: *const ml.graph.Node) bool {
    return node.op == .fused_rms_norm;
}

fn isPlainLinearNoBiasNode(node: *const ml.graph.Node) bool {
    return switch (node.op) {
        .fused_linear_no_bias => |attrs| attrs.num_projections == 0,
        else => false,
    };
}

fn isLinearNoBiasPairNode(node: *const ml.graph.Node) bool {
    return node.op == .fused_linear_no_bias_pair;
}

const MetalGraphRegionKind = enum {
    qkv,
    attention,
    ffn,
    ple,
    tail,
};

fn recordMetalGraphRegion(
    stats: *PartitionExecutor.ExecutionStats,
    kind: MetalGraphRegionKind,
    op_count: u64,
) void {
    stats.graph_regions += 1;
    stats.graph_region_ops += op_count;
    switch (kind) {
        .qkv => stats.metal_qkv_regions += 1,
        .attention => stats.metal_attention_regions += 1,
        .ffn => stats.metal_ffn_regions += 1,
        .ple => stats.metal_ple_regions += 1,
        .tail => stats.metal_tail_regions += 1,
    }
}

const RawLinearDotPattern = struct {
    id: NodeId,
    input_id: NodeId,
    transpose_id: NodeId,
    weight_id: NodeId,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
};

const RawLinearBiasPattern = struct {
    dot: RawLinearDotPattern,
    add_id: NodeId,
    bias_id: NodeId,
};

const RawLinearPairPattern = struct {
    first: RawLinearDotPattern,
    second: RawLinearDotPattern,
};

const RawLinearBiasPairPattern = struct {
    first: RawLinearBiasPattern,
    second: RawLinearBiasPattern,
};

const LoraLinearPattern = struct {
    add_id: NodeId,
    base_linear_id: NodeId,
    input_id: NodeId,
    dropout_mul_id: ?NodeId = null,
    dropout_mask_id: ?NodeId = null,
    lora_input_id: NodeId,
    lora_a_id: NodeId,
    lora_b_id: NodeId,
    after_a_id: NodeId,
    after_b_id: NodeId,
    scaled_id: NodeId,
    scale_id: NodeId,
    populate_scaled: bool = false,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    rank: usize,
};

const LoraLinearQkvPattern = struct {
    q: LoraLinearPattern,
    k: LoraLinearPattern,
    v: LoraLinearPattern,
    q_base_weight_id: NodeId,
    k_base_weight_id: NodeId,
    v_base_weight_id: NodeId,
    rows: usize,
    in_dim: usize,
    q_out_dim: usize,
    kv_out_dim: usize,
};

const DebertaAttentionPattern = struct {
    first_node_id: NodeId,
    output_id: NodeId,
    q_id: NodeId,
    k_id: NodeId,
    v_id: NodeId,
    q_r_id: NodeId,
    k_r_id: NodeId,
    attn_bias_id: NodeId,
    layer_index: usize,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
    hidden_size: usize,
};

const DebertaAttentionShape = struct {
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
};

const LoraBackwardPattern = struct {
    d_after_a_id: NodeId,
    grad_a_dot_id: NodeId,
    grad_a_id: NodeId,
    grad_b_dot_id: NodeId,
    grad_b_id: NodeId,
    input_transpose_id: NodeId,
    after_a_transpose_id: NodeId,
    b_transpose_id: NodeId,
    input_id: NodeId,
    after_a_id: NodeId,
    lora_b_id: NodeId,
    output_grad_id: NodeId,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    rank: usize,
};

const FfnGeluBackwardPattern = struct {
    first_dot_id: NodeId,
    second_branch_dot_id: NodeId,
    upstream_add_id: NodeId,
    gelu_backward_id: NodeId,
    output_dot_id: NodeId,
    rows: usize,
    hidden_size: usize,
    intermediate_size: usize,
};

const RuntimeDot2DInputs = struct {
    lhs: CT,
    rhs: CT,
    rhs_contract_axis: u32,
    k: usize,
};

const RuntimeDot2DRhs = struct {
    rhs: CT,
    rhs_contract_axis: u32,
    k: usize,
};

const RuntimeDot2DResolvedRhs = struct {
    rhs: CT,
    rhs_contract_axis: u32,
};

fn executeLoraLinearPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: LoraLinearPattern,
) !bool {
    const base = valueFor(values, pattern.base_linear_id) orelse return false;
    const lora_input = if (pattern.dropout_mul_id) |dropout_mul_id| blk: {
        const input = valueFor(values, pattern.input_id) orelse return false;
        const mask = valueFor(values, pattern.dropout_mask_id orelse return false) orelse return false;
        const masked = cb.multiply(input, mask) catch |err| switch (err) {
            error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => return false,
            else => return err,
        };
        values[@intCast(dropout_mul_id)] = masked;
        value_device[@intCast(dropout_mul_id)] = device_id;
        break :blk masked;
    } else valueFor(values, pattern.lora_input_id) orelse return false;
    const lora_a = valueFor(values, pattern.lora_a_id) orelse return false;
    const lora_b = valueFor(values, pattern.lora_b_id) orelse return false;
    const scale = valueFor(values, pattern.scale_id) orelse return false;
    const scale_value = scalarConstantF32(graph, pattern.scale_id);

    if (scale_value) |scale_f32| {
        if (try cb.loraLinearBranch(&.{
            .input = lora_input,
            .base = base,
            .lora_a = lora_a,
            .lora_b = lora_b,
            .rows = pattern.rows,
            .in_dim = pattern.in_dim,
            .rank = pattern.rank,
            .out_dim = pattern.out_dim,
            .scale = scale_f32,
        })) |fused| {
            values[@intCast(pattern.after_a_id)] = fused.after_a;
            value_device[@intCast(pattern.after_a_id)] = device_id;
            values[@intCast(pattern.after_b_id)] = fused.after_b;
            value_device[@intCast(pattern.after_b_id)] = device_id;
            if (pattern.populate_scaled) {
                const scaled = cb.multiply(fused.after_b, scale) catch |err| switch (err) {
                    error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
                    else => return err,
                };
                if (scaled) |scaled_tensor| {
                    values[@intCast(pattern.scaled_id)] = scaled_tensor;
                    value_device[@intCast(pattern.scaled_id)] = device_id;
                }
            }
            values[@intCast(pattern.add_id)] = fused.output;
            value_device[@intCast(pattern.add_id)] = device_id;

            if (stats) |s| {
                recordMetalGraphRegion(s, .ffn, 4);
                s.fused_graph_pattern_dispatches += 1;
                s.fused_graph_nodes_elided += 3 + @as(u64, if (pattern.dropout_mul_id != null) 1 else 0);
                recordGemmaRuntimeResidency(s, graph, pattern.add_id, isMetalResidentOrQuantizedDescriptor(cb, fused.output));
            }
            if (traceMetalGraphFusionsEnabled()) {
                std.debug.print(
                    "metal_graph_fusion_trace: lora_linear_region fused add={d} base={d} after_a={d} after_b={d} rows={d} in={d} rank={d} out={d}\n",
                    .{ pattern.add_id, pattern.base_linear_id, pattern.after_a_id, pattern.after_b_id, pattern.rows, pattern.in_dim, pattern.rank, pattern.out_dim },
                );
            }
            return true;
        }
    }

    const after_a = cb.linearNoBias(lora_input, lora_a, pattern.rows, pattern.in_dim, pattern.rank) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => return false,
        else => return err,
    };
    values[@intCast(pattern.after_a_id)] = after_a;
    value_device[@intCast(pattern.after_a_id)] = device_id;

    const after_b = cb.linearNoBias(after_a, lora_b, pattern.rows, pattern.rank, pattern.out_dim) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => return false,
        else => return err,
    };
    values[@intCast(pattern.after_b_id)] = after_b;
    value_device[@intCast(pattern.after_b_id)] = device_id;

    const output = scaled_add: {
        if (try cb.decoderRuntimeApplyScaledAddScale(&.{
            .lhs = after_b,
            .rhs = base,
            .dim = pattern.rows * pattern.out_dim,
            .lhs_scale = scale_value orelse break :scaled_add null,
            .output_scale = 1.0,
        })) |out| {
            if (pattern.populate_scaled) {
                const scaled = cb.multiply(after_b, scale) catch |err| switch (err) {
                    error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => break :scaled_add out,
                    else => return err,
                };
                values[@intCast(pattern.scaled_id)] = scaled;
                value_device[@intCast(pattern.scaled_id)] = device_id;
            }
            break :scaled_add out;
        }
        break :scaled_add null;
    } orelse fallback: {
        const scaled = cb.multiply(after_b, scale) catch |err| switch (err) {
            error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => return false,
            else => return err,
        };
        values[@intCast(pattern.scaled_id)] = scaled;
        value_device[@intCast(pattern.scaled_id)] = device_id;

        const out = cb.add(base, scaled) catch |err| switch (err) {
            error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => return false,
            else => return err,
        };
        break :fallback out;
    };
    values[@intCast(pattern.add_id)] = output;
    value_device[@intCast(pattern.add_id)] = device_id;

    if (stats) |s| {
        recordMetalGraphRegion(s, .ffn, 4);
        s.fused_graph_pattern_dispatches += 1;
        s.fused_graph_nodes_elided += 3 + @as(u64, if (pattern.dropout_mul_id != null) 1 else 0);
        recordGemmaRuntimeResidency(s, graph, pattern.add_id, isMetalResidentOrQuantizedDescriptor(cb, output));
    }
    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: lora_linear_region executed add={d} base={d} after_a={d} after_b={d} rows={d} in={d} rank={d} out={d}\n",
            .{ pattern.add_id, pattern.base_linear_id, pattern.after_a_id, pattern.after_b_id, pattern.rows, pattern.in_dim, pattern.rank, pattern.out_dim },
        );
    }
    return true;
}

fn executeLoraLinearQkvPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: LoraLinearQkvPattern,
    skipped_nodes: []bool,
) !bool {
    if (!try executeLoraLinearPattern(graph, cb, values, value_device, device_id, stats, pattern.q)) return false;
    if (!try executeLoraLinearPattern(graph, cb, values, value_device, device_id, stats, pattern.k)) return false;
    if (!try executeLoraLinearPattern(graph, cb, values, value_device, device_id, stats, pattern.v)) return false;
    markLoraLinearQkvSkipped(skipped_nodes, pattern);
    if (stats) |s| {
        s.gemma_qkv_hits += 3;
    }
    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: lora_qkv_region executed q={d} k={d} v={d} rows={d} in={d} q_out={d} kv_out={d}\n",
            .{ pattern.q.add_id, pattern.k.add_id, pattern.v.add_id, pattern.rows, pattern.in_dim, pattern.q_out_dim, pattern.kv_out_dim },
        );
    }
    return true;
}

fn executeDebertaAttentionPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    allocator: std.mem.Allocator,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: DebertaAttentionPattern,
    skipped_nodes: []bool,
) !bool {
    const q = valueFor(values, pattern.q_id) orelse {
        traceDebertaAttentionExecutionDecline("missing_q", pattern);
        return false;
    };
    const k = valueFor(values, pattern.k_id) orelse {
        traceDebertaAttentionExecutionDecline("missing_k", pattern);
        return false;
    };
    const v = valueFor(values, pattern.v_id) orelse {
        traceDebertaAttentionExecutionDecline("missing_v", pattern);
        return false;
    };
    const q_r = valueFor(values, pattern.q_r_id) orelse {
        traceDebertaAttentionExecutionDecline("missing_q_r", pattern);
        return false;
    };
    const k_r = valueFor(values, pattern.k_r_id) orelse {
        traceDebertaAttentionExecutionDecline("missing_k_r", pattern);
        return false;
    };
    const attn_bias = valueFor(values, pattern.attn_bias_id) orelse {
        traceDebertaAttentionExecutionDecline("missing_bias", pattern);
        return false;
    };

    const mask = try attentionMaskFromBias(allocator, cb, attn_bias, pattern.batch, pattern.seq_len, pattern.num_heads);
    defer allocator.free(mask);

    const output = cb.disentangledRelativeAttention(
        q,
        k,
        v,
        q_r,
        k_r,
        mask,
        pattern.batch,
        pattern.seq_len,
        pattern.num_heads,
        pattern.head_dim,
    ) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch, error.UnsupportedTensorType => {
            traceDebertaAttentionExecutionDecline(@errorName(err), pattern);
            return false;
        },
        else => return err,
    };
    errdefer cb.free(output);

    const output_shape = graph.node(pattern.output_id).output_shape;
    var output_shape_buf: [8]i32 = undefined;
    const output_rank = output_shape.rank();
    if (output_rank > output_shape_buf.len) return error.UnsupportedShape;
    for (0..output_rank) |dim_index| {
        output_shape_buf[dim_index] = @intCast(output_shape.dims[dim_index]);
    }
    const output_alias = (try cb.cloneTensorShape(output, output_shape_buf[0..output_rank])) orelse {
        traceDebertaAttentionExecutionDecline("output_alias", pattern);
        return false;
    };
    errdefer cb.free(output_alias);

    values[@intCast(pattern.first_node_id)] = output;
    value_device[@intCast(pattern.first_node_id)] = device_id;
    values[@intCast(pattern.output_id)] = output_alias;
    value_device[@intCast(pattern.output_id)] = device_id;
    markDebertaAttentionSkipped(graph, skipped_nodes, pattern);

    if (stats) |s| {
        recordMetalGraphRegion(s, .attention, 1);
        s.fused_graph_pattern_dispatches += 1;
        s.fused_graph_nodes_elided += @as(u64, pattern.output_id - pattern.first_node_id + 1);
        s.gemma_attention_matmul_hits += 1;
        recordGemmaRuntimeResidency(s, graph, pattern.output_id, isMetalResidentOrQuantizedDescriptor(cb, output));
    }
    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: deberta_attention_region executed first={d} output={d} q={d} k={d} v={d} qr={d} kr={d} bias={d} layer={d} batch={d} seq={d} heads={d} head_dim={d}\n",
            .{ pattern.first_node_id, pattern.output_id, pattern.q_id, pattern.k_id, pattern.v_id, pattern.q_r_id, pattern.k_r_id, pattern.attn_bias_id, pattern.layer_index, pattern.batch, pattern.seq_len, pattern.num_heads, pattern.head_dim },
        );
    }
    return true;
}

fn traceDebertaAttentionExecutionDecline(reason: []const u8, pattern: DebertaAttentionPattern) void {
    if (!traceMetalGraphFusionsEnabled()) return;
    std.debug.print(
        "metal_graph_fusion_trace: deberta_attention_region declined reason={s} first={d} output={d} q={d} k={d} v={d} qr={d} kr={d} bias={d} layer={d}\n",
        .{ reason, pattern.first_node_id, pattern.output_id, pattern.q_id, pattern.k_id, pattern.v_id, pattern.q_r_id, pattern.k_r_id, pattern.attn_bias_id, pattern.layer_index },
    );
}

fn attentionMaskFromBias(
    allocator: std.mem.Allocator,
    cb: *const ComputeBackend,
    attn_bias: CT,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
) ![]i64 {
    const bias = try cb.toFloat32(attn_bias, allocator);
    defer allocator.free(bias);
    if (bias.len < batch * num_heads * seq_len * seq_len) return error.InvalidAttentionShape;

    const mask = try allocator.alloc(i64, batch * seq_len);
    errdefer allocator.free(mask);
    for (0..batch) |b| {
        for (0..seq_len) |k| {
            const idx = ((b * num_heads) * seq_len + 0) * seq_len + k;
            mask[b * seq_len + k] = if (bias[idx] < -1.0e8) 0 else 1;
        }
    }
    return mask;
}

fn executeLoraBackwardPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: LoraBackwardPattern,
    skipped_nodes: []bool,
) !bool {
    const input = valueFor(values, pattern.input_id) orelse return false;
    const after_a = valueFor(values, pattern.after_a_id) orelse return false;
    const lora_b = valueFor(values, pattern.lora_b_id) orelse return false;
    const output_grad = valueFor(values, pattern.output_grad_id) orelse return false;
    const fused = (try cb.loraLinearBackward(&.{
        .input = input,
        .after_a = after_a,
        .lora_b = lora_b,
        .output_grad = output_grad,
        .rows = pattern.rows,
        .in_dim = pattern.in_dim,
        .rank = pattern.rank,
        .out_dim = pattern.out_dim,
        .scale = 1.0,
    })) orelse return false;

    values[@intCast(pattern.d_after_a_id)] = fused.grad_after_a;
    value_device[@intCast(pattern.d_after_a_id)] = device_id;
    values[@intCast(pattern.grad_a_id)] = fused.grad_a;
    value_device[@intCast(pattern.grad_a_id)] = device_id;
    values[@intCast(pattern.grad_b_id)] = fused.grad_b;
    value_device[@intCast(pattern.grad_b_id)] = device_id;
    markLoraBackwardSkipped(skipped_nodes, pattern);

    if (stats) |s| {
        recordMetalGraphRegion(s, .ffn, 5);
        s.fused_graph_pattern_dispatches += 1;
        s.fused_graph_nodes_elided += 4;
        recordGemmaRuntimeResidency(s, graph, pattern.grad_a_id, isMetalResidentOrQuantizedDescriptor(cb, fused.grad_a));
        recordGemmaRuntimeResidency(s, graph, pattern.grad_b_id, isMetalResidentOrQuantizedDescriptor(cb, fused.grad_b));
    }
    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: lora_backward_region executed d_after_a={d} grad_a={d} grad_b={d} input={d} after_a={d} lora_b={d} rows={d} in={d} rank={d} out={d}\n",
            .{ pattern.d_after_a_id, pattern.grad_a_id, pattern.grad_b_id, pattern.input_id, pattern.after_a_id, pattern.lora_b_id, pattern.rows, pattern.in_dim, pattern.rank, pattern.out_dim },
        );
    }
    return true;
}

fn executeFfnGeluBackwardPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    stats: ?*PartitionExecutor.ExecutionStats,
    partition_plan: ?*const partition_mod.PartitionPlan,
    pattern: FfnGeluBackwardPattern,
    skipped_nodes: []bool,
) !bool {
    if (try executeFfnGeluBackwardOutputPattern(graph, cb, values, value_device, device_id, stats, pattern, skipped_nodes)) {
        return true;
    }
    if (try executeFfnGeluBackwardChainPattern(graph, cb, values, value_device, device_id, stats, pattern, skipped_nodes)) {
        return true;
    }

    const first_dot_node = graph.node(pattern.first_dot_id);
    const first_attrs = switch (first_dot_node.op) {
        .dot_general => |attrs| attrs,
        else => return false,
    };
    const first = (try executeRuntimeDotGeneral(graph, cb, values, first_dot_node.getInputs(), first_attrs, operatorPlanForRegionNode(partition_plan, pattern.first_dot_id))) orelse return false;

    values[@intCast(pattern.first_dot_id)] = first;
    value_device[@intCast(pattern.first_dot_id)] = device_id;

    const second_branch_node = graph.node(pattern.second_branch_dot_id);
    const second_branch_attrs = switch (second_branch_node.op) {
        .dot_general => |attrs| attrs,
        else => return false,
    };
    const second_branch = (try executeRuntimeDotGeneral(graph, cb, values, second_branch_node.getInputs(), second_branch_attrs, operatorPlanForRegionNode(partition_plan, pattern.second_branch_dot_id))) orelse {
        values[@intCast(pattern.first_dot_id)] = null;
        cb.free(first);
        return false;
    };
    values[@intCast(pattern.second_branch_dot_id)] = second_branch;
    value_device[@intCast(pattern.second_branch_dot_id)] = device_id;

    const add_node = graph.node(pattern.upstream_add_id);
    const upstream = (try executeRuntimeAdd(graph, cb, values, add_node.getInputs(), add_node.output_shape)) orelse {
        values[@intCast(pattern.second_branch_dot_id)] = null;
        values[@intCast(pattern.first_dot_id)] = null;
        cb.free(second_branch);
        cb.free(first);
        return false;
    };
    values[@intCast(pattern.upstream_add_id)] = upstream;
    value_device[@intCast(pattern.upstream_add_id)] = device_id;

    const gelu_node = graph.node(pattern.gelu_backward_id);
    const gelu = (try executeRuntimeGeluBackward(cb, values, gelu_node.getInputs(), gelu_node.output_shape)) orelse {
        values[@intCast(pattern.upstream_add_id)] = null;
        values[@intCast(pattern.second_branch_dot_id)] = null;
        values[@intCast(pattern.first_dot_id)] = null;
        cb.free(upstream);
        cb.free(second_branch);
        cb.free(first);
        return false;
    };
    values[@intCast(pattern.gelu_backward_id)] = gelu;
    value_device[@intCast(pattern.gelu_backward_id)] = device_id;

    const output_dot_node = graph.node(pattern.output_dot_id);
    const output_attrs = switch (output_dot_node.op) {
        .dot_general => |attrs| attrs,
        else => return false,
    };
    const output = (try executeRuntimeDotGeneral(graph, cb, values, output_dot_node.getInputs(), output_attrs, operatorPlanForRegionNode(partition_plan, pattern.output_dot_id))) orelse {
        values[@intCast(pattern.gelu_backward_id)] = null;
        values[@intCast(pattern.upstream_add_id)] = null;
        values[@intCast(pattern.second_branch_dot_id)] = null;
        values[@intCast(pattern.first_dot_id)] = null;
        cb.free(gelu);
        cb.free(upstream);
        cb.free(second_branch);
        cb.free(first);
        return false;
    };

    values[@intCast(pattern.output_dot_id)] = output;
    value_device[@intCast(pattern.output_dot_id)] = device_id;
    markFfnGeluBackwardSkipped(skipped_nodes, pattern);

    if (stats) |s| {
        recordMetalGraphRegion(s, .ffn, 5);
        s.fused_graph_pattern_dispatches += 1;
        s.fused_graph_nodes_elided += 4;
        recordGemmaRuntimeResidency(s, graph, pattern.output_dot_id, isMetalResidentOrQuantizedDescriptor(cb, output));
    }
    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: ffn_gelu_backward_region executed first={d} second_branch={d} add={d} gelu={d} output={d} rows={d} hidden={d} intermediate={d}\n",
            .{ pattern.first_dot_id, pattern.second_branch_dot_id, pattern.upstream_add_id, pattern.gelu_backward_id, pattern.output_dot_id, pattern.rows, pattern.hidden_size, pattern.intermediate_size },
        );
    }
    return true;
}

fn executeFfnGeluBackwardOutputPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: FfnGeluBackwardPattern,
    skipped_nodes: []bool,
) !bool {
    if (platform.env.getenvBoolDefault("TERMITE_METAL_DISABLE_FFN_GELU_BACKWARD_OUTPUT_CHAIN", false)) {
        traceFfnGeluBackwardOutputDecline("disabled", pattern, 0, 0);
        return false;
    }

    const first_dot_node = graph.node(pattern.first_dot_id);
    const first_attrs = switch (first_dot_node.op) {
        .dot_general => |attrs| attrs,
        else => {
            traceFfnGeluBackwardOutputDecline("first_not_dot", pattern, 0, 0);
            return false;
        },
    };
    const first = (try runtimeDot2DInputs(graph, values, first_dot_node.getInputs(), first_attrs)) orelse {
        traceFfnGeluBackwardDot2DDecline(graph, "first_inputs", pattern.first_dot_id, first_dot_node.getInputs(), first_attrs);
        traceFfnGeluBackwardOutputDecline("first_inputs", pattern, 0, 0);
        return false;
    };
    if (first.k != 1) {
        traceFfnGeluBackwardOutputDecline("first_k", pattern, first.k, 0);
        return false;
    }

    const second_branch_node = graph.node(pattern.second_branch_dot_id);
    const second_branch_attrs = switch (second_branch_node.op) {
        .dot_general => |attrs| attrs,
        else => {
            traceFfnGeluBackwardOutputDecline("second_not_dot", pattern, first.k, 0);
            return false;
        },
    };
    const second_branch = (try runtimeDot2DInputs(graph, values, second_branch_node.getInputs(), second_branch_attrs)) orelse {
        traceFfnGeluBackwardDot2DDecline(graph, "second_inputs", pattern.second_branch_dot_id, second_branch_node.getInputs(), second_branch_attrs);
        traceFfnGeluBackwardOutputDecline("second_inputs", pattern, first.k, 0);
        return false;
    };

    const gelu_node = graph.node(pattern.gelu_backward_id);
    if (gelu_node.num_inputs < 2) {
        traceFfnGeluBackwardOutputDecline("gelu_arity", pattern, first.k, second_branch.k);
        return false;
    }
    const gelu_input = valueFor(values, gelu_node.inputs[0]) orelse {
        traceFfnGeluBackwardOutputDecline("gelu_input", pattern, first.k, second_branch.k);
        return false;
    };

    const output_dot_node = graph.node(pattern.output_dot_id);
    const output_attrs = switch (output_dot_node.op) {
        .dot_general => |attrs| attrs,
        else => {
            traceFfnGeluBackwardOutputDecline("output_not_dot", pattern, first.k, second_branch.k);
            return false;
        },
    };
    const output_dot = (try runtimeDot2DRhs(graph, values, output_dot_node.getInputs(), output_attrs)) orelse {
        traceFfnGeluBackwardDot2DDecline(graph, "output_rhs", pattern.output_dot_id, output_dot_node.getInputs(), output_attrs);
        traceFfnGeluBackwardOutputDecline("output_rhs", pattern, first.k, second_branch.k);
        return false;
    };

    const result = cb.decoderRuntimeFfnGeluBackwardOutput(&.{
        .first_lhs = first.lhs,
        .first_rhs = first.rhs,
        .first_rhs_contract_axis = first.rhs_contract_axis,
        .second_lhs = second_branch.lhs,
        .second_rhs = second_branch.rhs,
        .second_rhs_contract_axis = second_branch.rhs_contract_axis,
        .gelu_input = gelu_input,
        .output_rhs = output_dot.rhs,
        .output_rhs_contract_axis = output_dot.rhs_contract_axis,
        .rows = pattern.rows,
        .hidden_size = pattern.hidden_size,
        .intermediate_size = pattern.intermediate_size,
        .first_k = first.k,
        .second_k = second_branch.k,
    }) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    } orelse {
        traceFfnGeluBackwardOutputDecline("runtime_null", pattern, first.k, second_branch.k);
        return false;
    };

    values[@intCast(pattern.first_dot_id)] = result.first;
    values[@intCast(pattern.gelu_backward_id)] = result.gelu;
    values[@intCast(pattern.output_dot_id)] = result.output;
    value_device[@intCast(pattern.first_dot_id)] = device_id;
    value_device[@intCast(pattern.gelu_backward_id)] = device_id;
    value_device[@intCast(pattern.output_dot_id)] = device_id;
    markFfnGeluBackwardSkipped(skipped_nodes, pattern);

    if (stats) |s| {
        recordMetalGraphRegion(s, .ffn, 5);
        s.fused_graph_pattern_dispatches += 1;
        s.fused_graph_nodes_elided += 4;
        recordGemmaRuntimeResidency(s, graph, pattern.output_dot_id, isMetalResidentOrQuantizedDescriptor(cb, result.output));
    }
    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: ffn_gelu_backward_output_chain executed first={d} second_branch={d} add={d} gelu={d} output={d} rows={d} hidden={d} intermediate={d}\n",
            .{ pattern.first_dot_id, pattern.second_branch_dot_id, pattern.upstream_add_id, pattern.gelu_backward_id, pattern.output_dot_id, pattern.rows, pattern.hidden_size, pattern.intermediate_size },
        );
    }
    return true;
}

fn traceFfnGeluBackwardOutputDecline(reason: []const u8, pattern: FfnGeluBackwardPattern, first_k: usize, second_k: usize) void {
    if (!traceMetalGraphFusionsEnabled()) return;
    std.debug.print(
        "metal_graph_fusion_trace: ffn_gelu_backward_output_chain declined reason={s} first={d} second_branch={d} add={d} gelu={d} output={d} rows={d} hidden={d} intermediate={d} first_k={d} second_k={d}\n",
        .{ reason, pattern.first_dot_id, pattern.second_branch_dot_id, pattern.upstream_add_id, pattern.gelu_backward_id, pattern.output_dot_id, pattern.rows, pattern.hidden_size, pattern.intermediate_size, first_k, second_k },
    );
}

fn traceFfnGeluBackwardDot2DDecline(
    graph: *const Graph,
    reason: []const u8,
    node_id: NodeId,
    inputs: []const NodeId,
    attrs: anytype,
) void {
    if (!traceMetalGraphFusionsEnabled()) return;
    const lhs_id = if (inputs.len > 0) inputs[0] else null_node;
    const rhs_id = if (inputs.len > 1) inputs[1] else null_node;
    std.debug.print(
        "metal_graph_fusion_trace: ffn_gelu_backward_dot2d declined reason={s} node={d} inputs={d} lhs={d} rhs={d} num_contracting={d} num_batch={d} lhs_contract0={d} rhs_contract0={d} lhs_rank={d} lhs0={d} lhs1={d} rhs_rank={d} rhs0={d} rhs1={d}\n",
        .{
            reason,
            node_id,
            inputs.len,
            lhs_id,
            rhs_id,
            attrs.num_contracting,
            attrs.num_batch,
            if (attrs.num_contracting > 0) attrs.lhs_contracting[0] else 255,
            if (attrs.num_contracting > 0) attrs.rhs_contracting[0] else 255,
            shapeRankForNodeOr(graph, lhs_id, 0),
            shapeDimForNodeOr(graph, lhs_id, 0, -1),
            shapeDimForNodeOr(graph, lhs_id, 1, -1),
            shapeRankForNodeOr(graph, rhs_id, 0),
            shapeDimForNodeOr(graph, rhs_id, 0, -1),
            shapeDimForNodeOr(graph, rhs_id, 1, -1),
        },
    );
}

fn operatorPlanForRegionNode(partition_plan: ?*const partition_mod.PartitionPlan, node_id: NodeId) ?OperatorPlan {
    const plan = partition_plan orelse return null;
    return plan.operatorPlanForNode(node_id);
}

fn executeFfnGeluBackwardChainPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: FfnGeluBackwardPattern,
    skipped_nodes: []bool,
) !bool {
    if (!platform.env.getenvBoolDefault("TERMITE_METAL_ENABLE_FFN_GELU_BACKWARD_CHAIN", false)) return false;

    const first_dot_node = graph.node(pattern.first_dot_id);
    const first_attrs = switch (first_dot_node.op) {
        .dot_general => |attrs| attrs,
        else => return false,
    };
    const first = (try runtimeDot2DInputs(graph, values, first_dot_node.getInputs(), first_attrs)) orelse return false;

    const second_branch_node = graph.node(pattern.second_branch_dot_id);
    const second_branch_attrs = switch (second_branch_node.op) {
        .dot_general => |attrs| attrs,
        else => return false,
    };
    const second_branch = (try runtimeDot2DInputs(graph, values, second_branch_node.getInputs(), second_branch_attrs)) orelse return false;

    const gelu_node = graph.node(pattern.gelu_backward_id);
    if (gelu_node.num_inputs < 2) return false;
    const gelu_input = valueFor(values, gelu_node.inputs[0]) orelse return false;

    const output_dot_node = graph.node(pattern.output_dot_id);
    const output_attrs = switch (output_dot_node.op) {
        .dot_general => |attrs| attrs,
        else => return false,
    };
    const output_dot = (try runtimeDot2DRhs(graph, values, output_dot_node.getInputs(), output_attrs)) orelse return false;

    const chain = cb.decoderRuntimeFfnGeluBackwardChain(&.{
        .first_lhs = first.lhs,
        .first_rhs = first.rhs,
        .first_rhs_contract_axis = first.rhs_contract_axis,
        .second_lhs = second_branch.lhs,
        .second_rhs = second_branch.rhs,
        .second_rhs_contract_axis = second_branch.rhs_contract_axis,
        .gelu_input = gelu_input,
        .output_rhs = output_dot.rhs,
        .output_rhs_contract_axis = output_dot.rhs_contract_axis,
        .rows = pattern.rows,
        .hidden_size = pattern.hidden_size,
        .intermediate_size = pattern.intermediate_size,
        .first_k = first.k,
        .second_k = second_branch.k,
    }) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    } orelse return false;

    values[@intCast(pattern.first_dot_id)] = chain.first;
    values[@intCast(pattern.second_branch_dot_id)] = chain.second_branch;
    values[@intCast(pattern.upstream_add_id)] = chain.upstream;
    values[@intCast(pattern.gelu_backward_id)] = chain.gelu;
    values[@intCast(pattern.output_dot_id)] = chain.output;
    value_device[@intCast(pattern.first_dot_id)] = device_id;
    value_device[@intCast(pattern.second_branch_dot_id)] = device_id;
    value_device[@intCast(pattern.upstream_add_id)] = device_id;
    value_device[@intCast(pattern.gelu_backward_id)] = device_id;
    value_device[@intCast(pattern.output_dot_id)] = device_id;
    markFfnGeluBackwardSkipped(skipped_nodes, pattern);

    if (stats) |s| {
        recordMetalGraphRegion(s, .ffn, 5);
        s.fused_graph_pattern_dispatches += 1;
        s.fused_graph_nodes_elided += 4;
        recordGemmaRuntimeResidency(s, graph, pattern.output_dot_id, isMetalResidentOrQuantizedDescriptor(cb, chain.output));
    }
    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: ffn_gelu_backward_chain executed first={d} second_branch={d} add={d} gelu={d} output={d} rows={d} hidden={d} intermediate={d}\n",
            .{ pattern.first_dot_id, pattern.second_branch_dot_id, pattern.upstream_add_id, pattern.gelu_backward_id, pattern.output_dot_id, pattern.rows, pattern.hidden_size, pattern.intermediate_size },
        );
    }
    return true;
}

fn runtimeDot2DInputs(
    graph: *const Graph,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
) !?RuntimeDot2DInputs {
    if (inputs.len < 2) return null;
    if (attrs.num_contracting != 1 or attrs.num_batch != 0) return null;
    const lhs_contracting = attrs.lhs_contracting[0];
    const rhs_contracting = attrs.rhs_contracting[0];
    if (lhs_contracting != 1 or (rhs_contracting != 0 and rhs_contracting != 1)) return null;

    var lhs_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    var rhs_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    const lhs_shape = try fillShapeDims(graph.node(inputs[0]).output_shape, &lhs_shape_buf);
    const rhs_shape = try fillShapeDims(graph.node(inputs[1]).output_shape, &rhs_shape_buf);
    if (lhs_shape.len != 2 or rhs_shape.len != 2) return null;
    if (lhs_shape[0] <= 0 or lhs_shape[1] <= 0) return null;
    const rhs_axis: usize = @intCast(rhs_contracting);
    const rhs_k = rhs_shape[rhs_axis];
    if (rhs_k <= 0 or rhs_k != lhs_shape[1]) return null;
    const lhs = valueFor(values, inputs[0]) orelse return null;
    const rhs = resolvedRuntimeDot2DRhs(graph, values, inputs[1], rhs_contracting) orelse return null;
    return .{
        .lhs = lhs,
        .rhs = rhs.rhs,
        .rhs_contract_axis = rhs.rhs_contract_axis,
        .k = @intCast(lhs_shape[1]),
    };
}

fn runtimeDot2DRhs(
    graph: *const Graph,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
) !?RuntimeDot2DRhs {
    if (inputs.len < 2) return null;
    if (attrs.num_contracting != 1 or attrs.num_batch != 0) return null;
    const lhs_contracting = attrs.lhs_contracting[0];
    const rhs_contracting = attrs.rhs_contracting[0];
    if (lhs_contracting != 1 or (rhs_contracting != 0 and rhs_contracting != 1)) return null;

    var lhs_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    var rhs_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    const lhs_shape = try fillShapeDims(graph.node(inputs[0]).output_shape, &lhs_shape_buf);
    const rhs_shape = try fillShapeDims(graph.node(inputs[1]).output_shape, &rhs_shape_buf);
    if (lhs_shape.len != 2 or rhs_shape.len != 2) return null;
    if (lhs_shape[0] <= 0 or lhs_shape[1] <= 0) return null;
    const rhs_axis: usize = @intCast(rhs_contracting);
    const rhs_k = rhs_shape[rhs_axis];
    if (rhs_k <= 0 or rhs_k != lhs_shape[1]) return null;
    const rhs = resolvedRuntimeDot2DRhs(graph, values, inputs[1], rhs_contracting) orelse return null;
    return .{
        .rhs = rhs.rhs,
        .rhs_contract_axis = rhs.rhs_contract_axis,
        .k = @intCast(lhs_shape[1]),
    };
}

fn resolvedRuntimeDot2DRhs(
    graph: *const Graph,
    values: []?CT,
    rhs_id: NodeId,
    rhs_contracting: u32,
) ?RuntimeDot2DResolvedRhs {
    if (valueFor(values, rhs_id)) |rhs| {
        return .{ .rhs = rhs, .rhs_contract_axis = rhs_contracting };
    }
    if (rhs_id == null_node or rhs_id >= graph.nodeCount()) return null;
    const rhs_node = graph.node(rhs_id);
    const transpose_attrs = switch (rhs_node.op) {
        .transpose => |attrs| attrs,
        else => return null,
    };
    if (rhs_node.num_inputs == 0 or rhs_node.inputs[0] == null_node) return null;
    const source_id = rhs_node.inputs[0];
    if (source_id >= graph.nodeCount()) return null;
    if (!transposeIsSimple2D(transpose_attrs, graph.node(source_id).output_shape)) return null;
    const source = valueFor(values, source_id) orelse return null;
    const source_contract_axis: u32 = switch (rhs_contracting) {
        0 => 1,
        1 => 0,
        else => return null,
    };
    return .{ .rhs = source, .rhs_contract_axis = source_contract_axis };
}

fn matchFfnGeluBackwardPattern(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?FfnGeluBackwardPattern {
    const first_dot_id = node_ids[node_pos];
    if (!isReachableUnskippedNode(reachable, skipped_nodes, first_dot_id)) return null;
    const first_dot = graph.node(first_dot_id);
    const first_attrs = switch (first_dot.op) {
        .dot_general => |attrs| attrs,
        else => return null,
    };
    if (!isLinearDotAttrs(first_attrs) or first_dot.num_inputs < 2) return null;

    const first_shape = first_dot.output_shape;
    if (first_shape.rank() != 2) return null;
    const rows = shapeDimUsize(first_shape, 0) orelse return null;
    const intermediate_size = shapeDimUsize(first_shape, 1) orelse return null;
    if (rows <= 1 or intermediate_size < 512) return null;

    const upstream_add_id = singleAddConsumerForInput(graph, first_dot_id, reachable, skipped_nodes) orelse return null;
    const upstream_add = graph.node(upstream_add_id);
    if (!upstream_add.output_shape.eq(first_shape) or upstream_add.num_inputs < 2) return null;
    const second_branch_dot_id = if (upstream_add.inputs[0] == first_dot_id)
        upstream_add.inputs[1]
    else if (upstream_add.inputs[1] == first_dot_id)
        upstream_add.inputs[0]
    else
        return null;
    if (second_branch_dot_id == null_node or !isReachableUnskippedNode(reachable, skipped_nodes, second_branch_dot_id)) return null;
    const second_branch_dot = graph.node(second_branch_dot_id);
    const second_branch_attrs = switch (second_branch_dot.op) {
        .dot_general => |attrs| attrs,
        else => return null,
    };
    if (!isLinearDotAttrs(second_branch_attrs) or !second_branch_dot.output_shape.eq(first_shape)) return null;

    const gelu_backward_id = singleFfnGeluBackwardConsumer(graph, upstream_add_id, reachable, skipped_nodes) orelse return null;
    const gelu_backward = graph.node(gelu_backward_id);
    if (!gelu_backward.output_shape.eq(first_shape)) return null;

    const output_dot_id = singleLinearDotConsumerForInput(graph, gelu_backward_id, reachable, skipped_nodes) orelse return null;
    const output_dot = graph.node(output_dot_id);
    const output_attrs = switch (output_dot.op) {
        .dot_general => |attrs| attrs,
        else => return null,
    };
    if (!isLinearDotAttrs(output_attrs) or output_dot.num_inputs < 2) return null;
    if (output_dot.inputs[0] != gelu_backward_id) return null;
    const output_shape = output_dot.output_shape;
    if (output_shape.rank() != 2) return null;
    if ((shapeDimUsize(output_shape, 0) orelse return null) != rows) return null;
    const hidden_size = shapeDimUsize(output_shape, 1) orelse return null;
    if (hidden_size == 0 or hidden_size >= intermediate_size) return null;

    const rhs_id = output_dot.inputs[1];
    if (rhs_id == null_node) return null;
    if (!linearDotConsumesTranspose(graph, output_dot_id, rhs_id)) return null;
    const rhs_source_id = graph.node(rhs_id).inputs[0];
    if (rhs_source_id == null_node) return null;
    const rhs_source_shape = graph.node(rhs_source_id).output_shape;
    if ((shapeDimUsize(rhs_source_shape, 0) orelse return null) != hidden_size) return null;
    if ((shapeDimUsize(rhs_source_shape, 1) orelse return null) != intermediate_size) return null;

    return .{
        .first_dot_id = first_dot_id,
        .second_branch_dot_id = second_branch_dot_id,
        .upstream_add_id = upstream_add_id,
        .gelu_backward_id = gelu_backward_id,
        .output_dot_id = output_dot_id,
        .rows = rows,
        .hidden_size = hidden_size,
        .intermediate_size = intermediate_size,
    };
}

fn markFfnGeluBackwardSkipped(skipped: []bool, pattern: FfnGeluBackwardPattern) void {
    markSkipped(skipped, pattern.second_branch_dot_id);
    markSkipped(skipped, pattern.upstream_add_id);
    markSkipped(skipped, pattern.gelu_backward_id);
    markSkipped(skipped, pattern.output_dot_id);
}

fn isReachableUnskippedNode(reachable: []const bool, skipped_nodes: []const bool, node_id: NodeId) bool {
    const index: usize = @intCast(node_id);
    if (index >= reachable.len or !reachable[index]) return false;
    if (index < skipped_nodes.len and skipped_nodes[index]) return false;
    return true;
}

fn singleFfnGeluBackwardConsumer(
    graph: *const Graph,
    producer_id: NodeId,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?NodeId {
    var found: ?NodeId = null;
    var node_id: NodeId = 0;
    while (node_id < graph.nodeCount()) : (node_id += 1) {
        if (!isReachableUnskippedNode(reachable, skipped_nodes, node_id)) continue;
        const node = graph.node(node_id);
        switch (node.op) {
            .fused_gelu_backward => {},
            else => continue,
        }
        if (node.num_inputs < 2 or node.inputs[1] != producer_id) continue;
        if (found != null) return null;
        found = node_id;
    }
    return found;
}

fn singleAddConsumerForInput(
    graph: *const Graph,
    producer_id: NodeId,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?NodeId {
    var found: ?NodeId = null;
    var node_id: NodeId = 0;
    while (node_id < graph.nodeCount()) : (node_id += 1) {
        if (!isReachableUnskippedNode(reachable, skipped_nodes, node_id)) continue;
        const node = graph.node(node_id);
        switch (node.op) {
            .add => {},
            else => continue,
        }
        if (node.num_inputs < 2 or (node.inputs[0] != producer_id and node.inputs[1] != producer_id)) continue;
        if (found != null) return null;
        found = node_id;
    }
    return found;
}

fn singleLinearDotConsumerForInput(
    graph: *const Graph,
    producer_id: NodeId,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?NodeId {
    var found: ?NodeId = null;
    var node_id: NodeId = 0;
    while (node_id < graph.nodeCount()) : (node_id += 1) {
        if (!isReachableUnskippedNode(reachable, skipped_nodes, node_id)) continue;
        const node = graph.node(node_id);
        const attrs = switch (node.op) {
            .dot_general => |dot_attrs| dot_attrs,
            else => continue,
        };
        if (!isLinearDotAttrs(attrs) or node.num_inputs < 2 or node.inputs[0] != producer_id) continue;
        if (found != null) return null;
        found = node_id;
    }
    return found;
}

fn matchLoraBackwardPattern(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?LoraBackwardPattern {
    const d_after_a_id = node_ids[node_pos];
    const d_after_a = graph.node(d_after_a_id);
    const d_after_attrs = switch (d_after_a.op) {
        .dot_general => |attrs| attrs,
        else => return null,
    };
    if (!isLinearDotAttrs(d_after_attrs) or d_after_a.num_inputs < 2) return null;
    const output_grad_id = d_after_a.inputs[0];
    const b_transpose_id = d_after_a.inputs[1];
    if (output_grad_id == null_node or b_transpose_id == null_node) return null;
    const lora_b_id = loraBParameterFromBackwardTranspose(graph, b_transpose_id) orelse return null;

    const output_grad_shape = graph.node(output_grad_id).output_shape;
    const d_after_shape = d_after_a.output_shape;
    const lora_b_shape = graph.node(lora_b_id).output_shape;
    if (output_grad_shape.rank() != 2 or d_after_shape.rank() != 2 or lora_b_shape.rank() != 2) return null;
    const rows = shapeDimUsize(output_grad_shape, 0) orelse return null;
    const out_dim = shapeDimUsize(output_grad_shape, 1) orelse return null;
    const rank = shapeDimUsize(d_after_shape, 1) orelse return null;
    if (shapeDimUsize(d_after_shape, 0) != rows) return null;
    if (shapeDimUsize(lora_b_shape, 0) != out_dim or shapeDimUsize(lora_b_shape, 1) != rank) return null;

    const grad_b_match = findLoraBackwardGradB(graph, reachable, skipped_nodes, d_after_a_id, output_grad_id, rows, out_dim, rank) orelse return null;
    const grad_a_match = findLoraBackwardGradA(graph, reachable, skipped_nodes, d_after_a_id, rows, rank) orelse return null;
    const input_shape = graph.node(grad_a_match.input_id).output_shape;
    const after_a_shape = graph.node(grad_b_match.after_a_id).output_shape;
    if (input_shape.rank() != 2 or after_a_shape.rank() != 2) return null;
    const in_dim = shapeDimUsize(input_shape, 1) orelse return null;
    if (shapeDimUsize(input_shape, 0) != rows) return null;
    if (shapeDimUsize(after_a_shape, 0) != rows or shapeDimUsize(after_a_shape, 1) != rank) return null;
    if (shapeDimUsize(graph.node(grad_a_match.grad_a_id).output_shape, 0) != rank) return null;
    if (shapeDimUsize(graph.node(grad_a_match.grad_a_id).output_shape, 1) != in_dim) return null;

    return .{
        .d_after_a_id = d_after_a_id,
        .grad_a_dot_id = grad_a_match.dot_id,
        .grad_a_id = grad_a_match.grad_a_id,
        .grad_b_dot_id = grad_b_match.dot_id,
        .grad_b_id = grad_b_match.grad_b_id,
        .input_transpose_id = grad_a_match.input_transpose_id,
        .after_a_transpose_id = grad_b_match.after_a_transpose_id,
        .b_transpose_id = b_transpose_id,
        .input_id = grad_a_match.input_id,
        .after_a_id = grad_b_match.after_a_id,
        .lora_b_id = lora_b_id,
        .output_grad_id = output_grad_id,
        .rows = rows,
        .in_dim = in_dim,
        .out_dim = out_dim,
        .rank = rank,
    };
}

const LoraBackwardGradBMatch = struct {
    dot_id: NodeId,
    grad_b_id: NodeId,
    after_a_transpose_id: NodeId,
    after_a_id: NodeId,
};

const LoraBackwardGradAMatch = struct {
    dot_id: NodeId,
    grad_a_id: NodeId,
    input_transpose_id: NodeId,
    input_id: NodeId,
};

fn findLoraBackwardGradB(
    graph: *const Graph,
    reachable: []const bool,
    skipped_nodes: []const bool,
    d_after_a_id: NodeId,
    output_grad_id: NodeId,
    rows: usize,
    out_dim: usize,
    rank: usize,
) ?LoraBackwardGradBMatch {
    _ = d_after_a_id;
    var dot_id: NodeId = 0;
    while (dot_id < graph.nodeCount()) : (dot_id += 1) {
        const dot_index: usize = @intCast(dot_id);
        if (dot_index >= reachable.len or !reachable[dot_index]) continue;
        if (dot_index < skipped_nodes.len and skipped_nodes[dot_index]) continue;
        const dot = graph.node(dot_id);
        const attrs = switch (dot.op) {
            .dot_general => |a| a,
            else => continue,
        };
        if (!isLinearDotAttrs(attrs) or dot.num_inputs < 2 or dot.inputs[1] != output_grad_id) continue;
        if (shapeDimUsize(dot.output_shape, 0) != rank or shapeDimUsize(dot.output_shape, 1) != out_dim) continue;
        const after_a_id = sourceFromSimpleTranspose(graph, dot.inputs[0]) orelse continue;
        const after_a_shape = graph.node(after_a_id).output_shape;
        if (shapeDimUsize(after_a_shape, 0) != rows or shapeDimUsize(after_a_shape, 1) != rank) continue;
        const grad_b_id = findSimpleTransposeConsumer(graph, reachable, skipped_nodes, dot_id, out_dim, rank) orelse continue;
        return .{
            .dot_id = dot_id,
            .grad_b_id = grad_b_id,
            .after_a_transpose_id = dot.inputs[0],
            .after_a_id = after_a_id,
        };
    }
    return null;
}

fn findLoraBackwardGradA(
    graph: *const Graph,
    reachable: []const bool,
    skipped_nodes: []const bool,
    d_after_a_id: NodeId,
    rows: usize,
    rank: usize,
) ?LoraBackwardGradAMatch {
    var dot_id: NodeId = 0;
    while (dot_id < graph.nodeCount()) : (dot_id += 1) {
        const dot_index: usize = @intCast(dot_id);
        if (dot_index >= reachable.len or !reachable[dot_index]) continue;
        if (dot_index < skipped_nodes.len and skipped_nodes[dot_index]) continue;
        const dot = graph.node(dot_id);
        const attrs = switch (dot.op) {
            .dot_general => |a| a,
            else => continue,
        };
        if (!isLinearDotAttrs(attrs) or dot.num_inputs < 2 or dot.inputs[1] != d_after_a_id) continue;
        if (shapeDimUsize(dot.output_shape, 1) != rank) continue;
        const input_id = sourceFromSimpleTranspose(graph, dot.inputs[0]) orelse continue;
        const input_shape = graph.node(input_id).output_shape;
        const in_dim = shapeDimUsize(input_shape, 1) orelse continue;
        if (shapeDimUsize(input_shape, 0) != rows) continue;
        if (shapeDimUsize(dot.output_shape, 0) != in_dim) continue;
        const grad_a_id = findSimpleTransposeConsumer(graph, reachable, skipped_nodes, dot_id, rank, in_dim) orelse continue;
        return .{
            .dot_id = dot_id,
            .grad_a_id = grad_a_id,
            .input_transpose_id = dot.inputs[0],
            .input_id = input_id,
        };
    }
    return null;
}

fn findSimpleTransposeConsumer(
    graph: *const Graph,
    reachable: []const bool,
    skipped_nodes: []const bool,
    input_id: NodeId,
    rows: usize,
    cols: usize,
) ?NodeId {
    var consumer_id: NodeId = 0;
    while (consumer_id < graph.nodeCount()) : (consumer_id += 1) {
        const consumer_index: usize = @intCast(consumer_id);
        if (consumer_index >= reachable.len or !reachable[consumer_index]) continue;
        if (consumer_index < skipped_nodes.len and skipped_nodes[consumer_index]) continue;
        const consumer = graph.node(consumer_id);
        const attrs = switch (consumer.op) {
            .transpose => |a| a,
            else => continue,
        };
        if (consumer.num_inputs == 0 or consumer.inputs[0] != input_id) continue;
        if (!transposeIsSimple2D(attrs, graph.node(input_id).output_shape)) continue;
        if (shapeDimUsize(consumer.output_shape, 0) != rows or shapeDimUsize(consumer.output_shape, 1) != cols) continue;
        return consumer_id;
    }
    return null;
}

fn sourceFromSimpleTranspose(graph: *const Graph, node_id: NodeId) ?NodeId {
    if (node_id == null_node or node_id >= graph.nodeCount()) return null;
    const node = graph.node(node_id);
    const attrs = switch (node.op) {
        .transpose => |a| a,
        else => return null,
    };
    if (node.num_inputs == 0 or node.inputs[0] == null_node) return null;
    if (!transposeIsSimple2D(attrs, graph.node(node.inputs[0]).output_shape)) return null;
    return node.inputs[0];
}

fn loraBParameterFromBackwardTranspose(graph: *const Graph, node_id: NodeId) ?NodeId {
    const source_id = sourceFromSimpleTranspose(graph, node_id) orelse return null;
    if (isLoraParameter(graph, source_id, ".lora_B")) return source_id;
    const param_id = sourceFromSimpleTranspose(graph, source_id) orelse return null;
    if (isLoraParameter(graph, param_id, ".lora_B")) return param_id;
    return null;
}

fn markLoraBackwardSkipped(skipped: []bool, pattern: LoraBackwardPattern) void {
    const ids = [_]NodeId{
        pattern.grad_a_dot_id,
        pattern.grad_a_id,
        pattern.grad_b_dot_id,
        pattern.grad_b_id,
        pattern.input_transpose_id,
        pattern.after_a_transpose_id,
    };
    for (ids) |node_id| {
        const index: usize = @intCast(node_id);
        if (index < skipped.len) skipped[index] = true;
    }
}

fn matchLoraLinearQkvPattern(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?LoraLinearQkvPattern {
    const current = matchLoraLinearPattern(graph, node_ids, node_pos, reachable, skipped_nodes) orelse return null;
    const current_weight_name = loraBaseWeightParameterName(graph, current) orelse {
        traceLoraQkvMatchDecline("no_base_weight", current, null);
        return null;
    };
    traceLoraQkvMatchCandidate("candidate", current, current_weight_name);
    const q = if (isQueryProjectionWeightName(current_weight_name))
        current
    else
        findLoraQkvSibling(graph, node_ids, 0, reachable, skipped_nodes, current, &isQueryProjectionWeightName) orelse {
            traceLoraQkvMatchDecline("missing_q", current, current_weight_name);
            return null;
        };
    const k = if (isKeyProjectionWeightName(current_weight_name))
        current
    else
        findLoraQkvSibling(graph, node_ids, 0, reachable, skipped_nodes, current, &isKeyProjectionWeightName) orelse {
            traceLoraQkvMatchDecline("missing_k", current, current_weight_name);
            return null;
        };
    const v = if (isValueProjectionWeightName(current_weight_name))
        current
    else
        findLoraQkvSibling(graph, node_ids, 0, reachable, skipped_nodes, current, &isValueProjectionWeightName) orelse {
            traceLoraQkvMatchDecline("missing_v", current, current_weight_name);
            return null;
        };
    const q_pos = findNodePos(node_ids, q.add_id) orelse return null;
    const k_pos = findNodePos(node_ids, k.add_id) orelse return null;
    const v_pos = findNodePos(node_ids, v.add_id) orelse return null;
    if (node_pos != @min(q_pos, @min(k_pos, v_pos))) {
        traceLoraQkvMatchDecline("not_earliest", current, current_weight_name);
        return null;
    }
    const q_base_weight_id = loraBaseWeightId(graph, q) orelse return null;
    const k_base_weight_id = loraBaseWeightId(graph, k) orelse return null;
    const v_base_weight_id = loraBaseWeightId(graph, v) orelse return null;
    if (k.out_dim != v.out_dim) {
        traceLoraQkvMatchDecline("kv_dim_mismatch", current, current_weight_name);
        return null;
    }
    if (traceLoraQkvMatchingEnabled()) {
        std.debug.print(
            "lora_qkv_match: matched q={d} k={d} v={d} q_pos={d} k_pos={d} v_pos={d} rows={d} in={d} q_out={d} kv_out={d}\n",
            .{ q.add_id, k.add_id, v.add_id, q_pos, k_pos, v_pos, q.rows, q.in_dim, q.out_dim, k.out_dim },
        );
    }
    return .{
        .q = q,
        .k = k,
        .v = v,
        .q_base_weight_id = q_base_weight_id,
        .k_base_weight_id = k_base_weight_id,
        .v_base_weight_id = v_base_weight_id,
        .rows = q.rows,
        .in_dim = q.in_dim,
        .q_out_dim = q.out_dim,
        .kv_out_dim = k.out_dim,
    };
}

fn matchDebertaAttentionPattern(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
    regions: []const RuntimeRegion,
) ?DebertaAttentionPattern {
    const first_node_id = node_ids[node_pos];
    const first_node = graph.node(first_node_id);
    const first_shape = first_node.output_shape;
    if (std.meta.activeTag(first_node.op) != .reshape) return null;
    if (previousDebertaAttentionRegionCovers(regions, node_pos, first_node_id)) {
        traceDebertaAttentionMatchDecline("overlapping_region", first_node_id, first_shape, null, null);
        return null;
    }
    if (first_shape.rank() != 4) {
        traceDebertaAttentionMatchDecline("rank", first_node_id, first_shape, null, null);
        return null;
    }
    const qkv = previousLoraQkvRegion(regions, node_pos) orelse {
        traceDebertaAttentionMatchDecline("missing_qkv", first_node_id, first_shape, null, null);
        return null;
    };
    const shape = parseDebertaAttentionStartShape(first_shape, qkv.rows, qkv.q_out_dim) orelse {
        traceDebertaAttentionMatchDecline("shape", first_node_id, first_shape, qkv.q.add_id, null);
        return null;
    };
    traceDebertaAttentionMatchCandidate(first_node_id, first_shape, shape.num_heads, shape.seq_len, shape.seq_len, shape.head_dim);
    const hidden_size = shape.num_heads * shape.head_dim;
    const rel_len = shape.seq_len * 2 - 1;

    if (qkv.rows != shape.seq_len or qkv.q_out_dim != hidden_size or qkv.kv_out_dim != hidden_size) {
        traceDebertaAttentionMatchDecline("qkv_shape", first_node_id, first_shape, qkv.q.add_id, null);
        return null;
    }
    const layer_index = layerIndexForWeight(graph, qkv.q_base_weight_id) orelse {
        traceDebertaAttentionMatchDecline("qkv_layer", first_node_id, first_shape, qkv.q.add_id, null);
        return null;
    };

    const q_r_id = findCompactDebertaRelativeProjection(graph, node_ids, node_pos, reachable, layer_index, rel_len, hidden_size, &isQueryProjectionWeightName) orelse {
        traceDebertaAttentionMatchDecline("missing_q_r", first_node_id, first_shape, qkv.q.add_id, null);
        return null;
    };
    const k_r_id = findCompactDebertaRelativeProjection(graph, node_ids, node_pos, reachable, layer_index, rel_len, hidden_size, &isKeyProjectionWeightName) orelse {
        traceDebertaAttentionMatchDecline("missing_k_r", first_node_id, first_shape, qkv.q.add_id, q_r_id);
        return null;
    };
    if (q_r_id == k_r_id) {
        traceDebertaAttentionMatchDecline("same_relative_projection", first_node_id, first_shape, q_r_id, k_r_id);
        return null;
    }

    const attn_bias_id = findDebertaAttentionBiasInput(graph, node_ids, node_pos, reachable, skipped_nodes, shape.num_heads, shape.seq_len) orelse {
        traceDebertaAttentionMatchDecline("missing_bias", first_node_id, first_shape, q_r_id, k_r_id);
        return null;
    };
    const output_id = findDebertaAttentionMergedOutput(graph, node_ids, node_pos, reachable, skipped_nodes, shape.seq_len, hidden_size) orelse {
        traceDebertaAttentionMatchDecline("missing_output", first_node_id, first_shape, attn_bias_id, null);
        return null;
    };
    if (output_id <= first_node_id) {
        traceDebertaAttentionMatchDecline("output_before_start", first_node_id, first_shape, output_id, null);
        return null;
    }

    return .{
        .first_node_id = first_node_id,
        .output_id = output_id,
        .q_id = qkv.q.add_id,
        .k_id = qkv.k.add_id,
        .v_id = qkv.v.add_id,
        .q_r_id = q_r_id,
        .k_r_id = k_r_id,
        .attn_bias_id = attn_bias_id,
        .layer_index = layer_index,
        .batch = shape.batch,
        .seq_len = shape.seq_len,
        .num_heads = shape.num_heads,
        .head_dim = shape.head_dim,
        .hidden_size = hidden_size,
    };
}

fn parseDebertaAttentionStartShape(first_shape: Shape, qkv_rows: usize, hidden_size: usize) ?DebertaAttentionShape {
    if (first_shape.rank() != 4 or qkv_rows == 0 or hidden_size == 0) return null;
    const d0 = shapeDimUsize(first_shape, 0) orelse return null;
    const d1 = shapeDimUsize(first_shape, 1) orelse return null;
    const d2 = shapeDimUsize(first_shape, 2) orelse return null;
    const d3 = shapeDimUsize(first_shape, 3) orelse return null;
    if (d3 == 0 or hidden_size % d3 != 0) return null;
    const heads = hidden_size / d3;
    if (heads == 0) return null;

    if (d0 == heads and d1 == qkv_rows and d2 == qkv_rows) {
        return .{ .batch = 1, .seq_len = qkv_rows, .num_heads = heads, .head_dim = d3 };
    }
    if (d0 == 1 and d1 == qkv_rows and d2 == heads) {
        return .{ .batch = 1, .seq_len = qkv_rows, .num_heads = heads, .head_dim = d3 };
    }
    if (d0 == heads and d1 == 1 and d2 == qkv_rows) {
        return .{ .batch = 1, .seq_len = qkv_rows, .num_heads = heads, .head_dim = d3 };
    }
    if (d0 == 1 and d1 == heads and d2 == qkv_rows) {
        return .{ .batch = 1, .seq_len = qkv_rows, .num_heads = heads, .head_dim = d3 };
    }
    return null;
}

fn previousDebertaAttentionRegionCovers(regions: []const RuntimeRegion, node_pos: usize, node_id: NodeId) bool {
    var pos = node_pos;
    while (pos > 0) {
        pos -= 1;
        switch (regions[pos]) {
            .deberta_attention => |pattern| {
                if (node_id >= pattern.first_node_id and node_id <= pattern.output_id) return true;
            },
            else => {},
        }
    }
    return false;
}

fn previousLoraQkvRegion(regions: []const RuntimeRegion, node_pos: usize) ?LoraLinearQkvPattern {
    var pos = node_pos;
    while (pos > 0) {
        pos -= 1;
        switch (regions[pos]) {
            .lora_linear_qkv => |pattern| return pattern,
            else => {},
        }
    }
    return null;
}

fn findCompactDebertaRelativeProjection(
    graph: *const Graph,
    node_ids: []const NodeId,
    end_pos: usize,
    reachable: []const bool,
    layer_index: usize,
    rel_len: usize,
    hidden_size: usize,
    weight_name_predicate: *const fn ([]const u8) bool,
) ?NodeId {
    var pos = end_pos;
    while (pos > 0) {
        pos -= 1;
        const node_id = node_ids[pos];
        const node_index: usize = @intCast(node_id);
        if (node_index >= reachable.len or !reachable[node_index]) continue;
        const node = graph.node(node_id);
        const shape = node.output_shape;
        if (shape.rank() != 2) continue;
        if (shapeDimUsize(shape, 0) != rel_len or shapeDimUsize(shape, 1) != hidden_size) continue;
        const weight_id = projectionWeightParameterFromNode(graph, node_id, 8) orelse continue;
        const weight = graph.node(weight_id);
        if (std.meta.activeTag(weight.op) != .parameter) continue;
        if (layerIndexForWeight(graph, weight_id) != layer_index) continue;
        const weight_name = graph.parameterName(weight);
        if (!weight_name_predicate(weight_name)) continue;
        return node_id;
    }
    return null;
}

fn findDebertaAttentionBiasInput(
    graph: *const Graph,
    node_ids: []const NodeId,
    start_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
    num_heads: usize,
    seq_len: usize,
) ?NodeId {
    for (node_ids[start_pos..], start_pos..) |node_id, pos| {
        if (pos > start_pos + 160) break;
        const node_index: usize = @intCast(node_id);
        if (node_index >= reachable.len or !reachable[node_index]) continue;
        if (node_index < skipped_nodes.len and skipped_nodes[node_index]) continue;
        const node = graph.node(node_id);
        switch (node.op) {
            .add, .fused_elem_add => {},
            else => continue,
        }
        const shape = node.output_shape;
        if (shape.rank() != 3) continue;
        if (shapeDimUsize(shape, 0) != num_heads or shapeDimUsize(shape, 1) != seq_len or shapeDimUsize(shape, 2) != seq_len) continue;
        for (node.getInputs()) |input_id| {
            if (input_id == null_node or input_id >= graph.nodeCount()) continue;
            const input = graph.node(input_id);
            if (std.meta.activeTag(input.op) != .parameter) continue;
            const input_shape = input.output_shape;
            if (input_shape.rank() != 3) continue;
            if (shapeDimUsize(input_shape, 0) != num_heads or shapeDimUsize(input_shape, 1) != seq_len or shapeDimUsize(input_shape, 2) != seq_len) continue;
            if (std.mem.indexOf(u8, graph.parameterName(input), "attn_bias") == null) continue;
            return input_id;
        }
    }
    return findDebertaAttentionBiasParameter(graph, reachable, num_heads, seq_len);
}

fn findDebertaAttentionBiasParameter(
    graph: *const Graph,
    reachable: []const bool,
    num_heads: usize,
    seq_len: usize,
) ?NodeId {
    var node_id: NodeId = 0;
    while (node_id < graph.nodeCount()) : (node_id += 1) {
        const node_index: usize = @intCast(node_id);
        if (node_index >= reachable.len or !reachable[node_index]) continue;
        const node = graph.node(node_id);
        if (std.meta.activeTag(node.op) != .parameter) continue;
        const shape = node.output_shape;
        if (shape.rank() != 3) continue;
        if (shapeDimUsize(shape, 0) != num_heads or shapeDimUsize(shape, 1) != seq_len or shapeDimUsize(shape, 2) != seq_len) continue;
        if (std.mem.indexOf(u8, graph.parameterName(node), "attn_bias") == null) continue;
        return node_id;
    }
    return null;
}

fn findDebertaAttentionMergedOutput(
    graph: *const Graph,
    node_ids: []const NodeId,
    start_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
    seq_len: usize,
    hidden_size: usize,
) ?NodeId {
    for (node_ids[start_pos..], start_pos..) |node_id, pos| {
        if (pos > start_pos + 180) break;
        const node_index: usize = @intCast(node_id);
        if (node_index >= reachable.len or !reachable[node_index]) continue;
        if (node_index < skipped_nodes.len and skipped_nodes[node_index]) continue;
        const pattern = matchLoraLinearPattern(graph, node_ids, pos, reachable, skipped_nodes) orelse continue;
        const weight_name = loraBaseWeightParameterName(graph, pattern) orelse {
            traceDebertaAttentionOutputCandidate("no_weight", node_id, pattern, null);
            continue;
        };
        traceDebertaAttentionOutputCandidate("candidate", node_id, pattern, weight_name);
        if (!isDebertaAttentionOutputDenseName(weight_name)) continue;
        if (pattern.rows != seq_len or pattern.in_dim != hidden_size or pattern.out_dim != hidden_size) {
            traceDebertaAttentionOutputCandidate("shape_mismatch", node_id, pattern, weight_name);
            return null;
        }
        return pattern.input_id;
    }
    return null;
}

fn traceLoraQkvMatchCandidate(label: []const u8, pattern: LoraLinearPattern, weight_name: []const u8) void {
    if (!traceLoraQkvMatchingEnabled()) return;
    std.debug.print(
        "lora_qkv_match: {s} add={d} base={d} input={d} weight={s} rows={d} in={d} out={d} rank={d}\n",
        .{ label, pattern.add_id, pattern.base_linear_id, pattern.input_id, weight_name, pattern.rows, pattern.in_dim, pattern.out_dim, pattern.rank },
    );
}

fn traceLoraQkvMatchDecline(reason: []const u8, pattern: LoraLinearPattern, weight_name: ?[]const u8) void {
    if (!traceLoraQkvMatchingEnabled()) return;
    std.debug.print(
        "lora_qkv_match: declined reason={s} add={d} base={d} input={d} weight={s} rows={d} in={d} out={d} rank={d}\n",
        .{ reason, pattern.add_id, pattern.base_linear_id, pattern.input_id, weight_name orelse "none", pattern.rows, pattern.in_dim, pattern.out_dim, pattern.rank },
    );
}

fn traceDebertaAttentionMatchCandidate(node_id: NodeId, shape: ml.graph.Shape, dim0: usize, dim1: usize, dim2: usize, dim3: usize) void {
    if (!traceDebertaAttentionMatchingEnabled()) return;
    std.debug.print(
        "deberta_attention_match: candidate node={d} rank={d} dims={d},{d},{d},{d}\n",
        .{ node_id, shape.rank(), dim0, dim1, dim2, dim3 },
    );
}

fn traceDebertaAttentionMatchDecline(reason: []const u8, node_id: NodeId, shape: ml.graph.Shape, id0: ?NodeId, id1: ?NodeId) void {
    if (!traceDebertaAttentionMatchingEnabled()) return;
    std.debug.print(
        "deberta_attention_match: declined reason={s} node={d} rank={d} id0={d} id1={d}\n",
        .{ reason, node_id, shape.rank(), id0 orelse null_node, id1 orelse null_node },
    );
}

fn traceDebertaAttentionOutputCandidate(label: []const u8, node_id: NodeId, pattern: LoraLinearPattern, weight_name: ?[]const u8) void {
    if (!traceDebertaAttentionMatchingEnabled()) return;
    std.debug.print(
        "deberta_attention_output_match: {s} node={d} add={d} input={d} base={d} weight={s} rows={d} in={d} out={d}\n",
        .{ label, node_id, pattern.add_id, pattern.input_id, pattern.base_linear_id, weight_name orelse "none", pattern.rows, pattern.in_dim, pattern.out_dim },
    );
}

fn findLoraQkvSibling(
    graph: *const Graph,
    node_ids: []const NodeId,
    start_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
    q: LoraLinearPattern,
    weight_name_predicate: *const fn ([]const u8) bool,
) ?LoraLinearPattern {
    for (node_ids[start_pos..], start_pos..) |candidate_id, candidate_pos| {
        const candidate_index: usize = @intCast(candidate_id);
        if (candidate_index >= reachable.len or !reachable[candidate_index]) continue;
        if (candidate_index < skipped_nodes.len and skipped_nodes[candidate_index]) continue;
        const candidate = matchLoraLinearPattern(graph, node_ids, candidate_pos, reachable, skipped_nodes) orelse continue;
        if (candidate.input_id != q.input_id or candidate.rows != q.rows or candidate.in_dim != q.in_dim) continue;
        const weight_name = loraBaseWeightParameterName(graph, candidate) orelse continue;
        if (!weight_name_predicate(weight_name)) continue;
        return candidate;
    }
    return null;
}

fn loraBaseWeightId(graph: *const Graph, pattern: LoraLinearPattern) ?NodeId {
    if (projectionWeightParameterFromNode(graph, pattern.base_linear_id, 6)) |weight_id| return weight_id;
    const base = graph.node(pattern.base_linear_id);
    if (base.num_inputs < 2) return null;
    const weight_id = base.inputs[1];
    if (weight_id == null_node) return null;
    if (graph.node(weight_id).op == .parameter) return weight_id;
    return sourceParameterFromSimpleTranspose(graph, weight_id);
}

fn projectionWeightParameterFromNode(graph: *const Graph, node_id: NodeId, depth: usize) ?NodeId {
    if (node_id == null_node or node_id >= graph.nodeCount() or depth == 0) return null;
    const node = graph.node(node_id);
    switch (node.op) {
        .parameter => {
            const name = graph.parameterName(node);
            if (isQueryProjectionWeightName(name) or isKeyProjectionWeightName(name) or isValueProjectionWeightName(name)) return node_id;
            return null;
        },
        .transpose => {
            if (node.num_inputs < 1) return null;
            return projectionWeightParameterFromNode(graph, node.inputs[0], depth - 1);
        },
        .dot_general, .fused_linear, .fused_linear_no_bias => {
            if (node.num_inputs >= 2) {
                if (projectionWeightParameterFromNode(graph, node.inputs[1], depth - 1)) |weight_id| return weight_id;
            }
            if (node.num_inputs >= 1) return projectionWeightParameterFromNode(graph, node.inputs[0], depth - 1);
            return null;
        },
        .add, .fused_elem_add => {
            for (node.getInputs()) |input_id| {
                if (projectionWeightParameterFromNode(graph, input_id, depth - 1)) |weight_id| return weight_id;
            }
            return null;
        },
        else => return null,
    }
}

fn loraBaseWeightParameterName(graph: *const Graph, pattern: LoraLinearPattern) ?[]const u8 {
    const weight_id = loraBaseWeightId(graph, pattern) orelse return null;
    const weight = graph.node(weight_id);
    if (weight.op != .parameter) return null;
    return graph.parameterName(weight);
}

fn matchLoraLinearPattern(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?LoraLinearPattern {
    const add_id = node_ids[node_pos];
    const add_index: usize = @intCast(add_id);
    if (add_index < skipped_nodes.len and skipped_nodes[add_index]) return null;
    const add = graph.node(add_id);
    switch (add.op) {
        .add, .fused_elem_add => {},
        else => return null,
    }
    if (add.num_inputs < 2) return null;
    return matchLoraLinearPatternWithOrder(graph, add_id, add.inputs[0], add.inputs[1], reachable, skipped_nodes) orelse
        matchLoraLinearPatternWithOrder(graph, add_id, add.inputs[1], add.inputs[0], reachable, skipped_nodes);
}

fn matchLoraLinearPatternWithOrder(
    graph: *const Graph,
    add_id: NodeId,
    base_linear_id: NodeId,
    scaled_id: NodeId,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?LoraLinearPattern {
    if (matchLoweredLoraLinearPatternWithOrder(graph, add_id, base_linear_id, scaled_id, reachable, skipped_nodes)) |pattern| {
        return pattern;
    }
    if (base_linear_id == null_node or scaled_id == null_node) return null;
    const scaled_index: usize = @intCast(scaled_id);
    if (scaled_index >= reachable.len or !reachable[scaled_index]) return null;
    if (scaled_index < skipped_nodes.len and skipped_nodes[scaled_index]) return null;
    const base = graph.node(base_linear_id);
    const base_attrs = switch (base.op) {
        .fused_linear => |attrs| attrs,
        .fused_linear_no_bias => |attrs| attrs,
        else => return null,
    };
    if (base.num_inputs < 2 or base.inputs[0] == null_node) return null;

    const scaled = graph.node(scaled_id);
    switch (scaled.op) {
        .mul, .fused_elem_multiply => {},
        else => return null,
    }
    if (scaled.num_inputs < 2) return null;

    const after_b_id, const scale_id = blk: {
        if (isScalarF32Node(graph, scaled.inputs[0])) break :blk .{ scaled.inputs[1], scaled.inputs[0] };
        if (isScalarF32Node(graph, scaled.inputs[1])) break :blk .{ scaled.inputs[0], scaled.inputs[1] };
        return null;
    };
    if (after_b_id == null_node or scale_id == null_node) return null;

    const after_b = graph.node(after_b_id);
    const after_b_attrs = switch (after_b.op) {
        .fused_linear_no_bias => |attrs| attrs,
        else => return null,
    };
    if (after_b.num_inputs < 2) return null;
    const after_a_id = after_b.inputs[0];
    const lora_b_id = after_b.inputs[1];
    if (after_a_id == null_node or lora_b_id == null_node) return null;
    if (!isLoraParameter(graph, lora_b_id, ".lora_B")) return null;

    const after_a = graph.node(after_a_id);
    const after_a_attrs = switch (after_a.op) {
        .fused_linear_no_bias => |attrs| attrs,
        else => return null,
    };
    if (after_a.num_inputs < 2) return null;
    const lora_input_id = after_a.inputs[0];
    const lora_a_id = after_a.inputs[1];
    if (lora_input_id == null_node or lora_a_id == null_node) return null;
    if (!isLoraParameter(graph, lora_a_id, ".lora_A")) return null;

    var dropout_mask_id: ?NodeId = null;
    const dropout_mul_id: ?NodeId = if (lora_input_id != base.inputs[0]) blk: {
        const maybe_mul = graph.node(lora_input_id);
        switch (maybe_mul.op) {
            .mul, .fused_elem_multiply => {},
            else => return null,
        }
        if (maybe_mul.num_inputs < 2) return null;
        const matches_base_input =
            maybe_mul.inputs[0] == base.inputs[0] or maybe_mul.inputs[1] == base.inputs[0];
        if (!matches_base_input) return null;
        const other = if (maybe_mul.inputs[0] == base.inputs[0]) maybe_mul.inputs[1] else maybe_mul.inputs[0];
        if (!isLoraParameter(graph, other, ".lora_dropout_mask")) return null;
        dropout_mask_id = other;
        break :blk lora_input_id;
    } else null;

    const rows: usize = @intCast(base_attrs.rows);
    const in_dim: usize = @intCast(base_attrs.in_dim);
    const out_dim: usize = @intCast(base_attrs.out_dim);
    const rank: usize = @intCast(after_a_attrs.out_dim);
    if (rows == 0 or in_dim == 0 or out_dim == 0 or rank == 0) return null;
    if (after_a_attrs.rows != base_attrs.rows or after_a_attrs.in_dim != base_attrs.in_dim) return null;
    if (after_b_attrs.rows != base_attrs.rows or after_b_attrs.in_dim != after_a_attrs.out_dim or after_b_attrs.out_dim != base_attrs.out_dim) return null;
    const a_shape = graph.node(lora_a_id).output_shape;
    const b_shape = graph.node(lora_b_id).output_shape;
    if (shapeDimUsize(a_shape, 0) != rank or shapeDimUsize(a_shape, 1) != in_dim) return null;
    if (shapeDimUsize(b_shape, 0) != out_dim or shapeDimUsize(b_shape, 1) != rank) return null;

    return .{
        .add_id = add_id,
        .base_linear_id = base_linear_id,
        .input_id = base.inputs[0],
        .dropout_mul_id = dropout_mul_id,
        .dropout_mask_id = dropout_mask_id,
        .lora_input_id = lora_input_id,
        .lora_a_id = lora_a_id,
        .lora_b_id = lora_b_id,
        .after_a_id = after_a_id,
        .after_b_id = after_b_id,
        .scaled_id = scaled_id,
        .scale_id = scale_id,
        .populate_scaled = reachableUseCount(graph, scaled_id, reachable, 2) > 1,
        .rows = rows,
        .in_dim = in_dim,
        .out_dim = out_dim,
        .rank = rank,
    };
}

fn matchLoweredLoraLinearPatternWithOrder(
    graph: *const Graph,
    add_id: NodeId,
    base_id: NodeId,
    scaled_id: NodeId,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?LoraLinearPattern {
    if (base_id == null_node or scaled_id == null_node) return null;
    const scaled_index: usize = @intCast(scaled_id);
    if (scaled_index >= reachable.len or !reachable[scaled_index]) return null;
    if (scaled_index < skipped_nodes.len and skipped_nodes[scaled_index]) return null;
    const scaled = graph.node(scaled_id);
    switch (scaled.op) {
        .mul, .fused_elem_multiply => {},
        else => return null,
    }
    if (scaled.num_inputs < 2) return null;

    const after_b_id, const scale_id = blk: {
        if (isScalarF32Node(graph, scaled.inputs[0])) break :blk .{ scaled.inputs[1], scaled.inputs[0] };
        if (isScalarF32Node(graph, scaled.inputs[1])) break :blk .{ scaled.inputs[0], scaled.inputs[1] };
        return null;
    };
    if (after_b_id == null_node or scale_id == null_node) return null;
    const after_b = graph.node(after_b_id);
    const after_b_attrs = switch (after_b.op) {
        .dot_general => |attrs| attrs,
        else => return null,
    };
    if (!isLinearDotAttrs(after_b_attrs)) return null;
    if (after_b.num_inputs < 2) return null;
    const after_a_id = after_b.inputs[0];
    const lora_b_id = sourceParameterFromSimpleTranspose(graph, after_b.inputs[1]) orelse return null;
    if (!isLoraParameter(graph, lora_b_id, ".lora_B")) return null;

    const after_a = graph.node(after_a_id);
    const after_a_attrs = switch (after_a.op) {
        .dot_general => |attrs| attrs,
        else => return null,
    };
    if (!isLinearDotAttrs(after_a_attrs)) return null;
    if (after_a.num_inputs < 2) return null;
    const lora_input_id = after_a.inputs[0];
    const lora_a_id = sourceParameterFromSimpleTranspose(graph, after_a.inputs[1]) orelse return null;
    if (!isLoraParameter(graph, lora_a_id, ".lora_A")) return null;

    const lora_input_shape = graph.node(lora_input_id).output_shape;
    const after_a_shape = after_a.output_shape;
    const after_b_shape = after_b.output_shape;
    const base_shape = graph.node(base_id).output_shape;
    if (lora_input_shape.rank() != 2 or after_a_shape.rank() != 2 or after_b_shape.rank() != 2 or base_shape.rank() != 2) return null;
    if (!shapesEqual(after_b_shape, base_shape) or !shapesEqual(after_b_shape, graph.node(add_id).output_shape)) return null;
    const rows = shapeDimUsize(lora_input_shape, 0) orelse return null;
    const in_dim = shapeDimUsize(lora_input_shape, 1) orelse return null;
    const rank = shapeDimUsize(after_a_shape, 1) orelse return null;
    const out_dim = shapeDimUsize(after_b_shape, 1) orelse return null;
    if (shapeDimUsize(after_a_shape, 0) != rows or shapeDimUsize(after_b_shape, 0) != rows) return null;
    const a_shape = graph.node(lora_a_id).output_shape;
    const b_shape = graph.node(lora_b_id).output_shape;
    if (shapeDimUsize(a_shape, 0) != rank or shapeDimUsize(a_shape, 1) != in_dim) return null;
    if (shapeDimUsize(b_shape, 0) != out_dim or shapeDimUsize(b_shape, 1) != rank) return null;

    return .{
        .add_id = add_id,
        .base_linear_id = base_id,
        .input_id = lora_input_id,
        .lora_input_id = lora_input_id,
        .lora_a_id = lora_a_id,
        .lora_b_id = lora_b_id,
        .after_a_id = after_a_id,
        .after_b_id = after_b_id,
        .scaled_id = scaled_id,
        .scale_id = scale_id,
        .populate_scaled = reachableUseCount(graph, scaled_id, reachable, 2) > 1,
        .rows = rows,
        .in_dim = in_dim,
        .out_dim = out_dim,
        .rank = rank,
    };
}

fn isLinearDotAttrs(attrs: anytype) bool {
    return attrs.num_contracting == 1 and
        attrs.num_batch == 0 and
        attrs.lhs_contracting[0] == 1 and
        attrs.rhs_contracting[0] == 0;
}

fn sourceParameterFromSimpleTranspose(graph: *const Graph, node_id: NodeId) ?NodeId {
    if (node_id == null_node or node_id >= graph.nodeCount()) return null;
    const node = graph.node(node_id);
    const attrs = switch (node.op) {
        .transpose => |transpose_attrs| transpose_attrs,
        else => return null,
    };
    if (node.num_inputs == 0 or node.inputs[0] == null_node) return null;
    if (!transposeIsSimple2D(attrs, graph.node(node.inputs[0]).output_shape)) return null;
    const source_id = node.inputs[0];
    if (std.meta.activeTag(graph.node(source_id).op) != .parameter) return null;
    return source_id;
}

fn isScalarF32Node(graph: *const Graph, node_id: NodeId) bool {
    if (node_id == null_node or node_id >= graph.nodeCount()) return false;
    return graph.node(node_id).output_shape.dtype == .f32 and graph.node(node_id).output_shape.rank() == 0;
}

fn isLoraParameter(graph: *const Graph, node_id: NodeId, needle: []const u8) bool {
    if (node_id == null_node or node_id >= graph.nodeCount()) return false;
    const node = graph.node(node_id);
    if (std.meta.activeTag(node.op) != .parameter) return false;
    return std.mem.indexOf(u8, graph.parameterName(node), needle) != null;
}

fn executeRawLinearDotPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: RawLinearDotPattern,
    prepared: PreparedLinearRegion,
) !bool {
    const input = valueFor(values, pattern.input_id) orelse return false;
    const output = (try cb.decoderRuntimeApplyLinear(&.{
        .slot = prepared.linear_slot,
        .input = input,
        .in_dim = pattern.in_dim,
        .out_dim = pattern.out_dim,
    })) orelse return false;

    values[@intCast(pattern.id)] = output;
    value_device[@intCast(pattern.id)] = device_id;
    if (stats) |s| {
        recordMetalGraphRegion(s, .qkv, 1);
        s.fused_graph_pattern_dispatches += 1;
        s.fused_graph_nodes_elided += 1;
        recordGemmaRuntimeResidency(s, graph, pattern.id, isMetalResidentOrQuantizedDescriptor(cb, output));
    }
    if (traceMetalGraphFusionsEnabled()) {
        const weight_name = graph.parameterName(graph.node(pattern.weight_id));
        std.debug.print(
            "metal_graph_fusion_trace: raw_linear_dot executed dot={d} transpose={d} weight={s} rows={d} in={d} out={d}\n",
            .{ pattern.id, pattern.transpose_id, weight_name, pattern.rows, pattern.in_dim, pattern.out_dim },
        );
    }
    return true;
}

fn executeRawLinearPairPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: RawLinearPairPattern,
    prepared: PreparedLinearPairRegion,
    skipped_nodes: []bool,
) !bool {
    const input = valueFor(values, pattern.first.input_id) orelse return false;
    const pair = (try cb.decoderRuntimeApplyLinearPair(&.{
        .slot_a = prepared.first_slot,
        .slot_b = prepared.second_slot,
        .input = input,
        .in_dim = pattern.first.in_dim,
        .out_dim = pattern.first.out_dim,
    })) orelse return false;

    values[@intCast(pattern.first.id)] = pair.first;
    value_device[@intCast(pattern.first.id)] = device_id;
    values[@intCast(pattern.second.id)] = pair.second;
    value_device[@intCast(pattern.second.id)] = device_id;
    const second_index: usize = @intCast(pattern.second.id);
    if (second_index < skipped_nodes.len) skipped_nodes[second_index] = true;

    if (stats) |s| {
        recordMetalGraphRegion(s, .qkv, 2);
        s.fused_graph_pattern_dispatches += 1;
        s.fused_graph_nodes_elided += 1;
        recordGemmaRuntimeResidency(s, graph, pattern.first.id, isMetalResidentOrQuantizedDescriptor(cb, pair.first));
        recordGemmaRuntimeResidency(s, graph, pattern.second.id, isMetalResidentOrQuantizedDescriptor(cb, pair.second));
    }
    if (traceMetalGraphFusionsEnabled()) {
        const first_weight_name = graph.parameterName(graph.node(pattern.first.weight_id));
        const second_weight_name = graph.parameterName(graph.node(pattern.second.weight_id));
        std.debug.print(
            "metal_graph_fusion_trace: raw_linear_pair executed first={d} second={d} first_weight={s} second_weight={s} rows={d} in={d} out={d}\n",
            .{ pattern.first.id, pattern.second.id, first_weight_name, second_weight_name, pattern.first.rows, pattern.first.in_dim, pattern.first.out_dim },
        );
    }
    return true;
}

fn executeRawLinearBiasPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: RawLinearBiasPattern,
    prepared: PreparedLinearRegion,
    skipped_nodes: []bool,
) !bool {
    const input = valueFor(values, pattern.dot.input_id) orelse return false;
    const output = (try cb.decoderRuntimeApplyLinear(&.{
        .slot = prepared.linear_slot,
        .input = input,
        .in_dim = pattern.dot.in_dim,
        .out_dim = pattern.dot.out_dim,
    })) orelse return false;

    var dot_shape = [_]i32{
        @intCast(pattern.dot.rows),
        @intCast(pattern.dot.out_dim),
    };
    const dot_output = (try cb.cloneTensorShape(output, &dot_shape)) orelse return false;
    values[@intCast(pattern.dot.id)] = dot_output;
    value_device[@intCast(pattern.dot.id)] = device_id;
    values[@intCast(pattern.add_id)] = output;
    value_device[@intCast(pattern.add_id)] = device_id;
    const add_index: usize = @intCast(pattern.add_id);
    if (add_index < skipped_nodes.len) skipped_nodes[add_index] = true;
    if (stats) |s| {
        recordMetalGraphRegion(s, .qkv, 1);
        s.fused_graph_pattern_dispatches += 1;
        s.fused_graph_nodes_elided += 2;
        recordGemmaRuntimeResidency(s, graph, pattern.add_id, isMetalResidentOrQuantizedDescriptor(cb, output));
    }
    if (traceMetalGraphFusionsEnabled()) {
        const weight_name = graph.parameterName(graph.node(pattern.dot.weight_id));
        const bias_name = graph.parameterName(graph.node(pattern.bias_id));
        std.debug.print(
            "metal_graph_fusion_trace: raw_linear_bias executed dot={d} add={d} transpose={d} weight={s} bias={s} rows={d} in={d} out={d}\n",
            .{ pattern.dot.id, pattern.add_id, pattern.dot.transpose_id, weight_name, bias_name, pattern.dot.rows, pattern.dot.in_dim, pattern.dot.out_dim },
        );
    }
    return true;
}

fn executeRawLinearBiasPairPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: RawLinearBiasPairPattern,
    prepared: PreparedLinearPairRegion,
    skipped_nodes: []bool,
) !bool {
    const input = valueFor(values, pattern.first.dot.input_id) orelse return false;
    const pair = (try cb.decoderRuntimeApplyLinearPair(&.{
        .slot_a = prepared.first_slot,
        .slot_b = prepared.second_slot,
        .input = input,
        .in_dim = pattern.first.dot.in_dim,
        .out_dim = pattern.first.dot.out_dim,
    })) orelse return false;

    var first_dot_shape = [_]i32{
        @intCast(pattern.first.dot.rows),
        @intCast(pattern.first.dot.out_dim),
    };
    const first_dot_output = (try cb.cloneTensorShape(pair.first, &first_dot_shape)) orelse return false;
    values[@intCast(pattern.first.dot.id)] = first_dot_output;
    value_device[@intCast(pattern.first.dot.id)] = device_id;
    values[@intCast(pattern.first.add_id)] = pair.first;
    value_device[@intCast(pattern.first.add_id)] = device_id;
    values[@intCast(pattern.second.add_id)] = pair.second;
    value_device[@intCast(pattern.second.add_id)] = device_id;
    markRawLinearBiasPairSkipped(skipped_nodes, pattern);

    if (stats) |s| {
        recordMetalGraphRegion(s, .qkv, 4);
        s.fused_graph_pattern_dispatches += 1;
        s.fused_graph_nodes_elided += 3;
        recordGemmaRuntimeResidency(s, graph, pattern.first.add_id, isMetalResidentOrQuantizedDescriptor(cb, pair.first));
        recordGemmaRuntimeResidency(s, graph, pattern.second.add_id, isMetalResidentOrQuantizedDescriptor(cb, pair.second));
    }
    if (traceMetalGraphFusionsEnabled()) {
        const first_weight_name = graph.parameterName(graph.node(pattern.first.dot.weight_id));
        const second_weight_name = graph.parameterName(graph.node(pattern.second.dot.weight_id));
        const first_bias_name = graph.parameterName(graph.node(pattern.first.bias_id));
        const second_bias_name = graph.parameterName(graph.node(pattern.second.bias_id));
        std.debug.print(
            "metal_graph_fusion_trace: raw_linear_bias_pair executed first_dot={d} first_add={d} second_dot={d} second_add={d} first_weight={s} second_weight={s} first_bias={s} second_bias={s} rows={d} in={d} out={d}\n",
            .{
                pattern.first.dot.id,
                pattern.first.add_id,
                pattern.second.dot.id,
                pattern.second.add_id,
                first_weight_name,
                second_weight_name,
                first_bias_name,
                second_bias_name,
                pattern.first.dot.rows,
                pattern.first.dot.in_dim,
                pattern.first.dot.out_dim,
            },
        );
    }
    return true;
}

fn matchRawLinearDotPattern(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    last_use: []const u32,
) ?RawLinearDotPattern {
    const dot_id = node_ids[node_pos];
    const dot = graph.node(dot_id);
    const attrs = switch (dot.op) {
        .dot_general => |dot_attrs| dot_attrs,
        else => return null,
    };
    if (dot.num_inputs < 2) return null;
    if (attrs.num_contracting != 1 or attrs.num_batch != 0) return null;
    if (attrs.lhs_contracting[0] != 1 or attrs.rhs_contracting[0] != 0) return null;
    const lhs_id = dot.inputs[0];
    const transpose_id = dot.inputs[1];
    if (lhs_id == null_node or transpose_id == null_node) return null;
    if (!shouldDeferTransposeForLinearDot(graph, transpose_id, reachable, last_use)) return null;
    const transpose = graph.node(transpose_id);
    if (transpose.num_inputs == 0 or transpose.inputs[0] == null_node) return null;
    const weight_id = transpose.inputs[0];
    const weight_node = graph.node(weight_id);
    if (std.meta.activeTag(weight_node.op) != .parameter) return null;
    const weight_name = graph.parameterName(weight_node);
    if (isLoRAAdapterParameterName(weight_name)) return null;
    const lhs_shape = graph.node(lhs_id).output_shape;
    const rhs_shape = graph.node(transpose_id).output_shape;
    if (lhs_shape.rank() != 2 or rhs_shape.rank() != 2) return null;
    const rows = shapeDimUsize(lhs_shape, 0) orelse return null;
    const in_dim = shapeDimUsize(lhs_shape, 1) orelse return null;
    const out_dim = shapeDimUsize(rhs_shape, 1) orelse return null;
    const weight_shape = graph.node(weight_id).output_shape;
    const weight_out_dim = shapeDimUsize(weight_shape, 0) orelse return null;
    const weight_in_dim = shapeDimUsize(weight_shape, 1) orelse return null;
    if (weight_out_dim != out_dim or weight_in_dim != in_dim) return null;
    return .{
        .id = dot_id,
        .input_id = lhs_id,
        .transpose_id = transpose_id,
        .weight_id = weight_id,
        .rows = rows,
        .in_dim = in_dim,
        .out_dim = out_dim,
    };
}

fn matchRawLinearBiasPattern(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    last_use: []const u32,
    skipped_nodes: []const bool,
) ?RawLinearBiasPattern {
    const dot = matchRawLinearDotPattern(graph, node_ids, node_pos, reachable, last_use) orelse return null;
    if (reachableUseCount(graph, dot.id, reachable, 2) != 1) return null;

    var consumer_id: NodeId = 0;
    while (consumer_id < graph.nodeCount()) : (consumer_id += 1) {
        const consumer_index: usize = @intCast(consumer_id);
        if (consumer_index >= reachable.len or !reachable[consumer_index]) continue;
        if (consumer_index < skipped_nodes.len and skipped_nodes[consumer_index]) continue;
        const consumer = graph.node(consumer_id);
        switch (consumer.op) {
            .add, .fused_elem_add => {},
            else => continue,
        }
        if (consumer.num_inputs < 2) continue;
        if (consumer.inputs[0] != dot.id and consumer.inputs[1] != dot.id) continue;
        const bias_id = if (consumer.inputs[0] == dot.id) consumer.inputs[1] else consumer.inputs[0];
        if (bias_id == null_node) return null;
        const bias_node = graph.node(bias_id);
        if (std.meta.activeTag(bias_node.op) != .parameter) return null;
        const bias_name = graph.parameterName(bias_node);
        if (isLoRAAdapterParameterName(bias_name)) return null;
        const bias_shape = bias_node.output_shape;
        if (bias_shape.rank() != 1) return null;
        if (shapeDimUsize(bias_shape, 0) != dot.out_dim) return null;
        if (consumer.output_shape.rank() != 2) return null;
        if (shapeDimUsize(consumer.output_shape, 0) != dot.rows) return null;
        if (shapeDimUsize(consumer.output_shape, 1) != dot.out_dim) return null;
        return .{
            .dot = dot,
            .add_id = consumer_id,
            .bias_id = bias_id,
        };
    }
    return null;
}

fn matchRawLinearPairPattern(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    last_use: []const u32,
    skipped_nodes: []const bool,
) ?RawLinearPairPattern {
    const first = matchRawLinearDotPattern(graph, node_ids, node_pos, reachable, last_use) orelse return null;
    var candidate_pos = node_pos + 1;
    while (candidate_pos < node_ids.len) : (candidate_pos += 1) {
        const candidate_id = node_ids[candidate_pos];
        const candidate_index: usize = @intCast(candidate_id);
        if (candidate_index >= reachable.len or !reachable[candidate_index]) continue;
        if (candidate_index < skipped_nodes.len and skipped_nodes[candidate_index]) continue;
        const second = matchRawLinearDotPattern(graph, node_ids, candidate_pos, reachable, last_use) orelse return null;
        if (second.input_id != first.input_id) return null;
        if (second.rows != first.rows or second.in_dim != first.in_dim or second.out_dim != first.out_dim) return null;
        if (second.weight_id == first.weight_id) return null;
        return .{
            .first = first,
            .second = second,
        };
    }
    return null;
}

fn matchRawLinearBiasPairPattern(
    graph: *const Graph,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    last_use: []const u32,
    skipped_nodes: []const bool,
) ?RawLinearBiasPairPattern {
    const first = matchRawLinearBiasPattern(graph, node_ids, node_pos, reachable, last_use, skipped_nodes) orelse return null;
    var candidate_pos = node_pos + 1;
    while (candidate_pos < node_ids.len) : (candidate_pos += 1) {
        const candidate_id = node_ids[candidate_pos];
        const candidate_index: usize = @intCast(candidate_id);
        if (candidate_index >= reachable.len or !reachable[candidate_index]) continue;
        if (candidate_index < skipped_nodes.len and skipped_nodes[candidate_index]) continue;
        const second = matchRawLinearBiasPattern(graph, node_ids, candidate_pos, reachable, last_use, skipped_nodes) orelse return null;
        if (second.dot.input_id != first.dot.input_id) return null;
        if (second.dot.rows != first.dot.rows or second.dot.in_dim != first.dot.in_dim or second.dot.out_dim != first.dot.out_dim) return null;
        if (second.dot.weight_id == first.dot.weight_id) return null;
        if (second.bias_id == first.bias_id) return null;
        return .{
            .first = first,
            .second = second,
        };
    }
    return null;
}

fn markRawLinearBiasSkipped(skipped_nodes: []bool, pattern: RawLinearBiasPattern) void {
    const add_index: usize = @intCast(pattern.add_id);
    if (add_index < skipped_nodes.len) skipped_nodes[add_index] = true;
}

fn markRawLinearBiasPairSkipped(skipped_nodes: []bool, pattern: RawLinearBiasPairPattern) void {
    markRawLinearBiasSkipped(skipped_nodes, pattern.first);
    markRawLinearBiasSkipped(skipped_nodes, pattern.second);
    const second_dot_index: usize = @intCast(pattern.second.dot.id);
    if (second_dot_index < skipped_nodes.len) skipped_nodes[second_dot_index] = true;
}

fn markRawLinearPairSkipped(skipped_nodes: []bool, pattern: RawLinearPairPattern) void {
    const second_index: usize = @intCast(pattern.second.id);
    if (second_index < skipped_nodes.len) skipped_nodes[second_index] = true;
}

fn isLoRAAdapterParameterName(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".lora_A") != null or
        std.mem.indexOf(u8, name, ".lora_B") != null or
        std.mem.indexOf(u8, name, ".lora_dropout_mask") != null;
}

fn prepareRawLinearDotRegion(
    cb: *const ComputeBackend,
    values: []?CT,
    pattern: RawLinearDotPattern,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?PreparedLinearRegion {
    const linear_slot = (try ensurePreparedLinearSlot(
        cb,
        values,
        pattern.weight_id,
        pattern.in_dim,
        pattern.out_dim,
        stats,
    )) orelse return null;
    return .{ .linear_slot = linear_slot };
}

fn prepareRawLinearPairRegion(
    cb: *const ComputeBackend,
    values: []?CT,
    pattern: RawLinearPairPattern,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?PreparedLinearPairRegion {
    const first_slot = (try ensurePreparedLinearSlot(
        cb,
        values,
        pattern.first.weight_id,
        pattern.first.in_dim,
        pattern.first.out_dim,
        stats,
    )) orelse return null;
    const second_slot = (try ensurePreparedLinearSlot(
        cb,
        values,
        pattern.second.weight_id,
        pattern.second.in_dim,
        pattern.second.out_dim,
        stats,
    )) orelse return null;
    return .{
        .first_slot = first_slot,
        .second_slot = second_slot,
    };
}

fn prepareRawLinearBiasRegion(
    cb: *const ComputeBackend,
    values: []?CT,
    pattern: RawLinearBiasPattern,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?PreparedLinearRegion {
    const linear_slot = (try ensurePreparedLinearSlotWithOptionalBias(
        cb,
        values,
        pattern.dot.weight_id,
        pattern.bias_id,
        pattern.dot.in_dim,
        pattern.dot.out_dim,
        stats,
    )) orelse return null;
    return .{ .linear_slot = linear_slot };
}

fn prepareRawLinearBiasPairRegion(
    cb: *const ComputeBackend,
    values: []?CT,
    pattern: RawLinearBiasPairPattern,
    stats: ?*PartitionExecutor.ExecutionStats,
) !?PreparedLinearPairRegion {
    const first_slot = (try ensurePreparedLinearSlotWithOptionalBias(
        cb,
        values,
        pattern.first.dot.weight_id,
        pattern.first.bias_id,
        pattern.first.dot.in_dim,
        pattern.first.dot.out_dim,
        stats,
    )) orelse return null;
    const second_slot = (try ensurePreparedLinearSlotWithOptionalBias(
        cb,
        values,
        pattern.second.dot.weight_id,
        pattern.second.bias_id,
        pattern.second.dot.in_dim,
        pattern.second.dot.out_dim,
        stats,
    )) orelse return null;
    return .{
        .first_slot = first_slot,
        .second_slot = second_slot,
    };
}

const QLinearPattern = struct {
    id: NodeId,
    input_id: NodeId,
    weight_id: NodeId,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
};

const LinearNoBiasQkvPattern = struct {
    q_id: NodeId,
    k_id: NodeId,
    v_id: NodeId,
    input_id: NodeId,
    q_weight_id: NodeId,
    k_weight_id: NodeId,
    v_weight_id: NodeId,
    rows: usize,
    in_dim: usize,
    q_out_dim: usize,
    kv_out_dim: usize,
};

fn executeQLinearPattern(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    stats: ?*PartitionExecutor.ExecutionStats,
    pattern: QLinearPattern,
    prepared: PreparedLinearRegion,
) !bool {
    const input = valueFor(values, pattern.input_id) orelse return false;
    const output = (try cb.decoderRuntimeApplyLinear(&.{
        .slot = prepared.linear_slot,
        .input = input,
        .in_dim = pattern.in_dim,
        .out_dim = pattern.out_dim,
    })) orelse return traceQkvRegionDeclined("q_linear_backend_returned_null", pattern.id);

    values[@intCast(pattern.id)] = output;
    value_device[@intCast(pattern.id)] = device_id;
    if (stats) |s| {
        recordMetalGraphRegion(s, .qkv, 1);
        s.fused_graph_pattern_dispatches += 1;
        s.gemma_qkv_hits += 1;
        recordGemmaRuntimeResidency(s, graph, pattern.id, isMetalResidentOrQuantizedDescriptor(cb, output));
    }
    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: q_linear_region executed q={d} rows={d} in={d} out={d}\n",
            .{ pattern.id, pattern.rows, pattern.in_dim, pattern.out_dim },
        );
    }
    return true;
}

fn matchQLinearPattern(
    graph: *const Graph,
    values: []?CT,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?QLinearPattern {
    const q_id = node_ids[node_pos];
    const q_index: usize = @intCast(q_id);
    if (q_index < values.len and values[q_index] != null) return null;
    const q = graph.node(q_id);
    const q_attrs = switch (q.op) {
        .fused_linear_no_bias => |attrs| attrs,
        else => return null,
    };
    if (q_attrs.num_projections != 0 or q_attrs.rows == 0 or q_attrs.in_dim == 0 or q_attrs.out_dim == 0) return null;
    const q_inputs = q.getInputs();
    if (q_inputs.len < 2) return null;
    const q_weight_name = linearWeightParameterName(graph, q) orelse return null;
    if (!isGemmaQWeightName(q_weight_name)) return null;
    if (!hasExpectedReachableUseThrough(
        graph,
        reachable,
        skipped_nodes,
        q_id,
        &isAttentionNode,
        &isQLinearAttentionPathNode,
        8,
    )) return null;
    return .{
        .id = q_id,
        .input_id = q_inputs[0],
        .weight_id = q_inputs[1],
        .rows = q_attrs.rows,
        .in_dim = q_attrs.in_dim,
        .out_dim = q_attrs.out_dim,
    };
}

fn isQLinearAttentionPathNode(node: *const ml.graph.Node) bool {
    return switch (node.op) {
        .reshape,
        .transpose,
        .slice,
        .convert_dtype,
        .mul,
        .fused_rms_norm,
        .fused_elem_multiply,
        .fused_rope,
        .fused_to_float32,
        .fused_from_float32,
        => true,
        else => false,
    };
}

fn isAttentionNode(node: *const ml.graph.Node) bool {
    return switch (node.op) {
        .fused_causal_self_attention, .fused_gqa_causal_attention, .fused_sdpa => true,
        else => false,
    };
}

const GroupedLinearQkvSlicePattern = struct {
    linear_id: NodeId,
    q_slice_id: NodeId,
    k_slice_id: NodeId,
    v_slice_id: NodeId,
    input_id: NodeId,
    q_weight_id: NodeId,
    k_weight_id: NodeId,
    v_weight_id: NodeId,
    rows: usize,
    in_dim: usize,
    q_out_dim: usize,
    kv_out_dim: usize,

    fn elidedNodeCount(_: GroupedLinearQkvSlicePattern) u64 {
        return 3;
    }
};

const RmsNormGroupedLinearQkvSlicePattern = struct {
    norm_id: NodeId,
    norm_input_id: NodeId,
    norm_weight_id: NodeId,
    norm_dim: usize,
    norm_eps: f32,
    qkv: GroupedLinearQkvSlicePattern,

    fn elidedNodeCount(self: RmsNormGroupedLinearQkvSlicePattern) u64 {
        return 1 + self.qkv.elidedNodeCount();
    }
};

fn tryExecuteRmsNormGroupedLinearQkvSlicePattern(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    device_id: DeviceId,
    exec_ctx: PartitionExecutor.ExecutionContext,
    skipped_nodes: []bool,
    last_use: []const u32,
    rt_map: std.AutoHashMapUnmanaged(NodeId, CT),
    donated: std.AutoHashMapUnmanaged(NodeId, void),
) !bool {
    const pattern = matchRmsNormGroupedLinearQkvSlicePattern(graph, values, node_ids, node_pos, reachable, skipped_nodes) orelse return false;
    const prepared = (try prepareRmsNormGroupedLinearQkvSliceRegion(cb, values, pattern, exec_ctx.stats)) orelse return false;
    return executeRmsNormGroupedLinearQkvSlicePattern(
        allocator,
        graph,
        cb,
        values,
        value_device,
        device_id,
        exec_ctx,
        skipped_nodes,
        last_use,
        rt_map,
        donated,
        pattern,
        prepared,
    );
}

fn executeRmsNormGroupedLinearQkvSlicePattern(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    exec_ctx: PartitionExecutor.ExecutionContext,
    skipped_nodes: []bool,
    last_use: []const u32,
    rt_map: std.AutoHashMapUnmanaged(NodeId, CT),
    donated: std.AutoHashMapUnmanaged(NodeId, void),
    pattern: RmsNormGroupedLinearQkvSlicePattern,
    prepared: PreparedRmsNormGroupedQkvRegion,
) !bool {
    const input = valueFor(values, pattern.norm_input_id) orelse return false;

    const normed = cb.decoderRuntimeApplyRmsNorm(&.{
        .slot = prepared.norm_slot,
        .input = input,
        .hidden_size = pattern.norm_dim,
        .eps = pattern.norm_eps,
    }) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => return traceQkvRegionDeclined("rms_grouped_norm_unavailable", pattern.norm_id),
        else => return err,
    } orelse return traceQkvRegionDeclined("rms_grouped_norm_unavailable", pattern.norm_id);
    const normed_ct = normed;
    var normed_owned = true;
    errdefer if (normed_owned) cb.free(normed_ct);
    values[@intCast(pattern.norm_id)] = normed_ct;
    value_device[@intCast(pattern.norm_id)] = device_id;

    const qkv = (try cb.decoderRuntimeApplyLinearQkv(&.{
        .q_slot = prepared.qkv.q_slot,
        .k_slot = prepared.qkv.k_slot,
        .v_slot = prepared.qkv.v_slot,
        .input = normed_ct,
        .in_dim = pattern.qkv.in_dim,
        .q_out_dim = pattern.qkv.q_out_dim,
        .kv_out_dim = pattern.qkv.kv_out_dim,
    })) orelse return traceQkvRegionDeclined("rms_grouped_backend_returned_null", pattern.qkv.linear_id);

    try freeExpiredInputs(
        allocator,
        graph,
        cb,
        values,
        value_device,
        pattern.norm_id,
        device_id,
        last_use,
        null,
        rt_map,
        donated,
        exec_ctx,
    );
    try freeExpiredInputs(
        allocator,
        graph,
        cb,
        values,
        value_device,
        pattern.qkv.linear_id,
        device_id,
        last_use,
        null,
        rt_map,
        donated,
        exec_ctx,
    );

    if (values[@intCast(pattern.norm_id)]) |maybe_normed| {
        if (maybe_normed == normed_ct) {
            cb.free(normed_ct);
            values[@intCast(pattern.norm_id)] = null;
        }
    }
    normed_owned = false;

    values[@intCast(pattern.norm_id)] = qkv.first;
    values[@intCast(pattern.qkv.linear_id)] = null;
    values[@intCast(pattern.qkv.q_slice_id)] = qkv.first;
    values[@intCast(pattern.qkv.k_slice_id)] = qkv.second;
    values[@intCast(pattern.qkv.v_slice_id)] = qkv.third;
    value_device[@intCast(pattern.norm_id)] = device_id;
    value_device[@intCast(pattern.qkv.linear_id)] = device_id;
    value_device[@intCast(pattern.qkv.q_slice_id)] = device_id;
    value_device[@intCast(pattern.qkv.k_slice_id)] = device_id;
    value_device[@intCast(pattern.qkv.v_slice_id)] = device_id;
    skipped_nodes[@intCast(pattern.norm_id)] = true;
    skipped_nodes[@intCast(pattern.qkv.linear_id)] = true;
    skipped_nodes[@intCast(pattern.qkv.q_slice_id)] = true;
    skipped_nodes[@intCast(pattern.qkv.k_slice_id)] = true;
    skipped_nodes[@intCast(pattern.qkv.v_slice_id)] = true;

    if (exec_ctx.stats) |stats| {
        recordMetalGraphRegion(stats, .qkv, 5);
        stats.fused_graph_pattern_dispatches += 1;
        stats.fused_graph_nodes_elided += pattern.elidedNodeCount();
        stats.gemma_rms_norm_hits += 1;
        stats.gemma_qkv_hits += 3;
        const k_resident = isMetalResidentOrQuantizedDescriptor(cb, qkv.second);
        const v_resident = isMetalResidentOrQuantizedDescriptor(cb, qkv.third);
        if (k_resident) {
            stats.device_resident_outputs += 1;
        } else {
            stats.host_materialized_outputs += 1;
            if (traceMetalHostOutputsEnabled()) traceMetalHostOutput(graph, pattern.qkv.k_slice_id, "qkv_region_k_host_output");
        }
        if (v_resident) {
            stats.device_resident_outputs += 1;
        } else {
            stats.host_materialized_outputs += 1;
            if (traceMetalHostOutputsEnabled()) traceMetalHostOutput(graph, pattern.qkv.v_slice_id, "qkv_region_v_host_output");
        }
    }

    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: rms_grouped_qkv_region executed norm={d} linear={d} q={d} k={d} v={d} rows={d} in={d} q_out={d} kv_out={d}\n",
            .{ pattern.norm_id, pattern.qkv.linear_id, pattern.qkv.q_slice_id, pattern.qkv.k_slice_id, pattern.qkv.v_slice_id, pattern.qkv.rows, pattern.qkv.in_dim, pattern.qkv.q_out_dim, pattern.qkv.kv_out_dim },
        );
    }
    return true;
}

fn matchRmsNormGroupedLinearQkvSlicePattern(
    graph: *const Graph,
    values: []?CT,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?RmsNormGroupedLinearQkvSlicePattern {
    const norm_id = node_ids[node_pos];
    const norm = graph.node(norm_id);
    const norm_attrs = switch (norm.op) {
        .fused_rms_norm => |attrs| attrs,
        else => return null,
    };
    if (norm_attrs.dim == 0) return null;
    const norm_inputs = norm.getInputs();
    if (norm_inputs.len < 2) return null;

    const linear_id = findSingleInputNode(graph, node_ids, node_pos + 1, reachable, skipped_nodes, norm_id, &isGroupedLinearQkvCandidate) orelse return null;
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, norm_id, &.{linear_id})) return null;
    const linear_pos = findNodePos(node_ids, linear_id) orelse return null;
    const qkv = matchGroupedLinearQkvSlicePatternAt(graph, values, node_ids, linear_pos, reachable, skipped_nodes) orelse return null;

    return .{
        .norm_id = norm_id,
        .norm_input_id = norm_inputs[0],
        .norm_weight_id = norm_inputs[1],
        .norm_dim = norm_attrs.dim,
        .norm_eps = norm_attrs.eps,
        .qkv = qkv,
    };
}

fn tryExecuteGroupedLinearQkvSlicePattern(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    device_id: DeviceId,
    exec_ctx: PartitionExecutor.ExecutionContext,
    skipped_nodes: []bool,
    last_use: []const u32,
    rt_map: std.AutoHashMapUnmanaged(NodeId, CT),
    donated: std.AutoHashMapUnmanaged(NodeId, void),
) !bool {
    const pattern = matchGroupedLinearQkvSlicePattern(graph, values, node_ids, node_pos, reachable, skipped_nodes) orelse return false;
    const prepared = (try prepareGroupedLinearQkvSliceRegion(cb, values, pattern, exec_ctx.stats)) orelse return false;
    return executeGroupedLinearQkvSlicePattern(
        allocator,
        graph,
        cb,
        values,
        value_device,
        device_id,
        exec_ctx,
        skipped_nodes,
        last_use,
        rt_map,
        donated,
        pattern,
        prepared,
    );
}

fn executeGroupedLinearQkvSlicePattern(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    exec_ctx: PartitionExecutor.ExecutionContext,
    skipped_nodes: []bool,
    last_use: []const u32,
    rt_map: std.AutoHashMapUnmanaged(NodeId, CT),
    donated: std.AutoHashMapUnmanaged(NodeId, void),
    pattern: GroupedLinearQkvSlicePattern,
    prepared: PreparedQkvRegion,
) !bool {
    const input = valueFor(values, pattern.input_id) orelse return false;

    const qkv = (try cb.decoderRuntimeApplyLinearQkv(&.{
        .q_slot = prepared.q_slot,
        .k_slot = prepared.k_slot,
        .v_slot = prepared.v_slot,
        .input = input,
        .in_dim = pattern.in_dim,
        .q_out_dim = pattern.q_out_dim,
        .kv_out_dim = pattern.kv_out_dim,
    })) orelse return traceQkvRegionDeclined("grouped_backend_returned_null", pattern.linear_id);

    values[@intCast(pattern.linear_id)] = qkv.first;
    values[@intCast(pattern.q_slice_id)] = qkv.first;
    values[@intCast(pattern.k_slice_id)] = qkv.second;
    values[@intCast(pattern.v_slice_id)] = qkv.third;
    value_device[@intCast(pattern.linear_id)] = device_id;
    value_device[@intCast(pattern.q_slice_id)] = device_id;
    value_device[@intCast(pattern.k_slice_id)] = device_id;
    value_device[@intCast(pattern.v_slice_id)] = device_id;
    skipped_nodes[@intCast(pattern.linear_id)] = true;
    skipped_nodes[@intCast(pattern.q_slice_id)] = true;
    skipped_nodes[@intCast(pattern.k_slice_id)] = true;
    skipped_nodes[@intCast(pattern.v_slice_id)] = true;

    if (exec_ctx.stats) |stats| {
        recordMetalGraphRegion(stats, .qkv, 4);
        stats.fused_graph_pattern_dispatches += 1;
        stats.fused_graph_nodes_elided += pattern.elidedNodeCount();
        stats.gemma_qkv_hits += 3;
        const k_resident = isMetalResidentOrQuantizedDescriptor(cb, qkv.second);
        const v_resident = isMetalResidentOrQuantizedDescriptor(cb, qkv.third);
        if (k_resident) {
            stats.device_resident_outputs += 1;
        } else {
            stats.host_materialized_outputs += 1;
            if (traceMetalHostOutputsEnabled()) traceMetalHostOutput(graph, pattern.k_slice_id, "qkv_region_k_host_output");
        }
        if (v_resident) {
            stats.device_resident_outputs += 1;
        } else {
            stats.host_materialized_outputs += 1;
            if (traceMetalHostOutputsEnabled()) traceMetalHostOutput(graph, pattern.v_slice_id, "qkv_region_v_host_output");
        }
    }

    try freeExpiredInputs(
        allocator,
        graph,
        cb,
        values,
        value_device,
        pattern.linear_id,
        device_id,
        last_use,
        null,
        rt_map,
        donated,
        exec_ctx,
    );

    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: grouped_qkv_region executed linear={d} q={d} k={d} v={d} rows={d} in={d} q_out={d} kv_out={d}\n",
            .{ pattern.linear_id, pattern.q_slice_id, pattern.k_slice_id, pattern.v_slice_id, pattern.rows, pattern.in_dim, pattern.q_out_dim, pattern.kv_out_dim },
        );
    }
    return true;
}

fn matchGroupedLinearQkvSlicePattern(
    graph: *const Graph,
    values: []?CT,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?GroupedLinearQkvSlicePattern {
    return matchGroupedLinearQkvSlicePatternAt(graph, values, node_ids, node_pos, reachable, skipped_nodes);
}

fn matchGroupedLinearQkvSlicePatternAt(
    graph: *const Graph,
    values: []?CT,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?GroupedLinearQkvSlicePattern {
    const linear_id = node_ids[node_pos];
    const linear = graph.node(linear_id);
    const attrs = switch (linear.op) {
        .fused_linear_no_bias => |linear_attrs| linear_attrs,
        else => return null,
    };
    const rows = shapeDimUsize(linear.output_shape, 0) orelse return null;
    const total_out_dim = shapeDimUsize(linear.output_shape, 1) orelse return null;
    if (rows == 0 or attrs.in_dim == 0 or total_out_dim == 0) return null;
    const inputs = linear.getInputs();
    if (inputs.len < 2) return null;

    var leaves: [3]NodeId = undefined;
    if (!collectThreeRowConcatLeaves(graph, inputs[1], &leaves)) return null;
    const q_rows = shapeDimUsize(graph.node(leaves[0]).output_shape, 0) orelse return null;
    const k_rows = shapeDimUsize(graph.node(leaves[1]).output_shape, 0) orelse return null;
    const v_rows = shapeDimUsize(graph.node(leaves[2]).output_shape, 0) orelse return null;
    const q_cols = shapeDimUsize(graph.node(leaves[0]).output_shape, 1) orelse return null;
    const k_cols = shapeDimUsize(graph.node(leaves[1]).output_shape, 1) orelse return null;
    const v_cols = shapeDimUsize(graph.node(leaves[2]).output_shape, 1) orelse return null;
    if (q_cols != attrs.in_dim or k_cols != attrs.in_dim or v_cols != attrs.in_dim) return null;
    if (k_rows != v_rows or total_out_dim != q_rows + k_rows + v_rows) return null;

    const q_slice = findLinearSliceCandidate(graph, values, node_ids, node_pos + 1, reachable, skipped_nodes, linear_id, 0, q_rows) orelse return null;
    const k_slice = findLinearSliceCandidate(graph, values, node_ids, node_pos + 1, reachable, skipped_nodes, linear_id, q_rows, q_rows + k_rows) orelse return null;
    const v_slice = findLinearSliceCandidate(graph, values, node_ids, node_pos + 1, reachable, skipped_nodes, linear_id, q_rows + k_rows, q_rows + k_rows + v_rows) orelse return null;
    if (!hasOnlyExpectedUses(graph, reachable, skipped_nodes, linear_id, &.{ q_slice, k_slice, v_slice })) return null;

    return .{
        .linear_id = linear_id,
        .q_slice_id = q_slice,
        .k_slice_id = k_slice,
        .v_slice_id = v_slice,
        .input_id = inputs[0],
        .q_weight_id = leaves[0],
        .k_weight_id = leaves[1],
        .v_weight_id = leaves[2],
        .rows = rows,
        .in_dim = attrs.in_dim,
        .q_out_dim = q_rows,
        .kv_out_dim = k_rows,
    };
}

fn isGroupedLinearQkvCandidate(node: *const ml.graph.Node) bool {
    return switch (node.op) {
        .fused_linear_no_bias => true,
        else => false,
    };
}

fn findNodePos(node_ids: []const NodeId, needle: NodeId) ?usize {
    for (node_ids, 0..) |node_id, pos| {
        if (node_id == needle) return pos;
    }
    return null;
}

fn collectThreeRowConcatLeaves(graph: *const Graph, root_id: NodeId, out: *[3]NodeId) bool {
    var count: usize = 0;
    collectRowConcatLeaves(graph, root_id, out, &count) catch return false;
    return count == 3;
}

fn collectRowConcatLeaves(graph: *const Graph, node_id: NodeId, out: *[3]NodeId, count: *usize) !void {
    const node = graph.node(node_id);
    switch (node.op) {
        .concat_prim => |attrs| {
            if (attrs.axis != 0) return error.UnsupportedShape;
            const inputs = node.getInputs();
            if (inputs.len < 2) return error.UnsupportedShape;
            try collectRowConcatLeaves(graph, inputs[0], out, count);
            try collectRowConcatLeaves(graph, inputs[1], out, count);
        },
        .parameter => {
            if (count.* >= out.len) return error.UnsupportedShape;
            out[count.*] = node_id;
            count.* += 1;
        },
        else => return error.UnsupportedShape,
    }
}

fn findLinearSliceCandidate(
    graph: *const Graph,
    values: []?CT,
    node_ids: []const NodeId,
    start_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
    linear_id: NodeId,
    start: usize,
    limit: usize,
) ?NodeId {
    for (node_ids[start_pos..]) |candidate_id| {
        const candidate_index: usize = @intCast(candidate_id);
        if (candidate_index >= reachable.len or !reachable[candidate_index]) continue;
        if (candidate_index < skipped_nodes.len and skipped_nodes[candidate_index]) continue;
        if (values[candidate_index] != null) continue;
        const candidate = graph.node(candidate_id);
        const attrs = switch (candidate.op) {
            .slice => |slice_attrs| slice_attrs,
            else => continue,
        };
        const inputs = candidate.getInputs();
        if (inputs.len < 1 or inputs[0] != linear_id) continue;
        if (attrs.num_axes != 2 or attrs.starts[0] != 0 or attrs.strides[0] != 1 or attrs.strides[1] != 1) continue;
        if (std.math.cast(usize, attrs.starts[1]) orelse continue != start) continue;
        if (std.math.cast(usize, attrs.limits[1]) orelse continue != limit) continue;
        return candidate_id;
    }
    return null;
}

fn tryExecuteLinearNoBiasQkvPattern(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    device_id: DeviceId,
    exec_ctx: PartitionExecutor.ExecutionContext,
    skipped_nodes: []bool,
    last_use: []const u32,
    rt_map: std.AutoHashMapUnmanaged(NodeId, CT),
    donated: std.AutoHashMapUnmanaged(NodeId, void),
) !bool {
    const pattern = matchLinearNoBiasQkvPattern(graph, values, node_ids, node_pos, reachable, skipped_nodes) orelse return false;
    const prepared = (try prepareLinearNoBiasQkvRegion(cb, values, pattern, exec_ctx.stats)) orelse return false;
    return executeLinearNoBiasQkvPattern(
        allocator,
        graph,
        cb,
        values,
        value_device,
        device_id,
        exec_ctx,
        skipped_nodes,
        last_use,
        rt_map,
        donated,
        pattern,
        prepared,
    );
}

fn executeLinearNoBiasQkvPattern(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    device_id: DeviceId,
    exec_ctx: PartitionExecutor.ExecutionContext,
    skipped_nodes: []bool,
    last_use: []const u32,
    rt_map: std.AutoHashMapUnmanaged(NodeId, CT),
    donated: std.AutoHashMapUnmanaged(NodeId, void),
    pattern: LinearNoBiasQkvPattern,
    prepared: PreparedQkvRegion,
) !bool {
    const input = valueFor(values, pattern.input_id) orelse return false;

    const qkv = (try cb.decoderRuntimeApplyLinearQkv(&.{
        .q_slot = prepared.q_slot,
        .k_slot = prepared.k_slot,
        .v_slot = prepared.v_slot,
        .input = input,
        .in_dim = pattern.in_dim,
        .q_out_dim = pattern.q_out_dim,
        .kv_out_dim = pattern.kv_out_dim,
    })) orelse return traceQkvRegionDeclined("backend_returned_null", pattern.q_id);

    values[@intCast(pattern.q_id)] = qkv.first;
    values[@intCast(pattern.k_id)] = qkv.second;
    values[@intCast(pattern.v_id)] = qkv.third;
    value_device[@intCast(pattern.q_id)] = device_id;
    value_device[@intCast(pattern.k_id)] = device_id;
    value_device[@intCast(pattern.v_id)] = device_id;
    skipped_nodes[@intCast(pattern.k_id)] = true;
    skipped_nodes[@intCast(pattern.v_id)] = true;

    if (exec_ctx.stats) |stats| {
        recordMetalGraphRegion(stats, .qkv, 3);
        stats.fused_graph_pattern_dispatches += 1;
        stats.fused_graph_nodes_elided += 2;
        const k_resident = isMetalResidentOrQuantizedDescriptor(cb, qkv.second);
        const v_resident = isMetalResidentOrQuantizedDescriptor(cb, qkv.third);
        if (k_resident) {
            stats.device_resident_outputs += 1;
        } else {
            stats.host_materialized_outputs += 1;
            if (traceMetalHostOutputsEnabled()) traceMetalHostOutput(graph, pattern.k_id, "qkv_region_k_host_output");
        }
        if (v_resident) {
            stats.device_resident_outputs += 1;
        } else {
            stats.host_materialized_outputs += 1;
            if (traceMetalHostOutputsEnabled()) traceMetalHostOutput(graph, pattern.v_id, "qkv_region_v_host_output");
        }
        recordGemmaRuntimeResidency(stats, graph, pattern.k_id, k_resident);
        recordGemmaRuntimeResidency(stats, graph, pattern.v_id, v_resident);
    }

    try interpreter.cloneOutputIfAliasedInputWouldBeFreed(
        allocator,
        graph,
        cb,
        values,
        pattern.k_id,
        last_use,
        rt_map,
        donated,
    );
    try interpreter.cloneOutputIfAliasedInputWouldBeFreed(
        allocator,
        graph,
        cb,
        values,
        pattern.v_id,
        last_use,
        rt_map,
        donated,
    );
    try freeExpiredInputs(
        allocator,
        graph,
        cb,
        values,
        value_device,
        pattern.k_id,
        device_id,
        last_use,
        null,
        rt_map,
        donated,
        exec_ctx,
    );
    try freeExpiredInputs(
        allocator,
        graph,
        cb,
        values,
        value_device,
        pattern.v_id,
        device_id,
        last_use,
        null,
        rt_map,
        donated,
        exec_ctx,
    );

    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: qkv_region executed q={d} k={d} v={d} rows={d} in={d} q_out={d} kv_out={d}\n",
            .{ pattern.q_id, pattern.k_id, pattern.v_id, pattern.rows, pattern.in_dim, pattern.q_out_dim, pattern.kv_out_dim },
        );
    }
    return true;
}

fn traceQkvRegionDeclined(reason: []const u8, node_id: NodeId) bool {
    if (traceMetalGraphFusionsEnabled()) {
        std.debug.print(
            "metal_graph_fusion_trace: qkv_region declined reason={s} node={d}\n",
            .{ reason, node_id },
        );
    }
    return false;
}

fn matchLinearNoBiasQkvPattern(
    graph: *const Graph,
    values: []?CT,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
) ?LinearNoBiasQkvPattern {
    const q_id = node_ids[node_pos];
    const q = graph.node(q_id);
    const q_attrs = switch (q.op) {
        .fused_linear_no_bias => |attrs| attrs,
        else => return null,
    };
    if (q_attrs.num_projections != 0 or q_attrs.rows == 0 or q_attrs.in_dim == 0 or q_attrs.out_dim == 0) return null;
    const q_inputs = q.getInputs();
    if (q_inputs.len < 2) return null;
    const q_weight_name = linearWeightParameterName(graph, q) orelse return null;
    if (!isGemmaQWeightName(q_weight_name)) return null;

    const k_id = findQkvSiblingLinear(graph, values, node_ids, node_pos + 1, reachable, skipped_nodes, q_inputs[0], q_attrs, &isGemmaKWeightName) orelse return null;
    const v_id = findQkvSiblingLinear(graph, values, node_ids, node_pos + 1, reachable, skipped_nodes, q_inputs[0], q_attrs, &isGemmaVWeightName) orelse return null;
    const k = graph.node(k_id);
    const v = graph.node(v_id);
    const k_attrs = switch (k.op) {
        .fused_linear_no_bias => |attrs| attrs,
        else => return null,
    };
    const v_attrs = switch (v.op) {
        .fused_linear_no_bias => |attrs| attrs,
        else => return null,
    };
    if (k_attrs.out_dim != v_attrs.out_dim) return null;
    const k_inputs = k.getInputs();
    const v_inputs = v.getInputs();
    if (k_inputs.len < 2 or v_inputs.len < 2) return null;

    return .{
        .q_id = q_id,
        .k_id = k_id,
        .v_id = v_id,
        .input_id = q_inputs[0],
        .q_weight_id = q_inputs[1],
        .k_weight_id = k_inputs[1],
        .v_weight_id = v_inputs[1],
        .rows = q_attrs.rows,
        .in_dim = q_attrs.in_dim,
        .q_out_dim = q_attrs.out_dim,
        .kv_out_dim = k_attrs.out_dim,
    };
}

fn findQkvSiblingLinear(
    graph: *const Graph,
    values: []?CT,
    node_ids: []const NodeId,
    start_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
    input_id: NodeId,
    q_attrs: anytype,
    weight_name_predicate: *const fn ([]const u8) bool,
) ?NodeId {
    for (node_ids[start_pos..]) |candidate_id| {
        const candidate_index: usize = @intCast(candidate_id);
        if (candidate_index >= reachable.len or !reachable[candidate_index]) continue;
        if (candidate_index < skipped_nodes.len and skipped_nodes[candidate_index]) continue;
        if (values[candidate_index] != null) continue;

        const candidate = graph.node(candidate_id);
        const candidate_attrs = switch (candidate.op) {
            .fused_linear_no_bias => |attrs| attrs,
            else => continue,
        };
        if (candidate_attrs.num_projections != 0) continue;
        if (candidate_attrs.rows != q_attrs.rows or candidate_attrs.in_dim != q_attrs.in_dim) continue;
        const candidate_inputs = candidate.getInputs();
        if (candidate_inputs.len < 2 or candidate_inputs[0] != input_id) continue;
        const weight_name = linearWeightParameterName(graph, candidate) orelse continue;
        if (!weight_name_predicate(weight_name)) continue;
        return candidate_id;
    }
    return null;
}

fn isGemmaQWeightName(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".self_attn.q_proj.weight") != null;
}

fn isGemmaKWeightName(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".self_attn.k_proj.weight") != null;
}

fn isGemmaVWeightName(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".self_attn.v_proj.weight") != null;
}

fn isQueryProjectionWeightName(name: []const u8) bool {
    return isGemmaQWeightName(name) or std.mem.indexOf(u8, name, ".attention.self.query_proj.weight") != null;
}

fn isKeyProjectionWeightName(name: []const u8) bool {
    return isGemmaKWeightName(name) or std.mem.indexOf(u8, name, ".attention.self.key_proj.weight") != null;
}

fn isValueProjectionWeightName(name: []const u8) bool {
    return isGemmaVWeightName(name) or std.mem.indexOf(u8, name, ".attention.self.value_proj.weight") != null;
}

fn isDebertaAttentionOutputDenseName(name: []const u8) bool {
    return std.mem.indexOf(u8, name, ".attention.output.dense.") != null;
}

fn tryExecuteLinearNoBiasPairPattern(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    node_ids: []const NodeId,
    node_pos: usize,
    reachable: []const bool,
    device_id: DeviceId,
    exec_ctx: PartitionExecutor.ExecutionContext,
    skipped_nodes: []bool,
    last_use: []const u32,
    rt_map: std.AutoHashMapUnmanaged(NodeId, CT),
    donated: std.AutoHashMapUnmanaged(NodeId, void),
) !bool {
    const first_id = node_ids[node_pos];
    const first = graph.node(first_id);
    const first_attrs = switch (first.op) {
        .fused_linear_no_bias => |attrs| attrs,
        else => return false,
    };
    if (first_attrs.num_projections != 0) return false;
    const first_inputs = first.getInputs();
    if (first_inputs.len < 2) return false;
    const input_id = first_inputs[0];
    const weight_a_id = first_inputs[1];
    const input = valueFor(values, input_id) orelse return false;
    const weight_a = valueFor(values, weight_a_id) orelse return false;

    const second_id = findLinearNoBiasPairCandidate(
        graph,
        values,
        node_ids,
        node_pos + 1,
        reachable,
        skipped_nodes,
        first_inputs,
        first_attrs,
    ) orelse return false;
    const second = graph.node(second_id);
    const second_inputs = second.getInputs();
    if (second_inputs.len < 2) return false;
    const weight_b = valueFor(values, second_inputs[1]) orelse return false;

    const pair = cb.linearNoBiasPair(
        input,
        weight_a,
        weight_b,
        first_attrs.rows,
        first_attrs.in_dim,
        first_attrs.out_dim,
    ) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => return false,
        else => return err,
    };

    values[@intCast(first_id)] = pair.first;
    values[@intCast(second_id)] = pair.second;
    value_device[@intCast(first_id)] = device_id;
    value_device[@intCast(second_id)] = device_id;
    skipped_nodes[@intCast(second_id)] = true;

    if (exec_ctx.stats) |stats| {
        stats.fused_graph_pattern_dispatches += 1;
        stats.fused_graph_nodes_elided += 1;
        stats.metal_linear_pair_fusions += 1;
        const second_resident = isMetalResidentOrQuantizedDescriptor(cb, pair.second);
        if (second_resident) {
            stats.device_resident_outputs += 1;
        } else {
            stats.host_materialized_outputs += 1;
            if (traceMetalHostOutputsEnabled()) traceMetalHostOutput(graph, second_id, "linear_pair_second_host_output");
        }
        recordGemmaRuntimeResidency(stats, graph, second_id, second_resident);
    }

    try interpreter.cloneOutputIfAliasedInputWouldBeFreed(
        allocator,
        graph,
        cb,
        values,
        second_id,
        last_use,
        rt_map,
        donated,
    );
    try freeExpiredInputs(
        allocator,
        graph,
        cb,
        values,
        value_device,
        second_id,
        device_id,
        last_use,
        null,
        rt_map,
        donated,
        exec_ctx,
    );
    return true;
}

fn findLinearNoBiasPairCandidate(
    graph: *const Graph,
    values: []?CT,
    node_ids: []const NodeId,
    start_pos: usize,
    reachable: []const bool,
    skipped_nodes: []const bool,
    first_inputs: []const NodeId,
    first_attrs: anytype,
) ?NodeId {
    if (first_inputs.len < 2) return null;
    for (node_ids[start_pos..]) |candidate_id| {
        const candidate_index: usize = @intCast(candidate_id);
        if (candidate_index >= reachable.len or !reachable[candidate_index]) continue;
        if (candidate_index < skipped_nodes.len and skipped_nodes[candidate_index]) continue;
        if (values[candidate_index] != null) continue;

        const candidate = graph.node(candidate_id);
        const candidate_attrs = switch (candidate.op) {
            .fused_linear_no_bias => |attrs| attrs,
            else => continue,
        };
        if (candidate_attrs.num_projections != 0) continue;
        if (candidate_attrs.rows != first_attrs.rows or
            candidate_attrs.in_dim != first_attrs.in_dim or
            candidate_attrs.out_dim != first_attrs.out_dim)
        {
            continue;
        }

        const candidate_inputs = candidate.getInputs();
        if (candidate_inputs.len < 2) continue;
        if (candidate_inputs[0] != first_inputs[0]) continue;
        if (candidate_inputs[1] == first_inputs[1]) continue;
        return candidate_id;
    }
    return null;
}

fn classifyGemmaRuntimeResidencyNode(graph: *const Graph, node_id: NodeId) ?GemmaRuntimeResidencyCategory {
    const node = graph.node(node_id);
    switch (node.op) {
        .fused_linear, .fused_linear_no_bias => {
            const weight_name = linearWeightParameterName(graph, node) orelse return null;
            if (!isGemmaWeightName(weight_name)) return null;
            if (std.mem.indexOf(u8, weight_name, ".self_attn.q_proj.weight") != null or
                std.mem.indexOf(u8, weight_name, ".self_attn.k_proj.weight") != null or
                std.mem.indexOf(u8, weight_name, ".self_attn.v_proj.weight") != null)
            {
                return .qkv;
            }
            if (std.mem.indexOf(u8, weight_name, ".self_attn.o_proj.weight") != null) return .o_proj;
            if (std.mem.indexOf(u8, weight_name, ".mlp.gate_proj.weight") != null or
                std.mem.indexOf(u8, weight_name, ".mlp.up_proj.weight") != null or
                std.mem.indexOf(u8, weight_name, ".mlp.down_proj.weight") != null)
            {
                return .mlp_proj;
            }
            return null;
        },
        .dot_general, .fused_gqa_causal_attention => return if (nodeDependsOnGemmaParameter(graph, node_id, 64)) .attention_matmul else null,
        .fused_rms_norm => return if (nodeDependsOnGemmaParameter(graph, node_id, 8)) .rms_norm else null,
        .fused_softmax => return if (nodeDependsOnGemmaParameter(graph, node_id, 64)) .softmax else null,
        .add, .fused_elem_add => return if (nodeDependsOnGemmaParameter(graph, node_id, 64)) .residual_add else null,
        .mul, .fused_elem_multiply => return if (nodeDependsOnGemmaParameter(graph, node_id, 64)) .elementwise_mul else null,
        else => return null,
    }
}

fn linearWeightParameterName(graph: *const Graph, node: *const ml.graph.Node) ?[]const u8 {
    const inputs = node.getInputs();
    if (inputs.len < 2 or inputs[1] == null_node) return null;
    const weight = graph.node(inputs[1]);
    if (std.meta.activeTag(weight.op) != .parameter) return null;
    return graph.parameterName(weight);
}

fn layerIndexForWeight(graph: *const Graph, weight_id: NodeId) ?usize {
    return layerIndexForWeightDepth(graph, weight_id, 8);
}

fn layerIndexForWeightDepth(graph: *const Graph, weight_id: NodeId, depth: usize) ?usize {
    if (weight_id == null_node) return null;
    const weight = graph.node(weight_id);
    if (std.meta.activeTag(weight.op) == .parameter) {
        const name = graph.parameterName(weight);
        if (parseGemmaLayerIndex(name)) |layer_index| return layer_index;
        return parseDebertaLayerIndex(name);
    }
    if (depth == 0) return null;
    for (weight.getInputs()) |input_id| {
        if (layerIndexForWeightDepth(graph, input_id, depth - 1)) |layer_index| return layer_index;
    }
    return null;
}

fn parseGemmaLayerIndex(name: []const u8) ?usize {
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const rest = name[prefix.len..];
    var end: usize = 0;
    while (end < rest.len and std.ascii.isDigit(rest[end])) : (end += 1) {}
    if (end == 0) return null;
    return std.fmt.parseUnsigned(usize, rest[0..end], 10) catch null;
}

fn parseDebertaLayerIndex(name: []const u8) ?usize {
    const prefix = "encoder.layer.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    const rest = name[prefix.len..];
    var end: usize = 0;
    while (end < rest.len and std.ascii.isDigit(rest[end])) : (end += 1) {}
    if (end == 0) return null;
    return std.fmt.parseUnsigned(usize, rest[0..end], 10) catch null;
}

fn isGemmaWeightName(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "model.layers.") or
        std.mem.startsWith(u8, name, "model.norm.") or
        std.mem.startsWith(u8, name, "model.embed_tokens.");
}

const gemma_dependency_visit_limit = 8192;

fn nodeVisited(visited: []const NodeId, node_id: NodeId) bool {
    for (visited) |seen| {
        if (seen == node_id) return true;
    }
    return false;
}

fn nodeDependsOnGemmaParameter(graph: *const Graph, node_id: NodeId, max_depth: usize) bool {
    if (node_id == null_node) return false;
    const StackItem = struct {
        id: NodeId,
        depth: usize,
    };
    var stack: [gemma_dependency_visit_limit]StackItem = undefined;
    var stack_len: usize = 1;
    stack[0] = .{ .id = node_id, .depth = max_depth };
    var visited: [gemma_dependency_visit_limit]NodeId = undefined;
    var visited_len: usize = 0;

    while (stack_len != 0) {
        stack_len -= 1;
        const item = stack[stack_len];
        if (item.id == null_node) continue;
        if (nodeVisited(visited[0..visited_len], item.id)) continue;
        if (visited_len == visited.len) return false;
        visited[visited_len] = item.id;
        visited_len += 1;

        const node = graph.node(item.id);
        if (std.meta.activeTag(node.op) == .parameter) {
            if (isGemmaWeightName(graph.parameterName(node))) return true;
            continue;
        }
        if (item.depth == 0) continue;
        for (node.getInputs()) |input_id| {
            if (input_id == null_node) continue;
            if (stack_len == stack.len) return false;
            stack[stack_len] = .{ .id = input_id, .depth = item.depth - 1 };
            stack_len += 1;
        }
    }
    return false;
}

test "gemma runtime residency stats classify gemma graph nodes only" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 1;
    const dim: usize = 4;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ rows, dim }));
    const rms_weight = try b.parameter("model.layers.0.input_layernorm.weight", ml.graph.Shape.init(.f32, &.{dim}));
    const q_weight = try b.parameter("model.layers.0.self_attn.q_proj.weight", ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const k_weight = try b.parameter("model.layers.0.self_attn.k_proj.weight", ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const v_weight = try b.parameter("model.layers.0.self_attn.v_proj.weight", ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const o_weight = try b.parameter("model.layers.0.self_attn.o_proj.weight", ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const gate_weight = try b.parameter("model.layers.0.mlp.gate_proj.weight", ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const up_weight = try b.parameter("model.layers.0.mlp.up_proj.weight", ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const down_weight = try b.parameter("model.layers.0.mlp.down_proj.weight", ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const non_gemma_weight = try b.parameter("clip.text_projection.weight", ml.graph.Shape.init(.f32, &.{ dim, dim }));

    const normed = try b.rmsNorm(x, rms_weight, @intCast(dim), 1e-5);
    const q = try b.linearNoBias(normed, q_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    const k = try b.linearNoBias(normed, k_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    const v = try b.linearNoBias(normed, v_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    const q_4d = try b.reshape(q, ml.graph.Shape.init(.f32, &.{ 1, 1, rows, dim }));
    const k_4d = try b.reshape(k, ml.graph.Shape.init(.f32, &.{ 1, 1, rows, dim }));
    const v_4d = try b.reshape(v, ml.graph.Shape.init(.f32, &.{ 1, 1, rows, dim }));
    const scores = try g.addNode(.{
        .op = .{ .dot_general = .{
            .lhs_contracting = .{ 3, 0, 0, 0, 0, 0, 0, 0 },
            .rhs_contracting = .{ 3, 0, 0, 0, 0, 0, 0, 0 },
            .lhs_batch = .{ 0, 1, 0, 0, 0, 0, 0, 0 },
            .rhs_batch = .{ 0, 1, 0, 0, 0, 0, 0, 0 },
            .num_contracting = 1,
            .num_batch = 2,
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ 1, 1, rows, rows }),
        .inputs = .{ q_4d, k_4d, null_node, null_node },
        .num_inputs = 2,
    });
    const scale = try b.scalarConst(.f32, 0.5);
    const scaled_scores = try b.mul(scores, scale);
    const mask = try b.scalarConst(.f32, 0.0);
    const masked_scores = try b.add(scaled_scores, mask);
    const probs = try b.softmax(masked_scores);
    const attn = try g.addNode(.{
        .op = .{ .dot_general = .{
            .lhs_contracting = .{ 3, 0, 0, 0, 0, 0, 0, 0 },
            .rhs_contracting = .{ 2, 0, 0, 0, 0, 0, 0, 0 },
            .lhs_batch = .{ 0, 1, 0, 0, 0, 0, 0, 0 },
            .rhs_batch = .{ 0, 1, 0, 0, 0, 0, 0, 0 },
            .num_contracting = 1,
            .num_batch = 2,
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ 1, 1, rows, dim }),
        .inputs = .{ probs, v_4d, null_node, null_node },
        .num_inputs = 2,
    });
    const attn_flat = try b.reshape(attn, ml.graph.Shape.init(.f32, &.{ rows, dim }));
    const o = try b.linearNoBias(attn_flat, o_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    const resid = try b.add(x, o);
    const gate = try b.linearNoBias(resid, gate_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    const up = try b.linearNoBias(resid, up_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    const act = try b.gelu(gate);
    const gated = try b.mul(act, up);
    const down = try b.linearNoBias(gated, down_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    const non_gemma_linear = try b.linearNoBias(x, non_gemma_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    const non_gemma_add = try b.add(x, non_gemma_linear);

    var shared_tail = down;
    for (0..32) |_| {
        const lhs = try b.add(shared_tail, resid);
        const rhs = try b.mul(lhs, scale);
        shared_tail = try b.add(lhs, rhs);
    }
    try std.testing.expect(nodeDependsOnGemmaParameter(&g, shared_tail, 64));

    var stats: PartitionExecutor.ExecutionStats = .{};
    for (&[_]NodeId{ q, k, v, o, gate, up, down, scores, scaled_scores, masked_scores, probs, attn, normed, resid, gated }) |node_id| {
        recordGemmaRuntimeResidency(&stats, &g, node_id, true);
    }
    recordGemmaRuntimeResidency(&stats, &g, non_gemma_linear, true);
    recordGemmaRuntimeResidency(&stats, &g, non_gemma_add, true);

    try std.testing.expectEqual(@as(u64, 3), stats.gemma_qkv_hits);
    try std.testing.expectEqual(@as(u64, 1), stats.gemma_o_proj_hits);
    try std.testing.expectEqual(@as(u64, 3), stats.gemma_mlp_proj_hits);
    try std.testing.expectEqual(@as(u64, 2), stats.gemma_attention_matmul_hits);
    try std.testing.expectEqual(@as(u64, 1), stats.gemma_rms_norm_hits);
    try std.testing.expectEqual(@as(u64, 1), stats.gemma_softmax_hits);
    try std.testing.expectEqual(@as(u64, 2), stats.gemma_residual_add_hits);
    try std.testing.expectEqual(@as(u64, 2), stats.gemma_elementwise_mul_hits);
    try std.testing.expectEqual(@as(u64, 0), stats.gemma_qkv_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), stats.gemma_o_proj_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), stats.gemma_mlp_proj_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), stats.gemma_attention_matmul_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), stats.gemma_rms_norm_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), stats.gemma_softmax_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), stats.gemma_residual_add_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), stats.gemma_elementwise_mul_fallbacks);
}

fn partitionIndexForNodes(
    buffer_plan: *const buffer_plan_mod.BufferPlan,
    node_ids: []const NodeId,
) !u32 {
    if (node_ids.len == 0) return error.InvalidPartitionPlan;
    const first = buffer_plan.slotForNode(node_ids[0]) orelse return error.InvalidBufferPlan;
    const partition_index = first.partition_index;
    for (node_ids) |node_id| {
        const slot = buffer_plan.slotForNode(node_id) orelse return error.InvalidBufferPlan;
        if (slot.partition_index != partition_index) return error.InvalidPartitionPlan;
    }
    return partition_index;
}

fn validatePartitionView(
    view: buffer_plan_mod.PartitionBufferView,
    node_ids: []const NodeId,
) !void {
    if (view.backend != .metal) return error.InvalidPartitionPlan;
    for (node_ids) |node_id| {
        var found = false;
        for (view.slots) |slot_view| {
            if (slot_view.slot.node_id == node_id and slot_view.roles.local) {
                found = true;
                break;
            }
        }
        if (!found) return error.InvalidBufferPlan;
    }
}

const RuntimeUnaryOp = enum {
    negate,
    sqrt,
    rsqrt,
    exp,
    log,
    sin,
    cos,
    tanh,
    erf,
    abs,
};

fn executeRuntimeUnary(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    op: RuntimeUnaryOp,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    return switch (op) {
        .negate => cb.primNegate(input),
        .sqrt => cb.primSqrt(input),
        .rsqrt => cb.primRsqrt(input),
        .exp => cb.primExp(input),
        .log => cb.primLog(input),
        .sin => cb.primSin(input),
        .cos => cb.primCos(input),
        .tanh => cb.primTanh(input),
        .erf => cb.primErf(input),
        .abs => cb.primAbs(input),
    } catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn tryExecuteMetalCommand(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    node_id: NodeId,
    op_plan: ?OperatorPlan,
    exec_state: *interpreter.ExecState,
) !?CT {
    const n = graph.node(node_id);
    const inputs = n.getInputs();
    return switch (n.op) {
        .constant => |attrs| try executeRuntimeConstant(graph, cb, n.output_shape, attrs),
        .reshape => |attrs| blk: {
            const input = valueFor(values, inputs[0]) orelse break :blk null;
            var dims_buf: [ml.graph.shape.max_rank]i64 = undefined;
            const dims = try fillShapeDims(attrs.new_shape, &dims_buf);
            break :blk cb.primReshape(input, dims) catch |err| switch (err) {
                error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
                else => return err,
            };
        },
        .transpose => |attrs| blk: {
            var host_input_to_free: ?CT = null;
            defer if (host_input_to_free) |ct| cb.free(ct);
            const input_source = valueFor(values, inputs[0]) orelse break :blk null;
            const input_source_resident = isMetalDeviceResident(cb, input_source);
            const preserve_host_view = !input_source_resident;
            const input = if (preserve_host_view and input_source_resident) host_input: {
                var host_shape_buf: [ml.graph.shape.max_rank]i32 = undefined;
                const source_shape = graph.node(inputs[0]).output_shape;
                if (source_shape.rank() > host_shape_buf.len) break :host_input input_source;
                for (0..source_shape.rank()) |axis| {
                    const dim = source_shape.dim(@intCast(axis));
                    if (dim <= 0) break :host_input input_source;
                    host_shape_buf[axis] = @intCast(dim);
                }
                const host_data = cb.toFloat32(input_source, allocator) catch break :host_input input_source;
                defer allocator.free(host_data);
                const host_ct = cb.fromFloat32Shape(host_data, host_shape_buf[0..source_shape.rank()]) catch break :host_input input_source;
                host_input_to_free = host_ct;
                break :host_input host_ct;
            } else input_source;
            var in_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
            const in_shape = try fillShapeDims(graph.node(inputs[0]).output_shape, &in_shape_buf);
            var perm_buf: [ml.graph.shape.max_rank]u8 = undefined;
            const perm = transpose_utils.effectivePerm(attrs, graph.node(inputs[0]).output_shape.rank(), &perm_buf);
            const transposed = cb.primTranspose(input, perm, in_shape) catch |err| switch (err) {
                error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
                else => return err,
            };
            if (transposed) |ct| {
                if (!preserve_host_view and !isMetalDeviceResident(cb, ct)) {
                    if (try makeMetalDeviceResident(cb, ct)) |device_ct| {
                        cb.free(ct);
                        break :blk device_ct;
                    }
                }
            }
            break :blk transposed;
        },
        .broadcast_in_dim => |attrs| try executeRuntimeBroadcast(cb, values, inputs, graph.node(inputs[0]).output_shape, attrs),
        .neg => try executeRuntimeUnary(cb, values, inputs, .negate),
        .sqrt => try executeRuntimeUnary(cb, values, inputs, .sqrt),
        .rsqrt => try executeRuntimeUnary(cb, values, inputs, .rsqrt),
        .exp => try executeRuntimeUnary(cb, values, inputs, .exp),
        .log => try executeRuntimeUnary(cb, values, inputs, .log),
        .sin => try executeRuntimeUnary(cb, values, inputs, .sin),
        .cos => try executeRuntimeUnary(cb, values, inputs, .cos),
        .tanh => try executeRuntimeUnary(cb, values, inputs, .tanh),
        .erf => try executeRuntimeUnary(cb, values, inputs, .erf),
        .abs => try executeRuntimeUnary(cb, values, inputs, .abs),
        .slice => |attrs| try executeRuntimeSlice(graph, cb, values, inputs, attrs),
        .concat_prim => |attrs| try executeRuntimeConcatPrim(graph, cb, values, inputs, attrs),
        .gather => |attrs| try executeRuntimeGather(graph, cb, values, inputs, attrs),
        .scatter_add => |attrs| try executeRuntimeScatterAdd(graph, cb, values, inputs, attrs),
        .fused_gelu => try executeRuntimeActivation(cb, values, inputs, .gelu, n.output_shape),
        .fused_gelu_backward => try executeRuntimeGeluBackward(cb, values, inputs, n.output_shape),
        .fused_relu => try executeRuntimeActivation(cb, values, inputs, .relu, n.output_shape),
        .fused_silu => try executeRuntimeActivation(cb, values, inputs, .silu, n.output_shape),
        .fused_quick_gelu => try executeRuntimeActivation(cb, values, inputs, .quick_gelu, n.output_shape),
        .fused_sigmoid => try executeRuntimeFusedUnary(cb, values, inputs, .sigmoid),
        .fused_tanh_act => try executeRuntimeFusedUnary(cb, values, inputs, .tanh_act),
        .fused_elem_add, .add => try executeRuntimeAdd(graph, cb, values, inputs, n.output_shape),
        .fused_elem_multiply, .mul => try executeRuntimeBinary(cb, values, inputs, .multiply),
        .sub => try executeRuntimeBinary(cb, values, inputs, .subtract),
        .div => try executeRuntimeBinary(cb, values, inputs, .divide),
        .less_than => try executeRuntimeBinary(cb, values, inputs, .less_than),
        .where_select => try executeRuntimeWhereSelect(cb, values, inputs),
        .reduce_sum => |attrs| try executeRuntimeReduce(graph, cb, values, inputs, attrs, .sum),
        .reduce_max => |attrs| try executeRuntimeReduce(graph, cb, values, inputs, attrs, .max),
        .reduce_mean => |attrs| try executeRuntimeReduce(graph, cb, values, inputs, attrs, .mean),
        .fused_softmax => |attrs| try executeRuntimeSoftmax(cb, values, inputs, attrs.dim),
        .fused_log_softmax => |attrs| try executeRuntimeLogSoftmax(cb, values, inputs, attrs.dim),
        .fused_sdpa => |attrs| try executeRuntimeSdpa(cb, values, inputs, attrs, op_plan, exec_state),
        .fused_gqa_causal_attention => |attrs| try executeRuntimeGqaCausalAttention(cb, values, inputs, attrs, n.num_inputs, exec_state),
        .dot_general => |attrs| try executeRuntimeDotGeneral(graph, cb, values, inputs, attrs, op_plan),
        .conv_general => |attrs| try executeRuntimeConvGeneral(graph, cb, values, inputs, attrs),
        .fused_conv1d => |attrs| try executeRuntimeConv1d(graph, cb, values, inputs, attrs),
        .fused_conv2d => |attrs| try executeRuntimeConv2d(graph, cb, values, inputs, attrs),
        .fused_linear => |attrs| try executeRuntimeLinear(cb, values, inputs, attrs.rows, attrs.in_dim, attrs.out_dim, true, op_plan),
        .fused_linear_no_bias => |attrs| blk: {
            if (attrs.num_projections != 0) {
                break :blk try executeRuntimeLinearNoBiasGrouped(cb, values, inputs, attrs);
            }
            break :blk try executeRuntimeLinear(cb, values, inputs, attrs.rows, attrs.in_dim, attrs.out_dim, false, op_plan);
        },
        .fused_linear_no_bias_pair => |attrs| try executeRuntimeLinearNoBiasPair(cb, values, inputs, attrs, exec_state),
        .fused_to_float32 => blk: {
            if (exec_state.pair_second) |second| {
                exec_state.pair_second = null;
                break :blk second;
            }
            break :blk valueFor(values, inputs[0]);
        },
        .fused_embedding_lookup => |attrs| try executeRuntimeEmbeddingLookup(graph, cb, values, inputs, attrs, exec_state),
        .fused_take_rows => |attrs| try executeRuntimeTakeRows(cb, values, inputs, attrs.rows, attrs.dim, op_plan, exec_state),
        .fused_zero_tensor => |attrs| try executeRuntimeZeroTensor(cb, attrs.rows, attrs.out_dim),
        .fused_rope => |attrs| try executeRuntimeRope(cb, values, inputs, attrs, exec_state),
        .fused_layer_norm => |attrs| try executeRuntimeLayerNorm(cb, values, inputs, attrs.dim, attrs.eps, n.output_shape),
        .fused_rms_norm => |attrs| try executeRuntimeRmsNorm(cb, values, inputs, attrs.dim, attrs.eps, n.output_shape),
        else => null,
    };
}

fn executeRuntimeZeroTensor(
    cb: *const ComputeBackend,
    rows: usize,
    out_dim: usize,
) !?CT {
    const ct = (try cb.zeroTensor(rows, out_dim)) orelse return null;
    errdefer cb.free(ct);
    if (isMetalDeviceResident(cb, ct)) return ct;
    if (try makeMetalDeviceResident(cb, ct)) |device_ct| {
        if (device_ct != ct) cb.free(ct);
        return device_ct;
    }
    return ct;
}

fn executeRuntimeConstant(
    graph: *const Graph,
    cb: *const ComputeBackend,
    output_shape: ml.graph.Shape,
    attrs: anytype,
) !?CT {
    const constant = try graph.constantDataAsF32(
        graph.allocator,
        output_shape.dtype,
        attrs.data_offset,
        attrs.data_len,
    );
    defer constant.deinit(graph.allocator);

    var shape_buf: [ml.graph.shape.max_rank]i32 = undefined;
    const rank = output_shape.rank();
    const ct = if (rank > 1) blk: {
        for (0..rank) |axis| shape_buf[axis] = @intCast(output_shape.dim(@intCast(axis)));
        break :blk try cb.fromFloat32Shape(constant.data, shape_buf[0..rank]);
    } else try cb.fromFloat32(constant.data);
    errdefer cb.free(ct);

    if (isMetalDeviceResident(cb, ct)) return ct;
    if (try makeMetalDeviceResident(cb, ct)) |device_ct| {
        if (device_ct != ct) cb.free(ct);
        return device_ct;
    }
    return ct;
}

fn executeRuntimeGqaCausalAttention(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
    num_inputs: u8,
    exec_state: *interpreter.ExecState,
) !?CT {
    const q = valueFor(values, inputs[0]) orelse return null;
    const k = valueFor(values, inputs[1]) orelse return null;
    const v = valueFor(values, inputs[2]) orelse return null;
    const bias = valueFor(values, if (num_inputs > 3) inputs[3] else null_node);
    const kv_heads = if (attrs.num_kv_heads != 0) attrs.num_kv_heads else attrs.num_heads;

    if (exec_state.options.attention) |base_attn| {
        var attn = base_attn;
        attn.layer_index = if (attrs.layer_index == std.math.maxInt(u32))
            exec_state.attention_layer
        else
            attrs.layer_index;
        attn.skip_kv_write = attrs.skip_kv_write;
        const out = cb.gqaPagedAttention(
            q,
            k,
            v,
            bias,
            attn,
            attrs.batch,
            attrs.num_heads,
            kv_heads,
            attrs.head_dim,
        ) catch |err| switch (err) {
            error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch, error.UnsupportedTensorType, error.InvalidAttentionShape => null,
            else => return err,
        };
        if (out != null) exec_state.attention_layer += 1;
        return out;
    }

    return cb.gqaCausalAttention(
        q,
        k,
        v,
        bias,
        attrs.batch,
        attrs.seq_len,
        attrs.num_heads,
        kv_heads,
        attrs.head_dim,
    ) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch, error.UnsupportedTensorType, error.InvalidAttentionShape => null,
        else => return err,
    };
}

fn executeRuntimeSdpa(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
    op_plan: ?OperatorPlan,
    exec_state: *interpreter.ExecState,
) !?CT {
    const q = valueFor(values, inputs[0]) orelse return null;
    const k = valueFor(values, inputs[1]) orelse return null;
    const v = valueFor(values, inputs[2]) orelse return null;
    const kv_len = if (attrs.kv_seq_len != 0) attrs.kv_seq_len else attrs.seq_len;
    const attention_plan = try validatePlannedAttentionOp(attrs.seq_len, kv_len, attrs.head_dim, op_plan);
    const bias = valueFor(values, if (inputs.len > 3) inputs[3] else null_node);
    const kv_heads = if (attrs.num_kv_heads != 0) attrs.num_kv_heads else attrs.num_heads;

    if (attention_plan.operator == .attention_paged or attention_plan.operator == .attention_quantized_kv) {
        const attention = exec_state.options.attention orelse return null;
        return cb.gqaPagedAttention(
            q,
            k,
            v,
            bias,
            attention,
            attrs.batch,
            attrs.num_heads,
            kv_heads,
            attrs.head_dim,
        ) catch |err| switch (err) {
            error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch, error.UnsupportedTensorType, error.InvalidAttentionShape => null,
            else => return err,
        };
    }

    var synthesized_mask: ?[]i64 = null;
    defer if (synthesized_mask) |buf| std.heap.page_allocator.free(buf);
    const mask = blk: {
        if (exec_state.options.sdpa_mask) |runtime_mask| break :blk runtime_mask;
        if (attrs.batch == 0 or attrs.seq_len == 0) return error.MissingRuntimeInput;
        const full_mask = try std.heap.page_allocator.alloc(i64, @as(usize, attrs.batch) * @as(usize, attrs.seq_len));
        @memset(full_mask, 1);
        synthesized_mask = full_mask;
        break :blk full_mask;
    };

    return cb.scaledDotProductAttention(
        q,
        k,
        v,
        mask,
        bias,
        attrs.batch,
        attrs.seq_len,
        attrs.num_heads,
        attrs.head_dim,
    ) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

const RuntimeFusedUnaryOp = enum {
    sigmoid,
    tanh_act,
};

fn executeRuntimeFusedUnary(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    comptime op: RuntimeFusedUnaryOp,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    return switch (op) {
        .sigmoid => cb.sigmoid(input),
        .tanh_act => cb.tanh_act(input),
    } catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeConcatPrim(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
) !?CT {
    const lhs = valueFor(values, inputs[0]) orelse return null;
    const rhs = valueFor(values, inputs[1]) orelse return null;
    var lhs_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    var rhs_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    const lhs_shape = try fillShapeDims(graph.node(inputs[0]).output_shape, &lhs_shape_buf);
    const rhs_shape = try fillShapeDims(graph.node(inputs[1]).output_shape, &rhs_shape_buf);
    return cb.primConcatPrim(lhs, rhs, attrs.axis, lhs_shape, rhs_shape) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeScatterAdd(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
) !?CT {
    if (attrs.axis != 0 or inputs.len != 3) return null;
    const dest = valueFor(values, inputs[0]) orelse return null;
    const update_values = valueFor(values, inputs[1]) orelse return null;
    const indices = valueFor(values, inputs[2]) orelse return null;

    var dest_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    var values_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    var indices_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    const dest_shape = try fillShapeDims(graph.node(inputs[0]).output_shape, &dest_shape_buf);
    const values_shape = try fillShapeDims(graph.node(inputs[1]).output_shape, &values_shape_buf);
    const indices_shape = try fillShapeDims(graph.node(inputs[2]).output_shape, &indices_shape_buf);

    if (dest_shape.len != 2 or values_shape.len != 2 or indices_shape.len != 1) return null;
    if (dest_shape[0] <= 0 or dest_shape[1] <= 0 or values_shape[0] < 0 or values_shape[1] != dest_shape[1]) return null;

    const out_rows: usize = @intCast(dest_shape[0]);
    const value_rows: usize = @intCast(values_shape[0]);
    const dim: usize = @intCast(dest_shape[1]);

    const allocator = std.heap.page_allocator;
    const dest_data = try cb.toFloat32(dest, allocator);
    defer allocator.free(dest_data);
    const values_data = try cb.toFloat32(update_values, allocator);
    defer allocator.free(values_data);
    const index_data = try cb.toFloat32(indices, allocator);
    defer allocator.free(index_data);

    if (dest_data.len != out_rows * dim or values_data.len != value_rows * dim or index_data.len < value_rows) return error.ShapeMismatch;

    const output = try allocator.dupe(f32, dest_data);
    defer allocator.free(output);
    for (0..value_rows) |row_idx| {
        const out_row_f = @round(index_data[row_idx]);
        if (out_row_f < 0) return error.IndexOutOfBounds;
        const out_row: usize = @intFromFloat(out_row_f);
        if (out_row >= out_rows) return error.IndexOutOfBounds;
        const src = values_data[row_idx * dim ..][0..dim];
        const dst = output[out_row * dim ..][0..dim];
        for (src, dst) |v, *d| d.* += v;
    }

    const out_shape = [_]i32{ @intCast(out_rows), @intCast(dim) };
    const host_result = try cb.fromFloat32Shape(output, &out_shape);
    errdefer cb.free(host_result);
    if (isMetalDeviceResident(cb, host_result)) return host_result;
    if (try makeMetalDeviceResident(cb, host_result)) |device_result| {
        if (device_result != host_result) cb.free(host_result);
        return device_result;
    }
    return host_result;
}

fn executeRuntimeEmbeddingLookup(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
    exec_state: *interpreter.ExecState,
) !?CT {
    const weight = valueFor(values, inputs[0]) orelse return null;
    var owned_ids: ?[]i64 = null;
    defer if (owned_ids) |buf| std.heap.page_allocator.free(buf);
    const ids = blk: {
        if (graph.node(inputs[1]).op == .fused_from_float32) {
            break :blk exec_state.options.embedding_ids orelse return error.MissingRuntimeInput;
        }
        const ids_ct = valueFor(values, inputs[1]) orelse return null;
        const raw = try cb.toFloat32(ids_ct, std.heap.page_allocator);
        defer std.heap.page_allocator.free(raw);
        const converted = try std.heap.page_allocator.alloc(i64, raw.len);
        for (converted, raw) |*dst, value| dst.* = @intFromFloat(@round(value));
        owned_ids = converted;
        break :blk converted;
    };
    return cb.embeddingLookup(weight, ids, attrs.total, attrs.dim) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeBroadcast(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    input_shape: ml.graph.Shape,
    attrs: anytype,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    var in_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    const in_rank = input_shape.rank();
    if (in_rank > in_shape_buf.len) return error.UnsupportedShape;
    for (0..in_rank) |axis| in_shape_buf[axis] = input_shape.dim(@intCast(axis));

    var target_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    const target_rank = attrs.target_shape.rank();
    if (target_rank > target_shape_buf.len) return error.UnsupportedShape;
    for (0..target_rank) |axis| target_shape_buf[axis] = attrs.target_shape.dim(@intCast(axis));

    return cb.primBroadcastInDim(
        input,
        target_shape_buf[0..target_rank],
        attrs.broadcast_axes[0..attrs.num_axes],
        in_shape_buf[0..in_rank],
    ) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

const RuntimeReduceOp = enum {
    sum,
    max,
    mean,
};

fn executeRuntimeReduce(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
    comptime op: RuntimeReduceOp,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    var in_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    const in_shape = try fillShapeDims(graph.node(inputs[0]).output_shape, &in_shape_buf);
    const axes = attrs.axes[0..attrs.num_axes];
    return switch (op) {
        .sum => cb.primReduceSum(input, axes, in_shape),
        .max => cb.primReduceMax(input, axes, in_shape),
        .mean => cb.primReduceMean(input, axes, in_shape),
    } catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

const RuntimeBinaryOp = enum {
    multiply,
    subtract,
    divide,
    less_than,
};

fn executeRuntimeBinary(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    op: RuntimeBinaryOp,
) !?CT {
    var lhs = valueFor(values, inputs[0]) orelse return null;
    var rhs = valueFor(values, inputs[1]) orelse return null;
    var owned_lhs: ?CT = null;
    defer if (owned_lhs) |ct| cb.free(ct);
    var owned_rhs: ?CT = null;
    defer if (owned_rhs) |ct| cb.free(ct);
    if (cb.kind() == .metal) {
        if (!isMetalDeviceResident(cb, lhs)) {
            if (try makeMetalDeviceResident(cb, lhs)) |device_lhs| {
                if (device_lhs != lhs) owned_lhs = device_lhs;
                lhs = device_lhs;
            }
        }
        if (!isMetalDeviceResident(cb, rhs)) {
            if (try makeMetalDeviceResident(cb, rhs)) |device_rhs| {
                if (device_rhs != rhs) owned_rhs = device_rhs;
                rhs = device_rhs;
            }
        }
    }
    return switch (op) {
        .multiply => cb.multiply(lhs, rhs),
        .subtract => cb.primSubtract(lhs, rhs),
        .divide => cb.primDivide(lhs, rhs),
        .less_than => cb.primLessThan(lhs, rhs),
    } catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeWhereSelect(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
) !?CT {
    const cond = valueFor(values, inputs[0]) orelse return null;
    const on_true = valueFor(values, inputs[1]) orelse return null;
    const on_false = valueFor(values, inputs[2]) orelse return null;
    return cb.primWhereSelect(cond, on_true, on_false) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeSlice(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    var in_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    const in_shape = try fillShapeDims(graph.node(inputs[0]).output_shape, &in_shape_buf);
    const rank = @as(usize, attrs.num_axes);
    if (rank > ml.graph.shape.max_rank) return error.UnsupportedShape;
    var starts: [ml.graph.shape.max_rank]i64 = undefined;
    var limits: [ml.graph.shape.max_rank]i64 = undefined;
    var strides: [ml.graph.shape.max_rank]i64 = undefined;
    for (0..rank) |axis| {
        starts[axis] = attrs.starts[axis];
        limits[axis] = attrs.limits[axis];
        strides[axis] = attrs.strides[axis];
    }
    if (in_shape.len == 2 and rank == 2 and
        starts[0] == 0 and limits[0] == in_shape[0] and
        strides[0] == 1 and strides[1] == 1)
    {
        return cb.sliceLastDim(input, @intCast(starts[1]), @intCast(limits[1])) catch |err| switch (err) {
            error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
            else => return err,
        };
    }
    return cb.primSlice(input, starts[0..rank], limits[0..rank], strides[0..rank], in_shape) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeLinear(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    comptime has_bias: bool,
    op_plan: ?OperatorPlan,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    const weight = valueFor(values, inputs[1]) orelse return null;
    try validatePlannedLinearOp(cb, weight, rows, in_dim, out_dim, op_plan);
    if (has_bias) {
        const bias = valueFor(values, inputs[2]) orelse return null;
        const output = cb.linearWithPlan(input, weight, bias, rows, in_dim, out_dim, op_plan) catch |err| switch (err) {
            error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
            else => return err,
        };
        return try promoteMetalOutputIfNeeded(cb, output);
    } else {
        const output = cb.linearNoBiasWithPlan(input, weight, rows, in_dim, out_dim, op_plan) catch |err| switch (err) {
            error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
            else => return err,
        };
        return try promoteMetalOutputIfNeeded(cb, output);
    }
}

fn executeRuntimeLinearNoBiasGrouped(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    const weight = valueFor(values, inputs[1]) orelse return null;
    return cb.linearNoBiasGrouped(
        input,
        weight,
        attrs.rows,
        attrs.in_dim,
        attrs.out_dim,
        attrs.projection_out_dims[0..attrs.num_projections],
        attrs.num_projections,
    ) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeDotGeneral(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
    op_plan: ?OperatorPlan,
) !?CT {
    const lhs = valueFor(values, inputs[0]) orelse return null;
    var lhs_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    var rhs_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    const lhs_shape = try fillShapeDims(graph.node(inputs[0]).output_shape, &lhs_shape_buf);
    const rhs_shape = try fillShapeDims(graph.node(inputs[1]).output_shape, &rhs_shape_buf);
    const lhs_contracting = attrs.lhs_contracting[0..attrs.num_contracting];
    const rhs_contracting = attrs.rhs_contracting[0..attrs.num_contracting];
    const lhs_batch = attrs.lhs_batch[0..attrs.num_batch];
    const rhs_batch = attrs.rhs_batch[0..attrs.num_batch];
    if (try executeRuntimeLinearDotFromDeferredTranspose(graph, cb, values, inputs, lhs_shape, rhs_shape, lhs_contracting, rhs_contracting, lhs_batch, rhs_batch, op_plan)) |linear_output| {
        return linear_output;
    }
    const rhs = valueFor(values, inputs[1]) orelse return null;
    if (op_plan != null and attrs.num_contracting == 1 and attrs.num_batch == 0 and lhs_shape.len == 2 and rhs_shape.len == 2 and lhs_contracting[0] == 1 and rhs_contracting[0] == 1) {
        const rows = positiveI64ToUsize(lhs_shape[0]) orelse return null;
        const in_dim = positiveI64ToUsize(lhs_shape[1]) orelse return null;
        const out_dim = positiveI64ToUsize(rhs_shape[0]) orelse return null;
        if (rhs_shape[1] == lhs_shape[1]) {
            return executeRuntimeLinear(cb, values, inputs, rows, in_dim, out_dim, false, op_plan);
        }
    }
    const output = cb.primDotGeneral(lhs, rhs, lhs_shape, rhs_shape, lhs_contracting, rhs_contracting, lhs_batch, rhs_batch) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
    return try promoteMetalOutputIfNeeded(cb, output);
}

fn executeRuntimeLinearDotFromDeferredTranspose(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    lhs_shape: []const i64,
    rhs_shape: []const i64,
    lhs_contracting: []const u8,
    rhs_contracting: []const u8,
    lhs_batch: []const u8,
    rhs_batch: []const u8,
    op_plan: ?OperatorPlan,
) !?CT {
    if (inputs.len < 2) return null;
    if (lhs_batch.len != 0 or rhs_batch.len != 0) return null;
    if (lhs_contracting.len != 1 or rhs_contracting.len != 1) return null;
    if (lhs_shape.len != 2 or rhs_shape.len != 2) return null;
    if (lhs_contracting[0] != 1 or rhs_contracting[0] != 0) return null;

    const rhs_id = inputs[1];
    if (rhs_id == null_node) return null;
    const rhs_node = graph.node(rhs_id);
    const transpose_attrs = switch (rhs_node.op) {
        .transpose => |attrs| attrs,
        else => return null,
    };
    if (rhs_node.num_inputs == 0 or rhs_node.inputs[0] == null_node) return null;
    if (!transposeIsSimple2D(transpose_attrs, graph.node(rhs_node.inputs[0]).output_shape)) return null;

    const source_weight_id = rhs_node.inputs[0];
    const input = valueFor(values, inputs[0]) orelse return null;
    const weight = valueFor(values, source_weight_id) orelse return null;
    const rows = positiveI64ToUsize(lhs_shape[0]) orelse return null;
    const in_dim = positiveI64ToUsize(lhs_shape[1]) orelse return null;
    const out_dim = positiveI64ToUsize(rhs_shape[1]) orelse return null;
    const weight_shape = graph.node(source_weight_id).output_shape;
    const weight_out_dim = shapeDimUsize(weight_shape, 0) orelse return null;
    const weight_in_dim = shapeDimUsize(weight_shape, 1) orelse return null;
    if (weight_out_dim != out_dim or weight_in_dim != in_dim) return null;
    if (in_dim == 1 and rank1DotSpecializationEnabled()) {
        var weight_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
        const source_weight_shape = try fillShapeDims(weight_shape, &weight_shape_buf);
        const output = cb.primDotGeneral(input, weight, lhs_shape, source_weight_shape, lhs_contracting, &.{1}, lhs_batch, rhs_batch) catch |err| switch (err) {
            error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
            else => return err,
        };
        return try promoteMetalOutputIfNeeded(cb, output);
    }
    const output = cb.linearNoBiasWithPlan(input, weight, rows, in_dim, out_dim, op_plan) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
    return try promoteMetalOutputIfNeeded(cb, output);
}

fn shouldDeferTransposeForLinearDot(
    graph: *const Graph,
    node_id: NodeId,
    reachable: []const bool,
    last_use: []const u32,
) bool {
    _ = last_use;
    const node_index: usize = @intCast(node_id);
    if (node_index >= reachable.len or !reachable[node_index]) return false;
    const node = graph.node(node_id);
    const attrs = switch (node.op) {
        .transpose => |transpose_attrs| transpose_attrs,
        else => return false,
    };
    if (node.num_inputs == 0 or node.inputs[0] == null_node) return false;
    if (!transposeIsSimple2D(attrs, graph.node(node.inputs[0]).output_shape)) return false;
    var compatible_consumers: usize = 0;
    var consumer_id: NodeId = 0;
    while (consumer_id < graph.nodeCount()) : (consumer_id += 1) {
        const consumer_index: usize = @intCast(consumer_id);
        if (consumer_index >= reachable.len or !reachable[consumer_index]) continue;
        const consumer = graph.node(consumer_id);
        var consumes_transpose = false;
        for (consumer.getInputs()) |input_id| {
            if (input_id == node_id) {
                consumes_transpose = true;
                break;
            }
        }
        if (!consumes_transpose) continue;
        if (!linearDotConsumesTranspose(graph, consumer_id, node_id)) return false;
        compatible_consumers += 1;
    }
    return compatible_consumers > 0;
}

fn linearDotConsumesTranspose(graph: *const Graph, consumer_id: NodeId, transpose_id: NodeId) bool {
    if (consumer_id >= graph.nodeCount()) return false;
    const consumer = graph.node(consumer_id);
    const dot_attrs = switch (consumer.op) {
        .dot_general => |dot_general_attrs| dot_general_attrs,
        else => return false,
    };
    if (consumer.num_inputs < 2 or consumer.inputs[1] != transpose_id) return false;
    if (dot_attrs.num_contracting != 1 or dot_attrs.num_batch != 0) return false;
    if (dot_attrs.lhs_contracting[0] != 1 or dot_attrs.rhs_contracting[0] != 0) return false;
    const transpose = graph.node(transpose_id);
    if (transpose.num_inputs == 0 or transpose.inputs[0] == null_node) return false;
    const lhs_shape = graph.node(consumer.inputs[0]).output_shape;
    if (lhs_shape.rank() != 2 or consumer.output_shape.rank() != 2) return false;
    const weight_shape = graph.node(transpose.inputs[0]).output_shape;
    const in_dim = shapeDimUsize(lhs_shape, 1) orelse return false;
    const out_dim = shapeDimUsize(weight_shape, 0) orelse return false;
    const weight_in_dim = shapeDimUsize(weight_shape, 1) orelse return false;
    const output_out_dim = shapeDimUsize(consumer.output_shape, 1) orelse return false;
    if (weight_in_dim != in_dim or output_out_dim != out_dim) return false;
    return true;
}

fn transposeIsSimple2D(attrs: ml.graph.node.TransposeAttrs, input_shape: Shape) bool {
    if (input_shape.rank() != 2) return false;
    var perm_buf: [ml.graph.shape.max_rank]u8 = undefined;
    const perm = transpose_utils.effectivePerm(attrs, input_shape.rank(), &perm_buf);
    return perm.len == 2 and perm[0] == 1 and perm[1] == 0;
}

fn executeRuntimeConv1d(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    const weight = valueFor(values, inputs[1]) orelse return null;
    const bias = valueFor(values, inputs[2]) orelse return null;
    const input_shape = graph.node(inputs[0]).output_shape;
    return cb.conv1d(
        input,
        weight,
        bias,
        shapeDimUsize(input_shape, 0) orelse return null,
        shapeDimUsize(input_shape, 1) orelse return null,
        attrs.out_channels,
        shapeDimUsize(input_shape, 2) orelse return null,
        attrs.kernel_size,
        attrs.stride,
        attrs.padding,
    ) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeConv2d(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    const weight = valueFor(values, inputs[1]) orelse return null;
    const bias = valueFor(values, inputs[2]) orelse return null;
    const input_shape = graph.node(inputs[0]).output_shape;
    return cb.conv2d(
        input,
        weight,
        bias,
        shapeDimUsize(input_shape, 0) orelse return null,
        shapeDimUsize(input_shape, 1) orelse return null,
        attrs.out_channels,
        shapeDimUsize(input_shape, 2) orelse return null,
        shapeDimUsize(input_shape, 3) orelse return null,
        attrs.kernel_h,
        attrs.kernel_w,
        attrs.stride_h,
        attrs.stride_w,
        attrs.padding_h,
        attrs.padding_w,
        attrs.groups,
    ) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeConvGeneral(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    const weight = valueFor(values, inputs[1]) orelse return null;
    const input_shape = graph.node(inputs[0]).output_shape;
    const weight_shape = graph.node(inputs[1]).output_shape;
    if (attrs.num_spatial == 1 and attrs.groups == 1 and input_shape.rank() == 3 and weight_shape.rank() == 3 and attrs.padding[0][0] == attrs.padding[0][1]) {
        const out_channels = shapeDimUsize(weight_shape, 0) orelse return null;
        const bias_data = try std.heap.page_allocator.alloc(f32, out_channels);
        defer std.heap.page_allocator.free(bias_data);
        @memset(bias_data, 0.0);
        const bias = try cb.fromFloat32(bias_data);
        defer cb.free(bias);
        return cb.conv1d(
            input,
            weight,
            bias,
            shapeDimUsize(input_shape, 0) orelse return null,
            shapeDimUsize(input_shape, 1) orelse return null,
            out_channels,
            shapeDimUsize(input_shape, 2) orelse return null,
            shapeDimUsize(weight_shape, 2) orelse return null,
            std.math.cast(usize, attrs.strides[0]) orelse return null,
            std.math.cast(usize, attrs.padding[0][0]) orelse return null,
        ) catch |err| switch (err) {
            error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
            else => return err,
        };
    }
    if (attrs.num_spatial == 2 and input_shape.rank() == 4 and weight_shape.rank() == 4 and attrs.padding[0][0] == attrs.padding[0][1] and attrs.padding[1][0] == attrs.padding[1][1]) {
        const out_channels = shapeDimUsize(weight_shape, 0) orelse return null;
        const bias_data = try std.heap.page_allocator.alloc(f32, out_channels);
        defer std.heap.page_allocator.free(bias_data);
        @memset(bias_data, 0.0);
        const bias = try cb.fromFloat32(bias_data);
        defer cb.free(bias);
        return cb.conv2d(
            input,
            weight,
            bias,
            shapeDimUsize(input_shape, 0) orelse return null,
            shapeDimUsize(input_shape, 1) orelse return null,
            out_channels,
            shapeDimUsize(input_shape, 2) orelse return null,
            shapeDimUsize(input_shape, 3) orelse return null,
            shapeDimUsize(weight_shape, 2) orelse return null,
            shapeDimUsize(weight_shape, 3) orelse return null,
            std.math.cast(usize, attrs.strides[0]) orelse return null,
            std.math.cast(usize, attrs.strides[1]) orelse return null,
            std.math.cast(usize, attrs.padding[0][0]) orelse return null,
            std.math.cast(usize, attrs.padding[1][0]) orelse return null,
            std.math.cast(usize, attrs.groups) orelse return null,
        ) catch |err| switch (err) {
            error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
            else => return err,
        };
    }
    return null;
}

fn executeRuntimeLinearNoBiasPair(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
    exec_state: *interpreter.ExecState,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    const weight_a = valueFor(values, inputs[1]) orelse return null;
    const weight_b = valueFor(values, inputs[2]) orelse return null;
    const result = cb.linearNoBiasPair(
        input,
        weight_a,
        weight_b,
        attrs.rows,
        attrs.in_dim,
        attrs.out_dim,
    ) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => return null,
        else => return err,
    };
    exec_state.pair_second = result.second;
    return result.first;
}

fn validatePlannedLinearOp(
    cb: *const ComputeBackend,
    weight: CT,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    op_plan: ?OperatorPlan,
) !void {
    const plan = op_plan orelse return;
    switch (plan) {
        .quant_matmul => |quant| {
            if (quant.operator == .fallback or
                quant.rows != rows or
                quant.in_dim != in_dim or
                quant.out_dim != out_dim)
            {
                return error.InvalidPartitionPlan;
            }
            if (comptime build_options.enable_metal) {
                const storage = metal_compute_mod.MetalCompute.getQuantizedStorage(cb, weight) orelse
                    return error.InvalidPartitionPlan;
                const format = contracts.quantFormatFromGgufTensorType(storage.tensor_type) orelse
                    return error.InvalidPartitionPlan;
                if (format != quant.format) return error.InvalidPartitionPlan;
            }
        },
        else => return error.InvalidPartitionPlan,
    }
}

fn validatePlannedAttentionOp(
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    op_plan: ?OperatorPlan,
) !operator_plan_mod.AttentionOpPlan {
    const plan = op_plan orelse return error.InvalidPartitionPlan;
    switch (plan) {
        .attention => |attention| {
            if (attention.operator == .fallback or
                attention.q_len != q_len or
                attention.kv_len != kv_len or
                attention.head_dim != head_dim)
            {
                return error.InvalidPartitionPlan;
            }
            switch (attention.operator) {
                .attention_flash => {
                    if (attention.storage != .dense or attention.kv_format != .f32) return error.InvalidPartitionPlan;
                },
                .attention_paged => {
                    if (attention.storage != .paged) return error.InvalidPartitionPlan;
                },
                .attention_quantized_kv => {
                    if (attention.kv_format != .polar4 and
                        attention.kv_format != .turbo3 and
                        attention.kv_format != .quantized)
                    {
                        return error.InvalidPartitionPlan;
                    }
                },
                else => return error.InvalidPartitionPlan,
            }
            return attention;
        },
        else => return error.InvalidPartitionPlan,
    }
}

fn executeRuntimeTakeRows(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    rows: usize,
    dim: usize,
    op_plan: ?OperatorPlan,
    exec_state: *interpreter.ExecState,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    const grouped = exec_state.moe_grouped orelse return null;
    if (grouped.rows.len != rows) return error.InvalidPartitionPlan;
    try validatePlannedQuantRowOp(cb, input, rows, dim, op_plan);
    return cb.takeRows(input, grouped.rows, rows, dim) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeRope(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
    exec_state: *interpreter.ExecState,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    const rope_dim: usize = if (attrs.rope_dim > 0) attrs.rope_dim else attrs.head_dim;
    const position_offset = if (exec_state.options.attention) |attn|
        attn.total_sequence_len - attn.query_sequence_len
    else
        attrs.position_offset;
    return cb.rope(
        input,
        attrs.seq_len,
        attrs.head_dim,
        rope_dim,
        attrs.theta,
        attrs.freq_scale,
        position_offset,
        attrs.consecutive_pairs,
    ) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn validatePlannedQuantRowOp(
    cb: *const ComputeBackend,
    input: CT,
    rows: usize,
    dim: usize,
    op_plan: ?OperatorPlan,
) !void {
    const plan = op_plan orelse return error.InvalidPartitionPlan;
    switch (plan) {
        .quant_row => |row| {
            if (row.operator == .fallback or
                row.kind != .get_rows or
                row.rows != rows or
                row.dim != dim)
            {
                return error.InvalidPartitionPlan;
            }
            if (comptime build_options.enable_metal) {
                const storage = metal_compute_mod.MetalCompute.getQuantizedStorage(cb, input) orelse
                    return error.InvalidPartitionPlan;
                const format = contracts.quantFormatFromGgufTensorType(storage.tensor_type) orelse
                    return error.InvalidPartitionPlan;
                if (format != row.format) return error.InvalidPartitionPlan;
            }
        },
        else => return error.InvalidPartitionPlan,
    }
}

fn executeRuntimeLayerNorm(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    dim: usize,
    eps: f32,
    output_shape: ml.graph.Shape,
) !?CT {
    const output_elems = tensorElementCount(output_shape) orelse return null;
    if (dim == 0 or output_elems % dim != 0) return null;
    const input = valueFor(values, inputs[0]) orelse return null;
    const weight = valueFor(values, inputs[1]) orelse return null;
    const bias = valueFor(values, inputs[2]) orelse return null;
    return cb.layerNorm(input, weight, bias, dim, eps) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeRmsNorm(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    dim: usize,
    eps: f32,
    output_shape: ml.graph.Shape,
) !?CT {
    const output_elems = tensorElementCount(output_shape) orelse return null;
    if (dim == 0 or output_elems % dim != 0) return null;
    const input = valueFor(values, inputs[0]) orelse return null;
    const weight = valueFor(values, inputs[1]) orelse return null;
    return cb.rmsNorm(input, weight, dim, eps) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeActivation(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    kind: ops_mod.DecoderRuntimeActivationKind,
    output_shape: ml.graph.Shape,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    const dim = tensorElementCount(output_shape) orelse return null;
    if (try cb.decoderRuntimeApplyActivation(&.{
        .input = input,
        .kind = kind,
        .dim = dim,
    })) |result| return result;
    return switch (kind) {
        .gelu => cb.gelu(input),
        .relu => cb.relu(input),
        .silu => cb.silu(input),
        .quick_gelu => cb.quickGelu(input),
        else => return null,
    } catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeGeluBackward(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    output_shape: ml.graph.Shape,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    const upstream_grad = valueFor(values, inputs[1]) orelse return null;
    const dim = tensorElementCount(output_shape) orelse return null;
    return cb.decoderRuntimeApplyGeluBackward(&.{
        .input = input,
        .upstream_grad = upstream_grad,
        .dim = dim,
    }) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeGather(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    attrs: anytype,
) !?CT {
    if (inputs.len < 2) return null;
    var input_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
    const input_shape = try fillShapeDims(graph.node(inputs[0]).output_shape, &input_shape_buf);
    const input_value = valueFor(values, inputs[0]) orelse return null;
    const indices_value = valueFor(values, inputs[1]) orelse return null;
    const indices = try temporaryMetalResidentValue(cb, indices_value);
    defer indices.deinit(cb);

    if (attrs.axis == 0) gather_add_bias: {
        const input_node = graph.node(inputs[0]);
        if (input_node.op != .add) break :gather_add_bias;
        const add_inputs = input_node.getInputs();
        if (add_inputs.len != 2) break :gather_add_bias;

        const lhs = add_inputs[0];
        const rhs = add_inputs[1];
        if (lhs == null_node or rhs == null_node) break :gather_add_bias;

        var lhs_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
        var rhs_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
        const lhs_shape = try fillShapeDims(graph.node(lhs).output_shape, &lhs_shape_buf);
        const rhs_shape = try fillShapeDims(graph.node(rhs).output_shape, &rhs_shape_buf);

        var matrix_id: NodeId = null_node;
        var bias_id: NodeId = null_node;
        var matrix_shape_buf: [ml.graph.shape.max_rank]i64 = undefined;
        var matrix_shape: []const i64 = &.{};
        if (lhs_shape.len == 2 and rhs_shape.len == 1 and lhs_shape[1] == rhs_shape[0]) {
            matrix_id = lhs;
            bias_id = rhs;
            @memcpy(matrix_shape_buf[0..lhs_shape.len], lhs_shape);
            matrix_shape = matrix_shape_buf[0..lhs_shape.len];
        } else if (rhs_shape.len == 2 and lhs_shape.len == 1 and rhs_shape[1] == lhs_shape[0]) {
            matrix_id = rhs;
            bias_id = lhs;
            @memcpy(matrix_shape_buf[0..rhs_shape.len], rhs_shape);
            matrix_shape = matrix_shape_buf[0..rhs_shape.len];
        } else {
            break :gather_add_bias;
        }

        const matrix = valueFor(values, matrix_id) orelse break :gather_add_bias;
        const bias = valueFor(values, bias_id) orelse break :gather_add_bias;
        const fused = cb.primGatherAddBiasAxis0(matrix, bias, indices.value, matrix_shape) catch |err| switch (err) {
            error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedTensorType, error.UnsupportedShape, error.ShapeMismatch, error.InvalidTensorShape => null,
            else => return err,
        };
        if (fused) |output| return output;
    }

    return cb.primGather(input_value, indices.value, attrs.axis, input_shape) catch |err| switch (err) {
        error.UnsupportedOperation, error.UnsupportedPrimitiveOp, error.UnsupportedTensorType, error.UnsupportedShape, error.ShapeMismatch, error.InvalidTensorShape => null,
        else => return err,
    };
}

fn executeRuntimeAdd(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    output_shape: ml.graph.Shape,
) !?CT {
    const dim = tensorElementCount(output_shape) orelse return null;
    if (try executeRuntimeScaledAddFromDeferredMul(graph, cb, values, inputs, output_shape, dim)) |result| return result;
    if (try executeRuntimeMultiplyAddFromDeferredMul(graph, cb, values, inputs, output_shape, dim)) |result| return result;
    const lhs = valueFor(values, inputs[0]) orelse return null;
    const rhs = valueFor(values, inputs[1]) orelse return null;
    if (isMetalDeviceResident(cb, lhs) or isMetalDeviceResident(cb, rhs)) {
        return cb.add(lhs, rhs) catch |err| switch (err) {
            error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
            else => return err,
        };
    }
    if (try cb.decoderRuntimeApplyAdd(&.{ .lhs = lhs, .rhs = rhs, .dim = dim })) |result| return result;
    return cb.add(lhs, rhs) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

const DeferredScaleMul = struct {
    source_id: NodeId,
    scalar_id: NodeId,
    scale: f32,
};

const DeferredElementwiseMul = struct {
    lhs_id: NodeId,
    rhs_id: NodeId,
};

fn executeRuntimeScaledAddFromDeferredMul(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    output_shape: ml.graph.Shape,
    dim: usize,
) !?CT {
    if (inputs.len < 2) return null;
    if (deferredScaleMul(graph, inputs[0], output_shape)) |lhs_scaled| {
        if (deferredScaleMul(graph, inputs[1], output_shape)) |rhs_scaled| {
            const lhs = (try executeDeferredScaleMulValue(cb, values, lhs_scaled)) orelse return null;
            defer cb.free(lhs);
            const rhs = (try executeDeferredScaleMulValue(cb, values, rhs_scaled)) orelse return null;
            defer cb.free(rhs);
            const output = cb.add(lhs, rhs) catch |err| switch (err) {
                error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
                else => return err,
            };
            return try promoteMetalOutputIfNeeded(cb, output);
        }
    }
    if (deferredScaleMul(graph, inputs[0], output_shape)) |scaled| {
        const scaled_lhs = valueFor(values, scaled.source_id) orelse return null;
        const residual = valueFor(values, inputs[1]) orelse return null;
        const output = cb.decoderRuntimeApplyScaledAddScale(&.{
            .lhs = scaled_lhs,
            .rhs = residual,
            .dim = dim,
            .lhs_scale = scaled.scale,
            .output_scale = 1.0,
        }) catch |err| switch (err) {
            error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
            else => return err,
        };
        if (try promoteMetalOutputIfNeeded(cb, output)) |result| return result;
        return try executeDeferredScaleAddFallback(cb, values, scaled, residual);
    }
    if (deferredScaleMul(graph, inputs[1], output_shape)) |scaled| {
        const scaled_lhs = valueFor(values, scaled.source_id) orelse return null;
        const residual = valueFor(values, inputs[0]) orelse return null;
        const output = cb.decoderRuntimeApplyScaledAddScale(&.{
            .lhs = scaled_lhs,
            .rhs = residual,
            .dim = dim,
            .lhs_scale = scaled.scale,
            .output_scale = 1.0,
        }) catch |err| switch (err) {
            error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
            else => return err,
        };
        if (try promoteMetalOutputIfNeeded(cb, output)) |result| return result;
        return try executeDeferredScaleAddFallback(cb, values, scaled, residual);
    }
    return null;
}

fn executeRuntimeMultiplyAddFromDeferredMul(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    output_shape: ml.graph.Shape,
    dim: usize,
) !?CT {
    if (inputs.len < 2) return null;
    if (deferredElementwiseMul(graph, inputs[0], output_shape)) |mul| {
        if (deferredElementwiseMul(graph, inputs[1], output_shape)) |rhs_mul| {
            return try executeDeferredElementwiseMultiplyAdd2(cb, values, mul, rhs_mul, dim);
        }
        const addend = valueFor(values, inputs[1]) orelse return null;
        return try executeDeferredElementwiseMultiplyAdd(cb, values, mul, addend, dim);
    }
    if (deferredElementwiseMul(graph, inputs[1], output_shape)) |mul| {
        const addend = valueFor(values, inputs[0]) orelse return null;
        return try executeDeferredElementwiseMultiplyAdd(cb, values, mul, addend, dim);
    }
    return null;
}

fn executeDeferredElementwiseMultiplyAdd2(
    cb: *const ComputeBackend,
    values: []?CT,
    lhs_mul: DeferredElementwiseMul,
    rhs_mul: DeferredElementwiseMul,
    dim: usize,
) !?CT {
    var lhs0 = valueFor(values, lhs_mul.lhs_id) orelse return null;
    var rhs0 = valueFor(values, lhs_mul.rhs_id) orelse return null;
    var lhs1 = valueFor(values, rhs_mul.lhs_id) orelse return null;
    var rhs1 = valueFor(values, rhs_mul.rhs_id) orelse return null;
    var owned_lhs0: ?CT = null;
    defer if (owned_lhs0) |ct| cb.free(ct);
    var owned_rhs0: ?CT = null;
    defer if (owned_rhs0) |ct| cb.free(ct);
    var owned_lhs1: ?CT = null;
    defer if (owned_lhs1) |ct| cb.free(ct);
    var owned_rhs1: ?CT = null;
    defer if (owned_rhs1) |ct| cb.free(ct);
    if (cb.kind() == .metal) {
        if (!isMetalDeviceResident(cb, lhs0)) {
            if (try makeMetalDeviceResident(cb, lhs0)) |device_lhs0| {
                if (device_lhs0 != lhs0) owned_lhs0 = device_lhs0;
                lhs0 = device_lhs0;
            }
        }
        if (!isMetalDeviceResident(cb, rhs0)) {
            if (try makeMetalDeviceResident(cb, rhs0)) |device_rhs0| {
                if (device_rhs0 != rhs0) owned_rhs0 = device_rhs0;
                rhs0 = device_rhs0;
            }
        }
        if (!isMetalDeviceResident(cb, lhs1)) {
            if (try makeMetalDeviceResident(cb, lhs1)) |device_lhs1| {
                if (device_lhs1 != lhs1) owned_lhs1 = device_lhs1;
                lhs1 = device_lhs1;
            }
        }
        if (!isMetalDeviceResident(cb, rhs1)) {
            if (try makeMetalDeviceResident(cb, rhs1)) |device_rhs1| {
                if (device_rhs1 != rhs1) owned_rhs1 = device_rhs1;
                rhs1 = device_rhs1;
            }
        }
    }
    const output = cb.decoderRuntimeApplyMultiplyAdd2(&.{
        .lhs0 = lhs0,
        .rhs0 = rhs0,
        .lhs1 = lhs1,
        .rhs1 = rhs1,
        .dim = dim,
    }) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
    if (try promoteMetalOutputIfNeeded(cb, output)) |result| return result;

    const lhs_product = cb.multiply(lhs0, rhs0) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => return null,
        else => return err,
    };
    defer cb.free(lhs_product);
    const rhs_product = cb.multiply(lhs1, rhs1) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => return null,
        else => return err,
    };
    defer cb.free(rhs_product);
    const fallback = cb.add(lhs_product, rhs_product) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => return null,
        else => return err,
    };
    return try promoteMetalOutputIfNeeded(cb, fallback);
}

fn executeDeferredElementwiseMultiplyAdd(
    cb: *const ComputeBackend,
    values: []?CT,
    mul: DeferredElementwiseMul,
    addend: CT,
    dim: usize,
) !?CT {
    var lhs = valueFor(values, mul.lhs_id) orelse return null;
    var rhs = valueFor(values, mul.rhs_id) orelse return null;
    var residual = addend;
    var owned_lhs: ?CT = null;
    defer if (owned_lhs) |ct| cb.free(ct);
    var owned_rhs: ?CT = null;
    defer if (owned_rhs) |ct| cb.free(ct);
    var owned_residual: ?CT = null;
    defer if (owned_residual) |ct| cb.free(ct);
    if (cb.kind() == .metal) {
        if (!isMetalDeviceResident(cb, lhs)) {
            if (try makeMetalDeviceResident(cb, lhs)) |device_lhs| {
                if (device_lhs != lhs) owned_lhs = device_lhs;
                lhs = device_lhs;
            }
        }
        if (!isMetalDeviceResident(cb, rhs)) {
            if (try makeMetalDeviceResident(cb, rhs)) |device_rhs| {
                if (device_rhs != rhs) owned_rhs = device_rhs;
                rhs = device_rhs;
            }
        }
        if (!isMetalDeviceResident(cb, residual)) {
            if (try makeMetalDeviceResident(cb, residual)) |device_residual| {
                if (device_residual != residual) owned_residual = device_residual;
                residual = device_residual;
            }
        }
    }
    const output = cb.decoderRuntimeApplyMultiplyAdd(&.{
        .lhs = lhs,
        .rhs = rhs,
        .addend = residual,
        .dim = dim,
    }) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
    if (try promoteMetalOutputIfNeeded(cb, output)) |result| return result;

    const multiplied = cb.multiply(lhs, rhs) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => return null,
        else => return err,
    };
    defer cb.free(multiplied);
    const fallback = cb.add(multiplied, residual) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => return null,
        else => return err,
    };
    return try promoteMetalOutputIfNeeded(cb, fallback);
}

fn executeDeferredScaleMulValue(
    cb: *const ComputeBackend,
    values: []?CT,
    scaled: DeferredScaleMul,
) !?CT {
    var lhs = valueFor(values, scaled.source_id) orelse return null;
    var scalar = valueFor(values, scaled.scalar_id) orelse return null;
    var owned_lhs: ?CT = null;
    defer if (owned_lhs) |ct| cb.free(ct);
    var owned_scalar: ?CT = null;
    defer if (owned_scalar) |ct| cb.free(ct);
    if (cb.kind() == .metal) {
        if (!isMetalDeviceResident(cb, lhs)) {
            if (try makeMetalDeviceResident(cb, lhs)) |device_lhs| {
                if (device_lhs != lhs) owned_lhs = device_lhs;
                lhs = device_lhs;
            }
        }
        if (!isMetalDeviceResident(cb, scalar)) {
            if (try makeMetalDeviceResident(cb, scalar)) |device_scalar| {
                if (device_scalar != scalar) owned_scalar = device_scalar;
                scalar = device_scalar;
            }
        }
    }
    return cb.multiply(lhs, scalar) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeDeferredScaleAddFallback(
    cb: *const ComputeBackend,
    values: []?CT,
    scaled: DeferredScaleMul,
    residual: CT,
) !?CT {
    var rhs = residual;
    var owned_rhs: ?CT = null;
    defer if (owned_rhs) |ct| cb.free(ct);
    if (cb.kind() == .metal) {
        if (!isMetalDeviceResident(cb, rhs)) {
            if (try makeMetalDeviceResident(cb, rhs)) |device_rhs| {
                if (device_rhs != rhs) owned_rhs = device_rhs;
                rhs = device_rhs;
            }
        }
    }
    const multiplied_ct = (try executeDeferredScaleMulValue(cb, values, scaled)) orelse return null;
    defer cb.free(multiplied_ct);
    const output = cb.add(multiplied_ct, rhs) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
    return try promoteMetalOutputIfNeeded(cb, output);
}

fn deferredScaleMul(graph: *const Graph, node_id: NodeId, output_shape: ml.graph.Shape) ?DeferredScaleMul {
    if (node_id == null_node or node_id >= graph.nodeCount()) return null;
    const node = graph.node(node_id);
    switch (node.op) {
        .mul, .fused_elem_multiply => {},
        else => return null,
    }
    if (!shapesEqual(node.output_shape, output_shape)) return null;
    if (node.num_inputs < 2) return null;
    if (scalarConstantF32(graph, node.inputs[0])) |scale| {
        return .{ .source_id = node.inputs[1], .scalar_id = node.inputs[0], .scale = scale };
    }
    if (scalarConstantF32(graph, node.inputs[1])) |scale| {
        return .{ .source_id = node.inputs[0], .scalar_id = node.inputs[1], .scale = scale };
    }
    return null;
}

fn deferredElementwiseMul(graph: *const Graph, node_id: NodeId, output_shape: ml.graph.Shape) ?DeferredElementwiseMul {
    if (node_id == null_node or node_id >= graph.nodeCount()) return null;
    const node = graph.node(node_id);
    switch (node.op) {
        .mul, .fused_elem_multiply => {},
        else => return null,
    }
    if (!shapesEqual(node.output_shape, output_shape)) return null;
    if (node.output_shape.dtype != .f32 or node.num_inputs < 2) return null;
    if (deferredScaleMul(graph, node_id, output_shape) != null) return null;
    const lhs_id = node.inputs[0];
    const rhs_id = node.inputs[1];
    if (lhs_id == null_node or rhs_id == null_node or lhs_id >= graph.nodeCount() or rhs_id >= graph.nodeCount()) return null;
    if (!shapesEqual(graph.node(lhs_id).output_shape, output_shape)) return null;
    if (!shapesEqual(graph.node(rhs_id).output_shape, output_shape)) return null;
    if (isSameShapeElementwiseMul(graph, lhs_id, output_shape)) return null;
    if (isSameShapeElementwiseMul(graph, rhs_id, output_shape)) return null;
    return .{ .lhs_id = lhs_id, .rhs_id = rhs_id };
}

fn isSameShapeElementwiseMul(graph: *const Graph, node_id: NodeId, output_shape: ml.graph.Shape) bool {
    if (node_id == null_node or node_id >= graph.nodeCount()) return false;
    const node = graph.node(node_id);
    switch (node.op) {
        .mul, .fused_elem_multiply => {},
        else => return false,
    }
    return shapesEqual(node.output_shape, output_shape);
}

fn shouldDeferScaleMulForAdd(
    graph: *const Graph,
    node_id: NodeId,
    reachable: []const bool,
    last_use: []const u32,
) bool {
    const node_index: usize = @intCast(node_id);
    if (node_index >= reachable.len or !reachable[node_index]) return false;
    if (node_index >= last_use.len) return false;
    const consumer_id: NodeId = @intCast(last_use[node_index]);
    if (consumer_id == null_node or consumer_id >= graph.nodeCount()) return false;
    const consumer_index: usize = @intCast(consumer_id);
    if (consumer_index >= reachable.len or !reachable[consumer_index]) return false;
    const consumer = graph.node(consumer_id);
    switch (consumer.op) {
        .add, .fused_elem_add => {},
        else => return false,
    }
    if (consumer.num_inputs < 2 or (consumer.inputs[0] != node_id and consumer.inputs[1] != node_id)) return false;
    if (reachableUseCount(graph, node_id, reachable, 2) != 1) return false;
    return deferredScaleMul(graph, node_id, consumer.output_shape) != null;
}

fn shouldDeferElementwiseMulForAdd(
    graph: *const Graph,
    node_id: NodeId,
    reachable: []const bool,
    last_use: []const u32,
) bool {
    const node_index: usize = @intCast(node_id);
    if (node_index >= reachable.len or !reachable[node_index]) return false;
    if (node_index >= last_use.len) return false;
    const consumer_id: NodeId = @intCast(last_use[node_index]);
    if (consumer_id == null_node or consumer_id >= graph.nodeCount()) return false;
    const consumer_index: usize = @intCast(consumer_id);
    if (consumer_index >= reachable.len or !reachable[consumer_index]) return false;
    const consumer = graph.node(consumer_id);
    switch (consumer.op) {
        .add, .fused_elem_add => {},
        else => return false,
    }
    if (consumer.num_inputs < 2 or (consumer.inputs[0] != node_id and consumer.inputs[1] != node_id)) return false;
    if (reachableUseCount(graph, node_id, reachable, 2) != 1) return false;
    if (deferredElementwiseMul(graph, node_id, consumer.output_shape) == null) return false;
    return true;
}

fn reachableUseCount(graph: *const Graph, node_id: NodeId, reachable: []const bool, stop_after: usize) usize {
    var count: usize = 0;
    var candidate: NodeId = 0;
    while (candidate < graph.nodeCount()) : (candidate += 1) {
        const candidate_index: usize = @intCast(candidate);
        if (candidate_index >= reachable.len or !reachable[candidate_index]) continue;
        const node = graph.node(candidate);
        for (node.getInputs()) |input_id| {
            if (input_id != node_id) continue;
            count += 1;
            if (count >= stop_after) return count;
        }
    }
    return count;
}

fn scalarConstantF32(graph: *const Graph, node_id: NodeId) ?f32 {
    if (node_id == null_node or node_id >= graph.nodeCount()) return null;
    const node = graph.node(node_id);
    const attrs = switch (node.op) {
        .constant => |constant_attrs| constant_attrs,
        else => return null,
    };
    if (node.output_shape.dtype != .f32 or node.output_shape.rank() != 0 or attrs.data_len != 1) return null;
    const values = graph.constantDataAs(f32, attrs.data_offset, attrs.data_len);
    if (values.len != 1) return null;
    return values[0];
}

fn shapesEqual(lhs: ml.graph.Shape, rhs: ml.graph.Shape) bool {
    if (lhs.dtype != rhs.dtype or lhs.rank() != rhs.rank()) return false;
    for (0..lhs.rank()) |idx| {
        if (lhs.dims[idx] != rhs.dims[idx]) return false;
    }
    return true;
}

fn executeRuntimePlainAdd(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
) !?CT {
    const lhs = valueFor(values, inputs[0]) orelse return null;
    const rhs = valueFor(values, inputs[1]) orelse return null;
    return cb.add(lhs, rhs) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeSoftmax(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    dim: u32,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    return cb.primSoftmax(input, dim) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn executeRuntimeLogSoftmax(
    cb: *const ComputeBackend,
    values: []?CT,
    inputs: []const NodeId,
    dim: u32,
) !?CT {
    const input = valueFor(values, inputs[0]) orelse return null;
    return cb.primLogSoftmax(input, dim) catch |err| switch (err) {
        error.UnsupportedPrimitiveOp, error.UnsupportedShape, error.ShapeMismatch => null,
        else => return err,
    };
}

fn valueFor(values: []?CT, node_id: NodeId) ?CT {
    if (node_id == null_node) return null;
    const index: usize = @intCast(node_id);
    if (index >= values.len) return null;
    return values[index];
}

fn fillShapeDims(shape: ml.graph.Shape, buf: *[ml.graph.shape.max_rank]i64) ![]const i64 {
    const rank = shape.rank();
    if (rank > buf.len) return error.UnsupportedShape;
    for (0..rank) |axis| buf[axis] = shape.dim(@intCast(axis));
    return buf[0..rank];
}

fn tensorElementCount(shape: ml.graph.Shape) ?usize {
    const elems = shape.maxElements() orelse shape.numElements() orelse return null;
    if (elems <= 0) return null;
    return @intCast(elems);
}

fn positiveI64ToUsize(dim: i64) ?usize {
    if (dim <= 0) return null;
    return std.math.cast(usize, dim);
}

fn shapeDimUsize(shape: ml.graph.Shape, axis: usize) ?usize {
    if (axis >= shape.rank()) return null;
    return positiveI64ToUsize(shape.dim(@intCast(axis)));
}

fn shapeRankForNodeOr(graph: *const Graph, node_id: NodeId, fallback: usize) usize {
    if (node_id == null_node or node_id >= graph.nodeCount()) return fallback;
    return graph.node(node_id).output_shape.rank();
}

fn shapeDimForNodeOr(graph: *const Graph, node_id: NodeId, axis: usize, fallback: i64) i64 {
    if (node_id == null_node or node_id >= graph.nodeCount()) return fallback;
    const shape = graph.node(node_id).output_shape;
    if (axis >= shape.rank()) return fallback;
    return shape.dim(@intCast(axis));
}

fn buildMetalGraphPlan(
    allocator: std.mem.Allocator,
    buffer_plan: *const buffer_plan_mod.BufferPlan,
    view: buffer_plan_mod.PartitionBufferView,
) !MetalPartitionGraphPlan {
    var mappings = std.ArrayListUnmanaged(MetalGraphPlanAllocation).empty;
    errdefer mappings.deinit(allocator);

    for (view.slots) |slot_view| {
        if (!slot_view.roles.local) continue;
        const allocation_id = slot_view.slot.allocation;
        if (allocation_id == buffer_plan_mod.invalid_allocation) continue;
        const allocation = buffer_plan.allocations[@intCast(allocation_id)];
        if (allocation.kind != .tensor) continue;
        try addGraphPlanAllocation(allocator, &mappings, allocation_id, allocation.byte_size);
    }

    const slots = try allocator.alloc(GraphPlanSlot, mappings.items.len);
    errdefer allocator.free(slots);
    for (mappings.items, slots) |mapping, *slot| {
        slot.* = .{ .slot = mapping.graph_slot, .bytes = mapping.bytes };
    }

    return .{
        .slots = slots,
        .allocations = try mappings.toOwnedSlice(allocator),
    };
}

fn addGraphPlanAllocation(
    allocator: std.mem.Allocator,
    mappings: *std.ArrayListUnmanaged(MetalGraphPlanAllocation),
    allocation_id: buffer_plan_mod.AllocationId,
    bytes_u64: u64,
) !void {
    const bytes: usize = std.math.cast(usize, bytes_u64) orelse return error.OutOfMemory;
    if (bytes == 0) return;
    for (mappings.items) |*mapping| {
        if (mapping.allocation != allocation_id) continue;
        mapping.bytes = @max(mapping.bytes, bytes);
        return;
    }
    if (mappings.items.len >= max_graph_plan_slots) {
        var smallest_idx: usize = 0;
        for (mappings.items[1..], 1..) |mapping, idx| {
            if (mapping.bytes < mappings.items[smallest_idx].bytes) smallest_idx = idx;
        }
        if (bytes <= mappings.items[smallest_idx].bytes) return;
        mappings.items[smallest_idx] = .{
            .allocation = allocation_id,
            .graph_slot = smallest_idx,
            .bytes = bytes,
        };
        return;
    }
    const graph_slot = mappings.items.len;
    try mappings.append(allocator, .{
        .allocation = allocation_id,
        .graph_slot = graph_slot,
        .bytes = bytes,
    });
}

fn materializePartitionRuntimeInputs(
    allocator: std.mem.Allocator,
    values: []?CT,
    value_device: []DeviceId,
    node_ids: []const NodeId,
    device_id: DeviceId,
    exec_ctx: PartitionExecutor.ExecutionContext,
    cb: *const ComputeBackend,
    rt_map: std.AutoHashMapUnmanaged(NodeId, CT),
) !void {
    const trace_nodes = traceMetalGraphNodesEnabled();
    for (node_ids) |node_id| {
        const rt_val = rt_map.get(node_id) orelse continue;
        const i: usize = @intCast(node_id);
        const current_dev = value_device[i];
        if (trace_nodes) std.debug.print(
            "graph_executor_node_trace: materialize_runtime_input node={d} current_device={d} target_device={d} resident={}\n",
            .{ node_id, current_dev, device_id, isMetalDeviceResident(cb, rt_val) },
        );
        if (current_dev != device_id) {
            const mesh = exec_ctx.mesh orelse return error.DeviceNotFound;
            const src_entry = mesh.device(current_dev) orelse return error.DeviceNotFound;
            if (trace_nodes) std.debug.print(
                "graph_executor_node_trace: materialize_runtime_input_transfer node={d} from_backend={s} to_backend={s}\n",
                .{ node_id, @tagName(src_entry.backend.kind()), @tagName(cb.kind()) },
            );
            const transferred = try transferTensor(allocator, rt_val, src_entry.backend, cb);
            values[i] = transferred;
            if (exec_ctx.owned_runtime_transfers) |owned| try owned.put(allocator, node_id, {});
            if (exec_ctx.stats) |stats| {
                stats.runtime_input_transfers += 1;
                if (isMetalDeviceResident(cb, transferred)) stats.device_resident_transfers += 1;
            }
        } else {
            values[i] = rt_val;
        }
        const preserve_runtime_input_residency = if (exec_ctx.options) |options| options.preserve_runtime_input_residency else false;
        if (!preserve_runtime_input_residency and values[i] != null) {
            const current = values[i].?;
            if (!isMetalDeviceResident(cb, current)) {
                if (trace_nodes) std.debug.print(
                    "graph_executor_node_trace: materialize_runtime_input_make_resident node={d}\n",
                    .{node_id},
                );
                if (try makeMetalDeviceResident(cb, current)) |device_value| {
                    if (device_value != current) {
                        values[i] = device_value;
                        if (exec_ctx.owned_runtime_transfers) |owned| try owned.put(allocator, node_id, {});
                    }
                    if (exec_ctx.stats) |stats| stats.device_resident_transfers += 1;
                }
                if (trace_nodes) std.debug.print(
                    "graph_executor_node_trace: materialize_runtime_input_make_resident_done node={d} resident={}\n",
                    .{ node_id, if (values[i]) |updated| isMetalDeviceResident(cb, updated) else false },
                );
            }
        }
        value_device[i] = device_id;
    }
}

fn isPreMaterializedConstantOp(op: ml.graph.OpCode) bool {
    return switch (op) {
        .constant, .fused_zero_tensor => true,
        else => false,
    };
}

fn materializePartitionParameters(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    node_ids: []const NodeId,
    reachable: []const bool,
    device_id: DeviceId,
    rt_map: std.AutoHashMapUnmanaged(NodeId, CT),
    stats: ?*PartitionExecutor.ExecutionStats,
) !void {
    for (node_ids) |node_id| {
        const i: usize = @intCast(node_id);
        if (i >= reachable.len or !reachable[i]) continue;
        if (values[i] != null) continue;
        if (rt_map.contains(node_id)) continue;
        const node = graph.node(node_id);
        if (node.op != .parameter) continue;
        const materialized = try cb.getWeight(graph.parameterName(node));
        values[i] = materialized;
        value_device[i] = device_id;
        if (stats) |s| {
            s.descriptor_materializations += 1;
            if (isMetalResidentOrQuantizedDescriptor(cb, materialized)) {
                s.device_resident_outputs += 1;
            } else {
                s.host_materialized_outputs += 1;
                if (traceMetalHostOutputsEnabled()) traceMetalHostOutput(graph, node_id, "parameter_materialization_host_output");
            }
        }
    }
}

fn materializePartitionConstants(
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    node_ids: []const NodeId,
    reachable: []const bool,
    device_id: DeviceId,
) !void {
    for (node_ids) |node_id| {
        const i: usize = @intCast(node_id);
        if (i >= reachable.len or !reachable[i]) continue;
        if (values[i] != null) continue;
        const node = graph.node(node_id);
        const materialized = switch (node.op) {
            .constant => |attrs| try executeRuntimeConstant(graph, cb, node.output_shape, attrs),
            .fused_zero_tensor => |attrs| try executeRuntimeZeroTensor(cb, attrs.rows, attrs.out_dim),
            else => null,
        } orelse continue;
        values[i] = materialized;
        value_device[i] = device_id;
    }
}

fn freeExpiredInputs(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    cb: *const ComputeBackend,
    values: []?CT,
    value_device: []DeviceId,
    node_id: NodeId,
    device_id: DeviceId,
    last_use: []const u32,
    runtime_region_plan: ?RuntimeRegionPlan,
    rt_map: std.AutoHashMapUnmanaged(NodeId, CT),
    donated: std.AutoHashMapUnmanaged(NodeId, void),
    exec_ctx: PartitionExecutor.ExecutionContext,
) !void {
    const n = graph.node(node_id);
    const node_index: usize = @intCast(node_id);
    var released = std.AutoHashMapUnmanaged(usize, void).empty;
    defer released.deinit(allocator);
    for (n.getInputs()) |input_id| {
        if (input_id == null_node or input_id >= values.len) continue;
        const input_index: usize = @intCast(input_id);
        if (last_use[input_index] != node_index) continue;
        if (runtime_region_plan) |plan| {
            if (plan.needsAttentionInputAfterNode(input_id, node_id)) continue;
        }
        if (rt_map.contains(input_id) and
            !donated.contains(input_id) and
            !ownedRuntimeTransferContains(exec_ctx, input_id)) continue;
        const ct = values[input_index] orelse continue;
        if (values[node_index]) |out_ct| {
            if (ct == out_ct and interpreter.canKeepAliasedOutput(n.op)) {
                values[input_index] = null;
                continue;
            }
        }
        const ct_key = @intFromPtr(ct);
        if (released.contains(ct_key)) {
            values[input_index] = null;
            continue;
        }
        try released.put(allocator, ct_key, {});
        if (exec_ctx.mesh) |mesh| {
            const inp_dev = value_device[input_index];
            if (mesh.device(inp_dev)) |entry| {
                entry.backend.free(ct);
            } else {
                cb.free(ct);
            }
        } else if (value_device[input_index] == device_id) {
            cb.free(ct);
        } else {
            cb.free(ct);
        }
        values[input_index] = null;
    }
}

fn evalPartitionBoundaryOutputs(
    cb: *const ComputeBackend,
    values: []?CT,
    view: buffer_plan_mod.PartitionBufferView,
) !void {
    for (view.slots) |slot_view| {
        if (!slot_view.roles.output and !slot_view.roles.graph_output) continue;
        const index: usize = @intCast(slot_view.slot.node_id);
        if (index >= values.len) return error.InvalidBufferPlan;
        if (values[index]) |ct| try cb.evalTensor(ct);
    }
}

fn countPartitionBoundaryOutputs(view: buffer_plan_mod.PartitionBufferView) u64 {
    var count: u64 = 0;
    for (view.slots) |slot_view| {
        if (slot_view.roles.output or slot_view.roles.graph_output) count += 1;
    }
    return count;
}

fn ownedRuntimeTransferContains(
    exec_ctx: PartitionExecutor.ExecutionContext,
    node_id: NodeId,
) bool {
    const owned = exec_ctx.owned_runtime_transfers orelse return false;
    return owned.contains(node_id);
}

fn transferTensor(
    allocator: std.mem.Allocator,
    value: CT,
    from: *const ComputeBackend,
    to: *const ComputeBackend,
) !CT {
    const shape_i64 = try from.tensorShape(value, allocator);
    defer allocator.free(shape_i64);
    const shape_i32 = try tensorShapeI32(allocator, shape_i64);
    defer allocator.free(shape_i32);
    const f32_data = try from.toFloat32(value, allocator);
    defer allocator.free(f32_data);
    const transferred = try to.fromFloat32Shape(f32_data, shape_i32);
    errdefer to.free(transferred);
    if (try makeMetalDeviceResident(to, transferred)) |device_transferred| {
        to.free(transferred);
        return device_transferred;
    }
    return transferred;
}

fn tensorShapeI32(allocator: std.mem.Allocator, shape: []const i64) ![]i32 {
    const out = try allocator.alloc(i32, shape.len);
    errdefer allocator.free(out);
    for (shape, 0..) |dim, i| {
        out[i] = std.math.cast(i32, dim) orelse return error.UnsupportedShape;
    }
    return out;
}

const native_compute = @import("../ops/native_compute.zig");

fn deinitEmptyNativeWeightStore(weight_store: *native_compute.WeightStore, allocator: std.mem.Allocator) void {
    native_compute.deinitPrefetchQueue(weight_store);
    weight_store.resident_weights.deinit(allocator);
    weight_store.lazy_weights.deinit(allocator);
}

fn initEmptyMetalWeightStore(allocator: std.mem.Allocator) gpu_hosted_store_mod.WeightStore {
    if (comptime build_options.enable_mlx) {
        return .{
            .allocator = allocator,
            .resident_weights = .{},
            .stream = .{},
            .prefix = "",
            .lazy_weights = .empty,
        };
    }
    return .{
        .allocator = allocator,
        .resident_weights = {},
        .stream = {},
        .prefix = "",
        .lazy_weights = .empty,
    };
}

fn deinitEmptyMetalWeightStore(weight_store: *gpu_hosted_store_mod.WeightStore, allocator: std.mem.Allocator) void {
    metal_compute_mod.deinitPrefetchQueue(weight_store);
    weight_store.lazy_weights.deinit(allocator);
}

test "metal partition executor consumes buffer plan and evaluates partition" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{4}));
    const out = try b.gelu(x);
    try g.markOutput(out);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &partition_mod.supportsAll },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();
    var partition_view = try buffer_plan.partitionView(allocator, &partition_plan, 0);
    defer partition_view.deinit(allocator);
    var graph_plan = try buildMetalGraphPlan(allocator, &buffer_plan, partition_view);
    defer graph_plan.deinit(allocator);
    try std.testing.expect(graph_plan.slots.len > 0);
    try std.testing.expect(graph_plan.slots.len <= max_graph_plan_slots);
    for (graph_plan.slots) |slot| {
        try std.testing.expect(slot.slot < max_graph_plan_slots);
        try std.testing.expect(slot.bytes >= 16);
    }

    var weight_store = native_compute.WeightStore{ .allocator = allocator, .resident_weights = .{}, .lazy_weights = .{} };
    defer deinitEmptyNativeWeightStore(&weight_store, allocator);
    var compute = native_compute.NativeCompute.init(allocator, &weight_store, null);
    var cb = compute.computeBackend();

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const input_data = [_]f32{ -1.0, 0.0, 1.0, 2.0 };
    const input_ct = try cb.fromFloat32Shape(&input_data, &.{4});
    defer cb.free(input_ct);
    values[@intCast(x)] = input_ct;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    const partition_index = partition_plan.node_assignment[out];
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    cb.resetDebugTimingStats();
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .options = .{
            .runtime_inputs = &.{.{ .node_id = x, .value = input_ct }},
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
    });

    const raw = try cb.toFloat32(values[@intCast(out)].?, allocator);
    defer allocator.free(raw);
    defer cb.free(values[@intCast(out)].?);
    try std.testing.expect(raw[0] < 0.0);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), raw[1], 1e-6);
    try std.testing.expect(raw[2] > 0.8 and raw[2] < 0.9);
    try std.testing.expect(raw[3] > 1.9 and raw[3] < 2.0);
}

test "metal partition executor command path handles add softmax and reshape" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ 1, 4 }));
    const y = try b.parameter("y", ml.graph.Shape.init(.f32, &.{ 1, 4 }));
    const sum = try b.add(x, y);
    const probs = try b.softmax(sum);
    const out = try b.reshape(probs, ml.graph.Shape.init(.f32, &.{4}));
    try g.markOutput(out);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &partition_mod.supportsAll },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = native_compute.WeightStore{ .allocator = allocator, .resident_weights = .{}, .lazy_weights = .{} };
    defer deinitEmptyNativeWeightStore(&weight_store, allocator);
    var compute = native_compute.NativeCompute.init(allocator, &weight_store, null);
    var cb = compute.computeBackend();

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const x_ct = try cb.fromFloat32Shape(&.{ 1.0, 2.0, 3.0, 4.0 }, &.{ 1, 4 });
    defer cb.free(x_ct);
    const y_ct = try cb.fromFloat32Shape(&.{ 0.5, 0.5, 0.5, 0.5 }, &.{ 1, 4 });
    defer cb.free(y_ct);
    values[@intCast(x)] = x_ct;
    values[@intCast(y)] = y_ct;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    const partition_index = partition_plan.node_assignment[out];
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .options = .{
            .runtime_inputs = &.{
                .{ .node_id = x, .value = x_ct },
                .{ .node_id = y, .value = y_ct },
            },
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
    });

    const raw = try cb.toFloat32(values[@intCast(out)].?, allocator);
    defer allocator.free(raw);
    defer cb.free(values[@intCast(out)].?);
    try std.testing.expectEqual(@as(usize, 4), raw.len);
    var total: f32 = 0;
    for (raw) |v| total += v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), total, 1e-5);
    try std.testing.expect(raw[3] > raw[2] and raw[2] > raw[1] and raw[1] > raw[0]);
}

test "metal partition executor command path handles linear and norms" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);
    const hidden: usize = 16;

    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ 1, hidden }));
    const w = try b.parameter("w", ml.graph.Shape.init(.f32, &.{ hidden, hidden }));
    const bias = try b.parameter("bias", ml.graph.Shape.init(.f32, &.{hidden}));
    const gamma = try b.parameter("gamma", ml.graph.Shape.init(.f32, &.{hidden}));
    const beta = try b.parameter("beta", ml.graph.Shape.init(.f32, &.{hidden}));
    const rms_weight = try b.parameter("rms_weight", ml.graph.Shape.init(.f32, &.{hidden}));
    const lin = try b.linear(x, w, bias, 1, hidden, hidden);
    const ln = try b.layerNorm(lin, gamma, beta, hidden, 1e-5);
    const out = try b.rmsNorm(ln, rms_weight, hidden, 1e-5);
    try g.markOutput(out);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &partition_mod.supportsAll },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = native_compute.WeightStore{ .allocator = allocator, .resident_weights = .{}, .lazy_weights = .{} };
    defer deinitEmptyNativeWeightStore(&weight_store, allocator);
    var compute = native_compute.NativeCompute.init(allocator, &weight_store, null);
    var cb = compute.computeBackend();

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    var x_data: [hidden]f32 = undefined;
    var w_data: [hidden * hidden]f32 = .{0} ** (hidden * hidden);
    var bias_data: [hidden]f32 = .{0} ** hidden;
    var gamma_data: [hidden]f32 = .{1} ** hidden;
    var beta_data: [hidden]f32 = .{0} ** hidden;
    var rms_weight_data: [hidden]f32 = .{1} ** hidden;
    for (&x_data, 0..) |*value, i| value.* = @floatFromInt(i + 1);
    for (0..hidden) |i| w_data[i * hidden + i] = 1.0;

    const x_ct = try cb.fromFloat32Shape(&x_data, &.{ 1, hidden });
    defer cb.free(x_ct);
    const w_ct = try cb.fromFloat32Shape(&w_data, &.{ hidden, hidden });
    defer cb.free(w_ct);
    const bias_ct = try cb.fromFloat32Shape(&bias_data, &.{hidden});
    defer cb.free(bias_ct);
    const gamma_ct = try cb.fromFloat32Shape(&gamma_data, &.{hidden});
    defer cb.free(gamma_ct);
    const beta_ct = try cb.fromFloat32Shape(&beta_data, &.{hidden});
    defer cb.free(beta_ct);
    const rms_weight_ct = try cb.fromFloat32Shape(&rms_weight_data, &.{hidden});
    defer cb.free(rms_weight_ct);
    values[@intCast(x)] = x_ct;
    values[@intCast(w)] = w_ct;
    values[@intCast(bias)] = bias_ct;
    values[@intCast(gamma)] = gamma_ct;
    values[@intCast(beta)] = beta_ct;
    values[@intCast(rms_weight)] = rms_weight_ct;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    const partition_index = partition_plan.node_assignment[out];
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .options = .{
            .runtime_inputs = &.{
                .{ .node_id = x, .value = x_ct },
                .{ .node_id = w, .value = w_ct },
                .{ .node_id = bias, .value = bias_ct },
                .{ .node_id = gamma, .value = gamma_ct },
                .{ .node_id = beta, .value = beta_ct },
                .{ .node_id = rms_weight, .value = rms_weight_ct },
            },
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
    });

    const raw = try cb.toFloat32(values[@intCast(out)].?, allocator);
    defer allocator.free(raw);
    defer cb.free(values[@intCast(out)].?);
    try std.testing.expectEqual(hidden, raw.len);
    try std.testing.expect(raw[0] < -1.5);
    try std.testing.expect(raw[hidden - 1] > 1.5);
    try std.testing.expect(raw[hidden - 1] > raw[0]);
}

test "metal partition executor command path runs linear and norms on metal backend" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);
    const hidden: usize = 16;

    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ 1, hidden }));
    const w = try b.parameter("w", ml.graph.Shape.init(.f32, &.{ hidden, hidden }));
    const bias = try b.parameter("bias", ml.graph.Shape.init(.f32, &.{hidden}));
    const gamma = try b.parameter("gamma", ml.graph.Shape.init(.f32, &.{hidden}));
    const beta = try b.parameter("beta", ml.graph.Shape.init(.f32, &.{hidden}));
    const rms_weight = try b.parameter("rms_weight", ml.graph.Shape.init(.f32, &.{hidden}));
    const lin = try b.linear(x, w, bias, 1, hidden, hidden);
    const ln = try b.layerNorm(lin, gamma, beta, hidden, 1e-5);
    const out = try b.rmsNorm(ln, rms_weight, hidden, 1e-5);
    try g.markOutput(out);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &partition_mod.supportsAll },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    var x_data: [hidden]f32 = undefined;
    var w_data: [hidden * hidden]f32 = .{0} ** (hidden * hidden);
    var bias_data: [hidden]f32 = .{0} ** hidden;
    var gamma_data: [hidden]f32 = .{1} ** hidden;
    var beta_data: [hidden]f32 = .{0} ** hidden;
    var rms_weight_data: [hidden]f32 = .{1} ** hidden;
    for (&x_data, 0..) |*value, i| value.* = @floatFromInt(i + 1);
    for (0..hidden) |i| w_data[i * hidden + i] = 1.0;

    const x_ct = try cb.fromFloat32Shape(&x_data, &.{ 1, hidden });
    defer cb.free(x_ct);
    const w_ct = try cb.fromFloat32Shape(&w_data, &.{ hidden, hidden });
    defer cb.free(w_ct);
    const bias_ct = try cb.fromFloat32Shape(&bias_data, &.{hidden});
    defer cb.free(bias_ct);
    const gamma_ct = try cb.fromFloat32Shape(&gamma_data, &.{hidden});
    defer cb.free(gamma_ct);
    const beta_ct = try cb.fromFloat32Shape(&beta_data, &.{hidden});
    defer cb.free(beta_ct);
    const rms_weight_ct = try cb.fromFloat32Shape(&rms_weight_data, &.{hidden});
    defer cb.free(rms_weight_ct);
    values[@intCast(x)] = x_ct;
    values[@intCast(w)] = w_ct;
    values[@intCast(bias)] = bias_ct;
    values[@intCast(gamma)] = gamma_ct;
    values[@intCast(beta)] = beta_ct;
    values[@intCast(rms_weight)] = rms_weight_ct;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    var owned_runtime_transfers = std.AutoHashMapUnmanaged(NodeId, void).empty;
    defer owned_runtime_transfers.deinit(allocator);
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[0].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .options = .{
            .runtime_inputs = &.{
                .{ .node_id = x, .value = x_ct },
                .{ .node_id = w, .value = w_ct },
                .{ .node_id = bias, .value = bias_ct },
                .{ .node_id = gamma, .value = gamma_ct },
                .{ .node_id = beta, .value = beta_ct },
                .{ .node_id = rms_weight, .value = rms_weight_ct },
            },
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .owned_runtime_transfers = &owned_runtime_transfers,
    });

    const out_index: usize = @intCast(out);
    defer if (values[out_index]) |ct| cb.free(ct);
    const raw = try cb.toFloat32(values[out_index].?, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqual(hidden, raw.len);
    try std.testing.expect(raw[0] < -1.5);
    try std.testing.expect(raw[raw.len - 1] > 1.5);
    try std.testing.expect(raw[raw.len - 1] > raw[0]);
}

test "metal partition executor resident multi op chain matches host" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);
    const dim: usize = 8;

    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ 1, dim }));
    const w = try b.parameter("w", ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const bias = try b.parameter("bias", ml.graph.Shape.init(.f32, &.{dim}));
    const lin = try b.linear(x, w, bias, 1, dim, dim);
    const act = try b.silu(lin);
    const sum = try b.add(lin, act);
    const out = try b.softmax(sum);
    try g.markOutput(out);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &partition_mod.supportsAll },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    var native_weight_store = native_compute.WeightStore{ .allocator = allocator, .resident_weights = .{}, .lazy_weights = .{} };
    defer deinitEmptyNativeWeightStore(&native_weight_store, allocator);
    var native_compute_impl = native_compute.NativeCompute.init(allocator, &native_weight_store, null);
    var native_cb = native_compute_impl.computeBackend();
    var mesh = try device_mesh_mod.DeviceMesh.init(allocator, &.{
        .{ .id = 0, .backend = &native_cb, .kind = .native },
        .{ .id = 1, .backend = &cb, .kind = .metal },
    });
    defer mesh.deinit();

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const x_data = [_]f32{ -4.0, -2.0, -1.0, -0.25, 0.25, 1.0, 2.0, 4.0 };
    var w_data: [dim * dim]f32 = .{0} ** (dim * dim);
    const bias_data = [_]f32{ 0.5, -0.25, 0.125, -0.5, 0.25, 0.75, -0.125, 0.0 };
    for (0..dim) |i| w_data[i * dim + i] = 1.0;

    const x_ct = try native_cb.fromFloat32Shape(&x_data, &.{ 1, dim });
    defer native_cb.free(x_ct);
    const w_ct = try native_cb.fromFloat32Shape(&w_data, &.{ dim, dim });
    defer native_cb.free(w_ct);
    const bias_ct = try native_cb.fromFloat32Shape(&bias_data, &.{dim});
    defer native_cb.free(bias_ct);
    values[@intCast(x)] = x_ct;
    values[@intCast(w)] = w_ct;
    values[@intCast(bias)] = bias_ct;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    const partition_index = partition_plan.node_assignment[out];
    var owned_runtime_transfers = std.AutoHashMapUnmanaged(NodeId, void).empty;
    defer owned_runtime_transfers.deinit(allocator);
    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 1, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .mesh = &mesh,
        .options = .{
            .runtime_inputs = &.{
                .{ .node_id = x, .value = x_ct },
                .{ .node_id = w, .value = w_ct },
                .{ .node_id = bias, .value = bias_ct },
            },
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .owned_runtime_transfers = &owned_runtime_transfers,
        .materialize_boundary_outputs = false,
        .stats = &exec_stats,
    });

    const out_index: usize = @intCast(out);
    defer if (values[out_index]) |ct| cb.free(ct);
    const stats = cb.debugTimingSnapshot().provider;
    try std.testing.expectEqual(@as(u64, 1), stats.decoder_runtime_frame_begins);
    try std.testing.expectEqual(@as(u64, 1), stats.decoder_runtime_frame_submits);
    try std.testing.expect(exec_stats.runtime_input_transfers >= 3);
    try std.testing.expect(exec_stats.device_resident_transfers >= 3);
    try std.testing.expect(exec_stats.backend_command_dispatches >= 4);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.host_materialized_outputs);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.boundary_output_materializations);

    var logits: [dim]f32 = undefined;
    for (&logits, 0..) |*value, i| {
        const lin_value = x_data[i] + bias_data[i];
        const silu = lin_value / (1.0 + @exp(-lin_value));
        value.* = lin_value + silu;
    }
    var max_logit = logits[0];
    for (logits[1..]) |value| max_logit = @max(max_logit, value);
    var denom: f32 = 0.0;
    for (logits) |value| denom += @exp(value - max_logit);
    var expected: [dim]f32 = undefined;
    for (&expected, logits) |*value, logit| value.* = @exp(logit - max_logit) / denom;

    try std.testing.expect(metal_compute_mod.MetalCompute.debugHasDeviceTensor(&cb, values[out_index].?));

    const raw = try cb.toFloat32(values[out_index].?, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqual(dim, raw.len);
    for (expected, raw) |exp, actual| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-5);
    }
}

test "metal partition executor fuses sibling no-bias linears into one pair command" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ 1, 3 }));
    const w_a = try b.parameter("w_a", ml.graph.Shape.init(.f32, &.{ 2, 3 }));
    const w_b = try b.parameter("w_b", ml.graph.Shape.init(.f32, &.{ 2, 3 }));
    const a = try b.linearNoBias(x, w_a, 1, 3, 2);
    const c = try b.linearNoBias(x, w_b, 1, 3, 2);
    const out = try b.add(a, c);
    try g.markOutput(out);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &partition_mod.supportsAll },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = native_compute.WeightStore{ .allocator = allocator, .resident_weights = .{}, .lazy_weights = .{} };
    defer deinitEmptyNativeWeightStore(&weight_store, allocator);
    var compute = native_compute.NativeCompute.init(allocator, &weight_store, null);
    var cb = compute.computeBackend();

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const x_data = [_]f32{ 1.0, 2.0, 3.0 };
    const w_a_data = [_]f32{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
    };
    const w_b_data = [_]f32{
        0.0, 0.0, 1.0,
        1.0, 1.0, 1.0,
    };
    const x_ct = try cb.fromFloat32Shape(&x_data, &.{ 1, 3 });
    defer cb.free(x_ct);
    const w_a_ct = try cb.fromFloat32Shape(&w_a_data, &.{ 2, 3 });
    defer cb.free(w_a_ct);
    const w_b_ct = try cb.fromFloat32Shape(&w_b_data, &.{ 2, 3 });
    defer cb.free(w_b_ct);
    values[@intCast(x)] = x_ct;
    values[@intCast(w_a)] = w_a_ct;
    values[@intCast(w_b)] = w_b_ct;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    const partition_index = partition_plan.node_assignment[out];
    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .options = .{
            .runtime_inputs = &.{
                .{ .node_id = x, .value = x_ct },
                .{ .node_id = w_a, .value = w_a_ct },
                .{ .node_id = w_b, .value = w_b_ct },
            },
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .stats = &exec_stats,
    });

    const out_ct = values[@intCast(out)].?;
    defer cb.free(out_ct);
    const raw = try cb.toFloat32(out_ct, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqualSlices(f32, &[_]f32{ 4.0, 8.0 }, raw);
    try std.testing.expectEqual(@as(u64, 2), exec_stats.backend_command_dispatches);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);
}

test "metal partition executor recognizes pre-norm gated ffn residual graph pattern" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 2;
    const hidden: usize = 4;
    const intermediate: usize = 6;
    const residual = try b.parameter("residual", ml.graph.Shape.init(.f32, &.{ rows, hidden }));
    const norm_w = try b.parameter("norm_w", ml.graph.Shape.init(.f32, &.{hidden}));
    const gate_w = try b.parameter("gate_w", ml.graph.Shape.init(.f32, &.{ intermediate, hidden }));
    const up_w = try b.parameter("up_w", ml.graph.Shape.init(.f32, &.{ intermediate, hidden }));
    const down_w = try b.parameter("down_w", ml.graph.Shape.init(.f32, &.{ hidden, intermediate }));
    const post_w = try b.parameter("post_w", ml.graph.Shape.init(.f32, &.{hidden}));
    const normed = try b.rmsNorm(residual, norm_w, @intCast(hidden), 1e-5);
    const pair = try g.addNode(.{
        .op = .{ .fused_linear_no_bias_pair = .{
            .rows = @intCast(rows),
            .in_dim = @intCast(hidden),
            .out_dim = @intCast(intermediate),
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ rows, intermediate }),
        .inputs = .{ normed, gate_w, up_w, null_node },
        .num_inputs = 3,
    });
    const activated = try b.gelu(pair);
    const pair_second = try g.addNode(.{
        .op = .{ .fused_to_float32 = {} },
        .output_shape = ml.graph.Shape.init(.f32, &.{ rows, intermediate }),
        .inputs = .{ pair, null_node, null_node, null_node },
        .num_inputs = 1,
    });
    const gated = try b.elemMultiply(activated, pair_second);
    const down = try b.linearNoBias(gated, down_w, @intCast(rows), @intCast(intermediate), @intCast(hidden));
    const post = try b.rmsNorm(down, post_w, @intCast(hidden), 1e-6);
    const out = try b.add(post, residual);
    try g.markOutput(out);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);
    const node_ids = try allocator.alloc(NodeId, @intCast(g.nodeCount()));
    defer allocator.free(node_ids);
    for (node_ids, 0..) |*node_id, idx| node_id.* = @intCast(idx);
    const skipped = try allocator.alloc(bool, node_ids.len);
    defer allocator.free(skipped);
    @memset(skipped, false);

    const pattern = matchGatedFfnResidualPattern(&g, node_ids, @intCast(pair), reachable, skipped, last_use) orelse return error.ExpectedGatedFfnPattern;
    try std.testing.expectEqual(pair, pattern.pair_id);
    try std.testing.expectEqual(pair_second, pattern.pair_second_id);
    try std.testing.expectEqual(activated, pattern.activation_id);
    try std.testing.expectEqual(gated, pattern.multiply_id);
    try std.testing.expectEqual(down, pattern.down_id);
    try std.testing.expectEqual(post, pattern.post_down_norm_id.?);
    try std.testing.expectEqual(out, pattern.add_id);
    try std.testing.expectEqual(residual, pattern.residual_id);
    try std.testing.expectEqual(@as(usize, hidden), pattern.hidden_size);
    try std.testing.expectEqual(@as(usize, intermediate), pattern.intermediate_size);
    try std.testing.expectEqual(ops_mod.DecoderRuntimeActivationKind.gelu, pattern.activation);
}

test "metal partition executor recognizes attention output residual graph pattern" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 3;
    const hidden: usize = 8;
    const heads: usize = 2;
    const kv_heads: usize = 1;
    const head_dim: usize = 4;
    const attn_dim: usize = heads * head_dim;
    const q = try b.parameter("q", ml.graph.Shape.init(.f32, &.{ rows, attn_dim }));
    const k = try b.parameter("k", ml.graph.Shape.init(.f32, &.{ rows, kv_heads * head_dim }));
    const v = try b.parameter("v", ml.graph.Shape.init(.f32, &.{ rows, kv_heads * head_dim }));
    const residual = try b.parameter("residual", ml.graph.Shape.init(.f32, &.{ rows, hidden }));
    const pre_w = try b.parameter("pre_w", ml.graph.Shape.init(.f32, &.{attn_dim}));
    const out_w = try b.parameter("out_w", ml.graph.Shape.init(.f32, &.{ hidden, attn_dim }));
    const post_w = try b.parameter("post_w", ml.graph.Shape.init(.f32, &.{hidden}));
    const attention = try g.addNode(.{
        .op = .{ .fused_gqa_causal_attention = .{
            .batch = 1,
            .seq_len = @intCast(rows),
            .num_heads = @intCast(heads),
            .num_kv_heads = @intCast(kv_heads),
            .head_dim = @intCast(head_dim),
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ rows, attn_dim }),
        .inputs = .{ q, k, v, null_node },
        .num_inputs = 3,
    });
    const pre = try b.rmsNorm(attention, pre_w, @intCast(attn_dim), 1e-5);
    const projected = try b.linearNoBias(pre, out_w, @intCast(rows), @intCast(attn_dim), @intCast(hidden));
    const post = try b.rmsNorm(projected, post_w, @intCast(hidden), 1e-5);
    const out = try b.add(post, residual);
    try g.markOutput(out);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);
    const node_ids = try allocator.alloc(NodeId, @intCast(g.nodeCount()));
    defer allocator.free(node_ids);
    for (node_ids, 0..) |*node_id, idx| node_id.* = @intCast(idx);
    const skipped = try allocator.alloc(bool, node_ids.len);
    defer allocator.free(skipped);
    @memset(skipped, false);

    const pattern = matchAttentionOutputResidualPattern(&g, node_ids, @intCast(attention), reachable, skipped, last_use) orelse return error.ExpectedAttentionOutputResidualPattern;
    try std.testing.expectEqual(attention, pattern.attention_id);
    try std.testing.expectEqual(pre, pattern.pre_linear_norm_id.?);
    try std.testing.expectEqual(projected, pattern.linear_id);
    try std.testing.expectEqual(post, pattern.post_linear_norm_id.?);
    try std.testing.expectEqual(out, pattern.add_id);
    try std.testing.expectEqual(residual, pattern.residual_id);
    try std.testing.expectEqual(@as(usize, rows), pattern.rows);
    try std.testing.expectEqual(@as(usize, attn_dim), pattern.attention_input_size);
    try std.testing.expectEqual(@as(usize, hidden), pattern.hidden_size);

    var plan = try buildRuntimeRegionPlan(allocator, &g, node_ids, @intCast(g.nodeCount()), reachable, last_use);
    defer plan.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), plan.region_count);
    switch (plan.regionAt(@intCast(attention), attention, node_ids)) {
        .attention_output_residual => |planned| {
            try std.testing.expectEqual(attention, planned.attention_id);
            try std.testing.expectEqual(projected, planned.linear_id);
            try std.testing.expectEqual(out, planned.add_id);
        },
        else => return error.ExpectedPlannedAttentionOutputResidualRegion,
    }
}

test "metal partition executor recognizes gemma qkv sibling linear graph region" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 2;
    const hidden: usize = 8;
    const q_dim: usize = 16;
    const kv_dim: usize = 4;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ rows, hidden }));
    const q_w = try b.parameter("model.layers.0.self_attn.q_proj.weight", ml.graph.Shape.init(.f32, &.{ q_dim, hidden }));
    const k_w = try b.parameter("model.layers.0.self_attn.k_proj.weight", ml.graph.Shape.init(.f32, &.{ kv_dim, hidden }));
    const v_w = try b.parameter("model.layers.0.self_attn.v_proj.weight", ml.graph.Shape.init(.f32, &.{ kv_dim, hidden }));
    const q = try b.linearNoBias(x, q_w, @intCast(rows), @intCast(hidden), @intCast(q_dim));
    const k = try b.linearNoBias(x, k_w, @intCast(rows), @intCast(hidden), @intCast(kv_dim));
    const v = try b.linearNoBias(x, v_w, @intCast(rows), @intCast(hidden), @intCast(kv_dim));
    try g.markOutput(q);
    try g.markOutput(k);
    try g.markOutput(v);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);
    const node_ids = try allocator.alloc(NodeId, @intCast(g.nodeCount()));
    defer allocator.free(node_ids);
    for (node_ids, 0..) |*node_id, idx| node_id.* = @intCast(idx);
    const skipped = try allocator.alloc(bool, node_ids.len);
    defer allocator.free(skipped);
    @memset(skipped, false);
    const values = try allocator.alloc(?CT, node_ids.len);
    defer allocator.free(values);
    @memset(values, null);

    const pattern = matchLinearNoBiasQkvPattern(&g, values, node_ids, @intCast(q), reachable, skipped) orelse return error.ExpectedQkvRegion;
    try std.testing.expectEqual(q, pattern.q_id);
    try std.testing.expectEqual(k, pattern.k_id);
    try std.testing.expectEqual(v, pattern.v_id);
    try std.testing.expectEqual(x, pattern.input_id);
    try std.testing.expectEqual(q_w, pattern.q_weight_id);
    try std.testing.expectEqual(k_w, pattern.k_weight_id);
    try std.testing.expectEqual(v_w, pattern.v_weight_id);
    try std.testing.expectEqual(@as(usize, rows), pattern.rows);
    try std.testing.expectEqual(@as(usize, hidden), pattern.in_dim);
    try std.testing.expectEqual(@as(usize, q_dim), pattern.q_out_dim);
    try std.testing.expectEqual(@as(usize, kv_dim), pattern.kv_out_dim);

    var plan = try buildRuntimeRegionPlan(allocator, &g, node_ids, @intCast(g.nodeCount()), reachable, last_use);
    defer plan.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), plan.region_count);
    switch (plan.regionAt(@intCast(q), q, node_ids)) {
        .linear_qkv => |planned| {
            try std.testing.expectEqual(q, planned.q_id);
            try std.testing.expectEqual(k, planned.k_id);
            try std.testing.expectEqual(v, planned.v_id);
        },
        else => return error.ExpectedPlannedQkvRegion,
    }
}

test "metal partition executor recognizes gemma q-only linear through attention layout path" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 1;
    const hidden: usize = 1536;
    const heads: usize = 8;
    const kv_heads: usize = 1;
    const head_dim: usize = 256;
    const q_dim: usize = heads * head_dim;
    const kv_dim: usize = kv_heads * head_dim;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ rows, hidden }));
    const q_w = try b.parameter("model.layers.1.self_attn.q_proj.weight", ml.graph.Shape.init(.f32, &.{ q_dim, hidden }));
    const k = try b.parameter("k", ml.graph.Shape.init(.f32, &.{ rows, kv_dim }));
    const v = try b.parameter("v", ml.graph.Shape.init(.f32, &.{ rows, kv_dim }));
    const q_norm_w = try b.parameter("model.layers.1.self_attn.q_norm.weight", ml.graph.Shape.init(.f32, &.{head_dim}));
    const q_scale = try b.parameter("model.layers.1.self_attn.q_scale", ml.graph.Shape.init(.f32, &.{ rows, q_dim }));
    const cos = try b.tensorConst(&[_]f32{1.0}, ml.graph.Shape.init(.f32, &.{1}));
    const sin = try b.tensorConst(&[_]f32{0.0}, ml.graph.Shape.init(.f32, &.{1}));

    const q = try b.linearNoBias(x, q_w, @intCast(rows), @intCast(hidden), @intCast(q_dim));
    const q_heads = try b.reshape(q, ml.graph.Shape.init(.f32, &.{ @intCast(heads), @intCast(head_dim) }));
    const q_norm = try b.rmsNorm(q_heads, q_norm_w, @intCast(head_dim), 1e-5);
    const q_flat = try b.reshape(q_norm, ml.graph.Shape.init(.f32, &.{ rows, q_dim }));
    const q_scaled = try b.mul(q_flat, q_scale);
    const q_rope = try b.rope(q_scaled, cos, sin, @intCast(rows), @intCast(head_dim), @intCast(head_dim), 10000.0);
    const attention = try g.addNode(.{
        .op = .{ .fused_gqa_causal_attention = .{
            .batch = 1,
            .seq_len = @intCast(rows),
            .num_heads = @intCast(heads),
            .num_kv_heads = @intCast(kv_heads),
            .head_dim = @intCast(head_dim),
            .skip_kv_write = true,
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ rows, q_dim }),
        .inputs = .{ q_rope, k, v, null_node },
        .num_inputs = 3,
    });
    try g.markOutput(attention);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);
    const node_ids = try allocator.alloc(NodeId, @intCast(g.nodeCount()));
    defer allocator.free(node_ids);
    for (node_ids, 0..) |*node_id, idx| node_id.* = @intCast(idx);
    const skipped = try allocator.alloc(bool, node_ids.len);
    defer allocator.free(skipped);
    @memset(skipped, false);
    const values = try allocator.alloc(?CT, node_ids.len);
    defer allocator.free(values);
    @memset(values, null);

    const pattern = matchQLinearPattern(&g, values, node_ids, @intCast(q), reachable, skipped) orelse return error.ExpectedQLinearRegion;
    try std.testing.expectEqual(q, pattern.id);
    try std.testing.expectEqual(x, pattern.input_id);
    try std.testing.expectEqual(q_w, pattern.weight_id);
    try std.testing.expectEqual(@as(usize, rows), pattern.rows);
    try std.testing.expectEqual(@as(usize, hidden), pattern.in_dim);
    try std.testing.expectEqual(@as(usize, q_dim), pattern.out_dim);

    var plan = try buildRuntimeRegionPlan(allocator, &g, node_ids, @intCast(g.nodeCount()), reachable, last_use);
    defer plan.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), plan.region_count);
    switch (plan.regionAt(@intCast(q), q, node_ids)) {
        .q_linear => |planned| try std.testing.expectEqual(q, planned.id),
        else => return error.ExpectedPlannedQLinearRegion,
    }
}

test "metal partition executor recognizes raw transposed linear dot graph region" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 2;
    const in_dim: usize = 3;
    const out_dim: usize = 4;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ rows, in_dim }));
    const weight = try b.parameter("encoder.layer.0.attention.self.query_proj.weight", ml.graph.Shape.init(.f32, &.{ out_dim, in_dim }));
    const weight_t = try b.transpose(weight, &.{ 1, 0 });
    const dot = try g.addNode(.{
        .op = .{ .dot_general = .{
            .lhs_contracting = .{ 1, 0, 0, 0, 0, 0, 0, 0 },
            .rhs_contracting = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .lhs_batch = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .rhs_batch = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .num_contracting = 1,
            .num_batch = 0,
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ rows, out_dim }),
        .inputs = .{ x, weight_t, null_node, null_node },
        .num_inputs = 2,
    });
    try g.markOutput(dot);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);
    const node_ids = try allocator.alloc(NodeId, @intCast(g.nodeCount()));
    defer allocator.free(node_ids);
    for (node_ids, 0..) |*node_id, idx| node_id.* = @intCast(idx);

    var plan = try buildRuntimeRegionPlan(allocator, &g, node_ids, @intCast(g.nodeCount()), reachable, last_use);
    defer plan.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), plan.region_count);
    switch (plan.regionAt(@intCast(dot), dot, node_ids)) {
        .raw_linear_dot => |pattern| {
            try std.testing.expectEqual(dot, pattern.id);
            try std.testing.expectEqual(x, pattern.input_id);
            try std.testing.expectEqual(weight_t, pattern.transpose_id);
            try std.testing.expectEqual(weight, pattern.weight_id);
            try std.testing.expectEqual(@as(usize, rows), pattern.rows);
            try std.testing.expectEqual(@as(usize, in_dim), pattern.in_dim);
            try std.testing.expectEqual(@as(usize, out_dim), pattern.out_dim);
        },
        else => return error.ExpectedRawLinearDotRegion,
    }
}

test "metal partition executor recognizes raw transposed linear dot plus bias graph region" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 2;
    const in_dim: usize = 3;
    const out_dim: usize = 4;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ rows, in_dim }));
    const weight = try b.parameter("encoder.layer.0.attention.self.query_proj.weight", ml.graph.Shape.init(.f32, &.{ out_dim, in_dim }));
    const bias = try b.parameter("encoder.layer.0.attention.self.query_proj.bias", ml.graph.Shape.init(.f32, &.{out_dim}));
    const weight_t = try b.transpose(weight, &.{ 1, 0 });
    const dot = try g.addNode(.{
        .op = .{ .dot_general = .{
            .lhs_contracting = .{ 1, 0, 0, 0, 0, 0, 0, 0 },
            .rhs_contracting = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .lhs_batch = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .rhs_batch = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .num_contracting = 1,
            .num_batch = 0,
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ rows, out_dim }),
        .inputs = .{ x, weight_t, null_node, null_node },
        .num_inputs = 2,
    });
    const biased = try b.add(dot, bias);
    try g.markOutput(biased);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);
    const node_ids = try allocator.alloc(NodeId, @intCast(g.nodeCount()));
    defer allocator.free(node_ids);
    for (node_ids, 0..) |*node_id, idx| node_id.* = @intCast(idx);

    var plan = try buildRuntimeRegionPlan(allocator, &g, node_ids, @intCast(g.nodeCount()), reachable, last_use);
    defer plan.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), plan.region_count);
    switch (plan.regionAt(@intCast(dot), dot, node_ids)) {
        .raw_linear_bias => |pattern| {
            try std.testing.expectEqual(dot, pattern.dot.id);
            try std.testing.expectEqual(biased, pattern.add_id);
            try std.testing.expectEqual(bias, pattern.bias_id);
            try std.testing.expectEqual(x, pattern.dot.input_id);
            try std.testing.expectEqual(weight_t, pattern.dot.transpose_id);
            try std.testing.expectEqual(weight, pattern.dot.weight_id);
            try std.testing.expectEqual(@as(usize, rows), pattern.dot.rows);
            try std.testing.expectEqual(@as(usize, in_dim), pattern.dot.in_dim);
            try std.testing.expectEqual(@as(usize, out_dim), pattern.dot.out_dim);
        },
        else => return error.ExpectedRawLinearBiasRegion,
    }
    try std.testing.expectEqual(RuntimeRegion{ .none = {} }, plan.regionAt(@intCast(biased), biased, node_ids));
}

test "metal partition executor defers scalar scale mul only for single add consumer" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const shape = ml.graph.Shape.init(.f32, &.{ 2, 3 });
    const x = try b.parameter("x", shape);
    const y = try b.parameter("y", shape);
    const z = try b.parameter("z", shape);
    const scale = try b.scalarConst(.f32, 0.5);
    const scaled = try b.mul(x, scale);
    const first = try b.add(scaled, y);
    const second = try b.add(scaled, z);
    try g.markOutput(first);
    try g.markOutput(second);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    try std.testing.expectEqual(@as(usize, 2), reachableUseCount(&g, scaled, reachable, 2));
    try std.testing.expect(!shouldDeferScaleMulForAdd(&g, scaled, reachable, last_use));
}

test "metal partition executor defers two independent scalar scale mul add inputs" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const shape = ml.graph.Shape.init(.f32, &.{ 2, 3 });
    const x = try b.parameter("x", shape);
    const y = try b.parameter("y", shape);
    const scale = try b.scalarConst(.f32, 0.5);
    const lhs_scaled = try b.mul(x, scale);
    const rhs_scaled = try b.mul(y, scale);
    const out = try b.add(lhs_scaled, rhs_scaled);
    try g.markOutput(out);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    try std.testing.expectEqual(@as(usize, 1), reachableUseCount(&g, lhs_scaled, reachable, 2));
    try std.testing.expectEqual(@as(usize, 1), reachableUseCount(&g, rhs_scaled, reachable, 2));
    try std.testing.expect(shouldDeferScaleMulForAdd(&g, lhs_scaled, reachable, last_use));
    try std.testing.expect(shouldDeferScaleMulForAdd(&g, rhs_scaled, reachable, last_use));
    try std.testing.expect(deferredScaleMul(&g, lhs_scaled, g.node(out).output_shape) != null);
    try std.testing.expect(deferredScaleMul(&g, rhs_scaled, g.node(out).output_shape) != null);
}

test "metal partition executor defers same shape multiply for multiply-add" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const shape = ml.graph.Shape.init(.f32, &.{ 2, 3 });
    const x = try b.parameter("x", shape);
    const y = try b.parameter("y", shape);
    const z = try b.parameter("z", shape);
    const multiplied = try b.mul(x, y);
    const out = try b.add(multiplied, z);
    try g.markOutput(out);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    try std.testing.expectEqual(@as(usize, 1), reachableUseCount(&g, multiplied, reachable, 2));
    try std.testing.expect(deferredElementwiseMul(&g, multiplied, g.node(out).output_shape) != null);
    try std.testing.expect(shouldDeferElementwiseMulForAdd(&g, multiplied, reachable, last_use));
}

test "metal partition executor defers both same shape multiply add inputs" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const shape = ml.graph.Shape.init(.f32, &.{ 2, 3 });
    const x = try b.parameter("x", shape);
    const y = try b.parameter("y", shape);
    const z = try b.parameter("z", shape);
    const w = try b.parameter("w", shape);
    const lhs = try b.mul(x, y);
    const rhs = try b.mul(z, w);
    const out = try b.add(lhs, rhs);
    try g.markOutput(out);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    try std.testing.expectEqual(@as(usize, 1), reachableUseCount(&g, lhs, reachable, 2));
    try std.testing.expectEqual(@as(usize, 1), reachableUseCount(&g, rhs, reachable, 2));
    try std.testing.expect(deferredElementwiseMul(&g, lhs, g.node(out).output_shape) != null);
    try std.testing.expect(deferredElementwiseMul(&g, rhs, g.node(out).output_shape) != null);
    try std.testing.expect(shouldDeferElementwiseMulForAdd(&g, lhs, reachable, last_use));
    try std.testing.expect(shouldDeferElementwiseMulForAdd(&g, rhs, reachable, last_use));
}

test "metal partition executor does not defer nested same shape multiply add input" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const shape = ml.graph.Shape.init(.f32, &.{ 2, 3 });
    const x = try b.parameter("x", shape);
    const y = try b.parameter("y", shape);
    const z = try b.parameter("z", shape);
    const residual = try b.parameter("residual", shape);
    const inner = try b.mul(x, y);
    const outer = try b.mul(inner, z);
    const out = try b.add(outer, residual);
    try g.markOutput(out);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    try std.testing.expect(deferredElementwiseMul(&g, inner, g.node(out).output_shape) != null);
    try std.testing.expect(deferredElementwiseMul(&g, outer, g.node(out).output_shape) == null);
    try std.testing.expect(!shouldDeferElementwiseMulForAdd(&g, outer, reachable, last_use));
}

test "metal partition executor does not defer transpose when it escapes non-linear-dot consumers" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 2;
    const in_dim: usize = 3;
    const out_dim: usize = 4;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ rows, in_dim }));
    const weight = try b.parameter("encoder.layer.0.attention.self.query_proj.weight", ml.graph.Shape.init(.f32, &.{ out_dim, in_dim }));
    const residual = try b.parameter("residual", ml.graph.Shape.init(.f32, &.{ in_dim, out_dim }));
    const weight_t = try b.transpose(weight, &.{ 1, 0 });
    const dot = try g.addNode(.{
        .op = .{ .dot_general = .{
            .lhs_contracting = .{ 1, 0, 0, 0, 0, 0, 0, 0 },
            .rhs_contracting = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .lhs_batch = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .rhs_batch = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .num_contracting = 1,
            .num_batch = 0,
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ rows, out_dim }),
        .inputs = .{ x, weight_t, null_node, null_node },
        .num_inputs = 2,
    });
    const escaped = try b.add(weight_t, residual);
    try g.markOutput(dot);
    try g.markOutput(escaped);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    try std.testing.expectEqual(@as(usize, 2), reachableUseCount(&g, weight_t, reachable, 2));
    try std.testing.expect(!shouldDeferTransposeForLinearDot(&g, weight_t, reachable, last_use));
}

test "metal partition executor defers transpose shared by compatible linear dots" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 2;
    const in_dim: usize = 3;
    const out_dim: usize = 4;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ rows, in_dim }));
    const y = try b.parameter("y", ml.graph.Shape.init(.f32, &.{ rows, in_dim }));
    const weight = try b.parameter("encoder.layer.0.attention.self.query_proj.weight", ml.graph.Shape.init(.f32, &.{ out_dim, in_dim }));
    const weight_t = try b.transpose(weight, &.{ 1, 0 });
    const first = try g.addNode(.{
        .op = .{ .dot_general = .{
            .lhs_contracting = .{ 1, 0, 0, 0, 0, 0, 0, 0 },
            .rhs_contracting = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .lhs_batch = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .rhs_batch = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .num_contracting = 1,
            .num_batch = 0,
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ rows, out_dim }),
        .inputs = .{ x, weight_t, null_node, null_node },
        .num_inputs = 2,
    });
    const second = try g.addNode(.{
        .op = .{ .dot_general = .{
            .lhs_contracting = .{ 1, 0, 0, 0, 0, 0, 0, 0 },
            .rhs_contracting = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .lhs_batch = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .rhs_batch = .{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .num_contracting = 1,
            .num_batch = 0,
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ rows, out_dim }),
        .inputs = .{ y, weight_t, null_node, null_node },
        .num_inputs = 2,
    });
    try g.markOutput(first);
    try g.markOutput(second);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    try std.testing.expectEqual(@as(usize, 2), reachableUseCount(&g, weight_t, reachable, 2));
    try std.testing.expect(shouldDeferTransposeForLinearDot(&g, weight_t, reachable, last_use));
}

test "metal partition executor recognizes grouped qkv linear slice graph region" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 2;
    const hidden: usize = 8;
    const q_dim: usize = 16;
    const kv_dim: usize = 4;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ rows, hidden }));
    const q_w = try b.parameter("model.layers.0.self_attn.q_proj.weight", ml.graph.Shape.init(.f32, &.{ q_dim, hidden }));
    const k_w = try b.parameter("model.layers.0.self_attn.k_proj.weight", ml.graph.Shape.init(.f32, &.{ kv_dim, hidden }));
    const v_w = try b.parameter("model.layers.0.self_attn.v_proj.weight", ml.graph.Shape.init(.f32, &.{ kv_dim, hidden }));
    const qk_w = try b.concat(q_w, k_w, 0);
    const qkv_w = try b.concat(qk_w, v_w, 0);
    const qkv = try b.linearNoBias(x, qkv_w, @intCast(rows), @intCast(hidden), @intCast(q_dim + kv_dim * 2));
    const q = try b.sliceLastDim(qkv, 0, @intCast(q_dim));
    const k = try b.sliceLastDim(qkv, @intCast(q_dim), @intCast(q_dim + kv_dim));
    const v = try b.sliceLastDim(qkv, @intCast(q_dim + kv_dim), @intCast(q_dim + kv_dim * 2));
    const kv = try b.add(k, v);
    const out = try b.add(q, kv);
    try g.markOutput(out);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);
    const node_ids = try allocator.alloc(NodeId, @intCast(g.nodeCount()));
    defer allocator.free(node_ids);
    for (node_ids, 0..) |*node_id, idx| node_id.* = @intCast(idx);
    const skipped = try allocator.alloc(bool, node_ids.len);
    defer allocator.free(skipped);
    @memset(skipped, false);
    const values = try allocator.alloc(?CT, node_ids.len);
    defer allocator.free(values);
    @memset(values, null);

    const pattern = matchGroupedLinearQkvSlicePattern(&g, values, node_ids, @intCast(qkv), reachable, skipped) orelse return error.ExpectedGroupedQkvRegion;
    try std.testing.expectEqual(qkv, pattern.linear_id);
    try std.testing.expectEqual(q, pattern.q_slice_id);
    try std.testing.expectEqual(k, pattern.k_slice_id);
    try std.testing.expectEqual(v, pattern.v_slice_id);
    try std.testing.expectEqual(x, pattern.input_id);
    try std.testing.expectEqual(q_w, pattern.q_weight_id);
    try std.testing.expectEqual(k_w, pattern.k_weight_id);
    try std.testing.expectEqual(v_w, pattern.v_weight_id);
    try std.testing.expectEqual(@as(usize, q_dim), pattern.q_out_dim);
    try std.testing.expectEqual(@as(usize, kv_dim), pattern.kv_out_dim);

    var plan = try buildRuntimeRegionPlan(allocator, &g, node_ids, @intCast(g.nodeCount()), reachable, last_use);
    defer plan.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), plan.region_count);
    switch (plan.regionAt(@intCast(qkv), qkv, node_ids)) {
        .grouped_linear_qkv_slice => |planned| {
            try std.testing.expectEqual(qkv, planned.linear_id);
            try std.testing.expectEqual(q, planned.q_slice_id);
            try std.testing.expectEqual(v, planned.v_slice_id);
        },
        else => return error.ExpectedPlannedGroupedQkvRegion,
    }
}

test "metal partition executor recognizes rms norm grouped qkv graph region" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 2;
    const hidden: usize = 8;
    const q_dim: usize = 16;
    const kv_dim: usize = 4;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ rows, hidden }));
    const norm_w = try b.parameter("model.layers.0.input_layernorm.weight", ml.graph.Shape.init(.f32, &.{hidden}));
    const q_w = try b.parameter("model.layers.0.self_attn.q_proj.weight", ml.graph.Shape.init(.f32, &.{ q_dim, hidden }));
    const k_w = try b.parameter("model.layers.0.self_attn.k_proj.weight", ml.graph.Shape.init(.f32, &.{ kv_dim, hidden }));
    const v_w = try b.parameter("model.layers.0.self_attn.v_proj.weight", ml.graph.Shape.init(.f32, &.{ kv_dim, hidden }));
    const normed = try b.rmsNorm(x, norm_w, @intCast(hidden), 1e-5);
    const qk_w = try b.concat(q_w, k_w, 0);
    const qkv_w = try b.concat(qk_w, v_w, 0);
    const qkv = try b.linearNoBias(normed, qkv_w, @intCast(rows), @intCast(hidden), @intCast(q_dim + kv_dim * 2));
    const q = try b.sliceLastDim(qkv, 0, @intCast(q_dim));
    const k = try b.sliceLastDim(qkv, @intCast(q_dim), @intCast(q_dim + kv_dim));
    const v = try b.sliceLastDim(qkv, @intCast(q_dim + kv_dim), @intCast(q_dim + kv_dim * 2));
    const kv = try b.add(k, v);
    const out = try b.add(q, kv);
    try g.markOutput(out);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);
    const node_ids = try allocator.alloc(NodeId, @intCast(g.nodeCount()));
    defer allocator.free(node_ids);
    for (node_ids, 0..) |*node_id, idx| node_id.* = @intCast(idx);
    const skipped = try allocator.alloc(bool, node_ids.len);
    defer allocator.free(skipped);
    @memset(skipped, false);
    const values = try allocator.alloc(?CT, node_ids.len);
    defer allocator.free(values);
    @memset(values, null);

    const pattern = matchRmsNormGroupedLinearQkvSlicePattern(&g, values, node_ids, @intCast(normed), reachable, skipped) orelse return error.ExpectedRmsGroupedQkvRegion;
    try std.testing.expectEqual(normed, pattern.norm_id);
    try std.testing.expectEqual(x, pattern.norm_input_id);
    try std.testing.expectEqual(norm_w, pattern.norm_weight_id);
    try std.testing.expectEqual(@as(usize, hidden), pattern.norm_dim);
    try std.testing.expectEqual(qkv, pattern.qkv.linear_id);
    try std.testing.expectEqual(q, pattern.qkv.q_slice_id);
    try std.testing.expectEqual(k, pattern.qkv.k_slice_id);
    try std.testing.expectEqual(v, pattern.qkv.v_slice_id);
    try std.testing.expectEqual(q_w, pattern.qkv.q_weight_id);
    try std.testing.expectEqual(k_w, pattern.qkv.k_weight_id);
    try std.testing.expectEqual(v_w, pattern.qkv.v_weight_id);

    var plan = try buildRuntimeRegionPlan(allocator, &g, node_ids, @intCast(g.nodeCount()), reachable, last_use);
    defer plan.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), plan.region_count);
    switch (plan.regionAt(@intCast(normed), normed, node_ids)) {
        .rms_norm_grouped_linear_qkv_slice => |planned| {
            try std.testing.expectEqual(normed, planned.norm_id);
            try std.testing.expectEqual(qkv, planned.qkv.linear_id);
            try std.testing.expectEqual(k, planned.qkv.k_slice_id);
        },
        else => return error.ExpectedPlannedRmsGroupedQkvRegion,
    }
}

test "metal partition executor recognizes rms norm gated ffn graph region" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 2;
    const hidden: usize = 8;
    const intermediate: usize = 16;
    const residual = try b.parameter("residual", ml.graph.Shape.init(.f32, &.{ rows, hidden }));
    const norm_w = try b.parameter("model.layers.0.post_attention_layernorm.weight", ml.graph.Shape.init(.f32, &.{hidden}));
    const gate_w = try b.parameter("model.layers.0.mlp.gate_proj.weight", ml.graph.Shape.init(.f32, &.{ intermediate, hidden }));
    const up_w = try b.parameter("model.layers.0.mlp.up_proj.weight", ml.graph.Shape.init(.f32, &.{ intermediate, hidden }));
    const down_w = try b.parameter("model.layers.0.mlp.down_proj.weight", ml.graph.Shape.init(.f32, &.{ hidden, intermediate }));
    const normed = try b.rmsNorm(residual, norm_w, @intCast(hidden), 1e-5);
    const pair = try g.addNode(.{
        .op = .{ .fused_linear_no_bias_pair = .{
            .rows = @intCast(rows),
            .in_dim = @intCast(hidden),
            .out_dim = @intCast(intermediate),
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ rows, intermediate }),
        .inputs = .{ normed, gate_w, up_w, null_node },
        .num_inputs = 3,
    });
    const activated = try b.silu(pair);
    const pair_second = try g.addNode(.{
        .op = .{ .fused_to_float32 = {} },
        .output_shape = ml.graph.Shape.init(.f32, &.{ rows, intermediate }),
        .inputs = .{ pair, null_node, null_node, null_node },
        .num_inputs = 1,
    });
    const gated = try b.elemMultiply(activated, pair_second);
    const down = try b.linearNoBias(gated, down_w, @intCast(rows), @intCast(intermediate), @intCast(hidden));
    const out = try b.add(down, residual);
    try g.markOutput(out);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);
    const node_ids = try allocator.alloc(NodeId, @intCast(g.nodeCount()));
    defer allocator.free(node_ids);
    for (node_ids, 0..) |*node_id, idx| node_id.* = @intCast(idx);
    const skipped = try allocator.alloc(bool, node_ids.len);
    defer allocator.free(skipped);
    @memset(skipped, false);

    const pattern = matchRmsNormGatedFfnResidualPattern(&g, node_ids, @intCast(normed), reachable, skipped, last_use) orelse return error.ExpectedRmsGatedFfnRegion;
    try std.testing.expectEqual(normed, pattern.norm_id);
    try std.testing.expectEqual(residual, pattern.norm_input_id);
    try std.testing.expectEqual(norm_w, pattern.norm_weight_id);
    try std.testing.expectEqual(pair, pattern.ffn.pair_id);
    try std.testing.expectEqual(pair_second, pattern.ffn.pair_second_id);
    try std.testing.expectEqual(activated, pattern.ffn.activation_id);
    try std.testing.expectEqual(gated, pattern.ffn.multiply_id);
    try std.testing.expectEqual(down, pattern.ffn.down_id);
    try std.testing.expectEqual(out, pattern.ffn.add_id);
    try std.testing.expectEqual(residual, pattern.ffn.residual_id);
    try std.testing.expectEqual(ops_mod.DecoderRuntimeActivationKind.silu, pattern.ffn.activation);

    var plan = try buildRuntimeRegionPlan(allocator, &g, node_ids, @intCast(g.nodeCount()), reachable, last_use);
    defer plan.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), plan.region_count);
    switch (plan.regionAt(@intCast(normed), normed, node_ids)) {
        .rms_norm_gated_ffn_residual => |planned| {
            try std.testing.expectEqual(normed, planned.norm_id);
            try std.testing.expectEqual(pair, planned.ffn.pair_id);
            try std.testing.expectEqual(out, planned.ffn.add_id);
        },
        else => return error.ExpectedPlannedRmsNormGatedFfnRegion,
    }
}

test "metal partition executor recognizes ple residual graph region" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 2;
    const hidden: usize = 8;
    const ple_hidden: usize = 4;
    const hidden_in = try b.parameter("hidden", ml.graph.Shape.init(.f32, &.{ rows, hidden }));
    const ple = try b.parameter("ple", ml.graph.Shape.init(.f32, &.{ rows, ple_hidden }));
    const gate_w = try b.parameter("model.layers.0.ple.gate_proj.weight", ml.graph.Shape.init(.f32, &.{ ple_hidden, hidden }));
    const proj_w = try b.parameter("model.layers.0.ple.down_proj.weight", ml.graph.Shape.init(.f32, &.{ hidden, ple_hidden }));
    const norm_w = try b.parameter("model.layers.0.ple.post_norm.weight", ml.graph.Shape.init(.f32, &.{hidden}));
    const gate = try b.linearNoBias(hidden_in, gate_w, @intCast(rows), @intCast(hidden), @intCast(ple_hidden));
    const activated = try b.gelu(gate);
    const modulated = try b.elemMultiply(activated, ple);
    const projected = try b.linearNoBias(modulated, proj_w, @intCast(rows), @intCast(ple_hidden), @intCast(hidden));
    const post_norm = try b.rmsNorm(projected, norm_w, @intCast(hidden), 1e-5);
    const out = try b.add(hidden_in, post_norm);
    try g.markOutput(out);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const node_ids = try allocator.alloc(NodeId, @intCast(g.nodeCount()));
    defer allocator.free(node_ids);
    for (node_ids, 0..) |*node_id, idx| node_id.* = @intCast(idx);
    const skipped = try allocator.alloc(bool, node_ids.len);
    defer allocator.free(skipped);
    @memset(skipped, false);

    const pattern = matchPleResidualPattern(&g, node_ids, @intCast(gate), reachable, skipped) orelse return error.ExpectedPleResidualRegion;
    try std.testing.expectEqual(gate, pattern.gate_id);
    try std.testing.expectEqual(activated, pattern.activation_id);
    try std.testing.expectEqual(modulated, pattern.multiply_id);
    try std.testing.expectEqual(projected, pattern.projection_id);
    try std.testing.expectEqual(post_norm, pattern.post_norm_id);
    try std.testing.expectEqual(out, pattern.add_id);
    try std.testing.expectEqual(hidden_in, pattern.hidden_id);
    try std.testing.expectEqual(ple, pattern.ple_id);
    try std.testing.expectEqual(gate_w, pattern.gate_weight_id);
    try std.testing.expectEqual(proj_w, pattern.projection_weight_id);
    try std.testing.expectEqual(norm_w, pattern.post_norm_weight_id);
    try std.testing.expectEqual(@as(usize, rows), pattern.rows);
    try std.testing.expectEqual(@as(usize, hidden), pattern.hidden_size);
    try std.testing.expectEqual(@as(usize, ple_hidden), pattern.ple_hidden_size);
    try std.testing.expectEqual(ops_mod.DecoderRuntimeActivationKind.gelu, pattern.activation);

    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);
    var plan = try buildRuntimeRegionPlan(allocator, &g, node_ids, @intCast(g.nodeCount()), reachable, last_use);
    defer plan.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), plan.region_count);
    switch (plan.regionAt(@intCast(gate), gate, node_ids)) {
        .ple_residual => |planned| {
            try std.testing.expectEqual(gate, planned.gate_id);
            try std.testing.expectEqual(projected, planned.projection_id);
            try std.testing.expectEqual(out, planned.add_id);
        },
        else => return error.ExpectedPlannedPleResidualRegion,
    }
}

test "metal partition executor prepared runtime regions count cached slots" {
    const attention_prepared: PreparedRuntimeRegion = .{
        .attention_output_residual = .{
            .linear_slot = 10,
            .pre_linear_rms_norm_slot = 11,
            .post_linear_rms_norm_slot = 12,
        },
    };
    const ffn_prepared: PreparedRuntimeRegion = .{
        .gated_ffn_residual = .{
            .gate_slot = 20,
            .up_slot = 21,
            .down_slot = 22,
            .post_down_rms_norm_slot = null,
        },
    };
    const rms_ffn_prepared: PreparedRuntimeRegion = .{
        .rms_norm_gated_ffn_residual = .{
            .norm_slot = 30,
            .ffn = .{
                .gate_slot = 31,
                .up_slot = 32,
                .down_slot = 33,
                .post_down_rms_norm_slot = 34,
            },
        },
    };
    const ple_prepared: PreparedRuntimeRegion = .{
        .ple_residual = .{
            .gate_slot = 40,
            .projection_slot = 41,
            .post_norm_slot = 42,
        },
    };

    try std.testing.expectEqual(@as(u64, 3), preparedRuntimeRegionSlotCount(attention_prepared));
    try std.testing.expectEqual(@as(u64, 3), preparedRuntimeRegionSlotCount(ffn_prepared));
    try std.testing.expectEqual(@as(u64, 5), preparedRuntimeRegionSlotCount(rms_ffn_prepared));
    try std.testing.expectEqual(@as(u64, 3), preparedRuntimeRegionSlotCount(ple_prepared));
    try std.testing.expect(preparedRuntimeRegionMatches(.{
        .attention_output_residual = .{
            .attention_id = 1,
            .pre_linear_norm_id = null,
            .linear_id = 2,
            .post_linear_norm_id = null,
            .add_id = 3,
            .residual_id = 4,
            .rows = 1,
            .attention_input_size = 8,
            .hidden_size = 8,
            .eps = 1e-5,
        },
    }, attention_prepared));
}

test "metal partition executor owned runtime region plan reuses cached plan" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const input = try b.parameter("input", ml.graph.Shape.init(.f32, &.{ 1, 4 }));
    try g.markOutput(input);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);
    const node_ids = try allocator.alloc(NodeId, @intCast(g.nodeCount()));
    defer allocator.free(node_ids);
    for (node_ids, 0..) |*node_id, idx| node_id.* = @intCast(idx);

    var weight_store = native_compute.WeightStore{ .allocator = allocator, .resident_weights = .{}, .lazy_weights = .{} };
    defer deinitEmptyNativeWeightStore(&weight_store, allocator);
    var compute = native_compute.NativeCompute.init(allocator, &weight_store, null);
    var cb = compute.computeBackend();
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    exec.owned = true;
    defer if (exec.runtime_region_plan) |*plan| plan.deinit(allocator);

    var stats: PartitionExecutor.ExecutionStats = .{};
    var transient: ?RuntimeRegionPlan = null;
    _ = try exec.runtimeRegionPlan(allocator, &g, node_ids, @intCast(g.nodeCount()), reachable, last_use, &stats, &transient);
    try std.testing.expectEqual(@as(u64, 1), stats.runtime_region_plan_compiles);
    try std.testing.expectEqual(@as(u64, 0), stats.runtime_region_plan_reuses);

    _ = try exec.runtimeRegionPlan(allocator, &g, node_ids, @intCast(g.nodeCount()), reachable, last_use, &stats, &transient);
    try std.testing.expectEqual(@as(u64, 1), stats.runtime_region_plan_compiles);
    try std.testing.expectEqual(@as(u64, 1), stats.runtime_region_plan_reuses);
}

test "metal partition executor runtime frame eligibility recognizes layer triples" {
    var regions = [_]RuntimeRegion{
        .{ .linear_qkv = .{
            .q_id = 1,
            .k_id = 2,
            .v_id = 3,
            .input_id = 0,
            .q_weight_id = 4,
            .k_weight_id = 5,
            .v_weight_id = 6,
            .rows = 2,
            .in_dim = 8,
            .q_out_dim = 16,
            .kv_out_dim = 4,
        } },
        .{ .attention_output_residual = .{
            .attention_id = 10,
            .pre_linear_norm_id = null,
            .linear_id = 11,
            .post_linear_norm_id = null,
            .add_id = 12,
            .residual_id = 1,
            .rows = 2,
            .attention_input_size = 16,
            .hidden_size = 8,
            .eps = 1e-5,
        } },
        .{ .rms_norm_gated_ffn_residual = .{
            .norm_id = 20,
            .norm_input_id = 12,
            .norm_weight_id = 21,
            .norm_dim = 8,
            .norm_eps = 1e-5,
            .ffn = .{
                .pair_id = 22,
                .pair_second_id = 23,
                .activation_id = 24,
                .multiply_id = 25,
                .down_id = 26,
                .post_down_norm_id = null,
                .add_id = 27,
                .residual_id = 12,
                .activation = .silu,
                .hidden_size = 8,
                .intermediate_size = 32,
                .rows = 2,
                .eps = 1e-5,
            },
        } },
        .{ .ple_residual = .{
            .gate_id = 30,
            .activation_id = 31,
            .multiply_id = 32,
            .projection_id = 33,
            .post_norm_id = 34,
            .add_id = 35,
            .hidden_id = 27,
            .ple_id = 2,
            .gate_weight_id = 36,
            .projection_weight_id = 37,
            .post_norm_weight_id = 38,
            .rows = 2,
            .hidden_size = 8,
            .ple_hidden_size = 4,
            .eps = 1e-5,
            .activation = .gelu,
        } },
    };
    const plan = RuntimeRegionPlan{
        .regions_by_pos = regions[0..],
        .region_count = regions.len,
    };

    const eligibility = analyzeRuntimeFrameEligibility(plan);
    try std.testing.expectEqual(@as(usize, 1), eligibility.layers);
    try std.testing.expectEqual(RuntimeFrameIneligibleReason.missing_model_metadata, eligibility.reason);

    var stats: PartitionExecutor.ExecutionStats = .{};
    recordRuntimeFrameEligibilityStats(&stats, eligibility);
    try std.testing.expectEqual(@as(u64, 1), stats.runtime_frame_candidates);
    try std.testing.expectEqual(@as(u64, 0), stats.runtime_frame_eligible);
    try std.testing.expectEqual(@as(u64, 1), stats.runtime_frame_ineligible_missing_model_metadata);

    for (&regions) |*region| switch (region.*) {
        .linear_qkv => |*pattern| pattern.rows = 1,
        .attention_output_residual => |*pattern| pattern.rows = 1,
        .rms_norm_gated_ffn_residual => |*pattern| pattern.ffn.rows = 1,
        .gated_ffn_residual => |*pattern| pattern.rows = 1,
        .ple_residual => |*pattern| pattern.rows = 1,
        else => {},
    };
    const single_row_eligibility = analyzeRuntimeFrameEligibility(plan);
    try std.testing.expectEqual(@as(usize, 1), single_row_eligibility.layers);
    try std.testing.expectEqual(RuntimeFrameIneligibleReason.single_row, single_row_eligibility.reason);
}

test "metal partition executor runtime frame eligibility rejects incomplete layer triples" {
    var regions = [_]RuntimeRegion{
        .{ .linear_qkv = .{
            .q_id = 1,
            .k_id = 2,
            .v_id = 3,
            .input_id = 0,
            .q_weight_id = 4,
            .k_weight_id = 5,
            .v_weight_id = 6,
            .rows = 2,
            .in_dim = 8,
            .q_out_dim = 16,
            .kv_out_dim = 4,
        } },
        .{ .attention_output_residual = .{
            .attention_id = 10,
            .pre_linear_norm_id = null,
            .linear_id = 11,
            .post_linear_norm_id = null,
            .add_id = 12,
            .residual_id = 1,
            .rows = 2,
            .attention_input_size = 16,
            .hidden_size = 8,
            .eps = 1e-5,
        } },
        .{ .gated_ffn_residual = .{
            .pair_id = 22,
            .pair_second_id = 23,
            .activation_id = 24,
            .multiply_id = 25,
            .down_id = 26,
            .post_down_norm_id = null,
            .add_id = 27,
            .residual_id = 12,
            .activation = .silu,
            .hidden_size = 8,
            .intermediate_size = 32,
            .rows = 2,
            .eps = 1e-5,
        } },
    };
    const plan = RuntimeRegionPlan{
        .regions_by_pos = regions[0..],
        .region_count = regions.len,
    };

    const eligibility = analyzeRuntimeFrameEligibility(plan);
    try std.testing.expectEqual(@as(usize, 0), eligibility.layers);
    try std.testing.expectEqual(RuntimeFrameIneligibleReason.missing_ple, eligibility.reason);
}

test "metal partition executor derives runtime frame metadata with variable shared head dims" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const x0 = try b.parameter("x0", ml.graph.Shape.init(.f32, &.{ 2, 8 }));
    const q0_w = try b.parameter("model.layers.0.self_attn.q_proj.weight", ml.graph.Shape.init(.f32, &.{ 16, 8 }));
    const k0_w = try b.parameter("model.layers.0.self_attn.k_proj.weight", ml.graph.Shape.init(.f32, &.{ 4, 8 }));
    const v0_w = try b.parameter("model.layers.0.self_attn.v_proj.weight", ml.graph.Shape.init(.f32, &.{ 4, 8 }));
    const q1_w = try b.parameter("model.layers.1.self_attn.q_proj.weight", ml.graph.Shape.init(.f32, &.{ 32, 8 }));
    const attn0 = try g.addNode(.{
        .op = .{ .fused_gqa_causal_attention = .{
            .batch = 1,
            .seq_len = 2,
            .num_heads = 4,
            .num_kv_heads = 1,
            .head_dim = 4,
            .layer_index = 0,
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ 2, 16 }),
        .inputs = .{ 100, 101, 102, null_node },
        .num_inputs = 3,
    });
    const attn1 = try g.addNode(.{
        .op = .{ .fused_gqa_causal_attention = .{
            .batch = 1,
            .seq_len = 2,
            .num_heads = 4,
            .num_kv_heads = 1,
            .head_dim = 8,
            .layer_index = 0,
            .skip_kv_write = true,
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ 2, 32 }),
        .inputs = .{ 103, 101, 102, null_node },
        .num_inputs = 3,
    });
    _ = x0;

    var regions = [_]RuntimeRegion{
        .{ .linear_qkv = .{
            .q_id = 10,
            .k_id = 11,
            .v_id = 12,
            .input_id = 0,
            .q_weight_id = q0_w,
            .k_weight_id = k0_w,
            .v_weight_id = v0_w,
            .rows = 2,
            .in_dim = 8,
            .q_out_dim = 16,
            .kv_out_dim = 4,
        } },
        .{ .attention_output_residual = .{
            .attention_id = attn0,
            .pre_linear_norm_id = null,
            .linear_id = 13,
            .post_linear_norm_id = null,
            .add_id = 14,
            .residual_id = 0,
            .rows = 2,
            .attention_input_size = 16,
            .hidden_size = 8,
            .eps = 1e-5,
        } },
        .{ .gated_ffn_residual = .{
            .pair_id = 15,
            .pair_second_id = 16,
            .activation_id = 17,
            .multiply_id = 18,
            .down_id = 19,
            .post_down_norm_id = null,
            .add_id = 20,
            .residual_id = 14,
            .activation = .gelu,
            .hidden_size = 8,
            .intermediate_size = 32,
            .rows = 2,
            .eps = 1e-5,
        } },
        .{ .ple_residual = .{
            .gate_id = 21,
            .activation_id = 22,
            .multiply_id = 23,
            .projection_id = 24,
            .post_norm_id = 25,
            .add_id = 26,
            .hidden_id = 20,
            .ple_id = 2,
            .gate_weight_id = 27,
            .projection_weight_id = 28,
            .post_norm_weight_id = 29,
            .rows = 2,
            .hidden_size = 8,
            .ple_hidden_size = 4,
            .eps = 1e-5,
            .activation = .gelu,
        } },
        .{ .q_linear = .{
            .id = 30,
            .input_id = 26,
            .weight_id = q1_w,
            .rows = 2,
            .in_dim = 8,
            .out_dim = 32,
        } },
        .{ .attention_output_residual = .{
            .attention_id = attn1,
            .pre_linear_norm_id = null,
            .linear_id = 31,
            .post_linear_norm_id = null,
            .add_id = 32,
            .residual_id = 26,
            .rows = 2,
            .attention_input_size = 32,
            .hidden_size = 8,
            .eps = 1e-5,
        } },
        .{ .gated_ffn_residual = .{
            .pair_id = 33,
            .pair_second_id = 34,
            .activation_id = 35,
            .multiply_id = 36,
            .down_id = 37,
            .post_down_norm_id = null,
            .add_id = 38,
            .residual_id = 32,
            .activation = .gelu,
            .hidden_size = 8,
            .intermediate_size = 32,
            .rows = 2,
            .eps = 1e-5,
        } },
        .{ .ple_residual = .{
            .gate_id = 39,
            .activation_id = 40,
            .multiply_id = 41,
            .projection_id = 42,
            .post_norm_id = 43,
            .add_id = 44,
            .hidden_id = 38,
            .ple_id = 3,
            .gate_weight_id = 45,
            .projection_weight_id = 46,
            .post_norm_weight_id = 47,
            .rows = 2,
            .hidden_size = 8,
            .ple_hidden_size = 4,
            .eps = 1e-5,
            .activation = .gelu,
        } },
    };
    const plan = RuntimeRegionPlan{
        .regions_by_pos = regions[0..],
        .region_count = regions.len,
    };

    const metadata = runtimeFrameMetadataFromPlan(&g, plan) orelse return error.ExpectedRuntimeFrameMetadata;
    try std.testing.expectEqual(@as(usize, 2), metadata.layer_count);
    try std.testing.expectEqual(@as(usize, 2), metadata.rows);
    try std.testing.expectEqual(@as(usize, 8), metadata.hidden_size);
    try std.testing.expectEqual(@as(usize, 4), metadata.num_attention_heads);
    try std.testing.expectEqual(@as(usize, 8), metadata.global_head_dim);
    try std.testing.expectEqual(@as(usize, 4), metadata.ple_hidden_size);
    try std.testing.expectEqual(ops_mod.DecoderRuntimeActivationKind.gelu, metadata.activation);
}

test "metal partition executor rejects gated ffn pattern with escaped intermediate" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 1;
    const hidden: usize = 3;
    const intermediate: usize = 5;
    const residual = try b.parameter("residual", ml.graph.Shape.init(.f32, &.{ rows, hidden }));
    const norm_w = try b.parameter("norm_w", ml.graph.Shape.init(.f32, &.{hidden}));
    const gate_w = try b.parameter("gate_w", ml.graph.Shape.init(.f32, &.{ intermediate, hidden }));
    const up_w = try b.parameter("up_w", ml.graph.Shape.init(.f32, &.{ intermediate, hidden }));
    const down_w = try b.parameter("down_w", ml.graph.Shape.init(.f32, &.{ hidden, intermediate }));
    const normed = try b.rmsNorm(residual, norm_w, @intCast(hidden), 1e-5);
    const pair = try g.addNode(.{
        .op = .{ .fused_linear_no_bias_pair = .{
            .rows = @intCast(rows),
            .in_dim = @intCast(hidden),
            .out_dim = @intCast(intermediate),
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ rows, intermediate }),
        .inputs = .{ normed, gate_w, up_w, null_node },
        .num_inputs = 3,
    });
    const activated = try b.silu(pair);
    const pair_second = try g.addNode(.{
        .op = .{ .fused_to_float32 = {} },
        .output_shape = ml.graph.Shape.init(.f32, &.{ rows, intermediate }),
        .inputs = .{ pair, null_node, null_node, null_node },
        .num_inputs = 1,
    });
    const escaped = try b.add(pair, pair_second);
    const gated = try b.elemMultiply(activated, pair_second);
    const down = try b.linearNoBias(gated, down_w, @intCast(rows), @intCast(intermediate), @intCast(hidden));
    const out = try b.add(down, residual);
    try g.markOutput(out);
    try g.markOutput(escaped);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);
    const node_ids = try allocator.alloc(NodeId, @intCast(g.nodeCount()));
    defer allocator.free(node_ids);
    for (node_ids, 0..) |*node_id, idx| node_id.* = @intCast(idx);
    const skipped = try allocator.alloc(bool, node_ids.len);
    defer allocator.free(skipped);
    @memset(skipped, false);

    try std.testing.expect(matchGatedFfnResidualPattern(&g, node_ids, @intCast(pair), reachable, skipped, last_use) == null);
}

test "metal partition executor runtime add keeps resident input device backed" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const dim: usize = 4;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ 1, dim }));
    const y = try b.parameter("y", ml.graph.Shape.init(.f32, &.{ 1, dim }));
    const sum = try b.add(x, y);

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const values = try allocator.alloc(?CT, @intCast(g.nodeCount()));
    defer allocator.free(values);
    @memset(values, null);

    const x_host = try cb.fromFloat32Shape(&.{ 1.0, 2.0, 3.0, 4.0 }, &.{ 1, dim });
    defer cb.free(x_host);
    const x_device = (try makeMetalDeviceResident(&cb, x_host)) orelse return error.SkipZigTest;
    defer cb.free(x_device);
    const y_host = try cb.fromFloat32Shape(&.{ 10.0, 20.0, 30.0, 40.0 }, &.{ 1, dim });
    defer cb.free(y_host);
    values[@intCast(x)] = x_device;
    values[@intCast(y)] = y_host;

    var exec_state = interpreter.ExecState{
        .attention_layer = 0,
        .options = .{},
        .last_use = &.{},
    };
    const out = (try tryExecuteMetalCommand(allocator, &g, &cb, values, sum, null, &exec_state)) orelse return error.UnsupportedPrimitiveOp;
    defer cb.free(out);
    try std.testing.expect(isMetalDeviceResident(&cb, out));

    const raw = try cb.toFloat32(out, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqualSlices(f32, &.{ 11.0, 22.0, 33.0, 44.0 }, raw);
}

test "metal partition executor runtime rms norm supports row-wise resident shapes" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 2;
    const dim: usize = 4;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ rows, dim }));
    const weight = try b.parameter("weight", ml.graph.Shape.init(.f32, &.{dim}));
    const normed = try b.rmsNorm(x, weight, dim, 0.0);

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const values = try allocator.alloc(?CT, @intCast(g.nodeCount()));
    defer allocator.free(values);
    @memset(values, null);

    const x_data = [_]f32{ 1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0 };
    const x_host = try cb.fromFloat32Shape(&x_data, &.{ rows, dim });
    defer cb.free(x_host);
    const x_device = (try makeMetalDeviceResident(&cb, x_host)) orelse return error.SkipZigTest;
    defer cb.free(x_device);
    const weight_host = try cb.fromFloat32Shape(&.{ 1.0, 1.0, 1.0, 1.0 }, &.{dim});
    defer cb.free(weight_host);
    values[@intCast(x)] = x_device;
    values[@intCast(weight)] = weight_host;

    var exec_state = interpreter.ExecState{
        .attention_layer = 0,
        .options = .{},
        .last_use = &.{},
    };
    const out = (try tryExecuteMetalCommand(allocator, &g, &cb, values, normed, null, &exec_state)) orelse return error.UnsupportedPrimitiveOp;
    defer cb.free(out);
    try std.testing.expect(isMetalDeviceResident(&cb, out));

    const raw = try cb.toFloat32(out, allocator);
    defer allocator.free(raw);
    const denom0: f32 = @sqrt((1.0 + 4.0 + 9.0 + 16.0) / 4.0);
    const denom1: f32 = @sqrt((4.0 + 16.0 + 36.0 + 64.0) / 4.0);
    const expected = [_]f32{
        1.0 / denom0, 2.0 / denom0, 3.0 / denom0, 4.0 / denom0,
        2.0 / denom1, 4.0 / denom1, 6.0 / denom1, 8.0 / denom1,
    };
    for (expected, raw) |exp, actual| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-5);
    }
}

test "metal partition executor resident primitive chain stays device backed" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const dim: usize = 4;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ 2, dim }));
    const e = try b.expOp(x);
    const l = try b.logOp(e);
    const product = try b.mul(e, l);
    const divided = try b.div(product, l);
    const diff = try b.sub(divided, l);
    const t = try b.tanhOp(diff);
    const a = try b.absOp(t);
    const out = try b.sliceLastDim(a, 1, 3);
    try g.markOutput(out);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &partition_mod.supportsAll },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    var native_weight_store = native_compute.WeightStore{ .allocator = allocator, .resident_weights = .{}, .lazy_weights = .{} };
    defer deinitEmptyNativeWeightStore(&native_weight_store, allocator);
    var native_compute_impl = native_compute.NativeCompute.init(allocator, &native_weight_store, null);
    var native_cb = native_compute_impl.computeBackend();
    var mesh = try device_mesh_mod.DeviceMesh.init(allocator, &.{
        .{ .id = 0, .backend = &native_cb, .kind = .native },
        .{ .id = 1, .backend = &cb, .kind = .metal },
    });
    defer mesh.deinit();

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const x_data = [_]f32{ 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0 };
    const x_ct = try native_cb.fromFloat32Shape(&x_data, &.{ 2, dim });
    defer native_cb.free(x_ct);
    values[@intCast(x)] = x_ct;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    const partition_index = partition_plan.node_assignment[out];
    var owned_runtime_transfers = std.AutoHashMapUnmanaged(NodeId, void).empty;
    defer owned_runtime_transfers.deinit(allocator);
    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 1, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .mesh = &mesh,
        .options = .{
            .runtime_inputs = &.{.{ .node_id = x, .value = x_ct }},
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .owned_runtime_transfers = &owned_runtime_transfers,
        .materialize_boundary_outputs = false,
        .stats = &exec_stats,
    });

    const out_index: usize = @intCast(out);
    defer if (values[out_index]) |ct| cb.free(ct);
    try std.testing.expect(isMetalDeviceResident(&cb, values[out_index].?));
    try std.testing.expect(exec_stats.backend_command_dispatches >= 7);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.host_materialized_outputs);

    const raw = try cb.toFloat32(values[out_index].?, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqual(@as(usize, 4), raw.len);
    for (0..2) |row| {
        for (0..2) |col| {
            const original = x_data[row * dim + col + 1];
            const expected = @abs(std.math.tanh(@exp(original) - original));
            try std.testing.expectApproxEqAbs(expected, raw[row * 2 + col], 1e-5);
        }
    }
}

test "metal partition executor resident concat prim stays device backed" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const lhs = try b.parameter("lhs", ml.graph.Shape.init(.f32, &.{ 2, 2 }));
    const rhs = try b.parameter("rhs", ml.graph.Shape.init(.f32, &.{ 2, 3 }));
    const out = try b.concat(lhs, rhs, 1);
    try g.markOutput(out);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &metal_capabilities.supportsMetalEagerGraph },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    var native_weight_store = native_compute.WeightStore{ .allocator = allocator, .resident_weights = .{}, .lazy_weights = .{} };
    defer deinitEmptyNativeWeightStore(&native_weight_store, allocator);
    var native_compute_impl = native_compute.NativeCompute.init(allocator, &native_weight_store, null);
    var native_cb = native_compute_impl.computeBackend();
    var mesh = try device_mesh_mod.DeviceMesh.init(allocator, &.{
        .{ .id = 0, .backend = &native_cb, .kind = .native },
        .{ .id = 1, .backend = &cb, .kind = .metal },
    });
    defer mesh.deinit();

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const lhs_data = [_]f32{ 1, 2, 3, 4 };
    const rhs_data = [_]f32{ 10, 11, 12, 13, 14, 15 };
    const lhs_ct = try native_cb.fromFloat32Shape(&lhs_data, &.{ 2, 2 });
    defer native_cb.free(lhs_ct);
    const rhs_ct = try native_cb.fromFloat32Shape(&rhs_data, &.{ 2, 3 });
    defer native_cb.free(rhs_ct);
    values[@intCast(lhs)] = lhs_ct;
    values[@intCast(rhs)] = rhs_ct;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    var owned_runtime_transfers = std.AutoHashMapUnmanaged(NodeId, void).empty;
    defer owned_runtime_transfers.deinit(allocator);
    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    const partition_index = partition_plan.node_assignment[out];
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 1, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .mesh = &mesh,
        .options = .{
            .runtime_inputs = &.{
                .{ .node_id = lhs, .value = lhs_ct },
                .{ .node_id = rhs, .value = rhs_ct },
            },
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .owned_runtime_transfers = &owned_runtime_transfers,
        .materialize_boundary_outputs = false,
        .stats = &exec_stats,
    });

    const out_index: usize = @intCast(out);
    defer if (values[out_index]) |ct| cb.free(ct);
    try std.testing.expect(isMetalDeviceResident(&cb, values[out_index].?));
    try std.testing.expectEqual(@as(u64, 1), exec_stats.backend_command_dispatches);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.host_materialized_outputs);

    const raw = try cb.toFloat32(values[out_index].?, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 10, 11, 12, 3, 4, 13, 14, 15 }, raw);
}

test "metal partition executor planned sdpa stays device backed without interpreter fallback" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const q = try b.parameter("q", ml.graph.Shape.init(.f32, &.{ 1, 2, 2 }));
    const k = try b.parameter("k", ml.graph.Shape.init(.f32, &.{ 1, 2, 2 }));
    const v = try b.parameter("v", ml.graph.Shape.init(.f32, &.{ 1, 2, 2 }));
    const out = try b.sdpa(q, k, v, 1, 2, 1, 2);
    try g.markOutput(out);

    const seeds = try partition_mod.allocTensorDescriptorSeeds(allocator, &g);
    defer allocator.free(seeds);
    try partition_mod.seedAllParameterResidency(seeds, &g, .metal, 0);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 20, .decide = &metal_capabilities.decideMetalEagerGraph },
        .{ .backend = .native, .priority = 0, .decide = &partition_mod.decideNative },
    };
    var partition_plan = try partition_mod.partitionWithOptions(allocator, &g, &caps, .{
        .tensor_descs = seeds,
    });
    defer partition_plan.deinit();
    try std.testing.expectEqual(operator_plan_mod.Operator.attention_flash, partition_plan.operatorPlanForNode(out).?.operator());

    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const q_data = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const k_data = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    const v_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const q_ct = try cb.fromFloat32Shape(&q_data, &.{ 1, 2, 2 });
    defer cb.free(q_ct);
    const k_ct = try cb.fromFloat32Shape(&k_data, &.{ 1, 2, 2 });
    defer cb.free(k_ct);
    const v_ct = try cb.fromFloat32Shape(&v_data, &.{ 1, 2, 2 });
    defer cb.free(v_ct);
    const q_dev = (try makeMetalDeviceResident(&cb, q_ct)) orelse return error.SkipZigTest;
    defer cb.free(q_dev);
    const k_dev = (try makeMetalDeviceResident(&cb, k_ct)) orelse return error.SkipZigTest;
    defer cb.free(k_dev);
    const v_dev = (try makeMetalDeviceResident(&cb, v_ct)) orelse return error.SkipZigTest;
    defer cb.free(v_dev);
    values[@intCast(q)] = q_dev;
    values[@intCast(k)] = k_dev;
    values[@intCast(v)] = v_dev;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    const partition_index = partition_plan.node_assignment[out];
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .options = .{
            .runtime_inputs = &.{
                .{ .node_id = q, .value = q_dev },
                .{ .node_id = k, .value = k_dev },
                .{ .node_id = v, .value = v_dev },
            },
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .materialize_boundary_outputs = false,
        .stats = &exec_stats,
    });

    const out_index: usize = @intCast(out);
    defer if (values[out_index]) |ct| cb.free(ct);
    try std.testing.expect(isMetalDeviceResident(&cb, values[out_index].?));
    try std.testing.expectEqual(@as(u64, 1), exec_stats.backend_command_dispatches);
    try std.testing.expectEqual(@as(u64, 1), exec_stats.planned_operator_dispatches);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.host_materialized_outputs);

    const raw = try cb.toFloat32(values[out_index].?, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqual(@as(usize, 4), raw.len);
    const inv_sqrt_2: f32 = 1.0 / @sqrt(@as(f32, 2.0));
    const p_diag = @exp(inv_sqrt_2) / (@exp(inv_sqrt_2) + @exp(@as(f32, 0.0)));
    const p_off = 1.0 - p_diag;
    const expected = [_]f32{
        p_diag * 1.0 + p_off * 3.0,
        p_diag * 2.0 + p_off * 4.0,
        p_off * 1.0 + p_diag * 3.0,
        p_off * 2.0 + p_diag * 4.0,
    };
    for (expected, raw) |exp, actual| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-4);
    }
}

test "metal partition executor planned sdpa bias and mask layouts stay device backed" {
    try runPlannedSdpaBiasMaskCase(.shared_heads);
    try runPlannedSdpaBiasMaskCase(.batched_heads);
    try runPlannedSdpaBiasMaskCase(.broadcast_head);
}

const TestSdpaBiasMode = enum {
    shared_heads,
    batched_heads,
    broadcast_head,
};

fn runPlannedSdpaBiasMaskCase(mode: TestSdpaBiasMode) !void {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    const batch: usize = 2;
    const num_heads: usize = 3;
    const seq_len: usize = 2;
    const head_dim: usize = 2;
    const total = batch * num_heads * seq_len * head_dim;

    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const q = try b.parameter("q", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(batch * num_heads)), seq_len, head_dim }));
    const k = try b.parameter("k", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(batch * num_heads)), seq_len, head_dim }));
    const v = try b.parameter("v", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(batch * num_heads)), seq_len, head_dim }));
    const bias_shape = sdpaBiasShape(mode, batch, num_heads, seq_len);
    const bias = try b.parameter("bias", bias_shape);
    const out = try g.addNode(.{
        .op = .{ .fused_sdpa = .{
            .batch = @intCast(batch),
            .seq_len = @intCast(seq_len),
            .num_heads = @intCast(num_heads),
            .head_dim = @intCast(head_dim),
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(batch * num_heads)), seq_len, head_dim }),
        .inputs = .{ q, k, v, bias },
        .num_inputs = 4,
    });
    try g.markOutput(out);

    const seeds = try partition_mod.allocTensorDescriptorSeeds(allocator, &g);
    defer allocator.free(seeds);
    try partition_mod.seedAllParameterResidency(seeds, &g, .metal, 0);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 20, .decide = &metal_capabilities.decideMetalEagerGraph },
        .{ .backend = .native, .priority = 0, .decide = &partition_mod.decideNative },
    };
    var partition_plan = try partition_mod.partitionWithOptions(allocator, &g, &caps, .{
        .tensor_descs = seeds,
    });
    defer partition_plan.deinit();
    try std.testing.expectEqual(operator_plan_mod.Operator.attention_flash, partition_plan.operatorPlanForNode(out).?.operator());

    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    var q_data: [total]f32 = undefined;
    var k_data: [total]f32 = undefined;
    var v_data: [total]f32 = undefined;
    @memset(q_data[0..], 0.0);
    @memset(k_data[0..], 0.0);
    for (&v_data, 0..) |*value, idx| value.* = @as(f32, @floatFromInt(idx + 1)) * 0.25;

    const bias_len: usize = @intCast(bias_shape.numElements().?);
    const bias_data = try allocator.alloc(f32, bias_len);
    defer allocator.free(bias_data);
    fillSdpaBiasData(mode, bias_data, batch, num_heads, seq_len);

    const qkv_shape = [_]i32{ @intCast(batch * num_heads), @intCast(seq_len), @intCast(head_dim) };
    var bias_shape_i32: [ml.graph.shape.max_rank]i32 = undefined;
    for (0..bias_shape.rank()) |axis| bias_shape_i32[axis] = @intCast(bias_shape.dim(@intCast(axis)));

    const q_ct = try cb.fromFloat32Shape(&q_data, &qkv_shape);
    defer cb.free(q_ct);
    const k_ct = try cb.fromFloat32Shape(&k_data, &qkv_shape);
    defer cb.free(k_ct);
    const v_ct = try cb.fromFloat32Shape(&v_data, &qkv_shape);
    defer cb.free(v_ct);
    const bias_ct = try cb.fromFloat32Shape(bias_data, bias_shape_i32[0..bias_shape.rank()]);
    defer cb.free(bias_ct);

    const q_dev = (try makeMetalDeviceResident(&cb, q_ct)) orelse return error.SkipZigTest;
    defer cb.free(q_dev);
    const k_dev = (try makeMetalDeviceResident(&cb, k_ct)) orelse return error.SkipZigTest;
    defer cb.free(k_dev);
    const v_dev = (try makeMetalDeviceResident(&cb, v_ct)) orelse return error.SkipZigTest;
    defer cb.free(v_dev);
    const bias_dev = (try makeMetalDeviceResident(&cb, bias_ct)) orelse return error.SkipZigTest;
    defer cb.free(bias_dev);
    values[@intCast(q)] = q_dev;
    values[@intCast(k)] = k_dev;
    values[@intCast(v)] = v_dev;
    values[@intCast(bias)] = bias_dev;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    const mask = [_]i64{ 1, 0, 1, 1 };
    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    const partition_index = partition_plan.node_assignment[out];
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .options = .{
            .runtime_inputs = &.{
                .{ .node_id = q, .value = q_dev },
                .{ .node_id = k, .value = k_dev },
                .{ .node_id = v, .value = v_dev },
                .{ .node_id = bias, .value = bias_dev },
            },
            .sdpa_mask = &mask,
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .materialize_boundary_outputs = false,
        .stats = &exec_stats,
    });

    const out_index: usize = @intCast(out);
    defer if (values[out_index]) |ct| cb.free(ct);
    try std.testing.expect(isMetalDeviceResident(&cb, values[out_index].?));
    try std.testing.expectEqual(@as(u64, 1), exec_stats.backend_command_dispatches);
    try std.testing.expectEqual(@as(u64, 1), exec_stats.planned_operator_dispatches);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.host_materialized_outputs);

    const raw = try cb.toFloat32(values[out_index].?, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqual(@as(usize, total), raw.len);

    var expected: [total]f32 = undefined;
    computeExpectedSdpaBiasMask(mode, &expected, &v_data, &mask, batch, num_heads, seq_len, head_dim);
    for (expected, raw) |exp, actual| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-4);
    }
}

fn sdpaBiasShape(mode: TestSdpaBiasMode, batch: usize, num_heads: usize, seq_len: usize) ml.graph.Shape {
    return switch (mode) {
        .shared_heads => ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(num_heads)), @as(i64, @intCast(seq_len)), @as(i64, @intCast(seq_len)) }),
        .batched_heads => ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(batch)), @as(i64, @intCast(num_heads)), @as(i64, @intCast(seq_len)), @as(i64, @intCast(seq_len)) }),
        .broadcast_head => ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(batch)), 1, @as(i64, @intCast(seq_len)), @as(i64, @intCast(seq_len)) }),
    };
}

fn fillSdpaBiasData(mode: TestSdpaBiasMode, bias: []f32, batch: usize, num_heads: usize, seq_len: usize) void {
    for (0..batch) |b| {
        for (0..num_heads) |h| {
            for (0..seq_len) |qi| {
                for (0..seq_len) |ki| {
                    const value = sdpaBiasValue(mode, b, h, qi, ki);
                    switch (mode) {
                        .shared_heads => bias[(h * seq_len + qi) * seq_len + ki] = value,
                        .batched_heads => bias[((b * num_heads + h) * seq_len + qi) * seq_len + ki] = value,
                        .broadcast_head => bias[(b * seq_len + qi) * seq_len + ki] = value,
                    }
                }
            }
        }
    }
}

fn sdpaBiasValue(mode: TestSdpaBiasMode, batch: usize, head: usize, query: usize, key: usize) f32 {
    const b: f32 = @floatFromInt(batch);
    const h: f32 = @floatFromInt(head);
    const q: f32 = @floatFromInt(query);
    const k: f32 = @floatFromInt(key);
    return switch (mode) {
        .shared_heads => 0.10 * h + 0.20 * q - 0.15 * k,
        .batched_heads => 0.30 * b + 0.10 * h + 0.20 * q - 0.15 * k,
        .broadcast_head => 0.35 * b + 0.20 * q - 0.15 * k,
    };
}

fn computeExpectedSdpaBiasMask(
    mode: TestSdpaBiasMode,
    expected: []f32,
    values: []const f32,
    mask: []const i64,
    batch: usize,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) void {
    for (0..batch) |b| {
        for (0..num_heads) |h| {
            const bh = b * num_heads + h;
            for (0..seq_len) |qi| {
                var best = -std.math.inf(f32);
                for (0..seq_len) |ki| {
                    if (mask[b * seq_len + ki] == 0) continue;
                    best = @max(best, sdpaBiasValue(mode, b, h, qi, ki));
                }
                var sum: f32 = 0.0;
                var weights: [2]f32 = .{ 0.0, 0.0 };
                for (0..seq_len) |ki| {
                    if (mask[b * seq_len + ki] == 0) continue;
                    const weight = @exp(sdpaBiasValue(mode, b, h, qi, ki) - best);
                    weights[ki] = weight;
                    sum += weight;
                }
                for (0..head_dim) |d| {
                    var accum: f32 = 0.0;
                    for (0..seq_len) |ki| {
                        if (weights[ki] == 0.0) continue;
                        accum += weights[ki] * values[(bh * seq_len + ki) * head_dim + d];
                    }
                    expected[(bh * seq_len + qi) * head_dim + d] = accum / sum;
                }
            }
        }
    }
}

test "metal partition executor resident last-dim reductions stay device backed" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 2;
    const dim: usize = 4;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ rows, dim }));
    const sum = try b.reduceSum(x, &.{1});
    const max = try b.reduceMax(x, &.{1});
    const mean = try b.reduceMean(x, &.{1});
    try g.markOutput(sum);
    try g.markOutput(max);
    try g.markOutput(mean);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &partition_mod.supportsAll },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    var native_weight_store = native_compute.WeightStore{ .allocator = allocator, .resident_weights = .{}, .lazy_weights = .{} };
    defer deinitEmptyNativeWeightStore(&native_weight_store, allocator);
    var native_compute_impl = native_compute.NativeCompute.init(allocator, &native_weight_store, null);
    var native_cb = native_compute_impl.computeBackend();
    var mesh = try device_mesh_mod.DeviceMesh.init(allocator, &.{
        .{ .id = 0, .backend = &native_cb, .kind = .native },
        .{ .id = 1, .backend = &cb, .kind = .metal },
    });
    defer mesh.deinit();

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const x_data = [_]f32{ -1.0, 2.0, 4.0, -3.0, 0.5, 1.5, -2.5, 3.5 };
    const x_ct = try native_cb.fromFloat32Shape(&x_data, &.{ rows, dim });
    defer native_cb.free(x_ct);
    values[@intCast(x)] = x_ct;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    const partition_index = partition_plan.node_assignment[sum];
    var owned_runtime_transfers = std.AutoHashMapUnmanaged(NodeId, void).empty;
    defer owned_runtime_transfers.deinit(allocator);
    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 1, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .mesh = &mesh,
        .options = .{
            .runtime_inputs = &.{.{ .node_id = x, .value = x_ct }},
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .owned_runtime_transfers = &owned_runtime_transfers,
        .materialize_boundary_outputs = false,
        .stats = &exec_stats,
    });

    const output_nodes = [_]NodeId{ sum, max, mean };
    defer for (output_nodes) |node_id| {
        const idx: usize = @intCast(node_id);
        if (values[idx]) |ct| cb.free(ct);
    };
    try std.testing.expect(exec_stats.backend_command_dispatches >= 3);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);
    for (output_nodes) |node_id| {
        try std.testing.expect(isMetalDeviceResident(&cb, values[@intCast(node_id)].?));
    }

    const expected_sum = [_]f32{ 2.0, 3.0 };
    const expected_max = [_]f32{ 4.0, 3.5 };
    const expected_mean = [_]f32{ 0.5, 0.75 };
    const expected = [_][]const f32{ &expected_sum, &expected_max, &expected_mean };
    for (output_nodes, expected) |node_id, expected_values| {
        const raw = try cb.toFloat32(values[@intCast(node_id)].?, allocator);
        defer allocator.free(raw);
        try std.testing.expectEqual(@as(usize, rows), raw.len);
        for (expected_values, raw) |exp, actual| {
            try std.testing.expectApproxEqAbs(exp, actual, 1e-5);
        }
    }
}

fn addBroadcastReducedForTest(g: *Graph, input: NodeId, target_shape: ml.graph.Shape) !NodeId {
    const reduced_shape = g.node(input).output_shape;
    if (reduced_shape.numElements() == target_shape.numElements()) return input;

    var attrs = ml.graph.node.BroadcastAttrs{ .target_shape = target_shape };
    const rank = reduced_shape.rank();
    for (0..rank) |axis| attrs.broadcast_axes[axis] = @intCast(axis);
    attrs.num_axes = @intCast(rank);
    return g.addNode(.{
        .op = .{ .broadcast_in_dim = attrs },
        .output_shape = target_shape,
        .inputs = .{ input, null_node, null_node, null_node },
        .num_inputs = 1,
    });
}

test "metal partition executor decomposed softmax stays device resident" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 2;
    const dim: usize = 4;
    const x_shape = ml.graph.Shape.init(.f32, &.{ rows, dim });
    const x = try b.parameter("x", x_shape);
    const max = try b.reduceMax(x, &.{1});
    const max_bc = try addBroadcastReducedForTest(&g, max, x_shape);
    const shifted = try b.sub(x, max_bc);
    const exp_shifted = try b.expOp(shifted);
    const denom = try b.reduceSum(exp_shifted, &.{1});
    const denom_bc = try addBroadcastReducedForTest(&g, denom, x_shape);
    const out = try b.div(exp_shifted, denom_bc);
    try g.markOutput(out);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &partition_mod.supportsAll },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const x_data = [_]f32{ -1.0, 2.0, 4.0, -3.0, 0.5, 1.5, -2.5, 3.5 };
    const x_ct = try cb.fromFloat32Shape(&x_data, &.{ rows, dim });
    defer cb.free(x_ct);
    const x_dev = (try makeMetalDeviceResident(&cb, x_ct)) orelse return error.SkipZigTest;
    defer cb.free(x_dev);
    values[@intCast(x)] = x_dev;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    const partition_index = partition_plan.node_assignment[out];
    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .options = .{
            .runtime_inputs = &.{.{ .node_id = x, .value = values[@intCast(x)].? }},
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .materialize_boundary_outputs = false,
        .stats = &exec_stats,
    });

    const out_index: usize = @intCast(out);
    defer if (values[out_index]) |ct| cb.free(ct);
    try std.testing.expect(isMetalDeviceResident(&cb, values[out_index].?));
    try std.testing.expect(exec_stats.backend_command_dispatches >= 7);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.host_materialized_outputs);

    const raw = try cb.toFloat32(values[out_index].?, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqual(rows * dim, raw.len);
    for (0..rows) |row| {
        const base = row * dim;
        var row_max = x_data[base];
        for (x_data[base + 1 .. base + dim]) |value| row_max = @max(row_max, value);
        var denom_host: f32 = 0.0;
        for (x_data[base .. base + dim]) |value| denom_host += @exp(value - row_max);
        for (0..dim) |col| {
            const expected = @exp(x_data[base + col] - row_max) / denom_host;
            try std.testing.expectApproxEqAbs(expected, raw[base + col], 1e-5);
        }
    }
}

test "metal partition executor resident where select chain stays device backed" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const dim: usize = 6;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{dim}));
    const threshold = try b.scalarConst(.f32, 0.0);
    const neg_one = try b.scalarConst(.f32, -1.0);
    const pos_one = try b.scalarConst(.f32, 1.0);
    const cond = try g.addNode(.{
        .op = .{ .less_than = {} },
        .output_shape = g.node(x).output_shape,
        .inputs = .{ x, threshold, null_node, null_node },
        .num_inputs = 2,
    });
    const out = try g.addNode(.{
        .op = .{ .where_select = {} },
        .output_shape = g.node(x).output_shape,
        .inputs = .{ cond, neg_one, pos_one, null_node },
        .num_inputs = 3,
    });
    try g.markOutput(out);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &partition_mod.supportsAll },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const x_data = [_]f32{ -3.0, -0.25, 0.0, 0.5, 2.0, -1.0 };
    const x_ct = try cb.fromFloat32Shape(&x_data, &.{dim});
    defer cb.free(x_ct);
    const threshold_ct = try cb.fromFloat32Shape(&.{0.0}, &.{1});
    defer cb.free(threshold_ct);
    const neg_one_ct = try cb.fromFloat32Shape(&.{-1.0}, &.{1});
    defer cb.free(neg_one_ct);
    const pos_one_ct = try cb.fromFloat32Shape(&.{1.0}, &.{1});
    defer cb.free(pos_one_ct);
    const x_dev = (try makeMetalDeviceResident(&cb, x_ct)) orelse return error.SkipZigTest;
    defer cb.free(x_dev);
    const threshold_dev = (try makeMetalDeviceResident(&cb, threshold_ct)) orelse return error.SkipZigTest;
    defer cb.free(threshold_dev);
    const neg_one_dev = (try makeMetalDeviceResident(&cb, neg_one_ct)) orelse return error.SkipZigTest;
    defer cb.free(neg_one_dev);
    const pos_one_dev = (try makeMetalDeviceResident(&cb, pos_one_ct)) orelse return error.SkipZigTest;
    defer cb.free(pos_one_dev);
    values[@intCast(x)] = x_dev;
    values[@intCast(threshold)] = threshold_dev;
    values[@intCast(neg_one)] = neg_one_dev;
    values[@intCast(pos_one)] = pos_one_dev;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    const partition_index = partition_plan.node_assignment[out];
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .options = .{
            .runtime_inputs = &.{
                .{ .node_id = x, .value = values[@intCast(x)].? },
                .{ .node_id = threshold, .value = values[@intCast(threshold)].? },
                .{ .node_id = neg_one, .value = values[@intCast(neg_one)].? },
                .{ .node_id = pos_one, .value = values[@intCast(pos_one)].? },
            },
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .materialize_boundary_outputs = false,
        .stats = &exec_stats,
    });

    const out_index: usize = @intCast(out);
    defer if (values[out_index]) |ct| cb.free(ct);
    try std.testing.expect(isMetalDeviceResident(&cb, values[out_index].?));
    try std.testing.expect(exec_stats.backend_command_dispatches >= 2);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.host_materialized_outputs);

    const raw = try cb.toFloat32(values[out_index].?, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqualSlices(f32, &[_]f32{ -1.0, -1.0, 1.0, 1.0, 1.0, -1.0 }, raw);
}

test "metal partition executor resident pair and fused unary commands stay device backed" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const dim: usize = 4;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ 1, dim }));
    const w_a = try b.parameter("w_a", ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const w_b = try b.parameter("w_b", ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const pair = try g.addNode(.{
        .op = .{ .fused_linear_no_bias_pair = .{ .rows = 1, .in_dim = dim, .out_dim = dim } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ 1, dim }),
        .inputs = .{ x, w_a, w_b, null_node },
        .num_inputs = 3,
    });
    const pair_second = try g.addNode(.{
        .op = .{ .fused_to_float32 = {} },
        .output_shape = ml.graph.Shape.init(.f32, &.{ 1, dim }),
        .inputs = .{ pair, null_node, null_node, null_node },
        .num_inputs = 1,
    });
    const sigmoid = try g.addNode(.{
        .op = .{ .fused_sigmoid = {} },
        .output_shape = ml.graph.Shape.init(.f32, &.{ 1, dim }),
        .inputs = .{ x, null_node, null_node, null_node },
        .num_inputs = 1,
    });
    const tanh_act = try g.addNode(.{
        .op = .{ .fused_tanh_act = {} },
        .output_shape = ml.graph.Shape.init(.f32, &.{ 1, dim }),
        .inputs = .{ x, null_node, null_node, null_node },
        .num_inputs = 1,
    });
    try g.markOutput(pair);
    try g.markOutput(pair_second);
    try g.markOutput(sigmoid);
    try g.markOutput(tanh_act);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &partition_mod.supportsAll },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const x_data = [_]f32{ -2.0, -0.5, 0.5, 2.0 };
    const w_a_data = [_]f32{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    };
    const w_b_data = [_]f32{
        2.0, 0.0, 0.0, 0.0,
        0.0, 2.0, 0.0, 0.0,
        0.0, 0.0, 2.0, 0.0,
        0.0, 0.0, 0.0, 2.0,
    };
    const x_ct = try cb.fromFloat32Shape(&x_data, &.{ 1, dim });
    defer cb.free(x_ct);
    const w_a_ct = try cb.fromFloat32Shape(&w_a_data, &.{ dim, dim });
    defer cb.free(w_a_ct);
    const w_b_ct = try cb.fromFloat32Shape(&w_b_data, &.{ dim, dim });
    defer cb.free(w_b_ct);
    const x_dev = (try makeMetalDeviceResident(&cb, x_ct)) orelse return error.SkipZigTest;
    defer if (x_dev != x_ct) cb.free(x_dev);
    const w_a_dev = (try makeMetalDeviceResident(&cb, w_a_ct)) orelse return error.SkipZigTest;
    defer if (w_a_dev != w_a_ct) cb.free(w_a_dev);
    const w_b_dev = (try makeMetalDeviceResident(&cb, w_b_ct)) orelse return error.SkipZigTest;
    defer if (w_b_dev != w_b_ct) cb.free(w_b_dev);
    values[@intCast(x)] = x_dev;
    values[@intCast(w_a)] = w_a_dev;
    values[@intCast(w_b)] = w_b_dev;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    const partition_index = partition_plan.node_assignment[pair];
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .options = .{
            .runtime_inputs = &.{
                .{ .node_id = x, .value = x_dev },
                .{ .node_id = w_a, .value = w_a_dev },
                .{ .node_id = w_b, .value = w_b_dev },
            },
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .materialize_boundary_outputs = false,
        .stats = &exec_stats,
    });

    const output_nodes = [_]NodeId{ pair, pair_second, sigmoid, tanh_act };
    defer for (output_nodes) |node_id| {
        const idx: usize = @intCast(node_id);
        if (values[idx]) |ct| cb.free(ct);
    };
    try std.testing.expect(exec_stats.backend_command_dispatches >= output_nodes.len);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.host_materialized_outputs);
    for (output_nodes) |node_id| {
        try std.testing.expect(isMetalDeviceResident(&cb, values[@intCast(node_id)].?));
    }

    const first_raw = try cb.toFloat32(values[@intCast(pair)].?, allocator);
    defer allocator.free(first_raw);
    const second_raw = try cb.toFloat32(values[@intCast(pair_second)].?, allocator);
    defer allocator.free(second_raw);
    try std.testing.expectApproxEqAbs(@as(f32, -2.0), first_raw[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), first_raw[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -4.0), second_raw[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), second_raw[3], 1e-5);
}

test "metal partition executor resident masked softmax projection chain stays device backed" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 2;
    const dim: usize = 4;
    const out_dim: usize = 3;
    const scores = try b.parameter("scores", ml.graph.Shape.init(.f32, &.{ rows, dim }));
    const projection = try b.parameter("projection", ml.graph.Shape.init(.f32, &.{ out_dim, dim }));
    const threshold = try b.scalarConst(.f32, 0.0);
    const masked_value = try b.scalarConst(.f32, -1.0e9);
    const keep_value = try b.scalarConst(.f32, 0.0);
    const cond = try g.addNode(.{
        .op = .{ .less_than = {} },
        .output_shape = g.node(scores).output_shape,
        .inputs = .{ scores, threshold, null_node, null_node },
        .num_inputs = 2,
    });
    const bias = try g.addNode(.{
        .op = .{ .where_select = {} },
        .output_shape = g.node(scores).output_shape,
        .inputs = .{ cond, masked_value, keep_value, null_node },
        .num_inputs = 3,
    });
    const masked = try b.add(scores, bias);
    const probs = try b.softmax(masked);
    const out = try b.linearNoBias(probs, projection, @intCast(rows), @intCast(dim), @intCast(out_dim));
    try g.markOutput(out);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &partition_mod.supportsAll },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const score_data = [_]f32{
        -2.0, 0.5,  1.5,  -0.25,
        2.0,  -1.0, 0.25, 1.0,
    };
    const projection_data = [_]f32{
        1.0,   0.0,  -0.5, 0.25,
        -0.25, 0.75, 0.5,  -1.0,
        0.5,   -0.5, 0.25, 1.0,
    };
    const score_ct = try cb.fromFloat32Shape(&score_data, &.{ rows, dim });
    defer cb.free(score_ct);
    const projection_ct = try cb.fromFloat32Shape(&projection_data, &.{ out_dim, dim });
    defer cb.free(projection_ct);
    const threshold_ct = try cb.fromFloat32Shape(&.{0.0}, &.{1});
    defer cb.free(threshold_ct);
    const masked_value_ct = try cb.fromFloat32Shape(&.{-1.0e9}, &.{1});
    defer cb.free(masked_value_ct);
    const keep_value_ct = try cb.fromFloat32Shape(&.{0.0}, &.{1});
    defer cb.free(keep_value_ct);

    const score_dev = (try makeMetalDeviceResident(&cb, score_ct)) orelse return error.SkipZigTest;
    defer cb.free(score_dev);
    const projection_dev = (try makeMetalDeviceResident(&cb, projection_ct)) orelse return error.SkipZigTest;
    defer cb.free(projection_dev);
    const threshold_dev = (try makeMetalDeviceResident(&cb, threshold_ct)) orelse return error.SkipZigTest;
    defer cb.free(threshold_dev);
    const masked_value_dev = (try makeMetalDeviceResident(&cb, masked_value_ct)) orelse return error.SkipZigTest;
    defer cb.free(masked_value_dev);
    const keep_value_dev = (try makeMetalDeviceResident(&cb, keep_value_ct)) orelse return error.SkipZigTest;
    defer cb.free(keep_value_dev);
    values[@intCast(scores)] = score_dev;
    values[@intCast(projection)] = projection_dev;
    values[@intCast(threshold)] = threshold_dev;
    values[@intCast(masked_value)] = masked_value_dev;
    values[@intCast(keep_value)] = keep_value_dev;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    const partition_index = partition_plan.node_assignment[out];
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .options = .{
            .runtime_inputs = &.{
                .{ .node_id = scores, .value = values[@intCast(scores)].? },
                .{ .node_id = projection, .value = values[@intCast(projection)].? },
                .{ .node_id = threshold, .value = values[@intCast(threshold)].? },
                .{ .node_id = masked_value, .value = values[@intCast(masked_value)].? },
                .{ .node_id = keep_value, .value = values[@intCast(keep_value)].? },
            },
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .materialize_boundary_outputs = false,
        .stats = &exec_stats,
    });

    const out_index: usize = @intCast(out);
    defer if (values[out_index]) |ct| cb.free(ct);
    try std.testing.expect(isMetalDeviceResident(&cb, values[out_index].?));
    try std.testing.expect(exec_stats.backend_command_dispatches >= 5);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.host_materialized_outputs);
    try std.testing.expect(exec_stats.graph_plan_slots_reserved > 0);
    try std.testing.expect(exec_stats.graph_plan_bytes_reserved >= rows * dim * @sizeOf(f32));

    const raw = try cb.toFloat32(values[out_index].?, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqual(rows * out_dim, raw.len);
    var expected: [rows * out_dim]f32 = undefined;
    for (0..rows) |row| {
        const base = row * dim;
        var masked_scores: [dim]f32 = undefined;
        var row_max: f32 = -std.math.inf(f32);
        for (0..dim) |col| {
            const value = if (score_data[base + col] < 0.0) -1.0e9 else score_data[base + col];
            masked_scores[col] = value;
            row_max = @max(row_max, value);
        }
        var denom: f32 = 0.0;
        var probs_host: [dim]f32 = undefined;
        for (0..dim) |col| {
            probs_host[col] = @exp(masked_scores[col] - row_max);
            denom += probs_host[col];
        }
        for (0..dim) |col| probs_host[col] /= denom;
        for (0..out_dim) |out_col| {
            var acc: f32 = 0.0;
            for (0..dim) |col| acc += probs_host[col] * projection_data[out_col * dim + col];
            expected[row * out_dim + out_col] = acc;
        }
    }
    for (expected, raw) |exp, actual| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-5);
    }
}

test "metal partition executor resident rope stays device backed" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ 1, 4 }));
    const rope = try g.addNode(.{
        .op = .{ .fused_rope = .{
            .seq_len = 1,
            .head_dim = 4,
            .rope_dim = 4,
            .theta = 10000.0,
            .freq_scale = 1.0,
            .position_offset = 0,
            .consecutive_pairs = false,
        } },
        .output_shape = g.node(x).output_shape,
        .inputs = .{ x, null_node, null_node, null_node },
        .num_inputs = 1,
    });
    try g.markOutput(rope);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &metal_capabilities.supportsMetalEagerGraph },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const input_ct = try cb.fromFloat32Shape(&.{ 1.0, 2.0, 3.0, 4.0 }, &.{ 1, 4 });
    defer cb.free(input_ct);
    const input_dev = (try makeMetalDeviceResident(&cb, input_ct)) orelse return error.SkipZigTest;
    defer cb.free(input_dev);
    values[@intCast(x)] = input_dev;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    const partition_index = partition_plan.node_assignment[rope];
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .options = .{
            .runtime_inputs = &.{
                .{ .node_id = x, .value = values[@intCast(x)].? },
            },
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .materialize_boundary_outputs = false,
        .stats = &exec_stats,
    });

    const out_index: usize = @intCast(rope);
    defer if (values[out_index]) |ct| cb.free(ct);
    try std.testing.expect(isMetalDeviceResident(&cb, values[out_index].?));
    try std.testing.expect(exec_stats.backend_command_dispatches >= 1);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);

    const raw = try cb.toFloat32(values[out_index].?, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0, 3.0, 4.0 }, raw);
}

test "metal partition executor resident zero tensor materializes without fallback" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();

    const zero = try g.addNode(.{
        .op = .{ .fused_zero_tensor = .{ .rows = 1, .in_dim = 0, .out_dim = 4 } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ 1, 4 }),
        .inputs = .{ null_node, null_node, null_node, null_node },
        .num_inputs = 0,
    });
    try g.markOutput(zero);

    const seeds = try partition_mod.allocTensorDescriptorSeeds(allocator, &g);
    defer allocator.free(seeds);
    try partition_mod.seedAllUploadableResidency(seeds, &g, .metal, 0);
    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .decide = &metal_capabilities.decideMetalEagerGraph },
        .{ .backend = .native, .priority = 0, .decide = &partition_mod.decideNative },
    };
    var partition_plan = try partition_mod.partitionWithOptions(allocator, &g, &caps, .{ .tensor_descs = seeds });
    defer partition_plan.deinit();
    try std.testing.expectEqual(contracts.BackendKind.metal, partition_plan.partitions[partition_plan.node_assignment[zero]].backend);
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const values = try allocator.alloc(?CT, @intCast(g.nodeCount()));
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, @intCast(g.nodeCount()));
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    const partition_index = partition_plan.node_assignment[zero];
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .materialize_boundary_outputs = false,
        .stats = &exec_stats,
    });

    defer if (values[@intCast(zero)]) |ct| cb.free(ct);
    try std.testing.expect(isMetalDeviceResident(&cb, values[@intCast(zero)].?));
    try std.testing.expectEqual(@as(u64, 0), exec_stats.backend_command_dispatches);
    try std.testing.expectEqual(@as(u64, 1), exec_stats.constant_materializations);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);
    const raw = try cb.toFloat32(values[@intCast(zero)].?, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqualSlices(f32, &.{ 0.0, 0.0, 0.0, 0.0 }, raw);
}

test "metal partition executor resident gqa attention uses command path" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const batch: usize = 1;
    const seq_len: usize = 1;
    const num_heads: usize = 1;
    const head_dim: usize = 4;
    const dim: usize = num_heads * head_dim;
    const q = try b.parameter("q", ml.graph.Shape.init(.f32, &.{ seq_len, dim }));
    const k = try b.parameter("k", ml.graph.Shape.init(.f32, &.{ seq_len, dim }));
    const v = try b.parameter("v", ml.graph.Shape.init(.f32, &.{ seq_len, dim }));
    const attn = try g.addNode(.{
        .op = .{ .fused_gqa_causal_attention = .{
            .batch = @intCast(batch),
            .seq_len = @intCast(seq_len),
            .kv_seq_len = @intCast(seq_len),
            .num_heads = @intCast(num_heads),
            .num_kv_heads = @intCast(num_heads),
            .head_dim = @intCast(head_dim),
        } },
        .output_shape = ml.graph.Shape.init(.f32, &.{ seq_len, dim }),
        .inputs = .{ q, k, v, null_node },
        .num_inputs = 3,
    });
    try g.markOutput(attn);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &metal_capabilities.supportsMetalEagerGraph },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const q_ct = try cb.fromFloat32Shape(&.{ 1.0, 0.0, 0.0, 0.0 }, &.{ seq_len, dim });
    defer cb.free(q_ct);
    const k_ct = try cb.fromFloat32Shape(&.{ 1.0, 0.0, 0.0, 0.0 }, &.{ seq_len, dim });
    defer cb.free(k_ct);
    const v_ct = try cb.fromFloat32Shape(&.{ 5.0, 6.0, 7.0, 8.0 }, &.{ seq_len, dim });
    defer cb.free(v_ct);
    const q_dev = (try makeMetalDeviceResident(&cb, q_ct)) orelse return error.SkipZigTest;
    defer cb.free(q_dev);
    const k_dev = (try makeMetalDeviceResident(&cb, k_ct)) orelse return error.SkipZigTest;
    defer cb.free(k_dev);
    const v_dev = (try makeMetalDeviceResident(&cb, v_ct)) orelse return error.SkipZigTest;
    defer cb.free(v_dev);
    values[@intCast(q)] = q_dev;
    values[@intCast(k)] = k_dev;
    values[@intCast(v)] = v_dev;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    const partition_index = partition_plan.node_assignment[attn];
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .options = .{
            .runtime_inputs = &.{
                .{ .node_id = q, .value = values[@intCast(q)].? },
                .{ .node_id = k, .value = values[@intCast(k)].? },
                .{ .node_id = v, .value = values[@intCast(v)].? },
            },
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .materialize_boundary_outputs = false,
        .stats = &exec_stats,
    });

    const out_index: usize = @intCast(attn);
    defer if (values[out_index]) |ct| cb.free(ct);
    try std.testing.expect(isMetalDeviceResident(&cb, values[out_index].?));
    try std.testing.expect(exec_stats.backend_command_dispatches >= 1);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);

    const raw = try cb.toFloat32(values[out_index].?, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqualSlices(f32, &.{ 5.0, 6.0, 7.0, 8.0 }, raw);
}

fn putTestQuantizedWeight(
    allocator: std.mem.Allocator,
    weight_store: *gpu_hosted_store_mod.WeightStore,
    name: []const u8,
    raw: []const u8,
    shape: []const i64,
    format: @import("quant_matmul.zig").Format,
) !void {
    try weight_store.lazy_weights.put(allocator, name, .{
        .tensor_ref = undefined,
        .quantized_storage = QuantizedStorage{
            .tensor_type = try quantFormatTensorType(format),
            .raw_bytes = raw,
            .shape = shape,
            .raw_owned = false,
            .allocator = allocator,
        },
    });
}

fn quantFormatTensorType(format: @import("quant_matmul.zig").Format) !@import("../gguf/tensor_types.zig").TensorType {
    return switch (format) {
        .q4_0 => .{ .known = .Q4_0 },
        .q4_1 => .{ .known = .Q4_1 },
        .q5_k => .{ .known = .Q5_K },
        .q8_0 => .{ .known = .Q8_0 },
        else => error.UnsupportedTensorType,
    };
}

fn quantizedBlockSize(format: @import("quant_matmul.zig").Format) !struct { values: usize, bytes: usize } {
    return switch (format) {
        .q4_0 => .{ .values = 32, .bytes = 18 },
        .q4_1 => .{ .values = 32, .bytes = 20 },
        .q5_k => .{ .values = 256, .bytes = 176 },
        .q8_0 => .{ .values = 32, .bytes = 34 },
        else => error.UnsupportedTensorType,
    };
}

fn quantizeLinearRowsForTest(
    allocator: std.mem.Allocator,
    format: @import("quant_matmul.zig").Format,
    dense: []const f32,
    out_dim: usize,
    in_dim: usize,
) ![]u8 {
    const layout = try quantizedBlockSize(format);
    if (in_dim % layout.values != 0 or dense.len != out_dim * in_dim) return error.UnsupportedShape;
    const blocks = in_dim / layout.values;
    const raw = try allocator.alloc(u8, out_dim * blocks * layout.bytes);
    errdefer allocator.free(raw);
    for (0..out_dim) |out_col| {
        for (0..blocks) |block| {
            const src = dense[out_col * in_dim + block * layout.values ..][0..layout.values];
            const dst = raw[(out_col * blocks + block) * layout.bytes ..][0..layout.bytes];
            switch (format) {
                .q4_0 => quant_codec.quantizeQ4_0Block(src, dst),
                .q4_1 => quant_codec.quantizeQ4_1Block(src, dst),
                .q5_k => quant_codec.quantizeQ5_KBlock(src, dst),
                .q8_0 => quant_codec.quantizeQ8_0Block(src, dst),
                else => return error.UnsupportedTensorType,
            }
        }
    }
    return raw;
}

test "metal partition executor resident qkv rope softmax projection chain stays device backed" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 1;
    const dim: usize = 4;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ rows, dim }));
    const q_weight_name = "model.layers.0.self_attn.q_proj.weight";
    const k_weight_name = "model.layers.0.self_attn.k_proj.weight";
    const v_weight_name = "model.layers.0.self_attn.v_proj.weight";
    const out_weight_name = "model.layers.0.self_attn.o_proj.weight";
    const q_weight = try b.parameter(q_weight_name, ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const k_weight = try b.parameter(k_weight_name, ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const v_weight = try b.parameter(v_weight_name, ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const out_weight = try b.parameter(out_weight_name, ml.graph.Shape.init(.f32, &.{ dim, dim }));

    const q = try b.linearNoBias(x, q_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    const k = try b.linearNoBias(x, k_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    const v = try b.linearNoBias(x, v_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    const q_3d = try b.reshape(q, ml.graph.Shape.init(.f32, &.{ 1, rows, dim }));
    const q_t = try b.transpose(q_3d, &.{ 0, 2, 1 });
    const q_tt = try b.transpose(q_t, &.{ 0, 2, 1 });
    const q_rope = try g.addNode(.{
        .op = .{ .fused_rope = .{
            .seq_len = rows,
            .head_dim = dim,
            .rope_dim = dim,
            .theta = 10000.0,
            .freq_scale = 1.0,
            .position_offset = 0,
            .consecutive_pairs = false,
        } },
        .output_shape = g.node(q_tt).output_shape,
        .inputs = .{ q_tt, null_node, null_node, null_node },
        .num_inputs = 1,
    });
    const q_back = try b.reshape(q_rope, ml.graph.Shape.init(.f32, &.{ rows, dim }));
    const logits = try b.add(q_back, k);
    const probs = try b.softmax(logits);
    const mixed = try b.add(probs, v);
    const out = try b.linearNoBias(mixed, out_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    try g.markOutput(out);

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 10, .supports = &metal_capabilities.supportsMetalEagerGraph },
    };
    var partition_plan = try partition_mod.partition(allocator, &g, &caps);
    defer partition_plan.deinit();
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const input_data = [_]f32{
        0.1, 0.2, 0.3, 0.4,
    };
    const identity = [_]f32{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    };

    const input_ct = try cb.fromFloat32Shape(&input_data, &.{ rows, dim });
    defer cb.free(input_ct);
    const q_weight_ct = try cb.fromFloat32Shape(&identity, &.{ dim, dim });
    defer cb.free(q_weight_ct);
    const k_weight_ct = try cb.fromFloat32Shape(&identity, &.{ dim, dim });
    defer cb.free(k_weight_ct);
    const v_weight_ct = try cb.fromFloat32Shape(&identity, &.{ dim, dim });
    defer cb.free(v_weight_ct);
    const out_weight_ct = try cb.fromFloat32Shape(&identity, &.{ dim, dim });
    defer cb.free(out_weight_ct);

    const input_dev = (try makeMetalDeviceResident(&cb, input_ct)) orelse return error.SkipZigTest;
    defer cb.free(input_dev);
    const q_weight_dev = (try makeMetalDeviceResident(&cb, q_weight_ct)) orelse return error.SkipZigTest;
    defer cb.free(q_weight_dev);
    const k_weight_dev = (try makeMetalDeviceResident(&cb, k_weight_ct)) orelse return error.SkipZigTest;
    defer cb.free(k_weight_dev);
    const v_weight_dev = (try makeMetalDeviceResident(&cb, v_weight_ct)) orelse return error.SkipZigTest;
    defer cb.free(v_weight_dev);
    const out_weight_dev = (try makeMetalDeviceResident(&cb, out_weight_ct)) orelse return error.SkipZigTest;
    defer cb.free(out_weight_dev);

    values[@intCast(x)] = input_dev;
    values[@intCast(q_weight)] = q_weight_dev;
    values[@intCast(k_weight)] = k_weight_dev;
    values[@intCast(v_weight)] = v_weight_dev;
    values[@intCast(out_weight)] = out_weight_dev;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    const executed_partitions = try allocator.alloc(bool, partition_plan.partitions.len);
    defer allocator.free(executed_partitions);
    @memset(executed_partitions, false);
    for (0..@intCast(g.nodeCount())) |node_index| {
        const node_id: NodeId = @intCast(node_index);
        const partition_index = partition_plan.node_assignment[node_id];
        if (partition_index >= partition_plan.partitions.len or executed_partitions[partition_index]) continue;
        executed_partitions[partition_index] = true;
        const part = partition_plan.partitions[partition_index];
        var has_reachable_compute_node = false;
        for (part.node_ids) |part_node_id| {
            const part_index: usize = @intCast(part_node_id);
            if (part_index >= reachable.len or !reachable[part_index]) continue;
            switch (g.node(part_node_id).op) {
                .parameter, .constant => {},
                else => has_reachable_compute_node = true,
            }
        }
        if (!has_reachable_compute_node) continue;
        try std.testing.expectEqual(contracts.BackendKind.metal, part.backend);
        try exec.partitionExecutor().execute(values, value_device, part.node_ids, 0, .{
            .allocator = allocator,
            .graph = &g,
            .backend = &cb,
            .options = .{
                .runtime_inputs = &.{
                    .{ .node_id = x, .value = values[@intCast(x)].? },
                    .{ .node_id = q_weight, .value = values[@intCast(q_weight)].? },
                    .{ .node_id = k_weight, .value = values[@intCast(k_weight)].? },
                    .{ .node_id = v_weight, .value = values[@intCast(v_weight)].? },
                    .{ .node_id = out_weight, .value = values[@intCast(out_weight)].? },
                },
            },
            .reachable = reachable,
            .last_use = last_use,
            .partition_plan = &partition_plan,
            .buffer_plan = &buffer_plan,
            .materialize_boundary_outputs = false,
            .stats = &exec_stats,
        });
    }

    const out_index: usize = @intCast(out);
    defer if (values[out_index]) |ct| cb.free(ct);
    try std.testing.expect(isMetalDeviceResident(&cb, values[out_index].?));
    try std.testing.expect(exec_stats.backend_command_dispatches >= 1);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.host_materialized_outputs);
    try std.testing.expectEqual(@as(u64, 3), exec_stats.gemma_qkv_hits);
    try std.testing.expectEqual(@as(u64, 1), exec_stats.gemma_o_proj_hits);
    try std.testing.expectEqual(@as(u64, 1), exec_stats.gemma_softmax_hits);
    try std.testing.expectEqual(@as(u64, 2), exec_stats.gemma_residual_add_hits);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.gemma_qkv_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.gemma_o_proj_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.gemma_softmax_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.gemma_residual_add_fallbacks);

    const raw = try cb.toFloat32(values[out_index].?, allocator);
    defer allocator.free(raw);
    var expected: [rows * dim]f32 = undefined;
    for (0..rows) |row| {
        const base = row * dim;
        var row_max: f32 = -std.math.inf(f32);
        var logits_host: [dim]f32 = undefined;
        for (0..dim) |col| {
            logits_host[col] = input_data[base + col] * 2.0;
            row_max = @max(row_max, logits_host[col]);
        }
        var denom: f32 = 0.0;
        var probs_host: [dim]f32 = undefined;
        for (0..dim) |col| {
            probs_host[col] = @exp(logits_host[col] - row_max);
            denom += probs_host[col];
        }
        for (0..dim) |col| {
            probs_host[col] /= denom;
            expected[base + col] = probs_host[col] + input_data[base + col];
        }
    }
    for (expected, raw) |exp, actual| {
        try std.testing.expectApproxEqAbs(exp, actual, 1e-5);
    }
}

test "metal partition executor quantized qkv projection chain keeps activation transpose resident" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const rows: usize = 1;
    const dim: usize = 32;
    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ rows, dim }));
    const q_weight_name = "model.layers.0.self_attn.q_proj.weight";
    const k_weight_name = "model.layers.0.self_attn.k_proj.weight";
    const v_weight_name = "model.layers.0.self_attn.v_proj.weight";
    const out_weight_name = "model.layers.0.self_attn.o_proj.weight";
    const q_weight = try b.parameter(q_weight_name, ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const k_weight = try b.parameter(k_weight_name, ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const v_weight = try b.parameter(v_weight_name, ml.graph.Shape.init(.f32, &.{ dim, dim }));
    const out_weight = try b.parameter(out_weight_name, ml.graph.Shape.init(.f32, &.{ dim, dim }));

    const q = try b.linearNoBias(x, q_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    const k = try b.linearNoBias(x, k_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    const v = try b.linearNoBias(x, v_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    const q_3d = try b.reshape(q, ml.graph.Shape.init(.f32, &.{ 1, rows, dim }));
    const q_t = try b.transpose(q_3d, &.{ 0, 2, 1 });
    const q_tt = try b.transpose(q_t, &.{ 0, 2, 1 });
    const q_rope = try g.addNode(.{
        .op = .{ .fused_rope = .{
            .seq_len = rows,
            .head_dim = dim,
            .rope_dim = dim,
            .theta = 10000.0,
            .freq_scale = 1.0,
            .position_offset = 0,
            .consecutive_pairs = false,
        } },
        .output_shape = g.node(q_tt).output_shape,
        .inputs = .{ q_tt, null_node, null_node, null_node },
        .num_inputs = 1,
    });
    const q_back = try b.reshape(q_rope, ml.graph.Shape.init(.f32, &.{ rows, dim }));
    const logits = try b.add(q_back, k);
    const probs = try b.softmax(logits);
    const mixed = try b.add(probs, v);
    const out = try b.linearNoBias(mixed, out_weight, @intCast(rows), @intCast(dim), @intCast(dim));
    try g.markOutput(out);

    const seeds = try partition_mod.allocTensorDescriptorSeeds(allocator, &g);
    defer allocator.free(seeds);
    try partition_mod.seedParameterResidency(seeds, &g, x, .metal, 0);
    try std.testing.expect(try partition_mod.seedParameterQuantFormatByName(seeds, &g, q_weight_name, .q8_0));
    try std.testing.expect(try partition_mod.seedParameterQuantFormatByName(seeds, &g, k_weight_name, .q8_0));
    try std.testing.expect(try partition_mod.seedParameterQuantFormatByName(seeds, &g, v_weight_name, .q8_0));
    try std.testing.expect(try partition_mod.seedParameterQuantFormatByName(seeds, &g, out_weight_name, .q8_0));

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 20, .decide = &metal_capabilities.decideMetalEagerGraph },
        .{ .backend = .native, .priority = 0, .decide = &partition_mod.decideNative },
    };
    var partition_plan = try partition_mod.partitionWithOptions(allocator, &g, &caps, .{
        .tensor_descs = seeds,
    });
    defer partition_plan.deinit();
    try std.testing.expectEqual(operator_plan_mod.Operator.mul_mv, partition_plan.operatorPlanForNode(q).?.operator());
    try std.testing.expectEqual(operator_plan_mod.Operator.mul_mv, partition_plan.operatorPlanForNode(k).?.operator());
    try std.testing.expectEqual(operator_plan_mod.Operator.mul_mv, partition_plan.operatorPlanForNode(v).?.operator());
    try std.testing.expectEqual(operator_plan_mod.Operator.mul_mv, partition_plan.operatorPlanForNode(out).?.operator());

    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    const identity = try allocator.alloc(f32, dim * dim);
    defer allocator.free(identity);
    @memset(identity, 0.0);
    for (0..dim) |i| identity[i * dim + i] = 1.0;

    const q_raw = try quantizeLinearRowsForTest(allocator, .q8_0, identity, dim, dim);
    defer allocator.free(q_raw);
    const k_raw = try quantizeLinearRowsForTest(allocator, .q8_0, identity, dim, dim);
    defer allocator.free(k_raw);
    const v_raw = try quantizeLinearRowsForTest(allocator, .q8_0, identity, dim, dim);
    defer allocator.free(v_raw);
    const out_raw = try quantizeLinearRowsForTest(allocator, .q8_0, identity, dim, dim);
    defer allocator.free(out_raw);
    const weight_shape = [_]i64{ @intCast(dim), @intCast(dim) };

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    try putTestQuantizedWeight(allocator, &weight_store, q_weight_name, q_raw, &weight_shape, .q8_0);
    try putTestQuantizedWeight(allocator, &weight_store, k_weight_name, k_raw, &weight_shape, .q8_0);
    try putTestQuantizedWeight(allocator, &weight_store, v_weight_name, v_raw, &weight_shape, .q8_0);
    try putTestQuantizedWeight(allocator, &weight_store, out_weight_name, out_raw, &weight_shape, .q8_0);
    metal_compute_mod.initPrefetchQueue(&weight_store, allocator);

    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    var input_data: [rows * dim]f32 = undefined;
    for (&input_data, 0..) |*value, idx| value.* = @as(f32, @floatFromInt(idx + 1)) / 64.0;
    const input_ct = try cb.fromFloat32Shape(&input_data, &.{ rows, dim });
    defer cb.free(input_ct);
    const input_dev = (try makeMetalDeviceResident(&cb, input_ct)) orelse return error.SkipZigTest;
    defer cb.free(input_dev);
    const q_weight_ct = try cb.getWeight(q_weight_name);
    defer cb.free(q_weight_ct);
    const k_weight_ct = try cb.getWeight(k_weight_name);
    defer cb.free(k_weight_ct);
    const v_weight_ct = try cb.getWeight(v_weight_name);
    defer cb.free(v_weight_ct);
    const out_weight_ct = try cb.getWeight(out_weight_name);
    defer cb.free(out_weight_ct);

    values[@intCast(x)] = input_dev;
    values[@intCast(q_weight)] = q_weight_ct;
    values[@intCast(k_weight)] = k_weight_ct;
    values[@intCast(v_weight)] = v_weight_ct;
    values[@intCast(out_weight)] = out_weight_ct;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    var exec_stats: PartitionExecutor.ExecutionStats = .{};
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    const executed_partitions = try allocator.alloc(bool, partition_plan.partitions.len);
    defer allocator.free(executed_partitions);
    @memset(executed_partitions, false);
    for (0..@intCast(g.nodeCount())) |node_index| {
        const node_id: NodeId = @intCast(node_index);
        const partition_index = partition_plan.node_assignment[node_id];
        if (partition_index >= partition_plan.partitions.len or executed_partitions[partition_index]) continue;
        executed_partitions[partition_index] = true;
        const part = partition_plan.partitions[partition_index];
        var has_reachable_compute_node = false;
        for (part.node_ids) |part_node_id| {
            const part_index: usize = @intCast(part_node_id);
            if (part_index >= reachable.len or !reachable[part_index]) continue;
            switch (g.node(part_node_id).op) {
                .parameter, .constant => {},
                else => has_reachable_compute_node = true,
            }
        }
        if (!has_reachable_compute_node) continue;
        try std.testing.expectEqual(contracts.BackendKind.metal, part.backend);
        try exec.partitionExecutor().execute(values, value_device, part.node_ids, 0, .{
            .allocator = allocator,
            .graph = &g,
            .backend = &cb,
            .options = .{
                .runtime_inputs = &.{
                    .{ .node_id = x, .value = values[@intCast(x)].? },
                    .{ .node_id = q_weight, .value = values[@intCast(q_weight)].? },
                    .{ .node_id = k_weight, .value = values[@intCast(k_weight)].? },
                    .{ .node_id = v_weight, .value = values[@intCast(v_weight)].? },
                    .{ .node_id = out_weight, .value = values[@intCast(out_weight)].? },
                },
            },
            .reachable = reachable,
            .last_use = last_use,
            .partition_plan = &partition_plan,
            .buffer_plan = &buffer_plan,
            .materialize_boundary_outputs = false,
            .stats = &exec_stats,
        });
    }

    const out_index: usize = @intCast(out);
    defer if (values[out_index]) |ct| cb.free(ct);
    try std.testing.expect(isMetalDeviceResident(&cb, values[out_index].?));
    try std.testing.expect(exec_stats.planned_operator_dispatches >= 2);
    try std.testing.expectEqual(@as(u64, 1), exec_stats.metal_qkv_regions);
    try std.testing.expectEqual(@as(u64, 3), exec_stats.graph_region_ops);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.interpreter_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.host_materialized_outputs);
    try std.testing.expectEqual(@as(u64, 3), exec_stats.gemma_qkv_hits);
    try std.testing.expectEqual(@as(u64, 1), exec_stats.gemma_o_proj_hits);
    try std.testing.expectEqual(@as(u64, 1), exec_stats.gemma_softmax_hits);
    try std.testing.expectEqual(@as(u64, 2), exec_stats.gemma_residual_add_hits);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.gemma_qkv_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.gemma_o_proj_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.gemma_softmax_fallbacks);
    try std.testing.expectEqual(@as(u64, 0), exec_stats.gemma_residual_add_fallbacks);

    const raw = try cb.toFloat32(values[out_index].?, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqual(rows * dim, raw.len);
    for (raw) |value| try std.testing.expect(std.math.isFinite(value));
}
test "metal partition executor command path runs q8 quantized linear" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    try expectPlannedQ8LinearOnMetal(9, 32, 2, .mul_mm);
}

test "metal partition executor planned q8 linear uses tiled mm shape" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    try expectPlannedQ8LinearOnMetal(9, 64, 64, .mul_mm);
}

test "metal partition executor planned q8 linear covers mv and small batch buckets" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    try expectPlannedQ8LinearOnMetal(1, 32, 8, .mul_mv);
    try expectPlannedQ8LinearOnMetal(4, 32, 8, .mul_mv_ext);
}

test "metal partition executor planned q4 and q5k linear stay packed on metal" {
    if (comptime !build_options.enable_metal) return error.SkipZigTest;
    if (!metal_runtime_mod.metalDeviceAvailable()) return error.SkipZigTest;

    try expectPlannedQuantLinearOnMetal(.q4_0, 4, 32, 8, .mul_mv_ext, 0.35);
    try expectPlannedQuantLinearOnMetal(.q4_1, 9, 32, 8, .mul_mm, 0.35);
    try expectPlannedQuantLinearOnMetal(.q5_k, 9, 256, 8, .mul_mm, 0.18);
}

fn expectPlannedQ8LinearOnMetal(rows: usize, in_dim: usize, out_dim: usize, expected_operator: operator_plan_mod.Operator) !void {
    try expectPlannedQuantLinearOnMetal(.q8_0, rows, in_dim, out_dim, expected_operator, 1e-3);
}

fn expectPlannedQuantLinearOnMetal(
    format: @import("quant_matmul.zig").Format,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    expected_operator: operator_plan_mod.Operator,
    tolerance: f32,
) !void {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();
    var b = ml.graph.Builder.init(&g);

    const x = try b.parameter("x", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(rows)), @as(i64, @intCast(in_dim)) }));
    const w = try b.parameter("w", ml.graph.Shape.init(.f32, &.{ @as(i64, @intCast(out_dim)), @as(i64, @intCast(in_dim)) }));
    const out = try b.linearNoBias(x, w, @intCast(rows), @intCast(in_dim), @intCast(out_dim));
    try g.markOutput(out);

    const seeds = try partition_mod.allocTensorDescriptorSeeds(allocator, &g);
    defer allocator.free(seeds);
    try partition_mod.seedAllParameterResidency(seeds, &g, .metal, 0);
    try std.testing.expect(try partition_mod.seedParameterQuantFormatByName(seeds, &g, "w", format));

    const caps = [_]partition_mod.Capability{
        .{ .backend = .metal, .priority = 20, .decide = &metal_capabilities.decideMetalEagerGraph },
        .{ .backend = .native, .priority = 0, .decide = &partition_mod.decideNative },
    };
    var partition_plan = try partition_mod.partitionWithOptions(allocator, &g, &caps, .{
        .tensor_descs = seeds,
    });
    defer partition_plan.deinit();
    const selected_plan = partition_plan.operatorPlanForNode(out) orelse return error.InvalidPartitionPlan;
    try std.testing.expectEqual(expected_operator, selected_plan.operator());
    var buffer_plan = try buffer_plan_mod.build(allocator, &g, &partition_plan, .{});
    defer buffer_plan.deinit();

    const weight_dense = try allocator.alloc(f32, out_dim * in_dim);
    defer allocator.free(weight_dense);
    for (0..out_dim) |out_col| {
        const scale: f32 = @floatFromInt((out_col % 5) + 1);
        for (0..in_dim) |col| {
            const signed = @as(i32, @intCast((out_col * 17 + col * 11) % 23)) - 11;
            weight_dense[out_col * in_dim + col] = scale * @as(f32, @floatFromInt(signed)) / 7.0;
        }
    }
    const weight_raw = try quantizeLinearRowsForTest(allocator, format, weight_dense, out_dim, in_dim);
    defer allocator.free(weight_raw);
    const weight_shape = [_]i64{ @intCast(out_dim), @intCast(in_dim) };

    var weight_store = initEmptyMetalWeightStore(allocator);
    defer deinitEmptyMetalWeightStore(&weight_store, allocator);
    try putTestQuantizedWeight(allocator, &weight_store, "w", weight_raw, &weight_shape, format);
    metal_compute_mod.initPrefetchQueue(&weight_store, allocator);
    var metal_compute = try metal_compute_mod.MetalCompute.init(allocator, &weight_store, null);
    defer metal_compute.deinit();
    var cb = metal_compute.computeBackend();
    if (!cb.decoderRuntimeReady()) return error.SkipZigTest;

    const count: usize = @intCast(g.nodeCount());
    const values = try allocator.alloc(?CT, count);
    defer allocator.free(values);
    @memset(values, null);
    const value_device = try allocator.alloc(DeviceId, count);
    defer allocator.free(value_device);
    @memset(value_device, 0);

    const x_data = try allocator.alloc(f32, rows * in_dim);
    defer allocator.free(x_data);
    for (0..rows) |row| {
        for (0..in_dim) |col| {
            x_data[row * in_dim + col] = @as(f32, @floatFromInt(row + 1)) * @as(f32, @floatFromInt((col % 7) + 1));
        }
    }

    const x_ct = try cb.fromFloat32Shape(x_data, &.{ @as(i32, @intCast(rows)), @as(i32, @intCast(in_dim)) });
    defer cb.free(x_ct);
    const w_ct = try cb.getWeight("w");
    defer cb.free(w_ct);
    values[@intCast(x)] = x_ct;
    values[@intCast(w)] = w_ct;

    const reachable = try interpreter.computeReachable(allocator, &g);
    defer allocator.free(reachable);
    const last_use = try interpreter.computeLastUse(allocator, &g, reachable);
    defer allocator.free(last_use);

    var planned_exec_stats: PartitionExecutor.ExecutionStats = .{};
    const partition_index = partition_plan.node_assignment[out];
    var exec = MetalPartitionExecutor.initBorrowed(allocator, &g, &cb);
    try exec.partitionExecutor().execute(values, value_device, partition_plan.partitions[partition_index].node_ids, 0, .{
        .allocator = allocator,
        .graph = &g,
        .backend = &cb,
        .options = .{
            .runtime_inputs = &.{
                .{ .node_id = x, .value = x_ct },
                .{ .node_id = w, .value = w_ct },
            },
        },
        .reachable = reachable,
        .last_use = last_use,
        .partition_plan = &partition_plan,
        .buffer_plan = &buffer_plan,
        .stats = &planned_exec_stats,
    });
    try std.testing.expectEqual(@as(u64, 1), planned_exec_stats.planned_operator_dispatches);

    const out_index: usize = @intCast(out);
    defer if (values[out_index]) |ct| cb.free(ct);
    const raw = try cb.toFloat32(values[out_index].?, allocator);
    defer allocator.free(raw);
    try std.testing.expectEqual(rows * out_dim, raw.len);
    const dequantized_weight = try allocator.alloc(f32, out_dim * in_dim);
    defer allocator.free(dequantized_weight);
    try quant_codec.dequantizeToFloat32(try quantFormatTensorType(format), weight_raw, dequantized_weight);
    for (0..rows) |row| {
        for (0..out_dim) |out_col| {
            var expected: f32 = 0;
            for (0..in_dim) |col| {
                expected += x_data[row * in_dim + col] * dequantized_weight[out_col * in_dim + col];
            }
            try std.testing.expectApproxEqAbs(expected, raw[row * out_dim + out_col], tolerance);
        }
    }
}

test "metal partition executor owned lifecycle deinitializes cleanly" {
    const allocator = std.testing.allocator;
    var g = Graph.init(allocator);
    defer g.deinit();

    var weight_store = native_compute.WeightStore{ .allocator = allocator, .resident_weights = .{}, .lazy_weights = .{} };
    defer deinitEmptyNativeWeightStore(&weight_store, allocator);
    var compute = native_compute.NativeCompute.init(allocator, &weight_store, null);
    var cb = compute.computeBackend();

    const exec = try MetalPartitionExecutor.create(allocator, &g, &cb);
    const pe = exec.partitionExecutor();
    pe.deinitExecutor();
}
