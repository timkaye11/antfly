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

// Training step orchestration: forward → loss → gradient → execute → extract.
//
// Given a computation graph with a scalar loss output, computes the loss
// value and parameter gradients in a single call by:
//   1. Running autodiff to produce a gradient graph
//   2. Marking gradient nodes as additional outputs
//   3. Executing the combined graph through the interpreter
//   4. Extracting loss and gradient tensors as f32 slices
//
// Usage:
//   var result = try trainStep(allocator, &graph, loss_id, &cb, runtime_inputs, .{});
//   defer result.deinit();
//   // result.loss = scalar loss value
//   // result.gradients.get("weight") = []f32 gradient data

const std = @import("std");
const builtin = @import("builtin");
const ml = @import("ml");
const platform = @import("antfly_platform");
const Graph = ml.graph.Graph;
const NodeId = ml.graph.NodeId;
const null_node = ml.graph.null_node;
const Builder = ml.graph.Builder;
const Shape = ml.graph.Shape;
const autodiff = ml.graph.autodiff;
const checkpoint = ml.graph.checkpoint;
const ops_mod = @import("../ops/ops.zig");
const metal_compute_mod = @import("../ops/metal_compute.zig");
const native_compute = @import("../ops/native_compute.zig");
const contracts = @import("backend_contracts.zig");
const CT = contracts.CT;
const ComputeBackend = ops_mod.ComputeBackend;
const interpreter = @import("interpreter.zig");
const RuntimeInput = interpreter.RuntimeInput;
const partition_mod = @import("partition.zig");
const buffer_plan_mod = @import("buffer_plan.zig");
const metal_capabilities = @import("metal_capabilities.zig");
const metal_partition_executor = @import("metal_partition_executor.zig");
const device_mesh = @import("device_mesh.zig");
const multi_executor = @import("multi_executor.zig");

pub const TrainStepResult = struct {
    loss: f32,
    gradients: std.StringHashMapUnmanaged([]f32), // param_name -> gradient f32 data
    device_gradients: std.StringHashMapUnmanaged(CT) = .{},
    profile: TrainStepProfile = .{},
    checkpoint_summary: ?CheckpointSummary = null,
    allocator: std.mem.Allocator,
    compute_backend: ?*const ComputeBackend = null,

    pub fn deinit(self: *TrainStepResult) void {
        var it = self.gradients.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.gradients.deinit(self.allocator);
        if (self.compute_backend) |cb| {
            var device_it = self.device_gradients.iterator();
            while (device_it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                cb.free(entry.value_ptr.*);
            }
        }
        self.device_gradients.deinit(self.allocator);
    }
};

pub const TrainStepProfile = struct {
    autodiff_ns: u64 = 0,
    checkpoint_ns: u64 = 0,
    execute_ns: u64 = 0,
    extract_ns: u64 = 0,
    total_ns: u64 = 0,
    peak_resident_bytes: usize = 0,
    graph_executor_partitions: u64 = 0,
    graph_executor_command_dispatches: u64 = 0,
    graph_executor_planned_dispatches: u64 = 0,
    graph_executor_interpreter_fallbacks: u64 = 0,
    graph_executor_host_outputs: u64 = 0,
    graph_executor_device_outputs: u64 = 0,
    graph_executor_regions: u64 = 0,
    graph_executor_region_ops: u64 = 0,
    graph_executor_runtime_region_dispatches: u64 = 0,
    graph_executor_runtime_region_fallbacks: u64 = 0,
    graph_executor_runtime_region_active_regions: u64 = 0,
    graph_executor_runtime_region_covered_nodes: u64 = 0,
    graph_executor_runtime_region_elided_nodes: u64 = 0,
    graph_executor_runtime_region_plan_compiles: u64 = 0,
    graph_executor_runtime_region_plan_reuses: u64 = 0,
    graph_executor_runtime_frame_candidates: u64 = 0,
    graph_executor_runtime_frame_eligible: u64 = 0,
    graph_executor_plan_build_ns: u64 = 0,
    graph_executor_buffer_plan_build_ns: u64 = 0,
    graph_executor_plan_cache_hits: u64 = 0,
    graph_executor_plan_cache_misses: u64 = 0,
    metal_frame_wait_ns: u64 = 0,
    metal_frame_gpu_ns: u64 = 0,
    metal_last_frame_compute_encoders: u64 = 0,
    metal_last_frame_blit_encoders: u64 = 0,
    metal_last_frame_planned_scopes: u64 = 0,
    metal_last_frame_planned_barriers: u64 = 0,
    metal_last_frame_planned_command_ops: u64 = 0,
    metal_deberta_encoder_plan_attempts: u64 = 0,
    metal_deberta_encoder_plan_successes: u64 = 0,
    metal_deberta_encoder_plan_reuses: u64 = 0,
    metal_deberta_encoder_plan_failures: u64 = 0,
    metal_deberta_encoder_layer_attempts: u64 = 0,
    metal_deberta_encoder_layer_successes: u64 = 0,
    metal_deberta_encoder_layer_fallbacks: u64 = 0,
    metal_deberta_relative_qk_pair_calls: u64 = 0,
    metal_deberta_relative_qk_pair_fallbacks: u64 = 0,
    metal_deberta_ffn_fused_calls: u64 = 0,
    metal_deberta_ffn_fused_mps_matmuls: u64 = 0,
    metal_deberta_ffn_fused_fallbacks: u64 = 0,
    metal_deberta_attention_flash_calls: u64 = 0,
    metal_deberta_attention_gemm_calls: u64 = 0,
    metal_deberta_attention_gemm_fallbacks: u64 = 0,
    metal_deberta_attention_legacy_calls: u64 = 0,
};

pub const CheckpointSummary = struct {
    strategy: checkpoint.CheckpointStrategy,
    layer_interval: u32,
    total_forward_activations: u32,
    checkpointed_activations: u32,
    recomputable_activations: u32,
    savings_ratio: f32,
};

pub const TrainStepOptions = struct {
    /// Which parameters to compute gradients for. If null, all parameters.
    trainable_params: ?[]const []const u8 = null,
    checkpoint_config: ?checkpoint.CheckpointConfig = null,
    emit_checkpoint_analysis: bool = false,
};

fn beginOwnedExecutionFrame(cb: *const ComputeBackend) !bool {
    if (cb.decoderRuntimeHasActiveFrame()) return false;
    return try cb.decoderRuntimeBeginFrame();
}

fn deltaU64(before: u64, after: u64) u64 {
    return if (after >= before) after - before else 0;
}

fn deltaU128ToU64(before: u128, after: u128) u64 {
    const delta = if (after >= before) after - before else 0;
    return if (delta > std.math.maxInt(u64)) std.math.maxInt(u64) else @intCast(delta);
}

fn recordMetalDebugProfile(
    profile: *TrainStepProfile,
    before: ops_mod.NativeQuantTimingStats,
    after: ops_mod.NativeQuantTimingStats,
) void {
    profile.metal_frame_wait_ns = deltaU128ToU64(before.decoder_runtime_frame_wait_nanos, after.decoder_runtime_frame_wait_nanos);
    profile.metal_frame_gpu_ns = deltaU128ToU64(before.decoder_runtime_frame_gpu_nanos, after.decoder_runtime_frame_gpu_nanos);
    profile.metal_last_frame_compute_encoders = after.metal_runtime_last_frame_compute_encoder_count;
    profile.metal_last_frame_blit_encoders = after.metal_runtime_last_frame_blit_encoder_count;
    profile.metal_last_frame_planned_scopes = after.metal_runtime_last_frame_planned_compute_scope_count;
    profile.metal_last_frame_planned_barriers = after.metal_runtime_last_frame_planned_barrier_count;
    profile.metal_last_frame_planned_command_ops = after.metal_runtime_last_frame_planned_command_op_count;
    profile.metal_deberta_encoder_plan_attempts = deltaU64(before.metal_runtime_deberta_encoder_frame_plan_attempts, after.metal_runtime_deberta_encoder_frame_plan_attempts);
    profile.metal_deberta_encoder_plan_successes = deltaU64(before.metal_runtime_deberta_encoder_frame_plan_successes, after.metal_runtime_deberta_encoder_frame_plan_successes);
    profile.metal_deberta_encoder_plan_reuses = deltaU64(before.metal_runtime_deberta_encoder_frame_plan_reuses, after.metal_runtime_deberta_encoder_frame_plan_reuses);
    profile.metal_deberta_encoder_plan_failures = deltaU64(before.metal_runtime_deberta_encoder_frame_plan_failures, after.metal_runtime_deberta_encoder_frame_plan_failures);
    profile.metal_deberta_encoder_layer_attempts = deltaU64(before.metal_runtime_deberta_encoder_layer_attempts, after.metal_runtime_deberta_encoder_layer_attempts);
    profile.metal_deberta_encoder_layer_successes = deltaU64(before.metal_runtime_deberta_encoder_layer_successes, after.metal_runtime_deberta_encoder_layer_successes);
    profile.metal_deberta_encoder_layer_fallbacks = deltaU64(before.metal_runtime_deberta_encoder_layer_fallbacks, after.metal_runtime_deberta_encoder_layer_fallbacks);
    profile.metal_deberta_relative_qk_pair_calls = deltaU64(before.metal_runtime_deberta_relative_qk_pair_calls, after.metal_runtime_deberta_relative_qk_pair_calls);
    profile.metal_deberta_relative_qk_pair_fallbacks = deltaU64(before.metal_runtime_deberta_relative_qk_pair_fallbacks, after.metal_runtime_deberta_relative_qk_pair_fallbacks);
    profile.metal_deberta_ffn_fused_calls = deltaU64(before.metal_runtime_deberta_ffn_fused_calls, after.metal_runtime_deberta_ffn_fused_calls);
    profile.metal_deberta_ffn_fused_mps_matmuls = deltaU64(before.metal_runtime_deberta_ffn_fused_mps_matmuls, after.metal_runtime_deberta_ffn_fused_mps_matmuls);
    profile.metal_deberta_ffn_fused_fallbacks = deltaU64(before.metal_runtime_deberta_ffn_fused_fallbacks, after.metal_runtime_deberta_ffn_fused_fallbacks);
    profile.metal_deberta_attention_flash_calls = deltaU64(before.metal_runtime_deberta_attention_flash_calls, after.metal_runtime_deberta_attention_flash_calls);
    profile.metal_deberta_attention_gemm_calls = deltaU64(before.metal_runtime_deberta_attention_gemm_calls, after.metal_runtime_deberta_attention_gemm_calls);
    profile.metal_deberta_attention_gemm_fallbacks = deltaU64(before.metal_runtime_deberta_attention_gemm_fallbacks, after.metal_runtime_deberta_attention_gemm_fallbacks);
    profile.metal_deberta_attention_legacy_calls = deltaU64(before.metal_runtime_deberta_attention_legacy_calls, after.metal_runtime_deberta_attention_legacy_calls);
}

fn cancelOwnedExecutionFrame(cb: *const ComputeBackend, active: *bool) void {
    if (!active.*) return;
    cb.decoderRuntimeCancelFrame() catch {};
    active.* = false;
}

fn submitOwnedExecutionFrame(cb: *const ComputeBackend, active: *bool) !void {
    if (!active.*) return;
    try cb.decoderRuntimeSubmitAndWaitFrame();
    active.* = false;
}

fn beginTrainingPlannedScope(cb: *const ComputeBackend) !metal_compute_mod.MetalCompute.PlannedGraphScope {
    return metal_compute_mod.MetalCompute.beginPlannedGraphScope(cb, .ffn);
}

fn endTrainingPlannedScope(cb: *const ComputeBackend, scope: *metal_compute_mod.MetalCompute.PlannedGraphScope) !void {
    if (!scope.active and !scope.owns_frame) return;
    try metal_compute_mod.MetalCompute.endPlannedGraphScope(cb, scope.*);
    scope.* = .{};
}

pub const CompiledTrainSession = struct {
    allocator: std.mem.Allocator,
    graph: Graph,
    id_map: []NodeId,
    wrt_names: [][]const u8,
    param_grads: []NodeId,
    analysis: interpreter.CachedAnalysis,
    loss_output_index: usize = 0,
    build_profile: TrainStepProfile = .{},
    cached_metal_graph_executor_plan: ?CachedMetalGraphExecutorPlan = null,

    pub fn init(
        allocator: std.mem.Allocator,
        graph: *const Graph,
        loss_node: NodeId,
        options: TrainStepOptions,
    ) !CompiledTrainSession {
        var profile: TrainStepProfile = .{};
        profile.peak_resident_bytes = currentResidentBytes();
        compiledDiag(
            "init begin nodes={} params={} outputs={} requested_wrt={} rss={}",
            .{
                graph.nodeCount(),
                graph.parameters.items.len,
                graph.outputs.items.len,
                if (options.trainable_params) |params| params.len else @as(usize, 0),
                profile.peak_resident_bytes,
            },
        );

        const wrt = try resolveWrtParams(allocator, graph, options.trainable_params);
        defer allocator.free(wrt);
        compiledDiag("resolved wrt={} rss={}", .{ wrt.len, currentResidentBytes() });

        const autodiff_start = nowNs();
        compiledDiag("autodiff begin nodes={} loss_node={}", .{ graph.nodeCount(), loss_node });
        var grad_result = try autodiff.gradient(allocator, graph, loss_node, wrt);
        profile.autodiff_ns = elapsedNs(autodiff_start);
        profile.peak_resident_bytes = @max(profile.peak_resident_bytes, currentResidentBytes());
        errdefer grad_result.deinit();
        compiledDiag(
            "autodiff done grad_nodes={} grad_params={} autodiff_ms={d:.3} rss={}",
            .{
                grad_result.graph.nodeCount(),
                grad_result.param_grads.len,
                nsToMs(profile.autodiff_ns),
                profile.peak_resident_bytes,
            },
        );

        if (options.checkpoint_config) |cfg| {
            const checkpoint_start = nowNs();
            compiledDiag("checkpoint begin strategy={s} rss={}", .{ @tagName(cfg.strategy), currentResidentBytes() });
            try checkpoint.applyCheckpointing(allocator, &grad_result, cfg);
            profile.checkpoint_ns = elapsedNs(checkpoint_start);
            profile.peak_resident_bytes = @max(profile.peak_resident_bytes, currentResidentBytes());
            compiledDiag("checkpoint done checkpoint_ms={d:.3} rss={}", .{ nsToMs(profile.checkpoint_ns), profile.peak_resident_bytes });
        }

        const lowered_loss = grad_result.id_map[loss_node];
        compiledDiag("mark outputs begin lowered_loss={} existing_outputs={} rss={}", .{ lowered_loss, grad_result.graph.outputs.items.len, currentResidentBytes() });
        grad_result.graph.outputs.clearRetainingCapacity();
        try grad_result.graph.markOutput(lowered_loss);

        for (grad_result.param_grads) |grad_id| {
            if (grad_id != null_node) try grad_result.graph.markOutput(grad_id);
        }
        compiledDiag("mark outputs done outputs={} rss={}", .{ grad_result.graph.outputs.items.len, currentResidentBytes() });

        const id_map = try allocator.dupe(NodeId, grad_result.id_map);
        errdefer allocator.free(id_map);
        const param_grads = try allocator.dupe(NodeId, grad_result.param_grads);
        errdefer allocator.free(param_grads);

        const wrt_names = try allocator.alloc([]const u8, wrt.len);
        errdefer allocator.free(wrt_names);
        for (wrt, 0..) |param_id, i| {
            const param_node = graph.node(param_id);
            wrt_names[i] = try allocator.dupe(u8, graph.parameterName(param_node));
        }
        errdefer {
            for (wrt_names) |name| allocator.free(name);
        }

        const compiled_graph = grad_result.graph;
        grad_result.graph = Graph.init(allocator);
        grad_result.deinit();
        const analysis = try interpreter.CachedAnalysis.compute(allocator, &compiled_graph);
        errdefer {
            var mutable_analysis = analysis;
            mutable_analysis.deinit(allocator);
        }
        profile.total_ns = profile.autodiff_ns + profile.checkpoint_ns;
        compiledDiag(
            "init done compiled_nodes={} outputs={} total_ms={d:.3} peak_rss={}",
            .{
                compiled_graph.nodeCount(),
                compiled_graph.outputs.items.len,
                nsToMs(profile.total_ns),
                profile.peak_resident_bytes,
            },
        );

        return .{
            .allocator = allocator,
            .graph = compiled_graph,
            .id_map = id_map,
            .wrt_names = wrt_names,
            .param_grads = param_grads,
            .analysis = analysis,
            .build_profile = profile,
        };
    }

    pub fn deinit(self: *CompiledTrainSession) void {
        if (self.cached_metal_graph_executor_plan) |*cached| cached.deinit();
        self.analysis.deinit(self.allocator);
        self.graph.deinit();
        self.allocator.free(self.id_map);
        for (self.wrt_names) |name| self.allocator.free(name);
        self.allocator.free(self.wrt_names);
        self.allocator.free(self.param_grads);
        self.* = undefined;
    }

    const CachedMetalGraphExecutorPlan = struct {
        base_plan: partition_mod.PartitionPlan,
        buffer_plan: buffer_plan_mod.BufferPlan,
        assignments: []device_mesh.DeviceId,
        partitioned: bool,
        unsupported_ops: usize = 0,

        fn deinit(self: *CachedMetalGraphExecutorPlan) void {
            self.buffer_plan.deinit();
            self.base_plan.deinit();
            self.base_plan.allocator.free(self.assignments);
            self.* = undefined;
        }
    };

    pub fn execute(
        self: *CompiledTrainSession,
        cb: *const ComputeBackend,
        runtime_inputs: ?std.AutoHashMapUnmanaged(NodeId, CT),
    ) !TrainStepResult {
        return self.executeInternal(cb, runtime_inputs, false);
    }

    pub fn executeDeviceGradients(
        self: *CompiledTrainSession,
        cb: *const ComputeBackend,
        runtime_inputs: ?std.AutoHashMapUnmanaged(NodeId, CT),
    ) !TrainStepResult {
        return self.executeInternal(cb, runtime_inputs, true);
    }

    fn executeInternal(
        self: *CompiledTrainSession,
        cb: *const ComputeBackend,
        runtime_inputs: ?std.AutoHashMapUnmanaged(NodeId, CT),
        retain_device_gradients: bool,
    ) !TrainStepResult {
        const total_start = nowNs();
        var profile: TrainStepProfile = .{};
        profile.peak_resident_bytes = currentResidentBytes();

        var rt_list = std.ArrayListUnmanaged(RuntimeInput).empty;
        defer rt_list.deinit(self.allocator);
        if (runtime_inputs) |rt| {
            var it = rt.iterator();
            while (it.next()) |entry| {
                const old_id = entry.key_ptr.*;
                if (old_id >= self.id_map.len) continue;
                const new_id = self.id_map[old_id];
                if (new_id == null_node) continue;
                try rt_list.append(self.allocator, .{ .node_id = new_id, .value = entry.value_ptr.* });
            }
        }

        const rt_slice: ?[]const RuntimeInput = if (rt_list.items.len > 0) rt_list.items else null;

        const execute_start = nowNs();
        compiledDiag(
            "execute begin nodes={} outputs={} runtime_inputs={} retain_device_gradients={} rss={}",
            .{
                self.graph.nodeCount(),
                self.graph.outputs.items.len,
                rt_list.items.len,
                retain_device_gradients,
                currentResidentBytes(),
            },
        );
        if (trainingGraphExecutorParityNodeIds()) |node_ids_raw| {
            if (!try self.runSelectedNodeParityDiagnostic(cb, rt_slice, node_ids_raw)) {
                return error.TrainingGraphExecutorParityMismatch;
            }
        }

        if (trainingGraphExecutorParityCheckEnabled() and cb.kind() == .metal and platform.env.getenvBoolDefault("TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR", false) and !platform.env.getenvBoolDefault("TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR", false)) {
            var direct_result = try self.executeWithDirectInterpreter(cb, rt_slice, retain_device_gradients, nowNs());
            errdefer direct_result.deinit();
            const graph_result_opt = try self.executeWithSingleMetalGraphExecutor(cb, rt_slice, retain_device_gradients, total_start, execute_start, &profile);
            var graph_result = graph_result_opt orelse return error.TrainingGraphExecutorUnavailable;
            errdefer graph_result.deinit();
            const report = try self.compareTrainStepResults(cb, &direct_result, &graph_result);
            printTrainingGraphExecutorParityReport(report);
            if (!report.passed) return error.TrainingGraphExecutorParityMismatch;
            direct_result.deinit();
            return graph_result;
        }

        if (try self.executeWithSingleMetalGraphExecutor(cb, rt_slice, retain_device_gradients, total_start, execute_start, &profile)) |result| {
            return result;
        }

        return try self.executeWithDirectInterpreter(cb, rt_slice, retain_device_gradients, total_start);
    }

    fn runSelectedNodeParityDiagnostic(
        self: *CompiledTrainSession,
        cb: *const ComputeBackend,
        runtime_inputs: ?[]const RuntimeInput,
        node_ids_raw: []const u8,
    ) !bool {
        if (cb.kind() != .metal) return error.TrainingGraphExecutorParityDiagnosticRequiresMetal;
        if (!platform.env.getenvBoolDefault("TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR", false)) return error.TrainingGraphExecutorUnavailable;

        const selected = try parseParityNodeIds(self.allocator, node_ids_raw, self.graph.nodeCount());
        defer self.allocator.free(selected);
        if (selected.len == 0) return error.TrainingGraphExecutorParityNoNodes;

        const saved_outputs = try self.allocator.dupe(NodeId, self.graph.outputs.items);
        defer self.allocator.free(saved_outputs);
        self.graph.outputs.clearRetainingCapacity();
        for (selected) |node_id| try self.graph.markOutput(node_id);
        defer {
            self.graph.outputs.clearRetainingCapacity();
            self.graph.outputs.appendSlice(self.allocator, saved_outputs) catch {};
        }

        var owned_execution_frame = try beginOwnedExecutionFrame(cb);
        errdefer cancelOwnedExecutionFrame(cb, &owned_execution_frame);
        var planned_scope = try beginTrainingPlannedScope(cb);
        errdefer endTrainingPlannedScope(cb, &planned_scope) catch {};
        var direct_result = try interpreter.execute(
            self.allocator,
            &self.graph,
            cb,
            .{ .runtime_inputs = runtime_inputs },
        );
        try endTrainingPlannedScope(cb, &planned_scope);
        try submitOwnedExecutionFrame(cb, &owned_execution_frame);
        defer direct_result.deinit(cb);

        var diagnostics = partition_mod.CapabilityDiagnostics{};
        var base_plan = if (platform.env.getenvBoolDefault("TERMITE_TRAINING_GRAPH_EXECUTOR_PARTITIONED", false))
            try self.buildPartitionedMetalGraphExecutorPlan(&diagnostics)
        else
            try self.buildSingleMetalGraphExecutorPlan();
        defer base_plan.deinit();

        const assignments = try self.allocator.alloc(device_mesh.DeviceId, base_plan.partitions.len);
        defer self.allocator.free(assignments);
        for (base_plan.partitions, 0..) |part, idx| {
            assignments[idx] = if (part.backend == .native) 1 else 0;
        }
        var dpp = multi_executor.DevicePartitionPlan{
            .base = base_plan,
            .device_assignment = assignments,
            .allocator = self.allocator,
        };

        var fallback_native = try FallbackNativeBackend.init(self.allocator);
        defer fallback_native.deinit(self.allocator);
        var devices_buf = [_]device_mesh.DeviceEntry{
            .{ .id = 0, .backend = cb, .kind = .metal },
            .{ .id = 1, .backend = &fallback_native.backend, .kind = .native },
        };
        var mesh = try device_mesh.DeviceMesh.init(self.allocator, devices_buf[0..]);
        defer mesh.deinit();

        var graph_result = try multi_executor.executeMultiDevice(
            self.allocator,
            &self.graph,
            &dpp,
            &mesh,
            .{
                .runtime_inputs = runtime_inputs,
                .skip_metal_fused_patterns = trainingGraphExecutorSkipMetalFusedPatterns(),
                .collect_partition_stats = false,
                .preserve_runtime_input_residency = true,
            },
        );
        defer graph_result.deinit(&mesh);

        var passed = true;
        for (selected, 0..) |node_id, idx| {
            const direct_data = try cb.toFloat32(direct_result.outputs[idx], self.allocator);
            defer self.allocator.free(direct_data);
            const graph_device = mesh.device(graph_result.output_devices[idx]) orelse return error.DeviceNotFound;
            const graph_data = try graph_device.backend.toFloat32(graph_result.outputs[idx], self.allocator);
            defer self.allocator.free(graph_data);
            const diff = compareFloatSlices(direct_data, graph_data);
            if (diff.len_mismatch or (diff.max_abs > trainingGraphExecutorParityGradAbsTolerance() and diff.max_rel > trainingGraphExecutorParityGradRelTolerance())) {
                passed = false;
            }
            const node_passed = !diff.len_mismatch and (diff.max_abs <= trainingGraphExecutorParityGradAbsTolerance() or diff.max_rel <= trainingGraphExecutorParityGradRelTolerance());
            std.debug.print(
                "training_graph_executor_node_parity: node={} op={s} passed={} direct_len={} graph_len={} max_abs_diff={d:.9} max_rel_diff={d:.9} first_diff_index={}\n",
                .{
                    node_id,
                    @tagName(std.meta.activeTag(self.graph.node(node_id).op)),
                    node_passed,
                    direct_data.len,
                    graph_data.len,
                    diff.max_abs,
                    diff.max_rel,
                    diff.first_diff_index,
                },
            );
            if (platform.env.getenvBoolDefault("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_PRINT_VALUES", false) and graph_data.len > 0 and direct_data.len > 0) {
                const start = if (diff.first_diff_index != std.math.maxInt(usize)) diff.first_diff_index else 0;
                const limit = @min(@min(direct_data.len, graph_data.len), start + 8);
                var sample_idx = start;
                while (sample_idx < limit) : (sample_idx += 1) {
                    std.debug.print(
                        "training_graph_executor_node_parity_value: node={} index={} direct={d:.9} graph={d:.9} abs_diff={d:.9}\n",
                        .{
                            node_id,
                            sample_idx,
                            direct_data[sample_idx],
                            graph_data[sample_idx],
                            @abs(direct_data[sample_idx] - graph_data[sample_idx]),
                        },
                    );
                }
            }
            if (platform.env.getenv("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_SAMPLE_INDICES")) |sample_raw| {
                var parts = std.mem.splitScalar(u8, sample_raw, ',');
                while (parts.next()) |part_raw| {
                    const part = std.mem.trim(u8, part_raw, " \t\r\n");
                    if (part.len == 0) continue;
                    const sample_idx = std.fmt.parseInt(usize, part, 10) catch continue;
                    if (sample_idx >= direct_data.len or sample_idx >= graph_data.len) continue;
                    std.debug.print(
                        "training_graph_executor_node_parity_sample: node={} index={} direct={d:.9} graph={d:.9} abs_diff={d:.9}\n",
                        .{
                            node_id,
                            sample_idx,
                            direct_data[sample_idx],
                            graph_data[sample_idx],
                            @abs(direct_data[sample_idx] - graph_data[sample_idx]),
                        },
                    );
                }
            }
        }
        std.debug.print("training_graph_executor_node_parity_summary: passed={} nodes={}\n", .{ passed, selected.len });
        return passed;
    }

    fn executeWithDirectInterpreter(
        self: *CompiledTrainSession,
        cb: *const ComputeBackend,
        runtime_inputs: ?[]const RuntimeInput,
        retain_device_gradients: bool,
        total_start: u64,
    ) !TrainStepResult {
        var profile: TrainStepProfile = .{};
        profile.peak_resident_bytes = currentResidentBytes();
        const execute_start = nowNs();

        var owned_execution_frame = try beginOwnedExecutionFrame(cb);
        errdefer cancelOwnedExecutionFrame(cb, &owned_execution_frame);
        var planned_scope = try beginTrainingPlannedScope(cb);
        errdefer endTrainingPlannedScope(cb, &planned_scope) catch {};
        var exec_result = try interpreter.execute(
            self.allocator,
            &self.graph,
            cb,
            .{ .runtime_inputs = runtime_inputs, .cached_analysis = self.analysis },
        );
        try endTrainingPlannedScope(cb, &planned_scope);
        try submitOwnedExecutionFrame(cb, &owned_execution_frame);
        profile.execute_ns = elapsedNs(execute_start);
        profile.peak_resident_bytes = @max(profile.peak_resident_bytes, currentResidentBytes());
        compiledDiag("execute done execute_ms={d:.3} rss={}", .{ nsToMs(profile.execute_ns), profile.peak_resident_bytes });
        defer exec_result.deinit(cb);

        const extract_start = nowNs();
        compiledDiag("extract begin outputs={} retain_device_gradients={} rss={}", .{ exec_result.outputs.len, retain_device_gradients, currentResidentBytes() });
        const loss_data = try cb.toFloat32(exec_result.outputs[self.loss_output_index], self.allocator);
        defer self.allocator.free(loss_data);
        const loss_value: f32 = if (loss_data.len > 0) loss_data[0] else 0.0;

        var gradients = std.StringHashMapUnmanaged([]f32){};
        var device_gradients = std.StringHashMapUnmanaged(CT){};
        errdefer {
            var it = gradients.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                self.allocator.free(entry.value_ptr.*);
            }
            gradients.deinit(self.allocator);
            var device_it = device_gradients.iterator();
            while (device_it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                cb.free(entry.value_ptr.*);
            }
            device_gradients.deinit(self.allocator);
        }

        var grad_output_idx: usize = self.loss_output_index + 1;
        for (self.wrt_names, 0..) |name, i| {
            if (self.param_grads[i] == null_node) continue;
            const grad_ct = exec_result.outputs[grad_output_idx];
            const grad_output_node = self.graph.outputs.items[grad_output_idx];
            grad_output_idx += 1;
            const owned_name = try self.allocator.dupe(u8, name);
            errdefer self.allocator.free(owned_name);
            if (retain_device_gradients) {
                const retained = try cloneTensorForOutputShape(self.allocator, &self.graph, cb, grad_ct, grad_output_node);
                errdefer cb.free(retained);
                try device_gradients.put(self.allocator, owned_name, retained);
            } else {
                const grad_data = try cb.toFloat32(grad_ct, self.allocator);
                try gradients.put(self.allocator, owned_name, grad_data);
            }
        }

        profile.extract_ns = elapsedNs(extract_start);
        profile.total_ns = elapsedNs(total_start);
        profile.peak_resident_bytes = @max(profile.peak_resident_bytes, currentResidentBytes());
        compiledDiag(
            "extract done gradients={} device_gradients={} extract_ms={d:.3} total_ms={d:.3} rss={}",
            .{
                gradients.count(),
                device_gradients.count(),
                nsToMs(profile.extract_ns),
                nsToMs(profile.total_ns),
                profile.peak_resident_bytes,
            },
        );

        return .{
            .loss = loss_value,
            .gradients = gradients,
            .device_gradients = device_gradients,
            .profile = profile,
            .allocator = self.allocator,
            .compute_backend = if (retain_device_gradients) cb else null,
        };
    }

    fn executeWithSingleMetalGraphExecutor(
        self: *CompiledTrainSession,
        cb: *const ComputeBackend,
        runtime_inputs: ?[]const RuntimeInput,
        retain_device_gradients: bool,
        total_start: u64,
        execute_start: u64,
        profile: *TrainStepProfile,
    ) !?TrainStepResult {
        if (cb.kind() != .metal) return null;
        if (!platform.env.getenvBoolDefault("TERMITE_ENABLE_TRAINING_GRAPH_EXECUTOR", false)) return null;
        if (platform.env.getenvBoolDefault("TERMITE_DISABLE_TRAINING_GRAPH_EXECUTOR", false)) return null;

        const cached = try self.cachedMetalGraphExecutorPlan(cb, profile);
        if (retain_device_gradients and !self.planKeepsGradientOutputsOnMetal(&cached.base_plan)) {
            compiledDiag("graph executor fallback gradient output assigned to native partition unsupported_ops={}", .{cached.unsupported_ops});
            return null;
        }

        var dpp = multi_executor.DevicePartitionPlan{
            .base = cached.base_plan,
            .device_assignment = cached.assignments,
            .allocator = self.allocator,
        };

        var fallback_native = try FallbackNativeBackend.init(self.allocator);
        defer fallback_native.deinit(self.allocator);

        var devices_buf = [_]device_mesh.DeviceEntry{
            .{ .id = 0, .backend = cb, .kind = .metal },
            .{ .id = 1, .backend = &fallback_native.backend, .kind = .native },
        };
        var mesh = try device_mesh.DeviceMesh.init(self.allocator, devices_buf[0..]);
        defer mesh.deinit();

        const metal_debug_before = cb.debugTimingSnapshot().provider;
        var multi_result = multi_executor.executeMultiDevice(
            self.allocator,
            &self.graph,
            &dpp,
            &mesh,
            .{
                .runtime_inputs = runtime_inputs,
                .cached_analysis = self.analysis,
                .cached_buffer_plan = &cached.buffer_plan,
                .skip_metal_fused_patterns = trainingGraphExecutorSkipMetalFusedPatterns(),
                .collect_partition_stats = true,
                .preserve_runtime_input_residency = true,
            },
        ) catch |err| switch (err) {
            error.MissingRuntimeInput => {
                compiledDiag("graph executor fallback missing partition input; retrying regular compiled Metal path", .{});
                return null;
            },
            else => return err,
        };
        defer multi_result.deinit(&mesh);

        profile.execute_ns = elapsedNs(execute_start);
        recordMetalDebugProfile(profile, metal_debug_before, cb.debugTimingSnapshot().provider);
        profile.peak_resident_bytes = @max(profile.peak_resident_bytes, currentResidentBytes());
        profile.graph_executor_partitions = multi_result.stats.partitions_executed;
        profile.graph_executor_command_dispatches = multi_result.stats.backend_command_dispatches;
        profile.graph_executor_planned_dispatches = multi_result.stats.planned_operator_dispatches;
        profile.graph_executor_interpreter_fallbacks = multi_result.stats.interpreter_fallbacks;
        profile.graph_executor_host_outputs = multi_result.stats.host_materialized_outputs;
        profile.graph_executor_device_outputs = multi_result.stats.device_resident_outputs;
        profile.graph_executor_regions = multi_result.stats.graph_regions;
        profile.graph_executor_region_ops = multi_result.stats.graph_region_ops;
        profile.graph_executor_runtime_region_dispatches = multi_result.stats.runtime_region_plan_dispatches;
        profile.graph_executor_runtime_region_fallbacks = multi_result.stats.runtime_region_fallbacks;
        profile.graph_executor_runtime_region_active_regions = multi_result.stats.runtime_region_plan_active_regions;
        profile.graph_executor_runtime_region_covered_nodes = multi_result.stats.runtime_region_plan_covered_nodes;
        profile.graph_executor_runtime_region_elided_nodes = multi_result.stats.runtime_region_plan_elided_nodes;
        profile.graph_executor_runtime_region_plan_compiles = multi_result.stats.runtime_region_plan_compiles;
        profile.graph_executor_runtime_region_plan_reuses = multi_result.stats.runtime_region_plan_reuses;
        profile.graph_executor_runtime_frame_candidates = multi_result.stats.runtime_frame_candidates;
        profile.graph_executor_runtime_frame_eligible = multi_result.stats.runtime_frame_eligible;
        compiledDiag(
            "graph executor execute done partitions={} commands={} planned={} fallbacks={} plan_cache_hits={} plan_cache_misses={} plan_build_ms={d:.3} execute_ms={d:.3} rss={}",
            .{
                dpp.base.partitions.len,
                profile.graph_executor_command_dispatches,
                profile.graph_executor_planned_dispatches,
                profile.graph_executor_interpreter_fallbacks,
                profile.graph_executor_plan_cache_hits,
                profile.graph_executor_plan_cache_misses,
                nsToMs(profile.graph_executor_plan_build_ns),
                nsToMs(profile.execute_ns),
                profile.peak_resident_bytes,
            },
        );

        return try self.extractTrainStepResultFromGraphOutputs(
            cb,
            &mesh,
            &multi_result,
            retain_device_gradients,
            total_start,
            profile,
        );
    }

    fn cachedMetalGraphExecutorPlan(
        self: *CompiledTrainSession,
        cb: *const ComputeBackend,
        profile: *TrainStepProfile,
    ) !*const CachedMetalGraphExecutorPlan {
        const partitioned = platform.env.getenvBoolDefault("TERMITE_TRAINING_GRAPH_EXECUTOR_PARTITIONED", false);
        if (self.cached_metal_graph_executor_plan) |*cached| {
            if (cached.partitioned == partitioned) {
                profile.graph_executor_plan_cache_hits += 1;
                return cached;
            }
            cached.deinit();
            self.cached_metal_graph_executor_plan = null;
        }

        profile.graph_executor_plan_cache_misses += 1;
        const plan_start = nowNs();
        var diagnostics = partition_mod.CapabilityDiagnostics{};
        var base_plan = if (partitioned)
            try self.buildPartitionedMetalGraphExecutorPlan(&diagnostics)
        else
            try self.buildSingleMetalGraphExecutorPlan();
        errdefer base_plan.deinit();
        try self.attachCachedMetalExecutors(cb, &base_plan);

        const buffer_plan_start = nowNs();
        var graph_buffer_plan = try buffer_plan_mod.build(self.allocator, &self.graph, &base_plan, .{});
        errdefer graph_buffer_plan.deinit();
        try graph_buffer_plan.validate(&self.graph, &base_plan);
        profile.graph_executor_buffer_plan_build_ns += elapsedNs(buffer_plan_start);

        const assignments = try self.allocator.alloc(device_mesh.DeviceId, base_plan.partitions.len);
        errdefer self.allocator.free(assignments);
        for (base_plan.partitions, 0..) |part, idx| {
            assignments[idx] = if (part.backend == .native) 1 else 0;
        }

        self.cached_metal_graph_executor_plan = .{
            .base_plan = base_plan,
            .buffer_plan = graph_buffer_plan,
            .assignments = assignments,
            .partitioned = partitioned,
            .unsupported_ops = diagnostics.count(.unsupported_op),
        };
        profile.graph_executor_plan_build_ns += elapsedNs(plan_start);
        return &self.cached_metal_graph_executor_plan.?;
    }

    fn attachCachedMetalExecutors(
        self: *CompiledTrainSession,
        cb: *const ComputeBackend,
        plan: *partition_mod.PartitionPlan,
    ) !void {
        for (plan.partitions) |*part| {
            if (part.backend != .metal or part.executor != null) continue;
            const exec = try metal_partition_executor.MetalPartitionExecutor.create(self.allocator, &self.graph, cb);
            part.executor = exec.partitionExecutor();
        }
    }

    fn buildSingleMetalGraphExecutorPlan(self: *CompiledTrainSession) !partition_mod.PartitionPlan {
        const count: usize = @intCast(self.graph.nodeCount());
        const partitions = try self.allocator.alloc(partition_mod.Partition, 1);
        errdefer self.allocator.free(partitions);
        const node_ids = try self.allocator.alloc(NodeId, count);
        errdefer self.allocator.free(node_ids);
        const node_assignment = try self.allocator.alloc(u32, count);
        errdefer self.allocator.free(node_assignment);
        const node_operator_plans = try self.allocator.alloc(?@import("operator_plan.zig").OperatorPlan, count);
        errdefer self.allocator.free(node_operator_plans);
        const external_inputs = try self.allocator.alloc(partition_mod.ExternalInput, 0);
        errdefer self.allocator.free(external_inputs);

        for (0..count) |i| {
            node_ids[i] = @intCast(i);
            node_assignment[i] = 0;
            node_operator_plans[i] = null;
        }

        partitions[0] = .{
            .backend = .metal,
            .device_id = 0,
            .node_ids = node_ids,
            .external_inputs = external_inputs,
        };

        return .{
            .partitions = partitions,
            .node_assignment = node_assignment,
            .node_operator_plans = node_operator_plans,
            .allocator = self.allocator,
        };
    }

    fn buildPartitionedMetalGraphExecutorPlan(
        self: *CompiledTrainSession,
        diagnostics: *partition_mod.CapabilityDiagnostics,
    ) !partition_mod.PartitionPlan {
        const descriptor_seeds = try partition_mod.allocTensorDescriptorSeeds(self.allocator, &self.graph);
        defer self.allocator.free(descriptor_seeds);
        try partition_mod.seedAllUploadableResidency(descriptor_seeds, &self.graph, .metal, 0);

        const capabilities = [_]partition_mod.Capability{
            .{
                .backend = .metal,
                .priority = 10,
                .supports = &metal_capabilities.supportsMetalEagerGraph,
                .decide = &metal_capabilities.decideMetalEagerGraph,
            },
            .{
                .backend = .native,
                .priority = 0,
                .supports = &partition_mod.supportsAll,
                .decide = &partition_mod.decideNative,
            },
        };

        return partition_mod.partitionWithOptions(self.allocator, &self.graph, &capabilities, .{
            .tensor_descs = descriptor_seeds,
            .diagnostics = diagnostics,
        });
    }

    fn planKeepsGradientOutputsOnMetal(
        self: *const CompiledTrainSession,
        plan: *const partition_mod.PartitionPlan,
    ) bool {
        var grad_output_idx: usize = self.loss_output_index + 1;
        for (self.param_grads) |grad_id| {
            if (grad_id == null_node) continue;
            const output_node = self.graph.outputs.items[grad_output_idx];
            grad_output_idx += 1;
            const part_idx = plan.node_assignment[@intCast(output_node)];
            if (plan.partitions[@intCast(part_idx)].backend != .metal) return false;
        }
        return true;
    }

    fn extractTrainStepResultFromGraphOutputs(
        self: *CompiledTrainSession,
        cb: *const ComputeBackend,
        mesh: *const device_mesh.DeviceMesh,
        multi_result: *const multi_executor.MultiExecutionResult,
        retain_device_gradients: bool,
        total_start: u64,
        profile: *TrainStepProfile,
    ) !TrainStepResult {
        const extract_start = nowNs();
        const outputs = multi_result.outputs;
        compiledDiag("extract begin outputs={} retain_device_gradients={} rss={}", .{ outputs.len, retain_device_gradients, currentResidentBytes() });
        const loss_device = mesh.device(multi_result.output_devices[self.loss_output_index]) orelse return error.DeviceNotFound;
        const loss_data = try loss_device.backend.toFloat32(outputs[self.loss_output_index], self.allocator);
        defer self.allocator.free(loss_data);
        const loss_value: f32 = if (loss_data.len > 0) loss_data[0] else 0.0;

        var gradients = std.StringHashMapUnmanaged([]f32){};
        var device_gradients = std.StringHashMapUnmanaged(CT){};
        errdefer {
            var it = gradients.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                self.allocator.free(entry.value_ptr.*);
            }
            gradients.deinit(self.allocator);
            var device_it = device_gradients.iterator();
            while (device_it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                cb.free(entry.value_ptr.*);
            }
            device_gradients.deinit(self.allocator);
        }

        var grad_output_idx: usize = self.loss_output_index + 1;
        for (self.wrt_names, 0..) |name, i| {
            if (self.param_grads[i] == null_node) continue;
            const grad_ct = outputs[grad_output_idx];
            const grad_output_node = self.graph.outputs.items[grad_output_idx];
            const grad_device = mesh.device(multi_result.output_devices[grad_output_idx]) orelse return error.DeviceNotFound;
            grad_output_idx += 1;
            const owned_name = try self.allocator.dupe(u8, name);
            errdefer self.allocator.free(owned_name);
            if (retain_device_gradients) {
                if (grad_device.backend != cb) return error.UnexpectedGradientDevice;
                const retained = try cloneTensorForOutputShape(self.allocator, &self.graph, cb, grad_ct, grad_output_node);
                errdefer cb.free(retained);
                try device_gradients.put(self.allocator, owned_name, retained);
            } else {
                const grad_data = try grad_device.backend.toFloat32(grad_ct, self.allocator);
                try gradients.put(self.allocator, owned_name, grad_data);
            }
        }

        profile.extract_ns = elapsedNs(extract_start);
        profile.total_ns = elapsedNs(total_start);
        profile.peak_resident_bytes = @max(profile.peak_resident_bytes, currentResidentBytes());
        compiledDiag(
            "extract done gradients={} device_gradients={} extract_ms={d:.3} total_ms={d:.3} rss={}",
            .{
                gradients.count(),
                device_gradients.count(),
                nsToMs(profile.extract_ns),
                nsToMs(profile.total_ns),
                profile.peak_resident_bytes,
            },
        );

        return .{
            .loss = loss_value,
            .gradients = gradients,
            .device_gradients = device_gradients,
            .profile = profile.*,
            .allocator = self.allocator,
            .compute_backend = if (retain_device_gradients) cb else null,
        };
    }

    const ParityReport = struct {
        passed: bool = true,
        loss_abs_diff: f32 = 0,
        loss_rel_diff: f32 = 0,
        max_grad_abs_diff: f32 = 0,
        max_grad_rel_diff: f32 = 0,
        compared_gradients: usize = 0,
        compared_gradient_values: usize = 0,
        missing_gradients: usize = 0,
        direct_loss: f32 = 0,
        graph_loss: f32 = 0,
    };

    fn compareTrainStepResults(
        self: *CompiledTrainSession,
        cb: *const ComputeBackend,
        direct: *const TrainStepResult,
        graph_exec: *const TrainStepResult,
    ) !ParityReport {
        var report = ParityReport{
            .direct_loss = direct.loss,
            .graph_loss = graph_exec.loss,
            .loss_abs_diff = absF32(direct.loss - graph_exec.loss),
            .loss_rel_diff = relativeDiffF32(direct.loss, graph_exec.loss),
        };

        const loss_abs_tol = trainingGraphExecutorParityLossAbsTolerance();
        const loss_rel_tol = trainingGraphExecutorParityLossRelTolerance();
        const grad_abs_tol = trainingGraphExecutorParityGradAbsTolerance();
        const grad_rel_tol = trainingGraphExecutorParityGradRelTolerance();
        var remaining_values = trainingGraphExecutorParityMaxGradientValues();

        if (report.loss_abs_diff > loss_abs_tol and report.loss_rel_diff > loss_rel_tol) {
            report.passed = false;
        }

        if (direct.device_gradients.count() > 0 or graph_exec.device_gradients.count() > 0) {
            var it = direct.device_gradients.iterator();
            while (it.next()) |entry| {
                if (remaining_values == 0) break;
                const name = entry.key_ptr.*;
                const graph_ct = graph_exec.device_gradients.get(name) orelse {
                    report.missing_gradients += 1;
                    report.passed = false;
                    continue;
                };
                const direct_data = try cb.toFloat32(entry.value_ptr.*, self.allocator);
                defer self.allocator.free(direct_data);
                const graph_data = try cb.toFloat32(graph_ct, self.allocator);
                defer self.allocator.free(graph_data);
                self.compareGradientData(
                    direct_data,
                    graph_data,
                    grad_abs_tol,
                    grad_rel_tol,
                    &remaining_values,
                    &report,
                );
            }
            if (graph_exec.device_gradients.count() != direct.device_gradients.count()) {
                report.missing_gradients += graph_exec.device_gradients.count() -| direct.device_gradients.count();
                report.passed = false;
            }
            return report;
        }

        var it = direct.gradients.iterator();
        while (it.next()) |entry| {
            if (remaining_values == 0) break;
            const name = entry.key_ptr.*;
            const graph_data = graph_exec.gradients.get(name) orelse {
                report.missing_gradients += 1;
                report.passed = false;
                continue;
            };
            self.compareGradientData(
                entry.value_ptr.*,
                graph_data,
                grad_abs_tol,
                grad_rel_tol,
                &remaining_values,
                &report,
            );
        }
        if (graph_exec.gradients.count() != direct.gradients.count()) {
            report.missing_gradients += graph_exec.gradients.count() -| direct.gradients.count();
            report.passed = false;
        }
        return report;
    }

    fn compareGradientData(
        self: *CompiledTrainSession,
        direct: []const f32,
        graph_exec: []const f32,
        grad_abs_tol: f32,
        grad_rel_tol: f32,
        remaining_values: *usize,
        report: *ParityReport,
    ) void {
        _ = self;
        report.compared_gradients += 1;
        if (direct.len != graph_exec.len) {
            report.missing_gradients += 1;
            report.passed = false;
        }
        const n = @min(@min(direct.len, graph_exec.len), remaining_values.*);
        for (0..n) |i| {
            const abs_diff = absF32(direct[i] - graph_exec[i]);
            const rel_diff = relativeDiffF32(direct[i], graph_exec[i]);
            report.max_grad_abs_diff = @max(report.max_grad_abs_diff, abs_diff);
            report.max_grad_rel_diff = @max(report.max_grad_rel_diff, rel_diff);
            if (abs_diff > grad_abs_tol and rel_diff > grad_rel_tol) {
                report.passed = false;
            }
        }
        report.compared_gradient_values += n;
        remaining_values.* -= n;
    }
};

fn trainingGraphExecutorParityCheckEnabled() bool {
    return platform.env.getenvBoolDefault("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_CHECK", false);
}

fn trainingGraphExecutorSkipMetalFusedPatterns() bool {
    return platform.env.getenvBoolDefault("TERMITE_TRAINING_GRAPH_EXECUTOR_SKIP_METAL_FUSED_PATTERNS", false);
}

fn trainingGraphExecutorParityNodeIds() ?[]const u8 {
    return platform.env.getenv("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_NODE_IDS");
}

fn trainingGraphExecutorParityMaxGradientValues() usize {
    return platform.env.getenvUsize("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_MAX_GRAD_VALUES") orelse 4096;
}

fn trainingGraphExecutorParityLossAbsTolerance() f32 {
    return getenvF32("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_LOSS_ATOL", 1e-5);
}

fn trainingGraphExecutorParityLossRelTolerance() f32 {
    return getenvF32("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_LOSS_RTOL", 1e-4);
}

fn trainingGraphExecutorParityGradAbsTolerance() f32 {
    return getenvF32("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_GRAD_ATOL", 1e-4);
}

fn trainingGraphExecutorParityGradRelTolerance() f32 {
    return getenvF32("TERMITE_TRAINING_GRAPH_EXECUTOR_PARITY_GRAD_RTOL", 1e-3);
}

fn getenvF32(comptime name: [*:0]const u8, default: f32) f32 {
    const value = platform.env.getenv(name) orelse return default;
    return std.fmt.parseFloat(f32, value) catch default;
}

fn absF32(value: f32) f32 {
    return if (value < 0) -value else value;
}

fn relativeDiffF32(lhs: f32, rhs: f32) f32 {
    const denom = @max(@max(absF32(lhs), absF32(rhs)), 1e-12);
    return absF32(lhs - rhs) / denom;
}

const FloatSliceDiff = struct {
    max_abs: f32 = 0,
    max_rel: f32 = 0,
    first_diff_index: usize = std.math.maxInt(usize),
    len_mismatch: bool = false,
};

fn compareFloatSlices(lhs: []const f32, rhs: []const f32) FloatSliceDiff {
    var diff = FloatSliceDiff{ .len_mismatch = lhs.len != rhs.len };
    const n = @min(lhs.len, rhs.len);
    for (0..n) |i| {
        const abs_diff = absF32(lhs[i] - rhs[i]);
        const rel_diff = relativeDiffF32(lhs[i], rhs[i]);
        if (abs_diff > diff.max_abs) diff.max_abs = abs_diff;
        if (rel_diff > diff.max_rel) diff.max_rel = rel_diff;
        if (diff.first_diff_index == std.math.maxInt(usize) and abs_diff != 0) {
            diff.first_diff_index = i;
        }
    }
    return diff;
}

fn parseParityNodeIds(allocator: std.mem.Allocator, raw: []const u8, node_count: u32) ![]NodeId {
    var ids = std.ArrayListUnmanaged(NodeId).empty;
    errdefer ids.deinit(allocator);
    var split = std.mem.splitScalar(u8, raw, ',');
    while (split.next()) |part_raw| {
        const part = std.mem.trim(u8, part_raw, " \t\r\n");
        if (part.len == 0) continue;
        const node_id = try std.fmt.parseUnsigned(NodeId, part, 10);
        if (node_id >= node_count) return error.TrainingGraphExecutorParityInvalidNode;
        try ids.append(allocator, node_id);
    }
    return ids.toOwnedSlice(allocator);
}

fn printTrainingGraphExecutorParityReport(report: CompiledTrainSession.ParityReport) void {
    std.debug.print(
        "training_graph_executor_parity: passed={} direct_loss={d:.9} graph_loss={d:.9} loss_abs_diff={d:.9} loss_rel_diff={d:.9} max_grad_abs_diff={d:.9} max_grad_rel_diff={d:.9} compared_gradients={} compared_gradient_values={} missing_gradients={}\n",
        .{
            report.passed,
            report.direct_loss,
            report.graph_loss,
            report.loss_abs_diff,
            report.loss_rel_diff,
            report.max_grad_abs_diff,
            report.max_grad_rel_diff,
            report.compared_gradients,
            report.compared_gradient_values,
            report.missing_gradients,
        },
    );
}

const FallbackNativeBackend = struct {
    weight_store: *native_compute.WeightStore,
    compute: *native_compute.NativeCompute,
    backend: ComputeBackend,

    fn init(allocator: std.mem.Allocator) !FallbackNativeBackend {
        const weight_store = try allocator.create(native_compute.WeightStore);
        errdefer allocator.destroy(weight_store);
        weight_store.* = .{
            .allocator = allocator,
            .resident_weights = .{},
            .lazy_weights = .empty,
        };

        const compute = try allocator.create(native_compute.NativeCompute);
        errdefer allocator.destroy(compute);
        compute.* = native_compute.NativeCompute.init(allocator, weight_store, null);

        return .{
            .weight_store = weight_store,
            .compute = compute,
            .backend = compute.computeBackend(),
        };
    }

    fn deinit(self: *FallbackNativeBackend, allocator: std.mem.Allocator) void {
        self.backend.deinit();
        native_compute.deinitPrefetchQueue(self.weight_store);
        self.weight_store.resident_weights.deinit(allocator);
        self.weight_store.lazy_weights.deinit(allocator);
        allocator.destroy(self.weight_store);
    }
};

pub fn trainStep(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    loss_node: NodeId,
    cb: *const ComputeBackend,
    runtime_inputs: ?std.AutoHashMapUnmanaged(NodeId, CT),
    options: TrainStepOptions,
) !TrainStepResult {
    const total_start = nowNs();
    var profile: TrainStepProfile = .{};
    profile.peak_resident_bytes = currentResidentBytes();
    var checkpoint_summary: ?CheckpointSummary = null;

    // Determine which parameters to differentiate with respect to.
    const wrt = try resolveWrtParams(allocator, graph, options.trainable_params);
    defer allocator.free(wrt);

    // Run autodiff: lower fused ops and compute gradient graph.
    const autodiff_start = nowNs();
    var grad_result = try autodiff.gradient(allocator, graph, loss_node, wrt);
    profile.autodiff_ns = elapsedNs(autodiff_start);
    profile.peak_resident_bytes = @max(profile.peak_resident_bytes, currentResidentBytes());
    defer grad_result.deinit();

    if (options.checkpoint_config) |cfg| {
        const checkpoint_start = nowNs();
        if (options.emit_checkpoint_analysis) {
            var forward_count: u32 = 0;
            for (grad_result.id_map) |mapped| {
                if (mapped != null_node and mapped >= forward_count) {
                    forward_count = mapped + 1;
                }
            }
            if (forward_count > 0 and forward_count < grad_result.graph.nodeCount()) {
                const is_checkpoint = try checkpoint.identifyCheckpoints(allocator, &grad_result.graph, forward_count, cfg);
                defer allocator.free(is_checkpoint);
                const analysis = checkpoint.analyzeCheckpointSavings(&grad_result.graph, forward_count, is_checkpoint);
                checkpoint_summary = .{
                    .strategy = cfg.strategy,
                    .layer_interval = cfg.layer_interval,
                    .total_forward_activations = analysis.total_forward_activations,
                    .checkpointed_activations = analysis.checkpointed_activations,
                    .recomputable_activations = analysis.recomputable_activations,
                    .savings_ratio = analysis.savingsRatio(),
                };
            }
        }
        try checkpoint.applyCheckpointing(allocator, &grad_result, cfg);
        profile.checkpoint_ns = elapsedNs(checkpoint_start);
        profile.peak_resident_bytes = @max(profile.peak_resident_bytes, currentResidentBytes());
    }

    // Mark the loss (mapped through id_map) as an output if not already.
    // The lowered graph preserves the original outputs; we need the loss
    // output to extract its value, plus all gradient nodes.
    const lowered_loss = grad_result.id_map[loss_node];

    // Mark the lowered loss as the first output (clear existing outputs
    // and rebuild so loss is at index 0).
    grad_result.graph.outputs.clearRetainingCapacity();
    try grad_result.graph.markOutput(lowered_loss);

    // Mark each parameter gradient as an output. Skip null_node entries
    // (parameters unreachable from the loss get no gradient).
    for (grad_result.param_grads) |grad_id| {
        if (grad_id != null_node) {
            try grad_result.graph.markOutput(grad_id);
        }
    }

    // Build RuntimeInput slice, mapping original graph NodeIds through id_map.
    var rt_list = std.ArrayListUnmanaged(RuntimeInput).empty;
    defer rt_list.deinit(allocator);

    if (runtime_inputs) |rt| {
        var it = rt.iterator();
        while (it.next()) |entry| {
            const old_id = entry.key_ptr.*;
            const ct = entry.value_ptr.*;
            if (old_id < grad_result.id_map.len) {
                const new_id = grad_result.id_map[old_id];
                if (new_id != null_node) {
                    try rt_list.append(allocator, .{ .node_id = new_id, .value = ct });
                }
            }
        }
    }

    const rt_slice: ?[]const RuntimeInput = if (rt_list.items.len > 0)
        rt_list.items
    else
        null;

    // Execute the gradient graph.
    const execute_start = nowNs();
    const metal_debug_before = cb.debugTimingSnapshot().provider;
    var owned_execution_frame = try beginOwnedExecutionFrame(cb);
    errdefer cancelOwnedExecutionFrame(cb, &owned_execution_frame);
    var planned_scope = try beginTrainingPlannedScope(cb);
    errdefer endTrainingPlannedScope(cb, &planned_scope) catch {};
    var exec_result = try interpreter.execute(
        allocator,
        &grad_result.graph,
        cb,
        .{ .runtime_inputs = rt_slice },
    );
    try endTrainingPlannedScope(cb, &planned_scope);
    try submitOwnedExecutionFrame(cb, &owned_execution_frame);
    profile.execute_ns = elapsedNs(execute_start);
    recordMetalDebugProfile(&profile, metal_debug_before, cb.debugTimingSnapshot().provider);
    profile.peak_resident_bytes = @max(profile.peak_resident_bytes, currentResidentBytes());
    defer exec_result.deinit(cb);

    // Extract loss value from first output.
    const extract_start = nowNs();
    const loss_data = try cb.toFloat32(exec_result.outputs[0], allocator);
    defer allocator.free(loss_data);
    const loss_value: f32 = if (loss_data.len > 0) loss_data[0] else 0.0;

    // Extract gradient tensors keyed by parameter name.
    var gradients = std.StringHashMapUnmanaged([]f32){};
    errdefer {
        var it = gradients.iterator();
        while (it.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        gradients.deinit(allocator);
    }

    var grad_output_idx: usize = 1;
    for (wrt, 0..) |param_id, i| {
        const grad_id = grad_result.param_grads[i];
        if (grad_id == null_node) continue;

        // Output index: 0 = loss, followed by only the non-null gradients.
        const grad_data = try cb.toFloat32(exec_result.outputs[grad_output_idx], allocator);
        grad_output_idx += 1;

        const param_node = graph.node(param_id);
        const name = try allocator.dupe(u8, graph.parameterName(param_node));
        errdefer allocator.free(name);
        try gradients.put(allocator, name, grad_data);
    }

    profile.extract_ns = elapsedNs(extract_start);
    profile.total_ns = elapsedNs(total_start);
    profile.peak_resident_bytes = @max(profile.peak_resident_bytes, currentResidentBytes());

    return .{
        .loss = loss_value,
        .gradients = gradients,
        .profile = profile,
        .checkpoint_summary = checkpoint_summary,
        .allocator = allocator,
    };
}

fn elapsedNs(start_ns: u64) u64 {
    const end_ns = nowNs();
    return end_ns - start_ns;
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn compiledDiag(comptime fmt: []const u8, args: anytype) void {
    if (!platform.env.getenvBoolDefault("TERMITE_COMPILED_TRAIN_TRACE", false)) return;
    std.debug.print("[compiled-train] " ++ fmt ++ "\n", args);
}

fn nowNs() u64 {
    var timespec: std.posix.timespec = undefined;
    switch (std.posix.errno(std.posix.system.clock_gettime(std.posix.CLOCK.MONOTONIC, &timespec))) {
        .SUCCESS => return @intCast(@as(i128, timespec.sec) * std.time.ns_per_s + timespec.nsec),
        else => return 0,
    }
}

fn currentResidentBytes() usize {
    const usage = std.posix.getrusage(std.posix.rusage.SELF);
    if (usage.maxrss <= 0) return 0;

    const maxrss: usize = @intCast(usage.maxrss);
    return switch (builtin.os.tag) {
        // Darwin reports ru_maxrss in bytes; Linux reports KiB.
        .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => maxrss,
        .linux => std.math.mul(usize, maxrss, 1024) catch std.math.maxInt(usize),
        else => maxrss,
    };
}

fn cloneTensorForOutputShape(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    cb: *const ComputeBackend,
    tensor: CT,
    output_node: NodeId,
) !CT {
    const shape = graph.node(output_node).output_shape;
    var dims: [8]i32 = undefined;
    const rank = shape.rank();
    if (rank > dims.len) return error.UnsupportedShape;
    for (0..rank) |axis| {
        dims[axis] = std.math.cast(i32, shape.dim(@intCast(axis))) orelse return error.UnsupportedShape;
    }
    if (try cb.cloneTensorShape(tensor, dims[0..rank])) |cloned| return cloned;
    const data = try cb.toFloat32(tensor, allocator);
    defer allocator.free(data);
    return cb.fromFloat32Shape(data, dims[0..rank]);
}

/// Resolve which parameter NodeIds to differentiate. If trainable_params
/// is null, return all graph parameters. Otherwise filter by name.
fn resolveWrtParams(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    trainable_params: ?[]const []const u8,
) ![]NodeId {
    if (trainable_params) |names| {
        var result = std.ArrayListUnmanaged(NodeId).empty;
        errdefer result.deinit(allocator);

        for (graph.parameters.items) |param_id| {
            const param_node = graph.node(param_id);
            const param_name = graph.parameterName(param_node);
            for (names) |name| {
                if (std.mem.eql(u8, param_name, name)) {
                    try result.append(allocator, param_id);
                    break;
                }
            }
        }
        return try result.toOwnedSlice(allocator);
    } else {
        return allocator.dupe(NodeId, graph.parameters.items);
    }
}

// ── PJRT compile-once training session ─────────────────────────────

const pjrt_compiler_mod = @import("pjrt_compiler.zig");
const build_options = @import("build_options");
const pjrt_pkg = if (build_options.enable_pjrt) @import("pjrt") else struct {
    pub const pjrt = struct {
        pub const Client = void;
        pub const LoadedExecutable = void;
        pub const Buffer = void;
    };
};

/// Input for a PJRT training step (f32 data + shape).
pub const PjrtRuntimeInput = struct {
    data: []const f32,
    shape: []const i32,
};

/// A compiled training session: run autodiff once, then execute many times.
/// Inputs and outputs are identified by their original graph NodeIds.
pub const PjrtTrainSession = struct {
    executable: if (build_options.enable_pjrt) pjrt_pkg.pjrt.LoadedExecutable else void,
    client: if (build_options.enable_pjrt) *pjrt_pkg.pjrt.Client else void,
    /// Ordered list of original (pre-lowering) NodeIds that are runtime inputs.
    /// Caller must provide them in this exact order in execute().
    input_node_ids: []NodeId,
    /// Ordered list of output names: output_names[0] = "loss", output_names[1..] = param names.
    output_names: [][]u8,
    num_outputs: usize,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PjrtTrainSession) void {
        if (comptime build_options.enable_pjrt) {
            self.executable.deinit();
        }
        self.allocator.free(self.input_node_ids);
        for (self.output_names) |name| self.allocator.free(name);
        self.allocator.free(self.output_names);
    }

    /// Execute the compiled gradient program.
    /// `runtime_inputs`: map from original NodeId → f32 data + shape.
    /// Returns TrainStepResult with loss and gradients.
    pub fn execute(
        self: *const PjrtTrainSession,
        allocator: std.mem.Allocator,
        runtime_inputs: std.AutoHashMapUnmanaged(NodeId, PjrtRuntimeInput),
    ) !TrainStepResult {
        if (comptime !build_options.enable_pjrt) return error.PjrtNotCompiled;

        // Upload inputs to device in order.
        var device_inputs = try allocator.alloc(pjrt_pkg.pjrt.Buffer, self.input_node_ids.len);
        var device_inputs_init: usize = 0;
        defer {
            for (device_inputs[0..device_inputs_init]) |*buf| buf.deinit();
            allocator.free(device_inputs);
        }
        for (self.input_node_ids, 0..) |node_id, ii| {
            const rt = runtime_inputs.get(node_id) orelse return error.MissingRuntimeInput;
            var dims = try allocator.alloc(i64, rt.shape.len);
            defer allocator.free(dims);
            for (rt.shape, 0..) |d, j| dims[j] = @intCast(d);
            device_inputs[ii] = try self.client.bufferFromHostFloat32(rt.data, dims);
            device_inputs_init += 1;
        }

        // Execute.
        const output_buffers = try self.executable.execute(device_inputs, allocator);
        defer {
            for (output_buffers) |*buf| buf.deinit();
            allocator.free(output_buffers);
        }

        // Download outputs. Output 0 = loss, outputs 1..N = gradients.
        const loss_data = try output_buffers[0].toFloat32(allocator);
        defer allocator.free(loss_data);
        const loss_value: f32 = if (loss_data.len > 0) loss_data[0] else 0.0;

        var gradients = std.StringHashMapUnmanaged([]f32){};
        errdefer {
            var it = gradients.iterator();
            while (it.next()) |e| {
                allocator.free(e.key_ptr.*);
                allocator.free(e.value_ptr.*);
            }
            gradients.deinit(allocator);
        }
        for (1..self.num_outputs) |oi| {
            const grad_data = try output_buffers[oi].toFloat32(allocator);
            errdefer allocator.free(grad_data);
            const name = try allocator.dupe(u8, self.output_names[oi]); // output_names[0]="loss", [1..]=param names
            errdefer allocator.free(name);
            try gradients.put(allocator, name, grad_data);
        }

        return .{
            .loss = loss_value,
            .gradients = gradients,
            .allocator = allocator,
        };
    }
};

/// Compile a gradient training program for repeated PJRT execution.
///
/// Runs autodiff on `graph` to produce a lowered gradient graph,
/// compiles it to HLO via compileGradientGraph, then compiles to a
/// PJRT LoadedExecutable. The returned PjrtTrainSession can be executed
/// many times with different runtime inputs.
///
/// `runtime_input_node_ids`: ordered slice of NodeIds (in the ORIGINAL graph)
/// that will be provided as runtime inputs. These must cover all parameter nodes.
pub fn compilePjrtTrainSession(
    allocator: std.mem.Allocator,
    graph: *const Graph,
    loss_node: NodeId,
    pjrt_client: anytype,
    runtime_input_node_ids: []const NodeId,
    options: TrainStepOptions,
) !PjrtTrainSession {
    if (comptime !build_options.enable_pjrt) return error.PjrtNotCompiled;

    // Resolve trainable params.
    const wrt = try resolveWrtParams(allocator, graph, options.trainable_params);
    defer allocator.free(wrt);

    // Run autodiff → lowered graph + gradient node IDs.
    var grad_result = try autodiff.gradient(allocator, graph, loss_node, wrt);
    defer grad_result.deinit();

    // Mark loss + gradients as outputs in the lowered graph.
    grad_result.graph.outputs.clearRetainingCapacity();
    const lowered_loss = grad_result.id_map[loss_node];
    try grad_result.graph.markOutput(lowered_loss);
    for (grad_result.param_grads) |grad_id| {
        try grad_result.graph.markOutput(grad_id);
    }

    // Build input_ids: map original runtime input NodeIds through id_map.
    const input_ids = try allocator.alloc(NodeId, runtime_input_node_ids.len);
    defer allocator.free(input_ids);
    for (runtime_input_node_ids, 0..) |orig_id, ii| {
        input_ids[ii] = grad_result.id_map[orig_id];
    }

    // Build output_ids: [loss, grad_0, grad_1, ...]
    const num_outputs = 1 + grad_result.param_grads.len;
    const output_ids = try allocator.alloc(NodeId, num_outputs);
    defer allocator.free(output_ids);
    output_ids[0] = lowered_loss;
    for (grad_result.param_grads, 0..) |gid, gi| output_ids[gi + 1] = gid;

    // Compile the lowered graph to HLO.
    var compile_result = try pjrt_compiler_mod.compileGradientGraph(
        allocator,
        &grad_result.graph,
        input_ids,
        output_ids,
    );
    defer compile_result.deinit();

    // Compile to PJRT executable.
    var executable = try pjrt_client.compile(compile_result.hlo_bytes, num_outputs);
    errdefer executable.deinit();

    // Build output_names: ["loss", param_name_0, param_name_1, ...]
    const output_names = try allocator.alloc([]u8, num_outputs);
    var output_names_init: usize = 0;
    errdefer {
        for (output_names[0..output_names_init]) |n| allocator.free(n);
        allocator.free(output_names);
    }
    output_names[0] = try allocator.dupe(u8, "loss");
    output_names_init = 1;
    for (wrt, 0..) |param_id, pi| {
        const param_node = graph.node(param_id);
        const name = graph.parameterName(param_node);
        output_names[pi + 1] = try allocator.dupe(u8, name);
        output_names_init += 1;
    }

    // Copy runtime_input_node_ids for storage in the session.
    const stored_input_ids = try allocator.dupe(NodeId, runtime_input_node_ids);
    errdefer allocator.free(stored_input_ids);

    return .{
        .executable = executable,
        .client = pjrt_client,
        .input_node_ids = stored_input_ids,
        .output_names = output_names,
        .num_outputs = num_outputs,
        .allocator = allocator,
    };
}

// ── Tests ──────────────────────────────────────────────────────────────

const native_mod = @import("../ops/native_compute.zig");
const NativeCompute = native_mod.NativeCompute;
const WeightStore = native_mod.WeightStore;

test "trainStep computes loss and gradients for linear model" {
    // Build: y = Wx + b, loss = MSE(y, target) via reduceSum((y - target)^2) / n
    // Simplified: loss = reduceSum(linear(x, w, b))  (scalar loss)
    const allocator = std.testing.allocator;

    var g = Graph.init(allocator);
    defer g.deinit();
    var bld = Builder.init(&g);

    // x:[2,4], w:[3,4], bias:[3] -> y:[2,3] -> loss = sum(y)
    const x = try bld.parameter("x", Shape.init(.f32, &.{ 2, 4 }));
    const w = try bld.parameter("w", Shape.init(.f32, &.{ 3, 4 }));
    const bias = try bld.parameter("bias", Shape.init(.f32, &.{3}));
    const y = try bld.linear(x, w, bias, 2, 4, 3);
    const loss = try bld.reduceSum(y, &.{ 0, 1 });
    try g.markOutput(loss);

    // Set up native backend with empty WeightStore.
    var ws = WeightStore{ .allocator = allocator, .resident_weights = .{}, .lazy_weights = .{} };
    var compute = NativeCompute.init(allocator, &ws, null);
    var cb_val = compute.computeBackend();

    // Create parameter CTs.
    const x_ct = try cb_val.fromFloat32(&.{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer cb_val.free(x_ct);
    const w_ct = try cb_val.fromFloat32(&.{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 });
    defer cb_val.free(w_ct);
    const bias_ct = try cb_val.fromFloat32(&.{ 0.01, 0.02, 0.03 });
    defer cb_val.free(bias_ct);

    var rt = std.AutoHashMapUnmanaged(NodeId, CT){};
    defer rt.deinit(allocator);
    try rt.put(allocator, x, x_ct);
    try rt.put(allocator, w, w_ct);
    try rt.put(allocator, bias, bias_ct);

    var result = try trainStep(allocator, &g, loss, &cb_val, rt, .{});
    defer result.deinit();

    // Loss should be positive (sum of y values).
    try std.testing.expect(result.loss > 0.0);

    // Should have gradients for w and bias (and x, though x is also a "parameter" here).
    try std.testing.expect(result.gradients.get("w") != null);
    try std.testing.expect(result.gradients.get("bias") != null);
    try std.testing.expect(result.gradients.get("x") != null);

    // w gradient should have 12 elements (3x4)
    try std.testing.expectEqual(@as(usize, 12), result.gradients.get("w").?.len);
    // bias gradient should have 3 elements
    try std.testing.expectEqual(@as(usize, 3), result.gradients.get("bias").?.len);
}

test "CompiledTrainSession can retain gradient tensors without host extraction" {
    const allocator = std.testing.allocator;

    var g = Graph.init(allocator);
    defer g.deinit();
    var bld = Builder.init(&g);

    const x = try bld.parameter("x", Shape.init(.f32, &.{ 2, 4 }));
    const w = try bld.parameter("w", Shape.init(.f32, &.{ 3, 4 }));
    const bias = try bld.parameter("bias", Shape.init(.f32, &.{3}));
    const y = try bld.linear(x, w, bias, 2, 4, 3);
    const loss = try bld.reduceSum(y, &.{ 0, 1 });
    try g.markOutput(loss);

    var session = try CompiledTrainSession.init(allocator, &g, loss, .{ .trainable_params = &.{ "w", "bias" } });
    defer session.deinit();

    var ws = WeightStore{ .allocator = allocator, .resident_weights = .{}, .lazy_weights = .{} };
    var compute = NativeCompute.init(allocator, &ws, null);
    var cb_val = compute.computeBackend();

    const x_ct = try cb_val.fromFloat32Shape(&.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &.{ 2, 4 });
    defer cb_val.free(x_ct);
    const w_ct = try cb_val.fromFloat32Shape(&.{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 }, &.{ 3, 4 });
    defer cb_val.free(w_ct);
    const bias_ct = try cb_val.fromFloat32Shape(&.{ 0.01, 0.02, 0.03 }, &.{3});
    defer cb_val.free(bias_ct);

    var rt = std.AutoHashMapUnmanaged(NodeId, CT){};
    defer rt.deinit(allocator);
    try rt.put(allocator, x, x_ct);
    try rt.put(allocator, w, w_ct);
    try rt.put(allocator, bias, bias_ct);

    var result = try session.executeDeviceGradients(&cb_val, rt);
    defer result.deinit();

    try std.testing.expect(result.loss > 0.0);
    try std.testing.expect(result.gradients.count() == 0);
    const w_grad = result.device_gradients.get("w") orelse return error.MissingGradient;
    const w_data = try cb_val.toFloat32(w_grad, allocator);
    defer allocator.free(w_data);
    try std.testing.expectEqual(@as(usize, 12), w_data.len);
    try std.testing.expect(result.device_gradients.get("bias") != null);
}

test "trainStep on linear-gelu chain" {
    // Build: y = gelu(linear(x, w, b)), loss = reduceSum(y)
    const allocator = std.testing.allocator;

    var g = Graph.init(allocator);
    defer g.deinit();
    var bld = Builder.init(&g);

    // x:[2,4], w:[3,4], bias:[3] -> linear:[2,3] -> gelu -> loss = sum
    const x = try bld.parameter("x", Shape.init(.f32, &.{ 2, 4 }));
    const w = try bld.parameter("w", Shape.init(.f32, &.{ 3, 4 }));
    const bias = try bld.parameter("bias", Shape.init(.f32, &.{3}));
    const y = try bld.linear(x, w, bias, 2, 4, 3);
    const activated = try bld.gelu(y);
    const loss = try bld.reduceSum(activated, &.{ 0, 1 });
    try g.markOutput(loss);

    // Set up native backend.
    var ws = WeightStore{ .allocator = allocator, .resident_weights = .{}, .lazy_weights = .{} };
    var compute = NativeCompute.init(allocator, &ws, null);
    var cb_val = compute.computeBackend();

    // Create parameter CTs with small positive values.
    const x_ct = try cb_val.fromFloat32(&.{ 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 });
    defer cb_val.free(x_ct);
    const w_ct = try cb_val.fromFloat32(&.{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2 });
    defer cb_val.free(w_ct);
    const bias_ct = try cb_val.fromFloat32(&.{ 0.01, 0.02, 0.03 });
    defer cb_val.free(bias_ct);

    var rt = std.AutoHashMapUnmanaged(NodeId, CT){};
    defer rt.deinit(allocator);
    try rt.put(allocator, x, x_ct);
    try rt.put(allocator, w, w_ct);
    try rt.put(allocator, bias, bias_ct);

    var result = try trainStep(allocator, &g, loss, &cb_val, rt, .{});
    defer result.deinit();

    // Loss should be positive (gelu of positive values -> positive).
    try std.testing.expect(result.loss > 0.0);

    // All three params should have gradients.
    try std.testing.expect(result.gradients.get("x") != null);
    try std.testing.expect(result.gradients.get("w") != null);
    try std.testing.expect(result.gradients.get("bias") != null);

    // Gradient shapes should match parameter shapes.
    try std.testing.expectEqual(@as(usize, 8), result.gradients.get("x").?.len); // 2x4
    try std.testing.expectEqual(@as(usize, 12), result.gradients.get("w").?.len); // 3x4
    try std.testing.expectEqual(@as(usize, 3), result.gradients.get("bias").?.len); // 3
}

test "trainStep emits checkpoint summary when enabled" {
    const allocator = std.testing.allocator;

    var g = Graph.init(allocator);
    defer g.deinit();
    var bld = Builder.init(&g);

    const x = try bld.parameter("x", Shape.init(.f32, &.{ 2, 4 }));
    const w1 = try bld.parameter("w1", Shape.init(.f32, &.{ 4, 4 }));
    const w2 = try bld.parameter("w2", Shape.init(.f32, &.{ 4, 4 }));
    const y1 = try bld.linearNoBias(x, w1, 2, 4, 4);
    const y2 = try bld.linearNoBias(y1, w2, 2, 4, 4);
    const loss = try bld.reduceSum(y2, &.{ 0, 1 });
    try g.markOutput(loss);

    var ws = WeightStore{ .allocator = allocator, .resident_weights = .{}, .lazy_weights = .{} };
    var compute = NativeCompute.init(allocator, &ws, null);
    var cb_val = compute.computeBackend();

    const x_ct = try cb_val.fromFloat32(&.{ 1, 1, 1, 1, 1, 1, 1, 1 });
    defer cb_val.free(x_ct);
    const w1_ct = try cb_val.fromFloat32(&.{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6 });
    defer cb_val.free(w1_ct);
    const w2_ct = try cb_val.fromFloat32(&.{ 0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7, 1.0, 0.9, 1.2, 1.1, 1.4, 1.3, 1.6, 1.5 });
    defer cb_val.free(w2_ct);

    var rt = std.AutoHashMapUnmanaged(NodeId, CT){};
    defer rt.deinit(allocator);
    try rt.put(allocator, x, x_ct);
    try rt.put(allocator, w1, w1_ct);
    try rt.put(allocator, w2, w2_ct);

    var result = try trainStep(allocator, &g, loss, &cb_val, rt, .{
        .checkpoint_config = .{ .strategy = .every_n_layers, .layer_interval = 1 },
        .emit_checkpoint_analysis = true,
    });
    defer result.deinit();

    try std.testing.expect(result.checkpoint_summary != null);
    try std.testing.expect(result.checkpoint_summary.?.recomputable_activations > 0);
}
