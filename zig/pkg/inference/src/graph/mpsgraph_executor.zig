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
const ops_mod = @import("../ops/ops.zig");
const interpreter = @import("interpreter.zig");
const transpose_utils = @import("transpose_utils.zig");

const Graph = ml.graph.Graph;
const NodeId = ml.graph.NodeId;
const null_node = ml.graph.null_node;
const OpCode = ml.graph.OpCode;
const Shape = ml.graph.Shape;
const ComputeBackend = ops_mod.ComputeBackend;
const RuntimeInput = interpreter.RuntimeInput;
const CT = ops_mod.CT;

const CError = extern struct {
    code: c_int = 0,
    message: ?[*:0]const u8 = null,
};

extern fn termite_mpsgraph_error_message_free(message: ?[*:0]const u8) void;
extern fn termite_mpsgraph_context_create(error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_context_destroy(handle: ?*anyopaque) void;
extern fn termite_mpsgraph_placeholder_f32(handle: ?*anyopaque, shape: [*]const i64, rank: c_int, error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_constant_f32(handle: ?*anyopaque, data: [*]const f32, elems: i64, shape: [*]const i64, rank: c_int, error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_unary(handle: ?*anyopaque, op: c_int, x: ?*anyopaque, error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_binary(handle: ?*anyopaque, op: c_int, lhs: ?*anyopaque, rhs: ?*anyopaque, error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_where(handle: ?*anyopaque, cond: ?*anyopaque, on_true: ?*anyopaque, on_false: ?*anyopaque, error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_matmul(handle: ?*anyopaque, lhs: ?*anyopaque, rhs: ?*anyopaque, error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_reduce(handle: ?*anyopaque, op: c_int, x: ?*anyopaque, axes: [*]const c_int, num_axes: c_int, error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_reshape(handle: ?*anyopaque, x: ?*anyopaque, shape: [*]const i64, rank: c_int, error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_transpose(handle: ?*anyopaque, x: ?*anyopaque, perm: [*]const c_int, rank: c_int, error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_broadcast(handle: ?*anyopaque, x: ?*anyopaque, shape: [*]const i64, rank: c_int, error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_slice(handle: ?*anyopaque, x: ?*anyopaque, starts: [*]const i64, ends: [*]const i64, strides: [*]const i64, rank: c_int, error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_concat(handle: ?*anyopaque, tensors: [*]const ?*anyopaque, num_tensors: c_int, axis: c_int, error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_softmax(handle: ?*anyopaque, x: ?*anyopaque, axis: c_int, error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_compile(handle: ?*anyopaque, feeds: [*]const ?*anyopaque, feed_shapes: [*]const i64, feed_ranks: [*]const c_int, num_feeds: c_int, targets: [*]const ?*anyopaque, num_targets: c_int, error_out: *CError) ?*anyopaque;
extern fn termite_mpsgraph_executable_destroy(handle: ?*anyopaque) void;
extern fn termite_mpsgraph_execute_f32(handle: ?*anyopaque, input_data: [*]const [*]const f32, input_elems: [*]const i64, input_shapes: [*]const i64, input_ranks: [*]const c_int, num_inputs: c_int, output_data: [*]const [*]f32, output_elems: [*]const i64, output_shapes: [*]const i64, output_ranks: [*]const c_int, num_outputs: c_int, error_out: *CError) c_int;

const UnaryOp = enum(c_int) {
    neg = 1,
    sqrt = 2,
    rsqrt = 3,
    exp = 4,
    erf = 5,
};

const BinaryOp = enum(c_int) {
    add = 1,
    mul = 2,
    sub = 3,
    div = 4,
    less_than = 5,
};

const ReduceOp = enum(c_int) {
    sum = 1,
    mean = 2,
    max = 3,
};

/// A feed provided directly as a host f32 buffer. Skips the per-execute
/// ComputeBackend tensor round-trip (`fromFloat32Shape` + `toFloat32`) for
/// callers that already hold stable host data (e.g. cached frozen weights).
pub const HostInput = struct {
    node_id: NodeId,
    data: []const f32,
};

pub const CompiledGraph = struct {
    allocator: std.mem.Allocator,
    executable: ?*anyopaque,
    feed_node_ids: []NodeId,
    feed_shapes: []Shape,
    output_shapes: []Shape,

    pub fn compile(allocator: std.mem.Allocator, graph: *const Graph) !CompiledGraph {
        const support = supportSummary(graph);
        if (!support.allSupported()) {
            std.log.warn(
                "mpsgraph unsupported graph: {d}/{d} nodes unsupported, first op={s} node={d}",
                .{
                    support.unsupported,
                    support.nodes,
                    support.first_unsupported_op orelse "<unknown>",
                    support.first_unsupported_node orelse 0,
                },
            );
            return error.MpsGraphUnsupportedGraph;
        }

        var err: CError = .{};
        const ctx = termite_mpsgraph_context_create(&err) orelse return failCError(&err);
        defer termite_mpsgraph_context_destroy(ctx);

        var lowerer = Lowerer.init(allocator, graph, ctx);
        defer lowerer.deinit();
        try lowerer.lower();

        var target_handles = try allocator.alloc(?*anyopaque, graph.outputs.items.len);
        defer allocator.free(target_handles);
        for (graph.outputs.items, 0..) |out_id, i| {
            target_handles[i] = lowerer.value(out_id) orelse return error.MpsGraphMissingValue;
        }

        var err_compile: CError = .{};
        const exec = termite_mpsgraph_compile(
            ctx,
            lowerer.feed_handles.items.ptr,
            lowerer.feed_shape_flat.items.ptr,
            lowerer.feed_ranks.items.ptr,
            @intCast(lowerer.feed_handles.items.len),
            target_handles.ptr,
            @intCast(target_handles.len),
            &err_compile,
        ) orelse return failCError(&err_compile);
        errdefer termite_mpsgraph_executable_destroy(exec);

        const feed_node_ids = try lowerer.feed_node_ids.toOwnedSlice(allocator);
        errdefer allocator.free(feed_node_ids);
        const feed_shapes = try lowerer.feed_shapes.toOwnedSlice(allocator);
        errdefer allocator.free(feed_shapes);
        const output_shapes = try allocator.alloc(Shape, graph.outputs.items.len);
        errdefer allocator.free(output_shapes);
        for (graph.outputs.items, output_shapes) |out_id, *shape| {
            shape.* = graph.node(out_id).output_shape;
        }

        return .{
            .allocator = allocator,
            .executable = exec,
            .feed_node_ids = feed_node_ids,
            .feed_shapes = feed_shapes,
            .output_shapes = output_shapes,
        };
    }

    pub fn deinit(self: *CompiledGraph) void {
        termite_mpsgraph_executable_destroy(self.executable);
        self.allocator.free(self.feed_node_ids);
        self.allocator.free(self.feed_shapes);
        self.allocator.free(self.output_shapes);
        self.* = undefined;
    }

    pub fn execute(
        self: *CompiledGraph,
        allocator: std.mem.Allocator,
        cb: *const ComputeBackend,
        runtime_inputs: ?[]const RuntimeInput,
    ) !ExecutionResult {
        return self.executeWithHostInputs(allocator, cb, runtime_inputs, null);
    }

    /// Like `execute`, but feeds present in `host_inputs` are passed straight
    /// from the caller's host f32 buffers without materializing a backend
    /// tensor first. Byte-identical to the CT path (f32 in, f32 out); it only
    /// removes redundant host copies.
    pub fn executeWithHostInputs(
        self: *CompiledGraph,
        allocator: std.mem.Allocator,
        cb: *const ComputeBackend,
        runtime_inputs: ?[]const RuntimeInput,
        host_inputs: ?[]const HostInput,
    ) !ExecutionResult {
        const n_inputs = self.feed_node_ids.len;
        var input_buffers = try allocator.alloc([]f32, n_inputs);
        defer allocator.free(input_buffers);
        var input_owned = try allocator.alloc(bool, n_inputs);
        defer allocator.free(input_owned);
        @memset(input_owned, false);
        var materialized: usize = 0;
        errdefer {
            for (input_buffers[0..materialized], input_owned[0..materialized]) |buf, owned| {
                if (owned) allocator.free(buf);
            }
        }

        var input_ptrs = try allocator.alloc([*]const f32, n_inputs);
        defer allocator.free(input_ptrs);
        var input_elems = try allocator.alloc(i64, n_inputs);
        defer allocator.free(input_elems);
        var input_ranks = try allocator.alloc(c_int, n_inputs);
        defer allocator.free(input_ranks);
        var input_shape_flat = std.ArrayListUnmanaged(i64).empty;
        defer input_shape_flat.deinit(allocator);

        for (self.feed_node_ids, 0..) |node_id, i| {
            const expected = try shapeElementCount(self.feed_shapes[i]);
            if (findHostInput(host_inputs, node_id)) |host_data| {
                if (host_data.len != expected) return error.MpsGraphInputShapeMismatch;
                input_buffers[i] = @constCast(host_data);
                input_owned[i] = false;
                materialized += 1;
                input_ptrs[i] = host_data.ptr;
                input_elems[i] = @intCast(host_data.len);
            } else {
                const ct = findRuntimeInput(runtime_inputs, node_id) orelse return error.MpsGraphMissingRuntimeInput;
                const data = try cb.toFloat32(ct, allocator);
                errdefer allocator.free(data);
                if (data.len != expected) return error.MpsGraphInputShapeMismatch;
                input_buffers[i] = data;
                input_owned[i] = true;
                materialized += 1;
                input_ptrs[i] = data.ptr;
                input_elems[i] = @intCast(data.len);
            }
            input_ranks[i] = @intCast(self.feed_shapes[i].rank());
            try appendShape(&input_shape_flat, allocator, self.feed_shapes[i]);
        }

        const n_outputs = self.output_shapes.len;
        var outputs = try allocator.alloc([]f32, n_outputs);
        errdefer allocator.free(outputs);
        var output_count: usize = 0;
        errdefer {
            for (outputs[0..output_count]) |out| allocator.free(out);
        }

        var output_ptrs = try allocator.alloc([*]f32, n_outputs);
        defer allocator.free(output_ptrs);
        var output_elems = try allocator.alloc(i64, n_outputs);
        defer allocator.free(output_elems);
        var output_ranks = try allocator.alloc(c_int, n_outputs);
        defer allocator.free(output_ranks);
        var output_shape_flat = std.ArrayListUnmanaged(i64).empty;
        defer output_shape_flat.deinit(allocator);

        for (self.output_shapes, 0..) |shape, i| {
            const elems = try shapeElementCount(shape);
            const data = try allocator.alloc(f32, elems);
            outputs[i] = data;
            output_count += 1;
            output_ptrs[i] = data.ptr;
            output_elems[i] = @intCast(elems);
            output_ranks[i] = @intCast(shape.rank());
            try appendShape(&output_shape_flat, allocator, shape);
        }

        var err: CError = .{};
        const rc = termite_mpsgraph_execute_f32(
            self.executable,
            input_ptrs.ptr,
            input_elems.ptr,
            input_shape_flat.items.ptr,
            input_ranks.ptr,
            @intCast(n_inputs),
            output_ptrs.ptr,
            output_elems.ptr,
            output_shape_flat.items.ptr,
            output_ranks.ptr,
            @intCast(n_outputs),
            &err,
        );
        if (rc != 0) return failCError(&err);

        for (input_buffers[0..materialized], input_owned[0..materialized]) |buf, owned| {
            if (owned) allocator.free(buf);
        }
        materialized = 0;

        return .{
            .allocator = allocator,
            .outputs = outputs,
        };
    }
};

pub const ExecutionResult = struct {
    allocator: std.mem.Allocator,
    outputs: [][]f32,

    pub fn deinit(self: *ExecutionResult) void {
        for (self.outputs) |out| self.allocator.free(out);
        self.allocator.free(self.outputs);
        self.* = undefined;
    }
};

const Lowerer = struct {
    allocator: std.mem.Allocator,
    graph: *const Graph,
    ctx: ?*anyopaque,
    values: []?*anyopaque,
    feed_handles: std.ArrayListUnmanaged(?*anyopaque) = .empty,
    feed_node_ids: std.ArrayListUnmanaged(NodeId) = .empty,
    feed_shapes: std.ArrayListUnmanaged(Shape) = .empty,
    feed_ranks: std.ArrayListUnmanaged(c_int) = .empty,
    feed_shape_flat: std.ArrayListUnmanaged(i64) = .empty,

    fn init(allocator: std.mem.Allocator, graph: *const Graph, ctx: ?*anyopaque) Lowerer {
        return .{
            .allocator = allocator,
            .graph = graph,
            .ctx = ctx,
            .values = &.{},
        };
    }

    fn deinit(self: *Lowerer) void {
        self.allocator.free(self.values);
        self.feed_handles.deinit(self.allocator);
        self.feed_node_ids.deinit(self.allocator);
        self.feed_shapes.deinit(self.allocator);
        self.feed_ranks.deinit(self.allocator);
        self.feed_shape_flat.deinit(self.allocator);
    }

    fn lower(self: *Lowerer) !void {
        self.values = try self.allocator.alloc(?*anyopaque, self.graph.nodeCount());
        @memset(self.values, null);
        // Lower dependencies-first rather than in raw id order: graph
        // rewrites (e.g. LoRA injection redirecting consumers to the
        // combined output) can leave a node whose input has a HIGHER id
        // than itself, so a plain ascending-id pass would read a value
        // that has not been lowered yet (MpsGraphMissingInput).
        var stack: std.ArrayListUnmanaged(NodeId) = .empty;
        defer stack.deinit(self.allocator);
        for (0..self.graph.nodeCount()) |raw_id| {
            try self.lowerSubgraph(@intCast(raw_id), &stack);
        }
    }

    fn lowerSubgraph(self: *Lowerer, root: NodeId, stack: *std.ArrayListUnmanaged(NodeId)) !void {
        if (self.values[root] != null) return;
        try stack.append(self.allocator, root);
        while (stack.items.len > 0) {
            const node_id = stack.items[stack.items.len - 1];
            if (self.values[node_id] != null) {
                _ = stack.pop();
                continue;
            }
            const node = self.graph.node(node_id);
            var pending = false;
            for (node.inputs[0..node.num_inputs]) |in_id| {
                if (in_id == null_node) continue;
                if (self.values[in_id] == null) {
                    try stack.append(self.allocator, in_id);
                    pending = true;
                }
            }
            if (pending) continue;
            _ = stack.pop();
            self.values[node_id] = try self.lowerNode(node_id);
        }
    }

    fn value(self: *const Lowerer, node_id: NodeId) ?*anyopaque {
        if (node_id == null_node or node_id >= self.values.len) return null;
        return self.values[node_id];
    }

    fn input(self: *const Lowerer, node_id: NodeId, index: usize) !?*anyopaque {
        const node = self.graph.node(node_id);
        if (index >= node.num_inputs) return error.MpsGraphMissingInput;
        return self.value(node.inputs[index]) orelse error.MpsGraphMissingInput;
    }

    fn lowerNode(self: *Lowerer, node_id: NodeId) !?*anyopaque {
        const node = self.graph.node(node_id);
        switch (node.op) {
            .parameter => {
                var shape_buf: [8]i64 = undefined;
                const dims = shapeDims(node.output_shape, &shape_buf);
                var err: CError = .{};
                const handle = termite_mpsgraph_placeholder_f32(self.ctx, dims.ptr, @intCast(dims.len), &err) orelse return failCError(&err);
                try self.feed_handles.append(self.allocator, handle);
                try self.feed_node_ids.append(self.allocator, node_id);
                try self.feed_shapes.append(self.allocator, node.output_shape);
                try self.feed_ranks.append(self.allocator, @intCast(dims.len));
                try self.feed_shape_flat.appendSlice(self.allocator, dims);
                return handle;
            },
            .constant => |attrs| {
                const constant_view = try self.graph.constantDataAsF32(
                    self.allocator,
                    node.output_shape.dtype,
                    attrs.data_offset,
                    attrs.data_len,
                );
                defer constant_view.deinit(self.allocator);
                var shape_buf: [8]i64 = undefined;
                const dims = shapeDims(node.output_shape, &shape_buf);
                return self.constant(dims, constant_view.data);
            },
            .neg => return self.unary(.neg, try self.input(node_id, 0)),
            .sqrt => return self.unary(.sqrt, try self.input(node_id, 0)),
            .rsqrt => return self.unary(.rsqrt, try self.input(node_id, 0)),
            .exp => return self.unary(.exp, try self.input(node_id, 0)),
            .erf => return self.unary(.erf, try self.input(node_id, 0)),
            .add => return self.binary(.add, try self.input(node_id, 0), try self.input(node_id, 1)),
            .mul => return self.binary(.mul, try self.input(node_id, 0), try self.input(node_id, 1)),
            // Same equivalence the Metal partition executor uses:
            // fused_elem_multiply is exactly an element-wise mul of its two
            // inputs (the fused form only exists to carry a vjp_alternate).
            .fused_elem_multiply => return self.binary(.mul, try self.input(node_id, 0), try self.input(node_id, 1)),
            .sub => return self.binary(.sub, try self.input(node_id, 0), try self.input(node_id, 1)),
            .div => return self.binary(.div, try self.input(node_id, 0), try self.input(node_id, 1)),
            .less_than => return self.binary(.less_than, try self.input(node_id, 0), try self.input(node_id, 1)),
            .where_select => return self.where(try self.input(node_id, 0), try self.input(node_id, 1), try self.input(node_id, 2)),
            .reduce_sum => |attrs| return self.reduce(.sum, try self.input(node_id, 0), attrs.axes[0..attrs.num_axes], node.output_shape),
            .reduce_mean => |attrs| return self.reduce(.mean, try self.input(node_id, 0), attrs.axes[0..attrs.num_axes], node.output_shape),
            .reduce_max => |attrs| return self.reduce(.max, try self.input(node_id, 0), attrs.axes[0..attrs.num_axes], node.output_shape),
            .reshape => |attrs| return self.reshape(try self.input(node_id, 0), attrs.new_shape),
            .transpose => |attrs| return self.transpose(try self.input(node_id, 0), attrs, self.graph.node(node.inputs[0]).output_shape.rank()),
            .broadcast_in_dim => |attrs| return self.broadcastInDim(try self.input(node_id, 0), self.graph.node(node.inputs[0]).output_shape, attrs),
            .slice => |attrs| return self.slice(try self.input(node_id, 0), self.graph.node(node.inputs[0]).output_shape, attrs),
            .concat_prim => |attrs| return self.concat(try self.input(node_id, 0), try self.input(node_id, 1), attrs.axis),
            .dot_general => |attrs| return self.dotGeneral(try self.input(node_id, 0), self.graph.node(node.inputs[0]).output_shape, try self.input(node_id, 1), self.graph.node(node.inputs[1]).output_shape, attrs),
            .fused_linear => |attrs| return self.linear(try self.input(node_id, 0), try self.input(node_id, 1), try self.input(node_id, 2), attrs),
            .fused_linear_no_bias => |attrs| return self.linearNoBias(try self.input(node_id, 0), try self.input(node_id, 1), attrs),
            .fused_layer_norm => |attrs| return self.layerNorm(try self.input(node_id, 0), try self.input(node_id, 1), try self.input(node_id, 2), self.graph.node(node.inputs[0]).output_shape, attrs.eps),
            .fused_gelu => return self.gelu(try self.input(node_id, 0)),
            .fused_softmax => return self.softmax(try self.input(node_id, 0), self.graph.node(node.inputs[0]).output_shape),
            .fused_rope => |attrs| return self.rope(try self.input(node_id, 0), try self.input(node_id, 1), try self.input(node_id, 2), self.graph.node(node.inputs[0]).output_shape, attrs),
            else => return error.MpsGraphUnsupportedOp,
        }
    }

    fn constant(self: *Lowerer, dims: []const i64, data: []const f32) !?*anyopaque {
        var scalar_dims = [_]i64{1};
        const ranked_dims = if (dims.len == 0) scalar_dims[0..1] else dims;
        var err: CError = .{};
        return termite_mpsgraph_constant_f32(self.ctx, data.ptr, @intCast(data.len), ranked_dims.ptr, @intCast(ranked_dims.len), &err) orelse failCError(&err);
    }

    fn scalar(self: *Lowerer, value_: f32) !?*anyopaque {
        var dims_buf = [_]i64{1};
        var data_buf = [_]f32{value_};
        return self.constant(dims_buf[0..1], data_buf[0..1]);
    }

    fn unary(self: *Lowerer, op: UnaryOp, x: ?*anyopaque) !?*anyopaque {
        var err: CError = .{};
        return termite_mpsgraph_unary(self.ctx, @intFromEnum(op), x, &err) orelse failCError(&err);
    }

    fn binary(self: *Lowerer, op: BinaryOp, lhs: ?*anyopaque, rhs: ?*anyopaque) !?*anyopaque {
        var err: CError = .{};
        return termite_mpsgraph_binary(self.ctx, @intFromEnum(op), lhs, rhs, &err) orelse failCError(&err);
    }

    fn where(self: *Lowerer, cond: ?*anyopaque, on_true: ?*anyopaque, on_false: ?*anyopaque) !?*anyopaque {
        var err: CError = .{};
        return termite_mpsgraph_where(self.ctx, cond, on_true, on_false, &err) orelse failCError(&err);
    }

    fn matmul(self: *Lowerer, lhs: ?*anyopaque, rhs: ?*anyopaque) !?*anyopaque {
        var err: CError = .{};
        return termite_mpsgraph_matmul(self.ctx, lhs, rhs, &err) orelse failCError(&err);
    }

    fn reduce(self: *Lowerer, op: ReduceOp, x: ?*anyopaque, axes_u8: []const u8, output_shape: Shape) !?*anyopaque {
        var axes_buf: [8]c_int = undefined;
        for (axes_u8, 0..) |axis, i| axes_buf[i] = @intCast(axis);
        var err: CError = .{};
        const reduced = termite_mpsgraph_reduce(self.ctx, @intFromEnum(op), x, &axes_buf, @intCast(axes_u8.len), &err) orelse return failCError(&err);
        return self.reshape(reduced, output_shape);
    }

    fn reshape(self: *Lowerer, x: ?*anyopaque, shape: Shape) !?*anyopaque {
        var shape_buf: [8]i64 = undefined;
        const dims = shapeDims(shape, &shape_buf);
        var err: CError = .{};
        return termite_mpsgraph_reshape(self.ctx, x, dims.ptr, @intCast(dims.len), &err) orelse failCError(&err);
    }

    fn transpose(self: *Lowerer, x: ?*anyopaque, attrs: ml.graph.node.TransposeAttrs, input_rank: u8) !?*anyopaque {
        var perm_u8: [8]u8 = undefined;
        const perm = transpose_utils.effectivePerm(attrs, input_rank, &perm_u8);
        var perm_i32: [8]c_int = undefined;
        for (perm, 0..) |axis, i| perm_i32[i] = @intCast(axis);
        var err: CError = .{};
        return termite_mpsgraph_transpose(self.ctx, x, &perm_i32, @intCast(perm.len), &err) orelse failCError(&err);
    }

    fn transposeExplicit(self: *Lowerer, x: ?*anyopaque, perm: []const c_int) !?*anyopaque {
        var err: CError = .{};
        return termite_mpsgraph_transpose(self.ctx, x, perm.ptr, @intCast(perm.len), &err) orelse failCError(&err);
    }

    fn broadcastTo(self: *Lowerer, x: ?*anyopaque, dims: []const i64) !?*anyopaque {
        var err: CError = .{};
        return termite_mpsgraph_broadcast(self.ctx, x, dims.ptr, @intCast(dims.len), &err) orelse failCError(&err);
    }

    fn broadcastInDim(self: *Lowerer, x: ?*anyopaque, input_shape: Shape, attrs: ml.graph.node.BroadcastAttrs) !?*anyopaque {
        var reshape_dims: [8]i64 = .{1} ** 8;
        const target_rank = attrs.target_shape.rank();
        for (0..attrs.num_axes) |input_axis| {
            const target_axis = attrs.broadcast_axes[input_axis];
            reshape_dims[target_axis] = input_shape.dim(@intCast(input_axis));
        }
        var reshaped = x;
        if (input_shape.rank() != target_rank or !std.mem.eql(i64, input_shape.dims[0..input_shape.rank()], reshape_dims[0..target_rank])) {
            reshaped = try self.reshapeToDims(x, reshape_dims[0..target_rank]);
        }
        var target_dims: [8]i64 = undefined;
        const dims = shapeDims(attrs.target_shape, &target_dims);
        return self.broadcastTo(reshaped, dims);
    }

    fn reshapeToDims(self: *Lowerer, x: ?*anyopaque, dims: []const i64) !?*anyopaque {
        var err: CError = .{};
        return termite_mpsgraph_reshape(self.ctx, x, dims.ptr, @intCast(dims.len), &err) orelse failCError(&err);
    }

    fn slice(self: *Lowerer, x: ?*anyopaque, input_shape: Shape, attrs: ml.graph.node.SliceAttrs) !?*anyopaque {
        const rank = input_shape.rank();
        var starts: [8]i64 = .{0} ** 8;
        var ends: [8]i64 = .{0} ** 8;
        var strides: [8]i64 = .{1} ** 8;
        for (0..rank) |axis| {
            ends[axis] = input_shape.dim(@intCast(axis));
        }
        for (0..attrs.num_axes) |axis| {
            starts[axis] = attrs.starts[axis];
            ends[axis] = attrs.limits[axis];
            strides[axis] = attrs.strides[axis];
        }
        var err: CError = .{};
        return termite_mpsgraph_slice(self.ctx, x, &starts, &ends, &strides, @intCast(rank), &err) orelse failCError(&err);
    }

    fn sliceLast(self: *Lowerer, x: ?*anyopaque, shape: Shape, start: i64, end: i64) !?*anyopaque {
        var attrs = ml.graph.node.SliceAttrs{};
        attrs.num_axes = shape.rank();
        for (0..shape.rank()) |axis| {
            attrs.starts[axis] = 0;
            attrs.limits[axis] = shape.dim(@intCast(axis));
            attrs.strides[axis] = 1;
        }
        const last = shape.rank() - 1;
        attrs.starts[last] = start;
        attrs.limits[last] = end;
        return self.slice(x, shape, attrs);
    }

    fn sliceLastStrided(self: *Lowerer, x: ?*anyopaque, shape: Shape, start: i64, end: i64, stride: i64) !?*anyopaque {
        var attrs = ml.graph.node.SliceAttrs{};
        attrs.num_axes = shape.rank();
        for (0..shape.rank()) |axis| {
            attrs.starts[axis] = 0;
            attrs.limits[axis] = shape.dim(@intCast(axis));
            attrs.strides[axis] = 1;
        }
        const last = shape.rank() - 1;
        attrs.starts[last] = start;
        attrs.limits[last] = end;
        attrs.strides[last] = stride;
        return self.slice(x, shape, attrs);
    }

    fn concat(self: *Lowerer, lhs: ?*anyopaque, rhs: ?*anyopaque, axis: u8) !?*anyopaque {
        var handles = [_]?*anyopaque{ lhs, rhs };
        var err: CError = .{};
        return termite_mpsgraph_concat(self.ctx, &handles, 2, @intCast(axis), &err) orelse failCError(&err);
    }

    fn softmax(self: *Lowerer, x: ?*anyopaque, input_shape: Shape) !?*anyopaque {
        var err: CError = .{};
        return termite_mpsgraph_softmax(self.ctx, x, @intCast(input_shape.rank() - 1), &err) orelse failCError(&err);
    }

    fn dotGeneral(
        self: *Lowerer,
        lhs: ?*anyopaque,
        lhs_shape: Shape,
        rhs: ?*anyopaque,
        rhs_shape: Shape,
        attrs: ml.graph.node.DotGeneralAttrs,
    ) !?*anyopaque {
        if (attrs.num_contracting != 1 or attrs.num_batch > 1) return error.MpsGraphUnsupportedDotGeneral;
        if (attrs.num_batch == 0 and lhs_shape.rank() == 2 and rhs_shape.rank() == 2) {
            var lhs_m = lhs;
            var rhs_m = rhs;
            if (attrs.lhs_contracting[0] == 0) lhs_m = try self.transposeExplicit(lhs_m, &.{ 1, 0 });
            if (attrs.rhs_contracting[0] == 1) rhs_m = try self.transposeExplicit(rhs_m, &.{ 1, 0 });
            if (attrs.lhs_contracting[0] > 1 or attrs.rhs_contracting[0] > 1) return error.MpsGraphUnsupportedDotGeneral;
            return self.matmul(lhs_m, rhs_m);
        }
        if (attrs.num_batch == 1 and lhs_shape.rank() == 3 and rhs_shape.rank() == 3) {
            if (attrs.lhs_batch[0] != 0 or attrs.rhs_batch[0] != 0) return error.MpsGraphUnsupportedDotGeneral;
            var lhs_m = lhs;
            var rhs_m = rhs;
            if (attrs.lhs_contracting[0] == 1) lhs_m = try self.transposeExplicit(lhs_m, &.{ 0, 2, 1 });
            if (attrs.rhs_contracting[0] == 2) rhs_m = try self.transposeExplicit(rhs_m, &.{ 0, 2, 1 });
            if (attrs.lhs_contracting[0] > 2 or attrs.rhs_contracting[0] > 2) return error.MpsGraphUnsupportedDotGeneral;
            return self.matmul(lhs_m, rhs_m);
        }
        return error.MpsGraphUnsupportedDotGeneral;
    }

    fn linearNoBias(self: *Lowerer, input_: ?*anyopaque, weight: ?*anyopaque, attrs: ml.graph.node.LinearAttrs) !?*anyopaque {
        _ = attrs;
        const weight_t = try self.transposeExplicit(weight, &.{ 1, 0 });
        return self.matmul(input_, weight_t);
    }

    fn linear(self: *Lowerer, input_: ?*anyopaque, weight: ?*anyopaque, bias: ?*anyopaque, attrs: ml.graph.node.LinearAttrs) !?*anyopaque {
        const no_bias = try self.linearNoBias(input_, weight, attrs);
        return self.binary(.add, no_bias, bias);
    }

    fn layerNorm(self: *Lowerer, input_: ?*anyopaque, weight: ?*anyopaque, bias: ?*anyopaque, input_shape: Shape, eps: f32) !?*anyopaque {
        if (input_shape.rank() < 1) return error.MpsGraphUnsupportedShape;
        const last_axis: u8 = input_shape.rank() - 1;
        const mean_shape = shapeWithDim(input_shape, last_axis, 1);
        const mean = try self.reduce(.mean, input_, &.{last_axis}, mean_shape);
        const centered = try self.binary(.sub, input_, mean);
        const sq = try self.binary(.mul, centered, centered);
        const variance = try self.reduce(.mean, sq, &.{last_axis}, mean_shape);
        const eps_t = try self.scalar(eps);
        const inv = try self.unary(.rsqrt, try self.binary(.add, variance, eps_t));
        const norm = try self.binary(.mul, centered, inv);
        const scaled = try self.binary(.mul, norm, weight);
        return self.binary(.add, scaled, bias);
    }

    fn gelu(self: *Lowerer, x: ?*anyopaque) !?*anyopaque {
        const half = try self.scalar(0.5);
        const one = try self.scalar(1.0);
        const inv_sqrt2 = try self.scalar(0.7071067811865475);
        const erf_arg = try self.binary(.mul, x, inv_sqrt2);
        const erf_val = try self.unary(.erf, erf_arg);
        const one_plus = try self.binary(.add, one, erf_val);
        const scaled = try self.binary(.mul, x, half);
        return self.binary(.mul, scaled, one_plus);
    }

    fn rope(self: *Lowerer, x: ?*anyopaque, cos: ?*anyopaque, sin: ?*anyopaque, input_shape: Shape, attrs: ml.graph.node.RopeAttrs) !?*anyopaque {
        if (input_shape.rank() < 2) return error.MpsGraphUnsupportedRope;
        const head_dim: i64 = @intCast(attrs.head_dim);
        const rope_dim: i64 = if (attrs.rope_dim > 0) @intCast(attrs.rope_dim) else head_dim;
        if (rope_dim <= 0 or @rem(rope_dim, 2) != 0 or rope_dim > head_dim) return error.MpsGraphUnsupportedRope;
        const half = @divExact(rope_dim, 2);
        const last_axis = input_shape.rank() - 1;
        if (input_shape.dim(last_axis) != head_dim) return error.MpsGraphUnsupportedRope;

        var half_shape = input_shape;
        half_shape.dims[last_axis] = half;

        const cos_shape = Shape.init(.f32, &.{ attrs.seq_len, attrs.head_dim });
        const cos_half = try self.sliceLast(cos, cos_shape, 0, half);
        const sin_half = try self.sliceLast(sin, cos_shape, 0, half);
        var half_dims: [8]i64 = undefined;
        const half_target = shapeDims(half_shape, &half_dims);
        const bcos = try self.broadcastTo(cos_half, half_target);
        const bsin = try self.broadcastTo(sin_half, half_target);

        const x0 = if (attrs.consecutive_pairs)
            try self.sliceLastStrided(x, input_shape, 0, rope_dim, 2)
        else
            try self.sliceLast(x, input_shape, 0, half);
        const x1 = if (attrs.consecutive_pairs)
            try self.sliceLastStrided(x, input_shape, 1, rope_dim, 2)
        else
            try self.sliceLast(x, input_shape, half, rope_dim);

        const out0 = try self.binary(.sub, try self.binary(.mul, x0, bcos), try self.binary(.mul, x1, bsin));
        const out1 = try self.binary(.add, try self.binary(.mul, x0, bsin), try self.binary(.mul, x1, bcos));
        var rotated = try self.concat(out0, out1, last_axis);
        if (attrs.consecutive_pairs) {
            const rank: usize = input_shape.rank();
            if (rank >= 8) return error.MpsGraphUnsupportedRope;
            var pair_dims: [8]i64 = undefined;
            for (0..(rank - 1)) |axis| {
                pair_dims[axis] = input_shape.dim(@intCast(axis));
            }
            pair_dims[rank - 1] = 2;
            pair_dims[rank] = half;
            const paired = try self.reshapeToDims(rotated, pair_dims[0 .. rank + 1]);

            var perm: [8]c_int = undefined;
            for (0..(rank - 1)) |axis| {
                perm[axis] = @intCast(axis);
            }
            perm[rank - 1] = @intCast(rank);
            perm[rank] = @intCast(rank - 1);
            const transposed = try self.transposeExplicit(paired, perm[0 .. rank + 1]);

            var rotated_dims: [8]i64 = undefined;
            for (0..rank) |axis| {
                rotated_dims[axis] = input_shape.dim(@intCast(axis));
            }
            rotated_dims[last_axis] = rope_dim;
            rotated = try self.reshapeToDims(transposed, rotated_dims[0..rank]);
        }
        if (rope_dim < head_dim) {
            const passthrough = try self.sliceLast(x, input_shape, rope_dim, head_dim);
            rotated = try self.concat(rotated, passthrough, last_axis);
        }
        return rotated;
    }
};

fn findHostInput(host_inputs: ?[]const HostInput, node_id: NodeId) ?[]const f32 {
    const inputs = host_inputs orelse return null;
    for (inputs) |input| {
        if (input.node_id == node_id) return input.data;
    }
    return null;
}

fn findRuntimeInput(runtime_inputs: ?[]const RuntimeInput, node_id: NodeId) ?CT {
    const inputs = runtime_inputs orelse return null;
    for (inputs) |input| {
        if (input.node_id == node_id) return input.value;
    }
    return null;
}

fn appendShape(list: *std.ArrayListUnmanaged(i64), allocator: std.mem.Allocator, shape: Shape) !void {
    for (0..shape.rank()) |axis| {
        try list.append(allocator, shape.dim(@intCast(axis)));
    }
}

fn shapeDims(shape: Shape, buf: *[8]i64) []const i64 {
    for (0..shape.rank()) |axis| {
        buf[axis] = shape.dim(@intCast(axis));
    }
    return buf[0..shape.rank()];
}

fn shapeWithDim(shape: Shape, axis: u8, dim: i64) Shape {
    var out = shape;
    out.dims[axis] = dim;
    return out;
}

fn shapeElementCount(shape: Shape) !usize {
    const elems = shape.numElements() orelse return error.MpsGraphDynamicShape;
    if (elems < 0) return error.MpsGraphDynamicShape;
    return @intCast(elems);
}

fn failCError(err: *CError) error{MpsGraphFailure} {
    if (err.message) |message| {
        std.log.warn("mpsgraph error code={} message={s}", .{ err.code, std.mem.span(message) });
        termite_mpsgraph_error_message_free(message);
        err.message = null;
    } else {
        std.log.warn("mpsgraph error code={} message=<none>", .{err.code});
    }
    return error.MpsGraphFailure;
}

/// Conservative support summary for the MPSGraph training-graph lowerer.
///
/// This is intentionally separate from the primitive Metal partition support:
/// MPSGraph wants whole fixed-shape segment graphs, so a graph is useful only
/// when every reachable op can be lowered into MPSGraph.
pub const SupportSummary = struct {
    nodes: usize = 0,
    supported: usize = 0,
    unsupported: usize = 0,
    first_unsupported_node: ?NodeId = null,
    first_unsupported_op: ?[]const u8 = null,

    pub fn allSupported(self: SupportSummary) bool {
        return self.unsupported == 0;
    }
};

pub fn supportSummary(graph: *const Graph) SupportSummary {
    var summary = SupportSummary{ .nodes = graph.nodeCount() };
    for (0..graph.nodeCount()) |raw_id| {
        const node_id: NodeId = @intCast(raw_id);
        const node = graph.node(node_id);
        if (opSupported(node.op)) {
            summary.supported += 1;
        } else {
            summary.unsupported += 1;
            if (summary.first_unsupported_node == null) {
                summary.first_unsupported_node = node_id;
                summary.first_unsupported_op = @tagName(std.meta.activeTag(node.op));
            }
        }
    }
    return summary;
}

pub fn opSupported(op: OpCode) bool {
    return switch (op) {
        .parameter,
        .constant,
        .neg,
        .sqrt,
        .rsqrt,
        .exp,
        .erf,
        .add,
        .mul,
        .sub,
        .div,
        .less_than,
        .where_select,
        .reduce_sum,
        .reduce_max,
        .reduce_mean,
        .reshape,
        .transpose,
        .broadcast_in_dim,
        .slice,
        .concat_prim,
        .dot_general,
        .fused_linear,
        .fused_linear_no_bias,
        .fused_layer_norm,
        .fused_gelu,
        .fused_softmax,
        .fused_rope,
        .fused_elem_multiply,
        => true,

        // Keep everything else fail-closed until there is a concrete lowering
        // and at least a smoke/parity check for that op family.
        else => false,
    };
}

test "mpsgraph support classifier accepts current segment vjp op surface" {
    const supported_ops = [_]OpCode{
        .{ .parameter = .{ .name_offset = 0, .name_len = 1 } },
        .{ .constant = .{ .data_offset = 0, .data_len = 4 } },
        .{ .add = {} },
        .{ .mul = {} },
        .{ .less_than = {} },
        .{ .where_select = {} },
        .{ .concat_prim = .{ .axis = 0 } },
        .{ .fused_gelu = {} },
        .{ .fused_elem_multiply = {} },
    };
    for (supported_ops) |op| {
        try std.testing.expect(opSupported(op));
    }
    try std.testing.expect(!opSupported(.{ .gather = .{ .axis = 0 } }));
}
