// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Explicit, bounded qualification traces for the resident EG2 trainer.
//! This module is never called by the ordinary training path: activation and
//! gradient D2H copies occur only when the CLI opts into a trace step.

const std = @import("std");
const compat = @import("../io/compat.zig");
const platform = @import("antfly_platform");
const session_mod = @import("embedding_gemma2_cuda_session.zig");
const checkpoint_writer = @import("safetensors_checkpoint.zig");
const DeviceBuffer = @import("../ops/cuda/buffer.zig").DeviceBuffer;

pub const max_trace_steps: u64 = 10;

pub const Trace = struct {
    allocator: std.mem.Allocator,
    safetensors_path: []u8,
    json_path: []u8,
    json: []u8,

    pub fn deinit(self: *Trace) void {
        self.allocator.free(self.safetensors_path);
        self.allocator.free(self.json_path);
        self.allocator.free(self.json);
        self.* = undefined;
    }
};

pub const Pending = struct {
    allocator: std.mem.Allocator,
    state: *session_mod.ResidentAdapterState,
    initial_master: []f32,
    gradient: []f32,
    embeddings: []f32,

    pub fn capture(
        allocator: std.mem.Allocator,
        state: *session_mod.ResidentAdapterState,
        embeddings: DeviceBuffer,
    ) !Pending {
        if (!state.logical_batch_active or state.accumulated_microbatches != 0 or embeddings.ptr == 0 or
            embeddings.len == 0 or embeddings.len % @sizeOf(f32) != 0)
            return error.InvalidEmbeddingGemma2TraceBoundary;
        const count = std.math.cast(usize, state.layout.total_parameter_elements) orelse
            return error.EmbeddingGemma2TraceLimitExceeded;
        const initial_master = try allocator.alloc(f32, count);
        errdefer allocator.free(initial_master);
        const gradient = try allocator.alloc(f32, count);
        errdefer allocator.free(gradient);
        const embedding_values = try allocator.alloc(f32, embeddings.len / @sizeOf(f32));
        errdefer allocator.free(embedding_values);
        const ctx = state.compute.embeddingGemma2TrainingContext();
        try state.master.copyToHost(ctx, std.mem.sliceAsBytes(initial_master));
        try state.current_grad.copyToHost(ctx, std.mem.sliceAsBytes(gradient));
        try embeddings.copyToHost(ctx, std.mem.sliceAsBytes(embedding_values));
        try state.compute.synchronizeEmbeddingGemma2Training();
        try requireFinite(initial_master);
        try requireFinite(gradient);
        try requireFinite(embedding_values);
        return .{
            .allocator = allocator,
            .state = state,
            .initial_master = initial_master,
            .gradient = gradient,
            .embeddings = embedding_values,
        };
    }

    pub fn deinit(self: *Pending) void {
        self.allocator.free(self.initial_master);
        self.allocator.free(self.gradient);
        self.allocator.free(self.embeddings);
        self.* = undefined;
    }

    pub fn finish(
        self: *Pending,
        allocator: std.mem.Allocator,
        output_dir: []const u8,
        step: u64,
        loss: f64,
        learning_rate: f64,
    ) !Trace {
        if (step >= max_trace_steps or !std.math.isFinite(loss) or !std.math.isFinite(learning_rate) or learning_rate <= 0 or
            self.state.logical_batch_active or self.state.accumulated_microbatches != 0 or !std.fs.path.isAbsolute(output_dir))
            return error.InvalidEmbeddingGemma2TraceBoundary;
        const count = std.math.cast(usize, self.state.layout.total_parameter_elements) orelse
            return error.EmbeddingGemma2TraceLimitExceeded;
        const final_master = try allocator.alloc(f32, count);
        defer allocator.free(final_master);
        try self.state.master.copyToHost(self.state.compute.embeddingGemma2TrainingContext(), std.mem.sliceAsBytes(final_master));
        try self.state.compute.synchronizeEmbeddingGemma2Training();
        try requireFinite(final_master);

        const parameter_count = self.state.layout.descriptors.len * 2;
        const tensor_count = parameter_count * 3 + 1;
        const tensors = try allocator.alloc(checkpoint_writer.NamedTensor, tensor_count);
        defer allocator.free(tensors);
        const names = try allocator.alloc([]u8, tensor_count);
        defer {
            for (names) |name| allocator.free(name);
            allocator.free(names);
        }
        for (names) |*name| name.* = &.{};
        const shapes = try allocator.alloc([]usize, tensor_count);
        defer {
            for (shapes) |shape| allocator.free(shape);
            allocator.free(shapes);
        }
        for (shapes) |*shape| shape.* = &.{};

        var index: usize = 0;
        var name_buffer: [256]u8 = undefined;
        for (self.state.layout.descriptors) |descriptor| {
            const base = try descriptor.writeBaseName(&name_buffer);
            if (!std.mem.endsWith(u8, base, ".weight")) return error.InvalidEmbeddingGemma2AdapterLayout;
            const stem = base[0 .. base.len - ".weight".len];
            inline for (.{
                .{ "lora_A.weight", descriptor.a_offset, descriptor.aElements(), [_]usize{ @intCast(descriptor.rank), @intCast(descriptor.input_dim) } },
                .{ "lora_B.weight", descriptor.b_offset, descriptor.bElements(), [_]usize{ @intCast(descriptor.output_dim), @intCast(descriptor.rank) } },
            }) |parameter| {
                const offset: usize = @intCast(parameter[1]);
                const len: usize = @intCast(parameter[2]);
                inline for (.{
                    .{ "initial_adapter", self.initial_master },
                    .{ "gradient", self.gradient },
                    .{ "adapter", final_master },
                }) |value| {
                    const name = try std.fmt.allocPrint(allocator, "{s}.{s}.{s}", .{ value[0], stem, parameter[0] });
                    names[index] = name;
                    const shape = try allocator.dupe(usize, &parameter[3]);
                    shapes[index] = shape;
                    tensors[index] = .{ .name = name, .data = value[1][offset..][0..len], .shape = shape };
                    index += 1;
                }
            }
        }
        const embedding_rows = self.embeddings.len / 768;
        if (embedding_rows == 0 or embedding_rows * 768 != self.embeddings.len) return error.InvalidEmbeddingGemma2TraceBoundary;
        names[index] = try allocator.dupe(u8, "embedding.pooled");
        shapes[index] = try allocator.dupe(usize, &.{ embedding_rows, 768 });
        tensors[index] = .{ .name = names[index], .data = self.embeddings, .shape = shapes[index] };
        index += 1;
        if (index != tensor_count) return error.InvalidEmbeddingGemma2TraceBoundary;

        try compat.cwd().createDirPath(compat.io(), output_dir);
        const tensor_basename = try std.fmt.allocPrint(allocator, "trace-step-{d}.safetensors", .{step});
        defer allocator.free(tensor_basename);
        const tensor_path = try std.fs.path.join(allocator, &.{ output_dir, tensor_basename });
        errdefer allocator.free(tensor_path);
        try checkpoint_writer.save(allocator, tensor_path, tensors);
        const json_basename = try std.fmt.allocPrint(allocator, "trace-step-{d}.json", .{step});
        defer allocator.free(json_basename);
        const json_path = try std.fs.path.join(allocator, &.{ output_dir, json_basename });
        errdefer allocator.free(json_path);
        var json_writer: std.Io.Writer.Allocating = .init(allocator);
        defer json_writer.deinit();
        const writer = &json_writer.writer;
        try writer.print("{{\"step\":{d},\"loss\":{d},\"learning_rate\":{d},\"embedding\":{{\"pooled\":{{\"file\":", .{ step, loss, learning_rate });
        try std.json.Stringify.value(tensor_path, .{}, writer);
        try writer.writeAll(",\"tensor\":\"embedding.pooled\"}},\"gradients\":{");
        var first = true;
        for (0..parameter_count) |parameter_index| {
            // Tensors are emitted initial/gradient/final for each canonical
            // adapter parameter, in that order.
            const gradient_tensor = tensors[parameter_index * 3 + 1];
            const canonical = gradient_tensor.name["gradient.".len..];
            if (!first) try writer.writeByte(',');
            first = false;
            try std.json.Stringify.value(canonical, .{}, writer);
            try writer.writeAll(":{\"file\":");
            try std.json.Stringify.value(tensor_path, .{}, writer);
            try writer.writeAll(",\"tensor\":");
            try std.json.Stringify.value(gradient_tensor.name, .{}, writer);
            try writer.writeByte('}');
        }
        try writer.writeAll("},\"adapter_parameters\":{");
        first = true;
        for (0..parameter_count) |parameter_index| {
            const adapter_tensor = tensors[parameter_index * 3 + 2];
            const canonical = adapter_tensor.name["adapter.".len..];
            if (!first) try writer.writeByte(',');
            first = false;
            try std.json.Stringify.value(canonical, .{}, writer);
            try writer.writeAll(":{\"file\":");
            try std.json.Stringify.value(tensor_path, .{}, writer);
            try writer.writeAll(",\"tensor\":");
            try std.json.Stringify.value(adapter_tensor.name, .{}, writer);
            try writer.writeByte('}');
        }
        try writer.writeAll("}}\n");
        const json = try json_writer.toOwnedSlice();
        errdefer allocator.free(json);
        try writeFile(json_path, json);
        return .{ .allocator = allocator, .safetensors_path = tensor_path, .json_path = json_path, .json = json };
    }
};

fn requireFinite(values: []const f32) !void {
    for (values) |value| if (!std.math.isFinite(value)) return error.NonFiniteEmbeddingGemma2Trace;
}

fn writeFile(path: []const u8, data: []const u8) !void {
    var file = try compat.cwd().createFile(compat.io(), path, .{ .truncate = true });
    defer file.close(compat.io());
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(compat.io(), &buffer);
    try writer.interface.writeAll(data);
    try writer.end();
}

test "EmbeddingGemma2 trace limit is deliberately small" {
    try std.testing.expectEqual(@as(u64, 10), max_trace_steps);
}

pub fn compileProbe() !void {
    if (platform.env.getenvBoolDefault("ANTFLY_TEST_INSTANTIATE_EG2_TRACE", false)) {
        var pending = try Pending.capture(std.testing.allocator, undefined, undefined);
        defer pending.deinit();
        var trace = try pending.finish(std.testing.allocator, "/tmp/eg2-trace-probe", 0, 0, 1e-4);
        trace.deinit();
    }
}

test "EmbeddingGemma2 trace public methods remain instantiated" {
    try compileProbe();
}
