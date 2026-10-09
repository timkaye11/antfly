// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Official-weight CUDA smoke for the resident EG2 LoRA encoder.

const std = @import("std");
const build_options = @import("build_options");
const inference = @import("inference_internal");

const CudaCompute = if (build_options.enable_cuda) inference.native_compute.cuda.CudaCompute else void;
const CapabilityProfile = if (build_options.enable_cuda) inference.native_compute.cuda.CapabilityProfile else void;
const training = inference.finetune.embedding_gemma2_training;
const planning = inference.finetune.embedding_gemma2_cuda;
const session_mod = inference.finetune.embedding_gemma2_cuda_session;
const encoder_mod = inference.finetune.embedding_gemma2_cuda_encoder;
const projection_mod = inference.finetune.embedding_gemma2_cuda_projection;
const runner = inference.finetune.embedding_gemma2_runner;

pub fn main(init: std.process.Init) !void {
    if (comptime !build_options.enable_cuda) return error.CudaRequired;
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();
    const model_dir = args.next() orelse return error.UsageExpectedModelDirAndReportPath;
    const report_path = args.next() orelse return error.UsageExpectedModelDirAndReportPath;
    var max_shape = false;
    var projection_vjp = false;
    var projection_rows: usize = 17;
    var projection_index: usize = 0;
    var backward_trace = false;
    var input_path: ?[]const u8 = null;
    while (args.next()) |flag| {
        if (std.mem.eql(u8, flag, "--max-shape")) {
            max_shape = true;
        } else if (std.mem.eql(u8, flag, "--projection-vjp")) {
            projection_vjp = true;
        } else if (std.mem.eql(u8, flag, "--projection-rows")) {
            projection_rows = try std.fmt.parseUnsigned(usize, args.next() orelse return error.UsageExpectedModelDirAndReportPath, 10);
        } else if (std.mem.eql(u8, flag, "--projection-index")) {
            projection_index = try std.fmt.parseUnsigned(usize, args.next() orelse return error.UsageExpectedModelDirAndReportPath, 10);
        } else if (std.mem.eql(u8, flag, "--backward-trace")) {
            backward_trace = true;
        } else if (std.mem.eql(u8, flag, "--input-json")) {
            input_path = args.next() orelse return error.UsageExpectedModelDirAndReportPath;
        } else return error.UsageExpectedModelDirAndReportPath;
    }
    const DiagnosticInput = struct { token_ids: []const i64, attention_mask: []const i64, batch: u32, sequence: u32, selected_layer: u32 = 0 };
    var diagnostic: ?std.json.Parsed(DiagnosticInput) = null;
    defer if (diagnostic) |*value| value.deinit();
    if (input_path) |path| {
        const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, path, allocator, .limited(1024 * 1024));
        defer allocator.free(bytes);
        diagnostic = try std.json.parseFromSlice(DiagnosticInput, allocator, bytes, .{ .allocate = .alloc_always });
        const value = diagnostic.?.value;
        if (value.batch == 0 or value.batch > 2 or value.sequence == 0 or value.sequence > 512 or value.selected_layer >= 24 or
            value.token_ids.len != @as(usize, value.batch) * value.sequence or value.attention_mask.len != value.token_ids.len) return error.InvalidDiagnosticInput;
    }
    try runner.verifyBase(allocator, init.io, model_dir);
    const config = try runner.loadModelConfig(allocator, init.io, model_dir);

    var compute = try CudaCompute.initWithDeviceMemoryLimit(allocator, 20 * 1024 * 1024 * 1024);
    defer compute.deinit();
    compute.capability_profile = CapabilityProfile.embedding_gemma2;
    compute.resident_training_cache = true;
    try runner.loadTextWeights(allocator, &compute, model_dir);
    const loaded_bytes = compute.ctx.device_allocations.snapshot().live;
    if (loaded_bytes == 0) return error.EmbeddingGemma2WeightsMissing;

    const plan = training.TrainingPlan{};
    const limits = planning.Limits{ .resident_base_bytes = loaded_bytes };
    var state = try session_mod.ResidentAdapterState.init(&compute, config, plan, 2, if (max_shape) 512 else if (diagnostic) |value| value.value.sequence else 9, limits, 0x454732);
    defer state.deinit();
    if (projection_vjp) return projectionVjpProbe(init, &compute, &state, report_path, projection_rows, projection_index);
    var encoder = try encoder_mod.Encoder.init(allocator, &compute, config, plan, &state, limits);
    defer encoder.deinit();
    var forward_trace = TraceCapture{ .allocator = allocator, .compute = &compute };
    defer forward_trace.deinit();
    if (diagnostic) |parsed_input| {
        const value = parsed_input.value;
        forward_trace.selected_layer = value.selected_layer;
        var output = try encoder.encodeWithTrace(.{ .token_ids = value.token_ids, .attention_mask = value.attention_mask, .batch = value.batch, .sequence = value.sequence }, forward_trace.sink(), null);
        defer encoder.releaseOutput(&output);
        const vectors = try allocator.alloc(f32, @as(usize, value.batch) * 768);
        defer allocator.free(vectors);
        try output.buffer.copyToHost(compute.embeddingGemma2TrainingContext(), std.mem.sliceAsBytes(vectors));
        try compute.synchronizeEmbeddingGemma2Training();
        try writeF32Artifact(init, report_path, ".embeddings.f32", vectors);
        try writeArtifact(init, report_path, ".forward-trace.bf16", std.mem.sliceAsBytes(forward_trace.values.items));
        var backward_elements: usize = 0;
        if (backward_trace) {
            const d_embedding_host = try allocator.alloc(f32, @as(usize, value.batch) * 768);
            defer allocator.free(d_embedding_host);
            for (d_embedding_host, 0..) |*gradient, index| gradient.* = @as(f32, @floatFromInt(@as(i32, @intCast(index % 37)) - 18)) * 0.0001;
            var d_embedding = try compute.allocEmbeddingGemma2TrainingBuffer(d_embedding_host.len * @sizeOf(f32));
            defer compute.releaseEmbeddingGemma2TrainingBuffer(&d_embedding);
            try d_embedding.copyFromHost(compute.embeddingGemma2TrainingContext(), std.mem.sliceAsBytes(d_embedding_host));
            var backward_capture = BackwardTraceCapture{ .allocator = allocator, .compute = &compute, .selected_layer = value.selected_layer };
            defer backward_capture.deinit();
            try encoder.replayBackwardWithTrace(.{ .token_ids = value.token_ids, .attention_mask = value.attention_mask, .batch = value.batch, .sequence = value.sequence }, d_embedding, backward_capture.sink(), null);
            backward_elements = backward_capture.values.items.len;
            try writeF32Artifact(init, report_path, ".d-embedding.f32", d_embedding_host);
            try writeF32Artifact(init, report_path, ".backward-trace.f32", backward_capture.values.items);
            const gradient_count: usize = @intCast(state.layout.total_parameter_elements);
            const gradients = try allocator.alloc(f32, gradient_count);
            defer allocator.free(gradients);
            try state.current_grad.copyToHost(compute.embeddingGemma2TrainingContext(), std.mem.sliceAsBytes(gradients));
            try compute.synchronizeEmbeddingGemma2Training();
            try writeF32Artifact(init, report_path, ".gradients.f32", gradients);
        }
        const file = try std.Io.Dir.cwd().createFile(init.io, report_path, .{ .truncate = true });
        defer file.close(init.io);
        var bytes: [4096]u8 = undefined;
        var writer = file.writer(init.io, &bytes);
        try std.json.Stringify.value(.{ .diagnostic = true, .batch = value.batch, .sequence = value.sequence, .selected_layer = value.selected_layer, .bf16_elements = forward_trace.values.items.len, .backward_f32_elements = backward_elements }, .{}, &writer.interface);
        try writer.end();
        return;
    }
    const batches = [_][18]i64{
        .{ 2, 100, 1, 0, 0, 0, 0, 0, 0, 2, 200, 201, 202, 203, 204, 205, 1, 0 },
        .{ 2, 300, 301, 302, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    };
    const masks = [_][18]i64{
        .{ 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0 },
        .{ 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
    };
    const batch_sizes = [_]usize{ 2, 1 };
    var embeddings = try allocator.alloc(f32, 3 * 768);
    defer allocator.free(embeddings);
    const cotangent = try allocator.alloc(f32, 2 * 768);
    defer allocator.free(cotangent);
    for (cotangent, 0..) |*value, index| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(index % 17)) - 8)) * 0.0001;
    var embedding_row: usize = 0;
    for (batches, masks, batch_sizes) |ids, mask, batch| {
        const elements = batch * 9;
        const input = encoder_mod.PaddedInput{ .token_ids = ids[0..elements], .attention_mask = mask[0..elements], .batch = @intCast(batch), .sequence = 9 };
        var output = if (embedding_row == 0)
            try encoder.encodeWithTrace(input, forward_trace.sink(), null)
        else
            try encoder.encode(input, null);
        defer encoder.releaseOutput(&output);
        try output.buffer.copyToHost(compute.embeddingGemma2TrainingContext(), std.mem.sliceAsBytes(embeddings[embedding_row * 768 ..][0 .. batch * 768]));
        try compute.synchronizeEmbeddingGemma2Training();
        var d_embedding = try compute.allocEmbeddingGemma2TrainingBuffer(batch * 768 * @sizeOf(f32));
        defer compute.releaseEmbeddingGemma2TrainingBuffer(&d_embedding);
        try d_embedding.copyFromHost(compute.embeddingGemma2TrainingContext(), std.mem.sliceAsBytes(cotangent[0 .. batch * 768]));
        try encoder.replayBackward(input, d_embedding, null);
        embedding_row += batch;
    }
    if (max_shape) {
        const ids = try allocator.alloc(i64, 2 * 512);
        defer allocator.free(ids);
        const mask = try allocator.alloc(i64, 2 * 512);
        defer allocator.free(mask);
        for (ids, 0..) |*token, index| token.* = if (index % 512 == 0) 2 else if (index % 512 == 511) 1 else @intCast(100 + index % 300);
        @memset(mask, 1);
        mask[511] = 0;
        const input = encoder_mod.PaddedInput{ .token_ids = ids, .attention_mask = mask, .batch = 2, .sequence = 512 };
        var output = try encoder.encode(input, null);
        defer encoder.releaseOutput(&output);
        var d_embedding = try compute.allocEmbeddingGemma2TrainingBuffer(2 * 768 * @sizeOf(f32));
        defer compute.releaseEmbeddingGemma2TrainingBuffer(&d_embedding);
        try d_embedding.copyFromHost(compute.embeddingGemma2TrainingContext(), std.mem.sliceAsBytes(cotangent));
        try encoder.replayBackward(input, d_embedding, null);
    }
    const gradient_count: usize = @intCast(state.layout.total_parameter_elements);
    const gradients = try allocator.alloc(f32, gradient_count);
    defer allocator.free(gradients);
    try state.current_grad.copyToHost(compute.embeddingGemma2TrainingContext(), std.mem.sliceAsBytes(gradients));
    try compute.synchronizeEmbeddingGemma2Training();
    try writeF32Artifact(init, report_path, ".embeddings.f32", embeddings);
    try writeF32Artifact(init, report_path, ".gradients.f32", gradients);
    try writeArtifact(init, report_path, ".forward-trace.bf16", std.mem.sliceAsBytes(forward_trace.values.items));
    var finite_gradients: usize = 0;
    var nonzero_gradients: usize = 0;
    var gradient_l2: f64 = 0;
    for (gradients) |value| {
        if (std.math.isFinite(value)) finite_gradients += 1;
        if (value != 0) nonzero_gradients += 1;
        gradient_l2 += @as(f64, value) * value;
    }
    var min_embedding_norm: f64 = std.math.inf(f64);
    var max_embedding_norm: f64 = 0;
    for (0..3) |row| {
        var norm: f64 = 0;
        for (embeddings[row * 768 ..][0..768]) |value| norm += @as(f64, value) * value;
        norm = @sqrt(norm);
        min_embedding_norm = @min(min_embedding_norm, norm);
        max_embedding_norm = @max(max_embedding_norm, norm);
    }
    const report = try std.Io.Dir.cwd().createFile(init.io, report_path, .{ .truncate = true });
    defer report.close(init.io);
    var buffer: [4096]u8 = undefined;
    var writer = report.writer(init.io, &buffer);
    try writer.interface.print("{{\"verified_base\":true,\"resident_base_bytes\":{d},\"embedding_rows\":3,\"embedding_width\":768,\"padding_lengths\":[3,8,5],\"max_shape_b2_s512\":{},\"forward_trace\":{{\"rows\":18,\"bf16_elements\":{d},\"order\":\"token_embedding,layer0_substages,layer_output[0..24),final_norm,embedding_projection\"}},\"min_embedding_norm\":{d},\"max_embedding_norm\":{d},\"adapter_count\":169,\"gradient_count\":{d},\"finite_gradients\":{d},\"nonzero_gradients\":{d},\"gradient_l2\":{d}}}\n", .{ loaded_bytes, max_shape, forward_trace.values.items.len, min_embedding_norm, max_embedding_norm, gradient_count, finite_gradients, nonzero_gradients, @sqrt(gradient_l2) });
    try writer.end();
    if (finite_gradients != gradient_count or nonzero_gradients == 0) return error.InvalidEmbeddingGemma2Gradient;
}

fn writeF32Artifact(init: std.process.Init, report_path: []const u8, suffix: []const u8, values: []const f32) !void {
    return writeArtifact(init, report_path, suffix, std.mem.sliceAsBytes(values));
}

fn writeArtifact(init: std.process.Init, report_path: []const u8, suffix: []const u8, bytes: []const u8) !void {
    const path = try std.mem.concat(init.gpa, u8, &.{ report_path, suffix });
    defer init.gpa.free(path);
    const file = try std.Io.Dir.cwd().createFile(init.io, path, .{ .truncate = true });
    defer file.close(init.io);
    var buffer: [64 * 1024]u8 = undefined;
    var writer = file.writer(init.io, &buffer);
    try writer.interface.writeAll(bytes);
    try writer.end();
}

fn projectionVjpProbe(init: std.process.Init, compute: *CudaCompute, state: *session_mod.ResidentAdapterState, report_path: []const u8, rows: usize, adapter_index: usize) !void {
    if (rows == 0 or rows > 1024 or adapter_index >= state.layout.descriptors.len) return error.InvalidProjectionProbe;
    const descriptor = state.layout.descriptors[adapter_index];
    const input_dim: usize = descriptor.input_dim;
    const output_dim: usize = descriptor.output_dim;
    const rank: usize = descriptor.rank;
    var base_name_buffer: [128]u8 = undefined;
    const base_weight_name = try descriptor.writeBaseName(&base_name_buffer);
    const input_elements = rows * input_dim;
    const output_elements = rows * output_dim;
    const low_elements = rows * rank;
    const adapter = try state.adapterSlice(adapter_index);
    const gradients = try state.gradientSlice(adapter_index, false);
    const host_b = try init.gpa.alloc(f32, output_dim * rank);
    defer init.gpa.free(host_b);
    for (host_b, 0..) |*value, index| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(index % 23)) - 11)) * 0.0001;
    try adapter.b_master.copyFromHost(compute.embeddingGemma2TrainingContext(), std.mem.sliceAsBytes(host_b));
    try compute.convertEmbeddingGemma2TrainingF32ToBf16(adapter.b_bf16, adapter.b_master, host_b.len);

    const host_input = try init.gpa.alloc(f32, input_elements);
    defer init.gpa.free(host_input);
    for (host_input, 0..) |*value, index| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(index % 29)) - 14)) * 0.003;
    const host_dy = try init.gpa.alloc(f32, output_elements);
    defer init.gpa.free(host_dy);
    for (host_dy, 0..) |*value, index| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(index % 31)) - 15)) * 0.0002;

    var input_f32 = try compute.allocEmbeddingGemma2TrainingBuffer(input_elements * @sizeOf(f32));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&input_f32);
    var input_bf16 = try compute.allocEmbeddingGemma2TrainingBuffer(input_elements * @sizeOf(u16));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&input_bf16);
    var output_bf16 = try compute.allocEmbeddingGemma2TrainingBuffer(output_elements * @sizeOf(u16));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&output_bf16);
    var low_bf16 = try compute.allocEmbeddingGemma2TrainingBuffer(low_elements * @sizeOf(u16));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&low_bf16);
    var delta_bf16 = try compute.allocEmbeddingGemma2TrainingBuffer(output_elements * @sizeOf(u16));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&delta_bf16);
    var dy_f32 = try compute.allocEmbeddingGemma2TrainingBuffer(output_elements * @sizeOf(f32));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&dy_f32);
    var dx_f32 = try compute.allocEmbeddingGemma2TrainingBuffer(input_elements * @sizeOf(f32));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&dx_f32);
    var dy_scaled_f32 = try compute.allocEmbeddingGemma2TrainingBuffer(output_elements * @sizeOf(f32));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&dy_scaled_f32);
    var dy_bf16 = try compute.allocEmbeddingGemma2TrainingBuffer(output_elements * @sizeOf(u16));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&dy_bf16);
    var original_dy_bf16 = try compute.allocEmbeddingGemma2TrainingBuffer(output_elements * @sizeOf(u16));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&original_dy_bf16);
    var backward_input_f32 = try compute.allocEmbeddingGemma2TrainingBuffer(input_elements * @sizeOf(f32));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&backward_input_f32);
    var low_scratch_f32 = try compute.allocEmbeddingGemma2TrainingBuffer(input_elements * @sizeOf(f32));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&low_scratch_f32);
    var dz_f32 = try compute.allocEmbeddingGemma2TrainingBuffer(low_elements * @sizeOf(f32));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&dz_f32);
    var true_base_dx_bf16 = try compute.allocEmbeddingGemma2TrainingBuffer(input_elements * @sizeOf(u16));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&true_base_dx_bf16);
    var true_dz_bf16 = try compute.allocEmbeddingGemma2TrainingBuffer(low_elements * @sizeOf(u16));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&true_dz_bf16);
    var true_lora_dx_bf16 = try compute.allocEmbeddingGemma2TrainingBuffer(input_elements * @sizeOf(u16));
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&true_lora_dx_bf16);
    var workspace = try compute.allocEmbeddingGemma2TrainingBuffer(64 * 1024 * 1024);
    defer compute.releaseEmbeddingGemma2TrainingBuffer(&workspace);
    try input_f32.copyFromHost(compute.embeddingGemma2TrainingContext(), std.mem.sliceAsBytes(host_input));
    try compute.convertEmbeddingGemma2TrainingF32ToBf16(input_bf16, input_f32, input_elements);
    try dy_f32.copyFromHost(compute.embeddingGemma2TrainingContext(), std.mem.sliceAsBytes(host_dy));
    try compute.convertEmbeddingGemma2TrainingF32ToBf16(original_dy_bf16, dy_f32, output_elements);
    try projection_mod.forward(compute, state, adapter_index, base_weight_name, input_bf16, rows, input_dim, output_dim, 32, .{ .output_bf16 = output_bf16, .low_rank_bf16 = low_bf16, .delta_bf16 = delta_bf16, .workspace = workspace });
    try projection_mod.backward(compute, state, adapter_index, base_weight_name, input_bf16, low_bf16, dy_f32, rows, input_dim, output_dim, 32, .{ .dx_f32 = dx_f32, .dy_scaled_f32 = dy_scaled_f32, .dy_bf16 = dy_bf16, .input_f32 = backward_input_f32, .low_rank_f32 = low_scratch_f32, .dz_f32 = dz_f32, .workspace = workspace });
    const lt = try compute.embeddingGemma2TrainingCublasLt();
    const ctx = compute.embeddingGemma2TrainingContext();
    const base_weight = try compute.embeddingGemma2Bf16Weight(base_weight_name);
    try lt.matmulEmbeddingGemma2Attention(ctx, true_base_dx_bf16, original_dy_bf16, base_weight, workspace, rows, input_dim, output_dim, output_dim, input_dim, input_dim, 1, 0, 0, 0, false, false, false, 1);
    try lt.matmulEmbeddingGemma2Attention(ctx, true_dz_bf16, dy_bf16, adapter.b_bf16, workspace, rows, rank, output_dim, output_dim, rank, rank, 1, 0, 0, 0, false, false, false, 1);
    try lt.matmulEmbeddingGemma2Attention(ctx, true_lora_dx_bf16, true_dz_bf16, adapter.a_bf16, workspace, rows, input_dim, rank, rank, input_dim, input_dim, 1, 0, 0, 0, false, false, false, 1);
    try writeDeviceArtifact(init, compute, report_path, ".input.bf16", input_bf16, input_elements, u16);
    try writeDeviceArtifact(init, compute, report_path, ".output.bf16", output_bf16, output_elements, u16);
    try writeDeviceArtifact(init, compute, report_path, ".low.bf16", low_bf16, low_elements, u16);
    try writeDeviceArtifact(init, compute, report_path, ".dy.f32", dy_f32, output_elements, f32);
    try writeDeviceArtifact(init, compute, report_path, ".dx.f32", dx_f32, input_elements, f32);
    try writeDeviceArtifact(init, compute, report_path, ".true-base-dx.bf16", true_base_dx_bf16, input_elements, u16);
    try writeDeviceArtifact(init, compute, report_path, ".true-dz.bf16", true_dz_bf16, low_elements, u16);
    try writeDeviceArtifact(init, compute, report_path, ".true-lora-dx.bf16", true_lora_dx_bf16, input_elements, u16);
    try writeDeviceArtifact(init, compute, report_path, ".a.f32", adapter.a_master, rank * input_dim, f32);
    try writeDeviceArtifact(init, compute, report_path, ".b.f32", adapter.b_master, output_dim * rank, f32);
    try writeDeviceArtifact(init, compute, report_path, ".da.f32", gradients.a, rank * input_dim, f32);
    try writeDeviceArtifact(init, compute, report_path, ".db.f32", gradients.b, output_dim * rank, f32);
    const file = try std.Io.Dir.cwd().createFile(init.io, report_path, .{ .truncate = true });
    defer file.close(init.io);
    var buffer: [1024]u8 = undefined;
    var writer = file.writer(init.io, &buffer);
    try std.json.Stringify.value(.{ .mode = "projection_vjp", .adapter_index = adapter_index, .base_weight = base_weight_name, .rows = rows, .input_dim = input_dim, .output_dim = output_dim, .rank = rank, .alpha = 32 }, .{}, &writer.interface);
    try writer.end();
}

fn writeDeviceArtifact(init: std.process.Init, compute: *CudaCompute, report_path: []const u8, suffix: []const u8, buffer: anytype, count: usize, comptime T: type) !void {
    const values = try init.gpa.alloc(T, count);
    defer init.gpa.free(values);
    try buffer.copyToHost(compute.embeddingGemma2TrainingContext(), std.mem.sliceAsBytes(values));
    try compute.synchronizeEmbeddingGemma2Training();
    try writeArtifact(init, report_path, suffix, std.mem.sliceAsBytes(values));
}

const TraceCapture = struct {
    allocator: std.mem.Allocator,
    compute: *CudaCompute,
    selected_layer: u32 = 0,
    values: std.ArrayListUnmanaged(u16) = .empty,

    fn sink(self: *TraceCapture) encoder_mod.TraceSink {
        return .{ .context = self, .capture = capture, .selected_layer = self.selected_layer };
    }

    fn capture(context: *anyopaque, _: encoder_mod.TraceStage, _: u32, buffer: encoder_mod.TraceBuffer, batch: u32, sequence: u32, width: u32) !void {
        const self: *TraceCapture = @ptrCast(@alignCast(context));
        const count = try std.math.mul(usize, try std.math.mul(usize, batch, sequence), width);
        const start = self.values.items.len;
        try self.values.resize(self.allocator, start + count);
        try buffer.copyToHost(self.compute.embeddingGemma2TrainingContext(), std.mem.sliceAsBytes(self.values.items[start..][0..count]));
        try self.compute.synchronizeEmbeddingGemma2Training();
    }

    fn deinit(self: *TraceCapture) void {
        self.values.deinit(self.allocator);
        self.* = undefined;
    }
};

const BackwardTraceCapture = struct {
    allocator: std.mem.Allocator,
    compute: *CudaCompute,
    selected_layer: u32,
    values: std.ArrayListUnmanaged(f32) = .empty,

    fn sink(self: *BackwardTraceCapture) encoder_mod.BackwardTraceSink {
        return .{ .context = self, .capture = capture, .selected_layer = self.selected_layer };
    }
    fn capture(context: *anyopaque, _: encoder_mod.BackwardTraceStage, _: u32, buffer: encoder_mod.TraceBuffer, batch: u32, sequence: u32, width: u32) !void {
        const self: *BackwardTraceCapture = @ptrCast(@alignCast(context));
        const count = try std.math.mul(usize, try std.math.mul(usize, batch, sequence), width);
        const start = self.values.items.len;
        try self.values.resize(self.allocator, start + count);
        try buffer.copyToHost(self.compute.embeddingGemma2TrainingContext(), std.mem.sliceAsBytes(self.values.items[start..][0..count]));
        try self.compute.synchronizeEmbeddingGemma2Training();
    }
    fn deinit(self: *BackwardTraceCapture) void {
        self.values.deinit(self.allocator);
        self.* = undefined;
    }
};
