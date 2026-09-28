// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Whole-decoder parity across mixed sliding/global KV cache boundaries.
const std = @import("std");
const build_options = @import("build_options");
const compat = @import("../io/compat.zig");
const platform = @import("antfly_platform");
const checkpoint = @import("../finetune/safetensors_checkpoint.zig");
const gemma_backend = @import("../finetune/gemma4_real_autodiff.zig");
const factory = @import("../architectures/session_factory.zig");
const gpt = @import("../architectures/gpt.zig");
const direct = @import("decoder_gated_runtime.zig");
const metal = @import("metal_runtime.zig");
const metal_compute = @import("../ops/metal_compute.zig");
const kv = @import("../runtime/kv/storage_runtime.zig");

const model_config =
    \\{"model_type":"gemma4_text","dtype":"bfloat16","hidden_size":64,
    \\"num_hidden_layers":2,"num_attention_heads":2,"num_key_value_heads":1,
    \\"head_dim":256,"global_head_dim":256,"intermediate_size":128,"vocab_size":32,
    \\"max_position_embeddings":64,"sliding_window":4,"sliding_window_pattern":2,
    \\"num_kv_shared_layers":0,"hidden_size_per_layer_input":0,
    \\"hidden_activation":"gelu_pytorch_tanh","rms_norm_eps":0.000001,
    \\"rope_theta":10000.0,"tie_word_embeddings":true}
;

fn appendWeight(allocator: std.mem.Allocator, tensors: *std.ArrayList(checkpoint.NamedTensor), name: []const u8, shape: []const usize) !void {
    var count: usize = 1;
    for (shape) |dim| count *= dim;
    const owned_name = try allocator.dupe(u8, name);
    errdefer allocator.free(owned_name);
    const owned_shape = try allocator.dupe(usize, shape);
    errdefer allocator.free(owned_shape);
    const data = try allocator.alloc(f32, count);
    errdefer allocator.free(data);
    const norm = std.mem.indexOf(u8, name, "norm.weight") != null;
    for (data, 0..) |*value, index| {
        // Fixed-point values are exact in BF16, keeping the comparison
        // focused on the decoder and its cache rather than file conversion.
        const bucket: i32 = if (index % 32 == 0) 127 else @as(i32, @intCast((index * 11 + tensors.items.len * 7) % 255)) - 127;
        value.* = if (norm) 1.0 else @as(f32, @floatFromInt(bucket)) / 1024.0;
    }
    try tensors.append(allocator, .{ .name = owned_name, .shape = owned_shape, .data = data });
}

fn writeModel(allocator: std.mem.Allocator, root: []const u8) !void {
    var tensors: std.ArrayList(checkpoint.NamedTensor) = .empty;
    defer {
        for (tensors.items) |tensor| {
            allocator.free(tensor.name);
            allocator.free(tensor.shape);
            allocator.free(tensor.data);
        }
        tensors.deinit(allocator);
    }
    try appendWeight(allocator, &tensors, "model.embed_tokens.weight", &.{ 32, 64 });
    const Spec = struct { suffix: []const u8, shape: []const usize };
    const specs = [_]Spec{
        .{ .suffix = "input_layernorm.weight", .shape = &.{64} },
        .{ .suffix = "self_attn.q_proj.weight", .shape = &.{ 512, 64 } },
        .{ .suffix = "self_attn.k_proj.weight", .shape = &.{ 256, 64 } },
        .{ .suffix = "self_attn.v_proj.weight", .shape = &.{ 256, 64 } },
        .{ .suffix = "self_attn.q_norm.weight", .shape = &.{256} },
        .{ .suffix = "self_attn.k_norm.weight", .shape = &.{256} },
        .{ .suffix = "self_attn.o_proj.weight", .shape = &.{ 64, 512 } },
        .{ .suffix = "post_attention_layernorm.weight", .shape = &.{64} },
        .{ .suffix = "pre_feedforward_layernorm.weight", .shape = &.{64} },
        .{ .suffix = "mlp.gate_proj.weight", .shape = &.{ 128, 64 } },
        .{ .suffix = "mlp.up_proj.weight", .shape = &.{ 128, 64 } },
        .{ .suffix = "mlp.down_proj.weight", .shape = &.{ 64, 128 } },
        .{ .suffix = "post_feedforward_layernorm.weight", .shape = &.{64} },
    };
    for (0..2) |layer| for (specs) |spec| {
        var buf: [128]u8 = undefined;
        try appendWeight(allocator, &tensors, try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}", .{ layer, spec.suffix }), spec.shape);
    };
    try appendWeight(allocator, &tensors, "model.norm.weight", &.{64});
    const path = try std.fs.path.join(allocator, &.{ root, "model.safetensors" });
    defer allocator.free(path);
    // Keep the BF16 representation used by the production dense serving lane.
    var header: std.Io.Writer.Allocating = .init(allocator);
    defer header.deinit();
    const writer = &header.writer;
    try writer.writeByte('{');
    var offset: usize = 0;
    for (tensors.items, 0..) |tensor, index| {
        if (index != 0) try writer.writeByte(',');
        try writer.print("\"{s}\":{{\"dtype\":\"BF16\",\"shape\":[", .{tensor.name});
        for (tensor.shape, 0..) |dim, axis| {
            if (axis != 0) try writer.writeByte(',');
            try writer.print("{d}", .{dim});
        }
        try writer.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + tensor.data.len * 2 });
        offset += tensor.data.len * 2;
    }
    try writer.writeByte('}');
    while (header.written().len % 8 != 0) try writer.writeByte(' ');
    var file = try compat.cwd().createFile(compat.io(), path, .{ .truncate = true });
    defer file.close(compat.io());
    var buffer: [16384]u8 = undefined;
    var output = file.writer(compat.io(), &buffer);
    try output.interface.writeInt(u64, @intCast(header.written().len), .little);
    try output.interface.writeAll(header.written());
    for (tensors.items) |tensor| for (tensor.data) |value| {
        const bits: u32 = @bitCast(value);
        // All generated values are exactly representable in BF16.
        try std.testing.expectEqual(@as(u32, 0), bits & 0xffff);
        try output.interface.writeInt(u16, @intCast(bits >> 16), .little);
    };
    try output.interface.flush();
}

test "gemma4 serving mixed sliding and global layers match full-prefix logits through ring wrap" {
    const required = platform.env.getenvBoolDefault("TERMITE_REQUIRE_METAL_TESTS", false);
    if (!build_options.enable_metal or !metal.metalDeviceAvailable()) {
        if (required) return error.RequiredMetalDeviceUnavailable;
        return error.SkipZigTest;
    }
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = model_config });
    const root = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(root);
    try writeModel(allocator, root);
    const config = try factory.loadGptConfigMetadataFromModelDir(allocator, root);
    try std.testing.expect(config.layerUsesSlidingAttention(0));
    try std.testing.expect(!config.layerUsesSlidingAttention(1));
    var reference = try gemma_backend.loadBackendForModelDir(allocator, root, .native);
    defer reference.deinit();
    var device = try gemma_backend.loadBackendForModelDir(allocator, root, .metal);
    defer device.deinit();
    // Exercise the serving precision policy, not training's norm override.
    const compute: *metal_compute.MetalCompute = @ptrCast(@alignCast(device.backendPtr().ptr));
    compute.precise_training_rms_norm = false;
    const cb = device.backendPtr();
    try std.testing.expect(try direct.prepareDecodeRuntime(cb, allocator, config, 32, 2));
    var storage = try kv.KvStorageRuntime.init(allocator, .{
        .backend = .metal,
        .dtype = .f32,
        .page_size_tokens = 2,
        .num_layers_packed = 2,
        .num_kv_heads = 1,
        .head_dim = 256,
        .sliding_window_size = 4,
        .store_cpu_bytes = false,
    });
    defer storage.deinit();
    try cb.provisionKvDeviceWriteHook(&storage);
    const pool = storage.poolId();
    const sequence = try storage.attachSequence(pool);
    const tokens = [_]i64{ 2, 7, 13, 4, 19, 6, 23, 11, 3, 17, 29, 5, 14 };
    var window_effect: f32 = 0;
    var maximum_error: f32 = 0;
    for (3..tokens.len + 1) |length| {
        try storage.appendTokens(sequence, if (length == 3) 3 else 1);
        const table = storage.blockTable(sequence).?;
        const context: gpt.DecodeContext = .{
            .attention_mode = if (length == 3) .paged_prefill else .paged_decode,
            .total_sequence_len = length,
            .query_sequence_len = if (length == 3) 3 else 1,
            .kv_sequence_len = length,
            .kv_cache = .{ .sequence_id = sequence, .pool_id = pool, .logical_block_count = table.len(), .tail_tokens = table.tail_tokens, .logical_blocks = table.blocks.items, .kv_storage = &storage, .max_inflight_tokens = 3, .allow_swa_ring = true },
            .kv_storage = &storage,
        };
        const actual = (if (length == 3)
            try direct.forwardPrefillLastLogits(cb, allocator, config, 2, tokens[0..length], length, &context)
        else
            try direct.forwardLastLogits(cb, allocator, config, 2, tokens[length - 1], length, &context)) orelse return error.DirectServingPathDeclined;
        defer allocator.free(actual);
        const all_expected = try gpt.forward(reference.backendPtr(), allocator, config, tokens[0..length], 1, length, null);
        defer allocator.free(all_expected);
        const expected = all_expected[all_expected.len - config.vocab_size ..];
        try std.testing.expectEqual(expected.len, actual.len);
        var native_error: f32 = 0;
        for (expected, actual) |wanted, got| native_error = @max(native_error, @abs(wanted - got));
        std.debug.print("serving parity length={d} native_error={d}\n", .{ length, native_error });
        maximum_error = @max(maximum_error, native_error);
        if (length > config.sliding_window) {
            var global_config = config;
            // Preserve local/global head and RoPE selection; change only
            // the mask's history horizon for this negative control.
            global_config.sliding_window = tokens.len + 1;
            const global_logits = try gpt.forward(reference.backendPtr(), allocator, global_config, tokens[0..length], 1, length, null);
            defer allocator.free(global_logits);
            for (expected, global_logits[global_logits.len - config.vocab_size ..]) |windowed, global| {
                window_effect = @max(window_effect, @abs(windowed - global));
            }
        }
    }
    const hook = storage.device_write_hook orelse return error.MissingDeviceKvStorage;
    const sliding = try hook.pagedLayerKvDevice(.{ .sequence_id = sequence, .layer_index = 0, .token_count = tokens.len, .num_kv_heads = 1, .head_dim = 256 });
    const global = try hook.pagedLayerKvDevice(.{ .sequence_id = sequence, .layer_index = 1, .token_count = tokens.len, .num_kv_heads = 1, .head_dim = 256 });
    std.debug.print("serving parity max_error={d} window_effect={d} sliding_ring_pages={d} page_size={d} global_ring_pages={d}\n", .{ maximum_error, window_effect, sliding.ring_page_count, sliding.page_size_tokens, global.ring_page_count });
    try std.testing.expect(window_effect > 0.003);
    try std.testing.expect(maximum_error < 3e-4);
    try std.testing.expect(sliding.ring_page_count > 0);
    try std.testing.expect(sliding.ring_page_count * sliding.page_size_tokens < tokens.len);
    try std.testing.expectEqual(@as(usize, 0), global.ring_page_count);
}
