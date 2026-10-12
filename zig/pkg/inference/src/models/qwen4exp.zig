// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
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

//! Qwen4Exp GGUF execution contract. This parser does not grant serving
//! compatibility: architecture execution and exact-artifact qualification are
//! separate gates.
const std = @import("std");
const gguf = @import("../gguf/root.zig");

pub const artifact_revision = "766911a6b7369840a91dbcd95f9f997acaab6cd6";
pub const artifact_shards = [_]@import("../util/artifact_verification.zig").Identity{
    .{ .bytes = 10946624, .sha256 = "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082" },
    .{ .bytes = 49859583136, .sha256 = "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9" },
    .{ .bytes = 49376141504, .sha256 = "56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3" },
    .{ .bytes = 12087983520, .sha256 = "753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a" },
};

pub const max_layers = 48;
pub const max_heads = 16;
pub const Config = struct {
    hidden: usize,
    layers: usize,
    experts: usize,
    top_k: usize,
    expert_hidden: usize,
    shared_hidden: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rope_dim: usize,
    full_interval: usize,
    linear_heads: usize,
    linear_key_heads: usize,
    linear_dim: usize,
    conv_kernel: usize,
    hc_count: usize,
    hc_rank: usize,
    index_heads: usize,
    index_dim: usize,
    index_budget: usize,
    compress_ratio: usize,
    ple_layer: usize,
    ple_kernel: usize,
    ple_row_dim: usize,
    eos: u32,
    multipliers: [3]u64,
    head_offsets: [max_heads]u64,
    head_vocab_sizes: [max_heads]u64,
    epsilon: f32,
    theta: f32,

    pub fn fromGguf(file: *const gguf.format.File) !Config {
        const v = gguf.metadata.View.init(file);
        if (!std.mem.eql(u8, v.getString("general.architecture") orelse "", "qwen4exp")) return error.NotQwen4Exp;
        const ple = try array(1, v, "ple.layers");
        const ratios = try array(max_layers, v, "attention.compress_ratios");
        if (try integer(v, "ple.ngram_size") != 3 or try integer(v, "ple.heads_per_ngram") != 8) return error.UnsupportedQwen4ExpGeometry;
        const cfg = Config{
            .hidden = try integer(v, "embedding_length"),
            .layers = try integer(v, "block_count"),
            .experts = try integer(v, "expert_count"),
            .top_k = try integer(v, "expert_used_count"),
            .expert_hidden = try integer(v, "expert_feed_forward_length"),
            .shared_hidden = try integer(v, "expert_shared_feed_forward_length"),
            .heads = try integer(v, "attention.head_count"),
            .kv_heads = try integer(v, "attention.head_count_kv"),
            .head_dim = try integer(v, "attention.key_length"),
            .rope_dim = try integer(v, "rope.dimension_count"),
            .full_interval = try integer(v, "full_attention_interval"),
            .linear_heads = try integer(v, "ssm.time_step_rank"),
            .linear_key_heads = try integer(v, "ssm.group_count"),
            .linear_dim = try integer(v, "ssm.state_size"),
            .conv_kernel = try integer(v, "ssm.conv_kernel"),
            .hc_count = try integer(v, "hyper_connection.count"),
            .hc_rank = try integer(v, "hyper_connection.low_rank"),
            .index_heads = try integer(v, "attention.indexer.head_count"),
            .index_dim = try integer(v, "attention.indexer.key_length"),
            .index_budget = try integer(v, "attention.indexer.top_k"),
            .compress_ratio = std.math.cast(usize, ratios[3]) orelse return error.UnsupportedQwen4ExpGeometry,
            .ple_layer = std.math.cast(usize, ple[0]) orelse return error.UnsupportedQwen4ExpGeometry,
            .ple_kernel = try integer(v, "ple.conv_kernel"),
            .ple_row_dim = try integer(v, "embedding_length_per_layer_input"),
            .eos = std.math.cast(u32, try integer(v, "ple.eos_token_id")) orelse return error.UnsupportedQwen4ExpGeometry,
            .multipliers = try array(3, v, "ple.layer_multipliers"),
            .head_offsets = try array(max_heads, v, "ple.head_offsets"),
            .head_vocab_sizes = try array(max_heads, v, "ple.head_vocab_sizes"),
            .epsilon = v.getF32("qwen4exp.attention.layer_norm_rms_epsilon") orelse return error.MissingQwen4ExpMetadata,
            .theta = v.getF32("qwen4exp.rope.freq_base") orelse return error.MissingQwen4ExpMetadata,
        };
        try cfg.validate();
        for (ratios, 0..) |ratio, layer| if (ratio != (if (cfg.isLinear(layer)) @as(u64, 0) else cfg.compress_ratio)) return error.UnsupportedQwen4ExpGeometry;
        if (try integer(v, "ssm.inner_size") != cfg.linear_heads * cfg.linear_dim or
            try integer(v, "attention.value_length") != cfg.head_dim) return error.UnsupportedQwen4ExpGeometry;
        return cfg;
    }

    pub fn validate(c: Config) !void {
        // The first experimental runtime deliberately accepts only the geometry
        // whose SSD paging footprint and kernel shapes are qualified together.
        if (c.hidden != 2560 or c.layers != 48 or c.experts != 512 or c.top_k != 10 or
            c.expert_hidden != 640 or c.shared_hidden != 640 or c.heads != 24 or c.kv_heads != 2 or
            c.head_dim != 256 or c.rope_dim != 64 or c.full_interval != 4 or c.linear_heads != 48 or
            c.linear_key_heads != 16 or c.linear_dim != 128 or c.conv_kernel != 4 or c.hc_count != 4 or
            c.hc_rank != 320 or c.index_heads != 4 or c.index_dim != 128 or c.index_budget != 2048 or
            c.compress_ratio != 4 or c.ple_layer != 1 or c.ple_kernel != 4 or c.ple_row_dim != 160 or
            !std.math.isFinite(c.epsilon) or c.epsilon <= 0 or !std.math.isFinite(c.theta) or c.theta <= 0)
            return error.UnsupportedQwen4ExpGeometry;
        var end: u64 = 0;
        for (c.head_offsets, c.head_vocab_sizes) |offset, size| {
            if (size == 0 or size > std.math.maxInt(i32) or offset != end) return error.InvalidQwen4ExpNgramRange;
            end = std.math.add(u64, offset, size) catch return error.InvalidQwen4ExpNgramRange;
            if (end > std.math.maxInt(i32)) return error.InvalidQwen4ExpNgramRange;
        }
    }

    pub fn isLinear(c: Config, layer: usize) bool {
        return (layer + 1) % c.full_interval != 0;
    }

    pub fn stateBytes(c: Config, context: usize) !usize {
        try c.validate();
        if (context == 0 or context > 4096) return error.UnsupportedQwen4ExpContext;
        const attention_layers = c.layers / c.full_interval;
        const linear_layers = c.layers - attention_layers;
        const linear = linear_layers * (c.linear_heads * c.linear_dim * c.linear_dim +
            c.conv_kernel * (2 * c.linear_key_heads + c.linear_heads) * c.linear_dim);
        const attention = attention_layers * context * (2 * c.kv_heads * c.head_dim + c.index_dim * 2);
        const ple = (c.ple_kernel - 1) * 3 * c.hc_count * c.hidden;
        return (linear + attention + ple) * @sizeOf(f32);
    }
};

fn integer(v: gguf.metadata.View, comptime suffix: []const u8) !usize {
    const key = "qwen4exp." ++ suffix;
    const value = v.getU64(key) orelse if (v.getI64(key)) |n| std.math.cast(u64, n) orelse return error.InvalidQwen4ExpMetadata else return error.MissingQwen4ExpMetadata;
    return std.math.cast(usize, value) orelse error.InvalidQwen4ExpMetadata;
}

fn array(comptime n: usize, v: gguf.metadata.View, comptime suffix: []const u8) ![n]u64 {
    const entry = v.find("qwen4exp." ++ suffix) orelse return error.MissingQwen4ExpMetadata;
    if (entry.value != .array or entry.value.array.values.len != n) return error.InvalidQwen4ExpMetadata;
    var result: [n]u64 = undefined;
    for (entry.value.array.values, &result) |value, *out| out.* = switch (value) {
        .u64 => |x| x,
        .u32 => |x| x,
        .i32 => |x| std.math.cast(u64, x) orelse return error.InvalidQwen4ExpMetadata,
        .i64 => |x| std.math.cast(u64, x) orelse return error.InvalidQwen4ExpMetadata,
        else => return error.InvalidQwen4ExpMetadata,
    };
    return result;
}

pub const Inventory = struct {
    resident_bytes: usize = 0,
    ngram_bytes: usize = 0,
    expert_bytes: usize = 0,
    per_expert_bytes: [max_layers]usize = @splat(0),
    expert_projections: [max_layers]u8 = @splat(0),

    pub fn inspect(file: *const gguf.format.File, cfg: Config) !Inventory {
        try cfg.validate();
        var result: Inventory = .{};
        for (file.tensors) |tensor| {
            const bytes = std.math.cast(usize, gguf.tensor_types.byteLen(tensor.tensor_type, tensor.dimensions) orelse return error.UnsupportedTensorType) orelse return error.TensorTooLarge;
            if (std.mem.eql(u8, tensor.name, "per_layer_token_embd.weight")) {
                if (result.ngram_bytes != 0 or tensor.dimensions.len != 2 or tensor.dimensions[0] != cfg.ple_row_dim or
                    tensor.dimensions[1] < cfg.head_offsets[max_heads - 1] + cfg.head_vocab_sizes[max_heads - 1] or
                    tensor.tensor_type.raw() != @backingInt(gguf.tensor_types.KnownTensorType.IQ4_NL)) return error.InvalidQwen4ExpNgramTensor;
                result.ngram_bytes = bytes;
            } else if (std.mem.endsWith(u8, tensor.name, "_exps.weight")) {
                var parts = std.mem.splitScalar(u8, tensor.name, '.');
                if (!std.mem.eql(u8, parts.next() orelse "", "blk")) return error.InvalidQwen4ExpExpertTensor;
                const layer = try std.fmt.parseInt(usize, parts.next() orelse "", 10);
                if (layer >= cfg.layers or tensor.dimensions.len != 3 or tensor.dimensions[2] != cfg.experts) return error.InvalidQwen4ExpExpertTensor;
                const projection = parts.next() orelse return error.InvalidQwen4ExpExpertTensor;
                const mask: u8 = if (std.mem.eql(u8, projection, "ffn_gate_exps")) 1 else if (std.mem.eql(u8, projection, "ffn_up_exps")) 2 else if (std.mem.eql(u8, projection, "ffn_down_exps")) 4 else return error.InvalidQwen4ExpExpertTensor;
                if (tensor.dimensions[0] != (if (mask == 4) cfg.expert_hidden else cfg.hidden) or
                    tensor.dimensions[1] != (if (mask == 4) cfg.hidden else cfg.expert_hidden) or result.expert_projections[layer] & mask != 0)
                    return error.InvalidQwen4ExpExpertTensor;
                switch (tensor.tensor_type) {
                    .known => |t| switch (t) {
                        .Q4_K, .Q5_K, .Q5_1, .Q8_0 => {},
                        else => return error.UnsupportedQwen4ExpExpertFormat,
                    },
                    else => return error.UnsupportedQwen4ExpExpertFormat,
                }
                result.expert_projections[layer] |= mask;
                result.per_expert_bytes[layer] = try std.math.add(usize, result.per_expert_bytes[layer], bytes / cfg.experts);
                result.expert_bytes = try std.math.add(usize, result.expert_bytes, bytes);
            } else result.resident_bytes = try std.math.add(usize, result.resident_bytes, bytes);
        }
        if (result.ngram_bytes == 0) return error.MissingQwen4ExpNgramTensor;
        for (result.expert_projections) |mask| if (mask != 7) return error.MissingQwen4ExpExpertProjection;
        return result;
    }

    /// Validate every ordinary matrix before allocating any encoded weights.
    pub fn validateRequired(file: *const gguf.format.File, cfg: Config) !void {
        try cfg.validate();
        try require(file, "token_embd.weight", &.{ 2560, 248320 }, &.{8});
        try require(file, "output.weight", &.{ 2560, 248320 }, &.{8});
        try require(file, "output_hc_norm.weight", &.{10240}, &.{0});
        try require(file, "output_hc_down.weight", &.{ 10240, 320 }, &.{8});
        try require(file, "output_hc_up.weight", &.{ 320, 10240 }, &.{8});
        for (0..48) |layer| {
            for ([_][]const u8{ "attn", "ffn" }) |kind| {
                var prefix_buf: [64]u8 = undefined;
                const prefix = try std.fmt.bufPrint(&prefix_buf, "hc_{s}_", .{kind});
                inline for (.{ .{ "norm", &.{10240}, &.{0} }, .{ "down", &.{ 10240, 320 }, &.{8} }, .{ "up", &.{ 320, 10240 }, &.{8} }, .{ "inject", &.{ 10240, 4 }, &.{0} } }) |spec| {
                    var name_buf: [96]u8 = undefined;
                    try require(file, try std.fmt.bufPrint(&name_buf, "blk.{d}.{s}{s}.weight", .{ layer, prefix, spec[0] }), spec[1], spec[2]);
                }
            }
            inline for (.{ .{ "ffn_gate_inp.weight", &.{ 2560, 512 }, &.{0} }, .{ "ffn_gate_inp_shexp.weight", &.{2560}, &.{0} }, .{ "ffn_gate_shexp.weight", &.{ 2560, 640 }, &.{8} }, .{ "ffn_up_shexp.weight", &.{ 2560, 640 }, &.{8} }, .{ "ffn_down_shexp.weight", &.{ 640, 2560 }, &.{8} } }) |spec| {
                try requireLayer(file, layer, spec[0], spec[1], spec[2]);
            }
            if (cfg.isLinear(layer)) {
                inline for (.{ .{ "attn_qkv.weight", &.{ 2560, 10240 }, &.{8} }, .{ "attn_gate.weight", &.{ 2560, 6144 }, &.{8} }, .{ "ssm_a", &.{48}, &.{0} }, .{ "ssm_alpha.weight", &.{ 2560, 48 }, &.{0} }, .{ "ssm_beta.weight", &.{ 2560, 48 }, &.{0} }, .{ "ssm_dt.bias", &.{48}, &.{0} }, .{ "ssm_conv1d.weight", &.{ 4, 10240 }, &.{0} }, .{ "ssm_norm.weight", &.{128}, &.{0} }, .{ "ssm_out.weight", &.{ 6144, 2560 }, &.{8} } }) |spec| {
                    try requireLayer(file, layer, spec[0], spec[1], spec[2]);
                }
            } else {
                inline for (.{ .{ "attn_q.weight", &.{ 2560, 12288 }, &.{8} }, .{ "attn_k.weight", &.{ 2560, 512 }, &.{8} }, .{ "attn_v.weight", &.{ 2560, 512 }, &.{8} }, .{ "attn_output.weight", &.{ 6144, 2560 }, &.{8} }, .{ "attn_q_norm.weight", &.{256}, &.{0} }, .{ "attn_k_norm.weight", &.{256}, &.{0} }, .{ "indexer.q_proj.weight", &.{ 2560, 512 }, &.{30} }, .{ "indexer.k_proj.weight", &.{ 2560, 128 }, &.{30} }, .{ "indexer.q_norm.weight", &.{128}, &.{0} }, .{ "indexer.k_norm.weight", &.{128}, &.{0} } }) |spec| {
                    try requireLayer(file, layer, spec[0], spec[1], spec[2]);
                }
            }
        }
        inline for (.{ .{ "ple_key.weight", &.{ 2560, 10240 }, &.{8} }, .{ "ple_value.weight", &.{ 2560, 2560 }, &.{8} }, .{ "ple_norm_key.weight", &.{10240}, &.{0} }, .{ "ple_norm_query.weight", &.{10240}, &.{0} }, .{ "ple_norm_conv.weight", &.{10240}, &.{0} }, .{ "ple_conv1d.weight", &.{ 4, 10240 }, &.{0} } }) |spec| {
            try requireLayer(file, 1, spec[0], spec[1], spec[2]);
        }
        if (file.tensors.len != 1224) return error.UnsupportedQwen4ExpTensorInventory;
    }

    pub fn cacheSlots(self: Inventory, budget: usize, top_k: usize) ![max_layers]u8 {
        if (top_k == 0 or top_k > 128) return error.InvalidQwen4ExpCacheBudget;
        var slots: [max_layers]u8 = @splat(std.math.cast(u8, top_k) orelse return error.InvalidQwen4ExpCacheBudget);
        var used: usize = 0;
        for (self.per_expert_bytes) |bytes| {
            if (bytes == 0) return error.MissingQwen4ExpExpertProjection;
            used = try std.math.add(usize, used, try std.math.mul(usize, bytes, top_k));
        }
        if (used > budget) return error.InvalidQwen4ExpCacheBudget;
        while (true) {
            var progress = false;
            for (&slots, self.per_expert_bytes) |*count, bytes| {
                if (count.* >= 128 or bytes > budget - used) continue;
                count.* += 1;
                used += bytes;
                progress = true;
            }
            if (!progress) break;
        }
        return slots;
    }
};

fn requireLayer(file: *const gguf.format.File, layer: usize, suffix: []const u8, dims: []const u64, formats: []const u32) !void {
    var name: [96]u8 = undefined;
    try require(file, try std.fmt.bufPrint(&name, "blk.{d}.{s}", .{ layer, suffix }), dims, formats);
}
fn require(file: *const gguf.format.File, name: []const u8, dims: []const u64, formats: []const u32) !void {
    for (file.tensors) |tensor| {
        if (!std.mem.eql(u8, tensor.name, name)) continue;
        if (!std.mem.eql(u64, tensor.dimensions, dims)) return error.InvalidQwen4ExpTensorShape;
        for (formats) |format| if (tensor.tensor_type.raw() == format) return;
        return error.UnsupportedQwen4ExpTensorFormat;
    }
    return error.MissingQwen4ExpTensor;
}

/// Hash products use wrapping signed int64 arithmetic, matching Torch. An EOS
/// participates in its own n-gram; only subsequent tokens lose prior history.
pub const NgramState = struct {
    previous: [2]u32,
    pub fn init(eos: u32) NgramState {
        return .{ .previous = .{ eos, eos } };
    }
    pub fn next(self: *NgramState, token: u32, eos: u32, multipliers: [3]u64, offsets: [max_heads]u64, sizes: [max_heads]u64) ![max_heads]u64 {
        var result: [max_heads]u64 = undefined;
        var mixed = @as(u64, token) *% multipliers[0];
        for (self.previous, 1..) |prior, n| {
            mixed ^= @as(u64, prior) *% multipliers[n];
            for (0..8) |h| {
                const index = (n - 1) * 8 + h;
                const divisor = std.math.cast(i64, sizes[index]) orelse return error.InvalidQwen4ExpNgramRange;
                if (divisor <= 0) return error.InvalidQwen4ExpNgramRange;
                const remainder: u64 = @intCast(@mod(@as(i64, @bitCast(mixed)), divisor));
                result[index] = try std.math.add(u64, offsets[index], remainder);
            }
        }
        // Shift before replacing the newest entry. A result-location array
        // literal assigned to self.previous aliases its own RHS in Zig.
        self.previous[1] = if (token == eos) eos else self.previous[0];
        self.previous[0] = token;
        return result;
    }
};

test "Qwen4Exp ngram signed wrap and EOS history" {
    var state = NgramState.init(7);
    const multipliers = [3]u64{ 0xffffffffffffffff, 3, 5 };
    const rows = try state.next(11, 7, multipliers, @splat(0), @splat(13));
    // Python: ((-11) ^ 21) % 13 == 7; ((-11) ^ 21 ^ 35) % 13 == 4.
    try std.testing.expectEqual(@as(u64, 7), rows[0]);
    try std.testing.expectEqual(@as(u64, 4), rows[8]);
    try std.testing.expectEqualSlices(u32, &.{ 11, 7 }, &state.previous);
    const next = try state.next(5, 7, multipliers, @splat(0), @splat(13));
    try std.testing.expectEqual(@as(u64, 1), next[0]);
    try std.testing.expectEqual(@as(u64, 6), next[8]);
    try std.testing.expectEqualSlices(u32, &.{ 5, 11 }, &state.previous);
    _ = try state.next(7, 7, multipliers, @splat(0), @splat(13));
    try std.testing.expectEqualSlices(u32, &.{ 7, 7 }, &state.previous);
}

test "Qwen4Exp cache sizing fits mixed expert strides and rejects undersized budget" {
    var inventory: Inventory = .{};
    inventory.per_expert_bytes = @splat(3 * 1024 * 1024);
    inventory.per_expert_bytes[47] = 4 * 1024 * 1024;
    const slots = try inventory.cacheSlots(2 * 1024 * 1024 * 1024, 10);
    var bytes: usize = 0;
    for (slots, inventory.per_expert_bytes) |count, stride| {
        try std.testing.expect(count >= 10);
        bytes += count * stride;
    }
    try std.testing.expect(bytes <= 2 * 1024 * 1024 * 1024);
    try std.testing.expectError(error.InvalidQwen4ExpCacheBudget, inventory.cacheSlots(1024, 10));
}

test "Qwen4Exp installed split GGUF inventory uses bounded metadata reads" {
    const path = std.c.getenv("ANTFLY_QWEN4EXP_GGUF") orelse return error.SkipZigTest;
    const store = try @import("tensor_store.zig").SplitGgufStore.open(std.testing.allocator, std.mem.span(path));
    defer store.deinit();
    const file = store.ggufFile().?;
    const cfg = try Config.fromGguf(file);
    const inventory = try Inventory.inspect(file, cfg);
    try Inventory.validateRequired(file, cfg);
    const slots = try inventory.cacheSlots(2 * 1024 * 1024 * 1024, cfg.top_k);
    try std.testing.expectEqual(@as(usize, 1224), file.tensors.len);
    try std.testing.expectEqual(@as(usize, 5_505_584_640), inventory.resident_bytes);
    try std.testing.expectEqual(@as(usize, 28_800_138_240), inventory.ngram_bytes);
    try std.testing.expectEqual(@as(usize, 77_017_907_200), inventory.expert_bytes);
    try std.testing.expect(try cfg.stateBytes(4096) < 512 * 1024 * 1024);
    for (slots) |count| try std.testing.expect(count >= 10);
    std.debug.print("Qwen4Exp inventory: tensors={d} resident={d} experts={d} ngram={d} state4096={d}\n", .{
        file.tensors.len, inventory.resident_bytes, inventory.expert_bytes, inventory.ngram_bytes, try cfg.stateBytes(4096),
    });
}
