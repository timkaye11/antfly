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

//! Exact local Qwen3.8-27B contract. Only headers and explicit ranges are read.
const std = @import("std");
const gguf = @import("../gguf/root.zig");
const c_file = @import("../util/c_file.zig");
const verification = @import("../util/artifact_verification.zig");
pub const MiB = 1024 * 1024;
pub const layer_count = 64;
pub const layer_buffer_bytes = 288 * MiB;
pub const artifact_bytes = 16464440224;
pub const artifact_sha256 = "322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482";
pub const Span = struct { offset: u64, bytes: usize };
pub const Source = struct {
    allocator: std.mem.Allocator,
    reader: c_file.RangeReader,
    file: gguf.format.File,
    layers: [64]Span,
    /// The artifact digest holds for this descriptor: hashed by this process
    /// or vouched for by an accepted stamp (see `verification`).
    verified: bool = false,
    verification: verification.Outcome = .{},

    pub fn open(a: std.mem.Allocator, path: []const u8) !Source {
        var reader = try c_file.RangeReader.open(a, path, true);
        errdefer reader.deinit();
        if (reader.size != artifact_bytes) return error.Qwen35ArtifactSizeMismatch;
        // This exact artifact has a 10.49 MiB header. Never map tensor payloads.
        const header = try a.alloc(u8, 16 * MiB);
        defer a.free(header);
        try reader.read(0, header);
        var file = try gguf.format.parseStructure(a, header);
        errdefer file.deinit(a);
        try gguf.format.validateTensorDataRanges(&file, @intCast(reader.size));
        return .{ .allocator = a, .reader = reader, .file = file, .layers = try inspect(&file) };
    }
    pub fn deinit(self: *Source) void {
        self.file.deinit(self.allocator);
        self.reader.deinit();
    }
    pub fn tensor(self: *const Source, name: []const u8) !*const gguf.format.TensorInfo {
        for (self.file.tensors) |*t| if (std.mem.eql(u8, t.name, name)) return t;
        return error.MissingQwen35Tensor;
    }
    pub fn layerTensor(self: *const Source, layer: usize, suffix: []const u8) !*const gguf.format.TensorInfo {
        var buf: [96]u8 = undefined;
        return self.tensor(try std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ layer, suffix }));
    }
    /// Verify through the descriptor every later range read uses. `path` only
    /// keys the optional stamp.
    pub fn verify(self: *Source, path: []const u8, mode: verification.Mode, stamp_dir: ?[]const u8, cancelled: *const std.atomic.Value(bool)) !void {
        const files = [_]verification.File{.{ .reader = &self.reader, .path = path, .pin = .{ .bytes = artifact_bytes, .sha256 = artifact_sha256 } }};
        self.verification = verification.verifyFiles(self.allocator, &files, artifact_sha256, mode, stamp_dir, cancelled) catch |err| return switch (err) {
            error.ArtifactHashMismatch => error.Qwen35ArtifactHashMismatch,
            error.ArtifactSizeMismatch => error.Qwen35ArtifactSizeMismatch,
            else => err,
        };
        self.verified = true;
    }
};
pub fn byteLen(t: *const gguf.format.TensorInfo) !usize {
    return std.math.cast(usize, gguf.tensor_types.byteLen(t.tensor_type, t.dimensions) orelse return error.UnsupportedQwen35Format) orelse error.TensorTooLarge;
}
pub fn isLinear(layer: usize) bool {
    return (layer + 1) % 4 != 0;
}
/// State for `context` positions. The admitted context limit belongs to the
/// runtime's machine profile, not to this arithmetic.
pub fn stateBytes(context: usize) !usize {
    if (context == 0) return error.UnsupportedQwen35Context;
    const attention = std.math.mul(usize, context, 16 * 2 * 1024 * 4) catch return error.UnsupportedQwen35Context;
    return std.math.add(usize, 48 * (48 * 128 * 128 + 3 * 10240) * 4, attention) catch error.UnsupportedQwen35Context;
}
fn integer(v: gguf.metadata.View, key: []const u8) !u64 {
    return v.getU64(key) orelse return error.MissingQwen35Metadata;
}
fn require(file: *const gguf.format.File, layer: ?usize, suffix: []const u8, dims: []const u64) !void {
    var buf: [96]u8 = undefined;
    const name = if (layer) |l| try std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ l, suffix }) else suffix;
    for (file.tensors) |t| if (std.mem.eql(u8, t.name, name)) {
        if (!std.mem.eql(u64, t.dimensions, dims)) return error.InvalidQwen35TensorShape;
        return;
    };
    return error.MissingQwen35Tensor;
}
pub fn inspect(file: *const gguf.format.File) ![64]Span {
    const v = gguf.metadata.View.init(file);
    if (!std.mem.eql(u8, v.getString("general.architecture") orelse "", "qwen35")) return error.NotQwen35;
    const keys = .{
        .{ "block_count", 65 },            .{ "nextn_predict_layers", 1 },     .{ "embedding_length", 5120 },
        .{ "feed_forward_length", 17408 }, .{ "attention.head_count", 24 },    .{ "attention.head_count_kv", 4 },
        .{ "attention.key_length", 256 },  .{ "attention.value_length", 256 }, .{ "rope.dimension_count", 64 },
        .{ "ssm.conv_kernel", 4 },         .{ "ssm.state_size", 128 },         .{ "ssm.group_count", 16 },
        .{ "ssm.time_step_rank", 48 },     .{ "ssm.inner_size", 6144 },        .{ "full_attention_interval", 4 },
    };
    inline for (keys) |kv| if (try integer(v, "qwen35." ++ kv[0]) != kv[1]) return error.UnsupportedQwen35Geometry;
    if (v.getF32("qwen35.rope.freq_base") != 10000000 or v.getF32("qwen35.attention.layer_norm_rms_epsilon") != @as(f32, 1e-6)) return error.UnsupportedQwen35Geometry;
    if (file.tensors.len != 866) return error.UnsupportedQwen35Inventory;
    var spans: [64]Span = @splat(.{ .offset = std.math.maxInt(u64), .bytes = 0 });
    var ends: [64]u64 = @splat(0);
    var names: std.StringHashMapUnmanaged(void) = .{};
    defer names.deinit(std.heap.page_allocator);
    for (file.tensors) |*t| {
        const entry = try names.getOrPut(std.heap.page_allocator, t.name);
        if (entry.found_existing) return error.DuplicateQwen35Tensor;
        switch (t.tensor_type.raw()) {
            0, 8, 11, 12, 13, 14, 20, 21, 23 => {},
            else => return error.UnsupportedQwen35Format,
        }
        if (std.mem.startsWith(u8, t.name, "blk.")) {
            var parts = std.mem.splitScalar(u8, t.name, '.');
            _ = parts.next();
            const l = try std.fmt.parseInt(usize, parts.next() orelse return error.InvalidQwen35Layer, 10);
            if (l > 64) return error.InvalidQwen35Layer;
            if (l == 64) continue; // MTP is explicitly excluded from ordinary decoding.
            spans[l].offset = @min(spans[l].offset, t.data_offset);
            ends[l] = @max(ends[l], try std.math.add(u64, t.data_offset, try byteLen(t)));
        }
    }
    try require(file, null, "token_embd.weight", &.{ 5120, 248320 });
    try require(file, null, "output.weight", &.{ 5120, 248320 });
    try require(file, null, "output_norm.weight", &.{5120});
    for (&spans, 0..) |*span, l| {
        if (ends[l] <= span.offset) return error.MissingQwen35Layer;
        span.bytes = @intCast(ends[l] - span.offset);
        if (span.bytes > layer_buffer_bytes) return error.Qwen35LayerTooLarge;
        try require(file, l, "attn_norm.weight", &.{5120});
        try require(file, l, "post_attention_norm.weight", &.{5120});
        try require(file, l, "ffn_gate.weight", &.{ 5120, 17408 });
        try require(file, l, "ffn_up.weight", &.{ 5120, 17408 });
        try require(file, l, "ffn_down.weight", &.{ 17408, 5120 });
        if (isLinear(l)) {
            try require(file, l, "attn_qkv.weight", &.{ 5120, 10240 });
            try require(file, l, "attn_gate.weight", &.{ 5120, 6144 });
            try require(file, l, "ssm_alpha.weight", &.{ 5120, 48 });
            try require(file, l, "ssm_beta.weight", &.{ 5120, 48 });
            try require(file, l, "ssm_a", &.{48});
            try require(file, l, "ssm_dt.bias", &.{48});
            try require(file, l, "ssm_conv1d.weight", &.{ 4, 10240 });
            try require(file, l, "ssm_norm.weight", &.{128});
            try require(file, l, "ssm_out.weight", &.{ 6144, 5120 });
        } else {
            try require(file, l, "attn_q.weight", &.{ 5120, 12288 });
            try require(file, l, "attn_k.weight", &.{ 5120, 1024 });
            try require(file, l, "attn_v.weight", &.{ 5120, 1024 });
            try require(file, l, "attn_q_norm.weight", &.{256});
            try require(file, l, "attn_k_norm.weight", &.{256});
            try require(file, l, "attn_output.weight", &.{ 6144, 5120 });
        }
    }
    return spans;
}
test "Qwen35 streamed state accounts for 48 recurrent and 16 full attention layers" {
    try std.testing.expectEqual(@as(usize, 693764096), try stateBytes(4096));
    // Twice the context adds exactly the attention layers' share again.
    try std.testing.expectEqual(@as(usize, 693764096 + 16 * 4096 * 2 * 1024 * 4), try stateBytes(8192));
    try std.testing.expectError(error.UnsupportedQwen35Context, stateBytes(std.math.maxInt(usize)));
    try std.testing.expectError(error.UnsupportedQwen35Context, stateBytes(0));
}
