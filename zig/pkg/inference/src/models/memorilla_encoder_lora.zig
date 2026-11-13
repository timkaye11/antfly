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

//! Pinned run-time encoder LoRA of a hybrid research bundle
//! (docs/plans/memorilla-copy17-native.md section 8.7).
//!
//! The hybrid profile keeps ONE resident encoder, the stock managed
//! `qwen3_encoder_f32_v1` Qwen3-Embedding-4B. The copy17 candidate's encoder
//! adapters (`encoder-lora.safetensors`, exported by `encoder_lora_export.py
//! export-reader --include-encoder-lora`) are applied per request to the q/v
//! projections of the per-token record pass (`gpt.EncoderLora`), never merged
//! into resident weights. This module reads and verifies the adapter file
//! against the bundle's own pins and packs it into the run-input layout:
//!
//! - `manifest.json` (`antfly.memorilla.bundle.v1`, `release: false`) pins the
//!   size and sha256 of `encoder-lora.safetensors`, `candidate.json` and
//!   `protocol.json`, and `source.candidate_sha256` = sha256(`candidate.json`);
//! - `candidate.json` `encoder_lora_sha256` must equal the adapter file digest;
//! - `protocol.json` gives `rank`, `alpha` (scale `alpha / rank`) and
//!   `target_modules` (exactly `q_proj`, `v_proj`);
//! - the safetensors file holds exactly `layers.{l}.self_attn.{q_proj,v_proj}.{A,B}`
//!   for every layer, F32, `A [rank, hidden]`, `B [out, rank]`.
//!
//! Any mismatch fails closed (`MemorillaCandidateMismatch` for a pin that
//! names another candidate, `InvalidMemorillaEncoderLora` for a malformed file).
const std = @import("std");
const bundle = @import("memorilla.zig");
const c_file = @import("../util/c_file.zig");
const A = std.mem.Allocator;

pub const file_name = "encoder-lora.safetensors";
pub const max_file_bytes: usize = 64 * 1024 * 1024;
const manifest_max_bytes: usize = 2 * 1024 * 1024;
const header_max_bytes: usize = 1024 * 1024;

/// Projection geometry the adapters must fit.
pub const Geometry = struct {
    layers: usize,
    hidden: usize,
    q_out: usize,
    v_out: usize,
};

/// Qwen3-Embedding-4B: 36 layers, hidden 2560, 32 x 128 query and 8 x 128
/// key/value heads.
pub const qwen3_embedding_4b = Geometry{ .layers = 36, .hidden = 2560, .q_out = 4096, .v_out = 1024 };

/// Elements of one packed layer: `q_A, q_B, v_A, v_B` (gpt.EncoderLora layout).
pub fn layerElements(geometry: Geometry, rank: usize) usize {
    return rank * geometry.hidden * 2 + (geometry.q_out + geometry.v_out) * rank;
}

pub const Adapter = struct {
    allocator: A,
    /// Packed host F32 `[layers, layerElements]`.
    values: []f32,
    layers: usize,
    rank: usize,
    /// `alpha / rank` from `protocol.json`.
    scale: f32,
    /// sha256 of the adapter file bytes (= candidate `encoder_lora_sha256`).
    sha256: [64]u8,
    /// sha256 of the bundle's `candidate.json`.
    candidate_sha256: [64]u8,
    /// sha256 of the `manifest.json` bytes the files were verified against;
    /// the reader cache's opened bundle must report the same manifest.
    manifest_sha256: [64]u8,

    pub fn deinit(self: *Adapter) void {
        self.allocator.free(self.values);
        self.* = undefined;
    }

    pub fn bytes(self: *const Adapter) usize {
        return self.values.len * @sizeOf(f32);
    }
};

fn member(value: std.json.Value, name: []const u8) !std.json.Value {
    if (value != .object) return error.InvalidMemorillaManifest;
    return value.object.get(name) orelse error.InvalidMemorillaManifest;
}

fn readPinned(a: A, directory: []const u8, name: []const u8, maximum: usize, manifest: std.json.Value) ![]u8 {
    const path = try std.fs.path.join(a, &.{ directory, name });
    defer a.free(path);
    const bytes = try c_file.readFileMax(a, path, maximum);
    errdefer a.free(bytes);
    try bundle.verifyFile(manifest, name, bytes);
    return bytes;
}

/// Protocol hyperparameters of the adapters.
pub const Protocol = struct { rank: usize, scale: f32 };

pub fn parseProtocol(a: A, bytes: []const u8) !Protocol {
    var parsed = std.json.parseFromSlice(std.json.Value, a, bytes, .{}) catch return error.InvalidMemorillaProtocol;
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidMemorillaProtocol;
    const object = parsed.value.object;
    const rank = object.get("rank") orelse return error.InvalidMemorillaProtocol;
    if (rank != .integer or rank.integer <= 0 or rank.integer > 4096) return error.InvalidMemorillaProtocol;
    const alpha_value = object.get("alpha") orelse return error.InvalidMemorillaProtocol;
    const alpha: f64 = switch (alpha_value) {
        .integer => |v| @floatFromInt(v),
        .float => |v| v,
        else => return error.InvalidMemorillaProtocol,
    };
    if (!std.math.isFinite(alpha) or alpha <= 0) return error.InvalidMemorillaProtocol;
    const targets = object.get("target_modules") orelse return error.InvalidMemorillaProtocol;
    if (targets != .array or targets.array.items.len != 2) return error.InvalidMemorillaProtocol;
    for (targets.array.items, [_][]const u8{ "q_proj", "v_proj" }) |item, expected| {
        if (item != .string or !std.mem.eql(u8, item.string, expected)) return error.InvalidMemorillaProtocol;
    }
    const r: usize = @intCast(rank.integer);
    return .{ .rank = r, .scale = @floatCast(alpha / @as(f64, @floatFromInt(r))) };
}

/// Safetensors bytes -> packed `[layers, layerElements]` F32 values. The file
/// must hold exactly the q/v `A`/`B` tensors of every layer, little-endian F32
/// with the expected shapes and disjoint in-bounds data.
pub fn pack(a: A, file: []const u8, geometry: Geometry, rank: usize) ![]f32 {
    if (file.len < 8) return error.InvalidMemorillaEncoderLora;
    const header_len = std.mem.readInt(u64, file[0..8], .little);
    if (header_len == 0 or header_len > header_max_bytes or header_len > file.len - 8) return error.InvalidMemorillaEncoderLora;
    const data = file[8 + header_len ..];
    var parsed = std.json.parseFromSlice(std.json.Value, a, file[8..][0..header_len], .{}) catch return error.InvalidMemorillaEncoderLora;
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidMemorillaEncoderLora;
    const header = parsed.value.object;
    const tensors = geometry.layers * 4;
    var seen: usize = 0;
    var it = header.iterator();
    while (it.next()) |entry| {
        if (!std.mem.eql(u8, entry.key_ptr.*, "__metadata__")) seen += 1;
    }
    if (seen != tensors) return error.InvalidMemorillaEncoderLora;
    const per_layer = layerElements(geometry, rank);
    const values = try a.alloc(f32, try std.math.mul(usize, geometry.layers, per_layer));
    errdefer a.free(values);
    var name_buf: [96]u8 = undefined;
    const used = try a.alloc([2]usize, tensors);
    defer a.free(used);
    var used_count: usize = 0;
    for (0..geometry.layers) |layer| {
        var offset: usize = layer * per_layer;
        const parts = [_]struct { proj: []const u8, part: []const u8, rows: usize, cols: usize }{
            .{ .proj = "q_proj", .part = "A", .rows = rank, .cols = geometry.hidden },
            .{ .proj = "q_proj", .part = "B", .rows = geometry.q_out, .cols = rank },
            .{ .proj = "v_proj", .part = "A", .rows = rank, .cols = geometry.hidden },
            .{ .proj = "v_proj", .part = "B", .rows = geometry.v_out, .cols = rank },
        };
        for (parts) |spec| {
            const name = std.fmt.bufPrint(&name_buf, "layers.{d}.self_attn.{s}.{s}", .{ layer, spec.proj, spec.part }) catch return error.InvalidMemorillaEncoderLora;
            const info = header.get(name) orelse return error.InvalidMemorillaEncoderLora;
            if (info != .object) return error.InvalidMemorillaEncoderLora;
            const dtype = info.object.get("dtype") orelse return error.InvalidMemorillaEncoderLora;
            if (dtype != .string or !std.mem.eql(u8, dtype.string, "F32")) return error.InvalidMemorillaEncoderLora;
            const shape = info.object.get("shape") orelse return error.InvalidMemorillaEncoderLora;
            if (shape != .array or shape.array.items.len != 2) return error.InvalidMemorillaEncoderLora;
            for (shape.array.items, [_]usize{ spec.rows, spec.cols }) |dim, expected| {
                if (dim != .integer or dim.integer < 0 or @as(u64, @intCast(dim.integer)) != expected) return error.InvalidMemorillaEncoderLora;
            }
            const offsets = info.object.get("data_offsets") orelse return error.InvalidMemorillaEncoderLora;
            if (offsets != .array or offsets.array.items.len != 2) return error.InvalidMemorillaEncoderLora;
            const begin_value = offsets.array.items[0];
            const end_value = offsets.array.items[1];
            if (begin_value != .integer or end_value != .integer or begin_value.integer < 0 or end_value.integer < begin_value.integer) return error.InvalidMemorillaEncoderLora;
            const begin: usize = @intCast(begin_value.integer);
            const end: usize = @intCast(end_value.integer);
            const count = spec.rows * spec.cols;
            if (end > data.len or end - begin != count * @sizeOf(f32)) return error.InvalidMemorillaEncoderLora;
            for (used[0..used_count]) |range| if (begin < range[1] and range[0] < end) return error.InvalidMemorillaEncoderLora;
            used[used_count] = .{ begin, end };
            used_count += 1;
            const out = values[offset..][0..count];
            for (out, 0..) |*value, i| {
                const bits = std.mem.readInt(u32, data[begin + i * 4 ..][0..4], .little);
                value.* = @bitCast(bits);
                if (!std.math.isFinite(value.*)) return error.InvalidMemorillaEncoderLora;
            }
            offset += count;
        }
    }
    return values;
}

/// Reads, verifies and packs a hybrid bundle's encoder LoRA (see the module
/// comment for every pin). The bundle's reader files are verified by
/// `Bundle.open` when the research reader cache opens it.
pub fn loadBundle(a: A, directory: []const u8, geometry: Geometry) !Adapter {
    const manifest_path = try std.fs.path.join(a, &.{ directory, "manifest.json" });
    defer a.free(manifest_path);
    const manifest_bytes = try c_file.readFileMax(a, manifest_path, manifest_max_bytes);
    defer a.free(manifest_bytes);
    var manifest = std.json.parseFromSlice(std.json.Value, a, manifest_bytes, .{}) catch return error.InvalidMemorillaManifest;
    defer manifest.deinit();
    const schema = try member(manifest.value, "schema");
    if (schema != .string or !std.mem.eql(u8, schema.string, "antfly.memorilla.bundle.v1")) return error.InvalidMemorillaManifest;
    const release = try member(manifest.value, "release");
    if (release != .bool or release.bool) return error.UnsupportedMemorillaProfile;
    // A research bundle without the pinned adapter cannot serve the hybrid.
    const files = try member(manifest.value, "files");
    if (files != .object or files.object.get(file_name) == null) return error.UnsupportedMemorillaProfile;

    const adapter_bytes = try readPinned(a, directory, file_name, max_file_bytes, manifest.value);
    defer a.free(adapter_bytes);
    const candidate_bytes = try readPinned(a, directory, bundle.research_candidate_file, bundle.research_candidate_max_bytes, manifest.value);
    defer a.free(candidate_bytes);
    const protocol_bytes = try readPinned(a, directory, bundle.research_protocol_file, bundle.research_protocol_max_bytes, manifest.value);
    defer a.free(protocol_bytes);

    const adapter_sha256 = bundle.digest(adapter_bytes);
    const candidate_sha256 = bundle.digest(candidate_bytes);
    const declared = try member(try member(manifest.value, "source"), "candidate_sha256");
    if (declared != .string or !std.mem.eql(u8, declared.string, &candidate_sha256)) return error.MemorillaCandidateMismatch;
    {
        var candidate = std.json.parseFromSlice(std.json.Value, a, candidate_bytes, .{}) catch return error.InvalidMemorillaManifest;
        defer candidate.deinit();
        const pinned = try member(candidate.value, "encoder_lora_sha256");
        if (pinned != .string or !std.mem.eql(u8, pinned.string, &adapter_sha256)) return error.MemorillaCandidateMismatch;
    }
    const protocol = try parseProtocol(a, protocol_bytes);
    const values = try pack(a, adapter_bytes, geometry, protocol.rank);
    return .{
        .allocator = a,
        .values = values,
        .layers = geometry.layers,
        .rank = protocol.rank,
        .scale = protocol.scale,
        .sha256 = adapter_sha256,
        .candidate_sha256 = candidate_sha256,
        .manifest_sha256 = bundle.digest(manifest_bytes),
    };
}

// --- tests ---

const tiny = Geometry{ .layers = 2, .hidden = 3, .q_out = 4, .v_out = 2 };

/// A safetensors file of `tiny` adapters at `rank`, value = tensor index * 100
/// + element index; `skip` omits one tensor, `dtype` overrides every dtype.
fn writeTiny(a: A, rank: usize, skip: ?usize, dtype: []const u8) ![]u8 {
    var header: std.ArrayList(u8) = .empty;
    defer header.deinit(a);
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(a);
    try header.append(a, '{');
    var index: usize = 0;
    var first = true;
    for (0..tiny.layers) |layer| {
        for ([_][]const u8{ "q_proj", "v_proj" }) |proj| {
            for ([_][]const u8{ "A", "B" }) |part| {
                defer index += 1;
                if (skip != null and skip.? == index) continue;
                const rows: usize = if (part[0] == 'A') rank else if (proj[0] == 'q') tiny.q_out else tiny.v_out;
                const cols: usize = if (part[0] == 'A') tiny.hidden else rank;
                const begin = data.items.len;
                for (0..rows * cols) |i| {
                    const value: f32 = @floatFromInt(index * 100 + i);
                    try data.appendSlice(a, std.mem.asBytes(&value));
                }
                if (!first) try header.append(a, ',');
                first = false;
                try header.print(a, "\"layers.{d}.self_attn.{s}.{s}\":{{\"dtype\":\"{s}\",\"shape\":[{d},{d}],\"data_offsets\":[{d},{d}]}}", .{ layer, proj, part, dtype, rows, cols, begin, data.items.len });
            }
        }
    }
    try header.append(a, '}');
    const out = try a.alloc(u8, 8 + header.items.len + data.items.len);
    std.mem.writeInt(u64, out[0..8], header.items.len, .little);
    @memcpy(out[8..][0..header.items.len], header.items);
    @memcpy(out[8 + header.items.len ..], data.items);
    return out;
}

test "memorilla encoder LoRA packs q_A q_B v_A v_B per layer from safetensors" {
    const a = std.testing.allocator;
    const file = try writeTiny(a, 1, null, "F32");
    defer a.free(file);
    const values = try pack(a, file, tiny, 1);
    defer a.free(values);
    const per_layer = layerElements(tiny, 1);
    try std.testing.expectEqual(@as(usize, 3 + 4 + 3 + 2), per_layer);
    try std.testing.expectEqual(tiny.layers * per_layer, values.len);
    // Layer 0: tensors 0 (q.A), 1 (q.B), 2 (v.A), 3 (v.B); layer 1: 4..7.
    try std.testing.expectEqualSlices(f32, &.{ 0, 1, 2, 100, 101, 102, 103, 200, 201, 202, 300, 301 }, values[0..per_layer]);
    try std.testing.expectEqual(@as(f32, 400), values[per_layer]);
    try std.testing.expectEqual(@as(f32, 701), values[2 * per_layer - 1]);
}

test "memorilla encoder LoRA rejects missing tensors wrong dtype rank and truncation" {
    const a = std.testing.allocator;
    const missing = try writeTiny(a, 1, 5, "F32");
    defer a.free(missing);
    try std.testing.expectError(error.InvalidMemorillaEncoderLora, pack(a, missing, tiny, 1));
    const half = try writeTiny(a, 1, null, "F16");
    defer a.free(half);
    try std.testing.expectError(error.InvalidMemorillaEncoderLora, pack(a, half, tiny, 1));
    const good = try writeTiny(a, 1, null, "F32");
    defer a.free(good);
    try std.testing.expectError(error.InvalidMemorillaEncoderLora, pack(a, good, tiny, 2));
    try std.testing.expectError(error.InvalidMemorillaEncoderLora, pack(a, good[0 .. good.len - 4], tiny, 1));
    try std.testing.expectError(error.InvalidMemorillaEncoderLora, pack(a, good[0..6], tiny, 1));
    var other = tiny;
    other.layers = 3;
    try std.testing.expectError(error.InvalidMemorillaEncoderLora, pack(a, good, other, 1));
}

test "memorilla encoder LoRA protocol reads rank alpha and the q/v targets" {
    const a = std.testing.allocator;
    const protocol = try parseProtocol(a, "{\"rank\":8,\"alpha\":16,\"target_modules\":[\"q_proj\",\"v_proj\"],\"system_prompt\":\"x\"}");
    try std.testing.expectEqual(@as(usize, 8), protocol.rank);
    try std.testing.expectEqual(@as(f32, 2.0), protocol.scale);
    try std.testing.expectEqual(@as(f32, 0.5), (try parseProtocol(a, "{\"rank\":4,\"alpha\":2.0,\"target_modules\":[\"q_proj\",\"v_proj\"]}")).scale);
    for ([_][]const u8{
        "{\"alpha\":16,\"target_modules\":[\"q_proj\",\"v_proj\"]}",
        "{\"rank\":0,\"alpha\":16,\"target_modules\":[\"q_proj\",\"v_proj\"]}",
        "{\"rank\":8,\"alpha\":-1,\"target_modules\":[\"q_proj\",\"v_proj\"]}",
        "{\"rank\":8,\"alpha\":16,\"target_modules\":[\"q_proj\",\"k_proj\"]}",
        "{\"rank\":8,\"alpha\":16,\"target_modules\":[\"q_proj\"]}",
        "{\"rank\":8,\"alpha\":16}",
        "[]",
    }) |body| try std.testing.expectError(error.InvalidMemorillaProtocol, parseProtocol(a, body));
}

test "memorilla encoder LoRA bundle pins adapter candidate and manifest digests" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var temp = std.testing.tmpDir(.{});
    defer temp.cleanup();
    const adapter = try writeTiny(a, 1, null, "F32");
    defer a.free(adapter);
    const adapter_sha = bundle.digest(adapter);
    const protocol = "{\"rank\":1,\"alpha\":2,\"target_modules\":[\"q_proj\",\"v_proj\"]}";
    const Write = struct {
        fn files(allocator: A, dir_io: std.Io, dir: std.Io.Dir, adapter_bytes: []const u8, candidate: []const u8, protocol_bytes: []const u8, declared: []const u8, with_adapter: bool) !void {
            try dir.writeFile(dir_io, .{ .sub_path = file_name, .data = adapter_bytes });
            try dir.writeFile(dir_io, .{ .sub_path = "candidate.json", .data = candidate });
            try dir.writeFile(dir_io, .{ .sub_path = "protocol.json", .data = protocol_bytes });
            var manifest: std.ArrayList(u8) = .empty;
            defer manifest.deinit(allocator);
            try manifest.print(allocator, "{{\"schema\":\"antfly.memorilla.bundle.v1\",\"release\":false,\"source\":{{\"candidate_sha256\":\"{s}\"}},\"files\":{{", .{declared});
            if (with_adapter) try manifest.print(allocator, "\"{s}\":{{\"sha256\":\"{s}\",\"size_bytes\":{d}}},", .{ file_name, &bundle.digest(adapter_bytes), adapter_bytes.len });
            try manifest.print(allocator, "\"candidate.json\":{{\"sha256\":\"{s}\",\"size_bytes\":{d}}},\"protocol.json\":{{\"sha256\":\"{s}\",\"size_bytes\":{d}}}}}}}", .{ &bundle.digest(candidate), candidate.len, &bundle.digest(protocol_bytes), protocol_bytes.len });
            try dir.writeFile(dir_io, .{ .sub_path = "manifest.json", .data = manifest.items });
        }
    };
    const root = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", temp.sub_path[0..] });
    defer a.free(root);

    const candidate = try std.fmt.allocPrint(a, "{{\"encoder_lora_sha256\":\"{s}\"}}", .{&adapter_sha});
    defer a.free(candidate);
    try Write.files(a, io, temp.dir, adapter, candidate, protocol, &bundle.digest(candidate), true);
    var loaded = try loadBundle(a, root, tiny);
    defer loaded.deinit();
    try std.testing.expectEqual(@as(usize, 1), loaded.rank);
    try std.testing.expectEqual(@as(f32, 2), loaded.scale);
    try std.testing.expectEqualStrings(&adapter_sha, &loaded.sha256);
    try std.testing.expectEqualStrings(&bundle.digest(candidate), &loaded.candidate_sha256);
    try std.testing.expectEqual(tiny.layers * layerElements(tiny, 1), loaded.values.len);

    // The candidate names a different adapter.
    const other: [64]u8 = @splat('0');
    const stale = try std.fmt.allocPrint(a, "{{\"encoder_lora_sha256\":\"{s}\"}}", .{&other});
    defer a.free(stale);
    try Write.files(a, io, temp.dir, adapter, stale, protocol, &bundle.digest(stale), true);
    try std.testing.expectError(error.MemorillaCandidateMismatch, loadBundle(a, root, tiny));
    // The manifest source names a different candidate.
    try Write.files(a, io, temp.dir, adapter, candidate, protocol, &other, true);
    try std.testing.expectError(error.MemorillaCandidateMismatch, loadBundle(a, root, tiny));
    // A research bundle without the adapter pin is not a hybrid bundle.
    try Write.files(a, io, temp.dir, adapter, candidate, protocol, &bundle.digest(candidate), false);
    try std.testing.expectError(error.UnsupportedMemorillaProfile, loadBundle(a, root, tiny));
    // The file changed after the manifest pinned it.
    try Write.files(a, io, temp.dir, adapter, candidate, protocol, &bundle.digest(candidate), true);
    var mutated = try a.dupe(u8, adapter);
    defer a.free(mutated);
    mutated[mutated.len - 1] ^= 1;
    try temp.dir.writeFile(io, .{ .sub_path = file_name, .data = mutated });
    try std.testing.expectError(error.MemorillaArtifactMismatch, loadBundle(a, root, tiny));
}
