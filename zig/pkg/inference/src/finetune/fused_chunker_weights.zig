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

// Shared fused-chunker weight-store loading helpers.
//
// Loads ModernBERT-family safetensors files (base encoder weights plus the
// fused-chunker boundary/embedding/SPLADE head tensors exported by
// tools/export_fused_chunker_model.zig) into either a native CPU weight store
// or a Metal-hosted weight store. Weight names are normalized to the serving
// convention used by src/architectures/modern_bert.zig ("model.layers.*", …)
// and fused Wqkv tensors are additionally split into
// query_proj/key_proj/value_proj entries so both fused and split base exports
// load through the same path.
//
// Extracted from src/finetune/eval/eval_fused_chunker.zig so the serving
// pipeline (src/pipelines/fused_chunking.zig) and the eval harness share one
// loader implementation.

const std = @import("std");
const build_options = @import("build_options");
const native_compute = @import("../ops/native_compute.zig");
const metal_compute = if (build_options.enable_metal) @import("../ops/metal_compute.zig") else struct {};
const gpu_hosted_store = @import("../ops/gpu_hosted_store.zig");
const weight_source_mod = @import("../models/weight_source.zig");
const tensor_mod = @import("../backends/tensor.zig");

const SafetensorsSource = weight_source_mod.SafetensorsSource;
const LoadedWeight = weight_source_mod.LoadedWeight;

/// Metal-hosted weight store type. The type itself compiles without Metal
/// enabled (only Metal-specific entry points are comptime-gated), which keeps
/// call sites free of `void` special-casing.
pub const MetalWeightStore = gpu_hosted_store.WeightStore;

pub fn stripEncoderPrefix(name: []const u8) []const u8 {
    const prefix = "encoder.";
    if (std.mem.startsWith(u8, name, prefix)) return name[prefix.len..];
    return name;
}

pub fn normalizeModernBertWeightName(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    const stripped = stripEncoderPrefix(name);
    if (std.mem.startsWith(u8, stripped, "layers.") or
        std.mem.startsWith(u8, stripped, "embeddings.") or
        std.mem.startsWith(u8, stripped, "final_norm."))
    {
        return std.fmt.allocPrint(allocator, "model.{s}", .{stripped});
    }
    return allocator.dupe(u8, stripped);
}

pub fn cloneLoadedWeight(allocator: std.mem.Allocator, loaded: LoadedWeight) !LoadedWeight {
    if (loaded.quantized or loaded.quantized_storage != null) return error.UnsupportedQuantizedEvalWeight;
    const owned_data = try allocator.dupe(u8, loaded.tensor.data);
    errdefer allocator.free(owned_data);
    const owned_shape = try allocator.dupe(i64, loaded.tensor.shape);
    errdefer allocator.free(owned_shape);
    return .{
        .tensor = .{
            .data = owned_data,
            .dtype = loaded.tensor.dtype,
            .shape = owned_shape,
            .name = "",
            .allocator = allocator,
            .owns_data = true,
            .owns_shape = true,
        },
        .quantized = false,
    };
}

pub fn isGoFusedEmbeddingWeightName(name: []const u8) bool {
    return std.mem.eql(u8, name, "model.embeddings.tok_embeddings.weight");
}

pub fn transposeLoadedEmbeddingWeightLikeGoFused(allocator: std.mem.Allocator, loaded: *LoadedWeight) !void {
    if (loaded.tensor.dtype != .f32) return error.UnsupportedGoFusedEmbeddingTensor;
    if (loaded.tensor.shape.len != 2) return error.InvalidGoFusedEmbeddingTensor;
    if (loaded.tensor.shape[0] < 0 or loaded.tensor.shape[1] < 0) return error.InvalidGoFusedEmbeddingTensor;

    const rows: usize = @intCast(loaded.tensor.shape[0]);
    const cols: usize = @intCast(loaded.tensor.shape[1]);
    const values = loaded.tensor.asFloat32();
    if (values.len != rows * cols) return error.InvalidGoFusedEmbeddingTensor;

    const transposed = try allocator.alloc(f32, values.len);
    defer allocator.free(transposed);
    for (0..cols) |r| {
        for (0..rows) |c| {
            transposed[c * cols + r] = values[r * rows + c];
        }
    }

    var tensor = try tensor_mod.Tensor.initFloat32(allocator, loaded.tensor.name, loaded.tensor.shape, transposed);
    errdefer tensor.deinit();
    loaded.tensor.deinit();
    loaded.tensor = tensor;
}

pub fn modernBertQkvLayer(name: []const u8) ?[]const u8 {
    const prefix = "model.layers.";
    const suffix = ".attn.Wqkv.weight";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    if (!std.mem.endsWith(u8, name, suffix)) return null;
    return name[prefix.len .. name.len - suffix.len];
}

pub fn copyModernBertQkvPart(
    allocator: std.mem.Allocator,
    loaded: LoadedWeight,
    part: usize,
) !struct { data: []f32, hidden: usize } {
    if (loaded.tensor.dtype != .f32) return error.UnsupportedModernBertQkvTensor;
    if (loaded.tensor.shape.len != 2) return error.InvalidModernBertQkvTensor;
    if (loaded.tensor.shape[0] < 0 or loaded.tensor.shape[1] < 0) return error.InvalidModernBertQkvTensor;
    const rows: usize = @intCast(loaded.tensor.shape[0]);
    const hidden: usize = @intCast(loaded.tensor.shape[1]);
    if (rows != hidden * 3) return error.InvalidModernBertQkvTensor;

    const elems = hidden * hidden;
    const elem_start = part * elems;
    const byte_start = elem_start * @sizeOf(f32);
    const byte_end = byte_start + elems * @sizeOf(f32);
    if (byte_end > loaded.tensor.data.len) return error.InvalidModernBertQkvTensor;

    const data = try allocator.alloc(f32, elems);
    errdefer allocator.free(data);
    @memcpy(std.mem.sliceAsBytes(data), loaded.tensor.data[byte_start..byte_end]);
    return .{ .data = data, .hidden = hidden };
}

pub fn insertModernBertQkvSplitsIntoNativeStore(
    allocator: std.mem.Allocator,
    weight_store: *native_compute.WeightStore,
    normalized_name: []const u8,
    loaded: LoadedWeight,
) !usize {
    const layer = modernBertQkvLayer(normalized_name) orelse return 0;
    const projections = [_][]const u8{ "query_proj", "key_proj", "value_proj" };
    var inserted: usize = 0;
    for (projections, 0..) |projection, part| {
        const split = try copyModernBertQkvPart(allocator, loaded, part);
        defer allocator.free(split.data);
        const split_name = try std.fmt.allocPrint(allocator, "model.layers.{s}.attn.{s}.weight", .{ layer, projection });
        errdefer allocator.free(split_name);
        const shape = [_]i64{ @intCast(split.hidden), @intCast(split.hidden) };
        var tensor = try tensor_mod.Tensor.initFloat32(allocator, split_name, &shape, split.data);
        errdefer tensor.deinit();
        try weight_store.resident_weights.put(allocator, split_name, weight_source_mod.LoadedWeight{ .tensor = tensor });
        inserted += 1;
    }
    return inserted;
}

pub fn insertModernBertQkvSplitsIntoMetalStore(
    allocator: std.mem.Allocator,
    weight_store: *MetalWeightStore,
    normalized_name: []const u8,
    loaded: LoadedWeight,
) !usize {
    if (comptime !build_options.enable_metal) return error.MetalBackendUnavailable;
    const layer = modernBertQkvLayer(normalized_name) orelse return 0;
    const projections = [_][]const u8{ "query_proj", "key_proj", "value_proj" };
    var inserted: usize = 0;
    for (projections, 0..) |projection, part| {
        const split = try copyModernBertQkvPart(allocator, loaded, part);
        defer allocator.free(split.data);
        const split_name = try std.fmt.allocPrint(allocator, "model.layers.{s}.attn.{s}.weight", .{ layer, projection });
        errdefer allocator.free(split_name);
        const shape = [_]i64{ @intCast(split.hidden), @intCast(split.hidden) };
        var tensor = try tensor_mod.Tensor.initFloat32(allocator, split_name, &shape, split.data);
        errdefer tensor.deinit();
        try weight_store.lazy_weights.put(allocator, split_name, .{
            .tensor_ref = undefined,
            .host_loaded = .{ .tensor = tensor },
            .active_tier = .host,
            .loaded_bytes = tensor.data.len,
        });
        inserted += 1;
    }
    return inserted;
}

pub fn loadSafetensorsIntoNativeStore(
    allocator: std.mem.Allocator,
    weight_store: *native_compute.WeightStore,
    st_path: []const u8,
) !usize {
    var source = try SafetensorsSource.initAbsolute(allocator, st_path);
    errdefer source.weightSource().deinit();
    const ws = source.weightSource();
    const names = try ws.listNames(allocator);
    defer allocator.free(names);

    var loaded_count: usize = 0;
    for (names) |name| {
        var loaded = ws.getTensor(name) catch continue;
        defer loaded.deinit();
        var owned_loaded = try cloneLoadedWeight(allocator, loaded);
        errdefer owned_loaded.deinit();
        const owned_name = try normalizeModernBertWeightName(allocator, name);
        errdefer allocator.free(owned_name);
        if (isGoFusedEmbeddingWeightName(owned_name)) {
            try transposeLoadedEmbeddingWeightLikeGoFused(allocator, &owned_loaded);
        }
        try weight_store.resident_weights.put(allocator, owned_name, owned_loaded);
        loaded_count += 1;
        loaded_count += try insertModernBertQkvSplitsIntoNativeStore(allocator, weight_store, owned_name, loaded);
    }
    source.weightSource().deinit();
    return loaded_count;
}

pub fn loadSafetensorsIntoMetalStore(
    allocator: std.mem.Allocator,
    weight_store: *MetalWeightStore,
    st_path: []const u8,
) !usize {
    if (comptime !build_options.enable_metal) return error.MetalBackendUnavailable;
    var source = try SafetensorsSource.initAbsolute(allocator, st_path);
    errdefer source.weightSource().deinit();
    const ws = source.weightSource();
    const names = try ws.listNames(allocator);
    defer allocator.free(names);

    var loaded_count: usize = 0;
    for (names) |name| {
        var loaded = ws.getTensor(name) catch continue;
        defer loaded.deinit();
        var owned_loaded = try cloneLoadedWeight(allocator, loaded);
        errdefer owned_loaded.deinit();
        const owned_name = try normalizeModernBertWeightName(allocator, name);
        errdefer allocator.free(owned_name);
        if (isGoFusedEmbeddingWeightName(owned_name)) {
            try transposeLoadedEmbeddingWeightLikeGoFused(allocator, &owned_loaded);
        }
        try weight_store.lazy_weights.put(allocator, owned_name, .{
            .tensor_ref = undefined,
            .host_loaded = owned_loaded,
            .active_tier = .host,
            .loaded_bytes = owned_loaded.tensor.data.len,
        });
        loaded_count += 1;
        loaded_count += try insertModernBertQkvSplitsIntoMetalStore(allocator, weight_store, owned_name, loaded);
    }
    source.weightSource().deinit();
    return loaded_count;
}

pub fn deinitNativeWeightStore(allocator: std.mem.Allocator, weight_store: *native_compute.WeightStore) void {
    native_compute.deinitPrefetchQueue(weight_store);
    var it = weight_store.resident_weights.iterator();
    while (it.next()) |entry| {
        allocator.free(entry.key_ptr.*);
        entry.value_ptr.deinit();
    }
    weight_store.resident_weights.deinit(allocator);
    weight_store.lazy_weights.deinit(allocator);
}

pub fn deinitMetalWeightStore(allocator: std.mem.Allocator, weight_store: *MetalWeightStore) void {
    if (comptime build_options.enable_metal) metal_compute.deinitPrefetchQueue(weight_store);
    var it = weight_store.lazy_weights.iterator();
    while (it.next()) |entry| {
        allocator.free(entry.key_ptr.*);
        if (entry.value_ptr.host_loaded) |*loaded| loaded.deinit();
        if (entry.value_ptr.quantized_storage) |*storage| storage.deinit();
    }
    weight_store.lazy_weights.deinit(allocator);
}

test "normalizeModernBertWeightName maps names to serving convention" {
    const allocator = std.testing.allocator;

    const already = try normalizeModernBertWeightName(allocator, "model.layers.0.attn.Wqkv.weight");
    defer allocator.free(already);
    try std.testing.expectEqualStrings("model.layers.0.attn.Wqkv.weight", already);

    const prefixed = try normalizeModernBertWeightName(allocator, "encoder.layers.1.mlp.Wo.weight");
    defer allocator.free(prefixed);
    try std.testing.expectEqualStrings("model.layers.1.mlp.Wo.weight", prefixed);

    const bare = try normalizeModernBertWeightName(allocator, "final_norm.weight");
    defer allocator.free(bare);
    try std.testing.expectEqualStrings("model.final_norm.weight", bare);

    const head = try normalizeModernBertWeightName(allocator, "fused_chunker_embedder/boundary_head/mlp_dense1/weight");
    defer allocator.free(head);
    try std.testing.expectEqualStrings("fused_chunker_embedder/boundary_head/mlp_dense1/weight", head);
}

test "modernBertQkvLayer extracts the layer index" {
    try std.testing.expectEqualStrings("7", modernBertQkvLayer("model.layers.7.attn.Wqkv.weight").?);
    try std.testing.expectEqual(@as(?[]const u8, null), modernBertQkvLayer("model.layers.7.attn.Wo.weight"));
    try std.testing.expectEqual(@as(?[]const u8, null), modernBertQkvLayer("w1"));
}
