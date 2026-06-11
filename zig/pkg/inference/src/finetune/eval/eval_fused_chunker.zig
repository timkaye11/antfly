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

// Eval binary for the fused chunker-embedder boundary head.
//
// Usage:
//   eval_fused_chunker --data <path> --checkpoint <file> [options]
//
// Options:
//   --data <path>        JSONL eval data path (file or directory)
//   --checkpoint <file>  Checkpoint file written by train_fused_chunker
//   --split <name>       Dataset split filter (default: "val")
//   --batch-size <n>     Batch size (default: 32)
//   --hidden-size <n>    Hidden size (default: 768)
//   --max-seq-len <n>    Max seq len (default: 384)
//   --max-chunks <n>     Max chunks per sample (default: 32)
//   --backend native|metal|auto  Compute backend (default: auto)

const std = @import("std");
const build_options = @import("build_options");
const blas_compute = @import("../../ops/blas_compute.zig");
const metal_compute = if (build_options.enable_metal) @import("../../ops/metal_compute.zig") else struct {};
const gpu_hosted_store = @import("../../ops/gpu_hosted_store.zig");
const metal_runtime = if (build_options.enable_metal) @import("../../backends/metal_runtime.zig") else struct {
    pub fn metalDeviceAvailable() bool {
        return false;
    }
};
const mlx_mod = if (build_options.enable_mlx) @import("../../backends/mlx.zig") else struct {
    pub const c = struct {
        pub fn mlx_map_string_to_array_new() void {}
        pub fn mlx_map_string_to_array_free(_: void) void {}
    };
    pub fn openDefaultStream() struct { stream: void } {
        return .{ .stream = {} };
    }
};
const ops_mod = @import("../../ops/ops.zig");
const ComputeBackend = ops_mod.ComputeBackend;
const fused_chunker_data = @import("../fused_chunker_data.zig");
const fused_chunker_train = @import("../fused_chunker_train.zig");
const modern_bert = @import("../../architectures/modern_bert.zig");
const tokenizer_batch_mod = @import("../tokenizer_batch.zig");
const TokenizerBatch = tokenizer_batch_mod.TokenizerBatch;
const TokenFnCtx = tokenizer_batch_mod.TokenFnCtx;
const weight_source_mod = @import("../../models/weight_source.zig");
const SafetensorsSource = weight_source_mod.SafetensorsSource;
const LoadedWeight = weight_source_mod.LoadedWeight;
const safetensors = @import("../../models/safetensors.zig");
const tensor_mod = @import("../../backends/tensor.zig");
const compat = @import("../../io/compat.zig");
const FusedTrainer = fused_chunker_train.FusedTrainer;
const FusedTrainingConfig = fused_chunker_train.FusedTrainingConfig;
const MetalWeightStore = if (build_options.enable_metal) gpu_hosted_store.WeightStore else void;

const print = std.debug.print;

// ---------------------------------------------------------------------------
// Dense retrieval evaluation metrics
// ---------------------------------------------------------------------------

pub const RetrievalMetrics = struct {
    recall_at_1: f64,
    recall_at_10: f64,
    mrr: f64,
    num_queries: usize,
};

/// Compute retrieval metrics from chunk embeddings.
///
/// embeddings:  [num_chunks * embed_dim] — chunk embeddings (need not be pre-normalised)
/// doc_ids:     [num_chunks] — chunks sharing a doc_id are positives
/// chunk_mask:  [num_chunks] — 1.0 valid, 0.0 padding
/// num_chunks:  total number of chunks (including padding)
/// embed_dim:   embedding dimension
///
/// Algorithm (O(V²) where V = number of valid chunks):
///   1. Filter to valid chunks (chunk_mask > 0).
///   2. L2-normalise each valid embedding.
///   3. For each query i: compute dot-product similarity to all other valid chunks,
///      rank by descending similarity (excluding self), find positives (same doc_id).
///   4. Accumulate Recall@1, Recall@10, MRR; average over queries with at least one positive.
pub fn computeRetrievalMetrics(
    allocator: std.mem.Allocator,
    embeddings: []const f32,
    doc_ids: []const u32,
    chunk_mask: []const f32,
    num_chunks: usize,
    embed_dim: usize,
) !RetrievalMetrics {
    // Step 1: collect valid indices.
    var valid_idx = std.ArrayListUnmanaged(usize).empty;
    defer valid_idx.deinit(allocator);

    for (0..num_chunks) |i| {
        if (chunk_mask[i] > 0.5) {
            try valid_idx.append(allocator, i);
        }
    }
    const V = valid_idx.items.len;

    if (V < 2) {
        return RetrievalMetrics{
            .recall_at_1 = 0,
            .recall_at_10 = 0,
            .mrr = 0,
            .num_queries = 0,
        };
    }

    // Step 2: L2-normalise valid embeddings into a compact [V * embed_dim] buffer.
    const norm_vecs = try allocator.alloc(f32, V * embed_dim);
    defer allocator.free(norm_vecs);

    const compact_doc_ids = try allocator.alloc(u32, V);
    defer allocator.free(compact_doc_ids);

    for (0..V) |ci| {
        const orig_i = valid_idx.items[ci];
        compact_doc_ids[ci] = doc_ids[orig_i];

        const src = embeddings[orig_i * embed_dim .. orig_i * embed_dim + embed_dim];
        const dst = norm_vecs[ci * embed_dim .. ci * embed_dim + embed_dim];

        var sum_sq: f32 = 0;
        for (src) |v| sum_sq += v * v;
        const inv_norm: f32 = if (sum_sq > 1e-24) 1.0 / @sqrt(sum_sq) else 0;
        for (src, dst) |s, *d| d.* = s * inv_norm;
    }

    // Step 3: for each query, compute similarities, rank, accumulate metrics.
    const sims = try allocator.alloc(f32, V);
    defer allocator.free(sims);

    // Reusable scratch buffer for sorting indices.
    const rank_buf = try allocator.alloc(usize, V);
    defer allocator.free(rank_buf);

    var sum_r1: f64 = 0;
    var sum_r10: f64 = 0;
    var sum_mrr: f64 = 0;
    var num_queries: usize = 0;

    for (0..V) |qi| {
        // Check whether this query has any positives.
        var has_positive = false;
        for (0..V) |j| {
            if (j != qi and compact_doc_ids[j] == compact_doc_ids[qi]) {
                has_positive = true;
                break;
            }
        }
        if (!has_positive) continue;

        // Compute dot-product similarities to all other valid chunks.
        const qi_base = qi * embed_dim;
        for (0..V) |j| {
            if (j == qi) {
                sims[j] = -std.math.inf(f32); // exclude self from ranking
                continue;
            }
            const qj_base = j * embed_dim;
            var dot: f32 = 0;
            for (0..embed_dim) |k| {
                dot += norm_vecs[qi_base + k] * norm_vecs[qj_base + k];
            }
            sims[j] = dot;
        }

        // Build rank buffer [0..V] and partial-sort to find the top-10.
        // We use a simple selection approach: track the top-K indices by sim value.
        // For V up to ~512 a full sort is fine.
        for (0..V) |k| rank_buf[k] = k;
        // Sort descending by similarity.
        std.sort.pdq(usize, rank_buf, sims, struct {
            fn lessThan(sim_slice: []const f32, a: usize, b: usize) bool {
                return sim_slice[a] > sim_slice[b]; // descending
            }
        }.lessThan);

        // Recall@1
        if (compact_doc_ids[rank_buf[0]] == compact_doc_ids[qi]) {
            sum_r1 += 1.0;
        }

        // Recall@10 and MRR
        var found_r10 = false;
        var first_positive_rank: usize = 0; // 1-based, 0 means not found yet
        const top_k = @min(V - 1, 10); // at most V-1 non-self results
        for (0..top_k) |rank_idx| {
            const j = rank_buf[rank_idx];
            if (compact_doc_ids[j] == compact_doc_ids[qi]) {
                if (!found_r10) {
                    sum_r10 += 1.0;
                    found_r10 = true;
                }
                if (first_positive_rank == 0) {
                    first_positive_rank = rank_idx + 1; // 1-based
                }
            }
        }
        // If the first positive wasn't in top-10, search the rest for MRR.
        if (first_positive_rank == 0) {
            for (10..V - 1) |rank_idx| {
                const j = rank_buf[rank_idx];
                if (compact_doc_ids[j] == compact_doc_ids[qi]) {
                    first_positive_rank = rank_idx + 1;
                    break;
                }
            }
        }
        if (first_positive_rank > 0) {
            sum_mrr += 1.0 / @as(f64, @floatFromInt(first_positive_rank));
        }

        num_queries += 1;
    }

    if (num_queries == 0) {
        return RetrievalMetrics{
            .recall_at_1 = 0,
            .recall_at_10 = 0,
            .mrr = 0,
            .num_queries = 0,
        };
    }

    const nq_f: f64 = @floatFromInt(num_queries);
    return RetrievalMetrics{
        .recall_at_1 = sum_r1 / nq_f,
        .recall_at_10 = sum_r10 / nq_f,
        .mrr = sum_mrr / nq_f,
        .num_queries = num_queries,
    };
}

const FusedBackend = enum {
    auto,
    metal,
    native,
};

const Options = struct {
    data_path: []const u8,
    checkpoint_path: []const u8,
    model_dir: ?[]const u8 = null,
    split: []const u8 = "val",
    batch_size: u32 = 32,
    hidden_size: u32 = 768,
    num_layers: u32 = 22,
    max_seq_len: u32 = 384,
    max_chunks: u32 = 32,
    intermediate_size: u32 = 1152,
    backend: FusedBackend = .auto,
};

fn printUsage() void {
    print(
        \\Usage: eval-fused-chunker --data <path> --checkpoint <file> [options]
        \\
        \\Options:
        \\  --data <path>            JSONL eval data path (file or directory)
        \\  --checkpoint <file>      Checkpoint file written by train_fused_chunker
        \\  --model-dir <dir>        Model directory (tokenizer + encoder weights)
        \\  --split <name>           Dataset split filter (default: "val")
        \\  --batch-size <n>         Batch size (default: 32)
        \\  --hidden-size <n>        Hidden size (default: 768)
        \\  --num-layers <n>         ModernBERT layer count (default: 22)
        \\  --max-layers <n>         Alias for --num-layers
        \\  --max-seq-len <n>        Max seq len (default: 384)
        \\  --max-chunks <n>         Max chunks per sample (default: 32)
        \\  --intermediate-size <n>  ModernBERT intermediate_size (default: 1152)
        \\  --backend native|metal|auto  Compute backend (default: auto; auto prefers Metal)
        \\
    , .{});
}

fn deinitMetalWeightStore(allocator: std.mem.Allocator, weight_store: *MetalWeightStore) void {
    if (comptime !build_options.enable_metal) return;
    metal_compute.deinitPrefetchQueue(weight_store);
    var it = weight_store.lazy_weights.iterator();
    while (it.next()) |entry| {
        allocator.free(entry.key_ptr.*);
        if (entry.value_ptr.host_loaded) |*loaded| loaded.deinit();
        if (entry.value_ptr.quantized_storage) |*storage| storage.deinit();
    }
    weight_store.lazy_weights.deinit(allocator);
    if (comptime build_options.enable_mlx) {
        _ = mlx_mod.c.mlx_map_string_to_array_free(weight_store.resident_weights);
    }
}

fn deinitNativeWeightStore(allocator: std.mem.Allocator, weight_store: *blas_compute.WeightStore) void {
    var it = weight_store.resident_weights.iterator();
    while (it.next()) |entry| {
        allocator.free(entry.key_ptr.*);
        entry.value_ptr.deinit();
    }
    weight_store.resident_weights.deinit(allocator);
    weight_store.lazy_weights.deinit(allocator);
}

fn stripEncoderPrefix(name: []const u8) []const u8 {
    const prefix = "encoder.";
    if (std.mem.startsWith(u8, name, prefix)) return name[prefix.len..];
    return name;
}

fn normalizeModernBertWeightName(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    const stripped = stripEncoderPrefix(name);
    if (std.mem.startsWith(u8, stripped, "layers.") or
        std.mem.startsWith(u8, stripped, "embeddings.") or
        std.mem.startsWith(u8, stripped, "final_norm."))
    {
        return std.fmt.allocPrint(allocator, "model.{s}", .{stripped});
    }
    return allocator.dupe(u8, stripped);
}

fn cloneLoadedWeight(allocator: std.mem.Allocator, loaded: LoadedWeight) !LoadedWeight {
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

fn modernBertQkvLayer(name: []const u8) ?[]const u8 {
    const prefix = "model.layers.";
    const suffix = ".attn.Wqkv.weight";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    if (!std.mem.endsWith(u8, name, suffix)) return null;
    return name[prefix.len .. name.len - suffix.len];
}

fn copyModernBertQkvPart(
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

fn insertModernBertQkvSplitsIntoNativeStore(
    allocator: std.mem.Allocator,
    weight_store: *blas_compute.WeightStore,
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

fn insertModernBertQkvSplitsIntoMetalStore(
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

const FusedLoRALoadResult = struct {
    count: usize = 0,
    rank: u32 = 0,
    alpha: f32 = 0.0,
};

fn readOptionalScalarF32(reader: *const safetensors.MMapReader, name: []const u8) !?f32 {
    if (!reader.header.tensors.contains(name)) return null;
    var tensor = try reader.readTensor(name);
    defer tensor.deinit();
    if (tensor.dtype != .f32) return error.InvalidFusedLoRAScalar;
    const values = tensor.asFloat32();
    if (values.len != 1) return error.InvalidFusedLoRAScalar;
    return values[0];
}

fn rankFromLoRATensor(name: []const u8, shape: []const i64) !u32 {
    if (shape.len != 2 or shape[0] <= 0 or shape[1] <= 0) return error.InvalidFusedLoRATensor;
    if (std.mem.endsWith(u8, name, ".lora_a")) return @intCast(shape[0]);
    if (std.mem.endsWith(u8, name, ".lora_b")) return @intCast(shape[1]);
    return error.InvalidFusedLoRATensor;
}

const NormalizedLoRATensor = struct {
    key: []const u8,
    values: []const f32,
    shape: [2]i64,
    rank: u32,
    owned_values: ?[]f32 = null,

    fn deinit(self: *NormalizedLoRATensor, allocator: std.mem.Allocator) void {
        if (self.owned_values) |values| allocator.free(values);
        self.* = undefined;
    }
};

fn transpose2DF32Alloc(allocator: std.mem.Allocator, values: []const f32, rows: usize, cols: usize) ![]f32 {
    if (values.len != rows * cols) return error.InvalidFusedLoRATensor;
    const out = try allocator.alloc(f32, values.len);
    for (0..rows) |r| {
        for (0..cols) |c| {
            out[c * rows + r] = values[r * cols + c];
        }
    }
    return out;
}

fn normalizeGoFusedLoRAName(name: []const u8, key_buf: *[256]u8) !?struct {
    key: []const u8,
    is_a: bool,
} {
    var clean = name;
    if (std.mem.startsWith(u8, clean, "var:/")) clean = clean["var:/".len..];
    if (std.mem.startsWith(u8, clean, "fused_chunker_embedder/")) {
        clean = clean["fused_chunker_embedder/".len..];
    }
    if (!std.mem.startsWith(u8, clean, "encoder/layer/")) return null;
    var rest = clean["encoder/layer/".len..];
    const layer_end = std.mem.indexOfScalar(u8, rest, '/') orelse return null;
    const layer = try std.fmt.parseUnsigned(u32, rest[0..layer_end], 10);
    rest = rest[layer_end + 1 ..];

    var scope: []const u8 = undefined;
    var projection: []const u8 = undefined;
    var component: []const u8 = undefined;

    if (std.mem.startsWith(u8, rest, "attn/")) {
        rest = rest["attn/".len..];
        const proj_end = std.mem.indexOfScalar(u8, rest, '/') orelse return null;
        scope = "attn";
        projection = rest[0..proj_end];
        component = rest[proj_end + 1 ..];
    } else if (std.mem.startsWith(u8, rest, "attention/self/")) {
        rest = rest["attention/self/".len..];
        const proj_end = std.mem.indexOfScalar(u8, rest, '/') orelse return null;
        scope = "attn";
        projection = rest[0..proj_end];
        component = rest[proj_end + 1 ..];
    } else if (std.mem.startsWith(u8, rest, "mlp/Wo/")) {
        scope = "mlp";
        projection = "Wo";
        component = rest["mlp/Wo/".len..];
    } else {
        return null;
    }

    const suffix = if (std.mem.eql(u8, component, "lora_A"))
        "lora_a"
    else if (std.mem.eql(u8, component, "lora_B"))
        "lora_b"
    else
        return null;

    return .{
        .key = std.fmt.bufPrint(key_buf, "model.layers.{d}.{s}.{s}.{s}", .{ layer, scope, projection, suffix }) catch return error.NameTooLong,
        .is_a = std.mem.eql(u8, suffix, "lora_a"),
    };
}

fn normalizeFusedLoRATensor(
    allocator: std.mem.Allocator,
    name: []const u8,
    values: []const f32,
    shape: []const i64,
    key_buf: *[256]u8,
) !?NormalizedLoRATensor {
    if (shape.len != 2 or shape[0] <= 0 or shape[1] <= 0) return error.InvalidFusedLoRATensor;

    if (std.mem.endsWith(u8, name, ".lora_a") or std.mem.endsWith(u8, name, ".lora_b")) {
        return .{
            .key = name,
            .values = values,
            .shape = .{ shape[0], shape[1] },
            .rank = try rankFromLoRATensor(name, shape),
        };
    }

    const go_name = (try normalizeGoFusedLoRAName(name, key_buf)) orelse return null;
    const rows: usize = @intCast(shape[0]);
    const cols: usize = @intCast(shape[1]);
    const transposed = try transpose2DF32Alloc(allocator, values, rows, cols);
    return .{
        .key = go_name.key,
        .values = transposed,
        .shape = .{ shape[1], shape[0] },
        .rank = if (go_name.is_a) @intCast(shape[1]) else @intCast(shape[0]),
        .owned_values = transposed,
    };
}

test "eval fused LoRA normalization accepts Go checkpoint tensors" {
    const allocator = std.testing.allocator;

    const a_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const a_shape = [_]i64{ 3, 2 };
    var a_key_buf: [256]u8 = undefined;
    var normalized_a = (try normalizeFusedLoRATensor(
        allocator,
        "var:/fused_chunker_embedder/encoder/layer/7/mlp/Wo/lora_A",
        a_values[0..],
        a_shape[0..],
        &a_key_buf,
    )).?;
    defer normalized_a.deinit(allocator);

    try std.testing.expectEqualStrings("model.layers.7.mlp.Wo.lora_a", normalized_a.key);
    try std.testing.expectEqual(@as(u32, 2), normalized_a.rank);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, normalized_a.shape[0..]);
    try std.testing.expectEqualSlices(f32, &.{ 1, 3, 5, 2, 4, 6 }, normalized_a.values);

    const b_values = [_]f32{ 10, 11, 12, 13, 14, 15, 16, 17 };
    const b_shape = [_]i64{ 2, 4 };
    var b_key_buf: [256]u8 = undefined;
    var normalized_b = (try normalizeFusedLoRATensor(
        allocator,
        "fused_chunker_embedder/encoder/layer/3/attn/query_proj/lora_B",
        b_values[0..],
        b_shape[0..],
        &b_key_buf,
    )).?;
    defer normalized_b.deinit(allocator);

    try std.testing.expectEqualStrings("model.layers.3.attn.query_proj.lora_b", normalized_b.key);
    try std.testing.expectEqual(@as(u32, 2), normalized_b.rank);
    try std.testing.expectEqualSlices(i64, &.{ 4, 2 }, normalized_b.shape[0..]);
    try std.testing.expectEqualSlices(f32, &.{ 10, 14, 11, 15, 12, 16, 13, 17 }, normalized_b.values);

    const old_b_values = [_]f32{ 20, 21, 22, 23 };
    const old_b_shape = [_]i64{ 2, 2 };
    var old_b_key_buf: [256]u8 = undefined;
    var normalized_old_b = (try normalizeFusedLoRATensor(
        allocator,
        "var:/fused_chunker_embedder/encoder/layer/4/attention/self/value_proj/lora_B",
        old_b_values[0..],
        old_b_shape[0..],
        &old_b_key_buf,
    )).?;
    defer normalized_old_b.deinit(allocator);

    try std.testing.expectEqualStrings("model.layers.4.attn.value_proj.lora_b", normalized_old_b.key);
    try std.testing.expectEqual(@as(u32, 2), normalized_old_b.rank);
    try std.testing.expectEqualSlices(i64, &.{ 2, 2 }, normalized_old_b.shape[0..]);
    try std.testing.expectEqualSlices(f32, &.{ 20, 22, 21, 23 }, normalized_old_b.values);

    const zig_values = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const zig_shape = [_]i64{ 2, 3 };
    var zig_key_buf: [256]u8 = undefined;
    var normalized_zig = (try normalizeFusedLoRATensor(
        allocator,
        "model.layers.1.mlp.Wo.lora_a",
        zig_values[0..],
        zig_shape[0..],
        &zig_key_buf,
    )).?;
    defer normalized_zig.deinit(allocator);

    try std.testing.expectEqualStrings("model.layers.1.mlp.Wo.lora_a", normalized_zig.key);
    try std.testing.expectEqual(@as(u32, 2), normalized_zig.rank);
    try std.testing.expectEqualSlices(i64, &.{ 2, 3 }, normalized_zig.shape[0..]);
    try std.testing.expect(normalized_zig.values.ptr == zig_values[0..].ptr);
}

fn insertLoRATensorIntoNativeStore(
    allocator: std.mem.Allocator,
    weight_store: *blas_compute.WeightStore,
    key: []const u8,
    values: []const f32,
    shape: []const i64,
) !void {
    if (shape.len != 2 or shape[0] <= 0 or shape[1] <= 0) return error.InvalidFusedLoRATensor;
    const shape_i64 = [_]i64{ shape[0], shape[1] };
    const owned_key = try allocator.dupe(u8, key);
    errdefer allocator.free(owned_key);
    var tensor = try tensor_mod.Tensor.initFloat32(allocator, owned_key, &shape_i64, values);
    errdefer tensor.deinit();
    try weight_store.resident_weights.put(allocator, owned_key, LoadedWeight{ .tensor = tensor });
}

fn insertLoRATensorIntoMetalStore(
    allocator: std.mem.Allocator,
    weight_store: *MetalWeightStore,
    key: []const u8,
    values: []const f32,
    shape: []const i64,
) !void {
    if (comptime !build_options.enable_metal) return error.MetalBackendUnavailable;
    if (shape.len != 2 or shape[0] <= 0 or shape[1] <= 0) return error.InvalidFusedLoRATensor;
    const shape_i64 = [_]i64{ shape[0], shape[1] };
    const owned_key = try allocator.dupe(u8, key);
    errdefer allocator.free(owned_key);
    var tensor = try tensor_mod.Tensor.initFloat32(allocator, owned_key, &shape_i64, values);
    errdefer tensor.deinit();
    try weight_store.lazy_weights.put(allocator, owned_key, .{
        .tensor_ref = undefined,
        .host_loaded = .{ .tensor = tensor },
        .active_tier = .host,
        .loaded_bytes = tensor.data.len,
    });
}

fn loadFusedLoRAFromCheckpoint(
    allocator: std.mem.Allocator,
    checkpoint_path: []const u8,
    use_metal: bool,
    native_weight_store: *blas_compute.WeightStore,
    metal_weight_store: *MetalWeightStore,
) !FusedLoRALoadResult {
    const file_bytes = try compat.cwd().readFileAlloc(compat.io(), checkpoint_path, allocator, .unlimited);
    var reader = safetensors.MMapReader.fromBytes(allocator, file_bytes) catch |err| {
        allocator.free(file_bytes);
        return switch (err) {
            error.FileTooSmall,
            error.HeaderTooLarge,
            error.EmptyHeader,
            error.FileTruncated,
            error.InvalidHeader,
            error.UnsupportedDType,
            error.MissingShape,
            error.InvalidShape,
            error.MissingOffsets,
            error.InvalidOffset,
            => FusedLoRALoadResult{},
            else => err,
        };
    };
    defer reader.deinit();

    var result = FusedLoRALoadResult{};
    if (try readOptionalScalarF32(&reader, "lora_rank")) |rank_value| {
        if (rank_value < 0) return error.InvalidFusedLoRAScalar;
        result.rank = @intFromFloat(rank_value);
    }
    result.alpha = (try readOptionalScalarF32(&reader, "lora_alpha")) orelse 32.0;

    const names = try reader.header.tensorNames(allocator);
    defer allocator.free(names);

    for (names) |name| {
        if (std.mem.indexOf(u8, name, ".lora_") == null and
            std.mem.indexOf(u8, name, "/lora_") == null)
        {
            continue;
        }
        var tensor = try reader.readTensor(name);
        defer tensor.deinit();
        if (tensor.dtype != .f32 or tensor.shape.len != 2) return error.InvalidFusedLoRATensor;

        var key_buf: [256]u8 = undefined;
        var normalized = (try normalizeFusedLoRATensor(
            allocator,
            name,
            tensor.asFloat32(),
            tensor.shape,
            &key_buf,
        )) orelse continue;
        defer normalized.deinit(allocator);

        const tensor_rank = normalized.rank;
        if (result.rank == 0) {
            result.rank = tensor_rank;
        } else if (result.rank != tensor_rank) {
            return error.FusedLoRARankMismatch;
        }

        const expected_len: usize = @as(usize, @intCast(normalized.shape[0])) * @as(usize, @intCast(normalized.shape[1]));
        if (normalized.values.len != expected_len) return error.InvalidFusedLoRATensor;

        if (use_metal) {
            try insertLoRATensorIntoMetalStore(allocator, metal_weight_store, normalized.key, normalized.values, normalized.shape[0..]);
        } else {
            try insertLoRATensorIntoNativeStore(allocator, native_weight_store, normalized.key, normalized.values, normalized.shape[0..]);
        }
        result.count += 1;
    }

    if (result.count == 0) return FusedLoRALoadResult{};
    if (result.rank == 0 or result.alpha == 0.0) return error.InvalidFusedLoRAScalar;
    return result;
}

fn loadSafetensorsIntoNativeStore(
    allocator: std.mem.Allocator,
    weight_store: *blas_compute.WeightStore,
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
        try weight_store.resident_weights.put(allocator, owned_name, owned_loaded);
        loaded_count += 1;
        loaded_count += try insertModernBertQkvSplitsIntoNativeStore(allocator, weight_store, owned_name, loaded);
    }
    source.weightSource().deinit();
    return loaded_count;
}

fn loadSafetensorsIntoMetalStore(
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

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next(); // skip argv[0]

    var data_path: ?[]const u8 = null;
    var checkpoint_path: ?[]const u8 = null;
    var model_dir: ?[]const u8 = null;
    var split: []const u8 = "val";
    var batch_size: u32 = 32;
    var hidden_size: u32 = 768;
    var num_layers: u32 = 22;
    var max_seq_len: u32 = 384;
    var max_chunks: u32 = 32;
    var intermediate_size: u32 = 1152;
    var backend: @TypeOf((Options{
        .data_path = "",
        .checkpoint_path = "",
    }).backend) = .auto;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--data")) {
            data_path = args.next() orelse {
                print("error: --data requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--checkpoint")) {
            checkpoint_path = args.next() orelse {
                print("error: --checkpoint requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--model-dir")) {
            model_dir = args.next() orelse {
                print("error: --model-dir requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--split")) {
            split = args.next() orelse {
                print("error: --split requires a value\n", .{});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--batch-size")) {
            const value = args.next() orelse {
                print("error: --batch-size requires a value\n", .{});
                std.process.exit(1);
            };
            batch_size = std.fmt.parseUnsigned(u32, value, 10) catch {
                print("error: invalid --batch-size value: {s}\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--hidden-size")) {
            const value = args.next() orelse {
                print("error: --hidden-size requires a value\n", .{});
                std.process.exit(1);
            };
            hidden_size = std.fmt.parseUnsigned(u32, value, 10) catch {
                print("error: invalid --hidden-size value: {s}\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--num-layers") or std.mem.eql(u8, arg, "--max-layers")) {
            const value = args.next() orelse {
                print("error: {s} requires a value\n", .{arg});
                std.process.exit(1);
            };
            num_layers = std.fmt.parseUnsigned(u32, value, 10) catch {
                print("error: invalid {s} value: {s}\n", .{ arg, value });
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--max-seq-len")) {
            const value = args.next() orelse {
                print("error: --max-seq-len requires a value\n", .{});
                std.process.exit(1);
            };
            max_seq_len = std.fmt.parseUnsigned(u32, value, 10) catch {
                print("error: invalid --max-seq-len value: {s}\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--max-chunks")) {
            const value = args.next() orelse {
                print("error: --max-chunks requires a value\n", .{});
                std.process.exit(1);
            };
            max_chunks = std.fmt.parseUnsigned(u32, value, 10) catch {
                print("error: invalid --max-chunks value: {s}\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--intermediate-size")) {
            const value = args.next() orelse {
                print("error: --intermediate-size requires a value\n", .{});
                std.process.exit(1);
            };
            intermediate_size = std.fmt.parseUnsigned(u32, value, 10) catch {
                print("error: invalid --intermediate-size value: {s}\n", .{value});
                std.process.exit(1);
            };
        } else if (std.mem.eql(u8, arg, "--backend")) {
            const value = args.next() orelse {
                print("error: --backend requires a value\n", .{});
                std.process.exit(1);
            };
            if (std.mem.eql(u8, value, "native") or std.mem.eql(u8, value, "blas")) {
                backend = .native;
            } else if (std.mem.eql(u8, value, "metal")) {
                backend = .metal;
            } else if (std.mem.eql(u8, value, "auto")) {
                backend = .auto;
            } else {
                print("error: unknown backend '{s}': expected native, metal, or auto\n", .{value});
                std.process.exit(1);
            }
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return;
        } else {
            print("error: unknown argument: {s}\n", .{arg});
            printUsage();
            std.process.exit(1);
        }
    }

    const opts = Options{
        .data_path = data_path orelse {
            print("error: --data is required\n", .{});
            printUsage();
            std.process.exit(1);
        },
        .checkpoint_path = checkpoint_path orelse {
            print("error: --checkpoint is required\n", .{});
            printUsage();
            std.process.exit(1);
        },
        .model_dir = model_dir,
        .split = split,
        .batch_size = batch_size,
        .hidden_size = hidden_size,
        .num_layers = num_layers,
        .max_seq_len = max_seq_len,
        .max_chunks = max_chunks,
        .intermediate_size = intermediate_size,
        .backend = backend,
    };

    const metal_available = if (comptime build_options.enable_metal) metal_runtime.metalDeviceAvailable() else false;
    const selected_backend: FusedBackend = switch (opts.backend) {
        .auto => if (metal_available) .metal else .native,
        .metal => blk: {
            if (!metal_available) {
                print("error: --backend metal requested but Metal is not compiled in or no Metal device is available\n", .{});
                std.process.exit(1);
            }
            break :blk .metal;
        },
        .native => .native,
    };
    const use_metal = selected_backend == .metal;

    // Set up compute backend for optional encoder forward and boundary-head eval.
    var weight_store = blas_compute.WeightStore{
        .allocator = allocator,
        .resident_weights = .{},
        .lazy_weights = .{},
    };
    defer deinitNativeWeightStore(allocator, &weight_store);

    var blas_backend = blas_compute.BlasCompute.init(allocator, &weight_store, null);

    var metal_weight_store: MetalWeightStore = undefined;
    var metal_backend: if (build_options.enable_metal) metal_compute.MetalCompute else void = undefined;

    const cb: ComputeBackend = if (use_metal) blk: {
        if (comptime build_options.enable_metal) {
            metal_weight_store = MetalWeightStore{
                .allocator = allocator,
                .resident_weights = if (comptime build_options.enable_mlx) mlx_mod.c.mlx_map_string_to_array_new() else {},
                .stream = if (comptime build_options.enable_mlx) mlx_mod.openDefaultStream().stream else {},
                .prefix = "",
                .lazy_weights = .{},
            };
            metal_compute.initPrefetchQueue(&metal_weight_store, allocator);
            metal_backend = try metal_compute.MetalCompute.init(allocator, &metal_weight_store, null);
            break :blk metal_backend.computeBackend();
        } else unreachable;
    } else blas_backend.computeBackend();
    defer if (use_metal) {
        if (comptime build_options.enable_metal) deinitMetalWeightStore(allocator, &metal_weight_store);
    };
    defer if (use_metal) {
        if (comptime build_options.enable_metal) metal_backend.deinit();
    };

    print("backend: {s}\n", .{if (use_metal) "metal" else "native"});

    var tokenizer_opt: ?TokenizerBatch = null;
    defer if (tokenizer_opt) |*tb| tb.deinit();

    var encoder_loaded = false;
    if (opts.model_dir) |mdir| {
        tokenizer_opt = TokenizerBatch.loadFromDir(allocator, mdir, opts.max_seq_len) catch |err| blk: {
            print("warning: could not load tokenizer from {s}: {}\n", .{ mdir, err });
            break :blk null;
        };
        if (tokenizer_opt != null) print("tokenizer loaded from {s}\n", .{mdir});

        var st_path_buf: [512]u8 = undefined;
        const st_path = std.fmt.bufPrint(&st_path_buf, "{s}/model.safetensors", .{mdir}) catch null;
        if (st_path) |p| {
            if (use_metal) {
                if (loadSafetensorsIntoMetalStore(allocator, &metal_weight_store, p)) |loaded_count| {
                    encoder_loaded = true;
                    print("loaded {d} encoder weights into Metal store from {s}\n", .{ loaded_count, p });
                } else |err| {
                    print("warning: could not load encoder weights into Metal store from {s}: {}\n", .{ p, err });
                }
            } else {
                if (loadSafetensorsIntoNativeStore(allocator, &weight_store, p)) |loaded_count| {
                    encoder_loaded = true;
                    print("loaded {d} encoder weights from {s}\n", .{ loaded_count, p });
                } else |err| {
                    print("warning: could not load encoder weights from {s}: {}\n", .{ p, err });
                }
            }
        }
    } else {
        print("model_dir=none (encoder features and retrieval embeddings will be zero-filled)\n", .{});
    }

    // Set up trainer (owns the boundary head weights)
    const config = FusedTrainingConfig{
        .hidden_size = opts.hidden_size,
        .max_seq_len = opts.max_seq_len,
        .max_chunks = opts.max_chunks,
        .batch_size = opts.batch_size,
    };
    var trainer = try FusedTrainer.init(allocator, config, &cb);
    defer trainer.deinit();

    // Load checkpoint
    trainer.loadCheckpoint(allocator, opts.checkpoint_path) catch |err| {
        print("error: failed to load checkpoint '{s}': {}\n", .{ opts.checkpoint_path, err });
        std.process.exit(1);
    };

    var eval_lora_rank: u32 = 0;
    var eval_lora_alpha: f32 = 0.0;
    if (encoder_loaded) {
        const lora_load = loadFusedLoRAFromCheckpoint(
            allocator,
            opts.checkpoint_path,
            use_metal,
            &weight_store,
            &metal_weight_store,
        ) catch |err| blk: {
            print("warning: could not load LoRA tensors from checkpoint '{s}': {}\n", .{ opts.checkpoint_path, err });
            break :blk FusedLoRALoadResult{};
        };
        if (lora_load.count > 0) {
            eval_lora_rank = lora_load.rank;
            eval_lora_alpha = lora_load.alpha;
            print("loaded {d} LoRA tensors from checkpoint (rank={d} alpha={d:.3})\n", .{
                lora_load.count,
                lora_load.rank,
                lora_load.alpha,
            });
        }
    }

    // Load eval samples
    var loaded = fused_chunker_data.loadSamples(allocator, opts.data_path, opts.split) catch |err| {
        print("error: failed to load eval data from '{s}': {}\n", .{ opts.data_path, err });
        std.process.exit(1);
    };
    defer loaded.deinit();

    const samples = loaded.samples;
    if (samples.len == 0) {
        print("error: no samples found in '{s}' for split '{s}'\n", .{ opts.data_path, opts.split });
        std.process.exit(1);
    }

    print("Loaded {d} eval samples from '{s}' (split='{s}')\n", .{ samples.len, opts.data_path, opts.split });

    var boundary_acc = fused_chunker_train.BoundaryEvalAccumulator{};

    // Accumulate chunk-level data for dense retrieval evaluation.
    // chunk_embeddings_all: flat [total_chunks * hs] from mean-pooled encoder features.
    // chunk_mask_all:       flat [total_chunks] — 1.0 valid
    // chunk_doc_ids_all:    flat [total_chunks] — global sample index for each chunk
    var chunk_embeddings_all = std.ArrayListUnmanaged(f32).empty;
    defer chunk_embeddings_all.deinit(allocator);

    var chunk_mask_all = std.ArrayListUnmanaged(f32).empty;
    defer chunk_mask_all.deinit(allocator);

    var chunk_doc_ids_all = std.ArrayListUnmanaged(u32).empty;
    defer chunk_doc_ids_all.deinit(allocator);

    // Batch up samples
    const bs: usize = @intCast(opts.batch_size);
    const msl: usize = @intCast(opts.max_seq_len);
    const mc: usize = @intCast(opts.max_chunks);
    const hs: usize = @intCast(opts.hidden_size);

    // Dummy tokeniser: fills ids/mask with zeros, returns max_seq_len tokens
    const dummy_token_fn = struct {
        fn call(
            _ctx: void,
            text: []const u8,
            out_ids: []i32,
            out_mask: []i32,
            out_offsets: ?[][2]u32,
        ) usize {
            _ = _ctx;
            _ = text;
            @memset(out_ids, 0);
            @memset(out_mask, 1);
            if (out_offsets) |off| @memset(off, .{ 0, 0 });
            return out_ids.len;
        }
    }.call;

    var sample_idx: usize = 0;
    while (sample_idx < samples.len) {
        const end = @min(sample_idx + bs, samples.len);
        const count = end - sample_idx;

        // Build index slice for this batch
        const indices = try allocator.alloc(usize, count);
        defer allocator.free(indices);
        for (0..count) |k| indices[k] = sample_idx + k;

        var batch: fused_chunker_data.FusedBatch = undefined;
        if (tokenizer_opt) |*tb| {
            var tok_ctx = tb.makeTokenFnCtx();
            batch = try fused_chunker_data.assembleTokenBatch(
                allocator,
                samples,
                indices,
                msl,
                mc,
                &tok_ctx,
                TokenFnCtx.call,
            );
        } else {
            batch = try fused_chunker_data.assembleTokenBatch(
                allocator,
                samples,
                indices,
                msl,
                mc,
                {},
                dummy_token_fn,
            );
        }
        defer batch.deinit(allocator);

        // total tokens in this batch = batch_size * max_seq_len (all tokens active)
        const total_tokens: usize = count * msl;

        const features = if (encoder_loaded and tokenizer_opt != null) blk: {
            const ids_i64 = try allocator.alloc(i64, total_tokens);
            defer allocator.free(ids_i64);
            for (batch.input_ids[0..total_tokens], ids_i64) |id32, *id64| id64.* = @intCast(id32);

            const mask_i64 = try allocator.alloc(i64, total_tokens);
            defer allocator.free(mask_i64);
            for (batch.attention_mask[0..total_tokens], mask_i64) |m32, *m64| m64.* = @intCast(m32);

            const bert_config = modern_bert.Config{
                .hidden_size = opts.hidden_size,
                .num_hidden_layers = opts.num_layers,
                .intermediate_size = opts.intermediate_size,
                .lora_rank = eval_lora_rank,
                .lora_alpha = eval_lora_alpha,
            };
            break :blk modern_bert.forward(
                &cb,
                allocator,
                bert_config,
                ids_i64,
                mask_i64,
                count,
                msl,
            ) catch |err| fblk: {
                print("warning: encoder forward failed on eval batch starting at sample {d}: {}\n", .{ sample_idx, err });
                const zeros = try allocator.alloc(f32, total_tokens * hs);
                @memset(zeros, 0.0);
                break :fblk zeros;
            };
        } else zblk: {
            const zeros = try allocator.alloc(f32, total_tokens * hs);
            @memset(zeros, 0.0);
            break :zblk zeros;
        };
        defer allocator.free(features);

        // Build one-hot boundary labels [total_tokens * 2] from flat boundary_labels
        const labels = try allocator.alloc(f32, total_tokens * 2);
        defer allocator.free(labels);
        for (0..total_tokens) |t| {
            const is_boundary = batch.boundary_labels[t] > 0.5;
            labels[t * 2 + 0] = if (is_boundary) 0.0 else 1.0;
            labels[t * 2 + 1] = if (is_boundary) 1.0 else 0.0;
        }

        // Build f32 attention mask [total_tokens] from i32 attention_mask
        const mask = try allocator.alloc(f32, total_tokens);
        defer allocator.free(mask);
        for (0..total_tokens) |t| {
            mask[t] = if (batch.attention_mask[t] != 0) 1.0 else 0.0;
        }

        try trainer.evaluateBatchInto(allocator, &boundary_acc, features, labels, mask, total_tokens);

        const num_chunks_batch = count * mc;
        const chunk_embeddings = try fused_chunker_data.meanPoolChunkEmbeddings(allocator, features, &batch, hs);
        defer allocator.free(chunk_embeddings);
        try chunk_embeddings_all.appendSlice(allocator, chunk_embeddings);
        // chunk_mask from the assembled batch
        try chunk_mask_all.appendSlice(allocator, batch.chunk_mask[0..num_chunks_batch]);
        // doc_ids: the global sample index for each chunk position in the batch.
        // Chunk position c within the batch belongs to sample (sample_idx + c / mc).
        for (0..num_chunks_batch) |c| {
            const global_sample = sample_idx + c / mc;
            try chunk_doc_ids_all.append(allocator, @intCast(global_sample));
        }

        sample_idx = end;
    }

    const summary = boundary_acc.finish();

    // Print results
    print("\n=== Eval Results ===\n", .{});
    print("Checkpoint:  {s}\n", .{opts.checkpoint_path});
    print("Data:        {s}\n", .{opts.data_path});
    print("Batches:     {d}\n", .{summary.num_batches});
    print("F1:          {d:.4}\n", .{summary.boundary_f1});
    print("Precision:   {d:.4}\n", .{summary.boundary_precision});
    print("Recall:      {d:.4}\n", .{summary.boundary_recall});
    print("Counts:      tp={d} fp={d} fn={d}\n", .{ summary.boundary_tp, summary.boundary_fp, summary.boundary_fn });
    print("Best F1:     {d:.4} @ threshold={d:.2}\n", .{ summary.best_boundary_f1, summary.best_boundary_threshold });
    print("Best Prec:   {d:.4}\n", .{summary.best_boundary_precision});
    print("Best Recall: {d:.4}\n", .{summary.best_boundary_recall});
    print("Best Counts: tp={d} fp={d} fn={d}\n", .{ summary.best_boundary_tp, summary.best_boundary_fp, summary.best_boundary_fn });
    print("Valid Tok:   {d}\n", .{summary.valid_tokens});
    print("Gold Pos:    {d} ({d:.6})\n", .{ summary.gold_positives, summary.gold_positive_rate });
    print("Pred Pos:    {d} ({d:.6})\n", .{ summary.predicted_positives, summary.predicted_positive_rate });
    print("Best Pred:   {d} ({d:.6})\n", .{ summary.best_predicted_positives, summary.best_predicted_positive_rate });
    print("Mean P(+):   gold_pos={d:.4} gold_neg={d:.4}\n", .{
        summary.mean_positive_probability_gold_positive,
        summary.mean_positive_probability_gold_negative,
    });
    fused_chunker_train.printBoundaryQualityDiagnostics("eval_quality", summary);

    // Dense retrieval evaluation.
    // Only run when chunk embeddings are non-zero; without --model-dir the eval
    // path intentionally falls back to zero-filled features for boundary-only checks.
    const emb_slice = chunk_embeddings_all.items;
    var emb_sum: f32 = 0;
    for (emb_slice) |v| emb_sum += @abs(v);

    if (emb_sum > 0.0) {
        const total_chunks_all = chunk_mask_all.items.len;
        const retrieval = try computeRetrievalMetrics(
            allocator,
            emb_slice,
            chunk_doc_ids_all.items,
            chunk_mask_all.items,
            total_chunks_all,
            hs,
        );
        print("\n=== Dense Retrieval Metrics ===\n", .{});
        print("Queries:     {d}\n", .{retrieval.num_queries});
        print("Recall@1:    {d:.4}\n", .{retrieval.recall_at_1});
        print("Recall@10:   {d:.4}\n", .{retrieval.recall_at_10});
        print("MRR:         {d:.4}\n", .{retrieval.mrr});
    } else {
        print("\n(Dense retrieval metrics skipped: chunk embeddings are zero-filled)\n", .{});
    }
}
