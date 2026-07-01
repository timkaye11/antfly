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

// Export a trained fused-chunker checkpoint into a deployable serving model
// directory for the inference service (~/.antfly/inference/models/chunkers/<name>).
//
// Output layout:
//   model.safetensors      — base ModernBERT weights with LoRA adapters merged
//                            (W' = W + B·A·(alpha/rank)) plus the boundary-head
//                            MLP under the serving names used by
//                            src/finetune/fused_chunker.zig, and the optional
//                            embedding-projection / SPLADE heads if present in
//                            the checkpoint.
//   tokenizer.json         — copied from --model-dir
//   tokenizer_config.json  — copied from --model-dir (optional)
//   config.json            — copied from --model-dir
//   model_manifest.json    — chunker manifest parsed by src/models/manifest.zig
//
// Tensor name mapping (checkpoint → serving):
//   w1 [mlp,H], b1, w2 [2,mlp], b2
//     → fused_chunker_embedder/boundary_head/mlp_dense1/{weight,bias}
//       fused_chunker_embedder/boundary_head/mlp_dense2/{weight,bias}
//       (row-major [out,in] preserved — cb.linear transposes internally)
//   model.layers.{L}.attn.{query_proj,key_proj,value_proj}.lora_{a,b}
//     → merged into model.layers.{L}.attn.Wqkv.weight (q/k/v slice) or the
//       split model.layers.{L}.attn.{proj}.weight when the base stores splits
//   model.layers.{L}.attn.Wo.lora_{a,b} → merged into model.layers.{L}.attn.Wo.weight
//   model.layers.{L}.mlp.Wo.lora_{a,b}  → merged into model.layers.{L}.mlp.Wo.weight
//   lora_rank / lora_alpha              → consumed (merge scale), not emitted
//
// The tool is CPU-only: pure tensor arithmetic and file IO. Tensors are
// processed one at a time (mmap-backed reads, streaming writes) so peak
// memory stays near the largest single tensor.

const std = @import("std");
const inference = @import("inference_internal");
const compat = inference.io.compat;
const safetensors = inference.models.safetensors;
const fused_chunker = inference.finetune.fused_chunker;
const fused_chunker_splade = inference.finetune.fused_chunker_splade;
const lora = inference.finetune.lora;

const DType = @FieldType(safetensors.TensorMeta, "dtype");

const default_max_seq_len: u32 = 384;
const default_lora_alpha: f32 = 32.0;

const boundary_dense1_weight_name = "fused_chunker_embedder/boundary_head/mlp_dense1/weight";
const boundary_dense1_bias_name = "fused_chunker_embedder/boundary_head/mlp_dense1/bias";
const boundary_dense2_weight_name = "fused_chunker_embedder/boundary_head/mlp_dense2/weight";
const boundary_dense2_bias_name = "fused_chunker_embedder/boundary_head/mlp_dense2/bias";
const embedding_proj_weight_name = "fused_chunker_embedder/embedding_head/proj/weight";
const embedding_proj_bias_name = "fused_chunker_embedder/embedding_head/proj/bias";
const splade_bias_name = "fused_chunker_embedder/splade_head/proj/bias";

const Options = struct {
    checkpoint: []const u8,
    model_dir: []const u8,
    out: []const u8,
    model_version: ?[]const u8 = null,
    boundary_threshold: f32 = 0.5,
    force: bool = false,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const opts = try parseOptions(&args) orelse return;
    const stats = try exportModel(allocator, opts);
    try writeSummary(allocator, opts, stats);
}

fn parseOptions(args: *std.process.Args.Iterator) !?Options {
    var opts: Options = .{
        .checkpoint = "",
        .model_dir = "",
        .out = "",
    };
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--checkpoint")) {
            opts.checkpoint = args.next() orelse return error.MissingCheckpoint;
        } else if (std.mem.eql(u8, arg, "--model-dir")) {
            opts.model_dir = args.next() orelse return error.MissingModelDir;
        } else if (std.mem.eql(u8, arg, "--out")) {
            opts.out = args.next() orelse return error.MissingOut;
        } else if (std.mem.eql(u8, arg, "--model-version")) {
            opts.model_version = args.next() orelse return error.MissingModelVersion;
        } else if (std.mem.eql(u8, arg, "--boundary-threshold")) {
            const value = args.next() orelse return error.MissingBoundaryThreshold;
            opts.boundary_threshold = std.fmt.parseFloat(f32, value) catch return error.InvalidBoundaryThreshold;
        } else if (std.mem.eql(u8, arg, "--force")) {
            opts.force = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            usage();
            return null;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }

    if (opts.checkpoint.len == 0 or opts.model_dir.len == 0 or opts.out.len == 0) {
        usage();
        return error.InvalidArgument;
    }
    if (!(opts.boundary_threshold > 0.0 and opts.boundary_threshold < 1.0)) {
        std.debug.print("error: --boundary-threshold must be in (0, 1)\n", .{});
        return error.InvalidBoundaryThreshold;
    }
    return opts;
}

fn usage() void {
    std.debug.print(
        \\usage: export-fused-chunker-model --checkpoint <checkpoint_final.safetensors> --model-dir <base modernbert dir> --out <dir> [--model-version <str>] [--boundary-threshold <f>] [--force]
        \\
        \\Packages a trained fused-chunker checkpoint into a deployable model
        \\directory (model.safetensors with LoRA merged + boundary head,
        \\tokenizer files, config.json, model_manifest.json).
        \\
    , .{});
}

// ---------------------------------------------------------------------------
// Checkpoint inspection
// ---------------------------------------------------------------------------

const LoRATarget = enum {
    qkv_query,
    qkv_key,
    qkv_value,
    attn_wo,
    mlp_wo,

    fn qkvPart(self: LoRATarget) ?usize {
        return switch (self) {
            .qkv_query => 0,
            .qkv_key => 1,
            .qkv_value => 2,
            else => null,
        };
    }
};

const ParsedLoRAKey = struct {
    layer: u32,
    target: LoRATarget,
    is_a: bool,
};

/// Parse a Zig fused-chunker checkpoint LoRA tensor name:
///   model.layers.{L}.attn.{query_proj|key_proj|value_proj|Wo}.lora_{a,b}
///   model.layers.{L}.mlp.Wo.lora_{a,b}
fn parseCheckpointLoRAKey(name: []const u8) ?ParsedLoRAKey {
    const prefix = "model.layers.";
    if (!std.mem.startsWith(u8, name, prefix)) return null;
    var rest = name[prefix.len..];

    const layer_end = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const layer = std.fmt.parseUnsigned(u32, rest[0..layer_end], 10) catch return null;
    rest = rest[layer_end + 1 ..];

    const scope_end = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const scope = rest[0..scope_end];
    rest = rest[scope_end + 1 ..];

    const proj_end = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    const projection = rest[0..proj_end];
    const leaf = rest[proj_end + 1 ..];

    const is_a = if (std.mem.eql(u8, leaf, "lora_a"))
        true
    else if (std.mem.eql(u8, leaf, "lora_b"))
        false
    else
        return null;

    const target: LoRATarget = if (std.mem.eql(u8, scope, "attn")) blk: {
        if (std.mem.eql(u8, projection, "query_proj")) break :blk .qkv_query;
        if (std.mem.eql(u8, projection, "key_proj")) break :blk .qkv_key;
        if (std.mem.eql(u8, projection, "value_proj")) break :blk .qkv_value;
        if (std.mem.eql(u8, projection, "Wo")) break :blk .attn_wo;
        return null;
    } else if (std.mem.eql(u8, scope, "mlp") and std.mem.eql(u8, projection, "Wo"))
        .mlp_wo
    else
        return null;

    return .{ .layer = layer, .target = target, .is_a = is_a };
}

/// Map optional head tensors that may already carry serving-style names
/// (embedding projection, SPLADE) to their canonical serving names.
fn passthroughServingName(name: []const u8) ?[]const u8 {
    var clean = name;
    if (std.mem.startsWith(u8, clean, "var:/")) clean = clean["var:/".len..];
    if (std.mem.startsWith(u8, clean, "fused_chunker_embedder/")) {
        clean = clean["fused_chunker_embedder/".len..];
    }
    if (std.mem.eql(u8, clean, "embedding_head/proj/weight")) return embedding_proj_weight_name;
    if (std.mem.eql(u8, clean, "embedding_head/proj/bias")) return embedding_proj_bias_name;
    if (std.mem.eql(u8, clean, "splade_head/proj/weight")) return fused_chunker_splade.SPLADE_WEIGHT_KEY;
    if (std.mem.eql(u8, clean, "splade_head/proj/bias")) return splade_bias_name;
    return null;
}

const LoRAPair = struct {
    layer: u32,
    target: LoRATarget,
    a_name: ?[]const u8 = null,
    b_name: ?[]const u8 = null,
};

const CheckpointContents = struct {
    // Boundary head (required): checkpoint tensor names.
    w1: []const u8 = "",
    b1: []const u8 = "",
    w2: []const u8 = "",
    b2: []const u8 = "",
    // Optional passthrough head tensors: checkpoint name → serving name.
    passthrough: std.ArrayListUnmanaged(struct { src: []const u8, serving: []const u8 }) = .empty,
    // LoRA adapter pairs.
    pairs: std.ArrayListUnmanaged(LoRAPair) = .empty,
    lora_rank: u32 = 0,
    lora_alpha: f32 = default_lora_alpha,
    has_splade: bool = false,
    has_embedding_proj: bool = false,

    fn deinit(self: *CheckpointContents, allocator: std.mem.Allocator) void {
        self.passthrough.deinit(allocator);
        self.pairs.deinit(allocator);
    }

    fn findPair(self: *CheckpointContents, layer: u32, target: LoRATarget) ?*LoRAPair {
        for (self.pairs.items) |*pair| {
            if (pair.layer == layer and pair.target == target) return pair;
        }
        return null;
    }
};

/// Read an f32 tensor into a freshly-allocated aligned buffer.
///
/// MMapReader.readTensor returns borrowed views into the mapped file whose
/// data section is not guaranteed to be 4-byte aligned, so Tensor.asFloat32
/// cannot be used directly on mmap-backed tensors.
fn readTensorF32Alloc(
    allocator: std.mem.Allocator,
    reader: *const safetensors.MMapReader,
    name: []const u8,
) ![]f32 {
    var tensor = try reader.readTensor(name);
    defer tensor.deinit();
    if (tensor.dtype != .f32) return error.InvalidCheckpointTensor;
    if (tensor.data.len % @sizeOf(f32) != 0) return error.InvalidCheckpointTensor;
    const values = try allocator.alloc(f32, tensor.data.len / @sizeOf(f32));
    errdefer allocator.free(values);
    @memcpy(std.mem.sliceAsBytes(values), tensor.data);
    return values;
}

fn readScalarF32(allocator: std.mem.Allocator, reader: *const safetensors.MMapReader, name: []const u8) !?f32 {
    if (!reader.header.tensors.contains(name)) return null;
    const values = readTensorF32Alloc(allocator, reader, name) catch return error.InvalidCheckpointScalar;
    defer allocator.free(values);
    if (values.len != 1) return error.InvalidCheckpointScalar;
    return values[0];
}

fn inspectCheckpoint(
    allocator: std.mem.Allocator,
    reader: *const safetensors.MMapReader,
) !CheckpointContents {
    var contents = CheckpointContents{};
    errdefer contents.deinit(allocator);

    if (try readScalarF32(allocator, reader, "lora_rank")) |rank_value| {
        if (rank_value < 0) return error.InvalidCheckpointScalar;
        contents.lora_rank = @intFromFloat(rank_value);
    }
    if (try readScalarF32(allocator, reader, "lora_alpha")) |alpha_value| {
        contents.lora_alpha = alpha_value;
    }

    const names = try reader.header.tensorNames(allocator);
    defer allocator.free(names);

    var tensor_rank: u32 = 0;
    for (names) |name| {
        if (std.mem.eql(u8, name, "lora_rank") or std.mem.eql(u8, name, "lora_alpha")) continue;
        const meta = reader.header.tensors.get(name).?;
        if (meta.dtype != .f32) {
            std.debug.print("error: checkpoint tensor '{s}' is not f32\n", .{name});
            return error.InvalidCheckpointTensor;
        }

        if (std.mem.eql(u8, name, "w1")) {
            if (meta.shape.len != 2) return error.InvalidCheckpointTensor;
            contents.w1 = name;
            continue;
        }
        if (std.mem.eql(u8, name, "b1")) {
            if (meta.shape.len != 1) return error.InvalidCheckpointTensor;
            contents.b1 = name;
            continue;
        }
        if (std.mem.eql(u8, name, "w2")) {
            if (meta.shape.len != 2) return error.InvalidCheckpointTensor;
            contents.w2 = name;
            continue;
        }
        if (std.mem.eql(u8, name, "b2")) {
            if (meta.shape.len != 1) return error.InvalidCheckpointTensor;
            contents.b2 = name;
            continue;
        }

        if (parseCheckpointLoRAKey(name)) |parsed| {
            if (meta.shape.len != 2 or meta.shape[0] <= 0 or meta.shape[1] <= 0) {
                return error.InvalidCheckpointTensor;
            }
            const this_rank: u32 = if (parsed.is_a) @intCast(meta.shape[0]) else @intCast(meta.shape[1]);
            if (tensor_rank == 0) {
                tensor_rank = this_rank;
            } else if (tensor_rank != this_rank) {
                return error.LoRARankMismatch;
            }
            if (contents.findPair(parsed.layer, parsed.target)) |pair| {
                if (parsed.is_a) pair.a_name = name else pair.b_name = name;
            } else {
                var pair = LoRAPair{ .layer = parsed.layer, .target = parsed.target };
                if (parsed.is_a) pair.a_name = name else pair.b_name = name;
                try contents.pairs.append(allocator, pair);
            }
            continue;
        }

        if (passthroughServingName(name)) |serving| {
            try contents.passthrough.append(allocator, .{ .src = name, .serving = serving });
            if (std.mem.eql(u8, serving, fused_chunker_splade.SPLADE_WEIGHT_KEY) or
                std.mem.eql(u8, serving, splade_bias_name))
            {
                contents.has_splade = true;
            } else {
                contents.has_embedding_proj = true;
            }
            continue;
        }

        std.debug.print("error: unmapped checkpoint tensor '{s}'\n", .{name});
        return error.UnmappedCheckpointTensor;
    }

    if (contents.w1.len == 0 or contents.b1.len == 0 or contents.w2.len == 0 or contents.b2.len == 0) {
        std.debug.print("error: checkpoint is missing boundary head tensors (w1/b1/w2/b2)\n", .{});
        return error.MissingBoundaryHead;
    }

    for (contents.pairs.items) |pair| {
        if (pair.a_name == null or pair.b_name == null) {
            std.debug.print(
                "error: incomplete LoRA pair for layer {d} target {s}\n",
                .{ pair.layer, @tagName(pair.target) },
            );
            return error.IncompleteLoRAPair;
        }
    }

    if (contents.pairs.items.len > 0) {
        if (contents.lora_rank == 0) contents.lora_rank = tensor_rank;
        if (contents.lora_rank != tensor_rank) return error.LoRARankMismatch;
        if (contents.lora_rank == 0 or contents.lora_alpha == 0.0) return error.InvalidCheckpointScalar;
    }

    return contents;
}

// ---------------------------------------------------------------------------
// LoRA merge
// ---------------------------------------------------------------------------

/// Merge a LoRA delta into one [out_features, in_features] row-major slice of
/// `buf` starting at part index `part` (for fused Wqkv tensors parts 0/1/2 are
/// the query/key/value blocks; standalone weights use part 0).
///
///   W' = W + B·A·(alpha/rank)
///
/// a: [rank, in_features] row-major, b: [out_features, rank] row-major.
/// Reuses lora.mergeInto (delta = adapter_a·adapter_b with adapter_a = B and
/// adapter_b = A) through a scratch buffer since mergeInto requires a
/// non-aliased output.
fn mergeLoRAPart(
    allocator: std.mem.Allocator,
    buf: []f32,
    out_features: usize,
    in_features: usize,
    part: usize,
    a: []const f32,
    b: []const f32,
    rank: usize,
    alpha: f32,
) !void {
    const elems = out_features * in_features;
    if (a.len != rank * in_features) return error.InvalidLoRAShape;
    if (b.len != out_features * rank) return error.InvalidLoRAShape;
    if ((part + 1) * elems > buf.len) return error.InvalidLoRAShape;

    const slice = buf[part * elems ..][0..elems];
    const scratch = try allocator.alloc(f32, elems);
    defer allocator.free(scratch);

    lora.mergeInto(
        .{ .rows = out_features, .cols = in_features, .data = slice },
        .{ .rows = out_features, .cols = rank, .data = b },
        .{ .rows = rank, .cols = in_features, .data = a },
        alpha,
        scratch,
    );
    @memcpy(slice, scratch);
}

// ---------------------------------------------------------------------------
// Base weight name normalization (mirrors eval_fused_chunker.zig)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Output plan
// ---------------------------------------------------------------------------

const MergeOp = struct {
    part: usize,
    out_features: usize,
    in_features: usize,
    a_name: []const u8,
    b_name: []const u8,
};

const Source = union(enum) {
    /// Copy raw bytes of a base tensor (name in the base file).
    base_raw: []const u8,
    /// Base tensor with one or more LoRA deltas merged in.
    merged: struct {
        base_name: []const u8,
        ops: []const MergeOp,
    },
    /// Copy raw bytes of a checkpoint tensor (name in the checkpoint file).
    checkpoint_raw: []const u8,
};

const PlannedTensor = struct {
    out_name: []const u8,
    dtype: DType,
    shape: []const i64,
    byte_len: u64,
    source: Source,
};

fn dtypeName(d: DType) []const u8 {
    return switch (d) {
        .f32 => "F32",
        .f16 => "F16",
        .bf16 => "BF16",
        .f64 => "F64",
        .i8 => "I8",
        .i16 => "I16",
        .i32 => "I32",
        .i64 => "I64",
        .u8 => "U8",
        else => "F32",
    };
}

fn plannedTensorLessThan(_: void, a: PlannedTensor, b: PlannedTensor) bool {
    return std.mem.lessThan(u8, a.out_name, b.out_name);
}

const ExportStats = struct {
    base_tensors: usize = 0,
    merged_tensors: usize = 0,
    lora_pairs_merged: usize = 0,
    head_tensors: usize = 0,
    lora_rank: u32 = 0,
    lora_alpha: f32 = 0.0,
    hidden_size: u32 = 0,
    embedding_dim: u32 = 0,
    splade: bool = false,
    model_version_buf: [256]u8 = undefined,
    model_version_len: usize = 0,
    checkpoint_sha256: [64]u8 = undefined,

    fn modelVersion(self: *const ExportStats) []const u8 {
        return self.model_version_buf[0..self.model_version_len];
    }
};

// ---------------------------------------------------------------------------
// Export
// ---------------------------------------------------------------------------

fn exportModel(allocator: std.mem.Allocator, opts: Options) !ExportStats {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var stats = ExportStats{};

    // Refuse to clobber an existing export unless --force.
    if (!opts.force) {
        const guard_paths = [_][]const u8{ "model.safetensors", "model_manifest.json" };
        for (guard_paths) |leaf| {
            const path = try std.fmt.allocPrint(arena, "{s}/{s}", .{ opts.out, leaf });
            if (compat.cwd().statFile(compat.io(), path, .{})) |_| {
                std.debug.print("error: {s} already exists (use --force to overwrite)\n", .{path});
                return error.OutputExists;
            } else |_| {}
        }
    }

    // --- Checkpoint -------------------------------------------------------
    var checkpoint_reader = try safetensors.MMapReader.openFileAbsolute(allocator, opts.checkpoint);
    defer checkpoint_reader.deinit();

    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(checkpoint_reader.file_bytes, &digest, .{});
    const sha_hex = std.fmt.bytesToHex(digest, .lower);
    @memcpy(stats.checkpoint_sha256[0..], sha_hex[0..]);

    var contents = try inspectCheckpoint(allocator, &checkpoint_reader);
    defer contents.deinit(allocator);

    stats.lora_rank = contents.lora_rank;
    stats.lora_alpha = if (contents.pairs.items.len > 0) contents.lora_alpha else 0.0;
    stats.splade = contents.has_splade;

    const w1_meta = checkpoint_reader.header.tensors.get(contents.w1).?;
    stats.hidden_size = @intCast(w1_meta.shape[1]);
    stats.embedding_dim = stats.hidden_size;
    if (contents.has_embedding_proj) {
        if (checkpoint_reader.header.tensors.get("embedding_head/proj/weight") orelse
            checkpoint_reader.header.tensors.get(embedding_proj_weight_name)) |proj_meta|
        {
            if (proj_meta.shape.len == 2) stats.embedding_dim = @intCast(proj_meta.shape[0]);
        }
    }

    // Model version: --model-version, or "<checkpoint stem>-<sha256[0..12]>".
    {
        const version = opts.model_version orelse blk: {
            const stem = std.fs.path.stem(std.fs.path.basename(opts.checkpoint));
            break :blk try std.fmt.allocPrint(arena, "{s}-{s}", .{ stem, sha_hex[0..12] });
        };
        if (version.len > stats.model_version_buf.len) return error.ModelVersionTooLong;
        @memcpy(stats.model_version_buf[0..version.len], version);
        stats.model_version_len = version.len;
    }

    // --- Base model -------------------------------------------------------
    const base_st_path = try std.fmt.allocPrint(arena, "{s}/model.safetensors", .{opts.model_dir});
    var base_reader = try safetensors.MMapReader.openFileAbsolute(allocator, base_st_path);
    defer base_reader.deinit();

    // Normalized name → original base tensor name.
    var base_by_norm = std.StringHashMapUnmanaged([]const u8){};
    defer base_by_norm.deinit(allocator);

    const base_names = try base_reader.header.tensorNames(arena);
    for (base_names) |name| {
        const norm = try normalizeModernBertWeightName(arena, name);
        const entry = try base_by_norm.getOrPut(allocator, norm);
        if (entry.found_existing) return error.DuplicateBaseTensorName;
        entry.value_ptr.* = name;
    }

    // LoRA merge ops grouped by normalized base target name.
    var merges = std.StringHashMapUnmanaged(std.ArrayListUnmanaged(MergeOp)){};
    defer {
        var it = merges.valueIterator();
        while (it.next()) |list| list.deinit(allocator);
        merges.deinit(allocator);
    }

    for (contents.pairs.items) |pair| {
        const a_meta = checkpoint_reader.header.tensors.get(pair.a_name.?).?;
        const b_meta = checkpoint_reader.header.tensors.get(pair.b_name.?).?;
        const rank: usize = @intCast(contents.lora_rank);
        if (a_meta.shape[0] != @as(i64, @intCast(rank)) or b_meta.shape[1] != @as(i64, @intCast(rank))) {
            return error.LoRARankMismatch;
        }
        const in_features: usize = @intCast(a_meta.shape[1]);
        const out_features: usize = @intCast(b_meta.shape[0]);

        var target_norm: []const u8 = undefined;
        var part: usize = 0;
        var fused_qkv = false;
        if (pair.target.qkvPart()) |qkv_part| {
            // Prefer split projections when the base stores them; otherwise
            // merge into the corresponding third of the fused Wqkv tensor.
            const proj_name = switch (pair.target) {
                .qkv_query => "query_proj",
                .qkv_key => "key_proj",
                .qkv_value => "value_proj",
                else => unreachable,
            };
            const split_norm = try std.fmt.allocPrint(
                arena,
                "model.layers.{d}.attn.{s}.weight",
                .{ pair.layer, proj_name },
            );
            if (base_by_norm.contains(split_norm)) {
                target_norm = split_norm;
            } else {
                target_norm = try std.fmt.allocPrint(arena, "model.layers.{d}.attn.Wqkv.weight", .{pair.layer});
                part = qkv_part;
                fused_qkv = true;
            }
        } else {
            target_norm = switch (pair.target) {
                .attn_wo => try std.fmt.allocPrint(arena, "model.layers.{d}.attn.Wo.weight", .{pair.layer}),
                .mlp_wo => try std.fmt.allocPrint(arena, "model.layers.{d}.mlp.Wo.weight", .{pair.layer}),
                else => unreachable,
            };
        }

        const base_name = base_by_norm.get(target_norm) orelse {
            std.debug.print("error: LoRA target '{s}' not found in base model\n", .{target_norm});
            return error.MissingLoRATarget;
        };
        const base_meta = base_reader.header.tensors.get(base_name).?;
        if (base_meta.dtype != .f32) return error.InvalidBaseTensorDType;
        if (base_meta.shape.len != 2) return error.InvalidBaseTensorShape;
        const expected_rows: i64 = if (fused_qkv)
            @intCast(3 * out_features)
        else
            @intCast(out_features);
        if (base_meta.shape[0] != expected_rows or base_meta.shape[1] != @as(i64, @intCast(in_features))) {
            std.debug.print(
                "error: LoRA target '{s}' shape mismatch (base [{d},{d}], adapter out={d} in={d})\n",
                .{ target_norm, base_meta.shape[0], base_meta.shape[1], out_features, in_features },
            );
            return error.InvalidLoRAShape;
        }

        const entry = try merges.getOrPut(allocator, target_norm);
        if (!entry.found_existing) entry.value_ptr.* = .empty;
        try entry.value_ptr.append(allocator, .{
            .part = part,
            .out_features = out_features,
            .in_features = in_features,
            .a_name = pair.a_name.?,
            .b_name = pair.b_name.?,
        });
        stats.lora_pairs_merged += 1;
    }

    // --- Plan output tensors ----------------------------------------------
    var plan = std.ArrayListUnmanaged(PlannedTensor).empty;
    defer plan.deinit(allocator);

    var norm_it = base_by_norm.iterator();
    while (norm_it.next()) |entry| {
        const norm = entry.key_ptr.*;
        const base_name = entry.value_ptr.*;
        const meta = base_reader.header.tensors.get(base_name).?;
        const byte_len = meta.data_end - meta.data_start;
        if (merges.getPtr(norm)) |ops| {
            try plan.append(allocator, .{
                .out_name = norm,
                .dtype = meta.dtype,
                .shape = meta.shape,
                .byte_len = byte_len,
                .source = .{ .merged = .{ .base_name = base_name, .ops = ops.items } },
            });
            stats.merged_tensors += 1;
        } else {
            try plan.append(allocator, .{
                .out_name = norm,
                .dtype = meta.dtype,
                .shape = meta.shape,
                .byte_len = byte_len,
                .source = .{ .base_raw = base_name },
            });
            stats.base_tensors += 1;
        }
    }

    const head_mappings = [_]struct { src: []const u8, serving: []const u8 }{
        .{ .src = contents.w1, .serving = boundary_dense1_weight_name },
        .{ .src = contents.b1, .serving = boundary_dense1_bias_name },
        .{ .src = contents.w2, .serving = boundary_dense2_weight_name },
        .{ .src = contents.b2, .serving = boundary_dense2_bias_name },
    };
    for (head_mappings) |mapping| {
        const meta = checkpoint_reader.header.tensors.get(mapping.src).?;
        try plan.append(allocator, .{
            .out_name = mapping.serving,
            .dtype = meta.dtype,
            .shape = meta.shape,
            .byte_len = meta.data_end - meta.data_start,
            .source = .{ .checkpoint_raw = mapping.src },
        });
        stats.head_tensors += 1;
    }
    for (contents.passthrough.items) |mapping| {
        const meta = checkpoint_reader.header.tensors.get(mapping.src).?;
        try plan.append(allocator, .{
            .out_name = mapping.serving,
            .dtype = meta.dtype,
            .shape = meta.shape,
            .byte_len = meta.data_end - meta.data_start,
            .source = .{ .checkpoint_raw = mapping.src },
        });
        stats.head_tensors += 1;
    }

    std.mem.sort(PlannedTensor, plan.items, {}, plannedTensorLessThan);

    // --- Write model.safetensors -------------------------------------------
    try compat.cwd().createDirPath(compat.io(), opts.out);
    const out_st_path = try std.fmt.allocPrint(arena, "{s}/model.safetensors", .{opts.out});
    try writeSafetensors(allocator, out_st_path, plan.items, &base_reader, &checkpoint_reader, contents.lora_alpha);

    // --- Copy tokenizer/config files ---------------------------------------
    try copyModelFile(allocator, arena, opts.model_dir, opts.out, "tokenizer.json", true);
    try copyModelFile(allocator, arena, opts.model_dir, opts.out, "config.json", true);
    try copyModelFile(allocator, arena, opts.model_dir, opts.out, "tokenizer_config.json", false);

    // --- model_manifest.json ------------------------------------------------
    try writeManifest(allocator, arena, opts, &stats);

    return stats;
}

fn writeSafetensors(
    allocator: std.mem.Allocator,
    path: []const u8,
    plan: []const PlannedTensor,
    base_reader: *const safetensors.MMapReader,
    checkpoint_reader: *const safetensors.MMapReader,
    lora_alpha: f32,
) !void {
    // Header JSON with data offsets.
    var header_buf: std.Io.Writer.Allocating = .init(allocator);
    defer header_buf.deinit();
    const hw = &header_buf.writer;

    try hw.writeAll("{\"__metadata__\":{\"format\":\"pt\"}");
    var offset: u64 = 0;
    for (plan) |t| {
        try hw.writeAll(",\"");
        for (t.out_name) |c| {
            switch (c) {
                '"' => try hw.writeAll("\\\""),
                '\\' => try hw.writeAll("\\\\"),
                else => try hw.writeByte(c),
            }
        }
        try hw.print("\":{{\"dtype\":\"{s}\",\"shape\":[", .{dtypeName(t.dtype)});
        for (t.shape, 0..) |dim, i| {
            if (i > 0) try hw.writeByte(',');
            try hw.print("{d}", .{dim});
        }
        try hw.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + t.byte_len });
        offset += t.byte_len;
    }
    try hw.writeByte('}');
    const header_json = header_buf.written();

    var file = try compat.cwd().createFile(compat.io(), path, .{ .truncate = true });
    defer file.close(compat.io());
    var write_buf: [64 * 1024]u8 = undefined;
    var writer = file.writerStreaming(compat.io(), &write_buf);
    const w = &writer.interface;

    var size_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &size_buf, header_json.len, .little);
    try w.writeAll(&size_buf);
    try w.writeAll(header_json);

    // Stream tensor data one tensor at a time.
    for (plan) |t| {
        switch (t.source) {
            .base_raw => |base_name| {
                var tensor = try base_reader.readTensor(base_name);
                defer tensor.deinit();
                try w.writeAll(tensor.data);
            },
            .checkpoint_raw => |src_name| {
                var tensor = try checkpoint_reader.readTensor(src_name);
                defer tensor.deinit();
                try w.writeAll(tensor.data);
            },
            .merged => |spec| {
                const buf = try readTensorF32Alloc(allocator, base_reader, spec.base_name);
                defer allocator.free(buf);

                for (spec.ops) |op| {
                    const a_meta = checkpoint_reader.header.tensors.get(op.a_name) orelse return error.InvalidCheckpointTensor;
                    const rank: usize = @intCast(a_meta.shape[0]);
                    const a_values = try readTensorF32Alloc(allocator, checkpoint_reader, op.a_name);
                    defer allocator.free(a_values);
                    const b_values = try readTensorF32Alloc(allocator, checkpoint_reader, op.b_name);
                    defer allocator.free(b_values);
                    try mergeLoRAPart(
                        allocator,
                        buf,
                        op.out_features,
                        op.in_features,
                        op.part,
                        a_values,
                        b_values,
                        rank,
                        lora_alpha,
                    );
                }
                try w.writeAll(std.mem.sliceAsBytes(buf));
            },
        }
    }
    try w.flush();
}

fn copyModelFile(
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    model_dir: []const u8,
    out_dir: []const u8,
    leaf: []const u8,
    required: bool,
) !void {
    const src_path = try std.fmt.allocPrint(arena, "{s}/{s}", .{ model_dir, leaf });
    const data = compat.cwd().readFileAlloc(compat.io(), src_path, allocator, .unlimited) catch |err| {
        if (required) {
            std.debug.print("error: could not read required file {s}: {}\n", .{ src_path, err });
            return err;
        }
        std.debug.print("warning: skipping optional file {s}: {}\n", .{ src_path, err });
        return;
    };
    defer allocator.free(data);
    const dst_path = try std.fmt.allocPrint(arena, "{s}/{s}", .{ out_dir, leaf });
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = dst_path, .data = data });
}

fn writeJsonString(w: *std.Io.Writer, s: []const u8) !void {
    try w.writeByte('"');
    for (s) |c| {
        switch (c) {
            '"' => try w.writeAll("\\\""),
            '\\' => try w.writeAll("\\\\"),
            '\n' => try w.writeAll("\\n"),
            '\r' => try w.writeAll("\\r"),
            '\t' => try w.writeAll("\\t"),
            else => {
                if (c < 0x20) {
                    try w.print("\\u{x:0>4}", .{c});
                } else {
                    try w.writeByte(c);
                }
            },
        }
    }
    try w.writeByte('"');
}

fn writeManifest(
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    opts: Options,
    stats: *const ExportStats,
) !void {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    const w = &buf.writer;

    // Fields parsed by src/models/manifest.zig (parseModelManifestJson):
    // "type", "tasks", "inputs". Everything fused-chunker-specific lives under
    // the namespaced "fused_chunker" key so schema evolution stays additive.
    try w.writeAll("{\n");
    try w.writeAll("  \"schema_version\": 1,\n");
    try w.writeAll("  \"type\": \"chunker\",\n");
    try w.writeAll("  \"tasks\": [\"chunk\"],\n");
    try w.writeAll("  \"inputs\": [\"text\"],\n");
    try w.writeAll("  \"artifact_family\": ");
    try writeJsonString(w, fused_chunker.artifact_family_version);
    try w.writeAll(",\n  \"model_version\": ");
    try writeJsonString(w, stats.modelVersion());
    try w.writeAll(",\n  \"fused_chunker\": {\n");
    try w.writeAll("    \"artifact_family\": ");
    try writeJsonString(w, fused_chunker.artifact_family_version);
    try w.print(",\n    \"boundary_threshold\": {d},\n", .{opts.boundary_threshold});
    try w.print("    \"max_seq_len\": {d},\n", .{default_max_seq_len});
    try w.print("    \"hidden_size\": {d},\n", .{stats.hidden_size});
    try w.print("    \"embedding_dim\": {d},\n", .{stats.embedding_dim});
    try w.print("    \"splade\": {},\n", .{stats.splade});
    try w.print("    \"lora_merged\": {},\n", .{stats.lora_pairs_merged > 0});
    try w.print("    \"lora_rank\": {d},\n", .{stats.lora_rank});
    try w.print("    \"lora_alpha\": {d},\n", .{stats.lora_alpha});
    try w.writeAll("    \"source_checkpoint\": ");
    try writeJsonString(w, std.fs.path.basename(opts.checkpoint));
    try w.writeAll(",\n    \"checkpoint_sha256\": ");
    try writeJsonString(w, stats.checkpoint_sha256[0..]);
    try w.writeAll("\n  }\n}\n");

    const manifest_path = try std.fmt.allocPrint(arena, "{s}/model_manifest.json", .{opts.out});
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = manifest_path, .data = buf.written() });
}

fn writeSummary(allocator: std.mem.Allocator, opts: Options, stats: ExportStats) !void {
    var buf: std.Io.Writer.Allocating = .init(allocator);
    defer buf.deinit();
    const w = &buf.writer;

    try w.print(
        "{{\"tool\":\"export_fused_chunker_model\",\"schema_version\":1,\"status\":\"passed\",\"checkpoint\":\"{s}\",\"model_dir\":\"{s}\",\"out\":\"{s}\",\"model_version\":\"{s}\",\"artifact_family\":\"{s}\",\"base_tensors\":{d},\"merged_tensors\":{d},\"lora_pairs_merged\":{d},\"head_tensors\":{d},\"lora_rank\":{d},\"lora_alpha\":{d},\"splade\":{},\"checkpoint_sha256\":\"{s}\"}}\n",
        .{
            opts.checkpoint,
            opts.model_dir,
            opts.out,
            stats.modelVersion(),
            fused_chunker.artifact_family_version,
            stats.base_tensors,
            stats.merged_tensors,
            stats.lora_pairs_merged,
            stats.head_tensors,
            stats.lora_rank,
            stats.lora_alpha,
            stats.splade,
            stats.checkpoint_sha256[0..],
        },
    );

    var stdout_buf: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(compat.io(), &stdout_buf);
    try stdout.interface.writeAll(buf.written());
    try stdout.interface.flush();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "mergeLoRAPart produces W + B·A·(alpha/rank)" {
    const allocator = std.testing.allocator;

    // out_features=3, in_features=2, rank=2, alpha=4 → scale = 2.
    const out_features: usize = 3;
    const in_features: usize = 2;
    const rank: usize = 2;
    const alpha: f32 = 4.0;
    const scale: f32 = alpha / @as(f32, @floatFromInt(rank));

    const base = [_]f32{
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
    };
    // A [rank, in]
    const a = [_]f32{
        0.5,  -1.0,
        2.0,  0.25,
    };
    // B [out, rank]
    const b = [_]f32{
        1.0,  2.0,
        -0.5, 1.5,
        3.0,  -2.0,
    };

    var buf: [out_features * in_features]f32 = undefined;
    @memcpy(&buf, &base);
    try mergeLoRAPart(allocator, &buf, out_features, in_features, 0, &a, &b, rank, alpha);

    // Reference: W' = W + scale * (B @ A).
    for (0..out_features) |o| {
        for (0..in_features) |i| {
            var delta: f32 = 0.0;
            for (0..rank) |r| {
                delta += b[o * rank + r] * a[r * in_features + i];
            }
            const expected = base[o * in_features + i] + scale * delta;
            try std.testing.expectApproxEqAbs(expected, buf[o * in_features + i], 1e-6);
        }
    }
}

test "mergeLoRAPart only touches the selected fused Wqkv part" {
    const allocator = std.testing.allocator;

    // Fused tensor with 3 parts of out=2, in=2 each.
    const out_features: usize = 2;
    const in_features: usize = 2;
    const rank: usize = 1;
    const alpha: f32 = 1.0; // scale = 1

    var buf: [3 * out_features * in_features]f32 = undefined;
    for (&buf, 0..) |*v, i| v.* = @floatFromInt(i);
    const original = buf;

    const a = [_]f32{ 1.0, 2.0 }; // [1, 2]
    const b = [_]f32{ 10.0, 100.0 }; // [2, 1]

    try mergeLoRAPart(allocator, &buf, out_features, in_features, 1, &a, &b, rank, alpha);

    // Part 0 and part 2 untouched.
    for (0..4) |i| try std.testing.expectEqual(original[i], buf[i]);
    for (8..12) |i| try std.testing.expectEqual(original[i], buf[i]);
    // Part 1: delta = B@A = [[10,20],[100,200]].
    try std.testing.expectApproxEqAbs(original[4] + 10.0, buf[4], 1e-6);
    try std.testing.expectApproxEqAbs(original[5] + 20.0, buf[5], 1e-6);
    try std.testing.expectApproxEqAbs(original[6] + 100.0, buf[6], 1e-6);
    try std.testing.expectApproxEqAbs(original[7] + 200.0, buf[7], 1e-6);
}

test "parseCheckpointLoRAKey maps checkpoint tensor names" {
    const q = parseCheckpointLoRAKey("model.layers.3.attn.query_proj.lora_a").?;
    try std.testing.expectEqual(@as(u32, 3), q.layer);
    try std.testing.expectEqual(LoRATarget.qkv_query, q.target);
    try std.testing.expect(q.is_a);
    try std.testing.expectEqual(@as(?usize, 0), q.target.qkvPart());

    const k = parseCheckpointLoRAKey("model.layers.0.attn.key_proj.lora_b").?;
    try std.testing.expectEqual(LoRATarget.qkv_key, k.target);
    try std.testing.expect(!k.is_a);
    try std.testing.expectEqual(@as(?usize, 1), k.target.qkvPart());

    const v = parseCheckpointLoRAKey("model.layers.21.attn.value_proj.lora_a").?;
    try std.testing.expectEqual(@as(?usize, 2), v.target.qkvPart());

    const attn_wo = parseCheckpointLoRAKey("model.layers.7.attn.Wo.lora_b").?;
    try std.testing.expectEqual(LoRATarget.attn_wo, attn_wo.target);
    try std.testing.expectEqual(@as(?usize, null), attn_wo.target.qkvPart());

    const mlp_wo = parseCheckpointLoRAKey("model.layers.21.mlp.Wo.lora_a").?;
    try std.testing.expectEqual(LoRATarget.mlp_wo, mlp_wo.target);

    try std.testing.expectEqual(@as(?ParsedLoRAKey, null), parseCheckpointLoRAKey("model.layers.1.attn.Wqkv.weight"));
    try std.testing.expectEqual(@as(?ParsedLoRAKey, null), parseCheckpointLoRAKey("model.layers.x.attn.Wo.lora_a"));
    try std.testing.expectEqual(@as(?ParsedLoRAKey, null), parseCheckpointLoRAKey("w1"));
}

test "passthroughServingName maps optional head tensors" {
    try std.testing.expectEqualStrings(
        embedding_proj_weight_name,
        passthroughServingName("embedding_head/proj/weight").?,
    );
    try std.testing.expectEqualStrings(
        embedding_proj_weight_name,
        passthroughServingName("fused_chunker_embedder/embedding_head/proj/weight").?,
    );
    try std.testing.expectEqualStrings(
        fused_chunker_splade.SPLADE_WEIGHT_KEY,
        passthroughServingName("var:/fused_chunker_embedder/splade_head/proj/weight").?,
    );
    try std.testing.expectEqual(@as(?[]const u8, null), passthroughServingName("w1"));
    try std.testing.expectEqual(@as(?[]const u8, null), passthroughServingName("model.layers.0.attn.Wo.weight"));
}

test "normalizeModernBertWeightName mirrors eval loader" {
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
}
