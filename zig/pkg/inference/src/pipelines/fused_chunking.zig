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

// Fused chunker-embedder serving pipeline.
//
// Serves POST /chunk requests from a packaged fused-chunker model directory
// produced by src/finetune/tools/export_fused_chunker_model.zig:
//   model.safetensors      — ModernBERT base with LoRA merged + boundary head
//                            (+ optional embedding projection / SPLADE heads)
//   tokenizer.json         — HuggingFace tokenizer
//   config.json            — ModernBERT config
//   model_manifest.json    — chunker manifest with a "fused_chunker" block
//
// Per request: tokenize with char offsets → window long documents into
// max_seq_len token windows (each forward pass sees at most max_seq_len
// tokens, matching eval_fused_chunker.zig's max_seq_len semantics) → run
// fused_chunker.forward (or forwardBoundaryOnly when no embeddings are
// requested) per window → decode boundaries over the concatenated logits →
// late-chunking mean-pool for dense chunk embeddings (honoring
// output_dimension truncate+renormalize) → optional SPLADE sparse vectors.
//
// LoRA adapters are already merged into the exported base weights, so this
// loader only performs plain base-weight loading (no adapter application).
//
// Boundary feature mode: only the plain per-token feature view is
// implemented — this matches current training runs (BoundaryFeatureMode
// .token); the diff/concat feature views are eval/training-only experiments.
//
// Backend selection is automatic: Metal when the binary is built with
// -Dmetal=true and a Metal device is available, native CPU otherwise. All
// forward passes are serialized through an inference lock (the same pattern
// as model_manager.LoadedModel.lockRerankingSession) so concurrent requests
// never interleave GPU command submission or weight-store access.

const std = @import("std");
const build_options = @import("build_options");
const lib_chunker = @import("inference_chunker");
const hf_tokenizer = @import("inference_hf_tokenizer");
const fused_chunker = @import("../finetune/fused_chunker.zig");
const fused_chunker_splade = @import("../finetune/fused_chunker_splade.zig");
const fused_chunker_weights = @import("../finetune/fused_chunker_weights.zig");
const native_compute = @import("../ops/native_compute.zig");
const metal_compute = if (build_options.enable_metal) @import("../ops/metal_compute.zig") else struct {};
const metal_runtime = if (build_options.enable_metal) @import("../backends/metal_runtime.zig") else struct {
    pub fn metalDeviceAvailable() bool {
        return false;
    }
};
const ops_mod = @import("../ops/ops.zig");
const safetensors = @import("../models/safetensors.zig");
const c_file = @import("../util/c_file.zig");

const ComputeBackend = ops_mod.ComputeBackend;
const MetalComputeT = if (build_options.enable_metal) metal_compute.MetalCompute else void;

/// Upper bound on tokens processed per request. Text beyond this many tokens
/// is not chunked (the tokenizer truncates); at max_seq_len=384 this allows
/// ~85 forward windows per request.
pub const max_serving_tokens: usize = 32768;

const boundary_dense1_weight_name = "fused_chunker_embedder/boundary_head/mlp_dense1/weight";
const embedding_proj_weight_name = "fused_chunker_embedder/embedding_head/proj/weight";

pub const Backend = enum { auto, native, metal };

pub const LoadOptions = struct {
    backend: Backend = .auto,
};

/// Per-request options, mapped from lib_chunker.FixedChunkConfig by the server.
pub const ChunkRequestOptions = struct {
    /// Boundary probability threshold; overrides the exported default.
    threshold: ?f32 = null,
    /// Maximum number of chunks returned (0 = unlimited).
    max_chunks: usize = 50,
    include_embeddings: bool = false,
    include_sparse: bool = false,
    /// Truncate dense embeddings to this dimension and L2-renormalize.
    output_dimension: ?u32 = null,
    /// Keep only the top-k SPLADE activations per chunk (by weight).
    sparse_top_k: ?usize = null,
};

fn spinLock(m: *std.atomic.Mutex) void {
    while (!m.tryLock()) {
        std.atomic.spinLoopHint();
    }
}

pub const FusedChunkerPipeline = struct {
    allocator: std.mem.Allocator,
    tok: *hf_tokenizer.HfTokenizer,
    config: fused_chunker.Config,
    max_seq_len: usize,
    /// Owned; from model_manifest.json "model_version".
    model_version: []const u8,
    has_splade: bool,
    use_metal: bool,
    native_store: native_compute.WeightStore,
    native_backend: native_compute.NativeCompute,
    metal_store: fused_chunker_weights.MetalWeightStore = undefined,
    metal_backend: MetalComputeT = undefined,
    /// Lazily materialized f32 SPLADE projection [vocab_size * hidden_size].
    splade_weight: ?[]f32 = null,
    /// Serializes forward passes (GPU command submission / weight residency).
    inference_lock: std.atomic.Mutex = .unlocked,

    /// Load a packaged fused-chunker model directory. The returned pipeline is
    /// heap-allocated; release it with deinit().
    pub fn loadFromDir(
        allocator: std.mem.Allocator,
        model_dir: []const u8,
        load_opts: LoadOptions,
    ) !*FusedChunkerPipeline {
        const metal_available = if (comptime build_options.enable_metal)
            metal_runtime.metalDeviceAvailable()
        else
            false;
        const use_metal = switch (load_opts.backend) {
            .auto => metal_available,
            .metal => if (metal_available) true else return error.MetalBackendUnavailable,
            .native => false,
        };

        const manifest = try parseServingManifest(allocator, model_dir);
        errdefer allocator.free(manifest.model_version);

        var config = fused_chunker.Config{};
        try applyModelConfigJson(allocator, model_dir, &config);
        config.boundary_threshold = manifest.boundary_threshold;

        const st_path = try std.fmt.allocPrint(allocator, "{s}/model.safetensors", .{model_dir});
        defer allocator.free(st_path);
        const has_splade = try inspectExportedHeads(allocator, st_path, &config);

        const tok_path = try std.fmt.allocPrint(allocator, "{s}/tokenizer.json", .{model_dir});
        defer allocator.free(tok_path);
        const tok_bytes = try c_file.readFile(allocator, tok_path);
        defer allocator.free(tok_bytes);
        const tok = try hf_tokenizer.HfTokenizer.loadFromBytes(allocator, tok_bytes);
        errdefer tok.deinitSelf();

        const self = try allocator.create(FusedChunkerPipeline);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .tok = tok,
            .config = config,
            .max_seq_len = @intCast(@max(manifest.max_seq_len, 16)),
            .model_version = manifest.model_version,
            .has_splade = has_splade,
            .use_metal = use_metal,
            .native_store = .{
                .allocator = allocator,
                .resident_weights = .{},
                .lazy_weights = .{},
            },
            .native_backend = undefined,
        };
        self.native_backend = native_compute.NativeCompute.init(allocator, &self.native_store, null);
        errdefer fused_chunker_weights.deinitNativeWeightStore(allocator, &self.native_store);

        if (use_metal) {
            if (comptime build_options.enable_metal) {
                self.metal_store = .{
                    .allocator = allocator,
                    .prefix = "",
                    .lazy_weights = .{},
                };
                metal_compute.initPrefetchQueue(&self.metal_store, allocator);
                errdefer fused_chunker_weights.deinitMetalWeightStore(allocator, &self.metal_store);
                self.metal_backend = try metal_compute.MetalCompute.init(allocator, &self.metal_store, null);
                errdefer self.metal_backend.deinit();
                _ = try fused_chunker_weights.loadSafetensorsIntoMetalStore(allocator, &self.metal_store, st_path);
            } else {
                return error.MetalBackendUnavailable;
            }
        } else {
            _ = try fused_chunker_weights.loadSafetensorsIntoNativeStore(allocator, &self.native_store, st_path);
        }

        return self;
    }

    pub fn deinit(self: *FusedChunkerPipeline) void {
        const allocator = self.allocator;
        if (self.splade_weight) |w| allocator.free(w);
        if (self.use_metal) {
            if (comptime build_options.enable_metal) {
                fused_chunker_weights.deinitMetalWeightStore(allocator, &self.metal_store);
                self.metal_backend.deinit();
            }
        }
        fused_chunker_weights.deinitNativeWeightStore(allocator, &self.native_store);
        self.tok.deinitSelf();
        allocator.free(self.model_version);
        allocator.destroy(self);
    }

    fn computeBackend(self: *FusedChunkerPipeline) ComputeBackend {
        if (self.use_metal) {
            if (comptime build_options.enable_metal) return self.metal_backend.computeBackend();
            unreachable;
        }
        return self.native_backend.computeBackend();
    }

    /// Materialize the SPLADE projection weight as f32 (cached for the
    /// pipeline lifetime). Must be called with inference_lock held.
    fn ensureSpladeWeight(self: *FusedChunkerPipeline, cb: *const ComputeBackend) ![]const f32 {
        if (self.splade_weight) |w| return w;
        const ct = try cb.getWeight(fused_chunker_splade.SPLADE_WEIGHT_KEY);
        defer cb.free(ct);
        const w = try cb.toFloat32(ct, self.allocator);
        self.splade_weight = w;
        return w;
    }

    /// Chunk a text document. Returned chunks (and their owned embeddings)
    /// are allocated from `allocator`; free with lib_chunker.types.freeChunks.
    pub fn chunkText(
        self: *FusedChunkerPipeline,
        allocator: std.mem.Allocator,
        text: []const u8,
        opts: ChunkRequestOptions,
    ) ![]lib_chunker.Chunk {
        if (opts.include_sparse and !self.has_splade) return error.SparseEmbeddingsUnsupported;
        if (text.len == 0) return allocator.alloc(lib_chunker.Chunk, 0);

        var cfg = self.config;
        if (opts.threshold) |t| cfg.boundary_threshold = t;

        const emb_dim: usize = @intCast(cfg.embedding_dim);
        const hidden: usize = @intCast(cfg.hidden_size);
        // SPLADE pools raw encoder hidden states; the identity embedding head
        // (embedding_dim == hidden_size) exposes them as token embeddings.
        if (opts.include_sparse and emb_dim != hidden) return error.SparseEmbeddingsUnsupported;

        // 1. Tokenize the whole document once, with char offsets.
        const tok = self.tok.tokenizer();
        const token_cap = @max(@min(text.len + 2, max_serving_tokens), 16);
        var enc = try tok.encodeForModel(allocator, text, token_cap);
        defer enc.deinit();
        const offsets = enc.offsets orelse return error.TokenizerOffsetsUnavailable;

        var valid_len: usize = 0;
        for (enc.attention_mask) |m| {
            if (m == 0) break;
            valid_len += 1;
        }
        if (valid_len == 0) return allocator.alloc(lib_chunker.Chunk, 0);

        const need_token_outputs = opts.include_embeddings or opts.include_sparse;

        // 2. Forward each max_seq_len window, concatenating per-token outputs.
        const boundary_logits = try allocator.alloc(f32, valid_len * 2);
        defer allocator.free(boundary_logits);
        const token_embeddings: []f32 = if (need_token_outputs)
            try allocator.alloc(f32, valid_len * emb_dim)
        else
            &.{};
        defer if (need_token_outputs) allocator.free(token_embeddings);

        var splade_weight: []const f32 = &.{};
        {
            spinLock(&self.inference_lock);
            defer self.inference_lock.unlock();
            const cb = self.computeBackend();

            if (opts.include_sparse) splade_weight = try self.ensureSpladeWeight(&cb);

            const msl = self.max_seq_len;
            const windows = numWindows(valid_len, msl);
            var w: usize = 0;
            while (w < windows) : (w += 1) {
                const start = w * msl;
                const end = @min(start + msl, valid_len);
                const n = end - start;

                const ids_i64 = try allocator.alloc(i64, n);
                defer allocator.free(ids_i64);
                for (enc.ids[start..end], ids_i64) |id, *out| out.* = id;
                const mask_i64 = try allocator.alloc(i64, n);
                defer allocator.free(mask_i64);
                @memset(mask_i64, 1);

                if (need_token_outputs) {
                    var result = try fused_chunker.forward(&cb, allocator, cfg, ids_i64, mask_i64, 1, n);
                    defer result.deinit(allocator);
                    @memcpy(boundary_logits[start * 2 .. end * 2], result.boundary_logits[0 .. n * 2]);
                    @memcpy(
                        token_embeddings[start * emb_dim .. end * emb_dim],
                        result.token_embeddings[0 .. n * emb_dim],
                    );
                } else {
                    const logits = try fused_chunker.forwardBoundaryOnly(&cb, allocator, cfg, ids_i64, mask_i64, 1, n);
                    defer allocator.free(logits);
                    @memcpy(boundary_logits[start * 2 .. end * 2], logits[0 .. n * 2]);
                }
            }
        }

        // 3. Decode boundary logits into token spans and map to char ranges.
        const all_spans = try decodeSpansWithFallback(allocator, cfg, boundary_logits, valid_len);
        defer allocator.free(all_spans);

        const KeptSpan = struct {
            span: fused_chunker.ChunkSpan,
            chars: CharSpan,
        };
        var kept = std.ArrayListUnmanaged(KeptSpan).empty;
        defer kept.deinit(allocator);
        for (all_spans) |span| {
            if (opts.max_chunks > 0 and kept.items.len >= opts.max_chunks) break;
            const chars = charSpanFromTokens(offsets[0..valid_len], span.start_token, span.end_token) orelse continue;
            try kept.append(allocator, .{ .span = span, .chars = chars });
        }

        // 4. Dense embeddings via late-chunking mean pool over real tokens.
        var chunk_embeddings: []f32 = &.{};
        defer if (chunk_embeddings.len > 0) allocator.free(chunk_embeddings);
        var splade_vectors: []f32 = &.{};
        defer if (splade_vectors.len > 0) allocator.free(splade_vectors);

        if (kept.items.len > 0 and need_token_outputs) {
            const n_chunks = kept.items.len;
            const chunk_mask = try allocator.alloc(f32, n_chunks);
            defer allocator.free(chunk_mask);
            @memset(chunk_mask, 1.0);

            if (opts.include_embeddings) {
                const starts = try allocator.alloc(u32, n_chunks);
                defer allocator.free(starts);
                const ends = try allocator.alloc(u32, n_chunks);
                defer allocator.free(ends);
                for (kept.items, 0..) |k, i| {
                    starts[i] = k.chars.first_real;
                    ends[i] = k.chars.last_real_excl;
                }
                chunk_embeddings = try fused_chunker.lateChunkingPool(
                    allocator,
                    cfg,
                    token_embeddings,
                    starts,
                    ends,
                    chunk_mask,
                    1,
                    valid_len,
                    n_chunks,
                );
            }

            if (opts.include_sparse) {
                const starts_i32 = try allocator.alloc(i32, n_chunks);
                defer allocator.free(starts_i32);
                const ends_i32 = try allocator.alloc(i32, n_chunks);
                defer allocator.free(ends_i32);
                for (kept.items, 0..) |k, i| {
                    starts_i32[i] = @intCast(k.chars.first_real);
                    ends_i32[i] = @intCast(k.chars.last_real_excl);
                }
                splade_vectors = try fused_chunker.computeChunkSpladeVectors(
                    allocator,
                    cfg,
                    token_embeddings,
                    splade_weight,
                    starts_i32,
                    ends_i32,
                    chunk_mask,
                    1,
                    valid_len,
                    n_chunks,
                );
            }
        }

        const out_dim: usize = blk: {
            if (opts.output_dimension) |d| {
                const dd: usize = @intCast(d);
                if (dd > 0 and dd < emb_dim) break :blk dd;
            }
            break :blk emb_dim;
        };

        // 5. Build lib_chunker chunks.
        const chunks = try allocator.alloc(lib_chunker.Chunk, kept.items.len);
        var built: usize = 0;
        errdefer {
            for (chunks[0..built]) |*c| c.deinit(allocator);
            allocator.free(chunks);
        }
        for (kept.items, 0..) |k, i| {
            var chunk = lib_chunker.Chunk{
                .id = @intCast(i),
                .mime_type = "text/plain",
                .text = text[k.chars.start_char..k.chars.end_char],
                .start_char = k.chars.start_char,
                .end_char = k.chars.end_char,
                .start_token = k.span.start_token,
                .end_token = k.span.end_token,
                .boundary_score = boundaryProbability(boundary_logits, k.span.start_token),
                .model_version = self.model_version,
                .artifact_family = fused_chunker.artifact_family_version,
            };
            if (opts.include_embeddings) {
                const src = chunk_embeddings[i * emb_dim .. (i + 1) * emb_dim];
                chunk.embedding = try truncateAndRenormalize(allocator, src, out_dim);
                chunk.owns_embedding = true;
                chunk.embedding_dimension = @intCast(out_dim);
            }
            if (opts.include_sparse) {
                const vocab: usize = @intCast(cfg.splade_config.vocab_size);
                const dense = splade_vectors[i * vocab .. (i + 1) * vocab];
                chunk.sparse_embedding = try sparseFromDense(allocator, dense, opts.sparse_top_k);
            }
            chunks[i] = chunk;
            built += 1;
        }
        return chunks;
    }
};

// ---------------------------------------------------------------------------
// Model directory metadata
// ---------------------------------------------------------------------------

const ServingManifest = struct {
    model_version: []const u8,
    boundary_threshold: f32 = 0.5,
    max_seq_len: u32 = 384,
};

fn jsonF32(value: std.json.Value) ?f32 {
    return switch (value) {
        .float => |f| @floatCast(f),
        .integer => |i| @floatFromInt(i),
        else => null,
    };
}

fn jsonU32(value: std.json.Value) ?u32 {
    return switch (value) {
        .integer => |i| if (i > 0) std.math.cast(u32, i) else null,
        else => null,
    };
}

fn parseServingManifest(allocator: std.mem.Allocator, model_dir: []const u8) !ServingManifest {
    const path = try std.fmt.allocPrint(allocator, "{s}/model_manifest.json", .{model_dir});
    defer allocator.free(path);
    const bytes = try c_file.readFile(allocator, path);
    defer allocator.free(bytes);

    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidModelManifest;
    const obj = parsed.value.object;

    var out = ServingManifest{ .model_version = "" };
    var model_version: []const u8 = "";
    if (obj.get("model_version")) |v| {
        if (v == .string) model_version = v.string;
    }
    if (obj.get("fused_chunker")) |fc| {
        if (fc == .object) {
            if (fc.object.get("boundary_threshold")) |v| {
                if (jsonF32(v)) |t| {
                    if (t > 0.0 and t < 1.0) out.boundary_threshold = t;
                }
            }
            if (fc.object.get("max_seq_len")) |v| {
                if (jsonU32(v)) |msl| out.max_seq_len = msl;
            }
        }
    }
    out.model_version = try allocator.dupe(u8, model_version);
    return out;
}

/// Apply ModernBERT hyperparameters from config.json onto the fused config.
fn applyModelConfigJson(
    allocator: std.mem.Allocator,
    model_dir: []const u8,
    config: *fused_chunker.Config,
) !void {
    const path = try std.fmt.allocPrint(allocator, "{s}/config.json", .{model_dir});
    defer allocator.free(path);
    const bytes = c_file.readFile(allocator, path) catch return; // optional: defaults apply
    defer allocator.free(bytes);

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, bytes, .{}) catch return;
    defer parsed.deinit();
    if (parsed.value != .object) return;
    const obj = parsed.value.object;

    if (obj.get("vocab_size")) |v| config.vocab_size = jsonU32(v) orelse config.vocab_size;
    if (obj.get("hidden_size")) |v| config.hidden_size = jsonU32(v) orelse config.hidden_size;
    if (obj.get("num_hidden_layers")) |v| config.num_hidden_layers = jsonU32(v) orelse config.num_hidden_layers;
    if (obj.get("num_attention_heads")) |v| config.num_attention_heads = jsonU32(v) orelse config.num_attention_heads;
    if (obj.get("intermediate_size")) |v| config.intermediate_size = jsonU32(v) orelse config.intermediate_size;
    if (obj.get("max_position_embeddings")) |v| config.max_position_embeddings = jsonU32(v) orelse config.max_position_embeddings;
    if (obj.get("global_attn_every_n_layers")) |v| config.global_attn_every_n_layers = jsonU32(v) orelse config.global_attn_every_n_layers;
    if (obj.get("local_attention")) |v| config.local_attention_window = jsonU32(v) orelse config.local_attention_window;
    if (obj.get("global_rope_theta")) |v| config.global_rope_theta = jsonF32(v) orelse config.global_rope_theta;
    if (obj.get("local_rope_theta")) |v| config.local_rope_theta = jsonF32(v) orelse config.local_rope_theta;
    if (obj.get("norm_eps")) |v| config.layer_norm_eps = jsonF32(v) orelse config.layer_norm_eps;
}

/// Derive head geometry from the exported safetensors header: boundary MLP
/// width from the dense1 weight shape, embedding_dim from the optional
/// projection head, and SPLADE availability from the projection weight key.
fn inspectExportedHeads(
    allocator: std.mem.Allocator,
    st_path: []const u8,
    config: *fused_chunker.Config,
) !bool {
    var reader = try safetensors.MMapReader.openFileAbsolute(allocator, st_path);
    defer reader.deinit();

    const w1 = reader.header.tensors.get(boundary_dense1_weight_name) orelse
        return error.MissingBoundaryHead;
    if (w1.shape.len != 2 or w1.shape[0] <= 0) return error.InvalidBoundaryHead;
    config.boundary_mlp_dim = @intCast(w1.shape[0]);

    if (reader.header.tensors.get(embedding_proj_weight_name)) |proj| {
        if (proj.shape.len == 2 and proj.shape[0] > 0) {
            config.embedding_dim = @intCast(proj.shape[0]);
        }
    } else {
        config.embedding_dim = config.hidden_size;
    }

    const has_splade = reader.header.tensors.contains(fused_chunker_splade.SPLADE_WEIGHT_KEY);
    config.enable_splade = has_splade;
    config.splade_config.vocab_size = config.vocab_size;
    return has_splade;
}

// ---------------------------------------------------------------------------
// Pure helpers (unit-tested without a model)
// ---------------------------------------------------------------------------

pub fn numWindows(valid_len: usize, max_seq_len: usize) usize {
    if (valid_len == 0) return 0;
    return (valid_len + max_seq_len - 1) / max_seq_len;
}

/// Decode boundary logits over the full (unpadded) sequence. Short documents
/// that never cross min_chunk_tokens would otherwise produce zero chunks, so
/// fall back to a single span covering the whole sequence.
pub fn decodeSpansWithFallback(
    allocator: std.mem.Allocator,
    cfg: fused_chunker.Config,
    boundary_logits: []const f32,
    valid_len: usize,
) ![]fused_chunker.ChunkSpan {
    const mask = try allocator.alloc(i32, valid_len);
    defer allocator.free(mask);
    @memset(mask, 1);

    const spans = try fused_chunker.decodeBoundaries(allocator, cfg, boundary_logits, mask, valid_len);
    if (spans.len > 0) return spans;
    allocator.free(spans);

    const single = try allocator.alloc(fused_chunker.ChunkSpan, 1);
    single[0] = .{ .start_token = 0, .end_token = @intCast(valid_len) };
    return single;
}

/// Boundary probability (2-class softmax) at one token position.
pub fn boundaryProbability(boundary_logits: []const f32, token: usize) f32 {
    const l0 = boundary_logits[token * 2];
    const l1 = boundary_logits[token * 2 + 1];
    const m = @max(l0, l1);
    const e0 = @exp(l0 - m);
    const e1 = @exp(l1 - m);
    return e1 / (e0 + e1);
}

pub const CharSpan = struct {
    start_char: u32,
    end_char: u32,
    /// First token in the span with a real (non-degenerate) char offset.
    first_real: u32,
    /// One past the last real token (exclusive).
    last_real_excl: u32,
};

/// Map a token span to its character range, skipping special tokens whose
/// offsets are degenerate ([CLS]/[SEP]/padding report start == end). Returns
/// null when the span covers no real text.
pub fn charSpanFromTokens(offsets: []const [2]u32, start_token: u32, end_token: u32) ?CharSpan {
    var first: ?usize = null;
    var last: ?usize = null;
    var t: usize = start_token;
    const end = @min(@as(usize, end_token), offsets.len);
    while (t < end) : (t += 1) {
        const off = offsets[t];
        if (off[1] <= off[0]) continue;
        if (first == null) first = t;
        last = t;
    }
    const f = first orelse return null;
    const l = last.?;
    if (offsets[l][1] <= offsets[f][0]) return null;
    return .{
        .start_char = offsets[f][0],
        .end_char = offsets[l][1],
        .first_real = @intCast(f),
        .last_real_excl = @intCast(l + 1),
    };
}

/// Copy the first `dim` components of a dense embedding; when truncating,
/// L2-renormalize so the result stays unit-length.
pub fn truncateAndRenormalize(allocator: std.mem.Allocator, src: []const f32, dim: usize) ![]f32 {
    const n = @min(dim, src.len);
    const out = try allocator.alloc(f32, n);
    @memcpy(out, src[0..n]);
    if (n < src.len) {
        var sq_sum: f32 = 0.0;
        for (out) |v| sq_sum += v * v;
        const norm = @sqrt(sq_sum + 1e-12);
        if (norm > 1e-12) {
            for (out) |*v| v.* /= norm;
        }
    }
    return out;
}

const SparseEntry = struct {
    idx: u32,
    val: f32,
};

fn sparseEntryValDesc(_: void, a: SparseEntry, b: SparseEntry) bool {
    if (a.val != b.val) return a.val > b.val;
    return a.idx < b.idx;
}

fn sparseEntryIdxAsc(_: void, a: SparseEntry, b: SparseEntry) bool {
    return a.idx < b.idx;
}

/// Convert a dense SPLADE activation to a sparse vector (positive activations
/// only), optionally keeping just the top-k activations by weight. Indices
/// are emitted in ascending order.
pub fn sparseFromDense(
    allocator: std.mem.Allocator,
    dense: []const f32,
    top_k: ?usize,
) !lib_chunker.types.SparseVector {
    var entries = std.ArrayListUnmanaged(SparseEntry).empty;
    defer entries.deinit(allocator);
    for (dense, 0..) |v, i| {
        if (v > 0.0) try entries.append(allocator, .{ .idx = @intCast(i), .val = v });
    }
    if (top_k) |k| {
        if (k > 0 and entries.items.len > k) {
            std.mem.sort(SparseEntry, entries.items, {}, sparseEntryValDesc);
            entries.shrinkRetainingCapacity(k);
            std.mem.sort(SparseEntry, entries.items, {}, sparseEntryIdxAsc);
        }
    }

    const indices = try allocator.alloc(i32, entries.items.len);
    errdefer allocator.free(indices);
    const values = try allocator.alloc(f32, entries.items.len);
    errdefer allocator.free(values);
    for (entries.items, 0..) |e, i| {
        indices[i] = @intCast(e.idx);
        values[i] = e.val;
    }
    return .{
        .indices = indices,
        .values = values,
        .owns_indices = true,
        .owns_values = true,
    };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "numWindows covers exact and ragged sequence lengths" {
    try std.testing.expectEqual(@as(usize, 0), numWindows(0, 384));
    try std.testing.expectEqual(@as(usize, 1), numWindows(1, 384));
    try std.testing.expectEqual(@as(usize, 1), numWindows(384, 384));
    try std.testing.expectEqual(@as(usize, 2), numWindows(385, 384));
    try std.testing.expectEqual(@as(usize, 3), numWindows(1000, 384));
}

test "decodeSpansWithFallback honors the request threshold" {
    const allocator = std.testing.allocator;

    // 8 tokens; position 4 has boundary probability sigmoid(1.0) ~ 0.73.
    const seq_len = 8;
    var logits = [_]f32{0.0} ** (seq_len * 2);
    for (0..seq_len) |i| {
        logits[i * 2 + 0] = 2.0;
        logits[i * 2 + 1] = -2.0;
    }
    logits[4 * 2 + 0] = -0.5;
    logits[4 * 2 + 1] = 0.5;

    var cfg = fused_chunker.Config{
        .min_chunk_tokens = 2,
        .max_chunk_tokens = 512,
        .boundary_threshold = 0.5,
    };

    // Threshold below the boundary probability: two chunks.
    const split = try decodeSpansWithFallback(allocator, cfg, &logits, seq_len);
    defer allocator.free(split);
    try std.testing.expectEqual(@as(usize, 2), split.len);
    try std.testing.expectEqual(@as(u32, 4), split[0].end_token);

    // Threshold above the boundary probability: no boundary fires, and the
    // whole sequence falls back to a single chunk.
    cfg.boundary_threshold = 0.9;
    const merged = try decodeSpansWithFallback(allocator, cfg, &logits, seq_len);
    defer allocator.free(merged);
    try std.testing.expectEqual(@as(usize, 1), merged.len);
    try std.testing.expectEqual(@as(u32, 0), merged[0].start_token);
    try std.testing.expectEqual(@as(u32, seq_len), merged[0].end_token);
}

test "decodeSpansWithFallback returns a single span for short documents" {
    const allocator = std.testing.allocator;

    // 5 tokens, min_chunk_tokens = 32 → decode drops everything.
    const seq_len = 5;
    var logits = [_]f32{0.0} ** (seq_len * 2);
    for (0..seq_len) |i| {
        logits[i * 2 + 0] = 2.0;
        logits[i * 2 + 1] = -2.0;
    }
    const cfg = fused_chunker.Config{};
    const spans = try decodeSpansWithFallback(allocator, cfg, &logits, seq_len);
    defer allocator.free(spans);
    try std.testing.expectEqual(@as(usize, 1), spans.len);
    try std.testing.expectEqual(@as(u32, 0), spans[0].start_token);
    try std.testing.expectEqual(@as(u32, seq_len), spans[0].end_token);
}

test "charSpanFromTokens skips special tokens and empty spans" {
    // [CLS] "Hello" " world" [SEP]
    const offsets = [_][2]u32{
        .{ 0, 0 },
        .{ 0, 5 },
        .{ 5, 11 },
        .{ 11, 11 },
    };

    const full = charSpanFromTokens(&offsets, 0, 4).?;
    try std.testing.expectEqual(@as(u32, 0), full.start_char);
    try std.testing.expectEqual(@as(u32, 11), full.end_char);
    try std.testing.expectEqual(@as(u32, 1), full.first_real);
    try std.testing.expectEqual(@as(u32, 3), full.last_real_excl);

    const tail = charSpanFromTokens(&offsets, 2, 4).?;
    try std.testing.expectEqual(@as(u32, 5), tail.start_char);
    try std.testing.expectEqual(@as(u32, 11), tail.end_char);

    // Only special tokens: no chunk.
    try std.testing.expectEqual(@as(?CharSpan, null), charSpanFromTokens(&offsets, 3, 4));
    try std.testing.expectEqual(@as(?CharSpan, null), charSpanFromTokens(&offsets, 0, 1));
}

test "truncateAndRenormalize truncates and restores unit norm" {
    const allocator = std.testing.allocator;

    // Unit vector in 4 dims.
    const src = [_]f32{ 0.5, 0.5, 0.5, 0.5 };

    const same = try truncateAndRenormalize(allocator, &src, 4);
    defer allocator.free(same);
    try std.testing.expectEqual(@as(usize, 4), same.len);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), same[0], 1e-6);

    const truncated = try truncateAndRenormalize(allocator, &src, 2);
    defer allocator.free(truncated);
    try std.testing.expectEqual(@as(usize, 2), truncated.len);
    // [0.5, 0.5] renormalized → [1/sqrt(2), 1/sqrt(2)].
    const expected = 1.0 / @sqrt(2.0);
    try std.testing.expectApproxEqAbs(@as(f32, expected), truncated[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, expected), truncated[1], 1e-5);
    var norm: f32 = 0.0;
    for (truncated) |v| norm += v * v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 1e-5);
}

test "sparseFromDense keeps positives and applies top-k by weight" {
    const allocator = std.testing.allocator;

    const dense = [_]f32{ 0.0, 2.0, 0.0, 0.5, 3.0, -1.0, 0.25 };

    var all = try sparseFromDense(allocator, &dense, null);
    defer all.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 4), all.indices.len);
    try std.testing.expectEqual(@as(i32, 1), all.indices[0]);
    try std.testing.expectEqual(@as(i32, 3), all.indices[1]);
    try std.testing.expectEqual(@as(i32, 4), all.indices[2]);
    try std.testing.expectEqual(@as(i32, 6), all.indices[3]);

    var top2 = try sparseFromDense(allocator, &dense, 2);
    defer top2.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), top2.indices.len);
    // Top-2 by weight are indices 4 (3.0) and 1 (2.0), emitted ascending.
    try std.testing.expectEqual(@as(i32, 1), top2.indices[0]);
    try std.testing.expectEqual(@as(i32, 4), top2.indices[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), top2.values[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), top2.values[1], 1e-6);
}

test "boundaryProbability is a stable 2-class softmax" {
    const logits = [_]f32{ 0.0, 0.0, -5.0, 5.0, 100.0, -100.0 };
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), boundaryProbability(&logits, 0), 1e-6);
    try std.testing.expect(boundaryProbability(&logits, 1) > 0.99);
    try std.testing.expect(boundaryProbability(&logits, 2) < 0.01);
}
