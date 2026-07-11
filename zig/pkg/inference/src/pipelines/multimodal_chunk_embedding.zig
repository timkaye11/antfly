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

//! Route A: multimodal chunk embedding.
//!
//! The deterministic `fixed_multimodal` chunker splits a document into chunks
//! that may carry text (mime `text/*`), image (`image/*`) or audio (`audio/*`)
//! payloads. This module embeds each chunk through a single, request-selected
//! shared-space embedder (CLIP for text+image, CLAP for text+audio) so that a
//! text query can retrieve image/audio chunks (cross-modal retrieval).
//!
//! Every chunk is routed to the tower matching its modality:
//!   text chunk  -> text tower   (pipeline.embed)
//!   image chunk -> vision tower (pipeline.embedImages)
//!   audio chunk -> audio tower  (pipeline.embedEncodedAudio)
//! All chunk vectors therefore live in the same projection space of the one
//! chosen model. If a chunk's modality is not supported by the chosen model
//! (e.g. an audio chunk with a CLIP model) the caller receives a clear error.
//!
//! The same mechanism generalizes to SPLIT-ENCODER serving for the fused
//! chunker (`embedTextChunks`): boundaries come from the fine-tuned fused
//! model (boundary-only forward), while each resulting text chunk is embedded
//! through the raw frozen embed-base named by the request, optionally with a
//! retrieval document prefix (nomic "search_document: "). This preserves the
//! base encoder's retrieval quality, which the fused fine-tune catastrophically
//! forgot (doc-level NDCG@10 0.0096 fused vs 0.335+ raw base, 2026-07).

const std = @import("std");
const embedding_mod = @import("embedding.zig");
const fused_chunking = @import("fused_chunking.zig");
const manifest_mod = @import("../models/manifest.zig");
const chunker = @import("inference_chunker");

const Chunk = chunker.types.Chunk;

pub const Modality = enum { text, image, audio };

/// Which chunk modalities the chosen embedder can project into its shared space.
/// Text is always available for a text-capable embedder; image/audio depend on
/// the model advertising a vision/audio tower.
pub const ModalityCaps = struct {
    text: bool = true,
    image: bool = false,
    audio: bool = false,

    /// Derive capabilities from a resolved model manifest. CLIP exposes a
    /// vision tower (native `clip` hint or a `visual_model_path`); CLAP exposes
    /// an audio tower (native `clap` hint or an `audio_model_path`).
    ///
    /// A packaged CLIP→text bridge head (`has_bridge`) makes the embedder
    /// image-only: its bridged image vectors live in a text embedder's retrieval
    /// space, but its own text tower produces native CLIP text vectors (a
    /// different space), so text chunks must be rejected to prevent cross-modal
    /// misuse. Text queries are served by the separate text embedder.
    pub fn fromManifest(m: *const manifest_mod.ModelManifest) ModalityCaps {
        return .{
            .text = !m.has_bridge,
            .image = m.native_arch_hint == .clip or m.visual_model_path != null,
            .audio = m.native_arch_hint == .clap or m.audio_model_path != null,
        };
    }

    /// True when the model can embed at least one non-text modality; a plain
    /// text embedder cannot serve Route A and should be rejected upstream.
    pub fn isMultimodal(self: ModalityCaps) bool {
        return self.image or self.audio;
    }

    pub fn supports(self: ModalityCaps, m: Modality) bool {
        return switch (m) {
            .text => self.text,
            .image => self.image,
            .audio => self.audio,
        };
    }
};

pub const EmbedOptions = struct {
    /// Truncate each dense vector to this dimension and L2-renormalize
    /// (Matryoshka). Null keeps the model's native dimension.
    output_dimension: ?u32 = null,
};

pub const ClassifyError = error{
    /// mime type is neither text/*, image/* nor audio/* (e.g. a passthrough
    /// application/* binary chunk that has no embedding tower).
    UnknownModality,
    /// The chunk lacks the payload its modality requires (text/data).
    MissingChunkPayload,
};

pub const EmbedError = ClassifyError || error{
    /// The chosen model has no tower for a chunk's modality.
    UnsupportedModality,
    /// output_dimension exceeds the model's native embedding size.
    InvalidOutputDimension,
};

/// Classify a chunk by its mime type. Text chunks carry `text`; image and audio
/// chunks carry `data`.
pub fn classifyChunk(chunk: Chunk) ClassifyError!Modality {
    const mime = chunk.mime_type;
    if (std.mem.startsWith(u8, mime, "text/")) {
        if (chunk.text == null) return error.MissingChunkPayload;
        return .text;
    }
    if (std.mem.startsWith(u8, mime, "image/")) {
        if (chunk.data == null) return error.MissingChunkPayload;
        return .image;
    }
    if (std.mem.startsWith(u8, mime, "audio/")) {
        if (chunk.data == null) return error.MissingChunkPayload;
        return .audio;
    }
    return error.UnknownModality;
}

/// Embed every chunk in `chunks` through the single shared-space `pipeline`,
/// grouping by modality and dispatching each group to the matching tower.
/// Results are written into `chunk.embedding` (+ `embedding_dimension`,
/// `owns_embedding`) in the original chunk order.
///
/// `pipeline` is duck-typed (anytype) so tests can substitute a synthetic
/// implementation exposing `embed`, `embedImages` and `embedEncodedAudio`
/// without a real CLIP/CLAP forward pass. In production this is a
/// `*embedding_mod.EmbeddingPipeline`.
pub fn embedChunks(
    allocator: std.mem.Allocator,
    pipeline: anytype,
    chunks: []Chunk,
    caps: ModalityCaps,
    opts: EmbedOptions,
) !void {
    if (chunks.len == 0) return;

    // Partition chunk indices by modality, preserving the original order so we
    // can scatter each tower's batched results back to the right chunks.
    var text_idx = std.ArrayListUnmanaged(usize).empty;
    defer text_idx.deinit(allocator);
    var image_idx = std.ArrayListUnmanaged(usize).empty;
    defer image_idx.deinit(allocator);
    var audio_idx = std.ArrayListUnmanaged(usize).empty;
    defer audio_idx.deinit(allocator);

    for (chunks, 0..) |chunk, i| {
        const modality = try classifyChunk(chunk);
        if (!caps.supports(modality)) return error.UnsupportedModality;
        switch (modality) {
            .text => try text_idx.append(allocator, i),
            .image => try image_idx.append(allocator, i),
            .audio => try audio_idx.append(allocator, i),
        }
    }

    if (text_idx.items.len > 0) {
        const texts = try allocator.alloc([]const u8, text_idx.items.len);
        defer allocator.free(texts);
        for (text_idx.items, 0..) |ci, k| texts[k] = chunks[ci].text.?;
        const embeddings = try pipeline.embed(texts);
        try assignEmbeddings(allocator, chunks, text_idx.items, embeddings, opts);
    }

    if (image_idx.items.len > 0) {
        const images = try allocator.alloc([]const u8, image_idx.items.len);
        defer allocator.free(images);
        for (image_idx.items, 0..) |ci, k| images[k] = chunks[ci].data.?;
        const embeddings = try pipeline.embedImages(images);
        try assignEmbeddings(allocator, chunks, image_idx.items, embeddings, opts);
    }

    if (audio_idx.items.len > 0) {
        const clips = try allocator.alloc(embedding_mod.EncodedAudioClip, audio_idx.items.len);
        defer allocator.free(clips);
        for (audio_idx.items, 0..) |ci, k| clips[k] = .{
            .bytes = chunks[ci].data.?,
            .decode_options = .{ .mime_hint = chunks[ci].mime_type },
        };
        const embeddings = try pipeline.embedEncodedAudio(clips);
        try assignEmbeddings(allocator, chunks, audio_idx.items, embeddings, opts);
    }
}

pub const TextEmbedOptions = struct {
    /// Truncate each dense vector to this dimension and L2-renormalize
    /// (Matryoshka). Null keeps the model's native dimension.
    output_dimension: ?u32 = null,
    /// Prefix prepended to each chunk's text before embedding, for
    /// prefix-conditioned retrieval embedders (e.g. nomic's
    /// "search_document: "). Empty applies no extra prefix; the embedding
    /// pipeline's own manifest-level text_prefix (e.g. jina v5's "Document: ")
    /// still applies inside pipeline.embed, so the two never stack unless the
    /// model declares both.
    prefix: []const u8 = "",
};

/// Split-encoder serving: embed TEXT chunks (e.g. produced boundary-only by
/// the fused chunker) through a single frozen text embedder so retrieval
/// quality comes from the raw base encoder rather than the fine-tuned fused
/// weights. Results are written into `chunk.embedding` (+
/// `embedding_dimension`, `owns_embedding`) in the original chunk order.
///
/// `pipeline` is duck-typed (anytype) so tests can substitute a synthetic
/// implementation exposing `embed` without a real forward pass. In production
/// this is a `*embedding_mod.EmbeddingPipeline`.
pub fn embedTextChunks(
    allocator: std.mem.Allocator,
    pipeline: anytype,
    chunks: []Chunk,
    opts: TextEmbedOptions,
) !void {
    if (chunks.len == 0) return;

    for (chunks) |chunk| {
        if (try classifyChunk(chunk) != .text) return error.UnsupportedModality;
    }

    const texts = try allocator.alloc([]const u8, chunks.len);
    defer allocator.free(texts);
    var prefixed_count: usize = 0;
    defer if (opts.prefix.len > 0) for (texts[0..prefixed_count]) |t| allocator.free(t);
    for (chunks, texts) |chunk, *out| {
        if (opts.prefix.len > 0) {
            out.* = try std.fmt.allocPrint(allocator, "{s}{s}", .{ opts.prefix, chunk.text.? });
            prefixed_count += 1;
        } else {
            out.* = chunk.text.?;
        }
    }

    const embeddings = try pipeline.embed(texts);

    const indices = try allocator.alloc(usize, chunks.len);
    defer allocator.free(indices);
    for (indices, 0..) |*idx, i| idx.* = i;
    try assignEmbeddings(allocator, chunks, indices, embeddings, .{
        .output_dimension = opts.output_dimension,
    });
}

pub const SparseChunkOptions = struct {
    /// Keep only the top-k SPLADE activations per chunk (by weight); null keeps
    /// every positive activation. Indices are always emitted ascending.
    sparse_top_k: ?usize = null,
};

/// Split-encoder SPLADE serving: produce a sparse vocabulary-space vector for
/// each TEXT chunk from the SAME frozen embed-base that supplies the dense
/// vector, via its packaged SPLADE head (pipeline.embedSplade). Results are
/// written into `chunk.sparse_embedding` in the original chunk order.
///
/// Unlike the dense path, SPLADE is fed the RAW chunk text (no nomic
/// "search_document: " retrieval prefix): SPLADE max-pools over content tokens,
/// so a retrieval prefix would inject spurious "search"/"document" vocabulary
/// dimensions into every vector. The embedder's own unconditional manifest
/// text_prefix (empty for nomic) still applies inside the forward, matching the
/// dense path's non-stacking behavior.
///
/// `pipeline` is duck-typed (anytype) so tests can substitute a synthetic
/// implementation exposing `embedSplade` without a real forward pass.
pub fn embedTextChunksSparse(
    allocator: std.mem.Allocator,
    pipeline: anytype,
    chunks: []Chunk,
    opts: SparseChunkOptions,
) !void {
    if (chunks.len == 0) return;

    for (chunks) |chunk| {
        if (try classifyChunk(chunk) != .text) return error.UnsupportedModality;
    }

    const texts = try allocator.alloc([]const u8, chunks.len);
    defer allocator.free(texts);
    for (chunks, texts) |chunk, *out| out.* = chunk.text.?;

    const dense = try pipeline.embedSplade(texts);
    defer {
        for (dense) |vec| allocator.free(vec);
        allocator.free(dense);
    }
    if (dense.len != chunks.len) return error.UnexpectedOutputShape;

    for (chunks, 0..) |*chunk, i| {
        // Sparsify the dense SPLADE activation (positive-only, top_k, ascending
        // indices) via the shared chunker helper. Order is preserved because
        // dense[i] corresponds to chunks[i].
        var sparse = try fused_chunking.sparseFromDense(allocator, dense[i], opts.sparse_top_k);
        if (chunk.sparse_embedding) |*old| old.deinit(allocator);
        chunk.sparse_embedding = sparse;
        sparse = undefined;
    }
}

/// Take ownership of `embeddings` (an owned `[][]f32` returned by a tower) and
/// write each vector into the chunk at the matching original index. Applies
/// output_dimension truncation+renormalization when requested. Preserves order
/// because `indices[k]` is the original position of `embeddings[k]`.
fn assignEmbeddings(
    allocator: std.mem.Allocator,
    chunks: []Chunk,
    indices: []const usize,
    embeddings: [][]f32,
    opts: EmbedOptions,
) !void {
    defer allocator.free(embeddings);
    // Free any tower outputs we have not yet handed to a chunk if we bail out.
    var handled: usize = 0;
    errdefer for (embeddings[handled..]) |vec| allocator.free(vec);

    for (indices, 0..) |ci, k| {
        var vec = embeddings[k];
        if (opts.output_dimension) |dim| {
            if (dim > vec.len) return error.InvalidOutputDimension;
            if (dim < vec.len) {
                const truncated = try fused_chunking.truncateAndRenormalize(allocator, vec, dim);
                allocator.free(vec);
                vec = truncated;
            }
        }
        var chunk = &chunks[ci];
        if (chunk.owns_embedding) {
            if (chunk.embedding) |old| allocator.free(old);
        }
        chunk.embedding = vec;
        chunk.embedding_dimension = @intCast(vec.len);
        chunk.owns_embedding = true;
        handled += 1;
    }
}

// ---------------------------------------------------------------------------
// Tests: routing + order preservation using a synthetic pipeline (no GPU).
// ---------------------------------------------------------------------------

/// Minimal stand-in for EmbeddingPipeline. Each tower tags its vectors with a
/// per-modality base value plus the in-batch index so tests can assert both
/// which tower produced a chunk's vector and that batch order is preserved.
const MockPipeline = struct {
    allocator: std.mem.Allocator,
    dim: usize = 4,
    text_calls: usize = 0,
    image_calls: usize = 0,
    audio_calls: usize = 0,

    fn make(self: *MockPipeline, n: usize, tag: f32) ![][]f32 {
        const out = try self.allocator.alloc([]f32, n);
        errdefer self.allocator.free(out);
        for (out, 0..) |*v, i| {
            v.* = try self.allocator.alloc(f32, self.dim);
            v.*[0] = tag + @as(f32, @floatFromInt(i));
            for (v.*[1..]) |*x| x.* = 0.0;
        }
        return out;
    }
    pub fn embed(self: *MockPipeline, texts: []const []const u8) ![][]f32 {
        self.text_calls += 1;
        return self.make(texts.len, 100.0);
    }
    pub fn embedImages(self: *MockPipeline, images: []const []const u8) ![][]f32 {
        self.image_calls += 1;
        return self.make(images.len, 200.0);
    }
    pub fn embedEncodedAudio(self: *MockPipeline, clips: []const embedding_mod.EncodedAudioClip) ![][]f32 {
        self.audio_calls += 1;
        return self.make(clips.len, 300.0);
    }
};

test "classifyChunk maps mime types to modalities" {
    try std.testing.expectEqual(Modality.text, try classifyChunk(Chunk.initText(0, "hi", 0, 2)));
    try std.testing.expectEqual(Modality.image, try classifyChunk(Chunk.initBinary(0, "image/png", "x")));
    try std.testing.expectEqual(Modality.audio, try classifyChunk(Chunk.initBinary(0, "audio/wav", "x")));
    try std.testing.expectError(error.UnknownModality, classifyChunk(Chunk.initBinary(0, "application/pdf", "x")));
}

test "ModalityCaps derives towers from manifest" {
    var clip = manifest_mod.ModelManifest{ .allocator = std.testing.allocator, .native_arch_hint = .clip };
    const clip_caps = ModalityCaps.fromManifest(&clip);
    try std.testing.expect(clip_caps.text and clip_caps.image and !clip_caps.audio);
    try std.testing.expect(clip_caps.isMultimodal());

    var clap = manifest_mod.ModelManifest{ .allocator = std.testing.allocator, .native_arch_hint = .clap };
    const clap_caps = ModalityCaps.fromManifest(&clap);
    try std.testing.expect(clap_caps.text and !clap_caps.image and clap_caps.audio);

    var plain = manifest_mod.ModelManifest{ .allocator = std.testing.allocator };
    const plain_caps = ModalityCaps.fromManifest(&plain);
    try std.testing.expect(plain_caps.text and !plain_caps.isMultimodal());

    // A clipclap image embedder carrying a CLIP→text bridge head is image-only:
    // its bridged image vectors live in the text retrieval space, so text chunks
    // are rejected (served by the separate text embedder).
    var bridged = manifest_mod.ModelManifest{ .allocator = std.testing.allocator, .native_arch_hint = .clip, .has_bridge = true };
    const bridged_caps = ModalityCaps.fromManifest(&bridged);
    try std.testing.expect(!bridged_caps.text and bridged_caps.image and !bridged_caps.audio);
    try std.testing.expect(bridged_caps.isMultimodal());
    try std.testing.expect(!bridged_caps.supports(.text) and bridged_caps.supports(.image));
}

test "embedChunks routes each modality to its tower and preserves order" {
    const alloc = std.testing.allocator;
    var chunks = [_]Chunk{
        Chunk.initText(0, "first text", 0, 10),
        Chunk.initBinary(1, "image/png", "img0"),
        Chunk.initText(2, "second text", 10, 21),
        Chunk.initBinary(3, "image/png", "img1"),
    };
    defer for (&chunks) |*c| c.deinit(alloc);

    var pipeline = MockPipeline{ .allocator = alloc };
    try embedChunks(alloc, &pipeline, &chunks, .{ .text = true, .image = true }, .{});

    // Each tower called exactly once with its batch.
    try std.testing.expectEqual(@as(usize, 1), pipeline.text_calls);
    try std.testing.expectEqual(@as(usize, 1), pipeline.image_calls);
    try std.testing.expectEqual(@as(usize, 0), pipeline.audio_calls);

    // Order preserved: text batch is [chunk0, chunk2], image batch is [chunk1, chunk3].
    try std.testing.expectEqual(@as(f32, 100.0), chunks[0].embedding.?[0]); // text idx 0
    try std.testing.expectEqual(@as(f32, 200.0), chunks[1].embedding.?[0]); // image idx 0
    try std.testing.expectEqual(@as(f32, 101.0), chunks[2].embedding.?[0]); // text idx 1
    try std.testing.expectEqual(@as(f32, 201.0), chunks[3].embedding.?[0]); // image idx 1
    for (&chunks) |c| try std.testing.expectEqual(@as(?u32, 4), c.embedding_dimension);
}

test "embedChunks rejects modality unsupported by the chosen model" {
    const alloc = std.testing.allocator;
    var chunks = [_]Chunk{
        Chunk.initText(0, "text", 0, 4),
        Chunk.initBinary(1, "audio/wav", "aud0"),
    };
    defer for (&chunks) |*c| c.deinit(alloc);

    var pipeline = MockPipeline{ .allocator = alloc };
    // CLIP-like caps: text+image, no audio tower.
    try std.testing.expectError(
        error.UnsupportedModality,
        embedChunks(alloc, &pipeline, &chunks, .{ .text = true, .image = true }, .{}),
    );
}

test "embedChunks rejects unknown passthrough modality" {
    const alloc = std.testing.allocator;
    var chunks = [_]Chunk{Chunk.initBinary(0, "application/pdf", "%PDF")};
    defer for (&chunks) |*c| c.deinit(alloc);

    var pipeline = MockPipeline{ .allocator = alloc };
    try std.testing.expectError(
        error.UnknownModality,
        embedChunks(alloc, &pipeline, &chunks, .{ .text = true, .image = true, .audio = true }, .{}),
    );
}

test "embedChunks truncates and renormalizes to output_dimension" {
    const alloc = std.testing.allocator;
    var chunks = [_]Chunk{Chunk.initBinary(0, "image/png", "img")};
    defer for (&chunks) |*c| c.deinit(alloc);

    var pipeline = MockPipeline{ .allocator = alloc, .dim = 4 };
    try embedChunks(alloc, &pipeline, &chunks, .{ .text = true, .image = true }, .{ .output_dimension = 2 });

    const emb = chunks[0].embedding.?;
    try std.testing.expectEqual(@as(usize, 2), emb.len);
    try std.testing.expectEqual(@as(?u32, 2), chunks[0].embedding_dimension);
    // Vector was [200,0,0,0]; truncated+renormalized to unit length -> [1,0].
    var norm: f32 = 0;
    for (emb) |v| norm += v * v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 1e-5);
}

test "embedChunks rejects output_dimension larger than native size" {
    const alloc = std.testing.allocator;
    var chunks = [_]Chunk{Chunk.initBinary(0, "image/png", "img")};
    defer for (&chunks) |*c| c.deinit(alloc);

    var pipeline = MockPipeline{ .allocator = alloc, .dim = 4 };
    try std.testing.expectError(
        error.InvalidOutputDimension,
        embedChunks(alloc, &pipeline, &chunks, .{ .text = true, .image = true }, .{ .output_dimension = 8 }),
    );
}

// ---------------------------------------------------------------------------
// Split-encoder text embedding tests (mock text embedder, no forward pass).
// ---------------------------------------------------------------------------

/// Text-only stand-in for EmbeddingPipeline. Records every text it receives
/// so tests can assert prefix application and batch order, and tags each
/// vector with its in-batch index so order preservation is observable.
const MockTextPipeline = struct {
    allocator: std.mem.Allocator,
    dim: usize = 4,
    calls: usize = 0,
    seen_texts: std.ArrayListUnmanaged([]u8) = .empty,

    fn deinit(self: *MockTextPipeline) void {
        for (self.seen_texts.items) |t| self.allocator.free(t);
        self.seen_texts.deinit(self.allocator);
    }

    pub fn embed(self: *MockTextPipeline, texts: []const []const u8) ![][]f32 {
        self.calls += 1;
        for (texts) |t| try self.seen_texts.append(self.allocator, try self.allocator.dupe(u8, t));
        const out = try self.allocator.alloc([]f32, texts.len);
        errdefer self.allocator.free(out);
        for (out, 0..) |*v, i| {
            v.* = try self.allocator.alloc(f32, self.dim);
            v.*[0] = 1.0 + @as(f32, @floatFromInt(i));
            for (v.*[1..]) |*x| x.* = 0.0;
        }
        return out;
    }
};

test "embedTextChunks embeds one batch in order with the document prefix" {
    const alloc = std.testing.allocator;
    var chunks = [_]Chunk{
        Chunk.initText(0, "first chunk", 0, 11),
        Chunk.initText(1, "second chunk", 11, 23),
        Chunk.initText(2, "third chunk", 23, 34),
    };
    defer for (&chunks) |*c| c.deinit(alloc);

    var pipeline = MockTextPipeline{ .allocator = alloc };
    defer pipeline.deinit();
    try embedTextChunks(alloc, &pipeline, &chunks, .{ .prefix = "search_document: " });

    // One batched embed call; each text carries the nomic document prefix in
    // original chunk order.
    try std.testing.expectEqual(@as(usize, 1), pipeline.calls);
    try std.testing.expectEqual(@as(usize, 3), pipeline.seen_texts.items.len);
    try std.testing.expectEqualStrings("search_document: first chunk", pipeline.seen_texts.items[0]);
    try std.testing.expectEqualStrings("search_document: second chunk", pipeline.seen_texts.items[1]);
    try std.testing.expectEqualStrings("search_document: third chunk", pipeline.seen_texts.items[2]);

    // Order preserved: chunk i carries the i-th batch vector.
    try std.testing.expectEqual(@as(f32, 1.0), chunks[0].embedding.?[0]);
    try std.testing.expectEqual(@as(f32, 2.0), chunks[1].embedding.?[0]);
    try std.testing.expectEqual(@as(f32, 3.0), chunks[2].embedding.?[0]);
    for (&chunks) |c| {
        try std.testing.expectEqual(@as(?u32, 4), c.embedding_dimension);
        try std.testing.expect(c.owns_embedding);
    }
}

test "embedTextChunks without prefix passes chunk text verbatim" {
    const alloc = std.testing.allocator;
    var chunks = [_]Chunk{Chunk.initText(0, "plain text", 0, 10)};
    defer for (&chunks) |*c| c.deinit(alloc);

    var pipeline = MockTextPipeline{ .allocator = alloc };
    defer pipeline.deinit();
    try embedTextChunks(alloc, &pipeline, &chunks, .{});

    try std.testing.expectEqualStrings("plain text", pipeline.seen_texts.items[0]);
    try std.testing.expect(chunks[0].embedding != null);
}

test "embedTextChunks truncates and renormalizes to output_dimension" {
    const alloc = std.testing.allocator;
    var chunks = [_]Chunk{Chunk.initText(0, "chunk", 0, 5)};
    defer for (&chunks) |*c| c.deinit(alloc);

    var pipeline = MockTextPipeline{ .allocator = alloc, .dim = 4 };
    defer pipeline.deinit();
    try embedTextChunks(alloc, &pipeline, &chunks, .{ .output_dimension = 2 });

    const emb = chunks[0].embedding.?;
    try std.testing.expectEqual(@as(usize, 2), emb.len);
    try std.testing.expectEqual(@as(?u32, 2), chunks[0].embedding_dimension);
    var norm: f32 = 0;
    for (emb) |v| norm += v * v;
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 1e-5);
}

test "embedTextChunks rejects output_dimension larger than native size" {
    const alloc = std.testing.allocator;
    var chunks = [_]Chunk{Chunk.initText(0, "chunk", 0, 5)};
    defer for (&chunks) |*c| c.deinit(alloc);

    var pipeline = MockTextPipeline{ .allocator = alloc, .dim = 4 };
    defer pipeline.deinit();
    try std.testing.expectError(
        error.InvalidOutputDimension,
        embedTextChunks(alloc, &pipeline, &chunks, .{ .output_dimension = 8 }),
    );
}

test "embedTextChunks rejects non-text chunks" {
    const alloc = std.testing.allocator;
    var chunks = [_]Chunk{
        Chunk.initText(0, "text", 0, 4),
        Chunk.initBinary(1, "image/png", "img"),
    };
    defer for (&chunks) |*c| c.deinit(alloc);

    var pipeline = MockTextPipeline{ .allocator = alloc };
    defer pipeline.deinit();
    try std.testing.expectError(error.UnsupportedModality, embedTextChunks(alloc, &pipeline, &chunks, .{}));
    // The pipeline was never invoked for a rejected batch.
    try std.testing.expectEqual(@as(usize, 0), pipeline.calls);
}

// ---------------------------------------------------------------------------
// Split-encoder SPLADE serving tests (mock SPLADE embedder, no forward pass).
// The mock returns a synthetic dense vocab activation per chunk so the tests
// exercise the sparsify+top_k+order-preservation dispatch without a real
// encoder/projection.
// ---------------------------------------------------------------------------

/// Stand-in exposing `embedSplade`. Each chunk i gets a dense vocab vector with
/// a distinct activation pattern so tests can assert order preservation and
/// top_k selection. Records the texts seen (raw, un-prefixed) for assertion.
const MockSpladePipeline = struct {
    allocator: std.mem.Allocator,
    vocab: usize = 6,
    calls: usize = 0,
    seen_texts: std.ArrayListUnmanaged([]u8) = .empty,

    fn deinit(self: *MockSpladePipeline) void {
        for (self.seen_texts.items) |t| self.allocator.free(t);
        self.seen_texts.deinit(self.allocator);
    }

    pub fn embedSplade(self: *MockSpladePipeline, texts: []const []const u8) ![][]f32 {
        self.calls += 1;
        for (texts) |t| try self.seen_texts.append(self.allocator, try self.allocator.dupe(u8, t));
        const out = try self.allocator.alloc([]f32, texts.len);
        errdefer self.allocator.free(out);
        for (out, 0..) |*v, i| {
            v.* = try self.allocator.alloc(f32, self.vocab);
            @memset(v.*, 0);
            // Distinct positive activations per chunk: dims (i % vocab) and
            // ((i+2) % vocab) plus a small negative that must be dropped.
            v.*[i % self.vocab] = 3.0 + @as(f32, @floatFromInt(i));
            v.*[(i + 2) % self.vocab] = 1.0;
            v.*[(i + 4) % self.vocab] = -5.0; // dropped by log1p(relu)
        }
        return out;
    }
};

test "embedTextChunksSparse writes ascending-index sparse vectors in chunk order" {
    const alloc = std.testing.allocator;
    var chunks = [_]Chunk{
        Chunk.initText(0, "alpha", 0, 5),
        Chunk.initText(1, "beta", 5, 9),
        Chunk.initText(2, "gamma", 9, 14),
    };
    defer for (&chunks) |*c| c.deinit(alloc);

    var pipeline = MockSpladePipeline{ .allocator = alloc };
    defer pipeline.deinit();
    try embedTextChunksSparse(alloc, &pipeline, &chunks, .{});

    try std.testing.expectEqual(@as(usize, 1), pipeline.calls);
    // SPLADE is fed RAW chunk text (no document prefix).
    try std.testing.expectEqualStrings("alpha", pipeline.seen_texts.items[0]);
    try std.testing.expectEqualStrings("gamma", pipeline.seen_texts.items[2]);

    // Every chunk carries a sparse vector: 2 positive dims, ascending indices,
    // and the negative dim dropped.
    for (&chunks) |c| {
        const sv = c.sparse_embedding orelse return error.TestMissingSparse;
        try std.testing.expectEqual(@as(usize, 2), sv.indices.len);
        try std.testing.expect(sv.indices[0] < sv.indices[1]);
        for (sv.values) |val| try std.testing.expect(val > 0);
    }
    // Order preserved, and sparseFromDense keeps the already-activated value
    // verbatim (log1p(relu) was applied inside embedSplade, mocked here as the
    // dense activation directly): chunk 0 peaks at dim 0 (3.0), chunk 1 (4.0).
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), maxSparseValue(chunks[0].sparse_embedding.?), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), maxSparseValue(chunks[1].sparse_embedding.?), 1e-6);
}

test "embedTextChunksSparse honors sparse_top_k cap" {
    const alloc = std.testing.allocator;
    var chunks = [_]Chunk{Chunk.initText(0, "alpha", 0, 5)};
    defer for (&chunks) |*c| c.deinit(alloc);

    var pipeline = MockSpladePipeline{ .allocator = alloc };
    defer pipeline.deinit();
    // Chunk 0 has 2 positive dims; cap to 1 keeps the largest (dim 0).
    try embedTextChunksSparse(alloc, &pipeline, &chunks, .{ .sparse_top_k = 1 });

    const sv = chunks[0].sparse_embedding orelse return error.TestMissingSparse;
    try std.testing.expectEqual(@as(usize, 1), sv.indices.len);
    try std.testing.expectEqual(@as(i32, 0), sv.indices[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), sv.values[0], 1e-6);
}

test "embedTextChunksSparse rejects non-text chunks without invoking the head" {
    const alloc = std.testing.allocator;
    var chunks = [_]Chunk{
        Chunk.initText(0, "text", 0, 4),
        Chunk.initBinary(1, "image/png", "img"),
    };
    defer for (&chunks) |*c| c.deinit(alloc);

    var pipeline = MockSpladePipeline{ .allocator = alloc };
    defer pipeline.deinit();
    try std.testing.expectError(error.UnsupportedModality, embedTextChunksSparse(alloc, &pipeline, &chunks, .{}));
    try std.testing.expectEqual(@as(usize, 0), pipeline.calls);
}

fn maxSparseValue(sv: chunker.types.SparseVector) f32 {
    var m: f32 = 0;
    for (sv.values) |v| {
        if (v > m) m = v;
    }
    return m;
}
