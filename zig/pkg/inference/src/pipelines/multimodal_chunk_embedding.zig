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
    pub fn fromManifest(m: *const manifest_mod.ModelManifest) ModalityCaps {
        return .{
            .text = true,
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
