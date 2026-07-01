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

// Integration test for the fused chunker serving pipeline
// (src/pipelines/fused_chunking.zig) against an exported model directory.
//
// Requires a model exported by tools/export_fused_chunker_model.zig at
// $ANTFLY_FUSED_CHUNKER_EXPORT_DIR (default: /private/tmp/fused_export_smoke).
// Skips when the directory is absent. Runs CPU-only (native backend) so it is
// safe while the GPU is busy; the exported weights are ~600MB of RAM.

const std = @import("std");
const inference = @import("inference_internal");

const fused_chunking = inference.pipelines.fused_chunking;
const lib_chunker = inference.chunker;

const default_export_dir = "/private/tmp/fused_export_smoke";

fn resolveExportDir(buf: []u8) ?[]const u8 {
    const compat = inference.io.compat;
    const dir: []const u8 = if (std.c.getenv("ANTFLY_FUSED_CHUNKER_EXPORT_DIR")) |env|
        std.mem.span(env)
    else
        default_export_dir;
    if (dir.len + "/model.safetensors".len + 1 > buf.len) return null;
    const st_path = std.fmt.bufPrint(buf, "{s}/model.safetensors", .{dir}) catch return null;
    _ = compat.cwd().statFile(compat.io(), st_path, .{}) catch return null;
    return dir;
}

const sample_text =
    "The Antfly storage engine organizes data into shards, each backed by its " ++
    "own Pebble instance and Raft consensus group. Writes flow through the Raft " ++
    "log before being applied to the local key-value store, which keeps every " ++
    "replica consistent even when nodes fail or restart during heavy load.\n\n" ++
    "Vector search works differently. Each shard maintains an embedding index " ++
    "that is enriched asynchronously: documents are chunked, embedded by the " ++
    "inference service, and inserted into an approximate nearest neighbor " ++
    "structure. Queries combine BM25 scores with vector similarity to produce " ++
    "a single hybrid ranking across all shards of the table.\n\n" ++
    "Operationally, the metadata service coordinates topology changes. When a " ++
    "node joins the cluster it receives shard assignments, replays the Raft " ++
    "log, and begins serving reads once it has caught up with the leader.";

test "fused chunker pipeline serves an exported model CPU-only" {
    var path_buf: [512]u8 = undefined;
    const export_dir = resolveExportDir(&path_buf) orelse return error.SkipZigTest;

    const allocator = std.testing.allocator;

    var pipeline = try fused_chunking.FusedChunkerPipeline.loadFromDir(allocator, export_dir, .{
        .backend = .native,
    });
    defer pipeline.deinit();

    try std.testing.expect(pipeline.model_version.len > 0);
    try std.testing.expect(pipeline.max_seq_len > 0);

    // Boundary-only request (no embeddings).
    {
        const chunks = try pipeline.chunkText(allocator, sample_text, .{ .max_chunks = 16 });
        defer lib_chunker.types.freeChunks(allocator, chunks);

        try std.testing.expect(chunks.len > 0);
        for (chunks) |chunk| {
            try std.testing.expect(chunk.boundary_score != null);
            try std.testing.expect(chunk.text != null);
            try std.testing.expect(chunk.start_char != null);
            try std.testing.expect(chunk.end_char.? > chunk.start_char.?);
            try std.testing.expect(chunk.embedding == null);
            try std.testing.expectEqualStrings(pipeline.model_version, chunk.model_version.?);
            try std.testing.expectEqualStrings(
                inference.finetune.fused_chunker.artifact_family_version,
                chunk.artifact_family.?,
            );
        }
    }

    // Dense embeddings at full dimensionality.
    {
        const chunks = try pipeline.chunkText(allocator, sample_text, .{
            .include_embeddings = true,
            .max_chunks = 16,
        });
        defer lib_chunker.types.freeChunks(allocator, chunks);

        try std.testing.expect(chunks.len > 0);
        const expected_dim: usize = @intCast(pipeline.config.embedding_dim);
        try std.testing.expectEqual(@as(usize, 768), expected_dim);
        for (chunks) |chunk| {
            const embedding = chunk.embedding orelse return error.TestExpectedEmbedding;
            try std.testing.expectEqual(expected_dim, embedding.len);
            try std.testing.expectEqual(@as(u32, 768), chunk.embedding_dimension.?);
            var norm: f64 = 0.0;
            for (embedding) |v| norm += @as(f64, v) * @as(f64, v);
            try std.testing.expectApproxEqAbs(@as(f64, 1.0), norm, 1e-2);
        }
    }

    // Truncated output_dimension keeps unit norm.
    {
        const chunks = try pipeline.chunkText(allocator, sample_text, .{
            .include_embeddings = true,
            .output_dimension = 256,
            .max_chunks = 4,
        });
        defer lib_chunker.types.freeChunks(allocator, chunks);

        try std.testing.expect(chunks.len > 0);
        for (chunks) |chunk| {
            const embedding = chunk.embedding orelse return error.TestExpectedEmbedding;
            try std.testing.expectEqual(@as(usize, 256), embedding.len);
            try std.testing.expectEqual(@as(u32, 256), chunk.embedding_dimension.?);
            var norm: f64 = 0.0;
            for (embedding) |v| norm += @as(f64, v) * @as(f64, v);
            try std.testing.expectApproxEqAbs(@as(f64, 1.0), norm, 1e-2);
        }
    }

    // Requesting SPLADE from a model without a SPLADE head is a clear error.
    if (!pipeline.has_splade) {
        try std.testing.expectError(
            error.SparseEmbeddingsUnsupported,
            pipeline.chunkText(allocator, sample_text, .{ .include_sparse = true }),
        );
    }
}
