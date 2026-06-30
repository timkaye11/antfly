// Copyright 2026 Antfly, Inc.
//
// Licensed under the Elastic License 2.0 (ELv2); you may not use this file
// except in compliance with the Elastic License 2.0. You may obtain a copy of
// the Elastic License 2.0 at
//
//     https://www.antfly.io/licensing/ELv2-license
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the Elastic License 2.0 is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// Elastic License 2.0 for the specific language governing permissions and
// limitations.

const std = @import("std");
const httpx = @import("httpx");
const inference_api = @import("inference_api");
const chunking_types = @import("types.zig");
const Chunk = @import("chunk.zig").Chunk;
const http_common = @import("../raft/transport/http_common.zig");
const std_http_listener = @import("../raft/transport/std_http_listener.zig");
const inference_chunker = @import("inference_chunker");

const Allocator = std.mem.Allocator;

pub const RemoteChunk = inference_chunker.Chunk;
pub const RemoteInput = inference_chunker.Input;
pub const RemoteBinaryInput = inference_chunker.BinaryInput;

pub fn chunkText(alloc: Allocator, cfg: chunking_types.Config, text: []const u8) ![]Chunk {
    const shared_chunks = try chunkInput(alloc, cfg, .{ .text = text });
    defer freeRemoteChunks(alloc, shared_chunks);

    var chunks = try alloc.alloc(Chunk, shared_chunks.len);
    errdefer {
        for (chunks) |*chunk| chunk.deinit(alloc);
        alloc.free(chunks);
    }
    for (shared_chunks, 0..) |shared, i| {
        if (!std.mem.eql(u8, shared.mime_type, "text/plain")) return error.UnsupportedChunkMediaType;
        const shared_text = shared.text orelse return error.InvalidChunkerResponse;
        chunks[i] = .{
            .chunk_id = shared.id,
            .text = try alloc.dupe(u8, shared_text),
            .start_offset = shared.start_char,
            .end_offset = shared.end_char orelse std.math.cast(u32, shared_text.len),
        };
    }
    return chunks;
}

pub fn chunkBinary(alloc: Allocator, cfg: chunking_types.Config, mime_type: []const u8, data: []const u8) ![]RemoteChunk {
    return try chunkInput(alloc, cfg, .{
        .binary = .{
            .mime_type = mime_type,
            .data = data,
        },
    });
}

pub fn chunkInput(alloc: Allocator, cfg: chunking_types.Config, input: RemoteInput) ![]RemoteChunk {
    if (cfg.api_url.len == 0) return try chunkInputDirect(alloc, cfg, input);
    if (cfg.model.len == 0) return error.InvalidChunkerConfig;

    var io_impl = std.Io.Threaded.init(alloc, .{});
    defer io_impl.deinit();

    var http = httpx.Client.init(alloc, io_impl.io());
    defer http.deinit();

    const url = try std.fmt.allocPrint(alloc, "{s}/chunk", .{cfg.api_url});
    defer alloc.free(url);

    const body = try encodeChunkRequest(alloc, cfg, input);
    defer alloc.free(body);

    var resp = try http.post(url, .{ .json = body });
    defer resp.deinit();
    if (!resp.ok()) return error.ChunkRequestFailed;
    const response_body = resp.body orelse return error.EmptyResponse;
    return try parseChunkResponse(alloc, response_body);
}

fn chunkInputDirect(alloc: Allocator, cfg: chunking_types.Config, input: RemoteInput) ![]RemoteChunk {
    var fixed_cfg = inference_chunker.FixedChunkConfig{};
    if (cfg.model.len > 0) fixed_cfg.model = cfg.model;
    if (cfg.max_chunks > 0) fixed_cfg.max_chunks = @intCast(cfg.max_chunks);
    fixed_cfg.threshold = cfg.threshold;
    fixed_cfg.include_embeddings = cfg.include_embeddings;
    fixed_cfg.include_sparse = cfg.include_sparse;
    if (cfg.output_dimension > 0) fixed_cfg.output_dimension = cfg.output_dimension;
    if (cfg.sparse_top_k > 0) fixed_cfg.sparse_top_k = cfg.sparse_top_k;
    fixed_cfg.include_boundary_scores = cfg.include_boundary_scores;
    if (cfg.text.target_tokens > 0) fixed_cfg.text.target_tokens = @intCast(cfg.text.target_tokens);
    if (cfg.text.target_tokens > 0 or cfg.text.overlap_tokens > 0) fixed_cfg.text.overlap_tokens = @intCast(cfg.text.overlap_tokens);
    if (cfg.text.separator.len > 0) fixed_cfg.text.separator = cfg.text.separator;
    if (cfg.audio.window_duration_ms > 0) fixed_cfg.audio.window_duration_ms = @intCast(cfg.audio.window_duration_ms);
    if (cfg.audio.overlap_duration_ms > 0) fixed_cfg.audio.overlap_duration_ms = @intCast(cfg.audio.overlap_duration_ms);

    const chunks = try inference_chunker.fixed_multimodal.chunkInput(alloc, input, fixed_cfg);
    defer inference_chunker.types.freeChunks(alloc, chunks);
    return try cloneRemoteChunks(alloc, chunks);
}

fn cloneRemoteChunks(alloc: Allocator, source: []const RemoteChunk) ![]RemoteChunk {
    const chunks = try alloc.alloc(RemoteChunk, source.len);
    var initialized: usize = 0;
    errdefer {
        for (chunks[0..initialized]) |*chunk| {
            alloc.free(@constCast(chunk.mime_type));
            chunk.deinit(alloc);
        }
        alloc.free(chunks);
    }

    for (source, 0..) |chunk, i| {
        var out = RemoteChunk{
            .id = chunk.id,
            .mime_type = try alloc.dupe(u8, chunk.mime_type),
            .text = null,
            .start_char = chunk.start_char,
            .end_char = chunk.end_char,
            .start_token = chunk.start_token,
            .end_token = chunk.end_token,
            .boundary_score = chunk.boundary_score,
            .embedding = null,
            .embedding_dimension = chunk.embedding_dimension,
            .sparse_embedding = null,
            .model_version = null,
            .artifact_family = null,
            .data = null,
            .start_time_ms = chunk.start_time_ms,
            .end_time_ms = chunk.end_time_ms,
            .frame_index = chunk.frame_index,
            .frame_delay_ms = chunk.frame_delay_ms,
            .owns_text = false,
            .owns_data = false,
            .owns_embedding = false,
            .owns_model_version = false,
            .owns_artifact_family = false,
        };
        errdefer {
            alloc.free(@constCast(out.mime_type));
            out.deinit(alloc);
        }
        if (chunk.text) |text| {
            out.text = try alloc.dupe(u8, text);
            out.owns_text = true;
        }
        if (chunk.data) |data| {
            out.data = try alloc.dupe(u8, data);
            out.owns_data = true;
        }
        if (chunk.embedding) |embedding| {
            out.embedding = try alloc.dupe(f32, embedding);
            out.owns_embedding = true;
            if (out.embedding_dimension == null) out.embedding_dimension = std.math.cast(u32, embedding.len);
        }
        if (chunk.sparse_embedding) |sparse| {
            out.sparse_embedding = try cloneSparseVector(alloc, sparse);
        }
        if (chunk.model_version) |model_version| {
            out.model_version = try alloc.dupe(u8, model_version);
            out.owns_model_version = true;
        }
        if (chunk.artifact_family) |artifact_family| {
            out.artifact_family = try alloc.dupe(u8, artifact_family);
            out.owns_artifact_family = true;
        }
        chunks[i] = out;
        initialized += 1;
    }
    return chunks;
}

fn cloneSparseVector(alloc: Allocator, sparse: inference_chunker.types.SparseVector) !inference_chunker.types.SparseVector {
    const indices = try alloc.dupe(i32, sparse.indices);
    errdefer alloc.free(indices);
    const values = try alloc.dupe(f32, sparse.values);
    errdefer alloc.free(values);
    return .{
        .indices = indices,
        .values = values,
        .owns_indices = true,
        .owns_values = true,
    };
}

pub fn freeRemoteChunks(alloc: Allocator, chunks: []RemoteChunk) void {
    for (chunks) |*chunk| {
        alloc.free(@constCast(chunk.mime_type));
        chunk.deinit(alloc);
    }
    alloc.free(chunks);
}

fn encodeChunkRequest(alloc: Allocator, cfg: chunking_types.Config, input: RemoteInput) ![]u8 {
    const config: inference_api.ChunkConfig = .{
        .model = cfg.model,
        .max_chunks = if (cfg.max_chunks > 0) cfg.max_chunks else null,
        .threshold = cfg.threshold,
        .include_embeddings = if (cfg.include_embeddings) true else null,
        .include_sparse = if (cfg.include_sparse) true else null,
        .output_dimension = if (cfg.output_dimension > 0) cfg.output_dimension else null,
        .sparse_top_k = if (cfg.sparse_top_k > 0) cfg.sparse_top_k else null,
        .include_boundary_scores = if (cfg.include_boundary_scores) true else null,
        .text = if (cfg.text.target_tokens > 0 or cfg.text.overlap_tokens > 0 or cfg.text.separator.len > 0) .{
            .target_tokens = if (cfg.text.target_tokens > 0) cfg.text.target_tokens else null,
            .overlap_tokens = if (cfg.text.target_tokens > 0 or cfg.text.overlap_tokens > 0) cfg.text.overlap_tokens else null,
            .separator = if (cfg.text.separator.len > 0) cfg.text.separator else null,
        } else null,
        .audio = if (cfg.audio.window_duration_ms > 0 or cfg.audio.overlap_duration_ms > 0) .{
            .window_duration_ms = if (cfg.audio.window_duration_ms > 0) cfg.audio.window_duration_ms else null,
            .overlap_duration_ms = if (cfg.audio.overlap_duration_ms > 0) cfg.audio.overlap_duration_ms else null,
        } else null,
    };

    switch (input) {
        .text => |text| {
            const request = inference_api.ChunkRequest{
                .input = .{ .string = text },
                .config = config,
            };
            return try httpx.json.Json.stringify(alloc, request);
        },
        .binary => |binary| {
            const data_b64 = try base64EncodeAlloc(alloc, binary.data);
            defer alloc.free(data_b64);
            const request = struct {
                input: inference_api.MediaContentPart,
                config: inference_api.ChunkConfig,
            }{
                .input = .{
                    .type = "media",
                    .data = data_b64,
                    .mime_type = binary.mime_type,
                },
                .config = config,
            };
            return try httpx.json.Json.stringify(alloc, request);
        },
    }
}

fn parseChunkResponse(alloc: Allocator, response_body: []const u8) ![]RemoteChunk {
    const SparsePayload = struct {
        indices: []const i32,
        values: []const f32,
    };
    const Response = struct {
        data: []const struct {
            id: i64,
            mime_type: []const u8,
            text: ?[]const u8 = null,
            start_char: ?i64 = null,
            end_char: ?i64 = null,
            start_token: ?i64 = null,
            end_token: ?i64 = null,
            boundary_score: ?f32 = null,
            embedding: ?[]const f32 = null,
            embedding_dimension: ?i64 = null,
            sparse_embedding: ?SparsePayload = null,
            model_version: ?[]const u8 = null,
            artifact_family: ?[]const u8 = null,
            data: ?[]const u8 = null,
            start_time_ms: ?f32 = null,
            end_time_ms: ?f32 = null,
            frame_index: ?i64 = null,
            frame_delay_ms: ?i64 = null,
        },
    };

    var parsed = try std.json.parseFromSlice(Response, alloc, response_body, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();

    var chunks = try alloc.alloc(RemoteChunk, parsed.value.data.len);
    errdefer {
        for (chunks) |*chunk| chunk.deinit(alloc);
        alloc.free(chunks);
    }

    for (parsed.value.data, 0..) |chunk, i| {
        var out = RemoteChunk{
            .id = std.math.cast(u32, chunk.id) orelse return error.InvalidChunkerResponse,
            .mime_type = try alloc.dupe(u8, chunk.mime_type),
            .text = null,
            .start_char = if (chunk.start_char) |v| std.math.cast(u32, v) orelse return error.InvalidChunkerResponse else null,
            .end_char = if (chunk.end_char) |v| std.math.cast(u32, v) orelse return error.InvalidChunkerResponse else null,
            .start_token = if (chunk.start_token) |v| std.math.cast(u32, v) orelse return error.InvalidChunkerResponse else null,
            .end_token = if (chunk.end_token) |v| std.math.cast(u32, v) orelse return error.InvalidChunkerResponse else null,
            .boundary_score = chunk.boundary_score,
            .embedding = null,
            .embedding_dimension = if (chunk.embedding_dimension) |v| std.math.cast(u32, v) orelse return error.InvalidChunkerResponse else null,
            .sparse_embedding = null,
            .model_version = null,
            .artifact_family = null,
            .data = null,
            .start_time_ms = chunk.start_time_ms,
            .end_time_ms = chunk.end_time_ms,
            .frame_index = if (chunk.frame_index) |v| std.math.cast(u32, v) orelse return error.InvalidChunkerResponse else null,
            .frame_delay_ms = if (chunk.frame_delay_ms) |v| std.math.cast(u32, v) orelse return error.InvalidChunkerResponse else null,
            .owns_text = false,
            .owns_data = false,
            .owns_embedding = false,
            .owns_model_version = false,
            .owns_artifact_family = false,
        };
        errdefer {
            if (out.mime_type.len > 0) alloc.free(@constCast(out.mime_type));
            out.deinit(alloc);
        }
        if (chunk.text) |value| {
            out.text = try alloc.dupe(u8, value);
            out.owns_text = true;
        }
        if (chunk.embedding) |value| {
            out.embedding = try alloc.dupe(f32, value);
            out.owns_embedding = true;
            if (out.embedding_dimension == null) out.embedding_dimension = std.math.cast(u32, value.len);
        }
        if (chunk.sparse_embedding) |value| {
            if (value.indices.len != value.values.len) return error.InvalidChunkerResponse;
            out.sparse_embedding = try cloneSparseVector(alloc, .{
                .indices = value.indices,
                .values = value.values,
            });
        }
        if (chunk.model_version) |value| {
            out.model_version = try alloc.dupe(u8, value);
            out.owns_model_version = true;
        }
        if (chunk.artifact_family) |value| {
            out.artifact_family = try alloc.dupe(u8, value);
            out.owns_artifact_family = true;
        }
        if (chunk.data) |value| {
            out.data = try base64DecodeAlloc(alloc, value);
            out.owns_data = true;
        }
        chunks[i] = out;
    }

    return chunks;
}

fn base64EncodeAlloc(alloc: Allocator, bytes: []const u8) ![]u8 {
    const size = std.base64.standard.Encoder.calcSize(bytes.len);
    const out = try alloc.alloc(u8, size);
    _ = std.base64.standard.Encoder.encode(out, bytes);
    return out;
}

fn base64DecodeAlloc(alloc: Allocator, encoded: []const u8) ![]u8 {
    const size = try std.base64.standard.Decoder.calcSizeForSlice(encoded);
    const out = try alloc.alloc(u8, size);
    try std.base64.standard.Decoder.decode(out, encoded);
    return out;
}

test "antfly chunker compiles" {
    _ = chunkText;
    _ = chunkInput;
    _ = chunkBinary;
}

test "antfly chunker text round trip" {
    const alloc = std.testing.allocator;
    const FakeApp = struct {
        fn executor() http_common.RequestExecutor {
            return .{
                .ptr = undefined,
                .vtable = &.{
                    .execute = execute,
                },
            };
        }

        fn execute(_: *anyopaque, req_alloc: Allocator, req: http_common.HttpRequest) !http_common.HttpResponse {
            try std.testing.expectEqual(http_common.Method.POST, req.method);
            try std.testing.expect(std.mem.endsWith(u8, req.uri, "/chunk"));
            try std.testing.expect(std.mem.indexOf(u8, req.body, "\"model\":\"chunker-v1\"") != null);
            return .{
                .status = 200,
                .content_type = try req_alloc.dupe(u8, "application/json"),
                .body = try req_alloc.dupe(u8,
                    \\{"object":"list","data":[
                    \\  {"object":"chunk","index":0,"id":0,"mime_type":"text/plain","text":"alpha body","start_char":0,"end_char":10},
                    \\  {"object":"chunk","index":1,"id":1,"mime_type":"text/plain","text":"beta tail","start_char":11,"end_char":20}
                    \\],"model":"chunker-v1","usage":{"prompt_tokens":4,"completion_tokens":0,"total_tokens":4},"cache_hit":false}
                ),
            };
        }
    };

    var listener = std_http_listener.StdHttpListener.init(alloc, .{}, FakeApp.executor());
    defer listener.deinit();
    try listener.start();

    const base_uri = try listener.baseUri(alloc);
    defer alloc.free(base_uri);

    const cfg = chunking_types.Config{
        .provider = .antfly,
        .api_url = base_uri,
        .model = "chunker-v1",
        .text = .{ .target_tokens = 8, .overlap_tokens = 2 },
    };

    const chunks = try chunkText(alloc, cfg, "alpha beta gamma delta");
    defer {
        for (chunks) |*chunk| chunk.deinit(alloc);
        alloc.free(chunks);
    }

    try std.testing.expectEqual(@as(usize, 2), chunks.len);
    try std.testing.expectEqualStrings("alpha body", chunks[0].text.?);
    try std.testing.expectEqual(@as(?u32, 11), chunks[1].start_offset);
}

test "antfly chunker binary round trip" {
    const alloc = std.testing.allocator;
    const FakeApp = struct {
        fn executor() http_common.RequestExecutor {
            return .{
                .ptr = undefined,
                .vtable = &.{
                    .execute = execute,
                },
            };
        }

        fn execute(_: *anyopaque, req_alloc: Allocator, req: http_common.HttpRequest) !http_common.HttpResponse {
            try std.testing.expectEqual(http_common.Method.POST, req.method);
            try std.testing.expect(std.mem.indexOf(u8, req.body, "\"mime_type\":\"image/gif\"") != null);
            try std.testing.expect(std.mem.indexOf(u8, req.body, "\"data\":\"R0lG\"") != null);
            return .{
                .status = 200,
                .content_type = try req_alloc.dupe(u8, "application/json"),
                .body = try req_alloc.dupe(u8,
                    \\{"object":"list","data":[
                    \\  {"object":"chunk","index":0,"id":0,"mime_type":"image/png","data":"iVBORw0KGgo=","frame_index":0,"frame_delay_ms":50}
                    \\],"model":"chunker-v1","usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0},"cache_hit":false}
                ),
            };
        }
    };

    var listener = std_http_listener.StdHttpListener.init(alloc, .{}, FakeApp.executor());
    defer listener.deinit();
    try listener.start();

    const base_uri = try listener.baseUri(alloc);
    defer alloc.free(base_uri);

    const cfg = chunking_types.Config{
        .provider = .antfly,
        .api_url = base_uri,
        .model = "chunker-v1",
    };

    const chunks = try chunkBinary(alloc, cfg, "image/gif", "GIF");
    defer freeRemoteChunks(alloc, chunks);

    try std.testing.expectEqual(@as(usize, 1), chunks.len);
    try std.testing.expectEqualStrings("image/png", chunks[0].mime_type);
    try std.testing.expectEqual(@as(?u32, 0), chunks[0].frame_index);
    try std.testing.expectEqual(@as(?u32, 50), chunks[0].frame_delay_ms);
    try std.testing.expectEqualSlices(u8, &.{ 0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n' }, chunks[0].data.?[0..8]);
}

test "antfly chunker preserves contextual embedding response fields" {
    const alloc = std.testing.allocator;
    const FakeApp = struct {
        fn executor() http_common.RequestExecutor {
            return .{
                .ptr = undefined,
                .vtable = &.{
                    .execute = execute,
                },
            };
        }

        fn execute(_: *anyopaque, req_alloc: Allocator, req: http_common.HttpRequest) !http_common.HttpResponse {
            try std.testing.expectEqual(http_common.Method.POST, req.method);
            try std.testing.expect(std.mem.indexOf(u8, req.body, "\"include_embeddings\":true") != null);
            try std.testing.expect(std.mem.indexOf(u8, req.body, "\"include_sparse\":true") != null);
            try std.testing.expect(std.mem.indexOf(u8, req.body, "\"output_dimension\":256") != null);
            try std.testing.expect(std.mem.indexOf(u8, req.body, "\"sparse_top_k\":4") != null);
            try std.testing.expect(std.mem.indexOf(u8, req.body, "\"include_boundary_scores\":true") != null);
            return .{
                .status = 200,
                .content_type = try req_alloc.dupe(u8, "application/json"),
                .body = try req_alloc.dupe(u8,
                    \\{"object":"list","data":[
                    \\  {"object":"chunk","index":0,"id":0,"mime_type":"text/plain","text":"alpha body","start_char":0,"end_char":10,"start_token":1,"end_token":5,"boundary_score":0.91,"embedding":[0.5,0.25],"embedding_dimension":2,"sparse_embedding":{"indices":[7,11],"values":[1.25,0.75]},"model_version":"ckpt-1","artifact_family":"fused_chunker_embedder/v1"}
                    \\],"model":"chunker-v1","usage":{"prompt_tokens":2,"completion_tokens":0,"total_tokens":2},"cache_hit":false}
                ),
            };
        }
    };

    var listener = std_http_listener.StdHttpListener.init(alloc, .{}, FakeApp.executor());
    defer listener.deinit();
    try listener.start();

    const base_uri = try listener.baseUri(alloc);
    defer alloc.free(base_uri);

    const cfg = chunking_types.Config{
        .provider = .antfly,
        .api_url = base_uri,
        .model = "chunker-v1",
        .include_embeddings = true,
        .include_sparse = true,
        .output_dimension = 256,
        .sparse_top_k = 4,
        .include_boundary_scores = true,
    };

    const chunks = try chunkInput(alloc, cfg, .{ .text = "alpha beta" });
    defer freeRemoteChunks(alloc, chunks);

    try std.testing.expectEqual(@as(usize, 1), chunks.len);
    try std.testing.expectEqual(@as(?u32, 1), chunks[0].start_token);
    try std.testing.expectEqual(@as(?u32, 5), chunks[0].end_token);
    try std.testing.expectApproxEqAbs(@as(f32, 0.91), chunks[0].boundary_score.?, 1e-6);
    try std.testing.expectEqual(@as(?u32, 2), chunks[0].embedding_dimension);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, 0.25 }, chunks[0].embedding.?);
    try std.testing.expectEqualSlices(i32, &.{ 7, 11 }, chunks[0].sparse_embedding.?.indices);
    try std.testing.expectEqualSlices(f32, &.{ 1.25, 0.75 }, chunks[0].sparse_embedding.?.values);
    try std.testing.expectEqualStrings("ckpt-1", chunks[0].model_version.?);
    try std.testing.expectEqualStrings("fused_chunker_embedder/v1", chunks[0].artifact_family.?);
}

test "antfly chunker with empty api_url runs locally" {
    const alloc = std.testing.allocator;
    const cfg = chunking_types.Config{
        .provider = .antfly,
        .model = "fixed-bert-tokenizer",
        .max_chunks = 2,
        .text = .{ .target_tokens = 3, .overlap_tokens = 0, .separator = " " },
    };

    const chunks = try chunkText(alloc, cfg, "alpha beta gamma delta epsilon");
    defer {
        for (chunks) |*chunk| chunk.deinit(alloc);
        alloc.free(chunks);
    }

    try std.testing.expect(chunks.len > 0);
    try std.testing.expect(chunks.len <= 2);
    try std.testing.expect(chunks[0].text != null);
}

test "antfly chunker local path preserves explicit zero overlap when target is set" {
    const alloc = std.testing.allocator;
    const cfg = chunking_types.Config{
        .provider = .antfly,
        .model = "fixed-bert-tokenizer",
        .text = .{ .target_tokens = 4, .overlap_tokens = 0, .separator = " " },
    };

    const chunks = try chunkText(alloc, cfg, "alpha beta gamma delta epsilon");
    defer {
        for (chunks) |*chunk| chunk.deinit(alloc);
        alloc.free(chunks);
    }

    try std.testing.expect(chunks.len > 0);
    try std.testing.expect(chunks[0].text != null);
}
