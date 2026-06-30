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

const std = @import("std");
const compat = @import("termite_io_compat");

const Kind = enum {
    docs,
    queries,

    fn parse(raw: []const u8) !Kind {
        if (std.mem.eql(u8, raw, "docs") or std.mem.eql(u8, raw, "documents")) return .docs;
        if (std.mem.eql(u8, raw, "queries")) return .queries;
        return error.InvalidKind;
    }
};

const Options = struct {
    input_path: []const u8,
    out_path: []const u8,
    kind: Kind,
    require_dense: bool = true,
    require_sparse: bool = true,
    output_dimension: ?usize = null,
};

const MaterializeConfig = struct {
    kind: Kind,
    require_dense: bool = true,
    require_sparse: bool = true,
    output_dimension: ?usize = null,
};

const Stats = struct {
    input_records: usize = 0,
    output_records: usize = 0,
    dense_dimension: ?usize = null,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const opts = try parseOptions(&args) orelse return;
    const input_bytes = try compat.cwd().readFileAlloc(compat.io(), opts.input_path, allocator, .limited(2 * 1024 * 1024 * 1024));
    defer allocator.free(input_bytes);

    const result = try materializeRetrievalEmbeddingsJsonl(allocator, input_bytes, .{
        .kind = opts.kind,
        .require_dense = opts.require_dense,
        .require_sparse = opts.require_sparse,
        .output_dimension = opts.output_dimension,
    });
    defer allocator.free(result.jsonl);
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = opts.out_path, .data = result.jsonl });

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(compat.io(), &stdout_buf);
    try std.json.Stringify.value(.{
        .status = "written",
        .input = opts.input_path,
        .out = opts.out_path,
        .kind = @tagName(opts.kind),
        .input_records = result.stats.input_records,
        .output_records = result.stats.output_records,
        .dense_dimension = result.stats.dense_dimension,
        .require_dense = opts.require_dense,
        .require_sparse = opts.require_sparse,
    }, .{ .whitespace = .indent_2 }, &stdout.interface);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}

fn parseOptions(args: *std.process.Args.Iterator) !?Options {
    var input_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var kind: ?Kind = null;
    var require_dense = true;
    var require_sparse = true;
    var output_dimension: ?usize = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--input")) {
            input_path = args.next() orelse return error.MissingInputPath;
        } else if (std.mem.eql(u8, arg, "--out")) {
            out_path = args.next() orelse return error.MissingOutPath;
        } else if (std.mem.eql(u8, arg, "--kind")) {
            kind = try Kind.parse(args.next() orelse return error.MissingKind);
        } else if (std.mem.eql(u8, arg, "--allow-missing-dense")) {
            require_dense = false;
        } else if (std.mem.eql(u8, arg, "--allow-missing-sparse")) {
            require_sparse = false;
        } else if (std.mem.eql(u8, arg, "--output-dimension")) {
            output_dimension = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingOutputDimension, 10);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            usage();
            return null;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }

    if (input_path == null or out_path == null or kind == null) {
        usage();
        if (input_path == null) return error.MissingInputPath;
        if (out_path == null) return error.MissingOutPath;
        return error.MissingKind;
    }
    if (output_dimension != null and output_dimension.? == 0) return error.InvalidOutputDimension;
    return .{
        .input_path = input_path.?,
        .out_path = out_path.?,
        .kind = kind.?,
        .require_dense = require_dense,
        .require_sparse = require_sparse,
        .output_dimension = output_dimension,
    };
}

const MaterializeResult = struct {
    jsonl: []u8,
    stats: Stats,
};

pub fn materializeRetrievalEmbeddingsJsonl(
    allocator: std.mem.Allocator,
    input_jsonl: []const u8,
    config: MaterializeConfig,
) !MaterializeResult {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    var buffer: std.Io.Writer.Allocating = .init(allocator);
    errdefer buffer.deinit();
    var stats = Stats{ .dense_dimension = config.output_dimension };

    var lines = std.mem.tokenizeScalar(u8, input_jsonl, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r\n");
        if (line.len == 0) continue;
        stats.input_records += 1;
        const root = try std.json.parseFromSliceLeaky(std.json.Value, arena_alloc, line, .{ .ignore_unknown_fields = true });
        const object = try requireObject(root);
        switch (config.kind) {
            .docs => try materializeDocRecord(allocator, arena_alloc, &buffer.writer, object, config, &stats),
            .queries => try materializeQueryRecord(arena_alloc, &buffer.writer, object, config, &stats),
        }
    }

    if (stats.input_records == 0) return error.NoInputRecords;
    if (stats.output_records == 0) return error.NoOutputRecords;
    return .{ .jsonl = try buffer.toOwnedSlice(), .stats = stats };
}

fn materializeDocRecord(
    allocator: std.mem.Allocator,
    arena: std.mem.Allocator,
    writer: *std.Io.Writer,
    object: std.json.ObjectMap,
    config: MaterializeConfig,
    stats: *Stats,
) !void {
    const document_id = readFirstString(object, &.{ "document_id", "doc_id", "source_id", "id" }) orelse return error.MissingDocumentId;
    const chunks = try chunkArrayFromRecord(object);
    if (chunks) |chunk_values| {
        for (chunk_values, 0..) |chunk_value, ordinal| {
            const chunk = try requireObject(chunk_value);
            const chunk_id = try allocChunkId(allocator, document_id, chunk, ordinal);
            defer allocator.free(chunk_id);
            try writeDocEmbeddingRecord(arena, writer, document_id, chunk_id, chunk, ordinal, config, stats);
            try writer.writeByte('\n');
            stats.output_records += 1;
        }
    } else {
        const chunk_id = try allocChunkId(allocator, document_id, object, 0);
        defer allocator.free(chunk_id);
        try writeDocEmbeddingRecord(arena, writer, document_id, chunk_id, object, 0, config, stats);
        try writer.writeByte('\n');
        stats.output_records += 1;
    }
}

fn materializeQueryRecord(
    arena: std.mem.Allocator,
    writer: *std.Io.Writer,
    object: std.json.ObjectMap,
    config: MaterializeConfig,
    stats: *Stats,
) !void {
    const query_id = readFirstString(object, &.{ "query_id", "id" }) orelse return error.MissingQueryId;
    const relevant_ids = try readStringArrayAny(arena, object, &.{ "relevant_ids", "relevant" }) orelse return error.MissingRelevantIds;
    if (relevant_ids.len == 0) return error.EmptyRelevantIds;

    const embedding_source = if (hasEmbeddingFields(object))
        object
    else blk: {
        const chunks = try chunkArrayFromRecord(object) orelse return error.MissingChunkData;
        if (chunks.len == 0) return error.MissingChunkData;
        break :blk try requireObject(chunks[0]);
    };
    try writeQueryEmbeddingRecord(writer, query_id, relevant_ids, embedding_source, config, stats);
    try writer.writeByte('\n');
    stats.output_records += 1;
}

fn writeDocEmbeddingRecord(
    arena: std.mem.Allocator,
    writer: *std.Io.Writer,
    document_id: []const u8,
    chunk_id: []const u8,
    chunk: std.json.ObjectMap,
    ordinal: usize,
    config: MaterializeConfig,
    stats: *Stats,
) !void {
    try writer.writeAll("{\"id\":");
    try std.json.Stringify.value(chunk_id, .{}, writer);
    try writer.writeAll(",\"document_id\":");
    try std.json.Stringify.value(document_id, .{}, writer);
    try writer.writeAll(",\"chunk_index\":");
    try std.json.Stringify.value(ordinal, .{}, writer);
    if (jsonString(chunk.get("text"))) |text| {
        try writer.writeAll(",\"text\":");
        try std.json.Stringify.value(text, .{}, writer);
    }
    try writeOptionalIntField(writer, "start_char", chunk.get("start_char"));
    try writeOptionalIntField(writer, "end_char", chunk.get("end_char"));
    try writeOptionalIntField(writer, "start_token", chunk.get("start_token"));
    try writeOptionalIntField(writer, "end_token", chunk.get("end_token"));
    try writeOptionalStringField(writer, "model_version", chunk.get("model_version"));
    try writeOptionalStringField(writer, "artifact_family", chunk.get("artifact_family"));
    try writeEmbeddings(arena, writer, chunk, config, stats);
    try writer.writeByte('}');
}

fn writeQueryEmbeddingRecord(
    writer: *std.Io.Writer,
    query_id: []const u8,
    relevant_ids: []const []const u8,
    source: std.json.ObjectMap,
    config: MaterializeConfig,
    stats: *Stats,
) !void {
    try writer.writeAll("{\"query_id\":");
    try std.json.Stringify.value(query_id, .{}, writer);
    try writer.writeAll(",\"relevant_ids\":");
    try writeStringArray(writer, relevant_ids);
    try writeEmbeddings(std.heap.page_allocator, writer, source, config, stats);
    try writer.writeByte('}');
}

fn writeEmbeddings(
    arena: std.mem.Allocator,
    writer: *std.Io.Writer,
    object: std.json.ObjectMap,
    config: MaterializeConfig,
    stats: *Stats,
) !void {
    if (denseValue(object)) |dense| {
        const dim = try validateDenseVector(dense, object);
        if (stats.dense_dimension) |expected| {
            if (dim != expected) return error.DenseDimensionMismatch;
        } else {
            stats.dense_dimension = dim;
        }
        try writer.writeAll(",\"dense_embedding\":");
        try std.json.Stringify.value(dense, .{}, writer);
    } else if (config.require_dense) {
        return error.MissingDenseEmbedding;
    }

    if (sparseValue(object)) |sparse| {
        try validateSparseVector(sparse);
        try writer.writeAll(",\"sparse_embedding\":");
        try std.json.Stringify.value(sparse, .{}, writer);
    } else if (config.require_sparse) {
        _ = arena;
        return error.MissingSparseEmbedding;
    }
}

fn chunkArrayFromRecord(object: std.json.ObjectMap) !?[]const std.json.Value {
    if (object.get("data")) |data| {
        if (data == .array) return data.array.items;
        if (data == .object) {
            if (data.object.get("data")) |nested| {
                if (nested == .array) return nested.array.items;
            }
            if (data.object.get("chunks")) |nested| {
                if (nested == .array) return nested.array.items;
            }
        }
    }
    if (object.get("chunks")) |chunks| {
        if (chunks == .array) return chunks.array.items;
    }
    if (object.get("response")) |response| {
        const response_object = try requireObject(response);
        if (response_object.get("data")) |data| {
            if (data == .array) return data.array.items;
        }
        if (response_object.get("chunks")) |chunks| {
            if (chunks == .array) return chunks.array.items;
        }
    }
    if (object.get("chunk_response")) |response| {
        const response_object = try requireObject(response);
        if (response_object.get("data")) |data| {
            if (data == .array) return data.array.items;
        }
        if (response_object.get("chunks")) |chunks| {
            if (chunks == .array) return chunks.array.items;
        }
    }
    return null;
}

fn allocChunkId(
    allocator: std.mem.Allocator,
    document_id: []const u8,
    chunk: std.json.ObjectMap,
    ordinal: usize,
) ![]u8 {
    if (readFirstString(chunk, &.{ "chunk_id", "id" })) |id| {
        if (std.mem.indexOfScalar(u8, id, ':') != null or std.mem.indexOfScalar(u8, id, '#') != null) {
            return allocator.dupe(u8, id);
        }
        return std.fmt.allocPrint(allocator, "{s}#chunk:{s}", .{ document_id, id });
    }
    if (jsonI64(chunk.get("id"))) |id| {
        return std.fmt.allocPrint(allocator, "{s}#chunk:{d}", .{ document_id, id });
    }
    return std.fmt.allocPrint(allocator, "{s}#chunk:{d}", .{ document_id, ordinal });
}

fn hasEmbeddingFields(object: std.json.ObjectMap) bool {
    return denseValue(object) != null or sparseValue(object) != null;
}

fn denseValue(object: std.json.ObjectMap) ?std.json.Value {
    if (object.get("dense_embedding")) |value| return value;
    if (object.get("embedding")) |value| return value;
    return null;
}

fn sparseValue(object: std.json.ObjectMap) ?std.json.Value {
    if (object.get("sparse_embedding")) |value| return value;
    if (object.get("splade_embedding")) |value| return value;
    if (object.get("sparse")) |value| return value;
    return null;
}

fn validateDenseVector(value: std.json.Value, object: std.json.ObjectMap) !usize {
    if (value != .array) return error.ExpectedDenseArray;
    if (value.array.items.len == 0) return error.EmptyDenseEmbedding;
    for (value.array.items) |item| {
        const v = try jsonF64(item);
        if (!std.math.isFinite(v)) return error.InvalidDenseEmbedding;
    }
    if (jsonUsize(object.get("embedding_dimension"))) |declared| {
        if (declared != value.array.items.len) return error.DenseDimensionMismatch;
    }
    return value.array.items.len;
}

fn validateSparseVector(value: std.json.Value) !void {
    const object = try requireObject(value);
    const indices = object.get("indices") orelse return error.MissingSparseIndices;
    const values = object.get("values") orelse return error.MissingSparseValues;
    if (indices != .array or values != .array) return error.ExpectedSparseArrays;
    if (indices.array.items.len != values.array.items.len) return error.SparseLengthMismatch;
    if (indices.array.items.len == 0) return error.EmptySparseEmbedding;
    var previous: ?i64 = null;
    for (indices.array.items, values.array.items) |idx_value, val_value| {
        const idx = try jsonI64Required(idx_value);
        if (idx < 0 or idx > std.math.maxInt(i32)) return error.InvalidSparseIndex;
        if (previous) |prev| {
            if (idx <= prev) return error.SparseIndicesNotStrictlyAscending;
        }
        previous = idx;
        const weight = try jsonF64(val_value);
        if (!std.math.isFinite(weight) or weight <= 0.0) return error.InvalidSparseWeight;
    }
}

fn readFirstString(object: std.json.ObjectMap, keys: []const []const u8) ?[]const u8 {
    for (keys) |key| {
        if (object.get(key)) |value| {
            if (value == .string) return value.string;
        }
    }
    return null;
}

fn readStringArrayAny(
    arena: std.mem.Allocator,
    object: std.json.ObjectMap,
    keys: []const []const u8,
) !?[]const []const u8 {
    for (keys) |key| {
        if (object.get(key)) |value| {
            if (value != .array) return error.ExpectedJsonArray;
            const out = try arena.alloc([]const u8, value.array.items.len);
            for (value.array.items, 0..) |item, i| {
                if (item != .string) return error.ExpectedJsonString;
                out[i] = item.string;
            }
            return out;
        }
    }
    return null;
}

fn writeStringArray(writer: *std.Io.Writer, ids: []const []const u8) !void {
    try writer.writeByte('[');
    for (ids, 0..) |id, i| {
        if (i > 0) try writer.writeByte(',');
        try std.json.Stringify.value(id, .{}, writer);
    }
    try writer.writeByte(']');
}

fn writeOptionalIntField(writer: *std.Io.Writer, name: []const u8, value: ?std.json.Value) !void {
    const v = value orelse return;
    if (jsonI64(v)) |int| {
        try writer.writeByte(',');
        try std.json.Stringify.value(name, .{}, writer);
        try writer.writeByte(':');
        try std.json.Stringify.value(int, .{}, writer);
    }
}

fn writeOptionalStringField(writer: *std.Io.Writer, name: []const u8, value: ?std.json.Value) !void {
    if (jsonString(value)) |string| {
        try writer.writeByte(',');
        try std.json.Stringify.value(name, .{}, writer);
        try writer.writeByte(':');
        try std.json.Stringify.value(string, .{}, writer);
    }
}

fn requireObject(value: std.json.Value) !std.json.ObjectMap {
    return switch (value) {
        .object => |object| object,
        else => error.ExpectedJsonObject,
    };
}

fn jsonString(value: ?std.json.Value) ?[]const u8 {
    const v = value orelse return null;
    return switch (v) {
        .string => |s| s,
        .null => null,
        else => null,
    };
}

fn jsonI64(value: ?std.json.Value) ?i64 {
    const v = value orelse return null;
    return switch (v) {
        .integer => |i| i,
        else => null,
    };
}

fn jsonI64Required(value: std.json.Value) !i64 {
    return switch (value) {
        .integer => |i| i,
        else => error.ExpectedJsonInteger,
    };
}

fn jsonUsize(value: ?std.json.Value) ?usize {
    const v = value orelse return null;
    return switch (v) {
        .integer => |i| if (i >= 0) @intCast(i) else null,
        else => null,
    };
}

fn jsonF64(value: std.json.Value) !f64 {
    return switch (value) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.ExpectedJsonNumber,
    };
}

fn usage() void {
    std.debug.print(
        \\usage: materialize-fused-chunker-retrieval-embeddings --input <jsonl> --out <jsonl> --kind <docs|queries> [options]
        \\
        \\  --output-dimension <n>     Require dense embeddings to have this dimension
        \\  --allow-missing-dense      Permit records without dense_embedding/embedding
        \\  --allow-missing-sparse     Permit records without sparse_embedding/splade_embedding
        \\
        \\Input JSONL accepts raw /chunk responses, wrappers such as
        \\document_id/response wrappers, or direct embedding
        \\records. The default mode is strict: dense and SPLADE vectors are both
        \\required, dense dimensions must be consistent, and sparse indices must
        \\be strictly ascending.
        \\
    , .{});
}

test "materializes document chunk responses into ranker doc embeddings" {
    const input =
        \\{"document_id":"docA","response":{"data":[{"id":0,"text":"alpha","start_char":0,"end_char":5,"embedding":[1,0],"embedding_dimension":2,"sparse_embedding":{"indices":[3,9],"values":[1.0,2.0]},"model_version":"ckpt","artifact_family":"fused_chunker_embedder/v1alpha1"},{"id":1,"text":"beta","start_char":6,"end_char":10,"dense_embedding":[0,1],"sparse_embedding":{"indices":[4],"values":[3.0]}}]}}
        \\
    ;
    const result = try materializeRetrievalEmbeddingsJsonl(std.testing.allocator, input, .{ .kind = .docs, .output_dimension = 2 });
    defer std.testing.allocator.free(result.jsonl);
    try std.testing.expectEqual(@as(usize, 1), result.stats.input_records);
    try std.testing.expectEqual(@as(usize, 2), result.stats.output_records);
    try std.testing.expect(std.mem.indexOf(u8, result.jsonl, "\"id\":\"docA#chunk:0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.jsonl, "\"dense_embedding\":[1,0]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.jsonl, "\"sparse_embedding\":{\"indices\":[3,9],\"values\":[") != null);
}

test "materializes query response using first chunk embedding" {
    const input =
        \\{"query_id":"q1","relevant_ids":["docA#chunk:0"],"data":[{"id":0,"embedding":[0.5,0.5],"sparse_embedding":{"indices":[8],"values":[2.0]}}]}
        \\
    ;
    const result = try materializeRetrievalEmbeddingsJsonl(std.testing.allocator, input, .{ .kind = .queries, .output_dimension = 2 });
    defer std.testing.allocator.free(result.jsonl);
    try std.testing.expectEqual(@as(usize, 1), result.stats.output_records);
    try std.testing.expect(std.mem.indexOf(u8, result.jsonl, "\"query_id\":\"q1\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.jsonl, "\"relevant_ids\":[\"docA#chunk:0\"]") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.jsonl, "\"dense_embedding\":[0.5,0.5]") != null);
}

test "strict mode rejects missing sparse embeddings" {
    const input =
        \\{"document_id":"docA","data":[{"id":0,"embedding":[1,0]}]}
        \\
    ;
    try std.testing.expectError(error.MissingSparseEmbedding, materializeRetrievalEmbeddingsJsonl(std.testing.allocator, input, .{ .kind = .docs }));
}

test "strict mode rejects unsorted sparse indices" {
    const input =
        \\{"query_id":"q1","relevant_ids":["d1"],"embedding":[1,0],"sparse_embedding":{"indices":[9,3],"values":[1.0,1.0]}}
        \\
    ;
    try std.testing.expectError(error.SparseIndicesNotStrictlyAscending, materializeRetrievalEmbeddingsJsonl(std.testing.allocator, input, .{ .kind = .queries }));
}
