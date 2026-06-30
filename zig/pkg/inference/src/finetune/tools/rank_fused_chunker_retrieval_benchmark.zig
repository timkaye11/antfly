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

const RankingMode = enum {
    dense,
    sparse,
    hybrid,

    fn parse(raw: []const u8) !RankingMode {
        if (std.mem.eql(u8, raw, "dense")) return .dense;
        if (std.mem.eql(u8, raw, "sparse")) return .sparse;
        if (std.mem.eql(u8, raw, "hybrid")) return .hybrid;
        return error.InvalidRankingMode;
    }
};

const Options = struct {
    queries_path: []const u8,
    docs_path: []const u8,
    out_path: []const u8,
    mode: RankingMode = .hybrid,
    top_k: usize = 100,
    rrf_k: f64 = 60.0,
};

const RankConfig = struct {
    mode: RankingMode = .hybrid,
    top_k: usize = 100,
    rrf_k: f64 = 60.0,
};

const SparseVector = struct {
    indices: []const i64 = &.{},
    values: []const f64 = &.{},
};

const EmbeddingRecord = struct {
    id: []const u8,
    relevant_ids: []const []const u8 = &.{},
    dense: ?[]const f64 = null,
    sparse: ?SparseVector = null,
};

const SortContext = struct {
    scores: []const f64,
    docs: []const EmbeddingRecord,

    fn lessThan(ctx: SortContext, a: usize, b: usize) bool {
        const a_score = ctx.scores[a];
        const b_score = ctx.scores[b];
        if (a_score > b_score) return true;
        if (a_score < b_score) return false;
        return std.mem.lessThan(u8, ctx.docs[a].id, ctx.docs[b].id);
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const opts = try parseOptions(&args) orelse return;
    const query_bytes = try compat.cwd().readFileAlloc(compat.io(), opts.queries_path, allocator, .limited(512 * 1024 * 1024));
    defer allocator.free(query_bytes);
    const doc_bytes = try compat.cwd().readFileAlloc(compat.io(), opts.docs_path, allocator, .limited(2 * 1024 * 1024 * 1024));
    defer allocator.free(doc_bytes);

    const out = try rankRetrievalBenchmarkJsonl(allocator, query_bytes, doc_bytes, .{
        .mode = opts.mode,
        .top_k = opts.top_k,
        .rrf_k = opts.rrf_k,
    });
    defer allocator.free(out);
    try compat.cwd().writeFile(compat.io(), .{ .sub_path = opts.out_path, .data = out });

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(compat.io(), &stdout_buf);
    try std.json.Stringify.value(.{
        .status = "written",
        .queries = opts.queries_path,
        .docs = opts.docs_path,
        .out = opts.out_path,
        .mode = @tagName(opts.mode),
        .top_k = opts.top_k,
    }, .{ .whitespace = .indent_2 }, &stdout.interface);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}

fn parseOptions(args: *std.process.Args.Iterator) !?Options {
    var queries_path: ?[]const u8 = null;
    var docs_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var mode: RankingMode = .hybrid;
    var top_k: usize = 100;
    var rrf_k: f64 = 60.0;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--queries")) {
            queries_path = args.next() orelse return error.MissingQueriesPath;
        } else if (std.mem.eql(u8, arg, "--docs")) {
            docs_path = args.next() orelse return error.MissingDocsPath;
        } else if (std.mem.eql(u8, arg, "--out")) {
            out_path = args.next() orelse return error.MissingOutPath;
        } else if (std.mem.eql(u8, arg, "--mode")) {
            mode = try RankingMode.parse(args.next() orelse return error.MissingMode);
        } else if (std.mem.eql(u8, arg, "--top-k")) {
            top_k = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingTopK, 10);
        } else if (std.mem.eql(u8, arg, "--rrf-k")) {
            rrf_k = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingRrfK);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            usage();
            return null;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }
    if (queries_path == null or docs_path == null or out_path == null) {
        usage();
        if (queries_path == null) return error.MissingQueriesPath;
        if (docs_path == null) return error.MissingDocsPath;
        return error.MissingOutPath;
    }
    if (top_k == 0) return error.InvalidTopK;
    if (!std.math.isFinite(rrf_k) or rrf_k <= 0) return error.InvalidRrfK;
    return .{
        .queries_path = queries_path.?,
        .docs_path = docs_path.?,
        .out_path = out_path.?,
        .mode = mode,
        .top_k = top_k,
        .rrf_k = rrf_k,
    };
}

pub fn rankRetrievalBenchmarkJsonl(
    allocator: std.mem.Allocator,
    queries_jsonl: []const u8,
    docs_jsonl: []const u8,
    config: RankConfig,
) ![]u8 {
    if (config.top_k == 0) return error.InvalidTopK;
    if (!std.math.isFinite(config.rrf_k) or config.rrf_k <= 0) return error.InvalidRrfK;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    const queries = try parseEmbeddingJsonl(arena_alloc, queries_jsonl, true);
    const docs = try parseEmbeddingJsonl(arena_alloc, docs_jsonl, false);
    if (queries.len == 0) return error.NoQueries;
    if (docs.len == 0) return error.NoDocs;

    var buffer: std.Io.Writer.Allocating = .init(allocator);
    errdefer buffer.deinit();
    for (queries) |query| {
        const ranked = try rankDocsForQuery(allocator, query, docs, config);
        defer allocator.free(ranked.scores);
        defer allocator.free(ranked.ids);
        try writeRunRecord(&buffer.writer, query.id, query.relevant_ids, ranked.ids, ranked.scores);
        try buffer.writer.writeByte('\n');
    }
    return buffer.toOwnedSlice();
}

const RankedRun = struct {
    ids: []const []const u8,
    scores: []const f64,
};

fn rankDocsForQuery(
    allocator: std.mem.Allocator,
    query: EmbeddingRecord,
    docs: []const EmbeddingRecord,
    config: RankConfig,
) !RankedRun {
    const top_k = @min(config.top_k, docs.len);
    const scores = try allocator.alloc(f64, docs.len);
    errdefer allocator.free(scores);
    const order = try allocator.alloc(usize, docs.len);
    defer allocator.free(order);
    for (0..docs.len) |i| order[i] = i;

    switch (config.mode) {
        .dense => {
            try scoreDense(query, docs, scores);
        },
        .sparse => {
            try scoreSparse(query, docs, scores);
        },
        .hybrid => {
            try scoreHybrid(allocator, query, docs, scores, config.rrf_k);
        },
    }

    std.sort.pdq(usize, order, SortContext{ .scores = scores, .docs = docs }, SortContext.lessThan);
    const ids = try allocator.alloc([]const u8, top_k);
    errdefer allocator.free(ids);
    const out_scores = try allocator.alloc(f64, top_k);
    errdefer allocator.free(out_scores);
    for (0..top_k) |rank| {
        const doc_idx = order[rank];
        ids[rank] = docs[doc_idx].id;
        out_scores[rank] = scores[doc_idx];
    }
    allocator.free(scores);
    return .{ .ids = ids, .scores = out_scores };
}

fn scoreDense(query: EmbeddingRecord, docs: []const EmbeddingRecord, scores: []f64) !void {
    const q = query.dense orelse return error.MissingQueryDenseEmbedding;
    for (docs, 0..) |doc, i| {
        const d = doc.dense orelse return error.MissingDocDenseEmbedding;
        scores[i] = try cosineSimilarity(q, d);
    }
}

fn scoreSparse(query: EmbeddingRecord, docs: []const EmbeddingRecord, scores: []f64) !void {
    const q = query.sparse orelse return error.MissingQuerySparseEmbedding;
    for (docs, 0..) |doc, i| {
        const d = doc.sparse orelse return error.MissingDocSparseEmbedding;
        scores[i] = sparseDot(q, d);
    }
}

fn scoreHybrid(
    allocator: std.mem.Allocator,
    query: EmbeddingRecord,
    docs: []const EmbeddingRecord,
    scores: []f64,
    rrf_k: f64,
) !void {
    const dense_scores = try allocator.alloc(f64, docs.len);
    defer allocator.free(dense_scores);
    const sparse_scores = try allocator.alloc(f64, docs.len);
    defer allocator.free(sparse_scores);
    try scoreDense(query, docs, dense_scores);
    try scoreSparse(query, docs, sparse_scores);

    @memset(scores, 0.0);
    try addRrfScores(allocator, docs, dense_scores, scores, rrf_k);
    try addRrfScores(allocator, docs, sparse_scores, scores, rrf_k);
}

fn addRrfScores(
    allocator: std.mem.Allocator,
    docs: []const EmbeddingRecord,
    modality_scores: []const f64,
    out: []f64,
    rrf_k: f64,
) !void {
    const order = try allocator.alloc(usize, docs.len);
    defer allocator.free(order);
    for (0..docs.len) |i| order[i] = i;
    std.sort.pdq(usize, order, SortContext{ .scores = modality_scores, .docs = docs }, SortContext.lessThan);
    for (order, 0..) |doc_idx, zero_based_rank| {
        out[doc_idx] += 1.0 / (rrf_k + @as(f64, @floatFromInt(zero_based_rank + 1)));
    }
}

fn parseEmbeddingJsonl(
    arena: std.mem.Allocator,
    jsonl: []const u8,
    is_query: bool,
) ![]const EmbeddingRecord {
    var records = std.ArrayListUnmanaged(EmbeddingRecord).empty;
    errdefer records.deinit(arena);

    var lines = std.mem.tokenizeScalar(u8, jsonl, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r\n");
        if (line.len == 0) continue;
        const root = try std.json.parseFromSliceLeaky(std.json.Value, arena, line, .{ .ignore_unknown_fields = true });
        if (root != .object) return error.ExpectedJsonObject;
        const id = readFirstString(root.object, if (is_query)
            &.{ "query_id", "id" }
        else
            &.{ "id", "doc_id", "chunk_id" }) orelse return error.MissingId;
        try records.append(arena, .{
            .id = id,
            .relevant_ids = if (is_query) try readStringArrayAny(arena, root.object, &.{ "relevant_ids", "relevant" }) orelse &.{} else &.{},
            .dense = try readF64ArrayAny(arena, root.object, &.{ "dense_embedding", "embedding" }),
            .sparse = try readSparseAny(arena, root.object, &.{ "sparse_embedding", "splade_embedding", "sparse" }),
        });
    }
    return records.toOwnedSlice(arena);
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
            var out = try arena.alloc([]const u8, value.array.items.len);
            for (value.array.items, 0..) |item, i| {
                if (item != .string) return error.ExpectedJsonString;
                out[i] = item.string;
            }
            return out;
        }
    }
    return null;
}

fn readF64ArrayAny(
    arena: std.mem.Allocator,
    object: std.json.ObjectMap,
    keys: []const []const u8,
) !?[]const f64 {
    for (keys) |key| {
        if (object.get(key)) |value| {
            if (value != .array) return error.ExpectedJsonArray;
            var out = try arena.alloc(f64, value.array.items.len);
            for (value.array.items, 0..) |item, i| {
                out[i] = try jsonF64(item);
            }
            return out;
        }
    }
    return null;
}

fn readSparseAny(
    arena: std.mem.Allocator,
    object: std.json.ObjectMap,
    keys: []const []const u8,
) !?SparseVector {
    for (keys) |key| {
        if (object.get(key)) |value| {
            if (value != .object) return error.ExpectedJsonObject;
            const indices_value = value.object.get("indices") orelse return error.MissingSparseIndices;
            const values_value = value.object.get("values") orelse return error.MissingSparseValues;
            if (indices_value != .array or values_value != .array) return error.ExpectedJsonArray;
            if (indices_value.array.items.len != values_value.array.items.len) return error.SparseLengthMismatch;
            var indices = try arena.alloc(i64, indices_value.array.items.len);
            var values = try arena.alloc(f64, values_value.array.items.len);
            for (indices_value.array.items, 0..) |item, i| indices[i] = try jsonI64(item);
            for (values_value.array.items, 0..) |item, i| values[i] = try jsonF64(item);
            return .{ .indices = indices, .values = values };
        }
    }
    return null;
}

fn cosineSimilarity(a: []const f64, b: []const f64) !f64 {
    if (a.len != b.len) return error.DenseDimensionMismatch;
    var dot: f64 = 0.0;
    var a_norm_sq: f64 = 0.0;
    var b_norm_sq: f64 = 0.0;
    for (a, b) |av, bv| {
        dot += av * bv;
        a_norm_sq += av * av;
        b_norm_sq += bv * bv;
    }
    if (a_norm_sq <= 0.0 or b_norm_sq <= 0.0) return 0.0;
    return dot / (std.math.sqrt(a_norm_sq) * std.math.sqrt(b_norm_sq));
}

fn sparseDot(a: SparseVector, b: SparseVector) f64 {
    var i: usize = 0;
    var j: usize = 0;
    var out: f64 = 0.0;
    while (i < a.indices.len and j < b.indices.len) {
        if (a.indices[i] == b.indices[j]) {
            out += a.values[i] * b.values[j];
            i += 1;
            j += 1;
        } else if (a.indices[i] < b.indices[j]) {
            i += 1;
        } else {
            j += 1;
        }
    }
    return out;
}

fn writeRunRecord(
    writer: *std.Io.Writer,
    query_id: []const u8,
    relevant_ids: []const []const u8,
    ranked_ids: []const []const u8,
    scores: []const f64,
) !void {
    try writer.writeAll("{\"query_id\":");
    try std.json.Stringify.value(query_id, .{}, writer);
    try writer.writeAll(",\"relevant_ids\":");
    try writeStringArray(writer, relevant_ids);
    try writer.writeAll(",\"ranked_ids\":");
    try writeStringArray(writer, ranked_ids);
    try writer.writeAll(",\"results\":[");
    for (ranked_ids, scores, 0..) |id, score, i| {
        if (i > 0) try writer.writeByte(',');
        try writer.writeAll("{\"id\":");
        try std.json.Stringify.value(id, .{}, writer);
        try writer.writeAll(",\"score\":");
        try std.json.Stringify.value(score, .{}, writer);
        try writer.writeByte('}');
    }
    try writer.writeAll("]}");
}

fn writeStringArray(writer: *std.Io.Writer, ids: []const []const u8) !void {
    try writer.writeByte('[');
    for (ids, 0..) |id, i| {
        if (i > 0) try writer.writeByte(',');
        try std.json.Stringify.value(id, .{}, writer);
    }
    try writer.writeByte(']');
}

fn jsonF64(value: std.json.Value) !f64 {
    return switch (value) {
        .integer => |i| @floatFromInt(i),
        .float => |f| f,
        else => error.ExpectedJsonNumber,
    };
}

fn jsonI64(value: std.json.Value) !i64 {
    return switch (value) {
        .integer => |i| i,
        else => error.ExpectedJsonInteger,
    };
}

fn usage() void {
    std.debug.print(
        \\usage: rank-fused-chunker-retrieval-benchmark --queries <jsonl> --docs <jsonl> --out <jsonl> [options]
        \\
        \\  --mode <dense|sparse|hybrid>   Ranking mode; default hybrid
        \\  --top-k <n>                    Ranked ids to emit per query; default 100
        \\  --rrf-k <n>                    Reciprocal-rank-fusion constant; default 60
        \\
        \\Query JSONL records must include id/query_id and relevant_ids/relevant.
        \\Doc JSONL records must include id/doc_id/chunk_id. Dense vectors use
        \\embedding/dense_embedding arrays; SPLADE vectors use sparse_embedding
        \\objects with indices and values arrays.
        \\
    , .{});
}

test "dense ranking uses cosine similarity" {
    const queries =
        \\{"query_id":"q1","relevant_ids":["d2"],"embedding":[0,1]}
        \\
    ;
    const docs =
        \\{"id":"d1","embedding":[1,0]}
        \\{"id":"d2","embedding":[0,2]}
        \\
    ;
    const out = try rankRetrievalBenchmarkJsonl(std.testing.allocator, queries, docs, .{ .mode = .dense, .top_k = 2 });
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"ranked_ids\":[\"d2\",\"d1\"]") != null);
}

test "sparse ranking uses splade dot product" {
    const queries =
        \\{"query_id":"q1","relevant":["d3"],"sparse_embedding":{"indices":[3,9],"values":[1.0,2.0]}}
        \\
    ;
    const docs =
        \\{"id":"d1","sparse_embedding":{"indices":[1,2],"values":[10,10]}}
        \\{"id":"d3","sparse_embedding":{"indices":[9],"values":[3]}}
        \\
    ;
    const out = try rankRetrievalBenchmarkJsonl(std.testing.allocator, queries, docs, .{ .mode = .sparse, .top_k = 2 });
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"ranked_ids\":[\"d3\",\"d1\"]") != null);
}

test "hybrid reciprocal rank fusion can rescue sparse-positive document" {
    const queries =
        \\{"query_id":"q1","relevant_ids":["d_sparse"],"embedding":[1,0],"sparse_embedding":{"indices":[42],"values":[1]}}
        \\
    ;
    const docs =
        \\{"id":"d_dense","embedding":[1,0],"sparse_embedding":{"indices":[7],"values":[1]}}
        \\{"id":"d_sparse","embedding":[0,1],"sparse_embedding":{"indices":[42],"values":[5]}}
        \\{"id":"d_none","embedding":[0,-1],"sparse_embedding":{"indices":[9],"values":[1]}}
        \\
    ;
    const out = try rankRetrievalBenchmarkJsonl(std.testing.allocator, queries, docs, .{ .mode = .hybrid, .top_k = 3, .rrf_k = 1.0 });
    defer std.testing.allocator.free(out);
    try std.testing.expect(std.mem.indexOf(u8, out, "\"ranked_ids\":[\"d_dense\",\"d_sparse\",\"d_none\"]") != null);
}
