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

const default_max_total_tokens: usize = 1_000_000;
const default_max_record_tokens: usize = 384;

const Options = struct {
    input_path: []const u8,
    out_path: []const u8,
    dataset_name: []const u8,
    max_total_tokens: usize = default_max_total_tokens,
    max_record_tokens: usize = default_max_record_tokens,
};

pub const PrepareConfig = struct {
    dataset_name: []const u8,
    max_total_tokens: usize = default_max_total_tokens,
    max_record_tokens: usize = default_max_record_tokens,
};

pub const PrepareStats = struct {
    dataset_name: []const u8,
    input_records: usize = 0,
    output_records: usize = 0,
    paragraphs: usize = 0,
    chunks: usize = 0,
    boundary_targets: usize = 0,
    source_tokens: usize = 0,
    materialized_tokens: usize = 0,
    skipped_single_chunk_records: usize = 0,
};

pub const PreparedOutput = struct {
    jsonl: []u8,
    stats: PrepareStats,

    pub fn deinit(self: *PreparedOutput, allocator: std.mem.Allocator) void {
        allocator.free(self.jsonl);
        self.* = undefined;
    }
};

const OutputChunk = struct {
    start_char: u32,
    end_char: u32,
};

const OutputRecord = struct {
    text: []const u8,
    chunks: []const OutputChunk,
    dataset: []const u8,
    source_format: []const u8,
    benchmark_include_final_boundary: bool,
    token_count: usize,
};

const SampleBuilder = struct {
    allocator: std.mem.Allocator,
    out: *std.Io.Writer.Allocating,
    cfg: PrepareConfig,
    stats: PrepareStats,
    text: std.ArrayListUnmanaged(u8) = .empty,
    chunks: std.ArrayListUnmanaged(OutputChunk) = .empty,
    record_tokens: usize = 0,

    fn deinit(self: *SampleBuilder) void {
        self.text.deinit(self.allocator);
        self.chunks.deinit(self.allocator);
        self.* = undefined;
    }

    fn remainingTokens(self: *const SampleBuilder) usize {
        if (self.stats.materialized_tokens >= self.cfg.max_total_tokens) return 0;
        return self.cfg.max_total_tokens - self.stats.materialized_tokens;
    }

    fn addParagraph(self: *SampleBuilder, paragraph: []const u8, token_count: usize, source_format: []const u8) !bool {
        const trimmed = std.mem.trim(u8, paragraph, " \t\r\n");
        if (trimmed.len == 0 or token_count == 0) return true;
        if (self.remainingTokens() == 0) return false;
        if (token_count > self.remainingTokens()) return false;

        if (self.record_tokens > 0 and
            self.record_tokens + token_count > self.cfg.max_record_tokens and
            self.chunks.items.len >= 2)
        {
            try self.flush(source_format);
        }

        if (self.text.items.len > 0) {
            try self.text.append(self.allocator, ' ');
        }
        const start = self.text.items.len;
        try self.text.appendSlice(self.allocator, trimmed);
        const end = self.text.items.len;
        if (end > std.math.maxInt(u32)) return error.RecordTextTooLong;
        try self.chunks.append(self.allocator, .{
            .start_char = @intCast(start),
            .end_char = @intCast(end),
        });
        self.record_tokens += token_count;
        self.stats.paragraphs += 1;
        self.stats.materialized_tokens += token_count;
        return true;
    }

    fn flush(self: *SampleBuilder, source_format: []const u8) !void {
        const include_final_boundary = std.mem.eql(u8, source_format, "final");
        if (self.chunks.items.len < 2 and !(include_final_boundary and self.chunks.items.len == 1)) {
            if (self.chunks.items.len == 1) self.stats.skipped_single_chunk_records += 1;
            self.text.clearRetainingCapacity();
            self.chunks.clearRetainingCapacity();
            self.record_tokens = 0;
            return;
        }
        const record = OutputRecord{
            .text = self.text.items,
            .chunks = self.chunks.items,
            .dataset = self.cfg.dataset_name,
            .source_format = source_format,
            .benchmark_include_final_boundary = include_final_boundary,
            .token_count = self.record_tokens,
        };
        try std.json.Stringify.value(record, .{}, &self.out.writer);
        try self.out.writer.writeByte('\n');
        self.stats.output_records += 1;
        self.stats.chunks += self.chunks.items.len;
        self.stats.boundary_targets += self.chunks.items.len - 1;
        self.text.clearRetainingCapacity();
        self.chunks.clearRetainingCapacity();
        self.record_tokens = 0;
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const opts = try parseOptions(&args) orelse return;
    const input = try compat.cwd().readFileAlloc(compat.io(), opts.input_path, allocator, .limited(1024 * 1024 * 1024));
    defer allocator.free(input);

    var prepared = try prepareBoundaryEvalJsonl(allocator, input, .{
        .dataset_name = opts.dataset_name,
        .max_total_tokens = opts.max_total_tokens,
        .max_record_tokens = opts.max_record_tokens,
    });
    defer prepared.deinit(allocator);

    try compat.cwd().writeFile(compat.io(), .{ .sub_path = opts.out_path, .data = prepared.jsonl });

    var stdout_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(compat.io(), &stdout_buf);
    try std.json.Stringify.value(.{
        .status = "written",
        .input = opts.input_path,
        .out = opts.out_path,
        .stats = prepared.stats,
    }, .{ .whitespace = .indent_2 }, &stdout.interface);
    try stdout.interface.writeByte('\n');
    try stdout.interface.flush();
}

fn parseOptions(args: *std.process.Args.Iterator) !?Options {
    var input_path: ?[]const u8 = null;
    var out_path: ?[]const u8 = null;
    var dataset_name: ?[]const u8 = null;
    var max_total_tokens: usize = default_max_total_tokens;
    var max_record_tokens: usize = default_max_record_tokens;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--input")) {
            input_path = args.next() orelse return error.MissingInputPath;
        } else if (std.mem.eql(u8, arg, "--out")) {
            out_path = args.next() orelse return error.MissingOutPath;
        } else if (std.mem.eql(u8, arg, "--dataset-name")) {
            dataset_name = args.next() orelse return error.MissingDatasetName;
        } else if (std.mem.eql(u8, arg, "--max-total-tokens")) {
            max_total_tokens = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingMaxTotalTokens, 10);
        } else if (std.mem.eql(u8, arg, "--max-record-tokens")) {
            max_record_tokens = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingMaxRecordTokens, 10);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            usage();
            return null;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }
    if (input_path == null or out_path == null or dataset_name == null) {
        usage();
        return error.MissingRequiredArgument;
    }
    if (max_total_tokens == 0 or max_record_tokens == 0) return error.InvalidTokenLimit;
    return .{
        .input_path = input_path.?,
        .out_path = out_path.?,
        .dataset_name = dataset_name.?,
        .max_total_tokens = max_total_tokens,
        .max_record_tokens = max_record_tokens,
    };
}

pub fn prepareBoundaryEvalJsonl(
    allocator: std.mem.Allocator,
    input: []const u8,
    cfg: PrepareConfig,
) !PreparedOutput {
    if (cfg.max_total_tokens == 0 or cfg.max_record_tokens == 0) return error.InvalidTokenLimit;
    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();

    var builder = SampleBuilder{
        .allocator = allocator,
        .out = &out,
        .cfg = cfg,
        .stats = .{ .dataset_name = cfg.dataset_name },
    };
    defer builder.deinit();

    var lines = std.mem.tokenizeScalar(u8, input, '\n');
    while (lines.next()) |raw_line| {
        const line = std.mem.trim(u8, raw_line, " \t\r\n");
        if (line.len == 0) continue;
        builder.stats.input_records += 1;

        var parsed = try std.json.parseFromSlice(std.json.Value, allocator, line, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const obj = try requireObject(parsed.value);
        if (obj.get("tokens") != null and obj.get("ner_tags") != null) {
            try appendChonkyTokensRecord(allocator, &builder, obj);
        } else if (obj.get("paragraphs")) |paragraphs_value| {
            try appendParagraphArrayRecord(&builder, try requireArray(paragraphs_value), "paragraph-jsonl");
        } else if (jsonString(obj.get("text"))) |text| {
            try appendTextParagraphRecord(&builder, text);
        } else {
            return error.UnsupportedBoundaryEvalRecord;
        }
        if (builder.remainingTokens() == 0) break;
    }
    try builder.flush("final");

    return .{
        .jsonl = try out.toOwnedSlice(),
        .stats = builder.stats,
    };
}

fn appendChonkyTokensRecord(
    allocator: std.mem.Allocator,
    builder: *SampleBuilder,
    obj: std.json.ObjectMap,
) !void {
    const tokens = try requireArray(obj.get("tokens") orelse return error.MissingTokens);
    const tags = try requireArray(obj.get("ner_tags") orelse return error.MissingNerTags);
    if (tokens.len != tags.len) return error.TokenTagLengthMismatch;

    var paragraph = std.ArrayListUnmanaged(u8).empty;
    defer paragraph.deinit(allocator);
    var paragraph_tokens: usize = 0;

    for (tokens, tags) |token_value, tag_value| {
        if (builder.remainingTokens() == 0) break;
        const token = jsonString(token_value) orelse return error.InvalidToken;
        if (paragraph.items.len > 0) try paragraph.append(allocator, ' ');
        try paragraph.appendSlice(allocator, token);
        paragraph_tokens += 1;
        builder.stats.source_tokens += 1;
        if (jsonTagIsSeparator(tag_value)) {
            if (!try builder.addParagraph(paragraph.items, paragraph_tokens, "chonky-tokens")) return;
            paragraph.clearRetainingCapacity();
            paragraph_tokens = 0;
        }
    }
    if (paragraph_tokens > 0 and builder.remainingTokens() > 0) {
        _ = try builder.addParagraph(paragraph.items, paragraph_tokens, "chonky-tokens");
    }
}

fn appendParagraphArrayRecord(
    builder: *SampleBuilder,
    paragraphs: []const std.json.Value,
    source_format: []const u8,
) !void {
    for (paragraphs) |paragraph_value| {
        const paragraph = jsonString(paragraph_value) orelse return error.InvalidParagraph;
        const token_count = countWhitespaceTokens(paragraph);
        builder.stats.source_tokens += token_count;
        if (!try builder.addParagraph(paragraph, token_count, source_format)) return;
    }
}

fn appendTextParagraphRecord(builder: *SampleBuilder, text: []const u8) !void {
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw_line| {
        const paragraph = std.mem.trim(u8, raw_line, " \t\r\n");
        if (paragraph.len == 0) continue;
        const token_count = countWhitespaceTokens(paragraph);
        builder.stats.source_tokens += token_count;
        if (!try builder.addParagraph(paragraph, token_count, "text-lines")) return;
    }
}

fn countWhitespaceTokens(text: []const u8) usize {
    var count: usize = 0;
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n");
    while (it.next()) |_| count += 1;
    return count;
}

fn requireObject(value: std.json.Value) !std.json.ObjectMap {
    return switch (value) {
        .object => |obj| obj,
        else => error.ExpectedJsonObject,
    };
}

fn requireArray(value: std.json.Value) ![]const std.json.Value {
    return switch (value) {
        .array => |array| array.items,
        else => error.ExpectedJsonArray,
    };
}

fn jsonString(value: ?std.json.Value) ?[]const u8 {
    const v = value orelse return null;
    return switch (v) {
        .string => |s| s,
        else => null,
    };
}

fn jsonTagIsSeparator(value: std.json.Value) bool {
    return switch (value) {
        .integer => |i| i != 0,
        .float => |f| f != 0,
        .bool => |b| b,
        else => false,
    };
}

fn usage() void {
    std.debug.print(
        \\usage: prepare-fused-chunker-boundary-eval --input <jsonl> --out <jsonl> --dataset-name <name> [options]
        \\
        \\  --max-total-tokens <n>   Maximum source tokens to materialize (default: 1000000)
        \\  --max-record-tokens <n>  Target tokens per fused eval record (default: 384)
        \\
        \\Input JSONL auto-detects:
        \\  {{"tokens":[...],"ner_tags":[...]}}  Chonky/HF token-label records
        \\  {{"paragraphs":[...]}}               Paragraph records
        \\  {{"text":"para1\npara2"}}            Newline-separated paragraph text
        \\
    , .{});
}

test "prepare boundary eval converts chonky token tags into fused chunks" {
    const allocator = std.testing.allocator;
    const input =
        \\{"tokens":["alpha","beta","gamma","delta"],"ner_tags":[0,1,0,1]}
        \\
    ;
    var prepared = try prepareBoundaryEvalJsonl(allocator, input, .{
        .dataset_name = "toy",
        .max_total_tokens = 10,
        .max_record_tokens = 10,
    });
    defer prepared.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 1), prepared.stats.output_records);
    try std.testing.expectEqual(@as(usize, 2), prepared.stats.chunks);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, std.mem.trim(u8, prepared.jsonl, " \t\r\n"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const obj = parsed.value.object;
    try std.testing.expectEqualStrings("alpha beta gamma delta", obj.get("text").?.string);
    const chunks = obj.get("chunks").?.array.items;
    try std.testing.expectEqual(@as(i64, 0), chunks[0].object.get("start_char").?.integer);
    try std.testing.expectEqual(@as(i64, 10), chunks[0].object.get("end_char").?.integer);
    try std.testing.expectEqual(@as(i64, 11), chunks[1].object.get("start_char").?.integer);
    try std.testing.expectEqual(@as(i64, 22), chunks[1].object.get("end_char").?.integer);
    try std.testing.expect(obj.get("benchmark_include_final_boundary").?.bool);
}

test "prepare boundary eval groups paragraph records by token budget" {
    const allocator = std.testing.allocator;
    const input =
        \\{"paragraphs":["one two","three four","five six","seven eight"]}
        \\
    ;
    var prepared = try prepareBoundaryEvalJsonl(allocator, input, .{
        .dataset_name = "toy",
        .max_total_tokens = 20,
        .max_record_tokens = 4,
    });
    defer prepared.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 2), prepared.stats.output_records);
    try std.testing.expectEqual(@as(usize, 4), prepared.stats.chunks);

    var lines = std.mem.tokenizeScalar(u8, prepared.jsonl, '\n');
    const first = lines.next().?;
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, first, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqualStrings("one two three four", parsed.value.object.get("text").?.string);
    try std.testing.expect(!parsed.value.object.get("benchmark_include_final_boundary").?.bool);
    const second = lines.next().?;
    var parsed_second = try std.json.parseFromSlice(std.json.Value, allocator, second, .{ .ignore_unknown_fields = true });
    defer parsed_second.deinit();
    try std.testing.expect(parsed_second.value.object.get("benchmark_include_final_boundary").?.bool);
}

test "prepare boundary eval preserves final one-chunk record for Chonky final separator" {
    const allocator = std.testing.allocator;
    const input =
        \\{"paragraphs":["only final paragraph"]}
        \\
    ;
    var prepared = try prepareBoundaryEvalJsonl(allocator, input, .{
        .dataset_name = "toy",
        .max_total_tokens = 20,
        .max_record_tokens = 4,
    });
    defer prepared.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 1), prepared.stats.output_records);
    try std.testing.expectEqual(@as(usize, 1), prepared.stats.chunks);
    try std.testing.expectEqual(@as(usize, 0), prepared.stats.boundary_targets);
    try std.testing.expectEqual(@as(usize, 0), prepared.stats.skipped_single_chunk_records);

    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, std.mem.trim(u8, prepared.jsonl, " \t\r\n"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqualStrings("only final paragraph", parsed.value.object.get("text").?.string);
    try std.testing.expectEqual(@as(usize, 1), parsed.value.object.get("chunks").?.array.items.len);
    try std.testing.expect(parsed.value.object.get("benchmark_include_final_boundary").?.bool);
}

test "prepare boundary eval respects total token cap" {
    const allocator = std.testing.allocator;
    const input =
        \\{"tokens":["a","b","c","d","e","f"],"ner_tags":[0,1,0,1,0,1]}
        \\
    ;
    var prepared = try prepareBoundaryEvalJsonl(allocator, input, .{
        .dataset_name = "toy",
        .max_total_tokens = 4,
        .max_record_tokens = 8,
    });
    defer prepared.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 4), prepared.stats.materialized_tokens);
    try std.testing.expectEqual(@as(usize, 1), prepared.stats.output_records);
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, std.mem.trim(u8, prepared.jsonl, " \t\r\n"), .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    try std.testing.expectEqualStrings("a b c d", parsed.value.object.get("text").?.string);
}
