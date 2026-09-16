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

pub const TemplateKind = enum {
    llama3,
    gemma,
    chatml,
    alpaca,
};

pub const Role = enum { system, user, assistant, tool };

pub const Message = struct {
    role: Role,
    content: []const u8,
    name: ?[]const u8 = null,
    tool_call_id: ?[]const u8 = null,
    tool_calls_json: ?[]const u8 = null,
};

pub const AssistantSpan = struct {
    start: usize,
    end: usize,
};

pub const RenderResult = struct {
    allocator: std.mem.Allocator,
    text: []u8,
    assistant_spans: []AssistantSpan,

    pub fn deinit(self: *RenderResult) void {
        self.allocator.free(self.text);
        self.allocator.free(self.assistant_spans);
        self.* = undefined;
    }
};

pub const RenderOptions = struct {
    add_generation_prompt: bool = false,
};

const gemma4_bos = "<bos>";
const gemma4_turn_end = "<turn|>\n";
const gemma4_tool_string_delimiter = "<|\"|>";

pub const RenderError = error{
    UnsupportedRoleForTemplate,
    InvalidGemmaToolCalls,
    InvalidGemmaToolIdentifier,
    UnsupportedGemmaToolArgument,
    OutOfMemory,
};

fn roleStr(role: Role) []const u8 {
    return switch (role) {
        .system => "system",
        .user => "user",
        .assistant => "assistant",
        .tool => "tool",
    };
}

fn parseGemmaToolJson(
    allocator: std.mem.Allocator,
    source: []const u8,
) RenderError!std.json.Parsed(std.json.Value) {
    return std.json.parseFromSlice(std.json.Value, allocator, source, .{}) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidGemmaToolCalls,
    };
}

fn validGemmaToolIdentifier(value: []const u8) bool {
    if (value.len == 0) return false;
    for (value) |ch| {
        if (!std.ascii.isAlphanumeric(ch) and ch != '_' and ch != '-' and ch != '.') return false;
    }
    return true;
}

fn gemmaToolStringIsRepresentable(value: []const u8) bool {
    return std.mem.indexOf(u8, value, "<|") == null and
        std.mem.indexOf(u8, value, "|>") == null;
}

fn appendGemmaToolArgument(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    value: std.json.Value,
) RenderError!void {
    switch (value) {
        .null => try buf.appendSlice(allocator, "null"),
        .bool => |flag| try buf.appendSlice(allocator, if (flag) "true" else "false"),
        .integer => |number| {
            const rendered = try std.fmt.allocPrint(allocator, "{d}", .{number});
            defer allocator.free(rendered);
            try buf.appendSlice(allocator, rendered);
        },
        .float => |number| {
            const rendered = try std.fmt.allocPrint(allocator, "{d}", .{number});
            defer allocator.free(rendered);
            try buf.appendSlice(allocator, rendered);
        },
        .number_string => |number| try buf.appendSlice(allocator, number),
        .string => |string| {
            if (!gemmaToolStringIsRepresentable(string)) return error.UnsupportedGemmaToolArgument;
            try buf.appendSlice(allocator, gemma4_tool_string_delimiter);
            try buf.appendSlice(allocator, string);
            try buf.appendSlice(allocator, gemma4_tool_string_delimiter);
        },
        .array, .object => return error.UnsupportedGemmaToolArgument,
    }
}

fn appendGemmaToolArguments(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    value: std.json.Value,
) RenderError!void {
    if (value != .object) return error.InvalidGemmaToolCalls;

    const keys = try allocator.alloc([]const u8, value.object.count());
    defer allocator.free(keys);
    var key_idx: usize = 0;
    var iterator = value.object.iterator();
    while (iterator.next()) |entry| : (key_idx += 1) {
        keys[key_idx] = entry.key_ptr.*;
    }
    std.mem.sort([]const u8, keys, {}, struct {
        fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
            return std.mem.lessThan(u8, lhs, rhs);
        }
    }.lessThan);

    for (keys, 0..) |key, idx| {
        if (!validGemmaToolIdentifier(key)) return error.InvalidGemmaToolIdentifier;
        if (idx != 0) try buf.append(allocator, ',');
        try buf.appendSlice(allocator, key);
        try buf.append(allocator, ':');
        try appendGemmaToolArgument(allocator, buf, value.object.get(key).?);
    }
}

fn appendGemmaToolCall(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    value: std.json.Value,
) RenderError!void {
    if (value != .object) return error.InvalidGemmaToolCalls;
    const function_value = value.object.get("function") orelse value;
    if (function_value != .object) return error.InvalidGemmaToolCalls;

    const name_value = function_value.object.get("name") orelse return error.InvalidGemmaToolCalls;
    if (name_value != .string) return error.InvalidGemmaToolCalls;
    if (!validGemmaToolIdentifier(name_value.string)) return error.InvalidGemmaToolIdentifier;

    try buf.appendSlice(allocator, "<|tool_call>call:");
    try buf.appendSlice(allocator, name_value.string);
    try buf.append(allocator, '{');

    if (function_value.object.get("arguments") orelse function_value.object.get("arguments_json")) |arguments| {
        if (arguments == .string) {
            var parsed_arguments = try parseGemmaToolJson(allocator, arguments.string);
            defer parsed_arguments.deinit();
            try appendGemmaToolArguments(allocator, buf, parsed_arguments.value);
        } else {
            try appendGemmaToolArguments(allocator, buf, arguments);
        }
    }

    try buf.appendSlice(allocator, "}<tool_call|>");
}

fn appendGemmaToolCalls(
    allocator: std.mem.Allocator,
    buf: *std.ArrayList(u8),
    source: []const u8,
) RenderError!void {
    var parsed = try parseGemmaToolJson(allocator, source);
    defer parsed.deinit();
    if (parsed.value != .array or parsed.value.array.items.len == 0) {
        return error.InvalidGemmaToolCalls;
    }
    for (parsed.value.array.items) |tool_call| {
        try appendGemmaToolCall(allocator, buf, tool_call);
    }
}

pub fn render(
    allocator: std.mem.Allocator,
    kind: TemplateKind,
    messages: []const Message,
    options: RenderOptions,
) RenderError!RenderResult {
    return switch (kind) {
        .llama3 => renderLlama3(allocator, messages, options),
        .gemma => renderGemma(allocator, messages, options),
        .chatml => renderChatml(allocator, messages, options),
        .alpaca => renderAlpaca(allocator, messages, options),
    };
}

fn renderLlama3(
    allocator: std.mem.Allocator,
    messages: []const Message,
    options: RenderOptions,
) RenderError!RenderResult {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    var spans: std.ArrayList(AssistantSpan) = .empty;
    errdefer spans.deinit(allocator);

    if (messages.len > 0) {
        try buf.appendSlice(allocator, "<|begin_of_text|>");
    }

    for (messages) |msg| {
        const role_name = roleStr(msg.role);
        const span_start = buf.items.len;
        try buf.appendSlice(allocator, "<|start_header_id|>");
        try buf.appendSlice(allocator, role_name);
        try buf.appendSlice(allocator, "<|end_header_id|>\n\n");
        try buf.appendSlice(allocator, msg.content);
        try buf.appendSlice(allocator, "<|eot_id|>");
        if (msg.role == .assistant) {
            try spans.append(allocator, .{ .start = span_start, .end = buf.items.len });
        }
    }

    if (options.add_generation_prompt) {
        try buf.appendSlice(allocator, "<|start_header_id|>assistant<|end_header_id|>\n\n");
    }

    return RenderResult{
        .allocator = allocator,
        .text = try buf.toOwnedSlice(allocator),
        .assistant_spans = try spans.toOwnedSlice(allocator),
    };
}

// Match the canonical serving template's text/tool subset. Tool observations
// share the assistant turn but never enter completion-only loss spans.
fn renderGemma(
    allocator: std.mem.Allocator,
    messages: []const Message,
    options: RenderOptions,
) RenderError!RenderResult {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    var spans: std.ArrayList(AssistantSpan) = .empty;
    errdefer spans.deinit(allocator);
    try buf.appendSlice(allocator, gemma4_bos);
    var previous_role: ?Role = null;
    var pending_tool = false;
    for (messages, 0..) |msg, index| {
        if (msg.role == .tool) continue;
        pending_tool = false;
        if (msg.role != .assistant or previous_role != .assistant) {
            try buf.appendSlice(allocator, "<|turn>");
            try buf.appendSlice(allocator, if (msg.role == .assistant) "model" else roleStr(msg.role));
            try buf.append(allocator, '\n');
        }
        var span_start = buf.items.len;
        var has_response = false;
        if (msg.role == .assistant) {
            if (msg.tool_calls_json) |calls| {
                try appendGemmaToolCalls(allocator, &buf, calls);
                var next = index + 1;
                while (next < messages.len and messages[next].role == .tool) : (next += 1) {
                    const response = messages[next];
                    if (!has_response and buf.items.len > span_start) {
                        try spans.append(allocator, .{ .start = span_start, .end = buf.items.len });
                    }
                    has_response = true;
                    var parsed = try parseGemmaToolJson(allocator, calls);
                    defer parsed.deinit();
                    var name = response.name orelse "unknown";
                    if (parsed.value == .array) for (parsed.value.array.items) |call| {
                        if (call != .object) continue;
                        const id = call.object.get("id") orelse continue;
                        if (id == .string and response.tool_call_id != null and std.mem.eql(u8, id.string, response.tool_call_id.?)) {
                            const function = call.object.get("function") orelse continue;
                            if (function == .object) if (function.object.get("name")) |v| {
                                if (v == .string) name = v.string;
                            };
                        }
                    };
                    if (!validGemmaToolIdentifier(name)) return error.InvalidGemmaToolIdentifier;
                    try buf.appendSlice(allocator, "<|tool_response>response:");
                    try buf.appendSlice(allocator, name);
                    try buf.appendSlice(allocator, "{value:");
                    try appendGemmaToolArgument(allocator, &buf, .{ .string = response.content });
                    try buf.appendSlice(allocator, "}<tool_response|>");
                    span_start = buf.items.len;
                }
                pending_tool = true;
            }
        }
        const content_start = buf.items.len;
        if (msg.role == .assistant) {
            // Canonical strip_thinking removes channel annotations and private
            // content through the next channel terminator.
            var parts = std.mem.splitSequence(u8, msg.content, "<channel|>");
            while (parts.next()) |part| {
                const end = std.mem.indexOf(u8, part, "<|channel>") orelse part.len;
                try buf.appendSlice(allocator, part[0..end]);
            }
            const content = std.mem.trim(u8, buf.items[content_start..], &std.ascii.whitespace);
            std.mem.copyForwards(u8, buf.items[content_start..], content);
            buf.shrinkRetainingCapacity(content_start + content.len);
        } else {
            try buf.appendSlice(allocator, std.mem.trim(u8, msg.content, &std.ascii.whitespace));
        }
        const has_content = buf.items.len > content_start;
        var next_role: ?Role = null;
        for (messages[index + 1 ..]) |next| {
            if (next.role != .tool) {
                next_role = next.role;
                break;
            }
        }
        const continues = msg.role == .assistant and next_role == .assistant and (msg.tool_calls_json == null or has_response);
        if (pending_tool and !has_response) {
            try buf.appendSlice(allocator, "<|tool_response>");
        } else if (!continues and !(has_response and !has_content and next_role == null)) {
            try buf.appendSlice(allocator, gemma4_turn_end);
        }
        if (msg.role == .assistant and buf.items.len > span_start) {
            try spans.append(allocator, .{ .start = span_start, .end = buf.items.len });
        }
        previous_role = msg.role;
    }
    if (options.add_generation_prompt and !pending_tool) try buf.appendSlice(allocator, "<|turn>model\n");
    return .{
        .allocator = allocator,
        .text = try buf.toOwnedSlice(allocator),
        .assistant_spans = try spans.toOwnedSlice(allocator),
    };
}

fn renderChatml(
    allocator: std.mem.Allocator,
    messages: []const Message,
    options: RenderOptions,
) RenderError!RenderResult {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    var spans: std.ArrayList(AssistantSpan) = .empty;
    errdefer spans.deinit(allocator);

    for (messages) |msg| {
        const role_name = roleStr(msg.role);
        const span_start = buf.items.len;
        try buf.appendSlice(allocator, "<|im_start|>");
        try buf.appendSlice(allocator, role_name);
        try buf.append(allocator, '\n');
        try buf.appendSlice(allocator, msg.content);
        try buf.appendSlice(allocator, "<|im_end|>\n");
        if (msg.role == .assistant) {
            try spans.append(allocator, .{ .start = span_start, .end = buf.items.len });
        }
    }

    if (options.add_generation_prompt) {
        try buf.appendSlice(allocator, "<|im_start|>assistant\n");
    }

    return RenderResult{
        .allocator = allocator,
        .text = try buf.toOwnedSlice(allocator),
        .assistant_spans = try spans.toOwnedSlice(allocator),
    };
}

fn renderAlpaca(
    allocator: std.mem.Allocator,
    messages: []const Message,
    options: RenderOptions,
) RenderError!RenderResult {
    var buf: std.ArrayList(u8) = .empty;
    errdefer buf.deinit(allocator);
    var spans: std.ArrayList(AssistantSpan) = .empty;
    errdefer spans.deinit(allocator);

    for (messages) |msg| {
        switch (msg.role) {
            .tool => return error.UnsupportedRoleForTemplate,
            .system => {
                try buf.appendSlice(allocator, msg.content);
                try buf.appendSlice(allocator, "\n\n");
            },
            .user => {
                try buf.appendSlice(allocator, "### Instruction:\n");
                try buf.appendSlice(allocator, msg.content);
                try buf.appendSlice(allocator, "\n\n");
            },
            .assistant => {
                const span_start = buf.items.len;
                try buf.appendSlice(allocator, "### Response:\n");
                try buf.appendSlice(allocator, msg.content);
                try buf.appendSlice(allocator, "\n\n");
                try spans.append(allocator, .{ .start = span_start, .end = buf.items.len });
            },
        }
    }

    if (options.add_generation_prompt) {
        try buf.appendSlice(allocator, "### Response:\n");
    }

    return RenderResult{
        .allocator = allocator,
        .text = try buf.toOwnedSlice(allocator),
        .assistant_spans = try spans.toOwnedSlice(allocator),
    };
}

pub fn makeCompletionLabels(
    allocator: std.mem.Allocator,
    input_ids: []const i32,
    token_byte_offsets: []const usize,
    assistant_spans: []const AssistantSpan,
    ignore_label: i32,
) ![]i32 {
    std.debug.assert(input_ids.len == token_byte_offsets.len);
    const labels = try allocator.alloc(i32, input_ids.len);
    errdefer allocator.free(labels);

    for (input_ids, token_byte_offsets, 0..) |id, off, i| {
        var in_span = false;
        for (assistant_spans) |s| {
            if (off >= s.start and off < s.end) {
                in_span = true;
                break;
            }
        }
        labels[i] = if (in_span) id else ignore_label;
    }
    return labels;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "llama3 render system+user+assistant" {
    const allocator = std.testing.allocator;
    const messages = [_]Message{
        .{ .role = .system, .content = "You are helpful." },
        .{ .role = .user, .content = "Hi" },
        .{ .role = .assistant, .content = "Hello!" },
    };
    var result = try render(allocator, .llama3, &messages, .{});
    defer result.deinit();

    const expected =
        "<|begin_of_text|>" ++
        "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>" ++
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>" ++
        "<|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>";
    try std.testing.expectEqualStrings(expected, result.text);
    try std.testing.expectEqual(@as(usize, 1), result.assistant_spans.len);

    const span = result.assistant_spans[0];
    const expected_span_start =
        "<|begin_of_text|>".len +
        "<|start_header_id|>system<|end_header_id|>\n\nYou are helpful.<|eot_id|>".len +
        "<|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|>".len;
    try std.testing.expectEqual(expected_span_start, span.start);
    try std.testing.expectEqual(result.text.len, span.end);
    try std.testing.expectEqualStrings(
        "<|start_header_id|>assistant<|end_header_id|>\n\nHello!<|eot_id|>",
        result.text[span.start..span.end],
    );
}

test "gemma4 render matches inference wire format and assistant loss span" {
    const allocator = std.testing.allocator;
    const messages = [_]Message{
        .{ .role = .system, .content = "Be terse." },
        .{ .role = .user, .content = "2+2?" },
        .{ .role = .assistant, .content = "4" },
    };
    var result = try render(allocator, .gemma, &messages, .{});
    defer result.deinit();

    const expected =
        "<bos><|turn>system\nBe terse.<turn|>\n<|turn>user\n2+2?<turn|>\n" ++
        "<|turn>model\n4<turn|>\n";
    try std.testing.expectEqualStrings(expected, result.text);
    try std.testing.expectEqual(@as(usize, 1), result.assistant_spans.len);
    const span = result.assistant_spans[0];
    try std.testing.expectEqualStrings(
        "4<turn|>\n",
        result.text[span.start..span.end],
    );
    try std.testing.expect(!std.mem.startsWith(
        u8,
        result.text[span.start..span.end],
        "<|turn>model\n<|channel>thought\n<channel|>",
    ));
}

test "gemma4 render keeps tool calls private across a multi-turn loop" {
    const allocator = std.testing.allocator;
    const messages = [_]Message{
        .{ .role = .user, .content = "list files" },
        .{
            .role = .assistant,
            .content = "Checking",
            .tool_calls_json = "[{\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"shell\",\"arguments\":\"{\\\"z\\\":2,\\\"cmd\\\":\\\"ls\\\",\\\"force\\\":true}\"}}]",
        },
        .{ .role = .tool, .name = "shell", .tool_call_id = "call_1", .content = "file.txt" },
        .{ .role = .assistant, .content = "Found file.txt" },
    };
    var result = try render(allocator, .gemma, &messages, .{});
    defer result.deinit();

    try std.testing.expect(std.mem.indexOf(u8, result.text, "<|turn>tool") == null);
    try std.testing.expect(std.mem.indexOf(u8, result.text, "<|tool_response>response:shell{") != null);
    for (result.assistant_spans) |span| {
        try std.testing.expect(std.mem.indexOf(u8, result.text[span.start..span.end], "<|tool_response>response:") == null);
    }
}

test "gemma4 tool-call wire round trips through the inference parser" {
    const tool_parser = @import("../pipelines/tool_parser.zig");
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "genai_config.json",
        .data = "{\"tool_call_format\":\"functiongemma\"}",
    });
    try tmp.dir.writeFile(std.testing.io, .{
        .sub_path = "special_tokens_map.json",
        .data =
        \\{"gemma4-tool-call":true,"stc_token":"<|tool_call>","etc_token":"<tool_call|>","std_token":"<|tool>","etd_token":"<tool|>","escape_token":"<|\"|>"}
        ,
    });

    var model_dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const model_dir_len = try tmp.dir.realPath(std.testing.io, &model_dir_buf);
    const model_dir = model_dir_buf[0..model_dir_len];
    var parser = (try tool_parser.loadParser(allocator, model_dir)).?;
    defer parser.deinit();

    const messages = [_]Message{.{
        .role = .assistant,
        .content = "",
        .tool_calls_json = "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"priority\\\":2,\\\"order_id\\\":\\\"A-42\\\"}\"}}]",
    }};
    var rendered = try render(allocator, .gemma, &messages, .{});
    defer rendered.deinit();
    const assistant = rendered.text[rendered.assistant_spans[0].start..rendered.assistant_spans[0].end];
    const call_start = std.mem.indexOf(u8, assistant, "<|tool_call>").?;
    const call_end_start = std.mem.indexOfPos(u8, assistant, call_start, "<tool_call|>").?;
    const call_end = call_end_start + "<tool_call|>".len;

    const update = try parser.feed(assistant[call_start..call_end]);
    try std.testing.expectEqual(@as(usize, 1), update.new_calls.len);
    try std.testing.expectEqualStrings("lookup", update.new_calls[0].function.name);
    var arguments = try std.json.parseFromSlice(
        std.json.Value,
        allocator,
        update.new_calls[0].function.arguments,
        .{},
    );
    defer arguments.deinit();
    try std.testing.expectEqualStrings("A-42", arguments.value.object.get("order_id").?.string);
    try std.testing.expectEqual(@as(i64, 2), arguments.value.object.get("priority").?.integer);
}

test "gemma4 tool-call renderer rejects values the inference grammar cannot preserve" {
    const allocator = std.testing.allocator;
    const nested_arguments = [_]Message{.{
        .role = .assistant,
        .content = "",
        .tool_calls_json = "[{\"function\":{\"name\":\"lookup\",\"arguments\":\"{\\\"nested\\\":{\\\"id\\\":42}}\"}}]",
    }};
    try std.testing.expectError(
        error.UnsupportedGemmaToolArgument,
        render(allocator, .gemma, &nested_arguments, .{}),
    );

    const invalid_name = [_]Message{.{
        .role = .assistant,
        .content = "",
        .tool_calls_json = "[{\"function\":{\"name\":\"bad name\",\"arguments\":\"{}\"}}]",
    }};
    try std.testing.expectError(
        error.InvalidGemmaToolIdentifier,
        render(allocator, .gemma, &invalid_name, .{}),
    );
}

test "gemma4 render preserves explicit assistant channels without duplicating thought prompt" {
    const allocator = std.testing.allocator;
    const messages = [_]Message{
        .{ .role = .user, .content = "Explain" },
        .{ .role = .assistant, .content = "<|channel>thought\n<channel|>reason<|channel>final\n<channel|>answer" },
    };
    var result = try render(allocator, .gemma, &messages, .{});
    defer result.deinit();

    try std.testing.expectEqualStrings("<bos><|turn>user\nExplain<turn|>\n<|turn>model\nreasonanswer<turn|>\n", result.text);
}

test "chatml render with tool role" {
    const allocator = std.testing.allocator;
    const messages = [_]Message{
        .{ .role = .system, .content = "sys" },
        .{ .role = .user, .content = "call weather" },
        .{ .role = .assistant, .content = "calling..." },
        .{ .role = .tool, .content = "{\"temp\":72}" },
        .{ .role = .assistant, .content = "It's 72." },
    };
    var result = try render(allocator, .chatml, &messages, .{});
    defer result.deinit();

    const expected =
        "<|im_start|>system\nsys<|im_end|>\n" ++
        "<|im_start|>user\ncall weather<|im_end|>\n" ++
        "<|im_start|>assistant\ncalling...<|im_end|>\n" ++
        "<|im_start|>tool\n{\"temp\":72}<|im_end|>\n" ++
        "<|im_start|>assistant\nIt's 72.<|im_end|>\n";
    try std.testing.expectEqualStrings(expected, result.text);
    try std.testing.expectEqual(@as(usize, 2), result.assistant_spans.len);
}

test "alpaca rejects tool role" {
    const allocator = std.testing.allocator;
    const messages = [_]Message{
        .{ .role = .user, .content = "do a thing" },
        .{ .role = .tool, .content = "result" },
    };
    const err = render(allocator, .alpaca, &messages, .{});
    try std.testing.expectError(error.UnsupportedRoleForTemplate, err);
}

test "two assistant turns produce two spans" {
    const allocator = std.testing.allocator;
    const messages = [_]Message{
        .{ .role = .user, .content = "q1" },
        .{ .role = .assistant, .content = "a1" },
        .{ .role = .user, .content = "q2" },
        .{ .role = .assistant, .content = "a2" },
    };
    var result = try render(allocator, .chatml, &messages, .{});
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 2), result.assistant_spans.len);
    try std.testing.expectEqualStrings(
        "<|im_start|>assistant\na1<|im_end|>\n",
        result.text[result.assistant_spans[0].start..result.assistant_spans[0].end],
    );
    try std.testing.expectEqualStrings(
        "<|im_start|>assistant\na2<|im_end|>\n",
        result.text[result.assistant_spans[1].start..result.assistant_spans[1].end],
    );
}

test "add_generation_prompt appends suffix without adding a span" {
    const allocator = std.testing.allocator;
    const messages = [_]Message{
        .{ .role = .user, .content = "hi" },
    };

    {
        var r = try render(allocator, .llama3, &messages, .{ .add_generation_prompt = true });
        defer r.deinit();
        try std.testing.expect(std.mem.endsWith(u8, r.text, "<|start_header_id|>assistant<|end_header_id|>\n\n"));
        try std.testing.expectEqual(@as(usize, 0), r.assistant_spans.len);
    }
    {
        var r = try render(allocator, .gemma, &messages, .{ .add_generation_prompt = true });
        defer r.deinit();
        try std.testing.expectEqualStrings(
            "<bos><|turn>user\nhi<turn|>\n<|turn>model\n",
            r.text,
        );
        try std.testing.expectEqual(@as(usize, 0), r.assistant_spans.len);
    }
    {
        var r = try render(allocator, .chatml, &messages, .{ .add_generation_prompt = true });
        defer r.deinit();
        try std.testing.expect(std.mem.endsWith(u8, r.text, "<|im_start|>assistant\n"));
        try std.testing.expectEqual(@as(usize, 0), r.assistant_spans.len);
    }
    {
        var r = try render(allocator, .alpaca, &messages, .{ .add_generation_prompt = true });
        defer r.deinit();
        try std.testing.expect(std.mem.endsWith(u8, r.text, "### Response:\n"));
        try std.testing.expectEqual(@as(usize, 0), r.assistant_spans.len);
    }
}

test "makeCompletionLabels masks tokens outside assistant spans" {
    const allocator = std.testing.allocator;
    const input_ids = [_]i32{ 10, 11, 12, 13, 14, 15 };
    //   offsets:      0    5    10   20   25   30
    //   span:         [20, 30)  -> indices 3, 4
    const offsets = [_]usize{ 0, 5, 10, 20, 25, 30 };
    const spans = [_]AssistantSpan{.{ .start = 20, .end = 30 }};

    const labels = try makeCompletionLabels(allocator, &input_ids, &offsets, &spans, -100);
    defer allocator.free(labels);

    try std.testing.expectEqual(@as(i32, -100), labels[0]);
    try std.testing.expectEqual(@as(i32, -100), labels[1]);
    try std.testing.expectEqual(@as(i32, -100), labels[2]);
    try std.testing.expectEqual(@as(i32, 13), labels[3]);
    try std.testing.expectEqual(@as(i32, 14), labels[4]);
    try std.testing.expectEqual(@as(i32, -100), labels[5]);
}

test "gemma4 completion labels exclude role and include answer and turn end" {
    const allocator = std.testing.allocator;
    const messages = [_]Message{
        .{ .role = .user, .content = "2+2?" },
        .{ .role = .assistant, .content = "4" },
    };
    var rendered = try render(allocator, .gemma, &messages, .{});
    defer rendered.deinit();

    const role_offset = std.mem.indexOf(u8, rendered.text, "<|turn>model").?;
    const answer_offset = std.mem.indexOfPos(u8, rendered.text, role_offset, "4").?;
    const turn_end_offset = std.mem.indexOfPos(u8, rendered.text, answer_offset + 1, "<turn|>").?;
    const ids = [_]i32{ 10, 11, 12 };
    const offsets = [_]usize{ role_offset, answer_offset, turn_end_offset };
    const labels = try makeCompletionLabels(allocator, &ids, &offsets, rendered.assistant_spans, -100);
    defer allocator.free(labels);
    try std.testing.expectEqualSlices(i32, &.{ -100, 11, 12 }, labels);
}

test "empty messages renders empty string and empty spans" {
    const allocator = std.testing.allocator;
    const messages = [_]Message{};
    inline for (.{ TemplateKind.llama3, .gemma, .chatml, .alpaca }) |k| {
        var r = try render(allocator, k, &messages, .{});
        defer r.deinit();
        try std.testing.expectEqual(@as(usize, 0), r.text.len);
        try std.testing.expectEqual(@as(usize, 0), r.assistant_spans.len);
    }
}

test "gemma4 training renderer matches canonical Jinja system tool and channel history" {
    const jinja = @import("jinja");
    const allocator = std.testing.allocator;
    var template = try jinja.Template.initHuggingFace(allocator, jinja.gemma4_canonical_template);
    defer template.deinit();
    const messages = [_]Message{
        .{ .role = .system, .content = " Be terse. " },
        .{ .role = .user, .content = " Find the code. " },
        .{ .role = .assistant, .content = "Checking", .tool_calls_json = "[{\"id\":\"c1\",\"function\":{\"name\":\"lookup\",\"arguments\":{\"query\":\"code\",\"limit\":2}}}]" },
        .{ .role = .tool, .content = "AZURE-731", .tool_call_id = "c1" },
        .{ .role = .assistant, .content = "<|channel>thought\nprivate<channel|>Found it." },
        .{ .role = .user, .content = "Thanks" },
        .{ .role = .assistant, .content = "Done" },
    };
    const reference = [_]jinja.ChatMessage{
        .{ .role = "system", .content = messages[0].content },
        .{ .role = "user", .content = messages[1].content },
        .{ .role = "assistant", .content = "Checking", .tool_calls = &.{.{ .id = "c1", .name = "lookup", .arguments = "{\"query\":\"code\",\"limit\":2}" }} },
        .{ .role = "tool", .content = "AZURE-731", .tool_call_id = "c1" },
        .{ .role = "assistant", .content = messages[4].content },
        .{ .role = "user", .content = "Thanks" },
        .{ .role = "assistant", .content = "Done" },
    };
    for (1..messages.len + 1) |end| {
        for ([_]bool{ false, true }) |generation_prompt| {
            var arena = std.heap.ArenaAllocator.init(allocator);
            defer arena.deinit();
            var context = try jinja.chatTemplateContext(arena.allocator(), reference[0..end], .{ .bos_token = "<bos>", .add_generation_prompt = generation_prompt, .enable_thinking = false });
            const expected = try template.render(arena.allocator(), &context);
            var actual = try render(allocator, .gemma, messages[0..end], .{ .add_generation_prompt = generation_prompt });
            defer actual.deinit();
            try std.testing.expectEqualStrings(expected, actual.text);
            for (actual.assistant_spans) |span| {
                try std.testing.expect(std.mem.indexOf(u8, actual.text[span.start..span.end], "AZURE-731") == null);
            }
        }
    }
}
