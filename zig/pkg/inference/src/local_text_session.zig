// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
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

//! Single-owner JSONL sessions for the two bounded local Qwen decoders.
const std = @import("std");
const generation = @import("pipelines/generation.zig");
const tokenizer_mod = @import("inference_tokenizer");
const memory = @import("runtime/tier/memory.zig");
const platform = @import("antfly_platform");
pub const Config = struct {
    prompt_lookup: usize = 0,
    context: usize,
    max_tokens: usize,
    raw: bool,
    thinking: ?bool,
    eos: []const u32,
    ignore_eos: bool,
    model: []const u8,
};
const Request = struct { prompt: []const u8 = "", max_tokens: ?usize = null, reset: bool = false, quit: bool = false };
pub fn writeStdout(bytes: []const u8) !void {
    var offset: usize = 0;
    while (offset < bytes.len) {
        const n = std.c.write(1, bytes.ptr + offset, bytes.len - offset);
        if (n <= 0) return error.OutputWriteFailed;
        offset += @intCast(n);
    }
}
pub fn completeUtf8Prefix(bytes: []const u8) usize {
    var at: usize = 0;
    while (at < bytes.len) {
        const n = std.unicode.utf8ByteSequenceLength(bytes[at]) catch return at;
        if (n > bytes.len - at) return at;
        _ = std.unicode.utf8Decode(bytes[at..][0..n]) catch return at;
        at += n;
    }
    return at;
}
pub fn validText(a: std.mem.Allocator, bytes: []const u8) ![]u8 {
    var result: std.ArrayList(u8) = .empty;
    errdefer result.deinit(a);
    var at: usize = 0;
    while (at < bytes.len) {
        const good = completeUtf8Prefix(bytes[at..]);
        try result.appendSlice(a, bytes[at..][0..good]);
        at += good;
        if (at < bytes.len) {
            try result.appendSlice(a, "\xef\xbf\xbd");
            at += 1;
        }
    }
    return result.toOwnedSlice(a);
}
fn emit(a: std.mem.Allocator, value: anytype) !void {
    const json = try std.json.Stringify.valueAlloc(a, value, .{});
    defer a.free(json);
    try writeStdout(json);
    try writeStdout("\n");
}
const Input = struct {
    buffer: [8192]u8 = undefined,
    at: usize = 0,
    end: usize = 0,
    fn line(self: *Input, a: std.mem.Allocator) !?[]u8 {
        var result: std.ArrayList(u8) = .empty;
        errdefer result.deinit(a);
        while (true) {
            if (self.at == self.end) {
                const n = std.c.read(0, &self.buffer, self.buffer.len);
                if (n < 0) return error.InputReadFailed;
                if (n == 0) {
                    if (result.items.len == 0) {
                        result.deinit(a);
                        return null;
                    }
                    return try result.toOwnedSlice(a);
                }
                self.at = 0;
                self.end = @intCast(n);
            }
            const byte = self.buffer[self.at];
            self.at += 1;
            if (byte == '\n') return try result.toOwnedSlice(a);
            if (result.items.len == 1024 * 1024) return error.SessionRequestTooLarge;
            try result.append(a, byte);
        }
    }
};
pub fn exactPrefix(cached: []const i32, prompt: []const i32) bool {
    return cached.len <= prompt.len and std.mem.eql(i32, cached, prompt[0..cached.len]);
}
fn step(decoder: anytype, ids: []const i32) !u32 {
    if (comptime @hasDecl(@TypeOf(decoder.*), "prefill")) return decoder.prefill(ids);
    if (decoder.prefill_chunk > 0 and ids.len > 1) return decoder.prefillChunk(ids);
    var next: u32 = 0;
    for (ids, 0..) |id, i| next = try decoder.prefillToken(@intCast(id), i + 1 == ids.len);
    return next;
}
fn chunkSize(decoder: anytype) usize {
    if (comptime @hasDecl(@TypeOf(decoder.*), "prefill")) return decoder.plan.chunk;
    return @max(1, decoder.prefill_chunk);
}
fn readBytes(decoder: anytype) u64 {
    if (comptime @hasDecl(@TypeOf(decoder.*), "bytesRead")) return decoder.bytesRead();
    return decoder.pagingStats().bytes_read;
}
fn milliseconds(io: std.Io, start: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(std.Io.Timestamp.durationTo(start, std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e6;
}
pub fn run(a: std.mem.Allocator, io: std.Io, decoder: anytype, controller: *memory.AdmissionController, tokenizer: tokenizer_mod.Tokenizer, template: ?*generation.ChatTemplate, config: Config) !void {
    var session: Session(@TypeOf(decoder)) = .{ .a = a, .io = io, .decoder = decoder, .controller = controller, .tokenizer = tokenizer, .template = template, .config = config };
    defer session.deinit();
    try emit(a, .{ .event = "ready", .model = config.model, .context_size = config.context, .prefill_chunk = chunkSize(decoder), .protocol = "qwen-local-jsonl-v1" });
    var input: Input = .{};
    while (try input.line(a)) |line| {
        defer a.free(line);
        if (std.mem.trim(u8, line, " \r\t").len == 0) continue;
        const request = std.json.parseFromSlice(Request, a, line, .{}) catch |err| {
            try emit(a, .{ .event = "error", .error_name = @errorName(err), .fatal = false });
            continue;
        };
        defer request.deinit();
        if (request.value.quit) break;
        session.turn(request.value) catch |err| {
            // Execution errors may leave partially advanced state. Fail closed.
            const fatal = decoder.failed or session.executing;
            try emit(a, .{ .event = "error", .error_name = @errorName(err), .fatal = fatal });
            if (fatal) return err;
        };
    }
}
fn Session(comptime D: type) type {
    return struct {
        const Self = @This();
        a: std.mem.Allocator,
        io: std.Io,
        decoder: D,
        controller: *memory.AdmissionController,
        tokenizer: tokenizer_mod.Tokenizer,
        template: ?*generation.ChatTemplate,
        config: Config,
        history: std.ArrayList(generation.Message) = .empty,
        cached: std.ArrayList(i32) = .empty,
        next: ?u32 = null,
        executing: bool = false,
        fn clearHistory(self: *Self) void {
            for (self.history.items) |message| self.a.free(message.content);
            self.history.clearRetainingCapacity();
        }
        fn deinit(self: *Self) void {
            self.clearHistory();
            self.history.deinit(self.a);
            self.cached.deinit(self.a);
        }
        fn turn(self: *Self, request: Request) !void {
            const max_tokens = request.max_tokens orelse self.config.max_tokens;
            if (max_tokens == 0 or max_tokens > self.config.context) return error.InvalidSessionTokenLimit;
            if (request.reset) {
                try self.decoder.reset();
                self.clearHistory();
                self.cached.clearRetainingCapacity();
                self.next = null;
            }
            if (request.prompt.len == 0) {
                if (!request.reset) return error.EmptyPrompt;
                try emit(self.a, .{ .event = "reset" });
                return;
            }
            var messages: std.ArrayList(generation.Message) = .empty;
            defer messages.deinit(self.a);
            try messages.appendSlice(self.a, self.history.items);
            try messages.append(self.a, .{ .role = "user", .content = request.prompt });
            const prompt = if (self.template) |t| try t.applyWithOptions(self.a, messages.items, .{ .enable_thinking = self.config.thinking }) else try self.a.dupe(u8, request.prompt);
            defer self.a.free(prompt);
            const ids = try self.tokenizer.encode(self.a, prompt);
            defer self.a.free(ids);
            if (std.c.getenv("ANTFLY_QWEN_SESSION_TRACE") != null) std.debug.print("session prompt={s}\nids={any}\ncached={any}\n", .{ prompt, ids, self.cached.items });
            if (ids.len == 0) return error.EmptyPrompt;
            if (ids.len > self.config.context - max_tokens) return error.PromptTooLong;
            for (ids) |id| if (id < 0 or id >= 248320) return error.InvalidSessionToken;
            self.executing = true;
            const begin = std.Io.Timestamp.now(self.io, .awake);
            const before_bytes = readBytes(self.decoder);
            if (!exactPrefix(self.cached.items, ids)) {
                try self.decoder.reset();
                self.cached.clearRetainingCapacity();
                self.next = null;
            }
            const reused = self.cached.items.len;
            var at = reused;
            while (at < ids.len) {
                if (self.controller.isLiveHostUnderPressure()) return error.ResourceTemporarilyUnavailable;
                const end = @min(ids.len, at + chunkSize(self.decoder));
                self.next = try step(self.decoder, ids[at..end]);
                try self.cached.appendSlice(self.a, ids[at..end]);
                at = end;
            }
            const prefill_ms = milliseconds(self.io, begin);
            const prefill_bytes = readBytes(self.decoder) - before_bytes;
            var output: std.ArrayList(i32) = .empty;
            defer output.deinit(self.a);
            var printed: usize = 0;
            var finish: []const u8 = "length";
            const decode_start = std.Io.Timestamp.now(self.io, .awake);
            while (output.items.len < max_tokens) {
                if (self.controller.isLiveHostUnderPressure()) return error.ResourceTemporarilyUnavailable;
                const next = self.next orelse return error.MissingSessionLogits;
                if (!self.config.ignore_eos and std.mem.indexOfScalar(u32, self.config.eos, next) != null) {
                    finish = "stop";
                    break;
                }
                try output.append(self.a, @intCast(next));
                const text = try self.tokenizer.decode(self.a, output.items);
                defer self.a.free(text);
                const end = completeUtf8Prefix(text);
                if (end > printed) {
                    try emit(self.a, .{ .event = "delta", .text = text[printed..end] });
                    printed = end;
                }
                if (output.items.len == max_tokens) break;
                if (self.controller.isLiveHostUnderPressure()) return error.ResourceTemporarilyUnavailable;
                if (comptime @hasDecl(@TypeOf(self.decoder.*), "advancePromptLookup")) {
                    if (self.config.prompt_lookup > 0) {
                        var history: [4096]i32 = undefined;
                        @memcpy(history[0..self.cached.items.len], self.cached.items);
                        history[self.cached.items.len] = @intCast(next);
                        const advance = try self.decoder.advancePromptLookup(history[0 .. self.cached.items.len + 1], @min(self.config.prompt_lookup, max_tokens - output.items.len), if (self.config.ignore_eos) &.{} else self.config.eos);
                        self.next = advance.next;
                        try self.cached.append(self.a, @intCast(next));
                        try self.cached.appendSlice(self.a, advance.accepted[0..advance.count]);
                        try output.appendSlice(self.a, advance.accepted[0..advance.count]);
                        const accepted_text = try self.tokenizer.decode(self.a, output.items);
                        defer self.a.free(accepted_text);
                        const accepted_end = completeUtf8Prefix(accepted_text);
                        if (accepted_end > printed) {
                            try emit(self.a, .{ .event = "delta", .text = accepted_text[printed..accepted_end] });
                            printed = accepted_end;
                        }
                        continue;
                    }
                }
                self.next = try step(self.decoder, &.{@intCast(next)});
                try self.cached.append(self.a, @intCast(next));
            }
            const text = try self.tokenizer.decode(self.a, output.items);
            defer self.a.free(text);
            const printable = try validText(self.a, text);
            defer self.a.free(printable);
            if (printable.len > printed) try emit(self.a, .{ .event = "delta", .text = printable[printed..] });
            if (self.template != null) {
                const user = try self.a.dupe(u8, request.prompt);
                errdefer self.a.free(user);
                const assistant = try self.a.dupe(u8, printable);
                errdefer self.a.free(assistant);
                try self.history.ensureUnusedCapacity(self.a, 2);
                self.history.appendAssumeCapacity(.{ .role = "user", .content = user });
                self.history.appendAssumeCapacity(.{ .role = "assistant", .content = assistant });
            }
            try emit(self.a, .{ .event = "done", .text = printable, .generated_token_ids = output.items, .finish_reason = finish, .prompt_tokens = ids.len, .reused_tokens = reused, .prefilled_tokens = ids.len - reused, .position = self.decoder.position, .prefill_ms = prefill_ms, .decode_ms = milliseconds(self.io, decode_start), .first_token_ready_ms = prefill_ms, .prefill_bytes_read = prefill_bytes, .decode_bytes_read = readBytes(self.decoder) - before_bytes - prefill_bytes, .process_memory = platform.process_memory.pressureSnapshot() });
            self.executing = false;
        }
    };
}
test "Qwen35 Qwen4Exp local prefix reuse requires every consumed token and preserves extensions" {
    try std.testing.expect(exactPrefix(&.{ 1, 2 }, &.{ 1, 2, 3 }));
    try std.testing.expect(exactPrefix(&.{ 1, 2 }, &.{ 1, 2 }));
    try std.testing.expect(!exactPrefix(&.{ 1, 2 }, &.{ 1, 3, 2 }));
    try std.testing.expect(!exactPrefix(&.{ 1, 2 }, &.{1}));
    try std.testing.expectEqual(@as(usize, 1), completeUtf8Prefix("a\xe2\x82"));
    try std.testing.expectEqual(@as(usize, 4), completeUtf8Prefix("a\xe2\x82\xac"));
}

test "Qwen35 Qwen4Exp session final text replaces incomplete and invalid UTF8" {
    const text = try validText(std.testing.allocator, "a\xe2\x82");
    defer std.testing.allocator.free(text);
    try std.testing.expect(std.unicode.utf8ValidateSlice(text));
    try std.testing.expectEqualStrings("a\xef\xbf\xbd\xef\xbf\xbd", text);
}
