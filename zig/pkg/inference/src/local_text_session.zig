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
const control = @import("util/local_control.zig");
const platform = @import("antfly_platform");
pub const Config = struct {
    artifact_hash_verified: bool = false,
    /// A stamp from an earlier full pass on this host was accepted instead.
    artifact_stamp_verified: bool = false,
    verification_mode: []const u8 = "full",
    artifact_id: []const u8 = "unverified",
    process_cap_bytes: usize = 0,
    prompt_lookup: usize = 0,
    context: usize,
    max_tokens: usize,
    raw: bool,
    thinking: ?bool,
    eos: []const u32,
    ignore_eos: bool,
    model: []const u8,
};
const Request = struct { prompt: []const u8 = "", max_tokens: ?usize = null, reset: bool = false, quit: bool = false, cancel: bool = false };
pub fn writeStdout(bytes: []const u8) !void {
    var offset: usize = 0;
    while (offset < bytes.len) {
        const n = std.c.write(1, bytes.ptr + offset, bytes.len - offset);
        if (n <= 0) return error.OutputWriteFailed;
        offset += @intCast(n);
    }
}
/// Bytes that can be emitted now without being revised later: complete
/// sequences, plus invalid bytes, which become U+FFFD whatever follows. Only a
/// trailing sequence that may still complete is held back, so streaming never
/// stalls on bad output.
pub fn settledUtf8Prefix(bytes: []const u8) usize {
    var at: usize = 0;
    while (at < bytes.len) {
        const n = std.unicode.utf8ByteSequenceLength(bytes[at]) catch {
            at += 1;
            continue;
        };
        if (n > bytes.len - at) {
            // Wait only while every byte present could still belong to it.
            for (bytes[at + 1 ..]) |byte| {
                if (byte & 0xc0 != 0x80) break;
            } else return at;
            at += 1;
            continue;
        }
        if (std.unicode.utf8Decode(bytes[at..][0..n])) |_| {
            at += n;
        } else |_| at += 1;
    }
    return at;
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
/// True when `line` is exactly a cancel request. Anything else, including
/// malformed JSON, stays queued for the ordinary request path to report.
pub fn isCancelRequest(a: std.mem.Allocator, line: []const u8) bool {
    const parsed = std.json.parseFromSlice(Request, a, line, .{}) catch return false;
    defer parsed.deinit();
    const r = parsed.value;
    return r.cancel and r.prompt.len == 0 and r.max_tokens == null and !r.reset and !r.quit;
}
/// Buffered request lines. Between turns the owner blocks here; while a turn
/// executes, `pollCancel` reads without blocking so a queued cancel is seen.
pub const Input = struct {
    a: std.mem.Allocator,
    fd: std.posix.fd_t = 0,
    pending: std.ArrayList(u8) = .empty,
    eof: bool = false,
    /// Queued bytes before this offset are complete lines already known not
    /// to be cancel requests.
    inspected: usize = 0,
    pub const max_request_bytes = 1024 * 1024;

    pub fn deinit(self: *Input) void {
        self.pending.deinit(self.a);
    }
    /// Read what the descriptor offers within `timeout_ms`; false if nothing arrived.
    fn fill(self: *Input, timeout_ms: i32) !bool {
        if (self.eof) return false;
        // Signals may be delivered to an I/O worker. Poll in bounded
        // intervals so an idle owner still observes cancellation.
        var descriptors = [_]std.posix.pollfd{.{ .fd = self.fd, .events = std.posix.POLL.IN, .revents = 0 }};
        if (try std.posix.poll(&descriptors, timeout_ms) == 0) return false;
        var buffer: [8192]u8 = undefined;
        const n = std.c.read(self.fd, &buffer, buffer.len);
        if (n < 0) {
            if (std.posix.errno(n) == .INTR) return false;
            return error.InputReadFailed;
        }
        if (n == 0) {
            self.eof = true;
            return false;
        }
        try self.pending.appendSlice(self.a, buffer[0..@intCast(n)]);
        return true;
    }
    fn drop(self: *Input, consumed: usize) void {
        const rest = self.pending.items.len - consumed;
        std.mem.copyForwards(u8, self.pending.items[0..rest], self.pending.items[consumed..]);
        self.pending.shrinkRetainingCapacity(rest);
        self.inspected -|= consumed;
    }
    fn take(self: *Input, len: usize, consumed: usize) ![]u8 {
        const owned = try self.a.dupe(u8, self.pending.items[0..len]);
        self.drop(consumed);
        return owned;
    }
    pub fn line(self: *Input) !?[]u8 {
        while (true) {
            try control.check();
            if (std.mem.indexOfScalar(u8, self.pending.items, '\n')) |end| {
                if (end > max_request_bytes) return error.SessionRequestTooLarge;
                return try self.take(end, end + 1);
            }
            if (self.pending.items.len > max_request_bytes) return error.SessionRequestTooLarge;
            if (self.eof) {
                if (self.pending.items.len == 0) return null;
                return try self.take(self.pending.items.len, self.pending.items.len);
            }
            _ = try self.fill(100);
        }
    }
    /// Consume a queued cancel request, wherever it sits in the queue: it
    /// stops the request executing now, never one still queued. Other lines
    /// keep their order. Never blocks.
    pub fn pollCancel(self: *Input) bool {
        while (self.pending.items.len <= max_request_bytes) {
            if (!(self.fill(0) catch return false)) break;
        }
        var start = self.inspected;
        while (std.mem.indexOfScalarPos(u8, self.pending.items, start, '\n')) |end| {
            if (end - start <= max_request_bytes and isCancelRequest(self.a, self.pending.items[start..end])) {
                const rest = self.pending.items.len - (end + 1);
                std.mem.copyForwards(u8, self.pending.items[start..][0..rest], self.pending.items[end + 1 ..]);
                self.pending.shrinkRetainingCapacity(start + rest);
                return true;
            }
            start = end + 1;
            // Lines already checked are not parsed again on the next poll.
            self.inspected = start;
        }
        return false;
    }
    fn pollHook(context: *anyopaque) void {
        const self: *Input = @ptrCast(@alignCast(context));
        if (self.pollCancel()) control.request_cancelled.store(true, .release);
    }
};
pub fn exactPrefix(cached: []const i32, prompt: []const i32) bool {
    return cached.len <= prompt.len and std.mem.eql(i32, cached, prompt[0..cached.len]);
}
/// `final` marks the call that consumes the last prompt token. Only its
/// prediction is used, so a decoder may skip the output head before it; the
/// result is then null.
fn step(decoder: anytype, ids: []const i32, final: bool) !?u32 {
    if (comptime @hasDecl(@TypeOf(decoder.*), "prefill")) return try decoder.prefill(ids);
    if (decoder.prefill_chunk > 0 and ids.len > 1) {
        const next = try decoder.prefillChunk(ids, final);
        return if (final) next else null;
    }
    var next: u32 = 0;
    for (ids, 0..) |id, i| next = try decoder.prefillToken(@intCast(id), final and i + 1 == ids.len);
    return if (final) next else null;
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
fn cancelRequested() bool {
    control.checkRequest() catch return true;
    return false;
}
/// How a failed turn leaves the session.
pub const Recovery = enum {
    /// The process must exit: state or output can no longer be trusted.
    fatal,
    /// Nothing was consumed; the cached prefix is still exact.
    kept,
    /// Sequence state was cleared. The loaded model and its caches remain.
    reset,
};
pub fn run(a: std.mem.Allocator, io: std.Io, decoder: anytype, controller: *memory.AdmissionController, tokenizer: tokenizer_mod.Tokenizer, template: ?*generation.ChatTemplate, config: Config) !void {
    var session: Session(@TypeOf(decoder)) = .{ .a = a, .io = io, .decoder = decoder, .controller = controller, .tokenizer = tokenizer, .template = template, .config = config };
    defer session.deinit();
    try emit(a, .{ .event = "ready", .hash_verified = config.artifact_hash_verified, .stamp_verified = config.artifact_stamp_verified, .verification_mode = config.verification_mode, .artifact_id = config.artifact_id, .model = config.model, .context_size = config.context, .prefill_chunk = chunkSize(decoder), .process_cap_bytes = config.process_cap_bytes, .protocol = "qwen-local-jsonl-v2" });
    var input: Input = .{ .a = a };
    defer input.deinit();
    control.request_poll = .{ .context = &input, .call = Input.pollHook };
    defer control.request_poll = null;
    while (input.line() catch |err| {
        try emit(a, .{ .event = "error", .error_name = @errorName(err), .fatal = true });
        return err;
    }) |line| {
        defer a.free(line);
        if (std.mem.trim(u8, line, " \r\t").len == 0) continue;
        const request = std.json.parseFromSlice(Request, a, line, .{}) catch |err| {
            try emit(a, .{ .event = "error", .error_name = @errorName(err), .fatal = false });
            continue;
        };
        defer request.deinit();
        if (request.value.quit) break;
        // A cancel that arrives after its turn finished has nothing to stop.
        // It gets no event, so every other request keeps exactly one reply.
        if (request.value.cancel) continue;
        session.turn(request.value) catch |err| {
            const recovery = session.recover(err);
            try emit(a, .{ .event = "error", .error_name = @errorName(err), .fatal = recovery == .fatal, .state_reset = recovery == .reset, .retryable = recovery != .fatal and @TypeOf(decoder.*).recoverable(err) });
            if (recovery == .fatal) return err;
        };
    }
}
fn Session(comptime D: type) type {
    return struct {
        const Self = @This();
        const Decoder = @typeInfo(D).pointer.child;
        a: std.mem.Allocator,
        io: std.Io,
        decoder: D,
        controller: *memory.AdmissionController,
        tokenizer: tokenizer_mod.Tokenizer,
        template: ?*generation.ChatTemplate,
        config: Config,
        history: std.ArrayList(generation.Message) = .empty,
        /// Exactly the tokens the decoder has consumed.
        cached: std.ArrayList(i32) = .empty,
        /// The decoder's prediction after `cached`.
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
        /// Stream the newly settled part of the decoded output. `printed` is a
        /// raw byte offset; invalid bytes reach the client as U+FFFD exactly as
        /// they appear in the final text.
        fn emitSettled(self: *Self, ids: []const i32, printed: *usize) !void {
            const text = try self.tokenizer.decode(self.a, ids);
            defer self.a.free(text);
            const end = settledUtf8Prefix(text);
            if (end <= printed.*) return;
            const piece = try validText(self.a, text[printed.*..end]);
            defer self.a.free(piece);
            try emit(self.a, .{ .event = "delta", .text = piece });
            printed.* = end;
        }
        fn clearSequence(self: *Self) !void {
            try self.decoder.reset();
            self.cached.clearRetainingCapacity();
            self.next = null;
        }
        /// Classify a failed turn. Execution errors used to end the process
        /// unconditionally; a decoder that reports its state intact, or
        /// restorable by reset, now keeps the loaded model alive.
        fn recover(self: *Self, err: anyerror) Recovery {
            const executing = self.executing;
            self.executing = false;
            control.request_cancelled.store(false, .release);
            if (self.decoder.failed) return .fatal;
            // Rejected before execution began: nothing was touched.
            if (!executing) return .kept;
            if (!Decoder.recoverable(err)) return .fatal;
            if (!self.decoder.needs_reset) return .kept;
            self.clearSequence() catch return .fatal;
            return .reset;
        }
        fn turn(self: *Self, request: Request) !void {
            const max_tokens = request.max_tokens orelse self.config.max_tokens;
            if (max_tokens == 0 or max_tokens > self.config.context) return error.InvalidSessionTokenLimit;
            if (request.reset) {
                try self.clearSequence();
                self.clearHistory();
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
            try control.check();
            // A cancel belongs to the turn it interrupts, never to the next one.
            control.request_cancelled.store(false, .release);
            self.executing = true;
            const begin = std.Io.Timestamp.now(self.io, .awake);
            const before_bytes = readBytes(self.decoder);
            // A prefix stopped before its final chunk has no prediction; it is
            // reusable only by a prompt that extends it.
            if (!exactPrefix(self.cached.items, ids) or (self.next == null and self.cached.items.len == ids.len)) try self.clearSequence();
            const reused = self.cached.items.len;
            // Every step either completes with `cached` and `next` updated
            // together or leaves both untouched, so a stopped turn never
            // desynchronises the session from the decoder.
            try self.cached.ensureTotalCapacity(self.a, ids.len + max_tokens);
            var cancelled = false;
            var at = reused;
            while (at < ids.len) {
                try control.check();
                if (cancelRequested()) {
                    cancelled = true;
                    break;
                }
                if (self.controller.isLiveHostUnderPressure()) return error.ResourceTemporarilyUnavailable;
                const end = @min(ids.len, at + chunkSize(self.decoder));
                self.next = step(self.decoder, ids[at..end], end == ids.len) catch |err| switch (err) {
                    error.RequestCancelled => {
                        cancelled = true;
                        break;
                    },
                    else => return err,
                };
                self.cached.appendSliceAssumeCapacity(ids[at..end]);
                at = end;
            }
            const prefilled = at - reused;
            const prefill_ms = milliseconds(self.io, begin);
            const prefill_bytes = readBytes(self.decoder) - before_bytes;
            var output: std.ArrayList(i32) = .empty;
            defer output.deinit(self.a);
            var printed: usize = 0;
            var finish: []const u8 = "length";
            const decode_start = std.Io.Timestamp.now(self.io, .awake);
            while (!cancelled and output.items.len < max_tokens) {
                try control.check();
                if (cancelRequested()) {
                    cancelled = true;
                    break;
                }
                const next = self.next orelse return error.MissingSessionLogits;
                if (!self.config.ignore_eos and std.mem.indexOfScalar(u32, self.config.eos, next) != null) {
                    finish = "stop";
                    break;
                }
                try output.append(self.a, @intCast(next));
                try self.emitSettled(output.items, &printed);
                if (output.items.len == max_tokens) break;
                if (self.controller.isLiveHostUnderPressure()) return error.ResourceTemporarilyUnavailable;
                if (comptime @hasDecl(Decoder, "advancePromptLookup")) {
                    if (self.config.prompt_lookup > 0) {
                        // The pending token is the last history entry; the
                        // consumed prefix is exactly `cached`.
                        self.cached.appendAssumeCapacity(@intCast(next));
                        const advance = self.decoder.advancePromptLookup(self.cached.items, @min(self.config.prompt_lookup, max_tokens - output.items.len), if (self.config.ignore_eos) &.{} else self.config.eos) catch |err| {
                            _ = self.cached.pop();
                            if (err != error.RequestCancelled) return err;
                            cancelled = true;
                            break;
                        };
                        self.next = advance.next;
                        self.cached.appendSliceAssumeCapacity(advance.accepted[0..advance.count]);
                        try output.appendSlice(self.a, advance.accepted[0..advance.count]);
                        try self.emitSettled(output.items, &printed);
                        continue;
                    }
                }
                self.next = step(self.decoder, &.{@intCast(next)}, true) catch |err| switch (err) {
                    error.RequestCancelled => {
                        cancelled = true;
                        break;
                    },
                    else => return err,
                };
                self.cached.appendAssumeCapacity(@intCast(next));
            }
            // A cancel that landed inside a token leaves a partially advanced
            // sequence: clear it, keep the model and what was already emitted.
            var state_reset = false;
            if (cancelled) {
                finish = "cancelled";
                if (self.decoder.needs_reset) {
                    try self.clearSequence();
                    state_reset = true;
                }
            }
            const text = try self.tokenizer.decode(self.a, output.items);
            defer self.a.free(text);
            const printable = try validText(self.a, text);
            defer self.a.free(printable);
            // Deltas so far are validText of settled raw prefixes; the tail
            // completes the same concatenation as `printable`.
            if (printed < text.len) {
                const tail = try validText(self.a, text[printed..]);
                defer self.a.free(tail);
                try emit(self.a, .{ .event = "delta", .text = tail });
            }
            // A turn cancelled before any output is dropped from the conversation.
            if (self.template != null and (!cancelled or output.items.len > 0)) {
                const user = try self.a.dupe(u8, request.prompt);
                errdefer self.a.free(user);
                const assistant = try self.a.dupe(u8, printable);
                errdefer self.a.free(assistant);
                try self.history.ensureUnusedCapacity(self.a, 2);
                self.history.appendAssumeCapacity(.{ .role = "user", .content = user });
                self.history.appendAssumeCapacity(.{ .role = "assistant", .content = assistant });
            }
            try emit(self.a, .{ .event = "done", .text = printable, .generated_token_ids = output.items, .finish_reason = finish, .state_reset = state_reset, .prompt_tokens = ids.len, .reused_tokens = reused, .prefilled_tokens = prefilled, .position = self.decoder.position, .prefill_ms = prefill_ms, .decode_ms = milliseconds(self.io, decode_start), .first_token_ready_ms = prefill_ms, .prefill_bytes_read = prefill_bytes, .decode_bytes_read = readBytes(self.decoder) - before_bytes - prefill_bytes, .process_memory = platform.process_memory.pressureSnapshot() });
            control.request_cancelled.store(false, .release);
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

test "Qwen35 Qwen4Exp session input queues requests and consumes a cancel queued behind them" {
    const a = std.testing.allocator;
    var pipe: [2]std.posix.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), std.c.pipe(&pipe));
    defer _ = std.c.close(pipe[0]);
    var write_open = true;
    defer if (write_open) {
        _ = std.c.close(pipe[1]);
    };
    const send = struct {
        fn all(fd: std.posix.fd_t, bytes: []const u8) !void {
            try std.testing.expectEqual(@as(isize, @intCast(bytes.len)), std.c.write(fd, bytes.ptr, bytes.len));
        }
    }.all;
    var input: Input = .{ .a = a, .fd = pipe[0] };
    defer input.deinit();
    try std.testing.expect(!input.pollCancel());
    // A cancel queued behind the next request stops the executing one: it is
    // consumed exactly once and the queued request keeps its place.
    try send(pipe[1], "{\"prompt\":\"next\"}\n{\"cancel\":true}\n");
    try std.testing.expect(input.pollCancel());
    try std.testing.expect(!input.pollCancel());
    const first = (try input.line()).?;
    defer a.free(first);
    try std.testing.expectEqualStrings("{\"prompt\":\"next\"}", first);
    try std.testing.expect(!input.pollCancel());
    // Several queued requests around a cancel all survive in order.
    try send(pipe[1], "{\"prompt\":\"a\"}\n{\"prompt\":\"b\"}\n{\"cancel\":true}\n{\"prompt\":\"c\"}\n");
    try std.testing.expect(input.pollCancel());
    for ([_][]const u8{ "{\"prompt\":\"a\"}", "{\"prompt\":\"b\"}", "{\"prompt\":\"c\"}" }) |want| {
        const got = (try input.line()).?;
        defer a.free(got);
        try std.testing.expectEqualStrings(want, got);
    }
    // A partial line is not a request yet; ambiguous lines stay queued.
    try send(pipe[1], "{\"cancel\":tr");
    try std.testing.expect(!input.pollCancel());
    try send(pipe[1], "ue,\"reset\":true}\n{\"cancel\":true}");
    try std.testing.expect(!input.pollCancel());
    const mixed = (try input.line()).?;
    defer a.free(mixed);
    try std.testing.expectEqualStrings("{\"cancel\":true,\"reset\":true}", mixed);
    try std.testing.expect(!isCancelRequest(a, mixed));
    try std.testing.expect(isCancelRequest(a, "{\"cancel\":true}"));
    try std.testing.expect(!isCancelRequest(a, "{\"cancel\":false}"));
    try std.testing.expect(!isCancelRequest(a, "{\"cancel\":true"));
    // End of input still delivers the unterminated final line, then stops.
    _ = std.c.close(pipe[1]);
    write_open = false;
    const last = (try input.line()).?;
    defer a.free(last);
    try std.testing.expectEqualStrings("{\"cancel\":true}", last);
    try std.testing.expectEqual(@as(?[]u8, null), try input.line());
}

test "Qwen35 Qwen4Exp session input bounds a request line while probing for cancel" {
    const a = std.testing.allocator;
    var pipe: [2]std.posix.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), std.c.pipe(&pipe));
    defer _ = std.c.close(pipe[0]);
    defer _ = std.c.close(pipe[1]);
    var input: Input = .{ .a = a, .fd = pipe[0] };
    defer input.deinit();
    const chunk: [4096]u8 = @splat('x');
    var sent: usize = 0;
    while (sent <= Input.max_request_bytes) : (sent += chunk.len) {
        try std.testing.expectEqual(@as(isize, chunk.len), std.c.write(pipe[1], &chunk, chunk.len));
        try std.testing.expect(!input.pollCancel());
    }
    try std.testing.expectError(error.SessionRequestTooLarge, input.line());
}

const FakeDecoder = struct {
    failed: bool = false,
    needs_reset: bool = false,
    position: usize = 0,
    resets: usize = 0,
    fn recoverable(err: anyerror) bool {
        return err == error.RequestCancelled or err == error.ResourceTemporarilyUnavailable or err == error.InputOutput;
    }
    fn reset(self: *FakeDecoder) !void {
        if (self.failed) return error.StateInvalid;
        self.needs_reset = false;
        self.position = 0;
        self.resets += 1;
    }
};
test "Qwen35 Qwen4Exp session keeps the model after recoverable turn failures and fails closed otherwise" {
    const a = std.testing.allocator;
    var decoder: FakeDecoder = .{};
    var controller: memory.AdmissionController = .{};
    var session: Session(*FakeDecoder) = .{ .a = a, .io = std.testing.io, .decoder = &decoder, .controller = &controller, .tokenizer = undefined, .template = null, .config = undefined };
    defer session.deinit();
    try session.cached.appendSlice(a, &.{ 1, 2, 3 });
    session.next = 7;
    // Rejected before execution: nothing to recover, whatever the error.
    try std.testing.expectEqual(Recovery.kept, session.recover(error.PromptTooLong));
    // Pressure at a token boundary: the consumed prefix stays reusable.
    session.executing = true;
    try std.testing.expectEqual(Recovery.kept, session.recover(error.ResourceTemporarilyUnavailable));
    try std.testing.expectEqual(@as(usize, 3), session.cached.items.len);
    try std.testing.expectEqual(@as(?u32, 7), session.next);
    try std.testing.expect(!session.executing);
    // A transient read inside a token: sequence cleared, model kept.
    session.executing = true;
    decoder.needs_reset = true;
    try std.testing.expectEqual(Recovery.reset, session.recover(error.InputOutput));
    try std.testing.expectEqual(@as(usize, 0), session.cached.items.len);
    try std.testing.expectEqual(@as(?u32, null), session.next);
    try std.testing.expectEqual(@as(usize, 1), decoder.resets);
    // Unclassified execution errors and a poisoned decoder still end the session.
    session.executing = true;
    try std.testing.expectEqual(Recovery.fatal, session.recover(error.OutputWriteFailed));
    session.executing = true;
    try std.testing.expectEqual(Recovery.fatal, session.recover(error.Cancelled));
    session.executing = true;
    decoder.failed = true;
    try std.testing.expectEqual(Recovery.fatal, session.recover(error.InputOutput));
}

test "Qwen35 Qwen4Exp streaming settles invalid bytes and matches the final text" {
    try std.testing.expectEqual(@as(usize, 1), settledUtf8Prefix("a\xe2\x82"));
    try std.testing.expectEqual(@as(usize, 4), settledUtf8Prefix("a\xe2\x82\xac"));
    // An invalid byte never holds back what follows it.
    try std.testing.expectEqual(@as(usize, 3), settledUtf8Prefix("a\xffb"));
    try std.testing.expectEqual(@as(usize, 3), settledUtf8Prefix("\xe2ab"));
    try std.testing.expectEqual(@as(usize, 5), settledUtf8Prefix("\xc3\x28xyz"));
    // Streaming the settled prefixes token by token reproduces validText.
    const a = std.testing.allocator;
    const full = "ok \xff caf\xc3\xa9 \xe2\x82\xac \xe2(x \xf0\x9f\x8c\x8d";
    var streamed: std.ArrayList(u8) = .empty;
    defer streamed.deinit(a);
    var printed: usize = 0;
    for (1..full.len + 1) |n| {
        const end = settledUtf8Prefix(full[0..n]);
        if (end <= printed) continue;
        const piece = try validText(a, full[printed..end]);
        defer a.free(piece);
        try streamed.appendSlice(a, piece);
        printed = end;
    }
    const tail = try validText(a, full[printed..]);
    defer a.free(tail);
    try streamed.appendSlice(a, tail);
    const expected = try validText(a, full);
    defer a.free(expected);
    try std.testing.expectEqualStrings(expected, streamed.items);
    try std.testing.expect(std.unicode.utf8ValidateSlice(streamed.items));
}
