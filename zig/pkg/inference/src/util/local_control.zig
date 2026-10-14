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

//! Process-scoped control for single-owner local text CLIs. Handlers only set an atomic.
const std = @import("std");
pub var cancelled: std.atomic.Value(bool) = .init(false);
fn interrupt(_: std.posix.SIG) callconv(.c) void {
    cancelled.store(true, .release);
}
pub fn check() !void {
    if (cancelled.load(.acquire)) return error.Cancelled;
}
/// Stop for the request a session is executing. Unlike `cancelled` it never
/// ends the process; the session owner clears it when the turn ends.
pub var request_cancelled: std.atomic.Value(bool) = .init(false);
/// Session-owned probe for an out-of-band cancel. A single-threaded owner
/// cannot read its input while a token executes, so execution polls here.
pub const RequestPoll = struct { context: *anyopaque, call: *const fn (*anyopaque) void };
pub var request_poll: ?RequestPoll = null;
pub fn checkRequest() !void {
    if (request_poll) |poll| poll.call(poll.context);
    if (request_cancelled.load(.acquire)) return error.RequestCancelled;
}
/// Execution failures after which a local runner keeps its loaded weights,
/// caches and GPU pipeline: reads roll back and submitted work is drained
/// before they surface. A truncated or revoked artifact, a Metal failure and
/// process cancellation are not among them. Shared so the two decoders cannot
/// disagree about which failures a session survives.
pub fn recoverableExecutionError(err: anyerror) bool {
    return switch (err) {
        error.RequestCancelled, error.ResourceTemporarilyUnavailable, error.InputOutput, error.ReadTimedOut, error.ReadFailed => true,
        else => false,
    };
}
pub const Guard = struct {
    previous: [3]std.posix.Sigaction,
    const signals = [_]std.posix.SIG{ .INT, .TERM, .PIPE };
    pub fn init() Guard {
        cancelled.store(false, .release);
        var self: Guard = undefined;
        for (signals, 0..) |sig, i| {
            const action: std.posix.Sigaction = .{ .handler = .{ .handler = if (sig == .PIPE) std.posix.SIG.IGN else interrupt }, .mask = std.posix.sigemptyset(), .flags = 0 };
            std.posix.sigaction(sig, &action, &self.previous[i]);
        }
        return self;
    }
    pub fn deinit(self: *Guard) void {
        for (signals, 0..) |sig, i| std.posix.sigaction(sig, &self.previous[i], null);
        cancelled.store(false, .release);
        request_cancelled.store(false, .release);
        request_poll = null;
    }
};
test "Qwen35 Qwen4Exp local cancellation is observed and reset by scoped ownership" {
    var guard = Guard.init();
    defer guard.deinit();
    try check();
    interrupt(.INT);
    try std.testing.expectError(error.Cancelled, check());
    cancelled.store(false, .release);
    interrupt(.TERM);
    try std.testing.expectError(error.Cancelled, check());
}
test "Qwen35 Qwen4Exp request cancellation is polled and independent of process cancellation" {
    var guard = Guard.init();
    defer guard.deinit();
    try checkRequest();
    const Probe = struct {
        polls: usize = 0,
        fn poll(context: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.polls += 1;
            if (self.polls == 2) request_cancelled.store(true, .release);
        }
    };
    var probe: Probe = .{};
    request_poll = .{ .context = &probe, .call = Probe.poll };
    try checkRequest();
    try std.testing.expectError(error.RequestCancelled, checkRequest());
    // A cancelled request leaves the process-wide flag alone.
    try check();
    request_cancelled.store(false, .release);
    request_poll = null;
    try checkRequest();
    try std.testing.expectEqual(@as(usize, 2), probe.polls);
}
