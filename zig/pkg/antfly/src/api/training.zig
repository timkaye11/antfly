// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: ELv2
//! Node-local process owner. The manager's stdin is a lifetime pipe: backend
//! death closes it even if there are no active HTTP requests or UI clients.
const std = @import("std");
const sync = @import("antfly_platform").sync;
const Config = @import("../common/training_config.zig").Config;

pub const Manager = struct {
    mutex: std.atomic.Mutex = .unlocked,
    child: ?std.process.Child = null,
    closing: bool = false,
    requests: std.atomic.Value(u8) = .init(0),

    pub fn ensure(self: *Manager, a: std.mem.Allocator, io: std.Io, config: Config) !void {
        sync.lockYielding(&self.mutex);
        defer self.mutex.unlock();
        if (self.closing) return error.TrainingUnavailable;
        if (self.child != null) return;
        try config.validate();
        const script = try std.fs.path.join(a, &.{ config.toolchain_dir, "share/antfly/training/training_service.py" });
        defer a.free(script);
        const encoded = try std.json.Stringify.valueAlloc(a, config, .{});
        defer a.free(encoded);
        self.child = try std.process.spawn(io, .{
            .argv = &.{ config.python, script, "serve", encoded },
            .stdin = .pipe,
            .stdout = .ignore,
            .stderr = .inherit,
        });
    }

    pub fn request(self: *Manager, a: std.mem.Allocator, io: std.Io, config: Config, method: []const u8, path: []const u8, query: []const u8, body: []const u8) ![]u8 {
        var count = self.requests.load(.monotonic);
        while (true) {
            if (count >= 8) return error.TrainingBusy;
            count = self.requests.cmpxchgWeak(count, count + 1, .monotonic, .monotonic) orelse break;
        }
        defer _ = self.requests.fetchSub(1, .monotonic);
        if (body.len > 64 * 1024 or query.len > 1024 or path.len > 256) return error.TrainingRequestTooLarge;
        var parsed = try std.json.parseFromSlice(std.json.Value, a, if (body.len == 0) "{}" else body, .{});
        defer parsed.deinit();
        if (parsed.value != .object) return error.InvalidTrainingRequest;
        try self.ensure(a, io, config);
        const script = try std.fs.path.join(a, &.{ config.toolchain_dir, "share/antfly/training/training_service.py" });
        defer a.free(script);
        const encoded = try std.json.Stringify.valueAlloc(a, .{ .method = method, .path = path, .query = query, .body = parsed.value }, .{});
        defer a.free(encoded);
        const result = try std.process.run(a, io, .{
            .argv = &.{ config.python, script, "request", config.state_dir, encoded },
            .stdout_limit = .limited(4 * 1024 * 1024),
            .stderr_limit = .limited(8192),
            .timeout = .{ .duration = .{ .raw = .fromSeconds(20), .clock = .awake } },
        });
        defer a.free(result.stderr);
        errdefer a.free(result.stdout);
        if (result.term != .exited or result.term.exited != 0) return error.TrainingUnavailable;
        return result.stdout;
    }

    pub fn deinit(self: *Manager, io: std.Io) void {
        sync.lockYielding(&self.mutex);
        defer self.mutex.unlock();
        self.closing = true;
        if (self.child) |*child| {
            if (child.stdin) |input| input.close(io);
            child.stdin = null;
            _ = child.wait(io) catch {
                child.kill(io);
            };
        }
        self.child = null;
    }
};

/// Socket loopback alone does not prevent DNS rebinding from a browser.
pub fn localBrowserAllowed(host_header: ?[]const u8, origin: ?[]const u8) bool {
    const host = host_header orelse return false;
    const hostname = if (std.mem.startsWith(u8, host, "[")) blk: {
        const end = std.mem.indexOfScalar(u8, host, ']') orelse return false;
        break :blk host[0 .. end + 1];
    } else host[0 .. std.mem.indexOfScalar(u8, host, ':') orelse host.len];
    if (!std.ascii.eqlIgnoreCase(hostname, "localhost") and !std.mem.eql(u8, hostname, "127.0.0.1") and !std.mem.eql(u8, hostname, "[::1]")) return false;
    if (host.len > hostname.len) {
        if (host[hostname.len] != ':') return false;
        _ = std.fmt.parseUnsigned(u16, host[hostname.len + 1 ..], 10) catch return false;
    }
    if (origin) |value| {
        const offset: usize = if (std.mem.startsWith(u8, value, "http://")) 7 else if (std.mem.startsWith(u8, value, "https://")) 8 else return false;
        return std.mem.eql(u8, value[offset..], host);
    }
    return true;
}

test "httpx training local browser rejects foreign origins and rebinding hosts" {
    try std.testing.expect(localBrowserAllowed("localhost:8080", "http://localhost:8080"));
    try std.testing.expect(localBrowserAllowed("127.0.0.1", null));
    try std.testing.expect(localBrowserAllowed("[::1]:8080", "http://[::1]:8080"));
    try std.testing.expect(!localBrowserAllowed("evil.example:8080", "http://evil.example:8080"));
    try std.testing.expect(!localBrowserAllowed("localhost:8080", "https://evil.example"));
    try std.testing.expect(!localBrowserAllowed("localhost.attacker.example", null));
    try std.testing.expect(!localBrowserAllowed("[::1]attacker", null));
}
