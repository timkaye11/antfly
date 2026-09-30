// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");

pub fn start(a: std.mem.Allocator, io: std.Io, toolchain: []const u8) !std.process.Child {
    if (!std.fs.path.isAbsolute(toolchain)) return error.InvalidArguments;
    const script = try std.fs.path.join(a, &.{ toolchain, "share/antfly/training/training_discovery.py" });
    defer a.free(script);
    const state = try std.fs.path.join(a, &.{ toolchain, "var/discovery" });
    defer a.free(state);
    return std.process.spawn(io, .{ .argv = &.{ "/usr/bin/python3", script, state }, .stdin = .pipe, .stdout = .ignore, .stderr = .inherit });
}
