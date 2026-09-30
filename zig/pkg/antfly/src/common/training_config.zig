// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: ELv2
const std = @import("std");

pub const Config = struct {
    enabled: bool = false,
    state_dir: []const u8,
    toolchain_dir: []const u8,
    models_dir: []const u8,
    input_roots: []const []const u8,
    output_root: []const u8,
    python: []const u8 = "/usr/bin/python3",
    discovery: bool = false,
    ssh_port: u16 = 22,

    pub fn validate(self: Config) !void {
        for ([_][]const u8{ self.state_dir, self.toolchain_dir, self.models_dir, self.output_root, self.python }) |path| {
            if (!std.fs.path.isAbsolute(path) or std.mem.indexOfScalar(u8, path, 0) != null) return error.InvalidConfig;
        }
        if (self.state_dir.len > 85 or self.input_roots.len == 0 or self.input_roots.len > 32 or self.ssh_port == 0) return error.InvalidConfig;
        for (self.input_roots) |path| if (!std.fs.path.isAbsolute(path)) return error.InvalidConfig;
    }
};

test "common config training configuration requires explicit bounded absolute roots" {
    var config: Config = .{ .state_dir = "/tmp/training", .toolchain_dir = "/tmp/tools", .models_dir = "/tmp/models", .input_roots = &.{"/tmp/models"}, .output_root = "/tmp/runs" };
    try config.validate();
    config.input_roots = &.{"relative"};
    try std.testing.expectError(error.InvalidConfig, config.validate());
}
