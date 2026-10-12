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

//! Bound sidecar allocation by pinned size, then verify the exact bytes parsed.
const std = @import("std");
const c_file = @import("c_file.zig");

pub fn read(a: std.mem.Allocator, dir: []const u8, name: []const u8, size: usize, expected: []const u8) ![]u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    var reader = try c_file.RangeReader.open(a, path, true);
    defer reader.deinit();
    if (reader.size != size) return error.SidecarSizeMismatch;
    const bytes = try a.alloc(u8, size);
    errdefer a.free(bytes);
    try reader.read(0, bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), expected)) return error.SidecarHashMismatch;
    return bytes;
}

test "Qwen35 Qwen4Exp sidecar verification rejects changed and oversized files" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer a.free(dir);
    const hash = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08";
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "sidecar", .data = "test" });
    const bytes = try read(a, dir, "sidecar", 4, hash);
    defer a.free(bytes);
    try std.testing.expectEqualStrings("test", bytes);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "sidecar", .data = "text" });
    try std.testing.expectError(error.SidecarHashMismatch, read(a, dir, "sidecar", 4, hash));
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "sidecar", .data = "oversized" });
    try std.testing.expectError(error.SidecarSizeMismatch, read(a, dir, "sidecar", 4, hash));
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "sidecar", .data = "" });
    try std.testing.expectError(error.SidecarSizeMismatch, read(a, dir, "sidecar", 4, hash));
}
