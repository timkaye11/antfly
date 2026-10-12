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

//! Bounded, cancellable artifact hashing. Never trusts a cached verification flag.
const std = @import("std");
const c_file = @import("c_file.zig");
pub const Identity = struct { bytes: u64, sha256: []const u8 };
pub fn verify(reader: *const c_file.RangeReader, expected: Identity, scratch: []u8, stop: *const std.atomic.Value(bool)) !void {
    if (scratch.len == 0) return error.EmptyVerificationBuffer;
    if (reader.size != expected.bytes) return error.ArtifactSizeMismatch;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var offset: u64 = 0;
    while (offset < expected.bytes) {
        if (stop.load(.acquire)) return error.Cancelled;
        const n: usize = @intCast(@min(scratch.len, expected.bytes - offset));
        reader.read(offset, scratch[0..n]) catch |err| {
            if (stop.load(.acquire)) return error.Cancelled;
            return err;
        };
        hash.update(scratch[0..n]);
        offset += n;
    }
    if (stop.load(.acquire)) return error.Cancelled;
    if (!std.mem.eql(u8, &std.fmt.bytesToHex(hash.finalResult(), .lower), expected.sha256)) return error.ArtifactHashMismatch;
    // Detect growth as well as short reads while hashing this open descriptor.
    if (try reader.currentSize() != expected.bytes) return error.ArtifactSizeMismatch;
}
test "Qwen35 Qwen4Exp artifact verification rejects corruption truncation and cancellation" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "shard", .data = "test" });
    const path = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "shard" });
    defer a.free(path);
    var reader = try c_file.RangeReader.open(a, path, false);
    defer reader.deinit();
    const pin: Identity = .{ .bytes = 4, .sha256 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08" };
    var scratch: [3]u8 = undefined;
    var stop: std.atomic.Value(bool) = .init(false);
    try verify(&reader, pin, &scratch, &stop);
    stop.store(true, .release);
    try std.testing.expectError(error.Cancelled, verify(&reader, pin, &scratch, &stop));
    stop.store(false, .release);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "shard", .data = "text" });
    try std.testing.expectError(error.ArtifactHashMismatch, verify(&reader, pin, &scratch, &stop));
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "shard", .data = "tes" });
    try std.testing.expectError(error.IncompleteRead, verify(&reader, pin, &scratch, &stop));
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "shard", .data = "tests" });
    try std.testing.expectError(error.ArtifactSizeMismatch, verify(&reader, pin, &scratch, &stop));
}
