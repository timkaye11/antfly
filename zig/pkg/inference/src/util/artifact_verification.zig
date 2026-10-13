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

//! Bounded, cancellable artifact hashing. A full pass trusts no cached state.
//! The opt-in stamp mode accepts an earlier full pass on this host only while
//! every file keeps the local identity that pass recorded; it guards against
//! replacement and accidental change, not against deliberate tampering that
//! preserves size, timestamps and sampled content.
const std = @import("std");
const builtin = @import("builtin");
const c_file = @import("c_file.zig");
pub const Identity = struct { bytes: u64, sha256: []const u8 };
pub const Mode = enum { full, stamp };
pub const Outcome = struct {
    mode: Mode = .full,
    /// Every pinned byte was hashed by this process.
    hashed: bool = false,
    /// A stamp written by an earlier full pass was accepted instead.
    stamp_accepted: bool = false,
};
/// One pinned file. Later reads must go through the same reader.
pub const File = struct { reader: *const c_file.RangeReader, path: []const u8, pin: Identity };

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

/// Verify every file in order. `stamp_dir` enables stamp reads (in stamp
/// mode) and stamp writes (after any complete full pass); null disables both.
pub fn verifyFiles(a: std.mem.Allocator, files: []const File, artifact_id: []const u8, mode: Mode, stamp_dir: ?[]const u8, stop: *const std.atomic.Value(bool)) !Outcome {
    const stamp_path: ?[]u8 = if (stamp_dir) |dir| stampPath(a, dir, artifact_id, files) catch null else null;
    defer if (stamp_path) |path| a.free(path);
    if (mode == .stamp) if (stamp_path) |path| {
        if (stampMatches(a, path, artifact_id, files)) {
            std.debug.print("artifact stamp accepted for {d} file(s); pass --verify full to hash every byte\n", .{files.len});
            return .{ .mode = .stamp, .stamp_accepted = true };
        }
    };
    const before = try a.alloc(?c_file.FileIdentity, files.len);
    defer a.free(before);
    const scratch = try a.alloc(u8, 4 * 1024 * 1024);
    defer a.free(scratch);
    for (files, before, 0..) |file, *identity, i| {
        if (stop.load(.acquire)) return error.Cancelled;
        std.debug.print("verifying artifact file {d}/{d} ({d} bytes)\n", .{ i + 1, files.len, file.pin.bytes });
        identity.* = file.reader.identity(a) catch null;
        try verify(file.reader, file.pin, scratch, stop);
    }
    // Record identities only after every digest matched. A failed write just
    // costs the next start another full pass.
    if (stamp_path) |path| writeStamp(a, path, artifact_id, files, before) catch |err| {
        std.log.warn("artifact verification stamp not written: {s}", .{@errorName(err)});
    };
    return .{ .mode = mode, .hashed = true };
}

/// `$ANTFLY_QWEN_VERIFY_STAMP_DIR`, else `~/.antfly/inference/verification`.
pub fn defaultStampDir(a: std.mem.Allocator) ?[]u8 {
    if (comptime !builtin.link_libc) return null;
    if (std.c.getenv("ANTFLY_QWEN_VERIFY_STAMP_DIR")) |value| {
        const dir = std.mem.span(value);
        return if (dir.len == 0) null else a.dupe(u8, dir) catch null;
    }
    const home = std.mem.span(std.c.getenv("HOME") orelse return null);
    if (home.len == 0) return null;
    return std.fs.path.join(a, &.{ home, ".antfly", "inference", "verification" }) catch null;
}

const stamp_version: u32 = 1;
const StampEntry = struct {
    path: []const u8,
    bytes: u64,
    sha256: []const u8,
    inode: u64,
    mtime_seconds: i64,
    mtime_nanoseconds: u32,
    device_major: u32,
    device_minor: u32,
    quick_fingerprint_sha256: []const u8,
};
const Stamp = struct { version: u32, artifact: []const u8, files: []const StampEntry };

fn absolutePath(a: std.mem.Allocator, path: []const u8) ![]u8 {
    if (std.fs.path.isAbsolute(path)) return a.dupe(u8, path);
    if (comptime !builtin.link_libc) return error.StampUnavailable;
    var buffer: [4096]u8 = undefined;
    const cwd = std.c.getcwd(&buffer, buffer.len) orelse return error.StampUnavailable;
    return std.fs.path.join(a, &.{ std.mem.span(@as([*:0]const u8, @ptrCast(cwd))), path });
}

// One stamp per artifact and install location.
fn stampPath(a: std.mem.Allocator, dir: []const u8, artifact_id: []const u8, files: []const File) ![]u8 {
    var key = std.crypto.hash.sha2.Sha256.init(.{});
    key.update(artifact_id);
    for (files) |file| {
        const absolute = try absolutePath(a, file.path);
        defer a.free(absolute);
        key.update("\x00");
        key.update(absolute);
    }
    const name = std.fmt.bytesToHex(key.finalResult(), .lower);
    return std.fmt.allocPrint(a, "{s}/{s}.json", .{ dir, name[0..32] });
}

fn entryMatches(entry: StampEntry, absolute: []const u8, pin: Identity, identity: c_file.FileIdentity) bool {
    return std.mem.eql(u8, entry.path, absolute) and entry.bytes == pin.bytes and identity.size == pin.bytes and
        std.mem.eql(u8, entry.sha256, pin.sha256) and entry.inode == identity.inode and
        entry.mtime_seconds == identity.mtime_seconds and entry.mtime_nanoseconds == identity.mtime_nanoseconds and
        entry.device_major == identity.device_major and entry.device_minor == identity.device_minor and
        std.mem.eql(u8, entry.quick_fingerprint_sha256, &std.fmt.bytesToHex(identity.quick_fingerprint_sha256, .lower));
}

// Any unreadable, malformed or differing stamp is simply not a match.
fn stampMatches(a: std.mem.Allocator, path: []const u8, artifact_id: []const u8, files: []const File) bool {
    const bytes = c_file.readFileMax(a, path, 1024 * 1024) catch return false;
    defer a.free(bytes);
    const parsed = std.json.parseFromSlice(Stamp, a, bytes, .{}) catch return false;
    defer parsed.deinit();
    const stamp = parsed.value;
    if (stamp.version != stamp_version or !std.mem.eql(u8, stamp.artifact, artifact_id) or stamp.files.len != files.len) return false;
    for (stamp.files, files) |entry, file| {
        if (file.reader.size != file.pin.bytes) return false;
        const absolute = absolutePath(a, file.path) catch return false;
        defer a.free(absolute);
        const identity = file.reader.identity(a) catch return false;
        if (!entryMatches(entry, absolute, file.pin, identity)) return false;
    }
    return true;
}

fn writeStamp(a: std.mem.Allocator, path: []const u8, artifact_id: []const u8, files: []const File, before: []const ?c_file.FileIdentity) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const scratch = arena.allocator();
    const entries = try scratch.alloc(StampEntry, files.len);
    for (entries, files, before) |*entry, file, earlier| {
        const identity = try file.reader.identity(scratch);
        // The identity must have held for the whole hash pass.
        if (!std.meta.eql(earlier orelse return error.StampUnavailable, identity)) return error.ArtifactChangedDuringVerification;
        entry.* = .{
            .path = try absolutePath(scratch, file.path),
            .bytes = file.pin.bytes,
            .sha256 = file.pin.sha256,
            .inode = identity.inode,
            .mtime_seconds = identity.mtime_seconds,
            .mtime_nanoseconds = identity.mtime_nanoseconds,
            .device_major = identity.device_major,
            .device_minor = identity.device_minor,
            .quick_fingerprint_sha256 = try scratch.dupe(u8, &std.fmt.bytesToHex(identity.quick_fingerprint_sha256, .lower)),
        };
    }
    const json = try std.json.Stringify.valueAlloc(scratch, Stamp{ .version = stamp_version, .artifact = artifact_id, .files = entries }, .{ .whitespace = .indent_2 });
    try c_file.writeFileAtomic(scratch, path, json);
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

test "Qwen35 Qwen4Exp verification stamp skips hashing only while file identity is unchanged" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "shard", .data = "test" });
    const dir = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer a.free(dir);
    const path = try std.fs.path.join(a, &.{ dir, "shard" });
    defer a.free(path);
    const stamps = try std.fs.path.join(a, &.{ dir, "stamps", "nested" });
    defer a.free(stamps);
    const pin: Identity = .{ .bytes = 4, .sha256 = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08" };
    var stop: std.atomic.Value(bool) = .init(false);
    var reader = try c_file.RangeReader.open(a, path, false);
    defer reader.deinit();
    const files = [_]File{.{ .reader = &reader, .path = path, .pin = pin }};
    // No stamp yet: stamp mode falls back to a full pass and records it.
    const first = try verifyFiles(a, &files, "artifact-a", .stamp, stamps, &stop);
    try std.testing.expect(first.hashed and !first.stamp_accepted);
    const second = try verifyFiles(a, &files, "artifact-a", .stamp, stamps, &stop);
    try std.testing.expect(!second.hashed and second.stamp_accepted);
    try std.testing.expectEqual(Mode.stamp, second.mode);
    // Full mode never consults the stamp; a different artifact never reuses it.
    const full = try verifyFiles(a, &files, "artifact-a", .full, stamps, &stop);
    try std.testing.expect(full.hashed and !full.stamp_accepted);
    try std.testing.expect((try verifyFiles(a, &files, "artifact-b", .stamp, stamps, &stop)).hashed);
    try std.testing.expect((try verifyFiles(a, &files, "artifact-a", .stamp, null, &stop)).hashed);
    // Same-size replacement changes the sampled fingerprint (and usually the
    // inode or mtime): the stamp is rejected and the full pass fails closed.
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "shard", .data = "text" });
    var replaced = try c_file.RangeReader.open(a, path, false);
    defer replaced.deinit();
    const changed = [_]File{.{ .reader = &replaced, .path = path, .pin = pin }};
    try std.testing.expectError(error.ArtifactHashMismatch, verifyFiles(a, &changed, "artifact-a", .stamp, stamps, &stop));
    // A corrupt stamp is ignored rather than trusted or fatal.
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "shard", .data = "test" });
    var restored = try c_file.RangeReader.open(a, path, false);
    defer restored.deinit();
    const again = [_]File{.{ .reader = &restored, .path = path, .pin = pin }};
    try std.testing.expect((try verifyFiles(a, &again, "artifact-a", .stamp, stamps, &stop)).hashed);
    const stamp_file = try stampPath(a, stamps, "artifact-a", &again);
    defer a.free(stamp_file);
    try c_file.writeFileAtomic(a, stamp_file, "{\"version\":1");
    try std.testing.expect((try verifyFiles(a, &again, "artifact-a", .stamp, stamps, &stop)).hashed);
    try std.testing.expect((try verifyFiles(a, &again, "artifact-a", .stamp, stamps, &stop)).stamp_accepted);
}
