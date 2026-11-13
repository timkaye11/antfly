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

//! Minimal runtime binding for a parent-audited full release receipt. The exact
//! receipt SHA remains the source-build identity. Runtime parses only immutable
//! scope/status, executable identity and v2/v3 compiler profile; Python independently
//! audits the bound archive, toolchain, command, resource samples, logs and tests.
const std = @import("std");
const builtin = @import("builtin");

const Binary = struct { path: []const u8, sha256: []const u8, bytes: u64 };
const Receipt = struct {
    schema: []const u8,
    evidence_scope: []const u8,
    release_eligible: bool,
    serving_qualified: bool,
    source_verified_after_build: bool,
    binary: Binary,
    compiler_profile: ?std.json.Value = null,
};

const CompilerProfile = struct { strip_debug_info: bool, have_error_return_tracing: bool };

const Pin = struct { path: []const u8, sha256: []const u8 };
const FullCliBinary = struct { path: []const u8, sha256: []const u8, bytes: u64 };
const StagedFullCliReceipt = struct {
    schema: []const u8,
    build_kind: []const u8,
    diagnostic_worker: bool,
    source_freeze: Pin,
    source_state_before: std.json.Value,
    source_state_after: std.json.Value,
    source_stable: bool,
    full_context_release_proof: Pin,
    resource: struct { path: []const u8, sha256: []const u8, peak_group_rss_mib: f64 },
    binary: FullCliBinary,
    builder: Pin,
    command: []const []const u8,
    fresh_standalone_install_required: bool,
    release_eligible: bool,
    qualification_eligible: bool,
    serving_qualified: bool,
    registry_enrolled: bool,
};

pub const Binding = struct { binary_sha256: [32]u8, source_freeze_sha256: ?[32]u8 = null };

pub fn parse(allocator: std.mem.Allocator, bytes: []const u8) !Binding {
    return parseWithCompilerProfile(allocator, bytes, .{
        .strip_debug_info = builtin.strip_debug_info,
        .have_error_return_tracing = builtin.have_error_return_tracing,
    });
}

/// Parse the parent-audited standalone full CLI receipt used exclusively by a
/// staged operator candidate. This is deliberately separate from `parse`:
/// enrolled producers and trusted consumers still require the release-eligible
/// context-release receipt contract.
pub fn parseStagedFullCli(allocator: std.mem.Allocator, bytes: []const u8) !Binding {
    return parseStagedFullCliWithCompiler(allocator, bytes, .{
        .strip_debug_info = builtin.strip_debug_info,
        .have_error_return_tracing = builtin.have_error_return_tracing,
    });
}

/// Parse the V4 full-context proof only when it is nested under an already
/// authenticated staged-full-CLI receipt. Ordinary release binding remains
/// deliberately limited to its existing V1-V3 contract.
pub fn parseStagedFullContextProof(allocator: std.mem.Allocator, bytes: []const u8) !Binding {
    return parseStagedFullContextProofWithCompiler(allocator, bytes, .{
        .strip_debug_info = builtin.strip_debug_info,
        .have_error_return_tracing = builtin.have_error_return_tracing,
    });
}

fn parseStagedFullContextProofWithCompiler(allocator: std.mem.Allocator, bytes: []const u8, compiler: CompilerProfile) !Binding {
    var parsed = std.json.parseFromSlice(Receipt, allocator, bytes, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = true,
    }) catch return error.InvalidContextReleaseBuildReceipt;
    defer parsed.deinit();
    const value = parsed.value;
    if (!std.mem.eql(u8, value.schema, "antfly.context-release-build.v4") or
        !std.mem.eql(u8, value.evidence_scope, "full_context_release_build") or
        !value.release_eligible or value.serving_qualified or !value.source_verified_after_build or
        value.binary.bytes == 0 or !validRelative(value.binary.path))
        return error.InvalidContextReleaseBuildReceipt;
    const profile = value.compiler_profile orelse return error.InvalidContextReleaseBuildReceipt;
    if (profile != .object or profile.object.count() != 2)
        return error.InvalidContextReleaseBuildReceipt;
    const strip = profile.object.get("strip_debug_info") orelse return error.InvalidContextReleaseBuildReceipt;
    const tracing = profile.object.get("have_error_return_tracing") orelse return error.InvalidContextReleaseBuildReceipt;
    if (strip != .bool or tracing != .bool or !strip.bool or tracing.bool or
        compiler.strip_debug_info != strip.bool or compiler.have_error_return_tracing != tracing.bool)
        return error.InvalidContextReleaseBuildReceipt;
    return .{ .binary_sha256 = digest32(value.binary.sha256) catch return error.InvalidContextReleaseBuildReceipt };
}

fn parseStagedFullCliWithCompiler(allocator: std.mem.Allocator, bytes: []const u8, compiler: CompilerProfile) !Binding {
    var parsed = std.json.parseFromSlice(StagedFullCliReceipt, allocator, bytes, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = true,
    }) catch return error.InvalidStagedFullCliBuildReceipt;
    defer parsed.deinit();
    const value = parsed.value;
    if (!std.mem.eql(u8, value.schema, "antfly.full-inference-cli-build.v1") or
        !std.mem.eql(u8, value.build_kind, "full_inference_cli") or value.diagnostic_worker or
        !value.source_stable or !jsonEqual(value.source_state_before, value.source_state_after) or
        !value.fresh_standalone_install_required or value.release_eligible or value.qualification_eligible or
        value.serving_qualified or value.registry_enrolled or value.binary.bytes == 0 or
        !std.fs.path.isAbsolute(value.binary.path) or !validPin(value.source_freeze) or
        !validPin(value.full_context_release_proof) or !validPin(value.builder) or
        value.resource.peak_group_rss_mib <= 0 or !validAbsolutePin(value.resource.path, value.resource.sha256) or
        !compiler.strip_debug_info or compiler.have_error_return_tracing or !validFullCliCommand(value.command))
        return error.InvalidStagedFullCliBuildReceipt;
    return .{
        .binary_sha256 = digest32(value.binary.sha256) catch return error.InvalidStagedFullCliBuildReceipt,
        .source_freeze_sha256 = digest32(value.source_freeze.sha256) catch return error.InvalidStagedFullCliBuildReceipt,
    };
}

fn validPin(pin: Pin) bool {
    return validAbsolutePin(pin.path, pin.sha256);
}

fn validAbsolutePin(path: []const u8, digest: []const u8) bool {
    if (!std.fs.path.isAbsolute(path)) return false;
    _ = digest32(digest) catch return false;
    return true;
}

fn validFullCliCommand(command: []const []const u8) bool {
    // The release builder emits either the original exact argv or the same
    // argv with one compiler-driver `--verbose` immediately before
    // `--verbose-link`. Only authority-owned executable, helper, Zig, cache
    // and prefix paths vary.
    if (command.len != 17 and command.len != 18) return false;
    const suffix: usize = if (command.len == 18) 13 else 12;
    if (command.len == 18 and !std.mem.eql(u8, command[12], "--verbose")) return false;
    return std.fs.path.isAbsolute(command[0]) and std.fs.path.isAbsolute(command[1]) and
        std.mem.eql(u8, command[2], "--zig") and std.fs.path.isAbsolute(command[3]) and
        std.mem.eql(u8, command[4], "build") and std.mem.eql(u8, command[5], "install") and
        std.mem.eql(u8, command[6], "-j1") and std.mem.eql(u8, command[7], "-Dmetal=true") and
        std.mem.eql(u8, command[8], "-Dcuda=false") and std.mem.eql(u8, command[9], "-Doptimize=ReleaseFast") and
        std.mem.eql(u8, command[10], "-Dstrip=true") and
        std.mem.eql(u8, command[11], "-Dinference-compile-memory-mb=16384") and
        std.mem.eql(u8, command[suffix], "--verbose-link") and std.mem.eql(u8, command[suffix + 1], "--cache-dir") and
        std.fs.path.isAbsolute(command[suffix + 2]) and std.mem.eql(u8, command[suffix + 3], "--prefix") and
        std.fs.path.isAbsolute(command[suffix + 4]);
}

fn jsonEqual(a: std.json.Value, b: std.json.Value) bool {
    if (std.meta.activeTag(a) != std.meta.activeTag(b)) return false;
    return switch (a) {
        .null => true,
        .bool => |v| v == b.bool,
        .integer => |v| v == b.integer,
        .float => |v| v == b.float,
        .number_string => |v| std.mem.eql(u8, v, b.number_string),
        .string => |v| std.mem.eql(u8, v, b.string),
        .array => |v| blk: {
            if (v.items.len != b.array.items.len) break :blk false;
            for (v.items, b.array.items) |x, y| if (!jsonEqual(x, y)) break :blk false;
            break :blk true;
        },
        .object => |v| blk: {
            if (v.count() != b.object.count()) break :blk false;
            var it = v.iterator();
            while (it.next()) |entry| {
                const other = b.object.get(entry.key_ptr.*) orelse break :blk false;
                if (!jsonEqual(entry.value_ptr.*, other)) break :blk false;
            }
            break :blk true;
        },
    };
}

fn parseWithCompilerProfile(allocator: std.mem.Allocator, bytes: []const u8, compiler: CompilerProfile) !Binding {
    var parsed = std.json.parseFromSlice(Receipt, allocator, bytes, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = true,
    }) catch return error.InvalidContextReleaseBuildReceipt;
    defer parsed.deinit();
    const value = parsed.value;
    const v1 = std.mem.eql(u8, value.schema, "antfly.context-release-build.v1");
    const v2 = std.mem.eql(u8, value.schema, "antfly.context-release-build.v2");
    const v3 = std.mem.eql(u8, value.schema, "antfly.context-release-build.v3");
    if ((!v1 and !v2 and !v3) or
        !std.mem.eql(u8, value.evidence_scope, "full_context_release_build") or
        !value.release_eligible or value.serving_qualified or !value.source_verified_after_build or
        value.binary.bytes == 0 or !validRelative(value.binary.path))
        return error.InvalidContextReleaseBuildReceipt;
    if (v2 or v3) {
        const profile = value.compiler_profile orelse return error.InvalidContextReleaseBuildReceipt;
        if (profile != .object or profile.object.count() != 2)
            return error.InvalidContextReleaseBuildReceipt;
        const strip = profile.object.get("strip_debug_info") orelse return error.InvalidContextReleaseBuildReceipt;
        const tracing = profile.object.get("have_error_return_tracing") orelse return error.InvalidContextReleaseBuildReceipt;
        // V2/V3 are canonical stripped profiles. V3 removes the historical
        // build-option shape that Python audits; the runtime compiler contract
        // is intentionally identical. Require the actual executable's settings
        // as well as the parent-audited receipt's booleans.
        if (strip != .bool or tracing != .bool or !strip.bool or tracing.bool or
            strip.bool != compiler.strip_debug_info or tracing.bool != compiler.have_error_return_tracing)
            return error.InvalidContextReleaseBuildReceipt;
    }
    return .{ .binary_sha256 = digest32(value.binary.sha256) catch
        return error.InvalidContextReleaseBuildReceipt };
}

fn validRelative(value: []const u8) bool {
    if (value.len == 0 or value.len > 1024 or value[0] == '/') return false;
    var parts = std.mem.splitScalar(u8, value, '/');
    while (parts.next()) |part| if (part.len == 0 or std.mem.eql(u8, part, ".") or std.mem.eql(u8, part, "..")) return false;
    return true;
}

fn digest32(value: []const u8) ![32]u8 {
    if (value.len != 64) return error.InvalidDigest;
    var result: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&result, value) catch return error.InvalidDigest;
    return result;
}

test "context full release runtime binding rejects historical and scoped receipts" {
    const good =
        \\{"schema":"antfly.context-release-build.v1","evidence_scope":"full_context_release_build","release_eligible":true,"serving_qualified":false,"source_verified_after_build":true,"binary":{"path":"context-tests","sha256":"1111111111111111111111111111111111111111111111111111111111111111","bytes":7},"archive":{"ignored":true}}
    ;
    const binding = try parse(std.testing.allocator, good);
    try std.testing.expectEqual(@as(u8, 0x11), binding.binary_sha256[0]);
    const old =
        \\{"binary_sha256":"1111111111111111111111111111111111111111111111111111111111111111","source_verified_after_build":true}
    ;
    try std.testing.expectError(error.InvalidContextReleaseBuildReceipt, parse(std.testing.allocator, old));
    const scoped =
        \\{"schema":"antfly.context-release-build.v1","evidence_scope":"numerical_capture_and_metal_lifecycle_only","release_eligible":false,"serving_qualified":false,"source_verified_after_build":true,"binary":{"path":"context-tests","sha256":"1111111111111111111111111111111111111111111111111111111111111111","bytes":7}}
    ;
    try std.testing.expectError(error.InvalidContextReleaseBuildReceipt, parse(std.testing.allocator, scoped));
}

const v2_test_prefix =
    \\{"schema":"antfly.context-release-build.v2","evidence_scope":"full_context_release_build","release_eligible":true,"serving_qualified":false,"source_verified_after_build":true,"binary":{"path":"context-tests","sha256":"1111111111111111111111111111111111111111111111111111111111111111","bytes":7}
;
const v3_test_prefix =
    \\{"schema":"antfly.context-release-build.v3","evidence_scope":"full_context_release_build","release_eligible":true,"serving_qualified":false,"source_verified_after_build":true,"binary":{"path":"context-tests","sha256":"1111111111111111111111111111111111111111111111111111111111111111","bytes":7}
;
const v2_test_profile =
    \\,"compiler_profile":{"strip_debug_info":true,"have_error_return_tracing":false}
;

test "context full release runtime binding v2 matches actual compiler profile" {
    const receipt = v2_test_prefix ++ v2_test_profile ++ "}";
    const binding = try parseWithCompilerProfile(std.testing.allocator, receipt, .{
        .strip_debug_info = true,
        .have_error_return_tracing = false,
    });
    try std.testing.expectEqual(@as(u8, 0x11), binding.binary_sha256[0]);
    if (builtin.strip_debug_info and !builtin.have_error_return_tracing) {
        _ = try parse(std.testing.allocator, receipt);
    } else {
        try std.testing.expectError(error.InvalidContextReleaseBuildReceipt, parse(std.testing.allocator, receipt));
    }
    for ([_]CompilerProfile{
        .{ .strip_debug_info = false, .have_error_return_tracing = false },
        .{ .strip_debug_info = false, .have_error_return_tracing = true },
        .{ .strip_debug_info = true, .have_error_return_tracing = true },
    }) |compiler| {
        try std.testing.expectError(error.InvalidContextReleaseBuildReceipt, parseWithCompilerProfile(std.testing.allocator, receipt, compiler));
    }
}

test "context full release runtime binding v3 matches stripped compiler profile" {
    const receipt = v3_test_prefix ++ v2_test_profile ++ "}";
    const binding = try parseWithCompilerProfile(std.testing.allocator, receipt, .{
        .strip_debug_info = true,
        .have_error_return_tracing = false,
    });
    try std.testing.expectEqual(@as(u8, 0x11), binding.binary_sha256[0]);
    if (builtin.strip_debug_info and !builtin.have_error_return_tracing) {
        _ = try parse(std.testing.allocator, receipt);
    } else {
        try std.testing.expectError(error.InvalidContextReleaseBuildReceipt, parse(std.testing.allocator, receipt));
    }
    for ([_]CompilerProfile{
        .{ .strip_debug_info = false, .have_error_return_tracing = false },
        .{ .strip_debug_info = false, .have_error_return_tracing = true },
        .{ .strip_debug_info = true, .have_error_return_tracing = true },
    }) |compiler| {
        try std.testing.expectError(error.InvalidContextReleaseBuildReceipt, parseWithCompilerProfile(std.testing.allocator, receipt, compiler));
    }
    try std.testing.expectError(error.InvalidContextReleaseBuildReceipt, parseWithCompilerProfile(std.testing.allocator, v3_test_prefix ++ "}", .{
        .strip_debug_info = true,
        .have_error_return_tracing = false,
    }));
}

test "context full release runtime binding rejects unknown future schema" {
    const receipt =
        \\{"schema":"antfly.context-release-build.v4","evidence_scope":"full_context_release_build","release_eligible":true,"serving_qualified":false,"source_verified_after_build":true,"binary":{"path":"context-tests","sha256":"1111111111111111111111111111111111111111111111111111111111111111","bytes":7},"compiler_profile":{"strip_debug_info":true,"have_error_return_tracing":false}}
    ;
    try std.testing.expectError(error.InvalidContextReleaseBuildReceipt, parseWithCompilerProfile(std.testing.allocator, receipt, .{
        .strip_debug_info = true,
        .have_error_return_tracing = false,
    }));
}

const staged_context_v4_good =
    \\{"schema":"antfly.context-release-build.v4","evidence_scope":"full_context_release_build","release_eligible":true,"serving_qualified":false,"source_verified_after_build":true,"binary":{"path":"context-tests","sha256":"1111111111111111111111111111111111111111111111111111111111111111","bytes":7},"compiler_profile":{"strip_debug_info":true,"have_error_return_tracing":false}}
;

test "staged full CLI nested context proof accepts canonical V4 only" {
    const compiler: CompilerProfile = .{ .strip_debug_info = true, .have_error_return_tracing = false };
    const binding = try parseStagedFullContextProofWithCompiler(std.testing.allocator, staged_context_v4_good, compiler);
    try std.testing.expectEqual(@as(u8, 0x11), binding.binary_sha256[0]);
    try std.testing.expectError(error.InvalidContextReleaseBuildReceipt, parseWithCompilerProfile(std.testing.allocator, staged_context_v4_good, compiler));
    const v5 = try std.mem.replaceOwned(u8, std.testing.allocator, staged_context_v4_good, "context-release-build.v4", "context-release-build.v5");
    defer std.testing.allocator.free(v5);
    try std.testing.expectError(error.InvalidContextReleaseBuildReceipt, parseStagedFullContextProofWithCompiler(std.testing.allocator, v5, compiler));
    const absent_binary =
        \\{"schema":"antfly.context-release-build.v4","evidence_scope":"full_context_release_build","release_eligible":true,"serving_qualified":false,"source_verified_after_build":true,"compiler_profile":{"strip_debug_info":true,"have_error_return_tracing":false}}
    ;
    try std.testing.expectError(error.InvalidContextReleaseBuildReceipt, parseStagedFullContextProofWithCompiler(std.testing.allocator, absent_binary, compiler));
    inline for (.{
        .{ "\"compiler_profile\":{\"strip_debug_info\":true,\"have_error_return_tracing\":false}", "\"compiler_profile\":null" },
        .{ "\"release_eligible\":true", "\"release_eligible\":false" },
        .{ "\"path\":\"context-tests\"", "\"path\":\"\"" },
    }) |replacement| {
        const invalid = try std.mem.replaceOwned(u8, std.testing.allocator, staged_context_v4_good, replacement[0], replacement[1]);
        defer std.testing.allocator.free(invalid);
        try std.testing.expectError(error.InvalidContextReleaseBuildReceipt, parseStagedFullContextProofWithCompiler(std.testing.allocator, invalid, compiler));
    }
}

test "context full release runtime binding v2 and v3 reject absent malformed and noncanonical profile" {
    const invalid_profiles = [_][]const u8{
        "",
        \\,"compiler_profile":null
        ,
        \\,"compiler_profile":[]
        ,
        \\,"compiler_profile":{"strip_debug_info":true}
        ,
        \\,"compiler_profile":{"strip_debug_info":1,"have_error_return_tracing":false}
        ,
        \\,"compiler_profile":{"strip_debug_info":true,"have_error_return_tracing":0}
        ,
        \\,"compiler_profile":{"strip_debug_info":"true","have_error_return_tracing":false}
        ,
        \\,"compiler_profile":{"strip_debug_info":false,"have_error_return_tracing":false}
        ,
        \\,"compiler_profile":{"strip_debug_info":true,"have_error_return_tracing":true}
        ,
        \\,"compiler_profile":{"strip_debug_info":true,"have_error_return_tracing":false,"extra":false}
        ,
    };
    for ([_][]const u8{ v2_test_prefix, v3_test_prefix }) |prefix| {
        for (invalid_profiles) |profile| {
            const receipt = try std.fmt.allocPrint(std.testing.allocator, "{s}{s}}}", .{ prefix, profile });
            defer std.testing.allocator.free(receipt);
            try std.testing.expectError(error.InvalidContextReleaseBuildReceipt, parseWithCompilerProfile(std.testing.allocator, receipt, .{
                .strip_debug_info = true,
                .have_error_return_tracing = false,
            }));
        }
    }
}

test "context full release runtime binding v1 preserves unknown profile fields" {
    const receipt =
        \\{"schema":"antfly.context-release-build.v1","evidence_scope":"full_context_release_build","release_eligible":true,"serving_qualified":false,"source_verified_after_build":true,"binary":{"path":"context-tests","sha256":"1111111111111111111111111111111111111111111111111111111111111111","bytes":7},"compiler_profile":{"future_field":42}}
    ;
    _ = try parse(std.testing.allocator, receipt);
}

const staged_full_cli_good =
    \\{"schema":"antfly.full-inference-cli-build.v1","build_kind":"full_inference_cli","diagnostic_worker":false,"source_freeze":{"path":"/authority/freeze.json","sha256":"1111111111111111111111111111111111111111111111111111111111111111"},"source_state_before":{"head":"abc","dirty":[]},"source_state_after":{"dirty":[],"head":"abc"},"source_stable":true,"full_context_release_proof":{"path":"/authority/context-release-receipt.json","sha256":"2222222222222222222222222222222222222222222222222222222222222222"},"resource":{"path":"/authority/resource.json","sha256":"3333333333333333333333333333333333333333333333333333333333333333","peak_group_rss_mib":512.0},"binary":{"path":"/authority/antfly-inference-full","sha256":"4444444444444444444444444444444444444444444444444444444444444444","bytes":99},"builder":{"path":"/authority/build.py","sha256":"5555555555555555555555555555555555555555555555555555555555555555"},"command":["/usr/bin/python3","/authority/bounded.py","--zig","/authority/zig","build","install","-j1","-Dmetal=true","-Dcuda=false","-Doptimize=ReleaseFast","-Dstrip=true","-Dinference-compile-memory-mb=16384","--verbose-link","--cache-dir","/authority/cache","--prefix","/authority/out"],"fresh_standalone_install_required":true,"release_eligible":false,"qualification_eligible":false,"serving_qualified":false,"registry_enrolled":false}
;

test "staged full CLI binding accepts only exact stripped standalone authority" {
    const binding = try parseStagedFullCliWithCompiler(std.testing.allocator, staged_full_cli_good, .{
        .strip_debug_info = true,
        .have_error_return_tracing = false,
    });
    try std.testing.expectEqual(@as(u8, 0x44), binding.binary_sha256[0]);
    try std.testing.expectError(error.InvalidContextReleaseBuildReceipt, parseWithCompilerProfile(std.testing.allocator, staged_full_cli_good, .{
        .strip_debug_info = true,
        .have_error_return_tracing = false,
    }));
    try std.testing.expectError(error.InvalidStagedFullCliBuildReceipt, parseStagedFullCliWithCompiler(std.testing.allocator, staged_full_cli_good, .{
        .strip_debug_info = false,
        .have_error_return_tracing = false,
    }));
}

test "context staged full CLI binding accepts exact verbose compiler argv receipt" {
    const verbose_receipt = try std.mem.replaceOwned(
        u8,
        std.testing.allocator,
        staged_full_cli_good,
        "\"--verbose-link\"",
        "\"--verbose\",\"--verbose-link\"",
    );
    defer std.testing.allocator.free(verbose_receipt);
    const binding = try parseStagedFullCliWithCompiler(std.testing.allocator, verbose_receipt, .{
        .strip_debug_info = true,
        .have_error_return_tracing = false,
    });
    try std.testing.expectEqual(@as(u8, 0x44), binding.binary_sha256[0]);
}

test "context full CLI command admits only exact 17 or verbose 18 argument shapes" {
    const original = [_][]const u8{
        "/usr/bin/python3", "/authority/bounded.py",  "--zig",            "/authority/zig",
        "build",            "install",                "-j1",              "-Dmetal=true",
        "-Dcuda=false",     "-Doptimize=ReleaseFast", "-Dstrip=true",     "-Dinference-compile-memory-mb=16384",
        "--verbose-link",   "--cache-dir",            "/authority/cache", "--prefix",
        "/authority/out",
    };
    const verbose = [_][]const u8{
        "/usr/bin/python3", "/authority/bounded.py",  "--zig",        "/authority/zig",
        "build",            "install",                "-j1",          "-Dmetal=true",
        "-Dcuda=false",     "-Doptimize=ReleaseFast", "-Dstrip=true", "-Dinference-compile-memory-mb=16384",
        "--verbose",        "--verbose-link",         "--cache-dir",  "/authority/cache",
        "--prefix",         "/authority/out",
    };
    try std.testing.expect(validFullCliCommand(&original));
    try std.testing.expect(validFullCliCommand(&verbose));

    const unknown_extra = [_][]const u8{
        "/usr/bin/python3", "/authority/bounded.py",  "--zig",        "/authority/zig",
        "build",            "install",                "-j1",          "-Dmetal=true",
        "-Dcuda=false",     "-Doptimize=ReleaseFast", "-Dstrip=true", "-Dinference-compile-memory-mb=16384",
        "--unknown",        "--verbose-link",         "--cache-dir",  "/authority/cache",
        "--prefix",         "/authority/out",
    };
    const duplicated = [_][]const u8{
        "/usr/bin/python3", "/authority/bounded.py",  "--zig",        "/authority/zig",
        "build",            "install",                "-j1",          "-Dmetal=true",
        "-Dcuda=false",     "-Doptimize=ReleaseFast", "-Dstrip=true", "-Dinference-compile-memory-mb=16384",
        "--verbose-link",   "--verbose-link",         "--cache-dir",  "/authority/cache",
        "--prefix",         "/authority/out",
    };
    const reordered = [_][]const u8{
        "/usr/bin/python3", "/authority/bounded.py",  "--zig",        "/authority/zig",
        "build",            "install",                "-j1",          "-Dmetal=true",
        "-Dcuda=false",     "-Doptimize=ReleaseFast", "-Dstrip=true", "-Dinference-compile-memory-mb=16384",
        "--verbose-link",   "--verbose",              "--cache-dir",  "/authority/cache",
        "--prefix",         "/authority/out",
    };
    const duplicated_verbose = [_][]const u8{
        "/usr/bin/python3", "/authority/bounded.py",  "--zig",          "/authority/zig",
        "build",            "install",                "-j1",            "-Dmetal=true",
        "-Dcuda=false",     "-Doptimize=ReleaseFast", "-Dstrip=true",   "-Dinference-compile-memory-mb=16384",
        "--verbose",        "--verbose",              "--verbose-link", "--cache-dir",
        "/authority/cache", "--prefix",               "/authority/out",
    };
    try std.testing.expect(!validFullCliCommand(&unknown_extra));
    try std.testing.expect(!validFullCliCommand(&duplicated));
    try std.testing.expect(!validFullCliCommand(&reordered));
    try std.testing.expect(!validFullCliCommand(&duplicated_verbose));
}

test "staged full CLI binding rejects authority mutations" {
    const mutations = [_]struct { old: []const u8, new: []const u8 }{
        .{ .old = "\"source_stable\":true", .new = "\"source_stable\":false" },
        .{ .old = "\"head\":\"abc\",\"dirty\":[]", .new = "\"head\":\"def\",\"dirty\":[]" },
        .{ .old = "\"fresh_standalone_install_required\":true", .new = "\"fresh_standalone_install_required\":false" },
        .{ .old = "\"-Dstrip=true\"", .new = "\"-Dstrip=false\"" },
        .{ .old = "\"bytes\":99", .new = "\"bytes\":0" },
        .{ .old = "\"registry_enrolled\":false", .new = "\"registry_enrolled\":true" },
    };
    for (mutations) |mutation| {
        const changed = try std.mem.replaceOwned(u8, std.testing.allocator, staged_full_cli_good, mutation.old, mutation.new);
        defer std.testing.allocator.free(changed);
        try std.testing.expectError(error.InvalidStagedFullCliBuildReceipt, parseStagedFullCliWithCompiler(std.testing.allocator, changed, .{
            .strip_debug_info = true,
            .have_error_return_tracing = false,
        }));
    }
}
