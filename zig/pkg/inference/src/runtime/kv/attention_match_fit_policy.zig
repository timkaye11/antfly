// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Internal producer policy for attention-match checkpoint fitting. This is
//! evidence authority, not a public request surface or a qualification claim.
const std = @import("std");
const engine = @import("attention_match_engine.zig");
const checkpoint = @import("attention_match_checkpoint.zig");

const domain = "antfly.attention-match.fit-policy.v1\x00";

pub const Policy = struct {
    admission: engine.Options.Admission = .fixed,
    chunk_rows: u16 = 1024,
    reject_exact_active_fixed_point: bool = false,

    pub const legacy_fixed: Policy = .{};
    // A test/release candidate name, deliberately not a qualification claim.
    pub const cumulative256_exact_cycle_v1: Policy = .{
        .admission = .cumulative_ladder_4_3_2,
        .chunk_rows = 256,
        .reject_exact_active_fixed_point = true,
    };
    // Development-only diagnostic geometry. This does not widen the public
    // request surface or imply release qualification.
    pub const cumulative128_exact_cycle_v1: Policy = .{
        .admission = .cumulative_ladder_4_3_2,
        .chunk_rows = 128,
        .reject_exact_active_fixed_point = true,
    };

    pub fn validate(self: Policy, encoding: checkpoint.Encoding, metal: bool) !void {
        switch (self.admission) {
            .fixed => if (self.chunk_rows != 1024 or self.reject_exact_active_fixed_point)
                return error.InvalidNativeCheckpointRequest,
            .cumulative_ladder_4_3_2 => {
                if (encoding == .f32) return error.InvalidNativeCheckpointRequest;
                if (!metal or (self.chunk_rows != 128 and self.chunk_rows != 256 and self.chunk_rows != 1024))
                    return error.InvalidNativeCheckpointRequest;
            },
        }
    }

    pub fn digest(self: Policy) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(domain);
        var chunk: [2]u8 = undefined;
        std.mem.writeInt(u16, &chunk, self.chunk_rows, .little);
        hash.update(&.{@backingInt(self.admission)});
        hash.update(&chunk);
        hash.update(&.{@intFromBool(self.reject_exact_active_fixed_point)});
        var result: [32]u8 = undefined;
        hash.final(&result);
        return result;
    }
};

test "attention match fit policy legacy identity and candidate are distinct" {
    try Policy.legacy_fixed.validate(.f16, true);
    try Policy.legacy_fixed.validate(.f32, false);
    try Policy.cumulative256_exact_cycle_v1.validate(.f16, true);
    try Policy.cumulative256_exact_cycle_v1.validate(.fp8, true);
    try Policy.cumulative256_exact_cycle_v1.validate(.rotated4, true);
    try std.testing.expectError(error.InvalidNativeCheckpointRequest, Policy.cumulative256_exact_cycle_v1.validate(.f32, true));
    try std.testing.expect(!std.meta.eql(Policy.legacy_fixed.digest(), Policy.cumulative256_exact_cycle_v1.digest()));
    try std.testing.expectError(error.InvalidNativeCheckpointRequest, Policy.cumulative256_exact_cycle_v1.validate(.f16, false));
    try Policy.cumulative128_exact_cycle_v1.validate(.f16, true);
    try std.testing.expect(!std.meta.eql(Policy.cumulative128_exact_cycle_v1.digest(), Policy.cumulative256_exact_cycle_v1.digest()));
    try std.testing.expect(!std.meta.eql(Policy.cumulative128_exact_cycle_v1.digest(), Policy.legacy_fixed.digest()));
    try std.testing.expectError(error.InvalidNativeCheckpointRequest, Policy.cumulative128_exact_cycle_v1.validate(.f32, true));
    try std.testing.expectError(error.InvalidNativeCheckpointRequest, Policy.cumulative128_exact_cycle_v1.validate(.f16, false));
}

test "attention match fit policy digest binds every field" {
    const base = Policy.cumulative256_exact_cycle_v1;
    var changed = base;
    changed.chunk_rows = 1024;
    try std.testing.expect(!std.meta.eql(base.digest(), changed.digest()));
    changed = base;
    changed.reject_exact_active_fixed_point = false;
    try std.testing.expect(!std.meta.eql(base.digest(), changed.digest()));
    changed = Policy.legacy_fixed;
    try std.testing.expect(!std.meta.eql(base.digest(), changed.digest()));
}
