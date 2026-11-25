const std = @import("std");

pub const policy = "e4_32512_resident_shape_v1|source32512|dim512|consumers56|heads8|P256_Hdense512_Hselected256_F256_HHO64|prefix128_recent2048_chunk256|ranges119|candidates357|p91_p95_max3|page_required";
pub const policy_sha256: [32]u8 = blk: {
    @setEvalBranchQuota(100_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &out, .{});
    break :blk out;
};

pub const source_rows: usize = 32_512;
pub const dim: usize = 512;
pub const batch_count: usize = 56;
pub const eligible_heads: usize = 8;
pub const prefix_rows: usize = 128;
pub const recent_rows: usize = 2048;
pub const chunk_rows: usize = 256;
pub const primary_rows: usize = 256;
pub const dense_historical_rows: usize = 512;
pub const selected_historical_rows: usize = 256;
pub const future_rows: usize = 256;
pub const heldout_rows: usize = 64;
pub const range_count: usize = 119;
pub const candidate_count: usize = 357;
pub const primary_percent: u8 = 91;
pub const fallback_percent: u8 = 95;
pub const maximum_fallbacks: u8 = 3;

pub const Binding = struct {
    source_freeze_sha256: [32]u8,
    configuration_sha256: [32]u8,
    pressure_policy_sha256: [32]u8,
    packed_source_policy_sha256: [32]u8,
    expected_digest: [32]u8,
};

pub const Validated = struct {
    binding: Binding,
    middle_rows: usize,
    range_count: usize,
    candidate_count: usize,
    digest: [32]u8,
};

fn nonzero(value: [32]u8) bool {
    return !std.mem.allEqual(u8, &value, 0);
}

fn put(hash: *std.crypto.hash.sha2.Sha256, value: usize) !void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, std.math.cast(u64, value) orelse return error.Overflow, .little);
    hash.update(&bytes);
}

pub fn digest(binding: Binding) ![32]u8 {
    if (!nonzero(binding.source_freeze_sha256) or !nonzero(binding.configuration_sha256) or
        !nonzero(binding.pressure_policy_sha256) or !nonzero(binding.packed_source_policy_sha256))
        return error.InvalidTrainingAugmentationBinding;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy);
    hash.update(&policy_sha256);
    hash.update(&binding.source_freeze_sha256);
    hash.update(&binding.configuration_sha256);
    hash.update(&binding.pressure_policy_sha256);
    hash.update(&binding.packed_source_policy_sha256);
    inline for (.{ source_rows, dim, batch_count, eligible_heads, prefix_rows, recent_rows, chunk_rows, primary_rows, dense_historical_rows, selected_historical_rows, future_rows, heldout_rows, range_count, candidate_count }) |value| try put(&hash, value);
    hash.update(&.{ primary_percent, fallback_percent, maximum_fallbacks });
    return hash.finalResult();
}

pub fn validate(binding: Binding) !Validated {
    const middle = source_rows - prefix_rows - recent_rows;
    const ranges = std.math.divCeil(usize, source_rows - prefix_rows - recent_rows, chunk_rows) catch return error.InvalidFitConfig;
    const candidates = try std.math.mul(usize, ranges, 3);
    if (middle != 30_336 or ranges != range_count or candidates != candidate_count)
        return error.UnsupportedShape;
    const actual = try digest(binding);
    if (!nonzero(binding.expected_digest) or !std.crypto.timing_safe.eql([32]u8, actual, binding.expected_digest))
        return error.InvalidTrainingAugmentationBinding;
    return .{ .binding = binding, .middle_rows = middle, .range_count = ranges, .candidate_count = candidates, .digest = actual };
}

fn testBinding(seed: u8) Binding {
    var value: Binding = .{
        .source_freeze_sha256 = @splat(seed),
        .configuration_sha256 = @splat(seed + 1),
        .pressure_policy_sha256 = @splat(seed + 2),
        .packed_source_policy_sha256 = @splat(seed + 3),
        .expected_digest = @splat(0),
    };
    value.expected_digest = digest(value) catch unreachable;
    return value;
}

test "E4 32512 authority seals 119 ranges and 357 page candidates" {
    const value = try validate(testBinding(1));
    try std.testing.expectEqual(@as(usize, 30_336), value.middle_rows);
    try std.testing.expectEqual(@as(usize, 119), value.range_count);
    try std.testing.expectEqual(@as(usize, 357), value.candidate_count);
    try std.testing.expect(!std.mem.eql(u8, &value.digest, &policy_sha256));
}

test "E4 32512 authority binds source config pressure and packed producer independently" {
    const original = testBinding(1);
    _ = try validate(original);
    inline for (.{ "source_freeze_sha256", "configuration_sha256", "pressure_policy_sha256", "packed_source_policy_sha256" }) |field| {
        var changed = original;
        @field(changed, field)[0] +%= 1;
        try std.testing.expectError(error.InvalidTrainingAugmentationBinding, validate(changed));
    }
    var wrong = original;
    wrong.expected_digest[0] +%= 1;
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, validate(wrong));
}

test "E4 32512 authority rejects zero producer domains" {
    var value = testBinding(1);
    value.packed_source_policy_sha256 = @splat(0);
    value.expected_digest = @splat(0);
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, digest(value));
}
