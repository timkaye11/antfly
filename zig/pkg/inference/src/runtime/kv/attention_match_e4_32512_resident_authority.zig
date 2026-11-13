const std = @import("std");
const shape = @import("attention_match_e4_32512_shape_authority.zig");
const pressure = @import("attention_match_e4_32512_pressure_profile.zig");
const packed_source = @import("attention_match_e4_32512_packed_head_source.zig");

pub const policy = "e4_32512_resident_authority_v1|one_shot_full_prompt|page_train_fp64|tsqr_values|accelerated_heldout_fp64|postfit_exact|independent_publication|owner1152m|combined2g|guard8g";
pub const policy_sha256: [32]u8 = blk: {
    @setEvalBranchQuota(10_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &out, .{});
    break :blk out;
};

pub const owner_physical_cap: u64 = 1_207_959_552;
pub const combined_head_cap: u64 = 2 * 1024 * 1024 * 1024;
pub const process_guard: u64 = 8 * 1024 * 1024 * 1024;
pub const maximum_head_download_bytes: u64 = 256 * 1024 * 1024;
pub const candidate_device_max_bytes: u64 = combined_head_cap - owner_physical_cap;

pub fn digest() [32]u8 {
    return policy_sha256;
}

pub fn futurePolicyDigest() [32]u8 {
    var result: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash("antfly.e4-32512-resident-future-policy.v1|P256_Hdense512_Hselected256_F256_HHO64|independent-heldout", &result, .{});
    return result;
}

pub const Authority = struct {
    configuration_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    implementation_semantic_sha256: [32]u8,
    profile_semantic_sha256: [32]u8,
    shape_authority_sha256: [32]u8,
    pressure_policy_sha256: [32]u8,
    packed_source_policy_sha256: [32]u8,
    policy_module_sha256: [32]u8,
    logical_tokens: usize,
    range_count: usize,
    candidate_count: usize,
    owner_cap_bytes: u64,
    combined_cap_bytes: u64,
    process_guard_bytes: u64,
    maximum_head_download_bytes: u64,
    candidate_device_max_bytes: u64,

    pub fn validate(self: @This()) !void {
        inline for (.{ self.configuration_sha256, self.source_freeze_sha256, self.implementation_semantic_sha256, self.profile_semantic_sha256, self.shape_authority_sha256 }) |value|
            if (std.mem.allEqual(u8, &value, 0)) return error.UnqualifiedCompactionArtifact;
        if (!std.crypto.timing_safe.eql([32]u8, self.pressure_policy_sha256, pressure.policy_sha256) or
            !std.crypto.timing_safe.eql([32]u8, self.packed_source_policy_sha256, packed_source.policy_sha256) or
            !std.crypto.timing_safe.eql([32]u8, self.policy_module_sha256, policy_sha256) or
            self.logical_tokens != shape.source_rows or self.range_count != shape.range_count or self.candidate_count != shape.candidate_count or
            self.owner_cap_bytes != owner_physical_cap or self.combined_cap_bytes != combined_head_cap or self.process_guard_bytes != process_guard or
            self.maximum_head_download_bytes != maximum_head_download_bytes or self.candidate_device_max_bytes != candidate_device_max_bytes)
            return error.UnqualifiedCompactionArtifact;
        const shape_binding: shape.Binding = .{
            .source_freeze_sha256 = self.source_freeze_sha256,
            .configuration_sha256 = self.configuration_sha256,
            .pressure_policy_sha256 = self.pressure_policy_sha256,
            .packed_source_policy_sha256 = self.packed_source_policy_sha256,
            .expected_digest = self.shape_authority_sha256,
        };
        _ = try shape.validate(shape_binding);
    }
};

fn valid() Authority {
    var value: Authority = .{
        .configuration_sha256 = @splat(1),
        .source_freeze_sha256 = @splat(2),
        .implementation_semantic_sha256 = @splat(3),
        .profile_semantic_sha256 = @splat(4),
        .shape_authority_sha256 = @splat(0),
        .pressure_policy_sha256 = pressure.policy_sha256,
        .packed_source_policy_sha256 = packed_source.policy_sha256,
        .policy_module_sha256 = policy_sha256,
        .logical_tokens = shape.source_rows,
        .range_count = shape.range_count,
        .candidate_count = shape.candidate_count,
        .owner_cap_bytes = owner_physical_cap,
        .combined_cap_bytes = combined_head_cap,
        .process_guard_bytes = process_guard,
        .maximum_head_download_bytes = maximum_head_download_bytes,
        .candidate_device_max_bytes = candidate_device_max_bytes,
    };
    var binding: shape.Binding = .{ .source_freeze_sha256 = value.source_freeze_sha256, .configuration_sha256 = value.configuration_sha256, .pressure_policy_sha256 = value.pressure_policy_sha256, .packed_source_policy_sha256 = value.packed_source_policy_sha256, .expected_digest = @splat(0) };
    binding.expected_digest = shape.digest(binding) catch unreachable;
    value.shape_authority_sha256 = binding.expected_digest;
    return value;
}

test "E4 32512 cohort closes shape pressure packed source and unchanged caps" {
    try valid().validate();
}

test "E4 32512 cohort rejects max-policy-only and cap drift" {
    var value = valid();
    value.shape_authority_sha256 = @splat(9);
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, value.validate());
    value = valid();
    value.owner_cap_bytes += 1;
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, value.validate());
    value = valid();
    value.packed_source_policy_sha256[0] +%= 1;
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, value.validate());
}
