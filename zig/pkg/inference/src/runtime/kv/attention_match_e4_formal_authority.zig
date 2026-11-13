const std = @import("std");
pub const Cohort = enum { formal_7936_256_v1, formal_32512_256_v1 };
pub const Shape = struct {
    source_rows: u32,
    output_rows: u16,
    range_count: u16,
    candidate_count: u16,
    pub fn validate(self: Shape, cohort: Cohort) !void {
        const expected: Shape = switch (cohort) {
            .formal_7936_256_v1 => .{ .source_rows = 7_936, .output_rows = 256, .range_count = 23, .candidate_count = 69 },
            .formal_32512_256_v1 => .{ .source_rows = 32_512, .output_rows = 256, .range_count = 119, .candidate_count = 357 },
        };
        if (!std.meta.eql(self, expected)) return error.UnqualifiedCompactionArtifact;
    }
};
pub const Authority = struct {
    cohort: Cohort,
    shape: Shape,
    configuration_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    shape_authority_sha256: [32]u8,
    source_inventory_sha256: [32]u8,
    policy_module_sha256: [32]u8,
    wide_32512: ?@import("attention_match_e4_32512_resident_authority.zig").Authority = null,
    pub fn validate(self: Authority) !void {
        try self.shape.validate(self.cohort);
        inline for (.{ self.configuration_sha256, self.source_freeze_sha256, self.shape_authority_sha256, self.source_inventory_sha256, self.policy_module_sha256 }) |digest_value|
            if (std.mem.allEqual(u8, &digest_value, 0)) return error.UnqualifiedCompactionArtifact;
        if (!std.mem.eql(u8, &self.policy_module_sha256, &digest(self.cohort))) return error.UnqualifiedCompactionArtifact;
        switch (self.cohort) {
            .formal_7936_256_v1 => if (self.wide_32512 != null) return error.UnqualifiedCompactionArtifact,
            .formal_32512_256_v1 => {
                const wide = self.wide_32512 orelse return error.UnqualifiedCompactionArtifact;
                try wide.validate();
                if (!std.mem.eql(u8, &wide.configuration_sha256, &self.configuration_sha256) or
                    !std.mem.eql(u8, &wide.source_freeze_sha256, &self.source_freeze_sha256) or
                    !std.mem.eql(u8, &wide.shape_authority_sha256, &self.shape_authority_sha256))
                    return error.UnqualifiedCompactionArtifact;
            },
        }
    }
};
pub const OwnerFacts = struct {
    cohort: Cohort,
    shape: Shape,
    authority_configuration_sha256: [32]u8,
    installed_configuration_sha256: [32]u8,
    source_inventory_sha256: [32]u8,
    installed_source_inventory_sha256: [32]u8,
    full_prefill: bool,
    resumed: bool,
    worker_count: u16,
    cache_dtype_f16: bool,
};
/// Separate formal capability; never broaden the legacy exact-8128 owner.
pub fn permitsOwner(facts: OwnerFacts) bool {
    facts.shape.validate(facts.cohort) catch return false;
    return facts.full_prefill and !facts.resumed and facts.worker_count == 1 and facts.cache_dtype_f16 and
        !std.mem.allEqual(u8, &facts.authority_configuration_sha256, 0) and !std.mem.allEqual(u8, &facts.source_inventory_sha256, 0) and
        std.mem.eql(u8, &facts.authority_configuration_sha256, &facts.installed_configuration_sha256) and
        std.mem.eql(u8, &facts.source_inventory_sha256, &facts.installed_source_inventory_sha256);
}
pub fn digest(cohort: Cohort) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.e4-formal-attention-match-authority.v1\x00");
    hash.update(@tagName(cohort));
    var out: [32]u8 = undefined;
    hash.final(&out);
    return out;
}
test "formal E4 shapes and owner grants are cohort exact" {
    const shape: Shape = .{ .source_rows = 7_936, .output_rows = 256, .range_count = 23, .candidate_count = 69 };
    try shape.validate(.formal_7936_256_v1);
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, shape.validate(.formal_32512_256_v1));
    const good: OwnerFacts = .{ .cohort = .formal_7936_256_v1, .shape = shape, .authority_configuration_sha256 = @splat(1), .installed_configuration_sha256 = @splat(1), .source_inventory_sha256 = @splat(2), .installed_source_inventory_sha256 = @splat(2), .full_prefill = true, .resumed = false, .worker_count = 1, .cache_dtype_f16 = true };
    try std.testing.expect(permitsOwner(good));
    var bad = good;
    bad.shape.source_rows = 8_128;
    try std.testing.expect(!permitsOwner(bad));
    bad = good;
    bad.installed_source_inventory_sha256 = @splat(3);
    try std.testing.expect(!permitsOwner(bad));
    bad = good;
    bad.resumed = true;
    try std.testing.expect(!permitsOwner(bad));
}
