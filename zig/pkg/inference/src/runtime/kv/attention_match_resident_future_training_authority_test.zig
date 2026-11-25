const std = @import("std");
const authority = @import("attention_match_future_decode_training.zig");

test "context resident future authority keeps legacy digest byte identical and is distinct" {
    const legacy = authority.Authority{
        .source_identity_sha256 = @splat(1),
        .source_tokens_sha256 = @splat(2),
        .source_revision = 7,
        .generation_contract_sha256 = authority.policyDigest(),
        .generated_ids_sha256 = @splat(3),
        .primary_sha256 = @splat(4),
        .historical_sha256 = @splat(5),
        .future_sha256 = @splat(6),
        .heldout_sha256 = @splat(7),
        .early_eos_observations = 9,
    };
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(&authority.policyDigest());
    inline for (.{ legacy.source_identity_sha256, legacy.source_tokens_sha256, legacy.generation_contract_sha256, legacy.generated_ids_sha256, legacy.primary_sha256, legacy.historical_sha256, legacy.future_sha256, legacy.heldout_sha256 }) |value| hash.update(&value);
    var revision: [8]u8 = undefined;
    std.mem.writeInt(u64, &revision, legacy.source_revision, .little);
    hash.update(&revision);
    var eos: [2]u8 = undefined;
    std.mem.writeInt(u16, &eos, legacy.early_eos_observations, .little);
    hash.update(&eos);
    try std.testing.expectEqual(hash.finalResult(), legacy.digest());

    var resident = legacy;
    resident.generation_contract_sha256 = authority.residentHybridPolicyDigest();
    try std.testing.expect(authority.supportedPolicyDigest(resident.generation_contract_sha256));
    const resident_digest = resident.digest();
    const legacy_digest = legacy.digest();
    const resident_policy = authority.residentHybridPolicyDigest();
    const legacy_policy = authority.policyDigest();
    try std.testing.expect(!std.mem.eql(u8, &resident_digest, &legacy_digest));
    try std.testing.expect(!std.mem.eql(u8, &resident_policy, &legacy_policy));
}
