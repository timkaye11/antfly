const std = @import("std");
const bounded = @import("context_private_dense_profile.zig");
const base_mod = @import("context_resident_hybrid_compatibility_profile.zig");
const config_mod = @import("context_resident_hybrid_staged_configuration.zig");
const compaction_identity = @import("compaction_identity");

pub const Loaded = struct {
    profile: bounded.Loaded,
    authority: config_mod.Authority,
    pub fn deinit(self: *@This()) void {
        self.profile.deinit();
    }
};

fn validateJoin(profile: @import("runtime/kv/compaction_profile.zig").Profile, sealed: @import("runtime/kv/compaction_profile.zig").ResidentAuthorityBinding, base: base_mod.Loaded, authority: config_mod.Authority) !void {
    if (base.cohort != .e4_8128_v1 or authority.cohort != .e4_8128_v1 or
        !std.meta.eql(base.profile.identity, profile.identity) or
        !std.mem.eql(u8, &base.profile_semantic_sha256, &sealed.base_compatibility_semantic_sha256) or
        !std.mem.eql(u8, &base.source_freeze_sha256, &sealed.source_freeze_sha256) or
        !std.mem.eql(u8, &authority.configuration_sha256, &sealed.configuration_sha256) or
        !std.mem.eql(u8, &authority.source_freeze_sha256, &sealed.source_freeze_sha256) or
        !std.mem.eql(u8, &authority.shape_authority_sha256, &sealed.shape_authority_sha256) or
        !std.mem.eql(u8, &authority.policy_module_sha256, &sealed.policy_module_sha256) or
        !std.mem.eql(u8, &authority.profile_semantic_sha256, &sealed.base_compatibility_semantic_sha256)) return error.InvalidQualifiedResidentProfile;
}

fn readPinned(a: std.mem.Allocator, io: std.Io, path: []const u8, expected: [32]u8) ![]u8 {
    if (!std.fs.path.isAbsolute(path)) return error.InvalidQualifiedResidentProfile;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 * 1024 * 1024));
    errdefer a.free(bytes);
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &actual, .{});
    if (!std.mem.eql(u8, &actual, &expected)) return error.InvalidQualifiedResidentProfile;
    return bytes;
}

pub fn loadBytes(a: std.mem.Allocator, io: std.Io, profile_bytes: []const u8) !Loaded {
    var profile = bounded.parseBytes(a, profile_bytes) catch return error.InvalidQualifiedResidentProfile;
    errdefer profile.deinit();
    const artifact = profile.parsed.value;
    if (!std.mem.eql(u8, artifact.schema, bounded.qualified_resident_schema)) return error.InvalidQualifiedResidentProfile;
    const sealed = profile.profile.resident_authority orelse return error.InvalidQualifiedResidentProfile;
    const base_bytes = try readPinned(a, io, artifact.base_compatibility_profile_path.?, sealed.base_compatibility_artifact_sha256);
    defer a.free(base_bytes);
    var base = base_mod.parseBytes(a, base_bytes) catch return error.InvalidQualifiedResidentProfile;
    defer base.deinit();
    const config_bytes = try readPinned(a, io, artifact.resident_configuration_path.?, sealed.configuration_sha256);
    defer a.free(config_bytes);
    const authority = config_mod.parse(a, config_bytes, .{
        .configuration_sha256 = sealed.configuration_sha256,
        .implementation_semantic_sha256 = compaction_identity.digest,
        .shape_authority_sha256 = sealed.shape_authority_sha256,
        .policy_module_sha256 = sealed.policy_module_sha256,
        .profile_semantic_sha256 = sealed.base_compatibility_semantic_sha256,
    }) catch return error.InvalidQualifiedResidentProfile;
    try validateJoin(profile.profile, sealed, base, authority);
    return .{ .profile = profile, .authority = authority };
}

test "qualified bounded profile rejects missing and unpinned base configuration" {
    try std.testing.expectError(error.InvalidQualifiedResidentProfile, loadBytes(std.testing.allocator, std.testing.io, "{}"));
}

pub fn testBaseJson(a: std.mem.Allocator, source: [32]u8) ![]u8 {
    const one = "1111111111111111111111111111111111111111111111111111111111111111";
    return std.fmt.allocPrint(a,
        \\{{"schema":"{s}","stage":"{s}","diagnostic_only":true,"candidate_qualification":false,"serving_qualified":false,"release_authority":false,"registry_enrollment":false,"evaluation_opened":false,"evidence_present":false,"frozen_before_unseen_evaluation":true,"training_policy":"private_train_challenge_hybrid_resident_v1","fit_strategy":"identity_chunk_coefficients_adaptive_v18","identity":{{"artifact_sha256":"{s}","tokenizer_sha256":"{s}","template_sha256":"{s}","layout_sha256":"{s}","implementation_sha256":"{s}"}},"profile_semantic_sha256":"{s}","max_prompt_tokens":32768,"max_output_tokens":256,"owners":[{{"layer":5,"heads":[{{"head":0,"retained_rows":512}},{{"head":1,"retained_rows":512}}]}},{{"layer":11,"heads":[{{"head":0,"retained_rows":512}},{{"head":1,"retained_rows":512}}]}},{{"layer":17,"heads":[{{"head":0,"retained_rows":512}},{{"head":1,"retained_rows":512}}]}},{{"layer":23,"heads":[{{"head":0,"retained_rows":512}},{{"head":1,"retained_rows":512}}]}}],"source":{{"source_freeze":{{"path":"/sealed/source.json","sha256":"{s}"}}}}}}
    , .{ base_mod.schema, base_mod.stage, one, one, one, one, std.fmt.bytesToHex(base_mod.runtimeImplementationDigest(), .lower), std.fmt.bytesToHex(base_mod.expectedProfileSemantic(), .lower), std.fmt.bytesToHex(source, .lower) });
}

pub fn testConfigJson(a: std.mem.Allocator, source: [32]u8, shape: [32]u8) ![]u8 {
    return std.fmt.allocPrint(a,
        \\{{"schema":"{s}","stage":"resident_native_hybrid_pre_evaluation_configuration","diagnostic_only":true,"candidate_qualification":false,"serving_qualified":false,"release_authority":false,"registry_enrollment":false,"evaluation_opened":false,"frozen_before_unseen_evaluation":true,"evidence_present":false,"training_policy":"{s}","fit_strategy":"identity_chunk_coefficients_adaptive_v18","profile_semantic_sha256":"{s}","policy_module_sha256":"{s}","implementation_semantic_sha256":"{s}","shape_authority_sha256":"{s}","source_rows":8128,"range_count":24,"candidate_count":72,"eligible_heads":[{{"donor_layer":5,"head_index":0}},{{"donor_layer":5,"head_index":1}},{{"donor_layer":11,"head_index":0}},{{"donor_layer":11,"head_index":1}},{{"donor_layer":17,"head_index":0}},{{"donor_layer":17,"head_index":1}},{{"donor_layer":23,"head_index":0}},{{"donor_layer":23,"head_index":1}}],"resources":{{"per_head_host_workspace_bytes":2147483648,"maximum_transient_host_bytes":4026531840,"process_guard_bytes":8589934592,"fit_workers":1}},"physical_policy":{{"primary_percent":91,"fallback_percent":95,"maximum_fallbacks":3,"planned_exact_heads":1,"minimum_ratio":1.5}},"source":{{"source_freeze":{{"path":"/sealed/source.json","sha256":"{s}"}}}}}}
    , .{ config_mod.schema, config_mod.policy_name, std.fmt.bytesToHex(shape, .lower), std.fmt.bytesToHex(config_mod.policy_module_sha256, .lower), std.fmt.bytesToHex(compaction_identity.digest, .lower), std.fmt.bytesToHex(shape, .lower), std.fmt.bytesToHex(source, .lower) });
}

test "qualified bounded B C P files join and reject cross campaign and artifact tamper" {
    const a = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    defer a.free(root);
    const b_path = try std.fs.path.join(a, &.{ root, "B.json" });
    defer a.free(b_path);
    const c_path = try std.fs.path.join(a, &.{ root, "C.json" });
    defer a.free(c_path);
    const source: [32]u8 = @splat(7);
    const base = try testBaseJson(a, source);
    defer a.free(base);
    var base_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(base, &base_sha, .{});
    const base_semantic = base_mod.expectedProfileSemantic();
    const config = try testConfigJson(a, source, base_semantic);
    defer a.free(config);
    var config_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(config, &config_sha, .{});
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "B.json", .data = base });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "C.json", .data = config });
    const profile = try bounded.testQualifiedResidentJson(a, b_path, c_path, base_sha, base_semantic, config_sha, source, base_semantic);
    defer a.free(profile);
    var loaded = try loadBytes(a, std.testing.io, profile);
    loaded.deinit();

    var repeated_tuple_tamper = try a.dupe(u8, profile);
    defer a.free(repeated_tuple_tamper);
    const tuple_needle = "\"logical_tokens\":8128";
    const tuple_at = std.mem.indexOf(u8, repeated_tuple_tamper, tuple_needle).?;
    repeated_tuple_tamper[tuple_at + tuple_needle.len - 1] = '7';
    try std.testing.expectError(error.InvalidQualifiedResidentProfile, loadBytes(a, std.testing.io, repeated_tuple_tamper));

    var wrong_source = source;
    wrong_source[0] ^= 1;
    const cross = try bounded.testQualifiedResidentJson(a, b_path, c_path, base_sha, base_semantic, config_sha, wrong_source, base_semantic);
    defer a.free(cross);
    try std.testing.expectError(error.InvalidQualifiedResidentProfile, loadBytes(a, std.testing.io, cross));
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "B.json", .data = "tampered" });
    try std.testing.expectError(error.InvalidQualifiedResidentProfile, loadBytes(a, std.testing.io, profile));
}

test "bounded P rejects the ambiguous predecessor schema" {
    const old =
        \\{"schema":"antfly.context.qualified-resident-e4.profile.v1","stage":"qualified_resident_e4_inactive"}
    ;
    try std.testing.expectError(error.InvalidQualifiedResidentProfile, loadBytes(std.testing.allocator, std.testing.io, old));
}
