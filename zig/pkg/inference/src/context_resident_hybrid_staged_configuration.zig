// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Strict parser for the unqualified resident-hybrid operator configuration.
//! This artifact authorizes a future evaluation; it is neither evidence nor
//! registry enrollment and it never substitutes for a real compaction Profile.
const std = @import("std");

pub const schema = "antfly.context.resident-native-hybrid.staged-configuration.v1";
pub const e2_schema = "antfly.context.resident-native-hybrid-e2-8128.staged-configuration.v1";
pub const e2_guarded_trial_schema = "antfly.context.resident-native-hybrid-e2-8128.guarded-development-trial-configuration.v1";
pub const e2_formal_schema = "antfly.context.resident-native-hybrid-e2-formal.staged-configuration.v1";
pub const policy_name = "private_train_challenge_hybrid_resident_v1";
pub const policy_module_sha256 = hex("e0a407063023d89c40f516270081012329a8981dc5c64de2dd8a985d8469afee") catch unreachable;
pub const e2_policy_module_sha256 = @import("runtime/kv/attention_match_e2_resident_authority.zig").digest();
pub const e2_guarded_trial_policy_module_sha256 = @import("runtime/kv/attention_match_e2_resident_authority.zig").guardedTrialDigest();
pub const e2_formal_policy_module_sha256 = @import("runtime/kv/attention_match_e2_resident_authority.zig").formalFitDigest();

pub const Cohort = enum { e4_8128_v1, e2_8128_v1 };

pub const Authority = struct {
    configuration_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    shape_authority_sha256: [32]u8,
    policy_module_sha256: [32]u8,
    profile_semantic_sha256: [32]u8,
    cohort: Cohort = .e4_8128_v1,
    guarded_development_trial: bool = false,
    e2_fit_source_rows: usize = 8128,
    formal_owner_inventory_sha256: ?[32]u8 = null,
    formal_context_candidate_receipt_sha256: ?[32]u8 = null,
    formal_owner_inventory: ?@import("runtime/kv/attention_match_e2_formal_owner_inventory.zig").Artifact = null,
};

pub const Expected = struct {
    configuration_sha256: [32]u8,
    implementation_semantic_sha256: [32]u8,
    shape_authority_sha256: [32]u8,
    policy_module_sha256: [32]u8,
    profile_semantic_sha256: [32]u8,
};

const Head = struct { donor_layer: u32, head_index: u32 };
const Resources = struct {
    per_head_host_workspace_bytes: u64,
    maximum_transient_host_bytes: u64,
    process_guard_bytes: u64,
    fit_workers: u32,
};
const Physical = struct {
    primary_percent: u32,
    fallback_percent: u32,
    maximum_fallbacks: u32,
    planned_exact_heads: u32,
    minimum_ratio: f64,
};
const Ref = struct { path: []const u8, sha256: []const u8 };
const Source = struct { source_freeze: Ref };
const Configuration = struct {
    schema: []const u8,
    cohort: ?[]const u8 = null,
    stage: []const u8,
    diagnostic_only: bool,
    candidate_qualification: bool,
    serving_qualified: bool,
    release_authority: bool,
    registry_enrollment: bool,
    evaluation_opened: bool,
    frozen_before_unseen_evaluation: bool,
    evidence_present: bool,
    training_policy: []const u8,
    fit_strategy: []const u8,
    profile_semantic_sha256: []const u8,
    policy_module_sha256: []const u8,
    implementation_semantic_sha256: []const u8,
    shape_authority_sha256: []const u8,
    source_rows: u32,
    range_count: u32,
    candidate_count: u32,
    eligible_heads: []const Head,
    resources: Resources,
    physical_policy: Physical,
    source: Source,
    formal_owner_inventory: ?Ref = null,
    formal_context_candidate_receipt: ?Ref = null,
};

fn hex(value: []const u8) ![32]u8 {
    if (value.len != 64) return error.InvalidResidentHybridConfiguration;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, value) catch return error.InvalidResidentHybridConfiguration;
    return out;
}

fn nonzero(value: [32]u8) bool {
    const zero: [32]u8 = @splat(0);
    return !std.mem.eql(u8, &value, &zero);
}

pub fn parse(allocator: std.mem.Allocator, bytes: []const u8, expected: Expected) !Authority {
    var actual_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &actual_sha, .{});
    if (!std.mem.eql(u8, &actual_sha, &expected.configuration_sha256) or
        !nonzero(expected.configuration_sha256) or !nonzero(expected.implementation_semantic_sha256) or
        !nonzero(expected.shape_authority_sha256) or
        (!std.mem.eql(u8, &expected.policy_module_sha256, &policy_module_sha256) and
            !std.mem.eql(u8, &expected.policy_module_sha256, &e2_policy_module_sha256) and
            !std.mem.eql(u8, &expected.policy_module_sha256, &e2_guarded_trial_policy_module_sha256) and
            !std.mem.eql(u8, &expected.policy_module_sha256, &e2_formal_policy_module_sha256)))
        return error.InvalidResidentHybridConfiguration;

    var parsed = std.json.parseFromSlice(Configuration, allocator, bytes, .{
        .allocate = .alloc_always,
        .ignore_unknown_fields = false,
    }) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return error.InvalidResidentHybridConfiguration,
    };
    defer parsed.deinit();
    const c = parsed.value;
    const implementation = try hex(c.implementation_semantic_sha256);
    const shape = try hex(c.shape_authority_sha256);
    const module = try hex(c.policy_module_sha256);
    const source_freeze = try hex(c.source.source_freeze.sha256);
    const profile_semantic = try hex(c.profile_semantic_sha256);

    const guarded_trial = std.mem.eql(u8, c.schema, e2_guarded_trial_schema);
    const formal = std.mem.eql(u8, c.schema, e2_formal_schema);
    const cohort: Cohort = if (std.mem.eql(u8, c.schema, schema) and c.cohort == null)
        .e4_8128_v1
    else if ((std.mem.eql(u8, c.schema, e2_schema) or guarded_trial) and c.cohort != null and std.mem.eql(u8, c.cohort.?, "e2_8128_v1"))
        .e2_8128_v1
    else if (formal and c.cohort != null and std.mem.eql(u8, c.cohort.?, "e2_formal_v1"))
        .e2_8128_v1
    else
        return error.InvalidResidentHybridConfiguration;
    const e2 = cohort == .e2_8128_v1;
    const formal_range_count: usize = if (c.source_rows == 7_936) 23 else 119;
    const expected_module = if (formal) e2_formal_policy_module_sha256 else if (guarded_trial) e2_guarded_trial_policy_module_sha256 else if (e2) e2_policy_module_sha256 else policy_module_sha256;
    if (!std.mem.eql(u8, &expected.policy_module_sha256, &expected_module) or
        !std.mem.eql(u8, c.stage, "resident_native_hybrid_pre_evaluation_configuration") or
        !c.diagnostic_only or c.candidate_qualification or c.serving_qualified or c.release_authority or
        c.registry_enrollment or c.evaluation_opened or !c.frozen_before_unseen_evaluation or c.evidence_present or
        !std.mem.eql(u8, c.training_policy, policy_name) or
        !std.mem.eql(u8, c.fit_strategy, "identity_chunk_coefficients_adaptive_v18") or
        !std.mem.eql(u8, &implementation, &expected.implementation_semantic_sha256) or
        !std.mem.eql(u8, &shape, &expected.shape_authority_sha256) or
        !std.mem.eql(u8, &module, &expected.policy_module_sha256) or
        !std.mem.eql(u8, &profile_semantic, &expected.profile_semantic_sha256) or
        !nonzero(source_freeze) or !std.fs.path.isAbsolute(c.source.source_freeze.path) or
        (!formal and (c.source_rows != 8128 or c.range_count != 24 or c.candidate_count != 72)) or
        (formal and ((c.source_rows != 7_936 and c.source_rows != 32_512) or
            c.range_count != formal_range_count or c.candidate_count != c.range_count * 3)) or
        c.resources.per_head_host_workspace_bytes != 2 * 1024 * 1024 * 1024 or
        c.resources.maximum_transient_host_bytes != 4026531840 or
        c.resources.process_guard_bytes != 8 * 1024 * 1024 * 1024 or c.resources.fit_workers != 1 or
        c.physical_policy.planned_exact_heads != 1 or c.physical_policy.minimum_ratio != 1.5 or
        (e2 and (c.physical_policy.primary_percent != 60 or c.physical_policy.fallback_percent != 0 or c.physical_policy.maximum_fallbacks != 0 or c.eligible_heads.len != 3)) or
        (!e2 and (c.physical_policy.primary_percent != 91 or c.physical_policy.fallback_percent != 95 or c.physical_policy.maximum_fallbacks != 3 or c.eligible_heads.len != 8)))
        return error.InvalidResidentHybridConfiguration;
    const formal_inventory_sha256: ?[32]u8 = if (formal) blk: {
        const inventory = c.formal_owner_inventory orelse return error.InvalidResidentHybridConfiguration;
        const receipt = c.formal_context_candidate_receipt orelse return error.InvalidResidentHybridConfiguration;
        if (!std.fs.path.isAbsolute(inventory.path) or !std.fs.path.isAbsolute(receipt.path) or
            std.mem.eql(u8, inventory.path, receipt.path)) return error.InvalidResidentHybridConfiguration;
        break :blk try hex(inventory.sha256);
    } else blk: {
        if (c.formal_owner_inventory != null or c.formal_context_candidate_receipt != null) return error.InvalidResidentHybridConfiguration;
        break :blk null;
    };
    const formal_receipt_sha256: ?[32]u8 = if (formal)
        try hex(c.formal_context_candidate_receipt.?.sha256)
    else
        null;
    if (e2) {
        const donors = [_]u32{ 4, 9, 14 };
        for (c.eligible_heads, donors) |head, donor|
            if (head.donor_layer != donor or head.head_index != 0) return error.InvalidResidentHybridConfiguration;
    } else {
        const donors = [_]u32{ 5, 11, 17, 23 };
        for (c.eligible_heads, 0..) |head, i| {
            const donor = donors[i / 2];
            if (head.donor_layer != donor or head.head_index != i % 2)
                return error.InvalidResidentHybridConfiguration;
        }
    }
    return .{
        .configuration_sha256 = actual_sha,
        .source_freeze_sha256 = source_freeze,
        .shape_authority_sha256 = shape,
        .policy_module_sha256 = module,
        .profile_semantic_sha256 = profile_semantic,
        .cohort = cohort,
        .guarded_development_trial = guarded_trial,
        .e2_fit_source_rows = c.source_rows,
        .formal_owner_inventory_sha256 = formal_inventory_sha256,
        .formal_context_candidate_receipt_sha256 = formal_receipt_sha256,
    };
}

test "E2 guarded development trial has distinct schema and policy authority" {
    try std.testing.expect(!std.mem.eql(u8, &e2_policy_module_sha256, &e2_guarded_trial_policy_module_sha256));
    try std.testing.expect(!std.mem.eql(u8, e2_schema, e2_guarded_trial_schema));
    const a = std.testing.allocator;
    const one = "1111111111111111111111111111111111111111111111111111111111111111";
    const two = "2222222222222222222222222222222222222222222222222222222222222222";
    const three = "3333333333333333333333333333333333333333333333333333333333333333";
    const disabled = try validE2Json(a, one, two, three);
    defer a.free(disabled);
    const with_schema = try std.mem.replaceOwned(u8, a, disabled, e2_schema, e2_guarded_trial_schema);
    defer a.free(with_schema);
    const disabled_module = std.fmt.bytesToHex(e2_policy_module_sha256, .lower);
    const trial_module = std.fmt.bytesToHex(e2_guarded_trial_policy_module_sha256, .lower);
    const bytes = try std.mem.replaceOwned(u8, a, with_schema, &disabled_module, &trial_module);
    defer a.free(bytes);
    var configuration_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &configuration_sha256, .{});
    const got = try parse(a, bytes, .{ .configuration_sha256 = configuration_sha256, .implementation_semantic_sha256 = try hex(three), .shape_authority_sha256 = try hex(two), .policy_module_sha256 = e2_guarded_trial_policy_module_sha256, .profile_semantic_sha256 = try hex(two) });
    try std.testing.expect(got.guarded_development_trial);
    try std.testing.expectEqual(Cohort.e2_8128_v1, got.cohort);
}

fn validJson(allocator: std.mem.Allocator, source_sha: []const u8, shape_sha: []const u8, implementation_sha: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\{{"schema":"{s}","stage":"resident_native_hybrid_pre_evaluation_configuration","diagnostic_only":true,"candidate_qualification":false,"serving_qualified":false,"release_authority":false,"registry_enrollment":false,"evaluation_opened":false,"frozen_before_unseen_evaluation":true,"evidence_present":false,"training_policy":"{s}","fit_strategy":"identity_chunk_coefficients_adaptive_v18","profile_semantic_sha256":"{s}","policy_module_sha256":"{s}","implementation_semantic_sha256":"{s}","shape_authority_sha256":"{s}","source_rows":8128,"range_count":24,"candidate_count":72,"eligible_heads":[{{"donor_layer":5,"head_index":0}},{{"donor_layer":5,"head_index":1}},{{"donor_layer":11,"head_index":0}},{{"donor_layer":11,"head_index":1}},{{"donor_layer":17,"head_index":0}},{{"donor_layer":17,"head_index":1}},{{"donor_layer":23,"head_index":0}},{{"donor_layer":23,"head_index":1}}],"resources":{{"per_head_host_workspace_bytes":2147483648,"maximum_transient_host_bytes":4026531840,"process_guard_bytes":8589934592,"fit_workers":1}},"physical_policy":{{"primary_percent":91,"fallback_percent":95,"maximum_fallbacks":3,"planned_exact_heads":1,"minimum_ratio":1.5}},"source":{{"source_freeze":{{"path":"/sealed/source.json","sha256":"{s}"}}}}}}
    , .{ schema, policy_name, shape_sha, std.fmt.bytesToHex(policy_module_sha256, .lower), implementation_sha, shape_sha, source_sha });
}

fn validE2Json(allocator: std.mem.Allocator, source_sha: []const u8, shape_sha: []const u8, implementation_sha: []const u8) ![]u8 {
    return std.fmt.allocPrint(allocator,
        \\{{"schema":"{s}","cohort":"e2_8128_v1","stage":"resident_native_hybrid_pre_evaluation_configuration","diagnostic_only":true,"candidate_qualification":false,"serving_qualified":false,"release_authority":false,"registry_enrollment":false,"evaluation_opened":false,"frozen_before_unseen_evaluation":true,"evidence_present":false,"training_policy":"{s}","fit_strategy":"identity_chunk_coefficients_adaptive_v18","profile_semantic_sha256":"{s}","policy_module_sha256":"{s}","implementation_semantic_sha256":"{s}","shape_authority_sha256":"{s}","source_rows":8128,"range_count":24,"candidate_count":72,"eligible_heads":[{{"donor_layer":4,"head_index":0}},{{"donor_layer":9,"head_index":0}},{{"donor_layer":14,"head_index":0}}],"resources":{{"per_head_host_workspace_bytes":2147483648,"maximum_transient_host_bytes":4026531840,"process_guard_bytes":8589934592,"fit_workers":1}},"physical_policy":{{"primary_percent":60,"fallback_percent":0,"maximum_fallbacks":0,"planned_exact_heads":1,"minimum_ratio":1.5}},"source":{{"source_freeze":{{"path":"/sealed/source.json","sha256":"{s}"}}}}}}
    , .{ e2_schema, policy_name, shape_sha, std.fmt.bytesToHex(e2_policy_module_sha256, .lower), implementation_sha, shape_sha, source_sha });
}

test "resident hybrid configuration authenticates E2 8128 cohort independently" {
    const a = std.testing.allocator;
    const one = "1111111111111111111111111111111111111111111111111111111111111111";
    const two = "2222222222222222222222222222222222222222222222222222222222222222";
    const three = "3333333333333333333333333333333333333333333333333333333333333333";
    const bytes = try validE2Json(a, one, two, three);
    defer a.free(bytes);
    var configuration_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &configuration_sha256, .{});
    const got = try parse(a, bytes, .{ .configuration_sha256 = configuration_sha256, .implementation_semantic_sha256 = try hex(three), .shape_authority_sha256 = try hex(two), .policy_module_sha256 = e2_policy_module_sha256, .profile_semantic_sha256 = try hex(two) });
    try std.testing.expectEqual(Cohort.e2_8128_v1, got.cohort);
    try std.testing.expectError(error.InvalidResidentHybridConfiguration, parse(a, bytes, .{ .configuration_sha256 = configuration_sha256, .implementation_semantic_sha256 = try hex(three), .shape_authority_sha256 = try hex(two), .policy_module_sha256 = policy_module_sha256, .profile_semantic_sha256 = try hex(two) }));
}

test "resident hybrid configuration binds exact unqualified E4 8128 authority" {
    const a = std.testing.allocator;
    const one = "1111111111111111111111111111111111111111111111111111111111111111";
    const two = "2222222222222222222222222222222222222222222222222222222222222222";
    const three = "3333333333333333333333333333333333333333333333333333333333333333";
    const bytes = try validJson(a, one, two, three);
    defer a.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const got = try parse(a, bytes, .{ .configuration_sha256 = digest, .implementation_semantic_sha256 = try hex(three), .shape_authority_sha256 = try hex(two), .policy_module_sha256 = policy_module_sha256, .profile_semantic_sha256 = try hex(two) });
    try std.testing.expectEqualSlices(u8, &digest, &got.configuration_sha256);
    try std.testing.expectEqualSlices(u8, &(try hex(one)), &got.source_freeze_sha256);
}

test "resident hybrid configuration rejects enrollment qualification and authority mutation" {
    const a = std.testing.allocator;
    const one = "1111111111111111111111111111111111111111111111111111111111111111";
    const two = "2222222222222222222222222222222222222222222222222222222222222222";
    const three = "3333333333333333333333333333333333333333333333333333333333333333";
    const original = try validJson(a, one, two, three);
    defer a.free(original);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(original, &digest, .{});
    const needle = "\"registry_enrollment\":false";
    const at = std.mem.indexOf(u8, original, needle).?;
    var enrolled = try a.dupe(u8, original);
    defer a.free(enrolled);
    @memcpy(enrolled[at + needle.len - 5 .. at + needle.len], "true ");
    try std.testing.expectError(error.InvalidResidentHybridConfiguration, parse(a, enrolled, .{ .configuration_sha256 = digest, .implementation_semantic_sha256 = try hex(three), .shape_authority_sha256 = try hex(two), .policy_module_sha256 = policy_module_sha256, .profile_semantic_sha256 = try hex(two) }));
    var wrong = digest;
    wrong[0] ^= 1;
    try std.testing.expectError(error.InvalidResidentHybridConfiguration, parse(a, original, .{ .configuration_sha256 = wrong, .implementation_semantic_sha256 = try hex(three), .shape_authority_sha256 = try hex(two), .policy_module_sha256 = policy_module_sha256, .profile_semantic_sha256 = try hex(two) }));
}

test "resident hybrid parser is allocation failure atomic" {
    const a = std.testing.allocator;
    const one = "1111111111111111111111111111111111111111111111111111111111111111";
    const two = "2222222222222222222222222222222222222222222222222222222222222222";
    const three = "3333333333333333333333333333333333333333333333333333333333333333";
    const bytes = try validJson(a, one, two, three);
    defer a.free(bytes);
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    const C = struct {
        fn run(allocator: std.mem.Allocator, input: struct { bytes: []const u8, digest: [32]u8 }) !void {
            _ = try parse(allocator, input.bytes, .{ .configuration_sha256 = input.digest, .implementation_semantic_sha256 = try hex(three), .shape_authority_sha256 = try hex(two), .policy_module_sha256 = policy_module_sha256, .profile_semantic_sha256 = try hex(two) });
        }
    };
    try std.testing.checkAllAllocationFailures(a, C.run, .{.{ .bytes = bytes, .digest = digest }});
}

test "formal E2 configuration requires sealed capture and context receipt references" {
    const a = std.testing.allocator;
    const one = "1111111111111111111111111111111111111111111111111111111111111111";
    const two = "2222222222222222222222222222222222222222222222222222222222222222";
    const three = "3333333333333333333333333333333333333333333333333333333333333333";
    const module = std.fmt.bytesToHex(e2_formal_policy_module_sha256, .lower);
    const bytes = try std.fmt.allocPrint(a,
        \\{{"schema":"{s}","cohort":"e2_formal_v1","stage":"resident_native_hybrid_pre_evaluation_configuration","diagnostic_only":true,"candidate_qualification":false,"serving_qualified":false,"release_authority":false,"registry_enrollment":false,"evaluation_opened":false,"frozen_before_unseen_evaluation":true,"evidence_present":false,"training_policy":"{s}","fit_strategy":"identity_chunk_coefficients_adaptive_v18","profile_semantic_sha256":"{s}","policy_module_sha256":"{s}","implementation_semantic_sha256":"{s}","shape_authority_sha256":"{s}","source_rows":7936,"range_count":23,"candidate_count":69,"eligible_heads":[{{"donor_layer":4,"head_index":0}},{{"donor_layer":9,"head_index":0}},{{"donor_layer":14,"head_index":0}}],"resources":{{"per_head_host_workspace_bytes":2147483648,"maximum_transient_host_bytes":4026531840,"process_guard_bytes":8589934592,"fit_workers":1}},"physical_policy":{{"primary_percent":60,"fallback_percent":0,"maximum_fallbacks":0,"planned_exact_heads":1,"minimum_ratio":1.5}},"source":{{"source_freeze":{{"path":"/sealed/source.json","sha256":"{s}"}}}},"formal_owner_inventory":{{"path":"/sealed/inventory.json","sha256":"{s}"}},"formal_context_candidate_receipt":{{"path":"/sealed/context.json","sha256":"{s}"}}}}
    , .{ e2_formal_schema, policy_name, two, &module, three, two, one, one, two });
    defer a.free(bytes);
    var configuration_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &configuration_sha256, .{});
    const got = try parse(a, bytes, .{ .configuration_sha256 = configuration_sha256, .implementation_semantic_sha256 = try hex(three), .shape_authority_sha256 = try hex(two), .policy_module_sha256 = e2_formal_policy_module_sha256, .profile_semantic_sha256 = try hex(two) });
    try std.testing.expectEqual(@as(usize, 7_936), got.e2_fit_source_rows);
    try std.testing.expectEqualSlices(u8, &(try hex(one)), &got.formal_owner_inventory_sha256.?);
    var wrong = configuration_sha256;
    wrong[0] ^= 1;
    try std.testing.expectError(error.InvalidResidentHybridConfiguration, parse(a, bytes, .{ .configuration_sha256 = wrong, .implementation_semantic_sha256 = try hex(three), .shape_authority_sha256 = try hex(two), .policy_module_sha256 = e2_formal_policy_module_sha256, .profile_semantic_sha256 = try hex(two) }));
}
