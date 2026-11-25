// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

const std = @import("std");
const profiles = @import("runtime/kv/compaction_profile.zig");
const fit_policy = @import("runtime/kv/attention_match_fit_policy.zig");

pub const schema = "antfly.context.resident-native-hybrid.compatibility-profile.v1";
pub const e2_schema = "antfly.context.resident-native-hybrid-e2-8128.compatibility-profile.v1";
pub const stage = "resident_native_hybrid_compatibility_profile";
pub const Cohort = enum { e4_8128_v1, e2_8128_v1 };
const Ref = struct { path: []const u8, sha256: []const u8 };
const Identity = struct {
    artifact_sha256: []const u8,
    tokenizer_sha256: []const u8,
    template_sha256: []const u8,
    layout_sha256: []const u8,
    implementation_sha256: []const u8,
};
const Head = struct { head: u32, retained_rows: u32 };
const Owner = struct { layer: u32, heads: []const Head };
pub const Artifact = struct {
    schema: []const u8,
    stage: []const u8,
    diagnostic_only: bool,
    candidate_qualification: bool,
    serving_qualified: bool,
    release_authority: bool,
    registry_enrollment: bool,
    evaluation_opened: bool,
    evidence_present: bool,
    frozen_before_unseen_evaluation: bool,
    training_policy: []const u8,
    fit_strategy: []const u8,
    identity: Identity,
    profile_semantic_sha256: []const u8,
    max_prompt_tokens: u32,
    max_output_tokens: u32,
    owners: []const Owner,
    source: struct { source_freeze: Ref },
};

fn digest(value: []const u8) ![32]u8 {
    if (value.len != 64) return error.InvalidResidentHybridCompatibilityProfile;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, value) catch return error.InvalidResidentHybridCompatibilityProfile;
    const canonical = std.fmt.bytesToHex(out, .lower);
    if (!std.mem.eql(u8, value, &canonical) or std.mem.allEqual(u8, &out, 0)) return error.InvalidResidentHybridCompatibilityProfile;
    return out;
}

pub fn runtimeImplementationDigest() [32]u8 {
    return profiles.implementationDigest(profiles.servingImplementation(.native_whole_head_v2, true, true));
}

pub fn expectedProfileSemantic() [32]u8 {
    const rows = [_]u32{ 512, 512 };
    const owners = [_]profiles.Owner{
        .{ .layer = 5, .retained = &rows },
        .{ .layer = 11, .retained = &rows },
        .{ .layer = 17, .retained = &rows },
        .{ .layer = 23, .retained = &rows },
    };
    const profile: profiles.Profile = .{
        .identity = .{ .artifact = @splat(0), .tokenizer = @splat(0), .template = @splat(0), .layout = @splat(0), .runtime = @splat(0) },
        .engine = .native_whole_head_v2,
        .producer_policy = .cumulative256_exact_cycle_v1,
        .owners = &owners,
        .max_prompt_tokens = 32768,
        .max_output_tokens = 256,
        .minimum_saved_bytes = std.heap.pageSize(),
    };
    return profile.semanticDigest();
}

pub fn expectedE2ProfileSemantic() [32]u8 {
    const rows = [_]u32{512};
    const owners = [_]profiles.Owner{
        .{ .layer = 4, .retained = &rows },
        .{ .layer = 9, .retained = &rows },
        .{ .layer = 14, .retained = &rows },
    };
    const profile: profiles.Profile = .{
        .identity = .{ .artifact = @splat(0), .tokenizer = @splat(0), .template = @splat(0), .layout = @splat(0), .runtime = @splat(0) },
        .engine = .native_whole_head_v2,
        .producer_policy = .cumulative256_exact_cycle_v1,
        .owners = &owners,
        .max_prompt_tokens = 32768,
        .max_output_tokens = 256,
        .minimum_saved_bytes = std.heap.pageSize(),
    };
    return profile.semanticDigest();
}

fn testJson(allocator: std.mem.Allocator, diagnostic: bool, first_layer: u32, runtime: [32]u8, semantic: [32]u8, source: [32]u8, extra: []const u8) ![]u8 {
    const one = "1111111111111111111111111111111111111111111111111111111111111111";
    return std.fmt.allocPrint(allocator,
        \\{{"schema":"{s}","stage":"{s}","diagnostic_only":{},"candidate_qualification":false,"serving_qualified":false,"release_authority":false,"registry_enrollment":false,"evaluation_opened":false,"evidence_present":false,"frozen_before_unseen_evaluation":true,"training_policy":"private_train_challenge_hybrid_resident_v1","fit_strategy":"identity_chunk_coefficients_adaptive_v18","identity":{{"artifact_sha256":"{s}","tokenizer_sha256":"{s}","template_sha256":"{s}","layout_sha256":"{s}","implementation_sha256":"{s}"}},"profile_semantic_sha256":"{s}","max_prompt_tokens":32768,"max_output_tokens":256,"owners":[{{"layer":{},"heads":[{{"head":0,"retained_rows":512}},{{"head":1,"retained_rows":512}}]}},{{"layer":11,"heads":[{{"head":0,"retained_rows":512}},{{"head":1,"retained_rows":512}}]}},{{"layer":17,"heads":[{{"head":0,"retained_rows":512}},{{"head":1,"retained_rows":512}}]}},{{"layer":23,"heads":[{{"head":0,"retained_rows":512}},{{"head":1,"retained_rows":512}}]}}],"source":{{"source_freeze":{{"path":"/sealed/source.json","sha256":"{s}"}}}}{s}}}
    , .{ schema, stage, diagnostic, one, one, one, one, std.fmt.bytesToHex(runtime, .lower), std.fmt.bytesToHex(semantic, .lower), first_layer, std.fmt.bytesToHex(source, .lower), extra });
}

fn testE2Json(allocator: std.mem.Allocator, runtime: [32]u8, semantic: [32]u8, source: [32]u8) ![]u8 {
    const one = "1111111111111111111111111111111111111111111111111111111111111111";
    return std.fmt.allocPrint(allocator,
        \\{{"schema":"{s}","stage":"{s}","diagnostic_only":true,"candidate_qualification":false,"serving_qualified":false,"release_authority":false,"registry_enrollment":false,"evaluation_opened":false,"evidence_present":false,"frozen_before_unseen_evaluation":true,"training_policy":"private_train_challenge_hybrid_resident_v1","fit_strategy":"identity_chunk_coefficients_adaptive_v18","identity":{{"artifact_sha256":"{s}","tokenizer_sha256":"{s}","template_sha256":"{s}","layout_sha256":"{s}","implementation_sha256":"{s}"}},"profile_semantic_sha256":"{s}","max_prompt_tokens":32768,"max_output_tokens":256,"owners":[{{"layer":4,"heads":[{{"head":0,"retained_rows":512}}]}},{{"layer":9,"heads":[{{"head":0,"retained_rows":512}}]}},{{"layer":14,"heads":[{{"head":0,"retained_rows":512}}]}}],"source":{{"source_freeze":{{"path":"/sealed/source.json","sha256":"{s}"}}}}}}
    , .{ e2_schema, stage, one, one, one, one, std.fmt.bytesToHex(runtime, .lower), std.fmt.bytesToHex(semantic, .lower), std.fmt.bytesToHex(source, .lower) });
}

test "resident compatibility profile binds E2 physical owners separately from E4" {
    const source: [32]u8 = @splat(7);
    const bytes = try testE2Json(std.testing.allocator, runtimeImplementationDigest(), expectedE2ProfileSemantic(), source);
    defer std.testing.allocator.free(bytes);
    var loaded = try parseBytes(std.testing.allocator, bytes);
    defer loaded.deinit();
    try std.testing.expectEqual(Cohort.e2_8128_v1, loaded.cohort);
    try std.testing.expectEqual(@as(usize, 3), loaded.profile.owners.len);
    try std.testing.expectEqual(@as(u32, 14), loaded.profile.owners[2].layer);
    try std.testing.expectEqual(@as(usize, 1), loaded.profile.owners[2].retained.len);
    const semantic = expectedE2ProfileSemantic();
    try std.testing.expectEqualSlices(u8, &semantic, &loaded.profile_semantic_sha256);
}

test "resident compatibility profile parses only the unqualified fixed shape" {
    const a = std.testing.allocator;
    const source: [32]u8 = @splat(7);
    const expected_semantic = expectedProfileSemantic();
    const bytes = try testJson(a, true, 5, runtimeImplementationDigest(), expected_semantic, source, "");
    defer a.free(bytes);
    var loaded = try parseBytes(a, bytes);
    defer loaded.deinit();
    try std.testing.expectEqualSlices(u8, &expected_semantic, &loaded.profile_semantic_sha256);
    try std.testing.expectEqualSlices(u8, &source, &loaded.source_freeze_sha256);
    try std.testing.expectEqual(@as(usize, 4), loaded.profile.owners.len);
    try std.testing.expectEqual(@as(u32, 512), loaded.profile.owners[3].retained[1]);

    const wrong_owner = try testJson(a, true, 6, runtimeImplementationDigest(), expectedProfileSemantic(), source, "");
    defer a.free(wrong_owner);
    try std.testing.expectError(error.InvalidResidentHybridCompatibilityProfile, parseBytes(a, wrong_owner));
    const qualified = try testJson(a, false, 5, runtimeImplementationDigest(), expectedProfileSemantic(), source, "");
    defer a.free(qualified);
    try std.testing.expectError(error.InvalidResidentHybridCompatibilityProfile, parseBytes(a, qualified));
    var wrong_runtime = runtimeImplementationDigest();
    wrong_runtime[0] ^= 1;
    const runtime_mutation = try testJson(a, true, 5, wrong_runtime, expectedProfileSemantic(), source, "");
    defer a.free(runtime_mutation);
    try std.testing.expectError(error.InvalidResidentHybridCompatibilityProfile, parseBytes(a, runtime_mutation));
    var wrong_semantic = expectedProfileSemantic();
    wrong_semantic[0] ^= 1;
    const semantic_mutation = try testJson(a, true, 5, runtimeImplementationDigest(), wrong_semantic, source, "");
    defer a.free(semantic_mutation);
    try std.testing.expectError(error.InvalidResidentHybridCompatibilityProfile, parseBytes(a, semantic_mutation));
    const invented_metric = try testJson(a, true, 5, runtimeImplementationDigest(), expectedProfileSemantic(), source, ",\"max_mass_error\":0.01");
    defer a.free(invented_metric);
    try std.testing.expectError(error.InvalidResidentHybridCompatibilityProfile, parseBytes(a, invented_metric));
    const zero_source: [32]u8 = @splat(0);
    const source_mutation = try testJson(a, true, 5, runtimeImplementationDigest(), expectedProfileSemantic(), zero_source, "");
    defer a.free(source_mutation);
    try std.testing.expectError(error.InvalidResidentHybridCompatibilityProfile, parseBytes(a, source_mutation));
}

test "resident compatibility profile allocation failure is atomic" {
    const a = std.testing.allocator;
    const source: [32]u8 = @splat(7);
    const bytes = try testJson(a, true, 5, runtimeImplementationDigest(), expectedProfileSemantic(), source, "");
    defer a.free(bytes);
    const C = struct {
        fn run(allocator: std.mem.Allocator, input: []const u8) !void {
            var loaded = try parseBytes(allocator, input);
            loaded.deinit();
        }
    };
    try std.testing.checkAllAllocationFailures(a, C.run, .{bytes});
}

pub const Loaded = struct {
    allocator: std.mem.Allocator,
    parsed: std.json.Parsed(Artifact),
    owners: []profiles.Owner,
    retained: []u32,
    profile: profiles.Profile,
    profile_semantic_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    cohort: Cohort,
    pub fn deinit(self: *Loaded) void {
        self.allocator.free(self.retained);
        self.allocator.free(self.owners);
        self.parsed.deinit();
        self.* = undefined;
    }
};

pub fn parseBytes(allocator: std.mem.Allocator, bytes: []const u8) !Loaded {
    var artifact_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &artifact_sha, .{});
    var parsed = std.json.parseFromSlice(Artifact, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false }) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return error.InvalidResidentHybridCompatibilityProfile,
    };
    errdefer parsed.deinit();
    const value = parsed.value;
    const cohort: Cohort = if (std.mem.eql(u8, value.schema, schema)) .e4_8128_v1 else if (std.mem.eql(u8, value.schema, e2_schema)) .e2_8128_v1 else return error.InvalidResidentHybridCompatibilityProfile;
    const e2 = cohort == .e2_8128_v1;
    const owner_count: usize = if (e2) 3 else 4;
    const retained_count: usize = if (e2) 3 else 8;
    if (!std.mem.eql(u8, value.stage, stage) or
        !value.diagnostic_only or value.candidate_qualification or value.serving_qualified or value.release_authority or
        value.registry_enrollment or value.evaluation_opened or value.evidence_present or !value.frozen_before_unseen_evaluation or
        !std.mem.eql(u8, value.training_policy, "private_train_challenge_hybrid_resident_v1") or
        !std.mem.eql(u8, value.fit_strategy, "identity_chunk_coefficients_adaptive_v18") or
        value.max_prompt_tokens != 32768 or value.max_output_tokens != 256 or value.owners.len != owner_count or
        !std.fs.path.isAbsolute(value.source.source_freeze.path))
        return error.InvalidResidentHybridCompatibilityProfile;
    const runtime = try digest(value.identity.implementation_sha256);
    if (!std.mem.eql(u8, &runtime, &runtimeImplementationDigest())) return error.InvalidResidentHybridCompatibilityProfile;
    const owners = try allocator.alloc(profiles.Owner, owner_count);
    errdefer allocator.free(owners);
    const retained = try allocator.alloc(u32, retained_count);
    errdefer allocator.free(retained);
    for (value.owners, 0..) |owner, i| {
        const expected_layer: u32 = if (e2) switch (i) {
            0 => 4,
            1 => 9,
            2 => 14,
            else => unreachable,
        } else switch (i) {
            0 => 5,
            1 => 11,
            2 => 17,
            3 => 23,
            else => unreachable,
        };
        const heads: usize = if (e2) 1 else 2;
        if (owner.layer != expected_layer or owner.heads.len != heads) return error.InvalidResidentHybridCompatibilityProfile;
        for (owner.heads, 0..) |head, j| {
            if (head.head != j or head.retained_rows != 512) return error.InvalidResidentHybridCompatibilityProfile;
            retained[i * heads + j] = head.retained_rows;
        }
        owners[i] = .{ .layer = owner.layer, .retained = retained[i * heads ..][0..heads] };
    }
    const profile: profiles.Profile = .{
        .identity = .{
            .artifact = try digest(value.identity.artifact_sha256),
            .tokenizer = try digest(value.identity.tokenizer_sha256),
            .template = try digest(value.identity.template_sha256),
            .layout = try digest(value.identity.layout_sha256),
            .runtime = runtime,
        },
        .engine = .native_whole_head_v2,
        .producer_policy = fit_policy.Policy.cumulative256_exact_cycle_v1,
        .artifact_sha256 = artifact_sha,
        .owners = owners,
        .query_protocols = &.{},
        .max_prompt_tokens = value.max_prompt_tokens,
        .max_output_tokens = value.max_output_tokens,
        .minimum_saved_bytes = std.heap.pageSize(),
    };
    const semantic = profile.semanticDigest();
    const expected_semantic = if (e2) expectedE2ProfileSemantic() else expectedProfileSemantic();
    if (!std.mem.eql(u8, &semantic, &expected_semantic)) return error.InvalidResidentHybridCompatibilityProfile;
    if (!std.mem.eql(u8, &semantic, &try digest(value.profile_semantic_sha256))) return error.InvalidResidentHybridCompatibilityProfile;
    return .{
        .allocator = allocator,
        .parsed = parsed,
        .owners = owners,
        .retained = retained,
        .profile = profile,
        .profile_semantic_sha256 = semantic,
        .source_freeze_sha256 = try digest(value.source.source_freeze.sha256),
        .cohort = cohort,
    };
}

// The build receipt retains this executed observation. It is identity
// evidence only and does not open a model or grant qualification credit.
test "context resident compatibility compiled identity observation" {
    std.debug.print("RESIDENT_COMPATIBILITY_COMPILED_IDENTITY {{\"schema\":\"antfly.resident-hybrid-compiled-identity.v1\",\"compaction_identity_sha256\":\"{s}\",\"native_profile_implementation_sha256\":\"{s}\",\"compatibility_profile_semantic_sha256\":\"{s}\",\"allocation_page_bytes\":{d}}}\n", .{
        std.fmt.bytesToHex(@import("compaction_identity").digest, .lower),
        std.fmt.bytesToHex(runtimeImplementationDigest(), .lower),
        std.fmt.bytesToHex(expectedProfileSemantic(), .lower),
        std.heap.pageSize(),
    });
}
