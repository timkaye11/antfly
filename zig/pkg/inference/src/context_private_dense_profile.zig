//! Strict ingress for a staged-only adaptive-v18/private-Dense512 authority.
//! This format is deliberately not accepted by canonical native-repeat or any
//! enrolled/trusted producer path.
const std = @import("std");
const profiles = @import("runtime/kv/compaction_profile.zig");
const fit_policy = @import("runtime/kv/attention_match_fit_policy.zig");
const future_policy = @import("runtime/kv/attention_match_future_decode_training.zig");

pub const schema = "antfly.context.private-dense512.staged-profile.v1";
pub const stage = "private_dense512_adaptive_v18_candidate";
pub const e2_schema = "antfly.context.private-dense512-e2.staged-profile.v1";
pub const e2_stage = "private_dense512_e2_adaptive_v18_candidate";
pub const qualified_resident_schema = "antfly.context.qualified-resident-e4.bounded-profile.v1";
pub const qualified_resident_stage = "qualified_resident_e4_bounded_inactive";

const Ref = struct { path: []const u8, sha256: []const u8 };
const Identity = struct {
    artifact_sha256: []const u8,
    tokenizer_sha256: []const u8,
    template_sha256: []const u8,
    layout_sha256: []const u8,
    implementation_sha256: []const u8,
};
const Head = struct {
    donor: u32,
    head: u32,
    outcome: enum { compacted, exact_identity },
    source_rows: u32,
    candidate_rows: u32,
    source_head_sha256: []const u8,
    source_index_map_sha256: []const u8,
    candidate_sha256: ?[]const u8 = null,
    max_mass_error: ?f64 = null,
    output_rmse: ?f64 = null,
};
const Owner = struct { layer: u32, heads: []const Head };
pub const Artifact = struct {
    schema: []const u8,
    stage: []const u8,
    diagnostic_only: bool,
    candidate_qualification: bool,
    serving_qualified: bool,
    release_authority: bool,
    evaluation_opened: bool,
    frozen_before_unseen_evaluation: bool,
    training_policy: []const u8,
    /// Present only on the new future-mixture profile. The old profile parses
    /// byte-for-byte as before and is never relabelled.
    parent_private_dense_profile_sha256: ?[]const u8 = null,
    parent_private_dense_profile_json: ?[]const u8 = null,
    training_mix_policy_sha256: ?[]const u8 = null,
    prepare_host_owner_capability: profiles.PrepareHostOwnerCapability = .none,
    prepare_host_owner_physical_bytes: ?u64 = null,
    resident_configuration_sha256: ?[]const u8 = null,
    resident_source_freeze_sha256: ?[]const u8 = null,
    resident_shape_authority_sha256: ?[]const u8 = null,
    resident_policy_module_sha256: ?[]const u8 = null,
    resident_training_policy_sha256: ?[]const u8 = null,
    base_compatibility_profile_sha256: ?[]const u8 = null,
    base_compatibility_profile_semantic_sha256: ?[]const u8 = null,
    base_compatibility_profile_path: ?[]const u8 = null,
    resident_configuration_path: ?[]const u8 = null,
    encoding: ?enum { f16 } = null,
    logical_tokens: ?u32 = null,
    fit_workers: ?u32 = null,
    resume_allowed: ?bool = null,
    compiled_prefill_max_copy_bytes: ?u64 = null,
    compiled_prefill_max_host_staging_bytes: ?u64 = null,
    compiled_prefill_codec_scope: ?enum { f16_only } = null,
    compiled_prefill_max_logical_tokens: ?u32 = null,
    compiled_prefill_layout_policy: ?enum { full_history_exact } = null,
    fit_strategy: []const u8,
    identity: Identity,
    main_gguf_sha256: ?[]const u8 = null,
    profile_semantic_sha256: []const u8,
    max_prompt_tokens: u32,
    max_output_tokens: u32,
    conservative_owner_rows_per_head: u32,
    /// Required only by the E2 family. These are page-rounded prediction
    /// evidence, never a substitute for the post-materialization gate.
    ordinary_source_bytes: ?u64 = null,
    selected_page_rounded_prediction_bytes: ?u64 = null,
    policy: struct {
        initial_global_physical_ratio: f64,
        post_materialization_global_physical_ratio: f64,
        max_mass_error: f64,
        max_output_rmse: f64,
        physical_postcheck_required: bool,
        geometry_budget_primary_percent: u8,
        geometry_budget_fallback_percent: u8,
        max_fallbacks: u8,
    },
    source: struct {
        primary_capture: Ref,
        dense512_capture: Ref,
        sampler_recipe: Ref,
        calibration_campaign: Ref,
        full_cli_release_receipt: Ref,
        source_freeze: Ref,
    },
    owners: []const Owner,
};

fn validPolicy(p: @FieldType(Artifact, "policy")) bool {
    return p.initial_global_physical_ratio == 1.5 and
        p.post_materialization_global_physical_ratio == 1.000001 and
        p.max_mass_error == 0.05 and p.max_output_rmse == 0.05 and
        p.physical_postcheck_required and p.geometry_budget_primary_percent == 91 and
        p.geometry_budget_fallback_percent == 95 and p.max_fallbacks == 3;
}

fn validCompactedMetrics(mass: f64, output: f64, p: @FieldType(Artifact, "policy")) bool {
    return std.math.isFinite(mass) and std.math.isFinite(output) and
        mass >= 0 and output >= 0 and mass <= p.max_mass_error and output <= p.max_output_rmse;
}

fn digest(text: []const u8) ![32]u8 {
    if (text.len != 64) return error.InvalidPrivateDenseProfile;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, text) catch return error.InvalidPrivateDenseProfile;
    const canonical = std.fmt.bytesToHex(out, .lower);
    if (!std.mem.eql(u8, text, &canonical) or std.mem.allEqual(u8, &out, 0)) return error.InvalidPrivateDenseProfile;
    return out;
}
fn validRef(ref: Ref) bool {
    if (!std.fs.path.isAbsolute(ref.path)) return false;
    _ = digest(ref.sha256) catch return false;
    return true;
}

fn identityMapDigest(rows: u32) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    var bytes: [8]u8 = undefined;
    for (0..rows) |i| {
        std.mem.writeInt(u64, &bytes, @intCast(i), .little);
        hash.update(&bytes);
    }
    return hash.finalResult();
}

fn validIdentityCoordinate(head: Head) bool {
    return head.donor == 5 and head.head == 0;
}

fn roundedPage(bytes: u64) !u64 {
    const page: u64 = @intCast(std.heap.pageSize());
    return try std.math.mul(u64, try std.math.divCeil(u64, bytes, page), page);
}

fn e2HeadBytes(rows: u32) !u64 {
    const row_count: u64 = rows;
    const values = try std.math.mul(u64, row_count, 512 * 2);
    const metadata = try std.math.mul(u64, row_count, 20);
    return try std.math.add(u64, try std.math.mul(u64, 2, try roundedPage(values)), try roundedPage(metadata));
}

pub const Loaded = struct {
    allocator: std.mem.Allocator,
    parsed: std.json.Parsed(Artifact),
    owners: []profiles.Owner,
    retained: []u32,
    profile: profiles.Profile,
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
    var parsed = try std.json.parseFromSlice(Artifact, allocator, bytes, .{ .allocate = .alloc_always, .ignore_unknown_fields = false });
    errdefer parsed.deinit();
    const value = parsed.value;
    const e2 = std.mem.eql(u8, value.schema, e2_schema) and std.mem.eql(u8, value.stage, e2_stage);
    const e4 = std.mem.eql(u8, value.schema, schema) and std.mem.eql(u8, value.stage, stage);
    const qualified_resident = std.mem.eql(u8, value.schema, qualified_resident_schema) and std.mem.eql(u8, value.stage, qualified_resident_stage);
    if ((!e4 and !e2 and !qualified_resident) or
        !value.diagnostic_only or !value.candidate_qualification or value.serving_qualified or value.release_authority or
        value.evaluation_opened or !value.frozen_before_unseen_evaluation or
        (!std.mem.eql(u8, value.training_policy, "private_dense512_reference_v1") and
            !std.mem.eql(u8, value.training_policy, "private_future_decode_mix_v1") and
            !std.mem.eql(u8, value.training_policy, "private_train_challenge_hybrid_resident_v1")) or
        !std.mem.eql(u8, value.fit_strategy, "identity_chunk_coefficients_adaptive_v18") or
        value.max_prompt_tokens != @as(u32, if (e2 or qualified_resident) 8192 else 32768) or value.max_output_tokens != @as(u32, if (qualified_resident) 64 else 256) or
        value.conservative_owner_rows_per_head != 512 or value.owners.len != @as(usize, if (e2) 3 else 4))
        return error.InvalidPrivateDenseProfile;
    const resident_training = std.mem.eql(u8, value.training_policy, "private_train_challenge_hybrid_resident_v1");
    if (qualified_resident != resident_training) return error.InvalidPrivateDenseProfile;
    if (qualified_resident) {
        if (value.prepare_host_owner_capability != .bounded_page_e4_resident_1152m_v1 or
            value.prepare_host_owner_physical_bytes != profiles.bounded_page_e4_resident_1152m_bytes)
            return error.InvalidPrivateDenseProfile;
        if (!std.fs.path.isAbsolute(value.base_compatibility_profile_path orelse return error.InvalidPrivateDenseProfile) or
            !std.fs.path.isAbsolute(value.resident_configuration_path orelse return error.InvalidPrivateDenseProfile)) return error.InvalidPrivateDenseProfile;
        if (value.encoding != .f16 or value.logical_tokens != 8128 or value.fit_workers != 1 or value.resume_allowed != false or
            value.compiled_prefill_max_copy_bytes != 4 * 1024 * 1024 * 1024 or value.compiled_prefill_max_host_staging_bytes != 256 * 1024 * 1024 or
            value.compiled_prefill_codec_scope != .f16_only or value.compiled_prefill_max_logical_tokens != 8192 or
            value.compiled_prefill_layout_policy != .full_history_exact) return error.InvalidPrivateDenseProfile;
        inline for (.{ value.base_compatibility_profile_sha256, value.base_compatibility_profile_semantic_sha256, value.resident_training_policy_sha256, value.resident_configuration_sha256, value.resident_source_freeze_sha256, value.resident_shape_authority_sha256, value.resident_policy_module_sha256 }) |field|
            _ = try digest(field orelse return error.InvalidPrivateDenseProfile);
        const expected_module = "e0a407063023d89c40f516270081012329a8981dc5c64de2dd8a985d8469afee";
        if (!std.mem.eql(u8, value.resident_policy_module_sha256.?, expected_module)) return error.InvalidPrivateDenseProfile;
        if (!std.mem.eql(u8, &try digest(value.resident_training_policy_sha256.?), &@import("runtime/kv/attention_match_resident_native_hybrid_profile.zig").digest())) return error.InvalidPrivateDenseProfile;
    } else if (value.prepare_host_owner_capability != .none or value.prepare_host_owner_physical_bytes != null or
        value.base_compatibility_profile_sha256 != null or value.base_compatibility_profile_semantic_sha256 != null or value.base_compatibility_profile_path != null or value.resident_configuration_path != null or value.resident_training_policy_sha256 != null or value.resident_configuration_sha256 != null or value.resident_source_freeze_sha256 != null or
        value.resident_shape_authority_sha256 != null or value.resident_policy_module_sha256 != null or value.encoding != null or
        value.logical_tokens != null or value.fit_workers != null or value.resume_allowed != null or value.compiled_prefill_max_copy_bytes != null or
        value.compiled_prefill_max_host_staging_bytes != null or value.compiled_prefill_codec_scope != null or
        value.compiled_prefill_max_logical_tokens != null or value.compiled_prefill_layout_policy != null)
        return error.InvalidPrivateDenseProfile;
    const future_mix = std.mem.eql(u8, value.training_policy, "private_future_decode_mix_v1");
    if (e2 and future_mix) return error.InvalidPrivateDenseProfile;
    if (e2) {
        const expected = .{
            "0394d21a892ba0194e9bb6797eefc1846bf2d1df1cff86a26d24c45805efd163",
            "32e685c6648d25b7f696d1af0a4c16a58f8d8cad7123b2e48c81aa5284b6d1ab",
            "611d5f75642790e55749a538066e24a7f13b6a1b58697c855ed078a148ee5dbb",
            "b61587d2700c7e3d1c1b014ea0e0a7f7a47bd93ea1237972e88084d5683511e4",
            "fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634",
        };
        if (!std.mem.eql(u8, value.identity.artifact_sha256, expected[0]) or
            !std.mem.eql(u8, value.identity.tokenizer_sha256, expected[1]) or
            !std.mem.eql(u8, value.identity.template_sha256, expected[2]) or
            !std.mem.eql(u8, value.identity.layout_sha256, expected[3]) or
            value.main_gguf_sha256 == null or !std.mem.eql(u8, value.main_gguf_sha256.?, expected[4]))
            return error.InvalidPrivateDenseProfile;
    } else if (value.main_gguf_sha256 != null) return error.InvalidPrivateDenseProfile;
    if (future_mix) {
        const parent = value.parent_private_dense_profile_sha256 orelse return error.InvalidPrivateDenseProfile;
        const parent_json = value.parent_private_dense_profile_json orelse return error.InvalidPrivateDenseProfile;
        const policy = value.training_mix_policy_sha256 orelse return error.InvalidPrivateDenseProfile;
        const expected_parent = try digest(parent);
        var actual_parent: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(parent_json, &actual_parent, .{});
        if (!std.mem.eql(u8, &expected_parent, &actual_parent)) return error.InvalidPrivateDenseProfile;
        var loaded_parent = try parseBytes(allocator, parent_json);
        defer loaded_parent.deinit();
        if (!std.mem.eql(u8, loaded_parent.parsed.value.training_policy, "private_dense512_reference_v1") or
            loaded_parent.parsed.value.parent_private_dense_profile_sha256 != null or
            loaded_parent.parsed.value.parent_private_dense_profile_json != null or
            loaded_parent.parsed.value.training_mix_policy_sha256 != null)
            return error.InvalidPrivateDenseProfile;
        if (!std.mem.eql(u8, &try digest(policy), &future_policy.policyDigest())) return error.InvalidPrivateDenseProfile;
    } else if (value.parent_private_dense_profile_sha256 != null or value.parent_private_dense_profile_json != null or value.training_mix_policy_sha256 != null) {
        return error.InvalidPrivateDenseProfile;
    }
    const p = value.policy;
    if (!validPolicy(p))
        return error.InvalidPrivateDenseProfile;
    inline for (.{ value.source.primary_capture, value.source.dense512_capture, value.source.sampler_recipe, value.source.calibration_campaign, value.source.full_cli_release_receipt, value.source.source_freeze }) |ref|
        if (!validRef(ref)) return error.InvalidPrivateDenseProfile;

    const e4_layers = [_]u32{ 5, 11, 17, 23 };
    const e2_layers = [_]u32{ 4, 9, 14 };
    var identity_count: usize = 0;
    var compacted_count: usize = 0;
    var predicted: u64 = 0;
    for (value.owners, 0..) |owner, owner_index| {
        const expected_layer = if (e2) e2_layers[owner_index] else e4_layers[owner_index];
        const expected_heads: usize = if (e2) 1 else 2;
        if (owner.layer != expected_layer or owner.heads.len != expected_heads) return error.InvalidPrivateDenseProfile;
        for (owner.heads, 0..) |head, expected_head| {
            if (head.donor != owner.layer or head.head != expected_head or head.source_rows != 7936) return error.InvalidPrivateDenseProfile;
            _ = try digest(head.source_head_sha256);
            _ = try digest(head.source_index_map_sha256);
            switch (head.outcome) {
                .exact_identity => {
                    identity_count += 1;
                    const map_digest = try digest(head.source_index_map_sha256);
                    const expected_map_digest = identityMapDigest(head.source_rows);
                    if ((!e2 and !validIdentityCoordinate(head)) or head.candidate_rows != head.source_rows or
                        !std.mem.eql(u8, &map_digest, &expected_map_digest) or
                        head.candidate_sha256 != null or
                        head.max_mass_error != null or head.output_rmse != null)
                        return error.InvalidPrivateDenseProfile;
                },
                .compacted => {
                    compacted_count += 1;
                    if (head.candidate_rows == 0 or head.candidate_rows >= head.source_rows or head.candidate_sha256 == null or
                        head.max_mass_error == null or head.output_rmse == null or
                        !validCompactedMetrics(head.max_mass_error.?, head.output_rmse.?, p))
                        return error.InvalidPrivateDenseProfile;
                    _ = try digest(head.candidate_sha256.?);
                },
            }
            if (e2) predicted = try std.math.add(u64, predicted, try e2HeadBytes(head.candidate_rows));
        }
    }
    if (e2) {
        const source_bytes: u64 = 48_758_784;
        if (identity_count + compacted_count != 3 or value.ordinary_source_bytes != source_bytes or
            value.selected_page_rounded_prediction_bytes != predicted or
            try std.math.mul(u64, predicted, 3) > try std.math.mul(u64, source_bytes, 2))
            return error.InvalidPrivateDenseProfile;
    } else if (identity_count != 1 or compacted_count != 7 or
        value.ordinary_source_bytes != null or value.selected_page_rounded_prediction_bytes != null)
        return error.InvalidPrivateDenseProfile;
    const implementation = profiles.servingImplementation(.native_whole_head_v2, true, true);
    const implementation_sha = profiles.implementationDigest(implementation);
    if (!std.mem.eql(u8, &implementation_sha, &try digest(value.identity.implementation_sha256))) return error.InvalidPrivateDenseProfile;

    const retained_count: usize = if (e2) 3 else 8;
    const retained = try allocator.alloc(u32, retained_count);
    errdefer allocator.free(retained);
    @memset(retained, 512);
    const owner_count: usize = if (e2) 3 else 4;
    const owners = try allocator.alloc(profiles.Owner, owner_count);
    errdefer allocator.free(owners);
    for (owners, 0..) |*owner, i| {
        const layer = if (e2) e2_layers[i] else e4_layers[i];
        const heads: usize = if (e2) 1 else 2;
        owner.* = .{ .layer = layer, .retained = retained[i * heads ..][0..heads] };
    }
    const profile = profiles.Profile{
        .identity = .{ .artifact = try digest(value.identity.artifact_sha256), .tokenizer = try digest(value.identity.tokenizer_sha256), .template = try digest(value.identity.template_sha256), .layout = try digest(value.identity.layout_sha256), .runtime = implementation_sha },
        .engine = .native_whole_head_v2,
        .producer_policy = fit_policy.Policy.cumulative256_exact_cycle_v1,
        .prepare_host_owner_capability = value.prepare_host_owner_capability,
        .resident_authority = if (qualified_resident) .{
            .cohort = .e4_8128_v1,
            .base_compatibility_artifact_sha256 = try digest(value.base_compatibility_profile_sha256.?),
            .base_compatibility_semantic_sha256 = try digest(value.base_compatibility_profile_semantic_sha256.?),
            .training_policy_sha256 = try digest(value.resident_training_policy_sha256.?),
            .configuration_sha256 = try digest(value.resident_configuration_sha256.?),
            .source_freeze_sha256 = try digest(value.resident_source_freeze_sha256.?),
            .shape_authority_sha256 = try digest(value.resident_shape_authority_sha256.?),
            .policy_module_sha256 = try digest(value.resident_policy_module_sha256.?),
            .encoding = .f16,
            .logical_tokens = value.logical_tokens.?,
            .max_output_tokens = value.max_output_tokens,
            .fit_workers = value.fit_workers.?,
            .resume_allowed = value.resume_allowed.?,
            .compiled_max_copy_bytes = value.compiled_prefill_max_copy_bytes.?,
            .compiled_max_host_staging_bytes = value.compiled_prefill_max_host_staging_bytes.?,
            .compiled_codec_scope = .f16_only,
            .compiled_max_logical_tokens = value.compiled_prefill_max_logical_tokens.?,
            .compiled_layout_policy = .full_history_exact,
        } else null,
        .artifact_sha256 = artifact_sha,
        .owners = owners,
        .query_protocols = &.{},
        .max_prompt_tokens = value.max_prompt_tokens,
        .max_output_tokens = value.max_output_tokens,
        // Admission is governed by the separately authenticated dynamic policy
        // and its mandatory post-materialization physical check. This field is
        // a nonzero page-aligned compatibility floor, not a savings promise.
        .minimum_saved_bytes = std.heap.pageSize(),
    };
    if (!std.mem.eql(u8, &profile.semanticDigest(), &try digest(value.profile_semantic_sha256))) return error.InvalidPrivateDenseProfile;
    if (future_mix) {
        var loaded_parent = try parseBytes(allocator, value.parent_private_dense_profile_json.?);
        defer loaded_parent.deinit();
        if (!std.meta.eql(profile.identity, loaded_parent.profile.identity) or
            !std.mem.eql(u8, &profile.semanticDigest(), &loaded_parent.profile.semanticDigest()))
            return error.InvalidPrivateDenseProfile;
    }
    return .{ .allocator = allocator, .parsed = parsed, .owners = owners, .retained = retained, .profile = profile };
}

/// Test-only factory used by the cross-file B/C/P ingress fixture. It emits
/// the real P schema and passes through the same parser as production input;
/// it grants no serving qualification or enrollment.
pub fn testQualifiedResidentJson(
    allocator: std.mem.Allocator,
    base_path: []const u8,
    config_path: []const u8,
    base_artifact: [32]u8,
    base_semantic: [32]u8,
    configuration: [32]u8,
    source_freeze: [32]u8,
    shape: [32]u8,
) ![]u8 {
    const one = "1111111111111111111111111111111111111111111111111111111111111111";
    const identity_map = std.fmt.bytesToHex(identityMapDigest(7936), .lower);
    const runtime = profiles.implementationDigest(profiles.servingImplementation(.native_whole_head_v2, true, true));
    const training = @import("runtime/kv/attention_match_resident_native_hybrid_profile.zig").digest();
    var heads: [8]Head = undefined;
    for (&heads, 0..) |*head, i| head.* = .{
        .donor = ([_]u32{ 5, 5, 11, 11, 17, 17, 23, 23 })[i],
        .head = @intCast(i % 2),
        .outcome = if (i == 0) .exact_identity else .compacted,
        .source_rows = 7936,
        .candidate_rows = if (i == 0) 7936 else 5000,
        .source_head_sha256 = one,
        .source_index_map_sha256 = if (i == 0) &identity_map else one,
        .candidate_sha256 = if (i == 0) null else one,
        .max_mass_error = if (i == 0) null else 0.01,
        .output_rmse = if (i == 0) null else 0.01,
    };
    const owners = [_]Owner{
        .{ .layer = 5, .heads = heads[0..2] },  .{ .layer = 11, .heads = heads[2..4] },
        .{ .layer = 17, .heads = heads[4..6] }, .{ .layer = 23, .heads = heads[6..8] },
    };
    const ref: Ref = .{ .path = "/fixture", .sha256 = one };
    var artifact: Artifact = .{
        .schema = qualified_resident_schema,
        .stage = qualified_resident_stage,
        .diagnostic_only = true,
        .candidate_qualification = true,
        .serving_qualified = false,
        .release_authority = false,
        .evaluation_opened = false,
        .frozen_before_unseen_evaluation = true,
        .training_policy = "private_train_challenge_hybrid_resident_v1",
        .prepare_host_owner_capability = .bounded_page_e4_resident_1152m_v1,
        .prepare_host_owner_physical_bytes = profiles.bounded_page_e4_resident_1152m_bytes,
        .resident_configuration_sha256 = &std.fmt.bytesToHex(configuration, .lower),
        .resident_source_freeze_sha256 = &std.fmt.bytesToHex(source_freeze, .lower),
        .resident_shape_authority_sha256 = &std.fmt.bytesToHex(shape, .lower),
        .resident_policy_module_sha256 = &std.fmt.bytesToHex(@import("context_resident_hybrid_staged_configuration.zig").policy_module_sha256, .lower),
        .resident_training_policy_sha256 = &std.fmt.bytesToHex(training, .lower),
        .base_compatibility_profile_sha256 = &std.fmt.bytesToHex(base_artifact, .lower),
        .base_compatibility_profile_semantic_sha256 = &std.fmt.bytesToHex(base_semantic, .lower),
        .base_compatibility_profile_path = base_path,
        .resident_configuration_path = config_path,
        .encoding = .f16,
        .logical_tokens = 8128,
        .fit_workers = 1,
        .resume_allowed = false,
        .compiled_prefill_max_copy_bytes = 4 * 1024 * 1024 * 1024,
        .compiled_prefill_max_host_staging_bytes = 256 * 1024 * 1024,
        .compiled_prefill_codec_scope = .f16_only,
        .compiled_prefill_max_logical_tokens = 8192,
        .compiled_prefill_layout_policy = .full_history_exact,
        .fit_strategy = "identity_chunk_coefficients_adaptive_v18",
        .identity = .{ .artifact_sha256 = one, .tokenizer_sha256 = one, .template_sha256 = one, .layout_sha256 = one, .implementation_sha256 = &std.fmt.bytesToHex(runtime, .lower) },
        .profile_semantic_sha256 = one,
        .max_prompt_tokens = 8192,
        .max_output_tokens = 64,
        .conservative_owner_rows_per_head = 512,
        .policy = .{ .initial_global_physical_ratio = 1.5, .post_materialization_global_physical_ratio = 1.000001, .max_mass_error = 0.05, .max_output_rmse = 0.05, .physical_postcheck_required = true, .geometry_budget_primary_percent = 91, .geometry_budget_fallback_percent = 95, .max_fallbacks = 3 },
        .source = .{ .primary_capture = ref, .dense512_capture = ref, .sampler_recipe = ref, .calibration_campaign = ref, .full_cli_release_receipt = ref, .source_freeze = ref },
        .owners = &owners,
    };
    var retained = [_]u32{ 512, 512, 512, 512, 512, 512, 512, 512 };
    const profile_owners = [_]profiles.Owner{
        .{ .layer = 5, .retained = retained[0..2] },  .{ .layer = 11, .retained = retained[2..4] },
        .{ .layer = 17, .retained = retained[4..6] }, .{ .layer = 23, .retained = retained[6..8] },
    };
    const typed: profiles.Profile = .{
        .identity = .{ .artifact = @splat(0x11), .tokenizer = @splat(0x11), .template = @splat(0x11), .layout = @splat(0x11), .runtime = runtime },
        .engine = .native_whole_head_v2,
        .producer_policy = .cumulative256_exact_cycle_v1,
        .prepare_host_owner_capability = .bounded_page_e4_resident_1152m_v1,
        .resident_authority = .{
            .cohort = .e4_8128_v1,
            .base_compatibility_artifact_sha256 = base_artifact,
            .base_compatibility_semantic_sha256 = base_semantic,
            .training_policy_sha256 = training,
            .configuration_sha256 = configuration,
            .source_freeze_sha256 = source_freeze,
            .shape_authority_sha256 = shape,
            .policy_module_sha256 = @import("context_resident_hybrid_staged_configuration.zig").policy_module_sha256,
            .encoding = .f16,
            .logical_tokens = 8128,
            .max_output_tokens = 64,
            .fit_workers = 1,
            .resume_allowed = false,
            .compiled_max_copy_bytes = 4 * 1024 * 1024 * 1024,
            .compiled_max_host_staging_bytes = 256 * 1024 * 1024,
            .compiled_codec_scope = .f16_only,
            .compiled_max_logical_tokens = 8192,
            .compiled_layout_policy = .full_history_exact,
        },
        .owners = &profile_owners,
        .query_protocols = &.{},
        .max_prompt_tokens = 8192,
        .max_output_tokens = 64,
        .minimum_saved_bytes = std.heap.pageSize(),
    };
    const semantic = typed.semanticDigest();
    artifact.profile_semantic_sha256 = &std.fmt.bytesToHex(semantic, .lower);
    return std.json.Stringify.valueAlloc(allocator, artifact, .{});
}

/// Validate only the conservative compatibility profile emitted by this
/// staged authority. It intentionally does not claim a fixed per-block row
/// reduction: the adaptive route derives its geometry and enforces global
/// physical gates after materialization.
pub fn validateCompatibilityProfile(
    profile: profiles.Profile,
    actual: profiles.Identity,
    layout: @import("runtime/context_policy.zig").Layout,
) !void {
    return validateCompatibilityProfileWithE2Limit(profile, actual, layout, 8192);
}

fn validateCompatibilityProfileWithE2Limit(
    profile: profiles.Profile,
    actual: profiles.Identity,
    layout: @import("runtime/context_policy.zig").Layout,
    e2_prompt_limit: u32,
) !void {
    if (layout.len == 0 or layout.len > layout.layers.len or !std.meta.eql(profile.identity, actual) or
        profile.engine != .native_whole_head_v2 or
        !std.meta.eql(profile.producer_policy, fit_policy.Policy.cumulative256_exact_cycle_v1) or
        profile.query_protocols.len != 0 or
        profile.minimum_saved_bytes != std.heap.pageSize() or
        (profile.owners.len != 4 and profile.owners.len != 3))
        return error.UnsupportedCompactionProfile;
    const e2 = profile.owners.len == 3;
    const resident = profile.prepare_host_owner_capability == .bounded_page_e4_resident_1152m_v1;
    if (resident != (profile.resident_authority != null) or
        profile.max_prompt_tokens != (if (e2) e2_prompt_limit else @as(u32, if (resident) 8192 else 32768)) or
        profile.max_output_tokens != @as(u32, if (resident) 64 else 256)) return error.UnsupportedCompactionProfile;
    const e4_layers = [_]u32{ 5, 11, 17, 23 };
    const e2_layers = [_]u32{ 4, 9, 14 };
    var global_index: usize = 0;
    for (layout.layers[0..layout.len], 0..) |layer, i| {
        if (layer.owner != i or layer.sliding_window != 0) continue;
        const count: usize = if (e2) e2_layers.len else e4_layers.len;
        if (global_index >= count or i != (if (e2) e2_layers[global_index] else e4_layers[global_index]))
            return error.UnsupportedCompactionProfile;
        global_index += 1;
    }
    if (global_index != (if (e2) e2_layers.len else e4_layers.len)) return error.UnsupportedCompactionProfile;
    for (profile.owners, 0..) |owner, owner_index| {
        const layer = if (e2) e2_layers[owner_index] else e4_layers[owner_index];
        const heads: usize = if (e2) 1 else 2;
        if (owner.layer != layer or owner.retained.len != heads or
            owner.retained[0] != 512 or (!e2 and owner.retained[1] != 512) or
            layer >= layout.len or layout.layers[layer].owner != layer or
            layout.layers[layer].kv_heads != @as(u32, @intCast(heads)) or layout.layers[layer].head_dim != 512 or
            layout.layers[layer].sliding_window != 0)
            return error.UnsupportedCompactionProfile;
    }
}

/// Bind the staged profile to the portable native-checkpoint namespace rather
/// than the unrelated GGUF serving-compaction namespace.  Only the exact
/// compiled whole-head implementation is admitted.
pub fn validateNativeCheckpointCompatibilityProfile(
    profile: profiles.Profile,
    checkpoint: @import("runtime/kv/attention_match_checkpoint.zig").Identity,
    layout: @import("runtime/context_policy.zig").Layout,
    implementation: profiles.Implementation,
) !void {
    const required = profiles.servingImplementation(.native_whole_head_v2, true, true);
    if (!std.meta.eql(implementation, required))
        return error.UnsupportedCompactionProfile;
    const actual: profiles.Identity = .{
        .artifact = checkpoint.weights,
        .tokenizer = checkpoint.tokenizer,
        .template = checkpoint.template,
        .layout = checkpoint.layout,
        .runtime = profiles.implementationDigest(required),
    };
    // The resident E2 namespace has its own parser and exact semantic seal.
    // Its 32K contract must not be interpreted as the legacy private-Dense
    // E2 profile, whose 8K admission remains unchanged above.
    const resident_e2 = profile.owners.len == 3 and
        std.mem.eql(u8, &profile.semanticDigest(), &@import("context_resident_hybrid_compatibility_profile.zig").expectedE2ProfileSemantic());
    try validateCompatibilityProfileWithE2Limit(profile, actual, layout, if (resident_e2) 32768 else 8192);
}

test "context parsed resident E2 profile selects native checkpoint namespace with exact 32K scope" {
    const resident_profile = @import("context_resident_hybrid_compatibility_profile.zig");
    const checkpoint: @import("runtime/kv/attention_match_checkpoint.zig").Identity = .{
        .weights = @splat(1),
        .tokenizer = @splat(2),
        .template = @splat(3),
        .layout = @splat(4),
    };
    const required = profiles.servingImplementation(.native_whole_head_v2, true, true);
    const weights_hex = std.fmt.bytesToHex(checkpoint.weights, .lower);
    const tokenizer_hex = std.fmt.bytesToHex(checkpoint.tokenizer, .lower);
    const template_hex = std.fmt.bytesToHex(checkpoint.template, .lower);
    const layout_hex = std.fmt.bytesToHex(checkpoint.layout, .lower);
    const implementation_hex = std.fmt.bytesToHex(resident_profile.runtimeImplementationDigest(), .lower);
    const semantic_hex = std.fmt.bytesToHex(resident_profile.expectedE2ProfileSemantic(), .lower);
    const ResidentFixtureHead = struct { head: u32 = 0, retained_rows: u32 = 512 };
    const ResidentFixtureOwner = struct { layer: u32, heads: []const ResidentFixtureHead };
    const owners = [_]ResidentFixtureOwner{
        .{ .layer = 4, .heads = &.{.{}} },
        .{ .layer = 9, .heads = &.{.{}} },
        .{ .layer = 14, .heads = &.{.{}} },
    };
    const bytes = try std.json.Stringify.valueAlloc(std.testing.allocator, .{
        .schema = resident_profile.e2_schema,
        .stage = resident_profile.stage,
        .diagnostic_only = true,
        .candidate_qualification = false,
        .serving_qualified = false,
        .release_authority = false,
        .registry_enrollment = false,
        .evaluation_opened = false,
        .evidence_present = false,
        .frozen_before_unseen_evaluation = true,
        .training_policy = "private_train_challenge_hybrid_resident_v1",
        .fit_strategy = "identity_chunk_coefficients_adaptive_v18",
        .identity = .{
            .artifact_sha256 = weights_hex[0..],
            .tokenizer_sha256 = tokenizer_hex[0..],
            .template_sha256 = template_hex[0..],
            .layout_sha256 = layout_hex[0..],
            .implementation_sha256 = implementation_hex[0..],
        },
        .profile_semantic_sha256 = semantic_hex[0..],
        .max_prompt_tokens = @as(u32, 32768),
        .max_output_tokens = @as(u32, 256),
        .owners = owners[0..],
        .source = .{ .source_freeze = .{ .path = "/sealed/source.json", .sha256 = weights_hex[0..] } },
    }, .{});
    defer std.testing.allocator.free(bytes);
    var loaded = try resident_profile.parseBytes(std.testing.allocator, bytes);
    defer loaded.deinit();
    const context_policy = @import("runtime/context_policy.zig");
    var layout: context_policy.Layout = .{ .len = 35 };
    for (layout.layers[0..layout.len], 0..) |*layer, i| layer.* = .{
        .owner = @intCast(i),
        .kv_heads = 1,
        .query_heads = 8,
        .head_dim = 256,
        .sliding_window = 512,
        .rope_active_dim = 256,
        .rope_frequency_dim = 256,
        .rope_theta = 10_000,
    };
    inline for (.{ 4, 9, 14 }) |i| layout.layers[i] = .{
        .owner = i,
        .kv_heads = 1,
        .query_heads = 8,
        .head_dim = 512,
        .sliding_window = 0,
        .rope_active_dim = 512,
        .rope_frequency_dim = 512,
        .rope_theta = 10_000,
    };
    try validateNativeCheckpointCompatibilityProfile(loaded.profile, checkpoint, layout, required);
    try std.testing.expectError(error.UnsupportedCompactionProfile, validateCompatibilityProfile(loaded.profile, loaded.profile.identity, layout));
    var legacy = loaded.profile;
    legacy.max_prompt_tokens = 8192;
    try validateNativeCheckpointCompatibilityProfile(legacy, checkpoint, layout, required);
    var alias = loaded.profile;
    alias.max_prompt_tokens = 16384;
    try std.testing.expectError(error.UnsupportedCompactionProfile, validateNativeCheckpointCompatibilityProfile(alias, checkpoint, layout, required));
    var wrong_identity = checkpoint;
    wrong_identity.layout[0] ^= 1;
    try std.testing.expectError(error.UnsupportedCompactionProfile, validateNativeCheckpointCompatibilityProfile(loaded.profile, wrong_identity, layout, required));
    layout.layers[14].kv_heads = 2;
    try std.testing.expectError(error.UnsupportedCompactionProfile, validateNativeCheckpointCompatibilityProfile(loaded.profile, checkpoint, layout, required));
}

test "native checkpoint compatibility keeps its identity namespace and exact implementation" {
    const required = profiles.servingImplementation(.native_whole_head_v2, true, true);
    const checkpoint: @import("runtime/kv/attention_match_checkpoint.zig").Identity = .{
        .weights = @splat(1),
        .tokenizer = @splat(2),
        .template = @splat(3),
        .layout = @splat(4),
    };
    const actual: profiles.Identity = .{
        .artifact = checkpoint.weights,
        .tokenizer = checkpoint.tokenizer,
        .template = checkpoint.template,
        .layout = checkpoint.layout,
        .runtime = profiles.implementationDigest(required),
    };
    var retained = [_]u32{ 512, 512, 512, 512, 512, 512, 512, 512 };
    const owners = [_]profiles.Owner{
        .{ .layer = 5, .retained = retained[0..2] },  .{ .layer = 11, .retained = retained[2..4] },
        .{ .layer = 17, .retained = retained[4..6] }, .{ .layer = 23, .retained = retained[6..8] },
    };
    const policy = @import("runtime/context_policy.zig");
    var layout: policy.Layout = .{ .len = 24 };
    for (layout.layers[0..layout.len], 0..) |*layer, i| layer.* = .{ .owner = @intCast(i), .kv_heads = 2, .query_heads = 4, .head_dim = 512, .sliding_window = 4096, .rope_active_dim = 512, .rope_frequency_dim = 512, .rope_theta = 10_000 };
    inline for (.{ 5, 11, 17, 23 }) |i| layout.layers[i].sliding_window = 0;
    const profile: profiles.Profile = .{
        .identity = actual,
        .engine = .native_whole_head_v2,
        .producer_policy = fit_policy.Policy.cumulative256_exact_cycle_v1,
        .artifact_sha256 = @splat(9),
        .owners = &owners,
        .query_protocols = &.{},
        .max_prompt_tokens = 32768,
        .max_output_tokens = 256,
        .minimum_saved_bytes = std.heap.pageSize(),
    };
    try validateNativeCheckpointCompatibilityProfile(profile, checkpoint, layout, required);
    var generic_profile = profile;
    generic_profile.identity = try profiles.Identity.fromLoaded(.{
        .artifact = checkpoint.weights,
        .tokenizer = checkpoint.tokenizer,
        .template = checkpoint.template,
    }, layout, required);
    try std.testing.expectError(error.UnsupportedCompactionProfile, validateNativeCheckpointCompatibilityProfile(generic_profile, checkpoint, layout, required));
    var malformed_layout = layout;
    malformed_layout.len = malformed_layout.layers.len + 1;
    try std.testing.expectError(error.UnsupportedCompactionProfile, validateNativeCheckpointCompatibilityProfile(profile, checkpoint, malformed_layout, required));
    malformed_layout = layout;
    malformed_layout.layers[0].sliding_window = 0;
    try std.testing.expectError(error.UnsupportedCompactionProfile, validateNativeCheckpointCompatibilityProfile(profile, checkpoint, malformed_layout, required));
    var wrong = checkpoint;
    wrong.layout[0] ^= 1;
    try std.testing.expectError(error.UnsupportedCompactionProfile, validateNativeCheckpointCompatibilityProfile(profile, wrong, layout, required));
    try std.testing.expectError(error.UnsupportedCompactionProfile, validateNativeCheckpointCompatibilityProfile(profile, checkpoint, layout, profiles.servingImplementation(.block_v1, false, false)));
}

test "context source206 compiled identity observation" {
    const implementation = profiles.servingImplementation(.native_whole_head_v2, true, true);
    std.debug.print("SOURCE206_COMPILED_IDENTITY {{\"compaction_identity_sha256\":\"{s}\",\"native_profile_implementation_sha256\":\"{s}\",\"allocation_page_bytes\":{d}}}\n", .{
        std.fmt.bytesToHex(@import("compaction_identity").digest, .lower),
        std.fmt.bytesToHex(profiles.implementationDigest(implementation), .lower),
        std.heap.pageSize(),
    });
}

test "context private Dense parser positive roundtrip uses compiled page size" {
    const hex = "0101010101010101010101010101010101010101010101010101010101010101";
    var retained = [_]u32{ 512, 512, 512, 512, 512, 512, 512, 512 };
    const profile_owners = [_]profiles.Owner{
        .{ .layer = 5, .retained = retained[0..2] },  .{ .layer = 11, .retained = retained[2..4] },
        .{ .layer = 17, .retained = retained[4..6] }, .{ .layer = 23, .retained = retained[6..8] },
    };
    const implementation = profiles.servingImplementation(.native_whole_head_v2, true, true);
    const profile: profiles.Profile = .{
        .identity = .{ .artifact = @splat(1), .tokenizer = @splat(1), .template = @splat(1), .layout = @splat(1), .runtime = profiles.implementationDigest(implementation) },
        .engine = .native_whole_head_v2,
        .producer_policy = fit_policy.Policy.cumulative256_exact_cycle_v1,
        .artifact_sha256 = @splat(1),
        .owners = &profile_owners,
        .query_protocols = &.{},
        .max_prompt_tokens = 32768,
        .max_output_tokens = 256,
        .minimum_saved_bytes = std.heap.pageSize(),
    };
    const semantic = std.fmt.bytesToHex(profile.semanticDigest(), .lower);
    const implementation_hex = std.fmt.bytesToHex(profiles.implementationDigest(implementation), .lower);
    const identity_map = std.fmt.bytesToHex(identityMapDigest(7936), .lower);
    var heads: [8]Head = undefined;
    for (&heads, 0..) |*head, i| head.* = .{
        .donor = @intCast(([_]u32{ 5, 5, 11, 11, 17, 17, 23, 23 })[i]),
        .head = @intCast(i % 2),
        .outcome = if (i == 0) .exact_identity else .compacted,
        .source_rows = 7936,
        .candidate_rows = if (i == 0) 7936 else 5000,
        .source_head_sha256 = hex,
        .source_index_map_sha256 = if (i == 0) &identity_map else hex,
        .candidate_sha256 = if (i == 0) null else hex,
        .max_mass_error = if (i == 0) null else 0.01,
        .output_rmse = if (i == 0) null else 0.01,
    };
    const owners = [_]Owner{
        .{ .layer = 5, .heads = heads[0..2] },  .{ .layer = 11, .heads = heads[2..4] },
        .{ .layer = 17, .heads = heads[4..6] }, .{ .layer = 23, .heads = heads[6..8] },
    };
    const reference: Ref = .{ .path = "/source206", .sha256 = hex };
    const artifact: Artifact = .{
        .schema = schema,
        .stage = stage,
        .diagnostic_only = true,
        .candidate_qualification = true,
        .serving_qualified = false,
        .release_authority = false,
        .evaluation_opened = false,
        .frozen_before_unseen_evaluation = true,
        .training_policy = "private_dense512_reference_v1",
        .fit_strategy = "identity_chunk_coefficients_adaptive_v18",
        .identity = .{ .artifact_sha256 = hex, .tokenizer_sha256 = hex, .template_sha256 = hex, .layout_sha256 = hex, .implementation_sha256 = &implementation_hex },
        .profile_semantic_sha256 = &semantic,
        .max_prompt_tokens = 32768,
        .max_output_tokens = 256,
        .conservative_owner_rows_per_head = 512,
        .policy = .{ .initial_global_physical_ratio = 1.5, .post_materialization_global_physical_ratio = 1.000001, .max_mass_error = 0.05, .max_output_rmse = 0.05, .physical_postcheck_required = true, .geometry_budget_primary_percent = 91, .geometry_budget_fallback_percent = 95, .max_fallbacks = 3 },
        .source = .{ .primary_capture = reference, .dense512_capture = reference, .sampler_recipe = reference, .calibration_campaign = reference, .full_cli_release_receipt = reference, .source_freeze = reference },
        .owners = &owners,
    };
    const encoded = try std.json.Stringify.valueAlloc(std.testing.allocator, artifact, .{});
    defer std.testing.allocator.free(encoded);
    var loaded = try parseBytes(std.testing.allocator, encoded);
    defer loaded.deinit();
    try std.testing.expectEqual(std.heap.pageSize(), loaded.profile.minimum_saved_bytes);
    const expected_semantic = profile.semanticDigest();
    const actual_semantic = loaded.profile.semanticDigest();
    try std.testing.expectEqualSlices(u8, &expected_semantic, &actual_semantic);
}

test "context private Dense policy is exact and rejects weakened mixed tuples" {
    const exact: @FieldType(Artifact, "policy") = .{
        .initial_global_physical_ratio = 1.5,
        .post_materialization_global_physical_ratio = 1.000001,
        .max_mass_error = 0.05,
        .max_output_rmse = 0.05,
        .physical_postcheck_required = true,
        .geometry_budget_primary_percent = 91,
        .geometry_budget_fallback_percent = 95,
        .max_fallbacks = 3,
    };
    try std.testing.expect(validPolicy(exact));
    var changed = exact;
    changed.physical_postcheck_required = false;
    try std.testing.expect(!validPolicy(changed));
    changed = exact;
    changed.geometry_budget_primary_percent = 95;
    try std.testing.expect(!validPolicy(changed));
    changed = exact;
    changed.max_mass_error = 0.0500001;
    try std.testing.expect(!validPolicy(changed));
    try std.testing.expect(validCompactedMetrics(0, 0.05, exact));
    try std.testing.expect(!validCompactedMetrics(-0.000001, 0, exact));
    try std.testing.expect(!validCompactedMetrics(0, -0.000001, exact));
}

test "context private Dense exact identity map is canonical LE u64" {
    const expected = [_]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(&expected, &actual, .{});
    try std.testing.expectEqualSlices(u8, &actual, &identityMapDigest(2));
    const base: Head = .{ .donor = 5, .head = 0, .outcome = .exact_identity, .source_rows = 2, .candidate_rows = 2, .source_head_sha256 = "", .source_index_map_sha256 = "" };
    try std.testing.expect(validIdentityCoordinate(base));
    var swapped = base;
    swapped.head = 1;
    try std.testing.expect(!validIdentityCoordinate(swapped));
}

test "context private Dense parser does not reinterpret canonical native or legacy tags" {
    try std.testing.expectError(error.MissingField, parseBytes(std.testing.allocator, "{\"schema\":\"antfly.context.native.qualification-profile.v1\"}"));
    try std.testing.expectError(error.UnknownField, parseBytes(std.testing.allocator, "{\"schema_version\":1,\"stage\":\"offline_numerical\"}"));
}

test "context E2 private Dense parser binds three heads identity and physical prediction" {
    const artifact_hex = "0394d21a892ba0194e9bb6797eefc1846bf2d1df1cff86a26d24c45805efd163";
    const tokenizer_hex = "32e685c6648d25b7f696d1af0a4c16a58f8d8cad7123b2e48c81aa5284b6d1ab";
    const template_hex = "611d5f75642790e55749a538066e24a7f13b6a1b58697c855ed078a148ee5dbb";
    const layout_hex = "b61587d2700c7e3d1c1b014ea0e0a7f7a47bd93ea1237972e88084d5683511e4";
    const main_hex = "fa401b55b07ee70a54c6dae3903c783a6e65064312529ea57175cb5f8dec6634";
    const generic_hex = "0101010101010101010101010101010101010101010101010101010101010101";
    const implementation = profiles.servingImplementation(.native_whole_head_v2, true, true);
    const implementation_hex = std.fmt.bytesToHex(profiles.implementationDigest(implementation), .lower);
    var retained = [_]u32{ 512, 512, 512 };
    const profile_owners = [_]profiles.Owner{
        .{ .layer = 4, .retained = retained[0..1] },
        .{ .layer = 9, .retained = retained[1..2] },
        .{ .layer = 14, .retained = retained[2..3] },
    };
    const profile: profiles.Profile = .{
        .identity = .{ .artifact = try digest(artifact_hex), .tokenizer = try digest(tokenizer_hex), .template = try digest(template_hex), .layout = try digest(layout_hex), .runtime = profiles.implementationDigest(implementation) },
        .engine = .native_whole_head_v2,
        .producer_policy = fit_policy.Policy.cumulative256_exact_cycle_v1,
        .artifact_sha256 = @splat(1),
        .owners = &profile_owners,
        .query_protocols = &.{},
        .max_prompt_tokens = 8192,
        .max_output_tokens = 256,
        .minimum_saved_bytes = std.heap.pageSize(),
    };
    const semantic = std.fmt.bytesToHex(profile.semanticDigest(), .lower);
    const identity_map = std.fmt.bytesToHex(identityMapDigest(7936), .lower);
    var heads = [_]Head{
        .{ .donor = 4, .head = 0, .outcome = .compacted, .source_rows = 7936, .candidate_rows = 2500, .source_head_sha256 = generic_hex, .source_index_map_sha256 = generic_hex, .candidate_sha256 = generic_hex, .max_mass_error = 0.01, .output_rmse = 0.01 },
        .{ .donor = 9, .head = 0, .outcome = .exact_identity, .source_rows = 7936, .candidate_rows = 7936, .source_head_sha256 = generic_hex, .source_index_map_sha256 = &identity_map },
        .{ .donor = 14, .head = 0, .outcome = .compacted, .source_rows = 7936, .candidate_rows = 2500, .source_head_sha256 = generic_hex, .source_index_map_sha256 = generic_hex, .candidate_sha256 = generic_hex, .max_mass_error = 0.01, .output_rmse = 0.01 },
    };
    var owners = [_]Owner{
        .{ .layer = 4, .heads = heads[0..1] },
        .{ .layer = 9, .heads = heads[1..2] },
        .{ .layer = 14, .heads = heads[2..3] },
    };
    const reference: Ref = .{ .path = "/source332", .sha256 = generic_hex };
    const predicted = try std.math.add(u64, try e2HeadBytes(2500), try std.math.add(u64, try e2HeadBytes(7936), try e2HeadBytes(2500)));
    var artifact: Artifact = .{
        .schema = e2_schema,
        .stage = e2_stage,
        .diagnostic_only = true,
        .candidate_qualification = true,
        .serving_qualified = false,
        .release_authority = false,
        .evaluation_opened = false,
        .frozen_before_unseen_evaluation = true,
        .training_policy = "private_dense512_reference_v1",
        .fit_strategy = "identity_chunk_coefficients_adaptive_v18",
        .identity = .{ .artifact_sha256 = artifact_hex, .tokenizer_sha256 = tokenizer_hex, .template_sha256 = template_hex, .layout_sha256 = layout_hex, .implementation_sha256 = &implementation_hex },
        .main_gguf_sha256 = main_hex,
        .profile_semantic_sha256 = &semantic,
        .max_prompt_tokens = 8192,
        .max_output_tokens = 256,
        .conservative_owner_rows_per_head = 512,
        .ordinary_source_bytes = 48_758_784,
        .selected_page_rounded_prediction_bytes = predicted,
        .policy = .{ .initial_global_physical_ratio = 1.5, .post_materialization_global_physical_ratio = 1.000001, .max_mass_error = 0.05, .max_output_rmse = 0.05, .physical_postcheck_required = true, .geometry_budget_primary_percent = 91, .geometry_budget_fallback_percent = 95, .max_fallbacks = 3 },
        .source = .{ .primary_capture = reference, .dense512_capture = reference, .sampler_recipe = reference, .calibration_campaign = reference, .full_cli_release_receipt = reference, .source_freeze = reference },
        .owners = &owners,
    };
    const encoded = try std.json.Stringify.valueAlloc(std.testing.allocator, artifact, .{});
    defer std.testing.allocator.free(encoded);
    var loaded = try parseBytes(std.testing.allocator, encoded);
    defer loaded.deinit();
    try std.testing.expectEqual(@as(usize, 3), loaded.profile.owners.len);

    artifact.schema = schema;
    const mislabeled = try std.json.Stringify.valueAlloc(std.testing.allocator, artifact, .{});
    defer std.testing.allocator.free(mislabeled);
    try std.testing.expectError(error.InvalidPrivateDenseProfile, parseBytes(std.testing.allocator, mislabeled));
    artifact.schema = e2_schema;
    owners[1].layer = 4;
    const duplicate = try std.json.Stringify.valueAlloc(std.testing.allocator, artifact, .{});
    defer std.testing.allocator.free(duplicate);
    try std.testing.expectError(error.InvalidPrivateDenseProfile, parseBytes(std.testing.allocator, duplicate));
    owners[1].layer = 9;
    artifact.selected_page_rounded_prediction_bytes = predicted + 1;
    const ratio_mismatch = try std.json.Stringify.valueAlloc(std.testing.allocator, artifact, .{});
    defer std.testing.allocator.free(ratio_mismatch);
    try std.testing.expectError(error.InvalidPrivateDenseProfile, parseBytes(std.testing.allocator, ratio_mismatch));

    const context_policy = @import("runtime/context_policy.zig");
    var layout: context_policy.Layout = .{ .len = 35 };
    for (layout.layers[0..layout.len], 0..) |*layer, i| layer.* = .{ .owner = @intCast(i), .kv_heads = 1, .query_heads = 4, .head_dim = 256, .sliding_window = 512, .rope_active_dim = 256, .rope_frequency_dim = 256, .rope_theta = 10_000 };
    inline for (.{ 4, 9, 14 }) |i| layout.layers[i] = .{ .owner = i, .kv_heads = 1, .query_heads = 8, .head_dim = 512, .sliding_window = 0, .rope_active_dim = 512, .rope_frequency_dim = 512, .rope_theta = 10_000 };
    try validateCompatibilityProfile(loaded.profile, loaded.profile.identity, layout);
    layout.layers[14].kv_heads = 2;
    try std.testing.expectError(error.UnsupportedCompactionProfile, validateCompatibilityProfile(loaded.profile, loaded.profile.identity, layout));
}
