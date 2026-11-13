// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
const authority = @import("server/native_kv_producer_authority.zig");
const enrollment = @import("context_private_dense_enrollment.zig");
const policy = @import("private_native_checkpoint_policy.zig");

pub const AuthorityKind = enum { qualified_private_profile };

pub const Grant = struct {
    authority_kind: AuthorityKind = .qualified_private_profile,
    authority: authority.Authority,
    enrollment: enrollment.VerifiedEnrollment,
    runtime_policy: policy.Resolved,
    training_policy: @import("pipelines/generation.zig").NativeCheckpointTrainingPolicy = .private_dense512_reference_v1,
    training_mix_policy_sha256: [32]u8 = @splat(0),
    prepare_host_owner_capability: @import("runtime/kv/compaction_profile.zig").PrepareHostOwnerCapability = .none,
    prepare_host_owner_physical_bytes: u64 = 0,
    resident_hybrid_authority: ?@import("pipelines/generation.zig").ResidentHybridAuthority = null,
    source_full_cli_binding_sha256: [32]u8,
};
