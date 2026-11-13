// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Staged, unqualified TRAIN-only mixture for exact-source future-decode queries.
//! The heldout domain is never accepted from either added TRAIN source.
const std = @import("std");
const engine = @import("attention_match_engine.zig");
const Control = @import("attention_match.zig").Control;
pub const historical_selection = @import("attention_match_historical_selection.zig");

pub const schema = "antfly.attention-match-future-decode-training.v1";
pub const primary_rows: usize = 256;
pub const historical_rows: usize = 256;
pub const future_rows: usize = 256;
pub const training_rows: usize = primary_rows + historical_rows + future_rows;
pub const heldout_rows: usize = 64;
pub const generated_tokens: usize = 256;
pub const max_head_workspace_bytes: usize = 1024 * 1024 * 1024;

pub fn policyDigest() [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(schema ++ ";primary=256;historical=256;historical-selection=normalized-q-cosine-kcenter-v1;future=256;heldout=64;future-source=unmodified-exact-prefix;generation=greedy-fixed256-ignore-eos-observe;workspace=1GiB;cpu-fp64-heldout=unchanged;mass=0.05;output=0.05;global=1.5");
    hash.update(&historical_selection.policyDigest());
    return hash.finalResult();
}

pub fn residentHybridPolicyDigest() [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.attention-match-resident-hybrid-training-authority.v1;P=256;H=256-derived-and-rewritten;F=256-chronological-generated;HHO=64-original-primary;split=256,512;dim=512;batches=56;future-generation=greedy-fixed256-ignore-eos-observe;workspace=2GiB;cpu-fp64-validation=independent;mass=0.05;output=0.05;global=1.5\x00");
    hash.update(&@import("attention_match_native_resident_hybrid_bridge.zig").implementation_semantic_sha256);
    hash.update(&@import("attention_match_resident_native_hybrid_profile.zig").digest());
    return hash.finalResult();
}

pub fn residentE2HybridPolicyDigest() [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.attention-match-resident-e2-hybrid-training-authority.v2;P=256;H=256-derived-and-rewritten;F=256-chronological-generated;HHO=64-original-primary;split=256,512;dim=512;batches=56;per-owner=8,8,40;future-generation=greedy-fixed256-ignore-eos-observe;workspace=2GiB;cpu-fp64-validation=independent;mass=0.05;output=0.05;global=1.5\x00");
    hash.update(&@import("attention_match_e2_native_resident_hybrid_bridge.zig").implementation_semantic_sha256);
    hash.update(&@import("attention_match_e2_resident_authority.zig").digest());
    return hash.finalResult();
}

pub fn supportedPolicyDigest(value: [32]u8) bool {
    return std.mem.eql(u8, &value, &policyDigest()) or std.mem.eql(u8, &value, &residentHybridPolicyDigest()) or std.mem.eql(u8, &value, &residentE2HybridPolicyDigest());
}

pub const Authority = struct {
    source_identity_sha256: [32]u8,
    source_tokens_sha256: [32]u8,
    source_revision: u64,
    generation_contract_sha256: [32]u8,
    generated_ids_sha256: [32]u8,
    primary_sha256: [32]u8,
    historical_sha256: [32]u8,
    future_sha256: [32]u8,
    heldout_sha256: [32]u8,
    early_eos_observations: u16,

    pub fn digest(self: @This()) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        if (std.mem.eql(u8, &self.generation_contract_sha256, &residentHybridPolicyDigest())) {
            hash.update("antfly.attention-match-resident-hybrid-authority.v1\x00");
            hash.update(&residentHybridPolicyDigest());
        } else if (std.mem.eql(u8, &self.generation_contract_sha256, &residentE2HybridPolicyDigest())) {
            hash.update("antfly.attention-match-resident-e2-hybrid-authority.v1\x00");
            hash.update(&residentE2HybridPolicyDigest());
        } else hash.update(&policyDigest());
        inline for (.{ self.source_identity_sha256, self.source_tokens_sha256, self.generation_contract_sha256, self.generated_ids_sha256, self.primary_sha256, self.historical_sha256, self.future_sha256, self.heldout_sha256 }) |value| hash.update(&value);
        var revision: [8]u8 = undefined;
        std.mem.writeInt(u64, &revision, self.source_revision, .little);
        hash.update(&revision);
        var eos: [2]u8 = undefined;
        std.mem.writeInt(u16, &eos, self.early_eos_observations, .little);
        hash.update(&eos);
        return hash.finalResult();
    }
};

pub const Source = struct {
    train: []const f32,
    positions: []const u64,
};

pub const Input = struct {
    primary: Source,
    historical: Source,
    future: Source,
    heldout: []const f32,
    heldout_positions: []const u64,
    authority: Authority,
};

pub const OwnedBatch = struct {
    allocator: std.mem.Allocator,
    train: []f32,
    positions: []u64,
    heldout: []f32,
    heldout_positions: []u64,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.train);
        self.allocator.free(self.positions);
        self.allocator.free(self.heldout);
        self.allocator.free(self.heldout_positions);
        self.* = undefined;
    }

    pub fn reference(self: *const @This(), consumer_layer: u16, query_head: u16) engine.ReferenceBatch {
        return .{ .consumer_layer = consumer_layer, .query_head = query_head, .train = self.train, .heldout = self.heldout, .train_positions = self.positions, .heldout_positions = self.heldout_positions };
    }
};

fn hashFloats(values: []const f32) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(std.mem.sliceAsBytes(values));
    return hash.finalResult();
}

fn hashSource(source: Source) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(std.mem.sliceAsBytes(source.train));
    hash.update(std.mem.sliceAsBytes(source.positions));
    return hash.finalResult();
}

fn hashHeldout(values: []const f32, positions: []const u64) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(std.mem.sliceAsBytes(values));
    hash.update(std.mem.sliceAsBytes(positions));
    return hash.finalResult();
}

/// The only merge entrypoint used by production prepare and the maintained
/// quality worker. It binds the exact row values *and positions* into the
/// already-authenticated capture authority before allocating the replacement.
pub fn buildBound(allocator: std.mem.Allocator, input: Input, capture_authority: Authority, control: Control) !OwnedBatch {
    var bound = input;
    bound.authority = capture_authority;
    bound.authority.primary_sha256 = hashSource(input.primary);
    bound.authority.historical_sha256 = hashSource(input.historical);
    bound.authority.future_sha256 = hashSource(input.future);
    bound.authority.heldout_sha256 = hashHeldout(input.heldout, input.heldout_positions);
    return build(allocator, bound, control);
}

/// Builds the complete candidate batch off to the side. Callers may publish it
/// only after this returns; cancellation, OOM, bad hashes and row mismatch leave
/// no partially visible mixture.
pub fn build(allocator: std.mem.Allocator, input: Input, control: Control) !OwnedBatch {
    try control.check();
    if (input.primary.positions.len != primary_rows or input.historical.positions.len != historical_rows or input.future.positions.len != future_rows or input.heldout_positions.len != heldout_rows)
        return error.InvalidFutureDecodeTraining;
    const dim = input.primary.train.len / primary_rows;
    if (dim == 0 or input.primary.train.len != primary_rows * dim or input.historical.train.len != historical_rows * dim or input.future.train.len != future_rows * dim or input.heldout.len != heldout_rows * dim)
        return error.InvalidFutureDecodeTraining;
    if (!std.mem.eql(u8, &hashSource(input.primary), &input.authority.primary_sha256) or
        !std.mem.eql(u8, &hashSource(input.historical), &input.authority.historical_sha256) or
        !std.mem.eql(u8, &hashSource(input.future), &input.authority.future_sha256) or
        !std.mem.eql(u8, &hashHeldout(input.heldout, input.heldout_positions), &input.authority.heldout_sha256))
        return error.FutureDecodeTrainingDigestMismatch;
    var result: OwnedBatch = .{
        .allocator = allocator,
        .train = try allocator.alloc(f32, training_rows * dim),
        .positions = undefined,
        .heldout = undefined,
        .heldout_positions = undefined,
    };
    errdefer allocator.free(result.train);
    result.positions = try allocator.alloc(u64, training_rows);
    errdefer allocator.free(result.positions);
    result.heldout = try allocator.dupe(f32, input.heldout);
    errdefer allocator.free(result.heldout);
    result.heldout_positions = try allocator.dupe(u64, input.heldout_positions);
    errdefer allocator.free(result.heldout_positions);
    const sources = .{ input.primary, input.historical, input.future };
    var row: usize = 0;
    inline for (sources) |source| {
        try control.check();
        @memcpy(result.train[row * dim ..][0..source.train.len], source.train);
        @memcpy(result.positions[row..][0..source.positions.len], source.positions);
        row += source.positions.len;
    }
    for (0..training_rows) |train_row| for (0..heldout_rows) |heldout_row|
        if (result.positions[train_row] == result.heldout_positions[heldout_row] and
            std.mem.eql(u8, std.mem.sliceAsBytes(result.train[train_row * dim ..][0..dim]), std.mem.sliceAsBytes(result.heldout[heldout_row * dim ..][0..dim])))
            return error.FutureDecodeHeldoutLeak;
    _ = input.authority.digest();
    return result;
}

test "future decode mixture fixes 256 256 256 and preserves heldout byte exactly" {
    const dim = 2;
    var p: [primary_rows * dim]f32 = undefined;
    var h: [historical_rows * dim]f32 = undefined;
    var f: [future_rows * dim]f32 = undefined;
    var held: [heldout_rows * dim]f32 = undefined;
    var pp: [primary_rows]u64 = undefined;
    var hp: [historical_rows]u64 = undefined;
    var fp: [future_rows]u64 = undefined;
    var heldp: [heldout_rows]u64 = undefined;
    for (&p, 0..) |*v, i| v.* = @floatFromInt(i);
    for (&h, 0..) |*v, i| v.* = @floatFromInt(1000 + i);
    for (&f, 0..) |*v, i| v.* = @floatFromInt(2000 + i);
    for (&held, 0..) |*v, i| v.* = @floatFromInt(3000 + i);
    for (&pp, 0..) |*v, i| v.* = i;
    for (&hp, 0..) |*v, i| v.* = 1000 + i;
    for (&fp, 0..) |*v, i| v.* = 2000 + i;
    for (&heldp, 0..) |*v, i| v.* = 3000 + i;
    const zero: [32]u8 = @splat(0);
    const authority: Authority = .{ .source_identity_sha256 = zero, .source_tokens_sha256 = zero, .source_revision = 1, .generation_contract_sha256 = zero, .generated_ids_sha256 = zero, .primary_sha256 = hashSource(.{ .train = &p, .positions = &pp }), .historical_sha256 = hashSource(.{ .train = &h, .positions = &hp }), .future_sha256 = hashSource(.{ .train = &f, .positions = &fp }), .heldout_sha256 = hashHeldout(&held, &heldp), .early_eos_observations = 1 };
    var owned = try build(std.testing.allocator, .{ .primary = .{ .train = &p, .positions = &pp }, .historical = .{ .train = &h, .positions = &hp }, .future = .{ .train = &f, .positions = &fp }, .heldout = &held, .heldout_positions = &heldp, .authority = authority }, .{});
    defer owned.deinit();
    try std.testing.expectEqual(@as(usize, 768 * dim), owned.train.len);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&held), std.mem.sliceAsBytes(owned.heldout));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&heldp), std.mem.sliceAsBytes(owned.heldout_positions));
    try std.testing.expectEqual(@as(f32, 0), owned.train[0]);
    try std.testing.expectEqual(@as(f32, 1000), owned.train[primary_rows * dim]);
    try std.testing.expectEqual(@as(f32, 2000), owned.train[(primary_rows + historical_rows) * dim]);
}

test "future decode mixture rejects heldout leak and digest corruption before return" {
    var train: [primary_rows]f32 = @splat(1);
    var held: [heldout_rows]f32 = @splat(2);
    var positions: [primary_rows]u64 = undefined;
    var held_positions: [heldout_rows]u64 = undefined;
    for (&positions, 0..) |*v, i| v.* = i;
    for (&held_positions, 0..) |*v, i| v.* = 1000 + i;
    const zero: [32]u8 = @splat(0);
    var authority: Authority = .{ .source_identity_sha256 = zero, .source_tokens_sha256 = zero, .source_revision = 1, .generation_contract_sha256 = zero, .generated_ids_sha256 = zero, .primary_sha256 = hashSource(.{ .train = &train, .positions = &positions }), .historical_sha256 = hashSource(.{ .train = &train, .positions = &positions }), .future_sha256 = hashSource(.{ .train = &train, .positions = &positions }), .heldout_sha256 = hashHeldout(&held, &held_positions), .early_eos_observations = 0 };
    const input = Input{ .primary = .{ .train = &train, .positions = &positions }, .historical = .{ .train = &train, .positions = &positions }, .future = .{ .train = &train, .positions = &positions }, .heldout = &held, .heldout_positions = &held_positions, .authority = authority };
    authority.future_sha256[0] ^= 1;
    var bad = input;
    bad.authority = authority;
    try std.testing.expectError(error.FutureDecodeTrainingDigestMismatch, build(std.testing.allocator, bad, .{}));
    // A leak requires the same position AND the same query vector. Numeric
    // position overlap between independent trajectories is allowed.
    held_positions[0] = positions[0];
    held[0] = train[0];
    var leak = input;
    leak.authority.heldout_sha256 = hashHeldout(&held, &held_positions);
    if (build(std.testing.allocator, leak, .{})) |unexpected| {
        var owned = unexpected;
        defer owned.deinit();
        return error.TestExpectedError;
    } else |err| {
        try std.testing.expectEqual(error.FutureDecodeHeldoutLeak, err);
    }
}

test "bound caller authenticates positions and permits position-only overlap" {
    var primary: [primary_rows]f32 = @splat(1);
    var historical: [historical_rows]f32 = @splat(2);
    var future: [future_rows]f32 = @splat(3);
    var heldout: [heldout_rows]f32 = @splat(4);
    var pp: [primary_rows]u64 = undefined;
    var hp: [historical_rows]u64 = undefined;
    var fp: [future_rows]u64 = undefined;
    var op: [heldout_rows]u64 = undefined;
    for (&pp, 0..) |*p, i| p.* = i;
    for (&hp, 0..) |*p, i| p.* = i; // independent domain may reuse numeric positions
    for (&fp, 0..) |*p, i| p.* = i;
    for (&op, 0..) |*p, i| p.* = i;
    var authority: Authority = undefined;
    authority.source_identity_sha256 = @splat(1);
    authority.source_tokens_sha256 = @splat(2);
    authority.source_revision = 7;
    authority.generation_contract_sha256 = policyDigest();
    authority.generated_ids_sha256 = @splat(3);
    authority.early_eos_observations = 0;
    var owned = try buildBound(std.testing.allocator, .{ .primary = .{ .train = &primary, .positions = &pp }, .historical = .{ .train = &historical, .positions = &hp }, .future = .{ .train = &future, .positions = &fp }, .heldout = &heldout, .heldout_positions = &op, .authority = undefined }, authority, .{});
    defer owned.deinit();
    fp[0] += 1;
    try std.testing.expectError(error.FutureDecodeTrainingDigestMismatch, build(std.testing.allocator, .{ .primary = .{ .train = &primary, .positions = &pp }, .historical = .{ .train = &historical, .positions = &hp }, .future = .{ .train = &future, .positions = &fp }, .heldout = &heldout, .heldout_positions = &op, .authority = authority }, .{}));
}
