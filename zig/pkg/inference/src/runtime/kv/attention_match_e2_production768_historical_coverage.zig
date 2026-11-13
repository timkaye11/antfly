// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! TRAIN-only current-producer Dense512 -> H256 derivation for the offline
//! production768 replay. Artifact authentication remains with the caller.
const std = @import("std");
const solver = @import("attention_match.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const augmentation = @import("attention_match_training_augmentation.zig");
const selector = @import("attention_match_train_historical_coverage_selection.zig");
const pressure = @import("attention_match_train_pressure_exact_head_reservation.zig");
const e2_pressure = @import("attention_match_e2_historical_pressure_derivation.zig");
const accelerate_scan = @import("attention_match_accelerate_fp64_probability_scan.zig");

pub const threshold: f64 = 0.025;
pub const selector_policy_digest = selector.policy_digest;
pub const policy = "e2_production768_current_dense512_historical_coverage_v2|consumer_batches56|per_owner8,8,40|heads3|global_original_probability=scalar_fp64_attention_then_probability_rounded_f32_widened|threshold=0.025|baseline=primary256+future256|candidate=eligible_historical512|selected=256|heldout_role_positions_excluded_before_scoring|selector=train_historical_weighted_critical_coverage_then_normalized_q_kcenter_v1";
pub const policy_digest: [32]u8 = blk: {
    @setEvalBranchQuota(100_000);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(policy, &out, .{});
    break :blk out;
};
pub const accelerated_policy = "e2_production768_current_dense512_historical_coverage_accelerate_v1|eligible=gather_before_query_read|probability=f32_publication|shared_arithmetic_unchanged";

pub const ProbabilityOptions = struct {
    backend: ?accelerate_scan.Backend = null,
    gather_eligible_historical: bool = false,
    e2_shape_authority_sha256: ?[32]u8 = null,
    e2_source_physical_bytes: usize = 0,
};

pub const OuterBinding = struct {
    manifest_path: []const u8,
    manifest_sha256: [32]u8,
    nested_manifest_sha256: [32]u8,
    parent_manifest_sha256: [32]u8,
    identity: checkpoint.Identity,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
};

const OuterManifest = struct {
    schema: []const u8,
    diagnostic_only: bool,
    quality_credit: bool,
    training_only: bool,
    heldout_present: bool,
    retained_primary_manifest_sha256: []const u8,
    source_freeze_sha256: []const u8,
    binary_sha256: []const u8,
    exporter_sha256: []const u8,
    source_before: struct { identity: checkpoint.Identity, logical_tokens: usize },
    source_after: struct { identity: checkpoint.Identity, logical_tokens: usize },
    batch_count: usize,
    nested_training_path: []const u8,
    nested_training_manifest_sha256: []const u8,
};

fn decodeHex32(value: []const u8) ![32]u8 {
    if (value.len != 64) return error.InvalidTrainingAugmentationBinding;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, value) catch return error.InvalidTrainingAugmentationBinding;
    return out;
}

pub fn authenticateOuter(allocator: std.mem.Allocator, io: std.Io, binding: OuterBinding) !void {
    if (!std.fs.path.isAbsolute(binding.manifest_path)) return error.InvalidTrainingAugmentationBinding;
    const parent = std.fs.path.dirname(binding.manifest_path) orelse return error.InvalidTrainingAugmentationBinding;
    const name = std.fs.path.basename(binding.manifest_path);
    var dir = try std.Io.Dir.openDirAbsolute(io, parent, .{ .follow_symlinks = false });
    defer dir.close(io);
    const bytes = try dir.readFileAlloc(io, name, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    var actual: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &actual, .{});
    if (!std.mem.eql(u8, &actual, &binding.manifest_sha256)) return error.InvalidTrainingAugmentationBinding;
    const parsed = try std.json.parseFromSlice(OuterManifest, allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const m = parsed.value;
    if (!std.mem.eql(u8, m.schema, "antfly.attention-match-reference-coverage-training.v1") or !m.diagnostic_only or m.quality_credit or !m.training_only or m.heldout_present or m.batch_count != 56 or !std.mem.eql(u8, m.nested_training_path, "training") or m.source_before.logical_tokens != 8128 or m.source_after.logical_tokens != 8128 or !std.meta.eql(m.source_before.identity, binding.identity) or !std.meta.eql(m.source_after.identity, binding.identity)) return error.InvalidTrainingAugmentationBinding;
    const parent_digest = try decodeHex32(m.retained_primary_manifest_sha256);
    const nested_digest = try decodeHex32(m.nested_training_manifest_sha256);
    const freeze_digest = try decodeHex32(m.source_freeze_sha256);
    const binary_digest = try decodeHex32(m.binary_sha256);
    const exporter_digest = try decodeHex32(m.exporter_sha256);
    if (!std.mem.eql(u8, &parent_digest, &binding.parent_manifest_sha256) or !std.mem.eql(u8, &nested_digest, &binding.nested_manifest_sha256) or !std.mem.eql(u8, &freeze_digest, &binding.source_freeze_sha256) or !std.mem.eql(u8, &binary_digest, &binding.binary_sha256) or !std.mem.eql(u8, &exporter_digest, &binding.exporter_sha256)) return error.InvalidTrainingAugmentationBinding;
}

pub const Evidence = struct {
    batch_count: usize,
    eligible_rows: usize,
    excluded_rows: usize,
    baseline_covered_key_count: usize,
    newly_covered_key_count: usize,
    coverage_selected_rows: usize,
    kcenter_selected_rows: usize,
    selection_digest: [32]u8,
    derived_historical_sha256: [32]u8,
    coverage_authority_sha256: [32]u8,
    exact_head_policy_sha256: [32]u8,
    exact_head_evidence_sha256: [32]u8,
    chosen_exact_donor_layer: u32,
    chosen_exact_head_index: usize,
    pressure_infeasible_count: usize,
    exact_head_pressure: [3]pressure.HeadEvidence,
};

pub const Owned = struct {
    allocator: std.mem.Allocator,
    batches: []engine.ReferenceBatch,
    values: []f32,
    positions: []u64,
    coverage_offsets: []u32,
    coverage_indices: []u32,
    evidence: Evidence,
    pub fn liveBytes(self: @This()) !usize {
        var total = try mul(self.batches.len, @sizeOf(engine.ReferenceBatch));
        total = try add(total, try mul(self.values.len, @sizeOf(f32)));
        total = try add(total, try mul(self.positions.len, @sizeOf(u64)));
        total = try add(total, try mul(self.coverage_offsets.len, @sizeOf(u32)));
        return add(total, try mul(self.coverage_indices.len, @sizeOf(u32)));
    }
    pub fn coverageForBatch(self: @This(), batch_index: usize) ![]const u32 {
        if (self.coverage_offsets.len != self.batches.len + 1 or self.coverage_offsets[0] != 0 or batch_index >= self.batches.len or self.coverage_offsets[self.batches.len] != @as(u32, @intCast(self.coverage_indices.len))) return error.InvalidTrainingAugmentation;
        const begin: usize = @intCast(self.coverage_offsets[batch_index]);
        const end: usize = @intCast(self.coverage_offsets[batch_index + 1]);
        if (end < begin or end > self.coverage_indices.len) return error.InvalidTrainingAugmentation;
        const result = self.coverage_indices[begin..end];
        for (result, 0..) |raw, index| if (raw >= 512 or (index > 0 and result[index - 1] >= raw)) return error.InvalidTrainingAugmentation;
        return result;
    }
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.coverage_indices);
        self.allocator.free(self.coverage_offsets);
        self.allocator.free(self.positions);
        self.allocator.free(self.values);
        self.allocator.free(self.batches);
        self.* = undefined;
    }
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}

fn conservativeWorkspaceBytes(batch_count: usize, head_count: usize, maximum_rows: usize, maximum_normalized: usize) !usize {
    const value_count = try mul(try mul(batch_count, selector.max_selected), selector.max_dim);
    const position_count = try mul(batch_count, selector.max_selected);
    var peak = try mul(batch_count, @sizeOf(engine.ReferenceBatch));
    peak = try add(peak, try mul(value_count, @sizeOf(f32)));
    peak = try add(peak, try mul(position_count, @sizeOf(u64)));
    peak = try add(peak, try mul(try add(batch_count, 1), @sizeOf(u32)));
    peak = try add(peak, try mul(try mul(batch_count, selector.max_selected), @sizeOf(u32)));
    peak = try add(peak, try mul(try mul(head_count, maximum_rows), @sizeOf(f64) + @sizeOf(u32)));
    peak = try add(peak, maximum_normalized);
    peak = try add(peak, try mul(maximum_rows, 2 * @sizeOf(f64) + @sizeOf(bool)));
    peak = try add(peak, try mul(selector.max_dim, @sizeOf(f64)));
    peak = try add(peak, try mul(513, @sizeOf(u32)));
    peak = try add(peak, try mul(512 * selector.max_critical_keys_per_query, @sizeOf(u32)));
    return add(peak, try selector.workspaceBytes(512, selector.max_dim, maximum_rows, 256));
}
fn admitWorkspace(required: usize, cap: usize) !void {
    if (required > cap) return error.CompactionWorkspaceLimit;
}

const Normalized = struct {
    allocator: std.mem.Allocator,
    keys: []f64,
    values: []f64,
    bias: []f64,
    fn deinit(self: *@This()) void {
        self.allocator.free(self.bias);
        self.allocator.free(self.values);
        self.allocator.free(self.keys);
        self.* = undefined;
    }
};

fn normalize(allocator: std.mem.Allocator, head: checkpoint.Head) !Normalized {
    const keys = try allocator.alloc(f64, head.keys.len);
    errdefer allocator.free(keys);
    const values = try allocator.alloc(f64, head.values.len);
    errdefer allocator.free(values);
    const bias = try allocator.alloc(f64, head.bias.len);
    errdefer allocator.free(bias);
    for (keys, head.keys) |*out, value| out.* = value;
    for (values, head.values) |*out, value| out.* = value;
    for (bias, head.bias) |*out, value| out.* = value;
    return .{ .allocator = allocator, .keys = keys, .values = values, .bias = bias };
}

const MappedHead = struct { donor: usize, head_index: usize, head: checkpoint.Head, scale: f64, softcap: f64 };
fn headForBatch(state: checkpoint.State, layout: engine.Layout, batch: engine.ReferenceBatch) !MappedHead {
    if (batch.consumer_layer >= state.layer_donors.len or batch.consumer_layer >= layout.query_heads_by_layer.len) return error.InvalidTrainingAugmentation;
    const donor = state.layer_donors[batch.consumer_layer];
    if (donor >= state.donors.len or donor >= layout.scales_by_donor.len or donor >= layout.softcaps_by_donor.len) return error.InvalidTrainingAugmentation;
    const heads = state.donors[donor].heads;
    if (heads.len == 0 or layout.query_heads_by_layer[batch.consumer_layer] % heads.len != 0) return error.InvalidTrainingAugmentation;
    const group = layout.query_heads_by_layer[batch.consumer_layer] / heads.len;
    if (group == 0) return error.InvalidTrainingAugmentation;
    const head_index = batch.query_head / group;
    if (head_index >= heads.len) return error.InvalidTrainingAugmentation;
    return .{ .donor = donor, .head_index = head_index, .head = heads[head_index], .scale = layout.scales_by_donor[donor], .softcap = layout.softcaps_by_donor[donor] };
}

const HeadAuthority = struct {
    allocator: std.mem.Allocator,
    donor: usize,
    head_index: usize,
    source_key_weights: []f64,
    baseline_covered_keys: []u32,
    fn deinit(self: *@This()) void {
        self.allocator.free(self.baseline_covered_keys);
        self.allocator.free(self.source_key_weights);
        self.* = undefined;
    }
};

fn authorityFor(authorities: []HeadAuthority, mapped: MappedHead) !*HeadAuthority {
    for (authorities) |*authority| if (authority.donor == mapped.donor and authority.head_index == mapped.head_index) return authority;
    return error.InvalidTrainingAugmentation;
}

fn queryToF64(query: []const f32, out: []f64) !void {
    if (query.len != out.len) return error.InvalidTrainingAugmentation;
    for (query, out) |value, *dst| {
        if (!std.math.isFinite(value)) return error.InvalidTrainingAugmentation;
        dst.* = value;
    }
}

fn roleExcluded(position: u64, heldout_positions: []const u64) bool {
    for (heldout_positions) |heldout| if (heldout == position) return true;
    return false;
}

fn accumulate(
    allocator: std.mem.Allocator,
    head: solver.Head,
    queries: []const f32,
    dim: usize,
    weights: []f64,
    covered: ?[]bool,
    offsets: ?[]u32,
    keys: ?*std.ArrayList(u32),
    positions: ?[]const u64,
    heldout_positions: []const u64,
    query: []f64,
    probabilities: []f64,
    control: solver.Control,
) !void {
    if (queries.len == 0 or queries.len % dim != 0 or weights.len != head.rows or query.len != dim or probabilities.len != head.rows) return error.InvalidTrainingAugmentation;
    const rows = queries.len / dim;
    if (positions) |p| if (p.len != rows) return error.InvalidTrainingAugmentation;
    if (offsets) |o| if (o.len != rows + 1 or keys == null) return error.InvalidTrainingAugmentation;
    for (0..rows) |qi| {
        try control.check();
        if (positions) |p| if (roleExcluded(p[qi], heldout_positions)) {
            if (offsets) |o| o[qi + 1] = @intCast(keys.?.items.len);
            continue;
        };
        try queryToF64(queries[qi * dim ..][0..dim], query);
        _ = try solver.attend(head, query, probabilities, null);
        var count: usize = 0;
        for (probabilities, weights, 0..) |raw_probability, *maximum, source_row| {
            const probability: f64 = @floatCast(@as(f32, @floatCast(raw_probability)));
            if (!std.math.isFinite(probability) or probability < 0 or probability > 1) return error.NonFiniteAttention;
            maximum.* = @max(maximum.*, probability);
            if (probability >= threshold) {
                if (covered) |set| set[source_row] = true;
                if (keys) |list| {
                    if (count == selector.max_critical_keys_per_query) return error.InvalidTrainingAugmentation;
                    try list.append(allocator, @intCast(source_row));
                    count += 1;
                }
            }
        }
        if (offsets) |o| o[qi + 1] = @intCast(keys.?.items.len);
    }
}

fn accumulateAccelerated(
    allocator: std.mem.Allocator,
    head: solver.Head,
    queries: []const f32,
    weights: []f64,
    covered: ?[]bool,
    offsets: ?[]u32,
    keys: ?*std.ArrayList(u32),
    positions: ?[]const u64,
    heldout_positions: []const u64,
    backend: accelerate_scan.Backend,
    gather: bool,
    external_live_bytes: usize,
    maximum_workspace_bytes: usize,
    control: solver.Control,
) !void {
    if (queries.len == 0 or queries.len % head.dim != 0 or weights.len != head.rows) return error.InvalidTrainingAugmentation;
    const rows = queries.len / head.dim;
    if (positions) |p| if (p.len != rows) return error.InvalidTrainingAugmentation;
    if (offsets) |o| if (o.len != rows + 1 or keys == null) return error.InvalidTrainingAugmentation;
    const Callback = struct {
        allocator: std.mem.Allocator,
        weights: []f64,
        covered: ?[]bool,
        offsets: ?[]u32,
        keys: ?*std.ArrayList(u32),
        original_rows: ?[]const u32 = null,
        original_row_base: usize = 0,
        next_original_row: usize = 0,
        fn consume(raw: ?*anyopaque, local_row: usize, probabilities: []const f32) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            const row = if (self.original_rows) |mapping| mapping[local_row] else self.original_row_base + local_row;
            if (self.offsets) |o| {
                while (self.next_original_row < row) : (self.next_original_row += 1) o[self.next_original_row + 1] = @intCast(self.keys.?.items.len);
            }
            var count: usize = 0;
            for (probabilities, self.weights, 0..) |probability32, *maximum, source_row| {
                const probability: f64 = probability32;
                if (!std.math.isFinite(probability) or probability < 0 or probability > 1) return error.NonFiniteAttention;
                maximum.* = @max(maximum.*, probability);
                if (probability >= threshold) {
                    if (self.covered) |set| set[source_row] = true;
                    if (self.keys) |list| {
                        if (count == selector.max_critical_keys_per_query) return error.InvalidTrainingAugmentation;
                        try list.append(self.allocator, @intCast(source_row));
                        count += 1;
                    }
                }
            }
            if (self.offsets) |o| o[row + 1] = @intCast(self.keys.?.items.len);
            self.next_original_row = row + 1;
        }
    };
    var callback: Callback = .{ .allocator = allocator, .weights = weights, .covered = covered, .offsets = offsets, .keys = keys };
    if (gather and positions != null) {
        try control.check();
        var eligible: usize = 0;
        for (positions.?) |position| {
            try control.check();
            eligible += @intFromBool(!roleExcluded(position, heldout_positions));
        }
        if (eligible == 0) {
            if (offsets) |o| {
                for (1..o.len) |index| o[index] = @intCast(keys.?.items.len);
            }
            return;
        }
        const values_count = try mul(eligible, head.dim);
        const gather_bytes = try add(try mul(values_count, @sizeOf(f32)), try mul(eligible, @sizeOf(u32)));
        if (external_live_bytes > maximum_workspace_bytes or gather_bytes > maximum_workspace_bytes - external_live_bytes) return error.CompactionWorkspaceLimit;
        const gathered = try allocator.alloc(f32, values_count);
        defer allocator.free(gathered);
        const mapping = try allocator.alloc(u32, eligible);
        defer allocator.free(mapping);
        var at: usize = 0;
        for (positions.?, 0..) |position, row| {
            try control.check();
            if (roleExcluded(position, heldout_positions)) continue;
            mapping[at] = @intCast(row);
            @memcpy(gathered[at * head.dim ..][0..head.dim], queries[row * head.dim ..][0..head.dim]);
            at += 1;
        }
        if (at != eligible) return error.InvalidTrainingAugmentation;
        callback.original_rows = mapping;
        try backend.run(allocator, head, gathered, try add(external_live_bytes, gather_bytes), maximum_workspace_bytes, control, .{ .context = &callback, .row_fn = Callback.consume });
        if (offsets) |o| {
            while (callback.next_original_row < rows) : (callback.next_original_row += 1) o[callback.next_original_row + 1] = @intCast(keys.?.items.len);
        }
        return;
    }
    var row: usize = 0;
    while (row < rows) {
        try control.check();
        if (positions) |p| if (roleExcluded(p[row], heldout_positions)) {
            if (offsets) |o| o[row + 1] = @intCast(keys.?.items.len);
            row += 1;
            continue;
        };
        const begin = row;
        while (row < rows) : (row += 1) if (positions) |p| {
            if (roleExcluded(p[row], heldout_positions)) break;
        };
        callback.original_row_base = begin;
        callback.next_original_row = begin;
        try backend.run(allocator, head, queries[begin * head.dim .. row * head.dim], external_live_bytes, maximum_workspace_bytes, control, .{ .context = &callback, .row_fn = Callback.consume });
    }
}

fn appendCovered(allocator: std.mem.Allocator, covered: []const bool) ![]u32 {
    var count: usize = 0;
    for (covered) |value| if (value) {
        count += 1;
    };
    const out = try allocator.alloc(u32, count);
    var at: usize = 0;
    for (covered, 0..) |value, i| if (value) {
        out[at] = @intCast(i);
        at += 1;
    };
    return out;
}

fn pressureRanges() [23]@import("attention_match_cumulative_cache.zig").Range {
    var ranges: [23]@import("attention_match_cumulative_cache.zig").Range = undefined;
    var start: usize = 128;
    for (&ranges, 0..) |*range, i| {
        range.* = .{ .start = start, .rows = if (i + 1 == ranges.len) 128 else 256 };
        start += range.rows;
    }
    return ranges;
}

pub fn derive(
    allocator: std.mem.Allocator,
    state: checkpoint.State,
    layout: engine.Layout,
    primary: []const engine.ReferenceBatch,
    future: []const engine.ReferenceBatch,
    dense: *const augmentation.Owned,
    maximum_workspace_bytes: usize,
    control: solver.Control,
) !Owned {
    return deriveWithProbabilityBackend(allocator, state, layout, primary, future, dense, maximum_workspace_bytes, control, .{});
}

pub fn deriveWithProbabilityBackend(
    allocator: std.mem.Allocator,
    state: checkpoint.State,
    layout: engine.Layout,
    primary: []const engine.ReferenceBatch,
    future: []const engine.ReferenceBatch,
    dense: *const augmentation.Owned,
    maximum_workspace_bytes: usize,
    control: solver.Control,
    probability: ProbabilityOptions,
) !Owned {
    // The production accelerated authority is specifically the gathered,
    // role-safe path. Keep scalar default byte-identical and reject ambiguous
    // backend/mode combinations rather than mislabeling span execution.
    if ((probability.backend == null) != (!probability.gather_eligible_historical)) return error.InvalidTrainingAugmentation;
    if (primary.len == 0 or primary.len != future.len or primary.len != dense.batches.len or dense.counts.len != dense.batches.len) return error.InvalidTrainingAugmentation;
    const batch_count = primary.len;
    for (primary, future, dense.batches, dense.counts) |p, f, h, count| {
        if (count != 512 or p.consumer_layer != f.consumer_layer or p.consumer_layer != h.consumer_layer or p.query_head != f.query_head or p.query_head != h.query_head or p.train_positions.len != 256 or f.train_positions.len != 256 or h.train_positions.len != 512 or p.heldout_positions.len != 64 or p.train.len != 256 * selector.max_dim or f.train.len != 256 * selector.max_dim or h.train.len != 512 * selector.max_dim) return error.InvalidTrainingAugmentation;
    }
    const value_count = try mul(try mul(batch_count, selector.max_selected), selector.max_dim);
    const position_count = try mul(batch_count, selector.max_selected);
    var output_live = try mul(batch_count, @sizeOf(engine.ReferenceBatch));
    output_live = try add(output_live, try mul(value_count, @sizeOf(f32)));
    output_live = try add(output_live, try mul(position_count, @sizeOf(u64)));
    output_live = try add(output_live, try mul(try add(batch_count, 1), @sizeOf(u32)));
    output_live = try add(output_live, try mul(position_count, @sizeOf(u32)));
    var identities: [3][2]usize = undefined;
    var identity_count: usize = 0;
    var owner_counts: [3]usize = @splat(0);
    var maximum_rows: usize = 0;
    var maximum_normalized: usize = 0;
    for (primary) |batch| {
        const mapped = try headForBatch(state, layout, batch);
        const owner_index: usize = switch (state.donors[mapped.donor].layer) {
            4 => 0,
            9 => 1,
            14 => 2,
            else => return error.InvalidTrainingAugmentation,
        };
        owner_counts[owner_index] += 1;
        var found = false;
        for (identities[0..identity_count]) |identity| if (identity[0] == mapped.donor and identity[1] == mapped.head_index) {
            found = true;
            break;
        };
        if (!found) {
            if (identity_count == identities.len) return error.InvalidTrainingAugmentation;
            identities[identity_count] = .{ mapped.donor, mapped.head_index };
            identity_count += 1;
        }
        maximum_rows = @max(maximum_rows, mapped.head.positions.len);
        maximum_normalized = @max(maximum_normalized, try mul(try add(mapped.head.keys.len, try add(mapped.head.values.len, mapped.head.bias.len)), @sizeOf(f64)));
    }
    if (identity_count != 3 or batch_count != 56 or !std.meta.eql(owner_counts, [3]usize{ 8, 8, 40 })) return error.InvalidTrainingAugmentation;
    const conservative_peak = try conservativeWorkspaceBytes(batch_count, identity_count, maximum_rows, maximum_normalized);
    try admitWorkspace(conservative_peak, maximum_workspace_bytes);
    var authorities = std.ArrayList(HeadAuthority).empty;
    defer {
        for (authorities.items) |*authority| authority.deinit();
        authorities.deinit(allocator);
    }
    for (primary) |batch| {
        const mapped = try headForBatch(state, layout, batch);
        var found = false;
        for (authorities.items) |authority| if (authority.donor == mapped.donor and authority.head_index == mapped.head_index) {
            found = true;
            break;
        };
        if (found) continue;
        var normalized = try normalize(allocator, mapped.head);
        defer normalized.deinit();
        const original: solver.Head = .{ .keys = normalized.keys, .values = normalized.values, .bias = normalized.bias, .rows = mapped.head.positions.len, .dim = mapped.head.dim, .scale = mapped.scale, .softcap = mapped.softcap };
        const weights = try allocator.alloc(f64, original.rows);
        errdefer allocator.free(weights);
        @memset(weights, 0);
        const covered = try allocator.alloc(bool, original.rows);
        defer allocator.free(covered);
        @memset(covered, false);
        const query = try allocator.alloc(f64, original.dim);
        defer allocator.free(query);
        const probabilities = try allocator.alloc(f64, original.rows);
        defer allocator.free(probabilities);
        var scan_live = output_live;
        scan_live = try add(scan_live, try mul(authorities.capacity, @sizeOf(HeadAuthority)));
        for (authorities.items) |authority| {
            scan_live = try add(scan_live, try mul(authority.source_key_weights.len, @sizeOf(f64)));
            scan_live = try add(scan_live, try mul(authority.baseline_covered_keys.len, @sizeOf(u32)));
        }
        scan_live = try add(scan_live, try mul(try add(original.keys.len, try add(original.values.len, original.bias.?.len)), @sizeOf(f64)));
        scan_live = try add(scan_live, try mul(original.rows, @sizeOf(f64) + @sizeOf(bool) + @sizeOf(f64)));
        scan_live = try add(scan_live, try mul(original.dim, @sizeOf(f64)));
        for (primary, future, dense.batches) |p, f, h| {
            const other = try headForBatch(state, layout, p);
            if (other.donor != mapped.donor or other.head_index != mapped.head_index) continue;
            if (probability.backend) |backend| {
                try accumulateAccelerated(allocator, original, p.train, weights, covered, null, null, null, &.{}, backend, false, scan_live, maximum_workspace_bytes, control);
                try accumulateAccelerated(allocator, original, f.train, weights, covered, null, null, null, &.{}, backend, false, scan_live, maximum_workspace_bytes, control);
                try accumulateAccelerated(allocator, original, h.train, weights, null, null, null, h.train_positions, p.heldout_positions, backend, probability.gather_eligible_historical, scan_live, maximum_workspace_bytes, control);
            } else {
                try accumulate(allocator, original, p.train, original.dim, weights, covered, null, null, null, &.{}, query, probabilities, control);
                try accumulate(allocator, original, f.train, original.dim, weights, covered, null, null, null, &.{}, query, probabilities, control);
                try accumulate(allocator, original, h.train, original.dim, weights, null, null, null, h.train_positions, p.heldout_positions, query, probabilities, control);
            }
        }
        const baseline_keys = try appendCovered(allocator, covered);
        errdefer allocator.free(baseline_keys);
        try authorities.append(allocator, .{ .allocator = allocator, .donor = mapped.donor, .head_index = mapped.head_index, .source_key_weights = weights, .baseline_covered_keys = baseline_keys });
    }
    if (authorities.items.len != 3) return error.InvalidTrainingAugmentation;
    var authority_live: usize = 0;
    for (authorities.items) |authority| {
        authority_live = try add(authority_live, try mul(authority.source_key_weights.len, @sizeOf(f64)));
        authority_live = try add(authority_live, try mul(authority.baseline_covered_keys.len, @sizeOf(u32)));
    }
    const authority_storage_live = try add(authority_live, try mul(authorities.capacity, @sizeOf(HeadAuthority)));
    if (try add(output_live, authority_storage_live) > maximum_workspace_bytes) return error.CompactionWorkspaceLimit;
    var pressure_inputs: [3]pressure.HeadInput = undefined;
    for (authorities.items, &pressure_inputs) |authority, *input| input.* = .{
        .donor_layer = state.donors[authority.donor].layer,
        .head_index = authority.head_index,
        .maximum_probability = authority.source_key_weights,
    };
    const exact_reservation = blk: {
        if (maximum_rows != 7_936 and maximum_rows != 8_128 and maximum_rows != 32_512) return error.UnsupportedShape;
        const shape_digest = probability.e2_shape_authority_sha256 orelse return error.UnsupportedShape;
        const evidence = try e2_pressure.deriveForRows(allocator, &pressure_inputs, maximum_rows, probability.e2_source_physical_bytes, shape_digest, maximum_workspace_bytes - authority_storage_live, control);
        break :blk .{ .policy_sha256 = evidence.policy_sha256, .evidence_sha256 = evidence.evidence_sha256, .chosen_donor_layer = evidence.chosen_exact_donor_layer, .chosen_head_index = evidence.chosen_exact_head_index, .pressure_infeasible_count = evidence.pressure_infeasible_count, .heads = evidence.exact_head_pressure };
    };
    const batches = try allocator.alloc(engine.ReferenceBatch, batch_count);
    errdefer allocator.free(batches);
    const values = try allocator.alloc(f32, value_count);
    errdefer allocator.free(values);
    const positions = try allocator.alloc(u64, position_count);
    errdefer allocator.free(positions);
    const coverage_offsets = try allocator.alloc(u32, try add(batch_count, 1));
    errdefer allocator.free(coverage_offsets);
    coverage_offsets[0] = 0;
    var coverage_indices_list = std.ArrayList(u32).empty;
    errdefer coverage_indices_list.deinit(allocator);
    try coverage_indices_list.ensureTotalCapacityPrecise(allocator, position_count);
    var evidence: Evidence = .{ .batch_count = batch_count, .eligible_rows = 0, .excluded_rows = 0, .baseline_covered_key_count = 0, .newly_covered_key_count = 0, .coverage_selected_rows = 0, .kcenter_selected_rows = 0, .selection_digest = undefined, .derived_historical_sha256 = undefined, .coverage_authority_sha256 = undefined, .exact_head_policy_sha256 = exact_reservation.policy_sha256, .exact_head_evidence_sha256 = exact_reservation.evidence_sha256, .chosen_exact_donor_layer = exact_reservation.chosen_donor_layer, .chosen_exact_head_index = exact_reservation.chosen_head_index, .pressure_infeasible_count = exact_reservation.pressure_infeasible_count, .exact_head_pressure = exact_reservation.heads };
    var aggregate = std.crypto.hash.sha2.Sha256.init(.{});
    aggregate.update("antfly.e2-production768-historical-coverage-selection.v1\x00");
    if (probability.backend != null) {
        aggregate.update(accelerated_policy);
        aggregate.update(accelerate_scan.policy);
        aggregate.update(&.{@intFromBool(probability.gather_eligible_historical)});
    }
    var coverage_authority = std.crypto.hash.sha2.Sha256.init(.{});
    coverage_authority.update("antfly.e2-production768-historical-coverage-csr-authority.v1\x00");
    var batch_count_bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &batch_count_bytes, @intCast(batch_count), .little);
    coverage_authority.update(&batch_count_bytes);
    for (primary, future, dense.batches, batches, 0..) |p, f, h, *out, batch_index| {
        try control.check();
        if (dense.counts[batch_index] != 512 or p.consumer_layer != f.consumer_layer or p.consumer_layer != h.consumer_layer or p.query_head != f.query_head or p.query_head != h.query_head or p.train_positions.len != 256 or f.train_positions.len != 256 or h.train_positions.len != 512 or p.heldout_positions.len != 64 or p.train.len != 256 * selector.max_dim or f.train.len != 256 * selector.max_dim or h.train.len != 512 * selector.max_dim) return error.InvalidTrainingAugmentation;
        const mapped = try headForBatch(state, layout, p);
        const authority = try authorityFor(authorities.items, mapped);
        var normalized = try normalize(allocator, mapped.head);
        defer normalized.deinit();
        const original: solver.Head = .{ .keys = normalized.keys, .values = normalized.values, .bias = normalized.bias, .rows = mapped.head.positions.len, .dim = mapped.head.dim, .scale = mapped.scale, .softcap = mapped.softcap };
        if (original.dim != selector.max_dim or original.rows > std.math.maxInt(u32)) return error.InvalidTrainingAugmentation;
        var retained = try add(output_live, authority_storage_live);
        retained = try add(retained, try mul(try add(mapped.head.keys.len, try add(mapped.head.values.len, mapped.head.bias.len)), @sizeOf(f64)));
        retained = try add(retained, try mul(original.rows, 2 * @sizeOf(f64)));
        retained = try add(retained, try mul(original.dim, @sizeOf(f64)));
        retained = try add(retained, try mul(513, @sizeOf(u32)));
        retained = try add(retained, try mul(512 * selector.max_critical_keys_per_query, @sizeOf(u32)));
        const selector_workspace = try selector.workspaceBytes(512, original.dim, original.rows, 256);
        if (try add(retained, selector_workspace) > maximum_workspace_bytes) return error.CompactionWorkspaceLimit;
        const query = try allocator.alloc(f64, original.dim);
        defer allocator.free(query);
        const probabilities = try allocator.alloc(f64, original.rows);
        defer allocator.free(probabilities);
        const offsets = try allocator.alloc(u32, 513);
        defer allocator.free(offsets);
        @memset(offsets, 0);
        var candidate_keys = std.ArrayList(u32).empty;
        defer candidate_keys.deinit(allocator);
        try candidate_keys.ensureTotalCapacity(allocator, 512 * selector.max_critical_keys_per_query);
        // Recompute only this consumer's CSR. Global max-P weights and P/F
        // coverage were frozen across every consumer mapped to this source head.
        const scratch_weights = try allocator.dupe(f64, authority.source_key_weights);
        defer allocator.free(scratch_weights);
        if (probability.backend) |backend|
            try accumulateAccelerated(allocator, original, h.train, scratch_weights, null, offsets, &candidate_keys, h.train_positions, p.heldout_positions, backend, probability.gather_eligible_historical, retained, maximum_workspace_bytes, control)
        else
            try accumulate(allocator, original, h.train, original.dim, scratch_weights, null, offsets, &candidate_keys, h.train_positions, p.heldout_positions, query, probabilities, control);
        var selected = try selector.select(allocator, .{ .queries = h.train, .positions = h.train_positions, .heldout_positions = p.heldout_positions, .dim = original.dim, .selected_count = 256, .source_key_weights = authority.source_key_weights, .baseline_covered_keys = authority.baseline_covered_keys, .candidate_offsets = offsets, .candidate_keys = candidate_keys.items }, maximum_workspace_bytes - retained, control);
        defer selected.deinit();
        const value_begin = batch_index * 256 * original.dim;
        const position_begin = batch_index * 256;
        for (selected.indices, 0..) |raw, selected_index| {
            const source_index: usize = raw;
            @memcpy(values[value_begin + selected_index * original.dim ..][0..original.dim], h.train[source_index * original.dim ..][0..original.dim]);
            positions[position_begin + selected_index] = h.train_positions[source_index];
        }
        out.* = .{ .consumer_layer = p.consumer_layer, .query_head = p.query_head, .train = values[value_begin..][0 .. 256 * original.dim], .heldout = &.{}, .train_positions = positions[position_begin..][0..256], .heldout_positions = &.{} };
        evidence.eligible_rows = try add(evidence.eligible_rows, selected.evidence.eligible_rows);
        evidence.excluded_rows = try add(evidence.excluded_rows, selected.evidence.excluded_rows);
        evidence.baseline_covered_key_count = try add(evidence.baseline_covered_key_count, selected.evidence.baseline_covered_key_count);
        evidence.newly_covered_key_count = try add(evidence.newly_covered_key_count, selected.evidence.newly_covered_key_count);
        evidence.coverage_selected_rows = try add(evidence.coverage_selected_rows, selected.evidence.coverage_selected_rows);
        evidence.kcenter_selected_rows = try add(evidence.kcenter_selected_rows, selected.evidence.kcenter_selected_rows);
        aggregate.update(std.mem.asBytes(&p.consumer_layer));
        aggregate.update(std.mem.asBytes(&p.query_head));
        aggregate.update(&selected.evidence.selection_digest);
        for (selected.coverage_indices) |raw| if (std.mem.indexOfScalar(u32, selected.indices, raw) == null) return error.InvalidTrainingAugmentation;
        try coverage_indices_list.appendSlice(allocator, selected.coverage_indices);
        coverage_offsets[batch_index + 1] = @intCast(coverage_indices_list.items.len);
        var framing: [16]u8 = undefined;
        std.mem.writeInt(u32, framing[0..4], p.consumer_layer, .little);
        std.mem.writeInt(u32, framing[4..8], @intCast(p.query_head), .little);
        std.mem.writeInt(u32, framing[8..12], @intCast(batch_index), .little);
        std.mem.writeInt(u32, framing[12..16], @intCast(selected.coverage_indices.len), .little);
        coverage_authority.update(&framing);
        for (selected.coverage_indices) |raw| {
            var raw_bytes: [4]u8 = undefined;
            std.mem.writeInt(u32, &raw_bytes, raw, .little);
            coverage_authority.update(&raw_bytes);
        }
    }
    evidence.selection_digest = aggregate.finalResult();
    var domain = std.crypto.hash.sha2.Sha256.init(.{});
    domain.update("antfly.production768-derived-historical-domain.v1\x00");
    domain.update(std.mem.sliceAsBytes(values));
    domain.update(std.mem.sliceAsBytes(positions));
    evidence.derived_historical_sha256 = domain.finalResult();
    for (coverage_offsets) |offset| {
        var offset_bytes: [4]u8 = undefined;
        std.mem.writeInt(u32, &offset_bytes, offset, .little);
        coverage_authority.update(&offset_bytes);
    }
    evidence.coverage_authority_sha256 = coverage_authority.finalResult();
    // ArrayList may need an allocate-copy fallback while shrinking capacity;
    // admit that transient separately from the retained CSR owner.
    if (try add(try add(output_live, authority_storage_live), try mul(position_count, @sizeOf(u32))) > maximum_workspace_bytes) return error.CompactionWorkspaceLimit;
    const coverage_indices = try coverage_indices_list.toOwnedSlice(allocator);
    errdefer allocator.free(coverage_indices);
    return .{ .allocator = allocator, .batches = batches, .values = values, .positions = positions, .coverage_offsets = coverage_offsets, .coverage_indices = coverage_indices, .evidence = evidence };
}

test "covered key serialization is sorted and exact" {
    const out = try appendCovered(std.testing.allocator, &.{ false, true, false, true });
    defer std.testing.allocator.free(out);
    try std.testing.expectEqualSlices(u32, &.{ 1, 3 }, out);
}

test "coverage CSR validates offsets ordering and live ownership" {
    var owned: Owned = .{ .allocator = std.testing.allocator, .batches = try std.testing.allocator.alloc(engine.ReferenceBatch, 2), .values = try std.testing.allocator.alloc(f32, 3), .positions = try std.testing.allocator.alloc(u64, 2), .coverage_offsets = try std.testing.allocator.dupe(u32, &.{ 0, 2, 3 }), .coverage_indices = try std.testing.allocator.dupe(u32, &.{ 1, 7, 9 }), .evidence = undefined };
    defer owned.deinit();
    try std.testing.expectEqualSlices(u32, &.{ 1, 7 }, try owned.coverageForBatch(0));
    try std.testing.expectEqualSlices(u32, &.{9}, try owned.coverageForBatch(1));
    const expected = 2 * @sizeOf(engine.ReferenceBatch) + 3 * @sizeOf(f32) + 2 * @sizeOf(u64) + 3 * @sizeOf(u32) + 3 * @sizeOf(u32);
    try std.testing.expectEqual(expected, try owned.liveBytes());
}

test "historical derivation workspace rejects a one-byte cap before probability work" {
    const required = try conservativeWorkspaceBytes(56, 8, 7936, (2 * 7936 * 512 + 7936) * @sizeOf(f64));
    try std.testing.expectError(error.CompactionWorkspaceLimit, admitWorkspace(required, 1));
}

test "derive rejects cap before allocator or probability work" {
    const allocator = std.testing.allocator;
    const p_values = try allocator.alloc(f32, 256 * 512);
    defer allocator.free(p_values);
    const h_values = try allocator.alloc(f32, 512 * 512);
    defer allocator.free(h_values);
    const p_positions = try allocator.alloc(u64, 256);
    defer allocator.free(p_positions);
    const h_positions = try allocator.alloc(u64, 512);
    defer allocator.free(h_positions);
    const heldout_positions = try allocator.alloc(u64, 64);
    defer allocator.free(heldout_positions);
    @memset(p_values, 0);
    @memset(h_values, 0);
    for (p_positions, 0..) |*position, i| position.* = 7936 + i;
    for (h_positions, 0..) |*position, i| position.* = 7936 + i;
    for (heldout_positions, 0..) |*position, i| position.* = 7936 + 448 + i;
    // Match the real E2 8/8/40 consumer census while keeping payloads
    // tiny: admission must reject before it can inspect probabilities.
    var heads: [3]checkpoint.Head = undefined;
    var donors: [3]checkpoint.Donor = undefined;
    var layer_donors: [35]u32 = @splat(0);
    layer_donors[9] = 1;
    for ([_]usize{ 14, 19, 24, 29, 34 }) |layer| layer_donors[layer] = 2;
    const query_heads: [35]u32 = @splat(8);
    const scales: [3]f64 = @splat(1);
    const softcaps: [3]f64 = @splat(0);
    const head_keys: [2 * 512]f32 = @splat(0);
    const head_values: [2 * 512]f32 = @splat(0);
    const head_bias = [_]f32{ 0, 0 };
    const head_positions = [_]u64{ 0, 1 };
    const coverage = [_]u64{ 1, 2 };
    for ([_]u32{ 4, 9, 14 }, 0..) |layer, i| {
        heads[i] = .{ .dim = 512, .positions = &head_positions, .coverage_ends = &coverage, .keys = &head_keys, .values = &head_values, .bias = &head_bias };
        donors[i] = .{ .layer = layer, .heads = heads[i..][0..1] };
    }
    var primary: [56]engine.ReferenceBatch = undefined;
    var future: [56]engine.ReferenceBatch = undefined;
    var historical: [56]engine.ReferenceBatch = undefined;
    var counts: [56]u32 = @splat(512);
    for (0..56) |i| {
        const layer: u32 = @intCast(4 + 5 * (i / 8));
        const query_head: u32 = @intCast(i % 8);
        primary[i] = .{ .consumer_layer = layer, .query_head = query_head, .train = p_values, .heldout = &.{}, .train_positions = p_positions, .heldout_positions = heldout_positions };
        future[i] = .{ .consumer_layer = layer, .query_head = query_head, .train = p_values, .heldout = &.{}, .train_positions = p_positions, .heldout_positions = &.{} };
        historical[i] = .{ .consumer_layer = layer, .query_head = query_head, .train = h_values, .heldout = &.{}, .train_positions = h_positions, .heldout_positions = &.{} };
    }
    const state: checkpoint.State = .{ .identity = undefined, .logical_length = 2, .layer_donors = &layer_donors, .donors = &donors };
    const layout: engine.Layout = .{ .query_heads_by_layer = &query_heads, .scales_by_donor = &scales, .softcaps_by_donor = &softcaps };
    const dense: augmentation.Owned = .{ .allocator = undefined, .manifest_bytes = @constCast(&[_]u8{}), .batches = &historical, .training = @constCast(&[_]f32{}), .positions = @constCast(&[_]u64{}), .consumer_layers = @constCast(&[_]u32{}), .query_heads = @constCast(&[_]u32{}), .counts = &counts, .dimensions = @constCast(&[_]u32{}), .source_length = 7936, .repeat_length = 512 };
    try std.testing.expectError(error.CompactionWorkspaceLimit, derive(std.testing.failing_allocator, state, layout, &primary, &future, &dense, 1, .{}));
}

test "TRAIN probability scan emits sorted critical CSR and max weights" {
    const head: solver.Head = .{ .keys = &.{ 1, 0, 0, 1 }, .values = &.{ 0, 0, 0, 0 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 2, .scale = 1, .softcap = 0 };
    var weights = [_]f64{ 0, 0 };
    var covered = [_]bool{ false, false };
    var offsets = [_]u32{ 0, 0 };
    var keys = std.ArrayList(u32).empty;
    defer keys.deinit(std.testing.allocator);
    var query: [2]f64 = undefined;
    var probabilities: [2]f64 = undefined;
    try accumulate(std.testing.allocator, head, &.{ 1, 0 }, 2, &weights, &covered, &offsets, &keys, null, &.{}, &query, &probabilities, .{});
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, keys.items);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2 }, &offsets);
    try std.testing.expect(covered[0] and covered[1]);
    try std.testing.expect(weights[0] > weights[1]);
}

test "logical heldout positions never affect TRAIN weights or CSR" {
    const head: solver.Head = .{ .keys = &.{ 1, 0, 0, 1 }, .values = &.{ 0, 0, 0, 0 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 2, .scale = 1, .softcap = 0 };
    var weights = [_]f64{ 0, 0 };
    var offsets = [_]u32{ 0, 0, 0 };
    var keys = std.ArrayList(u32).empty;
    defer keys.deinit(std.testing.allocator);
    var query: [2]f64 = undefined;
    var probabilities: [2]f64 = undefined;
    try accumulate(std.testing.allocator, head, &.{ 1, 0, 0, 1 }, 2, &weights, null, &offsets, &keys, &.{ 10, 11 }, &.{10}, &query, &probabilities, .{});
    try std.testing.expectEqualSlices(u32, &.{ 0, 0, 2 }, &offsets);
    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, keys.items);
    try std.testing.expect(weights[1] > weights[0]);
}

test "accelerated gathered TRAIN scan matches scalar and never reads heldout query values" {
    const Gemm = struct {
        fn run(_: *accelerate_scan.Context, m: c_int, n: c_int, k: c_int, alpha: f64, a: [*]const f64, lda: c_int, b: [*]const f64, ldb: c_int, beta: f64, c: [*]f64, ldc: c_int) void {
            for (0..@intCast(m)) |i| for (0..@intCast(n)) |j| {
                var sum: f64 = 0;
                for (0..@intCast(k)) |x| sum += a[i * @as(usize, @intCast(lda)) + x] * b[j * @as(usize, @intCast(ldb)) + x];
                const out = i * @as(usize, @intCast(ldc)) + j;
                c[out] = alpha * sum + if (beta == 0) 0 else beta * c[out];
            };
        }
    };
    const head: solver.Head = .{ .keys = &.{ 1, 0, 0, 1 }, .values = &.{ 0, 0, 0, 0 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 2, .scale = 1, .softcap = 0 };
    const queries = [_]f32{ 1, 0, std.math.nan(f32), std.math.inf(f32), 0, 1 };
    const positions = [_]u64{ 10, 11, 12 };
    var scalar_weights = [_]f64{ 0, 0 };
    var scalar_offsets = [_]u32{ 0, 0, 0, 0 };
    var scalar_keys = std.ArrayList(u32).empty;
    defer scalar_keys.deinit(std.testing.allocator);
    var query: [2]f64 = undefined;
    var probabilities: [2]f64 = undefined;
    try accumulate(std.testing.allocator, head, &queries, 2, &scalar_weights, null, &scalar_offsets, &scalar_keys, &positions, &.{11}, &query, &probabilities, .{});
    var accelerated_weights = [_]f64{ 0, 0 };
    var accelerated_offsets = [_]u32{ 0, 0, 0, 0 };
    var accelerated_keys = std.ArrayList(u32).empty;
    defer accelerated_keys.deinit(std.testing.allocator);
    var context: accelerate_scan.Context = .{ .gemm_fn = Gemm.run };
    try accumulateAccelerated(std.testing.allocator, head, &queries, &accelerated_weights, null, &accelerated_offsets, &accelerated_keys, &positions, &.{11}, accelerate_scan.backend(&context), true, 0, 1 << 20, .{});
    try std.testing.expectEqualSlices(f64, &scalar_weights, &accelerated_weights);
    try std.testing.expectEqualSlices(u32, &scalar_offsets, &accelerated_offsets);
    try std.testing.expectEqualSlices(u32, scalar_keys.items, accelerated_keys.items);
    try std.testing.expectEqual(@as(usize, 1), context.calls);
    const cancelled = solver.Control{ .check_fn = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    }.check };
    try std.testing.expectError(error.Cancelled, accumulateAccelerated(std.testing.allocator, head, &queries, &accelerated_weights, null, &accelerated_offsets, &accelerated_keys, &positions, &.{11}, accelerate_scan.backend(&context), true, 0, 1 << 20, cancelled));
}
