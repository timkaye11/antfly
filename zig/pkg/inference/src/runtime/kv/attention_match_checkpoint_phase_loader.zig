// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Diagnostic/offline host checkpoint phase loading. This module does not
//! claim that native device KV has been retired. Every reload is authenticated
//! against the initially frozen full state before one head may escape.
const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const solver = @import("attention_match.zig");

const Sha256 = std.crypto.hash.sha2.Sha256;

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn updateSlice(hash: *Sha256, values: anytype) void {
    hash.update(std.mem.sliceAsBytes(values));
}
fn updateScalar(hash: *Sha256, value: anytype) void {
    hash.update(std.mem.asBytes(&value));
}

pub const Entry = struct {
    consumer_layer: u32,
    query_head: u32,
    donor_index: usize,
    donor_layer: u32,
    head_index: usize,
    rows: usize,
    dim: usize,
    scale: f64,
    softcap: f64,
};

pub const AuthorityOwned = struct {
    allocator: std.mem.Allocator,
    identity: checkpoint.Identity,
    payload_sha256: [32]u8,
    state_sha256: [32]u8,
    resident_bytes: usize,
    maximum_head_clone_bytes: usize,
    entries: []Entry,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.entries);
        self.* = undefined;
    }

    pub fn liveBytes(self: @This()) !usize {
        return mul(self.entries.len, @sizeOf(Entry));
    }

    pub fn entry(self: @This(), consumer_layer: u32, query_head: u32) !Entry {
        var found: ?Entry = null;
        for (self.entries) |candidate| {
            if (candidate.consumer_layer != consumer_layer or candidate.query_head != query_head) continue;
            if (found != null) return error.InvalidTrainingAugmentationBinding;
            found = candidate;
        }
        return found orelse error.InvalidTrainingAugmentationBinding;
    }
};

fn stateDigest(state: checkpoint.State, control: solver.Control) ![32]u8 {
    var hash = Sha256.init(.{});
    hash.update("antfly.attention-match-checkpoint-phase-state.v1");
    hash.update(&state.identity.weights);
    hash.update(&state.identity.tokenizer);
    hash.update(&state.identity.template);
    hash.update(&state.identity.layout);
    updateScalar(&hash, state.logical_length);
    updateSlice(&hash, state.tokens);
    updateSlice(&hash, state.logits);
    updateSlice(&hash, state.layer_donors);
    updateScalar(&hash, state.donors.len);
    for (state.donors, 0..) |donor, donor_index| {
        try control.check();
        updateScalar(&hash, donor_index);
        updateScalar(&hash, donor.layer);
        updateScalar(&hash, donor.window);
        updateScalar(&hash, donor.heads.len);
        for (donor.heads, 0..) |head, head_index| {
            try control.check();
            updateScalar(&hash, head_index);
            updateScalar(&hash, head.dim);
            updateScalar(&hash, head.compacted_until);
            updateSlice(&hash, head.positions);
            updateSlice(&hash, head.coverage_ends);
            updateSlice(&hash, head.keys);
            updateSlice(&hash, head.values);
            updateSlice(&hash, head.bias);
            updateScalar(&hash, head.fp8.first_row);
            updateScalar(&hash, head.fp8.rows);
            updateScalar(&hash, head.fp8.key_scale);
            updateScalar(&hash, head.fp8.value_scale);
            updateScalar(&hash, head.compressed.version);
            for (head.compressed.spans) |span| {
                updateScalar(&hash, span.first);
                updateScalar(&hash, span.rows);
                updateScalar(&hash, span.offset);
            }
            hash.update(head.compressed.keys);
            hash.update(head.compressed.values);
        }
    }
    return hash.finalResult();
}

fn mapEntry(state: checkpoint.State, layout: engine.Layout, batch: engine.ReferenceBatch) !Entry {
    if (batch.consumer_layer >= state.layer_donors.len or batch.consumer_layer >= layout.query_heads_by_layer.len)
        return error.InvalidTrainingAugmentationBinding;
    const donor_index = state.layer_donors[batch.consumer_layer];
    if (donor_index >= state.donors.len or donor_index >= layout.scales_by_donor.len or donor_index >= layout.softcaps_by_donor.len)
        return error.InvalidTrainingAugmentationBinding;
    const donor = state.donors[donor_index];
    if (donor.heads.len == 0 or layout.query_heads_by_layer[batch.consumer_layer] % donor.heads.len != 0)
        return error.InvalidTrainingAugmentationBinding;
    const group = layout.query_heads_by_layer[batch.consumer_layer] / donor.heads.len;
    if (group == 0) return error.InvalidTrainingAugmentationBinding;
    const head_index = batch.query_head / group;
    if (head_index >= donor.heads.len) return error.InvalidTrainingAugmentationBinding;
    const head = donor.heads[head_index];
    return .{ .consumer_layer = batch.consumer_layer, .query_head = batch.query_head, .donor_index = donor_index, .donor_layer = donor.layer, .head_index = head_index, .rows = head.positions.len, .dim = head.dim, .scale = layout.scales_by_donor[donor_index], .softcap = layout.softcaps_by_donor[donor_index] };
}

pub fn freezeAuthority(
    allocator: std.mem.Allocator,
    state: checkpoint.State,
    layout: engine.Layout,
    batches: []const engine.ReferenceBatch,
    payload_sha256: [32]u8,
    control: solver.Control,
) !AuthorityOwned {
    if (batches.len == 0 or std.mem.allEqual(u8, &payload_sha256, 0)) return error.InvalidTrainingAugmentationBinding;
    const entries = try allocator.alloc(Entry, batches.len);
    errdefer allocator.free(entries);
    var maximum_head_clone_bytes: usize = 0;
    for (batches, entries, 0..) |batch, *out, index| {
        try control.check();
        out.* = try mapEntry(state, layout, batch);
        for (entries[0..index]) |prior| if (prior.consumer_layer == out.consumer_layer and prior.query_head == out.query_head)
            return error.InvalidTrainingAugmentationBinding;
        const source = state.donors[out.donor_index].heads[out.head_index];
        var clone_bytes = try mul(try add(source.positions.len, source.coverage_ends.len), @sizeOf(u64));
        clone_bytes = try add(clone_bytes, try mul(try add(source.keys.len, try add(source.values.len, source.bias.len)), @sizeOf(f32)));
        clone_bytes = try add(clone_bytes, source.compressed.bytes());
        maximum_head_clone_bytes = @max(maximum_head_clone_bytes, clone_bytes);
    }
    return .{ .allocator = allocator, .identity = state.identity, .payload_sha256 = payload_sha256, .state_sha256 = try stateDigest(state, control), .resident_bytes = try state.residentBytes(), .maximum_head_clone_bytes = maximum_head_clone_bytes, .entries = entries };
}

pub const Reloaded = struct {
    checkpoint_owned: checkpoint.Owned,
    payload_sha256: [32]u8,
    /// Includes every temporary allocation made by the callback, including
    /// query payloads if it reuses a whole-input reader.
    peak_live_bytes: usize,
};

pub const ReloadFn = *const fn (?*anyopaque, std.mem.Allocator, solver.Control) anyerror!Reloaded;
pub const Loader = struct {
    context: ?*anyopaque,
    reload_fn: ReloadFn,
    maximum_reload_live_bytes: usize,
};

fn verifyReload(authority: AuthorityOwned, reload: *const Reloaded, control: solver.Control) !void {
    const digest = try stateDigest(reload.checkpoint_owned.state, control);
    if (!reload.checkpoint_owned.state.identity.eql(authority.identity) or
        !std.mem.eql(u8, &reload.payload_sha256, &authority.payload_sha256) or
        !std.mem.eql(u8, &digest, &authority.state_sha256))
        return error.InvalidTrainingAugmentationBinding;
}

pub const HeadOwned = struct {
    allocator: std.mem.Allocator,
    entry: Entry,
    positions: []u64,
    coverage_ends: []u64,
    keys: []f32,
    values: []f32,
    bias: []f32,
    compressed_spans: []checkpoint.packed_storage.Span,
    compressed_keys: []u8,
    compressed_values: []u8,
    compressed_version: u32,
    fp8: checkpoint.codec.Fp8,
    compacted_until: u64,

    pub fn head(self: *const @This()) checkpoint.Head {
        return .{ .dim = @intCast(self.entry.dim), .compacted_until = self.compacted_until, .positions = self.positions, .coverage_ends = self.coverage_ends, .keys = self.keys, .values = self.values, .bias = self.bias, .fp8 = self.fp8, .compressed = .{ .version = self.compressed_version, .spans = self.compressed_spans, .keys = self.compressed_keys, .values = self.compressed_values } };
    }
    pub fn liveBytes(self: @This()) !usize {
        var value = try mul(try add(self.positions.len, self.coverage_ends.len), @sizeOf(u64));
        value = try add(value, try mul(try add(self.keys.len, try add(self.values.len, self.bias.len)), @sizeOf(f32)));
        value = try add(value, try mul(self.compressed_spans.len, @sizeOf(checkpoint.packed_storage.Span)));
        return add(value, try add(self.compressed_keys.len, self.compressed_values.len));
    }
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.compressed_values);
        self.allocator.free(self.compressed_keys);
        self.allocator.free(self.compressed_spans);
        self.allocator.free(self.bias);
        self.allocator.free(self.values);
        self.allocator.free(self.keys);
        self.allocator.free(self.coverage_ends);
        self.allocator.free(self.positions);
        self.* = undefined;
    }
};

fn cloneHeadControlled(allocator: std.mem.Allocator, entry: Entry, source: checkpoint.Head, control: solver.Control) !HeadOwned {
    if (source.positions.len != entry.rows or @as(usize, source.dim) != entry.dim) return error.InvalidTrainingAugmentationBinding;
    const positions = try allocator.dupe(u64, source.positions);
    errdefer allocator.free(positions);
    try control.check();
    const coverage = try allocator.dupe(u64, source.coverage_ends);
    errdefer allocator.free(coverage);
    try control.check();
    const keys = try allocator.dupe(f32, source.keys);
    errdefer allocator.free(keys);
    try control.check();
    const values = try allocator.dupe(f32, source.values);
    errdefer allocator.free(values);
    try control.check();
    const bias = try allocator.dupe(f32, source.bias);
    errdefer allocator.free(bias);
    try control.check();
    const spans = try allocator.dupe(checkpoint.packed_storage.Span, source.compressed.spans);
    errdefer allocator.free(spans);
    try control.check();
    const compressed_keys = try allocator.dupe(u8, source.compressed.keys);
    errdefer allocator.free(compressed_keys);
    try control.check();
    const compressed_values = try allocator.dupe(u8, source.compressed.values);
    return .{ .allocator = allocator, .entry = entry, .positions = positions, .coverage_ends = coverage, .keys = keys, .values = values, .bias = bias, .compressed_spans = spans, .compressed_keys = compressed_keys, .compressed_values = compressed_values, .compressed_version = source.compressed.version, .fp8 = source.fp8, .compacted_until = source.compacted_until };
}

fn admitReload(external_without_checkpoint: usize, loader: Loader, combined_cap: usize) !void {
    if (loader.maximum_reload_live_bytes == 0 or external_without_checkpoint > combined_cap or
        loader.maximum_reload_live_bytes > combined_cap - external_without_checkpoint)
        return error.CompactionWorkspaceLimit;
}

fn admitHeadReload(external_without_checkpoint: usize, loader: Loader, authority: AuthorityOwned, combined_cap: usize) !void {
    const reload_and_clone = try add(loader.maximum_reload_live_bytes, authority.maximum_head_clone_bytes);
    if (external_without_checkpoint > combined_cap or reload_and_clone > combined_cap - external_without_checkpoint)
        return error.CompactionWorkspaceLimit;
}

pub fn loadHead(
    allocator: std.mem.Allocator,
    authority: AuthorityOwned,
    loader: Loader,
    consumer_layer: u32,
    query_head: u32,
    external_without_checkpoint: usize,
    combined_cap: usize,
    control: solver.Control,
) !HeadOwned {
    try control.check();
    const entry = try authority.entry(consumer_layer, query_head);
    try admitHeadReload(external_without_checkpoint, loader, authority, combined_cap);
    var reload = try loader.reload_fn(loader.context, allocator, control);
    defer reload.checkpoint_owned.deinit();
    if (reload.peak_live_bytes > loader.maximum_reload_live_bytes) return error.CompactionWorkspaceLimit;
    try verifyReload(authority, &reload, control);
    if (entry.donor_index >= reload.checkpoint_owned.state.donors.len or entry.head_index >= reload.checkpoint_owned.state.donors[entry.donor_index].heads.len)
        return error.InvalidTrainingAugmentationBinding;
    try control.check();
    const source = reload.checkpoint_owned.state.donors[entry.donor_index].heads[entry.head_index];
    const exact_clone_bytes = blk: {
        var bytes = try mul(try add(source.positions.len, source.coverage_ends.len), @sizeOf(u64));
        bytes = try add(bytes, try mul(try add(source.keys.len, try add(source.values.len, source.bias.len)), @sizeOf(f32)));
        break :blk try add(bytes, source.compressed.bytes());
    };
    if (reload.peak_live_bytes > combined_cap - external_without_checkpoint or
        exact_clone_bytes > combined_cap - external_without_checkpoint - reload.peak_live_bytes)
        return error.CompactionWorkspaceLimit;
    return cloneHeadControlled(allocator, entry, source, control);
}

pub fn loadFinal(
    allocator: std.mem.Allocator,
    authority: AuthorityOwned,
    loader: Loader,
    external_without_checkpoint: usize,
    combined_cap: usize,
    control: solver.Control,
) !checkpoint.Owned {
    try control.check();
    try admitReload(external_without_checkpoint, loader, combined_cap);
    var reload = try loader.reload_fn(loader.context, allocator, control);
    errdefer reload.checkpoint_owned.deinit();
    if (reload.peak_live_bytes > loader.maximum_reload_live_bytes) return error.CompactionWorkspaceLimit;
    try verifyReload(authority, &reload, control);
    return reload.checkpoint_owned;
}

test "checkpoint phase reload cap rejects before callback" {
    const Never = struct {
        fn load(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Control) !Reloaded {
            return error.TestUnexpectedResult;
        }
    };
    const authority: AuthorityOwned = undefined;
    try std.testing.expectError(error.CompactionWorkspaceLimit, loadFinal(std.testing.allocator, authority, .{ .context = null, .reload_fn = Never.load, .maximum_reload_live_bytes = 8 }, 9, 16, .{}));
}

test "checkpoint phase head reload exact cap includes clone" {
    const authority: AuthorityOwned = .{ .allocator = undefined, .identity = undefined, .payload_sha256 = @splat(1), .state_sha256 = @splat(2), .resident_bytes = 8, .maximum_head_clone_bytes = 7, .entries = &.{} };
    const loader: Loader = .{ .context = null, .reload_fn = undefined, .maximum_reload_live_bytes = 11 };
    try admitHeadReload(5, loader, authority, 23);
    try std.testing.expectError(error.CompactionWorkspaceLimit, admitHeadReload(5, loader, authority, 22));
}

test "checkpoint phase authority entries have exact live charge" {
    var entries: [2]Entry = undefined;
    const authority: AuthorityOwned = .{ .allocator = undefined, .identity = undefined, .payload_sha256 = @splat(1), .state_sha256 = @splat(2), .resident_bytes = 0, .maximum_head_clone_bytes = 0, .entries = &entries };
    try std.testing.expectEqual(2 * @sizeOf(Entry), try authority.liveBytes());
    const cap = try add(17, try authority.liveBytes());
    try std.testing.expect(cap - try authority.liveBytes() == 17);
    try std.testing.expectError(error.CompactionWorkspaceLimit, admitReload(cap - 10, .{ .context = null, .reload_fn = undefined, .maximum_reload_live_bytes = 11 }, cap));
}

fn testCheckpoint(allocator: std.mem.Allocator, key: f32) !checkpoint.Owned {
    const positions = try allocator.dupe(u64, &.{0});
    errdefer allocator.free(positions);
    const coverage = try allocator.dupe(u64, &.{1});
    errdefer allocator.free(coverage);
    const keys = try allocator.dupe(f32, &.{key});
    errdefer allocator.free(keys);
    const values = try allocator.dupe(f32, &.{3});
    errdefer allocator.free(values);
    const bias = try allocator.dupe(f32, &.{0});
    errdefer allocator.free(bias);
    const heads = try allocator.dupe(checkpoint.Head, &.{.{ .dim = 1, .positions = positions, .coverage_ends = coverage, .keys = keys, .values = values, .bias = bias }});
    errdefer allocator.free(heads);
    const donors = try allocator.dupe(checkpoint.Donor, &.{.{ .layer = 0, .heads = heads }});
    errdefer allocator.free(donors);
    const tokens = try allocator.dupe(u32, &.{0});
    errdefer allocator.free(tokens);
    const logits = try allocator.dupe(f32, &.{0});
    errdefer allocator.free(logits);
    const layer_donors = try allocator.dupe(u32, &.{0});
    return .{ .allocator = allocator, .state = .{ .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) }, .logical_length = 1, .tokens = tokens, .logits = logits, .layer_donors = layer_donors, .donors = donors } };
}

const TestReload = struct {
    key: f32 = 2,
    calls: usize = 0,
    fn load(raw: ?*anyopaque, allocator: std.mem.Allocator, _: solver.Control) !Reloaded {
        const self: *@This() = @ptrCast(@alignCast(raw.?));
        self.calls += 1;
        const owned = try testCheckpoint(allocator, self.key);
        return .{ .peak_live_bytes = try owned.state.residentBytes(), .payload_sha256 = @splat(9), .checkpoint_owned = owned };
    }
};

test "checkpoint phase happy head final substitution cancellation and OOM" {
    const allocator = std.testing.allocator;
    var original = try testCheckpoint(allocator, 2);
    defer original.deinit();
    const batches = [_]engine.ReferenceBatch{.{ .consumer_layer = 0, .query_head = 0, .train = &.{}, .heldout = &.{}, .train_positions = &.{}, .heldout_positions = &.{} }};
    const layout: engine.Layout = .{ .query_heads_by_layer = &.{1}, .scales_by_donor = &.{1}, .softcaps_by_donor = &.{0} };
    var authority = try freezeAuthority(allocator, original.state, layout, &batches, @splat(9), .{});
    defer authority.deinit();
    var context: TestReload = .{};
    const loader: Loader = .{ .context = &context, .reload_fn = TestReload.load, .maximum_reload_live_bytes = authority.resident_bytes };
    var head = try loadHead(allocator, authority, loader, 0, 0, 0, 4096, .{});
    try std.testing.expectEqual(@as(f32, 2), head.keys[0]);
    head.deinit();
    var final = try loadFinal(allocator, authority, loader, 0, 4096, .{});
    final.deinit();
    context.key = 8;
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, loadFinal(allocator, authority, loader, 0, 4096, .{}));
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.CompactionCancelled;
        }
    };
    try std.testing.expectError(error.CompactionCancelled, loadHead(allocator, authority, loader, 0, 0, 0, 4096, .{ .check_fn = Cancel.check }));
    context.key = 2;
    try std.testing.expectError(error.OutOfMemory, loadHead(std.testing.failing_allocator, authority, loader, 0, 0, 0, 4096, .{}));
}
