// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Authenticated one-head-at-a-time access to the frozen E4 32K packed source.
//! The callback exposes a borrowed f32/packed head; this module owns the only
//! clone and never constructs a whole-checkpoint f64 snapshot.
const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const phase_loader = @import("attention_match_checkpoint_phase_loader.zig");
const solver = @import("attention_match.zig");
const tracker_mod = @import("host_allocation_tracker.zig");

const Sha256 = std.crypto.hash.sha2.Sha256;
pub const rows: usize = 32_512;
pub const dim: usize = 512;
pub const combined_cap_bytes: usize = 2 * 1024 * 1024 * 1024;
pub const packed_source_policy = "antfly.e4-32512-packed-per-head-source.v1|rows=32512|dim=512|one-live-head|f32-borrowed-clone|exact-page-admission";
pub const policy_sha256: [32]u8 = blk: {
    @setEvalBranchQuota(100_000);
    var out: [32]u8 = undefined;
    Sha256.hash(packed_source_policy, &out, .{});
    break :blk out;
};

fn spinLock(mutex: *std.atomic.Mutex) void {
    while (!mutex.tryLock()) std.atomic.spinLoopHint();
}

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.CompactionWorkspaceLimit;
}
fn nonzero(value: [32]u8) bool {
    return !std.mem.allEqual(u8, &value, 0);
}
fn scalar(hash: *Sha256, value: anytype) void {
    hash.update(std.mem.asBytes(&value));
}
fn slice(hash: *Sha256, value: anytype) void {
    hash.update(std.mem.sliceAsBytes(value));
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
    source_revision: u64,
    payload_sha256: [32]u8,
    compressed_span_count: usize,
    compressed_key_bytes: usize,
    compressed_value_bytes: usize,

    pub fn phaseEntry(self: @This()) phase_loader.Entry {
        return .{ .consumer_layer = self.consumer_layer, .query_head = self.query_head, .donor_index = self.donor_index, .donor_layer = self.donor_layer, .head_index = self.head_index, .rows = self.rows, .dim = self.dim, .scale = self.scale, .softcap = self.softcap };
    }

    pub fn cloneLiveBytes(self: @This()) !usize {
        var value = try mul(try mul(2, self.rows), @sizeOf(u64));
        value = try add(value, try mul(try add(try mul(2, try mul(self.rows, self.dim)), self.rows), @sizeOf(f32)));
        value = try add(value, try mul(self.compressed_span_count, @sizeOf(checkpoint.packed_storage.Span)));
        return add(value, try add(self.compressed_key_bytes, self.compressed_value_bytes));
    }
};

pub const AuthorityOwned = struct {
    allocator: std.mem.Allocator,
    identity: checkpoint.Identity,
    token_sha256: [32]u8,
    binary_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    shape_authority_sha256: [32]u8,
    state_sha256: [32]u8,
    census_sha256: [32]u8,
    owner_identity: u64,
    maximum_head_live_bytes: usize,
    maximum_loader_live_bytes: usize,
    entries: []Entry,

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.entries);
        self.* = undefined;
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

const Geometry = struct { rows: usize, dim: usize };

fn entryValid(entry: Entry, geometry: Geometry) bool {
    return entry.rows == geometry.rows and entry.dim == geometry.dim and entry.source_revision != 0 and
        nonzero(entry.payload_sha256) and std.math.isFinite(entry.scale) and std.math.isFinite(entry.softcap);
}

fn sameSourceAuthority(a: Entry, b: Entry) bool {
    return a.donor_layer == b.donor_layer and a.rows == b.rows and a.dim == b.dim and
        @as(u64, @bitCast(a.scale)) == @as(u64, @bitCast(b.scale)) and
        @as(u64, @bitCast(a.softcap)) == @as(u64, @bitCast(b.softcap)) and
        a.source_revision == b.source_revision and std.mem.eql(u8, &a.payload_sha256, &b.payload_sha256) and
        a.compressed_span_count == b.compressed_span_count and a.compressed_key_bytes == b.compressed_key_bytes and
        a.compressed_value_bytes == b.compressed_value_bytes;
}

fn censusDigest(authority: AuthorityOwned) [32]u8 {
    var hash = Sha256.init(.{});
    hash.update("antfly.e4-32512-packed-head-census.v1");
    hash.update(&policy_sha256);
    hash.update(&authority.identity.weights);
    hash.update(&authority.identity.tokenizer);
    hash.update(&authority.identity.template);
    hash.update(&authority.identity.layout);
    hash.update(&authority.token_sha256);
    hash.update(&authority.binary_sha256);
    hash.update(&authority.source_freeze_sha256);
    hash.update(&authority.shape_authority_sha256);
    hash.update(&authority.state_sha256);
    scalar(&hash, authority.owner_identity);
    scalar(&hash, authority.maximum_head_live_bytes);
    scalar(&hash, authority.maximum_loader_live_bytes);
    scalar(&hash, authority.entries.len);
    for (authority.entries) |entry| {
        scalar(&hash, entry.consumer_layer);
        scalar(&hash, entry.query_head);
        scalar(&hash, entry.donor_index);
        scalar(&hash, entry.donor_layer);
        scalar(&hash, entry.head_index);
        scalar(&hash, entry.rows);
        scalar(&hash, entry.dim);
        scalar(&hash, @as(u64, @bitCast(entry.scale)));
        scalar(&hash, @as(u64, @bitCast(entry.softcap)));
        scalar(&hash, entry.source_revision);
        hash.update(&entry.payload_sha256);
        scalar(&hash, entry.compressed_span_count);
        scalar(&hash, entry.compressed_key_bytes);
        scalar(&hash, entry.compressed_value_bytes);
    }
    return hash.finalResult();
}

fn initAuthority(
    allocator: std.mem.Allocator,
    identity: checkpoint.Identity,
    token_sha256: [32]u8,
    binary_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    shape_authority_sha256: [32]u8,
    state_sha256: [32]u8,
    owner_identity: u64,
    maximum_loader_live_bytes: usize,
    source_entries: []const Entry,
    geometry: Geometry,
) !AuthorityOwned {
    if (!nonzero(token_sha256) or !nonzero(binary_sha256) or !nonzero(source_freeze_sha256) or
        !nonzero(shape_authority_sha256) or !nonzero(state_sha256) or owner_identity == 0 or
        maximum_loader_live_bytes == 0 or source_entries.len == 0)
        return error.InvalidTrainingAugmentationBinding;
    const entries = try allocator.dupe(Entry, source_entries);
    errdefer allocator.free(entries);
    var maximum_head_live_bytes: usize = 0;
    for (entries, 0..) |entry, index| {
        if (!entryValid(entry, geometry)) return error.InvalidTrainingAugmentationBinding;
        for (entries[0..index]) |prior| {
            if (prior.consumer_layer == entry.consumer_layer and prior.query_head == entry.query_head)
                return error.InvalidTrainingAugmentationBinding;
            // GQA consumers legitimately share one KV source head. They must
            // agree on every source-owned field; only the consumer tuple may
            // differ.
            if (prior.donor_index == entry.donor_index and prior.head_index == entry.head_index and
                !sameSourceAuthority(prior, entry))
                return error.InvalidTrainingAugmentationBinding;
        }
        maximum_head_live_bytes = @max(maximum_head_live_bytes, try entry.cloneLiveBytes());
    }
    var result: AuthorityOwned = .{ .allocator = allocator, .identity = identity, .token_sha256 = token_sha256, .binary_sha256 = binary_sha256, .source_freeze_sha256 = source_freeze_sha256, .shape_authority_sha256 = shape_authority_sha256, .state_sha256 = state_sha256, .census_sha256 = undefined, .owner_identity = owner_identity, .maximum_head_live_bytes = maximum_head_live_bytes, .maximum_loader_live_bytes = maximum_loader_live_bytes, .entries = entries };
    result.census_sha256 = censusDigest(result);
    return result;
}

fn validateAuthority(authority: AuthorityOwned, geometry: Geometry) !void {
    if (!nonzero(authority.token_sha256) or !nonzero(authority.binary_sha256) or
        !nonzero(authority.source_freeze_sha256) or !nonzero(authority.shape_authority_sha256) or
        !nonzero(authority.state_sha256) or authority.owner_identity == 0 or
        authority.maximum_loader_live_bytes == 0 or authority.entries.len == 0)
        return error.InvalidTrainingAugmentationBinding;
    var maximum: usize = 0;
    for (authority.entries, 0..) |entry, index| {
        if (!entryValid(entry, geometry)) return error.InvalidTrainingAugmentationBinding;
        for (authority.entries[0..index]) |prior| {
            if (prior.consumer_layer == entry.consumer_layer and prior.query_head == entry.query_head)
                return error.InvalidTrainingAugmentationBinding;
            if (prior.donor_index == entry.donor_index and prior.head_index == entry.head_index and
                !sameSourceAuthority(prior, entry))
                return error.InvalidTrainingAugmentationBinding;
        }
        maximum = @max(maximum, try entry.cloneLiveBytes());
    }
    if (maximum != authority.maximum_head_live_bytes or
        !std.mem.eql(u8, &authority.census_sha256, &censusDigest(authority)))
        return error.InvalidTrainingAugmentationBinding;
}

pub fn freezeAuthority(
    allocator: std.mem.Allocator,
    identity: checkpoint.Identity,
    token_sha256: [32]u8,
    binary_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    shape_authority_sha256: [32]u8,
    state_sha256: [32]u8,
    owner_identity: u64,
    maximum_loader_live_bytes: usize,
    entries: []const Entry,
) !AuthorityOwned {
    return initAuthority(allocator, identity, token_sha256, binary_sha256, source_freeze_sha256, shape_authority_sha256, state_sha256, owner_identity, maximum_loader_live_bytes, entries, .{ .rows = rows, .dim = dim });
}

pub const BorrowedHead = struct {
    entry: Entry,
    identity: checkpoint.Identity,
    token_sha256: [32]u8,
    binary_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    shape_authority_sha256: [32]u8,
    state_sha256: [32]u8,
    census_sha256: [32]u8,
    head: checkpoint.Head,
    loader_live_bytes: usize,
};

pub const OpenFn = *const fn (?*anyopaque, Entry, solver.Control) anyerror!BorrowedHead;
pub const CloseFn = *const fn (?*anyopaque, *BorrowedHead) void;
pub const Loader = struct {
    context: ?*anyopaque,
    open_fn: OpenFn,
    close_fn: CloseFn,
    maximum_loader_live_bytes: usize,
    head_allocation_copies: u8 = 1,
};

fn headDigest(entry: Entry, head: checkpoint.Head) [32]u8 {
    var hash = Sha256.init(.{});
    hash.update("antfly.e4-32512-packed-head-payload.v1");
    scalar(&hash, entry.consumer_layer);
    scalar(&hash, entry.query_head);
    scalar(&hash, entry.donor_index);
    scalar(&hash, entry.donor_layer);
    scalar(&hash, entry.head_index);
    scalar(&hash, entry.source_revision);
    scalar(&hash, head.dim);
    scalar(&hash, head.compacted_until);
    slice(&hash, head.positions);
    slice(&hash, head.coverage_ends);
    slice(&hash, head.keys);
    slice(&hash, head.values);
    slice(&hash, head.bias);
    scalar(&hash, head.fp8.first_row);
    scalar(&hash, head.fp8.rows);
    scalar(&hash, head.fp8.key_scale);
    scalar(&hash, head.fp8.value_scale);
    scalar(&hash, head.compressed.version);
    slice(&hash, head.compressed.spans);
    hash.update(head.compressed.keys);
    hash.update(head.compressed.values);
    return hash.finalResult();
}

/// Producer seam for a frozen packed head. It derives, rather than accepts,
/// the bit-exact payload digest recorded by the census.
pub fn censusEntry(
    consumer_layer: u32,
    query_head: u32,
    donor_index: usize,
    donor_layer: u32,
    head_index: usize,
    scale: f64,
    softcap: f64,
    source_revision: u64,
    head: checkpoint.Head,
) !Entry {
    if (@as(usize, head.dim) != dim or head.positions.len != rows or head.coverage_ends.len != rows or
        head.keys.len != try mul(rows, dim) or head.values.len != try mul(rows, dim) or head.bias.len != rows or
        source_revision == 0 or !std.math.isFinite(scale) or !std.math.isFinite(softcap))
        return error.InvalidTrainingAugmentationBinding;
    var result: Entry = .{ .consumer_layer = consumer_layer, .query_head = query_head, .donor_index = donor_index, .donor_layer = donor_layer, .head_index = head_index, .rows = rows, .dim = dim, .scale = scale, .softcap = softcap, .source_revision = source_revision, .payload_sha256 = undefined, .compressed_span_count = head.compressed.spans.len, .compressed_key_bytes = head.compressed.keys.len, .compressed_value_bytes = head.compressed.values.len };
    result.payload_sha256 = headDigest(result, head);
    return result;
}

fn borrowedValid(authority: AuthorityOwned, expected: Entry, borrowed: BorrowedHead, geometry: Geometry) bool {
    const head = borrowed.head;
    const elements = mul(geometry.rows, geometry.dim) catch return false;
    return std.meta.eql(expected, borrowed.entry) and borrowed.identity.eql(authority.identity) and
        std.mem.eql(u8, &borrowed.token_sha256, &authority.token_sha256) and
        std.mem.eql(u8, &borrowed.binary_sha256, &authority.binary_sha256) and
        std.mem.eql(u8, &borrowed.source_freeze_sha256, &authority.source_freeze_sha256) and
        std.mem.eql(u8, &borrowed.shape_authority_sha256, &authority.shape_authority_sha256) and
        std.mem.eql(u8, &borrowed.state_sha256, &authority.state_sha256) and
        std.mem.eql(u8, &borrowed.census_sha256, &authority.census_sha256) and
        @as(usize, head.dim) == geometry.dim and head.positions.len == geometry.rows and
        head.coverage_ends.len == geometry.rows and head.keys.len == elements and head.values.len == elements and
        head.bias.len == geometry.rows and head.compressed.spans.len == expected.compressed_span_count and
        head.compressed.keys.len == expected.compressed_key_bytes and head.compressed.values.len == expected.compressed_value_bytes and
        std.mem.eql(u8, &headDigest(expected, head), &expected.payload_sha256);
}

fn allocationRequests(entry: Entry, copies: u8, storage: *[16]tracker_mod.HostAllocationTracker.PreparedAllocationRequest) ![]const tracker_mod.HostAllocationTracker.PreparedAllocationRequest {
    const row_u64 = try mul(entry.rows, @sizeOf(u64));
    const elements = try mul(entry.rows, entry.dim);
    const element_f32 = try mul(elements, @sizeOf(f32));
    const row_f32 = try mul(entry.rows, @sizeOf(f32));
    const spans = try mul(entry.compressed_span_count, @sizeOf(checkpoint.packed_storage.Span));
    var count: usize = 0;
    const Candidate = struct {
        fn append(out: *[16]tracker_mod.HostAllocationTracker.PreparedAllocationRequest, at: *usize, bytes: usize, alignment: std.mem.Alignment) void {
            if (bytes == 0) return;
            out[at.*] = .{ .logical_bytes = bytes, .alignment = alignment };
            at.* += 1;
        }
    };
    if (copies == 0 or copies > 2) return error.InvalidTrainingAugmentationBinding;
    for (0..copies) |_| {
        Candidate.append(storage, &count, row_u64, .of(u64));
        Candidate.append(storage, &count, row_u64, .of(u64));
        Candidate.append(storage, &count, element_f32, .of(f32));
        Candidate.append(storage, &count, element_f32, .of(f32));
        Candidate.append(storage, &count, row_f32, .of(f32));
        Candidate.append(storage, &count, spans, .of(checkpoint.packed_storage.Span));
        Candidate.append(storage, &count, entry.compressed_key_bytes, .of(u8));
        Candidate.append(storage, &count, entry.compressed_value_bytes, .of(u8));
    }
    return storage[0..count];
}

fn cloneHead(allocator: std.mem.Allocator, entry: Entry, source: checkpoint.Head, control: solver.Control) !phase_loader.HeadOwned {
    const positions = try allocator.dupe(u64, source.positions);
    errdefer allocator.free(positions);
    try control.check();
    const coverage = try allocator.dupe(u64, source.coverage_ends);
    errdefer allocator.free(coverage);
    const keys = try allocator.dupe(f32, source.keys);
    errdefer allocator.free(keys);
    try control.check();
    const values = try allocator.dupe(f32, source.values);
    errdefer allocator.free(values);
    const bias = try allocator.dupe(f32, source.bias);
    errdefer allocator.free(bias);
    const spans = try allocator.dupe(checkpoint.packed_storage.Span, source.compressed.spans);
    errdefer allocator.free(spans);
    const packed_keys = try allocator.dupe(u8, source.compressed.keys);
    errdefer allocator.free(packed_keys);
    try control.check();
    const packed_values = try allocator.dupe(u8, source.compressed.values);
    return .{ .allocator = allocator, .entry = entry.phaseEntry(), .positions = positions, .coverage_ends = coverage, .keys = keys, .values = values, .bias = bias, .compressed_spans = spans, .compressed_keys = packed_keys, .compressed_values = packed_values, .compressed_version = source.compressed.version, .fp8 = source.fp8, .compacted_until = source.compacted_until };
}

pub const Source = struct {
    authority: *const AuthorityOwned,
    loader: Loader,
    tracker: *tracker_mod.HostAllocationTracker,
    mutex: std.atomic.Mutex = .unlocked,
    head_live: bool = false,

    pub const BindingIdentity = struct {
        source_rows: usize,
        source_dim: usize,
        shape_authority_sha256: [32]u8,
        packed_source_policy_sha256: [32]u8,
        census_sha256: [32]u8,
        source_freeze_sha256: [32]u8,
        owner_identity: u64,
    };

    pub fn bindingIdentity(self: *const Source) BindingIdentity {
        return .{ .source_rows = rows, .source_dim = dim, .shape_authority_sha256 = self.authority.shape_authority_sha256, .packed_source_policy_sha256 = policy_sha256, .census_sha256 = self.authority.census_sha256, .source_freeze_sha256 = self.authority.source_freeze_sha256, .owner_identity = self.authority.owner_identity };
    }

    pub fn init(authority: *const AuthorityOwned, loader: Loader, tracker: *tracker_mod.HostAllocationTracker) !Source {
        return initWithGeometry(authority, loader, tracker, .{ .rows = rows, .dim = dim });
    }

    fn initWithGeometry(authority: *const AuthorityOwned, loader: Loader, tracker: *tracker_mod.HostAllocationTracker, geometry: Geometry) !Source {
        try validateAuthority(authority.*, geometry);
        if (loader.maximum_loader_live_bytes == 0 or loader.maximum_loader_live_bytes > authority.maximum_loader_live_bytes or
            authority.owner_identity != tracker.owner_identity or loader.head_allocation_copies == 0 or
            loader.head_allocation_copies > 2)
            return error.InvalidTrainingAugmentationBinding;
        try tracker.verifyBoundedPrepare(authority.owner_identity);
        return .{ .authority = authority, .loader = loader, .tracker = tracker };
    }

    fn release(self: *Source) void {
        spinLock(&self.mutex);
        if (!self.head_live) @panic("packed head source release without live head");
        self.head_live = false;
        self.mutex.unlock();
    }

    /// Phase owners call this before returning or releasing the tracker. A
    /// packed arm must retain LoadedHead itself; moving out only `.owned`
    /// would strand this lease and is not a supported ownership transfer.
    pub fn verifyNoLiveHead(self: *Source) !void {
        spinLock(&self.mutex);
        defer self.mutex.unlock();
        if (self.head_live) return error.IncompleteContextAllocationScope;
    }

    pub fn loadHead(self: *Source, consumer_layer: u32, query_head: u32, external_live_bytes: usize, control: solver.Control) !LoadedHead {
        return self.loadHeadWithGeometry(consumer_layer, query_head, external_live_bytes, control, .{ .rows = rows, .dim = dim });
    }

    fn loadHeadWithGeometry(self: *Source, consumer_layer: u32, query_head: u32, external_live_bytes: usize, control: solver.Control, geometry: Geometry) !LoadedHead {
        try control.check();
        spinLock(&self.mutex);
        if (self.head_live) {
            self.mutex.unlock();
            return error.PackedHeadAlreadyLive;
        }
        self.head_live = true;
        self.mutex.unlock();
        errdefer self.release();
        const entry = try self.authority.entry(consumer_layer, query_head);
        const bounded_live = try add(self.loader.maximum_loader_live_bytes, try entry.cloneLiveBytes());
        if (external_live_bytes > combined_cap_bytes or bounded_live > combined_cap_bytes - external_live_bytes)
            return error.CompactionWorkspaceLimit;
        var requests: [16]tracker_mod.HostAllocationTracker.PreparedAllocationRequest = undefined;
        const admission = try self.tracker.boundedPrepareSequenceAdmission(try allocationRequests(entry, self.loader.head_allocation_copies, &requests));
        if (admission.owner_identity != self.authority.owner_identity or !admission.fits or
            admission.required_physical_bytes > combined_cap_bytes - external_live_bytes)
            return error.CompactionWorkspaceLimit;
        var borrowed = try self.loader.open_fn(self.loader.context, entry, control);
        defer self.loader.close_fn(self.loader.context, &borrowed);
        if (borrowed.loader_live_bytes > self.loader.maximum_loader_live_bytes or !borrowedValid(self.authority.*, entry, borrowed, geometry))
            return error.InvalidTrainingAugmentationBinding;
        const allocator = self.tracker.allocator();
        var owned = try cloneHead(allocator, entry, borrowed.head, control);
        errdefer owned.deinit();
        try control.check();
        return .{ .source = self, .owned = owned };
    }
};

pub const LoadedHead = struct {
    source: *Source,
    owned: phase_loader.HeadOwned,

    pub fn head(self: *const @This()) checkpoint.Head {
        return self.owned.head();
    }

    pub fn deinit(self: *@This()) void {
        self.owned.deinit();
        self.source.release();
        self.* = undefined;
    }
};

test "packed head authority rejects stale count layout and inconsistent shared source heads" {
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const good: Entry = .{ .consumer_layer = 1, .query_head = 2, .donor_index = 0, .donor_layer = 3, .head_index = 0, .rows = rows, .dim = dim, .scale = 1, .softcap = 0, .source_revision = 9, .payload_sha256 = @splat(7), .compressed_span_count = 0, .compressed_key_bytes = 0, .compressed_value_bytes = 0 };
    var stale = good;
    stale.rows -= 1;
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, freezeAuthority(std.testing.allocator, identity, @splat(5), @splat(6), @splat(7), @splat(8), @splat(9), 10, 1, &.{stale}));
    var layout = good;
    layout.dim -= 1;
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, freezeAuthority(std.testing.allocator, identity, @splat(5), @splat(6), @splat(7), @splat(8), @splat(9), 10, 1, &.{layout}));
    var duplicate = good;
    duplicate.consumer_layer = 8;
    duplicate.query_head = 9;
    var shared = try freezeAuthority(std.testing.allocator, identity, @splat(5), @splat(6), @splat(7), @splat(8), @splat(9), 10, 1, &.{ good, duplicate });
    shared.deinit();
    duplicate.source_revision += 1;
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, freezeAuthority(std.testing.allocator, identity, @splat(5), @splat(6), @splat(7), @splat(8), @splat(9), 10, 1, &.{ good, duplicate }));
}

test "packed head exact cap rejects before source callback" {
    const Never = struct {
        var calls: usize = 0;
        fn open(_: ?*anyopaque, _: Entry, _: solver.Control) !BorrowedHead {
            calls += 1;
            return error.TestUnexpectedResult;
        }
        fn close(_: ?*anyopaque, _: *BorrowedHead) void {}
    };
    const geometry: Geometry = .{ .rows = 2, .dim = 1 };
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const entry: Entry = .{ .consumer_layer = 1, .query_head = 2, .donor_index = 0, .donor_layer = 3, .head_index = 0, .rows = 2, .dim = 1, .scale = 1, .softcap = 0, .source_revision = 9, .payload_sha256 = @splat(7), .compressed_span_count = 0, .compressed_key_bytes = 0, .compressed_value_bytes = 0 };
    var tracker = tracker_mod.HostAllocationTracker.initSwitchablePrepare(0);
    const owner_identity = try tracker.activateBoundedPrepare(tracker_mod.bounded_prepare_e4_physical_bytes);
    var authority = try initAuthority(std.testing.allocator, identity, @splat(5), @splat(6), @splat(7), @splat(8), @splat(9), owner_identity, 16, &.{entry}, geometry);
    defer authority.deinit();
    var source = try Source.initWithGeometry(&authority, .{ .context = null, .open_fn = Never.open, .close_fn = Never.close, .maximum_loader_live_bytes = 16 }, &tracker, geometry);
    try std.testing.expectError(error.CompactionWorkspaceLimit, source.loadHeadWithGeometry(1, 2, combined_cap_bytes, .{}, geometry));
    try std.testing.expectEqual(@as(usize, 0), Never.calls);
    try std.testing.expect(!source.head_live);
}

test "packed head dual exact physical admission rejects before provider callback" {
    const Never = struct {
        var calls: usize = 0;
        fn open(_: ?*anyopaque, _: Entry, _: solver.Control) !BorrowedHead {
            calls += 1;
            return error.TestUnexpectedResult;
        }
        fn close(_: ?*anyopaque, _: *BorrowedHead) void {}
    };
    const geometry: Geometry = .{ .rows = 2, .dim = 1 };
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const entry: Entry = .{ .consumer_layer = 1, .query_head = 2, .donor_index = 0, .donor_layer = 3, .head_index = 0, .rows = 2, .dim = 1, .scale = 1, .softcap = 0, .source_revision = 9, .payload_sha256 = @splat(7), .compressed_span_count = 0, .compressed_key_bytes = 0, .compressed_value_bytes = 0 };
    var tracker = tracker_mod.HostAllocationTracker.initSwitchablePrepare(0);
    const owner_identity = try tracker.activateBoundedPrepare(tracker_mod.bounded_prepare_e4_physical_bytes);
    var authority = try initAuthority(std.testing.allocator, identity, @splat(5), @splat(6), @splat(7), @splat(8), @splat(9), owner_identity, 16, &.{entry}, geometry);
    defer authority.deinit();
    var source = try Source.initWithGeometry(&authority, .{ .context = null, .open_fn = Never.open, .close_fn = Never.close, .maximum_loader_live_bytes = 16, .head_allocation_copies = 2 }, &tracker, geometry);
    const logical_dual = try mul(2, try entry.cloneLiveBytes());
    const external = combined_cap_bytes - logical_dual;
    try std.testing.expectError(error.CompactionWorkspaceLimit, source.loadHeadWithGeometry(1, 2, external, .{}, geometry));
    try std.testing.expectEqual(@as(usize, 0), Never.calls);
    try std.testing.expect(!source.head_live);
    try std.testing.expectEqual(@as(u64, 0), tracker.snapshot().tracked_combined_live_bytes);
}

test "packed head authenticates wrapped source and releases one-head lease across cancel fault and stale reload" {
    const geometry: Geometry = .{ .rows = 2, .dim = 1 };
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const positions = [_]u64{ std.math.maxInt(u64), 0 };
    const coverage = [_]u64{ 0, 1 };
    const keys = [_]f32{ 1.25, -0.0 };
    const values = [_]f32{ -2.5, 3.75 };
    const bias = [_]f32{ 0.125, -0.25 };
    const head: checkpoint.Head = .{ .dim = 1, .compacted_until = std.math.maxInt(u64), .positions = &positions, .coverage_ends = &coverage, .keys = &keys, .values = &values, .bias = &bias };
    var entry: Entry = .{ .consumer_layer = 1, .query_head = 2, .donor_index = 0, .donor_layer = 3, .head_index = 0, .rows = 2, .dim = 1, .scale = 1, .softcap = 0, .source_revision = 9, .payload_sha256 = undefined, .compressed_span_count = 0, .compressed_key_bytes = 0, .compressed_value_bytes = 0 };
    entry.payload_sha256 = headDigest(entry, head);
    var tracker = tracker_mod.HostAllocationTracker.initSwitchablePrepare(0);
    const owner_identity = try tracker.activateBoundedPrepare(tracker_mod.bounded_prepare_e4_physical_bytes);
    var authority = try initAuthority(std.testing.allocator, identity, @splat(5), @splat(6), @splat(7), @splat(8), @splat(9), owner_identity, 16, &.{entry}, geometry);
    defer authority.deinit();
    const Context = struct {
        authority: *const AuthorityOwned,
        head: checkpoint.Head,
        opens: usize = 0,
        closes: usize = 0,
        stale: bool = false,
        fn open(raw: ?*anyopaque, expected: Entry, control: solver.Control) !BorrowedHead {
            try control.check();
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.opens += 1;
            var result: BorrowedHead = .{ .entry = expected, .identity = self.authority.identity, .token_sha256 = self.authority.token_sha256, .binary_sha256 = self.authority.binary_sha256, .source_freeze_sha256 = self.authority.source_freeze_sha256, .shape_authority_sha256 = self.authority.shape_authority_sha256, .state_sha256 = self.authority.state_sha256, .census_sha256 = self.authority.census_sha256, .head = self.head, .loader_live_bytes = 16 };
            if (self.stale) result.entry.source_revision += 1;
            return result;
        }
        fn close(raw: ?*anyopaque, _: *BorrowedHead) void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.closes += 1;
        }
    };
    var context: Context = .{ .authority = &authority, .head = head };
    var source = try Source.initWithGeometry(&authority, .{ .context = &context, .open_fn = Context.open, .close_fn = Context.close, .maximum_loader_live_bytes = 16 }, &tracker, geometry);
    var loaded = try source.loadHeadWithGeometry(1, 2, 0, .{}, geometry);
    try std.testing.expect(source.head_live);
    try std.testing.expectEqualSlices(u64, &positions, loaded.owned.positions);
    try std.testing.expectEqualSlices(f32, &keys, loaded.owned.keys);
    try std.testing.expect(tracker.snapshot().tracked_combined_live_bytes > 0);
    try std.testing.expectError(error.PackedHeadAlreadyLive, source.loadHeadWithGeometry(1, 2, 0, .{}, geometry));
    try std.testing.expectError(error.IncompleteContextAllocationScope, source.verifyNoLiveHead());
    loaded.deinit();
    try std.testing.expect(!source.head_live);
    try source.verifyNoLiveHead();
    try std.testing.expectEqual(@as(u64, 0), tracker.snapshot().tracked_combined_live_bytes);
    context.stale = true;
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, source.loadHeadWithGeometry(1, 2, 0, .{}, geometry));
    try std.testing.expect(!source.head_live);
    context.stale = false;
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.CompactionCancelled;
        }
    };
    const opens_before_cancel = context.opens;
    try std.testing.expectError(error.CompactionCancelled, source.loadHeadWithGeometry(1, 2, 0, .{ .check_fn = Cancel.check }, geometry));
    try std.testing.expectEqual(opens_before_cancel, context.opens);
    try std.testing.expect(!source.head_live);
    try tracker.armNextPositiveAllocationFailure();
    try std.testing.expectError(error.CompactionWorkspaceLimit, source.loadHeadWithGeometry(1, 2, 0, .{}, geometry));
    try std.testing.expect(!source.head_live);
    tracker.logical_limit_bytes = null;
    var recovered = try source.loadHeadWithGeometry(1, 2, 0, .{}, geometry);
    recovered.deinit();
    try std.testing.expectEqual(context.opens, context.closes);
    try std.testing.expectEqual(@as(u64, 0), tracker.snapshot().tracked_combined_live_bytes);
}
