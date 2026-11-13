// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Explicit test-driver contracts for successive installed-owner compaction.
//! No production request or environment variable constructs this request.
const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const Control = @import("attention_match.zig").Control;
const allocations = @import("host_allocation_tracker.zig");
const fit_policy = @import("attention_match_fit_policy.zig");

pub const max_cycles = 20;
pub const max_aggregate_bytes = 8 * 1024 * 1024 * 1024;
pub const query_tail_tokens = 512;

pub const Request = struct {
    encoding: checkpoint.Encoding = .f16,
    fit_policy: fit_policy.Policy = .legacy_fixed,
    cycles: usize = max_cycles,
    append_tokens: usize = 512,
    answer_tokens: usize = 16,
    initial_minimum_savings_ratio: f64 = 1.5,
    incremental_minimum_savings_ratio: f64 = 1.000001,
    /// Frozen gold tokens are consumed only after candidate publication. An
    /// empty reference explicitly selects diagnostic exact-greedy targets.
    reference: []const i64 = &.{},

    pub fn capacity(self: Request, initial: usize) !usize {
        try self.fit_policy.validate(self.encoding, true);
        if (self.encoding == .f32 or self.cycles == 0 or self.cycles > max_cycles or
            self.append_tokens == 0 or self.append_tokens > 2048 or self.append_tokens > initial or
            self.answer_tokens == 0 or self.answer_tokens > 32 or
            !std.math.isFinite(self.initial_minimum_savings_ratio) or self.initial_minimum_savings_ratio < 1.5 or self.initial_minimum_savings_ratio > 4 or
            !std.math.isFinite(self.incremental_minimum_savings_ratio) or self.incremental_minimum_savings_ratio <= 1 or self.incremental_minimum_savings_ratio > 4)
            return error.InvalidCompactionProbe;
        const answers = try std.math.mul(usize, self.cycles, self.answer_tokens);
        if (self.reference.len != 0 and self.reference.len != answers) return error.InvalidCompactionProbe;
        const per_cycle = try std.math.add(usize, self.append_tokens, self.answer_tokens);
        const capacity_value = try std.math.add(usize, initial, try std.math.mul(usize, self.cycles, per_cycle));
        // Every fitted owner must still support the full representative query
        // repeat in a private clone without extending the logical window.
        if (initial < 2 or capacity_value > checkpoint.max_context_tokens - query_tail_tokens)
            return error.CompactionQueryLimit;
        return capacity_value;
    }
};

/// This admission covers both checkpoint owners and all explicitly planned
/// transient allocations. The outer campaign separately guards process RSS,
/// swap and system headroom, including loaded model/runtime storage.
pub const PhasePlan = struct {
    owners: usize,
    host: usize = 0,
    queries: usize = 0,
    wire: usize = 0,
    import: usize = 0,
    device: usize = 0,

    pub fn admit(self: PhasePlan) !usize {
        var total: usize = 0;
        for ([_]usize{ self.owners, self.host, self.queries, self.wire, self.import, self.device }) |bytes|
            total = try std.math.add(usize, total, bytes);
        if (total > max_aggregate_bytes) return error.CompactionWorkspaceLimit;
        return total;
    }
};

pub const Row = struct {
    cycle: usize = 0,
    logical_tokens: usize = 0,
    owner_id: u64 = 0,
    source_revision: u64 = 0,
    installed_revision: u64 = 0,
    completed_chunks: usize = 0,
    validated_heads: usize = 0,
    max_head_mass_error: f64 = 0,
    max_head_output_rmse: f64 = 0,
    ragged_repeat_calls: usize = 0,
    fp64_accelerated_value_solves: u64 = 0,
    query_owner_id: u64 = 0,
    query_owner_peak_bytes: u64 = 0,
    query_scope_released: bool = false,
    query_scope_delta_bytes: ?u64 = null,
    query_uses_source_owner: bool = false,
    immutable_rows_sha256: [32]u8 = @splat(0),
    checkpoint_sha256: [32]u8 = @splat(0),
    source_global_bytes: usize = 0,
    candidate_global_bytes: usize = 0,
    admitted_peak_bytes: usize = 0,
    fp8_heads: usize = 0,
    rotated4_heads: usize = 0,
    attention_calls: usize = 0,
    donor_appends: usize = 0,
    greedy_matches: usize = 0,
    comparisons: usize = 0,
    exact_nll: f64 = 0,
    compacted_nll: f64 = 0,
    max_logit_relative_l2: f64 = 0,
};

pub const Report = struct {
    diagnostic_only: bool = true,
    production_ready: bool = false,
    passed: bool = false,
    fit_policy_sha256: [32]u8 = @splat(0),
    producer_policy: fit_policy.Policy = .legacy_fixed,
    completed: usize = 0,
    initial_tokens: usize,
    final_tokens: usize = 0,
    frozen_reference: bool,
    device_memory_scope: []const u8 = "MTLDevice.currentAllocatedSize sampled at every observed owned/alias MTLBuffer allocation and explicit cycle phase boundaries; excludes unobserved MPS/internal-driver transient allocations",
    device_memory_cap_bytes: u64 = 8 * 1024 * 1024 * 1024,
    device_current_bytes: u64 = 0,
    device_observed_peak_bytes: u64 = 0,
    device_cap_exceeded: bool = false,
    exact_release: allocations.RequestAllocationOwner.ReleaseReceipt = .{},
    compacted_release: allocations.RequestAllocationOwner.ReleaseReceipt = .{},
    rows: [max_cycles]Row = @splat(.{}),
};

pub const Descriptor = struct {
    donor: usize,
    head: usize,
    layer: u32,
    dim: u32,
    window: u32,
    rows: usize,
    packed_version: u32,
    spans: usize,
    keys: usize,
    values: usize,
    fp8: checkpoint.codec.Fp8,
};

pub fn capture(state: checkpoint.State, protected_prefix_rows: usize, output: []Descriptor, control: Control) ![]const Descriptor {
    var count: usize = 0;
    for (state.donors, 0..) |donor, di| {
        if (donor.window != 0) continue;
        for (donor.heads, 0..) |head, hi| {
            try control.check();
            if (count == output.len) return error.CompactionQueryLimit;
            const protected = @min(protected_prefix_rows, state.tokens.len);
            if (head.positions.len < protected) return error.CompactionPrefixUnavailable;
            for (0..protected) |i| {
                if (head.positions[i] != i or head.coverage_ends[i] != i + 1 or head.bias[i] != 0)
                    return error.CompactionPrefixUnavailable;
            }
            var rows: usize = protected;
            while (rows < head.coverage_ends.len and head.coverage_ends[rows] <= head.compacted_until) : (rows += 1) {}
            output[count] = .{ .donor = di, .head = hi, .layer = donor.layer, .dim = head.dim, .window = donor.window, .rows = rows, .packed_version = head.compressed.version, .spans = head.compressed.spans.len, .keys = head.compressed.keys.len, .values = head.compressed.values.len, .fp8 = head.fp8 };
            count += 1;
        }
    }
    return output[0..count];
}

fn integer(hash: *std.crypto.hash.sha2.Sha256, value: u64) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, value, .little);
    hash.update(&bytes);
}

pub fn immutableDigest(state: checkpoint.State, descriptors: []const Descriptor, control: Control) ![32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.installed-history.v2\n");
    inline for (.{ "weights", "tokenizer", "template", "layout" }) |field|
        hash.update(&@field(state.identity, field));
    integer(&hash, state.logical_length);
    integer(&hash, state.layer_donors.len);
    hash.update(std.mem.sliceAsBytes(state.layer_donors));
    for (descriptors) |old| {
        try control.check();
        if (old.donor >= state.donors.len or old.head >= state.donors[old.donor].heads.len)
            return error.CompactionHistoryChanged;
        const donor = state.donors[old.donor];
        const head = donor.heads[old.head];
        if (donor.layer != old.layer or head.dim != old.dim or donor.window != old.window or
            head.positions.len < old.rows or head.compressed.spans.len < old.spans or
            head.compressed.keys.len < old.keys or head.compressed.values.len < old.values or
            (old.spans != 0 and head.compressed.version != old.packed_version)) return error.CompactionHistoryChanged;
        integer(&hash, old.donor);
        integer(&hash, old.head);
        integer(&hash, donor.layer);
        integer(&hash, head.dim);
        integer(&hash, donor.window);
        integer(&hash, old.rows);
        // These bytes are compared on the same qualified destination host;
        // checkpoint transport itself remains canonically little endian.
        hash.update(std.mem.sliceAsBytes(head.positions[0..old.rows]));
        hash.update(std.mem.sliceAsBytes(head.coverage_ends[0..old.rows]));
        hash.update(std.mem.sliceAsBytes(head.bias[0..old.rows]));
        hash.update(std.mem.sliceAsBytes(head.keys[0 .. old.rows * head.dim]));
        hash.update(std.mem.sliceAsBytes(head.values[0 .. old.rows * head.dim]));
        // Adjacent packed appends canonically extend the last span. Seal its
        // mapping only within the previously fitted rows, while retaining all
        // prior payload bytes below. Growth into a prior exact row still
        // changes this mapping and therefore the digest.
        for (head.compressed.spans[0..old.spans]) |span| {
            if (span.first >= old.rows) return error.CompactionHistoryChanged;
            integer(&hash, span.first);
            integer(&hash, @min(@as(usize, span.rows), old.rows - span.first));
            integer(&hash, span.offset);
        }
        hash.update(head.compressed.keys[0..old.keys]);
        hash.update(head.compressed.values[0..old.values]);
        if (old.fp8.rows != 0) {
            if (head.fp8.first_row != old.fp8.first_row or head.fp8.rows < old.fp8.rows or
                head.fp8.key_scale != old.fp8.key_scale or head.fp8.value_scale != old.fp8.value_scale)
                return error.CompactionHistoryChanged;
            integer(&hash, old.fp8.first_row);
            integer(&hash, old.fp8.rows);
            integer(&hash, @as(u32, @bitCast(old.fp8.key_scale)));
            integer(&hash, @as(u32, @bitCast(old.fp8.value_scale)));
            const start = @as(usize, old.fp8.first_row) * head.dim;
            const end = start + @as(usize, old.fp8.rows) * head.dim;
            for (head.keys[start..end], head.values[start..end]) |key, value| {
                const bytes = [_]u8{ try checkpoint.codec.encode(key, old.fp8.key_scale), try checkpoint.codec.encode(value, old.fp8.value_scale) };
                hash.update(&bytes);
            }
        }
    }
    return hash.finalResult();
}

/// Compute the installed-history seal directly from an immutable streaming
/// provider.  Only one head lease is live at a time; no complete checkpoint or
/// complete donor is materialized on the host.  The byte sequence is exactly
/// the host `immutableDigest` sequence, so cycle gates remain unchanged.
pub fn immutableDigestFromProvider(metadata: checkpoint.StreamingMetadata, provider: checkpoint.StreamingHeadProvider, descriptors: []const Descriptor, control: Control) ![32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.installed-history.v2\n");
    inline for (.{ "weights", "tokenizer", "template", "layout" }) |field|
        hash.update(&@field(metadata.identity, field));
    integer(&hash, metadata.logical_length);
    integer(&hash, metadata.layer_donors.len);
    hash.update(std.mem.sliceAsBytes(metadata.layer_donors));
    for (descriptors) |old| {
        try control.check();
        if (old.donor >= metadata.donors.len) return error.CompactionHistoryChanged;
        const donor = metadata.donors[old.donor];
        if (old.head >= donor.head_count) return error.CompactionHistoryChanged;
        var lease = try provider.open(old.donor, old.head, control);
        defer lease.deinit();
        const head = lease.head;
        if (donor.layer != old.layer or head.dim != old.dim or donor.window != old.window or
            head.positions.len < old.rows or head.compressed.spans.len < old.spans or
            head.compressed.keys.len < old.keys or head.compressed.values.len < old.values or
            (old.spans != 0 and head.compressed.version != old.packed_version)) return error.CompactionHistoryChanged;
        integer(&hash, old.donor);
        integer(&hash, old.head);
        integer(&hash, donor.layer);
        integer(&hash, head.dim);
        integer(&hash, donor.window);
        integer(&hash, old.rows);
        hash.update(std.mem.sliceAsBytes(head.positions[0..old.rows]));
        hash.update(std.mem.sliceAsBytes(head.coverage_ends[0..old.rows]));
        hash.update(std.mem.sliceAsBytes(head.bias[0..old.rows]));
        hash.update(std.mem.sliceAsBytes(head.keys[0 .. old.rows * head.dim]));
        hash.update(std.mem.sliceAsBytes(head.values[0 .. old.rows * head.dim]));
        for (head.compressed.spans[0..old.spans]) |span| {
            if (span.first >= old.rows) return error.CompactionHistoryChanged;
            integer(&hash, span.first);
            integer(&hash, @min(@as(usize, span.rows), old.rows - span.first));
            integer(&hash, span.offset);
        }
        hash.update(head.compressed.keys[0..old.keys]);
        hash.update(head.compressed.values[0..old.values]);
        if (old.fp8.rows != 0) {
            if (head.fp8.first_row != old.fp8.first_row or head.fp8.rows < old.fp8.rows or
                head.fp8.key_scale != old.fp8.key_scale or head.fp8.value_scale != old.fp8.value_scale)
                return error.CompactionHistoryChanged;
            integer(&hash, old.fp8.first_row);
            integer(&hash, old.fp8.rows);
            integer(&hash, @as(u32, @bitCast(old.fp8.key_scale)));
            integer(&hash, @as(u32, @bitCast(old.fp8.value_scale)));
            const start = @as(usize, old.fp8.first_row) * head.dim;
            const end = start + @as(usize, old.fp8.rows) * head.dim;
            for (head.keys[start..end], head.values[start..end]) |key, value| hash.update(&[_]u8{ try checkpoint.codec.encode(key, old.fp8.key_scale), try checkpoint.codec.encode(value, old.fp8.value_scale) });
        }
    }
    try provider.verify_empty_fn(provider.context);
    return hash.finalResult();
}

pub const StreamingTransport = struct {
    encoded_bytes: usize,
    source_bytes: usize,
    savings_ratio: f64,
};

/// Truthful device transport accounting. `source_bytes` is the already
/// measured source-global authority; it is never inferred from row counts.
pub fn streamingTransport(metadata: checkpoint.StreamingMetadata, provider: checkpoint.StreamingHeadProvider, limits: checkpoint.Limits, encoding: checkpoint.Encoding, source_bytes: usize, control: Control) !StreamingTransport {
    if (source_bytes == 0) return error.InvalidCompactionProbe;
    const encoded = try checkpoint.encodedSizeFromProvider(metadata, provider, limits, encoding, control);
    if (encoded == 0) return error.InvalidKvCheckpoint;
    return .{ .encoded_bytes = encoded, .source_bytes = source_bytes, .savings_ratio = @as(f64, @floatFromInt(source_bytes)) / @as(f64, @floatFromInt(encoded)) };
}

test "context cycle streaming history seal matches host and closes leases on failure and cancellation" {
    const Fake = struct {
        state: checkpoint.State,
        live: bool = false,
        opens: usize = 0,
        fail_open: bool = false,
        fn open(raw: ?*anyopaque, donor: usize, head_index: usize, control: Control) !checkpoint.StreamingHeadLease {
            try control.check();
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (self.fail_open) return error.TestProviderFailed;
            if (self.live) return error.InvalidKvCheckpoint;
            self.live = true;
            self.opens += 1;
            return .{ .context = self, .close_fn = close, .donor = donor, .head_index = head_index, .head = self.state.donors[donor].heads[head_index] };
        }
        fn close(raw: ?*anyopaque, _: *checkpoint.StreamingHeadLease) void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            std.debug.assert(self.live);
            self.live = false;
        }
        fn empty(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (self.live) return error.InvalidKvCheckpoint;
        }
        fn provider(self: *@This()) checkpoint.StreamingHeadProvider {
            return .{ .context = self, .open_fn = open, .verify_empty_fn = empty };
        }
    };
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Canceled;
        }
    };
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = 4,
        .tokens = &.{ 0, 0, 0, 0 },
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &.{.{ .heads = &.{.{ .dim = 1, .compacted_until = 2, .positions = &.{ 0, 1, 2, 3 }, .coverage_ends = &.{ 1, 2, 3, 4 }, .keys = &.{ 0, 1, 2, 3 }, .values = &.{ 4, 5, 6, 7 }, .bias = &.{ 0, 0, 0, 0 } }} }},
    };
    const metadata: checkpoint.StreamingMetadata = .{ .identity = state.identity, .logical_length = state.logical_length, .tokens = state.tokens, .logits = state.logits, .layer_donors = state.layer_donors, .donors = &.{.{ .layer = 0, .window = 0, .head_count = 1 }} };
    var descriptors: [1]Descriptor = undefined;
    const captured = try capture(state, 1, &descriptors, .{});
    var fake: Fake = .{ .state = state };
    try std.testing.expectEqual(try immutableDigest(state, captured, .{}), try immutableDigestFromProvider(metadata, fake.provider(), captured, .{}));
    try std.testing.expect(!fake.live and fake.opens == 1);
    fake.fail_open = true;
    try std.testing.expectError(error.TestProviderFailed, immutableDigestFromProvider(metadata, fake.provider(), captured, .{}));
    try std.testing.expect(!fake.live);
    fake.fail_open = false;
    try std.testing.expectError(error.Canceled, immutableDigestFromProvider(metadata, fake.provider(), captured, .{ .check_fn = Cancel.check }));
    try std.testing.expect(!fake.live);
}

test "attention_match Metal cycle admission reserves continuation within logical window" {
    try std.testing.expectEqual(@as(usize, 18752), try (Request{}).capacity(8192));
    try std.testing.expectEqual(@as(usize, 32256), try (Request{ .append_tokens = 4, .answer_tokens = 4 }).capacity(32096));
    try std.testing.expectError(error.CompactionQueryLimit, (Request{}).capacity(32512));
    try std.testing.expectError(error.InvalidCompactionProbe, (Request{ .cycles = 21 }).capacity(8192));
    try std.testing.expectError(error.InvalidCompactionProbe, (Request{ .reference = &.{1} }).capacity(8192));
    try std.testing.expectError(error.CompactionWorkspaceLimit, (PhasePlan{ .owners = max_aggregate_bytes, .host = 1 }).admit());
    try std.testing.expectError(error.Overflow, (PhasePlan{ .owners = std.math.maxInt(usize), .host = 1 }).admit());
}

test "attention_match Metal cycle history seal detects changed prior vectors and FP8 scales" {
    var values = [_]f32{ 1, 2, 3, 4 };
    var heads = [_]checkpoint.Head{.{ .dim = 1, .compacted_until = 2, .positions = &.{ 0, 1, 2, 3 }, .coverage_ends = &.{ 1, 2, 3, 4 }, .keys = &.{ 0, 1, 2, 3 }, .values = &values, .bias = &.{ 0, 0, 0, 0 }, .fp8 = .{ .rows = 2 } }};
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = 4,
        .tokens = &.{ 0, 0, 0, 0 },
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &.{.{ .heads = &heads }},
    };
    var descriptors: [1]Descriptor = undefined;
    const old = try capture(state, 1, &descriptors, .{});
    const expected = try immutableDigest(state, old, .{});
    values[3] = 5; // Newly appended/exact rows are outside the immutable seal.
    try std.testing.expectEqual(expected, try immutableDigest(state, old, .{}));
    values[0] = 2;
    try std.testing.expect(!std.mem.eql(u8, &expected, &try immutableDigest(state, old, .{})));
    values[0] = 1;
    heads[0].fp8.key_scale = 2;
    try std.testing.expectError(error.CompactionHistoryChanged, immutableDigest(state, old, .{}));
    heads[0].fp8.key_scale = 1;
    heads[0].fp8.rows = 3;
    try std.testing.expectEqual(expected, try immutableDigest(state, old, .{}));
    heads[0].fp8.rows = 1;
    try std.testing.expectError(error.CompactionHistoryChanged, immutableDigest(state, old, .{}));
}

test "attention_match Metal cycle seal protects initial prefix and donor geometry" {
    var values = [_]f32{ 1, 2, 3, 4 };
    var heads = [_]checkpoint.Head{.{ .dim = 1, .positions = &.{ 0, 1, 2, 3 }, .coverage_ends = &.{ 1, 2, 3, 4 }, .keys = &.{ 0, 1, 2, 3 }, .values = &values, .bias = &.{ 0, 0, 0, 0 } }};
    var donors = [_]checkpoint.Donor{.{ .heads = &heads }};
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = 4,
        .tokens = &.{ 0, 0, 0, 0 },
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &donors,
    };
    var descriptors: [1]Descriptor = undefined;
    const old = try capture(state, 2, &descriptors, .{});
    try std.testing.expectEqual(@as(usize, 2), old[0].rows);
    const expected = try immutableDigest(state, old, .{});
    values[0] = 2;
    try std.testing.expect(!std.mem.eql(u8, &expected, &try immutableDigest(state, old, .{})));
    values[0] = 1;
    heads[0].dim = 2;
    try std.testing.expectError(error.CompactionHistoryChanged, immutableDigest(state, old, .{}));
    heads[0].dim = 1;
    donors[0].window = 512;
    try std.testing.expectError(error.CompactionHistoryChanged, immutableDigest(state, old, .{}));
}

test "attention_match Metal cycle seal permits adjacent packed extension and protects prior rows" {
    const a = std.testing.allocator;
    const dim = 64;
    const keys: [6 * dim]f32 = @splat(0);
    const values: [6 * dim]f32 = @splat(2);
    var previous = try (checkpoint.packed_storage.Packed{}).append(a, 1, dim, keys[dim .. 4 * dim], values[dim .. 4 * dim], .{});
    defer previous.deinit(a);
    var heads = [_]checkpoint.Head{.{ .dim = dim, .compacted_until = 4, .positions = &.{ 0, 1, 2, 3, 4, 5 }, .coverage_ends = &.{ 1, 2, 3, 4, 5, 6 }, .keys = &keys, .values = &values, .bias = &.{ 0, 0, 0, 0, 0, 0 }, .compressed = previous }};
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = 6,
        .tokens = &.{ 0, 0, 0, 0, 0, 0 },
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &.{.{ .heads = &heads }},
    };
    var descriptors: [1]Descriptor = undefined;
    const old = try capture(state, 1, &descriptors, .{});
    const expected = try immutableDigest(state, old, .{});
    var extended = try previous.append(a, 4, dim, keys[4 * dim .. 5 * dim], values[4 * dim .. 5 * dim], .{});
    defer extended.deinit(a);
    try std.testing.expectEqual(@as(usize, 1), extended.spans.len);
    try std.testing.expectEqual(previous.spans[0].rows + 1, extended.spans[0].rows);
    heads[0].compressed = extended;
    try std.testing.expectEqual(expected, try immutableDigest(state, old, .{}));

    // An extension cannot rewrite encoded history or turn a prior exact row
    // into a compressed row. Both remain part of the protected representation.
    const encoded_keys = @constCast(extended.keys);
    encoded_keys[0] ^= 1;
    try std.testing.expect(!std.mem.eql(u8, &expected, &try immutableDigest(state, old, .{})));
    encoded_keys[0] ^= 1;
    var spans = [_]checkpoint.packed_storage.Span{extended.spans[0]};
    heads[0].compressed.spans = &spans;
    spans[0].first = 0;
    try std.testing.expect(!std.mem.eql(u8, &expected, &try immutableDigest(state, old, .{})));
    spans[0] = extended.spans[0];
    spans[0].rows = previous.spans[0].rows - 1;
    try std.testing.expect(!std.mem.eql(u8, &expected, &try immutableDigest(state, old, .{})));

    // If an older fitted interval ended with an exact row, merging across
    // that row must still be rejected even though packed bytes are unchanged.
    spans[0] = previous.spans[0];
    spans[0].rows -= 1;
    const exact_tail_digest = try immutableDigest(state, old, .{});
    spans[0].rows += 1;
    try std.testing.expect(!std.mem.eql(u8, &exact_tail_digest, &try immutableDigest(state, old, .{})));
}
