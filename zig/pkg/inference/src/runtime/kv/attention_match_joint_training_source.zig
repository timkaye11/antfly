// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Private exact flat P/Y source for joint-mass diagnostics.
const std = @import("std");
const solver = @import("attention_match.zig");

pub const default_combined_bytes: usize = 512 * 1024 * 1024;
pub const supported_hard_maximum_combined_bytes: usize = 1024 * 1024 * 1024;
pub const resident_native_hybrid_combined_bytes: usize = 2 * 1024 * 1024 * 1024;
pub const initial_query_batch_rows: usize = 16;
pub const diagnostic_query_batch_rows: usize = 256;

pub const FallbackReason = enum { cap, allocation, authority_changed };
pub const AllocationAuthority = struct {
    context: ?*anyopaque,
    admits_fn: *const fn (?*anyopaque, usize, usize, solver.Control) anyerror!bool,
};
pub const ProbabilityStorage = enum { f64, exact_f32_widened };
pub const CaptureBackend = struct {
    context: ?*anyopaque,
    /// Called only after planning selects a tile and before any allocation or
    /// capture callback. Implementations must reject reconfiguration after use.
    configure_fn: ?*const fn (?*anyopaque, usize) anyerror!void = null,
    workspace_fn: *const fn (?*anyopaque, solver.Head, usize) anyerror!usize,
    begin_fn: *const fn (?*anyopaque, std.mem.Allocator, solver.Head, usize, solver.Control) anyerror!?*anyopaque,
    tile_fn: *const fn (?*anyopaque, []const f64, []f64, []f64, solver.Control) anyerror!void,
    end_fn: *const fn (?*anyopaque, std.mem.Allocator) void,
};
pub const Counters = struct {
    prepare_attempts: usize = 0,
    builds: usize = 0,
    fallbacks_cap: usize = 0,
    fallbacks_allocation: usize = 0,
    attention_calls: usize = 0,
    fit_hits: usize = 0,
    selection_hits: usize = 0,
    misses: usize = 0,
    query_batch_rows: usize = 0,
    cache_bytes: usize = 0,
    probability_bytes: usize = 0,
    probability_storage: ProbabilityStorage = .f64,
    capture_peak_bytes: usize = 0,
    steady_peak_bytes: usize = 0,
    capture_begins: usize = 0,
    capture_tiles: usize = 0,
    capture_ends: usize = 0,
};

pub const Options = struct {
    combined_cap: usize = default_combined_bytes,
    /// Every live byte outside this source: authenticated source/train/heldout,
    /// range/branch state, and caller-owned result history.
    external_live_bytes: usize,
    /// Exact maximum additional consumer lifetime supplied by replay's joint
    /// workspace profiler, including nested backend/candidate/validation peaks.
    consumer_peak_bytes: usize,
    query_batch_rows: usize = initial_query_batch_rows,
    probability_storage: ProbabilityStorage = .f64,
    capture_backend: ?CaptureBackend = null,
    allocation_authority: ?AllocationAuthority = null,
};

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.InvalidAttentionShape;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.InvalidAttentionShape;
}

fn backendEqual(a: ?solver.Backend, b: ?solver.Backend) bool {
    if (a == null or b == null) return a == null and b == null;
    const x = a.?;
    const y = b.?;
    return x.validation_batch == y.validation_batch and x.raw_range_policy == y.raw_range_policy and x.context == y.context and
        x.workspace_fn == y.workspace_fn and x.attend_fn == y.attend_fn and
        x.summary_fn == y.summary_fn and x.values_fn == y.values_fn and
        x.select_fn == y.select_fn and x.raw_select_fn == y.raw_select_fn and
        x.raw_range_workspace_fn == y.raw_range_workspace_fn and x.raw_range_begin_fn == y.raw_range_begin_fn and x.raw_range_end_fn == y.raw_range_end_fn;
}

pub const Source = struct {
    allocator: std.mem.Allocator,
    keys_ptr: [*]const f64,
    values_ptr: [*]const f64,
    bias_ptr: ?[*]const f64,
    train_ptr: [*]const f64,
    rows: usize,
    probability_row_start: usize = 0,
    probability_rows: usize = 0,
    dim: usize,
    query_count: usize,
    scale: f64,
    softcap: f64,
    backend: ?solver.Backend,
    query_batch_rows: usize,
    cache_bytes: usize,
    consumer_peak_bytes: usize,
    digest: [32]u8,
    fit_hits: usize = 0,
    selection_hits: usize = 0,
    probabilities: []f64 = &.{},
    probabilities_f32: []f32 = &.{},
    outputs: []f64,
    owns_outputs: bool = true,

    fn cachedRows(self: *const Source) usize {
        return if (self.probability_rows == 0) self.rows else self.probability_rows;
    }

    pub fn probability(self: *const Source, query: usize, row: usize) !f64 {
        const cached_rows = self.cachedRows();
        if (query >= self.query_count or row < self.probability_row_start or row - self.probability_row_start >= cached_rows) return error.InvalidAttentionShape;
        const index = try add(try mul(query, cached_rows), row - self.probability_row_start);
        return if (self.probabilities_f32.len != 0) self.probabilities_f32[index] else self.probabilities[index];
    }

    pub fn copyChunk(self: *const Source, query: usize, start: usize, destination: []f64) !void {
        const cached_rows = self.cachedRows();
        if (query >= self.query_count or start < self.probability_row_start or start - self.probability_row_start > cached_rows or destination.len > cached_rows - (start - self.probability_row_start)) return error.InvalidAttentionShape;
        const first = try add(try mul(query, cached_rows), start - self.probability_row_start);
        if (self.probabilities_f32.len != 0) {
            for (destination, self.probabilities_f32[first..][0..destination.len]) |*out, value| out.* = value;
        } else @memcpy(destination, self.probabilities[first..][0..destination.len]);
    }

    pub fn validate(self: *const Source, original: solver.Head, train: []const f64, cfg: solver.Config) !void {
        const cached_rows = self.cachedRows();
        const expected = try mul(self.query_count, cached_rows);
        if (original.keys.ptr != self.keys_ptr or original.values.ptr != self.values_ptr or
            (if (original.bias) |bias| bias.ptr else null) != self.bias_ptr or train.ptr != self.train_ptr or
            original.rows != self.rows or original.dim != self.dim or train.len != try mul(self.query_count, self.dim) or
            self.probability_row_start > self.rows or cached_rows > self.rows - self.probability_row_start or
            original.scale != self.scale or original.softcap != self.softcap or !backendEqual(self.backend, cfg.backend) or
            !((self.probabilities.len == expected and self.probabilities_f32.len == 0) or (self.probabilities.len == 0 and self.probabilities_f32.len == expected))) return error.InvalidFitConfig;
    }

    pub fn deinit(self: *Source) void {
        if (self.probabilities.len != 0) self.allocator.free(self.probabilities);
        if (self.probabilities_f32.len != 0) self.allocator.free(self.probabilities_f32);
        if (self.owns_outputs) self.allocator.free(self.outputs);
        self.* = undefined;
    }

    pub fn noteFitHit(self: *Source, counters: ?*Counters) void {
        self.fit_hits += 1;
        if (counters) |value| value.fit_hits += 1;
    }

    pub fn noteSelectionHit(self: *Source, counters: ?*Counters) void {
        self.selection_hits += 1;
        if (counters) |value| value.selection_hits += 1;
    }
};

/// Takes ownership of one exact-f32 probability span while borrowing the
/// already widened global Y. Probability lookups outside the authenticated
/// range fail closed; the original Head identity remains global.
fn initBorrowedRangeF32Impl(
    allocator: std.mem.Allocator,
    original: solver.Head,
    train: []const f64,
    cfg: solver.Config,
    range_start: usize,
    range_rows: usize,
    probabilities: []f32,
    outputs: []f64,
    consumer_peak_bytes: usize,
    validate_head: bool,
) !Source {
    if (validate_head) try original.validate();
    if (train.len == 0 or train.len % original.dim != 0 or range_rows == 0 or range_start > original.rows or range_rows > original.rows - range_start)
        return error.InvalidFitConfig;
    const q = train.len / original.dim;
    if (probabilities.len != try mul(q, range_rows) or outputs.len != try mul(q, original.dim) or consumer_peak_bytes == 0) return error.InvalidAttentionShape;
    for (probabilities) |p| if (!std.math.isFinite(p) or p < 0) return error.NonFiniteAttention;
    for (outputs) |y| if (!std.math.isFinite(y)) return error.NonFiniteAttention;
    return .{
        .allocator = allocator,
        .keys_ptr = original.keys.ptr,
        .values_ptr = original.values.ptr,
        .bias_ptr = if (original.bias) |bias| bias.ptr else null,
        .train_ptr = train.ptr,
        .rows = original.rows,
        .probability_row_start = range_start,
        .probability_rows = range_rows,
        .dim = original.dim,
        .query_count = q,
        .scale = original.scale,
        .softcap = original.softcap,
        .backend = cfg.backend,
        .query_batch_rows = initial_query_batch_rows,
        .cache_bytes = try mul(probabilities.len, @sizeOf(f32)),
        .consumer_peak_bytes = consumer_peak_bytes,
        .digest = @splat(0),
        .probabilities_f32 = probabilities,
        .outputs = outputs,
        .owns_outputs = false,
    };
}

pub fn initBorrowedRangeF32(
    allocator: std.mem.Allocator,
    original: solver.Head,
    train: []const f64,
    cfg: solver.Config,
    range_start: usize,
    range_rows: usize,
    probabilities: []f32,
    outputs: []f64,
    consumer_peak_bytes: usize,
) !Source {
    return initBorrowedRangeF32Impl(allocator, original, train, cfg, range_start, range_rows, probabilities, outputs, consumer_peak_bytes, true);
}

/// Internal bounded-route seam. The owning PreparedInput authenticated the
/// immutable complete Head before any range load. This variant preserves all
/// range, probability, output, backend, and pointer binding checks without
/// rescanning the same N*D head for each cache replacement.
pub fn initBorrowedRangeF32FromValidatedHead(
    allocator: std.mem.Allocator,
    original: solver.Head,
    train: []const f64,
    cfg: solver.Config,
    range_start: usize,
    range_rows: usize,
    probabilities: []f32,
    outputs: []f64,
    consumer_peak_bytes: usize,
) !Source {
    return initBorrowedRangeF32Impl(allocator, original, train, cfg, range_start, range_rows, probabilities, outputs, consumer_peak_bytes, false);
}

test "validated-head range constructor retains payload and binding checks" {
    const head: solver.Head = .{ .keys = &.{ 1, 2 }, .values = &.{ 3, 4 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const train = [_]f64{0.25};
    const bad_probabilities = try std.testing.allocator.dupe(f32, &.{std.math.nan(f32)});
    defer std.testing.allocator.free(bad_probabilities);
    var outputs = [_]f64{3.5};
    try std.testing.expectError(error.NonFiniteAttention, initBorrowedRangeF32FromValidatedHead(std.testing.allocator, head, &train, .{ .target_rows = 1 }, 0, 1, bad_probabilities, &outputs, 4096));

    const probabilities = try std.testing.allocator.dupe(f32, &.{0.5});
    var source = try initBorrowedRangeF32FromValidatedHead(std.testing.allocator, head, &train, .{ .target_rows = 1 }, 0, 1, probabilities, &outputs, 4096);
    defer source.deinit();
    try source.validate(head, &train, .{ .target_rows = 1 });
    const other_keys = try std.testing.allocator.dupe(f64, head.keys);
    defer std.testing.allocator.free(other_keys);
    const other_head: solver.Head = .{ .keys = other_keys, .values = head.values, .bias = head.bias, .rows = 2, .dim = 1, .scale = 1 };
    try std.testing.expectError(error.InvalidFitConfig, source.validate(other_head, &train, .{ .target_rows = 1 }));
}

test "borrowed range source matches full f32 span and rejects outside rows" {
    const head: solver.Head = .{ .keys = &.{ 1, 2, 3, 4 }, .values = &.{ 5, 6, 7, 8 }, .bias = &.{ 0, 0, 0, 0 }, .rows = 4, .dim = 1, .scale = 1 };
    const train = [_]f64{ 0.25, -0.5 };
    const probabilities = try std.testing.allocator.dupe(f32, &.{ 0.2, 0.3, 0.4, 0.5 });
    const outputs = [_]f64{ 6.5, 6.75 };
    var source = try initBorrowedRangeF32(std.testing.allocator, head, &train, .{ .target_rows = 1 }, 1, 2, probabilities, @constCast(outputs[0..]), 4096);
    defer source.deinit();
    try source.validate(head, &train, .{ .target_rows = 1 });
    try std.testing.expectEqual(@as(f64, @floatCast(@as(f32, 0.2))), try source.probability(0, 1));
    try std.testing.expectEqual(@as(f64, @floatCast(@as(f32, 0.5))), try source.probability(1, 2));
    try std.testing.expectError(error.InvalidAttentionShape, source.probability(0, 0));
    try std.testing.expectError(error.InvalidAttentionShape, source.probability(0, 3));
    var copied: [2]f64 = undefined;
    try source.copyChunk(1, 1, &copied);
    try std.testing.expectEqualSlices(f64, &.{ @as(f64, @floatCast(@as(f32, 0.4))), @as(f64, @floatCast(@as(f32, 0.5))) }, &copied);

    const full_probabilities = try std.testing.allocator.dupe(f32, &.{ 0.1, 0.2, 0.3, 0.4, 0.6, 0.4, 0.5, 0.5 });
    var full: Source = .{ .allocator = std.testing.allocator, .keys_ptr = head.keys.ptr, .values_ptr = head.values.ptr, .bias_ptr = head.bias.?.ptr, .train_ptr = train[0..].ptr, .rows = 4, .dim = 1, .query_count = 2, .scale = 1, .softcap = 0, .backend = null, .query_batch_rows = 16, .cache_bytes = full_probabilities.len * @sizeOf(f32), .consumer_peak_bytes = 0, .digest = @splat(0), .probabilities_f32 = full_probabilities, .outputs = @constCast(outputs[0..]), .owns_outputs = false };
    defer full.deinit();
    for (0..2) |query| for (1..3) |row| try std.testing.expectEqual(try full.probability(query, row), try source.probability(query, row));
    var full_copied: [2]f64 = undefined;
    try full.copyChunk(1, 1, &full_copied);
    try std.testing.expectEqualSlices(f64, &full_copied, &copied);
}

test "borrowed range source requires and preserves explicit consumer authority" {
    const head: solver.Head = .{ .keys = &.{ 1, 2 }, .values = &.{ 3, 4 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const train = [_]f64{0.25};
    var probabilities = [_]f32{0.5};
    var outputs = [_]f64{3.5};
    try std.testing.expectError(error.InvalidAttentionShape, initBorrowedRangeF32(std.testing.allocator, head, &train, .{ .target_rows = 1 }, 0, 1, &probabilities, &outputs, 0));
    const owned_probabilities = try std.testing.allocator.dupe(f32, &probabilities);
    var source = try initBorrowedRangeF32(std.testing.allocator, head, &train, .{ .target_rows = 1 }, 0, 1, owned_probabilities, &outputs, 12345);
    defer source.deinit();
    try std.testing.expectEqual(@as(usize, 12345), source.consumer_peak_bytes);
}

pub const Prepare = union(enum) { ready: Source, fallback: FallbackReason };

pub fn prepare(allocator: std.mem.Allocator, original: solver.Head, train: []const f64, cfg: solver.Config, options: Options, counters: ?*Counters) !Prepare {
    if (counters) |value| {
        value.prepare_attempts += 1;
        value.query_batch_rows = options.query_batch_rows;
        value.probability_storage = options.probability_storage;
    }
    try original.validate();
    try solver.validateConfigDiagnostic(cfg, original.rows);
    const admitted_cap = try cfg.admittedCombinedCap();
    if (train.len == 0 or train.len % original.dim != 0 or options.combined_cap == 0 or
        options.combined_cap > admitted_cap or
        (cfg.workspace_admission == .resident_native_hybrid_2g_v1 and options.combined_cap != resident_native_hybrid_combined_bytes) or
        (options.query_batch_rows != initial_query_batch_rows and options.query_batch_rows != 64 and
            (options.query_batch_rows != diagnostic_query_batch_rows or options.capture_backend == null)))
        return error.InvalidFitConfig;
    const q = train.len / original.dim;
    const p_count = try mul(q, original.rows);
    const y_count = try mul(q, original.dim);
    const probability_bytes = try mul(p_count, if (options.probability_storage == .f64) @sizeOf(f64) else @sizeOf(f32));
    const cache_bytes = try add(probability_bytes, try mul(y_count, @sizeOf(f64)));
    const scratch_bytes = try mul(try add(try mul(options.query_batch_rows, original.rows), try mul(options.query_batch_rows, original.dim)), @sizeOf(f64));
    const backend_bytes = if (options.capture_backend) |backend| try backend.workspace_fn(backend.context, original, options.query_batch_rows) else if (cfg.backend) |backend| try backend.workspace_fn(backend.context, original.rows, 1, options.query_batch_rows, original.dim) else 0;
    const exact_capture_scratch = try mul(try mul(options.query_batch_rows, original.rows), @sizeOf(f64));
    const capture_work = if (options.probability_storage == .f64) @max(scratch_bytes, backend_bytes) else try add(exact_capture_scratch, backend_bytes);
    const capture_peak = try add(options.external_live_bytes, try add(cache_bytes, capture_work));
    const steady_peak = try add(options.external_live_bytes, try add(cache_bytes, options.consumer_peak_bytes));
    if (counters) |value| {
        value.cache_bytes = cache_bytes;
        value.probability_bytes = probability_bytes;
        value.capture_peak_bytes = capture_peak;
        value.steady_peak_bytes = steady_peak;
    }
    if (@max(capture_peak, steady_peak) > options.combined_cap) {
        if (counters) |value| {
            value.fallbacks_cap += 1;
            value.misses += 1;
        }
        return .{ .fallback = .cap };
    }
    if (options.allocation_authority) |authority| {
        try cfg.control.check();
        if (!try authority.admits_fn(authority.context, probability_bytes, try mul(y_count, @sizeOf(f64)), cfg.control)) {
            if (counters) |value| value.misses += 1;
            return .{ .fallback = .authority_changed };
        }
        try cfg.control.check();
    }
    try cfg.control.check();
    const probabilities = allocator.alloc(f64, if (options.probability_storage == .f64) p_count else 0) catch |err| switch (err) {
        error.OutOfMemory => {
            if (counters) |value| {
                value.fallbacks_allocation += 1;
                value.misses += 1;
            }
            return .{ .fallback = .allocation };
        },
    };
    errdefer allocator.free(probabilities);
    const probabilities_f32 = allocator.alloc(f32, if (options.probability_storage == .exact_f32_widened) p_count else 0) catch |err| switch (err) {
        error.OutOfMemory => {
            allocator.free(probabilities);
            if (counters) |value| {
                value.fallbacks_allocation += 1;
                value.misses += 1;
            }
            return .{ .fallback = .allocation };
        },
    };
    errdefer allocator.free(probabilities_f32);
    const outputs = allocator.alloc(f64, y_count) catch |err| switch (err) {
        error.OutOfMemory => {
            allocator.free(probabilities_f32);
            allocator.free(probabilities);
            if (counters) |value| {
                value.fallbacks_allocation += 1;
                value.misses += 1;
            }
            return .{ .fallback = .allocation };
        },
    };
    errdefer allocator.free(outputs);
    const capture = allocator.alloc(f64, if (options.probability_storage == .exact_f32_widened) try mul(options.query_batch_rows, original.rows) else 0) catch |err| switch (err) {
        error.OutOfMemory => {
            allocator.free(outputs);
            allocator.free(probabilities_f32);
            allocator.free(probabilities);
            if (counters) |value| {
                value.fallbacks_allocation += 1;
                value.misses += 1;
            }
            return .{ .fallback = .allocation };
        },
    };
    defer allocator.free(capture);
    var capture_owner: ?*anyopaque = null;
    if (options.capture_backend) |backend| {
        if (cfg.backend == null or cfg.backend.?.attend_fn == null) return error.InvalidFitConfig;
        capture_owner = try backend.begin_fn(backend.context, allocator, original, options.query_batch_rows, cfg.control);
        if (capture_owner == null) return error.InvalidFitConfig;
        if (counters) |value| value.capture_begins += 1;
    }
    defer if (options.capture_backend) |backend| if (capture_owner) |owner| {
        backend.end_fn(owner, allocator);
        if (counters) |value| value.capture_ends += 1;
    };
    var offset: usize = 0;
    while (offset < q) {
        try cfg.control.check();
        const take = @min(options.query_batch_rows, q - offset);
        const count = try mul(take, original.rows);
        const destination = if (options.probability_storage == .f64) probabilities[offset * original.rows ..][0..count] else capture[0..count];
        if (options.capture_backend) |backend| {
            try backend.tile_fn(capture_owner.?, train[offset * original.dim ..][0 .. take * original.dim], destination, outputs[offset * original.dim ..][0 .. take * original.dim], cfg.control);
            if (counters) |value| value.capture_tiles += 1;
        } else try solver.attendBatch(allocator, original, train[offset * original.dim ..][0 .. take * original.dim], destination, outputs[offset * original.dim ..][0 .. take * original.dim], null, cfg);
        if (options.probability_storage == .exact_f32_widened) for (destination, probabilities_f32[offset * original.rows ..][0..count]) |value, *stored| {
            if (!std.math.isFinite(value)) return error.BackendPrecisionContractViolation;
            const narrowed: f32 = @floatCast(value);
            if (!std.math.isFinite(narrowed) or @as(f64, narrowed) != value) return error.BackendPrecisionContractViolation;
            stored.* = narrowed;
        };
        if (counters) |value| value.attention_calls += 1;
        offset += take;
    }
    const Canonical = struct {
        fn word(hash: *std.crypto.hash.sha2.Sha256, value: u64) void {
            var bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &bytes, value, .little);
            hash.update(&bytes);
        }
        fn floats(hash: *std.crypto.hash.sha2.Sha256, values: []const f64) void {
            for (values) |value| word(hash, @bitCast(value));
        }
        fn widenedF32(hash: *std.crypto.hash.sha2.Sha256, values: []const f32) void {
            for (values) |value| word(hash, @bitCast(@as(f64, value)));
        }
    };
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    Canonical.word(&hash, std.math.cast(u64, q) orelse return error.InvalidAttentionShape);
    Canonical.word(&hash, std.math.cast(u64, original.rows) orelse return error.InvalidAttentionShape);
    Canonical.word(&hash, std.math.cast(u64, original.dim) orelse return error.InvalidAttentionShape);
    Canonical.word(&hash, std.math.cast(u64, options.query_batch_rows) orelse return error.InvalidAttentionShape);
    if (options.probability_storage == .f64) Canonical.floats(&hash, probabilities) else Canonical.widenedF32(&hash, probabilities_f32);
    Canonical.floats(&hash, outputs);
    const digest = hash.finalResult();
    if (counters) |value| value.builds += 1;
    return .{ .ready = .{ .allocator = allocator, .keys_ptr = original.keys.ptr, .values_ptr = original.values.ptr, .bias_ptr = if (original.bias) |bias| bias.ptr else null, .train_ptr = train.ptr, .rows = original.rows, .probability_rows = original.rows, .dim = original.dim, .query_count = q, .scale = original.scale, .softcap = original.softcap, .backend = cfg.backend, .query_batch_rows = options.query_batch_rows, .cache_bytes = cache_bytes, .consumer_peak_bytes = options.consumer_peak_bytes, .digest = digest, .probabilities = probabilities, .probabilities_f32 = probabilities_f32, .outputs = outputs } };
}

test "training capture lifecycle preserves exact tiles and drains" {
    const Fake = struct {
        const Owner = struct { begins: *usize, tiles: *usize, ends: *usize, dim: usize, rows: usize };
        var begins: usize = 0;
        var tiles: usize = 0;
        var ends: usize = 0;
        var fail_tile: bool = false;
        fn values(head: solver.Head, queries: []const f64, p: []f64, y: []f64) void {
            const q = queries.len / head.dim;
            for (0..q) |qi| {
                for (0..head.rows) |ri| p[qi * head.rows + ri] = queries[qi * head.dim] + @as(f64, @floatFromInt(ri));
                for (0..head.dim) |di| y[qi * head.dim + di] = queries[qi * head.dim + di] + 1;
            }
        }
        fn workspace(_: ?*anyopaque, _: solver.Head, _: usize) !usize {
            return 128;
        }
        fn begin(_: ?*anyopaque, a: std.mem.Allocator, h: solver.Head, _: usize, c: solver.Control) !?*anyopaque {
            try c.check();
            const o = try a.create(Owner);
            begins += 1;
            o.* = .{ .begins = &begins, .tiles = &tiles, .ends = &ends, .dim = h.dim, .rows = h.rows };
            return o;
        }
        fn tile(raw: ?*anyopaque, q: []const f64, p: []f64, y: []f64, c: solver.Control) !void {
            try c.check();
            const o: *Owner = @ptrCast(@alignCast(raw.?));
            if (fail_tile) return error.CompactionCancelled;
            tiles += 1;
            const h: solver.Head = .{ .keys = &.{ 0, 0 }, .values = &.{ 0, 0 }, .rows = o.rows, .dim = o.dim, .scale = 1 };
            values(h, q, p, y);
        }
        fn end(raw: ?*anyopaque, a: std.mem.Allocator) void {
            const o: *Owner = @ptrCast(@alignCast(raw.?));
            ends += 1;
            a.destroy(o);
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, h: solver.Head, q: []const f64, p: []f64, y: ?[]f64, _: ?[]f64, c: solver.Control) !void {
            try c.check();
            values(h, q, p, y.?);
        }
        fn backend() CaptureBackend {
            return .{ .context = null, .workspace_fn = workspace, .begin_fn = begin, .tile_fn = tile, .end_fn = end };
        }
    };
    Fake.begins = 0;
    Fake.tiles = 0;
    Fake.ends = 0;
    const h: solver.Head = .{ .keys = &.{ 0, 0 }, .values = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    var train: [17]f64 = undefined;
    for (&train, 0..) |*value, i| value.* = @floatFromInt(i + 1);
    const cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20, .backend = .{ .workspace_fn = struct {
        fn f(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 128;
        }
    }.f, .attend_fn = Fake.attend } };
    var plain = (try prepare(std.testing.allocator, h, &train, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1, .query_batch_rows = 16 }, null)).ready;
    defer plain.deinit();
    var counts: Counters = .{};
    var captured = (try prepare(std.testing.allocator, h, &train, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1, .query_batch_rows = 16, .capture_backend = Fake.backend() }, &counts)).ready;
    defer captured.deinit();
    try std.testing.expectEqualSlices(f64, plain.probabilities, captured.probabilities);
    try std.testing.expectEqualSlices(f64, plain.outputs, captured.outputs);
    try std.testing.expectEqual(@as(usize, 1), counts.capture_begins);
    try std.testing.expectEqual(@as(usize, 2), counts.capture_tiles);
    try std.testing.expectEqual(@as(usize, 1), counts.capture_ends);
    Fake.fail_tile = true;
    defer Fake.fail_tile = false;
    try std.testing.expectError(error.CompactionCancelled, prepare(std.testing.allocator, h, &train, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1, .query_batch_rows = 16, .capture_backend = Fake.backend() }, null));
    try std.testing.expectEqual(@as(usize, 2), Fake.ends);
}

test "joint flat source cap and host allocation fallback occur before attention" {
    const Fake = struct {
        var calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, probabilities: []f64, outputs: ?[]f64, log_normalizers: ?[]f64, _: solver.Control) !void {
            calls += 1;
            @memset(probabilities, 0);
            if (outputs) |values| @memset(values, 0);
            if (log_normalizers) |values| @memset(values, 0);
        }
    };
    const head: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const cfg: solver.Config = .{ .target_rows = 1, .backend = .{ .workspace_fn = Fake.workspace, .attend_fn = Fake.attend }, .max_workspace_bytes = 64 };
    var counters: Counters = .{};
    Fake.calls = 0;
    const capped = try prepare(std.testing.allocator, head, &.{ 0, 1 }, cfg, .{ .combined_cap = 100, .external_live_bytes = 50, .consumer_peak_bytes = 64 }, &counters);
    try std.testing.expectEqual(FallbackReason.cap, capped.fallback);
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
    const exact_capped = try prepare(std.testing.allocator, head, &.{ 0, 1 }, cfg, .{ .combined_cap = 100, .external_live_bytes = 50, .consumer_peak_bytes = 64, .probability_storage = .exact_f32_widened }, &counters);
    try std.testing.expectEqual(FallbackReason.cap, exact_capped.fallback);
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
    const Authority = struct {
        calls: usize = 0,
        fn admits(raw: ?*anyopaque, probability_bytes: usize, output_bytes: usize, control: solver.Control) !bool {
            try control.check();
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            try std.testing.expectEqual(@as(usize, 4 * @sizeOf(f32)), probability_bytes);
            try std.testing.expectEqual(@as(usize, 2 * @sizeOf(f64)), output_bytes);
            return false;
        }
    };
    var authority: Authority = .{};
    const changed = try prepare(std.testing.allocator, head, &.{ 0, 1 }, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1, .probability_storage = .exact_f32_widened, .allocation_authority = .{ .context = &authority, .admits_fn = Authority.admits } }, &counters);
    try std.testing.expectEqual(FallbackReason.authority_changed, changed.fallback);
    try std.testing.expectEqual(@as(usize, 1), authority.calls);
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    const failed = try prepare(failing.allocator(), head, &.{ 0, 1 }, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 64 }, &counters);
    try std.testing.expectEqual(FallbackReason.allocation, failed.fallback);
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
    try std.testing.expectError(error.InvalidFitConfig, prepare(std.testing.allocator, head, &.{ 0, 1 }, cfg, .{ .combined_cap = 0, .external_live_bytes = 0, .consumer_peak_bytes = 64 }, null));
    try std.testing.expectError(error.InvalidFitConfig, prepare(std.testing.allocator, head, &.{ 0, 1 }, cfg, .{ .combined_cap = supported_hard_maximum_combined_bytes + 1, .external_live_bytes = 0, .consumer_peak_bytes = 64 }, null));
    var resident_cfg = cfg;
    resident_cfg.max_workspace_bytes = resident_native_hybrid_combined_bytes;
    resident_cfg.workspace_admission = .resident_native_hybrid_2g_v1;
    try std.testing.expectError(error.InvalidFitConfig, prepare(std.testing.allocator, head, &.{ 0, 1 }, resident_cfg, .{ .combined_cap = supported_hard_maximum_combined_bytes, .external_live_bytes = 0, .consumer_peak_bytes = 64 }, null));
    var resident = (try prepare(std.testing.allocator, head, &.{ 0, 1 }, resident_cfg, .{ .combined_cap = resident_native_hybrid_combined_bytes, .external_live_bytes = 0, .consumer_peak_bytes = 64 }, null)).ready;
    resident.deinit();
    const diagnostic_large = try prepare(std.testing.allocator, head, &.{ 0, 1 }, cfg, .{ .combined_cap = supported_hard_maximum_combined_bytes, .external_live_bytes = 0, .consumer_peak_bytes = 64 }, null);
    switch (diagnostic_large) {
        .ready => |ready_value| {
            var ready = ready_value;
            ready.deinit();
        },
        .fallback => return error.TestUnexpectedResult,
    }
}

test "query batch 64 preserves exact cache and output across a 65-query tail" {
    const Fake = struct {
        var calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, head: solver.Head, queries: []const f64, probabilities: []f64, outputs: ?[]f64, _: ?[]f64, control: solver.Control) !void {
            calls += 1;
            for (0..queries.len / head.dim) |qi| {
                _ = try solver.attend(head, queries[qi * head.dim ..][0..head.dim], probabilities[qi * head.rows ..][0..head.rows], outputs.?[qi * head.dim ..][0..head.dim]);
            }
            try control.check();
        }
    };
    const head: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    var train: [65]f64 = undefined;
    for (&train, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i)) / 64;
    const cfg: solver.Config = .{ .target_rows = 1, .backend = .{ .workspace_fn = Fake.workspace, .attend_fn = Fake.attend }, .max_workspace_bytes = 1 << 20 };
    try std.testing.expectError(error.InvalidFitConfig, prepare(std.testing.allocator, head, &train, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 4096, .query_batch_rows = 32 }, null));
    Fake.calls = 0;
    var small = (try prepare(std.testing.allocator, head, &train, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 4096, .query_batch_rows = 16 }, null)).ready;
    defer small.deinit();
    try std.testing.expectEqual(@as(usize, 5), Fake.calls);
    Fake.calls = 0;
    var large = (try prepare(std.testing.allocator, head, &train, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 4096, .query_batch_rows = 64 }, null)).ready;
    defer large.deinit();
    try std.testing.expectEqual(@as(usize, 2), Fake.calls);
    try std.testing.expectEqualSlices(f64, small.probabilities, large.probabilities);
    try std.testing.expectEqualSlices(f64, small.outputs, large.outputs);
}

test "query batch 64 capture cap and allocation fallback precede callback" {
    const Fake = struct {
        var calls: usize = 0;
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            calls += 1;
        }
    };
    const head: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const train: [65]f64 = @splat(0);
    const cfg: solver.Config = .{ .target_rows = 1, .backend = .{ .workspace_fn = Fake.workspace, .attend_fn = Fake.attend }, .max_workspace_bytes = 1 << 20 };
    Fake.calls = 0;
    const capped = try prepare(std.testing.allocator, head, &train, cfg, .{ .combined_cap = 2000, .external_live_bytes = 0, .consumer_peak_bytes = 1, .query_batch_rows = 64 }, null);
    try std.testing.expectEqual(FallbackReason.cap, capped.fallback);
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    const failed = try prepare(failing.allocator(), head, &train, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 4096, .query_batch_rows = 64 }, null);
    try std.testing.expectEqual(FallbackReason.allocation, failed.fallback);
    try std.testing.expectEqual(@as(usize, 0), Fake.calls);
}

test "diagnostic capture 256 is exact-f32 byte equivalent to 64 including tail 257" {
    const Fake = struct {
        const Owner = struct { head: solver.Head };
        fn workspace(_: ?*anyopaque, _: solver.Head, _: usize) !usize {
            return 0;
        }
        fn begin(_: ?*anyopaque, a: std.mem.Allocator, head: solver.Head, _: usize, _: solver.Control) !?*anyopaque {
            const owner = try a.create(Owner);
            owner.* = .{ .head = head };
            return owner;
        }
        fn tile(raw: ?*anyopaque, queries: []const f64, probabilities: []f64, outputs: []f64, control: solver.Control) !void {
            const owner: *Owner = @ptrCast(@alignCast(raw.?));
            for (0..queries.len / owner.head.dim) |qi|
                _ = try solver.attend(owner.head, queries[qi * owner.head.dim ..][0..owner.head.dim], probabilities[qi * owner.head.rows ..][0..owner.head.rows], outputs[qi * owner.head.dim ..][0..owner.head.dim]);
            // Match the real capture contract: probabilities are Metal f32
            // values widened to f64 before the exact-f32 storage check.
            for (probabilities) |*value| value.* = @as(f32, @floatCast(value.*));
            try control.check();
        }
        fn end(raw: ?*anyopaque, a: std.mem.Allocator) void {
            a.destroy(@as(*Owner, @ptrCast(@alignCast(raw.?))));
        }
        fn configure(_: ?*anyopaque, batch_rows: usize) !void {
            if (batch_rows != 64 and batch_rows != 256) return error.InvalidFitConfig;
        }
        fn unusedWorkspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn unexpectedAttend(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            return error.UnexpectedLegacyAttention;
        }
        fn backend() CaptureBackend {
            return .{ .context = null, .configure_fn = configure, .workspace_fn = workspace, .begin_fn = begin, .tile_fn = tile, .end_fn = end };
        }
    };
    const head: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    var train: [257]f64 = undefined;
    for (&train, 0..) |*value, i| value.* = @as(f64, @floatFromInt(i)) / 256;
    const cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20, .backend = .{ .workspace_fn = Fake.unusedWorkspace, .attend_fn = Fake.unexpectedAttend } };
    var counts64: Counters = .{};
    var source64 = (try prepare(std.testing.allocator, head, &train, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1, .probability_storage = .exact_f32_widened, .query_batch_rows = 64, .capture_backend = Fake.backend() }, &counts64)).ready;
    defer source64.deinit();
    var counts256: Counters = .{};
    var source256 = (try prepare(std.testing.allocator, head, &train, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1, .probability_storage = .exact_f32_widened, .query_batch_rows = 256, .capture_backend = Fake.backend() }, &counts256)).ready;
    defer source256.deinit();
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(source64.probabilities_f32), std.mem.sliceAsBytes(source256.probabilities_f32));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(source64.outputs), std.mem.sliceAsBytes(source256.outputs));
    try std.testing.expectEqual(@as(usize, 5), counts64.capture_tiles);
    try std.testing.expectEqual(@as(usize, 2), counts256.capture_tiles);
}

test "diagnostic capture 256 cap and allocation fallback precede every callback" {
    const Fake = struct {
        var begins: usize = 0;
        var tiles: usize = 0;
        fn workspace(_: ?*anyopaque, _: solver.Head, _: usize) !usize {
            return 0;
        }
        fn begin(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: usize, _: solver.Control) !?*anyopaque {
            begins += 1;
            return null;
        }
        fn tile(_: ?*anyopaque, _: []const f64, _: []f64, _: []f64, _: solver.Control) !void {
            tiles += 1;
        }
        fn end(_: ?*anyopaque, _: std.mem.Allocator) void {}
        fn backend() CaptureBackend {
            return .{ .context = null, .workspace_fn = workspace, .begin_fn = begin, .tile_fn = tile, .end_fn = end };
        }
    };
    const head: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const train: [257]f64 = @splat(0);
    const cfg: solver.Config = .{ .target_rows = 1, .max_workspace_bytes = 1 << 20 };
    Fake.begins = 0;
    Fake.tiles = 0;
    const capped = try prepare(std.testing.allocator, head, &train, cfg, .{ .combined_cap = 4096, .external_live_bytes = 0, .consumer_peak_bytes = 1, .probability_storage = .exact_f32_widened, .query_batch_rows = 256, .capture_backend = Fake.backend() }, null);
    try std.testing.expectEqual(FallbackReason.cap, capped.fallback);
    try std.testing.expectEqual(@as(usize, 0), Fake.begins);
    try std.testing.expectEqual(@as(usize, 0), Fake.tiles);
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    const failed = try prepare(failing.allocator(), head, &train, cfg, .{ .combined_cap = 1 << 20, .external_live_bytes = 0, .consumer_peak_bytes = 1, .probability_storage = .exact_f32_widened, .query_batch_rows = 256, .capture_backend = Fake.backend() }, null);
    try std.testing.expectEqual(FallbackReason.allocation, failed.fallback);
    try std.testing.expectEqual(@as(usize, 0), Fake.begins);
    try std.testing.expectEqual(@as(usize, 0), Fake.tiles);
}

test "joint flat source binds backend identity cancellation and allocation unwind" {
    const Fake = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, head: solver.Head, queries: []const f64, probabilities: []f64, outputs: ?[]f64, _: ?[]f64, control: solver.Control) !void {
            for (0..queries.len / head.dim) |i| _ = try solver.attend(head, queries[i * head.dim ..][0..head.dim], probabilities[i * head.rows ..][0..head.rows], if (outputs) |out| out[i * head.dim ..][0..head.dim] else null);
            try control.check();
        }
    };
    const head: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const cfg: solver.Config = .{ .target_rows = 1, .backend = .{ .workspace_fn = Fake.workspace, .attend_fn = Fake.attend }, .max_workspace_bytes = 1024 };
    const built = try prepare(std.testing.allocator, head, &.{ 0, 1 }, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1024 }, null);
    var source = built.ready;
    defer source.deinit();
    try source.validate(head, &.{ 0, 1 }, cfg);
    const Other = struct {
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, _: []f64, _: ?[]f64, _: ?[]f64, _: solver.Control) !void {}
    };
    try std.testing.expectError(error.InvalidFitConfig, source.validate(head, &.{ 0, 1 }, .{ .target_rows = 1, .backend = .{ .workspace_fn = Fake.workspace, .attend_fn = Other.attend }, .max_workspace_bytes = 1024 }));
    const Cancel = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls == 2) return error.Cancelled;
        }
    };
    var cancel: Cancel = .{};
    try std.testing.expectError(error.Cancelled, prepare(std.testing.allocator, head, &.{ 0, 1 }, .{ .target_rows = 1, .backend = cfg.backend, .max_workspace_bytes = 1024, .control = .{ .context = &cancel, .check_fn = Cancel.check } }, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1024, .probability_storage = .exact_f32_widened }, null));
    try std.testing.expectEqual(@as(usize, 2), cancel.calls);
    inline for (0..2) |fail_index| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const failed = try prepare(failing.allocator(), head, &.{ 0, 1 }, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1024 }, null);
        try std.testing.expectEqual(FallbackReason.allocation, failed.fallback);
    }
}

test "exact f32 probability storage preserves widened values digest and access paths" {
    const Fake = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, head: solver.Head, queries: []const f64, probabilities: []f64, outputs: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            for (0..queries.len / head.dim) |qi| {
                probabilities[qi * 2] = @as(f32, 0.25);
                probabilities[qi * 2 + 1] = @as(f32, 0.75);
                outputs.?[qi] = @as(f32, 2.75);
            }
        }
    };
    const head: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    const cfg: solver.Config = .{ .target_rows = 1, .backend = .{ .workspace_fn = Fake.workspace, .attend_fn = Fake.attend }, .max_workspace_bytes = 4096 };
    var f64_source = (try prepare(std.testing.allocator, head, &.{ 0, 1 }, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1024 }, null)).ready;
    defer f64_source.deinit();
    var counters: Counters = .{};
    var f32_source = (try prepare(std.testing.allocator, head, &.{ 0, 1 }, cfg, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1024, .probability_storage = .exact_f32_widened }, &counters)).ready;
    defer f32_source.deinit();
    try std.testing.expectEqual(ProbabilityStorage.exact_f32_widened, counters.probability_storage);
    try std.testing.expectEqual(@as(usize, 4 * @sizeOf(f32)), counters.probability_bytes);
    try std.testing.expectEqualSlices(u8, &f64_source.digest, &f32_source.digest);
    for (0..2) |qi| for (0..2) |row| try std.testing.expectEqual(try f64_source.probability(qi, row), try f32_source.probability(qi, row));
    var f64_chunk: [2]f64 = undefined;
    var f32_chunk: [2]f64 = undefined;
    try f64_source.copyChunk(1, 0, &f64_chunk);
    try f32_source.copyChunk(1, 0, &f32_chunk);
    try std.testing.expectEqualSlices(f64, &f64_chunk, &f32_chunk);
}

test "exact f32 probability contract violation and allocation failures fail before publish" {
    const Bad = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, probabilities: []f64, outputs: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            @memset(probabilities, 0.1);
            @memset(outputs.?, 0);
        }
    };
    const Good = struct {
        fn workspace(_: ?*anyopaque, _: usize, _: usize, _: usize, _: usize) !usize {
            return 0;
        }
        fn attend(_: ?*anyopaque, _: std.mem.Allocator, _: solver.Head, _: []const f64, probabilities: []f64, outputs: ?[]f64, _: ?[]f64, _: solver.Control) !void {
            @memset(probabilities, @as(f32, 0.5));
            @memset(outputs.?, 0);
        }
    };
    const head: solver.Head = .{ .keys = &.{ 0, 1 }, .values = &.{ 2, 3 }, .bias = &.{ 0, 0 }, .rows = 2, .dim = 1, .scale = 1 };
    var counters: Counters = .{};
    try std.testing.expectError(error.BackendPrecisionContractViolation, prepare(std.testing.allocator, head, &.{0}, .{ .target_rows = 1, .backend = .{ .workspace_fn = Bad.workspace, .attend_fn = Bad.attend }, .max_workspace_bytes = 4096 }, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1024, .probability_storage = .exact_f32_widened }, &counters));
    try std.testing.expectEqual(@as(usize, 0), counters.builds);
    inline for (0..3) |fail_index| {
        var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = fail_index });
        const failed = try prepare(failing.allocator(), head, &.{0}, .{ .target_rows = 1, .backend = .{ .workspace_fn = Good.workspace, .attend_fn = Good.attend }, .max_workspace_bytes = 4096 }, .{ .external_live_bytes = 0, .consumer_peak_bytes = 1024, .probability_storage = .exact_f32_widened }, null);
        try std.testing.expectEqual(FallbackReason.allocation, failed.fallback);
    }
}
