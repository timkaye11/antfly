// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Independently owned f16/FP8 Metal storage for portable attention-matching
//! checkpoints. Common FP8 intervals remain encoded in GPU-addressable no-copy
//! buffers, while attention and append execute only through Metal kernels.
const std = @import("std");
const checkpoint = @import("../runtime/kv/attention_match_checkpoint.zig");
const codec = checkpoint.codec;
const packed_storage = checkpoint.packed_storage;
const rotated = @import("../runtime/kv/turboquant_rotated.zig");
const Control = @import("../runtime/kv/attention_match.zig").Control;
const shared = @import("../backends/metal_generated_kernels.zig");
const metal_runtime = @import("../backends/metal_runtime.zig");

pub const Buffer = shared.Buffer;
pub const Ref = shared.Ref;
pub const AllocationTracker = shared.BufferAllocationTracker;

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.MetalMemoryLimitExceeded;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.MetalMemoryLimitExceeded;
}
fn rounded(bytes: usize) !usize {
    const page = std.heap.pageSize();
    return (try add(bytes, page - 1)) & ~(page - 1);
}
fn admitRunnerBaseline(max_device_bytes: usize, tracker: ?*AllocationTracker) !void {
    const baseline_bytes = try mul(6, std.heap.pageSize());
    if (baseline_bytes > max_device_bytes) return error.MetalMemoryLimitExceeded;
    if (tracker) |owner| {
        const snapshot = owner.snapshot();
        if (snapshot.tracked_combined_physical_available) {
            const live = std.math.cast(usize, snapshot.tracked_combined_live_bytes) orelse return error.MetalMemoryLimitExceeded;
            if (live > max_device_bytes - baseline_bytes) return error.MetalMemoryLimitExceeded;
        }
    }
}
const Metadata = extern struct { bias: f32, position_lo: u32, position_hi: u32, coverage_lo: u32, coverage_hi: u32 };
pub const StorageEncoding = enum(u32) { exact = 0, fp8 = 1, rotated4 = 2 };
pub const Fp8Header = extern struct {
    encoding: u32 = @backingInt(StorageEncoding.exact),
    first_row: u32 = 0,
    rows: u32 = 0,
    span_count: u32 = 0,
    compressed_rows: u32 = 0,
    key_payload_offset: u32 = 48,
    key_row_bytes: u32 = 0,
    value_row_bytes: u32 = 0,
    key_scale: f32 = 1,
    value_scale: f32 = 1,
    reserved0: u32 = 0,
    reserved1: u32 = 0,
};
comptime {
    if (@sizeOf(Metadata) != 20 or @sizeOf(Fp8Header) != 48) @compileError("Metal attention-match storage ABI mismatch");
}
fn bufferBytes(rows: usize, dim: usize, fp8_rows: usize) !usize {
    const exact_rows = std.math.sub(usize, rows, fp8_rows) catch return error.InvalidKvCheckpoint;
    var total = try rounded(try mul(try mul(exact_rows, dim), 2));
    total = try add(total, try rounded(try mul(try mul(exact_rows, dim), 2)));
    total = try add(total, try rounded(try mul(rows, @sizeOf(Metadata))));
    if (fp8_rows != 0) {
        total = try add(total, try rounded(try add(@sizeOf(Fp8Header), try mul(fp8_rows, dim))));
        total = try add(total, try rounded(try mul(fp8_rows, dim)));
    }
    return total;
}
fn rotatedBufferBytes(rows: usize, dim: usize, compressed_rows: usize, span_count: usize) !usize {
    const exact_rows = std.math.sub(usize, rows, compressed_rows) catch return error.InvalidKvCheckpoint;
    var total = try rounded(try mul(try mul(exact_rows, dim), 2));
    total = try add(total, try rounded(try mul(try mul(exact_rows, dim), 2)));
    total = try add(total, try rounded(try mul(rows, @sizeOf(Metadata))));
    if (compressed_rows != 0) {
        const prefix = try add(@sizeOf(Fp8Header), try mul(span_count, @sizeOf(packed_storage.Span)));
        total = try add(total, try rounded(try add(prefix, try mul(compressed_rows, rotated.rowBytes(dim)))));
        total = try add(total, try rounded(try mul(compressed_rows, packed_storage.valueBytes(@intCast(dim)))));
    }
    return total;
}
fn halfExact(value: f32) bool {
    const half: f16 = @floatCast(value);
    return std.math.isFinite(value) and std.math.isFinite(half) and
        @as(u32, @bitCast(value)) == @as(u32, @bitCast(@as(f32, half)));
}

fn allocationBytesImpl(state: checkpoint.State, limits: checkpoint.Limits, control: Control, global_only: bool) !usize {
    try state.validateWithControl(limits, control);
    if (try state.residentBytes() > limits.max_state_bytes) return error.KvCheckpointLimit;
    var bytes: usize = 0;
    for (state.donors) |donor| for (donor.heads) |head| {
        try control.check();
        for ([_][]const f32{ head.keys, head.values }) |values| {
            var runs = head.compressed.exactRuns(head.positions.len);
            if (head.compressed.spans.len == 0) runs = (packed_storage.Packed{}).exactRuns(head.positions.len);
            while (runs.next()) |run| {
                var offset: usize = 0;
                const exact = values[run.first * head.dim ..][0 .. run.rows * head.dim];
                while (offset < exact.len) {
                    try control.check();
                    const end = offset + @min(@as(usize, 4096), exact.len - offset);
                    for (exact[offset..end]) |value| if (!halfExact(value)) return error.KvStoragePrecisionMismatch;
                    offset = end;
                }
            }
        }
        if (head.fp8.rows != 0) {
            const first = try mul(head.fp8.first_row, head.dim);
            const count = try mul(head.fp8.rows, head.dim);
            for ([_]struct { values: []const f32, scale: f32 }{
                .{ .values = head.keys, .scale = head.fp8.key_scale },
                .{ .values = head.values, .scale = head.fp8.value_scale },
            }) |part| {
                const values = part.values[first..][0..count];
                var offset: usize = 0;
                while (offset < values.len) {
                    try control.check();
                    const end = offset + @min(@as(usize, 4096), values.len - offset);
                    for (values[offset..end]) |value| {
                        const bits = codec.encode(value, part.scale) catch return error.KvStoragePrecisionMismatch;
                        const decoded = codec.decode(bits, part.scale) catch return error.KvStoragePrecisionMismatch;
                        if (@as(u32, @bitCast(decoded)) != @as(u32, @bitCast(value))) return error.KvStoragePrecisionMismatch;
                    }
                    offset = end;
                }
            }
        }
        if (!global_only or donor.window == 0)
            bytes = try add(bytes, if (head.compressed.spans.len != 0)
                try rotatedBufferBytes(head.positions.len, head.dim, head.compressed.rows(), head.compressed.spans.len)
            else
                try bufferBytes(head.positions.len, head.dim, head.fp8.rows));
    };
    try control.check();
    return bytes;
}

pub fn allocationBytesWithControl(state: checkpoint.State, limits: checkpoint.Limits, control: Control) !usize {
    return allocationBytesImpl(state, limits, control, false);
}

/// Page-rounded resident bytes attributable to global attention-matching
/// donors. Local-window storage remains exact runtime state and is excluded
/// from compaction-savings accounting, while the complete checkpoint is still
/// validated before reporting the bound.
pub fn globalAllocationBytesWithControl(state: checkpoint.State, limits: checkpoint.Limits, control: Control) !usize {
    return allocationBytesImpl(state, limits, control, true);
}

/// Conservative resident lower bound for an ordinary exact f16 Metal cache.
/// The raw runtime stores one combined-head K buffer and one combined-head V
/// buffer per donor layer. It has no fitted bias/position/coverage metadata.
/// Metal private-buffer allocatedSize may equal the requested byte length, so
/// neither VM-page rounding nor geometric spare capacity may inflate a source
/// denominator used to authorize compaction.
pub fn ordinaryGlobalAllocationBytesWithControl(state: checkpoint.State, limits: checkpoint.Limits, control: Control) !usize {
    try state.validateWithControl(limits, control);
    if (try state.residentBytes() > limits.max_state_bytes) return error.KvCheckpointLimit;
    var bytes: usize = 0;
    for (state.donors) |donor| {
        var elements: usize = 0;
        for (donor.heads) |head| {
            try control.check();
            if (head.compressed.spans.len != 0 or head.compressed.keys.len != 0 or head.compressed.values.len != 0 or
                head.compressed.version != 0 or head.fp8.rows != 0 or head.compacted_until != 0)
                return error.UnsupportedMetalKvCheckpointEncoding;
            for ([_][]const f32{ head.keys, head.values }) |values| {
                var offset: usize = 0;
                while (offset < values.len) {
                    try control.check();
                    const end = offset + @min(@as(usize, 4096), values.len - offset);
                    for (values[offset..end]) |value| if (!halfExact(value)) return error.KvStoragePrecisionMismatch;
                    offset = end;
                }
            }
            if (donor.window == 0) elements = try add(elements, head.keys.len);
        }
        if (donor.window == 0) {
            const buffer_bytes = try mul(elements, @sizeOf(f16));
            bytes = try add(bytes, try mul(buffer_bytes, 2));
        }
    }
    try control.check();
    return bytes;
}

pub const Head = struct {
    layer: u32 = 0,
    head: u32 = 0,
    window: u32 = 0,
    dim: u32 = 0,
    rows: usize = 0,
    capacity: usize = 0,
    start: usize = 0,
    logical_length: u64 = 0,
    compacted_until: u64 = 0,
    rollback_floor: u64 = 0,
    encoding: StorageEncoding = .exact,
    compressed_rows: u32 = 0,
    span_count: u32 = 0,
    fp8_first_row: u32 = 0,
    fp8_rows: u32 = 0,
    key_scale: f32 = 1,
    value_scale: f32 = 1,
    keys: ?Buffer = null,
    values: ?Buffer = null,
    encoded_keys: ?Buffer = null,
    encoded_values: ?Buffer = null,
    metadata: ?Buffer = null,
};

pub const InventoryEntry = struct {
    layer: u32,
    head: u32,
    window: u32,
    dim: u32,
    rows: usize,
    capacity: usize,
    start: usize,
    encoding: StorageEncoding,
    physical_bytes: usize,
};
pub const InventoryOwned = struct {
    allocator: std.mem.Allocator,
    logical_length: u64,
    entries: []InventoryEntry,
    physical_bytes: usize,
    global_physical_bytes: usize,
    digest: [32]u8,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.entries);
        self.* = undefined;
    }
};
fn headPhysicalBytes(head: Head) !usize {
    var total: usize = 0;
    inline for (.{ "keys", "values", "encoded_keys", "encoded_values", "metadata" }) |field| if (@field(head, field)) |buffer| {
        total = try add(total, if (buffer.tracked_backing_bytes != 0) buffer.tracked_backing_bytes else buffer.bytes.len);
    };
    return total;
}

test "context local exact ring reuses window capacity and preserves chronological order" {
    var head: Head = .{ .window = 512, .dim = 1, .rows = 512, .capacity = 512, .start = 0, .logical_length = 512 };
    try std.testing.expectEqual(@as(usize, 512), try appendCapacity(head, 1, true));
    var positions: [512]u64 = undefined;
    for (&positions, 0..) |*position, i| position.* = i;
    const physical = (head.start + head.rows) % head.capacity;
    positions[physical] = 512;
    try publishAppend(&head, 1, 513, 512, true);
    try std.testing.expectEqual(@as(usize, 512), head.rows);
    try std.testing.expectEqual(@as(usize, 512), head.capacity);
    try std.testing.expectEqual(@as(usize, 1), head.start);
    // The overwritten row cannot be reconstructed; rollback before the new
    // logical end is no longer safe.
    try std.testing.expectEqual(@as(u64, 513), head.rollback_floor);
    for (0..head.rows) |logical| try std.testing.expectEqual(@as(u64, @intCast(logical + 1)), positions[(head.start + logical) % head.capacity]);
}

test "context local multirow append retains W minus one history for every causal query" {
    var head: Head = .{ .window = 512, .dim = 1, .rows = 512, .capacity = 512, .start = 0, .logical_length = 512 };
    const incoming: usize = 256;
    const capacity = try appendCapacity(head, incoming, true);
    try std.testing.expectEqual(@as(usize, 767), capacity);
    var positions: [767]u64 = undefined;
    for (0..512) |row| positions[row] = row;
    // grow() canonicalizes the old ring at start zero before this write.
    head.capacity = capacity;
    for (0..incoming) |row| positions[(head.start + head.rows + row) % head.capacity] = 512 + row;
    try publishAppend(&head, incoming, 768, 512, true);
    try std.testing.expectEqual(@as(usize, 767), head.rows);
    try std.testing.expectEqual(@as(usize, 1), head.start);
    try std.testing.expectEqual(@as(u64, 768), head.rollback_floor);
    for (0..incoming) |query_row| {
        const query_position = 512 + query_row;
        var visible: usize = 0;
        for (0..head.rows) |logical| {
            const position = positions[(head.start + logical) % head.capacity];
            if (position <= query_position and query_position - position < head.window) visible += 1;
        }
        try std.testing.expectEqual(@as(usize, 512), visible);
    }
    const drop = head.rows - head.window;
    head.start = (head.start + drop) % head.capacity;
    head.rows = head.window;
    for (0..head.rows) |logical| try std.testing.expectEqual(@as(u64, @intCast(256 + logical)), positions[(head.start + logical) % head.capacity]);
}

test "context local append capacity covers one oversized dispatch and otherwise requests causal rows" {
    const head: Head = .{ .window = 128, .dim = 1, .rows = 128, .capacity = 128, .logical_length = 128 };
    try std.testing.expectEqual(@as(usize, 383), try appendCapacity(head, 256, true));
    try std.testing.expectEqual(@as(usize, 128), try appendCapacity(head, 1, true));
}

test "context local exact-capacity opt-in preserves legacy spare capacity and publication by default" {
    var head: Head = .{ .window = 512, .dim = 1, .rows = 512, .capacity = 1024, .start = 0, .logical_length = 512 };
    try std.testing.expectEqual(@as(usize, 1024), try appendCapacity(head, 1, false));
    try publishAppend(&head, 1, 513, 512, false);
    try std.testing.expectEqual(@as(usize, 513), head.rows);
    try std.testing.expectEqual(@as(usize, 0), head.start);
}

fn deinitHead(head: *Head) void {
    if (head.keys) |*buffer| buffer.deinit();
    if (head.values) |*buffer| buffer.deinit();
    if (head.encoded_keys) |*buffer| buffer.deinit();
    if (head.encoded_values) |*buffer| buffer.deinit();
    if (head.metadata) |*buffer| buffer.deinit();
    head.* = .{};
}

fn appendCapacity(head: Head, incoming_rows: usize, exact_local_window: bool) !usize {
    if (incoming_rows == 0) return error.InvalidAttentionShape;
    const required = try add(head.rows, incoming_rows);
    if (head.window == 0) return try mul((try add(required, 255)) / 256, 256);
    if (!exact_local_window) return @max(required, @as(usize, head.window) + 512);
    if (head.rows > head.window) return error.InvalidAttentionShape;
    // All incoming rows are written before the earliest query attends. That
    // query needs W-1 prior rows because distance >= W is excluded. Retain the
    // causal union, not merely the final post-batch window.
    const history = @min(head.rows, @as(usize, head.window) - 1);
    return try add(history, incoming_rows);
}

fn publishAppend(head: *Head, incoming_rows: usize, end: u64, first_position: u64, exact_local_window: bool) !void {
    const required = try add(head.rows, incoming_rows);
    if (head.window == 0 or !exact_local_window) {
        if (required > head.capacity) return error.InvalidAttentionShape;
        head.rows = required;
    } else {
        // Expose every causal row to the following multi-query attention. A
        // later retainLocalWindow call trims to the final W rows. Only rows
        // physically overwritten by this dispatch are dropped here.
        const overwritten = required - @min(required, head.capacity);
        if (head.start >= head.capacity or incoming_rows > head.capacity) return error.InvalidAttentionShape;
        head.start = (head.start + overwritten) % head.capacity;
        head.rows = required - overwritten;
        if (overwritten != 0) head.rollback_floor = end else head.rollback_floor = first_position;
    }
    if (head.window == 0 or !exact_local_window) head.rollback_floor = first_position;
    head.logical_length = end;
}

/// Exact local rings retain only the causal union promised by the current
/// publication. Reject a historical query whose complete local window is no
/// longer resident instead of returning attention over a silently truncated
/// prefix. Global heads and the legacy spare-capacity policy are unchanged.
fn validateRetainedQueryCoverage(head: Head, first_query_position: u64, query_rows: usize, exact_local_window: bool) !void {
    if (!exact_local_window or head.window == 0) return;
    if (query_rows == 0 or head.rows == 0 or head.rows > head.capacity or head.rows > head.logical_length)
        return error.InvalidAttentionShape;
    const query_end = std.math.add(u64, first_query_position, query_rows - 1) catch return error.InvalidAttentionShape;
    if (query_end >= head.logical_length) return error.UnsupportedAttentionMatchRollback;
    const retained_first = head.logical_length - head.rows;
    const required_first = if (first_query_position +| 1 > head.window)
        first_query_position + 1 - head.window
    else
        0;
    if (required_first < retained_first) return error.UnsupportedAttentionMatchRollback;
}

test "context exact local retained coverage rejects historical queries" {
    const head: Head = .{ .window = 512, .rows = 512, .capacity = 512, .logical_length = 768 };
    try validateRetainedQueryCoverage(head, 767, 1, true);
    try std.testing.expectError(error.UnsupportedAttentionMatchRollback, validateRetainedQueryCoverage(head, 511, 1, true));
    try validateRetainedQueryCoverage(head, 0, 1, false);
    try validateRetainedQueryCoverage(.{ .window = 0, .rows = 1, .capacity = 1, .logical_length = 1 }, 0, 1, true);
}

fn headBytes(head: Head) !usize {
    return switch (head.encoding) {
        .exact, .fp8 => bufferBytes(head.capacity, head.dim, head.fp8_rows),
        .rotated4 => rotatedBufferBytes(head.capacity, head.dim, head.compressed_rows, head.span_count),
    };
}
fn validateFp8HeaderBytes(bytes: []const u8, head: Head) !void {
    if (bytes.len < @sizeOf(Fp8Header)) return error.InvalidAttentionShape;
    var stored: Fp8Header = undefined;
    @memcpy(std.mem.asBytes(&stored), bytes[0..@sizeOf(Fp8Header)]);
    if (stored.encoding != @backingInt(StorageEncoding.fp8) or stored.first_row != head.fp8_first_row or stored.rows != head.fp8_rows or
        stored.span_count != 0 or stored.compressed_rows != head.fp8_rows or
        stored.key_payload_offset != @sizeOf(Fp8Header) or stored.key_row_bytes != head.dim or stored.value_row_bytes != head.dim or
        @as(u32, @bitCast(stored.key_scale)) != @as(u32, @bitCast(head.key_scale)) or
        @as(u32, @bitCast(stored.value_scale)) != @as(u32, @bitCast(head.value_scale)))
        return error.InvalidAttentionShape;
}
fn validateRotatedHeaderBytes(bytes: []const u8, head: Head) !void {
    if (bytes.len < @sizeOf(Fp8Header)) return error.InvalidAttentionShape;
    var stored: Fp8Header = undefined;
    @memcpy(std.mem.asBytes(&stored), bytes[0..@sizeOf(Fp8Header)]);
    const payload = std.math.add(usize, @sizeOf(Fp8Header), std.math.mul(usize, head.span_count, @sizeOf(packed_storage.Span)) catch return error.InvalidAttentionShape) catch return error.InvalidAttentionShape;
    if (stored.encoding != @backingInt(StorageEncoding.rotated4) or stored.first_row != 0 or stored.rows != 0 or
        stored.span_count != head.span_count or stored.compressed_rows != head.compressed_rows or
        stored.key_payload_offset != payload or stored.key_row_bytes != rotated.rowBytes(head.dim) or
        stored.value_row_bytes != packed_storage.valueBytes(head.dim) or stored.key_scale != 1 or stored.value_scale != 1 or
        stored.reserved0 != 0 or stored.reserved1 != 0)
        return error.InvalidAttentionShape;
}
fn validateRotatedSpans(bytes: []const u8, head: Head) !void {
    if (!rotated.supported(head.dim) or head.span_count == 0 or head.compressed_rows == 0) return error.InvalidAttentionShape;
    const span_bytes = std.math.mul(usize, head.span_count, @sizeOf(packed_storage.Span)) catch return error.InvalidAttentionShape;
    const end = std.math.add(usize, @sizeOf(Fp8Header), span_bytes) catch return error.InvalidAttentionShape;
    if (bytes.len < end) return error.InvalidAttentionShape;
    const spans = std.mem.bytesAsSlice(packed_storage.Span, @as([]align(@alignOf(packed_storage.Span)) const u8, @alignCast(bytes[@sizeOf(Fp8Header)..end])));
    var previous_end: usize = 0;
    var count: usize = 0;
    for (spans, 0..) |span, index| {
        if (span.rows == 0 or span.first > head.rows or span.rows > head.rows - span.first or
            (index > 0 and span.first <= previous_end) or span.offset != count) return error.InvalidAttentionShape;
        previous_end = @as(usize, span.first) + span.rows;
        count = std.math.add(usize, count, span.rows) catch return error.InvalidAttentionShape;
    }
    if (count != head.compressed_rows) return error.InvalidAttentionShape;
}
fn validateStoredHead(head: Head, logical_length: u64, layer: u32, ordinal: usize, window: u32, max_tokens: u64) !void {
    const maximum_capacity = std.math.add(u64, max_tokens, 512) catch return error.InvalidAttentionShape;
    const maximum_capacity_usize = std.math.cast(usize, maximum_capacity) orelse return error.InvalidAttentionShape;
    if (head.layer != layer or head.head != ordinal or head.window != window or head.logical_length != logical_length or
        head.dim == 0 or head.dim > 1024 or head.rows == 0 or head.rows > head.capacity or
        head.capacity > maximum_capacity_usize or head.start >= head.capacity or head.compacted_until > logical_length or
        head.rollback_floor > logical_length or head.fp8_rows > head.rows or head.compressed_rows > head.rows or
        head.fp8_first_row > head.rows - head.fp8_rows or
        ((head.fp8_rows != 0 or head.compressed_rows != 0) and (head.window != 0 or head.start != 0)) or
        (head.encoding == .exact and (head.fp8_rows != 0 or head.compressed_rows != 0 or head.span_count != 0)) or
        (head.encoding == .fp8 and (head.compressed_rows != head.fp8_rows or head.span_count != 0)) or
        (head.encoding == .rotated4 and (head.fp8_rows != 0 or head.compressed_rows == 0 or head.span_count == 0)) or
        !codec.validScale(head.key_scale) or !codec.validScale(head.value_scale))
        return error.InvalidAttentionShape;
    const exact_rows = std.math.sub(usize, head.capacity, if (head.encoding == .rotated4) head.compressed_rows else head.fp8_rows) catch return error.InvalidAttentionShape;
    const exact_bytes = std.math.mul(usize, std.math.mul(usize, exact_rows, head.dim) catch return error.InvalidAttentionShape, 2) catch return error.InvalidAttentionShape;
    if (exact_bytes == 0) {
        if (head.keys != null or head.values != null) return error.InvalidAttentionShape;
    } else if (head.keys == null or head.values == null or
        head.keys.?.logical_bytes < exact_bytes or head.values.?.logical_bytes < exact_bytes) return error.InvalidAttentionShape;
    const metadata_bytes = std.math.mul(usize, head.capacity, @sizeOf(Metadata)) catch return error.InvalidAttentionShape;
    if (head.metadata == null or head.metadata.?.logical_bytes < metadata_bytes) return error.InvalidAttentionShape;
    if (head.encoding == .exact) {
        if (head.fp8_first_row != 0 or head.key_scale != 1 or head.value_scale != 1 or
            head.encoded_keys != null or head.encoded_values != null) return error.InvalidAttentionShape;
    } else if (head.encoding == .fp8) {
        const encoded_values = std.math.mul(usize, head.fp8_rows, head.dim) catch return error.InvalidAttentionShape;
        const encoded_keys = std.math.add(usize, @sizeOf(Fp8Header), encoded_values) catch return error.InvalidAttentionShape;
        if (head.encoded_keys == null or head.encoded_values == null or
            head.encoded_keys.?.logical_bytes < encoded_keys or head.encoded_values.?.logical_bytes < encoded_values or
            head.encoded_keys.?.bytes.len < encoded_keys or head.encoded_values.?.bytes.len < encoded_values)
            return error.InvalidAttentionShape;
        // The mapped header is consumed directly by Metal. Treat the Head
        // fields as immutable authority and reject any header mutation before
        // a GPU dispatch can observe different bounds or scale bits.
        try validateFp8HeaderBytes(
            head.encoded_keys.?.bytes,
            head,
        );
    } else {
        const key_payload_offset = std.math.add(usize, @sizeOf(Fp8Header), std.math.mul(usize, head.span_count, @sizeOf(packed_storage.Span)) catch return error.InvalidAttentionShape) catch return error.InvalidAttentionShape;
        const encoded_keys = std.math.add(usize, key_payload_offset, std.math.mul(usize, head.compressed_rows, rotated.rowBytes(head.dim)) catch return error.InvalidAttentionShape) catch return error.InvalidAttentionShape;
        const encoded_values = std.math.mul(usize, head.compressed_rows, packed_storage.valueBytes(head.dim)) catch return error.InvalidAttentionShape;
        if (head.encoded_keys == null or head.encoded_values == null or head.encoded_keys.?.logical_bytes < encoded_keys or
            head.encoded_values.?.logical_bytes < encoded_values or head.encoded_keys.?.bytes.len < encoded_keys or
            head.encoded_values.?.bytes.len < encoded_values) return error.InvalidAttentionShape;
        try validateRotatedHeaderBytes(head.encoded_keys.?.bytes, head);
        try validateRotatedSpans(head.encoded_keys.?.bytes, head);
    }
}

test "attention_match Metal deferred forward preserves sticky finite status and cancellation" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    for ([_]bool{ false, true }) |coalesced| for (0..4) |scenario| {
        var runner = try Runner.init(8 * 1024 * 1024, null);
        defer runner.deinit();
        runner.coalesced_frame_enabled = coalesced;
        const head: checkpoint.Head = .{
            .dim = 2,
            .positions = &.{ 0, 1, 2, 3 },
            .coverage_ends = &.{ 1, 2, 3, 4 },
            .keys = &.{ 0, 0, 0, 0, 0, 0, 0, 0 },
            .values = &.{ 1, 2, 1, 2, 1, 2, 1, 2 },
            .bias = &.{ 0, 0, 0, 0 },
        };
        const state: checkpoint.State = .{
            .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
            .logical_length = 4,
            .tokens = &.{ 0, 0, 0, 0 },
            .logits = &.{0},
            .layer_donors = &.{0},
            .donors = &.{.{ .window = 0, .heads = &.{head} }},
        };
        var device = try runner.backend(allocator, .{}).upload(state, 8 * 1024 * 1024, .{});
        defer device.deinit();
        var q1 = try runner.core.buffer(8);
        defer q1.deinit();
        var q2 = try runner.core.buffer(8);
        defer q2.deinit();
        var key = try runner.core.buffer(8);
        defer key.deinit();
        var value = try runner.core.buffer(8);
        defer value.deinit();
        var output1 = try runner.core.buffer(8);
        defer output1.deinit();
        var output2 = try runner.core.buffer(8);
        defer output2.deinit();
        @memset(q1.floats(), 0);
        if (scenario == 1) q1.floats()[0] = std.math.nan(f32);
        @memset(q2.floats(), 0);
        @memset(key.floats(), 0);
        @memcpy(value.floats(), &[_]f32{ 1, 2 });
        @memset(output2.floats(), -99);
        const runtime = runner.core.runtime;
        defer runner.cancelDeferredFrame();
        try runner.beginDeferredFrame(&device, runtime, 1, 4, .{});
        if (scenario == 3) {
            try std.testing.expectError(error.InvalidAttentionShape, runner.finishDeferredFrame(.{}));
            try std.testing.expect(runner.deferred_runtime == null);
            try std.testing.expect(!metal_runtime.hasActiveFrame(runtime));
            continue;
        }
        try std.testing.expectError(error.ContextTransactionBusy, device.synchronize());
        try std.testing.expectError(error.ContextStagingBudgetExceeded, runner.ensureScratch(runner.scratch.?.logical_bytes + 1));
        const request: Request = .{ .heads = try device.layer(0), .queries = q1.ref(), .output = output1.ref(), .query_rows = 1, .query_heads = 1, .first_query_position = 4, .scale = 1 };
        try runner.attendOnFrame(runtime, request, .{ .state = &device, .layer = 0, .keys = key.ref(), .values = value.ref(), .rows = 1, .first_position = 4 }, .{});
        var second = request;
        second.queries = q2.ref();
        second.output = output2.ref();
        try runner.attendOnFrame(runtime, second, null, .{});
        try std.testing.expect(metal_runtime.hasActiveFrame(runtime));
        try std.testing.expect(!metal_runtime.hasSubmittedFrame(runtime));
        if (scenario == 2) {
            const Cancel = struct {
                fn check(_: ?*anyopaque) !void {
                    return error.Cancelled;
                }
            };
            try std.testing.expectError(error.Cancelled, runner.finishDeferredFrame(.{ .check_fn = Cancel.check }));
            try std.testing.expectEqual(@as(f32, -99), output2.floats()[0]);
        } else if (scenario == 1) {
            try std.testing.expectError(error.NonFiniteAttention, runner.finishDeferredFrame(.{}));
        } else {
            try runner.finishDeferredFrame(.{});
            try std.testing.expectApproxEqAbs(@as(f32, 1), output2.floats()[0], 1e-6);
            try std.testing.expectApproxEqAbs(@as(f32, 2), output2.floats()[1], 1e-6);
            if (coalesced) {
                const snapshot = metal_runtime.runtimeMemorySnapshot(runtime);
                try std.testing.expectEqual(@as(u64, 1), snapshot.last_frame_compute_encoder_count);
                try std.testing.expectEqual(@as(u64, 1), snapshot.last_frame_planned_compute_scope_count);
                try std.testing.expect(runner.planned_dispatch_calls > 0);
            }
        }
        try std.testing.expect(runner.deferred_runtime == null);
        try std.testing.expect(!metal_runtime.hasActiveFrame(runtime));
        try std.testing.expect(!metal_runtime.hasSubmittedFrame(runtime));
    };
}

test "context attention_match Metal exact local deferred frame commits restores and poisons" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const initial: usize = 4;
    const head: checkpoint.Head = .{
        .dim = 2,
        .positions = &.{ 0, 1, 2, 3 },
        .coverage_ends = &.{ 1, 2, 3, 4 },
        .keys = &.{ 0, 0, 0, 0, 0, 0, 0, 0 },
        .values = &.{ 1, -1, 2, -2, 3, -3, 4, -4 },
        .bias = &.{ 0, 0, 0, 0 },
    };
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = initial,
        .tokens = &.{ 0, 0, 0, 0 },
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &.{.{ .window = initial, .heads = &.{head} }},
    };
    const CancelAfterPublication = struct {
        device: *DeviceState,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (self.device.logical_length != initial) return error.Cancelled;
        }
    };
    const Scenario = enum { journal_oom, cancel_during_attend, cancel_unsubmitted, early_submit_wait_cancel, commit, submitted_nonfinite };
    for ([_]bool{ false, true }) |coalesced| {
        for ([_]Scenario{ .journal_oom, .cancel_during_attend, .cancel_unsubmitted, .early_submit_wait_cancel, .commit, .submitted_nonfinite }) |scenario| {
            var runner = try Runner.initWithOptions(8 * 1024 * 1024, null, .{ .exact_local_window_capacity = true });
            defer runner.deinit();
            runner.coalesced_frame_enabled = coalesced;
            var device = try runner.backend(allocator, .{}).upload(state, 8 * 1024 * 1024, .{});
            defer device.deinit();
            const before = device.heads[0];
            if (scenario == .journal_oom) {
                var failing = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 0 });
                device.allocator = failing.allocator();
                try std.testing.expectError(error.OutOfMemory, runner.beginDeferredFrame(&device, runner.core.runtime, 1, initial, .{}));
                device.allocator = allocator;
                try std.testing.expectEqual(before.rows, device.heads[0].rows);
                try std.testing.expectEqual(before.start, device.heads[0].start);
                try std.testing.expectEqual(before.logical_length, device.heads[0].logical_length);
                try std.testing.expect(!device.invalidated);
                continue;
            }
            var query = try runner.core.buffer(2 * @sizeOf(f32));
            defer query.deinit();
            var key = try runner.core.buffer(2 * @sizeOf(f32));
            defer key.deinit();
            var value = try runner.core.buffer(2 * @sizeOf(f32));
            defer value.deinit();
            var output = try runner.core.buffer(2 * @sizeOf(f32));
            defer output.deinit();
            @memset(query.floats(), 0);
            @memset(key.floats(), 0);
            @memcpy(value.floats(), &[_]f32{ 5, -5 });
            if (scenario == .submitted_nonfinite) query.floats()[0] = std.math.nan(f32);
            try runner.beginDeferredFrame(&device, runner.core.runtime, 1, initial, .{});
            const request: Request = .{
                .heads = try device.layer(0),
                .queries = query.ref(),
                .output = output.ref(),
                .query_rows = 1,
                .query_heads = 1,
                .first_query_position = initial,
                .scale = 1,
            };
            const write: Runner.Append = .{
                .state = &device,
                .layer = 0,
                .keys = key.ref(),
                .values = value.ref(),
                .rows = 1,
                .first_position = initial,
            };
            if (scenario == .cancel_during_attend) {
                var cancel: CancelAfterPublication = .{ .device = &device };
                try std.testing.expectError(error.Cancelled, runner.attendOnFrame(
                    runner.core.runtime,
                    request,
                    write,
                    .{ .context = &cancel, .check_fn = CancelAfterPublication.check },
                ));
                try std.testing.expect(runner.deferred_runtime == null);
                try std.testing.expectEqual(before.rows, device.heads[0].rows);
                try std.testing.expectEqual(before.start, device.heads[0].start);
                try std.testing.expectEqual(before.logical_length, device.heads[0].logical_length);
                try std.testing.expectEqual(before.rollback_floor, device.heads[0].rollback_floor);
                try std.testing.expect(!device.invalidated);
                continue;
            }
            try runner.attendOnFrame(runner.core.runtime, request, write, .{});
            try std.testing.expect(metal_runtime.hasActiveFrame(runner.core.runtime));
            try std.testing.expect(!metal_runtime.hasSubmittedFrame(runner.core.runtime));
            if (scenario == .early_submit_wait_cancel) {
                try metal_runtime.submitFrame(runner.core.runtime);
                try metal_runtime.waitFrame(runner.core.runtime);
                try std.testing.expect(!metal_runtime.hasActiveFrame(runner.core.runtime));
                try std.testing.expect(!metal_runtime.hasSubmittedFrame(runner.core.runtime));
                runner.cancelDeferredFrame();
                try std.testing.expect(device.invalidated);
                try std.testing.expectError(error.InvalidAttentionMatchCache, device.layer(0));
            } else if (scenario == .cancel_unsubmitted) {
                runner.cancelDeferredFrame();
                try std.testing.expectEqual(before.rows, device.heads[0].rows);
                try std.testing.expectEqual(before.start, device.heads[0].start);
                try std.testing.expectEqual(before.logical_length, device.heads[0].logical_length);
                try std.testing.expectEqual(before.rollback_floor, device.heads[0].rollback_floor);
                try std.testing.expect(!device.invalidated);
            } else if (scenario == .submitted_nonfinite) {
                try std.testing.expectError(error.NonFiniteAttention, runner.finishDeferredFrame(.{}));
                try std.testing.expect(device.invalidated);
                try std.testing.expectError(error.InvalidAttentionMatchCache, device.layer(0));
            } else {
                try runner.finishDeferredFrame(.{});
                if (coalesced) {
                    const frame = metal_runtime.runtimeMemorySnapshot(runner.core.runtime);
                    try std.testing.expectEqual(@as(u64, 1), frame.last_frame_compute_encoder_count);
                    try std.testing.expect(runner.planned_dispatch_calls > 0);
                } else {
                    try std.testing.expectEqual(@as(usize, 0), runner.planned_dispatch_calls);
                }
                const stored = (try device.layer(0))[0];
                try std.testing.expectEqual(@as(usize, initial), stored.capacity);
                try std.testing.expectEqual(@as(usize, initial), stored.rows);
                try std.testing.expectEqual(@as(usize, 1), stored.start);
                try std.testing.expectEqual(@as(u64, initial + 1), stored.logical_length);
                try std.testing.expectEqual(@as(u64, initial + 1), stored.rollback_floor);
                try std.testing.expectEqual(@as(u64, initial + 1), device.logical_length);
                // Query zero gives uniform weights over the causal ring. The
                // overwritten row 0 is absent and rows 1,2,3,new4 average to 3.5.
                try std.testing.expectApproxEqAbs(@as(f32, 3.5), output.floats()[0], 1e-6);
                try std.testing.expectApproxEqAbs(@as(f32, -3.5), output.floats()[1], 1e-6);
                // The download template is validated as a real local checkpoint.
                // After appending position 4, a window-4 donor covers the valid
                // logical tail 1...4 rather than the pre-append 0...3 state.
                const snapshot_head: checkpoint.Head = .{
                    .dim = 2,
                    .positions = &.{ 1, 2, 3, 4 },
                    .coverage_ends = &.{ 2, 3, 4, 5 },
                    .keys = &.{ 0, 0, 0, 0, 0, 0, 0, 0 },
                    .values = &.{ 2, -2, 3, -3, 4, -4, 5, -5 },
                    .bias = &.{ 0, 0, 0, 0 },
                };
                const snapshot_state: checkpoint.State = .{
                    .identity = state.identity,
                    .logical_length = initial + 1,
                    .tokens = &.{ 0, 0, 0, 0, 0 },
                    .logits = &.{0},
                    .layer_donors = &.{0},
                    .donors = &.{.{ .window = initial, .heads = &.{snapshot_head} }},
                };
                var snapshot = try device.download(allocator, snapshot_state, .{}, 8 * 1024 * 1024, .{});
                defer snapshot.deinit();
                const restored = snapshot.state.donors[0].heads[0];
                try std.testing.expectEqualSlices(u64, &.{ 1, 2, 3, 4 }, restored.positions);
                try std.testing.expectEqualSlices(f32, &.{ 2, -2, 3, -3, 4, -4, 5, -5 }, restored.values);
            }
        }
    }
    const CancelAfterSubmit = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            // append start, pre-batch, per-head dispatch, post-submit.
            if (self.calls == 4) return error.Cancelled;
        }
    };
    for ([_]bool{ false, true }) |nonfinite| {
        var runner = try Runner.initWithOptions(8 * 1024 * 1024, null, .{ .exact_local_window_capacity = true });
        defer runner.deinit();
        var device = try runner.backend(allocator, .{}).upload(state, 8 * 1024 * 1024, .{});
        defer device.deinit();
        var key = try runner.core.buffer(2 * @sizeOf(f32));
        defer key.deinit();
        var value = try runner.core.buffer(2 * @sizeOf(f32));
        defer value.deinit();
        @memset(key.floats(), 0);
        @memcpy(value.floats(), &[_]f32{ 5, -5 });
        if (nonfinite) {
            key.floats()[0] = std.math.inf(f32);
            try std.testing.expectError(error.NonFiniteAttention, runner.append(&device, 0, key.ref(), value.ref(), 1, initial, .{}));
        } else {
            var cancel: CancelAfterSubmit = .{};
            try std.testing.expectError(error.Cancelled, runner.append(
                &device,
                0,
                key.ref(),
                value.ref(),
                1,
                initial,
                .{ .context = &cancel, .check_fn = CancelAfterSubmit.check },
            ));
        }
        try std.testing.expect(device.invalidated);
        try std.testing.expectError(error.InvalidAttentionMatchCache, device.layer(0));
    }
    for ([_]bool{ true, false }) |oom| {
        var runner = try Runner.initWithOptions(8 * 1024 * 1024, null, .{ .exact_local_window_capacity = true });
        defer runner.deinit();
        var device = try runner.backend(allocator, .{}).upload(state, 8 * 1024 * 1024, .{});
        defer device.deinit();
        var keys = try runner.core.buffer(2 * 2 * @sizeOf(f32));
        defer keys.deinit();
        var values = try runner.core.buffer(2 * 2 * @sizeOf(f32));
        defer values.deinit();
        @memset(keys.floats(), 0);
        @memcpy(values.floats(), &[_]f32{ 5, -5, 6, -6 });
        // Build the five-row causal-union publication through the real append
        // path. Such a transient is valid during a two-row prefill, but cannot
        // be uploaded as a portable local checkpoint (which must contain W).
        try runner.append(&device, 0, keys.ref(), values.ref(), 2, initial, .{});
        try std.testing.expectEqual(@as(usize, 5), device.heads[0].rows);
        try std.testing.expectEqual(@as(u64, 6), device.logical_length);
        const before = device.heads[0];
        if (oom) {
            var failing = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 0 });
            device.allocator = failing.allocator();
            try std.testing.expectError(error.OutOfMemory, runner.beginDeferredFrame(&device, runner.core.runtime, 1, 6, .{}));
            device.allocator = allocator;
        } else {
            try runner.beginDeferredFrame(&device, runner.core.runtime, 1, 6, .{});
            // begin trims the old view to W, but cancellation must restore the
            // complete pre-frame publication, not the post-trim snapshot.
            try std.testing.expectEqual(@as(usize, initial), device.heads[0].rows);
            runner.cancelDeferredFrame();
        }
        try std.testing.expectEqual(before.rows, device.heads[0].rows);
        try std.testing.expectEqual(before.start, device.heads[0].start);
        try std.testing.expectEqual(before.logical_length, device.heads[0].logical_length);
        try std.testing.expectEqual(before.rollback_floor, device.heads[0].rollback_floor);
        try std.testing.expect(!device.invalidated);
    }

    // Execute the same singleton decode through legacy spare capacity and the
    // exact ring. The Metal result and chronological snapshot agree; only the
    // tracked physical storage differs. Root must execute this gated fixture
    // before treating the estimate as measured evidence.
    var outputs: [3][2]f32 = undefined;
    var physical: [3]usize = undefined;
    for (0..3) |arm| {
        const exact = arm != 0;
        var runner = try Runner.initWithOptions(8 * 1024 * 1024, null, .{ .exact_local_window_capacity = exact });
        defer runner.deinit();
        runner.coalesced_frame_enabled = arm == 2;
        var device = try runner.backend(allocator, .{}).upload(state, 8 * 1024 * 1024, .{});
        defer device.deinit();
        var query = try runner.core.buffer(2 * @sizeOf(f32));
        defer query.deinit();
        var key = try runner.core.buffer(2 * @sizeOf(f32));
        defer key.deinit();
        var value = try runner.core.buffer(2 * @sizeOf(f32));
        defer value.deinit();
        var output = try runner.core.buffer(2 * @sizeOf(f32));
        defer output.deinit();
        @memset(query.floats(), 0);
        @memset(key.floats(), 0);
        @memcpy(value.floats(), &[_]f32{ 5, -5 });
        try runner.beginDeferredFrame(&device, runner.core.runtime, 1, initial, .{});
        try runner.attendOnFrame(runner.core.runtime, .{
            .heads = try device.layer(0),
            .queries = query.ref(),
            .output = output.ref(),
            .query_rows = 1,
            .query_heads = 1,
            .first_query_position = initial,
            .scale = 1,
        }, .{ .state = &device, .layer = 0, .keys = key.ref(), .values = value.ref(), .rows = 1, .first_position = initial }, .{});
        try std.testing.expect(!metal_runtime.hasSubmittedFrame(runner.core.runtime));
        try runner.finishDeferredFrame(.{});
        @memcpy(outputs[arm][0..], output.floats()[0..2]);
        physical[arm] = device.allocated_bytes;
        const stored = (try device.layer(0))[0];
        try std.testing.expectEqual(@as(u64, initial + 1), stored.logical_length);
        if (exact) try std.testing.expectEqual(initial, stored.rows) else try std.testing.expectEqual(initial + 1, stored.rows);
    }
    try std.testing.expectEqualSlices(f32, &outputs[0], &outputs[1]);
    try std.testing.expectEqualSlices(f32, &outputs[0], &outputs[2]);
    try std.testing.expectEqual(physical[1], physical[2]);
    // At D2/W4 both policies round to the same 16 KiB Metal pages.
    try std.testing.expectEqual(physical[0], physical[1]);

    // Production-shaped D256/W512 crosses backing-page boundaries. Exercise
    // the same real append/attention/download path and compare tracked bytes.
    const pd: usize = 256;
    const pw: usize = 512;
    const elements = pd * pw;
    const pkeys = try allocator.alloc(f32, elements);
    defer allocator.free(pkeys);
    const pvalues = try allocator.alloc(f32, elements);
    defer allocator.free(pvalues);
    const ppositions = try allocator.alloc(u64, pw);
    defer allocator.free(ppositions);
    const pends = try allocator.alloc(u64, pw);
    defer allocator.free(pends);
    const pbias = try allocator.alloc(f32, pw);
    defer allocator.free(pbias);
    const ptokens = try allocator.alloc(u32, pw);
    defer allocator.free(ptokens);
    @memset(pkeys, 0);
    @memset(pvalues, 1);
    @memset(pbias, 0);
    @memset(ptokens, 0);
    for (ppositions, pends, 0..) |*position, *end, i| {
        position.* = i;
        end.* = i + 1;
    }
    const phead: checkpoint.Head = .{ .dim = pd, .positions = ppositions, .coverage_ends = pends, .keys = pkeys, .values = pvalues, .bias = pbias };
    const pstate: checkpoint.State = .{ .identity = state.identity, .logical_length = pw, .tokens = ptokens, .logits = &.{0}, .layer_donors = &.{0}, .donors = &.{.{ .window = pw, .heads = &.{phead} }} };
    var production_outputs: [3][]f32 = undefined;
    var production_physical: [3]usize = undefined;
    for (0..3) |arm| {
        const exact = arm != 0;
        var runner = try Runner.initWithOptions(64 * 1024 * 1024, null, .{ .exact_local_window_capacity = exact });
        defer runner.deinit();
        runner.coalesced_frame_enabled = arm == 2;
        var device = try runner.backend(allocator, .{}).upload(pstate, 64 * 1024 * 1024, .{});
        defer device.deinit();
        var query = try runner.core.buffer(pd * @sizeOf(f32));
        defer query.deinit();
        var key = try runner.core.buffer(pd * @sizeOf(f32));
        defer key.deinit();
        var value = try runner.core.buffer(pd * @sizeOf(f32));
        defer value.deinit();
        var output = try runner.core.buffer(pd * @sizeOf(f32));
        defer output.deinit();
        @memset(query.floats(), 0);
        @memset(key.floats(), 0);
        @memset(value.floats(), 2);
        try runner.beginDeferredFrame(&device, runner.core.runtime, 1, pw, .{});
        try runner.attendOnFrame(runner.core.runtime, .{ .heads = try device.layer(0), .queries = query.ref(), .output = output.ref(), .query_rows = 1, .query_heads = 1, .first_query_position = pw, .scale = 1 }, .{ .state = &device, .layer = 0, .keys = key.ref(), .values = value.ref(), .rows = 1, .first_position = pw }, .{});
        try runner.finishDeferredFrame(.{});
        production_outputs[arm] = try allocator.dupe(f32, output.floats());
        production_physical[arm] = device.allocated_bytes;
        if (exact) {
            const tail_positions = try allocator.alloc(u64, pw);
            defer allocator.free(tail_positions);
            const tail_ends = try allocator.alloc(u64, pw);
            defer allocator.free(tail_ends);
            const tail_tokens = try allocator.alloc(u32, pw + 1);
            defer allocator.free(tail_tokens);
            @memset(tail_tokens, 0);
            for (tail_positions, tail_ends, 0..) |*position, *end, i| {
                position.* = i + 1;
                end.* = i + 2;
            }
            const tail_head: checkpoint.Head = .{ .dim = pd, .positions = tail_positions, .coverage_ends = tail_ends, .keys = pkeys, .values = pvalues, .bias = pbias };
            const tail_state: checkpoint.State = .{ .identity = state.identity, .logical_length = pw + 1, .tokens = tail_tokens, .logits = &.{0}, .layer_donors = &.{0}, .donors = &.{.{ .window = pw, .heads = &.{tail_head} }} };
            var downloaded = try device.download(allocator, tail_state, .{}, 64 * 1024 * 1024, .{});
            defer downloaded.deinit();
            const restored = downloaded.state.donors[0].heads[0];
            try std.testing.expectEqual(@as(u64, 1), restored.positions[0]);
            try std.testing.expectEqual(@as(u64, pw), restored.positions[pw - 1]);
        }
    }
    defer allocator.free(production_outputs[0]);
    defer allocator.free(production_outputs[1]);
    defer allocator.free(production_outputs[2]);
    try std.testing.expectEqualSlices(f32, production_outputs[0], production_outputs[1]);
    try std.testing.expectEqualSlices(f32, production_outputs[0], production_outputs[2]);
    try std.testing.expect(production_physical[1] < production_physical[0]);
    try std.testing.expectEqual(production_physical[1], production_physical[2]);
}

test "attention_match Metal FP8 header bytes match immutable head metadata" {
    const head: Head = .{
        .encoding = .fp8,
        .fp8_first_row = 17,
        .fp8_rows = 23,
        .key_scale = 0.5,
        .value_scale = 4,
    };
    var stored: Fp8Header = .{
        .encoding = @backingInt(StorageEncoding.fp8),
        .first_row = head.fp8_first_row,
        .rows = head.fp8_rows,
        .compressed_rows = head.fp8_rows,
        .key_row_bytes = head.dim,
        .value_row_bytes = head.dim,
        .key_scale = head.key_scale,
        .value_scale = head.value_scale,
    };
    try validateFp8HeaderBytes(std.mem.asBytes(&stored), head);

    const original = stored;
    stored.first_row += 1;
    try std.testing.expectError(error.InvalidAttentionShape, validateFp8HeaderBytes(std.mem.asBytes(&stored), head));
    stored = original;
    stored.rows -= 1;
    try std.testing.expectError(error.InvalidAttentionShape, validateFp8HeaderBytes(std.mem.asBytes(&stored), head));
    stored = original;
    stored.key_scale = @bitCast(@as(u32, @bitCast(stored.key_scale)) ^ 1);
    try std.testing.expectError(error.InvalidAttentionShape, validateFp8HeaderBytes(std.mem.asBytes(&stored), head));
    stored = original;
    stored.value_scale = @bitCast(@as(u32, @bitCast(stored.value_scale)) ^ 1);
    try std.testing.expectError(error.InvalidAttentionShape, validateFp8HeaderBytes(std.mem.asBytes(&stored), head));
    try std.testing.expectError(error.InvalidAttentionShape, validateFp8HeaderBytes(std.mem.asBytes(&stored)[0..15], head));
}

pub const DownloadedHead = struct {
    allocator: std.mem.Allocator,
    value: checkpoint.Head,

    pub fn head(self: *const @This()) checkpoint.Head {
        return self.value;
    }

    pub fn liveBytes(self: @This()) !usize {
        var bytes = try mul(try add(self.value.positions.len, self.value.coverage_ends.len), @sizeOf(u64));
        bytes = try add(bytes, try mul(try add(self.value.keys.len, try add(self.value.values.len, self.value.bias.len)), @sizeOf(f32)));
        bytes = try add(bytes, try mul(self.value.compressed.spans.len, @sizeOf(packed_storage.Span)));
        return add(bytes, try add(self.value.compressed.keys.len, self.value.compressed.values.len));
    }

    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.value.compressed.values);
        self.allocator.free(self.value.compressed.keys);
        self.allocator.free(self.value.compressed.spans);
        self.allocator.free(self.value.values);
        self.allocator.free(self.value.keys);
        self.allocator.free(self.value.bias);
        self.allocator.free(self.value.coverage_ends);
        self.allocator.free(self.value.positions);
        self.* = undefined;
    }
};

test "attention_match Metal FP8 storage rejects a rotated ring origin" {
    // FP8 payload rows are indexed by their absolute physical row. Unlike the
    // exact local-window representation, they cannot be interpreted through a
    // rotated ring origin without moving the compressed interval and header.
    const head: Head = .{
        .encoding = .fp8,
        .dim = 1,
        .rows = 1,
        .capacity = 2,
        .start = 1,
        .logical_length = 2,
        .fp8_rows = 1,
    };
    try std.testing.expectError(
        error.InvalidAttentionShape,
        validateStoredHead(head, 2, 0, 0, 0, checkpoint.max_context_tokens),
    );
}
fn ref(buffer: ?Buffer) !Ref {
    return if (buffer) |value| value.ref() else error.InvalidAttentionShape;
}
fn refOr(buffer: ?Buffer, fallback: Buffer) Ref {
    return if (buffer) |value| value.ref() else fallback.ref();
}
fn overlaps(a: Ref, b: Ref) bool {
    if (a.handle != b.handle) return false;
    const a_end = std.math.add(usize, a.offset, a.bytes) catch return true;
    const b_end = std.math.add(usize, b.offset, b.bytes) catch return true;
    return a.offset < b_end and b.offset < a_end;
}
fn ringCopy(destination: Buffer, source: Buffer, head: Head, width: usize) !void {
    if (destination.logical_bytes < try mul(head.rows, width) or source.logical_bytes < try mul(head.capacity, width))
        return error.InvalidAttentionShape;
    const first = @min(head.rows, head.capacity - head.start);
    @memcpy(destination.bytes[0 .. first * width], source.bytes[head.start * width ..][0 .. first * width]);
    if (first < head.rows)
        @memcpy(destination.bytes[first * width ..][0 .. (head.rows - first) * width], source.bytes[0 .. (head.rows - first) * width]);
}

pub const DeviceState = struct {
    allocator: std.mem.Allocator,
    runner: *Runner,
    heads: []Head = &.{},
    logical_length: u64,
    initial_logical_length: u64,
    allocated_bytes: usize = 0,
    invalidated: bool = false,

    pub fn structuralInventory(self: *DeviceState, allocator: std.mem.Allocator, control: Control) !InventoryOwned {
        try control.check();
        if (self.invalidated or self.logical_length == 0 or self.heads.len == 0) return error.InvalidAttentionMatchCache;
        try self.synchronize();
        const entries = try allocator.alloc(InventoryEntry, self.heads.len);
        errdefer allocator.free(entries);
        var total: usize = 0;
        var global: usize = 0;
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update("antfly.metal-attention-match-structural-inventory.v1\x00");
        hash.update(std.mem.asBytes(&self.logical_length));
        for (self.heads, entries) |head, *entry| {
            try control.check();
            try validateStoredHead(head, self.logical_length, head.layer, head.head, head.window, checkpoint.max_context_tokens);
            const physical = try headPhysicalBytes(head);
            if (physical == 0) return error.ContextPhysicalAccountingUnavailable;
            entry.* = .{ .layer = head.layer, .head = head.head, .window = head.window, .dim = head.dim, .rows = head.rows, .capacity = head.capacity, .start = head.start, .encoding = head.encoding, .physical_bytes = physical };
            total = try add(total, physical);
            if (head.window == 0) global = try add(global, physical);
            hash.update(std.mem.asBytes(&entry.layer));
            hash.update(std.mem.asBytes(&entry.head));
            hash.update(std.mem.asBytes(&entry.window));
            hash.update(std.mem.asBytes(&entry.dim));
            hash.update(std.mem.asBytes(&entry.rows));
            hash.update(std.mem.asBytes(&entry.capacity));
            hash.update(std.mem.asBytes(&entry.start));
            hash.update(std.mem.asBytes(&entry.encoding));
            hash.update(std.mem.asBytes(&entry.physical_bytes));
        }
        try control.check();
        return .{ .allocator = allocator, .logical_length = self.logical_length, .entries = entries, .physical_bytes = total, .global_physical_bytes = global, .digest = hash.finalResult() };
    }

    pub fn layer(self: *DeviceState, layer_index: u32) ![]Head {
        if (self.invalidated) return error.InvalidAttentionMatchCache;
        for (self.heads, 0..) |head, start| {
            if (head.layer != layer_index) continue;
            var end = start + 1;
            while (end < self.heads.len and self.heads[end].layer == layer_index) : (end += 1) {}
            return self.heads[start..end];
        }
        return error.UnsupportedKvSnapshotLayout;
    }

    pub fn clone(self: *DeviceState, allocator: std.mem.Allocator, max_bytes: usize, control: Control) !DeviceState {
        return self.cloneInto(allocator, self.runner, max_bytes, control);
    }

    /// Clone into a destination owner's runner/tracker. Cross-request clones
    /// must not retain the source runner because its storage owner may retire
    /// independently of the destination cache.
    pub fn cloneInto(self: *DeviceState, allocator: std.mem.Allocator, runner: *Runner, max_bytes: usize, control: Control) !DeviceState {
        try control.check();
        try self.synchronize();
        var bytes: usize = 0;
        for (self.heads) |head| {
            try validateStoredHead(head, self.logical_length, head.layer, head.head, head.window, checkpoint.max_context_tokens);
            bytes = try add(bytes, try headBytes(.{ .capacity = head.rows, .dim = head.dim, .encoding = head.encoding, .compressed_rows = head.compressed_rows, .span_count = head.span_count, .fp8_rows = head.fp8_rows }));
        }
        try runner.admit(bytes, max_bytes);
        var result: DeviceState = .{ .allocator = allocator, .runner = runner, .logical_length = self.logical_length, .initial_logical_length = self.initial_logical_length };
        errdefer result.deinit();
        result.heads = try allocator.alloc(Head, self.heads.len);
        @memset(result.heads, .{});
        for (self.heads, result.heads) |source, *destination| {
            try control.check();
            destination.* = .{ .layer = source.layer, .head = source.head, .window = source.window, .dim = source.dim, .rows = source.rows, .capacity = source.rows, .logical_length = source.logical_length, .compacted_until = source.compacted_until, .rollback_floor = source.rollback_floor, .encoding = source.encoding, .compressed_rows = source.compressed_rows, .span_count = source.span_count, .fp8_first_row = source.fp8_first_row, .fp8_rows = source.fp8_rows, .key_scale = source.key_scale, .value_scale = source.value_scale };
            try runner.allocateHead(destination, source.rows);
            if (source.encoding == .exact) {
                try ringCopy(destination.keys.?, source.keys.?, source, source.dim * 2);
                try ringCopy(destination.values.?, source.values.?, source, source.dim * 2);
            } else {
                if (source.start != 0 or source.window != 0) return error.InvalidAttentionShape;
                if (source.keys) |buffer| {
                    const count = destination.keys.?.logical_bytes;
                    @memcpy(destination.keys.?.bytes[0..count], buffer.bytes[0..count]);
                }
                if (source.values) |buffer| {
                    const count = destination.values.?.logical_bytes;
                    @memcpy(destination.values.?.bytes[0..count], buffer.bytes[0..count]);
                }
                @memcpy(destination.encoded_keys.?.bytes[0..destination.encoded_keys.?.logical_bytes], source.encoded_keys.?.bytes[0..destination.encoded_keys.?.logical_bytes]);
                @memcpy(destination.encoded_values.?.bytes[0..destination.encoded_values.?.logical_bytes], source.encoded_values.?.bytes[0..destination.encoded_values.?.logical_bytes]);
            }
            try ringCopy(destination.metadata.?, source.metadata.?, source, @sizeOf(Metadata));
        }
        result.allocated_bytes = bytes;
        try control.check();
        return result;
    }

    pub fn retainLocalWindow(self: *DeviceState, layer_index: u32) !void {
        if (self.invalidated) return error.InvalidAttentionMatchCache;
        if (self.runner.deferred_runtime == null) try self.synchronize();
        for (try self.layer(layer_index)) |*head| if (head.window != 0 and head.rows > head.window) {
            const drop = head.rows - head.window;
            head.start = (head.start + drop) % head.capacity;
            head.rows -= drop;
            head.rollback_floor = head.logical_length;
        };
    }

    pub fn truncate(self: *DeviceState, retained: u64, control: Control) !void {
        try control.check();
        if (self.invalidated) return error.InvalidAttentionMatchCache;
        if (retained < self.initial_logical_length or retained > self.logical_length or self.heads.len == 0)
            return error.UnsupportedAttentionMatchRollback;
        const removed: usize = @intCast(self.logical_length - retained);
        if (removed == 0) return;
        for (self.heads) |head| if (head.logical_length != self.logical_length or retained < head.rollback_floor or
            head.rows <= removed or head.rows > head.capacity or head.start >= head.capacity or
            (head.window > 0 and head.rows - removed < @min(head.window, retained))) return error.UnsupportedAttentionMatchRollback;
        try self.synchronize();
        try control.check();
        for (self.heads) |*head| {
            head.rows -= removed;
            head.logical_length = retained;
        }
        self.logical_length = retained;
    }

    pub fn download(self: *DeviceState, allocator: std.mem.Allocator, metadata: checkpoint.State, limits: checkpoint.Limits, max_host_bytes: usize, control: Control) !checkpoint.Owned {
        try control.check();
        if (self.invalidated) return error.InvalidAttentionMatchCache;
        if (metadata.logical_length != self.logical_length or metadata.logical_length == 0 or metadata.logical_length > limits.max_tokens or
            metadata.tokens.len != self.logical_length or metadata.logits.len == 0 or metadata.logits.len > limits.max_vocab or
            metadata.layer_donors.len == 0 or metadata.layer_donors.len > limits.max_layers or metadata.donors.len == 0 or
            metadata.donors.len > metadata.layer_donors.len) return error.InvalidKvCheckpoint;
        for (metadata.tokens, 0..) |token, i| {
            if (i % 4096 == 0) try control.check();
            if (token >= metadata.logits.len) return error.InvalidKvCheckpoint;
        }
        for (metadata.logits, 0..) |value, i| {
            if (i % 4096 == 0) try control.check();
            if (!std.math.isFinite(value)) return error.InvalidKvCheckpoint;
        }
        for (metadata.layer_donors, 0..) |index, consumer_layer| {
            if (index >= metadata.donors.len or metadata.donors[index].layer > consumer_layer) return error.InvalidKvCheckpoint;
        }
        try self.synchronize();
        var required = try mul(try add(try add(metadata.tokens.len, metadata.logits.len), metadata.layer_donors.len), 4);
        required = try add(required, try mul(metadata.donors.len, @sizeOf(checkpoint.Donor)));
        var count: usize = 0;
        for (metadata.donors, 0..) |donor, donor_index| {
            if (donor.layer >= metadata.layer_donors.len or metadata.layer_donors[donor.layer] != donor_index or
                (donor_index > 0 and donor.layer <= metadata.donors[donor_index - 1].layer) or donor.window > limits.max_tokens)
                return error.InvalidKvCheckpoint;
            const heads = try self.layer(donor.layer);
            if (heads.len != donor.heads.len or heads.len == 0 or heads.len > limits.max_kv_heads) return error.InvalidKvCheckpoint;
            count = try add(count, heads.len);
            required = try add(required, try mul(heads.len, @sizeOf(checkpoint.Head)));
            for (heads, 0..) |head, head_index| {
                validateStoredHead(head, self.logical_length, donor.layer, head_index, donor.window, limits.max_tokens) catch return error.InvalidKvCheckpoint;
                if (head.dim > limits.max_head_dim) return error.InvalidKvCheckpoint;
                required = try add(required, try mul(head.rows, try add(20, try mul(head.dim, 8))));
                if (head.encoding == .rotated4) {
                    required = try add(required, try mul(head.span_count, @sizeOf(packed_storage.Span)));
                    required = try add(required, try mul(head.compressed_rows, rotated.rowBytes(head.dim)));
                    required = try add(required, try mul(head.compressed_rows, packed_storage.valueBytes(head.dim)));
                }
            }
        }
        if (count != self.heads.len or required > limits.max_state_bytes) return error.KvCheckpointLimit;
        if (required > max_host_bytes) return error.CompactionWorkspaceLimit;
        var result = try checkpoint.cloneMetadata(allocator, metadata);
        errdefer result.deinit();
        for (result.state.donors) |donor| for (try self.layer(donor.layer), @constCast(donor.heads)) |stored, *head| {
            try control.check();
            var source = stored;
            if (source.window != 0 and source.rows > source.window) {
                source.start = (source.start + source.rows - source.window) % source.capacity;
                source.rows = source.window;
            }
            head.dim = source.dim;
            head.compacted_until = source.compacted_until;
            head.fp8 = .{ .first_row = source.fp8_first_row, .rows = source.fp8_rows, .key_scale = source.key_scale, .value_scale = source.value_scale };
            head.positions = try allocator.alloc(u64, source.rows);
            head.coverage_ends = try allocator.alloc(u64, source.rows);
            head.bias = try allocator.alloc(f32, source.rows);
            head.keys = try allocator.alloc(f32, try mul(source.rows, source.dim));
            head.values = try allocator.alloc(f32, head.keys.len);
            try readMetadata(source.metadata.?, source, @constCast(head.bias), @constCast(head.positions), @constCast(head.coverage_ends));
            if (source.encoding == .rotated4) {
                const encoded_keys = source.encoded_keys.?.bytes[0..source.encoded_keys.?.logical_bytes];
                const encoded_values = source.encoded_values.?.bytes[0..source.encoded_values.?.logical_bytes];
                const span_bytes = try mul(source.span_count, @sizeOf(packed_storage.Span));
                const payload_offset = try add(@sizeOf(Fp8Header), span_bytes);
                head.compressed.version = rotated.version;
                head.compressed.spans = try allocator.dupe(packed_storage.Span, std.mem.bytesAsSlice(packed_storage.Span, @as([]align(@alignOf(packed_storage.Span)) const u8, @alignCast(encoded_keys[@sizeOf(Fp8Header)..payload_offset]))));
                head.compressed.keys = try allocator.dupe(u8, encoded_keys[payload_offset..]);
                head.compressed.values = try allocator.dupe(u8, encoded_values);
                try head.compressed.validate(source.rows, source.dim);
                const exact_keys = if (source.keys) |buffer| std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(buffer.bytes[0..buffer.logical_bytes]))) else &.{};
                const exact_values = if (source.values) |buffer| std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(buffer.bytes[0..buffer.logical_bytes]))) else &.{};
                var decoded_keys: [512]f32 = undefined;
                var decoded_values: [512]f32 = undefined;
                for (0..source.rows) |row| {
                    if (row % 16 == 0) try control.check();
                    switch (head.compressed.index(row)) {
                        .compressed => |packed_row| {
                            try head.compressed.decodeRow(source.dim, packed_row, decoded_keys[0..source.dim], decoded_values[0..source.dim]);
                            @memcpy(@constCast(head.keys[row * source.dim ..][0..source.dim]), decoded_keys[0..source.dim]);
                            @memcpy(@constCast(head.values[row * source.dim ..][0..source.dim]), decoded_values[0..source.dim]);
                        },
                        .exact => |exact_row| for (@constCast(head.keys[row * source.dim ..][0..source.dim]), @constCast(head.values[row * source.dim ..][0..source.dim]), exact_keys[exact_row * source.dim ..][0..source.dim], exact_values[exact_row * source.dim ..][0..source.dim]) |*k, *v, sk, sv| {
                            k.* = sk;
                            v.* = sv;
                        },
                    }
                }
            } else {
                try readStoredValues(source.keys, source.encoded_keys, source, @constCast(head.keys), source.key_scale, true);
                try readStoredValues(source.values, source.encoded_values, source, @constCast(head.values), source.value_scale, false);
            }
        };
        try result.state.validateWithControl(limits, control);
        return result;
    }

    /// Download one authenticated source head without cloning checkpoint
    /// metadata or materializing any other donor. The decode is identical to
    /// the whole-checkpoint path and preserves ring order and packed bits.
    pub fn downloadHead(self: *DeviceState, allocator: std.mem.Allocator, donor_layer: u32, head_index: usize, window: u32, max_host_bytes: usize, control: Control) !DownloadedHead {
        try control.check();
        if (self.invalidated or self.logical_length == 0) return error.InvalidAttentionMatchCache;
        try self.synchronize();
        const layer_heads = try self.layer(donor_layer);
        if (head_index >= layer_heads.len) return error.InvalidKvCheckpoint;
        var source = layer_heads[head_index];
        validateStoredHead(source, self.logical_length, donor_layer, head_index, window, checkpoint.max_context_tokens) catch return error.InvalidKvCheckpoint;
        if (source.dim == 0 or source.dim > 512) return error.InvalidKvCheckpoint;
        if (source.window != 0 and source.rows > source.window) {
            source.start = (source.start + source.rows - source.window) % source.capacity;
            source.rows = source.window;
        }
        var required = try mul(source.rows, try add(20, try mul(source.dim, 8)));
        if (source.encoding == .rotated4) {
            required = try add(required, try mul(source.span_count, @sizeOf(packed_storage.Span)));
            required = try add(required, try mul(source.compressed_rows, rotated.rowBytes(source.dim)));
            required = try add(required, try mul(source.compressed_rows, packed_storage.valueBytes(source.dim)));
        }
        if (required > max_host_bytes) return error.CompactionWorkspaceLimit;
        const positions = try allocator.alloc(u64, source.rows);
        errdefer allocator.free(positions);
        const coverage = try allocator.alloc(u64, source.rows);
        errdefer allocator.free(coverage);
        const bias = try allocator.alloc(f32, source.rows);
        errdefer allocator.free(bias);
        const keys = try allocator.alloc(f32, try mul(source.rows, source.dim));
        errdefer allocator.free(keys);
        const values = try allocator.alloc(f32, keys.len);
        errdefer allocator.free(values);
        try control.check();
        try readMetadata(source.metadata.?, source, bias, positions, coverage);
        var spans: []packed_storage.Span = try allocator.alloc(packed_storage.Span, 0);
        errdefer allocator.free(spans);
        var compressed_keys: []u8 = try allocator.alloc(u8, 0);
        errdefer allocator.free(compressed_keys);
        var compressed_values: []u8 = try allocator.alloc(u8, 0);
        errdefer allocator.free(compressed_values);
        var compressed_version: u32 = 0;
        if (source.encoding == .rotated4) {
            const encoded_keys = source.encoded_keys.?.bytes[0..source.encoded_keys.?.logical_bytes];
            const encoded_values = source.encoded_values.?.bytes[0..source.encoded_values.?.logical_bytes];
            const span_bytes = try mul(source.span_count, @sizeOf(packed_storage.Span));
            const payload_offset = try add(@sizeOf(Fp8Header), span_bytes);
            allocator.free(spans);
            spans = try allocator.dupe(packed_storage.Span, std.mem.bytesAsSlice(packed_storage.Span, @as([]align(@alignOf(packed_storage.Span)) const u8, @alignCast(encoded_keys[@sizeOf(Fp8Header)..payload_offset]))));
            allocator.free(compressed_keys);
            compressed_keys = try allocator.dupe(u8, encoded_keys[payload_offset..]);
            allocator.free(compressed_values);
            compressed_values = try allocator.dupe(u8, encoded_values);
            compressed_version = rotated.version;
            const packed_payload: packed_storage.Packed = .{ .version = compressed_version, .spans = spans, .keys = compressed_keys, .values = compressed_values };
            try packed_payload.validate(source.rows, source.dim);
            const exact_keys = if (source.keys) |buffer| std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(buffer.bytes[0..buffer.logical_bytes]))) else &.{};
            const exact_values = if (source.values) |buffer| std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(buffer.bytes[0..buffer.logical_bytes]))) else &.{};
            var decoded_keys: [512]f32 = undefined;
            var decoded_values: [512]f32 = undefined;
            for (0..source.rows) |row| {
                if (row % 16 == 0) try control.check();
                switch (packed_payload.index(row)) {
                    .compressed => |packed_row| {
                        try packed_payload.decodeRow(source.dim, packed_row, decoded_keys[0..source.dim], decoded_values[0..source.dim]);
                        @memcpy(keys[row * source.dim ..][0..source.dim], decoded_keys[0..source.dim]);
                        @memcpy(values[row * source.dim ..][0..source.dim], decoded_values[0..source.dim]);
                    },
                    .exact => |exact_row| for (keys[row * source.dim ..][0..source.dim], values[row * source.dim ..][0..source.dim], exact_keys[exact_row * source.dim ..][0..source.dim], exact_values[exact_row * source.dim ..][0..source.dim]) |*k, *v, sk, sv| {
                        k.* = sk;
                        v.* = sv;
                    },
                }
            }
        } else {
            try readStoredValues(source.keys, source.encoded_keys, source, keys, source.key_scale, true);
            try readStoredValues(source.values, source.encoded_values, source, values, source.value_scale, false);
        }
        try control.check();
        return .{ .allocator = allocator, .value = .{ .dim = source.dim, .compacted_until = source.compacted_until, .positions = positions, .coverage_ends = coverage, .keys = keys, .values = values, .bias = bias, .fp8 = .{ .first_row = source.fp8_first_row, .rows = source.fp8_rows, .key_scale = source.key_scale, .value_scale = source.value_scale }, .compressed = .{ .version = compressed_version, .spans = spans, .keys = compressed_keys, .values = compressed_values } } };
    }

    pub fn synchronize(self: *DeviceState) !void {
        if (self.invalidated) return error.InvalidAttentionMatchCache;
        try self.runner.synchronize();
    }
    pub fn deinit(self: *DeviceState) void {
        if (self.runner.deferred_state == self) self.runner.cancelDeferredFrame();
        for (self.heads) |*head| deinitHead(head);
        self.allocator.free(self.heads);
        self.* = undefined;
    }
};

/// Conservative destination-owner peak for cloning the current packed state
/// and growing it by `extra_rows`. FP8 rows remain encoded; future appended
/// rows use exact f16 storage. A growth replaces one complete head at a time,
/// so two copies of the largest target allocation bound every intermediate
/// generation, including page rounding and metadata.
pub fn cloneCapacityAllocationBytes(state: *const DeviceState, extra_rows: usize) !usize {
    return cloneCapacityAllocationBytesWithPolicy(state, extra_rows, false);
}

/// Destination-owner bound for the explicit exact-local policy. The legacy
/// public helper above remains byte-identical. Local exact heads use the same
/// causal-union capacity as append; global and encoded heads retain the legacy
/// bound. Two target allocations still conservatively cover grow transfer.
pub fn cloneCapacityAllocationBytesWithPolicy(state: *const DeviceState, extra_rows: usize, exact_local_window: bool) !usize {
    if (state.heads.len == 0) return error.InvalidAttentionShape;
    var bytes: usize = 0;
    for (state.heads) |head| {
        if (head.logical_length != state.logical_length or head.rows == 0 or
            head.rows > head.capacity or head.start >= head.capacity or head.fp8_rows > head.rows)
            return error.InvalidAttentionShape;
        if (extra_rows == 0) {
            bytes = try add(bytes, try headBytes(.{ .capacity = head.rows, .dim = head.dim, .encoding = head.encoding, .compressed_rows = head.compressed_rows, .span_count = head.span_count, .fp8_rows = head.fp8_rows }));
            continue;
        }
        const target_rows = try add(head.rows, extra_rows);
        const target_capacity = if (exact_local_window and head.window != 0) blk: {
            if (head.encoding != .exact) return error.UnsupportedMetalKvCheckpointEncoding;
            break :blk try appendCapacity(head, extra_rows, true);
        } else if (head.window == 0)
            try mul(try std.math.divCeil(usize, target_rows, 256), 256)
        else
            @max(target_rows, try add(@as(usize, head.window), 512));
        const target_bytes = switch (head.encoding) {
            .exact, .fp8 => try bufferBytes(target_capacity, head.dim, head.fp8_rows),
            .rotated4 => try rotatedBufferBytes(target_capacity, head.dim, head.compressed_rows, head.span_count),
        };
        bytes = try add(bytes, try mul(target_bytes, 2));
    }
    return bytes;
}

test "context clone admission uses exact causal union without changing legacy bound" {
    var heads = [_]Head{.{ .window = 512, .dim = 256, .rows = 512, .capacity = 512, .logical_length = 512 }};
    const state: DeviceState = .{ .allocator = undefined, .runner = undefined, .heads = &heads, .logical_length = 512, .initial_logical_length = 512 };
    const exact_one = try cloneCapacityAllocationBytesWithPolicy(&state, 1, true);
    const exact_many = try cloneCapacityAllocationBytesWithPolicy(&state, 256, true);
    const legacy = try cloneCapacityAllocationBytes(&state, 1);
    try std.testing.expect(exact_one < legacy);
    try std.testing.expect(exact_many > exact_one);
    try std.testing.expectEqual(try mul(try bufferBytes(512, 256, 0), 2), exact_one);
    try std.testing.expectEqual(try mul(try bufferBytes(767, 256, 0), 2), exact_many);
}

fn readMetadata(buffer: Buffer, head: Head, bias: []f32, positions: []u64, coverage: []u64) !void {
    if (bias.len != head.rows or positions.len != head.rows or coverage.len != head.rows) return error.InvalidKvCheckpoint;
    const source = std.mem.bytesAsSlice(Metadata, @as([]align(@alignOf(Metadata)) u8, @alignCast(buffer.bytes[0..buffer.logical_bytes])));
    const first = @min(head.rows, head.capacity - head.start);
    for (0..head.rows) |row| {
        const meta = source[if (row < first) head.start + row else row - first];
        bias[row] = meta.bias;
        positions[row] = @as(u64, meta.position_lo) | (@as(u64, meta.position_hi) << 32);
        coverage[row] = @as(u64, meta.coverage_lo) | (@as(u64, meta.coverage_hi) << 32);
    }
}
fn readStoredValues(buffer: ?Buffer, encoded: ?Buffer, head: Head, output: []f32, scale: f32, is_key: bool) !void {
    if (output.len != try mul(head.rows, head.dim)) return error.InvalidKvCheckpoint;
    const source = if (buffer) |value|
        std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(value.bytes[0..value.logical_bytes])))
    else
        &.{};
    const encoded_bytes = if (encoded) |value| value.bytes[0..value.logical_bytes] else &.{};
    const encoded_offset: usize = if (is_key) @sizeOf(Fp8Header) else 0;
    for (0..head.rows) |row| {
        const physical = (head.start + row) % head.capacity;
        const compressed = physical >= head.fp8_first_row and physical < @as(usize, head.fp8_first_row) + head.fp8_rows;
        const storage_row = if (compressed) physical - head.fp8_first_row else if (physical < head.fp8_first_row) physical else physical - head.fp8_rows;
        for (0..head.dim) |d| {
            const out = &output[row * head.dim + d];
            out.* = if (compressed)
                try codec.decode(encoded_bytes[encoded_offset + storage_row * head.dim + d], scale)
            else
                @floatCast(source[storage_row * head.dim + d]);
        }
    }
}

fn uploadCheckpointHead(runner: *Runner, destination: *Head, source: checkpoint.Head, donor_layer: u32, head_index: usize, window: u32, logical_length: u64, control: Control) !void {
    try control.check();
    const encoding: StorageEncoding = if (source.compressed.spans.len != 0) .rotated4 else if (source.fp8.rows != 0) .fp8 else .exact;
    destination.* = .{ .layer = donor_layer, .head = @intCast(head_index), .window = window, .dim = source.dim, .rows = source.positions.len, .capacity = source.positions.len, .logical_length = logical_length, .compacted_until = source.compacted_until, .rollback_floor = logical_length, .encoding = encoding, .compressed_rows = if (encoding == .fp8) source.fp8.rows else @intCast(source.compressed.rows()), .span_count = @intCast(source.compressed.spans.len), .fp8_first_row = source.fp8.first_row, .fp8_rows = source.fp8.rows, .key_scale = source.fp8.key_scale, .value_scale = source.fp8.value_scale };
    errdefer {
        deinitHead(destination);
        destination.* = .{};
    }
    try runner.allocateHead(destination, destination.capacity);
    const keys = if (destination.keys) |buffer| std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(buffer.bytes[0..buffer.logical_bytes]))) else null;
    const values = if (destination.values) |buffer| std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(buffer.bytes[0..buffer.logical_bytes]))) else null;
    const encoded_keys = if (destination.encoded_keys) |buffer| buffer.bytes[0..buffer.logical_bytes] else runner.fp8_zero.bytes[0..0];
    const encoded_values = if (destination.encoded_values) |buffer| buffer.bytes[0..buffer.logical_bytes] else runner.fp8_zero.bytes[0..0];
    if (source.fp8.rows != 0) {
        const header: *Fp8Header = @ptrCast(@alignCast(encoded_keys.ptr));
        header.* = .{ .encoding = @backingInt(StorageEncoding.fp8), .first_row = source.fp8.first_row, .rows = source.fp8.rows, .compressed_rows = source.fp8.rows, .key_row_bytes = source.dim, .value_row_bytes = source.dim, .key_scale = source.fp8.key_scale, .value_scale = source.fp8.value_scale };
    } else if (source.compressed.spans.len != 0) {
        const header: *Fp8Header = @ptrCast(@alignCast(encoded_keys.ptr));
        const key_payload_offset = try add(@sizeOf(Fp8Header), try mul(source.compressed.spans.len, @sizeOf(packed_storage.Span)));
        header.* = .{ .encoding = @backingInt(StorageEncoding.rotated4), .span_count = @intCast(source.compressed.spans.len), .compressed_rows = @intCast(source.compressed.rows()), .key_payload_offset = @intCast(key_payload_offset), .key_row_bytes = @intCast(rotated.rowBytes(source.dim)), .value_row_bytes = @intCast(packed_storage.valueBytes(source.dim)) };
        @memcpy(encoded_keys[@sizeOf(Fp8Header)..key_payload_offset], std.mem.sliceAsBytes(source.compressed.spans));
        @memcpy(encoded_keys[key_payload_offset..][0..source.compressed.keys.len], source.compressed.keys);
        @memcpy(encoded_values[0..source.compressed.values.len], source.compressed.values);
    }
    for (0..source.positions.len) |row| {
        if (row % 256 == 0) try control.check();
        const index_in_storage: ?packed_storage.Index = if (source.compressed.spans.len != 0) source.compressed.index(row) else null;
        const is_packed = if (index_in_storage) |storage_index| switch (storage_index) {
            .compressed => true,
            .exact => false,
        } else false;
        const compressed = is_packed or (row >= source.fp8.first_row and row < @as(usize, source.fp8.first_row) + source.fp8.rows);
        const storage_row = if (index_in_storage) |storage_index| switch (storage_index) {
            .exact => |i| i,
            .compressed => |i| i,
        } else if (compressed) row - source.fp8.first_row else if (row < source.fp8.first_row) row else row - source.fp8.rows;
        for (0..source.dim) |d| {
            const source_index = row * source.dim + d;
            const storage_index = storage_row * source.dim + d;
            if (compressed and !is_packed) {
                encoded_keys[@sizeOf(Fp8Header) + storage_index] = try codec.encode(source.keys[source_index], source.fp8.key_scale);
                encoded_values[storage_index] = try codec.encode(source.values[source_index], source.fp8.value_scale);
            } else if (!is_packed) {
                keys.?[storage_index] = @floatCast(source.keys[source_index]);
                values.?[storage_index] = @floatCast(source.values[source_index]);
            }
        }
    }
    const metadata = std.mem.bytesAsSlice(Metadata, @as([]align(@alignOf(Metadata)) u8, @alignCast(destination.metadata.?.bytes[0..destination.metadata.?.logical_bytes])));
    for (metadata, source.bias, source.positions, source.coverage_ends) |*out, bias, position, coverage_end| out.* = .{ .bias = bias, .position_lo = @truncate(position), .position_hi = @truncate(position >> 32), .coverage_lo = @truncate(coverage_end), .coverage_hi = @truncate(coverage_end >> 32) };
}

pub const CandidateHeadDescriptor = struct { layer: u32, head: u32, window: u32 };

pub const CandidateBuilder = struct {
    allocator: std.mem.Allocator,
    state: DeviceState,
    descriptors: []CandidateHeadDescriptor,
    initialized: []bool,
    expected_structure_sha256: [32]u8,
    expected_state_sha256: [32]u8,
    max_bytes: usize,
    finished: bool = false,

    pub fn init(backend: Backend, logical_length: u64, descriptors: []const CandidateHeadDescriptor, expected_structure_sha256: [32]u8, expected_state_sha256: [32]u8, max_bytes: usize, control: Control) !CandidateBuilder {
        try control.check();
        if (logical_length == 0 or descriptors.len == 0 or max_bytes == 0 or std.mem.allEqual(u8, &expected_structure_sha256, 0) or std.mem.allEqual(u8, &expected_state_sha256, 0)) return error.InvalidTrainingAugmentationBinding;
        const owned_descriptors = try backend.allocator.dupe(CandidateHeadDescriptor, descriptors);
        errdefer backend.allocator.free(owned_descriptors);
        for (owned_descriptors, 0..) |descriptor, index| {
            if (index > 0 and (descriptor.layer < owned_descriptors[index - 1].layer or (descriptor.layer == owned_descriptors[index - 1].layer and descriptor.head != owned_descriptors[index - 1].head + 1))) return error.InvalidTrainingAugmentationBinding;
        }
        const heads = try backend.allocator.alloc(Head, descriptors.len);
        errdefer backend.allocator.free(heads);
        @memset(heads, .{});
        const initialized = try backend.allocator.alloc(bool, descriptors.len);
        errdefer backend.allocator.free(initialized);
        @memset(initialized, false);
        return .{ .allocator = backend.allocator, .state = .{ .allocator = backend.allocator, .runner = backend.runner, .heads = heads, .logical_length = logical_length, .initial_logical_length = logical_length }, .descriptors = owned_descriptors, .initialized = initialized, .expected_structure_sha256 = expected_structure_sha256, .expected_state_sha256 = expected_state_sha256, .max_bytes = max_bytes };
    }

    pub fn uploadHead(self: *CandidateBuilder, index: usize, source: checkpoint.Head, control: Control) !void {
        if (self.finished or index >= self.descriptors.len or self.initialized[index]) return error.InvalidTrainingAugmentationBinding;
        const descriptor = self.descriptors[index];
        const encoding: StorageEncoding = if (source.compressed.spans.len != 0) .rotated4 else if (source.fp8.rows != 0) .fp8 else .exact;
        const bytes = try headBytes(.{ .capacity = source.positions.len, .dim = source.dim, .encoding = encoding, .compressed_rows = if (encoding == .fp8) source.fp8.rows else @intCast(source.compressed.rows()), .span_count = @intCast(source.compressed.spans.len), .fp8_rows = source.fp8.rows });
        if (bytes > self.max_bytes - self.state.allocated_bytes) return error.CompactionWorkspaceLimit;
        try self.state.runner.admit(bytes, self.max_bytes - self.state.allocated_bytes);
        try uploadCheckpointHead(self.state.runner, &self.state.heads[index], source, descriptor.layer, descriptor.head, descriptor.window, self.state.logical_length, control);
        self.state.allocated_bytes = try add(self.state.allocated_bytes, bytes);
        if (self.state.allocated_bytes > self.max_bytes) {
            deinitHead(&self.state.heads[index]);
            self.state.heads[index] = .{};
            self.state.allocated_bytes -= bytes;
            return error.CompactionWorkspaceLimit;
        }
        self.initialized[index] = true;
    }

    pub fn finish(self: *CandidateBuilder, expected_structure_sha256: [32]u8, expected_state_sha256: [32]u8, control: Control) !DeviceState {
        if (self.finished or !std.mem.eql(u8, &self.expected_structure_sha256, &expected_structure_sha256) or !std.mem.eql(u8, &self.expected_state_sha256, &expected_state_sha256)) return error.InvalidTrainingAugmentationBinding;
        for (self.initialized, self.descriptors, self.state.heads) |ready, descriptor, head| {
            if (!ready or head.layer != descriptor.layer or head.head != descriptor.head or head.window != descriptor.window) return error.IncompleteContextAllocationScope;
            try validateStoredHead(head, self.state.logical_length, descriptor.layer, descriptor.head, descriptor.window, checkpoint.max_context_tokens);
        }
        if (self.state.allocated_bytes == 0 or self.state.allocated_bytes > self.max_bytes) return error.CompactionWorkspaceLimit;
        try control.check();
        try self.state.synchronize();
        self.finished = true;
        self.allocator.free(self.initialized);
        self.allocator.free(self.descriptors);
        self.initialized = &.{};
        self.descriptors = &.{};
        const result = self.state;
        self.state.heads = &.{};
        return result;
    }

    pub fn deinit(self: *CandidateBuilder) void {
        if (!self.finished) self.state.deinit();
        if (self.initialized.len != 0) self.allocator.free(self.initialized);
        if (self.descriptors.len != 0) self.allocator.free(self.descriptors);
        self.* = undefined;
    }
};

pub const Backend = struct {
    allocator: std.mem.Allocator,
    runner: *Runner,
    limits: checkpoint.Limits = checkpoint.serving_limits,

    pub fn allocationBytes(self: Backend, state: checkpoint.State) !usize {
        return allocationBytesWithControl(state, self.limits, .{});
    }
    pub fn upload(self: Backend, state: checkpoint.State, max_bytes: usize, control: Control) !DeviceState {
        const bytes = try allocationBytesWithControl(state, self.limits, control);
        if (self.runner.exact_local_window_capacity) for (state.donors) |donor| if (donor.window != 0) for (donor.heads) |source| {
            // Upload is already exact-sized (`positions.len`); make the
            // restricted contract explicit before the first allocation.
            if (source.positions.len == 0 or source.compressed.spans.len != 0 or source.fp8.rows != 0)
                return error.UnsupportedMetalKvCheckpointEncoding;
        };
        try self.runner.admit(bytes, max_bytes);
        var count: usize = 0;
        for (state.donors) |donor| count = try add(count, donor.heads.len);
        var result: DeviceState = .{ .allocator = self.allocator, .runner = self.runner, .logical_length = state.logical_length, .initial_logical_length = state.logical_length };
        errdefer result.deinit();
        result.heads = try self.allocator.alloc(Head, count);
        @memset(result.heads, .{});
        var index: usize = 0;
        for (state.donors) |donor| for (donor.heads, 0..) |source, head_index| {
            const destination = &result.heads[index];
            index += 1;
            try uploadCheckpointHead(self.runner, destination, source, donor.layer, head_index, donor.window, state.logical_length, control);
        };
        result.allocated_bytes = bytes;
        try control.check();
        return result;
    }
};

pub const Request = struct {
    heads: []const Head,
    queries: Ref,
    output: Ref,
    log_mass: ?Ref = null,
    query_rows: usize,
    query_heads: usize,
    first_query_position: u64,
    scale: f32,
    softcap: f32 = 0,
};

pub const LayerActivationPlan = struct { widths: [6]usize };
pub const RunnerOptions = struct { exact_local_window_capacity: bool = false };

const DeferredPublication = struct {
    rows: usize,
    start: usize,
    logical_length: u64,
    rollback_floor: u64,
};

fn restoreDeferredPublications(state: *DeviceState, publications: []const DeferredPublication, logical_length: u64) !void {
    if (publications.len != state.heads.len) return error.InvalidAttentionShape;
    for (state.heads, publications) |*head, entry| {
        head.rows = entry.rows;
        head.start = entry.start;
        head.logical_length = entry.logical_length;
        head.rollback_floor = entry.rollback_floor;
    }
    state.logical_length = logical_length;
}

test "context exact local deferred publication restores only an unsubmitted frame" {
    var heads = [_]Head{
        .{ .rows = 512, .capacity = 512, .start = 1, .logical_length = 513, .rollback_floor = 513 },
        .{ .rows = 512, .capacity = 512, .start = 1, .logical_length = 513, .rollback_floor = 513 },
    };
    var state: DeviceState = .{ .allocator = std.testing.allocator, .runner = undefined, .heads = &heads, .logical_length = 513, .initial_logical_length = 512 };
    const publications = [_]DeferredPublication{
        .{ .rows = 512, .start = 0, .logical_length = 512, .rollback_floor = 0 },
        .{ .rows = 512, .start = 0, .logical_length = 512, .rollback_floor = 0 },
    };
    try restoreDeferredPublications(&state, &publications, 512);
    try std.testing.expectEqual(@as(u64, 512), state.logical_length);
    for (state.heads) |head| {
        try std.testing.expectEqual(@as(usize, 0), head.start);
        try std.testing.expectEqual(@as(u64, 512), head.logical_length);
    }
    state.invalidated = true;
    try std.testing.expectError(error.InvalidAttentionMatchCache, state.layer(0));
}

pub const Runner = struct {
    core: shared.Kernels,
    copier: shared.Kernels,
    split_stage: shared.Kernels,
    split_variants: [12]?*metal_runtime.RawMetalGeneratedPipeline = @splat(null),
    // Bounded exact-F16 GQA8 stages: local SIMD and global matrix. Capture the opt-in once
    // per owner so deferred reservation and dispatch cannot disagree.
    matrix_stage_enabled: bool = false,
    coalesced_frame_enabled: bool = false,
    // Default-off staged policy. It may be enabled only before a DeviceState
    // is created so every head in that owner shares one capacity contract.
    exact_local_window_capacity: bool = false,
    planned_dispatch_calls: usize = 0,
    matrix_variants: [7]?*metal_runtime.RawMetalGeneratedPipeline = @splat(null),
    rotated_query_pipeline: ?*metal_runtime.RawMetalGeneratedPipeline = null,
    rotated_queries: ?Buffer = null,
    rotated_query_dispatches: usize = 0,
    matrix_stage_dispatches: usize = 0,
    split_reduce: shared.Kernels,
    status: Buffer,
    fp8_zero: Buffer,
    scratch: ?Buffer = null,
    // Request-owned, distinct per-layer outputs. They survive all token frames
    // and are charged to this owner's committed context, including page padding.
    layer_outputs: [256]?Buffer = @splat(null),
    layer_output_count: usize = 0,
    layer_output_width: usize = 0,
    layer_activations: [256][6]?Buffer = @splat(@splat(null)),
    // V7-only frontier: roles stay distinct and L aliases only L-2.
    layer_activation_frontier: [2][6]?Buffer = @splat(@splat(null)),
    layer_activation_frontier_enabled: bool = false,
    layer_activation_frontier_next_layer: usize = 0,
    layer_activation_widths: [256][6]usize = @splat(@splat(0)),
    layer_activation_count: usize = 0,
    allocation_tracker: ?*AllocationTracker,
    max_device_bytes: usize,
    // A forward owns one unsubmitted provider frame. Capacity is fixed before
    // encoding, and its sticky status is accepted only by finishDeferredFrame.
    deferred_runtime: ?*metal_runtime.RawMetalDecodeRuntime = null,
    deferred_state: ?*DeviceState = null,
    deferred_end: u64 = 0,
    deferred_query_heads: usize = 0,
    deferred_publications: ?[]DeferredPublication = null,
    deferred_prior_logical_length: u64 = 0,
    deferred_submitted: bool = false,
    deferred_submission_epoch: u64 = 0,

    pub fn init(max_device_bytes: usize, tracker: ?*AllocationTracker) !Runner {
        return initWithOptions(max_device_bytes, tracker, .{});
    }

    pub fn initWithOptions(max_device_bytes: usize, tracker: ?*AllocationTracker, options: RunnerOptions) !Runner {
        // Each generated-kernel owner has one page-rounded dummy buffer, and
        // status/FP8-zero each own another page. Admit that unavoidable live
        // baseline before constructing any native runtime or shared buffer.
        try admitRunnerBaseline(max_device_bytes, tracker);
        var core = try shared.Kernels.initSourceTracked(@embedFile("metal_attention_match_storage.metal"), "metal_attention_match_storage", tracker);
        errdefer core.deinit();
        var copier = try shared.Kernels.initSourceTracked(@embedFile("metal_attention_match_storage.metal"), "metal_attention_match_copy", tracker);
        errdefer copier.deinit();
        var split_stage = try shared.Kernels.initSourceTracked(@embedFile("metal_attention_match_parallel_split.metal"), "metal_attention_match_split_stage", tracker);
        errdefer split_stage.deinit();
        var split_reduce = try shared.Kernels.initSourceTracked(@embedFile("metal_attention_match_split_k.metal"), "metal_attention_match_split_reduce", tracker);
        errdefer split_reduce.deinit();
        var status = try core.buffer(4);
        errdefer status.deinit();
        var fp8_zero = try core.buffer(@sizeOf(Fp8Header));
        errdefer fp8_zero.deinit();
        @memset(fp8_zero.bytes[0..fp8_zero.logical_bytes], 0);
        const diagnostic = @import("../runtime/kv/attention_match_release.zig").Execution.diagnosticFromEnvironment();
        return .{ .core = core, .copier = copier, .split_stage = split_stage, .split_reduce = split_reduce, .status = status, .fp8_zero = fp8_zero, .allocation_tracker = tracker, .max_device_bytes = max_device_bytes, .matrix_stage_enabled = diagnostic.matrix_stage, .coalesced_frame_enabled = diagnostic.coalesced_stage, .exact_local_window_capacity = options.exact_local_window_capacity };
    }
    pub fn raiseMaxDeviceBytes(self: *Runner, maximum: usize) void {
        self.max_device_bytes = @max(self.max_device_bytes, maximum);
    }
    pub fn backend(self: *Runner, allocator: std.mem.Allocator, limits: checkpoint.Limits) Backend {
        return .{ .allocator = allocator, .runner = self, .limits = limits };
    }
    pub fn admitCapacity(self: *Runner, bytes: usize, call_limit: usize) !void {
        try self.admit(bytes, call_limit);
    }
    pub fn deinit(self: *Runner) void {
        self.cancelDeferredFrame();
        self.synchronize() catch {};
        for (self.split_variants) |pipeline| if (pipeline) |owned| metal_runtime.termite_metal_generated_pipeline_destroy(owned);
        for (self.matrix_variants) |pipeline| if (pipeline) |owned| metal_runtime.termite_metal_generated_pipeline_destroy(owned);
        if (self.rotated_query_pipeline) |pipeline| metal_runtime.termite_metal_generated_pipeline_destroy(pipeline);
        if (self.rotated_queries) |*buffer| buffer.deinit();
        if (self.scratch) |*buffer| buffer.deinit();
        self.releaseLayerActivations();
        self.releaseLayerOutputs();
        self.fp8_zero.deinit();
        self.status.deinit();
        self.split_reduce.deinit();
        self.split_stage.deinit();
        self.copier.deinit();
        self.core.deinit();
        self.* = undefined;
    }
    pub fn synchronize(self: *Runner) !void {
        if (self.deferred_runtime != null) return error.ContextTransactionBusy;
        if (self.core.batch_active or self.core.batch_submitted) try self.core.endBatch();
        if (self.copier.batch_active or self.copier.batch_submitted) try self.copier.endBatch();
        if (self.split_stage.batch_active or self.split_stage.batch_submitted) try self.split_stage.endBatch();
        if (self.split_reduce.batch_active or self.split_reduce.batch_submitted) try self.split_reduce.endBatch();
    }
    fn liveBytes(self: *Runner) ?usize {
        const tracker = self.allocation_tracker orelse return null;
        const snapshot = tracker.snapshot();
        if (!snapshot.tracked_combined_physical_available) return null;
        return std.math.cast(usize, snapshot.tracked_combined_live_bytes);
    }
    fn admit(self: *Runner, bytes: usize, call_limit: usize) !void {
        const limit = @min(call_limit, self.max_device_bytes);
        if (bytes > limit) return error.MetalMemoryLimitExceeded;
        if (self.liveBytes()) |live| if (live > limit - bytes) return error.MetalMemoryLimitExceeded;
    }
    fn allocateHead(self: *Runner, head: *Head, capacity: usize) !void {
        head.capacity = capacity;
        errdefer deinitHead(head);
        const compressed_rows = if (head.encoding == .rotated4) head.compressed_rows else head.fp8_rows;
        const exact_rows = std.math.sub(usize, capacity, compressed_rows) catch return error.InvalidKvCheckpoint;
        if (exact_rows != 0) {
            head.keys = try self.core.buffer(try mul(try mul(exact_rows, head.dim), 2));
            head.values = try self.core.buffer(try mul(try mul(exact_rows, head.dim), 2));
        }
        head.metadata = try self.core.buffer(try mul(capacity, @sizeOf(Metadata)));
        if (compressed_rows != 0) {
            const key_bytes = if (head.encoding == .rotated4)
                try add(try add(@sizeOf(Fp8Header), try mul(head.span_count, @sizeOf(packed_storage.Span))), try mul(compressed_rows, rotated.rowBytes(head.dim)))
            else
                try add(@sizeOf(Fp8Header), try mul(compressed_rows, head.dim));
            const value_bytes = if (head.encoding == .rotated4)
                try mul(compressed_rows, packed_storage.valueBytes(head.dim))
            else
                try mul(compressed_rows, head.dim);
            head.encoded_keys = try self.core.buffer(key_bytes);
            head.encoded_values = try self.core.buffer(value_bytes);
        }
    }
    fn grow(self: *Runner, state: *DeviceState, head: *Head, capacity: usize, control: Control) !void {
        if (capacity <= head.capacity) return;
        if (self.deferred_runtime != null) return error.ContextStagingBudgetExceeded;
        try control.check();
        const next_bytes = switch (head.encoding) {
            .exact, .fp8 => try bufferBytes(capacity, head.dim, head.fp8_rows),
            .rotated4 => try rotatedBufferBytes(capacity, head.dim, head.compressed_rows, head.span_count),
        };
        try self.admit(next_bytes, self.max_device_bytes);
        const old_bytes = try headBytes(head.*);
        var replacement = head.*;
        replacement.keys = null;
        replacement.values = null;
        replacement.encoded_keys = null;
        replacement.encoded_values = null;
        replacement.metadata = null;
        replacement.start = 0;
        try self.allocateHead(&replacement, capacity);
        errdefer deinitHead(&replacement);
        if (head.encoding == .exact) {
            try ringCopy(replacement.keys.?, head.keys.?, head.*, head.dim * 2);
            try ringCopy(replacement.values.?, head.values.?, head.*, head.dim * 2);
        } else {
            if (head.start != 0 or head.window != 0) return error.InvalidAttentionShape;
            if (head.keys) |source| @memcpy(replacement.keys.?.bytes[0..source.logical_bytes], source.bytes[0..source.logical_bytes]);
            if (head.values) |source| @memcpy(replacement.values.?.bytes[0..source.logical_bytes], source.bytes[0..source.logical_bytes]);
            @memcpy(replacement.encoded_keys.?.bytes[0..replacement.encoded_keys.?.logical_bytes], head.encoded_keys.?.bytes[0..head.encoded_keys.?.logical_bytes]);
            @memcpy(replacement.encoded_values.?.bytes[0..replacement.encoded_values.?.logical_bytes], head.encoded_values.?.bytes[0..head.encoded_values.?.logical_bytes]);
        }
        try ringCopy(replacement.metadata.?, head.metadata.?, head.*, @sizeOf(Metadata));
        deinitHead(head);
        head.* = replacement;
        state.allocated_bytes = state.allocated_bytes - old_bytes + next_bytes;
    }
    fn releaseLayerActivations(self: *Runner) void {
        for (&self.layer_activations) |*row| for (row) |*entry| if (entry.*) |*buffer| {
            buffer.deinit();
            entry.* = null;
        };
        for (&self.layer_activation_frontier) |*row| for (row) |*entry| if (entry.*) |*buffer| {
            buffer.deinit();
            entry.* = null;
        };
        self.layer_activation_frontier_enabled = false;
        self.layer_activation_frontier_next_layer = 0;
        self.layer_activation_count = 0;
        self.layer_activation_widths = @splat(@splat(0));
    }

    /// Admit all layers before allocating any backing. Every role has a
    /// distinct buffer in each layer; no same-frame activation reuse occurs.
    pub fn reserveLayerActivations(self: *Runner, plans: []const LayerActivationPlan) !void {
        return self.reserveLayerActivationsWithFrontier(plans, false);
    }

    pub fn reserveLayerActivationsWithFrontier(self: *Runner, plans: []const LayerActivationPlan, frontier: bool) !void {
        if (plans.len == 0 or plans.len > self.layer_activations.len or (frontier and plans.len < 2))
            return error.InvalidAttentionShape;
        if (self.deferred_runtime != null or self.core.batch_active or self.core.batch_submitted or
            self.copier.batch_active or self.copier.batch_submitted) return error.ContextStagingBudgetExceeded;
        var bytes: usize = 0;
        var frontier_widths: [2][6]usize = @splat(@splat(0));
        for (plans, 0..) |plan, layer| for (plan.widths, 0..) |width, role| {
            if (width == 0 or width > std.math.maxInt(i32)) return error.InvalidAttentionShape;
            if (self.layer_activation_count != 0 and (frontier != self.layer_activation_frontier_enabled or
                plans.len != self.layer_activation_count or width != self.layer_activation_widths[layer][role]))
                return error.InvalidAttentionShape;
            if (frontier)
                frontier_widths[layer % 2][role] = @max(frontier_widths[layer % 2][role], width)
            else
                bytes = try add(bytes, try rounded(try mul(width, @sizeOf(f32))));
        };
        if (self.layer_activation_count != 0) return;
        if (frontier) {
            for (frontier_widths) |row| {
                for (row) |width| bytes = try add(bytes, try rounded(try mul(width, @sizeOf(f32))));
            }
        }
        try self.admit(bytes, self.max_device_bytes);
        errdefer self.releaseLayerActivations();
        if (frontier) {
            for (&self.layer_activation_frontier, frontier_widths) |*row, widths| {
                for (row, widths) |*entry, width| entry.* = try self.core.buffer(try mul(width, @sizeOf(f32)));
            }
        } else {
            for (plans, 0..) |plan, layer| {
                for (plan.widths, 0..) |width, role| self.layer_activations[layer][role] = try self.core.buffer(try mul(width, @sizeOf(f32)));
            }
        }
        for (plans, 0..) |plan, layer| {
            for (plan.widths, 0..) |width, role| self.layer_activation_widths[layer][role] = width;
        }
        self.layer_activation_frontier_enabled = frontier;
        self.layer_activation_frontier_next_layer = 0;
        self.layer_activation_count = plans.len;
    }

    pub fn beginLayerActivationFrontierFrame(self: *Runner) !void {
        if (self.deferred_runtime != null or self.core.batch_active or self.core.batch_submitted or
            self.copier.batch_active or self.copier.batch_submitted) return error.ContextStagingBudgetExceeded;
        self.layer_activation_frontier_next_layer = 0;
    }

    pub fn enterLayerActivation(self: *Runner, layer: usize) !void {
        if (!self.layer_activation_frontier_enabled) return;
        if (layer != self.layer_activation_frontier_next_layer or layer >= self.layer_activation_count)
            return error.InvalidAttentionShape;
        self.layer_activation_frontier_next_layer += 1;
    }

    pub fn validateLayerActivationFrontierComplete(self: *const Runner) !void {
        if (self.layer_activation_frontier_enabled and self.layer_activation_frontier_next_layer != self.layer_activation_count)
            return error.InvalidAttentionShape;
    }

    pub fn layerActivation(self: *Runner, layer: usize, role: usize, width: usize) !Ref {
        const runtime = self.deferred_runtime orelse return error.ContextTransactionBusy;
        if (!metal_runtime.hasActiveFrame(runtime) or metal_runtime.hasSubmittedFrame(runtime)) return error.ContextTransactionBusy;
        if (layer >= self.layer_activation_count or role >= 6 or width == 0 or
            width > self.layer_activation_widths[layer][role]) return error.InvalidAttentionShape;
        const backing = if (self.layer_activation_frontier_enabled)
            self.layer_activation_frontier[layer % 2][role]
        else
            self.layer_activations[layer][role];
        return try (backing orelse return error.InvalidAttentionShape).ref().slice(0, try mul(width, @sizeOf(f32)));
    }

    fn releaseLayerOutputs(self: *Runner) void {
        for (&self.layer_outputs) |*entry| if (entry.*) |*buffer| {
            buffer.deinit();
            entry.* = null;
        };
        self.layer_output_count = 0;
        self.layer_output_width = 0;
    }

    pub fn reserveLayerOutputs(self: *Runner, layers: usize, width: usize) !void {
        if (layers == 0 or layers > self.layer_outputs.len or width == 0 or width > 65536)
            return error.InvalidAttentionShape;
        if (self.deferred_runtime != null or self.core.batch_active or self.core.batch_submitted or
            self.copier.batch_active or self.copier.batch_submitted)
            return error.ContextStagingBudgetExceeded;
        if (self.layer_output_count != 0) {
            if (self.layer_output_count != layers or self.layer_output_width != width)
                return error.InvalidAttentionShape;
            return;
        }
        const bytes = try mul(width, @sizeOf(f32));
        try self.admit(try mul(layers, try rounded(bytes)), self.max_device_bytes);
        errdefer self.releaseLayerOutputs();
        for (self.layer_outputs[0..layers]) |*entry| entry.* = try self.core.buffer(bytes);
        self.layer_output_count = layers;
        self.layer_output_width = width;
    }

    /// Only the owning active frame may borrow these allocations. Every layer
    /// has a different backing buffer, so queued residual reads never alias a
    /// later layer's output. The owner drains the frame before teardown.
    pub fn layerOutput(self: *Runner, layer: usize, width: usize) !Ref {
        const runtime = self.deferred_runtime orelse return error.ContextTransactionBusy;
        if (!metal_runtime.hasActiveFrame(runtime) or metal_runtime.hasSubmittedFrame(runtime))
            return error.ContextTransactionBusy;
        if (layer >= self.layer_output_count or width != self.layer_output_width)
            return error.InvalidAttentionShape;
        return (self.layer_outputs[layer] orelse return error.InvalidAttentionShape).ref();
    }

    fn ensureScratch(self: *Runner, bytes: usize) !void {
        if (self.scratch) |buffer| if (buffer.logical_bytes >= bytes) return;
        if (self.deferred_runtime != null) return error.ContextStagingBudgetExceeded;
        const physical = try rounded(bytes);
        try self.admit(physical, self.max_device_bytes);
        var replacement = try self.core.buffer(bytes);
        errdefer replacement.deinit();
        if (self.scratch) |*old| old.deinit();
        self.scratch = replacement;
    }
    fn ensureRotatedQueries(self: *Runner, query_heads: usize, dim: usize) !Ref {
        if (query_heads == 0 or query_heads > 256 or dim != 512) return error.InvalidAttentionShape;
        const bytes = try mul(try mul(try mul(query_heads, dim), 2), @sizeOf(f32));
        if (self.rotated_queries) |buffer| if (buffer.logical_bytes >= bytes)
            return try buffer.ref().slice(0, bytes);
        if (self.deferred_runtime != null) return error.ContextStagingBudgetExceeded;
        try self.admit(try rounded(bytes), self.max_device_bytes);
        var replacement = try self.core.buffer(bytes);
        errdefer replacement.deinit();
        if (self.rotated_queries) |*old| old.deinit();
        self.rotated_queries = replacement;
        return replacement.ref();
    }

    fn rotatedQueryPipeline(self: *Runner) !*metal_runtime.RawMetalGeneratedPipeline {
        if (self.rotated_query_pipeline) |pipeline| return pipeline;
        if (self.deferred_runtime != null) return error.ContextStagingBudgetExceeded;
        const source = @embedFile("metal_attention_match_rotated_matrix.metal");
        var message: [4096]u8 = @splat(0);
        const pipeline = metal_runtime.termite_metal_generated_pipeline_create(source.ptr, source.len, "metal_attention_match_rotate_queries", 1, &message, message.len) orelse {
            std.log.err("Metal rotated query kernel: {s}", .{std.mem.sliceTo(&message, 0)});
            return error.MetalKernelCompilationFailed;
        };
        self.rotated_query_pipeline = pipeline;
        return pipeline;
    }

    fn resetStatus(self: *Runner) void {
        if (self.deferred_runtime != null) return;
        std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(self.status.bytes[0..4])))[0] = 0;
    }

    /// Reserve every donor and the largest decode scratch before opening the
    /// frame. No allocation replacement or host status reset is allowed later.
    pub fn beginDeferredFrame(self: *Runner, state: *DeviceState, runtime: *metal_runtime.RawMetalDecodeRuntime, query_heads: usize, first_position: u64, control: Control) !void {
        try control.check();
        try self.requireOwnedBatchIdle();
        // Exact overwrite uses bounded coalesced entry points below. They prove
        // the provider frame stays active and unsubmitted around every encode.
        if (self.deferred_runtime != null or metal_runtime.hasActiveFrame(runtime) or metal_runtime.hasSubmittedFrame(runtime))
            return error.ContextTransactionBusy;
        if (state.invalidated) return error.InvalidAttentionMatchCache;
        if (state.runner != self or state.heads.len == 0 or query_heads == 0 or query_heads > 256 or first_position != state.logical_length or first_position >= checkpoint.max_context_tokens)
            return error.InvalidAttentionShape;
        var publications: ?[]DeferredPublication = null;
        if (self.exact_local_window_capacity) {
            const owned = try state.allocator.alloc(DeferredPublication, state.heads.len);
            for (state.heads, owned) |head, *entry| entry.* = .{
                .rows = head.rows,
                .start = head.start,
                .logical_length = head.logical_length,
                .rollback_floor = head.rollback_floor,
            };
            publications = owned;
        }
        errdefer {
            if (publications) |owned| {
                restoreDeferredPublications(state, owned, first_position) catch {
                    state.invalidated = true;
                };
                state.allocator.free(owned);
            }
        }
        var maximum_scratch: usize = 0;
        for (state.heads) |*head| {
            try validateStoredHead(head.*, first_position, head.layer, head.head, head.window, checkpoint.max_context_tokens);
            if (head.window != 0 and head.rows > head.window) {
                const drop = head.rows - head.window;
                head.start = (head.start + drop) % head.capacity;
                head.rows -= drop;
                head.rollback_floor = head.logical_length;
            }
            const capacity = try appendCapacity(head.*, 1, self.exact_local_window_capacity);
            try self.grow(state, head, capacity, control);
            const layer_heads = try state.layer(head.layer);
            if (query_heads % layer_heads.len != 0) return error.InvalidAttentionShape;
            var maximum_layer_rows = try add(head.rows, 1);
            for (layer_heads) |layer_head| {
                maximum_layer_rows = @max(maximum_layer_rows, try add(layer_head.rows, 1));
            }
            const matrix_partition = if (self.matrix_stage_enabled)
                matrixLayerPartition(layer_heads, head.dim, head.window, maximum_layer_rows, query_heads / layer_heads.len)
            else
                null;
            const partition: usize = matrix_partition orelse (if (head.window != 0) @as(usize, 32) else split_rows);
            const splits = if ((head.dim == 256 or head.dim == 512) and maximum_layer_rows >= (if (head.window != 0) @as(usize, 128) else 1024))
                std.math.divCeil(usize, maximum_layer_rows, partition) catch return error.MetalMemoryLimitExceeded
            else
                1;
            if (splits > 1) {
                if (matrix_partition != null) {
                    _ = try self.matrixStagePipeline(head.encoding, head.dim, partition);
                    if (head.encoding == .rotated4) {
                        _ = try self.ensureRotatedQueries(query_heads, head.dim);
                        _ = try self.rotatedQueryPipeline();
                    }
                } else {
                    _ = try self.splitStagePipeline(head.encoding, head.dim, partition);
                }
            }
            const padded_matrix = matrix_partition != null and head.dim == 512 and query_heads / layer_heads.len == 4;
            const scratch_heads = if (padded_matrix) try mul(query_heads, 2) else query_heads;
            maximum_scratch = @max(maximum_scratch, try mul(try mul(try mul(scratch_heads, splits), try add(head.dim, 2)), 4));
        }
        try self.ensureScratch(maximum_scratch);
        self.resetStatus();
        try control.check();
        try metal_runtime.beginFrame(runtime);
        self.deferred_runtime = runtime;
        self.deferred_state = state;
        self.deferred_end = first_position + 1;
        self.deferred_query_heads = query_heads;
        self.deferred_publications = publications;
        publications = null;
        self.deferred_prior_logical_length = first_position;
        self.deferred_submitted = false;
        self.deferred_submission_epoch = metal_runtime.frameCommandBufferCount(runtime);
        self.planned_dispatch_calls = 0;
    }

    pub fn cancelDeferredFrame(self: *Runner) void {
        const runtime = self.deferred_runtime orelse return;
        const active = metal_runtime.hasActiveFrame(runtime);
        const submitted = self.deferred_submitted or metal_runtime.hasSubmittedFrame(runtime) or
            metal_runtime.frameCommandBufferCount(runtime) != self.deferred_submission_epoch;
        if (active) metal_runtime.cancelFrame(runtime) catch {};
        if (metal_runtime.hasSubmittedFrame(runtime)) metal_runtime.waitFrame(runtime) catch {};
        if (self.deferred_publications) |owned| {
            if (self.deferred_state) |state| {
                // attendOnFrame may already have cancelled an unsubmitted
                // provider frame before control reaches this owner. Submission,
                // not the frame's current active bit, is the device-write
                // boundary: an unsubmitted frame cannot have changed the ring.
                if (!submitted) {
                    restoreDeferredPublications(state, owned, self.deferred_prior_logical_length) catch {
                        state.invalidated = true;
                    };
                } else {
                    // Submitted GPU writes may have partially overwritten ring
                    // rows. Never manufacture rollback authority without a
                    // device undo journal; poison the owner until deinit.
                    state.invalidated = true;
                }
                state.allocator.free(owned);
            }
        }
        self.deferred_publications = null;
        self.deferred_runtime = null;
        self.deferred_state = null;
        self.deferred_submitted = false;
        self.deferred_submission_epoch = 0;
    }

    pub fn finishDeferredFrame(self: *Runner, control: Control) !void {
        const runtime = self.deferred_runtime orelse return error.ContextTransactionBusy;
        errdefer self.cancelDeferredFrame();
        try control.check();
        const state = self.deferred_state orelse return error.ContextTransactionBusy;
        if (state.logical_length != self.deferred_end) return error.InvalidAttentionShape;
        for (state.heads) |head| if (head.logical_length != self.deferred_end) return error.InvalidAttentionShape;
        if (self.exact_local_window_capacity and self.coalesced_frame_enabled and
            (!metal_runtime.hasActiveFrame(runtime) or metal_runtime.hasSubmittedFrame(runtime) or
                metal_runtime.frameCommandBufferCount(runtime) != self.deferred_submission_epoch))
            return error.ContextTransactionBusy;
        if (metal_runtime.hasActiveFrame(runtime)) {
            // Submission can transfer device ownership before returning an
            // error or before the runtime exposes a submitted flag. Poisoning
            // is conservative; falsely restoring an overwritten ring is not.
            self.deferred_submitted = true;
            try metal_runtime.submitFrame(runtime);
        }
        if (metal_runtime.hasSubmittedFrame(runtime)) try metal_runtime.waitFrame(runtime);
        try self.checkStatus();
        try control.check();
        if (@import("builtin").is_test and self.coalesced_frame_enabled and @import("antfly_platform").env.getenvBool("ANTFLY_CONTEXT_PROFILE")) {
            const snapshot = metal_runtime.runtimeMemorySnapshot(runtime);
            std.debug.print("attention_match_coalesced_frame dispatch_calls={d} encoders={d} barriers={d}\n", .{ self.planned_dispatch_calls, snapshot.last_frame_compute_encoder_count, snapshot.last_frame_planned_barrier_count });
        }
        if (self.deferred_publications) |owned| state.allocator.free(owned);
        self.deferred_publications = null;
        self.deferred_runtime = null;
        self.deferred_state = null;
        self.deferred_submitted = false;
        self.deferred_submission_epoch = 0;
    }
    fn checkStatus(self: *Runner) !void {
        if (std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(self.status.bytes[0..4])))[0] != 0)
            return error.NonFiniteAttention;
    }
    fn requireOwnedBatchIdle(self: *Runner) !void {
        try self.core.requireOwnedBatchIdle();
        try self.copier.requireNoBatchWork();
        try self.split_stage.requireNoBatchWork();
        try self.split_reduce.requireNoBatchWork();
    }
    fn beginOwnedBatch(self: *Runner) !void {
        try self.requireOwnedBatchIdle();
        try self.core.beginBatch();
        if (!self.core.batchCounters().active)
            return error.GeneratedMetalBatchingDisabled;
    }

    // These masks describe the complete shader ABI, including atomic status
    // writes and optional log mass. They never depend on tensor values.
    fn providerWriteMask(self: *Runner, kernels: *shared.Kernels, params: shared.Params) u8 {
        if (kernels == &self.core) return switch (params.op) {
            1 => 0xcc, // append: resident K/V, metadata, sticky status
            2 => 0x82, // scalar attention: private scratch, sticky status
            else => 0xff,
        };
        if (kernels == &self.copier and (params.op == 3 or params.op == 4)) return 0x80;
        if (kernels == &self.split_stage) return 0xc0; // status and partials
        if (kernels == &self.split_reduce) return if (params.mode != 0) 0x86 else 0x84;
        return 0xff;
    }

    fn runProviderDispatch(self: *Runner, kernels: *shared.Kernels, runtime: *metal_runtime.RawMetalDecodeRuntime, params: shared.Params, inputs: []const Ref, output: Ref) !void {
        if (self.coalesced_frame_enabled and self.deferred_runtime == runtime) {
            if (self.exact_local_window_capacity)
                try kernels.runBoundedCoalescedOnActiveRuntime(runtime, self.deferred_submission_epoch, params, inputs, output, self.providerWriteMask(kernels, params))
            else
                try kernels.runCoalescedOnActiveRuntime(runtime, params, inputs, output, self.providerWriteMask(kernels, params));
            self.planned_dispatch_calls +|= 1;
        } else {
            try kernels.runOnActiveRuntime(runtime, params, inputs, output);
        }
    }

    fn runProviderThreadgroups(self: *Runner, kernels: *shared.Kernels, runtime: *metal_runtime.RawMetalDecodeRuntime, pipeline: *metal_runtime.RawMetalGeneratedPipeline, params: shared.Params, inputs: []const Ref, output: Ref, grid: [3]usize, group: [3]usize) !void {
        if (self.coalesced_frame_enabled and self.deferred_runtime == runtime) {
            if (self.exact_local_window_capacity)
                try kernels.runThreadgroupsPipelineBoundedCoalescedOnActiveRuntime(runtime, self.deferred_submission_epoch, pipeline, params, inputs, output, grid, group, self.providerWriteMask(kernels, params))
            else
                try kernels.runThreadgroupsPipelineCoalescedOnActiveRuntime(runtime, pipeline, params, inputs, output, grid, group, self.providerWriteMask(kernels, params));
            self.planned_dispatch_calls +|= 1;
        } else {
            try kernels.runThreadgroupsPipelineOnActiveRuntime(runtime, pipeline, params, inputs, output, grid, group);
        }
    }

    fn splitStagePipeline(self: *Runner, encoding: StorageEncoding, dim: usize, rows: usize) !*metal_runtime.RawMetalGeneratedPipeline {
        const dim_index: usize = switch (dim) {
            256 => 0,
            512 => 1,
            else => return error.InvalidAttentionShape,
        };
        const rows_index: usize = switch (rows) {
            32 => 0,
            256 => 1,
            else => return error.InvalidAttentionShape,
        };
        const index = @as(usize, @backingInt(encoding)) * 4 + dim_index * 2 + rows_index;
        if (self.split_variants[index]) |pipeline| return pipeline;
        var name_buffer: [128]u8 = undefined;
        const name = try std.fmt.bufPrintSentinel(&name_buffer, "metal_attention_match_split_stage_e{d}_d{d}_r{d}", .{ @backingInt(encoding), dim, rows }, 0);
        var prefix_buffer: [512]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&prefix_buffer, "#define ATTENTION_MATCH_SPECIAL_ENCODING {d}u\n#define ATTENTION_MATCH_SPECIAL_DIM {d}u\n#define ATTENTION_MATCH_SPECIAL_ROWS {d}u\n#define metal_attention_match_split_stage {s}\n", .{ @backingInt(encoding), dim, rows, name });
        const pieces = [_][]const u8{ prefix, @embedFile("metal_attention_match_parallel_split.metal") };
        const source_allocator = if (self.allocation_tracker) |tracker|
            tracker.boundedPrepareHostAllocator() orelse std.heap.page_allocator
        else
            std.heap.page_allocator;
        const source = try std.mem.concat(source_allocator, u8, &pieces);
        defer source_allocator.free(source);
        var message: [4096]u8 = @splat(0);
        const pipeline = metal_runtime.termite_metal_generated_pipeline_create(source.ptr, source.len, name.ptr, 1, &message, message.len) orelse {
            std.log.err("Metal kernel {s}: {s}", .{ name, std.mem.sliceTo(&message, 0) });
            return error.MetalKernelCompilationFailed;
        };
        self.split_variants[index] = pipeline;
        return pipeline;
    }

    fn matrixPartition(encoding: StorageEncoding, dim: usize, window: u32, rows: usize, gqa: usize) ?usize {
        // Each local SIMD group owns one query head; inactive groups leave
        // no records behind. The global matrix tile requires all eight heads.
        if (encoding == .exact and dim == 256 and window == 512 and rows >= 128 and (gqa == 4 or gqa == 8)) return 16;
        if ((encoding == .exact or encoding == .fp8 or encoding == .rotated4) and dim == 512 and window == 0 and rows >= 1024 and (gqa == 4 or gqa == 8))
            return if (rows < 8192) 64 else 128;
        return null;
    }

    fn matrixLayerPartition(heads: []const Head, dim: usize, window: u32, rows: usize, gqa: usize) ?usize {
        // The GQA4 matrix route loads all eight real queries for each KV tile.
        if (dim == 512 and gqa == 4 and heads.len != 2) return null;
        var partition: ?usize = null;
        for (heads) |head| {
            if (head.dim != dim or head.window != window) return null;
            const next = matrixPartition(head.encoding, dim, window, rows, gqa) orelse return null;
            if (partition) |previous| if (previous != next) return null;
            partition = next;
        }
        return partition;
    }

    fn matrixStagePipeline(self: *Runner, encoding: StorageEncoding, dim: usize, rows: usize) !*metal_runtime.RawMetalGeneratedPipeline {
        const index: usize = switch (dim) {
            256 => if (rows == 16 and encoding == .exact) 0 else return error.InvalidAttentionShape,
            512 => (switch (encoding) {
                .exact => @as(usize, 0),
                .fp8 => @as(usize, 2),
                .rotated4 => @as(usize, 4),
            }) + switch (rows) {
                64 => @as(usize, 1),
                128 => @as(usize, 2),
                else => return error.InvalidAttentionShape,
            },
            else => return error.InvalidAttentionShape,
        };
        if (self.matrix_variants[index]) |pipeline| return pipeline;
        // Pipeline creation must happen before opening the owned frame.
        if (self.deferred_runtime != null) return error.ContextStagingBudgetExceeded;
        var name_buffer: [128]u8 = undefined;
        const name = try std.fmt.bufPrintSentinel(&name_buffer, "metal_attention_match_matrix_stage_e{d}_d{d}_r{d}", .{ @backingInt(encoding), dim, rows }, 0);
        var prefix_buffer: [256]u8 = undefined;
        const prefix = try std.fmt.bufPrint(&prefix_buffer, "#define MATRIX_ENCODING {d}u\n#define MATRIX_DIM {d}u\n#define MATRIX_PARTITION {d}u\n#define metal_attention_match_matrix_stage {s}\n", .{ @backingInt(encoding), dim, rows, name });
        const pieces = [_][]const u8{ prefix, if (encoding == .rotated4) @embedFile("metal_attention_match_rotated_matrix.metal") else @embedFile("metal_attention_match_matrix_stage.metal") };
        const source_allocator = if (self.allocation_tracker) |tracker|
            tracker.boundedPrepareHostAllocator() orelse std.heap.page_allocator
        else
            std.heap.page_allocator;
        const source = try std.mem.concat(source_allocator, u8, &pieces);
        defer source_allocator.free(source);
        var message: [4096]u8 = @splat(0);
        const pipeline = metal_runtime.termite_metal_generated_pipeline_create(source.ptr, source.len, name.ptr, 1, &message, message.len) orelse {
            std.log.err("Metal kernel {s}: {s}", .{ name, std.mem.sliceTo(&message, 0) });
            return error.MetalKernelCompilationFailed;
        };
        self.matrix_variants[index] = pipeline;
        return pipeline;
    }

    const split_rows: usize = 256;
    const SplitLayout = struct {
        splits: usize,
        scratch_bytes: usize,
        rows_per_split: usize = split_rows,
        stage_threads: usize = 256,
        matrix: bool = false,
        padded_matrix: bool = false,
    };

    fn splitLayout(request: Request, matrix_enabled: bool) !?SplitLayout {
        if (request.query_rows != 1 or request.heads.len == 0) return null;
        const dim: usize = request.heads[0].dim;
        if (dim != 256 and dim != 512) return null;
        var maximum_rows: usize = 0;
        for (request.heads) |head| {
            if (head.dim != dim) return error.InvalidAttentionShape;
            maximum_rows = @max(maximum_rows, head.rows);
        }
        // Exact local windows need enough independent SIMD groups to keep
        // short decode attention off the serial row loop. Global heads retain
        // their larger partitions so 32K scratch and reduction stay bounded.
        const local = request.heads[0].window != 0;
        const gqa = request.query_heads / request.heads.len;
        const matrix_partition = if (matrix_enabled) matrixLayerPartition(request.heads, dim, request.heads[0].window, maximum_rows, gqa) else null;
        const matrix = matrix_partition != null;
        const padded_matrix = matrix and dim == 512 and gqa == 4;
        const rows_per_split: usize = if (matrix) matrix_partition.? else if (local) 32 else split_rows;
        if (maximum_rows < (if (local) @as(usize, 128) else 1024)) return null;
        const splits = std.math.divCeil(usize, maximum_rows, rows_per_split) catch return error.MetalMemoryLimitExceeded;
        const scratch_heads = if (padded_matrix) try mul(request.query_heads, 2) else request.query_heads;
        const records = try mul(scratch_heads, splits);
        const floats = try mul(records, try add(dim, 2));
        return .{ .splits = splits, .scratch_bytes = try mul(floats, @sizeOf(f32)), .rows_per_split = rows_per_split, .matrix = matrix, .padded_matrix = padded_matrix };
    }

    fn rejectPrivateAliases(self: *Runner, request: Request) !void {
        const authorities = [_]Ref{ self.status.ref(), self.fp8_zero.ref() };
        for ([_]Ref{ request.queries, request.output }) |external| {
            for (authorities) |authority| if (overlaps(external, authority)) return error.InvalidAttentionShape;
            if (self.scratch) |scratch| if (overlaps(external, scratch.ref())) return error.InvalidAttentionShape;
            if (self.rotated_queries) |queries| if (overlaps(external, queries.ref())) return error.InvalidAttentionShape;
            for (self.layer_outputs[0..self.layer_output_count]) |entry|
                if (entry) |buffer| if (overlaps(external, buffer.ref())) return error.InvalidAttentionShape;
        }
        if (request.log_mass) |mass| {
            for (authorities) |authority| if (overlaps(mass, authority)) return error.InvalidAttentionShape;
            if (self.scratch) |scratch| if (overlaps(mass, scratch.ref())) return error.InvalidAttentionShape;
            if (self.rotated_queries) |queries| if (overlaps(mass, queries.ref())) return error.InvalidAttentionShape;
            for (self.layer_outputs[0..self.layer_output_count]) |entry|
                if (entry) |buffer| if (overlaps(mass, buffer.ref())) return error.InvalidAttentionShape;
        }
    }

    fn attendSplitK(self: *Runner, request: Request, layout: SplitLayout, control: Control, frame_runtime: ?*metal_runtime.RawMetalDecodeRuntime) !void {
        if (layout.splits > 2048) return error.InvalidAttentionShape;
        try self.ensureScratch(layout.scratch_bytes);
        const scratch = self.scratch.?;
        if (overlaps(request.queries, scratch.ref()) or overlaps(request.output, scratch.ref()))
            return error.InvalidAttentionShape;
        if (request.log_mass) |mass| if (overlaps(mass, scratch.ref())) return error.InvalidAttentionShape;

        const dim: usize = request.heads[0].dim;
        var rotated_queries: ?Ref = null;
        if (layout.matrix) for (request.heads) |head| {
            if (head.encoding == .rotated4) {
                rotated_queries = try self.ensureRotatedQueries(request.query_heads, dim);
                _ = try self.rotatedQueryPipeline();
                break;
            }
        };
        const rotation_params: shared.Params = .{ .op = 0, .n = @intCast(request.query_heads * dim), .dim = @intCast(dim), .groups = @intCast(request.query_heads) };
        const rotation_inputs = [_]Ref{ request.queries, self.fp8_zero.ref(), self.fp8_zero.ref(), self.fp8_zero.ref(), self.fp8_zero.ref(), self.fp8_zero.ref(), self.status.ref() };
        if (rotated_queries) |queries| try self.split_stage.validateRun(rotation_params, &rotation_inputs, queries);
        const gqa = request.query_heads / request.heads.len;
        const stage_groups = if (layout.matrix) layout.splits else try mul(layout.splits, gqa);
        for (request.heads, 0..) |head, index| {
            _ = if (layout.matrix) try self.matrixStagePipeline(head.encoding, dim, layout.rows_per_split) else try self.splitStagePipeline(head.encoding, dim, layout.rows_per_split);
            const inputs = [_]Ref{ if (layout.matrix and head.encoding == .rotated4) rotated_queries.? else request.queries, refOr(head.keys, self.fp8_zero), refOr(head.values, self.fp8_zero), refOr(head.encoded_keys, self.fp8_zero), refOr(head.encoded_values, self.fp8_zero), try ref(head.metadata), self.status.ref() };
            const packed_stride: u32 = @as(u32, @intCast(index * gqa)) | (@as(u32, @intCast(gqa)) << 16);
            try self.split_stage.validateRun(.{
                .op = if (layout.rows_per_split == 32) 2 else 1,
                .n = @intCast(layout.splits),
                .dim = @intCast(dim),
                .groups = @intCast(request.query_heads),
                .position = @intCast(request.first_query_position),
                .stride = packed_stride,
                .mode = head.window,
                .index = @intCast(head.rows),
                .epsilon = request.softcap,
                .scale = request.scale,
                .theta = @bitCast(@as(u32, @intCast(head.start))),
                .reserved = @bitCast(@as(u32, @intCast(head.capacity))),
            }, &inputs, scratch.ref());
        }
        const mass = request.log_mass orelse self.fp8_zero.ref();
        const reduce_inputs = [_]Ref{ scratch.ref(), mass, self.status.ref(), self.fp8_zero.ref(), self.fp8_zero.ref(), self.fp8_zero.ref(), self.fp8_zero.ref() };
        const reduce_params: shared.Params = .{
            .op = if (layout.padded_matrix) 2 else 1,
            .n = @intCast(request.query_heads),
            .dim = @intCast(dim),
            .mode = @intFromBool(request.log_mass != null),
            .index = @intCast(layout.splits),
        };
        try self.split_reduce.validateRun(reduce_params, &reduce_inputs, request.output);

        // Heads with fewer physical rows do not encode their trailing split
        // records. Clear the persistent buffer before opening the frame so the
        // reducer can treat those records as empty without cross-call residue.
        if (self.deferred_runtime == null) @memset(scratch.bytes[0..layout.scratch_bytes], 0);
        self.resetStatus();
        try control.check();
        if (frame_runtime == null) try self.beginOwnedBatch();
        errdefer if (frame_runtime == null) self.core.cancelBatch();
        if (rotated_queries) |queries| {
            try self.runProviderThreadgroups(&self.split_stage, frame_runtime orelse self.core.runtime, try self.rotatedQueryPipeline(), rotation_params, &rotation_inputs, queries, .{ request.query_heads, 1, 1 }, .{ 32, 1, 1 });
            self.rotated_query_dispatches +|= 1;
        }
        for (request.heads, 0..) |head, index| {
            try control.check();
            const inputs = [_]Ref{ if (layout.matrix and head.encoding == .rotated4) rotated_queries.? else request.queries, refOr(head.keys, self.fp8_zero), refOr(head.values, self.fp8_zero), refOr(head.encoded_keys, self.fp8_zero), refOr(head.encoded_values, self.fp8_zero), try ref(head.metadata), self.status.ref() };
            const packed_stride: u32 = @as(u32, @intCast(index * gqa)) | (@as(u32, @intCast(gqa)) << 16);
            if (layout.matrix) {
                self.matrix_stage_dispatches +|= 1;
                if (@import("builtin").is_test and @import("antfly_platform").env.getenvBool("ANTFLY_CONTEXT_PROFILE"))
                    std.debug.print("attention_match_matrix_stage layer={d} head={d} rows={d} partition={d} kernel={s} encoding={s}\n", .{ head.layer, index, head.rows, layout.rows_per_split, if (dim == 256) "local_simd" else "global_matrix", @tagName(head.encoding) });
            }
            try self.runProviderThreadgroups(
                &self.split_stage,
                frame_runtime orelse self.core.runtime,
                if (layout.matrix) try self.matrixStagePipeline(head.encoding, dim, layout.rows_per_split) else try self.splitStagePipeline(head.encoding, dim, layout.rows_per_split),
                .{
                    .op = if (layout.rows_per_split == 32) 2 else 1,
                    .n = @intCast(layout.splits),
                    .dim = @intCast(dim),
                    .groups = @intCast(request.query_heads),
                    .position = @intCast(request.first_query_position),
                    .stride = packed_stride,
                    .mode = head.window,
                    .index = @intCast(head.rows),
                    .epsilon = request.softcap,
                    .scale = request.scale,
                    .theta = @bitCast(@as(u32, @intCast(head.start))),
                    .reserved = @bitCast(@as(u32, @intCast(head.capacity))),
                },
                &inputs,
                scratch.ref(),
                .{ stage_groups, 1, 1 },
                .{ layout.stage_threads, 1, 1 },
            );
        }
        try control.check();
        try self.runProviderThreadgroups(
            &self.split_reduce,
            frame_runtime orelse self.core.runtime,
            self.split_reduce.pipeline,
            reduce_params,
            &reduce_inputs,
            request.output,
            .{ request.query_heads, 1, 1 },
            .{ 256, 1, 1 },
        );
        if (frame_runtime == null) {
            try self.core.endBatch();
            try self.checkStatus();
        }
        try control.check();
    }

    pub const Append = struct {
        state: *DeviceState,
        layer: u32,
        keys: Ref,
        values: Ref,
        rows: usize,
        first_position: u64,
    };

    /// Consume one active provider frame, including its pending Q/K/V work.
    /// Append and attention remain private until the single terminal wait and
    /// finite-status check. Success and failure both leave that frame drained.
    pub fn attendOnFrame(self: *Runner, runtime: *metal_runtime.RawMetalDecodeRuntime, request: Request, append_request: ?Append, control: Control) !void {
        if (!metal_runtime.hasActiveFrame(runtime) or metal_runtime.hasSubmittedFrame(runtime))
            return error.ContextTransactionBusy;
        errdefer {
            // The deferred owner must sample submitted state before anyone
            // waits/drains the provider. Otherwise a completed submitted frame
            // can look unsubmitted and incorrectly restore only host metadata.
            if (self.deferred_runtime == runtime) {
                self.cancelDeferredFrame();
            } else {
                if (metal_runtime.hasActiveFrame(runtime)) metal_runtime.cancelFrame(runtime) catch {};
                if (metal_runtime.hasSubmittedFrame(runtime)) metal_runtime.waitFrame(runtime) catch {};
            }
        }
        try control.check();
        try self.requireOwnedBatchIdle();
        if (self.deferred_runtime) |deferred| {
            if (deferred != runtime or request.query_rows != 1 or request.query_heads != self.deferred_query_heads or
                request.first_query_position != self.deferred_end - 1 or request.heads.len == 0)
                return error.ContextTransactionBusy;
            const state = self.deferred_state orelse return error.ContextTransactionBusy;
            const owned_heads = try state.layer(request.heads[0].layer);
            if (request.heads.ptr != owned_heads.ptr or request.heads.len != owned_heads.len) return error.InvalidAttentionShape;
            if (append_request) |write| if (write.state != state) return error.InvalidAttentionShape;
            if (append_request) |write|
                try self.appendImpl(write.state, write.layer, write.keys, write.values, write.rows, write.first_position, control, runtime);
            try self.attendImpl(request, control, runtime);
            return;
        }
        // Test-only profiling separates pending model work from storage GPU
        // time. Production keeps the single-frame transaction unchanged.
        const profile = @import("builtin").is_test and self.core.profile.enabled;
        var input_frame_gpu_ns: u64 = 0;
        if (profile and metal_runtime.activeFrameHasWork(runtime)) {
            try metal_runtime.submitFrame(runtime);
            try metal_runtime.waitFrame(runtime);
            input_frame_gpu_ns = metal_runtime.lastFrameGpuNanos(runtime);
            try metal_runtime.beginFrame(runtime);
        }
        if (append_request) |write|
            try self.appendImpl(write.state, write.layer, write.keys, write.values, write.rows, write.first_position, control, runtime);
        try self.attendImpl(request, control, runtime);
        try metal_runtime.submitFrame(runtime);
        try metal_runtime.waitFrame(runtime);
        try self.checkStatus();
        try control.check();
        if (profile) std.debug.print("attention_match_decode_profile {{\"layer\":{d},\"query_rows\":{d},\"query_heads\":{d},\"kv_heads\":{d},\"dim\":{d},\"window\":{d},\"rows_first\":{d},\"encoding\":\"{s}\",\"append\":{},\"input_frame_gpu_ns\":{d},\"storage_frame_gpu_ns\":{d}}}\n", .{
            request.heads[0].layer, request.query_rows,      request.query_heads,                      request.heads.len,
            request.heads[0].dim,   request.heads[0].window, request.heads[0].rows,                    @tagName(request.heads[0].encoding),
            append_request != null, input_frame_gpu_ns,      metal_runtime.lastFrameGpuNanos(runtime),
        });
    }

    pub fn append(self: *Runner, state: *DeviceState, layer_index: u32, keys: Ref, values: Ref, rows: usize, first_position: u64, control: Control) !void {
        return self.appendImpl(state, layer_index, keys, values, rows, first_position, control, null);
    }

    fn appendImpl(self: *Runner, state: *DeviceState, layer_index: u32, keys: Ref, values: Ref, rows: usize, first_position: u64, control: Control, frame_runtime: ?*metal_runtime.RawMetalDecodeRuntime) !void {
        var standalone_exact_submitted = false;
        errdefer {
            if (standalone_exact_submitted) state.invalidated = true;
        }
        try control.check();
        try self.requireOwnedBatchIdle();
        if (state.runner != self or rows == 0 or rows > 512 or first_position != state.logical_length or first_position < state.initial_logical_length)
            return error.InvalidAttentionShape;
        const end = std.math.add(u64, first_position, rows) catch return error.InvalidAttentionShape;
        if (end > checkpoint.max_context_tokens) return error.CompactionAppendLimit;
        const heads = try state.layer(layer_index);
        if (heads.len == 0 or heads.len > 64 or heads[0].dim == 0 or heads[0].dim > 1024) return error.InvalidAttentionShape;
        const dim: usize = heads[0].dim;
        const elements = try mul(try mul(rows, heads.len), dim);
        if (keys.bytes < try mul(elements, 4) or values.bytes < try mul(elements, 4)) return error.InvalidAttentionShape;
        // Validate all heads and aliases before any capacity mutation.
        for (heads, 0..) |head, i| {
            try validateStoredHead(head, first_position, layer_index, i, heads[0].window, checkpoint.max_context_tokens);
            if (head.dim != dim)
                return error.InvalidAttentionShape;
            for ([_]Ref{ refOr(head.keys, self.fp8_zero), refOr(head.values, self.fp8_zero), refOr(head.encoded_keys, self.fp8_zero), refOr(head.encoded_values, self.fp8_zero), try ref(head.metadata) }) |destination|
                if (overlaps(keys, destination) or overlaps(values, destination)) return error.InvalidAttentionShape;
            _ = try add(head.rows, rows);
        }
        for (heads) |*head| {
            const capacity = try appendCapacity(head.*, rows, self.exact_local_window_capacity);
            try self.grow(state, head, capacity, control);
        }
        for (heads, 0..) |head, index| {
            const inputs = [_]Ref{ keys, values, refOr(head.keys, self.fp8_zero), refOr(head.values, self.fp8_zero), refOr(head.encoded_keys, self.fp8_zero), refOr(head.encoded_values, self.fp8_zero), try ref(head.metadata) };
            try self.core.validateRun(.{ .op = 1, .n = @intCast(rows * dim), .dim = @intCast(dim), .groups = @intCast(heads.len), .position = @intCast(first_position), .stride = @intCast(index), .mode = @intCast(head.capacity), .index = @intCast(head.rows), .theta = @bitCast(@as(u32, @intCast(head.start))) }, &inputs, self.status.ref());
        }
        self.resetStatus();
        try control.check();
        if (frame_runtime == null) try self.beginOwnedBatch();
        errdefer if (frame_runtime == null) self.core.cancelBatch();
        for (heads, 0..) |head, index| {
            try control.check();
            const inputs = [_]Ref{ keys, values, refOr(head.keys, self.fp8_zero), refOr(head.values, self.fp8_zero), refOr(head.encoded_keys, self.fp8_zero), refOr(head.encoded_values, self.fp8_zero), try ref(head.metadata) };
            if (frame_runtime) |runtime| {
                try self.runProviderDispatch(&self.core, runtime, .{ .op = 1, .n = @intCast(rows * dim), .dim = @intCast(dim), .groups = @intCast(heads.len), .position = @intCast(first_position), .stride = @intCast(index), .mode = @intCast(head.capacity), .index = @intCast(head.rows), .theta = @bitCast(@as(u32, @intCast(head.start))) }, &inputs, self.status.ref());
            } else {
                try self.core.run(.{ .op = 1, .n = @intCast(rows * dim), .dim = @intCast(dim), .groups = @intCast(heads.len), .position = @intCast(first_position), .stride = @intCast(index), .mode = @intCast(head.capacity), .index = @intCast(head.rows), .theta = @bitCast(@as(u32, @intCast(head.start))) }, &inputs, self.status.ref());
            }
        }
        if (frame_runtime == null) {
            // endBatch may submit before reporting a device/status error. From
            // this boundary through host publication, an exact local ring can
            // no longer be rolled back without restoring overwritten bytes.
            standalone_exact_submitted = self.exact_local_window_capacity and heads[0].window != 0;
            try self.core.endBatch();
            try self.checkStatus();
        }
        try control.check();
        for (heads) |*head| {
            try publishAppend(head, rows, end, first_position, self.exact_local_window_capacity);
        }
        for (state.heads) |head| if (head.logical_length != end) return;
        state.logical_length = end;
    }

    pub fn attend(self: *Runner, request: Request, control: Control) !void {
        return self.attendImpl(request, control, null);
    }

    fn attendImpl(self: *Runner, request: Request, control: Control, frame_runtime: ?*metal_runtime.RawMetalDecodeRuntime) !void {
        try control.check();
        try self.requireOwnedBatchIdle();
        if (request.heads.len == 0 or request.heads.len > 64 or request.query_heads == 0 or request.query_heads > 256 or
            request.query_heads % request.heads.len != 0 or request.query_rows == 0 or request.query_rows > checkpoint.max_context_tokens or
            !std.math.isFinite(request.scale) or request.scale <= 0 or !std.math.isFinite(request.softcap) or request.softcap < 0)
            return error.InvalidAttentionShape;
        const query_end = std.math.add(u64, request.first_query_position, request.query_rows - 1) catch return error.InvalidAttentionShape;
        if (query_end >= checkpoint.max_context_tokens) return error.InvalidAttentionShape;
        const dim: usize = request.heads[0].dim;
        if (dim == 0 or dim > 1024) return error.InvalidAttentionShape;
        const query_count = try mul(request.query_rows, request.query_heads);
        const elements = try mul(query_count, dim);
        if (request.queries.bytes < try mul(elements, 4) or request.output.bytes < try mul(elements, 4)) return error.InvalidAttentionShape;
        if (request.log_mass) |mass| if (mass.bytes < try mul(query_count, 4)) return error.InvalidAttentionShape;
        // Reject aliases against the current private buffers before either
        // path can grow and free scratch. Status and FP8-zero are immutable
        // dispatch authority, not caller workspace.
        try self.rejectPrivateAliases(request);
        const gqa = request.query_heads / request.heads.len;
        for (request.heads, 0..) |head, index| {
            try validateStoredHead(head, request.heads[0].logical_length, request.heads[0].layer, index, request.heads[0].window, checkpoint.max_context_tokens);
            try validateRetainedQueryCoverage(head, request.first_query_position, request.query_rows, self.exact_local_window_capacity);
            if (head.dim != dim) return error.InvalidAttentionShape;
            for ([_]Ref{ refOr(head.keys, self.fp8_zero), refOr(head.values, self.fp8_zero), refOr(head.encoded_keys, self.fp8_zero), refOr(head.encoded_values, self.fp8_zero), try ref(head.metadata) }) |source| {
                if (overlaps(request.output, source) or overlaps(request.queries, source)) return error.InvalidAttentionShape;
                if (request.log_mass) |mass| if (overlaps(mass, source)) return error.InvalidAttentionShape;
            }
        }
        if (overlaps(request.queries, request.output)) return error.InvalidAttentionShape;
        if (request.log_mass) |mass| if (overlaps(mass, request.queries) or overlaps(mass, request.output)) return error.InvalidAttentionShape;
        if (try splitLayout(request, self.matrix_stage_enabled)) |layout| return self.attendSplitK(request, layout, control, frame_runtime);
        const chunk_limit: usize = 16;
        const scratch_rows = @min(request.query_rows, chunk_limit);
        try self.ensureScratch(try mul(try mul(try mul(scratch_rows, request.query_heads), try add(dim, 2)), 4));
        const query_row_bytes = try mul(try mul(request.query_heads, dim), 4);
        const mass_row_bytes = try mul(request.query_heads, 4);
        var offset: usize = 0;
        while (offset < request.query_rows) {
            try control.check();
            const rows = @min(chunk_limit, request.query_rows - offset);
            const chunk_queries = try request.queries.slice(try mul(offset, query_row_bytes), try mul(rows, query_row_bytes));
            const chunk_output = try request.output.slice(try mul(offset, query_row_bytes), try mul(rows, query_row_bytes));
            const position = std.math.add(u64, request.first_query_position, @intCast(offset)) catch return error.InvalidAttentionShape;
            for (request.heads, 0..) |head, index| {
                const inputs = [_]Ref{ chunk_queries, self.scratch.?.ref(), refOr(head.keys, self.fp8_zero), refOr(head.values, self.fp8_zero), refOr(head.encoded_keys, self.fp8_zero), refOr(head.encoded_values, self.fp8_zero), try ref(head.metadata) };
                const packed_stride: u32 = @as(u32, @intCast(index * gqa)) | (@as(u32, @intCast(gqa)) << 16);
                const dispatch_threads = try mul(try mul(rows, gqa), 32);
                try self.core.validateRun(.{ .op = 2, .n = @intCast(dispatch_threads), .dim = @intCast(dim), .groups = @intCast(request.query_heads), .position = @intCast(position), .stride = packed_stride, .mode = head.window, .index = @intCast(head.rows), .epsilon = request.softcap, .scale = request.scale, .theta = @bitCast(@as(u32, @intCast(head.start))), .reserved = @bitCast(@as(u32, @intCast(head.capacity))) }, &inputs, self.status.ref());
            }
            const copy_inputs = [_]Ref{self.scratch.?.ref()};
            const chunk_elements = try mul(try mul(rows, request.query_heads), dim);
            try self.copier.validateRun(.{ .op = 3, .n = @intCast(chunk_elements), .dim = @intCast(dim) }, &copy_inputs, chunk_output);
            const chunk_mass = if (request.log_mass) |mass|
                try mass.slice(try mul(offset, mass_row_bytes), try mul(rows, mass_row_bytes))
            else
                null;
            if (chunk_mass) |mass|
                try self.copier.validateRun(.{ .op = 4, .n = @intCast(try mul(rows, request.query_heads)), .dim = @intCast(dim) }, &copy_inputs, mass);
            self.resetStatus();
            try control.check();
            if (frame_runtime == null) try self.beginOwnedBatch();
            errdefer if (frame_runtime == null) self.core.cancelBatch();
            for (request.heads, 0..) |head, index| {
                const inputs = [_]Ref{ chunk_queries, self.scratch.?.ref(), refOr(head.keys, self.fp8_zero), refOr(head.values, self.fp8_zero), refOr(head.encoded_keys, self.fp8_zero), refOr(head.encoded_values, self.fp8_zero), try ref(head.metadata) };
                const packed_stride: u32 = @as(u32, @intCast(index * gqa)) | (@as(u32, @intCast(gqa)) << 16);
                const dispatch_threads = try mul(try mul(rows, gqa), 32);
                if (frame_runtime) |runtime| {
                    try self.runProviderDispatch(&self.core, runtime, .{ .op = 2, .n = @intCast(dispatch_threads), .dim = @intCast(dim), .groups = @intCast(request.query_heads), .position = @intCast(position), .stride = packed_stride, .mode = head.window, .index = @intCast(head.rows), .epsilon = request.softcap, .scale = request.scale, .theta = @bitCast(@as(u32, @intCast(head.start))), .reserved = @bitCast(@as(u32, @intCast(head.capacity))) }, &inputs, self.status.ref());
                } else {
                    try self.core.run(.{ .op = 2, .n = @intCast(dispatch_threads), .dim = @intCast(dim), .groups = @intCast(request.query_heads), .position = @intCast(position), .stride = packed_stride, .mode = head.window, .index = @intCast(head.rows), .epsilon = request.softcap, .scale = request.scale, .theta = @bitCast(@as(u32, @intCast(head.start))), .reserved = @bitCast(@as(u32, @intCast(head.capacity))) }, &inputs, self.status.ref());
                }
            }
            try self.runProviderDispatch(&self.copier, frame_runtime orelse self.core.runtime, .{ .op = 3, .n = @intCast(chunk_elements), .dim = @intCast(dim) }, &copy_inputs, chunk_output);
            if (chunk_mass) |mass|
                try self.runProviderDispatch(&self.copier, frame_runtime orelse self.core.runtime, .{ .op = 4, .n = @intCast(try mul(rows, request.query_heads)), .dim = @intCast(dim) }, &copy_inputs, mass);
            if (frame_runtime == null) {
                try self.core.endBatch();
                try self.checkStatus();
            }
            offset += rows;
            try control.check();
        }
    }
};

test "attention_match Metal runner rejects a bound below its shared baseline" {
    try std.testing.expectError(error.MetalMemoryLimitExceeded, admitRunnerBaseline(6 * std.heap.pageSize() - 1, null));
    try admitRunnerBaseline(6 * std.heap.pageSize(), null);
}

test "attention_match Metal split K layout bounds ragged E4B heads" {
    const heads = [_]Head{
        .{ .dim = 256, .rows = 1024 },
        .{ .dim = 256, .rows = 8192 },
    };
    const layout = (try Runner.splitLayout(.{
        .heads = &heads,
        .queries = undefined,
        .output = undefined,
        .query_rows = 1,
        .query_heads = 8,
        .first_query_position = 0,
        .scale = 1,
    }, false)).?;
    try std.testing.expectEqual(@as(usize, 32), layout.splits);
    try std.testing.expectEqual(@as(usize, 8 * 32 * 258 * 4), layout.scratch_bytes);
    var small = heads;
    small[0].rows = 255;
    small[1].rows = 257;
    try std.testing.expect((try Runner.splitLayout(.{ .heads = &small, .queries = undefined, .output = undefined, .query_rows = 1, .query_heads = 8, .first_query_position = 0, .scale = 1 }, false)) == null);
    small[1].rows = 1024;
    small[1].dim = 512;
    try std.testing.expectError(error.InvalidAttentionShape, Runner.splitLayout(.{ .heads = &small, .queries = undefined, .output = undefined, .query_rows = 1, .query_heads = 8, .first_query_position = 0, .scale = 1 }, false));
}

test "attention_match Metal GQA4 global scratch isolates owners and rejects incomplete query tiles" {
    const heads = [_]Head{ .{ .dim = 512, .rows = 4097 }, .{ .dim = 512, .rows = 1025 } };
    var request: Request = .{ .heads = &heads, .queries = undefined, .output = undefined, .query_rows = 1, .query_heads = 8, .first_query_position = 4097, .scale = 1 };
    const layout = (try Runner.splitLayout(request, true)).?;
    try std.testing.expect(layout.matrix and layout.padded_matrix);
    try std.testing.expectEqual(@as(usize, 65), layout.splits);
    try std.testing.expectEqual(@as(usize, 16 * 65 * 514 * 4), layout.scratch_bytes);
    request.heads = heads[0..1];
    request.query_heads = 4;
    try std.testing.expect(!(try Runner.splitLayout(request, true)).?.matrix);
    request.heads = &heads;
    request.query_heads = 16;
    const gqa8 = (try Runner.splitLayout(request, true)).?;
    try std.testing.expect(gqa8.matrix and !gqa8.padded_matrix);
    try std.testing.expectEqual(layout.scratch_bytes, gqa8.scratch_bytes);
}

test "attention_match Metal storage preflight accounts f16 and resident FP8 exactly" {
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = 2,
        .tokens = &.{ 0, 1 },
        .logits = &.{ 0, 1 },
        .layer_donors = &.{0},
        .donors = &.{.{ .layer = 0, .window = 2, .heads = &.{.{ .dim = 1, .positions = &.{ 0, 1 }, .coverage_ends = &.{ 1, 2 }, .keys = &.{ 0, 1 }, .values = &.{ -2, 3 }, .bias = &.{ 0, 0 } }} }},
    };
    const page = std.heap.pageSize();
    try std.testing.expectEqual(3 * page, try allocationBytesWithControl(state, .{}, .{}));
    try std.testing.expectEqual(0, try globalAllocationBytesWithControl(state, .{}, .{}));
    var changed = state;
    var donor = state.donors[0];
    var head = donor.heads[0];
    head.keys = &.{ 0, 0.1 };
    donor.heads = (&head)[0..1];
    changed.donors = (&donor)[0..1];
    try std.testing.expectError(error.KvStoragePrecisionMismatch, allocationBytesWithControl(changed, .{}, .{}));
    head = state.donors[0].heads[0];
    head.compacted_until = 1;
    head.fp8 = .{ .first_row = 0, .rows = 1 };
    donor.window = 0;
    donor.heads = (&head)[0..1];
    changed.donors = (&donor)[0..1];
    try std.testing.expectEqual(5 * page, try allocationBytesWithControl(changed, .{}, .{}));
    try std.testing.expectEqual(5 * page, try globalAllocationBytesWithControl(changed, .{}, .{}));
}

test "attention_match Metal FP8 preflight cancellation is chunk bounded" {
    const rows = 5000;
    var positions: [rows]u64 = undefined;
    var coverage: [rows]u64 = undefined;
    const tokens: [rows]u32 = @splat(0);
    const keys: [rows]f32 = @splat(0);
    const values: [rows]f32 = @splat(0);
    const bias: [rows]f32 = @splat(0);
    for (&positions, &coverage, 0..) |*position, *end, i| {
        position.* = i;
        end.* = i + 1;
    }
    var head: checkpoint.Head = .{ .dim = 1, .compacted_until = rows, .positions = &positions, .coverage_ends = &coverage, .keys = &keys, .values = &values, .bias = &bias };
    const donor: checkpoint.Donor = .{ .heads = (&head)[0..1] };
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = rows,
        .tokens = &tokens,
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = (&donor)[0..1],
    };
    const Counter = struct {
        calls: usize = 0,
        fail_at: usize = std.math.maxInt(usize),
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls == self.fail_at) return error.CompactionCancelled;
        }
        fn control(self: *@This()) Control {
            return .{ .context = self, .check_fn = check };
        }
    };
    var exact_counter: Counter = .{};
    _ = try allocationBytesWithControl(state, .{}, exact_counter.control());
    head.fp8 = .{ .first_row = 0, .rows = rows };
    var fp8_counter: Counter = .{};
    _ = try allocationBytesWithControl(state, .{}, fp8_counter.control());
    try std.testing.expectEqual(exact_counter.calls + 4, fp8_counter.calls);
    var cancel: Counter = .{ .fail_at = exact_counter.calls };
    try std.testing.expectError(error.CompactionCancelled, allocationBytesWithControl(state, .{}, cancel.control()));
}

test "attention_match ordinary Metal global allocation combines GQA heads without fitted metadata" {
    const head: checkpoint.Head = .{
        .dim = 1,
        .positions = &.{ 0, 1 },
        .coverage_ends = &.{ 1, 2 },
        .keys = &.{ 0, 1 },
        .values = &.{ -2, 3 },
        .bias = &.{ 0, 0 },
    };
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = 2,
        .tokens = &.{ 0, 1 },
        .logits = &.{ 0, 1 },
        .layer_donors = &.{0},
        .donors = &.{.{ .layer = 0, .heads = &.{ head, head } }},
    };
    // Ordinary storage has one combined-head K buffer and one V buffer.
    try std.testing.expectEqual(@as(usize, 16), try ordinaryGlobalAllocationBytesWithControl(state, .{}, .{}));
    // The installed shared-buffer representation explicitly page-rounds
    // K/V/metadata allocations for every head.
    try std.testing.expectEqual(6 * std.heap.pageSize(), try globalAllocationBytesWithControl(state, .{}, .{}));
}

test "attention_match Metal storage admits canonical rotated4 bytes and excludes them from ordinary f16 accounting" {
    const allocator = std.testing.allocator;
    var keys: [3 * 64]f32 = @splat(0.25);
    var values: [3 * 64]f32 = @splat(2);
    var first = try (checkpoint.packed_storage.Packed{}).append(allocator, 0, 64, keys[0..64], values[0..64], .{});
    defer first.deinit(allocator);
    var encoded = try first.append(allocator, 2, 64, keys[128..192], values[128..192], .{});
    defer encoded.deinit(allocator);
    try encoded.decodeRow(64, 0, keys[0..64], values[0..64]);
    try encoded.decodeRow(64, 1, keys[128..192], values[128..192]);
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = 3,
        .tokens = &.{ 0, 0, 0 },
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &.{.{ .heads = &.{.{
            .dim = 64,
            .compacted_until = 3,
            .positions = &.{ 0, 1, 2 },
            .coverage_ends = &.{ 1, 2, 3 },
            .keys = &keys,
            .values = &values,
            .bias = &.{ 0, 0, 0 },
            .compressed = encoded,
        }} }},
    };
    try std.testing.expectEqual(
        try rotatedBufferBytes(3, 64, 2, 2),
        try allocationBytesWithControl(state, .{}, .{}),
    );
    try std.testing.expectError(error.UnsupportedMetalKvCheckpointEncoding, ordinaryGlobalAllocationBytesWithControl(state, .{}, .{}));
}

fn attentionOracle(
    heads: []const checkpoint.Head,
    queries: []const f32,
    query_rows: usize,
    query_heads: usize,
    first_query_position: u64,
    window: u32,
    scale: f32,
    softcap: f32,
    output: []f64,
    log_mass: []f64,
) !void {
    const dim: usize = heads[0].dim;
    const gqa = query_heads / heads.len;
    for (0..query_rows) |query_row| for (0..query_heads) |query_head| {
        const head = heads[query_head / gqa];
        const query_position = std.math.add(u64, first_query_position, @intCast(query_row)) catch return error.InvalidAttentionShape;
        const query = queries[(query_row * query_heads + query_head) * dim ..][0..dim];
        var maximum = -std.math.inf(f64);
        var denominator: f64 = 0;
        const destination = output[(query_row * query_heads + query_head) * dim ..][0..dim];
        @memset(destination, 0);
        for (head.positions, head.coverage_ends, head.bias, 0..) |position, coverage_end, bias, row| {
            if (position > query_position or coverage_end == 0 or coverage_end > query_position + 1) continue;
            if (window != 0 and query_position - position >= @as(u64, window)) continue;
            var dot: f64 = 0;
            for (query, head.keys[row * dim ..][0..dim]) |q, key| dot += @as(f64, q) * key;
            var score = dot * scale;
            if (softcap > 0) score = @as(f64, softcap) * std.math.tanh(score / softcap);
            score += bias;
            const next_maximum = @max(maximum, score);
            const previous_scale = @exp(maximum - next_maximum);
            const weight = @exp(score - next_maximum);
            denominator = denominator * previous_scale + weight;
            for (destination, head.values[row * dim ..][0..dim]) |*value, source|
                value.* = value.* * previous_scale + weight * source;
            maximum = next_maximum;
        }
        if (denominator <= 0) return error.InvalidAttentionShape;
        for (destination) |*value| value.* /= denominator;
        log_mass[query_row * query_heads + query_head] = @log(denominator) + maximum;
    };
}

test "attention_match Metal f16 FP8 GQA bias causal append clone checkpoint cycles" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const positions = [_]u64{ 0, 1, 2 };
    const coverage = [_]u64{ 1, 2, 3 };
    const fp8: checkpoint.codec.Fp8 = .{ .first_row = 1, .rows = 1 };
    const keys0 = [_]f32{ 1, 0, 2, 0, 3, 0 };
    const values0 = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const keys1 = [_]f32{ 0, 1, 0, 2, 0, 3 };
    const values1 = [_]f32{ -1, 1, -2, 2, -3, 3 };
    const bias0 = [_]f32{ 0.5, -0.25, 0 };
    const bias1 = [_]f32{ -0.5, 0.25, 0 };
    const heads = [_]checkpoint.Head{
        .{ .dim = 2, .compacted_until = 2, .positions = &positions, .coverage_ends = &coverage, .keys = &keys0, .values = &values0, .bias = &bias0, .fp8 = fp8 },
        .{ .dim = 2, .compacted_until = 2, .positions = &positions, .coverage_ends = &coverage, .keys = &keys1, .values = &values1, .bias = &bias1, .fp8 = fp8 },
    };
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = 3,
        .tokens = &.{ 0, 0, 0 },
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &.{.{ .layer = 0, .heads = &heads }},
    };
    var runner = try Runner.init(64 * 1024 * 1024, null);
    defer runner.deinit();
    const backend = runner.backend(std.testing.allocator, .{});
    const allocation = try backend.allocationBytes(state);
    try std.testing.expectError(error.MetalMemoryLimitExceeded, backend.upload(state, allocation - 1, .{}));
    var device = try backend.upload(state, 64 * 1024 * 1024, .{});
    defer device.deinit();

    const queries = [_]f32{
        1,    0, 0.5,  0, 0, 1,    0, 0.5,
        0.25, 0, -0.5, 0, 0, 0.25, 0, -0.5,
    };
    var query = try runner.core.buffer(queries.len * 4);
    defer query.deinit();
    @memcpy(query.floats(), &queries);
    var output = try runner.core.buffer(queries.len * 4);
    defer output.deinit();
    var mass = try runner.core.buffer(8 * 4);
    defer mass.deinit();
    const attend_core_before = runner.core.batchCounters();
    const attend_copy_before = runner.copier.batchCounters();
    try runner.attend(.{ .heads = try device.layer(0), .queries = query.ref(), .output = output.ref(), .log_mass = mass.ref(), .query_rows = 2, .query_heads = 4, .first_query_position = 1, .scale = 0.5, .softcap = 0.75 }, .{});
    const attend_core = runner.core.batchCounters();
    const attend_copy = runner.copier.batchCounters();
    try std.testing.expectEqual(attend_core_before.submissions + 1, attend_core.submissions);
    try std.testing.expectEqual(attend_core_before.waits + 1, attend_core.waits);
    try std.testing.expectEqual(attend_copy_before.submissions, attend_copy.submissions);
    try std.testing.expectEqual(attend_copy_before.waits, attend_copy.waits);
    try std.testing.expect(!attend_core.active and !attend_core.submitted);
    var expected: [16]f64 = undefined;
    var expected_mass: [8]f64 = undefined;
    try attentionOracle(&heads, &queries, 2, 4, 1, 0, 0.5, 0.75, &expected, &expected_mass);
    for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 3e-5);
    for (expected_mass, mass.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 3e-5);
    try std.testing.expectError(error.InvalidAttentionShape, runner.attend(.{ .heads = try device.layer(0), .queries = query.ref(), .output = output.ref(), .query_rows = 1, .query_heads = 257, .first_query_position = 1, .scale = 1 }, .{}));

    const saved_capacity = device.heads[0].capacity;
    device.heads[0].capacity = 0;
    try std.testing.expectError(error.InvalidKvCheckpoint, device.download(std.testing.allocator, state, .{}, 1024 * 1024, .{}));
    device.heads[0].capacity = saved_capacity;

    var chunk_queries = try runner.core.buffer(17 * 4 * 2 * 4);
    defer chunk_queries.deinit();
    @memset(chunk_queries.floats(), 0);
    var chunk_output = try runner.core.buffer(17 * 4 * 2 * 4);
    defer chunk_output.deinit();
    @memset(chunk_output.floats(), -99);
    const ChunkCancel = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls == 4) return error.CompactionCancelled;
        }
    };
    const chunk_core_before = runner.core.batchCounters();
    const chunk_copy_before = runner.copier.batchCounters();
    var chunk_cancel: ChunkCancel = .{};
    try std.testing.expectError(error.CompactionCancelled, runner.attend(.{ .heads = try device.layer(0), .queries = chunk_queries.ref(), .output = chunk_output.ref(), .query_rows = 17, .query_heads = 4, .first_query_position = 0, .scale = 1 }, .{ .context = &chunk_cancel, .check_fn = ChunkCancel.check }));
    for (chunk_output.floats()[16 * 4 * 2 ..]) |value| try std.testing.expectEqual(@as(f32, -99), value);
    const chunk_core = runner.core.batchCounters();
    const chunk_copy = runner.copier.batchCounters();
    try std.testing.expectEqual(chunk_core_before.submissions + 1, chunk_core.submissions);
    try std.testing.expectEqual(chunk_core_before.waits + 1, chunk_core.waits);
    try std.testing.expectEqual(chunk_copy_before.submissions, chunk_copy.submissions);
    try std.testing.expectEqual(chunk_copy_before.waits, chunk_copy.waits);
    try std.testing.expect(!chunk_core.active and !chunk_core.submitted);

    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.CompactionCancelled;
        }
    };
    try std.testing.expectError(error.CompactionCancelled, device.clone(std.testing.allocator, 64 * 1024 * 1024, .{ .check_fn = Cancel.check }));

    const appended_keys0 = [_]f32{ 4, 0, 0, 4 };
    const appended_values0 = [_]f32{ 7, 8, -4, 4 };
    var append_keys0 = try runner.core.buffer(appended_keys0.len * 4);
    defer append_keys0.deinit();
    var append_values0 = try runner.core.buffer(appended_values0.len * 4);
    defer append_values0.deinit();
    @memcpy(append_keys0.floats(), &appended_keys0);
    @memcpy(append_values0.floats(), &appended_values0);
    const AppendCancel = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            // initial, two grow checks, pre-batch, first-head, second-head
            if (self.calls == 6) return error.CompactionCancelled;
        }
    };
    const logical_before_cancel = device.logical_length;
    const rows_before_cancel = device.heads[0].rows;
    const cancel_core_before = runner.core.batchCounters();
    var append_cancel: AppendCancel = .{};
    try std.testing.expectError(
        error.CompactionCancelled,
        runner.append(&device, 0, append_keys0.ref(), append_values0.ref(), 1, 3, .{ .context = &append_cancel, .check_fn = AppendCancel.check }),
    );
    const cancel_core = runner.core.batchCounters();
    try std.testing.expectEqual(cancel_core_before.submissions, cancel_core.submissions);
    try std.testing.expectEqual(cancel_core_before.waits, cancel_core.waits);
    try std.testing.expect(!cancel_core.active and !cancel_core.submitted);
    try std.testing.expectEqual(logical_before_cancel, device.logical_length);
    for (device.heads) |head| {
        try std.testing.expectEqual(rows_before_cancel, head.rows);
        try std.testing.expectEqual(logical_before_cancel, head.logical_length);
    }
    const append_core_before = runner.core.batchCounters();
    const append_copy_before = runner.copier.batchCounters();
    try runner.append(&device, 0, append_keys0.ref(), append_values0.ref(), 1, 3, .{});
    const append_core = runner.core.batchCounters();
    const append_copy = runner.copier.batchCounters();
    try std.testing.expectEqual(append_core_before.submissions + 1, append_core.submissions);
    try std.testing.expectEqual(append_core_before.waits + 1, append_core.waits);
    try std.testing.expectEqual(append_copy_before.submissions, append_copy.submissions);
    try std.testing.expectEqual(append_copy_before.waits, append_copy.waits);

    var destination = try Runner.init(64 * 1024 * 1024, null);
    defer destination.deinit();
    var cycle1 = try device.cloneInto(std.testing.allocator, &destination, 64 * 1024 * 1024, .{});
    defer cycle1.deinit();
    const metadata4: checkpoint.State = .{ .identity = state.identity, .logical_length = 4, .tokens = &.{ 0, 0, 0, 0 }, .logits = &.{0}, .layer_donors = &.{0}, .donors = state.donors };
    var downloaded1 = try cycle1.download(std.testing.allocator, metadata4, .{}, 1024 * 1024, .{});
    defer downloaded1.deinit();
    const expected_keys0 = [_]f32{ 1, 0, 2, 0, 3, 0, 4, 0 };
    const expected_values0 = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const expected_keys1 = [_]f32{ 0, 1, 0, 2, 0, 3, 0, 4 };
    const expected_values1 = [_]f32{ -1, 1, -2, 2, -3, 3, -4, 4 };
    try std.testing.expectEqualDeep(fp8, downloaded1.state.donors[0].heads[0].fp8);
    try std.testing.expectEqualSlices(f32, &expected_keys0, downloaded1.state.donors[0].heads[0].keys);
    try std.testing.expectEqualSlices(f32, &expected_values0, downloaded1.state.donors[0].heads[0].values);
    try std.testing.expectEqualSlices(f32, &expected_keys1, downloaded1.state.donors[0].heads[1].keys);
    try std.testing.expectEqualSlices(f32, &expected_values1, downloaded1.state.donors[0].heads[1].values);

    const appended_keys1 = [_]f32{ 5, 0, 0, 5 };
    const appended_values1 = [_]f32{ 9, 10, -5, 5 };
    var append_keys1 = try destination.core.buffer(appended_keys1.len * 4);
    defer append_keys1.deinit();
    var append_values1 = try destination.core.buffer(appended_values1.len * 4);
    defer append_values1.deinit();
    @memcpy(append_keys1.floats(), &appended_keys1);
    @memcpy(append_values1.floats(), &appended_values1);
    try destination.append(&cycle1, 0, append_keys1.ref(), append_values1.ref(), 1, 4, .{});
    var cycle2 = try cycle1.cloneInto(std.testing.allocator, &runner, 64 * 1024 * 1024, .{});
    defer cycle2.deinit();
    const metadata5: checkpoint.State = .{ .identity = state.identity, .logical_length = 5, .tokens = &.{ 0, 0, 0, 0, 0 }, .logits = &.{0}, .layer_donors = &.{0}, .donors = state.donors };
    var downloaded2 = try cycle2.download(std.testing.allocator, metadata5, .{}, 1024 * 1024, .{});
    defer downloaded2.deinit();
    try std.testing.expectEqualDeep(fp8, downloaded2.state.donors[0].heads[0].fp8);
    try std.testing.expectEqualSlices(f32, &.{ 1, 0, 2, 0, 3, 0, 4, 0, 5, 0 }, downloaded2.state.donors[0].heads[0].keys);
    try std.testing.expectEqualSlices(f32, &.{ -1, 1, -2, 2, -3, 3, -4, 4, -5, 5 }, downloaded2.state.donors[0].heads[1].values);
    // Each tiny head still occupies five independent pages. Encoding bytes
    // alone cannot establish a physical saving for this fixture.
    try std.testing.expectEqual(10 * std.heap.pageSize(), cycle2.allocated_bytes);
}

test "device candidate builder publishes complete state and abort leaves canonical source unchanged" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const positions = [_]u64{ 0, 1 };
    const coverage = [_]u64{ 1, 2 };
    const keys = [_]f32{ 1, 0, 0, 1 };
    const values = [_]f32{ 2, 3, 4, 5 };
    const bias = [_]f32{ 0, 0 };
    const head: checkpoint.Head = .{ .dim = 2, .positions = &positions, .coverage_ends = &coverage, .keys = &keys, .values = &values, .bias = &bias };
    const state: checkpoint.State = .{ .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) }, .logical_length = 2, .tokens = &.{ 0, 0 }, .logits = &.{0}, .layer_donors = &.{0}, .donors = &.{.{ .layer = 0, .heads = &.{head} }} };
    var runner = try Runner.init(16 * 1024 * 1024, null);
    defer runner.deinit();
    const backend = runner.backend(std.testing.allocator, .{});
    var canonical = try backend.upload(state, 8 * 1024 * 1024, .{});
    defer canonical.deinit();
    const canonical_bytes = canonical.allocated_bytes;
    const canonical_rows = canonical.heads[0].rows;
    const descriptors = [_]CandidateHeadDescriptor{.{ .layer = 0, .head = 0, .window = 0 }};
    {
        var aborted = try CandidateBuilder.init(backend, 2, &descriptors, @splat(7), @splat(8), 8 * 1024 * 1024, .{});
        defer aborted.deinit();
        try aborted.uploadHead(0, head, .{});
        try std.testing.expectError(error.InvalidTrainingAugmentationBinding, aborted.finish(@splat(7), @splat(9), .{}));
    }
    try std.testing.expectEqual(canonical_bytes, canonical.allocated_bytes);
    try std.testing.expectEqual(canonical_rows, canonical.heads[0].rows);
    var builder = try CandidateBuilder.init(backend, 2, &descriptors, @splat(7), @splat(8), 8 * 1024 * 1024, .{});
    defer builder.deinit();
    try builder.uploadHead(0, head, .{});
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, builder.uploadHead(0, head, .{}));
    var candidate = try builder.finish(@splat(7), @splat(8), .{});
    defer candidate.deinit();
    try std.testing.expectEqual(@as(usize, 1), candidate.heads.len);
    try std.testing.expectEqual(@as(usize, 2), candidate.heads[0].rows);
    try std.testing.expectEqual(canonical_rows, canonical.heads[0].rows);

    const fp8_head: checkpoint.Head = .{
        .dim = 2,
        .positions = &positions,
        .coverage_ends = &coverage,
        .keys = &keys,
        .values = &values,
        .bias = &bias,
        .fp8 = .{ .first_row = 0, .rows = 2, .key_scale = 1, .value_scale = 1 },
    };
    var fp8_builder = try CandidateBuilder.init(backend, 2, &descriptors, @splat(9), @splat(10), 8 * 1024 * 1024, .{});
    defer fp8_builder.deinit();
    try fp8_builder.uploadHead(0, fp8_head, .{});
    var fp8_candidate = try fp8_builder.finish(@splat(9), @splat(10), .{});
    defer fp8_candidate.deinit();
    var fp8_download = try fp8_candidate.downloadHead(std.testing.allocator, 0, 0, 0, 1024 * 1024, .{});
    defer fp8_download.deinit();
    try std.testing.expectEqualDeep(fp8_head.fp8, fp8_download.head().fp8);
    try std.testing.expectEqualSlices(f32, fp8_head.keys, fp8_download.head().keys);
    try std.testing.expectEqualSlices(f32, fp8_head.values, fp8_download.head().values);

    const rotated_dim = 64;
    const rotated_keys: [2 * rotated_dim]f32 = @splat(0.25);
    const rotated_values: [2 * rotated_dim]f32 = @splat(0.5);
    var encoded = try (packed_storage.Packed{}).append(std.testing.allocator, 0, rotated_dim, rotated_keys[0..rotated_dim], rotated_values[0..rotated_dim], .{});
    defer encoded.deinit(std.testing.allocator);
    const rotated_head: checkpoint.Head = .{
        .dim = rotated_dim,
        .positions = &positions,
        .coverage_ends = &coverage,
        .keys = &rotated_keys,
        .values = &rotated_values,
        .bias = &bias,
        .compressed = encoded,
    };
    var rotated_builder = try CandidateBuilder.init(backend, 2, &descriptors, @splat(11), @splat(12), 8 * 1024 * 1024, .{});
    defer rotated_builder.deinit();
    try rotated_builder.uploadHead(0, rotated_head, .{});
    var rotated_candidate = try rotated_builder.finish(@splat(11), @splat(12), .{});
    defer rotated_candidate.deinit();
    var rotated_download = try rotated_candidate.downloadHead(std.testing.allocator, 0, 0, 0, 1024 * 1024, .{});
    defer rotated_download.deinit();
    try std.testing.expectEqualSlices(packed_storage.Span, rotated_head.compressed.spans, rotated_download.head().compressed.spans);
    try std.testing.expectEqualSlices(u8, rotated_head.compressed.keys, rotated_download.head().compressed.keys);
    try std.testing.expectEqualSlices(u8, rotated_head.compressed.values, rotated_download.head().compressed.values);
}

test "attention_match Metal FP8 reduces actual page-rounded residency while protecting exact rows" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const rows = 512;
    const dim = 512;
    const keys = try allocator.alloc(f32, rows * dim);
    defer allocator.free(keys);
    const values = try allocator.alloc(f32, rows * dim);
    defer allocator.free(values);
    @memset(keys, 1);
    @memset(values, 2);
    var positions: [rows]u64 = undefined;
    var coverage: [rows]u64 = undefined;
    for (&positions, &coverage, 0..) |*position, *end, i| {
        position.* = i;
        end.* = i + 1;
    }
    const bias: [rows]f32 = @splat(0);
    const tokens: [rows]u32 = @splat(0);
    var head: checkpoint.Head = .{
        .dim = dim,
        .compacted_until = rows - 64,
        .positions = &positions,
        .coverage_ends = &coverage,
        .keys = keys,
        .values = values,
        .bias = &bias,
    };
    var donor: checkpoint.Donor = .{ .heads = (&head)[0..1] };
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = rows,
        .tokens = &tokens,
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = (&donor)[0..1],
    };
    var runner = try Runner.init(64 * 1024 * 1024, null);
    defer runner.deinit();
    const backend = runner.backend(allocator, .{});
    var exact = try backend.upload(state, 64 * 1024 * 1024, .{});
    defer exact.deinit();
    head.fp8 = .{ .first_row = 64, .rows = rows - 128 };
    var encoded = try backend.upload(state, 64 * 1024 * 1024, .{});
    defer encoded.deinit();
    const stored = encoded.heads[0];
    const actual_backing = stored.keys.?.bytes.len + stored.values.?.bytes.len +
        stored.encoded_keys.?.bytes.len + stored.encoded_values.?.bytes.len + stored.metadata.?.bytes.len;
    try std.testing.expectEqual(encoded.allocated_bytes, actual_backing);
    try std.testing.expect(encoded.allocated_bytes * 3 <= exact.allocated_bytes * 2);
    var restored = try encoded.download(allocator, state, .{}, 16 * 1024 * 1024, .{});
    defer restored.deinit();
    try std.testing.expectEqualDeep(head.fp8, restored.state.donors[0].heads[0].fp8);
    try std.testing.expectEqualSlices(f32, keys, restored.state.donors[0].heads[0].keys);
    try std.testing.expectEqualSlices(f32, values, restored.state.donors[0].heads[0].values);
}

fn realShapeFp8AttentionDifferential(dim: usize) !void {
    const allocator = std.testing.allocator;
    const rows = 8;
    const kv_heads = 2;
    const query_heads = 8;
    const query_rows = 3;
    const first_query_position = 2;
    const scale: f32 = 0.03125;
    const softcap: f32 = 0.75;
    const positions = [_]u64{ 0, 1, 2, 3, 4, 5, 6, 7 };
    // Row two represents a fitted span and is deliberately invisible to the
    // first query despite its original position being causal.
    const coverage = [_]u64{ 1, 2, 4, 4, 5, 6, 7, 8 };
    const bias = [_]f32{ 0.25, -0.125, 0.5, -0.25, 0.125, -0.5, 0, 0 };
    const canonical = [_]f32{ -1, -0.5, -0.25, 0, 0.25, 0.5, 1 };

    var keys: [kv_heads][]f32 = undefined;
    var values: [kv_heads][]f32 = undefined;
    var initialized: usize = 0;
    defer for (0..initialized) |head| {
        allocator.free(keys[head]);
        allocator.free(values[head]);
    };
    var heads: [kv_heads]checkpoint.Head = undefined;
    for (0..kv_heads) |head| {
        keys[head] = try allocator.alloc(f32, rows * dim);
        errdefer allocator.free(keys[head]);
        values[head] = try allocator.alloc(f32, rows * dim);
        errdefer allocator.free(values[head]);
        initialized += 1;
        for (0..rows) |row| for (0..dim) |d| {
            keys[head][row * dim + d] = canonical[(head * 3 + row + d) % canonical.len];
            values[head][row * dim + d] = canonical[(head * 5 + row * 2 + d * 3) % canonical.len];
        };
        heads[head] = .{
            .dim = @intCast(dim),
            .compacted_until = 6,
            .positions = &positions,
            .coverage_ends = &coverage,
            .keys = keys[head],
            .values = values[head],
            .bias = &bias,
        };
    }
    const tokens: [rows]u32 = @splat(0);
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = rows,
        .tokens = &tokens,
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &.{.{ .layer = 0, .heads = &heads }},
    };
    try state.validate(.{});

    const query_elements = query_rows * query_heads * dim;
    const queries = try allocator.alloc(f32, query_elements);
    defer allocator.free(queries);
    for (queries, 0..) |*query, i| query.* = canonical[(i * 5 + i / dim) % canonical.len];
    const expected = try allocator.alloc(f64, query_elements);
    defer allocator.free(expected);
    var expected_mass: [query_rows * query_heads]f64 = undefined;
    try attentionOracle(&heads, queries, query_rows, query_heads, first_query_position, 0, scale, softcap, expected, &expected_mass);

    var runner = try Runner.init(128 * 1024 * 1024, null);
    defer runner.deinit();
    const backend = runner.backend(allocator, .{});
    var query = try runner.core.buffer(queries.len * @sizeOf(f32));
    defer query.deinit();
    @memcpy(query.floats(), queries);
    var output = try runner.core.buffer(queries.len * @sizeOf(f32));
    defer output.deinit();
    var mass = try runner.core.buffer(expected_mass.len * @sizeOf(f32));
    defer mass.deinit();
    const attention_request: Request = .{
        .heads = &.{},
        .queries = query.ref(),
        .output = output.ref(),
        .log_mass = mass.ref(),
        .query_rows = query_rows,
        .query_heads = query_heads,
        .first_query_position = first_query_position,
        .scale = scale,
        .softcap = softcap,
    };

    // Run the identical real-shape request through exact f16 and resident FP8
    // storage. The encoded interval has a nonzero offset and exact rows on
    // both sides, so this covers all storage-row mapping branches.
    var exact = try backend.upload(state, 128 * 1024 * 1024, .{});
    defer exact.deinit();
    var request = attention_request;
    request.heads = try exact.layer(0);
    try runner.attend(request, .{});
    for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 3e-4);
    for (expected_mass, mass.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 3e-4);

    heads[0].fp8 = .{ .first_row = 2, .rows = 4, .key_scale = 1, .value_scale = 1 };
    try state.validate(.{});
    var mixed = try backend.upload(state, 128 * 1024 * 1024, .{});
    defer mixed.deinit();
    request.heads = try mixed.layer(0);
    try runner.attend(request, .{});
    for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 3e-4);
    for (expected_mass, mass.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 3e-4);

    heads[1].fp8 = .{ .first_row = 2, .rows = 4, .key_scale = 1, .value_scale = 1 };
    try state.validate(.{});
    var encoded = try backend.upload(state, 128 * 1024 * 1024, .{});
    defer encoded.deinit();
    request.heads = try encoded.layer(0);
    try runner.attend(request, .{});
    for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 3e-4);
    for (expected_mass, mass.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 3e-4);

    // Both signed reserved E4M3FN encodings must trip the GPU status path,
    // never decode as the finite magnitude that exponent/mantissa arithmetic
    // alone would produce.
    const key_byte = &encoded.heads[0].encoded_keys.?.bytes[@sizeOf(Fp8Header)];
    const original_key = key_byte.*;
    key_byte.* = 0x7f;
    try std.testing.expectError(error.NonFiniteAttention, runner.attend(request, .{}));
    key_byte.* = original_key;
    const value_byte = &encoded.heads[0].encoded_values.?.bytes[0];
    const original_value = value_byte.*;
    value_byte.* = 0xff;
    try std.testing.expectError(error.NonFiniteAttention, runner.attend(request, .{}));
    value_byte.* = original_value;

    // Header corruption is rejected on the CPU before dispatch. Keep output
    // sentinel-filled to prove that no partially executed GPU work escaped.
    const header: *Fp8Header = @ptrCast(@alignCast(encoded.heads[0].encoded_keys.?.bytes.ptr));
    const original_header = header.*;
    @memset(output.floats(), -77);
    header.first_row += 1;
    try std.testing.expectError(error.InvalidAttentionShape, runner.attend(request, .{}));
    try std.testing.expectError(error.InvalidAttentionShape, encoded.clone(allocator, 128 * 1024 * 1024, .{}));
    try std.testing.expectError(error.InvalidKvCheckpoint, encoded.download(allocator, state, .{}, 128 * 1024 * 1024, .{}));
    try std.testing.expectError(error.InvalidAttentionShape, runner.append(&encoded, 0, query.ref(), query.ref(), 1, state.logical_length, .{}));
    for (output.floats()) |value| try std.testing.expectEqual(@as(f32, -77), value);
    header.* = original_header;
    header.rows -= 1;
    try std.testing.expectError(error.InvalidAttentionShape, runner.attend(request, .{}));
    header.* = original_header;
    header.key_scale = @bitCast(@as(u32, @bitCast(header.key_scale)) ^ 1);
    try std.testing.expectError(error.InvalidAttentionShape, runner.attend(request, .{}));
    header.* = original_header;
    header.value_scale = @bitCast(@as(u32, @bitCast(header.value_scale)) ^ 1);
    try std.testing.expectError(error.InvalidAttentionShape, runner.attend(request, .{}));
    header.* = original_header;
    try runner.attend(request, .{});
}

test "attention_match Metal E4B shape f16 FP8 attention matches independent FP64 oracle" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    try realShapeFp8AttentionDifferential(256);
    try realShapeFp8AttentionDifferential(512);
}

fn rotated4AttentionDifferential(dim: usize, rows: usize, comptime query_heads: usize, comptime kv_heads: usize, matrix: bool) !void {
    const allocator = std.testing.allocator;
    const canonical = [_]f32{ -1, -0.5, -0.25, 0, 0.25, 0.5, 1 };
    const positions = try allocator.alloc(u64, rows);
    defer allocator.free(positions);
    const coverage = try allocator.alloc(u64, rows);
    defer allocator.free(coverage);
    const bias = try allocator.alloc(f32, rows);
    defer allocator.free(bias);
    const keys = try allocator.alloc(f32, rows * dim);
    defer allocator.free(keys);
    const values = try allocator.alloc(f32, rows * dim);
    defer allocator.free(values);
    const tokens = try allocator.alloc(u32, rows);
    defer allocator.free(tokens);
    @memset(tokens, 0);
    for (0..rows) |row| {
        positions[row] = row;
        coverage[row] = row + 1;
        bias[row] = @as(f32, @floatFromInt(@as(i32, @intCast(row % 5)) - 2)) * 0.03125;
        for (0..dim) |d| {
            keys[row * dim + d] = canonical[(row * 3 + d) % canonical.len];
            values[row * dim + d] = canonical[(row + d * 5) % canonical.len];
        }
    }
    const first_a: usize = 1;
    const first_b: usize = if (rows >= 1024) 512 else 5;
    const span_rows: usize = if (rows >= 1024) 256 else 2;
    // Exercise the valid zero-scale key representation in every dimension.
    @memset(keys[first_a * dim ..][0..dim], 0);
    var first = try (packed_storage.Packed{}).append(allocator, first_a, @intCast(dim), keys[first_a * dim ..][0 .. span_rows * dim], values[first_a * dim ..][0 .. span_rows * dim], .{});
    defer first.deinit(allocator);
    var compressed = try first.append(allocator, first_b, @intCast(dim), keys[first_b * dim ..][0 .. span_rows * dim], values[first_b * dim ..][0 .. span_rows * dim], .{});
    defer compressed.deinit(allocator);
    var decoded_keys: [512]f32 = undefined;
    var decoded_values: [512]f32 = undefined;
    for (compressed.spans) |span| for (0..span.rows) |i| {
        const packed_row = span.offset + i;
        const row = span.first + i;
        try compressed.decodeRow(@intCast(dim), packed_row, decoded_keys[0..dim], decoded_values[0..dim]);
        @memcpy(keys[row * dim ..][0..dim], decoded_keys[0..dim]);
        @memcpy(values[row * dim ..][0..dim], decoded_values[0..dim]);
    };
    const head: checkpoint.Head = .{ .dim = @intCast(dim), .compacted_until = @intCast(rows), .positions = positions, .coverage_ends = coverage, .keys = keys, .values = values, .bias = bias, .compressed = compressed };
    var heads: [kv_heads]checkpoint.Head = @splat(head);
    var donor: checkpoint.Donor = .{ .heads = &heads };
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = @intCast(rows),
        .tokens = tokens,
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = (&donor)[0..1],
    };
    try state.validate(.{});

    const queries = try allocator.alloc(f32, query_heads * dim);
    defer allocator.free(queries);
    for (queries, 0..) |*query, i| query.* = canonical[(i * 3 + i / dim) % canonical.len] / (if (matrix) @as(f32, 0.731239) else 1);
    const expected = try allocator.alloc(f64, queries.len);
    defer allocator.free(expected);
    var expected_mass: [query_heads]f64 = undefined;
    const query_position: u64 = @intCast(rows - 1);
    try attentionOracle(&heads, queries, 1, query_heads, query_position, 0, 0.03125, 0.75, expected, &expected_mass);

    var runner = try Runner.init(256 * 1024 * 1024, null);
    defer runner.deinit();
    runner.matrix_stage_enabled = matrix;
    const backend = runner.backend(allocator, .{});
    var device = try backend.upload(state, 256 * 1024 * 1024, .{});
    defer device.deinit();
    var query = try runner.core.buffer((queries.len + 1) * @sizeOf(f32));
    defer query.deinit();
    @memcpy(query.floats()[1..][0..queries.len], queries);
    const query_ref = try query.ref().slice(4, queries.len * @sizeOf(f32));
    var output = try runner.core.buffer(queries.len * @sizeOf(f32));
    defer output.deinit();
    var mass = try runner.core.buffer(query_heads * @sizeOf(f32));
    defer mass.deinit();
    try runner.attend(.{ .heads = try device.layer(0), .queries = query_ref, .output = output.ref(), .log_mass = mass.ref(), .query_rows = 1, .query_heads = query_heads, .first_query_position = query_position, .scale = 0.03125, .softcap = 0.75 }, .{});
    for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-3);
    for (expected_mass, mass.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-3);

    if (matrix) {
        try std.testing.expectEqual(@as(usize, 1), runner.rotated_query_dispatches);
        try std.testing.expectEqual(@as(usize, kv_heads), runner.matrix_stage_dispatches);
        var alias: Request = .{ .heads = try device.layer(0), .queries = query_ref, .output = runner.rotated_queries.?.ref(), .query_rows = 1, .query_heads = query_heads, .first_query_position = query_position, .scale = 0.03125 };
        try std.testing.expectError(error.InvalidAttentionShape, runner.attend(alias, .{}));
        alias.output = output.ref();
        alias.queries = try runner.rotated_queries.?.ref().slice(0, queries.len * @sizeOf(f32));
        try std.testing.expectError(error.InvalidAttentionShape, runner.attend(alias, .{}));
        var provider = try shared.Kernels.initSource(@embedFile("metal_attention_match_storage.metal"), "metal_attention_match_copy");
        defer provider.deinit();
        for ([_]bool{ false, true }) |coalesced| {
            runner.coalesced_frame_enabled = coalesced;
            @memset(runner.rotated_queries.?.bytes, 0xff);
            try runner.beginDeferredFrame(&device, provider.runtime, query_heads, rows, .{});
            defer runner.cancelDeferredFrame();
            @memset(runner.scratch.?.bytes, 0xff);
            try std.testing.expectError(error.ContextStagingBudgetExceeded, runner.ensureRotatedQueries(query_heads * 2, dim));
            try runner.attendOnFrame(provider.runtime, .{ .heads = try device.layer(0), .queries = query_ref, .output = output.ref(), .log_mass = mass.ref(), .query_rows = 1, .query_heads = query_heads, .first_query_position = rows, .scale = 0.03125, .softcap = 0.75 }, null, .{});
            try metal_runtime.submitFrame(provider.runtime);
            try metal_runtime.waitFrame(provider.runtime);
            try runner.checkStatus();
            runner.cancelDeferredFrame();
            for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-3);
            for (expected_mass, mass.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-3);
        }
    }

    var restored = try device.download(allocator, state, .{}, 256 * 1024 * 1024, .{});
    defer restored.deinit();
    const restored_head = restored.state.donors[0].heads[0];
    try std.testing.expectEqualSlices(packed_storage.Span, compressed.spans, restored_head.compressed.spans);
    try std.testing.expectEqualSlices(u8, compressed.keys, restored_head.compressed.keys);
    try std.testing.expectEqualSlices(u8, compressed.values, restored_head.compressed.values);
    var cloned = try device.clone(allocator, 256 * 1024 * 1024, .{});
    defer cloned.deinit();
    const before = cloned.allocated_bytes;
    const append_values = try allocator.alloc(f32, kv_heads * dim);
    defer allocator.free(append_values);
    @memset(append_values, 0.25);
    var append_buffer = try runner.core.buffer(kv_heads * dim * @sizeOf(f32));
    defer append_buffer.deinit();
    @memcpy(append_buffer.floats(), append_values);
    try runner.append(&cloned, 0, append_buffer.ref(), append_buffer.ref(), 1, @intCast(rows), .{});
    try std.testing.expect(cloned.allocated_bytes >= before);
    try std.testing.expectEqualSlices(u8, compressed.keys, cloned.heads[0].encoded_keys.?.bytes[cloned.heads[0].encoded_keys.?.logical_bytes - compressed.keys.len ..][0..compressed.keys.len]);
    try std.testing.expectEqualSlices(u8, compressed.values, cloned.heads[0].encoded_values.?.bytes[0..compressed.values.len]);
    try runner.attend(.{ .heads = try cloned.layer(0), .queries = query_ref, .output = output.ref(), .log_mass = mass.ref(), .query_rows = 1, .query_heads = query_heads, .first_query_position = @intCast(rows), .scale = 0.03125, .softcap = 0.75 }, .{});
    for (0..query_heads) |query_head| {
        var appended_score: f64 = 0;
        for (queries[query_head * dim ..][0..dim]) |q| appended_score += @as(f64, q) * 0.25;
        appended_score *= 0.03125;
        appended_score = 0.75 * std.math.tanh(appended_score / 0.75);
        const maximum = @max(expected_mass[query_head], appended_score);
        const old_weight = @exp(expected_mass[query_head] - maximum);
        const new_weight = @exp(appended_score - maximum);
        const denominator = old_weight + new_weight;
        for (0..dim) |d| {
            const want = (expected[query_head * dim + d] * old_weight + 0.25 * new_weight) / denominator;
            try std.testing.expectApproxEqAbs(want, @as(f64, output.floats()[query_head * dim + d]), 2e-3);
        }
        try std.testing.expectApproxEqAbs(maximum + @log(denominator), @as(f64, mass.floats()[query_head]), 2e-3);
    }

    const stored = &device.heads[0];
    const span_bytes = std.mem.sliceAsBytes(compressed.spans);
    try std.testing.expectEqualSlices(u8, span_bytes, stored.encoded_keys.?.bytes[@sizeOf(Fp8Header)..][0..span_bytes.len]);
    const spans = std.mem.bytesAsSlice(packed_storage.Span, @as([]align(@alignOf(packed_storage.Span)) u8, @alignCast(stored.encoded_keys.?.bytes[@sizeOf(Fp8Header)..][0..span_bytes.len])));
    const original_offset = spans[0].offset;
    spans[0].offset = 1;
    try std.testing.expectError(error.InvalidAttentionShape, runner.attend(.{ .heads = try device.layer(0), .queries = query_ref, .output = output.ref(), .query_rows = 1, .query_heads = query_heads, .first_query_position = query_position, .scale = 0.03125 }, .{}));
    spans[0].offset = original_offset;
    const header: *Fp8Header = @ptrCast(@alignCast(stored.encoded_keys.?.bytes.ptr));
    const key_scale = stored.encoded_keys.?.bytes[header.key_payload_offset..][0..4];
    var original_scale: [4]u8 = undefined;
    @memcpy(&original_scale, key_scale);
    std.mem.writeInt(u32, key_scale, @bitCast(std.math.nan(f32)), .little);
    try std.testing.expectError(error.NonFiniteAttention, runner.attend(.{ .heads = try device.layer(0), .queries = query_ref, .output = output.ref(), .query_rows = 1, .query_heads = query_heads, .first_query_position = query_position, .scale = 0.03125 }, .{}));
    @memcpy(key_scale, &original_scale);
    const value_code = &stored.encoded_values.?.bytes[4];
    const original_code = value_code.*;
    const packed_metadata: *Metadata = @ptrCast(@alignCast(stored.metadata.?.bytes.ptr + first_a * @sizeOf(Metadata)));
    const original_bias = packed_metadata.bias;
    packed_metadata.bias = -10000;
    value_code.* = 0x80;
    try std.testing.expectError(error.NonFiniteAttention, runner.attend(.{ .heads = try device.layer(0), .queries = query_ref, .output = output.ref(), .query_rows = 1, .query_heads = query_heads, .first_query_position = query_position, .scale = 0.03125 }, .{}));
    value_code.* = original_code;
    packed_metadata.bias = original_bias;
}

test "attention_match Metal resident rotated4 scalar and split K match independent decoded oracle" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    try rotated4AttentionDifferential(64, 8, 2, 1, false);
    try rotated4AttentionDifferential(128, 8, 2, 1, false);
    try rotated4AttentionDifferential(256, 1024, 2, 1, false);
    try rotated4AttentionDifferential(512, 1024, 2, 1, false);
}

test "attention_match Metal rotated4 matrix keeps FP32 queries packed bytes and GQA owner rows" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    for ([_]usize{ 1025, 4097, 8193 }) |rows| {
        try rotated4AttentionDifferential(512, rows, 8, 1, true);
        try rotated4AttentionDifferential(512, rows, 8, 2, true);
    }
}

fn splitKBoundaryDifferential(runner: *Runner, dim: usize, rows: usize) !void {
    const allocator = std.testing.allocator;
    const query_heads = 2;
    const canonical = [_]f32{ -1, -0.5, -0.25, 0, 0.25, 0.5, 1 };
    const positions = try allocator.alloc(u64, rows);
    defer allocator.free(positions);
    const coverage = try allocator.alloc(u64, rows);
    defer allocator.free(coverage);
    const bias = try allocator.alloc(f32, rows);
    defer allocator.free(bias);
    const keys = try allocator.alloc(f32, rows * dim);
    defer allocator.free(keys);
    const values = try allocator.alloc(f32, rows * dim);
    defer allocator.free(values);
    const tokens = try allocator.alloc(u32, rows);
    defer allocator.free(tokens);
    @memset(tokens, 0);
    for (0..rows) |row| {
        positions[row] = row;
        coverage[row] = row + 1;
        bias[row] = @as(f32, @floatFromInt(@as(i32, @intCast(row % 5)) - 2)) * 0.0625;
        for (0..dim) |d| {
            keys[row * dim + d] = canonical[(row + d * 3) % canonical.len];
            values[row * dim + d] = canonical[(row * 5 + d) % canonical.len];
        }
    }
    // A fitted span ending after this causal query must remain invisible.
    if (rows > 1) coverage[rows - 2] = rows;
    var head: checkpoint.Head = .{
        .dim = @intCast(dim),
        .compacted_until = @intCast(rows),
        .positions = positions,
        .coverage_ends = coverage,
        .keys = keys,
        .values = values,
        .bias = bias,
    };
    var donor: checkpoint.Donor = .{ .heads = (&head)[0..1] };
    var state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = @intCast(rows),
        .tokens = tokens,
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = (&donor)[0..1],
    };
    try state.validate(.{});

    const queries = try allocator.alloc(f32, query_heads * dim);
    defer allocator.free(queries);
    for (queries, 0..) |*query, i| query.* = canonical[(i * 5 + i / dim) % canonical.len];
    const expected = try allocator.alloc(f64, queries.len);
    defer allocator.free(expected);
    var expected_mass: [query_heads]f64 = undefined;
    const query_position: u64 = @intCast(rows -| 2);
    try attentionOracle((&head)[0..1], queries, 1, query_heads, query_position, 0, 0.03125, 0.75, expected, &expected_mass);

    var query = try runner.core.buffer(queries.len * @sizeOf(f32));
    defer query.deinit();
    @memcpy(query.floats(), queries);
    var output = try runner.core.buffer(queries.len * @sizeOf(f32));
    defer output.deinit();
    var mass = try runner.core.buffer(query_heads * @sizeOf(f32));
    defer mass.deinit();
    const backend = runner.backend(allocator, .{});

    var exact = try backend.upload(state, 256 * 1024 * 1024, .{});
    var exact_live = true;
    defer if (exact_live) exact.deinit();
    if (rows == 8192 and dim == 512) {
        const alias_query_heads = 8;
        const alias_elements = alias_query_heads * dim;
        const old_scratch_bytes = 16 * alias_query_heads * (dim + 2) * @sizeOf(f32);
        try runner.ensureScratch(old_scratch_bytes);
        const old_scratch = runner.scratch.?;
        old_scratch.bytes[0] = 0x5a;
        var alias_query = try runner.core.buffer(alias_elements * @sizeOf(f32));
        defer alias_query.deinit();
        var alias_output = try runner.core.buffer(alias_elements * @sizeOf(f32));
        defer alias_output.deinit();
        var alias_mass = try runner.core.buffer(alias_query_heads * @sizeOf(f32));
        defer alias_mass.deinit();
        const base: Request = .{
            .heads = try exact.layer(0),
            .queries = alias_query.ref(),
            .output = alias_output.ref(),
            .log_mass = alias_mass.ref(),
            .query_rows = 1,
            .query_heads = alias_query_heads,
            .first_query_position = query_position,
            .scale = 0.03125,
            .softcap = 0.75,
        };
        const before_aliases = runner.core.batchCounters();
        var aliased = base;
        aliased.queries = try old_scratch.ref().slice(0, alias_elements * @sizeOf(f32));
        try std.testing.expectError(error.InvalidAttentionShape, runner.attend(aliased, .{}));
        aliased = base;
        aliased.output = try old_scratch.ref().slice(0, alias_elements * @sizeOf(f32));
        try std.testing.expectError(error.InvalidAttentionShape, runner.attend(aliased, .{}));
        aliased = base;
        aliased.log_mass = try old_scratch.ref().slice(0, alias_query_heads * @sizeOf(f32));
        try std.testing.expectError(error.InvalidAttentionShape, runner.attend(aliased, .{}));
        aliased = base;
        aliased.output = .{ .handle = runner.status.handle, .bytes = alias_elements * @sizeOf(f32) };
        try std.testing.expectError(error.InvalidAttentionShape, runner.attend(aliased, .{}));
        aliased = base;
        aliased.queries = .{ .handle = runner.fp8_zero.handle, .bytes = alias_elements * @sizeOf(f32) };
        try std.testing.expectError(error.InvalidAttentionShape, runner.attend(aliased, .{}));
        const after_aliases = runner.core.batchCounters();
        try std.testing.expectEqual(before_aliases.submissions, after_aliases.submissions);
        try std.testing.expectEqual(before_aliases.waits, after_aliases.waits);
        try std.testing.expect(runner.scratch.?.handle == old_scratch.handle);
        try std.testing.expectEqual(old_scratch.logical_bytes, runner.scratch.?.logical_bytes);
        try std.testing.expectEqual(@as(u8, 0x5a), runner.scratch.?.bytes[0]);
    }
    const split_before = runner.split_stage.batchCounters();
    const core_before = runner.core.batchCounters();
    try runner.attend(.{
        .heads = try exact.layer(0),
        .queries = query.ref(),
        .output = output.ref(),
        .log_mass = mass.ref(),
        .query_rows = 1,
        .query_heads = query_heads,
        .first_query_position = query_position,
        .scale = 0.03125,
        .softcap = 0.75,
    }, .{});
    const split_after = runner.split_stage.batchCounters();
    const core_after = runner.core.batchCounters();
    if (rows >= 1024) {
        try std.testing.expect((try Runner.splitLayout(.{ .heads = try exact.layer(0), .queries = query.ref(), .output = output.ref(), .query_rows = 1, .query_heads = query_heads, .first_query_position = query_position, .scale = 0.03125 }, false)) != null);
    } else {
        try std.testing.expect((try Runner.splitLayout(.{ .heads = try exact.layer(0), .queries = query.ref(), .output = output.ref(), .query_rows = 1, .query_heads = query_heads, .first_query_position = query_position, .scale = 0.03125 }, false)) == null);
    }
    try std.testing.expectEqual(split_before.submissions, split_after.submissions);
    try std.testing.expectEqual(split_before.waits, split_after.waits);
    try std.testing.expectEqual(core_before.submissions + 1, core_after.submissions);
    try std.testing.expectEqual(core_before.waits + 1, core_after.waits);
    for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-3);
    for (expected_mass, mass.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-3);
    exact.deinit();
    exact_live = false;

    head.fp8 = if (rows == 1024)
        .{ .first_row = 0, .rows = @intCast(rows), .key_scale = 1, .value_scale = 1 }
    else
        .{ .first_row = @intCast(rows / 4), .rows = @intCast(rows / 2), .key_scale = 1, .value_scale = 1 };
    donor.heads = (&head)[0..1];
    state.donors = (&donor)[0..1];
    try state.validate(.{});
    var encoded = try backend.upload(state, 256 * 1024 * 1024, .{});
    defer encoded.deinit();
    try runner.attend(.{
        .heads = try encoded.layer(0),
        .queries = query.ref(),
        .output = output.ref(),
        .log_mass = mass.ref(),
        .query_rows = 1,
        .query_heads = query_heads,
        .first_query_position = query_position,
        .scale = 0.03125,
        .softcap = 0.75,
    }, .{});
    for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-3);
    for (expected_mass, mass.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-3);

    if (rows == 1024) {
        const request: Request = .{
            .heads = try encoded.layer(0),
            .queries = query.ref(),
            .output = output.ref(),
            .query_rows = 1,
            .query_heads = query_heads,
            .first_query_position = query_position,
            .scale = 0.03125,
            .softcap = 0.75,
        };
        var aliased = request;
        aliased.output = query.ref();
        const split_before_alias = runner.core.batchCounters();
        try std.testing.expectError(error.InvalidAttentionShape, runner.attend(aliased, .{}));
        const split_after_alias = runner.core.batchCounters();
        try std.testing.expectEqual(split_before_alias.submissions, split_after_alias.submissions);
        try std.testing.expectEqual(split_before_alias.waits, split_after_alias.waits);
        const key_byte = &encoded.heads[0].encoded_keys.?.bytes[@sizeOf(Fp8Header)];
        const original_key = key_byte.*;
        key_byte.* = 0x7f;
        try std.testing.expectError(error.NonFiniteAttention, runner.attend(request, .{}));
        key_byte.* = original_key;
        const value_byte = &encoded.heads[0].encoded_values.?.bytes[0];
        const original_value = value_byte.*;
        value_byte.* = 0xff;
        try std.testing.expectError(error.NonFiniteAttention, runner.attend(request, .{}));
        value_byte.* = original_value;

        const header: *Fp8Header = @ptrCast(@alignCast(encoded.heads[0].encoded_keys.?.bytes.ptr));
        const original_header = header.*;
        @memset(output.floats(), -73);
        header.rows -= 1;
        try std.testing.expectError(error.InvalidAttentionShape, runner.attend(request, .{}));
        for (output.floats()) |value| try std.testing.expectEqual(@as(f32, -73), value);
        header.* = original_header;
    }
}

test "attention_match Metal split K boundaries f16 and resident FP8 match independent FP64 oracle" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    var runner = try Runner.init(256 * 1024 * 1024, null);
    defer runner.deinit();
    for ([_]usize{ 256, 512 }) |dim| for ([_]usize{ 255, 256, 257, 1024, 8192 }) |rows|
        try splitKBoundaryDifferential(&runner, dim, rows);

    const old_scratch = runner.scratch.?;
    old_scratch.bytes[0] = 0xa5;
    const old_limit = runner.max_device_bytes;
    runner.max_device_bytes = old_scratch.bytes.len;
    try std.testing.expectError(
        error.MetalMemoryLimitExceeded,
        runner.ensureScratch(old_scratch.logical_bytes + std.heap.pageSize()),
    );
    try std.testing.expect(runner.scratch.?.handle == old_scratch.handle);
    try std.testing.expectEqual(old_scratch.logical_bytes, runner.scratch.?.logical_bytes);
    try std.testing.expectEqual(@as(u8, 0xa5), runner.scratch.?.bytes[0]);
    runner.max_device_bytes = old_limit;
}

test "attention_match Metal split K supports mixed FP8 ragged heads and cancellation" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const dim = 256;
    const logical_rows = 1280;
    const head_rows = [_]usize{ 1024, logical_rows };
    const compacted = [_]usize{ 1024, logical_rows };
    const query_heads = 4;
    const canonical = [_]f32{ -1, -0.5, -0.25, 0, 0.25, 0.5, 1 };
    var positions: [2][]u64 = undefined;
    var coverage: [2][]u64 = undefined;
    var bias: [2][]f32 = undefined;
    var keys: [2][]f32 = undefined;
    var values: [2][]f32 = undefined;
    var initialized: usize = 0;
    defer for (0..initialized) |head| {
        allocator.free(positions[head]);
        allocator.free(coverage[head]);
        allocator.free(bias[head]);
        allocator.free(keys[head]);
        allocator.free(values[head]);
    };
    var heads: [2]checkpoint.Head = undefined;
    for (0..2) |head_index| {
        const rows = head_rows[head_index];
        positions[head_index] = try allocator.alloc(u64, rows);
        coverage[head_index] = try allocator.alloc(u64, rows);
        bias[head_index] = try allocator.alloc(f32, rows);
        keys[head_index] = try allocator.alloc(f32, rows * dim);
        values[head_index] = try allocator.alloc(f32, rows * dim);
        initialized += 1;
        for (0..rows) |row| {
            const position = if (head_index == 0 and row >= 768) 1024 + row - 768 else row;
            positions[head_index][row] = position;
            coverage[head_index][row] = if (head_index == 0 and row == 767) 1024 else position + 1;
            bias[head_index][row] = if (position >= compacted[head_index]) 0 else @as(f32, @floatFromInt(@as(i32, @intCast((row + head_index) % 3)) - 1)) * 0.125;
            for (0..dim) |d| {
                keys[head_index][row * dim + d] = canonical[(head_index * 3 + row + d) % canonical.len];
                values[head_index][row * dim + d] = canonical[(head_index * 5 + row * 3 + d) % canonical.len];
            }
        }
        heads[head_index] = .{
            .dim = dim,
            .compacted_until = compacted[head_index],
            .positions = positions[head_index],
            .coverage_ends = coverage[head_index],
            .keys = keys[head_index],
            .values = values[head_index],
            .bias = bias[head_index],
            .fp8 = if (head_index == 0) .{ .first_row = 0, .rows = 768, .key_scale = 1, .value_scale = 1 } else .{},
        };
    }
    const tokens = try allocator.alloc(u32, logical_rows);
    defer allocator.free(tokens);
    @memset(tokens, 0);
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = logical_rows,
        .tokens = tokens,
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &.{.{ .heads = &heads }},
    };
    try state.validate(.{});

    const queries = try allocator.alloc(f32, query_heads * dim);
    defer allocator.free(queries);
    for (queries, 0..) |*query, i| query.* = canonical[(i * 3 + i / dim) % canonical.len];
    const expected = try allocator.alloc(f64, queries.len);
    defer allocator.free(expected);
    var expected_mass: [query_heads]f64 = undefined;
    try attentionOracle(&heads, queries, 1, query_heads, logical_rows - 1, 0, 0.03125, 0.5, expected, &expected_mass);

    var runner = try Runner.init(128 * 1024 * 1024, null);
    defer runner.deinit();
    var device = try runner.backend(allocator, .{}).upload(state, 128 * 1024 * 1024, .{});
    defer device.deinit();
    var query = try runner.core.buffer(queries.len * @sizeOf(f32));
    defer query.deinit();
    @memcpy(query.floats(), queries);
    var output = try runner.core.buffer(queries.len * @sizeOf(f32));
    defer output.deinit();
    var mass = try runner.core.buffer(query_heads * @sizeOf(f32));
    defer mass.deinit();
    const request: Request = .{
        .heads = try device.layer(0),
        .queries = query.ref(),
        .output = output.ref(),
        .log_mass = mass.ref(),
        .query_rows = 1,
        .query_heads = query_heads,
        .first_query_position = logical_rows - 1,
        .scale = 0.03125,
        .softcap = 0.5,
    };

    const Cancel = struct {
        calls: usize = 0,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            self.calls += 1;
            if (self.calls == 4) return error.CompactionCancelled;
        }
    };
    @memset(output.floats(), -91);
    const before = runner.core.batchCounters();
    var cancel: Cancel = .{};
    try std.testing.expectError(error.CompactionCancelled, runner.attend(request, .{ .context = &cancel, .check_fn = Cancel.check }));
    const after = runner.core.batchCounters();
    try std.testing.expectEqual(before.submissions, after.submissions);
    try std.testing.expectEqual(before.waits, after.waits);
    try std.testing.expect(!after.active and !after.submitted);
    for (output.floats()) |value| try std.testing.expectEqual(@as(f32, -91), value);

    try runner.attend(request, .{});
    for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-3);
    for (expected_mass, mass.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-3);

    // No visible row is a hard error. Mutating only mapped test metadata keeps
    // the state geometry intact while exercising the GPU fail-closed path.
    for (device.heads) |stored| {
        const metadata = std.mem.bytesAsSlice(Metadata, @as([]align(@alignOf(Metadata)) u8, @alignCast(stored.metadata.?.bytes[0..stored.metadata.?.logical_bytes])));
        for (metadata[0..stored.capacity]) |*entry| {
            entry.coverage_lo = logical_rows + 1;
            entry.coverage_hi = 0;
        }
    }
    try std.testing.expectError(error.NonFiniteAttention, runner.attend(request, .{}));
}

test "attention_match Metal split K preserves sliding-window causality" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const dim = 256;
    const logical_rows = 2048;
    const window = 1024;
    const positions = try allocator.alloc(u64, window);
    defer allocator.free(positions);
    const coverage = try allocator.alloc(u64, window);
    defer allocator.free(coverage);
    const bias = try allocator.alloc(f32, window);
    defer allocator.free(bias);
    const keys = try allocator.alloc(f32, window * dim);
    defer allocator.free(keys);
    const values = try allocator.alloc(f32, window * dim);
    defer allocator.free(values);
    const tokens = try allocator.alloc(u32, logical_rows);
    defer allocator.free(tokens);
    @memset(tokens, 0);
    for (0..window) |row| {
        positions[row] = logical_rows - window + row;
        coverage[row] = positions[row] + 1;
        bias[row] = 0;
        for (0..dim) |d| {
            keys[row * dim + d] = if ((row + d) % 2 == 0) 0.5 else -0.5;
            values[row * dim + d] = if ((row * 3 + d) % 2 == 0) 1 else -1;
        }
    }
    const head: checkpoint.Head = .{ .dim = dim, .positions = positions, .coverage_ends = coverage, .keys = keys, .values = values, .bias = bias };
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = logical_rows,
        .tokens = tokens,
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &.{.{ .window = window, .heads = &.{head} }},
    };
    try state.validate(.{});
    const queries: [2 * dim]f32 = @splat(0.25);
    var expected: [2 * dim]f64 = undefined;
    var expected_mass: [2]f64 = undefined;
    try attentionOracle(&.{head}, &queries, 1, 2, logical_rows - 1, window, 0.03125, 0.75, &expected, &expected_mass);
    var runner = try Runner.init(64 * 1024 * 1024, null);
    defer runner.deinit();
    var device = try runner.backend(allocator, .{}).upload(state, 64 * 1024 * 1024, .{});
    defer device.deinit();
    var query = try runner.core.buffer(@sizeOf(@TypeOf(queries)));
    defer query.deinit();
    @memcpy(query.floats(), &queries);
    var output = try runner.core.buffer(@sizeOf(@TypeOf(queries)));
    defer output.deinit();
    var mass = try runner.core.buffer(@sizeOf(@TypeOf(expected_mass)) / 2);
    defer mass.deinit();
    try runner.attend(.{ .heads = try device.layer(0), .queries = query.ref(), .output = output.ref(), .log_mass = mass.ref(), .query_rows = 1, .query_heads = 2, .first_query_position = logical_rows - 1, .scale = 0.03125, .softcap = 0.75 }, .{});
    for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-3);
    for (expected_mass, mass.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-3);
}

test "attention_match Metal local-window split partitions match independent oracle" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const rows = 512;
    const logical = 4096;
    const query_heads = 8;
    for ([_]usize{ 256, 512 }) |dim| {
        const positions = try allocator.alloc(u64, rows);
        defer allocator.free(positions);
        const coverage = try allocator.alloc(u64, rows);
        defer allocator.free(coverage);
        const bias = try allocator.alloc(f32, rows);
        defer allocator.free(bias);
        const keys = try allocator.alloc(f32, rows * dim);
        defer allocator.free(keys);
        const values = try allocator.alloc(f32, rows * dim);
        defer allocator.free(values);
        const queries = try allocator.alloc(f32, query_heads * dim);
        defer allocator.free(queries);
        const expected = try allocator.alloc(f64, queries.len);
        defer allocator.free(expected);
        var expected_mass: [query_heads]f64 = undefined;
        for (0..rows) |row| {
            positions[row] = logical - rows + row;
            coverage[row] = positions[row] + 1;
            bias[row] = 0;
            for (0..dim) |d| {
                keys[row * dim + d] = @as(f32, @floatFromInt(@as(i32, @intCast((row * 3 + d) % 17)) - 8)) / 16;
                values[row * dim + d] = @as(f32, @floatFromInt(@as(i32, @intCast((row + d * 5) % 19)) - 9)) / 16;
            }
        }
        for (queries, 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 16;
        const head: checkpoint.Head = .{ .dim = @intCast(dim), .positions = positions, .coverage_ends = coverage, .keys = keys, .values = values, .bias = bias };
        const tokens: [logical]u32 = @splat(0);
        const state: checkpoint.State = .{
            .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
            .logical_length = logical,
            .tokens = &tokens,
            .logits = &.{0},
            .layer_donors = &.{0},
            .donors = &.{.{ .window = rows, .heads = &.{head} }},
        };
        try attentionOracle(&.{head}, queries, 1, query_heads, logical - 2, rows, 0.0625, 0.75, expected, &expected_mass);
        var runner = try Runner.init(64 * 1024 * 1024, null);
        defer runner.deinit();
        var device = try runner.backend(allocator, .{}).upload(state, 64 * 1024 * 1024, .{});
        defer device.deinit();
        var query = try runner.core.buffer(queries.len * 4);
        defer query.deinit();
        @memcpy(query.floats(), queries);
        var output = try runner.core.buffer(queries.len * 4);
        defer output.deinit();
        var mass = try runner.core.buffer(query_heads * 4);
        defer mass.deinit();
        const request: Request = .{ .heads = try device.layer(0), .queries = query.ref(), .output = output.ref(), .log_mass = mass.ref(), .query_rows = 1, .query_heads = query_heads, .first_query_position = logical - 2, .scale = 0.0625, .softcap = 0.75 };
        try runner.attend(request, .{});
        const layout = (try Runner.splitLayout(request, false)) orelse return error.TestUnexpectedResult;
        try std.testing.expectEqual(@as(usize, 32), layout.rows_per_split);
        try std.testing.expectEqual(@as(usize, 16), layout.splits);
        var provider = try shared.Kernels.initSource(@embedFile("metal_attention_match_storage.metal"), "metal_attention_match_copy");
        defer provider.deinit();
        const before = runner.core.batchCounters();
        try metal_runtime.beginFrame(provider.runtime);
        try runner.attendOnFrame(provider.runtime, request, null, .{});
        try std.testing.expect(!metal_runtime.hasActiveFrame(provider.runtime));
        try std.testing.expect(!metal_runtime.hasSubmittedFrame(provider.runtime));
        const after = runner.core.batchCounters();
        try std.testing.expectEqual(before.submissions, after.submissions);
        try std.testing.expectEqual(before.waits, after.waits);
        for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-3);
        for (expected_mass, mass.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-3);
    }
}

test "attention_match Metal provider frame owns chunked append attention and finite acceptance" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const dim = 2;
    const initial = 4;
    const suffix = 17;
    const query_heads = 2;
    const positions = [_]u64{ 0, 1, 2, 3 };
    const coverage = [_]u64{ 1, 2, 3, 4 };
    const keys = [_]f32{ 1, 0, 1, 0, 1, 0, 1, 0 };
    const values = [_]f32{ 0, 0, 1, -1, 2, -2, 3, -3 };
    const bias: [initial]f32 = @splat(0);
    const head: checkpoint.Head = .{ .dim = dim, .positions = &positions, .coverage_ends = &coverage, .keys = &keys, .values = &values, .bias = &bias };
    const state: checkpoint.State = .{
        .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
        .logical_length = initial,
        .tokens = &.{ 0, 0, 0, 0 },
        .logits = &.{0},
        .layer_donors = &.{0},
        .donors = &.{.{ .heads = &.{head} }},
    };
    var provider = try shared.Kernels.initSource(@embedFile("metal_attention_match_storage.metal"), "metal_attention_match_copy");
    defer provider.deinit();
    const Scenario = enum { clean, poison, cancel_after_append };
    const StopAfterAppend = struct {
        device: *DeviceState,
        fn check(raw: ?*anyopaque) !void {
            const self: *@This() = @ptrCast(@alignCast(raw.?));
            if (self.device.logical_length != initial) return error.Cancelled;
        }
    };
    for ([_]Scenario{ .clean, .poison, .cancel_after_append }) |scenario| {
        var runner = try Runner.init(16 * 1024 * 1024, null);
        defer runner.deinit();
        var device = try runner.backend(allocator, .{}).upload(state, 16 * 1024 * 1024, .{});
        defer device.deinit();
        var query = try runner.core.buffer(suffix * query_heads * dim * 4);
        defer query.deinit();
        var output = try runner.core.buffer(suffix * query_heads * dim * 4);
        defer output.deinit();
        var new_keys = try runner.core.buffer(suffix * dim * 4);
        defer new_keys.deinit();
        var new_values = try runner.core.buffer(suffix * dim * 4);
        defer new_values.deinit();
        for (0..suffix) |row| {
            new_keys.floats()[row * dim] = 1;
            new_keys.floats()[row * dim + 1] = 0;
            const value: f32 = @floatFromInt(initial + row);
            new_values.floats()[row * dim] = value;
            new_values.floats()[row * dim + 1] = -value;
            for (0..query_heads) |qh| {
                query.floats()[(row * query_heads + qh) * dim] = 1;
                query.floats()[(row * query_heads + qh) * dim + 1] = 0;
            }
        }
        if (scenario == .poison) new_keys.floats()[0] = std.math.inf(f32);
        const request: Request = .{ .heads = try device.layer(0), .queries = query.ref(), .output = output.ref(), .query_rows = suffix, .query_heads = query_heads, .first_query_position = initial, .scale = 1 };
        const write: Runner.Append = .{ .state = &device, .layer = 0, .keys = new_keys.ref(), .values = new_values.ref(), .rows = suffix, .first_position = initial };
        const before = runner.core.batchCounters();
        try metal_runtime.beginFrame(provider.runtime);
        if (scenario == .poison) {
            try std.testing.expectError(error.NonFiniteAttention, runner.attendOnFrame(provider.runtime, request, write, .{}));
        } else if (scenario == .cancel_after_append) {
            var stop: StopAfterAppend = .{ .device = &device };
            try std.testing.expectError(error.Cancelled, runner.attendOnFrame(provider.runtime, request, write, .{ .context = &stop, .check_fn = StopAfterAppend.check }));
        } else {
            try runner.attendOnFrame(provider.runtime, request, write, .{});
            for (0..suffix) |row| for (0..query_heads) |qh| {
                const expected = @as(f32, @floatFromInt(initial + row)) / 2;
                try std.testing.expectApproxEqAbs(expected, output.floats()[(row * query_heads + qh) * dim], 1e-4);
                try std.testing.expectApproxEqAbs(-expected, output.floats()[(row * query_heads + qh) * dim + 1], 1e-4);
            };
            try std.testing.expectEqual(@as(u64, initial + suffix), device.logical_length);
        }
        try std.testing.expect(!metal_runtime.hasActiveFrame(provider.runtime));
        try std.testing.expect(!metal_runtime.hasSubmittedFrame(provider.runtime));
        const after = runner.core.batchCounters();
        try std.testing.expectEqual(before.submissions, after.submissions);
        try std.testing.expectEqual(before.waits, after.waits);
    }
}

test "attention_match Metal F16 and FP8 matrix GQA4 and GQA8 preserve float queries ragged rings and finite checks" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const a = std.testing.allocator;
    inline for (.{
        .{ .dim = 512, .window = 0, .gqa = 8, .fp8 = false, .rows = [_]usize{ 1025, 8193 } },
        .{ .dim = 512, .window = 0, .gqa = 4, .fp8 = false, .rows = [_]usize{ 1025, 8193 } },
        .{ .dim = 256, .window = 512, .gqa = 8, .fp8 = false, .rows = [_]usize{ 128, 129, 511, 512, 513 } },
        .{ .dim = 256, .window = 512, .gqa = 4, .fp8 = false, .rows = [_]usize{ 128, 129, 511, 512, 513 } },
        .{ .dim = 512, .window = 0, .gqa = 8, .fp8 = true, .rows = [_]usize{ 4097, 8193 } },
        .{ .dim = 512, .window = 0, .gqa = 4, .fp8 = true, .rows = [_]usize{ 4097, 8193 } },
    }) |fixture| {
        const dim = fixture.dim;
        const query_heads = fixture.gqa * 2;
        for (fixture.rows) |rows| {
            const positions = try a.alloc(u64, rows);
            defer a.free(positions);
            const coverage = try a.alloc(u64, rows);
            defer a.free(coverage);
            const bias = try a.alloc(f32, rows);
            defer a.free(bias);
            const keys = try a.alloc(f32, rows * dim);
            defer a.free(keys);
            const values = try a.alloc(f32, rows * dim);
            defer a.free(values);
            const tokens = try a.alloc(u32, rows);
            defer a.free(tokens);
            @memset(tokens, 0);
            for (0..rows) |r| {
                positions[r] = r;
                coverage[r] = r + 1;
                bias[r] = if (r == 3) -1000 else @as(f32, @floatFromInt(@as(i32, @intCast(r % 7)) - 3)) / 8;
                for (0..dim) |d| {
                    keys[r * dim + d] = @as(f32, @floatFromInt(@as(i32, @intCast((r * 3 + d) % 17)) - 8)) / 16;
                    values[r * dim + d] = @as(f32, @floatFromInt(@as(i32, @intCast((r + d * 5) % 19)) - 9)) / 16;
                }
            }
            coverage[rows - 2] = rows; // Fitted coverage extending beyond the query.
            const short = rows / 2;
            const short_coverage = try a.dupe(u64, coverage[0..short]);
            defer a.free(short_coverage);
            short_coverage[short - 1] = rows; // One fitted span covers the remaining prefix.
            var heads = [_]checkpoint.Head{
                .{ .dim = dim, .compacted_until = @intCast(rows), .positions = positions, .coverage_ends = coverage, .bias = bias, .keys = keys, .values = values },
                .{ .dim = dim, .compacted_until = @intCast(rows), .positions = positions[0..short], .coverage_ends = short_coverage, .bias = bias[0..short], .keys = keys[0 .. short * dim], .values = values[0 .. short * dim] },
            };
            if (fixture.fp8) heads[0].fp8 = .{ .first_row = 128, .rows = @intCast(rows - 2176), .key_scale = 1, .value_scale = 1 };
            const state: checkpoint.State = .{
                .identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) },
                .logical_length = @intCast(rows),
                .tokens = tokens,
                .logits = &.{0},
                .layer_donors = &.{0},
                .donors = &.{.{ .heads = &heads }},
            };
            var queries: [query_heads * dim]f32 = undefined;
            for (&queries, 0..) |*q, i| q.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 37;
            // This fixture would detect silently converting queries to half.
            try std.testing.expect(queries[0] != @as(f32, @as(f16, @floatCast(queries[0]))));
            var expected: [query_heads * dim]f64 = undefined;
            var expected_mass: [query_heads]f64 = undefined;
            try attentionOracle(&heads, &queries, 1, query_heads, rows - 2, fixture.window, 0.03125, 0.75, &expected, &expected_mass);
            var runner = try Runner.init(128 * 1024 * 1024, null);
            defer runner.deinit();
            runner.matrix_stage_enabled = true;
            runner.coalesced_frame_enabled = true;
            var device = try runner.backend(a, .{}).upload(state, 128 * 1024 * 1024, .{});
            defer device.deinit();
            for (device.heads) |*head| {
                if (head.encoding == .fp8) continue; // Packed global storage has a fixed origin.
                // Upload the legal ragged global fixture, then exercise the same
                // visibility/ring kernel with a local window as well. The native
                // local append fixtures separately verify exact window ownership.
                head.window = fixture.window;
                const start = head.capacity - 3;
                for ([_]struct { buffer: *Buffer, stride: usize }{
                    .{ .buffer = &head.keys.?, .stride = dim * 2 },
                    .{ .buffer = &head.values.?, .stride = dim * 2 },
                    .{ .buffer = &head.metadata.?, .stride = @sizeOf(Metadata) },
                }) |part| {
                    const copy = try a.dupe(u8, part.buffer.bytes[0 .. head.rows * part.stride]);
                    defer a.free(copy);
                    @memset(part.buffer.bytes, 0);
                    for (0..head.rows) |r| {
                        const physical = (start + r) % head.capacity;
                        @memcpy(part.buffer.bytes[physical * part.stride ..][0..part.stride], copy[r * part.stride ..][0..part.stride]);
                    }
                }
                head.start = start;
            }
            var query = try runner.core.buffer(@sizeOf(@TypeOf(queries)) + 4);
            defer query.deinit();
            query.floats()[0] = std.math.nan(f32); // Prefix sentinel is outside the four-byte-shifted query view.
            @memcpy(query.floats()[1..], &queries);
            var output = try runner.core.buffer(@sizeOf(@TypeOf(queries)));
            defer output.deinit();
            var mass = try runner.core.buffer(query_heads * 4);
            defer mass.deinit();
            const request: Request = .{ .heads = try device.layer(0), .queries = try query.ref().slice(4, @sizeOf(@TypeOf(queries))), .output = output.ref(), .log_mass = mass.ref(), .query_rows = 1, .query_heads = query_heads, .first_query_position = rows - 2, .scale = 0.03125, .softcap = 0.75 };
            const layout = (try Runner.splitLayout(request, true)).?;
            try std.testing.expectEqual(dim == 512 and fixture.gqa == 4, layout.padded_matrix);
            try std.testing.expect(layout.matrix);
            try std.testing.expectEqual(@as(usize, if (dim == 256) 16 else if (rows < 8192) 64 else 128), layout.rows_per_split);
            // Reuse one scratch allocation across ordinary and deferred calls.
            try runner.ensureScratch(layout.scratch_bytes);
            @memset(runner.scratch.?.bytes, 0xff);
            try runner.attend(request, .{});
            try std.testing.expectEqual(@as(usize, 2), runner.matrix_stage_dispatches);
            for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-5);
            for (expected_mass, mass.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-5);
            query.floats()[1] = std.math.nan(f32);
            try std.testing.expectError(error.NonFiniteAttention, runner.attend(request, .{}));
            query.floats()[1] = queries[0];
            const stored = &device.heads[0];
            if (fixture.fp8) {
                try std.testing.expectEqual(StorageEncoding.fp8, stored.encoding);
                try std.testing.expectEqual(StorageEncoding.exact, device.heads[1].encoding);
                const key_byte = &stored.encoded_keys.?.bytes[@sizeOf(Fp8Header)];
                const saved_key = key_byte.*;
                key_byte.* = 0x7f;
                try std.testing.expectError(error.NonFiniteAttention, runner.attend(request, .{}));
                key_byte.* = saved_key;
                const value_byte = &stored.encoded_values.?.bytes[0];
                const saved_value = value_byte.*;
                const metadata: *Metadata = @ptrCast(@alignCast(stored.metadata.?.bytes.ptr + 128 * @sizeOf(Metadata)));
                const saved_bias = metadata.bias;
                metadata.bias = -1000; // A reserved FP8 value must fail even at zero weight.
                value_byte.* = 0xff;
                try std.testing.expectError(error.NonFiniteAttention, runner.attend(request, .{}));
                value_byte.* = saved_value;
                metadata.bias = saved_bias;
            }
            const physical = (stored.start + 3) % stored.capacity;
            const half_values = std.mem.bytesAsSlice(f16, @as([]align(2) u8, @alignCast(stored.values.?.bytes)));
            const previous = half_values[physical * dim];
            half_values[physical * dim] = std.math.nan(f16);
            // Bias -1000 makes this value's weight underflow to zero. It must
            // still be checked; ignoring zero-weight NaNs would hide corruption.
            try std.testing.expectError(error.NonFiniteAttention, runner.attend(request, .{}));
            half_values[physical * dim] = previous;
            try runner.attend(request, .{});
            for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-5);
            var provider = try shared.Kernels.initSource(@embedFile("metal_attention_match_storage.metal"), "metal_attention_match_copy");
            defer provider.deinit();
            try runner.beginDeferredFrame(&device, provider.runtime, query_heads, rows, .{});
            defer runner.cancelDeferredFrame();
            // Deferred dispatch cannot clear host scratch. Empty trailing records
            // must overwrite poison on the GPU, including the shorter KV head.
            @memset(runner.scratch.?.bytes, 0xff);
            var deferred_request = request;
            deferred_request.first_query_position = rows;
            try attentionOracle(&heads, &queries, 1, query_heads, rows, fixture.window, 0.03125, 0.75, &expected, &expected_mass);
            try runner.attendOnFrame(provider.runtime, deferred_request, null, .{});
            try metal_runtime.submitFrame(provider.runtime);
            try metal_runtime.waitFrame(provider.runtime);
            const snapshot = metal_runtime.runtimeMemorySnapshot(provider.runtime);
            try std.testing.expectEqual(@as(u64, 1), snapshot.last_frame_compute_encoder_count);
            try std.testing.expectEqual(@as(u64, 1), snapshot.last_frame_planned_compute_scope_count);
            try std.testing.expectEqual(@as(usize, 3), runner.planned_dispatch_calls);
            try runner.checkStatus();
            runner.cancelDeferredFrame();
            for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-5);
            for (expected_mass, mass.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-5);
            var fallback = request;
            fallback.query_heads = 4; // Two query heads per KV head.
            try std.testing.expect(!(try Runner.splitLayout(fallback, true)).?.matrix);
            stored.window = 1024;
            try std.testing.expect(!(try Runner.splitLayout(request, true)).?.matrix);
            stored.window = fixture.window;
        }
    }
}

test "attention_match Metal parallel reducer preserves ragged empty underflow and maximum split bounds" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const Run = struct {
        fn execute(runner: *Runner, partial: Ref, mass: Ref, output: Ref, heads: usize, dim: usize, splits: usize) !void {
            runner.resetStatus();
            const inputs = [_]Ref{ partial, mass, runner.status.ref(), runner.fp8_zero.ref(), runner.fp8_zero.ref(), runner.fp8_zero.ref(), runner.fp8_zero.ref() };
            _ = try runner.split_reduce.runThreadgroupsPipeline(runner.split_reduce.pipeline, .{
                .op = 1,
                .n = @intCast(heads),
                .dim = @intCast(dim),
                .mode = 1,
                .index = @intCast(splits),
            }, &inputs, output, .{ heads, 1, 1 }, .{ 256, 1, 1 });
        }
    };
    const query_heads = 3;
    var runner = try Runner.init(128 * 1024 * 1024, null);
    defer runner.deinit();
    inline for (.{ 256, 512 }) |dim| {
        for ([_]usize{ 1, 2, 31, 32, 33, 255, 256, 257, 2048 }) |splits| {
            const width = dim + 2;
            var partial = try runner.core.buffer(query_heads * splits * width * 4);
            defer partial.deinit();
            var output = try runner.core.buffer(query_heads * dim * 4);
            defer output.deinit();
            var mass = try runner.core.buffer(query_heads * 4);
            defer mass.deinit();
            const values = partial.floats();
            @memset(values, std.math.nan(f32));
            @memset(output.floats(), std.math.nan(f32));
            @memset(mass.floats(), std.math.nan(f32));
            var expected: [query_heads * dim]f64 = @splat(0);
            var expected_mass: [query_heads]f64 = undefined;
            for (0..query_heads) |head| {
                for (0..splits) |split| {
                    const record = (head * splits + split) * width;
                    const empty = split != 0 and split % 7 == 0;
                    const den: f32 = if (empty) 0 else 1 + @as(f32, @floatFromInt(split % 5)) / 16;
                    const maximum: f32 = if (empty) std.math.nan(f32) else if (split == 1) -1000 else @as(f32, @floatFromInt(@as(i32, @intCast((split * 11 + head * 7) % 29)) - 14)) / 16;
                    values[record + dim] = den;
                    values[record + dim + 1] = maximum;
                    if (!empty) for (0..dim) |d| {
                        values[record + d] = den * @as(f32, @floatFromInt(@as(i32, @intCast((split * 13 + d * 7 + head * 3) % 31)) - 15)) / 32;
                    };
                }
                var maximum: f64 = -std.math.inf(f64);
                for (0..splits) |split| {
                    const record = (head * splits + split) * width;
                    if (values[record + dim] > 0) maximum = @max(maximum, values[record + dim + 1]);
                }
                var denominator: f64 = 0;
                for (0..splits) |split| {
                    const record = (head * splits + split) * width;
                    if (values[record + dim] == 0) continue;
                    const weight = @exp(@as(f64, values[record + dim + 1]) - maximum);
                    denominator += @as(f64, values[record + dim]) * weight;
                    for (0..dim) |d| expected[head * dim + d] += @as(f64, values[record + d]) * weight;
                }
                for (0..dim) |d| expected[head * dim + d] /= denominator;
                expected_mass[head] = maximum + @log(denominator);
            }
            try Run.execute(&runner, partial.ref(), mass.ref(), output.ref(), query_heads, dim, splits);
            try runner.checkStatus();
            for (expected, output.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-5);
            for (expected_mass, mass.floats()) |want, got| try std.testing.expectApproxEqAbs(want, @as(f64, got), 2e-5);
            if (splits > 1) {
                // This split is active with exp(-1000 - maximum) == 0. Its
                // poisoned numerator must still fail, while empty records
                // containing NaNs above must remain ignorable.
                const saved = values[width];
                values[width] = std.math.nan(f32);
                try Run.execute(&runner, partial.ref(), mass.ref(), output.ref(), query_heads, dim, splits);
                try std.testing.expectError(error.NonFiniteAttention, runner.checkStatus());
                values[width] = saved;
            }
            const saved_den = values[dim];
            for ([_]f32{ -1, std.math.nan(f32), std.math.inf(f32) }) |bad| {
                values[dim] = bad;
                try Run.execute(&runner, partial.ref(), mass.ref(), output.ref(), query_heads, dim, splits);
                try std.testing.expectError(error.NonFiniteAttention, runner.checkStatus());
            }
            values[dim] = saved_den;
            const saved_max = values[dim + 1];
            values[dim + 1] = std.math.nan(f32);
            try Run.execute(&runner, partial.ref(), mass.ref(), output.ref(), query_heads, dim, splits);
            try std.testing.expectError(error.NonFiniteAttention, runner.checkStatus());
            values[dim + 1] = saved_max;
            if (splits == 2048) {
                // Rejection occurs before reading a 2049th partial record.
                try Run.execute(&runner, partial.ref(), mass.ref(), output.ref(), query_heads, dim, splits + 1);
                try std.testing.expectError(error.NonFiniteAttention, runner.checkStatus());
            }
        }
    }
}

test "attention_match Metal request layer outputs retain GPU bytes across frames and release their owner" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    var tracker = AllocationTracker.init(null, 0);
    var runner = try Runner.init(8 * 1024 * 1024, &tracker);
    var owns_runner = true;
    defer if (owns_runner) runner.deinit();
    const baseline = tracker.snapshot();
    const original_limit = runner.max_device_bytes;
    const page = std.heap.pageSize();
    runner.max_device_bytes = @intCast(baseline.tracked_combined_live_bytes + 2 * page);
    try std.testing.expectError(error.MetalMemoryLimitExceeded, runner.reserveLayerOutputs(3, 17));
    try std.testing.expectEqual(@as(usize, 0), runner.layer_output_count);
    try std.testing.expectEqual(baseline, tracker.snapshot());
    runner.max_device_bytes = original_limit;
    try std.testing.expectError(error.InvalidAttentionShape, runner.reserveLayerOutputs(0, 17));
    try std.testing.expectError(error.InvalidAttentionShape, runner.reserveLayerOutputs(257, 17));
    try runner.reserveLayerOutputs(3, 17);
    const committed = tracker.snapshot();
    try std.testing.expectEqual(baseline.shared_backing_live_bytes + 3 * page, committed.shared_backing_live_bytes);
    var handles: [3]*anyopaque = undefined;
    for (0..3) |layer| {
        const buffer = runner.layer_outputs[layer].?;
        handles[layer] = buffer.handle;
        for (handles[0..layer]) |prior| try std.testing.expect(prior != buffer.handle);
        for ([_]Ref{ runner.status.ref(), runner.fp8_zero.ref() }) |authority|
            try std.testing.expect(!overlaps(authority, buffer.ref()));
    }
    try runner.reserveLayerOutputs(3, 17);
    try std.testing.expectEqual(committed, tracker.snapshot());
    try std.testing.expectError(error.InvalidAttentionShape, runner.reserveLayerOutputs(4, 17));
    try std.testing.expectError(error.ContextTransactionBusy, runner.layerOutput(0, 17));
    {
        var source = try runner.copier.buffer(19 * @sizeOf(f32));
        defer source.deinit();
        for (0..3) |frame| {
            for (source.floats(), 0..) |*value, i| value.* = @as(f32, @floatFromInt(i + frame * 100)) / 7;
            for (runner.layer_outputs[0..3]) |entry| @memset(entry.?.floats(), std.math.nan(f32));
            try runner.copier.beginBatch();
            runner.deferred_runtime = runner.copier.runtime;
            defer {
                runner.deferred_runtime = null;
                runner.copier.cancelBatch();
            }
            try std.testing.expectError(error.ContextStagingBudgetExceeded, runner.reserveLayerOutputs(3, 17));
            try std.testing.expectError(error.InvalidAttentionShape, runner.layerOutput(3, 17));
            try std.testing.expectError(error.InvalidAttentionShape, runner.layerOutput(0, 18));
            const before_encode = tracker.snapshot();
            for (0..3) |layer| {
                const output = try runner.layerOutput(layer, 17);
                try std.testing.expect(output.handle == handles[layer]);
                const alias: Request = .{ .heads = &.{}, .queries = source.ref(), .output = output, .query_rows = 1, .query_heads = 1, .first_query_position = 0, .scale = 1 };
                try std.testing.expectError(error.InvalidAttentionShape, runner.rejectPrivateAliases(alias));
                var alias_query = alias;
                alias_query.queries = output;
                alias_query.output = source.ref();
                try std.testing.expectError(error.InvalidAttentionShape, runner.rejectPrivateAliases(alias_query));
                var alias_mass = alias_query;
                alias_mass.queries = source.ref();
                alias_mass.log_mass = output;
                try std.testing.expectError(error.InvalidAttentionShape, runner.rejectPrivateAliases(alias_mass));
                try runner.copier.run(.{ .op = 3, .n = 17, .dim = 17 }, &.{source.ref()}, output);
            }
            try std.testing.expectEqual(before_encode, tracker.snapshot());
            try runner.copier.endBatch();
            try std.testing.expectError(error.ContextTransactionBusy, runner.layerOutput(0, 17));
            runner.deferred_runtime = null;
            for (runner.layer_outputs[0..3]) |entry|
                try std.testing.expectEqualSlices(f32, source.floats()[0..17], entry.?.floats());
        }
    }
    runner.deinit();
    owns_runner = false;
    try std.testing.expectEqual(@as(u64, 0), tracker.snapshot().tracked_combined_live_bytes);
}

test "attention_match Metal V7 two layer frontier wraps on GPU and matches separate backing" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const one = [_]LayerActivationPlan{.{ .widths = .{ 256, 128, 64, 256, 128, 32 } }};
    var one_runner = try Runner.init(16 * 1024 * 1024, null);
    defer one_runner.deinit();
    try std.testing.expectError(error.InvalidAttentionShape, one_runner.reserveLayerActivationsWithFrontier(&one, true));
    const plans = [_]LayerActivationPlan{
        .{ .widths = .{ 256, 128, 64, 256, 128, 32 } },
        .{ .widths = .{ 512, 256, 128, 512, 256, 64 } },
        .{ .widths = .{ 256, 128, 64, 256, 128, 32 } },
        .{ .widths = .{ 512, 256, 128, 512, 256, 64 } },
        .{ .widths = .{ 256, 128, 64, 256, 128, 32 } },
    };
    const Run = struct {
        fn execute(runner: *Runner, plans_: []const LayerActivationPlan, frontier: bool, step: usize) !void {
            var sources: [5][6]?Buffer = @splat(@splat(null));
            var audits: [5][6]?Buffer = @splat(@splat(null));
            defer for (&sources) |*row| for (row) |*entry| if (entry.*) |*b| b.deinit();
            defer for (&audits) |*row| for (row) |*entry| if (entry.*) |*b| b.deinit();
            for (plans_, 0..) |plan, layer| for (plan.widths, 0..) |width, role| {
                sources[layer][role] = try Buffer.init(runner.copier.runtime, width * @sizeOf(f32));
                audits[layer][role] = try Buffer.init(runner.copier.runtime, width * @sizeOf(f32));
                for (sources[layer][role].?.floats(), 0..) |*value, i|
                    value.* = @as(f32, @floatFromInt(step * 100000 + layer * 10000 + role * 1000 + i));
                @memset(audits[layer][role].?.floats(), std.math.nan(f32));
            };
            try runner.beginLayerActivationFrontierFrame();
            try runner.copier.beginBatch();
            runner.deferred_runtime = runner.copier.runtime;
            defer {
                runner.deferred_runtime = null;
                runner.copier.cancelBatch();
            }
            for (plans_, 0..) |plan, layer| {
                try runner.enterLayerActivation(layer);
                for (plan.widths, 0..) |width, role| {
                    if (frontier and layer >= 2) {
                        const old_width = plans_[layer - 2].widths[role];
                        const old = try runner.layerActivation(layer - 2, role, old_width);
                        try runner.copier.run(.{ .op = 3, .n = @intCast(old_width), .dim = @intCast(old_width) }, &.{old}, audits[layer - 2][role].?.ref());
                    }
                    const output = try runner.layerActivation(layer, role, width);
                    try runner.copier.run(.{ .op = 3, .n = @intCast(width), .dim = @intCast(width) }, &.{sources[layer][role].?.ref()}, output);
                }
            }
            for (3..5) |layer| for (plans_[layer].widths, 0..) |width, role| {
                const output = try runner.layerActivation(layer, role, width);
                try runner.copier.run(.{ .op = 3, .n = @intCast(width), .dim = @intCast(width) }, &.{output}, audits[layer][role].?.ref());
            };
            if (!frontier) for (0..3) |layer| for (plans_[layer].widths, 0..) |width, role| {
                const output = try runner.layerActivation(layer, role, width);
                try runner.copier.run(.{ .op = 3, .n = @intCast(width), .dim = @intCast(width) }, &.{output}, audits[layer][role].?.ref());
            };
            try runner.validateLayerActivationFrontierComplete();
            try runner.copier.endBatch();
            runner.deferred_runtime = null;
            for (plans_, 0..) |plan, layer| for (plan.widths, 0..) |width, role|
                try std.testing.expectEqualSlices(f32, sources[layer][role].?.floats()[0..width], audits[layer][role].?.floats()[0..width]);
        }
    };
    var legacy_tracker = AllocationTracker.init(null, 0);
    var legacy = try Runner.init(16 * 1024 * 1024, &legacy_tracker);
    defer legacy.deinit();
    const legacy_base = legacy_tracker.snapshot();
    try legacy.reserveLayerActivations(&plans);
    const legacy_bytes = legacy_tracker.snapshot().shared_backing_live_bytes - legacy_base.shared_backing_live_bytes;
    var frontier_tracker = AllocationTracker.init(null, 0);
    var frontier = try Runner.init(16 * 1024 * 1024, &frontier_tracker);
    defer frontier.deinit();
    const frontier_base = frontier_tracker.snapshot();
    const saved_limit = frontier.max_device_bytes;
    frontier.max_device_bytes = @intCast(frontier_base.tracked_combined_live_bytes + 11 * std.heap.pageSize());
    try std.testing.expectError(error.MetalMemoryLimitExceeded, frontier.reserveLayerActivationsWithFrontier(&plans, true));
    try std.testing.expectEqual(frontier_base, frontier_tracker.snapshot());
    frontier.max_device_bytes = saved_limit;
    try frontier.reserveLayerActivationsWithFrontier(&plans, true);
    const frontier_bytes = frontier_tracker.snapshot().shared_backing_live_bytes - frontier_base.shared_backing_live_bytes;
    try std.testing.expectEqual(@as(u64, 30 * std.heap.pageSize()), legacy_bytes);
    try std.testing.expectEqual(@as(u64, 12 * std.heap.pageSize()), frontier_bytes);
    try std.testing.expectEqual(@as(u64, 18 * std.heap.pageSize()), legacy_bytes - frontier_bytes);
    for (0..6) |role| {
        try std.testing.expect(frontier.layer_activation_frontier[0][role].?.handle != frontier.layer_activation_frontier[1][role].?.handle);
        for (role + 1..6) |other|
            try std.testing.expect(frontier.layer_activation_frontier[0][role].?.handle != frontier.layer_activation_frontier[0][other].?.handle);
    }
    // An unsubmitted partial frame is cancelled; the next two complete frames
    // prove the same backing survives and parity wraps without stale data.
    try frontier.beginLayerActivationFrontierFrame();
    try frontier.copier.beginBatch();
    frontier.deferred_runtime = frontier.copier.runtime;
    try frontier.enterLayerActivation(0);
    frontier.copier.cancelBatch();
    frontier.deferred_runtime = null;
    var cancelled_source = try Buffer.init(frontier.copier.runtime, 256 * @sizeOf(f32));
    defer cancelled_source.deinit();
    @memset(cancelled_source.floats(), 7);
    try frontier.beginLayerActivationFrontierFrame();
    try frontier.copier.beginBatch();
    frontier.deferred_runtime = frontier.copier.runtime;
    try frontier.enterLayerActivation(0);
    const cancelled_output = try frontier.layerActivation(0, 0, 256);
    try frontier.copier.run(.{ .op = 3, .n = 256, .dim = 256 }, &.{cancelled_source.ref()}, cancelled_output);
    frontier.copier.cancelBatch();
    frontier.deferred_runtime = null;
    for (0..2) |step| {
        try Run.execute(&legacy, &plans, false, step);
        try Run.execute(&frontier, &plans, true, step);
    }
    const before_poison = frontier_tracker.snapshot();
    std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(frontier.status.bytes[0..4])))[0] = 1;
    try std.testing.expectError(error.NonFiniteAttention, frontier.checkStatus());
    try std.testing.expectEqual(before_poison, frontier_tracker.snapshot());
    std.mem.bytesAsSlice(u32, @as([]align(4) u8, @alignCast(frontier.status.bytes[0..4])))[0] = 0;
}

test "attention_match Metal layer activation arena admits the whole plan and releases distinct frame backing" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    var tracker = AllocationTracker.init(null, 0);
    var runner = try Runner.init(8 * 1024 * 1024, &tracker);
    var owns_runner = true;
    defer if (owns_runner) runner.deinit();
    const baseline = tracker.snapshot();
    const page = std.heap.pageSize();
    const plans = [_]LayerActivationPlan{ .{ .widths = .{ 32, 16, 16, 32, 16, 8 } }, .{ .widths = .{ 64, 32, 32, 64, 32, 8 } } };
    const saved_limit = runner.max_device_bytes;
    runner.max_device_bytes = @intCast(baseline.tracked_combined_live_bytes + 11 * page);
    try std.testing.expectError(error.MetalMemoryLimitExceeded, runner.reserveLayerActivations(&plans));
    try std.testing.expectEqual(baseline, tracker.snapshot());
    try std.testing.expectEqual(@as(usize, 0), runner.layer_activation_count);
    runner.max_device_bytes = saved_limit;
    try runner.reserveLayerActivations(&plans);
    const committed = tracker.snapshot();
    try std.testing.expectEqual(baseline.shared_backing_live_bytes + 12 * page, committed.shared_backing_live_bytes);
    try runner.reserveLayerActivations(&plans);
    try std.testing.expectEqual(committed, tracker.snapshot());
    var wrong = plans;
    wrong[1].widths[0] += 1;
    try std.testing.expectError(error.InvalidAttentionShape, runner.reserveLayerActivations(&wrong));
    try std.testing.expectEqual(committed, tracker.snapshot());
    try std.testing.expectError(error.ContextTransactionBusy, runner.layerActivation(0, 0, 1));
    var handles: [12]*anyopaque = undefined;
    var count: usize = 0;
    for (runner.layer_activations[0..2]) |row| for (row) |entry| {
        const buffer = entry.?;
        for (handles[0..count]) |prior| try std.testing.expect(prior != buffer.handle);
        handles[count] = buffer.handle;
        count += 1;
    };
    runner.deinit();
    owns_runner = false;
    try std.testing.expectEqual(@as(u64, 0), tracker.snapshot().tracked_combined_live_bytes);
}
