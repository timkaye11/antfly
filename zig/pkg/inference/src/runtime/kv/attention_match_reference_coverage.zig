const std = @import("std");
const queries = @import("attention_match_queries.zig");
const Control = @import("attention_match.zig").Control;

pub const policy = "reference_coverage_v1";
pub const dense512_policy = "reference_coverage_dense512_v1";
pub const source_study_tokens: usize = 512;
pub const rotation_tokens: usize = 1;
pub const sample_count: usize = 320;
pub const dense512_sample_count: usize = 512;
pub const maximum_workspace_bytes: usize = 1024 * 1024;

pub fn policyDigest() [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(policy ++ "\nsource_study_tokens=512\nrotation_tokens=1\nsamples=320\nclass_selection=first_distinct_maximal_then_uniform_unused_v1\n");
    return h.finalResult();
}

pub fn dense512PolicyDigest() [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(dense512_policy ++ "\nsource_study_tokens=512\nrotation_tokens=1\nsamples=512\nselection=all_reference_positions_v1\n");
    return h.finalResult();
}

pub fn workspaceBytes(source_length: usize) !usize {
    const source_scratch = try std.math.mul(usize, source_length, 2 * @sizeOf(u32));
    const retained = source_study_tokens * 2 * @sizeOf(i64);
    const selection = sample_count * (@sizeOf(i64) + @sizeOf(u32) + @sizeOf(u64)) + source_study_tokens * @sizeOf(bool);
    return std.math.add(usize, @max(source_scratch + source_study_tokens * (@sizeOf(f64) + @sizeOf(u16)), retained), selection);
}

pub fn dense512WorkspaceBytes(source_length: usize) !usize {
    const source_scratch = try std.math.mul(usize, source_length, 2 * @sizeOf(u32));
    const retained = source_study_tokens * 2 * @sizeOf(i64);
    const selection = dense512_sample_count * (@sizeOf(u32) + @sizeOf(u64));
    return std.math.add(usize, @max(source_scratch + source_study_tokens * (@sizeOf(f64) + @sizeOf(u16)), retained), selection);
}

pub const Selection = struct {
    allocator: std.mem.Allocator,
    reference_tokens: []i64,
    repeat_tokens: []i64,
    offsets: []u32,
    positions: []u64,
    source_token_class_count: usize,
    selected_token_class_count: usize,
    all_classes_covered: bool,
    reference_sha256: [32]u8,
    repeat_sha256: [32]u8,
    offsets_sha256: [32]u8,
    sampler_policy_sha256: [32]u8,
    digest: [32]u8,
    pub fn deinit(self: *Selection) void {
        self.allocator.free(self.reference_tokens);
        self.allocator.free(self.repeat_tokens);
        self.allocator.free(self.offsets);
        self.allocator.free(self.positions);
        self.* = undefined;
    }
};
fn hashI64(values: []const i64) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (values) |v| {
        var b: [8]u8 = undefined;
        std.mem.writeInt(i64, &b, v, .little);
        h.update(&b);
    }
    return h.finalResult();
}
fn hashU32(values: []const u32) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    for (values) |v| {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, v, .little);
        h.update(&b);
    }
    return h.finalResult();
}
fn seal(a: [32]u8, b: [32]u8, c: [32]u8, source_length: usize, classes: usize, covered: usize) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(policy ++ "\n");
    h.update(&a);
    h.update(&b);
    h.update(&c);
    for ([_]usize{ source_length, source_study_tokens, rotation_tokens, sample_count, classes, covered }) |v| {
        var x: [8]u8 = undefined;
        std.mem.writeInt(u64, &x, @intCast(v), .little);
        h.update(&x);
    }
    return h.finalResult();
}

fn sealDense512(a: [32]u8, b: [32]u8, c: [32]u8, source_length: usize, classes: usize) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(dense512_policy ++ "\n");
    h.update(&a);
    h.update(&b);
    h.update(&c);
    for ([_]usize{ source_length, source_study_tokens, rotation_tokens, dense512_sample_count, classes, classes }) |v| {
        var x: [8]u8 = undefined;
        std.mem.writeInt(u64, &x, @intCast(v), .little);
        h.update(&x);
    }
    return h.finalResult();
}

/// Source token IDs are the only selection input. Heldout/query/metric data is
/// deliberately absent from this API.
pub fn select(allocator: std.mem.Allocator, source: []const i64, max_repeat_tokens: usize, samples: u16, control: Control) !Selection {
    if (source.len < source_study_tokens or source.len > std.math.maxInt(u32) or max_repeat_tokens != source_study_tokens or samples != sample_count) return error.InvalidCompactionQueries;
    if ((workspaceBytes(source.len) catch return error.InvalidCompactionQueries) > maximum_workspace_bytes) return error.CompactionQueryLimit;
    try control.check();
    const reference = try queries.studyTokensPass(allocator, source, source_study_tokens, 0);
    errdefer allocator.free(reference);
    const repeat = try allocator.alloc(i64, reference.len);
    errdefer allocator.free(repeat);
    @memcpy(repeat[0 .. reference.len - 1], reference[1..]);
    repeat[reference.len - 1] = reference[0];
    const chosen = try allocator.alloc(bool, repeat.len);
    defer allocator.free(chosen);
    @memset(chosen, false);
    const offsets = try allocator.alloc(u32, sample_count);
    errdefer allocator.free(offsets);
    var distinct: usize = 0;
    var count: usize = 0;
    for (repeat, 0..) |token, index| {
        if ((index & 63) == 0) try control.check();
        var seen = false;
        for (repeat[0..index]) |old| if (old == token) {
            seen = true;
            break;
        };
        if (seen) continue;
        distinct += 1;
        if (count < sample_count) {
            offsets[count] = @intCast(index);
            chosen[index] = true;
            count += 1;
        }
    }
    if (distinct == 0) return error.InvalidCompactionQueries;
    const fill = sample_count - count;
    for (0..fill) |slot| {
        try control.check();
        var index = slot * repeat.len / @max(fill, 1);
        var searched: usize = 0;
        while (chosen[index] and searched < repeat.len) : (searched += 1) index = (index + 1) % repeat.len;
        if (searched == repeat.len) return error.InvalidCompactionQueries;
        chosen[index] = true;
        offsets[count] = @intCast(index);
        count += 1;
    }
    std.mem.sort(u32, offsets, {}, struct {
        fn less(_: void, a: u32, b: u32) bool {
            return a < b;
        }
    }.less);
    for (offsets, 0..) |offset, index| if (offset >= repeat.len or (index > 0 and offsets[index - 1] >= offset)) return error.InvalidCompactionQueries;
    const positions = try allocator.alloc(u64, sample_count);
    errdefer allocator.free(positions);
    for (offsets, positions) |offset, *position| position.* = std.math.add(u64, @intCast(source.len), offset) catch return error.InvalidCompactionQueries;
    const a = hashI64(reference);
    const b = hashI64(repeat);
    const c = hashU32(offsets);
    try control.check();
    const covered = @min(distinct, sample_count);
    return .{ .allocator = allocator, .reference_tokens = reference, .repeat_tokens = repeat, .offsets = offsets, .positions = positions, .source_token_class_count = distinct, .selected_token_class_count = covered, .all_classes_covered = distinct == covered, .reference_sha256 = a, .repeat_sha256 = b, .offsets_sha256 = c, .sampler_policy_sha256 = policyDigest(), .digest = seal(a, b, c, source.len, distinct, covered) };
}

pub fn verify(allocator: std.mem.Allocator, source: []const i64, reference: []const i64, repeat: []const i64, offsets: []const u32, control: Control) !void {
    var expected = try select(allocator, source, source_study_tokens, sample_count, control);
    defer expected.deinit();
    if (!std.mem.eql(i64, reference, expected.reference_tokens) or !std.mem.eql(i64, repeat, expected.repeat_tokens) or !std.mem.eql(u32, offsets, expected.offsets))
        return error.InvalidCompactionQueries;
}

pub fn selectDense512(allocator: std.mem.Allocator, source: []const i64, max_repeat_tokens: usize, control: Control) !Selection {
    if (source.len < source_study_tokens or source.len > std.math.maxInt(u32) or max_repeat_tokens != source_study_tokens) return error.InvalidCompactionQueries;
    if ((dense512WorkspaceBytes(source.len) catch return error.InvalidCompactionQueries) > maximum_workspace_bytes) return error.CompactionQueryLimit;
    try control.check();
    const reference = try queries.studyTokensPass(allocator, source, source_study_tokens, 0);
    errdefer allocator.free(reference);
    const repeat = try allocator.alloc(i64, reference.len);
    errdefer allocator.free(repeat);
    @memcpy(repeat[0 .. reference.len - 1], reference[1..]);
    repeat[reference.len - 1] = reference[0];
    const offsets = try allocator.alloc(u32, dense512_sample_count);
    errdefer allocator.free(offsets);
    const positions = try allocator.alloc(u64, dense512_sample_count);
    errdefer allocator.free(positions);
    var distinct: usize = 0;
    for (repeat, offsets, positions, 0..) |token, *offset, *position, index| {
        if ((index & 63) == 0) try control.check();
        var seen = false;
        for (repeat[0..index]) |old| if (old == token) {
            seen = true;
            break;
        };
        if (!seen) distinct += 1;
        offset.* = @intCast(index);
        position.* = std.math.add(u64, @intCast(source.len), index) catch return error.InvalidCompactionQueries;
    }
    const a = hashI64(reference);
    const b = hashI64(repeat);
    const c = hashU32(offsets);
    try control.check();
    return .{ .allocator = allocator, .reference_tokens = reference, .repeat_tokens = repeat, .offsets = offsets, .positions = positions, .source_token_class_count = distinct, .selected_token_class_count = distinct, .all_classes_covered = true, .reference_sha256 = a, .repeat_sha256 = b, .offsets_sha256 = c, .sampler_policy_sha256 = dense512PolicyDigest(), .digest = sealDense512(a, b, c, source.len, distinct) };
}

pub fn verifyDense512(allocator: std.mem.Allocator, source: []const i64, reference: []const i64, repeat: []const i64, offsets: []const u32, control: Control) !void {
    var expected = try selectDense512(allocator, source, source_study_tokens, control);
    defer expected.deinit();
    if (!std.mem.eql(i64, reference, expected.reference_tokens) or !std.mem.eql(i64, repeat, expected.repeat_tokens) or !std.mem.eql(u32, offsets, expected.offsets))
        return error.InvalidCompactionQueries;
}

test "dense512 keeps legacy320 authority unchanged and selects every position" {
    try std.testing.expectEqualStrings("49a5ea01455da82bc3082c7415d20201c5d4c475d2fb7e544cea63198b5533f2", &std.fmt.bytesToHex(policyDigest(), .lower));
    try std.testing.expectEqualStrings("72f2b172ac301aa341e89ce678f043e8ac3e10c607631c6226c5d3f081f0efa0", &std.fmt.bytesToHex(dense512PolicyDigest(), .lower));
    var source: [768]i64 = undefined;
    for (&source, 0..) |*value, i| value.* = @intCast(i % 266);
    var dense = try selectDense512(std.testing.allocator, &source, 512, .{});
    defer dense.deinit();
    try std.testing.expectEqual(@as(usize, 512), dense.offsets.len);
    try std.testing.expectEqual(@as(usize, 266), dense.source_token_class_count);
    try std.testing.expectEqual(@as(usize, 266), dense.selected_token_class_count);
    try std.testing.expect(dense.all_classes_covered);
    for (dense.offsets, 0..) |offset, i| try std.testing.expectEqual(@as(u32, @intCast(i)), offset);
}

fn denseAllocationFailure(allocator: std.mem.Allocator) !void {
    var source: [768]i64 = undefined;
    for (&source, 0..) |*value, i| value.* = @intCast(i % 17);
    var selection = try selectDense512(allocator, &source, 512, .{});
    selection.deinit();
}

test "dense512 allocation unwind cancellation and shape" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, denseAllocationFailure, .{});
    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.Cancelled;
        }
    };
    var source: [768]i64 = @splat(1);
    try std.testing.expectError(error.Cancelled, selectDense512(std.testing.allocator, &source, 512, .{ .check_fn = Cancel.check }));
    try std.testing.expectError(error.InvalidCompactionQueries, selectDense512(std.testing.allocator, &source, 511, .{}));
}

test "coverage rotation uniqueness classes and anti alias" {
    var source: [768]i64 = undefined;
    for (&source, 0..) |*v, i| v.* = @intCast(i % 266);
    var s = try select(std.testing.allocator, &source, 512, 320, .{});
    defer s.deinit();
    try std.testing.expectEqual(@as(usize, 266), s.source_token_class_count);
    try std.testing.expectEqual(@as(usize, 266), s.selected_token_class_count);
    try std.testing.expect(s.all_classes_covered);
    try std.testing.expectEqualSlices(i64, s.reference_tokens[1..], s.repeat_tokens[0..511]);
    try std.testing.expectEqual(s.reference_tokens[0], s.repeat_tokens[511]);
    var found: [266]bool = @splat(false);
    var mod8: [8]bool = @splat(false);
    for (s.offsets) |o| {
        found[@intCast(s.repeat_tokens[o])] = true;
        mod8[o % 8] = true;
    }
    for (found) |v| try std.testing.expect(v);
    for (mod8) |v| try std.testing.expect(v);
}
test "coverage above budget is maximal deterministic and truthful" {
    var source: [768]i64 = undefined;
    for (&source, 0..) |*v, i| v.* = @intCast(i);
    var s = try select(std.testing.allocator, &source, 512, 320, .{});
    defer s.deinit();
    try std.testing.expectEqual(@as(usize, 512), s.source_token_class_count);
    try std.testing.expectEqual(@as(usize, 320), s.selected_token_class_count);
    try std.testing.expect(!s.all_classes_covered);
    for (s.offsets, 0..) |offset, i| try std.testing.expectEqual(@as(u32, @intCast(i)), offset);
}
test "coverage workspace cap is bounded" {
    try std.testing.expect((try workspaceBytes(768)) < maximum_workspace_bytes);
    try std.testing.expect((try workspaceBytes(200000)) > maximum_workspace_bytes);
}
fn allocationFailure(a: std.mem.Allocator) !void {
    var source: [768]i64 = undefined;
    for (&source, 0..) |*v, i| v.* = @intCast(i % 17);
    var s = try select(a, &source, 512, 320, .{});
    s.deinit();
}
test "coverage allocation unwind cancellation and shape" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailure, .{});
    const Cancel = struct {
        fn check(_: ?*anyopaque) anyerror!void {
            return error.Cancelled;
        }
    };
    var source: [768]i64 = @splat(1);
    try std.testing.expectError(error.Cancelled, select(std.testing.allocator, &source, 512, 320, .{ .check_fn = Cancel.check }));
    try std.testing.expectError(error.InvalidCompactionQueries, select(std.testing.allocator, &source, 511, 320, .{}));
}
