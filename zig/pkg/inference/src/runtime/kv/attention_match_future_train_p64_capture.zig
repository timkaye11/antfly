const std = @import("std");
const engine = @import("attention_match_engine.zig");

pub const schema = "antfly.attention-match-teacher-forced-future-train-p64.v1";
pub const source_tokens: usize = 8128;
pub const continuation_rows: usize = 512;
pub const sample_rows: usize = 64;
pub const sample_stride: usize = 8;
pub const sample_first_row: usize = 4;
pub const first_training_position: usize = source_tokens + sample_first_row;
pub const query_heads: usize = 8;
pub const head_dim: usize = 512;
pub const consumer_count: usize = 7;
pub const batch_count: usize = consumer_count * query_heads;

pub const Identity = struct { weights: [32]u8, tokenizer: [32]u8, template: [32]u8, layout: [32]u8 };
pub const Consumer = struct { donor_layer: usize, consumer_layer: usize };
pub const Control = struct {
    context: ?*anyopaque = null,
    check_fn: ?*const fn (?*anyopaque) anyerror!void = null,
    pub fn check(self: @This()) !void {
        if (self.check_fn) |f| try f(self.context);
    }
};

pub const Authority = struct {
    source_freeze_sha256: [32]u8,
    implementation_semantic_sha256: [32]u8,
    execution_sha256: [32]u8,
    source_checkpoint_sha256: [32]u8,
    source_prefix_sha256: [32]u8,
    continuation_ids_sha256: [32]u8,
    continuation_origin_sha256: [32]u8,
    profile_sha256: [32]u8,
    identity: Identity,
    source_revision: u64,
    capture_epoch: u64,
    device_registry_id: u64,
    device_name_sha256: [32]u8,
    compiler_identity_sha256: [32]u8,
    backend_environment_sha256: [32]u8,

    pub fn validate(self: @This()) !void {
        if (self.source_revision == 0 or self.capture_epoch == 0 or self.device_registry_id == 0) return error.InvalidFutureTrainP64Authority;
        inline for (.{ self.source_freeze_sha256, self.implementation_semantic_sha256, self.execution_sha256, self.source_checkpoint_sha256, self.source_prefix_sha256, self.continuation_ids_sha256, self.continuation_origin_sha256, self.profile_sha256, self.identity.weights, self.identity.tokenizer, self.identity.template, self.identity.layout, self.device_name_sha256, self.compiler_identity_sha256, self.backend_environment_sha256 }) |field_digest|
            if (std.mem.allEqual(u8, &field_digest, 0)) return error.InvalidFutureTrainP64Authority;
    }

    pub fn digest(self: @This()) [32]u8 {
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(schema ++ "\x00");
        inline for (.{ self.source_freeze_sha256, self.implementation_semantic_sha256, self.execution_sha256, self.source_checkpoint_sha256, self.source_prefix_sha256, self.continuation_ids_sha256, self.continuation_origin_sha256, self.profile_sha256, self.identity.weights, self.identity.tokenizer, self.identity.template, self.identity.layout, self.device_name_sha256, self.compiler_identity_sha256, self.backend_environment_sha256 }) |field_digest| hash.update(&field_digest);
        inline for (.{ self.source_revision, self.capture_epoch, self.device_registry_id }) |value| {
            var bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &bytes, value, .little);
            hash.update(&bytes);
        }
        return hash.finalResult();
    }
};

pub const Entry = struct { tuple: Consumer, captured: bool = false };
pub const TrainingOutput = struct {
    allocator: std.mem.Allocator,
    batches: []engine.ReferenceBatch,
    values: []f32,
    positions: []u64,
    authority_sha256: [32]u8,
    payload_sha256: [32]u8,
    pub fn deinit(self: *@This()) void {
        self.allocator.free(self.positions);
        self.allocator.free(self.values);
        self.allocator.free(self.batches);
        self.* = undefined;
    }
};

pub const Owner = struct {
    allocator: std.mem.Allocator,
    authority: Authority,
    entries: []Entry,
    values: []f32,
    control: Control,
    sealed: bool = false,

    pub fn requiredBytes() !usize {
        return std.math.add(usize, consumer_count * @sizeOf(Entry), try std.math.mul(usize, consumer_count * sample_rows * query_heads * head_dim, @sizeOf(f32)));
    }
    pub fn init(allocator: std.mem.Allocator, authority: Authority, topology: [consumer_count]Consumer, maximum_bytes: usize, control: Control) !Owner {
        try authority.validate();
        try control.check();
        if (maximum_bytes != try requiredBytes()) return error.FutureTrainP64CaptureCapMismatch;
        const entries = try allocator.alloc(Entry, consumer_count);
        errdefer allocator.free(entries);
        for (entries, topology, 0..) |*entry, tuple, i| {
            for (topology[0..i]) |prior| if (std.meta.eql(prior, tuple)) return error.DuplicateFutureTrainP64Consumer;
            entry.* = .{ .tuple = tuple };
        }
        const values = try allocator.alloc(f32, consumer_count * sample_rows * query_heads * head_dim);
        errdefer allocator.free(values);
        return .{ .allocator = allocator, .authority = authority, .entries = entries, .values = values, .control = control };
    }
    pub fn deinit(self: *Owner) void {
        self.allocator.free(self.values);
        self.allocator.free(self.entries);
        self.* = undefined;
    }
    fn index(self: *const Owner, donor: usize, consumer: usize) !usize {
        for (self.entries, 0..) |entry, i| if (entry.tuple.donor_layer == donor and entry.tuple.consumer_layer == consumer) return i;
        return error.UnexpectedFutureTrainP64Consumer;
    }
    pub fn capture(self: *Owner, donor: usize, consumer: usize, first_position: usize, heads: usize, dim: usize, incoming: []const f32) !void {
        try self.control.check();
        if (self.sealed or heads != query_heads or dim != head_dim or first_position != first_training_position or incoming.len != sample_rows * query_heads * head_dim) return error.InvalidFutureTrainP64Capture;
        for (incoming) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttentionInput;
        const i = try self.index(donor, consumer);
        if (self.entries[i].captured) return error.DuplicateFutureTrainP64Consumer;
        @memcpy(self.values[i * sample_rows * query_heads * head_dim ..][0..incoming.len], incoming);
        self.entries[i].captured = true;
    }
    pub fn seal(self: *Owner, allocator: std.mem.Allocator) !TrainingOutput {
        if (self.sealed) return error.FutureTrainP64CaptureSealed;
        const batches = try allocator.alloc(engine.ReferenceBatch, batch_count);
        errdefer allocator.free(batches);
        const positions = try allocator.alloc(u64, batch_count * sample_rows);
        errdefer allocator.free(positions);
        const values = try allocator.alloc(f32, batch_count * sample_rows * head_dim);
        errdefer allocator.free(values);
        var at: usize = 0;
        for (self.entries, 0..) |entry, ei| {
            if (!entry.captured) return error.IncompleteFutureTrainP64Capture;
            for (0..query_heads) |head| {
                const p = positions[at * sample_rows ..][0..sample_rows];
                for (p, 0..) |*value, row| value.* = first_training_position + row * sample_stride;
                const out = values[at * sample_rows * head_dim ..][0 .. sample_rows * head_dim];
                const src = self.values[ei * sample_rows * query_heads * head_dim ..][0 .. sample_rows * query_heads * head_dim];
                for (0..sample_rows) |row| @memcpy(out[row * head_dim ..][0..head_dim], src[(row * query_heads + head) * head_dim ..][0..head_dim]);
                batches[at] = .{ .consumer_layer = @intCast(entry.tuple.consumer_layer), .query_head = @intCast(head), .train = out, .train_positions = p, .heldout = &.{}, .heldout_positions = &.{} };
                at += 1;
            }
        }
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(schema ++ ".payload\x00");
        hash.update(std.mem.sliceAsBytes(values));
        hash.update(std.mem.sliceAsBytes(positions));
        self.sealed = true;
        return .{ .allocator = allocator, .batches = batches, .values = values, .positions = positions, .authority_sha256 = self.authority.digest(), .payload_sha256 = hash.finalResult() };
    }
};

fn fixtureAuthority() Authority {
    return .{ .source_freeze_sha256 = @splat(1), .implementation_semantic_sha256 = @splat(2), .execution_sha256 = @splat(3), .source_checkpoint_sha256 = @splat(4), .source_prefix_sha256 = @splat(5), .continuation_ids_sha256 = @splat(6), .continuation_origin_sha256 = @splat(7), .profile_sha256 = @splat(8), .identity = .{ .weights = @splat(9), .tokenizer = @splat(10), .template = @splat(11), .layout = @splat(12) }, .source_revision = 1, .capture_epoch = 2, .device_registry_id = 3, .device_name_sha256 = @splat(13), .compiler_identity_sha256 = @splat(14), .backend_environment_sha256 = @splat(15) };
}
fn fixtureTopology() [consumer_count]Consumer {
    return .{ .{ .donor_layer = 5, .consumer_layer = 5 }, .{ .donor_layer = 11, .consumer_layer = 11 }, .{ .donor_layer = 17, .consumer_layer = 17 }, .{ .donor_layer = 23, .consumer_layer = 23 }, .{ .donor_layer = 23, .consumer_layer = 29 }, .{ .donor_layer = 23, .consumer_layer = 35 }, .{ .donor_layer = 23, .consumer_layer = 41 } };
}

test "future TRAIN P64 owner emits train-only stride-eight positions from row four" {
    var owner = try Owner.init(std.testing.allocator, fixtureAuthority(), fixtureTopology(), try Owner.requiredBytes(), .{});
    defer owner.deinit();
    const values = try std.testing.allocator.alloc(f32, sample_rows * query_heads * head_dim);
    defer std.testing.allocator.free(values);
    @memset(values, 1);
    for (fixtureTopology()) |tuple| try owner.capture(tuple.donor_layer, tuple.consumer_layer, first_training_position, query_heads, head_dim, values);
    var training = try owner.seal(std.testing.allocator);
    defer training.deinit();
    try std.testing.expectEqual(batch_count, training.batches.len);
    for (training.batches) |batch| {
        try std.testing.expectEqual(@as(usize, 0), batch.heldout.len);
        try std.testing.expectEqual(@as(usize, sample_rows * head_dim), batch.train.len);
        try std.testing.expectEqual(@as(u64, 8132), batch.train_positions[0]);
        try std.testing.expectEqual(@as(u64, 8636), batch.train_positions[63]);
    }
}

test "future TRAIN P64 owner rejects position domain and duplicate tuple" {
    var topology = fixtureTopology();
    topology[1] = topology[0];
    try std.testing.expectError(error.DuplicateFutureTrainP64Consumer, Owner.init(std.testing.allocator, fixtureAuthority(), topology, try Owner.requiredBytes(), .{}));
    var owner = try Owner.init(std.testing.allocator, fixtureAuthority(), fixtureTopology(), try Owner.requiredBytes(), .{});
    defer owner.deinit();
    var values: [sample_rows * query_heads * head_dim]f32 = @splat(0);
    try std.testing.expectError(error.InvalidFutureTrainP64Capture, owner.capture(5, 5, 8128, query_heads, head_dim, &values));
    try owner.capture(5, 5, first_training_position, query_heads, head_dim, &values);
    try std.testing.expectError(error.DuplicateFutureTrainP64Consumer, owner.capture(5, 5, first_training_position, query_heads, head_dim, &values));
}
