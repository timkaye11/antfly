const std = @import("std");
const engine = @import("attention_match_engine.zig");

pub const policy = "prefill_recent64_train_v1|E4_8128|post_rope_f32|final64|train_only|no_heldout|default_off";
pub const horizon: usize = 64;
pub const query_heads: usize = 8;
pub const head_dim: usize = 512;
pub const consumer_count: usize = 7;
pub const batch_count: usize = consumer_count * query_heads;

pub const Control = struct {
    context: ?*anyopaque = null,
    check_fn: ?*const fn (?*anyopaque) anyerror!void = null,
    pub fn check(self: Control) !void {
        if (self.check_fn) |f| try f(self.context);
    }
};

pub const Identity = struct {
    weights: [32]u8,
    tokenizer: [32]u8,
    template: [32]u8,
    layout: [32]u8,
};

pub const Authority = struct {
    source_freeze_sha256: [32]u8,
    implementation_semantic_sha256: [32]u8,
    execution_sha256: [32]u8,
    source_checkpoint_sha256: [32]u8,
    prompt_tokens_sha256: [32]u8,
    profile_sha256: [32]u8,
    policy_sha256: [32]u8,
    identity: Identity,
    source_revision: u64,
    capture_epoch: u64,
    device_registry_id: u64,
    device_name_sha256: [32]u8,
    compiler_identity_sha256: [32]u8,
    backend_environment_sha256: [32]u8,
    logical_tokens: usize,
    fitted_middle_end_exclusive: usize,

    pub fn validate(self: Authority) !void {
        if (self.logical_tokens != 8128 or self.fitted_middle_end_exclusive != 6080 or
            self.source_revision == 0 or self.capture_epoch == 0 or self.device_registry_id == 0)
            return error.InvalidRecentQueryAuthority;
        inline for (.{ self.source_freeze_sha256, self.implementation_semantic_sha256, self.execution_sha256, self.source_checkpoint_sha256, self.prompt_tokens_sha256, self.profile_sha256, self.policy_sha256, self.identity.weights, self.identity.tokenizer, self.identity.template, self.identity.layout, self.device_name_sha256, self.compiler_identity_sha256, self.backend_environment_sha256 }) |digest| {
            var nonzero = false;
            for (digest) |byte| nonzero = nonzero or byte != 0;
            if (!nonzero) return error.InvalidRecentQueryAuthority;
        }
    }
};

pub const Consumer = struct { donor_layer: usize, consumer_layer: usize };

const Entry = struct {
    tuple: Consumer,
    count: usize = 0,
};

pub const Seal = struct {
    authority_sha256: [32]u8,
    portfolio_sha256: [32]u8,
    logical_live_bytes: usize,
    capture_epoch: u64,
};

pub const Training = struct {
    allocator: std.mem.Allocator,
    batches: []engine.ReferenceBatch,
    values: []f32,
    positions: []u64,
    authority_sha256: [32]u8,
    portfolio_sha256: [32]u8,

    pub fn deinit(self: *Training) void {
        self.allocator.free(self.positions);
        self.allocator.free(self.values);
        self.allocator.free(self.batches);
        self.* = undefined;
    }
};

pub fn trainingRequiredBytes() !usize {
    var bytes = try std.math.mul(usize, batch_count, @sizeOf(engine.ReferenceBatch));
    bytes = try std.math.add(usize, bytes, try std.math.mul(usize, batch_count * horizon, @sizeOf(u64)));
    return std.math.add(usize, bytes, try std.math.mul(usize, batch_count * horizon * head_dim, @sizeOf(f32)));
}

pub const Owner = struct {
    allocator: std.mem.Allocator,
    authority: Authority,
    authority_sha256: [32]u8,
    entries: []Entry,
    positions: []u64,
    values: []f32,
    maximum_live_bytes: usize,
    control: Control,
    sealed: bool = false,

    pub fn requiredBytes() !usize {
        var bytes = try std.math.mul(usize, consumer_count, @sizeOf(Entry));
        bytes = try std.math.add(usize, bytes, try std.math.mul(usize, consumer_count * horizon, @sizeOf(u64)));
        return std.math.add(usize, bytes, try std.math.mul(usize, consumer_count * horizon * query_heads * head_dim, @sizeOf(f32)));
    }

    pub fn init(allocator: std.mem.Allocator, authority: Authority, topology: [consumer_count]Consumer, maximum_live_bytes: usize, control: Control) !Owner {
        try authority.validate();
        try validateTopology(topology);
        try control.check();
        const required = try requiredBytes();
        if (maximum_live_bytes != required) return error.RecentQueryCaptureCapMismatch;
        const entries = try allocator.alloc(Entry, consumer_count);
        errdefer allocator.free(entries);
        const positions = try allocator.alloc(u64, consumer_count * horizon);
        errdefer allocator.free(positions);
        const values = try allocator.alloc(f32, consumer_count * horizon * query_heads * head_dim);
        errdefer allocator.free(values);
        for (entries, topology) |*entry, tuple| entry.* = .{ .tuple = tuple };
        return .{ .allocator = allocator, .authority = authority, .authority_sha256 = authorityDigest(authority), .entries = entries, .positions = positions, .values = values, .maximum_live_bytes = maximum_live_bytes, .control = control };
    }

    pub fn deinit(self: *Owner) void {
        self.allocator.free(self.values);
        self.allocator.free(self.positions);
        self.allocator.free(self.entries);
        self.* = undefined;
    }

    pub fn liveBytes(self: *const Owner) usize {
        return self.entries.len * @sizeOf(Entry) + self.positions.len * @sizeOf(u64) + self.values.len * @sizeOf(f32);
    }

    fn entryIndex(self: *const Owner, donor_layer: usize, consumer_layer: usize) !usize {
        var found: ?usize = null;
        for (self.entries, 0..) |entry, index| if (entry.tuple.donor_layer == donor_layer and entry.tuple.consumer_layer == consumer_layer) {
            if (found != null) return error.DuplicateRecentQueryConsumer;
            found = index;
        };
        return found orelse error.UnexpectedRecentQueryConsumer;
    }

    pub fn capture(self: *Owner, donor_layer: usize, consumer_layer: usize, first_position: usize, num_heads: usize, dim: usize, incoming: []const f32) !void {
        if (self.sealed) return error.RecentQueryCaptureSealed;
        try self.control.check();
        if (num_heads != query_heads or dim != head_dim or incoming.len == 0 or incoming.len % (query_heads * head_dim) != 0)
            return error.InvalidRecentQueryShape;
        for (incoming) |value| if (!std.math.isFinite(value)) return error.NonFiniteAttentionInput;
        const rows = incoming.len / (query_heads * head_dim);
        if (rows > horizon or first_position > self.authority.logical_tokens or rows > self.authority.logical_tokens - first_position)
            return error.InvalidRecentQueryPosition;
        const index = try self.entryIndex(donor_layer, consumer_layer);
        const entry = &self.entries[index];
        const pos = self.positions[index * horizon ..][0..horizon];
        const value_width = query_heads * head_dim;
        const vals = self.values[index * horizon * value_width ..][0 .. horizon * value_width];

        // An exact retry is accepted only when every overlapping position and
        // f32 bit agrees. Different epochs require a different owner.
        for (0..entry.count) |old_row| if (pos[old_row] >= first_position and pos[old_row] < first_position + rows) {
            const new_row = pos[old_row] - first_position;
            if (!std.mem.eql(u8, std.mem.sliceAsBytes(vals[old_row * value_width ..][0..value_width]), std.mem.sliceAsBytes(incoming[new_row * value_width ..][0..value_width])))
                return error.RecentQueryRetryMismatch;
        };

        if (rows == horizon) {
            entry.count = 0;
        } else {
            while (entry.count > 0 and pos[entry.count - 1] >= first_position) entry.count -= 1;
            if (entry.count > 0 and pos[entry.count - 1] + 1 != first_position)
                return error.NonContiguousRecentQueries;
            const keep = @min(entry.count, horizon - rows);
            if (keep < entry.count) {
                const begin = entry.count - keep;
                std.mem.copyForwards(u64, pos[0..keep], pos[begin..entry.count]);
                std.mem.copyForwards(f32, vals[0 .. keep * value_width], vals[begin * value_width .. entry.count * value_width]);
                entry.count = keep;
            }
        }
        for (0..rows) |row| pos[entry.count + row] = first_position + row;
        @memcpy(vals[entry.count * value_width ..][0..incoming.len], incoming);
        entry.count += rows;
    }

    pub fn captureCallback(ctx: *anyopaque, donor_layer: usize, consumer_layer: usize, first_position: usize, num_heads: usize, dim: usize, values: []const f32) anyerror!void {
        const self: *Owner = @ptrCast(@alignCast(ctx));
        try self.capture(donor_layer, consumer_layer, first_position, num_heads, dim, values);
    }

    pub fn seal(self: *Owner) !Seal {
        if (self.sealed) return error.RecentQueryCaptureSealed;
        try self.control.check();
        const expected_first = self.authority.logical_tokens - horizon;
        if (expected_first < self.authority.fitted_middle_end_exclusive) return error.RecentQueryNotFullCausalPast;
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(policy ++ "\x00");
        hash.update(&self.authority_sha256);
        for (self.entries, 0..) |entry, index| {
            if (entry.count != horizon) return error.IncompleteRecentQueryCapture;
            const positions = self.positions[index * horizon ..][0..horizon];
            for (positions, 0..) |position, row| if (position != expected_first + row) return error.InvalidRecentQueryPosition;
            var donor_bytes: [8]u8 = undefined;
            var consumer_bytes: [8]u8 = undefined;
            std.mem.writeInt(u64, &donor_bytes, @intCast(entry.tuple.donor_layer), .little);
            std.mem.writeInt(u64, &consumer_bytes, @intCast(entry.tuple.consumer_layer), .little);
            hash.update(&donor_bytes);
            hash.update(&consumer_bytes);
            hash.update(std.mem.sliceAsBytes(positions));
            const width = horizon * query_heads * head_dim;
            hash.update(std.mem.sliceAsBytes(self.values[index * width ..][0..width]));
        }
        self.sealed = true;
        return .{ .authority_sha256 = self.authority_sha256, .portfolio_sha256 = hash.finalResult(), .logical_live_bytes = self.liveBytes(), .capture_epoch = self.authority.capture_epoch };
    }

    pub fn training(self: *const Owner, allocator: std.mem.Allocator, seal_value: Seal) !Training {
        if (!self.sealed or !std.mem.eql(u8, &seal_value.authority_sha256, &self.authority_sha256)) return error.InvalidRecentQuerySeal;
        const batches = try allocator.alloc(engine.ReferenceBatch, batch_count);
        errdefer allocator.free(batches);
        const positions = try allocator.alloc(u64, batch_count * horizon);
        errdefer allocator.free(positions);
        const values = try allocator.alloc(f32, batch_count * horizon * head_dim);
        errdefer allocator.free(values);
        var batch_at: usize = 0;
        for (self.entries, 0..) |entry, entry_index| for (0..query_heads) |head| {
            try self.control.check();
            const output_positions = positions[batch_at * horizon ..][0..horizon];
            @memcpy(output_positions, self.positions[entry_index * horizon ..][0..horizon]);
            const output_values = values[batch_at * horizon * head_dim ..][0 .. horizon * head_dim];
            const source_values = self.values[entry_index * horizon * query_heads * head_dim ..][0 .. horizon * query_heads * head_dim];
            for (0..horizon) |row| @memcpy(output_values[row * head_dim ..][0..head_dim], source_values[(row * query_heads + head) * head_dim ..][0..head_dim]);
            batches[batch_at] = .{ .consumer_layer = @intCast(entry.tuple.consumer_layer), .query_head = @intCast(head), .train = output_values, .train_positions = output_positions, .heldout = &.{}, .heldout_positions = &.{} };
            batch_at += 1;
        };
        return .{ .allocator = allocator, .batches = batches, .values = values, .positions = positions, .authority_sha256 = seal_value.authority_sha256, .portfolio_sha256 = seal_value.portfolio_sha256 };
    }
};

fn validateTopology(topology: [consumer_count]Consumer) !void {
    for (topology, 0..) |tuple, index| {
        if (tuple.donor_layer > tuple.consumer_layer) return error.InvalidRecentQueryTopology;
        for (topology[0..index]) |prior| if (std.meta.eql(prior, tuple)) return error.DuplicateRecentQueryConsumer;
    }
}

fn authorityDigest(authority: Authority) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(policy ++ "\x00");
    inline for (.{ authority.source_freeze_sha256, authority.implementation_semantic_sha256, authority.execution_sha256, authority.source_checkpoint_sha256, authority.prompt_tokens_sha256, authority.profile_sha256, authority.policy_sha256, authority.identity.weights, authority.identity.tokenizer, authority.identity.template, authority.identity.layout, authority.device_name_sha256, authority.compiler_identity_sha256, authority.backend_environment_sha256 }) |digest| hash.update(&digest);
    inline for (.{ authority.source_revision, authority.capture_epoch, authority.device_registry_id, @as(u64, @intCast(authority.logical_tokens)), @as(u64, @intCast(authority.fitted_middle_end_exclusive)) }) |number| {
        var encoded: [8]u8 = undefined;
        std.mem.writeInt(u64, &encoded, number, .little);
        hash.update(&encoded);
    }
    return hash.finalResult();
}

fn fixtureAuthority() Authority {
    return .{ .source_freeze_sha256 = @splat(1), .implementation_semantic_sha256 = @splat(2), .execution_sha256 = @splat(3), .source_checkpoint_sha256 = @splat(4), .prompt_tokens_sha256 = @splat(5), .profile_sha256 = @splat(6), .policy_sha256 = @splat(7), .identity = .{ .weights = @splat(8), .tokenizer = @splat(9), .template = @splat(10), .layout = @splat(11) }, .source_revision = 1, .capture_epoch = 2, .device_registry_id = 3, .device_name_sha256 = @splat(12), .compiler_identity_sha256 = @splat(13), .backend_environment_sha256 = @splat(14), .logical_tokens = 8128, .fitted_middle_end_exclusive = 6080 };
}

fn fixtureTopology() [consumer_count]Consumer {
    return .{ .{ .donor_layer = 5, .consumer_layer = 5 }, .{ .donor_layer = 11, .consumer_layer = 11 }, .{ .donor_layer = 17, .consumer_layer = 17 }, .{ .donor_layer = 23, .consumer_layer = 23 }, .{ .donor_layer = 23, .consumer_layer = 29 }, .{ .donor_layer = 23, .consumer_layer = 35 }, .{ .donor_layer = 23, .consumer_layer = 41 } };
}

test "recent64 capture seals causal exact retries and adapts train-only batches" {
    var owner = try Owner.init(std.testing.allocator, fixtureAuthority(), fixtureTopology(), try Owner.requiredBytes(), .{});
    defer owner.deinit();
    const values = try std.testing.allocator.alloc(f32, horizon * query_heads * head_dim);
    defer std.testing.allocator.free(values);
    for (values, 0..) |*value, index| value.* = @floatFromInt(index % 997);
    for (fixtureTopology()) |tuple| {
        try owner.capture(tuple.donor_layer, tuple.consumer_layer, 8064, query_heads, head_dim, values);
        try owner.capture(tuple.donor_layer, tuple.consumer_layer, 8064, query_heads, head_dim, values);
    }
    const sealed = try owner.seal();
    var training = try owner.training(std.testing.allocator, sealed);
    defer training.deinit();
    try std.testing.expectEqual(@as(usize, 56), training.batches.len);
    for (training.batches) |batch| {
        try std.testing.expectEqual(@as(usize, 64), batch.train_positions.len);
        try std.testing.expectEqual(@as(usize, 64 * 512), batch.train.len);
        try std.testing.expectEqual(@as(usize, 0), batch.heldout.len);
    }
}

test "recent64 capture rejects causal shape retry and cancellation mutations" {
    var owner = try Owner.init(std.testing.allocator, fixtureAuthority(), fixtureTopology(), try Owner.requiredBytes(), .{});
    defer owner.deinit();
    var values: [query_heads * head_dim]f32 = @splat(1);
    try std.testing.expectError(error.InvalidRecentQueryShape, owner.capture(5, 5, 8127, 7, head_dim, &values));
    try owner.capture(5, 5, 8127, query_heads, head_dim, &values);
    values[0] = 2;
    try std.testing.expectError(error.RecentQueryRetryMismatch, owner.capture(5, 5, 8127, query_heads, head_dim, &values));
    try std.testing.expectError(error.IncompleteRecentQueryCapture, owner.seal());

    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.CompactionCancelled;
        }
    };
    var cancelled = try Owner.init(std.testing.allocator, fixtureAuthority(), fixtureTopology(), try Owner.requiredBytes(), .{});
    defer cancelled.deinit();
    cancelled.control = .{ .check_fn = Cancel.check };
    try std.testing.expectError(error.CompactionCancelled, cancelled.capture(5, 5, 8127, query_heads, head_dim, &values));
    try std.testing.expectEqual(@as(usize, 0), cancelled.entries[0].count);
}

fn allocationFailure(allocator: std.mem.Allocator) !void {
    var owner = try Owner.init(allocator, fixtureAuthority(), fixtureTopology(), try Owner.requiredBytes(), .{});
    defer owner.deinit();
}

test "recent64 capture initialization is allocation failure atomic" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailure, .{});
}
