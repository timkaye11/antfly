const std = @import("std");
const engine = @import("attention_match_engine.zig");
const checkpoint = @import("attention_match_checkpoint.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;
pub const maximum_bytes: usize = 128 * 1024 * 1024;
const manifest_reservation: usize = 64 * 1024;
const chunk_bytes: usize = 1024 * 1024;
const Claim = enum(u8) { idle, writing, committed, failed };
var claim: std.atomic.Value(u8) = .init(@backingInt(Claim.idle));

pub const Binding = struct { output_dir: []const u8, identity: checkpoint.Identity, config_sha256: [32]u8, max_bytes: usize = maximum_bytes };
pub const Result = struct { committed: bool, required_bytes: usize };
pub const Writer = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    binding: Binding,
    capture_error: ?anyerror = null,
    required_bytes: usize = 0,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, binding: Binding) !Writer {
        if (binding.output_dir.len == 0 or binding.max_bytes == 0 or binding.max_bytes > maximum_bytes) return error.InvalidRejectedHeadFixtureBinding;
        return .{ .allocator = allocator, .io = io, .binding = binding };
    }
    pub fn deinit(_: *Writer) void {}
    pub fn callback(self: *Writer) engine.RejectedHeadCapture {
        return .{ .context = self, .capture_fn = capture };
    }
    pub fn finish(self: *Writer) !Result {
        if (self.capture_error) |err| return err;
        if (claim.load(.acquire) != @backingInt(Claim.committed)) return error.RejectedHeadFixtureNotCommitted;
        return .{ .committed = true, .required_bytes = self.required_bytes };
    }
    fn capture(raw: ?*anyopaque, view: engine.RejectedHeadFixtureView) void {
        const self: *Writer = @ptrCast(@alignCast(raw orelse return));
        if (!tryClaim()) return;
        self.write(view) catch |err| {
            self.capture_error = err;
            self.writeFailure(err) catch {};
            claim.store(@backingInt(Claim.failed), .release);
            return;
        };
        claim.store(@backingInt(Claim.committed), .release);
    }

    const Payload = struct { name: []const u8, bytes: usize, count: usize, sha256: [32]u8 };
    fn payload(self: *Writer, dir: std.Io.Dir, name: []const u8, bytes: []const u8, count: usize) !Payload {
        var file = try dir.createFile(self.io, name, .{ .exclusive = true, .truncate = false });
        defer file.close(self.io);
        var hash = Sha256.init(.{});
        var offset: usize = 0;
        while (offset < bytes.len) {
            const take = @min(chunk_bytes, bytes.len - offset);
            const part = bytes[offset..][0..take];
            try file.writeStreamingAll(self.io, part);
            hash.update(part);
            offset += take;
        }
        try file.sync(self.io);
        return .{ .name = name, .bytes = bytes.len, .count = count, .sha256 = hash.finalResult() };
    }

    fn write(self: *Writer, view: engine.RejectedHeadFixtureView) !void {
        self.required_bytes = try requiredBytes(view, manifest_reservation);
        if (self.required_bytes > self.binding.max_bytes) return error.RejectedHeadFixtureLimit;
        try std.Io.Dir.createDirAbsolute(self.io, self.binding.output_dir, std.Io.File.Permissions.fromMode(0o700));
        var dir = try std.Io.Dir.openDirAbsolute(self.io, self.binding.output_dir, .{ .follow_symlinks = false });
        defer dir.close(self.io);
        const ob = view.original.bias orelse return error.InvalidRejectedHeadFixture;
        const cb = view.candidate.bias orelse return error.InvalidRejectedHeadFixture;
        var p: [18]Payload = undefined;
        p[0] = try self.payload(dir, "original.keys.f64", std.mem.sliceAsBytes(view.original.keys), view.original.keys.len);
        p[1] = try self.payload(dir, "original.values.f64", std.mem.sliceAsBytes(view.original.values), view.original.values.len);
        p[2] = try self.payload(dir, "original.bias.f64", std.mem.sliceAsBytes(ob), ob.len);
        p[3] = try self.payload(dir, "original.positions.u64", std.mem.sliceAsBytes(view.original_positions), view.original_positions.len);
        p[4] = try self.payload(dir, "original.coverage.u64", std.mem.sliceAsBytes(view.original_coverage_ends), view.original_coverage_ends.len);
        p[5] = try self.payload(dir, "candidate.keys.f64", std.mem.sliceAsBytes(view.candidate.keys), view.candidate.keys.len);
        p[6] = try self.payload(dir, "candidate.values.f64", std.mem.sliceAsBytes(view.candidate.values), view.candidate.values.len);
        p[7] = try self.payload(dir, "candidate.bias.f64", std.mem.sliceAsBytes(cb), cb.len);
        p[8] = try self.payload(dir, "candidate.positions.u64", std.mem.sliceAsBytes(view.candidate_positions), view.candidate_positions.len);
        p[9] = try self.payload(dir, "candidate.coverage.u64", std.mem.sliceAsBytes(view.candidate_coverage_ends), view.candidate_coverage_ends.len);
        p[10] = try self.payload(dir, "train.f64", std.mem.sliceAsBytes(view.train_queries), view.train_queries.len);
        p[11] = try self.payload(dir, "heldout.f64", std.mem.sliceAsBytes(view.heldout_queries), view.heldout_queries.len);
        p[12] = try self.payload(dir, "train.positions.u64", std.mem.sliceAsBytes(view.train_positions), view.train_positions.len);
        p[13] = try self.payload(dir, "heldout.positions.u64", std.mem.sliceAsBytes(view.heldout_positions), view.heldout_positions.len);
        p[14] = try self.payload(dir, "query.consumer_layers.u32", std.mem.sliceAsBytes(view.query_consumer_layers), view.query_consumer_layers.len);
        p[15] = try self.payload(dir, "query.heads.u32", std.mem.sliceAsBytes(view.query_heads), view.query_heads.len);
        p[16] = try self.payload(dir, "query.train_counts.u32", std.mem.sliceAsBytes(view.query_train_counts), view.query_train_counts.len);
        p[17] = try self.payload(dir, "query.heldout_counts.u32", std.mem.sliceAsBytes(view.query_heldout_counts), view.query_heldout_counts.len);
        const id = self.binding.identity;
        const body = try std.json.Stringify.valueAlloc(self.allocator, .{
            .schema = "antfly.source61-first-rejected-head-fixture.v1",
            .diagnostic_only = true,
            .quality_credit = false,
            .encoding = "native-little-endian-ieee754",
            .required_bytes = self.required_bytes,
            .maximum_bytes = self.binding.max_bytes,
            .identity = .{ .weights = std.fmt.bytesToHex(id.weights, .lower), .tokenizer = std.fmt.bytesToHex(id.tokenizer, .lower), .template = std.fmt.bytesToHex(id.template, .lower), .layout = std.fmt.bytesToHex(id.layout, .lower) },
            .config_sha256 = std.fmt.bytesToHex(self.binding.config_sha256, .lower),
            .geometry = .{ .donor_layer = view.donor_layer, .kv_head = view.kv_head, .logical_length = view.logical_length, .head_dim = view.original.dim, .original_rows = view.original.rows, .candidate_rows = view.candidate.rows, .prefix_rows = view.prefix_rows, .recent_rows = view.recent_rows, .chunk_rows = view.chunk_rows, .retention_divisor = view.retention_divisor, .scale = view.scale, .softcap = view.softcap },
            .recorded_metrics = view.metrics,
            .payloads = p,
        }, .{});
        defer self.allocator.free(body);
        if (body.len > manifest_reservation) return error.RejectedHeadFixtureLimit;
        var file = try dir.createFile(self.io, "manifest.json.part", .{ .exclusive = true, .truncate = false });
        try file.writeStreamingAll(self.io, body);
        try file.writeStreamingAll(self.io, "\n");
        try file.sync(self.io);
        file.close(self.io);
        try std.Io.Dir.rename(dir, "manifest.json.part", dir, "manifest.json", self.io);
    }

    fn writeFailure(self: *Writer, err: anyerror) !void {
        std.Io.Dir.createDirAbsolute(self.io, self.binding.output_dir, std.Io.File.Permissions.fromMode(0o700)) catch {};
        var dir = try std.Io.Dir.openDirAbsolute(self.io, self.binding.output_dir, .{ .follow_symlinks = false });
        defer dir.close(self.io);
        const body = try std.json.Stringify.valueAlloc(self.allocator, .{ .schema = "antfly.source61-first-rejected-head-fixture-failure.v1", .diagnostic_only = true, .quality_credit = false, .error_name = @errorName(err), .required_bytes = self.required_bytes, .maximum_bytes = self.binding.max_bytes }, .{});
        defer self.allocator.free(body);
        var file = try dir.createFile(self.io, "failure.json", .{ .exclusive = true, .truncate = false });
        defer file.close(self.io);
        try file.writeStreamingAll(self.io, body);
        try file.writeStreamingAll(self.io, "\n");
        try file.sync(self.io);
    }
};

pub fn requiredBytes(view: engine.RejectedHeadFixtureView, manifest_bytes: usize) !usize {
    var total = manifest_bytes;
    inline for (.{ view.original.keys.len, view.original.values.len, (view.original.bias orelse return error.InvalidRejectedHeadFixture).len, view.candidate.keys.len, view.candidate.values.len, (view.candidate.bias orelse return error.InvalidRejectedHeadFixture).len, view.train_queries.len, view.heldout_queries.len }) |count|
        total = try std.math.add(usize, total, try std.math.mul(usize, count, @sizeOf(f64)));
    inline for (.{ view.original_positions.len, view.original_coverage_ends.len, view.candidate_positions.len, view.candidate_coverage_ends.len, view.train_positions.len, view.heldout_positions.len }) |count|
        total = try std.math.add(usize, total, try std.math.mul(usize, count, @sizeOf(u64)));
    inline for (.{ view.query_consumer_layers.len, view.query_heads.len, view.query_train_counts.len, view.query_heldout_counts.len }) |count|
        total = try std.math.add(usize, total, try std.math.mul(usize, count, @sizeOf(u32)));
    return total;
}

const PayloadManifest = struct { name: []const u8, bytes: usize, count: usize, sha256: [32]u8 };
pub const Geometry = struct { donor_layer: u32, kv_head: usize, logical_length: u64, head_dim: usize, original_rows: usize, candidate_rows: usize, prefix_rows: usize, recent_rows: usize, chunk_rows: usize, retention_divisor: usize, scale: f64, softcap: f64 };
const IdentityManifest = struct { weights: [64]u8, tokenizer: [64]u8, template: [64]u8, layout: [64]u8 };
const Manifest = struct { schema: []const u8, diagnostic_only: bool, quality_credit: bool, encoding: []const u8, required_bytes: usize, maximum_bytes: usize, identity: IdentityManifest, config_sha256: [64]u8, geometry: Geometry, payloads: []const PayloadManifest };

pub const Fixture = struct {
    allocator: std.mem.Allocator,
    manifest_bytes: []u8,
    geometry: Geometry,
    original_keys: []f64,
    original_values: []f64,
    original_bias: []f64,
    original_positions: []u64,
    original_coverage: []u64,
    candidate_keys: []f64,
    candidate_values: []f64,
    candidate_bias: []f64,
    candidate_positions: []u64,
    candidate_coverage: []u64,
    train: []f64,
    heldout: []f64,
    train_positions: []u64,
    heldout_positions: []u64,
    query_consumer_layers: []u32,
    query_heads: []u32,
    query_train_counts: []u32,
    query_heldout_counts: []u32,

    pub fn originalHead(self: *const Fixture) @import("attention_match.zig").Head {
        return .{ .keys = self.original_keys, .values = self.original_values, .bias = self.original_bias, .rows = self.geometry.original_rows, .dim = self.geometry.head_dim, .scale = self.geometry.scale, .softcap = self.geometry.softcap };
    }
    pub fn candidateHead(self: *const Fixture) @import("attention_match.zig").Head {
        return .{ .keys = self.candidate_keys, .values = self.candidate_values, .bias = self.candidate_bias, .rows = self.geometry.candidate_rows, .dim = self.geometry.head_dim, .scale = self.geometry.scale, .softcap = self.geometry.softcap };
    }
    pub const Batches = struct {
        allocator: std.mem.Allocator,
        rows: []engine.ReferenceBatch,
        train: []f32,
        heldout: []f32,

        pub fn deinit(self: *Batches) void {
            self.allocator.free(self.rows);
            self.allocator.free(self.train);
            self.allocator.free(self.heldout);
            self.* = undefined;
        }
    };

    fn exactF32(allocator: std.mem.Allocator, source: []const f64) ![]f32 {
        const output = try allocator.alloc(f32, source.len);
        errdefer allocator.free(output);
        for (output, source) |*dst, value| {
            if (!std.math.isFinite(value)) return error.InvalidRejectedHeadFixture;
            dst.* = @floatCast(value);
            if (@as(f64, dst.*) != value) return error.InvalidRejectedHeadFixture;
        }
        return output;
    }

    pub fn batches(self: *const Fixture, allocator: std.mem.Allocator) !Batches {
        const train = try exactF32(allocator, self.train);
        errdefer allocator.free(train);
        const heldout = try exactF32(allocator, self.heldout);
        errdefer allocator.free(heldout);
        const rows = try allocator.alloc(engine.ReferenceBatch, self.query_heads.len);
        errdefer allocator.free(rows);
        var train_row: usize = 0;
        var heldout_row: usize = 0;
        for (rows, 0..) |*batch, i| {
            const nt: usize = self.query_train_counts[i];
            const nh: usize = self.query_heldout_counts[i];
            batch.* = .{ .consumer_layer = self.query_consumer_layers[i], .query_head = self.query_heads[i], .train = train[train_row * self.geometry.head_dim ..][0 .. nt * self.geometry.head_dim], .heldout = heldout[heldout_row * self.geometry.head_dim ..][0 .. nh * self.geometry.head_dim], .train_positions = self.train_positions[train_row..][0..nt], .heldout_positions = self.heldout_positions[heldout_row..][0..nh] };
            train_row += nt;
            heldout_row += nh;
        }
        return .{ .allocator = allocator, .rows = rows, .train = train, .heldout = heldout };
    }

    pub fn deinit(self: *Fixture) void {
        inline for (.{ "original_keys", "original_values", "original_bias", "original_positions", "original_coverage", "candidate_keys", "candidate_values", "candidate_bias", "candidate_positions", "candidate_coverage", "train", "heldout", "train_positions", "heldout_positions", "query_consumer_layers", "query_heads", "query_train_counts", "query_heldout_counts" }) |field| self.allocator.free(@field(self, field));
        self.allocator.free(self.manifest_bytes);
        self.* = undefined;
    }
};

fn findPayload(rows: []const PayloadManifest, name: []const u8) !PayloadManifest {
    var found: ?PayloadManifest = null;
    for (rows) |row| if (std.mem.eql(u8, row.name, name)) {
        if (found != null) return error.InvalidRejectedHeadFixture;
        found = row;
    };
    return found orelse error.InvalidRejectedHeadFixture;
}

fn readTyped(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, rows: []const PayloadManifest, name: []const u8) ![]T {
    const row = try findPayload(rows, name);
    if (row.bytes != try std.math.mul(usize, row.count, @sizeOf(T))) return error.InvalidRejectedHeadFixture;
    const output = try allocator.alloc(T, row.count);
    errdefer allocator.free(output);
    var file = try dir.openFile(io, name, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    const bytes = std.mem.sliceAsBytes(output);
    const read_bytes = try file.readPositionalAll(io, bytes, 0);
    var extra: [1]u8 = undefined;
    const trailing_bytes = try file.readPositionalAll(io, &extra, bytes.len);
    try verifyPayloadLength(read_bytes, bytes.len, trailing_bytes);
    try verifyPayload(bytes, row.sha256);
    return output;
}

fn verifyPayloadLength(read_bytes: usize, expected_bytes: usize, trailing_bytes: usize) !void {
    if (read_bytes != expected_bytes or trailing_bytes != 0) return error.InvalidRejectedHeadFixture;
}

fn verifyPayload(bytes: []const u8, expected_sha256: [32]u8) !void {
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &digest, &expected_sha256)) return error.InvalidRejectedHeadFixture;
}

fn tryClaim() bool {
    return claim.cmpxchgStrong(@backingInt(Claim.idle), @backingInt(Claim.writing), .acq_rel, .acquire) == null;
}

fn readInternal(allocator: std.mem.Allocator, io: std.Io, path: []const u8, expected: Binding, defer_heldout: bool) !Fixture {
    if (!std.mem.eql(u8, path, expected.output_dir)) return error.InvalidRejectedHeadFixtureBinding;
    var dir = try std.Io.Dir.openDirAbsolute(io, path, .{ .follow_symlinks = false });
    defer dir.close(io);
    const manifest_bytes = try dir.readFileAlloc(io, "manifest.json", allocator, .limited(manifest_reservation));
    errdefer allocator.free(manifest_bytes);
    const parsed = try std.json.parseFromSlice(Manifest, allocator, manifest_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const m = parsed.value;
    if (!std.mem.eql(u8, m.schema, "antfly.source61-first-rejected-head-fixture.v1") or
        !std.mem.eql(u8, m.encoding, "native-little-endian-ieee754") or
        !m.diagnostic_only or m.quality_credit or m.required_bytes > expected.max_bytes or m.maximum_bytes != expected.max_bytes or m.payloads.len != 18) return error.InvalidRejectedHeadFixture;
    const expected_identity = IdentityManifest{ .weights = std.fmt.bytesToHex(expected.identity.weights, .lower), .tokenizer = std.fmt.bytesToHex(expected.identity.tokenizer, .lower), .template = std.fmt.bytesToHex(expected.identity.template, .lower), .layout = std.fmt.bytesToHex(expected.identity.layout, .lower) };
    const expected_config = std.fmt.bytesToHex(expected.config_sha256, .lower);
    if (!std.meta.eql(m.identity, expected_identity) or !std.mem.eql(u8, &m.config_sha256, &expected_config)) return error.InvalidRejectedHeadFixture;
    var declared: usize = manifest_reservation;
    for (m.payloads) |row| declared = std.math.add(usize, declared, row.bytes) catch return error.InvalidRejectedHeadFixture;
    if (declared != m.required_bytes or declared > expected.max_bytes) return error.InvalidRejectedHeadFixture;
    const names = [_][]const u8{ "original.keys.f64", "original.values.f64", "original.bias.f64", "original.positions.u64", "original.coverage.u64", "candidate.keys.f64", "candidate.values.f64", "candidate.bias.f64", "candidate.positions.u64", "candidate.coverage.u64", "train.f64", "heldout.f64", "train.positions.u64", "heldout.positions.u64", "query.consumer_layers.u32", "query.heads.u32", "query.train_counts.u32", "query.heldout_counts.u32" };
    for (m.payloads) |row| {
        var known = false;
        for (names) |name| known = known or std.mem.eql(u8, row.name, name);
        if (!known) return error.InvalidRejectedHeadFixture;
    }
    for (names) |name| _ = try findPayload(m.payloads, name);
    if (m.geometry.retention_divisor != 2 or m.geometry.head_dim == 0 or
        (try findPayload(m.payloads, "original.keys.f64")).count != try std.math.mul(usize, m.geometry.original_rows, m.geometry.head_dim) or
        (try findPayload(m.payloads, "original.values.f64")).count != try std.math.mul(usize, m.geometry.original_rows, m.geometry.head_dim) or
        (try findPayload(m.payloads, "original.bias.f64")).count != m.geometry.original_rows or
        (try findPayload(m.payloads, "original.positions.u64")).count != m.geometry.original_rows or
        (try findPayload(m.payloads, "original.coverage.u64")).count != m.geometry.original_rows or
        (try findPayload(m.payloads, "candidate.keys.f64")).count != try std.math.mul(usize, m.geometry.candidate_rows, m.geometry.head_dim) or
        (try findPayload(m.payloads, "candidate.values.f64")).count != try std.math.mul(usize, m.geometry.candidate_rows, m.geometry.head_dim) or
        (try findPayload(m.payloads, "candidate.bias.f64")).count != m.geometry.candidate_rows or
        (try findPayload(m.payloads, "candidate.positions.u64")).count != m.geometry.candidate_rows or
        (try findPayload(m.payloads, "candidate.coverage.u64")).count != m.geometry.candidate_rows)
        return error.InvalidRejectedHeadFixture;
    const descriptor_count = (try findPayload(m.payloads, "query.consumer_layers.u32")).count;
    if ((try findPayload(m.payloads, "query.heads.u32")).count != descriptor_count or (try findPayload(m.payloads, "query.train_counts.u32")).count != descriptor_count or (try findPayload(m.payloads, "query.heldout_counts.u32")).count != descriptor_count) return error.InvalidRejectedHeadFixture;
    var result: Fixture = undefined;
    result.allocator = allocator;
    result.manifest_bytes = manifest_bytes;
    result.geometry = m.geometry;
    result.original_keys = try readTyped(f64, allocator, io, dir, m.payloads, "original.keys.f64");
    errdefer allocator.free(result.original_keys);
    result.original_values = try readTyped(f64, allocator, io, dir, m.payloads, "original.values.f64");
    errdefer allocator.free(result.original_values);
    result.original_bias = try readTyped(f64, allocator, io, dir, m.payloads, "original.bias.f64");
    errdefer allocator.free(result.original_bias);
    result.original_positions = try readTyped(u64, allocator, io, dir, m.payloads, "original.positions.u64");
    errdefer allocator.free(result.original_positions);
    result.original_coverage = try readTyped(u64, allocator, io, dir, m.payloads, "original.coverage.u64");
    errdefer allocator.free(result.original_coverage);
    result.candidate_keys = if (defer_heldout) try allocator.alloc(f64, 0) else try readTyped(f64, allocator, io, dir, m.payloads, "candidate.keys.f64");
    errdefer allocator.free(result.candidate_keys);
    result.candidate_values = if (defer_heldout) try allocator.alloc(f64, 0) else try readTyped(f64, allocator, io, dir, m.payloads, "candidate.values.f64");
    errdefer allocator.free(result.candidate_values);
    result.candidate_bias = if (defer_heldout) try allocator.alloc(f64, 0) else try readTyped(f64, allocator, io, dir, m.payloads, "candidate.bias.f64");
    errdefer allocator.free(result.candidate_bias);
    result.candidate_positions = if (defer_heldout) try allocator.alloc(u64, 0) else try readTyped(u64, allocator, io, dir, m.payloads, "candidate.positions.u64");
    errdefer allocator.free(result.candidate_positions);
    result.candidate_coverage = if (defer_heldout) try allocator.alloc(u64, 0) else try readTyped(u64, allocator, io, dir, m.payloads, "candidate.coverage.u64");
    errdefer allocator.free(result.candidate_coverage);
    result.train = try readTyped(f64, allocator, io, dir, m.payloads, "train.f64");
    errdefer allocator.free(result.train);
    result.heldout = if (defer_heldout) try allocator.alloc(f64, 0) else try readTyped(f64, allocator, io, dir, m.payloads, "heldout.f64");
    errdefer allocator.free(result.heldout);
    result.train_positions = try readTyped(u64, allocator, io, dir, m.payloads, "train.positions.u64");
    errdefer allocator.free(result.train_positions);
    result.heldout_positions = if (defer_heldout) try allocator.alloc(u64, 0) else try readTyped(u64, allocator, io, dir, m.payloads, "heldout.positions.u64");
    errdefer allocator.free(result.heldout_positions);
    result.query_consumer_layers = try readTyped(u32, allocator, io, dir, m.payloads, "query.consumer_layers.u32");
    errdefer allocator.free(result.query_consumer_layers);
    result.query_heads = try readTyped(u32, allocator, io, dir, m.payloads, "query.heads.u32");
    errdefer allocator.free(result.query_heads);
    result.query_train_counts = try readTyped(u32, allocator, io, dir, m.payloads, "query.train_counts.u32");
    errdefer allocator.free(result.query_train_counts);
    result.query_heldout_counts = try readTyped(u32, allocator, io, dir, m.payloads, "query.heldout_counts.u32");
    errdefer allocator.free(result.query_heldout_counts);
    var train_rows: usize = 0;
    for (result.query_train_counts) |count| train_rows = try std.math.add(usize, train_rows, count);
    var heldout_rows: usize = 0;
    for (result.query_heldout_counts) |count| heldout_rows = try std.math.add(usize, heldout_rows, count);
    if (result.train_positions.len != train_rows or result.train.len != try std.math.mul(usize, train_rows, m.geometry.head_dim) or
        (!defer_heldout and (result.heldout_positions.len != heldout_rows or result.heldout.len != try std.math.mul(usize, heldout_rows, m.geometry.head_dim))) or
        (defer_heldout and (result.heldout_positions.len != 0 or result.heldout.len != 0))) return error.InvalidRejectedHeadFixture;
    return result;
}

pub fn read(allocator: std.mem.Allocator, io: std.Io, path: []const u8, expected: Binding) !Fixture {
    return readInternal(allocator, io, path, expected, false);
}

/// Diagnostic two-phase reader. It authenticates the manifest and all source
/// and TRAIN payloads but deliberately does not open placeholder-candidate or
/// heldout files.
/// Callers must later call the ordinary `read` and compare manifest bytes
/// before using heldout solely for validation.
pub fn readTrainingOnly(allocator: std.mem.Allocator, io: std.Io, path: []const u8, expected: Binding) !Fixture {
    return readInternal(allocator, io, path, expected, true);
}

pub fn resetForTest() void {
    if (@import("builtin").is_test) claim.store(@backingInt(Claim.idle), .release);
}

test "rejected fixture payload hash rejects mutation" {
    const bytes = "bounded rejected fixture";
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    try verifyPayload(bytes, digest);
    digest[0] ^= 1;
    try std.testing.expectError(error.InvalidRejectedHeadFixture, verifyPayload(bytes, digest));
}

test "rejected fixture payload length rejects truncation and trailing bytes" {
    try verifyPayloadLength(8, 8, 0);
    try std.testing.expectError(error.InvalidRejectedHeadFixture, verifyPayloadLength(7, 8, 0));
    try std.testing.expectError(error.InvalidRejectedHeadFixture, verifyPayloadLength(8, 8, 1));
}

test "rejected fixture manifest lookup rejects duplicate and missing payloads" {
    const digest: [32]u8 = @splat(0);
    const duplicate = [_]PayloadManifest{
        .{ .name = "x", .bytes = 8, .count = 1, .sha256 = digest },
        .{ .name = "x", .bytes = 8, .count = 1, .sha256 = digest },
    };
    try std.testing.expectError(error.InvalidRejectedHeadFixture, findPayload(&duplicate, "x"));
    try std.testing.expectError(error.InvalidRejectedHeadFixture, findPayload(&duplicate, "missing"));
}

test "rejected fixture byte accounting overflows and enforces the bounded cap" {
    const near_cap = maximum_bytes - manifest_reservation;
    try std.testing.expect(try std.math.add(usize, manifest_reservation, near_cap) == maximum_bytes);
    try std.testing.expect((try std.math.add(usize, maximum_bytes, 1)) > maximum_bytes);
    try std.testing.expectError(error.Overflow, std.math.add(usize, std.math.maxInt(usize), 1));
}

test "rejected fixture first claim is process shared and no overwrite is possible" {
    resetForTest();
    defer resetForTest();
    const Worker = struct {
        fn run(winners: *std.atomic.Value(u32)) void {
            if (tryClaim()) _ = winners.fetchAdd(1, .acq_rel);
        }
    };
    var winners: std.atomic.Value(u32) = .init(0);
    var threads: [8]std.Thread = undefined;
    for (&threads) |*thread| thread.* = try std.Thread.spawn(.{}, Worker.run, .{&winners});
    for (&threads) |*thread| thread.join();
    try std.testing.expectEqual(@as(u32, 1), winners.load(.acquire));
    try std.testing.expect(!tryClaim());
}

test "rejected fixture writer reader and owned batches round trip" {
    resetForTest();
    defer resetForTest();
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const base_rel = try std.fs.path.join(allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
    defer allocator.free(base_rel);
    const base_sentinel = try std.Io.Dir.cwd().realPathFileAlloc(io, base_rel, allocator);
    defer allocator.free(base_sentinel);
    const output_dir = try std.fs.path.join(allocator, &.{ base_sentinel, "fixture" });
    defer allocator.free(output_dir);
    const zero: [32]u8 = @splat(0);
    const binding: Binding = .{ .output_dir = output_dir, .identity = .{ .weights = zero, .tokenizer = zero, .template = zero, .layout = zero }, .config_sha256 = zero, .max_bytes = 1024 * 1024 };
    var writer = try Writer.init(allocator, io, binding);
    defer writer.deinit();
    const original_keys = [_]f64{ 0.25, -0.5, 0.75, 1.0 };
    const original_values = [_]f64{ -1.0, 0.5, 0.25, 0.125 };
    const original_bias = [_]f64{ 0.0, -0.25 };
    const candidate_keys = [_]f64{ 0.25, -0.5 };
    const candidate_values = [_]f64{ -1.0, 0.5 };
    const candidate_bias = [_]f64{0.0};
    const original_positions = [_]u64{ 3, 7 };
    const original_coverage = [_]u64{ 4, 8 };
    const candidate_positions = [_]u64{3};
    const candidate_coverage = [_]u64{8};
    const train = [_]f64{ 0.25, -0.5, 0.75, 1.0 };
    const heldout = [_]f64{ -1.0, 0.125 };
    const train_positions = [_]u64{ 11, 12 };
    const heldout_positions = [_]u64{13};
    const consumer_layers = [_]u32{5};
    const query_heads = [_]u32{2};
    const train_counts = [_]u32{2};
    const heldout_counts = [_]u32{1};
    const callback = writer.callback();
    callback.capture_fn(callback.context, .{
        .donor_layer = 4,
        .kv_head = 1,
        .retention_divisor = 2,
        .logical_length = 8,
        .prefix_rows = 1,
        .recent_rows = 1,
        .chunk_rows = 2,
        .scale = 0.5,
        .softcap = 0,
        .original = .{ .keys = &original_keys, .values = &original_values, .bias = &original_bias, .rows = 2, .dim = 2, .scale = 0.5 },
        .candidate = .{ .keys = &candidate_keys, .values = &candidate_values, .bias = &candidate_bias, .rows = 1, .dim = 2, .scale = 0.5 },
        .original_positions = &original_positions,
        .original_coverage_ends = &original_coverage,
        .candidate_positions = &candidate_positions,
        .candidate_coverage_ends = &candidate_coverage,
        .train_queries = &train,
        .heldout_queries = &heldout,
        .train_positions = &train_positions,
        .heldout_positions = &heldout_positions,
        .query_consumer_layers = &consumer_layers,
        .query_heads = &query_heads,
        .query_train_counts = &train_counts,
        .query_heldout_counts = &heldout_counts,
        .metrics = .{},
    });
    try std.testing.expect((try writer.finish()).committed);
    var fixture_dir = try std.Io.Dir.openDirAbsolute(io, output_dir, .{ .follow_symlinks = false });
    defer fixture_dir.close(io);
    const deferred_names = [_][]const u8{ "candidate.keys.f64", "candidate.values.f64", "candidate.bias.f64", "candidate.positions.u64", "candidate.coverage.u64", "heldout.f64", "heldout.positions.u64" };
    for (deferred_names) |name| {
        const hidden = try std.fmt.allocPrint(allocator, "{s}.absent", .{name});
        defer allocator.free(hidden);
        try std.Io.Dir.rename(fixture_dir, name, fixture_dir, hidden, io);
    }
    var training_only = try readTrainingOnly(allocator, io, output_dir, binding);
    try std.testing.expectEqual(@as(usize, 0), training_only.heldout.len);
    try std.testing.expectEqual(@as(usize, 0), training_only.candidate_keys.len);
    training_only.deinit();
    try std.testing.expectError(error.FileNotFound, read(allocator, io, output_dir, binding));
    for (deferred_names) |name| {
        const hidden = try std.fmt.allocPrint(allocator, "{s}.absent", .{name});
        defer allocator.free(hidden);
        try std.Io.Dir.rename(fixture_dir, hidden, fixture_dir, name, io);
    }
    var fixture = try read(allocator, io, output_dir, binding);
    defer fixture.deinit();
    var batches = try fixture.batches(allocator);
    defer batches.deinit();
    try std.testing.expectEqual(@as(usize, 1), batches.rows.len);
    try std.testing.expectEqualSlices(f32, &.{ 0.25, -0.5, 0.75, 1.0 }, batches.rows[0].train);
    try std.testing.expectEqualSlices(f32, &.{ -1.0, 0.125 }, batches.rows[0].heldout);
    try std.testing.expectEqualSlices(u64, &train_positions, batches.rows[0].train_positions);
    try std.testing.expectEqualSlices(u64, &heldout_positions, batches.rows[0].heldout_positions);
    try std.testing.expectEqual(@as(u32, 5), batches.rows[0].consumer_layer);
    try std.testing.expectEqual(@as(u32, 2), batches.rows[0].query_head);
}
