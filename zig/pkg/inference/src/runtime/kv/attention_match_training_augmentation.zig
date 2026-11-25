// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

//! Authenticated, diagnostic-only second-study training queries.  This bundle
//! deliberately has no held-out domain; authorization remains with its parent.
const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const queries = @import("attention_match_queries.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const maximum_bytes: usize = 128 * 1024 * 1024;
pub const legacy_samples_per_stratum: u16 = 320;
pub const portfolio_samples_per_stratum: u16 = 32;
pub const extended_portfolio_samples_per_stratum: u16 = 128;
pub const independent_domain_samples_per_stratum: u16 = 256;
pub const dense_reference_samples_per_stratum: u16 = 512;
const manifest_reservation: usize = 64 * 1024;
const schema = "antfly.attention-match-training-augmentation.v1";

pub const Binding = struct {
    output_dir: []const u8,
    parent_manifest_sha256: [32]u8,
    identity: checkpoint.Identity,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
    parent_study_tokens_sha256: [32]u8,
    study_tokens_sha256: [32]u8,
    samples_per_stratum: u16 = legacy_samples_per_stratum,
    max_bytes: usize = maximum_bytes,
};

pub const View = struct {
    batches: []const engine.ReferenceBatch,
    consumers: []const queries.Consumer,
    study_pass: u8 = 1,
    samples_per_stratum: u16 = legacy_samples_per_stratum,
    parent_study_tokens_sha256: [32]u8 = @splat(0),
    study_tokens_sha256: [32]u8 = @splat(0),
    source_length: usize,
    repeat_length: usize,
};

pub const Result = struct { required_bytes: usize, manifest_sha256: [32]u8 };
const Payload = struct { name: []const u8, bytes: usize, count: usize, sha256: [32]u8 };
const IdentityManifest = struct { weights: [64]u8, tokenizer: [64]u8, template: [64]u8, layout: [64]u8 };
const Manifest = struct {
    schema: []const u8,
    diagnostic_only: bool,
    quality_credit: bool,
    training_only: bool,
    heldout_present: bool,
    encoding: []const u8,
    required_bytes: usize,
    maximum_bytes: usize,
    parent_manifest_sha256: [64]u8,
    identity: IdentityManifest,
    source_freeze_sha256: [64]u8,
    binary_sha256: [64]u8,
    exporter_sha256: [64]u8,
    parent_study_tokens_sha256: [64]u8,
    study_tokens_sha256: [64]u8,
    study_pass: u8,
    samples_per_stratum: u16,
    source_length: usize,
    repeat_length: usize,
    batch_count: usize,
    dimensions: []const u32,
    payloads: []const Payload,
};

const payload_names = [_][]const u8{
    "queries.training.f32",
    "queries.training_positions.u64",
    "descriptors.consumer_layers.u32",
    "descriptors.query_heads.u32",
    "descriptors.training_counts.u32",
};

pub fn allowedSamples(samples: u16) bool {
    return samples == legacy_samples_per_stratum or samples == portfolio_samples_per_stratum or samples == extended_portfolio_samples_per_stratum or samples == independent_domain_samples_per_stratum;
}

fn checkedAdd(total: usize, count: usize, size: usize) !usize {
    return std.math.add(usize, total, try std.math.mul(usize, count, size));
}

const Plan = struct { required: usize, elements: usize, positions: usize };
fn planWithPolicy(view: View, cap: usize, allow_dense_reference: bool) !Plan {
    const samples: usize = view.samples_per_stratum;
    if (view.study_pass != 1 or (!allowedSamples(view.samples_per_stratum) and !(allow_dense_reference and view.samples_per_stratum == dense_reference_samples_per_stratum)) or view.source_length == 0 or view.repeat_length == 0 or view.batches.len == 0 or view.batches.len > std.math.maxInt(u32)) return error.InvalidTrainingAugmentation;
    var total: usize = manifest_reservation;
    var elements: usize = 0;
    var positions: usize = 0;
    for (view.batches) |batch| {
        if (batch.heldout.len != 0 or batch.heldout_positions.len != 0 or batch.train_positions.len != samples or batch.train.len == 0 or batch.train.len % samples != 0) return error.InvalidTrainingAugmentation;
        const dim = batch.train.len / samples;
        if (dim == 0 or dim > std.math.maxInt(u32)) return error.InvalidTrainingAugmentation;
        for (batch.train) |value| if (!std.math.isFinite(value)) return error.InvalidTrainingAugmentation;
        for (batch.train_positions[1..], batch.train_positions[0 .. batch.train_positions.len - 1]) |next, prior|
            if (next <= prior) return error.InvalidTrainingAugmentation;
        elements = try std.math.add(usize, elements, batch.train.len);
        positions = try std.math.add(usize, positions, batch.train_positions.len);
    }
    total = try checkedAdd(total, elements, @sizeOf(f32));
    total = try checkedAdd(total, positions, @sizeOf(u64));
    inline for (0..3) |_| total = try checkedAdd(total, view.batches.len, @sizeOf(u32));
    if (total > cap) return error.TrainingAugmentationLimit;
    return .{ .required = total, .elements = elements, .positions = positions };
}

fn plan(view: View, cap: usize) !Plan {
    return planWithPolicy(view, cap, false);
}

fn writeHashed(io: std.Io, file: *std.Io.File, hash: *Sha256, bytes: []const u8) !void {
    const chunk = 1024 * 1024;
    var offset: usize = 0;
    while (offset < bytes.len) {
        const part = bytes[offset..][0..@min(chunk, bytes.len - offset)];
        try file.writeStreamingAll(io, part);
        hash.update(part);
        offset += part.len;
    }
}

fn writeFields(io: std.Io, dir: std.Io.Dir, name: []const u8, batches: []const engine.ReferenceBatch, comptime field: enum { train, positions }) !Payload {
    var file = try dir.createFile(io, name, .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    var hash = Sha256.init(.{});
    var count: usize = 0;
    for (batches) |batch| {
        const values = if (field == .train) std.mem.sliceAsBytes(batch.train) else std.mem.sliceAsBytes(batch.train_positions);
        count = try std.math.add(usize, count, if (field == .train) batch.train.len else batch.train_positions.len);
        try writeHashed(io, &file, &hash, values);
    }
    try file.sync(io);
    return .{ .name = name, .bytes = if (field == .train) try std.math.mul(usize, count, @sizeOf(f32)) else try std.math.mul(usize, count, @sizeOf(u64)), .count = count, .sha256 = hash.finalResult() };
}

fn writeSlice(io: std.Io, dir: std.Io.Dir, name: []const u8, values: anytype) !Payload {
    var file = try dir.createFile(io, name, .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    const bytes = std.mem.sliceAsBytes(values);
    var hash = Sha256.init(.{});
    try writeHashed(io, &file, &hash, bytes);
    try file.sync(io);
    return .{ .name = name, .bytes = bytes.len, .count = values.len, .sha256 = hash.finalResult() };
}

pub const OwnedTraining = struct {
    allocator: std.mem.Allocator,
    batches: []engine.ReferenceBatch,
    values: []f32,
    positions: []u64,

    pub fn deinit(self: *OwnedTraining) void {
        self.allocator.free(self.batches);
        self.allocator.free(self.values);
        self.allocator.free(self.positions);
        self.* = undefined;
    }
};

/// Converts the collector's deterministic train/heldout split into a
/// training-only second-study domain.  The original chronological order is
/// reconstructed from positions; duplicate positions are rejected.
fn mergeCollectorBatchesWithPolicy(allocator: std.mem.Allocator, source: []const engine.ReferenceBatch, samples_per_stratum: u16, allow_dense_reference: bool) !OwnedTraining {
    const samples: usize = samples_per_stratum;
    if (!allowedSamples(samples_per_stratum) and !(allow_dense_reference and samples_per_stratum == dense_reference_samples_per_stratum)) return error.InvalidTrainingAugmentation;
    if (source.len == 0) return error.InvalidTrainingAugmentation;
    var total_rows: usize = 0;
    var total_values: usize = 0;
    for (source) |batch| {
        if (batch.train_positions.len == 0 or batch.heldout_positions.len == 0 or
            batch.train.len % batch.train_positions.len != 0 or batch.heldout.len % batch.heldout_positions.len != 0)
            return error.InvalidTrainingAugmentation;
        const dim = batch.train.len / batch.train_positions.len;
        if (dim == 0 or batch.heldout.len / batch.heldout_positions.len != dim or
            batch.train_positions.len + batch.heldout_positions.len != samples)
            return error.InvalidTrainingAugmentation;
        total_rows = try std.math.add(usize, total_rows, samples);
        total_values = try std.math.add(usize, total_values, try std.math.mul(usize, samples, dim));
    }
    const values = try allocator.alloc(f32, total_values);
    errdefer allocator.free(values);
    const positions = try allocator.alloc(u64, total_rows);
    errdefer allocator.free(positions);
    const batches = try allocator.alloc(engine.ReferenceBatch, source.len);
    errdefer allocator.free(batches);
    var value_at: usize = 0;
    var position_at: usize = 0;
    for (source, batches) |batch, *output| {
        const dim = batch.train.len / batch.train_positions.len;
        const value_begin = value_at;
        const position_begin = position_at;
        var train_at: usize = 0;
        var heldout_at: usize = 0;
        while (train_at < batch.train_positions.len or heldout_at < batch.heldout_positions.len) {
            const take_train = if (heldout_at == batch.heldout_positions.len)
                true
            else if (train_at == batch.train_positions.len)
                false
            else
                batch.train_positions[train_at] < batch.heldout_positions[heldout_at];
            if (train_at < batch.train_positions.len and heldout_at < batch.heldout_positions.len and
                batch.train_positions[train_at] == batch.heldout_positions[heldout_at])
                return error.InvalidTrainingAugmentation;
            const row = if (take_train) train_at else heldout_at;
            const source_values = if (take_train) batch.train else batch.heldout;
            positions[position_at] = if (take_train) batch.train_positions[train_at] else batch.heldout_positions[heldout_at];
            if (position_at > position_begin and positions[position_at] <= positions[position_at - 1]) return error.InvalidTrainingAugmentation;
            @memcpy(values[value_at..][0..dim], source_values[row * dim ..][0..dim]);
            position_at += 1;
            value_at += dim;
            if (take_train) train_at += 1 else heldout_at += 1;
        }
        output.* = .{
            .consumer_layer = batch.consumer_layer,
            .query_head = batch.query_head,
            .train = values[value_begin..value_at],
            .heldout = &.{},
            .train_positions = positions[position_begin..position_at],
            .heldout_positions = &.{},
        };
    }
    return .{ .allocator = allocator, .batches = batches, .values = values, .positions = positions };
}

pub fn mergeCollectorBatchesExact(allocator: std.mem.Allocator, source: []const engine.ReferenceBatch, samples_per_stratum: u16) !OwnedTraining {
    return mergeCollectorBatchesWithPolicy(allocator, source, samples_per_stratum, false);
}

pub fn mergeReferenceCoverageDense512(allocator: std.mem.Allocator, source: []const engine.ReferenceBatch) !OwnedTraining {
    return mergeCollectorBatchesWithPolicy(allocator, source, dense_reference_samples_per_stratum, true);
}

pub fn mergeCollectorBatches(allocator: std.mem.Allocator, source: []const engine.ReferenceBatch) !OwnedTraining {
    return mergeCollectorBatchesExact(allocator, source, legacy_samples_per_stratum);
}

fn writeWithPolicy(allocator: std.mem.Allocator, io: std.Io, binding: Binding, view: View, allow_dense_reference: bool) !Result {
    if (binding.output_dir.len == 0 or !std.fs.path.isAbsolute(binding.output_dir) or binding.max_bytes == 0 or binding.max_bytes > maximum_bytes) return error.InvalidTrainingAugmentationBinding;
    if (view.samples_per_stratum != binding.samples_per_stratum or !std.mem.eql(u8, &view.parent_study_tokens_sha256, &binding.parent_study_tokens_sha256) or !std.mem.eql(u8, &view.study_tokens_sha256, &binding.study_tokens_sha256) or std.mem.eql(u8, &view.parent_study_tokens_sha256, &view.study_tokens_sha256)) return error.InvalidTrainingAugmentationBinding;
    if (!allowedSamples(binding.samples_per_stratum) and !(allow_dense_reference and binding.samples_per_stratum == dense_reference_samples_per_stratum)) return error.InvalidTrainingAugmentationBinding;
    const p = try planWithPolicy(view, binding.max_bytes, allow_dense_reference);
    const n = view.batches.len;
    const layers = try allocator.alloc(u32, n);
    defer allocator.free(layers);
    const heads = try allocator.alloc(u32, n);
    defer allocator.free(heads);
    const counts = try allocator.alloc(u32, n);
    defer allocator.free(counts);
    const dims = try allocator.alloc(u32, n);
    defer allocator.free(dims);
    for (view.batches, 0..) |batch, i| {
        layers[i] = batch.consumer_layer;
        heads[i] = batch.query_head;
        counts[i] = view.samples_per_stratum;
        dims[i] = @intCast(batch.train.len / @as(usize, view.samples_per_stratum));
        for (view.batches[0..i]) |prior| if (prior.consumer_layer == batch.consumer_layer and prior.query_head == batch.query_head) return error.InvalidTrainingAugmentation;
        var matched = false;
        for (view.consumers) |consumer| if (consumer.layer == batch.consumer_layer and batch.query_head < consumer.query_heads and dims[i] == consumer.dim) {
            matched = true;
            break;
        };
        if (!matched) return error.InvalidTrainingAugmentation;
    }
    try std.Io.Dir.createDirAbsolute(io, binding.output_dir, std.Io.File.Permissions.fromMode(0o700));
    var dir = try std.Io.Dir.openDirAbsolute(io, binding.output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    var payloads: [5]Payload = undefined;
    payloads[0] = try writeFields(io, dir, payload_names[0], view.batches, .train);
    payloads[1] = try writeFields(io, dir, payload_names[1], view.batches, .positions);
    payloads[2] = try writeSlice(io, dir, payload_names[2], layers);
    payloads[3] = try writeSlice(io, dir, payload_names[3], heads);
    payloads[4] = try writeSlice(io, dir, payload_names[4], counts);
    if (payloads[0].count != p.elements or payloads[1].count != p.positions) return error.InvalidTrainingAugmentation;
    const id = binding.identity;
    const body = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = schema,
        .diagnostic_only = true,
        .quality_credit = false,
        .training_only = true,
        .heldout_present = false,
        .encoding = "native-little-endian-ieee754",
        .required_bytes = p.required,
        .maximum_bytes = binding.max_bytes,
        .parent_manifest_sha256 = std.fmt.bytesToHex(binding.parent_manifest_sha256, .lower),
        .identity = .{ .weights = std.fmt.bytesToHex(id.weights, .lower), .tokenizer = std.fmt.bytesToHex(id.tokenizer, .lower), .template = std.fmt.bytesToHex(id.template, .lower), .layout = std.fmt.bytesToHex(id.layout, .lower) },
        .source_freeze_sha256 = std.fmt.bytesToHex(binding.source_freeze_sha256, .lower),
        .binary_sha256 = std.fmt.bytesToHex(binding.binary_sha256, .lower),
        .exporter_sha256 = std.fmt.bytesToHex(binding.exporter_sha256, .lower),
        .parent_study_tokens_sha256 = std.fmt.bytesToHex(view.parent_study_tokens_sha256, .lower),
        .study_tokens_sha256 = std.fmt.bytesToHex(view.study_tokens_sha256, .lower),
        .study_pass = view.study_pass,
        .samples_per_stratum = view.samples_per_stratum,
        .source_length = view.source_length,
        .repeat_length = view.repeat_length,
        .batch_count = n,
        .dimensions = dims,
        .payloads = payloads,
    }, .{});
    defer allocator.free(body);
    if (body.len + 1 > manifest_reservation) return error.TrainingAugmentationLimit;
    var manifest_hash = Sha256.init(.{});
    manifest_hash.update(body);
    manifest_hash.update("\n");
    {
        var file = try dir.createFile(io, "manifest.json.part", .{ .exclusive = true, .truncate = false });
        defer file.close(io);
        try file.writeStreamingAll(io, body);
        try file.writeStreamingAll(io, "\n");
        try file.sync(io);
    }
    try std.Io.Dir.rename(dir, "manifest.json.part", dir, "manifest.json", io);
    return .{ .required_bytes = p.required, .manifest_sha256 = manifest_hash.finalResult() };
}

pub fn write(allocator: std.mem.Allocator, io: std.Io, binding: Binding, view: View) !Result {
    return writeWithPolicy(allocator, io, binding, view, false);
}

pub fn writeReferenceCoverageDense512(allocator: std.mem.Allocator, io: std.Io, binding: Binding, view: View) !Result {
    if (binding.samples_per_stratum != dense_reference_samples_per_stratum or view.samples_per_stratum != dense_reference_samples_per_stratum)
        return error.InvalidTrainingAugmentationBinding;
    return writeWithPolicy(allocator, io, binding, view, true);
}

pub const Owned = struct {
    allocator: std.mem.Allocator,
    manifest_bytes: []u8,
    batches: []engine.ReferenceBatch,
    training: []f32,
    positions: []u64,
    consumer_layers: []u32,
    query_heads: []u32,
    counts: []u32,
    dimensions: []u32,
    source_length: usize,
    repeat_length: usize,
    pub fn deinit(self: *Owned) void {
        self.allocator.free(self.batches);
        self.allocator.free(self.training);
        self.allocator.free(self.positions);
        self.allocator.free(self.consumer_layers);
        self.allocator.free(self.query_heads);
        self.allocator.free(self.counts);
        self.allocator.free(self.dimensions);
        self.allocator.free(self.manifest_bytes);
        self.* = undefined;
    }
};

fn identityManifest(identity: checkpoint.Identity) IdentityManifest {
    return .{
        .weights = std.fmt.bytesToHex(identity.weights, .lower),
        .tokenizer = std.fmt.bytesToHex(identity.tokenizer, .lower),
        .template = std.fmt.bytesToHex(identity.template, .lower),
        .layout = std.fmt.bytesToHex(identity.layout, .lower),
    };
}
fn findPayload(rows: []const Payload, name: []const u8) !Payload {
    var out: ?Payload = null;
    for (rows) |row| if (std.mem.eql(u8, row.name, name)) {
        if (out != null) return error.InvalidTrainingAugmentation;
        out = row;
    };
    return out orelse error.InvalidTrainingAugmentation;
}
fn readTyped(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, rows: []const Payload, name: []const u8) ![]T {
    const row = try findPayload(rows, name);
    if (row.bytes != try std.math.mul(usize, row.count, @sizeOf(T))) return error.InvalidTrainingAugmentation;
    const out = try allocator.alloc(T, row.count);
    errdefer allocator.free(out);
    var file = try dir.openFile(io, name, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    const bytes = std.mem.sliceAsBytes(out);
    if (try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.InvalidTrainingAugmentation;
    var extra: [1]u8 = undefined;
    if (try file.readPositionalAll(io, &extra, bytes.len) != 0) return error.InvalidTrainingAugmentation;
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &digest, &row.sha256)) return error.InvalidTrainingAugmentation;
    return out;
}

fn readWithPolicy(allocator: std.mem.Allocator, io: std.Io, binding: Binding, allow_dense_reference: bool) !Owned {
    if (binding.output_dir.len == 0 or !std.fs.path.isAbsolute(binding.output_dir) or binding.max_bytes == 0 or binding.max_bytes > maximum_bytes) return error.InvalidTrainingAugmentationBinding;
    var dir = try std.Io.Dir.openDirAbsolute(io, binding.output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    const manifest_bytes = try dir.readFileAlloc(io, "manifest.json", allocator, .limited(manifest_reservation));
    errdefer allocator.free(manifest_bytes);
    const parsed = try std.json.parseFromSlice(Manifest, allocator, manifest_bytes, .{ .ignore_unknown_fields = false });
    defer parsed.deinit();
    const m = parsed.value;
    if (!std.mem.eql(u8, m.schema, schema) or !m.diagnostic_only or m.quality_credit or !m.training_only or m.heldout_present or !std.mem.eql(u8, m.encoding, "native-little-endian-ieee754") or m.study_pass != 1 or m.samples_per_stratum != binding.samples_per_stratum or (!allowedSamples(m.samples_per_stratum) and !(allow_dense_reference and m.samples_per_stratum == dense_reference_samples_per_stratum)) or m.maximum_bytes != binding.max_bytes or m.required_bytes > binding.max_bytes or m.batch_count == 0 or m.batch_count != m.dimensions.len or m.payloads.len != payload_names.len) return error.InvalidTrainingAugmentation;
    const parent = std.fmt.bytesToHex(binding.parent_manifest_sha256, .lower);
    const freeze = std.fmt.bytesToHex(binding.source_freeze_sha256, .lower);
    const binary = std.fmt.bytesToHex(binding.binary_sha256, .lower);
    const exporter = std.fmt.bytesToHex(binding.exporter_sha256, .lower);
    const parent_tokens = std.fmt.bytesToHex(binding.parent_study_tokens_sha256, .lower);
    const study_tokens = std.fmt.bytesToHex(binding.study_tokens_sha256, .lower);
    if (std.mem.eql(u8, &parent_tokens, &study_tokens) or !std.mem.eql(u8, &m.parent_manifest_sha256, &parent) or !std.meta.eql(m.identity, identityManifest(binding.identity)) or !std.mem.eql(u8, &m.source_freeze_sha256, &freeze) or !std.mem.eql(u8, &m.binary_sha256, &binary) or !std.mem.eql(u8, &m.exporter_sha256, &exporter) or !std.mem.eql(u8, &m.parent_study_tokens_sha256, &parent_tokens) or !std.mem.eql(u8, &m.study_tokens_sha256, &study_tokens)) return error.InvalidTrainingAugmentationBinding;
    var sum: usize = manifest_reservation;
    for (m.payloads) |row| sum = try std.math.add(usize, sum, row.bytes);
    if (sum != m.required_bytes) return error.InvalidTrainingAugmentation;
    for (payload_names) |name| _ = try findPayload(m.payloads, name);
    // Shape-derived payload sizes are authenticated before allocation so a
    // forged manifest cannot turn its declared count into an allocation probe.
    var expected_elements: usize = 0;
    for (m.dimensions) |dimension| expected_elements = try std.math.add(usize, expected_elements, try std.math.mul(usize, m.samples_per_stratum, dimension));
    const expected_positions = try std.math.mul(usize, m.batch_count, m.samples_per_stratum);
    const expected_rows = [_]struct { name: []const u8, count: usize, size: usize }{
        .{ .name = payload_names[0], .count = expected_elements, .size = @sizeOf(f32) },
        .{ .name = payload_names[1], .count = expected_positions, .size = @sizeOf(u64) },
        .{ .name = payload_names[2], .count = m.batch_count, .size = @sizeOf(u32) },
        .{ .name = payload_names[3], .count = m.batch_count, .size = @sizeOf(u32) },
        .{ .name = payload_names[4], .count = m.batch_count, .size = @sizeOf(u32) },
    };
    for (expected_rows) |expected| {
        const row = try findPayload(m.payloads, expected.name);
        if (row.count != expected.count or row.bytes != try std.math.mul(usize, expected.count, expected.size)) return error.InvalidTrainingAugmentation;
    }
    const training = try readTyped(f32, allocator, io, dir, m.payloads, payload_names[0]);
    errdefer allocator.free(training);
    const positions = try readTyped(u64, allocator, io, dir, m.payloads, payload_names[1]);
    errdefer allocator.free(positions);
    const layers = try readTyped(u32, allocator, io, dir, m.payloads, payload_names[2]);
    errdefer allocator.free(layers);
    const heads = try readTyped(u32, allocator, io, dir, m.payloads, payload_names[3]);
    errdefer allocator.free(heads);
    const counts = try readTyped(u32, allocator, io, dir, m.payloads, payload_names[4]);
    errdefer allocator.free(counts);
    const dimensions = try allocator.dupe(u32, m.dimensions);
    errdefer allocator.free(dimensions);
    if (layers.len != m.batch_count or heads.len != m.batch_count or counts.len != m.batch_count) return error.InvalidTrainingAugmentation;
    const batches = try allocator.alloc(engine.ReferenceBatch, m.batch_count);
    errdefer allocator.free(batches);
    var element_at: usize = 0;
    var position_at: usize = 0;
    for (batches, 0..) |*batch, i| {
        if (counts[i] != m.samples_per_stratum or dimensions[i] == 0) return error.InvalidTrainingAugmentation;
        for (0..i) |j| if (layers[j] == layers[i] and heads[j] == heads[i]) return error.InvalidTrainingAugmentation;
        const elems = try std.math.mul(usize, counts[i], dimensions[i]);
        const element_end = try std.math.add(usize, element_at, elems);
        const position_end = try std.math.add(usize, position_at, counts[i]);
        if (element_end > training.len or position_end > positions.len) return error.InvalidTrainingAugmentation;
        for (training[element_at..element_end]) |value| if (!std.math.isFinite(value)) return error.InvalidTrainingAugmentation;
        const pos = positions[position_at..position_end];
        for (pos[1..], pos[0 .. pos.len - 1]) |next, prior| if (next <= prior) return error.InvalidTrainingAugmentation;
        batch.* = .{ .consumer_layer = layers[i], .query_head = heads[i], .train = training[element_at..element_end], .heldout = &.{}, .train_positions = pos, .heldout_positions = &.{} };
        element_at = element_end;
        position_at = position_end;
    }
    if (element_at != training.len or position_at != positions.len) return error.InvalidTrainingAugmentation;
    return .{ .allocator = allocator, .manifest_bytes = manifest_bytes, .batches = batches, .training = training, .positions = positions, .consumer_layers = layers, .query_heads = heads, .counts = counts, .dimensions = dimensions, .source_length = m.source_length, .repeat_length = m.repeat_length };
}

pub fn read(allocator: std.mem.Allocator, io: std.Io, binding: Binding) !Owned {
    return readWithPolicy(allocator, io, binding, false);
}

pub fn readReferenceCoverageDense512(allocator: std.mem.Allocator, io: std.Io, binding: Binding) !Owned {
    if (binding.samples_per_stratum != dense_reference_samples_per_stratum) return error.InvalidTrainingAugmentationBinding;
    return readWithPolicy(allocator, io, binding, true);
}

pub const HistoricalOverrideBinding = struct {
    output_dir: []const u8,
    manifest_sha256: [32]u8,
    policy_sha256: [32]u8,
    parent_manifest_sha256: [32]u8,
    complete_manifest_sha256: [32]u8,
    identity: checkpoint.Identity,
    config_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
    base_historical_sha256: [32]u8,
    primary_sha256: [32]u8,
    future_sha256: [32]u8,
    heldout_sha256: [32]u8,
    dense_output_dir: []const u8,
    dense_outer_manifest_sha256: [32]u8,
    dense_nested_manifest_sha256: [32]u8,
    dense_parent_study_tokens_sha256: [32]u8,
    dense_study_tokens_sha256: [32]u8,
    v4_proof_sha256: [32]u8,
    policy_manifest_sha256: [32]u8,
};
pub const HistoricalOverrideOwned = struct {
    owned: Owned,
    derived_historical_sha256: [32]u8,
    eligible_indices_sha256: [32]u8,
    selection_indices_sha256: [32]u8,
    dense_outer_manifest_sha256: [32]u8,
    dense_nested_manifest_sha256: [32]u8,
    v4_proof_sha256: [32]u8,
    policy_manifest_sha256: [32]u8,
    pub fn deinit(self: *@This()) void {
        self.owned.deinit();
        self.* = undefined;
    }
};
const OverridePayload = struct { name: []const u8, bytes: usize, count: usize, sha256: []const u8 };
const OverrideRef = struct { path: []const u8, sha256: []const u8 };
const OverrideManifest = struct {
    schema: []const u8,
    diagnostic_only: bool,
    quality_credit: bool,
    production_qualified: bool,
    training_only: bool,
    heldout_present: bool,
    encoding: []const u8,
    policy_sha256: []const u8,
    policy: std.json.Value,
    source_length: usize,
    repeat_length: usize,
    batch_count: usize,
    samples_per_stratum: u16,
    dimensions: []const u32,
    base_historical_sha256: []const u8,
    derived_historical_sha256: []const u8,
    eligible_indices_sha256: []const u8,
    selection_indices_sha256: []const u8,
    eligible_rows_per_batch: usize,
    selected_rows_per_batch: usize,
    excluded_rows_per_batch: usize,
    identity: struct { weights: []const u8, tokenizer: []const u8, template: []const u8, layout: []const u8 },
    config_sha256: []const u8,
    source_freeze_sha256: []const u8,
    binary_sha256: []const u8,
    exporter_sha256: []const u8,
    parent_study_tokens_sha256: []const u8,
    study_tokens_sha256: []const u8,
    origins: struct { parent: OverrideRef, complete: OverrideRef, base_historical: OverrideRef, dense_outer: OverrideRef, dense_nested: OverrideRef, v4_proof: OverrideRef, policy: OverrideRef },
    complete_unchanged_domains: struct { primary_sha256: []const u8, future_sha256: []const u8, heldout_sha256: []const u8 },
    payloads: []const OverridePayload,
};
fn decodeHex32(value: []const u8) ![32]u8 {
    if (value.len != 64) return error.InvalidTrainingAugmentation;
    var out: [32]u8 = undefined;
    _ = std.fmt.hexToBytes(&out, value) catch return error.InvalidTrainingAugmentation;
    return out;
}
fn expectHex32(value: []const u8, expected: [32]u8) !void {
    const actual = try decodeHex32(value);
    if (!std.mem.eql(u8, &actual, &expected)) return error.InvalidTrainingAugmentationBinding;
}
fn overridePayload(rows: []const OverridePayload, name: []const u8) !OverridePayload {
    var found: ?OverridePayload = null;
    for (rows) |row| if (std.mem.eql(u8, row.name, name)) {
        if (found != null) return error.InvalidTrainingAugmentation;
        found = row;
    };
    return found orelse error.InvalidTrainingAugmentation;
}
fn readOverrideTyped(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, rows: []const OverridePayload, name: []const u8) ![]T {
    const row = try overridePayload(rows, name);
    if (row.bytes != try std.math.mul(usize, row.count, @sizeOf(T))) return error.InvalidTrainingAugmentation;
    const expected = try decodeHex32(row.sha256);
    const out = try allocator.alloc(T, row.count);
    errdefer allocator.free(out);
    var file = try dir.openFile(io, name, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    const bytes = std.mem.sliceAsBytes(out);
    if (try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.InvalidTrainingAugmentation;
    var extra: [1]u8 = undefined;
    if (try file.readPositionalAll(io, &extra, bytes.len) != 0) return error.InvalidTrainingAugmentation;
    var actual: [32]u8 = undefined;
    Sha256.hash(bytes, &actual, .{});
    if (!std.mem.eql(u8, &actual, &expected)) return error.InvalidTrainingAugmentation;
    return out;
}
fn authenticateOverrideRef(allocator: std.mem.Allocator, io: std.Io, ref: OverrideRef) ![32]u8 {
    if (!std.fs.path.isAbsolute(ref.path)) return error.InvalidTrainingAugmentation;
    const expected = try decodeHex32(ref.sha256);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, ref.path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    var actual: [32]u8 = undefined;
    Sha256.hash(bytes, &actual, .{});
    if (!std.mem.eql(u8, &actual, &expected)) return error.InvalidTrainingAugmentation;
    return actual;
}
const DenseOuterState = struct {
    identity: checkpoint.Identity,
    sequence_id: u64,
    logical_tokens: usize,
    kv_revision: u64,
    tokens_sha256: [32]u8,
};
const DenseOuterManifest = struct {
    schema: []const u8,
    nested_training_path: []const u8,
    nested_training_manifest_sha256: []const u8,
    source_before: DenseOuterState,
    source_after: DenseOuterState,
    source_freeze_sha256: []const u8,
    binary_sha256: []const u8,
    exporter_sha256: []const u8,
};
fn authenticateDenseOuter(allocator: std.mem.Allocator, io: std.Io, ref: OverrideRef, binding: HistoricalOverrideBinding) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, ref.path, allocator, .limited(manifest_reservation));
    defer allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(DenseOuterManifest, allocator, bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const m = parsed.value;
    if (!std.mem.eql(u8, m.schema, "antfly.attention-match-reference-coverage-training.v1") or !std.mem.eql(u8, m.nested_training_path, "training") or !std.meta.eql(m.source_before, m.source_after) or !checkpoint.Identity.eql(m.source_before.identity, binding.identity)) return error.InvalidTrainingAugmentationBinding;
    try expectHex32(m.nested_training_manifest_sha256, binding.dense_nested_manifest_sha256);
    try expectHex32(m.source_freeze_sha256, binding.source_freeze_sha256);
    try expectHex32(m.binary_sha256, binding.binary_sha256);
    try expectHex32(m.exporter_sha256, binding.exporter_sha256);
}
fn verifySelectedDenseRows(values: []const f32, positions: []const u64, selected: []const u32, dense_values: []const f32, dense_positions: []const u64, heldout_positions: []const u64, dim: usize) !void {
    if (values.len != try std.math.mul(usize, selected.len, dim) or positions.len != selected.len or dense_values.len != try std.math.mul(usize, dense_positions.len, dim)) return error.InvalidTrainingAugmentationBinding;
    for (positions, selected, 0..) |position, index_u32, j| {
        const index: usize = index_u32;
        if (index >= dense_positions.len or position != dense_positions[index] or !std.mem.eql(u8, std.mem.sliceAsBytes(values[j * dim ..][0..dim]), std.mem.sliceAsBytes(dense_values[index * dim ..][0..dim]))) return error.InvalidTrainingAugmentationBinding;
        for (heldout_positions) |heldout_position| if (position == heldout_position) return error.InvalidTrainingAugmentationBinding;
    }
}
pub fn readHistoricalOverride(allocator: std.mem.Allocator, io: std.Io, binding: HistoricalOverrideBinding, base_batches: []const engine.ReferenceBatch) !HistoricalOverrideOwned {
    if (!std.fs.path.isAbsolute(binding.output_dir)) return error.InvalidTrainingAugmentationBinding;
    var dir = try std.Io.Dir.openDirAbsolute(io, binding.output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    const manifest_bytes = try dir.readFileAlloc(io, "manifest.json", allocator, .limited(manifest_reservation));
    errdefer allocator.free(manifest_bytes);
    var manifest_digest: [32]u8 = undefined;
    Sha256.hash(manifest_bytes, &manifest_digest, .{});
    if (!std.mem.eql(u8, &manifest_digest, &binding.manifest_sha256)) return error.InvalidTrainingAugmentationBinding;
    const parsed = try std.json.parseFromSlice(OverrideManifest, allocator, manifest_bytes, .{ .ignore_unknown_fields = false });
    defer parsed.deinit();
    const m = parsed.value;
    if (!std.mem.eql(u8, m.schema, "antfly.attention-match-historical-kcenter-override.v1") or !m.diagnostic_only or m.quality_credit or m.production_qualified or !m.training_only or m.heldout_present or !std.mem.eql(u8, m.encoding, "native-little-endian-ieee754") or m.source_length != 7936 or m.repeat_length != 512 or m.batch_count != 56 or m.samples_per_stratum != 256 or m.dimensions.len != 56 or m.payloads.len != 6 or m.eligible_rows_per_batch != 448 or m.selected_rows_per_batch != 256 or m.excluded_rows_per_batch != 64) return error.InvalidTrainingAugmentation;
    for (m.dimensions) |dim| if (dim != 512) return error.InvalidTrainingAugmentation;
    const expected_identity = identityManifest(binding.identity);
    if (!std.mem.eql(u8, m.identity.weights, &expected_identity.weights) or !std.mem.eql(u8, m.identity.tokenizer, &expected_identity.tokenizer) or !std.mem.eql(u8, m.identity.template, &expected_identity.template) or !std.mem.eql(u8, m.identity.layout, &expected_identity.layout)) return error.InvalidTrainingAugmentationBinding;
    try expectHex32(m.policy_sha256, binding.policy_sha256);
    try expectHex32(m.config_sha256, binding.config_sha256);
    try expectHex32(m.source_freeze_sha256, binding.source_freeze_sha256);
    try expectHex32(m.binary_sha256, binding.binary_sha256);
    try expectHex32(m.exporter_sha256, binding.exporter_sha256);
    try expectHex32(m.base_historical_sha256, binding.base_historical_sha256);
    try expectHex32(m.complete_unchanged_domains.primary_sha256, binding.primary_sha256);
    try expectHex32(m.complete_unchanged_domains.future_sha256, binding.future_sha256);
    try expectHex32(m.complete_unchanged_domains.heldout_sha256, binding.heldout_sha256);
    try expectHex32(m.origins.parent.sha256, binding.parent_manifest_sha256);
    try expectHex32(m.origins.complete.sha256, binding.complete_manifest_sha256);
    try expectHex32(m.origins.dense_outer.sha256, binding.dense_outer_manifest_sha256);
    try expectHex32(m.origins.dense_nested.sha256, binding.dense_nested_manifest_sha256);
    try expectHex32(m.origins.v4_proof.sha256, binding.v4_proof_sha256);
    try expectHex32(m.origins.policy.sha256, binding.policy_manifest_sha256);
    inline for (.{ m.origins.parent, m.origins.complete, m.origins.base_historical, m.origins.dense_outer, m.origins.dense_nested, m.origins.v4_proof, m.origins.policy }) |ref| _ = try authenticateOverrideRef(allocator, io, ref);
    try authenticateDenseOuter(allocator, io, m.origins.dense_outer, binding);
    inline for ([_][]const u8{ "queries.training.f32", "queries.training_positions.u64", "descriptors.consumer_layers.u32", "descriptors.query_heads.u32", "descriptors.training_counts.u32", "selection.dense_indices.u32" }) |name| _ = try overridePayload(m.payloads, name);
    // Reject attacker-controlled counts before any payload allocation.
    const fixed = [_]struct { name: []const u8, count: usize, size: usize }{
        .{ .name = "queries.training.f32", .count = 56 * 256 * 512, .size = @sizeOf(f32) },
        .{ .name = "queries.training_positions.u64", .count = 56 * 256, .size = @sizeOf(u64) },
        .{ .name = "descriptors.consumer_layers.u32", .count = 56, .size = @sizeOf(u32) },
        .{ .name = "descriptors.query_heads.u32", .count = 56, .size = @sizeOf(u32) },
        .{ .name = "descriptors.training_counts.u32", .count = 56, .size = @sizeOf(u32) },
        .{ .name = "selection.dense_indices.u32", .count = 56 * 256, .size = @sizeOf(u32) },
    };
    for (fixed) |expected| {
        const row = try overridePayload(m.payloads, expected.name);
        if (row.count != expected.count or row.bytes != try std.math.mul(usize, expected.count, expected.size)) return error.InvalidTrainingAugmentation;
    }
    if (base_batches.len != 56) return error.InvalidTrainingAugmentationBinding;
    var dense = try readReferenceCoverageDense512(allocator, io, .{
        .output_dir = binding.dense_output_dir,
        .max_bytes = maximum_bytes - manifest_reservation,
        // The nested Dense training artifact is a sibling of the complete mix
        // under the same authenticated whole-fit parent, not a child of the
        // Dense outer sidecar.
        .parent_manifest_sha256 = binding.parent_manifest_sha256,
        .identity = binding.identity,
        .source_freeze_sha256 = binding.source_freeze_sha256,
        .binary_sha256 = binding.binary_sha256,
        .exporter_sha256 = binding.exporter_sha256,
        .parent_study_tokens_sha256 = binding.dense_parent_study_tokens_sha256,
        .study_tokens_sha256 = binding.dense_study_tokens_sha256,
        .samples_per_stratum = dense_reference_samples_per_stratum,
    });
    defer dense.deinit();
    var dense_manifest_digest: [32]u8 = undefined;
    Sha256.hash(dense.manifest_bytes, &dense_manifest_digest, .{});
    if (!std.mem.eql(u8, &dense_manifest_digest, &binding.dense_nested_manifest_sha256)) return error.InvalidTrainingAugmentationBinding;
    const training = try readOverrideTyped(f32, allocator, io, dir, m.payloads, "queries.training.f32");
    errdefer allocator.free(training);
    const positions = try readOverrideTyped(u64, allocator, io, dir, m.payloads, "queries.training_positions.u64");
    errdefer allocator.free(positions);
    const layers = try readOverrideTyped(u32, allocator, io, dir, m.payloads, "descriptors.consumer_layers.u32");
    errdefer allocator.free(layers);
    const heads = try readOverrideTyped(u32, allocator, io, dir, m.payloads, "descriptors.query_heads.u32");
    errdefer allocator.free(heads);
    const counts = try readOverrideTyped(u32, allocator, io, dir, m.payloads, "descriptors.training_counts.u32");
    errdefer allocator.free(counts);
    const selected = try readOverrideTyped(u32, allocator, io, dir, m.payloads, "selection.dense_indices.u32");
    defer allocator.free(selected);
    if (layers.len != 56 or heads.len != 56 or counts.len != 56 or positions.len != 56 * 256 or training.len != 56 * 256 * 512 or selected.len != 56 * 256) return error.InvalidTrainingAugmentation;
    const dimensions = try allocator.dupe(u32, m.dimensions);
    errdefer allocator.free(dimensions);
    const batches = try allocator.alloc(engine.ReferenceBatch, 56);
    errdefer allocator.free(batches);
    var domain = Sha256.init(.{});
    var selection_digest = Sha256.init(.{});
    selection_digest.update("antfly.historical-kcenter.selection-indices.v1\x00");
    for (batches, 0..) |*batch, i| {
        if (counts[i] != 256) return error.InvalidTrainingAugmentation;
        for (0..i) |j| if (layers[i] == layers[j] and heads[i] == heads[j]) return error.InvalidTrainingAugmentation;
        const values = training[i * 256 * 512 ..][0 .. 256 * 512];
        const pos = positions[i * 256 ..][0..256];
        const indices = selected[i * 256 ..][0..256];
        const dense_batch = dense.batches[i];
        const base_batch = base_batches[i];
        if (dense_batch.consumer_layer != layers[i] or dense_batch.query_head != heads[i] or base_batch.consumer_layer != layers[i] or base_batch.query_head != heads[i] or dense_batch.train_positions.len != 512 or dense_batch.train.len != 512 * 512 or base_batch.heldout_positions.len != 64) return error.InvalidTrainingAugmentationBinding;
        for (values) |value| if (!std.math.isFinite(value)) return error.InvalidTrainingAugmentation;
        for (pos, indices, 0..) |position, index, j| {
            if (position < 7936 or position >= 8448 or index >= 512 or (j != 0 and (position <= pos[j - 1] or index <= indices[j - 1]))) return error.InvalidTrainingAugmentation;
        }
        try verifySelectedDenseRows(values, pos, indices, dense_batch.train, dense_batch.train_positions, base_batch.heldout_positions, 512);
        domain.update(std.mem.sliceAsBytes(values));
        domain.update(std.mem.sliceAsBytes(pos));
        selection_digest.update(std.mem.asBytes(&layers[i]));
        selection_digest.update(std.mem.asBytes(&heads[i]));
        selection_digest.update(std.mem.sliceAsBytes(indices));
        batch.* = .{ .consumer_layer = layers[i], .query_head = heads[i], .train = values, .heldout = &.{}, .train_positions = pos, .heldout_positions = &.{} };
    }
    const derived = domain.finalResult();
    if (!std.mem.eql(u8, &derived, &(try decodeHex32(m.derived_historical_sha256)))) return error.InvalidTrainingAugmentation;
    const actual_selection = selection_digest.finalResult();
    if (!std.mem.eql(u8, &actual_selection, &(try decodeHex32(m.selection_indices_sha256)))) return error.InvalidTrainingAugmentation;
    return .{ .owned = .{ .allocator = allocator, .manifest_bytes = manifest_bytes, .batches = batches, .training = training, .positions = positions, .consumer_layers = layers, .query_heads = heads, .counts = counts, .dimensions = dimensions, .source_length = m.source_length, .repeat_length = m.repeat_length }, .derived_historical_sha256 = derived, .eligible_indices_sha256 = try decodeHex32(m.eligible_indices_sha256), .selection_indices_sha256 = try decodeHex32(m.selection_indices_sha256), .dense_outer_manifest_sha256 = try decodeHex32(m.origins.dense_outer.sha256), .dense_nested_manifest_sha256 = try decodeHex32(m.origins.dense_nested.sha256), .v4_proof_sha256 = try decodeHex32(m.origins.v4_proof.sha256), .policy_manifest_sha256 = try decodeHex32(m.origins.policy.sha256) };
}

test "historical override rejects selected dense query byte mismatch" {
    const dense = [_]f32{ 1, 2, 3, 4 };
    const selected_values = [_]f32{ 1, 9 };
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, verifySelectedDenseRows(&selected_values, &.{10}, &.{0}, &dense, &.{ 10, 11 }, &.{12}, 2));
}

test "historical override rejects logical heldout position contamination regardless bytes" {
    const dense = [_]f32{ 1, 2, 3, 4 };
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, verifySelectedDenseRows(dense[0..2], &.{10}, &.{0}, &dense, &.{ 10, 11 }, &.{10}, 2));
}

test "training augmentation checked cap and strict shape" {
    var positions: [320]u64 = undefined;
    for (&positions, 0..) |*position, i| position.* = @intCast(i + 1);
    const training_rows: [320]f32 = @splat(1);
    const batch: engine.ReferenceBatch = .{ .consumer_layer = 1, .query_head = 0, .train = &training_rows, .heldout = &.{}, .train_positions = &positions, .heldout_positions = &.{} };
    try std.testing.expectError(error.InvalidTrainingAugmentation, plan(.{ .batches = &.{batch}, .consumers = &.{}, .study_pass = 0, .source_length = 1, .repeat_length = 1 }, maximum_bytes));
    const consumer: queries.Consumer = .{ .layer = 1, .donor = 0, .query_heads = 1, .kv_heads = 1, .dim = 1 };
    try std.testing.expectError(error.TrainingAugmentationLimit, plan(.{ .batches = &.{batch}, .consumers = &.{consumer}, .source_length = 1, .repeat_length = 1 }, manifest_reservation));
    positions[160] = positions[159];
    try std.testing.expectError(error.InvalidTrainingAugmentation, plan(.{ .batches = &.{batch}, .consumers = &.{consumer}, .source_length = 1, .repeat_length = 1 }, maximum_bytes));
}

test "study pass1 portfolio admits exactly 32 TRAIN rows and no heldout" {
    var values: [32]f32 = @splat(1);
    var positions: [32]u64 = undefined;
    for (&positions, 0..) |*position, i| position.* = @intCast(i + 1);
    const batches = [_]engine.ReferenceBatch{.{
        .consumer_layer = 0,
        .query_head = 0,
        .train = &values,
        .heldout = &.{},
        .train_positions = &positions,
        .heldout_positions = &.{},
    }};
    const p = try plan(.{
        .batches = &batches,
        .consumers = &.{},
        .samples_per_stratum = portfolio_samples_per_stratum,
        .source_length = 1,
        .repeat_length = 1,
    }, maximum_bytes);
    try std.testing.expect(p.required > manifest_reservation);
    var forbidden = batches;
    forbidden[0].heldout = &.{1};
    forbidden[0].heldout_positions = &.{1};
    try std.testing.expectError(error.InvalidTrainingAugmentation, plan(.{
        .batches = &forbidden,
        .consumers = &.{},
        .samples_per_stratum = portfolio_samples_per_stratum,
        .source_length = 1,
        .repeat_length = 1,
    }, maximum_bytes));
}

test "study pass1 extended portfolio admits exactly 128 TRAIN rows only" {
    var values: [128]f32 = @splat(1);
    var positions: [128]u64 = undefined;
    for (&positions, 0..) |*position, i| position.* = @intCast(i + 1);
    var batches = [_]engine.ReferenceBatch{.{ .consumer_layer = 0, .query_head = 0, .train = &values, .heldout = &.{}, .train_positions = &positions, .heldout_positions = &.{} }};
    const view: View = .{ .batches = &batches, .consumers = &.{}, .samples_per_stratum = extended_portfolio_samples_per_stratum, .source_length = 1, .repeat_length = 1 };
    try std.testing.expect((try plan(view, maximum_bytes)).required > manifest_reservation);
    batches[0].heldout = &.{1};
    try std.testing.expectError(error.InvalidTrainingAugmentation, plan(view, maximum_bytes));
    batches[0].heldout = &.{};
    batches[0].train = values[0..127];
    batches[0].train_positions = positions[0..127];
    try std.testing.expectError(error.InvalidTrainingAugmentation, plan(view, maximum_bytes));
    try std.testing.expectError(error.InvalidTrainingAugmentation, plan(.{ .batches = &batches, .consumers = &.{}, .samples_per_stratum = 64, .source_length = 1, .repeat_length = 1 }, maximum_bytes));
}

test "dense512 is rejected generically and admitted only through typed merge" {
    var train: [410]f32 = @splat(1);
    var heldout: [102]f32 = @splat(2);
    var train_positions: [410]u64 = undefined;
    var heldout_positions: [102]u64 = undefined;
    for (&train_positions, 0..) |*position, i| position.* = @intCast(i + i / 4 + 1);
    for (&heldout_positions, 0..) |*position, i| position.* = @intCast(i * 5 + 5);
    const source: engine.ReferenceBatch = .{ .consumer_layer = 0, .query_head = 0, .train = &train, .heldout = &heldout, .train_positions = &train_positions, .heldout_positions = &heldout_positions };
    try std.testing.expectError(error.InvalidTrainingAugmentation, mergeCollectorBatchesExact(std.testing.allocator, &.{source}, dense_reference_samples_per_stratum));
    var merged = try mergeReferenceCoverageDense512(std.testing.allocator, &.{source});
    defer merged.deinit();
    try std.testing.expectEqual(@as(usize, 512), merged.batches[0].train_positions.len);
    try std.testing.expectEqual(@as(usize, 0), merged.batches[0].heldout_positions.len);
}

test "training augmentation filesystem round trip hash cap and no overwrite" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const outer_len = try tmp.dir.realPath(io, &path_buffer);
    const output_dir = try std.fs.path.join(a, &.{ path_buffer[0..outer_len], "augmentation" });
    defer a.free(output_dir);
    const capped_dir = try std.fs.path.join(a, &.{ path_buffer[0..outer_len], "cap" });
    defer a.free(capped_dir);

    var train: [640]f32 = undefined;
    var positions: [320]u64 = undefined;
    for (&positions, 0..) |*position, i| position.* = @intCast(i + 100);
    for (&train, 0..) |*value, i| value.* = @floatFromInt(i % 31);
    const batch: engine.ReferenceBatch = .{
        .consumer_layer = 3,
        .query_head = 1,
        .train = &train,
        .heldout = &.{},
        .train_positions = &positions,
        .heldout_positions = &.{},
    };
    const consumer: queries.Consumer = .{ .layer = 3, .donor = 0, .query_heads = 2, .kv_heads = 1, .dim = 2 };
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const binding: Binding = .{
        .output_dir = output_dir,
        .parent_manifest_sha256 = @splat(5),
        .identity = identity,
        .source_freeze_sha256 = @splat(6),
        .binary_sha256 = @splat(7),
        .exporter_sha256 = @splat(8),
        .parent_study_tokens_sha256 = @splat(10),
        .study_tokens_sha256 = @splat(11),
    };
    const view: View = .{ .batches = &.{batch}, .consumers = &.{consumer}, .parent_study_tokens_sha256 = binding.parent_study_tokens_sha256, .study_tokens_sha256 = binding.study_tokens_sha256, .source_length = 7936, .repeat_length = 512 };
    const expected = try plan(view, maximum_bytes);
    const written = try write(a, io, binding, view);
    try std.testing.expectEqual(expected.required, written.required_bytes);
    var restored = try read(a, io, binding);
    defer restored.deinit();
    try std.testing.expectEqual(@as(usize, 1), restored.batches.len);
    try std.testing.expectEqualSlices(f32, &train, restored.batches[0].train);
    try std.testing.expectEqualSlices(u64, &positions, restored.batches[0].train_positions);
    try std.testing.expectEqual(@as(usize, 0), restored.batches[0].heldout.len);
    try std.testing.expectError(error.PathAlreadyExists, write(a, io, binding, view));

    const capped_binding: Binding = .{
        .output_dir = capped_dir,
        .parent_manifest_sha256 = binding.parent_manifest_sha256,
        .identity = identity,
        .source_freeze_sha256 = binding.source_freeze_sha256,
        .binary_sha256 = binding.binary_sha256,
        .exporter_sha256 = binding.exporter_sha256,
        .parent_study_tokens_sha256 = binding.parent_study_tokens_sha256,
        .study_tokens_sha256 = binding.study_tokens_sha256,
        .max_bytes = expected.required - 1,
    };
    try std.testing.expectError(error.TrainingAugmentationLimit, write(a, io, capped_binding, view));

    var dir = try std.Io.Dir.openDirAbsolute(io, output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    var payload = try dir.openFile(io, payload_names[0], .{ .mode = .read_write, .allow_directory = false, .follow_symlinks = false });
    defer payload.close(io);
    try payload.writePositionalAll(io, &.{0xff}, 0);
    try payload.sync(io);
    try std.testing.expectError(error.InvalidTrainingAugmentation, read(a, io, binding));

    const wrong_parent: Binding = .{
        .output_dir = output_dir,
        .parent_manifest_sha256 = @splat(9),
        .identity = identity,
        .source_freeze_sha256 = binding.source_freeze_sha256,
        .binary_sha256 = binding.binary_sha256,
        .exporter_sha256 = binding.exporter_sha256,
        .parent_study_tokens_sha256 = binding.parent_study_tokens_sha256,
        .study_tokens_sha256 = binding.study_tokens_sha256,
    };
    try std.testing.expectError(error.InvalidTrainingAugmentationBinding, read(a, io, wrong_parent));
}

test "training augmentation collector merge is chronological lossless and rejects duplicates" {
    const a = std.testing.allocator;
    const source: engine.ReferenceBatch = .{
        .consumer_layer = 2,
        .query_head = 3,
        .train = &.{ 10, 11, 30, 31 },
        .heldout = &.{ 20, 21, 40, 41 },
        .train_positions = &.{ 10, 30 },
        .heldout_positions = &.{ 20, 40 },
    };
    // The production contract is 320 rows. Build the same interleaving at
    // that cardinality so this test exercises the real shape, not a bypass.
    var train: [512]f32 = undefined;
    var heldout: [128]f32 = undefined;
    var train_positions: [256]u64 = undefined;
    var heldout_positions: [64]u64 = undefined;
    for (&train_positions, 0..) |*position, i| position.* = @intCast(i + 1 + i / 4);
    for (&heldout_positions, 0..) |*position, i| position.* = @intCast(5 * i + 5);
    for (&train, 0..) |*value, i| value.* = @floatFromInt(i);
    for (&heldout, 0..) |*value, i| value.* = @floatFromInt(i + 1000);
    const full: engine.ReferenceBatch = .{ .consumer_layer = source.consumer_layer, .query_head = source.query_head, .train = &train, .heldout = &heldout, .train_positions = &train_positions, .heldout_positions = &heldout_positions };
    var merged = try mergeCollectorBatches(a, &.{full});
    defer merged.deinit();
    try std.testing.expectEqual(@as(usize, 320), merged.batches[0].train_positions.len);
    try std.testing.expectEqual(@as(usize, 640), merged.batches[0].train.len);
    try std.testing.expectEqual(@as(usize, 0), merged.batches[0].heldout.len);
    for (merged.batches[0].train_positions[1..], merged.batches[0].train_positions[0..319]) |next, prior| try std.testing.expect(next > prior);

    const duplicate_heldout_positions: [64]u64 = @splat(5);
    const duplicate: engine.ReferenceBatch = .{ .consumer_layer = 2, .query_head = 3, .train = &train, .heldout = &heldout, .train_positions = &train_positions, .heldout_positions = &duplicate_heldout_positions };
    try std.testing.expectError(error.InvalidTrainingAugmentation, mergeCollectorBatches(a, &.{duplicate}));
}

test "extended portfolio merge reconstructs exactly its independent 128 rows" {
    var train: [102]f32 = undefined;
    var heldout: [26]f32 = undefined;
    var train_positions: [102]u64 = undefined;
    var heldout_positions: [26]u64 = undefined;
    var ti: usize = 0;
    var hi: usize = 0;
    for (0..128) |i| {
        const position: u64 = @intCast(i + 1);
        if (i % 5 == 0) {
            heldout_positions[hi] = position;
            heldout[hi] = @floatFromInt(i);
            hi += 1;
        } else {
            train_positions[ti] = position;
            train[ti] = @floatFromInt(i);
            ti += 1;
        }
    }
    try std.testing.expectEqual(@as(usize, 102), ti);
    try std.testing.expectEqual(@as(usize, 26), hi);
    const source: engine.ReferenceBatch = .{ .consumer_layer = 1, .query_head = 2, .train = &train, .heldout = &heldout, .train_positions = &train_positions, .heldout_positions = &heldout_positions };
    var merged = try mergeCollectorBatchesExact(std.testing.allocator, &.{source}, extended_portfolio_samples_per_stratum);
    defer merged.deinit();
    try std.testing.expectEqual(@as(usize, 128), merged.batches[0].train_positions.len);
    try std.testing.expectEqual(@as(usize, 0), merged.batches[0].heldout.len);
    for (merged.batches[0].train_positions, 0..) |position, i| try std.testing.expectEqual(@as(u64, @intCast(i + 1)), position);
}
