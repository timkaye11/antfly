// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const engine = @import("attention_match_engine.zig");
const Control = @import("attention_match.zig").Control;
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const maximum_bytes: usize = 256 * 1024 * 1024;
pub const manifest_reservation: usize = 64 * 1024;
const chunk_bytes: usize = 1024 * 1024;

pub const Method = enum { top1, top2 };

pub const Binding = struct {
    output_dir: []const u8,
    identity: checkpoint.Identity,
    config_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    exporter_sha256: [32]u8,
    limits: checkpoint.Limits,
    max_bytes: usize = maximum_bytes,
};

pub const View = struct {
    source: checkpoint.State,
    batches: []const engine.ReferenceBatch,
    layout: engine.Layout,
    query_samples: u16,
    method: Method,
    fit: FitConfig,
    source_has_metadata: bool,
    original_global_device_bytes: usize,
    control: Control,
};

pub const FitConfig = struct {
    prefix_rows: usize,
    recent_rows: usize,
    chunk_rows: usize,
    retention_divisor: usize,
    min_retention_divisor: usize,
    max_relative_mass_error: f64,
    max_relative_output_rmse: f64,
    max_sweeps: usize,
    nnls_tolerance: f64,
    min_weight: f64,
    selection: enum { residual_omp },
    mass_solver: enum { active_set },
    validation: enum { whole_head },
    rejected_head_policy: enum { preserve_original },
    storage_precision: enum { f16 },
    independent_fp64_publication: bool,
};

pub const Result = struct {
    required_bytes: usize,
    checkpoint_bytes: usize,
    checkpoint_sha256: [32]u8,
};

const Payload = struct {
    name: []const u8,
    bytes: usize,
    count: usize,
    sha256: [32]u8,
};

pub const Writer = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    binding: Binding,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, binding: Binding) !Writer {
        if (binding.output_dir.len == 0 or !std.fs.path.isAbsolute(binding.output_dir) or
            binding.max_bytes == 0 or binding.max_bytes > maximum_bytes)
            return error.InvalidWholeFitInputBinding;
        return .{ .allocator = allocator, .io = io, .binding = binding };
    }

    pub fn write(self: *Writer, view: View) !Result {
        if (view.query_samples != 320 and view.query_samples != 512)
            return error.InvalidWholeFitInput;
        if (!view.source.identity.eql(self.binding.identity))
            return error.InvalidWholeFitInputBinding;
        if (view.original_global_device_bytes != try engine.globalStorageBytes(view.source, .f16, view.source_has_metadata))
            return error.InvalidWholeFitInput;
        const encoding_plan = try plan(view, self.binding.limits, self.binding.max_bytes);
        try std.Io.Dir.createDirAbsolute(self.io, self.binding.output_dir, std.Io.File.Permissions.fromMode(0o700));
        var dir = try std.Io.Dir.openDirAbsolute(self.io, self.binding.output_dir, .{ .follow_symlinks = false });
        defer dir.close(self.io);

        var payloads: [11]Payload = undefined;
        const exported = blk: {
            var checkpoint_file = try dir.createFile(self.io, "original.checkpoint.f16", .{ .exclusive = true, .truncate = false });
            defer checkpoint_file.close(self.io);
            var bounded: BoundedFileSink = .{ .file = &checkpoint_file, .io = self.io, .remaining = encoding_plan.checkpoint_bytes };
            const value = try checkpoint.exportEncoded(view.source, self.binding.limits, .f16, &bounded, view.control);
            try checkpoint_file.sync(self.io);
            if (bounded.remaining != 0 or value.bytes != encoding_plan.checkpoint_bytes) return error.InvalidWholeFitInput;
            break :blk value;
        };
        payloads[0] = .{ .name = "original.checkpoint.f16", .bytes = exported.bytes, .count = 1, .sha256 = exported.file_sha256 };

        payloads[1] = try self.writeBatchField(dir, "queries.train.f32", view.batches, .train, encoding_plan.train_elements);
        payloads[2] = try self.writeBatchField(dir, "queries.heldout.f32", view.batches, .heldout, encoding_plan.heldout_elements);
        payloads[3] = try self.writeBatchField(dir, "queries.train_positions.u64", view.batches, .train_positions, encoding_plan.train_positions);
        payloads[4] = try self.writeBatchField(dir, "queries.heldout_positions.u64", view.batches, .heldout_positions, encoding_plan.heldout_positions);

        const batch_descriptors = try self.descriptors(view.batches);
        defer batch_descriptors.deinit(self.allocator);
        payloads[5] = try self.writeSlice(dir, "queries.consumer_layers.u32", batch_descriptors.consumer_layers);
        payloads[6] = try self.writeSlice(dir, "queries.heads.u32", batch_descriptors.heads);
        payloads[7] = try self.writeSlice(dir, "queries.train_counts.u32", batch_descriptors.train_counts);
        payloads[8] = try self.writeSlice(dir, "queries.heldout_counts.u32", batch_descriptors.heldout_counts);
        payloads[9] = try self.writeSlice(dir, "layout.query_heads_by_layer.u32", view.layout.query_heads_by_layer);

        // Scale and softcap stay in one typed file; the manifest records the
        // split so the reader cannot silently exchange the two equal domains.
        var layout_file = try dir.createFile(self.io, "layout.scales_softcaps.f64", .{ .exclusive = true, .truncate = false });
        defer layout_file.close(self.io);
        var layout_hash = Sha256.init(.{});
        try writeHashed(self.io, &layout_file, &layout_hash, std.mem.sliceAsBytes(view.layout.scales_by_donor));
        try writeHashed(self.io, &layout_file, &layout_hash, std.mem.sliceAsBytes(view.layout.softcaps_by_donor));
        try layout_file.sync(self.io);
        payloads[10] = .{ .name = "layout.scales_softcaps.f64", .bytes = try std.math.mul(usize, view.layout.scales_by_donor.len + view.layout.softcaps_by_donor.len, @sizeOf(f64)), .count = view.layout.scales_by_donor.len + view.layout.softcaps_by_donor.len, .sha256 = layout_hash.finalResult() };
        try validatePayloadClosure(&payloads, encoding_plan.required_bytes);

        const id = self.binding.identity;
        const manifest = try std.json.Stringify.valueAlloc(self.allocator, .{
            .schema = "antfly.attention-match-whole-fit-input.v1",
            .diagnostic_only = true,
            .quality_credit = false,
            .encoding = "native-little-endian-ieee754",
            .checkpoint_encoding = "f16-v3",
            .required_bytes = encoding_plan.required_bytes,
            .maximum_bytes = self.binding.max_bytes,
            .identity = .{ .weights = std.fmt.bytesToHex(id.weights, .lower), .tokenizer = std.fmt.bytesToHex(id.tokenizer, .lower), .template = std.fmt.bytesToHex(id.template, .lower), .layout = std.fmt.bytesToHex(id.layout, .lower) },
            .config_sha256 = std.fmt.bytesToHex(self.binding.config_sha256, .lower),
            .source_freeze_sha256 = std.fmt.bytesToHex(self.binding.source_freeze_sha256, .lower),
            .binary_sha256 = std.fmt.bytesToHex(self.binding.binary_sha256, .lower),
            .exporter_sha256 = std.fmt.bytesToHex(self.binding.exporter_sha256, .lower),
            .query_samples = view.query_samples,
            .method = @tagName(view.method),
            .fit = view.fit,
            .source_has_metadata = view.source_has_metadata,
            .original_global_device_bytes = view.original_global_device_bytes,
            .logical_length = view.source.logical_length,
            .batch_count = view.batches.len,
            .layer_count = view.layout.query_heads_by_layer.len,
            .donor_count = view.layout.scales_by_donor.len,
            .layout_scale_count = view.layout.scales_by_donor.len,
            .layout_softcap_count = view.layout.softcaps_by_donor.len,
            .payloads = payloads,
        }, .{});
        defer self.allocator.free(manifest);
        if (manifest.len + 1 > manifest_reservation) return error.WholeFitInputLimit;
        {
            var manifest_file = try dir.createFile(self.io, "manifest.json.part", .{ .exclusive = true, .truncate = false });
            defer manifest_file.close(self.io);
            try manifest_file.writeStreamingAll(self.io, manifest);
            try manifest_file.writeStreamingAll(self.io, "\n");
            try manifest_file.sync(self.io);
        }
        try std.Io.Dir.rename(dir, "manifest.json.part", dir, "manifest.json", self.io);
        return .{ .required_bytes = encoding_plan.required_bytes, .checkpoint_bytes = exported.bytes, .checkpoint_sha256 = exported.file_sha256 };
    }

    const BatchField = enum { train, heldout, train_positions, heldout_positions };
    fn writeBatchField(self: *Writer, dir: std.Io.Dir, name: []const u8, batches: []const engine.ReferenceBatch, comptime field: BatchField, expected_count: usize) !Payload {
        var file = try dir.createFile(self.io, name, .{ .exclusive = true, .truncate = false });
        defer file.close(self.io);
        var hash = Sha256.init(.{});
        var count: usize = 0;
        for (batches) |batch| {
            const values = switch (field) {
                .train => std.mem.sliceAsBytes(batch.train),
                .heldout => std.mem.sliceAsBytes(batch.heldout),
                .train_positions => std.mem.sliceAsBytes(batch.train_positions),
                .heldout_positions => std.mem.sliceAsBytes(batch.heldout_positions),
            };
            const elements = switch (field) {
                .train => batch.train.len,
                .heldout => batch.heldout.len,
                .train_positions => batch.train_positions.len,
                .heldout_positions => batch.heldout_positions.len,
            };
            count = try std.math.add(usize, count, elements);
            try writeHashed(self.io, &file, &hash, values);
        }
        if (count != expected_count) return error.InvalidWholeFitInput;
        try file.sync(self.io);
        const element_size: usize = if (field == .train or field == .heldout) @sizeOf(f32) else @sizeOf(u64);
        return .{ .name = name, .bytes = try std.math.mul(usize, count, element_size), .count = count, .sha256 = hash.finalResult() };
    }

    fn writeSlice(self: *Writer, dir: std.Io.Dir, name: []const u8, values: anytype) !Payload {
        var file = try dir.createFile(self.io, name, .{ .exclusive = true, .truncate = false });
        defer file.close(self.io);
        const bytes = std.mem.sliceAsBytes(values);
        var hash = Sha256.init(.{});
        try writeHashed(self.io, &file, &hash, bytes);
        try file.sync(self.io);
        return .{ .name = name, .bytes = bytes.len, .count = values.len, .sha256 = hash.finalResult() };
    }

    const Descriptors = struct {
        consumer_layers: []u32,
        heads: []u32,
        train_counts: []u32,
        heldout_counts: []u32,
        fn deinit(self: @This(), allocator: std.mem.Allocator) void {
            allocator.free(self.consumer_layers);
            allocator.free(self.heads);
            allocator.free(self.train_counts);
            allocator.free(self.heldout_counts);
        }
    };
    fn descriptors(self: *Writer, batches: []const engine.ReferenceBatch) !Descriptors {
        const layers = try self.allocator.alloc(u32, batches.len);
        errdefer self.allocator.free(layers);
        const heads = try self.allocator.alloc(u32, batches.len);
        errdefer self.allocator.free(heads);
        const train = try self.allocator.alloc(u32, batches.len);
        errdefer self.allocator.free(train);
        const heldout = try self.allocator.alloc(u32, batches.len);
        errdefer self.allocator.free(heldout);
        for (batches, 0..) |batch, i| {
            layers[i] = batch.consumer_layer;
            heads[i] = batch.query_head;
            train[i] = std.math.cast(u32, batch.train_positions.len) orelse return error.WholeFitInputLimit;
            heldout[i] = std.math.cast(u32, batch.heldout_positions.len) orelse return error.WholeFitInputLimit;
        }
        return .{ .consumer_layers = layers, .heads = heads, .train_counts = train, .heldout_counts = heldout };
    }
};

const Plan = struct { required_bytes: usize, checkpoint_bytes: usize, train_elements: usize, heldout_elements: usize, train_positions: usize, heldout_positions: usize };
pub fn plan(view: View, limits: checkpoint.Limits, maximum: usize) !Plan {
    if (maximum == 0 or maximum > maximum_bytes or view.batches.len == 0 or
        view.layout.scales_by_donor.len != view.layout.softcaps_by_donor.len)
        return error.InvalidWholeFitInput;
    const checkpoint_bytes = try checkpoint.encodedSize(view.source, limits, .f16, view.control);
    var train_elements: usize = 0;
    var heldout_elements: usize = 0;
    var train_positions: usize = 0;
    var heldout_positions: usize = 0;
    for (view.batches) |batch| {
        const sample_count = try std.math.add(usize, batch.train_positions.len, batch.heldout_positions.len);
        if (batch.train_positions.len == 0 or batch.heldout_positions.len == 0 or
            sample_count != @as(usize, view.query_samples) or
            batch.train.len % batch.train_positions.len != 0 or batch.heldout.len % batch.heldout_positions.len != 0 or
            batch.train.len / batch.train_positions.len != batch.heldout.len / batch.heldout_positions.len)
            return error.InvalidWholeFitInput;
        train_elements = try std.math.add(usize, train_elements, batch.train.len);
        heldout_elements = try std.math.add(usize, heldout_elements, batch.heldout.len);
        train_positions = try std.math.add(usize, train_positions, batch.train_positions.len);
        heldout_positions = try std.math.add(usize, heldout_positions, batch.heldout_positions.len);
    }
    var total = try std.math.add(usize, manifest_reservation, checkpoint_bytes);
    total = try addTyped(total, try std.math.add(usize, train_elements, heldout_elements), @sizeOf(f32));
    total = try addTyped(total, try std.math.add(usize, train_positions, heldout_positions), @sizeOf(u64));
    const descriptor_count = try std.math.add(usize, try std.math.mul(usize, view.batches.len, 4), view.layout.query_heads_by_layer.len);
    total = try addTyped(total, descriptor_count, @sizeOf(u32));
    total = try addTyped(total, try std.math.add(usize, view.layout.scales_by_donor.len, view.layout.softcaps_by_donor.len), @sizeOf(f64));
    if (total > maximum) return error.WholeFitInputLimit;
    return .{ .required_bytes = total, .checkpoint_bytes = checkpoint_bytes, .train_elements = train_elements, .heldout_elements = heldout_elements, .train_positions = train_positions, .heldout_positions = heldout_positions };
}

fn addTyped(total: usize, count: usize, size: usize) !usize {
    return std.math.add(usize, total, try std.math.mul(usize, count, size));
}

const BoundedFileSink = struct {
    file: *std.Io.File,
    io: std.Io,
    remaining: usize,
    pub fn writeAll(self: *@This(), bytes: []const u8) !void {
        if (bytes.len > self.remaining) return error.WholeFitInputLimit;
        try self.file.writeStreamingAll(self.io, bytes);
        self.remaining -= bytes.len;
    }
};

fn writeHashed(io: std.Io, file: *std.Io.File, hash: *Sha256, bytes: []const u8) !void {
    var offset: usize = 0;
    while (offset < bytes.len) {
        const take = @min(chunk_bytes, bytes.len - offset);
        const part = bytes[offset..][0..take];
        try file.writeStreamingAll(io, part);
        hash.update(part);
        offset += take;
    }
}

const payload_names = [_][]const u8{
    "original.checkpoint.f16",
    "queries.train.f32",
    "queries.heldout.f32",
    "queries.train_positions.u64",
    "queries.heldout_positions.u64",
    "queries.consumer_layers.u32",
    "queries.heads.u32",
    "queries.train_counts.u32",
    "queries.heldout_counts.u32",
    "layout.query_heads_by_layer.u32",
    "layout.scales_softcaps.f64",
};

fn validatePayloadClosure(payloads: []const Payload, required_bytes: usize) !void {
    if (payloads.len != payload_names.len) return error.InvalidWholeFitInput;
    var total: usize = manifest_reservation;
    for (payload_names) |expected| {
        var matches: usize = 0;
        for (payloads) |payload| if (std.mem.eql(u8, payload.name, expected)) {
            matches += 1;
            total = try std.math.add(usize, total, payload.bytes);
        };
        if (matches != 1) return error.InvalidWholeFitInput;
    }
    if (total != required_bytes) return error.InvalidWholeFitInput;
}

const IdentityManifest = struct { weights: [64]u8, tokenizer: [64]u8, template: [64]u8, layout: [64]u8 };
const Manifest = struct {
    schema: []const u8,
    diagnostic_only: bool,
    quality_credit: bool,
    encoding: []const u8,
    checkpoint_encoding: []const u8,
    required_bytes: usize,
    maximum_bytes: usize,
    identity: IdentityManifest,
    config_sha256: [64]u8,
    source_freeze_sha256: [64]u8,
    binary_sha256: [64]u8,
    exporter_sha256: [64]u8,
    query_samples: u16,
    method: []const u8,
    fit: FitConfig,
    source_has_metadata: bool,
    original_global_device_bytes: usize,
    logical_length: u64,
    batch_count: usize,
    layer_count: usize,
    donor_count: usize,
    layout_scale_count: usize,
    layout_softcap_count: usize,
    payloads: []const Payload,
};

pub const OwnedInput = struct {
    allocator: std.mem.Allocator,
    checkpoint_owned: checkpoint.Owned,
    checkpoint_identity: checkpoint.Identity,
    checkpoint_payload_sha256: [32]u8,
    checkpoint_payload_bytes: usize,
    checkpoint_live: bool = true,
    batches: []engine.ReferenceBatch,
    train: []f32,
    heldout: []f32,
    train_positions: []u64,
    heldout_positions: []u64,
    consumer_layers: []u32,
    heads: []u32,
    train_counts: []u32,
    heldout_counts: []u32,
    query_heads_by_layer: []u32,
    scales_softcaps: []f64,
    layout: engine.Layout,
    fit: FitConfig,
    method: Method,
    query_samples: u16,
    source_has_metadata: bool,
    original_global_device_bytes: usize,

    pub fn deinit(self: *OwnedInput) void {
        if (self.checkpoint_live) self.checkpoint_owned.deinit();
        inline for (.{ "batches", "train", "heldout", "train_positions", "heldout_positions", "consumer_layers", "heads", "train_counts", "heldout_counts", "query_heads_by_layer", "scales_softcaps" }) |field|
            self.allocator.free(@field(self, field));
        self.* = undefined;
    }

    pub fn releaseCheckpoint(self: *OwnedInput) !void {
        if (!self.checkpoint_live) return error.InvalidWholeFitInput;
        self.checkpoint_owned.deinit();
        self.checkpoint_live = false;
    }

    pub fn takeCheckpoint(self: *OwnedInput) !checkpoint.Owned {
        if (!self.checkpoint_live) return error.InvalidWholeFitInput;
        self.checkpoint_live = false;
        return self.checkpoint_owned;
    }

    pub fn liveBytes(self: *const OwnedInput) !usize {
        var bytes: usize = if (self.checkpoint_live) try self.checkpoint_owned.state.residentBytes() else 0;
        inline for (.{ self.train, self.heldout }) |slice|
            bytes = try std.math.add(usize, bytes, try std.math.mul(usize, slice.len, @sizeOf(f32)));
        inline for (.{ self.train_positions, self.heldout_positions }) |slice|
            bytes = try std.math.add(usize, bytes, try std.math.mul(usize, slice.len, @sizeOf(u64)));
        bytes = try std.math.add(usize, bytes, try std.math.mul(usize, self.batches.len, @sizeOf(engine.ReferenceBatch)));
        inline for (.{ self.consumer_layers, self.heads, self.train_counts, self.heldout_counts, self.query_heads_by_layer }) |slice|
            bytes = try std.math.add(usize, bytes, try std.math.mul(usize, slice.len, @sizeOf(u32)));
        bytes = try std.math.add(usize, bytes, try std.math.mul(usize, self.scales_softcaps.len, @sizeOf(f64)));
        return bytes;
    }

    pub fn installCheckpoint(self: *OwnedInput, owned: checkpoint.Owned, payload_sha256: [32]u8) !void {
        if (self.checkpoint_live or !owned.state.identity.eql(self.checkpoint_identity) or
            !std.mem.eql(u8, &payload_sha256, &self.checkpoint_payload_sha256))
            return error.InvalidWholeFitInputBinding;
        self.checkpoint_owned = owned;
        self.checkpoint_live = true;
    }
};

pub const CheckpointOnly = struct {
    owned: checkpoint.Owned,
    payload_sha256: [32]u8,
    peak_live_bytes: usize,
};

pub fn read(allocator: std.mem.Allocator, io: std.Io, expected: Binding) !OwnedInput {
    var dir = try std.Io.Dir.openDirAbsolute(io, expected.output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    const manifest_bytes = try dir.readFileAlloc(io, "manifest.json", allocator, .limited(manifest_reservation));
    defer allocator.free(manifest_bytes);
    const parsed = try std.json.parseFromSlice(Manifest, allocator, manifest_bytes, .{ .ignore_unknown_fields = false });
    defer parsed.deinit();
    const m = parsed.value;
    if (!std.mem.eql(u8, m.schema, "antfly.attention-match-whole-fit-input.v1") or
        !m.diagnostic_only or m.quality_credit or
        !std.mem.eql(u8, m.encoding, "native-little-endian-ieee754") or
        !std.mem.eql(u8, m.checkpoint_encoding, "f16-v3") or
        m.maximum_bytes != expected.max_bytes or m.required_bytes > expected.max_bytes or
        (m.query_samples != 320 and m.query_samples != 512) or m.payloads.len != payload_names.len)
        return error.InvalidWholeFitInput;
    const expected_identity: IdentityManifest = .{
        .weights = std.fmt.bytesToHex(expected.identity.weights, .lower),
        .tokenizer = std.fmt.bytesToHex(expected.identity.tokenizer, .lower),
        .template = std.fmt.bytesToHex(expected.identity.template, .lower),
        .layout = std.fmt.bytesToHex(expected.identity.layout, .lower),
    };
    const expected_config = std.fmt.bytesToHex(expected.config_sha256, .lower);
    const expected_freeze = std.fmt.bytesToHex(expected.source_freeze_sha256, .lower);
    const expected_binary = std.fmt.bytesToHex(expected.binary_sha256, .lower);
    const expected_exporter = std.fmt.bytesToHex(expected.exporter_sha256, .lower);
    const config_json = try std.json.Stringify.valueAlloc(allocator, .{ .schema = "antfly.attention-match-whole-fit-config.v1", .fit = m.fit, .query_samples = m.query_samples, .method = m.method }, .{});
    defer allocator.free(config_json);
    var config_digest: [32]u8 = undefined;
    Sha256.hash(config_json, &config_digest, .{});
    const canonical_config = std.fmt.bytesToHex(config_digest, .lower);
    if (!std.meta.eql(m.identity, expected_identity) or
        !std.mem.eql(u8, &m.config_sha256, &expected_config) or
        !std.mem.eql(u8, &m.config_sha256, &canonical_config) or
        !std.mem.eql(u8, &m.source_freeze_sha256, &expected_freeze) or
        !std.mem.eql(u8, &m.binary_sha256, &expected_binary) or
        !std.mem.eql(u8, &m.exporter_sha256, &expected_exporter))
        return error.InvalidWholeFitInputBinding;
    try validatePayloadClosure(m.payloads, m.required_bytes);

    const checkpoint_row = try findPayload(m.payloads, "original.checkpoint.f16");
    try verifyFile(io, dir, checkpoint_row);
    var checkpoint_file = try dir.openFile(io, checkpoint_row.name, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer checkpoint_file.close(io);
    var reader: FileReader = .{ .file = &checkpoint_file, .io = io };
    var decoded = try checkpoint.decodeReaderWithEncodingAndControl(allocator, &reader, checkpoint_row.bytes, expected.identity, expected.limits, .{});
    var decoded_live = true;
    defer if (decoded_live) decoded.deinit();
    if (decoded.encoding != .f16 or decoded.checkpoint.state.logical_length != m.logical_length)
        return error.InvalidWholeFitInput;

    var result: OwnedInput = undefined;
    result.allocator = allocator;
    result.checkpoint_owned = decoded.checkpoint;
    result.checkpoint_identity = decoded.checkpoint.state.identity;
    result.checkpoint_payload_sha256 = checkpoint_row.sha256;
    result.checkpoint_payload_bytes = checkpoint_row.bytes;
    result.checkpoint_live = true;
    decoded_live = false;
    errdefer result.checkpoint_owned.deinit();
    result.train = try readTyped(f32, allocator, io, dir, m.payloads, "queries.train.f32");
    errdefer allocator.free(result.train);
    result.heldout = try readTyped(f32, allocator, io, dir, m.payloads, "queries.heldout.f32");
    errdefer allocator.free(result.heldout);
    result.train_positions = try readTyped(u64, allocator, io, dir, m.payloads, "queries.train_positions.u64");
    errdefer allocator.free(result.train_positions);
    result.heldout_positions = try readTyped(u64, allocator, io, dir, m.payloads, "queries.heldout_positions.u64");
    errdefer allocator.free(result.heldout_positions);
    result.consumer_layers = try readTyped(u32, allocator, io, dir, m.payloads, "queries.consumer_layers.u32");
    errdefer allocator.free(result.consumer_layers);
    result.heads = try readTyped(u32, allocator, io, dir, m.payloads, "queries.heads.u32");
    errdefer allocator.free(result.heads);
    result.train_counts = try readTyped(u32, allocator, io, dir, m.payloads, "queries.train_counts.u32");
    errdefer allocator.free(result.train_counts);
    result.heldout_counts = try readTyped(u32, allocator, io, dir, m.payloads, "queries.heldout_counts.u32");
    errdefer allocator.free(result.heldout_counts);
    result.query_heads_by_layer = try readTyped(u32, allocator, io, dir, m.payloads, "layout.query_heads_by_layer.u32");
    errdefer allocator.free(result.query_heads_by_layer);
    result.scales_softcaps = try readTyped(f64, allocator, io, dir, m.payloads, "layout.scales_softcaps.f64");
    errdefer allocator.free(result.scales_softcaps);
    if (m.layout_scale_count != m.layout_softcap_count or
        result.scales_softcaps.len != m.layout_scale_count + m.layout_softcap_count or
        result.query_heads_by_layer.len != m.layer_count or m.donor_count != m.layout_scale_count or
        result.consumer_layers.len != m.batch_count or result.heads.len != m.batch_count or
        result.train_counts.len != m.batch_count or result.heldout_counts.len != m.batch_count)
        return error.InvalidWholeFitInput;
    result.batches = try allocator.alloc(engine.ReferenceBatch, m.batch_count);
    errdefer allocator.free(result.batches);
    var to: usize = 0;
    var ho: usize = 0;
    var tp: usize = 0;
    var hp: usize = 0;
    for (result.batches, 0..) |*batch, i| {
        const nt: usize = result.train_counts[i];
        const nh: usize = result.heldout_counts[i];
        if (try std.math.add(usize, nt, nh) != m.query_samples) return error.InvalidWholeFitInput;
        if (result.consumer_layers[i] >= result.query_heads_by_layer.len or result.heads[i] >= result.query_heads_by_layer[result.consumer_layers[i]]) return error.InvalidWholeFitInput;
        const dim = try headDim(result.checkpoint_owned.state, result.consumer_layers[i], result.heads[i]);
        const ne = try std.math.mul(usize, nt, dim);
        const he = try std.math.mul(usize, nh, dim);
        if (to > result.train.len or ho > result.heldout.len or tp > result.train_positions.len or hp > result.heldout_positions.len or
            ne > result.train.len - to or he > result.heldout.len - ho or nt > result.train_positions.len - tp or nh > result.heldout_positions.len - hp)
            return error.InvalidWholeFitInput;
        batch.* = .{ .consumer_layer = result.consumer_layers[i], .query_head = result.heads[i], .train = result.train[to..][0..ne], .heldout = result.heldout[ho..][0..he], .train_positions = result.train_positions[tp..][0..nt], .heldout_positions = result.heldout_positions[hp..][0..nh] };
        to = try std.math.add(usize, to, ne);
        ho = try std.math.add(usize, ho, he);
        tp = try std.math.add(usize, tp, nt);
        hp = try std.math.add(usize, hp, nh);
    }
    if (to != result.train.len or ho != result.heldout.len or tp != result.train_positions.len or hp != result.heldout_positions.len)
        return error.InvalidWholeFitInput;
    result.layout = .{ .query_heads_by_layer = result.query_heads_by_layer, .scales_by_donor = result.scales_softcaps[0..m.layout_scale_count], .softcaps_by_donor = result.scales_softcaps[m.layout_scale_count..] };
    result.fit = m.fit;
    result.method = if (std.mem.eql(u8, m.method, "top1")) .top1 else if (std.mem.eql(u8, m.method, "top2")) .top2 else return error.InvalidWholeFitInput;
    result.query_samples = m.query_samples;
    result.source_has_metadata = m.source_has_metadata;
    result.original_global_device_bytes = m.original_global_device_bytes;
    try engine.validateQueries(result.checkpoint_owned.state, result.batches, result.layout, .{ .max_queries_per_consumer = 512 });
    if (result.original_global_device_bytes != try engine.globalStorageBytes(result.checkpoint_owned.state, .f16, result.source_has_metadata))
        return error.InvalidWholeFitInput;
    return result;
}

/// Reload only the initially authenticated checkpoint payload. The caller
/// freezes byte count and digest from `OwnedInput`; no manifest or query
/// payload is parsed or allocated on this path.
pub fn reloadCheckpointOnly(
    allocator: std.mem.Allocator,
    io: std.Io,
    expected: Binding,
    payload_bytes: usize,
    payload_sha256: [32]u8,
    maximum_state_live_bytes: usize,
    control: Control,
) !CheckpointOnly {
    if (payload_bytes == 0 or payload_bytes > expected.max_bytes or maximum_state_live_bytes == 0 or std.mem.allEqual(u8, &payload_sha256, 0))
        return error.InvalidWholeFitInputBinding;
    var dir = try std.Io.Dir.openDirAbsolute(io, expected.output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    try control.check();
    var file = try dir.openFile(io, "original.checkpoint.f16", .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    var file_reader: FileReader = .{ .file = &file, .io = io };
    var reader: HashingReader(FileReader) = .{ .inner = &file_reader, .remaining = payload_bytes };
    var limits = expected.limits;
    limits.max_state_bytes = @min(limits.max_state_bytes, maximum_state_live_bytes);
    var decoded = try checkpoint.decodeReaderWithEncodingAndControl(allocator, &reader, payload_bytes, expected.identity, limits, control);
    errdefer decoded.deinit();
    if (decoded.encoding != .f16) return error.InvalidWholeFitInput;
    var extra: [1]u8 = undefined;
    const consumed_digest = reader.hash.finalResult();
    if (reader.remaining != 0 or try file.readPositionalAll(io, &extra, payload_bytes) != 0 or
        !std.mem.eql(u8, &consumed_digest, &payload_sha256))
        return error.InvalidWholeFitInputBinding;
    const peak = try decoded.checkpoint.state.residentBytes();
    if (peak > maximum_state_live_bytes) return error.KvCheckpointLimit;
    return .{ .owned = decoded.checkpoint, .payload_sha256 = payload_sha256, .peak_live_bytes = peak };
}

fn headDim(state: checkpoint.State, consumer_layer: u32, query_head: u32) !usize {
    if (consumer_layer >= state.layer_donors.len) return error.InvalidWholeFitInput;
    const donor_index = state.layer_donors[consumer_layer];
    if (donor_index >= state.donors.len) return error.InvalidWholeFitInput;
    const donor = state.donors[donor_index];
    if (donor.heads.len == 0 or query_head >= 256) return error.InvalidWholeFitInput;
    return donor.heads[0].dim;
}

fn findPayload(rows: []const Payload, name: []const u8) !Payload {
    var found: ?Payload = null;
    for (rows) |row| if (std.mem.eql(u8, row.name, name)) {
        if (found != null) return error.InvalidWholeFitInput;
        found = row;
    };
    return found orelse error.InvalidWholeFitInput;
}

fn readTyped(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, rows: []const Payload, name: []const u8) ![]T {
    const row = try findPayload(rows, name);
    if (row.bytes != try std.math.mul(usize, row.count, @sizeOf(T))) return error.InvalidWholeFitInput;
    const values = try allocator.alloc(T, row.count);
    errdefer allocator.free(values);
    var file = try dir.openFile(io, name, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    const bytes = std.mem.sliceAsBytes(values);
    if (try file.readPositionalAll(io, bytes, 0) != bytes.len) return error.InvalidWholeFitInput;
    var extra: [1]u8 = undefined;
    if (try file.readPositionalAll(io, &extra, bytes.len) != 0) return error.InvalidWholeFitInput;
    var digest: [32]u8 = undefined;
    Sha256.hash(bytes, &digest, .{});
    if (!std.mem.eql(u8, &digest, &row.sha256)) return error.InvalidWholeFitInput;
    return values;
}

fn verifyFile(io: std.Io, dir: std.Io.Dir, row: Payload) !void {
    var file = try dir.openFile(io, row.name, .{ .mode = .read_only, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    var hash = Sha256.init(.{});
    var offset: usize = 0;
    var buffer: [16384]u8 = undefined;
    while (offset < row.bytes) {
        const take = @min(buffer.len, row.bytes - offset);
        const got = try file.readPositionalAll(io, buffer[0..take], offset);
        if (got != take) return error.InvalidWholeFitInput;
        hash.update(buffer[0..take]);
        offset += take;
    }
    var extra: [1]u8 = undefined;
    const digest = hash.finalResult();
    if (try file.readPositionalAll(io, &extra, row.bytes) != 0 or !std.mem.eql(u8, &digest, &row.sha256)) return error.InvalidWholeFitInput;
}

const FileReader = struct {
    file: *std.Io.File,
    io: std.Io,
    offset: u64 = 0,
    pub fn read(self: *@This(), destination: []u8) !usize {
        const got = try self.file.readPositionalAll(self.io, destination, self.offset);
        self.offset += got;
        return got;
    }
};

fn HashingReader(comptime Reader: type) type {
    return struct {
        inner: *Reader,
        remaining: usize,
        hash: Sha256 = Sha256.init(.{}),
        pub fn read(self: *@This(), destination: []u8) !usize {
            if (self.remaining == 0) return 0;
            const got = try self.inner.read(destination[0..@min(destination.len, self.remaining)]);
            if (got > self.remaining) return error.InvalidWholeFitInput;
            self.hash.update(destination[0..got]);
            self.remaining -= got;
            return got;
        }
    };
}

test "whole fit checkpoint hashing reader binds consumed bytes" {
    const SliceReader = struct {
        bytes: []const u8,
        offset: usize = 0,
        fn read(self: *@This(), destination: []u8) !usize {
            const take = @min(destination.len, self.bytes.len - self.offset);
            @memcpy(destination[0..take], self.bytes[self.offset..][0..take]);
            self.offset += take;
            return take;
        }
    };
    var source: SliceReader = .{ .bytes = "payload-a" };
    var reader: HashingReader(SliceReader) = .{ .inner = &source, .remaining = source.bytes.len };
    var out: [9]u8 = undefined;
    try std.testing.expectEqual(@as(usize, out.len), try reader.read(&out));
    try std.testing.expectEqualSlices(u8, "payload-a", &out);
    var expected: [32]u8 = undefined;
    Sha256.hash("payload-a", &expected, .{});
    const actual = reader.hash.finalResult();
    try std.testing.expectEqualSlices(u8, &expected, &actual);
    var substituted: [32]u8 = undefined;
    Sha256.hash("payload-b", &substituted, .{});
    try std.testing.expect(!std.mem.eql(u8, &substituted, &actual));
}

test "whole fit input checked cap and overflow helpers" {
    try std.testing.expectEqual(@as(usize, 24), try addTyped(8, 2, 8));
    try std.testing.expectError(error.Overflow, addTyped(std.math.maxInt(usize), 1, 1));
    try std.testing.expectError(error.Overflow, addTyped(0, std.math.maxInt(usize), 2));
}

test "whole fit input payload closure rejects duplicate and missing names" {
    var rows: [payload_names.len]Payload = undefined;
    for (&rows, payload_names) |*row, name| row.* = .{ .name = name, .bytes = 1, .count = 1, .sha256 = @splat(0) };
    try validatePayloadClosure(&rows, manifest_reservation + rows.len);
    rows[1].name = rows[0].name;
    try std.testing.expectError(error.InvalidWholeFitInput, validatePayloadClosure(&rows, manifest_reservation + rows.len));
}

test "whole fit input SHA is content sensitive" {
    var left: [32]u8 = undefined;
    var right: [32]u8 = undefined;
    Sha256.hash("whole-cache", &left, .{});
    Sha256.hash("whole-cachf", &right, .{});
    try std.testing.expect(!std.mem.eql(u8, &left, &right));
}

test "whole fit input filesystem round trip and bad payload unwind" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const outer_len = try tmp.dir.realPath(io, &path_buffer);
    const output_dir = try std.fs.path.join(a, &.{ path_buffer[0..outer_len], "bundle" });
    defer a.free(output_dir);
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const head: checkpoint.Head = .{ .dim = 1, .compacted_until = 2, .positions = &.{0}, .coverage_ends = &.{2}, .keys = &.{0}, .values = &.{1}, .bias = &.{0.6931472} };
    const state: checkpoint.State = .{ .identity = identity, .logical_length = 2, .tokens = &.{ 0, 1 }, .logits = &.{ 0.5, -0.5 }, .layer_donors = &.{0}, .donors = &.{.{ .layer = 0, .heads = &.{head} }} };
    var train: [240]f32 = undefined;
    var heldout: [80]f32 = undefined;
    var train_positions: [240]u64 = undefined;
    var heldout_positions: [80]u64 = undefined;
    for (&train, &train_positions, 0..) |*value, *position, i| {
        value.* = @floatFromInt(i);
        position.* = @intCast(i + 2);
    }
    for (&heldout, &heldout_positions, 0..) |*value, *position, i| {
        value.* = @floatFromInt(i + 1);
        position.* = @intCast(i + 242);
    }
    const batches = [_]engine.ReferenceBatch{.{ .consumer_layer = 0, .query_head = 0, .train = &train, .heldout = &heldout, .train_positions = &train_positions, .heldout_positions = &heldout_positions }};
    const fit: FitConfig = .{ .prefix_rows = 128, .recent_rows = 2048, .chunk_rows = 1024, .retention_divisor = 4, .min_retention_divisor = 2, .max_relative_mass_error = 0.05, .max_relative_output_rmse = 0.05, .max_sweeps = 32, .nnls_tolerance = 1e-8, .min_weight = 0, .selection = .residual_omp, .mass_solver = .active_set, .validation = .whole_head, .rejected_head_policy = .preserve_original, .storage_precision = .f16, .independent_fp64_publication = true };
    const config_json = try std.json.Stringify.valueAlloc(a, .{ .schema = "antfly.attention-match-whole-fit-config.v1", .fit = fit, .query_samples = @as(u16, 320), .method = "top1" }, .{});
    defer a.free(config_json);
    var config_sha: [32]u8 = undefined;
    Sha256.hash(config_json, &config_sha, .{});
    const binding: Binding = .{ .output_dir = output_dir, .identity = identity, .config_sha256 = config_sha, .source_freeze_sha256 = @splat(5), .binary_sha256 = @splat(6), .exporter_sha256 = @splat(7), .limits = .{} };
    var writer = try Writer.init(a, io, binding);
    _ = try writer.write(.{ .source = state, .batches = &batches, .layout = .{ .query_heads_by_layer = &.{1}, .scales_by_donor = &.{1}, .softcaps_by_donor = &.{0} }, .query_samples = 320, .method = .top1, .fit = fit, .source_has_metadata = false, .original_global_device_bytes = try engine.globalStorageBytes(state, .f16, false), .control = .{} });
    var restored = try read(a, io, binding);
    try std.testing.expectEqual(@as(usize, 1), restored.batches.len);
    try std.testing.expectEqualSlices(f32, &train, restored.batches[0].train);
    restored.deinit();
    var dir = try std.Io.Dir.openDirAbsolute(io, output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    var file = try dir.openFile(io, "queries.heldout.f32", .{ .mode = .read_write, .allow_directory = false, .follow_symlinks = false });
    defer file.close(io);
    try file.writePositionalAll(io, &.{0xff}, 0);
    try std.testing.expectError(error.InvalidWholeFitInput, read(a, io, binding));
    try file.writePositionalAll(io, std.mem.asBytes(&heldout[0]), 0);
    try file.sync(io);
    var counts_file = try dir.openFile(io, "queries.train_counts.u32", .{ .mode = .read_write, .allow_directory = false, .follow_symlinks = false });
    defer counts_file.close(io);
    const malformed_count: u32 = 241;
    try counts_file.writePositionalAll(io, std.mem.asBytes(&malformed_count), 0);
    try counts_file.sync(io);
    var malformed_sha: [32]u8 = undefined;
    Sha256.hash(std.mem.asBytes(&malformed_count), &malformed_sha, .{});
    const old_manifest = try dir.readFileAlloc(io, "manifest.json", a, .limited(manifest_reservation));
    defer a.free(old_manifest);
    const parsed = try std.json.parseFromSlice(Manifest, a, old_manifest, .{ .ignore_unknown_fields = false });
    defer parsed.deinit();
    for (@constCast(parsed.value.payloads)) |*payload| {
        if (std.mem.eql(u8, payload.name, "queries.train_counts.u32")) payload.sha256 = malformed_sha;
    }
    const malformed_manifest = try std.json.Stringify.valueAlloc(a, parsed.value, .{});
    defer a.free(malformed_manifest);
    var manifest_file = try dir.createFile(io, "manifest.json", .{ .truncate = true });
    defer manifest_file.close(io);
    try manifest_file.writeStreamingAll(io, malformed_manifest);
    try manifest_file.writeStreamingAll(io, "\n");
    try manifest_file.sync(io);
    try std.testing.expectError(error.InvalidWholeFitInput, read(a, io, binding));
}
