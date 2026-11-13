//! Sealed, content-independent inventory for guarded E2 formal-fit sources.
//! The digest is useful only with `verifyState`: a caller supplied digest alone
//! is never authority.
const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const shape = @import("attention_match_e2_flat_shape_authority.zig");

pub const schema = "antfly.e2.formal-owner-inventory.v1";

pub const Context = struct {
    source_freeze_sha256: [32]u8,
    base_configuration_sha256: [32]u8,
    shape_sha256: [32]u8,
    /// Binary proven by the parent full-CLI/context-release receipt.
    binary_sha256: [32]u8,
    /// Distinct executable running the capture-only context test.
    capture_binary_sha256: [32]u8,
    hardware_sha256: [32]u8,
    owner_identity: u64,
    owner_logical_bytes: u64,
    owner_physical_bytes: u64,
};

pub const CaptureBinding = struct {
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    capture_binary_sha256: [32]u8,
    hardware_sha256: [32]u8,
    owner_identity: u64,
    owner_logical_bytes: u64,
    owner_physical_bytes: u64,
};

pub const RawEntry = struct { layer: u32, window: u32, rows: u32, capacity_rows: u32, kv_heads: u32, head_dim: u32, key_row_bytes: u64, value_row_bytes: u64, key_capacity_bytes: u64, value_capacity_bytes: u64, installed_physical_bytes: u64 };
pub const RawInventory = struct { context_instance_id: u64, sequence_id: u64, source_revision: u64, logical_tokens: u32, entries: []const RawEntry, installed_physical_bytes: u64, descriptor_sha256: [32]u8 };

pub const Entry = struct {
    donor_layer: u32,
    head_index: u16,
    role: enum(u8) { global },
    rows: u32,
    dim: u32,
    compacted_until: u64,
    exact_suffix_rows: u32,
    position_bytes: u64,
    coverage_bytes: u64,
    key_bytes: u64,
    value_bytes: u64,
    bias_bytes: u64,
    packed_bytes: u64,
};

pub const Artifact = struct {
    context: Context,
    native_identity: checkpoint.Identity,
    logical_length: u64,
    source_rows: u32,
    range_count: u16,
    candidate_count: u16,
    range_geometry_sha256: [32]u8,
    raw_context_instance_id: u64,
    raw_sequence_id: u64,
    raw_source_revision: u64,
    raw_total_physical_bytes: u64,
    raw_descriptor_sha256: [32]u8,
    raw_storage_profile_sha256: [32]u8,
    raw_entries: [shape.canonical_heads.len]RawEntry,
    entries: [shape.canonical_heads.len]Entry,
    physical_bytes: u64,
    digest: [32]u8,

    pub fn validate(self: @This()) !void {
        if (self.source_rows != 7_936 and self.source_rows != 32_512) return error.UnqualifiedCompactionArtifact;
        inline for (.{ self.context.source_freeze_sha256, self.context.base_configuration_sha256, self.context.shape_sha256, self.context.binary_sha256, self.context.capture_binary_sha256, self.context.hardware_sha256 }) |value|
            if (std.mem.allEqual(u8, &value, 0)) return error.UnqualifiedCompactionArtifact;
        if (self.context.owner_identity == 0 or self.context.owner_logical_bytes == 0 or
            self.context.owner_physical_bytes == 0 or self.context.owner_physical_bytes < self.physical_bytes)
            return error.UnqualifiedCompactionArtifact;
        const expected_ranges: u16 = if (self.source_rows == 7_936) 23 else 119;
        if (self.range_count != expected_ranges or self.candidate_count != expected_ranges * 3) return error.UnqualifiedCompactionArtifact;
        if (self.raw_context_instance_id == 0 or self.raw_sequence_id == 0 or self.raw_source_revision == 0 or
            self.raw_total_physical_bytes < self.physical_bytes or std.mem.allEqual(u8, &self.raw_descriptor_sha256, 0) or
            std.mem.allEqual(u8, &self.raw_storage_profile_sha256, 0))
            return error.UnqualifiedCompactionArtifact;
        if (!std.mem.eql(u8, &self.raw_storage_profile_sha256, &rawStorageProfile(self.raw_entries)))
            return error.UnqualifiedCompactionArtifact;
        var raw_total: u64 = 0;
        for (self.raw_entries, shape.canonical_heads) |entry, canonical| {
            if (entry.layer != canonical.donor_layer or entry.window != 0 or entry.rows != self.source_rows or
                entry.capacity_rows < entry.rows or entry.kv_heads != 1 or entry.head_dim != 512 or
                entry.installed_physical_bytes < entry.key_capacity_bytes + entry.value_capacity_bytes)
                return error.UnqualifiedCompactionArtifact;
            raw_total = add(raw_total, entry.installed_physical_bytes) catch return error.UnqualifiedCompactionArtifact;
        }
        if (raw_total != self.physical_bytes or self.raw_total_physical_bytes < raw_total)
            return error.UnqualifiedCompactionArtifact;
        const expected_geometry = rangeGeometryDigest(self.source_rows);
        if (!std.mem.eql(u8, &self.range_geometry_sha256, &expected_geometry)) return error.UnqualifiedCompactionArtifact;
        var total: u64 = 0;
        for (self.entries, shape.canonical_heads) |entry, canonical| {
            if (entry.donor_layer != canonical.donor_layer or entry.head_index != @as(u16, @intCast(canonical.head_index)) or
                entry.role != .global or entry.rows != self.source_rows or entry.dim != 512 or
                entry.compacted_until > self.logical_length or self.logical_length - entry.compacted_until != @as(u64, entry.exact_suffix_rows))
                return error.UnqualifiedCompactionArtifact;
            inline for (.{ entry.position_bytes, entry.coverage_bytes, entry.key_bytes, entry.value_bytes, entry.bias_bytes, entry.packed_bytes }) |part|
                total = add(total, part) catch return error.UnqualifiedCompactionArtifact;
        }
        if (total != self.physical_bytes) return error.UnqualifiedCompactionArtifact;
        const expected_digest = digestArtifact(self);
        if (!std.mem.eql(u8, &self.digest, &expected_digest)) return error.UnqualifiedCompactionArtifact;
    }

    pub fn validateExpected(self: @This(), expected: Context, expected_rows: usize) !void {
        try self.validate();
        if (!std.meta.eql(self.context, expected) or @as(usize, self.source_rows) != expected_rows)
            return error.UnqualifiedCompactionArtifact;
    }
};

/// Exclusive diagnostic writer. The artifact has already been validated and
/// is capture authority only; this file does not activate a fit profile.
pub fn writeCapturedArtifact(
    allocator: std.mem.Allocator,
    io: std.Io,
    directory: std.Io.Dir,
    name: []const u8,
    artifact: Artifact,
) !void {
    try artifact.validate();
    if (name.len == 0 or std.fs.path.isAbsolute(name) or std.mem.indexOf(u8, name, "..") != null)
        return error.InvalidFormalOwnerInventoryPath;
    const body = try std.json.Stringify.valueAlloc(allocator, .{
        .schema = "antfly.e2.formal-owner-inventory.capture.v1",
        .capture_only = true,
        .fit_authority = false,
        .release_authority = false,
        .artifact = artifact,
        .final_configuration_sha256 = std.fmt.bytesToHex(finalConfigurationDigest(artifact), .lower),
    }, .{ .whitespace = .indent_2 });
    defer allocator.free(body);
    var file = try directory.createFile(io, name, .{ .exclusive = true, .truncate = false });
    defer file.close(io);
    try file.writeStreamingAll(io, body);
    try file.writeStreamingAll(io, "\n");
    try file.sync(io);
}

fn add(a: u64, b: u64) !u64 {
    return std.math.add(u64, a, b) catch error.UnqualifiedCompactionArtifact;
}

fn bytes(count: usize, width: usize) !u64 {
    return std.math.mul(u64, @intCast(count), @intCast(width)) catch error.UnqualifiedCompactionArtifact;
}

fn digestArtifact(value: Artifact) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(schema);
    hash.update(&value.context.source_freeze_sha256);
    hash.update(&value.context.base_configuration_sha256);
    hash.update(&value.context.shape_sha256);
    hash.update(&value.context.binary_sha256);
    hash.update(&value.context.capture_binary_sha256);
    hash.update(&value.context.hardware_sha256);
    hashInt(&hash, value.context.owner_identity);
    hashInt(&hash, value.context.owner_logical_bytes);
    hashInt(&hash, value.context.owner_physical_bytes);
    hash.update(&value.native_identity.weights);
    hash.update(&value.native_identity.tokenizer);
    hash.update(&value.native_identity.template);
    hash.update(&value.native_identity.layout);
    hashInt(&hash, value.logical_length);
    hashInt(&hash, value.source_rows);
    hashInt(&hash, value.range_count);
    hashInt(&hash, value.candidate_count);
    hash.update(&value.range_geometry_sha256);
    hashInt(&hash, value.raw_context_instance_id);
    hashInt(&hash, value.raw_sequence_id);
    hashInt(&hash, value.raw_source_revision);
    hashInt(&hash, value.raw_total_physical_bytes);
    hash.update(&value.raw_descriptor_sha256);
    hash.update(&value.raw_storage_profile_sha256);
    for (value.raw_entries) |entry| hashRawEntry(&hash, entry);
    for (value.entries) |entry| {
        hashInt(&hash, entry.donor_layer);
        hashInt(&hash, entry.head_index);
        hashInt(&hash, @backingInt(entry.role));
        hashInt(&hash, entry.rows);
        hashInt(&hash, entry.dim);
        hashInt(&hash, entry.compacted_until);
        hashInt(&hash, entry.exact_suffix_rows);
        hashInt(&hash, entry.position_bytes);
        hashInt(&hash, entry.coverage_bytes);
        hashInt(&hash, entry.key_bytes);
        hashInt(&hash, entry.value_bytes);
        hashInt(&hash, entry.bias_bytes);
        hashInt(&hash, entry.packed_bytes);
    }
    hashInt(&hash, value.physical_bytes);
    return hash.finalResult();
}

fn hashInt(hash: *std.crypto.hash.sha2.Sha256, value: anytype) void {
    var frame: [8]u8 = undefined;
    std.mem.writeInt(u64, &frame, @intCast(value), .little);
    hash.update(&frame);
}

fn hashRawEntry(hash: *std.crypto.hash.sha2.Sha256, entry: RawEntry) void {
    inline for (@typeInfo(RawEntry).@"struct".field_names) |name| hashInt(hash, @field(entry, name));
}

fn rawStorageProfile(entries: [shape.canonical_heads.len]RawEntry) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.e2.raw-prefill-storage-profile.v1");
    for (entries) |entry| hashRawEntry(&hash, entry);
    return hash.finalResult();
}

fn canonicalRawEntries(raw: RawInventory, source_rows: usize) ![shape.canonical_heads.len]RawEntry {
    var result: [shape.canonical_heads.len]RawEntry = undefined;
    for (shape.canonical_heads, &result) |wanted, *out| {
        var found: ?RawEntry = null;
        for (raw.entries) |entry| if (entry.layer == wanted.donor_layer) {
            if (found != null) return error.UnqualifiedCompactionArtifact;
            found = entry;
        };
        const entry = found orelse return error.UnqualifiedCompactionArtifact;
        if (entry.window != 0 or entry.rows != source_rows or entry.kv_heads != 1 or entry.head_dim != 512 or
            entry.capacity_rows < entry.rows or entry.installed_physical_bytes < entry.key_capacity_bytes + entry.value_capacity_bytes)
            return error.UnqualifiedCompactionArtifact;
        out.* = entry;
    }
    return result;
}

fn rangeGeometryDigest(source_rows: u32) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.e2.formal-owner-ranges.v1");
    hashInt(&hash, source_rows);
    var start: usize = 128;
    const end: usize = @as(usize, source_rows) - 2_048;
    while (start < end) {
        const rows = @min(@as(usize, 256), end - start);
        hashInt(&hash, start);
        hashInt(&hash, rows);
        inline for (.{ 4, 8, 16 }) |divisor| hashInt(&hash, divisor);
        start += rows;
    }
    return hash.finalResult();
}

/// Constrained diagnostic capture. It inventories only the three authenticated
/// global E2 owners and never hashes K/V content or instance pointers.
fn captureWithContext(state: checkpoint.State, context: Context, source_rows: usize) !Artifact {
    if (source_rows != 7_936 and source_rows != 32_512) return error.UnqualifiedCompactionArtifact;
    inline for (.{ context.source_freeze_sha256, context.base_configuration_sha256, context.shape_sha256, context.binary_sha256, context.capture_binary_sha256, context.hardware_sha256 }) |value|
        if (std.mem.allEqual(u8, &value, 0)) return error.UnqualifiedCompactionArtifact;
    if (context.owner_identity == 0 or context.owner_logical_bytes == 0 or context.owner_physical_bytes == 0)
        return error.UnqualifiedCompactionArtifact;
    const validated = try shape.validate(.{ .source_rows = source_rows, .authority_sha256 = context.source_freeze_sha256 });
    if (!std.mem.eql(u8, &validated.digest, &context.shape_sha256)) return error.UnqualifiedCompactionArtifact;
    var entries: [shape.canonical_heads.len]Entry = undefined;
    var total: u64 = 0;
    for (shape.canonical_heads, &entries) |wanted, *entry| {
        var found: ?checkpoint.Head = null;
        for (state.donors) |donor| if (donor.layer == wanted.donor_layer) {
            if (donor.window != 0 or wanted.head_index >= donor.heads.len) return error.UnqualifiedCompactionArtifact;
            found = donor.heads[wanted.head_index];
        };
        const head = found orelse return error.UnqualifiedCompactionArtifact;
        const rows = head.positions.len;
        if (rows != source_rows or head.dim != 512 or head.compacted_until > state.logical_length) return error.UnqualifiedCompactionArtifact;
        const exact_suffix = state.logical_length - head.compacted_until;
        if (exact_suffix > rows) return error.UnqualifiedCompactionArtifact;
        entry.* = .{
            .donor_layer = wanted.donor_layer,
            .head_index = @intCast(wanted.head_index),
            .role = .global,
            .rows = @intCast(rows),
            .dim = head.dim,
            .compacted_until = head.compacted_until,
            .exact_suffix_rows = @intCast(exact_suffix),
            .position_bytes = try bytes(head.positions.len, @sizeOf(u64)),
            .coverage_bytes = try bytes(head.coverage_ends.len, @sizeOf(u64)),
            .key_bytes = try bytes(head.keys.len, @sizeOf(f32)),
            .value_bytes = try bytes(head.values.len, @sizeOf(f32)),
            .bias_bytes = try bytes(head.bias.len, @sizeOf(f32)),
            .packed_bytes = @intCast(head.compressed.bytes()),
        };
        inline for (.{ entry.position_bytes, entry.coverage_bytes, entry.key_bytes, entry.value_bytes, entry.bias_bytes, entry.packed_bytes }) |part| total = try add(total, part);
    }
    var result: Artifact = .{
        .context = context,
        .native_identity = state.identity,
        .logical_length = state.logical_length,
        .source_rows = @intCast(source_rows),
        .range_count = @intCast(validated.range_count),
        .candidate_count = @intCast(validated.candidate_count),
        .range_geometry_sha256 = rangeGeometryDigest(@intCast(source_rows)),
        .raw_context_instance_id = 0,
        .raw_sequence_id = 0,
        .raw_source_revision = 0,
        .raw_total_physical_bytes = 0,
        .raw_descriptor_sha256 = @splat(0),
        .raw_storage_profile_sha256 = @splat(0),
        .raw_entries = @splat(std.mem.zeroes(RawEntry)),
        .entries = entries,
        .physical_bytes = total,
        .digest = undefined,
    };
    result.digest = digestArtifact(result);
    if (context.owner_physical_bytes < result.physical_bytes) return error.UnqualifiedCompactionArtifact;
    return result;
}

pub fn baseConfigurationDigest(source_rows: usize, source_freeze_sha256: [32]u8, shape_sha256: [32]u8) ![32]u8 {
    if (source_rows != 7_936 and source_rows != 32_512) return error.UnqualifiedCompactionArtifact;
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.e2.formal-fit.base-configuration.v1");
    hashInt(&hash, source_rows);
    hash.update(&source_freeze_sha256);
    hash.update(&shape_sha256);
    hash.update("dim512|batches56|owners8,8,40|chunk256|primary60|fallback0|workspace2GiB|activation=false");
    return hash.finalResult();
}

pub fn finalConfigurationDigest(artifact: Artifact) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update("antfly.e2.formal-fit.final-configuration.v1");
    hash.update(&artifact.context.base_configuration_sha256);
    hash.update(&artifact.digest);
    return hash.finalResult();
}

pub fn captureBound(state: checkpoint.State, binding: CaptureBinding, source_rows: usize) !Artifact {
    const validated = try shape.validate(.{ .source_rows = source_rows, .authority_sha256 = binding.source_freeze_sha256 });
    const context: Context = .{
        .source_freeze_sha256 = binding.source_freeze_sha256,
        .base_configuration_sha256 = try baseConfigurationDigest(source_rows, binding.source_freeze_sha256, validated.digest),
        .shape_sha256 = validated.digest,
        .binary_sha256 = binding.binary_sha256,
        .capture_binary_sha256 = binding.capture_binary_sha256,
        .hardware_sha256 = binding.hardware_sha256,
        .owner_identity = binding.owner_identity,
        .owner_logical_bytes = binding.owner_logical_bytes,
        .owner_physical_bytes = binding.owner_physical_bytes,
    };
    return captureWithContext(state, context, source_rows);
}

pub fn captureRaw(identity: checkpoint.Identity, raw: RawInventory, binding: CaptureBinding, source_rows: usize) !Artifact {
    if (raw.context_instance_id == 0 or raw.sequence_id == 0 or raw.source_revision == 0 or raw.logical_tokens != source_rows or
        raw.entries.len == 0 or raw.installed_physical_bytes == 0 or std.mem.allEqual(u8, &raw.descriptor_sha256, 0))
        return error.UnqualifiedCompactionArtifact;
    const validated = try shape.validate(.{ .source_rows = source_rows, .authority_sha256 = binding.source_freeze_sha256 });
    const raw_entries = try canonicalRawEntries(raw, source_rows);
    var entries: [shape.canonical_heads.len]Entry = undefined;
    var canonical_physical: u64 = 0;
    for (shape.canonical_heads, raw_entries, &entries) |wanted, entry, *out| {
        out.* = .{ .donor_layer = wanted.donor_layer, .head_index = @intCast(wanted.head_index), .role = .global, .rows = entry.rows, .dim = entry.head_dim, .compacted_until = 0, .exact_suffix_rows = entry.rows, .position_bytes = 0, .coverage_bytes = 0, .key_bytes = entry.key_capacity_bytes, .value_bytes = entry.value_capacity_bytes, .bias_bytes = 0, .packed_bytes = entry.installed_physical_bytes - entry.key_capacity_bytes - entry.value_capacity_bytes };
        canonical_physical = try add(canonical_physical, entry.installed_physical_bytes);
    }
    var artifact: Artifact = .{ .context = .{ .source_freeze_sha256 = binding.source_freeze_sha256, .base_configuration_sha256 = try baseConfigurationDigest(source_rows, binding.source_freeze_sha256, validated.digest), .shape_sha256 = validated.digest, .binary_sha256 = binding.binary_sha256, .capture_binary_sha256 = binding.capture_binary_sha256, .hardware_sha256 = binding.hardware_sha256, .owner_identity = binding.owner_identity, .owner_logical_bytes = binding.owner_logical_bytes, .owner_physical_bytes = binding.owner_physical_bytes }, .native_identity = identity, .logical_length = raw.logical_tokens, .source_rows = @intCast(source_rows), .range_count = @intCast(validated.range_count), .candidate_count = @intCast(validated.candidate_count), .range_geometry_sha256 = rangeGeometryDigest(@intCast(source_rows)), .raw_context_instance_id = raw.context_instance_id, .raw_sequence_id = raw.sequence_id, .raw_source_revision = raw.source_revision, .raw_total_physical_bytes = raw.installed_physical_bytes, .raw_descriptor_sha256 = raw.descriptor_sha256, .raw_storage_profile_sha256 = rawStorageProfile(raw_entries), .raw_entries = raw_entries, .entries = entries, .physical_bytes = canonical_physical, .digest = undefined };
    artifact.digest = digestArtifact(artifact);
    try artifact.validate();
    return artifact;
}

pub fn verifyState(artifact: Artifact, state: checkpoint.State, current_raw: RawInventory, expected: Context, expected_rows: usize) ![32]u8 {
    try artifact.validateExpected(expected, expected_rows);
    if (current_raw.context_instance_id == 0 or current_raw.sequence_id == 0 or current_raw.source_revision == 0 or
        current_raw.logical_tokens != expected_rows or std.mem.allEqual(u8, &current_raw.descriptor_sha256, 0))
        return error.UnqualifiedCompactionArtifact;
    const current_entries = try canonicalRawEntries(current_raw, expected_rows);
    if (!std.mem.eql(u8, &rawStorageProfile(current_entries), &artifact.raw_storage_profile_sha256))
        return error.UnqualifiedCompactionArtifact;
    if (!checkpoint.Identity.eql(artifact.native_identity, state.identity) or state.logical_length != expected_rows)
        return error.UnqualifiedCompactionArtifact;
    for (shape.canonical_heads) |wanted| {
        var found: ?checkpoint.Head = null;
        for (state.donors) |donor| if (donor.layer == wanted.donor_layer) {
            if (donor.window != 0 or wanted.head_index >= donor.heads.len or found != null)
                return error.UnqualifiedCompactionArtifact;
            found = donor.heads[wanted.head_index];
        };
        const head = found orelse return error.UnqualifiedCompactionArtifact;
        if (head.positions.len != expected_rows or head.coverage_ends.len != expected_rows or head.dim != 512 or
            head.keys.len != expected_rows * 512 or head.values.len != expected_rows * 512 or
            (head.bias.len != 0 and head.bias.len != expected_rows))
            return error.UnqualifiedCompactionArtifact;
    }
    var receipt = std.crypto.hash.sha2.Sha256.init(.{});
    receipt.update("antfly.e2.raw-to-checkpoint-conversion.v1");
    receipt.update(&artifact.digest);
    hashInt(&receipt, current_raw.context_instance_id);
    hashInt(&receipt, current_raw.sequence_id);
    hashInt(&receipt, current_raw.source_revision);
    receipt.update(&current_raw.descriptor_sha256);
    receipt.update(&artifact.raw_storage_profile_sha256);
    receipt.update(&state.identity.weights);
    receipt.update(&state.identity.tokenizer);
    receipt.update(&state.identity.template);
    receipt.update(&state.identity.layout);
    return receipt.finalResult();
}

fn verifyCaptured(expected: Artifact, actual: Artifact) !void {
    try expected.validate();
    try actual.validate();
    if (!std.meta.eql(expected, actual)) return error.UnqualifiedCompactionArtifact;
}

test "formal inventory capture rejects stale identity rows shape and ranges" {
    // Full state construction is exercised by the bridge fixture. These early
    // failures prove no arbitrary nonzero digest can stand in for an artifact.
    var artifact: Artifact = undefined;
    artifact = std.mem.zeroes(Artifact);
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, artifact.validate());
    var context: Context = .{ .source_freeze_sha256 = @splat(1), .base_configuration_sha256 = @splat(2), .shape_sha256 = @splat(3), .binary_sha256 = @splat(4), .capture_binary_sha256 = @splat(6), .hardware_sha256 = @splat(5), .owner_identity = 1, .owner_logical_bytes = 1, .owner_physical_bytes = 1 };
    context.shape_sha256 = @splat(0);
    const empty: checkpoint.State = .{ .identity = std.mem.zeroes(checkpoint.Identity), .logical_length = 32_512 };
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, captureWithContext(empty, context, 32_512));
    context.shape_sha256 = @splat(3);
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, captureWithContext(empty, context, 8_128));
}

test "sealed formal inventory rejects stale context identity rows and range census" {
    const context: Context = .{ .source_freeze_sha256 = @splat(1), .base_configuration_sha256 = @splat(2), .shape_sha256 = @splat(3), .binary_sha256 = @splat(4), .capture_binary_sha256 = @splat(15), .hardware_sha256 = @splat(5), .owner_identity = 1, .owner_logical_bytes = 401_458_176, .owner_physical_bytes = 401_458_176 };
    var artifact: Artifact = .{
        .context = context,
        .native_identity = .{ .weights = @splat(6), .tokenizer = @splat(7), .template = @splat(8), .layout = @splat(9) },
        .logical_length = 32_512,
        .source_rows = 32_512,
        .range_count = 119,
        .candidate_count = 357,
        .range_geometry_sha256 = rangeGeometryDigest(32_512),
        .raw_context_instance_id = 11,
        .raw_sequence_id = 12,
        .raw_source_revision = 13,
        .raw_total_physical_bytes = 401_458_176,
        .raw_descriptor_sha256 = @splat(14),
        .raw_storage_profile_sha256 = undefined,
        .raw_entries = @splat(.{ .layer = 4, .window = 0, .rows = 32_512, .capacity_rows = 32_512, .kv_heads = 1, .head_dim = 512, .key_row_bytes = 1024, .value_row_bytes = 1024, .key_capacity_bytes = 66_584_576, .value_capacity_bytes = 66_584_576, .installed_physical_bytes = 133_819_392 }),
        .entries = @splat(.{ .donor_layer = 4, .head_index = 0, .role = .global, .rows = 32_512, .dim = 512, .compacted_until = 30_464, .exact_suffix_rows = 2_048, .position_bytes = 260_096, .coverage_bytes = 260_096, .key_bytes = 66_584_576, .value_bytes = 66_584_576, .bias_bytes = 130_048, .packed_bytes = 0 }),
        .physical_bytes = 401_458_176,
        .digest = undefined,
    };
    artifact.entries[1].donor_layer = 9;
    artifact.entries[2].donor_layer = 14;
    artifact.raw_entries[1].layer = 9;
    artifact.raw_entries[2].layer = 14;
    artifact.raw_storage_profile_sha256 = rawStorageProfile(artifact.raw_entries);
    artifact.digest = digestArtifact(artifact);
    try artifact.validateExpected(context, 32_512);
    const finalized = finalConfigurationDigest(artifact);
    try std.testing.expect(!std.mem.eql(u8, &finalized, &artifact.context.base_configuration_sha256));
    var stale = artifact;
    stale.context.source_freeze_sha256 = @splat(10);
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, stale.validateExpected(context, 32_512));
    stale = artifact;
    stale.native_identity.weights = @splat(10);
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, stale.validate());
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, artifact.validateExpected(context, 7_936));
    stale = artifact;
    stale.range_count = 118;
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, stale.validate());
    var actual = artifact;
    actual.native_identity.weights = @splat(10);
    actual.digest = digestArtifact(actual);
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, verifyCaptured(artifact, actual));
    actual = artifact;
    actual.source_rows = 7_936;
    actual.range_count = 23;
    actual.candidate_count = 69;
    actual.digest = digestArtifact(actual);
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, verifyCaptured(artifact, actual));
}

test "formal inventory base projection is cycle free and finalization binds artifact" {
    const source: [32]u8 = @splat(1);
    const shape_digest: [32]u8 = @splat(2);
    const first = try baseConfigurationDigest(32_512, source, shape_digest);
    const second = try baseConfigurationDigest(32_512, source, shape_digest);
    try std.testing.expectEqual(first, second);
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, baseConfigurationDigest(8_128, source, shape_digest));
}

test "raw issuer converts native f16 census to checkpoint semantics and binds fresh runtime revision" {
    const allocator = std.testing.allocator;
    const rows: usize = 7_936;
    const vector_count = rows * 512;
    const positions = try allocator.alloc(u64, rows);
    defer allocator.free(positions);
    const coverage = try allocator.alloc(u64, rows);
    defer allocator.free(coverage);
    const bias = try allocator.alloc(f32, rows);
    defer allocator.free(bias);
    const keys = try allocator.alloc(f32, vector_count);
    defer allocator.free(keys);
    const values = try allocator.alloc(f32, vector_count);
    defer allocator.free(values);
    const identity: checkpoint.Identity = .{ .weights = @splat(1), .tokenizer = @splat(2), .template = @splat(3), .layout = @splat(4) };
    const f16_vector_bytes: u64 = rows * 512 * @sizeOf(f16);
    const one_physical = f16_vector_bytes * 2;
    var raw_entries = [_]RawEntry{
        .{ .layer = 4, .window = 0, .rows = @intCast(rows), .capacity_rows = @intCast(rows), .kv_heads = 1, .head_dim = 512, .key_row_bytes = 1024, .value_row_bytes = 1024, .key_capacity_bytes = f16_vector_bytes, .value_capacity_bytes = f16_vector_bytes, .installed_physical_bytes = one_physical },
        .{ .layer = 9, .window = 0, .rows = @intCast(rows), .capacity_rows = @intCast(rows), .kv_heads = 1, .head_dim = 512, .key_row_bytes = 1024, .value_row_bytes = 1024, .key_capacity_bytes = f16_vector_bytes, .value_capacity_bytes = f16_vector_bytes, .installed_physical_bytes = one_physical },
        .{ .layer = 14, .window = 0, .rows = @intCast(rows), .capacity_rows = @intCast(rows), .kv_heads = 1, .head_dim = 512, .key_row_bytes = 1024, .value_row_bytes = 1024, .key_capacity_bytes = f16_vector_bytes, .value_capacity_bytes = f16_vector_bytes, .installed_physical_bytes = one_physical },
    };
    var raw: RawInventory = .{ .context_instance_id = 10, .sequence_id = 11, .source_revision = 12, .logical_tokens = @intCast(rows), .entries = &raw_entries, .installed_physical_bytes = one_physical * 3, .descriptor_sha256 = @splat(5) };
    const binding: CaptureBinding = .{ .source_freeze_sha256 = @splat(6), .binary_sha256 = @splat(7), .capture_binary_sha256 = @splat(8), .hardware_sha256 = @splat(9), .owner_identity = 13, .owner_logical_bytes = one_physical * 3, .owner_physical_bytes = one_physical * 3 };
    const artifact = try captureRaw(identity, raw, binding, rows);
    const head: checkpoint.Head = .{ .dim = 512, .positions = positions, .coverage_ends = coverage, .keys = keys, .values = values, .bias = bias };
    var heads = [_]checkpoint.Head{head};
    var donors = [_]checkpoint.Donor{ .{ .layer = 4, .heads = &heads }, .{ .layer = 9, .heads = &heads }, .{ .layer = 14, .heads = &heads } };
    const state: checkpoint.State = .{ .identity = identity, .logical_length = rows, .donors = &donors };
    const receipt = try verifyState(artifact, state, raw, artifact.context, rows);
    try std.testing.expect(!std.mem.allEqual(u8, &receipt, 0));
    raw.source_revision = 0;
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, verifyState(artifact, state, raw, artifact.context, rows));
    raw.source_revision = 12;
    raw_entries[0].key_capacity_bytes += 2;
    try std.testing.expectError(error.UnqualifiedCompactionArtifact, verifyState(artifact, state, raw, artifact.context, rows));
}
