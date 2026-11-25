// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
//! TRAIN-only, diagnostic snapshot of the exact cached source and frozen plan.
//! Heldout data is absent from this API by construction.
const std = @import("std");
const solver = @import("attention_match.zig");
const training = @import("attention_match_joint_training_source.zig");
const replay = @import("attention_match_joint_chunk_mass_replay.zig");

pub const schema = "antfly.attention-match-train-snapshot.v2";
pub const maximum_bytes: usize = 1024 * 1024 * 1024;

pub const Authority = struct {
    output_dir: []const u8,
    source_sha256: [32]u8,
    config_sha256: [32]u8,
    model_sha256: [32]u8,
    source_freeze_sha256: [32]u8,
    binary_sha256: [32]u8,
    producer_input_manifest_sha256: [32]u8,
    complete_future_mix_manifest_sha256: [32]u8,
    complete_future_mix_policy_sha256: [32]u8,
    consumer_source_manifest_sha256: [32]u8,
    consumer_binary_sha256: [32]u8,
};
pub const Result = struct { manifest_sha256: [32]u8, payload_bytes: usize };
pub const Domain = replay.TrainSnapshotDomain;
pub const DomainRun = replay.TrainSnapshotDomainRun;
const Payload = struct { name: []const u8, bytes: usize, sha256: [64]u8 };

fn add(a: usize, b: usize) !usize {
    return std.math.add(usize, a, b) catch error.TrainSnapshotLimit;
}
fn mul(a: usize, b: usize) !usize {
    return std.math.mul(usize, a, b) catch error.TrainSnapshotLimit;
}
fn validateAuthority(a: Authority) !void {
    if (!std.fs.path.isAbsolute(a.output_dir) or std.mem.indexOf(u8, a.output_dir, "/../") != null or std.mem.endsWith(u8, a.output_dir, "/.."))
        return error.InvalidTrainSnapshotPath;
    inline for (.{ a.source_sha256, a.config_sha256, a.model_sha256, a.source_freeze_sha256, a.binary_sha256, a.producer_input_manifest_sha256, a.complete_future_mix_manifest_sha256, a.complete_future_mix_policy_sha256, a.consumer_source_manifest_sha256, a.consumer_binary_sha256 }) |digest| {
        var nonzero = false;
        for (digest) |byte| nonzero = nonzero or byte != 0;
        if (!nonzero) return error.InvalidTrainSnapshotAuthority;
    }
}
const PayloadWriter = struct {
    file: *std.Io.File,
    io: std.Io,
    hash: std.crypto.hash.sha2.Sha256 = std.crypto.hash.sha2.Sha256.init(.{}),
    buffer: [64 * 1024]u8 = undefined,
    used: usize = 0,

    fn write(self: *PayloadWriter, bytes: []const u8) !void {
        self.hash.update(bytes);
        var remaining = bytes;
        while (remaining.len != 0) {
            if (self.used == self.buffer.len) try self.flush();
            const count = @min(remaining.len, self.buffer.len - self.used);
            @memcpy(self.buffer[self.used..][0..count], remaining[0..count]);
            self.used += count;
            remaining = remaining[count..];
        }
    }

    fn flush(self: *PayloadWriter) !void {
        if (self.used == 0) return;
        try self.file.writeStreamingAll(self.io, self.buffer[0..self.used]);
        self.used = 0;
    }
};

fn putU64(writer: *PayloadWriter, value: usize) !void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, @intCast(value), .little);
    try writer.write(&bytes);
}
fn putF64(writer: *PayloadWriter, value: f64) !void {
    if (!std.math.isFinite(value)) return error.NonFiniteAttention;
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, @bitCast(value), .little);
    try writer.write(&bytes);
}
fn putF32(writer: *PayloadWriter, value: f32) !void {
    if (!std.math.isFinite(value)) return error.NonFiniteAttention;
    var bytes: [4]u8 = undefined;
    std.mem.writeInt(u32, &bytes, @bitCast(value), .little);
    try writer.write(&bytes);
}
fn payload(name: []const u8, bytes: usize, h: *std.crypto.hash.sha2.Sha256) Payload {
    return .{ .name = name, .bytes = bytes, .sha256 = std.fmt.bytesToHex(h.finalResult(), .lower) };
}

pub fn requiredBytes(source: *const training.Source, plan: *const replay.AdaptivePlan) !usize {
    if (source.probabilities_f32.len != try mul(source.query_count, source.rows) or source.probabilities.len != 0 or source.outputs.len != try mul(source.query_count, source.dim) or plan.scores.len != plan.scored.len)
        return error.InvalidTrainSnapshotShape;
    var bytes = try add(try mul(source.probabilities_f32.len, 4), try mul(source.outputs.len, 8));
    bytes = try add(bytes, try mul(try add(try mul(source.rows, source.dim), source.rows), 8));
    // Header plus exact frozen support and rounded fit payload for every level.
    bytes = try add(bytes, 24);
    for (plan.scores, plan.scored) |score, levels| {
        bytes = try add(bytes, 16);
        for (levels) |candidate| {
            const owned = candidate orelse return error.InvalidTrainSnapshotShape;
            if (owned.frozen.chunks.len != 1 or owned.fitted.chunks.len != 1) return error.InvalidTrainSnapshotShape;
            const chunk = owned.frozen.chunks[0];
            const range = owned.frozen.ranges[0];
            const fit = owned.fitted.chunks[0];
            if (range.start != score.start or range.rows != score.rows or chunk.start != score.start or chunk.rows != score.rows or chunk.selected_local.len != chunk.weights.len or fit.selected_count != chunk.selected_local.len or owned.fitted.bias.len != fit.selected_count or owned.fitted.values.len != try mul(fit.selected_count, source.dim)) return error.InvalidTrainSnapshotShape;
            for (chunk.selected_local, chunk.weights, 0..) |local, weight, i| if (local >= chunk.rows or !std.math.isFinite(weight) or weight < 0 or std.mem.indexOfScalar(usize, chunk.selected_local[0..i], local) != null) return error.InvalidTrainSnapshotShape;
            for (owned.fitted.bias) |value| if (!std.math.isFinite(value) or @abs(value) > std.math.floatMax(f32)) return error.NonFiniteAttention;
            bytes = try add(bytes, 24);
            bytes = try add(bytes, try mul(chunk.selected_local.len, 8 + 8 + 4 + try mul(source.dim, 2)));
        }
    }
    for (plan.branches) |branch| {
        if (branch.targets.len != plan.scores.len) return error.InvalidTrainSnapshotShape;
        bytes = try add(bytes, try add(32, try mul(branch.targets.len, 8)));
    }
    return bytes;
}

pub fn write(allocator: std.mem.Allocator, io: std.Io, authority: Authority, domain_runs: []const DomainRun, original: solver.Head, source: *const training.Source, plan: *const replay.AdaptivePlan, control: solver.Control) !Result {
    try validateAuthority(authority);
    try control.check();
    if (original.rows != source.rows or original.dim != source.dim or original.keys.ptr != source.keys_ptr or original.values.ptr != source.values_ptr or original.bias == null or original.bias.?.len != original.rows) return error.InvalidTrainSnapshotShape;
    const required = try requiredBytes(source, plan);
    const manifest_reservation: usize = 64 * 1024;
    if (required > maximum_bytes - manifest_reservation) return error.TrainSnapshotLimit;
    var domain_at: usize = 0;
    var domain_counts: [3]usize = @splat(0);
    for (domain_runs) |run| {
        if (run.start != domain_at or run.count == 0) return error.InvalidTrainSnapshotShape;
        domain_at = try add(domain_at, run.count);
        domain_counts[@backingInt(run.domain)] = try add(domain_counts[@backingInt(run.domain)], run.count);
    }
    if (domain_at != source.query_count) return error.InvalidTrainSnapshotShape;
    var dir = try std.Io.Dir.openDirAbsolute(io, authority.output_dir, .{ .follow_symlinks = false });
    defer dir.close(io);
    const names = .{ "probabilities.f32", "original-y.f64", "original-values.f64", "original-bias.f64", "frozen-plan.bin", "manifest.json.part" };
    var created: [names.len]bool = @splat(false);
    errdefer {
        inline for (names, 0..) |name, i| if (created[i]) dir.deleteFile(io, name) catch {};
    }
    var payloads: [5]Payload = undefined;
    {
        var file = try dir.createFile(io, "original-values.f64", .{ .exclusive = true, .truncate = false });
        created[2] = true;
        defer file.close(io);
        var writer: PayloadWriter = .{ .file = &file, .io = io };
        for (original.values, 0..) |value, i| {
            if ((i & 4095) == 0) try control.check();
            try putF64(&writer, value);
        }
        try writer.flush();
        try file.sync(io);
        payloads[2] = payload("original-values.f64", original.values.len * 8, &writer.hash);
    }
    {
        const bias = original.bias orelse return error.InvalidTrainSnapshotShape;
        var file = try dir.createFile(io, "original-bias.f64", .{ .exclusive = true, .truncate = false });
        created[3] = true;
        defer file.close(io);
        var writer: PayloadWriter = .{ .file = &file, .io = io };
        for (bias, 0..) |value, i| {
            if ((i & 4095) == 0) try control.check();
            try putF64(&writer, value);
        }
        try writer.flush();
        try file.sync(io);
        payloads[3] = payload("original-bias.f64", bias.len * 8, &writer.hash);
    }
    {
        var file = try dir.createFile(io, "probabilities.f32", .{ .exclusive = true, .truncate = false });
        created[0] = true;
        defer file.close(io);
        var writer: PayloadWriter = .{ .file = &file, .io = io };
        for (source.probabilities_f32, 0..) |value, i| {
            if ((i & 4095) == 0) try control.check();
            try putF32(&writer, value);
        }
        try writer.flush();
        try file.sync(io);
        payloads[0] = payload("probabilities.f32", source.probabilities_f32.len * 4, &writer.hash);
    }
    {
        var file = try dir.createFile(io, "original-y.f64", .{ .exclusive = true, .truncate = false });
        created[1] = true;
        defer file.close(io);
        var writer: PayloadWriter = .{ .file = &file, .io = io };
        for (source.outputs, 0..) |value, i| {
            if ((i & 4095) == 0) try control.check();
            try putF64(&writer, value);
        }
        try writer.flush();
        try file.sync(io);
        payloads[1] = payload("original-y.f64", source.outputs.len * 8, &writer.hash);
    }
    {
        var file = try dir.createFile(io, "frozen-plan.bin", .{ .exclusive = true, .truncate = false });
        created[4] = true;
        defer file.close(io);
        var writer: PayloadWriter = .{ .file = &file, .io = io };
        try putU64(&writer, source.dim);
        try putU64(&writer, plan.scores.len);
        try putU64(&writer, 3);
        for (plan.scores, plan.scored, 0..) |score, levels, range_index| {
            try control.check();
            try putU64(&writer, score.start);
            try putU64(&writer, score.rows);
            for (levels, 0..) |candidate, level| {
                const owned = candidate.?;
                const chunk = owned.frozen.chunks[0];
                try putU64(&writer, range_index);
                try putU64(&writer, level);
                try putU64(&writer, chunk.selected_local.len);
                for (chunk.selected_local, chunk.weights, owned.fitted.bias, 0..) |local, weight, bias, selected| {
                    try putU64(&writer, local);
                    try putF64(&writer, weight);
                    try putF32(&writer, @floatCast(bias));
                    for (owned.fitted.values[selected * source.dim ..][0..source.dim]) |value| {
                        if (!std.math.isFinite(value) or @abs(value) > std.math.floatMax(f16)) return error.NonFiniteAttention;
                        var bits: [2]u8 = undefined;
                        std.mem.writeInt(u16, &bits, @bitCast(@as(f16, @floatCast(value))), .little);
                        try writer.write(&bits);
                    }
                }
            }
        }
        for (plan.branches) |branch| {
            try putU64(&writer, branch.middle_cap_rows);
            try putU64(&writer, branch.middle_rows);
            try putF64(&writer, branch.train_harm_before);
            try putF64(&writer, branch.train_harm_after);
            for (branch.targets) |target| try putU64(&writer, target);
        }
        try writer.flush();
        try file.sync(io);
        payloads[4] = payload("frozen-plan.bin", required - payloads[0].bytes - payloads[1].bytes - payloads[2].bytes - payloads[3].bytes, &writer.hash);
    }
    const body = try std.json.Stringify.valueAlloc(allocator, .{ .schema = schema, .encoding = "canonical-little-endian", .objective_arithmetic = "f32_serialized_probabilities_widened_to_f64", .training_only = true, .heldout_present = false, .diagnostic_only = true, .quality_credit = false, .production_qualified = false, .numerator_preservation_evaluated = false, .rounded_fitted_values_authenticated = true, .maximum_bytes = maximum_bytes, .payload_bytes = required, .rows = source.rows, .dim = source.dim, .train_queries = source.query_count, .domain_runs = domain_runs, .domain_counts = .{ .primary = domain_counts[0], .historical = domain_counts[1], .future = domain_counts[2] }, .cache_sha256 = std.fmt.bytesToHex(source.digest, .lower), .geometry_policy = plan.geometry_policy, .geometry_storage = @tagName(plan.geometry_storage), .source_sha256 = std.fmt.bytesToHex(authority.source_sha256, .lower), .config_sha256 = std.fmt.bytesToHex(authority.config_sha256, .lower), .model_sha256 = std.fmt.bytesToHex(authority.model_sha256, .lower), .source_freeze_sha256 = std.fmt.bytesToHex(authority.source_freeze_sha256, .lower), .binary_sha256 = std.fmt.bytesToHex(authority.binary_sha256, .lower), .producer_input_manifest_sha256 = std.fmt.bytesToHex(authority.producer_input_manifest_sha256, .lower), .complete_future_mix_manifest_sha256 = std.fmt.bytesToHex(authority.complete_future_mix_manifest_sha256, .lower), .complete_future_mix_policy_sha256 = std.fmt.bytesToHex(authority.complete_future_mix_policy_sha256, .lower), .consumer_source_manifest_sha256 = std.fmt.bytesToHex(authority.consumer_source_manifest_sha256, .lower), .consumer_binary_sha256 = std.fmt.bytesToHex(authority.consumer_binary_sha256, .lower), .branch_target_candidate_payload_sha256 = payloads[4].sha256, .payloads = payloads }, .{});
    defer allocator.free(body);
    if (body.len + 1 > manifest_reservation) return error.TrainSnapshotLimit;
    var mh = std.crypto.hash.sha2.Sha256.init(.{});
    mh.update(body);
    mh.update("\n");
    var manifest = try dir.createFile(io, "manifest.json.part", .{ .exclusive = true, .truncate = false });
    created[5] = true;
    defer manifest.close(io);
    try manifest.writeStreamingAll(io, body);
    try manifest.writeStreamingAll(io, "\n");
    try manifest.sync(io);
    try dir.renamePreserve("manifest.json.part", dir, "manifest.json", io);
    return .{ .manifest_sha256 = mh.finalResult(), .payload_bytes = required };
}

test "TRAIN snapshot buffered payload preserves exact bytes across flush boundaries" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const expected = try a.alloc(u8, 64 * 1024 + 37);
    defer a.free(expected);
    for (expected, 0..) |*byte, i| byte.* = @truncate(i *% 131 +% 17);
    var file = try tmp.dir.createFile(io, "buffered.bin", .{ .read = true, .exclusive = true, .truncate = false });
    defer file.close(io);
    var writer: PayloadWriter = .{ .file = &file, .io = io };
    try writer.write(expected[0..3]);
    try writer.write(expected[3 .. 64 * 1024 + 11]);
    try writer.write(expected[64 * 1024 + 11 ..]);
    try writer.flush();
    try file.sync(io);
    const actual = try a.alloc(u8, expected.len);
    defer a.free(actual);
    try std.testing.expectEqual(expected.len, try file.readPositionalAll(io, actual, 0));
    try std.testing.expectEqualSlices(u8, expected, actual);
    var direct: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(expected, &direct, .{});
    const buffered_digest = writer.hash.finalResult();
    try std.testing.expectEqualSlices(u8, &direct, &buffered_digest);
}

test "TRAIN snapshot cap is checked before publication" {
    var source: training.Source = undefined;
    var probabilities_f32 = [_]f32{1};
    var outputs = [_]f64{ 1, 2 };
    source.query_count = maximum_bytes;
    source.dim = 2;
    source.rows = 2;
    source.probabilities = &.{};
    source.probabilities_f32 = &probabilities_f32;
    source.outputs = &outputs;
    var plan: replay.AdaptivePlan = undefined;
    plan.scores = &.{};
    plan.scored = &.{};
    try std.testing.expectError(error.InvalidTrainSnapshotShape, requiredBytes(&source, &plan));
}

test "TRAIN snapshot cancellation is observed before filesystem publication" {
    const Cancel = struct {
        fn check(_: ?*anyopaque) !void {
            return error.Cancelled;
        }
    };
    var source: training.Source = undefined;
    var plan: replay.AdaptivePlan = undefined;
    const one: [32]u8 = @splat(1);
    try std.testing.expectError(error.Cancelled, write(std.testing.allocator, std.testing.io, .{ .output_dir = "/never-opened-train-snapshot", .source_sha256 = one, .config_sha256 = one, .model_sha256 = one, .source_freeze_sha256 = one, .binary_sha256 = one, .producer_input_manifest_sha256 = one, .complete_future_mix_manifest_sha256 = one, .complete_future_mix_policy_sha256 = one, .consumer_source_manifest_sha256 = one, .consumer_binary_sha256 = one }, &.{.{ .domain = .primary, .start = 0, .count = 1 }}, undefined, &source, &plan, .{ .check_fn = Cancel.check }));
}

fn filesystemFixture(allocator: std.mem.Allocator, output_dir: []const u8, control: solver.Control) !void {
    const keys = [_]f64{0};
    const values = [_]f64{2};
    const bias = [_]f64{0};
    const train = [_]f64{1};
    const probabilities = [_]f32{1};
    const outputs = [_]f64{2};
    var source: training.Source = .{ .allocator = allocator, .keys_ptr = &keys, .values_ptr = &values, .bias_ptr = &bias, .train_ptr = &train, .rows = 1, .dim = 1, .query_count = 1, .scale = 1, .softcap = 0, .backend = null, .query_batch_rows = 16, .cache_bytes = 12, .consumer_peak_bytes = 1, .digest = @splat(7), .probabilities_f32 = @constCast(&probabilities), .outputs = @constCast(&outputs) };
    _ = &source;
    const selected = [_]usize{0};
    const weights = [_]f64{1};
    const frozen_chunks = [_]@import("attention_match_joint_chunk_mass_diagnostic.zig").FrozenChunk{.{ .start = 0, .rows = 1, .selected_local = &selected, .weights = &weights }};
    const ranges = [_]@import("attention_match_cumulative_cache.zig").Range{.{ .start = 0, .rows = 1 }};
    var index_slices = [_][]usize{@constCast(&selected)};
    var weight_slices = [_][]f64{@constCast(&weights)};
    const fit_chunks = [_]@import("attention_match_joint_chunk_mass_diagnostic.zig").ChunkResult{.{ .start = 0, .rows = 1, .value_offset = 0, .selected_count = 1, .alpha = 1 }};
    var fit_values = [_]f64{2};
    var fit_bias = [_]f64{0};
    const candidate: replay.ScoredCandidate = .{ .frozen = .{ .allocator = allocator, .chunks = @constCast(&frozen_chunks), .ranges = @constCast(&ranges), .indices = &index_slices, .weights = &weight_slices }, .fitted = .{ .allocator = allocator, .chunks = @constCast(&fit_chunks), .values = &fit_values, .bias = &fit_bias, .nnls_sweeps = 0 } };
    var scores = [_]@import("attention_match_joint_adaptive_geometry.zig").RangeScores{.{ .start = 0, .rows = 1, .target_rows = .{ 1, 1, 1, 1 }, .mass_harm = .{ 0, 0, 0, 0 }, .output_harm = .{ 0, 0, 0, 0 } }};
    var scored = [_][3]?replay.ScoredCandidate{.{ candidate, candidate, candidate }};
    var levels = [_]@import("attention_match_joint_adaptive_geometry.zig").Level{.divisor2};
    var targets = [_]usize{1};
    const geometry: @import("attention_match_joint_adaptive_geometry.zig").Geometry = .{ .allocator = allocator, .levels = &levels, .targets = &targets, .middle_cap_rows = 1, .middle_rows = 1, .preserved_ranges = 0, .fitted_ranges = 1, .train_harm_before = 0, .train_harm_after = 0, .digest = @splat(8) };
    var plan: replay.AdaptivePlan = .{ .allocator = allocator, .scores = &scores, .scored = &scored, .branches = .{ geometry, geometry, geometry }, .geometry_policy = "fixture" };
    const one: [32]u8 = @splat(1);
    _ = try write(allocator, std.testing.io, .{ .output_dir = output_dir, .source_sha256 = one, .config_sha256 = one, .model_sha256 = one, .source_freeze_sha256 = one, .binary_sha256 = one, .producer_input_manifest_sha256 = one, .complete_future_mix_manifest_sha256 = one, .complete_future_mix_policy_sha256 = one, .consumer_source_manifest_sha256 = one, .consumer_binary_sha256 = one }, &.{.{ .domain = .primary, .start = 0, .count = 1 }}, .{ .keys = &keys, .values = &values, .bias = &bias, .rows = 1, .dim = 1, .scale = 1, .softcap = 0 }, &source, &plan, control);
}

fn allocationFixture(allocator: std.mem.Allocator) !void {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &root);
    const output = try std.fs.path.join(a, &.{ root[0..n], "snapshot" });
    defer a.free(output);
    try std.Io.Dir.createDirAbsolute(io, output, std.Io.File.Permissions.fromMode(0o700));
    try filesystemFixture(allocator, output, .{});
}

test "TRAIN snapshot unwinds every allocator failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFixture, .{});
}

test "TRAIN snapshot mid-stream cancellation removes partial payloads" {
    const Cancel = struct {
        var calls: usize = 0;
        fn check(_: ?*anyopaque) !void {
            calls += 1;
            if (calls >= 2) return error.Cancelled;
        }
    };
    Cancel.calls = 0;
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &root);
    const output = try std.fs.path.join(a, &.{ root[0..n], "cancel" });
    defer a.free(output);
    try std.Io.Dir.createDirAbsolute(io, output, std.Io.File.Permissions.fromMode(0o700));
    try std.testing.expectError(error.Cancelled, filesystemFixture(a, output, .{ .check_fn = Cancel.check }));
    var dir = try std.Io.Dir.openDirAbsolute(io, output, .{});
    defer dir.close(io);
    try std.testing.expectError(error.FileNotFound, dir.openFile(io, "manifest.json", .{}));
    try std.testing.expectError(error.FileNotFound, dir.openFile(io, "probabilities.f32", .{}));
}

test "TRAIN snapshot collision preserves pre-existing payload and removes only owned files" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &root);
    const output = try std.fs.path.join(a, &.{ root[0..n], "collision" });
    defer a.free(output);
    try std.Io.Dir.createDirAbsolute(io, output, std.Io.File.Permissions.fromMode(0o700));
    var dir = try std.Io.Dir.openDirAbsolute(io, output, .{});
    defer dir.close(io);
    const marker = "pre-existing-owned-by-caller";
    var existing = try dir.createFile(io, "original-y.f64", .{ .exclusive = true, .truncate = false });
    try existing.writeStreamingAll(io, marker);
    try existing.sync(io);
    existing.close(io);

    try std.testing.expectError(error.PathAlreadyExists, filesystemFixture(a, output, .{}));
    var preserved = try dir.openFile(io, "original-y.f64", .{});
    defer preserved.close(io);
    var bytes: [marker.len]u8 = undefined;
    try std.testing.expectEqual(marker.len, try preserved.readPositionalAll(io, &bytes, 0));
    try std.testing.expectEqualStrings(marker, &bytes);
    try std.testing.expectEqual(@as(u64, marker.len), (try preserved.stat(io)).size);
    try std.testing.expectError(error.FileNotFound, dir.openFile(io, "original-values.f64", .{}));
    try std.testing.expectError(error.FileNotFound, dir.openFile(io, "manifest.json", .{}));
}

test "TRAIN snapshot commit preserves pre-existing manifest and removes owned payloads" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &root);
    const output = try std.fs.path.join(a, &.{ root[0..n], "manifest-collision" });
    defer a.free(output);
    try std.Io.Dir.createDirAbsolute(io, output, std.Io.File.Permissions.fromMode(0o700));
    var dir = try std.Io.Dir.openDirAbsolute(io, output, .{});
    defer dir.close(io);
    const marker = "pre-existing-manifest-owned-by-caller";
    var existing = try dir.createFile(io, "manifest.json", .{ .exclusive = true, .truncate = false });
    try existing.writeStreamingAll(io, marker);
    try existing.sync(io);
    existing.close(io);

    try std.testing.expectError(error.PathAlreadyExists, filesystemFixture(a, output, .{}));
    var preserved = try dir.openFile(io, "manifest.json", .{});
    defer preserved.close(io);
    var bytes: [marker.len]u8 = undefined;
    try std.testing.expectEqual(marker.len, try preserved.readPositionalAll(io, &bytes, 0));
    try std.testing.expectEqualStrings(marker, &bytes);
    try std.testing.expectEqual(@as(u64, marker.len), (try preserved.stat(io)).size);
    inline for (.{ "probabilities.f32", "original-y.f64", "original-values.f64", "original-bias.f64", "frozen-plan.bin", "manifest.json.part" }) |name| {
        try std.testing.expectError(error.FileNotFound, dir.openFile(io, name, .{}));
    }
}

test "TRAIN snapshot real filesystem roundtrip and allocation failure publish manifest last" {
    const a = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var root: [std.fs.max_path_bytes]u8 = undefined;
    const n = try tmp.dir.realPath(io, &root);
    const good = try std.fs.path.join(a, &.{ root[0..n], "good" });
    defer a.free(good);
    try std.Io.Dir.createDirAbsolute(io, good, std.Io.File.Permissions.fromMode(0o700));
    try filesystemFixture(a, good, .{});
    var dir = try std.Io.Dir.openDirAbsolute(io, good, .{});
    defer dir.close(io);
    var manifest = try dir.openFile(io, "manifest.json", .{});
    manifest.close(io);
    const failed = try std.fs.path.join(a, &.{ root[0..n], "failed" });
    defer a.free(failed);
    try std.Io.Dir.createDirAbsolute(io, failed, std.Io.File.Permissions.fromMode(0o700));
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    try std.testing.expectError(error.OutOfMemory, filesystemFixture(failing.allocator(), failed, .{}));
    var failed_dir = try std.Io.Dir.openDirAbsolute(io, failed, .{});
    defer failed_dir.close(io);
    try std.testing.expectError(error.FileNotFound, failed_dir.openFile(io, "manifest.json", .{}));
    try std.testing.expectError(error.FileNotFound, failed_dir.openFile(io, "probabilities.f32", .{}));
}
