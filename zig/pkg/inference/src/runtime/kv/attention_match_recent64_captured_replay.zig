const std = @import("std");
const checkpoint = @import("attention_match_checkpoint.zig");
const direct = @import("attention_match_recent64_direct_diagnostic.zig");
const fixture = @import("attention_match_recent64_captured_fixture.zig");
const heldout_bundle = @import("attention_match_prepared_heldout_bundle.zig");
const solver = @import("attention_match.zig");
const Fitter = @import("../../ops/metal_attention_match_fit.zig").Fitter;

pub const Binding = struct {
    manifest_path: []const u8,
    expected_manifest_sha256: [32]u8,
    expected_binary_sha256: [32]u8,
    source_afkv_path: []const u8,
    heldout_path: []const u8,
    heldout_authority_sha256: [32]u8,
    mode: direct.Mode,
};

fn readChecked(allocator: std.mem.Allocator, io: std.Io, path: []const u8, maximum: usize, expected: ?[32]u8) ![]u8 {
    return readCheckedFromDir(std.Io.Dir.cwd(), allocator, io, path, maximum, expected);
}

fn readCheckedFromDir(dir: std.Io.Dir, allocator: std.mem.Allocator, io: std.Io, path: []const u8, maximum: usize, expected: ?[32]u8) ![]u8 {
    // Zig's read limit is exclusive; an exact-sized valid payload needs one
    // byte of headroom. The authenticated maximum remains inclusive below.
    const exclusive_limit = try std.math.add(usize, maximum, 1);
    const bytes = try dir.readFileAlloc(io, path, allocator, .limited(exclusive_limit));
    errdefer allocator.free(bytes);
    if (bytes.len > maximum) return error.StreamTooLong;
    if (expected) |digest| {
        var actual: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(bytes, &actual, .{});
        if (!std.mem.eql(u8, &actual, &digest)) return error.Recent64CapturedFixtureChanged;
    }
    return bytes;
}

pub fn run(allocator: std.mem.Allocator, io: std.Io, binding: Binding, control: solver.Control) !direct.Result {
    const manifest_bytes = try readChecked(allocator, io, binding.manifest_path, 256 * 1024, binding.expected_manifest_sha256);
    defer allocator.free(manifest_bytes);
    var parsed = try std.json.parseFromSlice(fixture.Manifest, allocator, manifest_bytes, .{ .allocate = .alloc_always });
    defer parsed.deinit();
    try fixture.validateManifest(parsed.value);
    if (!std.mem.eql(u8, &parsed.value.binary_sha256, &binding.expected_binary_sha256)) return error.Recent64CapturedFixtureChanged;
    const root = std.fs.path.dirname(binding.manifest_path) orelse return error.InvalidRecent64CapturedFixture;
    const values_path = try std.fs.path.join(allocator, &.{ root, fixture.values_name });
    defer allocator.free(values_path);
    const positions_path = try std.fs.path.join(allocator, &.{ root, fixture.positions_name });
    defer allocator.free(positions_path);
    const value_bytes = try readChecked(allocator, io, values_path, fixture.value_count * @sizeOf(f32), parsed.value.values.sha256);
    defer allocator.free(value_bytes);
    const position_bytes = try readChecked(allocator, io, positions_path, fixture.position_count * @sizeOf(u64), parsed.value.positions.sha256);
    defer allocator.free(position_bytes);
    var training = try fixture.reconstruct(allocator, parsed.value, value_bytes, position_bytes);
    defer training.deinit();

    const source_bytes = try readChecked(allocator, io, binding.source_afkv_path, checkpoint.serving_limits.max_file_bytes, parsed.value.source_checkpoint_sha256);
    defer allocator.free(source_bytes);
    var source = try checkpoint.decodeWithControl(allocator, source_bytes, .{
        .weights = parsed.value.identity.weights,
        .tokenizer = parsed.value.identity.tokenizer,
        .template = parsed.value.identity.template,
        .layout = parsed.value.identity.layout,
    }, checkpoint.serving_limits, control);
    defer source.deinit();
    const heldout_bytes = try readChecked(allocator, io, binding.heldout_path, 16 * 1024 * 1024, null);
    defer allocator.free(heldout_bytes);
    var heldout = try heldout_bundle.decodeOwned(allocator, heldout_bytes, binding.heldout_authority_sha256);
    defer heldout.deinit();
    var fitter = try Fitter.initWorker(allocator, null, direct.head_workspace_bytes, direct.head_workspace_bytes);
    defer _ = fitter.deinitAndTakeValueSolves();
    try fitter.setPhaseProfiling(true);
    return direct.run(allocator, source.state, training, heldout, fitter.backend(), control, binding.mode) catch |err| {
        std.debug.print("recent64-captured-replay-fitter-failure-profile-json {f}\n", .{std.json.fmt(.{ .schema = "antfly.recent64-captured-replay-failure-profile.v1", .mode = @tagName(binding.mode), .error_name = @errorName(err), .phase_counters = fitter.phaseCounters(), .diagnostic_only = true, .qualification_credit = false }, .{})});
        return err;
    };
}

test "captured replay reads exact sized payload and rejects excess or changed bytes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "payload", .data = "abcd" });
    const bytes = try readCheckedFromDir(tmp.dir, std.testing.allocator, std.testing.io, "payload", 4, null);
    defer std.testing.allocator.free(bytes);
    try std.testing.expectEqualStrings("abcd", bytes);
    const wrong: [32]u8 = @splat(1);
    try std.testing.expectError(error.Recent64CapturedFixtureChanged, readCheckedFromDir(tmp.dir, std.testing.allocator, std.testing.io, "payload", 4, wrong));
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "payload", .data = "abcde" });
    try std.testing.expectError(error.StreamTooLong, readCheckedFromDir(tmp.dir, std.testing.allocator, std.testing.io, "payload", 4, null));
}
