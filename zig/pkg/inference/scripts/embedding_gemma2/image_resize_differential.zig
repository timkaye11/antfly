// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0

// Generate exact CPU Pillow-compatible RGB8 resize and patch-packing goldens.
// Files are little endian; run with the CUDA differential on a 64-bit host.
const std = @import("std");
const processing = @import("processing");
fn write(init: std.process.Init, directory: []const u8, name: []const u8, bytes: []const u8) !void {
    const path = try std.fs.path.join(init.gpa, &.{ directory, name });
    defer init.gpa.free(path);
    const file = try std.Io.Dir.cwd().createFile(init.io, path, .{});
    defer file.close(init.io);
    var buffer: [4096]u8 = undefined;
    var writer = file.writer(init.io, &buffer);
    try writer.interface.writeAll(bytes);
    try writer.end();
}
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 2) return error.ExpectedOutputDirectory;
    const output_directory = args[1];
    const cases = [_][4]usize{ .{ 1, 1, 16, 16 }, .{ 7, 5, 16, 16 }, .{ 128, 96, 912, 672 }, .{ 256, 1536, 320, 1856 }, .{ 1536, 256, 1856, 320 }, .{ 64, 64, 16, 16 }, .{ 65535, 1, 38304, 16 } };
    for (cases, 0..) |shape, index| {
        var arena = std.heap.ArenaAllocator.init(init.gpa);
        defer arena.deinit();
        const allocator = arena.allocator();
        const source_w = shape[0];
        const source_h = shape[1];
        const target_w = shape[2];
        const target_h = shape[3];
        const stride = source_w * 3 + 7;
        const source = try allocator.alloc(u8, stride * source_h);
        for (source, 0..) |*value, i| value.* = @intCast((i * 37 + i / 3) % 256);
        const view = processing.ImageU8{ .data = source, .width = @intCast(source_w), .height = @intCast(source_h), .format = .rgb8, .row_stride_bytes = stride };
        const chw = try processing.preprocessDecodedRectScaledWithResample(allocator, view, @intCast(target_w), @intCast(target_h), .{ 0, 0, 0 }, .{ 1, 1, 1 }, 1.0 / 255.0, .pillow_bicubic);
        const expected = try allocator.alloc(f32, target_w * target_h * 3);
        for (0..target_h / 16) |gy| for (0..target_w / 16) |gx| {
            const base = (gy * (target_w / 16) + gx) * 768;
            for (0..16) |py| for (0..16) |px| for (0..3) |channel| {
                expected[base + (py * 16 + px) * 3 + channel] = 2 *
                    (chw[channel * target_w * target_h + (gy * 16 + py) * target_w + gx * 16 + px] - 0.5);
            };
        };
        var x = try processing.buildPillowBicubicAxis(allocator, source_w, target_w, false);
        defer x.deinit();
        var y = try processing.buildPillowBicubicAxis(allocator, source_h, target_h, false);
        defer y.deinit();
        const path = try std.fmt.allocPrint(allocator, "{s}/case-{d}", .{ output_directory, index });
        try std.Io.Dir.cwd().createDirPath(init.io, path);
        const metadata = [_]u64{ source_w, source_h, stride, target_w, target_h, 16, x.precision_bits, y.precision_bits };
        try write(init, path, "shape.u64", std.mem.sliceAsBytes(&metadata));
        try write(init, path, "source.u8", source);
        try write(init, path, "expected.f32", std.mem.sliceAsBytes(expected));
        inline for (.{ .{ "x", x }, .{ "y", y } }) |axis| {
            try write(init, path, axis[0] ++ "_starts.u64", std.mem.sliceAsBytes(axis[1].starts));
            try write(init, path, axis[0] ++ "_offsets.u64", std.mem.sliceAsBytes(axis[1].offsets));
            try write(init, path, axis[0] ++ "_weights.i32", std.mem.sliceAsBytes(axis[1].weights));
        }
    }
}
