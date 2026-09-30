// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
const std = @import("std");
const data = @import("inference_internal").finetune.gliner_boundary_dataset;

pub fn main(init: std.process.Init) !void {
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, init.gpa);
    defer args.deinit();
    _ = args.next();
    const path = args.next() orelse return error.InvalidArguments;
    if (args.next() != null) return error.InvalidArguments;
    var failure: data.Failure = .{};
    var dataset = data.Dataset.open(init.gpa, path, .{}, .{ .io = init.io }, &failure) catch |err| {
        std.debug.print("GLiNER2.5 dataset validation failed at line {?d} ({s}): {s}\n", .{ failure.line, @tagName(failure.stage), @errorName(err) });
        return err;
    };
    defer dataset.deinit();
    var buffer: [4096]u8 = undefined;
    var writer = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    try std.json.Stringify.value(.{ .examples = dataset.index.len, .sha256 = std.fmt.bytesToHex(dataset.sha256, .lower) }, .{}, &writer.interface);
    try writer.interface.writeByte('\n');
    try writer.interface.flush();
}
