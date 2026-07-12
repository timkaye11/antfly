// Copyright 2026 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
//
//! Debug harness: run an arbitrary ONNX model through the native ONNX importer
//! (CPU NativeCompute backend, graph interpreter) and dump every graph output
//! tensor to raw little-endian f32 files, plus a `_shapes.txt` manifest.
//!
//! Usage:
//!   onnx-forward-dump <model.onnx> <pixel_values.f32> <out_dir>
//!
//! `pixel_values.f32` is raw little-endian f32 for a [1,3,224,224] tensor.
//! Intended for op-by-op parity debugging against an onnxruntime reference:
//! run a probe model whose intermediates are promoted to graph outputs.

const std = @import("std");
const inference = @import("inference_internal");
const backends = inference.backends;
const imported = backends.imported_onnx_session;
const Tensor = backends.Tensor;

fn readFileAllocAtPath(io: std.Io, path: []const u8, allocator: std.mem.Allocator) ![]u8 {
    if (!std.fs.path.isAbsolute(path)) {
        return std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(std.math.maxInt(usize)));
    }
    const parent = std.fs.path.dirname(path) orelse return error.BadPathName;
    const base = std.fs.path.basename(path);
    var dir = try std.Io.Dir.openDirAbsolute(io, parent, .{});
    defer dir.close(io);
    return dir.readFileAlloc(io, base, allocator, .limited(std.math.maxInt(usize)));
}

fn writeFileAtPath(io: std.Io, path: []const u8, data: []const u8) !void {
    if (!std.fs.path.isAbsolute(path)) {
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = data });
        return;
    }
    const parent = std.fs.path.dirname(path) orelse return error.BadPathName;
    const base = std.fs.path.basename(path);
    var dir = try std.Io.Dir.openDirAbsolute(io, parent, .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = base, .data = data });
}

fn sanitize(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    const out = try allocator.alloc(u8, name.len);
    for (name, 0..) |c, i| {
        out[i] = switch (c) {
            'A'...'Z', 'a'...'z', '0'...'9', '_', '.', '-' => c,
            else => '_',
        };
    }
    return out;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = std.heap.page_allocator;

    var args_iter = std.process.Args.Iterator.init(init.minimal.args);
    _ = args_iter.next();
    const onnx_path = args_iter.next() orelse return error.MissingOnnxPath;
    const input_path = args_iter.next() orelse return error.MissingInputPath;
    const out_dir = args_iter.next() orelse return error.MissingOutDir;

    const input_bytes = try readFileAllocAtPath(io, input_path, allocator);
    defer allocator.free(input_bytes);
    if (input_bytes.len % 4 != 0) return error.BadInputByteLength;
    const n = input_bytes.len / 4;
    const floats = try allocator.alloc(f32, n);
    defer allocator.free(floats);
    for (0..n) |i| {
        floats[i] = @bitCast(std.mem.readInt(u32, input_bytes[i * 4 ..][0..4], .little));
    }
    // Infer a square [1,3,S,S] pixel tensor from the element count so this harness
    // works for any vision resolution (CLIP 224, SigLIP 512, ...). An optional 4th
    // positional arg overrides the spatial side explicitly.
    var side: usize = 224;
    if (args_iter.next()) |side_arg| {
        side = std.fmt.parseInt(usize, side_arg, 10) catch 224;
    } else if (n % 3 == 0) {
        const hw = n / 3;
        const s: usize = std.math.sqrt(hw);
        if (s * s == hw) side = s;
    }
    if (n != 3 * side * side) {
        std.debug.print("warning: input element count {d} != {d} (expected [1,3,{d},{d}])\n", .{ n, 3 * side * side, side, side });
    }
    const shape = [_]i64{ 1, 3, @intCast(side), @intCast(side) };
    std.debug.print("input shape [1,3,{d},{d}] ({d} elems)\n", .{ side, side, n });

    var pv = try Tensor.initFloat32(allocator, "pixel_values", &shape, floats);
    defer pv.deinit();

    var session = try imported.createSessionWithOptions(allocator, onnx_path, .native, .{ .io = io });
    defer session.close();

    const outputs = try session.run(&.{pv}, allocator);
    defer {
        for (outputs) |*t| t.deinit();
        allocator.free(outputs);
    }

    var shapes_text = std.ArrayListUnmanaged(u8).empty;
    defer shapes_text.deinit(allocator);

    for (outputs) |*t| {
        const safe = try sanitize(allocator, t.name);
        defer allocator.free(safe);
        const fname = try std.fmt.allocPrint(allocator, "{s}/{s}.f32", .{ out_dir, safe });
        defer allocator.free(fname);

        if (t.dtype == .f32) {
            const vals = t.asFloat32();
            try writeFileAtPath(io, fname, std.mem.sliceAsBytes(vals));
        } else {
            std.debug.print("skip non-f32 output {s} dtype={s}\n", .{ t.name, @tagName(t.dtype) });
        }

        const line = try std.fmt.allocPrint(allocator, "{s}\t{s}\t{any}\n", .{ safe, @tagName(t.dtype), t.shape });
        defer allocator.free(line);
        try shapes_text.appendSlice(allocator, line);
        std.debug.print("dumped {s} dtype={s} shape={any} n={d}\n", .{ t.name, @tagName(t.dtype), t.shape, t.elementCount() });
    }

    const shapes_path = try std.fmt.allocPrint(allocator, "{s}/_shapes.txt", .{out_dir});
    defer allocator.free(shapes_path);
    try writeFileAtPath(io, shapes_path, shapes_text.items);
    std.debug.print("wrote {d} outputs to {s}\n", .{ outputs.len, out_dir });
}
