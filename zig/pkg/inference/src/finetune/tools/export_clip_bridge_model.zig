// Copyright 2026 Antfly, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package a base clipclap image embedder plus a trained CLIP→text bridge head
// into a NEW deployable serving model directory (e.g.
// clipclap-vit-b-32-nomic-bridge). The bridge is a 3-layer MLP that projects a
// native, L2-normalized clipclap image embedding (512-d, post visual_projection)
// into the target text embedder's retrieval space (nomic modernbert-embed-base
// `search_document:`, 768-d).
//
// The base clipclap dir is READ-ONLY here: all base assets are copied into the
// new --out dir, and the bridge head is written alongside as
// `bridge_head.safetensors`. The base dir is NEVER modified — tack-on training's
// CLIP precompute needs native, un-bridged vectors, so the base clipclap dir
// must keep producing byte-identical native CLIP output.
//
// Output layout (--out):
//   <all files from --clipclap-dir, copied verbatim>
//   bridge_head.safetensors  — 3-layer MLP under the `clip_bridge_head/` namespace
//
// The bridge weights are byte-copied (F32) from the training artifact's
//   net.{0,2,4}.{weight,bias}
// into
//   clip_bridge_head/net.{0,2,4}.{weight,bias}
// (PyTorch Linear layout [out, in], y = x·Wᵀ + b). Model discovery
// (src/models/manifest.zig) keys has_bridge purely off the presence of
// bridge_head.safetensors, and the serving forward runs in
// pipelines/embedding.zig applyClipBridge.

const std = @import("std");
const inference = @import("inference_internal");
const compat = inference.io.compat;
const safetensors = inference.models.safetensors;

const bridge_prefix = "clip_bridge_head/";

const BridgeTensor = struct {
    src: []const u8,
    dst: []const u8,
    /// 2 for weights ([out, in]), 1 for biases ([dim]).
    rank: usize,
};

// Source names from the training artifact (nn.Sequential of Linear/GELU): the
// three Linear layers land at module indices 0, 2, 4.
const bridge_tensors = [_]BridgeTensor{
    .{ .src = "net.0.weight", .dst = bridge_prefix ++ "net.0.weight", .rank = 2 },
    .{ .src = "net.0.bias", .dst = bridge_prefix ++ "net.0.bias", .rank = 1 },
    .{ .src = "net.2.weight", .dst = bridge_prefix ++ "net.2.weight", .rank = 2 },
    .{ .src = "net.2.bias", .dst = bridge_prefix ++ "net.2.bias", .rank = 1 },
    .{ .src = "net.4.weight", .dst = bridge_prefix ++ "net.4.weight", .rank = 2 },
    .{ .src = "net.4.bias", .dst = bridge_prefix ++ "net.4.bias", .rank = 1 },
};

const Options = struct {
    clipclap_dir: []const u8,
    bridge: []const u8,
    out: []const u8,
    force: bool = false,
};

const Stats = struct {
    files_copied: usize = 0,
    in_dim: usize = 0,
    hidden0: usize = 0,
    hidden1: usize = 0,
    out_dim: usize = 0,
    bridge_bytes: u64 = 0,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const opts = try parseOptions(&args) orelse return;
    const stats = try exportModel(allocator, opts);
    try writeSummary(opts, stats);
}

fn parseOptions(args: *std.process.Args.Iterator) !?Options {
    var opts: Options = .{ .clipclap_dir = "", .bridge = "", .out = "" };
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--clipclap-dir")) {
            opts.clipclap_dir = args.next() orelse return error.MissingClipclapDir;
        } else if (std.mem.eql(u8, arg, "--bridge")) {
            opts.bridge = args.next() orelse return error.MissingBridge;
        } else if (std.mem.eql(u8, arg, "--out")) {
            opts.out = args.next() orelse return error.MissingOut;
        } else if (std.mem.eql(u8, arg, "--force")) {
            opts.force = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            usage();
            return null;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }
    if (opts.clipclap_dir.len == 0 or opts.bridge.len == 0 or opts.out.len == 0) {
        usage();
        return error.InvalidArgument;
    }
    return opts;
}

fn usage() void {
    std.debug.print(
        \\Usage: export-clip-bridge-model --clipclap-dir <dir> --bridge <bridge_mlp.safetensors> --out <dir> [--force]
        \\
        \\Packages a base clipclap image embedder plus a trained CLIP→text bridge
        \\head into a NEW serving model directory. The base --clipclap-dir is
        \\read-only (copied verbatim into --out); bridge_head.safetensors is written
        \\alongside under the clip_bridge_head/ namespace. Do NOT point --out at the
        \\base clipclap dir — the base must keep emitting native (un-bridged) CLIP
        \\vectors for tack-on training precompute.
        \\
    , .{});
}

fn exportModel(allocator: std.mem.Allocator, opts: Options) !Stats {
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    var stats = Stats{};

    // Guard against packaging INTO the base dir (would corrupt native precompute).
    if (std.mem.eql(u8, std.mem.trimEnd(u8, opts.clipclap_dir, "/"), std.mem.trimEnd(u8, opts.out, "/"))) {
        std.debug.print("error: --out must differ from --clipclap-dir (the base dir must stay bridge-less)\n", .{});
        return error.OutEqualsBase;
    }

    // Refuse to clobber an existing bridge export unless --force.
    if (!opts.force) {
        const guard = try std.fmt.allocPrint(arena, "{s}/bridge_head.safetensors", .{opts.out});
        if (compat.cwd().statFile(compat.io(), guard, .{})) |_| {
            std.debug.print("error: {s} already exists (use --force to overwrite)\n", .{guard});
            return error.OutputExists;
        } else |_| {}
    }

    // --- Copy base clipclap assets verbatim (read-only source) ---------------
    try copyTree(arena, opts.clipclap_dir, opts.out, &stats);

    // --- Write bridge_head.safetensors --------------------------------------
    var reader = try safetensors.MMapReader.openFileAbsolute(allocator, opts.bridge);
    defer reader.deinit();

    // Read + validate arch dims (in → hidden0 → hidden1 → out).
    const w0 = reader.header.tensors.get("net.0.weight") orelse return error.BridgeTensorMissing;
    const w2 = reader.header.tensors.get("net.2.weight") orelse return error.BridgeTensorMissing;
    const w4 = reader.header.tensors.get("net.4.weight") orelse return error.BridgeTensorMissing;
    if (w0.shape.len != 2 or w2.shape.len != 2 or w4.shape.len != 2) return error.BridgeShapeMismatch;
    stats.hidden0 = @intCast(w0.shape[0]);
    stats.in_dim = @intCast(w0.shape[1]);
    stats.hidden1 = @intCast(w2.shape[0]);
    stats.out_dim = @intCast(w4.shape[0]);
    if (@as(usize, @intCast(w2.shape[1])) != stats.hidden0) return error.BridgeShapeMismatch;
    if (@as(usize, @intCast(w4.shape[1])) != stats.hidden1) return error.BridgeShapeMismatch;

    const bridge_out = try std.fmt.allocPrint(arena, "{s}/bridge_head.safetensors", .{opts.out});
    stats.bridge_bytes = try writeBridgeSafetensors(allocator, bridge_out, &reader);

    return stats;
}

/// Recursively copy every file under `src_dir` into `dst_dir`, preserving
/// subdirectory structure. A pre-existing `bridge_head.safetensors` in the
/// source is skipped (the base dir must be bridge-less). Uses copyFile with
/// make_path so `dst_dir` and any subdirs are created on demand.
fn copyTree(arena: std.mem.Allocator, src_dir: []const u8, dst_dir: []const u8, stats: *Stats) !void {
    const io = compat.io();
    var dir = compat.cwd().openDir(io, src_dir, .{ .iterate = true }) catch |err| {
        std.debug.print("error: could not open base clipclap dir {s}: {}\n", .{ src_dir, err });
        return err;
    };
    defer dir.close(io);

    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.name.len == 0 or entry.name[0] == '.') continue;
        const child_src = try std.fmt.allocPrint(arena, "{s}/{s}", .{ src_dir, entry.name });
        const child_dst = try std.fmt.allocPrint(arena, "{s}/{s}", .{ dst_dir, entry.name });
        switch (entry.kind) {
            .directory => try copyTree(arena, child_src, child_dst, stats),
            .file, .sym_link => {
                if (std.mem.eql(u8, entry.name, "bridge_head.safetensors")) continue;
                try compat.cwd().copyFile(child_src, compat.cwd(), child_dst, io, .{ .make_path = true });
                stats.files_copied += 1;
            },
            else => {},
        }
    }
}

/// Write the 6 bridge tensors (byte-preserved F32) into a safetensors file under
/// the `clip_bridge_head/` namespace. Returns the file size in bytes.
fn writeBridgeSafetensors(
    allocator: std.mem.Allocator,
    path: []const u8,
    reader: *const safetensors.MMapReader,
) !u64 {
    // Header JSON with sequential data offsets.
    var header_buf: std.Io.Writer.Allocating = .init(allocator);
    defer header_buf.deinit();
    const hw = &header_buf.writer;

    try hw.writeAll("{\"__metadata__\":{\"format\":\"pt\"}");
    var offset: u64 = 0;
    for (bridge_tensors) |bt| {
        const meta = reader.header.tensors.get(bt.src) orelse return error.BridgeTensorMissing;
        if (meta.dtype != .f32) return error.BridgeDtypeUnsupported;
        if (meta.shape.len != bt.rank) return error.BridgeShapeMismatch;
        const byte_len = meta.data_end - meta.data_start;
        try hw.print(",\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{bt.dst});
        for (meta.shape, 0..) |dim, i| {
            if (i > 0) try hw.writeByte(',');
            try hw.print("{d}", .{dim});
        }
        try hw.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + byte_len });
        offset += byte_len;
    }
    try hw.writeByte('}');
    const header_json = header_buf.written();

    var file = try compat.cwd().createFile(compat.io(), path, .{ .truncate = true });
    defer file.close(compat.io());
    var write_buf: [64 * 1024]u8 = undefined;
    var writer = file.writerStreaming(compat.io(), &write_buf);
    const w = &writer.interface;

    var size_buf: [8]u8 = undefined;
    std.mem.writeInt(u64, &size_buf, header_json.len, .little);
    try w.writeAll(&size_buf);
    try w.writeAll(header_json);

    var total: u64 = 8 + header_json.len;
    for (bridge_tensors) |bt| {
        var tensor = try reader.readTensor(bt.src);
        defer tensor.deinit();
        try w.writeAll(tensor.data);
        total += tensor.data.len;
    }
    try w.flush();
    return total;
}

fn writeSummary(opts: Options, stats: Stats) !void {
    var stdout_buf: [4 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(compat.io(), &stdout_buf);
    const w = &stdout.interface;
    try w.print("export-clip-bridge-model: wrote {s}\n", .{opts.out});
    try w.print("  base clipclap dir : {s} ({d} files copied)\n", .{ opts.clipclap_dir, stats.files_copied });
    try w.print("  bridge artifact   : {s}\n", .{opts.bridge});
    try w.print("  bridge arch       : {d} → {d} → {d} → {d} (Linear/GELU/Linear/GELU/Linear)\n", .{ stats.in_dim, stats.hidden0, stats.hidden1, stats.out_dim });
    try w.print("  bridge_head.safetensors: {d} bytes ({s}net.{{0,2,4}}.{{weight,bias}})\n", .{ stats.bridge_bytes, bridge_prefix });
    try w.print("  NOTE: base clipclap dir left bridge-less (native CLIP precompute intact)\n", .{});
    try w.flush();
}
