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

const std = @import("std");
const inference = @import("inference_internal");
const gemma4 = inference.finetune.gemma4;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    var argv = std.ArrayListUnmanaged([]const u8).empty;
    defer argv.deinit(allocator);
    while (args.next()) |arg| try argv.append(allocator, arg);
    try runFromArgs(allocator, init.io, argv.items);
}

pub fn runFromArgs(allocator: std.mem.Allocator, io: std.Io, argv: []const []const u8) !void {
    if (argv.len == 1 and isHelpArg(argv[0])) {
        printUsage();
        return;
    }

    var model: ?[]const u8 = null;
    var adapter: ?[]const u8 = null;
    var out: ?[]const u8 = null;
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= argv.len or model != null) return usageError();
            model = argv[i];
        } else if (std.mem.eql(u8, arg, "--adapter")) {
            i += 1;
            if (i >= argv.len or adapter != null) return usageError();
            adapter = argv[i];
        } else if (std.mem.eql(u8, arg, "--out")) {
            i += 1;
            if (i >= argv.len or out != null) return usageError();
            out = argv[i];
        } else {
            return usageError();
        }
    }

    var summary = try gemma4.exportPeftAdapter(
        allocator,
        model orelse return usageError(),
        adapter orelse return usageError(),
        out orelse return usageError(),
    );
    defer gemma4.freePeftExportSummary(allocator, &summary);

    const stdout = std.Io.File.stdout();
    var buffer: [4096]u8 = undefined;
    var writer = stdout.writer(io, &buffer);
    try std.json.Stringify.value(summary, .{ .whitespace = .indent_2 }, &writer.interface);
    try writer.interface.writeByte('\n');
    try writer.interface.flush();
}

pub fn printUsage() void {
    std.debug.print(
        \\usage: antfly inference finetune adapter export gemma4-peft --model <dir> \\
        \\       --adapter <antfly-adapter-dir> --out <new-peft-dir>
        \\
        \\The source remains unchanged. The immutable output contains stock PEFT
        \\adapter tensor keys, adapter_config.json, and an Antfly provenance sidecar.
        \\
    , .{});
}

fn usageError() error{InvalidArguments} {
    printUsage();
    return error.InvalidArguments;
}

fn isHelpArg(arg: []const u8) bool {
    return std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "help");
}

test "Gemma4 PEFT export parser requires the named production contract" {
    try std.testing.expectError(error.InvalidArguments, runFromArgs(std.testing.allocator, std.testing.io, &.{ "base", "adapter", "out" }));
    try std.testing.expectError(error.InvalidArguments, runFromArgs(std.testing.allocator, std.testing.io, &.{ "--model", "base", "--adapter", "adapter" }));
}
