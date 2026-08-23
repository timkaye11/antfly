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
const finetune = inference.finetune.gemma4;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    var eval_program: ?[]const u8 = null;
    var positional = std.ArrayListUnmanaged([]const u8).empty;
    defer positional.deinit(allocator);

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--eval")) {
            eval_program = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            return usageError();
        } else {
            try positional.append(allocator, arg);
        }
    }

    if (positional.items.len != 3) return usageError();
    try validateEvalAdmission(eval_program);
    const base_model_dir = positional.items[0];
    const adapter_model_dir = positional.items[1];
    const out_dir = positional.items[2];

    var summary = try finetune.materializeMergedModel(allocator, base_model_dir, adapter_model_dir, out_dir);
    defer finetune.freeMaterializeSummary(allocator, &summary);

    try printJson(init, summary);
}

fn validateEvalAdmission(eval_program: ?[]const u8) !void {
    if (eval_program != null) return error.MaterializeEvalRequiresStagedEvaluator;
}

test "gemma4 materialize eval fails closed before publication" {
    try validateEvalAdmission(null);
    try std.testing.expectError(
        error.MaterializeEvalRequiresStagedEvaluator,
        validateEvalAdmission("eval-program"),
    );
}

fn printJson(init: std.process.Init, value: anytype) !void {
    const stdout = std.Io.File.stdout();
    var buf: [4096]u8 = undefined;
    var writer = stdout.writer(init.io, &buf);
    try std.json.Stringify.value(value, .{ .whitespace = .indent_2 }, &writer.interface);
    try writer.interface.writeByte('\n');
    try writer.interface.flush();
}

fn usageError() error{InvalidArguments} {
    std.debug.print(
        \\usage: materialize-gemma4-lora <base_model_dir> <adapter_model_dir> <out_dir>
        \\       materialize-gemma4-lora [--eval <program>] <base_model_dir> <adapter_model_dir> <out_dir>
        \\example: materialize-gemma4-lora /tmp/gemma4-base /tmp/gemma4-lora /tmp/gemma4-merged
        \\
        \\--eval is currently rejected: evaluation must target the staged artifact
        \\and succeed before immutable publication.
        \\
    , .{});
    return error.InvalidArguments;
}
