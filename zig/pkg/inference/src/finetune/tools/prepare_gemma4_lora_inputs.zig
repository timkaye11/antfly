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
const gemma_chat_data = inference.finetune.gemma_chat_data;

pub const Options = struct {
    model_dir: []const u8,
    dataset_path: []const u8,
    split: ?[]const u8,
    out_path: []const u8,
    max_examples: usize = 0,
    max_seq_len: usize = 512,
    gguf_projector_path: ?[]const u8 = null,
    dataset_revision: ?[]const u8 = null,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    const model_dir = args.next() orelse return usageError();
    const dataset_path = args.next() orelse return usageError();
    const split_arg = args.next() orelse return usageError();
    const out_path = args.next() orelse return usageError();

    var max_examples: usize = 0;
    var max_seq_len: usize = 512;
    var gguf_projector_path: ?[]const u8 = null;
    var dataset_revision: ?[]const u8 = null;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--max-examples")) {
            const val = args.next() orelse return usageError();
            max_examples = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--max-seq-len")) {
            const val = args.next() orelse return usageError();
            max_seq_len = try std.fmt.parseUnsigned(usize, val, 10);
        } else if (std.mem.eql(u8, arg, "--gguf-projector")) {
            gguf_projector_path = args.next() orelse return usageError();
        } else if (std.mem.eql(u8, arg, "--dataset-revision")) {
            dataset_revision = args.next() orelse return usageError();
        } else {
            return usageError();
        }
    }

    const split = if (std.mem.eql(u8, split_arg, "-")) null else split_arg;

    const options = Options{
        .model_dir = model_dir,
        .dataset_path = dataset_path,
        .split = split,
        .out_path = out_path,
        .max_examples = max_examples,
        .max_seq_len = max_seq_len,
        .gguf_projector_path = gguf_projector_path,
        .dataset_revision = dataset_revision,
    };

    var summary = try prepare(allocator, options);
    defer finetune.freePreparedInputsSummary(allocator, &summary);
    try finetune.savePreparedInputsSummary(allocator, out_path, summary);

    const stdout = std.Io.File.stdout();
    var buf: [4096]u8 = undefined;
    var writer = stdout.writer(init.io, &buf);
    try std.json.Stringify.value(summary, .{ .whitespace = .indent_2 }, &writer.interface);
    try writer.interface.writeByte('\n');
    try writer.interface.flush();

    std.debug.print("examples_written: {d}\n", .{summary.examples_seen});
    std.debug.print("saved_summary: {s}\n", .{out_path});
}

/// Typed entrypoint used by recipes and workflows. It deliberately owns no
/// process-global argv state and publishes the prepared artifact immutably.
pub fn run(allocator: std.mem.Allocator, io: std.Io, options: Options) !void {
    _ = io;
    var summary = try prepare(allocator, options);
    defer finetune.freePreparedInputsSummary(allocator, &summary);
    try finetune.savePreparedInputsSummary(allocator, options.out_path, summary);
}

/// Prepare without publishing. The caller owns the returned summary and must
/// release it with `finetune.freePreparedInputsSummary`.
pub fn prepare(allocator: std.mem.Allocator, options: Options) !finetune.PreparedInputsSummary {
    if (options.model_dir.len == 0 or options.dataset_path.len == 0 or options.out_path.len == 0) {
        return error.InvalidArguments;
    }

    var loaded = try gemma_chat_data.loadExamples(allocator, options.dataset_path, options.split);
    defer loaded.deinit();

    const has_multimodal = messagesHaveMedia(loaded.examples);
    if (has_multimodal and options.gguf_projector_path == null) return error.MissingGgufProjector;
    const source = finetune.PreparedSourceIdentity{
        .dataset_path = options.dataset_path,
        .split = options.split,
        .revision = options.dataset_revision,
    };
    return if (has_multimodal)
        try finetune.prepareMultimodalInputsFromChatDataWithSource(
            allocator,
            options.model_dir,
            options.gguf_projector_path.?,
            loaded.examples,
            options.max_examples,
            options.max_seq_len,
            source,
        )
    else
        try finetune.prepareInputsFromChatDataWithSource(
            allocator,
            options.model_dir,
            loaded.examples,
            options.max_examples,
            options.max_seq_len,
            source,
        );
}

fn usageError() error{InvalidArguments} {
    std.debug.print(
        \\usage: prepare-gemma4-lora-inputs <model_dir> <dataset_path> <split|-> <out_summary_json> [options]
        \\
        \\  <model_dir>         Path to Gemma4 model directory (must contain tokenizer.json)
        \\  <dataset_path>      JSONL file, directory, or manifest.json
        \\  <split|->           Dataset split to filter (e.g. "train") or "-" for all
        \\  <out_summary_json>  Output path for prepared inputs JSON
        \\
        \\Options:
        \\  --max-examples N    Maximum number of examples to prepare (default: 0 = all)
        \\  --max-seq-len N     Maximum sequence length in tokens (default: 512)
        \\  --gguf-projector P  Required when the dataset contains image/audio parts; path to Gemma4 projector GGUF
        \\  --dataset-revision R Immutable dataset revision; defaults to the resolved split digest
        \\
        \\example: prepare-gemma4-lora-inputs /tmp/gemma4-base /tmp/data.jsonl train /tmp/gemma4_inputs.json --max-examples 256 --max-seq-len 512
        \\
    , .{});
    return error.InvalidArguments;
}

fn messagesHaveMedia(examples: []const gemma_chat_data.Example) bool {
    for (examples) |example| {
        if (example.image_paths.len > 0 or example.audio_paths.len > 0) return true;
    }
    return false;
}
