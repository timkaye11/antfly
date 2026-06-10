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

const fused_chunker_data = inference.finetune.fused_chunker_data;
const TokenizerBatch = inference.finetune.tokenizer_batch.TokenizerBatch;
const TokenFnCtx = inference.finetune.tokenizer_batch.TokenFnCtx;

const fnv_offset: u64 = 14695981039346656037;
const fnv_prime: u64 = 1099511628211;

fn hashU64(hash: *u64, value: u64) void {
    var v = value;
    for (0..8) |_| {
        hash.* ^= v & 0xff;
        hash.* *%= fnv_prime;
        v >>= 8;
    }
}

fn hashI32(hash: *u64, value: i32) void {
    hashU64(hash, @as(u32, @bitCast(value)));
}

fn usage() void {
    std.debug.print(
        \\usage: count-fused-tokenization --data <jsonl> --model-dir <dir> [--max-seq-len <n>] [--max-chunks <n>] [--batch-size <n>]
        \\
    , .{});
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    var data_path: ?[]const u8 = null;
    var model_dir: ?[]const u8 = null;
    var max_seq_len: usize = 384;
    var max_chunks: usize = 32;
    var batch_size: usize = 8;
    var dump_first = false;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--data")) {
            data_path = args.next() orelse return error.MissingData;
        } else if (std.mem.eql(u8, arg, "--model-dir")) {
            model_dir = args.next() orelse return error.MissingModelDir;
        } else if (std.mem.eql(u8, arg, "--max-seq-len")) {
            max_seq_len = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingMaxSeqLen, 10);
        } else if (std.mem.eql(u8, arg, "--max-chunks")) {
            max_chunks = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingMaxChunks, 10);
        } else if (std.mem.eql(u8, arg, "--batch-size")) {
            batch_size = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingBatchSize, 10);
        } else if (std.mem.eql(u8, arg, "--dump-first")) {
            dump_first = true;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            usage();
            return;
        } else {
            std.debug.print("unknown argument: {s}\n", .{arg});
            usage();
            return error.InvalidArgument;
        }
    }

    const path = data_path orelse {
        usage();
        return error.MissingData;
    };
    const mdir = model_dir orelse {
        usage();
        return error.MissingModelDir;
    };

    var loaded = try fused_chunker_data.loadSamples(allocator, path, null);
    defer loaded.deinit();

    var tokenizer = try TokenizerBatch.loadFromDir(allocator, mdir, max_seq_len);
    defer tokenizer.deinit();
    var tok_ctx = tokenizer.makeTokenFnCtx();

    var valid: u64 = 0;
    var gold: u64 = 0;
    var batches: u64 = 0;
    var ids_hash: u64 = fnv_offset;
    var mask_hash: u64 = fnv_offset;
    var offsets_hash: u64 = fnv_offset;
    var labels_hash: u64 = fnv_offset;

    var start: usize = 0;
    while (start < loaded.samples.len) {
        const end = @min(start + batch_size, loaded.samples.len);
        const count = end - start;
        const indices = try allocator.alloc(usize, count);
        defer allocator.free(indices);
        for (indices, 0..) |*idx, i| idx.* = start + i;

        var batch = try fused_chunker_data.assembleTokenBatch(
            allocator,
            loaded.samples,
            indices,
            max_seq_len,
            max_chunks,
            &tok_ctx,
            TokenFnCtx.call,
        );
        defer batch.deinit(allocator);

        batches += 1;
        const total = count * max_seq_len;
        for (0..total) |i| {
            hashI32(&ids_hash, batch.input_ids[i]);
            hashI32(&mask_hash, batch.attention_mask[i]);
            hashU64(&labels_hash, if (batch.boundary_labels[i] > 0.5) 1 else 0);
            if (batch.attention_mask[i] != 0) {
                valid += 1;
                if (batch.boundary_labels[i] > 0.5) gold += 1;
            }
        }

        start = end;
    }

    const offsets = try allocator.alloc([2]u32, max_seq_len);
    defer allocator.free(offsets);
    const ids = try allocator.alloc(i32, max_seq_len);
    defer allocator.free(ids);
    const mask = try allocator.alloc(i32, max_seq_len);
    defer allocator.free(mask);

    for (loaded.samples) |sample| {
        @memset(offsets, .{ 0, 0 });
        @memset(ids, 0);
        @memset(mask, 0);
        _ = TokenFnCtx.call(&tok_ctx, sample.text, ids, mask, offsets);
        for (offsets) |off| {
            hashU64(&offsets_hash, off[0]);
            hashU64(&offsets_hash, off[1]);
        }
    }

    const gold_rate = if (valid == 0) 0.0 else @as(f64, @floatFromInt(gold)) / @as(f64, @floatFromInt(valid));
    std.debug.print(
        "zig samples={d} batches={d} valid={d} gold={d} gold_rate={d:.6} ids_hash={x} mask_hash={x} offsets_hash={x} labels_hash={x}\n",
        .{
            loaded.samples.len,
            batches,
            valid,
            gold,
            gold_rate,
            ids_hash,
            mask_hash,
            offsets_hash,
            labels_hash,
        },
    );

    if (dump_first and loaded.samples.len > 0) {
        const one = [_]usize{0};
        var batch = try fused_chunker_data.assembleTokenBatch(
            allocator,
            loaded.samples,
            &one,
            max_seq_len,
            max_chunks,
            &tok_ctx,
            TokenFnCtx.call,
        );
        defer batch.deinit(allocator);

        var valid_first: usize = 0;
        for (batch.attention_mask[0..max_seq_len]) |m| {
            if (m == 0) break;
            valid_first += 1;
        }
        std.debug.print("first valid={d} chunks={d}\n", .{ valid_first, loaded.samples[0].chunk_boundaries.len });
        for (0..@min(loaded.samples[0].chunk_boundaries.len, max_chunks)) |i| {
            const label = if (batch.chunk_starts[i] >= 0 and @as(usize, @intCast(batch.chunk_starts[i])) < max_seq_len)
                batch.boundary_labels[@intCast(batch.chunk_starts[i])]
            else
                0.0;
            std.debug.print(
                "first chunk {d} char=[{d},{d}) token=[{d},{d}) mask={d} label_at_start={d}\n",
                .{
                    i,
                    loaded.samples[0].chunk_boundaries[i].start_char,
                    loaded.samples[0].chunk_boundaries[i].end_char,
                    batch.chunk_starts[i],
                    batch.chunk_ends[i],
                    batch.chunk_mask[i],
                    label,
                },
            );
        }
    }
}
