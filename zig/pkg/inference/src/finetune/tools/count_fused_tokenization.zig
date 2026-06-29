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
        \\usage: count-fused-tokenization --data <jsonl> --model-dir <dir> [--split <name>] [--max-seq-len <n>] [--max-chunks <n>] [--batch-size <n>] [--offset <n>] [--limit <n>] [--json] [--dump-first] [--dump-batch]
        \\
    , .{});
}

fn printI32Array(values: []const i32) void {
    std.debug.print("[", .{});
    for (values, 0..) |v, i| {
        if (i > 0) std.debug.print(",", .{});
        std.debug.print("{d}", .{v});
    }
    std.debug.print("]", .{});
}

fn printUsizeArray(values: []const usize) void {
    std.debug.print("[", .{});
    for (values, 0..) |v, i| {
        if (i > 0) std.debug.print(",", .{});
        std.debug.print("{d}", .{v});
    }
    std.debug.print("]", .{});
}

fn printUsizeArrayWithOffset(values: []const usize, offset: usize) void {
    std.debug.print("[", .{});
    for (values, 0..) |v, i| {
        if (i > 0) std.debug.print(",", .{});
        std.debug.print("{d}", .{offset + v});
    }
    std.debug.print("]", .{});
}

fn printBoundaryPositions(labels: []const f32, mask: []const i32) void {
    std.debug.print("[", .{});
    var first = true;
    for (labels, 0..) |label, i| {
        if (i >= mask.len or mask[i] == 0) break;
        if (label <= 0.5) continue;
        if (!first) std.debug.print(",", .{});
        first = false;
        std.debug.print("{d}", .{i});
    }
    std.debug.print("]", .{});
}

fn printChunkSpans(starts: []const i32, ends: []const i32, mask: []const f32) void {
    std.debug.print("[", .{});
    for (starts, 0..) |start, i| {
        if (i > 0) std.debug.print(",", .{});
        const end = if (i < ends.len) ends[i] else 0;
        const m: u8 = if (i < mask.len and mask[i] > 0.5) 1 else 0;
        std.debug.print("{{\"idx\":{d},\"start\":{d},\"end\":{d},\"mask\":{d}}}", .{ i, start, end, m });
    }
    std.debug.print("]", .{});
}

fn printSourceChunkBoundaries(chunks: []const fused_chunker_data.FusedChunkBoundary) void {
    std.debug.print("[", .{});
    for (chunks, 0..) |chunk, i| {
        if (i > 0) std.debug.print(",", .{});
        std.debug.print(
            "{{\"idx\":{d},\"start_char\":{d},\"end_char\":{d},\"start_token\":{d},\"end_token\":{d}}}",
            .{ i, chunk.start_char, chunk.end_char, chunk.start_token, chunk.end_token },
        );
    }
    std.debug.print("]", .{});
}

fn printSourceTokenMappings(chunks: []const fused_chunker_data.FusedChunkBoundary, offsets: [][2]u32) void {
    std.debug.print("[", .{});
    for (chunks, 0..) |chunk, i| {
        if (i > 0) std.debug.print(",", .{});
        const span = fused_chunker_data.charToTokenBoundary(chunk.start_char, chunk.end_char, offsets);
        const resolved_start = if (chunk.start_token > 0) chunk.start_token else span.start_token;
        const resolved_end = if (chunk.end_token > 0) chunk.end_token else span.end_token;
        const valid: u8 = if (resolved_end > resolved_start) 1 else 0;
        const prev_start = if (span.end_token > 0 and span.end_token - 1 < offsets.len) offsets[span.end_token - 1][0] else 0;
        const prev_end = if (span.end_token > 0 and span.end_token - 1 < offsets.len) offsets[span.end_token - 1][1] else 0;
        const end_start = if (span.end_token < offsets.len) offsets[span.end_token][0] else 0;
        const end_end = if (span.end_token < offsets.len) offsets[span.end_token][1] else 0;
        std.debug.print(
            "{{\"idx\":{d},\"start_char\":{d},\"end_char\":{d},\"raw_start\":{d},\"raw_end\":{d},\"resolved_start\":{d},\"resolved_end\":{d},\"valid\":{d},\"prev_off\":[{d},{d}],\"end_off\":[{d},{d}]}}",
            .{
                i,
                chunk.start_char,
                chunk.end_char,
                span.start_token,
                span.end_token,
                resolved_start,
                resolved_end,
                valid,
                prev_start,
                prev_end,
                end_start,
                end_end,
            },
        );
    }
    std.debug.print("]", .{});
}

fn validTokenCount(mask: []const i32) usize {
    var n: usize = 0;
    for (mask) |m| {
        if (m == 0) break;
        n += 1;
    }
    return n;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;

    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, allocator);
    defer args.deinit();
    _ = args.next();

    var data_path: ?[]const u8 = null;
    var model_dir: ?[]const u8 = null;
    var split: ?[]const u8 = null;
    var max_seq_len: usize = 384;
    var max_chunks: usize = 32;
    var batch_size: usize = 8;
    var sample_offset: usize = 0;
    var limit: usize = 0;
    var dump_first = false;
    var dump_batch = false;
    var json_output = false;

    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--data")) {
            data_path = args.next() orelse return error.MissingData;
        } else if (std.mem.eql(u8, arg, "--model-dir")) {
            model_dir = args.next() orelse return error.MissingModelDir;
        } else if (std.mem.eql(u8, arg, "--split")) {
            split = args.next() orelse return error.MissingSplit;
        } else if (std.mem.eql(u8, arg, "--max-seq-len")) {
            max_seq_len = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingMaxSeqLen, 10);
        } else if (std.mem.eql(u8, arg, "--max-chunks")) {
            max_chunks = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingMaxChunks, 10);
        } else if (std.mem.eql(u8, arg, "--batch-size")) {
            batch_size = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingBatchSize, 10);
        } else if (std.mem.eql(u8, arg, "--offset")) {
            sample_offset = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingOffset, 10);
        } else if (std.mem.eql(u8, arg, "--limit")) {
            limit = try std.fmt.parseUnsigned(usize, args.next() orelse return error.MissingLimit, 10);
        } else if (std.mem.eql(u8, arg, "--dump-first")) {
            dump_first = true;
        } else if (std.mem.eql(u8, arg, "--dump-batch")) {
            dump_batch = true;
        } else if (std.mem.eql(u8, arg, "--json")) {
            json_output = true;
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

    var loaded = try fused_chunker_data.loadSamples(allocator, path, split);
    defer loaded.deinit();
    if (sample_offset > loaded.samples.len) return error.InvalidOffset;
    const sample_end = if (limit > 0)
        @min(sample_offset + limit, loaded.samples.len)
    else
        loaded.samples.len;
    const samples = loaded.samples[sample_offset..sample_end];
    const dataset_stats = fused_chunker_data.computeStats(samples);

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
    var chunks_hash: u64 = fnv_offset;
    var sample_indices_hash: u64 = fnv_offset;
    var samples_with_active_boundary_labels: u64 = 0;

    var start: usize = 0;
    while (start < samples.len) {
        const end = @min(start + batch_size, samples.len);
        const count = end - start;
        const indices = try allocator.alloc(usize, count);
        defer allocator.free(indices);
        for (indices, 0..) |*idx, i| idx.* = start + i;
        for (indices) |idx| hashU64(&sample_indices_hash, sample_offset + idx);

        var batch = try fused_chunker_data.assembleTokenBatch(
            allocator,
            samples,
            indices,
            max_seq_len,
            max_chunks,
            &tok_ctx,
            TokenFnCtx.call,
        );
        defer batch.deinit(allocator);

        batches += 1;
        const total = count * max_seq_len;
        var batch_gold_by_sample = try allocator.alloc(u64, count);
        defer allocator.free(batch_gold_by_sample);
        @memset(batch_gold_by_sample, 0);
        for (0..total) |i| {
            hashI32(&ids_hash, batch.input_ids[i]);
            hashI32(&mask_hash, batch.attention_mask[i]);
            hashU64(&labels_hash, if (batch.boundary_labels[i] > 0.5) 1 else 0);
            if (batch.attention_mask[i] != 0) {
                valid += 1;
                if (batch.boundary_labels[i] > 0.5) {
                    gold += 1;
                    batch_gold_by_sample[i / max_seq_len] += 1;
                }
            }
        }
        for (batch_gold_by_sample) |sample_gold| {
            if (sample_gold > 0) samples_with_active_boundary_labels += 1;
        }
        for (0..count * max_chunks) |i| {
            hashI32(&chunks_hash, batch.chunk_starts[i]);
            hashI32(&chunks_hash, batch.chunk_ends[i]);
            hashU64(&chunks_hash, if (batch.chunk_mask[i] > 0.5) 1 else 0);
        }

        start = end;
    }

    const offsets = try allocator.alloc([2]u32, max_seq_len);
    defer allocator.free(offsets);
    const ids = try allocator.alloc(i32, max_seq_len);
    defer allocator.free(ids);
    const mask = try allocator.alloc(i32, max_seq_len);
    defer allocator.free(mask);

    for (samples) |sample| {
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
    if (json_output) {
        std.debug.print(
            "{{\"tool\":\"zig_fused_tokenization_parity\",\"schema_version\":2,\"offset\":{d},\"samples\":{d},\"batches\":{d},\"max_seq_len\":{d},\"max_chunks\":{d},\"valid_tokens\":{d},\"boundary_gold_tokens\":{d},\"gold_rate\":{d:.9},\"samples_with_active_boundary_labels\":{d},\"contrastive_pos_samples\":{d},\"boundary_target_samples\":{d},\"boundary_targets\":{d},\"hashes\":{{\"sample_indices\":\"{x}\",\"input_ids\":\"{x}\",\"attention_mask\":\"{x}\",\"offsets\":\"{x}\",\"labels\":\"{x}\",\"chunks\":\"{x}\"}}",
            .{
                sample_offset,
                samples.len,
                batches,
                max_seq_len,
                max_chunks,
                valid,
                gold,
                gold_rate,
                samples_with_active_boundary_labels,
                dataset_stats.samples_with_contrastive_positives,
                dataset_stats.samples_with_boundary_targets,
                dataset_stats.total_boundary_targets,
                sample_indices_hash,
                ids_hash,
                mask_hash,
                offsets_hash,
                labels_hash,
                chunks_hash,
            },
        );
    } else {
        std.debug.print(
            "zig offset={d} samples={d} batches={d} valid={d} gold={d} gold_rate={d:.6} active_boundary_samples={d} contrastive_pos_samples={d} boundary_target_samples={d} boundary_targets={d} sample_indices_hash={x} ids_hash={x} mask_hash={x} offsets_hash={x} labels_hash={x} chunks_hash={x}\n",
            .{
                sample_offset,
                samples.len,
                batches,
                valid,
                gold,
                gold_rate,
                samples_with_active_boundary_labels,
                dataset_stats.samples_with_contrastive_positives,
                dataset_stats.samples_with_boundary_targets,
                dataset_stats.total_boundary_targets,
                sample_indices_hash,
                ids_hash,
                mask_hash,
                offsets_hash,
                labels_hash,
                chunks_hash,
            },
        );
    }

    if (dump_first and samples.len > 0) {
        const one = [_]usize{0};
        var batch = try fused_chunker_data.assembleTokenBatch(
            allocator,
            samples,
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
        if (json_output) {
            std.debug.print(",\"first_sample\":{{\"valid_tokens\":{d},\"chunks\":[", .{valid_first});
            for (0..@min(samples[0].chunk_boundaries.len, max_chunks)) |i| {
                if (i > 0) std.debug.print(",", .{});
                const label = if (batch.chunk_starts[i] >= 0 and @as(usize, @intCast(batch.chunk_starts[i])) < max_seq_len)
                    batch.boundary_labels[@intCast(batch.chunk_starts[i])]
                else
                    0.0;
                std.debug.print(
                    "{{\"idx\":{d},\"start_char\":{d},\"end_char\":{d},\"start_token\":{d},\"end_token\":{d},\"mask\":{d},\"label_at_start\":{d}}}",
                    .{
                        i,
                        samples[0].chunk_boundaries[i].start_char,
                        samples[0].chunk_boundaries[i].end_char,
                        batch.chunk_starts[i],
                        batch.chunk_ends[i],
                        batch.chunk_mask[i],
                        label,
                    },
                );
            }
            std.debug.print("],\"boundary_positions\":[", .{});
            var first_position = true;
            for (0..valid_first) |i| {
                if (batch.boundary_labels[i] <= 0.5) continue;
                if (!first_position) std.debug.print(",", .{});
                first_position = false;
                std.debug.print("{d}", .{i});
            }
            std.debug.print("]}}", .{});
        } else {
            std.debug.print("first valid={d} chunks={d}\n", .{ valid_first, samples[0].chunk_boundaries.len });
            for (0..@min(samples[0].chunk_boundaries.len, max_chunks)) |i| {
                const label = if (batch.chunk_starts[i] >= 0 and @as(usize, @intCast(batch.chunk_starts[i])) < max_seq_len)
                    batch.boundary_labels[@intCast(batch.chunk_starts[i])]
                else
                    0.0;
                std.debug.print(
                    "first chunk {d} char=[{d},{d}) token=[{d},{d}) mask={d} label_at_start={d}\n",
                    .{
                        i,
                        samples[0].chunk_boundaries[i].start_char,
                        samples[0].chunk_boundaries[i].end_char,
                        batch.chunk_starts[i],
                        batch.chunk_ends[i],
                        batch.chunk_mask[i],
                        label,
                    },
                );
            }
        }
    }
    if (dump_batch and samples.len > 0) {
        const count = @min(batch_size, samples.len);
        const indices = try allocator.alloc(usize, count);
        defer allocator.free(indices);
        for (indices, 0..) |*idx, i| idx.* = i;
        var batch = try fused_chunker_data.assembleTokenBatch(
            allocator,
            samples,
            indices,
            max_seq_len,
            max_chunks,
            &tok_ctx,
            TokenFnCtx.call,
        );
        defer batch.deinit(allocator);

        if (json_output) {
            std.debug.print(",\"first_batch\":{{\"sample_indices\":", .{});
            printUsizeArrayWithOffset(batch.sample_indices, sample_offset);
            std.debug.print(",\"samples\":[", .{});
            for (0..count) |bi| {
                if (bi > 0) std.debug.print(",", .{});
                const seq_start = bi * max_seq_len;
                const seq_end = seq_start + max_seq_len;
                const chunk_start = bi * max_chunks;
                const chunk_end = chunk_start + max_chunks;
                const ids_slice = batch.input_ids[seq_start..seq_end];
                const mask_slice = batch.attention_mask[seq_start..seq_end];
                const labels_slice = batch.boundary_labels[seq_start..seq_end];
                std.debug.print("{{\"sample_index\":{d},\"valid_tokens\":{d},\"input_ids\":", .{
                    sample_offset + batch.sample_indices[bi],
                    validTokenCount(mask_slice),
                });
                printI32Array(ids_slice);
                std.debug.print(",\"attention_mask\":", .{});
                printI32Array(mask_slice);
                std.debug.print(",\"boundary_positions\":", .{});
                printBoundaryPositions(labels_slice, mask_slice);
                std.debug.print(",\"source_chunk_boundaries\":", .{});
                printSourceChunkBoundaries(samples[batch.sample_indices[bi]].chunk_boundaries);
                std.debug.print(",\"chunk_spans\":", .{});
                printChunkSpans(
                    batch.chunk_starts[chunk_start..chunk_end],
                    batch.chunk_ends[chunk_start..chunk_end],
                    batch.chunk_mask[chunk_start..chunk_end],
                );
                std.debug.print("}}", .{});
            }
            std.debug.print("]}}", .{});
            std.debug.print(",\"debug_source_token_mappings\":[", .{});
            for (0..count) |bi| {
                if (bi > 0) std.debug.print(",", .{});
                const sample_index = batch.sample_indices[bi];
                @memset(offsets, .{ 0, 0 });
                @memset(ids, 0);
                @memset(mask, 0);
                const n_tokens = TokenFnCtx.call(&tok_ctx, samples[sample_index].text, ids, mask, offsets);
                const active_offsets = offsets[0..@min(n_tokens, max_seq_len)];
                std.debug.print("{{\"sample_index\":{d},\"mappings\":", .{sample_offset + sample_index});
                printSourceTokenMappings(samples[sample_index].chunk_boundaries, active_offsets);
                std.debug.print("}}", .{});
            }
            std.debug.print("]", .{});
        } else {
            std.debug.print("first_batch sample_indices=", .{});
            printUsizeArrayWithOffset(batch.sample_indices, sample_offset);
            std.debug.print("\n", .{});
            for (0..count) |bi| {
                const seq_start = bi * max_seq_len;
                const chunk_start = bi * max_chunks;
                std.debug.print("batch sample {d} valid={d} boundary_positions=", .{
                    sample_offset + batch.sample_indices[bi],
                    validTokenCount(batch.attention_mask[seq_start .. seq_start + max_seq_len]),
                });
                printBoundaryPositions(
                    batch.boundary_labels[seq_start .. seq_start + max_seq_len],
                    batch.attention_mask[seq_start .. seq_start + max_seq_len],
                );
                std.debug.print(" chunks=", .{});
                printChunkSpans(
                    batch.chunk_starts[chunk_start .. chunk_start + max_chunks],
                    batch.chunk_ends[chunk_start .. chunk_start + max_chunks],
                    batch.chunk_mask[chunk_start .. chunk_start + max_chunks],
                );
                std.debug.print("\n", .{});
            }
        }
    }
    if (json_output) std.debug.print("}}\n", .{});
}
