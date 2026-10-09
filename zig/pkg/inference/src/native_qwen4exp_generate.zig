// Copyright 2026 Antfly, Inc.
// SPDX-License-Identifier: Apache-2.0
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

//! Explicit local CLI entry point; intentionally not advertised as serving support.
const std = @import("std");
const decoder_mod = @import("architectures/qwen4exp_runtime.zig");
const store_mod = @import("models/tensor_store.zig");
const manifest_mod = @import("models/manifest.zig");
const memory = @import("runtime/tier/memory.zig");
const c_file = @import("util/c_file.zig");
const generation = @import("pipelines/generation.zig");
const hf = @import("inference_hf_tokenizer");
const platform = @import("antfly_platform");

pub const Options = struct {
    model: []const u8,
    prompt: []const u8,
    max_tokens: usize = 128,
    context: usize = 4096,
    cache_mb: usize = 2048,
    combined_mb: usize = 0,
    raw: bool = false,
    enable_thinking: ?bool = null,
    ignore_eos: bool = false,
    print_prompt: bool = false,
    print_prompt_ids: bool = false,
    print_ids: bool = false,
    print_count: bool = false,
    print_finish: bool = false,
    print_timing: bool = false,
    print_template: bool = false,
    json_path: ?[]const u8 = null,
    logits_path: ?[]const u8 = null,
};

pub fn run(a: std.mem.Allocator, io: std.Io, opts: Options) !void {
    if (opts.context == 0 or opts.context > 4096 or opts.max_tokens == 0 or opts.max_tokens > opts.context) return error.UnsupportedQwen4ExpContext;
    const start = std.Io.Timestamp.now(io, .awake);
    // A shard path is unambiguous, including the metadata-only first shard.
    const path = if (std.mem.endsWith(u8, opts.model, ".gguf")) try a.dupe(u8, opts.model) else blk: {
        var manifest = try manifest_mod.loadFromDir(a, opts.model);
        defer manifest.deinit();
        break :blk try a.dupe(u8, manifest.gguf_path orelse return error.Qwen4ExpGgufPathRequired);
    };
    defer a.free(path);
    const store = try store_mod.SplitGgufStore.open(a, path);
    defer store.deinit();
    const plan = try decoder_mod.Plan.inspect(store, opts.context, try std.math.mul(usize, opts.cache_mb, 1024 * 1024));
    var controller: memory.AdmissionController = .{};
    var lease = try plan.acquire(&controller, try std.math.mul(usize, opts.combined_mb, 1024 * 1024));
    defer lease.release();
    var sidecar_dir = std.fs.path.dirname(path) orelse ".";
    if (!c_file.fileExistsInDir(a, sidecar_dir, "tokenizer.json")) sidecar_dir = std.fs.path.dirname(sidecar_dir) orelse return error.MissingQwen4ExpTokenizer;
    // Bound tokenizer parsing, vocabulary and caches independently of the
    // caller's allocator (debug allocation metadata can otherwise dwarf the
    // vocabulary). This allocation is included in Plan's host allowance.
    var tokenizer_allocator = @import("runtime/bounded_allocator.zig").BoundedAllocator{
        .backing = std.heap.c_allocator,
        .limit = 512 * 1024 * 1024,
    };
    defer std.debug.assert(tokenizer_allocator.live == 0);
    const tokenizer = blk: {
        const bytes = try c_file.readFileFromDir(a, sidecar_dir, "tokenizer.json");
        defer a.free(bytes);
        break :blk try hf.HfTokenizer.loadFromBytes(tokenizer_allocator.allocator(), bytes);
    };
    defer tokenizer.deinitSelf();
    const template_bytes = try c_file.readFileFromDir(a, sidecar_dir, "tokenizer_config.json");
    defer a.free(template_bytes);
    const template_config = try std.json.parseFromSlice(struct { chat_template: []const u8, bos_token: ?[]const u8 = null, eos_token: ?[]const u8 = null, unk_token: ?[]const u8 = null, pad_token: ?[]const u8 = null }, a, template_bytes, .{ .ignore_unknown_fields = true });
    defer template_config.deinit();
    const tc = template_config.value;
    var template: ?generation.ChatTemplate = null;
    defer if (template) |*t| t.deinit();
    if (!opts.raw) template = try generation.ChatTemplate.init(a, tc.chat_template, tc.bos_token orelse "", tc.eos_token orelse "", tc.unk_token orelse "", tc.pad_token orelse "");
    const prompt = if (template) |*t| try t.applyWithOptions(a, &.{.{ .role = "user", .content = opts.prompt }}, .{ .enable_thinking = opts.enable_thinking }) else try a.dupe(u8, opts.prompt);
    defer a.free(prompt);
    const prompt_ids = try tokenizer.tokenizer().encode(a, prompt);
    defer a.free(prompt_ids);
    if (prompt_ids.len == 0) return error.EmptyPrompt;
    if (prompt_ids.len > opts.context - opts.max_tokens) return error.PromptTooLong;
    for (prompt_ids) |id| if (id < 0 or id >= 248320) return error.InvalidQwen4ExpToken;
    // EOS is a generation-sidecar contract, separate from the ngram reset token.
    const generation_bytes = try c_file.readFileFromDir(a, sidecar_dir, "generation_config.json");
    defer a.free(generation_bytes);
    const config = try std.json.parseFromSlice(struct { eos_token_id: []const u32 }, a, generation_bytes, .{ .ignore_unknown_fields = true });
    defer config.deinit();
    if (opts.print_prompt) std.debug.print("{s}\n", .{prompt});
    if (opts.print_prompt_ids) std.debug.print("prompt_token_ids={any}\n", .{prompt_ids});
    if (opts.print_template) std.debug.print("chat_template={}\n", .{template != null});
    std.debug.print("experimental qwen4exp: context={d}, expert_cache_mb={d}, admitted_bytes={d}\n", .{ opts.context, opts.cache_mb, plan.total() });
    // Command-scoped execution owner. Bounded worker stacks plus executor
    // bookkeeping are reserved in Plan before any workers can be scheduled.
    const reader = decoder_mod.pager;
    var read_executor = std.Io.Threaded.init(a, .{ .async_limit = .limited(reader.read_lane_count), .concurrent_limit = .limited(reader.read_lane_count), .stack_size = reader.read_stack_bytes });
    defer read_executor.deinit();
    const decoder = try decoder_mod.Decoder.create(a, store, plan);
    defer decoder.deinit();
    if (std.c.getenv("ANTFLY_QWEN4EXP_DISABLE_PARALLEL_READS") == null) decoder.read_io = read_executor.io();
    const loaded = std.Io.Timestamp.now(io, .awake);
    const loaded_profile = decoder.kernels.profile;
    var next: u32 = 0;
    if (decoder.prefill_chunk > 0 and prompt_ids.len > 1) {
        var offset: usize = 0;
        while (offset < prompt_ids.len) {
            if (controller.isLiveHostUnderPressure()) return error.ResourceTemporarilyUnavailable;
            const end = @min(prompt_ids.len, offset + decoder.prefill_chunk);
            next = try decoder.prefillChunk(prompt_ids[offset..end]);
            offset = end;
        }
    } else for (prompt_ids, 0..) |id, i| {
        if (controller.isLiveHostUnderPressure()) return error.ResourceTemporarilyUnavailable;
        next = try decoder.prefillToken(@intCast(id), i + 1 == prompt_ids.len);
    }
    const prefilled = std.Io.Timestamp.now(io, .awake);
    const prefilled_profile = decoder.kernels.profile;
    const prefilled_paging = decoder.pagingStats();
    const token_ms = try a.alloc(f64, opts.max_tokens);
    defer a.free(token_ms);
    var timed_steps: usize = 0;
    var output: std.ArrayList(i32) = .empty;
    defer output.deinit(a);
    var reason: []const u8 = "length";
    while (output.items.len < opts.max_tokens) {
        const eos = std.mem.indexOfScalar(u32, config.value.eos_token_id, next) != null;
        if (eos and !opts.ignore_eos) {
            reason = "stop";
            break;
        }
        try output.append(a, @intCast(next));
        if (output.items.len == opts.max_tokens) break;
        // Memory pressure remains observable throughout a long sequence.
        if (controller.isLiveHostUnderPressure()) {
            std.debug.print("Qwen4Exp memory pressure: process={any} live={any}\n", .{ platform.process_memory.pressureSnapshot(), memory.currentSystemMemoryInfoForLimit(controller.process_memory_limit_bytes, controller.process_memory_limit_provenance) });
            return error.ResourceTemporarilyUnavailable;
        }
        const step_start = std.Io.Timestamp.now(io, .awake);
        next = try decoder.step(next);
        token_ms[timed_steps] = @as(f64, @floatFromInt(std.Io.Timestamp.durationTo(step_start, std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e6;
        timed_steps += 1;
    }
    const finished = std.Io.Timestamp.now(io, .awake);
    if (std.c.getenv("ANTFLY_QWEN4EXP_ROUTE_TRACE")) |p| {
        if (decoder.route_trace) |trace| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = std.mem.span(p), .data = std.mem.sliceAsBytes(trace[0 .. decoder.position * 48 * 10]) });
    }
    const text = try tokenizer.tokenizer().decode(a, output.items);
    defer a.free(text);
    std.debug.print("{s}\n", .{text});
    if (opts.print_ids) std.debug.print("token_ids={any}\n", .{output.items});
    if (opts.print_count) std.debug.print("prompt_tokens={d} generated_tokens={d}\n", .{ prompt_ids.len, output.items.len });
    if (opts.print_finish) std.debug.print("finish_reason={s}\n", .{reason});
    if (opts.logits_path) |p| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = decoder.logits.?.bytes[0 .. 248320 * 4] });
    const ms = struct {
        fn between(from: std.Io.Timestamp, to: std.Io.Timestamp) f64 {
            return @as(f64, @floatFromInt(std.Io.Timestamp.durationTo(from, to).nanoseconds)) / 1e6;
        }
    }.between;
    var hits: u64 = 0;
    var misses: u64 = 0;
    var bytes: u64 = 0;
    for (decoder.layers) |layer| {
        hits += layer.stats.hits;
        misses += layer.stats.misses;
        bytes += layer.stats.bytes_read;
    }
    const receipt = .{ .prefill_chunk = decoder.prefill_chunk, .prefill_read_group = 10, .expert_slot_counts = blk: {
        var counts: [48]usize = undefined;
        for (plan.slots, 0..) |count, i| counts[i] = count;
        break :blk counts;
    }, .expert_strides = blk: {
        var strides: [48]usize = undefined;
        for (decoder.layers, 0..) |layer, i| strides[i] = layer.stride;
        break :blk strides;
    }, .grouped_experts = decoder.overlapEnabled() and std.c.getenv("ANTFLY_QWEN4EXP_GROUPED_EXPERTS") != null, .build_mode = @tagName(@import("builtin").mode), .io_overlap = decoder.overlapEnabled(), .prefill_head_skip = std.c.getenv("ANTFLY_QWEN4EXP_DISABLE_PREFILL_HEAD_SKIP") == null, .prefill_expert_io = prefilled_paging, .decode_expert_io = decoder.pagingStats().since(prefilled_paging), .decode_token_ms = token_ms[0..timed_steps], .first_token_ready_ms = ms(start, prefilled), .architecture = "qwen4exp", .experimental = true, .backend = "metal", .metal_command_batching = decoder.kernels.batching_enabled, .metal_residency = decoder.kernels.residency != null, .parallel_expert_reads = decoder.read_io != null, .expert_read_lanes = if (decoder.read_io != null) reader.read_lane_count else 1, .expert_cache_mb = opts.cache_mb, .decode_steps = decoder.position - prompt_ids.len, .context_size = opts.context, .prompt_tokens = prompt_ids.len, .generated_tokens = output.items.len, .load_ms = ms(start, loaded), .prefill_ms = ms(loaded, prefilled), .decode_ms = ms(prefilled, finished), .prefill_profile = prefilled_profile.since(loaded_profile), .decode_profile = decoder.kernels.profile.since(prefilled_profile), .admitted_bytes = plan.total(), .expert_hits = hits, .expert_misses = misses, .expert_bytes_read = bytes, .ngram_bytes_read = decoder.ngrams.?.stats.bytes_read, .tokenizer_live_bytes = tokenizer_allocator.live, .tokenizer_peak_bytes = tokenizer_allocator.peak, .process_memory = platform.process_memory.pressureSnapshot(), .finish_reason = reason, .prompt_token_ids = prompt_ids, .generated_token_ids = output.items };
    if (opts.print_timing) std.debug.print("load_ms={d:.2} prefill_ms={d:.2} decode_ms={d:.2} expert_hits={d} expert_misses={d} expert_bytes_read={d}\n", .{ receipt.load_ms, receipt.prefill_ms, receipt.decode_ms, hits, misses, bytes });
    if (opts.json_path) |p| {
        const json = try std.json.Stringify.valueAlloc(a, receipt, .{ .whitespace = .indent_2 });
        defer a.free(json);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = json });
    }
}

test "Qwen4Exp installed CLI uses exact tokenizer and emits bounded receipts" {
    if (comptime !@import("build_options").enable_metal) return error.SkipZigTest;
    const path = std.c.getenv("ANTFLY_QWEN4EXP_CLI_MODEL") orelse return error.SkipZigTest;
    try run(std.testing.allocator, std.testing.io, .{
        .model = std.mem.span(path),
        .prompt = "Hello",
        .raw = true,
        .max_tokens = 2,
        .context = 4096,
        .combined_mb = 10240,
        .print_ids = true,
        .print_prompt_ids = true,
        .print_timing = true,
        .json_path = if (std.c.getenv("ANTFLY_QWEN4EXP_CLI_RECEIPT")) |value| std.mem.span(value) else null,
        .logits_path = if (std.c.getenv("ANTFLY_QWEN4EXP_LOGITS")) |value| std.mem.span(value) else null,
    });
}
