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
const sidecar = @import("util/verified_sidecar.zig");
const generation = @import("pipelines/generation.zig");
const hf = @import("inference_hf_tokenizer");
const control = @import("util/local_control.zig");
const model = @import("models/qwen4exp.zig");
const profile = @import("runtime/machine_profile.zig");
const verification = @import("util/artifact_verification.zig");
const platform = @import("antfly_platform");

pub const Options = struct {
    model: []const u8,
    prompt: []const u8,
    max_tokens: usize = 128,
    context: usize = profile.max_context,
    /// Zero derives the largest expert cache the machine profile admits.
    cache_mb: usize = 0,
    /// Zero derives the process envelope from the finished plan.
    combined_mb: usize = 0,
    verify: verification.Mode = .full,
    /// Report the memory plan and machine profile, then exit without
    /// verifying or loading weights.
    dry_run: bool = false,
    stream: bool = false,
    session: bool = false,
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

const RunStatus = struct { stage: []const u8 = "arguments", verification: verification.Outcome = .{} };
pub fn run(a: std.mem.Allocator, io: std.Io, opts: Options) !void {
    var guard = control.Guard.init();
    defer guard.deinit();
    var status: RunStatus = .{};
    runInner(a, io, opts, &status) catch |failure| {
        if (opts.json_path) |p| {
            const report = .{ .architecture = "qwen4exp", .experimental = true, .finish_reason = @errorName(failure), .failure_stage = status.stage, .hash_verified = status.verification.hashed, .stamp_verified = status.verification.stamp_accepted, .verification_mode = @tagName(opts.verify), .artifact_revision = model.artifact_revision, .process_memory = platform.process_memory.pressureSnapshot() };
            if (std.json.Stringify.valueAlloc(a, report, .{ .whitespace = .indent_2 })) |json| {
                defer a.free(json);
                std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = json }) catch |err| std.log.err("failure receipt: {s}", .{@errorName(err)});
            } else |_| {}
        }
        return failure;
    };
}
fn runInner(a: std.mem.Allocator, io: std.Io, opts: Options, status: *RunStatus) !void {
    if (opts.context == 0 or opts.context > profile.max_context or opts.max_tokens == 0 or opts.max_tokens > opts.context) return error.UnsupportedQwen4ExpContext;
    const start = std.Io.Timestamp.now(io, .awake);
    // A shard path is unambiguous, including the metadata-only first shard.
    const path = if (std.mem.endsWith(u8, opts.model, ".gguf")) try a.dupe(u8, opts.model) else blk: {
        var manifest = try manifest_mod.loadFromDir(a, opts.model);
        defer manifest.deinit();
        break :blk try a.dupe(u8, manifest.gguf_path orelse return error.Qwen4ExpGgufPathRequired);
    };
    defer a.free(path);
    status.stage = "inventory";
    const store = try store_mod.SplitGgufStore.open(a, path);
    defer store.deinit();
    const host = try profile.Host.detect();
    const plan = try decoder_mod.Plan.inspect(store, host, .{
        .context = opts.context,
        .cache_bytes = try std.math.mul(usize, opts.cache_mb, profile.MiB),
        .cap_bytes = try std.math.mul(usize, opts.combined_mb, profile.MiB),
    });
    if (opts.dry_run) {
        const inventory = try model.Inventory.inspect(store.ggufFile().?, plan.cfg);
        var counts: [48]usize = undefined;
        for (plan.slots, &counts) |count, *out| out.* = count;
        const report = .{
            .model = "Qwen3.8-Flash-Next",
            .architecture = "qwen4exp",
            .artifact_revision = model.artifact_revision,
            .hash_verified = false,
            .context = plan.context,
            .amounts = plan.amounts,
            .admitted_bytes = plan.total(),
            .process_cap_bytes = plan.envelope.cap_bytes,
            .process_cap_source = @tagName(plan.envelope.source),
            .machine_profile = profile.report(host, plan.envelope),
            .expert_cache_bytes = plan.cache_bytes,
            .expert_cache_source = @tagName(plan.cache_source),
            .expert_slot_counts = counts,
            .expert_strides = inventory.per_expert_bytes,
            .experts_per_layer = plan.cfg.experts,
            .resident_expert_fraction = @as(f64, @floatFromInt(plan.cache_bytes)) / @as(f64, @floatFromInt(try inventory.cacheBytes(plan.cfg.experts))),
        };
        const json = try std.json.Stringify.valueAlloc(a, report, .{ .whitespace = .indent_2 });
        defer a.free(json);
        try @import("local_text_session.zig").writeStdout(json);
        try @import("local_text_session.zig").writeStdout("\n");
        if (opts.json_path) |p| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = json });
        return;
    }
    var controller: memory.AdmissionController = .{};
    status.stage = "admission";
    var lease = try plan.acquire(&controller);
    defer lease.release();
    status.stage = "artifact_verification";
    // Hashing and every later weight read share these descriptors.
    try store_mod.SplitGgufStore.bindReaders(store);
    const stamp_dir = verification.defaultStampDir(a);
    defer if (stamp_dir) |dir| a.free(dir);
    std.debug.print("Qwen4Exp: verifying {d} artifact shards ({s})\n", .{ model.artifact_shards.len, @tagName(opts.verify) });
    status.verification = try store_mod.SplitGgufStore.verifyShardIdentity(store, &model.artifact_shards, model.artifact_revision, opts.verify, stamp_dir, &control.cancelled);
    status.stage = "tokenizer";
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
        const bytes = try sidecar.read(a, sidecar_dir, "tokenizer.json", 12809320, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3");
        defer a.free(bytes);
        break :blk try hf.HfTokenizer.loadFromBytes(tokenizer_allocator.allocator(), bytes);
    };
    defer tokenizer.deinitSelf();
    const template_bytes = try sidecar.read(a, sidecar_dir, "tokenizer_config.json", 17928, "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27");
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
    if (!opts.session and prompt_ids.len > opts.context - opts.max_tokens) return error.PromptTooLong;
    for (prompt_ids) |id| if (id < 0 or id >= 248320) return error.InvalidQwen4ExpToken;
    // EOS is a generation-sidecar contract, separate from the ngram reset token.
    const generation_bytes = try sidecar.read(a, sidecar_dir, "generation_config.json", 202, "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e");
    defer a.free(generation_bytes);
    const config = try std.json.parseFromSlice(struct { eos_token_id: []const u32 }, a, generation_bytes, .{ .ignore_unknown_fields = true });
    defer config.deinit();
    if (opts.print_prompt) std.debug.print("{s}\n", .{prompt});
    if (opts.print_prompt_ids) std.debug.print("prompt_token_ids={any}\n", .{prompt_ids});
    if (opts.print_template) std.debug.print("chat_template={}\n", .{template != null});
    std.debug.print("experimental qwen4exp: context={d}, expert_cache_bytes={d} ({s}), process_cap_bytes={d} ({s}), admitted_bytes={d}\n", .{ opts.context, plan.cache_bytes, @tagName(plan.cache_source), plan.envelope.cap_bytes, @tagName(plan.envelope.source), plan.total() });
    // Command-scoped execution owner. Bounded worker stacks plus executor
    // bookkeeping are reserved in Plan before any workers can be scheduled.
    const reader = decoder_mod.pager;
    var read_executor = std.Io.Threaded.init(a, .{ .async_limit = .limited(reader.read_lane_count), .concurrent_limit = .limited(reader.read_lane_count), .stack_size = reader.read_stack_bytes });
    defer read_executor.deinit();
    status.stage = "decoder_initialization";
    try control.check();
    const decoder = try decoder_mod.Decoder.create(a, store, plan);
    defer decoder.deinit();
    decoder.controller = &controller;
    status.stage = if (opts.session) "session" else "prefill";
    if (std.c.getenv("ANTFLY_QWEN4EXP_DISABLE_PARALLEL_READS") == null) decoder.read_io = read_executor.io();
    if (opts.session) {
        const result = @import("local_text_session.zig").run(a, io, decoder, &controller, tokenizer.tokenizer(), if (template) |*t| t else null, .{
            .context = opts.context,
            .max_tokens = opts.max_tokens,
            .raw = opts.raw,
            .thinking = opts.enable_thinking,
            .eos = config.value.eos_token_id,
            .ignore_eos = opts.ignore_eos,
            .model = "Qwen3.8-Flash-Next",
            .artifact_hash_verified = status.verification.hashed,
            .artifact_stamp_verified = status.verification.stamp_accepted,
            .verification_mode = @tagName(status.verification.mode),
            .artifact_id = model.artifact_revision,
            .process_cap_bytes = plan.envelope.cap_bytes,
        });
        // The trace covers the sequence the decoder holds when the session ends.
        writeRouteTrace(io, decoder) catch |err| std.log.err("route trace: {s}", .{@errorName(err)});
        return result;
    }
    const loaded = std.Io.Timestamp.now(io, .awake);
    const loaded_profile = decoder.kernels.profile;
    std.debug.print("Qwen4Exp: prefilling {d} tokens\n", .{prompt_ids.len});
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
    status.stage = "decode";
    std.debug.print("Qwen4Exp: decoding\n", .{});
    const prefilled = std.Io.Timestamp.now(io, .awake);
    const prefilled_profile = decoder.kernels.profile;
    const prefilled_paging = decoder.pagingStats();
    const token_ms = try a.alloc(f64, opts.max_tokens);
    defer a.free(token_ms);
    var timed_steps: usize = 0;
    var output: std.ArrayList(i32) = .empty;
    defer output.deinit(a);
    var reason: []const u8 = "length";
    var printed: usize = 0;
    while (output.items.len < opts.max_tokens) {
        try control.check();
        const eos = std.mem.indexOfScalar(u32, config.value.eos_token_id, next) != null;
        if (eos and !opts.ignore_eos) {
            reason = "stop";
            break;
        }
        try output.append(a, @intCast(next));
        if (opts.stream) {
            const text = try tokenizer.tokenizer().decode(a, output.items);
            defer a.free(text);
            const end = @import("local_text_session.zig").completeUtf8Prefix(text);
            if (end > printed) {
                try @import("local_text_session.zig").writeStdout(text[printed..end]);
                printed = end;
            }
        }
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
    status.stage = "output";
    const finished = std.Io.Timestamp.now(io, .awake);
    try writeRouteTrace(io, decoder);
    const text = try tokenizer.tokenizer().decode(a, output.items);
    defer a.free(text);
    try @import("local_text_session.zig").writeStdout(text[if (opts.stream) printed else 0..]);
    try @import("local_text_session.zig").writeStdout("\n");
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
    const receipt = .{ .hash_verified = status.verification.hashed, .stamp_verified = status.verification.stamp_accepted, .verification_mode = @tagName(status.verification.mode), .process_cap_bytes = plan.envelope.cap_bytes, .process_cap_source = @tagName(plan.envelope.source), .machine_profile = profile.report(plan.host, plan.envelope), .expert_cache_bytes = decoder.expertCacheBytes(), .expert_cache_source = @tagName(plan.cache_source), .expert_cache_requested_mb = if (opts.cache_mb == 0) @as(?usize, null) else opts.cache_mb, .expert_read_retries = decoder.pagingStats().read_retries, .artifact_revision = model.artifact_revision, .artifact_shards = model.artifact_shards, .sidecars_hash_verified = true, .prefill_mps = decoder.prefill_tile != null, .batch_linear = decoder.batch_linear, .prefill_chunk = decoder.prefill_chunk, .prefill_read_group = 10, .expert_slot_counts = blk: {
        var counts: [48]usize = undefined;
        for (decoder.layers, 0..) |layer, i| counts[i] = layer.slotCount();
        break :blk counts;
    }, .expert_strides = blk: {
        var strides: [48]usize = undefined;
        for (decoder.layers, 0..) |layer, i| strides[i] = layer.stride;
        break :blk strides;
    }, .grouped_experts = decoder.overlapEnabled() and std.c.getenv("ANTFLY_QWEN4EXP_GROUPED_EXPERTS") != null, .build_mode = @tagName(@import("builtin").mode), .io_overlap = decoder.overlapEnabled(), .prefill_head_skip = std.c.getenv("ANTFLY_QWEN4EXP_DISABLE_PREFILL_HEAD_SKIP") == null, .prefill_expert_io = prefilled_paging, .decode_expert_io = decoder.pagingStats().since(prefilled_paging), .decode_token_ms = token_ms[0..timed_steps], .first_token_ready_ms = ms(start, prefilled), .architecture = "qwen4exp", .experimental = true, .backend = "metal", .metal_command_batching = decoder.kernels.batching_enabled, .metal_residency = decoder.kernels.residency != null, .parallel_expert_reads = decoder.read_io != null, .expert_read_lanes = if (decoder.read_io != null) reader.read_lane_count else 1, .decode_steps = decoder.position - prompt_ids.len, .context_size = opts.context, .prompt_tokens = prompt_ids.len, .generated_tokens = output.items.len, .load_ms = ms(start, loaded), .prefill_ms = ms(loaded, prefilled), .decode_ms = ms(prefilled, finished), .prefill_profile = prefilled_profile.since(loaded_profile), .decode_profile = decoder.kernels.profile.since(prefilled_profile), .admitted_bytes = plan.total(), .expert_hits = hits, .expert_misses = misses, .expert_bytes_read = bytes, .ngram_bytes_read = decoder.ngrams.?.stats.bytes_read, .tokenizer_live_bytes = tokenizer_allocator.live, .tokenizer_peak_bytes = tokenizer_allocator.peak, .process_memory = platform.process_memory.pressureSnapshot(), .finish_reason = reason, .prompt_token_ids = prompt_ids, .generated_token_ids = output.items };
    if (opts.print_timing) std.debug.print("load_ms={d:.2} prefill_ms={d:.2} decode_ms={d:.2} expert_hits={d} expert_misses={d} expert_bytes_read={d}\n", .{ receipt.load_ms, receipt.prefill_ms, receipt.decode_ms, hits, misses, bytes });
    if (opts.json_path) |p| {
        const json = try std.json.Stringify.valueAlloc(a, receipt, .{ .whitespace = .indent_2 });
        defer a.free(json);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = json });
    }
}

fn writeRouteTrace(io: std.Io, decoder: *decoder_mod.Decoder) !void {
    const path = std.c.getenv("ANTFLY_QWEN4EXP_ROUTE_TRACE") orelse return;
    const trace = decoder.route_trace orelse return;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = std.mem.span(path), .data = std.mem.sliceAsBytes(trace[0 .. decoder.position * 48 * 10]) });
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
        .cache_mb = 2048,
        .combined_mb = 10240,
        .print_ids = true,
        .print_prompt_ids = true,
        .print_timing = true,
        .json_path = if (std.c.getenv("ANTFLY_QWEN4EXP_CLI_RECEIPT")) |value| std.mem.span(value) else null,
        .logits_path = if (std.c.getenv("ANTFLY_QWEN4EXP_LOGITS")) |value| std.mem.span(value) else null,
    });
}
