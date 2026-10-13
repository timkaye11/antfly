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

//! Opt-in local text generation; no general serving compatibility promotion.
const std = @import("std");
const model = @import("models/qwen35_streamed.zig");
const decoder_mod = @import("architectures/qwen35_streamed.zig");
const memory = @import("runtime/tier/memory.zig");
const c_file = @import("util/c_file.zig");
const generation = @import("pipelines/generation.zig");
const hf = @import("inference_hf_tokenizer");
const profile = @import("runtime/machine_profile.zig");
const verification = @import("util/artifact_verification.zig");
const platform = @import("antfly_platform");
pub const Options = struct {
    model: []const u8,
    prompt: []const u8,
    max_tokens: usize = 128,
    context: usize = profile.max_context,
    chunk: usize = 512,
    /// Zero derives the process envelope from the finished plan.
    combined_mb: usize = 0,
    verify: verification.Mode = .full,
    dry_run: bool = false,
    stream: bool = false,
    session: bool = false,
    prompt_lookup: usize = 0,
    raw: bool = false,
    enable_thinking: bool = false,
    ignore_eos: bool = false,
    print_prompt: bool = false,
    print_prompt_ids: bool = false,
    print_ids: bool = false,
    print_count: bool = false,
    print_finish: bool = false,
    print_timing: bool = false,
    print_template: bool = false,
    json_path: ?[]const u8 = null,
};
const control = @import("util/local_control.zig");
const cancelled = &control.cancelled;
fn writeStdout(bytes: []const u8) !void {
    var offset: usize = 0;
    while (offset < bytes.len) {
        const n = std.c.write(1, bytes.ptr + offset, bytes.len - offset);
        if (n <= 0) return error.OutputWriteFailed;
        offset += @intCast(n);
    }
}
fn elapsed(io: std.Io, start: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(std.Io.Timestamp.durationTo(start, std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e6;
}
const RunStatus = struct { receipt_written: bool = false, receipt_reason: []const u8 = "", verification: verification.Outcome = .{}, stage: []const u8 = "arguments" };
pub fn run(a: std.mem.Allocator, io: std.Io, opts: Options) !void {
    var guard = control.Guard.init();
    defer guard.deinit();
    var status: RunStatus = .{};
    runInner(a, io, opts, &status) catch |failure| {
        // Preserve a detailed execution failure, but replace a success receipt
        // if streaming output or another unanticipated operation failed.
        if (!status.receipt_written or !std.mem.eql(u8, status.receipt_reason, @errorName(failure))) if (opts.json_path) |p| {
            const report = .{ .model = "Qwen3.8-27B", .experimental = true, .execution = "streamed_layers", .sha256 = model.artifact_sha256, .hash_verified = status.verification.hashed, .stamp_verified = status.verification.stamp_accepted, .verification_mode = @tagName(opts.verify), .finish_reason = @errorName(failure), .failure_stage = status.stage, .process_memory = platform.process_memory.pressureSnapshot() };
            if (std.json.Stringify.valueAlloc(a, report, .{ .whitespace = .indent_2 })) |json| {
                defer a.free(json);
                std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = json }) catch {};
            } else |_| {}
        };
        return failure;
    };
    if (opts.json_path != null and !opts.session and !opts.dry_run and !status.receipt_written) return error.ReceiptWriteFailed;
}
fn runInner(a: std.mem.Allocator, io: std.Io, opts: Options, status: *RunStatus) !void {
    if (opts.context == 0 or opts.context > profile.max_context or opts.max_tokens == 0 or opts.max_tokens > opts.context) return error.UnsupportedQwen35Context;
    const start = std.Io.Timestamp.now(io, .awake);
    const path = opts.model;
    status.stage = "inventory";
    var source = try model.Source.open(a, path);
    defer source.deinit();
    const host = try profile.Host.detect();
    const plan = try decoder_mod.Plan.inspect(&source, host, .{
        .context = opts.context,
        .chunk = opts.chunk,
        .lookup = opts.prompt_lookup > 0,
        .cap_bytes = try std.math.mul(usize, opts.combined_mb, model.MiB),
    });
    const envelope: profile.Envelope = .{ .cap_bytes = plan.cap, .source = plan.cap_source };
    if (opts.dry_run) {
        const report = .{ .model = "Qwen3.8-27B", .architecture = "qwen35", .sha256 = model.artifact_sha256, .hash_verified = false, .file_bytes = source.reader.size, .main_layers = 64, .excluded_mtp_layers = 1, .context = plan.context, .prefill_chunk = plan.chunk, .amounts = plan.amounts, .admitted_bytes = plan.total(), .cap_bytes = plan.cap, .process_cap_bytes = plan.cap, .process_cap_source = @tagName(plan.cap_source), .machine_profile = profile.report(host, envelope), .layer_spans = source.layers };
        const json = try std.json.Stringify.valueAlloc(a, report, .{ .whitespace = .indent_2 });
        defer a.free(json);
        try writeStdout(json);
        try writeStdout("\n");
        if (opts.json_path) |p| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = json });
        return;
    }
    var controller: memory.AdmissionController = .{};
    status.stage = "admission";
    var admission = try plan.acquire(&controller);
    defer admission.release();
    std.debug.print("Qwen3.8-27B: verifying exact artifact with bounded reads ({s})...\n", .{@tagName(opts.verify)});
    status.stage = "artifact_verification";
    const stamp_dir = verification.defaultStampDir(a);
    defer if (stamp_dir) |dir| a.free(dir);
    try source.verify(path, opts.verify, stamp_dir, cancelled);
    status.verification = source.verification;
    status.stage = "tokenizer";
    var sidecar_dir = std.fs.path.dirname(path) orelse ".";
    if (!c_file.fileExistsInDir(a, sidecar_dir, "tokenizer.json")) sidecar_dir = std.fs.path.dirname(sidecar_dir) orelse return error.MissingQwen35Tokenizer;
    // Bound tokenizer parsing, vocabulary and caches independently of the
    // caller's allocator (debug allocation metadata can otherwise dwarf the
    // vocabulary). This allocation is included in Plan's host allowance.
    var tokenizer_allocator = @import("runtime/bounded_allocator.zig").BoundedAllocator{
        .backing = std.heap.c_allocator,
        .limit = 512 * 1024 * 1024,
    };
    defer std.debug.assert(tokenizer_allocator.live == 0);
    const tokenizer = blk: {
        const bytes = try readSidecar(a, sidecar_dir, "tokenizer.json", 12809320, "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3");
        defer a.free(bytes);
        break :blk try hf.HfTokenizer.loadFromBytes(tokenizer_allocator.allocator(), bytes);
    };
    defer tokenizer.deinitSelf();
    const template_bytes = try readSidecar(a, sidecar_dir, "tokenizer_config.json", 17928, "b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27");
    defer a.free(template_bytes);
    const template_config = try std.json.parseFromSlice(struct { chat_template: []const u8, bos_token: ?[]const u8 = null, eos_token: ?[]const u8 = null, unk_token: ?[]const u8 = null, pad_token: ?[]const u8 = null }, a, template_bytes, .{ .ignore_unknown_fields = true });
    defer template_config.deinit();
    const tc = template_config.value;
    var template: ?generation.ChatTemplate = null;
    defer if (template) |*t| t.deinit();
    if (!opts.raw) template = try generation.ChatTemplate.init(a, tc.chat_template, tc.bos_token orelse "", tc.eos_token orelse "", tc.unk_token orelse "", tc.pad_token orelse "");
    const prompt = if (template) |*t| try t.applyWithOptions(a, &.{.{ .role = "user", .content = opts.prompt }}, .{ .enable_thinking = @as(?bool, opts.enable_thinking) }) else try a.dupe(u8, opts.prompt);
    defer a.free(prompt);
    const prompt_ids = try tokenizer.tokenizer().encode(a, prompt);
    defer a.free(prompt_ids);
    if (prompt_ids.len == 0) return error.EmptyPrompt;
    if (!opts.session and prompt_ids.len > opts.context - opts.max_tokens) return error.PromptTooLong;
    for (prompt_ids) |id| if (id < 0 or id >= 248320) return error.InvalidQwen35Token;
    // EOS is a generation-sidecar contract, separate from the ngram reset token.
    const generation_bytes = try readSidecar(a, sidecar_dir, "generation_config.json", 202, "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e");
    defer a.free(generation_bytes);
    const config = try std.json.parseFromSlice(struct { eos_token_id: []const u32 }, a, generation_bytes, .{ .ignore_unknown_fields = true });
    defer config.deinit();
    if (opts.print_prompt) std.debug.print("{s}\n", .{prompt});
    if (opts.print_prompt_ids) std.debug.print("prompt_token_ids={any}\n", .{prompt_ids});
    if (opts.print_template) std.debug.print("chat_template={}\n", .{template != null});

    var executor = std.Io.Threaded.init(a, .{ .async_limit = .limited(1), .concurrent_limit = .limited(1), .stack_size = 256 * 1024 });
    defer executor.deinit();
    status.stage = "decoder_initialization";
    const decoder = try decoder_mod.Decoder.create(a, &source, plan, &controller, cancelled);
    defer decoder.deinit();
    if (std.c.getenv("ANTFLY_QWEN35_DISABLE_IO_OVERLAP") == null) decoder.io = executor.io();
    if (opts.session) status.stage = "session";
    if (opts.session) return @import("local_text_session.zig").run(a, io, decoder, &controller, tokenizer.tokenizer(), if (template) |*t| t else null, .{
        .context = opts.context,
        .max_tokens = opts.max_tokens,
        .raw = opts.raw,
        .thinking = opts.enable_thinking,
        .eos = config.value.eos_token_id,
        .ignore_eos = opts.ignore_eos,
        .model = "Qwen3.8-27B",
        .artifact_hash_verified = source.verification.hashed,
        .artifact_stamp_verified = source.verification.stamp_accepted,
        .verification_mode = @tagName(source.verification.mode),
        .artifact_id = model.artifact_sha256,
        .process_cap_bytes = plan.cap,
        .prompt_lookup = opts.prompt_lookup,
    });
    const load_ms = elapsed(io, start);
    var output: std.ArrayList(i32) = .empty;
    defer output.deinit(a);
    var step_ms: std.ArrayList(f64) = .empty;
    defer step_ms.deinit(a);
    var prefill_ms: f64 = 0;
    var decode_ms: f64 = 0;
    var prefill_bytes: u64 = 0;
    var prefill_cpu_ns: u64 = 0;
    var prefill_gpu_wait_ns: u64 = 0;
    var prefill_gpu_ns: u64 = 0;
    var prefill_batches: u64 = 0;
    var prefill_read_wait_ns: u64 = 0;
    var reason: []const u8 = "error";
    defer {
        const report = .{
            .model = "Qwen3.8-27B",
            .architecture = "qwen35",
            .execution = "streamed_layers",
            .experimental = true,
            .build_mode = @tagName(@import("builtin").mode),
            .sha256 = model.artifact_sha256,
            .hash_verified = source.verification.hashed,
            .stamp_verified = source.verification.stamp_accepted,
            .verification_mode = @tagName(source.verification.mode),
            .backend = "metal",
            .cpu_weight_dequantization = !decoder.kernels.gpu_dequant,
            .context_size = plan.context,
            .prefill_chunk = plan.chunk,
            .io_overlap = decoder.io != null,
            .admitted_bytes = plan.total(),
            .cap_bytes = plan.cap,
            .process_cap_bytes = plan.cap,
            .process_cap_source = @tagName(plan.cap_source),
            .machine_profile = profile.report(host, envelope),
            .load_ms = load_ms,
            .prefill_ms = prefill_ms,
            .decode_ms = decode_ms,
            .first_token_ready_ms = if (prefill_ms > 0 and !std.mem.eql(u8, status.stage, "prefill")) @as(?f64, load_ms + prefill_ms) else null,
            .decode_steps = step_ms.items.len,
            .decode_token_ms = step_ms.items,
            .prefill_bytes_read = if (std.mem.eql(u8, status.stage, "prefill")) decoder.bytesRead() else prefill_bytes,
            .decode_bytes_read = if (std.mem.eql(u8, status.stage, "prefill")) 0 else decoder.bytesRead() - prefill_bytes,
            .embedding_bytes_read = decoder.embedding_bytes_read,
            .cpu_dequant_bytes = decoder.kernels.cpu_dequant_bytes,
            .cpu_dequant_ns = decoder.kernels.cpu_dequant_ns,
            .profile_enabled = decoder.kernels.profile_enabled,
            .prefill_profile = .{ .cpu_dequant_ns = prefill_cpu_ns, .gpu_wait_ns = prefill_gpu_wait_ns, .gpu_ns = prefill_gpu_ns, .batches = prefill_batches, .read_wait_ns = prefill_read_wait_ns },
            .decode_profile = .{ .cpu_dequant_ns = decoder.kernels.cpu_dequant_ns - prefill_cpu_ns, .gpu_wait_ns = decoder.kernels.gpu_wait_ns - prefill_gpu_wait_ns, .gpu_ns = decoder.kernels.gpu_ns - prefill_gpu_ns, .batches = decoder.kernels.gpu_batches - prefill_batches, .read_wait_ns = decoder.read_wait_ns - prefill_read_wait_ns },
            .prompt_lookup = opts.prompt_lookup,
            .lookup_stats = decoder.lookup_stats,
            .process_memory = platform.process_memory.pressureSnapshot(),
            .prompt_token_ids = prompt_ids,
            .generated_token_ids = output.items,
            .generated_tokens = output.items.len,
            .finish_reason = reason,
            .failure_stage = if (std.mem.eql(u8, reason, "stop") or std.mem.eql(u8, reason, "length")) @as(?[]const u8, null) else status.stage,
            .last_layer = decoder.current_layer,
            .position = decoder.position,
        };
        if (opts.json_path) |p| {
            if (std.json.Stringify.valueAlloc(a, report, .{ .whitespace = .indent_2 })) |json| {
                defer a.free(json);
                if (std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = json })) |_| {
                    status.receipt_written = true;
                    status.receipt_reason = reason;
                } else |err| std.log.err("receipt: {s}", .{@errorName(err)});
            } else |err| std.log.err("receipt: {s}", .{@errorName(err)});
        }
        if (opts.print_timing) std.debug.print("load_ms={d:.2} prefill_ms={d:.2} decode_ms={d:.2} bytes_read={d}\n", .{ load_ms, prefill_ms, decode_ms, decoder.bytesRead() });
    }
    std.debug.print("Qwen3.8-27B: admitted={d} MiB, context={d}, chunk={d}; prefilling {d} tokens\n", .{ plan.total() / model.MiB, plan.context, plan.chunk, prompt_ids.len });
    status.stage = "prefill";
    var next: u32 = 0;
    const prefill_start = std.Io.Timestamp.now(io, .awake);
    // This defer runs before the receipt defer. A cancelled/failed prefill
    // must retain its elapsed time and counters in the prefill phase.
    defer if (std.mem.eql(u8, status.stage, "prefill")) {
        prefill_ms = elapsed(io, prefill_start);
        prefill_bytes = decoder.bytesRead();
        prefill_cpu_ns = decoder.kernels.cpu_dequant_ns;
        prefill_gpu_wait_ns = decoder.kernels.gpu_wait_ns;
        prefill_gpu_ns = decoder.kernels.gpu_ns;
        prefill_batches = decoder.kernels.gpu_batches;
        prefill_read_wait_ns = decoder.read_wait_ns;
    };
    var at: usize = 0;
    while (at < prompt_ids.len) {
        const end = @min(at + plan.chunk, prompt_ids.len);
        next = decoder.prefill(prompt_ids[at..end]) catch |err| {
            reason = @errorName(err);
            return err;
        };
        at = end;
        std.debug.print("prefill {d}/{d}\n", .{ at, prompt_ids.len });
    }
    prefill_ms = elapsed(io, prefill_start);
    prefill_bytes = decoder.bytesRead();
    prefill_cpu_ns = decoder.kernels.cpu_dequant_ns;
    prefill_gpu_wait_ns = decoder.kernels.gpu_wait_ns;
    prefill_gpu_ns = decoder.kernels.gpu_ns;
    prefill_batches = decoder.kernels.gpu_batches;
    prefill_read_wait_ns = decoder.read_wait_ns;
    status.stage = "decode";
    std.debug.print("Qwen35: decoding\n", .{});
    const decode_start = std.Io.Timestamp.now(io, .awake);
    defer decode_ms = elapsed(io, decode_start);
    // Sized from the admitted context, never from a fixed sequence limit.
    var history: std.ArrayList(i32) = .empty;
    defer history.deinit(a);
    if (opts.prompt_lookup > 0) try history.ensureTotalCapacity(a, opts.context);
    var printed: usize = 0;
    while (output.items.len < opts.max_tokens) {
        if (!opts.ignore_eos and std.mem.indexOfScalar(u32, config.value.eos_token_id, next) != null) {
            reason = "stop";
            break;
        }
        try output.append(a, @intCast(next));
        if (opts.stream) {
            const text = try tokenizer.tokenizer().decode(a, output.items);
            defer a.free(text);
            // The byte-level decoder returns raw bytes; retain incomplete tails.
            const end = printed + completeUtf8Prefix(text[printed..]);
            if (end > printed) {
                try writeStdout(text[printed..end]);
                printed = end;
            }
        }
        if (output.items.len == opts.max_tokens) {
            reason = "length";
            break;
        }
        const tick = std.Io.Timestamp.now(io, .awake);
        if (opts.prompt_lookup > 0) {
            history.clearRetainingCapacity();
            try history.appendSlice(a, prompt_ids);
            try history.appendSlice(a, output.items);
            const advance = decoder.advancePromptLookup(history.items, @min(opts.prompt_lookup, opts.max_tokens - output.items.len), if (opts.ignore_eos) &.{} else config.value.eos_token_id) catch |err| {
                reason = @errorName(err);
                return err;
            };
            next = advance.next;
            try output.appendSlice(a, advance.accepted[0..advance.count]);
            if (opts.stream and advance.count > 0) {
                const accepted_text = try tokenizer.tokenizer().decode(a, output.items);
                defer a.free(accepted_text);
                const end = printed + completeUtf8Prefix(accepted_text[printed..]);
                try writeStdout(accepted_text[printed..end]);
                printed = end;
            }
            if (output.items.len == opts.max_tokens) reason = "length";
        } else {
            next = decoder.prefill(&.{@intCast(next)}) catch |err| {
                reason = @errorName(err);
                return err;
            };
        }
        try step_ms.append(a, elapsed(io, tick));
    }
    status.stage = "output";
    const text = try tokenizer.tokenizer().decode(a, output.items);
    defer a.free(text);
    try writeStdout(text[if (opts.stream) printed else 0..]);
    try writeStdout("\n");
    if (opts.print_ids) std.debug.print("token_ids={any}\n", .{output.items});
    if (opts.print_count) std.debug.print("prompt_tokens={d} generated_tokens={d}\n", .{ prompt_ids.len, output.items.len });
    if (opts.print_finish) std.debug.print("finish_reason={s}\n", .{reason});
}

fn readSidecar(a: std.mem.Allocator, dir: []const u8, name: []const u8, size: usize, expected: []const u8) ![]u8 {
    return @import("util/verified_sidecar.zig").read(a, dir, name, size, expected) catch |err| switch (err) {
        error.SidecarSizeMismatch => error.Qwen35SidecarSizeMismatch,
        error.SidecarHashMismatch => error.Qwen35SidecarHashMismatch,
        else => err,
    };
}

fn completeUtf8Prefix(bytes: []const u8) usize {
    var at: usize = 0;
    while (at < bytes.len) {
        const n = std.unicode.utf8ByteSequenceLength(bytes[at]) catch {
            at += 1;
            continue;
        };
        if (bytes.len - at < n) return at;
        _ = std.unicode.utf8Decode(bytes[at..][0..n]) catch {
            at += 1;
            continue;
        };
        at += n;
    }
    return at;
}
test "Qwen35 streaming holds incomplete UTF8 but emits complete replacement characters" {
    try std.testing.expectEqual(@as(usize, 1), completeUtf8Prefix("a\xe2\x82"));
    try std.testing.expectEqual(@as(usize, 4), completeUtf8Prefix("a\xe2\x82\xac"));
    try std.testing.expectEqual(@as(usize, 3), completeUtf8Prefix("\xef\xbf\xbd"));
}
