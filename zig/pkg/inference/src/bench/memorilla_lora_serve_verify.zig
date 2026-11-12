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

//! Drives the actual managed research service (`memorilla_research_service`)
//! through a real `ModelManager` over the Python reference cases and reports
//! answer agreement, admission refunds and per-case latency. Companion to
//! `memorilla_lora_verify` (stage-level parity) for milestone 2 of
//! docs/plans/memorilla-copy17-native.md.
const std = @import("std");
const inference = @import("inference_internal");
const options_mod = @import("build_options");
const service = inference.server.memorilla_research_service;
const common = inference.server.memorilla_service;
const manager_mod = inference.server.model_manager;
const qualified = inference.models.memorilla_qualification;
const memory = inference.runtime.tier.memory;
const Control = inference.InferenceExecutionControl;
const file = inference.util.c_file;
const A = std.mem.Allocator;

const Args = struct {
    encoder: []const u8 = "",
    decoder: []const u8 = "",
    bundle: []const u8 = "",
    reference: []const u8 = "",
    output: []const u8 = "",
    max_cases: usize = 0,
    budget_gib: usize = 96,
    max_new_tokens: usize = 24,
};

fn parse(a: A, init: std.process.Init) !Args {
    const args = try init.minimal.args.toSlice(a);
    defer a.free(args);
    var result = Args{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (i + 1 >= args.len) return error.MissingArgument;
        const flag = args[i];
        i += 1;
        const value = try a.dupe(u8, args[i]);
        if (std.mem.eql(u8, flag, "--encoder")) {
            result.encoder = value;
        } else if (std.mem.eql(u8, flag, "--decoder")) {
            result.decoder = value;
        } else if (std.mem.eql(u8, flag, "--bundle")) {
            result.bundle = value;
        } else if (std.mem.eql(u8, flag, "--reference")) {
            result.reference = value;
        } else if (std.mem.eql(u8, flag, "--output")) {
            result.output = value;
        } else if (std.mem.eql(u8, flag, "--max-cases")) {
            result.max_cases = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--budget-gib")) {
            result.budget_gib = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--max-new-tokens")) {
            result.max_new_tokens = try std.fmt.parseInt(usize, value, 10);
        } else return error.UnknownArgument;
    }
    if (result.encoder.len == 0 or result.decoder.len == 0 or result.bundle.len == 0 or result.reference.len == 0 or
        result.budget_gib == 0 or result.budget_gib > 112 or result.max_new_tokens == 0 or result.max_new_tokens > 128) return error.InvalidArguments;
    return result;
}

const Case = struct {
    name: []const u8,
    documents: []const []const u8,
    question: []const u8,
    expected_answer: []const u8 = "",
    answer: []const u8,
};

const Reference = struct {
    schema: []const u8,
    cases: []const Case,
};

fn emit(a: A, io: std.Io, value: anytype) !void {
    const bytes = try std.json.Stringify.valueAlloc(a, value, .{});
    defer a.free(bytes);
    try std.Io.File.stdout().writeStreamingAll(io, bytes);
    try std.Io.File.stdout().writeStreamingAll(io, "\n");
}

fn elapsedMs(io: std.Io, started: std.Io.Timestamp) f64 {
    const finished = std.Io.Timestamp.now(io, .awake);
    const nanoseconds = started.durationTo(finished).nanoseconds;
    return if (nanoseconds <= 0) 0 else @as(f64, @floatFromInt(nanoseconds)) / 1_000_000.0;
}

const Stages = struct {
    prepared_ms: f64 = 0,
    encoder_loaded_ms: f64 = 0,
    question_encoded_ms: f64 = 0,
    decoder_loaded_ms: f64 = 0,
    reader_loaded_ms: f64 = 0,
    decoded_ms: f64 = 0,
};

const Context = struct {
    io: std.Io,
    started: std.Io.Timestamp,
    stages: Stages = .{},

    fn stage(raw: ?*anyopaque, which: common.Stage, _: *manager_mod.ModelManager) anyerror!void {
        const self: *Context = @ptrCast(@alignCast(raw.?));
        const at = elapsedMs(self.io, self.started);
        switch (which) {
            .prepared => self.stages.prepared_ms = at,
            .encoder_loaded => self.stages.encoder_loaded_ms = at,
            .question_encoded => self.stages.question_encoded_ms = at,
            .decoder_loaded => self.stages.decoder_loaded_ms = at,
            .reader_loaded => self.stages.reader_loaded_ms = at,
            .decoded => self.stages.decoded_ms = at,
        }
    }
};

const Summary = struct {
    event: []const u8 = "complete",
    cases: usize = 0,
    errors: usize = 0,
    answer_matches_reference: usize = 0,
    answer_correct: usize = 0,
    reference_correct: usize = 0,
    admission_refunded: usize = 0,
    cold_ms: f64 = 0,
    mean_warm_ms: f64 = 0,
    passed: bool = false,
};

pub fn main(init: std.process.Init) !void {
    if (comptime !options_mod.enable_cuda) return error.CudaNotEnabled;
    const a = std.heap.c_allocator;
    const io = init.io;
    const args = try parse(a, init);
    const cap = try std.math.mul(usize, args.budget_gib, 1024 * 1024 * 1024);
    const limits = memory.Limits{ .host_limit_bytes = cap, .backend_limit_bytes = cap, .combined_limit_bytes = cap, .kv_limit_bytes = 2 * 1024 * 1024 * 1024, .scratch_limit_bytes = 12 * 1024 * 1024 * 1024 };

    const receipt_path = try std.fs.path.join(a, &.{ args.reference, "reference.json" });
    defer a.free(receipt_path);
    const receipt_bytes = try file.readFileMax(a, receipt_path, 64 * 1024 * 1024);
    defer a.free(receipt_bytes);
    var parsed = try std.json.parseFromSlice(Reference, a, receipt_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    if (parsed.value.cases.len == 0) return error.InvalidReference;

    const selection = inference.model_selection;
    // `fromDirectory` adopts the resolved path; free it once through `selected_*.path`.
    const encoder_path = try std.Io.Dir.cwd().realPathFileAlloc(io, args.encoder, a);
    const selected_encoder = try selection.fromDirectory(a, io, encoder_path);
    defer a.free(selected_encoder.path);
    const decoder_path = try std.Io.Dir.cwd().realPathFileAlloc(io, args.decoder, a);
    const selected_decoder = try selection.fromDirectory(a, io, decoder_path);
    defer a.free(selected_decoder.path);
    const bundle_path = try std.Io.Dir.cwd().realPathFileAlloc(io, args.bundle, a);
    defer a.free(bundle_path);
    const paths = common.Paths{ .encoder = selected_encoder, .decoder = selected_decoder };
    const profile = common.Profile{
        .name = "research-copy17",
        .encoder = selected_encoder.path,
        .decoder = selected_decoder.path,
        .memory_module_path = bundle_path,
        .max_new_tokens = 128,
        .reader_kind = .encoder_lora_research,
    };
    const identity = qualified.executionIdentity(service.encoder_profile);

    var sessions = inference.backends.SessionManager.initWithIo(a, io);
    sessions.preferred_backends = &.{.cuda};
    sessions.required_backend = .cuda;
    sessions.native_call_policy = .in_process;
    var models = manager_mod.ModelManager.init(a, sessions);
    defer models.deinit();
    models.configureServingPolicy(.{});
    models.configureAdmissionLimits(limits);
    models.configureProcessMemoryLimit(cap, .explicit);
    models.configureModelCache(600_000, 4);
    try models.ensureResourceOwnerReady();

    try emit(a, io, .{
        .event = "ready",
        .scope = "actual-managed-memorilla-research-service",
        .cases = parsed.value.cases.len,
        .encoder_publication = selected_encoder.publication != null,
        .decoder_publication = selected_decoder.publication != null,
        .operator_cap_bytes = cap,
        .encoder_execution_identity = identity[0..],
        .max_new_tokens = args.max_new_tokens,
    });

    var summary = Summary{};
    var warm_total: f64 = 0;
    var warm_count: usize = 0;
    var retained: ?memory.AdmissionAmounts = null;
    const limit = if (args.max_cases == 0) parsed.value.cases.len else @min(args.max_cases, parsed.value.cases.len);
    for (parsed.value.cases[0..limit], 0..) |case, index| {
        summary.cases += 1;
        var context = Context{ .io = io, .started = std.Io.Timestamp.now(io, .awake) };
        const control = Control{ .io = io, .native_call_policy = .in_process };
        const options = common.Options{ .context = &context, .stage = Context.stage };
        var result = service.generate(a, io, &models, profile, paths, case.question, case.documents, &identity, args.max_new_tokens, limits, .{}, control, options) catch |err| {
            summary.errors += 1;
            try emit(a, io, .{ .event = "case_error", .name = case.name, .@"error" = @errorName(err), .stages_ms = context.stages });
            continue;
        };
        var released = false;
        defer if (!released) result.deinit();
        const total_ms = elapsedMs(io, context.started);
        const answer = std.mem.trim(u8, result.generation.text, " \t\r\n");
        const reference_answer = std.mem.trim(u8, case.answer, " \t\r\n");
        const matches = std.mem.eql(u8, answer, reference_answer);
        const correct = std.mem.eql(u8, answer, case.expected_answer);
        const reference_correct = std.mem.eql(u8, reference_answer, case.expected_answer);
        summary.answer_matches_reference += @intFromBool(matches);
        summary.answer_correct += @intFromBool(correct);
        summary.reference_correct += @intFromBool(reference_correct);
        if (index == 0) summary.cold_ms = total_ms else {
            warm_total += total_ms;
            warm_count += 1;
        }
        try emit(a, io, .{
            .event = "case",
            .name = case.name,
            .documents = case.documents.len,
            .answer = answer,
            .reference_answer = reference_answer,
            .expected_answer = case.expected_answer,
            .answer_matches_reference = matches,
            .answer_correct = correct,
            .reference_correct = reference_correct,
            .finish_reason = result.generation.finish_reason,
            .question_tokens = result.question_tokens,
            .candidate_sha256 = result.candidate_sha256[0..],
            .memory_module_sha256 = result.module_sha256[0..],
            .total_ms = total_ms,
            .stages_ms = context.stages,
        });
        // Every request must retire its own admission; what remains is the
        // resident encoder, decoder and reader.
        result.deinit();
        released = true;
        const now = models.currentAdmissionAmounts();
        if (retained) |expected| {
            summary.admission_refunded += @intFromBool(std.meta.eql(expected, now));
        } else {
            retained = now;
            summary.admission_refunded += 1;
        }
    }
    summary.mean_warm_ms = if (warm_count == 0) 0 else warm_total / @as(f64, @floatFromInt(warm_count));
    summary.passed = summary.errors == 0 and summary.admission_refunded == summary.cases;
    try emit(a, io, .{ .event = "retained", .amounts = retained });
    try emit(a, io, summary);
    if (args.output.len > 0) {
        const json = try std.json.Stringify.valueAlloc(a, summary, .{ .whitespace = .indent_2 });
        defer a.free(json);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args.output, .data = json });
    }
    if (!summary.passed) return error.ResearchServiceVerificationFailed;
}

test "memorilla lora serve verify parses stage timings" {
    var context = Context{ .io = std.testing.io, .started = std.Io.Timestamp.now(std.testing.io, .awake) };
    var manager: manager_mod.ModelManager = undefined;
    try Context.stage(&context, .decoded, &manager);
    try std.testing.expect(context.stages.decoded_ms >= 0);
}
