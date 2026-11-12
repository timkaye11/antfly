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

//! Offline end-to-end parity tool for the encoder-LoRA research candidate
//! (per-token dual-layer features, copy head, snapped contiguous spans). It
//! runs the strict-F32 CUDA pipeline on every fixture case and compares each
//! stage against the Python reference written by
//! `scripts/memorilla/encoder_lora_export.py reference`. Nothing here changes
//! serving policy: the merged encoder loads through the diagnostic reference
//! session, the reader bundle is a non-release research bundle.
const std = @import("std");
const internal = @import("inference_internal");
const options = @import("build_options");
const factory = internal.architectures.session_factory;
const gpt = internal.architectures.gpt;
const model = internal.models.memorilla;
const reader_pipeline = internal.pipelines.memorilla;
const decode = internal.pipelines.memorilla_generation;
const generation = internal.pipelines.generation;
const copy = internal.pipelines.memorilla_copy;
const features_mod = internal.pipelines.memorilla_token_features;
const memorilla_encoder = internal.pipelines.memorilla_encoder;
const safe = internal.models.safetensors;
const manifest_mod = internal.models.manifest;
const qualification = internal.models.memorilla_qualification;
const memory = internal.runtime.tier.memory;
const Tensor = internal.backends.Tensor;
const ops = internal.ops;
const A = std.mem.Allocator;

const slots: usize = 16;

const Args = struct {
    candidate: []const u8 = "",
    reader_bundle: []const u8 = "",
    encoder: []const u8 = "",
    decoder: []const u8 = "",
    reference: []const u8 = "",
    output: []const u8 = "",
    max_cases: usize = 0,
    skip_decoder: bool = false,
    /// Feed the fixture's document rows, query and copy ids to the reader
    /// instead of running the encoder: isolates reader/copy/decoder parity.
    reference_features: bool = false,
    /// Feed the fixture's memories to the copy head instead of the reader's.
    reference_memories: bool = false,
    atol: f64 = 2e-2,
    rtol: f64 = 2e-2,
    /// Bridge margins of the contiguous-run rule to evaluate per case
    /// (`copy.contiguousRunsWithMargin`); stages 1-3 run once, stages 4-6
    /// once per margin. The entry for margin 0 (or the first entry when 0 is
    /// absent) fills the original per-case and summary fields.
    copy_bridge_margins: []const f64 = &.{0},
};

fn parseMargins(a: A, text: []const u8) ![]const f64 {
    var list = std.ArrayListUnmanaged(f64).empty;
    errdefer list.deinit(a);
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |piece| {
        const trimmed = std.mem.trim(u8, piece, " \t");
        if (trimmed.len == 0) continue;
        const margin = try std.fmt.parseFloat(f64, trimmed);
        if (!(margin >= 0 and margin <= 1)) return error.InvalidArguments;
        for (list.items) |seen| if (seen == margin) return error.InvalidArguments;
        try list.append(a, margin);
    }
    if (list.items.len == 0) return error.InvalidArguments;
    return list.toOwnedSlice(a);
}

/// Index of the margin whose results fill the legacy single-margin fields.
fn primaryMargin(margins: []const f64) usize {
    for (margins, 0..) |margin, i| if (margin == 0) return i;
    return 0;
}

fn parseArgs(a: A, init: std.process.Init) !Args {
    const args = try init.minimal.args.toSlice(a);
    defer a.free(args);
    var result = Args{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        if (std.mem.eql(u8, flag, "--skip-decoder")) {
            result.skip_decoder = true;
            continue;
        }
        if (std.mem.eql(u8, flag, "--reference-features")) {
            result.reference_features = true;
            continue;
        }
        if (std.mem.eql(u8, flag, "--reference-memories")) {
            result.reference_memories = true;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingArgument;
        i += 1;
        const value = try a.dupe(u8, args[i]);
        if (std.mem.eql(u8, flag, "--candidate")) {
            result.candidate = value;
        } else if (std.mem.eql(u8, flag, "--reader-bundle")) {
            result.reader_bundle = value;
        } else if (std.mem.eql(u8, flag, "--encoder")) {
            result.encoder = value;
        } else if (std.mem.eql(u8, flag, "--decoder")) {
            result.decoder = value;
        } else if (std.mem.eql(u8, flag, "--reference")) {
            result.reference = value;
        } else if (std.mem.eql(u8, flag, "--output")) {
            result.output = value;
        } else if (std.mem.eql(u8, flag, "--max-cases")) {
            result.max_cases = try std.fmt.parseInt(usize, value, 10);
        } else if (std.mem.eql(u8, flag, "--atol")) {
            result.atol = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, flag, "--rtol")) {
            result.rtol = try std.fmt.parseFloat(f64, value);
        } else if (std.mem.eql(u8, flag, "--copy-bridge-margins")) {
            result.copy_bridge_margins = try parseMargins(a, value);
        } else return error.UnknownArgument;
    }
    if (result.candidate.len == 0 or result.reader_bundle.len == 0 or result.reference.len == 0 or
        (!result.reference_features and result.encoder.len == 0) or result.decoder.len == 0 or
        !std.math.isFinite(result.atol) or !std.math.isFinite(result.rtol) or result.atol < 0 or result.rtol < 0)
        return error.InvalidArguments;
    return result;
}

const Case = struct {
    name: []const u8,
    documents: []const []const u8,
    question: []const u8,
    expected_answer: []const u8 = "",
    copy_ids: []const i64,
    spans: []const [3]i64,
    snapped_ids: []const i64,
    completed_ids: []const i64,
    prompt_ids: []const i64,
    output_ids: []const i64,
    answer: []const u8,
};

const Reference = struct {
    schema: []const u8,
    dual_layer: usize = 24,
    system_prompt: []const u8,
    pad_id: i32 = 220,
    max_new_tokens: usize = 24,
    cases: []const Case,
    tensors: struct { path: []const u8 = "reference.safetensors", sha256: []const u8 = "" } = .{},
};

const expected_schema = "antfly.memorilla.encoder-lora-pipeline-reference.v1";

fn emit(a: A, io: std.Io, value: anytype) !void {
    const bytes = try std.json.Stringify.valueAlloc(a, value, .{});
    defer a.free(bytes);
    try std.Io.File.stdout().writeStreamingAll(io, bytes);
    try std.Io.File.stdout().writeStreamingAll(io, "\n");
}

const Stat = struct {
    elements: usize = 0,
    mismatches: usize = 0,
    max_absolute_difference: f64 = 0,
    relative_l2_error: f64 = 0,
    /// Minimum per-row cosine between actual and expected rows (1 = identical direction).
    min_row_cosine: f64 = 1,
    passed: bool = true,
};

fn compare(actual: []const f32, expected: []const f32, width: usize, atol: f64, rtol: f64) !Stat {
    if (actual.len != expected.len or actual.len == 0 or width == 0 or actual.len % width != 0) return error.ShapeMismatch;
    var stat = Stat{ .elements = actual.len };
    var squared_error: f64 = 0;
    var squared_reference: f64 = 0;
    for (actual, expected) |x, y| {
        if (!std.math.isFinite(x) or !std.math.isFinite(y)) return error.NonFiniteValue;
        const difference = @abs(@as(f64, x) - y);
        if (difference > atol + rtol * @abs(@as(f64, y))) stat.mismatches += 1;
        stat.max_absolute_difference = @max(stat.max_absolute_difference, difference);
        squared_error += difference * difference;
        squared_reference += @as(f64, y) * y;
    }
    stat.relative_l2_error = if (squared_reference > 0) @sqrt(squared_error / squared_reference) else @sqrt(squared_error);
    var row: usize = 0;
    while (row < actual.len) : (row += width) {
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        for (actual[row..][0..width], expected[row..][0..width]) |x, y| {
            dot += @as(f64, x) * y;
            na += @as(f64, x) * x;
            nb += @as(f64, y) * y;
        }
        const cosine = if (na > 0 and nb > 0) dot / (@sqrt(na) * @sqrt(nb)) else if (na == 0 and nb == 0) @as(f64, 1) else @as(f64, 0);
        stat.min_row_cosine = @min(stat.min_row_cosine, cosine);
    }
    stat.passed = stat.mismatches == 0;
    return stat;
}

fn idsEqual(actual: []const i32, expected: []const i64) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |x, y| if (@as(i64, x) != y) return false;
    return true;
}

fn idsEqualUnsigned(actual: []const u32, expected: []const i64) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |x, y| if (@as(i64, x) != y) return false;
    return true;
}

fn idsEqualWide(actual: []const i64, expected: []const i64) bool {
    return std.mem.eql(i64, actual, expected);
}

fn spansEqual(actual: []const copy.Span, expected: []const [3]i64) bool {
    if (actual.len != expected.len) return false;
    for (actual, expected) |span, triple| {
        if (@as(i64, @intCast(span.begin)) != triple[0] or @as(i64, @intCast(span.end)) != triple[1] or @as(i64, @intCast(span.start)) != triple[2]) return false;
    }
    return true;
}

/// Offline tools run every native call in-process; serving supervises them.
fn inProcessControl(io: std.Io) internal.InferenceExecutionControl {
    return .{ .io = io, .native_call_policy = .in_process };
}

fn elapsedMs(io: std.Io, started: std.Io.Timestamp) f64 {
    const finished = std.Io.Timestamp.now(io, .awake);
    const nanoseconds = started.durationTo(finished).nanoseconds;
    return if (nanoseconds <= 0) 0 else @as(f64, @floatFromInt(nanoseconds)) / 1_000_000.0;
}

fn readTensor(reader: *const safe.MMapReader, case: []const u8, suffix: []const u8) !Tensor {
    var buf: [256]u8 = undefined;
    const name = try std.fmt.bufPrint(&buf, "{s}.{s}", .{ case, suffix });
    var tensor = try reader.readTensor(name);
    errdefer tensor.deinit();
    if (tensor.dtype != .f32) return error.InvalidFixtureTensor;
    return tensor;
}

const Timings = struct {
    encoder_ms: f64 = 0,
    features_ms: f64 = 0,
    reader_ms: f64 = 0,
    copy_ms: f64 = 0,
    snap_ms: f64 = 0,
    complete_ms: f64 = 0,
    decoder_ms: f64 = 0,
};

/// Stages 4-6 under one bridge margin.
const MarginReport = struct {
    margin: f64,
    spans_match: bool,
    spans: []const copy.Span,
    snapped_match: bool,
    completed_match: bool,
    completed_ids: []const i32,
    answer: ?[]const u8,
    answer_correct: ?bool,
    answer_matches_reference: ?bool,
    finish_reason: ?[]const u8,
    decoder_ms: f64,
};

const CaseReport = struct {
    event: []const u8 = "case",
    name: []const u8,
    documents: usize,
    feature_rows: usize,
    reference_feature_rows: usize,
    copy_sources: usize,
    copy_ids_match: bool,
    docs: ?Stat,
    query: ?Stat,
    memories: ?Stat,
    copy_logits: ?Stat,
    copy_gate: ?Stat,
    copy_output: ?Stat,
    spans_match: bool,
    spans: []const copy.Span,
    reference_spans: []const [3]i64,
    snapped_ids: []const i32,
    snapped_match: bool,
    snapped_agreeing_slots: usize,
    completed_ids: []const i32,
    completed_match: bool,
    prompt_match: ?bool,
    answer: ?[]const u8,
    reference_answer: []const u8,
    expected_answer: []const u8,
    answer_matches_reference: ?bool,
    answer_correct: ?bool,
    reference_correct: bool,
    output_ids_match: ?bool,
    finish_reason: ?[]const u8,
    timings: Timings,
    margins: []const MarginReport,
};

const Decoder = struct {
    session: internal.backends.Session,
    backend: ops.ComputeBackend,
    budget: memory.RunBudget,
    tokenizer: internal.tokenizer.Tokenizer,
    manifest: manifest_mod.ModelManifest,
    template: generation.ChatTemplate,
    config: gpt.Config,
    table_norms: []f32,
    mutex: std.atomic.Mutex = .unlocked,
    limits: memory.Limits,
};

/// The merged encoder is a managed research publication, so it loads through
/// a `ModelManager` exactly as serving does and the per-token features come
/// from the production `encodeTokenFeatures` entry point.
const Encoder = struct {
    manager: *internal.server.model_manager.ModelManager,
    handle: internal.server.model_manager.ModelHandle,
    limits: memory.Limits,
};

const Reader = struct {
    bundle: model.Bundle,
    store: internal.native_compute.native.WeightStore,
    compute: *internal.native_compute.cuda.CudaCompute,
    backend: ops.ComputeBackend,
};

const Fixture = struct {
    reference: Reference,
    reader: safe.MMapReader,
};

/// Document rows, query and copy sources for one case; either computed by the
/// encoder or copied from the fixture.
const CaseFeatures = struct {
    a: A,
    docs: []f32,
    rows: usize,
    query: []f32,
    copy_ids: []u32,
    encoder_ms: f64 = 0,
    features_ms: f64 = 0,

    fn deinit(self: *CaseFeatures) void {
        self.a.free(self.docs);
        self.a.free(self.query);
        self.a.free(self.copy_ids);
    }
};

fn encodeCase(a: A, io: std.Io, encoder: *Encoder, case: Case) !CaseFeatures {
    const started = std.Io.Timestamp.now(io, .awake);
    const control = inProcessControl(io);
    const canonical = try qualification.canonicalQuestion(case.question);
    var encoded = try memorilla_encoder.encodeTokenFeatures(a, &encoder.handle, case.documents, canonical, encoder.limits, control);
    defer encoded.deinit();
    const encoder_ms = elapsedMs(io, started);
    const docs = try a.dupe(f32, encoded.features.docs);
    errdefer a.free(docs);
    const query = try a.dupe(f32, encoded.features.query);
    errdefer a.free(query);
    const copy_ids = try a.dupe(u32, encoded.features.copy_ids);
    return .{ .a = a, .docs = docs, .rows = encoded.features.rows, .query = query, .copy_ids = copy_ids, .encoder_ms = encoder_ms, .features_ms = 0 };
}

fn fixtureFeatures(a: A, fixture: *Fixture, case: Case) !CaseFeatures {
    var docs = try readTensor(&fixture.reader, case.name, "docs");
    defer docs.deinit();
    var query = try readTensor(&fixture.reader, case.name, "query");
    defer query.deinit();
    if (docs.shape.len != 2 or docs.shape[1] != features_mod.width or query.asFloat32().len != features_mod.width) return error.InvalidFixtureTensor;
    const copied_docs = try a.dupe(f32, docs.asFloat32());
    errdefer a.free(copied_docs);
    const copied_query = try a.dupe(f32, query.asFloat32());
    errdefer a.free(copied_query);
    const copy_ids = try a.alloc(u32, case.copy_ids.len);
    for (case.copy_ids, copy_ids) |id, *out| {
        if (id < 0) return error.InvalidFixtureTensor;
        out.* = @intCast(id);
    }
    return .{ .a = a, .docs = copied_docs, .rows = @intCast(docs.shape[0]), .query = copied_query, .copy_ids = copy_ids };
}

fn runReader(a: A, io: std.Io, reader: *Reader, feats: *const CaseFeatures, timings: *Timings) ![]f32 {
    const started = std.Io.Timestamp.now(io, .awake);
    const cb = &reader.backend;
    const docs = try cb.fromFloat32Shape(feats.docs, &.{ @intCast(feats.rows), @intCast(features_mod.width) });
    defer cb.free(docs);
    const query = try cb.fromFloat32Shape(feats.query, &.{ 1, @intCast(features_mod.width) });
    defer cb.free(query);
    const mask = try a.alloc(bool, feats.rows);
    defer a.free(mask);
    @memset(mask, false);
    const memories = try reader_pipeline.forward(cb, a, reader.bundle.config, .{
        .documents = docs,
        .question = query,
        .padding_mask = mask,
        .batch = 1,
        .num_documents = feats.rows,
        .max_document_rows = 65536,
    }, null);
    defer cb.free(memories);
    const host = try cb.toFloat32(memories, a);
    errdefer a.free(host);
    if (host.len != slots * reader.bundle.config.output_dim) return error.UnexpectedReaderOutput;
    timings.reader_ms = elapsedMs(io, started);
    return host;
}

fn runCase(a: A, io: std.Io, args: Args, fixture: *Fixture, case: Case, encoder: ?*Encoder, reader: *Reader, head: *const copy.CopyHead, decoder: *Decoder) !CaseReport {
    var timings = Timings{};
    var feats = if (args.reference_features) try fixtureFeatures(a, fixture, case) else try encodeCase(a, io, encoder.?, case);
    defer feats.deinit();
    timings.encoder_ms = feats.encoder_ms;
    timings.features_ms = feats.features_ms;

    // Stage 1: features against the fixture (shape may differ when the
    // tokenization differs; then only the row counts are reported).
    var docs_stat: ?Stat = null;
    var query_stat: ?Stat = null;
    var reference_rows: usize = 0;
    {
        var docs = try readTensor(&fixture.reader, case.name, "docs");
        defer docs.deinit();
        var query = try readTensor(&fixture.reader, case.name, "query");
        defer query.deinit();
        if (docs.shape.len == 2) reference_rows = @intCast(docs.shape[0]);
        if (docs.asFloat32().len == feats.docs.len) docs_stat = try compare(feats.docs, docs.asFloat32(), features_mod.width, args.atol, args.rtol);
        if (query.asFloat32().len == feats.query.len) query_stat = try compare(feats.query, query.asFloat32(), features_mod.width, args.atol, args.rtol);
    }
    const copy_ids_match = idsEqualUnsigned(feats.copy_ids, case.copy_ids);

    // Stage 2: reader.
    const memories = try runReader(a, io, reader, &feats, &timings);
    defer a.free(memories);
    var memories_stat: ?Stat = null;
    var reference_memories: ?[]f32 = null;
    defer if (reference_memories) |values| a.free(values);
    {
        var expected = try readTensor(&fixture.reader, case.name, "memories");
        defer expected.deinit();
        if (expected.asFloat32().len == memories.len) memories_stat = try compare(memories, expected.asFloat32(), decoder.config.hidden_size, args.atol, args.rtol);
        if (args.reference_memories) reference_memories = try a.dupe(f32, expected.asFloat32());
    }
    const copy_input = reference_memories orelse memories;

    // Stage 3 (copy head) runs once; stages 4-6 own every device tensor they
    // create and run once per bridge margin. The decoder's generate path
    // requires the decoder compute to hold nothing but resident weights.
    var stage = try copyForward(a, io, args, fixture, case, decoder, head, &feats, copy_input, &timings);
    defer stage.deinit();

    const margins = args.copy_bridge_margins;
    const primary = primaryMargin(margins);
    const reports = try a.alloc(MarginReport, margins.len);
    var primary_timings = Timings{};
    var report_primary: ?PrimaryFields = null;
    for (margins, reports, 0..) |margin, *out, index| {
        var margin_timings = Timings{};
        var copied = try copyAndSnap(a, io, fixture, case, decoder, &stage, copy_input, @floatCast(margin), &margin_timings);
        defer copied.deinit();
        out.* = .{
            .margin = margin,
            .spans_match = spansEqual(copied.spans, case.spans),
            .spans = try a.dupe(copy.Span, copied.spans),
            .snapped_match = idsEqual(copied.snapped, case.snapped_ids),
            .completed_match = idsEqual(copied.completed, case.completed_ids),
            .completed_ids = try a.dupe(i32, copied.completed),
            .answer = null,
            .answer_correct = null,
            .answer_matches_reference = null,
            .finish_reason = null,
            .decoder_ms = 0,
        };
        var generated: ?Generated = null;
        if (!args.skip_decoder) {
            generated = try generateAnswer(a, io, fixture, case, decoder, reader, copied.completed, &margin_timings);
            out.answer = generated.?.answer;
            out.finish_reason = generated.?.finish_reason;
            out.answer_matches_reference = generated.?.answer_matches_reference;
            out.answer_correct = generated.?.answer_correct;
            out.decoder_ms = margin_timings.decoder_ms;
        }
        if (index == primary) {
            primary_timings = margin_timings;
            report_primary = .{ .copied = .{
                .snapped_ids = try a.dupe(i32, copied.snapped),
                .snapped_agreeing = copied.snapped_agreeing,
            }, .generated = generated };
        }
    }
    timings.copy_ms += primary_timings.copy_ms;
    timings.snap_ms = primary_timings.snap_ms;
    timings.complete_ms = primary_timings.complete_ms;
    timings.decoder_ms = primary_timings.decoder_ms;
    const lead = reports[primary];
    const primary_fields = report_primary.?;

    return CaseReport{
        .name = case.name,
        .documents = case.documents.len,
        .feature_rows = feats.rows,
        .reference_feature_rows = reference_rows,
        .copy_sources = feats.copy_ids.len,
        .copy_ids_match = copy_ids_match,
        .docs = docs_stat,
        .query = query_stat,
        .memories = memories_stat,
        .copy_logits = stage.logits_stat,
        .copy_gate = stage.gate_stat,
        .copy_output = stage.output_stat,
        .spans_match = lead.spans_match,
        .spans = lead.spans,
        .reference_spans = case.spans,
        .snapped_ids = primary_fields.copied.snapped_ids,
        .snapped_match = idsEqual(primary_fields.copied.snapped_ids, case.snapped_ids),
        .snapped_agreeing_slots = primary_fields.copied.snapped_agreeing,
        .completed_ids = lead.completed_ids,
        .completed_match = lead.completed_match,
        .prompt_match = if (primary_fields.generated) |g| g.prompt_match else null,
        .answer = lead.answer,
        .reference_answer = case.answer,
        .expected_answer = case.expected_answer,
        .answer_matches_reference = lead.answer_matches_reference,
        .answer_correct = lead.answer_correct,
        .reference_correct = std.mem.eql(u8, std.mem.trim(u8, case.answer, " \t\r\n"), case.expected_answer),
        .output_ids_match = if (primary_fields.generated) |g| g.output_ids_match else null,
        .finish_reason = lead.finish_reason,
        .timings = timings,
        .margins = reports,
    };
}

/// Fields of the legacy single-margin report that are not part of
/// `MarginReport`.
const PrimaryFields = struct {
    copied: struct { snapped_ids: []i32, snapped_agreeing: usize },
    generated: ?Generated,
};

const Generated = struct {
    answer: []const u8,
    finish_reason: []const u8,
    prompt_match: bool,
    output_ids_match: ?bool,
    answer_matches_reference: bool,
    answer_correct: bool,
};

/// Stage 6: decoder with the completed ids' embedding rows as memory. The
/// rows travel through the reader backend, as in serving, so the decoder
/// compute starts generation holding only its resident weights.
fn generateAnswer(a: A, io: std.Io, fixture: *Fixture, case: Case, decoder: *Decoder, reader: *Reader, completed: []const i32, timings: *Timings) !Generated {
    const decoder_started = std.Io.Timestamp.now(io, .awake);
    const hidden = decoder.config.hidden_size;
    const canonical = try qualification.canonicalQuestion(case.question);
    const limits = decode.Limits{ .max_new_tokens = 256 };
    const control = inProcessControl(io);
    var text = try decode.prepareText(a, decoder.tokenizer, &decoder.template, decoder.config, canonical, .{ .system_prompt = fixture.reference.system_prompt }, limits, control);
    defer text.deinit();
    const prompt_match = idsEqualWide(text.input_ids, case.prompt_ids);
    const rows_host = try embeddingRows(a, decoder, completed);
    defer a.free(rows_host);
    const rows_ct = try reader.backend.fromFloat32Shape(rows_host, &.{ @intCast(slots), @intCast(hidden) });
    defer reader.backend.free(rows_ct);
    const plan = try decode.ResourcePlan.create(decoder.config, .cuda, text.input_ids.len, text.rendered.len, slots, fixture.reference.max_new_tokens, limits);
    var reservation = try plan.reserve(&decoder.budget);
    defer reservation.deinit();
    const engine = try decode.Decoder.fromQualifiedSession(decoder.session, &decoder.backend, decoder.tokenizer, &decoder.mutex);
    var result = try decode.generate(a, engine, &text, .{ .backend = &reader.backend, .tensor = rows_ct }, .{ .max_new_tokens = fixture.reference.max_new_tokens }, &reservation, limits, control);
    defer result.deinit();
    timings.decoder_ms = elapsedMs(io, decoder_started);
    const answer = std.mem.trim(u8, result.text, " \t\r\n");
    return .{
        .answer = try a.dupe(u8, answer),
        .finish_reason = try a.dupe(u8, result.finish_reason),
        .prompt_match = prompt_match,
        .output_ids_match = if (result.token_ids) |ids| idsEqual(ids, case.output_ids) else null,
        .answer_matches_reference = std.mem.eql(u8, answer, std.mem.trim(u8, case.answer, " \t\r\n")),
        .answer_correct = std.mem.eql(u8, answer, case.expected_answer),
    };
}

/// Host F32 embedding rows of `ids` from the decoder table; the device handles
/// are released before returning.
fn embeddingRows(a: A, decoder: *Decoder, ids: []const i32) ![]f32 {
    const cb = &decoder.backend;
    const wide = try a.alloc(i64, ids.len);
    defer a.free(wide);
    for (ids, wide) |id, *out| {
        if (id < 0 or id >= decoder.config.vocab_size) return error.InvalidCopySource;
        out.* = id;
    }
    const weight = try gpt.getEmbeddingWeight(cb, decoder.config);
    defer cb.free(weight);
    const rows = try cb.embeddingLookup(weight, wide, ids.len, decoder.config.hidden_size);
    defer cb.free(rows);
    return cb.toFloat32(rows, a);
}

/// Stage 3 outputs shared by every bridge margin.
const CopyStage = struct {
    a: A,
    source_ids: []i32,
    sources: []f32,
    fwd: copy.Forward,
    logits_stat: ?Stat,
    gate_stat: ?Stat,
    output_stat: ?Stat,

    fn deinit(self: *CopyStage) void {
        self.fwd.deinit();
        self.a.free(self.sources);
        self.a.free(self.source_ids);
    }
};

const Copied = struct {
    a: A,
    spans: []copy.Span,
    snapped: []i32,
    snapped_agreeing: usize,
    completed: []i32,

    fn deinit(self: *Copied) void {
        self.a.free(self.spans);
        self.a.free(self.snapped);
        self.a.free(self.completed);
    }
};

/// Stage 3: copy head over the decoder's embedding rows of the sources,
/// compared against the fixture.
fn copyForward(a: A, io: std.Io, args: Args, fixture: *Fixture, case: Case, decoder: *Decoder, head: *const copy.CopyHead, feats: *const CaseFeatures, copy_input: []const f32, timings: *Timings) !CopyStage {
    const hidden = decoder.config.hidden_size;
    const source_count = feats.copy_ids.len;
    if (source_count == 0) return error.EmptyCopySources;
    const source_ids = try a.alloc(i32, source_count);
    errdefer a.free(source_ids);
    for (feats.copy_ids, source_ids) |id, *narrow| {
        if (id >= decoder.config.vocab_size) return error.InvalidCopySource;
        narrow.* = @intCast(id);
    }
    const copy_started = std.Io.Timestamp.now(io, .awake);
    const sources = try embeddingRows(a, decoder, source_ids);
    errdefer a.free(sources);
    var fwd = try copy.forward(a, head, copy_input, slots, sources, source_count);
    errdefer fwd.deinit();
    var logits_stat: ?Stat = null;
    var gate_stat: ?Stat = null;
    var output_stat: ?Stat = null;
    {
        var logits = try readTensor(&fixture.reader, case.name, "logits");
        defer logits.deinit();
        var gate = try readTensor(&fixture.reader, case.name, "gate");
        defer gate.deinit();
        var output = try readTensor(&fixture.reader, case.name, "copy_output");
        defer output.deinit();
        if (logits.asFloat32().len == fwd.logits.len) logits_stat = try compare(fwd.logits, logits.asFloat32(), source_count, args.atol * 10, args.rtol);
        if (gate.asFloat32().len == fwd.gate.len) gate_stat = try compare(fwd.gate, gate.asFloat32(), 1, args.atol, args.rtol);
        if (output.asFloat32().len == fwd.soft_output.len) output_stat = try compare(fwd.soft_output, output.asFloat32(), hidden, args.atol, args.rtol);
    }
    timings.copy_ms = elapsedMs(io, copy_started);
    return .{ .a = a, .source_ids = source_ids, .sources = sources, .fwd = fwd, .logits_stat = logits_stat, .gate_stat = gate_stat, .output_stat = output_stat };
}

/// Stages 4-5 under one bridge margin: contiguous runs, snapping, word
/// completion.
fn copyAndSnap(a: A, io: std.Io, fixture: *Fixture, case: Case, decoder: *Decoder, stage: *const CopyStage, copy_input: []const f32, bridge_margin: f32, timings: *Timings) !Copied {
    const hidden = decoder.config.hidden_size;
    const runs_started = std.Io.Timestamp.now(io, .awake);
    var runs = try copy.contiguousRunsForwardWithMargin(a, &stage.fwd, copy_input, stage.sources, copy.default_min_gate, bridge_margin);
    defer runs.deinit();
    timings.copy_ms = elapsedMs(io, runs_started);

    // Stage 4: snap every slot to its nearest vocabulary row (dot products on
    // the GPU against the resident embedding table, norms on the host).
    const snap_started = std.Io.Timestamp.now(io, .awake);
    const scores = blk: {
        const cb = &decoder.backend;
        const weight = try gpt.getEmbeddingWeight(cb, decoder.config);
        defer cb.free(weight);
        const output_ct = try cb.fromFloat32Shape(runs.output, &.{ @intCast(slots), @intCast(hidden) });
        defer cb.free(output_ct);
        const scores_ct = try cb.linearNoBias(output_ct, weight, slots, hidden, decoder.config.vocab_size);
        defer cb.free(scores_ct);
        break :blk try cb.toFloat32(scores_ct, a);
    };
    defer a.free(scores);
    const output_norms = try copy.rowNorms(a, runs.output, slots, hidden);
    defer a.free(output_norms);
    const snapped = try copy.snapIds(a, scores, decoder.table_norms, output_norms, slots, decoder.config.vocab_size);
    errdefer a.free(snapped);
    timings.snap_ms = elapsedMs(io, snap_started);
    var snapped_agreeing: usize = 0;
    if (case.snapped_ids.len == slots) {
        for (snapped, case.snapped_ids) |x, y| snapped_agreeing += @intFromBool(@as(i64, x) == y);
    }

    // Stage 5: word completion with the decoder tokenizer.
    const complete_started = std.Io.Timestamp.now(io, .awake);
    const completed = blk: {
        var owned = try copy.completeCopiedWords(a, decoder.tokenizer, snapped, runs.spans, stage.source_ids, fixture.reference.pad_id);
        defer owned.deinit();
        break :blk try a.dupe(i32, owned.ids);
    };
    errdefer a.free(completed);
    timings.complete_ms = elapsedMs(io, complete_started);
    const spans = try a.dupe(copy.Span, runs.spans);
    return .{ .a = a, .spans = spans, .snapped = snapped, .snapped_agreeing = snapped_agreeing, .completed = completed };
}

fn openReader(a: A, path: []const u8) !Reader {
    if (comptime !options.enable_cuda) return error.CudaNotEnabled;
    var bundle = try model.Bundle.open(a, path, true);
    errdefer bundle.deinit();
    if (bundle.release) return error.ReleaseBundleNotAResearchCandidate;
    var store = try bundle.loadFloat32Store();
    errdefer store.deinitOwned();
    const compute = try a.create(internal.native_compute.cuda.CudaCompute);
    errdefer a.destroy(compute);
    compute.* = try internal.native_compute.cuda.CudaCompute.init(a);
    errdefer compute.deinit();
    compute.strict_f32_weights = true;
    try compute.resident_weights.ensureUnusedCapacity(a, store.resident_weights.count());
    var iter = store.resident_weights.iterator();
    while (iter.next()) |entry| {
        const key = try a.dupe(u8, entry.key_ptr.*);
        errdefer a.free(key);
        try compute.insertWeightFromLoaded(key, entry.value_ptr);
    }
    const backend = compute.computeBackend();
    if (backend.kind() != .cuda) return error.RequiredCudaBackendUnavailable;
    return .{ .bundle = bundle, .store = store, .compute = compute, .backend = backend };
}

fn closeReader(a: A, reader: *Reader) void {
    reader.compute.deinit();
    a.destroy(reader.compute);
    reader.store.deinitOwned();
    reader.bundle.deinit();
}

fn openDecoder(a: A, path: []const u8) !Decoder {
    if (comptime !options.enable_cuda) return error.CudaNotEnabled;
    const session = try factory.createCudaSessionWithManagedMemorillaProfile(a, path, .{}, .dynamic, null, null);
    errdefer session.close();
    const config = try factory.requireMemorillaDecoderF32(session);
    const hf = try internal.server.model_manager.loadHuggingFaceTokenizerFromDir(a, path);
    const tokenizer = hf.tokenizer();
    errdefer tokenizer.deinitTokenizer();
    var manifest = try manifest_mod.loadFromDir(a, path);
    errdefer manifest.deinit();
    const source = manifest.chat_template orelse return error.MemorillaChatTemplateRequired;
    const template = try generation.ChatTemplate.init(a, source, manifest.bos_token, manifest.eos_token, manifest.unk_token, manifest.pad_token);
    const limits = memory.Limits{
        .host_limit_bytes = 16 * 1024 * 1024 * 1024,
        .backend_limit_bytes = 48 * 1024 * 1024 * 1024,
        .combined_limit_bytes = 64 * 1024 * 1024 * 1024,
    };
    var result = Decoder{
        .session = session,
        .backend = undefined,
        .budget = memory.RunBudget.init(limits),
        .tokenizer = tokenizer,
        .manifest = manifest,
        .template = template,
        .config = config,
        .table_norms = &.{},
        .limits = limits,
    };
    result.backend = try factory.getCoordinatedCudaComputeBackend(session, &result.budget, @intFromPtr(&result.budget));
    errdefer result.backend.deinit();
    // Table norms once: the snapper divides GPU dot products by them. The
    // weight handle is released again so generation sees only resident weights.
    const weight = try gpt.getEmbeddingWeight(&result.backend, config);
    defer result.backend.free(weight);
    const table = try result.backend.toFloat32(weight, a);
    defer a.free(table);
    if (table.len != config.vocab_size * config.hidden_size) return error.UnexpectedEmbeddingTable;
    result.table_norms = try copy.rowNorms(a, table, config.vocab_size, config.hidden_size);
    return result;
}

fn closeDecoder(a: A, decoder: *Decoder) void {
    a.free(decoder.table_norms);
    decoder.backend.deinit();
    decoder.manifest.deinit();
    decoder.tokenizer.deinitTokenizer();
    decoder.session.close();
}

fn openEncoder(a: A, io: std.Io, path: []const u8) !Encoder {
    if (comptime !options.enable_cuda) return error.CudaNotEnabled;
    const cap: usize = 96 * 1024 * 1024 * 1024;
    const limits = memory.Limits{ .host_limit_bytes = cap, .backend_limit_bytes = cap, .combined_limit_bytes = cap, .kv_limit_bytes = 2 * 1024 * 1024 * 1024, .scratch_limit_bytes = 12 * 1024 * 1024 * 1024 };
    var sessions = internal.backends.SessionManager.initWithIo(a, io);
    sessions.preferred_backends = &.{.cuda};
    sessions.required_backend = .cuda;
    sessions.native_call_policy = .in_process;
    const manager = try a.create(internal.server.model_manager.ModelManager);
    errdefer a.destroy(manager);
    manager.* = internal.server.model_manager.ModelManager.init(a, sessions);
    errdefer manager.deinit();
    manager.configureServingPolicy(.{});
    manager.configureAdmissionLimits(limits);
    manager.configureProcessMemoryLimit(cap, .explicit);
    manager.configureModelCache(600_000, 4);
    try manager.ensureResourceOwnerReady();
    const real = try std.Io.Dir.cwd().realPathFileAlloc(io, path, a);
    defer a.free(real);
    const handle = try manager.acquireFromDirWithControl(real, inProcessControl(io));
    return .{ .manager = manager, .handle = handle, .limits = limits };
}

fn closeEncoder(a: A, encoder: *Encoder) void {
    encoder.handle.release();
    encoder.manager.deinit();
    a.destroy(encoder.manager);
}

const MarginTotals = struct {
    margin: f64,
    decoded: usize = 0,
    spans_match: usize = 0,
    snapped_match: usize = 0,
    completed_match: usize = 0,
    answer_correct: usize = 0,
    answer_matches_reference: usize = 0,
    /// Decoded answers equal to "UNKNOWN".
    abstained: usize = 0,
    /// Decoded answers that are neither correct nor an abstention.
    wrong: usize = 0,
    mean_decoder_ms: f64 = 0,
};

const Summary = struct {
    event: []const u8 = "complete",
    cases: usize = 0,
    errors: usize = 0,
    copy_ids_match: usize = 0,
    spans_match: usize = 0,
    snapped_match: usize = 0,
    completed_match: usize = 0,
    prompt_match: usize = 0,
    answer_matches_reference: usize = 0,
    answer_correct: usize = 0,
    reference_correct: usize = 0,
    decoded: usize = 0,
    max_memories_relative_l2: f64 = 0,
    min_memories_row_cosine: f64 = 1,
    max_docs_relative_l2: f64 = 0,
    mean_timings_ms: Timings = .{},
    margins: []const MarginTotals = &.{},
};

pub fn main(init: std.process.Init) !void {
    if (comptime !options.enable_cuda) return error.CudaNotEnabled;
    const a = std.heap.c_allocator;
    const io = init.io;
    const args = try parseArgs(a, init);
    const receipt_path = try std.fs.path.join(a, &.{ args.reference, "reference.json" });
    defer a.free(receipt_path);
    const receipt_bytes = try internal.util.c_file.readFileMax(a, receipt_path, 64 * 1024 * 1024);
    defer a.free(receipt_bytes);
    var parsed = try std.json.parseFromSlice(Reference, a, receipt_bytes, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    if (!std.mem.eql(u8, parsed.value.schema, expected_schema) or parsed.value.cases.len == 0) return error.InvalidReference;
    const tensor_path = try std.fs.path.join(a, &.{ args.reference, parsed.value.tensors.path });
    defer a.free(tensor_path);
    var fixture = Fixture{ .reference = parsed.value, .reader = try safe.MMapReader.openFileAbsoluteLimited(a, tensor_path, 8 * 1024 * 1024 * 1024, 16 * 1024 * 1024) };
    defer fixture.reader.deinit();
    try safe.validateReader(a, &fixture.reader);
    if (fixture.reference.tensors.sha256.len == 64) {
        const digest = model.digest(fixture.reader.file_bytes);
        if (!std.mem.eql(u8, &digest, fixture.reference.tensors.sha256)) return error.ReferenceIdentityMismatch;
    }

    var head = try copy.CopyHead.load(a, args.candidate, null);
    defer head.deinit();
    var reader = try openReader(a, args.reader_bundle);
    defer closeReader(a, &reader);
    if (reader.bundle.config.num_memories != slots) return error.UnsupportedSlotCount;
    var decoder = try openDecoder(a, args.decoder);
    defer closeDecoder(a, &decoder);
    if (head.width != decoder.config.hidden_size or reader.bundle.config.output_dim != decoder.config.hidden_size) return error.WidthMismatch;
    var encoder: ?Encoder = if (args.reference_features) null else try openEncoder(a, io, args.encoder);
    defer if (encoder) |*active| closeEncoder(a, active);

    try emit(a, io, .{
        .event = "ready",
        .scope = "memorilla_encoder_lora_pipeline_parity",
        .schema = fixture.reference.schema,
        .cases = fixture.reference.cases.len,
        .reader_bundle_sha256 = reader.bundle.weights_sha256[0..],
        .reader_release = reader.bundle.release,
        .reference_features = args.reference_features,
        .reference_memories = args.reference_memories,
        .skip_decoder = args.skip_decoder,
        .copy_bridge_margins = args.copy_bridge_margins,
        .dual_layer = fixture.reference.dual_layer,
        .atol = args.atol,
        .rtol = args.rtol,
        .arithmetic = "source_weights_widened_f32_activations_f32",
    });

    var summary = Summary{};
    var total = Timings{};
    const totals = try a.alloc(MarginTotals, args.copy_bridge_margins.len);
    defer a.free(totals);
    for (totals, args.copy_bridge_margins) |*t, margin| t.* = .{ .margin = margin };
    const decoder_total = try a.alloc(f64, totals.len);
    defer a.free(decoder_total);
    @memset(decoder_total, 0);
    const limit = if (args.max_cases == 0) fixture.reference.cases.len else @min(args.max_cases, fixture.reference.cases.len);
    for (fixture.reference.cases[0..limit]) |case| {
        summary.cases += 1;
        const report = runCase(a, io, args, &fixture, case, if (encoder) |*active| active else null, &reader, &head, &decoder) catch |err| {
            summary.errors += 1;
            try emit(a, io, .{ .event = "case_error", .name = case.name, .@"error" = @errorName(err) });
            continue;
        };
        try emit(a, io, report);
        summary.copy_ids_match += @intFromBool(report.copy_ids_match);
        summary.spans_match += @intFromBool(report.spans_match);
        summary.snapped_match += @intFromBool(report.snapped_match);
        summary.completed_match += @intFromBool(report.completed_match);
        summary.reference_correct += @intFromBool(report.reference_correct);
        if (report.prompt_match) |flag| summary.prompt_match += @intFromBool(flag);
        if (report.answer_matches_reference) |flag| summary.answer_matches_reference += @intFromBool(flag);
        if (report.answer_correct) |flag| {
            summary.decoded += 1;
            summary.answer_correct += @intFromBool(flag);
        }
        if (report.memories) |stat| {
            summary.max_memories_relative_l2 = @max(summary.max_memories_relative_l2, stat.relative_l2_error);
            summary.min_memories_row_cosine = @min(summary.min_memories_row_cosine, stat.min_row_cosine);
        }
        if (report.docs) |stat| summary.max_docs_relative_l2 = @max(summary.max_docs_relative_l2, stat.relative_l2_error);
        for (report.margins, totals, decoder_total) |m, *t, *ms| {
            t.spans_match += @intFromBool(m.spans_match);
            t.snapped_match += @intFromBool(m.snapped_match);
            t.completed_match += @intFromBool(m.completed_match);
            if (m.answer_matches_reference) |flag| t.answer_matches_reference += @intFromBool(flag);
            if (m.answer_correct) |flag| {
                t.decoded += 1;
                t.answer_correct += @intFromBool(flag);
                const abstained = std.mem.eql(u8, m.answer.?, "UNKNOWN");
                t.abstained += @intFromBool(abstained);
                t.wrong += @intFromBool(!flag and !abstained);
            }
            ms.* += m.decoder_ms;
        }
        total.encoder_ms += report.timings.encoder_ms;
        total.features_ms += report.timings.features_ms;
        total.reader_ms += report.timings.reader_ms;
        total.copy_ms += report.timings.copy_ms;
        total.snap_ms += report.timings.snap_ms;
        total.complete_ms += report.timings.complete_ms;
        total.decoder_ms += report.timings.decoder_ms;
    }
    const completed_cases: f64 = @floatFromInt(@max(summary.cases - summary.errors, 1));
    summary.mean_timings_ms = .{
        .encoder_ms = total.encoder_ms / completed_cases,
        .features_ms = total.features_ms / completed_cases,
        .reader_ms = total.reader_ms / completed_cases,
        .copy_ms = total.copy_ms / completed_cases,
        .snap_ms = total.snap_ms / completed_cases,
        .complete_ms = total.complete_ms / completed_cases,
        .decoder_ms = total.decoder_ms / completed_cases,
    };
    for (totals, decoder_total) |*t, ms| t.mean_decoder_ms = ms / completed_cases;
    summary.margins = totals;
    try emit(a, io, summary);
    if (args.output.len > 0) {
        const json = try std.json.Stringify.valueAlloc(a, summary, .{ .whitespace = .indent_2 });
        defer a.free(json);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args.output, .data = json });
    }
    if (summary.errors != 0) return error.CaseFailures;
}

test "memorilla lora verify parses bridge margins" {
    const a = std.testing.allocator;
    const margins = try parseMargins(a, "0,0.05, 0.1,0.15");
    defer a.free(margins);
    try std.testing.expectEqualSlices(f64, &.{ 0, 0.05, 0.1, 0.15 }, margins);
    try std.testing.expectEqual(@as(usize, 0), primaryMargin(margins));
    try std.testing.expectEqual(@as(usize, 0), primaryMargin(&.{ 0.05, 0.1 }));
    try std.testing.expectEqual(@as(usize, 1), primaryMargin(&.{ 0.05, 0 }));
    try std.testing.expectError(error.InvalidArguments, parseMargins(a, ""));
    try std.testing.expectError(error.InvalidArguments, parseMargins(a, "0,1.5"));
    try std.testing.expectError(error.InvalidArguments, parseMargins(a, "0.1,0.1"));
    try std.testing.expectError(error.InvalidCharacter, parseMargins(a, "x"));
}

test "memorilla lora verify comparison reports mismatches and row cosines" {
    const exact = try compare(&.{ 1, 2, 3, 4 }, &.{ 1, 2, 3, 4 }, 2, 1e-6, 0);
    try std.testing.expectEqual(@as(usize, 0), exact.mismatches);
    try std.testing.expect(exact.passed);
    try std.testing.expectApproxEqAbs(@as(f64, 1), exact.min_row_cosine, 1e-12);
    const off = try compare(&.{ 1, 2, 3, 4 }, &.{ 1, 2, 3, 5 }, 2, 1e-3, 0);
    try std.testing.expectEqual(@as(usize, 1), off.mismatches);
    try std.testing.expect(!off.passed and off.min_row_cosine < 1);
    try std.testing.expectError(error.ShapeMismatch, compare(&.{ 1, 2 }, &.{1}, 1, 1e-3, 0));
    try std.testing.expect(spansEqual(&.{.{ .begin = 0, .end = 3, .start = 2 }}, &.{.{ 0, 3, 2 }}));
    try std.testing.expect(!spansEqual(&.{.{ .begin = 0, .end = 3, .start = 2 }}, &.{.{ 0, 3, 1 }}));
    try std.testing.expect(idsEqual(&.{ 1, 2 }, &.{ 1, 2 }) and !idsEqual(&.{1}, &.{ 1, 2 }));
}
