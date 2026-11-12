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

//! Host-side copy head and decoding rules of the Memorilla copy17 candidate
//! (docs/plans/memorilla-copy17-native.md, sections 1.5 and 1.6). Everything
//! here is plain F32 code over caller-provided slices and mirrors the Python
//! research implementation in `scripts/memorilla/encoder_lora.py`
//! (`CopyHead.forward_detailed`, `copy_with_contiguous_runs` with the default
//! arguments, `snap_memory_to_tokens`, `token_continues_word` and
//! `complete_copied_words` with `overflow="truncate"`).
//!
//! Layouts are row-major: memories `[slots, width]`, source embeddings
//! `[sources, width]`, logits `[slots, sources]`.
const std = @import("std");
const tok = @import("inference_tokenizer");
const safetensors = @import("../models/safetensors.zig");
const A = std.mem.Allocator;

pub const weights_file_name = "copy-head.safetensors";
pub const default_temperature: f32 = 0.05;
pub const default_min_gate: f32 = 0.5;
/// Width of the "uncertain" gate band above `min_gate` in which a slot may
/// not bridge two confident runs (see `contiguousRunsWithMargin`); 0 keeps
/// the plain maximal-run rule of the Python reference.
pub const default_bridge_margin: f32 = 0;
/// Qwen3 id of the single space " ", the record pad of the training data.
pub const default_pad_id: i32 = 220;
/// `torch.nn.functional.normalize` clamps the norm at this value.
const normalize_eps: f32 = 1e-12;

/// A run of copying record slots `begin..end` reading source positions
/// `start..start + (end - begin)`.
pub const Span = struct {
    begin: usize,
    end: usize,
    start: usize,

    pub fn length(self: Span) usize {
        return self.end - self.begin;
    }
};

// --- Copy head weights -----------------------------------------------------

pub const CopyHead = struct {
    allocator: A,
    width: usize,
    /// `query.weight` in nn.Linear layout `[out, in]`: output i of the
    /// projection is the dot product of row i with the memory vector.
    query_weight: []f32,
    /// `gate.weight[0]`, one weight per memory dimension.
    gate_weight: []f32,
    gate_bias: f32,
    temperature: f32,

    /// Copies the supplied weights; the caller keeps ownership of its slices.
    pub fn init(a: A, width: usize, query_weight: []const f32, gate_weight: []const f32, gate_bias: f32, temperature: f32) !CopyHead {
        if (width == 0 or query_weight.len != width * width or gate_weight.len != width) return error.InvalidCopyHeadShape;
        if (!(temperature > 0 and temperature <= 1)) return error.InvalidCopyHeadTemperature;
        for (query_weight) |v| if (!std.math.isFinite(v)) return error.InvalidCopyHeadTensor;
        for (gate_weight) |v| if (!std.math.isFinite(v)) return error.InvalidCopyHeadTensor;
        if (!std.math.isFinite(gate_bias)) return error.InvalidCopyHeadTensor;
        const query = try a.dupe(f32, query_weight);
        errdefer a.free(query);
        const gate = try a.dupe(f32, gate_weight);
        return .{ .allocator = a, .width = width, .query_weight = query, .gate_weight = gate, .gate_bias = gate_bias, .temperature = temperature };
    }

    /// Opens `<directory>/copy-head.safetensors`.
    pub fn load(a: A, directory: []const u8, expected_width: ?usize) !CopyHead {
        const path = try std.fs.path.join(a, &.{ directory, weights_file_name });
        defer a.free(path);
        return loadFile(a, path, expected_width);
    }

    pub fn loadFile(a: A, path: []const u8, expected_width: ?usize) !CopyHead {
        var reader = try safetensors.MMapReader.openFileAbsoluteLimited(a, path, 1024 * 1024 * 1024, 1024 * 1024);
        defer reader.deinit();
        return fromReader(a, &reader, expected_width);
    }

    /// Validates `query.weight [W,W]`, `gate.weight [1,W]` and `gate.bias [1]`
    /// (all F32, finite) and copies them out of the reader.
    pub fn fromReader(a: A, reader: *const safetensors.MMapReader, expected_width: ?usize) !CopyHead {
        try safetensors.validateReader(a, reader);
        const query_meta = reader.header.tensors.get("query.weight") orelse return error.MissingCopyHeadTensor;
        if (query_meta.shape.len != 2 or query_meta.shape[0] != query_meta.shape[1] or query_meta.shape[0] < 1) return error.InvalidCopyHeadShape;
        const width: usize = @intCast(query_meta.shape[0]);
        if (expected_width) |expected| if (expected != width) return error.InvalidCopyHeadShape;
        const w: i64 = @intCast(width);
        const query = try readFloat32(a, reader, "query.weight", &[_]i64{ w, w });
        defer a.free(query);
        const gate = try readFloat32(a, reader, "gate.weight", &[_]i64{ 1, w });
        defer a.free(gate);
        const bias = try readFloat32(a, reader, "gate.bias", &[_]i64{1});
        defer a.free(bias);
        return init(a, width, query, gate, bias[0], default_temperature);
    }

    pub fn deinit(self: *CopyHead) void {
        self.allocator.free(self.query_weight);
        self.allocator.free(self.gate_weight);
        self.* = undefined;
    }
};

fn readFloat32(a: A, reader: *const safetensors.MMapReader, name: []const u8, shape: []const i64) ![]f32 {
    var tensor = reader.readTensor(name) catch |err| return if (err == error.TensorNotFound) error.MissingCopyHeadTensor else err;
    defer tensor.deinit();
    if (tensor.dtype != .f32) return error.InvalidCopyHeadDType;
    if (!std.mem.eql(i64, tensor.shape, shape)) return error.InvalidCopyHeadShape;
    const count = tensor.data.len / @sizeOf(f32);
    if (count * @sizeOf(f32) != tensor.data.len) return error.InvalidCopyHeadTensor;
    const out = try a.alloc(f32, count);
    errdefer a.free(out);
    // Byte copy: mmap'd tensor data need not be 4-byte aligned.
    @memcpy(std.mem.sliceAsBytes(out), tensor.data[0 .. count * @sizeOf(f32)]);
    for (out) |v| if (!std.math.isFinite(v)) return error.InvalidCopyHeadTensor;
    return out;
}

// --- Numerics --------------------------------------------------------------

pub fn dot(x: []const f32, y: []const f32) f32 {
    std.debug.assert(x.len == y.len);
    const lanes = 16;
    const V = @Vector(lanes, f32);
    var acc: V = @splat(0);
    var i: usize = 0;
    while (i + lanes <= x.len) : (i += lanes) {
        const xv: V = x[i..][0..lanes].*;
        const yv: V = y[i..][0..lanes].*;
        acc += xv * yv;
    }
    var sum = @reduce(.Add, acc);
    while (i < x.len) : (i += 1) sum += x[i] * y[i];
    return sum;
}

pub fn norm(x: []const f32) f32 {
    return @sqrt(dot(x, x));
}

/// `torch.nn.functional.normalize(x, dim=-1)`: `x / max(|x|, 1e-12)` per row.
pub fn normalizeRows(data: []f32, rows: usize, width: usize) void {
    std.debug.assert(data.len == rows * width);
    var r: usize = 0;
    while (r < rows) : (r += 1) {
        const row = data[r * width ..][0..width];
        const scale = 1.0 / @max(norm(row), normalize_eps);
        for (row) |*v| v.* *= scale;
    }
}

/// In-place softmax of one row (max-subtracted).
pub fn softmaxRow(row: []f32) void {
    if (row.len == 0) return;
    var m = row[0];
    for (row[1..]) |v| m = @max(m, v);
    var sum: f32 = 0;
    for (row) |*v| {
        v.* = @exp(v.* - m);
        sum += v.*;
    }
    for (row) |*v| v.* /= sum;
}

/// In-place log-softmax of one row: `x - max - log(sum(exp(x - max)))`.
pub fn logSoftmaxRow(row: []f32) void {
    if (row.len == 0) return;
    var m = row[0];
    for (row[1..]) |v| m = @max(m, v);
    var sum: f32 = 0;
    for (row) |v| sum += @exp(v - m);
    const shift = m + @log(sum);
    for (row) |*v| v.* -= shift;
}

pub fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

/// First index of the maximum (torch.argmax tie rule on contiguous rows).
pub fn argmaxFirst(row: []const f32) usize {
    var best: usize = 0;
    for (row[1..], 1..) |v, i| {
        if (v > row[best]) best = i;
    }
    return best;
}

// --- Forward -----------------------------------------------------------------

pub const Forward = struct {
    allocator: A,
    slots: usize,
    sources: usize,
    width: usize,
    /// `[slots, sources]` cosine / temperature scores (pre-softmax).
    logits: []f32,
    /// `[slots]` sigmoid gate.
    gate: []f32,
    /// `[slots, width]` `g * (probs @ E) + (1 - g) * m`.
    soft_output: []f32,

    pub fn deinit(self: *Forward) void {
        self.allocator.free(self.logits);
        self.allocator.free(self.gate);
        self.allocator.free(self.soft_output);
        self.* = undefined;
    }
};

/// `memories @ query.weight^T` as `[slots, width]`, not normalized. This is
/// the one large matmul of the head (16 x 4096 x 4096); it runs over a few
/// threads when the machine has them and falls back to the calling thread.
pub fn projectQuery(a: A, head: *const CopyHead, memories: []const f32, slots: usize) ![]f32 {
    const width = head.width;
    if (slots == 0 or memories.len != slots * width) return error.InvalidCopyGeometry;
    const out = try a.alloc(f32, slots * width);
    errdefer a.free(out);
    const job = ProjectJob{ .head = head, .memories = memories, .slots = slots, .out = out };
    const work = slots * width * width;
    var threads: usize = 1;
    if (work >= (1 << 22)) {
        threads = @min(std.Thread.getCpuCount() catch 1, @as(usize, 8));
        threads = @min(threads, width);
    }
    if (threads <= 1) {
        job.run(0, width);
        return out;
    }
    var handles: [8]?std.Thread = @splat(null);
    const chunk = (width + threads - 1) / threads;
    var begin: usize = 0;
    var index: usize = 0;
    while (begin < width) : (index += 1) {
        const end = @min(width, begin + chunk);
        if (index + 1 < threads and end < width) {
            handles[index] = std.Thread.spawn(.{}, ProjectJob.run, .{ job, begin, end }) catch null;
            if (handles[index] == null) job.run(begin, end);
        } else {
            job.run(begin, end);
        }
        begin = end;
    }
    for (handles) |handle| {
        if (handle) |h| h.join();
    }
    return out;
}

const ProjectJob = struct {
    head: *const CopyHead,
    memories: []const f32,
    slots: usize,
    out: []f32,

    /// Fills output dimensions `row_begin..row_end` of every slot.
    fn run(self: ProjectJob, row_begin: usize, row_end: usize) void {
        const width = self.head.width;
        var row = row_begin;
        while (row < row_end) : (row += 1) {
            const weights = self.head.query_weight[row * width ..][0..width];
            var s: usize = 0;
            while (s < self.slots) : (s += 1) {
                self.out[s * width + row] = dot(weights, self.memories[s * width ..][0..width]);
            }
        }
    }
};

/// Section 1.5: `uq = normalize(m @ Wq^T)`, `ut = normalize(E)`,
/// `logits = uq @ ut^T / tau`, `probs = softmax(logits)`, `copied = probs @ E`,
/// `g = sigmoid(m . w + b)`, `soft = g * copied + (1 - g) * m`.
pub fn forward(a: A, head: *const CopyHead, memories: []const f32, slots: usize, sources: []const f32, source_count: usize) !Forward {
    const projected = try projectQuery(a, head, memories, slots);
    defer a.free(projected);
    return forwardProjected(a, head, projected, memories, slots, sources, source_count);
}

/// Same as `forward` for a caller that already computed `memories @ Wq^T`
/// (for example on the device). `projected_query` is `[slots, width]` and
/// may be unnormalized; normalization happens here.
pub fn forwardProjected(a: A, head: *const CopyHead, projected_query: []const f32, memories: []const f32, slots: usize, sources: []const f32, source_count: usize) !Forward {
    const width = head.width;
    if (slots == 0 or source_count == 0) return error.InvalidCopyGeometry;
    if (memories.len != slots * width or projected_query.len != slots * width or sources.len != source_count * width) return error.InvalidCopyGeometry;

    const unit_query = try a.dupe(f32, projected_query);
    defer a.free(unit_query);
    normalizeRows(unit_query, slots, width);
    const unit_tokens = try a.dupe(f32, sources);
    defer a.free(unit_tokens);
    normalizeRows(unit_tokens, source_count, width);

    const logits = try a.alloc(f32, slots * source_count);
    errdefer a.free(logits);
    const gate = try a.alloc(f32, slots);
    errdefer a.free(gate);
    const soft = try a.alloc(f32, slots * width);
    errdefer a.free(soft);
    const probs = try a.alloc(f32, source_count);
    defer a.free(probs);

    const inv_tau = 1.0 / head.temperature;
    var s: usize = 0;
    while (s < slots) : (s += 1) {
        const q = unit_query[s * width ..][0..width];
        const m = memories[s * width ..][0..width];
        const row = logits[s * source_count ..][0..source_count];
        for (row, 0..) |*l, t| l.* = dot(q, unit_tokens[t * width ..][0..width]) * inv_tau;
        @memcpy(probs, row);
        softmaxRow(probs);
        const g = sigmoid(dot(m, head.gate_weight) + head.gate_bias);
        gate[s] = g;
        const out = soft[s * width ..][0..width];
        @memset(out, 0);
        for (probs, 0..) |p, t| {
            const e = sources[t * width ..][0..width];
            for (out, e) |*o, v| o.* += p * v;
        }
        for (out, m) |*o, v| o.* = g * o.* + (1 - g) * v;
    }
    return .{ .allocator = a, .slots = slots, .sources = source_count, .width = width, .logits = logits, .gate = gate, .soft_output = soft };
}

// --- Contiguous runs (section 1.6 a) -----------------------------------------

pub const Runs = struct {
    allocator: A,
    /// `[slots, width]`: hard gate-mixed copies on run slots, soft output elsewhere.
    output: []f32,
    /// Sorted by `begin`.
    spans: []Span,
    /// Run slots whose chosen source differs from their own argmax.
    moved: usize,

    pub fn deinit(self: *Runs) void {
        self.allocator.free(self.output);
        self.allocator.free(self.spans);
        self.* = undefined;
    }
};

pub fn contiguousRunsForward(a: A, fwd: *const Forward, memories: []const f32, sources: []const f32, min_gate: f32) !Runs {
    return contiguousRunsForwardWithMargin(a, fwd, memories, sources, min_gate, default_bridge_margin);
}

pub fn contiguousRunsForwardWithMargin(a: A, fwd: *const Forward, memories: []const f32, sources: []const f32, min_gate: f32, bridge_margin: f32) !Runs {
    return contiguousRunsWithMargin(a, fwd.logits, fwd.gate, fwd.soft_output, memories, sources, fwd.slots, fwd.sources, fwd.width, min_gate, bridge_margin);
}

/// `copy_with_contiguous_runs(min_gate)` with `extend=False`, no boundary,
/// label or jump handling. `log_probs = log_softmax(logits)`; slots with
/// `gate >= min_gate` form maximal runs; a run of length L <= T scores each
/// start `s in 0..T-L` by the summed diagonal log probability. Runs are
/// processed by descending best score (stable, so ties keep slot order);
/// within a run starts are tried by descending score and the first finite
/// start whose positions are all unclaimed wins, else the first finite start
/// (overlap allowed), else the run is skipped. Python orders the starts with
/// `torch.argsort(descending=True)`, whose tie order is unspecified; here a
/// stable sort keeps the lower start first among equal scores. Runs longer
/// than T keep the soft output and yield no span.
pub fn contiguousRuns(a: A, logits: []const f32, gate: []const f32, soft_output: []const f32, memories: []const f32, sources: []const f32, slots: usize, source_count: usize, width: usize, min_gate: f32) !Runs {
    return contiguousRunsWithMargin(a, logits, gate, soft_output, memories, sources, slots, source_count, width, min_gate, default_bridge_margin);
}

/// `contiguousRuns` with the "an uncertain slot cannot bridge two confident
/// runs" rule. A copying slot is *uncertain* when
/// `min_gate <= gate < min_gate + bridge_margin` and *confident* when
/// `gate >= min_gate + bridge_margin`. Within every maximal run of copying
/// slots, each uncertain slot that has a confident slot on both sides inside
/// that run stops copying: it keeps the soft output, claims no source
/// position and splits the run there. Uncertain slots at a run's ends and
/// runs made only of uncertain slots are unchanged, as is everything after
/// the run partition (scoring, claiming, spans). `bridge_margin = 0` is the
/// plain rule.
pub fn contiguousRunsWithMargin(a: A, logits: []const f32, gate: []const f32, soft_output: []const f32, memories: []const f32, sources: []const f32, slots: usize, source_count: usize, width: usize, min_gate: f32, bridge_margin: f32) !Runs {
    if (slots == 0 or source_count == 0) return error.InvalidCopyGeometry;
    if (logits.len != slots * source_count or gate.len != slots or soft_output.len != slots * width or memories.len != slots * width or sources.len != source_count * width) return error.InvalidCopyGeometry;
    if (!(min_gate >= 0 and min_gate <= 1)) return error.InvalidCopyGateThreshold;
    if (!(bridge_margin >= 0 and bridge_margin <= 1)) return error.InvalidCopyBridgeMargin;

    const output = try a.dupe(f32, soft_output);
    errdefer a.free(output);
    const log_probs = try a.dupe(f32, logits);
    defer a.free(log_probs);
    const argmax = try a.alloc(usize, slots);
    defer a.free(argmax);
    var s: usize = 0;
    while (s < slots) : (s += 1) {
        const row = log_probs[s * source_count ..][0..source_count];
        logSoftmaxRow(row);
        argmax[s] = argmaxFirst(row);
    }

    const copying = try a.alloc(bool, slots);
    defer a.free(copying);
    for (copying, gate) |*flag, g| flag.* = g >= min_gate;
    if (bridge_margin > 0) splitBridgedRuns(copying, gate, min_gate + bridge_margin);

    // Maximal runs of copying slots.
    var runs = std.ArrayListUnmanaged(ScoredRun).empty;
    defer {
        for (runs.items) |run| a.free(run.scores);
        runs.deinit(a);
    }
    var run_start: ?usize = null;
    s = 0;
    while (s <= slots) : (s += 1) {
        const flag = s < slots and copying[s];
        if (flag and run_start == null) {
            run_start = s;
        } else if (!flag and run_start != null) {
            const begin = run_start.?;
            run_start = null;
            const length = s - begin;
            if (length > source_count) continue;
            const starts = source_count - length + 1;
            const scores = try a.alloc(f32, starts);
            errdefer a.free(scores);
            var best: f32 = -std.math.inf(f32);
            for (scores, 0..) |*score, start| {
                var sum: f32 = 0;
                var k: usize = 0;
                while (k < length) : (k += 1) sum += log_probs[(begin + k) * source_count + start + k];
                score.* = sum;
                best = @max(best, sum);
            }
            try runs.append(a, .{ .begin = begin, .end = s, .scores = scores, .best = best });
        }
    }
    std.mem.sort(ScoredRun, runs.items, {}, ScoredRun.betterFirst);

    const taken = try a.alloc(bool, source_count);
    defer a.free(taken);
    @memset(taken, false);
    const order = try a.alloc(usize, source_count);
    defer a.free(order);
    var spans = std.ArrayListUnmanaged(Span).empty;
    defer spans.deinit(a);
    var moved: usize = 0;
    for (runs.items) |run| {
        const length = run.end - run.begin;
        const starts = run.scores.len;
        const candidates = order[0..starts];
        for (candidates, 0..) |*c, i| c.* = i;
        std.mem.sort(usize, candidates, @as([]const f32, run.scores), descendingByScore);
        var chosen: ?usize = null;
        for (candidates) |start| {
            if (!std.math.isFinite(run.scores[start])) continue;
            var free = true;
            var k: usize = 0;
            while (k < length) : (k += 1) {
                if (taken[start + k]) {
                    free = false;
                    break;
                }
            }
            if (free) {
                chosen = start;
                break;
            }
        }
        if (chosen == null) {
            for (candidates) |start| {
                if (std.math.isFinite(run.scores[start])) {
                    chosen = start;
                    break;
                }
            }
        }
        const start = chosen orelse continue;
        try spans.append(a, .{ .begin = run.begin, .end = run.end, .start = start });
        var k: usize = 0;
        while (k < length) : (k += 1) {
            const slot = run.begin + k;
            const source = start + k;
            taken[source] = true;
            if (argmax[slot] != source) moved += 1;
            const g = gate[slot];
            const out = output[slot * width ..][0..width];
            const e = sources[source * width ..][0..width];
            const m = memories[slot * width ..][0..width];
            for (out, e, m) |*o, ev, mv| o.* = g * ev + (1 - g) * mv;
        }
    }
    std.mem.sort(Span, spans.items, {}, spanLessThan);
    return .{ .allocator = a, .output = output, .spans = try spans.toOwnedSlice(a), .moved = moved };
}

/// Clears `copying` for every uncertain slot (copying, `gate < confident`)
/// strictly between the first and the last confident slot of its maximal run.
/// After one pass every remaining uncertain slot of a run lies before its
/// first or after its last confident slot, so no second pass is needed.
fn splitBridgedRuns(copying: []bool, gate: []const f32, confident: f32) void {
    var s: usize = 0;
    while (s < copying.len) {
        if (!copying[s]) {
            s += 1;
            continue;
        }
        const begin = s;
        while (s < copying.len and copying[s]) s += 1;
        const end = s;
        var first: ?usize = null;
        var last: usize = begin;
        var i = begin;
        while (i < end) : (i += 1) {
            if (gate[i] >= confident) {
                if (first == null) first = i;
                last = i;
            }
        }
        const fc = first orelse continue;
        i = fc + 1;
        while (i < last) : (i += 1) {
            if (gate[i] < confident) copying[i] = false;
        }
    }
}

const ScoredRun = struct {
    begin: usize,
    end: usize,
    scores: []f32,
    best: f32,

    fn betterFirst(_: void, x: ScoredRun, y: ScoredRun) bool {
        return x.best > y.best;
    }
};

fn descendingByScore(scores: []const f32, x: usize, y: usize) bool {
    return scores[x] > scores[y];
}

fn spanLessThan(_: void, x: Span, y: Span) bool {
    if (x.begin != y.begin) return x.begin < y.begin;
    if (x.end != y.end) return x.end < y.end;
    return x.start < y.start;
}

// --- Snapping (section 1.6 b) -----------------------------------------------

/// L2 norm of every row of a `[rows, width]` table, computed once per table.
pub fn rowNorms(a: A, table: []const f32, rows: usize, width: usize) ![]f32 {
    if (table.len != rows * width) return error.InvalidCopyGeometry;
    const norms = try a.alloc(f32, rows);
    for (norms, 0..) |*n, r| n.* = norm(table[r * width ..][0..width]);
    return norms;
}

/// `scores[s*vocab + v] = output[s] . table[v]` (raw dot products, typically
/// from a device matmul) -> the row id maximizing
/// `score / (max(|output|, eps) * max(|table|, eps))` per slot; the first
/// maximum wins on ties, as `torch.max` does.
pub fn snapIds(a: A, scores: []const f32, table_norms: []const f32, output_norms: []const f32, slots: usize, vocab: usize) ![]i32 {
    if (slots == 0 or vocab == 0 or scores.len != slots * vocab or table_norms.len != vocab or output_norms.len != slots) return error.InvalidCopyGeometry;
    const ids = try a.alloc(i32, slots);
    errdefer a.free(ids);
    const inv_table = try a.alloc(f32, vocab);
    defer a.free(inv_table);
    for (inv_table, table_norms) |*inv, n| inv.* = 1.0 / @max(n, normalize_eps);
    for (ids, 0..) |*id, s| {
        const inv_out = 1.0 / @max(output_norms[s], normalize_eps);
        const row = scores[s * vocab ..][0..vocab];
        var best: usize = 0;
        var best_cos = row[0] * inv_table[0] * inv_out;
        for (row[1..], 1..) |score, v| {
            const cosine = score * inv_table[v] * inv_out;
            if (cosine > best_cos) {
                best_cos = cosine;
                best = v;
            }
        }
        id.* = @intCast(best);
    }
    return ids;
}

/// Host reference: nearest table row by cosine for every output row.
pub fn cosineArgmax(a: A, output: []const f32, table: []const f32, slots: usize, vocab: usize, width: usize) ![]i32 {
    if (output.len != slots * width or table.len != vocab * width) return error.InvalidCopyGeometry;
    const table_norms = try rowNorms(a, table, vocab, width);
    defer a.free(table_norms);
    const output_norms = try rowNorms(a, output, slots, width);
    defer a.free(output_norms);
    const scores = try a.alloc(f32, slots * vocab);
    defer a.free(scores);
    for (0..slots) |s| {
        for (0..vocab) |v| scores[s * vocab + v] = dot(output[s * width ..][0..width], table[v * width ..][0..width]);
    }
    return snapIds(a, scores, table_norms, output_norms, slots, vocab);
}

/// Callback that maps `[slots, width]` output rows to token ids, so the
/// embedding-table matmul can live on a device.
pub const Snapper = struct {
    ptr: *anyopaque,
    snap: *const fn (ptr: *anyopaque, a: A, output: []const f32, slots: usize, width: usize) anyerror![]i32,
};

/// Host snapper over an in-memory `[vocab, width]` table (tests and small
/// tools).
pub const TableSnapper = struct {
    table: []const f32,
    vocab: usize,

    pub fn snapper(self: *TableSnapper) Snapper {
        return .{ .ptr = self, .snap = snap };
    }

    fn snap(ptr: *anyopaque, a: A, output: []const f32, slots: usize, width: usize) anyerror![]i32 {
        const self: *TableSnapper = @ptrCast(@alignCast(ptr));
        return cosineArgmax(a, output, self.table, slots, self.vocab, width);
    }
};

// --- Word boundaries (section 1.6 c) -----------------------------------------

/// Python `str.isalnum` is Unicode-aware. ASCII letters and digits are exact;
/// outside ASCII this approximates: Latin-1 Supplement letters and
/// superscript digits, Latin Extended-A/B (U+0100..U+024F) count as
/// alphanumeric and every other code point does not.
pub fn isWordCodepoint(cp: u21) bool {
    if (cp < 0x80) return std.ascii.isAlphanumeric(@intCast(cp));
    return switch (cp) {
        0xAA, 0xB2, 0xB3, 0xB5, 0xB9, 0xBA => true,
        0xC0...0xD6, 0xD8...0xF6, 0xF8...0x24F => true,
        else => false,
    };
}

fn allWordCodepoints(body: []const u8, allow_hyphen: bool) bool {
    const view = std.unicode.Utf8View.init(body) catch return false;
    var it = view.iterator();
    while (it.nextCodepoint()) |cp| {
        if (allow_hyphen and cp == '-') continue;
        if (!isWordCodepoint(cp)) return false;
    }
    return true;
}

/// `token_continues_word`: the decoded piece has no leading space and its
/// trimmed body is non-empty and made only of alphanumerics or '-'. Special
/// tokens decode to "" and are never word continuations. `strip()` is
/// approximated with ASCII whitespace.
pub fn tokenContinuesWord(a: A, tokenizer: tok.Tokenizer, id: i32) !bool {
    const text = try tokenizer.decodeToken(a, id, false);
    defer a.free(text);
    const body = std.mem.trim(u8, text, &std.ascii.whitespace);
    return body.len > 0 and !std.mem.startsWith(u8, text, " ") and allWordCodepoints(body, true);
}

/// `token_starts_word`: leading space and an alphanumeric body.
pub fn tokenStartsWord(a: A, tokenizer: tok.Tokenizer, id: i32) !bool {
    const text = try tokenizer.decodeToken(a, id, false);
    defer a.free(text);
    const body = std.mem.trim(u8, text, &std.ascii.whitespace);
    return std.mem.startsWith(u8, text, " ") and body.len > 0 and allWordCodepoints(body, false);
}

pub const Completed = struct {
    allocator: A,
    /// `[slots]` token ids.
    ids: []i32,
    /// Source tokens added by word extension.
    inserted: usize,

    pub fn deinit(self: *Completed) void {
        self.allocator.free(self.ids);
        self.* = undefined;
    }
};

/// `complete_copied_words(overflow="truncate")`. Spans are walked in
/// `(begin, end, start)` order: snapped ids before `begin` are kept, each
/// span is extended backwards while `continues(src[first])` and forwards
/// while `continues(src[last])`, the extended SOURCE ids are emitted, and the
/// cursor moves to `end`. The tail follows; trailing pads are dropped while
/// the record is too long, then the record is truncated to `slot_ids.len`
/// and padded with `pad_id`. Spans must satisfy `0 <= begin < end <= S` and
/// `start + L <= T`; overlapping spans are not rejected (as in Python) and
/// emit nothing for the already-consumed prefix.
pub fn completeCopiedWords(a: A, tokenizer: tok.Tokenizer, slot_ids: []const i32, spans: []const Span, source_ids: []const i32, pad_id: i32) !Completed {
    const size = slot_ids.len;
    const sorted = try a.dupe(Span, spans);
    defer a.free(sorted);
    std.mem.sort(Span, sorted, {}, spanLessThan);
    for (sorted) |span| {
        if (!(span.begin < span.end and span.end <= size)) return error.InvalidCopySpan;
        if (span.length() > source_ids.len or span.start > source_ids.len - span.length()) return error.InvalidCopySpan;
    }

    var pieces = std.ArrayListUnmanaged(i32).empty;
    defer pieces.deinit(a);
    var cursor: usize = 0;
    var inserted: usize = 0;
    for (sorted) |span| {
        if (span.begin > cursor) try pieces.appendSlice(a, slot_ids[cursor..span.begin]);
        var first = span.start;
        var last = span.start + span.length();
        while (first > 0 and try tokenContinuesWord(a, tokenizer, source_ids[first])) {
            first -= 1;
            inserted += 1;
        }
        while (last < source_ids.len and try tokenContinuesWord(a, tokenizer, source_ids[last])) {
            last += 1;
            inserted += 1;
        }
        try pieces.appendSlice(a, source_ids[first..last]);
        cursor = span.end;
    }
    if (cursor < size) try pieces.appendSlice(a, slot_ids[cursor..]);
    while (pieces.items.len > size and pieces.items.len > 0 and pieces.items[pieces.items.len - 1] == pad_id) _ = pieces.pop();
    if (pieces.items.len > size) pieces.shrinkRetainingCapacity(size);
    while (pieces.items.len < size) try pieces.append(a, pad_id);
    return .{ .allocator = a, .ids = try pieces.toOwnedSlice(a), .inserted = inserted };
}

// --- Decode ------------------------------------------------------------------

pub const Decoded = struct {
    allocator: A,
    runs: Runs,
    /// `[slots]` nearest-token ids of `runs.output`.
    snapped_ids: []i32,
    /// `[slots]` ids after word completion (equal to `snapped_ids` without spans).
    completed_ids: []i32,
    inserted: usize,

    pub fn deinit(self: *Decoded) void {
        self.runs.deinit();
        self.allocator.free(self.snapped_ids);
        self.allocator.free(self.completed_ids);
        self.* = undefined;
    }
};

/// Section 1.6 end to end: contiguous runs -> snap every slot -> complete
/// copied words (only when spans exist). `source_ids` are the encoder ids of
/// the source rows, in the order of `sources`.
pub fn decode(a: A, tokenizer: tok.Tokenizer, fwd: *const Forward, memories: []const f32, sources: []const f32, source_ids: []const i32, snapper: Snapper, min_gate: f32, pad_id: i32) !Decoded {
    if (source_ids.len != fwd.sources) return error.InvalidCopyGeometry;
    var runs = try contiguousRunsForward(a, fwd, memories, sources, min_gate);
    errdefer runs.deinit();
    const snapped = try snapper.snap(snapper.ptr, a, runs.output, fwd.slots, fwd.width);
    errdefer a.free(snapped);
    if (snapped.len != fwd.slots) return error.InvalidCopyGeometry;
    return decodeWithIds(a, tokenizer, runs, snapped, source_ids, pad_id);
}

/// Second half of `decode` for a caller that snapped `runs.output` itself.
/// Takes ownership of `runs` and `snapped_ids`, also on error.
pub fn decodeWithIds(a: A, tokenizer: tok.Tokenizer, runs: Runs, snapped_ids: []i32, source_ids: []const i32, pad_id: i32) !Decoded {
    var owned_runs = runs;
    errdefer owned_runs.deinit();
    errdefer a.free(snapped_ids);
    if (owned_runs.spans.len == 0) {
        const copy = try a.dupe(i32, snapped_ids);
        return .{ .allocator = a, .runs = owned_runs, .snapped_ids = snapped_ids, .completed_ids = copy, .inserted = 0 };
    }
    const completed = try completeCopiedWords(a, tokenizer, snapped_ids, owned_runs.spans, source_ids, pad_id);
    return .{ .allocator = a, .runs = owned_runs, .snapped_ids = snapped_ids, .completed_ids = completed.ids, .inserted = completed.inserted };
}

// --- Tests -------------------------------------------------------------------

const testing = std.testing;

/// Fake decoder tokenizer: a fixed id -> piece table (pieces keep their
/// leading space; the empty piece stands for a special token).
const WordTokenizer = struct {
    words: []const []const u8,

    fn encode(_: *anyopaque, _: A, _: []const u8) ![]i32 {
        return error.NotUsed;
    }
    fn append(_: *anyopaque, _: A, _: []const u8, _: *std.ArrayListUnmanaged(i32)) !void {
        return error.NotUsed;
    }
    fn decodeIds(raw: *anyopaque, a: A, ids: []const i32) ![]u8 {
        const self: *WordTokenizer = @ptrCast(@alignCast(raw));
        var out = std.ArrayListUnmanaged(u8).empty;
        errdefer out.deinit(a);
        for (ids) |id| {
            if (id < 0 or @as(usize, @intCast(id)) >= self.words.len) return error.UnknownToken;
            try out.appendSlice(a, self.words[@intCast(id)]);
        }
        return out.toOwnedSlice(a);
    }
    fn unused(_: *anyopaque, _: A, _: []const u8, _: usize) !tok.EncodeResult {
        return error.NotUsed;
    }
    fn unusedGeneration(_: *anyopaque, _: A, _: []const u8, _: usize, _: bool) !tok.EncodeResult {
        return error.NotUsed;
    }
    fn specials(_: *anyopaque) tok.SpecialTokens {
        return .{};
    }
    fn size(raw: *anyopaque) usize {
        const self: *WordTokenizer = @ptrCast(@alignCast(raw));
        return self.words.len;
    }
    fn deinit(_: *anyopaque) void {}
    fn handle(self: *WordTokenizer) tok.Tokenizer {
        return .{ .ptr = self, .vtable = &.{ .encode = encode, .encodeInto = append, .encodeForModel = unused, .encodeGeneration = unusedGeneration, .decode = decodeIds, .specialTokens = specials, .vocabSize = size, .deinit = deinit } };
    }
};

/// The id table of `test_completed_words_extend_copied_spans_to_word_boundaries`
/// plus 17 = "é" (Latin-1 letter), 18 = "日本" (outside the approximation),
/// 19 = "" (a special token) and 20 = " -00".
const python_words = [_][]const u8{ " ", " key", " holder", " Kit", "ima", " Dab", "oco", " at", " Mo", "zeba", " Depot", ".", " code", " N", "q6", "-00", "1844", "é", "日本", "", " -00" };

fn expectSpans(expected: []const Span, actual: []const Span) !void {
    try testing.expectEqual(expected.len, actual.len);
    for (expected, actual) |e, x| {
        try testing.expectEqual(e.begin, x.begin);
        try testing.expectEqual(e.end, x.end);
        try testing.expectEqual(e.start, x.start);
    }
}

fn expectRowMix(expected_e: []const f32, expected_m: []const f32, g: f32, actual: []const f32) !void {
    for (actual, expected_e, expected_m) |o, e, m| try testing.expectApproxEqAbs(g * e + (1 - g) * m, o, 1e-6);
}

/// Identity-row sources of `width` and a head whose gate is high for unit
/// memories and low for zero memories.
fn basisHead(a: A, width: usize) !CopyHead {
    const query = try a.alloc(f32, width * width);
    defer a.free(query);
    @memset(query, 0);
    for (0..width) |i| query[i * width + i] = 1;
    const gate = try a.alloc(f32, width);
    defer a.free(gate);
    @memset(gate, 12);
    return CopyHead.init(a, width, query, gate, -6, default_temperature);
}

fn basisRows(a: A, width: usize, rows: []const ?usize) ![]f32 {
    const out = try a.alloc(f32, rows.len * width);
    @memset(out, 0);
    for (rows, 0..) |row, i| {
        if (row) |r| out[i * width + r] = 1;
    }
    return out;
}

test "memorilla copy normalize softmax and log-softmax match hand-computed values" {
    var rows = [_]f32{ 3, 4, 0, 0, 0, 0 };
    normalizeRows(&rows, 2, 3);
    try testing.expectApproxEqAbs(@as(f32, 0.6), rows[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.8), rows[1], 1e-6);
    try testing.expectEqual(@as(f32, 0), rows[3]);

    var soft = [_]f32{ 1, 2, 3 };
    softmaxRow(&soft);
    try testing.expectApproxEqAbs(@as(f32, 0.09003057), soft[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.24472847), soft[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.66524096), soft[2], 1e-6);

    var logs = [_]f32{ 1, 2, 3 };
    logSoftmaxRow(&logs);
    try testing.expectApproxEqAbs(@as(f32, -2.40760596), logs[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -0.40760596), logs[2], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, 0.5), sigmoid(0), 1e-7);
    try testing.expectEqual(@as(usize, 1), argmaxFirst(&.{ 0, 2, 2, 1 }));
    try testing.expectApproxEqAbs(@as(f32, 5), norm(&.{ 3, 4 }), 1e-6);
    var long: [37]f32 = undefined;
    for (&long, 0..) |*v, i| v.* = @floatFromInt(i);
    try testing.expectApproxEqAbs(@as(f32, 36 * 37 * 73 / 6), dot(&long, &long), 1e-2);
}

test "memorilla copy head forward matches the torch reference" {
    const a = testing.allocator;
    // Values from scripts/memorilla/encoder_lora.py CopyHead.forward_detailed
    // on these inputs (torch 2.9.1, CPU, FP32).
    const query = [_]f32{ 1, 0.5, 0, 0, 0, 1, 0, 0.25, 0, 0, 1, 0, 0.5, 0, 0, 1 };
    const gate_w = [_]f32{ 0.5, -0.25, 1, 0 };
    var head = try CopyHead.init(a, 4, &query, &gate_w, 0.1, 0.05);
    defer head.deinit();
    const memories = [_]f32{ 1, 2, 0, -1, 0, 1, 1, 0.5 };
    const sources = [_]f32{ 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1 };

    const projected = try projectQuery(a, &head, &memories, 2);
    defer a.free(projected);
    const expected_projected = [_]f32{ 2, 1.75, 0, -0.5, 0.5, 1.125, 1, 0.5 };
    for (projected, expected_projected) |p, e| try testing.expectApproxEqAbs(e, p, 1e-6);

    var fwd = try forward(a, &head, &memories, 2, &sources, 3);
    defer fwd.deinit();
    const expected_logits = [_]f32{ 14.79200459, 9.15208530, 12.01850414, 6.01316786, 18.07079506, 18.79115105 };
    for (fwd.logits, expected_logits) |l, e| try testing.expectApproxEqAbs(e, l, 1e-4);
    try testing.expectApproxEqAbs(@as(f32, 0.52497917), fwd.gate[0], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, 0.70056719), fwd.gate[1], 1e-6);
    const expected_soft = [_]f32{ 0.99825019, 0.98254329, 0.03250165, -0.44426903, 0.47126177, 0.99999869, 0.99999869, 0.62097681 };
    for (fwd.soft_output, expected_soft) |o, e| try testing.expectApproxEqAbs(e, o, 1e-5);

    // The pre-projected entry point gives the same result.
    var again = try forwardProjected(a, &head, projected, &memories, 2, &sources, 3);
    defer again.deinit();
    try testing.expectEqualSlices(f32, fwd.logits, again.logits);
    try testing.expectEqualSlices(f32, fwd.soft_output, again.soft_output);

    // log_softmax of the first row, as the run scorer computes it.
    var row = [_]f32{ fwd.logits[0], fwd.logits[1], fwd.logits[2] };
    logSoftmaxRow(&row);
    try testing.expectApproxEqAbs(@as(f32, -0.06390973), row[0], 1e-5);
    try testing.expectApproxEqAbs(@as(f32, -5.70382881), row[1], 1e-4);
    try testing.expectApproxEqAbs(@as(f32, -2.83741021), row[2], 1e-4);

    try testing.expectError(error.InvalidCopyGeometry, forward(a, &head, &memories, 2, &sources, 2));
    try testing.expectError(error.InvalidCopyGeometry, forward(a, &head, memories[0..4], 2, &sources, 3));
    try testing.expectError(error.InvalidCopyHeadShape, CopyHead.init(a, 4, &query, gate_w[0..3], 0.1, 0.05));
    try testing.expectError(error.InvalidCopyHeadTemperature, CopyHead.init(a, 4, &query, &gate_w, 0.1, 0));
}

test "memorilla copy head query projection agrees across thread counts" {
    const a = testing.allocator;
    // Large enough to take the threaded path (slots*width*width >= 2^22).
    const width: usize = 512;
    const slots: usize = 16;
    const query = try a.alloc(f32, width * width);
    defer a.free(query);
    var prng = std.Random.DefaultPrng.init(17);
    const random = prng.random();
    for (query) |*v| v.* = random.float(f32) - 0.5;
    const gate = try a.alloc(f32, width);
    defer a.free(gate);
    @memset(gate, 0);
    var head = try CopyHead.init(a, width, query, gate, 0, default_temperature);
    defer head.deinit();
    const memories = try a.alloc(f32, slots * width);
    defer a.free(memories);
    for (memories) |*v| v.* = random.float(f32) - 0.5;
    const projected = try projectQuery(a, &head, memories, slots);
    defer a.free(projected);
    const serial = try a.alloc(f32, slots * width);
    defer a.free(serial);
    (ProjectJob{ .head = &head, .memories = memories, .slots = slots, .out = serial }).run(0, width);
    try testing.expectEqualSlices(f32, serial, projected);
}

fn safetensorsBytes(a: A, header: []const u8, payload: []const f32) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(a);
    var size: [8]u8 = undefined;
    std.mem.writeInt(u64, &size, header.len, .little);
    try out.appendSlice(a, &size);
    try out.appendSlice(a, header);
    try out.appendSlice(a, std.mem.sliceAsBytes(payload));
    return out.toOwnedSlice(a);
}

const copy_head_header_w2 =
    \\{"gate.bias":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"gate.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[4,12]},"query.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[12,28]}}
;

test "memorilla copy head loads validated safetensors weights" {
    const a = testing.allocator;
    const payload = [_]f32{ -2, 0.5, 0.25, 1, 0, 0, 1 };
    const bytes = try safetensorsBytes(a, copy_head_header_w2, &payload);
    defer a.free(bytes);
    var reader = try safetensors.MMapReader.fromBorrowedBytesLimited(a, bytes, 1024);
    defer reader.deinit();
    var head = try CopyHead.fromReader(a, &reader, 2);
    defer head.deinit();
    try testing.expectEqual(@as(usize, 2), head.width);
    try testing.expectEqualSlices(f32, &.{ 1, 0, 0, 1 }, head.query_weight);
    try testing.expectEqualSlices(f32, &.{ 0.5, 0.25 }, head.gate_weight);
    try testing.expectEqual(@as(f32, -2), head.gate_bias);
    try testing.expectEqual(default_temperature, head.temperature);
    try testing.expectError(error.InvalidCopyHeadShape, CopyHead.fromReader(a, &reader, 4096));

    // Non-finite weights are rejected.
    const bad_payload = [_]f32{ -2, 0.5, std.math.nan(f32), 1, 0, 0, 1 };
    const bad = try safetensorsBytes(a, copy_head_header_w2, &bad_payload);
    defer a.free(bad);
    var bad_reader = try safetensors.MMapReader.fromBorrowedBytesLimited(a, bad, 1024);
    defer bad_reader.deinit();
    try testing.expectError(error.InvalidCopyHeadTensor, CopyHead.fromReader(a, &bad_reader, null));

    // Wrong dtype, wrong gate shape and a missing tensor are rejected.
    const bf16_header =
        \\{"gate.bias":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"gate.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[4,12]},"query.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[12,20]}}
    ;
    const bf16 = try safetensorsBytes(a, bf16_header, payload[0..5]);
    defer a.free(bf16);
    var bf16_reader = try safetensors.MMapReader.fromBorrowedBytesLimited(a, bf16, 1024);
    defer bf16_reader.deinit();
    try testing.expectError(error.InvalidCopyHeadDType, CopyHead.fromReader(a, &bf16_reader, null));
    const gate_header =
        \\{"gate.bias":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"gate.weight":{"dtype":"F32","shape":[2],"data_offsets":[4,12]},"query.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[12,28]}}
    ;
    const gate = try safetensorsBytes(a, gate_header, &payload);
    defer a.free(gate);
    var gate_reader = try safetensors.MMapReader.fromBorrowedBytesLimited(a, gate, 1024);
    defer gate_reader.deinit();
    try testing.expectError(error.InvalidCopyHeadShape, CopyHead.fromReader(a, &gate_reader, null));
    const missing_header =
        \\{"gate.weight":{"dtype":"F32","shape":[1,2],"data_offsets":[0,8]},"query.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[8,24]}}
    ;
    const missing = try safetensorsBytes(a, missing_header, payload[1..]);
    defer a.free(missing);
    var missing_reader = try safetensors.MMapReader.fromBorrowedBytesLimited(a, missing, 1024);
    defer missing_reader.deinit();
    try testing.expectError(error.MissingCopyHeadTensor, CopyHead.fromReader(a, &missing_reader, null));

    // The directory loader reads copy-head.safetensors.
    const io = testing.io;
    var temp = testing.tmpDir(.{});
    defer temp.cleanup();
    try temp.dir.writeFile(io, .{ .sub_path = weights_file_name, .data = bytes });
    const directory = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", temp.sub_path[0..] });
    defer a.free(directory);
    var loaded = try CopyHead.load(a, directory, 2);
    defer loaded.deinit();
    try testing.expectEqualSlices(f32, head.query_weight, loaded.query_weight);
    try testing.expectEqual(head.gate_bias, loaded.gate_bias);
}

test "memorilla copy contiguous runs read one source span per copying run" {
    // Port of test_contiguous_runs_read_one_source_span_per_copying_run with
    // basis-vector sources: run A = slots 0..2 should read positions 2,3,4 but
    // slot 1 points at position 6; slot 3 generates; run B = slots 4..5
    // reads positions 0,1.
    const a = testing.allocator;
    const width: usize = 8;
    var head = try basisHead(a, width);
    defer head.deinit();
    const sources = try basisRows(a, width, &.{ 0, 1, 2, 3, 4, 5, 6, 7 });
    defer a.free(sources);
    const memories = try basisRows(a, width, &.{ 2, 6, 4, null, 0, 1 });
    defer a.free(memories);
    var fwd = try forward(a, &head, memories, 6, sources, 8);
    defer fwd.deinit();
    try testing.expect(fwd.gate[0] > 0.99 and fwd.gate[3] < 0.01);

    var runs = try contiguousRunsForward(a, &fwd, memories, sources, default_min_gate);
    defer runs.deinit();
    try testing.expectEqual(@as(usize, 1), runs.moved);
    try expectSpans(&.{ .{ .begin = 0, .end = 3, .start = 2 }, .{ .begin = 4, .end = 6, .start = 0 } }, runs.spans);
    const row = struct {
        fn of(data: []const f32, i: usize) []const f32 {
            return data[i * 8 ..][0..8];
        }
    }.of;
    try expectRowMix(row(sources, 3), row(memories, 1), fwd.gate[1], row(runs.output, 1));
    try expectRowMix(row(sources, 2), row(memories, 0), fwd.gate[0], row(runs.output, 0));
    try expectRowMix(row(sources, 4), row(memories, 2), fwd.gate[2], row(runs.output, 2));
    try expectRowMix(row(sources, 0), row(memories, 4), fwd.gate[4], row(runs.output, 4));
    try expectRowMix(row(sources, 1), row(memories, 5), fwd.gate[5], row(runs.output, 5));
    // The generating slot keeps the plain soft output exactly.
    try testing.expectEqualSlices(f32, row(fwd.soft_output, 3), row(runs.output, 3));

    // Slots below the threshold never claim positions.
    var none = try contiguousRunsForward(a, &fwd, memories, sources, 1.0);
    defer none.deinit();
    try testing.expectEqual(@as(usize, 0), none.spans.len);
    try testing.expectEqual(@as(usize, 0), none.moved);
    try testing.expectEqualSlices(f32, fwd.soft_output, none.output);

    try testing.expectError(error.InvalidCopyGeometry, contiguousRuns(a, fwd.logits, fwd.gate, fwd.soft_output, memories[0..width], sources, 6, 8, width, 0.5));
    try testing.expectError(error.InvalidCopyGateThreshold, contiguousRunsForward(a, &fwd, memories, sources, 1.5));
}

test "memorilla copy contiguous runs fall back to overlapping starts and skip runs longer than the source" {
    const a = testing.allocator;
    const width: usize = 4;
    var head = try basisHead(a, width);
    defer head.deinit();
    const sources = try basisRows(a, width, &.{ 0, 1 });
    defer a.free(sources);

    // Run A = slots 0..1 reads 0,1 with full confidence; slot 2 generates;
    // run B = slot 3 wants position 1 with less confidence (memory e1 +
    // 0.5 e0). A claims both positions, so B falls back to its best start
    // although it overlaps.
    const memories = try a.alloc(f32, 4 * width);
    defer a.free(memories);
    @memset(memories, 0);
    memories[0 * width + 0] = 1;
    memories[1 * width + 1] = 1;
    memories[3 * width + 1] = 1;
    memories[3 * width + 0] = 0.5;
    var fwd = try forward(a, &head, memories, 4, sources, 2);
    defer fwd.deinit();
    try testing.expect(fwd.gate[2] < 0.5 and fwd.gate[3] > 0.5);
    var runs = try contiguousRunsForward(a, &fwd, memories, sources, default_min_gate);
    defer runs.deinit();
    try expectSpans(&.{ .{ .begin = 0, .end = 2, .start = 0 }, .{ .begin = 3, .end = 4, .start = 1 } }, runs.spans);
    try testing.expectEqual(@as(usize, 0), runs.moved);
    try expectRowMix(sources[width..][0..width], memories[3 * width ..][0..width], fwd.gate[3], runs.output[3 * width ..][0..width]);

    // Two equally confident single-slot runs wanting the same position: the
    // first run (stable order) claims it and the second moves to the other.
    const twins = try basisRows(a, width, &.{ 1, 1 });
    defer a.free(twins);
    var twin_fwd = try forward(a, &head, twins, 2, sources, 2);
    defer twin_fwd.deinit();
    var twin_runs = try contiguousRunsForward(a, &twin_fwd, twins, sources, default_min_gate);
    defer twin_runs.deinit();
    // Both slots copy, so they form one run of length 2 whose only start is
    // 0: slot 0 is moved off its own argmax (1) onto position 0.
    try expectSpans(&.{.{ .begin = 0, .end = 2, .start = 0 }}, twin_runs.spans);
    try testing.expectEqual(@as(usize, 1), twin_runs.moved);

    // A run longer than the source keeps the soft output and yields no span.
    const long = try basisRows(a, width, &.{ 0, 1, 0 });
    defer a.free(long);
    var long_fwd = try forward(a, &head, long, 3, sources, 2);
    defer long_fwd.deinit();
    var long_runs = try contiguousRunsForward(a, &long_fwd, long, sources, default_min_gate);
    defer long_runs.deinit();
    try testing.expectEqual(@as(usize, 0), long_runs.spans.len);
    try testing.expectEqual(@as(usize, 0), long_runs.moved);
    try testing.expectEqualSlices(f32, long_fwd.soft_output, long_runs.output);
}

test "memorilla copy contiguous runs process runs by descending best score" {
    const a = testing.allocator;
    const width: usize = 4;
    var head = try basisHead(a, width);
    defer head.deinit();
    const sources = try basisRows(a, width, &.{ 0, 1, 2 });
    defer a.free(sources);
    // Run A = slot 0 wants position 1 weakly (e1 + 0.9 e0); slot 1 generates;
    // run B = slot 2 wants position 1 with full confidence. B is scored
    // higher and claims 1 first; A then moves to its second choice 0.
    const memories = try a.alloc(f32, 3 * width);
    defer a.free(memories);
    @memset(memories, 0);
    memories[0 * width + 1] = 1;
    memories[0 * width + 0] = 0.9;
    memories[2 * width + 1] = 1;
    var fwd = try forward(a, &head, memories, 3, sources, 3);
    defer fwd.deinit();
    var runs = try contiguousRunsForward(a, &fwd, memories, sources, default_min_gate);
    defer runs.deinit();
    try expectSpans(&.{ .{ .begin = 0, .end = 1, .start = 0 }, .{ .begin = 2, .end = 3, .start = 1 } }, runs.spans);
    try testing.expectEqual(@as(usize, 1), runs.moved);
}

/// Direct inputs for the run rule: basis sources (`width == source_count`),
/// each copying slot's memory pointing at `targets[s]` (null = zero memory,
/// uniform log probabilities), logits `= dot / temperature`, and a soft
/// output row filled with the slot index so untouched rows are recognizable.
const RunFixture = struct {
    a: A,
    slots: usize,
    width: usize,
    logits: []f32,
    sources: []f32,
    memories: []f32,
    soft: []f32,

    fn init(a: A, width: usize, targets: []const ?usize) !RunFixture {
        const slots = targets.len;
        const basis = try a.alloc(?usize, width);
        defer a.free(basis);
        for (basis, 0..) |*b, i| b.* = i;
        const sources = try basisRows(a, width, basis);
        errdefer a.free(sources);
        const memories = try basisRows(a, width, targets);
        errdefer a.free(memories);
        const logits = try a.alloc(f32, slots * width);
        errdefer a.free(logits);
        for (0..slots) |s| {
            for (0..width) |t| logits[s * width + t] = dot(memories[s * width ..][0..width], sources[t * width ..][0..width]) / default_temperature;
        }
        const soft = try a.alloc(f32, slots * width);
        for (soft, 0..) |*v, i| v.* = @floatFromInt(i / width);
        return .{ .a = a, .slots = slots, .width = width, .logits = logits, .sources = sources, .memories = memories, .soft = soft };
    }

    fn deinit(self: *RunFixture) void {
        self.a.free(self.logits);
        self.a.free(self.sources);
        self.a.free(self.memories);
        self.a.free(self.soft);
    }

    fn runs(self: *const RunFixture, gate: []const f32, margin: f32) !Runs {
        return contiguousRunsWithMargin(self.a, self.logits, gate, self.soft, self.memories, self.sources, self.slots, self.width, self.width, default_min_gate, margin);
    }
};

test "memorilla copy bridge margin splits a run at an uncertain slot between confident slots" {
    // Gate pattern of wide case postfreeze-phrasing:ood-0008-owner (seed
    // 20261110) as the F32 path sees it: the " Z" slot (7) sits just above
    // 0.5 and glues the name run [2,7) to the site run [8,11).
    const a = testing.allocator;
    const gate = [16]f32{ 0, 0, 1, 1, 1, 1, 0.587, 0.52, 1, 1, 1, 0.026, 0, 0, 0, 0 };
    // Name slots read positions 0..4, the site slots read 12..14; the glue
    // slot points at position 8, off both diagonals.
    const targets = [16]?usize{ null, null, 0, 1, 2, 3, 4, 8, 12, 13, 14, null, null, null, null, null };
    var fx = try RunFixture.init(a, 16, &targets);
    defer fx.deinit();
    const width = fx.width;

    // Margin 0.05 (confident >= 0.55): slot 7 is the only uncertain slot
    // and it stops copying, which restores the Python spans.
    var split = try fx.runs(&gate, 0.05);
    defer split.deinit();
    try expectSpans(&.{ .{ .begin = 2, .end = 7, .start = 0 }, .{ .begin = 8, .end = 11, .start = 12 } }, split.spans);
    try testing.expectEqual(@as(usize, 0), split.moved);
    // The uncertain slot keeps the soft output exactly and claims nothing.
    try testing.expectEqualSlices(f32, fx.soft[7 * width ..][0..width], split.output[7 * width ..][0..width]);
    // Slot 6 (0.587) is confident at this margin and stays in the name run.
    try expectRowMix(fx.sources[4 * width ..][0..width], fx.memories[6 * width ..][0..width], gate[6], split.output[6 * width ..][0..width]);

    // Margin 0.1 (confident >= 0.6) also makes slot 6 uncertain; it sits
    // between confident slots 5 and 8 inside the merged run, so both glue
    // slots stop copying and the name run ends one slot earlier.
    var wider = try fx.runs(&gate, 0.1);
    defer wider.deinit();
    try expectSpans(&.{ .{ .begin = 2, .end = 6, .start = 0 }, .{ .begin = 8, .end = 11, .start = 12 } }, wider.spans);
    try testing.expectEqualSlices(f32, fx.soft[6 * width ..][0..width], wider.output[6 * width ..][0..width]);
    try testing.expectEqualSlices(f32, fx.soft[7 * width ..][0..width], wider.output[7 * width ..][0..width]);

    // Margin 0 is the plain rule: one nine-slot run.
    var merged = try fx.runs(&gate, 0);
    defer merged.deinit();
    try testing.expectEqual(@as(usize, 1), merged.spans.len);
    try testing.expectEqual(@as(usize, 2), merged.spans[0].begin);
    try testing.expectEqual(@as(usize, 11), merged.spans[0].end);
    // The old entry points are the margin-0 rule.
    var plain = try contiguousRuns(a, fx.logits, &gate, fx.soft, fx.memories, fx.sources, 16, 16, 16, default_min_gate);
    defer plain.deinit();
    try expectSpans(merged.spans, plain.spans);
    try testing.expectEqualSlices(f32, merged.output, plain.output);

    try testing.expectError(error.InvalidCopyBridgeMargin, fx.runs(&gate, -0.1));
    try testing.expectError(error.InvalidCopyBridgeMargin, fx.runs(&gate, 1.5));
}

test "memorilla copy bridge margin leaves run ends and all-uncertain runs alone" {
    const a = testing.allocator;
    // An uncertain slot at the end of a run (slot 2) and at the start of a
    // run (slot 5) is not between confident slots: both runs stay whole.
    const gate = [8]f32{ 1, 1, 0.52, 0, 0, 0.51, 1, 0 };
    const targets = [8]?usize{ 0, 1, 2, null, null, 5, 6, null };
    var fx = try RunFixture.init(a, 8, &targets);
    defer fx.deinit();
    var ends = try fx.runs(&gate, 0.1);
    defer ends.deinit();
    try expectSpans(&.{ .{ .begin = 0, .end = 3, .start = 0 }, .{ .begin = 5, .end = 7, .start = 5 } }, ends.spans);
    try expectRowMix(fx.sources[2 * fx.width ..][0..fx.width], fx.memories[2 * fx.width ..][0..fx.width], gate[2], ends.output[2 * fx.width ..][0..fx.width]);

    // A run made only of uncertain slots has no confident slot to bridge.
    const weak = [8]f32{ 0.52, 0.55, 0.51, 0, 0, 0, 0, 0 };
    var soft = try fx.runs(&weak, 0.1);
    defer soft.deinit();
    try expectSpans(&.{.{ .begin = 0, .end = 3, .start = 0 }}, soft.spans);
    var same = try fx.runs(&weak, 0);
    defer same.deinit();
    try expectSpans(same.spans, soft.spans);
    try testing.expectEqualSlices(f32, same.output, soft.output);

    // Several uncertain slots between confident ones all stop copying; a
    // slot exactly at min_gate is copying and uncertain.
    const many = [8]f32{ 1, 0.55, 0.5, 0.65, 1, 0.52, 1, 0 };
    var diagonal = try RunFixture.init(a, 8, &.{ 0, 1, 2, 3, 4, 5, 6, null });
    defer diagonal.deinit();
    var split = try diagonal.runs(&many, 0.1);
    defer split.deinit();
    try expectSpans(&.{ .{ .begin = 0, .end = 1, .start = 0 }, .{ .begin = 3, .end = 5, .start = 3 }, .{ .begin = 6, .end = 7, .start = 6 } }, split.spans);
    try testing.expectEqualSlices(f32, diagonal.soft[1 * 8 ..][0..8], split.output[1 * 8 ..][0..8]);
    try testing.expectEqualSlices(f32, diagonal.soft[2 * 8 ..][0..8], split.output[2 * 8 ..][0..8]);
    try testing.expectEqualSlices(f32, diagonal.soft[5 * 8 ..][0..8], split.output[5 * 8 ..][0..8]);
}

test "memorilla copy snapping chooses the nearest table row by cosine" {
    const a = testing.allocator;
    const width: usize = 3;
    const table = [_]f32{ 1, 0, 0, 0, 2, 0, 1, 1, 0, 0, 0, 0 };
    const output = [_]f32{ 0.1, 5, 0, 3, 3, 0.1, 0, 0, 0 };
    const ids = try cosineArgmax(a, &output, &table, 3, 4, width);
    defer a.free(ids);
    // Row 0: nearest is the scaled e1 row (1); row 1: the diagonal row (2);
    // row 2: a zero output row ties at cosine 0 everywhere, first wins.
    try testing.expectEqualSlices(i32, &.{ 1, 2, 0 }, ids);

    const norms = try rowNorms(a, &table, 4, width);
    defer a.free(norms);
    try testing.expectApproxEqAbs(@as(f32, 2), norms[1], 1e-6);
    try testing.expectApproxEqAbs(@as(f32, std.math.sqrt2), norms[2], 1e-6);
    try testing.expectEqual(@as(f32, 0), norms[3]);
    const output_norms = try rowNorms(a, &output, 3, width);
    defer a.free(output_norms);
    // Raw dot products as a device matmul would deliver them.
    var scores: [12]f32 = undefined;
    for (0..3) |s| {
        for (0..4) |v| scores[s * 4 + v] = dot(output[s * width ..][0..width], table[v * width ..][0..width]);
    }
    const snapped = try snapIds(a, &scores, norms, output_norms, 3, 4);
    defer a.free(snapped);
    try testing.expectEqualSlices(i32, ids, snapped);
    try testing.expectError(error.InvalidCopyGeometry, snapIds(a, &scores, norms, output_norms, 2, 4));
}

test "memorilla copy token continues word mirrors the Python predicate" {
    const a = testing.allocator;
    var words = WordTokenizer{ .words = &python_words };
    const tokenizer = words.handle();
    try testing.expect(try tokenContinuesWord(a, tokenizer, 4)); // "ima"
    try testing.expect(try tokenContinuesWord(a, tokenizer, 16)); // "1844"
    try testing.expect(try tokenContinuesWord(a, tokenizer, 15)); // "-00"
    try testing.expect(!try tokenContinuesWord(a, tokenizer, 3)); // " Kit"
    try testing.expect(!try tokenContinuesWord(a, tokenizer, 11)); // "."
    try testing.expect(!try tokenContinuesWord(a, tokenizer, 0)); // " "
    try testing.expect(try tokenContinuesWord(a, tokenizer, 17)); // "é" (Latin-1 letter)
    try testing.expect(!try tokenContinuesWord(a, tokenizer, 18)); // "日本": outside the approximation
    try testing.expect(!try tokenContinuesWord(a, tokenizer, 19)); // special token decodes to ""
    try testing.expect(!try tokenContinuesWord(a, tokenizer, 20)); // " -00" has a leading space

    try testing.expect(try tokenStartsWord(a, tokenizer, 3)); // " Kit"
    try testing.expect(try tokenStartsWord(a, tokenizer, 13)); // " N"
    try testing.expect(!try tokenStartsWord(a, tokenizer, 4)); // "ima"
    try testing.expect(!try tokenStartsWord(a, tokenizer, 20)); // " -00": hyphen is not alnum
    try testing.expect(!try tokenStartsWord(a, tokenizer, 0)); // " "
    try testing.expect(!try tokenStartsWord(a, tokenizer, 11)); // "."
    try testing.expect(isWordCodepoint('Z') and isWordCodepoint('7') and !isWordCodepoint('_') and isWordCodepoint(0xE9) and !isWordCodepoint(0xD7));
}

test "memorilla copy completed words extend copied spans to word boundaries" {
    // Port of test_completed_words_extend_copied_spans_to_word_boundaries
    // (overflow="truncate" cases).
    const a = testing.allocator;
    var words = WordTokenizer{ .words = &python_words };
    const tokenizer = words.handle();
    // Source: " Kit ima  Dab oco  at  Mo zeba  Depot ."
    const source = [_]i32{ 3, 4, 5, 6, 7, 8, 9, 10, 11 };
    // Record (10 slots): key holder [Kit ima Dab] at [zeba Depot] pad pad pad
    const record = [_]i32{ 1, 2, 3, 4, 5, 7, 9, 10, 0, 0 };
    const spans = [_]Span{ .{ .begin = 2, .end = 5, .start = 0 }, .{ .begin = 6, .end = 8, .start = 6 } };
    var full = try completeCopiedWords(a, tokenizer, &record, &spans, &source, 0);
    defer full.deinit();
    try testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, full.ids);
    try testing.expectEqual(@as(usize, 2), full.inserted);

    // Overflow drops trailing pads first, then the final tokens.
    var short = try completeCopiedWords(a, tokenizer, record[0..9], &spans, &source, 0);
    defer short.deinit();
    try testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4, 5, 6, 7, 8, 9 }, short.ids);

    // Spans arrive unsorted and are walked in record order anyway.
    const reversed = [_]Span{ spans[1], spans[0] };
    var sorted = try completeCopiedWords(a, tokenizer, &record, &reversed, &source, 0);
    defer sorted.deinit();
    try testing.expectEqualSlices(i32, full.ids, sorted.ids);

    // A span already on word boundaries is unchanged.
    var same = try completeCopiedWords(a, tokenizer, &.{ 1, 2, 3, 4, 0 }, &.{.{ .begin = 2, .end = 4, .start = 0 }}, &source, 0);
    defer same.deinit();
    try testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4, 0 }, same.ids);
    try testing.expectEqual(@as(usize, 0), same.inserted);

    // A pad in the middle survives; only trailing pads are dropped on overflow.
    // Source " Kit ima oco": the span [Kit ima] grows by "oco".
    const glued = [_]i32{ 3, 4, 6, 7 };
    var middle = try completeCopiedWords(a, tokenizer, &.{ 1, 0, 3, 4, 0 }, &.{.{ .begin = 2, .end = 4, .start = 0 }}, &glued, 0);
    defer middle.deinit();
    try testing.expectEqualSlices(i32, &.{ 1, 0, 3, 4, 6 }, middle.ids);
    try testing.expectEqual(@as(usize, 1), middle.inserted);

    // Backward extension: the span starts mid-word (" Mo" "zeba" read from "zeba").
    var back = try completeCopiedWords(a, tokenizer, &.{ 1, 9, 10, 0, 0 }, &.{.{ .begin = 1, .end = 3, .start = 6 }}, &source, 0);
    defer back.deinit();
    try testing.expectEqualSlices(i32, &.{ 1, 8, 9, 10, 0 }, back.ids);
    try testing.expectEqual(@as(usize, 1), back.inserted);

    // A full record with nothing to extend is returned as is.
    const full_record = [_]i32{ 12, 13, 14, 15, 16, 7, 8, 9 };
    const full_source = [_]i32{ 12, 13, 14, 15, 16, 7, 8, 9, 10, 11 };
    var kept = try completeCopiedWords(a, tokenizer, &full_record, &.{ .{ .begin = 1, .end = 5, .start = 1 }, .{ .begin = 6, .end = 8, .start = 6 } }, &full_source, 0);
    defer kept.deinit();
    try testing.expectEqualSlices(i32, &full_record, kept.ids);

    // No spans: the snapped ids pass through.
    var plain = try completeCopiedWords(a, tokenizer, &record, &.{}, &source, 0);
    defer plain.deinit();
    try testing.expectEqualSlices(i32, &record, plain.ids);

    // Span validation.
    try testing.expectError(error.InvalidCopySpan, completeCopiedWords(a, tokenizer, &record, &.{.{ .begin = 2, .end = 5, .start = 8 }}, &source, 0));
    try testing.expectError(error.InvalidCopySpan, completeCopiedWords(a, tokenizer, &record, &.{.{ .begin = 2, .end = 2, .start = 0 }}, &source, 0));
    try testing.expectError(error.InvalidCopySpan, completeCopiedWords(a, tokenizer, &record, &.{.{ .begin = 9, .end = 11, .start = 0 }}, &source, 0));
    try testing.expectError(error.InvalidCopySpan, completeCopiedWords(a, tokenizer, &record, &.{.{ .begin = 0, .end = 10, .start = 0 }}, &source, 0));
}

test "memorilla copy decode chains runs snapping and word completion" {
    const a = testing.allocator;
    const width: usize = 4;
    // Table rows double as token ids: 0 " Kit", 1 "ima", 2 " Dab", 3 " at",
    // 4 " " (the pad).
    const table = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0.5, 0.5, 0.5, 0.5 };
    const words = [_][]const u8{ " Kit", "ima", " Dab", " at", " " };
    var word_tokenizer = WordTokenizer{ .words = &words };
    const tokenizer = word_tokenizer.handle();
    // The document " Kit ima Dab at" = source rows 0..3.
    const sources = table[0 .. 4 * width];
    const source_ids = [_]i32{ 0, 1, 2, 3 };
    // Gate high for the e0 memory, low for the pad memory.
    const query = [_]f32{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
    var head = try CopyHead.init(a, width, &query, &.{ 12, -12, 0, 0 }, -3, default_temperature);
    defer head.deinit();
    // Slot 0 copies " Kit"; slots 1 and 2 are pads.
    const memories = [_]f32{ 1, 0, 0, 0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5 };
    var fwd = try forward(a, &head, &memories, 3, sources, 4);
    defer fwd.deinit();
    try testing.expect(fwd.gate[0] > 0.99 and fwd.gate[1] < 0.1);

    var table_snapper = TableSnapper{ .table = &table, .vocab = 5 };
    var decoded = try decode(a, tokenizer, &fwd, &memories, sources, &source_ids, table_snapper.snapper(), default_min_gate, 4);
    defer decoded.deinit();
    try expectSpans(&.{.{ .begin = 0, .end = 1, .start = 0 }}, decoded.runs.spans);
    try testing.expectEqualSlices(i32, &.{ 0, 4, 4 }, decoded.snapped_ids);
    // Word completion pulls in "ima" and the overflow drops a trailing pad.
    try testing.expectEqualSlices(i32, &.{ 0, 1, 4 }, decoded.completed_ids);
    try testing.expectEqual(@as(usize, 1), decoded.inserted);

    // Without any copying slot the snapped ids pass through unchanged.
    var head_closed = try CopyHead.init(a, width, &query, &.{ 0, 0, 0, 0 }, -6, default_temperature);
    defer head_closed.deinit();
    var closed = try forward(a, &head_closed, &memories, 3, sources, 4);
    defer closed.deinit();
    var passthrough = try decode(a, tokenizer, &closed, &memories, sources, &source_ids, table_snapper.snapper(), default_min_gate, 4);
    defer passthrough.deinit();
    try testing.expectEqual(@as(usize, 0), passthrough.runs.spans.len);
    try testing.expectEqualSlices(i32, passthrough.snapped_ids, passthrough.completed_ids);
    try testing.expectEqual(@as(usize, 0), passthrough.inserted);

    try testing.expectError(error.InvalidCopyGeometry, decode(a, tokenizer, &fwd, &memories, sources, source_ids[0..3], table_snapper.snapper(), default_min_gate, 4));
}
