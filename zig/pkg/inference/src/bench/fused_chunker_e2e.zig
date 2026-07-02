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

// Fused chunker-embedder serving benchmark.
//
// Loads a packaged fused-chunker export directory (the same layout POST
// /chunk serves via src/pipelines/fused_chunking.zig), synthesizes documents
// at configurable token lengths, and measures the serving path end to end:
// cold model load, then per-iteration tokenize / forward / decode / pool /
// SPLADE phase timings plus e2e latency percentiles, in boundary-only and
// full-embedding modes. Reports peak RSS and optionally writes a JSON
// results file.
//
// Example:
//   zig build -Dskip-openapi=true bench-fused-chunker-e2e -- \
//     --model-dir /private/tmp/fused_export_smoke \
//     --iterations 3 --doc-tokens 512,2048 --backend native \
//     --results-out /tmp/fused_chunker_bench.json

const std = @import("std");
const builtin = @import("builtin");

const inference = @import("inference_internal");
const fused_chunking = inference.pipelines.fused_chunking;
const lib_chunker = inference.chunker;

const default_model_dir = "/private/tmp/fused_export_smoke";
const results_schema = "fused_chunker_e2e_bench/v1";
/// Hard cap on synthesized document bytes (~8 bytes/token headroom over the
/// serving token cap) so a degenerate tokenizer cannot grow text unboundedly.
const max_document_bytes: usize = fused_chunking.max_serving_tokens * 16;

const BackendChoice = enum { native, metal };

const BenchMode = enum { boundary, full, both };

const Options = struct {
    model_dir: []const u8 = default_model_dir,
    iterations: usize = 10,
    warmup_iters: usize = 1,
    doc_tokens: std.ArrayListUnmanaged(usize) = .empty,
    backend: BackendChoice = .native,
    mode: BenchMode = .both,
    results_out: ?[]const u8 = null,

    fn deinit(self: *Options, allocator: std.mem.Allocator) void {
        self.doc_tokens.deinit(allocator);
    }
};

const Row = struct {
    doc_tokens: usize,
    actual_tokens: u64,
    mode: []const u8,
    windows: u64,
    chunks: u64,
    iterations: usize,
    e2e_avg_ms: f64,
    e2e_p50_ms: f64,
    e2e_p95_ms: f64,
    e2e_min_ms: f64,
    e2e_max_ms: f64,
    tokenize_avg_ms: f64,
    forward_avg_ms: f64,
    decode_avg_ms: f64,
    pool_avg_ms: f64,
    splade_avg_ms: f64,
};

const ResultsJson = struct {
    schema_version: []const u8 = results_schema,
    model_dir: []const u8,
    backend: []const u8,
    model_version: []const u8,
    max_seq_len: usize,
    has_splade: bool,
    iterations: usize,
    warmup_iterations: usize,
    cold_load_ms: f64,
    peak_rss_bytes: u64,
    rows: []const Row,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    var opts = try parseArgs(allocator, init) orelse return;
    defer opts.deinit(allocator);
    if (opts.doc_tokens.items.len == 0) {
        try opts.doc_tokens.appendSlice(allocator, &.{ 512, 2048, 8192 });
    }
    if (opts.iterations == 0) return error.InvalidIterations;

    const load_backend: fused_chunking.Backend = switch (opts.backend) {
        .native => .native,
        .metal => .metal,
    };

    const load_start = nowNs();
    const pipeline = fused_chunking.FusedChunkerPipeline.loadFromDir(
        allocator,
        opts.model_dir,
        .{ .backend = load_backend },
    ) catch |err| {
        std.debug.print("failed to load fused chunker model dir {s}: {s}\n", .{ opts.model_dir, @errorName(err) });
        return err;
    };
    defer pipeline.deinit();
    const cold_load_ms = nsToMs(nowNs() - load_start);

    const emb_dim: usize = @intCast(pipeline.config.embedding_dim);
    const hidden: usize = @intCast(pipeline.config.hidden_size);
    const sparse_supported = pipeline.has_splade and emb_dim == hidden;

    var rows = std.ArrayListUnmanaged(Row).empty;
    defer rows.deinit(allocator);

    for (opts.doc_tokens.items, 0..) |target_tokens, doc_idx| {
        const text = try synthesizeDocument(allocator, pipeline, target_tokens, 0x5eed + doc_idx);
        defer allocator.free(text);

        if (opts.mode == .boundary or opts.mode == .both) {
            const row = try runMode(allocator, pipeline, text, target_tokens, .{
                .include_embeddings = false,
                .include_sparse = false,
                .max_chunks = 0,
            }, "boundary", opts.warmup_iters, opts.iterations);
            try rows.append(allocator, row);
        }
        if (opts.mode == .full or opts.mode == .both) {
            const row = try runMode(allocator, pipeline, text, target_tokens, .{
                .include_embeddings = true,
                .include_sparse = sparse_supported,
                .max_chunks = 0,
            }, "full", opts.warmup_iters, opts.iterations);
            try rows.append(allocator, row);
        }
    }

    const peak_rss = peakRssBytes();

    std.debug.print(
        "fused-chunker-e2e: model_dir={s} backend={s} model_version={s} max_seq_len={d} has_splade={} sparse_in_full_mode={}\n",
        .{ opts.model_dir, @tagName(opts.backend), pipeline.model_version, pipeline.max_seq_len, pipeline.has_splade, sparse_supported },
    );
    std.debug.print(
        "cold_load_ms={d:.1} peak_rss_mb={d:.1} iterations={d} warmup={d}\n\n",
        .{ cold_load_ms, @as(f64, @floatFromInt(peak_rss)) / (1024.0 * 1024.0), opts.iterations, opts.warmup_iters },
    );
    std.debug.print(
        "{s:>10} {s:>7} {s:>9} {s:>7} {s:>6} {s:>9} {s:>9} {s:>9} {s:>9} {s:>9} {s:>11} {s:>10} {s:>9} {s:>8} {s:>9}\n",
        .{ "doc_tokens", "actual", "mode", "windows", "chunks", "p50_ms", "p95_ms", "avg_ms", "min_ms", "max_ms", "tokenize_ms", "forward_ms", "decode_ms", "pool_ms", "splade_ms" },
    );
    for (rows.items) |row| {
        std.debug.print(
            "{d:>10} {d:>7} {s:>9} {d:>7} {d:>6} {d:>9.2} {d:>9.2} {d:>9.2} {d:>9.2} {d:>9.2} {d:>11.2} {d:>10.2} {d:>9.3} {d:>8.3} {d:>9.3}\n",
            .{
                row.doc_tokens,
                row.actual_tokens,
                row.mode,
                row.windows,
                row.chunks,
                row.e2e_p50_ms,
                row.e2e_p95_ms,
                row.e2e_avg_ms,
                row.e2e_min_ms,
                row.e2e_max_ms,
                row.tokenize_avg_ms,
                row.forward_avg_ms,
                row.decode_avg_ms,
                row.pool_avg_ms,
                row.splade_avg_ms,
            },
        );
    }

    if (opts.results_out) |out_path| {
        const results = ResultsJson{
            .model_dir = opts.model_dir,
            .backend = @tagName(opts.backend),
            .model_version = pipeline.model_version,
            .max_seq_len = pipeline.max_seq_len,
            .has_splade = pipeline.has_splade,
            .iterations = opts.iterations,
            .warmup_iterations = opts.warmup_iters,
            .cold_load_ms = cold_load_ms,
            .peak_rss_bytes = @intCast(peak_rss),
            .rows = rows.items,
        };
        var buffer: std.Io.Writer.Allocating = .init(allocator);
        defer buffer.deinit();
        try std.json.Stringify.value(results, .{ .whitespace = .indent_2 }, &buffer.writer);
        try buffer.writer.writeByte('\n');
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = out_path, .data = buffer.writer.buffered() });
        std.debug.print("\nresults written to {s}\n", .{out_path});
    }
}

fn runMode(
    allocator: std.mem.Allocator,
    pipeline: *fused_chunking.FusedChunkerPipeline,
    text: []const u8,
    target_tokens: usize,
    chunk_opts: fused_chunking.ChunkRequestOptions,
    mode_name: []const u8,
    warmup_iters: usize,
    iterations: usize,
) !Row {
    for (0..warmup_iters) |_| {
        var timings = fused_chunking.PhaseTimings{};
        const chunks = try pipeline.chunkTextTimed(allocator, text, chunk_opts, &timings);
        lib_chunker.types.freeChunks(allocator, chunks);
    }

    const samples = try allocator.alloc(u64, iterations);
    defer allocator.free(samples);
    var tokenize_ns: u128 = 0;
    var forward_ns: u128 = 0;
    var decode_ns: u128 = 0;
    var pool_ns: u128 = 0;
    var splade_ns: u128 = 0;
    var last = fused_chunking.PhaseTimings{};

    for (samples) |*sample| {
        var timings = fused_chunking.PhaseTimings{};
        const start = nowNs();
        const chunks = try pipeline.chunkTextTimed(allocator, text, chunk_opts, &timings);
        sample.* = nowNs() - start;
        lib_chunker.types.freeChunks(allocator, chunks);
        tokenize_ns += timings.tokenize_ns;
        forward_ns += timings.forward_ns;
        decode_ns += timings.decode_ns;
        pool_ns += timings.pool_ns;
        splade_ns += timings.splade_ns;
        last = timings;
    }

    const sorted = try allocator.dupe(u64, samples);
    defer allocator.free(sorted);
    std.mem.sort(u64, sorted, {}, std.sort.asc(u64));
    var total_ns: u128 = 0;
    for (samples) |s| total_ns += s;
    const p50_idx = sorted.len / 2;
    const p95_idx = @min(sorted.len - 1, (sorted.len * 95 + 99) / 100 - 1);
    const iters_f: f64 = @floatFromInt(iterations);

    return .{
        .doc_tokens = target_tokens,
        .actual_tokens = last.tokens,
        .mode = mode_name,
        .windows = last.windows,
        .chunks = last.chunks,
        .iterations = iterations,
        .e2e_avg_ms = nsToMs(@intCast(total_ns / iterations)),
        .e2e_p50_ms = nsToMs(sorted[p50_idx]),
        .e2e_p95_ms = nsToMs(sorted[p95_idx]),
        .e2e_min_ms = nsToMs(sorted[0]),
        .e2e_max_ms = nsToMs(sorted[sorted.len - 1]),
        .tokenize_avg_ms = @as(f64, @floatFromInt(tokenize_ns)) / iters_f / 1.0e6,
        .forward_avg_ms = @as(f64, @floatFromInt(forward_ns)) / iters_f / 1.0e6,
        .decode_avg_ms = @as(f64, @floatFromInt(decode_ns)) / iters_f / 1.0e6,
        .pool_avg_ms = @as(f64, @floatFromInt(pool_ns)) / iters_f / 1.0e6,
        .splade_avg_ms = @as(f64, @floatFromInt(splade_ns)) / iters_f / 1.0e6,
    };
}

const word_bank = [_][]const u8{
    "search",   "vector",   "storage",  "cluster",   "replica",  "index",
    "shard",    "query",    "boundary", "document",  "token",    "window",
    "latency",  "backend",  "compute",  "pipeline",  "manifest", "encoder",
    "pooling",  "database", "network",  "consensus", "snapshot", "metadata",
    "chunking", "sequence", "sparse",   "dense",     "matrix",   "kernel",
};

/// Build a plain-text document whose tokenization (with the pipeline's own
/// tokenizer) yields approximately `target_tokens` valid tokens. The text is
/// sliced at the char offset of the target-th real token so long targets are
/// hit precisely rather than estimated.
fn synthesizeDocument(
    allocator: std.mem.Allocator,
    pipeline: *fused_chunking.FusedChunkerPipeline,
    target_tokens: usize,
    seed: usize,
) ![]u8 {
    if (target_tokens < 8) return error.TargetTokensTooSmall;
    if (target_tokens > fused_chunking.max_serving_tokens) return error.TargetTokensTooLarge;
    // Leave room for [CLS]/[SEP] so the padded valid length lands near target.
    const target_real_tokens = target_tokens - 2;

    var prng = std.Random.DefaultPrng.init(@intCast(seed));
    const random = prng.random();
    const tok = pipeline.tok.tokenizer();

    var text = std.ArrayListUnmanaged(u8).empty;
    defer text.deinit(allocator);

    // Roughly 7 chars/token for this word bank; grow until the tokenizer
    // confirms we crossed the target.
    var goal_bytes: usize = @min(target_tokens * 8, max_document_bytes);
    while (true) {
        while (text.items.len < goal_bytes) {
            try appendSentence(allocator, &text, random);
        }

        var enc = try tok.encodeForModel(allocator, text.items, fused_chunking.max_serving_tokens);
        defer enc.deinit();
        const offsets = enc.offsets orelse return error.TokenizerOffsetsUnavailable;

        var real_tokens: usize = 0;
        const n = @min(offsets.len, enc.attention_mask.len);
        for (offsets[0..n], enc.attention_mask[0..n]) |off, m| {
            if (m == 0) break;
            if (off[1] <= off[0]) continue;
            real_tokens += 1;
            if (real_tokens == target_real_tokens) {
                return allocator.dupe(u8, text.items[0..off[1]]);
            }
        }

        if (text.items.len >= max_document_bytes) return error.CannotReachTargetTokens;
        goal_bytes = @min(goal_bytes * 2, max_document_bytes);
    }
}

fn appendSentence(
    allocator: std.mem.Allocator,
    text: *std.ArrayListUnmanaged(u8),
    random: std.Random,
) !void {
    const words = 8 + random.uintLessThan(usize, 7);
    for (0..words) |i| {
        const word = word_bank[random.uintLessThan(usize, word_bank.len)];
        if (i == 0) {
            try text.append(allocator, std.ascii.toUpper(word[0]));
            try text.appendSlice(allocator, word[1..]);
        } else {
            try text.append(allocator, ' ');
            try text.appendSlice(allocator, word);
        }
    }
    // Occasional paragraph breaks give the boundary head something to find.
    if (random.uintLessThan(usize, 4) == 0) {
        try text.appendSlice(allocator, ".\n\n");
    } else {
        try text.appendSlice(allocator, ". ");
    }
}

fn parseArgs(allocator: std.mem.Allocator, init: std.process.Init) !?Options {
    var opts = Options{};
    errdefer opts.deinit(allocator);
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.next();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model-dir")) {
            opts.model_dir = args.next() orelse return error.MissingModelDir;
        } else if (std.mem.eql(u8, arg, "--iterations")) {
            opts.iterations = try std.fmt.parseInt(usize, args.next() orelse return error.MissingIterations, 10);
        } else if (std.mem.eql(u8, arg, "--warmup-iters")) {
            opts.warmup_iters = try std.fmt.parseInt(usize, args.next() orelse return error.MissingWarmupIters, 10);
        } else if (std.mem.eql(u8, arg, "--doc-tokens")) {
            const list = args.next() orelse return error.MissingDocTokens;
            var it = std.mem.splitScalar(u8, list, ',');
            while (it.next()) |part| {
                const trimmed = std.mem.trim(u8, part, " \t");
                if (trimmed.len == 0) continue;
                try opts.doc_tokens.append(allocator, try std.fmt.parseInt(usize, trimmed, 10));
            }
        } else if (std.mem.eql(u8, arg, "--backend")) {
            opts.backend = parseBackend(args.next() orelse return error.MissingBackend) orelse return error.InvalidBackend;
        } else if (std.mem.eql(u8, arg, "--mode")) {
            opts.mode = parseMode(args.next() orelse return error.MissingMode) orelse return error.InvalidMode;
        } else if (std.mem.eql(u8, arg, "--results-out")) {
            opts.results_out = args.next() orelse return error.MissingResultsOut;
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            opts.deinit(allocator);
            return null;
        } else {
            printUsage();
            return error.InvalidArguments;
        }
    }
    return opts;
}

fn parseBackend(value: []const u8) ?BackendChoice {
    if (std.ascii.eqlIgnoreCase(value, "native")) return .native;
    if (std.ascii.eqlIgnoreCase(value, "metal")) return .metal;
    return null;
}

fn parseMode(value: []const u8) ?BenchMode {
    if (std.ascii.eqlIgnoreCase(value, "boundary")) return .boundary;
    if (std.ascii.eqlIgnoreCase(value, "full")) return .full;
    if (std.ascii.eqlIgnoreCase(value, "both")) return .both;
    return null;
}

fn printUsage() void {
    std.debug.print(
        \\usage: zig build bench-fused-chunker-e2e -- [--model-dir <dir>] [--iterations N]
        \\         [--warmup-iters N] [--doc-tokens 512,2048,8192] [--backend native|metal]
        \\         [--mode boundary|full|both] [--results-out <path>]
        \\
        \\Benchmarks the fused chunker /chunk serving path against a packaged
        \\export directory (default {s}): cold load, per-phase
        \\timings (tokenize/forward/decode/pool/splade), e2e latency
        \\percentiles, and peak RSS.
        \\
    , .{default_model_dir});
}

fn peakRssBytes() usize {
    const usage = std.posix.getrusage(std.posix.rusage.SELF);
    if (usage.maxrss <= 0) return 0;
    const maxrss: usize = @intCast(usage.maxrss);
    return switch (builtin.os.tag) {
        // Darwin reports ru_maxrss in bytes; Linux reports KiB.
        .driverkit, .ios, .maccatalyst, .macos, .tvos, .visionos, .watchos => maxrss,
        .linux => std.math.mul(usize, maxrss, 1024) catch std.math.maxInt(usize),
        else => maxrss,
    };
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1.0e6;
}

fn nowNs() u64 {
    var ts: std.posix.timespec = undefined;
    switch (std.posix.errno(std.posix.system.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts))) {
        .SUCCESS => return @intCast(@as(i128, ts.sec) * std.time.ns_per_s + ts.nsec),
        else => return 0,
    }
}
