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

// BGE-M3 pretokenized CUDA encoder benchmark.
//
// This deliberately measures the comparable encoder contract: pretokenized
// [batch, 256] IDs -> normalized dense CLS vector. Tokenization, model load,
// and response serialization are excluded. CUDA RuntimeStats make the
// device-resident boundary explicit: one final vector download per request,
// rather than a [batch, sequence, hidden] activation download.

const std = @import("std");
const build_options = @import("build_options");
const inference = @import("inference_internal");
const backends = inference.backends;
const kernel_jit = inference.graph.kernel_jit;
const model_manager_mod = inference.server.model_manager;
const session_factory = inference.architectures.session_factory;

extern fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

const BackendChoice = enum { native, cuda };
const AttentionKernel = enum { generic, exact, exact_q8, tensor_core };

const Options = struct {
    model_dir: []const u8 = "",
    backend: BackendChoice = .cuda,
    batch: usize = 1,
    seq_len: usize = 256,
    warmup_iters: usize = 2,
    measure_iters: usize = 10,
    attention_kernel: AttentionKernel = .exact,
    validate_specialized_attention: bool = false,
    min_attention_cosine: ?f64 = null,
    max_attention_abs: ?f32 = null,
    kernel_jit: kernel_jit.Config = .{},
};

const Timing = struct {
    avg_ns: u64,
    p50_ns: u64,
    p95_ns: u64,
    min_ns: u64,
    max_ns: u64,
};

const CudaDelta = struct {
    h2d_bytes: usize = 0,
    d2h_bytes: usize = 0,
    to_float32_calls: usize = 0,
    to_float32_bytes: usize = 0,
    bf16_cublaslt_linear_calls: usize = 0,
    bf16_cublaslt_qkv_calls: usize = 0,
    bf16_activation_staging_calls: usize = 0,
    bf16_activation_mirror_hits: usize = 0,
    f16_cublaslt_linear_calls: usize = 0,
    f16_cublaslt_qkv_calls: usize = 0,
    f16_activation_staging_calls: usize = 0,
    f16_cublaslt_fallbacks: usize = 0,
    generated_mm_hits: usize = 0,
    linear_launches: usize = 0,
    attention_launches: usize = 0,
    prefill_profile_events: usize = 0,
    prefill_bf16_linear_us: u64 = 0,
    prefill_bf16_qkv_us: u64 = 0,
    prefill_attention_us: u64 = 0,
    prefill_staging_us: u64 = 0,
    prefill_norm_us: u64 = 0,
};

const AttentionValidation = struct {
    max_abs: f32,
    cosine: f64,
};

pub fn main(init: std.process.Init) !void {
    const allocator = std.heap.page_allocator;
    const opts = try parseArgs(init);
    if (opts.model_dir.len == 0 or opts.batch == 0 or opts.seq_len == 0 or opts.measure_iters == 0) {
        printUsage();
        return error.InvalidArguments;
    }
    if (opts.backend == .cuda and !build_options.enable_cuda) return error.CudaNotEnabled;
    if (opts.validate_specialized_attention and opts.attention_kernel == .generic) return error.InvalidAttentionValidationKernel;

    var session_manager = backends.SessionManager.initWithIo(allocator, init.io);
    session_manager.preferred_backends = switch (opts.backend) {
        .native => &.{backends.BackendType.native},
        .cuda => &.{backends.BackendType.cuda},
    };
    session_manager.kernel_jit = opts.kernel_jit;

    var model_manager = model_manager_mod.ModelManager.init(allocator, session_manager);
    defer model_manager.deinit();
    const model = model_manager.loadFromDir(opts.model_dir) catch |err| {
        std.debug.print("bge_m3_e2e: model_load_error={s}\n", .{@errorName(err)});
        return err;
    };
    try model.ensureEmbeddingAssets(true, false, false);
    const expected_backend: backends.BackendType = switch (opts.backend) {
        .native => .native,
        .cuda => .cuda,
    };
    if (model.session.backend() != expected_backend) return error.UnexpectedBackend;

    var pipeline = model.embeddingPipeline(allocator);
    pipeline.config.resident_projection_required = opts.backend == .cuda;
    if (opts.backend == .cuda and !pipeline.config.resident_text_encoder) {
        return error.ResidentTextEncoderUnavailable;
    }
    if (opts.backend == .cuda) try setAttentionKernel(opts.attention_kernel);

    const token_count = std.math.mul(usize, opts.batch, opts.seq_len) catch return error.InvalidInputShape;
    const input_ids = try allocator.alloc(i64, token_count);
    defer allocator.free(input_ids);
    const attention_mask = try allocator.alloc(i64, token_count);
    defer allocator.free(attention_mask);
    for (input_ids, 0..) |*token, index| {
        // Valid non-special BGE-M3 vocabulary IDs. Repeating a deterministic
        // range makes this benchmark portable without requiring a tokenizer.
        token.* = @intCast(4 + (index % 1024));
    }
    @memset(attention_mask, 1);

    for (0..opts.warmup_iters) |_| {
        const embeddings = try pipeline.embedTokenized(input_ids, attention_mask, opts.batch, opts.seq_len);
        freeEmbeddings(allocator, embeddings);
    }

    const attention_validation: ?AttentionValidation = if (opts.validate_specialized_attention and opts.backend == .cuda and opts.seq_len == 256) blk: {
        try setAttentionKernelEnabled(false);
        const generic = try pipeline.embedTokenized(input_ids, attention_mask, opts.batch, opts.seq_len);
        defer freeEmbeddings(allocator, generic);
        try setAttentionKernel(opts.attention_kernel);
        const specialized = try pipeline.embedTokenized(input_ids, attention_mask, opts.batch, opts.seq_len);
        defer freeEmbeddings(allocator, specialized);
        break :blk compareEmbeddings(generic, specialized);
    } else null;

    if (opts.min_attention_cosine != null or opts.max_attention_abs != null) {
        const validation = attention_validation orelse return error.AttentionValidationRequired;
        if (opts.min_attention_cosine) |minimum| {
            if (validation.cosine < minimum) {
                std.debug.print(
                    "bge_m3_e2e attention_quality_gate=failed metric=cosine actual={d:.8} minimum={d:.8}\n",
                    .{ validation.cosine, minimum },
                );
                return error.AttentionCosineBelowThreshold;
            }
        }
        if (opts.max_attention_abs) |maximum| {
            if (validation.max_abs > maximum) {
                std.debug.print(
                    "bge_m3_e2e attention_quality_gate=failed metric=max_abs actual={d:.7} maximum={d:.7}\n",
                    .{ validation.max_abs, maximum },
                );
                return error.AttentionMaxAbsAboveThreshold;
            }
        }
    }

    const before_cuda = session_factory.getCudaRuntimeStats(model.session);
    const before_resident = model.resident_projection_stats.snapshot();
    const samples = try allocator.alloc(u64, opts.measure_iters);
    defer allocator.free(samples);
    var checksum: f64 = 0;
    for (samples) |*sample| {
        const start = nowNs();
        const embeddings = try pipeline.embedTokenized(input_ids, attention_mask, opts.batch, opts.seq_len);
        sample.* = nowNs() - start;
        checksum += embeddingChecksum(embeddings);
        freeEmbeddings(allocator, embeddings);
    }
    const after_cuda = session_factory.getCudaRuntimeStats(model.session);
    const resident = model.resident_projection_stats.snapshot();
    const timing = try summarize(allocator, samples);

    const cuda: CudaDelta = if (before_cuda) |before| blk: {
        const after = after_cuda orelse return error.CudaStatsUnavailable;
        break :blk .{
            .h2d_bytes = after.h2d_bytes - before.h2d_bytes,
            .d2h_bytes = after.d2h_bytes - before.d2h_bytes,
            .to_float32_calls = after.to_float32_calls - before.to_float32_calls,
            .to_float32_bytes = after.to_float32_bytes - before.to_float32_bytes,
            .bf16_cublaslt_linear_calls = after.bf16_cublaslt_linear_calls - before.bf16_cublaslt_linear_calls,
            .bf16_cublaslt_qkv_calls = after.bf16_cublaslt_qkv_calls - before.bf16_cublaslt_qkv_calls,
            .bf16_activation_staging_calls = after.bf16_cublaslt_activation_staging_calls - before.bf16_cublaslt_activation_staging_calls,
            .bf16_activation_mirror_hits = after.bf16_cublaslt_activation_mirror_hits - before.bf16_cublaslt_activation_mirror_hits,
            .f16_cublaslt_linear_calls = after.f16_cublaslt_linear_calls - before.f16_cublaslt_linear_calls,
            .f16_cublaslt_qkv_calls = after.f16_cublaslt_qkv_calls - before.f16_cublaslt_qkv_calls,
            .f16_activation_staging_calls = after.f16_cublaslt_activation_staging_calls - before.f16_cublaslt_activation_staging_calls,
            .f16_cublaslt_fallbacks = after.f16_cublaslt_fallbacks - before.f16_cublaslt_fallbacks,
            .generated_mm_hits = after.q4_0_generated_mm_hits - before.q4_0_generated_mm_hits,
            .linear_launches = after.launch_linear - before.launch_linear,
            .attention_launches = after.launch_attention - before.launch_attention,
            .prefill_profile_events = after.prefill_profile_events - before.prefill_profile_events,
            .prefill_bf16_linear_us = after.prefill_profile_bf16_linear_us - before.prefill_profile_bf16_linear_us,
            .prefill_bf16_qkv_us = after.prefill_profile_bf16_qkv_us - before.prefill_profile_bf16_qkv_us,
            .prefill_attention_us = after.prefill_profile_attention_us - before.prefill_profile_attention_us,
            .prefill_staging_us = after.prefill_profile_staging_us - before.prefill_profile_staging_us,
            .prefill_norm_us = after.prefill_profile_norm_us - before.prefill_profile_norm_us,
        };
    } else .{
        .h2d_bytes = @as(usize, 0),
        .d2h_bytes = @as(usize, 0),
        .to_float32_calls = @as(usize, 0),
        .to_float32_bytes = @as(usize, 0),
        .bf16_cublaslt_linear_calls = @as(usize, 0),
        .bf16_cublaslt_qkv_calls = @as(usize, 0),
        .bf16_activation_staging_calls = @as(usize, 0),
        .bf16_activation_mirror_hits = @as(usize, 0),
        .f16_cublaslt_linear_calls = @as(usize, 0),
        .f16_cublaslt_qkv_calls = @as(usize, 0),
        .f16_activation_staging_calls = @as(usize, 0),
        .f16_cublaslt_fallbacks = @as(usize, 0),
        .generated_mm_hits = @as(usize, 0),
        .linear_launches = @as(usize, 0),
        .attention_launches = @as(usize, 0),
        .prefill_profile_events = @as(usize, 0),
        .prefill_bf16_linear_us = @as(u64, 0),
        .prefill_bf16_qkv_us = @as(u64, 0),
        .prefill_attention_us = @as(u64, 0),
        .prefill_staging_us = @as(u64, 0),
        .prefill_norm_us = @as(u64, 0),
    };

    const total_ns: u64 = total(samples);
    const embeddings_per_second = if (total_ns == 0) 0.0 else @as(f64, @floatFromInt(opts.batch * opts.measure_iters)) / (@as(f64, @floatFromInt(total_ns)) / 1.0e9);
    var resident_text_buf: [48]u8 = undefined;
    const resident_text = try std.fmt.bufPrint(
        &resident_text_buf,
        "{}/{}",
        .{ resident.text_success - before_resident.text_success, resident.text_fallback - before_resident.text_fallback },
    );
    std.debug.print(
        "bge_m3_e2e backend={s} attention_kernel={s} batch={} seq_len={} iters={} avg_ms={d:.3} p50_ms={d:.3} p95_ms={d:.3} min_ms={d:.3} max_ms={d:.3} embeddings_s={d:.2} resident_text={s} h2d_bytes={} d2h_bytes={} to_float32_calls={} to_float32_bytes={} bf16_cublaslt_linear_calls={} bf16_cublaslt_qkv_calls={} bf16_activation_staging_calls={} bf16_activation_mirror_hits={} generated_q4_0_mm_hits={} linear_launches={} attention_launches={} prefill_profile_events={} prefill_bf16_linear_us={} prefill_bf16_qkv_us={} prefill_attention_us={} prefill_staging_us={} prefill_norm_us={} checksum={d:.6} attention_max_abs={d:.7} attention_cosine={d:.8}\n",
        .{
            @tagName(opts.backend),
            attentionKernelName(opts.attention_kernel),
            opts.batch,
            opts.seq_len,
            opts.measure_iters,
            nsToMs(timing.avg_ns),
            nsToMs(timing.p50_ns),
            nsToMs(timing.p95_ns),
            nsToMs(timing.min_ns),
            nsToMs(timing.max_ns),
            embeddings_per_second,
            resident_text,
            cuda.h2d_bytes,
            cuda.d2h_bytes,
            cuda.to_float32_calls,
            cuda.to_float32_bytes,
            cuda.bf16_cublaslt_linear_calls,
            cuda.bf16_cublaslt_qkv_calls,
            cuda.bf16_activation_staging_calls,
            cuda.bf16_activation_mirror_hits,
            cuda.generated_mm_hits,
            cuda.linear_launches,
            cuda.attention_launches,
            cuda.prefill_profile_events,
            cuda.prefill_bf16_linear_us,
            cuda.prefill_bf16_qkv_us,
            cuda.prefill_attention_us,
            cuda.prefill_staging_us,
            cuda.prefill_norm_us,
            checksum,
            if (attention_validation) |validation| validation.max_abs else @as(f32, -1),
            if (attention_validation) |validation| validation.cosine else @as(f64, -1),
        },
    );
    std.debug.print(
        "bge_m3_e2e f16_cublaslt_linear_calls={} f16_cublaslt_qkv_calls={} f16_activation_staging_calls={} f16_cublaslt_fallbacks={}\n",
        .{
            cuda.f16_cublaslt_linear_calls,
            cuda.f16_cublaslt_qkv_calls,
            cuda.f16_activation_staging_calls,
            cuda.f16_cublaslt_fallbacks,
        },
    );
}

fn parseArgs(init: std.process.Init) !Options {
    var opts = Options{};
    var args = std.process.Args.Iterator.init(init.minimal.args);
    _ = args.next();
    while (args.next()) |arg| {
        if (std.mem.eql(u8, arg, "--model-dir")) {
            opts.model_dir = args.next() orelse return error.MissingModelDir;
        } else if (std.mem.eql(u8, arg, "--backend")) {
            const value = args.next() orelse return error.MissingBackend;
            opts.backend = std.meta.stringToEnum(BackendChoice, value) orelse return error.InvalidBackend;
        } else if (std.mem.eql(u8, arg, "--batch")) {
            opts.batch = try std.fmt.parseInt(usize, args.next() orelse return error.MissingBatch, 10);
        } else if (std.mem.eql(u8, arg, "--seq-len")) {
            opts.seq_len = try std.fmt.parseInt(usize, args.next() orelse return error.MissingSequenceLength, 10);
        } else if (std.mem.eql(u8, arg, "--warmup-iters")) {
            opts.warmup_iters = try std.fmt.parseInt(usize, args.next() orelse return error.MissingWarmupIters, 10);
        } else if (std.mem.eql(u8, arg, "--measure-iters")) {
            opts.measure_iters = try std.fmt.parseInt(usize, args.next() orelse return error.MissingMeasureIters, 10);
        } else if (std.mem.eql(u8, arg, "--attention-kernel")) {
            opts.attention_kernel = parseAttentionKernel(args.next() orelse return error.MissingAttentionKernel) orelse return error.InvalidAttentionKernel;
        } else if (std.mem.eql(u8, arg, "--validate-specialized-attention")) {
            opts.validate_specialized_attention = true;
        } else if (std.mem.eql(u8, arg, "--min-attention-cosine")) {
            opts.min_attention_cosine = try std.fmt.parseFloat(f64, args.next() orelse return error.MissingAttentionCosine);
        } else if (std.mem.eql(u8, arg, "--max-attention-abs")) {
            opts.max_attention_abs = try std.fmt.parseFloat(f32, args.next() orelse return error.MissingAttentionMaxAbs);
        } else if (std.mem.eql(u8, arg, "--kernel-jit-mode")) {
            opts.kernel_jit.mode = std.meta.stringToEnum(kernel_jit.Mode, args.next() orelse return error.MissingKernelJitMode) orelse return error.InvalidKernelJitMode;
        } else if (std.mem.eql(u8, arg, "--kernel-jit-cache-dir")) {
            opts.kernel_jit.cache_dir = args.next() orelse return error.MissingKernelJitCacheDir;
        } else if (std.mem.eql(u8, arg, "--kernel-jit-max-cache-mb")) {
            opts.kernel_jit.max_cache_bytes_mb = try std.fmt.parseInt(usize, args.next() orelse return error.MissingKernelJitMaxCacheMb, 10);
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return error.InvalidArguments;
        } else {
            printUsage();
            return error.InvalidArguments;
        }
    }
    try opts.kernel_jit.validate();
    if (opts.min_attention_cosine) |minimum| {
        if (minimum < -1 or minimum > 1) return error.InvalidAttentionCosine;
    }
    if (opts.max_attention_abs) |maximum| {
        if (maximum < 0) return error.InvalidAttentionMaxAbs;
    }
    return opts;
}

fn parseAttentionKernel(value: []const u8) ?AttentionKernel {
    if (std.ascii.eqlIgnoreCase(value, "generic")) return .generic;
    if (std.ascii.eqlIgnoreCase(value, "exact")) return .exact;
    if (std.ascii.eqlIgnoreCase(value, "exact-q8") or std.ascii.eqlIgnoreCase(value, "exact_q8")) return .exact_q8;
    if (std.ascii.eqlIgnoreCase(value, "tensor-core") or
        std.ascii.eqlIgnoreCase(value, "tensor_core") or
        std.ascii.eqlIgnoreCase(value, "mma")) return .tensor_core;
    return null;
}

fn attentionKernelName(kernel: AttentionKernel) []const u8 {
    return switch (kernel) {
        .generic => "generic",
        .exact => "exact",
        .exact_q8 => "exact-q8",
        .tensor_core => "tensor-core",
    };
}

fn attentionKernelEnvValue(kernel: AttentionKernel) [*:0]const u8 {
    return switch (kernel) {
        .generic => "generic",
        .exact => "exact",
        .exact_q8 => "exact-q8",
        .tensor_core => "tensor-core",
    };
}

fn setAttentionKernel(kernel: AttentionKernel) !void {
    if (setenv("ANTFLY_INFERENCE_CUDA_BERT_PREFILL_ATTENTION", "1", 1) != 0) {
        return error.EnvironmentMutationFailed;
    }
    if (setenv("ANTFLY_INFERENCE_CUDA_BERT_PREFILL_ATTENTION_MODE", attentionKernelEnvValue(kernel), 1) != 0) {
        return error.EnvironmentMutationFailed;
    }
}

fn setAttentionKernelEnabled(enabled: bool) !void {
    if (setenv("ANTFLY_INFERENCE_CUDA_BERT_PREFILL_ATTENTION", if (enabled) "1" else "0", 1) != 0) {
        return error.EnvironmentMutationFailed;
    }
}

fn compareEmbeddings(reference: []const []const f32, candidate: []const []const f32) AttentionValidation {
    var max_abs: f32 = 0;
    var dot: f64 = 0;
    var reference_norm: f64 = 0;
    var candidate_norm: f64 = 0;
    for (reference, candidate) |reference_embedding, candidate_embedding| {
        for (reference_embedding, candidate_embedding) |reference_value, candidate_value| {
            max_abs = @max(max_abs, @abs(reference_value - candidate_value));
            dot += @as(f64, reference_value) * @as(f64, candidate_value);
            reference_norm += @as(f64, reference_value) * @as(f64, reference_value);
            candidate_norm += @as(f64, candidate_value) * @as(f64, candidate_value);
        }
    }
    return .{
        .max_abs = max_abs,
        .cosine = if (reference_norm == 0 or candidate_norm == 0) 0 else dot / @sqrt(reference_norm * candidate_norm),
    };
}

fn summarize(allocator: std.mem.Allocator, samples: []const u64) !Timing {
    const sorted = try allocator.dupe(u64, samples);
    defer allocator.free(sorted);
    std.mem.sort(u64, sorted, {}, std.sort.asc(u64));
    return .{
        .avg_ns = total(samples) / samples.len,
        .p50_ns = sorted[percentileIndex(sorted.len, 50)],
        .p95_ns = sorted[percentileIndex(sorted.len, 95)],
        .min_ns = sorted[0],
        .max_ns = sorted[sorted.len - 1],
    };
}

fn percentileIndex(len: usize, percentile: usize) usize {
    if (len <= 1) return 0;
    const rank = (len * percentile + 99) / 100;
    return @min(len - 1, if (rank == 0) 0 else rank - 1);
}

fn total(samples: []const u64) u64 {
    var value: u64 = 0;
    for (samples) |sample| value += sample;
    return value;
}

fn freeEmbeddings(allocator: std.mem.Allocator, embeddings: [][]f32) void {
    for (embeddings) |embedding| allocator.free(embedding);
    allocator.free(embeddings);
}

fn embeddingChecksum(embeddings: []const []const f32) f64 {
    var sum: f64 = 0;
    for (embeddings) |embedding| {
        for (embedding[0..@min(embedding.len, 16)]) |value| sum += value;
    }
    return sum;
}

fn nowNs() u64 {
    var ts: std.posix.timespec = undefined;
    switch (std.posix.errno(std.posix.system.clock_gettime(std.posix.CLOCK.MONOTONIC, &ts))) {
        .SUCCESS => return @intCast(@as(i128, ts.sec) * std.time.ns_per_s + ts.nsec),
        else => return 0,
    }
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1.0e6;
}

fn printUsage() void {
    std.debug.print(
        "usage: zig build bench-bge-m3-e2e -- --model-dir <bge-m3-q4_0.gguf|dir> [--backend cuda|native] [--batch N] [--seq-len 256] [--warmup-iters N] [--measure-iters N] [--attention-kernel generic|exact|exact-q8|tensor-core] [--validate-specialized-attention] [--min-attention-cosine N] [--max-attention-abs N] [--kernel-jit-mode off|shadow|on|required]\n",
        .{},
    );
}
