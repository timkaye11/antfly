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
const build_options = @import("build_options");

const cuda_buffer = if (build_options.enable_cuda) @import("../ops/cuda/buffer.zig") else struct {};
const cuda_context = if (build_options.enable_cuda) @import("../ops/cuda/context.zig") else struct {};
const cuda_driver = if (build_options.enable_cuda) @import("../ops/cuda/driver.zig") else struct {};
const cuda_artifact = if (build_options.enable_cuda) @import("../ops/cuda/artifact.zig") else struct {
    const image = "";
};
const native_embed = @import("../native_embed.zig");
const quant_kernel_compiler = @import("../graph/quant_kernel_compiler.zig");
const quant_codec = @import("../gguf/quant_codec.zig");
const compat = @import("../io/compat.zig");

const print = std.debug.print;

const q4_k_values_per_block: usize = 256;
const q4_k_block_bytes: usize = 144;
const q8_0_values_per_block: usize = 32;
const q8_0_block_bytes: usize = 34;
const q4_0_values_per_block: usize = 32;
const q4_0_block_bytes: usize = 18;

const Shape = struct {
    label: []const u8,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
};

const shapes = [_]Shape{
    .{ .label = "CLIP text proj/QKV", .rows = 77, .in_dim = 768, .out_dim = 768 },
    .{ .label = "CLIP vision proj/QKV", .rows = 257, .in_dim = 768, .out_dim = 768 },
    .{ .label = "CLIP text MLP up", .rows = 77, .in_dim = 768, .out_dim = 3072 },
    .{ .label = "CLIP text MLP down", .rows = 77, .in_dim = 3072, .out_dim = 768 },
    .{ .label = "CLAP audio MLP up", .rows = 256, .in_dim = 768, .out_dim = 3072 },
    .{ .label = "pooled projection", .rows = 1, .in_dim = 768, .out_dim = 768 },
};

const gemma4_shapes = [_]Shape{
    .{ .label = "Gemma4 Q proj", .rows = 1, .in_dim = 3840, .out_dim = 4096 },
    .{ .label = "Gemma4 KV proj", .rows = 1, .in_dim = 3840, .out_dim = 2048 },
    .{ .label = "Gemma4 attn out", .rows = 1, .in_dim = 4096, .out_dim = 3840 },
    .{ .label = "Gemma4 full attn out", .rows = 1, .in_dim = 8192, .out_dim = 3840 },
    .{ .label = "Gemma4 FFN gate/up", .rows = 1, .in_dim = 3840, .out_dim = 15360 },
    .{ .label = "Gemma4 FFN down", .rows = 1, .in_dim = 15360, .out_dim = 3840 },
    .{ .label = "Gemma4 LM head", .rows = 1, .in_dim = 3840, .out_dim = 262144 },
};

const quant_compiler_lazy_shape = Shape{ .label = "Q4_K compiler lazy", .rows = 8, .in_dim = 512, .out_dim = 768 };

// Gemma4 E2B QAT q4_0 linear_no_bias shapes observed via ANTFLY_CUDA_DISPATCH_STATS.
const quant_compiler_q4_0_dims = [_]Shape{
    .{ .label = "E2B PLE proj", .rows = 0, .in_dim = 256, .out_dim = 1536 },
    .{ .label = "E2B attn out", .rows = 0, .in_dim = 2048, .out_dim = 1536 },
    .{ .label = "E2B FFN down", .rows = 0, .in_dim = 12288, .out_dim = 1536 },
    .{ .label = "E2B FFN up", .rows = 0, .in_dim = 1536, .out_dim = 8960 },
};
const quant_compiler_q4_0_mmv_rows: usize = 1;
const quant_compiler_q4_0_mm_rows: usize = 22;

// mmv-only coverage for the largest production rows==1 dispatch (Gemma4 LM
// head). The CPU reference dequantizes the whole weight tensor to dense f32,
// which is impractical above this threshold; larger shapes are cross-checked
// generated-vs-baseline on device instead.
const quant_compiler_q4_0_mmv_extra_dims = [_]Shape{
    .{ .label = "E2B LM head", .rows = 0, .in_dim = 1536, .out_dim = 262144 },
};
const quant_compiler_q4_0_max_reference_out_dim: usize = 16384;

const QuantCompilerQ4_0Kind = enum { mmv, mm };

const Config = struct {
    warmup_iters: usize = 5,
    measure_iters: usize = 50,
    model_path: ?[]const u8 = null,
    text: []const u8 = "a photo of a document with audio metadata",
    full_iters: usize = 1,
    gemma4_shapes: bool = false,
    quant_compiler_lazy_target: bool = false,
    quant_compiler_generated_ptx_path: ?[]const u8 = null,
    quant_compiler_q4_0_mmv_ptx_path: ?[]const u8 = null,
    quant_compiler_q4_0_mm_ptx_path: ?[]const u8 = null,
    quant_compiler_q4_0_pair_ptx_path: ?[]const u8 = null,
    quant_compiler_q4_0_pair_q8_ptx_path: ?[]const u8 = null,
    quant_compiler_q4_0_down_q8_ptx_path: ?[]const u8 = null,
    quant_compiler_evidence_out_path: ?[]const u8 = null,
    quant_compiler_check_evidence_path: ?[]const u8 = null,
    quant_compiler_require_promotion_ready: bool = false,
    quant_compiler_repeat_runs: usize = 1,
    json_out_path: ?[]const u8 = null,
};

const quant_compiler_evidence_repeat_runs: usize = 3;

const BenchModule = if (build_options.enable_cuda) struct {
    module: cuda_driver.CUmodule = null,
    linear_q4_k_f32: cuda_driver.CUfunction = null,
    linear_q4_k_bias_f32: cuda_driver.CUfunction = null,
    linear_q8_0_f32: cuda_driver.CUfunction = null,
    linear_q8_0_f32_tile4: cuda_driver.CUfunction = null,
    linear_q4_k_f32_tiled: cuda_driver.CUfunction = null,
    linear_q4_k_bias_f32_tiled: cuda_driver.CUfunction = null,
    linear_q4_k_bias_quick_gelu_f32_tiled: cuda_driver.CUfunction = null,
    linear_q4_k_f32_tile4: cuda_driver.CUfunction = null,
    linear_q4_k_f32_tile8: cuda_driver.CUfunction = null,
    linear_q4_k_bias_f32_tile4: cuda_driver.CUfunction = null,
    linear_q4_k_bias_gelu_f32_tile4_r2: cuda_driver.CUfunction = null,
    linear_q4_k_bias_quick_gelu_f32_tile4: cuda_driver.CUfunction = null,
    linear_q4_k_triple_bias_f32: cuda_driver.CUfunction = null,
    linear_q4_k_triple_bias_f32_tiled: cuda_driver.CUfunction = null,
    linear_q4_0_f32: cuda_driver.CUfunction = null,
    linear_q4_0_f32_tile4: cuda_driver.CUfunction = null,
    linear_q4_0_pair_nobias_f32_tile4_w4: cuda_driver.CUfunction = null,
    linear_q4_0_pair_activation_q8_1_e4b: cuda_driver.CUfunction = null,
    linear_q4_0_q8_1_e4b_down: cuda_driver.CUfunction = null,

    fn load(ctx: *cuda_context.CudaContext) cuda_driver.Error!BenchModule {
        try ctx.makeCurrent();
        var module: cuda_driver.CUmodule = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleLoadDataEx(&module, cuda_artifact.image.ptr, 0, null, null));
        errdefer _ = ctx.driver.fns.cuModuleUnload(module);

        var linear_q4_k_f32: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_k_f32, module, "termite_linear_q4_k_f32"));
        var linear_q4_k_bias_f32: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_k_bias_f32, module, "termite_linear_q4_k_bias_f32"));
        var linear_q8_0_f32: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q8_0_f32, module, "termite_linear_q8_0_f32"));
        var linear_q8_0_f32_tile4: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q8_0_f32_tile4, module, "termite_linear_q8_0_f32_tile4"));
        var linear_q4_k_f32_tiled: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_k_f32_tiled, module, "termite_linear_q4_k_f32_tiled"));
        var linear_q4_k_bias_f32_tiled: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_k_bias_f32_tiled, module, "termite_linear_q4_k_bias_f32_tiled"));
        var linear_q4_k_bias_quick_gelu_f32_tiled: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_k_bias_quick_gelu_f32_tiled, module, "termite_linear_q4_k_bias_quick_gelu_f32_tiled"));
        var linear_q4_k_f32_tile4: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_k_f32_tile4, module, "termite_linear_q4_k_f32_tile4"));
        var linear_q4_k_f32_tile8: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_k_f32_tile8, module, "termite_linear_q4_k_f32_tile8"));
        var linear_q4_k_bias_f32_tile4: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_k_bias_f32_tile4, module, "termite_linear_q4_k_bias_f32_tile4"));
        var linear_q4_k_bias_gelu_f32_tile4_r2: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_k_bias_gelu_f32_tile4_r2, module, "termite_linear_q4_k_bias_gelu_f32_tile4_r2"));
        var linear_q4_k_bias_quick_gelu_f32_tile4: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_k_bias_quick_gelu_f32_tile4, module, "termite_linear_q4_k_bias_quick_gelu_f32_tile4"));
        var linear_q4_k_triple_bias_f32: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_k_triple_bias_f32, module, "termite_linear_q4_k_triple_bias_f32"));
        var linear_q4_k_triple_bias_f32_tiled: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_k_triple_bias_f32_tiled, module, "termite_linear_q4_k_triple_bias_f32_tiled"));
        var linear_q4_0_f32: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_0_f32, module, "termite_linear_q4_0_f32"));
        var linear_q4_0_f32_tile4: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_0_f32_tile4, module, "termite_linear_q4_0_f32_tile4"));
        var linear_q4_0_pair_nobias_f32_tile4_w4: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_0_pair_nobias_f32_tile4_w4, module, "termite_linear_q4_0_pair_nobias_f32_tile4_w4"));
        var linear_q4_0_pair_activation_q8_1_e4b: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_0_pair_activation_q8_1_e4b, module, "termite_linear_q4_0_pair_activation_q8_1_q8_1_tile32_w5_e4b_ffn"));
        var linear_q4_0_q8_1_e4b_down: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&linear_q4_0_q8_1_e4b_down, module, "termite_linear_q4_0_q8_1_f32_tile4_w8_e4b_down"));

        return .{
            .module = module,
            .linear_q4_k_f32 = linear_q4_k_f32,
            .linear_q4_k_bias_f32 = linear_q4_k_bias_f32,
            .linear_q8_0_f32 = linear_q8_0_f32,
            .linear_q8_0_f32_tile4 = linear_q8_0_f32_tile4,
            .linear_q4_k_f32_tiled = linear_q4_k_f32_tiled,
            .linear_q4_k_bias_f32_tiled = linear_q4_k_bias_f32_tiled,
            .linear_q4_k_bias_quick_gelu_f32_tiled = linear_q4_k_bias_quick_gelu_f32_tiled,
            .linear_q4_k_f32_tile4 = linear_q4_k_f32_tile4,
            .linear_q4_k_f32_tile8 = linear_q4_k_f32_tile8,
            .linear_q4_k_bias_f32_tile4 = linear_q4_k_bias_f32_tile4,
            .linear_q4_k_bias_gelu_f32_tile4_r2 = linear_q4_k_bias_gelu_f32_tile4_r2,
            .linear_q4_k_bias_quick_gelu_f32_tile4 = linear_q4_k_bias_quick_gelu_f32_tile4,
            .linear_q4_k_triple_bias_f32 = linear_q4_k_triple_bias_f32,
            .linear_q4_k_triple_bias_f32_tiled = linear_q4_k_triple_bias_f32_tiled,
            .linear_q4_0_f32 = linear_q4_0_f32,
            .linear_q4_0_f32_tile4 = linear_q4_0_f32_tile4,
            .linear_q4_0_pair_nobias_f32_tile4_w4 = linear_q4_0_pair_nobias_f32_tile4_w4,
            .linear_q4_0_pair_activation_q8_1_e4b = linear_q4_0_pair_activation_q8_1_e4b,
            .linear_q4_0_q8_1_e4b_down = linear_q4_0_q8_1_e4b_down,
        };
    }

    fn unload(self: *BenchModule, ctx: *cuda_context.CudaContext) void {
        if (self.module != null) {
            ctx.makeCurrent() catch {};
            _ = ctx.driver.fns.cuModuleUnload(self.module);
            self.module = null;
            self.linear_q4_k_f32 = null;
            self.linear_q4_k_bias_f32 = null;
            self.linear_q8_0_f32 = null;
            self.linear_q8_0_f32_tile4 = null;
            self.linear_q4_k_f32_tiled = null;
            self.linear_q4_k_bias_f32_tiled = null;
            self.linear_q4_k_bias_quick_gelu_f32_tiled = null;
            self.linear_q4_k_f32_tile4 = null;
            self.linear_q4_k_f32_tile8 = null;
            self.linear_q4_k_bias_f32_tile4 = null;
            self.linear_q4_k_bias_gelu_f32_tile4_r2 = null;
            self.linear_q4_k_bias_quick_gelu_f32_tile4 = null;
            self.linear_q4_k_triple_bias_f32 = null;
            self.linear_q4_k_triple_bias_f32_tiled = null;
            self.linear_q4_0_f32 = null;
            self.linear_q4_0_f32_tile4 = null;
            self.linear_q4_0_pair_nobias_f32_tile4_w4 = null;
            self.linear_q4_0_pair_activation_q8_1_e4b = null;
            self.linear_q4_0_q8_1_e4b_down = null;
        }
    }
} else struct {};

pub fn main(allocator: std.mem.Allocator, io: std.Io, args: []const []const u8) !void {
    if (wantsHelp(args)) {
        printUsage();
        return;
    }
    const cfg = try parseArgs(args);
    if (cfg.quant_compiler_check_evidence_path) |path| {
        try checkQuantCompilerEvidenceFile(allocator, io, path, cfg.quant_compiler_require_promotion_ready);
        print("bench-cuda quant compiler evidence_check={s} ok\n", .{path});
        return;
    }
    if (!build_options.enable_cuda) {
        print("bench-cuda requires a build with -Dcuda=true\n", .{});
        return error.CudaUnavailable;
    }

    try runKernelBench(allocator, io, cfg);
    if (cfg.model_path) |model_path| {
        try runFullTextEmbedBench(allocator, io, cfg, model_path);
    }
}

fn parseArgs(args: []const []const u8) !Config {
    var cfg = Config{};
    var seen_quant_compiler_lazy_target = false;
    var seen_benchmark_option = false;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--warmup-iters")) {
            seen_benchmark_option = true;
            i += 1;
            if (i >= args.len) return error.MissingWarmupIters;
            cfg.warmup_iters = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--measure-iters")) {
            seen_benchmark_option = true;
            i += 1;
            if (i >= args.len) return error.MissingMeasureIters;
            cfg.measure_iters = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--model")) {
            seen_benchmark_option = true;
            i += 1;
            if (i >= args.len) return error.MissingModelPath;
            cfg.model_path = args[i];
        } else if (std.mem.eql(u8, arg, "--text")) {
            seen_benchmark_option = true;
            i += 1;
            if (i >= args.len) return error.MissingText;
            cfg.text = args[i];
        } else if (std.mem.eql(u8, arg, "--full-iters")) {
            seen_benchmark_option = true;
            i += 1;
            if (i >= args.len) return error.MissingFullIters;
            cfg.full_iters = try std.fmt.parseInt(usize, args[i], 10);
        } else if (std.mem.eql(u8, arg, "--gemma4-shapes")) {
            seen_benchmark_option = true;
            cfg.gemma4_shapes = true;
        } else if (std.mem.eql(u8, arg, "--quant-compiler-lazy-target")) {
            if (seen_quant_compiler_lazy_target) return error.DuplicateQuantCompilerLazyTarget;
            seen_quant_compiler_lazy_target = true;
            seen_benchmark_option = true;
            cfg.quant_compiler_lazy_target = true;
        } else if (std.mem.eql(u8, arg, "--quant-compiler-generated-ptx")) {
            if (cfg.quant_compiler_generated_ptx_path != null) return error.DuplicateGeneratedPtxPath;
            seen_benchmark_option = true;
            i += 1;
            if (i >= args.len) return error.MissingGeneratedPtxPath;
            cfg.quant_compiler_lazy_target = true;
            cfg.quant_compiler_generated_ptx_path = args[i];
        } else if (std.mem.eql(u8, arg, "--quant-compiler-q4-0-mmv-ptx")) {
            if (cfg.quant_compiler_q4_0_mmv_ptx_path != null) return error.DuplicateQuantCompilerQ4_0MmvPtxPath;
            seen_benchmark_option = true;
            i += 1;
            if (i >= args.len) return error.MissingQuantCompilerQ4_0MmvPtxPath;
            cfg.quant_compiler_q4_0_mmv_ptx_path = args[i];
        } else if (std.mem.eql(u8, arg, "--quant-compiler-q4-0-mm-ptx")) {
            if (cfg.quant_compiler_q4_0_mm_ptx_path != null) return error.DuplicateQuantCompilerQ4_0MmPtxPath;
            seen_benchmark_option = true;
            i += 1;
            if (i >= args.len) return error.MissingQuantCompilerQ4_0MmPtxPath;
            cfg.quant_compiler_q4_0_mm_ptx_path = args[i];
        } else if (std.mem.eql(u8, arg, "--quant-compiler-q4-0-pair-ptx")) {
            if (cfg.quant_compiler_q4_0_pair_ptx_path != null) return error.DuplicateQuantCompilerQ4_0PairPtxPath;
            seen_benchmark_option = true;
            i += 1;
            if (i >= args.len) return error.MissingQuantCompilerQ4_0PairPtxPath;
            cfg.quant_compiler_q4_0_pair_ptx_path = args[i];
        } else if (std.mem.eql(u8, arg, "--quant-compiler-q4-0-pair-q8-ptx")) {
            if (cfg.quant_compiler_q4_0_pair_q8_ptx_path != null) return error.DuplicateQuantCompilerQ4_0PairQ8PtxPath;
            seen_benchmark_option = true;
            i += 1;
            if (i >= args.len) return error.MissingQuantCompilerQ4_0PairQ8PtxPath;
            cfg.quant_compiler_q4_0_pair_q8_ptx_path = args[i];
        } else if (std.mem.eql(u8, arg, "--quant-compiler-q4-0-down-q8-ptx")) {
            if (cfg.quant_compiler_q4_0_down_q8_ptx_path != null) return error.DuplicateQuantCompilerQ4_0DownQ8PtxPath;
            seen_benchmark_option = true;
            i += 1;
            if (i >= args.len) return error.MissingQuantCompilerQ4_0DownQ8PtxPath;
            cfg.quant_compiler_q4_0_down_q8_ptx_path = args[i];
        } else if (std.mem.eql(u8, arg, "--quant-compiler-evidence-out")) {
            if (cfg.quant_compiler_evidence_out_path != null) return error.DuplicateQuantCompilerEvidenceOutPath;
            seen_benchmark_option = true;
            i += 1;
            if (i >= args.len) return error.MissingQuantCompilerEvidenceOutPath;
            cfg.quant_compiler_evidence_out_path = args[i];
        } else if (std.mem.eql(u8, arg, "--quant-compiler-check-evidence")) {
            if (cfg.quant_compiler_check_evidence_path != null) return error.DuplicateQuantCompilerCheckEvidencePath;
            i += 1;
            if (i >= args.len) return error.MissingQuantCompilerCheckEvidencePath;
            cfg.quant_compiler_check_evidence_path = args[i];
        } else if (std.mem.eql(u8, arg, "--quant-compiler-require-promotion-ready")) {
            if (cfg.quant_compiler_require_promotion_ready) return error.DuplicateQuantCompilerRequirePromotionReady;
            cfg.quant_compiler_require_promotion_ready = true;
        } else if (std.mem.eql(u8, arg, "--quant-compiler-repeat-runs")) {
            if (cfg.quant_compiler_repeat_runs != 1) return error.DuplicateQuantCompilerRepeatRuns;
            seen_benchmark_option = true;
            i += 1;
            if (i >= args.len) return error.MissingQuantCompilerRepeatRuns;
            cfg.quant_compiler_repeat_runs = try parseQuantCompilerRepeatRuns(args[i]);
        } else if (std.mem.eql(u8, arg, "--json-out")) {
            seen_benchmark_option = true;
            i += 1;
            if (i >= args.len) return error.MissingJsonOutPath;
            cfg.json_out_path = args[i];
        } else {
            print("unknown bench-cuda argument: {s}\n", .{arg});
            printUsage();
            return error.InvalidArgument;
        }
    }
    if (cfg.quant_compiler_require_promotion_ready and cfg.quant_compiler_check_evidence_path == null) return error.QuantCompilerPromotionReadyRequiresCheckEvidence;
    if (cfg.quant_compiler_check_evidence_path != null) {
        if (seen_benchmark_option) return error.QuantCompilerCheckEvidenceConflictsWithBenchmark;
        return cfg;
    }
    if (cfg.measure_iters == 0) return error.InvalidArgument;
    if (cfg.full_iters == 0) return error.InvalidArgument;
    if (cfg.quant_compiler_lazy_target and cfg.quant_compiler_generated_ptx_path == null) return error.MissingGeneratedPtxPath;
    if (cfg.quant_compiler_lazy_target and (cfg.quant_compiler_q4_0_mmv_ptx_path != null or cfg.quant_compiler_q4_0_mm_ptx_path != null or cfg.quant_compiler_q4_0_pair_ptx_path != null or cfg.quant_compiler_q4_0_pair_q8_ptx_path != null or cfg.quant_compiler_q4_0_down_q8_ptx_path != null)) return error.QuantCompilerQ4_0ConflictsWithLazyTarget;
    const q4_0_target_count: usize = @as(usize, @intFromBool(cfg.quant_compiler_q4_0_mmv_ptx_path != null)) + @as(usize, @intFromBool(cfg.quant_compiler_q4_0_mm_ptx_path != null)) + @as(usize, @intFromBool(cfg.quant_compiler_q4_0_pair_ptx_path != null)) + @as(usize, @intFromBool(cfg.quant_compiler_q4_0_pair_q8_ptx_path != null)) + @as(usize, @intFromBool(cfg.quant_compiler_q4_0_down_q8_ptx_path != null));
    if (cfg.quant_compiler_evidence_out_path != null and !cfg.quant_compiler_lazy_target and q4_0_target_count == 0) return error.QuantCompilerEvidenceRequiresLazyTarget;
    if (cfg.quant_compiler_evidence_out_path != null and q4_0_target_count > 1) return error.QuantCompilerEvidenceRequiresSingleQ4_0Target;
    if (cfg.quant_compiler_evidence_out_path != null and (cfg.warmup_iters != 5 or cfg.measure_iters != 50)) return error.QuantCompilerEvidenceRequiresManifestIterations;
    if (cfg.quant_compiler_evidence_out_path != null and cfg.quant_compiler_repeat_runs != quant_compiler_evidence_repeat_runs) return error.QuantCompilerEvidenceRequiresManifestRepeatRuns;
    return cfg;
}

fn parseQuantCompilerRepeatRuns(text: []const u8) !usize {
    const runs = try std.fmt.parseInt(usize, text, 10);
    if (runs == 0 or runs > 31) return error.InvalidQuantCompilerRepeatRuns;
    return runs;
}

fn wantsHelp(args: []const []const u8) bool {
    for (args) |arg| {
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) return true;
    }
    return false;
}

test "cuda microbench quant compiler flags select lazy target" {
    const cfg = try parseArgs(&.{ "--quant-compiler-generated-ptx", "/tmp/generated.ptx", "--quant-compiler-repeat-runs", "3", "--quant-compiler-evidence-out", "/tmp/evidence.json" });
    try std.testing.expect(cfg.quant_compiler_lazy_target);
    try std.testing.expectEqualStrings("/tmp/generated.ptx", cfg.quant_compiler_generated_ptx_path.?);
    try std.testing.expectEqualStrings("/tmp/evidence.json", cfg.quant_compiler_evidence_out_path.?);
    try std.testing.expectEqual(@as(usize, 3), cfg.quant_compiler_repeat_runs);
    try std.testing.expectError(error.MissingGeneratedPtxPath, parseArgs(&.{"--quant-compiler-generated-ptx"}));
    try std.testing.expectError(error.MissingGeneratedPtxPath, parseArgs(&.{"--quant-compiler-lazy-target"}));
    try std.testing.expectError(error.MissingQuantCompilerEvidenceOutPath, parseArgs(&.{"--quant-compiler-evidence-out"}));
    const check_cfg = try parseArgs(&.{ "--quant-compiler-check-evidence", "/tmp/evidence.json" });
    try std.testing.expectEqualStrings("/tmp/evidence.json", check_cfg.quant_compiler_check_evidence_path.?);
    const promotion_check_cfg = try parseArgs(&.{ "--quant-compiler-check-evidence", "/tmp/evidence.json", "--quant-compiler-require-promotion-ready" });
    try std.testing.expect(promotion_check_cfg.quant_compiler_require_promotion_ready);
    try std.testing.expectError(error.MissingQuantCompilerCheckEvidencePath, parseArgs(&.{"--quant-compiler-check-evidence"}));
    try std.testing.expectError(error.DuplicateQuantCompilerLazyTarget, parseArgs(&.{ "--quant-compiler-lazy-target", "--quant-compiler-lazy-target" }));
    try std.testing.expectError(error.DuplicateGeneratedPtxPath, parseArgs(&.{ "--quant-compiler-generated-ptx", "/tmp/a.ptx", "--quant-compiler-generated-ptx", "/tmp/b.ptx" }));
    try std.testing.expectError(error.DuplicateQuantCompilerEvidenceOutPath, parseArgs(&.{ "--quant-compiler-evidence-out", "/tmp/a.json", "--quant-compiler-evidence-out", "/tmp/b.json" }));
    try std.testing.expectError(error.DuplicateQuantCompilerCheckEvidencePath, parseArgs(&.{ "--quant-compiler-check-evidence", "/tmp/a.json", "--quant-compiler-check-evidence", "/tmp/b.json" }));
    try std.testing.expectError(error.DuplicateQuantCompilerRequirePromotionReady, parseArgs(&.{ "--quant-compiler-check-evidence", "/tmp/evidence.json", "--quant-compiler-require-promotion-ready", "--quant-compiler-require-promotion-ready" }));
    try std.testing.expectError(error.QuantCompilerPromotionReadyRequiresCheckEvidence, parseArgs(&.{"--quant-compiler-require-promotion-ready"}));
    try std.testing.expectError(error.QuantCompilerCheckEvidenceConflictsWithBenchmark, parseArgs(&.{ "--quant-compiler-check-evidence", "/tmp/evidence.json", "--quant-compiler-lazy-target" }));
    try std.testing.expectError(error.QuantCompilerCheckEvidenceConflictsWithBenchmark, parseArgs(&.{ "--quant-compiler-check-evidence", "/tmp/evidence.json", "--quant-compiler-generated-ptx", "/tmp/generated.ptx" }));
    try std.testing.expectError(error.QuantCompilerCheckEvidenceConflictsWithBenchmark, parseArgs(&.{ "--quant-compiler-check-evidence", "/tmp/evidence.json", "--quant-compiler-evidence-out", "/tmp/out.json" }));
    try std.testing.expectError(error.QuantCompilerCheckEvidenceConflictsWithBenchmark, parseArgs(&.{ "--quant-compiler-check-evidence", "/tmp/evidence.json", "--warmup-iters", "5" }));
    try std.testing.expectError(error.QuantCompilerCheckEvidenceConflictsWithBenchmark, parseArgs(&.{ "--quant-compiler-check-evidence", "/tmp/evidence.json", "--gemma4-shapes" }));
    try std.testing.expectError(error.QuantCompilerCheckEvidenceConflictsWithBenchmark, parseArgs(&.{ "--quant-compiler-check-evidence", "/tmp/evidence.json", "--json-out", "/tmp/out.json" }));
    try std.testing.expectError(error.QuantCompilerCheckEvidenceConflictsWithBenchmark, parseArgs(&.{ "--quant-compiler-check-evidence", "/tmp/evidence.json", "--quant-compiler-repeat-runs", "3" }));
    try std.testing.expectError(error.MissingQuantCompilerRepeatRuns, parseArgs(&.{"--quant-compiler-repeat-runs"}));
    try std.testing.expectError(error.InvalidQuantCompilerRepeatRuns, parseArgs(&.{ "--quant-compiler-repeat-runs", "0" }));
    try std.testing.expectError(error.DuplicateQuantCompilerRepeatRuns, parseArgs(&.{ "--quant-compiler-repeat-runs", "2", "--quant-compiler-repeat-runs", "3" }));
    try std.testing.expectError(error.QuantCompilerEvidenceRequiresLazyTarget, parseArgs(&.{ "--quant-compiler-evidence-out", "/tmp/evidence.json" }));
    try std.testing.expectError(error.QuantCompilerEvidenceRequiresManifestRepeatRuns, parseArgs(&.{ "--quant-compiler-generated-ptx", "/tmp/generated.ptx", "--quant-compiler-evidence-out", "/tmp/evidence.json" }));
    try std.testing.expectError(error.QuantCompilerEvidenceRequiresManifestIterations, parseArgs(&.{ "--quant-compiler-generated-ptx", "/tmp/generated.ptx", "--warmup-iters", "1", "--quant-compiler-evidence-out", "/tmp/evidence.json" }));
    try std.testing.expectError(error.QuantCompilerEvidenceRequiresManifestIterations, parseArgs(&.{ "--quant-compiler-generated-ptx", "/tmp/generated.ptx", "--measure-iters", "49", "--quant-compiler-evidence-out", "/tmp/evidence.json" }));

    const repeat_cfg = try parseArgs(&.{ "--quant-compiler-generated-ptx", "/tmp/generated.ptx", "--quant-compiler-repeat-runs", "3", "--quant-compiler-evidence-out", "/tmp/evidence.json" });
    try std.testing.expectEqual(@as(usize, 3), repeat_cfg.quant_compiler_repeat_runs);

    const q4_0_cfg = try parseArgs(&.{ "--quant-compiler-q4-0-mmv-ptx", "/tmp/mmv.ptx", "--quant-compiler-q4-0-mm-ptx", "/tmp/mm.ptx" });
    try std.testing.expectEqualStrings("/tmp/mmv.ptx", q4_0_cfg.quant_compiler_q4_0_mmv_ptx_path.?);
    try std.testing.expectEqualStrings("/tmp/mm.ptx", q4_0_cfg.quant_compiler_q4_0_mm_ptx_path.?);
    try std.testing.expect(!q4_0_cfg.quant_compiler_lazy_target);
    try std.testing.expectError(error.MissingQuantCompilerQ4_0MmvPtxPath, parseArgs(&.{"--quant-compiler-q4-0-mmv-ptx"}));
    try std.testing.expectError(error.MissingQuantCompilerQ4_0MmPtxPath, parseArgs(&.{"--quant-compiler-q4-0-mm-ptx"}));
    try std.testing.expectError(error.DuplicateQuantCompilerQ4_0MmvPtxPath, parseArgs(&.{ "--quant-compiler-q4-0-mmv-ptx", "/tmp/a.ptx", "--quant-compiler-q4-0-mmv-ptx", "/tmp/b.ptx" }));
    try std.testing.expectError(error.DuplicateQuantCompilerQ4_0MmPtxPath, parseArgs(&.{ "--quant-compiler-q4-0-mm-ptx", "/tmp/a.ptx", "--quant-compiler-q4-0-mm-ptx", "/tmp/b.ptx" }));
    try std.testing.expectError(error.QuantCompilerQ4_0ConflictsWithLazyTarget, parseArgs(&.{ "--quant-compiler-generated-ptx", "/tmp/generated.ptx", "--quant-compiler-q4-0-mmv-ptx", "/tmp/mmv.ptx" }));
    try std.testing.expectError(error.QuantCompilerCheckEvidenceConflictsWithBenchmark, parseArgs(&.{ "--quant-compiler-check-evidence", "/tmp/evidence.json", "--quant-compiler-q4-0-mmv-ptx", "/tmp/mmv.ptx" }));
}

fn printUsage() void {
    print(
        \\usage: antfly inference bench-cuda [--warmup-iters N] [--measure-iters N]
        \\                         [--model <clipclap-model-dir>] [--text <prompt>] [--full-iters N]
        \\                         [--gemma4-shapes] [--json-out PATH]
        \\                         [--quant-compiler-lazy-target --quant-compiler-generated-ptx PATH]
        \\                         [--quant-compiler-q4-0-mmv-ptx PATH] [--quant-compiler-q4-0-mm-ptx PATH]
        \\                         [--quant-compiler-q4-0-pair-ptx PATH]
        \\                         [--quant-compiler-q4-0-pair-q8-ptx PATH] [--quant-compiler-q4-0-down-q8-ptx PATH]
        \\                         [--quant-compiler-evidence-out PATH]
        \\                         [--quant-compiler-check-evidence PATH]
        \\                         [--quant-compiler-require-promotion-ready]
        \\                         [--quant-compiler-repeat-runs N]
        \\
        \\Benchmarks CUDA Q4_K linear kernels on CLIP/CLAP-sized shapes.
        \\With --gemma4-shapes, also benchmarks Gemma4 Q8_0/Q4_K decode-sized matmuls.
        \\With --quant-compiler-generated-ptx, compares the dev generated Q4_K rows 2..8 bias_gelu
        \\candidate against the checked-in handwritten CUDA baseline and CPU quant reference.
        \\With --quant-compiler-q4-0-mmv-ptx / --quant-compiler-q4-0-mm-ptx, compares the
        \\generated Q4_0 rows=1 / rows 9..64 no-bias kernels against the handwritten
        \\q4_simt kernels on Gemma4 E2B QAT shapes.
        \\With --quant-compiler-q4-0-pair-ptx, compares the generated Q4_0 rows=1 FFN
        \\gate+up pair kernel against the handwritten pair baseline.
        \\With --quant-compiler-q4-0-pair-q8-ptx / --quant-compiler-q4-0-down-q8-ptx,
        \\compares the generated q8_1/DP4A E4B fused-FFN kernels against the tuned
        \\handwritten baselines.
        \\If --model is provided, also runs full ClipCLAP text embedding through the CUDA backend.
        \\
    , .{});
}

fn runKernelBench(allocator: std.mem.Allocator, io: std.Io, cfg: Config) !void {
    var ctx = try cuda_context.CudaContext.initDefault();
    defer ctx.deinit();

    var module = try BenchModule.load(&ctx);
    defer module.unload(&ctx);

    if (cfg.quant_compiler_lazy_target) {
        try benchQuantCompilerLazyTarget(allocator, io, &ctx, &module, cfg);
        return;
    }

    if (cfg.quant_compiler_q4_0_mmv_ptx_path != null or cfg.quant_compiler_q4_0_mm_ptx_path != null or cfg.quant_compiler_q4_0_pair_ptx_path != null or cfg.quant_compiler_q4_0_pair_q8_ptx_path != null or cfg.quant_compiler_q4_0_down_q8_ptx_path != null) {
        if (cfg.quant_compiler_q4_0_mmv_ptx_path) |ptx_path| {
            try benchQuantCompilerQ4_0Target(allocator, io, &ctx, &module, cfg, .mmv, ptx_path);
        }
        if (cfg.quant_compiler_q4_0_mm_ptx_path) |ptx_path| {
            try benchQuantCompilerQ4_0Target(allocator, io, &ctx, &module, cfg, .mm, ptx_path);
        }
        if (cfg.quant_compiler_q4_0_pair_ptx_path) |ptx_path| {
            try benchQuantCompilerQ4_0PairTarget(allocator, io, &ctx, &module, cfg, ptx_path);
        }
        if (cfg.quant_compiler_q4_0_pair_q8_ptx_path) |ptx_path| {
            try benchQuantCompilerQ4_0PairQ8Target(allocator, io, &ctx, &module, cfg, ptx_path);
        }
        if (cfg.quant_compiler_q4_0_down_q8_ptx_path) |ptx_path| {
            try benchQuantCompilerQ4_0DownQ8Target(allocator, io, &ctx, &module, cfg, ptx_path);
        }
        return;
    }

    print("CUDA Q4_K microbench: device={s} cc={d}.{d} warmup={d} measure={d}\n", .{
        ctx.info.nameSlice(),
        ctx.info.compute_major,
        ctx.info.compute_minor,
        cfg.warmup_iters,
        cfg.measure_iters,
    });
    print("{s:<24} {s:>8} {s:>8} {s:>8} {s:>14} {s:>14} {s:>14} {s:>14} {s:>14} {s:>12}\n", .{
        "shape",
        "rows",
        "in",
        "out",
        "scalar ns",
        "tiled ns",
        "scalar+b ns",
        "tiled+b ns",
        "qgelu ns",
        "checksum",
    });
    print("{s:<24} {s:>8} {s:>8} {s:>8} {s:>14} {s:>14} {s:>14} {s:>14} {s:>14} {s:>12}\n", .{
        "-----",
        "----",
        "--",
        "---",
        "------",
        "------",
        "-----------",
        "---------",
        "--------",
        "--------",
    });

    for (shapes) |shape| {
        try benchShape(allocator, &ctx, &module, cfg, shape);
    }

    print("\n{s:<24} {s:>8} {s:>8} {s:>8} {s:>14} {s:>14} {s:>12}\n", .{
        "triple shape",
        "rows",
        "in",
        "out",
        "scalar ns",
        "tiled ns",
        "checksum",
    });
    print("{s:<24} {s:>8} {s:>8} {s:>8} {s:>14} {s:>14} {s:>12}\n", .{
        "------------",
        "----",
        "--",
        "---",
        "---------",
        "--------",
        "--------",
    });
    try benchTripleShape(allocator, &ctx, &module, cfg, .{ .label = "CLIP text QKV", .rows = 77, .in_dim = 768, .out_dim = 768 });
    try benchTripleShape(allocator, &ctx, &module, cfg, .{ .label = "CLIP vision QKV", .rows = 257, .in_dim = 768, .out_dim = 768 });

    if (cfg.gemma4_shapes) {
        try runGemma4KernelBench(allocator, io, &ctx, &module, cfg);
    }
}

fn benchShape(
    allocator: std.mem.Allocator,
    ctx: *cuda_context.CudaContext,
    module: *BenchModule,
    cfg: Config,
    shape: Shape,
) !void {
    if (shape.in_dim % q4_k_values_per_block != 0) return error.InvalidArgument;

    const input_count = try std.math.mul(usize, shape.rows, shape.in_dim);
    const output_count = try std.math.mul(usize, shape.rows, shape.out_dim);
    const row_blocks = shape.in_dim / q4_k_values_per_block;
    const weight_bytes = try std.math.mul(usize, try std.math.mul(usize, shape.out_dim, row_blocks), q4_k_block_bytes);

    const input_host = try allocator.alloc(f32, input_count);
    defer allocator.free(input_host);
    const bias_host = try allocator.alloc(f32, shape.out_dim);
    defer allocator.free(bias_host);
    const weight_host = try allocator.alloc(u8, weight_bytes);
    defer allocator.free(weight_host);
    fillInput(input_host);
    fillBias(bias_host);
    fillQ4KWeights(weight_host);

    var input = try cuda_buffer.DeviceBuffer.alloc(ctx, input_count * @sizeOf(f32));
    defer input.free(ctx);
    var weight = try cuda_buffer.DeviceBuffer.alloc(ctx, weight_host.len);
    defer weight.free(ctx);
    var bias = try cuda_buffer.DeviceBuffer.alloc(ctx, bias_host.len * @sizeOf(f32));
    defer bias.free(ctx);
    var output = try cuda_buffer.DeviceBuffer.alloc(ctx, output_count * @sizeOf(f32));
    defer output.free(ctx);

    try input.copyFromHost(ctx, std.mem.sliceAsBytes(input_host));
    try weight.copyFromHost(ctx, weight_host);
    try bias.copyFromHost(ctx, std.mem.sliceAsBytes(bias_host));
    try ctx.synchronize();

    const no_bias_ns = try timeCudaStep(ctx, cfg, launchQ4K, .{ module, ctx, output, input, weight, shape.rows, shape.in_dim, shape.out_dim });
    const no_bias_tiled_ns = try timeCudaStep(ctx, cfg, launchQ4KTiled, .{ module, ctx, output, input, weight, shape.rows, shape.in_dim, shape.out_dim });
    const bias_ns = try timeCudaStep(ctx, cfg, launchQ4KBias, .{ module, ctx, output, input, weight, bias, shape.rows, shape.in_dim, shape.out_dim });
    const bias_tiled_ns = try timeCudaStep(ctx, cfg, launchQ4KBiasTiled, .{ module, ctx, output, input, weight, bias, shape.rows, shape.in_dim, shape.out_dim });
    const quick_gelu_ns = try timeCudaStep(ctx, cfg, launchQ4KBiasQuickGeluTiled, .{ module, ctx, output, input, weight, bias, shape.rows, shape.in_dim, shape.out_dim });

    var sample: [1]f32 = undefined;
    try output.copyToHost(ctx, std.mem.sliceAsBytes(&sample));
    try ctx.synchronize();

    print("{s:<24} {d:>8} {d:>8} {d:>8} {d:>14} {d:>14} {d:>14} {d:>14} {d:>14} {d:>12.4}\n", .{
        shape.label,
        shape.rows,
        shape.in_dim,
        shape.out_dim,
        no_bias_ns,
        no_bias_tiled_ns,
        bias_ns,
        bias_tiled_ns,
        quick_gelu_ns,
        sample[0],
    });
}

const GeneratedCandidateModule = if (build_options.enable_cuda) struct {
    module: cuda_driver.CUmodule = null,
    function: cuda_driver.CUfunction = null,

    fn load(
        allocator: std.mem.Allocator,
        io: std.Io,
        ctx: *cuda_context.CudaContext,
        ptx_path: []const u8,
        kernel_id: []const u8,
    ) !GeneratedCandidateModule {
        const ptx = try std.Io.Dir.cwd().readFileAlloc(io, ptx_path, allocator, .limited(16 * 1024 * 1024));
        defer allocator.free(ptx);
        const image = try allocator.alloc(u8, ptx.len + 1);
        defer allocator.free(image);
        @memcpy(image[0..ptx.len], ptx);
        image[ptx.len] = 0;

        try ctx.makeCurrent();
        var module: cuda_driver.CUmodule = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleLoadDataEx(&module, image.ptr, 0, null, null));
        errdefer _ = ctx.driver.fns.cuModuleUnload(module);

        const kernel_name = try allocator.dupeZ(u8, kernel_id);
        defer allocator.free(kernel_name);
        var function: cuda_driver.CUfunction = null;
        try ctx.driver.check(ctx.driver.fns.cuModuleGetFunction(&function, module, kernel_name));

        return .{
            .module = module,
            .function = function,
        };
    }

    fn unload(self: *GeneratedCandidateModule, ctx: *cuda_context.CudaContext) void {
        if (self.module != null) {
            ctx.makeCurrent() catch {};
            _ = ctx.driver.fns.cuModuleUnload(self.module);
            self.module = null;
            self.function = null;
        }
    }
} else struct {};

fn benchQuantCompilerLazyTarget(
    allocator: std.mem.Allocator,
    io: std.Io,
    ctx: *cuda_context.CudaContext,
    module: *BenchModule,
    cfg: Config,
) !void {
    const bench = quant_kernel_compiler.first_lazy_benchmark;
    const lowering = quant_kernel_compiler.registryLoweringFor(.cuda, bench.format, bench.row_bucket, bench.epilogue, .small_batch);
    const route_diagnostic = try quant_kernel_compiler.loweringDiagnostic(allocator, lowering);
    defer allocator.free(route_diagnostic);
    const shape = quant_compiler_lazy_shape;
    const input_count = try std.math.mul(usize, shape.rows, shape.in_dim);
    const output_count = try std.math.mul(usize, shape.rows, shape.out_dim);
    const row_blocks = shape.in_dim / q4_k_values_per_block;
    const weight_bytes = try std.math.mul(usize, try std.math.mul(usize, shape.out_dim, row_blocks), q4_k_block_bytes);

    const input_host = try allocator.alloc(f32, input_count);
    defer allocator.free(input_host);
    const bias_host = try allocator.alloc(f32, shape.out_dim);
    defer allocator.free(bias_host);
    const weight_host = try allocator.alloc(u8, weight_bytes);
    defer allocator.free(weight_host);
    fillInput(input_host);
    fillBias(bias_host);
    fillQ4KWeights(weight_host);

    const reference_host = try allocator.alloc(f32, output_count);
    defer allocator.free(reference_host);
    try quant_kernel_compiler.referenceMatmulBiasGelu(
        allocator,
        bench.format,
        weight_host,
        input_host,
        bias_host,
        shape.rows,
        shape.in_dim,
        shape.out_dim,
        reference_host,
    );

    var input = try cuda_buffer.DeviceBuffer.alloc(ctx, input_count * @sizeOf(f32));
    defer input.free(ctx);
    var weight = try cuda_buffer.DeviceBuffer.alloc(ctx, weight_host.len);
    defer weight.free(ctx);
    var bias = try cuda_buffer.DeviceBuffer.alloc(ctx, bias_host.len * @sizeOf(f32));
    defer bias.free(ctx);
    var output = try cuda_buffer.DeviceBuffer.alloc(ctx, output_count * @sizeOf(f32));
    defer output.free(ctx);

    try input.copyFromHost(ctx, std.mem.sliceAsBytes(input_host));
    try weight.copyFromHost(ctx, weight_host);
    try bias.copyFromHost(ctx, std.mem.sliceAsBytes(bias_host));
    try ctx.synchronize();

    const baseline_ns = try repeatCudaStep(allocator, ctx, cfg, launchQ4KBiasGeluRows2, .{ module, ctx, output, input, weight, bias, shape.rows, shape.in_dim, shape.out_dim });
    const baseline_host = try allocator.alloc(f32, output_count);
    defer allocator.free(baseline_host);
    try output.copyToHost(ctx, std.mem.sliceAsBytes(baseline_host));
    try ctx.synchronize();
    const baseline_cpu_max_abs_diff = try maxAbsDiffFinite(baseline_host, reference_host);
    if (baseline_cpu_max_abs_diff > bench.correctness_tolerance_abs) return error.HandwrittenBaselineMismatch;

    // Poison the shared output buffer so a generated kernel that silently
    // early-returns cannot inherit the baseline's results and pass.
    try poisonDeviceOutput(allocator, ctx, output, output_count);

    print("CUDA quant compiler lazy target: {s} rows={d} in={d} out={d}\n", .{ route_diagnostic, shape.rows, shape.in_dim, shape.out_dim });
    print("baseline kernel={s} ns={d} checksum={d:.6} cpu_max_abs_diff={d:.6}\n", .{ bench.handwritten_baseline, baseline_ns, baseline_host[0], baseline_cpu_max_abs_diff });
    print("candidate kernel={s} source={s} production=false\n", .{ bench.generated_kernel_id, bench.generated_source_path });

    const ptx_path = cfg.quant_compiler_generated_ptx_path orelse return error.MissingGeneratedPtxPath;
    if (cfg.quant_compiler_evidence_out_path) |evidence_path| try validateQuantCompilerEvidenceRequest(bench, ptx_path, evidence_path);

    var generated = try GeneratedCandidateModule.load(allocator, io, ctx, ptx_path, bench.generated_kernel_id);
    defer generated.unload(ctx);
    const generated_ns = try repeatCudaStep(allocator, ctx, cfg, launchGeneratedQ4KBiasGeluRows2, .{ &generated, ctx, output, input, weight, bias, shape.rows, shape.in_dim, shape.out_dim });

    const generated_host = try allocator.alloc(f32, output_count);
    defer allocator.free(generated_host);
    try output.copyToHost(ctx, std.mem.sliceAsBytes(generated_host));
    try ctx.synchronize();

    const baseline_max_abs_diff = try maxAbsDiffFinite(generated_host, baseline_host);
    const cpu_max_abs_diff = try maxAbsDiffFinite(generated_host, reference_host);
    if (cpu_max_abs_diff > bench.correctness_tolerance_abs) return error.GeneratedCandidateMismatch;
    const candidate_speedup = speedup(baseline_ns, generated_ns);
    print("generated ptx={s} ns={d} speedup={d:.6} min_speedup={d:.6} checksum={d:.6} baseline_max_abs_diff={d:.6} cpu_max_abs_diff={d:.6} tolerance={d:.6}\n", .{
        ptx_path,
        generated_ns,
        candidate_speedup,
        bench.minimum_speedup,
        generated_host[0],
        baseline_max_abs_diff,
        cpu_max_abs_diff,
        bench.correctness_tolerance_abs,
    });
    if (candidate_speedup < bench.minimum_speedup) return error.GeneratedCandidateSlowerThanBaseline;
    if (cfg.quant_compiler_evidence_out_path) |evidence_path| {
        try writeQuantCompilerEvidence(
            allocator,
            io,
            evidence_path,
            bench,
            shape,
            ptx_path,
            cfg,
            baseline_ns,
            generated_ns,
            baseline_host[0],
            generated_host[0],
            baseline_cpu_max_abs_diff,
            baseline_max_abs_diff,
            cpu_max_abs_diff,
        );
        print("wrote quant compiler evidence: {s}\n", .{evidence_path});
    }
}

fn benchQuantCompilerQ4_0Target(
    allocator: std.mem.Allocator,
    io: std.Io,
    ctx: *cuda_context.CudaContext,
    module: *BenchModule,
    cfg: Config,
    kind: QuantCompilerQ4_0Kind,
    ptx_path: []const u8,
) !void {
    const bench = switch (kind) {
        .mmv => quant_kernel_compiler.first_q4_0_mmv_benchmark,
        .mm => quant_kernel_compiler.first_q4_0_mm_benchmark,
    };
    const kernel_id = bench.generated_kernel_id;
    const rows: usize = switch (kind) {
        .mmv => quant_compiler_q4_0_mmv_rows,
        .mm => quant_compiler_q4_0_mm_rows,
    };
    const baseline_name = bench.handwritten_baseline;
    const baseline_function = switch (kind) {
        .mmv => module.linear_q4_0_f32_tile4,
        .mm => module.linear_q4_0_f32,
    };
    const lowering = switch (kind) {
        .mmv => quant_kernel_compiler.registryLoweringFor(.cuda, .q4_0, .rows_1, .none, .mmv),
        .mm => quant_kernel_compiler.registryLoweringFor(.cuda, .q4_0, .rows_9_64, .none, .mm),
    };
    const route_diagnostic = try quant_kernel_compiler.loweringDiagnostic(allocator, lowering);
    defer allocator.free(route_diagnostic);
    const correctness_tolerance_abs: f32 = bench.correctness_tolerance_abs;
    if (cfg.quant_compiler_evidence_out_path) |evidence_path| {
        if (!std.mem.eql(u8, ptx_path, bench.generated_ptx_path)) return error.GeneratedPtxPathMismatch;
        if (!std.mem.eql(u8, evidence_path, bench.benchmark_evidence_path)) return error.QuantCompilerEvidencePathMismatch;
    }

    var generated = try GeneratedCandidateModule.load(allocator, io, ctx, ptx_path, kernel_id);
    defer generated.unload(ctx);

    print("CUDA quant compiler q4_0 {s} target: {s} rows={d} ptx={s}\n", .{ @tagName(kind), route_diagnostic, rows, ptx_path });
    print("candidate kernel={s} baseline kernel={s} production={} tolerance={d:.6}\n", .{ kernel_id, baseline_name, bench.production_enabled, correctness_tolerance_abs });

    var dims_list = std.ArrayListUnmanaged(Shape).empty;
    defer dims_list.deinit(allocator);
    try dims_list.appendSlice(allocator, &quant_compiler_q4_0_dims);
    if (kind == .mmv) try dims_list.appendSlice(allocator, &quant_compiler_q4_0_mmv_extra_dims);

    var worst_speedup: f64 = std.math.inf(f64);
    var speedup_log_sum: f64 = 0.0;
    var shape_details = std.ArrayListUnmanaged(u8).empty;
    defer shape_details.deinit(allocator);
    for (dims_list.items, 0..) |dims, shape_index| {
        const shape = Shape{ .label = dims.label, .rows = rows, .in_dim = dims.in_dim, .out_dim = dims.out_dim };
        const has_cpu_reference = shape.out_dim <= quant_compiler_q4_0_max_reference_out_dim;
        const input_count = try std.math.mul(usize, shape.rows, shape.in_dim);
        const output_count = try std.math.mul(usize, shape.rows, shape.out_dim);
        const row_blocks = shape.in_dim / q4_0_values_per_block;
        const weight_bytes = try std.math.mul(usize, try std.math.mul(usize, shape.out_dim, row_blocks), q4_0_block_bytes);

        const input_host = try allocator.alloc(f32, input_count);
        defer allocator.free(input_host);
        const weight_host = try allocator.alloc(u8, weight_bytes);
        defer allocator.free(weight_host);
        fillInput(input_host);
        fillQ4_0Weights(weight_host);

        const reference_host = try allocator.alloc(f32, output_count);
        defer allocator.free(reference_host);
        if (has_cpu_reference) {
            try quant_kernel_compiler.referenceMatmulNoBias(
                allocator,
                .q4_0,
                weight_host,
                input_host,
                shape.rows,
                shape.in_dim,
                shape.out_dim,
                reference_host,
            );
        }

        var input = try cuda_buffer.DeviceBuffer.alloc(ctx, input_count * @sizeOf(f32));
        defer input.free(ctx);
        var weight = try cuda_buffer.DeviceBuffer.alloc(ctx, weight_host.len);
        defer weight.free(ctx);
        var output = try cuda_buffer.DeviceBuffer.alloc(ctx, output_count * @sizeOf(f32));
        defer output.free(ctx);

        try input.copyFromHost(ctx, std.mem.sliceAsBytes(input_host));
        try weight.copyFromHost(ctx, weight_host);
        try ctx.synchronize();

        const baseline_ns = try repeatCudaStep(allocator, ctx, cfg, launchQ4_0Baseline, .{ baseline_function, ctx, kind, output, input, weight, shape.rows, shape.in_dim, shape.out_dim });
        const baseline_host = try allocator.alloc(f32, output_count);
        defer allocator.free(baseline_host);
        try output.copyToHost(ctx, std.mem.sliceAsBytes(baseline_host));
        try ctx.synchronize();
        const baseline_cpu_max_abs_diff = if (has_cpu_reference) try maxAbsDiffFinite(baseline_host, reference_host) else 0.0;
        if (has_cpu_reference and baseline_cpu_max_abs_diff > correctness_tolerance_abs) return error.HandwrittenBaselineMismatch;

        // Poison the shared output buffer so a generated kernel that silently
        // early-returns cannot inherit the baseline's results and pass.
        try poisonDeviceOutput(allocator, ctx, output, output_count);

        const generated_ns = try repeatCudaStep(allocator, ctx, cfg, launchGeneratedQ4_0, .{ &generated, ctx, kind, output, input, weight, shape.rows, shape.in_dim, shape.out_dim });
        const generated_host = try allocator.alloc(f32, output_count);
        defer allocator.free(generated_host);
        try output.copyToHost(ctx, std.mem.sliceAsBytes(generated_host));
        try ctx.synchronize();

        const baseline_max_abs_diff = try maxAbsDiffFinite(generated_host, baseline_host);
        if (baseline_max_abs_diff > correctness_tolerance_abs) return error.GeneratedCandidateMismatch;
        const cpu_max_abs_diff = if (has_cpu_reference) try maxAbsDiffFinite(generated_host, reference_host) else 0.0;
        if (has_cpu_reference and cpu_max_abs_diff > correctness_tolerance_abs) return error.GeneratedCandidateMismatch;

        const candidate_speedup = speedup(baseline_ns, generated_ns);
        worst_speedup = @min(worst_speedup, candidate_speedup);
        speedup_log_sum += @log(candidate_speedup);
        print("shape={s} rows={d} in={d} out={d} baseline_ns={d} generated_ns={d} speedup={d:.6} cpu_reference={} baseline_cpu_max_abs_diff={d:.6} generated_cpu_max_abs_diff={d:.6} generated_baseline_max_abs_diff={d:.6}\n", .{
            shape.label,
            shape.rows,
            shape.in_dim,
            shape.out_dim,
            baseline_ns,
            generated_ns,
            candidate_speedup,
            has_cpu_reference,
            baseline_cpu_max_abs_diff,
            cpu_max_abs_diff,
            baseline_max_abs_diff,
        });
        try appendJsonFmt(allocator, &shape_details,
            \\{s}{{"label":{f},"rows":{d},"in_dim":{d},"out_dim":{d},"baseline_ns":{d},"generated_ns":{d},"speedup":{d:.6},"cpu_reference":{},"baseline_cpu_max_abs_diff":{d:.6},"generated_cpu_max_abs_diff":{d:.6},"generated_baseline_max_abs_diff":{d:.6}}}
        , .{
            if (shape_index == 0) "" else ",",
            std.json.fmt(shape.label, .{}),
            shape.rows,
            shape.in_dim,
            shape.out_dim,
            baseline_ns,
            generated_ns,
            candidate_speedup,
            has_cpu_reference,
            baseline_cpu_max_abs_diff,
            cpu_max_abs_diff,
            baseline_max_abs_diff,
        });
    }
    const geomean_speedup = @exp(speedup_log_sum / @as(f64, @floatFromInt(dims_list.items.len)));
    print("q4_0 {s} summary: geomean_speedup={d:.6} worst_speedup={d:.6} shapes={d}\n", .{ @tagName(kind), geomean_speedup, worst_speedup, dims_list.items.len });
    if (cfg.quant_compiler_evidence_out_path) |evidence_path| {
        try writeQuantCompilerQ4_0Evidence(allocator, io, evidence_path, bench, cfg, geomean_speedup, worst_speedup, shape_details.items);
        print("wrote quant compiler evidence: {s}\n", .{evidence_path});
    }
}

const quant_compiler_q4_0_pair_dims = [_]Shape{
    .{ .label = "E2B FFN gate+up", .rows = 1, .in_dim = 1536, .out_dim = 8960 },
    .{ .label = "E4B FFN gate+up", .rows = 1, .in_dim = 2560, .out_dim = 10240 },
};

fn benchQuantCompilerQ4_0PairTarget(
    allocator: std.mem.Allocator,
    io: std.Io,
    ctx: *cuda_context.CudaContext,
    module: *BenchModule,
    cfg: Config,
    ptx_path: []const u8,
) !void {
    const bench = quant_kernel_compiler.first_q4_0_pair_benchmark;
    const kernel_id = bench.generated_kernel_id;
    const baseline_name = bench.handwritten_baseline;
    const lowering = quant_kernel_compiler.registryLoweringFor(.cuda, .q4_0, .rows_1, .pair, .mmv);
    const route_diagnostic = try quant_kernel_compiler.loweringDiagnostic(allocator, lowering);
    defer allocator.free(route_diagnostic);
    const correctness_tolerance_abs: f32 = bench.correctness_tolerance_abs;
    if (cfg.quant_compiler_evidence_out_path) |evidence_path| {
        if (!std.mem.eql(u8, ptx_path, bench.generated_ptx_path)) return error.GeneratedPtxPathMismatch;
        if (!std.mem.eql(u8, evidence_path, bench.benchmark_evidence_path)) return error.QuantCompilerEvidencePathMismatch;
    }

    var generated = try GeneratedCandidateModule.load(allocator, io, ctx, ptx_path, kernel_id);
    defer generated.unload(ctx);

    print("CUDA quant compiler q4_0 pair target: {s} rows=1 ptx={s}\n", .{ route_diagnostic, ptx_path });
    print("candidate kernel={s} baseline kernel={s} production={} tolerance={d:.6}\n", .{ kernel_id, baseline_name, bench.production_enabled, correctness_tolerance_abs });

    var worst_speedup: f64 = std.math.inf(f64);
    var speedup_log_sum: f64 = 0.0;
    var shape_details = std.ArrayListUnmanaged(u8).empty;
    defer shape_details.deinit(allocator);
    for (quant_compiler_q4_0_pair_dims, 0..) |shape, shape_index| {
        const input_count = shape.in_dim;
        const output_count = shape.out_dim;
        const row_blocks = shape.in_dim / q4_0_values_per_block;
        const weight_bytes = try std.math.mul(usize, try std.math.mul(usize, shape.out_dim, row_blocks), q4_0_block_bytes);

        const input_host = try allocator.alloc(f32, input_count);
        defer allocator.free(input_host);
        const weight_a_host = try allocator.alloc(u8, weight_bytes);
        defer allocator.free(weight_a_host);
        const weight_b_host = try allocator.alloc(u8, weight_bytes);
        defer allocator.free(weight_b_host);
        fillInput(input_host);
        fillQ4_0Weights(weight_a_host);
        fillQ4_0WeightsAlt(weight_b_host);

        const reference_a = try allocator.alloc(f32, output_count);
        defer allocator.free(reference_a);
        const reference_b = try allocator.alloc(f32, output_count);
        defer allocator.free(reference_b);
        try quant_kernel_compiler.referenceMatmulNoBias(allocator, .q4_0, weight_a_host, input_host, 1, shape.in_dim, shape.out_dim, reference_a);
        try quant_kernel_compiler.referenceMatmulNoBias(allocator, .q4_0, weight_b_host, input_host, 1, shape.in_dim, shape.out_dim, reference_b);

        var input = try cuda_buffer.DeviceBuffer.alloc(ctx, input_count * @sizeOf(f32));
        defer input.free(ctx);
        var weight_a = try cuda_buffer.DeviceBuffer.alloc(ctx, weight_a_host.len);
        defer weight_a.free(ctx);
        var weight_b = try cuda_buffer.DeviceBuffer.alloc(ctx, weight_b_host.len);
        defer weight_b.free(ctx);
        var output_a = try cuda_buffer.DeviceBuffer.alloc(ctx, output_count * @sizeOf(f32));
        defer output_a.free(ctx);
        var output_b = try cuda_buffer.DeviceBuffer.alloc(ctx, output_count * @sizeOf(f32));
        defer output_b.free(ctx);

        try input.copyFromHost(ctx, std.mem.sliceAsBytes(input_host));
        try weight_a.copyFromHost(ctx, weight_a_host);
        try weight_b.copyFromHost(ctx, weight_b_host);
        try ctx.synchronize();

        const baseline_ns = try repeatCudaStep(allocator, ctx, cfg, launchQ4_0PairBaseline, .{ module, ctx, output_a, output_b, input, weight_a, weight_b, shape.in_dim, shape.out_dim });
        const baseline_a = try allocator.alloc(f32, output_count);
        defer allocator.free(baseline_a);
        const baseline_b = try allocator.alloc(f32, output_count);
        defer allocator.free(baseline_b);
        try output_a.copyToHost(ctx, std.mem.sliceAsBytes(baseline_a));
        try output_b.copyToHost(ctx, std.mem.sliceAsBytes(baseline_b));
        try ctx.synchronize();
        if (try maxAbsDiffFinite(baseline_a, reference_a) > correctness_tolerance_abs) return error.HandwrittenBaselineMismatch;
        if (try maxAbsDiffFinite(baseline_b, reference_b) > correctness_tolerance_abs) return error.HandwrittenBaselineMismatch;

        // Poison the shared output buffers so a generated kernel that silently
        // early-returns cannot inherit the baseline's results and pass.
        try poisonDeviceOutput(allocator, ctx, output_a, output_count);
        try poisonDeviceOutput(allocator, ctx, output_b, output_count);

        const generated_ns = try repeatCudaStep(allocator, ctx, cfg, launchGeneratedQ4_0Pair, .{ &generated, ctx, output_a, output_b, input, weight_a, weight_b, shape.in_dim, shape.out_dim });
        const generated_a = try allocator.alloc(f32, output_count);
        defer allocator.free(generated_a);
        const generated_b = try allocator.alloc(f32, output_count);
        defer allocator.free(generated_b);
        try output_a.copyToHost(ctx, std.mem.sliceAsBytes(generated_a));
        try output_b.copyToHost(ctx, std.mem.sliceAsBytes(generated_b));
        try ctx.synchronize();

        const cpu_max_abs_diff_a = try maxAbsDiffFinite(generated_a, reference_a);
        const cpu_max_abs_diff_b = try maxAbsDiffFinite(generated_b, reference_b);
        if (cpu_max_abs_diff_a > correctness_tolerance_abs or cpu_max_abs_diff_b > correctness_tolerance_abs) return error.GeneratedCandidateMismatch;

        const candidate_speedup = speedup(baseline_ns, generated_ns);
        worst_speedup = @min(worst_speedup, candidate_speedup);
        speedup_log_sum += @log(candidate_speedup);
        print("shape={s} rows=1 in={d} out={d} baseline_ns={d} generated_ns={d} speedup={d:.6} cpu_max_abs_diff_a={d:.6} cpu_max_abs_diff_b={d:.6}\n", .{
            shape.label,
            shape.in_dim,
            shape.out_dim,
            baseline_ns,
            generated_ns,
            candidate_speedup,
            cpu_max_abs_diff_a,
            cpu_max_abs_diff_b,
        });
        try appendJsonFmt(allocator, &shape_details,
            \\{s}{{"label":{f},"rows":1,"in_dim":{d},"out_dim":{d},"baseline_ns":{d},"generated_ns":{d},"speedup":{d:.6},"generated_cpu_max_abs_diff_a":{d:.6},"generated_cpu_max_abs_diff_b":{d:.6}}}
        , .{
            if (shape_index == 0) "" else ",",
            std.json.fmt(shape.label, .{}),
            shape.in_dim,
            shape.out_dim,
            baseline_ns,
            generated_ns,
            candidate_speedup,
            cpu_max_abs_diff_a,
            cpu_max_abs_diff_b,
        });
    }
    const geomean_speedup = @exp(speedup_log_sum / @as(f64, @floatFromInt(quant_compiler_q4_0_pair_dims.len)));
    print("q4_0 pair summary: geomean_speedup={d:.6} worst_speedup={d:.6} shapes={d}\n", .{ geomean_speedup, worst_speedup, quant_compiler_q4_0_pair_dims.len });
    if (cfg.quant_compiler_evidence_out_path) |evidence_path| {
        try writeQuantCompilerQ4_0Evidence(allocator, io, evidence_path, bench, cfg, geomean_speedup, worst_speedup, shape_details.items);
        print("wrote quant compiler evidence: {s}\n", .{evidence_path});
    }
}

// Experimental race harness for the tuned Gemma4 E4B decode FFN gate+up
// kernel (q8_1 activations in, fused activation multiply, q8_1 out). The
// candidate is compiler-owned (plan cuda/q4_0/rows_1/pair_activation/mmv)
// and shares the baseline's exact launch contract.
const quant_compiler_q4_0_pair_q8_kernel_id = "antfly_q4_0_pair_activation_q8_1_mmv_v1";
const quant_compiler_q4_0_pair_q8_in_dim: usize = 2560;
const quant_compiler_q4_0_pair_q8_out_dim: usize = 10240;
const quant_compiler_q4_0_pair_q8_activation: u32 = 0; // GELU-tanh
const q8_1_values_per_block: usize = 32;
const q8_1_block_bytes: usize = 36;

fn geluTanhF32(x: f32) f32 {
    const inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}

fn benchQuantCompilerQ4_0PairQ8Target(
    allocator: std.mem.Allocator,
    io: std.Io,
    ctx: *cuda_context.CudaContext,
    module: *BenchModule,
    cfg: Config,
    ptx_path: []const u8,
) !void {
    const bench = quant_kernel_compiler.first_q4_0_pair_q8_benchmark;
    const in_dim = quant_compiler_q4_0_pair_q8_in_dim;
    const out_dim = quant_compiler_q4_0_pair_q8_out_dim;
    const out_blocks = out_dim / q8_1_values_per_block;
    const row_blocks = in_dim / q4_0_values_per_block;
    const weight_bytes = out_dim * row_blocks * q4_0_block_bytes;
    const baseline_name = bench.handwritten_baseline;
    if (cfg.quant_compiler_evidence_out_path) |evidence_path| {
        if (!std.mem.eql(u8, ptx_path, bench.generated_ptx_path)) return error.GeneratedPtxPathMismatch;
        if (!std.mem.eql(u8, evidence_path, bench.benchmark_evidence_path)) return error.QuantCompilerEvidencePathMismatch;
    }

    var generated = try GeneratedCandidateModule.load(allocator, io, ctx, ptx_path, bench.generated_kernel_id);
    defer generated.unload(ctx);

    print("CUDA quant compiler q4_0 pair-q8 target: rows=1 in={d} out={d} activation=gelu_tanh ptx={s}\n", .{ in_dim, out_dim, ptx_path });
    print("candidate kernel={s} baseline kernel={s} production={}\n", .{ bench.generated_kernel_id, baseline_name, bench.production_enabled });

    const input_host = try allocator.alloc(f32, in_dim);
    defer allocator.free(input_host);
    fillInput(input_host);
    const q8_input_host = try quant_codec.quantizeQ8_1FromF32(allocator, input_host);
    defer allocator.free(q8_input_host);
    const weight_gate_host = try allocator.alloc(u8, weight_bytes);
    defer allocator.free(weight_gate_host);
    const weight_up_host = try allocator.alloc(u8, weight_bytes);
    defer allocator.free(weight_up_host);
    fillQ4_0Weights(weight_gate_host);
    fillQ4_0WeightsAlt(weight_up_host);

    // CPU reference: dequantize the q8 activations the kernels actually
    // consume, run both projections through the q4_0 reference matmul, then
    // apply the fused activation multiply.
    const act_dequant = try allocator.alloc(f32, in_dim);
    defer allocator.free(act_dequant);
    try quant_codec.dequantizeToFloat32(.{ .known = .Q8_1 }, q8_input_host, act_dequant);
    const gate_ref = try allocator.alloc(f32, out_dim);
    defer allocator.free(gate_ref);
    const up_ref = try allocator.alloc(f32, out_dim);
    defer allocator.free(up_ref);
    try quant_kernel_compiler.referenceMatmulNoBias(allocator, .q4_0, weight_gate_host, act_dequant, 1, in_dim, out_dim, gate_ref);
    try quant_kernel_compiler.referenceMatmulNoBias(allocator, .q4_0, weight_up_host, act_dequant, 1, in_dim, out_dim, up_ref);
    const activated_ref = try allocator.alloc(f32, out_dim);
    defer allocator.free(activated_ref);
    var activated_amax: f32 = 0.0;
    for (activated_ref, 0..) |*value, i| {
        value.* = geluTanhF32(gate_ref[i]) * up_ref[i];
        activated_amax = @max(activated_amax, @abs(value.*));
    }
    // The kernel output is per-block q8_1-quantized; allow one quantization
    // step at the global amax plus dot-product float noise.
    const correctness_tolerance_abs: f32 = activated_amax / 127.0 + 0.05;

    var q8_input = try cuda_buffer.DeviceBuffer.alloc(ctx, q8_input_host.len);
    defer q8_input.free(ctx);
    var weight_gate = try cuda_buffer.DeviceBuffer.alloc(ctx, weight_bytes);
    defer weight_gate.free(ctx);
    var weight_up = try cuda_buffer.DeviceBuffer.alloc(ctx, weight_bytes);
    defer weight_up.free(ctx);
    var dst_q8 = try cuda_buffer.DeviceBuffer.alloc(ctx, out_blocks * q8_1_block_bytes);
    defer dst_q8.free(ctx);
    try q8_input.copyFromHost(ctx, q8_input_host);
    try weight_gate.copyFromHost(ctx, weight_gate_host);
    try weight_up.copyFromHost(ctx, weight_up_host);
    try ctx.synchronize();

    const baseline_ns = try repeatCudaStep(allocator, ctx, cfg, launchQ4_0PairQ8, .{ module.linear_q4_0_pair_activation_q8_1_e4b, ctx, dst_q8, q8_input, weight_gate, weight_up, out_blocks });
    const baseline_q8_host = try allocator.alloc(u8, out_blocks * q8_1_block_bytes);
    defer allocator.free(baseline_q8_host);
    try dst_q8.copyToHost(ctx, baseline_q8_host);
    try ctx.synchronize();
    const baseline_dequant = try allocator.alloc(f32, out_dim);
    defer allocator.free(baseline_dequant);
    try quant_codec.dequantizeToFloat32(.{ .known = .Q8_1 }, baseline_q8_host, baseline_dequant);
    const baseline_cpu_max_abs_diff = try maxAbsDiffFinite(baseline_dequant, activated_ref);
    if (baseline_cpu_max_abs_diff > correctness_tolerance_abs) return error.HandwrittenBaselineMismatch;

    try poisonDeviceBytes(allocator, ctx, dst_q8, out_blocks * q8_1_block_bytes);

    const generated_ns = try repeatCudaStep(allocator, ctx, cfg, launchQ4_0PairQ8, .{ generated.function, ctx, dst_q8, q8_input, weight_gate, weight_up, out_blocks });
    const generated_q8_host = try allocator.alloc(u8, out_blocks * q8_1_block_bytes);
    defer allocator.free(generated_q8_host);
    try dst_q8.copyToHost(ctx, generated_q8_host);
    try ctx.synchronize();
    const generated_dequant = try allocator.alloc(f32, out_dim);
    defer allocator.free(generated_dequant);
    try quant_codec.dequantizeToFloat32(.{ .known = .Q8_1 }, generated_q8_host, generated_dequant);
    const cpu_max_abs_diff = try maxAbsDiffFinite(generated_dequant, activated_ref);
    if (cpu_max_abs_diff > correctness_tolerance_abs) return error.GeneratedCandidateMismatch;
    const baseline_max_abs_diff = try maxAbsDiffFinite(generated_dequant, baseline_dequant);

    const candidate_speedup = speedup(baseline_ns, generated_ns);
    print("shape=E4B FFN gate+up q8_1 rows=1 in={d} out={d} baseline_ns={d} generated_ns={d} speedup={d:.6} tolerance={d:.6} baseline_cpu_max_abs_diff={d:.6} generated_cpu_max_abs_diff={d:.6} generated_baseline_max_abs_diff={d:.6}\n", .{
        in_dim,
        out_dim,
        baseline_ns,
        generated_ns,
        candidate_speedup,
        correctness_tolerance_abs,
        baseline_cpu_max_abs_diff,
        cpu_max_abs_diff,
        baseline_max_abs_diff,
    });
    print("q4_0 pair-q8 summary: speedup={d:.6}\n", .{candidate_speedup});
    if (cfg.quant_compiler_evidence_out_path) |evidence_path| {
        var shape_details = std.ArrayListUnmanaged(u8).empty;
        defer shape_details.deinit(allocator);
        try appendJsonFmt(allocator, &shape_details,
            \\{{"label":"E4B FFN gate+up q8_1","rows":1,"in_dim":{d},"out_dim":{d},"baseline_ns":{d},"generated_ns":{d},"speedup":{d:.6},"quantized_output_tolerance":{d:.6},"baseline_cpu_max_abs_diff":{d:.6},"generated_cpu_max_abs_diff":{d:.6},"generated_baseline_max_abs_diff":{d:.6}}}
        , .{
            in_dim,
            out_dim,
            baseline_ns,
            generated_ns,
            candidate_speedup,
            correctness_tolerance_abs,
            baseline_cpu_max_abs_diff,
            cpu_max_abs_diff,
            baseline_max_abs_diff,
        });
        try writeQuantCompilerQ4_0Evidence(allocator, io, evidence_path, bench, cfg, candidate_speedup, candidate_speedup, shape_details.items);
        print("wrote quant compiler evidence: {s}\n", .{evidence_path});
    }
}

const quant_compiler_q4_0_down_q8_kernel_id = "antfly_q4_0_down_q8_1_mmv_v1";
const quant_compiler_q4_0_down_q8_in_dim: usize = 10240;
const quant_compiler_q4_0_down_q8_out_dim: usize = 2560;

fn benchQuantCompilerQ4_0DownQ8Target(
    allocator: std.mem.Allocator,
    io: std.Io,
    ctx: *cuda_context.CudaContext,
    module: *BenchModule,
    cfg: Config,
    ptx_path: []const u8,
) !void {
    const bench = quant_kernel_compiler.first_q4_0_down_q8_benchmark;
    const in_dim = quant_compiler_q4_0_down_q8_in_dim;
    const out_dim = quant_compiler_q4_0_down_q8_out_dim;
    const row_blocks = in_dim / q4_0_values_per_block;
    const weight_bytes = out_dim * row_blocks * q4_0_block_bytes;
    const baseline_name = bench.handwritten_baseline;
    const correctness_tolerance_abs: f32 = bench.correctness_tolerance_abs;
    if (cfg.quant_compiler_evidence_out_path) |evidence_path| {
        if (!std.mem.eql(u8, ptx_path, bench.generated_ptx_path)) return error.GeneratedPtxPathMismatch;
        if (!std.mem.eql(u8, evidence_path, bench.benchmark_evidence_path)) return error.QuantCompilerEvidencePathMismatch;
    }

    var generated = try GeneratedCandidateModule.load(allocator, io, ctx, ptx_path, bench.generated_kernel_id);
    defer generated.unload(ctx);

    print("CUDA quant compiler q4_0 down-q8 target: rows=1 in={d} out={d} ptx={s}\n", .{ in_dim, out_dim, ptx_path });
    print("candidate kernel={s} baseline kernel={s} production={} tolerance={d:.6}\n", .{ bench.generated_kernel_id, baseline_name, bench.production_enabled, correctness_tolerance_abs });

    const input_host = try allocator.alloc(f32, in_dim);
    defer allocator.free(input_host);
    fillInput(input_host);
    const q8_input_host = try quant_codec.quantizeQ8_1FromF32(allocator, input_host);
    defer allocator.free(q8_input_host);
    const weight_host = try allocator.alloc(u8, weight_bytes);
    defer allocator.free(weight_host);
    fillQ4_0Weights(weight_host);

    const act_dequant = try allocator.alloc(f32, in_dim);
    defer allocator.free(act_dequant);
    try quant_codec.dequantizeToFloat32(.{ .known = .Q8_1 }, q8_input_host, act_dequant);
    const reference_host = try allocator.alloc(f32, out_dim);
    defer allocator.free(reference_host);
    try quant_kernel_compiler.referenceMatmulNoBias(allocator, .q4_0, weight_host, act_dequant, 1, in_dim, out_dim, reference_host);

    var q8_input = try cuda_buffer.DeviceBuffer.alloc(ctx, q8_input_host.len);
    defer q8_input.free(ctx);
    var weight = try cuda_buffer.DeviceBuffer.alloc(ctx, weight_bytes);
    defer weight.free(ctx);
    var output = try cuda_buffer.DeviceBuffer.alloc(ctx, out_dim * @sizeOf(f32));
    defer output.free(ctx);
    try q8_input.copyFromHost(ctx, q8_input_host);
    try weight.copyFromHost(ctx, weight_host);
    try ctx.synchronize();

    const baseline_ns = try repeatCudaStep(allocator, ctx, cfg, launchQ4_0DownQ8, .{ module.linear_q4_0_q8_1_e4b_down, ctx, output, q8_input, weight, out_dim });
    const baseline_host = try allocator.alloc(f32, out_dim);
    defer allocator.free(baseline_host);
    try output.copyToHost(ctx, std.mem.sliceAsBytes(baseline_host));
    try ctx.synchronize();
    const baseline_cpu_max_abs_diff = try maxAbsDiffFinite(baseline_host, reference_host);
    if (baseline_cpu_max_abs_diff > correctness_tolerance_abs) return error.HandwrittenBaselineMismatch;

    try poisonDeviceOutput(allocator, ctx, output, out_dim);

    const generated_ns = try repeatCudaStep(allocator, ctx, cfg, launchQ4_0DownQ8, .{ generated.function, ctx, output, q8_input, weight, out_dim });
    const generated_host = try allocator.alloc(f32, out_dim);
    defer allocator.free(generated_host);
    try output.copyToHost(ctx, std.mem.sliceAsBytes(generated_host));
    try ctx.synchronize();
    const cpu_max_abs_diff = try maxAbsDiffFinite(generated_host, reference_host);
    if (cpu_max_abs_diff > correctness_tolerance_abs) return error.GeneratedCandidateMismatch;
    const baseline_max_abs_diff = try maxAbsDiffFinite(generated_host, baseline_host);

    const candidate_speedup = speedup(baseline_ns, generated_ns);
    print("shape=E4B FFN down q8_1 rows=1 in={d} out={d} baseline_ns={d} generated_ns={d} speedup={d:.6} baseline_cpu_max_abs_diff={d:.6} generated_cpu_max_abs_diff={d:.6} generated_baseline_max_abs_diff={d:.6}\n", .{
        in_dim,
        out_dim,
        baseline_ns,
        generated_ns,
        candidate_speedup,
        baseline_cpu_max_abs_diff,
        cpu_max_abs_diff,
        baseline_max_abs_diff,
    });
    print("q4_0 down-q8 summary: speedup={d:.6}\n", .{candidate_speedup});
    if (cfg.quant_compiler_evidence_out_path) |evidence_path| {
        var shape_details = std.ArrayListUnmanaged(u8).empty;
        defer shape_details.deinit(allocator);
        try appendJsonFmt(allocator, &shape_details,
            \\{{"label":"E4B FFN down q8_1","rows":1,"in_dim":{d},"out_dim":{d},"baseline_ns":{d},"generated_ns":{d},"speedup":{d:.6},"baseline_cpu_max_abs_diff":{d:.6},"generated_cpu_max_abs_diff":{d:.6},"generated_baseline_max_abs_diff":{d:.6}}}
        , .{
            in_dim,
            out_dim,
            baseline_ns,
            generated_ns,
            candidate_speedup,
            baseline_cpu_max_abs_diff,
            cpu_max_abs_diff,
            baseline_max_abs_diff,
        });
        try writeQuantCompilerQ4_0Evidence(allocator, io, evidence_path, bench, cfg, candidate_speedup, candidate_speedup, shape_details.items);
        print("wrote quant compiler evidence: {s}\n", .{evidence_path});
    }
}

fn launchQ4_0DownQ8(
    function: cuda_driver.CUfunction,
    ctx: *cuda_context.CudaContext,
    output: cuda_buffer.DeviceBuffer,
    q8_input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    out_dim: usize,
) cuda_driver.Error!void {
    var dst_ptr = output.ptr;
    var input_ptr = q8_input.ptr;
    var weight_ptr = weight.ptr;
    var params = [_]?*anyopaque{
        @ptrCast(&dst_ptr),
        @ptrCast(&input_ptr),
        @ptrCast(&weight_ptr),
    };
    try launchBlocks(ctx, function, out_dim / 4, 256, &params);
}

fn launchQ4_0PairQ8(
    function: cuda_driver.CUfunction,
    ctx: *cuda_context.CudaContext,
    dst_q8: cuda_buffer.DeviceBuffer,
    q8_input: cuda_buffer.DeviceBuffer,
    weight_gate: cuda_buffer.DeviceBuffer,
    weight_up: cuda_buffer.DeviceBuffer,
    out_blocks: usize,
) cuda_driver.Error!void {
    var dst_ptr = dst_q8.ptr;
    var input_ptr = q8_input.ptr;
    var weight_gate_ptr = weight_gate.ptr;
    var weight_up_ptr = weight_up.ptr;
    var activation_u32: u32 = quant_compiler_q4_0_pair_q8_activation;
    var params = [_]?*anyopaque{
        @ptrCast(&dst_ptr),
        @ptrCast(&input_ptr),
        @ptrCast(&weight_gate_ptr),
        @ptrCast(&weight_up_ptr),
        @ptrCast(&activation_u32),
    };
    try launchBlocks(ctx, function, out_blocks, 640, &params);
}

fn poisonDeviceBytes(
    allocator: std.mem.Allocator,
    ctx: *cuda_context.CudaContext,
    buffer: cuda_buffer.DeviceBuffer,
    byte_count: usize,
) !void {
    const poison = try allocator.alloc(u8, byte_count);
    defer allocator.free(poison);
    @memset(poison, 0x7f);
    try buffer.copyFromHost(ctx, poison);
    try ctx.synchronize();
}

fn launchQ4_0PairBaseline(
    module: *BenchModule,
    ctx: *cuda_context.CudaContext,
    output_a: cuda_buffer.DeviceBuffer,
    output_b: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight_a: cuda_buffer.DeviceBuffer,
    weight_b: cuda_buffer.DeviceBuffer,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    try validateQ4_0Buffers(output_a, input, weight_a, 1, in_dim, out_dim);
    try validateQ4_0Buffers(output_b, input, weight_b, 1, in_dim, out_dim);
    var dst_a_ptr = output_a.ptr;
    var dst_b_ptr = output_b.ptr;
    var input_ptr = input.ptr;
    var weight_a_ptr = weight_a.ptr;
    var weight_b_ptr = weight_b.ptr;
    var rows_u32: u32 = 1;
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&dst_a_ptr),
        @ptrCast(&dst_b_ptr),
        @ptrCast(&input_ptr),
        @ptrCast(&weight_a_ptr),
        @ptrCast(&weight_b_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    const total_tiles = try checkedMul((out_dim + 3) / 4, 2);
    try launchBlocks(ctx, module.linear_q4_0_pair_nobias_f32_tile4_w4, total_tiles, 128, &params);
}

fn launchGeneratedQ4_0Pair(
    module: *GeneratedCandidateModule,
    ctx: *cuda_context.CudaContext,
    output_a: cuda_buffer.DeviceBuffer,
    output_b: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight_a: cuda_buffer.DeviceBuffer,
    weight_b: cuda_buffer.DeviceBuffer,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    try validateQ4_0Buffers(output_a, input, weight_a, 1, in_dim, out_dim);
    try validateQ4_0Buffers(output_b, input, weight_b, 1, in_dim, out_dim);
    var input_ptr = input.ptr;
    var weight_a_ptr = weight_a.ptr;
    var weight_b_ptr = weight_b.ptr;
    var dst_a_ptr = output_a.ptr;
    var dst_b_ptr = output_b.ptr;
    var rows_u32: u32 = 1;
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&input_ptr),
        @ptrCast(&weight_a_ptr),
        @ptrCast(&weight_b_ptr),
        @ptrCast(&dst_a_ptr),
        @ptrCast(&dst_b_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    try launch2d(ctx, module.function, (out_dim + 3) / 4, 1, 256, &params);
}

fn writeQuantCompilerQ4_0Evidence(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    bench: quant_kernel_compiler.BenchmarkCase,
    cfg: Config,
    geomean_speedup: f64,
    worst_speedup: f64,
    shape_details_json: []const u8,
) !void {
    const benchmark_passed = std.math.isFinite(geomean_speedup) and geomean_speedup >= bench.minimum_speedup;
    var out = std.ArrayListUnmanaged(u8).empty;
    defer out.deinit(allocator);
    try appendJsonFmt(allocator, &out,
        \\{{
        \\"schema":"antfly.quant_kernel_q4_0_benchmark_evidence.v1",
        \\"kernel_id":{f},
        \\"generated_source_path":{f},
        \\"generated_source_fingerprint":{d},
        \\"generated_ptx_path":{f},
        \\"generated_ptx_command":{f},
        \\"benchmark_command":{f},
        \\"correctness_evidence_path":{f},
        \\"benchmark_evidence_path":{f},
        \\"benchmark_mode":"sequential",
        \\"repeat_runs":{d},
        \\"timing_aggregation":"median",
        \\"production_enabled":{},
        \\"correctness_passed":true,
        \\"benchmark_passed":{},
        \\"promotion_ready":{},
        \\"measured_speedup":{d:.6},
        \\"worst_shape_speedup":{d:.6},
        \\"minimum_speedup":{d:.6},
        \\"correctness_tolerance_abs":{d:.6},
        \\"baseline_kernel":{f},
        \\"warmup_iters":{d},
        \\"measure_iters":{d},
        \\"shapes":[{s}]
        \\}}
        \\
    , .{
        std.json.fmt(bench.generated_kernel_id, .{}),
        std.json.fmt(bench.generated_source_path, .{}),
        bench.generated_source_fingerprint,
        std.json.fmt(bench.generated_ptx_path, .{}),
        std.json.fmt(bench.generated_ptx_command, .{}),
        std.json.fmt(bench.benchmark_command, .{}),
        std.json.fmt(bench.correctness_evidence_path, .{}),
        std.json.fmt(bench.benchmark_evidence_path, .{}),
        cfg.quant_compiler_repeat_runs,
        bench.production_enabled,
        benchmark_passed,
        benchmark_passed,
        geomean_speedup,
        worst_speedup,
        bench.minimum_speedup,
        bench.correctness_tolerance_abs,
        std.json.fmt(bench.handwritten_baseline, .{}),
        cfg.warmup_iters,
        cfg.measure_iters,
        shape_details_json,
    });
    try writeFileCreatingParent(io, path, out.items);
}

fn launchQ4_0Baseline(
    function: cuda_driver.CUfunction,
    ctx: *cuda_context.CudaContext,
    kind: QuantCompilerQ4_0Kind,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    try validateQ4_0Buffers(output, input, weight, rows, in_dim, out_dim);
    var dst_ptr = output.ptr;
    var input_ptr = input.ptr;
    var weight_ptr = weight.ptr;
    var rows_u32 = try toU32(rows);
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&dst_ptr),
        @ptrCast(&input_ptr),
        @ptrCast(&weight_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    switch (kind) {
        .mmv => try launch2d(ctx, function, (out_dim + 3) / 4, rows, 256, &params),
        .mm => try launch1d(ctx, function, try checkedMul(rows, out_dim), &params),
    }
}

fn launchGeneratedQ4_0(
    module: *GeneratedCandidateModule,
    ctx: *cuda_context.CudaContext,
    kind: QuantCompilerQ4_0Kind,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    try validateQ4_0Buffers(output, input, weight, rows, in_dim, out_dim);
    var dst_ptr = output.ptr;
    var input_ptr = input.ptr;
    var weight_ptr = weight.ptr;
    var rows_u32 = try toU32(rows);
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&input_ptr),
        @ptrCast(&weight_ptr),
        @ptrCast(&dst_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    switch (kind) {
        .mmv => try launch2d(ctx, module.function, (out_dim + 3) / 4, 1, 256, &params),
        .mm => try launch2d(ctx, module.function, (out_dim + 3) / 4, (rows + 7) / 8, 256, &params),
    }
}

fn validateQ4_0Buffers(
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    if (in_dim % q4_0_values_per_block != 0) return error.InvalidCudaState;
    try checkF32Bytes(output, try checkedMul(rows, out_dim));
    try checkF32Bytes(input, try checkedMul(rows, in_dim));
    const row_blocks = in_dim / q4_0_values_per_block;
    try checkRawBytes(weight, try checkedMul(try checkedMul(out_dim, row_blocks), q4_0_block_bytes));
}

fn timeCudaStep(
    ctx: *cuda_context.CudaContext,
    cfg: Config,
    comptime step: anytype,
    args: anytype,
) !u64 {
    for (0..cfg.warmup_iters) |_| {
        try @call(.auto, step, args);
        try ctx.synchronize();
    }

    var total_ns: u64 = 0;
    for (0..cfg.measure_iters) |_| {
        const started = nowNs();
        try @call(.auto, step, args);
        try ctx.synchronize();
        total_ns += nowNs() - started;
    }
    return total_ns / cfg.measure_iters;
}

fn repeatCudaStep(
    allocator: std.mem.Allocator,
    ctx: *cuda_context.CudaContext,
    cfg: Config,
    comptime step: anytype,
    args: anytype,
) !u64 {
    if (cfg.quant_compiler_repeat_runs == 1) return timeCudaStep(ctx, cfg, step, args);
    const runs = try allocator.alloc(u64, cfg.quant_compiler_repeat_runs);
    defer allocator.free(runs);
    for (runs) |*run| {
        run.* = try timeCudaStep(ctx, cfg, step, args);
    }
    std.mem.sort(u64, runs, {}, std.sort.asc(u64));
    return runs[runs.len / 2];
}

fn benchTripleShape(
    allocator: std.mem.Allocator,
    ctx: *cuda_context.CudaContext,
    module: *BenchModule,
    cfg: Config,
    shape: Shape,
) !void {
    const input_count = try std.math.mul(usize, shape.rows, shape.in_dim);
    const output_count = try std.math.mul(usize, shape.rows, shape.out_dim);
    const row_blocks = shape.in_dim / q4_k_values_per_block;
    const weight_bytes = try std.math.mul(usize, try std.math.mul(usize, shape.out_dim, row_blocks), q4_k_block_bytes);

    const input_host = try allocator.alloc(f32, input_count);
    defer allocator.free(input_host);
    const bias_host = try allocator.alloc(f32, shape.out_dim);
    defer allocator.free(bias_host);
    const weight_host = try allocator.alloc(u8, weight_bytes);
    defer allocator.free(weight_host);
    fillInput(input_host);
    fillBias(bias_host);
    fillQ4KWeights(weight_host);

    var input = try cuda_buffer.DeviceBuffer.alloc(ctx, input_count * @sizeOf(f32));
    defer input.free(ctx);
    var weight = try cuda_buffer.DeviceBuffer.alloc(ctx, weight_host.len);
    defer weight.free(ctx);
    var bias = try cuda_buffer.DeviceBuffer.alloc(ctx, bias_host.len * @sizeOf(f32));
    defer bias.free(ctx);
    var output_a = try cuda_buffer.DeviceBuffer.alloc(ctx, output_count * @sizeOf(f32));
    defer output_a.free(ctx);
    var output_b = try cuda_buffer.DeviceBuffer.alloc(ctx, output_count * @sizeOf(f32));
    defer output_b.free(ctx);
    var output_c = try cuda_buffer.DeviceBuffer.alloc(ctx, output_count * @sizeOf(f32));
    defer output_c.free(ctx);

    try input.copyFromHost(ctx, std.mem.sliceAsBytes(input_host));
    try weight.copyFromHost(ctx, weight_host);
    try bias.copyFromHost(ctx, std.mem.sliceAsBytes(bias_host));
    try ctx.synchronize();

    const scalar_ns = try timeCudaStep(ctx, cfg, launchQ4KTripleBias, .{ module, ctx, output_a, output_b, output_c, input, weight, bias, shape.rows, shape.in_dim, shape.out_dim });
    const tiled_ns = try timeCudaStep(ctx, cfg, launchQ4KTripleBiasTiled, .{ module, ctx, output_a, output_b, output_c, input, weight, bias, shape.rows, shape.in_dim, shape.out_dim });

    var sample: [1]f32 = undefined;
    try output_a.copyToHost(ctx, std.mem.sliceAsBytes(&sample));
    try ctx.synchronize();
    print("{s:<24} {d:>8} {d:>8} {d:>8} {d:>14} {d:>14} {d:>12.4}\n", .{
        shape.label,
        shape.rows,
        shape.in_dim,
        shape.out_dim,
        scalar_ns,
        tiled_ns,
        sample[0],
    });
}

const Gemma4Q8BenchResult = struct {
    scalar_ns: u64,
    tile4_ns: u64,
    checksum: f32,
};

const Gemma4Q4BenchResult = struct {
    scalar_ns: u64,
    tile4_ns: u64,
    tile8_ns: u64,
    checksum: f32,
};

fn speedup(base_ns: u64, candidate_ns: u64) f64 {
    if (base_ns == 0 or candidate_ns == 0) return 0;
    return @as(f64, @floatFromInt(base_ns)) / @as(f64, @floatFromInt(candidate_ns));
}

fn validateQuantCompilerEvidencePtxPath(bench: quant_kernel_compiler.BenchmarkCase, ptx_path: []const u8) !void {
    if (!std.mem.eql(u8, ptx_path, bench.generated_ptx_path)) return error.GeneratedPtxPathMismatch;
}

fn validateQuantCompilerEvidenceRequest(bench: quant_kernel_compiler.BenchmarkCase, ptx_path: []const u8, evidence_path: []const u8) !void {
    try validateQuantCompilerEvidencePtxPath(bench, ptx_path);
    if (!std.mem.eql(u8, evidence_path, quant_kernel_compiler.first_lazy_benchmark_evidence_path)) return error.QuantCompilerEvidencePathMismatch;
}

fn poisonDeviceOutput(
    allocator: std.mem.Allocator,
    ctx: *cuda_context.CudaContext,
    buffer: cuda_buffer.DeviceBuffer,
    count: usize,
) !void {
    const poison = try allocator.alloc(f32, count);
    defer allocator.free(poison);
    @memset(poison, std.math.nan(f32));
    try buffer.copyFromHost(ctx, std.mem.sliceAsBytes(poison));
    try ctx.synchronize();
}

fn maxAbsDiffFinite(a: []const f32, b: []const f32) !f32 {
    if (a.len != b.len) return error.InvalidArgument;
    var max_abs_diff: f32 = 0.0;
    for (a, b) |left, right| {
        if (!std.math.isFinite(left) or !std.math.isFinite(right)) return error.NonFiniteBenchmarkOutput;
        max_abs_diff = @max(max_abs_diff, @abs(left - right));
    }
    return max_abs_diff;
}

fn writeQuantCompilerEvidence(
    allocator: std.mem.Allocator,
    io: std.Io,
    path: []const u8,
    bench: quant_kernel_compiler.BenchmarkCase,
    shape: Shape,
    ptx_path: []const u8,
    cfg: Config,
    baseline_ns: u64,
    generated_ns: u64,
    baseline_checksum: f32,
    generated_checksum: f32,
    baseline_cpu_max_abs_diff: f32,
    baseline_max_abs_diff: f32,
    cpu_max_abs_diff: f32,
) !void {
    const benchmark_command = try std.fmt.allocPrint(allocator, "zig-out/bin/antfly-inference bench-cuda --warmup-iters {d} --measure-iters {d} --quant-compiler-lazy-target {s} {s} --quant-compiler-repeat-runs {d} --quant-compiler-evidence-out {s}", .{
        cfg.warmup_iters,
        cfg.measure_iters,
        bench.generated_ptx_arg,
        ptx_path,
        cfg.quant_compiler_repeat_runs,
        path,
    });
    defer allocator.free(benchmark_command);

    const measured_speedup = speedup(baseline_ns, generated_ns);
    const benchmark_passed = std.math.isFinite(measured_speedup) and measured_speedup >= bench.minimum_speedup;
    const promotion_blocker = if (benchmark_passed) "dev_only_candidate" else "generated_slower_than_handwritten";
    var out = std.ArrayListUnmanaged(u8).empty;
    defer out.deinit(allocator);
    try appendJsonFmt(allocator, &out,
        \\{{
        \\"schema":"antfly.quant_kernel_benchmark_evidence.v1",
        \\"kernel_id":{f},
        \\"generated_source_path":{f},
        \\"generated_source_fingerprint":{d},
        \\"generated_ptx_path":{f},
        \\"generated_ptx_command":{f},
        \\"benchmark_command":{f},
        \\"correctness_evidence_path":{f},
        \\"benchmark_evidence_path":{f},
        \\"benchmark_mode":"sequential",
        \\"repeat_runs":{d},
        \\"timing_aggregation":{f},
        \\"production_enabled":false,
        \\"correctness_passed":true,
        \\"benchmark_passed":{s},
        \\"promotion_ready":false,
        \\"promotion_blocker":{f},
        \\"measured_speedup":{d:.6},
        \\"minimum_speedup":{d:.6},
        \\"correctness_tolerance_abs":{d:.6},
        \\"baseline_kernel":{f},
        \\"baseline_ns":{d},
        \\"generated_ns":{d},
        \\"baseline_checksum":{d:.6},
        \\"generated_checksum":{d:.6},
        \\"baseline_cpu_max_abs_diff":{d:.6},
        \\"baseline_max_abs_diff":{d:.6},
        \\"cpu_max_abs_diff":{d:.6},
        \\"warmup_iters":{d},
        \\"measure_iters":{d},
        \\"shape":{{"label":{f},"rows":{d},"in_dim":{d},"out_dim":{d}}}
        \\}}
        \\
    , .{
        std.json.fmt(bench.generated_kernel_id, .{}),
        std.json.fmt(bench.generated_source_path, .{}),
        bench.generated_source_fingerprint,
        std.json.fmt(ptx_path, .{}),
        std.json.fmt(bench.generated_ptx_command, .{}),
        std.json.fmt(benchmark_command, .{}),
        std.json.fmt(path, .{}),
        std.json.fmt(path, .{}),
        cfg.quant_compiler_repeat_runs,
        std.json.fmt(if (cfg.quant_compiler_repeat_runs == 1) "single" else "median", .{}),
        if (benchmark_passed) "true" else "false",
        std.json.fmt(promotion_blocker, .{}),
        measured_speedup,
        bench.minimum_speedup,
        bench.correctness_tolerance_abs,
        std.json.fmt(bench.handwritten_baseline, .{}),
        baseline_ns,
        generated_ns,
        baseline_checksum,
        generated_checksum,
        baseline_cpu_max_abs_diff,
        baseline_max_abs_diff,
        cpu_max_abs_diff,
        cfg.warmup_iters,
        cfg.measure_iters,
        std.json.fmt(shape.label, .{}),
        shape.rows,
        shape.in_dim,
        shape.out_dim,
    });
    try checkQuantCompilerEvidenceJson(allocator, out.items, false);
    try writeFileCreatingParent(io, path, out.items);
}

fn checkQuantCompilerEvidenceJson(allocator: std.mem.Allocator, bytes: []const u8, require_promotion_ready: bool) !void {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();
    const root = parsed.value;
    if (root != .object) return error.InvalidQuantCompilerEvidence;
    const bench = quant_kernel_compiler.first_lazy_benchmark;

    const schema = jsonString(root.object.get("schema")) orelse return error.InvalidQuantCompilerEvidence;
    if (!std.mem.eql(u8, schema, "antfly.quant_kernel_benchmark_evidence.v1")) return error.InvalidQuantCompilerEvidence;
    const benchmark_mode = jsonString(root.object.get("benchmark_mode")) orelse return error.InvalidQuantCompilerEvidence;
    if (!std.mem.eql(u8, benchmark_mode, "sequential")) return error.InvalidQuantCompilerEvidence;
    const repeat_runs = jsonUsize(root.object.get("repeat_runs")) orelse return error.InvalidQuantCompilerEvidence;
    if (repeat_runs == 0 or repeat_runs > 31) return error.InvalidQuantCompilerEvidence;
    const timing_aggregation = jsonString(root.object.get("timing_aggregation")) orelse return error.InvalidQuantCompilerEvidence;
    const expected_timing_aggregation = if (repeat_runs == 1) "single" else "median";
    if (!std.mem.eql(u8, timing_aggregation, expected_timing_aggregation)) return error.InvalidQuantCompilerEvidence;
    const production_enabled = jsonBoolValue(root.object.get("production_enabled")) orelse return error.InvalidQuantCompilerEvidence;
    if (require_promotion_ready) {
        if (!production_enabled) return error.QuantCompilerEvidencePromotionNotReady;
    } else if (production_enabled) {
        return error.InvalidQuantCompilerEvidence;
    }
    const correctness_passed = jsonBoolValue(root.object.get("correctness_passed")) orelse return error.InvalidQuantCompilerEvidence;
    if (!correctness_passed) return error.InvalidQuantCompilerEvidence;
    const promotion_ready = jsonBoolValue(root.object.get("promotion_ready")) orelse return error.InvalidQuantCompilerEvidence;
    if (require_promotion_ready) {
        if (!promotion_ready) return error.QuantCompilerEvidencePromotionNotReady;
        if (repeat_runs < quant_compiler_evidence_repeat_runs) return error.QuantCompilerEvidencePromotionNotReady;
    } else if (promotion_ready) {
        return error.InvalidQuantCompilerEvidence;
    }

    const baseline_ns = jsonU64(root.object.get("baseline_ns")) orelse return error.InvalidQuantCompilerEvidence;
    const generated_ns = jsonU64(root.object.get("generated_ns")) orelse return error.InvalidQuantCompilerEvidence;
    if (baseline_ns == 0 or generated_ns == 0) return error.InvalidQuantCompilerEvidence;
    const measured_speedup = jsonF64(root.object.get("measured_speedup")) orelse return error.InvalidQuantCompilerEvidence;
    if (!approximately(measured_speedup, speedup(baseline_ns, generated_ns), 0.000001)) return error.InvalidQuantCompilerEvidence;

    const minimum_speedup = jsonF64(root.object.get("minimum_speedup")) orelse return error.InvalidQuantCompilerEvidence;
    if (!std.math.isFinite(minimum_speedup) or minimum_speedup < 1.0) return error.InvalidQuantCompilerEvidence;
    const benchmark_passed = jsonBoolValue(root.object.get("benchmark_passed")) orelse return error.InvalidQuantCompilerEvidence;
    if (benchmark_passed != (measured_speedup >= minimum_speedup)) return error.InvalidQuantCompilerEvidence;
    const blocker = jsonString(root.object.get("promotion_blocker")) orelse return error.InvalidQuantCompilerEvidence;
    if (promotion_ready) {
        if (!benchmark_passed) return error.InvalidQuantCompilerEvidence;
        if (blocker.len != 0) return error.InvalidQuantCompilerEvidence;
    } else {
        const expected_blocker = if (benchmark_passed) "dev_only_candidate" else "generated_slower_than_handwritten";
        if (!std.mem.eql(u8, blocker, expected_blocker)) return error.InvalidQuantCompilerEvidence;
    }

    const tolerance = jsonF64(root.object.get("correctness_tolerance_abs")) orelse return error.InvalidQuantCompilerEvidence;
    if (!std.math.isFinite(tolerance) or tolerance <= 0.0) return error.InvalidQuantCompilerEvidence;
    if (!approximately(tolerance, @as(f64, @floatCast(bench.correctness_tolerance_abs)), 0.000001)) return error.InvalidQuantCompilerEvidence;
    const baseline_cpu_max_abs_diff = jsonF64(root.object.get("baseline_cpu_max_abs_diff")) orelse return error.InvalidQuantCompilerEvidence;
    const baseline_max_abs_diff = jsonF64(root.object.get("baseline_max_abs_diff")) orelse return error.InvalidQuantCompilerEvidence;
    const cpu_max_abs_diff = jsonF64(root.object.get("cpu_max_abs_diff")) orelse return error.InvalidQuantCompilerEvidence;
    if (!std.math.isFinite(baseline_cpu_max_abs_diff) or baseline_cpu_max_abs_diff > tolerance) return error.InvalidQuantCompilerEvidence;
    if (!std.math.isFinite(baseline_max_abs_diff) or baseline_max_abs_diff > tolerance) return error.InvalidQuantCompilerEvidence;
    if (!std.math.isFinite(cpu_max_abs_diff) or cpu_max_abs_diff > tolerance) return error.InvalidQuantCompilerEvidence;

    const kernel_id = jsonString(root.object.get("kernel_id")) orelse return error.InvalidQuantCompilerEvidence;
    if (!std.mem.eql(u8, kernel_id, bench.generated_kernel_id)) return error.InvalidQuantCompilerEvidence;
    const generated_source_path = jsonString(root.object.get("generated_source_path")) orelse return error.InvalidQuantCompilerEvidence;
    if (!std.mem.eql(u8, generated_source_path, bench.generated_source_path)) return error.InvalidQuantCompilerEvidence;
    const generated_source_fingerprint = jsonU64(root.object.get("generated_source_fingerprint")) orelse return error.InvalidQuantCompilerEvidence;
    if (generated_source_fingerprint != bench.generated_source_fingerprint) return error.InvalidQuantCompilerEvidence;
    const generated_ptx_path = jsonString(root.object.get("generated_ptx_path")) orelse return error.InvalidQuantCompilerEvidence;
    if (!std.mem.eql(u8, generated_ptx_path, bench.generated_ptx_path)) return error.InvalidQuantCompilerEvidence;
    const generated_ptx_command = jsonString(root.object.get("generated_ptx_command")) orelse return error.InvalidQuantCompilerEvidence;
    if (!std.mem.eql(u8, generated_ptx_command, bench.generated_ptx_command)) return error.InvalidQuantCompilerEvidence;
    const baseline_kernel = jsonString(root.object.get("baseline_kernel")) orelse return error.InvalidQuantCompilerEvidence;
    if (!std.mem.eql(u8, baseline_kernel, bench.handwritten_baseline)) return error.InvalidQuantCompilerEvidence;
    inline for (&.{ "kernel_id", "generated_source_path", "generated_ptx_path", "generated_ptx_command", "benchmark_command", "correctness_evidence_path", "benchmark_evidence_path", "baseline_kernel" }) |field| {
        const text = jsonString(root.object.get(field)) orelse return error.InvalidQuantCompilerEvidence;
        if (text.len == 0) return error.InvalidQuantCompilerEvidence;
    }
    const correctness_path = jsonString(root.object.get("correctness_evidence_path")) orelse return error.InvalidQuantCompilerEvidence;
    const benchmark_path = jsonString(root.object.get("benchmark_evidence_path")) orelse return error.InvalidQuantCompilerEvidence;
    if (!std.mem.eql(u8, correctness_path, benchmark_path)) return error.InvalidQuantCompilerEvidence;
    const benchmark_command = jsonString(root.object.get("benchmark_command")) orelse return error.InvalidQuantCompilerEvidence;
    const expected_benchmark_command = try std.fmt.allocPrint(
        allocator,
        "zig-out/bin/antfly-inference bench-cuda --warmup-iters 5 --measure-iters 50 --quant-compiler-lazy-target {s} {s} --quant-compiler-repeat-runs {d} --quant-compiler-evidence-out {s}",
        .{ bench.generated_ptx_arg, bench.generated_ptx_path, repeat_runs, benchmark_path },
    );
    defer allocator.free(expected_benchmark_command);
    if (!std.mem.eql(u8, benchmark_command, expected_benchmark_command)) return error.InvalidQuantCompilerEvidence;

    if ((jsonUsize(root.object.get("warmup_iters")) orelse 0) != 5) return error.InvalidQuantCompilerEvidence;
    if ((jsonUsize(root.object.get("measure_iters")) orelse 0) != 50) return error.InvalidQuantCompilerEvidence;
    if (repeat_runs != quant_compiler_evidence_repeat_runs) return error.InvalidQuantCompilerEvidence;
    const shape_value = root.object.get("shape") orelse return error.InvalidQuantCompilerEvidence;
    if (shape_value != .object) return error.InvalidQuantCompilerEvidence;
    const shape_label = jsonString(shape_value.object.get("label")) orelse return error.InvalidQuantCompilerEvidence;
    if (!std.mem.eql(u8, shape_label, quant_compiler_lazy_shape.label)) return error.InvalidQuantCompilerEvidence;
    if ((jsonUsize(shape_value.object.get("rows")) orelse 0) != quant_compiler_lazy_shape.rows) return error.InvalidQuantCompilerEvidence;
    if ((jsonUsize(shape_value.object.get("in_dim")) orelse 0) != quant_compiler_lazy_shape.in_dim) return error.InvalidQuantCompilerEvidence;
    if ((jsonUsize(shape_value.object.get("out_dim")) orelse 0) != quant_compiler_lazy_shape.out_dim) return error.InvalidQuantCompilerEvidence;
}

fn checkQuantCompilerEvidenceFile(allocator: std.mem.Allocator, io: std.Io, path: []const u8, require_promotion_ready: bool) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, allocator, .limited(1024 * 1024));
    defer allocator.free(bytes);
    try checkQuantCompilerEvidenceJson(allocator, bytes, require_promotion_ready);
    try checkQuantCompilerEvidenceJsonPath(allocator, bytes, path);
}

fn checkQuantCompilerEvidenceJsonPath(allocator: std.mem.Allocator, bytes: []const u8, path: []const u8) !void {
    var parsed = try std.json.parseFromSlice(std.json.Value, allocator, bytes, .{});
    defer parsed.deinit();
    const root = parsed.value;
    if (root != .object) return error.InvalidQuantCompilerEvidence;
    const correctness_path = jsonString(root.object.get("correctness_evidence_path")) orelse return error.InvalidQuantCompilerEvidence;
    const benchmark_path = jsonString(root.object.get("benchmark_evidence_path")) orelse return error.InvalidQuantCompilerEvidence;
    if (!std.mem.eql(u8, correctness_path, path)) return error.InvalidQuantCompilerEvidence;
    if (!std.mem.eql(u8, benchmark_path, path)) return error.InvalidQuantCompilerEvidence;
}

fn writeFileCreatingParent(io: std.Io, path: []const u8, data: []const u8) !void {
    if (std.fs.path.dirname(path)) |parent| {
        if (parent.len > 0) try compat.cwd().createDirPath(io, parent);
    }
    if (std.fs.path.isAbsolute(path)) {
        var file = try std.Io.Dir.createFileAbsolute(io, path, .{ .truncate = true });
        defer file.close(io);
        try file.writeStreamingAll(io, data);
    } else {
        try compat.cwd().writeFile(io, .{ .sub_path = path, .data = data });
    }
}

fn approximately(actual: f64, expected: f64, tolerance: f64) bool {
    return std.math.isFinite(actual) and std.math.isFinite(expected) and @abs(actual - expected) <= tolerance;
}

fn jsonString(value: ?std.json.Value) ?[]const u8 {
    const actual = value orelse return null;
    return switch (actual) {
        .string => |text| text,
        else => null,
    };
}

fn jsonBoolValue(value: ?std.json.Value) ?bool {
    const actual = value orelse return null;
    return switch (actual) {
        .bool => |val| val,
        else => null,
    };
}

fn jsonUsize(value: ?std.json.Value) ?usize {
    const actual = value orelse return null;
    return switch (actual) {
        .integer => |val| if (val >= 0) @intCast(val) else null,
        else => null,
    };
}

fn jsonU64(value: ?std.json.Value) ?u64 {
    const actual = value orelse return null;
    return switch (actual) {
        .integer => |val| if (val >= 0) @intCast(val) else null,
        .number_string => |text| std.fmt.parseUnsigned(u64, text, 10) catch null,
        else => null,
    };
}

fn jsonF64(value: ?std.json.Value) ?f64 {
    const actual = value orelse return null;
    return switch (actual) {
        .float => |val| val,
        .integer => |val| @floatFromInt(val),
        .number_string => |text| std.fmt.parseFloat(f64, text) catch null,
        else => null,
    };
}

test "cuda microbench finite diff guards reference comparisons" {
    try std.testing.expectEqual(@as(f32, 0.25), try maxAbsDiffFinite(&.{ 1.0, 2.0 }, &.{ 1.25, 1.75 }));
    try std.testing.expectError(error.InvalidArgument, maxAbsDiffFinite(&.{1.0}, &.{ 1.0, 2.0 }));
    try std.testing.expectError(error.NonFiniteBenchmarkOutput, maxAbsDiffFinite(&.{std.math.nan(f32)}, &.{0.0}));
}

test "cuda microbench promotion evidence requires manifest ptx path" {
    const bench = quant_kernel_compiler.first_lazy_benchmark;
    try validateQuantCompilerEvidencePtxPath(bench, bench.generated_ptx_path);
    try std.testing.expectError(error.GeneratedPtxPathMismatch, validateQuantCompilerEvidencePtxPath(bench, "/tmp/other.ptx"));
    try validateQuantCompilerEvidenceRequest(bench, bench.generated_ptx_path, quant_kernel_compiler.first_lazy_benchmark_evidence_path);
    try std.testing.expectError(error.QuantCompilerEvidencePathMismatch, validateQuantCompilerEvidenceRequest(bench, bench.generated_ptx_path, "/tmp/other-evidence.json"));
}

test "cuda microbench writes quant compiler evidence json" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const evidence_path = try std.fs.path.join(std.testing.allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "evidence", "q4.json" });
    defer std.testing.allocator.free(evidence_path);

    try writeQuantCompilerEvidence(
        std.testing.allocator,
        std.testing.io,
        evidence_path,
        quant_kernel_compiler.first_lazy_benchmark,
        quant_compiler_lazy_shape,
        quant_kernel_compiler.first_lazy_benchmark.generated_ptx_path,
        .{ .warmup_iters = 5, .measure_iters = 50, .quant_compiler_repeat_runs = quant_compiler_evidence_repeat_runs },
        100,
        80,
        1.0,
        1.0,
        0.001,
        0.001,
        0.001,
    );

    const actual = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, evidence_path, std.testing.allocator, .limited(4096));
    defer std.testing.allocator.free(actual);
    try checkQuantCompilerEvidenceJson(std.testing.allocator, actual, false);
    try checkQuantCompilerEvidenceFile(std.testing.allocator, std.testing.io, evidence_path, false);
    try std.testing.expectError(error.QuantCompilerEvidencePromotionNotReady, checkQuantCompilerEvidenceJson(std.testing.allocator, actual, true));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "\"schema\":\"antfly.quant_kernel_benchmark_evidence.v1\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "\"generated_source_fingerprint\":"));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "\"generated_ptx_command\":"));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, quant_kernel_compiler.first_lazy_benchmark.generated_ptx_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "\"benchmark_command\":"));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "--warmup-iters 5"));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "--measure-iters 50"));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "--quant-compiler-repeat-runs 3"));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "\"benchmark_mode\":\"sequential\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "\"repeat_runs\":3"));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "\"timing_aggregation\":\"median\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "\"production_enabled\":false"));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "\"correctness_passed\":true"));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "\"benchmark_passed\":true"));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "\"promotion_ready\":false"));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "\"promotion_blocker\":\"dev_only_candidate\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, "\"measured_speedup\":1.250000"));
    try std.testing.expect(std.mem.containsAtLeast(u8, actual, 1, evidence_path));

    const copied_evidence_path = try std.fs.path.join(std.testing.allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "evidence", "q4_copied.json" });
    defer std.testing.allocator.free(copied_evidence_path);
    try writeFileCreatingParent(std.testing.io, copied_evidence_path, actual);
    try std.testing.expectError(error.InvalidQuantCompilerEvidence, checkQuantCompilerEvidenceFile(std.testing.allocator, std.testing.io, copied_evidence_path, false));

    const wrong_speedup = try replaceOnce(std.testing.allocator, actual, "\"measured_speedup\":1.250000", "\"measured_speedup\":1.100000");
    defer std.testing.allocator.free(wrong_speedup);
    try std.testing.expectError(error.InvalidQuantCompilerEvidence, checkQuantCompilerEvidenceJson(std.testing.allocator, wrong_speedup, false));
    const wrong_command = try replaceOnce(std.testing.allocator, actual, "--quant-compiler-lazy-target", "--gemma4-shapes");
    defer std.testing.allocator.free(wrong_command);
    try std.testing.expectError(error.InvalidQuantCompilerEvidence, checkQuantCompilerEvidenceJson(std.testing.allocator, wrong_command, false));
    const wrong_ptx_path = try replaceOnce(std.testing.allocator, actual, quant_kernel_compiler.first_lazy_benchmark.generated_ptx_path, "/tmp/wrong.ptx");
    defer std.testing.allocator.free(wrong_ptx_path);
    try std.testing.expectError(error.InvalidQuantCompilerEvidence, checkQuantCompilerEvidenceJson(std.testing.allocator, wrong_ptx_path, false));
    const wrong_shape = try replaceOnce(std.testing.allocator, actual, "\"rows\":8", "\"rows\":7");
    defer std.testing.allocator.free(wrong_shape);
    try std.testing.expectError(error.InvalidQuantCompilerEvidence, checkQuantCompilerEvidenceJson(std.testing.allocator, wrong_shape, false));
    const loose_tolerance = try replaceOnce(std.testing.allocator, actual, "\"correctness_tolerance_abs\":0.010000", "\"correctness_tolerance_abs\":0.020000");
    defer std.testing.allocator.free(loose_tolerance);
    try std.testing.expectError(error.InvalidQuantCompilerEvidence, checkQuantCompilerEvidenceJson(std.testing.allocator, loose_tolerance, false));
    const bad_evidence_path = try std.fs.path.join(std.testing.allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "evidence", "q4_bad.json" });
    defer std.testing.allocator.free(bad_evidence_path);
    try writeFileCreatingParent(std.testing.io, bad_evidence_path, wrong_speedup);
    try std.testing.expectError(error.InvalidQuantCompilerEvidence, checkQuantCompilerEvidenceFile(std.testing.allocator, std.testing.io, bad_evidence_path, false));

    const slow_evidence_path = try std.fs.path.join(std.testing.allocator, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "evidence", "q4_slow.json" });
    defer std.testing.allocator.free(slow_evidence_path);
    try writeQuantCompilerEvidence(
        std.testing.allocator,
        std.testing.io,
        slow_evidence_path,
        quant_kernel_compiler.first_lazy_benchmark,
        quant_compiler_lazy_shape,
        quant_kernel_compiler.first_lazy_benchmark.generated_ptx_path,
        .{ .warmup_iters = 5, .measure_iters = 50, .quant_compiler_repeat_runs = quant_compiler_evidence_repeat_runs },
        100,
        125,
        1.0,
        1.0,
        0.001,
        0.001,
        0.001,
    );
    const slow_actual = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, slow_evidence_path, std.testing.allocator, .limited(4096));
    defer std.testing.allocator.free(slow_actual);
    try checkQuantCompilerEvidenceJson(std.testing.allocator, slow_actual, false);
    try std.testing.expect(std.mem.containsAtLeast(u8, slow_actual, 1, "\"benchmark_passed\":false"));
    try std.testing.expect(std.mem.containsAtLeast(u8, slow_actual, 1, "\"promotion_blocker\":\"generated_slower_than_handwritten\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, slow_actual, 1, "\"measured_speedup\":0.800000"));

    const wrong_blocker = try replaceOnce(std.testing.allocator, slow_actual, "\"promotion_blocker\":\"generated_slower_than_handwritten\"", "\"promotion_blocker\":\"dev_only_candidate\"");
    defer std.testing.allocator.free(wrong_blocker);
    try std.testing.expectError(error.InvalidQuantCompilerEvidence, checkQuantCompilerEvidenceJson(std.testing.allocator, wrong_blocker, false));
    const production_slow = try replaceOnce(std.testing.allocator, slow_actual, "\"production_enabled\":false", "\"production_enabled\":true");
    defer std.testing.allocator.free(production_slow);
    const promoted_slow_tmp = try replaceOnce(std.testing.allocator, production_slow, "\"promotion_ready\":false", "\"promotion_ready\":true");
    defer std.testing.allocator.free(promoted_slow_tmp);
    const promoted_slow = try replaceOnce(std.testing.allocator, promoted_slow_tmp, "\"promotion_blocker\":\"generated_slower_than_handwritten\"", "\"promotion_blocker\":\"\"");
    defer std.testing.allocator.free(promoted_slow);
    try std.testing.expectError(error.InvalidQuantCompilerEvidence, checkQuantCompilerEvidenceJson(std.testing.allocator, promoted_slow, true));
}

fn replaceOnce(allocator: std.mem.Allocator, input: []const u8, needle: []const u8, replacement: []const u8) ![]u8 {
    const index = std.mem.indexOf(u8, input, needle) orelse return error.InvalidArgument;
    return std.fmt.allocPrint(allocator, "{s}{s}{s}", .{ input[0..index], replacement, input[index + needle.len ..] });
}

test "cuda microbench evidence writer creates absolute parent directories" {
    const root = try std.fmt.allocPrint(std.testing.allocator, "/tmp/antfly_quant_evidence_test_{d}", .{std.posix.system.getpid()});
    defer std.testing.allocator.free(root);
    compat.cwd().deleteTree(std.testing.io, root) catch {};
    defer compat.cwd().deleteTree(std.testing.io, root) catch {};

    const evidence_path = try std.fmt.allocPrint(std.testing.allocator, "{s}/nested/q4.json", .{root});
    defer std.testing.allocator.free(evidence_path);

    try writeFileCreatingParent(std.testing.io, evidence_path, "evidence");
    const actual = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, evidence_path, std.testing.allocator, .limited(64));
    defer std.testing.allocator.free(actual);
    try std.testing.expectEqualStrings("evidence", actual);
}

fn q8CandidateName(q8: Gemma4Q8BenchResult) []const u8 {
    return if (q8.tile4_ns <= q8.scalar_ns) "tile4" else "scalar";
}

fn q8CandidateNs(q8: Gemma4Q8BenchResult) u64 {
    return if (q8.tile4_ns <= q8.scalar_ns) q8.tile4_ns else q8.scalar_ns;
}

fn q4CandidateName(q4: Gemma4Q4BenchResult) []const u8 {
    if (q4.tile4_ns <= q4.scalar_ns and q4.tile4_ns <= q4.tile8_ns) return "tile4";
    if (q4.tile8_ns <= q4.scalar_ns and q4.tile8_ns <= q4.tile4_ns) return "tile8";
    return "scalar";
}

fn q4CandidateNs(q4: Gemma4Q4BenchResult) u64 {
    if (q4.tile4_ns <= q4.scalar_ns and q4.tile4_ns <= q4.tile8_ns) return q4.tile4_ns;
    if (q4.tile8_ns <= q4.scalar_ns and q4.tile8_ns <= q4.tile4_ns) return q4.tile8_ns;
    return q4.scalar_ns;
}

fn runGemma4KernelBench(
    allocator: std.mem.Allocator,
    io: std.Io,
    ctx: *cuda_context.CudaContext,
    module: *BenchModule,
    cfg: Config,
) !void {
    var json_out = std.ArrayListUnmanaged(u8).empty;
    defer json_out.deinit(allocator);
    if (cfg.json_out_path != null) {
        try appendJsonFmt(allocator, &json_out,
            \\{{
            \\"warmup_iters":{d},
            \\"measure_iters":{d},
            \\"gemma4_shapes":[
        , .{ cfg.warmup_iters, cfg.measure_iters });
    }

    print("\nCUDA Gemma4 decode matvec microbench: synthetic rows=1 weights\n", .{});
    print("{s:<24} {s:>8} {s:>8} {s:>8} {s:>14} {s:>14} {s:>14} {s:>14} {s:>14} {s:>12} {s:>12}\n", .{
        "shape",
        "rows",
        "in",
        "out",
        "q8 scalar",
        "q8 tile4",
        "q4 scalar",
        "q4 tile4",
        "q4 tile8",
        "q8 check",
        "q4 check",
    });
    print("{s:<24} {s:>8} {s:>8} {s:>8} {s:>14} {s:>14} {s:>14} {s:>14} {s:>14} {s:>12} {s:>12}\n", .{
        "-----",
        "----",
        "--",
        "---",
        "---------",
        "--------",
        "---------",
        "--------",
        "--------",
        "--------",
        "--------",
    });

    for (gemma4_shapes, 0..) |shape, idx| {
        const q8 = try benchGemma4Q8Shape(allocator, ctx, module, cfg, shape);
        const q4 = try benchGemma4Q4Shape(allocator, ctx, module, cfg, shape);
        print("{s:<24} {d:>8} {d:>8} {d:>8} {d:>14} {d:>14} {d:>14} {d:>14} {d:>14} {d:>12.4} {d:>12.4}\n", .{
            shape.label,
            shape.rows,
            shape.in_dim,
            shape.out_dim,
            q8.scalar_ns,
            q8.tile4_ns,
            q4.scalar_ns,
            q4.tile4_ns,
            q4.tile8_ns,
            q8.checksum,
            q4.checksum,
        });
        if (cfg.json_out_path != null) {
            const q8_candidate = q8CandidateName(q8);
            const q8_candidate_ns = q8CandidateNs(q8);
            const q4_candidate = q4CandidateName(q4);
            const q4_candidate_ns = q4CandidateNs(q4);
            if (idx != 0) try json_out.appendSlice(allocator, ",");
            try appendJsonFmt(allocator, &json_out,
                \\{{
                \\"label":{f},
                \\"rows":{d},
                \\"in_dim":{d},
                \\"out_dim":{d},
                \\"q8_scalar_ns":{d},
                \\"q8_tile4_ns":{d},
                \\"q4_scalar_ns":{d},
                \\"q4_tile4_ns":{d},
                \\"q4_tile8_ns":{d},
                \\"q8_baseline_ns":{d},
                \\"q8_candidate":{f},
                \\"q8_candidate_ns":{d},
                \\"q8_candidate_speedup":{d:.6},
                \\"q4_baseline_ns":{d},
                \\"q4_candidate":{f},
                \\"q4_candidate_ns":{d},
                \\"q4_candidate_speedup":{d:.6},
                \\"q8_checksum":{d:.6},
                \\"q4_checksum":{d:.6}
                \\}}
            , .{
                std.json.fmt(shape.label, .{}),
                shape.rows,
                shape.in_dim,
                shape.out_dim,
                q8.scalar_ns,
                q8.tile4_ns,
                q4.scalar_ns,
                q4.tile4_ns,
                q4.tile8_ns,
                q8.scalar_ns,
                std.json.fmt(q8_candidate, .{}),
                q8_candidate_ns,
                speedup(q8.scalar_ns, q8_candidate_ns),
                q4.scalar_ns,
                std.json.fmt(q4_candidate, .{}),
                q4_candidate_ns,
                speedup(q4.scalar_ns, q4_candidate_ns),
                q8.checksum,
                q4.checksum,
            });
        }
    }

    if (cfg.json_out_path) |path| {
        try json_out.appendSlice(allocator,
            \\]
            \\}
            \\
        );
        try compat.cwd().writeFile(io, .{ .sub_path = path, .data = json_out.items });
    }
}

fn appendJsonFmt(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    comptime fmt: []const u8,
    args: anytype,
) !void {
    const chunk = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(chunk);
    try out.appendSlice(allocator, chunk);
}

fn benchGemma4Q8Shape(
    allocator: std.mem.Allocator,
    ctx: *cuda_context.CudaContext,
    module: *BenchModule,
    cfg: Config,
    shape: Shape,
) !Gemma4Q8BenchResult {
    if (shape.in_dim % q8_0_values_per_block != 0) return error.InvalidArgument;

    const input_count = try std.math.mul(usize, shape.rows, shape.in_dim);
    const output_count = try std.math.mul(usize, shape.rows, shape.out_dim);
    const row_blocks = shape.in_dim / q8_0_values_per_block;
    const weight_bytes = try std.math.mul(usize, try std.math.mul(usize, shape.out_dim, row_blocks), q8_0_block_bytes);

    const input_host = try allocator.alloc(f32, input_count);
    defer allocator.free(input_host);
    const weight_host = try allocator.alloc(u8, weight_bytes);
    defer allocator.free(weight_host);
    fillInput(input_host);
    fillQ8Weights(weight_host);

    var input = try cuda_buffer.DeviceBuffer.alloc(ctx, input_count * @sizeOf(f32));
    defer input.free(ctx);
    var weight = try cuda_buffer.DeviceBuffer.alloc(ctx, weight_host.len);
    defer weight.free(ctx);
    var output = try cuda_buffer.DeviceBuffer.alloc(ctx, output_count * @sizeOf(f32));
    defer output.free(ctx);

    try input.copyFromHost(ctx, std.mem.sliceAsBytes(input_host));
    try weight.copyFromHost(ctx, weight_host);
    try ctx.synchronize();

    const scalar_ns = try timeCudaStep(ctx, cfg, launchQ8, .{ module, ctx, output, input, weight, shape.rows, shape.in_dim, shape.out_dim });
    const tile4_ns = try timeCudaStep(ctx, cfg, launchQ8Tile4, .{ module, ctx, output, input, weight, shape.rows, shape.in_dim, shape.out_dim });

    var sample: [1]f32 = undefined;
    try output.copyToHost(ctx, std.mem.sliceAsBytes(&sample));
    try ctx.synchronize();
    return .{
        .scalar_ns = scalar_ns,
        .tile4_ns = tile4_ns,
        .checksum = sample[0],
    };
}

fn benchGemma4Q4Shape(
    allocator: std.mem.Allocator,
    ctx: *cuda_context.CudaContext,
    module: *BenchModule,
    cfg: Config,
    shape: Shape,
) !Gemma4Q4BenchResult {
    if (shape.in_dim % q4_k_values_per_block != 0) return error.InvalidArgument;

    const input_count = try std.math.mul(usize, shape.rows, shape.in_dim);
    const output_count = try std.math.mul(usize, shape.rows, shape.out_dim);
    const row_blocks = shape.in_dim / q4_k_values_per_block;
    const weight_bytes = try std.math.mul(usize, try std.math.mul(usize, shape.out_dim, row_blocks), q4_k_block_bytes);

    const input_host = try allocator.alloc(f32, input_count);
    defer allocator.free(input_host);
    const weight_host = try allocator.alloc(u8, weight_bytes);
    defer allocator.free(weight_host);
    fillInput(input_host);
    fillQ4KWeights(weight_host);

    var input = try cuda_buffer.DeviceBuffer.alloc(ctx, input_count * @sizeOf(f32));
    defer input.free(ctx);
    var weight = try cuda_buffer.DeviceBuffer.alloc(ctx, weight_host.len);
    defer weight.free(ctx);
    var output = try cuda_buffer.DeviceBuffer.alloc(ctx, output_count * @sizeOf(f32));
    defer output.free(ctx);

    try input.copyFromHost(ctx, std.mem.sliceAsBytes(input_host));
    try weight.copyFromHost(ctx, weight_host);
    try ctx.synchronize();

    const scalar_ns = try timeCudaStep(ctx, cfg, launchQ4K, .{ module, ctx, output, input, weight, shape.rows, shape.in_dim, shape.out_dim });
    const tile4_ns = try timeCudaStep(ctx, cfg, launchQ4KTiled, .{ module, ctx, output, input, weight, shape.rows, shape.in_dim, shape.out_dim });
    const tile8_ns = try timeCudaStep(ctx, cfg, launchQ4KTile8, .{ module, ctx, output, input, weight, shape.rows, shape.in_dim, shape.out_dim });

    var sample: [1]f32 = undefined;
    try output.copyToHost(ctx, std.mem.sliceAsBytes(&sample));
    try ctx.synchronize();
    return .{
        .scalar_ns = scalar_ns,
        .tile4_ns = tile4_ns,
        .tile8_ns = tile8_ns,
        .checksum = sample[0],
    };
}

fn launchQ8(
    module: *BenchModule,
    ctx: *cuda_context.CudaContext,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !void {
    try launchLinearQ8Raw(ctx, module.linear_q8_0_f32, output, input, weight, rows, in_dim, out_dim);
}

fn launchQ8Tile4(
    module: *BenchModule,
    ctx: *cuda_context.CudaContext,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !void {
    try launchLinearQ8Tile4(ctx, module.linear_q8_0_f32_tile4, output, input, weight, rows, in_dim, out_dim);
}

fn launchQ4K(
    module: *BenchModule,
    ctx: *cuda_context.CudaContext,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !void {
    try launchLinearQ4KRaw(ctx, module.linear_q4_k_f32, output, input, weight, rows, in_dim, out_dim);
}

fn launchQ4KBias(
    module: *BenchModule,
    ctx: *cuda_context.CudaContext,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    bias: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !void {
    try launchLinearQ4KBiasRaw(ctx, module.linear_q4_k_bias_f32, output, input, weight, bias, rows, in_dim, out_dim);
}

fn launchQ4KTiled(
    module: *BenchModule,
    ctx: *cuda_context.CudaContext,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !void {
    try launchLinearQ4KTile4(ctx, module.linear_q4_k_f32_tile4, output, input, weight, .{}, rows, in_dim, out_dim, false, false);
}

fn launchQ4KTile8(
    module: *BenchModule,
    ctx: *cuda_context.CudaContext,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !void {
    try launchLinearQ4KTile8(ctx, module.linear_q4_k_f32_tile8, output, input, weight, rows, in_dim, out_dim);
}

fn launchQ4KBiasTiled(
    module: *BenchModule,
    ctx: *cuda_context.CudaContext,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    bias: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !void {
    try launchLinearQ4KTile4(ctx, module.linear_q4_k_bias_f32_tile4, output, input, weight, bias, rows, in_dim, out_dim, true, false);
}

fn launchQ4KBiasGeluRows2(
    module: *BenchModule,
    ctx: *cuda_context.CudaContext,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    bias: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !void {
    try launchLinearQ4KBiasGeluRows2(ctx, module.linear_q4_k_bias_gelu_f32_tile4_r2, output, input, weight, bias, rows, in_dim, out_dim);
}

fn launchGeneratedQ4KBiasGeluRows2(
    module: *GeneratedCandidateModule,
    ctx: *cuda_context.CudaContext,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    bias: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !void {
    try launchGeneratedLinearQ4KBiasGeluRows2(ctx, module.function, output, input, weight, bias, rows, in_dim, out_dim);
}

fn launchQ4KBiasQuickGeluTiled(
    module: *BenchModule,
    ctx: *cuda_context.CudaContext,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    bias: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !void {
    try launchLinearQ4KTile4(ctx, module.linear_q4_k_bias_quick_gelu_f32_tile4, output, input, weight, bias, rows, in_dim, out_dim, true, false);
}

fn launchLinearQ4KTile4(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    bias: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    has_bias: bool,
    has_residual: bool,
) cuda_driver.Error!void {
    try validateQ4KBuffers(output, input, weight, if (has_bias) bias else null, rows, in_dim, out_dim);
    if (has_residual) return error.InvalidCudaState;
    var dst_ptr = output.ptr;
    var input_ptr = input.ptr;
    var weight_ptr = weight.ptr;
    var bias_ptr = bias.ptr;
    var rows_u32 = try toU32(rows);
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&dst_ptr),
        @ptrCast(&input_ptr),
        @ptrCast(&weight_ptr),
        @ptrCast(&bias_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    if (!has_bias) {
        params[3] = @ptrCast(&rows_u32);
        params[4] = @ptrCast(&in_dim_u32);
        params[5] = @ptrCast(&out_dim_u32);
    }
    try launch2d(ctx, function, (out_dim + 3) / 4, rows, 256, &params);
}

fn launchLinearQ4KBiasGeluRows2(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    bias: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    try validateQ4KBuffers(output, input, weight, bias, rows, in_dim, out_dim);
    var dst_ptr = output.ptr;
    var input_ptr = input.ptr;
    var weight_ptr = weight.ptr;
    var bias_ptr = bias.ptr;
    var rows_u32 = try toU32(rows);
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&dst_ptr),
        @ptrCast(&input_ptr),
        @ptrCast(&weight_ptr),
        @ptrCast(&bias_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    try launch2d(ctx, function, (out_dim + 3) / 4, (rows + 1) / 2, 256, &params);
}

fn launchGeneratedLinearQ4KBiasGeluRows2(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    bias: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    try validateQ4KBuffers(output, input, weight, bias, rows, in_dim, out_dim);
    var dst_ptr = output.ptr;
    var input_ptr = input.ptr;
    var weight_ptr = weight.ptr;
    var bias_ptr = bias.ptr;
    var rows_u32 = try toU32(rows);
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&input_ptr),
        @ptrCast(&weight_ptr),
        @ptrCast(&bias_ptr),
        @ptrCast(&dst_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    try launch2d(ctx, function, out_dim, rows, 128, &params);
}

fn launchLinearQ4KTile8(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    try validateQ4KBuffers(output, input, weight, null, rows, in_dim, out_dim);
    var dst_ptr = output.ptr;
    var input_ptr = input.ptr;
    var weight_ptr = weight.ptr;
    var rows_u32 = try toU32(rows);
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&dst_ptr),
        @ptrCast(&input_ptr),
        @ptrCast(&weight_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    try launch2d(ctx, function, (out_dim + 7) / 8, rows, 256, &params);
}

fn launchLinearQ8Raw(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    try validateQ8Buffers(output, input, weight, rows, in_dim, out_dim);
    var dst_ptr = output.ptr;
    var input_ptr = input.ptr;
    var weight_ptr = weight.ptr;
    var rows_u32 = try toU32(rows);
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&dst_ptr),
        @ptrCast(&input_ptr),
        @ptrCast(&weight_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    try launch1d(ctx, function, try checkedMul(rows, out_dim), &params);
}

fn launchLinearQ8Tile4(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    try validateQ8Buffers(output, input, weight, rows, in_dim, out_dim);
    var dst_ptr = output.ptr;
    var input_ptr = input.ptr;
    var weight_ptr = weight.ptr;
    var rows_u32 = try toU32(rows);
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&dst_ptr),
        @ptrCast(&input_ptr),
        @ptrCast(&weight_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    try launch2d(ctx, function, (out_dim + 3) / 4, rows, 256, &params);
}

fn launchQ4KTripleBias(
    module: *BenchModule,
    ctx: *cuda_context.CudaContext,
    output_a: cuda_buffer.DeviceBuffer,
    output_b: cuda_buffer.DeviceBuffer,
    output_c: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    bias: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !void {
    try launchLinearQ4KTripleBiasRaw(ctx, module.linear_q4_k_triple_bias_f32, output_a, output_b, output_c, input, weight, bias, rows, in_dim, out_dim, false);
}

fn launchQ4KTripleBiasTiled(
    module: *BenchModule,
    ctx: *cuda_context.CudaContext,
    output_a: cuda_buffer.DeviceBuffer,
    output_b: cuda_buffer.DeviceBuffer,
    output_c: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    bias: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) !void {
    try launchLinearQ4KTripleBiasRaw(ctx, module.linear_q4_k_triple_bias_f32_tiled, output_a, output_b, output_c, input, weight, bias, rows, in_dim, out_dim, true);
}

fn launchLinearQ4KRaw(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    try validateQ4KBuffers(output, input, weight, null, rows, in_dim, out_dim);
    var dst_ptr = output.ptr;
    var input_ptr = input.ptr;
    var weight_ptr = weight.ptr;
    var rows_u32 = try toU32(rows);
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&dst_ptr),
        @ptrCast(&input_ptr),
        @ptrCast(&weight_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    try launch1d(ctx, function, try checkedMul(rows, out_dim), &params);
}

fn launchLinearQ4KBiasRaw(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    bias: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    try validateQ4KBuffers(output, input, weight, bias, rows, in_dim, out_dim);
    var dst_ptr = output.ptr;
    var input_ptr = input.ptr;
    var weight_ptr = weight.ptr;
    var bias_ptr = bias.ptr;
    var rows_u32 = try toU32(rows);
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&dst_ptr),
        @ptrCast(&input_ptr),
        @ptrCast(&weight_ptr),
        @ptrCast(&bias_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    try launch1d(ctx, function, try checkedMul(rows, out_dim), &params);
}

fn launchLinearQ4KBlocks(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    try validateQ4KBuffers(output, input, weight, null, rows, in_dim, out_dim);
    var dst_ptr = output.ptr;
    var input_ptr = input.ptr;
    var weight_ptr = weight.ptr;
    var rows_u32 = try toU32(rows);
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&dst_ptr),
        @ptrCast(&input_ptr),
        @ptrCast(&weight_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    try launchBlocks(ctx, function, try checkedMul(rows, out_dim), 256, &params);
}

fn launchLinearQ4KBiasBlocks(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    bias: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    try validateQ4KBuffers(output, input, weight, bias, rows, in_dim, out_dim);
    var dst_ptr = output.ptr;
    var input_ptr = input.ptr;
    var weight_ptr = weight.ptr;
    var bias_ptr = bias.ptr;
    var rows_u32 = try toU32(rows);
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&dst_ptr),
        @ptrCast(&input_ptr),
        @ptrCast(&weight_ptr),
        @ptrCast(&bias_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    try launchBlocks(ctx, function, try checkedMul(rows, out_dim), 256, &params);
}

fn launchLinearQ4KTripleBiasRaw(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    output_a: cuda_buffer.DeviceBuffer,
    output_b: cuda_buffer.DeviceBuffer,
    output_c: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    bias: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    tiled: bool,
) cuda_driver.Error!void {
    try validateQ4KBuffers(output_a, input, weight, bias, rows, in_dim, out_dim);
    try validateQ4KBuffers(output_b, input, weight, bias, rows, in_dim, out_dim);
    try validateQ4KBuffers(output_c, input, weight, bias, rows, in_dim, out_dim);
    var dst_a_ptr = output_a.ptr;
    var dst_b_ptr = output_b.ptr;
    var dst_c_ptr = output_c.ptr;
    var input_ptr = input.ptr;
    var weight_a_ptr = weight.ptr;
    var bias_a_ptr = bias.ptr;
    var weight_b_ptr = weight.ptr;
    var bias_b_ptr = bias.ptr;
    var weight_c_ptr = weight.ptr;
    var bias_c_ptr = bias.ptr;
    var rows_u32 = try toU32(rows);
    var in_dim_u32 = try toU32(in_dim);
    var out_dim_u32 = try toU32(out_dim);
    var params = [_]?*anyopaque{
        @ptrCast(&dst_a_ptr),
        @ptrCast(&dst_b_ptr),
        @ptrCast(&dst_c_ptr),
        @ptrCast(&input_ptr),
        @ptrCast(&weight_a_ptr),
        @ptrCast(&bias_a_ptr),
        @ptrCast(&weight_b_ptr),
        @ptrCast(&bias_b_ptr),
        @ptrCast(&weight_c_ptr),
        @ptrCast(&bias_c_ptr),
        @ptrCast(&rows_u32),
        @ptrCast(&in_dim_u32),
        @ptrCast(&out_dim_u32),
    };
    const count = try checkedMul(try checkedMul(rows, out_dim), 3);
    if (tiled) {
        try launchBlocks(ctx, function, count, 256, &params);
    } else {
        try launch1d(ctx, function, count, &params);
    }
}

fn validateQ4KBuffers(
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    bias: ?cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    if (in_dim == 0 or in_dim % q4_k_values_per_block != 0) return error.InvalidCudaState;
    const row_blocks = in_dim / q4_k_values_per_block;
    try checkF32Bytes(output, try checkedMul(rows, out_dim));
    try checkF32Bytes(input, try checkedMul(rows, in_dim));
    try checkRawBytes(weight, try checkedMul(try checkedMul(out_dim, row_blocks), q4_k_block_bytes));
    if (bias) |bias_buf| try checkF32Bytes(bias_buf, out_dim);
}

fn validateQ8Buffers(
    output: cuda_buffer.DeviceBuffer,
    input: cuda_buffer.DeviceBuffer,
    weight: cuda_buffer.DeviceBuffer,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) cuda_driver.Error!void {
    if (in_dim == 0 or in_dim % q8_0_values_per_block != 0) return error.InvalidCudaState;
    const row_blocks = in_dim / q8_0_values_per_block;
    try checkF32Bytes(output, try checkedMul(rows, out_dim));
    try checkF32Bytes(input, try checkedMul(rows, in_dim));
    try checkRawBytes(weight, try checkedMul(try checkedMul(out_dim, row_blocks), q8_0_block_bytes));
}

fn launch1d(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    count: usize,
    params: [*]?*anyopaque,
) cuda_driver.Error!void {
    const block: c_uint = 256;
    const grid = try toU32((count + block - 1) / block);
    try ctx.makeCurrent();
    try ctx.driver.check(ctx.driver.fns.cuLaunchKernel(
        function,
        grid,
        1,
        1,
        block,
        1,
        1,
        0,
        ctx.stream,
        params,
        null,
    ));
}

fn launchBlocks(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    blocks: usize,
    threads: usize,
    params: [*]?*anyopaque,
) cuda_driver.Error!void {
    const grid = try toU32(blocks);
    const block = try toU32(threads);
    try ctx.makeCurrent();
    try ctx.driver.check(ctx.driver.fns.cuLaunchKernel(
        function,
        grid,
        1,
        1,
        block,
        1,
        1,
        0,
        ctx.stream,
        params,
        null,
    ));
}

fn launch2d(
    ctx: *cuda_context.CudaContext,
    function: cuda_driver.CUfunction,
    grid_x: usize,
    grid_y: usize,
    threads: usize,
    params: [*]?*anyopaque,
) cuda_driver.Error!void {
    const gx = try toU32(grid_x);
    const gy = try toU32(grid_y);
    const block = try toU32(threads);
    try ctx.makeCurrent();
    try ctx.driver.check(ctx.driver.fns.cuLaunchKernel(
        function,
        gx,
        gy,
        1,
        block,
        1,
        1,
        0,
        ctx.stream,
        params,
        null,
    ));
}

fn checkedMul(a: usize, b: usize) cuda_driver.Error!usize {
    return std.math.mul(usize, a, b) catch error.InvalidCudaState;
}

fn toU32(value: usize) cuda_driver.Error!u32 {
    if (value > std.math.maxInt(u32)) return error.InvalidCudaState;
    return @intCast(value);
}

fn checkF32Bytes(buffer: cuda_buffer.DeviceBuffer, count: usize) cuda_driver.Error!void {
    try checkRawBytes(buffer, try checkedMul(count, @sizeOf(f32)));
}

fn checkRawBytes(buffer: cuda_buffer.DeviceBuffer, byte_count: usize) cuda_driver.Error!void {
    if (byte_count > buffer.len) return error.InvalidCudaState;
}

fn fillInput(values: []f32) void {
    for (values, 0..) |*value, i| {
        const lane: f32 = @floatFromInt(i % 97);
        value.* = (lane - 48.0) * 0.005;
    }
}

fn fillBias(values: []f32) void {
    for (values, 0..) |*value, i| {
        const lane: f32 = @floatFromInt(i % 23);
        value.* = (lane - 11.0) * 0.001;
    }
}

fn fillQ4KWeights(bytes: []u8) void {
    var block_input: [q4_k_values_per_block]f32 = undefined;
    var offset: usize = 0;
    var block_index: usize = 0;
    while (offset < bytes.len) : ({
        offset += q4_k_block_bytes;
        block_index += 1;
    }) {
        for (&block_input, 0..) |*value, i| {
            const lane: f32 = @floatFromInt((i + block_index * 13) % 31);
            value.* = (lane - 15.0) * 0.01;
        }
        quant_codec.quantizeQ4_KBlock(&block_input, bytes[offset..][0..q4_k_block_bytes]);
    }
}

fn fillQ4_0Weights(bytes: []u8) void {
    var block_input: [q4_0_values_per_block]f32 = undefined;
    var offset: usize = 0;
    var block_index: usize = 0;
    while (offset < bytes.len) : ({
        offset += q4_0_block_bytes;
        block_index += 1;
    }) {
        for (&block_input, 0..) |*value, i| {
            const lane: f32 = @floatFromInt((i + block_index * 11) % 29);
            value.* = (lane - 14.0) * 0.01;
        }
        quant_codec.quantizeQ4_0Block(&block_input, bytes[offset..][0..q4_0_block_bytes]);
    }
}

fn fillQ4_0WeightsAlt(bytes: []u8) void {
    var block_input: [q4_0_values_per_block]f32 = undefined;
    var offset: usize = 0;
    var block_index: usize = 0;
    while (offset < bytes.len) : ({
        offset += q4_0_block_bytes;
        block_index += 1;
    }) {
        for (&block_input, 0..) |*value, i| {
            const lane: f32 = @floatFromInt((i * 3 + block_index * 17) % 23);
            value.* = (lane - 11.0) * 0.012;
        }
        quant_codec.quantizeQ4_0Block(&block_input, bytes[offset..][0..q4_0_block_bytes]);
    }
}

fn fillQ8Weights(bytes: []u8) void {
    var block_input: [q8_0_values_per_block]f32 = undefined;
    var offset: usize = 0;
    var block_index: usize = 0;
    while (offset < bytes.len) : ({
        offset += q8_0_block_bytes;
        block_index += 1;
    }) {
        for (&block_input, 0..) |*value, i| {
            const lane: f32 = @floatFromInt((i + block_index * 7) % 37);
            value.* = (lane - 18.0) * 0.008;
        }
        quant_codec.quantizeQ8_0Block(&block_input, bytes[offset..][0..q8_0_block_bytes]);
    }
}

fn runFullTextEmbedBench(allocator: std.mem.Allocator, io: std.Io, cfg: Config, model_path: []const u8) !void {
    print("\nfull ClipCLAP text embed via termite embed --backend cuda: model={s} iters={d}\n", .{ model_path, cfg.full_iters });

    var total_ns: u64 = 0;
    for (0..cfg.full_iters) |iter| {
        const embed_args = [_][]const u8{
            model_path,
            "--backend",
            "cuda",
            "--text",
            cfg.text,
            "--print-timing",
        };
        const started = nowNs();
        try native_embed.main(allocator, io, &embed_args);
        const elapsed = nowNs() - started;
        total_ns += elapsed;
        print("full_text_embed_iter={d} elapsed_ms={d}\n", .{ iter, elapsed / std.time.ns_per_ms });
    }
    print("full_text_embed_avg_ms={d}\n", .{(total_ns / cfg.full_iters) / std.time.ns_per_ms});
}

fn nowNs() u64 {
    var timespec: std.posix.timespec = undefined;
    switch (std.posix.errno(std.posix.system.clock_gettime(std.posix.CLOCK.MONOTONIC, &timespec))) {
        .SUCCESS => return @intCast(@as(i128, timespec.sec) * std.time.ns_per_s + timespec.nsec),
        else => return 0,
    }
}
