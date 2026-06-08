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
const ops = @import("ops/ops.zig");
const tensor_store_mod = @import("models/tensor_store.zig");

const print = std.debug.print;

const cuda_context = if (build_options.enable_cuda) @import("ops/cuda/context.zig") else struct {};
const cuda_compute = if (build_options.enable_cuda) @import("ops/cuda/cuda_compute.zig") else struct {};
const cuda_kernels = if (build_options.enable_cuda) @import("ops/cuda/kernels.zig") else struct {};

const NormRopeParityCase = struct {
    weight_name: []const u8,
    layer: usize,
    proj: []const u8,
    rows: usize,
    total_dim: usize,
    head_dim: usize,
    rope_dim: usize,
    theta: f32,
    scale: f32,
};

pub fn main(allocator: std.mem.Allocator, _: std.Io, args: []const []const u8) !void {
    var smoke = false;
    var gemma4_parity_path: ?[]const u8 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--smoke")) {
            smoke = true;
        } else if (std.mem.eql(u8, arg, "--gemma4-parity")) {
            i += 1;
            if (i >= args.len) {
                print("missing value for --gemma4-parity\n", .{});
                printUsage();
                std.process.exit(1);
            }
            gemma4_parity_path = args[i];
        } else if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            return;
        } else {
            print("unknown cuda-info option: {s}\n", .{arg});
            printUsage();
            std.process.exit(1);
        }
    }

    if (!build_options.enable_cuda) {
        print("cuda: unavailable\nreason: backend not built; rebuild with -Dcuda=true\n", .{});
        std.process.exit(1);
    }

    if (comptime build_options.enable_cuda) {
        const info = cuda_context.probeDefault() catch |err| {
            print("cuda: unavailable\nreason: {s}\n", .{@errorName(err)});
            std.process.exit(1);
        };

        print("cuda: available\n", .{});
        print("driver_version: {d}\n", .{info.driver_version});
        print("device_count: {d}\n", .{info.device_count});
        print("selected_device: {d}\n", .{info.selected_device});
        print("device_name: {s}\n", .{info.nameSlice()});
        print("compute_capability: sm_{d}{d}\n", .{ info.compute_major, info.compute_minor });
        print("artifacts: {s}\n", .{build_options.cuda_artifacts});
        var compute = cuda_compute.CudaCompute.init(allocator) catch |err| {
            print("capabilities: unavailable\nreason: {s}\n", .{@errorName(err)});
            std.process.exit(1);
        };
        defer compute.deinit();
        print("capability_clipclap: {}\n", .{compute.supportsProfile(.clipclap)});
        print("capability_gliner2: {}\n", .{compute.supportsProfile(.gliner2)});
        print("capability_gemma4: {}\n", .{compute.supportsProfile(.gemma4)});

        if (smoke) {
            cuda_kernels.smokeFill(allocator) catch |err| {
                print("smoke: fill_f32 failed\nreason: {s}\n", .{@errorName(err)});
                std.process.exit(1);
            };
            print("smoke: fill_f32 ok\n", .{});
            cuda_kernels.smokeDenseF32(allocator) catch |err| {
                print("smoke: dense_f32 failed\nreason: {s}\n", .{@errorName(err)});
                std.process.exit(1);
            };
            print("smoke: dense_f32 ok\n", .{});
            cuda_kernels.smokeQ8_0(allocator) catch |err| {
                print("smoke: q8_0_f32 failed\nreason: {s}\n", .{@errorName(err)});
                std.process.exit(1);
            };
            print("smoke: q8_0_f32 ok\n", .{});
            cuda_kernels.smokeQ4_0(allocator) catch |err| {
                print("smoke: q4_0_f32 failed\nreason: {s}\n", .{@errorName(err)});
                std.process.exit(1);
            };
            print("smoke: q4_0_f32 ok\n", .{});
            cuda_kernels.smokeQ4_K(allocator) catch |err| {
                print("smoke: q4_k_f32 failed\nreason: {s}\n", .{@errorName(err)});
                std.process.exit(1);
            };
            print("smoke: q4_k_f32 ok\n", .{});
            cuda_kernels.smokeGemma4Primitives(allocator) catch |err| {
                print("smoke: gemma4_primitives failed\nreason: {s}\n", .{@errorName(err)});
                std.process.exit(1);
            };
            print("smoke: gemma4_primitives ok\n", .{});
        }

        if (gemma4_parity_path) |path| {
            runGemma4Parity(allocator, path) catch |err| {
                print("gemma4_parity: failed\nreason: {s}\n", .{@errorName(err)});
                std.process.exit(1);
            };
            print("gemma4_parity: ok\n", .{});
        }
    }
}

fn printUsage() void {
    print(
        \\usage: antfly inference cuda-info [--smoke] [--gemma4-parity <gguf>]
        \\
        \\  --smoke   Run embedded PTX smoke checks for fill, dense f32 ops, Q8_0, Q4_0, Q4_K, RoPE, and GQA.
        \\  --gemma4-parity <gguf>
        \\            Compare real Gemma 4 GGUF projection tensors on CUDA against CPU dequantized matmul.
        \\
    , .{});
}

const Gemma4ParityCase = struct {
    name: []const u8,
};

const gemma4_parity_cases = [_]Gemma4ParityCase{
    .{ .name = "blk.0.attn_q.weight" },
    .{ .name = "blk.0.attn_k.weight" },
    .{ .name = "blk.0.attn_v.weight" },
    .{ .name = "blk.0.attn_output.weight" },
    .{ .name = "blk.0.ffn_gate.weight" },
    .{ .name = "blk.0.ffn_up.weight" },
    .{ .name = "blk.0.ffn_down.weight" },
    .{ .name = "blk.5.attn_q.weight" },
    .{ .name = "blk.5.attn_k.weight" },
    .{ .name = "blk.5.attn_output.weight" },
    .{ .name = "blk.5.ffn_gate.weight" },
    .{ .name = "blk.5.ffn_up.weight" },
    .{ .name = "blk.5.ffn_down.weight" },
};

const DiffStats = struct {
    max_abs: f32 = 0,
    mean_abs: f64 = 0,
    max_index: usize = 0,
};

fn runGemma4Parity(allocator: std.mem.Allocator, gguf_path: []const u8) !void {
    const store = try tensor_store_mod.GgufStore.initAbsolute(allocator, gguf_path);
    defer store.tensorStore().deinit();
    const tensor_store = store.tensorStore();

    for (gemma4_parity_cases) |case| {
        try runGemma4LinearParityCase(allocator, tensor_store, case);
    }
    try runGemma4NormRopeParity(allocator, tensor_store);
    try runGqaParity(allocator);
    try runGemma4Layer0Parity(allocator, tensor_store);
    try runGemma4Layer5AttentionParity(allocator, tensor_store);
}

fn runGemma4LinearParityCase(
    allocator: std.mem.Allocator,
    tensor_store: tensor_store_mod.TensorStore,
    case: Gemma4ParityCase,
) !void {
    var tensor_ref = try tensor_store.describeTensor(allocator, case.name);
    defer tensor_ref.deinit(allocator);

    var loaded = try tensor_store.loadTensorRef(&tensor_ref);
    defer loaded.deinit();
    if (loaded.tensor.dtype != .f32) return error.UnsupportedTensorType;
    if (loaded.tensor.shape.len != 2) return error.InvalidTensorShape;

    const out_dim: usize = @intCast(loaded.tensor.shape[0]);
    const in_dim: usize = @intCast(loaded.tensor.shape[1]);
    const rows: usize = 1;
    const input = try makeParityInput(allocator, in_dim);
    defer allocator.free(input);

    const expected = try allocator.alloc(f32, out_dim);
    defer allocator.free(expected);
    cpuLinearNoBias(input, loaded.tensor.asFloat32(), expected, in_dim, out_dim);

    var compute = try cuda_compute.CudaCompute.init(allocator);
    defer compute.deinit();
    const key = try allocator.dupe(u8, case.name);
    var inserted = false;
    errdefer if (!inserted) allocator.free(key);
    try compute.insertWeightFromLoaded(key, &loaded);
    inserted = true;
    const cb = compute.computeBackend();

    const input_shape = [_]i32{ @intCast(rows), @intCast(in_dim) };
    const input_ct = try cb.fromFloat32Shape(input, &input_shape);
    defer cb.free(input_ct);
    const weight_ct = try cb.getWeight(case.name);
    const output_ct = try cb.linearNoBias(input_ct, weight_ct, rows, in_dim, out_dim);
    defer cb.free(output_ct);
    const actual = try cb.toFloat32(output_ct, allocator);
    defer allocator.free(actual);
    if (actual.len != expected.len) return error.InvalidTensorShape;

    const stats = diffStats(actual, expected);
    const tolerance = toleranceForParityCase(loaded.quantized);
    const quant_name = if (loaded.quantized_storage) |storage| storage.tensor_type.name() else "F32";
    print(
        "gemma4_parity: {s} type={s} shape=[{d},{d}] max_abs={d:.6} mean_abs={d:.6} max_index={d}\n",
        .{ case.name, quant_name, out_dim, in_dim, stats.max_abs, stats.mean_abs, stats.max_index },
    );
    if (stats.max_abs > tolerance) return error.CudaParityMismatch;
}

fn runGemma4NormRopeParity(
    allocator: std.mem.Allocator,
    tensor_store: tensor_store_mod.TensorStore,
) !void {
    const cases = [_]NormRopeParityCase{
        .{
            .weight_name = "blk.0.attn_q_norm.weight",
            .layer = 0,
            .proj = "q",
            .rows = 2,
            .total_dim = 4096,
            .head_dim = 256,
            .rope_dim = 256,
            .theta = 10000.0,
            .scale = @sqrt(@as(f32, 256.0)),
        },
        .{
            .weight_name = "blk.0.attn_k_norm.weight",
            .layer = 0,
            .proj = "k",
            .rows = 2,
            .total_dim = 2048,
            .head_dim = 256,
            .rope_dim = 256,
            .theta = 10000.0,
            .scale = 1.0,
        },
        .{
            .weight_name = "blk.5.attn_q_norm.weight",
            .layer = 5,
            .proj = "q",
            .rows = 2,
            .total_dim = 8192,
            .head_dim = 512,
            .rope_dim = 512,
            .theta = 1000000.0,
            .scale = @sqrt(@as(f32, 512.0)),
        },
        .{
            .weight_name = "blk.5.attn_k_norm.weight",
            .layer = 5,
            .proj = "k",
            .rows = 2,
            .total_dim = 512,
            .head_dim = 512,
            .rope_dim = 512,
            .theta = 1000000.0,
            .scale = 1.0,
        },
    };

    for (cases) |case| {
        try runNormRopeParityCase(allocator, tensor_store, case);
    }
}

fn runNormRopeParityCase(
    allocator: std.mem.Allocator,
    tensor_store: tensor_store_mod.TensorStore,
    case: NormRopeParityCase,
) !void {
    var tensor_ref = try tensor_store.describeTensor(allocator, case.weight_name);
    defer tensor_ref.deinit(allocator);

    var loaded = try tensor_store.loadTensorRef(&tensor_ref);
    defer loaded.deinit();
    if (loaded.tensor.dtype != .f32) return error.UnsupportedTensorType;
    if (loaded.tensor.elementCount() != case.head_dim) return error.InvalidTensorShape;

    const count = case.rows * case.total_dim;
    const input = try makeParityInput(allocator, count);
    defer allocator.free(input);

    const expected = try allocator.alloc(f32, count);
    defer allocator.free(expected);
    cpuRmsNormHeadsRope(
        input,
        loaded.tensor.asFloat32(),
        expected,
        case.rows,
        case.total_dim,
        case.head_dim,
        case.rope_dim,
        0.000001,
        case.theta,
        1.0,
        0,
        case.rows,
        false,
        case.scale,
    );

    var compute = try cuda_compute.CudaCompute.init(allocator);
    defer compute.deinit();
    const key = try allocator.dupe(u8, case.weight_name);
    var inserted = false;
    errdefer if (!inserted) allocator.free(key);
    try compute.insertWeightFromLoaded(key, &loaded);
    inserted = true;
    const cb = compute.computeBackend();

    const input_shape = [_]i32{ @intCast(case.rows), @intCast(case.total_dim) };
    const input_ct = try cb.fromFloat32Shape(input, &input_shape);
    defer cb.free(input_ct);
    const weight_ct = try cb.getWeight(case.weight_name);
    const output_ct = (try cb.rmsNormHeadsRope(
        input_ct,
        weight_ct,
        case.rows,
        case.total_dim,
        case.head_dim,
        case.rope_dim,
        0.000001,
        case.theta,
        1.0,
        0,
        case.rows,
        false,
        case.scale,
    )) orelse return error.CudaKernelUnavailable;
    defer cb.free(output_ct);

    const actual = try cb.toFloat32(output_ct, allocator);
    defer allocator.free(actual);
    if (actual.len != expected.len) return error.InvalidTensorShape;
    const stats = diffStats(actual, expected);
    print(
        "gemma4_parity: rms_norm_heads_rope layer={d} proj={s} shape=[{d},{d}] max_abs={d:.6} mean_abs={d:.6} max_index={d}\n",
        .{ case.layer, case.proj, case.rows, case.total_dim, stats.max_abs, stats.mean_abs, stats.max_index },
    );
    if (stats.max_abs > 0.001) return error.CudaParityMismatch;
}

fn runGqaParity(allocator: std.mem.Allocator) !void {
    const batch: usize = 1;
    const seq_len: usize = 3;
    const num_heads: usize = 4;
    const num_kv_heads: usize = 2;
    const head_dim: usize = 8;
    const q_count = batch * seq_len * num_heads * head_dim;
    const kv_count = batch * seq_len * num_kv_heads * head_dim;
    const q = try makeParityInput(allocator, q_count);
    defer allocator.free(q);
    const k = try makeParityInput(allocator, kv_count);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, kv_count);
    defer allocator.free(v);
    for (v, 0..) |*value, idx| {
        const centered: i32 = @as(i32, @intCast((idx * 7 + 11) % 199)) - 99;
        value.* = @as(f32, @floatFromInt(centered)) / 99.0;
    }

    const expected = try allocator.alloc(f32, q_count);
    defer allocator.free(expected);
    cpuGqaCausalAttention(q, k, v, expected, batch, seq_len, num_heads, num_kv_heads, head_dim);

    var compute = try cuda_compute.CudaCompute.init(allocator);
    defer compute.deinit();
    const cb = compute.computeBackend();
    const q_shape = [_]i32{ @intCast(batch * seq_len), @intCast(num_heads * head_dim) };
    const kv_shape = [_]i32{ @intCast(batch * seq_len), @intCast(num_kv_heads * head_dim) };
    const q_ct = try cb.fromFloat32Shape(q, &q_shape);
    defer cb.free(q_ct);
    const k_ct = try cb.fromFloat32Shape(k, &kv_shape);
    defer cb.free(k_ct);
    const v_ct = try cb.fromFloat32Shape(v, &kv_shape);
    defer cb.free(v_ct);
    const output_ct = try cb.gqaCausalAttention(q_ct, k_ct, v_ct, null, batch, seq_len, num_heads, num_kv_heads, head_dim);
    defer cb.free(output_ct);
    const actual = try cb.toFloat32(output_ct, allocator);
    defer allocator.free(actual);
    if (actual.len != expected.len) return error.InvalidTensorShape;
    const stats = diffStats(actual, expected);
    print(
        "gemma4_parity: gqa_causal_attention shape=[{d},{d},{d},{d}] max_abs={d:.6} mean_abs={d:.6} max_index={d}\n",
        .{ batch, seq_len, num_heads, head_dim, stats.max_abs, stats.mean_abs, stats.max_index },
    );
    if (stats.max_abs > 0.001) return error.CudaParityMismatch;
}

const Gemma4Layer0Weights = struct {
    attn_norm: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    q_norm: []f32,
    k_norm: []f32,
    attn_output: []f32,
    post_attention_norm: []f32,
    ffn_norm: []f32,
    ffn_gate: []f32,
    ffn_up: []f32,
    ffn_down: []f32,
    post_ffw_norm: []f32,
    layer_output_scale: []f32,

    fn deinit(self: *Gemma4Layer0Weights, allocator: std.mem.Allocator) void {
        allocator.free(self.attn_norm);
        allocator.free(self.q);
        allocator.free(self.k);
        allocator.free(self.v);
        allocator.free(self.q_norm);
        allocator.free(self.k_norm);
        allocator.free(self.attn_output);
        allocator.free(self.post_attention_norm);
        allocator.free(self.ffn_norm);
        allocator.free(self.ffn_gate);
        allocator.free(self.ffn_up);
        allocator.free(self.ffn_down);
        allocator.free(self.post_ffw_norm);
        allocator.free(self.layer_output_scale);
        self.* = undefined;
    }
};

fn runGemma4Layer0Parity(
    allocator: std.mem.Allocator,
    tensor_store: tensor_store_mod.TensorStore,
) !void {
    const rows: usize = 3;
    const hidden: usize = 3840;
    const q_dim: usize = 4096;
    const kv_dim: usize = 2048;
    const num_heads: usize = 16;
    const num_kv_heads: usize = 8;
    const head_dim: usize = 256;
    const inter: usize = 15360;
    const eps: f32 = 0.000001;
    const theta: f32 = 10000.0;

    var compute = try cuda_compute.CudaCompute.init(allocator);
    defer compute.deinit();
    var weights = try loadGemma4Layer0Weights(allocator, tensor_store, &compute);
    defer weights.deinit(allocator);
    const cb = compute.computeBackend();

    const input = try makeParityInput(allocator, rows * hidden);
    defer allocator.free(input);
    const input_shape = [_]i32{ @intCast(rows), @intCast(hidden) };
    const input_ct = try cb.fromFloat32Shape(input, &input_shape);
    defer cb.free(input_ct);

    const expected_attn_norm = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_attn_norm);
    cpuRmsNormRows(input, weights.attn_norm, expected_attn_norm, rows, hidden, eps);
    const attn_norm_w = try cb.getWeight("blk.0.attn_norm.weight");
    const attn_norm_ct = try cb.rmsNorm(input_ct, attn_norm_w, hidden, eps);
    defer cb.free(attn_norm_ct);
    try compareCudaStage(allocator, &cb, "layer0.attn_norm", attn_norm_ct, expected_attn_norm, 0.001);

    const expected_q = try allocator.alloc(f32, rows * q_dim);
    defer allocator.free(expected_q);
    const expected_k = try allocator.alloc(f32, rows * kv_dim);
    defer allocator.free(expected_k);
    const expected_v = try allocator.alloc(f32, rows * kv_dim);
    defer allocator.free(expected_v);
    cpuLinearNoBiasRows(expected_attn_norm, weights.q, expected_q, rows, hidden, q_dim);
    cpuLinearNoBiasRows(expected_attn_norm, weights.k, expected_k, rows, hidden, kv_dim);
    cpuLinearNoBiasRows(expected_attn_norm, weights.v, expected_v, rows, hidden, kv_dim);
    const q_w = try cb.getWeight("blk.0.attn_q.weight");
    const k_w = try cb.getWeight("blk.0.attn_k.weight");
    const v_w = try cb.getWeight("blk.0.attn_v.weight");
    const qkv_ct = try cb.linearNoBiasQkv(attn_norm_ct, q_w, k_w, v_w, rows, hidden, q_dim, kv_dim);
    const q_ct = qkv_ct.first;
    defer cb.free(q_ct);
    const k_ct = qkv_ct.second;
    defer cb.free(k_ct);
    const v_ct = qkv_ct.third;
    defer cb.free(v_ct);
    try compareCudaStage(allocator, &cb, "layer0.q", q_ct, expected_q, 0.001);
    try compareCudaStage(allocator, &cb, "layer0.k", k_ct, expected_k, 0.001);
    try compareCudaStage(allocator, &cb, "layer0.v", v_ct, expected_v, 0.001);

    const expected_q_rope = try allocator.alloc(f32, rows * q_dim);
    defer allocator.free(expected_q_rope);
    const expected_k_rope = try allocator.alloc(f32, rows * kv_dim);
    defer allocator.free(expected_k_rope);
    cpuRmsNormHeadsRope(expected_q, weights.q_norm, expected_q_rope, rows, q_dim, head_dim, head_dim, eps, theta, 1.0, 0, rows, false, @sqrt(@as(f32, @floatFromInt(head_dim))));
    cpuRmsNormHeadsRope(expected_k, weights.k_norm, expected_k_rope, rows, kv_dim, head_dim, head_dim, eps, theta, 1.0, 0, rows, false, 1.0);
    const q_norm_w = try cb.getWeight("blk.0.attn_q_norm.weight");
    const k_norm_w = try cb.getWeight("blk.0.attn_k_norm.weight");
    const q_rope_ct = (try cb.rmsNormHeadsRope(q_ct, q_norm_w, rows, q_dim, head_dim, head_dim, eps, theta, 1.0, 0, rows, false, @sqrt(@as(f32, @floatFromInt(head_dim))))) orelse return error.CudaKernelUnavailable;
    defer cb.free(q_rope_ct);
    const k_rope_ct = (try cb.rmsNormHeadsRope(k_ct, k_norm_w, rows, kv_dim, head_dim, head_dim, eps, theta, 1.0, 0, rows, false, 1.0)) orelse return error.CudaKernelUnavailable;
    defer cb.free(k_rope_ct);
    try compareCudaStage(allocator, &cb, "layer0.q_rope", q_rope_ct, expected_q_rope, 0.001);
    try compareCudaStage(allocator, &cb, "layer0.k_rope", k_rope_ct, expected_k_rope, 0.001);

    const expected_v_norm = try allocator.alloc(f32, rows * kv_dim);
    defer allocator.free(expected_v_norm);
    cpuRmsNormBareRows(expected_v, expected_v_norm, rows * num_kv_heads, head_dim, eps);
    const v_flat_ct = (try cb.reshape2d(v_ct, rows * num_kv_heads, head_dim)) orelse return error.ReshapeFailed;
    defer cb.free(v_flat_ct);
    const v_norm_flat_ct = (try cb.rmsNormBare(v_flat_ct, head_dim, eps)) orelse return error.CudaKernelUnavailable;
    defer cb.free(v_norm_flat_ct);
    const v_norm_ct = (try cb.reshape2d(v_norm_flat_ct, rows, kv_dim)) orelse return error.ReshapeFailed;
    defer cb.free(v_norm_ct);
    try compareCudaStage(allocator, &cb, "layer0.v_norm", v_norm_ct, expected_v_norm, 0.001);

    const expected_attn = try allocator.alloc(f32, rows * q_dim);
    defer allocator.free(expected_attn);
    cpuGqaCausalAttention(expected_q_rope, expected_k_rope, expected_v_norm, expected_attn, 1, rows, num_heads, num_kv_heads, head_dim);
    const attn_ct = try cb.gqaCausalAttention(q_rope_ct, k_rope_ct, v_norm_ct, null, 1, rows, num_heads, num_kv_heads, head_dim);
    defer cb.free(attn_ct);
    try compareCudaStage(allocator, &cb, "layer0.attn_out", attn_ct, expected_attn, 0.001);

    const expected_attn_proj = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_attn_proj);
    cpuLinearNoBiasRows(expected_attn, weights.attn_output, expected_attn_proj, rows, q_dim, hidden);
    const attn_out_w = try cb.getWeight("blk.0.attn_output.weight");
    const attn_proj_ct = try cb.linearNoBias(attn_ct, attn_out_w, rows, q_dim, hidden);
    defer cb.free(attn_proj_ct);
    try compareCudaStage(allocator, &cb, "layer0.attn_proj", attn_proj_ct, expected_attn_proj, 0.001);

    const expected_attn_post = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_attn_post);
    cpuRmsNormRows(expected_attn_proj, weights.post_attention_norm, expected_attn_post, rows, hidden, eps);
    const post_attn_w = try cb.getWeight("blk.0.post_attention_norm.weight");
    const attn_post_ct = try cb.rmsNorm(attn_proj_ct, post_attn_w, hidden, eps);
    defer cb.free(attn_post_ct);
    try compareCudaStage(allocator, &cb, "layer0.attn_post", attn_post_ct, expected_attn_post, 0.001);

    const expected_sa = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_sa);
    cpuAdd(expected_attn_post, input, expected_sa);
    const sa_ct = try cb.add(attn_post_ct, input_ct);
    defer cb.free(sa_ct);
    try compareCudaStage(allocator, &cb, "layer0.attn_residual", sa_ct, expected_sa, 0.001);

    const expected_ffn_norm = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_ffn_norm);
    cpuRmsNormRows(expected_sa, weights.ffn_norm, expected_ffn_norm, rows, hidden, eps);
    const ffn_norm_w = try cb.getWeight("blk.0.ffn_norm.weight");
    const ffn_norm_ct = try cb.rmsNorm(sa_ct, ffn_norm_w, hidden, eps);
    defer cb.free(ffn_norm_ct);
    try compareCudaStage(allocator, &cb, "layer0.ffn_norm", ffn_norm_ct, expected_ffn_norm, 0.001);

    const expected_gate = try allocator.alloc(f32, rows * inter);
    defer allocator.free(expected_gate);
    const expected_up = try allocator.alloc(f32, rows * inter);
    defer allocator.free(expected_up);
    const expected_gated = try allocator.alloc(f32, rows * inter);
    defer allocator.free(expected_gated);
    cpuLinearNoBiasRows(expected_ffn_norm, weights.ffn_gate, expected_gate, rows, hidden, inter);
    cpuLinearNoBiasRows(expected_ffn_norm, weights.ffn_up, expected_up, rows, hidden, inter);
    cpuSiluMultiply(expected_gate, expected_up, expected_gated);
    const gate_w = try cb.getWeight("blk.0.ffn_gate.weight");
    const up_w = try cb.getWeight("blk.0.ffn_up.weight");
    const gate_up_ct = try cb.linearNoBiasPair(ffn_norm_ct, gate_w, up_w, rows, hidden, inter);
    const gate_ct = gate_up_ct.first;
    defer cb.free(gate_ct);
    const up_ct = gate_up_ct.second;
    defer cb.free(up_ct);
    try compareCudaStage(allocator, &cb, "layer0.ffn_gate", gate_ct, expected_gate, 0.001);
    try compareCudaStage(allocator, &cb, "layer0.ffn_up", up_ct, expected_up, 0.001);
    const gated_ct = (try cb.activationMultiply(gate_ct, up_ct, .silu)) orelse return error.CudaKernelUnavailable;
    defer cb.free(gated_ct);
    try compareCudaStage(allocator, &cb, "layer0.ffn_gated", gated_ct, expected_gated, 0.005);

    const expected_ffn_raw = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_ffn_raw);
    cpuLinearNoBiasRows(expected_gated, weights.ffn_down, expected_ffn_raw, rows, inter, hidden);
    const down_w = try cb.getWeight("blk.0.ffn_down.weight");
    const ffn_raw_ct = try cb.linearNoBias(gated_ct, down_w, rows, inter, hidden);
    defer cb.free(ffn_raw_ct);
    try compareCudaStage(allocator, &cb, "layer0.ffn_raw", ffn_raw_ct, expected_ffn_raw, 0.005);

    const expected_ffn_post = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_ffn_post);
    cpuRmsNormRows(expected_ffn_raw, weights.post_ffw_norm, expected_ffn_post, rows, hidden, eps);
    const post_ffw_w = try cb.getWeight("blk.0.post_ffw_norm.weight");
    const ffn_post_ct = try cb.rmsNorm(ffn_raw_ct, post_ffw_w, hidden, eps);
    defer cb.free(ffn_post_ct);
    try compareCudaStage(allocator, &cb, "layer0.ffn_post", ffn_post_ct, expected_ffn_post, 0.001);

    const expected_out = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_out);
    cpuAddScaled(expected_ffn_post, expected_sa, weights.layer_output_scale[0], expected_out);
    const scale_w = try cb.getWeight("blk.0.layer_output_scale.weight");
    const out_ct = if (try cb.addMultiplyScalarTensor(ffn_post_ct, sa_ct, scale_w)) |fused|
        fused
    else blk: {
        const scaled = try cb.multiply(ffn_post_ct, scale_w);
        defer cb.free(scaled);
        break :blk try cb.add(scaled, sa_ct);
    };
    defer cb.free(out_ct);
    try compareCudaStage(allocator, &cb, "layer0.out", out_ct, expected_out, 0.001);

    print("gemma4_parity: layer0_decoder_block ok\n", .{});
}

fn runGemma4Layer5AttentionParity(
    allocator: std.mem.Allocator,
    tensor_store: tensor_store_mod.TensorStore,
) !void {
    const rows: usize = 3;
    const hidden: usize = 3840;
    const q_dim: usize = 8192;
    const kv_dim: usize = 512;
    const num_heads: usize = 16;
    const num_kv_heads: usize = 1;
    const head_dim: usize = 512;
    const eps: f32 = 0.000001;
    const theta: f32 = 1000000.0;

    var compute = try cuda_compute.CudaCompute.init(allocator);
    defer compute.deinit();
    const attn_norm = try loadLayerWeight(allocator, tensor_store, &compute, "blk.5.attn_norm.weight");
    defer allocator.free(attn_norm);
    const q_w_host = try loadLayerWeight(allocator, tensor_store, &compute, "blk.5.attn_q.weight");
    defer allocator.free(q_w_host);
    const k_w_host = try loadLayerWeight(allocator, tensor_store, &compute, "blk.5.attn_k.weight");
    defer allocator.free(k_w_host);
    const q_norm = try loadLayerWeight(allocator, tensor_store, &compute, "blk.5.attn_q_norm.weight");
    defer allocator.free(q_norm);
    const k_norm = try loadLayerWeight(allocator, tensor_store, &compute, "blk.5.attn_k_norm.weight");
    defer allocator.free(k_norm);
    const attn_output = try loadLayerWeight(allocator, tensor_store, &compute, "blk.5.attn_output.weight");
    defer allocator.free(attn_output);
    const post_attention_norm = try loadLayerWeight(allocator, tensor_store, &compute, "blk.5.post_attention_norm.weight");
    defer allocator.free(post_attention_norm);
    const cb = compute.computeBackend();

    const input = try makeParityInput(allocator, rows * hidden);
    defer allocator.free(input);
    const input_shape = [_]i32{ @intCast(rows), @intCast(hidden) };
    const input_ct = try cb.fromFloat32Shape(input, &input_shape);
    defer cb.free(input_ct);

    const expected_attn_norm = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_attn_norm);
    cpuRmsNormRows(input, attn_norm, expected_attn_norm, rows, hidden, eps);
    const attn_norm_w = try cb.getWeight("blk.5.attn_norm.weight");
    const attn_norm_ct = try cb.rmsNorm(input_ct, attn_norm_w, hidden, eps);
    defer cb.free(attn_norm_ct);
    try compareCudaStage(allocator, &cb, "layer5.attn_norm", attn_norm_ct, expected_attn_norm, 0.001);

    const expected_q = try allocator.alloc(f32, rows * q_dim);
    defer allocator.free(expected_q);
    const expected_k = try allocator.alloc(f32, rows * kv_dim);
    defer allocator.free(expected_k);
    cpuLinearNoBiasRows(expected_attn_norm, q_w_host, expected_q, rows, hidden, q_dim);
    cpuLinearNoBiasRows(expected_attn_norm, k_w_host, expected_k, rows, hidden, kv_dim);
    const q_w = try cb.getWeight("blk.5.attn_q.weight");
    const k_w = try cb.getWeight("blk.5.attn_k.weight");
    const q_ct = try cb.linearNoBias(attn_norm_ct, q_w, rows, hidden, q_dim);
    defer cb.free(q_ct);
    const k_ct = try cb.linearNoBias(attn_norm_ct, k_w, rows, hidden, kv_dim);
    defer cb.free(k_ct);
    try compareCudaStage(allocator, &cb, "layer5.q", q_ct, expected_q, 0.005);
    try compareCudaStage(allocator, &cb, "layer5.k", k_ct, expected_k, 0.005);

    const expected_q_rope = try allocator.alloc(f32, rows * q_dim);
    defer allocator.free(expected_q_rope);
    const expected_k_rope = try allocator.alloc(f32, rows * kv_dim);
    defer allocator.free(expected_k_rope);
    cpuRmsNormHeadsRope(expected_q, q_norm, expected_q_rope, rows, q_dim, head_dim, head_dim, eps, theta, 1.0, 0, rows, false, @sqrt(@as(f32, @floatFromInt(head_dim))));
    cpuRmsNormHeadsRope(expected_k, k_norm, expected_k_rope, rows, kv_dim, head_dim, head_dim, eps, theta, 1.0, 0, rows, false, 1.0);
    const q_norm_w = try cb.getWeight("blk.5.attn_q_norm.weight");
    const k_norm_w = try cb.getWeight("blk.5.attn_k_norm.weight");
    const q_rope_ct = (try cb.rmsNormHeadsRope(q_ct, q_norm_w, rows, q_dim, head_dim, head_dim, eps, theta, 1.0, 0, rows, false, @sqrt(@as(f32, @floatFromInt(head_dim))))) orelse return error.CudaKernelUnavailable;
    defer cb.free(q_rope_ct);
    const k_rope_ct = (try cb.rmsNormHeadsRope(k_ct, k_norm_w, rows, kv_dim, head_dim, head_dim, eps, theta, 1.0, 0, rows, false, 1.0)) orelse return error.CudaKernelUnavailable;
    defer cb.free(k_rope_ct);
    try compareCudaStage(allocator, &cb, "layer5.q_rope", q_rope_ct, expected_q_rope, 0.005);
    try compareCudaStage(allocator, &cb, "layer5.k_rope", k_rope_ct, expected_k_rope, 0.001);

    const expected_v_norm = try allocator.alloc(f32, rows * kv_dim);
    defer allocator.free(expected_v_norm);
    cpuRmsNormBareRows(expected_k, expected_v_norm, rows * num_kv_heads, head_dim, eps);
    const v_flat_ct = (try cb.reshape2d(k_ct, rows * num_kv_heads, head_dim)) orelse return error.ReshapeFailed;
    defer cb.free(v_flat_ct);
    const v_norm_flat_ct = (try cb.rmsNormBare(v_flat_ct, head_dim, eps)) orelse return error.CudaKernelUnavailable;
    defer cb.free(v_norm_flat_ct);
    const v_norm_ct = (try cb.reshape2d(v_norm_flat_ct, rows, kv_dim)) orelse return error.ReshapeFailed;
    defer cb.free(v_norm_ct);
    try compareCudaStage(allocator, &cb, "layer5.v_norm_from_k", v_norm_ct, expected_v_norm, 0.001);

    const expected_attn = try allocator.alloc(f32, rows * q_dim);
    defer allocator.free(expected_attn);
    cpuGqaCausalAttention(expected_q_rope, expected_k_rope, expected_v_norm, expected_attn, 1, rows, num_heads, num_kv_heads, head_dim);
    const attn_ct = try cb.gqaCausalAttention(q_rope_ct, k_rope_ct, v_norm_ct, null, 1, rows, num_heads, num_kv_heads, head_dim);
    defer cb.free(attn_ct);
    try compareCudaStage(allocator, &cb, "layer5.attn_out", attn_ct, expected_attn, 0.001);

    const expected_attn_proj = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_attn_proj);
    cpuLinearNoBiasRows(expected_attn, attn_output, expected_attn_proj, rows, q_dim, hidden);
    const attn_out_w = try cb.getWeight("blk.5.attn_output.weight");
    const attn_proj_ct = try cb.linearNoBias(attn_ct, attn_out_w, rows, q_dim, hidden);
    defer cb.free(attn_proj_ct);
    try compareCudaStage(allocator, &cb, "layer5.attn_proj", attn_proj_ct, expected_attn_proj, 0.005);

    const expected_attn_post = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_attn_post);
    cpuRmsNormRows(expected_attn_proj, post_attention_norm, expected_attn_post, rows, hidden, eps);
    const post_attn_w = try cb.getWeight("blk.5.post_attention_norm.weight");
    const attn_post_ct = try cb.rmsNorm(attn_proj_ct, post_attn_w, hidden, eps);
    defer cb.free(attn_post_ct);
    try compareCudaStage(allocator, &cb, "layer5.attn_post", attn_post_ct, expected_attn_post, 0.001);

    print("gemma4_parity: layer5_attention_omitted_v ok\n", .{});
}

fn loadGemma4Layer0Weights(
    allocator: std.mem.Allocator,
    tensor_store: tensor_store_mod.TensorStore,
    compute: *cuda_compute.CudaCompute,
) !Gemma4Layer0Weights {
    return .{
        .attn_norm = try loadLayerWeight(allocator, tensor_store, compute, "blk.0.attn_norm.weight"),
        .q = try loadLayerWeight(allocator, tensor_store, compute, "blk.0.attn_q.weight"),
        .k = try loadLayerWeight(allocator, tensor_store, compute, "blk.0.attn_k.weight"),
        .v = try loadLayerWeight(allocator, tensor_store, compute, "blk.0.attn_v.weight"),
        .q_norm = try loadLayerWeight(allocator, tensor_store, compute, "blk.0.attn_q_norm.weight"),
        .k_norm = try loadLayerWeight(allocator, tensor_store, compute, "blk.0.attn_k_norm.weight"),
        .attn_output = try loadLayerWeight(allocator, tensor_store, compute, "blk.0.attn_output.weight"),
        .post_attention_norm = try loadLayerWeight(allocator, tensor_store, compute, "blk.0.post_attention_norm.weight"),
        .ffn_norm = try loadLayerWeight(allocator, tensor_store, compute, "blk.0.ffn_norm.weight"),
        .ffn_gate = try loadLayerWeight(allocator, tensor_store, compute, "blk.0.ffn_gate.weight"),
        .ffn_up = try loadLayerWeight(allocator, tensor_store, compute, "blk.0.ffn_up.weight"),
        .ffn_down = try loadLayerWeight(allocator, tensor_store, compute, "blk.0.ffn_down.weight"),
        .post_ffw_norm = try loadLayerWeight(allocator, tensor_store, compute, "blk.0.post_ffw_norm.weight"),
        .layer_output_scale = try loadLayerWeight(allocator, tensor_store, compute, "blk.0.layer_output_scale.weight"),
    };
}

fn loadLayerWeight(
    allocator: std.mem.Allocator,
    tensor_store: tensor_store_mod.TensorStore,
    compute: *cuda_compute.CudaCompute,
    name: []const u8,
) ![]f32 {
    var tensor_ref = try tensor_store.describeTensor(allocator, name);
    defer tensor_ref.deinit(allocator);
    var loaded = try tensor_store.loadTensorRef(&tensor_ref);
    defer loaded.deinit();
    if (loaded.tensor.dtype != .f32) return error.UnsupportedTensorType;
    const host = try allocator.dupe(f32, loaded.tensor.asFloat32());
    errdefer allocator.free(host);
    const key = try allocator.dupe(u8, name);
    var inserted = false;
    errdefer if (!inserted) allocator.free(key);
    try compute.insertWeightFromLoaded(key, &loaded);
    inserted = true;
    return host;
}

fn compareCudaStage(
    allocator: std.mem.Allocator,
    cb: *const ops.ComputeBackend,
    stage: []const u8,
    tensor: ops.CT,
    expected: []const f32,
    tolerance: f32,
) !void {
    const actual = try cb.toFloat32(tensor, allocator);
    defer allocator.free(actual);
    if (actual.len != expected.len) return error.InvalidTensorShape;
    const stats = diffStats(actual, expected);
    print(
        "gemma4_parity: {s} max_abs={d:.6} mean_abs={d:.6} max_index={d}\n",
        .{ stage, stats.max_abs, stats.mean_abs, stats.max_index },
    );
    if (stats.max_abs > tolerance) {
        print(
            "gemma4_parity: first_divergence stage={s} got={d:.6} expected={d:.6}\n",
            .{ stage, actual[stats.max_index], expected[stats.max_index] },
        );
        return error.CudaParityMismatch;
    }
}

fn makeParityInput(allocator: std.mem.Allocator, in_dim: usize) ![]f32 {
    const input = try allocator.alloc(f32, in_dim);
    for (input, 0..) |*value, idx| {
        const centered: i32 = @as(i32, @intCast(idx % 251)) - 125;
        const sign: f32 = if ((idx / 251) % 2 == 0) 1.0 else -1.0;
        value.* = sign * @as(f32, @floatFromInt(centered)) / 125.0;
    }
    return input;
}

fn cpuLinearNoBias(input: []const f32, weight: []const f32, output: []f32, in_dim: usize, out_dim: usize) void {
    for (0..out_dim) |row| {
        var sum: f32 = 0;
        const w_row = weight[row * in_dim ..][0..in_dim];
        for (0..in_dim) |col| {
            sum += input[col] * w_row[col];
        }
        output[row] = sum;
    }
}

fn cpuLinearNoBiasRows(input: []const f32, weight: []const f32, output: []f32, rows: usize, in_dim: usize, out_dim: usize) void {
    for (0..rows) |row| {
        const in_row = input[row * in_dim ..][0..in_dim];
        const out_row = output[row * out_dim ..][0..out_dim];
        cpuLinearNoBias(in_row, weight, out_row, in_dim, out_dim);
    }
}

fn cpuRmsNormRows(input: []const f32, weight: []const f32, output: []f32, rows: usize, dim: usize, eps: f32) void {
    for (0..rows) |row| {
        const src = input[row * dim ..][0..dim];
        const dst = output[row * dim ..][0..dim];
        var sumsq: f32 = 0;
        for (src) |value| sumsq += value * value;
        const scale = 1.0 / @sqrt(sumsq / @as(f32, @floatFromInt(dim)) + eps);
        for (src, dst, 0..) |value, *out, idx| out.* = value * scale * weight[idx];
    }
}

fn cpuRmsNormBareRows(input: []const f32, output: []f32, rows: usize, dim: usize, eps: f32) void {
    for (0..rows) |row| {
        const src = input[row * dim ..][0..dim];
        const dst = output[row * dim ..][0..dim];
        var sumsq: f32 = 0;
        for (src) |value| sumsq += value * value;
        const scale = 1.0 / @sqrt(sumsq / @as(f32, @floatFromInt(dim)) + eps);
        for (src, dst) |value, *out| out.* = value * scale;
    }
}

fn cpuAdd(a: []const f32, b: []const f32, output: []f32) void {
    for (a, b, output) |av, bv, *out| out.* = av + bv;
}

fn cpuAddScaled(a: []const f32, b: []const f32, scale: f32, output: []f32) void {
    for (a, b, output) |av, bv, *out| out.* = av * scale + bv;
}

fn cpuSiluMultiply(gate: []const f32, up: []const f32, output: []f32) void {
    for (gate, up, output) |g, u, *out| out.* = (g / (1.0 + @exp(-g))) * u;
}

fn cpuRmsNormHeadsRope(
    input: []const f32,
    weight: []const f32,
    output: []f32,
    rows: usize,
    total_dim: usize,
    head_dim: usize,
    rope_dim: usize,
    eps: f32,
    theta: f32,
    freq_scale: f32,
    position_offset: usize,
    seq_len: usize,
    consecutive_pairs: bool,
    scale: f32,
) void {
    const total_chunks = rows * (total_dim / head_dim);
    const chunks_per_position = total_chunks / seq_len;
    for (0..total_chunks) |chunk| {
        const base = chunk * head_dim;
        var sumsq: f32 = 0;
        for (0..head_dim) |i| {
            const x = input[base + i];
            sumsq += x * x;
        }
        const norm_scale = 1.0 / @sqrt(sumsq / @as(f32, @floatFromInt(head_dim)) + eps);
        for (0..head_dim) |d| {
            var value = input[base + d] * norm_scale * weight[d];
            if (d < rope_dim and rope_dim >= 2) {
                var idx0: usize = 0;
                var idx1: usize = 0;
                var pair_index: usize = 0;
                var second = false;
                if (consecutive_pairs) {
                    pair_index = d / 2;
                    idx0 = pair_index * 2;
                    idx1 = idx0 + 1;
                    second = (d & 1) != 0;
                } else {
                    const half = rope_dim / 2;
                    if (d < half) {
                        pair_index = d;
                        idx0 = d;
                        idx1 = d + half;
                    } else {
                        pair_index = d - half;
                        idx0 = d - half;
                        idx1 = d;
                        second = true;
                    }
                }
                if (idx1 < head_dim) {
                    const token_pos = (chunk / chunks_per_position) % seq_len;
                    const position = position_offset + token_pos;
                    const exponent = @as(f32, @floatFromInt(2 * pair_index)) / @as(f32, @floatFromInt(rope_dim));
                    const angle = @as(f32, @floatFromInt(position)) * freq_scale * (1.0 / std.math.pow(f32, theta, exponent));
                    const s = @sin(angle);
                    const c = @cos(angle);
                    const x0 = input[base + idx0] * norm_scale * weight[idx0];
                    const x1 = input[base + idx1] * norm_scale * weight[idx1];
                    value = if (second) x0 * s + x1 * c else x0 * c - x1 * s;
                }
            }
            output[base + d] = value * scale;
        }
    }
}

fn cpuGqaCausalAttention(
    q: []const f32,
    k: []const f32,
    v: []const f32,
    output: []f32,
    batch: usize,
    seq_len: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
) void {
    const heads_per_kv = num_heads / num_kv_heads;
    const q_row_dim = num_heads * head_dim;
    const kv_row_dim = num_kv_heads * head_dim;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    for (0..batch) |b| {
        for (0..seq_len) |qi| {
            for (0..num_heads) |h| {
                const kv_head = h / heads_per_kv;
                var max_score: f32 = -std.math.inf(f32);
                var scores: [16]f32 = undefined;
                for (0..seq_len) |ki| {
                    if (ki > qi) {
                        scores[ki] = -std.math.inf(f32);
                        continue;
                    }
                    var dot: f32 = 0;
                    const q_base = (b * seq_len + qi) * q_row_dim + h * head_dim;
                    const k_base = (b * seq_len + ki) * kv_row_dim + kv_head * head_dim;
                    for (0..head_dim) |d| dot += q[q_base + d] * k[k_base + d];
                    const score = dot * scale;
                    scores[ki] = score;
                    max_score = @max(max_score, score);
                }
                var denom: f32 = 0;
                for (0..seq_len) |ki| {
                    if (ki > qi) continue;
                    const e = @exp(scores[ki] - max_score);
                    scores[ki] = e;
                    denom += e;
                }
                const out_base = (b * seq_len + qi) * q_row_dim + h * head_dim;
                for (0..head_dim) |d| {
                    var sum: f32 = 0;
                    for (0..seq_len) |ki| {
                        if (ki > qi) continue;
                        const v_base = (b * seq_len + ki) * kv_row_dim + kv_head * head_dim;
                        sum += (scores[ki] / denom) * v[v_base + d];
                    }
                    output[out_base + d] = sum;
                }
            }
        }
    }
}

fn diffStats(actual: []const f32, expected: []const f32) DiffStats {
    var stats: DiffStats = .{};
    var sum_abs: f64 = 0;
    for (actual, 0..) |got, idx| {
        const diff = @abs(got - expected[idx]);
        sum_abs += diff;
        if (diff > stats.max_abs) {
            stats.max_abs = diff;
            stats.max_index = idx;
        }
    }
    stats.mean_abs = sum_abs / @as(f64, @floatFromInt(actual.len));
    return stats;
}

fn toleranceForParityCase(quantized: bool) f32 {
    return if (quantized) 0.05 else 0.001;
}
