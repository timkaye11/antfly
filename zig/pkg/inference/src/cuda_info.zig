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
const weight_source_mod = @import("models/weight_source.zig");

const print = std.debug.print;

const cuda_context = if (build_options.enable_cuda) @import("ops/cuda/context.zig") else struct {};
const cuda_compute = if (build_options.enable_cuda) @import("ops/cuda/cuda_compute.zig") else struct {};
const cuda_kernels = if (build_options.enable_cuda) @import("ops/cuda/kernels.zig") else struct {};
const cuda_buffer = if (build_options.enable_cuda) @import("ops/cuda/buffer.zig") else struct {};
const cuda_cublaslt = if (build_options.enable_cuda) @import("ops/cuda/cublaslt.zig") else struct {};

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
    var gemma4_hf_parity_path: ?[]const u8 = null;
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
        } else if (std.mem.eql(u8, arg, "--gemma4-hf-parity")) {
            i += 1;
            if (i >= args.len) {
                print("missing value for --gemma4-hf-parity\n", .{});
                printUsage();
                std.process.exit(1);
            }
            gemma4_hf_parity_path = args[i];
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
            const cublaslt_ok = smokeCublasLtBf16(allocator) catch |err| {
                print("smoke: cublaslt_bf16 failed\nreason: {s}\n", .{@errorName(err)});
                std.process.exit(1);
            };
            if (cublaslt_ok) {
                print("smoke: cublaslt_bf16 ok\n", .{});
            } else {
                print("smoke: cublaslt_bf16 skipped\n", .{});
            }
        }

        if (gemma4_parity_path) |path| {
            runGemma4Parity(allocator, path) catch |err| {
                print("gemma4_parity: failed\nreason: {s}\n", .{@errorName(err)});
                std.process.exit(1);
            };
            print("gemma4_parity: ok\n", .{});
        }
        if (gemma4_hf_parity_path) |path| {
            runGemma4HfParity(allocator, path) catch |err| {
                print("gemma4_hf_parity: failed\nreason: {s}\n", .{@errorName(err)});
                std.process.exit(1);
            };
            print("gemma4_hf_parity: ok\n", .{});
        }
    }
}

fn smokeCublasLtBf16(allocator: std.mem.Allocator) !bool {
    if (comptime !build_options.enable_cuda) return false;
    var ctx = try cuda_context.CudaContext.initDefault();
    defer ctx.deinit();
    if (ctx.info.compute_major < 8) return false;

    var module = try cuda_kernels.KernelModule.load(&ctx);
    defer module.unload(&ctx);
    var blas = cuda_cublaslt.CublasLt.open() catch return false;
    defer blas.deinit();

    const rows: usize = 2;
    const in_dim: usize = 16;
    const out_dim: usize = 16;
    var input_data: [rows * in_dim]f32 = undefined;
    for (&input_data, 0..) |*value, i| value.* = @floatFromInt((i % 7) + 1);
    var weight_data: [out_dim * in_dim]u16 = .{0} ** (out_dim * in_dim);
    for (0..out_dim) |row| weight_data[row * in_dim + row] = bf16Bits(1.0);

    var input = try cuda_buffer.DeviceBuffer.alloc(&ctx, input_data.len * @sizeOf(f32));
    defer input.free(&ctx);
    var input_bf16 = try cuda_buffer.DeviceBuffer.alloc(&ctx, input_data.len * @sizeOf(u16));
    defer input_bf16.free(&ctx);
    var weight = try cuda_buffer.DeviceBuffer.alloc(&ctx, weight_data.len * @sizeOf(u16));
    defer weight.free(&ctx);
    var output = try cuda_buffer.DeviceBuffer.alloc(&ctx, rows * out_dim * @sizeOf(f32));
    defer output.free(&ctx);

    try input.copyFromHost(&ctx, std.mem.sliceAsBytes(&input_data));
    try weight.copyFromHost(&ctx, std.mem.sliceAsBytes(&weight_data));
    try module.launchF32ToBf16(&ctx, input_bf16, input, input_data.len);
    try blas.matmulBf16WeightF32Out(&ctx, output, input_bf16, weight, rows, in_dim, out_dim);
    try ctx.synchronize();

    const actual = try allocator.alloc(f32, rows * out_dim);
    defer allocator.free(actual);
    try output.copyToHost(&ctx, std.mem.sliceAsBytes(actual));
    try ctx.synchronize();
    for (actual, input_data) |got, expected| {
        if (@abs(got - expected) > 0.0001) return error.CudaParityMismatch;
    }
    return true;
}

fn bf16Bits(value: f32) u16 {
    return @intCast(@as(u32, @bitCast(value)) >> 16);
}

fn printUsage() void {
    print(
        \\usage: antfly inference cuda-info [--smoke] [--gemma4-parity <gguf>] [--gemma4-hf-parity <model-dir>]
        \\
        \\  --smoke   Run embedded PTX smoke checks for fill, dense f32 ops, Q8_0, Q4_0, Q4_K, RoPE, and GQA.
        \\  --gemma4-parity <gguf>
        \\            Compare real Gemma 4 GGUF projection tensors on CUDA against CPU dequantized matmul.
        \\  --gemma4-hf-parity <model-dir>
        \\            Compare real Gemma 4 HF safetensors tensors on CUDA against CPU BF16->F32 reference math.
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
    try runGemma4ParityOnStore(allocator, store.tensorStore());
}

fn runGemma4HfParity(allocator: std.mem.Allocator, model_dir: []const u8) !void {
    const safetensors_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors" });
    defer allocator.free(safetensors_path);
    const single = tensor_store_mod.SafetensorsStore.initAbsolute(allocator, safetensors_path) catch |single_err| blk: {
        if (single_err != error.FileNotFound and single_err != error.NotFound and single_err != error.AccessDenied) return single_err;
        break :blk null;
    };
    if (single) |store| {
        defer store.tensorStore().deinit();
        try runGemma4ParityOnStore(allocator, store.tensorStore());
        return;
    }

    const index_path = try std.fs.path.join(allocator, &.{ model_dir, "model.safetensors.index.json" });
    defer allocator.free(index_path);
    const sharded = try tensor_store_mod.ShardedSafetensorsStore.initAbsolute(allocator, index_path);
    defer sharded.tensorStore().deinit();
    try runGemma4ParityOnStore(allocator, sharded.tensorStore());
}

fn runGemma4ParityOnStore(allocator: std.mem.Allocator, tensor_store: tensor_store_mod.TensorStore) !void {
    for (gemma4_parity_cases) |case| {
        try runGemma4LinearParityCase(allocator, tensor_store, case);
    }
    try runGemma4NormRopeParity(allocator, tensor_store);
    try runGqaParity(allocator);
    try runGemma4Layer0Parity(allocator, tensor_store);
    try runGemma4Layer5AttentionParity(allocator, tensor_store);
    if (tensor_store.kind() == .safetensors) try runGemma4FinalProjectionParity(allocator, tensor_store);
}

fn runGemma4LinearParityCase(
    allocator: std.mem.Allocator,
    tensor_store: tensor_store_mod.TensorStore,
    case: Gemma4ParityCase,
) !void {
    const source_name = try resolveGemma4ParityTensorName(allocator, tensor_store, case.name);
    defer allocator.free(source_name);
    var tensor_ref = try tensor_store.describeTensor(allocator, source_name);
    defer tensor_ref.deinit(allocator);

    var loaded = try tensor_store.loadTensorRef(&tensor_ref);
    defer loaded.deinit();
    if (loaded.tensor.shape.len != 2) return error.InvalidTensorShape;

    const out_dim: usize = @intCast(loaded.tensor.shape[0]);
    const in_dim: usize = @intCast(loaded.tensor.shape[1]);
    const rows: usize = 1;
    const input = try makeParityInput(allocator, in_dim);
    defer allocator.free(input);

    const expected = try allocator.alloc(f32, out_dim);
    defer allocator.free(expected);
    const weight_host = try tensorToFloat32Owned(allocator, &loaded.tensor);
    defer allocator.free(weight_host);
    if (loaded.tensor.dtype == .bf16)
        cpuLinearNoBiasBf16Input(input, weight_host, expected, in_dim, out_dim)
    else
        cpuLinearNoBias(input, weight_host, expected, in_dim, out_dim);

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
    const tolerance = toleranceForLoadedParityCase(&loaded);
    const quant_name = if (loaded.quantized_storage) |storage| storage.tensor_type.name() else "F32";
    print(
        "gemma4_parity: {s} source={s} type={s} dtype={s} shape=[{d},{d}] max_abs={d:.6} mean_abs={d:.6} max_index={d}\n",
        .{ case.name, source_name, quant_name, @tagName(loaded.tensor.dtype), out_dim, in_dim, stats.max_abs, stats.mean_abs, stats.max_index },
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
    const source_name = try resolveGemma4ParityTensorName(allocator, tensor_store, case.weight_name);
    defer allocator.free(source_name);
    var tensor_ref = try tensor_store.describeTensor(allocator, source_name);
    defer tensor_ref.deinit(allocator);

    var loaded = try tensor_store.loadTensorRef(&tensor_ref);
    defer loaded.deinit();
    if (loaded.tensor.elementCount() != case.head_dim) return error.InvalidTensorShape;
    const weight_host = try tensorToFloat32Owned(allocator, &loaded.tensor);
    defer allocator.free(weight_host);

    const count = case.rows * case.total_dim;
    const input = try makeParityInput(allocator, count);
    defer allocator.free(input);

    const expected = try allocator.alloc(f32, count);
    defer allocator.free(expected);
    cpuRmsNormHeadsRope(
        input,
        weight_host,
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
        "gemma4_parity: rms_norm_heads_rope layer={d} proj={s} source={s} shape=[{d},{d}] max_abs={d:.6} mean_abs={d:.6} max_index={d}\n",
        .{ case.layer, case.proj, source_name, case.rows, case.total_dim, stats.max_abs, stats.mean_abs, stats.max_index },
    );
    if (stats.max_abs > toleranceForLoadedElementwiseCase(&loaded)) return error.CudaParityMismatch;
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
    const bf16_linear_inputs = tensor_store.kind() == .safetensors;
    const elem_tolerance = elementwiseStageTolerance(tensor_store, 0.001);
    const linear_tolerance = linearStageTolerance(tensor_store, 0.001);
    const ffn_linear_tolerance = linearStageTolerance(tensor_store, 0.005);
    const activation_tolerance = activationStageTolerance(tensor_store, 0.005);
    const ffn_down_tolerance = ffnDownStageTolerance(tensor_store, 0.005);
    const ffn_post_tolerance = ffnPostStageTolerance(tensor_store, 0.001);

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
    try compareCudaStage(allocator, &cb, "layer0.attn_norm", attn_norm_ct, expected_attn_norm, elem_tolerance);

    const expected_q = try allocator.alloc(f32, rows * q_dim);
    defer allocator.free(expected_q);
    const expected_k = try allocator.alloc(f32, rows * kv_dim);
    defer allocator.free(expected_k);
    const expected_v = try allocator.alloc(f32, rows * kv_dim);
    defer allocator.free(expected_v);
    cpuLinearNoBiasRowsForCudaDense(bf16_linear_inputs, expected_attn_norm, weights.q, expected_q, rows, hidden, q_dim);
    cpuLinearNoBiasRowsForCudaDense(bf16_linear_inputs, expected_attn_norm, weights.k, expected_k, rows, hidden, kv_dim);
    cpuLinearNoBiasRowsForCudaDense(bf16_linear_inputs, expected_attn_norm, weights.v, expected_v, rows, hidden, kv_dim);
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
    try compareCudaStage(allocator, &cb, "layer0.q", q_ct, expected_q, linear_tolerance);
    try compareCudaStage(allocator, &cb, "layer0.k", k_ct, expected_k, linear_tolerance);
    try compareCudaStage(allocator, &cb, "layer0.v", v_ct, expected_v, linear_tolerance);

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
    cpuLinearNoBiasRowsForCudaDense(bf16_linear_inputs, expected_attn, weights.attn_output, expected_attn_proj, rows, q_dim, hidden);
    const attn_out_w = try cb.getWeight("blk.0.attn_output.weight");
    const attn_proj_ct = try cb.linearNoBias(attn_ct, attn_out_w, rows, q_dim, hidden);
    defer cb.free(attn_proj_ct);
    try compareCudaStage(allocator, &cb, "layer0.attn_proj", attn_proj_ct, expected_attn_proj, linear_tolerance);

    const expected_attn_post = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_attn_post);
    cpuRmsNormRows(expected_attn_proj, weights.post_attention_norm, expected_attn_post, rows, hidden, eps);
    const post_attn_w = try cb.getWeight("blk.0.post_attention_norm.weight");
    const attn_post_ct = try cb.rmsNorm(attn_proj_ct, post_attn_w, hidden, eps);
    defer cb.free(attn_post_ct);
    try compareCudaStage(allocator, &cb, "layer0.attn_post", attn_post_ct, expected_attn_post, elem_tolerance);

    const expected_sa = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_sa);
    cpuAdd(expected_attn_post, input, expected_sa);
    const sa_ct = try cb.add(attn_post_ct, input_ct);
    defer cb.free(sa_ct);
    try compareCudaStage(allocator, &cb, "layer0.attn_residual", sa_ct, expected_sa, elem_tolerance);

    const expected_ffn_norm = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_ffn_norm);
    cpuRmsNormRows(expected_sa, weights.ffn_norm, expected_ffn_norm, rows, hidden, eps);
    const ffn_norm_w = try cb.getWeight("blk.0.ffn_norm.weight");
    const ffn_norm_ct = try cb.rmsNorm(sa_ct, ffn_norm_w, hidden, eps);
    defer cb.free(ffn_norm_ct);
    try compareCudaStage(allocator, &cb, "layer0.ffn_norm", ffn_norm_ct, expected_ffn_norm, elem_tolerance);

    const expected_gate = try allocator.alloc(f32, rows * inter);
    defer allocator.free(expected_gate);
    const expected_up = try allocator.alloc(f32, rows * inter);
    defer allocator.free(expected_up);
    const expected_gated = try allocator.alloc(f32, rows * inter);
    defer allocator.free(expected_gated);
    cpuLinearNoBiasRowsForCudaDense(bf16_linear_inputs, expected_ffn_norm, weights.ffn_gate, expected_gate, rows, hidden, inter);
    cpuLinearNoBiasRowsForCudaDense(bf16_linear_inputs, expected_ffn_norm, weights.ffn_up, expected_up, rows, hidden, inter);
    cpuSiluMultiply(expected_gate, expected_up, expected_gated);
    const gate_w = try cb.getWeight("blk.0.ffn_gate.weight");
    const up_w = try cb.getWeight("blk.0.ffn_up.weight");
    const gate_up_ct = try cb.linearNoBiasPair(ffn_norm_ct, gate_w, up_w, rows, hidden, inter);
    const gate_ct = gate_up_ct.first;
    defer cb.free(gate_ct);
    const up_ct = gate_up_ct.second;
    defer cb.free(up_ct);
    try compareCudaStage(allocator, &cb, "layer0.ffn_gate", gate_ct, expected_gate, ffn_linear_tolerance);
    try compareCudaStage(allocator, &cb, "layer0.ffn_up", up_ct, expected_up, ffn_linear_tolerance);
    const gated_ct = (try cb.activationMultiply(gate_ct, up_ct, .silu)) orelse return error.CudaKernelUnavailable;
    defer cb.free(gated_ct);
    try compareCudaStage(allocator, &cb, "layer0.ffn_gated", gated_ct, expected_gated, activation_tolerance);

    const expected_ffn_raw = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_ffn_raw);
    cpuLinearNoBiasRowsForCudaDense(bf16_linear_inputs, expected_gated, weights.ffn_down, expected_ffn_raw, rows, inter, hidden);
    const down_w = try cb.getWeight("blk.0.ffn_down.weight");
    const ffn_raw_ct = try cb.linearNoBias(gated_ct, down_w, rows, inter, hidden);
    defer cb.free(ffn_raw_ct);
    try compareCudaStage(allocator, &cb, "layer0.ffn_raw", ffn_raw_ct, expected_ffn_raw, ffn_down_tolerance);

    const expected_ffn_post = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_ffn_post);
    cpuRmsNormRows(expected_ffn_raw, weights.post_ffw_norm, expected_ffn_post, rows, hidden, eps);
    const post_ffw_w = try cb.getWeight("blk.0.post_ffw_norm.weight");
    const ffn_post_ct = try cb.rmsNorm(ffn_raw_ct, post_ffw_w, hidden, eps);
    defer cb.free(ffn_post_ct);
    try compareCudaStage(allocator, &cb, "layer0.ffn_post", ffn_post_ct, expected_ffn_post, ffn_post_tolerance);

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
    try compareCudaStage(allocator, &cb, "layer0.out", out_ct, expected_out, ffn_post_tolerance);

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
    const bf16_linear_inputs = tensor_store.kind() == .safetensors;
    const elem_tolerance = elementwiseStageTolerance(tensor_store, 0.001);
    const linear_tolerance = linearStageTolerance(tensor_store, 0.005);

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
    try compareCudaStage(allocator, &cb, "layer5.attn_norm", attn_norm_ct, expected_attn_norm, elem_tolerance);

    const expected_q = try allocator.alloc(f32, rows * q_dim);
    defer allocator.free(expected_q);
    const expected_k = try allocator.alloc(f32, rows * kv_dim);
    defer allocator.free(expected_k);
    cpuLinearNoBiasRowsForCudaDense(bf16_linear_inputs, expected_attn_norm, q_w_host, expected_q, rows, hidden, q_dim);
    cpuLinearNoBiasRowsForCudaDense(bf16_linear_inputs, expected_attn_norm, k_w_host, expected_k, rows, hidden, kv_dim);
    const q_w = try cb.getWeight("blk.5.attn_q.weight");
    const k_w = try cb.getWeight("blk.5.attn_k.weight");
    const q_ct = try cb.linearNoBias(attn_norm_ct, q_w, rows, hidden, q_dim);
    defer cb.free(q_ct);
    const k_ct = try cb.linearNoBias(attn_norm_ct, k_w, rows, hidden, kv_dim);
    defer cb.free(k_ct);
    try compareCudaStage(allocator, &cb, "layer5.q", q_ct, expected_q, linear_tolerance);
    try compareCudaStage(allocator, &cb, "layer5.k", k_ct, expected_k, linear_tolerance);

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
    cpuLinearNoBiasRowsForCudaDense(bf16_linear_inputs, expected_attn, attn_output, expected_attn_proj, rows, q_dim, hidden);
    const attn_out_w = try cb.getWeight("blk.5.attn_output.weight");
    const attn_proj_ct = try cb.linearNoBias(attn_ct, attn_out_w, rows, q_dim, hidden);
    defer cb.free(attn_proj_ct);
    try compareCudaStage(allocator, &cb, "layer5.attn_proj", attn_proj_ct, expected_attn_proj, linear_tolerance);

    const expected_attn_post = try allocator.alloc(f32, rows * hidden);
    defer allocator.free(expected_attn_post);
    cpuRmsNormRows(expected_attn_proj, post_attention_norm, expected_attn_post, rows, hidden, eps);
    const post_attn_w = try cb.getWeight("blk.5.post_attention_norm.weight");
    const attn_post_ct = try cb.rmsNorm(attn_proj_ct, post_attn_w, hidden, eps);
    defer cb.free(attn_post_ct);
    try compareCudaStage(allocator, &cb, "layer5.attn_post", attn_post_ct, expected_attn_post, elem_tolerance);

    print("gemma4_parity: layer5_attention_omitted_v ok\n", .{});
}

fn runGemma4FinalProjectionParity(
    allocator: std.mem.Allocator,
    tensor_store: tensor_store_mod.TensorStore,
) !void {
    const hidden: usize = 3840;
    const vocab: usize = 262144;
    const eps: f32 = 0.000001;
    const sampled_ids = [_]usize{ 0, 1, 2, 50, 100, 101, 106, 236761, 258882, 258883 };

    var compute = try cuda_compute.CudaCompute.init(allocator);
    defer compute.deinit();
    const norm = try loadLayerWeight(allocator, tensor_store, &compute, "output_norm.weight");
    defer allocator.free(norm);

    const embed_source_name = try resolveGemma4ParityTensorName(allocator, tensor_store, "token_embd.weight");
    defer allocator.free(embed_source_name);
    var embed_ref = try tensor_store.describeTensor(allocator, embed_source_name);
    defer embed_ref.deinit(allocator);
    var embed_loaded = try tensor_store.loadTensorRef(&embed_ref);
    defer embed_loaded.deinit();
    if (embed_loaded.tensor.dtype != .bf16) return error.UnsupportedTensorType;
    if (embed_loaded.tensor.shape.len != 2) return error.InvalidTensorShape;
    if (@as(usize, @intCast(embed_loaded.tensor.shape[0])) != vocab) return error.InvalidTensorShape;
    if (@as(usize, @intCast(embed_loaded.tensor.shape[1])) != hidden) return error.InvalidTensorShape;
    const embed_key = try allocator.dupe(u8, "token_embd.weight");
    var inserted = false;
    errdefer if (!inserted) allocator.free(embed_key);
    try compute.insertWeightFromLoaded(embed_key, &embed_loaded);
    inserted = true;

    const cb = compute.computeBackend();
    const input = try makeParityInput(allocator, hidden);
    defer allocator.free(input);
    const expected_norm = try allocator.alloc(f32, hidden);
    defer allocator.free(expected_norm);
    cpuRmsNormRows(input, norm, expected_norm, 1, hidden, eps);

    const input_shape = [_]i32{ 1, @intCast(hidden) };
    const input_ct = try cb.fromFloat32Shape(input, &input_shape);
    defer cb.free(input_ct);
    const norm_w = try cb.getWeight("output_norm.weight");
    const norm_ct = try cb.rmsNorm(input_ct, norm_w, hidden, eps);
    defer cb.free(norm_ct);
    try compareCudaStage(allocator, &cb, "final.norm", norm_ct, expected_norm, 0.005);

    const embed_w = try cb.getWeight("token_embd.weight");
    const logits_ct = try cb.linearNoBias(norm_ct, embed_w, 1, hidden, vocab);
    defer cb.free(logits_ct);
    const actual_logits = try cb.toFloat32(logits_ct, allocator);
    defer allocator.free(actual_logits);
    if (actual_logits.len != vocab) return error.InvalidTensorShape;

    var max_abs: f32 = 0;
    var mean_abs: f64 = 0;
    var max_token: usize = 0;
    var max_expected: f32 = 0;
    var max_actual: f32 = 0;
    for (sampled_ids) |token_id| {
        const expected = cpuEmbeddingRowDotBf16Input(expected_norm, embed_loaded.tensor.data, token_id, hidden);
        const actual = actual_logits[token_id];
        const diff = @abs(actual - expected);
        mean_abs += diff;
        if (diff > max_abs) {
            max_abs = diff;
            max_token = token_id;
            max_expected = expected;
            max_actual = actual;
        }
    }
    mean_abs /= @as(f64, @floatFromInt(sampled_ids.len));
    print(
        "gemma4_parity: final.tied_lm_head sampled={d} max_abs={d:.6} mean_abs={d:.6} token={d}\n",
        .{ sampled_ids.len, max_abs, mean_abs, max_token },
    );
    if (max_abs > 0.02) {
        print(
            "gemma4_parity: first_divergence stage=final.tied_lm_head got={d:.6} expected={d:.6}\n",
            .{ max_actual, max_expected },
        );
        return error.CudaParityMismatch;
    }
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
    const source_name = try resolveGemma4ParityTensorName(allocator, tensor_store, name);
    defer allocator.free(source_name);
    var tensor_ref = try tensor_store.describeTensor(allocator, source_name);
    defer tensor_ref.deinit(allocator);
    var loaded = try tensor_store.loadTensorRef(&tensor_ref);
    defer loaded.deinit();
    const host = try tensorToFloat32Owned(allocator, &loaded.tensor);
    errdefer allocator.free(host);
    const key = try allocator.dupe(u8, name);
    var inserted = false;
    errdefer if (!inserted) allocator.free(key);
    try compute.insertWeightFromLoaded(key, &loaded);
    inserted = true;
    return host;
}

fn tensorToFloat32Owned(allocator: std.mem.Allocator, tensor: *const @import("backends/tensor.zig").Tensor) ![]f32 {
    if (tensor.dtype == .f32) return try allocator.dupe(f32, tensor.asFloat32());
    if (tensor.dtype == .f16 or tensor.dtype == .bf16) {
        var converted = try weight_source_mod.convertToF32(allocator, tensor);
        defer converted.deinit();
        return try allocator.dupe(f32, converted.asFloat32());
    }
    return error.UnsupportedTensorType;
}

fn resolveGemma4ParityTensorName(
    allocator: std.mem.Allocator,
    tensor_store: tensor_store_mod.TensorStore,
    gguf_name: []const u8,
) ![]u8 {
    if (tensor_store.kind() != .safetensors) return try allocator.dupe(u8, gguf_name);
    if (std.mem.eql(u8, gguf_name, "output_norm.weight")) return try allocator.dupe(u8, "model.language_model.norm.weight");
    if (std.mem.eql(u8, gguf_name, "token_embd.weight")) return try allocator.dupe(u8, "model.language_model.embed_tokens.weight");
    if (!std.mem.startsWith(u8, gguf_name, "blk.")) return try allocator.dupe(u8, gguf_name);

    const rest = gguf_name["blk.".len..];
    const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return error.InvalidTensorName;
    const layer = rest[0..dot];
    const suffix = rest[dot + 1 ..];
    const hf_suffix = gemma4HfLayerSuffixForGguf(suffix) orelse return error.TensorNotFound;
    return try std.fmt.allocPrint(allocator, "model.language_model.layers.{s}.{s}", .{ layer, hf_suffix });
}

fn gemma4HfLayerSuffixForGguf(suffix: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, suffix, "attn_norm.weight")) return "input_layernorm.weight";
    if (std.mem.eql(u8, suffix, "attn_q.weight")) return "self_attn.q_proj.weight";
    if (std.mem.eql(u8, suffix, "attn_k.weight")) return "self_attn.k_proj.weight";
    if (std.mem.eql(u8, suffix, "attn_v.weight")) return "self_attn.v_proj.weight";
    if (std.mem.eql(u8, suffix, "attn_output.weight")) return "self_attn.o_proj.weight";
    if (std.mem.eql(u8, suffix, "attn_q_norm.weight")) return "self_attn.q_norm.weight";
    if (std.mem.eql(u8, suffix, "attn_k_norm.weight")) return "self_attn.k_norm.weight";
    if (std.mem.eql(u8, suffix, "post_attention_norm.weight")) return "post_attention_layernorm.weight";
    if (std.mem.eql(u8, suffix, "ffn_norm.weight")) return "pre_feedforward_layernorm.weight";
    if (std.mem.eql(u8, suffix, "ffn_gate.weight")) return "mlp.gate_proj.weight";
    if (std.mem.eql(u8, suffix, "ffn_up.weight")) return "mlp.up_proj.weight";
    if (std.mem.eql(u8, suffix, "ffn_down.weight")) return "mlp.down_proj.weight";
    if (std.mem.eql(u8, suffix, "post_ffw_norm.weight")) return "post_feedforward_layernorm.weight";
    if (std.mem.eql(u8, suffix, "layer_output_scale.weight")) return "layer_scalar";
    return null;
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

fn cpuLinearNoBiasBf16Input(input: []const f32, weight: []const f32, output: []f32, in_dim: usize, out_dim: usize) void {
    for (0..out_dim) |row| {
        var sum: f32 = 0;
        const w_row = weight[row * in_dim ..][0..in_dim];
        for (0..in_dim) |col| {
            sum += roundF32ToBf16(input[col]) * w_row[col];
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

fn cpuLinearNoBiasRowsForCudaDense(
    bf16_linear_inputs: bool,
    input: []const f32,
    weight: []const f32,
    output: []f32,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
) void {
    for (0..rows) |row| {
        const in_row = input[row * in_dim ..][0..in_dim];
        const out_row = output[row * out_dim ..][0..out_dim];
        if (bf16_linear_inputs) {
            cpuLinearNoBiasBf16Input(in_row, weight, out_row, in_dim, out_dim);
        } else {
            cpuLinearNoBias(in_row, weight, out_row, in_dim, out_dim);
        }
    }
}

fn roundF32ToBf16(value: f32) f32 {
    const bits: u32 = @bitCast(value);
    const rounded = bits + 0x8000;
    return @bitCast(rounded & 0xffff0000);
}

fn cpuEmbeddingRowDotBf16Input(input: []const f32, bf16_weight_bytes: []const u8, row: usize, dim: usize) f32 {
    var sum: f32 = 0;
    const row_offset = row * dim;
    for (0..dim) |col| {
        const bits = std.mem.readInt(u16, bf16_weight_bytes[(row_offset + col) * @sizeOf(u16) ..][0..@sizeOf(u16)], .little);
        const weight = bf16BitsToF32(bits);
        sum += roundF32ToBf16(input[col]) * weight;
    }
    return sum;
}

fn bf16BitsToF32(bits: u16) f32 {
    return @bitCast(@as(u32, bits) << 16);
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

fn toleranceForLoadedParityCase(loaded: *const weight_source_mod.LoadedWeight) f32 {
    if (loaded.quantized) return 0.05;
    return switch (loaded.tensor.dtype) {
        .bf16, .f16 => 0.25,
        else => 0.001,
    };
}

fn toleranceForLoadedElementwiseCase(loaded: *const weight_source_mod.LoadedWeight) f32 {
    return switch (loaded.tensor.dtype) {
        .bf16, .f16 => 0.005,
        else => 0.001,
    };
}

fn elementwiseStageTolerance(tensor_store: tensor_store_mod.TensorStore, base: f32) f32 {
    return if (tensor_store.kind() == .safetensors) @max(base, 0.005) else base;
}

fn linearStageTolerance(tensor_store: tensor_store_mod.TensorStore, base: f32) f32 {
    return if (tensor_store.kind() == .safetensors) @max(base, 0.02) else base;
}

fn activationStageTolerance(tensor_store: tensor_store_mod.TensorStore, base: f32) f32 {
    return if (tensor_store.kind() == .safetensors) @max(base, 0.25) else base;
}

fn ffnDownStageTolerance(tensor_store: tensor_store_mod.TensorStore, base: f32) f32 {
    return if (tensor_store.kind() == .safetensors) @max(base, 0.5) else base;
}

fn ffnPostStageTolerance(tensor_store: tensor_store_mod.TensorStore, base: f32) f32 {
    return if (tensor_store.kind() == .safetensors) @max(base, 0.25) else base;
}
