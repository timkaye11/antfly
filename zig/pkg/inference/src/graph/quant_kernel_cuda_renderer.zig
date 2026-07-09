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

//! Typed renderer for generated CUDA quant kernels.
//!
//! A `RenderPlan` owns the route identity and the complete launch contract.
//! Rendering composes common metadata, include directives, helper fragments,
//! and one kernel body. Production-bundle validation consumes the same helper
//! and body fragments, so standalone candidates and promoted kernels cannot
//! silently diverge in support code.

const std = @import("std");
const quant_matmul = @import("quant_matmul.zig");
const quant_kernel_op = @import("quant_kernel_op.zig");

pub const EpilogueKind = quant_kernel_op.Epilogue;

pub const KernelKind = enum {
    q4_k_small_batch_bias_gelu,
    q4_0_mmv,
    q4_0_mm,
    q4_0_pair_mmv,
    q4_0_pair_activation_q8_1,
    q4_0_down_q8_1,
};

pub const GridMapping = enum {
    /// grid.x selects an output-column tile; grid.y selects an input row tile.
    output_cols_by_rows,
    /// grid.x selects an output-column tile. The route owns one input row.
    output_cols,
    /// grid.x flattens `(input row, output-column block)`.
    flattened_rows_by_output_blocks,
    /// grid.x flattens `(batch, query row, attention head)`.
    batch_query_heads,
};

pub const DimensionConstraint = struct {
    min: u32 = 0,
    max: u32 = 0,
    multiple_of: u32 = 1,
    fixed: u32 = 0,

    pub fn validate(self: DimensionConstraint) !void {
        if (self.multiple_of == 0) return error.InvalidDimensionMultiple;
        if (self.max != 0 and self.min > self.max) return error.InvalidDimensionBounds;
        if (self.fixed != 0) {
            if (self.fixed < self.min) return error.FixedDimensionBelowMinimum;
            if (self.max != 0 and self.fixed > self.max) return error.FixedDimensionAboveMaximum;
            if (self.fixed % self.multiple_of != 0) return error.FixedDimensionMisaligned;
        }
    }
};

/// Exact launch ABI for a generated CUDA kernel. Unlike the generic compiler
/// schedule, this records specialized blocks such as the 640-thread fused FFN
/// route and its 32-output tile.
pub const LaunchMetadata = struct {
    grid: GridMapping,
    threads_per_block: u16,
    output_rows_per_block: u16,
    output_cols_per_block: u16,
    static_shared_memory_bytes: u32,
    /// All current generated CUDA families use static `__shared__` arrays.
    /// Future attention/tiling plans must record their launch-time allocation.
    dynamic_shared_memory_bytes: u32 = 0,
    rows: DimensionConstraint,
    input_dim: DimensionConstraint,
    output_dim: DimensionConstraint,

    pub fn validate(self: LaunchMetadata) !void {
        if (self.threads_per_block == 0 or self.threads_per_block > 1024 or self.threads_per_block % 32 != 0) {
            return error.InvalidCudaThreadCount;
        }
        if (self.output_rows_per_block == 0 or self.output_cols_per_block == 0) {
            return error.InvalidCudaOutputTile;
        }
        try self.rows.validate();
        try self.input_dim.validate();
        try self.output_dim.validate();
    }
};

pub const Route = struct {
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: EpilogueKind,
    dispatch: quant_matmul.DispatchKind,
};

pub const QuantDecodePrimitive = enum {
    q4_k_scale_min,
    q4_0_f16_nibbles,
    q4_0_q8_1_dp4a,
};

pub const ReductionPrimitive = enum {
    shared_tree,
    warp_then_shared,
    warp_sum_and_max,
};

pub const Q4KSmallBatchLowering = struct {
    decode: QuantDecodePrimitive = .q4_k_scale_min,
    reduction: ReductionPrimitive = .shared_tree,
    block_values: u16 = 256,
};

/// Parameterized lowering shared by q4_0 MMV, 8-row MM, and paired MMV.
/// `row_tile` and `projection_count` drive every accumulator/shared-memory
/// dimension and output write in the generated body.
pub const Q4ProjectionLowering = struct {
    decode: QuantDecodePrimitive = .q4_0_f16_nibbles,
    reduction: ReductionPrimitive = .warp_then_shared,
    row_tile: u8,
    projection_count: u8,
    columns_per_block: u8 = 4,
    warps_per_block: u8 = 8,

    pub fn validate(self: Q4ProjectionLowering) !void {
        if (self.decode != .q4_0_f16_nibbles) return error.InvalidQ4ProjectionDecode;
        if (self.reduction != .warp_then_shared) return error.InvalidQ4ProjectionReduction;
        if (self.row_tile != 1 and self.row_tile != 8) return error.InvalidQ4ProjectionRowTile;
        if (self.projection_count != 1 and self.projection_count != 2) return error.InvalidQ4ProjectionCount;
        if (self.row_tile > 1 and self.projection_count != 1) return error.UnsupportedQ4ProjectionShape;
        if (self.columns_per_block != 4 or self.warps_per_block != 8) return error.InvalidQ4ProjectionTopology;
    }
};

pub const PairActivationQ8Lowering = struct {
    decode: QuantDecodePrimitive = .q4_0_q8_1_dp4a,
    reduction: ReductionPrimitive = .warp_sum_and_max,
    input_blocks: u16,
    output_blocks: u16,
    output_values_per_block: u8,
    warps_per_projection_group: u8,
};

pub const DownQ8Lowering = struct {
    decode: QuantDecodePrimitive = .q4_0_q8_1_dp4a,
    reduction: ReductionPrimitive = .warp_then_shared,
    input_blocks: u16,
    columns_per_block: u8,
};

pub const KernelLowering = union(enum) {
    q4_k_small_batch: Q4KSmallBatchLowering,
    q4_0_projection: Q4ProjectionLowering,
    q4_0_pair_activation_q8_1: PairActivationQ8Lowering,
    q4_0_down_q8_1: DownQ8Lowering,

    pub fn validate(self: KernelLowering) !void {
        switch (self) {
            .q4_k_small_batch => |lowering| {
                if (lowering.decode != .q4_k_scale_min or lowering.reduction != .shared_tree or lowering.block_values != 256) {
                    return error.InvalidQ4KLowering;
                }
            },
            .q4_0_projection => |lowering| try lowering.validate(),
            .q4_0_pair_activation_q8_1 => |lowering| {
                if (lowering.decode != .q4_0_q8_1_dp4a or lowering.reduction != .warp_sum_and_max or
                    lowering.input_blocks != 80 or lowering.output_blocks != 320 or
                    lowering.output_values_per_block != 32 or lowering.warps_per_projection_group != 5)
                {
                    return error.InvalidPairActivationQ8Lowering;
                }
            },
            .q4_0_down_q8_1 => |lowering| {
                if (lowering.decode != .q4_0_q8_1_dp4a or lowering.reduction != .warp_then_shared or
                    lowering.input_blocks != 320 or lowering.columns_per_block != 4)
                {
                    return error.InvalidDownQ8Lowering;
                }
            },
        }
    }
};

pub const RenderPlan = struct {
    kind: KernelKind,
    route: Route,
    kernel_id: []const u8,
    production_baseline: []const u8,
    production_enabled: bool,
    launch: LaunchMetadata,
    lowering: KernelLowering,

    pub fn validate(self: RenderPlan) !void {
        try self.launch.validate();
        try self.lowering.validate();
        const expected = planFor(self.kind);
        if (!std.meta.eql(self.route, expected.route)) return error.CudaRouteDoesNotMatchKind;
        if (!std.mem.eql(u8, self.kernel_id, expected.kernel_id)) return error.CudaKernelIdDoesNotMatchKind;
        if (!std.mem.eql(u8, self.production_baseline, expected.production_baseline)) return error.CudaBaselineDoesNotMatchKind;
        if (self.production_enabled != expected.production_enabled) return error.CudaPromotionStateDoesNotMatchKind;
        if (!std.meta.eql(self.launch, expected.launch)) return error.CudaLaunchDoesNotMatchKind;
        if (!std.meta.eql(self.lowering, expected.lowering)) return error.CudaLoweringDoesNotMatchKind;
    }
};

pub const AttentionKernelKind = enum {
    gqa_decode_scalars_hd256,
};

pub const AttentionReduction = enum {
    shared_tree,
};

pub const AttentionSoftmax = enum {
    stable_two_pass,
};

pub const AttentionLowering = struct {
    kind: quant_kernel_op.AttentionKind,
    reduction: AttentionReduction,
    softmax: AttentionSoftmax,
    head_dim: u16,
    device_decode_scalars: bool,

    pub fn validate(self: AttentionLowering) !void {
        if (self.kind != .decode_1x or self.reduction != .shared_tree or
            self.softmax != .stable_two_pass or self.head_dim != 256 or
            !self.device_decode_scalars)
        {
            return error.InvalidCudaAttentionLowering;
        }
    }
};

/// Typed plan for generated CUDA attention. It is intentionally separate from
/// the quant-matmul `RenderPlan`: attention has no quant format, row bucket, or
/// epilogue, and must not acquire placeholder values for those fields.
pub const AttentionRenderPlan = struct {
    kind: AttentionKernelKind,
    kernel_id: []const u8,
    production_baseline: []const u8,
    production_enabled: bool,
    launch: LaunchMetadata,
    lowering: AttentionLowering,

    pub fn validate(self: AttentionRenderPlan) !void {
        try self.launch.validate();
        try self.lowering.validate();
        const expected = attentionPlanFor(self.kind);
        if (!std.mem.eql(u8, self.kernel_id, expected.kernel_id)) return error.CudaKernelIdDoesNotMatchKind;
        if (!std.mem.eql(u8, self.production_baseline, expected.production_baseline)) return error.CudaBaselineDoesNotMatchKind;
        if (self.production_enabled != expected.production_enabled) return error.CudaPromotionStateDoesNotMatchKind;
        if (!std.meta.eql(self.launch, expected.launch)) return error.CudaLaunchDoesNotMatchKind;
        if (!std.meta.eql(self.lowering, expected.lowering)) return error.CudaLoweringDoesNotMatchKind;
    }
};

pub const SourceFragment = struct {
    name: []const u8,
    source: []const u8,
};

pub const KernelSupport = struct {
    includes: []const []const u8,
    helpers: []const SourceFragment,
};

const license_header =
    \\// Copyright 2026 Antfly, Inc.
    \\//
    \\// Licensed under the Apache License, Version 2.0 (the "License");
    \\// you may not use this file except in compliance with the License.
    \\// You may obtain a copy of the License at
    \\//
    \\//     http://www.apache.org/licenses/LICENSE-2.0
    \\//
    \\// Unless required by applicable law or agreed to in writing, software
    \\// distributed under the License is distributed on an "AS IS" BASIS,
    \\// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    \\// See the License for the specific language governing permissions and
    \\// limitations under the License.
;

const includes_cuda_fp16_stdint = [_][]const u8{ "<cuda_fp16.h>", "<stdint.h>" };
const includes_cuda_fp16_math_stdint = [_][]const u8{ "<cuda_fp16.h>", "<math.h>", "<stdint.h>" };
const includes_cuda_fp16 = [_][]const u8{"<cuda_fp16.h>"};
const includes_math = [_][]const u8{"<math.h>"};

pub fn planFor(kind: KernelKind) RenderPlan {
    return switch (kind) {
        .q4_k_small_batch_bias_gelu => .{
            .kind = kind,
            .route = .{ .format = .q4_k, .row_bucket = .rows_2_8, .epilogue = .bias_gelu, .dispatch = .small_batch },
            .kernel_id = "antfly_q4_k_small_batch_bias_gelu_f32_v1",
            .production_baseline = "termite_linear_q4_k_bias_gelu_f32_tile4_r2",
            .production_enabled = false,
            .launch = .{
                .grid = .output_cols_by_rows,
                .threads_per_block = 128,
                .output_rows_per_block = 1,
                .output_cols_per_block = 1,
                .static_shared_memory_bytes = 512,
                .rows = .{ .min = 2, .max = 8 },
                .input_dim = .{ .multiple_of = 256 },
                .output_dim = .{},
            },
            .lowering = .{ .q4_k_small_batch = .{} },
        },
        .q4_0_mmv => .{
            .kind = kind,
            .route = .{ .format = .q4_0, .row_bucket = .rows_1, .epilogue = .none, .dispatch = .mmv },
            .kernel_id = "antfly_q4_0_mmv_f32_v1",
            .production_baseline = "termite_linear_q4_0_f32_tile4",
            .production_enabled = true,
            .launch = .{
                .grid = .output_cols,
                .threads_per_block = 256,
                .output_rows_per_block = 1,
                .output_cols_per_block = 4,
                .static_shared_memory_bytes = 128,
                .rows = .{ .min = 1, .max = 1, .fixed = 1 },
                .input_dim = .{ .multiple_of = 32 },
                .output_dim = .{},
            },
            .lowering = .{ .q4_0_projection = .{ .row_tile = 1, .projection_count = 1 } },
        },
        .q4_0_mm => .{
            .kind = kind,
            .route = .{ .format = .q4_0, .row_bucket = .rows_9_64, .epilogue = .none, .dispatch = .mm },
            .kernel_id = "antfly_q4_0_mm_f32_v1",
            .production_baseline = "termite_linear_q4_0_f32",
            .production_enabled = true,
            .launch = .{
                .grid = .output_cols_by_rows,
                .threads_per_block = 256,
                .output_rows_per_block = 8,
                .output_cols_per_block = 4,
                .static_shared_memory_bytes = 1024,
                .rows = .{ .min = 9, .max = 64 },
                .input_dim = .{ .multiple_of = 32 },
                .output_dim = .{},
            },
            .lowering = .{ .q4_0_projection = .{ .row_tile = 8, .projection_count = 1 } },
        },
        .q4_0_pair_mmv => .{
            .kind = kind,
            .route = .{ .format = .q4_0, .row_bucket = .rows_1, .epilogue = .pair, .dispatch = .mmv },
            .kernel_id = "antfly_q4_0_pair_mmv_f32_v1",
            .production_baseline = "termite_linear_q4_0_pair_nobias_f32_tile4_w4",
            .production_enabled = true,
            .launch = .{
                .grid = .output_cols,
                .threads_per_block = 256,
                .output_rows_per_block = 1,
                .output_cols_per_block = 4,
                .static_shared_memory_bytes = 256,
                .rows = .{ .min = 1, .max = 1, .fixed = 1 },
                .input_dim = .{ .multiple_of = 32 },
                .output_dim = .{},
            },
            .lowering = .{ .q4_0_projection = .{ .row_tile = 1, .projection_count = 2 } },
        },
        .q4_0_pair_activation_q8_1 => .{
            .kind = kind,
            .route = .{ .format = .q4_0, .row_bucket = .rows_1, .epilogue = .pair_activation, .dispatch = .mmv },
            .kernel_id = "antfly_q4_0_pair_activation_q8_1_mmv_v1",
            .production_baseline = "termite_linear_q4_0_pair_activation_q8_1_q8_1_tile32_w5_e4b_ffn",
            .production_enabled = true,
            .launch = .{
                .grid = .flattened_rows_by_output_blocks,
                .threads_per_block = 640,
                .output_rows_per_block = 1,
                .output_cols_per_block = 32,
                .static_shared_memory_bytes = 768,
                .rows = .{ .min = 1 },
                .input_dim = .{ .multiple_of = 32, .fixed = 2560 },
                .output_dim = .{ .multiple_of = 32, .fixed = 10240 },
            },
            .lowering = .{ .q4_0_pair_activation_q8_1 = .{ .input_blocks = 80, .output_blocks = 320, .output_values_per_block = 32, .warps_per_projection_group = 5 } },
        },
        .q4_0_down_q8_1 => .{
            .kind = kind,
            .route = .{ .format = .q4_0, .row_bucket = .rows_1, .epilogue = .gated_down, .dispatch = .mmv },
            .kernel_id = "antfly_q4_0_down_q8_1_mmv_v1",
            .production_baseline = "termite_linear_q4_0_q8_1_f32_tile4_w8_e4b_down",
            .production_enabled = true,
            .launch = .{
                .grid = .output_cols,
                .threads_per_block = 256,
                .output_rows_per_block = 1,
                .output_cols_per_block = 4,
                .static_shared_memory_bytes = 128,
                .rows = .{ .min = 1, .max = 1, .fixed = 1 },
                .input_dim = .{ .multiple_of = 32, .fixed = 10240 },
                .output_dim = .{},
            },
            .lowering = .{ .q4_0_down_q8_1 = .{ .input_blocks = 320, .columns_per_block = 4 } },
        },
    };
}

pub fn attentionPlanFor(kind: AttentionKernelKind) AttentionRenderPlan {
    return switch (kind) {
        .gqa_decode_scalars_hd256 => .{
            .kind = kind,
            .kernel_id = "antfly_gqa_attention_decode_scalars_hd256_f32_v1",
            .production_baseline = "termite_gqa_attention_decode_scalars_f32",
            .production_enabled = false,
            .launch = .{
                .grid = .batch_query_heads,
                .threads_per_block = 256,
                .output_rows_per_block = 1,
                .output_cols_per_block = 256,
                .static_shared_memory_bytes = 1032,
                .rows = .{ .min = 1, .max = 1, .fixed = 1 },
                .input_dim = .{ .min = 256, .max = 256, .fixed = 256 },
                .output_dim = .{},
            },
            .lowering = .{
                .kind = .decode_1x,
                .reduction = .shared_tree,
                .softmax = .stable_two_pass,
                .head_dim = 256,
                .device_decode_scalars = true,
            },
        },
    };
}

fn sourceKind(plan: RenderPlan) []const u8 {
    return if (plan.production_enabled)
        "Promoted generated kernel"
    else
        "Dev-only generated kernel candidate";
}

fn promotionComment(kind: KernelKind) []const u8 {
    return switch (kind) {
        .q4_k_small_batch_bias_gelu =>
        \\// Not compiled into production artifacts until correctness and benchmark gates
        \\// beat the handwritten CUDA baseline.
        ,
        .q4_0_mmv =>
        \\// Promoted on sequential benchmark evidence vs the handwritten CUDA baseline;
        \\// runtime dispatch is default-on behind ANTFLY_INFERENCE_CUDA_DISABLE_GENERATED_Q4_0_MMV.
        ,
        .q4_0_mm =>
        \\// Promoted on sequential benchmark evidence vs the handwritten CUDA baseline;
        \\// runtime dispatch is default-on behind ANTFLY_INFERENCE_CUDA_DISABLE_GENERATED_Q4_0_MM.
        ,
        .q4_0_pair_mmv =>
        \\// Promoted on sequential benchmark evidence vs the handwritten CUDA baseline;
        \\// runtime dispatch is default-on behind ANTFLY_INFERENCE_CUDA_DISABLE_GENERATED_Q4_0_PAIR.
        ,
        .q4_0_pair_activation_q8_1 =>
        \\// Promoted on sequential benchmark evidence vs the handwritten CUDA baseline;
        \\// runtime dispatch is default-on behind ANTFLY_INFERENCE_CUDA_DISABLE_GENERATED_Q4_0_PAIR_Q8.
        ,
        .q4_0_down_q8_1 =>
        \\// Promoted on sequential benchmark evidence vs the handwritten CUDA baseline;
        \\// runtime dispatch is default-on behind ANTFLY_INFERENCE_CUDA_DISABLE_GENERATED_Q4_0_DOWN_Q8.
        ,
    };
}

pub fn planId(plan: RenderPlan, allocator: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(allocator, "cuda/{s}/{s}/{s}/{s}", .{
        @tagName(plan.route.format),
        @tagName(plan.route.row_bucket),
        @tagName(plan.route.epilogue),
        @tagName(plan.route.dispatch),
    });
}

pub fn renderKernel(allocator: std.mem.Allocator, plan: RenderPlan) ![]u8 {
    try plan.validate();
    const support = supportFor(plan.lowering);
    const route_id = try planId(plan, allocator);
    defer allocator.free(route_id);

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, license_header);
    try out.appendSlice(allocator, "\n\n// ");
    try out.appendSlice(allocator, sourceKind(plan));
    try out.appendSlice(allocator, " from graph/quant_kernel_compiler.zig.\n// plan_id=");
    try out.appendSlice(allocator, route_id);
    try out.appendSlice(allocator, "\n// kernel_id=");
    try out.appendSlice(allocator, plan.kernel_id);
    try out.appendSlice(allocator, "\n// production_baseline=");
    try out.appendSlice(allocator, plan.production_baseline);
    try out.appendSlice(allocator, if (plan.production_enabled) "\n// production_enabled=true\n" else "\n// production_enabled=false\n");
    try out.appendSlice(allocator, promotionComment(plan.kind));
    try out.appendSlice(allocator, "\n\n");
    for (support.includes) |include| {
        try out.appendSlice(allocator, "#include ");
        try out.appendSlice(allocator, include);
        try out.append(allocator, '\n');
    }
    try out.append(allocator, '\n');
    for (support.helpers) |helper| {
        try out.appendSlice(allocator, helper.source);
        try out.appendSlice(allocator, "\n\n");
    }
    try renderBody(allocator, &out, plan);
    if (out.items.len == 0 or out.items[out.items.len - 1] != '\n') try out.append(allocator, '\n');
    return out.toOwnedSlice(allocator);
}

pub fn attentionPlanId(plan: AttentionRenderPlan, allocator: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(allocator, "cuda/attention/{s}/hd{d}/device_scalars", .{
        @tagName(plan.lowering.kind),
        plan.lowering.head_dim,
    });
}

pub fn renderAttentionKernel(allocator: std.mem.Allocator, plan: AttentionRenderPlan) ![]u8 {
    try plan.validate();
    const route_id = try attentionPlanId(plan, allocator);
    defer allocator.free(route_id);

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.appendSlice(allocator, license_header);
    try out.appendSlice(allocator, "\n\n// Dev-only generated attention kernel from graph/quant_kernel_compiler.zig.\n// plan_id=");
    try out.appendSlice(allocator, route_id);
    try out.appendSlice(allocator, "\n// kernel_id=");
    try out.appendSlice(allocator, plan.kernel_id);
    try out.appendSlice(allocator, "\n// production_baseline=");
    try out.appendSlice(allocator, plan.production_baseline);
    try out.appendSlice(allocator, "\n// production_enabled=false\n");
    try out.appendSlice(allocator, "// Runtime opt-in: ANTFLY_INFERENCE_CUDA_GENERATED_ATTENTION_DECODE=1.\n\n");
    for (includes_math) |include| {
        try out.appendSlice(allocator, "#include ");
        try out.appendSlice(allocator, include);
        try out.append(allocator, '\n');
    }
    try out.append(allocator, '\n');
    try out.appendSlice(allocator, body_attention_decode_scalars_hd256.source);
    try out.append(allocator, '\n');
    return out.toOwnedSlice(allocator);
}

const helper_half_le_to_float = SourceFragment{
    .name = "antfly_half_le_to_float",
    .source =
    \\static __device__ __forceinline__ float antfly_half_le_to_float(const uint8_t *p) {
    \\    const uint16_t bits = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    \\    return __half2float(__ushort_as_half(bits));
    \\}
    ,
};

const helper_warp_reduce_sum = SourceFragment{
    .name = "antfly_warp_reduce_sum",
    .source =
    \\static __device__ __forceinline__ float antfly_warp_reduce_sum(float value) {
    \\    value += __shfl_down_sync(0xffffffffu, value, 16);
    \\    value += __shfl_down_sync(0xffffffffu, value, 8);
    \\    value += __shfl_down_sync(0xffffffffu, value, 4);
    \\    value += __shfl_down_sync(0xffffffffu, value, 2);
    \\    value += __shfl_down_sync(0xffffffffu, value, 1);
    \\    return value;
    \\}
    ,
};

const helper_q4_k_block_view = SourceFragment{
    .name = "antfly_q4_k_block_view",
    .source =
    \\struct antfly_q4_k_block_view {
    \\    const uint8_t *d;
    \\    const uint8_t *dmin;
    \\    const uint8_t *scales;
    \\    const uint8_t *qs;
    \\};
    ,
};

const helper_gelu = SourceFragment{
    .name = "antfly_gelu",
    .source =
    \\static __device__ __forceinline__ float antfly_gelu(float x) {
    \\    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    \\    return 0.5f * x * (1.0f + tanhf(inner));
    \\}
    ,
};

const helper_q4_k_unpack_scale_min = SourceFragment{
    .name = "antfly_q4_k_unpack_scale_min",
    .source =
    \\static __device__ __forceinline__ void antfly_q4_k_unpack_scale_min(
    \\    const uint8_t *scales,
    \\    int sub,
    \\    float *scale,
    \\    float *min_v
    \\) {
    \\    if (sub < 4) {
    \\        *scale = (float)(scales[sub] & 63u);
    \\        *min_v = (float)(scales[sub + 4] & 63u);
    \\        return;
    \\    }
    \\
    \\    *scale = (float)((scales[sub + 4] & 0x0fu) | ((scales[sub - 4] >> 6) << 4));
    \\    *min_v = (float)((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
    \\}
    ,
};

const helper_q4_k_dequant_lane = SourceFragment{
    .name = "antfly_q4_k_dequant_lane",
    .source =
    \\static __device__ __forceinline__ float antfly_q4_k_dequant_lane(const uint8_t *block, int lane) {
    \\    antfly_q4_k_block_view view = {
    \\        block,
    \\        block + 2,
    \\        block + 4,
    \\        block + 16,
    \\    };
    \\    const int sub = lane >> 5;
    \\    const int q_index = (sub >> 1) * 32 + (lane & 31);
    \\    const uint8_t packed = view.qs[q_index];
    \\    const uint8_t q = (sub & 1) == 0 ? (packed & 0x0fu) : (packed >> 4);
    \\    const float d = antfly_half_le_to_float(view.d);
    \\    const float dmin = antfly_half_le_to_float(view.dmin);
    \\    float raw_scale = 0.0f;
    \\    float raw_min = 0.0f;
    \\    antfly_q4_k_unpack_scale_min(view.scales, sub, &raw_scale, &raw_min);
    \\    return d * raw_scale * (float)q - dmin * raw_min;
    \\}
    ,
};

const q4_k_helpers = [_]SourceFragment{
    helper_q4_k_block_view,
    helper_half_le_to_float,
    helper_gelu,
    helper_q4_k_unpack_scale_min,
    helper_q4_k_dequant_lane,
};

const q4_0_helpers = [_]SourceFragment{ helper_half_le_to_float, helper_warp_reduce_sum };

const helper_half_bits_to_float = SourceFragment{
    .name = "antfly_half_bits_to_float",
    .source =
    \\static __device__ __forceinline__ float antfly_half_bits_to_float(unsigned short bits) {
    \\    return __half2float(__ushort_as_half(bits));
    \\}
    ,
};

const helper_warp_reduce_sum_f32 = SourceFragment{
    .name = "antfly_warp_reduce_sum_f32",
    .source =
    \\static __device__ __forceinline__ float antfly_warp_reduce_sum_f32(float value) {
    \\    value += __shfl_down_sync(0xffffffffu, value, 16);
    \\    value += __shfl_down_sync(0xffffffffu, value, 8);
    \\    value += __shfl_down_sync(0xffffffffu, value, 4);
    \\    value += __shfl_down_sync(0xffffffffu, value, 2);
    \\    value += __shfl_down_sync(0xffffffffu, value, 1);
    \\    return value;
    \\}
    ,
};

const helper_warp_reduce_max_f32 = SourceFragment{
    .name = "antfly_warp_reduce_max_f32",
    .source =
    \\static __device__ __forceinline__ float antfly_warp_reduce_max_f32(float value) {
    \\    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 16));
    \\    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 8));
    \\    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 4));
    \\    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 2));
    \\    value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, 1));
    \\    return __shfl_sync(0xffffffffu, value, 0);
    \\}
    ,
};

const helper_decoder_activation_f32 = SourceFragment{
    .name = "antfly_decoder_activation_f32",
    .source =
    \\static __device__ __forceinline__ float antfly_decoder_activation_f32(float x, unsigned int activation) {
    \\    if (activation <= 1u) {
    \\        const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    \\        return 0.5f * x * (1.0f + tanhf(inner));
    \\    }
    \\    if (activation == 2u) return x / (1.0f + __expf(-x));
    \\    if (activation == 3u) return fmaxf(x, 0.0f);
    \\    if (activation == 4u) return x / (1.0f + __expf(-1.702f * x));
    \\    const float r = fmaxf(x, 0.0f);
    \\    return r * r;
    \\}
    ,
};

const helper_q4_0_word_u16 = SourceFragment{
    .name = "antfly_q4_0_word_u16",
    .source =
    \\// q4_0 payload bytes live at bp+2; bp is always 2-byte aligned (18-byte
    \\// blocks), so every 4-byte word can be assembled from two aligned u16 loads.
    \\static __device__ __forceinline__ unsigned int antfly_q4_0_word_u16(const unsigned char *payload) {
    \\    const unsigned short *halves = (const unsigned short *)payload;
    \\    return (unsigned int)halves[0] | ((unsigned int)halves[1] << 16);
    \\}
    ,
};

const helper_q4_0_word_u16_no_comment = SourceFragment{
    .name = "antfly_q4_0_word_u16",
    .source =
    \\static __device__ __forceinline__ unsigned int antfly_q4_0_word_u16(const unsigned char *payload) {
    \\    const unsigned short *halves = (const unsigned short *)payload;
    \\    return (unsigned int)halves[0] | ((unsigned int)halves[1] << 16);
    \\}
    ,
};

const helper_q4_0_q8_dot16 = SourceFragment{
    .name = "antfly_q4_0_q8_dot16",
    .source =
    \\static __device__ __forceinline__ float antfly_q4_0_q8_dot16(
    \\    const unsigned char *q4_bp,
    \\    float q8_d,
    \\    unsigned int iqs,
    \\    int q8_low0,
    \\    int q8_high0,
    \\    int q8_low1,
    \\    int q8_high1
    \\) {
    \\    const float q4_d = antfly_half_bits_to_float(((const unsigned short *)q4_bp)[0]);
    \\    const unsigned int base0 = iqs * 4u;
    \\    const unsigned int word0 = antfly_q4_0_word_u16(q4_bp + 2u + base0);
    \\    const unsigned int word1 = antfly_q4_0_word_u16(q4_bp + 2u + base0 + 4u);
    \\    const unsigned int low0 = __vadd4(word0 & 0x0f0f0f0fu, 0xf8f8f8f8u);
    \\    const unsigned int high0 = __vadd4((word0 >> 4) & 0x0f0f0f0fu, 0xf8f8f8f8u);
    \\    const unsigned int low1 = __vadd4(word1 & 0x0f0f0f0fu, 0xf8f8f8f8u);
    \\    const unsigned int high1 = __vadd4((word1 >> 4) & 0x0f0f0f0fu, 0xf8f8f8f8u);
    \\    int sumi = __dp4a((int)low0, q8_low0, 0);
    \\    sumi = __dp4a((int)high0, q8_high0, sumi);
    \\    sumi = __dp4a((int)low1, q8_low1, sumi);
    \\    sumi = __dp4a((int)high1, q8_high1, sumi);
    \\    return q4_d * q8_d * (float)sumi;
    \\}
    ,
};

const q4_0_q8_helpers = [_]SourceFragment{
    helper_half_bits_to_float,
    helper_warp_reduce_sum_f32,
    helper_warp_reduce_max_f32,
    helper_decoder_activation_f32,
    helper_q4_0_word_u16,
    helper_q4_0_q8_dot16,
};

const q4_0_down_q8_helpers = [_]SourceFragment{
    helper_half_bits_to_float,
    helper_warp_reduce_sum_f32,
    helper_q4_0_word_u16_no_comment,
    helper_q4_0_q8_dot16,
};

pub const body_attention_decode_scalars_hd256 = SourceFragment{
    .name = "antfly_gqa_attention_decode_scalars_hd256_f32_v1",
    .source =
    \\extern "C" __global__ void antfly_gqa_attention_decode_scalars_hd256_f32_v1(
    \\    float* dst,
    \\    const float* q,
    \\    const float* k,
    \\    const float* v,
    \\    const unsigned char* attn_or_mask,
    \\    const float* bias,
    \\    unsigned int batch,
    \\    unsigned int q_seq_len,
    \\    unsigned int kv_seq_len,
    \\    unsigned int num_heads,
    \\    unsigned int num_kv_heads,
    \\    unsigned int head_dim,
    \\    unsigned int query_position_offset,
    \\    unsigned int kv_position_offset,
    \\    unsigned int sliding_window,
    \\    unsigned int total_sequence_len,
    \\    unsigned int mask_len,
    \\    unsigned int bias_mode,
    \\    const unsigned int* decode_scalars
    \\) {
    \\    if (decode_scalars != 0) {
    \\        kv_position_offset = decode_scalars[4];
    \\        query_position_offset = decode_scalars[1];
    \\        kv_seq_len = decode_scalars[2];
    \\        total_sequence_len = decode_scalars[3];
    \\    }
    \\
    \\    __shared__ float warp_sums[4];
    \\    __shared__ float shared_max_score;
    \\    __shared__ float shared_denom;
    \\    __shared__ float shared_alpha;
    \\    __shared__ float shared_beta;
    \\    unsigned int lane = threadIdx.x;
    \\    unsigned int block = blockIdx.x;
    \\    unsigned int total_blocks = batch * q_seq_len * num_heads;
    \\    if (block >= total_blocks || q_seq_len != 1u || head_dim != 256u || blockDim.x != 128u || num_kv_heads == 0u || (num_heads % num_kv_heads) != 0u) return;
    \\
    \\    unsigned int head = block % num_heads;
    \\    unsigned int tmp = block / num_heads;
    \\    unsigned int qi = tmp % q_seq_len;
    \\    unsigned int b = tmp / q_seq_len;
    \\    unsigned int heads_per_group = num_heads / num_kv_heads;
    \\    unsigned int kv_head = head / heads_per_group;
    \\    unsigned int q_hidden = num_heads * head_dim;
    \\    unsigned int kv_hidden = num_kv_heads * head_dim;
    \\    unsigned int query_pos = query_position_offset + qi;
    \\    unsigned int q_base = (b * q_seq_len + qi) * q_hidden + head * head_dim;
    \\    float scale = rsqrtf((float)head_dim);
    \\    unsigned int warp = lane >> 5;
    \\    unsigned int warp_lane = lane & 31u;
    \\    if (lane == 0u) {
    \\        shared_max_score = -3.402823466e+38f;
    \\        shared_denom = 0.0f;
    \\    }
    \\    __syncthreads();
    \\
    \\    float acc0 = 0.0f;
    \\    float acc1 = 0.0f;
    \\    for (unsigned int ki = 0; ki < kv_seq_len; ++ki) {
    \\        unsigned int key_pos = kv_position_offset + ki;
    \\        unsigned int mask_idx = query_pos * total_sequence_len + key_pos;
    \\        bool future_allowed = attn_or_mask != 0 && mask_idx < mask_len && attn_or_mask[mask_idx] != 0u;
    \\        bool future_blocked = key_pos > query_pos && !future_allowed;
    \\        bool past_blocked = key_pos > query_pos || (sliding_window != 0u && (query_pos - key_pos) >= sliding_window);
    \\        bool valid = !(future_blocked || past_blocked);
    \\        unsigned int k_base = (b * kv_seq_len + ki) * kv_hidden + kv_head * head_dim;
    \\        float dot = valid ? q[q_base + lane] * k[k_base + lane] + q[q_base + lane + 128u] * k[k_base + lane + 128u] : 0.0f;
    \\        for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
    \\            dot += __shfl_down_sync(0xffffffffu, dot, offset);
    \\        }
    \\        if (warp_lane == 0u) warp_sums[warp] = dot;
    \\        __syncthreads();
    \\        if (warp == 0u) {
    \\            float block_dot = warp_lane < 4u ? warp_sums[warp_lane] : 0.0f;
    \\            for (unsigned int offset = 16u; offset > 0u; offset >>= 1) {
    \\                block_dot += __shfl_down_sync(0xffffffffu, block_dot, offset);
    \\            }
    \\            if (warp_lane == 0u) {
    \\                float score = valid ? block_dot * scale : -3.402823466e+38f;
    \\                if (valid && bias_mode == 1u) score += bias[(head * q_seq_len + qi) * kv_seq_len + ki];
    \\                if (valid && bias_mode == 2u) score += bias[((b * num_heads + head) * q_seq_len + qi) * kv_seq_len + ki];
    \\                float next_max = fmaxf(shared_max_score, score);
    \\                float alpha = shared_denom > 0.0f ? expf(shared_max_score - next_max) : 0.0f;
    \\                float beta = valid ? expf(score - next_max) : 0.0f;
    \\                shared_denom = shared_denom * alpha + beta;
    \\                shared_max_score = next_max;
    \\                shared_alpha = alpha;
    \\                shared_beta = beta;
    \\            }
    \\        }
    \\        __syncthreads();
    \\        unsigned int v_idx = (b * kv_seq_len + ki) * kv_hidden + kv_head * head_dim + lane;
    \\        acc0 = acc0 * shared_alpha + shared_beta * v[v_idx];
    \\        acc1 = acc1 * shared_alpha + shared_beta * v[v_idx + 128u];
    \\    }
    \\
    \\    unsigned int out_idx = (b * q_seq_len + qi) * q_hidden + head * head_dim + lane;
    \\    dst[out_idx] = shared_denom > 0.0f ? acc0 / shared_denom : 0.0f;
    \\    dst[out_idx + 128u] = shared_denom > 0.0f ? acc1 / shared_denom : 0.0f;
    \\}
    ,
};

pub fn renderAttentionBodyAlloc(allocator: std.mem.Allocator, plan: AttentionRenderPlan) ![]u8 {
    try plan.validate();
    return allocator.dupe(u8, body_attention_decode_scalars_hd256.source);
}

const body_q4_k_small_batch_bias_gelu = SourceFragment{
    .name = "antfly_q4_k_small_batch_bias_gelu_f32_v1",
    .source =
    \\extern "C" __global__ void antfly_q4_k_small_batch_bias_gelu_f32_v1(
    \\    const float *input,
    \\    const uint8_t *weight_q4_k,
    \\    const float *bias,
    \\    float *output,
    \\    int rows,
    \\    int in_dim,
    \\    int out_dim
    \\) {
    \\    const int row = blockIdx.y;
    \\    const int col = blockIdx.x;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim) return;
    \\    if (blockDim.x != 128) return;
    \\    if ((in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\        const uint8_t *block = weight_q4_k + ((col * block_count + block_idx) * 144);
    \\        const int base = block_idx << 8;
    \\        for (int lane = threadIdx.x; lane < 256; lane += blockDim.x) {
    \\            acc += input[row * in_dim + base + lane] * antfly_q4_k_dequant_lane(block, lane);
    \\        }
    \\    }
    \\
    \\    __shared__ float partial[128];
    \\    partial[threadIdx.x] = acc;
    \\    __syncthreads();
    \\    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
    \\        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    \\        __syncthreads();
    \\    }
    \\    if (threadIdx.x == 0) output[row * out_dim + col] = antfly_gelu(partial[0] + bias[col]);
    \\}
    ,
};

const body_q4_0_pair_activation_q8_1 = SourceFragment{
    .name = "antfly_q4_0_pair_activation_q8_1_mmv_v1",
    .source =
    \\extern "C" __global__ void antfly_q4_0_pair_activation_q8_1_mmv_v1(
    \\    unsigned char *dst_q8,
    \\    const unsigned char *q8_input,
    \\    const unsigned char *weight_gate,
    \\    const unsigned char *weight_up,
    \\    unsigned int activation,
    \\    unsigned int rows,
    \\    unsigned int in_dim,
    \\    unsigned int out_dim
    \\) {
    \\    if (rows == 0u || (in_dim & 31u) != 0u || (out_dim & 31u) != 0u) return;
    \\    const unsigned int row_blocks = in_dim >> 5;
    \\    const unsigned int out_row_blocks = out_dim >> 5;
    \\    const unsigned int group_cols = 4u;
    \\    const unsigned int groups_per_wave = 4u;
    \\    const unsigned int waves = 2u;
    \\
    \\    const unsigned int out_block = blockIdx.x % out_row_blocks;
    \\    const unsigned int row = blockIdx.x / out_row_blocks;
    \\    const unsigned int col_block = out_block * 32u;
    \\    const unsigned int tid = threadIdx.x;
    \\    const unsigned int lane = tid & 31u;
    \\    const unsigned int warp = tid >> 5u;
    \\    const unsigned int group = warp / 5u;
    \\    const unsigned int group_warp = warp - group * 5u;
    \\    if (blockDim.x != 640u || row >= rows) return;
    \\
    \\    __shared__ float gate_partial[4][4][5];
    \\    __shared__ float up_partial[4][4][5];
    \\    __shared__ float activated[32];
    \\
    \\    #pragma unroll
    \\    for (unsigned int wave = 0u; wave < waves; ++wave) {
    \\        if (group < groups_per_wave) {
    \\            const unsigned int local_tid = group_warp * 32u + lane;
    \\            const unsigned int col_tile = col_block + (wave * groups_per_wave + group) * group_cols;
    \\            float gate_acc[4];
    \\            float up_acc[4];
    \\            #pragma unroll
    \\            for (unsigned int c = 0u; c < group_cols; ++c) {
    \\                gate_acc[c] = 0.0f;
    \\                up_acc[c] = 0.0f;
    \\            }
    \\
    \\            const unsigned int iqs = (local_tid & 1u) * 2u;
    \\            for (unsigned int block = local_tid >> 1u; block < row_blocks; block += 80u) {
    \\                const unsigned char *q8_bp = q8_input + (row * row_blocks + block) * 36u;
    \\                const float q8_d = antfly_half_bits_to_float(((const unsigned short *)q8_bp)[0]);
    \\                const signed char *q8_values = (const signed char *)(q8_bp + 4u);
    \\                const unsigned int q8_base0 = iqs * 4u;
    \\                const unsigned int q8_base1 = q8_base0 + 4u;
    \\                const int q8_low0 = *(const int *)(q8_values + q8_base0);
    \\                const int q8_high0 = *(const int *)(q8_values + q8_base0 + 16u);
    \\                const int q8_low1 = *(const int *)(q8_values + q8_base1);
    \\                const int q8_high1 = *(const int *)(q8_values + q8_base1 + 16u);
    \\
    \\                #pragma unroll
    \\                for (unsigned int c = 0u; c < group_cols; ++c) {
    \\                    const unsigned int col = col_tile + c;
    \\                    const unsigned char *gate_bp = weight_gate + ((size_t)col * row_blocks + block) * 18u;
    \\                    const unsigned char *up_bp = weight_up + ((size_t)col * row_blocks + block) * 18u;
    \\                    gate_acc[c] += antfly_q4_0_q8_dot16(gate_bp, q8_d, iqs, q8_low0, q8_high0, q8_low1, q8_high1);
    \\                    up_acc[c] += antfly_q4_0_q8_dot16(up_bp, q8_d, iqs, q8_low0, q8_high0, q8_low1, q8_high1);
    \\                }
    \\            }
    \\
    \\            #pragma unroll
    \\            for (unsigned int c = 0u; c < group_cols; ++c) {
    \\                const float gate_sum = antfly_warp_reduce_sum_f32(gate_acc[c]);
    \\                const float up_sum = antfly_warp_reduce_sum_f32(up_acc[c]);
    \\                if (lane == 0u) {
    \\                    gate_partial[group][c][group_warp] = gate_sum;
    \\                    up_partial[group][c][group_warp] = up_sum;
    \\                }
    \\            }
    \\        }
    \\        __syncthreads();
    \\        if (tid < 16u) {
    \\            const unsigned int out_group = tid >> 2u;
    \\            const unsigned int c = tid & 3u;
    \\            float gate_y = 0.0f;
    \\            float up_y = 0.0f;
    \\            #pragma unroll
    \\            for (unsigned int w = 0u; w < 5u; ++w) {
    \\                gate_y += gate_partial[out_group][c][w];
    \\                up_y += up_partial[out_group][c][w];
    \\            }
    \\            activated[wave * 16u + out_group * group_cols + c] = antfly_decoder_activation_f32(gate_y, activation) * up_y;
    \\        }
    \\        __syncthreads();
    \\    }
    \\
    \\    if (warp == 0u) {
    \\        const float x = activated[lane];
    \\        const float amax = antfly_warp_reduce_max_f32(fabsf(x));
    \\        const float d = amax > 0.0f ? amax / 127.0f : 0.0f;
    \\        int q = 0;
    \\        if (d > 0.0f) {
    \\            q = __float2int_rn(x / d);
    \\            q = max(-127, min(127, q));
    \\        }
    \\        unsigned char *bp = dst_q8 + ((size_t)row * out_row_blocks + out_block) * 36u;
    \\        bp[4u + lane] = (unsigned char)(signed char)q;
    \\        if (lane == 0u) {
    \\            const unsigned short d_bits = __half_as_ushort(__float2half(d));
    \\            bp[0] = (unsigned char)(d_bits & 0xffu);
    \\            bp[1] = (unsigned char)(d_bits >> 8);
    \\            bp[2] = 0u;
    \\            bp[3] = 0u;
    \\        }
    \\    }
    \\}
    ,
};

const body_q4_0_down_q8_1 = SourceFragment{
    .name = "antfly_q4_0_down_q8_1_mmv_v1",
    .source =
    \\extern "C" __global__ void antfly_q4_0_down_q8_1_mmv_v1(
    \\    float *dst,
    \\    const unsigned char *q8_input,
    \\    const unsigned char *weight,
    \\    unsigned int rows,
    \\    unsigned int in_dim,
    \\    unsigned int out_dim
    \\) {
    \\    const unsigned int cols = 4u;
    \\    if (rows == 0u || (in_dim & 31u) != 0u || out_dim == 0u) return;
    \\    const unsigned int row_blocks = in_dim >> 5;
    \\    const unsigned int col_tiles = (out_dim + cols - 1u) / cols;
    \\    const unsigned int row = blockIdx.x / col_tiles;
    \\    const unsigned int col_tile = (blockIdx.x % col_tiles) * cols;
    \\    const unsigned int tid = threadIdx.x;
    \\    const unsigned int lane = tid & 31u;
    \\    const unsigned int warp = tid >> 5u;
    \\    if (blockDim.x != 256u || row >= rows) return;
    \\
    \\    __shared__ float warp_partial[4][8];
    \\    float acc[4];
    \\    #pragma unroll
    \\    for (unsigned int c = 0u; c < cols; ++c) acc[c] = 0.0f;
    \\
    \\    const unsigned int iqs = (tid & 1u) * 2u;
    \\    for (unsigned int block = tid >> 1u; block < row_blocks; block += 128u) {
    \\        const unsigned char *q8_bp = q8_input + ((size_t)row * row_blocks + block) * 36u;
    \\        const float q8_d = antfly_half_bits_to_float(((const unsigned short *)q8_bp)[0]);
    \\        const signed char *q8_values = (const signed char *)(q8_bp + 4u);
    \\        const unsigned int q8_base0 = iqs * 4u;
    \\        const unsigned int q8_base1 = q8_base0 + 4u;
    \\        const int q8_low0 = *(const int *)(q8_values + q8_base0);
    \\        const int q8_high0 = *(const int *)(q8_values + q8_base0 + 16u);
    \\        const int q8_low1 = *(const int *)(q8_values + q8_base1);
    \\        const int q8_high1 = *(const int *)(q8_values + q8_base1 + 16u);
    \\
    \\        #pragma unroll
    \\        for (unsigned int c = 0u; c < cols; ++c) {
    \\            const unsigned int col = col_tile + c;
    \\            if (col < out_dim) {
    \\                const unsigned char *bp = weight + ((size_t)col * row_blocks + block) * 18u;
    \\                acc[c] += antfly_q4_0_q8_dot16(bp, q8_d, iqs, q8_low0, q8_high0, q8_low1, q8_high1);
    \\            }
    \\        }
    \\    }
    \\
    \\    #pragma unroll
    \\    for (unsigned int c = 0u; c < cols; ++c) {
    \\        const float sum = antfly_warp_reduce_sum_f32(acc[c]);
    \\        if (lane == 0u) warp_partial[c][warp] = sum;
    \\    }
    \\    __syncthreads();
    \\    if (tid < 4u) {
    \\        float y = 0.0f;
    \\        #pragma unroll
    \\        for (unsigned int w = 0u; w < 8u; ++w) y += warp_partial[tid][w];
    \\        const unsigned int col = col_tile + tid;
    \\        if (col < out_dim) dst[(size_t)row * out_dim + col] = y;
    \\    }
    \\}
    ,
};

fn appendFmt(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    comptime fmt: []const u8,
    args: anytype,
) !void {
    const chunk = try std.fmt.allocPrint(allocator, fmt, args);
    defer allocator.free(chunk);
    try out.appendSlice(allocator, chunk);
}

pub fn supportFor(lowering: KernelLowering) KernelSupport {
    return switch (lowering) {
        .q4_k_small_batch => .{ .includes = &includes_cuda_fp16_math_stdint, .helpers = &q4_k_helpers },
        .q4_0_projection => .{ .includes = &includes_cuda_fp16_stdint, .helpers = &q4_0_helpers },
        .q4_0_pair_activation_q8_1 => .{ .includes = &includes_cuda_fp16, .helpers = &q4_0_q8_helpers },
        .q4_0_down_q8_1 => .{ .includes = &includes_cuda_fp16, .helpers = &q4_0_down_q8_helpers },
    };
}

pub fn renderBodyAlloc(allocator: std.mem.Allocator, plan: RenderPlan) ![]u8 {
    try plan.validate();
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try renderBody(allocator, &out, plan);
    return out.toOwnedSlice(allocator);
}

fn renderBody(allocator: std.mem.Allocator, out: *std.ArrayListUnmanaged(u8), plan: RenderPlan) !void {
    switch (plan.lowering) {
        .q4_k_small_batch => try out.appendSlice(allocator, body_q4_k_small_batch_bias_gelu.source),
        .q4_0_projection => |lowering| try renderQ4ProjectionBody(allocator, out, plan.kernel_id, lowering),
        .q4_0_pair_activation_q8_1 => |lowering| try renderPairActivationQ8Body(allocator, out, plan.kernel_id, lowering),
        .q4_0_down_q8_1 => |lowering| try renderDownQ8Body(allocator, out, plan.kernel_id, lowering),
    }
}

fn renderQ4ProjectionBody(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    kernel_id: []const u8,
    lowering: Q4ProjectionLowering,
) !void {
    try lowering.validate();
    const rows = lowering.row_tile;
    const projections = lowering.projection_count;
    const cols = lowering.columns_per_block;
    const warps = lowering.warps_per_block;

    try appendFmt(allocator, out, "extern \"C\" __global__ void {s}(\n", .{kernel_id});
    try out.appendSlice(allocator, "    const float *input,\n");
    if (projections == 1) {
        try out.appendSlice(allocator, "    const uint8_t *weight_q4_0,\n    float *output,\n");
    } else {
        try out.appendSlice(
            allocator,
            "    const uint8_t *weight_a_q4_0,\n" ++
                "    const uint8_t *weight_b_q4_0,\n" ++
                "    float *output_a,\n" ++
                "    float *output_b,\n",
        );
    }
    try out.appendSlice(allocator, "    int rows,\n    int in_dim,\n    int out_dim\n) {\n");
    try appendFmt(allocator, out, "    const int col0 = blockIdx.x << {d};\n", .{@as(u8, 2)});
    if (rows == 1) {
        try out.appendSlice(allocator, "    if (rows != 1 || col0 >= out_dim) return;\n");
    } else {
        try appendFmt(
            allocator,
            out,
            "    const int row0 = blockIdx.y << {d};\n" ++
                "    if (rows < 9 || rows > 64) return;\n" ++
                "    if (col0 >= out_dim || row0 >= rows) return;\n",
            .{@as(u8, 3)},
        );
    }
    try appendFmt(
        allocator,
        out,
        "    if (blockDim.x != {d}) return;\n" ++
            "    if ((in_dim & 31) != 0) return;\n\n" ++
            "    const int row_blocks = in_dim >> 5;\n" ++
            "    const int half_bytes = in_dim >> 1;\n",
        .{@as(u16, warps) * 32},
    );

    if (rows == 1 and projections == 1) {
        try appendFmt(allocator, out, "    float acc[{d}] = {{0.0f, 0.0f, 0.0f, 0.0f}};\n", .{cols});
    } else if (rows == 1) {
        try appendFmt(
            allocator,
            out,
            "    float acc[{d}][{d}];\n" ++
                "#pragma unroll\n" ++
                "    for (int w = 0; w < {d}; ++w) {{\n" ++
                "#pragma unroll\n" ++
                "        for (int c = 0; c < {d}; ++c) acc[w][c] = 0.0f;\n" ++
                "    }}\n",
            .{ projections, cols, projections, cols },
        );
    } else {
        try appendFmt(
            allocator,
            out,
            "    float acc[{d}][{d}];\n" ++
                "#pragma unroll\n" ++
                "    for (int c = 0; c < {d}; ++c) {{\n" ++
                "#pragma unroll\n" ++
                "        for (int r = 0; r < {d}; ++r) acc[c][r] = 0.0f;\n" ++
                "    }}\n",
            .{ cols, rows, cols, rows },
        );
    }

    if (rows != 1 or projections != 1) try out.append(allocator, '\n');
    try appendFmt(
        allocator,
        out,
        "    for (int byte_idx = threadIdx.x; byte_idx < half_bytes; byte_idx += {d}) {{\n" ++
            "        const int block_idx = byte_idx >> 4;\n" ++
            "        const int offset = byte_idx & 15;\n" ++
            "        const int base = block_idx << 5;\n",
        .{@as(u16, warps) * 32},
    );
    if (rows == 1) {
        try out.appendSlice(
            allocator,
            "        const float x_lo = input[base + offset];\n" ++
                "        const float x_hi = input[base + offset + 16];\n",
        );
    } else {
        try appendFmt(
            allocator,
            out,
            "        float x_lo[{d}];\n" ++
                "        float x_hi[{d}];\n" ++
                "#pragma unroll\n" ++
                "        for (int r = 0; r < {d}; ++r) {{\n" ++
                "            const int row = row0 + r;\n" ++
                "            x_lo[r] = row < rows ? input[(size_t)row * in_dim + base + offset] : 0.0f;\n" ++
                "            x_hi[r] = row < rows ? input[(size_t)row * in_dim + base + offset + 16] : 0.0f;\n" ++
                "        }}\n",
            .{ rows, rows, rows },
        );
    }
    if (projections == 2) {
        try appendFmt(
            allocator,
            out,
            "#pragma unroll\n" ++
                "        for (int w = 0; w < {d}; ++w) {{\n" ++
                "            const uint8_t *weight = w == 0 ? weight_a_q4_0 : weight_b_q4_0;\n",
            .{projections},
        );
    }
    const projection_loop_indent = if (projections == 2) "            " else "        ";
    const projection_body_indent = if (projections == 2) "                " else "            ";
    try appendFmt(
        allocator,
        out,
        "#pragma unroll\n" ++
            "{s}for (int c = 0; c < {d}; ++c) {{\n" ++
            "{s}if (col0 + c >= out_dim) continue;\n",
        .{ projection_loop_indent, cols, projection_body_indent },
    );
    const weight_name = if (projections == 2) "weight" else "weight_q4_0";
    try appendFmt(
        allocator,
        out,
        "{s}const uint8_t *block = {s} + ((size_t)(col0 + c) * row_blocks + block_idx) * 18;\n" ++
            "{s}const float d = antfly_half_le_to_float(block);\n" ++
            "{s}const int packed = (int)block[2 + offset];\n",
        .{ projection_body_indent, weight_name, projection_body_indent, projection_body_indent },
    );
    if (rows == 1) {
        if (projections == 1) {
            try out.appendSlice(allocator, "            acc[c] += d * (x_lo * (float)((packed & 15) - 8) + x_hi * (float)((packed >> 4) - 8));\n");
        } else {
            try out.appendSlice(allocator, "                acc[w][c] += d * (x_lo * (float)((packed & 15) - 8) + x_hi * (float)((packed >> 4) - 8));\n");
        }
    } else {
        try out.appendSlice(
            allocator,
            "            const float w_lo = d * (float)((packed & 15) - 8);\n" ++
                "            const float w_hi = d * (float)((packed >> 4) - 8);\n" ++
                "#pragma unroll\n",
        );
        try appendFmt(allocator, out, "            for (int r = 0; r < {d}; ++r) acc[c][r] += w_lo * x_lo[r] + w_hi * x_hi[r];\n", .{rows});
    }
    try out.appendSlice(allocator, if (projections == 2) "            }\n        }\n    }\n" else "        }\n    }\n");

    if (rows == 1 and projections == 1) {
        try appendFmt(allocator, out, "\n    __shared__ float partial[{d}][{d}];\n", .{ cols, warps });
    } else if (rows == 1) {
        try appendFmt(allocator, out, "\n    __shared__ float partial[{d}][{d}][{d}];\n", .{ projections, cols, warps });
    } else {
        try appendFmt(allocator, out, "\n    __shared__ float partial[{d}][{d}][{d}];\n", .{ cols, rows, warps });
    }
    try out.appendSlice(
        allocator,
        "    const int lane = threadIdx.x & 31;\n" ++
            "    const int warp = threadIdx.x >> 5;\n",
    );
    if (projections == 2) {
        try appendFmt(allocator, out, "#pragma unroll\n    for (int w = 0; w < {d}; ++w) {{\n", .{projections});
    }
    const reduction_loop_indent = if (projections == 2) "        " else "    ";
    try appendFmt(allocator, out, "#pragma unroll\n{s}for (int c = 0; c < {d}; ++c) {{\n", .{ reduction_loop_indent, cols });
    if (rows > 1) {
        try appendFmt(allocator, out, "#pragma unroll\n        for (int r = 0; r < {d}; ++r) {{\n", .{rows});
    }
    const acc_expr = if (rows > 1) "acc[c][r]" else if (projections == 2) "acc[w][c]" else "acc[c]";
    const partial_expr = if (rows > 1) "partial[c][r][warp]" else if (projections == 2) "partial[w][c][warp]" else "partial[c][warp]";
    const indent = if (rows > 1) "            " else if (projections == 2) "            " else "        ";
    try appendFmt(
        allocator,
        out,
        "{s}const float total = antfly_warp_reduce_sum({s});\n" ++
            "{s}if (lane == 0) {s} = total;\n",
        .{ indent, acc_expr, indent, partial_expr },
    );
    if (rows > 1) try out.appendSlice(allocator, "        }\n");
    try out.appendSlice(allocator, if (projections == 2) "        }\n" else "    }\n");
    if (projections == 2) try out.appendSlice(allocator, "    }\n");
    try out.appendSlice(allocator, "    __syncthreads();\n");

    const final_threads: u16 = @as(u16, projections) * @as(u16, cols) * @as(u16, rows);
    try appendFmt(allocator, out, "    if (threadIdx.x < {d}) {{\n", .{final_threads});
    if (projections == 2) {
        try appendFmt(
            allocator,
            out,
            "        const int w = threadIdx.x >> 2;\n" ++
                "        const int c = threadIdx.x & 3;\n" ++
                "        float total = 0.0f;\n" ++
                "#pragma unroll\n" ++
                "        for (int i = 0; i < {d}; ++i) total += partial[w][c][i];\n" ++
                "        if (col0 + c < out_dim) {{\n" ++
                "            float *output = w == 0 ? output_a : output_b;\n" ++
                "            output[col0 + c] = total;\n" ++
                "        }}\n",
            .{warps},
        );
    } else if (rows == 1) {
        try appendFmt(
            allocator,
            out,
            "        float total = 0.0f;\n" ++
                "#pragma unroll\n" ++
                "        for (int w = 0; w < {d}; ++w) total += partial[threadIdx.x][w];\n" ++
                "        if (col0 + threadIdx.x < out_dim) output[col0 + threadIdx.x] = total;\n",
            .{warps},
        );
    } else {
        try appendFmt(
            allocator,
            out,
            "        const int c = threadIdx.x >> 3;\n" ++
                "        const int r = threadIdx.x & 7;\n" ++
                "        float total = 0.0f;\n" ++
                "#pragma unroll\n" ++
                "        for (int w = 0; w < {d}; ++w) total += partial[c][r][w];\n" ++
                "        const int col = col0 + c;\n" ++
                "        const int row = row0 + r;\n" ++
                "        if (col < out_dim && row < rows) output[(size_t)row * out_dim + col] = total;\n",
            .{warps},
        );
    }
    try out.appendSlice(allocator, "    }\n}\n");
}

fn renderPairActivationQ8Body(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    kernel_id: []const u8,
    lowering: PairActivationQ8Lowering,
) !void {
    const expected = planFor(.q4_0_pair_activation_q8_1);
    const expected_lowering = expected.lowering.q4_0_pair_activation_q8_1;
    if (!std.meta.eql(lowering, expected_lowering) or !std.mem.eql(u8, kernel_id, expected.kernel_id)) {
        return error.InvalidPairActivationQ8Lowering;
    }
    try out.appendSlice(allocator, body_q4_0_pair_activation_q8_1.source);
}

fn renderDownQ8Body(
    allocator: std.mem.Allocator,
    out: *std.ArrayListUnmanaged(u8),
    kernel_id: []const u8,
    lowering: DownQ8Lowering,
) !void {
    const expected = planFor(.q4_0_down_q8_1);
    const expected_lowering = expected.lowering.q4_0_down_q8_1;
    if (!std.meta.eql(lowering, expected_lowering) or !std.mem.eql(u8, kernel_id, expected.kernel_id)) {
        return error.InvalidDownQ8Lowering;
    }
    try out.appendSlice(allocator, body_q4_0_down_q8_1.source);
}

test "cuda renderer exposes exact launch metadata for every kernel family" {
    const expected = [_]struct {
        kind: KernelKind,
        threads: u16,
        rows: u16,
        cols: u16,
        shared: u32,
    }{
        .{ .kind = .q4_k_small_batch_bias_gelu, .threads = 128, .rows = 1, .cols = 1, .shared = 512 },
        .{ .kind = .q4_0_mmv, .threads = 256, .rows = 1, .cols = 4, .shared = 128 },
        .{ .kind = .q4_0_mm, .threads = 256, .rows = 8, .cols = 4, .shared = 1024 },
        .{ .kind = .q4_0_pair_mmv, .threads = 256, .rows = 1, .cols = 4, .shared = 256 },
        .{ .kind = .q4_0_pair_activation_q8_1, .threads = 640, .rows = 1, .cols = 32, .shared = 768 },
        .{ .kind = .q4_0_down_q8_1, .threads = 256, .rows = 1, .cols = 4, .shared = 128 },
    };
    for (expected) |item| {
        const plan = planFor(item.kind);
        try plan.validate();
        try std.testing.expectEqual(item.threads, plan.launch.threads_per_block);
        try std.testing.expectEqual(item.rows, plan.launch.output_rows_per_block);
        try std.testing.expectEqual(item.cols, plan.launch.output_cols_per_block);
        try std.testing.expectEqual(item.shared, plan.launch.static_shared_memory_bytes);
    }
}

test "cuda renderer parameterizes the common q4_0 projection lowering" {
    const allocator = std.testing.allocator;
    const mmv = try renderBodyAlloc(allocator, planFor(.q4_0_mmv));
    defer allocator.free(mmv);
    const mm = try renderBodyAlloc(allocator, planFor(.q4_0_mm));
    defer allocator.free(mm);
    const pair = try renderBodyAlloc(allocator, planFor(.q4_0_pair_mmv));
    defer allocator.free(pair);

    try std.testing.expect(std.mem.containsAtLeast(u8, mmv, 1, "float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};"));
    try std.testing.expect(std.mem.containsAtLeast(u8, mm, 1, "float acc[4][8];"));
    try std.testing.expect(std.mem.containsAtLeast(u8, mm, 1, "const int row0 = blockIdx.y << 3;"));
    try std.testing.expect(std.mem.containsAtLeast(u8, pair, 1, "float acc[2][4];"));
    try std.testing.expect(std.mem.containsAtLeast(u8, pair, 1, "const uint8_t *weight = w == 0 ? weight_a_q4_0 : weight_b_q4_0;"));
}

test "cuda renderer rejects launch and lowering drift" {
    var bad_launch = planFor(.q4_0_pair_activation_q8_1);
    bad_launch.launch.threads_per_block = 256;
    try std.testing.expectError(error.CudaLaunchDoesNotMatchKind, bad_launch.validate());

    var bad_lowering = planFor(.q4_0_mmv);
    bad_lowering.lowering.q4_0_projection.row_tile = 8;
    try std.testing.expectError(error.CudaLoweringDoesNotMatchKind, bad_lowering.validate());
}

test "cuda renderer emits each helper once" {
    const allocator = std.testing.allocator;
    const source = try renderKernel(allocator, planFor(.q4_0_pair_activation_q8_1));
    defer allocator.free(source);
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, source, "float antfly_warp_reduce_sum_f32("));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, source, "float antfly_q4_0_q8_dot16("));
}

test "cuda renderer emits typed hd256 device-scalar decode attention" {
    const plan = attentionPlanFor(.gqa_decode_scalars_hd256);
    try plan.validate();
    try std.testing.expectEqual(GridMapping.batch_query_heads, plan.launch.grid);
    try std.testing.expectEqual(@as(u16, 256), plan.launch.threads_per_block);
    try std.testing.expectEqual(@as(u32, 1032), plan.launch.static_shared_memory_bytes);
    try std.testing.expect(plan.lowering.device_decode_scalars);

    const source = try renderAttentionKernel(std.testing.allocator, plan);
    defer std.testing.allocator.free(source);
    try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "extern \"C\" __global__ void antfly_gqa_attention_decode_scalars_hd256_f32_v1("));
    try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "q_seq_len != 1u"));
    try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "head_dim != 256u"));
    try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "const unsigned int* decode_scalars"));
}

test "cuda renderer rejects attention launch drift" {
    var plan = attentionPlanFor(.gqa_decode_scalars_hd256);
    plan.launch.threads_per_block = 128;
    try std.testing.expectError(error.CudaLaunchDoesNotMatchKind, plan.validate());
}
