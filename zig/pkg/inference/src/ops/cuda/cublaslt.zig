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
const buffer_mod = @import("buffer.zig");
const context_mod = @import("context.zig");
const driver_mod = @import("driver.zig");

const Status = c_int;
const Handle = ?*anyopaque;
const MatmulDesc = ?*anyopaque;
const MatrixLayout = ?*anyopaque;
const MatmulPreference = ?*anyopaque;

const CUBLAS_STATUS_SUCCESS: Status = 0;
const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;
const CUDA_R_32F: c_int = 0;
const CUDA_R_16F: c_int = 2;
const CUDA_R_16BF: c_int = 14;
const CUBLAS_COMPUTE_32F: c_int = 68;
const CUBLASLT_MATMUL_DESC_TRANSA: c_int = 3;
const CUBLASLT_MATMUL_DESC_TRANSB: c_int = 4;
const CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES: c_int = 1;
const CUBLASLT_MATRIX_LAYOUT_ORDER: c_int = 1;
const CUBLASLT_ORDER_ROW: c_int = 1;
const CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT: c_int = 5;
const CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET: c_int = 6;

pub const Error = error{
    CublasLtUnavailable,
    CublasLtSymbolMissing,
    CublasLtError,
    CublasLtUnsupported,
};

const MatmulAlgo = extern struct {
    data: [8]u64,
};

const MatmulHeuristicResult = extern struct {
    algo: MatmulAlgo,
    workspace_size: usize,
    state: Status,
    waves_count: f32,
    reserved: [4]c_int,
};

const TensorCorePlanKey = struct {
    input_type: c_int,
    batch_count: u32,
    rows: u32,
    in_dim: u32,
    out_dim: u32,
    workspace_bytes: usize,
};

const TensorCorePlan = struct {
    algo: MatmulAlgo,
    workspace_size: usize,
};

pub const CublasLt = struct {
    allocator: ?std.mem.Allocator = null,
    lib: std.DynLib,
    handle: Handle,
    fns: Table,
    tensor_core_plans: std.AutoHashMapUnmanaged(TensorCorePlanKey, TensorCorePlan) = .empty,

    const Table = struct {
        cublasLtCreate: *const fn (*Handle) callconv(.c) Status,
        cublasLtDestroy: *const fn (Handle) callconv(.c) Status,
        cublasLtMatmul: *const fn (
            Handle,
            MatmulDesc,
            ?*const anyopaque,
            ?*const anyopaque,
            MatrixLayout,
            ?*const anyopaque,
            MatrixLayout,
            ?*const anyopaque,
            ?*const anyopaque,
            MatrixLayout,
            ?*anyopaque,
            MatrixLayout,
            ?*const MatmulAlgo,
            ?*anyopaque,
            usize,
            driver_mod.CUstream,
        ) callconv(.c) Status,
        cublasLtMatmulDescCreate: *const fn (*MatmulDesc, c_int, c_int) callconv(.c) Status,
        cublasLtMatmulDescDestroy: *const fn (MatmulDesc) callconv(.c) Status,
        cublasLtMatmulDescSetAttribute: *const fn (MatmulDesc, c_int, ?*const anyopaque, usize) callconv(.c) Status,
        cublasLtMatrixLayoutCreate: *const fn (*MatrixLayout, c_int, u64, u64, i64) callconv(.c) Status,
        cublasLtMatrixLayoutDestroy: *const fn (MatrixLayout) callconv(.c) Status,
        cublasLtMatrixLayoutSetAttribute: *const fn (MatrixLayout, c_int, ?*const anyopaque, usize) callconv(.c) Status,
        cublasLtMatmulPreferenceCreate: *const fn (*MatmulPreference) callconv(.c) Status,
        cublasLtMatmulPreferenceDestroy: *const fn (MatmulPreference) callconv(.c) Status,
        cublasLtMatmulPreferenceSetAttribute: *const fn (MatmulPreference, c_int, ?*const anyopaque, usize) callconv(.c) Status,
        cublasLtMatmulAlgoGetHeuristic: *const fn (
            Handle,
            MatmulDesc,
            MatrixLayout,
            MatrixLayout,
            MatrixLayout,
            MatrixLayout,
            MatmulPreference,
            c_int,
            [*]MatmulHeuristicResult,
            *c_int,
        ) callconv(.c) Status,
    };

    pub fn open() Error!CublasLt {
        return openWithAllocator(null);
    }

    pub fn openWithAllocator(allocator: ?std.mem.Allocator) Error!CublasLt {
        var lib = openLibrary() catch return error.CublasLtUnavailable;
        errdefer lib.close();
        const fns = Table{
            .cublasLtCreate = lookup(&lib, @TypeOf(@as(Table, undefined).cublasLtCreate), "cublasLtCreate") catch return error.CublasLtSymbolMissing,
            .cublasLtDestroy = lookup(&lib, @TypeOf(@as(Table, undefined).cublasLtDestroy), "cublasLtDestroy") catch return error.CublasLtSymbolMissing,
            .cublasLtMatmul = lookup(&lib, @TypeOf(@as(Table, undefined).cublasLtMatmul), "cublasLtMatmul") catch return error.CublasLtSymbolMissing,
            .cublasLtMatmulDescCreate = lookup(&lib, @TypeOf(@as(Table, undefined).cublasLtMatmulDescCreate), "cublasLtMatmulDescCreate") catch return error.CublasLtSymbolMissing,
            .cublasLtMatmulDescDestroy = lookup(&lib, @TypeOf(@as(Table, undefined).cublasLtMatmulDescDestroy), "cublasLtMatmulDescDestroy") catch return error.CublasLtSymbolMissing,
            .cublasLtMatmulDescSetAttribute = lookup(&lib, @TypeOf(@as(Table, undefined).cublasLtMatmulDescSetAttribute), "cublasLtMatmulDescSetAttribute") catch return error.CublasLtSymbolMissing,
            .cublasLtMatrixLayoutCreate = lookup(&lib, @TypeOf(@as(Table, undefined).cublasLtMatrixLayoutCreate), "cublasLtMatrixLayoutCreate") catch return error.CublasLtSymbolMissing,
            .cublasLtMatrixLayoutDestroy = lookup(&lib, @TypeOf(@as(Table, undefined).cublasLtMatrixLayoutDestroy), "cublasLtMatrixLayoutDestroy") catch return error.CublasLtSymbolMissing,
            .cublasLtMatrixLayoutSetAttribute = lookup(&lib, @TypeOf(@as(Table, undefined).cublasLtMatrixLayoutSetAttribute), "cublasLtMatrixLayoutSetAttribute") catch return error.CublasLtSymbolMissing,
            .cublasLtMatmulPreferenceCreate = lookup(&lib, @TypeOf(@as(Table, undefined).cublasLtMatmulPreferenceCreate), "cublasLtMatmulPreferenceCreate") catch return error.CublasLtSymbolMissing,
            .cublasLtMatmulPreferenceDestroy = lookup(&lib, @TypeOf(@as(Table, undefined).cublasLtMatmulPreferenceDestroy), "cublasLtMatmulPreferenceDestroy") catch return error.CublasLtSymbolMissing,
            .cublasLtMatmulPreferenceSetAttribute = lookup(&lib, @TypeOf(@as(Table, undefined).cublasLtMatmulPreferenceSetAttribute), "cublasLtMatmulPreferenceSetAttribute") catch return error.CublasLtSymbolMissing,
            .cublasLtMatmulAlgoGetHeuristic = lookup(&lib, @TypeOf(@as(Table, undefined).cublasLtMatmulAlgoGetHeuristic), "cublasLtMatmulAlgoGetHeuristic") catch return error.CublasLtSymbolMissing,
        };
        var handle: Handle = null;
        if (fns.cublasLtCreate(&handle) != CUBLAS_STATUS_SUCCESS) return error.CublasLtUnavailable;
        return .{
            .allocator = allocator,
            .lib = lib,
            .handle = handle,
            .fns = fns,
        };
    }

    pub fn deinit(self: *CublasLt) void {
        if (self.allocator) |allocator| {
            self.tensor_core_plans.deinit(allocator);
        }
        if (self.handle != null) {
            _ = self.fns.cublasLtDestroy(self.handle);
            self.handle = null;
        }
        self.lib.close();
    }

    pub fn matmulBf16WeightF32Out(
        self: *CublasLt,
        ctx: *context_mod.CudaContext,
        dst: buffer_mod.DeviceBuffer,
        input_bf16: buffer_mod.DeviceBuffer,
        weight_bf16: buffer_mod.DeviceBuffer,
        workspace: buffer_mod.DeviceBuffer,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) Error!void {
        return self.matmul16WeightF32Out(ctx, dst, input_bf16, weight_bf16, workspace, rows, in_dim, out_dim, CUDA_R_16BF);
    }

    /// Tensor-core matmul for FP16 GGUF weights. Activations are staged from
    /// the F32 graph representation, while accumulation and the output stay
    /// in F32 so this is numerically compatible with the existing encoder
    /// execution contract.
    pub fn matmulF16WeightF32Out(
        self: *CublasLt,
        ctx: *context_mod.CudaContext,
        dst: buffer_mod.DeviceBuffer,
        input_f16: buffer_mod.DeviceBuffer,
        weight_f16: buffer_mod.DeviceBuffer,
        workspace: buffer_mod.DeviceBuffer,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) Error!void {
        return self.matmul16WeightF32Out(ctx, dst, input_f16, weight_f16, workspace, rows, in_dim, out_dim, CUDA_R_16F);
    }

    fn matmul16WeightF32Out(
        self: *CublasLt,
        ctx: *context_mod.CudaContext,
        dst: buffer_mod.DeviceBuffer,
        input: buffer_mod.DeviceBuffer,
        weight: buffer_mod.DeviceBuffer,
        workspace: buffer_mod.DeviceBuffer,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
        input_type: c_int,
    ) Error!void {
        if (rows == 0 or in_dim == 0 or out_dim == 0) return;
        if (rows > std.math.maxInt(u32) or in_dim > std.math.maxInt(u32) or out_dim > std.math.maxInt(u32)) return error.CublasLtUnsupported;
        try checkRawBytes(input, rows * in_dim * @sizeOf(u16));
        try checkRawBytes(weight, out_dim * in_dim * @sizeOf(u16));
        try checkRawBytes(dst, rows * out_dim * @sizeOf(f32));

        var op_desc: MatmulDesc = null;
        try self.check(self.fns.cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        defer _ = self.fns.cublasLtMatmulDescDestroy(op_desc);

        var transa = CUBLAS_OP_T;
        var transb = CUBLAS_OP_N;
        try self.check(self.fns.cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, @sizeOf(c_int)));
        try self.check(self.fns.cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, @sizeOf(c_int)));

        var a_desc: MatrixLayout = null;
        var b_desc: MatrixLayout = null;
        var c_desc: MatrixLayout = null;
        var d_desc: MatrixLayout = null;
        try self.check(self.fns.cublasLtMatrixLayoutCreate(&a_desc, input_type, in_dim, out_dim, @intCast(in_dim)));
        defer _ = self.fns.cublasLtMatrixLayoutDestroy(a_desc);
        try self.check(self.fns.cublasLtMatrixLayoutCreate(&b_desc, input_type, in_dim, rows, @intCast(in_dim)));
        defer _ = self.fns.cublasLtMatrixLayoutDestroy(b_desc);
        try self.check(self.fns.cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, out_dim, rows, @intCast(out_dim)));
        defer _ = self.fns.cublasLtMatrixLayoutDestroy(c_desc);
        try self.check(self.fns.cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_32F, out_dim, rows, @intCast(out_dim)));
        defer _ = self.fns.cublasLtMatrixLayoutDestroy(d_desc);

        const plan = try self.tensorCorePlanFor(input_type, 1, rows, in_dim, out_dim, workspace.len, op_desc, a_desc, b_desc, c_desc, d_desc);
        try checkRawBytes(workspace, plan.workspace_size);

        var alpha: f32 = 1.0;
        var beta: f32 = 0.0;
        const workspace_ptr: ?*anyopaque = if (workspace.ptr != 0 and plan.workspace_size > 0)
            @ptrFromInt(workspace.ptr)
        else
            null;
        ctx.makeCurrent() catch return error.CublasLtError;
        try self.check(self.fns.cublasLtMatmul(
            self.handle,
            op_desc,
            &alpha,
            @ptrFromInt(weight.ptr),
            a_desc,
            @ptrFromInt(input.ptr),
            b_desc,
            &beta,
            @ptrFromInt(dst.ptr),
            c_desc,
            @ptrFromInt(dst.ptr),
            d_desc,
            &plan.algo,
            workspace_ptr,
            plan.workspace_size,
            ctx.stream,
        ));
    }

    fn tensorCorePlanFor(
        self: *CublasLt,
        input_type: c_int,
        batch_count: usize,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
        workspace_bytes: usize,
        op_desc: MatmulDesc,
        a_desc: MatrixLayout,
        b_desc: MatrixLayout,
        c_desc: MatrixLayout,
        d_desc: MatrixLayout,
    ) Error!TensorCorePlan {
        if (batch_count == 0 or batch_count > std.math.maxInt(u32)) return error.CublasLtUnsupported;
        const key = TensorCorePlanKey{
            .input_type = input_type,
            .batch_count = @intCast(batch_count),
            .rows = @intCast(rows),
            .in_dim = @intCast(in_dim),
            .out_dim = @intCast(out_dim),
            .workspace_bytes = workspace_bytes,
        };
        if (self.tensor_core_plans.get(key)) |plan| return plan;

        var pref: MatmulPreference = null;
        try self.check(self.fns.cublasLtMatmulPreferenceCreate(&pref));
        defer _ = self.fns.cublasLtMatmulPreferenceDestroy(pref);
        var max_workspace = workspace_bytes;
        try self.check(self.fns.cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, @sizeOf(usize)));

        var heuristic: [1]MatmulHeuristicResult = undefined;
        var returned: c_int = 0;
        try self.check(self.fns.cublasLtMatmulAlgoGetHeuristic(self.handle, op_desc, a_desc, b_desc, c_desc, d_desc, pref, 1, &heuristic, &returned));
        if (returned <= 0 or heuristic[0].state != CUBLAS_STATUS_SUCCESS) return error.CublasLtUnsupported;

        const plan = TensorCorePlan{
            .algo = heuristic[0].algo,
            .workspace_size = heuristic[0].workspace_size,
        };
        if (self.allocator) |allocator| {
            self.tensor_core_plans.put(allocator, key, plan) catch {};
        }
        return plan;
    }

    /// Tensor-core strided-batched row-major GEMM. The logical operation is
    /// `dst[batch, rows, out_dim] = input[batch, rows, in_dim] *
    /// weight[batch, out_dim, in_dim]^T`; F16 inputs accumulate and store F32.
    pub fn matmulF16StridedBatchedF32Out(
        self: *CublasLt,
        ctx: *context_mod.CudaContext,
        dst: buffer_mod.DeviceBuffer,
        input_f16: buffer_mod.DeviceBuffer,
        weight_f16: buffer_mod.DeviceBuffer,
        workspace: buffer_mod.DeviceBuffer,
        batch_count: usize,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) Error!void {
        if (batch_count == 0 or batch_count > std.math.maxInt(i32) or rows == 0 or in_dim == 0 or out_dim == 0) return error.CublasLtUnsupported;
        if (rows > std.math.maxInt(u32) or in_dim > std.math.maxInt(u32) or out_dim > std.math.maxInt(u32)) return error.CublasLtUnsupported;
        const input_stride = std.math.mul(usize, rows, in_dim) catch return error.CublasLtUnsupported;
        const weight_stride = std.math.mul(usize, out_dim, in_dim) catch return error.CublasLtUnsupported;
        const output_stride = std.math.mul(usize, rows, out_dim) catch return error.CublasLtUnsupported;
        const input_count = std.math.mul(usize, batch_count, input_stride) catch return error.CublasLtUnsupported;
        const weight_count = std.math.mul(usize, batch_count, weight_stride) catch return error.CublasLtUnsupported;
        const output_count = std.math.mul(usize, batch_count, output_stride) catch return error.CublasLtUnsupported;
        try checkRawBytes(input_f16, std.math.mul(usize, input_count, @sizeOf(u16)) catch return error.CublasLtUnsupported);
        try checkRawBytes(weight_f16, std.math.mul(usize, weight_count, @sizeOf(u16)) catch return error.CublasLtUnsupported);
        try checkRawBytes(dst, std.math.mul(usize, output_count, @sizeOf(f32)) catch return error.CublasLtUnsupported);

        var op_desc: MatmulDesc = null;
        try self.check(self.fns.cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        defer _ = self.fns.cublasLtMatmulDescDestroy(op_desc);
        var transa = CUBLAS_OP_N;
        var transb = CUBLAS_OP_T;
        try self.check(self.fns.cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, @sizeOf(c_int)));
        try self.check(self.fns.cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, @sizeOf(c_int)));

        var a_desc: MatrixLayout = null;
        var b_desc: MatrixLayout = null;
        var c_desc: MatrixLayout = null;
        var d_desc: MatrixLayout = null;
        try self.check(self.fns.cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, out_dim, in_dim, @intCast(in_dim)));
        defer _ = self.fns.cublasLtMatrixLayoutDestroy(a_desc);
        try self.check(self.fns.cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, rows, in_dim, @intCast(in_dim)));
        defer _ = self.fns.cublasLtMatrixLayoutDestroy(b_desc);
        try self.check(self.fns.cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, out_dim, rows, @intCast(out_dim)));
        defer _ = self.fns.cublasLtMatrixLayoutDestroy(c_desc);
        try self.check(self.fns.cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_32F, out_dim, rows, @intCast(out_dim)));
        defer _ = self.fns.cublasLtMatrixLayoutDestroy(d_desc);
        var row_order = CUBLASLT_ORDER_ROW;
        try self.check(self.fns.cublasLtMatrixLayoutSetAttribute(a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, @sizeOf(c_int)));
        try self.check(self.fns.cublasLtMatrixLayoutSetAttribute(b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, @sizeOf(c_int)));

        const batch_i32: i32 = @intCast(batch_count);
        try self.configureStridedBatch(a_desc, batch_i32, @intCast(weight_stride));
        try self.configureStridedBatch(b_desc, batch_i32, @intCast(input_stride));
        try self.configureStridedBatch(c_desc, batch_i32, @intCast(output_stride));
        try self.configureStridedBatch(d_desc, batch_i32, @intCast(output_stride));

        const plan = try self.tensorCorePlanFor(CUDA_R_16F, batch_count, rows, in_dim, out_dim, workspace.len, op_desc, a_desc, b_desc, c_desc, d_desc);
        try checkRawBytes(workspace, plan.workspace_size);
        var alpha: f32 = 1.0;
        var beta: f32 = 0.0;
        const workspace_ptr: ?*anyopaque = if (workspace.ptr != 0 and plan.workspace_size > 0) @ptrFromInt(workspace.ptr) else null;
        ctx.makeCurrent() catch return error.CublasLtError;
        try self.check(self.fns.cublasLtMatmul(
            self.handle,
            op_desc,
            &alpha,
            @ptrFromInt(weight_f16.ptr),
            a_desc,
            @ptrFromInt(input_f16.ptr),
            b_desc,
            &beta,
            @ptrFromInt(dst.ptr),
            c_desc,
            @ptrFromInt(dst.ptr),
            d_desc,
            &plan.algo,
            workspace_ptr,
            plan.workspace_size,
            ctx.stream,
        ));
    }

    fn configureStridedBatch(self: *CublasLt, layout: MatrixLayout, batch_count: i32, stride: i64) Error!void {
        try self.check(self.fns.cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch_count, @sizeOf(i32)));
        try self.check(self.fns.cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &stride, @sizeOf(i64)));
    }

    pub fn matmulF32WeightF32Out(
        self: *CublasLt,
        ctx: *context_mod.CudaContext,
        dst: buffer_mod.DeviceBuffer,
        input_f32: buffer_mod.DeviceBuffer,
        weight_f32: buffer_mod.DeviceBuffer,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
    ) Error!void {
        if (rows == 0 or in_dim == 0 or out_dim == 0) return;
        if (rows > std.math.maxInt(u32) or in_dim > std.math.maxInt(u32) or out_dim > std.math.maxInt(u32)) return error.CublasLtUnsupported;
        try checkRawBytes(input_f32, rows * in_dim * @sizeOf(f32));
        try checkRawBytes(weight_f32, out_dim * in_dim * @sizeOf(f32));
        try checkRawBytes(dst, rows * out_dim * @sizeOf(f32));

        var op_desc: MatmulDesc = null;
        try self.check(self.fns.cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        defer _ = self.fns.cublasLtMatmulDescDestroy(op_desc);

        var transa = CUBLAS_OP_T;
        var transb = CUBLAS_OP_N;
        try self.check(self.fns.cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, @sizeOf(c_int)));
        try self.check(self.fns.cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, @sizeOf(c_int)));

        var a_desc: MatrixLayout = null;
        var b_desc: MatrixLayout = null;
        var c_desc: MatrixLayout = null;
        var d_desc: MatrixLayout = null;
        try self.check(self.fns.cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_32F, in_dim, out_dim, @intCast(in_dim)));
        defer _ = self.fns.cublasLtMatrixLayoutDestroy(a_desc);
        try self.check(self.fns.cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_32F, in_dim, rows, @intCast(in_dim)));
        defer _ = self.fns.cublasLtMatrixLayoutDestroy(b_desc);
        try self.check(self.fns.cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, out_dim, rows, @intCast(out_dim)));
        defer _ = self.fns.cublasLtMatrixLayoutDestroy(c_desc);
        try self.check(self.fns.cublasLtMatrixLayoutCreate(&d_desc, CUDA_R_32F, out_dim, rows, @intCast(out_dim)));
        defer _ = self.fns.cublasLtMatrixLayoutDestroy(d_desc);

        var pref: MatmulPreference = null;
        try self.check(self.fns.cublasLtMatmulPreferenceCreate(&pref));
        defer _ = self.fns.cublasLtMatmulPreferenceDestroy(pref);
        var max_workspace: usize = 0;
        try self.check(self.fns.cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, @sizeOf(usize)));

        var heuristic: [1]MatmulHeuristicResult = undefined;
        var returned: c_int = 0;
        try self.check(self.fns.cublasLtMatmulAlgoGetHeuristic(self.handle, op_desc, a_desc, b_desc, c_desc, d_desc, pref, 1, &heuristic, &returned));
        if (returned <= 0 or heuristic[0].state != CUBLAS_STATUS_SUCCESS) return error.CublasLtUnsupported;

        var alpha: f32 = 1.0;
        var beta: f32 = 0.0;
        ctx.makeCurrent() catch return error.CublasLtError;
        try self.check(self.fns.cublasLtMatmul(
            self.handle,
            op_desc,
            &alpha,
            @ptrFromInt(weight_f32.ptr),
            a_desc,
            @ptrFromInt(input_f32.ptr),
            b_desc,
            &beta,
            @ptrFromInt(dst.ptr),
            c_desc,
            @ptrFromInt(dst.ptr),
            d_desc,
            &heuristic[0].algo,
            null,
            0,
            ctx.stream,
        ));
    }

    fn check(self: *const CublasLt, status: Status) Error!void {
        _ = self;
        if (status != CUBLAS_STATUS_SUCCESS) return error.CublasLtError;
    }
};

fn openLibrary() !std.DynLib {
    return std.DynLib.open("libcublasLt.so") catch
        std.DynLib.open("libcublasLt.so.11") catch
        std.DynLib.open("libcublasLt.so.12");
}

fn lookup(lib: *std.DynLib, comptime T: type, name: [:0]const u8) Error!T {
    return lib.lookup(T, name) orelse error.CublasLtSymbolMissing;
}

fn checkRawBytes(buffer: buffer_mod.DeviceBuffer, bytes: usize) Error!void {
    if (bytes > buffer.len) return error.CublasLtUnsupported;
}
