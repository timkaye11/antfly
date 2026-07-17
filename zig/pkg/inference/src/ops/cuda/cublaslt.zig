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

const TensorCorePlanKind = enum(u8) {
    dense,
    strided_batched,
};

const TensorCorePlanKey = struct {
    kind: TensorCorePlanKind,
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
    op_desc: MatmulDesc,
    a_desc: MatrixLayout,
    b_desc: MatrixLayout,
    c_desc: MatrixLayout,
    d_desc: MatrixLayout,
};

const TensorCorePlanLease = struct {
    plan: TensorCorePlan,
    owns_descriptors: bool,

    fn deinit(self: *TensorCorePlanLease, cublas: *CublasLt) void {
        if (self.owns_descriptors) cublas.destroyTensorCorePlan(self.plan);
        self.* = undefined;
    }
};

// Shape diversity is small for a loaded model, but untrusted request shapes
// must not be able to grow this process-lifetime cache without bound.
const maximum_tensor_core_plans = 256;

pub const CublasLt = struct {
    allocator: ?std.mem.Allocator = null,
    lib: std.DynLib,
    handle: Handle,
    fns: Table,
    tensor_core_plans: std.AutoHashMapUnmanaged(TensorCorePlanKey, TensorCorePlan) = .empty,
    tensor_core_plans_mutex: std.atomic.Mutex = .unlocked,

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
            var plans = self.tensor_core_plans.valueIterator();
            while (plans.next()) |plan| self.destroyTensorCorePlan(plan.*);
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
        try checkRawBytes(input, try checkedMatrixBytes(rows, in_dim, @sizeOf(u16)));
        try checkRawBytes(weight, try checkedMatrixBytes(out_dim, in_dim, @sizeOf(u16)));
        try checkRawBytes(dst, try checkedMatrixBytes(rows, out_dim, @sizeOf(f32)));

        var plan_lease = try self.tensorCorePlanFor(.dense, input_type, 1, rows, in_dim, out_dim, workspace.len);
        defer plan_lease.deinit(self);
        const plan = plan_lease.plan;
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
            plan.op_desc,
            &alpha,
            @ptrFromInt(weight.ptr),
            plan.a_desc,
            @ptrFromInt(input.ptr),
            plan.b_desc,
            &beta,
            @ptrFromInt(dst.ptr),
            plan.c_desc,
            @ptrFromInt(dst.ptr),
            plan.d_desc,
            &plan.algo,
            workspace_ptr,
            plan.workspace_size,
            ctx.stream,
        ));
    }

    fn tensorCorePlanFor(
        self: *CublasLt,
        kind: TensorCorePlanKind,
        input_type: c_int,
        batch_count: usize,
        rows: usize,
        in_dim: usize,
        out_dim: usize,
        workspace_bytes: usize,
    ) Error!TensorCorePlanLease {
        if (batch_count == 0 or batch_count > std.math.maxInt(u32)) return error.CublasLtUnsupported;
        if (rows == 0 or rows > std.math.maxInt(u32) or in_dim == 0 or in_dim > std.math.maxInt(u32) or out_dim == 0 or out_dim > std.math.maxInt(u32)) return error.CublasLtUnsupported;
        const key = TensorCorePlanKey{
            .kind = kind,
            .input_type = input_type,
            .batch_count = @intCast(batch_count),
            .rows = @intCast(rows),
            .in_dim = @intCast(in_dim),
            .out_dim = @intCast(out_dim),
            .workspace_bytes = workspace_bytes,
        };
        spinLock(&self.tensor_core_plans_mutex);
        const cached_plan = self.tensor_core_plans.get(key);
        self.tensor_core_plans_mutex.unlock();
        if (cached_plan) |plan| return .{ .plan = plan, .owns_descriptors = false };

        const candidate = try self.createTensorCorePlan(key);
        if (self.allocator) |allocator| {
            spinLock(&self.tensor_core_plans_mutex);
            // Another request may have populated the same plan while this
            // request was running the heuristic. Prefer that canonical entry.
            if (self.tensor_core_plans.get(key)) |existing| {
                self.tensor_core_plans_mutex.unlock();
                self.destroyTensorCorePlan(candidate);
                return .{ .plan = existing, .owns_descriptors = false };
            }
            if (self.tensor_core_plans.count() < maximum_tensor_core_plans) {
                self.tensor_core_plans.put(allocator, key, candidate) catch {
                    self.tensor_core_plans_mutex.unlock();
                    return .{ .plan = candidate, .owns_descriptors = true };
                };
                self.tensor_core_plans_mutex.unlock();
                return .{ .plan = candidate, .owns_descriptors = false };
            }
            self.tensor_core_plans_mutex.unlock();
        }
        return .{ .plan = candidate, .owns_descriptors = true };
    }

    fn createTensorCorePlan(self: *CublasLt, key: TensorCorePlanKey) Error!TensorCorePlan {
        var plan = TensorCorePlan{
            .algo = undefined,
            .workspace_size = 0,
            .op_desc = null,
            .a_desc = null,
            .b_desc = null,
            .c_desc = null,
            .d_desc = null,
        };
        errdefer self.destroyTensorCorePlan(plan);

        try self.check(self.fns.cublasLtMatmulDescCreate(&plan.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        var transa: c_int = undefined;
        var transb: c_int = undefined;
        switch (key.kind) {
            .dense => {
                transa = CUBLAS_OP_T;
                transb = CUBLAS_OP_N;
                try self.check(self.fns.cublasLtMatrixLayoutCreate(&plan.a_desc, key.input_type, key.in_dim, key.out_dim, key.in_dim));
                try self.check(self.fns.cublasLtMatrixLayoutCreate(&plan.b_desc, key.input_type, key.in_dim, key.rows, key.in_dim));
            },
            .strided_batched => {
                if (key.batch_count > std.math.maxInt(i32)) return error.CublasLtUnsupported;
                transa = CUBLAS_OP_N;
                transb = CUBLAS_OP_T;
                try self.check(self.fns.cublasLtMatrixLayoutCreate(&plan.a_desc, key.input_type, key.out_dim, key.in_dim, key.in_dim));
                try self.check(self.fns.cublasLtMatrixLayoutCreate(&plan.b_desc, key.input_type, key.rows, key.in_dim, key.in_dim));
            },
        }
        try self.check(self.fns.cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, @sizeOf(c_int)));
        try self.check(self.fns.cublasLtMatmulDescSetAttribute(plan.op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, @sizeOf(c_int)));
        try self.check(self.fns.cublasLtMatrixLayoutCreate(&plan.c_desc, CUDA_R_32F, key.out_dim, key.rows, key.out_dim));
        try self.check(self.fns.cublasLtMatrixLayoutCreate(&plan.d_desc, CUDA_R_32F, key.out_dim, key.rows, key.out_dim));

        if (key.kind == .strided_batched) {
            var row_order = CUBLASLT_ORDER_ROW;
            try self.check(self.fns.cublasLtMatrixLayoutSetAttribute(plan.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, @sizeOf(c_int)));
            try self.check(self.fns.cublasLtMatrixLayoutSetAttribute(plan.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_order, @sizeOf(c_int)));
            const batch_i32: i32 = @intCast(key.batch_count);
            const weight_stride = try checkedI64Product(key.out_dim, key.in_dim);
            const input_stride = try checkedI64Product(key.rows, key.in_dim);
            const output_stride = try checkedI64Product(key.rows, key.out_dim);
            try self.configureStridedBatch(plan.a_desc, batch_i32, weight_stride);
            try self.configureStridedBatch(plan.b_desc, batch_i32, input_stride);
            try self.configureStridedBatch(plan.c_desc, batch_i32, output_stride);
            try self.configureStridedBatch(plan.d_desc, batch_i32, output_stride);
        }

        var pref: MatmulPreference = null;
        try self.check(self.fns.cublasLtMatmulPreferenceCreate(&pref));
        defer _ = self.fns.cublasLtMatmulPreferenceDestroy(pref);
        var max_workspace = key.workspace_bytes;
        try self.check(self.fns.cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace, @sizeOf(usize)));

        var heuristic: [1]MatmulHeuristicResult = undefined;
        var returned: c_int = 0;
        try self.check(self.fns.cublasLtMatmulAlgoGetHeuristic(self.handle, plan.op_desc, plan.a_desc, plan.b_desc, plan.c_desc, plan.d_desc, pref, 1, &heuristic, &returned));
        if (returned <= 0 or heuristic[0].state != CUBLAS_STATUS_SUCCESS) return error.CublasLtUnsupported;

        plan.algo = heuristic[0].algo;
        plan.workspace_size = heuristic[0].workspace_size;
        return plan;
    }

    fn destroyTensorCorePlan(self: *CublasLt, plan: TensorCorePlan) void {
        if (plan.a_desc != null) _ = self.fns.cublasLtMatrixLayoutDestroy(plan.a_desc);
        if (plan.b_desc != null) _ = self.fns.cublasLtMatrixLayoutDestroy(plan.b_desc);
        if (plan.c_desc != null) _ = self.fns.cublasLtMatrixLayoutDestroy(plan.c_desc);
        if (plan.d_desc != null) _ = self.fns.cublasLtMatrixLayoutDestroy(plan.d_desc);
        if (plan.op_desc != null) _ = self.fns.cublasLtMatmulDescDestroy(plan.op_desc);
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

        var plan_lease = try self.tensorCorePlanFor(.strided_batched, CUDA_R_16F, batch_count, rows, in_dim, out_dim, workspace.len);
        defer plan_lease.deinit(self);
        const plan = plan_lease.plan;
        try checkRawBytes(workspace, plan.workspace_size);
        var alpha: f32 = 1.0;
        var beta: f32 = 0.0;
        const workspace_ptr: ?*anyopaque = if (workspace.ptr != 0 and plan.workspace_size > 0) @ptrFromInt(workspace.ptr) else null;
        ctx.makeCurrent() catch return error.CublasLtError;
        try self.check(self.fns.cublasLtMatmul(
            self.handle,
            plan.op_desc,
            &alpha,
            @ptrFromInt(weight_f16.ptr),
            plan.a_desc,
            @ptrFromInt(input_f16.ptr),
            plan.b_desc,
            &beta,
            @ptrFromInt(dst.ptr),
            plan.c_desc,
            @ptrFromInt(dst.ptr),
            plan.d_desc,
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
        try checkRawBytes(input_f32, try checkedMatrixBytes(rows, in_dim, @sizeOf(f32)));
        try checkRawBytes(weight_f32, try checkedMatrixBytes(out_dim, in_dim, @sizeOf(f32)));
        try checkRawBytes(dst, try checkedMatrixBytes(rows, out_dim, @sizeOf(f32)));

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

fn checkedMatrixBytes(rows: usize, cols: usize, element_bytes: usize) Error!usize {
    const elements = std.math.mul(usize, rows, cols) catch return error.CublasLtUnsupported;
    return std.math.mul(usize, elements, element_bytes) catch return error.CublasLtUnsupported;
}

fn checkedI64Product(a: u32, b: u32) Error!i64 {
    const product = std.math.mul(u64, a, b) catch return error.CublasLtUnsupported;
    if (product > std.math.maxInt(i64)) return error.CublasLtUnsupported;
    return @intCast(product);
}

fn spinLock(mutex: *std.atomic.Mutex) void {
    while (!mutex.tryLock()) std.atomic.spinLoopHint();
}

test "tensor core plan keys distinguish incompatible layouts" {
    const common = TensorCorePlanKey{
        .kind = .dense,
        .input_type = CUDA_R_16F,
        .batch_count = 1,
        .rows = 256,
        .in_dim = 768,
        .out_dim = 768,
        .workspace_bytes = 4 * 1024 * 1024,
    };
    var batched = common;
    batched.kind = .strided_batched;
    try std.testing.expect(!std.meta.eql(common, batched));
}

test "matrix byte checks reject overflow" {
    try std.testing.expectError(error.CublasLtUnsupported, checkedMatrixBytes(std.math.maxInt(usize), 2, 2));
}
