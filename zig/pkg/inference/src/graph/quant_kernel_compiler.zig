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

//! Domain-specific quant matmul compiler scaffolding.
//!
//! This is intentionally tiny: packed GGUF block decode, matmul schedules, and
//! common epilogues only. Production dispatch still uses existing handwritten
//! CUDA/Metal paths until a generated kernel is promoted into checked-in
//! backend artifacts.

const std = @import("std");
const backend_contracts = @import("backend_contracts.zig");
const quant_matmul = @import("quant_matmul.zig");
const quant_codec = @import("../gguf/quant_codec.zig");
const tensor_types = @import("../gguf/tensor_types.zig");

pub const metal_promotion_min_speedup: f64 = 1.10;
pub const metal_promotion_speedup_tolerance: f64 = 0.001;
pub const metal_promotion_repeat_runs: usize = 5;
pub const metal_promotion_warmup_repeat_runs: u32 = 2;
pub const metal_promotion_measure_iters: u32 = 500;
const metal_promotion_repeat_runs_text = "5";
const metal_promotion_measure_iters_text = "500";
const metal_promotion_args = " --repeat-runs " ++ metal_promotion_repeat_runs_text ++ " --measure-iters " ++ metal_promotion_measure_iters_text ++ " --promotion-ready-kernel ";
pub const metal_blocker_none = "none";
pub const metal_blocker_speedup_gate_missing = "speedup_gate_missing";
pub const metal_blocker_unstable_benchmark_timing = "unstable_benchmark_timing";
pub const metal_blocker_runtime_route_only = "runtime_route_only";
pub const metal_blocker_missing_generated_route = "missing_metal_generated_route_evidence";
pub const metal_blocker_missing_provider_route = "missing_metal_provider_route_evidence";
pub const metal_blocker_unsupported_handwritten = "unsupported_handwritten_baseline";
pub const metal_blocker_dev_only_candidate = "dev_only_candidate";

pub const Backend = enum(u8) {
    cuda,
    metal,
};

pub const Epilogue = enum(u8) {
    none,
    bias,
    bias_gelu,
    pair,
    triple,
    relu,
    gelu,
    add,
    argmax,
};

pub const DType = enum(u8) {
    f32,
    f16,
    bf16,
};

pub const DecodeOpKind = enum(u8) {
    load_packed_bytes,
    extract_scale,
    extract_min,
    extract_quant_lane,
    dequant_lane,
};

pub const DecodeOp = struct {
    kind: DecodeOpKind,
    expression: []const u8,
};

pub const BlockField = struct {
    name: []const u8,
    offset: usize,
    bytes: usize,
};

pub const QuantKernelSpec = struct {
    format: quant_matmul.Format,
    block_values: usize,
    block_bytes: usize,
    block_fields: []const BlockField,
    decode_ops: []const DecodeOp,
    supported_schedules: []const quant_matmul.DispatchKind,
    supported_epilogues: []const Epilogue,
    accumulator_dtype: DType,
    output_dtype: DType,
    supported_backends: []const Backend,

    pub fn supportsSchedule(self: QuantKernelSpec, schedule: quant_matmul.DispatchKind) bool {
        return contains(quant_matmul.DispatchKind, self.supported_schedules, schedule);
    }

    pub fn supportsEpilogue(self: QuantKernelSpec, epilogue: Epilogue) bool {
        return contains(Epilogue, self.supported_epilogues, epilogue);
    }

    pub fn supportsBackend(self: QuantKernelSpec, backend: Backend) bool {
        return contains(Backend, self.supported_backends, backend);
    }
};

pub const IROp = enum(u8) {
    load_input_row,
    load_quant_block,
    decode_quant_lane,
    multiply_accumulate,
    reduce_accumulator,
    apply_bias,
    apply_gelu,
    write_output,
};

pub const QuantKernelIR = struct {
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    dispatch: quant_matmul.DispatchKind,
    epilogue: Epilogue,
    ops: []const IROp,
};

pub const QuantKernelSchedule = struct {
    dispatch: quant_matmul.DispatchKind,
    row_bucket: quant_matmul.RowBucket,
    tile_rows: usize,
    tile_cols: usize,
    vector_width: usize,
    threads_per_block: usize,
    shared_memory_bytes: usize,
    register_pressure_hint: u8,
    tensor_core_eligible: bool,
};

pub const QuantKernelPlanId = struct {
    backend: Backend,
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
    dispatch: quant_matmul.DispatchKind,
};

pub const LoweringRoute = enum(u8) {
    generated_production,
    generated_dev_candidate,
    handwritten_production,
    unsupported,
};

pub const FallbackReason = enum(u8) {
    none,
    unsupported_format,
    unsupported_shape,
    unsupported_epilogue,
    unsupported_backend,
    generated_artifact_missing,
    generated_runtime_not_wired,
    tensor_core_repack_required,
};

pub const QuantKernelLowering = struct {
    plan_id: QuantKernelPlanId,
    backend: Backend,
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
    schedule: QuantKernelSchedule,
    production_route: LoweringRoute,
    candidate_route: LoweringRoute,
    production_kernel_id: []const u8,
    fallback_reason: FallbackReason,
    kernel_id: []const u8,
    candidate_source_path: []const u8,
};

pub const PlanCounters = struct {
    quant_kernel_planned_ops: usize = 0,
    quant_kernel_handwritten_production: usize = 0,
    quant_kernel_generated_production: usize = 0,
    quant_kernel_unsupported_routes: usize = 0,
    quant_kernel_generated_candidates: usize = 0,
    quant_kernel_fallback_generated_artifact_missing: usize = 0,
    quant_kernel_fallback_generated_runtime_not_wired: usize = 0,
    quant_kernel_fallback_unsupported_format: usize = 0,
    quant_kernel_fallback_unsupported_shape: usize = 0,
    quant_kernel_fallback_unsupported_epilogue: usize = 0,
    quant_kernel_fallback_unsupported_backend: usize = 0,
    quant_kernel_fallback_tensor_core_repack_required: usize = 0,
    quant_kernel_fallback_unsupported: usize = 0,
};

pub const QuantKernelRegistry = struct {
    entries: []const QuantKernelLowering,

    pub fn lookup(
        self: QuantKernelRegistry,
        backend: Backend,
        format: quant_matmul.Format,
        row_bucket: quant_matmul.RowBucket,
        epilogue: Epilogue,
        dispatch: quant_matmul.DispatchKind,
    ) ?QuantKernelLowering {
        for (self.entries) |entry| {
            if (entry.backend == backend and
                entry.format == format and
                entry.row_bucket == row_bucket and
                entry.epilogue == epilogue and
                entry.schedule.dispatch == dispatch)
            {
                return entry;
            }
        }
        return null;
    }
};

pub const CoverageCase = struct {
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
};

pub const ConformanceCase = struct {
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
    dispatch: quant_matmul.DispatchKind,
    reference_supported: bool,
    reference_tensor_type: tensor_types.TensorType,
    tolerance_abs: f32,
    cuda_route: LoweringRoute,
    cuda_candidate_route: LoweringRoute,
    cuda_fallback_reason: FallbackReason,
    metal_route: LoweringRoute,
    metal_candidate_route: LoweringRoute,
    metal_fallback_reason: FallbackReason,
};

pub const BenchmarkCase = struct {
    name: []const u8,
    backend: Backend,
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
    generated_kernel_id: []const u8,
    generated_source_path: []const u8,
    generated_source_fingerprint: u64,
    generated_ptx_path: []const u8,
    generated_ptx_command: []const u8,
    benchmark_command: []const u8,
    generated_ptx_arg: []const u8,
    handwritten_baseline: []const u8,
    correctness_tolerance_abs: f32,
    minimum_speedup: f64,
    correctness_evidence_path: []const u8,
    benchmark_evidence_path: []const u8,
    benchmark_mode: []const u8,
    production_enabled: bool,
};

pub const MetalBenchmarkShape = enum {
    small,
    wide,
};

pub const MetalBenchmarkDims = struct {
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    tolerance_abs: f32,
};

pub const MetalProductionBenchmarkCase = struct {
    name: []const u8,
    kernel_id: []const u8,
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
    dispatch: quant_matmul.DispatchKind,
    shape: MetalBenchmarkShape,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    threads_per_threadgroup: usize,
    cols_per_threadgroup: usize,
    tolerance_abs: f32,
    generated_source_path: []const u8,
    generated_source_fingerprint: u64,
    check_command: []const u8,
    production_kernel_id: []const u8,
    benchmark_command: []const u8,
};

const BenchmarkEvidence = struct {
    kernel_id: []const u8,
    generated_source_path: []const u8,
    generated_source_fingerprint: u64,
    generated_ptx_path: []const u8,
    generated_ptx_command: []const u8,
    benchmark_command: []const u8,
    correctness_evidence_path: []const u8,
    benchmark_evidence_path: []const u8,
    benchmark_mode: []const u8,
    repeat_runs: usize,
    correctness_passed: bool,
    benchmark_passed: bool,
    measured_speedup: f64,
};

const MetalRuntimeEvidence = struct {
    kernel_id: []const u8,
    source_path: []const u8,
    artifact_source_path: []const u8,
    source_fingerprint: u64,
    check_command: []const u8,
    runtime_evidence_command: []const u8,
    promotion_check_command: []const u8,
    repeat_runs: usize,
    correctness_passed: bool,
    generated_route_checked: bool = false,
    provider_route_checked: bool = false,
    benchmark_passed: bool,
    measured_speedup: f64,
    // Minimum speedup across every promoted case in the evidence file.
    minimum_repeat_speedup: f64,
    production_enabled: bool,
    promotion_ready: bool,
};

pub const MetalPromotionBlockerEvidence = struct {
    kernel_id: []const u8,
    blocker: []const u8,
    evidence_path: []const u8 = "",
    requires_production_regression_clear: bool = false,
};

const BenchmarkManifest = struct {
    schema: []const u8,
    benchmark_count: usize,
    evidence_count: usize,
    metal_evidence_count: usize,
    metal_promotion_warmup_repeat_runs: u32,
    metal_production_regression_expected_kernel_count: usize,
    metal_production_regression_expected_case_count: usize,
    metal_production_regression_expected_route_ready_count: usize,
    metal_production_regression_case_fingerprint: u64,
    metal_production_regression_build_command: []const u8,
    metal_production_regression_evidence_command: []const u8,
    metal_production_regression_cases: []const MetalProductionBenchmarkManifestRecord,
    benchmarks: []const BenchmarkManifestRecord,
    evidence_records: []const BenchmarkEvidence,
    metal_evidence_records: []const MetalRuntimeEvidence,
};

const BenchmarkManifestRecord = struct {
    name: []const u8,
    backend: []const u8,
    format: []const u8,
    row_bucket: []const u8,
    epilogue: []const u8,
    dispatch: []const u8,
    generated_kernel_id: []const u8,
    generated_source_path: []const u8,
    generated_source_fingerprint: u64,
    generated_ptx_path: []const u8,
    generated_ptx_command: []const u8,
    benchmark_command: []const u8,
    generated_ptx_arg: []const u8,
    handwritten_baseline: []const u8,
    correctness_tolerance_abs: f32,
    minimum_speedup: f64,
    measured_speedup: f64,
    correctness_evidence_path: []const u8,
    benchmark_evidence_path: []const u8,
    benchmark_mode: []const u8,
    production_enabled: bool,
    promotion_ready: bool,
    promotion_blocker: []const u8,
};

const MetalProductionBenchmarkManifestRecord = struct {
    name: []const u8,
    kernel_id: []const u8,
    format: []const u8,
    row_bucket: []const u8,
    epilogue: []const u8,
    dispatch: []const u8,
    shape: []const u8,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    threads_per_threadgroup: usize,
    cols_per_threadgroup: usize,
    tolerance_abs: f32,
    generated_source_path: []const u8,
    generated_source_fingerprint: u64,
    check_command: []const u8,
    production_kernel_id: []const u8,
    benchmark_command: []const u8,
};

const ArtifactManifest = struct {
    schema: []const u8,
    artifact_count: usize,
    checked_in_metal_evidence_count: usize,
    metal_promotion_blocker_evidence_count: usize,
    metal_promotion_blocker_evidence_path_count: usize,
    metal_promotion_blocker_evidence_expected_case_count: usize,
    metal_promotion_blocker_evidence_expected_route_ready_count: usize,
    metal_promotion_blocker_check_command_count: usize,
    metal_promotion_blocker_skipped_no_path_count: usize,
    metal_promotion_blocker_cleared_requires_checked_in_evidence: bool,
    metal_promotion_blocker_speedup_gate_missing_count: usize,
    metal_promotion_blocker_unstable_benchmark_timing_count: usize,
    metal_promotion_blocker_unsupported_handwritten_count: usize,
    metal_unsupported_handwritten_baseline_blocks_promotion: bool,
    metal_unsupported_handwritten_baseline_uses_runtime_route_all_evidence: bool,
    metal_unsupported_handwritten_baseline_has_promotion_evidence_path: bool,
    metal_local_check_command: []const u8,
    metal_model_local_check_command: []const u8,
    metal_model_generated_route_check_command: []const u8,
    metal_model_generated_q8_0_small_batch_min: usize,
    metal_model_generated_q4_0_small_batch_min: usize,
    metal_industry_local_check_command: []const u8,
    metal_runtime_route_all_build_command: []const u8,
    metal_runtime_route_all_evidence_command: []const u8,
    metal_runtime_route_all_check_command: []const u8,
    metal_runtime_route_all_expected_case_count: usize,
    metal_runtime_route_all_expected_route_ready_count: usize,
    metal_runtime_route_all_expected_provider_route_count: usize,
    metal_production_regression_expected_kernel_count: usize,
    metal_production_regression_expected_case_count: usize,
    metal_production_regression_expected_route_ready_count: usize,
    metal_promotion_warmup_repeat_runs: u32,
    metal_production_regression_route_ready_is_hard_gate: bool,
    metal_production_regression_missing_provider_route_is_hard_gate: bool,
    metal_production_regression_speedup_gate_missing_is_hard_gate: bool,
    metal_production_regression_unstable_benchmark_timing_is_hard_gate: bool,
    metal_production_regression_build_command: []const u8,
    metal_production_regression_evidence_command: []const u8,
    metal_blocker_strict_check_command: []const u8,
    artifacts: []const ArtifactManifestRecord,
    metal_evidence_records: []const MetalRuntimeEvidence,
};

const ArtifactManifestRecord = struct {
    backend: []const u8,
    format: []const u8,
    row_bucket: []const u8,
    epilogue: []const u8,
    kernel_id: []const u8,
    source_path: []const u8,
    generated_source_path: []const u8,
    artifact_source_path: []const u8,
    generated_source_fingerprint: u64,
    check_command: []const u8,
    generated_check_command: []const u8,
    runtime_evidence_command: []const u8,
    runtime_route_evidence_command: []const u8,
    promotion_evidence_command: []const u8,
    promotion_check_command: []const u8,
    promotion_policy: []const u8,
    production_enabled: bool,
    runtime_wired: bool,
    runtime_gate_env: []const u8,
    production_regression_checked: bool,
    production_regression_command: []const u8,
    metal_promotion_min_speedup: f64,
    metal_promotion_repeat_runs: usize,
    metal_promotion_warmup_repeat_runs: u32,
    candidate_status: []const u8,
    promotion_ready: bool,
    promotion_blocker: []const u8,
    promotion_blocker_evidence_path: []const u8,
    promotion_blocker_check_command: []const u8,
    promotion_blocker_requires_production_regression_clear: bool,
};

const SpecManifest = struct {
    schema: []const u8,
    format_count: usize,
    specs: []const SpecManifestRecord,
};

const SpecManifestRecord = struct {
    format: []const u8,
    reference_tensor_type: []const u8,
    block_values: usize,
    block_bytes: usize,
    block_fields: []const BlockField,
    decode_ops: []const DecodeOp,
    supported_schedules: []const quant_matmul.DispatchKind,
    supported_epilogues: []const Epilogue,
    supported_backends: []const Backend,
    accumulator_dtype: []const u8,
    output_dtype: []const u8,
};

const ConformanceManifest = struct {
    schema: []const u8,
    case_count: usize,
    format_count: usize,
    row_bucket_count: usize,
    epilogue_count: usize,
    backend_count: usize,
    cuda_route_summary: PlanCounters,
    metal_route_summary: PlanCounters,
    cases: []const ConformanceManifestRecord,
};

const ConformanceManifestRecord = struct {
    format: []const u8,
    block_values: usize,
    block_bytes: usize,
    accumulator_dtype: []const u8,
    output_dtype: []const u8,
    row_bucket: []const u8,
    epilogue: []const u8,
    dispatch: []const u8,
    tile_rows: usize,
    tile_cols: usize,
    vector_width: usize,
    threads_per_block: usize,
    shared_memory_bytes: usize,
    register_pressure_hint: u8,
    tensor_core_eligible: bool,
    cuda_candidate_tile_rows: usize,
    cuda_candidate_tile_cols: usize,
    cuda_candidate_threads_per_block: usize,
    metal_candidate_tile_rows: usize,
    metal_candidate_tile_cols: usize,
    metal_candidate_threads_per_block: usize,
    reference_supported: bool,
    reference_tensor_type: []const u8,
    tolerance_abs: f32,
    cuda_route: []const u8,
    cuda_candidate_route: []const u8,
    cuda_production_kernel_id: []const u8,
    cuda_candidate_kernel_id: []const u8,
    cuda_candidate_source_path: []const u8,
    cuda_candidate_source_fingerprint: u64,
    cuda_fallback_reason: []const u8,
    metal_route: []const u8,
    metal_candidate_route: []const u8,
    metal_production_kernel_id: []const u8,
    metal_candidate_kernel_id: []const u8,
    metal_candidate_source_path: []const u8,
    metal_candidate_source_fingerprint: u64,
    metal_fallback_reason: []const u8,
};

pub const GeneratedArtifact = struct {
    backend: Backend,
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
    kernel_id: []const u8,
    source_path: []const u8,
    check_command: []const u8,
    generated_source_path: []const u8 = "",
    generated_check_command: []const u8 = "",
    runtime_evidence_command: []const u8 = "",
    promotion_evidence_command: []const u8 = "",
    promotion_check_command: []const u8 = "",
    production_enabled: bool,
};

pub const QuantKernelCompileRequest = struct {
    backend: Backend,
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
};

pub const QuantKernelCompiledSource = struct {
    request: QuantKernelCompileRequest,
    spec: QuantKernelSpec,
    ir: QuantKernelIR,
    lowering: QuantKernelLowering,
    artifact: GeneratedArtifact,
    source: []const u8,
    source_path: []const u8,
    artifact_source_path: []const u8,
    check_command: []const u8,
    runtime_gate_env: ?[*:0]const u8,
    production_enabled: bool,
};

pub const EmittedCompiledSource = struct {
    data: []const u8,
    owned: bool = false,

    pub fn deinit(self: EmittedCompiledSource, allocator: std.mem.Allocator) void {
        if (self.owned) allocator.free(self.data);
    }
};

const CodecFormatCoverage = struct {
    tensor_type: tensor_types.KnownTensorType,
    format: quant_matmul.Format,
};

const RouteExpectation = struct {
    backend: Backend,
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
    dispatch: quant_matmul.DispatchKind,
    production_route: LoweringRoute,
    candidate_route: LoweringRoute,
    fallback_reason: FallbackReason,
};

const no_backends = [_]Backend{};

const q1_0_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "qs", .offset = 2, .bytes = 16 },
};

const i2_s_block_fields = [_]BlockField{
    .{ .name = "qs", .offset = 0, .bytes = 32 },
};

const i8_s_block_fields = [_]BlockField{
    .{ .name = "q", .offset = 0, .bytes = 1 },
};

const iq4_nl_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "qs", .offset = 2, .bytes = 16 },
};

const iq4_xs_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "scales_h", .offset = 2, .bytes = 2 },
    .{ .name = "scales_l", .offset = 4, .bytes = 4 },
    .{ .name = "qs", .offset = 8, .bytes = 128 },
};

const mxfp4_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 1 },
    .{ .name = "qs", .offset = 1, .bytes = 16 },
};

const nvfp4_block_fields = [_]BlockField{
    .{ .name = "scales", .offset = 0, .bytes = 4 },
    .{ .name = "qs", .offset = 4, .bytes = 32 },
};

const tq1_0_block_fields = [_]BlockField{
    .{ .name = "qs", .offset = 0, .bytes = 48 },
    .{ .name = "qh", .offset = 48, .bytes = 4 },
    .{ .name = "d", .offset = 52, .bytes = 2 },
};

const tq2_0_block_fields = [_]BlockField{
    .{ .name = "qs", .offset = 0, .bytes = 64 },
    .{ .name = "d", .offset = 64, .bytes = 2 },
};

const iq2_xxs_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "qs", .offset = 2, .bytes = 64 },
};

const iq2_xs_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "qs", .offset = 2, .bytes = 64 },
    .{ .name = "scales", .offset = 66, .bytes = 8 },
};

const iq2_s_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "qs", .offset = 2, .bytes = 64 },
    .{ .name = "qh", .offset = 66, .bytes = 8 },
    .{ .name = "scales", .offset = 74, .bytes = 8 },
};

const iq3_xxs_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "qs", .offset = 2, .bytes = 96 },
};

const iq3_s_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "qs", .offset = 2, .bytes = 64 },
    .{ .name = "qh", .offset = 66, .bytes = 8 },
    .{ .name = "signs", .offset = 74, .bytes = 32 },
    .{ .name = "scales", .offset = 106, .bytes = 4 },
};

const iq1_s_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "qs", .offset = 2, .bytes = 32 },
    .{ .name = "qh", .offset = 34, .bytes = 16 },
};

const iq1_m_block_fields = [_]BlockField{
    .{ .name = "qs", .offset = 0, .bytes = 32 },
    .{ .name = "qh", .offset = 32, .bytes = 16 },
    .{ .name = "scales", .offset = 48, .bytes = 8 },
};

const q4_0_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "qs", .offset = 2, .bytes = 16 },
};

const q4_1_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "m", .offset = 2, .bytes = 2 },
    .{ .name = "qs", .offset = 4, .bytes = 16 },
};

const q5_0_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "qh", .offset = 2, .bytes = 4 },
    .{ .name = "qs", .offset = 6, .bytes = 16 },
};

const q5_1_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "m", .offset = 2, .bytes = 2 },
    .{ .name = "qh", .offset = 4, .bytes = 4 },
    .{ .name = "qs", .offset = 8, .bytes = 16 },
};

const q2_k_block_fields = [_]BlockField{
    .{ .name = "scales", .offset = 0, .bytes = 16 },
    .{ .name = "d", .offset = 16, .bytes = 2 },
    .{ .name = "dmin", .offset = 18, .bytes = 2 },
    .{ .name = "qs", .offset = 20, .bytes = 64 },
};

const q3_k_block_fields = [_]BlockField{
    .{ .name = "hmask", .offset = 0, .bytes = 32 },
    .{ .name = "qs", .offset = 32, .bytes = 64 },
    .{ .name = "scales", .offset = 96, .bytes = 12 },
    .{ .name = "d", .offset = 108, .bytes = 2 },
};

const q4_k_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "dmin", .offset = 2, .bytes = 2 },
    .{ .name = "scales", .offset = 4, .bytes = 12 },
    .{ .name = "qs", .offset = 16, .bytes = 128 },
};

const q5_k_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "dmin", .offset = 2, .bytes = 2 },
    .{ .name = "scales", .offset = 4, .bytes = 12 },
    .{ .name = "qh", .offset = 16, .bytes = 32 },
    .{ .name = "ql", .offset = 48, .bytes = 128 },
};

const q6_k_block_fields = [_]BlockField{
    .{ .name = "ql", .offset = 0, .bytes = 128 },
    .{ .name = "qh", .offset = 128, .bytes = 64 },
    .{ .name = "scales", .offset = 192, .bytes = 16 },
    .{ .name = "d", .offset = 208, .bytes = 2 },
};

const q8_0_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "qs", .offset = 2, .bytes = 32 },
};

const q8_1_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 2 },
    .{ .name = "sum", .offset = 2, .bytes = 2 },
    .{ .name = "qs", .offset = 4, .bytes = 32 },
};

const q8_k_block_fields = [_]BlockField{
    .{ .name = "d", .offset = 0, .bytes = 4 },
    .{ .name = "qs", .offset = 4, .bytes = 256 },
    .{ .name = "bsums", .offset = 260, .bytes = 32 },
};

const q4_0_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_q4_0 {d,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d)" },
    .{ .kind = .extract_quant_lane, .expression = "q = low_or_high_nibble(qs[lane]) - 8" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const q4_1_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_q4_1 {d,m,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d)" },
    .{ .kind = .extract_min, .expression = "min = fp16(m)" },
    .{ .kind = .extract_quant_lane, .expression = "q = low_or_high_nibble(qs[lane])" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q + min" },
};

const q5_0_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_q5_0 {d,qh,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d)" },
    .{ .kind = .extract_quant_lane, .expression = "q = (low_or_high_nibble(qs[lane]) | high_bit(qh[lane])) - 16" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const q5_1_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_q5_1 {d,m,qh,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d)" },
    .{ .kind = .extract_min, .expression = "min = fp16(m)" },
    .{ .kind = .extract_quant_lane, .expression = "q = low_or_high_nibble(qs[lane]) | high_bit(qh[lane])" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q + min" },
};

const q1_0_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_q1_0 {d,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d)" },
    .{ .kind = .extract_quant_lane, .expression = "q = bit(qs[lane]) ? 1 : -1" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const i2_s_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_i2_s {qs}" },
    .{ .kind = .extract_scale, .expression = "scale = 1.0" },
    .{ .kind = .extract_quant_lane, .expression = "q = ternary(two_bit(qs[group]))" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const i8_s_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_i8_s {q}" },
    .{ .kind = .extract_scale, .expression = "scale = 1.0" },
    .{ .kind = .extract_quant_lane, .expression = "q = i8(q)" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const iq4_nl_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_iq4_nl {d,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d)" },
    .{ .kind = .extract_quant_lane, .expression = "q = iq4_nl_table[low_or_high_nibble(qs[lane])]" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const iq4_xs_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_iq4_xs {d,scales_h,scales_l,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d) * (packed_i6_scale[group] - 32)" },
    .{ .kind = .extract_quant_lane, .expression = "q = iq4_nl_table[low_or_high_nibble(qs[lane])]" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const mxfp4_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_mxfp4 {d,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = e8m0_half(d)" },
    .{ .kind = .extract_quant_lane, .expression = "q = e2m1_table[low_or_high_nibble(qs[lane])]" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const nvfp4_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_nvfp4 {scales,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = ue4m3_half(scales[subblock])" },
    .{ .kind = .extract_quant_lane, .expression = "q = e2m1_table[low_or_high_nibble(qs[lane])]" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const tq1_0_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_tq1_0 {qs,qh,d}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d)" },
    .{ .kind = .extract_quant_lane, .expression = "q = ternary_base3(qs,qh,lane) - 1" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const tq2_0_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_tq2_0 {qs,d}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d)" },
    .{ .kind = .extract_quant_lane, .expression = "q = two_bit(qs[lane]) - 1" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const iq2_xxs_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_iq2_xxs {d,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d) * packed_4bit_scale[group] * 0.25" },
    .{ .kind = .extract_quant_lane, .expression = "q = iq2_xxs_grid[packed_grid] * sign_mask" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const iq2_xs_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_iq2_xs {d,qs,scales}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d) * packed_nibble_scale[pair] * 0.25" },
    .{ .kind = .extract_quant_lane, .expression = "q = iq2_xs_grid[packed_grid] * sign_mask" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const iq2_s_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_iq2_s {d,qs,qh,scales}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d) * packed_nibble_scale[pair] * 0.25" },
    .{ .kind = .extract_quant_lane, .expression = "q = iq2_s_grid[grid_low|high_bits] * sign_bits" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const iq3_xxs_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_iq3_xxs {d,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d) * packed_4bit_scale[group] * 0.5" },
    .{ .kind = .extract_quant_lane, .expression = "q = iq3_xxs_grid[grid_index] * sign_mask" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const iq3_s_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_iq3_s {d,qs,qh,signs,scales}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d) * packed_nibble_scale[pair]" },
    .{ .kind = .extract_quant_lane, .expression = "q = iq3_s_grid[grid_low|high_bits] * sign_bits" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const iq1_s_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_iq1_s {d,qs,qh}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d) * packed_scale[group]" },
    .{ .kind = .extract_quant_lane, .expression = "q = iq1_s_grid[grid_low|high_bits] + delta" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const iq1_m_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_iq1_m {qs,qh,scales}" },
    .{ .kind = .extract_scale, .expression = "scale = packed_fp16(scales) * subscale[group]" },
    .{ .kind = .extract_quant_lane, .expression = "q = iq1_s_grid[grid_low|high_bits] + delta" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const q2_k_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_q2_K {scales,d,dmin,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d) * packed_scale[subblock]" },
    .{ .kind = .extract_min, .expression = "min = fp16(dmin) * packed_min[subblock]" },
    .{ .kind = .extract_quant_lane, .expression = "q = two_bit(qs[lane])" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q - min" },
};

const q3_k_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_q3_K {hmask,qs,scales,d}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d) * packed_i6_scale[subblock]" },
    .{ .kind = .extract_quant_lane, .expression = "q = two_bit(qs[lane]) | high_bit(hmask[lane])" },
    .{ .kind = .dequant_lane, .expression = "value = scale * (q - 4)" },
};

const q4_k_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_q4_K {d,dmin,scales,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d) * packed_scale[subblock]" },
    .{ .kind = .extract_min, .expression = "min = fp16(dmin) * packed_min[subblock]" },
    .{ .kind = .extract_quant_lane, .expression = "q = low_or_high_nibble(qs[lane])" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q - min" },
};

const q5_k_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_q5_K {d,dmin,scales,qh,ql}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d) * packed_scale[subblock]" },
    .{ .kind = .extract_min, .expression = "min = fp16(dmin) * packed_min[subblock]" },
    .{ .kind = .extract_quant_lane, .expression = "q = low_or_high_nibble(ql[lane]) | high_bit(qh[lane])" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q - min" },
};

const q6_k_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_q6_K {ql,qh,scales,d}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d) * i8(scales[subblock])" },
    .{ .kind = .extract_quant_lane, .expression = "q = (low4(ql[lane]) | high2(qh[lane])) - 32" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const q8_0_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_q8_0 {d,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d)" },
    .{ .kind = .extract_quant_lane, .expression = "q = i8(qs[lane])" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const q8_1_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_q8_1 {d,sum,qs}" },
    .{ .kind = .extract_scale, .expression = "scale = fp16(d)" },
    .{ .kind = .extract_quant_lane, .expression = "q = i8(qs[lane])" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const q8_k_decode_ops = [_]DecodeOp{
    .{ .kind = .load_packed_bytes, .expression = "load block_q8_K {d,qs,bsums}" },
    .{ .kind = .extract_scale, .expression = "scale = f32(d)" },
    .{ .kind = .extract_quant_lane, .expression = "q = i8(qs[lane])" },
    .{ .kind = .dequant_lane, .expression = "value = scale * q" },
};

const codec_format_coverage = [_]CodecFormatCoverage{
    .{ .tensor_type = .Q1_0, .format = .q1_0 },
    .{ .tensor_type = .I2_S, .format = .i2_s },
    .{ .tensor_type = .I8_S, .format = .i8_s },
    .{ .tensor_type = .Q4_0, .format = .q4_0 },
    .{ .tensor_type = .Q4_1, .format = .q4_1 },
    .{ .tensor_type = .Q5_0, .format = .q5_0 },
    .{ .tensor_type = .Q5_1, .format = .q5_1 },
    .{ .tensor_type = .Q8_0, .format = .q8_0 },
    .{ .tensor_type = .Q8_1, .format = .q8_1 },
    .{ .tensor_type = .Q2_K, .format = .q2_k },
    .{ .tensor_type = .Q3_K, .format = .q3_k },
    .{ .tensor_type = .Q4_K, .format = .q4_k },
    .{ .tensor_type = .Q5_K, .format = .q5_k },
    .{ .tensor_type = .Q6_K, .format = .q6_k },
    .{ .tensor_type = .Q8_K, .format = .q8_k },
    .{ .tensor_type = .TQ1_0, .format = .tq1_0 },
    .{ .tensor_type = .TQ2_0, .format = .tq2_0 },
    .{ .tensor_type = .IQ2_XXS, .format = .iq2_xxs },
    .{ .tensor_type = .IQ2_XS, .format = .iq2_xs },
    .{ .tensor_type = .IQ2_S, .format = .iq2_s },
    .{ .tensor_type = .IQ3_XXS, .format = .iq3_xxs },
    .{ .tensor_type = .IQ3_S, .format = .iq3_s },
    .{ .tensor_type = .IQ1_S, .format = .iq1_s },
    .{ .tensor_type = .IQ1_M, .format = .iq1_m },
    .{ .tensor_type = .IQ4_NL, .format = .iq4_nl },
    .{ .tensor_type = .IQ4_XS, .format = .iq4_xs },
    .{ .tensor_type = .MXFP4, .format = .mxfp4 },
    .{ .tensor_type = .NVFP4, .format = .nvfp4 },
};
const descriptor_formats = buildDescriptorFormats();
const first_formats = descriptor_formats;
const first_row_buckets = [_]quant_matmul.RowBucket{ .rows_1, .rows_2_8, .rows_9_64, .rows_65_plus };
const first_schedules = [_]quant_matmul.DispatchKind{ .mmv, .small_batch, .mm };
const first_epilogues = [_]Epilogue{ .none, .bias, .bias_gelu, .pair };
const q8_0_epilogues = [_]Epilogue{ .none, .bias, .bias_gelu, .pair, .relu };
const coverage_epilogues = [_]Epilogue{ .none, .bias, .bias_gelu, .pair, .triple, .relu, .gelu, .add, .argmax };
const no_bias_epilogues = [_]Epilogue{.none};
const q2_k_epilogues = [_]Epilogue{ .none, .bias, .bias_gelu };
const q3_k_epilogues = [_]Epilogue{ .none, .bias, .bias_gelu };
const k_quant_epilogues = [_]Epilogue{ .none, .bias, .bias_gelu };
const first_backends = [_]Backend{ .cuda, .metal };
const metal_backends = [_]Backend{.metal};

const q4_0_spec = QuantKernelSpec{
    .format = .q4_0,
    .block_values = blockValuesForFormat(.q4_0),
    .block_bytes = blockBytesForFormat(.q4_0),
    .block_fields = &q4_0_block_fields,
    .decode_ops = &q4_0_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &first_backends,
};

const q4_1_spec = QuantKernelSpec{
    .format = .q4_1,
    .block_values = blockValuesForFormat(.q4_1),
    .block_bytes = blockBytesForFormat(.q4_1),
    .block_fields = &q4_1_block_fields,
    .decode_ops = &q4_1_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &metal_backends,
};

const q5_0_spec = QuantKernelSpec{
    .format = .q5_0,
    .block_values = blockValuesForFormat(.q5_0),
    .block_bytes = blockBytesForFormat(.q5_0),
    .block_fields = &q5_0_block_fields,
    .decode_ops = &q5_0_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &metal_backends,
};

const q5_1_spec = QuantKernelSpec{
    .format = .q5_1,
    .block_values = blockValuesForFormat(.q5_1),
    .block_bytes = blockBytesForFormat(.q5_1),
    .block_fields = &q5_1_block_fields,
    .decode_ops = &q5_1_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &metal_backends,
};

const q1_0_spec = QuantKernelSpec{
    .format = .q1_0,
    .block_values = blockValuesForFormat(.q1_0),
    .block_bytes = blockBytesForFormat(.q1_0),
    .block_fields = &q1_0_block_fields,
    .decode_ops = &q1_0_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &metal_backends,
};

const i2_s_spec = QuantKernelSpec{
    .format = .i2_s,
    .block_values = blockValuesForFormat(.i2_s),
    .block_bytes = blockBytesForFormat(.i2_s),
    .block_fields = &i2_s_block_fields,
    .decode_ops = &i2_s_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const i8_s_spec = QuantKernelSpec{
    .format = .i8_s,
    .block_values = blockValuesForFormat(.i8_s),
    .block_bytes = blockBytesForFormat(.i8_s),
    .block_fields = &i8_s_block_fields,
    .decode_ops = &i8_s_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const iq4_nl_spec = QuantKernelSpec{
    .format = .iq4_nl,
    .block_values = blockValuesForFormat(.iq4_nl),
    .block_bytes = blockBytesForFormat(.iq4_nl),
    .block_fields = &iq4_nl_block_fields,
    .decode_ops = &iq4_nl_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const iq4_xs_spec = QuantKernelSpec{
    .format = .iq4_xs,
    .block_values = blockValuesForFormat(.iq4_xs),
    .block_bytes = blockBytesForFormat(.iq4_xs),
    .block_fields = &iq4_xs_block_fields,
    .decode_ops = &iq4_xs_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const mxfp4_spec = QuantKernelSpec{
    .format = .mxfp4,
    .block_values = blockValuesForFormat(.mxfp4),
    .block_bytes = blockBytesForFormat(.mxfp4),
    .block_fields = &mxfp4_block_fields,
    .decode_ops = &mxfp4_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const nvfp4_spec = QuantKernelSpec{
    .format = .nvfp4,
    .block_values = blockValuesForFormat(.nvfp4),
    .block_bytes = blockBytesForFormat(.nvfp4),
    .block_fields = &nvfp4_block_fields,
    .decode_ops = &nvfp4_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const tq1_0_spec = QuantKernelSpec{
    .format = .tq1_0,
    .block_values = blockValuesForFormat(.tq1_0),
    .block_bytes = blockBytesForFormat(.tq1_0),
    .block_fields = &tq1_0_block_fields,
    .decode_ops = &tq1_0_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const tq2_0_spec = QuantKernelSpec{
    .format = .tq2_0,
    .block_values = blockValuesForFormat(.tq2_0),
    .block_bytes = blockBytesForFormat(.tq2_0),
    .block_fields = &tq2_0_block_fields,
    .decode_ops = &tq2_0_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const iq2_xxs_spec = QuantKernelSpec{
    .format = .iq2_xxs,
    .block_values = blockValuesForFormat(.iq2_xxs),
    .block_bytes = blockBytesForFormat(.iq2_xxs),
    .block_fields = &iq2_xxs_block_fields,
    .decode_ops = &iq2_xxs_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const iq2_xs_spec = QuantKernelSpec{
    .format = .iq2_xs,
    .block_values = blockValuesForFormat(.iq2_xs),
    .block_bytes = blockBytesForFormat(.iq2_xs),
    .block_fields = &iq2_xs_block_fields,
    .decode_ops = &iq2_xs_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const iq2_s_spec = QuantKernelSpec{
    .format = .iq2_s,
    .block_values = blockValuesForFormat(.iq2_s),
    .block_bytes = blockBytesForFormat(.iq2_s),
    .block_fields = &iq2_s_block_fields,
    .decode_ops = &iq2_s_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const iq3_xxs_spec = QuantKernelSpec{
    .format = .iq3_xxs,
    .block_values = blockValuesForFormat(.iq3_xxs),
    .block_bytes = blockBytesForFormat(.iq3_xxs),
    .block_fields = &iq3_xxs_block_fields,
    .decode_ops = &iq3_xxs_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const iq3_s_spec = QuantKernelSpec{
    .format = .iq3_s,
    .block_values = blockValuesForFormat(.iq3_s),
    .block_bytes = blockBytesForFormat(.iq3_s),
    .block_fields = &iq3_s_block_fields,
    .decode_ops = &iq3_s_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const iq1_s_spec = QuantKernelSpec{
    .format = .iq1_s,
    .block_values = blockValuesForFormat(.iq1_s),
    .block_bytes = blockBytesForFormat(.iq1_s),
    .block_fields = &iq1_s_block_fields,
    .decode_ops = &iq1_s_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const iq1_m_spec = QuantKernelSpec{
    .format = .iq1_m,
    .block_values = blockValuesForFormat(.iq1_m),
    .block_bytes = blockBytesForFormat(.iq1_m),
    .block_fields = &iq1_m_block_fields,
    .decode_ops = &iq1_m_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &no_backends,
};

const q2_k_spec = QuantKernelSpec{
    .format = .q2_k,
    .block_values = blockValuesForFormat(.q2_k),
    .block_bytes = blockBytesForFormat(.q2_k),
    .block_fields = &q2_k_block_fields,
    .decode_ops = &q2_k_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &q2_k_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &metal_backends,
};

const q3_k_spec = QuantKernelSpec{
    .format = .q3_k,
    .block_values = blockValuesForFormat(.q3_k),
    .block_bytes = blockBytesForFormat(.q3_k),
    .block_fields = &q3_k_block_fields,
    .decode_ops = &q3_k_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &q3_k_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &metal_backends,
};

const q4_k_spec = QuantKernelSpec{
    .format = .q4_k,
    .block_values = blockValuesForFormat(.q4_k),
    .block_bytes = blockBytesForFormat(.q4_k),
    .block_fields = &q4_k_block_fields,
    .decode_ops = &q4_k_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &first_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &first_backends,
};

const q5_k_spec = QuantKernelSpec{
    .format = .q5_k,
    .block_values = blockValuesForFormat(.q5_k),
    .block_bytes = blockBytesForFormat(.q5_k),
    .block_fields = &q5_k_block_fields,
    .decode_ops = &q5_k_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &k_quant_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &metal_backends,
};

const q6_k_spec = QuantKernelSpec{
    .format = .q6_k,
    .block_values = blockValuesForFormat(.q6_k),
    .block_bytes = blockBytesForFormat(.q6_k),
    .block_fields = &q6_k_block_fields,
    .decode_ops = &q6_k_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &k_quant_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &first_backends,
};

const q8_0_spec = QuantKernelSpec{
    .format = .q8_0,
    .block_values = blockValuesForFormat(.q8_0),
    .block_bytes = blockBytesForFormat(.q8_0),
    .block_fields = &q8_0_block_fields,
    .decode_ops = &q8_0_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &q8_0_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &first_backends,
};

const q8_1_spec = QuantKernelSpec{
    .format = .q8_1,
    .block_values = blockValuesForFormat(.q8_1),
    .block_bytes = blockBytesForFormat(.q8_1),
    .block_fields = &q8_1_block_fields,
    .decode_ops = &q8_1_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &metal_backends,
};

const q8_k_spec = QuantKernelSpec{
    .format = .q8_k,
    .block_values = blockValuesForFormat(.q8_k),
    .block_bytes = blockBytesForFormat(.q8_k),
    .block_fields = &q8_k_block_fields,
    .decode_ops = &q8_k_decode_ops,
    .supported_schedules = &first_schedules,
    .supported_epilogues = &no_bias_epilogues,
    .accumulator_dtype = .f32,
    .output_dtype = .f32,
    .supported_backends = &metal_backends,
};

const ir_ops_basic = [_]IROp{
    .load_input_row,
    .load_quant_block,
    .decode_quant_lane,
    .multiply_accumulate,
    .reduce_accumulator,
    .write_output,
};

const ir_ops_bias = [_]IROp{
    .load_input_row,
    .load_quant_block,
    .decode_quant_lane,
    .multiply_accumulate,
    .reduce_accumulator,
    .apply_bias,
    .write_output,
};

const ir_ops_bias_gelu = [_]IROp{
    .load_input_row,
    .load_quant_block,
    .decode_quant_lane,
    .multiply_accumulate,
    .reduce_accumulator,
    .apply_bias,
    .apply_gelu,
    .write_output,
};

pub const first_lazy_benchmark_evidence_path = "src/ops/cuda/generated/evidence/q4_k_small_batch_bias_gelu_benchmark.json";
pub const first_lazy_cuda_source_fingerprint = sourceFingerprint(first_lazy_cuda_source);

pub const first_lazy_benchmark = BenchmarkCase{
    .name = "q4_k_small_batch_bias_gelu",
    .backend = .cuda,
    .format = .q4_k,
    .row_bucket = .rows_2_8,
    .epilogue = .bias_gelu,
    .generated_kernel_id = "antfly_q4_k_small_batch_bias_gelu_f32_v1",
    .generated_source_path = "src/ops/cuda/generated/quant_kernel_q4_k_small_batch_bias_gelu.cu",
    .generated_source_fingerprint = first_lazy_cuda_source_fingerprint,
    .generated_ptx_path = "/tmp/antfly_q4_k_small_batch_bias_gelu_f32_v1.ptx",
    .generated_ptx_command = "nvcc -ptx -arch=compute_75 src/ops/cuda/generated/quant_kernel_q4_k_small_batch_bias_gelu.cu -o /tmp/antfly_q4_k_small_batch_bias_gelu_f32_v1.ptx",
    .benchmark_command = "zig-out/bin/antfly-inference bench-cuda --warmup-iters 5 --measure-iters 50 --quant-compiler-lazy-target --quant-compiler-generated-ptx /tmp/antfly_q4_k_small_batch_bias_gelu_f32_v1.ptx --quant-compiler-repeat-runs 3 --quant-compiler-evidence-out " ++ first_lazy_benchmark_evidence_path,
    .generated_ptx_arg = "--quant-compiler-generated-ptx",
    .handwritten_baseline = "termite_linear_q4_k_bias_gelu_f32_tile4_r2",
    .correctness_tolerance_abs = 0.01,
    .minimum_speedup = 1.0,
    .correctness_evidence_path = "",
    .benchmark_evidence_path = "",
    .benchmark_mode = "",
    .production_enabled = false,
};

pub const first_benchmarks = [_]BenchmarkCase{first_lazy_benchmark};
const first_benchmark_evidence = [_]BenchmarkEvidence{};
const first_metal_runtime_evidence = [_]MetalRuntimeEvidence{
    .{
        .kernel_id = first_general_metal_q2_kernel_id,
        .source_path = first_general_metal_q2_source_path,
        .artifact_source_path = first_general_metal_q2_artifact_source_path,
        .source_fingerprint = sourceFingerprint(first_general_metal_q2_source),
        .check_command = first_general_metal_q2_check_command,
        .runtime_evidence_command = first_general_metal_q2_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q2_promotion_check_command,
        .repeat_runs = metal_promotion_repeat_runs,
        .correctness_passed = true,
        .generated_route_checked = true,
        .provider_route_checked = true,
        .benchmark_passed = true,
        .measured_speedup = 2.080139,
        .minimum_repeat_speedup = 1.458720,
        .production_enabled = true,
        .promotion_ready = true,
    },
    .{
        .kernel_id = first_general_metal_q3_kernel_id,
        .source_path = first_general_metal_q3_source_path,
        .artifact_source_path = first_general_metal_q3_artifact_source_path,
        .source_fingerprint = sourceFingerprint(first_general_metal_q3_source),
        .check_command = first_general_metal_q3_check_command,
        .runtime_evidence_command = first_general_metal_q3_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q3_promotion_check_command,
        .repeat_runs = metal_promotion_repeat_runs,
        .correctness_passed = true,
        .generated_route_checked = true,
        .provider_route_checked = true,
        .benchmark_passed = true,
        .measured_speedup = 3.461401,
        .minimum_repeat_speedup = 2.642921,
        .production_enabled = true,
        .promotion_ready = true,
    },
    .{
        .kernel_id = first_general_metal_q5_bias_kernel_id,
        .source_path = first_general_metal_q5_bias_source_path,
        .artifact_source_path = first_general_metal_q5_bias_artifact_source_path,
        .source_fingerprint = sourceFingerprint(first_general_metal_q5_bias_source),
        .check_command = first_general_metal_q5_bias_check_command,
        .runtime_evidence_command = first_general_metal_q5_bias_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q5_bias_promotion_check_command,
        .repeat_runs = metal_promotion_repeat_runs,
        .correctness_passed = true,
        .generated_route_checked = true,
        .provider_route_checked = true,
        .benchmark_passed = true,
        .measured_speedup = 2.257019,
        .minimum_repeat_speedup = 1.857760,
        .production_enabled = true,
        .promotion_ready = true,
    },
    .{
        .kernel_id = first_general_metal_q6_bias_kernel_id,
        .source_path = first_general_metal_q6_bias_source_path,
        .artifact_source_path = first_general_metal_q6_bias_artifact_source_path,
        .source_fingerprint = sourceFingerprint(first_general_metal_q6_bias_source),
        .check_command = first_general_metal_q6_bias_check_command,
        .runtime_evidence_command = first_general_metal_q6_bias_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q6_bias_promotion_check_command,
        .repeat_runs = metal_promotion_repeat_runs,
        .correctness_passed = true,
        .generated_route_checked = true,
        .provider_route_checked = true,
        .benchmark_passed = true,
        .measured_speedup = 6.150455,
        .minimum_repeat_speedup = 5.261137,
        .production_enabled = true,
        .promotion_ready = true,
    },
    .{
        .kernel_id = first_general_metal_q6_kernel_id,
        .source_path = first_general_metal_q6_source_path,
        .artifact_source_path = first_general_metal_q6_artifact_source_path,
        .source_fingerprint = sourceFingerprint(first_general_metal_q6_source),
        .check_command = first_general_metal_q6_check_command,
        .runtime_evidence_command = first_general_metal_q6_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q6_promotion_check_command,
        .repeat_runs = metal_promotion_repeat_runs,
        .correctness_passed = true,
        .generated_route_checked = true,
        .provider_route_checked = true,
        .benchmark_passed = true,
        .measured_speedup = 4.319982,
        .minimum_repeat_speedup = 3.881579,
        .production_enabled = true,
        .promotion_ready = true,
    },
    .{
        .kernel_id = first_general_metal_q4_bias_kernel_id,
        .source_path = first_general_metal_q4_bias_source_path,
        .artifact_source_path = first_general_metal_q4_bias_artifact_source_path,
        .source_fingerprint = sourceFingerprint(first_general_metal_q4_bias_source),
        .check_command = first_general_metal_q4_bias_check_command,
        .runtime_evidence_command = first_general_metal_q4_bias_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q4_bias_promotion_check_command,
        .repeat_runs = metal_promotion_repeat_runs,
        .correctness_passed = true,
        .generated_route_checked = true,
        .provider_route_checked = true,
        .benchmark_passed = true,
        .measured_speedup = 1.798550,
        .minimum_repeat_speedup = 1.566395,
        .production_enabled = true,
        .promotion_ready = true,
    },
    .{
        .kernel_id = first_general_metal_q8_kernel_id,
        .source_path = first_general_metal_q8_source_path,
        .artifact_source_path = first_general_metal_q8_artifact_source_path,
        .source_fingerprint = sourceFingerprint(first_general_metal_q8_source),
        .check_command = first_general_metal_q8_check_command,
        .runtime_evidence_command = first_general_metal_q8_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q8_promotion_check_command,
        .repeat_runs = metal_promotion_repeat_runs,
        .correctness_passed = true,
        .generated_route_checked = true,
        .provider_route_checked = true,
        .benchmark_passed = true,
        .measured_speedup = 3.408203,
        .minimum_repeat_speedup = 2.411878,
        .production_enabled = true,
        .promotion_ready = true,
    },
    .{
        .kernel_id = first_general_metal_q8_bias_kernel_id,
        .source_path = first_general_metal_q8_bias_source_path,
        .artifact_source_path = first_general_metal_q8_bias_artifact_source_path,
        .source_fingerprint = sourceFingerprint(first_general_metal_q8_bias_source),
        .check_command = first_general_metal_q8_bias_check_command,
        .runtime_evidence_command = first_general_metal_q8_bias_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q8_bias_promotion_check_command,
        .repeat_runs = metal_promotion_repeat_runs,
        .correctness_passed = true,
        .generated_route_checked = true,
        .provider_route_checked = true,
        .benchmark_passed = true,
        .measured_speedup = 1.914274,
        .minimum_repeat_speedup = 1.776806,
        .production_enabled = true,
        .promotion_ready = true,
    },
};
pub const first_metal_runtime_evidence_count = first_metal_runtime_evidence.len;
// A cleared blocker refresh is only a signal to investigate; promotion still
// requires checked-in runtime evidence plus a passing production-regression run.
pub const first_metal_promotion_blocker_evidence = [_]MetalPromotionBlockerEvidence{
    .{ .kernel_id = first_general_metal_q4_0_kernel_id, .blocker = "speedup_gate_missing", .evidence_path = first_general_metal_q4_0_promotion_evidence_path, .requires_production_regression_clear = true },
    .{ .kernel_id = first_general_metal_q4_1_kernel_id, .blocker = "unstable_benchmark_timing", .evidence_path = first_general_metal_q4_1_promotion_evidence_path, .requires_production_regression_clear = true },
    .{ .kernel_id = first_general_metal_q5_0_kernel_id, .blocker = "speedup_gate_missing", .evidence_path = first_general_metal_q5_0_promotion_evidence_path, .requires_production_regression_clear = true },
    .{ .kernel_id = first_general_metal_q5_1_kernel_id, .blocker = "speedup_gate_missing", .evidence_path = first_general_metal_q5_1_promotion_evidence_path, .requires_production_regression_clear = true },
    .{ .kernel_id = first_lazy_metal_kernel_id, .blocker = "unstable_benchmark_timing", .evidence_path = first_lazy_metal_promotion_evidence_path, .requires_production_regression_clear = true },
    .{ .kernel_id = first_general_metal_q4_kernel_id, .blocker = "speedup_gate_missing", .evidence_path = first_general_metal_q4_promotion_evidence_path, .requires_production_regression_clear = true },
    .{ .kernel_id = first_general_metal_q5_kernel_id, .blocker = "unstable_benchmark_timing", .evidence_path = first_general_metal_q5_promotion_evidence_path, .requires_production_regression_clear = true },
    .{ .kernel_id = first_general_metal_q5_bias_gelu_kernel_id, .blocker = "unstable_benchmark_timing", .evidence_path = first_general_metal_q5_bias_gelu_promotion_evidence_path, .requires_production_regression_clear = true },
    .{ .kernel_id = first_general_metal_q8_bias_gelu_kernel_id, .blocker = "speedup_gate_missing", .evidence_path = first_general_metal_q8_bias_gelu_promotion_evidence_path, .requires_production_regression_clear = true },
    .{ .kernel_id = first_general_metal_q8_k_kernel_id, .blocker = "unstable_benchmark_timing", .evidence_path = first_general_metal_q8_k_promotion_evidence_path, .requires_production_regression_clear = true },
    .{ .kernel_id = first_general_metal_q6_bias_gelu_kernel_id, .blocker = "speedup_gate_missing", .evidence_path = first_general_metal_q6_bias_gelu_promotion_evidence_path, .requires_production_regression_clear = true },
    .{ .kernel_id = first_general_metal_q8_1_kernel_id, .blocker = "unstable_benchmark_timing", .evidence_path = first_general_metal_q8_1_promotion_evidence_path, .requires_production_regression_clear = true },
    .{ .kernel_id = first_general_metal_q2_bias_kernel_id, .blocker = "unsupported_handwritten_baseline" },
    .{ .kernel_id = first_general_metal_q2_bias_gelu_kernel_id, .blocker = "unsupported_handwritten_baseline" },
    .{ .kernel_id = first_general_metal_q3_bias_kernel_id, .blocker = "unsupported_handwritten_baseline" },
    .{ .kernel_id = first_general_metal_q3_bias_gelu_kernel_id, .blocker = "unsupported_handwritten_baseline" },
    .{ .kernel_id = first_general_metal_q8_relu_kernel_id, .blocker = "unsupported_handwritten_baseline" },
};
pub const first_metal_promotion_blocker_evidence_count = first_metal_promotion_blocker_evidence.len;
pub const first_metal_promotion_blocker_evidence_cases_per_kernel: usize = 2;
pub const first_metal_promotion_blocker_evidence_expected_case_count = metalPromotionBlockerEvidenceExpectedCaseCount();
pub const first_metal_promotion_blocker_evidence_expected_route_ready_count = first_metal_promotion_blocker_evidence_expected_case_count;
pub const first_metal_production_benchmark_cases = buildMetalProductionBenchmarkCases();
pub const first_metal_production_benchmark_case_count = first_metal_production_benchmark_cases.len;
pub const first_artifact_manifest_schema = "antfly.quant_kernel_artifacts.v2";
pub const first_benchmark_manifest_schema = "antfly.quant_kernel_benchmarks.v4";
pub const first_lazy_benchmark_check_command = "zig-out/bin/antfly-inference bench-cuda --quant-compiler-check-evidence " ++ first_lazy_benchmark_evidence_path ++ " --quant-compiler-require-promotion-ready";
pub const first_spec_manifest_path = "src/ops/cuda/generated/quant_kernel_specs.json";
pub const first_artifact_manifest_path = "src/ops/cuda/generated/quant_kernel_artifacts.json";
pub const first_benchmark_manifest_path = "src/ops/cuda/generated/quant_kernel_benchmarks.json";
pub const first_conformance_manifest_path = "src/ops/cuda/generated/quant_kernel_conformance.json";

pub const first_lazy_metal_kernel_id = "antfly_q4_k_small_batch_bias_gelu_msl_v1";
pub const first_lazy_metal_source_path = "src/ops/metal/generated/quant_kernel_q4_k_small_batch_bias_gelu.metal";
pub const first_lazy_metal_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q4_k_small_batch_bias_gelu.metal";
pub const first_lazy_metal_air_path = "/tmp/antfly_q4_k_small_batch_bias_gelu_msl_v1.air";
pub const first_lazy_metal_artifact_air_path = "/tmp/antfly_q4_k_small_batch_bias_gelu_msl_v1_artifact.air";
pub const first_lazy_metal_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q4_k_small_batch_bias_gelu.metal -o /tmp/antfly_q4_k_small_batch_bias_gelu_msl_v1.air";
pub const first_lazy_metal_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q4_k_small_batch_bias_gelu.metal -o /tmp/antfly_q4_k_small_batch_bias_gelu_msl_v1_artifact.air";
pub const first_general_metal_q4_0_kernel_id = "antfly_q4_0_small_batch_msl_v1";
pub const first_general_metal_q4_0_source_path = "src/ops/metal/generated/quant_kernel_q4_0_small_batch.metal";
pub const first_general_metal_q4_0_air_path = "/tmp/antfly_q4_0_small_batch_msl_v1.air";
pub const first_general_metal_q4_0_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q4_0_small_batch.metal -o /tmp/antfly_q4_0_small_batch_msl_v1.air";
pub const first_general_metal_q4_1_kernel_id = "antfly_q4_1_small_batch_msl_v1";
pub const first_general_metal_q4_1_source_path = "src/ops/metal/generated/quant_kernel_q4_1_small_batch.metal";
pub const first_general_metal_q4_1_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q4_1_small_batch.metal";
pub const first_general_metal_q4_1_air_path = "/tmp/antfly_q4_1_small_batch_msl_v1.air";
pub const first_general_metal_q4_1_artifact_air_path = "/tmp/antfly_q4_1_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q4_1_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q4_1_small_batch.metal -o /tmp/antfly_q4_1_small_batch_msl_v1.air";
pub const first_general_metal_q4_1_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q4_1_small_batch.metal -o /tmp/antfly_q4_1_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q5_0_kernel_id = "antfly_q5_0_small_batch_msl_v1";
pub const first_general_metal_q5_0_source_path = "src/ops/metal/generated/quant_kernel_q5_0_small_batch.metal";
pub const first_general_metal_q5_0_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q5_0_small_batch.metal";
pub const first_general_metal_q5_0_air_path = "/tmp/antfly_q5_0_small_batch_msl_v1.air";
pub const first_general_metal_q5_0_artifact_air_path = "/tmp/antfly_q5_0_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q5_0_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q5_0_small_batch.metal -o /tmp/antfly_q5_0_small_batch_msl_v1.air";
pub const first_general_metal_q5_0_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q5_0_small_batch.metal -o /tmp/antfly_q5_0_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q5_1_kernel_id = "antfly_q5_1_small_batch_msl_v1";
pub const first_general_metal_q5_1_source_path = "src/ops/metal/generated/quant_kernel_q5_1_small_batch.metal";
pub const first_general_metal_q5_1_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q5_1_small_batch.metal";
pub const first_general_metal_q5_1_air_path = "/tmp/antfly_q5_1_small_batch_msl_v1.air";
pub const first_general_metal_q5_1_artifact_air_path = "/tmp/antfly_q5_1_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q5_1_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q5_1_small_batch.metal -o /tmp/antfly_q5_1_small_batch_msl_v1.air";
pub const first_general_metal_q5_1_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q5_1_small_batch.metal -o /tmp/antfly_q5_1_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q2_kernel_id = "antfly_q2_k_small_batch_msl_v1";
pub const first_general_metal_q2_source_path = "src/ops/metal/generated/quant_kernel_q2_k_small_batch.metal";
pub const first_general_metal_q2_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q2_k_small_batch.metal";
pub const first_general_metal_q2_air_path = "/tmp/antfly_q2_k_small_batch_msl_v1.air";
pub const first_general_metal_q2_artifact_air_path = "/tmp/antfly_q2_k_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q2_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q2_k_small_batch.metal -o /tmp/antfly_q2_k_small_batch_msl_v1.air";
pub const first_general_metal_q2_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q2_k_small_batch.metal -o /tmp/antfly_q2_k_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q2_bias_kernel_id = "antfly_q2_k_small_batch_bias_msl_v1";
pub const first_general_metal_q2_bias_source_path = "src/ops/metal/generated/quant_kernel_q2_k_small_batch_bias.metal";
pub const first_general_metal_q2_bias_air_path = "/tmp/antfly_q2_k_small_batch_bias_msl_v1.air";
pub const first_general_metal_q2_bias_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q2_k_small_batch_bias.metal -o /tmp/antfly_q2_k_small_batch_bias_msl_v1.air";
pub const first_general_metal_q2_bias_gelu_kernel_id = "antfly_q2_k_small_batch_bias_gelu_msl_v1";
pub const first_general_metal_q2_bias_gelu_source_path = "src/ops/metal/generated/quant_kernel_q2_k_small_batch_bias_gelu.metal";
pub const first_general_metal_q2_bias_gelu_air_path = "/tmp/antfly_q2_k_small_batch_bias_gelu_msl_v1.air";
pub const first_general_metal_q2_bias_gelu_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q2_k_small_batch_bias_gelu.metal -o /tmp/antfly_q2_k_small_batch_bias_gelu_msl_v1.air";
pub const first_general_metal_q3_kernel_id = "antfly_q3_k_small_batch_msl_v1";
pub const first_general_metal_q3_source_path = "src/ops/metal/generated/quant_kernel_q3_k_small_batch.metal";
pub const first_general_metal_q3_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q3_k_small_batch.metal";
pub const first_general_metal_q3_air_path = "/tmp/antfly_q3_k_small_batch_msl_v1.air";
pub const first_general_metal_q3_artifact_air_path = "/tmp/antfly_q3_k_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q3_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q3_k_small_batch.metal -o /tmp/antfly_q3_k_small_batch_msl_v1.air";
pub const first_general_metal_q3_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q3_k_small_batch.metal -o /tmp/antfly_q3_k_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q3_bias_kernel_id = "antfly_q3_k_small_batch_bias_msl_v1";
pub const first_general_metal_q3_bias_source_path = "src/ops/metal/generated/quant_kernel_q3_k_small_batch_bias.metal";
pub const first_general_metal_q3_bias_air_path = "/tmp/antfly_q3_k_small_batch_bias_msl_v1.air";
pub const first_general_metal_q3_bias_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q3_k_small_batch_bias.metal -o /tmp/antfly_q3_k_small_batch_bias_msl_v1.air";
pub const first_general_metal_q3_bias_gelu_kernel_id = "antfly_q3_k_small_batch_bias_gelu_msl_v1";
pub const first_general_metal_q3_bias_gelu_source_path = "src/ops/metal/generated/quant_kernel_q3_k_small_batch_bias_gelu.metal";
pub const first_general_metal_q3_bias_gelu_air_path = "/tmp/antfly_q3_k_small_batch_bias_gelu_msl_v1.air";
pub const first_general_metal_q3_bias_gelu_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q3_k_small_batch_bias_gelu.metal -o /tmp/antfly_q3_k_small_batch_bias_gelu_msl_v1.air";
pub const first_general_metal_q4_bias_kernel_id = "antfly_q4_k_small_batch_bias_msl_v1";
pub const first_general_metal_q4_bias_source_path = "src/ops/metal/generated/quant_kernel_q4_k_small_batch_bias.metal";
pub const first_general_metal_q4_bias_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q4_k_small_batch_bias.metal";
pub const first_general_metal_q4_bias_air_path = "/tmp/antfly_q4_k_small_batch_bias_msl_v1.air";
pub const first_general_metal_q4_bias_artifact_air_path = "/tmp/antfly_q4_k_small_batch_bias_msl_v1_artifact.air";
pub const first_general_metal_q4_bias_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q4_k_small_batch_bias.metal -o /tmp/antfly_q4_k_small_batch_bias_msl_v1.air";
pub const first_general_metal_q4_bias_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q4_k_small_batch_bias.metal -o /tmp/antfly_q4_k_small_batch_bias_msl_v1_artifact.air";
pub const first_general_metal_q4_kernel_id = "antfly_q4_k_small_batch_msl_v1";
pub const first_general_metal_q4_source_path = "src/ops/metal/generated/quant_kernel_q4_k_small_batch.metal";
pub const first_general_metal_q4_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q4_k_small_batch.metal";
pub const first_general_metal_q4_air_path = "/tmp/antfly_q4_k_small_batch_msl_v1.air";
pub const first_general_metal_q4_artifact_air_path = "/tmp/antfly_q4_k_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q4_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q4_k_small_batch.metal -o /tmp/antfly_q4_k_small_batch_msl_v1.air";
pub const first_general_metal_q4_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q4_k_small_batch.metal -o /tmp/antfly_q4_k_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q8_kernel_id = "antfly_q8_0_small_batch_msl_v1";
pub const first_general_metal_q8_source_path = "src/ops/metal/generated/quant_kernel_q8_0_small_batch.metal";
pub const first_general_metal_q8_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q8_0_small_batch.metal";
pub const first_general_metal_q8_air_path = "/tmp/antfly_q8_0_small_batch_msl_v1.air";
pub const first_general_metal_q8_artifact_air_path = "/tmp/antfly_q8_0_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q8_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q8_0_small_batch.metal -o /tmp/antfly_q8_0_small_batch_msl_v1.air";
pub const first_general_metal_q8_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q8_0_small_batch.metal -o /tmp/antfly_q8_0_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q8_bias_kernel_id = "antfly_q8_0_small_batch_bias_msl_v1";
pub const first_general_metal_q8_bias_source_path = "src/ops/metal/generated/quant_kernel_q8_0_small_batch_bias.metal";
pub const first_general_metal_q8_bias_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q8_0_small_batch_bias.metal";
pub const first_general_metal_q8_bias_air_path = "/tmp/antfly_q8_0_small_batch_bias_msl_v1.air";
pub const first_general_metal_q8_bias_artifact_air_path = "/tmp/antfly_q8_0_small_batch_bias_msl_v1_artifact.air";
pub const first_general_metal_q8_bias_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q8_0_small_batch_bias.metal -o /tmp/antfly_q8_0_small_batch_bias_msl_v1.air";
pub const first_general_metal_q8_bias_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q8_0_small_batch_bias.metal -o /tmp/antfly_q8_0_small_batch_bias_msl_v1_artifact.air";
pub const first_general_metal_q8_bias_gelu_kernel_id = "antfly_q8_0_small_batch_bias_gelu_msl_v1";
pub const first_general_metal_q8_bias_gelu_source_path = "src/ops/metal/generated/quant_kernel_q8_0_small_batch_bias_gelu.metal";
pub const first_general_metal_q8_bias_gelu_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q8_0_small_batch_bias_gelu.metal";
pub const first_general_metal_q8_bias_gelu_air_path = "/tmp/antfly_q8_0_small_batch_bias_gelu_msl_v1.air";
pub const first_general_metal_q8_bias_gelu_artifact_air_path = "/tmp/antfly_q8_0_small_batch_bias_gelu_msl_v1_artifact.air";
pub const first_general_metal_q8_bias_gelu_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q8_0_small_batch_bias_gelu.metal -o /tmp/antfly_q8_0_small_batch_bias_gelu_msl_v1.air";
pub const first_general_metal_q8_bias_gelu_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q8_0_small_batch_bias_gelu.metal -o /tmp/antfly_q8_0_small_batch_bias_gelu_msl_v1_artifact.air";
pub const first_general_metal_q8_bias_gelu_source_fingerprint = sourceFingerprint(first_general_metal_q8_bias_gelu_source);
pub const first_general_metal_q8_relu_kernel_id = "antfly_q8_0_small_batch_relu_msl_v1";
pub const first_general_metal_q8_relu_source_path = "src/ops/metal/generated/quant_kernel_q8_0_small_batch_relu.metal";
pub const first_general_metal_q8_relu_air_path = "/tmp/antfly_q8_0_small_batch_relu_msl_v1.air";
pub const first_general_metal_q8_relu_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q8_0_small_batch_relu.metal -o /tmp/antfly_q8_0_small_batch_relu_msl_v1.air";
pub const first_general_metal_q8_1_kernel_id = "antfly_q8_1_small_batch_msl_v1";
pub const first_general_metal_q8_1_source_path = "src/ops/metal/generated/quant_kernel_q8_1_small_batch.metal";
pub const first_general_metal_q8_1_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q8_1_small_batch.metal";
pub const first_general_metal_q8_1_air_path = "/tmp/antfly_q8_1_small_batch_msl_v1.air";
pub const first_general_metal_q8_1_artifact_air_path = "/tmp/antfly_q8_1_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q8_1_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q8_1_small_batch.metal -o /tmp/antfly_q8_1_small_batch_msl_v1.air";
pub const first_general_metal_q8_1_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q8_1_small_batch.metal -o /tmp/antfly_q8_1_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q8_k_kernel_id = "antfly_q8_k_small_batch_msl_v1";
pub const first_general_metal_q8_k_source_path = "src/ops/metal/generated/quant_kernel_q8_k_small_batch.metal";
pub const first_general_metal_q8_k_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q8_k_small_batch.metal";
pub const first_general_metal_q8_k_air_path = "/tmp/antfly_q8_k_small_batch_msl_v1.air";
pub const first_general_metal_q8_k_artifact_air_path = "/tmp/antfly_q8_k_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q8_k_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q8_k_small_batch.metal -o /tmp/antfly_q8_k_small_batch_msl_v1.air";
pub const first_general_metal_q8_k_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q8_k_small_batch.metal -o /tmp/antfly_q8_k_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q5_kernel_id = "antfly_q5_k_small_batch_msl_v1";
pub const first_general_metal_q5_source_path = "src/ops/metal/generated/quant_kernel_q5_k_small_batch.metal";
pub const first_general_metal_q5_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q5_k_small_batch.metal";
pub const first_general_metal_q5_air_path = "/tmp/antfly_q5_k_small_batch_msl_v1.air";
pub const first_general_metal_q5_artifact_air_path = "/tmp/antfly_q5_k_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q5_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q5_k_small_batch.metal -o /tmp/antfly_q5_k_small_batch_msl_v1.air";
pub const first_general_metal_q5_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q5_k_small_batch.metal -o /tmp/antfly_q5_k_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q5_bias_kernel_id = "antfly_q5_k_small_batch_bias_msl_v1";
pub const first_general_metal_q5_bias_source_path = "src/ops/metal/generated/quant_kernel_q5_k_small_batch_bias.metal";
pub const first_general_metal_q5_bias_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q5_k_small_batch_bias.metal";
pub const first_general_metal_q5_bias_air_path = "/tmp/antfly_q5_k_small_batch_bias_msl_v1.air";
pub const first_general_metal_q5_bias_artifact_air_path = "/tmp/antfly_q5_k_small_batch_bias_msl_v1_artifact.air";
pub const first_general_metal_q5_bias_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q5_k_small_batch_bias.metal -o /tmp/antfly_q5_k_small_batch_bias_msl_v1.air";
pub const first_general_metal_q5_bias_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q5_k_small_batch_bias.metal -o /tmp/antfly_q5_k_small_batch_bias_msl_v1_artifact.air";
pub const first_general_metal_q5_bias_gelu_kernel_id = "antfly_q5_k_small_batch_bias_gelu_msl_v1";
pub const first_general_metal_q5_bias_gelu_source_path = "src/ops/metal/generated/quant_kernel_q5_k_small_batch_bias_gelu.metal";
pub const first_general_metal_q5_bias_gelu_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q5_k_small_batch_bias_gelu.metal";
pub const first_general_metal_q5_bias_gelu_air_path = "/tmp/antfly_q5_k_small_batch_bias_gelu_msl_v1.air";
pub const first_general_metal_q5_bias_gelu_artifact_air_path = "/tmp/antfly_q5_k_small_batch_bias_gelu_msl_v1_artifact.air";
pub const first_general_metal_q5_bias_gelu_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q5_k_small_batch_bias_gelu.metal -o /tmp/antfly_q5_k_small_batch_bias_gelu_msl_v1.air";
pub const first_general_metal_q5_bias_gelu_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q5_k_small_batch_bias_gelu.metal -o /tmp/antfly_q5_k_small_batch_bias_gelu_msl_v1_artifact.air";
pub const first_general_metal_q6_kernel_id = "antfly_q6_k_small_batch_msl_v1";
pub const first_general_metal_q6_source_path = "src/ops/metal/generated/quant_kernel_q6_k_small_batch.metal";
pub const first_general_metal_q6_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q6_k_small_batch.metal";
pub const first_general_metal_q6_air_path = "/tmp/antfly_q6_k_small_batch_msl_v1.air";
pub const first_general_metal_q6_artifact_air_path = "/tmp/antfly_q6_k_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q6_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q6_k_small_batch.metal -o /tmp/antfly_q6_k_small_batch_msl_v1.air";
pub const first_general_metal_q6_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q6_k_small_batch.metal -o /tmp/antfly_q6_k_small_batch_msl_v1_artifact.air";
pub const first_general_metal_q6_bias_kernel_id = "antfly_q6_k_small_batch_bias_msl_v1";
pub const first_general_metal_q6_bias_source_path = "src/ops/metal/generated/quant_kernel_q6_k_small_batch_bias.metal";
pub const first_general_metal_q6_bias_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q6_k_small_batch_bias.metal";
pub const first_general_metal_q6_bias_air_path = "/tmp/antfly_q6_k_small_batch_bias_msl_v1.air";
pub const first_general_metal_q6_bias_artifact_air_path = "/tmp/antfly_q6_k_small_batch_bias_msl_v1_artifact.air";
pub const first_general_metal_q6_bias_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q6_k_small_batch_bias.metal -o /tmp/antfly_q6_k_small_batch_bias_msl_v1.air";
pub const first_general_metal_q6_bias_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q6_k_small_batch_bias.metal -o /tmp/antfly_q6_k_small_batch_bias_msl_v1_artifact.air";
pub const first_general_metal_q6_bias_gelu_kernel_id = "antfly_q6_k_small_batch_bias_gelu_msl_v1";
pub const first_general_metal_q6_bias_gelu_source_path = "src/ops/metal/generated/quant_kernel_q6_k_small_batch_bias_gelu.metal";
pub const first_general_metal_q6_bias_gelu_artifact_source_path = "src/ops/metal/artifacts/quant_kernel_q6_k_small_batch_bias_gelu.metal";
pub const first_general_metal_q6_bias_gelu_air_path = "/tmp/antfly_q6_k_small_batch_bias_gelu_msl_v1.air";
pub const first_general_metal_q6_bias_gelu_artifact_air_path = "/tmp/antfly_q6_k_small_batch_bias_gelu_msl_v1_artifact.air";
pub const first_general_metal_q6_bias_gelu_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/generated/quant_kernel_q6_k_small_batch_bias_gelu.metal -o /tmp/antfly_q6_k_small_batch_bias_gelu_msl_v1.air";
pub const first_general_metal_q6_bias_gelu_artifact_check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q6_k_small_batch_bias_gelu.metal -o /tmp/antfly_q6_k_small_batch_bias_gelu_msl_v1_artifact.air";
pub const first_metal_runtime_evidence_path = "/private/tmp/antfly-quant-metal-evidence.json";
pub const first_metal_runtime_evidence_command = "zig build quant-kernel-metal-runtime-check -Dmetal=true -Dcuda=false -- --evidence-out " ++ first_metal_runtime_evidence_path ++ " --repeat-runs " ++ metal_promotion_repeat_runs_text;
pub const first_metal_local_check_command = "zig build quant-kernel-metal-local-check -Dmetal=true -Dcuda=false";
pub const first_metal_model_local_check_command = "zig build quant-kernel-metal-model-local-check -Dmetal=true -Dcuda=false";
pub const first_metal_model_generated_route_check_command = "zig build test-metal-gemma4-prefill-frame-e4b-generated-q8-q4-0 -Dmetal=true -Dcuda=false";
pub const first_metal_model_generated_q8_0_small_batch_min: usize = 1;
pub const first_metal_model_generated_q4_0_small_batch_min: usize = 1;
pub const first_metal_industry_local_check_command = "zig build quant-kernel-metal-industry-local-check -Dmetal=true -Dcuda=false";
pub const first_metal_runtime_route_all_evidence_path = "/private/tmp/antfly-quant-metal-runtime-route-all-evidence.json";
pub const first_metal_runtime_route_all_build_command = "zig build quant-kernel-metal-runtime-route-all -Dmetal=true -Dcuda=false";
pub const first_metal_runtime_route_all_evidence_command = "zig build quant-kernel-metal-runtime-check -Dmetal=true -Dcuda=false -- --evidence-out " ++ first_metal_runtime_route_all_evidence_path ++ " --runtime-route-all";
pub const first_metal_runtime_route_all_check_command = "zig build quant-kernel-metal-runtime-check -Dmetal=true -Dcuda=false -- --check-evidence " ++ first_metal_runtime_route_all_evidence_path ++ " --require-runtime-route-all";
pub const first_metal_runtime_route_all_expected_case_count = metalRuntimeRouteAllExpectedCaseCount();
pub const first_metal_runtime_route_all_expected_route_ready_count = first_metal_runtime_route_all_expected_case_count;
pub const first_metal_runtime_route_all_expected_provider_route_count = metalRuntimeRouteAllExpectedProviderRouteCount();
pub const first_metal_production_regression_evidence_path = "/private/tmp/antfly-quant-metal-production-regression-evidence.json";
pub const first_metal_production_regression_build_command = "zig build quant-kernel-metal-production-regression-check -Dmetal=true -Dcuda=false";
pub const first_metal_production_regression_evidence_command = "zig build quant-kernel-metal-runtime-check -Dmetal=true -Dcuda=false -- --evidence-out " ++ first_metal_production_regression_evidence_path ++ " --repeat-runs " ++ metal_promotion_repeat_runs_text ++ " --measure-iters " ++ metal_promotion_measure_iters_text ++ " --production-regression-check";
pub const first_metal_production_regression_expected_kernel_count = metalProductionRegressionExpectedKernelCount();
pub const first_metal_production_regression_expected_case_count = metalProductionRegressionExpectedCaseCount();
pub const first_metal_production_regression_expected_route_ready_count = first_metal_production_regression_expected_case_count;
pub const first_metal_production_regression_route_ready_is_hard_gate = true;
pub const first_metal_production_regression_missing_provider_route_is_hard_gate = true;
pub const first_metal_production_regression_speedup_gate_missing_is_hard_gate = true;
pub const first_metal_production_regression_unstable_benchmark_timing_is_hard_gate = true;
pub const first_metal_unsupported_handwritten_baseline_blocks_promotion = true;
pub const first_metal_unsupported_handwritten_baseline_uses_runtime_route_all_evidence = true;
pub const first_metal_unsupported_handwritten_baseline_has_promotion_evidence_path = false;
pub const first_metal_blocker_strict_check_command = "zig build quant-kernel-metal-blocker-strict-check -Dmetal=true -Dcuda=false";
pub const first_metal_promotion_evidence_command = "zig build quant-kernel-metal-runtime-check -Dmetal=true -Dcuda=false -- --evidence-out ";
pub const first_metal_promotion_check_command = "zig build quant-kernel-metal-runtime-check -Dmetal=true -Dcuda=false -- --check-evidence ";
pub const first_lazy_metal_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_lazy_metal_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q4_0_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q4_0_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q4_1_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q4_1_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q5_0_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q5_0_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q5_1_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q5_1_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q2_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q2_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q2_bias_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q2_bias_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q2_bias_gelu_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q2_bias_gelu_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q3_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q3_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q3_bias_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q3_bias_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q3_bias_gelu_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q3_bias_gelu_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q4_bias_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q4_bias_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q4_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q4_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q8_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q8_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q8_bias_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q8_bias_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q8_bias_gelu_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q8_bias_gelu_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q8_relu_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q8_relu_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q8_1_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q8_1_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q8_k_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q8_k_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q5_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q5_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q5_bias_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q5_bias_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q5_bias_gelu_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q5_bias_gelu_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q6_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q6_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q6_bias_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q6_bias_kernel_id ++ "-promotion-evidence.json";
pub const first_general_metal_q6_bias_gelu_promotion_evidence_path = "/private/tmp/antfly-quant-metal-" ++ first_general_metal_q6_bias_gelu_kernel_id ++ "-promotion-evidence.json";
pub const first_lazy_metal_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_lazy_metal_promotion_evidence_path ++ metal_promotion_args ++ first_lazy_metal_kernel_id;
pub const first_general_metal_q4_0_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q4_0_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q4_0_kernel_id;
pub const first_general_metal_q4_1_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q4_1_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q4_1_kernel_id;
pub const first_general_metal_q5_0_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q5_0_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q5_0_kernel_id;
pub const first_general_metal_q5_1_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q5_1_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q5_1_kernel_id;
pub const first_general_metal_q2_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q2_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q2_kernel_id;
pub const first_general_metal_q2_bias_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q2_bias_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q2_bias_kernel_id;
pub const first_general_metal_q2_bias_gelu_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q2_bias_gelu_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q2_bias_gelu_kernel_id;
pub const first_general_metal_q3_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q3_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q3_kernel_id;
pub const first_general_metal_q3_bias_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q3_bias_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q3_bias_kernel_id;
pub const first_general_metal_q3_bias_gelu_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q3_bias_gelu_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q3_bias_gelu_kernel_id;
pub const first_general_metal_q4_bias_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q4_bias_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q4_bias_kernel_id;
pub const first_general_metal_q4_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q4_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q4_kernel_id;
pub const first_general_metal_q8_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q8_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q8_kernel_id;
pub const first_general_metal_q8_bias_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q8_bias_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q8_bias_kernel_id;
pub const first_general_metal_q8_bias_gelu_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q8_bias_gelu_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q8_bias_gelu_kernel_id;
pub const first_general_metal_q8_relu_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q8_relu_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q8_relu_kernel_id;
pub const first_general_metal_q8_1_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q8_1_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q8_1_kernel_id;
pub const first_general_metal_q8_k_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q8_k_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q8_k_kernel_id;
pub const first_general_metal_q5_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q5_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q5_kernel_id;
pub const first_general_metal_q5_bias_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q5_bias_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q5_bias_kernel_id;
pub const first_general_metal_q5_bias_gelu_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q5_bias_gelu_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q5_bias_gelu_kernel_id;
pub const first_general_metal_q6_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q6_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q6_kernel_id;
pub const first_general_metal_q6_bias_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q6_bias_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q6_bias_kernel_id;
pub const first_general_metal_q6_bias_gelu_promotion_evidence_command = first_metal_promotion_evidence_command ++ first_general_metal_q6_bias_gelu_promotion_evidence_path ++ metal_promotion_args ++ first_general_metal_q6_bias_gelu_kernel_id;
pub const first_lazy_metal_promotion_check_command = first_metal_promotion_check_command ++ first_lazy_metal_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_lazy_metal_kernel_id;
pub const first_general_metal_q4_0_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q4_0_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q4_0_kernel_id;
pub const first_general_metal_q4_1_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q4_1_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q4_1_kernel_id;
pub const first_general_metal_q5_0_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q5_0_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q5_0_kernel_id;
pub const first_general_metal_q5_1_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q5_1_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q5_1_kernel_id;
pub const first_general_metal_q2_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q2_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q2_kernel_id;
pub const first_general_metal_q2_bias_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q2_bias_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q2_bias_kernel_id;
pub const first_general_metal_q2_bias_gelu_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q2_bias_gelu_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q2_bias_gelu_kernel_id;
pub const first_general_metal_q3_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q3_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q3_kernel_id;
pub const first_general_metal_q3_bias_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q3_bias_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q3_bias_kernel_id;
pub const first_general_metal_q3_bias_gelu_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q3_bias_gelu_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q3_bias_gelu_kernel_id;
pub const first_general_metal_q4_bias_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q4_bias_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q4_bias_kernel_id;
pub const first_general_metal_q4_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q4_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q4_kernel_id;
pub const first_general_metal_q8_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q8_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q8_kernel_id;
pub const first_general_metal_q8_bias_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q8_bias_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q8_bias_kernel_id;
pub const first_general_metal_q8_bias_gelu_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q8_bias_gelu_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q8_bias_gelu_kernel_id;
pub const first_general_metal_q8_relu_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q8_relu_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q8_relu_kernel_id;
pub const first_general_metal_q8_1_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q8_1_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q8_1_kernel_id;
pub const first_general_metal_q8_k_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q8_k_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q8_k_kernel_id;
pub const first_general_metal_q5_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q5_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q5_kernel_id;
pub const first_general_metal_q5_bias_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q5_bias_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q5_bias_kernel_id;
pub const first_general_metal_q5_bias_gelu_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q5_bias_gelu_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q5_bias_gelu_kernel_id;
pub const first_general_metal_q6_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q6_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q6_kernel_id;
pub const first_general_metal_q6_bias_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q6_bias_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q6_bias_kernel_id;
pub const first_general_metal_q6_bias_gelu_promotion_check_command = first_metal_promotion_check_command ++ first_general_metal_q6_bias_gelu_promotion_evidence_path ++ " --require-promotion-ready --require-kernel " ++ first_general_metal_q6_bias_gelu_kernel_id;

pub const first_generated_artifacts = [_]GeneratedArtifact{
    .{
        .backend = .cuda,
        .format = first_lazy_benchmark.format,
        .row_bucket = first_lazy_benchmark.row_bucket,
        .epilogue = first_lazy_benchmark.epilogue,
        .kernel_id = first_lazy_benchmark.generated_kernel_id,
        .source_path = first_lazy_benchmark.generated_source_path,
        .check_command = first_lazy_benchmark.generated_ptx_command,
        .runtime_evidence_command = first_lazy_benchmark.benchmark_command,
        .promotion_check_command = first_lazy_benchmark_check_command,
        .production_enabled = first_lazy_benchmark.production_enabled,
    },
    .{
        .backend = .metal,
        .format = first_lazy_benchmark.format,
        .row_bucket = first_lazy_benchmark.row_bucket,
        .epilogue = first_lazy_benchmark.epilogue,
        .kernel_id = first_lazy_metal_kernel_id,
        .source_path = first_lazy_metal_artifact_source_path,
        .check_command = first_lazy_metal_artifact_check_command,
        .generated_source_path = first_lazy_metal_source_path,
        .generated_check_command = first_lazy_metal_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_lazy_metal_promotion_evidence_command,
        .promotion_check_command = first_lazy_metal_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q4_0,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .kernel_id = first_general_metal_q4_0_kernel_id,
        .source_path = first_general_metal_q4_0_source_path,
        .check_command = first_general_metal_q4_0_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q4_0_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q4_0_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q4_1,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .kernel_id = first_general_metal_q4_1_kernel_id,
        .source_path = first_general_metal_q4_1_artifact_source_path,
        .check_command = first_general_metal_q4_1_artifact_check_command,
        .generated_source_path = first_general_metal_q4_1_source_path,
        .generated_check_command = first_general_metal_q4_1_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q4_1_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q4_1_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q5_0,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .kernel_id = first_general_metal_q5_0_kernel_id,
        .source_path = first_general_metal_q5_0_artifact_source_path,
        .check_command = first_general_metal_q5_0_artifact_check_command,
        .generated_source_path = first_general_metal_q5_0_source_path,
        .generated_check_command = first_general_metal_q5_0_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q5_0_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q5_0_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q5_1,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .kernel_id = first_general_metal_q5_1_kernel_id,
        .source_path = first_general_metal_q5_1_artifact_source_path,
        .check_command = first_general_metal_q5_1_artifact_check_command,
        .generated_source_path = first_general_metal_q5_1_source_path,
        .generated_check_command = first_general_metal_q5_1_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q5_1_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q5_1_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q2_k,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .kernel_id = first_general_metal_q2_kernel_id,
        .source_path = first_general_metal_q2_artifact_source_path,
        .check_command = first_general_metal_q2_artifact_check_command,
        .generated_source_path = first_general_metal_q2_source_path,
        .generated_check_command = first_general_metal_q2_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q2_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q2_promotion_check_command,
        .production_enabled = true,
    },
    .{
        .backend = .metal,
        .format = .q2_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias,
        .kernel_id = first_general_metal_q2_bias_kernel_id,
        .source_path = first_general_metal_q2_bias_source_path,
        .check_command = first_general_metal_q2_bias_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q2_bias_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q2_bias_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q2_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .kernel_id = first_general_metal_q2_bias_gelu_kernel_id,
        .source_path = first_general_metal_q2_bias_gelu_source_path,
        .check_command = first_general_metal_q2_bias_gelu_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q2_bias_gelu_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q2_bias_gelu_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q3_k,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .kernel_id = first_general_metal_q3_kernel_id,
        .source_path = first_general_metal_q3_artifact_source_path,
        .check_command = first_general_metal_q3_artifact_check_command,
        .generated_source_path = first_general_metal_q3_source_path,
        .generated_check_command = first_general_metal_q3_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q3_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q3_promotion_check_command,
        .production_enabled = true,
    },
    .{
        .backend = .metal,
        .format = .q3_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias,
        .kernel_id = first_general_metal_q3_bias_kernel_id,
        .source_path = first_general_metal_q3_bias_source_path,
        .check_command = first_general_metal_q3_bias_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q3_bias_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q3_bias_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q3_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .kernel_id = first_general_metal_q3_bias_gelu_kernel_id,
        .source_path = first_general_metal_q3_bias_gelu_source_path,
        .check_command = first_general_metal_q3_bias_gelu_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q3_bias_gelu_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q3_bias_gelu_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q4_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias,
        .kernel_id = first_general_metal_q4_bias_kernel_id,
        .source_path = first_general_metal_q4_bias_artifact_source_path,
        .check_command = first_general_metal_q4_bias_artifact_check_command,
        .generated_source_path = first_general_metal_q4_bias_source_path,
        .generated_check_command = first_general_metal_q4_bias_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q4_bias_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q4_bias_promotion_check_command,
        .production_enabled = true,
    },
    .{
        .backend = .metal,
        .format = .q4_k,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .kernel_id = first_general_metal_q4_kernel_id,
        .source_path = first_general_metal_q4_artifact_source_path,
        .check_command = first_general_metal_q4_artifact_check_command,
        .generated_source_path = first_general_metal_q4_source_path,
        .generated_check_command = first_general_metal_q4_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q4_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q4_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q8_0,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .kernel_id = first_general_metal_q8_kernel_id,
        .source_path = first_general_metal_q8_artifact_source_path,
        .check_command = first_general_metal_q8_artifact_check_command,
        .generated_source_path = first_general_metal_q8_source_path,
        .generated_check_command = first_general_metal_q8_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q8_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q8_promotion_check_command,
        .production_enabled = true,
    },
    .{
        .backend = .metal,
        .format = .q8_0,
        .row_bucket = .rows_2_8,
        .epilogue = .bias,
        .kernel_id = first_general_metal_q8_bias_kernel_id,
        .source_path = first_general_metal_q8_bias_artifact_source_path,
        .check_command = first_general_metal_q8_bias_artifact_check_command,
        .generated_source_path = first_general_metal_q8_bias_source_path,
        .generated_check_command = first_general_metal_q8_bias_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q8_bias_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q8_bias_promotion_check_command,
        .production_enabled = true,
    },
    .{
        .backend = .metal,
        .format = .q8_0,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .kernel_id = first_general_metal_q8_bias_gelu_kernel_id,
        .source_path = first_general_metal_q8_bias_gelu_artifact_source_path,
        .check_command = first_general_metal_q8_bias_gelu_artifact_check_command,
        .generated_source_path = first_general_metal_q8_bias_gelu_source_path,
        .generated_check_command = first_general_metal_q8_bias_gelu_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q8_bias_gelu_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q8_bias_gelu_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q8_0,
        .row_bucket = .rows_2_8,
        .epilogue = .relu,
        .kernel_id = first_general_metal_q8_relu_kernel_id,
        .source_path = first_general_metal_q8_relu_source_path,
        .check_command = first_general_metal_q8_relu_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q8_relu_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q8_relu_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q8_1,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .kernel_id = first_general_metal_q8_1_kernel_id,
        .source_path = first_general_metal_q8_1_artifact_source_path,
        .check_command = first_general_metal_q8_1_artifact_check_command,
        .generated_source_path = first_general_metal_q8_1_source_path,
        .generated_check_command = first_general_metal_q8_1_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q8_1_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q8_1_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q8_k,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .kernel_id = first_general_metal_q8_k_kernel_id,
        .source_path = first_general_metal_q8_k_artifact_source_path,
        .check_command = first_general_metal_q8_k_artifact_check_command,
        .generated_source_path = first_general_metal_q8_k_source_path,
        .generated_check_command = first_general_metal_q8_k_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q8_k_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q8_k_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q5_k,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .kernel_id = first_general_metal_q5_kernel_id,
        .source_path = first_general_metal_q5_artifact_source_path,
        .check_command = first_general_metal_q5_artifact_check_command,
        .generated_source_path = first_general_metal_q5_source_path,
        .generated_check_command = first_general_metal_q5_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q5_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q5_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q5_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias,
        .kernel_id = first_general_metal_q5_bias_kernel_id,
        .source_path = first_general_metal_q5_bias_artifact_source_path,
        .check_command = first_general_metal_q5_bias_artifact_check_command,
        .generated_source_path = first_general_metal_q5_bias_source_path,
        .generated_check_command = first_general_metal_q5_bias_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q5_bias_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q5_bias_promotion_check_command,
        .production_enabled = true,
    },
    .{
        .backend = .metal,
        .format = .q5_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .kernel_id = first_general_metal_q5_bias_gelu_kernel_id,
        .source_path = first_general_metal_q5_bias_gelu_artifact_source_path,
        .check_command = first_general_metal_q5_bias_gelu_artifact_check_command,
        .generated_source_path = first_general_metal_q5_bias_gelu_source_path,
        .generated_check_command = first_general_metal_q5_bias_gelu_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q5_bias_gelu_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q5_bias_gelu_promotion_check_command,
        .production_enabled = false,
    },
    .{
        .backend = .metal,
        .format = .q6_k,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .kernel_id = first_general_metal_q6_kernel_id,
        .source_path = first_general_metal_q6_artifact_source_path,
        .check_command = first_general_metal_q6_artifact_check_command,
        .generated_source_path = first_general_metal_q6_source_path,
        .generated_check_command = first_general_metal_q6_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q6_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q6_promotion_check_command,
        .production_enabled = true,
    },
    .{
        .backend = .metal,
        .format = .q6_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias,
        .kernel_id = first_general_metal_q6_bias_kernel_id,
        .source_path = first_general_metal_q6_bias_artifact_source_path,
        .check_command = first_general_metal_q6_bias_artifact_check_command,
        .generated_source_path = first_general_metal_q6_bias_source_path,
        .generated_check_command = first_general_metal_q6_bias_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q6_bias_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q6_bias_promotion_check_command,
        .production_enabled = true,
    },
    .{
        .backend = .metal,
        .format = .q6_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .kernel_id = first_general_metal_q6_bias_gelu_kernel_id,
        .source_path = first_general_metal_q6_bias_gelu_artifact_source_path,
        .check_command = first_general_metal_q6_bias_gelu_artifact_check_command,
        .generated_source_path = first_general_metal_q6_bias_gelu_source_path,
        .generated_check_command = first_general_metal_q6_bias_gelu_check_command,
        .runtime_evidence_command = first_metal_runtime_evidence_command,
        .promotion_evidence_command = first_general_metal_q6_bias_gelu_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q6_bias_gelu_promotion_check_command,
        .production_enabled = false,
    },
};

const first_route_expectations = [_]RouteExpectation{
    .{
        .backend = .cuda,
        .format = .q4_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .dispatch = .small_batch,
        .production_route = .handwritten_production,
        .candidate_route = .generated_dev_candidate,
        .fallback_reason = .generated_artifact_missing,
    },
    .{
        .backend = .metal,
        .format = .q4_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .dispatch = .small_batch,
        .production_route = .handwritten_production,
        .candidate_route = .generated_dev_candidate,
        .fallback_reason = .generated_artifact_missing,
    },
    .{
        .backend = .cuda,
        .format = .q8_0,
        .row_bucket = .rows_1,
        .epilogue = .none,
        .dispatch = .mmv,
        .production_route = .handwritten_production,
        .candidate_route = .unsupported,
        .fallback_reason = .none,
    },
    .{
        .backend = .cuda,
        .format = .q5_k,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .dispatch = .small_batch,
        .production_route = .unsupported,
        .candidate_route = .unsupported,
        .fallback_reason = .unsupported_backend,
    },
    .{
        .backend = .cuda,
        .format = .q5_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .dispatch = .small_batch,
        .production_route = .unsupported,
        .candidate_route = .unsupported,
        .fallback_reason = .unsupported_backend,
    },
    .{
        .backend = .metal,
        .format = .q1_0,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .dispatch = .small_batch,
        .production_route = .handwritten_production,
        .candidate_route = .unsupported,
        .fallback_reason = .none,
    },
    .{
        .backend = .metal,
        .format = .q2_k,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .dispatch = .small_batch,
        .production_route = .generated_production,
        .candidate_route = .unsupported,
        .fallback_reason = .none,
    },
    .{
        .backend = .metal,
        .format = .q2_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias,
        .dispatch = .small_batch,
        .production_route = .handwritten_production,
        .candidate_route = .generated_dev_candidate,
        .fallback_reason = .generated_artifact_missing,
    },
    .{
        .backend = .metal,
        .format = .q2_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .dispatch = .small_batch,
        .production_route = .handwritten_production,
        .candidate_route = .generated_dev_candidate,
        .fallback_reason = .generated_artifact_missing,
    },
    .{
        .backend = .metal,
        .format = .q3_k,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .dispatch = .small_batch,
        .production_route = .generated_production,
        .candidate_route = .unsupported,
        .fallback_reason = .none,
    },
    .{
        .backend = .metal,
        .format = .q3_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias,
        .dispatch = .small_batch,
        .production_route = .handwritten_production,
        .candidate_route = .generated_dev_candidate,
        .fallback_reason = .generated_artifact_missing,
    },
    .{
        .backend = .metal,
        .format = .q3_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .dispatch = .small_batch,
        .production_route = .handwritten_production,
        .candidate_route = .generated_dev_candidate,
        .fallback_reason = .generated_artifact_missing,
    },
    .{
        .backend = .metal,
        .format = .q5_0,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .dispatch = .small_batch,
        .production_route = .handwritten_production,
        .candidate_route = .generated_dev_candidate,
        .fallback_reason = .generated_artifact_missing,
    },
    .{
        .backend = .metal,
        .format = .q5_1,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .dispatch = .small_batch,
        .production_route = .handwritten_production,
        .candidate_route = .generated_dev_candidate,
        .fallback_reason = .generated_artifact_missing,
    },
    .{
        .backend = .metal,
        .format = .q8_k,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .dispatch = .small_batch,
        .production_route = .handwritten_production,
        .candidate_route = .generated_dev_candidate,
        .fallback_reason = .generated_artifact_missing,
    },
    .{
        .backend = .metal,
        .format = .q8_1,
        .row_bucket = .rows_2_8,
        .epilogue = .none,
        .dispatch = .small_batch,
        .production_route = .handwritten_production,
        .candidate_route = .generated_dev_candidate,
        .fallback_reason = .generated_artifact_missing,
    },
    .{
        .backend = .metal,
        .format = .q8_0,
        .row_bucket = .rows_2_8,
        .epilogue = .bias,
        .dispatch = .small_batch,
        .production_route = .generated_production,
        .candidate_route = .unsupported,
        .fallback_reason = .none,
    },
    .{
        .backend = .metal,
        .format = .q8_0,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .dispatch = .small_batch,
        .production_route = .handwritten_production,
        .candidate_route = .generated_dev_candidate,
        .fallback_reason = .generated_artifact_missing,
    },
    .{
        .backend = .metal,
        .format = .q5_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias,
        .dispatch = .small_batch,
        .production_route = .generated_production,
        .candidate_route = .unsupported,
        .fallback_reason = .none,
    },
    .{
        .backend = .metal,
        .format = .q5_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .dispatch = .small_batch,
        .production_route = .handwritten_production,
        .candidate_route = .generated_dev_candidate,
        .fallback_reason = .generated_artifact_missing,
    },
    .{
        .backend = .metal,
        .format = .q6_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias,
        .dispatch = .small_batch,
        .production_route = .generated_production,
        .candidate_route = .unsupported,
        .fallback_reason = .none,
    },
    .{
        .backend = .metal,
        .format = .q6_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .dispatch = .small_batch,
        .production_route = .handwritten_production,
        .candidate_route = .generated_dev_candidate,
        .fallback_reason = .generated_artifact_missing,
    },
    .{
        .backend = .cuda,
        .format = .q6_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .dispatch = .small_batch,
        .production_route = .unsupported,
        .candidate_route = .unsupported,
        .fallback_reason = .unsupported_epilogue,
    },
    .{
        .backend = .cuda,
        .format = .unknown,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .dispatch = .small_batch,
        .production_route = .unsupported,
        .candidate_route = .unsupported,
        .fallback_reason = .unsupported_format,
    },
    .{
        .backend = .cuda,
        .format = .q4_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
        .dispatch = .mm,
        .production_route = .unsupported,
        .candidate_route = .unsupported,
        .fallback_reason = .unsupported_shape,
    },
};

const first_lazy_cuda_source =
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
    \\
    \\// Dev-only generated kernel candidate from graph/quant_kernel_compiler.zig.
    \\// plan_id=cuda/q4_k/rows_2_8/bias_gelu/small_batch
    \\// kernel_id=antfly_q4_k_small_batch_bias_gelu_f32_v1
    \\// production_baseline=termite_linear_q4_k_bias_gelu_f32_tile4_r2
    \\// production_enabled=false
    \\// Not compiled into production artifacts until correctness and benchmark gates
    \\// beat the handwritten CUDA baseline.
    \\
    \\#include <cuda_fp16.h>
    \\#include <math.h>
    \\#include <stdint.h>
    \\
    \\struct antfly_q4_k_block_view {
    \\    const uint8_t *d;
    \\    const uint8_t *dmin;
    \\    const uint8_t *scales;
    \\    const uint8_t *qs;
    \\};
    \\
    \\static __device__ __forceinline__ float antfly_half_le_to_float(const uint8_t *p) {
    \\    const uint16_t bits = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
    \\    return __half2float(__ushort_as_half(bits));
    \\}
    \\
    \\static __device__ __forceinline__ float antfly_gelu(float x) {
    \\    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    \\    return 0.5f * x * (1.0f + tanhf(inner));
    \\}
    \\
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
    \\
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
    \\
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
    \\
;

const first_lazy_metal_source =
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
    \\
    \\// Generated Metal candidate artifact from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q4_k/rows_2_8/bias_gelu/small_batch
    \\// kernel_id=antfly_q4_k_small_batch_bias_gelu_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, provider-route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_gelu(float x) {
    \\    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    \\    return 0.5f * x * (1.0f + fast::tanh(inner));
    \\}
    \\
    \\static inline void antfly_q4_k_unpack_scale_min(
    \\    const device uchar *scales,
    \\    int sub,
    \\    thread float &scale,
    \\    thread float &min_v
    \\) {
    \\    if (sub < 4) {
    \\        scale = (float)(scales[sub] & 63u);
    \\        min_v = (float)(scales[sub + 4] & 63u);
    \\        return;
    \\    }
    \\    scale = (float)((scales[sub + 4] & 0x0fu) | ((scales[sub - 4] >> 6) << 4));
    \\    min_v = (float)((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
    \\}
    \\
    \\static inline float antfly_q4_k_dequant_lane(const device uchar *block, int lane) {
    \\    const device uchar *d = block;
    \\    const device uchar *dmin = block + 2;
    \\    const device uchar *scales = block + 4;
    \\    const device uchar *qs = block + 16;
    \\    const int sub = lane >> 5;
    \\    const int q_index = (sub >> 1) * 32 + (lane & 31);
    \\    const uchar packed = qs[q_index];
    \\    const uchar q = (sub & 1) == 0 ? (packed & 0x0fu) : (packed >> 4);
    \\    float raw_scale = 0.0f;
    \\    float raw_min = 0.0f;
    \\    antfly_q4_k_unpack_scale_min(scales, sub, raw_scale, raw_min);
    \\    return antfly_half_le_to_float(d) * raw_scale * (float)q - antfly_half_le_to_float(dmin) * raw_min;
    \\}
    \\
    \\kernel void antfly_q4_k_small_batch_bias_gelu_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q4_k [[buffer(1)]],
    \\    const device float *bias [[buffer(2)]],
    \\    device float *output [[buffer(3)]],
    \\    constant int &rows [[buffer(4)]],
    \\    constant int &in_dim [[buffer(5)]],
    \\    constant int &out_dim [[buffer(6)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 64) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q4_k + ((col * block_count + block_idx) * 144);
    \\            const int base = block_idx << 8;
    \\            for (int lane = (int)tid; lane < 256; lane += 64) {
    \\                acc += input[row * in_dim + base + lane] * antfly_q4_k_dequant_lane(block, lane);
    \\            }
    \\        }
    \\    }
    \\
    \\    threadgroup float partial[64];
    \\    if (tid < 64) partial[tid] = acc;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = 32; stride > 0; stride >>= 1) {
    \\        if (tid < stride) partial[tid] += partial[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (tid == 0) output[row * out_dim + col] = antfly_gelu(partial[0] + bias[col]);
    \\}
    \\
;

const first_general_metal_q4_0_source =
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
    \\
    \\// Generated Metal candidate artifact from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q4_0/rows_2_8/none/small_batch
    \\// kernel_id=antfly_q4_0_small_batch_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=false
    \\// General MSL lowering smoke for descriptor-driven quant matmul.
    \\// Production Metal dispatch stays on native handwritten MSL until this
    \\// candidate clears correctness and benchmark gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_q4_0_dequant_lane(const device uchar *block, int lane) {
    \\    const float d = antfly_half_le_to_float(block);
    \\    const int packed_index = lane & 15;
    \\    const uchar packed = block[2 + packed_index];
    \\    const int q = lane < 16 ? (int)(packed & 0x0fu) - 8 : (int)(packed >> 4) - 8;
    \\    return d * (float)q;
    \\}
    \\
    \\kernel void antfly_q4_0_small_batch_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q4_0 [[buffer(1)]],
    \\    device float *output [[buffer(2)]],
    \\    constant int &rows [[buffer(3)]],
    \\    constant int &in_dim [[buffer(4)]],
    \\    constant int &out_dim [[buffer(5)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 31) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 5;
    \\    if (tid < 32) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q4_0 + ((col * block_count + block_idx) * 18);
    \\            const int lane = (int)tid;
    \\            acc += input[row * in_dim + (block_idx << 5) + lane] * antfly_q4_0_dequant_lane(block, lane);
    \\        }
    \\    }
    \\
    \\    acc = simd_sum(acc);
    \\    if (tid == 0) output[row * out_dim + col] = acc;
    \\}
    \\
;

const first_general_metal_q4_1_source =
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
    \\
    \\// Generated Metal artifact source from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q4_1/rows_2_8/none/small_batch
    \\// kernel_id=antfly_q4_1_small_batch_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=false
    \\// General MSL lowering smoke for descriptor-driven quant matmul.
    \\// Production Metal dispatch stays on native handwritten MSL until this
    \\// candidate clears correctness and benchmark gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_q4_1_dequant_lane(const device uchar *block, int lane) {
    \\    const float d = antfly_half_le_to_float(block);
    \\    const float m = antfly_half_le_to_float(block + 2);
    \\    const int packed_index = lane & 15;
    \\    const uchar packed = block[4 + packed_index];
    \\    const int q = lane < 16 ? (int)(packed & 0x0fu) : (int)(packed >> 4);
    \\    return d * (float)q + m;
    \\}
    \\
    \\kernel void antfly_q4_1_small_batch_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q4_1 [[buffer(1)]],
    \\    device float *output [[buffer(2)]],
    \\    constant int &rows [[buffer(3)]],
    \\    constant int &in_dim [[buffer(4)]],
    \\    constant int &out_dim [[buffer(5)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col0 = (int)(group_pos.x << 1);
    \\    const int col1 = col0 + 1;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col0 >= out_dim || (in_dim & 31) != 0) return;
    \\
    \\    float acc0 = 0.0f;
    \\    float acc1 = 0.0f;
    \\    const int block_count = in_dim >> 5;
    \\    const int lane = (int)tid;
    \\    const device float *row_input = input + row * in_dim;
    \\    const device uchar *col0_weight = weight_q4_1 + col0 * block_count * 20;
    \\    const bool has_col1 = col1 < out_dim;
    \\    const device uchar *col1_weight = has_col1 ? weight_q4_1 + col1 * block_count * 20 : col0_weight;
    \\    for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\        const float x = row_input[(block_idx << 5) + lane];
    \\        const device uchar *block0 = col0_weight + block_idx * 20;
    \\        acc0 += x * antfly_q4_1_dequant_lane(block0, lane);
    \\        if (has_col1) {
    \\            const device uchar *block1 = col1_weight + block_idx * 20;
    \\            acc1 += x * antfly_q4_1_dequant_lane(block1, lane);
    \\        }
    \\    }
    \\
    \\    acc0 = simd_sum(acc0);
    \\    acc1 = simd_sum(acc1);
    \\    if (tid == 0) {
    \\        output[row * out_dim + col0] = acc0;
    \\        if (has_col1) output[row * out_dim + col1] = acc1;
    \\    }
    \\}
    \\
;

const first_general_metal_q5_0_source =
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
    \\
    \\// Generated Metal artifact source from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q5_0/rows_2_8/none/small_batch
    \\// kernel_id=antfly_q5_0_small_batch_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=false
    \\// Dev-only generated Metal candidate from graph/quant_kernel_compiler.zig.
    \\// Route metadata is checked in, but production dispatch stays on the
    \\// handwritten Metal path until this candidate clears correctness and
    \\// benchmark gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline uint antfly_u32_le(const device uchar *p) {
    \\    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
    \\}
    \\
    \\static inline float antfly_q5_0_dequant_lane(const device uchar *block, int lane) {
    \\    const float d = antfly_half_le_to_float(block);
    \\    const uint qh = antfly_u32_le(block + 2);
    \\    const int packed_index = lane & 15;
    \\    const uchar packed = block[6 + packed_index];
    \\    const int low4 = lane < 16 ? (int)(packed & 0x0fu) : (int)(packed >> 4);
    \\    const int high = (int)((qh >> (uint)lane) & 1u);
    \\    return d * (float)((low4 | (high << 4)) - 16);
    \\}
    \\
    \\kernel void antfly_q5_0_small_batch_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q5_0 [[buffer(1)]],
    \\    device float *output [[buffer(2)]],
    \\    constant int &rows [[buffer(3)]],
    \\    constant int &in_dim [[buffer(4)]],
    \\    constant int &out_dim [[buffer(5)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 31) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 5;
    \\    if (tid < 32) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q5_0 + ((col * block_count + block_idx) * 22);
    \\            const int lane = (int)tid;
    \\            acc += input[row * in_dim + (block_idx << 5) + lane] * antfly_q5_0_dequant_lane(block, lane);
    \\        }
    \\    }
    \\
    \\    acc = simd_sum(acc);
    \\    if (tid == 0) output[row * out_dim + col] = acc;
    \\}
    \\
;

const first_general_metal_q5_1_source =
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
    \\
    \\// Generated Metal artifact source from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q5_1/rows_2_8/none/small_batch
    \\// kernel_id=antfly_q5_1_small_batch_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=false
    \\// Dev-only generated Metal candidate from graph/quant_kernel_compiler.zig.
    \\// Route metadata is checked in, but production dispatch stays on the
    \\// handwritten Metal path until this candidate clears correctness and
    \\// benchmark gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline uint antfly_u32_le(const device uchar *p) {
    \\    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
    \\}
    \\
    \\static inline float antfly_q5_1_dequant_lane(const device uchar *block, int lane) {
    \\    const float d = antfly_half_le_to_float(block);
    \\    const float m = antfly_half_le_to_float(block + 2);
    \\    const uint qh = antfly_u32_le(block + 4);
    \\    const int packed_index = lane & 15;
    \\    const uchar packed = block[8 + packed_index];
    \\    const int low4 = lane < 16 ? (int)(packed & 0x0fu) : (int)(packed >> 4);
    \\    const int high = (int)((qh >> (uint)lane) & 1u);
    \\    return d * (float)(low4 | (high << 4)) + m;
    \\}
    \\
    \\kernel void antfly_q5_1_small_batch_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q5_1 [[buffer(1)]],
    \\    device float *output [[buffer(2)]],
    \\    constant int &rows [[buffer(3)]],
    \\    constant int &in_dim [[buffer(4)]],
    \\    constant int &out_dim [[buffer(5)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col0 = (int)(group_pos.x << 1);
    \\    const int col1 = col0 + 1;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col0 >= out_dim || (in_dim & 31) != 0) return;
    \\
    \\    float acc0 = 0.0f;
    \\    float acc1 = 0.0f;
    \\    const int block_count = in_dim >> 5;
    \\    const int lane = (int)tid;
    \\    const device float *row_input = input + row * in_dim;
    \\    const device uchar *col0_weight = weight_q5_1 + col0 * block_count * 24;
    \\    const bool has_col1 = col1 < out_dim;
    \\    const device uchar *col1_weight = has_col1 ? weight_q5_1 + col1 * block_count * 24 : col0_weight;
    \\    for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\        const float x = row_input[(block_idx << 5) + lane];
    \\        const device uchar *block0 = col0_weight + block_idx * 24;
    \\        acc0 += x * antfly_q5_1_dequant_lane(block0, lane);
    \\        if (has_col1) {
    \\            const device uchar *block1 = col1_weight + block_idx * 24;
    \\            acc1 += x * antfly_q5_1_dequant_lane(block1, lane);
    \\        }
    \\    }
    \\
    \\    acc0 = simd_sum(acc0);
    \\    acc1 = simd_sum(acc1);
    \\    if (tid == 0) {
    \\        output[row * out_dim + col0] = acc0;
    \\        if (has_col1) output[row * out_dim + col1] = acc1;
    \\    }
    \\}
    \\
;

const first_general_metal_q4_source =
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
    \\
    \\// Generated Metal candidate artifact from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q4_k/rows_2_8/none/small_batch
    \\// kernel_id=antfly_q4_k_small_batch_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, provider-route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline void antfly_q4_k_unpack_scale_min(
    \\    const device uchar *scales,
    \\    int sub,
    \\    thread float &scale,
    \\    thread float &min_v
    \\) {
    \\    if (sub < 4) {
    \\        scale = (float)(scales[sub] & 63u);
    \\        min_v = (float)(scales[sub + 4] & 63u);
    \\        return;
    \\    }
    \\    scale = (float)((scales[sub + 4] & 0x0fu) | ((scales[sub - 4] >> 6) << 4));
    \\    min_v = (float)((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
    \\}
    \\
    \\static inline float antfly_q4_k_dequant_lane(const device uchar *block, int lane) {
    \\    const device uchar *d = block;
    \\    const device uchar *dmin = block + 2;
    \\    const device uchar *scales = block + 4;
    \\    const device uchar *qs = block + 16;
    \\    const int sub = lane >> 5;
    \\    const int q_index = (sub >> 1) * 32 + (lane & 31);
    \\    const uchar packed = qs[q_index];
    \\    const uchar q = (sub & 1) == 0 ? (packed & 0x0fu) : (packed >> 4);
    \\    float raw_scale = 0.0f;
    \\    float raw_min = 0.0f;
    \\    antfly_q4_k_unpack_scale_min(scales, sub, raw_scale, raw_min);
    \\    return antfly_half_le_to_float(d) * raw_scale * (float)q - antfly_half_le_to_float(dmin) * raw_min;
    \\}
    \\
    \\kernel void antfly_q4_k_small_batch_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q4_k [[buffer(1)]],
    \\    device float *output [[buffer(2)]],
    \\    constant int &rows [[buffer(3)]],
    \\    constant int &in_dim [[buffer(4)]],
    \\    constant int &out_dim [[buffer(5)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    threadgroup float partial[64];
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 64) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q4_k + ((col * block_count + block_idx) * 144);
    \\            const int base = block_idx << 8;
    \\            for (int lane = (int)tid; lane < 256; lane += 64) {
    \\                acc += input[row * in_dim + base + lane] * antfly_q4_k_dequant_lane(block, lane);
    \\            }
    \\        }
    \\    }
    \\
    \\    if (tid < 64) partial[tid] = acc;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = 32; stride > 0; stride >>= 1) {
    \\        if (tid < stride) partial[tid] += partial[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (tid == 0) output[row * out_dim + col] = partial[0];
    \\}
    \\
;

const first_general_metal_q4_bias_source =
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
    \\
    \\// Generated Metal candidate artifact from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q4_k/rows_2_8/bias/small_batch
    \\// kernel_id=antfly_q4_k_small_batch_bias_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, provider-route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline void antfly_q4_k_unpack_scale_min(
    \\    const device uchar *scales,
    \\    int sub,
    \\    thread float &scale,
    \\    thread float &min_v
    \\) {
    \\    if (sub < 4) {
    \\        scale = (float)(scales[sub] & 63u);
    \\        min_v = (float)(scales[sub + 4] & 63u);
    \\        return;
    \\    }
    \\    scale = (float)((scales[sub + 4] & 0x0fu) | ((scales[sub - 4] >> 6) << 4));
    \\    min_v = (float)((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
    \\}
    \\
    \\static inline float antfly_q4_k_dequant_lane(const device uchar *block, int lane) {
    \\    const device uchar *d = block;
    \\    const device uchar *dmin = block + 2;
    \\    const device uchar *scales = block + 4;
    \\    const device uchar *qs = block + 16;
    \\    const int sub = lane >> 5;
    \\    const int q_index = (sub >> 1) * 32 + (lane & 31);
    \\    const uchar packed = qs[q_index];
    \\    const uchar q = (sub & 1) == 0 ? (packed & 0x0fu) : (packed >> 4);
    \\    float raw_scale = 0.0f;
    \\    float raw_min = 0.0f;
    \\    antfly_q4_k_unpack_scale_min(scales, sub, raw_scale, raw_min);
    \\    return antfly_half_le_to_float(d) * raw_scale * (float)q - antfly_half_le_to_float(dmin) * raw_min;
    \\}
    \\
    \\kernel void antfly_q4_k_small_batch_bias_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q4_k [[buffer(1)]],
    \\    const device float *bias [[buffer(2)]],
    \\    device float *output [[buffer(3)]],
    \\    constant int &rows [[buffer(4)]],
    \\    constant int &in_dim [[buffer(5)]],
    \\    constant int &out_dim [[buffer(6)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]],
    \\    ushort lane_id [[thread_index_in_simdgroup]],
    \\    ushort simdgroup_id [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 64) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q4_k + ((col * block_count + block_idx) * 144);
    \\            const int base = block_idx << 8;
    \\            for (int lane = (int)tid; lane < 256; lane += 64) {
    \\                acc += input[row * in_dim + base + lane] * antfly_q4_k_dequant_lane(block, lane);
    \\            }
    \\        }
    \\    }
    \\
    \\    threadgroup float partial[64];
    \\    if (tid < 64) partial[tid] = acc;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = 32; stride > 0; stride >>= 1) {
    \\        if (tid < stride) partial[tid] += partial[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (tid == 0) output[row * out_dim + col] = partial[0] + bias[col];
    \\}
    \\
;

const first_general_metal_q8_source =
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
    \\
    \\// Generated Metal artifact source from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q8_0/rows_2_8/none/small_batch
    \\// kernel_id=antfly_q8_0_small_batch_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, provider-route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_q8_0_dequant_lane(const device uchar *block, int lane) {
    \\    const float d = antfly_half_le_to_float(block);
    \\    const int q = (int)as_type<char>(block[2 + lane]);
    \\    return d * (float)q;
    \\}
    \\
    \\kernel void antfly_q8_0_small_batch_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q8_0 [[buffer(1)]],
    \\    device float *output [[buffer(2)]],
    \\    constant int &rows [[buffer(3)]],
    \\    constant int &in_dim [[buffer(4)]],
    \\    constant int &out_dim [[buffer(5)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 31) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 5;
    \\    const int lane = (int)tid;
    \\    const device float *row_input = input + row * in_dim;
    \\    const device uchar *col_weight = weight_q8_0 + col * block_count * 34;
    \\    for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\        const device uchar *block = col_weight + block_idx * 34;
    \\        acc += row_input[(block_idx << 5) + lane] * antfly_q8_0_dequant_lane(block, lane);
    \\    }
    \\
    \\    acc = simd_sum(acc);
    \\    if (tid == 0) output[row * out_dim + col] = acc;
    \\}
    \\
;

const first_general_metal_q8_bias_source =
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
    \\
    \\// Generated Metal artifact source from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q8_0/rows_2_8/bias/small_batch
    \\// kernel_id=antfly_q8_0_small_batch_bias_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, provider-route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_q8_0_dequant_lane(const device uchar *block, int lane) {
    \\    const float d = antfly_half_le_to_float(block);
    \\    const int q = (int)as_type<char>(block[2 + lane]);
    \\    return d * (float)q;
    \\}
    \\
    \\kernel void antfly_q8_0_small_batch_bias_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q8_0 [[buffer(1)]],
    \\    const device float *bias [[buffer(2)]],
    \\    device float *output [[buffer(3)]],
    \\    constant int &rows [[buffer(4)]],
    \\    constant int &in_dim [[buffer(5)]],
    \\    constant int &out_dim [[buffer(6)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]],
    \\    ushort lane_id [[thread_index_in_simdgroup]],
    \\    ushort simdgroup_id [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 31) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 5;
    \\    const int lane = (int)tid;
    \\    const device float *row_input = input + row * in_dim;
    \\    const device uchar *col_weight = weight_q8_0 + col * block_count * 34;
    \\    for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\        const device uchar *block = col_weight + block_idx * 34;
    \\        acc += row_input[(block_idx << 5) + lane] * antfly_q8_0_dequant_lane(block, lane);
    \\    }
    \\
    \\    acc = simd_sum(acc);
    \\    if (tid == 0) output[row * out_dim + col] = acc + bias[col];
    \\}
    \\
;

const first_general_metal_q8_bias_gelu_source =
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
    \\
    \\// Generated Metal artifact source from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q8_0/rows_2_8/bias_gelu/small_batch
    \\// kernel_id=antfly_q8_0_small_batch_bias_gelu_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=false
    \\// General MSL lowering smoke for descriptor-driven quant matmul epilogues.
    \\// Production Metal dispatch stays on native handwritten MSL until this
    \\// candidate clears correctness and benchmark gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_q8_0_dequant_lane(const device uchar *block, int lane) {
    \\    const float d = antfly_half_le_to_float(block);
    \\    const int q = (int)as_type<char>(block[2 + lane]);
    \\    return d * (float)q;
    \\}
    \\
    \\static inline float antfly_gelu(float x) {
    \\    return 0.5f * x * (1.0f + fast::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    \\}
    \\
    \\kernel void antfly_q8_0_small_batch_bias_gelu_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q8_0 [[buffer(1)]],
    \\    const device float *bias [[buffer(2)]],
    \\    device float *output [[buffer(3)]],
    \\    constant int &rows [[buffer(4)]],
    \\    constant int &in_dim [[buffer(5)]],
    \\    constant int &out_dim [[buffer(6)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 31) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 5;
    \\    const int lane = (int)tid;
    \\    const device float *row_input = input + row * in_dim;
    \\    const device uchar *col_weight = weight_q8_0 + col * block_count * 34;
    \\    for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\        const float x = row_input[(block_idx << 5) + lane];
    \\        const device uchar *block = col_weight + block_idx * 34;
    \\        acc += x * antfly_q8_0_dequant_lane(block, lane);
    \\    }
    \\
    \\    acc = simd_sum(acc);
    \\    if (tid == 0) {
    \\        output[row * out_dim + col] = antfly_gelu(acc + bias[col]);
    \\    }
    \\}
    \\
;

const first_general_metal_q8_relu_source =
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
    \\
    \\// Dev-only generated Metal candidate from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q8_0/rows_2_8/relu/small_batch
    \\// kernel_id=antfly_q8_0_small_batch_relu_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=false
    \\// General MSL lowering smoke for descriptor-driven quant matmul epilogues.
    \\// Production Metal dispatch stays on native handwritten MSL until this
    \\// candidate clears correctness and benchmark gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_q8_0_dequant_lane(const device uchar *block, int lane) {
    \\    const float d = antfly_half_le_to_float(block);
    \\    const int q = (int)as_type<char>(block[2 + lane]);
    \\    return d * (float)q;
    \\}
    \\
    \\kernel void antfly_q8_0_small_batch_relu_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q8_0 [[buffer(1)]],
    \\    device float *output [[buffer(2)]],
    \\    constant int &rows [[buffer(3)]],
    \\    constant int &in_dim [[buffer(4)]],
    \\    constant int &out_dim [[buffer(5)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 31) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 5;
    \\    if (tid < 32) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q8_0 + ((col * block_count + block_idx) * 34);
    \\            const int lane = (int)tid;
    \\            acc += input[row * in_dim + (block_idx << 5) + lane] * antfly_q8_0_dequant_lane(block, lane);
    \\        }
    \\    }
    \\
    \\    acc = simd_sum(acc);
    \\    if (tid == 0) output[row * out_dim + col] = max(acc, 0.0f);
    \\}
    \\
;

const first_general_metal_q2_source =
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
    \\
    \\// Dev-only generated Metal candidate from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q2_k/rows_2_8/none/small_batch
    \\// kernel_id=antfly_q2_k_small_batch_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_q2_k_dequant_lane(const device uchar *block, int lane) {
    \\    const uint sub = (uint)lane >> 4;
    \\    const uint i = (uint)lane & 15u;
    \\    const uchar scale_byte = block[sub];
    \\    const float dsc = antfly_half_le_to_float(block + 16) * (float)(scale_byte & 0x0Fu);
    \\    const float dmn = antfly_half_le_to_float(block + 18) * (float)(scale_byte >> 4);
    \\    const uint chunk = sub >> 3;
    \\    const uint group = (sub & 7u) >> 1;
    \\    const uint l_base = (sub & 1u) << 4;
    \\    const uint q_base = chunk << 5;
    \\    const uint shift = group << 1;
    \\    const uint q = ((uint)block[20 + q_base + l_base + i] >> shift) & 0x03u;
    \\    return dsc * (float)q - dmn;
    \\}
    \\
    \\kernel void antfly_q2_k_small_batch_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q2_k [[buffer(1)]],
    \\    device float *output [[buffer(2)]],
    \\    constant int &rows [[buffer(3)]],
    \\    constant int &in_dim [[buffer(4)]],
    \\    constant int &out_dim [[buffer(5)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 32) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q2_k + ((col * block_count + block_idx) * 84);
    \\            for (int lane = (int)tid; lane < 256; lane += 32) {
    \\                acc += input[row * in_dim + (block_idx << 8) + lane] * antfly_q2_k_dequant_lane(block, lane);
    \\            }
    \\        }
    \\    }
    \\
    \\    acc = simd_sum(acc);
    \\    if (tid == 0) output[row * out_dim + col] = acc;
    \\}
    \\
;

const first_general_metal_q2_bias_source =
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
    \\
    \\// Dev-only generated Metal candidate from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q2_k/rows_2_8/bias/small_batch
    \\// kernel_id=antfly_q2_k_small_batch_bias_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=false
    \\// General MSL lowering smoke for descriptor-driven quant matmul epilogues.
    \\// Production Metal dispatch stays on native handwritten MSL until this
    \\// candidate clears correctness and benchmark gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_q2_k_dequant_lane(const device uchar *block, int lane) {
    \\    const uint sub = (uint)lane >> 4;
    \\    const uint i = (uint)lane & 15u;
    \\    const uchar scale_byte = block[sub];
    \\    const float dsc = antfly_half_le_to_float(block + 16) * (float)(scale_byte & 0x0Fu);
    \\    const float dmn = antfly_half_le_to_float(block + 18) * (float)(scale_byte >> 4);
    \\    const uint chunk = sub >> 3;
    \\    const uint group = (sub & 7u) >> 1;
    \\    const uint l_base = (sub & 1u) << 4;
    \\    const uint q_base = chunk << 5;
    \\    const uint shift = group << 1;
    \\    const uint q = ((uint)block[20 + q_base + l_base + i] >> shift) & 0x03u;
    \\    return dsc * (float)q - dmn;
    \\}
    \\
    \\kernel void antfly_q2_k_small_batch_bias_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q2_k [[buffer(1)]],
    \\    const device float *bias [[buffer(2)]],
    \\    device float *output [[buffer(3)]],
    \\    constant int &rows [[buffer(4)]],
    \\    constant int &in_dim [[buffer(5)]],
    \\    constant int &out_dim [[buffer(6)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 32) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q2_k + ((col * block_count + block_idx) * 84);
    \\            for (int lane = (int)tid; lane < 256; lane += 32) {
    \\                acc += input[row * in_dim + (block_idx << 8) + lane] * antfly_q2_k_dequant_lane(block, lane);
    \\            }
    \\        }
    \\    }
    \\
    \\    acc = simd_sum(acc);
    \\    if (tid == 0) output[row * out_dim + col] = acc + bias[col];
    \\}
    \\
;

const first_general_metal_q2_bias_gelu_source =
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
    \\
    \\// Dev-only generated Metal candidate from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q2_k/rows_2_8/bias_gelu/small_batch
    \\// kernel_id=antfly_q2_k_small_batch_bias_gelu_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=false
    \\// General MSL lowering smoke for descriptor-driven quant matmul epilogues.
    \\// Production Metal dispatch stays on native handwritten MSL until this
    \\// candidate clears correctness and benchmark gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_gelu(float x) {
    \\    return 0.5f * x * (1.0f + fast::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    \\}
    \\
    \\static inline float antfly_q2_k_dequant_lane(const device uchar *block, int lane) {
    \\    const uint sub = (uint)lane >> 4;
    \\    const uint i = (uint)lane & 15u;
    \\    const uchar scale_byte = block[sub];
    \\    const float dsc = antfly_half_le_to_float(block + 16) * (float)(scale_byte & 0x0Fu);
    \\    const float dmn = antfly_half_le_to_float(block + 18) * (float)(scale_byte >> 4);
    \\    const uint chunk = sub >> 3;
    \\    const uint group = (sub & 7u) >> 1;
    \\    const uint l_base = (sub & 1u) << 4;
    \\    const uint q_base = chunk << 5;
    \\    const uint shift = group << 1;
    \\    const uint q = ((uint)block[20 + q_base + l_base + i] >> shift) & 0x03u;
    \\    return dsc * (float)q - dmn;
    \\}
    \\
    \\kernel void antfly_q2_k_small_batch_bias_gelu_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q2_k [[buffer(1)]],
    \\    const device float *bias [[buffer(2)]],
    \\    device float *output [[buffer(3)]],
    \\    constant int &rows [[buffer(4)]],
    \\    constant int &in_dim [[buffer(5)]],
    \\    constant int &out_dim [[buffer(6)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 32) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q2_k + ((col * block_count + block_idx) * 84);
    \\            for (int lane = (int)tid; lane < 256; lane += 32) {
    \\                acc += input[row * in_dim + (block_idx << 8) + lane] * antfly_q2_k_dequant_lane(block, lane);
    \\            }
    \\        }
    \\    }
    \\
    \\    acc = simd_sum(acc);
    \\    if (tid == 0) output[row * out_dim + col] = antfly_gelu(acc + bias[col]);
    \\}
    \\
;

const first_general_metal_q3_source =
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
    \\
    \\// Generated Metal candidate artifact from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q3_k/rows_2_8/none/small_batch
    \\// kernel_id=antfly_q3_k_small_batch_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, provider-route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline int antfly_q3_k_raw_scale(const device uchar *scale_data, uint sub) {
    \\    const uint i = sub & 3u;
    \\    uint low = 0u;
    \\    uint high = 0u;
    \\    if (sub < 4u) {
    \\        low = (uint)(scale_data[i] & 0x0Fu);
    \\        high = (uint)(scale_data[8 + i] & 0x03u);
    \\    } else if (sub < 8u) {
    \\        low = (uint)(scale_data[4 + i] & 0x0Fu);
    \\        high = (uint)((scale_data[8 + i] >> 2) & 0x03u);
    \\    } else if (sub < 12u) {
    \\        low = (uint)((scale_data[i] >> 4) & 0x0Fu);
    \\        high = (uint)((scale_data[8 + i] >> 4) & 0x03u);
    \\    } else {
    \\        low = (uint)((scale_data[4 + i] >> 4) & 0x0Fu);
    \\        high = (uint)((scale_data[8 + i] >> 6) & 0x03u);
    \\    }
    \\    return (int)(low | (high << 4)) - 32;
    \\}
    \\
    \\static inline float antfly_q3_k_dequant_lane(const device uchar *block, int lane) {
    \\    const uint sub = (uint)lane >> 4;
    \\    const uint i = (uint)lane & 15u;
    \\    const uint chunk = sub >> 3;
    \\    const uint group = (sub & 7u) >> 1;
    \\    const uint l_base = (sub & 1u) << 4;
    \\    const uint l = l_base + i;
    \\    const uint q_base = chunk << 5;
    \\    const uint shift = group << 1;
    \\    const uint hm_bit = (chunk << 2) + group;
    \\    const int low2 = (int)(((uint)block[32 + q_base + l] >> shift) & 0x03u);
    \\    const int high1 = (int)(((uint)block[l] >> hm_bit) & 0x01u);
    \\    const int q = low2 + high1 * 4 - 4;
    \\    const float scale = antfly_half_le_to_float(block + 108) * (float)antfly_q3_k_raw_scale(block + 96, sub);
    \\    return scale * (float)q;
    \\}
    \\
    \\kernel void antfly_q3_k_small_batch_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q3_k [[buffer(1)]],
    \\    device float *output [[buffer(2)]],
    \\    constant int &rows [[buffer(3)]],
    \\    constant int &in_dim [[buffer(4)]],
    \\    constant int &out_dim [[buffer(5)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 32) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q3_k + ((col * block_count + block_idx) * 110);
    \\            for (int lane = (int)tid; lane < 256; lane += 32) {
    \\                acc += input[row * in_dim + (block_idx << 8) + lane] * antfly_q3_k_dequant_lane(block, lane);
    \\            }
    \\        }
    \\    }
    \\
    \\    acc = simd_sum(acc);
    \\    if (tid == 0) output[row * out_dim + col] = acc;
    \\}
    \\
;

const first_general_metal_q3_bias_source =
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
    \\
    \\// Generated Metal artifact source from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q3_k/rows_2_8/bias/small_batch
    \\// kernel_id=antfly_q3_k_small_batch_bias_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=false
    \\// General MSL lowering smoke for descriptor-driven quant matmul epilogues.
    \\// Production Metal dispatch stays on native handwritten MSL until this
    \\// candidate clears correctness and benchmark gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline int antfly_q3_k_raw_scale(const device uchar *scale_data, uint sub) {
    \\    const uint i = sub & 3u;
    \\    uint low = 0u;
    \\    uint high = 0u;
    \\    if (sub < 4u) {
    \\        low = (uint)(scale_data[i] & 0x0Fu);
    \\        high = (uint)(scale_data[8 + i] & 0x03u);
    \\    } else if (sub < 8u) {
    \\        low = (uint)(scale_data[4 + i] & 0x0Fu);
    \\        high = (uint)((scale_data[8 + i] >> 2) & 0x03u);
    \\    } else if (sub < 12u) {
    \\        low = (uint)((scale_data[i] >> 4) & 0x0Fu);
    \\        high = (uint)((scale_data[8 + i] >> 4) & 0x03u);
    \\    } else {
    \\        low = (uint)((scale_data[4 + i] >> 4) & 0x0Fu);
    \\        high = (uint)((scale_data[8 + i] >> 6) & 0x03u);
    \\    }
    \\    return (int)(low | (high << 4)) - 32;
    \\}
    \\
    \\static inline float antfly_q3_k_dequant_lane(const device uchar *block, int lane) {
    \\    const uint sub = (uint)lane >> 4;
    \\    const uint i = (uint)lane & 15u;
    \\    const uint chunk = sub >> 3;
    \\    const uint group = (sub & 7u) >> 1;
    \\    const uint l_base = (sub & 1u) << 4;
    \\    const uint l = l_base + i;
    \\    const uint q_base = chunk << 5;
    \\    const uint shift = group << 1;
    \\    const uint hm_bit = (chunk << 2) + group;
    \\    const int low2 = (int)(((uint)block[32 + q_base + l] >> shift) & 0x03u);
    \\    const int high1 = (int)(((uint)block[l] >> hm_bit) & 0x01u);
    \\    const int q = low2 + high1 * 4 - 4;
    \\    const float scale = antfly_half_le_to_float(block + 108) * (float)antfly_q3_k_raw_scale(block + 96, sub);
    \\    return scale * (float)q;
    \\}
    \\
    \\kernel void antfly_q3_k_small_batch_bias_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q3_k [[buffer(1)]],
    \\    const device float *bias [[buffer(2)]],
    \\    device float *output [[buffer(3)]],
    \\    constant int &rows [[buffer(4)]],
    \\    constant int &in_dim [[buffer(5)]],
    \\    constant int &out_dim [[buffer(6)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 32) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q3_k + ((col * block_count + block_idx) * 110);
    \\            for (int lane = (int)tid; lane < 256; lane += 32) {
    \\                acc += input[row * in_dim + (block_idx << 8) + lane] * antfly_q3_k_dequant_lane(block, lane);
    \\            }
    \\        }
    \\    }
    \\
    \\    acc = simd_sum(acc);
    \\    if (tid == 0) output[row * out_dim + col] = acc + bias[col];
    \\}
    \\
;

const first_general_metal_q3_bias_gelu_source =
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
    \\
    \\// Dev-only generated Metal candidate from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q3_k/rows_2_8/bias_gelu/small_batch
    \\// kernel_id=antfly_q3_k_small_batch_bias_gelu_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=false
    \\// General MSL lowering smoke for descriptor-driven quant matmul epilogues.
    \\// Production Metal dispatch stays on native handwritten MSL until this
    \\// candidate clears correctness and benchmark gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_gelu(float x) {
    \\    return 0.5f * x * (1.0f + fast::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    \\}
    \\
    \\static inline int antfly_q3_k_raw_scale(const device uchar *scale_data, uint sub) {
    \\    const uint i = sub & 3u;
    \\    uint low = 0u;
    \\    uint high = 0u;
    \\    if (sub < 4u) {
    \\        low = (uint)(scale_data[i] & 0x0Fu);
    \\        high = (uint)(scale_data[8 + i] & 0x03u);
    \\    } else if (sub < 8u) {
    \\        low = (uint)(scale_data[4 + i] & 0x0Fu);
    \\        high = (uint)((scale_data[8 + i] >> 2) & 0x03u);
    \\    } else if (sub < 12u) {
    \\        low = (uint)((scale_data[i] >> 4) & 0x0Fu);
    \\        high = (uint)((scale_data[8 + i] >> 4) & 0x03u);
    \\    } else {
    \\        low = (uint)((scale_data[4 + i] >> 4) & 0x0Fu);
    \\        high = (uint)((scale_data[8 + i] >> 6) & 0x03u);
    \\    }
    \\    return (int)(low | (high << 4)) - 32;
    \\}
    \\
    \\static inline float antfly_q3_k_dequant_lane(const device uchar *block, int lane) {
    \\    const uint sub = (uint)lane >> 4;
    \\    const uint i = (uint)lane & 15u;
    \\    const uint chunk = sub >> 3;
    \\    const uint group = (sub & 7u) >> 1;
    \\    const uint l_base = (sub & 1u) << 4;
    \\    const uint l = l_base + i;
    \\    const uint q_base = chunk << 5;
    \\    const uint shift = group << 1;
    \\    const uint hm_bit = (chunk << 2) + group;
    \\    const int low2 = (int)(((uint)block[32 + q_base + l] >> shift) & 0x03u);
    \\    const int high1 = (int)(((uint)block[l] >> hm_bit) & 0x01u);
    \\    const int q = low2 + high1 * 4 - 4;
    \\    const float scale = antfly_half_le_to_float(block + 108) * (float)antfly_q3_k_raw_scale(block + 96, sub);
    \\    return scale * (float)q;
    \\}
    \\
    \\kernel void antfly_q3_k_small_batch_bias_gelu_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q3_k [[buffer(1)]],
    \\    const device float *bias [[buffer(2)]],
    \\    device float *output [[buffer(3)]],
    \\    constant int &rows [[buffer(4)]],
    \\    constant int &in_dim [[buffer(5)]],
    \\    constant int &out_dim [[buffer(6)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 32) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q3_k + ((col * block_count + block_idx) * 110);
    \\            for (int lane = (int)tid; lane < 256; lane += 32) {
    \\                acc += input[row * in_dim + (block_idx << 8) + lane] * antfly_q3_k_dequant_lane(block, lane);
    \\            }
    \\        }
    \\    }
    \\
    \\    acc = simd_sum(acc);
    \\    if (tid == 0) output[row * out_dim + col] = antfly_gelu(acc + bias[col]);
    \\}
    \\
;

const first_general_metal_q8_1_source =
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
    \\
    \\// Dev-only generated Metal candidate from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q8_1/rows_2_8/none/small_batch
    \\// kernel_id=antfly_q8_1_small_batch_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_q8_1_dequant_lane(const device uchar *block, int lane) {
    \\    const float d = antfly_half_le_to_float(block);
    \\    const int q = (int)as_type<char>(block[4 + lane]);
    \\    return d * (float)q;
    \\}
    \\
    \\kernel void antfly_q8_1_small_batch_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q8_1 [[buffer(1)]],
    \\    device float *output [[buffer(2)]],
    \\    constant int &rows [[buffer(3)]],
    \\    constant int &in_dim [[buffer(4)]],
    \\    constant int &out_dim [[buffer(5)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 31) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 5;
    \\    if (tid < 32) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q8_1 + ((col * block_count + block_idx) * 36);
    \\            const int lane = (int)tid;
    \\            acc += input[row * in_dim + (block_idx << 5) + lane] * antfly_q8_1_dequant_lane(block, lane);
    \\        }
    \\    }
    \\
    \\    acc = simd_sum(acc);
    \\    if (tid == 0) output[row * out_dim + col] = acc;
    \\}
    \\
;

const first_general_metal_q8_k_source =
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
    \\
    \\// Generated Metal artifact source from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q8_k/rows_2_8/none/small_batch
    \\// kernel_id=antfly_q8_k_small_batch_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_f32_le_to_float(const device uchar *p) {
    \\    const uint bits = (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
    \\    return as_type<float>(bits);
    \\}
    \\
    \\static inline float antfly_q8_k_dequant_lane(const device uchar *block, int lane) {
    \\    const float d = antfly_f32_le_to_float(block);
    \\    const int q = (int)as_type<char>(block[4 + lane]);
    \\    return d * (float)q;
    \\}
    \\
    \\kernel void antfly_q8_k_small_batch_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q8_k [[buffer(1)]],
    \\    device float *output [[buffer(2)]],
    \\    constant int &rows [[buffer(3)]],
    \\    constant int &in_dim [[buffer(4)]],
    \\    constant int &out_dim [[buffer(5)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 32) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q8_k + ((col * block_count + block_idx) * 292);
    \\            for (int lane = (int)tid; lane < 256; lane += 32) {
    \\                acc += input[row * in_dim + (block_idx << 8) + lane] * antfly_q8_k_dequant_lane(block, lane);
    \\            }
    \\        }
    \\    }
    \\
    \\    acc = simd_sum(acc);
    \\    if (tid == 0) output[row * out_dim + col] = acc;
    \\}
    \\
;

const first_general_metal_q5_source =
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
    \\
    \\// Generated Metal artifact source from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q5_k/rows_2_8/none/small_batch
    \\// kernel_id=antfly_q5_k_small_batch_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, provider-route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline void antfly_q5_k_unpack_scale_min(
    \\    const device uchar *scales,
    \\    int sub,
    \\    thread float &scale,
    \\    thread float &min_v
    \\) {
    \\    if (sub < 4) {
    \\        scale = (float)(scales[sub] & 63u);
    \\        min_v = (float)(scales[sub + 4] & 63u);
    \\        return;
    \\    }
    \\    scale = (float)((scales[sub + 4] & 0x0fu) | ((scales[sub - 4] >> 6) << 4));
    \\    min_v = (float)((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
    \\}
    \\
    \\static inline float antfly_q5_k_dequant_lane(const device uchar *block, int lane, float d, float dmin) {
    \\    const device uchar *scales = block + 4;
    \\    const device uchar *qh = block + 16;
    \\    const device uchar *ql = block + 48;
    \\    const int sub = lane >> 5;
    \\    const int i = lane & 31;
    \\    const int q_index = (sub >> 1) * 32 + i;
    \\    const uchar packed = ql[q_index];
    \\    const int low = (sub & 1) == 0 ? (int)(packed & 0x0fu) : (int)(packed >> 4);
    \\    const int high = (int)((qh[i] >> sub) & 1u);
    \\    const int q = low + high * 16;
    \\    float raw_scale = 0.0f;
    \\    float raw_min = 0.0f;
    \\    antfly_q5_k_unpack_scale_min(scales, sub, raw_scale, raw_min);
    \\    return d * raw_scale * (float)q - dmin * raw_min;
    \\}
    \\
    \\kernel void antfly_q5_k_small_batch_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q5_k [[buffer(1)]],
    \\    device float *output [[buffer(2)]],
    \\    constant int &rows [[buffer(3)]],
    \\    constant int &in_dim [[buffer(4)]],
    \\    constant int &out_dim [[buffer(5)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 64) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q5_k + ((col * block_count + block_idx) * 176);
    \\            const int base = block_idx << 8;
    \\            const float d = antfly_half_le_to_float(block);
    \\            const float dmin = antfly_half_le_to_float(block + 2);
    \\            for (int lane = (int)tid; lane < 256; lane += 64) {
    \\                acc += input[row * in_dim + base + lane] * antfly_q5_k_dequant_lane(block, lane, d, dmin);
    \\            }
    \\        }
    \\    }
    \\
    \\    threadgroup float partial[64];
    \\    if (tid < 64) partial[tid] = acc;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = 32; stride > 0; stride >>= 1) {
    \\        if (tid < stride) partial[tid] += partial[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (tid == 0) output[row * out_dim + col] = partial[0];
    \\}
    \\
;

const first_general_metal_q5_bias_source =
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
    \\
    \\// Generated Metal artifact source from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q5_k/rows_2_8/bias/small_batch
    \\// kernel_id=antfly_q5_k_small_batch_bias_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, provider-route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline void antfly_q5_k_unpack_scale_min(
    \\    const device uchar *scales,
    \\    int sub,
    \\    thread float &scale,
    \\    thread float &min_v
    \\) {
    \\    if (sub < 4) {
    \\        scale = (float)(scales[sub] & 63u);
    \\        min_v = (float)(scales[sub + 4] & 63u);
    \\        return;
    \\    }
    \\    scale = (float)((scales[sub + 4] & 0x0fu) | ((scales[sub - 4] >> 6) << 4));
    \\    min_v = (float)((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
    \\}
    \\
    \\static inline float antfly_q5_k_dequant_lane(const device uchar *block, int lane, float d, float dmin) {
    \\    const device uchar *scales = block + 4;
    \\    const device uchar *qh = block + 16;
    \\    const device uchar *ql = block + 48;
    \\    const int sub = lane >> 5;
    \\    const int i = lane & 31;
    \\    const int q_index = (sub >> 1) * 32 + i;
    \\    const uchar packed = ql[q_index];
    \\    const int low = (sub & 1) == 0 ? (int)(packed & 0x0fu) : (int)(packed >> 4);
    \\    const int high = (int)((qh[i] >> sub) & 1u);
    \\    const int q = low + high * 16;
    \\    float raw_scale = 0.0f;
    \\    float raw_min = 0.0f;
    \\    antfly_q5_k_unpack_scale_min(scales, sub, raw_scale, raw_min);
    \\    return d * raw_scale * (float)q - dmin * raw_min;
    \\}
    \\
    \\kernel void antfly_q5_k_small_batch_bias_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q5_k [[buffer(1)]],
    \\    const device float *bias [[buffer(2)]],
    \\    device float *output [[buffer(3)]],
    \\    constant int &rows [[buffer(4)]],
    \\    constant int &in_dim [[buffer(5)]],
    \\    constant int &out_dim [[buffer(6)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]],
    \\    ushort lane_id [[thread_index_in_simdgroup]],
    \\    ushort simdgroup_id [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 128) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q5_k + ((col * block_count + block_idx) * 176);
    \\            const int base = block_idx << 8;
    \\            const float d = antfly_half_le_to_float(block);
    \\            const float dmin = antfly_half_le_to_float(block + 2);
    \\            for (int lane = (int)tid; lane < 256; lane += 128) {
    \\                acc += input[row * in_dim + base + lane] * antfly_q5_k_dequant_lane(block, lane, d, dmin);
    \\            }
    \\        }
    \\    }
    \\
    \\    threadgroup float partial[32];
    \\    acc = simd_sum(acc);
    \\    if (lane_id == 0u) partial[simdgroup_id] = acc;
    \\    if (simdgroup_id == 0u && lane_id >= 4u) partial[lane_id] = 0.0f;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    const float total = simd_sum(partial[lane_id]);
    \\    if (lane_id == 0u && simdgroup_id == 0u) output[row * out_dim + col] = total + bias[col];
    \\}
    \\
;

const first_general_metal_q5_bias_gelu_source =
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
    \\
    \\// Generated Metal artifact source from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q5_k/rows_2_8/bias_gelu/small_batch
    \\// kernel_id=antfly_q5_k_small_batch_bias_gelu_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, provider-route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_gelu(float x) {
    \\    return 0.5f * x * (1.0f + fast::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    \\}
    \\
    \\static inline void antfly_q5_k_unpack_scale_min(
    \\    const device uchar *scales,
    \\    int sub,
    \\    thread float &scale,
    \\    thread float &min_v
    \\) {
    \\    if (sub < 4) {
    \\        scale = (float)(scales[sub] & 63u);
    \\        min_v = (float)(scales[sub + 4] & 63u);
    \\        return;
    \\    }
    \\    scale = (float)((scales[sub + 4] & 0x0fu) | ((scales[sub - 4] >> 6) << 4));
    \\    min_v = (float)((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
    \\}
    \\
    \\static inline float antfly_q5_k_dequant_lane(const device uchar *block, int lane, float d, float dmin) {
    \\    const device uchar *scales = block + 4;
    \\    const device uchar *qh = block + 16;
    \\    const device uchar *ql = block + 48;
    \\    const int sub = lane >> 5;
    \\    const int i = lane & 31;
    \\    const int q_index = (sub >> 1) * 32 + i;
    \\    const uchar packed = ql[q_index];
    \\    const int low = (sub & 1) == 0 ? (int)(packed & 0x0fu) : (int)(packed >> 4);
    \\    const int high = (int)((qh[i] >> sub) & 1u);
    \\    const int q = low + high * 16;
    \\    float raw_scale = 0.0f;
    \\    float raw_min = 0.0f;
    \\    antfly_q5_k_unpack_scale_min(scales, sub, raw_scale, raw_min);
    \\    return d * raw_scale * (float)q - dmin * raw_min;
    \\}
    \\
    \\kernel void antfly_q5_k_small_batch_bias_gelu_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q5_k [[buffer(1)]],
    \\    const device float *bias [[buffer(2)]],
    \\    device float *output [[buffer(3)]],
    \\    constant int &rows [[buffer(4)]],
    \\    constant int &in_dim [[buffer(5)]],
    \\    constant int &out_dim [[buffer(6)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]],
    \\    ushort lane_id [[thread_index_in_simdgroup]],
    \\    ushort simdgroup_id [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 128) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q5_k + ((col * block_count + block_idx) * 176);
    \\            const int base = block_idx << 8;
    \\            const float d = antfly_half_le_to_float(block);
    \\            const float dmin = antfly_half_le_to_float(block + 2);
    \\            for (int lane = (int)tid; lane < 256; lane += 128) {
    \\                acc += input[row * in_dim + base + lane] * antfly_q5_k_dequant_lane(block, lane, d, dmin);
    \\            }
    \\        }
    \\    }
    \\
    \\    threadgroup float partial[32];
    \\    acc = simd_sum(acc);
    \\    if (lane_id == 0u) partial[simdgroup_id] = acc;
    \\    if (simdgroup_id == 0u && lane_id >= 4u) partial[lane_id] = 0.0f;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    const float total = simd_sum(partial[lane_id]);
    \\    if (lane_id == 0u && simdgroup_id == 0u) output[row * out_dim + col] = antfly_gelu(total + bias[col]);
    \\}
    \\
;

const first_general_metal_q6_source =
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
    \\
    \\// Generated Metal candidate artifact from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q6_k/rows_2_8/none/small_batch
    \\// kernel_id=antfly_q6_k_small_batch_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, provider-route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_q6_k_dequant_lane(const device uchar *block, int lane, float d) {
    \\    const device uchar *ql = block;
    \\    const device uchar *qh = block + 128;
    \\    const device uchar *scales = block + 192;
    \\    const int sub = lane >> 4;
    \\    const int i = lane & 15;
    \\    const int half_idx = sub >> 3;
    \\    const int group = (sub & 7) >> 1;
    \\    const int l = ((sub & 1) << 4) + i;
    \\    const int ql_off = half_idx * 64 + (group & 1) * 32;
    \\    const int qh_off = half_idx * 32;
    \\    const int qh_shift = group * 2;
    \\    const int nibble_shift = (group >> 1) * 4;
    \\    const int low4 = (int)((ql[ql_off + l] >> nibble_shift) & 0x0fu);
    \\    const int high2 = (int)((qh[qh_off + l] >> qh_shift) & 0x03u);
    \\    const int q = (low4 | (high2 << 4)) - 32;
    \\    const int scale_u = (int)scales[sub];
    \\    const int scale = scale_u >= 128 ? scale_u - 256 : scale_u;
    \\    return d * (float)scale * (float)q;
    \\}
    \\
    \\kernel void antfly_q6_k_small_batch_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q6_k [[buffer(1)]],
    \\    device float *output [[buffer(2)]],
    \\    constant int &rows [[buffer(3)]],
    \\    constant int &in_dim [[buffer(4)]],
    \\    constant int &out_dim [[buffer(5)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 128) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q6_k + ((col * block_count + block_idx) * 210);
    \\            const int base = block_idx << 8;
    \\            const float d = antfly_half_le_to_float(block + 208);
    \\            for (int lane = (int)tid; lane < 256; lane += 128) {
    \\                acc += input[row * in_dim + base + lane] * antfly_q6_k_dequant_lane(block, lane, d);
    \\            }
    \\        }
    \\    }
    \\
    \\    threadgroup float partial[128];
    \\    if (tid < 128) partial[tid] = acc;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    for (uint stride = 64; stride > 0; stride >>= 1) {
    \\        if (tid < stride) partial[tid] += partial[tid + stride];
    \\        threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    }
    \\    if (tid == 0) output[row * out_dim + col] = partial[0];
    \\}
    \\
;

const first_general_metal_q6_bias_source =
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
    \\
    \\// Generated Metal candidate artifact from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q6_k/rows_2_8/bias/small_batch
    \\// kernel_id=antfly_q6_k_small_batch_bias_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, provider-route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_q6_k_dequant_lane(const device uchar *block, int lane, float d) {
    \\    const device uchar *ql = block;
    \\    const device uchar *qh = block + 128;
    \\    const device uchar *scales = block + 192;
    \\    const int sub = lane >> 4;
    \\    const int i = lane & 15;
    \\    const int half_idx = sub >> 3;
    \\    const int group = (sub & 7) >> 1;
    \\    const int l = ((sub & 1) << 4) + i;
    \\    const int ql_off = half_idx * 64 + (group & 1) * 32;
    \\    const int qh_off = half_idx * 32;
    \\    const int qh_shift = group * 2;
    \\    const int nibble_shift = (group >> 1) * 4;
    \\    const int low4 = (int)((ql[ql_off + l] >> nibble_shift) & 0x0fu);
    \\    const int high2 = (int)((qh[qh_off + l] >> qh_shift) & 0x03u);
    \\    const int q = (low4 | (high2 << 4)) - 32;
    \\    const int scale_u = (int)scales[sub];
    \\    const int scale = scale_u >= 128 ? scale_u - 256 : scale_u;
    \\    return d * (float)scale * (float)q;
    \\}
    \\
    \\kernel void antfly_q6_k_small_batch_bias_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q6_k [[buffer(1)]],
    \\    const device float *bias [[buffer(2)]],
    \\    device float *output [[buffer(3)]],
    \\    constant int &rows [[buffer(4)]],
    \\    constant int &in_dim [[buffer(5)]],
    \\    constant int &out_dim [[buffer(6)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]],
    \\    ushort lane_id [[thread_index_in_simdgroup]],
    \\    ushort simdgroup_id [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 128) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q6_k + ((col * block_count + block_idx) * 210);
    \\            const int base = block_idx << 8;
    \\            const float d = antfly_half_le_to_float(block + 208);
    \\            for (int lane = (int)tid; lane < 256; lane += 128) {
    \\                acc += input[row * in_dim + base + lane] * antfly_q6_k_dequant_lane(block, lane, d);
    \\            }
    \\        }
    \\    }
    \\
    \\    threadgroup float partial[32];
    \\    acc = simd_sum(acc);
    \\    if (lane_id == 0u) partial[simdgroup_id] = acc;
    \\    if (simdgroup_id == 0u && lane_id >= 4u) partial[lane_id] = 0.0f;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    const float total = simd_sum(partial[lane_id]);
    \\    if (lane_id == 0u && simdgroup_id == 0u) output[row * out_dim + col] = total + bias[col];
    \\}
    \\
;

const first_general_metal_q6_bias_gelu_source =
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
    \\
    \\// Generated Metal candidate artifact from graph/quant_kernel_compiler.zig.
    \\// plan_id=metal/q6_k/rows_2_8/bias_gelu/small_batch
    \\// kernel_id=antfly_q6_k_small_batch_bias_gelu_msl_v1
    \\// production_baseline=metal_handwritten_quant_matmul
    \\// production_enabled=true
    \\// Promoted after sequential Metal runtime evidence cleared correctness,
    \\// route, provider-route, and speedup gates.
    \\
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\
    \\static inline float antfly_half_le_to_float(const device uchar *p) {
    \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
    \\    return (float)as_type<half>(bits);
    \\}
    \\
    \\static inline float antfly_gelu(float x) {
    \\    return 0.5f * x * (1.0f + fast::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    \\}
    \\
    \\static inline float antfly_q6_k_dequant_lane(const device uchar *block, int lane, float d) {
    \\    const device uchar *ql = block;
    \\    const device uchar *qh = block + 128;
    \\    const device uchar *scales = block + 192;
    \\    const int sub = lane >> 4;
    \\    const int i = lane & 15;
    \\    const int half_idx = sub >> 3;
    \\    const int group = (sub & 7) >> 1;
    \\    const int l = ((sub & 1) << 4) + i;
    \\    const int ql_off = half_idx * 64 + (group & 1) * 32;
    \\    const int qh_off = half_idx * 32;
    \\    const int qh_shift = group * 2;
    \\    const int nibble_shift = (group >> 1) * 4;
    \\    const int low4 = (int)((ql[ql_off + l] >> nibble_shift) & 0x0fu);
    \\    const int high2 = (int)((qh[qh_off + l] >> qh_shift) & 0x03u);
    \\    const int q = (low4 | (high2 << 4)) - 32;
    \\    const int scale_u = (int)scales[sub];
    \\    const int scale = scale_u >= 128 ? scale_u - 256 : scale_u;
    \\    return d * (float)scale * (float)q;
    \\}
    \\
    \\kernel void antfly_q6_k_small_batch_bias_gelu_msl_v1(
    \\    const device float *input [[buffer(0)]],
    \\    const device uchar *weight_q6_k [[buffer(1)]],
    \\    const device float *bias [[buffer(2)]],
    \\    device float *output [[buffer(3)]],
    \\    constant int &rows [[buffer(4)]],
    \\    constant int &in_dim [[buffer(5)]],
    \\    constant int &out_dim [[buffer(6)]],
    \\    uint3 thread_pos [[thread_position_in_threadgroup]],
    \\    uint3 group_pos [[threadgroup_position_in_grid]],
    \\    ushort lane_id [[thread_index_in_simdgroup]],
    \\    ushort simdgroup_id [[simdgroup_index_in_threadgroup]]
    \\) {
    \\    const uint tid = thread_pos.x;
    \\    const int col = (int)group_pos.x;
    \\    const int row = (int)group_pos.y;
    \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & 255) != 0) return;
    \\
    \\    float acc = 0.0f;
    \\    const int block_count = in_dim >> 8;
    \\    if (tid < 128) {
    \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {
    \\            const device uchar *block = weight_q6_k + ((col * block_count + block_idx) * 210);
    \\            const int base = block_idx << 8;
    \\            const float d = antfly_half_le_to_float(block + 208);
    \\            for (int lane = (int)tid; lane < 256; lane += 128) {
    \\                acc += input[row * in_dim + base + lane] * antfly_q6_k_dequant_lane(block, lane, d);
    \\            }
    \\        }
    \\    }
    \\
    \\    threadgroup float partial[32];
    \\    acc = simd_sum(acc);
    \\    if (lane_id == 0u) partial[simdgroup_id] = acc;
    \\    if (simdgroup_id == 0u && lane_id >= 4u) partial[lane_id] = 0.0f;
    \\    threadgroup_barrier(mem_flags::mem_threadgroup);
    \\    const float total = simd_sum(partial[lane_id]);
    \\    if (lane_id == 0u && simdgroup_id == 0u) output[row * out_dim + col] = antfly_gelu(total + bias[col]);
    \\}
    \\
;

pub const first_coverage = buildFirstCoverage();

const first_registry_entries = buildFirstRegistry();
pub const first_registry = QuantKernelRegistry{ .entries = &first_registry_entries };

pub const first_conformance = buildFirstConformance();

pub fn specFor(format: quant_matmul.Format) ?QuantKernelSpec {
    return switch (format) {
        .q1_0 => q1_0_spec,
        .i2_s => i2_s_spec,
        .i8_s => i8_s_spec,
        .q2_k => q2_k_spec,
        .q3_k => q3_k_spec,
        .q4_0 => q4_0_spec,
        .q4_1 => q4_1_spec,
        .q5_0 => q5_0_spec,
        .q5_1 => q5_1_spec,
        .q4_k => q4_k_spec,
        .q5_k => q5_k_spec,
        .q6_k => q6_k_spec,
        .q8_0 => q8_0_spec,
        .q8_1 => q8_1_spec,
        .q8_k => q8_k_spec,
        .tq1_0 => tq1_0_spec,
        .tq2_0 => tq2_0_spec,
        .iq2_xxs => iq2_xxs_spec,
        .iq2_xs => iq2_xs_spec,
        .iq2_s => iq2_s_spec,
        .iq3_xxs => iq3_xxs_spec,
        .iq3_s => iq3_s_spec,
        .iq1_s => iq1_s_spec,
        .iq1_m => iq1_m_spec,
        .iq4_nl => iq4_nl_spec,
        .iq4_xs => iq4_xs_spec,
        .mxfp4 => mxfp4_spec,
        .nvfp4 => nvfp4_spec,
        else => null,
    };
}

pub fn buildIr(format: quant_matmul.Format, row_bucket: quant_matmul.RowBucket, epilogue: Epilogue) ?QuantKernelIR {
    const spec = specFor(format) orelse return null;
    const dispatch = dispatchForRowBucket(row_bucket) orelse return null;
    if (!spec.supportsSchedule(dispatch) or !spec.supportsEpilogue(epilogue)) return null;
    return .{
        .format = format,
        .row_bucket = row_bucket,
        .dispatch = dispatch,
        .epilogue = epilogue,
        .ops = irOpsForEpilogue(epilogue),
    };
}

fn supportsEpilogueForBackend(spec: QuantKernelSpec, backend: Backend, epilogue: Epilogue) bool {
    if (!spec.supportsEpilogue(epilogue)) return false;
    if (backend == .cuda and spec.format == .q6_k and epilogue != .none) return false;
    return true;
}

pub fn emitFirstLazyCudaSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_lazy_cuda_source);
}

pub fn emitFirstLazyMetalSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_lazy_metal_source);
}

pub fn emitFirstGeneralMetalQ40Source(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q4_0_source);
}

pub fn emitFirstGeneralMetalQ41Source(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q4_1_source);
}

pub fn emitFirstGeneralMetalQ50Source(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q5_0_source);
}

pub fn emitFirstGeneralMetalQ51Source(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q5_1_source);
}

pub fn emitFirstGeneralMetalQ4Source(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q4_source);
}

pub fn emitFirstGeneralMetalQ4BiasSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q4_bias_source);
}

pub fn emitFirstGeneralMetalQ8Source(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q8_source);
}

pub fn emitFirstGeneralMetalQ8BiasSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q8_bias_source);
}

pub fn emitFirstGeneralMetalQ8BiasGeluSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q8_bias_gelu_source);
}

pub fn emitFirstGeneralMetalQ8ReluSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q8_relu_source);
}

pub fn emitFirstGeneralMetalQ2Source(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q2_source);
}

pub fn emitFirstGeneralMetalQ2BiasSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q2_bias_source);
}

pub fn emitFirstGeneralMetalQ2BiasGeluSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q2_bias_gelu_source);
}

pub fn emitFirstGeneralMetalQ3Source(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q3_source);
}

pub fn emitFirstGeneralMetalQ3BiasSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q3_bias_source);
}

pub fn emitFirstGeneralMetalQ3BiasGeluSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q3_bias_gelu_source);
}

pub fn emitFirstGeneralMetalQ81Source(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q8_1_source);
}

pub fn emitFirstGeneralMetalQ8KSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q8_k_source);
}

pub fn emitFirstGeneralMetalQ5Source(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q5_source);
}

pub fn emitFirstGeneralMetalQ5BiasSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q5_bias_source);
}

pub fn emitFirstGeneralMetalQ5BiasGeluSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q5_bias_gelu_source);
}

pub fn emitFirstGeneralMetalQ6Source(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q6_source);
}

pub fn emitFirstGeneralMetalQ6BiasSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q6_bias_source);
}

pub fn emitFirstGeneralMetalQ6BiasGeluSource(allocator: std.mem.Allocator) ![]u8 {
    return allocator.dupe(u8, first_general_metal_q6_bias_gelu_source);
}

pub fn firstLazyCudaSource() []const u8 {
    return first_lazy_cuda_source;
}

pub fn firstLazyMetalSource() []const u8 {
    return first_lazy_metal_source;
}

pub fn firstGeneralMetalQ40Source() []const u8 {
    return first_general_metal_q4_0_source;
}

pub fn firstGeneralMetalQ41Source() []const u8 {
    return first_general_metal_q4_1_source;
}

pub fn firstGeneralMetalQ50Source() []const u8 {
    return first_general_metal_q5_0_source;
}

pub fn firstGeneralMetalQ51Source() []const u8 {
    return first_general_metal_q5_1_source;
}

pub fn firstGeneralMetalQ4Source() []const u8 {
    return first_general_metal_q4_source;
}

pub fn firstGeneralMetalQ4BiasSource() []const u8 {
    return first_general_metal_q4_bias_source;
}

pub fn firstGeneralMetalQ8Source() []const u8 {
    return first_general_metal_q8_source;
}

pub fn firstGeneralMetalQ8BiasSource() []const u8 {
    return first_general_metal_q8_bias_source;
}

pub fn firstGeneralMetalQ8BiasGeluSource() []const u8 {
    return first_general_metal_q8_bias_gelu_source;
}

pub fn firstGeneralMetalQ8ReluSource() []const u8 {
    return first_general_metal_q8_relu_source;
}

pub fn firstGeneralMetalQ2Source() []const u8 {
    return first_general_metal_q2_source;
}

pub fn firstGeneralMetalQ2BiasSource() []const u8 {
    return first_general_metal_q2_bias_source;
}

pub fn firstGeneralMetalQ2BiasGeluSource() []const u8 {
    return first_general_metal_q2_bias_gelu_source;
}

pub fn firstGeneralMetalQ3Source() []const u8 {
    return first_general_metal_q3_source;
}

pub fn firstGeneralMetalQ3BiasSource() []const u8 {
    return first_general_metal_q3_bias_source;
}

pub fn firstGeneralMetalQ3BiasGeluSource() []const u8 {
    return first_general_metal_q3_bias_gelu_source;
}

pub fn firstGeneralMetalQ81Source() []const u8 {
    return first_general_metal_q8_1_source;
}

pub fn firstGeneralMetalQ8KSource() []const u8 {
    return first_general_metal_q8_k_source;
}

pub fn firstGeneralMetalQ5Source() []const u8 {
    return first_general_metal_q5_source;
}

pub fn firstGeneralMetalQ5BiasSource() []const u8 {
    return first_general_metal_q5_bias_source;
}

pub fn firstGeneralMetalQ5BiasGeluSource() []const u8 {
    return first_general_metal_q5_bias_gelu_source;
}

pub fn firstGeneralMetalQ6Source() []const u8 {
    return first_general_metal_q6_source;
}

pub fn firstGeneralMetalQ6BiasSource() []const u8 {
    return first_general_metal_q6_bias_source;
}

pub fn firstGeneralMetalQ6BiasGeluSource() []const u8 {
    return first_general_metal_q6_bias_gelu_source;
}

pub fn benchmarkManifestJson(allocator: std.mem.Allocator) ![]u8 {
    var records: [first_benchmarks.len]BenchmarkManifestRecord = undefined;
    for (first_benchmarks, 0..) |bench, index| {
        records[index] = benchmarkManifestRecord(bench);
    }
    var metal_production_records: [first_metal_production_benchmark_cases.len]MetalProductionBenchmarkManifestRecord = undefined;
    for (first_metal_production_benchmark_cases, 0..) |case, index| {
        metal_production_records[index] = metalProductionBenchmarkManifestRecord(case);
    }
    return std.json.Stringify.valueAlloc(allocator, BenchmarkManifest{
        .schema = first_benchmark_manifest_schema,
        .benchmark_count = first_benchmarks.len,
        .evidence_count = first_benchmark_evidence.len,
        .metal_evidence_count = first_metal_runtime_evidence.len,
        .metal_promotion_warmup_repeat_runs = metal_promotion_warmup_repeat_runs,
        .metal_production_regression_expected_kernel_count = first_metal_production_regression_expected_kernel_count,
        .metal_production_regression_expected_case_count = first_metal_production_regression_expected_case_count,
        .metal_production_regression_expected_route_ready_count = first_metal_production_regression_expected_route_ready_count,
        .metal_production_regression_case_fingerprint = metalProductionBenchmarkCaseManifestFingerprint(),
        .metal_production_regression_build_command = first_metal_production_regression_build_command,
        .metal_production_regression_evidence_command = first_metal_production_regression_evidence_command,
        .metal_production_regression_cases = &metal_production_records,
        .benchmarks = &records,
        .evidence_records = &first_benchmark_evidence,
        .metal_evidence_records = &first_metal_runtime_evidence,
    }, .{ .whitespace = .indent_2 });
}

pub fn artifactManifestJson(allocator: std.mem.Allocator) ![]u8 {
    var records: [first_generated_artifacts.len]ArtifactManifestRecord = undefined;
    var owned_route_commands = [_][]const u8{""} ** first_generated_artifacts.len;
    var owned_blocker_check_commands = [_][]const u8{""} ** first_generated_artifacts.len;
    defer for (owned_route_commands) |command| {
        if (command.len != 0) allocator.free(command);
    };
    defer for (owned_blocker_check_commands) |command| {
        if (command.len != 0) allocator.free(command);
    };
    for (first_generated_artifacts, 0..) |artifact, index| {
        owned_route_commands[index] = try artifactRuntimeRouteEvidenceCommand(allocator, artifact);
        owned_blocker_check_commands[index] = try artifactPromotionBlockerCheckCommand(allocator, artifact);
        records[index] = artifactManifestRecord(artifact, owned_route_commands[index], owned_blocker_check_commands[index]);
    }
    return std.json.Stringify.valueAlloc(allocator, ArtifactManifest{
        .schema = first_artifact_manifest_schema,
        .artifact_count = first_generated_artifacts.len,
        .checked_in_metal_evidence_count = first_metal_runtime_evidence_count,
        .metal_promotion_blocker_evidence_count = first_metal_promotion_blocker_evidence_count,
        .metal_promotion_blocker_evidence_path_count = metalPromotionBlockerEvidencePathCount(),
        .metal_promotion_blocker_evidence_expected_case_count = first_metal_promotion_blocker_evidence_expected_case_count,
        .metal_promotion_blocker_evidence_expected_route_ready_count = first_metal_promotion_blocker_evidence_expected_route_ready_count,
        .metal_promotion_blocker_check_command_count = metalPromotionBlockerEvidencePathCount(),
        .metal_promotion_blocker_skipped_no_path_count = metalPromotionBlockerSkippedNoPathCount(),
        .metal_promotion_blocker_cleared_requires_checked_in_evidence = true,
        .metal_promotion_blocker_speedup_gate_missing_count = metalPromotionBlockerEvidenceCount(metal_blocker_speedup_gate_missing),
        .metal_promotion_blocker_unstable_benchmark_timing_count = metalPromotionBlockerEvidenceCount(metal_blocker_unstable_benchmark_timing),
        .metal_promotion_blocker_unsupported_handwritten_count = metalPromotionBlockerEvidenceCount(metal_blocker_unsupported_handwritten),
        .metal_unsupported_handwritten_baseline_blocks_promotion = first_metal_unsupported_handwritten_baseline_blocks_promotion,
        .metal_unsupported_handwritten_baseline_uses_runtime_route_all_evidence = first_metal_unsupported_handwritten_baseline_uses_runtime_route_all_evidence,
        .metal_unsupported_handwritten_baseline_has_promotion_evidence_path = first_metal_unsupported_handwritten_baseline_has_promotion_evidence_path,
        .metal_local_check_command = first_metal_local_check_command,
        .metal_model_local_check_command = first_metal_model_local_check_command,
        .metal_model_generated_route_check_command = first_metal_model_generated_route_check_command,
        .metal_model_generated_q8_0_small_batch_min = first_metal_model_generated_q8_0_small_batch_min,
        .metal_model_generated_q4_0_small_batch_min = first_metal_model_generated_q4_0_small_batch_min,
        .metal_industry_local_check_command = first_metal_industry_local_check_command,
        .metal_runtime_route_all_build_command = first_metal_runtime_route_all_build_command,
        .metal_runtime_route_all_evidence_command = first_metal_runtime_route_all_evidence_command,
        .metal_runtime_route_all_check_command = first_metal_runtime_route_all_check_command,
        .metal_runtime_route_all_expected_case_count = first_metal_runtime_route_all_expected_case_count,
        .metal_runtime_route_all_expected_route_ready_count = first_metal_runtime_route_all_expected_route_ready_count,
        .metal_runtime_route_all_expected_provider_route_count = first_metal_runtime_route_all_expected_provider_route_count,
        .metal_production_regression_expected_kernel_count = first_metal_production_regression_expected_kernel_count,
        .metal_production_regression_expected_case_count = first_metal_production_regression_expected_case_count,
        .metal_production_regression_expected_route_ready_count = first_metal_production_regression_expected_route_ready_count,
        .metal_promotion_warmup_repeat_runs = metal_promotion_warmup_repeat_runs,
        .metal_production_regression_route_ready_is_hard_gate = first_metal_production_regression_route_ready_is_hard_gate,
        .metal_production_regression_missing_provider_route_is_hard_gate = first_metal_production_regression_missing_provider_route_is_hard_gate,
        .metal_production_regression_speedup_gate_missing_is_hard_gate = first_metal_production_regression_speedup_gate_missing_is_hard_gate,
        .metal_production_regression_unstable_benchmark_timing_is_hard_gate = first_metal_production_regression_unstable_benchmark_timing_is_hard_gate,
        .metal_production_regression_build_command = first_metal_production_regression_build_command,
        .metal_production_regression_evidence_command = first_metal_production_regression_evidence_command,
        .metal_blocker_strict_check_command = first_metal_blocker_strict_check_command,
        .artifacts = &records,
        .metal_evidence_records = &first_metal_runtime_evidence,
    }, .{ .whitespace = .indent_2 });
}

pub fn specManifestJson(allocator: std.mem.Allocator) ![]u8 {
    var records: [first_formats.len]SpecManifestRecord = undefined;
    for (first_formats, 0..) |format, index| {
        records[index] = specManifestRecord(specFor(format).?);
    }
    return std.json.Stringify.valueAlloc(allocator, SpecManifest{
        .schema = "antfly.quant_kernel_specs.v1",
        .format_count = first_formats.len,
        .specs = &records,
    }, .{ .whitespace = .indent_2 });
}

pub fn conformanceManifestJson(allocator: std.mem.Allocator) ![]u8 {
    var records: [first_conformance.len]ConformanceManifestRecord = undefined;
    for (first_conformance, 0..) |case, index| {
        records[index] = conformanceManifestRecord(case);
    }
    return std.json.Stringify.valueAlloc(allocator, ConformanceManifest{
        .schema = "antfly.quant_kernel_conformance.v1",
        .case_count = first_conformance.len,
        .format_count = first_formats.len,
        .row_bucket_count = first_row_buckets.len,
        .epilogue_count = coverage_epilogues.len,
        .backend_count = first_backends.len,
        .cuda_route_summary = routeSummaryForBackend(.cuda),
        .metal_route_summary = routeSummaryForBackend(.metal),
        .cases = &records,
    }, .{ .whitespace = .indent_2 });
}

fn metalProductionBenchmarkCaseCount() comptime_int {
    @setEvalBranchQuota(10_000);
    var count: comptime_int = 0;
    inline for (first_generated_artifacts) |artifact| {
        if (artifact.backend == .metal and artifactHasPromotionEvidence(artifact)) count += 2;
    }
    return count;
}

fn buildMetalProductionBenchmarkCases() [metalProductionBenchmarkCaseCount()]MetalProductionBenchmarkCase {
    @setEvalBranchQuota(10_000);
    var cases: [metalProductionBenchmarkCaseCount()]MetalProductionBenchmarkCase = undefined;
    var index: usize = 0;
    inline for (first_generated_artifacts) |artifact| {
        if (artifact.backend == .metal and artifactHasPromotionEvidence(artifact)) {
            cases[index] = metalProductionBenchmarkCaseForArtifactShape(artifact, .small);
            index += 1;
            cases[index] = metalProductionBenchmarkCaseForArtifactShape(artifact, .wide);
            index += 1;
        }
    }
    return cases;
}

pub fn metalProductionBenchmarkCaseManifestFingerprint() u64 {
    var hasher = std.hash.Wyhash.init(0);
    std.hash.autoHash(&hasher, first_metal_production_benchmark_case_count);
    for (first_metal_production_benchmark_cases) |case| {
        hasher.update(case.name);
        hasher.update(case.kernel_id);
        std.hash.autoHash(&hasher, case.format);
        std.hash.autoHash(&hasher, case.row_bucket);
        std.hash.autoHash(&hasher, case.epilogue);
        std.hash.autoHash(&hasher, case.dispatch);
        std.hash.autoHash(&hasher, case.shape);
        std.hash.autoHash(&hasher, case.rows);
        std.hash.autoHash(&hasher, case.in_dim);
        std.hash.autoHash(&hasher, case.out_dim);
        std.hash.autoHash(&hasher, case.threads_per_threadgroup);
        std.hash.autoHash(&hasher, case.cols_per_threadgroup);
        std.hash.autoHash(&hasher, @as(u32, @bitCast(case.tolerance_abs)));
        hasher.update(case.generated_source_path);
        std.hash.autoHash(&hasher, case.generated_source_fingerprint);
        hasher.update(case.check_command);
        hasher.update(case.production_kernel_id);
        hasher.update(case.benchmark_command);
    }
    return hasher.final();
}

fn metalProductionBenchmarkCaseForArtifactShape(comptime artifact: GeneratedArtifact, comptime shape: MetalBenchmarkShape) MetalProductionBenchmarkCase {
    const dims = metalBenchmarkDimsForArtifact(artifact, shape);
    return .{
        .name = metalBenchmarkCaseName(artifact, shape),
        .kernel_id = artifact.kernel_id,
        .format = artifact.format,
        .row_bucket = artifact.row_bucket,
        .epilogue = artifact.epilogue,
        .dispatch = dispatchForRowBucket(artifact.row_bucket) orelse .scalar,
        .shape = shape,
        .rows = dims.rows,
        .in_dim = dims.in_dim,
        .out_dim = dims.out_dim,
        .threads_per_threadgroup = metalGeneratedThreadsPerThreadgroup(artifact.format, artifact.row_bucket, artifact.epilogue),
        .cols_per_threadgroup = metalGeneratedColsPerThreadgroup(artifact.format, artifact.row_bucket, artifact.epilogue),
        .tolerance_abs = dims.tolerance_abs,
        .generated_source_path = generatedSourcePathForArtifact(artifact),
        .generated_source_fingerprint = artifactSourceFingerprint(artifact),
        .check_command = generatedCheckCommandForArtifact(artifact),
        .production_kernel_id = artifact.kernel_id,
        .benchmark_command = first_metal_production_regression_evidence_command,
    };
}

pub fn metalBenchmarkCaseName(comptime artifact: GeneratedArtifact, comptime shape: MetalBenchmarkShape) []const u8 {
    return switch (shape) {
        .small => std.fmt.comptimePrint("{s}_rows_2_8_{s}", .{ @tagName(artifact.format), @tagName(artifact.epilogue) }),
        .wide => std.fmt.comptimePrint("{s}_rows_8_cols_7_{s}", .{ @tagName(artifact.format), @tagName(artifact.epilogue) }),
    };
}

pub fn metalBenchmarkDimsForArtifact(comptime artifact: GeneratedArtifact, comptime shape: MetalBenchmarkShape) MetalBenchmarkDims {
    return switch (shape) {
        .small => .{
            .rows = if (artifact.format == .q4_k and artifact.epilogue == .bias_gelu) 4 else 3,
            .in_dim = 512,
            .out_dim = if (artifact.format == .q4_k and artifact.epilogue == .bias_gelu) 3 else 2,
            .tolerance_abs = metalBenchmarkTolerance(artifact.format, shape),
        },
        .wide => .{
            .rows = 8,
            .in_dim = 768,
            .out_dim = 7,
            .tolerance_abs = metalBenchmarkTolerance(artifact.format, shape),
        },
    };
}

pub fn metalBenchmarkTolerance(comptime format: quant_matmul.Format, comptime shape: MetalBenchmarkShape) f32 {
    const loose = format == .q2_k or format == .q3_k or format == .q8_k;
    return switch (shape) {
        .small => if (loose) 0.0005 else 0.0002,
        .wide => if (loose) 0.001 else 0.0005,
    };
}

fn buildDescriptorFormats() [codec_format_coverage.len]quant_matmul.Format {
    var formats: [codec_format_coverage.len]quant_matmul.Format = undefined;
    for (codec_format_coverage, 0..) |entry, index| {
        formats[index] = entry.format;
    }
    return formats;
}

fn buildFirstCoverage() [first_formats.len * first_row_buckets.len * coverage_epilogues.len]CoverageCase {
    @setEvalBranchQuota(4096);
    var cases: [first_formats.len * first_row_buckets.len * coverage_epilogues.len]CoverageCase = undefined;
    var index: usize = 0;
    for (first_formats) |format| {
        for (first_row_buckets) |row_bucket| {
            for (coverage_epilogues) |epilogue| {
                cases[index] = .{ .format = format, .row_bucket = row_bucket, .epilogue = epilogue };
                index += 1;
            }
        }
    }
    return cases;
}

fn buildFirstRegistry() [first_coverage.len * first_backends.len]QuantKernelLowering {
    @setEvalBranchQuota(131072);
    var entries: [first_coverage.len * first_backends.len]QuantKernelLowering = undefined;
    var index: usize = 0;
    for (first_coverage) |case| {
        for (first_backends) |backend| {
            entries[index] = loweringFor(backend, case.format, case.row_bucket, case.epilogue);
            index += 1;
        }
    }
    return entries;
}

fn routeSummaryForBackend(backend: Backend) PlanCounters {
    var total = PlanCounters{};
    for (first_registry.entries) |entry| {
        if (entry.backend != backend) continue;
        addCountersToStats(&total, countersForLowering(entry));
    }
    return total;
}

fn benchmarkManifestRecord(bench: BenchmarkCase) BenchmarkManifestRecord {
    return .{
        .name = bench.name,
        .backend = @tagName(bench.backend),
        .format = @tagName(bench.format),
        .row_bucket = @tagName(bench.row_bucket),
        .epilogue = @tagName(bench.epilogue),
        .dispatch = @tagName(dispatchForRowBucket(bench.row_bucket) orelse .scalar),
        .generated_kernel_id = bench.generated_kernel_id,
        .generated_source_path = bench.generated_source_path,
        .generated_source_fingerprint = bench.generated_source_fingerprint,
        .generated_ptx_path = bench.generated_ptx_path,
        .generated_ptx_command = bench.generated_ptx_command,
        .benchmark_command = bench.benchmark_command,
        .generated_ptx_arg = bench.generated_ptx_arg,
        .handwritten_baseline = bench.handwritten_baseline,
        .correctness_tolerance_abs = bench.correctness_tolerance_abs,
        .minimum_speedup = bench.minimum_speedup,
        .measured_speedup = benchmarkMeasuredSpeedup(bench),
        .correctness_evidence_path = bench.correctness_evidence_path,
        .benchmark_evidence_path = bench.benchmark_evidence_path,
        .benchmark_mode = bench.benchmark_mode,
        .production_enabled = bench.production_enabled,
        .promotion_ready = benchmarkHasPromotionEvidence(bench),
        .promotion_blocker = benchmarkPromotionBlocker(bench),
    };
}

fn metalProductionBenchmarkManifestRecord(case: MetalProductionBenchmarkCase) MetalProductionBenchmarkManifestRecord {
    return .{
        .name = case.name,
        .kernel_id = case.kernel_id,
        .format = @tagName(case.format),
        .row_bucket = @tagName(case.row_bucket),
        .epilogue = @tagName(case.epilogue),
        .dispatch = @tagName(case.dispatch),
        .shape = @tagName(case.shape),
        .rows = case.rows,
        .in_dim = case.in_dim,
        .out_dim = case.out_dim,
        .threads_per_threadgroup = case.threads_per_threadgroup,
        .cols_per_threadgroup = case.cols_per_threadgroup,
        .tolerance_abs = case.tolerance_abs,
        .generated_source_path = case.generated_source_path,
        .generated_source_fingerprint = case.generated_source_fingerprint,
        .check_command = case.check_command,
        .production_kernel_id = case.production_kernel_id,
        .benchmark_command = case.benchmark_command,
    };
}

fn artifactManifestRecord(artifact: GeneratedArtifact, runtime_route_evidence_command: []const u8, promotion_blocker_check_command: []const u8) ArtifactManifestRecord {
    return .{
        .backend = @tagName(artifact.backend),
        .format = @tagName(artifact.format),
        .row_bucket = @tagName(artifact.row_bucket),
        .epilogue = @tagName(artifact.epilogue),
        .kernel_id = artifact.kernel_id,
        .source_path = artifact.source_path,
        .generated_source_path = generatedSourcePathForArtifact(artifact),
        .artifact_source_path = if (artifact.backend == .metal) metalArtifactSourcePathForKernel(artifact.kernel_id) orelse "" else "",
        .generated_source_fingerprint = artifactSourceFingerprint(artifact),
        .check_command = artifact.check_command,
        .generated_check_command = generatedCheckCommandForArtifact(artifact),
        .runtime_evidence_command = artifact.runtime_evidence_command,
        .runtime_route_evidence_command = runtime_route_evidence_command,
        .promotion_evidence_command = artifact.promotion_evidence_command,
        .promotion_check_command = artifact.promotion_check_command,
        .promotion_policy = artifactPromotionPolicy(artifact),
        .production_enabled = artifact.production_enabled,
        .runtime_wired = artifactRuntimeWired(artifact),
        .runtime_gate_env = artifactRuntimeGateEnvText(artifact),
        .production_regression_checked = artifactProductionRegressionChecked(artifact),
        .production_regression_command = artifactProductionRegressionCommand(artifact),
        .metal_promotion_min_speedup = if (artifact.backend == .metal) metal_promotion_min_speedup else 0.0,
        .metal_promotion_repeat_runs = if (artifact.backend == .metal) metal_promotion_repeat_runs else 0,
        .metal_promotion_warmup_repeat_runs = if (artifact.backend == .metal) metal_promotion_warmup_repeat_runs else 0,
        .candidate_status = artifactCandidateStatus(artifact),
        .promotion_ready = artifactHasPromotionEvidence(artifact),
        .promotion_blocker = artifactPromotionBlocker(artifact),
        .promotion_blocker_evidence_path = artifactPromotionBlockerEvidencePath(artifact),
        .promotion_blocker_check_command = promotion_blocker_check_command,
        .promotion_blocker_requires_production_regression_clear = artifactPromotionBlockerRequiresProductionRegressionClear(artifact),
    };
}

fn artifactRuntimeRouteEvidenceCommand(allocator: std.mem.Allocator, artifact: GeneratedArtifact) ![]const u8 {
    if (!artifactNeedsRuntimeRouteEvidence(artifact)) return "";
    return std.fmt.allocPrint(
        allocator,
        "zig build quant-kernel-metal-runtime-check -Dmetal=true -Dcuda=false -- --evidence-out /private/tmp/antfly-quant-metal-{s}-runtime-route-evidence.json --runtime-route-kernel {s}",
        .{ artifact.kernel_id, artifact.kernel_id },
    );
}

pub fn artifactNeedsRuntimeRouteEvidence(artifact: GeneratedArtifact) bool {
    return artifact.backend == .metal and artifactRuntimeWired(artifact) and !artifactHasPromotionEvidence(artifact);
}

fn artifactPromotionPolicy(artifact: GeneratedArtifact) []const u8 {
    if (artifact.backend == .metal and std.mem.eql(u8, artifactPromotionBlocker(artifact), metal_blocker_unsupported_handwritten)) {
        return "route_evidence_only_no_promotion";
    }
    if (artifact.backend == .metal and artifactHasPromotionEvidence(artifact)) {
        return "promoted_speedup_vs_handwritten";
    }
    if (artifact.backend == .metal and artifactRuntimeWired(artifact)) {
        return "speedup_vs_handwritten";
    }
    if (artifact.backend == .metal) {
        return "production_disabled";
    }
    if (artifact.backend == .cuda) {
        return "driver_artifact_policy";
    }
    return "unsupported";
}

fn artifactPromotionBlockerCheckCommand(allocator: std.mem.Allocator, artifact: GeneratedArtifact) ![]const u8 {
    const path = artifactPromotionBlockerEvidencePath(artifact);
    if (path.len == 0) return "";
    return std.fmt.allocPrint(
        allocator,
        "zig build quant-kernel-metal-runtime-check -Dmetal=true -Dcuda=false -- --check-evidence {s} --require-evidence-kernel {s}",
        .{ path, artifact.kernel_id },
    );
}

pub fn artifactRuntimeWired(artifact: GeneratedArtifact) bool {
    if (artifact.backend != .metal or artifact.row_bucket != .rows_2_8) return false;
    if (artifact.format == .q4_0 and artifact.epilogue == .none) return true;
    if (artifact.format == .q4_1 and artifact.epilogue == .none) return true;
    if ((artifact.format == .q5_0 or artifact.format == .q5_1) and artifact.epilogue == .none) return true;
    if ((artifact.format == .q8_1 or artifact.format == .q8_k) and artifact.epilogue == .none) return true;
    if (artifact.format == .q2_k and (artifact.epilogue == .bias or artifact.epilogue == .bias_gelu)) return true;
    if (artifact.format == .q3_k and (artifact.epilogue == .bias or artifact.epilogue == .bias_gelu)) return true;
    if (artifact.format == .q4_k and artifact.epilogue == .bias) return true;
    if (artifact.format == .q4_k and artifact.epilogue == .bias_gelu) return true;
    if (artifact.format == .q8_0 and (artifact.epilogue == .bias or artifact.epilogue == .bias_gelu or artifact.epilogue == .relu)) return true;
    if (artifact.format == .q5_k and (artifact.epilogue == .bias or artifact.epilogue == .bias_gelu)) return true;
    if (artifact.format == .q6_k and (artifact.epilogue == .bias or artifact.epilogue == .bias_gelu)) return true;
    if (artifact.epilogue != .none) return false;
    return switch (artifact.format) {
        .q8_0, .q2_k, .q3_k, .q4_k, .q5_k, .q6_k => true,
        else => false,
    };
}

fn metalRuntimeRouteAllExpectedCaseCount() usize {
    var count: usize = 0;
    for (first_generated_artifacts) |artifact| {
        if (artifactRuntimeWired(artifact)) count += 2;
    }
    return count;
}

fn metalRuntimeRouteAllExpectedProviderRouteCount() usize {
    var count: usize = 0;
    for (first_generated_artifacts) |artifact| {
        if (artifactHasMetalProviderRouteEvidence(artifact)) count += 2;
    }
    return count;
}

fn metalPromotionBlockerSkippedNoPathCount() usize {
    return first_metal_promotion_blocker_evidence_count - metalPromotionBlockerEvidencePathCount();
}

fn metalPromotionBlockerEvidenceExpectedCaseCount() usize {
    return metalPromotionBlockerEvidencePathCount() * first_metal_promotion_blocker_evidence_cases_per_kernel;
}

fn metalProductionRegressionExpectedKernelCount() usize {
    var count: usize = 0;
    for (first_generated_artifacts) |artifact| {
        if (artifactProductionRegressionChecked(artifact)) count += 1;
    }
    return count;
}

fn metalProductionRegressionExpectedCaseCount() usize {
    return metalProductionRegressionExpectedKernelCount() * 2;
}

fn artifactRuntimeGateEnvText(artifact: GeneratedArtifact) []const u8 {
    return if (artifactRuntimeGateEnv(artifact)) |env| std.mem.span(env) else "";
}

fn artifactCandidateOptInGateEnv(artifact: GeneratedArtifact) ?[*:0]const u8 {
    if (!artifactRuntimeWired(artifact)) return null;
    return switch (artifact.format) {
        .q8_0 => switch (artifact.epilogue) {
            .bias => "TERMITE_METAL_ENABLE_ANTFLY_Q8_0_SMALL_BATCH_BIAS",
            .bias_gelu => "TERMITE_METAL_ENABLE_ANTFLY_Q8_0_SMALL_BATCH_BIAS_GELU",
            .relu => "TERMITE_METAL_ENABLE_ANTFLY_Q8_0_SMALL_BATCH_RELU",
            else => "TERMITE_METAL_ENABLE_ANTFLY_Q8_0_SMALL_BATCH",
        },
        .q2_k => switch (artifact.epilogue) {
            .bias => "TERMITE_METAL_ENABLE_ANTFLY_Q2_K_SMALL_BATCH_BIAS",
            .bias_gelu => "TERMITE_METAL_ENABLE_ANTFLY_Q2_K_SMALL_BATCH_BIAS_GELU",
            else => "TERMITE_METAL_ENABLE_ANTFLY_Q2_K_SMALL_BATCH",
        },
        .q3_k => switch (artifact.epilogue) {
            .bias => "TERMITE_METAL_ENABLE_ANTFLY_Q3_K_SMALL_BATCH_BIAS",
            .bias_gelu => "TERMITE_METAL_ENABLE_ANTFLY_Q3_K_SMALL_BATCH_BIAS_GELU",
            else => "TERMITE_METAL_ENABLE_ANTFLY_Q3_K_SMALL_BATCH",
        },
        .q4_0 => "TERMITE_METAL_ENABLE_ANTFLY_Q4_0_SMALL_BATCH",
        .q4_1 => "TERMITE_METAL_ENABLE_ANTFLY_Q4_1_SMALL_BATCH",
        .q5_0 => "TERMITE_METAL_ENABLE_ANTFLY_Q5_0_SMALL_BATCH",
        .q5_1 => "TERMITE_METAL_ENABLE_ANTFLY_Q5_1_SMALL_BATCH",
        .q8_1 => "TERMITE_METAL_ENABLE_ANTFLY_Q8_1_SMALL_BATCH",
        .q8_k => "TERMITE_METAL_ENABLE_ANTFLY_Q8_K_SMALL_BATCH",
        .q4_k => switch (artifact.epilogue) {
            .bias => "TERMITE_METAL_ENABLE_ANTFLY_Q4_K_SMALL_BATCH_BIAS",
            .bias_gelu => "TERMITE_METAL_ENABLE_ANTFLY_Q4_K_SMALL_BATCH_BIAS_GELU",
            else => "TERMITE_METAL_ENABLE_ANTFLY_Q4_K_SMALL_BATCH",
        },
        .q5_k => switch (artifact.epilogue) {
            .bias => "TERMITE_METAL_ENABLE_ANTFLY_Q5_K_SMALL_BATCH_BIAS",
            .bias_gelu => "TERMITE_METAL_ENABLE_ANTFLY_Q5_K_SMALL_BATCH_BIAS_GELU",
            else => "TERMITE_METAL_ENABLE_ANTFLY_Q5_K_SMALL_BATCH",
        },
        .q6_k => switch (artifact.epilogue) {
            .bias => "TERMITE_METAL_ENABLE_ANTFLY_Q6_K_SMALL_BATCH_BIAS",
            .bias_gelu => "TERMITE_METAL_ENABLE_ANTFLY_Q6_K_SMALL_BATCH_BIAS_GELU",
            else => "TERMITE_METAL_ENABLE_ANTFLY_Q6_K_SMALL_BATCH",
        },
        else => null,
    };
}

fn artifactProductionRegressionChecked(artifact: GeneratedArtifact) bool {
    return artifact.backend == .metal and artifactHasPromotionEvidence(artifact);
}

fn artifactProductionRegressionCommand(artifact: GeneratedArtifact) []const u8 {
    return if (artifactProductionRegressionChecked(artifact)) first_metal_production_regression_build_command else "";
}

pub fn artifactRuntimeGateEnv(artifact: GeneratedArtifact) ?[*:0]const u8 {
    if (!artifactRuntimeWired(artifact)) return null;
    const promotion_ready = artifactHasPromotionEvidence(artifact);
    return switch (artifact.format) {
        .q8_0 => switch (artifact.epilogue) {
            .bias, .bias_gelu => if (promotion_ready) null else artifactCandidateOptInGateEnv(artifact),
            .relu => artifactCandidateOptInGateEnv(artifact),
            else => if (promotion_ready) "TERMITE_METAL_DISABLE_ANTFLY_Q8_0_SMALL_BATCH" else artifactCandidateOptInGateEnv(artifact),
        },
        .q2_k => switch (artifact.epilogue) {
            .bias, .bias_gelu => artifactCandidateOptInGateEnv(artifact),
            else => if (promotion_ready) "TERMITE_METAL_DISABLE_ANTFLY_Q2_K_SMALL_BATCH" else artifactCandidateOptInGateEnv(artifact),
        },
        .q3_k => switch (artifact.epilogue) {
            .bias, .bias_gelu => artifactCandidateOptInGateEnv(artifact),
            else => if (promotion_ready) "TERMITE_METAL_DISABLE_ANTFLY_Q3_K_SMALL_BATCH" else artifactCandidateOptInGateEnv(artifact),
        },
        .q4_0 => artifactCandidateOptInGateEnv(artifact),
        .q4_1 => if (promotion_ready) "TERMITE_METAL_DISABLE_ANTFLY_Q4_1_SMALL_BATCH" else artifactCandidateOptInGateEnv(artifact),
        .q5_0 => if (promotion_ready) "TERMITE_METAL_DISABLE_ANTFLY_Q5_0_SMALL_BATCH" else artifactCandidateOptInGateEnv(artifact),
        .q8_1 => if (promotion_ready) "TERMITE_METAL_DISABLE_ANTFLY_Q8_1_SMALL_BATCH" else artifactCandidateOptInGateEnv(artifact),
        .q8_k => if (promotion_ready) "TERMITE_METAL_DISABLE_ANTFLY_Q8_K_SMALL_BATCH" else artifactCandidateOptInGateEnv(artifact),
        .q5_1 => if (promotion_ready) "TERMITE_METAL_DISABLE_ANTFLY_Q5_1_SMALL_BATCH" else artifactCandidateOptInGateEnv(artifact),
        .q4_k => switch (artifact.epilogue) {
            .bias, .bias_gelu => if (promotion_ready) null else artifactCandidateOptInGateEnv(artifact),
            else => if (promotion_ready) "TERMITE_METAL_DISABLE_ANTFLY_Q4_K_SMALL_BATCH" else artifactCandidateOptInGateEnv(artifact),
        },
        .q5_k => switch (artifact.epilogue) {
            .bias, .bias_gelu => if (promotion_ready) null else artifactCandidateOptInGateEnv(artifact),
            else => if (promotion_ready) "TERMITE_METAL_DISABLE_ANTFLY_Q5_K_SMALL_BATCH" else artifactCandidateOptInGateEnv(artifact),
        },
        .q6_k => switch (artifact.epilogue) {
            .bias, .bias_gelu => if (promotion_ready) null else artifactCandidateOptInGateEnv(artifact),
            else => if (promotion_ready) "TERMITE_METAL_DISABLE_ANTFLY_Q6_K_SMALL_BATCH" else artifactCandidateOptInGateEnv(artifact),
        },
        else => null,
    };
}

fn artifactCandidateStatus(artifact: GeneratedArtifact) []const u8 {
    if (artifactHasPromotionEvidence(artifact)) return "promoted";
    if (metalPromotionBlockerEvidenceFor(artifact) != null) return "blocked_by_evidence";
    if (artifact.production_enabled) return "promotion_blocked";
    return "dev_only_candidate";
}

fn artifactSourceFingerprint(artifact: GeneratedArtifact) u64 {
    @setEvalBranchQuota(24_000_000);
    const source = generatedSourceForArtifact(artifact) orelse return 0;
    return std.hash.Wyhash.hash(0, source);
}

fn candidateSourceFingerprint(lowering: QuantKernelLowering) u64 {
    if (lowering.candidate_route != .generated_dev_candidate) return 0;
    for (first_generated_artifacts) |artifact| {
        if (artifact.backend == lowering.backend and
            artifact.format == lowering.format and
            artifact.row_bucket == lowering.row_bucket and
            artifact.epilogue == lowering.epilogue and
            std.mem.eql(u8, artifact.kernel_id, lowering.kernel_id) and
            std.mem.eql(u8, artifact.source_path, lowering.candidate_source_path))
        {
            return artifactSourceFingerprint(artifact);
        }
    }
    return 0;
}

fn specManifestRecord(spec: QuantKernelSpec) SpecManifestRecord {
    return .{
        .format = @tagName(spec.format),
        .reference_tensor_type = tensorTypeName(tensorTypeForFormat(spec.format).?),
        .block_values = spec.block_values,
        .block_bytes = spec.block_bytes,
        .block_fields = spec.block_fields,
        .decode_ops = spec.decode_ops,
        .supported_schedules = spec.supported_schedules,
        .supported_epilogues = spec.supported_epilogues,
        .supported_backends = spec.supported_backends,
        .accumulator_dtype = @tagName(spec.accumulator_dtype),
        .output_dtype = @tagName(spec.output_dtype),
    };
}

fn conformanceManifestRecord(case: ConformanceCase) ConformanceManifestRecord {
    const spec = specFor(case.format).?;
    const cuda = loweringFor(.cuda, case.format, case.row_bucket, case.epilogue);
    const metal = loweringFor(.metal, case.format, case.row_bucket, case.epilogue);
    const schedule = cuda.schedule;
    const cuda_candidate_schedule = candidateScheduleFor(cuda);
    const metal_candidate_schedule = candidateScheduleFor(metal);
    return .{
        .format = @tagName(case.format),
        .block_values = spec.block_values,
        .block_bytes = spec.block_bytes,
        .accumulator_dtype = @tagName(spec.accumulator_dtype),
        .output_dtype = @tagName(spec.output_dtype),
        .row_bucket = @tagName(case.row_bucket),
        .epilogue = @tagName(case.epilogue),
        .dispatch = @tagName(case.dispatch),
        .tile_rows = schedule.tile_rows,
        .tile_cols = schedule.tile_cols,
        .vector_width = schedule.vector_width,
        .threads_per_block = schedule.threads_per_block,
        .shared_memory_bytes = schedule.shared_memory_bytes,
        .register_pressure_hint = schedule.register_pressure_hint,
        .tensor_core_eligible = schedule.tensor_core_eligible,
        .cuda_candidate_tile_rows = cuda_candidate_schedule.tile_rows,
        .cuda_candidate_tile_cols = cuda_candidate_schedule.tile_cols,
        .cuda_candidate_threads_per_block = cuda_candidate_schedule.threads_per_block,
        .metal_candidate_tile_rows = metal_candidate_schedule.tile_rows,
        .metal_candidate_tile_cols = metal_candidate_schedule.tile_cols,
        .metal_candidate_threads_per_block = metal_candidate_schedule.threads_per_block,
        .reference_supported = case.reference_supported,
        .reference_tensor_type = tensorTypeName(case.reference_tensor_type),
        .tolerance_abs = case.tolerance_abs,
        .cuda_route = loweringRouteName(case.cuda_route),
        .cuda_candidate_route = loweringRouteName(case.cuda_candidate_route),
        .cuda_production_kernel_id = cuda.production_kernel_id,
        .cuda_candidate_kernel_id = cuda.kernel_id,
        .cuda_candidate_source_path = cuda.candidate_source_path,
        .cuda_candidate_source_fingerprint = candidateSourceFingerprint(cuda),
        .cuda_fallback_reason = fallbackReasonName(case.cuda_fallback_reason),
        .metal_route = loweringRouteName(case.metal_route),
        .metal_candidate_route = loweringRouteName(case.metal_candidate_route),
        .metal_production_kernel_id = metal.production_kernel_id,
        .metal_candidate_kernel_id = metal.kernel_id,
        .metal_candidate_source_path = metal.candidate_source_path,
        .metal_candidate_source_fingerprint = candidateSourceFingerprint(metal),
        .metal_fallback_reason = fallbackReasonName(case.metal_fallback_reason),
    };
}

fn buildFirstConformance() [first_coverage.len]ConformanceCase {
    @setEvalBranchQuota(131072);
    var cases: [first_coverage.len]ConformanceCase = undefined;
    for (first_coverage, 0..) |case, index| {
        const dispatch = dispatchForRowBucket(case.row_bucket) orelse .scalar;
        const cuda = loweringFor(.cuda, case.format, case.row_bucket, case.epilogue);
        const metal = loweringFor(.metal, case.format, case.row_bucket, case.epilogue);
        cases[index] = .{
            .format = case.format,
            .row_bucket = case.row_bucket,
            .epilogue = case.epilogue,
            .dispatch = dispatch,
            .reference_supported = referenceSupportedForEpilogue(case.epilogue),
            .reference_tensor_type = tensorTypeForFormat(case.format).?,
            .tolerance_abs = toleranceFor(case.format, case.epilogue),
            .cuda_route = cuda.production_route,
            .cuda_candidate_route = cuda.candidate_route,
            .cuda_fallback_reason = cuda.fallback_reason,
            .metal_route = metal.production_route,
            .metal_candidate_route = metal.candidate_route,
            .metal_fallback_reason = metal.fallback_reason,
        };
    }
    return cases;
}

pub fn generatedArtifactForCandidate(
    backend: Backend,
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
) ?GeneratedArtifact {
    for (first_generated_artifacts) |artifact| {
        if (artifact.backend == backend and
            artifact.format == format and
            artifact.row_bucket == row_bucket and
            artifact.epilogue == epilogue)
        {
            return artifact;
        }
    }
    return null;
}

pub fn generatedArtifactForKernel(backend: Backend, kernel_id: []const u8) ?GeneratedArtifact {
    for (first_generated_artifacts) |artifact| {
        if (artifact.backend == backend and std.mem.eql(u8, artifact.kernel_id, kernel_id)) {
            return artifact;
        }
    }
    return null;
}

fn generatedSourcePathForArtifact(artifact: GeneratedArtifact) []const u8 {
    return if (artifact.generated_source_path.len != 0) artifact.generated_source_path else artifact.source_path;
}

fn generatedCheckCommandForArtifact(artifact: GeneratedArtifact) []const u8 {
    return if (artifact.generated_check_command.len != 0) artifact.generated_check_command else artifact.check_command;
}

pub fn loweringFor(
    backend: Backend,
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
) QuantKernelLowering {
    const spec = specFor(format) orelse return unsupportedLowering(backend, format, row_bucket, epilogue, .scalar, .unsupported_format);

    const dispatch = dispatchForRowBucket(row_bucket) orelse return unsupportedLowering(backend, format, row_bucket, epilogue, .scalar, .unsupported_shape);

    if (!spec.supportsSchedule(dispatch)) return unsupportedLowering(backend, format, row_bucket, epilogue, dispatch, .unsupported_shape);

    if (!spec.supportsBackend(backend)) return .{
        .plan_id = planId(backend, format, row_bucket, epilogue, dispatch),
        .backend = backend,
        .format = format,
        .row_bucket = row_bucket,
        .epilogue = epilogue,
        .schedule = scheduleFor(row_bucket, dispatch),
        .production_route = .unsupported,
        .candidate_route = .unsupported,
        .production_kernel_id = "",
        .fallback_reason = .unsupported_backend,
        .kernel_id = "",
        .candidate_source_path = "",
    };

    if (!supportsEpilogueForBackend(spec, backend, epilogue)) return .{
        .plan_id = planId(backend, format, row_bucket, epilogue, dispatch),
        .backend = backend,
        .format = format,
        .row_bucket = row_bucket,
        .epilogue = epilogue,
        .schedule = scheduleFor(row_bucket, dispatch),
        .production_route = .unsupported,
        .candidate_route = .unsupported,
        .production_kernel_id = "",
        .fallback_reason = .unsupported_epilogue,
        .kernel_id = "",
        .candidate_source_path = "",
    };

    const candidate_artifact = generatedArtifactForCandidate(backend, format, row_bucket, epilogue);
    const generated_candidate = candidate_artifact != null;
    const kernel_id = if (candidate_artifact) |artifact| artifact.kernel_id else "";
    const source_path = if (candidate_artifact) |artifact| artifact.source_path else "";

    return promoteLoweringIfArtifactReady(.{
        .plan_id = planId(backend, format, row_bucket, epilogue, dispatch),
        .backend = backend,
        .format = format,
        .row_bucket = row_bucket,
        .epilogue = epilogue,
        .schedule = scheduleFor(row_bucket, dispatch),
        .production_route = .handwritten_production,
        .candidate_route = if (generated_candidate) .generated_dev_candidate else .unsupported,
        .production_kernel_id = productionKernelId(backend, format, row_bucket, epilogue),
        .fallback_reason = if (generated_candidate) generatedCandidateFallbackReason(backend, kernel_id) else .none,
        .kernel_id = if (generated_candidate) kernel_id else "",
        .candidate_source_path = if (generated_candidate) source_path else "",
    });
}

fn generatedCandidateFallbackReason(backend: Backend, kernel_id: []const u8) FallbackReason {
    if (backend == .metal and checkedInMetalEvidenceForKernel(kernel_id) != null) {
        return .generated_runtime_not_wired;
    }
    return .generated_artifact_missing;
}

fn unsupportedLowering(
    backend: Backend,
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
    dispatch: quant_matmul.DispatchKind,
    reason: FallbackReason,
) QuantKernelLowering {
    return .{
        .plan_id = planId(backend, format, row_bucket, epilogue, dispatch),
        .backend = backend,
        .format = format,
        .row_bucket = row_bucket,
        .epilogue = epilogue,
        .schedule = unsupportedSchedule(row_bucket, dispatch),
        .production_route = .unsupported,
        .candidate_route = .unsupported,
        .production_kernel_id = "",
        .fallback_reason = reason,
        .kernel_id = "",
        .candidate_source_path = "",
    };
}

fn unsupportedSchedule(row_bucket: quant_matmul.RowBucket, dispatch: quant_matmul.DispatchKind) QuantKernelSchedule {
    return .{
        .dispatch = dispatch,
        .row_bucket = row_bucket,
        .tile_rows = 0,
        .tile_cols = 0,
        .vector_width = 0,
        .threads_per_block = 0,
        .shared_memory_bytes = 0,
        .register_pressure_hint = 0,
        .tensor_core_eligible = false,
    };
}

fn planId(
    backend: Backend,
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
    dispatch: quant_matmul.DispatchKind,
) QuantKernelPlanId {
    return .{
        .backend = backend,
        .format = format,
        .row_bucket = row_bucket,
        .epilogue = epilogue,
        .dispatch = dispatch,
    };
}

pub fn planIdName(allocator: std.mem.Allocator, id: QuantKernelPlanId) ![]u8 {
    return std.fmt.allocPrint(
        allocator,
        "{s}/{s}/{s}/{s}/{s}",
        .{
            @tagName(id.backend),
            @tagName(id.format),
            @tagName(id.row_bucket),
            @tagName(id.epilogue),
            @tagName(id.dispatch),
        },
    );
}

pub fn loweringRouteName(route: LoweringRoute) []const u8 {
    return switch (route) {
        .generated_production => "generated_production",
        .generated_dev_candidate => "generated_dev_candidate",
        .handwritten_production => "handwritten_production",
        .unsupported => "unsupported",
    };
}

pub fn fallbackReasonName(reason: FallbackReason) []const u8 {
    return switch (reason) {
        .none => "none",
        .unsupported_format => "unsupported_format",
        .unsupported_shape => "unsupported_shape",
        .unsupported_epilogue => "unsupported_epilogue",
        .unsupported_backend => "unsupported_backend",
        .generated_artifact_missing => "generated_artifact_missing",
        .generated_runtime_not_wired => "generated_runtime_not_wired",
        .tensor_core_repack_required => "tensor_core_repack_required",
    };
}

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

pub fn metalGeneratedCounterNameForArtifact(artifact: GeneratedArtifact) ?[]const u8 {
    if (artifact.backend != .metal or !artifactRuntimeWired(artifact)) return null;
    return switch (artifact.format) {
        .q8_0 => switch (artifact.epilogue) {
            .none => "q8_0_small_batch",
            .bias => "q8_0_small_batch_bias",
            .bias_gelu => "q8_0_small_batch_bias_gelu",
            .relu => "q8_0_small_batch_relu",
            else => null,
        },
        .q8_1 => if (artifact.epilogue == .none) "q8_1_small_batch" else null,
        .q8_k => if (artifact.epilogue == .none) "q8_k_small_batch" else null,
        .q2_k => switch (artifact.epilogue) {
            .none => "q2_k_small_batch",
            .bias => "q2_k_small_batch_bias",
            .bias_gelu => "q2_k_small_batch_bias_gelu",
            else => null,
        },
        .q3_k => switch (artifact.epilogue) {
            .none => "q3_k_small_batch",
            .bias => "q3_k_small_batch_bias",
            .bias_gelu => "q3_k_small_batch_bias_gelu",
            else => null,
        },
        .q4_0 => if (artifact.epilogue == .none) "q4_0_small_batch" else null,
        .q4_1 => if (artifact.epilogue == .none) "q4_1_small_batch" else null,
        .q5_0 => if (artifact.epilogue == .none) "q5_0_small_batch" else null,
        .q5_1 => if (artifact.epilogue == .none) "q5_1_small_batch" else null,
        .q4_k => switch (artifact.epilogue) {
            .none => "q4_k_small_batch",
            .bias => "q4_k_small_batch_bias",
            .bias_gelu => "q4_k_small_batch_bias_gelu",
            else => null,
        },
        .q5_k => switch (artifact.epilogue) {
            .none => "q5_k_small_batch",
            .bias => "q5_k_small_batch_bias",
            .bias_gelu => "q5_k_small_batch_bias_gelu",
            else => null,
        },
        .q6_k => switch (artifact.epilogue) {
            .none => "q6_k_small_batch",
            .bias => "q6_k_small_batch_bias",
            .bias_gelu => "q6_k_small_batch_bias_gelu",
            else => null,
        },
        else => null,
    };
}

pub fn metalRuntimeRouteSummaryJson(allocator: std.mem.Allocator) ![]u8 {
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try out.append(allocator, '{');
    var first = true;
    for (first_generated_artifacts) |artifact| {
        const counter_name = metalGeneratedCounterNameForArtifact(artifact) orelse continue;
        const lowering = loweringFor(.metal, artifact.format, artifact.row_bucket, artifact.epilogue);
        const plan_name = try planIdName(allocator, lowering.plan_id);
        defer allocator.free(plan_name);
        if (!first) try out.append(allocator, ',');
        first = false;
        const runtime_gate_env = if (artifactRuntimeGateEnv(artifact)) |env| std.mem.span(env) else "";
        const generated_source_path = generatedMetalSourcePathForKernel(artifact.kernel_id) orelse artifact.generated_source_path;
        try appendFmt(
            allocator,
            &out,
            \\{f}:{{"plan_id":{f},"format":{f},"row_bucket":{f},"epilogue":{f},"dispatch":{f},"kernel_id":{f},"production_kernel_id":{f},"production_route":{f},"candidate_route":{f},"fallback_reason":{f},"source_path":{f},"generated_source_path":{f},"runtime_gate_env":{f},"promotion_ready":{}}}
        ,
            .{
                std.json.fmt(counter_name, .{}),
                std.json.fmt(plan_name, .{}),
                std.json.fmt(@tagName(artifact.format), .{}),
                std.json.fmt(@tagName(artifact.row_bucket), .{}),
                std.json.fmt(@tagName(artifact.epilogue), .{}),
                std.json.fmt(@tagName(lowering.schedule.dispatch), .{}),
                std.json.fmt(artifact.kernel_id, .{}),
                std.json.fmt(lowering.production_kernel_id, .{}),
                std.json.fmt(loweringRouteName(lowering.production_route), .{}),
                std.json.fmt(loweringRouteName(lowering.candidate_route), .{}),
                std.json.fmt(fallbackReasonName(lowering.fallback_reason), .{}),
                std.json.fmt(artifact.source_path, .{}),
                std.json.fmt(generated_source_path, .{}),
                std.json.fmt(runtime_gate_env, .{}),
                artifactHasPromotionEvidence(artifact),
            },
        );
    }
    try out.append(allocator, '}');
    return try out.toOwnedSlice(allocator);
}

fn tensorTypeName(tensor_type: tensor_types.TensorType) []const u8 {
    return switch (tensor_type) {
        .known => |known| @tagName(known),
        .bitnet_tl2 => "bitnet_tl2",
        .unknown => "unknown",
    };
}

pub fn loweringDiagnostic(allocator: std.mem.Allocator, lowering: QuantKernelLowering) ![]u8 {
    const plan_name = try planIdName(allocator, lowering.plan_id);
    defer allocator.free(plan_name);
    return std.fmt.allocPrint(
        allocator,
        "plan={s} production={s} candidate={s} fallback={s} production_kernel={s} candidate_kernel={s} candidate_source={s}",
        .{
            plan_name,
            loweringRouteName(lowering.production_route),
            loweringRouteName(lowering.candidate_route),
            fallbackReasonName(lowering.fallback_reason),
            lowering.production_kernel_id,
            lowering.kernel_id,
            lowering.candidate_source_path,
        },
    );
}

pub fn registryLoweringFor(
    backend: Backend,
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
    dispatch: quant_matmul.DispatchKind,
) QuantKernelLowering {
    if (first_registry.lookup(backend, format, row_bucket, epilogue, dispatch)) |lowering| return lowering;
    if (specFor(format) == null) {
        return unsupportedLowering(backend, format, row_bucket, epilogue, dispatch, .unsupported_format);
    }
    const expected_dispatch = dispatchForRowBucket(row_bucket) orelse
        return unsupportedLowering(backend, format, row_bucket, epilogue, dispatch, .unsupported_shape);
    if (expected_dispatch != dispatch) {
        return unsupportedLowering(backend, format, row_bucket, epilogue, dispatch, .unsupported_shape);
    }
    return loweringFor(backend, format, row_bucket, epilogue);
}

pub fn countersForLowering(lowering: QuantKernelLowering) PlanCounters {
    var counters = PlanCounters{ .quant_kernel_planned_ops = 1 };
    switch (lowering.production_route) {
        .generated_production => counters.quant_kernel_generated_production = 1,
        .handwritten_production => counters.quant_kernel_handwritten_production = 1,
        .generated_dev_candidate => unreachable,
        .unsupported => counters.quant_kernel_unsupported_routes = 1,
    }
    if (lowering.candidate_route == .generated_dev_candidate) counters.quant_kernel_generated_candidates = 1;
    switch (lowering.fallback_reason) {
        .generated_artifact_missing => counters.quant_kernel_fallback_generated_artifact_missing = 1,
        .generated_runtime_not_wired => counters.quant_kernel_fallback_generated_runtime_not_wired = 1,
        .unsupported_format => {
            counters.quant_kernel_fallback_unsupported_format = 1;
            counters.quant_kernel_fallback_unsupported = 1;
        },
        .unsupported_shape => {
            counters.quant_kernel_fallback_unsupported_shape = 1;
            counters.quant_kernel_fallback_unsupported = 1;
        },
        .unsupported_epilogue => {
            counters.quant_kernel_fallback_unsupported_epilogue = 1;
            counters.quant_kernel_fallback_unsupported = 1;
        },
        .unsupported_backend => {
            counters.quant_kernel_fallback_unsupported_backend = 1;
            counters.quant_kernel_fallback_unsupported = 1;
        },
        .tensor_core_repack_required => counters.quant_kernel_fallback_tensor_core_repack_required = 1,
        .none => {},
    }
    return counters;
}

pub fn addCountersToStats(stats: anytype, counters: PlanCounters) void {
    inline for (@typeInfo(PlanCounters).@"struct".fields) |field| {
        @field(stats.*, field.name) += @intCast(@field(counters, field.name));
    }
}

fn promotedArtifactFor(lowering: QuantKernelLowering) ?GeneratedArtifact {
    if (lowering.production_route != .generated_production) return null;
    for (first_generated_artifacts) |artifact| {
        if (artifactHasPromotionEvidence(artifact) and
            artifact.backend == lowering.backend and
            artifact.format == lowering.format and
            artifact.row_bucket == lowering.row_bucket and
            artifact.epilogue == lowering.epilogue and
            std.mem.eql(u8, artifact.kernel_id, lowering.production_kernel_id))
        {
            return artifact;
        }
    }
    return null;
}

fn promoteLoweringIfArtifactReady(lowering: QuantKernelLowering) QuantKernelLowering {
    if (lowering.candidate_route != .generated_dev_candidate) return lowering;
    for (first_generated_artifacts) |artifact| {
        if (!artifactHasPromotionEvidence(artifact)) continue;
        if (promotedLoweringForArtifact(lowering, artifact)) |promoted| return promoted;
    }
    return lowering;
}

fn promotedLoweringForArtifact(lowering: QuantKernelLowering, artifact: GeneratedArtifact) ?QuantKernelLowering {
    if (lowering.candidate_route != .generated_dev_candidate) return null;
    if (artifact.backend != lowering.backend or
        artifact.format != lowering.format or
        artifact.row_bucket != lowering.row_bucket or
        artifact.epilogue != lowering.epilogue or
        !std.mem.eql(u8, artifact.kernel_id, lowering.kernel_id) or
        isDevGeneratedSourcePath(artifact.source_path))
    {
        return null;
    }

    var promoted = lowering;
    promoted.production_route = .generated_production;
    promoted.candidate_route = .unsupported;
    promoted.production_kernel_id = artifact.kernel_id;
    promoted.fallback_reason = .none;
    promoted.kernel_id = "";
    promoted.candidate_source_path = "";
    return promoted;
}

pub fn artifactHasPromotionEvidence(artifact: GeneratedArtifact) bool {
    return std.mem.eql(u8, artifactPromotionBlocker(artifact), metal_blocker_none);
}

fn artifactPromotionBlocker(artifact: GeneratedArtifact) []const u8 {
    if (!artifact.production_enabled) return disabledArtifactPromotionBlocker(artifact);
    if (isDevGeneratedSourcePath(artifact.source_path)) return "dev_generated_source";
    return switch (artifact.backend) {
        .cuda => blk: {
            const bench = benchmarkForArtifact(artifact) orelse return "missing_benchmark_record";
            break :blk benchmarkPromotionBlocker(bench);
        },
        .metal => metalArtifactPromotionBlocker(artifact),
    };
}

fn artifactPromotionBlockerEvidencePath(artifact: GeneratedArtifact) []const u8 {
    if (metalPromotionBlockerEvidenceFor(artifact)) |evidence| return evidence.evidence_path;
    return "";
}

fn artifactPromotionBlockerRequiresProductionRegressionClear(artifact: GeneratedArtifact) bool {
    if (metalPromotionBlockerEvidenceFor(artifact)) |evidence| return evidence.requires_production_regression_clear;
    return false;
}

fn disabledArtifactPromotionBlocker(artifact: GeneratedArtifact) []const u8 {
    if (metalPromotionBlockerEvidenceFor(artifact)) |evidence| return evidence.blocker;
    return if (artifactRuntimeWired(artifact)) "awaiting_metal_promotion_evidence" else "production_disabled";
}

fn metalPromotionBlockerEvidenceFor(artifact: GeneratedArtifact) ?MetalPromotionBlockerEvidence {
    @setEvalBranchQuota(10_000);
    if (artifact.backend != .metal) return null;
    for (first_metal_promotion_blocker_evidence) |evidence| {
        if (std.mem.eql(u8, artifact.kernel_id, evidence.kernel_id)) return evidence;
    }
    return null;
}

fn metalPromotionBlockerEvidenceCount(blocker: []const u8) usize {
    var count: usize = 0;
    for (first_metal_promotion_blocker_evidence) |evidence| {
        if (std.mem.eql(u8, evidence.blocker, blocker)) count += 1;
    }
    return count;
}

fn metalPromotionBlockerEvidencePathCount() usize {
    var count: usize = 0;
    for (first_metal_promotion_blocker_evidence) |evidence| {
        if (evidence.evidence_path.len != 0) count += 1;
    }
    return count;
}

fn metalArtifactPromotionBlocker(artifact: GeneratedArtifact) []const u8 {
    return metalArtifactPromotionBlockerWithEvidence(artifact, &first_metal_runtime_evidence);
}

fn metalArtifactPromotionBlockerWithEvidence(artifact: GeneratedArtifact, evidence_records: []const MetalRuntimeEvidence) []const u8 {
    if (artifact.backend != .metal) return "non_metal_artifact";
    if (!artifact.production_enabled) return disabledArtifactPromotionBlocker(artifact);
    if (!std.mem.endsWith(u8, artifact.source_path, ".metal")) return "missing_metal_source_path";
    if (artifactSourceFingerprint(artifact) == 0) return "missing_source_fingerprint";
    if (!commandFirstTokenEquals(artifact.check_command, "xcrun") or !commandHasToken(artifact.check_command, "metal")) return "missing_xcrun_metal_command";
    if (!commandHasArgValue(artifact.check_command, "-c", artifact.source_path)) return "metal_check_missing_source";
    if (!commandHasArgValue(artifact.check_command, "-o", metalArtifactAirPathForKernel(artifact.kernel_id) orelse return "missing_metal_air_path")) return "metal_check_missing_output";
    if (!commandHasToken(artifact.runtime_evidence_command, "quant-kernel-metal-runtime-check")) return "missing_metal_runtime_evidence_command";
    if (!commandHasToken(artifact.runtime_evidence_command, "--evidence-out")) return "missing_metal_evidence_out_arg";
    if (!commandHasArgValue(artifact.runtime_evidence_command, "--repeat-runs", metal_promotion_repeat_runs_text)) return "missing_metal_repeat_runs";
    if (!metalPromotionEvidenceCommandMatchesPolicy(artifact)) return "metal_promotion_evidence_command_mismatch";
    if (!commandHasToken(artifact.promotion_evidence_command, "quant-kernel-metal-runtime-check")) return "missing_metal_promotion_evidence_command";
    if (!commandHasToken(artifact.promotion_evidence_command, "--promotion-ready-kernel")) return "missing_metal_promotion_ready_kernel";
    if (!commandHasArgValue(artifact.promotion_evidence_command, "--promotion-ready-kernel", artifact.kernel_id)) return "wrong_metal_promotion_ready_kernel";
    if (!commandHasToken(artifact.promotion_check_command, "quant-kernel-metal-runtime-check")) return "missing_metal_promotion_check_command";
    if (!commandHasToken(artifact.promotion_check_command, "--check-evidence")) return "missing_metal_check_evidence_arg";
    if (!commandHasToken(artifact.promotion_check_command, "--require-promotion-ready")) return "missing_metal_require_promotion_ready";
    if (!commandHasArgValue(artifact.promotion_check_command, "--require-kernel", artifact.kernel_id)) return "missing_metal_require_kernel";
    if (!metalPromotionCommandsUseSameEvidencePath(artifact)) return "metal_promotion_evidence_path_mismatch";
    const evidence = metalRuntimeEvidenceFor(artifact, evidence_records) orelse return "missing_metal_runtime_evidence";
    if (!evidence.production_enabled) return "production_disabled";
    if (!evidence.correctness_passed) return "correctness_evidence_failed";
    if (!evidence.generated_route_checked) return metal_blocker_missing_generated_route;
    if (metalProviderRouteRequiredForKernel(artifact.kernel_id) and !evidence.provider_route_checked) return metal_blocker_missing_provider_route;
    if (evidence.repeat_runs < metal_promotion_repeat_runs) return "insufficient_metal_repeats";
    const speedup_blocker = metalPromotionSpeedupBlocker(evidence.measured_speedup, evidence.minimum_repeat_speedup);
    if (!std.mem.eql(u8, speedup_blocker, metal_blocker_none)) return speedup_blocker;
    if (!evidence.benchmark_passed) return "benchmark_evidence_failed";
    if (!evidence.promotion_ready) return "metal_promotion_not_ready";
    return metal_blocker_none;
}

pub fn metalPromotionSpeedupBlocker(measured_speedup: f64, minimum_repeat_speedup: f64) []const u8 {
    if (!metalPromotionSpeedupPasses(measured_speedup)) return metal_blocker_speedup_gate_missing;
    if (!metalPromotionSpeedupPasses(minimum_repeat_speedup)) return metal_blocker_unstable_benchmark_timing;
    return metal_blocker_none;
}

pub fn metalPromotionSpeedupPasses(speedup_value: f64) bool {
    if (!std.math.isFinite(speedup_value)) return false;
    return speedup_value >= metal_promotion_min_speedup;
}

pub fn metalProviderRouteRequiredForKernel(kernel_id: []const u8) bool {
    return std.mem.eql(u8, kernel_id, first_general_metal_q8_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q8_bias_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q8_bias_gelu_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q8_relu_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q8_1_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q8_k_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q2_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q2_bias_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q2_bias_gelu_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q3_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q3_bias_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q3_bias_gelu_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q4_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q4_bias_kernel_id) or
        std.mem.eql(u8, kernel_id, first_lazy_metal_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q5_0_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q5_1_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q5_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q5_bias_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q5_bias_gelu_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q6_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q6_bias_kernel_id) or
        std.mem.eql(u8, kernel_id, first_general_metal_q6_bias_gelu_kernel_id);
}

pub fn artifactHasMetalProviderRouteEvidence(artifact: GeneratedArtifact) bool {
    if (artifact.backend != .metal or !artifactRuntimeWired(artifact)) return false;
    return switch (artifact.format) {
        .q8_0 => artifact.epilogue == .none or artifact.epilogue == .bias or artifact.epilogue == .bias_gelu or artifact.epilogue == .relu,
        .q8_1, .q8_k => artifact.epilogue == .none,
        .q2_k => artifact.epilogue == .none or artifact.epilogue == .bias or artifact.epilogue == .bias_gelu,
        .q5_0, .q5_1 => artifact.epilogue == .none,
        .q3_k, .q4_k, .q5_k, .q6_k => artifact.epilogue == .none or artifact.epilogue == .bias or artifact.epilogue == .bias_gelu,
        else => false,
    };
}

fn metalPromotionEvidenceCommandMatchesPolicy(artifact: GeneratedArtifact) bool {
    const evidence_path = commandArgValue(artifact.promotion_evidence_command, "--evidence-out") orelse return false;
    return std.mem.startsWith(u8, artifact.promotion_evidence_command, first_metal_promotion_evidence_command) and
        std.mem.containsAtLeast(u8, evidence_path, 1, artifact.kernel_id) and
        commandHasArgValue(artifact.promotion_evidence_command, "--repeat-runs", metal_promotion_repeat_runs_text) and
        commandHasArgValue(artifact.promotion_evidence_command, "--measure-iters", metal_promotion_measure_iters_text);
}

fn metalPromotionCommandsUseSameEvidencePath(artifact: GeneratedArtifact) bool {
    const evidence_path = commandArgValue(artifact.promotion_evidence_command, "--evidence-out") orelse return false;
    return commandHasArgValue(artifact.promotion_check_command, "--check-evidence", evidence_path);
}

fn benchmarkHasPromotionEvidence(bench: BenchmarkCase) bool {
    return std.mem.eql(u8, benchmarkPromotionBlocker(bench), "none");
}

fn benchmarkPromotionBlocker(bench: BenchmarkCase) []const u8 {
    return benchmarkPromotionBlockerWithEvidence(bench, &first_benchmark_evidence);
}

fn benchmarkPromotionBlockerWithEvidence(bench: BenchmarkCase, evidence_records: []const BenchmarkEvidence) []const u8 {
    if (!bench.production_enabled) return "production_disabled";
    if (bench.backend != .cuda) return "non_cuda_benchmark";
    if (isDevGeneratedSourcePath(bench.generated_source_path)) return "dev_generated_source";
    if (bench.generated_kernel_id.len == 0) return "missing_kernel_id";
    if (bench.generated_source_path.len == 0) return "missing_source_path";
    if (bench.generated_source_fingerprint == 0) return "missing_source_fingerprint";
    if (!std.mem.eql(u8, bench.generated_ptx_arg, "--quant-compiler-generated-ptx")) return "wrong_generated_ptx_arg";
    if (bench.handwritten_baseline.len == 0) return "missing_handwritten_baseline";
    if (!isPtxPath(bench.generated_ptx_path)) return "missing_generated_ptx_path";
    if (!commandHasToken(bench.generated_ptx_command, "nvcc") or !commandHasToken(bench.generated_ptx_command, "-ptx")) return "missing_nvcc_ptx_command";
    if (!commandHasToken(bench.generated_ptx_command, "-arch=compute_75")) return "wrong_ptx_arch";
    if (!commandHasToken(bench.generated_ptx_command, bench.generated_source_path)) return "ptx_command_missing_source";
    if (!commandHasArgValue(bench.generated_ptx_command, "-o", bench.generated_ptx_path)) return "ptx_command_missing_output";
    if (!commandFirstTokenEquals(bench.benchmark_command, "zig-out/bin/antfly-inference") or !commandHasToken(bench.benchmark_command, "bench-cuda")) return "missing_bench_cuda_command";
    if (!commandHasArgValue(bench.benchmark_command, "--warmup-iters", "5")) return "missing_warmup_iters";
    if (!commandHasArgValue(bench.benchmark_command, "--measure-iters", "50")) return "missing_measure_iters";
    if (!commandHasToken(bench.benchmark_command, "--quant-compiler-lazy-target")) return "missing_lazy_target_flag";
    if (!commandHasToken(bench.benchmark_command, bench.generated_ptx_arg)) return "benchmark_missing_generated_ptx_arg";
    if (!commandHasArgValue(bench.benchmark_command, bench.generated_ptx_arg, bench.generated_ptx_path)) return "benchmark_missing_generated_ptx_path";
    if (!commandHasArgValue(bench.benchmark_command, "--quant-compiler-repeat-runs", "3")) return "missing_benchmark_repeat_runs";
    if (!commandHasToken(bench.benchmark_command, "--quant-compiler-evidence-out")) return "benchmark_missing_evidence_out_arg";
    if (!std.math.isFinite(bench.correctness_tolerance_abs) or bench.correctness_tolerance_abs <= 0.0 or bench.correctness_tolerance_abs > 0.01) return "correctness_tolerance_missing_or_loose";
    if (bench.correctness_evidence_path.len == 0) return "missing_correctness_evidence";
    if (bench.benchmark_evidence_path.len == 0) return "missing_benchmark_evidence";
    if (!std.mem.eql(u8, bench.correctness_evidence_path, bench.benchmark_evidence_path)) return "evidence_path_mismatch";
    if (!commandHasArgValue(bench.benchmark_command, "--quant-compiler-evidence-out", bench.benchmark_evidence_path)) return "benchmark_missing_evidence_out_path";
    if (!std.mem.eql(u8, bench.benchmark_mode, "sequential")) return "missing_sequential_benchmark_evidence";
    if (!std.math.isFinite(bench.minimum_speedup) or bench.minimum_speedup < 1.0) return "speedup_gate_missing";
    const evidence = benchmarkEvidenceFor(bench, evidence_records) orelse return "missing_matching_evidence_record";
    if (!evidence.correctness_passed) return "correctness_evidence_failed";
    if (!evidence.benchmark_passed) return "benchmark_evidence_failed";
    if (evidence.repeat_runs < 3) return "insufficient_benchmark_repeats";
    if (!std.math.isFinite(evidence.measured_speedup)) return "speedup_gate_missing";
    if (evidence.measured_speedup < bench.minimum_speedup) return "speedup_gate_missing";
    return "none";
}

fn commandFirstTokenEquals(command: []const u8, expected: []const u8) bool {
    var tokens = std.mem.tokenizeScalar(u8, command, ' ');
    const first = tokens.next() orelse return false;
    return std.mem.eql(u8, first, expected);
}

fn commandHasToken(command: []const u8, expected: []const u8) bool {
    var tokens = std.mem.tokenizeScalar(u8, command, ' ');
    while (tokens.next()) |token| {
        if (std.mem.eql(u8, token, expected)) return true;
    }
    return false;
}

fn commandHasArgValue(command: []const u8, arg: []const u8, value: []const u8) bool {
    const actual = commandArgValue(command, arg) orelse return false;
    return std.mem.eql(u8, actual, value);
}

fn commandArgValue(command: []const u8, arg: []const u8) ?[]const u8 {
    var tokens = std.mem.tokenizeScalar(u8, command, ' ');
    while (tokens.next()) |token| {
        if (std.mem.eql(u8, token, arg)) {
            return tokens.next();
        }
    }
    return null;
}

fn benchmarkMeasuredSpeedup(bench: BenchmarkCase) f64 {
    const evidence = benchmarkEvidenceFor(bench, &first_benchmark_evidence) orelse return 0.0;
    return evidence.measured_speedup;
}

fn benchmarkEvidenceFor(bench: BenchmarkCase, evidence_records: []const BenchmarkEvidence) ?BenchmarkEvidence {
    for (evidence_records) |evidence| {
        if (std.mem.eql(u8, evidence.kernel_id, bench.generated_kernel_id) and
            std.mem.eql(u8, evidence.generated_source_path, bench.generated_source_path) and
            evidence.generated_source_fingerprint == bench.generated_source_fingerprint and
            std.mem.eql(u8, evidence.generated_ptx_path, bench.generated_ptx_path) and
            std.mem.eql(u8, evidence.generated_ptx_command, bench.generated_ptx_command) and
            std.mem.eql(u8, evidence.benchmark_command, bench.benchmark_command) and
            std.mem.eql(u8, evidence.correctness_evidence_path, bench.correctness_evidence_path) and
            std.mem.eql(u8, evidence.benchmark_evidence_path, bench.benchmark_evidence_path) and
            std.mem.eql(u8, evidence.benchmark_mode, bench.benchmark_mode))
        {
            return evidence;
        }
    }
    return null;
}

fn metalRuntimeEvidenceFor(artifact: GeneratedArtifact, evidence_records: []const MetalRuntimeEvidence) ?MetalRuntimeEvidence {
    const expected_source_path = generatedMetalSourcePathForKernel(artifact.kernel_id) orelse return null;
    const expected_artifact_source_path = metalArtifactSourcePathForKernel(artifact.kernel_id);
    const expected_check_command = generatedMetalCheckCommandForKernel(artifact.kernel_id) orelse return null;
    for (evidence_records) |evidence| {
        if (expected_artifact_source_path) |artifact_source_path| {
            if (!std.mem.eql(u8, evidence.artifact_source_path, artifact_source_path)) continue;
        } else if (evidence.artifact_source_path.len != 0) {
            continue;
        }
        if (std.mem.eql(u8, evidence.kernel_id, artifact.kernel_id) and
            // Promotion evidence is usually collected before copying identical source into artifacts/.
            std.mem.eql(u8, evidence.source_path, expected_source_path) and
            evidence.source_fingerprint == artifactSourceFingerprint(artifact) and
            std.mem.eql(u8, evidence.check_command, expected_check_command) and
            metalRuntimeEvidenceCommandMatchesArtifact(evidence, artifact) and
            std.mem.eql(u8, evidence.promotion_check_command, artifact.promotion_check_command))
        {
            return evidence;
        }
    }
    return null;
}

fn checkedInMetalEvidenceForKernel(kernel_id: []const u8) ?MetalRuntimeEvidence {
    for (first_metal_runtime_evidence) |evidence| {
        if (std.mem.eql(u8, evidence.kernel_id, kernel_id) and
            evidence.artifact_source_path.len != 0 and
            evidence.correctness_passed and
            evidence.source_fingerprint != 0)
        {
            return evidence;
        }
    }
    return null;
}

pub fn metalArtifactSourcePathForKernel(kernel_id: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, kernel_id, first_lazy_metal_kernel_id)) return first_lazy_metal_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q4_kernel_id)) return first_general_metal_q4_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q3_kernel_id)) return first_general_metal_q3_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q4_bias_kernel_id)) return first_general_metal_q4_bias_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q8_kernel_id)) return first_general_metal_q8_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q8_bias_kernel_id)) return first_general_metal_q8_bias_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q8_bias_gelu_kernel_id)) return first_general_metal_q8_bias_gelu_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q5_kernel_id)) return first_general_metal_q5_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q5_bias_kernel_id)) return first_general_metal_q5_bias_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q5_bias_gelu_kernel_id)) return first_general_metal_q5_bias_gelu_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q5_1_kernel_id)) return first_general_metal_q5_1_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q5_0_kernel_id)) return first_general_metal_q5_0_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q6_kernel_id)) return first_general_metal_q6_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q6_bias_kernel_id)) return first_general_metal_q6_bias_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q6_bias_gelu_kernel_id)) return first_general_metal_q6_bias_gelu_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q2_kernel_id)) return first_general_metal_q2_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q4_1_kernel_id)) return first_general_metal_q4_1_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q8_1_kernel_id)) return first_general_metal_q8_1_artifact_source_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q8_k_kernel_id)) return first_general_metal_q8_k_artifact_source_path;
    return null;
}

fn metalArtifactAirPathForKernel(kernel_id: []const u8) ?[]const u8 {
    if (std.mem.eql(u8, kernel_id, first_lazy_metal_kernel_id)) return first_lazy_metal_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q4_kernel_id)) return first_general_metal_q4_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q3_kernel_id)) return first_general_metal_q3_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q4_bias_kernel_id)) return first_general_metal_q4_bias_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q8_kernel_id)) return first_general_metal_q8_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q8_bias_kernel_id)) return first_general_metal_q8_bias_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q8_bias_gelu_kernel_id)) return first_general_metal_q8_bias_gelu_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q5_kernel_id)) return first_general_metal_q5_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q5_bias_kernel_id)) return first_general_metal_q5_bias_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q5_bias_gelu_kernel_id)) return first_general_metal_q5_bias_gelu_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q5_1_kernel_id)) return first_general_metal_q5_1_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q5_0_kernel_id)) return first_general_metal_q5_0_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q6_kernel_id)) return first_general_metal_q6_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q6_bias_kernel_id)) return first_general_metal_q6_bias_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q6_bias_gelu_kernel_id)) return first_general_metal_q6_bias_gelu_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q2_kernel_id)) return first_general_metal_q2_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q4_1_kernel_id)) return first_general_metal_q4_1_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q8_1_kernel_id)) return first_general_metal_q8_1_artifact_air_path;
    if (std.mem.eql(u8, kernel_id, first_general_metal_q8_k_kernel_id)) return first_general_metal_q8_k_artifact_air_path;
    return metalAirPathForKernel(kernel_id);
}

pub fn generatedMetalSourcePathForKernel(kernel_id: []const u8) ?[]const u8 {
    const artifact = generatedArtifactForKernel(.metal, kernel_id) orelse return null;
    return generatedSourcePathForArtifact(artifact);
}

pub fn generatedMetalCheckCommandForKernel(kernel_id: []const u8) ?[]const u8 {
    const artifact = generatedArtifactForKernel(.metal, kernel_id) orelse return null;
    return generatedCheckCommandForArtifact(artifact);
}

fn metalRuntimeEvidenceCommandMatchesArtifact(evidence: MetalRuntimeEvidence, artifact: GeneratedArtifact) bool {
    if (evidence.production_enabled or evidence.promotion_ready) {
        return std.mem.eql(u8, evidence.runtime_evidence_command, artifact.promotion_evidence_command);
    }
    return std.mem.eql(u8, evidence.runtime_evidence_command, artifact.runtime_evidence_command);
}

fn benchmarkForArtifact(artifact: GeneratedArtifact) ?BenchmarkCase {
    for (first_benchmarks) |bench| {
        if (artifact.backend == bench.backend and
            artifact.format == bench.format and
            artifact.row_bucket == bench.row_bucket and
            artifact.epilogue == bench.epilogue and
            std.mem.eql(u8, artifact.source_path, bench.generated_source_path) and
            std.mem.eql(u8, artifact.kernel_id, bench.generated_kernel_id))
        {
            return bench;
        }
    }
    return null;
}

fn isDevGeneratedSourcePath(path: []const u8) bool {
    @setEvalBranchQuota(10_000);
    return std.mem.containsAtLeast(u8, path, 1, "src/ops/cuda/generated/") or
        std.mem.containsAtLeast(u8, path, 1, "src/ops/metal/generated/") or
        std.mem.startsWith(u8, path, "generated/");
}

fn isPtxPath(path: []const u8) bool {
    return std.mem.endsWith(u8, path, ".ptx");
}

fn metalAirPathForKernel(kernel_id: []const u8) ?[]const u8 {
    const command = generatedMetalCheckCommandForKernel(kernel_id) orelse return null;
    return commandArgValue(command, "-o");
}

pub fn compileQuantKernelSource(request: QuantKernelCompileRequest) ?QuantKernelCompiledSource {
    const spec = specFor(request.format) orelse return null;
    const ir = buildIr(request.format, request.row_bucket, request.epilogue) orelse return null;
    const artifact = generatedArtifactForCandidate(request.backend, request.format, request.row_bucket, request.epilogue) orelse return null;
    const source = generatedSourceForArtifact(artifact) orelse return null;
    const lowering = registryLoweringFor(request.backend, request.format, request.row_bucket, request.epilogue, ir.dispatch);
    const artifact_source_path = if (request.backend == .metal)
        metalArtifactSourcePathForKernel(artifact.kernel_id) orelse ""
    else
        "";
    const runtime_gate_env = if (request.backend == .metal)
        artifactRuntimeGateEnv(artifact)
    else
        null;
    const compiled = QuantKernelCompiledSource{
        .request = request,
        .spec = spec,
        .ir = ir,
        .lowering = lowering,
        .artifact = artifact,
        .source = source,
        .source_path = generatedSourcePathForArtifact(artifact),
        .artifact_source_path = artifact_source_path,
        .check_command = generatedCheckCommandForArtifact(artifact),
        .runtime_gate_env = runtime_gate_env,
        .production_enabled = artifact.production_enabled,
    };
    if (!compiledSourceMatchesRoute(compiled)) return null;
    return compiled;
}

pub fn compiledSourceMatchesRoute(compiled: QuantKernelCompiledSource) bool {
    if (compiled.artifact.backend != compiled.request.backend or
        compiled.artifact.format != compiled.request.format or
        compiled.artifact.row_bucket != compiled.request.row_bucket or
        compiled.artifact.epilogue != compiled.request.epilogue)
    {
        return false;
    }
    const expected_source = generatedSourceForArtifact(compiled.artifact) orelse return false;
    if (!std.mem.eql(u8, compiled.source, expected_source) or
        !std.mem.eql(u8, compiled.source_path, generatedSourcePathForArtifact(compiled.artifact)) or
        !std.mem.eql(u8, compiled.check_command, generatedCheckCommandForArtifact(compiled.artifact)) or
        compiled.production_enabled != compiled.artifact.production_enabled)
    {
        return false;
    }
    if (compiled.request.backend == .metal) {
        const expected_artifact_source_path = metalArtifactSourcePathForKernel(compiled.artifact.kernel_id) orelse "";
        if (!std.mem.eql(u8, compiled.artifact_source_path, expected_artifact_source_path)) return false;
        const expected_gate = artifactRuntimeGateEnv(compiled.artifact);
        if (!optionalCStringEquals(compiled.runtime_gate_env, expected_gate)) return false;
    } else {
        if (compiled.artifact_source_path.len != 0 or compiled.runtime_gate_env != null) return false;
    }
    if (compiled.spec.format != compiled.request.format or
        compiled.ir.format != compiled.request.format or
        compiled.ir.row_bucket != compiled.request.row_bucket or
        compiled.ir.epilogue != compiled.request.epilogue)
    {
        return false;
    }
    if (compiled.lowering.backend != compiled.request.backend or
        compiled.lowering.format != compiled.request.format or
        compiled.lowering.row_bucket != compiled.request.row_bucket or
        compiled.lowering.epilogue != compiled.request.epilogue or
        compiled.lowering.schedule.dispatch != compiled.ir.dispatch)
    {
        return false;
    }

    const promotion_ready = artifactHasPromotionEvidence(compiled.artifact);
    if (promotion_ready) {
        return compiled.lowering.production_route == .generated_production and
            compiled.lowering.candidate_route == .unsupported and
            compiled.lowering.fallback_reason == .none and
            std.mem.eql(u8, compiled.lowering.production_kernel_id, compiled.artifact.kernel_id) and
            compiled.lowering.kernel_id.len == 0 and
            compiled.lowering.candidate_source_path.len == 0;
    }

    return compiled.lowering.production_route == .handwritten_production and
        compiled.lowering.candidate_route == .generated_dev_candidate and
        std.mem.eql(u8, compiled.lowering.kernel_id, compiled.artifact.kernel_id) and
        std.mem.eql(u8, compiled.lowering.candidate_source_path, compiled.artifact.source_path);
}

fn optionalCStringEquals(a: ?[*:0]const u8, b: ?[*:0]const u8) bool {
    if (a == null or b == null) return a == null and b == null;
    return std.mem.eql(u8, std.mem.span(a.?), std.mem.span(b.?));
}

pub fn compileMetalKernelSource(
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
) ?QuantKernelCompiledSource {
    return compileQuantKernelSource(.{
        .backend = .metal,
        .format = format,
        .row_bucket = row_bucket,
        .epilogue = epilogue,
    });
}

pub fn emitCompiledSource(allocator: std.mem.Allocator, compiled: QuantKernelCompiledSource) !EmittedCompiledSource {
    if (compiled.request.backend == .metal and
        compiled.request.format == .q8_0 and
        compiled.request.row_bucket == .rows_2_8 and
        metalQ8SmallBatchEpilogueSupported(compiled.request.epilogue))
    {
        return .{
            .data = try emitMetalQ8SmallBatchSource(allocator, compiled),
            .owned = true,
        };
    }
    if (compiled.request.backend == .metal and
        (compiled.request.format == .q8_1 or compiled.request.format == .q8_k) and
        compiled.request.row_bucket == .rows_2_8 and
        compiled.request.epilogue == .none)
    {
        return .{
            .data = try emitMetalQ8FamilySmallBatchSource(allocator, compiled),
            .owned = true,
        };
    }
    if (compiled.request.backend == .metal and
        compiled.request.format == .q4_k and
        compiled.request.row_bucket == .rows_2_8 and
        metalQ4KSmallBatchEpilogueSupported(compiled.request.epilogue))
    {
        return .{
            .data = try emitMetalQ4KSmallBatchSource(allocator, compiled),
            .owned = true,
        };
    }
    if (compiled.request.backend == .metal and
        (compiled.request.format == .q4_0 or compiled.request.format == .q5_0) and
        compiled.request.row_bucket == .rows_2_8 and
        compiled.request.epilogue == .none)
    {
        return .{
            .data = try emitMetalLegacyScalarSmallBatchSource(allocator, compiled),
            .owned = true,
        };
    }
    if (compiled.request.backend == .metal and
        compiled.request.format == .q4_1 and
        compiled.request.row_bucket == .rows_2_8 and
        compiled.request.epilogue == .none)
    {
        return .{
            .data = try emitMetalQ4_1SmallBatchSource(allocator, compiled),
            .owned = true,
        };
    }
    if (compiled.request.backend == .metal and
        compiled.request.format == .q5_1 and
        compiled.request.row_bucket == .rows_2_8 and
        compiled.request.epilogue == .none)
    {
        return .{
            .data = try emitMetalQ5_1SmallBatchSource(allocator, compiled),
            .owned = true,
        };
    }
    if (compiled.request.backend == .metal and
        compiled.request.format == .q2_k and
        compiled.request.row_bucket == .rows_2_8 and
        metalQ2KSmallBatchEpilogueSupported(compiled.request.epilogue))
    {
        return .{
            .data = try emitMetalQ2KSmallBatchSource(allocator, compiled),
            .owned = true,
        };
    }
    if (compiled.request.backend == .metal and
        compiled.request.format == .q3_k and
        compiled.request.row_bucket == .rows_2_8 and
        metalQ3KSmallBatchEpilogueSupported(compiled.request.epilogue))
    {
        return .{
            .data = try emitMetalQ3KSmallBatchSource(allocator, compiled),
            .owned = true,
        };
    }
    if (compiled.request.backend == .metal and
        compiled.request.format == .q6_k and
        compiled.request.row_bucket == .rows_2_8 and
        compiled.request.epilogue == .none)
    {
        return .{
            .data = try emitMetalQ6KSmallBatchSource(allocator, compiled),
            .owned = true,
        };
    }
    if (compiled.request.backend == .metal and
        compiled.request.format == .q6_k and
        compiled.request.row_bucket == .rows_2_8 and
        metalQ6KSmallBatchBiasEpilogueSupported(compiled.request.epilogue))
    {
        return .{
            .data = try emitMetalQ6KSmallBatchBiasSource(allocator, compiled),
            .owned = true,
        };
    }
    if (compiled.request.backend == .metal and
        compiled.request.format == .q5_k and
        compiled.request.row_bucket == .rows_2_8 and
        compiled.request.epilogue == .none)
    {
        return .{
            .data = try emitMetalQ5KSmallBatchSource(allocator, compiled),
            .owned = true,
        };
    }
    if (compiled.request.backend == .metal and
        compiled.request.format == .q5_k and
        compiled.request.row_bucket == .rows_2_8 and
        metalQ5KSmallBatchBiasEpilogueSupported(compiled.request.epilogue))
    {
        return .{
            .data = try emitMetalQ5KSmallBatchBiasSource(allocator, compiled),
            .owned = true,
        };
    }
    return .{ .data = compiled.source };
}

pub fn compiledSourceHeaderMatchesPlan(allocator: std.mem.Allocator, compiled: QuantKernelCompiledSource) !bool {
    return sourceHeaderMatchesCompiledPlan(allocator, compiled, compiled.source);
}

pub fn compiledSourceHeaderMatchesSource(
    allocator: std.mem.Allocator,
    compiled: QuantKernelCompiledSource,
    source: []const u8,
) !bool {
    return sourceHeaderMatchesCompiledPlan(allocator, compiled, source);
}

fn sourceHeaderMatchesCompiledPlan(
    allocator: std.mem.Allocator,
    compiled: QuantKernelCompiledSource,
    source: []const u8,
) !bool {
    const plan_name = try planIdName(allocator, compiled.lowering.plan_id);
    defer allocator.free(plan_name);

    const plan_metadata = try std.fmt.allocPrint(allocator, "plan_id={s}", .{plan_name});
    defer allocator.free(plan_metadata);
    if (!std.mem.containsAtLeast(u8, source, 1, plan_metadata)) return false;

    const kernel_metadata = try std.fmt.allocPrint(allocator, "kernel_id={s}", .{compiled.artifact.kernel_id});
    defer allocator.free(kernel_metadata);
    if (!std.mem.containsAtLeast(u8, source, 1, kernel_metadata)) return false;

    const baseline_id = productionKernelId(
        compiled.artifact.backend,
        compiled.artifact.format,
        compiled.artifact.row_bucket,
        compiled.artifact.epilogue,
    );
    const baseline_metadata = try std.fmt.allocPrint(allocator, "production_baseline={s}", .{baseline_id});
    defer allocator.free(baseline_metadata);
    if (!std.mem.containsAtLeast(u8, source, 1, baseline_metadata)) return false;

    const enabled_metadata = if (compiled.artifact.production_enabled)
        "production_enabled=true"
    else
        "production_enabled=false";
    if (!std.mem.containsAtLeast(u8, source, 1, enabled_metadata)) return false;

    if (!std.mem.containsAtLeast(u8, source, 1, compiled.artifact.kernel_id)) return false;
    return true;
}

fn metalQ8SmallBatchEpilogueSupported(epilogue: Epilogue) bool {
    return switch (epilogue) {
        .none, .bias, .bias_gelu, .relu => true,
        else => false,
    };
}

fn metalQ8SmallBatchSourceKind(epilogue: Epilogue) []const u8 {
    return switch (epilogue) {
        .relu => "Dev-only generated Metal candidate",
        else => "Generated Metal artifact source",
    };
}

fn metalQ8SmallBatchPromotionComment(artifact: GeneratedArtifact, epilogue: Epilogue) []const u8 {
    if (!artifactHasPromotionEvidence(artifact)) {
        return "// General MSL lowering smoke for descriptor-driven quant matmul epilogues.\n// Production Metal dispatch stays on native handwritten MSL until this\n// candidate clears correctness and benchmark gates.";
    }
    return switch (epilogue) {
        .none, .bias => "// Promoted after sequential Metal runtime evidence cleared correctness,\n// route, provider-route, and speedup gates.",
        .bias_gelu => "// Promoted after active-frame decode-runtime evidence cleared the\n// sequential benchmark and production-regression gates.",
        .relu => unreachable,
        else => "",
    };
}

fn metalQ4KSmallBatchEpilogueSupported(epilogue: Epilogue) bool {
    return switch (epilogue) {
        .none, .bias, .bias_gelu => true,
        else => false,
    };
}

fn metalQ4KSmallBatchPromotionComment(artifact: GeneratedArtifact, epilogue: Epilogue) []const u8 {
    if (!artifactHasPromotionEvidence(artifact)) {
        return "// General MSL lowering smoke for descriptor-driven quant matmul epilogues.\n// Production Metal dispatch stays on native handwritten MSL until this\n// candidate clears correctness and benchmark gates.";
    }
    return switch (epilogue) {
        .none, .bias, .bias_gelu => "// Promoted after sequential Metal runtime evidence cleared correctness,\n// route, provider-route, and speedup gates.",
        else => "",
    };
}

fn metalLegacyScalarSmallBatchSourceKind(format: quant_matmul.Format) []const u8 {
    return switch (format) {
        .q4_0 => "Generated Metal candidate artifact",
        .q4_1 => "Generated Metal artifact source",
        .q5_0, .q5_1 => "Dev-only generated Metal candidate",
        else => "",
    };
}

fn metalLegacyScalarSmallBatchPromotionComment(artifact: GeneratedArtifact) []const u8 {
    if (artifact.production_enabled) {
        return "// Promoted after sequential Metal runtime evidence cleared correctness,\n// route, and speedup gates.";
    }
    return "// General MSL lowering smoke for descriptor-driven quant matmul.\n// Production Metal dispatch stays on native handwritten MSL until this\n// candidate clears correctness and benchmark gates.";
}

fn metalQ2KSmallBatchEpilogueSupported(epilogue: Epilogue) bool {
    return switch (epilogue) {
        .none, .bias, .bias_gelu => true,
        else => false,
    };
}

fn metalQ2KSmallBatchPromotionComment(epilogue: Epilogue) []const u8 {
    return switch (epilogue) {
        .none => "// Promoted after sequential Metal runtime evidence cleared correctness,\n// route, and speedup gates.",
        .bias, .bias_gelu => "// General MSL lowering smoke for descriptor-driven quant matmul epilogues.\n// Production Metal dispatch stays on native handwritten MSL until this\n// candidate clears correctness and benchmark gates.",
        else => "",
    };
}

fn metalQ3KSmallBatchEpilogueSupported(epilogue: Epilogue) bool {
    return switch (epilogue) {
        .none, .bias, .bias_gelu => true,
        else => false,
    };
}

fn metalQ3KSmallBatchSourceKind(epilogue: Epilogue) []const u8 {
    return switch (epilogue) {
        .none => "Generated Metal candidate artifact",
        .bias, .bias_gelu => "Dev-only generated Metal candidate",
        else => "",
    };
}

fn metalQ3KSmallBatchPromotionComment(epilogue: Epilogue) []const u8 {
    return switch (epilogue) {
        .none => "// Promoted after sequential Metal runtime evidence cleared correctness,\n// route, provider-route, and speedup gates.",
        .bias, .bias_gelu => "// General MSL lowering smoke for descriptor-driven quant matmul epilogues.\n// Production Metal dispatch stays on native handwritten MSL until this\n// candidate clears correctness and benchmark gates.",
        else => "",
    };
}

fn metalQ6KSmallBatchPromotionComment(artifact: GeneratedArtifact, epilogue: Epilogue) []const u8 {
    if (!artifactHasPromotionEvidence(artifact)) {
        return "// General MSL lowering smoke for descriptor-driven quant matmul epilogues.\n// Production Metal dispatch stays on native handwritten MSL until this\n// candidate clears correctness and benchmark gates.";
    }
    return switch (epilogue) {
        .none, .bias, .bias_gelu => "// Promoted after sequential Metal runtime evidence cleared correctness,\n// route, provider-route, and speedup gates.",
        else => "",
    };
}

fn metalQ6KSmallBatchBiasEpilogueSupported(epilogue: Epilogue) bool {
    return switch (epilogue) {
        .bias, .bias_gelu => true,
        else => false,
    };
}

fn metalQ5KSmallBatchBiasEpilogueSupported(epilogue: Epilogue) bool {
    return switch (epilogue) {
        .bias, .bias_gelu => true,
        else => false,
    };
}

fn metalQ5KSmallBatchPromotionComment(artifact: GeneratedArtifact, epilogue: Epilogue) []const u8 {
    if (!artifactHasPromotionEvidence(artifact)) {
        return "// General MSL lowering smoke for descriptor-driven quant matmul epilogues.\n// Production Metal dispatch stays on native handwritten MSL until this\n// candidate clears correctness and benchmark gates.";
    }
    return switch (epilogue) {
        .none, .bias, .bias_gelu => "// Promoted after sequential Metal runtime evidence cleared correctness,\n// route, provider-route, and speedup gates.",
        else => "",
    };
}

fn blockFieldOffset(spec: QuantKernelSpec, name: []const u8) usize {
    for (spec.block_fields) |field| {
        if (std.mem.eql(u8, field.name, name)) return field.offset;
    }
    unreachable;
}

fn blockPointerOffsetExpr(allocator: std.mem.Allocator, offset: usize) ![]u8 {
    if (offset == 0) return allocator.dupe(u8, "block");
    return std.fmt.allocPrint(allocator, "block + {d}", .{offset});
}

fn emitMetalQ8SmallBatchSource(
    allocator: std.mem.Allocator,
    compiled: QuantKernelCompiledSource,
) ![]u8 {
    const plan_name = try planIdName(allocator, compiled.lowering.plan_id);
    defer allocator.free(plan_name);

    const block_shift = blockValueShift(compiled.spec.block_values);
    const block_mask = compiled.spec.block_values - 1;
    const baseline_id = productionKernelId(
        compiled.artifact.backend,
        compiled.artifact.format,
        compiled.artifact.row_bucket,
        compiled.artifact.epilogue,
    );
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try appendFmt(
        allocator,
        &out,
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
        \\
        \\// {s} from graph/quant_kernel_compiler.zig.
        \\// plan_id={s}
        \\// kernel_id={s}
        \\// production_baseline={s}
        \\// production_enabled={}
        \\{s}
        \\
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\static inline float antfly_half_le_to_float(const device uchar *p) {{
        \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
        \\    return (float)as_type<half>(bits);
        \\}}
        \\
        \\static inline float antfly_q8_0_dequant_lane(const device uchar *block, int lane) {{
        \\    const float d = antfly_half_le_to_float(block);
        \\    const int q = (int)as_type<char>(block[2 + lane]);
        \\    return d * (float)q;
        \\}}
        \\
    ,
        .{
            metalQ8SmallBatchSourceKind(compiled.ir.epilogue),
            plan_name,
            compiled.artifact.kernel_id,
            baseline_id,
            compiled.artifact.production_enabled,
            metalQ8SmallBatchPromotionComment(compiled.artifact, compiled.ir.epilogue),
        },
    );
    try out.append(allocator, '\n');

    if (compiled.ir.epilogue == .bias_gelu) {
        try appendFmt(
            allocator,
            &out,
            \\static inline float antfly_gelu(float x) {{
            \\    return 0.5f * x * (1.0f + fast::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
            \\}}
            \\
        ,
            .{},
        );
        try out.append(allocator, '\n');
    }

    switch (compiled.ir.epilogue) {
        .none, .relu => try appendFmt(
            allocator,
            &out,
            \\kernel void {s}(
            \\    const device float *input [[buffer(0)]],
            \\    const device uchar *weight_q8_0 [[buffer(1)]],
            \\    device float *output [[buffer(2)]],
            \\    constant int &rows [[buffer(3)]],
            \\    constant int &in_dim [[buffer(4)]],
            \\    constant int &out_dim [[buffer(5)]],
            \\    uint3 thread_pos [[thread_position_in_threadgroup]],
            \\    uint3 group_pos [[threadgroup_position_in_grid]]
            \\) {{
            \\    const uint tid = thread_pos.x;
            \\    const int col = (int)group_pos.x;
            \\    const int row = (int)group_pos.y;
            \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & {d}) != 0) return;
            \\
        ,
            .{
                compiled.artifact.kernel_id,
                block_mask,
            },
        ),
        .bias, .bias_gelu => try appendFmt(
            allocator,
            &out,
            \\kernel void {s}(
            \\    const device float *input [[buffer(0)]],
            \\    const device uchar *weight_q8_0 [[buffer(1)]],
            \\    const device float *bias [[buffer(2)]],
            \\    device float *output [[buffer(3)]],
            \\    constant int &rows [[buffer(4)]],
            \\    constant int &in_dim [[buffer(5)]],
            \\    constant int &out_dim [[buffer(6)]],
            \\    uint3 thread_pos [[thread_position_in_threadgroup]],
            \\    uint3 group_pos [[threadgroup_position_in_grid]]{s}
            \\
        ,
            .{
                compiled.artifact.kernel_id,
                if (compiled.ir.epilogue == .bias) "," else "",
            },
        ),
        else => return error.UnsupportedQuantKernelEpilogue,
    }

    if (compiled.ir.epilogue == .bias) {
        try appendFmt(
            allocator,
            &out,
            \\    ushort lane_id [[thread_index_in_simdgroup]],
            \\    ushort simdgroup_id [[simdgroup_index_in_threadgroup]]
            \\
        ,
            .{},
        );
    }

    if (compiled.ir.epilogue == .bias or compiled.ir.epilogue == .bias_gelu) {
        try appendFmt(
            allocator,
            &out,
            \\) {{
            \\    const uint tid = thread_pos.x;
            \\    const int col = (int)group_pos.x;
            \\    const int row = (int)group_pos.y;
            \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & {d}) != 0) return;
            \\
        ,
            .{block_mask},
        );
    }
    try out.append(allocator, '\n');

    if (compiled.ir.epilogue == .relu) {
        try appendFmt(
            allocator,
            &out,
            \\    float acc = 0.0f;
            \\    const int block_count = in_dim >> {d};
            \\    if (tid < {d}) {{
            \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {{
            \\            const device uchar *block = weight_q8_0 + ((col * block_count + block_idx) * {d});
            \\            const int lane = (int)tid;
            \\            acc += input[row * in_dim + (block_idx << {d}) + lane] * antfly_q8_0_dequant_lane(block, lane);
            \\        }}
            \\    }}
            \\
            \\    acc = simd_sum(acc);
            \\    if (tid == 0) output[row * out_dim + col] = max(acc, 0.0f);
            \\}}
            \\
        ,
            .{
                block_shift,
                compiled.spec.block_values,
                compiled.spec.block_bytes,
                block_shift,
            },
        );
        return try out.toOwnedSlice(allocator);
    }

    try appendFmt(
        allocator,
        &out,
        \\    float acc = 0.0f;
        \\    const int block_count = in_dim >> {d};
        \\    const int lane = (int)tid;
        \\    const device float *row_input = input + row * in_dim;
        \\    const device uchar *col_weight = weight_q8_0 + col * block_count * {d};
        \\    for (int block_idx = 0; block_idx < block_count; ++block_idx) {{
        \\
    ,
        .{
            block_shift,
            compiled.spec.block_bytes,
        },
    );

    if (compiled.ir.epilogue == .bias_gelu) {
        try appendFmt(
            allocator,
            &out,
            \\        const float x = row_input[(block_idx << {d}) + lane];
            \\        const device uchar *block = col_weight + block_idx * {d};
            \\        acc += x * antfly_q8_0_dequant_lane(block, lane);
            \\    }}
            \\
            \\    acc = simd_sum(acc);
            \\    if (tid == 0) {{
            \\        output[row * out_dim + col] = antfly_gelu(acc + bias[col]);
            \\    }}
            \\}}
            \\
        ,
            .{
                block_shift,
                compiled.spec.block_bytes,
            },
        );
    } else {
        const output_expr = switch (compiled.ir.epilogue) {
            .none => "acc",
            .bias => "acc + bias[col]",
            else => unreachable,
        };
        try appendFmt(
            allocator,
            &out,
            \\        const device uchar *block = col_weight + block_idx * {d};
            \\        acc += row_input[(block_idx << {d}) + lane] * antfly_q8_0_dequant_lane(block, lane);
            \\    }}
            \\
            \\    acc = simd_sum(acc);
            \\    if (tid == 0) output[row * out_dim + col] = {s};
            \\}}
            \\
        ,
            .{
                compiled.spec.block_bytes,
                block_shift,
                output_expr,
            },
        );
    }

    return try out.toOwnedSlice(allocator);
}

fn emitMetalLegacyScalarSmallBatchSource(
    allocator: std.mem.Allocator,
    compiled: QuantKernelCompiledSource,
) ![]u8 {
    const plan_name = try planIdName(allocator, compiled.lowering.plan_id);
    defer allocator.free(plan_name);

    const block_shift = blockValueShift(compiled.spec.block_values);
    const block_mask = compiled.spec.block_values - 1;
    const d_offset = blockFieldOffset(compiled.spec, "d");
    const qs_offset = blockFieldOffset(compiled.spec, "qs");
    const d_pointer = try blockPointerOffsetExpr(allocator, d_offset);
    defer allocator.free(d_pointer);
    const format_suffix = switch (compiled.request.format) {
        .q4_0 => "q4_0",
        .q5_0 => "q5_0",
        .q5_1 => "q5_1",
        else => return error.UnsupportedQuantKernelFormat,
    };
    const weight_name = switch (compiled.request.format) {
        .q4_0 => "weight_q4_0",
        .q5_0 => "weight_q5_0",
        .q5_1 => "weight_q5_1",
        else => return error.UnsupportedQuantKernelFormat,
    };
    const baseline_id = productionKernelId(
        compiled.artifact.backend,
        compiled.artifact.format,
        compiled.artifact.row_bucket,
        compiled.artifact.epilogue,
    );
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try appendFmt(
        allocator,
        &out,
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
        \\
        \\// {s} from graph/quant_kernel_compiler.zig.
        \\// plan_id={s}
        \\// kernel_id={s}
        \\// production_baseline={s}
        \\// production_enabled={}
        \\{s}
        \\
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\static inline float antfly_half_le_to_float(const device uchar *p) {{
        \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
        \\    return (float)as_type<half>(bits);
        \\}}
        \\
    ,
        .{
            if (compiled.artifact.production_enabled) "Generated Metal artifact source" else metalLegacyScalarSmallBatchSourceKind(compiled.request.format),
            plan_name,
            compiled.artifact.kernel_id,
            baseline_id,
            compiled.artifact.production_enabled,
            metalLegacyScalarSmallBatchPromotionComment(compiled.artifact),
        },
    );
    try out.append(allocator, '\n');

    if (compiled.request.format == .q5_0 or compiled.request.format == .q5_1) {
        try appendFmt(
            allocator,
            &out,
            \\static inline uint antfly_u32_le(const device uchar *p) {{
            \\    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
            \\}}
            \\
        ,
            .{},
        );
        try out.append(allocator, '\n');
    }

    switch (compiled.request.format) {
        .q4_0 => try appendFmt(
            allocator,
            &out,
            \\static inline float antfly_q4_0_dequant_lane(const device uchar *block, int lane) {{
            \\    const float d = antfly_half_le_to_float({s});
            \\    const int packed_index = lane & 15;
            \\    const uchar packed = block[{d} + packed_index];
            \\    const int q = lane < 16 ? (int)(packed & 0x0fu) - 8 : (int)(packed >> 4) - 8;
            \\    return d * (float)q;
            \\}}
            \\
        ,
            .{ d_pointer, qs_offset },
        ),
        .q5_0 => {
            const qh_offset = blockFieldOffset(compiled.spec, "qh");
            const qh_pointer = try blockPointerOffsetExpr(allocator, qh_offset);
            defer allocator.free(qh_pointer);
            try appendFmt(
                allocator,
                &out,
                \\static inline float antfly_q5_0_dequant_lane(const device uchar *block, int lane) {{
                \\    const float d = antfly_half_le_to_float({s});
                \\    const uint qh = antfly_u32_le({s});
                \\    const int packed_index = lane & 15;
                \\    const uchar packed = block[{d} + packed_index];
                \\    const int low4 = lane < 16 ? (int)(packed & 0x0fu) : (int)(packed >> 4);
                \\    const int high = (int)((qh >> (uint)lane) & 1u);
                \\    return d * (float)((low4 | (high << 4)) - 16);
                \\}}
                \\
            ,
                .{ d_pointer, qh_pointer, qs_offset },
            );
        },
        .q5_1 => {
            const m_offset = blockFieldOffset(compiled.spec, "m");
            const qh_offset = blockFieldOffset(compiled.spec, "qh");
            const m_pointer = try blockPointerOffsetExpr(allocator, m_offset);
            defer allocator.free(m_pointer);
            const qh_pointer = try blockPointerOffsetExpr(allocator, qh_offset);
            defer allocator.free(qh_pointer);
            try appendFmt(
                allocator,
                &out,
                \\static inline float antfly_q5_1_dequant_lane(const device uchar *block, int lane) {{
                \\    const float d = antfly_half_le_to_float({s});
                \\    const float m = antfly_half_le_to_float({s});
                \\    const uint qh = antfly_u32_le({s});
                \\    const int packed_index = lane & 15;
                \\    const uchar packed = block[{d} + packed_index];
                \\    const int low4 = lane < 16 ? (int)(packed & 0x0fu) : (int)(packed >> 4);
                \\    const int high = (int)((qh >> (uint)lane) & 1u);
                \\    return d * (float)(low4 | (high << 4)) + m;
                \\}}
                \\
            ,
                .{ d_pointer, m_pointer, qh_pointer, qs_offset },
            );
        },
        else => return error.UnsupportedQuantKernelFormat,
    }
    try out.append(allocator, '\n');

    try appendFmt(
        allocator,
        &out,
        \\kernel void {s}(
        \\    const device float *input [[buffer(0)]],
        \\    const device uchar *{s} [[buffer(1)]],
        \\    device float *output [[buffer(2)]],
        \\    constant int &rows [[buffer(3)]],
        \\    constant int &in_dim [[buffer(4)]],
        \\    constant int &out_dim [[buffer(5)]],
        \\    uint3 thread_pos [[thread_position_in_threadgroup]],
        \\    uint3 group_pos [[threadgroup_position_in_grid]]
        \\) {{
        \\    const uint tid = thread_pos.x;
        \\    const int col = (int)group_pos.x;
        \\    const int row = (int)group_pos.y;
        \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & {d}) != 0) return;
        \\
        \\    float acc = 0.0f;
        \\    const int block_count = in_dim >> {d};
        \\    if (tid < 32) {{
        \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {{
        \\            const device uchar *block = {s} + ((col * block_count + block_idx) * {d});
        \\            const int lane = (int)tid;
        \\            acc += input[row * in_dim + (block_idx << {d}) + lane] * antfly_{s}_dequant_lane(block, lane);
        \\        }}
        \\    }}
        \\
        \\    acc = simd_sum(acc);
        \\    if (tid == 0) output[row * out_dim + col] = acc;
        \\}}
        \\
    ,
        .{
            compiled.artifact.kernel_id,
            weight_name,
            block_mask,
            block_shift,
            weight_name,
            compiled.spec.block_bytes,
            block_shift,
            format_suffix,
        },
    );

    return try out.toOwnedSlice(allocator);
}

fn emitMetalQ4_1SmallBatchSource(
    allocator: std.mem.Allocator,
    compiled: QuantKernelCompiledSource,
) ![]u8 {
    const plan_name = try planIdName(allocator, compiled.lowering.plan_id);
    defer allocator.free(plan_name);

    const block_shift = blockValueShift(compiled.spec.block_values);
    const block_mask = compiled.spec.block_values - 1;
    const d_offset = blockFieldOffset(compiled.spec, "d");
    const m_offset = blockFieldOffset(compiled.spec, "m");
    const qs_offset = blockFieldOffset(compiled.spec, "qs");
    const d_pointer = try blockPointerOffsetExpr(allocator, d_offset);
    defer allocator.free(d_pointer);
    const m_pointer = try blockPointerOffsetExpr(allocator, m_offset);
    defer allocator.free(m_pointer);
    const baseline_id = productionKernelId(
        compiled.artifact.backend,
        compiled.artifact.format,
        compiled.artifact.row_bucket,
        compiled.artifact.epilogue,
    );
    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try appendFmt(
        allocator,
        &out,
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
        \\
        \\// {s} from graph/quant_kernel_compiler.zig.
        \\// plan_id={s}
        \\// kernel_id={s}
        \\// production_baseline={s}
        \\// production_enabled={}
        \\{s}
        \\
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\static inline float antfly_half_le_to_float(const device uchar *p) {{
        \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
        \\    return (float)as_type<half>(bits);
        \\}}
        \\
        \\static inline float antfly_q4_1_dequant_lane(const device uchar *block, int lane) {{
        \\    const float d = antfly_half_le_to_float({s});
        \\    const float m = antfly_half_le_to_float({s});
        \\    const int packed_index = lane & 15;
        \\    const uchar packed = block[{d} + packed_index];
        \\    const int q = lane < 16 ? (int)(packed & 0x0fu) : (int)(packed >> 4);
        \\    return d * (float)q + m;
        \\}}
        \\
        \\kernel void {s}(
        \\    const device float *input [[buffer(0)]],
        \\    const device uchar *weight_q4_1 [[buffer(1)]],
        \\    device float *output [[buffer(2)]],
        \\    constant int &rows [[buffer(3)]],
        \\    constant int &in_dim [[buffer(4)]],
        \\    constant int &out_dim [[buffer(5)]],
        \\    uint3 thread_pos [[thread_position_in_threadgroup]],
        \\    uint3 group_pos [[threadgroup_position_in_grid]]
        \\) {{
        \\    const uint tid = thread_pos.x;
        \\    const int col0 = (int)(group_pos.x << 1);
        \\    const int col1 = col0 + 1;
        \\    const int row = (int)group_pos.y;
        \\    if (row >= rows || rows < 2 || rows > 8 || col0 >= out_dim || (in_dim & {d}) != 0) return;
        \\
        \\    float acc0 = 0.0f;
        \\    float acc1 = 0.0f;
        \\    const int block_count = in_dim >> {d};
        \\    const int lane = (int)tid;
        \\    const device float *row_input = input + row * in_dim;
        \\    const device uchar *col0_weight = weight_q4_1 + col0 * block_count * {d};
        \\    const bool has_col1 = col1 < out_dim;
        \\    const device uchar *col1_weight = has_col1 ? weight_q4_1 + col1 * block_count * {d} : col0_weight;
        \\    for (int block_idx = 0; block_idx < block_count; ++block_idx) {{
        \\        const float x = row_input[(block_idx << {d}) + lane];
        \\        const device uchar *block0 = col0_weight + block_idx * {d};
        \\        acc0 += x * antfly_q4_1_dequant_lane(block0, lane);
        \\        if (has_col1) {{
        \\            const device uchar *block1 = col1_weight + block_idx * {d};
        \\            acc1 += x * antfly_q4_1_dequant_lane(block1, lane);
        \\        }}
        \\    }}
        \\
        \\    acc0 = simd_sum(acc0);
        \\    acc1 = simd_sum(acc1);
        \\    if (tid == 0) {{
        \\        output[row * out_dim + col0] = acc0;
        \\        if (has_col1) output[row * out_dim + col1] = acc1;
        \\    }}
        \\}}
        \\
    ,
        .{
            if (compiled.artifact.production_enabled) "Generated Metal artifact source" else metalLegacyScalarSmallBatchSourceKind(compiled.request.format),
            plan_name,
            compiled.artifact.kernel_id,
            baseline_id,
            compiled.artifact.production_enabled,
            metalLegacyScalarSmallBatchPromotionComment(compiled.artifact),
            d_pointer,
            m_pointer,
            qs_offset,
            compiled.artifact.kernel_id,
            block_mask,
            block_shift,
            compiled.spec.block_bytes,
            compiled.spec.block_bytes,
            block_shift,
            compiled.spec.block_bytes,
            compiled.spec.block_bytes,
        },
    );

    return try out.toOwnedSlice(allocator);
}

fn emitMetalQ5_1SmallBatchSource(
    allocator: std.mem.Allocator,
    compiled: QuantKernelCompiledSource,
) ![]u8 {
    const plan_name = try planIdName(allocator, compiled.lowering.plan_id);
    defer allocator.free(plan_name);

    const block_shift = blockValueShift(compiled.spec.block_values);
    const block_mask = compiled.spec.block_values - 1;
    const d_offset = blockFieldOffset(compiled.spec, "d");
    const m_offset = blockFieldOffset(compiled.spec, "m");
    const qh_offset = blockFieldOffset(compiled.spec, "qh");
    const qs_offset = blockFieldOffset(compiled.spec, "qs");
    const d_pointer = try blockPointerOffsetExpr(allocator, d_offset);
    defer allocator.free(d_pointer);
    const m_pointer = try blockPointerOffsetExpr(allocator, m_offset);
    defer allocator.free(m_pointer);
    const qh_pointer = try blockPointerOffsetExpr(allocator, qh_offset);
    defer allocator.free(qh_pointer);
    const baseline_id = productionKernelId(
        compiled.artifact.backend,
        compiled.artifact.format,
        compiled.artifact.row_bucket,
        compiled.artifact.epilogue,
    );

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try appendFmt(
        allocator,
        &out,
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
        \\
        \\// {s} from graph/quant_kernel_compiler.zig.
        \\// plan_id={s}
        \\// kernel_id={s}
        \\// production_baseline={s}
        \\// production_enabled={}
        \\{s}
        \\
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\static inline float antfly_half_le_to_float(const device uchar *p) {{
        \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
        \\    return (float)as_type<half>(bits);
        \\}}
        \\
        \\static inline uint antfly_u32_le(const device uchar *p) {{
        \\    return (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
        \\}}
        \\
        \\static inline float antfly_q5_1_dequant_lane(const device uchar *block, int lane) {{
        \\    const float d = antfly_half_le_to_float({s});
        \\    const float m = antfly_half_le_to_float({s});
        \\    const uint qh = antfly_u32_le({s});
        \\    const int packed_index = lane & 15;
        \\    const uchar packed = block[{d} + packed_index];
        \\    const int low4 = lane < 16 ? (int)(packed & 0x0fu) : (int)(packed >> 4);
        \\    const int high = (int)((qh >> (uint)lane) & 1u);
        \\    return d * (float)(low4 | (high << 4)) + m;
        \\}}
        \\
        \\kernel void {s}(
        \\    const device float *input [[buffer(0)]],
        \\    const device uchar *weight_q5_1 [[buffer(1)]],
        \\    device float *output [[buffer(2)]],
        \\    constant int &rows [[buffer(3)]],
        \\    constant int &in_dim [[buffer(4)]],
        \\    constant int &out_dim [[buffer(5)]],
        \\    uint3 thread_pos [[thread_position_in_threadgroup]],
        \\    uint3 group_pos [[threadgroup_position_in_grid]]
        \\) {{
        \\    const uint tid = thread_pos.x;
        \\    const int col0 = (int)(group_pos.x << 1);
        \\    const int col1 = col0 + 1;
        \\    const int row = (int)group_pos.y;
        \\    if (row >= rows || rows < 2 || rows > 8 || col0 >= out_dim || (in_dim & {d}) != 0) return;
        \\
        \\    float acc0 = 0.0f;
        \\    float acc1 = 0.0f;
        \\    const int block_count = in_dim >> {d};
        \\    const int lane = (int)tid;
        \\    const device float *row_input = input + row * in_dim;
        \\    const device uchar *col0_weight = weight_q5_1 + col0 * block_count * {d};
        \\    const bool has_col1 = col1 < out_dim;
        \\    const device uchar *col1_weight = has_col1 ? weight_q5_1 + col1 * block_count * {d} : col0_weight;
        \\    for (int block_idx = 0; block_idx < block_count; ++block_idx) {{
        \\        const float x = row_input[(block_idx << {d}) + lane];
        \\        const device uchar *block0 = col0_weight + block_idx * {d};
        \\        acc0 += x * antfly_q5_1_dequant_lane(block0, lane);
        \\        if (has_col1) {{
        \\            const device uchar *block1 = col1_weight + block_idx * {d};
        \\            acc1 += x * antfly_q5_1_dequant_lane(block1, lane);
        \\        }}
        \\    }}
        \\
        \\    acc0 = simd_sum(acc0);
        \\    acc1 = simd_sum(acc1);
        \\    if (tid == 0) {{
        \\        output[row * out_dim + col0] = acc0;
        \\        if (has_col1) output[row * out_dim + col1] = acc1;
        \\    }}
        \\}}
        \\
    ,
        .{
            if (compiled.artifact.production_enabled) "Generated Metal artifact source" else metalLegacyScalarSmallBatchSourceKind(compiled.request.format),
            plan_name,
            compiled.artifact.kernel_id,
            baseline_id,
            compiled.artifact.production_enabled,
            metalLegacyScalarSmallBatchPromotionComment(compiled.artifact),
            d_pointer,
            m_pointer,
            qh_pointer,
            qs_offset,
            compiled.artifact.kernel_id,
            block_mask,
            block_shift,
            compiled.spec.block_bytes,
            compiled.spec.block_bytes,
            block_shift,
            compiled.spec.block_bytes,
            compiled.spec.block_bytes,
        },
    );

    return try out.toOwnedSlice(allocator);
}

fn emitMetalQ8FamilySmallBatchSource(
    allocator: std.mem.Allocator,
    compiled: QuantKernelCompiledSource,
) ![]u8 {
    const plan_name = try planIdName(allocator, compiled.lowering.plan_id);
    defer allocator.free(plan_name);

    const block_shift = blockValueShift(compiled.spec.block_values);
    const block_mask = compiled.spec.block_values - 1;
    const d_offset = blockFieldOffset(compiled.spec, "d");
    const qs_offset = blockFieldOffset(compiled.spec, "qs");
    const d_pointer = try blockPointerOffsetExpr(allocator, d_offset);
    defer allocator.free(d_pointer);
    const format_suffix = switch (compiled.request.format) {
        .q8_1 => "q8_1",
        .q8_k => "q8_k",
        else => return error.UnsupportedQuantKernelFormat,
    };
    const weight_name = switch (compiled.request.format) {
        .q8_1 => "weight_q8_1",
        .q8_k => "weight_q8_k",
        else => return error.UnsupportedQuantKernelFormat,
    };
    const baseline_id = productionKernelId(
        compiled.artifact.backend,
        compiled.artifact.format,
        compiled.artifact.row_bucket,
        compiled.artifact.epilogue,
    );
    const source_kind = if (compiled.artifact.production_enabled)
        "Generated Metal artifact source"
    else
        "Dev-only generated Metal candidate";
    const promotion_comment = if (compiled.artifact.production_enabled)
        "// Promoted after sequential Metal runtime evidence cleared correctness,\n// route, and speedup gates."
    else
        "// General MSL lowering smoke for descriptor-driven quant matmul.\n// Production Metal dispatch stays on native handwritten MSL until this\n// candidate clears correctness and benchmark gates.";

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try appendFmt(
        allocator,
        &out,
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
        \\
        \\// {s} from graph/quant_kernel_compiler.zig.
        \\// plan_id={s}
        \\// kernel_id={s}
        \\// production_baseline={s}
        \\// production_enabled={}
        \\{s}
        \\
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
    ,
        .{
            source_kind,
            plan_name,
            compiled.artifact.kernel_id,
            baseline_id,
            compiled.artifact.production_enabled,
            promotion_comment,
        },
    );
    try out.append(allocator, '\n');

    switch (compiled.request.format) {
        .q8_1 => try appendFmt(
            allocator,
            &out,
            \\static inline float antfly_half_le_to_float(const device uchar *p) {{
            \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
            \\    return (float)as_type<half>(bits);
            \\}}
            \\
            \\static inline float antfly_{s}_dequant_lane(const device uchar *block, int lane) {{
            \\    const float d = antfly_half_le_to_float({s});
            \\    const int q = (int)as_type<char>(block[{d} + lane]);
            \\    return d * (float)q;
            \\}}
            \\
        ,
            .{ format_suffix, d_pointer, qs_offset },
        ),
        .q8_k => try appendFmt(
            allocator,
            &out,
            \\static inline float antfly_f32_le_to_float(const device uchar *p) {{
            \\    const uint bits = (uint)p[0] | ((uint)p[1] << 8) | ((uint)p[2] << 16) | ((uint)p[3] << 24);
            \\    return as_type<float>(bits);
            \\}}
            \\
            \\static inline float antfly_{s}_dequant_lane(const device uchar *block, int lane) {{
            \\    const float d = antfly_f32_le_to_float({s});
            \\    const int q = (int)as_type<char>(block[{d} + lane]);
            \\    return d * (float)q;
            \\}}
            \\
        ,
            .{ format_suffix, d_pointer, qs_offset },
        ),
        else => return error.UnsupportedQuantKernelFormat,
    }
    try out.append(allocator, '\n');

    try appendFmt(
        allocator,
        &out,
        \\kernel void {s}(
        \\    const device float *input [[buffer(0)]],
        \\    const device uchar *{s} [[buffer(1)]],
        \\    device float *output [[buffer(2)]],
        \\    constant int &rows [[buffer(3)]],
        \\    constant int &in_dim [[buffer(4)]],
        \\    constant int &out_dim [[buffer(5)]],
        \\    uint3 thread_pos [[thread_position_in_threadgroup]],
        \\    uint3 group_pos [[threadgroup_position_in_grid]]
        \\) {{
        \\    const uint tid = thread_pos.x;
        \\    const int col = (int)group_pos.x;
        \\    const int row = (int)group_pos.y;
        \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & {d}) != 0) return;
        \\
        \\    float acc = 0.0f;
        \\    const int block_count = in_dim >> {d};
        \\    if (tid < 32) {{
        \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {{
        \\            const device uchar *block = {s} + ((col * block_count + block_idx) * {d});
    ,
        .{
            compiled.artifact.kernel_id,
            weight_name,
            block_mask,
            block_shift,
            weight_name,
            compiled.spec.block_bytes,
        },
    );

    if (compiled.spec.block_values == 32) {
        try appendFmt(
            allocator,
            &out,
            \\
            \\            const int lane = (int)tid;
            \\            acc += input[row * in_dim + (block_idx << {d}) + lane] * antfly_{s}_dequant_lane(block, lane);
        ,
            .{ block_shift, format_suffix },
        );
    } else {
        try appendFmt(
            allocator,
            &out,
            \\
            \\            for (int lane = (int)tid; lane < {d}; lane += 32) {{
            \\                acc += input[row * in_dim + (block_idx << {d}) + lane] * antfly_{s}_dequant_lane(block, lane);
            \\            }}
        ,
            .{ compiled.spec.block_values, block_shift, format_suffix },
        );
    }

    try appendFmt(
        allocator,
        &out,
        \\
        \\        }}
        \\    }}
        \\
        \\    acc = simd_sum(acc);
        \\    if (tid == 0) output[row * out_dim + col] = acc;
        \\}}
        \\
    ,
        .{},
    );

    return try out.toOwnedSlice(allocator);
}

fn emitMetalQ2KSmallBatchSource(
    allocator: std.mem.Allocator,
    compiled: QuantKernelCompiledSource,
) ![]u8 {
    const plan_name = try planIdName(allocator, compiled.lowering.plan_id);
    defer allocator.free(plan_name);

    const block_shift = blockValueShift(compiled.spec.block_values);
    const block_mask = compiled.spec.block_values - 1;
    const scales_offset = blockFieldOffset(compiled.spec, "scales");
    const d_offset = blockFieldOffset(compiled.spec, "d");
    const dmin_offset = blockFieldOffset(compiled.spec, "dmin");
    const qs_offset = blockFieldOffset(compiled.spec, "qs");
    const scale_byte_expr = if (scales_offset == 0)
        try allocator.dupe(u8, "block[sub]")
    else
        try std.fmt.allocPrint(allocator, "block[{d} + sub]", .{scales_offset});
    defer allocator.free(scale_byte_expr);
    const d_pointer = try blockPointerOffsetExpr(allocator, d_offset);
    defer allocator.free(d_pointer);
    const dmin_pointer = try blockPointerOffsetExpr(allocator, dmin_offset);
    defer allocator.free(dmin_pointer);
    const baseline_id = productionKernelId(
        compiled.artifact.backend,
        compiled.artifact.format,
        compiled.artifact.row_bucket,
        compiled.artifact.epilogue,
    );
    const output_expr = switch (compiled.ir.epilogue) {
        .none => "acc",
        .bias => "acc + bias[col]",
        .bias_gelu => "antfly_gelu(acc + bias[col])",
        else => return error.UnsupportedQuantKernelEpilogue,
    };

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try appendFmt(
        allocator,
        &out,
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
        \\
        \\// Dev-only generated Metal candidate from graph/quant_kernel_compiler.zig.
        \\// plan_id={s}
        \\// kernel_id={s}
        \\// production_baseline={s}
        \\// production_enabled={}
        \\{s}
        \\
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\static inline float antfly_half_le_to_float(const device uchar *p) {{
        \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
        \\    return (float)as_type<half>(bits);
        \\}}
        \\
    ,
        .{
            plan_name,
            compiled.artifact.kernel_id,
            baseline_id,
            compiled.artifact.production_enabled,
            metalQ2KSmallBatchPromotionComment(compiled.ir.epilogue),
        },
    );
    try out.append(allocator, '\n');

    if (compiled.ir.epilogue == .bias_gelu) {
        try appendFmt(
            allocator,
            &out,
            \\static inline float antfly_gelu(float x) {{
            \\    return 0.5f * x * (1.0f + fast::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
            \\}}
            \\
        ,
            .{},
        );
        try out.append(allocator, '\n');
    }

    try appendFmt(
        allocator,
        &out,
        \\static inline float antfly_q2_k_dequant_lane(const device uchar *block, int lane) {{
        \\    const uint sub = (uint)lane >> 4;
        \\    const uint i = (uint)lane & 15u;
        \\    const uchar scale_byte = {s};
        \\    const float dsc = antfly_half_le_to_float({s}) * (float)(scale_byte & 0x0Fu);
        \\    const float dmn = antfly_half_le_to_float({s}) * (float)(scale_byte >> 4);
        \\    const uint chunk = sub >> 3;
        \\    const uint group = (sub & 7u) >> 1;
        \\    const uint l_base = (sub & 1u) << 4;
        \\    const uint q_base = chunk << 5;
        \\    const uint shift = group << 1;
        \\    const uint q = ((uint)block[{d} + q_base + l_base + i] >> shift) & 0x03u;
        \\    return dsc * (float)q - dmn;
        \\}}
        \\
    ,
        .{
            scale_byte_expr,
            d_pointer,
            dmin_pointer,
            qs_offset,
        },
    );
    try out.append(allocator, '\n');

    switch (compiled.ir.epilogue) {
        .none => try appendFmt(
            allocator,
            &out,
            \\kernel void {s}(
            \\    const device float *input [[buffer(0)]],
            \\    const device uchar *weight_q2_k [[buffer(1)]],
            \\    device float *output [[buffer(2)]],
            \\    constant int &rows [[buffer(3)]],
            \\    constant int &in_dim [[buffer(4)]],
            \\    constant int &out_dim [[buffer(5)]],
            \\    uint3 thread_pos [[thread_position_in_threadgroup]],
            \\    uint3 group_pos [[threadgroup_position_in_grid]]
            \\) {{
            \\
        ,
            .{compiled.artifact.kernel_id},
        ),
        .bias, .bias_gelu => try appendFmt(
            allocator,
            &out,
            \\kernel void {s}(
            \\    const device float *input [[buffer(0)]],
            \\    const device uchar *weight_q2_k [[buffer(1)]],
            \\    const device float *bias [[buffer(2)]],
            \\    device float *output [[buffer(3)]],
            \\    constant int &rows [[buffer(4)]],
            \\    constant int &in_dim [[buffer(5)]],
            \\    constant int &out_dim [[buffer(6)]],
            \\    uint3 thread_pos [[thread_position_in_threadgroup]],
            \\    uint3 group_pos [[threadgroup_position_in_grid]]
            \\) {{
            \\
        ,
            .{compiled.artifact.kernel_id},
        ),
        else => return error.UnsupportedQuantKernelEpilogue,
    }

    try appendFmt(
        allocator,
        &out,
        \\    const uint tid = thread_pos.x;
        \\    const int col = (int)group_pos.x;
        \\    const int row = (int)group_pos.y;
        \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & {d}) != 0) return;
        \\
        \\    float acc = 0.0f;
        \\    const int block_count = in_dim >> {d};
        \\    if (tid < 32) {{
        \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {{
        \\            const device uchar *block = weight_q2_k + ((col * block_count + block_idx) * {d});
        \\            for (int lane = (int)tid; lane < {d}; lane += 32) {{
        \\                acc += input[row * in_dim + (block_idx << {d}) + lane] * antfly_q2_k_dequant_lane(block, lane);
        \\            }}
        \\        }}
        \\    }}
        \\
        \\    acc = simd_sum(acc);
        \\    if (tid == 0) output[row * out_dim + col] = {s};
        \\}}
        \\
    ,
        .{
            block_mask,
            block_shift,
            compiled.spec.block_bytes,
            compiled.spec.block_values,
            block_shift,
            output_expr,
        },
    );

    return try out.toOwnedSlice(allocator);
}

fn emitMetalQ3KSmallBatchSource(
    allocator: std.mem.Allocator,
    compiled: QuantKernelCompiledSource,
) ![]u8 {
    const plan_name = try planIdName(allocator, compiled.lowering.plan_id);
    defer allocator.free(plan_name);

    const block_shift = blockValueShift(compiled.spec.block_values);
    const block_mask = compiled.spec.block_values - 1;
    const hmask_offset = blockFieldOffset(compiled.spec, "hmask");
    const qs_offset = blockFieldOffset(compiled.spec, "qs");
    const scales_offset = blockFieldOffset(compiled.spec, "scales");
    const d_offset = blockFieldOffset(compiled.spec, "d");
    const hmask_expr = if (hmask_offset == 0)
        try allocator.dupe(u8, "block[l]")
    else
        try std.fmt.allocPrint(allocator, "block[{d} + l]", .{hmask_offset});
    defer allocator.free(hmask_expr);
    const scales_pointer = try blockPointerOffsetExpr(allocator, scales_offset);
    defer allocator.free(scales_pointer);
    const d_pointer = try blockPointerOffsetExpr(allocator, d_offset);
    defer allocator.free(d_pointer);
    const baseline_id = productionKernelId(
        compiled.artifact.backend,
        compiled.artifact.format,
        compiled.artifact.row_bucket,
        compiled.artifact.epilogue,
    );
    const output_expr = switch (compiled.ir.epilogue) {
        .none => "acc",
        .bias => "acc + bias[col]",
        .bias_gelu => "antfly_gelu(acc + bias[col])",
        else => return error.UnsupportedQuantKernelEpilogue,
    };

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try appendFmt(
        allocator,
        &out,
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
        \\
        \\// {s} from graph/quant_kernel_compiler.zig.
        \\// plan_id={s}
        \\// kernel_id={s}
        \\// production_baseline={s}
        \\// production_enabled={}
        \\{s}
        \\
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\static inline float antfly_half_le_to_float(const device uchar *p) {{
        \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
        \\    return (float)as_type<half>(bits);
        \\}}
        \\
    ,
        .{
            metalQ3KSmallBatchSourceKind(compiled.ir.epilogue),
            plan_name,
            compiled.artifact.kernel_id,
            baseline_id,
            compiled.artifact.production_enabled,
            metalQ3KSmallBatchPromotionComment(compiled.ir.epilogue),
        },
    );
    try out.append(allocator, '\n');

    if (compiled.ir.epilogue == .bias_gelu) {
        try appendFmt(
            allocator,
            &out,
            \\static inline float antfly_gelu(float x) {{
            \\    return 0.5f * x * (1.0f + fast::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
            \\}}
            \\
        ,
            .{},
        );
        try out.append(allocator, '\n');
    }

    try appendFmt(
        allocator,
        &out,
        \\static inline int antfly_q3_k_raw_scale(const device uchar *scale_data, uint sub) {{
        \\    const uint i = sub & 3u;
        \\    uint low = 0u;
        \\    uint high = 0u;
        \\    if (sub < 4u) {{
        \\        low = (uint)(scale_data[i] & 0x0Fu);
        \\        high = (uint)(scale_data[8 + i] & 0x03u);
        \\    }} else if (sub < 8u) {{
        \\        low = (uint)(scale_data[4 + i] & 0x0Fu);
        \\        high = (uint)((scale_data[8 + i] >> 2) & 0x03u);
        \\    }} else if (sub < 12u) {{
        \\        low = (uint)((scale_data[i] >> 4) & 0x0Fu);
        \\        high = (uint)((scale_data[8 + i] >> 4) & 0x03u);
        \\    }} else {{
        \\        low = (uint)((scale_data[4 + i] >> 4) & 0x0Fu);
        \\        high = (uint)((scale_data[8 + i] >> 6) & 0x03u);
        \\    }}
        \\    return (int)(low | (high << 4)) - 32;
        \\}}
        \\
        \\static inline float antfly_q3_k_dequant_lane(const device uchar *block, int lane) {{
        \\    const uint sub = (uint)lane >> 4;
        \\    const uint i = (uint)lane & 15u;
        \\    const uint chunk = sub >> 3;
        \\    const uint group = (sub & 7u) >> 1;
        \\    const uint l_base = (sub & 1u) << 4;
        \\    const uint l = l_base + i;
        \\    const uint q_base = chunk << 5;
        \\    const uint shift = group << 1;
        \\    const uint hm_bit = (chunk << 2) + group;
        \\    const int low2 = (int)(((uint)block[{d} + q_base + l] >> shift) & 0x03u);
        \\    const int high1 = (int)(((uint){s} >> hm_bit) & 0x01u);
        \\    const int q = low2 + high1 * 4 - 4;
        \\    const float scale = antfly_half_le_to_float({s}) * (float)antfly_q3_k_raw_scale({s}, sub);
        \\    return scale * (float)q;
        \\}}
        \\
    ,
        .{
            qs_offset,
            hmask_expr,
            d_pointer,
            scales_pointer,
        },
    );
    try out.append(allocator, '\n');

    switch (compiled.ir.epilogue) {
        .none => try appendFmt(
            allocator,
            &out,
            \\kernel void {s}(
            \\    const device float *input [[buffer(0)]],
            \\    const device uchar *weight_q3_k [[buffer(1)]],
            \\    device float *output [[buffer(2)]],
            \\    constant int &rows [[buffer(3)]],
            \\    constant int &in_dim [[buffer(4)]],
            \\    constant int &out_dim [[buffer(5)]],
            \\    uint3 thread_pos [[thread_position_in_threadgroup]],
            \\    uint3 group_pos [[threadgroup_position_in_grid]]
            \\) {{
            \\
        ,
            .{compiled.artifact.kernel_id},
        ),
        .bias, .bias_gelu => try appendFmt(
            allocator,
            &out,
            \\kernel void {s}(
            \\    const device float *input [[buffer(0)]],
            \\    const device uchar *weight_q3_k [[buffer(1)]],
            \\    const device float *bias [[buffer(2)]],
            \\    device float *output [[buffer(3)]],
            \\    constant int &rows [[buffer(4)]],
            \\    constant int &in_dim [[buffer(5)]],
            \\    constant int &out_dim [[buffer(6)]],
            \\    uint3 thread_pos [[thread_position_in_threadgroup]],
            \\    uint3 group_pos [[threadgroup_position_in_grid]]
            \\) {{
            \\
        ,
            .{compiled.artifact.kernel_id},
        ),
        else => return error.UnsupportedQuantKernelEpilogue,
    }

    try appendFmt(
        allocator,
        &out,
        \\    const uint tid = thread_pos.x;
        \\    const int col = (int)group_pos.x;
        \\    const int row = (int)group_pos.y;
        \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & {d}) != 0) return;
        \\
        \\    float acc = 0.0f;
        \\    const int block_count = in_dim >> {d};
        \\    if (tid < 32) {{
        \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {{
        \\            const device uchar *block = weight_q3_k + ((col * block_count + block_idx) * {d});
        \\            for (int lane = (int)tid; lane < {d}; lane += 32) {{
        \\                acc += input[row * in_dim + (block_idx << {d}) + lane] * antfly_q3_k_dequant_lane(block, lane);
        \\            }}
        \\        }}
        \\    }}
        \\
        \\    acc = simd_sum(acc);
        \\    if (tid == 0) output[row * out_dim + col] = {s};
        \\}}
        \\
    ,
        .{
            block_mask,
            block_shift,
            compiled.spec.block_bytes,
            compiled.spec.block_values,
            block_shift,
            output_expr,
        },
    );

    return try out.toOwnedSlice(allocator);
}

fn emitMetalQ4KSmallBatchSource(
    allocator: std.mem.Allocator,
    compiled: QuantKernelCompiledSource,
) ![]u8 {
    const plan_name = try planIdName(allocator, compiled.lowering.plan_id);
    defer allocator.free(plan_name);

    const block_shift = blockValueShift(compiled.spec.block_values);
    const block_mask = compiled.spec.block_values - 1;
    const d_offset = blockFieldOffset(compiled.spec, "d");
    const dmin_offset = blockFieldOffset(compiled.spec, "dmin");
    const scales_offset = blockFieldOffset(compiled.spec, "scales");
    const qs_offset = blockFieldOffset(compiled.spec, "qs");
    const d_pointer = try blockPointerOffsetExpr(allocator, d_offset);
    defer allocator.free(d_pointer);
    const dmin_pointer = try blockPointerOffsetExpr(allocator, dmin_offset);
    defer allocator.free(dmin_pointer);
    const thread_count: usize = 64;
    const reduction_start = thread_count / 2;
    const baseline_id = productionKernelId(
        compiled.artifact.backend,
        compiled.artifact.format,
        compiled.artifact.row_bucket,
        compiled.artifact.epilogue,
    );
    const output_expr = switch (compiled.ir.epilogue) {
        .none => "partial[0]",
        .bias => "partial[0] + bias[col]",
        .bias_gelu => "antfly_gelu(partial[0] + bias[col])",
        else => return error.UnsupportedQuantKernelEpilogue,
    };

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try appendFmt(
        allocator,
        &out,
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
        \\
        \\// Generated Metal candidate artifact from graph/quant_kernel_compiler.zig.
        \\// plan_id={s}
        \\// kernel_id={s}
        \\// production_baseline={s}
        \\// production_enabled={}
        \\{s}
        \\
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\static inline float antfly_half_le_to_float(const device uchar *p) {{
        \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
        \\    return (float)as_type<half>(bits);
        \\}}
        \\
    ,
        .{
            plan_name,
            compiled.artifact.kernel_id,
            baseline_id,
            compiled.artifact.production_enabled,
            metalQ4KSmallBatchPromotionComment(compiled.artifact, compiled.ir.epilogue),
        },
    );
    try out.append(allocator, '\n');

    if (compiled.ir.epilogue == .bias_gelu) {
        try appendFmt(
            allocator,
            &out,
            \\static inline float antfly_gelu(float x) {{
            \\    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
            \\    return 0.5f * x * (1.0f + fast::tanh(inner));
            \\}}
            \\
        ,
            .{},
        );
        try out.append(allocator, '\n');
    }

    try appendFmt(
        allocator,
        &out,
        \\static inline void antfly_q4_k_unpack_scale_min(
        \\    const device uchar *scales,
        \\    int sub,
        \\    thread float &scale,
        \\    thread float &min_v
        \\) {{
        \\    if (sub < 4) {{
        \\        scale = (float)(scales[sub] & 63u);
        \\        min_v = (float)(scales[sub + 4] & 63u);
        \\        return;
        \\    }}
        \\    scale = (float)((scales[sub + 4] & 0x0fu) | ((scales[sub - 4] >> 6) << 4));
        \\    min_v = (float)((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
        \\}}
        \\
        \\static inline float antfly_q4_k_dequant_lane(const device uchar *block, int lane) {{
        \\    const device uchar *d = {s};
        \\    const device uchar *dmin = {s};
        \\    const device uchar *scales = block + {d};
        \\    const device uchar *qs = block + {d};
        \\    const int sub = lane >> 5;
        \\    const int q_index = (sub >> 1) * 32 + (lane & 31);
        \\    const uchar packed = qs[q_index];
        \\    const uchar q = (sub & 1) == 0 ? (packed & 0x0fu) : (packed >> 4);
        \\    float raw_scale = 0.0f;
        \\    float raw_min = 0.0f;
        \\    antfly_q4_k_unpack_scale_min(scales, sub, raw_scale, raw_min);
        \\    return antfly_half_le_to_float(d) * raw_scale * (float)q - antfly_half_le_to_float(dmin) * raw_min;
        \\}}
        \\
    ,
        .{
            d_pointer,
            dmin_pointer,
            scales_offset,
            qs_offset,
        },
    );
    try out.append(allocator, '\n');

    switch (compiled.ir.epilogue) {
        .none => try appendFmt(
            allocator,
            &out,
            \\kernel void {s}(
            \\    const device float *input [[buffer(0)]],
            \\    const device uchar *weight_q4_k [[buffer(1)]],
            \\    device float *output [[buffer(2)]],
            \\    constant int &rows [[buffer(3)]],
            \\    constant int &in_dim [[buffer(4)]],
            \\    constant int &out_dim [[buffer(5)]],
            \\    uint3 thread_pos [[thread_position_in_threadgroup]],
            \\    uint3 group_pos [[threadgroup_position_in_grid]]
            \\) {{
            \\
        ,
            .{compiled.artifact.kernel_id},
        ),
        .bias => try appendFmt(
            allocator,
            &out,
            \\kernel void {s}(
            \\    const device float *input [[buffer(0)]],
            \\    const device uchar *weight_q4_k [[buffer(1)]],
            \\    const device float *bias [[buffer(2)]],
            \\    device float *output [[buffer(3)]],
            \\    constant int &rows [[buffer(4)]],
            \\    constant int &in_dim [[buffer(5)]],
            \\    constant int &out_dim [[buffer(6)]],
            \\    uint3 thread_pos [[thread_position_in_threadgroup]],
            \\    uint3 group_pos [[threadgroup_position_in_grid]],
            \\    ushort lane_id [[thread_index_in_simdgroup]],
            \\    ushort simdgroup_id [[simdgroup_index_in_threadgroup]]
            \\) {{
            \\
        ,
            .{compiled.artifact.kernel_id},
        ),
        .bias_gelu => try appendFmt(
            allocator,
            &out,
            \\kernel void {s}(
            \\    const device float *input [[buffer(0)]],
            \\    const device uchar *weight_q4_k [[buffer(1)]],
            \\    const device float *bias [[buffer(2)]],
            \\    device float *output [[buffer(3)]],
            \\    constant int &rows [[buffer(4)]],
            \\    constant int &in_dim [[buffer(5)]],
            \\    constant int &out_dim [[buffer(6)]],
            \\    uint3 thread_pos [[thread_position_in_threadgroup]],
            \\    uint3 group_pos [[threadgroup_position_in_grid]]
            \\) {{
            \\
        ,
            .{compiled.artifact.kernel_id},
        ),
        else => return error.UnsupportedQuantKernelEpilogue,
    }

    try appendFmt(
        allocator,
        &out,
        \\    const uint tid = thread_pos.x;
        \\    const int col = (int)group_pos.x;
        \\    const int row = (int)group_pos.y;
        \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & {d}) != 0) return;
        \\
    ,
        .{block_mask},
    );
    try out.append(allocator, '\n');

    if (compiled.ir.epilogue == .none) {
        try appendFmt(
            allocator,
            &out,
            \\    threadgroup float partial[{d}];
            \\    float acc = 0.0f;
            \\
        ,
            .{thread_count},
        );
    } else {
        try appendFmt(
            allocator,
            &out,
            \\    float acc = 0.0f;
            \\
        ,
            .{},
        );
    }

    try appendFmt(
        allocator,
        &out,
        \\    const int block_count = in_dim >> {d};
        \\    if (tid < {d}) {{
        \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {{
        \\            const device uchar *block = weight_q4_k + ((col * block_count + block_idx) * {d});
        \\            const int base = block_idx << {d};
        \\            for (int lane = (int)tid; lane < {d}; lane += {d}) {{
        \\                acc += input[row * in_dim + base + lane] * antfly_q4_k_dequant_lane(block, lane);
        \\            }}
        \\        }}
        \\    }}
        \\
    ,
        .{
            block_shift,
            thread_count,
            compiled.spec.block_bytes,
            block_shift,
            compiled.spec.block_values,
            thread_count,
        },
    );
    try out.append(allocator, '\n');

    if (compiled.ir.epilogue != .none) {
        try appendFmt(
            allocator,
            &out,
            \\    threadgroup float partial[{d}];
            \\
        ,
            .{thread_count},
        );
    }

    try appendFmt(
        allocator,
        &out,
        \\    if (tid < {d}) partial[tid] = acc;
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    for (uint stride = {d}; stride > 0; stride >>= 1) {{
        \\        if (tid < stride) partial[tid] += partial[tid + stride];
        \\        threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    }}
        \\    if (tid == 0) output[row * out_dim + col] = {s};
        \\}}
        \\
    ,
        .{
            thread_count,
            reduction_start,
            output_expr,
        },
    );

    return try out.toOwnedSlice(allocator);
}

fn emitMetalQ5KSmallBatchBiasSource(
    allocator: std.mem.Allocator,
    compiled: QuantKernelCompiledSource,
) ![]u8 {
    const plan_name = try planIdName(allocator, compiled.lowering.plan_id);
    defer allocator.free(plan_name);

    const block_shift = blockValueShift(compiled.spec.block_values);
    const block_mask = compiled.spec.block_values - 1;
    const d_offset = blockFieldOffset(compiled.spec, "d");
    const dmin_offset = blockFieldOffset(compiled.spec, "dmin");
    const scales_offset = blockFieldOffset(compiled.spec, "scales");
    const qh_offset = blockFieldOffset(compiled.spec, "qh");
    const ql_offset = blockFieldOffset(compiled.spec, "ql");
    const d_pointer = try blockPointerOffsetExpr(allocator, d_offset);
    defer allocator.free(d_pointer);
    const thread_count: usize = 128;
    const simd_width: usize = 32;
    const active_simdgroups = thread_count / simd_width;
    const baseline_id = productionKernelId(
        compiled.artifact.backend,
        compiled.artifact.format,
        compiled.artifact.row_bucket,
        compiled.artifact.epilogue,
    );
    const output_expr = switch (compiled.ir.epilogue) {
        .bias => "total + bias[col]",
        .bias_gelu => "antfly_gelu(total + bias[col])",
        else => return error.UnsupportedQuantKernelEpilogue,
    };

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try appendFmt(
        allocator,
        &out,
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
        \\
        \\// Generated Metal artifact source from graph/quant_kernel_compiler.zig.
        \\// plan_id={s}
        \\// kernel_id={s}
        \\// production_baseline={s}
        \\// production_enabled={}
        \\{s}
        \\
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\static inline float antfly_half_le_to_float(const device uchar *p) {{
        \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
        \\    return (float)as_type<half>(bits);
        \\}}
        \\
    ,
        .{
            plan_name,
            compiled.artifact.kernel_id,
            baseline_id,
            compiled.artifact.production_enabled,
            metalQ5KSmallBatchPromotionComment(compiled.artifact, compiled.ir.epilogue),
        },
    );
    try out.append(allocator, '\n');

    if (compiled.ir.epilogue == .bias_gelu) {
        try appendFmt(
            allocator,
            &out,
            \\static inline float antfly_gelu(float x) {{
            \\    return 0.5f * x * (1.0f + fast::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
            \\}}
            \\
        ,
            .{},
        );
        try out.append(allocator, '\n');
    }

    try appendFmt(
        allocator,
        &out,
        \\static inline void antfly_q5_k_unpack_scale_min(
        \\    const device uchar *scales,
        \\    int sub,
        \\    thread float &scale,
        \\    thread float &min_v
        \\) {{
        \\    if (sub < 4) {{
        \\        scale = (float)(scales[sub] & 63u);
        \\        min_v = (float)(scales[sub + 4] & 63u);
        \\        return;
        \\    }}
        \\    scale = (float)((scales[sub + 4] & 0x0fu) | ((scales[sub - 4] >> 6) << 4));
        \\    min_v = (float)((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
        \\}}
        \\
        \\static inline float antfly_q5_k_dequant_lane(const device uchar *block, int lane, float d, float dmin) {{
        \\    const device uchar *scales = block + {d};
        \\    const device uchar *qh = block + {d};
        \\    const device uchar *ql = block + {d};
        \\    const int sub = lane >> 5;
        \\    const int i = lane & 31;
        \\    const int q_index = (sub >> 1) * 32 + i;
        \\    const uchar packed = ql[q_index];
        \\    const int low = (sub & 1) == 0 ? (int)(packed & 0x0fu) : (int)(packed >> 4);
        \\    const int high = (int)((qh[i] >> sub) & 1u);
        \\    const int q = low + high * 16;
        \\    float raw_scale = 0.0f;
        \\    float raw_min = 0.0f;
        \\    antfly_q5_k_unpack_scale_min(scales, sub, raw_scale, raw_min);
        \\    return d * raw_scale * (float)q - dmin * raw_min;
        \\}}
        \\
        \\kernel void {s}(
        \\    const device float *input [[buffer(0)]],
        \\    const device uchar *weight_q5_k [[buffer(1)]],
        \\    const device float *bias [[buffer(2)]],
        \\    device float *output [[buffer(3)]],
        \\    constant int &rows [[buffer(4)]],
        \\    constant int &in_dim [[buffer(5)]],
        \\    constant int &out_dim [[buffer(6)]],
        \\    uint3 thread_pos [[thread_position_in_threadgroup]],
        \\    uint3 group_pos [[threadgroup_position_in_grid]],
        \\    ushort lane_id [[thread_index_in_simdgroup]],
        \\    ushort simdgroup_id [[simdgroup_index_in_threadgroup]]
        \\) {{
        \\    const uint tid = thread_pos.x;
        \\    const int col = (int)group_pos.x;
        \\    const int row = (int)group_pos.y;
        \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & {d}) != 0) return;
        \\
        \\    float acc = 0.0f;
        \\    const int block_count = in_dim >> {d};
        \\    if (tid < {d}) {{
        \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {{
        \\            const device uchar *block = weight_q5_k + ((col * block_count + block_idx) * {d});
        \\            const int base = block_idx << {d};
        \\            const float d = antfly_half_le_to_float({s});
        \\            const float dmin = antfly_half_le_to_float(block + {d});
        \\            for (int lane = (int)tid; lane < {d}; lane += {d}) {{
        \\                acc += input[row * in_dim + base + lane] * antfly_q5_k_dequant_lane(block, lane, d, dmin);
        \\            }}
        \\        }}
        \\    }}
        \\
        \\    threadgroup float partial[{d}];
        \\    acc = simd_sum(acc);
        \\    if (lane_id == 0u) partial[simdgroup_id] = acc;
        \\    if (simdgroup_id == 0u && lane_id >= {d}u) partial[lane_id] = 0.0f;
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    const float total = simd_sum(partial[lane_id]);
        \\    if (lane_id == 0u && simdgroup_id == 0u) output[row * out_dim + col] = {s};
        \\}}
        \\
    ,
        .{
            scales_offset,
            qh_offset,
            ql_offset,
            compiled.artifact.kernel_id,
            block_mask,
            block_shift,
            thread_count,
            compiled.spec.block_bytes,
            block_shift,
            d_pointer,
            dmin_offset,
            compiled.spec.block_values,
            thread_count,
            simd_width,
            active_simdgroups,
            output_expr,
        },
    );

    return try out.toOwnedSlice(allocator);
}

fn emitMetalQ5KSmallBatchSource(
    allocator: std.mem.Allocator,
    compiled: QuantKernelCompiledSource,
) ![]u8 {
    const plan_name = try planIdName(allocator, compiled.lowering.plan_id);
    defer allocator.free(plan_name);

    const block_shift = blockValueShift(compiled.spec.block_values);
    const block_mask = compiled.spec.block_values - 1;
    const d_offset = blockFieldOffset(compiled.spec, "d");
    const dmin_offset = blockFieldOffset(compiled.spec, "dmin");
    const scales_offset = blockFieldOffset(compiled.spec, "scales");
    const qh_offset = blockFieldOffset(compiled.spec, "qh");
    const ql_offset = blockFieldOffset(compiled.spec, "ql");
    const d_pointer = try blockPointerOffsetExpr(allocator, d_offset);
    defer allocator.free(d_pointer);
    const thread_count: usize = 64;
    const reduction_start = thread_count / 2;
    const baseline_id = productionKernelId(
        compiled.artifact.backend,
        compiled.artifact.format,
        compiled.artifact.row_bucket,
        compiled.artifact.epilogue,
    );

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try appendFmt(
        allocator,
        &out,
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
        \\
        \\// Generated Metal artifact source from graph/quant_kernel_compiler.zig.
        \\// plan_id={s}
        \\// kernel_id={s}
        \\// production_baseline={s}
        \\// production_enabled={}
        \\{s}
        \\
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\static inline float antfly_half_le_to_float(const device uchar *p) {{
        \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
        \\    return (float)as_type<half>(bits);
        \\}}
        \\
        \\static inline void antfly_q5_k_unpack_scale_min(
        \\    const device uchar *scales,
        \\    int sub,
        \\    thread float &scale,
        \\    thread float &min_v
        \\) {{
        \\    if (sub < 4) {{
        \\        scale = (float)(scales[sub] & 63u);
        \\        min_v = (float)(scales[sub + 4] & 63u);
        \\        return;
        \\    }}
        \\    scale = (float)((scales[sub + 4] & 0x0fu) | ((scales[sub - 4] >> 6) << 4));
        \\    min_v = (float)((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4));
        \\}}
        \\
        \\static inline float antfly_q5_k_dequant_lane(const device uchar *block, int lane, float d, float dmin) {{
        \\    const device uchar *scales = block + {d};
        \\    const device uchar *qh = block + {d};
        \\    const device uchar *ql = block + {d};
        \\    const int sub = lane >> 5;
        \\    const int i = lane & 31;
        \\    const int q_index = (sub >> 1) * 32 + i;
        \\    const uchar packed = ql[q_index];
        \\    const int low = (sub & 1) == 0 ? (int)(packed & 0x0fu) : (int)(packed >> 4);
        \\    const int high = (int)((qh[i] >> sub) & 1u);
        \\    const int q = low + high * 16;
        \\    float raw_scale = 0.0f;
        \\    float raw_min = 0.0f;
        \\    antfly_q5_k_unpack_scale_min(scales, sub, raw_scale, raw_min);
        \\    return d * raw_scale * (float)q - dmin * raw_min;
        \\}}
        \\
        \\kernel void {s}(
        \\    const device float *input [[buffer(0)]],
        \\    const device uchar *weight_q5_k [[buffer(1)]],
        \\    device float *output [[buffer(2)]],
        \\    constant int &rows [[buffer(3)]],
        \\    constant int &in_dim [[buffer(4)]],
        \\    constant int &out_dim [[buffer(5)]],
        \\    uint3 thread_pos [[thread_position_in_threadgroup]],
        \\    uint3 group_pos [[threadgroup_position_in_grid]]
        \\) {{
        \\    const uint tid = thread_pos.x;
        \\    const int col = (int)group_pos.x;
        \\    const int row = (int)group_pos.y;
        \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & {d}) != 0) return;
        \\
        \\    float acc = 0.0f;
        \\    const int block_count = in_dim >> {d};
        \\    if (tid < {d}) {{
        \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {{
        \\            const device uchar *block = weight_q5_k + ((col * block_count + block_idx) * {d});
        \\            const int base = block_idx << {d};
        \\            const float d = antfly_half_le_to_float({s});
        \\            const float dmin = antfly_half_le_to_float(block + {d});
        \\            for (int lane = (int)tid; lane < {d}; lane += {d}) {{
        \\                acc += input[row * in_dim + base + lane] * antfly_q5_k_dequant_lane(block, lane, d, dmin);
        \\            }}
        \\        }}
        \\    }}
        \\
        \\    threadgroup float partial[{d}];
        \\    if (tid < {d}) partial[tid] = acc;
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    for (uint stride = {d}; stride > 0; stride >>= 1) {{
        \\        if (tid < stride) partial[tid] += partial[tid + stride];
        \\        threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    }}
        \\    if (tid == 0) output[row * out_dim + col] = partial[0];
        \\}}
        \\
    ,
        .{
            plan_name,
            compiled.artifact.kernel_id,
            baseline_id,
            compiled.artifact.production_enabled,
            metalQ5KSmallBatchPromotionComment(compiled.artifact, compiled.ir.epilogue),
            scales_offset,
            qh_offset,
            ql_offset,
            compiled.artifact.kernel_id,
            block_mask,
            block_shift,
            thread_count,
            compiled.spec.block_bytes,
            block_shift,
            d_pointer,
            dmin_offset,
            compiled.spec.block_values,
            thread_count,
            thread_count,
            thread_count,
            reduction_start,
        },
    );

    return try out.toOwnedSlice(allocator);
}

fn emitMetalQ6KSmallBatchSource(
    allocator: std.mem.Allocator,
    compiled: QuantKernelCompiledSource,
) ![]u8 {
    const plan_name = try planIdName(allocator, compiled.lowering.plan_id);
    defer allocator.free(plan_name);

    const block_shift = blockValueShift(compiled.spec.block_values);
    const block_mask = compiled.spec.block_values - 1;
    const qh_offset = blockFieldOffset(compiled.spec, "qh");
    const scales_offset = blockFieldOffset(compiled.spec, "scales");
    const d_offset = blockFieldOffset(compiled.spec, "d");
    const thread_count: usize = 128;
    const reduction_start = thread_count / 2;
    const baseline_id = productionKernelId(
        compiled.artifact.backend,
        compiled.artifact.format,
        compiled.artifact.row_bucket,
        compiled.artifact.epilogue,
    );

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try appendFmt(
        allocator,
        &out,
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
        \\
        \\// Generated Metal candidate artifact from graph/quant_kernel_compiler.zig.
        \\// plan_id={s}
        \\// kernel_id={s}
        \\// production_baseline={s}
        \\// production_enabled={}
        \\{s}
        \\
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\static inline float antfly_half_le_to_float(const device uchar *p) {{
        \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
        \\    return (float)as_type<half>(bits);
        \\}}
        \\
        \\static inline float antfly_q6_k_dequant_lane(const device uchar *block, int lane, float d) {{
        \\    const device uchar *ql = block;
        \\    const device uchar *qh = block + {d};
        \\    const device uchar *scales = block + {d};
        \\    const int sub = lane >> 4;
        \\    const int i = lane & 15;
        \\    const int half_idx = sub >> 3;
        \\    const int group = (sub & 7) >> 1;
        \\    const int l = ((sub & 1) << 4) + i;
        \\    const int ql_off = half_idx * 64 + (group & 1) * 32;
        \\    const int qh_off = half_idx * 32;
        \\    const int qh_shift = group * 2;
        \\    const int nibble_shift = (group >> 1) * 4;
        \\    const int low4 = (int)((ql[ql_off + l] >> nibble_shift) & 0x0fu);
        \\    const int high2 = (int)((qh[qh_off + l] >> qh_shift) & 0x03u);
        \\    const int q = (low4 | (high2 << 4)) - 32;
        \\    const int scale_u = (int)scales[sub];
        \\    const int scale = scale_u >= 128 ? scale_u - 256 : scale_u;
        \\    return d * (float)scale * (float)q;
        \\}}
        \\
        \\kernel void {s}(
        \\    const device float *input [[buffer(0)]],
        \\    const device uchar *weight_q6_k [[buffer(1)]],
        \\    device float *output [[buffer(2)]],
        \\    constant int &rows [[buffer(3)]],
        \\    constant int &in_dim [[buffer(4)]],
        \\    constant int &out_dim [[buffer(5)]],
        \\    uint3 thread_pos [[thread_position_in_threadgroup]],
        \\    uint3 group_pos [[threadgroup_position_in_grid]]
        \\) {{
        \\    const uint tid = thread_pos.x;
        \\    const int col = (int)group_pos.x;
        \\    const int row = (int)group_pos.y;
        \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & {d}) != 0) return;
        \\
        \\    float acc = 0.0f;
        \\    const int block_count = in_dim >> {d};
        \\    if (tid < {d}) {{
        \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {{
        \\            const device uchar *block = weight_q6_k + ((col * block_count + block_idx) * {d});
        \\            const int base = block_idx << {d};
        \\            const float d = antfly_half_le_to_float(block + {d});
        \\            for (int lane = (int)tid; lane < {d}; lane += {d}) {{
        \\                acc += input[row * in_dim + base + lane] * antfly_q6_k_dequant_lane(block, lane, d);
        \\            }}
        \\        }}
        \\    }}
        \\
        \\    threadgroup float partial[{d}];
        \\    if (tid < {d}) partial[tid] = acc;
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    for (uint stride = {d}; stride > 0; stride >>= 1) {{
        \\        if (tid < stride) partial[tid] += partial[tid + stride];
        \\        threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    }}
        \\    if (tid == 0) output[row * out_dim + col] = partial[0];
        \\}}
        \\
    ,
        .{
            plan_name,
            compiled.artifact.kernel_id,
            baseline_id,
            compiled.artifact.production_enabled,
            metalQ6KSmallBatchPromotionComment(compiled.artifact, compiled.ir.epilogue),
            qh_offset,
            scales_offset,
            compiled.artifact.kernel_id,
            block_mask,
            block_shift,
            thread_count,
            compiled.spec.block_bytes,
            block_shift,
            d_offset,
            compiled.spec.block_values,
            thread_count,
            thread_count,
            thread_count,
            reduction_start,
        },
    );

    return try out.toOwnedSlice(allocator);
}

fn emitMetalQ6KSmallBatchBiasSource(
    allocator: std.mem.Allocator,
    compiled: QuantKernelCompiledSource,
) ![]u8 {
    const plan_name = try planIdName(allocator, compiled.lowering.plan_id);
    defer allocator.free(plan_name);

    const block_shift = blockValueShift(compiled.spec.block_values);
    const block_mask = compiled.spec.block_values - 1;
    const qh_offset = blockFieldOffset(compiled.spec, "qh");
    const scales_offset = blockFieldOffset(compiled.spec, "scales");
    const d_offset = blockFieldOffset(compiled.spec, "d");
    const thread_count: usize = 128;
    const simd_width: usize = 32;
    const active_simdgroups = thread_count / simd_width;
    const baseline_id = productionKernelId(
        compiled.artifact.backend,
        compiled.artifact.format,
        compiled.artifact.row_bucket,
        compiled.artifact.epilogue,
    );
    const output_expr = switch (compiled.ir.epilogue) {
        .bias => "total + bias[col]",
        .bias_gelu => "antfly_gelu(total + bias[col])",
        else => return error.UnsupportedQuantKernelEpilogue,
    };

    var out = std.ArrayListUnmanaged(u8).empty;
    errdefer out.deinit(allocator);
    try appendFmt(
        allocator,
        &out,
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
        \\
        \\// Generated Metal candidate artifact from graph/quant_kernel_compiler.zig.
        \\// plan_id={s}
        \\// kernel_id={s}
        \\// production_baseline={s}
        \\// production_enabled={}
        \\{s}
        \\
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\
        \\static inline float antfly_half_le_to_float(const device uchar *p) {{
        \\    const ushort bits = (ushort)p[0] | ((ushort)p[1] << 8);
        \\    return (float)as_type<half>(bits);
        \\}}
        \\
    ,
        .{
            plan_name,
            compiled.artifact.kernel_id,
            baseline_id,
            compiled.artifact.production_enabled,
            metalQ6KSmallBatchPromotionComment(compiled.artifact, compiled.ir.epilogue),
        },
    );
    try out.append(allocator, '\n');

    if (compiled.ir.epilogue == .bias_gelu) {
        try appendFmt(
            allocator,
            &out,
            \\static inline float antfly_gelu(float x) {{
            \\    return 0.5f * x * (1.0f + fast::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
            \\}}
            \\
        ,
            .{},
        );
        try out.append(allocator, '\n');
    }

    try appendFmt(
        allocator,
        &out,
        \\static inline float antfly_q6_k_dequant_lane(const device uchar *block, int lane, float d) {{
        \\    const device uchar *ql = block;
        \\    const device uchar *qh = block + {d};
        \\    const device uchar *scales = block + {d};
        \\    const int sub = lane >> 4;
        \\    const int i = lane & 15;
        \\    const int half_idx = sub >> 3;
        \\    const int group = (sub & 7) >> 1;
        \\    const int l = ((sub & 1) << 4) + i;
        \\    const int ql_off = half_idx * 64 + (group & 1) * 32;
        \\    const int qh_off = half_idx * 32;
        \\    const int qh_shift = group * 2;
        \\    const int nibble_shift = (group >> 1) * 4;
        \\    const int low4 = (int)((ql[ql_off + l] >> nibble_shift) & 0x0fu);
        \\    const int high2 = (int)((qh[qh_off + l] >> qh_shift) & 0x03u);
        \\    const int q = (low4 | (high2 << 4)) - 32;
        \\    const int scale_u = (int)scales[sub];
        \\    const int scale = scale_u >= 128 ? scale_u - 256 : scale_u;
        \\    return d * (float)scale * (float)q;
        \\}}
        \\
        \\kernel void {s}(
        \\    const device float *input [[buffer(0)]],
        \\    const device uchar *weight_q6_k [[buffer(1)]],
        \\    const device float *bias [[buffer(2)]],
        \\    device float *output [[buffer(3)]],
        \\    constant int &rows [[buffer(4)]],
        \\    constant int &in_dim [[buffer(5)]],
        \\    constant int &out_dim [[buffer(6)]],
        \\    uint3 thread_pos [[thread_position_in_threadgroup]],
        \\    uint3 group_pos [[threadgroup_position_in_grid]],
        \\    ushort lane_id [[thread_index_in_simdgroup]],
        \\    ushort simdgroup_id [[simdgroup_index_in_threadgroup]]
        \\) {{
        \\    const uint tid = thread_pos.x;
        \\    const int col = (int)group_pos.x;
        \\    const int row = (int)group_pos.y;
        \\    if (row >= rows || rows < 2 || rows > 8 || col >= out_dim || (in_dim & {d}) != 0) return;
        \\
        \\    float acc = 0.0f;
        \\    const int block_count = in_dim >> {d};
        \\    if (tid < {d}) {{
        \\        for (int block_idx = 0; block_idx < block_count; ++block_idx) {{
        \\            const device uchar *block = weight_q6_k + ((col * block_count + block_idx) * {d});
        \\            const int base = block_idx << {d};
        \\            const float d = antfly_half_le_to_float(block + {d});
        \\            for (int lane = (int)tid; lane < {d}; lane += {d}) {{
        \\                acc += input[row * in_dim + base + lane] * antfly_q6_k_dequant_lane(block, lane, d);
        \\            }}
        \\        }}
        \\    }}
        \\
        \\    threadgroup float partial[{d}];
        \\    acc = simd_sum(acc);
        \\    if (lane_id == 0u) partial[simdgroup_id] = acc;
        \\    if (simdgroup_id == 0u && lane_id >= {d}u) partial[lane_id] = 0.0f;
        \\    threadgroup_barrier(mem_flags::mem_threadgroup);
        \\    const float total = simd_sum(partial[lane_id]);
        \\    if (lane_id == 0u && simdgroup_id == 0u) output[row * out_dim + col] = {s};
        \\}}
        \\
    ,
        .{
            qh_offset,
            scales_offset,
            compiled.artifact.kernel_id,
            block_mask,
            block_shift,
            thread_count,
            compiled.spec.block_bytes,
            block_shift,
            d_offset,
            compiled.spec.block_values,
            thread_count,
            simd_width,
            active_simdgroups,
            output_expr,
        },
    );

    return try out.toOwnedSlice(allocator);
}

pub fn generatedSourceForArtifact(artifact: GeneratedArtifact) ?[]const u8 {
    if (artifact.backend == .cuda and std.mem.eql(u8, artifact.kernel_id, first_lazy_benchmark.generated_kernel_id)) {
        return first_lazy_cuda_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_lazy_metal_kernel_id)) {
        return first_lazy_metal_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q4_0_kernel_id)) {
        return first_general_metal_q4_0_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q4_1_kernel_id)) {
        return first_general_metal_q4_1_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q5_0_kernel_id)) {
        return first_general_metal_q5_0_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q5_1_kernel_id)) {
        return first_general_metal_q5_1_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q4_kernel_id)) {
        return first_general_metal_q4_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q4_bias_kernel_id)) {
        return first_general_metal_q4_bias_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q8_kernel_id)) {
        return first_general_metal_q8_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q8_bias_kernel_id)) {
        return first_general_metal_q8_bias_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q8_bias_gelu_kernel_id)) {
        return first_general_metal_q8_bias_gelu_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q8_relu_kernel_id)) {
        return first_general_metal_q8_relu_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q2_kernel_id)) {
        return first_general_metal_q2_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q2_bias_kernel_id)) {
        return first_general_metal_q2_bias_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q2_bias_gelu_kernel_id)) {
        return first_general_metal_q2_bias_gelu_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q3_kernel_id)) {
        return first_general_metal_q3_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q3_bias_kernel_id)) {
        return first_general_metal_q3_bias_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q3_bias_gelu_kernel_id)) {
        return first_general_metal_q3_bias_gelu_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q8_1_kernel_id)) {
        return first_general_metal_q8_1_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q8_k_kernel_id)) {
        return first_general_metal_q8_k_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q5_kernel_id)) {
        return first_general_metal_q5_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q5_bias_kernel_id)) {
        return first_general_metal_q5_bias_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q5_bias_gelu_kernel_id)) {
        return first_general_metal_q5_bias_gelu_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q6_kernel_id)) {
        return first_general_metal_q6_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q6_bias_kernel_id)) {
        return first_general_metal_q6_bias_source;
    }
    if (artifact.backend == .metal and std.mem.eql(u8, artifact.kernel_id, first_general_metal_q6_bias_gelu_kernel_id)) {
        return first_general_metal_q6_bias_gelu_source;
    }
    return null;
}

fn productionKernelId(
    backend: Backend,
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
) []const u8 {
    if (backend == .cuda and format == first_lazy_benchmark.format and row_bucket == first_lazy_benchmark.row_bucket and epilogue == first_lazy_benchmark.epilogue) {
        return first_lazy_benchmark.handwritten_baseline;
    }
    if (backend == .cuda) {
        if (format == .q4_k and epilogue == .none) return "termite_linear_q4_k_f32_tile4";
        if (format == .q4_k and epilogue == .bias) {
            return switch (row_bucket) {
                .rows_1 => "termite_linear_q4_k_bias_f32_tile4",
                .rows_2_8 => "termite_linear_q4_k_bias_f32_tile4_r2",
                else => "cuda_handwritten_quant_matmul",
            };
        }
        if (format == .q6_k and epilogue == .none) return "termite_linear_q6_k_f32_tile4";
    }
    return switch (backend) {
        .cuda => "cuda_handwritten_quant_matmul",
        .metal => "metal_handwritten_quant_matmul",
    };
}

pub fn referenceMatmulBiasGelu(
    allocator: std.mem.Allocator,
    format: quant_matmul.Format,
    raw_weight: []const u8,
    input: []const f32,
    bias: []const f32,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    output: []f32,
) !void {
    return referenceMatmulEpilogue(allocator, format, raw_weight, input, bias, rows, in_dim, out_dim, .bias_gelu, output);
}

pub fn referenceMatmulNoBias(
    allocator: std.mem.Allocator,
    format: quant_matmul.Format,
    raw_weight: []const u8,
    input: []const f32,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    output: []f32,
) !void {
    return referenceMatmulEpilogue(allocator, format, raw_weight, input, null, rows, in_dim, out_dim, .none, output);
}

pub fn referenceMatmulEpilogue(
    allocator: std.mem.Allocator,
    format: quant_matmul.Format,
    raw_weight: []const u8,
    input: []const f32,
    bias: ?[]const f32,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    epilogue: Epilogue,
    output: []f32,
) !void {
    if (input.len != rows * in_dim or output.len != rows * out_dim) return error.InvalidShape;
    switch (epilogue) {
        .none, .bias, .bias_gelu, .relu, .gelu => {},
        else => return error.UnsupportedEpilogue,
    }
    if ((epilogue == .bias or epilogue == .bias_gelu) and (bias == null or bias.?.len != out_dim)) return error.InvalidShape;
    const tensor_type = tensorTypeForFormat(format) orelse return error.UnsupportedFormat;
    const dense_weight = try allocator.alloc(f32, in_dim * out_dim);
    defer allocator.free(dense_weight);
    try quant_codec.dequantizeToFloat32(tensor_type, raw_weight, dense_weight);

    for (0..rows) |r| {
        for (0..out_dim) |o| {
            var acc: f32 = 0.0;
            for (0..in_dim) |i| {
                acc += input[r * in_dim + i] * dense_weight[o * in_dim + i];
            }
            output[r * out_dim + o] = switch (epilogue) {
                .none => acc,
                .bias => acc + bias.?[o],
                .bias_gelu => gelu(acc + bias.?[o]),
                .relu => @max(acc, 0.0),
                .gelu => gelu(acc),
                else => return error.UnsupportedEpilogue,
            };
        }
    }
}

pub fn generatedMathMatmulBiasGelu(
    format: quant_matmul.Format,
    raw_weight: []const u8,
    input: []const f32,
    bias: []const f32,
    rows: usize,
    in_dim: usize,
    out_dim: usize,
    output: []f32,
) !void {
    if (format != .q4_k) return error.UnsupportedFormat;
    if (input.len != rows * in_dim or bias.len != out_dim or output.len != rows * out_dim) return error.InvalidShape;
    if (in_dim % q4_k_spec.block_values != 0) return error.InvalidShape;

    const block_count = in_dim / q4_k_spec.block_values;
    if (raw_weight.len != out_dim * block_count * q4_k_spec.block_bytes) return error.InvalidShape;

    for (0..rows) |r| {
        for (0..out_dim) |o| {
            var acc: f32 = bias[o];
            for (0..block_count) |block_idx| {
                const block_off = (o * block_count + block_idx) * q4_k_spec.block_bytes;
                const block = raw_weight[block_off .. block_off + q4_k_spec.block_bytes];
                const input_off = r * in_dim + block_idx * q4_k_spec.block_values;
                for (0..q4_k_spec.block_values) |lane| {
                    acc += input[input_off + lane] * generatedMathQ4KDequantLane(block, lane);
                }
            }
            output[r * out_dim + o] = gelu(acc);
        }
    }
}

const ScaleMin = struct {
    scale: f32,
    min: f32,
};

fn generatedMathQ4KDequantLane(block: []const u8, lane: usize) f32 {
    const d = quant_codec.decodeFp16Le(block[0], block[1]);
    const dmin = quant_codec.decodeFp16Le(block[2], block[3]);
    const scales = block[4..16];
    const qs = block[16..144];

    const sub = lane >> 5;
    const q_index = (sub >> 1) * 32 + (lane & 31);
    const packed_byte = qs[q_index];
    const q = if ((sub & 1) == 0) packed_byte & 0x0F else packed_byte >> 4;
    const scale_min = generatedMathQ4KScaleMin(scales, sub);
    return d * scale_min.scale * @as(f32, @floatFromInt(q)) - dmin * scale_min.min;
}

fn generatedMathQ4KScaleMin(scales: []const u8, sub: usize) ScaleMin {
    if (sub < 4) return .{
        .scale = @floatFromInt(scales[sub] & 63),
        .min = @floatFromInt(scales[sub + 4] & 63),
    };

    return .{
        .scale = @floatFromInt((scales[sub + 4] & 0x0F) | ((scales[sub - 4] >> 6) << 4)),
        .min = @floatFromInt((scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4)),
    };
}

fn scheduleFor(row_bucket: quant_matmul.RowBucket, dispatch: quant_matmul.DispatchKind) QuantKernelSchedule {
    return switch (row_bucket) {
        .rows_1 => .{
            .dispatch = dispatch,
            .row_bucket = row_bucket,
            .tile_rows = 1,
            .tile_cols = 4,
            .vector_width = 4,
            .threads_per_block = 256,
            .shared_memory_bytes = 0,
            .register_pressure_hint = 4,
            .tensor_core_eligible = false,
        },
        .rows_2_8 => .{
            .dispatch = dispatch,
            .row_bucket = row_bucket,
            .tile_rows = 2,
            .tile_cols = 4,
            .vector_width = 4,
            .threads_per_block = 256,
            .shared_memory_bytes = 0,
            .register_pressure_hint = 6,
            .tensor_core_eligible = false,
        },
        .rows_9_64, .rows_65_plus => .{
            .dispatch = dispatch,
            .row_bucket = row_bucket,
            .tile_rows = 8,
            .tile_cols = 4,
            .vector_width = 4,
            .threads_per_block = 256,
            .shared_memory_bytes = 0,
            .register_pressure_hint = 8,
            .tensor_core_eligible = false,
        },
        .rows_0 => .{
            .dispatch = .scalar,
            .row_bucket = row_bucket,
            .tile_rows = 0,
            .tile_cols = 0,
            .vector_width = 0,
            .threads_per_block = 0,
            .shared_memory_bytes = 0,
            .register_pressure_hint = 0,
            .tensor_core_eligible = false,
        },
    };
}

fn candidateScheduleFor(lowering: QuantKernelLowering) QuantKernelSchedule {
    if (lowering.candidate_route != .generated_dev_candidate) return emptyCandidateSchedule(lowering.row_bucket, lowering.schedule.dispatch);
    const tile_cols: usize = if (lowering.backend == .metal)
        metalGeneratedColsPerThreadgroup(lowering.format, lowering.row_bucket, lowering.epilogue)
    else
        1;
    const threads_per_block: usize = if (lowering.backend == .metal)
        metalGeneratedThreadsPerThreadgroup(lowering.format, lowering.row_bucket, lowering.epilogue)
    else
        128;
    return .{
        .dispatch = lowering.schedule.dispatch,
        .row_bucket = lowering.row_bucket,
        .tile_rows = 1,
        .tile_cols = tile_cols,
        .vector_width = 1,
        .threads_per_block = threads_per_block,
        .shared_memory_bytes = 0,
        .register_pressure_hint = 6,
        .tensor_core_eligible = false,
    };
}

pub fn metalGeneratedColsPerThreadgroup(
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
) usize {
    if (row_bucket != .rows_2_8) return 1;
    if (format == .q4_1 and epilogue == .none) return 2;
    if (format == .q5_1 and epilogue == .none) return 2;
    return 1;
}

pub fn metalGeneratedThreadsPerThreadgroup(
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
) usize {
    if (metalCandidateUsesThreadgroup64(format, row_bucket, epilogue)) {
        return 64;
    }
    if (metalCandidateUsesThreadgroup32(format, row_bucket, epilogue)) {
        return 32;
    }
    return 128;
}

fn metalCandidateUsesThreadgroup64(
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
) bool {
    if (row_bucket != .rows_2_8) return false;
    return switch (format) {
        .q4_k => epilogue == .none or epilogue == .bias or epilogue == .bias_gelu,
        else => false,
    };
}

fn metalCandidateUsesThreadgroup32(
    format: quant_matmul.Format,
    row_bucket: quant_matmul.RowBucket,
    epilogue: Epilogue,
) bool {
    if (row_bucket != .rows_2_8) return false;
    return switch (format) {
        .q4_0, .q4_1, .q5_0, .q5_1 => epilogue == .none,
        .q8_0 => epilogue == .none or epilogue == .bias or epilogue == .bias_gelu or epilogue == .relu,
        .q2_k => epilogue == .none or epilogue == .bias or epilogue == .bias_gelu,
        .q3_k => epilogue == .none or epilogue == .bias or epilogue == .bias_gelu,
        .q8_1, .q8_k => epilogue == .none,
        else => false,
    };
}

fn emptyCandidateSchedule(row_bucket: quant_matmul.RowBucket, dispatch: quant_matmul.DispatchKind) QuantKernelSchedule {
    return .{
        .dispatch = dispatch,
        .row_bucket = row_bucket,
        .tile_rows = 0,
        .tile_cols = 0,
        .vector_width = 0,
        .threads_per_block = 0,
        .shared_memory_bytes = 0,
        .register_pressure_hint = 0,
        .tensor_core_eligible = false,
    };
}

fn sourceFingerprint(comptime source: []const u8) u64 {
    @setEvalBranchQuota(131072);
    return std.hash.Wyhash.hash(0, source);
}

fn blockValueShift(block_values: usize) usize {
    var value = block_values;
    var shift: usize = 0;
    while (value > 1) : (value >>= 1) {
        shift += 1;
    }
    return shift;
}

fn dispatchForRowBucket(row_bucket: quant_matmul.RowBucket) ?quant_matmul.DispatchKind {
    return switch (row_bucket) {
        .rows_0 => null,
        .rows_1 => .mmv,
        .rows_2_8 => .small_batch,
        .rows_9_64, .rows_65_plus => .mm,
    };
}

fn irOpsForEpilogue(epilogue: Epilogue) []const IROp {
    return switch (epilogue) {
        .none, .pair => &ir_ops_basic,
        .bias => &ir_ops_bias,
        .bias_gelu => &ir_ops_bias_gelu,
        else => &ir_ops_basic,
    };
}

fn tensorTypeForFormat(format: quant_matmul.Format) ?tensor_types.TensorType {
    return switch (format) {
        .q1_0 => .{ .known = .Q1_0 },
        .i2_s => .{ .known = .I2_S },
        .i8_s => .{ .known = .I8_S },
        .q2_k => .{ .known = .Q2_K },
        .q3_k => .{ .known = .Q3_K },
        .q4_0 => .{ .known = .Q4_0 },
        .q4_1 => .{ .known = .Q4_1 },
        .q5_0 => .{ .known = .Q5_0 },
        .q5_1 => .{ .known = .Q5_1 },
        .q4_k => .{ .known = .Q4_K },
        .q5_k => .{ .known = .Q5_K },
        .q6_k => .{ .known = .Q6_K },
        .q8_0 => .{ .known = .Q8_0 },
        .q8_1 => .{ .known = .Q8_1 },
        .q8_k => .{ .known = .Q8_K },
        .tq1_0 => .{ .known = .TQ1_0 },
        .tq2_0 => .{ .known = .TQ2_0 },
        .iq2_xxs => .{ .known = .IQ2_XXS },
        .iq2_xs => .{ .known = .IQ2_XS },
        .iq2_s => .{ .known = .IQ2_S },
        .iq3_xxs => .{ .known = .IQ3_XXS },
        .iq3_s => .{ .known = .IQ3_S },
        .iq1_s => .{ .known = .IQ1_S },
        .iq1_m => .{ .known = .IQ1_M },
        .iq4_nl => .{ .known = .IQ4_NL },
        .iq4_xs => .{ .known = .IQ4_XS },
        .mxfp4 => .{ .known = .MXFP4 },
        .nvfp4 => .{ .known = .NVFP4 },
        else => null,
    };
}

fn toleranceFor(format: quant_matmul.Format, epilogue: Epilogue) f32 {
    _ = format;
    _ = epilogue;
    return 0.0001;
}

fn referenceSupportedForEpilogue(epilogue: Epilogue) bool {
    return switch (epilogue) {
        .none, .bias, .bias_gelu, .relu, .gelu => true,
        else => false,
    };
}

fn blockValuesForFormat(comptime format: quant_matmul.Format) usize {
    const tensor_type = tensorTypeForFormat(format) orelse @compileError("missing GGUF tensor type for quant kernel format");
    return tensor_types.valuesPerBlock(tensor_type) orelse @compileError("missing GGUF block value count");
}

fn blockBytesForFormat(comptime format: quant_matmul.Format) usize {
    const tensor_type = tensorTypeForFormat(format) orelse @compileError("missing GGUF tensor type for quant kernel format");
    return tensor_types.bytesPerBlock(tensor_type) orelse @compileError("missing GGUF block byte count");
}

fn gelu(x: f32) f32 {
    const inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
    return 0.5 * x * (1.0 + std.math.tanh(inner));
}

fn contains(comptime T: type, haystack: []const T, needle: T) bool {
    for (haystack) |item| {
        if (item == needle) return true;
    }
    return false;
}

fn quantizeForFormat(allocator: std.mem.Allocator, format: quant_matmul.Format, dense: []const f32) ![]u8 {
    return switch (format) {
        .q1_0 => quant_codec.quantizeQ1_0FromF32(allocator, dense),
        .i2_s => quantizeI2SFromF32(allocator, dense),
        .i8_s => quantizeI8SFromF32(allocator, dense),
        .q2_k => quant_codec.quantizeQ2_KFromF32(allocator, dense),
        .q3_k => quant_codec.quantizeQ3_KFromF32(allocator, dense),
        .q4_0 => quant_codec.quantizeQ4_0FromF32(allocator, dense),
        .q4_1 => quant_codec.quantizeQ4_1FromF32(allocator, dense),
        .q5_0 => quant_codec.quantizeQ5_0FromF32(allocator, dense),
        .q5_1 => quant_codec.quantizeQ5_1FromF32(allocator, dense),
        .q4_k => quant_codec.quantizeQ4_KFromF32(allocator, dense),
        .q5_k => quant_codec.quantizeQ5_KFromF32(allocator, dense),
        .q6_k => quant_codec.quantizeQ6_KFromF32(allocator, dense),
        .q8_0 => quant_codec.quantizeQ8_0FromF32(allocator, dense),
        .q8_1 => quant_codec.quantizeQ8_1FromF32(allocator, dense),
        .q8_k => quant_codec.quantizeQ8_KFromF32(allocator, dense),
        .tq1_0,
        .tq2_0,
        .iq2_xxs,
        .iq2_xs,
        .iq2_s,
        .iq3_xxs,
        .iq3_s,
        .iq1_s,
        .iq1_m,
        => quantizeOpaqueValidBlocks(allocator, format, dense),
        .iq4_nl => quantizeIQ4NLFromF32(allocator, dense),
        .iq4_xs => quantizeIQ4XSFromF32(allocator, dense),
        .mxfp4 => quantizeMXFP4FromF32(allocator, dense),
        .nvfp4 => quantizeNVFP4FromF32(allocator, dense),
        else => error.UnsupportedFormat,
    };
}

fn quantizeI2SFromF32(allocator: std.mem.Allocator, dense: []const f32) ![]u8 {
    const values_per_block = 128;
    const bytes_per_block = 32;
    if (dense.len == 0 or dense.len % values_per_block != 0) return error.InvalidShape;
    const blocks = dense.len / values_per_block;
    const raw = try allocator.alloc(u8, blocks * bytes_per_block);
    errdefer allocator.free(raw);
    for (0..blocks) |block| {
        const base = block * values_per_block;
        for (0..32) |group| {
            var packed_byte: u8 = 0;
            for (0..4) |lane| {
                const value = dense[base + lane * 32 + group];
                const code: u8 = if (value > 0.5) 2 else if (value < -0.5) 0 else 1;
                const shift: u3 = @intCast((3 - lane) * 2);
                packed_byte |= code << shift;
            }
            raw[block * bytes_per_block + group] = packed_byte;
        }
    }
    return raw;
}

fn quantizeI8SFromF32(allocator: std.mem.Allocator, dense: []const f32) ![]u8 {
    if (dense.len == 0) return error.InvalidShape;
    const raw = try allocator.alloc(u8, dense.len);
    errdefer allocator.free(raw);
    for (dense, raw) |value, *dst| {
        const clamped = std.math.clamp(@round(value), -128.0, 127.0);
        const q: i8 = @intFromFloat(clamped);
        dst.* = @bitCast(q);
    }
    return raw;
}

fn quantizeOpaqueValidBlocks(allocator: std.mem.Allocator, format: quant_matmul.Format, dense: []const f32) ![]u8 {
    const tensor_type = tensorTypeForFormat(format) orelse return error.UnsupportedFormat;
    const values_per_block: usize = @intCast(tensor_types.valuesPerBlock(tensor_type) orelse return error.UnsupportedFormat);
    const bytes_per_block: usize = @intCast(tensor_types.bytesPerBlock(tensor_type) orelse return error.UnsupportedFormat);
    if (dense.len == 0 or dense.len % values_per_block != 0) return error.InvalidShape;
    const blocks = dense.len / values_per_block;
    const raw = try allocator.alloc(u8, blocks * bytes_per_block);
    errdefer allocator.free(raw);
    @memset(raw, 0);
    for (0..blocks) |block_index| {
        const block = raw[block_index * bytes_per_block ..][0..bytes_per_block];
        switch (format) {
            .tq1_0 => {
                block[52] = 0x00;
                block[53] = 0x3C;
            },
            .tq2_0 => {
                block[64] = 0x00;
                block[65] = 0x3C;
            },
            .iq2_xxs,
            .iq2_xs,
            .iq2_s,
            .iq3_xxs,
            .iq3_s,
            .iq1_s,
            => {
                block[0] = 0x00;
                block[1] = 0x3C;
            },
            .iq1_m => {},
            else => return error.UnsupportedFormat,
        }
    }
    return raw;
}

fn quantizeIQ4NLFromF32(allocator: std.mem.Allocator, dense: []const f32) ![]u8 {
    return quantizeNibbleTableFromF32(allocator, dense, 32, 18, 2, &quant_codec.iq4_nl_values, .iq4_nl);
}

fn quantizeMXFP4FromF32(allocator: std.mem.Allocator, dense: []const f32) ![]u8 {
    return quantizeNibbleTableFromF32(allocator, dense, 32, 17, 1, &quant_codec.mxfp4_values, .mxfp4);
}

fn quantizeIQ4XSFromF32(allocator: std.mem.Allocator, dense: []const f32) ![]u8 {
    const values_per_block = 256;
    const bytes_per_block = 136;
    if (dense.len == 0 or dense.len % values_per_block != 0) return error.InvalidShape;
    const blocks = dense.len / values_per_block;
    const raw = try allocator.alloc(u8, blocks * bytes_per_block);
    errdefer allocator.free(raw);
    for (0..blocks) |block| {
        const dst = raw[block * bytes_per_block ..][0..bytes_per_block];
        @memset(dst, 0);
        dst[0] = 0x00;
        dst[1] = 0x3C;
        std.mem.writeInt(u16, dst[2..4], 0xAAAA, .little);
        @memset(dst[4..8], 0x11);
        const src = dense[block * values_per_block ..][0..values_per_block];
        for (0..8) |group| {
            packNibbles(src[group * 32 ..][0..32], dst[8 + group * 16 ..][0..16], &quant_codec.iq4_nl_values);
        }
    }
    return raw;
}

fn quantizeNVFP4FromF32(allocator: std.mem.Allocator, dense: []const f32) ![]u8 {
    const values_per_block = 64;
    const bytes_per_block = 36;
    if (dense.len == 0 or dense.len % values_per_block != 0) return error.InvalidShape;
    const blocks = dense.len / values_per_block;
    const raw = try allocator.alloc(u8, blocks * bytes_per_block);
    errdefer allocator.free(raw);
    for (0..blocks) |block| {
        const dst = raw[block * bytes_per_block ..][0..bytes_per_block];
        @memset(dst[0..4], 0x40);
        const src = dense[block * values_per_block ..][0..values_per_block];
        for (0..4) |sub| {
            packNibbles(src[sub * 16 ..][0..16], dst[4 + sub * 8 ..][0..8], &quant_codec.mxfp4_values);
        }
    }
    return raw;
}

fn quantizeNibbleTableFromF32(
    allocator: std.mem.Allocator,
    dense: []const f32,
    values_per_block: usize,
    bytes_per_block: usize,
    qs_offset: usize,
    table: []const i8,
    format: quant_matmul.Format,
) ![]u8 {
    if (dense.len == 0 or dense.len % values_per_block != 0) return error.InvalidShape;
    const blocks = dense.len / values_per_block;
    const raw = try allocator.alloc(u8, blocks * bytes_per_block);
    errdefer allocator.free(raw);
    for (0..blocks) |block| {
        const dst = raw[block * bytes_per_block ..][0..bytes_per_block];
        @memset(dst, 0);
        switch (format) {
            .iq4_nl => {
                dst[0] = 0x00;
                dst[1] = 0x3C;
            },
            .mxfp4 => dst[0] = 128,
            else => return error.UnsupportedFormat,
        }
        packNibbles(dense[block * values_per_block ..][0..values_per_block], dst[qs_offset..][0 .. values_per_block / 2], table);
    }
    return raw;
}

fn packNibbles(values: []const f32, dst: []u8, table: []const i8) void {
    for (dst, 0..) |*byte, i| {
        const lo = nearestTableIndex(table, values[i]);
        const hi = nearestTableIndex(table, values[i + dst.len]);
        byte.* = lo | (hi << 4);
    }
}

fn nearestTableIndex(table: []const i8, value: f32) u8 {
    var best_index: u8 = 0;
    var best_diff = std.math.inf(f32);
    for (table, 0..) |candidate, i| {
        const diff = @abs(value - @as(f32, @floatFromInt(candidate)));
        if (diff < best_diff) {
            best_diff = diff;
            best_index = @intCast(i);
        }
    }
    return best_index;
}

test "quant kernel compiler descriptors mirror packed format metadata" {
    for (descriptor_formats) |format| {
        const spec = specFor(format).?;
        const desc = quant_matmul.packedFormatDescriptor(format);
        const tensor_type = tensorTypeForFormat(format).?;
        try std.testing.expectEqual(desc.values_per_block, spec.block_values);
        try std.testing.expectEqual(desc.bytes_per_block, spec.block_bytes);
        try std.testing.expectEqual(@as(usize, @intCast(tensor_types.valuesPerBlock(tensor_type).?)), spec.block_values);
        try std.testing.expectEqual(@as(usize, @intCast(tensor_types.bytesPerBlock(tensor_type).?)), spec.block_bytes);
        try std.testing.expectEqual(contains(Epilogue, spec.supported_epilogues, .bias_gelu), spec.supportsEpilogue(.bias_gelu));
        try std.testing.expectEqual(contains(Backend, spec.supported_backends, .cuda), spec.supportsBackend(.cuda));
        try std.testing.expectEqual(contains(Backend, spec.supported_backends, .metal), spec.supportsBackend(.metal));

        var offset: usize = 0;
        for (spec.block_fields) |field| {
            try std.testing.expectEqual(offset, field.offset);
            offset += field.bytes;
        }
        try std.testing.expectEqual(spec.block_bytes, offset);
    }

    try std.testing.expectEqualStrings("qs", specFor(.q1_0).?.block_fields[1].name);
    try std.testing.expectEqualStrings("qs", specFor(.i2_s).?.block_fields[0].name);
    try std.testing.expectEqualStrings("q", specFor(.i8_s).?.block_fields[0].name);
    try std.testing.expectEqualStrings("dmin", specFor(.q2_k).?.block_fields[2].name);
    try std.testing.expectEqualStrings("hmask", specFor(.q3_k).?.block_fields[0].name);
    try std.testing.expectEqualStrings("qs", specFor(.q4_0).?.block_fields[1].name);
    try std.testing.expectEqualStrings("m", specFor(.q4_1).?.block_fields[1].name);
    try std.testing.expectEqualStrings("qh", specFor(.q5_0).?.block_fields[1].name);
    try std.testing.expectEqualStrings("m", specFor(.q5_1).?.block_fields[1].name);
    try std.testing.expectEqualStrings("qs", specFor(.q4_k).?.block_fields[3].name);
    try std.testing.expectEqualStrings("qh", specFor(.q5_k).?.block_fields[3].name);
    try std.testing.expectEqualStrings("scales", specFor(.q6_k).?.block_fields[2].name);
    try std.testing.expectEqualStrings("d", specFor(.q8_0).?.block_fields[0].name);
    try std.testing.expectEqualStrings("sum", specFor(.q8_1).?.block_fields[1].name);
    try std.testing.expectEqualStrings("bsums", specFor(.q8_k).?.block_fields[2].name);
    try std.testing.expectEqualStrings("qh", specFor(.tq1_0).?.block_fields[1].name);
    try std.testing.expectEqualStrings("d", specFor(.tq2_0).?.block_fields[1].name);
    try std.testing.expectEqualStrings("qs", specFor(.iq2_xxs).?.block_fields[1].name);
    try std.testing.expectEqualStrings("scales", specFor(.iq2_xs).?.block_fields[2].name);
    try std.testing.expectEqualStrings("qh", specFor(.iq2_s).?.block_fields[2].name);
    try std.testing.expectEqualStrings("qs", specFor(.iq3_xxs).?.block_fields[1].name);
    try std.testing.expectEqualStrings("signs", specFor(.iq3_s).?.block_fields[3].name);
    try std.testing.expectEqualStrings("qh", specFor(.iq1_s).?.block_fields[2].name);
    try std.testing.expectEqualStrings("scales", specFor(.iq1_m).?.block_fields[2].name);
    try std.testing.expectEqualStrings("qs", specFor(.iq4_nl).?.block_fields[1].name);
    try std.testing.expectEqualStrings("scales_l", specFor(.iq4_xs).?.block_fields[2].name);
    try std.testing.expectEqualStrings("d", specFor(.mxfp4).?.block_fields[0].name);
    try std.testing.expectEqualStrings("scales", specFor(.nvfp4).?.block_fields[0].name);
}

test "quant kernel compiler classifies every codec quant format" {
    try std.testing.expectEqual(@as(usize, 28), codec_format_coverage.len);
    try std.testing.expectEqual(codec_format_coverage.len, descriptor_formats.len);

    for (codec_format_coverage, 0..) |entry, index| {
        const tensor_type: tensor_types.TensorType = .{ .known = entry.tensor_type };
        try std.testing.expectEqual(entry.format, descriptor_formats[index]);
        try std.testing.expectEqual(@as(?quant_matmul.Format, entry.format), backend_contracts.quantFormatFromGgufTensorType(tensor_type));
        try std.testing.expect(tensor_types.valuesPerBlock(tensor_type) != null);
        try std.testing.expect(tensor_types.bytesPerBlock(tensor_type) != null);

        for (codec_format_coverage[0..index]) |previous| {
            try std.testing.expect(previous.tensor_type != entry.tensor_type);
            try std.testing.expect(previous.format != entry.format);
        }

        try std.testing.expect(specFor(entry.format) != null);
    }
}

fn expectManifestArrayCount(manifest: []const u8, count_field: []const u8, array_field: []const u8, expected: usize) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, manifest, .{});
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidManifestJson;
    const count_value = parsed.value.object.get(count_field) orelse return error.InvalidManifestJson;
    const array_value = parsed.value.object.get(array_field) orelse return error.InvalidManifestJson;
    if (count_value != .integer or array_value != .array) return error.InvalidManifestJson;

    try std.testing.expectEqual(@as(i64, @intCast(expected)), count_value.integer);
    try std.testing.expectEqual(expected, array_value.array.items.len);
}

fn expectManifestNestedInteger(manifest: []const u8, object_field: []const u8, count_field: []const u8, expected: usize) !void {
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, manifest, .{});
    defer parsed.deinit();

    if (parsed.value != .object) return error.InvalidManifestJson;
    const object_value = parsed.value.object.get(object_field) orelse return error.InvalidManifestJson;
    if (object_value != .object) return error.InvalidManifestJson;
    const count_value = object_value.object.get(count_field) orelse return error.InvalidManifestJson;
    if (count_value != .integer) return error.InvalidManifestJson;

    try std.testing.expectEqual(@as(i64, @intCast(expected)), count_value.integer);
}

test "quant kernel compiler spec manifest serializes descriptors" {
    const manifest = try specManifestJson(std.testing.allocator);
    defer std.testing.allocator.free(manifest);

    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "antfly.quant_kernel_specs.v1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"format_count\": 28"));
    try expectManifestArrayCount(manifest, "format_count", "specs", descriptor_formats.len);
    try std.testing.expectEqual(descriptor_formats.len, std.mem.count(u8, manifest, "\"format\": "));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"format\": \"q4_k\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"reference_tensor_type\": \"Q4_K\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"block_values\": 256"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"block_bytes\": 144"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"name\": \"qs\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"offset\": 16"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "load block_q4_K {d,dmin,scales,qs}"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"supported_epilogues\": ["));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"bias_gelu\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"supported_backends\": ["));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"cuda\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"accumulator_dtype\": \"f32\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"output_dtype\": \"f32\""));
}

test "quant kernel compiler artifact manifest serializes generated candidates" {
    const manifest = try artifactManifestJson(std.testing.allocator);
    defer std.testing.allocator.free(manifest);

    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_artifact_manifest_schema));
    const artifact_count = try std.fmt.allocPrint(std.testing.allocator, "\"artifact_count\": {d}", .{first_generated_artifacts.len});
    defer std.testing.allocator.free(artifact_count);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, artifact_count));
    const metal_evidence_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"checked_in_metal_evidence_count\": {d}", .{first_metal_runtime_evidence_count});
    defer std.testing.allocator.free(metal_evidence_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, metal_evidence_count_field));
    const blocker_evidence_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_promotion_blocker_evidence_count\": {d}", .{first_metal_promotion_blocker_evidence_count});
    defer std.testing.allocator.free(blocker_evidence_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, blocker_evidence_count_field));
    const blocker_path_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_promotion_blocker_evidence_path_count\": {d}", .{metalPromotionBlockerEvidencePathCount()});
    defer std.testing.allocator.free(blocker_path_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, blocker_path_count_field));
    const blocker_case_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_promotion_blocker_evidence_expected_case_count\": {d}", .{first_metal_promotion_blocker_evidence_expected_case_count});
    defer std.testing.allocator.free(blocker_case_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, blocker_case_count_field));
    const blocker_route_ready_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_promotion_blocker_evidence_expected_route_ready_count\": {d}", .{first_metal_promotion_blocker_evidence_expected_route_ready_count});
    defer std.testing.allocator.free(blocker_route_ready_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, blocker_route_ready_count_field));
    try std.testing.expectEqual(first_metal_promotion_blocker_evidence_expected_case_count, first_metal_promotion_blocker_evidence_expected_route_ready_count);
    const blocker_check_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_promotion_blocker_check_command_count\": {d}", .{metalPromotionBlockerEvidencePathCount()});
    defer std.testing.allocator.free(blocker_check_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, blocker_check_count_field));
    const blocker_skipped_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_promotion_blocker_skipped_no_path_count\": {d}", .{metalPromotionBlockerSkippedNoPathCount()});
    defer std.testing.allocator.free(blocker_skipped_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, blocker_skipped_count_field));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_promotion_blocker_cleared_requires_checked_in_evidence\": true"));
    try std.testing.expectEqual(metalPromotionBlockerSkippedNoPathCount(), metalPromotionBlockerEvidenceCount(metal_blocker_unsupported_handwritten));
    const blocker_speedup_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_promotion_blocker_speedup_gate_missing_count\": {d}", .{metalPromotionBlockerEvidenceCount(metal_blocker_speedup_gate_missing)});
    defer std.testing.allocator.free(blocker_speedup_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, blocker_speedup_count_field));
    const blocker_unstable_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_promotion_blocker_unstable_benchmark_timing_count\": {d}", .{metalPromotionBlockerEvidenceCount(metal_blocker_unstable_benchmark_timing)});
    defer std.testing.allocator.free(blocker_unstable_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, blocker_unstable_count_field));
    const blocker_unsupported_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_promotion_blocker_unsupported_handwritten_count\": {d}", .{metalPromotionBlockerEvidenceCount(metal_blocker_unsupported_handwritten)});
    defer std.testing.allocator.free(blocker_unsupported_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, blocker_unsupported_count_field));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_unsupported_handwritten_baseline_blocks_promotion\": true"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_unsupported_handwritten_baseline_uses_runtime_route_all_evidence\": true"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_unsupported_handwritten_baseline_has_promotion_evidence_path\": false"));
    const route_all_case_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_runtime_route_all_expected_case_count\": {d}", .{first_metal_runtime_route_all_expected_case_count});
    defer std.testing.allocator.free(route_all_case_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, route_all_case_count_field));
    const route_all_ready_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_runtime_route_all_expected_route_ready_count\": {d}", .{first_metal_runtime_route_all_expected_route_ready_count});
    defer std.testing.allocator.free(route_all_ready_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, route_all_ready_count_field));
    try std.testing.expectEqual(first_metal_runtime_route_all_expected_case_count, first_metal_runtime_route_all_expected_route_ready_count);
    const route_all_provider_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_runtime_route_all_expected_provider_route_count\": {d}", .{first_metal_runtime_route_all_expected_provider_route_count});
    defer std.testing.allocator.free(route_all_provider_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, route_all_provider_count_field));
    try std.testing.expect(first_metal_runtime_route_all_expected_case_count > first_metal_runtime_route_all_expected_provider_route_count);
    const production_regression_kernel_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_production_regression_expected_kernel_count\": {d}", .{metalProductionRegressionExpectedKernelCount()});
    defer std.testing.allocator.free(production_regression_kernel_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, production_regression_kernel_count_field));
    const production_regression_case_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_production_regression_expected_case_count\": {d}", .{metalProductionRegressionExpectedCaseCount()});
    defer std.testing.allocator.free(production_regression_case_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, production_regression_case_count_field));
    const production_regression_ready_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_production_regression_expected_route_ready_count\": {d}", .{first_metal_production_regression_expected_route_ready_count});
    defer std.testing.allocator.free(production_regression_ready_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, production_regression_ready_count_field));
    try std.testing.expectEqual(first_metal_production_regression_expected_case_count, first_metal_production_regression_expected_route_ready_count);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_production_regression_route_ready_is_hard_gate\": true"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_production_regression_missing_provider_route_is_hard_gate\": true"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_production_regression_speedup_gate_missing_is_hard_gate\": true"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_production_regression_unstable_benchmark_timing_is_hard_gate\": true"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_promotion_warmup_repeat_runs\": 2"));
    try std.testing.expectEqual(first_metal_runtime_evidence_count, metalProductionRegressionExpectedKernelCount());
    try expectManifestArrayCount(manifest, "artifact_count", "artifacts", first_generated_artifacts.len);
    try expectManifestArrayCount(manifest, "checked_in_metal_evidence_count", "metal_evidence_records", first_metal_runtime_evidence.len);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_blocker_strict_check_command\": \"" ++ first_metal_blocker_strict_check_command ++ "\""));
    var runtime_evidence_count: usize = 0;
    var runtime_route_evidence_count: usize = 0;
    var promotion_evidence_count: usize = 0;
    var promotion_check_count: usize = 0;
    var metal_evidence_count: usize = 0;
    var route_only_no_promotion_policy_count: usize = 0;
    var speedup_policy_count: usize = 0;
    var promoted_speedup_policy_count: usize = 0;
    var production_disabled_policy_count: usize = 0;
    var cuda_policy_count: usize = 0;
    for (first_generated_artifacts) |artifact| {
        if (artifact.runtime_evidence_command.len != 0) runtime_evidence_count += 1;
        if (artifact.backend == .metal and artifactRuntimeWired(artifact) and !artifactHasPromotionEvidence(artifact)) runtime_route_evidence_count += 1;
        if (artifact.promotion_evidence_command.len != 0) promotion_evidence_count += 1;
        if (artifact.promotion_check_command.len != 0) promotion_check_count += 1;
        if (artifact.backend == .metal and artifact.runtime_evidence_command.len != 0) metal_evidence_count += 1;
        const promotion_policy = artifactPromotionPolicy(artifact);
        if (std.mem.eql(u8, promotion_policy, "route_evidence_only_no_promotion")) route_only_no_promotion_policy_count += 1;
        if (std.mem.eql(u8, promotion_policy, "speedup_vs_handwritten")) speedup_policy_count += 1;
        if (std.mem.eql(u8, promotion_policy, "promoted_speedup_vs_handwritten")) promoted_speedup_policy_count += 1;
        if (std.mem.eql(u8, promotion_policy, "production_disabled")) production_disabled_policy_count += 1;
        if (std.mem.eql(u8, promotion_policy, "driver_artifact_policy")) cuda_policy_count += 1;
    }
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"backend\": \"cuda\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"backend\": \"metal\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_lazy_benchmark.generated_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_lazy_benchmark.generated_ptx_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_lazy_benchmark.benchmark_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_lazy_benchmark_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, first_generated_artifacts.len, "\"generated_source_fingerprint\":"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_lazy_metal_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_lazy_metal_artifact_check_command));
    try std.testing.expectEqual(first_generated_artifacts.len, std.mem.count(u8, manifest, "\"generated_source_path\":"));
    try std.testing.expectEqual(first_generated_artifacts.len + first_metal_runtime_evidence.len, std.mem.count(u8, manifest, "\"artifact_source_path\":"));
    try std.testing.expectEqual(first_generated_artifacts.len, std.mem.count(u8, manifest, "\"generated_check_command\":"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"generated_source_path\": \"" ++ first_general_metal_q5_1_source_path ++ "\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"artifact_source_path\": \"" ++ first_general_metal_q5_1_artifact_source_path ++ "\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"generated_check_command\": \"" ++ first_general_metal_q5_1_check_command ++ "\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"generated_source_path\": \"" ++ first_lazy_benchmark.generated_source_path ++ "\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"artifact_source_path\": \"\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q4_0_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q4_0_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q4_1_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q4_1_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_0_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_0_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_1_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_1_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q4_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q4_artifact_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_artifact_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_bias_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_bias_artifact_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_bias_gelu_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_bias_gelu_artifact_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_relu_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_relu_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q2_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q2_artifact_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q2_bias_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q2_bias_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q2_bias_gelu_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q2_bias_gelu_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q3_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q3_artifact_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q3_bias_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q3_bias_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q3_bias_gelu_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q3_bias_gelu_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_1_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_1_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_k_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_k_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_artifact_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_bias_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_bias_artifact_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_bias_gelu_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_bias_gelu_artifact_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q6_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q6_artifact_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q6_bias_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q6_bias_artifact_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q6_bias_gelu_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q6_bias_gelu_artifact_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, runtime_evidence_count, "\"runtime_evidence_command\":"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, runtime_route_evidence_count, "\"runtime_route_evidence_command\": \"zig build quant-kernel-metal-runtime-check"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, runtime_route_evidence_count, "--runtime-route-kernel"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, promotion_evidence_count, "\"promotion_evidence_command\":"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, promotion_check_count, "\"promotion_check_command\":"));
    const blocked_unsupported_handwritten_count = metalPromotionBlockerEvidenceCount("unsupported_handwritten_baseline");
    try std.testing.expectEqual(first_generated_artifacts.len, std.mem.count(u8, manifest, "\"promotion_policy\":"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, route_only_no_promotion_policy_count, "\"promotion_policy\": \"route_evidence_only_no_promotion\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, speedup_policy_count, "\"promotion_policy\": \"speedup_vs_handwritten\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, promoted_speedup_policy_count, "\"promotion_policy\": \"promoted_speedup_vs_handwritten\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, production_disabled_policy_count, "\"promotion_policy\": \"production_disabled\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, cuda_policy_count, "\"promotion_policy\": \"driver_artifact_policy\""));
    try std.testing.expectEqual(blocked_unsupported_handwritten_count, route_only_no_promotion_policy_count);
    try std.testing.expectEqual(first_metal_runtime_evidence_count, promoted_speedup_policy_count);
    try std.testing.expect(route_only_no_promotion_policy_count > 0);
    try std.testing.expect(speedup_policy_count > 0);
    try std.testing.expect(promoted_speedup_policy_count > 0);
    try std.testing.expect(cuda_policy_count > 0);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, metal_evidence_count, first_metal_runtime_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_metal_local_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_metal_runtime_route_all_build_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_metal_runtime_route_all_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_metal_runtime_route_all_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "--require-runtime-route-all"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_metal_production_regression_build_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_metal_production_regression_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_metal_production_regression_evidence_path));
    try std.testing.expectEqual(first_metal_runtime_evidence_count, std.mem.count(u8, manifest, "\"production_regression_checked\": true"));
    try std.testing.expectEqual(first_generated_artifacts.len - first_metal_runtime_evidence_count, std.mem.count(u8, manifest, "\"production_regression_checked\": false"));
    try std.testing.expectEqual(first_metal_runtime_evidence_count + 1, std.mem.count(u8, manifest, first_metal_production_regression_build_command));
    try std.testing.expectEqual(first_generated_artifacts.len - first_metal_runtime_evidence_count, std.mem.count(u8, manifest, "\"production_regression_command\": \"\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, metal_evidence_count, first_metal_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, metal_evidence_count, first_metal_promotion_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, metal_evidence_count, "--require-kernel"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_evidence_records\": ["));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q4_0_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q4_1_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_0_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_1_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_bias_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_bias_artifact_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_bias_gelu_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_bias_gelu_artifact_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_relu_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q2_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q2_bias_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q2_bias_gelu_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q3_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q3_bias_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q3_bias_gelu_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q4_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q4_artifact_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_1_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_k_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q8_artifact_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_artifact_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_bias_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q5_bias_gelu_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q6_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q6_bias_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q6_bias_gelu_promotion_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q6_artifact_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_general_metal_q6_bias_gelu_artifact_source_path));
    const blocked_metal_promotion_count = first_metal_promotion_blocker_evidence_count;
    const dev_only_candidate_count = first_generated_artifacts.len - first_metal_runtime_evidence_count - blocked_metal_promotion_count;
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, dev_only_candidate_count, "\"candidate_status\": \"dev_only_candidate\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, first_metal_runtime_evidence_count, "\"candidate_status\": \"promoted\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, blocked_metal_promotion_count, "\"candidate_status\": \"blocked_by_evidence\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, first_generated_artifacts.len - first_metal_runtime_evidence_count, "\"promotion_ready\": false"));
    const runtime_wired_artifacts = first_metal_runtime_route_all_expected_case_count / 2;
    const awaiting_metal_promotion_count = runtime_wired_artifacts - first_metal_runtime_evidence_count - blocked_metal_promotion_count;
    const blocked_speedup_count = metalPromotionBlockerEvidenceCount("speedup_gate_missing");
    const blocked_evidence_path_count = metalPromotionBlockerEvidencePathCount();
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, awaiting_metal_promotion_count, "\"promotion_blocker\": \"awaiting_metal_promotion_evidence\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, blocked_speedup_count, "\"promotion_blocker\": \"speedup_gate_missing\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, blocked_unsupported_handwritten_count, "\"promotion_blocker\": \"unsupported_handwritten_baseline\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, first_generated_artifacts.len - runtime_wired_artifacts, "\"promotion_blocker\": \"production_disabled\""));
    try std.testing.expectEqual(first_generated_artifacts.len, std.mem.count(u8, manifest, "\"promotion_blocker_evidence_path\":"));
    try std.testing.expectEqual(blocked_evidence_path_count, std.mem.count(u8, manifest, "\"promotion_blocker_evidence_path\": \"/private/tmp/antfly-quant-metal-"));
    try std.testing.expectEqual(first_generated_artifacts.len - blocked_evidence_path_count, std.mem.count(u8, manifest, "\"promotion_blocker_evidence_path\": \"\""));
    try std.testing.expectEqual(first_generated_artifacts.len, std.mem.count(u8, manifest, "\"promotion_blocker_check_command\":"));
    try std.testing.expectEqual(blocked_evidence_path_count, std.mem.count(u8, manifest, "\"promotion_blocker_check_command\": \"zig build quant-kernel-metal-runtime-check"));
    try std.testing.expectEqual(blocked_evidence_path_count, std.mem.count(u8, manifest, "--require-evidence-kernel"));
    try std.testing.expectEqual(first_generated_artifacts.len - blocked_evidence_path_count, std.mem.count(u8, manifest, "\"promotion_blocker_check_command\": \"\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "--require-evidence-kernel " ++ first_general_metal_q4_0_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, runtime_wired_artifacts, "\"runtime_wired\": true"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, first_generated_artifacts.len - runtime_wired_artifacts, "\"runtime_wired\": false"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, first_generated_artifacts.len - runtime_wired_artifacts, "\"runtime_gate_env\": \"\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q8_0_SMALL_BATCH\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_DISABLE_ANTFLY_Q8_0_SMALL_BATCH\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q8_0_SMALL_BATCH_RELU\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q2_K_SMALL_BATCH\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_DISABLE_ANTFLY_Q2_K_SMALL_BATCH\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q2_K_SMALL_BATCH_BIAS\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q2_K_SMALL_BATCH_BIAS_GELU\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q8_0_SMALL_BATCH_BIAS\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q8_0_SMALL_BATCH_BIAS_GELU\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q3_K_SMALL_BATCH\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_DISABLE_ANTFLY_Q3_K_SMALL_BATCH\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q3_K_SMALL_BATCH_BIAS\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q3_K_SMALL_BATCH_BIAS_GELU\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q4_0_SMALL_BATCH\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q4_1_SMALL_BATCH\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_DISABLE_ANTFLY_Q4_1_SMALL_BATCH\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q5_0_SMALL_BATCH\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_DISABLE_ANTFLY_Q5_0_SMALL_BATCH\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q5_1_SMALL_BATCH\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_DISABLE_ANTFLY_Q5_1_SMALL_BATCH\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q8_1_SMALL_BATCH\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_DISABLE_ANTFLY_Q8_1_SMALL_BATCH\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q8_K_SMALL_BATCH\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_DISABLE_ANTFLY_Q8_K_SMALL_BATCH\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q4_K_SMALL_BATCH\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_DISABLE_ANTFLY_Q4_K_SMALL_BATCH\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q4_K_SMALL_BATCH_BIAS\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q4_K_SMALL_BATCH_BIAS_GELU\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q5_K_SMALL_BATCH\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_DISABLE_ANTFLY_Q5_K_SMALL_BATCH\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q5_K_SMALL_BATCH_BIAS\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q5_K_SMALL_BATCH_BIAS_GELU\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q6_K_SMALL_BATCH\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_DISABLE_ANTFLY_Q6_K_SMALL_BATCH\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q6_K_SMALL_BATCH_BIAS\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"runtime_gate_env\": \"TERMITE_METAL_ENABLE_ANTFLY_Q6_K_SMALL_BATCH_BIAS_GELU\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, metal_evidence_count, "\"metal_promotion_min_speedup\": 1.1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, metal_evidence_count, "\"metal_promotion_repeat_runs\": 5"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, metal_evidence_count, "\"metal_promotion_warmup_repeat_runs\": 2"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_metal_model_local_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_metal_model_generated_route_check_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_model_generated_q8_0_small_batch_min\": 1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_model_generated_q4_0_small_batch_min\": 1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_metal_industry_local_check_command));
}

test "quant kernel compiler checked-in Metal evidence matches generated source" {
    try std.testing.expectEqual(@as(usize, 8), first_metal_runtime_evidence_count);
    for (first_metal_runtime_evidence) |evidence| {
        try std.testing.expectEqual(metal_promotion_repeat_runs, evidence.repeat_runs);
        try std.testing.expect(evidence.correctness_passed);
        try std.testing.expect(evidence.generated_route_checked);
        try std.testing.expectEqual(metalProviderRouteRequiredForKernel(evidence.kernel_id), evidence.provider_route_checked);
        try std.testing.expect(evidence.benchmark_passed);
        try std.testing.expect(evidence.production_enabled);
        try std.testing.expect(evidence.promotion_ready);

        const artifact_source_path = metalArtifactSourcePathForKernel(evidence.kernel_id) orelse return error.MissingMetalArtifactPath;
        try std.testing.expectEqualStrings(artifact_source_path, evidence.artifact_source_path);

        var found = false;
        for (first_generated_artifacts) |artifact| {
            if (!std.mem.eql(u8, artifact.kernel_id, evidence.kernel_id)) continue;
            found = true;
            try std.testing.expectEqual(artifactSourceFingerprint(artifact), evidence.source_fingerprint);
            try std.testing.expect(metalRuntimeEvidenceFor(artifact, &first_metal_runtime_evidence) != null);
        }
        try std.testing.expect(found);
    }
}

test "quant kernel compiler checked-in Metal blocker evidence matches generated candidates" {
    try std.testing.expectEqual(@as(usize, 17), first_metal_promotion_blocker_evidence_count);
    try std.testing.expectEqual(@as(usize, 6), metalPromotionBlockerEvidenceCount("speedup_gate_missing"));
    try std.testing.expectEqual(@as(usize, 5), metalPromotionBlockerEvidenceCount("unsupported_handwritten_baseline"));
    try std.testing.expectEqual(@as(usize, 6), metalPromotionBlockerEvidenceCount("unstable_benchmark_timing"));
    try std.testing.expectEqual(@as(usize, 12), metalPromotionBlockerEvidencePathCount());
    try std.testing.expectEqual(@as(usize, 24), first_metal_promotion_blocker_evidence_expected_case_count);
    try std.testing.expectEqual(first_metal_promotion_blocker_evidence_expected_case_count, first_metal_promotion_blocker_evidence_expected_route_ready_count);
    for (first_metal_promotion_blocker_evidence) |evidence| {
        const artifact = generatedArtifactForKernel(.metal, evidence.kernel_id) orelse return error.MissingMetalBlockerArtifact;
        try std.testing.expectEqual(Backend.metal, artifact.backend);
        try std.testing.expect(artifactRuntimeWired(artifact));
        try std.testing.expect(!artifactHasPromotionEvidence(artifact));
        try std.testing.expectEqualStrings(evidence.blocker, artifactPromotionBlocker(artifact));
        try std.testing.expectEqualStrings(evidence.evidence_path, artifactPromotionBlockerEvidencePath(artifact));
        if (std.mem.eql(u8, evidence.blocker, metal_blocker_unsupported_handwritten)) {
            try std.testing.expectEqualStrings("", evidence.evidence_path);
            try std.testing.expect(artifactRuntimeWired(artifact));
        }
        if (evidence.evidence_path.len != 0) {
            try std.testing.expect(std.mem.startsWith(u8, evidence.evidence_path, "/private/tmp/antfly-quant-metal-"));
            try std.testing.expect(std.mem.containsAtLeast(u8, evidence.evidence_path, 1, evidence.kernel_id));
            try std.testing.expect(std.mem.endsWith(u8, evidence.evidence_path, "-promotion-evidence.json"));
        }
    }
}

test "quant kernel compiler Metal promotion requires route evidence where routed" {
    var artifact: GeneratedArtifact = undefined;
    var found_artifact = false;
    for (first_generated_artifacts) |candidate| {
        if (!std.mem.eql(u8, candidate.kernel_id, first_general_metal_q6_kernel_id)) continue;
        artifact = candidate;
        found_artifact = true;
        break;
    }
    try std.testing.expect(found_artifact);
    artifact.production_enabled = true;
    artifact.source_path = first_general_metal_q6_artifact_source_path;
    artifact.check_command = first_general_metal_q6_artifact_check_command;
    const evidence = MetalRuntimeEvidence{
        .kernel_id = first_general_metal_q6_kernel_id,
        .source_path = first_general_metal_q6_source_path,
        .artifact_source_path = first_general_metal_q6_artifact_source_path,
        .source_fingerprint = artifactSourceFingerprint(artifact),
        .check_command = first_general_metal_q6_check_command,
        .runtime_evidence_command = first_general_metal_q6_promotion_evidence_command,
        .promotion_check_command = first_general_metal_q6_promotion_check_command,
        .repeat_runs = metal_promotion_repeat_runs,
        .correctness_passed = true,
        .generated_route_checked = true,
        .provider_route_checked = true,
        .benchmark_passed = true,
        .measured_speedup = metal_promotion_min_speedup + 0.1,
        .minimum_repeat_speedup = metal_promotion_min_speedup + 0.1,
        .production_enabled = true,
        .promotion_ready = true,
    };
    var missing_generated_evidence = [_]MetalRuntimeEvidence{evidence};
    missing_generated_evidence[0].generated_route_checked = false;
    try std.testing.expectEqualStrings(
        "missing_metal_generated_route_evidence",
        metalArtifactPromotionBlockerWithEvidence(artifact, &missing_generated_evidence),
    );
    var stale_evidence = [_]MetalRuntimeEvidence{evidence};
    stale_evidence[0].provider_route_checked = false;
    try std.testing.expectEqualStrings(
        "missing_metal_provider_route_evidence",
        metalArtifactPromotionBlockerWithEvidence(artifact, &stale_evidence),
    );
}

test "quant kernel compiler route-all covers every generated Metal artifact" {
    var metal_artifacts: usize = 0;
    for (first_generated_artifacts) |artifact| {
        if (artifact.backend != .metal) continue;
        metal_artifacts += 1;
        try std.testing.expect(artifactRuntimeWired(artifact));
    }
    try std.testing.expect(metal_artifacts > first_metal_runtime_evidence_count);
    try std.testing.expectEqual(metal_artifacts * 2, first_metal_runtime_route_all_expected_case_count);
}

test "quant kernel compiler decode descriptors cover required stages" {
    for (descriptor_formats) |format| {
        const spec = specFor(format).?;
        try std.testing.expect(spec.decode_ops.len >= 4);
        try std.testing.expectEqual(DecodeOpKind.load_packed_bytes, spec.decode_ops[0].kind);
        try std.testing.expectEqual(DecodeOpKind.dequant_lane, spec.decode_ops[spec.decode_ops.len - 1].kind);

        for (spec.block_fields) |field| {
            try std.testing.expect(std.mem.containsAtLeast(u8, spec.decode_ops[0].expression, 1, field.name));
        }

        var has_scale = false;
        var has_min = false;
        var has_lane = false;
        for (spec.decode_ops) |op| {
            try std.testing.expect(op.expression.len != 0);
            has_scale = has_scale or op.kind == .extract_scale;
            has_min = has_min or op.kind == .extract_min;
            has_lane = has_lane or op.kind == .extract_quant_lane;
        }
        try std.testing.expect(has_scale);
        try std.testing.expect(has_lane);
        try std.testing.expectEqual(format == .q2_k or format == .q4_1 or format == .q5_1 or format == .q4_k or format == .q5_k, has_min);
    }
}

test "quant kernel compiler builds deterministic IR for the first lazy target" {
    const ir = buildIr(.q4_k, .rows_2_8, .bias_gelu).?;
    try std.testing.expectEqual(quant_matmul.DispatchKind.small_batch, ir.dispatch);
    try std.testing.expectEqual(Epilogue.bias_gelu, ir.epilogue);
    try std.testing.expectEqualSlices(IROp, &ir_ops_bias_gelu, ir.ops);

    const route = loweringFor(.cuda, .q4_k, .rows_2_8, .bias_gelu);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, route.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, route.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, route.fallback_reason);
    try std.testing.expectEqualStrings(first_lazy_benchmark.handwritten_baseline, route.production_kernel_id);
    try std.testing.expectEqualStrings(first_lazy_benchmark.generated_kernel_id, route.kernel_id);
}

test "quant kernel compiler IR covers descriptor-supported matrix" {
    for (descriptor_formats) |format| {
        const spec = specFor(format).?;
        for (first_row_buckets) |row_bucket| {
            for (first_epilogues) |epilogue| {
                const ir = buildIr(format, row_bucket, epilogue);
                if (!spec.supportsEpilogue(epilogue)) {
                    try std.testing.expect(ir == null);
                    continue;
                }

                const planned = ir orelse return error.MissingQuantKernelIR;
                try std.testing.expectEqual(format, planned.format);
                try std.testing.expectEqual(row_bucket, planned.row_bucket);
                try std.testing.expectEqual(dispatchForRowBucket(row_bucket).?, planned.dispatch);
                try std.testing.expectEqual(epilogue, planned.epilogue);
                try std.testing.expectEqualSlices(IROp, irOpsForEpilogue(epilogue), planned.ops);
            }
        }
    }

    try std.testing.expect(buildIr(.q4_k, .rows_0, .none) == null);
    try std.testing.expect(buildIr(.unknown, .rows_2_8, .none) == null);
}

test "quant kernel compiler exposes the first static route registry" {
    try std.testing.expectEqual(first_coverage.len * first_backends.len, first_registry.entries.len);

    const route = first_registry.lookup(.cuda, .q4_k, .rows_2_8, .bias_gelu, .small_batch).?;
    try std.testing.expectEqual(LoweringRoute.handwritten_production, route.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, route.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, route.fallback_reason);

    try std.testing.expect(first_registry.lookup(.cuda, .unknown, .rows_2_8, .bias_gelu, .small_batch) == null);
}

test "quant kernel compiler registry entries have unique route keys" {
    for (first_registry.entries, 0..) |a, i| {
        for (first_registry.entries[i + 1 ..]) |b| {
            try std.testing.expect(!(a.backend == b.backend and
                a.format == b.format and
                a.row_bucket == b.row_bucket and
                a.epilogue == b.epilogue and
                a.schedule.dispatch == b.schedule.dispatch));
        }
    }
}

test "quant kernel compiler plan ids match route keys" {
    for (first_registry.entries) |entry| {
        try std.testing.expectEqual(entry.backend, entry.plan_id.backend);
        try std.testing.expectEqual(entry.format, entry.plan_id.format);
        try std.testing.expectEqual(entry.row_bucket, entry.plan_id.row_bucket);
        try std.testing.expectEqual(entry.epilogue, entry.plan_id.epilogue);
        try std.testing.expectEqual(entry.schedule.dispatch, entry.plan_id.dispatch);
    }

    const unsupported_shape = loweringFor(.cuda, .q4_k, .rows_0, .bias_gelu);
    try std.testing.expectEqual(quant_matmul.DispatchKind.scalar, unsupported_shape.plan_id.dispatch);
    try std.testing.expectEqual(.rows_0, unsupported_shape.plan_id.row_bucket);
}

test "quant kernel compiler plan ids have stable diagnostic names" {
    const lazy = loweringFor(.cuda, .q4_k, .rows_2_8, .bias_gelu);
    const lazy_name = try planIdName(std.testing.allocator, lazy.plan_id);
    defer std.testing.allocator.free(lazy_name);
    try std.testing.expectEqualStrings("cuda/q4_k/rows_2_8/bias_gelu/small_batch", lazy_name);

    const unsupported = loweringFor(.metal, .q2_k, .rows_0, .argmax);
    const unsupported_name = try planIdName(std.testing.allocator, unsupported.plan_id);
    defer std.testing.allocator.free(unsupported_name);
    try std.testing.expectEqualStrings("metal/q2_k/rows_0/argmax/scalar", unsupported_name);
}

test "quant kernel compiler routes and fallback reasons have stable diagnostic names" {
    try std.testing.expectEqualStrings("generated_production", loweringRouteName(.generated_production));
    try std.testing.expectEqualStrings("generated_dev_candidate", loweringRouteName(.generated_dev_candidate));
    try std.testing.expectEqualStrings("handwritten_production", loweringRouteName(.handwritten_production));
    try std.testing.expectEqualStrings("unsupported", loweringRouteName(.unsupported));

    try std.testing.expectEqualStrings("none", fallbackReasonName(.none));
    try std.testing.expectEqualStrings("unsupported_format", fallbackReasonName(.unsupported_format));
    try std.testing.expectEqualStrings("unsupported_shape", fallbackReasonName(.unsupported_shape));
    try std.testing.expectEqualStrings("unsupported_epilogue", fallbackReasonName(.unsupported_epilogue));
    try std.testing.expectEqualStrings("unsupported_backend", fallbackReasonName(.unsupported_backend));
    try std.testing.expectEqualStrings("generated_artifact_missing", fallbackReasonName(.generated_artifact_missing));
    try std.testing.expectEqualStrings("generated_runtime_not_wired", fallbackReasonName(.generated_runtime_not_wired));
    try std.testing.expectEqualStrings("tensor_core_repack_required", fallbackReasonName(.tensor_core_repack_required));
}

test "quant kernel compiler lowering diagnostic includes route identity" {
    const lowering = registryLoweringFor(.cuda, .q4_k, .rows_2_8, .bias_gelu, .small_batch);
    const diagnostic = try loweringDiagnostic(std.testing.allocator, lowering);
    defer std.testing.allocator.free(diagnostic);

    try std.testing.expect(std.mem.containsAtLeast(u8, diagnostic, 1, "plan=cuda/q4_k/rows_2_8/bias_gelu/small_batch"));
    try std.testing.expect(std.mem.containsAtLeast(u8, diagnostic, 1, "production=handwritten_production"));
    try std.testing.expect(std.mem.containsAtLeast(u8, diagnostic, 1, "candidate=generated_dev_candidate"));
    try std.testing.expect(std.mem.containsAtLeast(u8, diagnostic, 1, "fallback=generated_artifact_missing"));
    try std.testing.expect(std.mem.containsAtLeast(u8, diagnostic, 1, first_lazy_benchmark.handwritten_baseline));
    try std.testing.expect(std.mem.containsAtLeast(u8, diagnostic, 1, first_lazy_benchmark.generated_kernel_id));
    try std.testing.expect(std.mem.containsAtLeast(u8, diagnostic, 1, first_lazy_benchmark.generated_source_path));
}

test "quant kernel compiler Metal runtime route summary maps counters to route metadata" {
    const json = try metalRuntimeRouteSummaryJson(std.testing.allocator);
    defer std.testing.allocator.free(json);
    var parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, json, .{});
    defer parsed.deinit();

    const routes = parsed.value.object;
    const q8_bias = routes.get("q8_0_small_batch_bias").?.object;
    try std.testing.expectEqualStrings("metal/q8_0/rows_2_8/bias/small_batch", q8_bias.get("plan_id").?.string);
    try std.testing.expectEqualStrings(first_general_metal_q8_bias_kernel_id, q8_bias.get("kernel_id").?.string);
    try std.testing.expectEqualStrings(first_general_metal_q8_bias_kernel_id, q8_bias.get("production_kernel_id").?.string);
    try std.testing.expectEqualStrings("generated_production", q8_bias.get("production_route").?.string);
    try std.testing.expectEqualStrings("unsupported", q8_bias.get("candidate_route").?.string);
    try std.testing.expectEqualStrings("none", q8_bias.get("fallback_reason").?.string);
    try std.testing.expectEqual(true, q8_bias.get("promotion_ready").?.bool);

    const q4_bias_gelu = routes.get("q4_k_small_batch_bias_gelu").?.object;
    try std.testing.expectEqualStrings(first_lazy_metal_kernel_id, q4_bias_gelu.get("kernel_id").?.string);
    try std.testing.expectEqualStrings("metal_handwritten_quant_matmul", q4_bias_gelu.get("production_kernel_id").?.string);
    try std.testing.expectEqualStrings("handwritten_production", q4_bias_gelu.get("production_route").?.string);
    try std.testing.expectEqualStrings("generated_dev_candidate", q4_bias_gelu.get("candidate_route").?.string);
    try std.testing.expectEqualStrings("generated_artifact_missing", q4_bias_gelu.get("fallback_reason").?.string);
    try std.testing.expectEqual(false, q4_bias_gelu.get("promotion_ready").?.bool);

    const q4_0 = routes.get("q4_0_small_batch").?.object;
    try std.testing.expectEqualStrings(first_general_metal_q4_0_source_path, q4_0.get("generated_source_path").?.string);
}

test "quant kernel compiler planned production routes are named" {
    for (first_registry.entries) |entry| {
        switch (entry.production_route) {
            .generated_production, .handwritten_production => try std.testing.expect(entry.production_kernel_id.len != 0),
            .generated_dev_candidate => try std.testing.expect(false),
            .unsupported => try std.testing.expect(entry.production_kernel_id.len == 0),
        }
    }
}

test "quant kernel compiler names checked-in CUDA production route symbols" {
    const cases = [_]struct {
        format: quant_matmul.Format,
        row_bucket: quant_matmul.RowBucket,
        epilogue: Epilogue,
        symbol: []const u8,
    }{
        .{ .format = .q4_k, .row_bucket = .rows_1, .epilogue = .none, .symbol = "termite_linear_q4_k_f32_tile4" },
        .{ .format = .q4_k, .row_bucket = .rows_2_8, .epilogue = .none, .symbol = "termite_linear_q4_k_f32_tile4" },
        .{ .format = .q4_k, .row_bucket = .rows_1, .epilogue = .bias, .symbol = "termite_linear_q4_k_bias_f32_tile4" },
        .{ .format = .q4_k, .row_bucket = .rows_2_8, .epilogue = .bias, .symbol = "termite_linear_q4_k_bias_f32_tile4_r2" },
        .{ .format = .q4_k, .row_bucket = .rows_2_8, .epilogue = .bias_gelu, .symbol = first_lazy_benchmark.handwritten_baseline },
        .{ .format = .q6_k, .row_bucket = .rows_2_8, .epilogue = .none, .symbol = "termite_linear_q6_k_f32_tile4" },
    };

    const contents = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "src/ops/cuda/artifacts/inference_cuda_kernels.cu", std.testing.allocator, .limited(1024 * 1024));
    defer std.testing.allocator.free(contents);
    for (cases) |case| {
        const route = loweringFor(.cuda, case.format, case.row_bucket, case.epilogue);
        try std.testing.expectEqual(LoweringRoute.handwritten_production, route.production_route);
        try std.testing.expectEqualStrings(case.symbol, route.production_kernel_id);
        try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, case.symbol));
    }
}

test "quant kernel compiler keeps shape-dependent CUDA routes generic" {
    try std.testing.expectEqualStrings("cuda_handwritten_quant_matmul", loweringFor(.cuda, .q8_0, .rows_2_8, .bias).production_kernel_id);
    try std.testing.expectEqualStrings("cuda_handwritten_quant_matmul", loweringFor(.cuda, .q4_k, .rows_9_64, .bias).production_kernel_id);
}

test "quant kernel compiler dev candidates never occupy production route slots" {
    for (first_registry.entries) |entry| {
        try std.testing.expect(entry.production_route != .generated_dev_candidate);
        try std.testing.expect(entry.candidate_route != .generated_production);
    }
}

test "quant kernel compiler generated candidates are named only when present" {
    for (first_registry.entries) |entry| {
        switch (entry.candidate_route) {
            .generated_dev_candidate => {
                try std.testing.expect(entry.kernel_id.len != 0);
                try std.testing.expect(entry.candidate_source_path.len != 0);
            },
            .generated_production, .handwritten_production, .unsupported => {
                try std.testing.expect(entry.kernel_id.len == 0);
                try std.testing.expect(entry.candidate_source_path.len == 0);
            },
        }
    }
}

test "quant kernel compiler routes honor descriptor schedules" {
    for (first_registry.entries) |entry| {
        const spec = specFor(entry.format) orelse {
            try std.testing.expectEqual(LoweringRoute.unsupported, entry.production_route);
            continue;
        };
        if (!spec.supportsSchedule(entry.schedule.dispatch)) {
            try std.testing.expectEqual(LoweringRoute.unsupported, entry.production_route);
            try std.testing.expectEqual(LoweringRoute.unsupported, entry.candidate_route);
            try std.testing.expectEqual(FallbackReason.unsupported_shape, entry.fallback_reason);
            continue;
        }
        if (entry.production_route != .unsupported or entry.candidate_route != .unsupported) {
            try std.testing.expect(spec.supportsSchedule(entry.schedule.dispatch));
        }
    }
}

test "quant kernel compiler coverage includes every non-empty row bucket" {
    const required = [_]quant_matmul.RowBucket{ .rows_1, .rows_2_8, .rows_9_64, .rows_65_plus };
    for (required) |row_bucket| {
        var found = false;
        for (first_coverage) |case| {
            if (case.row_bucket == row_bucket) {
                found = true;
                break;
            }
        }
        try std.testing.expect(found);
    }
}

test "quant kernel compiler row buckets follow quant matmul planner" {
    const samples = [_]struct {
        rows: usize,
        bucket: quant_matmul.RowBucket,
        dispatch: ?quant_matmul.DispatchKind,
    }{
        .{ .rows = 0, .bucket = .rows_0, .dispatch = null },
        .{ .rows = 1, .bucket = .rows_1, .dispatch = .mmv },
        .{ .rows = 2, .bucket = .rows_2_8, .dispatch = .small_batch },
        .{ .rows = 8, .bucket = .rows_2_8, .dispatch = .small_batch },
        .{ .rows = 9, .bucket = .rows_9_64, .dispatch = .mm },
        .{ .rows = 64, .bucket = .rows_9_64, .dispatch = .mm },
        .{ .rows = 65, .bucket = .rows_65_plus, .dispatch = .mm },
    };

    for (samples) |sample| {
        try std.testing.expectEqual(sample.bucket, quant_matmul.rowBucket(sample.rows));
        try std.testing.expectEqual(sample.dispatch, dispatchForRowBucket(sample.bucket));
        const planned = quant_matmul.plan(.{
            .rows = sample.rows,
            .in_dim = 256,
            .out_dim = 128,
            .format = .q4_k,
        });
        try std.testing.expectEqual(sample.bucket, planned.row_bucket);
        try std.testing.expectEqual(sample.dispatch orelse .scalar, planned.dispatch);
    }
}

test "quant kernel compiler conformance matrix pins route expectations" {
    try std.testing.expectEqual(first_coverage.len, first_conformance.len);
    for (first_conformance, first_coverage) |conf, cov| {
        try std.testing.expectEqual(cov.format, conf.format);
        try std.testing.expectEqual(cov.row_bucket, conf.row_bucket);
        try std.testing.expectEqual(cov.epilogue, conf.epilogue);
        try std.testing.expectEqual(dispatchForRowBucket(cov.row_bucket).?, conf.dispatch);
        try std.testing.expectEqual(tensorTypeForFormat(cov.format).?, conf.reference_tensor_type);
        try std.testing.expect(conf.tolerance_abs > 0.0 and conf.tolerance_abs <= 0.001);

        const cuda = loweringFor(.cuda, cov.format, cov.row_bucket, cov.epilogue);
        try std.testing.expectEqual(cuda.production_route, conf.cuda_route);
        try std.testing.expectEqual(cuda.candidate_route, conf.cuda_candidate_route);
        try std.testing.expectEqual(cuda.fallback_reason, conf.cuda_fallback_reason);

        const metal = loweringFor(.metal, cov.format, cov.row_bucket, cov.epilogue);
        try std.testing.expectEqual(metal.production_route, conf.metal_route);
        try std.testing.expectEqual(metal.candidate_route, conf.metal_candidate_route);
        try std.testing.expectEqual(metal.fallback_reason, conf.metal_fallback_reason);
    }
}

test "quant kernel compiler registry route summary is golden" {
    var by_backend = [_]PlanCounters{ .{}, .{} };
    for (first_registry.entries) |entry| {
        const counters = countersForLowering(entry);
        const index = @intFromEnum(entry.backend);
        addCountersToStats(&by_backend[index], counters);
    }

    const cuda = by_backend[@intFromEnum(@as(Backend, .cuda))];
    try std.testing.expectEqual(@as(usize, 1008), cuda.quant_kernel_planned_ops);
    try std.testing.expectEqual(@as(usize, 44), cuda.quant_kernel_handwritten_production);
    try std.testing.expectEqual(@as(usize, 0), cuda.quant_kernel_generated_production);
    try std.testing.expectEqual(@as(usize, 964), cuda.quant_kernel_unsupported_routes);
    try std.testing.expectEqual(@as(usize, 1), cuda.quant_kernel_generated_candidates);
    try std.testing.expectEqual(@as(usize, 1), cuda.quant_kernel_fallback_generated_artifact_missing);
    try std.testing.expectEqual(@as(usize, 0), cuda.quant_kernel_fallback_generated_runtime_not_wired);
    try std.testing.expectEqual(@as(usize, 0), cuda.quant_kernel_fallback_unsupported_format);
    try std.testing.expectEqual(@as(usize, 0), cuda.quant_kernel_fallback_unsupported_shape);
    try std.testing.expectEqual(@as(usize, 100), cuda.quant_kernel_fallback_unsupported_epilogue);
    try std.testing.expectEqual(@as(usize, 864), cuda.quant_kernel_fallback_unsupported_backend);
    try std.testing.expectEqual(@as(usize, 0), cuda.quant_kernel_fallback_tensor_core_repack_required);
    try std.testing.expectEqual(@as(usize, 964), cuda.quant_kernel_fallback_unsupported);

    const metal = by_backend[@intFromEnum(@as(Backend, .metal))];
    try std.testing.expectEqual(@as(usize, 1008), metal.quant_kernel_planned_ops);
    try std.testing.expectEqual(@as(usize, 104), metal.quant_kernel_handwritten_production);
    try std.testing.expectEqual(@as(usize, 8), metal.quant_kernel_generated_production);
    try std.testing.expectEqual(@as(usize, 896), metal.quant_kernel_unsupported_routes);
    try std.testing.expectEqual(@as(usize, 17), metal.quant_kernel_generated_candidates);
    try std.testing.expectEqual(@as(usize, 17), metal.quant_kernel_fallback_generated_artifact_missing);
    try std.testing.expectEqual(@as(usize, 0), metal.quant_kernel_fallback_generated_runtime_not_wired);
    try std.testing.expectEqual(@as(usize, 0), metal.quant_kernel_fallback_unsupported_format);
    try std.testing.expectEqual(@as(usize, 0), metal.quant_kernel_fallback_unsupported_shape);
    try std.testing.expectEqual(@as(usize, 356), metal.quant_kernel_fallback_unsupported_epilogue);
    try std.testing.expectEqual(@as(usize, 540), metal.quant_kernel_fallback_unsupported_backend);
    try std.testing.expectEqual(@as(usize, 0), metal.quant_kernel_fallback_tensor_core_repack_required);
    try std.testing.expectEqual(@as(usize, 896), metal.quant_kernel_fallback_unsupported);
}

test "quant kernel compiler route expectations are hand pinned" {
    for (first_route_expectations) |want| {
        const got = registryLoweringFor(want.backend, want.format, want.row_bucket, want.epilogue, want.dispatch);
        try std.testing.expectEqual(want.production_route, got.production_route);
        try std.testing.expectEqual(want.candidate_route, got.candidate_route);
        try std.testing.expectEqual(want.fallback_reason, got.fallback_reason);
        try std.testing.expectEqual(want.dispatch, got.plan_id.dispatch);
        try std.testing.expectEqual(want.dispatch, got.schedule.dispatch);
    }
}

test "quant kernel compiler benchmark manifest maps to conformance rows" {
    const manifest = try benchmarkManifestJson(std.testing.allocator);
    defer std.testing.allocator.free(manifest);
    try expectManifestArrayCount(manifest, "benchmark_count", "benchmarks", first_benchmarks.len);
    try expectManifestArrayCount(manifest, "evidence_count", "evidence_records", first_benchmark_evidence.len);
    try expectManifestArrayCount(manifest, "metal_evidence_count", "metal_evidence_records", first_metal_runtime_evidence.len);
    try expectManifestArrayCount(manifest, "metal_production_regression_expected_case_count", "metal_production_regression_cases", first_metal_production_benchmark_case_count);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"schema\": \"" ++ first_benchmark_manifest_schema ++ "\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_promotion_warmup_repeat_runs\": 2"));
    const production_regression_case_fingerprint_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_production_regression_case_fingerprint\": {d}", .{metalProductionBenchmarkCaseManifestFingerprint()});
    defer std.testing.allocator.free(production_regression_case_fingerprint_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, production_regression_case_fingerprint_field));
    try std.testing.expectEqual(first_metal_runtime_evidence_count * 2, first_metal_production_benchmark_case_count);
    const production_regression_kernel_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_production_regression_expected_kernel_count\": {d}", .{first_metal_production_regression_expected_kernel_count});
    defer std.testing.allocator.free(production_regression_kernel_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, production_regression_kernel_count_field));
    const production_regression_case_count_field = try std.fmt.allocPrint(std.testing.allocator, "\"metal_production_regression_expected_case_count\": {d}", .{first_metal_production_regression_expected_case_count});
    defer std.testing.allocator.free(production_regression_case_count_field);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, production_regression_case_count_field));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_metal_production_regression_build_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, first_metal_production_regression_evidence_command));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"kernel_id\": \"antfly_q6_k_small_batch_bias_msl_v1\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"name\": \"q6_k_rows_8_cols_7_bias\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"shape\": \"wide\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"out_dim\": 7"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"production_kernel_id\": \"antfly_q6_k_small_batch_bias_msl_v1\""));
    try std.testing.expect(!std.mem.containsAtLeast(u8, manifest, 1, "\"minimum_repeat_speedup\": 1.271992"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"generated_source_fingerprint\":"));

    for (first_benchmarks) |bench| {
        try std.testing.expect(!bench.production_enabled);
        var matched = false;
        for (first_conformance) |conf| {
            if (conf.format == bench.format and conf.row_bucket == bench.row_bucket and conf.epilogue == bench.epilogue) {
                matched = true;
                try std.testing.expectEqual(Backend.cuda, bench.backend);
                try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, conf.cuda_candidate_route);
                try std.testing.expectEqual(FallbackReason.generated_artifact_missing, conf.cuda_fallback_reason);
                try std.testing.expect(bench.generated_kernel_id.len != 0);
                try std.testing.expect(bench.generated_source_path.len != 0);
                try std.testing.expect(bench.generated_source_fingerprint != 0);
                try std.testing.expect(isPtxPath(bench.generated_ptx_path));
                try std.testing.expect(std.mem.containsAtLeast(u8, bench.generated_ptx_command, 1, "nvcc -ptx"));
                try std.testing.expect(std.mem.containsAtLeast(u8, bench.generated_ptx_command, 1, bench.generated_source_path));
                try std.testing.expect(std.mem.containsAtLeast(u8, bench.generated_ptx_command, 1, bench.generated_ptx_path));
                try std.testing.expect(std.mem.containsAtLeast(u8, bench.benchmark_command, 1, "zig-out/bin/antfly-inference bench-cuda"));
                try std.testing.expect(std.mem.containsAtLeast(u8, bench.benchmark_command, 1, "--warmup-iters 5"));
                try std.testing.expect(std.mem.containsAtLeast(u8, bench.benchmark_command, 1, "--measure-iters 50"));
                try std.testing.expect(std.mem.containsAtLeast(u8, bench.benchmark_command, 1, "--quant-compiler-lazy-target"));
                try std.testing.expectEqualStrings("--quant-compiler-generated-ptx", bench.generated_ptx_arg);
                try std.testing.expect(std.mem.containsAtLeast(u8, bench.benchmark_command, 1, bench.generated_ptx_arg));
                try std.testing.expect(std.mem.containsAtLeast(u8, bench.benchmark_command, 1, bench.generated_ptx_path));
                try std.testing.expect(std.mem.containsAtLeast(u8, bench.benchmark_command, 1, "--quant-compiler-evidence-out"));
                try std.testing.expect(bench.handwritten_baseline.len != 0);
                try std.testing.expect(bench.correctness_tolerance_abs > 0.0 and bench.correctness_tolerance_abs <= 0.01);
                try std.testing.expect(bench.minimum_speedup >= 1.0);
                try std.testing.expectEqual(@as(f64, 0.0), benchmarkMeasuredSpeedup(bench));
                try std.testing.expect(bench.correctness_evidence_path.len == 0);
                try std.testing.expect(bench.benchmark_evidence_path.len == 0);
                try std.testing.expect(bench.benchmark_mode.len == 0);
                try std.testing.expect(!benchmarkHasPromotionEvidence(bench));
                try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"promotion_ready\": false"));
                try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"promotion_blocker\": \"production_disabled\""));
                break;
            }
        }
        try std.testing.expect(matched);
    }
}

test "quant kernel compiler conformance manifest serializes the route matrix" {
    const manifest = try conformanceManifestJson(std.testing.allocator);
    defer std.testing.allocator.free(manifest);

    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "antfly.quant_kernel_conformance.v1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"case_count\": 1008"));
    try expectManifestArrayCount(manifest, "case_count", "cases", first_conformance.len);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"format_count\": 28"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"row_bucket_count\": 4"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"epilogue_count\": 9"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"backend_count\": 2"));
    try expectManifestNestedInteger(manifest, "cuda_route_summary", "quant_kernel_planned_ops", 1008);
    try expectManifestNestedInteger(manifest, "cuda_route_summary", "quant_kernel_generated_candidates", 1);
    try expectManifestNestedInteger(manifest, "cuda_route_summary", "quant_kernel_fallback_generated_artifact_missing", 1);
    try expectManifestNestedInteger(manifest, "cuda_route_summary", "quant_kernel_fallback_generated_runtime_not_wired", 0);
    try expectManifestNestedInteger(manifest, "cuda_route_summary", "quant_kernel_fallback_unsupported_epilogue", 100);
    try expectManifestNestedInteger(manifest, "metal_route_summary", "quant_kernel_planned_ops", 1008);
    try expectManifestNestedInteger(manifest, "metal_route_summary", "quant_kernel_generated_production", 8);
    try expectManifestNestedInteger(manifest, "metal_route_summary", "quant_kernel_generated_candidates", 17);
    try expectManifestNestedInteger(manifest, "metal_route_summary", "quant_kernel_fallback_generated_artifact_missing", 17);
    try expectManifestNestedInteger(manifest, "metal_route_summary", "quant_kernel_fallback_generated_runtime_not_wired", 0);
    try expectManifestNestedInteger(manifest, "metal_route_summary", "quant_kernel_fallback_unsupported_epilogue", 356);
    try std.testing.expectEqual(first_conformance.len, std.mem.count(u8, manifest, "\"format\": "));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"format\": \"q4_k\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"block_values\": 256"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"block_bytes\": 144"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"accumulator_dtype\": \"f32\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"output_dtype\": \"f32\""));
    try std.testing.expectEqual(first_formats.len * first_row_buckets.len * 5, std.mem.count(u8, manifest, "\"reference_supported\": true"));
    try std.testing.expectEqual(first_formats.len * first_row_buckets.len * (coverage_epilogues.len - 5), std.mem.count(u8, manifest, "\"reference_supported\": false"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"row_bucket\": \"rows_2_8\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"epilogue\": \"pair\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"epilogue\": \"bias_gelu\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"tile_rows\": 2"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"tile_cols\": 4"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"vector_width\": 4"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"threads_per_block\": 256"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"cuda_candidate_tile_rows\": 1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"cuda_candidate_tile_cols\": 1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"cuda_candidate_threads_per_block\": 128"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_candidate_tile_rows\": 1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_candidate_tile_cols\": 1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_candidate_threads_per_block\": 32"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"shared_memory_bytes\": 0"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"register_pressure_hint\": 6"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"tensor_core_eligible\": false"));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"cuda_candidate_route\": \"generated_dev_candidate\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"cuda_production_kernel_id\": \"termite_linear_q4_k_bias_gelu_f32_tile4_r2\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"cuda_candidate_kernel_id\": \"antfly_q4_k_small_batch_bias_gelu_f32_v1\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"cuda_candidate_source_path\": \"src/ops/cuda/generated/quant_kernel_q4_k_small_batch_bias_gelu.cu\""));
    const cuda_candidate_fingerprint = try std.fmt.allocPrint(std.testing.allocator, "\"cuda_candidate_source_fingerprint\": {d}", .{artifactSourceFingerprint(first_generated_artifacts[0])});
    defer std.testing.allocator.free(cuda_candidate_fingerprint);
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, cuda_candidate_fingerprint));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"cuda_fallback_reason\": \"generated_artifact_missing\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_candidate_route\": \"generated_dev_candidate\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_production_kernel_id\": \"metal_handwritten_quant_matmul\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_candidate_kernel_id\": \"antfly_q4_0_small_batch_msl_v1\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_fallback_reason\": \"generated_artifact_missing\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_candidate_kernel_id\": \"antfly_q2_k_small_batch_bias_msl_v1\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_candidate_kernel_id\": \"antfly_q2_k_small_batch_bias_gelu_msl_v1\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_candidate_kernel_id\": \"antfly_q3_k_small_batch_bias_msl_v1\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_candidate_kernel_id\": \"antfly_q3_k_small_batch_bias_gelu_msl_v1\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, manifest, 1, "\"metal_candidate_kernel_id\": \"antfly_q8_0_small_batch_relu_msl_v1\""));
}

test "quant kernel compiler conformance fingerprints match generated artifacts" {
    var cuda_candidates: usize = 0;
    var metal_candidates: usize = 0;
    for (first_conformance) |conf| {
        const record = conformanceManifestRecord(conf);
        const cuda = loweringFor(.cuda, conf.format, conf.row_bucket, conf.epilogue);
        const metal = loweringFor(.metal, conf.format, conf.row_bucket, conf.epilogue);

        try std.testing.expectEqual(candidateSourceFingerprint(cuda), record.cuda_candidate_source_fingerprint);
        try std.testing.expectEqual(candidateSourceFingerprint(metal), record.metal_candidate_source_fingerprint);

        if (cuda.candidate_route == .generated_dev_candidate) {
            cuda_candidates += 1;
            try std.testing.expect(record.cuda_candidate_source_fingerprint != 0);
        } else {
            try std.testing.expectEqual(@as(u64, 0), record.cuda_candidate_source_fingerprint);
        }

        if (metal.candidate_route == .generated_dev_candidate) {
            metal_candidates += 1;
            try std.testing.expect(record.metal_candidate_source_fingerprint != 0);
        } else {
            try std.testing.expectEqual(@as(u64, 0), record.metal_candidate_source_fingerprint);
        }
    }
    try std.testing.expectEqual(@as(usize, 1), cuda_candidates);
    try std.testing.expectEqual(@as(usize, 17), metal_candidates);
}

test "quant kernel compiler benchmark promotion evidence is complete" {
    var bench = first_lazy_benchmark;
    try std.testing.expect(!benchmarkHasPromotionEvidence(bench));
    try std.testing.expectEqualStrings("production_disabled", benchmarkPromotionBlocker(bench));

    bench.production_enabled = true;
    try std.testing.expect(!benchmarkHasPromotionEvidence(bench));
    try std.testing.expectEqualStrings("dev_generated_source", benchmarkPromotionBlocker(bench));

    bench.generated_source_path = "src/ops/cuda/artifacts/inference_cuda_kernels.cu";
    try std.testing.expect(!benchmarkHasPromotionEvidence(bench));
    try std.testing.expectEqualStrings("ptx_command_missing_source", benchmarkPromotionBlocker(bench));

    bench.generated_ptx_command = "nvcc -ptx -arch=compute_75 src/ops/cuda/artifacts/inference_cuda_kernels.cu -o /tmp/antfly_q4_k_small_batch_bias_gelu_f32_v1.ptx";
    try std.testing.expect(!benchmarkHasPromotionEvidence(bench));
    try std.testing.expectEqualStrings("missing_correctness_evidence", benchmarkPromotionBlocker(bench));

    bench.correctness_evidence_path = first_lazy_benchmark_evidence_path;
    try std.testing.expect(!benchmarkHasPromotionEvidence(bench));
    try std.testing.expectEqualStrings("missing_benchmark_evidence", benchmarkPromotionBlocker(bench));

    var mismatched_evidence_path = bench;
    mismatched_evidence_path.benchmark_evidence_path = "src/ops/cuda/generated/evidence/q4_k_small_batch_bias_gelu_other.json";
    try std.testing.expect(!benchmarkHasPromotionEvidence(mismatched_evidence_path));
    try std.testing.expectEqualStrings("evidence_path_mismatch", benchmarkPromotionBlocker(mismatched_evidence_path));

    bench.benchmark_evidence_path = first_lazy_benchmark_evidence_path;
    try std.testing.expect(!benchmarkHasPromotionEvidence(bench));
    try std.testing.expectEqualStrings("missing_sequential_benchmark_evidence", benchmarkPromotionBlocker(bench));

    bench.benchmark_mode = "sequential";
    try std.testing.expect(!benchmarkHasPromotionEvidence(bench));
    try std.testing.expectEqualStrings("missing_matching_evidence_record", benchmarkPromotionBlocker(bench));

    const passing_evidence = [_]BenchmarkEvidence{.{
        .kernel_id = bench.generated_kernel_id,
        .generated_source_path = bench.generated_source_path,
        .generated_source_fingerprint = bench.generated_source_fingerprint,
        .generated_ptx_path = bench.generated_ptx_path,
        .generated_ptx_command = bench.generated_ptx_command,
        .benchmark_command = bench.benchmark_command,
        .correctness_evidence_path = bench.correctness_evidence_path,
        .benchmark_evidence_path = bench.benchmark_evidence_path,
        .benchmark_mode = bench.benchmark_mode,
        .repeat_runs = metal_promotion_repeat_runs,
        .correctness_passed = true,
        .benchmark_passed = true,
        .measured_speedup = 1.0,
    }};
    try std.testing.expect(std.mem.eql(u8, benchmarkPromotionBlockerWithEvidence(bench, &passing_evidence), "none"));

    var wrong_evidence_source = passing_evidence;
    wrong_evidence_source[0].generated_source_path = "src/ops/cuda/artifacts/other.cu";
    try std.testing.expectEqualStrings("missing_matching_evidence_record", benchmarkPromotionBlockerWithEvidence(bench, &wrong_evidence_source));

    var wrong_evidence_source_fingerprint = passing_evidence;
    wrong_evidence_source_fingerprint[0].generated_source_fingerprint = bench.generated_source_fingerprint +% 1;
    try std.testing.expectEqualStrings("missing_matching_evidence_record", benchmarkPromotionBlockerWithEvidence(bench, &wrong_evidence_source_fingerprint));

    var wrong_evidence_ptx = passing_evidence;
    wrong_evidence_ptx[0].generated_ptx_path = "/tmp/other.ptx";
    try std.testing.expectEqualStrings("missing_matching_evidence_record", benchmarkPromotionBlockerWithEvidence(bench, &wrong_evidence_ptx));

    var wrong_evidence_ptx_command = passing_evidence;
    wrong_evidence_ptx_command[0].generated_ptx_command = "nvcc -ptx -arch=compute_90 src/ops/cuda/generated/quant_kernel_q4_k_small_batch_bias_gelu.cu -o /tmp/antfly_q4_k_small_batch_bias_gelu_f32_v1.ptx";
    try std.testing.expectEqualStrings("missing_matching_evidence_record", benchmarkPromotionBlockerWithEvidence(bench, &wrong_evidence_ptx_command));

    var wrong_evidence_benchmark_command = passing_evidence;
    wrong_evidence_benchmark_command[0].benchmark_command = "zig-out/bin/antfly-inference bench-cuda --quant-compiler-lazy-target --quant-compiler-generated-ptx /tmp/antfly_q4_k_small_batch_bias_gelu_f32_v1.ptx";
    try std.testing.expectEqualStrings("missing_matching_evidence_record", benchmarkPromotionBlockerWithEvidence(bench, &wrong_evidence_benchmark_command));

    var single_repeat_evidence = passing_evidence;
    single_repeat_evidence[0].repeat_runs = 1;
    try std.testing.expectEqualStrings("insufficient_benchmark_repeats", benchmarkPromotionBlockerWithEvidence(bench, &single_repeat_evidence));

    var wrong_backend = bench;
    wrong_backend.backend = .metal;
    try std.testing.expect(!benchmarkHasPromotionEvidence(wrong_backend));
    try std.testing.expectEqualStrings("non_cuda_benchmark", benchmarkPromotionBlocker(wrong_backend));

    var missing_baseline = bench;
    missing_baseline.handwritten_baseline = "";
    try std.testing.expect(!benchmarkHasPromotionEvidence(missing_baseline));
    try std.testing.expectEqualStrings("missing_handwritten_baseline", benchmarkPromotionBlocker(missing_baseline));

    var missing_source_fingerprint = bench;
    missing_source_fingerprint.generated_source_fingerprint = 0;
    try std.testing.expect(!benchmarkHasPromotionEvidence(missing_source_fingerprint));
    try std.testing.expectEqualStrings("missing_source_fingerprint", benchmarkPromotionBlocker(missing_source_fingerprint));

    var empty_ptx_arg = bench;
    empty_ptx_arg.generated_ptx_arg = "";
    try std.testing.expect(!benchmarkHasPromotionEvidence(empty_ptx_arg));
    try std.testing.expectEqualStrings("wrong_generated_ptx_arg", benchmarkPromotionBlocker(empty_ptx_arg));

    var wrong_ptx_arg = bench;
    wrong_ptx_arg.generated_ptx_arg = "--ptx";
    wrong_ptx_arg.benchmark_command = "zig-out/bin/antfly-inference bench-cuda --warmup-iters 5 --measure-iters 50 --quant-compiler-lazy-target --ptx /tmp/antfly_q4_k_small_batch_bias_gelu_f32_v1.ptx";
    try std.testing.expect(!benchmarkHasPromotionEvidence(wrong_ptx_arg));
    try std.testing.expectEqualStrings("wrong_generated_ptx_arg", benchmarkPromotionBlocker(wrong_ptx_arg));

    var wrong_arch = bench;
    wrong_arch.generated_ptx_command = "nvcc -ptx -arch=compute_90 src/ops/cuda/artifacts/inference_cuda_kernels.cu -o /tmp/antfly_q4_k_small_batch_bias_gelu_f32_v1.ptx";
    try std.testing.expect(!benchmarkHasPromotionEvidence(wrong_arch));
    try std.testing.expectEqualStrings("wrong_ptx_arch", benchmarkPromotionBlocker(wrong_arch));

    var wrong_ptx_extension = bench;
    wrong_ptx_extension.generated_ptx_path = "/tmp/antfly_q4_k_small_batch_bias_gelu_f32_v1.ptx.txt";
    try std.testing.expect(!benchmarkHasPromotionEvidence(wrong_ptx_extension));
    try std.testing.expectEqualStrings("missing_generated_ptx_path", benchmarkPromotionBlocker(wrong_ptx_extension));

    var missing_ptx_arg = bench;
    missing_ptx_arg.benchmark_command = "zig-out/bin/antfly-inference bench-cuda --warmup-iters 5 --measure-iters 50 --quant-compiler-lazy-target";
    try std.testing.expect(!benchmarkHasPromotionEvidence(missing_ptx_arg));
    try std.testing.expectEqualStrings("benchmark_missing_generated_ptx_arg", benchmarkPromotionBlocker(missing_ptx_arg));

    var missing_repeat = bench;
    missing_repeat.benchmark_command = "zig-out/bin/antfly-inference bench-cuda --warmup-iters 5 --measure-iters 50 --quant-compiler-lazy-target --quant-compiler-generated-ptx /tmp/antfly_q4_k_small_batch_bias_gelu_f32_v1.ptx --quant-compiler-evidence-out src/ops/cuda/generated/evidence/q4_k_small_batch_bias_gelu_benchmark.json";
    try std.testing.expect(!benchmarkHasPromotionEvidence(missing_repeat));
    try std.testing.expectEqualStrings("missing_benchmark_repeat_runs", benchmarkPromotionBlocker(missing_repeat));

    var wrong_ptx_path_value = bench;
    wrong_ptx_path_value.benchmark_command = "zig-out/bin/antfly-inference bench-cuda --warmup-iters 5 --measure-iters 50 --quant-compiler-lazy-target --quant-compiler-generated-ptx /tmp/wrong.ptx /tmp/antfly_q4_k_small_batch_bias_gelu_f32_v1.ptx --quant-compiler-repeat-runs 3 --quant-compiler-evidence-out src/ops/cuda/generated/evidence/q4_k_small_batch_bias_gelu_benchmark.json";
    try std.testing.expect(!benchmarkHasPromotionEvidence(wrong_ptx_path_value));
    try std.testing.expectEqualStrings("benchmark_missing_generated_ptx_path", benchmarkPromotionBlocker(wrong_ptx_path_value));

    var missing_evidence_out = bench;
    missing_evidence_out.benchmark_command = "zig-out/bin/antfly-inference bench-cuda --warmup-iters 5 --measure-iters 50 --quant-compiler-lazy-target --quant-compiler-generated-ptx /tmp/antfly_q4_k_small_batch_bias_gelu_f32_v1.ptx --quant-compiler-repeat-runs 3";
    try std.testing.expect(!benchmarkHasPromotionEvidence(missing_evidence_out));
    try std.testing.expectEqualStrings("benchmark_missing_evidence_out_arg", benchmarkPromotionBlocker(missing_evidence_out));

    var wrong_evidence_path = bench;
    wrong_evidence_path.benchmark_command = "zig-out/bin/antfly-inference bench-cuda --warmup-iters 5 --measure-iters 50 --quant-compiler-lazy-target --quant-compiler-generated-ptx /tmp/antfly_q4_k_small_batch_bias_gelu_f32_v1.ptx --quant-compiler-repeat-runs 3 --quant-compiler-evidence-out /tmp/wrong-evidence.json";
    try std.testing.expect(!benchmarkHasPromotionEvidence(wrong_evidence_path));
    try std.testing.expectEqualStrings("benchmark_missing_evidence_out_path", benchmarkPromotionBlocker(wrong_evidence_path));

    var missing_iters = bench;
    missing_iters.benchmark_command = "zig-out/bin/antfly-inference bench-cuda --quant-compiler-lazy-target --quant-compiler-generated-ptx /tmp/antfly_q4_k_small_batch_bias_gelu_f32_v1.ptx";
    try std.testing.expect(!benchmarkHasPromotionEvidence(missing_iters));
    try std.testing.expectEqualStrings("missing_warmup_iters", benchmarkPromotionBlocker(missing_iters));

    var loose_correctness = bench;
    loose_correctness.correctness_tolerance_abs = 0.02;
    try std.testing.expect(!benchmarkHasPromotionEvidence(loose_correctness));
    try std.testing.expectEqualStrings("correctness_tolerance_missing_or_loose", benchmarkPromotionBlocker(loose_correctness));

    var nan_correctness = bench;
    nan_correctness.correctness_tolerance_abs = std.math.nan(f32);
    try std.testing.expect(!benchmarkHasPromotionEvidence(nan_correctness));
    try std.testing.expectEqualStrings("correctness_tolerance_missing_or_loose", benchmarkPromotionBlocker(nan_correctness));

    var no_speedup = bench;
    no_speedup.minimum_speedup = 0.99;
    try std.testing.expect(!benchmarkHasPromotionEvidence(no_speedup));
    try std.testing.expectEqualStrings("speedup_gate_missing", benchmarkPromotionBlocker(no_speedup));

    var nan_min_speedup = bench;
    nan_min_speedup.minimum_speedup = std.math.nan(f64);
    try std.testing.expect(!benchmarkHasPromotionEvidence(nan_min_speedup));
    try std.testing.expectEqualStrings("speedup_gate_missing", benchmarkPromotionBlocker(nan_min_speedup));

    const nan_speedup_evidence = [_]BenchmarkEvidence{.{
        .kernel_id = bench.generated_kernel_id,
        .generated_source_path = bench.generated_source_path,
        .generated_source_fingerprint = bench.generated_source_fingerprint,
        .generated_ptx_path = bench.generated_ptx_path,
        .generated_ptx_command = bench.generated_ptx_command,
        .benchmark_command = bench.benchmark_command,
        .correctness_evidence_path = bench.correctness_evidence_path,
        .benchmark_evidence_path = bench.benchmark_evidence_path,
        .benchmark_mode = bench.benchmark_mode,
        .repeat_runs = metal_promotion_repeat_runs,
        .correctness_passed = true,
        .benchmark_passed = true,
        .measured_speedup = std.math.nan(f64),
    }};
    try std.testing.expectEqualStrings("speedup_gate_missing", benchmarkPromotionBlockerWithEvidence(bench, &nan_speedup_evidence));
}

test "quant kernel compiler production artifacts require promotion evidence" {
    for (first_generated_artifacts) |artifact| {
        try std.testing.expectEqual(artifact.production_enabled, artifactHasPromotionEvidence(artifact));
    }

    var cuda_artifact = first_generated_artifacts[0];
    try std.testing.expect(benchmarkForArtifact(cuda_artifact) != null);
    cuda_artifact.production_enabled = true;
    cuda_artifact.source_path = "src/ops/cuda/artifacts/inference_cuda_kernels.cu";
    try std.testing.expect(benchmarkForArtifact(cuda_artifact) == null);
    try std.testing.expect(!artifactHasPromotionEvidence(cuda_artifact));

    var unbenchmarked = cuda_artifact;
    unbenchmarked.kernel_id = "unbenchmarked_generated_kernel";
    try std.testing.expect(benchmarkForArtifact(unbenchmarked) == null);
    try std.testing.expect(!artifactHasPromotionEvidence(unbenchmarked));

    var unstable_general_metal = generatedArtifactForKernel(.metal, first_general_metal_q4_0_kernel_id).?;
    unstable_general_metal.production_enabled = true;
    try std.testing.expect(!artifactHasPromotionEvidence(unstable_general_metal));
    try std.testing.expectEqualStrings("TERMITE_METAL_ENABLE_ANTFLY_Q4_0_SMALL_BATCH", std.mem.span(artifactRuntimeGateEnv(unstable_general_metal).?));

    var unsupported_baseline_metal = generatedArtifactForKernel(.metal, first_general_metal_q8_relu_kernel_id).?;
    unsupported_baseline_metal.production_enabled = true;
    try std.testing.expect(!artifactHasPromotionEvidence(unsupported_baseline_metal));
    try std.testing.expectEqualStrings("TERMITE_METAL_ENABLE_ANTFLY_Q8_0_SMALL_BATCH_RELU", std.mem.span(artifactRuntimeGateEnv(unsupported_baseline_metal).?));

    const metal_artifact = generatedArtifactForKernel(.metal, first_general_metal_q6_bias_kernel_id).?;
    try std.testing.expect(benchmarkForArtifact(metal_artifact) == null);
    try std.testing.expect(artifactHasPromotionEvidence(metal_artifact));
    try std.testing.expect(artifactRuntimeGateEnv(metal_artifact) == null);
    try std.testing.expectEqualStrings("missing_metal_runtime_evidence", metalArtifactPromotionBlockerWithEvidence(metal_artifact, &.{}));

    const metal_evidence = [_]MetalRuntimeEvidence{.{
        .kernel_id = metal_artifact.kernel_id,
        .source_path = first_general_metal_q6_bias_source_path,
        .artifact_source_path = first_general_metal_q6_bias_artifact_source_path,
        .source_fingerprint = artifactSourceFingerprint(metal_artifact),
        .check_command = first_general_metal_q6_bias_check_command,
        .runtime_evidence_command = metal_artifact.promotion_evidence_command,
        .promotion_check_command = metal_artifact.promotion_check_command,
        .repeat_runs = metal_promotion_repeat_runs,
        .correctness_passed = true,
        .generated_route_checked = true,
        .provider_route_checked = true,
        .benchmark_passed = true,
        .measured_speedup = metal_promotion_min_speedup + 0.01,
        .minimum_repeat_speedup = metal_promotion_min_speedup + 0.01,
        .production_enabled = true,
        .promotion_ready = true,
    }};
    try std.testing.expectEqualStrings("none", metalArtifactPromotionBlockerWithEvidence(metal_artifact, &metal_evidence));

    var missing_route_evidence = metal_evidence;
    missing_route_evidence[0].generated_route_checked = false;
    try std.testing.expectEqualStrings("missing_metal_generated_route_evidence", metalArtifactPromotionBlockerWithEvidence(metal_artifact, &missing_route_evidence));

    var stale_source_evidence = metal_evidence;
    stale_source_evidence[0].source_fingerprint = 0;
    try std.testing.expectEqualStrings("missing_metal_runtime_evidence", metalArtifactPromotionBlockerWithEvidence(metal_artifact, &stale_source_evidence));

    var wrong_source_path_evidence = metal_evidence;
    wrong_source_path_evidence[0].source_path = first_lazy_metal_source_path;
    try std.testing.expectEqualStrings("missing_metal_runtime_evidence", metalArtifactPromotionBlockerWithEvidence(metal_artifact, &wrong_source_path_evidence));

    var wrong_check_command_evidence = metal_evidence;
    wrong_check_command_evidence[0].check_command = first_lazy_metal_check_command;
    try std.testing.expectEqualStrings("missing_metal_runtime_evidence", metalArtifactPromotionBlockerWithEvidence(metal_artifact, &wrong_check_command_evidence));

    const wrong_promotion_path_command = try std.fmt.allocPrint(
        std.testing.allocator,
        "zig build quant-kernel-metal-runtime-check -Dmetal=true -Dcuda=false -- --evidence-out /private/tmp/antfly-quant-metal-{s}-wrong-promotion-evidence.json" ++ metal_promotion_args ++ "{s}",
        .{ metal_artifact.kernel_id, metal_artifact.kernel_id },
    );
    defer std.testing.allocator.free(wrong_promotion_path_command);
    var mismatched_promotion_path = metal_artifact;
    mismatched_promotion_path.promotion_evidence_command = wrong_promotion_path_command;
    try std.testing.expectEqualStrings("metal_promotion_evidence_path_mismatch", metalArtifactPromotionBlockerWithEvidence(mismatched_promotion_path, &metal_evidence));

    const invalid_promotion_path_command = try std.fmt.allocPrint(
        std.testing.allocator,
        "zig build quant-kernel-metal-runtime-check -Dmetal=true -Dcuda=false -- --evidence-out /tmp/wrong.json --repeat-runs " ++ metal_promotion_repeat_runs_text ++ " --promotion-ready-kernel {s}",
        .{metal_artifact.kernel_id},
    );
    defer std.testing.allocator.free(invalid_promotion_path_command);
    var mismatched_promotion_command = metal_artifact;
    mismatched_promotion_command.promotion_evidence_command = invalid_promotion_path_command;
    try std.testing.expectEqualStrings("metal_promotion_evidence_command_mismatch", metalArtifactPromotionBlockerWithEvidence(mismatched_promotion_command, &metal_evidence));

    var missing_promotion_command = metal_artifact;
    missing_promotion_command.promotion_evidence_command = "";
    try std.testing.expectEqualStrings("metal_promotion_evidence_command_mismatch", metalArtifactPromotionBlockerWithEvidence(missing_promotion_command, &metal_evidence));

    const wrong_promotion_kernel_command = first_metal_promotion_evidence_command ++ first_general_metal_q6_bias_promotion_evidence_path ++ metal_promotion_args ++ "wrong_kernel";
    var wrong_promotion_kernel = metal_artifact;
    wrong_promotion_kernel.promotion_evidence_command = wrong_promotion_kernel_command;
    try std.testing.expectEqualStrings("wrong_metal_promotion_ready_kernel", metalArtifactPromotionBlockerWithEvidence(wrong_promotion_kernel, &metal_evidence));

    var dev_command_evidence = metal_evidence;
    dev_command_evidence[0].runtime_evidence_command = metal_artifact.runtime_evidence_command;
    try std.testing.expectEqualStrings("missing_metal_runtime_evidence", metalArtifactPromotionBlockerWithEvidence(metal_artifact, &dev_command_evidence));

    const loose_repeat_command = try std.fmt.allocPrint(
        std.testing.allocator,
        "{s}0 --promotion-ready-kernel {s}",
        .{ metal_artifact.runtime_evidence_command, metal_artifact.kernel_id },
    );
    defer std.testing.allocator.free(loose_repeat_command);
    var loose_repeat_evidence = metal_evidence;
    loose_repeat_evidence[0].runtime_evidence_command = loose_repeat_command;
    try std.testing.expectEqualStrings("missing_metal_runtime_evidence", metalArtifactPromotionBlockerWithEvidence(metal_artifact, &loose_repeat_evidence));

    var disabled_metal_artifact = metal_artifact;
    disabled_metal_artifact.production_enabled = false;
    try std.testing.expectEqualStrings("awaiting_metal_promotion_evidence", metalArtifactPromotionBlockerWithEvidence(disabled_metal_artifact, &metal_evidence));

    var single_repeat_evidence = metal_evidence;
    single_repeat_evidence[0].repeat_runs = 1;
    try std.testing.expectEqualStrings("insufficient_metal_repeats", metalArtifactPromotionBlockerWithEvidence(metal_artifact, &single_repeat_evidence));

    var weak_average_evidence = metal_evidence;
    weak_average_evidence[0].measured_speedup = metal_promotion_min_speedup - 0.01;
    weak_average_evidence[0].minimum_repeat_speedup = metal_promotion_min_speedup - 0.02;
    try std.testing.expectEqualStrings("speedup_gate_missing", metalArtifactPromotionBlockerWithEvidence(metal_artifact, &weak_average_evidence));

    var unstable_repeat_evidence = metal_evidence;
    unstable_repeat_evidence[0].measured_speedup = metal_promotion_min_speedup + 0.2;
    unstable_repeat_evidence[0].minimum_repeat_speedup = metal_promotion_min_speedup - 0.01;
    try std.testing.expectEqualStrings("unstable_benchmark_timing", metalArtifactPromotionBlockerWithEvidence(metal_artifact, &unstable_repeat_evidence));

    var failed_benchmark_flag_evidence = metal_evidence;
    failed_benchmark_flag_evidence[0].benchmark_passed = false;
    try std.testing.expectEqualStrings("benchmark_evidence_failed", metalArtifactPromotionBlockerWithEvidence(metal_artifact, &failed_benchmark_flag_evidence));

    var not_ready_evidence = metal_evidence;
    not_ready_evidence[0].promotion_ready = false;
    try std.testing.expectEqualStrings("metal_promotion_not_ready", metalArtifactPromotionBlockerWithEvidence(metal_artifact, &not_ready_evidence));

    var wrong_backend = cuda_artifact;
    wrong_backend.backend = .metal;
    try std.testing.expect(benchmarkForArtifact(wrong_backend) == null);
    try std.testing.expect(!artifactHasPromotionEvidence(wrong_backend));
}

test "quant kernel compiler generated artifacts have unique ids and paths" {
    for (first_generated_artifacts, 0..) |artifact, i| {
        try std.testing.expect(artifact.kernel_id.len != 0);
        try std.testing.expect(artifact.source_path.len != 0);
        const lookup = generatedArtifactForCandidate(artifact.backend, artifact.format, artifact.row_bucket, artifact.epilogue) orelse return error.MissingGeneratedArtifactLookup;
        try std.testing.expectEqualStrings(artifact.kernel_id, lookup.kernel_id);
        try std.testing.expectEqualStrings(artifact.source_path, lookup.source_path);
        const kernel_lookup = generatedArtifactForKernel(artifact.backend, artifact.kernel_id) orelse return error.MissingGeneratedArtifactLookup;
        try std.testing.expectEqualStrings(artifact.source_path, kernel_lookup.source_path);
        if (artifact.generated_source_path.len == 0) {
            try std.testing.expectEqualStrings(artifact.source_path, generatedSourcePathForArtifact(artifact));
            try std.testing.expectEqualStrings(artifact.check_command, generatedCheckCommandForArtifact(artifact));
        } else {
            try std.testing.expectEqualStrings(artifact.generated_source_path, generatedSourcePathForArtifact(artifact));
            try std.testing.expectEqualStrings(artifact.generated_check_command, generatedCheckCommandForArtifact(artifact));
        }
        for (first_generated_artifacts[i + 1 ..]) |other| {
            try std.testing.expect(!std.mem.eql(u8, artifact.source_path, other.source_path));
            try std.testing.expect(!std.mem.eql(u8, artifact.kernel_id, other.kernel_id));
            try std.testing.expect(!(artifact.backend == other.backend and
                artifact.format == other.format and
                artifact.row_bucket == other.row_bucket and
                artifact.epilogue == other.epilogue));
        }
    }
}

test "quant kernel compiler promoted artifacts keep generated evidence metadata" {
    var promoted = first_generated_artifacts[1];
    promoted.source_path = "src/ops/metal/artifacts/quant_kernel_q4_k_small_batch_bias_gelu.metal";
    promoted.check_command = "xcrun --toolchain Metal metal -c src/ops/metal/artifacts/quant_kernel_q4_k_small_batch_bias_gelu.metal -o src/ops/metal/artifacts/quant_kernel_q4_k_small_batch_bias_gelu.air";
    promoted.generated_source_path = first_lazy_metal_source_path;
    promoted.generated_check_command = first_lazy_metal_check_command;

    try std.testing.expectEqualStrings(first_lazy_metal_source_path, generatedSourcePathForArtifact(promoted));
    try std.testing.expectEqualStrings(first_lazy_metal_check_command, generatedCheckCommandForArtifact(promoted));
}

test "quant kernel compiler generated artifact manifest maps to route candidates" {
    for (first_generated_artifacts) |artifact| {
        try std.testing.expect(artifactSourceFingerprint(artifact) != 0);
        try std.testing.expect(artifact.check_command.len != 0);
        try std.testing.expect(std.mem.containsAtLeast(u8, artifact.check_command, 1, artifact.source_path));

        const route = loweringFor(artifact.backend, artifact.format, artifact.row_bucket, artifact.epilogue);
        if (artifact.production_enabled) {
            try std.testing.expect(!isDevGeneratedSourcePath(artifact.source_path));
            if (artifact.backend == .metal) {
                try std.testing.expect(artifact.generated_source_path.len != 0);
                try std.testing.expect(isDevGeneratedSourcePath(artifact.generated_source_path));
                try std.testing.expect(std.mem.containsAtLeast(u8, artifact.generated_check_command, 1, artifact.generated_source_path));
            }
            try std.testing.expectEqual(LoweringRoute.generated_production, route.production_route);
            try std.testing.expectEqual(LoweringRoute.unsupported, route.candidate_route);
            try std.testing.expectEqual(FallbackReason.none, route.fallback_reason);
            try std.testing.expectEqualStrings(artifact.kernel_id, route.production_kernel_id);
            try std.testing.expectEqualStrings("", route.kernel_id);
            try std.testing.expectEqualStrings("", route.candidate_source_path);
            const promoted = promotedArtifactFor(route) orelse return error.MissingPromotedArtifact;
            try std.testing.expectEqualStrings(artifact.source_path, promoted.source_path);
        } else {
            if (artifact.generated_source_path.len == 0) {
                try std.testing.expect(isDevGeneratedSourcePath(artifact.source_path));
            } else {
                try std.testing.expect(std.mem.containsAtLeast(u8, artifact.check_command, 1, artifact.source_path));
                try std.testing.expect(isDevGeneratedSourcePath(artifact.generated_source_path));
                try std.testing.expect(std.mem.containsAtLeast(u8, artifact.generated_check_command, 1, artifact.generated_source_path));
            }
            try std.testing.expectEqual(LoweringRoute.handwritten_production, route.production_route);
            try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, route.candidate_route);
            const expected_fallback: FallbackReason = if (artifact.backend == .metal and checkedInMetalEvidenceForKernel(artifact.kernel_id) != null)
                .generated_runtime_not_wired
            else
                .generated_artifact_missing;
            try std.testing.expectEqual(expected_fallback, route.fallback_reason);
            try std.testing.expectEqualStrings(artifact.kernel_id, route.kernel_id);
            try std.testing.expectEqualStrings(artifact.source_path, route.candidate_source_path);
            try std.testing.expectEqual(artifactSourceFingerprint(artifact), candidateSourceFingerprint(route));
            const candidate_schedule = candidateScheduleFor(route);
            const expected_tile_cols: usize = if (artifact.backend == .metal)
                metalGeneratedColsPerThreadgroup(artifact.format, artifact.row_bucket, artifact.epilogue)
            else
                1;
            const expected_threads_per_block: usize = if (artifact.backend == .metal)
                metalGeneratedThreadsPerThreadgroup(artifact.format, artifact.row_bucket, artifact.epilogue)
            else
                128;
            try std.testing.expectEqual(@as(usize, 1), candidate_schedule.tile_rows);
            try std.testing.expectEqual(expected_tile_cols, candidate_schedule.tile_cols);
            try std.testing.expectEqual(expected_threads_per_block, candidate_schedule.threads_per_block);
        }

        const contents = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, artifact.source_path, std.testing.allocator, .limited(128 * 1024));
        defer std.testing.allocator.free(contents);
        const compiled = compileQuantKernelSource(.{
            .backend = artifact.backend,
            .format = artifact.format,
            .row_bucket = artifact.row_bucket,
            .epilogue = artifact.epilogue,
        }) orelse return error.MissingGeneratedSource;
        const emitted = try emitCompiledSource(std.testing.allocator, compiled);
        defer emitted.deinit(std.testing.allocator);
        try std.testing.expectEqualStrings(emitted.data, contents);
        try std.testing.expectEqualStrings(artifact.kernel_id, compiled.artifact.kernel_id);
        try std.testing.expect(try compiledSourceHeaderMatchesSource(std.testing.allocator, compiled, emitted.data));
        switch (artifact.backend) {
            .cuda => {
                try std.testing.expect(std.mem.containsAtLeast(u8, artifact.check_command, 1, "nvcc -ptx"));
                try std.testing.expect(std.mem.containsAtLeast(u8, artifact.check_command, 1, ".ptx"));
                try std.testing.expectEqualStrings(first_lazy_benchmark.benchmark_command, artifact.runtime_evidence_command);
                try std.testing.expectEqualStrings(first_lazy_benchmark_check_command, artifact.promotion_check_command);
            },
            .metal => {
                try std.testing.expect(std.mem.containsAtLeast(u8, artifact.check_command, 1, "xcrun --toolchain Metal metal -c"));
                try std.testing.expect(std.mem.containsAtLeast(u8, artifact.check_command, 1, ".air"));
                try std.testing.expect(std.mem.containsAtLeast(u8, artifact.runtime_evidence_command, 1, "--repeat-runs " ++ metal_promotion_repeat_runs_text));
                try std.testing.expectEqualStrings(first_metal_runtime_evidence_command, artifact.runtime_evidence_command);
                try std.testing.expect(std.mem.containsAtLeast(u8, artifact.promotion_check_command, 1, first_metal_promotion_check_command));
                try std.testing.expect(commandHasArgValue(artifact.promotion_check_command, "--require-kernel", artifact.kernel_id));
                try std.testing.expect(commandHasArgValue(artifact.promotion_evidence_command, "--measure-iters", metal_promotion_measure_iters_text));
            },
        }
    }
}

test "quant kernel compiler generated production routes require promoted artifacts" {
    for (first_registry.entries) |entry| {
        if (entry.production_route != .generated_production) continue;
        try std.testing.expectEqual(LoweringRoute.unsupported, entry.candidate_route);
        try std.testing.expectEqual(FallbackReason.none, entry.fallback_reason);
        const artifact = promotedArtifactFor(entry) orelse return error.MissingPromotedArtifact;
        try std.testing.expect(!isDevGeneratedSourcePath(artifact.source_path));
    }

    var unpromoted = loweringFor(.cuda, .q4_k, .rows_2_8, .bias_gelu);
    unpromoted.production_route = .generated_production;
    unpromoted.production_kernel_id = first_lazy_benchmark.generated_kernel_id;
    unpromoted.candidate_route = .unsupported;
    unpromoted.fallback_reason = .none;
    try std.testing.expect(promotedArtifactFor(unpromoted) == null);

    var dev_route = loweringFor(.metal, .q4_k, .rows_2_8, .bias_gelu);
    dev_route.production_route = .handwritten_production;
    dev_route.candidate_route = .generated_dev_candidate;
    dev_route.production_kernel_id = "metal_handwritten_quant_matmul";
    dev_route.fallback_reason = .generated_artifact_missing;
    dev_route.kernel_id = first_lazy_metal_kernel_id;
    dev_route.candidate_source_path = first_lazy_metal_source_path;
    var promoted_artifact = first_generated_artifacts[1];
    promoted_artifact.source_path = "src/ops/metal/artifacts/quant_kernel_q4_k_small_batch_bias_gelu.metal";
    const promoted = promotedLoweringForArtifact(dev_route, promoted_artifact).?;
    try std.testing.expectEqual(LoweringRoute.generated_production, promoted.production_route);
    try std.testing.expectEqual(LoweringRoute.unsupported, promoted.candidate_route);
    try std.testing.expectEqual(FallbackReason.none, promoted.fallback_reason);
    try std.testing.expectEqualStrings(promoted_artifact.kernel_id, promoted.production_kernel_id);
    try std.testing.expectEqualStrings("", promoted.kernel_id);
    try std.testing.expectEqualStrings("", promoted.candidate_source_path);

    var dev_artifact = promoted_artifact;
    dev_artifact.source_path = first_lazy_metal_source_path;
    try std.testing.expect(promotedLoweringForArtifact(dev_route, dev_artifact) == null);
}

test "quant kernel compiler registry helper is the dispatch-facing route source" {
    const planned = registryLoweringFor(.cuda, .q4_k, .rows_2_8, .bias_gelu, .small_batch);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, planned.candidate_route);
    try std.testing.expectEqualStrings(first_lazy_benchmark.generated_kernel_id, planned.kernel_id);
    try std.testing.expectEqualStrings(first_lazy_benchmark.generated_source_path, planned.candidate_source_path);

    const metal_q4_0 = registryLoweringFor(.metal, .q4_0, .rows_2_8, .none, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q4_0.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q4_0.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q4_0.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q4_0_kernel_id, metal_q4_0.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q4_0_source_path, metal_q4_0.candidate_source_path);

    const metal_q4_1 = registryLoweringFor(.metal, .q4_1, .rows_2_8, .none, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q4_1.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q4_1.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q4_1.fallback_reason);
    try std.testing.expectEqualStrings("metal_handwritten_quant_matmul", metal_q4_1.production_kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q4_1_kernel_id, metal_q4_1.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q4_1_artifact_source_path, metal_q4_1.candidate_source_path);

    const metal_q5_0 = registryLoweringFor(.metal, .q5_0, .rows_2_8, .none, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q5_0.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q5_0.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q5_0.fallback_reason);
    try std.testing.expectEqualStrings("metal_handwritten_quant_matmul", metal_q5_0.production_kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q5_0_kernel_id, metal_q5_0.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q5_0_artifact_source_path, metal_q5_0.candidate_source_path);

    const metal_q5_1 = registryLoweringFor(.metal, .q5_1, .rows_2_8, .none, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q5_1.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q5_1.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q5_1.fallback_reason);
    try std.testing.expectEqualStrings("metal_handwritten_quant_matmul", metal_q5_1.production_kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q5_1_kernel_id, metal_q5_1.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q5_1_artifact_source_path, metal_q5_1.candidate_source_path);

    const metal_q4 = registryLoweringFor(.metal, .q4_k, .rows_2_8, .none, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q4.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q4.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q4.fallback_reason);
    try std.testing.expectEqualStrings("metal_handwritten_quant_matmul", metal_q4.production_kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q4_kernel_id, metal_q4.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q4_artifact_source_path, metal_q4.candidate_source_path);

    const metal_q4_bias = registryLoweringFor(.metal, .q4_k, .rows_2_8, .bias, .small_batch);
    try std.testing.expectEqual(LoweringRoute.generated_production, metal_q4_bias.production_route);
    try std.testing.expectEqual(LoweringRoute.unsupported, metal_q4_bias.candidate_route);
    try std.testing.expectEqual(FallbackReason.none, metal_q4_bias.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q4_bias_kernel_id, metal_q4_bias.production_kernel_id);
    try std.testing.expectEqualStrings("", metal_q4_bias.kernel_id);
    try std.testing.expectEqualStrings("", metal_q4_bias.candidate_source_path);

    const metal_q8 = registryLoweringFor(.metal, .q8_0, .rows_2_8, .none, .small_batch);
    try std.testing.expectEqual(LoweringRoute.generated_production, metal_q8.production_route);
    try std.testing.expectEqual(LoweringRoute.unsupported, metal_q8.candidate_route);
    try std.testing.expectEqual(FallbackReason.none, metal_q8.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q8_kernel_id, metal_q8.production_kernel_id);
    try std.testing.expectEqualStrings("", metal_q8.kernel_id);
    try std.testing.expectEqualStrings("", metal_q8.candidate_source_path);

    const metal_q8_bias = registryLoweringFor(.metal, .q8_0, .rows_2_8, .bias, .small_batch);
    try std.testing.expectEqual(LoweringRoute.generated_production, metal_q8_bias.production_route);
    try std.testing.expectEqual(LoweringRoute.unsupported, metal_q8_bias.candidate_route);
    try std.testing.expectEqual(FallbackReason.none, metal_q8_bias.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q8_bias_kernel_id, metal_q8_bias.production_kernel_id);
    try std.testing.expectEqualStrings("", metal_q8_bias.kernel_id);
    try std.testing.expectEqualStrings("", metal_q8_bias.candidate_source_path);

    const metal_q8_bias_gelu = registryLoweringFor(.metal, .q8_0, .rows_2_8, .bias_gelu, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q8_bias_gelu.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q8_bias_gelu.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q8_bias_gelu.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q8_bias_gelu_kernel_id, metal_q8_bias_gelu.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q8_bias_gelu_artifact_source_path, metal_q8_bias_gelu.candidate_source_path);

    const metal_q8_relu = registryLoweringFor(.metal, .q8_0, .rows_2_8, .relu, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q8_relu.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q8_relu.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q8_relu.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q8_relu_kernel_id, metal_q8_relu.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q8_relu_source_path, metal_q8_relu.candidate_source_path);

    const metal_q2 = registryLoweringFor(.metal, .q2_k, .rows_2_8, .none, .small_batch);
    try std.testing.expectEqual(LoweringRoute.generated_production, metal_q2.production_route);
    try std.testing.expectEqual(LoweringRoute.unsupported, metal_q2.candidate_route);
    try std.testing.expectEqual(FallbackReason.none, metal_q2.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q2_kernel_id, metal_q2.production_kernel_id);
    try std.testing.expectEqualStrings("", metal_q2.kernel_id);
    try std.testing.expectEqualStrings("", metal_q2.candidate_source_path);

    const metal_q2_bias = registryLoweringFor(.metal, .q2_k, .rows_2_8, .bias, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q2_bias.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q2_bias.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q2_bias.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q2_bias_kernel_id, metal_q2_bias.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q2_bias_source_path, metal_q2_bias.candidate_source_path);

    const metal_q2_bias_gelu = registryLoweringFor(.metal, .q2_k, .rows_2_8, .bias_gelu, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q2_bias_gelu.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q2_bias_gelu.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q2_bias_gelu.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q2_bias_gelu_kernel_id, metal_q2_bias_gelu.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q2_bias_gelu_source_path, metal_q2_bias_gelu.candidate_source_path);

    const metal_q3 = registryLoweringFor(.metal, .q3_k, .rows_2_8, .none, .small_batch);
    try std.testing.expectEqual(LoweringRoute.generated_production, metal_q3.production_route);
    try std.testing.expectEqual(LoweringRoute.unsupported, metal_q3.candidate_route);
    try std.testing.expectEqual(FallbackReason.none, metal_q3.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q3_kernel_id, metal_q3.production_kernel_id);
    try std.testing.expectEqualStrings("", metal_q3.kernel_id);
    try std.testing.expectEqualStrings("", metal_q3.candidate_source_path);

    const metal_q3_bias = registryLoweringFor(.metal, .q3_k, .rows_2_8, .bias, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q3_bias.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q3_bias.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q3_bias.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q3_bias_kernel_id, metal_q3_bias.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q3_bias_source_path, metal_q3_bias.candidate_source_path);

    const metal_q3_bias_gelu = registryLoweringFor(.metal, .q3_k, .rows_2_8, .bias_gelu, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q3_bias_gelu.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q3_bias_gelu.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q3_bias_gelu.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q3_bias_gelu_kernel_id, metal_q3_bias_gelu.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q3_bias_gelu_source_path, metal_q3_bias_gelu.candidate_source_path);

    const metal_q8_1 = registryLoweringFor(.metal, .q8_1, .rows_2_8, .none, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q8_1.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q8_1.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q8_1.fallback_reason);
    try std.testing.expectEqualStrings("metal_handwritten_quant_matmul", metal_q8_1.production_kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q8_1_kernel_id, metal_q8_1.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q8_1_artifact_source_path, metal_q8_1.candidate_source_path);

    const metal_q8_k = registryLoweringFor(.metal, .q8_k, .rows_2_8, .none, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q8_k.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q8_k.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q8_k.fallback_reason);
    try std.testing.expectEqualStrings("metal_handwritten_quant_matmul", metal_q8_k.production_kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q8_k_kernel_id, metal_q8_k.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q8_k_artifact_source_path, metal_q8_k.candidate_source_path);

    const metal_q5 = registryLoweringFor(.metal, .q5_k, .rows_2_8, .none, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q5.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q5.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q5.fallback_reason);
    try std.testing.expectEqualStrings("metal_handwritten_quant_matmul", metal_q5.production_kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q5_kernel_id, metal_q5.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q5_artifact_source_path, metal_q5.candidate_source_path);

    const metal_q5_bias = registryLoweringFor(.metal, .q5_k, .rows_2_8, .bias, .small_batch);
    try std.testing.expectEqual(LoweringRoute.generated_production, metal_q5_bias.production_route);
    try std.testing.expectEqual(LoweringRoute.unsupported, metal_q5_bias.candidate_route);
    try std.testing.expectEqual(FallbackReason.none, metal_q5_bias.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q5_bias_kernel_id, metal_q5_bias.production_kernel_id);
    try std.testing.expectEqualStrings("", metal_q5_bias.kernel_id);
    try std.testing.expectEqualStrings("", metal_q5_bias.candidate_source_path);
    try std.testing.expect(std.mem.containsAtLeast(u8, first_general_metal_q5_bias_source, 1, "threadgroup float partial[32];"));
    try std.testing.expect(std.mem.containsAtLeast(u8, first_general_metal_q5_bias_source, 1, "if (simdgroup_id == 0u && lane_id >= 4u) partial[lane_id] = 0.0f;"));
    try std.testing.expect(std.mem.containsAtLeast(u8, first_general_metal_q5_bias_source, 1, "simdgroup_index_in_threadgroup"));

    const metal_q5_bias_gelu = registryLoweringFor(.metal, .q5_k, .rows_2_8, .bias_gelu, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q5_bias_gelu.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q5_bias_gelu.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q5_bias_gelu.fallback_reason);
    try std.testing.expectEqualStrings("metal_handwritten_quant_matmul", metal_q5_bias_gelu.production_kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q5_bias_gelu_kernel_id, metal_q5_bias_gelu.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q5_bias_gelu_artifact_source_path, metal_q5_bias_gelu.candidate_source_path);
    try std.testing.expect(std.mem.containsAtLeast(u8, first_general_metal_q5_bias_gelu_source, 1, "threadgroup float partial[32];"));
    try std.testing.expect(std.mem.containsAtLeast(u8, first_general_metal_q5_bias_gelu_source, 1, "if (simdgroup_id == 0u && lane_id >= 4u) partial[lane_id] = 0.0f;"));
    try std.testing.expect(std.mem.containsAtLeast(u8, first_general_metal_q5_bias_gelu_source, 1, "simdgroup_index_in_threadgroup"));

    const metal_q6 = registryLoweringFor(.metal, .q6_k, .rows_2_8, .none, .small_batch);
    try std.testing.expectEqual(LoweringRoute.generated_production, metal_q6.production_route);
    try std.testing.expectEqual(LoweringRoute.unsupported, metal_q6.candidate_route);
    try std.testing.expectEqual(FallbackReason.none, metal_q6.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q6_kernel_id, metal_q6.production_kernel_id);
    try std.testing.expectEqualStrings("", metal_q6.kernel_id);
    try std.testing.expectEqualStrings("", metal_q6.candidate_source_path);

    const metal_q6_bias = registryLoweringFor(.metal, .q6_k, .rows_2_8, .bias, .small_batch);
    try std.testing.expectEqual(LoweringRoute.generated_production, metal_q6_bias.production_route);
    try std.testing.expectEqual(LoweringRoute.unsupported, metal_q6_bias.candidate_route);
    try std.testing.expectEqual(FallbackReason.none, metal_q6_bias.fallback_reason);
    try std.testing.expectEqualStrings(first_general_metal_q6_bias_kernel_id, metal_q6_bias.production_kernel_id);
    try std.testing.expectEqualStrings("", metal_q6_bias.kernel_id);
    try std.testing.expectEqualStrings("", metal_q6_bias.candidate_source_path);
    try std.testing.expect(std.mem.containsAtLeast(u8, first_general_metal_q6_bias_source, 1, "threadgroup float partial[32];"));
    try std.testing.expect(std.mem.containsAtLeast(u8, first_general_metal_q6_bias_source, 1, "if (simdgroup_id == 0u && lane_id >= 4u) partial[lane_id] = 0.0f;"));
    try std.testing.expect(std.mem.containsAtLeast(u8, first_general_metal_q6_bias_source, 1, "simdgroup_index_in_threadgroup"));

    const metal_q6_bias_gelu = registryLoweringFor(.metal, .q6_k, .rows_2_8, .bias_gelu, .small_batch);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, metal_q6_bias_gelu.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal_q6_bias_gelu.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, metal_q6_bias_gelu.fallback_reason);
    try std.testing.expectEqualStrings("metal_handwritten_quant_matmul", metal_q6_bias_gelu.production_kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q6_bias_gelu_kernel_id, metal_q6_bias_gelu.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q6_bias_gelu_artifact_source_path, metal_q6_bias_gelu.candidate_source_path);
    try std.testing.expect(std.mem.containsAtLeast(u8, first_general_metal_q6_bias_gelu_source, 1, "threadgroup float partial[32];"));
    try std.testing.expect(std.mem.containsAtLeast(u8, first_general_metal_q6_bias_gelu_source, 1, "if (simdgroup_id == 0u && lane_id >= 4u) partial[lane_id] = 0.0f;"));
    try std.testing.expect(std.mem.containsAtLeast(u8, first_general_metal_q6_bias_gelu_source, 1, "simdgroup_index_in_threadgroup"));

    const miss = registryLoweringFor(.cuda, .unknown, .rows_2_8, .bias_gelu, .small_batch);
    try std.testing.expectEqual(LoweringRoute.unsupported, miss.production_route);
    try std.testing.expectEqual(FallbackReason.unsupported_format, miss.fallback_reason);
    try std.testing.expectEqual(quant_matmul.DispatchKind.small_batch, miss.plan_id.dispatch);
    try std.testing.expectEqual(quant_matmul.DispatchKind.small_batch, miss.schedule.dispatch);

    const format_miss_with_bad_shape = registryLoweringFor(.cuda, .unknown, .rows_0, .bias_gelu, .small_batch);
    try std.testing.expectEqual(LoweringRoute.unsupported, format_miss_with_bad_shape.production_route);
    try std.testing.expectEqual(FallbackReason.unsupported_format, format_miss_with_bad_shape.fallback_reason);
    try std.testing.expectEqual(quant_matmul.DispatchKind.small_batch, format_miss_with_bad_shape.plan_id.dispatch);
}

test "quant kernel compiler registry rejects mismatched dispatch keys" {
    const miss = registryLoweringFor(.cuda, .q4_k, .rows_2_8, .bias_gelu, .mm);
    try std.testing.expectEqual(LoweringRoute.unsupported, miss.production_route);
    try std.testing.expectEqual(LoweringRoute.unsupported, miss.candidate_route);
    try std.testing.expectEqual(FallbackReason.unsupported_shape, miss.fallback_reason);
    try std.testing.expectEqual(quant_matmul.DispatchKind.mm, miss.plan_id.dispatch);
    try std.testing.expectEqual(quant_matmul.DispatchKind.mm, miss.schedule.dispatch);
    try std.testing.expectEqual(@as(usize, 0), miss.schedule.tile_rows);
    try std.testing.expectEqualStrings("", miss.production_kernel_id);
}

test "quant kernel compiler first CUDA candidate stays dev-only but checked in" {
    try std.testing.expect(!first_lazy_benchmark.production_enabled);
    try std.testing.expect(!std.mem.containsAtLeast(u8, first_lazy_benchmark.generated_source_path, 1, "/artifacts/"));

    const contents = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, first_lazy_benchmark.generated_source_path, std.testing.allocator, .limited(128 * 1024));
    defer std.testing.allocator.free(contents);
    const emitted = try emitFirstLazyCudaSource(std.testing.allocator);
    defer std.testing.allocator.free(emitted);

    try std.testing.expectEqualStrings(emitted, contents);

    const route = loweringFor(.cuda, .q4_k, .rows_2_8, .bias_gelu);
    try std.testing.expectEqualStrings(first_lazy_benchmark.generated_source_path, route.candidate_source_path);
    try std.testing.expectEqual(@as(usize, 2), route.schedule.tile_rows);
    try std.testing.expectEqual(@as(usize, 4), route.schedule.tile_cols);
    try std.testing.expectEqual(@as(usize, 256), route.schedule.threads_per_block);

    const candidate_schedule = candidateScheduleFor(route);
    try std.testing.expectEqual(@as(usize, 1), candidate_schedule.tile_rows);
    try std.testing.expectEqual(@as(usize, 1), candidate_schedule.tile_cols);
    try std.testing.expectEqual(@as(usize, 128), candidate_schedule.threads_per_block);
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "if (blockDim.x != 128) return;"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "__shared__ float partial[128];"));
}

test "quant kernel compiler first Metal lazy target stays blocked by timing drift evidence" {
    try std.testing.expect(!std.mem.containsAtLeast(u8, first_lazy_metal_source_path, 1, "/artifacts/"));

    const route = loweringFor(.metal, .q4_k, .rows_2_8, .bias_gelu);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, route.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, route.candidate_route);
    try std.testing.expectEqual(FallbackReason.generated_artifact_missing, route.fallback_reason);
    try std.testing.expectEqualStrings("metal_handwritten_quant_matmul", route.production_kernel_id);
    try std.testing.expectEqualStrings(first_lazy_metal_kernel_id, route.kernel_id);
    try std.testing.expectEqualStrings(first_lazy_metal_artifact_source_path, route.candidate_source_path);

    const artifact = generatedArtifactForKernel(.metal, first_lazy_metal_kernel_id).?;
    try std.testing.expect(artifactRuntimeWired(artifact));
    try std.testing.expect(artifactHasMetalProviderRouteEvidence(artifact));
    try std.testing.expect(!artifactHasPromotionEvidence(artifact));
    try std.testing.expectEqualStrings("unstable_benchmark_timing", artifactPromotionBlocker(artifact));
    try std.testing.expectEqualStrings("TERMITE_METAL_ENABLE_ANTFLY_Q4_K_SMALL_BATCH_BIAS_GELU", std.mem.span(artifactRuntimeGateEnv(artifact).?));

    const counters = countersForLowering(route);
    try std.testing.expectEqual(@as(usize, 1), counters.quant_kernel_planned_ops);
    try std.testing.expectEqual(@as(usize, 1), counters.quant_kernel_handwritten_production);
    try std.testing.expectEqual(@as(usize, 0), counters.quant_kernel_generated_production);
    try std.testing.expectEqual(@as(usize, 1), counters.quant_kernel_generated_candidates);

    const contents = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, first_lazy_metal_source_path, std.testing.allocator, .limited(128 * 1024));
    defer std.testing.allocator.free(contents);
    const compiled_lazy = compileMetalKernelSource(.q4_k, .rows_2_8, .bias_gelu).?;
    const emitted = try emitCompiledSource(std.testing.allocator, compiled_lazy);
    defer emitted.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings(emitted.data, contents);
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "for (int lane = (int)tid; lane < 256; lane += 64)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "threadgroup float partial[64];"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "if (tid < 64) partial[tid] = acc;"));
}

test "quant kernel compiler compiles promoted Metal source from descriptor route" {
    const compiled = compileMetalKernelSource(.q6_k, .rows_2_8, .bias).?;

    try std.testing.expectEqual(Backend.metal, compiled.request.backend);
    try std.testing.expectEqual(quant_matmul.Format.q6_k, compiled.spec.format);
    try std.testing.expectEqual(quant_matmul.Format.q6_k, compiled.ir.format);
    try std.testing.expectEqual(Epilogue.bias, compiled.ir.epilogue);
    try std.testing.expectEqualSlices(IROp, &ir_ops_bias, compiled.ir.ops);
    try std.testing.expectEqualStrings(first_general_metal_q6_bias_kernel_id, compiled.artifact.kernel_id);
    try std.testing.expectEqualStrings(first_general_metal_q6_bias_source_path, compiled.source_path);
    try std.testing.expectEqualStrings(first_general_metal_q6_bias_artifact_source_path, compiled.artifact_source_path);
    try std.testing.expectEqualStrings(first_general_metal_q6_bias_check_command, compiled.check_command);
    try std.testing.expectEqualStrings(first_general_metal_q6_bias_source, compiled.source);
    try std.testing.expect(compiled.production_enabled);
    try std.testing.expect(compiled.runtime_gate_env == null);
    try std.testing.expectEqual(LoweringRoute.generated_production, compiled.lowering.production_route);
    try std.testing.expectEqual(LoweringRoute.unsupported, compiled.lowering.candidate_route);
    try std.testing.expectEqual(FallbackReason.none, compiled.lowering.fallback_reason);
    try std.testing.expect(try compiledSourceHeaderMatchesPlan(std.testing.allocator, compiled));

    const q5_1 = compileMetalKernelSource(.q5_1, .rows_2_8, .none).?;
    try std.testing.expect(!q5_1.production_enabled);
    try std.testing.expectEqualStrings(first_general_metal_q5_1_kernel_id, q5_1.artifact.kernel_id);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, q5_1.lowering.production_route);
    try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, q5_1.lowering.candidate_route);
    try std.testing.expect(q5_1.runtime_gate_env != null);
    try std.testing.expectEqualStrings("TERMITE_METAL_ENABLE_ANTFLY_Q5_1_SMALL_BATCH", std.mem.span(q5_1.runtime_gate_env.?));
    try std.testing.expect(std.mem.containsAtLeast(u8, q5_1.source, 1, "Dev-only generated Metal candidate"));
    try std.testing.expect(try compiledSourceHeaderMatchesPlan(std.testing.allocator, q5_1));

    const wrong_kernel_header = try std.mem.replaceOwned(
        u8,
        std.testing.allocator,
        compiled.source,
        first_general_metal_q6_bias_kernel_id,
        first_general_metal_q5_bias_kernel_id,
    );
    defer std.testing.allocator.free(wrong_kernel_header);
    try std.testing.expect(!try sourceHeaderMatchesCompiledPlan(std.testing.allocator, compiled, wrong_kernel_header));
}

fn expectSameMetalKernelBody(expected: []const u8, actual: []const u8) !void {
    const marker = "#include <metal_stdlib>";
    const expected_start = std.mem.indexOf(u8, expected, marker) orelse return error.MissingMetalSourceBody;
    const actual_start = std.mem.indexOf(u8, actual, marker) orelse return error.MissingMetalSourceBody;
    try std.testing.expectEqualStrings(expected[expected_start..], actual[actual_start..]);
}

test "quant kernel compiler emits migrated Metal source from descriptor data" {
    const q8_cases = [_]struct {
        epilogue: Epilogue,
        expected_source: []const u8,
    }{
        .{ .epilogue = .none, .expected_source = first_general_metal_q8_source },
        .{ .epilogue = .bias, .expected_source = first_general_metal_q8_bias_source },
        .{ .epilogue = .bias_gelu, .expected_source = first_general_metal_q8_bias_gelu_source },
        .{ .epilogue = .relu, .expected_source = first_general_metal_q8_relu_source },
    };

    for (q8_cases) |case| {
        const q8 = compileMetalKernelSource(.q8_0, .rows_2_8, case.epilogue).?;
        const emitted_q8 = try emitCompiledSource(std.testing.allocator, q8);
        defer emitted_q8.deinit(std.testing.allocator);

        try std.testing.expect(emitted_q8.owned);
        try expectSameMetalKernelBody(case.expected_source, emitted_q8.data);
        try std.testing.expect(try compiledSourceHeaderMatchesSource(std.testing.allocator, q8, emitted_q8.data));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q8.data, 1, "const int block_count = in_dim >> 5;"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q8.data, 1, "* 34"));
    }

    const q8_family_cases = [_]struct {
        format: quant_matmul.Format,
        expected_source: []const u8,
        expected_block_shift: []const u8,
        expected_block_bytes: []const u8,
    }{
        .{ .format = .q8_1, .expected_source = first_general_metal_q8_1_source, .expected_block_shift = "const int block_count = in_dim >> 5;", .expected_block_bytes = "* 36" },
        .{ .format = .q8_k, .expected_source = first_general_metal_q8_k_source, .expected_block_shift = "const int block_count = in_dim >> 8;", .expected_block_bytes = "* 292" },
    };

    for (q8_family_cases) |case| {
        const q8_family = compileMetalKernelSource(case.format, .rows_2_8, .none).?;
        const emitted_q8_family = try emitCompiledSource(std.testing.allocator, q8_family);
        defer emitted_q8_family.deinit(std.testing.allocator);

        try std.testing.expect(emitted_q8_family.owned);
        try expectSameMetalKernelBody(case.expected_source, emitted_q8_family.data);
        try std.testing.expect(try compiledSourceHeaderMatchesSource(std.testing.allocator, q8_family, emitted_q8_family.data));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q8_family.data, 1, "block[4 + lane]"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q8_family.data, 1, case.expected_block_shift));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q8_family.data, 1, case.expected_block_bytes));
    }

    const q2_cases = [_]struct {
        epilogue: Epilogue,
        expected_source: []const u8,
    }{
        .{ .epilogue = .none, .expected_source = first_general_metal_q2_source },
        .{ .epilogue = .bias, .expected_source = first_general_metal_q2_bias_source },
        .{ .epilogue = .bias_gelu, .expected_source = first_general_metal_q2_bias_gelu_source },
    };

    for (q2_cases) |case| {
        const q2 = compileMetalKernelSource(.q2_k, .rows_2_8, case.epilogue).?;
        const emitted_q2 = try emitCompiledSource(std.testing.allocator, q2);
        defer emitted_q2.deinit(std.testing.allocator);

        try std.testing.expect(emitted_q2.owned);
        try expectSameMetalKernelBody(case.expected_source, emitted_q2.data);
        try std.testing.expect(try compiledSourceHeaderMatchesSource(std.testing.allocator, q2, emitted_q2.data));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q2.data, 1, "block + 16"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q2.data, 1, "block + 18"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q2.data, 1, "block[20 +"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q2.data, 1, "const int block_count = in_dim >> 8;"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q2.data, 1, "* 84"));
    }

    const q3_cases = [_]struct {
        epilogue: Epilogue,
        expected_source: []const u8,
    }{
        .{ .epilogue = .none, .expected_source = first_general_metal_q3_source },
        .{ .epilogue = .bias, .expected_source = first_general_metal_q3_bias_source },
        .{ .epilogue = .bias_gelu, .expected_source = first_general_metal_q3_bias_gelu_source },
    };

    for (q3_cases) |case| {
        const q3 = compileMetalKernelSource(.q3_k, .rows_2_8, case.epilogue).?;
        const emitted_q3 = try emitCompiledSource(std.testing.allocator, q3);
        defer emitted_q3.deinit(std.testing.allocator);

        try std.testing.expect(emitted_q3.owned);
        try expectSameMetalKernelBody(case.expected_source, emitted_q3.data);
        try std.testing.expect(try compiledSourceHeaderMatchesSource(std.testing.allocator, q3, emitted_q3.data));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q3.data, 1, "block + 108"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q3.data, 1, "block + 96"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q3.data, 1, "block[32 +"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q3.data, 1, "const int block_count = in_dim >> 8;"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q3.data, 1, "* 110"));
    }

    const q4_cases = [_]struct {
        epilogue: Epilogue,
        expected_source: []const u8,
    }{
        .{ .epilogue = .none, .expected_source = first_general_metal_q4_source },
        .{ .epilogue = .bias, .expected_source = first_general_metal_q4_bias_source },
        .{ .epilogue = .bias_gelu, .expected_source = first_lazy_metal_source },
    };

    for (q4_cases) |case| {
        const q4 = compileMetalKernelSource(.q4_k, .rows_2_8, case.epilogue).?;
        const emitted_q4 = try emitCompiledSource(std.testing.allocator, q4);
        defer emitted_q4.deinit(std.testing.allocator);

        try std.testing.expect(emitted_q4.owned);
        try expectSameMetalKernelBody(case.expected_source, emitted_q4.data);
        try std.testing.expect(try compiledSourceHeaderMatchesSource(std.testing.allocator, q4, emitted_q4.data));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q4.data, 1, "block + 2"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q4.data, 1, "block + 4"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q4.data, 1, "block + 16"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q4.data, 1, "const int block_count = in_dim >> 8;"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q4.data, 1, "* 144"));
    }

    const q5_cases = [_]struct {
        epilogue: Epilogue,
        expected_source: []const u8,
    }{
        .{ .epilogue = .none, .expected_source = first_general_metal_q5_source },
        .{ .epilogue = .bias, .expected_source = first_general_metal_q5_bias_source },
        .{ .epilogue = .bias_gelu, .expected_source = first_general_metal_q5_bias_gelu_source },
    };

    for (q5_cases) |case| {
        const q5 = compileMetalKernelSource(.q5_k, .rows_2_8, case.epilogue).?;
        const emitted_q5 = try emitCompiledSource(std.testing.allocator, q5);
        defer emitted_q5.deinit(std.testing.allocator);

        try std.testing.expect(emitted_q5.owned);
        try expectSameMetalKernelBody(case.expected_source, emitted_q5.data);
        try std.testing.expect(try compiledSourceHeaderMatchesSource(std.testing.allocator, q5, emitted_q5.data));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q5.data, 1, "block + 2"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q5.data, 1, "block + 4"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q5.data, 1, "block + 16"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q5.data, 1, "block + 48"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q5.data, 1, "const int block_count = in_dim >> 8;"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q5.data, 1, "* 176"));
    }

    const q6_cases = [_]struct {
        epilogue: Epilogue,
        expected_source: []const u8,
    }{
        .{ .epilogue = .none, .expected_source = first_general_metal_q6_source },
        .{ .epilogue = .bias, .expected_source = first_general_metal_q6_bias_source },
        .{ .epilogue = .bias_gelu, .expected_source = first_general_metal_q6_bias_gelu_source },
    };

    for (q6_cases) |case| {
        const q6 = compileMetalKernelSource(.q6_k, .rows_2_8, case.epilogue).?;
        const emitted_q6 = try emitCompiledSource(std.testing.allocator, q6);
        defer emitted_q6.deinit(std.testing.allocator);

        try std.testing.expect(emitted_q6.owned);
        try expectSameMetalKernelBody(case.expected_source, emitted_q6.data);
        try std.testing.expect(try compiledSourceHeaderMatchesSource(std.testing.allocator, q6, emitted_q6.data));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q6.data, 1, "block + 128"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q6.data, 1, "block + 192"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q6.data, 1, "block + 208"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q6.data, 1, "const int block_count = in_dim >> 8;"));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_q6.data, 1, "* 210"));
    }

    const legacy_scalar_cases = [_]struct {
        format: quant_matmul.Format,
        expected_source: []const u8,
        expected_qs: []const u8,
        expected_block_shift: []const u8,
        expected_block_bytes: []const u8,
    }{
        .{ .format = .q4_0, .expected_source = first_general_metal_q4_0_source, .expected_qs = "block[2 + packed_index]", .expected_block_shift = "const int block_count = in_dim >> 5;", .expected_block_bytes = "* 18" },
        .{ .format = .q4_1, .expected_source = first_general_metal_q4_1_source, .expected_qs = "block[4 + packed_index]", .expected_block_shift = "const int block_count = in_dim >> 5;", .expected_block_bytes = "* 20" },
        .{ .format = .q5_0, .expected_source = first_general_metal_q5_0_source, .expected_qs = "block[6 + packed_index]", .expected_block_shift = "const int block_count = in_dim >> 5;", .expected_block_bytes = "* 22" },
        .{ .format = .q5_1, .expected_source = first_general_metal_q5_1_source, .expected_qs = "block[8 + packed_index]", .expected_block_shift = "const int block_count = in_dim >> 5;", .expected_block_bytes = "* 24" },
    };

    for (legacy_scalar_cases) |case| {
        const legacy_scalar = compileMetalKernelSource(case.format, .rows_2_8, .none).?;
        const emitted_legacy_scalar = try emitCompiledSource(std.testing.allocator, legacy_scalar);
        defer emitted_legacy_scalar.deinit(std.testing.allocator);

        try std.testing.expect(emitted_legacy_scalar.owned);
        try expectSameMetalKernelBody(case.expected_source, emitted_legacy_scalar.data);
        try std.testing.expect(try compiledSourceHeaderMatchesSource(std.testing.allocator, legacy_scalar, emitted_legacy_scalar.data));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_legacy_scalar.data, 1, case.expected_qs));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_legacy_scalar.data, 1, case.expected_block_shift));
        try std.testing.expect(std.mem.containsAtLeast(u8, emitted_legacy_scalar.data, 1, case.expected_block_bytes));
    }

    const cuda_q4 = compileQuantKernelSource(.{
        .backend = .cuda,
        .format = .q4_k,
        .row_bucket = .rows_2_8,
        .epilogue = .bias_gelu,
    }).?;
    const emitted_cuda_q4 = try emitCompiledSource(std.testing.allocator, cuda_q4);
    defer emitted_cuda_q4.deinit(std.testing.allocator);

    try std.testing.expect(!emitted_cuda_q4.owned);
    try std.testing.expectEqualStrings(cuda_q4.source, emitted_cuda_q4.data);
}

test "quant kernel compiler compile API requires a generated artifact" {
    try std.testing.expect(compileMetalKernelSource(.q6_k, .rows_9_64, .bias_gelu) == null);
    try std.testing.expect(compileMetalKernelSource(.q8_0, .rows_2_8, .argmax) == null);

    const lowering = loweringFor(.metal, .q6_k, .rows_9_64, .bias_gelu);
    try std.testing.expectEqual(LoweringRoute.handwritten_production, lowering.production_route);
    try std.testing.expectEqual(LoweringRoute.unsupported, lowering.candidate_route);
}

test "quant kernel compiler compile API rejects route metadata drift" {
    var checked: usize = 0;
    var promoted_checked: usize = 0;
    var candidate_checked: usize = 0;
    for (first_generated_artifacts) |artifact| {
        const compiled = compileQuantKernelSource(.{
            .backend = artifact.backend,
            .format = artifact.format,
            .row_bucket = artifact.row_bucket,
            .epilogue = artifact.epilogue,
        }) orelse return error.MissingCompiledQuantKernelSource;
        try std.testing.expect(compiledSourceMatchesRoute(compiled));
        checked += 1;
        if (artifactHasPromotionEvidence(artifact)) {
            promoted_checked += 1;
            try std.testing.expectEqual(LoweringRoute.generated_production, compiled.lowering.production_route);
            try std.testing.expectEqualStrings(artifact.kernel_id, compiled.lowering.production_kernel_id);
        } else {
            candidate_checked += 1;
            try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, compiled.lowering.candidate_route);
            try std.testing.expectEqualStrings(artifact.kernel_id, compiled.lowering.kernel_id);
        }
    }
    try std.testing.expectEqual(first_generated_artifacts.len, checked);
    try std.testing.expect(promoted_checked > 0);
    try std.testing.expect(candidate_checked > 0);

    var wrong_artifact = compileMetalKernelSource(.q6_k, .rows_2_8, .bias_gelu).?;
    wrong_artifact.artifact.kernel_id = first_general_metal_q5_bias_gelu_kernel_id;
    try std.testing.expect(!compiledSourceMatchesRoute(wrong_artifact));

    var wrong_candidate = compileMetalKernelSource(.q4_0, .rows_2_8, .none).?;
    wrong_candidate.lowering.kernel_id = first_general_metal_q6_kernel_id;
    try std.testing.expect(!compiledSourceMatchesRoute(wrong_candidate));

    var wrong_source_path = compileMetalKernelSource(.q6_k, .rows_2_8, .bias_gelu).?;
    wrong_source_path.source_path = first_general_metal_q5_bias_gelu_source_path;
    try std.testing.expect(!compiledSourceMatchesRoute(wrong_source_path));

    var wrong_check_command = compileMetalKernelSource(.q6_k, .rows_2_8, .bias_gelu).?;
    wrong_check_command.check_command = first_general_metal_q5_bias_gelu_check_command;
    try std.testing.expect(!compiledSourceMatchesRoute(wrong_check_command));

    var wrong_gate = compileMetalKernelSource(.q5_1, .rows_2_8, .none).?;
    wrong_gate.runtime_gate_env = null;
    try std.testing.expect(!compiledSourceMatchesRoute(wrong_gate));

    var wrong_production_state = compileMetalKernelSource(.q6_k, .rows_2_8, .bias).?;
    wrong_production_state.production_enabled = false;
    try std.testing.expect(!compiledSourceMatchesRoute(wrong_production_state));
}

test "quant kernel compiler docs describe compile API guardrail" {
    const contents = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "QUANT_KERNEL_COMPILER.md", std.testing.allocator, .limited(64 * 1024));
    defer std.testing.allocator.free(contents);

    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "compileQuantKernelSource(...)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "is also a guardrail"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "descriptor, IR, route lowering, generated"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "production bit, and Metal runtime gate"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "Drift returns"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "compileMetalKernelSource(.q6_k, .rows_2_8, .bias_gelu)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "emitCompiledSource(allocator, compiled)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "artifact_source_path, check_command, runtime_gate_env"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "Repeated promotion and production-regression checks run two unrecorded warmup"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "`warmup_repeat_runs`"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "`metal_promotion_warmup_repeat_runs`"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "Route-all evidence is an observability check"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "The route-all evidence covers 50 generated cases"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "all 50 must be route-ready"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "46 must have provider-route evidence"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "17 candidate kernels are guarded"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "`speedup_gate_missing` for Q4_0, Q5_0, Q5_1, Q4_K none, Q6_K bias+GELU, and\nQ8_0 bias+GELU"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "`unstable_benchmark_timing` for Q4_1, Q4_K bias+GELU"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "Q5_K none, Q5_K bias+GELU, Q8_1 none, and Q8_K none"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "and 5 are"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "route-evidence-only"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "because their handwritten baseline is unsupported"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "Timing drift from\n  an individual repeated run is reported as `production_regression_timing_drift`"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "does not hide the route/provider evidence"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "Promoted"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "generated-production routes report an empty `promotion_blocker`"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "`runtime_route_only`,"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "Unsupported-handwritten-baseline candidates are route-evidence-only"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "cannot be promoted"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "by the sequential speedup gate"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "benchmark_manifest=antfly.quant_kernel_benchmarks.v4:34:<fingerprint>"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "Runtime Observability"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "`fast_path_misses`: the sum of explicit, mutually exclusive fallback reasons"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "Actual generated Metal dispatch counters are not treated as handwritten\nfallbacks"));
}

fn metalRuntimeQuantFormatConstant(format: quant_matmul.Format) ?[]const u8 {
    return switch (format) {
        .q2_k => "TERMITE_METAL_QUANT_FORMAT_Q2_K",
        .q3_k => "TERMITE_METAL_QUANT_FORMAT_Q3_K",
        .q4_0 => "TERMITE_METAL_QUANT_FORMAT_Q4_0",
        .q4_1 => "TERMITE_METAL_QUANT_FORMAT_Q4_1",
        .q4_k => "TERMITE_METAL_QUANT_FORMAT_Q4_K",
        .q5_0 => "TERMITE_METAL_QUANT_FORMAT_Q5_0",
        .q5_1 => "TERMITE_METAL_QUANT_FORMAT_Q5_1",
        .q5_k => "TERMITE_METAL_QUANT_FORMAT_Q5_K",
        .q6_k => "TERMITE_METAL_QUANT_FORMAT_Q6_K",
        .q8_0 => "TERMITE_METAL_QUANT_FORMAT_Q8_0",
        .q8_1 => "TERMITE_METAL_QUANT_FORMAT_Q8_1",
        .q8_k => "TERMITE_METAL_QUANT_FORMAT_Q8_K",
        else => null,
    };
}

fn metalRuntimeGeneratedEpilogueConstant(epilogue: Epilogue) ?[]const u8 {
    return switch (epilogue) {
        .none => "TERMITE_METAL_GENERATED_QUANT_EPILOGUE_NONE",
        .bias => "TERMITE_METAL_GENERATED_QUANT_EPILOGUE_BIAS",
        .bias_gelu => "TERMITE_METAL_GENERATED_QUANT_EPILOGUE_BIAS_GELU",
        .relu => "TERMITE_METAL_GENERATED_QUANT_EPILOGUE_RELU",
        else => null,
    };
}

test "quant kernel compiler production Metal source includes only runtime-wired generated kernels" {
    const contents = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "src/backends/metal_kernels.m", std.testing.allocator, .limited(3 * 1024 * 1024));
    defer std.testing.allocator.free(contents);

    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, "src/ops/metal/generated"));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_lazy_metal_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_lazy_metal_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q4_0_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q4_0_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q4_1_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q4_1_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q5_0_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q5_0_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q5_1_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q5_1_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q4_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q4_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q4_bias_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q4_bias_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q8_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q8_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q8_bias_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q8_bias_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q8_bias_gelu_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q8_bias_gelu_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q8_relu_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q8_relu_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q2_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q2_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q2_bias_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q2_bias_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q2_bias_gelu_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q2_bias_gelu_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q3_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q3_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q3_bias_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q3_bias_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q3_bias_gelu_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q3_bias_gelu_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q8_1_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q8_1_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q8_k_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q8_k_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q5_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q5_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q5_bias_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q5_bias_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q5_bias_gelu_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q5_bias_gelu_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q6_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q6_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q6_bias_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q6_bias_kernel_id));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q6_bias_gelu_source_path));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, first_general_metal_q6_bias_gelu_kernel_id));
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q8_0_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, contents, "TERMITE_METAL_DISABLE_ANTFLY_Q8_0_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q2_K_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, contents, "TERMITE_METAL_DISABLE_ANTFLY_Q2_K_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 4), std.mem.count(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q3_K_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, contents, "TERMITE_METAL_DISABLE_ANTFLY_Q3_K_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q4_0_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q4_1_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, contents, "TERMITE_METAL_DISABLE_ANTFLY_Q4_1_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q5_0_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, contents, "TERMITE_METAL_DISABLE_ANTFLY_Q5_0_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q5_1_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, contents, "TERMITE_METAL_DISABLE_ANTFLY_Q5_1_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q8_1_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, contents, "TERMITE_METAL_DISABLE_ANTFLY_Q8_1_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q8_K_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, contents, "TERMITE_METAL_DISABLE_ANTFLY_Q8_K_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q4_K_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, contents, "TERMITE_METAL_DISABLE_ANTFLY_Q4_K_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 3), std.mem.count(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q5_K_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, contents, "TERMITE_METAL_DISABLE_ANTFLY_Q5_K_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 2), std.mem.count(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q6_K_SMALL_BATCH"));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, contents, "TERMITE_METAL_DISABLE_ANTFLY_Q6_K_SMALL_BATCH"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "typedef enum termite_metal_generated_quant_epilogue"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "TERMITE_METAL_GENERATED_QUANT_EPILOGUE_NONE"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "TERMITE_METAL_GENERATED_QUANT_EPILOGUE_BIAS"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "TERMITE_METAL_GENERATED_QUANT_EPILOGUE_BIAS_GELU"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "TERMITE_METAL_GENERATED_QUANT_EPILOGUE_RELU"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1,
        \\TERMITE_METAL_QUANT_FORMAT_Q8_0,
        \\        NULL,
        \\        runtime != NULL ? runtime->antfly_q8_0_small_batch_bias_pipeline : nil,
        \\        runtime != NULL ? &runtime->antfly_q8_0_small_batch_bias_dispatches : NULL,
        \\        TERMITE_METAL_GENERATED_QUANT_EPILOGUE_BIAS,
    ));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1,
        \\TERMITE_METAL_QUANT_FORMAT_Q8_0,
        \\        "TERMITE_METAL_ENABLE_ANTFLY_Q8_0_SMALL_BATCH_BIAS_GELU",
        \\        runtime != NULL ? runtime->antfly_q8_0_small_batch_bias_gelu_pipeline : nil,
        \\        runtime != NULL ? &runtime->antfly_q8_0_small_batch_bias_gelu_dispatches : NULL,
        \\        TERMITE_METAL_GENERATED_QUANT_EPILOGUE_BIAS_GELU,
    ));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1,
        \\TERMITE_METAL_QUANT_FORMAT_Q5_K,
        \\        NULL,
        \\        runtime != NULL ? runtime->antfly_q5_k_small_batch_bias_pipeline : nil,
        \\        runtime != NULL ? &runtime->antfly_q5_k_small_batch_bias_dispatches : NULL,
        \\        TERMITE_METAL_GENERATED_QUANT_EPILOGUE_BIAS,
    ));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1,
        \\TERMITE_METAL_QUANT_FORMAT_Q5_K,
        \\        "TERMITE_METAL_ENABLE_ANTFLY_Q5_K_SMALL_BATCH_BIAS_GELU",
        \\        runtime != NULL ? runtime->antfly_q5_k_small_batch_bias_gelu_pipeline : nil,
        \\        runtime != NULL ? &runtime->antfly_q5_k_small_batch_bias_gelu_dispatches : NULL,
        \\        TERMITE_METAL_GENERATED_QUANT_EPILOGUE_BIAS_GELU,
    ));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1,
        \\TERMITE_METAL_QUANT_FORMAT_Q6_K,
        \\        NULL,
        \\        runtime != NULL ? runtime->antfly_q6_k_small_batch_bias_pipeline : nil,
        \\        runtime != NULL ? &runtime->antfly_q6_k_small_batch_bias_dispatches : NULL,
        \\        TERMITE_METAL_GENERATED_QUANT_EPILOGUE_BIAS,
    ));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1,
        \\TERMITE_METAL_QUANT_FORMAT_Q6_K,
        \\        "TERMITE_METAL_ENABLE_ANTFLY_Q6_K_SMALL_BATCH_BIAS_GELU",
        \\        runtime != NULL ? runtime->antfly_q6_k_small_batch_bias_gelu_pipeline : nil,
        \\        runtime != NULL ? &runtime->antfly_q6_k_small_batch_bias_gelu_dispatches : NULL,
        \\        TERMITE_METAL_GENERATED_QUANT_EPILOGUE_BIAS_GELU,
    ));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1,
        \\runtime != NULL ? &runtime->antfly_q2_k_small_batch_bias_dispatches : NULL,
        \\        128u,
        \\        1u,
    ));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1,
        \\runtime != NULL ? &runtime->antfly_q8_0_small_batch_bias_gelu_dispatches : NULL,
        \\        32u,
        \\        2u,
    ));
    const none_encoder = std.mem.indexOf(u8, contents, "static int termite_metal_encode_quant_matmul_generic_none_on_encoder") orelse return error.MissingMetalNoneEncoder;
    const q8_encoder = std.mem.indexOfPos(u8, contents, none_encoder, "static int termite_metal_encode_q8_0_linear(") orelse return error.MissingMetalQ8Encoder;
    const q8_disable = std.mem.indexOf(u8, contents, "getenv(\"TERMITE_METAL_DISABLE_ANTFLY_Q8_0_SMALL_BATCH\")") orelse return error.MissingMetalQ8Disable;
    const q2_disable = std.mem.indexOf(u8, contents, "getenv(\"TERMITE_METAL_DISABLE_ANTFLY_Q2_K_SMALL_BATCH\")") orelse return error.MissingMetalQ2Disable;
    const q4_0_enable = std.mem.indexOf(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q4_0_SMALL_BATCH") orelse return error.MissingMetalQ4_0Enable;
    const q4_1_enable = std.mem.indexOf(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q4_1_SMALL_BATCH") orelse return error.MissingMetalQ4_1Enable;
    const q5_0_enable = std.mem.indexOf(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q5_0_SMALL_BATCH") orelse return error.MissingMetalQ5_0Enable;
    const q5_1_enable = std.mem.indexOf(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q5_1_SMALL_BATCH") orelse return error.MissingMetalQ5_1Enable;
    const q8_1_enable = std.mem.indexOf(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q8_1_SMALL_BATCH") orelse return error.MissingMetalQ8_1Enable;
    const q8_k_enable = std.mem.indexOf(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q8_K_SMALL_BATCH") orelse return error.MissingMetalQ8_KEnable;
    const q3_disable = std.mem.indexOf(u8, contents, "getenv(\"TERMITE_METAL_DISABLE_ANTFLY_Q3_K_SMALL_BATCH\")") orelse return error.MissingMetalQ3Disable;
    const q4_enable = std.mem.indexOf(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q4_K_SMALL_BATCH") orelse return error.MissingMetalQ4Enable;
    const q5_enable = std.mem.indexOf(u8, contents, "TERMITE_METAL_ENABLE_ANTFLY_Q5_K_SMALL_BATCH") orelse return error.MissingMetalQ5Enable;
    const q6_disable = std.mem.indexOf(u8, contents, "getenv(\"TERMITE_METAL_DISABLE_ANTFLY_Q6_K_SMALL_BATCH\")") orelse return error.MissingMetalQ6Disable;
    try std.testing.expect(q8_disable > none_encoder and q8_disable < q8_encoder);
    try std.testing.expect(q2_disable > none_encoder and q2_disable < q8_encoder);
    try std.testing.expect(q3_disable > none_encoder and q3_disable < q8_encoder);
    try std.testing.expect(q4_0_enable > none_encoder and q4_0_enable < q8_encoder);
    try std.testing.expect(q4_1_enable > none_encoder and q4_1_enable < q8_encoder);
    try std.testing.expect(q5_0_enable > none_encoder and q5_0_enable < q8_encoder);
    try std.testing.expect(q5_1_enable > none_encoder and q5_1_enable < q8_encoder);
    try std.testing.expect(q8_1_enable > none_encoder and q8_1_enable < q8_encoder);
    try std.testing.expect(q8_k_enable > none_encoder and q8_k_enable < q8_encoder);
    try std.testing.expect(q4_enable > none_encoder and q4_enable < q8_encoder);
    try std.testing.expect(q5_enable > none_encoder and q5_enable < q8_encoder);
    try std.testing.expect(q6_disable > none_encoder and q6_disable < q8_encoder);
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, "strcmp(kernel_name, \"antfly_q4_k_small_batch_msl_v1\") == 0"));
    const q5_1_kernel = std.mem.indexOf(u8, contents, "kernel void antfly_q5_1_small_batch_msl_v1") orelse return error.MissingMetalQ5_1Kernel;
    const q5_1_kernel_end = std.mem.indexOfPos(u8, contents, q5_1_kernel, "inline float antfly_q8_1_half_le_to_float") orelse return error.MissingMetalQ5_1KernelEnd;
    const q5_1_kernel_body = contents[q5_1_kernel..q5_1_kernel_end];
    try std.testing.expect(std.mem.containsAtLeast(u8, q5_1_kernel_body, 1, "int col0 = int(group_pos.x << 1); int col1 = col0 + 1;"));
    try std.testing.expect(std.mem.containsAtLeast(u8, q5_1_kernel_body, 1, "device const uchar *col1_weight = has_col1 ? weight_q5_1 + col1 * block_count * 24 : col0_weight;"));
    try std.testing.expect(!std.mem.containsAtLeast(u8, q5_1_kernel_body, 1, "int col = int(group_pos.x); int row = int(group_pos.y);"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "uint32_t threads_per_threadgroup,\n    uint32_t cols_per_threadgroup,"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "const NSUInteger threads_per_threadgroup_size = (NSUInteger)threads_per_threadgroup;"));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, "const NSUInteger threads_per_threadgroup = (use_antfly_q2_k_small_batch || use_antfly_q3_k_small_batch"));
    const launch_helper = std.mem.indexOf(u8, contents, "static bool termite_metal_generated_quant_launch_shape_for") orelse return error.MissingMetalGeneratedLaunchShapeHelper;
    const launch_helper_end = std.mem.indexOfPos(u8, contents, launch_helper, "static uint8_t termite_metal_quant_matmul_descriptor_planned_dispatch") orelse return error.MissingMetalGeneratedLaunchShapeHelperEnd;
    try std.testing.expect(launch_helper < none_encoder);
    const launch_helper_body = contents[launch_helper..launch_helper_end];
    for (first_generated_artifacts) |artifact| {
        if (artifact.backend != .metal or artifact.row_bucket != .rows_2_8) continue;
        const format_constant = metalRuntimeQuantFormatConstant(artifact.format) orelse continue;
        const epilogue_constant = metalRuntimeGeneratedEpilogueConstant(artifact.epilogue) orelse continue;
        try std.testing.expect(std.mem.containsAtLeast(u8, launch_helper_body, 1, format_constant));
        try std.testing.expect(std.mem.containsAtLeast(u8, launch_helper_body, 1, epilogue_constant));
    }
    try std.testing.expect(std.mem.containsAtLeast(u8, launch_helper_body, 1, "case TERMITE_METAL_GENERATED_QUANT_EPILOGUE_NONE:"));
    try std.testing.expect(std.mem.containsAtLeast(u8, launch_helper_body, 1, "case TERMITE_METAL_GENERATED_QUANT_EPILOGUE_BIAS:"));
    try std.testing.expect(std.mem.containsAtLeast(u8, launch_helper_body, 1, "case TERMITE_METAL_GENERATED_QUANT_EPILOGUE_BIAS_GELU:"));
    try std.testing.expect(std.mem.containsAtLeast(u8, launch_helper_body, 1, "case TERMITE_METAL_GENERATED_QUANT_EPILOGUE_RELU:"));
    try std.testing.expect(std.mem.containsAtLeast(u8, launch_helper_body, 1, "case TERMITE_METAL_QUANT_FORMAT_Q4_1:"));
    try std.testing.expect(std.mem.containsAtLeast(u8, launch_helper_body, 1, "case TERMITE_METAL_QUANT_FORMAT_Q5_1:"));
    try std.testing.expect(std.mem.containsAtLeast(u8, launch_helper_body, 1, "shape->cols_per_threadgroup = 2u;"));
    try std.testing.expect(std.mem.containsAtLeast(u8, launch_helper_body, 1, "case TERMITE_METAL_QUANT_FORMAT_Q4_K:"));
    try std.testing.expect(std.mem.containsAtLeast(u8, launch_helper_body, 1, "case TERMITE_METAL_QUANT_FORMAT_Q5_K:\n                case TERMITE_METAL_QUANT_FORMAT_Q6_K:\n                    return true;"));
    try std.testing.expect(!std.mem.containsAtLeast(u8, launch_helper_body, 1, "case TERMITE_METAL_QUANT_FORMAT_Q6_K:\n                    shape->threads_per_threadgroup = 32u;"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 2, "termite_metal_generated_quant_launch_shape_for(descriptor->format, TERMITE_METAL_GENERATED_QUANT_EPILOGUE_NONE, descriptor->rows, &launch_shape)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 2, "launch_shape.threads_per_threadgroup"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 2, "launch_shape.cols_per_threadgroup"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "return termite_metal_dispatch_quant_matmul_none(provider, TERMITE_METAL_QUANT_FORMAT_Q4_K"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "return termite_metal_dispatch_quant_matmul_none(provider, TERMITE_METAL_QUANT_FORMAT_Q5_K"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "return termite_metal_dispatch_quant_matmul_none(provider, TERMITE_METAL_QUANT_FORMAT_Q6_K"));
}

test "quant kernel compiler promoted Metal wrappers drop opt-in gates" {
    const contents = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "src/backends/metal_kernels.m", std.testing.allocator, .limited(3 * 1024 * 1024));
    defer std.testing.allocator.free(contents);

    var checked: usize = 0;
    for (first_generated_artifacts) |artifact| {
        if (artifact.backend != .metal or !artifactHasPromotionEvidence(artifact)) continue;
        if (artifactRuntimeGateEnv(artifact) != null) continue;
        const opt_in_gate = artifactCandidateOptInGateEnv(artifact) orelse continue;
        checked += 1;
        const quoted_gate = try std.fmt.allocPrint(std.testing.allocator, "\"{s}\"", .{std.mem.span(opt_in_gate)});
        defer std.testing.allocator.free(quoted_gate);
        try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, quoted_gate));
    }
    try std.testing.expect(checked > 0);
}

fn metalCEnvArgForArtifact(allocator: std.mem.Allocator, artifact: GeneratedArtifact) ![]u8 {
    if (artifactRuntimeGateEnv(artifact)) |env| {
        return try std.fmt.allocPrint(allocator, "\"{s}\"", .{std.mem.span(env)});
    }
    return try allocator.dupe(u8, "NULL");
}

fn metalCBespokeRuntimeWrapperHasArtifact(
    allocator: std.mem.Allocator,
    contents: []const u8,
    artifact: GeneratedArtifact,
    counter_name: []const u8,
    format_constant: []const u8,
    epilogue_constant: []const u8,
) !bool {
    if (artifact.format != .q4_k or (artifact.epilogue != .bias and artifact.epilogue != .bias_gelu)) return false;

    const wrapper_suffix: []const u8 = switch (artifact.epilogue) {
        .bias => "q4_k_bias",
        .bias_gelu => "q4_k_bias_gelu",
        else => unreachable,
    };
    const function_name = try std.fmt.allocPrint(
        allocator,
        "int termite_metal_decode_runtime_apply_quantized_linear_{s}_slot_device(",
        .{wrapper_suffix},
    );
    defer allocator.free(function_name);
    const start = std.mem.indexOf(u8, contents, function_name) orelse return false;
    const end = std.mem.indexOfPos(u8, contents, start + function_name.len, "\n}\n\n") orelse return false;
    const body = contents[start..end];

    if (artifactRuntimeGateEnv(artifact)) |env| {
        const gate = try std.fmt.allocPrint(allocator, "termite_metal_generated_quant_candidate_enabled(\"{s}\")", .{std.mem.span(env)});
        defer allocator.free(gate);
        if (!std.mem.containsAtLeast(u8, body, 1, gate)) return false;
    } else if (artifactCandidateOptInGateEnv(artifact)) |env| {
        const stale_gate = try std.fmt.allocPrint(allocator, "getenv(\"{s}\")", .{std.mem.span(env)});
        defer allocator.free(stale_gate);
        if (std.mem.containsAtLeast(u8, body, 1, stale_gate)) return false;
    }

    const pipeline = try std.fmt.allocPrint(allocator, "runtime->antfly_{s}_pipeline", .{counter_name});
    defer allocator.free(pipeline);
    const dispatch_counter = try std.fmt.allocPrint(allocator, "runtime->antfly_{s}_dispatches += 1;", .{counter_name});
    defer allocator.free(dispatch_counter);
    return std.mem.containsAtLeast(u8, body, 1, format_constant) and
        std.mem.containsAtLeast(u8, body, 1, epilogue_constant) and
        std.mem.containsAtLeast(u8, body, 1, pipeline) and
        std.mem.containsAtLeast(u8, body, 1, dispatch_counter);
}

test "quant kernel compiler generated Metal wrapper gates follow artifact metadata" {
    const contents = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "src/backends/metal_kernels.m", std.testing.allocator, .limited(3 * 1024 * 1024));
    defer std.testing.allocator.free(contents);

    var runtime_checked: usize = 0;
    var provider_checked: usize = 0;
    for (first_generated_artifacts) |artifact| {
        if (artifact.backend != .metal or !artifactRuntimeWired(artifact)) continue;
        const provider_helper: []const u8 = switch (artifact.epilogue) {
            .bias, .bias_gelu => "termite_metal_dispatch_generated_quant_bias",
            .relu => "termite_metal_dispatch_generated_quant_relu",
            else => continue,
        };
        const counter_name = metalGeneratedCounterNameForArtifact(artifact) orelse continue;
        const format_constant = metalRuntimeQuantFormatConstant(artifact.format) orelse continue;
        const epilogue_constant = metalRuntimeGeneratedEpilogueConstant(artifact.epilogue) orelse continue;
        const env_arg = try metalCEnvArgForArtifact(std.testing.allocator, artifact);
        defer std.testing.allocator.free(env_arg);

        const runtime_snippet = try std.fmt.allocPrint(
            std.testing.allocator,
            "{s},\n        {s},\n        runtime != NULL ? runtime->antfly_{s}_pipeline : nil,\n        runtime != NULL ? &runtime->antfly_{s}_dispatches : NULL,\n        {s},",
            .{ format_constant, env_arg, counter_name, counter_name, epilogue_constant },
        );
        defer std.testing.allocator.free(runtime_snippet);
        const runtime_found = std.mem.containsAtLeast(u8, contents, 1, runtime_snippet) or
            try metalCBespokeRuntimeWrapperHasArtifact(std.testing.allocator, contents, artifact, counter_name, format_constant, epilogue_constant);
        try std.testing.expect(runtime_found);
        runtime_checked += 1;

        if (artifactHasMetalProviderRouteEvidence(artifact)) {
            const provider_snippet = try std.fmt.allocPrint(
                std.testing.allocator,
                "return {s}(provider, {s}, provider->antfly_{s}_pipeline, {s}, {s},",
                .{ provider_helper, env_arg, counter_name, format_constant, epilogue_constant },
            );
            defer std.testing.allocator.free(provider_snippet);
            try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, provider_snippet));
            provider_checked += 1;
        }
    }
    try std.testing.expect(runtime_checked > 0);
    try std.testing.expect(provider_checked > 0);
}

test "quant kernel compiler embedded Metal source keeps generated q5 q6 bias reduction" {
    const contents = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "src/backends/metal_kernels.m", std.testing.allocator, .limited(3 * 1024 * 1024));
    defer std.testing.allocator.free(contents);

    const optimized_reduction = "threadgroup float partial[32]; acc = simd_sum(acc); if (lane_id == 0u) partial[simdgroup_id] = acc; if (simdgroup_id == 0u && lane_id >= 4u) partial[lane_id] = 0.0f; threadgroup_barrier(mem_flags::mem_threadgroup); float total = simd_sum(partial[lane_id]);";
    const old_reduction = "threadgroup float partial[32]; if (simdgroup_id == 0u) partial[lane_id] = 0.0f; acc = simd_sum(acc); threadgroup_barrier(mem_flags::mem_threadgroup); if (lane_id == 0u) partial[simdgroup_id] = acc; threadgroup_barrier(mem_flags::mem_threadgroup); float total = simd_sum(partial[lane_id]);";
    const kernels = [_][]const u8{
        "antfly_q5_k_small_batch_bias_msl_v1",
        "antfly_q5_k_small_batch_bias_gelu_msl_v1",
        "antfly_q6_k_small_batch_bias_msl_v1",
        "antfly_q6_k_small_batch_bias_gelu_msl_v1",
    };

    for (kernels) |kernel| {
        const start = std.mem.indexOf(u8, contents, kernel) orelse return error.MissingEmbeddedQ5Q6BiasKernel;
        const end = std.mem.indexOfPos(u8, contents, start, "\"}\\n\"") orelse return error.MissingEmbeddedQ5Q6BiasKernelEnd;
        const body = contents[start..end];
        try std.testing.expect(std.mem.containsAtLeast(u8, body, 1, optimized_reduction));
        try std.testing.expect(!std.mem.containsAtLeast(u8, body, 1, old_reduction));
    }
}

test "quant kernel compiler generated q5 k source hoists block half scales" {
    const q5_sources = [_][]const u8{
        first_general_metal_q5_source,
        first_general_metal_q5_bias_source,
        first_general_metal_q5_bias_gelu_source,
    };

    for (q5_sources) |source| {
        try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "antfly_q5_k_dequant_lane(const device uchar *block, int lane, float d, float dmin)"));
        try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "const float d = antfly_half_le_to_float(block);"));
        try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "const float dmin = antfly_half_le_to_float(block + 2);"));
        try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "antfly_q5_k_dequant_lane(block, lane, d, dmin)"));
        try std.testing.expect(!std.mem.containsAtLeast(u8, source, 1, "return antfly_half_le_to_float(d)"));
    }
}

test "quant kernel compiler generated q6 k source hoists block half scale" {
    const q6_sources = [_][]const u8{
        first_general_metal_q6_source,
        first_general_metal_q6_bias_source,
        first_general_metal_q6_bias_gelu_source,
    };

    for (q6_sources) |source| {
        try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "antfly_q6_k_dequant_lane(const device uchar *block, int lane, float d)"));
        try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "const float d = antfly_half_le_to_float(block + 208);"));
        try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "antfly_q6_k_dequant_lane(block, lane, d)"));
        try std.testing.expect(!std.mem.containsAtLeast(u8, source, 1, "return antfly_half_le_to_float(d)"));
    }
}

test "quant kernel compiler generated Metal headers match production state" {
    for (first_generated_artifacts) |artifact| {
        if (artifact.backend != .metal) continue;
        const compiled = compileQuantKernelSource(.{
            .backend = artifact.backend,
            .format = artifact.format,
            .row_bucket = artifact.row_bucket,
            .epilogue = artifact.epilogue,
        }) orelse return error.MissingGeneratedSource;
        const emitted = try emitCompiledSource(std.testing.allocator, compiled);
        defer emitted.deinit(std.testing.allocator);
        const source = emitted.data;
        if (artifact.production_enabled) {
            try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "production_enabled=true"));
            try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "Promoted after"));
            try std.testing.expect(!std.mem.containsAtLeast(u8, source, 1, "stays on native handwritten MSL until this"));
        } else {
            try std.testing.expect(std.mem.containsAtLeast(u8, source, 1, "production_enabled=false"));
            try std.testing.expect(!std.mem.containsAtLeast(u8, source, 1, "Promoted after"));
        }
    }
}

test "quant kernel compiler Metal build check covers generated and promoted artifacts" {
    const contents = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, "build.zig", std.testing.allocator, .limited(256 * 1024));
    defer std.testing.allocator.free(contents);

    const macos_gate = std.mem.indexOf(u8, contents, "if (target.result.os.tag == .macos) {\n        const quant_kernel_metal_artifact_check = b.addRunArtifact(quant_kernel_codegen_exe);") orelse return error.MissingMetalBuildMacosGate;
    const non_macos_fail_closed = std.mem.indexOf(u8, contents, "} else {\n        const quant_kernel_metal_unavailable = b.addFail(metal_unavailable_message);\n        quant_kernel_metal_unavailable_step = &quant_kernel_metal_unavailable.step;\n        quant_kernel_metal_check_step.dependOn(&quant_kernel_metal_unavailable.step);\n    }\n\n    const quant_kernel_metal_runtime_check_step") orelse return error.MissingMetalBuildNonMacosFailClosed;
    try std.testing.expect(macos_gate < non_macos_fail_closed);
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "if (std.mem.startsWith(u8, args[0], \"-\")) return default_filters"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "fn targetRunsOnBuildHost(b: *std.Build, target: std.Build.ResolvedTarget) bool"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, ".target = b.graph.host"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_metal_artifact_check.addArg(\"--check-metal\")"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_metal_artifact_check.step.dependOn(&quant_kernel_codegen_test_check.step)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_metal_artifact_check_step = &quant_kernel_metal_artifact_check.step"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "Metal quant kernel evidence targets require a macOS target with xcrun/Metal; no Metal runtime evidence was run."));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "const metal_unavailable_step = quant_kernel_metal_unavailable_step orelse unreachable;"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_metal_production_regression_step.dependOn(metal_unavailable_step)"));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, "run_quant_kernel_metal_runtime_default_check"));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_metal_runtime_default_check_step"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant-kernel-metal-runtime-route-all"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "--runtime-route-all"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "--require-runtime-route-all"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant-kernel-metal-production-regression-check"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "\"500\",\n            \"--production-regression-check\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "--production-regression-check"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "/private/tmp/antfly-quant-metal-production-regression-evidence.json"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant-kernel-metal-blocker-evidence-refresh"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "--refresh-blocker-evidence"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant-kernel-metal-blocker-strict-check"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "--confirm-cleared-blockers"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "--fail-on-cleared-blocker"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "refresh_quant_kernel_metal_blocker_evidence.step.dependOn(&check_quant_kernel_metal_runtime_route_all.step)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "check_quant_kernel_metal_blocker_evidence.step.dependOn(&refresh_quant_kernel_metal_blocker_evidence.step)"));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, "run_quant_kernel_metal_runtime_route_all.step.dependOn(&run_quant_kernel_metal_production_regression.step)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "if (target.result.os.tag == .macos and targetRunsOnBuildHost(b, target)) {\n        if (quant_kernel_metal_production_regression_run_step) |production_regression_step| {\n            production_regression_step.dependOn(&run_quant_kernel_metal_runtime_check_tests.step);\n        }\n        quant_kernel_metal_runtime_check_step.dependOn(&run_quant_kernel_metal_runtime_check_tests.step);\n    }"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant-kernel-metal-local-check"));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_metal_local_check_step.dependOn(quant_kernel_metal_runtime_default_check_step orelse quant_kernel_metal_runtime_check_step)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "if (target.result.os.tag == .macos and targetRunsOnBuildHost(b, target)) {\n        quant_kernel_metal_local_check_step.dependOn(&run_quant_kernel_metal_runtime_check_tests.step);\n    }"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_metal_local_check_step.dependOn(quant_kernel_metal_runtime_route_all_step)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_metal_local_check_step.dependOn(quant_kernel_metal_production_regression_step)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_metal_local_check_step.dependOn(&run_quant_kernel_compiler_tests.step)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_local_check_step.dependOn(&quant_kernel_codegen_test_check.step)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_local_check_step.dependOn(&cuda_artifact_source_policy_check.step)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "if (target.result.os.tag == .macos) {\n        quant_kernel_local_check_step.dependOn(quant_kernel_metal_local_check_step);\n    }"));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_local_check_step.dependOn(quant_kernel_metal_runtime_route_all_step)"));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_local_check_step.dependOn(quant_kernel_metal_production_regression_step)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "if (targetRunsOnBuildHost(b, target)) {\n        quant_kernel_local_check_step.dependOn(&run_quant_kernel_compiler_tests.step);\n        quant_kernel_metal_local_check_step.dependOn(&run_quant_kernel_compiler_tests.step);\n    }"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "if (targetRunsOnBuildHost(b, target)) {\n        quant_kernel_local_check_step.dependOn(&run_quant_kernel_cuda_microbench_tests.step);\n    }"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "test-metal-gemma4-prefill-frame-e4b-generated-q8"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "\"--e4b-smoke\",\n        \"--generated-q8-smoke\""));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "test-metal-gemma4-prefill-frame-e4b-generated-q8-q4-0"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "ANTFLY_INFERENCE_GEMMA4_MIN_GENERATED_Q4_0_SMALL_BATCH=1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "ANTFLY_INFERENCE_GEMMA4_MIN_GENERATED_FAMILY_COUNT=2"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "ANTFLY_INFERENCE_GEMMA4_EXPECTED_GENERATED_TOP_FAMILY=q4_0"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "ANTFLY_INFERENCE_GEMMA4_MIN_GENERATED_TOP_COUNT=1"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant-kernel-metal-model-local-check"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_metal_model_local_check_step.dependOn(quant_kernel_metal_local_check_step)"));
    try std.testing.expect(!std.mem.containsAtLeast(u8, contents, 1, "metal_gemma4_prefill_frame_e4b_generated_q8_q4_0_test.step.dependOn(quant_kernel_metal_local_check_step)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "if (target.result.os.tag == .macos and targetRunsOnBuildHost(b, target)) {\n        quant_kernel_metal_model_local_check_step.dependOn(&metal_gemma4_prefill_frame_e4b_generated_q8_q4_0_test.step);\n    }"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant-kernel-metal-industry-local-check"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_metal_industry_local_check_step.dependOn(quant_kernel_metal_local_check_step)"));
    try std.testing.expect(std.mem.containsAtLeast(u8, contents, 1, "quant_kernel_metal_industry_local_check_step.dependOn(quant_kernel_metal_model_local_check_step)"));

    var metal_artifact_count: usize = 0;
    for (first_generated_artifacts) |artifact| {
        if (artifact.backend != .metal) continue;
        metal_artifact_count += 1;
        try std.testing.expect(std.mem.containsAtLeast(u8, artifact.check_command, 1, artifact.source_path));
    }
    try std.testing.expect(metal_artifact_count > 0);
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, contents, "src/ops/metal/generated/quant_kernel_"));
    try std.testing.expectEqual(@as(usize, 0), std.mem.count(u8, contents, "src/ops/metal/artifacts/quant_kernel_"));
}

test "quant kernel compiler coverage keeps CUDA and Metal route metadata aligned" {
    for (first_coverage) |case| {
        const cuda = loweringFor(.cuda, case.format, case.row_bucket, case.epilogue);
        const metal = loweringFor(.metal, case.format, case.row_bucket, case.epilogue);
        const spec = specFor(case.format).?;
        const cuda_dev_candidate = generatedArtifactForCandidate(.cuda, case.format, case.row_bucket, case.epilogue) != null;
        const metal_dev_candidate = generatedArtifactForCandidate(.metal, case.format, case.row_bucket, case.epilogue) != null;
        const metal_generated_production = metal.production_route == .generated_production;
        try std.testing.expectEqual(case.format, cuda.format);
        try std.testing.expectEqual(case.row_bucket, cuda.row_bucket);
        try std.testing.expectEqual(case.epilogue, cuda.epilogue);

        if (!spec.supportsBackend(.cuda)) {
            try std.testing.expectEqual(LoweringRoute.unsupported, cuda.production_route);
            try std.testing.expectEqual(FallbackReason.unsupported_backend, cuda.fallback_reason);
        } else if (!supportsEpilogueForBackend(spec, .cuda, case.epilogue)) {
            try std.testing.expectEqual(LoweringRoute.unsupported, cuda.production_route);
            try std.testing.expectEqual(FallbackReason.unsupported_epilogue, cuda.fallback_reason);
        } else if (cuda_dev_candidate) {
            try std.testing.expectEqual(LoweringRoute.handwritten_production, cuda.production_route);
            try std.testing.expectEqual(FallbackReason.generated_artifact_missing, cuda.fallback_reason);
            try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, cuda.candidate_route);
        } else {
            try std.testing.expectEqual(LoweringRoute.handwritten_production, cuda.production_route);
            try std.testing.expectEqual(FallbackReason.none, cuda.fallback_reason);
        }

        if (!spec.supportsBackend(.metal)) {
            try std.testing.expectEqual(LoweringRoute.unsupported, metal.production_route);
            try std.testing.expectEqual(FallbackReason.unsupported_backend, metal.fallback_reason);
        } else if (!supportsEpilogueForBackend(spec, .metal, case.epilogue)) {
            try std.testing.expectEqual(LoweringRoute.unsupported, metal.production_route);
            try std.testing.expectEqual(FallbackReason.unsupported_epilogue, metal.fallback_reason);
        } else if (metal_generated_production) {
            try std.testing.expectEqual(LoweringRoute.generated_production, metal.production_route);
            try std.testing.expectEqual(FallbackReason.none, metal.fallback_reason);
            try std.testing.expectEqual(LoweringRoute.unsupported, metal.candidate_route);
        } else if (metal_dev_candidate) {
            try std.testing.expectEqual(LoweringRoute.handwritten_production, metal.production_route);
            const artifact = generatedArtifactForKernel(.metal, metal.kernel_id) orelse return error.MissingMetalCandidateArtifact;
            const expected_fallback: FallbackReason = if (checkedInMetalEvidenceForKernel(artifact.kernel_id) != null)
                .generated_runtime_not_wired
            else
                .generated_artifact_missing;
            try std.testing.expectEqual(expected_fallback, metal.fallback_reason);
            try std.testing.expectEqual(LoweringRoute.generated_dev_candidate, metal.candidate_route);
        } else {
            try std.testing.expectEqual(LoweringRoute.handwritten_production, metal.production_route);
            try std.testing.expectEqual(FallbackReason.none, metal.fallback_reason);
        }

        try std.testing.expect(cuda.schedule.dispatch != .scalar);
        try std.testing.expect(metal.schedule.dispatch != .scalar);
    }
}

test "quant kernel compiler rejects unsupported shapes and epilogues explicitly" {
    const empty = loweringFor(.cuda, .q4_k, .rows_0, .bias_gelu);
    try std.testing.expectEqual(LoweringRoute.unsupported, empty.production_route);
    try std.testing.expectEqual(FallbackReason.unsupported_shape, empty.fallback_reason);

    const future_epilogues = [_]Epilogue{ .triple, .relu, .gelu, .add, .argmax };
    for (future_epilogues) |epilogue| {
        const unsupported_epilogue = loweringFor(.cuda, .q4_k, .rows_2_8, epilogue);
        try std.testing.expectEqual(LoweringRoute.unsupported, unsupported_epilogue.production_route);
        try std.testing.expectEqual(FallbackReason.unsupported_epilogue, unsupported_epilogue.fallback_reason);
    }
    const q6_bias_gelu = loweringFor(.cuda, .q6_k, .rows_2_8, .bias_gelu);
    try std.testing.expectEqual(LoweringRoute.unsupported, q6_bias_gelu.production_route);
    try std.testing.expectEqual(FallbackReason.unsupported_epilogue, q6_bias_gelu.fallback_reason);

    const q5_cuda = loweringFor(.cuda, .q5_k, .rows_2_8, .none);
    try std.testing.expectEqual(LoweringRoute.unsupported, q5_cuda.production_route);
    try std.testing.expectEqual(FallbackReason.unsupported_backend, q5_cuda.fallback_reason);

    const q5_cuda_bias_gelu = loweringFor(.cuda, .q5_k, .rows_2_8, .bias_gelu);
    try std.testing.expectEqual(LoweringRoute.unsupported, q5_cuda_bias_gelu.production_route);
    try std.testing.expectEqual(FallbackReason.unsupported_backend, q5_cuda_bias_gelu.fallback_reason);

    const unsupported_format = loweringFor(.cuda, .unknown, .rows_2_8, .bias_gelu);
    try std.testing.expectEqual(LoweringRoute.unsupported, unsupported_format.production_route);
    try std.testing.expectEqual(FallbackReason.unsupported_format, unsupported_format.fallback_reason);
}

test "quant kernel compiler counters classify route lowerings" {
    const lazy = countersForLowering(loweringFor(.cuda, .q4_k, .rows_2_8, .bias_gelu));
    try std.testing.expectEqual(@as(usize, 1), lazy.quant_kernel_planned_ops);
    try std.testing.expectEqual(@as(usize, 1), lazy.quant_kernel_handwritten_production);
    try std.testing.expectEqual(@as(usize, 0), lazy.quant_kernel_generated_production);
    try std.testing.expectEqual(@as(usize, 0), lazy.quant_kernel_unsupported_routes);
    try std.testing.expectEqual(@as(usize, 1), lazy.quant_kernel_generated_candidates);
    try std.testing.expectEqual(@as(usize, 1), lazy.quant_kernel_fallback_generated_artifact_missing);
    try std.testing.expectEqual(@as(usize, 0), lazy.quant_kernel_fallback_generated_runtime_not_wired);
    try std.testing.expectEqual(@as(usize, 0), lazy.quant_kernel_fallback_unsupported);

    const q6_ready = countersForLowering(loweringFor(.metal, .q6_k, .rows_2_8, .bias));
    try std.testing.expectEqual(@as(usize, 1), q6_ready.quant_kernel_planned_ops);
    try std.testing.expectEqual(@as(usize, 0), q6_ready.quant_kernel_handwritten_production);
    try std.testing.expectEqual(@as(usize, 1), q6_ready.quant_kernel_generated_production);
    try std.testing.expectEqual(@as(usize, 0), q6_ready.quant_kernel_unsupported_routes);
    try std.testing.expectEqual(@as(usize, 0), q6_ready.quant_kernel_generated_candidates);
    try std.testing.expectEqual(@as(usize, 0), q6_ready.quant_kernel_fallback_generated_artifact_missing);
    try std.testing.expectEqual(@as(usize, 0), q6_ready.quant_kernel_fallback_generated_runtime_not_wired);
    try std.testing.expectEqual(@as(usize, 0), q6_ready.quant_kernel_fallback_unsupported);

    const handwritten_only = countersForLowering(loweringFor(.cuda, .q8_0, .rows_1, .none));
    try std.testing.expectEqual(@as(usize, 1), handwritten_only.quant_kernel_planned_ops);
    try std.testing.expectEqual(@as(usize, 1), handwritten_only.quant_kernel_handwritten_production);
    try std.testing.expectEqual(@as(usize, 0), handwritten_only.quant_kernel_generated_production);
    try std.testing.expectEqual(@as(usize, 0), handwritten_only.quant_kernel_unsupported_routes);
    try std.testing.expectEqual(@as(usize, 0), handwritten_only.quant_kernel_generated_candidates);
    try std.testing.expectEqual(@as(usize, 0), handwritten_only.quant_kernel_fallback_generated_artifact_missing);
    try std.testing.expectEqual(@as(usize, 0), handwritten_only.quant_kernel_fallback_generated_runtime_not_wired);
    try std.testing.expectEqual(@as(usize, 0), handwritten_only.quant_kernel_fallback_unsupported);

    const unsupported = countersForLowering(loweringFor(.cuda, .unknown, .rows_2_8, .bias_gelu));
    try std.testing.expectEqual(@as(usize, 1), unsupported.quant_kernel_planned_ops);
    try std.testing.expectEqual(@as(usize, 0), unsupported.quant_kernel_handwritten_production);
    try std.testing.expectEqual(@as(usize, 0), unsupported.quant_kernel_generated_production);
    try std.testing.expectEqual(@as(usize, 1), unsupported.quant_kernel_unsupported_routes);
    try std.testing.expectEqual(@as(usize, 0), unsupported.quant_kernel_generated_candidates);
    try std.testing.expectEqual(@as(usize, 0), unsupported.quant_kernel_fallback_generated_artifact_missing);
    try std.testing.expectEqual(@as(usize, 0), unsupported.quant_kernel_fallback_generated_runtime_not_wired);
    try std.testing.expectEqual(@as(usize, 1), unsupported.quant_kernel_fallback_unsupported_format);
    try std.testing.expectEqual(@as(usize, 0), unsupported.quant_kernel_fallback_unsupported_shape);
    try std.testing.expectEqual(@as(usize, 0), unsupported.quant_kernel_fallback_unsupported_epilogue);
    try std.testing.expectEqual(@as(usize, 0), unsupported.quant_kernel_fallback_unsupported_backend);
    try std.testing.expectEqual(@as(usize, 1), unsupported.quant_kernel_fallback_unsupported);

    const unsupported_backend = countersForLowering(loweringFor(.cuda, .q5_k, .rows_2_8, .none));
    try std.testing.expectEqual(@as(usize, 0), unsupported_backend.quant_kernel_fallback_unsupported_format);
    try std.testing.expectEqual(@as(usize, 1), unsupported_backend.quant_kernel_fallback_unsupported_backend);
    try std.testing.expectEqual(@as(usize, 1), unsupported_backend.quant_kernel_fallback_unsupported);

    var promoted = loweringFor(.cuda, .q4_k, .rows_2_8, .bias_gelu);
    promoted.production_route = .generated_production;
    promoted.candidate_route = .unsupported;
    promoted.fallback_reason = .none;
    const promoted_counters = countersForLowering(promoted);
    try std.testing.expectEqual(@as(usize, 0), promoted_counters.quant_kernel_handwritten_production);
    try std.testing.expectEqual(@as(usize, 1), promoted_counters.quant_kernel_generated_production);
    try std.testing.expectEqual(@as(usize, 0), promoted_counters.quant_kernel_generated_candidates);
}

test "quant kernel compiler adds every plan counter to runtime stats" {
    const Stats64 = struct {
        quant_kernel_planned_ops: u64 = 0,
        quant_kernel_handwritten_production: u64 = 0,
        quant_kernel_generated_production: u64 = 0,
        quant_kernel_unsupported_routes: u64 = 0,
        quant_kernel_generated_candidates: u64 = 0,
        quant_kernel_fallback_generated_artifact_missing: u64 = 0,
        quant_kernel_fallback_generated_runtime_not_wired: u64 = 0,
        quant_kernel_fallback_unsupported_format: u64 = 0,
        quant_kernel_fallback_unsupported_shape: u64 = 0,
        quant_kernel_fallback_unsupported_epilogue: u64 = 0,
        quant_kernel_fallback_unsupported_backend: u64 = 0,
        quant_kernel_fallback_tensor_core_repack_required: u64 = 0,
        quant_kernel_fallback_unsupported: u64 = 0,
    };
    const StatsUsize = struct {
        quant_kernel_planned_ops: usize = 0,
        quant_kernel_handwritten_production: usize = 0,
        quant_kernel_generated_production: usize = 0,
        quant_kernel_unsupported_routes: usize = 0,
        quant_kernel_generated_candidates: usize = 0,
        quant_kernel_fallback_generated_artifact_missing: usize = 0,
        quant_kernel_fallback_generated_runtime_not_wired: usize = 0,
        quant_kernel_fallback_unsupported_format: usize = 0,
        quant_kernel_fallback_unsupported_shape: usize = 0,
        quant_kernel_fallback_unsupported_epilogue: usize = 0,
        quant_kernel_fallback_unsupported_backend: usize = 0,
        quant_kernel_fallback_tensor_core_repack_required: usize = 0,
        quant_kernel_fallback_unsupported: usize = 0,
    };

    var counters = PlanCounters{};
    inline for (@typeInfo(PlanCounters).@"struct".fields, 1..) |field, value| {
        @field(counters, field.name) = value;
    }

    var stats64 = Stats64{};
    addCountersToStats(&stats64, counters);
    var stats_usize = StatsUsize{};
    addCountersToStats(&stats_usize, counters);

    inline for (@typeInfo(PlanCounters).@"struct".fields) |field| {
        try std.testing.expectEqual(@as(u64, @intCast(@field(counters, field.name))), @field(stats64, field.name));
        try std.testing.expectEqual(@field(counters, field.name), @field(stats_usize, field.name));
    }
}

test "quant kernel compiler fallback counters cover every reason" {
    const base = loweringFor(.cuda, .q4_k, .rows_2_8, .bias_gelu);
    inline for (std.meta.tags(FallbackReason)) |reason| {
        var lowering = base;
        lowering.production_route = .unsupported;
        lowering.candidate_route = .unsupported;
        lowering.fallback_reason = reason;
        const counters = countersForLowering(lowering);

        try std.testing.expectEqual(@as(usize, 1), counters.quant_kernel_planned_ops);
        try std.testing.expectEqual(@as(usize, 1), counters.quant_kernel_unsupported_routes);
        try std.testing.expectEqual(@as(usize, if (reason == .generated_artifact_missing) 1 else 0), counters.quant_kernel_fallback_generated_artifact_missing);
        try std.testing.expectEqual(@as(usize, if (reason == .generated_runtime_not_wired) 1 else 0), counters.quant_kernel_fallback_generated_runtime_not_wired);
        try std.testing.expectEqual(@as(usize, if (reason == .unsupported_format) 1 else 0), counters.quant_kernel_fallback_unsupported_format);
        try std.testing.expectEqual(@as(usize, if (reason == .unsupported_shape) 1 else 0), counters.quant_kernel_fallback_unsupported_shape);
        try std.testing.expectEqual(@as(usize, if (reason == .unsupported_epilogue) 1 else 0), counters.quant_kernel_fallback_unsupported_epilogue);
        try std.testing.expectEqual(@as(usize, if (reason == .unsupported_backend) 1 else 0), counters.quant_kernel_fallback_unsupported_backend);
        try std.testing.expectEqual(@as(usize, if (reason == .tensor_core_repack_required) 1 else 0), counters.quant_kernel_fallback_tensor_core_repack_required);
        try std.testing.expectEqual(@as(usize, switch (reason) {
            .unsupported_format, .unsupported_shape, .unsupported_epilogue, .unsupported_backend => 1,
            .none, .generated_artifact_missing, .generated_runtime_not_wired, .tensor_core_repack_required => 0,
        }), counters.quant_kernel_fallback_unsupported);
    }
}

test "quant kernel compiler reference checker covers Q4_K small-batch bias gelu" {
    const allocator = std.testing.allocator;
    const rows = 8;
    const in_dim = 512;
    const out_dim = 3;

    var dense_weight: [in_dim * out_dim]f32 = undefined;
    for (&dense_weight, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 23)) - 11)) / 16.0;
    }
    const raw_weight = try quant_codec.quantizeQ4_KFromF32(allocator, &dense_weight);
    defer allocator.free(raw_weight);

    var input: [rows * in_dim]f32 = undefined;
    for (&input, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) / 32.0;
    }
    const bias = [_]f32{ 0.125, -0.25, 0.375 };
    var generated_actual: [rows * out_dim]f32 = undefined;
    try generatedMathMatmulBiasGelu(.q4_k, raw_weight, &input, &bias, rows, in_dim, out_dim, &generated_actual);

    var reference_actual: [rows * out_dim]f32 = undefined;
    try referenceMatmulBiasGelu(allocator, .q4_k, raw_weight, &input, &bias, rows, in_dim, out_dim, &reference_actual);

    var dequantized: [in_dim * out_dim]f32 = undefined;
    try quant_codec.dequantizeToFloat32(.{ .known = .Q4_K }, raw_weight, &dequantized);
    for (0..rows) |r| {
        for (0..out_dim) |o| {
            var expected = bias[o];
            for (0..in_dim) |i| expected += input[r * in_dim + i] * dequantized[o * in_dim + i];
            expected = gelu(expected);
            try std.testing.expectApproxEqAbs(expected, reference_actual[r * out_dim + o], 0.00001);
            try std.testing.expectApproxEqAbs(expected, generated_actual[r * out_dim + o], 0.00001);
        }
    }
}

test "quant kernel compiler CPU reference covers descriptor formats no-bias" {
    const allocator = std.testing.allocator;
    const rows = 3;
    const in_dim = 512;
    const out_dim = 2;

    var dense_weight: [in_dim * out_dim]f32 = undefined;
    for (&dense_weight, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 5 + 3) % 29)) - 14)) / 24.0;
    }

    var input: [rows * in_dim]f32 = undefined;
    for (&input, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 3) % 17)) - 8)) / 40.0;
    }

    for (descriptor_formats) |format| {
        const raw_weight = try quantizeForFormat(allocator, format, &dense_weight);
        defer allocator.free(raw_weight);

        var actual: [rows * out_dim]f32 = undefined;
        try referenceMatmulNoBias(allocator, format, raw_weight, &input, rows, in_dim, out_dim, &actual);

        var dequantized: [in_dim * out_dim]f32 = undefined;
        try quant_codec.dequantizeToFloat32(tensorTypeForFormat(format).?, raw_weight, &dequantized);
        for (0..rows) |r| {
            for (0..out_dim) |o| {
                var expected: f32 = 0.0;
                for (0..in_dim) |i| expected += input[r * in_dim + i] * dequantized[o * in_dim + i];
                try std.testing.expectApproxEqAbs(expected, actual[r * out_dim + o], toleranceFor(format, .none));
            }
        }
    }
}

test "quant kernel compiler CPU reference covers single-output epilogues" {
    const allocator = std.testing.allocator;
    const rows = 2;
    const in_dim = 512;
    const out_dim = 2;
    const formats = [_]quant_matmul.Format{ .q4_k, .q8_0 };
    const reference_epilogues = [_]Epilogue{ .none, .bias, .bias_gelu, .relu, .gelu };

    var dense_weight: [in_dim * out_dim]f32 = undefined;
    for (&dense_weight, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 11 + 1) % 31)) - 15)) / 20.0;
    }

    var input: [rows * in_dim]f32 = undefined;
    for (&input, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 7) % 19)) - 9)) / 48.0;
    }
    const bias = [_]f32{ 0.2, -0.3 };

    for (formats) |format| {
        const raw_weight = try quantizeForFormat(allocator, format, &dense_weight);
        defer allocator.free(raw_weight);

        var dequantized: [in_dim * out_dim]f32 = undefined;
        try quant_codec.dequantizeToFloat32(tensorTypeForFormat(format).?, raw_weight, &dequantized);

        for (reference_epilogues) |epilogue| {
            var actual: [rows * out_dim]f32 = undefined;
            try referenceMatmulEpilogue(allocator, format, raw_weight, &input, &bias, rows, in_dim, out_dim, epilogue, &actual);

            for (0..rows) |r| {
                for (0..out_dim) |o| {
                    var acc: f32 = 0.0;
                    for (0..in_dim) |i| acc += input[r * in_dim + i] * dequantized[o * in_dim + i];
                    const expected = switch (epilogue) {
                        .none => acc,
                        .bias => acc + bias[o],
                        .bias_gelu => gelu(acc + bias[o]),
                        .relu => @max(acc, 0.0),
                        .gelu => gelu(acc),
                        else => unreachable,
                    };
                    try std.testing.expectApproxEqAbs(expected, actual[r * out_dim + o], toleranceFor(format, epilogue));
                }
            }
        }
    }

    var scratch: [rows * out_dim]f32 = undefined;
    try std.testing.expectError(error.UnsupportedEpilogue, referenceMatmulEpilogue(allocator, .q4_k, &.{}, &input, null, rows, in_dim, out_dim, .pair, &scratch));
}
